"""Generic recurrent agents.  No slots, no semantic heads, no reconstruction.

Speaker: permutation-invariant scene encoder -> autoregressive GRU emitting a
variable-length string over a small alphabet, terminated by EOS.
Listener: GRU over the received integers -> vector; scores whole public
candidate scenes by dot product with its own learned scene encoder.
Only integer symbols cross between the two.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .world import ELEMENT_DIM


class SceneEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.element = nn.Sequential(nn.Linear(ELEMENT_DIM, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(width, width), nn.Tanh())

    def forward(self, scene):                       # [..., elements, dim]; zero rows are padding
        mask = (scene.abs().sum(-1, keepdim=True) > 0).float()
        return self.out((self.element(scene) * mask).sum(-2))


class Speaker(nn.Module):
    def __init__(self, vocab=6, max_len=6, width=64, emb=32):
        super().__init__()
        self.vocab, self.eos, self.max_len = vocab, vocab, max_len      # symbols 0..vocab-1, EOS = vocab
        self.encoder = SceneEncoder(width)
        # shared context: how the private target differs from each public room (generic, permutation-invariant)
        self.contrast = nn.Sequential(nn.Linear(2 * ELEMENT_DIM, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh())
        self.embedding = nn.Embedding(vocab + 2, emb)                   # + EOS + BOS
        self.cell = nn.GRUCell(emb + width, width)
        self.out = nn.Linear(width, vocab + 1)

    def relaxed(self, scene, tau=1.0):
        """Straight-through Gumbel sampling.  Forward values are exact one-hot
        integers; the backward pass uses the soft relaxation.  Returns the
        one-hot wire [n, L, vocab+1] and alive[n, L] = 1 while no EOS so far."""
        ctx = self.perceive(scene); h = ctx; n = len(scene)
        prev = self.embedding.weight[self.vocab + 1].expand(n, -1)
        alive = torch.ones(n, device=scene.device); wire = []; alives = []
        for t in range(self.max_len):
            h = self.cell(torch.cat((prev, ctx), -1), h)
            soft = F.gumbel_softmax(self.out(h), tau=tau, hard=False)
            hard = F.one_hot(soft.argmax(-1), self.vocab + 1).float()
            sym = hard + (soft - soft.detach())
            alive = alive * (1 - sym[:, self.eos]); wire.append(sym); alives.append(alive)
            prev = sym @ self.embedding.weight[:self.vocab + 1]
        return torch.stack(wire, 1), torch.stack(alives, 1)

    def imitate(self, scene, tokens):
        """Teacher-forced negative log-likelihood of observed utterances
        (a child hearing adults describe scenes it can also see)."""
        ctx = self.perceive(scene); h = ctx; n = len(scene)
        prev = torch.full((n,), self.vocab + 1, device=scene.device, dtype=torch.long)
        alive = torch.ones(n, device=scene.device); nll = torch.zeros(n, device=scene.device)
        for t in range(self.max_len):
            h = self.cell(torch.cat((self.embedding(prev), ctx), -1), h)
            nll = nll - alive * F.log_softmax(self.out(h), -1).gather(1, tokens[:, t:t + 1]).squeeze(1)
            alive = alive * (tokens[:, t] != self.eos).float(); prev = tokens[:, t]
        return nll.mean()

    def perceive(self, scene, candidates=None):
        """Target percept, optionally together with the public rooms.  Without
        `candidates` the speaker is blind to the context it is speaking in."""
        target = self.encoder(scene)
        if candidates is None: return self.mix(torch.cat((target, torch.zeros_like(target)), -1))
        pairs = torch.cat((scene[:, None].expand_as(candidates), candidates), -1)       # [n, rooms, elements, 2*dim]
        return self.mix(torch.cat((target, self.contrast(pairs).sum((1, 2))), -1))

    def forward(self, scene, candidates=None, mode='greedy', forced=None):
        """Returns tokens [n, max_len] (EOS-padded), log-prob, entropy, length.
        `forced` replays stored tokens to recompute their log-probability."""
        ctx = self.perceive(scene, candidates); h = ctx; n = len(scene)
        prev = torch.full((n,), self.vocab + 1, device=scene.device, dtype=torch.long)
        alive = torch.ones(n, device=scene.device)
        logp = torch.zeros(n, device=scene.device); ent = torch.zeros_like(logp); length = torch.zeros_like(logp); steps = torch.zeros_like(logp)
        toks = []
        for t in range(self.max_len):
            h = self.cell(torch.cat((self.embedding(prev), ctx), -1), h)
            dist = Categorical(logits=self.out(h))
            tok = forced[:, t] if forced is not None else dist.sample() if mode == 'sample' else dist.logits.argmax(-1)
            tok = torch.where(alive.bool(), tok, torch.full_like(tok, self.eos))
            logp = logp + alive * dist.log_prob(tok); ent = ent + alive * dist.entropy(); steps = steps + alive
            alive = alive * (tok != self.eos).float(); length = length + alive
            toks.append(tok); prev = tok
        return dict(tokens=torch.stack(toks, 1), logp=logp, entropy=ent, length=length)


class Listener(nn.Module):
    def __init__(self, vocab=10, width=64, emb=32):
        super().__init__()
        self.vocab, self.eos = vocab, vocab
        self.embedding = nn.Embedding(vocab + 1, emb)
        self.gru = nn.GRU(emb, width, batch_first=True)
        self.proj = nn.Linear(width, width)
        self.scenes = SceneEncoder(width)

    def message(self, tokens):
        """Hidden state at the last payload symbol; an empty message gives zeros."""
        length = (tokens != self.eos).long().cumprod(1).sum(1)
        out, _ = self.gru(self.embedding(tokens))
        idx = (length - 1).clamp_min(0)
        last = out[torch.arange(len(tokens), device=tokens.device), idx] * (length > 0).float()[:, None]
        return self.proj(last)

    def forward(self, candidates, tokens):
        m = self.message(tokens); c = self.scenes(candidates)
        return torch.einsum('nd,ncd->nc', m, c) / m.shape[-1] ** 0.5

    def anytime_wire(self, candidates, wire):
        """As `anytime`, for a one-hot (straight-through) wire."""
        out, _ = self.gru(wire @ self.embedding.weight)
        m = self.proj(out); c = self.scenes(candidates)
        return torch.einsum('nld,ncd->nlc', m, c) / m.shape[-1] ** 0.5

    def anytime(self, candidates, tokens):
        """Guess after every symbol: logits [n, L, candidates] and a mask of the
        positions that hold a payload symbol.  The collector can be forced to
        act at any tick, so it keeps a best guess throughout the message."""
        mask = (tokens != self.eos).long().cumprod(1).float()
        out, _ = self.gru(self.embedding(tokens))
        m = self.proj(out); c = self.scenes(candidates)
        return torch.einsum('nld,ncd->nlc', m, c) / m.shape[-1] ** 0.5, mask


def strip(tokens, eos):
    """Tensor [n, L] -> list of python tuples without EOS padding."""
    out = []
    for row in tokens.tolist():
        out.append(tuple(row[:row.index(eos)] if eos in row else row))
    return out


def pack(messages, eos, max_len, device='cpu'):
    """List of tuples -> EOS-padded tensor (messages longer than max_len are cut)."""
    t = torch.full((len(messages), max_len), eos, dtype=torch.long, device=device)
    for i, m in enumerate(messages):
        m = list(m)[:max_len]
        if m: t[i, :len(m)] = torch.tensor(m, device=device)
    return t
