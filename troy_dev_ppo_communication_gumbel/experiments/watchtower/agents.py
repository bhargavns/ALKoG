"""Scout and forager networks.  Each is a recurrent actor-critic that can listen
(message encoder) and, where the stage allows, speak (word decoder).  Only the
integer symbols they emit ever reach the other agent."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .env import NCH, SIZE, NACT

VOCAB, MAX_LEN = 16, 8           # symbols 0..15; STOP = 16; BOS = 17
STOP = VOCAB


class Ears(nn.Module):
    """Message encoder: symbol embeddings read by a GRU -> 64 numbers (zeros for silence)."""
    def __init__(self, width=64, emb=32):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB + 1, emb); self.gru = nn.GRU(emb, width, batch_first=True)

    def forward(self, tokens):
        length = (tokens != STOP).long().cumprod(1).sum(1)
        out, _ = self.gru(self.embedding(tokens))
        last = out[torch.arange(len(tokens), device=tokens.device), (length - 1).clamp_min(0)]
        return last * (length > 0).float()[:, None]


class Mouth(nn.Module):
    """Word decoder: from the agent's memory, sample one symbol at a time until STOP."""
    def __init__(self, core=128, width=64, emb=32):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(core, width), nn.Tanh())
        self.embedding = nn.Embedding(VOCAB + 2, emb); self.cell = nn.GRUCell(emb + width, width)
        self.out = nn.Linear(width, VOCAB + 1)

    def forward(self, h_core, forced=None, greedy=False):
        ctx = self.start(h_core); h = ctx; n = len(h_core); dev = h_core.device
        prev = torch.full((n,), VOCAB + 1, device=dev, dtype=torch.long)
        alive = torch.ones(n, device=dev); logp = torch.zeros(n, device=dev); ent = torch.zeros(n, device=dev)
        length = torch.zeros(n, device=dev); toks = []
        for t in range(MAX_LEN):
            h = self.cell(torch.cat((self.embedding(prev), ctx), -1), h)
            dist = Categorical(logits=self.out(h))
            if forced is not None: tok = forced[:, t]
            elif greedy: tok = dist.logits.argmax(-1)
            else: tok = dist.sample()
            tok = torch.where(alive.bool(), tok, torch.full_like(tok, STOP))
            logp = logp + alive * dist.log_prob(tok); ent = ent + alive * dist.entropy()
            alive = alive * (tok != STOP).float(); length = length + alive
            toks.append(tok); prev = tok
        return torch.stack(toks, 1), logp, ent, length


class Scout(nn.Module):
    """Sees the whole map; remembers across ticks; speaks; never moves."""
    def __init__(self, core=128):
        super().__init__()
        self.see = nn.Sequential(nn.Conv2d(NCH, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(32 * SIZE * SIZE, core), nn.ReLU())
        self.ears = Ears()
        self.memory = nn.GRUCell(core + 11 + 64 + 1, core)
        self.mouth = Mouth(core)
        self.value = nn.Sequential(nn.Linear(core + 8, 64), nn.Tanh(), nn.Linear(64, 1))
        self.core = core

    def step(self, h, full_map, vec, heard, fresh):
        x = torch.cat((self.see(full_map), vec, self.ears(heard), fresh[:, None]), -1)
        return self.memory(x, h)


class Forager(nn.Module):
    """Sees 3 x 3 around itself; remembers; moves, eats, picks up; may speak."""
    def __init__(self, core=128):
        super().__init__()
        self.see = nn.Sequential(nn.Linear(NCH * 9 + 2 * SIZE + 4 + 4 + 1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.ears = Ears()
        self.memory = nn.GRUCell(64 + 64 + 1, core)
        self.policy = nn.Linear(core, NACT)
        self.mouth = Mouth(core)
        self.value = nn.Sequential(nn.Linear(core + 8, 64), nn.Tanh(), nn.Linear(64, 1))
        self.core = core

    def step(self, h, view_vec, heard_emb, fresh):
        """`heard_emb`: self.ears(tokens) of the latest utterance, computed once when it arrives."""
        x = torch.cat((self.see(view_vec), heard_emb, fresh[:, None]), -1)
        return self.memory(x, h)
