"""Transformer-encoder policy (v2, promoted from experiments/oracle).

The per-triple encodings are treated as a token set (no cross-token positional
encoding, so the encoder is permutation-equivariant) with a learned CLS token
prepended. Two pre-LN self-attention blocks contextualize the tokens -- redundant
same-object entries explain each other away, and co-occurrence patterns (lion
token alongside an inside-relation token) are computed natively.

Readout: the CLS output concatenated with a masked mean-pool of the triple-token
outputs, so heads see [obs (4) || CLS (12) || pool (12)] = 28 dims, permutation-
invariant and dynamic in triple count/vocabulary. No final LayerNorm -- delta
magnitude (encoded by the dist_proj gate as token magnitude) must survive to the
heads. Per spec: no token projection (attention in the full 12-dim token space),
2 heads, 2 blocks. Reuses lib.SymbolPolicy.SymbolTable, so drift reporting and
the PPOTrainer symbol-table optimization path are identical to TripleActorCritic.

This matched slot attention on the oracle (0.957 vs 0.98 on the plain distractor,
0.586 vs 0.574 on the conflict layout) once value-loss normalization removed the
critic-swamps-actor gradient imbalance.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from lib.SymbolPolicy import N_TRIPLE_SLOTS, SymbolTable, _mlp
from lib.SymbolicKG import SYMBOL_DIM

D = N_TRIPLE_SLOTS * SYMBOL_DIM  # 12


class Block(nn.Module):
    """Pre-LN transformer encoder block: x + MHA(LN(x)), x + FFN(LN(x))."""

    def __init__(self, d, n_heads=2, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.Tanh(), nn.Linear(ffn_mult * d, d)
        )

    def forward(self, x, pad_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


class TransformerActorCritic(nn.Module):
    def __init__(self, kg, obs_dim=4, n_actions=4, n_blocks=2, n_heads=2,
                 hidden=38, **_):
        super().__init__()
        self.symbol_table = SymbolTable(kg)
        self.cls = nn.Parameter(torch.randn(D))
        self.blocks = nn.ModuleList(Block(D, n_heads) for _ in range(n_blocks))
        in_dim = obs_dim + 2 * D  # obs || CLS-out || masked mean-pool of tokens
        self.actor = _mlp(in_dim, n_actions, hidden)
        self.critic = _mlp(in_dim, 1, hidden)

    def _features(self, obs, triples, deltas):
        enc, valid = self.symbol_table(triples, deltas)  # [B,K,12], [B,K]
        b = enc.shape[0]
        x = torch.cat([self.cls.expand(b, 1, D), enc], dim=1)  # [B,K+1,12]
        # True = ignore as key; CLS (col 0) is always attendable, so even a
        # row with zero valid triples never softmaxes over an empty key set
        pad = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=enc.device), ~valid], dim=1
        )
        for blk in self.blocks:
            x = blk(x, pad)
        # readout: CLS output || masked mean-pool of the triple-token outputs.
        # An all-invalid row pools to zeros (denom clamped), matching the
        # always-attendable CLS fallback above.
        m = valid.unsqueeze(-1).to(x.dtype)  # [B,K,1]
        pool = (x[:, 1:] * m).sum(1) / m.sum(1).clamp(min=1.0)  # [B,12]
        return torch.cat([obs, x[:, 0], pool], dim=-1)

    def dist(self, obs, triples, deltas):
        return Categorical(logits=self.actor(self._features(obs, triples, deltas)))

    def value(self, obs, triples, deltas):
        return self.critic(self._features(obs, triples, deltas)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, triples, deltas):
        d = self.dist(obs, triples, deltas)
        action = d.sample()
        return action, d.log_prob(action), self.value(obs, triples, deltas)

    def evaluate(self, obs, triples, deltas, actions):
        d = self.dist(obs, triples, deltas)
        return d.log_prob(actions), d.entropy(), self.value(obs, triples, deltas)
