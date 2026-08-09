import torch
import torch.nn as nn
from torch.distributions import Categorical

from lib.Grounding import TRIPLE_PAD
from lib.SymbolicKG import SYMBOL_DIM

N_TRIPLE_SLOTS = 3  # src / relation / dst positions within a triple


class SymbolTable(nn.Module):
    """Trainable symbol machinery for the phase-2 triple encoder.

    Holds the 4-dim node and relation symbols (initialized from a phase-1
    SymbolicKG), a 3x4 positional embedding table (src/rel/dst slot within a
    triple), and the shared 2->4 distance projection that gates node tokens
    by their agent-frame (forward, left) ground-plane offset. All of it
    receives PPO gradients; the KG's structure and visual embeddings stay
    frozen.
    """

    def __init__(self, kg):
        super().__init__()
        self.node_symbols = nn.Parameter(kg.symbols.clone())
        self.relation_symbols = nn.Parameter(kg.relation_symbols.clone())
        self.pos_embeddings = nn.Parameter(torch.randn(N_TRIPLE_SLOTS, SYMBOL_DIM))
        self.dist_proj = nn.Linear(2, SYMBOL_DIM)  # one set of weights for all nodes
        # identity (FiLM-style) init: the gate starts as a pass-through so
        # symbols receive full-strength gradient from step one, and distance
        # modulation is learned as a perturbation on top
        nn.init.zeros_(self.dist_proj.weight)
        nn.init.ones_(self.dist_proj.bias)

    def forward(self, triples, deltas):
        """triples: LongTensor [B, K, 3] with TRIPLE_PAD for empty slots.
        deltas: FloatTensor [B, K, 2, 2] -- agent->src and agent->dst offsets.

        Per token: symbol + positional embedding, then (node tokens only)
        elementwise-multiplied by the projected offset. PAD tokens are zeroed.
        Returns [B, K * 3 * SYMBOL_DIM] with each triple laid out
        [src || rel || dst].
        """
        b, k, _ = triples.shape

        def token(table, idx, pos, delta=None):
            mask = (idx != TRIPLE_PAD).unsqueeze(-1).float()
            tok = table[idx.clamp(min=0)] + self.pos_embeddings[pos]
            if delta is not None:
                tok = tok * self.dist_proj(delta)
            return tok * mask

        parts = [
            token(self.node_symbols, triples[..., 0], 0, deltas[..., 0, :]),
            token(self.relation_symbols, triples[..., 1], 1),
            token(self.node_symbols, triples[..., 2], 2, deltas[..., 1, :]),
        ]
        return torch.cat(parts, dim=-1).reshape(b, k * N_TRIPLE_SLOTS * SYMBOL_DIM)


def _mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class TripleActorCritic(nn.Module):
    """Separate discrete actor/critic MLPs over [env vector, symbol triples].

    Input: [x/8, y/8, cos(yaw), sin(yaw)] (4) concatenated with the
    distance-gated triple encoding (k_triples * 12) -- 76 dims at defaults.
    One hidden layer each; actor outputs 4 action logits (fwd/back/left/right),
    critic a scalar value.
    """

    def __init__(self, kg, obs_dim=4, n_actions=4, k_triples=6, hidden=38):
        super().__init__()
        self.symbol_table = SymbolTable(kg)
        in_dim = obs_dim + k_triples * N_TRIPLE_SLOTS * SYMBOL_DIM
        self.actor = _mlp(in_dim, n_actions, hidden)
        self.critic = _mlp(in_dim, 1, hidden)

    def _features(self, obs, triples, deltas):
        return torch.cat([obs, self.symbol_table(triples, deltas)], dim=-1)

    def dist(self, obs, triples, deltas):
        return Categorical(logits=self.actor(self._features(obs, triples, deltas)))

    def value(self, obs, triples, deltas):
        return self.critic(self._features(obs, triples, deltas)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, triples, deltas):
        """Single-step sampling. obs [B,obs_dim], triples [B,K,3], deltas [B,K,2,2]."""
        d = self.dist(obs, triples, deltas)
        action = d.sample()
        return action, d.log_prob(action), self.value(obs, triples, deltas)

    def evaluate(self, obs, triples, deltas, actions):
        d = self.dist(obs, triples, deltas)
        return d.log_prob(actions), d.entropy(), self.value(obs, triples, deltas)
