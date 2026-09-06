"""Blind receiving agent: a transformer encoder over the symbol stream.

The receiver never observes the world. Its entire input is the sequence of
symbols the transmitting agent has emitted so far, which is the whole point --
anything it learns to do correctly must have travelled through the 10-symbol
channel.

Context layout (11 tokens of SYMBOL_DIM=4, so 44 values) -- the window holds
MAX_SYMBOLS past symbols regardless of how large the vocabulary is:

    [CLS] [s_1] [s_2] ... [s_n] [NULL] x (10 - n)

CLS is a learned token; its contextualized output is the only thing the policy
heads read. After 10 symbols the window slides -- the oldest symbol drops off
and the newest is appended, so position 1 always holds the oldest symbol still
in context and position 10 the newest.

NULL positions are excluded from attention via src_key_padding_mask rather than
fed as literal zero vectors: a zero embedding plus a positional encoding is not
zero, so masking is the faithful way to express "this slot carries nothing".
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from lib.SymbolicKG import SYMBOL_DIM

# Emittable vocabulary. Started at 10; cut to 4 to match the receiver's 4
# actions, so a one-to-one code exists for the pair to find. With 10 symbols
# neither agent got a causal learning signal in a 10-iteration probe (receiver
# explained-var pinned at 0.00, symbol entropy drifting at ln(10)), which is the
# usual emergent-communication bootstrap problem: the transmitter has no reason
# to make symbols informative until the receiver reads them, and vice versa.
# Shrinking the vocabulary shrinks that joint search from 10x4 to 4x4.
N_SYMBOLS = 4
SYMBOL_NAMES = tuple("abcd")
MAX_SYMBOLS = 10  # symbols retained in context
CONTEXT_LEN = MAX_SYMBOLS + 1  # + CLS
CONTEXT_VALUES = CONTEXT_LEN * SYMBOL_DIM  # 44

CLS_ID = N_SYMBOLS  # 10
NULL_ID = N_SYMBOLS + 1  # 11
VOCAB = N_SYMBOLS + 2


def build_context(symbols, device=None):
    """Token ids [CONTEXT_LEN] for a list of emitted symbol ids (oldest first).

    Only the most recent MAX_SYMBOLS are kept, which is the sliding window.
    """
    recent = list(symbols)[-MAX_SYMBOLS:]
    ids = [CLS_ID] + recent + [NULL_ID] * (MAX_SYMBOLS - len(recent))
    return torch.tensor(ids, dtype=torch.long, device=device)


def format_context(context):
    """Render a [CONTEXT_LEN] token-id tensor for the video HUD.

    Takes the ACTUAL tensor handed to the model rather than rebuilding one from
    the symbol history, so what the overlay shows is by construction what the
    receiver read: 'CLS' first, then a slot per symbol, '_' for NULL padding.
    """
    out = []
    for tok in context.tolist():
        if tok == CLS_ID:
            out.append("CLS")
        elif tok == NULL_ID:
            out.append("_")
        else:
            out.append(SYMBOL_NAMES[tok])
    return " ".join(out)


class ReceiverActorCritic(nn.Module):
    """Symbol-sequence transformer + actor/critic heads on the CLS output.

    3 encoder blocks, 1 head each. One head rather than two because d_model is
    only 4: a single head gets the full 4 dims, where two would split into
    head_dim=2 each.
    """

    def __init__(
        self,
        n_actions=4,
        d_model=SYMBOL_DIM,
        n_blocks=3,
        n_heads=1,
        hidden=SYMBOL_DIM,
        ff_dim=None,
    ):
        super().__init__()
        self.d_model = d_model
        # symbols a-j, CLS, NULL -- all randomly initialized, all trained by PPO
        self.token_embeddings = nn.Embedding(VOCAB, d_model)
        self.pos_embeddings = nn.Parameter(torch.randn(CONTEXT_LEN, d_model))
        nn.init.normal_(self.token_embeddings.weight, std=0.5)
        with torch.no_grad():  # NULL is masked out of attention; keep it literally zero
            self.token_embeddings.weight[NULL_ID].zero_()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model if ff_dim is None else ff_dim,
            dropout=0.0,  # PPO's on-policy ratio needs a deterministic forward pass
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)

        self.actor = _mlp(d_model, n_actions, hidden)
        self.critic = _mlp(d_model, 1, hidden)

    def _cls(self, context):
        """context [B, CONTEXT_LEN] token ids -> contextualized CLS [B, d_model]."""
        pad = context == NULL_ID
        x = self.token_embeddings(context) + self.pos_embeddings
        out = self.encoder(x, src_key_padding_mask=pad)
        return out[:, 0]

    def dist(self, context):
        return Categorical(logits=self.actor(self._cls(context)))

    def value(self, context):
        return self.critic(self._cls(context)).squeeze(-1)

    @torch.no_grad()
    def act(self, context, greedy=False):
        """Sample an action (argmax when greedy). context [B, CONTEXT_LEN]."""
        cls = self._cls(context)
        d = Categorical(logits=self.actor(cls))
        action = d.probs.argmax(dim=-1) if greedy else d.sample()
        return action, d.log_prob(action), self.critic(cls).squeeze(-1)

    def evaluate(self, context, actions):
        cls = self._cls(context)
        d = Categorical(logits=self.actor(cls))
        return d.log_prob(actions), d.entropy(), self.critic(cls).squeeze(-1)


def _mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )
