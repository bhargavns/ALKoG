"""Actor-critic policy over fixed soft semantic category slots."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from lib.ObjectCategories import NUM_OBJECT_CATEGORIES
from lib.SoftCategoryGrounding import RELATION_FEATURE_DIM, SLOT_FEATURE_DIM


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class SoftCategoryActorCritic(nn.Module):
    """Encode each category with a learned symbol and continuous evidence."""

    def __init__(
        self,
        obs_dim=4,
        n_actions=4,
        num_categories=NUM_OBJECT_CATEGORIES,
        slot_dim=SLOT_FEATURE_DIM,
        symbol_dim=8,
        hidden_dim=128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.num_categories = int(num_categories)
        self.slot_dim = int(slot_dim)
        self.symbol_dim = int(symbol_dim)

        self.category_symbols = nn.Parameter(torch.randn(num_categories, symbol_dim) * 0.1)
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_dim, symbol_dim),
            nn.Tanh(),
        )
        feature_dim = obs_dim + num_categories * symbol_dim + RELATION_FEATURE_DIM
        self.actor = _mlp(feature_dim, n_actions, hidden_dim)
        self.critic = _mlp(feature_dim, 1, hidden_dim)

    def _features(self, obs, slots, relations):
        if slots.ndim != 3:
            raise ValueError(f"slots must have shape [B,C,F], received {tuple(slots.shape)}")
        presence = slots[..., :1]
        tokens = (self.category_symbols.unsqueeze(0) + self.slot_encoder(slots)) * presence
        return torch.cat([obs, tokens.flatten(1), relations], dim=-1)

    def dist(self, obs, slots, relations):
        return Categorical(logits=self.actor(self._features(obs, slots, relations)))

    def value(self, obs, slots, relations):
        return self.critic(self._features(obs, slots, relations)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, slots, relations, deterministic=False):
        distribution = self.dist(obs, slots, relations)
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), self.value(obs, slots, relations)

    def evaluate(self, obs, slots, relations, actions):
        distribution = self.dist(obs, slots, relations)
        return distribution.log_prob(actions), distribution.entropy(), self.value(obs, slots, relations)


def save_policy_checkpoint(path, model, *, iteration, category_checkpoint, config):
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "iteration": int(iteration),
            "category_checkpoint": str(category_checkpoint),
            "config": dict(config),
        },
        path,
    )


def load_policy_checkpoint(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    config = blob["config"]
    model = SoftCategoryActorCritic(
        obs_dim=int(config.get("obs_dim", 4)),
        n_actions=int(config.get("n_actions", 4)),
        num_categories=int(config.get("num_categories", NUM_OBJECT_CATEGORIES)),
        slot_dim=int(config.get("slot_dim", SLOT_FEATURE_DIM)),
        symbol_dim=int(config.get("symbol_dim", 8)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob
