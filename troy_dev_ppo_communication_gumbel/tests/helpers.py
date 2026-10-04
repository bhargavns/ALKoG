"""Small deterministic environments for Gumbel mechanism tests."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from communication.GumbelPPOTrainer import GumbelPPOTrainer
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic


class ToyEnv:
    observation_space = SimpleNamespace(shape=(4,))
    action_space = SimpleNamespace(n=4)

    def __init__(self, length=5, seed=0):
        self.length = self.max_steps = length
        self.rng = np.random.default_rng(seed)
        self.resets = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.resets += 1
        self.target = int(self.rng.integers(4))
        self.step_count = 0
        return np.eye(4)[self.target], {}

    def step(self, action):
        self.step_count += 1
        correct = action == self.target
        end = self.step_count == self.length
        return np.eye(4)[self.target], float(correct), end, False, {
            'food_reached': bool(correct and end), 'lion_caught': False, 'lion_caged': True}


class ToyInput:
    triples = torch.tensor([[0, -1, -1]])

    def reset(self):
        pass

    def refresh(self, obs):
        pass

    def deltas(self, obs, env):
        return torch.zeros(1, 2, 2)


class Sender(nn.Module):
    def __init__(self, n_symbols=16):
        super().__init__()
        self.message_head = nn.Linear(4, n_symbols)

    def forward(self, obs, triples, deltas):
        return self.message_head(obs)


def trainer(length=5, horizon=8, **kwargs):
    torch.manual_seed(17)
    return GumbelPPOTrainer(Sender(), CurrentMessageReceiverActorCritic(n_symbols=16),
        ToyEnv(length), ToyInput, device='cpu', horizon=horizon, epochs=2,
        minibatch_size=4, **kwargs)
