"""Scout and forager for the errand world.  The message encoder (Ears) and word
decoder (Mouth) are the watchtower ones: 16 symbols, up to 8 per utterance, then stop."""
import torch
from torch import nn
from .env import NCH, SIZE, NACT
from experiments.watchtower.agents import Ears, Mouth, VOCAB, MAX_LEN, STOP

SCOUT_VEC = 4 + 1 + 2 + 1 + 1          # carrying (none/apple/berry/mushroom), carry ripe?, fed flags (shared only), time, items handed in
FORAGER_VEC = NCH * 9 + 2 * SIZE + 4 + 1 + 4 + 2 + 1   # view, position, carrying, carry ripe, objective left, fed flags, time
PRIV = 2 + 4                          # the critics also see the fed flags and the objective


class Scout(nn.Module):
    """Sees the whole map every tick; remembers; speaks only at briefings; never moves."""
    def __init__(self, core=128):
        super().__init__()
        self.see = nn.Sequential(nn.Conv2d(NCH, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(32 * SIZE * SIZE, core), nn.ReLU())
        self.ears = Ears()
        self.memory = nn.GRUCell(core + SCOUT_VEC + 64 + 1, core)
        self.mouth = Mouth(core)
        self.value = nn.Sequential(nn.Linear(core + PRIV, 64), nn.Tanh(), nn.Linear(64, 1))
        self.core = core

    def step(self, h, full_map, vec, heard_emb, fresh):
        return self.memory(torch.cat((self.see(full_map), vec, heard_emb, fresh[:, None]), -1), h)


class Forager(nn.Module):
    """Sees 3 x 3 around itself, its objective and who is fed; remembers; moves; may speak at briefings."""
    def __init__(self, core=128):
        super().__init__()
        self.see = nn.Sequential(nn.Linear(FORAGER_VEC, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.ears = Ears()
        self.memory = nn.GRUCell(64 + 64 + 1, core)
        self.policy = nn.Linear(core, NACT)
        self.mouth = Mouth(core)
        self.talk = nn.Sequential(nn.Linear(core + 64, core), nn.Tanh())   # what it says draws on memory and what it sees now
        self.value = nn.Sequential(nn.Linear(core + PRIV, 64), nn.Tanh(), nn.Linear(64, 1))
        self.core = core

    def step(self, h, vec, heard_emb, fresh):
        return self.memory(torch.cat((self.see(vec), heard_emb, fresh[:, None]), -1), h)

    def speaker_state(self, h, vec):
        return self.talk(torch.cat((h, self.see(vec)), -1))
