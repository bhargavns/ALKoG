"""Scout and foragers for the community world.  The scout is the errand scout; each forager is the
errand forager with two extra inputs (which species it knows about) and its own parameters."""
from torch import nn
from foraging.errand.agents import Scout, Forager as ErrandForager, VOCAB, MAX_LEN, STOP
from foraging.errand.env import NCH, SIZE

FORAGER_VEC = NCH * 9 + 2 * SIZE + 4 + 1 + 4 + 2 + 2 + 1   # view, position, carrying, carry ripe, need, known fed, knows, time


class Forager(ErrandForager):
    def __init__(self, core=128):
        super().__init__(core)
        self.see = nn.Sequential(nn.Linear(FORAGER_VEC, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())


__all__ = ['Scout', 'Forager', 'VOCAB', 'MAX_LEN', 'STOP', 'FORAGER_VEC']
