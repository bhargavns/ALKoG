"""Balanced single-decision oracle episodes; private labels never enter the receiver."""
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

ENTITIES = ('lion', 'tiger', 'sheep')
N_ENTITIES = len(ENTITIES)
N_CASES = 2 * N_ENTITIES
SHEEP = ENTITIES.index('sheep')
APPROACH, WARN = 0, 1


@dataclass
class Episodes:
    identities: torch.Tensor      # [B,N], public category IDs in random order
    positions: torch.Tensor       # [B,N,2], public independent positions
    cages: torch.Tensor           # [B,N], PRIVATE
    target: torch.Tensor          # [B], PRIVATE candidate index

    @property
    def target_entity(self):
        return self.identities.gather(1, self.target[:, None]).squeeze(1)

    @property
    def target_caged(self):
        return self.cages.gather(1, self.target[:, None]).squeeze(1)

    @property
    def dangerous(self):
        return (self.target_entity != SHEEP) & ~self.target_caged

    def sender_input(self):
        # Oracle evidence supplies the safety rule, including for loose sheep.
        return torch.cat((F.one_hot(self.target_entity, N_ENTITIES).float(),
                          self.target_caged[:, None].float(),
                          self.dangerous[:, None].float()), -1)

    def receiver_input(self):
        return torch.cat((F.one_hot(self.identities, N_ENTITIES).float(), self.positions), -1)

    def reward(self, actions):
        return ((actions // 2 == self.target) & (actions % 2 == self.dangerous.long())).float()

    def select(self, indices):
        return Episodes(**{f.name: getattr(self, f.name)[indices] for f in fields(self)})


def sample_episodes(count, generator, *, matched_layouts=False):
    """Balance six entity/cage meanings (also for sheep's two safe cage states).

    Evaluation uses six episodes with identical PUBLIC layout per group. Private
    cage assignments vary; shuffle controls permute messages within these groups.
    Training generates every layout independently and shuffles target meanings.
    """
    if count < N_CASES or count % N_CASES:
        raise ValueError(f'Episode count must be a positive multiple of {N_CASES}')
    meaning = torch.arange(count) % N_CASES
    if not matched_layouts:
        meaning = meaning[torch.randperm(count, generator=generator)]
    layouts = count // N_CASES if matched_layouts else count
    identities = torch.rand(layouts, N_ENTITIES, generator=generator).argsort(-1)
    positions = torch.rand(layouts, N_ENTITIES, 2, generator=generator) * 2 - 1
    if matched_layouts:
        identities = identities.repeat_interleave(N_CASES, 0)
        positions = positions.repeat_interleave(N_CASES, 0)
    target = (identities == (meaning // 2)[:, None]).long().argmax(-1)
    cages = torch.randint(2, (count, N_ENTITIES), generator=generator).bool()
    cages[torch.arange(count), target] = meaning % 2 == 1
    return Episodes(identities, positions, cages, target)


def matched_shuffle(count, generator):
    """Nonzero cyclic shifts within each six-case matched public layout."""
    if count < N_CASES or count % N_CASES:
        raise ValueError('Incomplete matched layout group')
    shifts = torch.randint(1, N_CASES, (count // N_CASES, 1), generator=generator)
    return (torch.arange(count).reshape(-1, N_CASES) // N_CASES * N_CASES
            + (torch.arange(N_CASES)[None, :] + shifts) % N_CASES).flatten()
