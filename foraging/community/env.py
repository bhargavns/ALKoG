"""The community world: the errand world with sparse food, named objectives and several foragers.

Changes from `foraging.errand.env` (see README.md):
  * food is in only some glades, different every episode;
  * the objective names specific foods, drawn from the ripe food on the map, so it can always be done;
  * each episode belongs to one of M foragers, who differ in what they know: some know whether lions
    are fed today, others whether tigers are;
  * more held-out combinations: six (glade, animal, state) and three (glade, food) combinations never
    occur in training.
"""
import torch
from foraging.errand.env import (World as ErrandWorld, LION, TIGER, ASLEEP, AWAKE, CAGED, GLADES, FOODS, ANIMALS,
                                 STATES, FOOD_PER_GLADE, NFOOD, P_ANIMAL, P_RIPE, P_FED, STATE_P, STAT_KEYS, START)

APPLE, BERRY, MUSHROOM = range(3)
P_FOOD_GLADE = 0.5
# what forager m knows: (lions fed?, tigers fed?)
KNOWS = ((1, 0), (0, 1), (1, 0), (0, 1))
M = len(KNOWS)
# each (animal, state) pair is held out in exactly one glade and seen in the other three
HELD_ANIMAL = ((0, TIGER, ASLEEP), (0, LION, CAGED), (1, LION, AWAKE), (1, TIGER, CAGED), (2, TIGER, AWAKE), (3, LION, ASLEEP))
# each food is held out in exactly one glade
HELD_FOOD = ((1, BERRY), (2, APPLE), (3, MUSHROOM))


class World(ErrandWorld):
    """B episodes; episode b belongs to forager b * M // B (contiguous blocks)."""

    def __init__(self, B, rules='private', scout_ripe=True, device='cpu', generator=None, max_items=2):
        super().__init__(B, rules, 'described', scout_ripe, device=device, generator=generator)
        self.max_items = max_items
        self.forager = torch.arange(B, device=device) * M // B
        self.knows = torch.tensor(KNOWS, dtype=torch.bool, device=device)[self.forager]          # [B, 2]

    # ------------------------------------------------------------------ reset
    def reset(self, max_items=None, held_out=None):
        B, dev = self.B, self.device
        max_items = self.max_items if max_items is None else max_items
        cells = self._rand(B, 4, 9).argsort(-1)[..., :3]
        xy = self.origin[None, :, None] + torch.stack((cells // 3, cells % 3), -1)
        self.food_pos = xy[:, :, :FOOD_PER_GLADE].reshape(B, NFOOD, 2)
        self.food_glade = torch.arange(4, device=dev).repeat_interleave(FOOD_PER_GLADE).expand(B, -1)
        self.food_type = self._randint(3, B, NFOOD)
        self.food_ripe = self._rand(B, NFOOD) < P_RIPE
        has = self._rand(B, 4) < P_FOOD_GLADE                                                   # glades with food
        none = ~has.any(1)
        pick = self._randint(4, B); nz = none.nonzero().squeeze(1)
        has[nz, pick[nz]] = True
        self.food_alive = has.repeat_interleave(FOOD_PER_GLADE, 1).clone()
        self.an_pos = xy[:, :, 2]
        self.an_on = self._rand(B, 4) < P_ANIMAL
        self.an_type = self._randint(2, B, 4)
        self.an_state = torch.multinomial(STATE_P.to(dev).expand(B * 4, 3), 1, generator=self.g).view(B, 4)
        if held_out == 'exclude': self._exclude_held_out()
        elif held_out == 'only': self._force_held_out()
        # at least one ripe food on the map
        ripe = self.food_alive & self.food_ripe
        no_ripe = ~ripe.any(1)
        if no_ripe.any():
            first = self.food_alive.float().argmax(1)
            nz = no_ripe.nonzero().squeeze(1)
            self.food_ripe[nz, first[nz]] = True
        self.held = self.held_out_mask()
        self.fed = torch.zeros(B, 2, dtype=torch.bool, device=dev) if self.rules == 'fixed' else self._rand(B, 2) < P_FED
        self._objective(max_items, force_held_food=held_out == 'only')
        self.pos = torch.tensor(START, device=dev).expand(B, 2).clone()
        self.was_home = torch.ones(B, dtype=torch.bool, device=dev)
        self.carry = torch.full((B,), -1, device=dev); self.carry_ripe = torch.zeros(B, dtype=torch.bool, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev); self.t = 0
        self.stats = {k: torch.zeros(B, device=dev) for k in STAT_KEYS}
        return self

    def _objective(self, max_items, force_held_food=False):
        """need[b, 0:3]: named items, drawn without replacement from the ripe food on the map."""
        B, dev = self.B, self.device
        ripe = self.food_alive & self.food_ripe                                                 # [B, NFOOD]
        n = (1 + self._randint(max_items, B)).minimum(ripe.sum(1))
        score = torch.where(ripe, self._rand(B, NFOOD), torch.full((B, NFOOD), -1.0, device=dev))
        if force_held_food:                                     # the held-out food is always wanted
            score = torch.where(ripe & self.food_held_mask(), torch.full_like(score, 2.0), score)
        rank = score.argsort(1, descending=True).argsort(1)
        take = ripe & (rank < n[:, None])
        self.need = torch.zeros(B, 4, dtype=torch.long, device=dev)
        self.need[:, :3].scatter_add_(1, self.food_type, take.long())

    # ------------------------------------------------------------------ held-out combinations
    def animal_held_mask(self):
        m = torch.zeros(self.B, 4, dtype=torch.bool, device=self.device)
        for g, a, s in HELD_ANIMAL:
            m[:, g] |= self.an_on[:, g] & (self.an_type[:, g] == a) & (self.an_state[:, g] == s)
        return m

    def food_held_mask(self):
        """[B, NFOOD]: food items whose (glade, type) is held out."""
        m = torch.zeros(self.B, NFOOD, dtype=torch.bool, device=self.device)
        for g, f in HELD_FOOD:
            m |= self.food_alive & (self.food_glade == g) & (self.food_type == f)
        return m

    def held_out_mask(self):
        """[B] bool: the map contains at least one held-out combination."""
        return self.animal_held_mask().any(1) | self.food_held_mask().any(1)

    def _exclude_held_out(self):
        while True:
            ma = self.animal_held_mask()
            if ma.any():
                redraw = torch.multinomial(STATE_P.to(self.device).expand(self.B * 4, 3), 1, generator=self.g).view(self.B, 4)
                self.an_state = torch.where(ma, redraw, self.an_state)
            mf = self.food_held_mask()
            if mf.any():
                self.food_type = torch.where(mf, self._randint(3, self.B, NFOOD), self.food_type)
            if not ma.any() and not mf.any(): return

    def _force_held_out(self):
        """Every map gets one held-out animal combination and one ripe held-out food."""
        B, dev = self.B, self.device; b = torch.arange(B, device=dev)
        ca = torch.tensor(HELD_ANIMAL, device=dev)[self._randint(len(HELD_ANIMAL), B)]
        self.an_on[b, ca[:, 0]] = True; self.an_type[b, ca[:, 0]] = ca[:, 1]; self.an_state[b, ca[:, 0]] = ca[:, 2]
        cf = torch.tensor(HELD_FOOD, device=dev)[self._randint(len(HELD_FOOD), B)]
        item = cf[:, 0] * FOOD_PER_GLADE + self._randint(FOOD_PER_GLADE, B)
        for i in range(FOOD_PER_GLADE):                                       # the glade has food
            self.food_alive[b, cf[:, 0] * FOOD_PER_GLADE + i] = True
        self.food_type[b, item] = cf[:, 1]; self.food_ripe[b, item] = True

    # ------------------------------------------------------------------ what each forager knows
    def known_fed(self):
        """[B, 2]: the forager's knowledge of who is fed: +1 fed, -1 hungry, 0 unknown."""
        return torch.where(self.knows, self.fed.float() * 2 - 1, torch.zeros(self.B, 2, device=self.device))

    def describe(self, b):
        out = []
        for g in range(4):
            parts = [('ripe ' if self.food_ripe[b, i] else 'rotten ') + FOODS[int(self.food_type[b, i])]
                     for i in range(NFOOD) if self.food_glade[b, i] == g and self.food_alive[b, i]]
            if self.an_on[b, g]: parts.append(f"{ANIMALS[int(self.an_type[b, g])]} {STATES[int(self.an_state[b, g])]}")
            out.append(f"{GLADES[g]}: {', '.join(parts) or 'empty'}")
        fed = ', '.join(f"{a}s {'fed' if self.fed[b, i] else 'hungry'}" for i, a in enumerate(ANIMALS))
        knows = ' and '.join(a + 's' for i, a in enumerate(ANIMALS) if self.knows[b, i])
        want = ', '.join(f"{n} {FOODS[i]}" for i, n in enumerate(self.need[b, :3].tolist()) if n)
        return '; '.join(out) + f" | {fed} | forager {int(self.forager[b])} knows about {knows} | objective: {want}"
