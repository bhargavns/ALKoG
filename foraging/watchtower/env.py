"""The watchtower world: a batch of 11 x 11 grids simulated together.

Everything that matters happens through behaviour: predators sleep, wake,
guard, hunt and catch; food is eaten or carried home.  See README.md.
"""
import torch

SIZE = 11
FOODS = ('apple', 'berry', 'mushroom')
ANIMALS = ('lion', 'tiger')
STATES = ('asleep', 'caged', 'guarding', 'hunting')
ASLEEP, CAGED, GUARDING, HUNTING = range(4)
LION, TIGER = range(2)
GLADES = ('NW', 'NE', 'SW', 'SE')
GLADE_ORIGIN = torch.tensor([[1, 1], [1, 7], [7, 1], [7, 7]])
HOME_LO, HOME_HI = 4, 6
START = (5, 5)
CHANNELS = ('wall', 'home', 'apple', 'berry', 'mushroom', 'ripe', 'rotten', 'lion', 'tiger',
            'asleep', 'caged', 'guarding', 'hunting', 'forager')
NCH = len(CHANNELS)
# actions
# Third revision (after the third pilot): there are no separate eat / pick-up actions.  A hungry
# forager eats whatever food it steps on; a forager that has eaten (or is not hungry) picks up food
# it steps on when the scout is hungry.  Choosing food is choosing where to walk.
UP, DOWN, LEFT, RIGHT, STAY = range(5)
NACT = 5
MOVES = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
STATE_P = torch.tensor([0.3, 0.2, 0.3, 0.2])
P_PREDATOR, P_RIPE = 0.7, 0.6
R_WANTED, R_OTHER, R_ROTTEN = 1.0, 0.2, -0.5
R_CAUGHT = torch.tensor([-1.0, -2.0])          # lion, tiger
R_TICK = -0.01
HUNT_RADIUS, WAKE_RADIUS = 3, 2      # revised after the pilot (was 4): see README
R_STARVE = -1.0                      # per hungry agent unfed at the end; revised after the second pilot


def cheb(a, b):
    return (a - b).abs().max(-1).values


class World:
    """B episodes.  `stage` 1, 2 or 3 (see README).  All state is tensors of batch size B."""

    def __init__(self, B, stage=1, T=None, device='cpu', generator=None):
        self.B, self.stage, self.device = B, stage, device
        self.T = T or (40 if stage == 1 else 60)
        self.g = generator
        base = torch.zeros(NCH, SIZE, SIZE, device=device)
        base[0, 0, :] = base[0, -1, :] = base[0, :, 0] = base[0, :, -1] = 1
        base[1, HOME_LO:HOME_HI + 1, HOME_LO:HOME_HI + 1] = 1
        self.base = base
        self.origin = GLADE_ORIGIN.to(device)

    # ------------------------------------------------------------------ reset
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device)

    def _randint(self, n, *shape):
        return torch.randint(n, shape, generator=self.g, device=self.device)

    def reset(self, single=None):
        """`single`: bool [B] - only one random glade is occupied (curriculum)."""
        B, dev = self.B, self.device
        cell = self._randint(9, B, 4)
        self.food_pos = self.origin[None] + torch.stack((cell // 3, cell % 3), -1)
        self.food_type = self._randint(3, B, 4)
        self.food_ripe = self._rand(B, 4) < P_RIPE
        self.food_alive = torch.ones(B, 4, dtype=torch.bool, device=dev)
        self.pred_on = self._rand(B, 4) < P_PREDATOR
        self.pred_type = self._randint(2, B, 4)
        self.pred_state = torch.multinomial(STATE_P.to(dev).expand(B * 4, 4), 1, generator=self.g).view(B, 4)
        self.pred_pos = self._place_predators()
        if single is not None:
            keep = self._randint(4, B)
            only = torch.arange(4, device=dev)[None] == keep[:, None]
            mask = ~single[:, None] | only
            self.food_alive &= mask; self.pred_on &= mask
        self.pos = torch.tensor(START, device=dev).expand(B, 2).clone()
        self.carry = torch.full((B,), -1, device=dev); self.carry_ripe = torch.zeros(B, dtype=torch.bool, device=dev)
        self.hunger_f, self.hunger_s = self._hungers()
        self.f_done = self.hunger_f < 0; self.s_done = self.hunger_s < 0
        self.done = torch.zeros(B, dtype=torch.bool, device=dev); self.t = 0
        self.stats = {k: torch.zeros(B, device=dev) for k in ('wanted', 'other', 'rotten', 'lion', 'tiger', 'delivered', 'starved')}
        return self

    def _place_predators(self):
        B, dev = self.B, self.device
        pos = torch.zeros(B, 4, 2, dtype=torch.long, device=dev)
        need = torch.ones(B, 4, dtype=torch.bool, device=dev)
        while need.any():
            cell = self._randint(9, B, 4)
            cand = self.origin[None] + torch.stack((cell // 3, cell % 3), -1)
            d = cheb(cand, self.food_pos)
            ok = torch.where(self.pred_state == GUARDING, d == 1, d >= 1)
            take = need & ok
            pos[take] = cand[take]; need &= ~ok
        return pos

    def _sample_present(self):
        """A food type present on the map (uniform over present types)."""
        present = torch.zeros(self.B, 3, device=self.device)
        present.scatter_add_(1, self.food_type, self.food_alive.float())
        present = (present > 0).float()
        return torch.multinomial(present + 1e-9 * (present.sum(1, keepdim=True) == 0), 1, generator=self.g).squeeze(1)

    def _hungers(self):
        B, dev = self.B, self.device
        none = torch.full((B,), -1, device=dev)
        if self.stage == 1: return self._sample_present(), none
        if self.stage == 2:
            scout = self._rand(B) < 0.5; h = self._sample_present()
            return torch.where(scout, none, h), torch.where(scout, h, none)
        return self._sample_present(), self._sample_present()

    # ------------------------------------------------------------------ percepts
    def render(self, scout=False):
        """[B, NCH, SIZE, SIZE].  In stage 3 the scout cannot tell ripe from rotten."""
        B = self.B; dev = self.device
        m = self.base.expand(B, -1, -1, -1).clone()
        b = torch.arange(B, device=dev)[:, None].expand(B, 4)
        fa = self.food_alive; fr, fc = self.food_pos[..., 0], self.food_pos[..., 1]
        m[b[fa], 2 + self.food_type[fa], fr[fa], fc[fa]] = 1
        if not (scout and self.stage == 3):
            ripe = fa & self.food_ripe; rot = fa & ~self.food_ripe
            m[b[ripe], 5, fr[ripe], fc[ripe]] = 1; m[b[rot], 6, fr[rot], fc[rot]] = 1
        po = self.pred_on; pr, pc = self.pred_pos[..., 0], self.pred_pos[..., 1]
        m[b[po], 7 + self.pred_type[po], pr[po], pc[po]] = 1
        m[b[po], 9 + self.pred_state[po], pr[po], pc[po]] = 1
        m[torch.arange(B, device=dev), 13, self.pos[:, 0], self.pos[:, 1]] = 1
        return m

    def view(self, full=None):
        """The forager's 3 x 3 surroundings: [B, NCH, 3, 3]."""
        full = self.render() if full is None else full
        off = torch.arange(-1, 2, device=self.device)
        rr = (self.pos[:, 0, None] + off); cc = (self.pos[:, 1, None] + off)
        b = torch.arange(self.B, device=self.device)[:, None, None]
        return full[b, :, rr[:, :, None], cc[:, None, :]].permute(0, 3, 1, 2)

    # ------------------------------------------------------------------ dynamics
    def _eat_reward(self, hunger, ftype, ripe):
        return torch.where(~ripe, torch.full_like(ftype, R_ROTTEN, dtype=torch.float),
                           torch.where(ftype == hunger, torch.full_like(ftype, R_WANTED, dtype=torch.float),
                                       torch.full_like(ftype, R_OTHER, dtype=torch.float)))

    def _record_eat(self, mask, hunger, ftype, ripe):
        self.stats['wanted'] += (mask & ripe & (ftype == hunger)).float()
        self.stats['other'] += (mask & ripe & (ftype != hunger)).float()
        self.stats['rotten'] += (mask & ~ripe).float()

    def step(self, action):
        B, dev = self.B, self.device
        live = ~self.done
        r = torch.zeros(B, device=dev)
        # move (walls: the floor is rows/cols 1..9)
        step = MOVES.to(dev)[action] * live[:, None]
        self.pos = (self.pos + step).clamp(1, SIZE - 2)
        here = (self.food_pos == self.pos[:, None]).all(-1) & self.food_alive            # [B, 4]
        on = here.any(1); which = here.float().argmax(1); bi = torch.arange(B, device=dev)
        ftype, fripe = self.food_type[bi, which], self.food_ripe[bi, which]
        # eat: a hungry forager eats the food it steps on
        eat = live & on & ~self.f_done
        r += torch.where(eat, self._eat_reward(self.hunger_f, ftype, fripe), torch.zeros_like(r))
        self._record_eat(eat, self.hunger_f, ftype, fripe)
        self.f_done |= eat; self.food_alive[bi[eat], which[eat]] = False
        # pick up: once the forager is fed (or not hungry), it carries food it steps on home to a hungry scout
        pick = live & on & ~eat & self.f_done & (self.carry < 0) & ~self.s_done
        self.carry = torch.where(pick, ftype, self.carry); self.carry_ripe = torch.where(pick, fripe, self.carry_ripe)
        self.food_alive[bi[pick], which[pick]] = False
        # deliver: the scout eats when the forager reaches home with food
        home = ((self.pos >= HOME_LO) & (self.pos <= HOME_HI)).all(1)
        deliver = live & home & (self.carry >= 0) & ~self.s_done
        r += torch.where(deliver, self._eat_reward(self.hunger_s, self.carry.clamp_min(0), self.carry_ripe), torch.zeros_like(r))
        self._record_eat(deliver, self.hunger_s, self.carry, self.carry_ripe)
        self.stats['delivered'] += deliver.float()
        self.s_done |= deliver; self.carry = torch.where(deliver, torch.full_like(self.carry, -1), self.carry)
        # predators.  Home is a refuge: nothing wakes, chases or catches a forager standing in it.
        home_now = ((self.pos >= HOME_LO) & (self.pos <= HOME_HI)).all(1)
        out = (live & ~home_now)[:, None]
        d = cheb(self.pred_pos, self.pos[:, None])                                      # [B, 4]
        wake = self.pred_on & (self.pred_state == ASLEEP) & (self.pred_type == TIGER) & (d <= WAKE_RADIUS) & out
        self.pred_state = torch.where(wake, torch.full_like(self.pred_state, HUNTING), self.pred_state)
        moves = self.pred_on & (self.pred_state == HUNTING) & (d <= HUNT_RADIUS) & out & ((self.pred_type == TIGER) | (self.t % 2 == 0))
        delta = (self.pos[:, None] - self.pred_pos).sign()
        self.pred_pos = torch.where(moves[..., None], self.pred_pos + delta, self.pred_pos)
        d = cheb(self.pred_pos, self.pos[:, None])
        near_food = cheb(self.food_pos, self.pos[:, None]) <= 1
        active = self.pred_on & ((self.pred_state == HUNTING) | (self.pred_state == GUARDING))
        catch = active & ((d <= 1) | ((self.pred_state == GUARDING) & near_food)) & out
        caught = catch.any(1)
        pen = torch.where(catch, R_CAUGHT.to(dev)[self.pred_type], torch.zeros_like(d, dtype=torch.float)).min(1).values
        r += torch.where(caught, pen, torch.zeros_like(r))
        by_tiger = (catch & (self.pred_type == TIGER)).any(1)
        self.stats['tiger'] += (caught & by_tiger).float(); self.stats['lion'] += (caught & ~by_tiger).float()
        r += R_TICK * live.float()
        self.t += 1
        ending = live & ((self.f_done & self.s_done) | caught | (self.t >= self.T))
        # an agent that is still hungry when the episode ends goes without food
        r += torch.where(ending, R_STARVE * ((~self.f_done).float() + (~self.s_done).float()), torch.zeros_like(r))
        self.stats['starved'] += torch.where(ending, (~self.f_done).float() + (~self.s_done).float(), torch.zeros_like(r))
        self.done = self.done | ending
        return r * live.float()

    # ------------------------------------------------------------------ helpers
    STATE_KEYS = ('food_pos', 'food_type', 'food_ripe', 'food_alive', 'pred_on', 'pred_type', 'pred_state', 'pred_pos',
                  'pos', 'carry', 'carry_ripe', 'hunger_f', 'hunger_s', 'f_done', 's_done', 'done')

    def snapshot(self):
        return {k: getattr(self, k).clone() for k in self.STATE_KEYS} | {'t': self.t}

    def restore(self, s):
        for k in self.STATE_KEYS: setattr(self, k, s[k].clone())
        self.t = s['t']; self.B = s['pos'].shape[0]
        self.stats = {k: torch.zeros(self.B, device=self.device) for k in ('wanted', 'other', 'rotten', 'lion', 'tiger', 'delivered', 'starved')}
        return self

    def describe(self, b):
        out = []
        for g in range(4):
            parts = []
            if self.food_alive[b, g]:
                parts.append(('ripe ' if self.food_ripe[b, g] else 'rotten ') + FOODS[int(self.food_type[b, g])])
            if self.pred_on[b, g]:
                parts.append(f"{ANIMALS[int(self.pred_type[b, g])]} {STATES[int(self.pred_state[b, g])]}")
            out.append(f"{GLADES[g]}: {', '.join(parts) or 'empty'}")
        h = lambda x: FOODS[int(x)] if x >= 0 else '-'
        return '; '.join(out) + f" | forager wants {h(self.hunger_f[b])}, scout wants {h(self.hunger_s[b])}"
