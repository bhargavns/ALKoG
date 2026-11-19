"""The errand world: a batch of 11 x 11 grids simulated together (CPU or GPU).

A forager with a private objective and private knowledge of which animals are
fed; a scout on a tower that sees where everything is.  See README.md.
"""
import torch

SIZE = 11
FOODS = ('apple', 'berry', 'mushroom')
ANIMALS = ('lion', 'tiger')
STATES = ('asleep', 'awake', 'caged')
LION, TIGER = range(2)
ASLEEP, AWAKE, CAGED = range(3)
GLADES = ('NW', 'NE', 'SW', 'SE')
GLADE_ORIGIN = torch.tensor([[1, 1], [1, 7], [7, 1], [7, 7]])
HOME_LO, HOME_HI = 4, 6
START = (5, 5)
CHANNELS = ('wall', 'home', 'apple', 'berry', 'mushroom', 'ripe', 'rotten', 'lion', 'tiger',
            'asleep', 'awake', 'caged', 'forager')
NCH = len(CHANNELS)
C_RIPE, C_ROTTEN = 5, 6
UP, DOWN, LEFT, RIGHT, STAY = range(5)
NACT = 5
MOVES = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
FOOD_PER_GLADE = 2
NFOOD = 4 * FOOD_PER_GLADE
P_ANIMAL, P_RIPE, P_FED = 0.7, 0.6, 0.5
STATE_P = torch.tensor([0.4, 0.4, 0.2])
R_ITEM, R_ROTTEN, R_DONE = 0.3, -0.3, 2.0
R_CAUGHT = torch.tensor([-1.0, -2.0])          # lion, tiger
R_TICK = -0.01
HUNT_RADIUS, WAKE_RADIUS = 3, 2
T_MAX = 60
# (glade, animal, state) combinations that never occur in training
HELD_OUT = ((1, TIGER, ASLEEP), (2, LION, AWAKE), (0, TIGER, CAGED), (3, LION, ASLEEP))
STAT_KEYS = ('items', 'rotten', 'unneeded', 'complete', 'lion', 'tiger', 'briefings')


def cheb(a, b):
    return (a - b).abs().max(-1).values


class World:
    """B episodes.

    rules: 'fixed' (every animal hungry), 'private' or 'shared' (lions and tigers each fed with
           probability 0.5 per episode; 'shared' also shows this to the scout).
    phase: 'count' (bring N ripe items, any type) or 'described' (bring named items).
    scout_ripe: whether the scout's map shows ripeness.
    held_out: None (any scene), 'exclude' (training: held-out combinations never occur) or
              'only' (every scene contains at least one held-out combination).
    """

    def __init__(self, B, rules='private', phase='count', scout_ripe=True, T=T_MAX, device='cpu', generator=None):
        self.B, self.rules, self.phase, self.scout_ripe, self.T = B, rules, phase, scout_ripe, T
        self.device, self.g = device, generator
        base = torch.zeros(NCH, SIZE, SIZE, device=device)
        base[0, 0, :] = base[0, -1, :] = base[0, :, 0] = base[0, :, -1] = 1
        base[1, HOME_LO:HOME_HI + 1, HOME_LO:HOME_HI + 1] = 1
        self.base = base
        self.origin = GLADE_ORIGIN.to(device)
        self.r_caught = R_CAUGHT.to(device); self.moves = MOVES.to(device)

    # ------------------------------------------------------------------ reset
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device)

    def _randint(self, n, *shape):
        return torch.randint(n, shape, generator=self.g, device=self.device)

    def reset(self, max_items=3, held_out=None):
        B, dev = self.B, self.device
        # three distinct cells per glade: two foods and the animal's spot
        cells = self._rand(B, 4, 9).argsort(-1)[..., :3]
        xy = self.origin[None, :, None] + torch.stack((cells // 3, cells % 3), -1)        # [B, 4, 3, 2]
        self.food_pos = xy[:, :, :FOOD_PER_GLADE].reshape(B, NFOOD, 2)
        self.food_glade = torch.arange(4, device=dev).repeat_interleave(FOOD_PER_GLADE).expand(B, -1)
        self.food_type = self._randint(3, B, NFOOD)
        self.food_ripe = self._rand(B, NFOOD) < P_RIPE
        self.food_alive = torch.ones(B, NFOOD, dtype=torch.bool, device=dev)
        self.an_pos = xy[:, :, 2]
        self.an_on = self._rand(B, 4) < P_ANIMAL
        self.an_type = self._randint(2, B, 4)
        self.an_state = torch.multinomial(STATE_P.to(dev).expand(B * 4, 3), 1, generator=self.g).view(B, 4)
        if held_out == 'exclude': self._exclude_held_out()
        elif held_out == 'only': self._force_held_out()
        self.held = self.held_out_mask()
        # which species are fed today
        if self.rules == 'fixed': self.fed = torch.zeros(B, 2, dtype=torch.bool, device=dev)
        else: self.fed = self._rand(B, 2) < P_FED
        # the objective: need[b, 0:3] items of each named type, need[b, 3] items of any type
        n = 1 + self._randint(max_items, B)
        self.need = torch.zeros(B, 4, dtype=torch.long, device=dev)
        if self.phase == 'count':
            self.need[:, 3] = n
        else:
            pick = self._randint(3, B, 3)
            take = torch.arange(3, device=dev)[None] < n[:, None]
            self.need[:, :3].scatter_add_(1, pick, take.long())
        self.pos = torch.tensor(START, device=dev).expand(B, 2).clone()
        self.was_home = torch.ones(B, dtype=torch.bool, device=dev)
        self.carry = torch.full((B,), -1, device=dev); self.carry_ripe = torch.zeros(B, dtype=torch.bool, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev); self.t = 0
        self.stats = {k: torch.zeros(B, device=dev) for k in STAT_KEYS}
        return self

    def held_out_mask(self):
        """[B, 4]: glades whose (glade, animal, state) is a held-out combination."""
        m = torch.zeros(self.B, 4, dtype=torch.bool, device=self.device)
        for g, a, s in HELD_OUT:
            m[:, g] = self.an_on[:, g] & (self.an_type[:, g] == a) & (self.an_state[:, g] == s)
        return m

    def _exclude_held_out(self):
        while True:
            m = self.held_out_mask()
            if not m.any(): return
            redraw = torch.multinomial(STATE_P.to(self.device).expand(self.B * 4, 3), 1, generator=self.g).view(self.B, 4)
            self.an_state = torch.where(m, redraw, self.an_state)

    def _force_held_out(self):
        g = self._randint(len(HELD_OUT), self.B)
        combo = torch.tensor(HELD_OUT, device=self.device)[g]                                       # [B, 3]
        b = torch.arange(self.B, device=self.device)
        self.an_on[b, combo[:, 0]] = True; self.an_type[b, combo[:, 0]] = combo[:, 1]; self.an_state[b, combo[:, 0]] = combo[:, 2]

    # ------------------------------------------------------------------ percepts
    def render(self, scout=False):
        """[B, NCH, SIZE, SIZE].  The scout's map lacks ripeness when scout_ripe is False."""
        B, dev = self.B, self.device
        m = self.base.expand(B, -1, -1, -1).clone()
        bF = torch.arange(B, device=dev)[:, None].expand(B, NFOOD)
        fa = self.food_alive; fr, fc = self.food_pos[..., 0], self.food_pos[..., 1]
        m[bF[fa], 2 + self.food_type[fa], fr[fa], fc[fa]] = 1
        if not (scout and not self.scout_ripe):
            ripe = fa & self.food_ripe; rot = fa & ~self.food_ripe
            m[bF[ripe], C_RIPE, fr[ripe], fc[ripe]] = 1; m[bF[rot], C_ROTTEN, fr[rot], fc[rot]] = 1
        bA = torch.arange(B, device=dev)[:, None].expand(B, 4)
        on = self.an_on; ar, ac = self.an_pos[..., 0], self.an_pos[..., 1]
        m[bA[on], 7 + self.an_type[on], ar[on], ac[on]] = 1
        m[bA[on], 9 + self.an_state[on], ar[on], ac[on]] = 1
        m[torch.arange(B, device=dev), 12, self.pos[:, 0], self.pos[:, 1]] = 1
        return m

    def view(self, full=None):
        """The forager's 3 x 3 surroundings (ripeness always visible up close): [B, NCH, 3, 3]."""
        full = self.render() if full is None else full
        off = torch.arange(-1, 2, device=self.device)
        rr = self.pos[:, 0, None] + off; cc = self.pos[:, 1, None] + off
        b = torch.arange(self.B, device=self.device)[:, None, None]
        return full[b, :, rr[:, :, None], cc[:, None, :]].permute(0, 3, 1, 2)

    def at_home(self):
        return ((self.pos >= HOME_LO) & (self.pos <= HOME_HI)).all(1)

    def briefing(self):
        """[B] bool: the scout may speak now (tick 0, or the forager has just stepped into home)."""
        return ~self.done & self.at_home() & (~self.was_home | (self.t == 0))

    def remaining(self):
        return self.need.clone()

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        B, dev = self.B, self.device
        live = ~self.done
        r = torch.zeros(B, device=dev)
        self.was_home = self.at_home()
        self.pos = (self.pos + self.moves[action] * live[:, None]).clamp(1, SIZE - 2)
        bi = torch.arange(B, device=dev)
        # pick up: empty hands + stepping on food
        here = (self.food_pos == self.pos[:, None]).all(-1) & self.food_alive
        which = here.float().argmax(1)
        pick = live & here.any(1) & (self.carry < 0)
        self.carry = torch.where(pick, self.food_type[bi, which], self.carry)
        self.carry_ripe = torch.where(pick, self.food_ripe[bi, which], self.carry_ripe)
        self.food_alive[bi[pick], which[pick]] = False
        # hand in at home
        home = self.at_home()
        give = live & home & (self.carry >= 0)
        typ = self.carry.clamp_min(0)
        named = self.need[bi, typ] > 0
        anyn = self.need[:, 3] > 0
        good = give & self.carry_ripe & (named | anyn)
        rotten = give & ~self.carry_ripe
        unneeded = give & self.carry_ripe & ~named & ~anyn
        dec_named = good & named; dec_any = good & ~named
        self.need[bi[dec_named], typ[dec_named]] -= 1
        self.need[bi[dec_any], 3] -= 1
        r += R_ITEM * good.float() + R_ROTTEN * rotten.float()
        self.stats['items'] += good.float(); self.stats['rotten'] += rotten.float(); self.stats['unneeded'] += unneeded.float()
        self.carry = torch.where(give, torch.full_like(self.carry, -1), self.carry)
        complete = live & (self.need.sum(1) == 0)
        r += R_DONE * complete.float(); self.stats['complete'] += complete.float()
        # animals: only hungry ones are dangerous; home is a refuge
        out = (live & ~home & ~complete)[:, None]
        hungry = self.an_on & ~self.fed.gather(1, self.an_type)
        d = cheb(self.an_pos, self.pos[:, None])
        wake = self.an_on & (self.an_state == ASLEEP) & (self.an_type == TIGER) & (d <= WAKE_RADIUS) & out
        self.an_state = torch.where(wake, torch.full_like(self.an_state, AWAKE), self.an_state)
        hunting = hungry & (self.an_state == AWAKE)
        moves = hunting & (d <= HUNT_RADIUS) & out & ((self.an_type == TIGER) | (self.t % 2 == 0))
        self.an_pos = torch.where(moves[..., None], self.an_pos + (self.pos[:, None] - self.an_pos).sign(), self.an_pos)
        d = cheb(self.an_pos, self.pos[:, None])
        catch = hunting & (d <= 1) & out
        caught = catch.any(1)
        pen = torch.where(catch, self.r_caught[self.an_type], torch.zeros_like(d, dtype=torch.float)).min(1).values
        r += pen * caught.float()
        by_tiger = (catch & (self.an_type == TIGER)).any(1)
        self.stats['tiger'] += (caught & by_tiger).float(); self.stats['lion'] += (caught & ~by_tiger).float()
        r += R_TICK * live.float()
        self.t += 1
        self.done = self.done | (live & (complete | caught | (self.t >= self.T)))
        return r * live.float()

    # ------------------------------------------------------------------ helpers
    STATE_KEYS = ('food_pos', 'food_glade', 'food_type', 'food_ripe', 'food_alive', 'an_pos', 'an_on', 'an_type', 'an_state',
                  'held', 'fed', 'need', 'pos', 'was_home', 'carry', 'carry_ripe', 'done')

    def snapshot(self):
        return {k: getattr(self, k).clone() for k in self.STATE_KEYS} | {'t': self.t}

    def restore(self, s):
        for k in self.STATE_KEYS: setattr(self, k, s[k].clone())
        self.t = s['t']; self.B = s['pos'].shape[0]
        self.stats = {k: torch.zeros(self.B, device=self.device) for k in STAT_KEYS}
        return self

    def danger(self):
        """[B, 4] bool: the glade's animal can hurt the forager today (knowing who is fed)."""
        hungry = self.an_on & ~self.fed.gather(1, self.an_type)
        risky = (self.an_state == AWAKE) | ((self.an_state == ASLEEP) & (self.an_type == TIGER))
        return hungry & risky

    def describe(self, b):
        out = []
        for g in range(4):
            parts = [('ripe ' if self.food_ripe[b, i] else 'rotten ') + FOODS[int(self.food_type[b, i])]
                     for i in range(NFOOD) if self.food_glade[b, i] == g and self.food_alive[b, i]]
            if self.an_on[b, g]: parts.append(f"{ANIMALS[int(self.an_type[b, g])]} {STATES[int(self.an_state[b, g])]}")
            out.append(f"{GLADES[g]}: {', '.join(parts) or 'empty'}")
        fed = ', '.join(f"{a}s {'fed' if self.fed[b, i] else 'hungry'}" for i, a in enumerate(ANIMALS))
        need = self.need[b].tolist()
        want = ', '.join(f"{n} {FOODS[i]}" for i, n in enumerate(need[:3]) if n) or ''
        if need[3]: want = (want + ', ' if want else '') + f"{need[3]} any"
        return '; '.join(out) + f" | {fed} | objective: {want}"
