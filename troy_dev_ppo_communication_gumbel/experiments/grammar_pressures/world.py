"""Event world for the grammar-pressures study.

A scene is an *event*, not a bag of attributes:

    intransitive:  PRED1(S)          e.g. "the red lion is loose"
    transitive:    PRED2(A, P)       e.g. "the red lion guards the green apple"

Every participant is an entity = (type, colour).  Transitive events are fully
reversible (any entity can be agent or patient), so who-does-what-to-whom can
never be recovered from the entities alone.  Events have one or two
participants, so a fixed-length holistic packet is never forced by the task.

Nothing in the percept tells an agent how to talk about it.  A percept is an
unordered set of element vectors (one per participant, one for the predicate);
the participant's role (S / A / P) is a perceptual feature of the participant,
exactly like the left/right coordinates of the earlier binding task.  S is its
own perceptual role, so whether the language treats S like A (accusative
alignment), like P (ergative) or neither is left to the agents.

Held-out structure (all primitives are seen in training):

    novel_entity   4 (type, colour) descriptions never occur in training at all
    novel_pair     regular agent/patient pairs with (colour_A - colour_P) % 4 == 1
    construction   2 entities occur only in intransitive events in training,
                   tested inside transitive events
    novel_role     2 entities never occur as P in training (tested as P),
                   2 never occur as A (tested as A)
"""
import numpy as np
import torch

TYPES = ('apple', 'berry', 'grain', 'mushroom', 'lion', 'tiger')
COLOURS = ('red', 'green', 'yellow', 'blue')
PRED1 = ('loose', 'caged', 'ripe', 'gone')          # intransitive
PRED2 = ('guards', 'chases', 'near', 'hides')       # transitive
NT, NC, NP1, NP2 = len(TYPES), len(COLOURS), len(PRED1), len(PRED2)
NE = NT * NC
ROLES = ('S', 'A', 'P')
ELEMENT_DIM = NT + NC + 3 + NP1 + NP2               # one-hot blocks, see features()

# entity classes (index = type * NC + colour)
def _e(t, c): return t * NC + c
NOVEL = {_e(t, c) for t in range(NT) for c in range(NC) if (t + c) % 6 == 5}
S_ONLY = {_e(0, 0), _e(1, 3)}
A_ONLY = {_e(0, 2), _e(3, 0)}       # seen as S and A, never as P
P_ONLY = {_e(1, 1), _e(5, 2)}       # seen as S and P, never as A
REGULAR = set(range(NE)) - NOVEL - S_ONLY - A_ONLY - P_ONLY
PARTITIONS = ('train', 'novel_entity', 'novel_pair', 'construction', 'novel_role')
TEST_PARTITIONS = PARTITIONS[1:]


def _partition(valency, a, b):
    """Partition name of one event, or None if it is unused (mixed novelty)."""
    ents = [a] if valency == 1 else [a, b]
    special = [e for e in ents if e not in REGULAR]
    if valency == 1:
        return 'novel_entity' if a in NOVEL else 'train'
    if len(special) > 1:
        return None
    if not special:
        ca, cb = a % NC, b % NC
        return 'novel_pair' if (ca - cb) % NC == 1 else 'train'
    e = special[0]
    if e in NOVEL: return 'novel_entity'
    if e in S_ONLY: return 'construction'
    if e in A_ONLY: return 'train' if e == a else 'novel_role'
    return 'train' if e == b else 'novel_role'          # P_ONLY


class World:
    """Enumerates all events, their percepts, partitions and edit neighbours."""

    def __init__(self, device='cpu'):
        rows, part = [], []
        for p in range(NP1):
            for a in range(NE):
                rows.append((1, p, a, -1)); part.append(_partition(1, a, -1))
        for p in range(NP2):
            for a in range(NE):
                for b in range(NE):
                    if a != b:
                        rows.append((2, p, a, b)); part.append(_partition(2, a, b))
        self.events = np.array(rows)                       # valency, pred, e1, e2
        self.n = len(rows)
        self.partition = np.array([PARTITIONS.index(x) if x else -1 for x in part])
        self.lookup = {tuple(r): i for i, r in enumerate(rows)}
        self.features = self._features().to(device)
        self.neighbours, self.swap = self._neighbours()
        self.device = device
        self.pools = {name: torch.tensor(np.nonzero(self.partition == k)[0], device=device)
                      for k, name in enumerate(PARTITIONS)}
        self.neigh_t = torch.tensor(self.neighbours, device=device)
        self.swap_t = torch.tensor(self.swap, device=device)
        self.nslot_t = torch.tensor(self.neighbour_slot, device=device)
        self.part_t = torch.tensor(self.partition, device=device)
        self.val_t = torch.tensor(self.events[:, 0], device=device); self.events_t = torch.tensor(self.events, device=device)
        self.similarity = self._similarity().to(device)

    # ------------------------------------------------------------------ percepts
    def _features(self):
        f = torch.zeros(self.n, 3, ELEMENT_DIM)
        for i, (v, p, a, b) in enumerate(self.events):
            f[i, 0, NT + NC + 3 + (p if v == 1 else NP1 + p)] = 1      # predicate element
            for slot, e, role in ((1, a, 0 if v == 1 else 1), (2, b, 2)):
                if e < 0: continue
                f[i, slot, e // NC] = 1; f[i, slot, NT + e % NC] = 1; f[i, slot, NT + NC + role] = 1
        return f

    def _similarity(self):
        """Share of semantic slots two events have in common (0 across valencies).
        Used only by the graded-payoff condition: a near miss is worth more than a wild one."""
        ev = torch.tensor(self.events); v, p, a, b = ev.T
        eq = lambda x: (x[:, None] == x[None, :]).float()
        same = eq(p) + eq(a // NC) + eq(a % NC)
        two = (v == 2).float()
        same = same + two[:, None] * two[None, :] * (eq(b // NC) + eq(b % NC))
        return same / (3 + 2 * two)[:, None] * eq(v)

    # ------------------------------------------------------- minimal-edit graph
    def slots(self, i):
        """Semantic slots of event i as a dict; used by the analysis."""
        v, p, a, b = self.events[i]
        d = {'pred': p}
        if v == 1:
            d.update(S_type=a // NC, S_colour=a % NC)
        else:
            d.update(A_type=a // NC, A_colour=a % NC, P_type=b // NC, P_colour=b % NC)
        return d

    def edit(self, i, slot, value):
        """Event id obtained by setting one slot to a value (None if invalid)."""
        v, p, a, b = self.events[i]
        if slot == 'pred': p = value
        else:
            role, kind = slot.split('_')
            e = a if role in 'SA' else b
            t, c = e // NC, e % NC
            if kind == 'type': t = value
            else: c = value
            if role in 'SA': a = t * NC + c
            else: b = t * NC + c
        return self.lookup.get((v, p, a, b))

    def slot_sizes(self, i):
        v = self.events[i, 0]
        if v == 1: return {'pred': NP1, 'S_type': NT, 'S_colour': NC}
        return {'pred': NP2, 'A_type': NT, 'A_colour': NC, 'P_type': NT, 'P_colour': NC}

    def _neighbours(self):
        width = (NP2 - 1) + 2 * (NT - 1) + 2 * (NC - 1)
        table = -np.ones((self.n, width), dtype=np.int64); swap = -np.ones(self.n, dtype=np.int64)
        self.neighbour_slot = -np.ones((self.n, width + 1), dtype=np.int64)     # which slot each neighbour edits; 5 = role swap
        for i in range(self.n):
            cur = self.slots(i); out = []; which = []
            for s, (slot, size) in enumerate(self.slot_sizes(i).items()):
                for value in range(size):
                    if value != cur[slot]:
                        j = self.edit(i, slot, value)
                        if j is not None: out.append(j); which.append(s)
            table[i, :len(out)] = out; self.neighbour_slot[i, :len(out)] = which; self.neighbour_slot[i, -1] = 5
            v, p, a, b = self.events[i]
            if v == 2: swap[i] = self.lookup[(2, p, b, a)]
        return table, swap

    # ------------------------------------------------------------ episode sampler
    def sample(self, sets, partition='train', generator=None, n_cand=4, focus=(1, 2, 3), far=0, displaced=False, keyhole=None):
        """Balanced reference episodes.  Each set is an anchor event plus
        single-edit neighbours from the same partition.  `focus` lists how many
        different slots the distractors may contrast with the anchor (drawn
        uniformly per set): with focus 1 all rooms differ in the same respect
        (say, only the agent's colour), with 3 they differ in up to three.  The
        role-reversed event counts as a slot.  Every member of a set is used
        once as the private target, so ignoring the message scores 1/n_cand.
        `far` of the distractors are unrelated same-valency events instead of
        neighbours, so a room can be a near miss or a wild miss.
        `displaced`: the speaker's view of the rooms ('speaker_view') is a second,
        independent draw of rooms that contrast the same respects, so it knows
        what is at stake but not the exact alternatives the collector faces.
        `keyhole`: rooms are only glimpsed (see `view`)."""
        dev = self.device; pool = self.pools[partition]; k = PARTITIONS.index(partition)
        anchor = pool[torch.randint(len(pool), (sets,), generator=generator, device=dev)]
        neigh = torch.cat((self.neigh_t[anchor], self.swap_t[anchor][:, None]), 1); slot = self.nslot_t[anchor]
        ok = (neigh >= 0) & (self.part_t[neigh.clamp_min(0)] == k)
        avail = torch.zeros(sets, 6, device=dev).scatter_add(1, slot.clamp_min(0), ok.float()) > 0
        want = torch.tensor(focus, device=dev)[torch.randint(len(focus), (sets,), generator=generator, device=dev)]
        rank = (torch.rand(sets, 6, generator=generator, device=dev) + avail.float()).argsort(1, descending=True).argsort(1)
        chosen = (rank < want[:, None]) & avail                      # the slots this context contrasts
        ok = ok & chosen.gather(1, slot.clamp_min(0))
        def draw():
            score = torch.where(ok, torch.rand(neigh.shape, generator=generator, device=dev), torch.full(neigh.shape, -1., device=dev))
            top = score.topk(n_cand - 1, dim=1)
            ids = torch.cat((anchor[:, None], neigh.gather(1, top.indices)), 1)
            fill = torch.cat((torch.zeros(sets, 1, dtype=torch.bool, device=dev), top.values < 0), 1)
            if far: fill[:, n_cand - far:] = True
            need = fill.clone()
            while need.any():       # too few such neighbours: fill with unrelated same-valency events
                ids[need] = pool[torch.randint(len(pool), (int(need.sum()),), generator=generator, device=dev)]
                bad = self.val_t[ids] != self.val_t[ids[:, :1]]
                for c in range(1, ids.shape[1]):
                    bad[:, c] |= (ids[:, c:c + 1] == ids[:, :c]).any(1) | (ids[:, c:c + 1] == ids[:, c + 1:]).any(1)
                need = fill & bad
            return ids
        ids = draw(); other = draw() if displaced else None
        order = torch.rand(ids.shape, generator=generator, device=dev).argsort(1)
        batch = self.episodes(ids.gather(1, order))
        if displaced:       # episode (set, t): the other draw, with the room in t's original column replaced by the target
            view = other.repeat_interleave(n_cand, 0)
            view[torch.arange(len(view), device=dev), order.flatten()] = batch['goal']
            batch['speaker_view'] = self.features[view]
        else: view = batch['cand_ids']
        if keyhole is not None:
            batch['candidates'] = self.view(batch['cand_ids'], keyhole, generator)
            batch['speaker_view'] = self.view(view, keyhole, generator)
        else: batch['speaker_view'] = self.features[view]
        return batch

    def view(self, ids, extra=0., generator=None):
        """Rooms glimpsed through a keyhole.  ids [n, rooms] -> percepts in which a
        property (predicate; first participant's type, colour; second
        participant's type, colour) is visible only if the rooms differ in it,
        or by chance with probability `extra`.  Whatever is the same in every
        room is not seen at all, so a signal can only be matched against the
        properties themselves, never against the whole event."""
        ev = self.events_t[ids]; a, b = ev[..., 2], ev[..., 3]
        vals = torch.stack((ev[..., 1], a // NC, a % NC, b // NC, b % NC), -1)
        vis = (vals != vals[:, :1]).any(1)
        if extra > 0: vis = vis | (torch.rand(vis.shape, generator=generator, device=vis.device) < extra)
        m = torch.zeros(len(ids), 3, ELEMENT_DIM, device=ids.device)
        m[:, 0] = vis[:, 0:1]
        for el, (t, c) in ((1, (1, 2)), (2, (3, 4))):
            m[:, el, :NT] = vis[:, t:t + 1]; m[:, el, NT:NT + NC] = vis[:, c:c + 1]; m[:, el, NT + NC:NT + NC + 3] = (vis[:, t] | vis[:, c])[:, None]
        return self.features[ids] * m[:, None]

    def episodes(self, ids):
        """ids: [sets, c] candidate event ids -> balanced batch of c*sets episodes."""
        sets, c = ids.shape
        cand = ids.repeat_interleave(c, 0)
        target = torch.arange(c, device=ids.device).repeat(sets)
        goal = ids.flatten()
        return dict(private=self.features[goal], candidates=self.features[cand], target=target, goal=goal, cand_ids=cand)

    # ------------------------------------------------------------------- display
    def describe(self, i):
        v, p, a, b = self.events[i]
        ent = lambda e: f'{COLOURS[e % NC]} {TYPES[e // NC]}'
        return f'{PRED1[p]}({ent(a)})' if v == 1 else f'{PRED2[p]}({ent(a)}, {ent(b)})'


if __name__ == '__main__':
    w = World()
    print('events', w.n, {name: int((w.partition == k).sum()) for k, name in enumerate(PARTITIONS)}, 'unused', int((w.partition < 0).sum()))
    g = torch.Generator().manual_seed(0)
    for name in PARTITIONS:
        b = w.sample(64, name, g)
        c = b['cand_ids'].reshape(-1, 4, 4)[:, 0]
        assert (w.part_t[c] == PARTITIONS.index(name)).all() and all(len(set(r.tolist())) == 4 for r in c)
        assert (w.val_t[c] == w.val_t[c[:, :1]]).all()
        print(name, [w.describe(int(j)) for j in c[0]])
