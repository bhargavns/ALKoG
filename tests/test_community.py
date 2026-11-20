"""Checks for the community world and trainer."""
import torch
from foraging.community.env import World, M, KNOWS, HELD_ANIMAL, HELD_FOOD
from foraging.community.train import play, replay_scout, replay_forager, blocks, imitate, first_tokens
from foraging.community.agents import Scout, Forager


def make(B=256, seed=0, **kw):
    return World(B, generator=torch.Generator().manual_seed(seed)).reset(**kw)


def test_food_only_in_some_glades_and_objective_is_feasible():
    w = make(512)
    glades_with_food = w.food_alive.view(512, 4, 2).any(2).sum(1)
    assert (glades_with_food >= 1).all() and glades_with_food.float().mean() < 3.5
    assert (w.need[:, 3] == 0).all() and (w.need[:, :3].sum(1) >= 1).all()
    ripe = w.food_alive & w.food_ripe
    for f in range(3):
        have = (ripe & (w.food_type == f)).sum(1)
        assert (w.need[:, f] <= have).all()


def test_held_out_combinations():
    w = make(2048, held_out='exclude')
    assert not w.held_out_mask().any()
    w = make(2048, held_out='only')
    assert w.held_out_mask().all() and w.animal_held_mask().any(1).all() and w.food_held_mask().any(1).all()
    wanted = (w.food_held_mask() & w.food_ripe)                       # the held-out food is part of the objective
    held_types = torch.where(wanted, w.food_type, torch.full_like(w.food_type, -1))
    for b in range(0, 2048, 97):
        types = set(held_types[b][held_types[b] >= 0].tolist())
        assert any(w.need[b, t] > 0 for t in types)


def test_each_forager_knows_only_its_species():
    w = make(64)
    for b in range(64):
        m = int(w.forager[b]); k = w.known_fed()[b]
        for i in range(2):
            assert (k[i] != 0) == bool(KNOWS[m][i])
    assert [s.stop - s.start for s in blocks(64, 'cpu')] == [16] * M


def test_replay_matches_play():
    torch.manual_seed(0)
    S, Fos = Scout(), [Forager() for _ in range(M)]
    w = make(32, seed=3)
    d = play(w, S, Fos, dict(rules='private', silent=False))
    assert d['brief'][1:].any()
    sl = blocks(32, 'cpu')
    idx = torch.arange(32)
    o = replay_scout(S, d, idx, dict(silent=False))
    br = d['brief']
    assert torch.allclose(o['lpS'][br], d['lpS'][br], atol=1e-4)
    for m, s in enumerate(sl):
        ix = torch.arange(s.start, s.stop)
        of = replay_forager(Fos[m], d, ix)
        a = d['alive'][:, ix]
        assert torch.allclose(of['lp_act'][a], d['lp_act'][:, ix][a], atol=1e-4)


def test_imitation_copies_a_scout():
    torch.manual_seed(0)
    old = Scout(); gen = torch.Generator().manual_seed(5)
    wf = lambda n: World(n, generator=gen).reset(held_out='exclude')
    new, same = imitate(old, wf, 64, 150, 3e-3, 'cpu', wf(64))
    assert same > 0.5
