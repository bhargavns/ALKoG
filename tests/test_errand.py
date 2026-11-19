"""Rule checks for the errand world."""
import torch
from foraging.errand.env import (World, UP, DOWN, LEFT, RIGHT, STAY, LION, TIGER, ASLEEP, AWAKE, CAGED,
                                    R_ITEM, R_DONE, R_TICK, HELD_OUT)


def scene(food=(((2, 2), 0, True),), animal=None, fed=(False, False), pos=(5, 5), need_any=1, need=(0, 0, 0),
          rules='private', phase='count', scout_ripe=True):
    w = World(1, rules, phase, scout_ripe, generator=torch.Generator().manual_seed(0)).reset()
    w.food_alive[:] = False; w.an_on[:] = False
    for i, ((r, c), ft, ripe) in enumerate(food):
        w.food_pos[0, i] = torch.tensor([r, c]); w.food_type[0, i] = ft; w.food_ripe[0, i] = ripe; w.food_alive[0, i] = True
    if animal:
        (r, c), at, st = animal
        w.an_pos[0, 0] = torch.tensor([r, c]); w.an_type[0, 0] = at; w.an_state[0, 0] = st; w.an_on[0, 0] = True
    w.fed[0] = torch.tensor(fed)
    w.pos[0] = torch.tensor(pos); w.was_home[0] = True
    w.need[0] = torch.tensor(list(need) + [need_any])
    return w


def act(w, *actions):
    return sum(float(w.step(torch.tensor([a]))[0]) for a in actions)


def test_fetch_one_item_completes():
    w = scene(pos=(2, 3))
    r = act(w, LEFT)                                  # pick up at (2,2)
    assert w.carry[0] == 0 and not w.done[0]
    r += act(w, DOWN, DOWN, RIGHT, RIGHT)             # (4,4): home
    assert w.done[0] and w.stats['items'][0] == 1 and w.stats['complete'][0] == 1
    assert abs(r - (R_ITEM + R_DONE + 5 * R_TICK)) < 1e-6


def test_rotten_does_not_count():
    w = scene(food=(((2, 2), 0, False),), pos=(2, 3))
    r = act(w, LEFT, DOWN, DOWN, RIGHT, RIGHT)
    assert not w.done[0] and w.stats['rotten'][0] == 1 and abs(r - (-0.3 + 5 * R_TICK)) < 1e-6


def test_two_items_need_two_trips():
    w = scene(food=(((2, 2), 0, True), ((2, 3), 1, True)), pos=(2, 4), need_any=2)
    act(w, LEFT)                                      # picks berry at (2,3)
    act(w, LEFT)                                      # hands full: walks over the apple without taking it
    assert w.carry[0] == 1 and w.food_alive[0, 0]
    act(w, DOWN, DOWN, RIGHT, RIGHT)                  # home with the berry
    assert not w.done[0] and w.stats['items'][0] == 1
    act(w, UP, UP, LEFT, LEFT, DOWN, DOWN, RIGHT, RIGHT)
    assert w.done[0] and w.stats['complete'][0] == 1


def test_described_objective_ignores_unneeded_type():
    w = scene(food=(((2, 2), 0, True),), pos=(2, 3), phase='described', need_any=0, need=(0, 1, 0))
    r = act(w, LEFT, DOWN, DOWN, RIGHT, RIGHT)        # an apple, but a berry was wanted
    assert not w.done[0] and w.stats['unneeded'][0] == 1 and abs(r - 5 * R_TICK) < 1e-6


def test_fed_animals_ignore_the_forager():
    for t in (LION, TIGER):
        w = scene(animal=((3, 3), t, AWAKE), fed=(True, True), pos=(3, 4))
        act(w, STAY, STAY, STAY)
        assert not w.done[0]


def test_hungry_awake_tiger_catches_lion_is_slower():
    w = scene(animal=((1, 9), TIGER, AWAKE), pos=(4, 9))   # (4,9) is outside home (cols 4..6)
    act(w, STAY, STAY)
    assert w.done[0] and w.stats['tiger'][0] == 1
    w = scene(animal=((1, 9), LION, AWAKE), pos=(4, 9))
    act(w, STAY, STAY)
    assert w.an_pos[0, 0].tolist() == [2, 9] and not w.done[0]


def test_sleeping_lion_safe_sleeping_tiger_depends_on_fed():
    w = scene(animal=((3, 3), LION, ASLEEP), pos=(3, 4))
    act(w, STAY, STAY, STAY)
    assert not w.done[0]
    hungry = scene(animal=((3, 3), TIGER, ASLEEP), pos=(3, 5))
    r = act(hungry, STAY)
    assert hungry.done[0] and abs(r - (-2.0 + R_TICK)) < 1e-6
    fed = scene(animal=((3, 3), TIGER, ASLEEP), fed=(False, True), pos=(3, 5))
    act(fed, STAY, STAY, STAY)
    assert not fed.done[0] and fed.an_state[0, 0] == AWAKE      # woke, but it has eaten


def test_caged_is_harmless_and_home_is_a_refuge():
    w = scene(animal=((3, 3), TIGER, CAGED), pos=(3, 4))
    act(w, STAY, STAY)
    assert not w.done[0]
    w = scene(animal=((3, 4), TIGER, AWAKE), pos=(4, 4))
    act(w, STAY, STAY)
    assert not w.done[0]


def test_fixed_rules_everyone_hungry():
    w = World(512, 'fixed', generator=torch.Generator().manual_seed(1)).reset()
    assert not w.fed.any()
    w = World(512, 'private', generator=torch.Generator().manual_seed(1)).reset()
    assert 0.3 < w.fed.float().mean() < 0.7


def test_briefing_only_on_arrival_home():
    w = scene(pos=(4, 4))
    assert w.briefing()[0]                            # tick 0
    act(w, STAY)
    assert not w.briefing()[0]                        # still home, no new briefing
    act(w, UP)                                        # (3,4): out
    assert not w.briefing()[0]
    act(w, DOWN)                                      # back in
    assert w.briefing()[0]


def test_held_out_combinations():
    w = World(4096, generator=torch.Generator().manual_seed(2)).reset(held_out='exclude')
    assert not w.held_out_mask().any()
    w = World(4096, generator=torch.Generator().manual_seed(2)).reset(held_out='only')
    assert w.held_out_mask().any(1).all()


def test_scout_map_without_ripeness():
    w = scene(scout_ripe=False)
    assert w.render()[0, 5].sum() == 1 and w.render(scout=True)[0, 5:7].sum() == 0
    assert w.view()[0].shape == (13, 3, 3)


def test_fast_replay_matches_stepwise():
    from foraging.errand.train import play, replay, replay_stepwise
    from foraging.errand.agents import Scout, Forager
    torch.manual_seed(0)
    S, Fo = Scout(), Forager()
    for talks, phase in ((False, 'count'), (True, 'described')):
        cfg = dict(rules='private', phase=phase, scout_ripe=False, forager_talks=talks, silent=False, cost_now=0.)
        w = World(24, 'private', phase, False, generator=torch.Generator().manual_seed(3)).reset()
        d = play(w, S, Fo, cfg)
        assert d['brief'][1:].any()                      # some briefings after tick 0
        idx = torch.arange(0, 24, 2)
        a, b = replay(S, Fo, d, idx, cfg), replay_stepwise(S, Fo, d, idx, cfg)
        for k in a:
            assert torch.allclose(a[k], b[k], atol=1e-4), k
        assert torch.allclose(a['lp_act'], d['lp_act'][:, idx], atol=1e-4)
