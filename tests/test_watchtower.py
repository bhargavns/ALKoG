"""Rule checks for the watchtower world: each consequence comes from behaviour."""
import torch
from foraging.watchtower.env import (World, UP, DOWN, LEFT, RIGHT, STAY, LION, TIGER,
                                        ASLEEP, CAGED, GUARDING, HUNTING)


def scene(stage=1, food=((2, 2), 0, True), pred=None, pos=(5, 5), hunger_f=0, hunger_s=-1):
    """One episode with a single food in the NW glade and optionally one predator."""
    w = World(1, stage, generator=torch.Generator().manual_seed(0)).reset()
    w.food_alive[:] = False; w.pred_on[:] = False
    (fr, fc), ft, ripe = food
    w.food_pos[0, 0] = torch.tensor([fr, fc]); w.food_type[0, 0] = ft; w.food_ripe[0, 0] = ripe; w.food_alive[0, 0] = True
    if pred:
        (pr, pc), ptype, pstate = pred
        w.pred_pos[0, 0] = torch.tensor([pr, pc]); w.pred_type[0, 0] = ptype; w.pred_state[0, 0] = pstate; w.pred_on[0, 0] = True
    w.pos[0] = torch.tensor(pos)
    w.hunger_f[0] = hunger_f; w.hunger_s[0] = hunger_s
    w.f_done[:] = hunger_f < 0; w.s_done[:] = hunger_s < 0; w.done[:] = False
    return w


def act(w, *actions):
    total = 0.
    for a in actions: total += float(w.step(torch.tensor([a]))[0])
    return total


def test_eat_ripe_wanted_food_ends_episode():
    w = scene(pos=(2, 3))
    r = act(w, LEFT)                     # stepping onto the food eats it
    assert w.done[0] and abs(r - (1.0 - 0.01)) < 1e-6


def test_rotten_food_costs():
    w = scene(food=((2, 2), 0, False), pos=(2, 3))
    assert abs(act(w, LEFT) - (-0.5 - 0.01)) < 1e-6


def test_sleeping_lion_is_harmless_sleeping_tiger_wakes_and_catches():
    lion = scene(pred=((3, 3), LION, ASLEEP), pos=(3, 4))
    act(lion, STAY, STAY, STAY)
    assert not lion.done[0] and lion.pred_state[0, 0] == ASLEEP
    tiger = scene(pred=((3, 3), TIGER, ASLEEP), pos=(3, 5))
    r = act(tiger, STAY)                 # within 2 cells: wakes, steps closer, catches
    assert tiger.done[0] and abs(r - (-2.0 - 0.01 - 1.0)) < 1e-6      # caught, and never ate


def test_caged_predators_are_harmless():
    for t in (LION, TIGER):
        w = scene(pred=((3, 3), t, CAGED), pos=(3, 4))
        act(w, STAY, STAY)
        assert not w.done[0]


def test_guard_catches_near_its_food_and_lion_costs_less():
    w = scene(pred=((2, 3), LION, GUARDING), pos=(4, 1))
    r = act(w, UP)                       # now at (3,1): next to the food at (2,2)
    assert w.done[0] and abs(r - (-1.0 - 0.01 - 1.0)) < 1e-6


def test_hunting_lion_is_slow_tiger_is_fast():
    lion = scene(pred=((1, 9), LION, HUNTING), pos=(4, 9))
    act(lion, STAY)                      # t=0: within 3 cells, lion moves to (2,9)
    act(lion, STAY)                      # t=1: lion rests
    assert lion.pred_pos[0, 0].tolist() == [2, 9] and not lion.done[0]
    tiger = scene(pred=((1, 9), TIGER, HUNTING), pos=(4, 9))
    act(tiger, STAY, STAY)               # moves every tick: (2,9), (3,9) -> adjacent, catches
    assert tiger.pred_pos[0, 0].tolist() == [3, 9] and tiger.done[0]


def test_home_is_a_refuge():
    w = scene(pred=((3, 4), TIGER, HUNTING), pos=(4, 4))   # tiger right next to the home square
    act(w, STAY, STAY, STAY)
    assert not w.done[0] and w.pred_pos[0, 0].tolist() == [3, 4]
    w = scene(pred=((3, 3), TIGER, ASLEEP), pos=(5, 5))    # start square: a sleeping tiger 2 cells away stays asleep
    act(w, STAY)
    assert w.pred_state[0, 0] == ASLEEP and not w.done[0]


def test_delivery_feeds_the_scout():
    w = scene(stage=2, pos=(2, 3), hunger_f=-1, hunger_s=0)
    act(w, LEFT)                         # a forager that is not hungry picks the food up
    assert w.carry[0] == 0 and not w.food_alive[0, 0]
    r = act(w, DOWN, DOWN, RIGHT, RIGHT)  # to (4,4): home
    assert w.done[0] and w.stats['delivered'][0] == 1 and r > 0.9


def test_staying_home_hungry_starves():
    w = scene(pos=(5, 5))
    r = act(w, *[STAY] * 40)
    assert w.done[0] and abs(r - (-0.40 - 1.0)) < 1e-6 and w.stats['starved'][0] == 1


def test_stage3_scout_cannot_see_ripeness():
    w = scene(stage=3, hunger_s=1)
    full, scout = w.render(), w.render(scout=True)
    assert full[0, 5].sum() == 1 and scout[0, 5:7].sum() == 0
    assert w.view()[0].shape == (14, 3, 3)
