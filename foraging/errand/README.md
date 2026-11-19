# Errand: a scout briefs a forager who has a job to do

Follows the watchtower study
(`foraging/watchtower`), where the scout knew everything it needed to make
the decision itself and so learned only "go to glade X".  Here the knowledge is
split so that neither agent can decide alone, and the scout may speak only
while the forager is at home.

## The world (`env.py`)

The watchtower grid: 11 x 11 with walls (9 x 9 floor), a 3 x 3 **home** in the
centre under the scout's tower, four 3 x 3 **glades** in the corners.  The
forager starts in the middle of home.  Everything runs as a batch of torch
tensors on the GPU.

Each glade holds **two foods** (apple, berry or mushroom; each ripe with
probability 0.6) and, with probability 0.7, **one animal** (lion or tiger,
50/50) that is asleep (0.4), awake (0.4) or caged (0.2).

**Fed or hungry.**  Each episode, lions as a species are either fed or hungry
(50/50), and so are tigers, independently.  A fed animal ignores the forager.
A hungry one behaves as follows:

| | lion | tiger |
|---|---|---|
| asleep | deep sleeper: never wakes, harmless | light sleeper: wakes if the forager comes within 2 cells, then hunts |
| awake | hunts: moves toward a forager within 3 cells, one cell every **2** ticks | hunts: same, but every tick |
| caged | harmless | harmless |

A hunting animal catches the forager when it is within 1 cell.  Home is a
refuge: nothing wakes, hunts or catches a forager standing in it.  A fed tiger
that is woken is awake but still ignores the forager.

So a sleeping lion and a caged animal are always safe; everything else depends
on whether that species is fed today.

**Carrying.**  The forager carries one item at a time.  Stepping on food with
empty hands picks it up (there is no separate pick-up action).  Reaching home
with an item hands it in.  Actions: up, down, left, right, stay.

## The objective

Each episode draws an objective, shown only to the forager.  It is not an
agent; it is the episode's goal.

* **Phase A (count):** bring home N ripe items, N in {1, 2, 3}, any type.
* **Phase B (described):** bring home specific items, e.g. "2 berries" or
  "1 apple and 1 mushroom": up to 3 items, each of a named type.

## Rewards (shared by the team)

| event | reward |
|---|---|
| an item handed in that is ripe and still needed by the objective | +0.3 |
| a rotten item handed in (does not count) | -0.3 |
| a ripe item that is not needed (phase B, does not count) | 0 |
| the whole objective completed | +2, episode ends |
| caught by a lion | -1, episode ends |
| caught by a tiger | -2, episode ends |
| each tick until the episode ends | -0.01 |
| each symbol spoken | 0 (was -0.005 from 40% of training on; removed after pilot 1, see below) |

Episodes last at most 60 ticks.  Staying home scores -0.60; there is no
hunger and no starvation.

## Who knows what

| | scout (tower) | forager (ground) |
|---|---|---|
| where the food and animals are, each animal's type and state | yes, the whole map | only the 3 x 3 cells around it |
| ripeness | ablation: yes / no | up close |
| whether lions and tigers are fed today | **no** (except in the "shared" ablation) | yes |
| the objective | no | yes |
| progress (what has been handed in) | yes | yes |

The scout cannot judge which glade is safe, because it does not know who is
fed; it can only report what is where.  The forager cannot see where things are,
but can judge danger and knows what it needs.

**Who speaks when.**  A *briefing* happens at tick 0 and on every tick the
forager steps back into home.  At a briefing the scout speaks: up to 8 symbols
from 16, ending with stop (stop first = silence).  In phase B's two-way
condition the forager speaks first at each briefing and the scout hears it
before replying.  The forager keeps the latest briefing in memory.

## Conditions

Phase A, each with 3 seeds and a silent control (no symbols ever pass):

| rules | what varies | who knows if lions/tigers are fed |
|---|---|---|
| fixed | nothing: every animal is hungry, every episode | everyone (they are always hungry) |
| **private** | fed/hungry per species, per episode | the forager only |
| shared | fed/hungry per species, per episode | both |

crossed with **scout sees ripeness: yes / no**.  The fixed and shared
conditions are the controls for private: fixed shows what happens when the
scout can judge danger itself; shared separates "the rules vary" from "only the
forager knows them".

Phase B (after phase A) uses private rules and whichever ripeness setting
worked in phase A, with the forager mute vs speaking.

## Learning (`agents.py`, `train.py`)

As in the watchtower: recurrent PPO from the team reward only, independent
learners (no gradient crosses), the message encoder and word decoder from
`watchtower/agents.py`, centralised critics during training (value heads also
see the fed flags and the objective).  The scout's memory updates every tick;
its words are scored only at briefings, and the reward that follows a briefing
is the team reward until the next one.  Learning aids, fixed in advance: for the
first 20% of training the objective is N = 1; the symbol cost starts at 40%.

**Held-out combinations.**  Four (glade, animal, state) combinations never occur
in training: tiger asleep in NE, lion awake in SW, tiger caged in NW, lion asleep
in SE.  Each word for each part occurs in training, just never in that
combination.

## Pilots and revisions (before any reported run)

**Pilot 1** (`results/errand/pilot`, 300 updates of 2,048 episodes; private
speaking, private silent, fixed speaking).  No hiding: every pair foraged and
completed objectives.  But speaking did not help: at update 300, private
speaking +0.42 vs private silent +0.71; fixed speaking -0.53.  The scouts spoke
6 symbols per briefing until the symbol cost started (update 120) and fell
silent within about 100 updates (0.2 symbols); the foragers had not yet learned
to use the briefings, so speaking was pure cost.  Revision: **no symbol cost**,
runs of 1,000 updates.  Training was also made about 2.7x faster without
changing what is computed (`replay` batches everything except the two memory
updates; `test_fast_replay_matches_stepwise` checks it against the tick-by-tick
version).

**Pilot 2** (`results/errand/pilot2`): private speaking vs private silent,
seed 0, 1,000 updates (about 43 and 35 minutes, one run at a time).  The
briefings help: return +1.33 vs silent +0.88 (+0.45); blocking or shuffling
them drops the pair to +0.69 / +0.70, below silent, so the forager relies on
them.  Most of the gain is fewer tiger catches (0.10 vs 0.20 per episode).  On
held-out maps +1.24 vs silent +0.47.  But the language is not yet descriptive:
the scout always speaks 8 symbols, mostly one symbol repeated ("10 10 10 10
..."), 155 distinct first briefings, topsim 0.10; sleeping lion -> tiger changes
the briefing 36% of the time vs 13% for caged (the right direction, below the
70% line); flipping the forager's fed flags changes its first destination 34%
of the time (line: 50%), and it walks first into a dangerous glade only 4% of
the time.  Pilot lines 1 and 4 pass, 2, 3, 5 fail (one seed; not a reported
result).  The speaking run was still improving at 1,000 updates (0.92 at 600,
1.23 at 800, 1.28 at 1,000); the silent one had levelled off (about 0.8).

## Running

    python -m foraging.errand.train --rules private --seed 0 --updates 2000 --output results/errand/X/private_0
    python -m foraging.errand.train --rules private --silent --seed 0 --updates 2000 --output results/errand/X/private_silent_0
    python -m foraging.errand.analysis --summarise results/errand/X

`--mem-fraction` caps the share of GPU memory per process (default 0.5).  On a
Slurm node, `scripts/slurm/errand_private2k.sh` runs the three-seed private
study with all six runs on one GPU.

## Pass lines, fixed before any run

"Return" is mean team reward per episode on fresh maps, sampling as in
training.  A condition passes a line if at least 2 of 3 seeds do.

1. **Useful.**  Return at least 0.3 above the silent control of the same seed,
   and blocking or shuffling the briefings removes at least half of that gain.
2. **The forager decides.**  On maps where flipping the forager's private
   fed/hungry flags changes which glades are dangerous, the forager's first
   destination changes in at least 50% of cases.  (If the scout were giving
   orders, the flags would not matter.)  Private and shared rules only.
3. **Animals are described.**  The scout's first briefing changes for at least
   70% of maps when a sleeping lion becomes a sleeping tiger (which can matter)
   and for at most 30% when a caged lion becomes a caged tiger (which cannot).
4. **Held-out combinations.**  On maps containing a held-out combination, the
   return is at least 0.3 above the silent control's return on the same maps.
5. **Structure** (reported for every run, a pass for at least 2 seeds):
   topographic similarity >= 0.3 between scenes and first briefings.

Predictions written before running: fixed rules give a pointer language as in
the watchtower (lines 1 and maybe 3 only); private rules are the condition that
should pass 2-4; shared rules sit in between.
