# Community: one scout, a population of foragers, and transmission

Follows the [errand study](../errand/README.md).  There, briefings were useful (return +0.47 to
+0.61 above silent, tiger catches nearly gone) but the code was holistic: each scout reported mainly on
one or two glades, a change to one glade rewrote the whole message, and positional / bag-of-symbols
disentanglement were 0.03-0.06 (`foraging/errand/words.py`).  A pair that grows up together can get by
with a private code and a routine ("is my usual glade safe?").  This study changes the world so that
routine fails (step 1), then adds transmission so the language must be learnable by newcomers (step 2).

## Step 1: a world where description is needed (`env.py`)

Everything not listed here is as in the errand world (grid, home, animals and their rules, fed/hungry
per species per episode, rewards, 60 ticks, briefings at tick 0 and on each return home, 16 symbols, up
to 8 per briefing).

* **Sparse food.** Each glade holds food (two items) with probability 0.5, at least one glade per map;
  which glades differ every episode.  Items are apple, berry or mushroom, ripe with probability 0.6;
  every map has at least one ripe item.
* **Named objectives.** The objective is 1 or 2 named items (1 only, during the first 20% of training),
  drawn from the ripe food on the map, so it can always be done.  Ripe items not named score 0; rotten
  ones -0.3.
* **Four foragers who know different things.** Episode b belongs to forager b*4//B, each with its own
  parameters.  Foragers 0 and 2 know whether lions are fed today; foragers 1 and 3 know whether tigers
  are.  The scout knows neither, and does not know which forager it is talking to or what it wants.
* **More held out.**  Six (glade, animal, state) combinations, each (animal, state) pair held out in
  exactly one glade: tiger asleep NW, lion caged NW, lion awake NE, tiger caged NE, tiger awake SW,
  lion asleep SE.  Three (glade, food) combinations: berry in NE, apple in SW, mushroom in SE.  None occurs
  in training.  Held-out test maps contain one of each, and the held-out food is part of the objective.

Why this should force description: with food in different glades every episode and objectives naming
foods the scout never sees, the only briefing that serves every forager is one that says what is where.
A forager who knows only about lions needs to hear where the tigers are, and the other way round.

## Step 2: transmission (`train.py`)

* **Forager turnover** (`--turnover 100`): every 100 updates the oldest forager is replaced by a fresh
  one (round robin), so each lives 400 updates and the scout keeps teaching newcomers.
* **Scout generations** (`--generation 500 --bottleneck 512`): at updates 500, 1000 and 1500 a fresh scout
  first imitates the old scout's first briefings on a fixed set of 512 maps (300 Adam steps), then
  carries on with PPO alongside the foragers.  The 512 maps are the bottleneck: the new scout must
  generalise the rest.  Control: `--bottleneck 0`, imitation on fresh maps at every step.

## Conditions (3 seeds each, 2,000 updates of 2,048 episodes)

| batch | runs | what it tests |
|---|---|---|
| A | `speak_s`, `silent_s` | step 1 against a silent community |
| B | `turnover_s`, `generations_s` (turnover + scout generations with bottleneck) | step 2 |
| C | `nobottleneck_s` (turnover + scout generations, no bottleneck) | is it the bottleneck? |

## Pass lines, fixed before any run (a condition passes if 2 of 3 seeds do)

Measured on 4,096 fresh maps; "decoding gain" is the held-out accuracy of a linear decoder reading the
fact from the first briefing, minus always guessing the most common value.

1. **Useful.** Return at least 0.3 above the silent community of the same seed, and blocking or
   shuffling the briefings removes at least half the gain.
2. **Coverage.** Whether each glade holds a possibly dangerous animal decodes with gain >= 0.10 in all
   four glades, and at least 8 of the 12 (glade, ripe food type) facts decode with gain >= 0.05.  (Errand:
   1-2 glades.)
3. **Held out.** On held-out maps, return at least 0.3 above the silent community on the same maps.
4. **Structure.** max(positional, bag-of-symbols disentanglement) >= 0.15 and topographic similarity
   >= 0.3.
5. **Learnability** (turnover runs).  New foragers in the second half of the run reach 80% of the other
   foragers' return in at least 25% fewer updates (median) than those in the first half.
6. **Transmission helps structure** (batch B vs A, same seed).  Disentanglement up by >= 0.05 and topsim
   up by >= 0.05.  For scout generations, also better than `nobottleneck` on the same measures.

Predictions: step 1 passes 1 and 2 and does better than the errand study on structure, without reaching
line 4.  Turnover passes 5; scout generations with the bottleneck pass 6 and beat the no-bottleneck
control.

## Running

    python -m foraging.community.train --output results/community/speak_0 --seed 0
    python -m foraging.community.train --output results/community/silent_0 --seed 0 --silent
    python -m foraging.community.train --output results/community/turnover_0 --seed 0 --turnover 100
    python -m foraging.community.train --output results/community/generations_0 --seed 0 --turnover 100 --generation 500
    python -m foraging.community.analysis --summarise results/community

On a Slurm node: `scripts/slurm/community.sh` with `BATCH=A`, `B` or `C`.  About 3.3 s per update on an
RTX 4060 Ti for one run alone.
