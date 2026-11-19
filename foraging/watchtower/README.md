# Watchtower: communication in a world where things happen

The first scout-and-forager study.  It replaced an earlier symbolic game whose
event world contained impossible scenes such as `chases(apple, lion)`.  Here
every consequence comes from simulated behaviour, not from a table of meanings.

## The world (`env.py`)

An 11 x 11 grid with walls around the edge (9 x 9 floor).  A 3 x 3 **home** in
the centre with the scout's tower; four 3 x 3 **glades** in the corners.  The
forager starts in the middle of home.

Each glade holds one **food** (apple, berry or mushroom; ripe with probability
0.6) and, with probability 0.7, one **predator** (lion or tiger, 50/50) in one
of four states:

| state | lion | tiger |
|---|---|---|
| asleep | deep sleeper: harmless, never wakes | light sleeper: wakes and starts hunting if the forager comes within 2 cells |
| caged | harmless | harmless |
| guarding (sits next to the glade's food) | catches a forager that comes within 1 cell of it or of the food | same |
| hunting | waits until the forager is within 3 cells (and outside home), then moves 1 cell toward it every **2** ticks | same, but moves **every** tick |

A predator that is guarding or hunting catches the forager when it is within
1 cell (Chebyshev distance).  Being caught ends the episode.

**Revision after the stage-1 pilot (before any reported run):** home is a
refuge (nothing wakes, chases or catches a forager standing in it) and the
hunting radius is 3 cells, not 4.  With the original numbers every glade cell
was within reach of the start square, so a hunting tiger caught the forager at
home before it could act, in roughly 40% of episodes and whatever the agents
did.  Pilot (800 updates, original world): catches 30-60% per episode,
speaking seed 0 about -0.6 vs silent -0.9, both worse than staying home (-0.4).

**Second revision, after a second pilot on the refuge world:** both speaking
pairs learned to stay home for the whole episode (-0.40 exactly) and their
scouts fell silent; the silent pair foraged and scored -0.62.  Hiding was a
stable optimum because going hungry cost nothing.  A hungry agent that has not
eaten when the episode ends now pays -1 (starvation), so hiding scores -1.40,
a catch costs -2 (lion) or -3 (tiger) in total, and a successful trip earns +1.

**Third revision, after a third pilot:** with starvation, every pair (speaking
and silent) still hid for the whole episode (-1.40).  Even untrained they ate
almost never (0.00-0.01 meals per episode): eating required standing on food
*and* choosing the eat action, 1 in 7, so a randomly moving forager never
discovered that food pays.  The eat and pick-up actions are removed: a hungry
forager eats whatever food it steps on, and a forager that is fed (or not
hungry) picks up food it steps on when the scout is hungry.  Actions are now
up, down, left, right, stay; choosing food means choosing where to walk.

## Agents

**Scout** (on the tower): sees the whole map every tick (14 binary channels per
cell: wall, home, apple, berry, mushroom, ripe, rotten, lion, tiger, asleep,
caged, guarding, hunting, forager), the forager's load, the time, and its own
hunger (stages 2-3).  It cannot move or eat.  It speaks at ticks 0, 5, 10, ...:
an utterance of up to 8 symbols from 16, ending with *stop* (stop first =
silence).

**Forager** (on the ground): sees the 3 x 3 cells around itself (same channels;
it can always see and smell ripeness up close), its position, its own hunger,
what it carries, the time, and everything it has heard.  Each tick: up, down,
left, right, stay (third revision: stepping on food eats it when hungry, or
picks it up for a hungry scout).  In stage 3 it also speaks, at ticks 2, 7,
12, ..., with the same 16 symbols.

Only integer symbols pass between the agents.  No symbol is assigned any
meaning.

## Stages

| stage | who is hungry (privately) | who speaks | ripeness visible to scout | episode length |
|---|---|---|---|---|
| 1 | forager | scout | yes | 40 |
| 2 | forager **or** scout (50/50 per episode) | scout | yes | 60 |
| 3 | forager **and** scout | both | **no** (only the forager can tell, up close) | 60 |

Hungers are drawn from the food types present on the map.  The scout eats when
the forager reaches home carrying food (pick up, then walk home).

## Rewards (shared by the team)

| event | reward |
|---|---|
| a hungry agent eats ripe food of the wanted type | +1 |
| ... ripe food of another type | +0.2 |
| ... rotten food | -0.5 |
| forager caught by a lion | -1, episode ends |
| forager caught by a tiger | -2, episode ends |
| a hungry agent that has not eaten when the episode ends (including when caught) | -1 (added after the second pilot) |
| each tick before the episode ends | -0.01 |
| each symbol spoken (either agent) | -0.005, from 40% of training on |

An agent eats once per episode.  The episode ends when every hungry agent has
eaten, when the forager is caught, or at the time limit.

## Learning (`agents.py`, `train.py`)

Recurrent PPO, 256 episodes per update, whole episodes replayed for each
update.  Scout: map encoder (two 3 x 3 convolutions, 32 channels, then 128
units), a GRU memory updated every tick, a GRU word decoder sampling one
symbol at a time, a value head.  Forager: message encoder (symbol embeddings
and a GRU), view encoder, GRU memory updated every tick, action head, value
head, and in stage 3 a word decoder.  Each agent is updated only from the team
reward through its own PPO loss; no gradient crosses between them; no
auxiliary losses.  Value heads (used only in training) also see both hungers
(centralised critic).  Learning aids, fixed in advance: for the first 20% of
training only one glade has anything in it; the symbol cost starts at 40%.

Silent controls: identical runs in which every utterance is replaced by
silence (`--silent`).

## Pass lines, fixed before any run

Each stage: 3 seeds; a stage passes if at least 2 seeds pass.  "Return" is mean
team reward per episode on fresh maps, with policies sampling as in training.

**Stage 1**
1. Return at least 0.2 above the silent control of the same seed, and blocking
   or shuffling the scout's messages removes at least half of that gain.
2. The scout's first utterance changes for at least 70% of maps when a sleeping
   lion is replaced by a sleeping tiger (which matters) and for at most 30%
   when a caged lion is replaced by a caged tiger (which does not).
3. In maps with one occupied glade, topographic similarity >= 0.3, and for each
   factor (glade, food, ripeness, predator, predator state) a symbol whose
   presence separates at least one value from the rest by >= 0.5.

**Stage 2**
1. Return at least 0.2 above the silent control, in both kinds of episode.
2. A "my hunger" marker: a symbol present in >= 90% of the scout's first
   utterances when it is hungry and <= 10% when the forager is.
3. Reuse: on the same map, the symbols of the two kinds of utterance (marker
   removed) overlap with Jaccard >= 0.5.

**Stage 3**
1. Return at least 0.2 above the silent control, and silencing only the
   forager costs at least 0.1 (its speech is used).
2. Shared vocabulary: for at least 2 of the 3 food types, the symbol most
   associated with "I want this food" is the same for scout and forager.

## Results (2,500 updates per run; `results/watchtower/summary.txt`, `summary.json`)

No stage passes.  Stage 2 seed 2 crashed silently at update 925 (process gone,
no error; log kept as `s2_2.crashed.log`) and was restarted from scratch with
the same seed; stage 2 fails on seeds 0 and 1 whatever it shows.

| run | return | blocked | shuffled | silent ctrl | verdict |
|---|---|---|---|---|---|
| s1_0 | +0.02 | -1.16 | -1.30 | -1.09 | useful PASS, lion/tiger fail, structure fail |
| s1_1 | +0.04 | -1.18 | -1.21 | -1.16 | useful PASS, lion/tiger fail, structure fail |
| s1_2 | -0.15 | -1.43 | -1.17 | -1.18 | useful PASS, lion/tiger fail, structure fail |
| s2_0 | -0.71 | -1.61 | -1.37 | -1.41 | useful PASS, marker fail, reuse fail |
| s2_1 | -0.02 | -1.59 | -1.27 | -1.52 | useful PASS, marker fail, reuse PASS |
| s2_2 (restarted) | -0.73 | -1.70 | -1.44 | -1.49 | useful PASS, marker fail, reuse PASS |
| s3_0 | -2.60 | -2.60 | -2.60 | -2.60 | fail (both agents stay home and starve) |
| s3_1 | -1.99 | -2.48 | -2.54 | -2.60 | fail (forager never speaks) |
| s3_2 | -2.60 | -2.60 | -2.60 | -2.60 | fail (both agents stay home and starve) |

What the scouts say: essentially **which glade to go to**.  The glade is the
only factor with a separating symbol (gap 0.42-0.56 in every stage-1 run);
food type (gap <= 0.04), ripeness (<= 0.19) and predator type (<= 0.18) have
none.  Seed 2 uses only 6 distinct utterances.  Replacing a sleeping lion by a
sleeping tiger changes the first utterance in 46-63% of maps (line: 70%);
caged lion -> caged tiger in 2-11% (line: at most 30%, met).  Foragers still eat
rotten food in about 20% of episodes.

Stage 2: the scout's first utterance is identical in 89-99% of maps (seeds 0-2) whether
it or the forager is hungry; there is no hunger marker.  None is needed: the
forager knows its own hunger, and the world makes it eat or carry
automatically, so "go to glade X" serves both cases (returns -0.02 in both
kinds of episode for seed 1).  This is a design flaw in the test, not only a
failure of the agents.

Stage 3: four of six pairs, including all silent controls, stay home and starve
(-2.60).  In seed 1 the scout points to glades but the forager is silent
(mean length 0), so silencing it costs nothing.
