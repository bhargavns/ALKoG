# Grammar from environmental pressure: event world, pressure switches, elicitation tests

Branch `experiment-grammar-pressures`, started from `experiment-spontaneous-grammar` (4ae2eeb).
Nothing in the earlier packages is changed.

**Status: the target claim is not supported yet.** Reward-driven agents learn to
communicate about who-does-what-to-whom events and generalise to held-out
events, and time pressure reliably makes them brief. None of the pressure
combinations tried here produced words, word order or morphology by the
elicitation tests below. One of three seeds in the strongest-urgency condition
shows a partial lexicon and is the most promising lead. Details and the next
steps are at the end.

## Why a new world

The two earlier studies ended with codes that work but are not analysable:
the fixed-slot protocol was compositional because the roles were supplied, and
the unassigned-slot protocol generalised (98.8% on binding) while failing local
edits (23%) and new cardinality (55%). In both, structure came from the
architecture or from an auxiliary loss (reconstruction, inventory invariance),
not from anything the world demanded.

This package removes every such term and makes the *world* the experimental
variable. It also changes what there is to talk about, so that grammar has
something to do:

* A scene is an event. Intransitive `PRED(S)` or transitive `PRED(A, P)`, where
  each participant is a (type, colour) entity. 6 types x 4 colours, 4 + 4
  predicates, 2,304 events.
* Transitive events are fully reversible (`guards(lion, tiger)` and
  `guards(tiger, lion)` are both valid and both occur), so roles can never be
  read off the entities. This is what a role marker or word order would be for.
* One or two participants, so no fixed-length packet is forced.
* S, A and P are three distinct perceptual roles. Whether the language treats S
  like A or like P is left to the agents (alignment is measurable).
* Held-out structure, all primitives seen in training: 4 entity descriptions
  never seen (`novel_entity`), agent/patient colour pairings never seen
  (`novel_pair`), 2 entities seen only in intransitive events and tested in
  transitive ones (`construction`), 2 entities never seen as patient and 2 never
  as agent, tested in the missing role (`novel_role`). 840 training events.

The task is the same reference game as before: an observer knows the rewarded
room, a collector sees the public rooms and picks one. Rooms are single-edit
neighbours of each other (one feature changed, another predicate, or the roles
reversed). Every room of a set is used once as the target, so a collector that
ignores the message scores exactly 1/rooms.

## What is and is not supplied

Supplied: a symbolic percept (unordered element vectors with role as a
perceptual feature), generic GRU speaker and listener, a 6-symbol inventory
plus end-of-utterance, the task reward.

Not supplied: word slots, semantic heads, reconstruction or invariance losses,
role or position labels, anchored perception, cross-agent gradients. The
speaker learns by REINFORCE on the binary team outcome minus time cost. The
listener learns from the revealed outcome of each episode (which room held the
reward), keeping a guess after every symbol.

## The pressures (`train.py`)

| switch | what it is in the world |
|---|---|
| `--cost` | every symbol takes a tick, ticks are charged to the team |
| `--hazard` | after every symbol the danger arrives with probability h and the collector must act on what it has heard so far |
| `--blind` / default | whether the speaker sees the public rooms (shared context) |
| `--focus`, `--curriculum` | how many respects the rooms differ in; simple contexts first |
| `--cands` | how many confusable rooms |
| `--vocab` | size of the signal inventory |
| `--generations`, `--bottleneck` | population turnover: newcomers overhear a limited sample of the old speaker, then learn by doing |

## The tests (`analysis.py`)

All behavioural, on the frozen pair, in constructed contexts:

* **Lexicon.** Rooms that differ from the target in one respect only. If the
  same short form recurs for a value whatever the rest of the event is, it is a
  word. `word_consistency` is the share of the modal form.
* **Word classes.** Is the form for `lion` the same as S, A and P?
* **Combination.** Rooms differing in two respects: is the utterance the two
  words side by side (`concatenative`), in a consistent order?
* **Productivity.** Lexicon and order rules induced from training events only,
  utterances synthesised for held-out events and given to the listener.
* **Brevity.** Does length track how much the context demands?
* **Roles.** With the role-reversed twin in the room: accuracy, and whether
  reversing the string moves the listener to the twin.
* **Entanglement** (blind speakers): how much of each symbol position is
  explained by its best single slot (`position_purity`).
* **Learnability.** A naive listener learning from the frozen speaker.

## Results: 5 conditions x 3 seeds, 6,000 updates each, every run kept

`results/grammar_pressures/sweep/`. Mean (min-max) over seeds. Chance is 0.06
for the 16-room blind conditions and 0.25 for the 4-room context conditions.

| condition | train | held-out | utterance length | word consistency | concatenative | role twin in room |
|---|---|---|---|---|---|---|
| blind, calm (16 rooms) | 0.82 (0.76-0.85) | 0.76 (0.72-0.79) | 10.0 of 10 | 0.07 | 0.00 | 0.81 |
| blind, urgent (cost .003, hazard .03) | 0.49 (0.35-0.74) | 0.46 (0.36-0.66) | 8.5 (6.9-10) | 0.32 | 0.00 | 0.54 |
| context, calm (4 rooms) | 0.92 (0.91-0.95) | 0.89 (0.86-0.93) | 6.0 of 6 | 0.14 | 0.00 | 0.85 |
| context, urgent (cost .01, hazard .03) | 0.94 (0.92-0.96) | 0.93 (0.90-0.96) | 3.2 (2.8-3.5) | 0.11 | 0.00 | 0.87 |
| context, emergency (cost .01, hazard .3, curriculum) | 0.66 (0.52-0.92) | 0.66 (0.53-0.91) | 1.0 (0.4-2.0) | 0.75 | 0.29 | 0.43 |

Held-out by kind for context/urgent: novel entity 0.88, novel pair 0.96,
construction 0.93, novel role 0.94. Shuffling messages between episodes returns
every condition to chance.

What these support:

1. **Useful event communication from reward alone.** With no cross-agent
   gradient and no auxiliary loss, pairs reach 92-94% on training events and
   89-93% on held-out events, including entities in roles and constructions
   they never occupied in training, and 85-87% when the role-reversed twin is
   among the rooms.
2. **Urgency produces brevity.** A small time cost and interruption risk halve
   utterance length (6.0 to 3.2 symbols) with no loss of accuracy, in all three
   seeds.
3. **Brevity is not grammar.** The short codes are holistic. No value has a
   stable form (word consistency 0.11), two-contrast utterances are never the
   two one-contrast forms combined, length does not track what the context
   demands (3.20 vs 3.28 symbols for one vs three contrasts), and reversing the
   string moves the listener to the role-reversed room in only 18-25% of cases.
   Blind speakers' positions each explain 4-16% of their entropy by any one
   slot.
4. **Too much urgency destroys the code.** At hazard 0.3 two of three seeds
   collapse to three signals (silence and two symbols) at about 50%. Their
   word consistency of 0.9 is an artefact of having three signals, not a
   lexicon.

The lead worth following: `context_emergency_2` did not collapse. It reaches
92%, its utterances lengthen with context complexity (1.7, 1.9, 2.1 symbols for
one, two, three contrasts), and about half of its one-contrast forms are stable
single symbols (`lion` = 4, `tiger` = 2, `blue` = 5, `loose` = 2). It does not
combine them (concatenative 0.06). One seed is an observation, not a result.

## Development log (single-seed pilots, earlier code states, not in the sweep)

Kept for the record; do not cite as results.

* A straight-through Gumbel learner with these recurrent agents reached 41%
  without pressure and 8% with it. The sampled-reward learner was far better,
  matching the earlier study (sampled PPO 93-95% vs recurrent Gumbel 49-66%).
* An earlier blind/urgent run reached 93% with 5.1-symbol utterances, close to
  the 5-symbol minimum for a 6-symbol inventory, with near-maximal entropy per
  position and under 0.26 bits shared with any slot: an efficient hash. The
  final code did not reproduce this in three seeds (table row 2).
* With only four rooms and a blind speaker, urgency cut utterances to 2.4
  symbols at 89%: the task needs two bits and the code delivers a short hash.
* Generational turnover (10 generations, bottleneck 120 or 250 events,
  starting from a working founder pair) lost the language within four
  generations in both runs. The code path is kept (`--generations`).
* In pure one-contrast contexts with mild cost, the speaker reached 100% with
  2.8-symbol forms and was still shortening slowly at 6,000 updates.

## Why it has not worked, and what follows from that

* **The reference game rewards hashes.** When the rooms are near neighbours in
  meaning and success is all-or-nothing, the best code under noise or time
  pressure puts neighbouring meanings far apart in form. That is the opposite
  of compositional. The earlier studies shared this property.
* **Brevity pressure selects for efficiency, not reuse.** The shortest code
  for a fixed set of events is a dense one. Reusable parts only pay when what
  must be said varies from moment to moment and the speaker exploits it. The
  context-seeing speaker did not: its utterances barely change with context.
* **The learner gets stuck.** REINFORCE on a binary outcome settles into
  pooling equilibria (the three-signal collapse) and trims length very slowly.
  The conditions under which words would be optimal are exactly the ones where
  it collapsed in two of three seeds.
* **Turnover needs an expressivity counterweight** that 400 reward updates per
  generation did not provide.

## Next experiments, in the order I would run them

1. **Follow the emergency lead properly.** 10 seeds, 20,000 updates, hazard
   swept over 0.1-0.3, entropy held up longer. Success criterion fixed now:
   word consistency >= 0.8 with >= 15 distinct forms, in >= 7 of 10 seeds.
2. **Graded payoff.** Reward the collector by how much of the chosen room
   matches the target instead of all-or-nothing, with rooms drawn from near and
   far. With symbol noise this makes form-meaning smoothness the optimum.
3. **Open-ended world.** 12 types x 8 colours and three-participant events, so
   most events an agent meets are new. Memorising whole-event forms stops paying.
4. **Role alternation with one network per agent** (the proposal's 8.3.4),
   then a population of four, then turnover with an imitation budget large
   enough to keep expressivity.
5. **A better sampled learner.** PPO with a learned value baseline worked in
   the earlier study and should reduce the collapse rate.

Only after one of these passes the lexicon and combination tests is the
MuJoCo step worth running. Its design is fixed here so it can be executed
without new decisions: reuse `experiments/grammar_arena` and the five frozen
navigators; place the event's participants in the arena (the predicate decides
which gate the agent animal occupies and whether the patient food is
reachable); the collector's room choice becomes a food-and-route choice scored
by the existing food bonus and hazard penalty; ticks become physical time, so
`--cost` and `--hazard` are no longer abstractions: a loose animal that reaches
the collector while it is still listening is the interruption. Train the
communication game against that decision interface with motors frozen, hold
out the same four partitions, compare normal, blocked and shuffled, and run the
elicitation tests on the frozen pair. None of this has been run.

## If it works: where it goes

* **Gloss it.** The elicitation output is already a paradigm table. Feed the
  utterances and their contexts to CWoMP / GlossAssist as an unknown language
  and score recovered morphemes against the simulator. This turns ALKoG into a
  generator of low-resource documentation problems with known answers.
* **Q4 as a typology experiment.** Train newcomers on languages from different
  pressure conditions and compare interactions to criterion. The claim that
  structured systems are easier to acquire becomes testable within one world.
* **Word order against case.** Add symbol loss in transit. Order-based role
  marking should be fragile to deletion and marker-based marking robust,
  which is the trade-off Lian, Bisazza and Verhoef (2023) had to seed with a
  taught language.
* **Alignment.** S, A and P are separate percepts here. Vary how often
  intransitive subjects are agent-like and see whether the language goes
  accusative or ergative.
* **Abbreviation.** Make event frequencies Zipfian and test whether frequent
  meanings get shorter and less regular forms.
* **Knowledge graph ablation.** Replace the symbolic percept with the ALKoG
  graph and with a frozen or random graph, which is the comparison Q3 of the
  proposal asks for.

## Run

From `troy_dev_ppo_communication_gumbel/`:

```powershell
python -m pytest tests/experiments/test_grammar_pressures.py -q -p no:cacheprovider
python -m experiments.grammar_pressures.sweep --output results/grammar_pressures/sweep10 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 6
python -m experiments.grammar_pressures.train --output runs/lead --cost 0.01 --hazard 0.2 --curriculum --updates 20000
python -m experiments.grammar_pressures.analysis runs/lead
```

A run is 3-6 minutes on one CPU core; the models are too small to benefit from
the GPU. The sweep in `results/` was run on a 2-core Linux sandbox with torch
2.15, not on the RTX machine, in 85 minutes.

## Sources

Galke, Ram and Raviv (2022), Emergent communication for understanding human language evolution: what's missing? https://arxiv.org/abs/2204.10590
Li and Bowling (2019), Ease-of-teaching and language structure from emergent communication. https://arxiv.org/abs/1906.02403
Lian, Bisazza and Verhoef (2023), Communication drives the emergence of language universals in neural agents. https://arxiv.org/abs/2301.13083

## Follow-up runs (prepared, launch with `scripts/run_grammar_followups.bat`)

Two questions, with the pass line fixed before any of these runs exist.

1. **Was `context_emergency_2` luck?** `lead_h10`, `lead_h20`, `lead_h30`: the
   emergency condition at interruption risk 0.1, 0.2, 0.3, exploration kept on
   for 90% of training, 10 seeds, 20,000 updates. Pass: in at least 7 of 10
   seeds of some risk level, word consistency >= 0.8 with at least 15 distinct
   one-contrast forms and training accuracy >= 0.85.
2. **Does a world with near and wild misses favour parts?** Eight rooms, three
   of them unrelated events. `mixed_graded_noise` pays by how much of the chosen
   room matches the target and mishears 10% of symbols; `mixed_plain`,
   `mixed_graded` and `mixed_noise` are its controls. 5 seeds, 12,000 updates.
   Pass: `concatenative` >= 0.5 and `synthesis_understood` >= 0.8 on held-out
   events, and clearly above all three controls.

Results land in `results/grammar_pressures/lead` and `.../mixed`
(`summary.md` in each). These are CPU jobs; six run at once.

## Second round (2026-11-17)

Results of the two follow-ups: neither passed.  `lead_*`: 1 of 30 runs met the
lexicon line (the original lead was luck).  `mixed_*`: concatenation 0.00 in
all 20 runs; noise lengthened messages to the maximum (repetition, not parts).

Searched since, 3 seeds each (`report.py` and `structure.py` give per-run
tables; `structure.py --lineage RUN` follows a run across generations):

| folder | what changed | outcome |
|---|---|---|
| `inventory` | 12 / 24 / 40 signals instead of 6 | words did not get more consistent; concatenation fell to 0.02 |
| `pressures24` | speaker sees other rooms of the same kind (`--displaced`), collectors replaced every 1,000 updates, community of 3, 8 rooms | displaced and 8 rooms: accurate holistic codes, consistency 0.06-0.11. Turnover: consistency 0.71 at lower accuracy |
| `keyhole` | rooms are glimpsed: only what differs between them is seen (`--keyhole`) | most consistent words so far (0.80, predicate words 0.97); with `--displaced`, hand-built two-word messages understood 0.79; speaker still does not combine |
| `kgrid` | variations on keyhole + displaced, 20,000 updates | without urgency messages lengthen to the maximum; graded structure rises (topsim ~0.3) but no concatenation |

Fixed along the way: a newcomer speaker in the generational runs used to
imitate its predecessor without the rooms in view, though it spoke with them
in view, and the final checkpoint saved the founder.  Both affected the
earlier turnover pilots.

### Prepared next, pass lines fixed before the runs

7. **Iterated learning** (`il_*`): founder trained 6,000 updates, then 10
   generations of 1,500; each new speaker first imitates 200 (tight) or 3,000
   (loose) of the old speaker's utterances, in context.  Pass: in the tight
   condition, the last generation's `part2` or `concatenative` >= 0.3 in at
   least 2 of 3 seeds at training accuracy >= 0.85, and above the loose
   condition.
   **Result: fail.**  Concatenation 0.00 in all 12 runs, `part2` <= 0.13 in
   every generation.  Languages now survive all 10 generations at stable
   accuracy (0.86 glimpsed rooms, 0.93 urgent), so the earlier collapse was the
   imitation bug.  The tight bottleneck lowered word consistency (0.15 vs 0.62
   loose, glimpsed rooms) instead of regularising; topsim drifted up in one
   seed only (0.19 to 0.29-0.38).
8. **Sound before meaning** (`bab_*`): 3,000 updates of babbling first.  The
   speaker produces sequences from a random inner mood that refers to nothing;
   a song-listener learns to predict them; the speaker is paid for babbles that
   are easy to predict yet unlike the rest of the batch (diverse but rule-
   governed song).  Then the task as usual.  Pass: `part2` or `concatenative`
   >= 0.3 in at least 2 of 3 seeds at training accuracy >= 0.85, and above its
   control without babbling (`k_h03`, `inv24`).
   **Result: fail.**  Concatenation <= 0.07 and `part2` <= 0.15 in all 9 runs;
   word consistency equal to the controls (0.49 vs 0.50 for `inv24`, 0.51 vs
   0.42 for `k_h03`).  The song habits were not recruited for meaning.
9. **Wanting versus reporting** (`persp*`, `perspective.py`): the speaker has
   an inner state.  It has either seen an event (the collector should look at
   that room) or wants something with a property (the collector should fetch
   the one room that has it).  The collector chooses room and deed; the team is
   paid only if both are right.  Four properties (blue, tiger, 'hides', 'gone')
   are never wanted in training, only seen.  Pass, per run: a want marker in
   >= 90% of wanting utterances and <= 10% of reports; at least 80% of
   properties have their own non-empty report word; the property word is the
   same in both moods for >= 70% of properties; and a message built from the
   marker plus the report word for a never-wanted property is understood
   (room and deed) >= 70% of the time.

   **Result: fail on the pass line, with one clear partial success.**
   All 6 runs (vocab 12 and 24) win 1.00 on training.  The *mood* is fully
   productive: for properties never wanted in training, the collector picks the
   right deed (fetch) 100% of the time in 6/6 runs.  This is not a bias: on
   random strings it says fetch only 6-63%.  And it follows the speaker's felt
   state, not the percept: give the speaker a wish percept but switch its inner
   mood to 'report' and the collector chooses look 99-100% of the time.  One
   run (`persp12_0`) has a clean prefix marker: symbol 0, first, in 100% of
   wanting and 0% of reporting utterances ('tiger' seen = `7`, wanted = `0 7`).
   The *content* does not transfer: the room is right for only 0-6% of never-
   wanted properties, and the word for a property differs between moods
   (same in both <= 0.33).  So the speaker's state is marked separably and
   generalises; what it wants is not expressed with the words for what it sees.

This is the first step toward speakers talking about their own states: a
marker for the speaker's mood that is separable from the words for the world,
and that extends to wants it has never expressed.

## Third round: a world with consequences (`consequences.py`)

Until now nothing in the event world mattered: a lion was a one-hot column,
and the only job of a message was to pick out a room.  Here events have
consequences and the collector has needs.

* **Consequences.** Lions and tigers are predators; the other four types are
  food.  A room is *dangerous* if a predator is free: `loose(lion)`, or a
  predator agent of `guards`, `chases` or `near`, or a predator patient of
  `near`.  A predator that is present but not free (`caged`, a hiding agent,
  or on the receiving end of `guards`/`chases`/`hides`) is *tame*; `ripe` and
  `gone` predators count as nothing.
  Food is *available* if it is `ripe` (intransitive) or a participant of a
  transitive event that does not guard or hide it.  Roles now change outcomes:
  `chases(lion, apple)` is dangerous, `chases(apple, lion)` is not.  Colour
  never changes an outcome.
* **Needs.** The collector is privately *hungry* (+1 for entering a room with
  available food) or a *keeper* (+1 for a tame predator).  Either way a
  dangerous room costs -1, staying home is 0, and each symbol costs 0.01.
* **What each agent knows.** The speaker has been in all four rooms and sees
  them fully, as an unordered set, so it cannot point by position.  It does
  not know the collector's need.  The collector knows its need and sees each
  room through fog: who is there (type, colour, role) but not what is
  happening (the predicate is hidden).  So the message must carry the hidden
  states, tied to visible identities, for whichever need the collector has.
* **Learning.** Speaker: REINFORCE on the payoff of the collector's choice.
  Collector: after each episode every room's payoff for its need is revealed
  and it learns to choose the best option.  No auxiliary losses.
* **Conditions** (3 seeds, 12 symbols, up to 8 per message, 12,000 updates):
  `cons2` (need private, two needs), `cons1` (always hungry: the speaker can
  in effect just name the food room), `cons2_silent` (speaker says nothing; the
  best the collector can do from what it sees).

Tests (`consequences.py --analyse`):

* **Payoff** normal / blocked / shuffled message, against the silent control
  and a full-information oracle, per need.
* **Consequence sensitivity.** Change one room's predicate so that its
  outcome changes for some need (`caged` to `loose` lion) or so that it
  changes for no need (`gone` to `caged` apple).  Does the message change?
  A language about consequences changes for the first and not the second.
* **Status markers.** Scenes with one eventful room and three uneventful ones:
  is there a symbol for *danger*, *food* and *tame* that is present whatever
  the entity, and absent otherwise?

**Pass line, fixed before any run, per `cons2` seed; pass if 2 of 3 seeds:**
(a) mean payoff at least 0.3 above the silent control; (b) the message changes
for >= 80% of consequential edits and <= 30% of inconsequential ones; (c) for
each of danger, food and tame, a symbol present in >= 80% of that status's
utterances for every entity type and in <= 20% of the others.
Prediction: `cons1` passes (a) and fails (c).

```
python -m experiments.grammar_pressures.sweep --output results/grammar_pressures/iterated --seeds 0 1 2 --updates 6000 --workers 12 --conditions il_k_tight il_k_loose il_l_tight il_l_loose
python -m experiments.grammar_pressures.sweep --output results/grammar_pressures/babble --seeds 0 1 2 --updates 12000 --workers 12 --conditions bab_l bab_l_nov
python -m experiments.grammar_pressures.sweep --output results/grammar_pressures/babble --seeds 0 1 2 --updates 20000 --workers 12 --conditions bab_k
python -m experiments.grammar_pressures.sweep --output results/grammar_pressures/perspective --seeds 0 1 2 --updates 12000 --workers 12 --conditions persp24 persp12
python -m experiments.grammar_pressures.perspective --summarise results/grammar_pressures/perspective
```
