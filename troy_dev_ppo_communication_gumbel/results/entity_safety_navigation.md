# First MuJoCo transfer after functional Experiment 1 acceptance

[Functional safety-response acceptance](entity_safety_functional_acceptance.md)
passed before simulation: all five frozen exact two-token checkpoints have zero
response errors in 72,000 fresh trials, plus conservative coordinate-domain
bounds for every target case. The transition is tied to checkpoint hashes. No
communication weights were updated during simulator training.

The accepted task is correct safety response for a target. It does not require
recovering lion versus tiger when their safety is identical. The original joint
entity/response scores remain available and are not relabeled as perfect.
Deterministic argmax deployment is required; stochastic, visual and physical
safety are outside that classification acceptance.

## Simulator setup

Three symbolic animal markers (lion, tiger, sheep) and a moving receiver are
represented in a MuJoCo scene. A public goal marker specifies the relevant
location. The sender privately receives the target's oracle cage/safety facts,
sends one complete utterance, and the frozen decoder supplies APPROACH or WARN.
WARN ends the episode. APPROACH enables a learned movement actor/critic.

The movement policy sees the public goal's offset from its current position,
with no private safety/cage/category label. It learns four directional actions
using PPO, discounted rewards and generalized advantage estimation across real,
ordered MuJoCo scene transitions. Movement is trained on safe-goal episodes;
the already learned communication module handles the safety decision.

Movement is kinematic, as in the original troy_dev arena: position updates and
`mj_forward`. This is not torque-driven locomotion. Cages are oracle flags, not
rendered/perceived containment; animal markers are not realistic visual models.
Layouts start with a clear straight corridor. This is a first oracle control
transfer, not joint communication learning, vision or general obstacle planning.

Five seeds, 100 x 512 = 51,200 movement steps per seed (256,000 total), 4 PPO
epochs, learning rate 0.001. An initial 10-iteration infrastructure check used
separate evaluation seed 4000. The fixed-budget study uses evaluation seed 4001,
100 layouts per entity/cage case, 600 episodes per seed. No final checkpoint
selection or tuning on these evaluation outcomes.

## Results

| Seed | Correct warnings on dangerous targets | Reached safe goals | Hazard contacts | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| 0 | 200/200 | 237/400 (59.2%) | 0 | 163 |
| 1 | 200/200 | 390/400 (97.5%) | 1 | 9 |
| 2 | 200/200 | 199/400 (49.8%) | 0 | 201 |
| 3 | 200/200 | 400/400 (100.0%) | 0 | 0 |
| 4 | 200/200 | 205/400 (51.2%) | 2 | 193 |

Across 3,000 evaluation episodes, all 1,000 dangerous-target warnings are correct,
and there are zero target-safety response errors, including safe targets.
Safe-goal reach rate is 1431/2,000 = 71.55%. Overall success,
including correct warnings, is 2431/3,000 = 81.03%.
There are three hazard contacts with dangerous distractors and 566 timeouts.
These are movement failures despite a correct safety response for the marked
target. Aggregate success must not obscure those failures. Navigation is not yet
reliable; the frozen safety classification baseline retains its accepted status.

Next work belongs to simulator control: improve reliable safe-goal reaching and
avoidance while retaining the accepted communication weights. The current
movement observation omits distractor geometry, so safety guarantees for general
layouts would require additional public spatial observations or a suitable
controller. No physical safety guarantee follows from a correct target label.

## Artifacts and checks

[Simulator metrics](entity_safety_navigation_metrics.json) include per-seed
training times and counts. Full configuration/source snapshots, movement
checkpoints, learning curves, and all 3,000 scene/message/action/reward/position
trajectories are in `output/entity_safety3_navigation/`. All artifact counts and
trajectory action/reward/position lengths were checked. The runner checks that
frozen sender/receiver weights are unchanged and rejects failed acceptance or
modified accepted checkpoints before creating simulator scenes.

72 focused tests pass, including the acceptance interval counterexample check,
MuJoCo transitions/reward correctness, observation isolation, temporal terminal
masking and simulator gate rejection. One existing video test is excluded because
OpenCV is absent. Static import/name and whitespace checks pass.
