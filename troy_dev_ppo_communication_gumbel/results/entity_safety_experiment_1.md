# Experiment 1: initial implementation and five-seed run

Implementation base: `origin/troy_dev`, commit
`40ac429897d543094aa120c0fab8da93ae686de9`. Local implementation is on
`experiment-1-safety-and-navigation`; source hashes/snapshots capture the changes
used in this run. See the [protocol and commands](../experiments/entity_safety/README.md).

This is an initial fixed-configuration experiment, not evidence of stable
compositional communication. Two-token agents outperform one-token agents, and
performance depends on messages, but training is unstable and joint accuracy
remains well below reliable entity/safety communication. Establish a stronger,
reproducible Experiment 1 baseline before drawing transfer conclusions or
proceeding to graph, relational, newcomer, or visual experiments.

## Configuration

Five training seeds (0–4) per length, vocabulary six, unrestricted ordered message
positions, 1,000 updates × 500 one-decision episodes = 500,000 interactions per
pair, four PPO epochs, learning rate 0.001, Gumbel temperature 1.2, receiver action
entropy coefficient 0.01. No label/token/reuse supervision. The fixed safety rule
is explicit private oracle evidence. Public candidate identities/positions contain
no target or safety information. All entity/cage meanings are in training.

Frozen deterministic evaluation uses 1,000 balanced episodes per seed and control,
100 public layouts repeated across the ten entity/cage cases, layout seed
2147483646. Configurations were fixed before this comparison. No adaptation or
selection of the best checkpoint; all numbers use the final checkpoint. The
straight-through update is the documented biased surrogate, not exact categorical
message PPO. Parameter counts are different across lengths.

## Joint entity/response accuracy

Mean ± sample standard deviation across five independently trained pairs.

| Control | One token | Two tokens |
| --- | ---: | ---: |
| Normal | 20.0% ± 10.0% | 54.0% ± 8.9% |
| No message | 17.9% ± 4.4% | 13.7% ± 5.1% |
| Constant | 16.0% ± 5.5% | 12.0% ± 4.5% |
| Shuffled within matched layouts | 11.7% ± 3.5% | 6.5% ± 1.0% |

Normal-message entity accuracy is 28.0% ± 11.0% (one token) and 70.0% ± 7.1%
(two tokens). Response accuracy is 54.0% ± 8.9% and 72.0% ± 13.0%, respectively.
Unsafe APPROACH rate, conditional on a dangerous target, is 80.0% ± 44.7% versus
20.0% ± 20.9%. Entity/response mistakes remain substantial. Per-meaning results
retain loose sheep and caged predators; aggregate accuracy cannot replace those
checks. The shuffled control intentionally always moves messages to another
meaning, so its score need not equal chance. Token associations remain descriptive.

One-token per-seed joint accuracy: 20%, 10%, 10%, 30%, 30%.
Two-token per-seed joint accuracy: 60%, 60%, 40%, 60%, 50%.

## Artifacts and validation

[Portable aggregate metrics](entity_safety_experiment_1_metrics.json) include
all requested metrics overall and by entity/safety/cage case. Full local artifacts
are in `output/entity_safety_exp1/`: study config, source snapshot/diff/hashes,
per-pair manifests/checkpoints, 1,000 saved complete rollout batches and learning
curve records per pair, and 4,000 scene/message/action/outcome evaluation records
per pair. All ten artifact sets were checked for completeness. Message frequencies
stay per pair to avoid pooling arbitrary public token labels.

Sender/receiver parameter counts: 902/2,291 for one token and 1,292/3,315 for two.
CPU training takes about 15–20 seconds per pair on this host; evaluation and logging
costs are separate from the recorded training wall time.

All 12 new diagnostic tests pass. The focused communication suite has 58 passing
tests including the new tests; its one video test fails to import the environment's
missing `cv2`. Additional oracle/entrypoint checks yield 33 passes and eight
`cv2` import failures. On macOS those checks require `MUJOCO_GL=glfw`, rather than
the test fixture's default `egl`. No new diagnostic depends on OpenCV or rendering.
Static undefined-name/import checks and `git diff --check` pass.

Experiment 2 (withheld dangerous tiger, causal token tests, and capacity/holistic
controls) and Experiments 3–6 are not implemented in this change.
