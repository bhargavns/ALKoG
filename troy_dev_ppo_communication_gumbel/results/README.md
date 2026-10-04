# Saved Gumbel arena results

The compact [evaluation summary](arena_summary.json) preserves configurations,
held-out metrics, evaluation-file hashes, and final checkpoint hashes from the
completed separate-start experiments. All use 16-symbol Gumbel communication,
current-message receivers, oracle inputs, shared yaw, and training seed 0.

| Iterations | Scheduled sender entropy: food success | Zero bonus: food success |
|---|---:|---:|
| 800 | 81.6% | 67.6% |
| 3200 | 98.2% | 84.4% |

These are hard-sender evaluations on 500 scenes per condition with sampled
receiver actions. At 3200 iterations, loose-lion death rates were 3.6% and 10.4%
on 250 uncaged scenes. The JSON also includes sampled, constant, and random
message interventions and uncertainty estimates.

These are saved experimental observations, not reruns performed during cleanup.
One training seed does not establish robustness, and oracle-input results do not
establish performance with visual perception. Raw Gumbel checkpoints, metrics,
and evaluations remain local under ignored `output/experiments/`. They are not
required to read this summary or run the source. Copied pre-cleanup implementation
snapshots were removed; original run manifests describe their original execution.
