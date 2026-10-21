# Experiment 1: functional safety-response acceptance

The functional criterion is the appropriate APPROACH/WARN response for the
sender's target. Target identification remains a separate diagnostic. This
acceptance does not claim that the original identity-sensitive composition
experiment has been solved, nor that a receiver may safely execute an arbitrary
selected animal index. Simulator transfer uses an explicitly public goal marker.

No training weights or training reward were changed for this audit. All five
previous exact-reward two-token checkpoints were frozen and audited. They were
originally trained on joint identity/response reward; their safety response is
now evaluated separately, as required by the functional objective.

## Fresh exhaustive-state evaluation

Audit seed 3001 is distinct from training, development and earlier evaluation.
100 coordinate layouts x 6 candidate orderings x 8 cage-state assignments x 3
target choices = 14,400 trials per checkpoint, 72,000 trials total. Every
entity/cage case passes, including loose sheep and caged predators. Both unsafe
APPROACH and false WARN count as errors. Deterministic argmax tokens and responses
are the deployment contract.

| Training seed | Fresh trials | Safety-response errors | Unsafe APPROACH | Minimum conservative logit margin |
| --- | ---: | ---: | ---: | ---: |
| 0 | 14,400 | 0 | 0 | 8.04 |
| 1 | 14,400 | 0 | 0 | 7.93 |
| 2 | 14,400 | 0 | 0 | 5.42 |
| 3 | 14,400 | 0 | 0 | 12.75 |
| 4 | 14,400 | 0 | 0 | 6.40 |

Constant and zero-message controls are at or below the 66.7% majority-response
baseline. Thus perfect response performance requires information in the learned
messages. Identification scores and earlier joint-accuracy failures are retained;
merging lion/tiger when their safety is equal is allowed by this functional task.

## Continuous-position check

For each of the six target entity/cage cases, interval bounds propagate every
possible candidate coordinate in [-1,1] through the frozen Linear-Tanh-Linear
actor. At least one correct-response action score has a lower bound above the
upper bound of every wrong-response action score. All 30 seed/case checks pass
with large positive margins. Shared candidate scoring handles every permutation.
Distractor cage states and the private target cue never enter the receiver.

These are conservative floating-point bounds, not a machine-verified arithmetic
proof. The guarantee is limited to the specified architecture, three categories,
one candidate per category, oracle facts, bounded coordinates, and deterministic
deployment. Sampling softmax messages/actions retains nonzero error probability;
we do not claim perfect stochastic deployment, learned safety inference, visual
robustness, physical navigation, or compositionality.

## Acceptance and transition

The Experiment 1 functional gate passes for all five checkpoints: zero observed
errors, correct response bounds for all cases, and informative-message dependence.
The next stage is therefore a separate oracle MuJoCo transfer with a public goal
marker and frozen communication weights. Movement learning and simulator success
must be evaluated separately from this acceptance.

[Acceptance data](entity_safety_functional_acceptance.json) records checkpoint
hashes, cases, bounds, negative controls, and the acceptance decision. Full trial
inputs, output probabilities, messages and actions are in
`output/entity_safety3_acceptance/`.

Reproduce from the Gumbel project directory:

```sh
python -m experiments.entity_safety.audit --output output/entity_safety3_acceptance_NEW
```
