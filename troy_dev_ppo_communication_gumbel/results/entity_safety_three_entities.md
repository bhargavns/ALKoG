# Experiment 1 follow-up: lion, tiger, sheep

Only lion, tiger and sheep are used in the current task. Earlier five-entity
results and source snapshots are retained, but are not directly comparable to
this follow-up. The embodied navigation baseline is preserved.

The diagnostic has three candidates and six joint actions. There are six balanced
entity/cage cases but five required meanings: lion safe/dangerous, tiger
safe/dangerous, sheep safe. Six public tokens can solve the one-token task;
two-token messages have 36 possibilities. No position or token has an assigned
semantic role in learned-code conditions. Safety is explicit private oracle
evidence. No visual model, graph-memory model, trajectory, or safety inference
is trained.

## Fixed protocol and implementation

Four conditions, five seeds each (0–4), 1,000 updates over batches of 60 generated
contexts, 60,000 contexts per pair. Frozen final-checkpoint evaluation uses 600
balanced episodes, 100 independent public layouts, seed 2147483645. Development
used seed 123 and evaluation seed 1001. No checkpoint selection or final-set tuning.
Defaults were fixed after the development sanity check and before these runs.

- Frozen code: seed 734 selects five distinct arbitrary whole utterances. Sheep
  shares a code across cage states. Only the receiver learns. This is one fixed
  code sanity check, not an emergent or compositional condition.
- Exact reward: enumerate discrete utterances and compute the expected shared
  0/1 reward, summing action probabilities. Differentiate the exact expectation.
  No critic loss, PPO, Gumbel surrogate, or token regularizer. Both agents learn.
- Gumbel/PPO: earlier approximate learner, rerun on exactly the same simplified
  task/layout budgets. Four PPO epochs; the exact methods have one update per
  batch. Hyperparameters are in each saved config.

Exact methods use the complete reward function for all utterances/actions;
Gumbel/PPO uses sampled action rewards. Equal context counts do not match reward
queries, gradient steps, or compute. This comparison diagnoses optimization and
is not a sample-efficiency comparison. Two-token networks also have more
parameters than one-token networks.

## Results

Mean ± sample standard deviation across five trained pairs. Greedy chooses both
sender tokens and receiver action by argmax; sampled draws both. Unsafe APPROACH
is conditional on a dangerous target, regardless of selected candidate identity.

| Condition | Tokens | Greedy joint accuracy | Sampled joint accuracy | Unsafe APPROACH, greedy |
| --- | ---: | ---: | ---: | ---: |
| Fixed arbitrary code | 2 | 100.0% ± 0.0% | 100.0% ± 0.0% | 0.0% ± 0.0% |
| Learned code, exact reward | 1 | 70.0% ± 13.9% | 70.1% ± 14.2% | 0.0% ± 0.0% |
| Learned code, exact reward | 2 | 86.7% ± 18.3% | 87.8% ± 16.7% | 0.0% ± 0.0% |
| Learned code, Gumbel/PPO | 2 | 76.7% ± 14.9% | 76.5% ± 14.5% | 50.0% ± 50.0% |

The fixed code reaches 100% greedy accuracy in all five seeds. The receiver and
task can therefore support correct behavior for this code without changing the
architecture. This does not prove that every code is equally easy to learn.

Learned two-token exact-reward runs reach 100%, 100%, 66.7%, 100%, 66.7%. In the
two unsuccessful seeds, lion and tiger receive identical messages within each
safety state. Those messages cannot identify which predator matters. The sender
retains safety information but collapses animal identity. This failure persists
without the Gumbel approximation, so it cannot be attributed solely to that
approximation. Three perfect seeds do not establish reliable convergence across
seeds or compositionality. The proposed every-seed accuracy gate is not met.

The one-token exact-reward condition has enough representational capacity, but
also remains below its 100% ceiling. Gumbel/PPO has large safety errors and some
learning-curve deterioration; full comparisons are in the linked metrics.
Per-meaning results and utterance maps are retained for every seed. A low target-
unsafe APPROACH rate does not compensate for identifying the wrong entity.

Next work should remain in Experiment 1: keep this three-entity task fixed and
address early loss of distinctions between meanings in learned codes. A small
controlled exploration or update-schedule change is more informative than
expanding the environment, adding graph structure, or starting Experiment 2.
Neither high reward nor distinct whole utterances proves component reuse;
transfer/intervention tests are still required later.

## Running and artifacts

[Protocol and commands](../experiments/entity_safety/README.md) reproduce the
conditions. [Portable metrics and utterance maps](entity_safety_three_entities_metrics.json)
include all controls, per-seed/per-case accuracy, safety rates, and computation.
Full configs, source hashes/snapshots, checkpoint/RNG state, learning curves,
training records and evaluation scene/message/action/outcome logs are in:

- `output/entity_safety3_fixed/`
- `output/entity_safety3_exact/`
- `output/entity_safety3_gumbel/`

All 20 artifact sets have 1,000 rollout batches, 1,000 curve records and 3,000
evaluation records each. Earlier five-entity artifacts remain untouched.

Training ran locally with PyTorch on the Mac's CPU, one worker thread, with
small networks (a few thousand parameters). Each generated episode contains one
utterance and one response; batch updates do not imply temporal rollouts.
CPU training times were 2.5–8.1 seconds per pair.

A GPU is optional acceleration; it is not required for learning across time
steps. Larger perception models and many embodied rollouts would change the
compute requirements. Exact enumeration will become impractical as utterance
length/vocabulary grow, so this baseline is a diagnostic for the small task.

Validation: 16 diagnostic tests and 46 existing communication tests pass (62
total). One existing video test remains excluded because OpenCV is missing in
the environment. New checks verify the exact expected reward and its sender
gradient against an explicit sum over discrete messages and actions, frozen
whole-code behavior, and exact-run logs. Static import/name checks and whitespace
checks pass. Experiment 2 and later experiments remain unimplemented.
