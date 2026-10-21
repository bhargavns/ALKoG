# Experiment 1: lion, tiger, sheep

The current diagnostic uses only lion, tiger and sheep. It is a single-decision
oracle task on CPU, separate from the preserved navigation baseline. Earlier
five-entity results remain in `results/entity_safety_experiment_1.md`, with their
original source snapshots in `output/entity_safety_exp1/`.

Every batch balances six entity/cage cases. Loose lion/tiger are dangerous;
caged lion/tiger and both sheep states are safe. Thus there are five required
entity/response meanings. Six public tokens suffice for a perfect one-token
code; two-token messages have 36 possibilities. Neither length prescribes token
meanings or entity/state positions.

The sender privately sees target category, cage flag and correct oracle safety.
The receiver sees only three candidate category indicators, random coordinates
and the entire utterance. Order/layout/target are randomized. A joint action
selects one candidate and APPROACH or WARN. Shared reward is 1 iff both choices
are correct, otherwise 0. No reward favors tokens, reuse, or a prescribed code.
No safety inference, graph memory, vision, navigation trajectory, or time-step
history is being trained in this diagnostic.

## Follow-up conditions

Run from this directory with Python 3.11+, torch and numpy installed:

```sh
python -m experiments.entity_safety.run --learner fixed --lengths 2 --output output/entity_safety3_fixed
python -m experiments.entity_safety.run --learner exact --lengths 1 2 --output output/entity_safety3_exact
python -m experiments.entity_safety.run --learner gumbel --lengths 2 --output output/entity_safety3_gumbel
```

Outputs must be new directories. Defaults: seeds 0–4, 1,000 batches × 60 contexts,
learning rate 0.001, hidden width 64, public embedding width 8, six tokens. Runs
are sequential, on CPU, with one torch worker thread. Each episode is a generated
scene, an utterance and a response, not a MuJoCo trajectory. A GPU is optional
acceleration, not a requirement for gradient updates or time-dependent training.

- **fixed:** Frozen arbitrary holistic two-token code for the five meanings,
  generated once with seed 734; only the receiver learns. Safe sheep has one
  utterance across cage states. This is a decoder sanity check, not emergence.
- **exact:** Both agents learn by maximizing mean expected shared 0/1 reward.
  Enumerate all 6 or 36 discrete utterances, weight each by the sender's product
  of position probabilities, and sum the receiver's correct-action probability.
  No Gumbel surrogate, PPO or critic loss is used. All receiver message inputs
  are discrete. There is one optimizer step per batch and no entropy bonus.
- **gumbel:** The earlier biased straight-through Gumbel/PPO method on the new
  three-entity task. Replay complete messages from saved source inputs and noise.
  Four PPO epochs, temperature 1.2, action-entropy coefficient 0.01. This is not
  exact marginal categorical-message PPO.

Exact learning accesses the full reward function for all messages/actions;
Gumbel/PPO uses sampled rewards. Equal generated-context budgets do not imply
matched reward-query budgets or computation. These are diagnostic learner
comparisons, not sample-efficiency claims. Parameter counts and wall time are
recorded separately. The receiver critic exists for architecture compatibility
but receives no loss in exact/fixed training.

## Evaluation and evidence

Each final checkpoint is frozen. Evaluate on 600 balanced episodes with 100
public layouts, each repeated for all six entity/cage cases. Default layout seed
2147483645 is separate from prior five-entity evaluation and development seed
1001. Development uses training seed 123; study runs use 0–4. Final evaluation is
not used for tuning or selecting a checkpoint.

Controls: deterministic normal messages/actions; zero message embeddings;
constant token-0 utterances; messages cyclically shuffled between different
cases within matched public layouts; and normally sampled messages/actions.
The zero-message ablation is out of distribution. Shuffling deliberately moves
every message to another case, so it need not produce chance performance.

Record joint/entity/response accuracy, utterance and position-token frequencies
by entity/safety/cage, and unsafe APPROACH rate conditional on a dangerous target
(regardless of selected identity). An additional metric counts actual dangerous
selected-entity approaches among all episodes. Null marks strata with no dangerous
targets. Aggregate numeric metrics report mean and sample standard deviation over
training seeds; token counts remain per pair because labels are arbitrary.

Save configurations, split manifests, commit/diff/hashed source snapshots, model,
optimizer/RNG checkpoints, learning curves, complete training records, and scene/
message/action/outcome evaluation logs. Exact logs include all message weights,
correct-action probabilities and expected rewards; sampled actions are logged
for auditing but are not used to compute the exact objective. Extra computation
from enumeration must be considered before any comparison with embodied RL.

For a quick check use a separate development seed and output:

```sh
python -m experiments.entity_safety.run --learner exact --lengths 2 --seeds 123 \
  --iterations 5 --batch-size 12 --evaluation-episodes 12 --evaluation-seed 1001 \
  --output output/entity_safety3_smoke
python -m pytest tests/experiments/test_entity_safety.py
```

Joint entity/response accuracy remains a diagnostic for the original
identity-sensitive task. The functional safety-only acceptance and simulator
transition are specified below. High response accuracy alone does not establish
compositionality. Experiment 2 and later language experiments are not implemented.

## Functional safety acceptance and first simulator transfer

The current functional gate measures the response for the sender's target;
entity identification remains a separate diagnostic. It does not authorize
movement toward an arbitrary predicted animal index. Existing exact two-token
checkpoints retain their original joint-reward training weights. Audit their
safety responses with:

```sh
python -m experiments.entity_safety.audit --output output/entity_safety3_acceptance
```

The audit requires zero deterministic response errors across all states/orderings
on fresh layouts, positive conservative coordinate-domain response bounds in all
six cases for all five seeds, and degradation with constant/zero messages. It
saves scene/action/probability records, checkpoint hashes and acceptance data.
This functional criterion supersedes the earlier proposed joint-accuracy gate
for the safety-only task. The original identity-sensitive results remain valid
as a separate diagnostic. Perfect stochastic behavior or visual safety inference
is not claimed.

Only after acceptance, run the first oracle MuJoCo transfer:

```sh
MUJOCO_GL=glfw python -m experiments.entity_safety.run_navigation --output output/entity_safety3_navigation
```

Use the appropriate MuJoCo backend for the host. No renderer or OpenCV is needed
for headless state stepping. The runner verifies all five accepted checkpoint
hashes before creating simulator scenes. It freezes their weights and trains a
small movement actor/critic using PPO, ordered environment transitions,
discounted rewards and generalized advantage estimation. Default movement budget
is 100 x 512 = 51,200 steps per seed, with 600 fresh evaluation episodes per seed.

The three animals are symbolic MuJoCo markers. Cage state is an oracle flag,
not inferred from an image. A public goal marker identifies which location to
consider. The message supplies that goal's safety; a WARN ends the episode,
whereas APPROACH enables a learned four-direction movement controller using the
public goal offset and current pose. Movement training uses safe goals; safety
handling is supplied by the already accepted communication module. All six
entity/cage cases are evaluated afterward, with target approaches, distractor
hazards, false warnings and timeouts scored separately.

This is a deliberately limited hierarchical transfer: the movement policy is
learned over time, while message generation/interpretation stays frozen. It is
not joint temporal communication learning, visual transfer, general obstacle
planning, or torque-based locomotion. As in the original troy_dev arena, direct
position changes plus MuJoCo `mj_forward` implement kinematic steps. Layouts
provide an initially clear straight path; the policy can still fail through
poor movement or encounters with dangerous distractors. Communication acceptance
and navigation success are reported separately. The runner saves movement
checkpoints, source/config metadata, learning curves and complete evaluation
trajectories, and checks that communication weights are unchanged.
