# Plan: differentiable communication with PPO

Date: 2026-09-12. Scope: the implementation in this directory. This document records the original design and experiment plan. The implementation is now available; see [DIAL.md](DIAL.md) for usage and validation status. Full arena experiments remain separate from implementation verification.

## Recommendation and paper basis

Implement a DIAL-inspired PPO variant alongside the existing independent-PPO baseline. Retain the perception stack, transmitter slot encoder, receiver transformer, ten-message context, rewards, and four-symbol execution channel for the first comparison.

Foerster et al.'s DIAL passes recipient learning gradients into the sender through a continuous training channel. Its discretise/regularise unit uses `sigmoid(z + Gaussian noise)` for training and `1[z > 0]` for execution; their default noise standard deviation is 2. Their implementation uses recurrent Q-learning, not PPO. Parameter sharing is optional. Thus, adopting their channel with PPO is an adaptation, not an exact reproduction. Straight-through Gumbel-softmax, previously suggested in `RESULTS.md`, is not their method. Source: [Foerster et al., sections 5.2–6](https://proceedings.neurips.cc/paper_files/paper/2016/file/c7635bfd99248a2cdef8249ef7bfbef4-Paper.pdf).

Everything below is a project-specific design proposal. The objective is to establish a useful gradient path and test whether it produces useful **discrete** communication; differentiability alone does not guarantee successful navigation.

## 1. Fix all three gradient breaks

Current `CommunicationTrainer.collect_and_update()` calls `tx.act()` under `no_grad`, converts the sampled result to a Python integer, and stores receiver inputs as integer contexts. Its receiver update only evaluates `rx_ctx[idx]`. Even replacing the categorical sampler with a differentiable function would not reconnect those stored contexts to the transmitter.

The new update must compute this complete graph:

```text
stored transmitter inputs for each message in the window
    -> current transmitter encoder and message head
    -> noisy continuous channel, with recorded noise
    -> differentiable receiver message embeddings
    -> receiver transformer and action distribution
    -> PPO loss -> gradients into both agents
```

Collecting experience under `no_grad` remains appropriate. Reconstruct the differentiable graph during optimization; do not retain a 2,048-step simulator-time autograd graph. SAM, category inference, discrete triples, environment transitions, rewards, and sampled movement actions do not need to become differentiable.

## 2. Use two message bits and preserve four execution symbols

Add a transmitter message head with two unrestricted real outputs `z=(z0,z1)`, replacing its four-way categorical communication head in the new variant. Keep its existing perception/symbol/attention trunk.

Training channel:

```python
epsilon = normal_sample(shape=(2,))  # recorded once per emitted message
b = sigmoid(z + sigma * epsilon)
```

Execution channel:

```python
b = (z > 0).to(float)
symbol_id = 2 * b[0] + b[1]
```

The bit pairs `00,01,10,11` map to `a,b,c,d`. Hard execution therefore still transmits exactly one of four symbols, with a maximum capacity of two bits per step. Continuous training messages can carry more information, which is why hard-channel evaluation is mandatory.

Start with `sigma=2` as the paper-derived reference setting; treat its suitability here as untested. Keep sigma fixed throughout a rollout and every PPO update on that rollout. If necessary, compare predeclared alternatives such as 0.5 and 1.0 on validation seeds. Avoid introducing temperature annealing, hard straight-through gradients, and extra message objectives in the first experiment.

Initialize the new head with small weights and zero bias to avoid saturating its sigmoid immediately. Log both pre-noise logits and post-noise bit values.

## 3. Give the receiver a differentiable input adapter

Retain the receiver's four learned symbol embeddings and construct a continuous interpolation over them. For bit values `b0,b1`, define:

```text
w = [(1-b0)(1-b1), (1-b0)b1, b0(1-b1), b0b1]
message_embedding = w @ symbol_embedding_table[:4]
```

This is a proposed adapter for the existing architecture, rather than a claim about the paper's implementation. It keeps message embeddings four-dimensional. At binary corners it is exactly the existing symbol lookup, so hard inference and training use compatible receiver representations without increasing transformer width.

Add a receiver forward path accepting `[batch, 10, 2]` message bits plus an explicit validity mask. Construct message embeddings, prepend the learned CLS embedding, add the existing positional embeddings, and run the existing transformer and heads.

Padding must remain separate from bit values: `00` is a valid symbol, not an empty slot. Preserve NULL masking and reset context at episode boundaries. Keep the current integer-input method for the baseline and hard-execution equivalence checks.

## 4. Redesign the rollout buffer around message provenance

For each emitted message, store its transmitter pose, triple IDs, deltas, Gaussian noise sample, episode ID, and timestep. For each receiver action, store the IDs of the up-to-ten message records that formed its context, a validity mask, movement action, old receiver log probability, old value, reward, and termination/truncation flags.

During each minibatch update:

1. Gather all message records referenced by the sampled receiver steps.
2. Recompute their logits through the **current** transmitter with gradients enabled.
3. Apply the recorded noise and continuous channel.
4. Reassemble complete receiver windows, then recompute action log probabilities and values.
5. Backpropagate the joint loss through the receiver and all contributing messages.

The current transmitter is feed-forward and the receiver has a fixed input window, so a full recurrent episode unroll is unnecessary. Randomly sampled receiver timesteps are acceptable only when each sample reconstructs its entire valid window. Independent timesteps with detached past message embeddings are insufficient.

Deduplicate message records inside a minibatch to reduce repeated transmitter computation. Preserve the last nine prior message-input records across horizon boundaries so the first action in a new rollout can reconstruct its context. Do not allow windows to cross episode resets. Fixed source features can be copied into buffers; differentiable message activations must be recomputed.

## 5. Train the composed movement policy with PPO

Treat the transmitter/channel/receiver composition as the policy that chooses environment actions. Remove the transmitter's independent categorical-message PPO objective in the new variant: its message output is now an internal differentiable activation.

For fixed recorded history `h` and channel noise `epsilon`, use:

```text
ratio = pi_current(action | h, epsilon) / pi_rollout(action | h, epsilon)
loss = clipped_PPO_actor_loss
       + value_coefficient * normalized_receiver_value_loss
       - entropy_coefficient * receiver_action_entropy
```

The numerator recomputes the complete transmitter-to-receiver path. The denominator is the stored old **movement** log probability. Noise has a fixed, parameter-independent distribution; conditioning both evaluations on the same recorded samples avoids comparing different stochastic forwards. This is a PPO surrogate for the continuous training system, not a likelihood-ratio derivation for the eventual hard binary policy.

Do not add a transmitter message probability ratio, use stale message tensors in the numerator, or redraw noise each update. Recompute past messages even if their original rollout predates the latest parameter update, analogously to recomputing a history-dependent policy's internal features.

Use one Adam optimizer with named transmitter/receiver parameter groups and one backward/step per minibatch. This is an implementation choice for synchronized updates, not a requirement to share weights. Two optimizers could also work if both gradients are computed before either steps. The agents remain separate networks with separate execution interfaces; optimizer count does not determine whether execution is decentralized.

Initially let both the receiver actor and normalized receiver critic loss backpropagate through the channel. Retain value normalization and gradient clipping. Remove the unused transmitter critic from the new model, rather than training a second return predictor without a defined role. Log actor-only and value-only gradient norms at the message head separately, since a code useful only for predicting return would not establish useful control. If critic gradients dominate, compare a declared ablation that detaches messages only on the critic branch.

Retain the baseline's learning rate, discount, GAE lambda, clipping ratio, and entropy coefficient initially. Add action-policy KL monitoring and early stopping on excessive update drift. Do not infer healthy communication from an increasing transmitter gradient norm alone.

### Value timing and episode boundaries

The receiver value is evaluated **after** the current message is appended, just before its movement action. Bootstrap values must use that same phase. At a nonterminal horizon boundary, prepare the next transmitter input and one noise draw, append its message for the bootstrap, and cache that pending input/noise so the next rollout consumes the same logical message once. Recompute its activation under updated weights when collecting the next rollout.

Keep true termination distinct from truncation. For a finite 300-step task, explicitly retain terminal treatment at the time limit to match the existing baseline. If instead treating the limit as a collection cutoff, bootstrap the final pre-reset state and rerun the baseline with that same convention. Never bootstrap from the next episode's reset state.

## 6. File-level implementation sequence

| File/change | Work |
|---|---|
| New `lib/CommunicationChannel.py` | Noisy sigmoid and hard binary modes, symbol conversion, controlled noise input, and channel diagnostics. |
| `lib/SymbolPolicy.py` | Expose shared encoded features and add a transmitter-specific two-logit model while preserving `TripleActorCritic` for baseline/single-agent runs. |
| `lib/ReceiverPolicy.py` | Continuous-bit adapter, explicit padding mask, shared transformer core, and retained integer inference. |
| New `lib/DIALPPOTrainer.py` | Provenance-aware buffer, full-window replay, joint PPO update, and correctly timed bootstrap. Reuse `TransmitterInput` from the existing trainer. |
| `scripts/train_communication.py` | Add an explicit `method=independent_ppo|dial_ppo` selection, channel settings, and construction of the appropriate model/trainer. Preserve the existing baseline behavior. |
| `lib/VideoRecorder.py` | Explicit hard/continuous recording mode, using the same channel and input assembly as training/evaluation. Default demonstration videos to hard execution. |
| `lib/Diagnostics.py` | Plot actual method-specific metrics, including cross-agent gradients, KL, saturation, and hard-channel results. |
| New `scripts/evaluate_communication.py` | Separate evaluation environment, fixed scene seeds, hard channel, controlled action selection, and message ablations. |
| Run metadata/checkpoints | Save method/version, channel settings, vocabulary, context length, model shapes, perception settings, seeds, and optimizer/RNG state if exact resumption is supported. |

New two-output message heads are incompatible with existing four-logit transmitter checkpoints. Default to fresh training for comparisons; any trunk warm start must be explicit and applied to matched experimental arms.

## 7. Verify the mechanism before expensive training

Create small tests that require no SAM checkpoint or renderer:

- A receiver action loss produces finite, nonzero gradients on transmitter message-head parameters for a nondegenerate synthetic batch; an explicit detached-channel control removes that gradient.
- Receiver loss reaches valid earlier messages in the ten-message window, including prefix records from the preceding rollout, and never crosses an episode boundary.
- With unchanged parameters and recorded noise, recomputed action log probabilities match rollout log probabilities and PPO ratios equal one within numerical tolerance.
- All four binary corners produce the same receiver outputs through continuous and integer paths. `00` stays unmasked; NULL stays masked.
- Serialization, noise reuse, prefix handling, and bootstrap preparation do not duplicate or drop a message.

Then run a tiny cooperative identification task: transmitter observes one of four uniformly sampled targets; receiver receives only the message and chooses one of four actions for a correctness reward. Train with the same PPO/channel code and evaluate with hard bits. A reasonable engineering gate is at least 95% hard-channel accuracy across three seeds, against a 25% chance level. This is a plumbing/optimization check, not evidence of solving the arena.

## 8. Arena experiment and acceptance criteria

Use matched seeds, environment-step budgets, evaluation settings, perception checkpoints, and model settings. Keep evaluation/video environments separate from training so diagnostics do not change training resets or RNG streams.

Run these arms:

1. Existing independent PPO baseline, freshly run under the comparison setup.
2. Proposed noisy differentiable channel with hard-channel evaluation.
3. The same new architecture and noisy channel with messages detached during optimization, isolating cross-agent gradients from architecture/channel changes.

Start with short diagnostic runs to catch exploding KL, saturation, disconnected gradients, or incorrect masks. Short flat learning curves alone should not terminate a candidate: the existing task can learn slowly. Once the mechanism checks pass, use at least seeds 0, 1, and 2 and the same 450×2,048-step budget as the baseline; expand seeds if uncertainty remains large.

Evaluate each frozen checkpoint on a prespecified held-out scene set, for example 500 episodes per training seed, using:

- Hard two-bit communication as the primary deployment condition.
- Noisy continuous communication with the training sigma as a train/evaluation-gap diagnostic.
- Constant and independently randomized hard messages as communication-use ablations.
- Message-stream shuffling across episodes as an additional intervention, with its distribution shift reported.

Choose stochastic versus greedy movement evaluation in advance and apply the same choice to all arms; reporting both is useful. Report food success, return, loose-lion death rate, episode length, per-seed estimates, and confidence intervals. Separate training-seed variability from within-seed episode sampling uncertainty. Log hard symbol usage and channel saturation, but do not use them as substitutes for task performance.

Success requires: verified actor gradients across the channel; improved held-out **hard-channel** task performance over matched independent and detached controls without hiding worse lion safety; and a meaningful performance drop when messages are removed or randomized. As a provisional engineering target, require the hard-channel food rate to be within five percentage points of continuous-channel performance, subject to sampling uncertainty. Prespecify checkpoint selection using validation scenes, leaving final test scenes untouched.

If continuous performance improves but hard performance collapses, the gradient fix works but the learned code exploits analog precision. Investigate noise strength and logit margins before claiming success. If only the critic improves, inspect actor gradient attribution and the critic-detach ablation. If the synthetic test succeeds but neither arena channel learns, investigate task observability and perception rather than merely increasing training duration.

## 9. A separate issue differentiability cannot remove

The receiver's yaw is randomized independently each episode, absent from the transmitter input, and absent from the receiver input. The receiver also has no explicit previous-action input. Thus, a four-symbol direction code cannot immediately map transmitter-frame directions to receiver-frame moves without some inferred calibration/history. The transmitter currently has no recurrent state to remember its own previous emissions or receiver motion.

Keep this limitation unchanged for the first matched comparison. If it becomes the next bottleneck, use a clearly labeled shared/fixed-yaw diagnostic to separate credit assignment from orientation inference. A subsequent architecture experiment could add transmitter temporal memory over receiver positions and emitted messages, and receiver previous-action embeddings. Do not silently supply yaw, target coordinates, or simulator state to the receiver; such changes would alter the task and invalidate the original comparison.

The first deliverable is therefore a tested differentiable channel and correct PPO recomputation, followed by evidence that its learned protocol survives the original four-symbol execution constraint. Broader memory or perception changes come after that result is measurable.
