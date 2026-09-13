# DIAL PPO implementation

The DIAL path is implemented alongside independent PPO. It trains a two-bit noisy sigmoid channel jointly through the receiver's movement-policy loss, reconstructing all ten messages in each receiver window from stored transmitter inputs and channel noise. Execution thresholds the logits into four symbols. The sender has no private critic or categorical-message PPO objective.

This is the PPO adaptation described in [DIAL_PLAN.md](DIAL_PLAN.md), not a reproduction of the paper's Q-learning algorithm. The implementation does not establish that the original arena task is solved.

## Training

From `troy_dev_ppo_communication`:

```bash
python3 scripts/train_communication.py method=dial_ppo seed=0 run_name=dial_s0
```

The communication entry point now uses this directory's logger and argument parser and does not require `CORE`. Dependencies and checkpoints are the same as the existing perception pipeline. CUDA is the default; `device=cpu` works but SAM inference is expensive.

The method defaults to `independent_ppo`, preserving the prior algorithm. For matched experiments:

```bash
python3 scripts/train_communication.py method=independent_ppo seed=0 run_name=baseline_s0
python3 scripts/train_communication.py method=dial_ppo seed=0 run_name=dial_s0
python3 scripts/train_communication.py method=dial_ppo detach_channel=1 seed=0 run_name=detached_s0
```

Repeat with seeds 1 and 2. Defaults remain 450 iterations × 2,048 environment steps, learning rate 0.0003, ten PPO epochs, minibatch 256, value normalization, and gradient clipping. DIAL-specific arguments:

| Argument | Default | Meaning |
|---|---|---|
| `channel_sigma` | `2.0` | Gaussian noise standard deviation before sigmoid. |
| `target_kl` | `0.03` | Stop PPO updates when minibatch action KL estimate exceeds this value; zero disables. |
| `detach_channel` | `0` | Set to 1 for the architecture-matched control with no receiver-to-transmitter gradients. |
| `critic_detach` | `0` | Set to 1 to stop only the value-loss gradient at the message boundary. |
| `video_channel` | `hard` | `continuous` is available for DIAL inspection only; hard is the execution condition. |

Videos use a separate environment and preserve PyTorch RNG state, so recording does not reset training episodes or consume their random stream. A separate environment changes historical video-enabled baseline trajectories; rerun matched baselines rather than treating old logs as exact controls.

Each run writes `config.json`, initial/trained KG files, policy checkpoints, JSONL metrics, reports, channel/loss plots, and optional videos. Config records method, shapes, vocabulary, channel settings, perception path, seed, and the finite-horizon terminal convention. `init_from=<run-directory>` restores the final policy weights only; optimizer, simulator, and RNG state are not resumed. Four-logit baseline transmitter checkpoints cannot load into the two-logit DIAL model.

## Shared-yaw diagnostic

Add `receiver_yaw_mode=shared` to both the DIAL and detached-control commands.
For example, keeping the September 13 experiment's sigma of 0.5:

```bash
python3 scripts/train_communication.py method=dial_ppo receiver_yaw_mode=shared channel_sigma=0.5 seed=0 run_name=dial_shared_s0
python3 scripts/train_communication.py method=dial_ppo detach_channel=1 receiver_yaw_mode=shared channel_sigma=0.5 seed=0 run_name=detached_shared_s0
```

The default `receiver_yaw_mode=independent` preserves the original task. Shared
mode copies the transmitter's randomly sampled yaw at each reset; both stay fixed
throughout the episode. Both modes consume the same RNG draws, preserving layout
sequences for matched reset sequences. The setting is saved in `config.json`, used
by videos, and automatically restored by evaluation. Older run configurations
without this field evaluate with independent yaw. Start fresh runs for the matched
diagnostic; an explicit weight warm start is a separate transfer experiment.

Verification on September 13: all 18 tests passed, including shared-frame movement,
fixed within-episode yaw, matched layout RNG streams, and independent-mode defaults.
The yaw tests use real MuJoCo state and movement with a stub renderer. No new training
experiment was launched.

## Held-out evaluation

Use the new evaluator in a separate process. It reconstructs the model from `config.json` and loads `kg_init.pt` plus the selected policy weights:

```bash
python3 scripts/evaluate_communication.py \
  --run-dir output/runs/<dial-run> \
  --episodes 500 --scene-seed 200000 \
  --modes hard continuous constant random shuffle
```

For independent PPO omit `continuous`. `--checkpoint 25` selects iteration 25 instead of final weights. `--device cpu` enables CPU evaluation. Receiver actions are sampled by default; add `--greedy` for greedy movement, and use the same setting for all comparisons.

Hard mode sends exactly one of a/b/c/d. Continuous mode uses training noise. Constant mode sends a, random mode sends independent uniform symbols, and shuffle mode supplies a different held-out episode's hard message stream (holding its final symbol if the recipient runs longer). Shuffle changes trajectory/message alignment and is an intervention, not an on-distribution policy.

Output contains per-episode outcomes, symbol streams, scene seeds, food/return/length/loose-death metrics, and Wilson intervals for event rates and episode-bootstrap intervals for mean returns/lengths within one training seed. It does not estimate uncertainty across independently trained seeds. Evaluation refuses to overwrite an existing output file; use `--out` for a distinct artifact. Use separate validation and final test scene seed ranges when selecting checkpoints.

## Focused verification and learning probe

Run the mechanism tests from this directory without inheriting repository-wide pytest configuration:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  -c /dev/null --rootdir=. --confcutdir=. -q tests/test_dial.py -p no:cacheprovider
```

The tests require installed Python dependencies but no renderer, SAM inference, or pretrained weights. They cover gradient connectivity and detachment, historical-message gradients, masks, binary/integer equivalence, rollout prefixes and resets, pending bootstrap messages, old-policy replay, KL rejection, checkpoint round trips, interventions, and the retained independent trainer.

The four-target probe uses the production channel, receiver, window reconstruction, and joint PPO update, with a small sender receiving a one-hot target and one-step reward. It batches independent one-step episodes for speed; rollout lifecycle is covered separately by the tests.

```bash
python3 scripts/probe_dial.py --seeds 0 1 2 --out output/dial_probe.json
python3 scripts/probe_dial.py --seeds 0 --detach --out output/dial_probe_detached.json
```

Probe defaults are 300 updates, batch 512, four PPO epochs, learning rate 0.003, sigma 2, and CPU. These intentionally differ from the arena budget. It exits nonzero when any non-detached seed fails the 95% hard-channel greedy-accuracy gate. It writes the result even on a failed gate, so failures remain reviewable. A passing toy probe is not evidence of arena navigation or temporal communication.

## Implementation boundaries

- `CommunicationChannel.py`: continuous/hard modes and four-corner embedding weights.
- `SymbolPolicy.py`: separate `DIALTransmitter` using the existing symbolic trunk.
- `ReceiverPolicy.py`: continuous-bit input with explicit validity masking.
- `DIALPPOTrainer.py`: message provenance, full-window replay, pending bootstrap input/noise, one synchronized optimizer, actor/value message-gradient diagnostics, and KL stopping.
- `CommunicationEvaluation.py`: shared inference path for videos and evaluation.
- `evaluate_communication.py` and `probe_dial.py`: deployment/intervention evaluation and the toy learning gate.

Static scene grounding remains frozen, receiver localization remains oracle-fed, and receiver orientation is independently randomized by default; the shared-yaw diagnostic is optional. No recurrence, previous-action inputs, auxiliary communication objectives, or straight-through gradients were added. Time-limit truncation retains the baseline's finite-task terminal treatment. The observed hard-symbol histogram during training thresholds noiseless logits for diagnostics; it is not the continuous message distribution used by PPO.

## Validation recorded on 2026-09-12

Twelve focused tests passed on CPU. A real MuJoCo/SAM/category-model run completed one DIAL iteration of eight steps, wrote both policy checkpoints and the trained KG, and generated training/channel plots. Its message-head actor and value gradient norms were approximately 0.0212 and 0.0423. This short run completed no episodes and measures integration, not navigation performance. A separate hard-channel evaluation successfully reloaded its checkpoints and completed one 300-step episode with real perception; the untrained policy timed out. Separate two-step video checks verified three-frame MP4 files for DIAL hard, DIAL continuous, and independent hard execution using synthetic empty detections with the real renderer.

The four-target probe produced these hard-channel greedy accuracies after 300 updates:

| Channel setting | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Sigma 2, reference | 100% | 50% | 50% |
| Sigma 1 | 100% | 75% | 75% |
| Sigma 0.5 | 100% | 100% | 75% |
| Sigma 2, detached control | 25% | Not run | Not run |

The three-seed 95% gate **has not passed**. Some continuous-channel accuracies approach 100% while hard execution remains worse, demonstrating the analog-to-discrete gap that the plan calls for measuring. The reference sigma remains 2; the comparison did not justify silently selecting a successful seed or declaring the task solved. Full 450-iteration arena experiments were not run.

Machine-readable settings, outcomes, smoke metrics, and failed gates are preserved in [output/dial_validation.json](output/dial_validation.json).
