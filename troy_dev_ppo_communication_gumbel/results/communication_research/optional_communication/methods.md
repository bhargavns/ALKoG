# Optional temporal communication study

Run from `troy_dev_ppo_communication_gumbel`. Existing safety acceptance and
MuJoCo navigation modules are untouched. This module introduces a separate
vectorized PyTorch **oracle kinematic grid**, not a replacement for those tasks.

## Windows setup

From the repository root:

```powershell
python -m venv .venv-event
.venv-event/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-event/Scripts/python.exe -m pip install numpy pytest 'mujoco>=3.2,<4' gymnasium matplotlib opencv-python
.venv-event/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

`requirements-lock.txt` records the actual environment. CUDA wheels include the
runtime; the study does not require compiling CUDA or changing NVIDIA drivers.
The tiny sequential networks can also run with `--device cpu`.

```powershell
cd troy_dev_ppo_communication_gumbel
../.venv-event/Scripts/python.exe -m experiments.event_communication.study --output output/NEW_DIRECTORY --device cuda --seeds 0 1 2 --iterations 75 --batch-size 256 --evaluation-episodes 1024 --warmup 1500
../.venv-event/Scripts/python.exe -m pytest tests/experiments/test_event_communication.py -q
```

## Task and information boundary

11 by 7 cells, four movements, 24-step finite episodes, absorbing success/death.
Start and goal share a random row; the goal is at x=10 and the lion at x=5.
Half of lions are on that initial corridor; others are three rows away. Status
is initially dangerous with probability .5. Independently, half of episodes
flip status once at a uniformly sampled time 2 through 7, unless already ended.
The lion is stationary; becoming loose activates its occupied cell as a hazard.
There is always a detour. Communication therefore competes against conservative
navigation; it is not logically necessary to reach every goal.

The receiver sees its pose and relative goal/lion geometry (six floats). Only
the sender sees danger status. The sender gate additionally observes its own
last sent status, elapsed silence and episode time. It never reads receiver
hidden state. The centralized critic sees true danger and sender history during
training only. Public geometry is independent of private status.

A single categorical token has four possible values, with no assigned token
meanings. Receiver memory predicts one scalar danger belief; movement consumes
this belief plus public geometry. This **imposes a safety belief bottleneck**.
It does not establish unconstrained semantic emergence or compositionality.
GRU memory is 16-dimensional, fact-cache memory holds the latest learned scalar,
and memoryless decoding reverts to .5 on silence. The `graph` arm is that single
fact cache, not the repository's full learned knowledge graph.

## Stages and supervision

1. BFS behavior cloning establishes spatial navigation. Teacher states include
   true danger. Cell/action balancing emphasizes rare avoidance decisions.
2. Sender/receiver warm-up enumerates all four messages for two private states,
   maximizing exact expected correct safety-response reward. GRU/fact cache
   additionally predict the response over eight steps after a single message.
   No token receives a predefined label, but safety correctness is privileged
   supervision. This is intentionally a modular bootstrap.
3. Joint temporal PPO samples gate, message, and movement actions. The joint
   log probability includes a message only when sent. Ordered 24-step histories
   are replayed exactly with BPTT; three PPO epochs, clip .2, gamma .99,
   lambda-GAE .95, Adam .0003, gradient clipping .5. Central value baseline is
   independent of the sampled current message and gate. Gate entropy bonus .01
   maintains exploration. Transmission cost ramps over the first 25 updates.

Reward: -.1 per active movement, +.1 per Manhattan-distance improvement, +5 on
success, -8 on collision, minus cost per attempted transmission. This distance
shaping is a practical dense reward, not claimed policy-invariant shaping.
Default communication cost .05; `free` uses zero and `expensive` uses .2.
All methods can adapt navigation during PPO, including conservative silence.

## Methods

| Arm | Change |
| --- | --- |
| gru | Learned optional gate and recurrent memory |
| graph | Learned gate and one-fact persistent cache |
| memoryless | Learned gate, no persistence during silence |
| continuous | Send every active step |
| periodic | Send at steps 0, 4, 8, ... |
| random | Bernoulli(.25) timing |
| opening | One opening transmission |
| none | No messages; uncertain belief .5 |
| oracle | True danger directly available; no transmissions |
| free | GRU, zero transmission cost |
| expensive | GRU, .2 transmission cost |
| auxiliary | GRU, additional .1 binary state-prediction loss |
| counterfactual | GRU plus fitted communication Q values and detached Q-difference gate surrogate |
| cold | Transfer spatial skill, initialize all communication components randomly |

The counterfactual arm estimates Q for each gate choice from bootstrapped return
targets on the observed choice. Its .05 Q-difference surrogate is an approximate
model-based gate update, **not exact MACC or ground-truth counterfactual reward**.
Unchosen Q estimates can be poor. The cold arm still has spatial pretraining.

## Evaluation

Final checkpoints only. Initial scene seed 900001 overlapped the development
checks and is retained as development evidence. Primary report results use a
fresh final scene seed 1900001 after every method and weight was frozen, with
no subsequent tuning. Training seeds 0-2, development training seed 123.
Final re-evaluation runs on CPU. Evaluation uses deterministic gate,
symbol and movement argmax. Random timing/dropout use seed 987654.

Four scenarios: dynamic held-out layouts, no status changes, rotation by 90
degrees, and status changes four steps later. Rotation changes public geometry
and movement coordinates consistently; navigation itself has not been pretrained
on rotated goals, so oracle failure is an essential interpretation control.

Seven message interventions: normal, suppress all updates after t=0, constant
token, shuffle across lanes, one-step delay, independent 25% packet loss, and
suppress repeated unchanged status. Attempted sends are still charged; sender
history assumes transmission without acknowledgment. Silence and timing remain
observable, including under constant payloads.

Two extra dynamic interventions replace the gate with periodic/random timing at
the learned normal policy's observed send rate. Periodic phases are randomized
per episode. This matches **expected step rate**, not exact realized messages or
episode length; report actual counts. These are timing interventions on the same
trained model, in addition to independently trained timing baselines.

Outputs include manifest and exact source snapshot/hash, environment lock,
warm-up weights, final model/optimizer/RNG states, per-update curves, evaluations
and paired normal/update-deletion trajectories for 32 held-out episodes per run.
Scene streams are matched across arms. Results are descriptive across three
training seeds; evaluation episodes are not independent training replicates.

No visual perception, physical locomotion, private-location signaling, multiple
hazards, role exchange, newcomer learning, natural language or full graph-learning
claim follows from this study. Such extensions need separate tasks and budgets.

## Follow-up studies and report

After the main study completes, run the prescribed event control, then both
learned fork-based gates. The paired methods freeze the control's learned code
and navigator, so their extra simulation queries are not budget-matched to PPO.
Regression fits return magnitude with positive-state oversampling;
classification predicts the sign using ordinary binary cross entropy.
Both compare SEND versus SILENT with an otherwise silent continuation.

```powershell
foreach ($seed in 0,1,2) {
  ../.venv-event/Scripts/python.exe -m experiments.event_communication.event_control --study output/event_study_v1 --output "output/event_control_seed$seed" --seeds $seed
  ../.venv-event/Scripts/python.exe -m experiments.event_communication.paired_gate --control "output/event_control_seed$seed" --seed $seed --output "output/event_paired_value_final$seed"
  ../.venv-event/Scripts/python.exe -m experiments.event_communication.paired_gate --control "output/event_control_seed$seed" --seed $seed --objective classification --output "output/event_paired_class_final$seed"
}
../.venv-event/Scripts/python.exe -m experiments.event_communication.probe --study output/event_study_v1 --output output/token-probes.json
../.venv-event/Scripts/python.exe -m experiments.event_communication.fresh_evaluation --study output/event_study_v1
../.venv-event/Scripts/python.exe -m experiments.event_communication.report --study output/event_study_v1 --output output/event_report
../.venv-event/Scripts/python.exe -m experiments.event_communication.finish_report --study output/event_study_v1 --output output/event_report
```

The HTML report dependency is `markdown` (included in the final lock). Report
generation uses all three seeds; incomplete runs should not be presented as a
completed study. `finish_report` runs after `report` and appends the follow-ups.
