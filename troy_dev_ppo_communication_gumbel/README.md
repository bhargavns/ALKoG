# Gumbel communication pipeline

A transmitter observes scene triples and emits a learned categorical symbol.
A blind, current-message receiver chooses one of four movements. Joint PPO
trains both agents through a straight-through Gumbel channel. The default
vocabulary has 16 symbols, with no prescribed symbol meanings.

Visual perception is the normal input path. `perception=oracle_state` supplies
exact scene state for faster policy testing while retaining the same input
interface. Oracle results do not establish visual-perception performance.

## Setup

Run commands from this directory. Python 3.11+ is required. Choose a PyTorch
build appropriate for your CPU/CUDA environment, then install the dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
export MPLCONFIGDIR="$PWD/output/matplotlib"
```

MuJoCo needs a working rendering backend, including for environment creation
in oracle mode. Use the backend appropriate to your system. No sibling project,
`CORE` environment variable, or parent repository test configuration is needed.
SAM and torchvision models are imported only when visual inference is created.
Oracle testing requires no downloaded visual checkpoints.

## Fast oracle smoke test

```bash
python communication/train_communication.py perception=oracle_state device=cpu \
  iterations=2 horizon=32 checkpoint_every=1 videos=0
```

This checks execution and checkpoint generation, not learning. Runs appear in
`output/runs/`; every invocation creates a new directory. Defaults are Gumbel,
16 symbols, temperature 1.2, a current-message MLP, separate starting positions,
shared yaw, 800 iterations, horizon 2048, zero sender entropy bonus, and no videos.
`--help` lists all training defaults. Unknown key=value options fail explicitly.

## Visual perception

```bash
python perception/download_sam_checkpoint.py
python perception/generate_category_dataset.py device=cuda scenes=200
python perception/train_category_model.py device=cuda epochs=30
python perception/evaluate_category_model.py device=cuda
python communication/train_communication.py perception=softcat device=cuda \
  sender_entropy_beta=0.1 sender_entropy_schedule=annealed
```

The SAM checkpoint is stored under `models/`; ResNet-18 ImageNet weights are
obtained through torchvision's cache on first use. Dataset generation saves
`output/category_dataset.pt`; training saves `output/category_model.pt`.
Use `category_model=<path>` to select another category checkpoint.
The category vocabulary includes food, lion, cage, water, poison, receiver, and
background. Simulator segmentation supervises category preparation; inference
uses RGB proposals, 576-dimensional visual/color features, category predictions,
box relations, and ground projection.

Both visual and oracle modes include the receiver's exact live position in a
reserved input slot. Visual scene identities, relations, and object positions
come from perception. Oracle mode additionally supplies exact scene state,
including objects hidden from the cameras. `colocated=1` follows the receiver
and refreshes scene perception every 75 steps. Stationary mode perceives at
reset and for the configured opening `perceive_passes`; anchors persist within
an episode and clear at reset.

## Evaluation and experiments

[Entity–safety Experiment 1](experiments/entity_safety/README.md) adds an oracle
selection diagnostic with complete one- and two-token utterances, a six-token
vocabulary, matched communication controls, and a sequential five-seed runner.
It has its own task, policies, checkpoints, and logs alongside this navigation
baseline. It requires torch and numpy, with no rendering or visual checkpoints.
The [functional acceptance](results/entity_safety_functional_acceptance.md) and
[first oracle MuJoCo transfer](results/entity_safety_navigation.md) report safety
classification and multi-step movement separately.

```bash
python evaluation/evaluate_communication.py --run-dir output/runs/REPLACE_WITH_RUN \
  --episodes 500 --modes hard sampled constant random shuffle --device cpu
```

`hard` selects the sender argmax; `sampled` samples sender categories as in
training. Receiver movements are sampled unless `--greedy` is supplied.
The other modes intervene on messages. `--checkpoint 25` evaluates an intermediate
checkpoint; `--video-dir <path>` records episodes. For relocated visual runs,
`--category-model <path>` overrides the saved category checkpoint location.
Outputs are never silently overwritten.

[Matched experiments](evaluation/run_entropy_pair.py) compare scheduled sender entropy with
zero bonus, or a coupled channel with a detached channel. The direct-loss probe
is a diagnostic, not policy pretraining:

```bash
python evaluation/probe_gumbel_direct.py --symbols 16 --entropy-beta 0.1 \
  --entropy-schedule annealed --seeds 0 1 2 3 4 --out output/direct_loss.json
```

Vocabulary comparisons, entropy comparisons, failed-code inspection, and receiver
symbol intervention tools are under `evaluation/`. Read their `--help` for options.

## Contents and validation

- `environment/`: scene simulation, semantic categories, and `assets/kg_world.xml`.
- `perception/`: visual features, category grounding, dataset generation, classifier training/evaluation.
- `representation/`: fixed scene vocabulary, triples, anchors, oracle adapter, and refresh cadence.
- `communication/`: sender, Gumbel channel, current-message receiver, joint PPO, sender entropy, and communication training.
- `evaluation/`: checkpoint evaluation, diagnostics, and matched experiment launching.
- `shared/`: argument parsing and logging.
- `observability/`: metrics, plots, and video recording.
- `tests/`: tests grouped by the corresponding stage, with shared fixtures at the root.
- `docs/`: architecture documentation.
- `results/`: portable summaries of saved Gumbel arena evaluations.
- `output/`, `models/`: ignored local artifacts, not required source files.

```bash
python -m pytest
```

The local pytest configuration does not inherit parent coverage options. Tests
use controlled perception proposals rather than downloading pretrained weights;
full SAM/ResNet inference must be checked separately with the visual checkpoints.
See [architecture](docs/ARCHITECTURE.md) and [saved results](results/README.md).

New runs use configuration version 2 and one-message replay. Existing Gumbel
current-message checkpoints in version 1 remain evaluable. `init_from=<run>`
restores weights only, not optimizer, environment, RNG, or training progress.

## Matched experiment launch

```bash
python evaluation/run_entropy_pair.py --prepare --iterations 800 --device cuda
python evaluation/run_entropy_pair.py --launch output/experiments/REPLACE_WITH_PREPARED_DIRECTORY
```

Preparation snapshots all pipeline stages and runs the five-seed Gumbel preflight.
Launch checks snapshot hashes before starting both arms and their final evaluations.
Use `--control entropy` (default) for scheduled versus zero sender entropy, or
`--control detached` for coupled versus detached channels with zero sender entropy.
Source snapshots and existing checkpoints under ignored `output/` retain their
original layouts; new snapshots use the stage directories above.

Scene vocabulary checkpoints are named `scene_vocabulary_init.pt` and
`scene_vocabulary_trained.pt`. Evaluation also accepts existing runs with
`kg_init.pt`; transmitter and receiver weight formats are unchanged.
