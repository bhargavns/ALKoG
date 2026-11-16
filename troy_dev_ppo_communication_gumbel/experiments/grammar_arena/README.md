# Fixed-slot communication transferred to the MuJoCo arena

This experiment freezes the existing `modular_curriculum` neural sender/listener,
then trains a separate waypoint-following neural PPO controller on the NVIDIA
GPU. The actual arena inherits `CommunicationKGWorldEnv._apply_action` and
extends its original `kg_world.xml` asset with four candidate foods, a tiger and
three passages. Original environment files remain byte-identical.

The original unfiltered transfer and a later training-only protocol-admission
follow-up are reported separately. The follow-up keeps a checkpoint only when
all four facts decode correctly on all 48 TRAINING meanings; otherwise it retries
the same fixed-slot PPO procedure at predetermined seeds, retaining every failed
attempt. It does not select on held-out scenes or introduce another grammar
induction method. See `protocol-quality-control.json` for the fixed rule.

This is exact-state, kinematic arena navigation. The receiver's motor policy sees
only a public waypoint direction and distance. Public passage polylines and
their interpolation are supplied. Visual perception, learned path planning,
torque locomotion and new grammar-induction methods are outside this experiment.

## Reproduce

From `project/`, using Python 3.11 with the installed torch CUDA, mujoco, numpy,
gymnasium and pytest packages:

```powershell
python -m experiments.grammar_arena.study --source-study PATH_TO_ORIGINAL_GRAMMAR_STUDY --protocol ../protocol.json --output ../reproduced_study --workers 3
python -m pytest tests/experiments/test_grammar_arena.py -q -p no:cacheprovider
```

The source study must contain `context_modular_curriculum_0` through seed 4,
`food_pair_modular_curriculum_0` through seed 4, and the corresponding independently
trained `silent_task` folders. The full download includes these checkpoints under
`source_protocols/`; use that path when reproducing. Output folders must be new.

For the follow-up, from the bundle root:

```powershell
python calibrate_protocols.py --source source_protocols --output new_admission
python run_admitted.py --quality new_admission --first-study new_zero_shot --source-study source_protocols --output new_admitted --workers 2
```

The original repository also includes the training-only check as
`python -m experiments.grammar_arena.quality_control --source SOURCE --output OUTPUT`.

One navigation run uses:

```powershell
python -m experiments.grammar_arena.train_navigation --seed 0 --updates 60 --environments 32 --horizon 128 --device cuda --output ../one_navigator
```

The final study uses five navigation seeds, two prior meaning splits, two geometry
families and seven channel conditions, with 32 contexts per private meaning.
All final checkpoints are retained. No checkpoint or final seed is selected by
test performance. `protocol.json` was fixed after development seed 97 and before
training final seed 0. The earlier failed waypoint interfaces are retained.

`study_final/` includes movement checkpoints, frozen-source snapshots, configs,
curves, all scene and episode records, and complete compressed trajectories for
normal, oracle and inverted-state arms. `audit_trajectories.py` independently
checks geometry and rewards and replays saved actor decisions. `report/` contains
the conclusions, tables and an actual MuJoCo trajectory replay.

In the original repository, the code is the same package under
`troy_dev_ppo_communication_gumbel/experiments/grammar_arena`. Run those commands
from `troy_dev_ppo_communication_gumbel`; provide explicit paths to protocol and
source checkpoints. On Windows the module uses the GLFW MuJoCo backend.
