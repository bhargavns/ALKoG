# Unassigned neural symbols and emergent order

This research package branches from preserved baseline `8152d91`. It does not
change the existing environment, arena XML, fixed-role protocol or navigator.

The new grid tasks are **neural referential games**: an observer knows the target
scene, a collector sees four public scene descriptions, and an initial discrete
packet selects the rewarded destination. Movement on the 13 x 13 grid is
supplied; the new grid training is not locomotion RL. Food and animal conditions
are semantic scene properties here. Actual food/hazard rewards are tested in a
separate frozen transfer through the existing MuJoCo arena.

No word meanings, semantic word positions, English targets or fact-specific
listener heads are assigned. The listener compares a whole-message embedding
against whole candidate scenes. This removes the old forced lexical roles,
while retaining explicit perceptual/architectural biases described below.

## Methods and evidence

Final comparison: five seeds, two tasks, 6,000 updates x 256 episodes per model.
Methods: sampled PPO, recurrent Gumbel, curriculum, iterated learning, population
learning and a parallel unassigned-slot neural channel. A seventh method uses
training-only spatial role reversals and inventory invariance, on binding only.
Total: 65 final models. Pilot anchoring, reconstruction and attention experiments
are retained separately and are not pooled with final seeds.

The parallel/invariance channels have **eight fixed positions**. Every position
depends on the whole scene; none is labeled as a fact. They also use anchored
perception and whole-scene reconstruction. Inventory invariance supplies the
additional bias that changing object roles should preserve the symbol histogram
while remaining distinguishable by the listener. It receives extra auxiliary
examples. These are disclosed inductive biases, not unbiased emergence.

The report separates useful communication, unseen combinations, causal
order-sensitive role binding, local symbolic edits, new object cardinality and
embodied task success. A generic sorting drop is not called syntax. The stronger
probe requires two correct role-swapped meanings with exactly the same symbol
multiset and different order. No human-language claim is made.

## Run

From the project folder with Python 3.11 and the experiment dependencies:

```powershell
$env:MUJOCO_GL = 'glfw'
python -m experiments.spontaneous_grammar.study --protocol PATH/protocol.json --output PATH/new_study --workers 6
python -m experiments.spontaneous_grammar.arena_transfer --study PATH/new_study --output PATH/new_transfer
python -m experiments.spontaneous_grammar.audit --study PATH/new_study --protocol PATH/protocol.json --transfer PATH/new_transfer --output PATH/new_audit
python -m pytest tests -q -p no:cacheprovider
```

The study verifies frozen source hashes before and after all runs, evaluates the
last checkpoint only, and retains every seed. Output directories must be new.
The baseline navigation checkpoints under `grammar_arena/checkpoints/navigation`
are needed for the optional MuJoCo transfer. This uses CUDA for frozen motor
inference and does not retrain the motor controller.

To inspect one final model:

```powershell
python -m experiments.spontaneous_grammar.evaluate --checkpoint PATH/checkpoint.pt --cache PATH/attributes-shift1.pt --output PATH/new_evaluation
```

## Source map

- `task.py`: meanings, split rules, target-balanced rectangles, three-object test,
  and supplied cardinal paths.
- `models.py`: scene encoders, recurrent/parallel messages, whole-scene listeners,
  critic, and message interventions.
- `train.py`: joint training methods and complete final checkpoints.
- `evaluate.py`: paired channel controls, codebooks, training-only rewrite probe,
  role-binding contrasts, and fresh-listener learning.
- `study.py`: frozen 65-run comparison and failure retention.
- `arena_transfer.py`: public 16-hypothesis retrieval and actual MuJoCo execution.
- `audit.py`: source/checkpoint checks and independent retained-codebook replay.

Read `results/spontaneous_grammar/report.md` or the accompanying PDF for results,
mathematics, failures and limitations. Checkpoints and raw runs are also provided
in the downloadable study archive. Inference copies omit optimizer/RNG state;
full training checkpoints retain them.
