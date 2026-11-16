# Neural food and hazard communication

This module adds sampled neural PPO communication experiments without modifying
the existing MuJoCo experiments. The collector learns destination and route
choices; BFS navigation and primitive observations are supplied. The strongest
compositional reference supplies attribute-to-position roles and learns word
meanings. See the final report for the complete limits and failed methods.

Run from `troy_dev_ppo_communication_gumbel` with the experiment environment:

```powershell
python -m experiments.grammar_food.study --output output/grammar_reproduction --workers 4
python -m experiments.grammar_food.analyze --study output/grammar_reproduction --output output/grammar_summary
```

The sweep creates a NEW output directory and trains all nine methods, two splits
and five seeds using the budget in protocol.json. For one model:

```powershell
python -m experiments.grammar_food.train --method modular_curriculum --seed 0 --split context --shift 0 --steps 1800 --batch-size 512 --warmup 600 --slot-coefficient 5 --device cpu --output output/grammar_one
python -m experiments.grammar_food.evaluate --run output/grammar_one
python -m experiments.grammar_food.export_records --run output/grammar_one
```

`--device cuda` is supported for individual training runs. The final sweep used
CPU because the small networks do not benefit clearly from GPU batching at this
scale. The four-word channel and flat-word channel both have eight payload bits.
The fixed sweep runner sets the FINAL parameters; the individual CLI defaults
are intended for development.

Optional-tool and new-map execution are separate frozen-protocol modules:

```powershell
python -m experiments.grammar_food.optional_tool --run output/grammar_one --silent PATH_TO_SAME_SEED_SILENT_MODEL --output output/grammar_gate
python -m experiments.grammar_food.transfer --run output/grammar_one --layouts PATH_TO_MAZE_LAYOUTS_JSON --output output/grammar_maps
```

The downloadable standalone bundle includes all 128 layouts, runners, final and
development checkpoints, complete results, per-episode records and report. In
that copy the package namespace is `grammar_food` instead of
`experiments.grammar_food`. No low-level maze exploration, camera perception,
MuJoCo training, morphology or recursive grammar is claimed by this module.

Windows project test command (choose a new writable basetemp directory):

```powershell
$env:MUJOCO_GL = 'glfw'
python -m pytest -q -p no:cacheprovider --basetemp PATH_TO_NEW_WRITABLE_TEST_DIRECTORY
```
