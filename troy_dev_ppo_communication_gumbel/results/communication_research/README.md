# Communication research baseline

These reports preserve the optional-communication study, four split-information
tasks, neural fixed-slot study and validated MuJoCo transfer. All failed methods
and the distinction between raw frozen and training-admitted results are retained
in the reports. MuJoCo uses exact state, supplied public routes and kinematic
movement, not visual perception or torque locomotion. New grammar methods are
not part of this baseline.

Small inference checkpoints are under experiments/grammar_arena/checkpoints.
They contain the exact original model tensors. The manifest records both the
original full-checkpoint SHA-256 and a hash of the model tensors; optimizer and
RNG state are omitted to keep Git small. Full checkpoints, raw scene/trajectory
records and complete downloadable archives remain in the original research
output workspace and are not copied into Git.

To replay a validated pair, run from troy_dev_ppo_communication_gumbel:

```powershell
python -m experiments.grammar_arena.evaluate --protocol experiments/grammar_arena/checkpoints/admitted/context_modular_curriculum_0 --silent experiments/grammar_arena/checkpoints/original/context_silent_task_0 --navigation experiments/grammar_arena/checkpoints/navigation/seed0 --output output/validated_replay --seed 94000019 --repeats 32
```
