# Four communication options

This is a controlled finite-state study outside vision and MuJoCo. It tests:
private targets, changing safe passages, asking versus inspecting, and interactive
lookup of a private hazard map. It does not claim sampled neural PPO learning.
Policies are small categorical probability tables trained with exact expected
shared reward. A receiver-local BFS planner executes selected public waypoints;
movement is not jointly learned. All logical states occur in training.

The self-contained generated report explains observations, reward, policy
factorization, exact silent bounds, temporal dynamic programming, binary-entropy
bit bounds, warm-ups, and limitations. `protocol.json` contains the criteria
written before final optimization seeds 0–4. Development uses seed 97.

From `troy_dev_ppo_communication_gumbel`, using the installed environment:

```powershell
../.venv-event/Scripts/python.exe -m experiments.communication_options.run --output output/NEW_STUDY --steps 1500 --seeds 0 1 2 3 4
../.venv-event/Scripts/python.exe -m experiments.communication_options.interventions --study output/NEW_STUDY
../.venv-event/Scripts/python.exe -m experiments.communication_options.validate --study output/NEW_STUDY
../.venv-event/Scripts/python.exe -m experiments.communication_options.report --study output/NEW_STUDY --output output/NEW_REPORT
```

Use a new output directory; runs refuse to overwrite existing artifacts.
`requirements-lock.txt` captures installed versions. Needed packages are PyTorch,
NumPy, Matplotlib, Markdown and pytest. The reported study uses CPU and one
PyTorch thread; CUDA is optional, not required. GPU-capable Torch is already
installed in `.venv-event` on the user's machine.

## Reproduction notes

- `run` preserves manifest, source snapshot/hash, every final checkpoint,
  optimizer/RNG state, curve, normal score and diagnostic intervention rows.
- `interventions` is a required postprocessing step. Under channel corruption,
  it maintains separate sender-predicted and receiver-delivered caches, without
  acknowledgment. It checks that every normal score agrees with the reliable
  one-cache training dynamic program. It writes `validated_results.json`.
- `validate` independently samples one million episodes across 20 primary
  policies, and checks 128 new public obstacle layouts for the local planner.
- `report` reads validated results. Inapplicable intervention names are removed
  from the delivered CSV. The raw generic diagnostic records remain archived.
- Argmax deployment is primary. Exact sampled-policy expectations are secondary.
- The warm scouting method has 600 extra code-learning steps. The warm dialogue
  method has 1,000 extra reference-learning steps. They are not equal-total-budget
  comparisons with cold starts.
- A final reproducibility rerun corrected only default responses to unused words
  in fixed reference controls. Learned weights and normal results were unchanged.
  Earlier development and first-pass artifacts are retained separately.

## Tests

```powershell
$env:MUJOCO_GL='glfw'
../.venv-event/Scripts/python.exe -m pytest tests/experiments/test_communication_options.py -q
../.venv-event/Scripts/python.exe -m pytest -q
```

Tests check analytic bounds, exhaustive route-state enumeration, independent
unacknowledged packet-loss execution, temporal recursion, bit charging, gradients,
and public planner constraints. The broader suite checks existing ALKoG modules.

No remote branch is pushed. New code is isolated from the previous experiments.
