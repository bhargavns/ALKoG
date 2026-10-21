"""Run the first MuJoCo transfer only after the functional Experiment 1 gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .audit import load_pair
from .run import source_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, default=Path('output/entity_safety3_acceptance/acceptance.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=512)
    parser.add_argument('--evaluation-seed', type=int, default=4001)
    parser.add_argument('--layouts-per-case', type=int, default=100)
    args = parser.parse_args()
    if min(args.iterations, args.horizon, args.layouts_per_case) < 1:
        parser.error('Positive budgets required')
    acceptance = json.loads(args.acceptance.read_text())
    if not acceptance.get('accepted') or set(acceptance['runs']) != set(map(str, range(5))):
        parser.error('Five accepted frozen communication checkpoints are required')
    checkpoints = []
    for seed in range(5):
        row = acceptance['runs'][str(seed)]
        path = Path(row['checkpoint'])
        if (not row['accepted'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['checkpoint_sha256']):
            parser.error('Accepted communication checkpoint has changed')
        checkpoints.append(path)
    # No MuJoCo import/scene creation occurs before the gate is validated.
    from .navigation import train_movement, evaluate_navigation
    import mujoco
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_metadata(args.output)
    source['method'] = 'frozen_accepted_safety_code_plus_temporal_movement_PPO'
    source['mujoco_version'] = str(mujoco.__version__)
    write_json(args.output / 'source.json', source)
    write_json(args.output / 'config.json', dict(
        acceptance=str(args.acceptance.resolve()),
        acceptance_sha256=hashlib.sha256(args.acceptance.read_bytes()).hexdigest(),
        movement_iterations=args.iterations, horizon=args.horizon, seed_count=5,
        evaluation_seed=args.evaluation_seed, layouts_per_case=args.layouts_per_case,
        movement='kinematic MuJoCo position updates; no torque locomotion',
        task='public marked goal, private oracle cage/safety evidence, clear-path layouts',
        movement_training='safe-goal episodes, PPO with discounted returns and GAE',
        communication_training='none; accepted weights remain frozen',
        movement_observation='public goal offset divided by arena diameter; no cage/safety/category labels',
        deployment='frozen message response gates WARN termination or learned movement policy'))
    results = {}
    for seed, checkpoint in enumerate(checkpoints):
        sender, receiver, _ = load_pair(checkpoint)
        before_sender = {k: v.clone() for k, v in sender.state_dict().items()}
        before_receiver = {k: v.clone() for k, v in receiver.state_dict().items()}
        start = time.perf_counter()
        movement, curve = train_movement(seed, args.iterations, args.horizon)
        training_seconds = time.perf_counter() - start
        result, records = evaluate_navigation(sender, receiver, movement, args.evaluation_seed,
                                              args.layouts_per_case)
        assert all(torch.equal(v, before_sender[k]) for k, v in sender.state_dict().items())
        assert all(torch.equal(v, before_receiver[k]) for k, v in receiver.state_dict().items())
        result.update(training_seconds=training_seconds, communication_weights_unchanged=True)
        results[str(seed)] = result
        run = args.output / f'seed{seed}'
        run.mkdir()
        torch.save(dict(movement=movement.state_dict(), movement_seed=seed,
                        frozen_communication_checkpoint=str(checkpoint.resolve()),
                        frozen_communication_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        source=source, torch_rng_state=torch.get_rng_state()), run / 'checkpoint.pt')
        (run / 'learning_curve.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in curve))
        (run / 'evaluation_trajectories.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        write_json(run / 'metrics.json', result)
        print(f'seed={seed} success={result["success_rate"]:.3f} safety_errors={result["safety_response_errors"]} '
              f'caught={result["catches"]} timeout={result["timeouts"]}', flush=True)
    write_json(args.output / 'metrics.json', dict(runs=results,
               note='First clear-path oracle transfer; not perception, language composition or dynamic locomotion'))


if __name__ == '__main__':
    main()
