#!/usr/bin/env python3
"""Held-out communication evaluation; uses a separate process/environment.

Example: python3 scripts/evaluate_communication.py --run-dir output/runs/comm_...
  --episodes 500 --modes hard constant random shuffle --device cuda
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lib.CategoryModel import load_category_checkpoint
from lib.CommunicationEvaluation import CommunicationActor, evaluate_episode, summarize
from lib.CommunicationTrainer import TransmitterInput
from lib.KGWorldEnv import CommunicationKGWorldEnv
from lib.OracleGrounding import oracle_bodies, perceive_scene_softcat
from lib.Perception import PerceptionPipeline
from lib.ReceiverPolicy import ReceiverActorCritic
from lib.SymbolicKG import SymbolicKG
from lib.SymbolPolicy import DIALTransmitter, TripleActorCritic



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='final', help='final or an iteration number')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--scene-seed', type=int, default=200_000)
    parser.add_argument('--policy-seed', type=int, default=300_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modes', nargs='+', default=['hard', 'constant', 'random'],
                        choices=['hard', 'continuous', 'constant', 'random', 'shuffle'])
    parser.add_argument('--greedy', action='store_true', help='greedy receiver movement; otherwise sampled')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or ('shuffle' in args.modes and args.episodes < 2):
        parser.error('positive episodes required; shuffle requires at least two')
    with (args.run_dir/'config.json').open() as handle:
        config = json.load(handle)
    if config.get('format_version') != 1 or config.get('context_length') != 10:
        parser.error('Unsupported checkpoint configuration version or context length')
    method = config['method']
    if method not in ('independent_ppo', 'dial_ppo'):
        parser.error('Unsupported checkpoint method')
    if method == 'independent_ppo' and 'continuous' in args.modes:
        parser.error('continuous mode requires a DIAL checkpoint')
    out = args.out or args.run_dir/f'evaluation_{args.checkpoint}_{args.scene_seed}.json'
    if out.exists():
        parser.error(f'Output already exists: {out}; choose a distinct --out path')
    device = args.device
    receiver_yaw_mode = config.get('receiver_yaw_mode', 'independent')
    env = CommunicationKGWorldEnv(resources=config['resources'] == '1',
        receiver_yaw_mode=receiver_yaw_mode,
        food_near_lion_prob=float(config['food_near_lion']),
        food_near_lion_offset=float(config['food_near_lion_offset']))
    try:
        kg = SymbolicKG.load(args.run_dir/'kg_init.pt', device=device)
        bodies = oracle_bodies(env)
        if kg.num_nodes != len(bodies):
            raise ValueError('Checkpoint KG does not match environment body vocabulary')
        body_to_node = {body: i for i, body in enumerate(bodies)}
        category, _ = load_category_checkpoint(config['category_model'], device)
        category.requires_grad_(False)
        pipeline = PerceptionPipeline(device=device)
        def make_input():
            return TransmitterInput(kg, lambda: perceive_scene_softcat(
                env, pipeline, kg, category, body_to_node), body_to_node['receiver'],
                int(config['k_triples']))
        cls = DIALTransmitter if method == 'dial_ppo' else TripleActorCritic
        tx = cls(kg, k_triples=int(config['k_triples'])+1).to(device)
        rx = ReceiverActorCritic().to(device)
        if args.checkpoint == 'final':
            tx_path, rx_path = args.run_dir/'transmitter_final.pt', args.run_dir/'receiver_final.pt'
        else:
            prefix = args.run_dir/'checkpoints'/f'iter{int(args.checkpoint):04d}'
            tx_path, rx_path = Path(str(prefix)+'_transmitter.pt'), Path(str(prefix)+'_receiver.pt')
        tx.load_state_dict(torch.load(tx_path, map_location=device, weights_only=True))
        rx.load_state_dict(torch.load(rx_path, map_location=device, weights_only=True))
        tx.eval()
        rx.eval()
        modes = list(dict.fromkeys(args.modes))
        if 'shuffle' in modes:
            modes = ['hard'] + [m for m in modes if m != 'hard']
        all_rows, summaries = {}, {}
        for mode in modes:
            rows = []
            for i in range(args.episodes):
                torch.manual_seed(args.policy_seed+i)
                donor = (all_rows['hard'][(i+1) % args.episodes]['messages']
                         if mode == 'shuffle' else None)
                actor = CommunicationActor(tx, rx, device, method,
                    float(config['channel_sigma']), mode, args.greedy, donor)
                row = evaluate_episode(env, make_input, actor, int(config['perceive_passes']),
                                       args.scene_seed+i)
                rows.append(row)
                if (i+1) % 25 == 0:
                    print(f'{mode}: {i+1}/{args.episodes}', flush=True)
            all_rows[mode] = rows
            summaries[mode] = summarize(rows)
            print(mode, json.dumps(summaries[mode]), flush=True)
        payload = {'format_version': 1, 'training_seed': config['seed'], 'method': method,
            'receiver_yaw_mode': receiver_yaw_mode,
            'checkpoint': str(tx_path), 'scene_seed': args.scene_seed,
            'policy_seed': args.policy_seed, 'greedy_receiver': args.greedy,
            'ci_scope': 'Wilson rate / episode-bootstrap mean intervals within this training seed; not across seeds',
            'shuffle_policy': 'next hard episode as donor, hold last symbol when donor ends',
            'summaries': summaries, 'episodes': all_rows}
        out.parent.mkdir(parents=True, exist_ok=True)
        # Avoid silently replacing a previous held-out evaluation.
        with out.open('x') as handle:
            json.dump(payload, handle, indent=2)
        print(f'Saved {out}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
