#!/usr/bin/env python3
"""Held-out communication evaluation; uses a separate process/environment.

Example: python3 evaluation/evaluate_communication.py --run-dir output/runs/comm_...
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

from perception.CategoryModel import load_category_checkpoint
from evaluation.CommunicationEvaluation import CommunicationActor, evaluate_episode, summarize
from representation.TransmitterInput import TransmitterInput
from representation.OracleTransmitterInput import OracleTransmitterInput
from environment.KGWorldEnv import CommunicationKGWorldEnv
from representation.SceneVocabulary import scene_bodies
from perception.ScenePerception import perceive_scene_softcat
from perception.Perception import PerceptionPipeline
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic
from representation.SceneVocabulary import SceneVocabulary
from communication.SymbolPolicy import GumbelTransmitter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='final', help='final or an iteration number')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--scene-seed', type=int, default=200_000)
    parser.add_argument('--policy-seed', type=int, default=300_000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modes', nargs='+', default=['hard', 'constant', 'random'],
                        choices=['hard', 'sampled', 'constant', 'random', 'shuffle'])
    parser.add_argument('--greedy', action='store_true', help='greedy receiver movement; otherwise sampled')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--category-model', type=Path, help='Override the category checkpoint path for relocated visual runs')
    parser.add_argument('--video-dir', type=Path, help='Record every evaluated episode as an MP4')
    args = parser.parse_args()
    if args.episodes < 1 or ('shuffle' in args.modes and args.episodes < 2):
        parser.error('positive episodes required; shuffle requires at least two')
    with (args.run_dir/'config.json').open() as handle:
        config = json.load(handle)
    if config.get('format_version') not in (1, 2):
        parser.error('Unsupported checkpoint configuration version')
    method = config['method']
    if method != 'st_gumbel_ppo' or config.get('receiver_arch') != 'current_mlp':
        parser.error('A Gumbel current-message checkpoint is required')
    out = args.out or args.run_dir/f'evaluation_{args.checkpoint}_{args.scene_seed}.json'
    if out.exists():
        parser.error(f'Output already exists: {out}; choose a distinct --out path')
    device = args.device
    colocated = str(config.get('colocated', '0')) == '1'
    receiver_yaw_mode = config.get('receiver_yaw_mode', 'independent')
    env = CommunicationKGWorldEnv(resources=config['resources'] == '1',
        receiver_yaw_mode=receiver_yaw_mode, colocated=colocated,
        food_near_lion_prob=float(config['food_near_lion']),
        food_near_lion_offset=float(config['food_near_lion_offset']))
    try:
        vocabulary_path = args.run_dir / 'scene_vocabulary_init.pt'
        if not vocabulary_path.exists():
            # Earlier runs stored the same tensor dictionary under this name.
            vocabulary_path = args.run_dir / 'kg_init.pt'
        vocabulary = SceneVocabulary.load(vocabulary_path, device=device)
        bodies = scene_bodies(env)
        if vocabulary.num_objects != len(bodies):
            raise ValueError('Checkpoint scene vocabulary does not match environment bodies')
        body_to_node = {body: i for i, body in enumerate(bodies)}
        perception = config.get('perception', 'softcat')
        if perception == 'softcat':
            category, _ = load_category_checkpoint(args.category_model or config['category_model'], device)
            category.requires_grad_(False)
            pipeline = PerceptionPipeline(device=device)
        elif perception != 'oracle_state':
            raise ValueError(f'Unsupported perception mode: {perception}')
        def make_input():
            if perception == 'oracle_state':
                return OracleTransmitterInput(vocabulary, env, body_to_node, int(config['k_triples']))
            return TransmitterInput(vocabulary, lambda: perceive_scene_softcat(
                env, pipeline, vocabulary, category, body_to_node), body_to_node['receiver'],
                int(config['k_triples']))
        n_symbols = int(config['n_symbols'])
        tx = GumbelTransmitter(vocabulary, k_triples=int(config['k_triples'])+1, n_symbols=n_symbols).to(device)
        rx = CurrentMessageReceiverActorCritic(n_symbols=n_symbols).to(device)
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
                actor = CommunicationActor(tx, rx, device, mode, args.greedy, donor)
                recorder = None
                if args.video_dir:
                    from observability.VideoRecorder import PanoramaRecorder
                    args.video_dir.mkdir(parents=True, exist_ok=True)
                    video = args.video_dir/f'{mode}_scene{args.scene_seed+i}.mp4'
                    if video.exists():
                        raise FileExistsError(video)
                    recorder = PanoramaRecorder(str(video), frame_size=env.render_size, fps=6)
                    if not recorder.writer.isOpened():
                        raise RuntimeError(f'Cannot open video writer: {video}')
                try:
                    row = evaluate_episode(env, make_input, actor, int(config['perceive_passes']),
                                           args.scene_seed+i, recorder=recorder)
                finally:
                    if recorder is not None:
                        recorder.close()
                rows.append(row)
                if (i+1) % 25 == 0:
                    print(f'{mode}: {i+1}/{args.episodes}', flush=True)
            all_rows[mode] = rows
            summaries[mode] = summarize(rows)
            print(mode, json.dumps(summaries[mode]), flush=True)
        payload = {'format_version': 1, 'training_seed': config['seed'], 'method': method,
            'perception': perception,
            'receiver_arch': config['receiver_arch'],
            'receiver_yaw_mode': receiver_yaw_mode, 'colocated': colocated,
            'perceive_every': 75 if colocated else 0,
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
