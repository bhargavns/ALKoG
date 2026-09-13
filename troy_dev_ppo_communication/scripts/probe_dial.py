#!/usr/bin/env python3
"""Fast four-target communication probe using the production channel/PPO update.

No renderer, SAM inference, or pretrained weights. The sender sees a one-hot
class; the receiver sees only its message. Success is a correct receiver action.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from lib.DIALPPOTrainer import DIALPPOTrainer, evaluate_windows
from lib.ReceiverPolicy import ReceiverActorCritic


class ProbeTransmitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_head = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 2))
        nn.init.normal_(self.message_head[-1].weight, std=.01)
        nn.init.zeros_(self.message_head[-1].bias)

    def forward(self, obs, triples, deltas):
        return self.message_head(obs)


@torch.no_grad()
def accuracy(tx, rx, channel, device, noisy=False):
    targets = torch.arange(4, device=device).repeat(256)
    obs = torch.eye(4, device=device)[targets]
    logits = tx(obs, None, None)
    bits = torch.zeros(len(targets), 10, 2, device=device)
    valid = torch.zeros(len(targets), 10, dtype=torch.bool, device=device)
    bits[:, 0] = channel(logits, hard=not noisy)
    valid[:, 0] = True
    dist, _ = rx.distribution_bits(bits, valid)
    return float((dist.probs.argmax(-1) == targets).float().mean())


def run(seed, args):
    torch.manual_seed(seed)
    tx, rx = ProbeTransmitter(), ReceiverActorCritic()
    trainer = DIALPPOTrainer(tx, rx, None, lambda: None, device=args.device,
        horizon=args.batch_size, minibatch_size=args.batch_size, epochs=4,
        lr=args.lr, sigma=args.sigma, target_kl=.03, detach_channel=args.detach)
    last = {}
    for iteration in range(1, args.iterations+1):
        targets = torch.randint(4, (args.batch_size,), device=args.device)
        records = {
            'obs': torch.eye(4, device=args.device)[targets],
            'triples': torch.zeros(args.batch_size, 1, 3, dtype=torch.long, device=args.device),
            'deltas': torch.zeros(args.batch_size, 1, 2, 2, device=args.device),
            'noise': torch.randn(args.batch_size, 2, device=args.device),
        }
        windows = torch.full((args.batch_size, 10), -1, dtype=torch.long, device=args.device)
        windows[:, 0] = torch.arange(args.batch_size, device=args.device)
        with torch.no_grad():
            dist, value, _, _ = evaluate_windows(tx, rx, trainer.channel, records, windows)
            actions = dist.sample()
            rewards = (actions == targets).float()
            advantages = rewards-value
            advantages = (advantages-advantages.mean()) / advantages.std().clamp_min(1e-8)
            batch = dict(records=records, windows=windows, actions=actions,
                         logp=dist.log_prob(actions), values=value, returns=rewards,
                         advantages=advantages)
        last = trainer.update(batch)
        if iteration % 50 == 0:
            print(f'seed={seed} iteration={iteration} reward={float(rewards.mean()):.3f}', flush=True)
    # Evaluation draws do not affect training because training is finished.
    return {'seed': seed, 'hard_accuracy': accuracy(tx, rx, trainer.channel, args.device),
            'continuous_accuracy': accuracy(tx, rx, trainer.channel, args.device, True),
            'last_actor_gradient': last['tx_actor_grad_norm']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--sigma', type=float, default=2.)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.batch_size < 2:
        parser.error('iterations must be positive and batch-size >= 2')
    torch.set_num_threads(1)
    rows = [run(seed, args) for seed in args.seeds]
    result = {'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'results': rows, 'passed_95_percent_gate': all(r['hard_accuracy'] >= .95 for r in rows)}
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as handle:
            json.dump(result, handle, indent=2)
    if not args.detach and not result['passed_95_percent_gate']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
