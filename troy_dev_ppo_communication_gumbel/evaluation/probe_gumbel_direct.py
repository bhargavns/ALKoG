#!/usr/bin/env python3
"""Direct-loss diagnostic: learn a discrete-symbol code with no prescribed meanings.

Balanced four-direction labels go only to the sender and the loss. The receiver
sees one discrete symbol. This is a mechanism diagnostic, not arena pretraining.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from communication.CommunicationChannel import StraightThroughGumbelChannel
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic


from communication.SenderEntropy import entropy_coefficient, sender_entropy


def run(seed, steps=2000, temperature=1.2, detach=False, on_trained=None, n_symbols=4,
        entropy_beta=0., entropy_schedule='constant'):
    entropy_coefficient(1, steps, entropy_beta, entropy_schedule)
    if detach and entropy_beta:
        raise ValueError('Detached diagnostic must not train sender through an entropy loss')
    torch.manual_seed(seed)
    tx = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, n_symbols))
    rx = CurrentMessageReceiverActorCritic(n_symbols=n_symbols)
    channel = StraightThroughGumbelChannel(temperature, n_symbols=n_symbols)
    optimizer = torch.optim.Adam(list(tx.parameters()) + list(rx.parameters()), lr=.003)
    labels = torch.arange(4).repeat(32)
    obs = F.one_hot(labels, 4).float()
    initial = {k: v.clone() for k, v in tx.state_dict().items()}
    losses = []
    for step in range(steps):
        sender_logits = tx(obs)
        messages = channel(sender_logits)
        if detach:
            messages = messages.detach()
        dist, _ = rx.distribution_messages(messages)
        prediction_loss = F.nll_loss(dist.logits, labels)
        beta = entropy_coefficient(step+1, steps, entropy_beta, entropy_schedule)
        loss = prediction_loss
        if entropy_beta:
            entropy = sender_entropy(sender_logits)
            loss = loss - beta * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(tx.parameters()) + list(rx.parameters()), .5)
        optimizer.step()
        if (step+1) % 200 == 0:
            row = {'step': step+1, 'cross_entropy': float(prediction_loss.detach())}
            if entropy_beta:
                row.update(entropy_beta=beta, sender_entropy=float(entropy.detach()), total_loss=float(loss.detach()))
            losses.append(row)
    if on_trained is not None:
        on_trained(tx, rx, channel)
    # Separate draws for evaluation; report both greedy and sampled sender modes.
    torch.manual_seed(100000 + seed)
    targets = torch.arange(4).repeat(1024)
    with torch.no_grad():
        logits = tx(F.one_hot(targets, 4).float())
        result = {}
        for mode in ('hard', 'sampled'):
            messages = channel(logits, hard=mode == 'hard')
            dist, _ = rx.distribution_messages(messages)
            result[mode+'_accuracy'] = float((dist.probs.argmax(-1) == targets).float().mean())
            result[mode+'_target_probability'] = float(dist.probs.gather(1, targets[:, None]).mean())
        result['symbol_by_target'] = tx(torch.eye(4)).argmax(-1).tolist()
        result['tx_weight_change'] = sum(float((v-initial[k]).square().sum()) for k,v in tx.state_dict().items())**.5
    return dict(seed=seed, detach=detach, **result, learning_curve=losses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--temperature', type=float, default=1.2)
    p.add_argument('--symbols', type=int, default=4)
    p.add_argument('--entropy-beta', type=float, default=0.)
    p.add_argument('--entropy-schedule', choices=('constant', 'annealed'), default='constant')
    p.add_argument('--detach', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.iterations < 1 or len(set(a.seeds)) != len(a.seeds):
        p.error('positive iterations and distinct seeds required')
    if a.out.exists():
        p.error('Output exists; choose a new path')
    torch.set_num_threads(1)
    rows=[]
    for seed in a.seeds:
        result=run(seed, a.iterations, a.temperature, a.detach, n_symbols=a.symbols, entropy_beta=a.entropy_beta, entropy_schedule=a.entropy_schedule)
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='learning_curve'}), flush=True)
    passed = len(rows)>=3 and all(min(x['hard_accuracy'], x['sampled_accuracy'])>=.95 for x in rows)
    payload=dict(diagnostic='direct_cross_entropy_four_directions', receiver_arch='current_mlp',
                 temperature=a.temperature, iterations=a.iterations, detached=a.detach, n_symbols=a.symbols,
                 entropy_beta=a.entropy_beta, entropy_schedule=a.entropy_schedule,
                 seeds=a.seeds, results=rows, passed_95_percent_gate=passed,
                 scope='Mechanism diagnostic; four classes, not held-out navigation; no weights transferred to arena.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f: json.dump(payload, f, indent=2)
    print('PASS' if passed else 'FAIL', flush=True)
    return 0 if passed else 2

if __name__ == '__main__':
    raise SystemExit(main())
