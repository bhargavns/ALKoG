#!/usr/bin/env python3
"""Exhaustively intervene on each symbol of a trained current-message receiver."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic

ACTIONS = ('forward', 'backward', 'left', 'right')

@torch.no_grad()
def probe(run):
    config = json.loads((run/'config.json').read_text())
    if config['receiver_arch'] != 'current_mlp':
        raise ValueError('This exhaustive state-independent probe requires current_mlp')
    n = int(config['n_symbols'])
    checkpoint = run/'receiver_final.pt'
    rx = CurrentMessageReceiverActorCritic(n_symbols=n)
    rx.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    rx.eval()
    messages = torch.eye(n)
    dist, _ = rx.distribution_messages(messages)
    # Independent embedding lookup agrees with the actual channel interface.
    direct = rx.actor(rx.token_embeddings(torch.arange(n))).softmax(-1)
    torch.testing.assert_close(dist.probs, direct)
    torch.manual_seed(42000)
    samples = dist.sample((10000,))
    training = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()][-100:]
    rows = []
    for i in range(n):
        probs = dist.probs[i].tolist()
        empirical = torch.bincount(samples[:, i], minlength=4).float()/10000
        # Wide reproducible sampling check; exact probabilities are the primary result.
        assert (empirical - dist.probs[i]).abs().max() < .03
        rows.append({'symbol': f's{i}', 'probabilities': dict(zip(ACTIONS, probs)),
                     'most_likely': ACTIONS[dist.probs[i].argmax().item()],
                     'sampled_frequencies': dict(zip(ACTIONS, empirical.tolist())),
                     'last100_training_usage': sum(r['symbol_fracs'][i] for r in training)/len(training)})
    return {'run_dir': str(run), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'checks': ['channel vs direct embedding', '10000 sampled actions per symbol'],
            'rows': rows}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    results = [probe(run) for run in sorted((args.experiment/'runs').glob('comm_*'))]
    lines = ['# Receiver symbol interventions', '',
             'Final checkpoints; exact movement probabilities for each forced symbol. Directions are relative to receiver yaw. Actions are sampled, so a symbol does not guarantee a movement. This tests receiver decoding, not which scenes cause the sender to choose a symbol.', '']
    for result in results:
        lines += [f'## {Path(result["run_dir"]).name}', '', '| Symbol | Forward | Backward | Left | Right | Most likely | Training usage (last 100 iterations) |', '|---|---:|---:|---:|---:|---|---:|']
        for row in result['rows']:
            values = ' | '.join(f'{100*v:.2f}%' for v in row['probabilities'].values())
            lines.append(f'| {row["symbol"]} | {values} | {row["most_likely"]} | {100*row["last100_training_usage"]:.2f}% |')
        lines += ['', 'Passed: direct embedding equivalence and 10,000 sampled actions per symbol.', '']
    (args.experiment/'receiver_symbol_probe.json').write_text(json.dumps(results, indent=2)+'\n')
    (args.experiment/'receiver_symbol_probe.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
