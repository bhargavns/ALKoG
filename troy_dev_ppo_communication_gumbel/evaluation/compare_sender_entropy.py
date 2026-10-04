#!/usr/bin/env python3
"""Four symbols: no bonus vs constant/annealed sender entropy, seeds 0-9."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.probe_gumbel_direct import run, sender_entropy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference', type=Path, help='Optional four-symbol direct-loss JSON to reproduce exactly')
    a=p.parse_args()
    if a.out.exists():p.error('Output directory exists')
    a.out.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    hashes={}
    for name in ['evaluation/compare_sender_entropy.py','evaluation/probe_gumbel_direct.py',
                 'communication/CommunicationChannel.py','communication/ReceiverPolicy.py','communication/SenderEntropy.py','representation/SceneVocabulary.py']:
        target=a.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target);hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    arms=[('baseline',0.,'constant')]+[(f'{schedule}_{beta}',beta,schedule) for beta in [.01,.05,.1] for schedule in ['constant','annealed']]
    settings=dict(seeds=list(range(10)),updates=2000,n_symbols=4,temperature=1.2,
        lr=.003,batch_size=128,gradient_clip=.5,arms=arms,annealing='hold through 500; linear decay to zero at 1500; zero through 2000',
        scope='Fresh training, no checkpoint rescue or diagnostic pretraining transfer; no arena run',source_sha256=hashes,torch_version=torch.__version__)
    (a.out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    prior={x['seed']:x for x in json.loads(a.reference.read_text())['results']} if a.reference else None
    torch.set_num_threads(1)
    results=[]
    for arm,beta,schedule in arms:
        for seed in settings['seeds']:
            captured=[]
            def capture(tx,rx,channel):captured.extend((tx,rx,channel))
            row=run(seed,on_trained=capture,entropy_beta=beta,entropy_schedule=schedule)
            if prior is not None and arm=='baseline' and seed in prior and row!=prior[seed]:raise RuntimeError('Baseline failed to reproduce')
            tx,rx,channel=captured
            with torch.no_grad():
                logits=tx(torch.eye(4));probs=logits.softmax(-1)
                decode=rx.actor(rx.token_embeddings.weight).log_softmax(-1)
                exact_loss=(probs*(-decode.T)).sum(-1).mean()
                row.update(final_sender_entropy=float(sender_entropy(logits)),sender_probabilities=probs.tolist(),
                           receiver_probabilities_by_symbol=decode.exp().tolist(),exact_expected_cross_entropy=float(exact_loss))
            row.update(arm=arm,beta=beta,schedule=schedule,passed=min(row['hard_accuracy'],row['sampled_accuracy'])>=.95)
            torch.save(dict(transmitter=tx.state_dict(),receiver=rx.state_dict(),n_symbols=4,temperature=1.2),a.out/f'{arm}_seed{seed}.pt')
            results.append(row)
            (a.out/'results.json').write_text(json.dumps(dict(settings=settings,results=results),indent=2)+'\n')
        subset=[x for x in results if x['arm']==arm]
        print(json.dumps(dict(arm=arm,passed=sum(x['passed'] for x in subset),
            hard=[x['hard_accuracy'] for x in subset],sampled=[x['sampled_accuracy'] for x in subset],
            mean_entropy=sum(x['final_sender_entropy'] for x in subset)/10)),flush=True)
    lines=['# Sender entropy bonus comparison','',
        'Four symbols, seeds 0–9, 2000 updates each. All settings unchanged except sender entropy bonus. Each condition starts fresh with the same initialization/random stream per seed. Betas and schedules were specified before running.','',
        '| Schedule | Starting beta | Seeds passing both >=95% | Mean greedy accuracy | Mean sampled accuracy | Final mean sender entropy | Failed seeds |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arm,beta,schedule in arms:
        rs=[x for x in results if x['arm']==arm]
        lines.append(f"| {schedule if beta else 'none'} | {beta} | {sum(x['passed'] for x in rs)}/10 | {sum(x['hard_accuracy'] for x in rs)*10:.3f}% | {sum(x['sampled_accuracy'] for x in rs)*10:.3f}% | {sum(x['final_sender_entropy'] for x in rs)/10:.6f} | {[x['seed'] for x in rs if not x['passed']]} |")
    lines+=['','Annealed schedule holds beta through update 500, decays linearly to zero at 1500, and remains zero through 2000. The bonus is conditional sender entropy from categorical logits, not receiver action entropy or marginal symbol-use balance. Reported accuracies use greedy receiver predictions. These are exploratory hyperparameter comparisons on ten seeds, not independent confirmation, checkpoint rescue, or navigation evidence. Full learning curves, probabilities and model weights are preserved.']
    (a.out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
