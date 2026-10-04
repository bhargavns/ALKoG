#!/usr/bin/env python3
"""Fixed-budget direct-loss vocabulary comparison; no arena training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.probe_gumbel_direct import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference', type=Path, help='Optional four-symbol direct-loss JSON to reproduce exactly')
    args=p.parse_args()
    if args.out.exists():p.error('Output directory already exists')
    args.out.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    source_files=['evaluation/probe_gumbel_direct.py','evaluation/compare_gumbel_vocabularies.py',
                  'communication/CommunicationChannel.py','communication/ReceiverPolicy.py','communication/SenderEntropy.py','representation/SceneVocabulary.py']
    hashes={}
    for name in source_files:
        target=args.out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    settings=dict(vocabularies=[4,8,16],seeds=[0,1,2,3,4],updates=2000,
        batch_size=128,examples_per_class_per_batch=32,temperature=1.2,lr=.003,
        optimizer='Adam defaults',gradient_clip_norm=.5,sender_hidden=32,
        receiver_embedding=4,receiver_hidden=4,receiver_actions=4,
        evaluation_examples=4096,torch_version=torch.__version__,source_sha256=hashes,
        initialization='Same seed and initialization rules; dimension changes alter RNG consumption, so weights/draws are not identical across vocabularies.')
    (args.out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    torch.set_num_threads(1)
    original={x['seed']:x for x in json.loads(args.reference.read_text())['results']} if args.reference else None
    results=[]
    for symbols in settings['vocabularies']:
        for seed in settings['seeds']:
            captured=[]
            def capture(tx,rx,c):captured.extend((tx,rx,c))
            result=run(seed,on_trained=capture,n_symbols=symbols)
            if original is not None and symbols==4 and result!=original[seed]:raise RuntimeError(f'Four-symbol baseline changed: seed {seed}')
            tx,rx,c=captured
            with torch.no_grad():
                result['sender_probabilities']=tx(torch.eye(4)).softmax(-1).tolist()
                result['receiver_probabilities_by_symbol']=rx.actor(rx.token_embeddings.weight).softmax(-1).tolist()
            result.update(n_symbols=symbols,passed=min(result['hard_accuracy'],result['sampled_accuracy'])>=.95,
                          sender_parameters=sum(p.numel() for p in tx.parameters()),
                          receiver_parameters=sum(p.numel() for p in rx.parameters()))
            torch.save(dict(transmitter=tx.state_dict(),receiver=rx.state_dict(),n_symbols=symbols,temperature=c.temperature),args.out/f'v{symbols}_seed{seed}.pt')
            results.append(result)
            (args.out/'results.json').write_text(json.dumps(dict(settings=settings,results=results),indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ['learning_curve','sender_probabilities','receiver_probabilities_by_symbol']}),flush=True)
    lines=['# Vocabulary-size diagnostic comparison','',
           'Five seeds per vocabulary; 2000 direct-loss updates per seed. Only vocabulary size and required sender output/receiver embedding-table dimensions change.','',
           '| Vocabulary | Hard accuracy, seeds 0–4 | Mean hard accuracy | Mean sampled accuracy | Seeds passing both >=95% |',
           '|---|---|---:|---:|---:|']
    for symbols in settings['vocabularies']:
        rows=[x for x in results if x['n_symbols']==symbols]
        lines.append(f"| {symbols} | "+', '.join(f"{100*x['hard_accuracy']:.0f}%" for x in rows)+f" | {100*sum(x['hard_accuracy'] for x in rows)/5:.2f}% | {100*sum(x['sampled_accuracy'] for x in rows)/5:.2f}% | {sum(x['passed'] for x in rows)}/5 |")
    lines+=['','This is a four-class communication diagnostic, not navigation or held-out task generalization. Same seed does not imply identical initial weights or noise draws after tensor shapes change. Vocabulary-sized parameters necessarily increase; embedding/hidden widths and four output actions stay fixed. No tuning or extra training was applied to failing seeds. Checkpoints, class probabilities and loss curves are saved.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
