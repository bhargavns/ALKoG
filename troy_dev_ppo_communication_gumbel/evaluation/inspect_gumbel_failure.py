#!/usr/bin/env python3
"""Reproduce direct-loss seeds and audit exact vs straight-through gradients.

Only four categories: enumerate the exact expected discrete loss and its sender
logit gradient with the receiver held fixed. Compare against repeated ST Monte
Carlo gradients. Saves trained weights and every per-class counterfactual.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.probe_gumbel_direct import run


def norm(x): return float(x.detach().norm())
def cosine(a,b):
    return float(F.cosine_similarity(a[None],b[None],dim=-1)) if norm(a)*norm(b)>0 else None


def audit(tx,rx,channel,seed):
    torch.manual_seed(900000+seed)
    z=tx(torch.eye(4)).detach()
    probs=z.softmax(-1)
    decode_logits=rx.actor(rx.token_embeddings.weight)
    decode_probs=decode_logits.softmax(-1).detach()
    # Rows are targets; columns are symbols. No sampled estimator required.
    losses=-decode_logits.log_softmax(-1).T.detach()
    expected=(probs*losses).sum(-1)
    exact=probs*(losses-expected[:,None])
    rows=[]
    for target in range(4):
        st_grads=[]; msg_norms=[]; sample_losses=[]
        # 32 independent batches: estimate mean and MC uncertainty.
        for _ in range(32):
            logits=z[target].clone().requires_grad_()
            messages=channel(logits.expand(1024,-1))
            out=rx.actor(messages@rx.token_embeddings.weight)
            loss=F.cross_entropy(out,torch.full((1024,),target,dtype=torch.long))
            g,gm=torch.autograd.grad(loss,(logits,messages))
            st_grads.append(g.detach());msg_norms.append(norm(gm)*32)
            sample_losses.append(float(loss.detach()))
        gradients=torch.stack(st_grads)
        st=gradients.mean(0)
        symbol=int(z[target].argmax())
        raw=rx.actor(rx.token_embeddings(torch.tensor(symbol)))
        hard_loss=F.cross_entropy(raw[None],torch.tensor([target]))
        rxgrad=torch.autograd.grad(hard_loss,tuple(rx.actor.parameters())+(rx.token_embeddings.weight,))
        # Chain the mean estimated dL/dlogits into actual sender parameters.
        sender_logits=tx(torch.eye(4)[target:target+1])[0]
        sender_grad=torch.autograd.grad(sender_logits,tuple(tx.parameters()),grad_outputs=st)
        receiver_output_grad=decode_probs[symbol]-F.one_hot(torch.tensor(target),4)
        rows.append(dict(target=target,symbol=symbol,logits=z[target].tolist(),
            sender_probabilities=probs[target].tolist(),
            receiver_probabilities=decode_probs[symbol].tolist(),
            hard_loss=float(hard_loss.detach()),expected_discrete_loss=float(expected[target]),
            counterfactual_loss_by_symbol=losses[target].tolist(),
            receiver_output_gradient=receiver_output_grad.tolist(),
            receiver_parameter_gradient_norm=sum(norm(g)**2 for g in rxgrad)**.5,
            st_sender_logit_gradient=st.tolist(),
            st_sender_logit_gradient_standard_error=(gradients.std(0)/32**.5).tolist(),
            st_sender_logit_gradient_norm=norm(st),
            st_sender_parameter_gradient_norm=sum(norm(g)**2 for g in sender_grad)**.5,
            exact_sender_logit_gradient=exact[target].tolist(),
            exact_sender_logit_gradient_norm=norm(exact[target]),
            st_exact_cosine=cosine(st,exact[target]),
            st_descent_directional_derivative=float(-(exact[target]*st).sum()),
            sampled_loss_mean=sum(sample_losses)/len(sample_losses)))
    collision=[]
    for symbol in range(4):
        targets=[i for i in range(4) if int(z[i].argmax())==symbol]
        if len(targets)<2: continue
        individual=[]
        for target in targets:
            out=rx.actor(rx.token_embeddings(torch.tensor(symbol)))
            loss=F.cross_entropy(out[None],torch.tensor([target]))
            gs=torch.autograd.grad(loss,tuple(rx.actor.parameters())+(rx.token_embeddings.weight,))
            individual.append(torch.cat([g.flatten() for g in gs]))
        collision.append(dict(symbol=symbol,targets=targets,
            individual_receiver_gradient_norms=[norm(g) for g in individual],
            mean_receiver_gradient_norm=norm(torch.stack(individual).mean(0)),
            receiver_gradient_cosine=cosine(individual[0],individual[1])))
    return dict(seed=seed,receiver_action_probabilities_by_symbol=decode_probs.tolist(),
                classes=rows,collisions=collision)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--reference', type=Path, help='Optional direct-loss JSON to reproduce exactly')
    a=p.parse_args()
    if a.out.exists():p.error('Output directory already exists')
    a.out.mkdir(parents=True)
    torch.set_num_threads(1)
    original={x['seed']:x for x in json.loads(a.reference.read_text())['results']} if a.reference else None
    reports=[]
    for seed in a.seeds:
        captured=[]
        def capture(tx,rx,c): captured.extend((tx,rx,c))
        result=run(seed,on_trained=capture)
        matches=result==original[seed] if original is not None else None
        if matches is False: raise RuntimeError(f'Reproduction mismatch for seed {seed}')
        tx,rx,c=captured
        torch.save(dict(transmitter=tx.state_dict(),receiver=rx.state_dict(),temperature=c.temperature),a.out/f'seed{seed}.pt')
        report=audit(tx,rx,c,seed)
        report['exactly_reproduced_original_metrics_and_learning_curve']=matches
        reports.append(report)
        print(json.dumps(dict(seed=seed,reproduced=matches,collisions=report['collisions'],
            classes=[{k:v for k,v in row.items() if k in ['target','symbol','sender_probabilities','hard_loss','st_sender_logit_gradient_norm','exact_sender_logit_gradient_norm','st_exact_cosine','st_descent_directional_derivative','counterfactual_loss_by_symbol']} for row in report['classes']])),flush=True)
    (a.out/'audit.json').write_text(json.dumps(dict(
        scope='2000-update diagnostic; 32768 Gumbel samples per class for ST estimates; exact categorical enumeration holds receiver fixed; gradients before clipping',
        results=reports),indent=2)+'\n')

if __name__=='__main__':main()
