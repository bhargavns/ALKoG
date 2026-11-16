"""Follow-up: learn a gate from actual simulator-fork return differences.

Freeze the prescribed-event control's learned code and navigator. Train only
the gate on counterfactual outcomes. Extra simulator queries are NOT matched to
the initial PPO study. Continuation in both branches stays silent; this is a
one-packet value approximation, not an optimal long-term communication value.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from .study import Grid, Pair, metrics, rollout


def act(model, env, h, send):
    token=model.encoder(env.risk[:, None]).argmax(-1)
    h,belief=model.receive(h,token,send)
    action=model.actor(torch.cat((env.observation(),belief[:,None]),-1)).argmax(-1)
    reward,_,_=env.step(action,send,.05)
    return h,reward


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,default=80)
    p.add_argument('--objective',choices=('regression','classification'),default='regression')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(a.seed+700)
    model=Pair('graph')
    model.load_state_dict(torch.load(a.control/f'event_control_{a.seed}'/'checkpoint.pt',
                                    map_location='cpu',weights_only=True)['model'])
    start=time.perf_counter(); xs=[]; deltas=[]; query_steps=0
    with torch.no_grad():
        for batch in range(a.batches):
            env=Grid(256,30000000+a.seed*100000+batch)
            h=model.initial(256,'cpu')
            # Explore visited pre-message states under random sparse histories.
            t=batch%10
            for _ in range(t):
                h,_=act(model,env,h,(torch.rand(256)<.2).float())
            valid=~env.done; x=env.sender().clone(); values=[]
            for choice in (0.,1.):
                fork=copy.deepcopy(env); fh=h.clone(); total=torch.zeros(256)
                for k in range(24-t):
                    send=torch.full((256,),choice if k==0 else 0.)
                    query_steps += int((~fork.done).sum())
                    fh,reward=act(model,fork,fh,send)
                    total += (.99**k)*reward
                values.append(total)
            xs.append(x[valid]); deltas.append((values[1]-values[0])[valid])
    x,y=torch.cat(xs),torch.cat(deltas)
    opt=torch.optim.Adam(model.gate.parameters(),lr=.001)
    positive=y>0
    weights=torch.where(positive,1/positive.float().mean().clamp_min(.01),
                        1/(~positive).float().mean().clamp_min(.01))
    # Balanced regression emphasizes rare useful packets. Inputs remain local
    # sender observations/history; receiver hidden state is not a gate input.
    curve=[]
    for step in range(2000):
        idx=(torch.multinomial(weights,512,replacement=True) if a.objective=='regression'
             else torch.randint(len(x),(512,)))
        pred=model.gate(x[idx]).squeeze(-1)
        loss=((pred-y[idx]).square().mean() if a.objective=='regression' else
              torch.nn.functional.binary_cross_entropy_with_logits(pred,positive[idx].float()))
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0: curve.append(dict(step=step,loss=float(loss.detach())))
    evaluations=[]
    with torch.no_grad():
        for scenario in ('dynamic','static','rotated','late'):
            for iv in ('normal','suppress_updates','constant','shuffle','delay','drop','suppress_redundant'):
                torch.manual_seed(987654)
                b=rollout(model,'graph',Grid(1024,900001,'cpu',scenario),.05,False,iv)
                evaluations.append(dict(scenario=scenario,intervention=iv,**metrics(b)))
        normal=next(e for e in evaluations if e['scenario']=='dynamic' and e['intervention']=='normal')
        for mode in ('periodic','random'):
            torch.manual_seed(987654); env=Grid(1024,900001)
            env.schedule_override=(normal['send_rate'],mode)
            env.schedule_phase=torch.rand(1024)
            b=rollout(model,'graph',env,.05,False)
            evaluations.append(dict(scenario='dynamic',intervention='matched_'+mode,**metrics(b)))
    torch.save(dict(model=model.state_dict(),x=x,return_difference=y),a.output/'checkpoint.pt')
    result=dict(arm='paired_counterfactual_'+a.objective,seed=a.seed,states=len(x),
                positive_states=int(positive.sum()),extra_active_query_steps=query_steps,
                seconds=time.perf_counter()-start,evaluation=evaluations,curve=curve)
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    (a.output/'source.py').write_text(Path(__file__).read_text())
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
