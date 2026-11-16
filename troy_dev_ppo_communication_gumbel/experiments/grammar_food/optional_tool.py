"""Freeze a learned rich protocol and learn DIRECT / INSPECT / ASK with PPO.

Only the gate is trained here. It observes visibility and inspection cost, not
the hidden scene. Direct action when visible and inspection grant exact facts.
Unknown direct action uses the independently trained silent same-seed policy.
"""
import argparse,json,time
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from .evaluate import load
from .task import Sampler,outcome,oracle,exhaustive_contexts,held_out

INSPECTION_COSTS=(.01,.12,.40)
REQUEST_COST=.02

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.actor=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,3))
        self.critic=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,1))

@torch.no_grad()
def options(protocol,silent,batch,visible,inspection):
    batch={**batch,'query_game':torch.zeros(len(visible),dtype=torch.bool)}
    _,_,best=oracle(batch)
    sa=silent.act(batch,True)
    direct=outcome(batch,sa['food'],sa['route'],bits=0)
    pa=protocol.act(batch,True)
    asked=outcome(batch,pa['food'],pa['route'],bits=8)
    rewards=torch.stack((torch.where(visible,best['reward'],direct['reward']),
                         best['reward']-inspection,asked['reward']-REQUEST_COST),-1)
    success=torch.stack((torch.where(visible,best['success'],direct['success']),best['success'],asked['success']),-1)
    return rewards,success

def run(folder,silent_folder,output,updates=600):
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    model,c=load(folder);silent,_=load(silent_folder);torch.manual_seed(7300+c['seed'])
    gate=Gate();opt=torch.optim.Adam(gate.parameters(),lr=.003)
    sampler=Sampler(9100000+c['seed'],split=c['split'],shift=c['shift']);curve=[];start=time.perf_counter()
    costs=torch.tensor(INSPECTION_COSTS)
    for t in range(updates):
        b=sampler.sample(512);g=sampler.generator
        visible=torch.rand(512,generator=g)<.5
        inspect=costs[torch.randint(3,(512,),generator=g)]
        obs=torch.stack((visible.float(),inspect),-1)
        with torch.no_grad():
            dist=Categorical(logits=gate.actor(obs));action=dist.sample();old=dist.log_prob(action)
            # The learner uses only the sampled branch reward, not unchosen values.
            reward,_=options(model,silent,b,visible,inspect)
            selected=reward[torch.arange(512),action]
            advantage=selected-gate.critic(obs).squeeze(-1)
            advantage=(advantage-advantage.mean())/advantage.std().clamp_min(1e-6)
        for _ in range(2):
            dist=Categorical(logits=gate.actor(obs));ratio=(dist.log_prob(action)-old).exp()
            loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            loss+=.5*(gate.critic(obs).squeeze(-1)-selected).square().mean()-.001*dist.entropy().mean()
            opt.zero_grad();loss.backward();opt.step()
        if t%100==0:curve.append(dict(update=t+1,reward=float(selected.mean()),loss=float(loss.detach())))
    data=exhaustive_contexts();n=len(data['facts']);mask=held_out(data['facts'],c['split'],c['shift'])
    rows=[]
    for visible_flag in (False,True):
        for inspection_cost in INSPECTION_COSTS:
            visible=torch.full((n,),visible_flag);inspection=torch.full((n,),inspection_cost)
            obs=torch.stack((visible.float(),inspection),-1)
            with torch.no_grad():
                action=gate.actor(obs).argmax(-1)
                rewards,success=options(model,silent,data,visible,inspection)
                selected=rewards[torch.arange(n),action];correct=success[torch.arange(n),action]
            for partition,select in [('all',torch.ones(n,dtype=torch.bool)),('train_meanings',~mask),('heldout_meanings',mask)]:
                means=rewards[select].mean(0)
                rows.append(dict(visible=visible_flag,inspection_cost=inspection_cost,partition=partition,
                                 action=int(action[0]),success=float(correct[select].float().mean()),utility=float(selected[select].mean()),
                                 context_reference_utility=float(means.max()),context_reference_action=int(means.argmax()),
                                 ask_fraction=float((action[select]==2).float().mean()),
                                 inspect_fraction=float((action[select]==1).float().mean()),direct_fraction=float((action[select]==0).float().mean())))
    result=dict(method=c['method'],split=c['split'],seed=c['seed'],updates=updates,rows=rows,curve=curve,
                training_seconds=time.perf_counter()-start,request_cost=REQUEST_COST,inspection_costs=INSPECTION_COSTS,
                protocol_frozen=True,primary_evaluation='mean over six visibility/cost contexts; argmax gate and frozen protocol')
    torch.save(dict(gate=gate.state_dict(),config=c),output/'checkpoint.pt')
    (output/'results.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--silent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=600)
    a=p.parse_args();torch.set_num_threads(1);r=run(a.run,a.silent,a.output,a.updates)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','curve')}),flush=True)

if __name__=='__main__':main()
