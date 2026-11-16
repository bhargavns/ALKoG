"""Frozen communication on new public obstacle layouts with scalar path checks.

The path planner and map-to-waypoint interface are supplied. This validates
execution of learned meanings in new geometry, not learned maze exploration.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .evaluate import load
from .task import Sampler,all_facts,held_out,plan,TIME_COST,BIT_COST

def generate_layouts(count=128,seed=92000019):
    rng=np.random.default_rng(seed);layouts=[]
    width=height=13;start=(1,6);goals=((11,1),(11,4),(11,8),(11,11));gates=((6,2),(6,6),(6,10))
    wall={(6,y) for y in range(height)}-set(gates)
    while len(layouts)<count:
        obstacles={(int(x),int(y)) for x,y in rng.integers(0,13,(10,2))}
        obstacles-=set(goals)|set(gates)|{start};obstacles|=wall
        routes=[];hit_steps=[]
        try:
            for goal in goals:
                paths=[]
                for j,gate in enumerate(gates):
                    blocked=obstacles|(set(gates)-{gate})
                    prefix=plan(start,gate,blocked,width,height);suffix=plan(gate,goal,blocked,width,height)
                    path=prefix+suffix[1:];paths.append(path)
                    assert set(path)&set(gates)=={gate}
                routes.append(paths)
            for j in range(2):
                blocked=obstacles|(set(gates)-{gates[j]})
                hit_steps.append(len(plan(start,gates[j],blocked,width,height))-1)
        except ValueError:continue
        layouts.append(dict(start=start,goals=goals,gates=gates,blocked=sorted(obstacles),paths=routes,
                            lengths=[[len(p)-1 for p in ps] for ps in routes],hit_steps=hit_steps))
    return layouts

@torch.no_grad()
def receiver_choices(model,batch,layout):
    words=model.speak(batch['facts'],True)[0]
    logits=model.listen(words);lp=[x.log_softmax(-1) for x in logits]
    cand=batch['candidates'];n=len(cand);ids=torch.arange(n)[:,None]
    food=(lp[0][ids,cand[:,:,0]]+lp[1][ids,cand[:,:,1]]).argmax(-1)
    probs=[x.softmax(-1) for x in logits]
    animals=torch.stack((batch['swap'],1-batch['swap']),-1)
    danger=torch.nn.functional.pad(probs[2].gather(1,animals)*probs[3][:,1,None],(0,1))
    lengths=torch.tensor(layout['lengths']).float()[food]
    stops=torch.tensor((*layout['hit_steps'],0)).float()
    predicted_steps=(1-danger)*lengths+danger*stops
    route=((1-2*danger)-TIME_COST*predicted_steps).argmax(-1)
    return words,food,route

def execute(layout,facts,candidates,swap,food,route,bits=8):
    path=layout['paths'][food][route];risk_cell=layout['gates'][0 if facts[2]==swap else 1] if facts[3] else None
    correct=tuple(candidates[food])==tuple(facts[:2]);total=-bits*BIT_COST;trace=[];hit=False
    for t,p in enumerate(path[1:]):
        hit=tuple(p)==tuple(risk_cell) if risk_cell is not None else False
        reward=-TIME_COST-int(hit)
        if t==len(path)-2 and correct and not hit:reward+=1
        total+=reward;trace.append(dict(t=t+1,position=p,reward=reward))
        if hit:break
    return dict(success=correct and not hit,hazard=hit,reward=total,steps=len(trace),trajectory=trace)

def run(folder,output,layouts):
    model,c=load(folder);sampler=Sampler(94000019,split=c['split'],shift=c['shift'],partition='all')
    facts=all_facts();test=held_out(facts,c['split'],c['shift']).numpy()
    records=[];examples=[]
    for k,layout in enumerate(layouts):
        batch=sampler.for_facts(facts);words,food,route=receiver_choices(model,batch,layout)
        for i in range(64):
            result=execute(layout,facts[i].tolist(),batch['candidates'][i].tolist(),int(batch['swap'][i]),int(food[i]),int(route[i]),0 if c['channel']=='silent' else 8)
            records.append(dict(layout=k,meaning=i,heldout=bool(test[i]),**{a:b for a,b in result.items() if a!='trajectory'}))
            if k<2 and i in (0,1,10,15,31,47,63):examples.append(dict(layout=k,facts=facts[i].tolist(),candidates=batch['candidates'][i].tolist(),
                swap=int(batch['swap'][i]),words=words[i].tolist(),chosen_food=int(food[i]),chosen_route=int(route[i]),**result))
    rows=[]
    for partition in ('all','train_meanings','heldout_meanings'):
        selected=[r for r in records if partition=='all' or r['heldout']==(partition=='heldout_meanings')]
        rows.append(dict(partition=partition,episodes=len(selected),**{a:float(np.mean([r[a] for r in selected])) for a in ('success','hazard','reward','steps')}))
    result=dict(method=c['method'],split=c['split'],seed=c['seed'],rows=rows,examples=examples,layouts=len(layouts),
                geometry='13x13 public walls and random obstacles; all policies and words frozen; BFS supplied')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layouts',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    layouts=json.loads(a.layouts.read_text());r=run(a.run,a.output,layouts)
    print(json.dumps({k:v for k,v in r.items() if k not in ('examples',)}),flush=True)

if __name__=='__main__':main()
