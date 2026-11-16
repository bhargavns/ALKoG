"""Independent sampled execution and receiver-local navigation checks."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .models import TASKS
from .navigation import plan,room_route


PRIMARY={'private_goal':'learned','route_safety':'learned','scouting':'learned_warm','complementary':'dialogue_warm'}


def simulate(model,task,n,generator):
    rand=lambda high:torch.randint(high,(n,),generator=generator)
    details={}
    if task=='private_goal':
        target=rand(4);word=model.sender.argmax(-1)[target]
        chosen=model.receiver.argmax(-1)[word]
        correct=chosen==target;packets=(word>0).float();utility=correct.float()-.05*packets
        details=dict(target=target,word=word,chosen=chosen)
    elif task=='route_safety':
        state=rand(2);memory=torch.zeros(n,dtype=torch.long)
        active=torch.ones(n,dtype=torch.bool);packets=torch.zeros(n)
        decisions=[];sender=model.sender.argmax(-1);decoder=model.receiver.argmax(-1)
        for t in range(9):
            word=sender[t,state,memory]
            packets+=active*(word>0)
            memory=torch.where(word>0,word,memory)
            if t in (0,4,8):
                choice=decoder[memory]
                decisions.append(dict(t=t,safe=state[:32].tolist(),choice=choice[:32].tolist(),word=word[:32].tolist()))
                active &= choice==state
            if t<8:
                state=torch.where(torch.rand(n,generator=generator)<.08,1-state,state)
        correct=active;utility=active.float()-.02*packets;details=dict(decisions=decisions)
    elif task=='scouting':
        target=rand(4);c=rand(3);visible=rand(2).bool()
        obs=torch.where(visible,target,4)
        action=model.first.argmax(-1)[c,obs]
        word=model.sender.argmax(-1)[target]
        answered=model.answer.argmax(-1)[word]
        chosen=torch.where(action==4,target,torch.where(action==5,answered,action))
        correct=chosen==target;packets=2*(action==5).float()
        cost=torch.tensor(model.costs)[c]*(action==4)+.08*(action==5)
        utility=correct.float()-cost
        details=dict(target=target,visible=visible,scout_cost=torch.tensor(model.costs)[c],action=action,word=word,chosen=chosen)
    else:
        query=rand(8);map_id=rand(256);maps=model.maps[map_id]
        word=model.query.argmax(-1)[query];ask=word>0
        chosen_index=model.lookup.argmax(-1)[(word-1).clamp_min(0)]
        fact=maps[torch.arange(n),chosen_index]
        reply=model.bit_encoder.argmax(-1)[fact]
        reply=torch.where(ask,reply,2)
        chosen=model.answer.argmax(-1)[query,reply]
        correct=chosen==maps[torch.arange(n),query]
        packets=2*ask.float();utility=correct.float()-.02*(1+4*ask.float())
        details=dict(query=query,map_id=map_id,query_word=word,reply=reply,chosen=chosen)
    data={k:v[:32].tolist() if isinstance(v,torch.Tensor) else v for k,v in details.items()}
    return dict(success=float(correct.float().mean()),packets=float(packets.mean()),utility=float(utility.mean())),data


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=50000);a=p.parse_args()
    torch.set_num_threads(1)
    rows=json.loads((a.study/'results.json').read_text())
    assert len(rows)==130
    output=[];traces=[]
    with torch.no_grad():
        for row in rows:
            task,method,seed=row['task'],row['method'],row['seed']
            if method!=PRIMARY[task]:continue
            model=TASKS[task][0](method)
            model.load_state_dict(torch.load(a.study/f'{task}_{method}_{seed}'/'checkpoint.pt',weights_only=True,map_location='cpu')['model'])
            g=torch.Generator().manual_seed(80000019+seed)
            sampled,trace=simulate(model,task,a.episodes,g)
            exact=next(r for r in row['evaluations'] if r['deployment']=='argmax' and r['intervention']=='normal')
            for key in ('success','packets','utility'):
                # Bounded sample mean check (validation tolerance, not a claim
                # of five independent population estimates).
                assert abs(sampled[key]-exact[key])<.035,(task,seed,key,sampled,exact)
            output.append(dict(task=task,seed=seed,episodes=a.episodes,exact=exact,sampled=sampled))
            traces.append(dict(task=task,seed=seed,examples=trace))
    rng=np.random.default_rng(80000019);navigation=[]
    goals=((1,1),(11,1),(1,11),(11,11))
    for case in range(128):
        start=(6,6);blocked={(int(x),int(y)) for x,y in rng.integers(0,13,(12,2))}
        blocked-=set(goals)|{start}
        try: paths=[plan(start,g,blocked) for g in goals]
        except ValueError: continue
        for goal,path in zip(goals,paths):
            assert path[0]==start and path[-1]==goal
            assert not any(p in blocked for p in path)
            assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))
        navigation.append(dict(start=start,goals=goals,blocked=sorted(blocked),paths=paths))
    assert len(navigation)>=100
    for choice in (0,1):
        path=room_route(choice)
        assert (3,1 if choice==0 else 5) in path
    (a.study/'independent_validation.json').write_text(json.dumps(output,indent=2))
    (a.study/'execution_examples.json').write_text(json.dumps(traces,indent=2))
    (a.study/'navigation_examples.json').write_text(json.dumps(navigation,indent=2))
    print(json.dumps(dict(validated_policies=len(output),sampled_episodes=len(output)*a.episodes,
                          navigation_layouts=len(navigation)),indent=2))


if __name__=='__main__':main()
