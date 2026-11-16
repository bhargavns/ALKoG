"""Whole-message hypothesis retrieval into the unchanged MuJoCo motor bridge.

No word positions, semantic token mappings or decoder heads are introduced.
The public 4-food x 4-hazard candidate hypotheses are scored by the frozen
neural listener. This is an explicit supplied action interface, not learned
planning, visual perception or end-to-end motor/language training.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from .evaluate import load_models,packets
from .task import features,split_mask
from .models import intervene
from ..grammar_arena.arena import FoodArena,BIT_COST,FOOD_BONUS,HAZARD_PENALTY,MOVE_COST,source_identity
from ..grammar_arena.bridge import public_lengths
from ..grammar_arena.evaluate import cases
from ..grammar_arena.train_navigation import Navigator

def new_facts(old):return torch.stack((old[:,0],old[:,1],torch.zeros_like(old[:,0]),old[:,2],old[:,3]),-1)

@torch.no_grad()
def select(pair,batch,words,geometries,yaws,oracle=False):
    n=len(words);public=batch['candidates'];hazards=torch.cartesian_prod(torch.arange(2),torch.arange(2));hypotheses=torch.zeros(n,4,4,5,dtype=torch.long)
    hypotheses[...,:2]=public[:,:,None,:2];hypotheses[...,3:]=hazards[None,None,:,:]
    candidate_scenes=features(hypotheses.reshape(-1,5),'attributes').reshape(n,16,1,24)
    logits=pair.listener(candidate_scenes,tokens=words);posterior=logits.softmax(-1).reshape(n,4,2,2)
    room=posterior.sum((2,3)).argmax(-1);animal_state=posterior.sum(1)
    if oracle:
        room=batch['target'];animal_state=torch.zeros_like(animal_state)
        animal_state[torch.arange(n),batch['facts'][:,2],batch['facts'][:,3]]=1.
    animals=torch.stack((batch['swap'],1-batch['swap']),-1)
    danger=torch.nn.functional.pad(animal_state[:,:,1].gather(1,animals),(0,1))
    lengths,prefix=map(np.stack,zip(*(public_lengths(g,y) for g,y in zip(geometries,yaws))))
    lengths=torch.tensor(lengths).float()[torch.arange(n),room];prefix=torch.tensor(prefix).float()
    utility=FOOD_BONUS*(1-danger)-HAZARD_PENALTY*danger-MOVE_COST*((1-danger)*lengths+danger*prefix)
    return dict(food=room.numpy(),route=utility.argmax(-1).numpy(),posterior=posterior.numpy())

@torch.no_grad()
def execute(envs,navigator,batch,geometries,yaws,selected,words,condition,device):
    n=len(words);records=[None]*n;active={};next_case=0
    def reset(lane,index):
        e=envs[lane];e.reset_task(batch['facts'][index].tolist(),batch['candidates'][index].tolist(),int(batch['swap'][index]),geometries[index],yaws[index])
        active[lane]=dict(index=index,obs=e.select(int(selected['food'][index]),int(selected['route'][index])),total=0.,bumps=0)
    for lane in range(min(n,len(envs))):reset(lane,next_case);next_case+=1
    while active:
        lanes=list(active);obs=torch.as_tensor(np.stack([active[lane]['obs'] for lane in lanes]),device=device)
        actions=navigator(obs)[0].logits.argmax(-1).cpu().numpy()
        for lane,action in zip(lanes,actions):
            a=active[lane];i=a['index'];e=envs[lane]
            observation,reward,terminal,timeout,info=e.step(int(action));a['obs']=observation;a['total']+=reward;a['bumps']+=int(info['bump'])
            if terminal or timeout:
                facts=batch['facts'][i].tolist();risk_gate=0 if int(batch['swap'][i])==facts[2] else 1
                safe=not(facts[3] and int(selected['route'][i])==risk_gate)
                correct=int(selected['food'][i])==int(batch['target'][i]);expected=correct and safe
                bits=0 if condition in ('oracle','blocked') else 24
                records[i]=dict(index=i,success=bool(info['reached']),hazard=bool(info['hazard']),timeout=bool(timeout),food_correct=correct,route_safe=bool(safe),
                    expected_success=bool(expected),execution_error=bool(expected and not info['reached']),steps=e.steps,bumps=a['bumps'],return_=a['total']-BIT_COST*bits,
                    food=int(selected['food'][i]),route=int(selected['route'][i]),words=words[i].tolist())
                del active[lane]
                if next_case<n:reset(lane,next_case);next_case+=1
    return records

def run(study,output,seeds=range(5),repeats=4,device='cuda'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);rows=[];start=time.perf_counter()
    navroot=Path(__file__).parents[1]/'grammar_arena/checkpoints/navigation'
    manifest=dict(method='parallel',selection='Fixed on development seed97 before final seeds',hypotheses=16,bits=24,repeats=repeats,seeds=list(seeds),device=device,gpu=torch.cuda.get_device_name(),source=source_identity(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoints=[])
    for seed in seeds:
        cp=Path(study)/f'attributes_parallel_{seed}/checkpoint.pt';models,c=load_models(cp);pair=models[0]
        navpath=navroot/f'seed{seed}/checkpoint.pt';saved=torch.load(navpath,map_location=device,weights_only=False)
        nav=Navigator(saved['config']['width']).to(device);nav.load_state_dict(saved['model']);nav.eval()
        manifest['checkpoints'].append(dict(seed=seed,communication_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),navigation_sha256=hashlib.sha256(navpath.read_bytes()).hexdigest()))
        envs=[FoodArena(i) for i in range(32)]
        try:
            for family in ('standard','transformed'):
                batch,geometry,yaws=cases('food_pair',1,repeats,9600001,family);nf=new_facts(batch['facts']);held=split_mask(nf,'attributes',1).numpy()
                original=packets(pair.speaker,features(nf,'attributes'))
                scenes=dict(facts=batch['facts'].tolist(),candidates=batch['candidates'].tolist(),target=batch['target'].tolist(),swap=batch['swap'].tolist(),geometry=geometry,yaw=yaws.tolist(),held_out=held.tolist())
                (root/f'seed{seed}-{family}-scenes.json').write_text(json.dumps(scenes))
                for condition in ('normal','blocked','shuffle','sort','oracle'):
                    words=intervene(original,'normal' if condition=='oracle' else condition,8,torch.Generator().manual_seed(9700001))
                    selected=select(pair,batch,words,geometry,yaws,oracle=condition=='oracle');episodes=execute(envs,nav,batch,geometry,yaws,selected,words,condition,device)
                    (root/f'seed{seed}-{family}-{condition}-episodes.json').write_text(json.dumps(episodes))
                    for part,mask in [('all',np.ones(len(held),dtype=bool)),('train_meanings',~held),('heldout_meanings',held)]:
                        records=[e for e,m in zip(episodes,mask) if m]
                        rows.append(dict(seed=seed,family=family,condition=condition,partition=part,episodes=len(records),**{k:float(np.mean([e[k] for e in records])) for k in ('success','hazard','timeout','food_correct','route_safe','expected_success','execution_error','steps','bumps','return_')}))
                    (root/'progress.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
        finally:
            for e in envs:e.close()
        assert hashlib.sha256(cp.read_bytes()).hexdigest()==manifest['checkpoints'][-1]['communication_sha256']
        assert hashlib.sha256(navpath.read_bytes()).hexdigest()==manifest['checkpoints'][-1]['navigation_sha256']
    result=dict(manifest=manifest,rows=rows,total_seconds=time.perf_counter()-start,all_checkpoints_unchanged=True)
    (root/'results.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=4);a=p.parse_args();run(a.study,a.output,repeats=a.repeats)
