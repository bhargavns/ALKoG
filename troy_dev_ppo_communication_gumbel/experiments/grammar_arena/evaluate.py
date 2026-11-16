"""Frozen MuJoCo arena evaluation; independent path outcomes and paired controls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from ..grammar_food.task import Sampler,all_facts,held_out
from .arena import FoodArena,BIT_COST,source_identity
from .bridge import freeze_protocol,messages,choose
from .train_navigation import Navigator

def cases(split,shift,repeats,seed,geometry_family):
    facts=all_facts().repeat_interleave(repeats,0)
    batch=Sampler(seed,split=split,shift=shift,partition='all').for_facts(facts)
    rng=np.random.default_rng(seed+13);geometries=[];yaws=[]
    for i in range(len(facts)):
        if geometry_family=='standard':scale=rng.uniform(.95,1.05);translation=rng.uniform(-.25,.25,2);angle=0.
        elif geometry_family=='transformed':scale=rng.uniform(.85,1.15);translation=rng.uniform(-.45,.45,2);angle=int(rng.integers(1,4))*np.pi/2
        else:raise ValueError(geometry_family)
        geometries.append(dict(scale=float(scale),translation=translation.tolist(),rotation=float(angle)))
        yaws.append(float(rng.uniform(-np.pi,np.pi)))
    return batch,geometries,np.array(yaws)

@torch.no_grad()
def evaluate_condition(envs,navigator,protocol,batch,geometries,yaws,condition,device,seed,keep_traces=False):
    n=len(batch['facts']);words=messages(protocol,batch['facts'],condition,seed)
    selected=choose(protocol,batch,words,geometries,yaws,oracle=condition=='oracle')
    episodes=[None]*n;traces={};active={};next_case=0;start=time.perf_counter()
    def reset(lane,index):
        env=envs[lane];env.reset_task(batch['facts'][index].tolist(),batch['candidates'][index].tolist(),int(batch['swap'][index]),geometries[index],yaws[index])
        obs=env.select(int(selected['food'][index]),int(selected['route'][index]))
        active[lane]=dict(index=index,obs=obs,total=0.,bumps=0,positions=[env.receiver_xy.tolist()],actions=[],rewards=[])
    for lane in range(min(len(envs),n)):reset(lane,next_case);next_case+=1
    while active:
        lanes=list(active);observations=np.stack([active[j]['obs'] for j in lanes])
        actions=navigator(torch.as_tensor(observations,device=device))[0].logits.argmax(-1).cpu().numpy()
        for lane,action in zip(lanes,actions):
            lane_state=active[lane];index=lane_state['index'];env=envs[lane]
            obs,reward,terminal,timeout,info=env.step(int(action))
            lane_state['obs']=obs;lane_state['total']+=reward;lane_state['bumps']+=int(info['bump'])
            if keep_traces:
                lane_state['positions'].append(env.receiver_xy.tolist());lane_state['actions'].append(int(action));lane_state['rewards'].append(reward)
            if terminal or timeout:
                bits=0 if condition in ('silent','oracle') else 8
                facts=batch['facts'][index].tolist()
                risk_gate=0 if int(batch['swap'][index])==facts[2] else 1
                route_safe=not(facts[3] and int(selected['route'][index])==risk_gate)
                expected=(int(selected['food'][index])==int(batch['target'][index])) and route_safe
                episodes[index]=dict(index=index,success=info['reached'],hazard=info['hazard'],timeout=bool(timeout),wrong_food=info['food']>=0 and not info['reached'],
                    food_correct=int(selected['food'][index])==int(batch['target'][index]),food=int(selected['food'][index]),route=int(selected['route'][index]),
                    route_safe=bool(route_safe),expected_success=bool(expected),execution_error=bool(expected and not info['reached']),
                    return_=lane_state['total']-BIT_COST*bits,steps=env.steps,bumps=lane_state['bumps'],words=words[index].tolist())
                if keep_traces:traces[str(index)]={k:lane_state[k] for k in ('positions','actions','rewards')}
                del active[lane]
                if next_case<n:reset(lane,next_case);next_case+=1
    return episodes,traces,dict(seconds=time.perf_counter()-start,episodes=n,environment_steps=sum(x['steps'] for x in episodes))

def run(protocol_folder,silent_folder,navigation_folder,output,repeats=8,seed=93000019,device='cuda',families=('standard','transformed'),conditions=('normal','silent','oracle','shuffle','blocked','reverse','invert_state'),lanes=32):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    protocol,c,phash=freeze_protocol(protocol_folder);silent,sc,shash=freeze_protocol(silent_folder)
    assert c['seed']==sc['seed'] and c['split']==sc['split'] and c['shift']==sc['shift']
    navpath=Path(navigation_folder)/'checkpoint.pt';nhash=hashlib.sha256(navpath.read_bytes()).hexdigest()
    navstate=torch.load(navpath,map_location=device,weights_only=False)
    navigator=Navigator(navstate['config']['width']).to(device);navigator.load_state_dict(navstate['model']);navigator.eval()
    envs=[FoodArena(j) for j in range(lanes)];rows=[];start=time.perf_counter();alltimings=[]
    source=source_identity()
    config=dict(protocol=str(Path(protocol_folder).resolve()),silent_protocol=str(Path(silent_folder).resolve()),navigation=str(navpath.resolve()),
                protocol_sha256=phash,silent_sha256=shash,navigation_sha256=nhash,split=c['split'],shift=c['shift'],protocol_seed=c['seed'],
                repeats=repeats,evaluation_seed=seed,device=device,geometry_families=families,conditions=conditions,lanes=lanes,**source)
    config['integration_source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    source_folder=output/'source';source_folder.mkdir()
    for p in Path(__file__).parent.glob('*.py'):(source_folder/p.name).write_bytes(p.read_bytes())
    (output/'config.json').write_text(json.dumps(config,indent=2))
    for family in families:
        batch,geometries,yaws=cases(c['split'],c['shift'],repeats,seed,family)
        held=held_out(batch['facts'],c['split'],c['shift']).numpy()
        (output/f'{family}-scenes.json').write_text(json.dumps(dict(facts=batch['facts'].tolist(),candidates=batch['candidates'].tolist(),swap=batch['swap'].tolist(),target=batch['target'].tolist(),geometry=geometries,yaw=yaws.tolist()),indent=2))
        for condition in conditions:
            chosen=silent if condition=='silent' else protocol
            records,traces,timing=evaluate_condition(envs,navigator,chosen,batch,geometries,yaws,condition,device,seed,keep_traces=condition in ('normal','oracle','invert_state'))
            (output/f'{family}-{condition}-episodes.json').write_text(json.dumps(records))
            if traces:
                # All selected arms retain complete movements for auditing.
                import gzip
                with gzip.open(output/f'{family}-{condition}-trajectories.json.gz','wt',encoding='utf-8') as f:json.dump(traces,f)
            for label,mask in [('all',np.ones(len(held),dtype=bool)),('train_meanings',~held),('heldout_meanings',held)]:
                part=[r for r,m in zip(records,mask) if m]
                row=dict(family=family,condition=condition,partition=label,episodes=len(part),**{key:float(np.mean([r[key] for r in part])) for key in ('success','hazard','timeout','wrong_food','food_correct','return_','steps','bumps','expected_success','execution_error')})
                expected_cases=[r for r in part if r['expected_success']]
                row['execution_fidelity']=float(np.mean([r['success'] for r in expected_cases])) if expected_cases else None
                rows.append(row)
            alltimings.append(dict(family=family,condition=condition,**timing))
            (output/'progress.json').write_text(json.dumps(dict(rows=rows,timings=alltimings),indent=2))
            print(json.dumps(dict(family=family,condition=condition,**{k:v for k,v in rows[-1].items() if k not in ('family','condition')})),flush=True)
    for env in envs:env.close()
    assert hashlib.sha256((Path(protocol_folder)/'checkpoint.pt').read_bytes()).hexdigest()==phash
    assert hashlib.sha256((Path(silent_folder)/'checkpoint.pt').read_bytes()).hexdigest()==shash
    assert hashlib.sha256(navpath.read_bytes()).hexdigest()==nhash
    result=dict(config=config,rows=rows,timings=alltimings,total_seconds=time.perf_counter()-start,all_checkpoints_unchanged=True)
    (output/'results.json').write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--silent',type=Path,required=True);p.add_argument('--navigation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=8);p.add_argument('--seed',type=int,default=93000019);p.add_argument('--device',default='cuda');p.add_argument('--family',choices=('standard','transformed','both'),default='both');p.add_argument('--conditions',nargs='+',default=['normal','silent','oracle','shuffle','blocked','reverse','invert_state'])
    a=p.parse_args();run(a.protocol,a.silent,a.navigation,a.output,a.repeats,a.seed,a.device,('standard','transformed') if a.family=='both' else (a.family,),tuple(a.conditions))

if __name__=='__main__':main()
