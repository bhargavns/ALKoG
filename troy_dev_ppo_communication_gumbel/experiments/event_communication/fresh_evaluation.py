"""Final untouched scene split, evaluated only after all model designs freeze."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.distributions import Bernoulli
from . import study


SEED=1900001


def evaluate(model,arm,cost,prescribed=False,save_path=None):
    original=study.schedule
    if prescribed:
        def timing(arm,env,model,sample=True):
            if hasattr(env,'schedule_override'): return original(arm,env,model,sample)
            send=((env.t==0)|((env.risk!=env.last)&env.relevant&(env.pos[:,0]<env.lion[:,0]))).float()
            return send,Bernoulli(probs=torch.full_like(send,.5))
        study.schedule=timing
    rows=[]
    with torch.no_grad():
        for scenario in ('dynamic','static','rotated','late'):
            for iv in ('normal','suppress_updates','constant','shuffle','delay','drop','suppress_redundant'):
                torch.manual_seed(987654)
                save=save_path is not None and scenario=='dynamic' and iv in ('normal','suppress_updates')
                b=study.rollout(model,arm,study.Grid(1024,SEED,'cpu',scenario),cost,False,iv,save=save)
                rows.append(dict(scenario=scenario,intervention=iv,**study.metrics(b)))
                if save:
                    traces=[{k:v[:32] if isinstance(v,list) else v for k,v in row.items()}
                            for row in b['traces']]
                    (save_path/f'fresh_trajectories_{iv}.json').write_text(json.dumps(traces))
        normal=next(r for r in rows if r['scenario']=='dynamic' and r['intervention']=='normal')
        for mode in ('periodic','random'):
            torch.manual_seed(987654); env=study.Grid(1024,SEED)
            env.schedule_override=(normal['send_rate'],mode); env.schedule_phase=torch.rand(1024)
            b=study.rollout(model,arm,env,cost,False)
            rows.append(dict(scenario='dynamic',intervention='matched_'+mode,**study.metrics(b)))
    study.schedule=original
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(1)
    main_results=json.loads((a.study/'results.json').read_text())
    assert len(main_results)==42,'Wait for all main training runs to finish'
    # Validate every required final checkpoint before using the final split.
    root=a.study.parent
    for seed in (0,1,2):
        for kind in ('value','class'):
            assert (root/f'event_paired_{kind}_final{seed}'/'checkpoint.pt').exists()
    final=[]
    for row in main_results:
        arm,seed=row['arm'],row['seed']
        memory='graph' if arm=='graph' else 'memoryless' if arm=='memoryless' else 'gru'
        model=study.Pair(memory)
        model.load_state_dict(torch.load(a.study/f'{arm}_{seed}'/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
        cost=0. if arm in ('free','oracle') else .2 if arm=='expensive' else .05
        final.append(dict(row,evaluation=evaluate(model,arm,cost,save_path=a.study/f'{arm}_{seed}')))
        print('FRESH',arm,seed,flush=True)
    (a.study/'fresh_results.json').write_text(json.dumps(final,indent=2))
    for seed in (0,1,2):
        folder=root/f'event_control_seed{seed}'
        model=study.Pair('graph')
        model.load_state_dict(torch.load(folder/f'event_control_{seed}'/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
        row=json.loads((folder/'results.json').read_text())[0]
        (folder/'fresh_results.json').write_text(json.dumps([dict(row,evaluation=evaluate(model,'periodic',.05,True,folder))],indent=2))
        for kind in ('value','class'):
            folder=root/f'event_paired_{kind}_final{seed}'
            model=study.Pair('graph')
            model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)['model'])
            row=json.loads((folder/'results.json').read_text())
            (folder/'fresh_results.json').write_text(json.dumps(dict(row,evaluation=evaluate(model,'graph',.05,save_path=folder)),indent=2))
    (a.study/'final_evaluation_manifest.json').write_text(json.dumps(dict(scene_seed=SEED,device='cpu',
        episodes_per_condition=1024,main_runs=42,followup_runs=9,
        note='900001 was used in development and is not the final untouched split.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))


if __name__=='__main__': main()
