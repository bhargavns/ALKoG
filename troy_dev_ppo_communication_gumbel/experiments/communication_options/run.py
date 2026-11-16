"""Train complete finite communication games and preserve exact artifacts."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from .models import TASKS, entropy_regularizer, silent_route_ceiling, broadcast_bound


def numbers(values):
    return {k:float(v.detach()) for k,v in values.items()}


def train(task,method,seed,steps,device,folder):
    torch.manual_seed(seed)
    model=TASKS[task][0](method).to(device)
    start=time.perf_counter()
    warmup_steps=0
    if method in ('learned_warm','dialogue_warm'):
        warmup_steps=600 if task=='scouting' else 1000
        model.warmup(warmup_steps)
    fixed=method in ('fixed','oracle','fixed_event','fixed_dialogue','fixed_broadcast','scout_only')
    opt=torch.optim.Adam(model.parameters(),lr=.04)
    curve=[]
    if not fixed:
        for t in range(steps):
            values=model()
            # Symmetry breaking/exploration, annealed away before final fitting.
            entropy=.02*max(0,1-t/(steps*.4))
            loss=-values['utility']-entropy*entropy_regularizer(model)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            if t%25==0 or t==steps-1:
                curve.append(dict(step=t+1,loss=float(loss.detach()),**numbers(values)))
    evaluations=[]
    with torch.no_grad():
        for hard in (False,True):
            for iv in ('normal','blocked','constant','permute','suppress_updates','reset_memory','drop25','remove_query'):
                evaluations.append(dict(deployment='argmax' if hard else 'sampled',
                                        intervention=iv,**numbers(model(hard,iv))))
    row=dict(task=task,method=method,seed=seed,steps=0 if fixed else steps,
             warmup_steps=warmup_steps,seconds=time.perf_counter()-start,evaluations=evaluations)
    folder.mkdir(parents=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),
                    task=task,method=method,seed=seed),folder/'checkpoint.pt')
    (folder/'curve.json').write_text(json.dumps(curve,indent=2))
    (folder/'evaluation.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS))
    p.add_argument('--methods',nargs='+')
    p.add_argument('--development',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    manifest=dict(device=a.device,steps=a.steps,seeds=a.seeds,tasks=a.tasks,methods=a.methods,
                  development=a.development,python=platform.python_version(),torch=torch.__version__,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  sources=sources,optimizer='Adam lr .04',word_costs={'goal':.05,'route':.02,'scouting_request':.02,'scouting_reply':.06,'dialogue_bit':.02},
                  evaluation='Exact enumeration over finite logical states; argmax primary, sampled secondary.',
                  silent_bounds={'private_goal':.25,'route_safety':silent_route_ceiling(),
                                 'scouting_utility':.91,'complementary_accuracy':.5},
                  one_way_5bit_accuracy_upper_bound=broadcast_bound())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    source_dir=a.output/'source';source_dir.mkdir()
    for f in Path(__file__).parent.glob('*.py'): (source_dir/f.name).write_bytes(f.read_bytes())
    rows=[]
    for task in a.tasks:
        for method in (a.methods or TASKS[task][1]):
            for seed in a.seeds:
                row=train(task,method,seed,a.steps,a.device,a.output/f'{task}_{method}_{seed}')
                rows.append(row)
                (a.output/'results.json').write_text(json.dumps(rows,indent=2))
                normal=next(x for x in row['evaluations'] if x['deployment']=='argmax' and x['intervention']=='normal')
                print(json.dumps(dict(task=task,method=method,seed=seed,seconds=row['seconds'],**normal)),flush=True)


if __name__=='__main__':main()
