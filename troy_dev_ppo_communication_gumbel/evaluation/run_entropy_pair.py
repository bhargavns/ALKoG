#!/usr/bin/env python3
"""Archive/validate a matched sender-entropy pair, then launch it detached."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
KEYS=('food_rate','loose_death_rate','mean_return','mean_length','sender_entropy',
      'sender_entropy_beta','rx_entropy','rx_explained_var','tx_actor_grad_norm',
      'tx_value_grad_norm','tx_entropy_grad_norm')

def save(p,data):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def env_for(root):
    return dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MUJOCO_GL='egl',MPLCONFIGDIR=str(root/'matplotlib'))
def metrics(root,arm):
    paths=list((root/'runs').glob(f'*_{arm}/metrics.jsonl'))
    rows=[]
    if paths:
        for line in paths[0].read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    return paths[0].parent if paths else None,rows

def report(root,jobs,label):
    manifest=json.loads((root/'launch.json').read_text())
    data={'time_utc':utc(),'planned_iterations':manifest['iterations'],'arms':{}}
    lines=[f'# {label}: {manifest["control"]}','',f'Recorded {utc()}. Means weight iterations equally; single-seed training metrics.','']
    for arm,proc in jobs.items():
        run,rows=metrics(root,arm)
        recent=rows[-25:]
        avg={k:sum(x[k] for x in recent if x.get(k) is not None)/max(1,sum(x.get(k) is not None for x in recent)) for k in KEYS}
        data['arms'][arm]={'pid':proc.pid,'exit_code':proc.poll(),'iterations':len(rows),'run_dir':str(run) if run else None,'last25':avg,'episodes':sum(x.get('episodes',0) for x in rows)}
        lines += [f'## {arm}', '',f'Iterations {len(rows)}/{manifest["iterations"]}; exit code {proc.poll()}.','', '| Metric | Last 25 iterations |','|---|---:|']
        lines += [f'| {k} | {v:.6f} |' for k,v in avg.items()]
        if recent:lines+=['',f'Latest 16-symbol frequencies: {recent[-1]["symbol_fracs"]}']
        lines+=['']
    save(root/f'{label}.json',data);(root/f'{label}.md').write_text('\n'.join(lines)+'\n')

def prepare(args):
    root=ROOT/'output/experiments'/datetime.now().strftime(f'gumbel16_{args.control}_pair_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source';source.mkdir()
    for folder in ('environment', 'perception', 'representation', 'communication', 'evaluation', 'shared', 'observability', 'tests'):
        shutil.copytree(ROOT/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    for name in ('requirements.txt', 'pytest.ini'):
        shutil.copy2(ROOT/name, source/name)
    commands={}
    arms = (('scheduled_s0', .1, 0), ('no_entropy_s0', 0., 0)) if args.control == 'entropy' else (
        ('coupled_s0', 0., 0), ('detached_s0', 0., 1))
    control = ('Scheduled sender entropy versus zero bonus; both channels learn' if args.control == 'entropy'
               else 'Coupled versus detached channel; both sender entropy bonuses are zero')
    for arm,beta,detach in arms:
        commands[arm]=[sys.executable,str(source/'communication/train_communication.py'),
            'n_symbols=16','channel_temperature=1.2',
            f'sender_entropy_beta={beta}','sender_entropy_schedule=annealed',
            'perception=oracle_state',f'colocated={args.colocated}','receiver_yaw_mode=shared','seed=0',
            f'detach_channel={detach}',f'iterations={args.iterations}',f'horizon={args.horizon}',
            f'device={args.device}','videos=0',f'checkpoint_every={args.checkpoint_every}',
            f'run_name={arm}',f'out_dir={root/"runs"}']
    manifest={'created_utc':utc(),'iterations':args.iterations,'horizon':args.horizon,
        'colocated':args.colocated,'receiver_yaw_mode':'shared','device':args.device,'eval_episodes':args.eval_episodes,'commands':commands,
        'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},
        'schedule':'beta .1 for first quarter, linear decay to zero at three quarters, then zero',
        'control':control,
        'resume_semantics':'weights only; fresh seed 0, no diagnostic pretraining'}
    save(root/'launch.json',manifest)
    diagnostic=[sys.executable,str(source/'evaluation/probe_gumbel_direct.py'),'--symbols','16',
        '--entropy-beta','0.1','--entropy-schedule','annealed','--seeds','0','1','2','3','4',
        '--out',str(root/'preflight_diagnostic.json')]
    with (root/'preflight_diagnostic.log').open('w') as log:
        code=subprocess.run(diagnostic,env=env_for(root),stdout=log,stderr=subprocess.STDOUT).returncode
    if code:raise RuntimeError(f'Preflight diagnostic failed; no arena launched: {root}')
    manifest['preflight_passed']=True;save(root/'launch.json',manifest)
    (root/'README.md').write_text(f'''# Matched 16-symbol PPO experiment

{args.iterations} iterations × {args.horizon} steps per arm, fresh seed 0, shared yaw,
{'shared position' if args.colocated else 'separate starting positions; stationary transmitter'}, exact oracle inputs, current-message MLP, ST Gumbel temperature 1.2.
{control}. When enabled, beta .1 holds through the first quarter, then decays
to zero at three quarters. Receiver action entropy and other PPO settings match.
No diagnostic weights are transferred. Five-seed source-snapshot preflight passed.
Checkpoints every {args.checkpoint_every} iterations; final weights saved. Warm
starts restore weights only. Reports are single-seed training metrics.
After both arms finish, each successful arm is evaluated on {args.eval_episodes}
unseen scenes per mode: hard, sampled, constant, random. Receiver actions are
sampled, matching training. Training and evaluation run independently of the client.
See launch.json for exact commands/source hashes and status.json for progress.
''')
    print(json.dumps({'prepared_experiment':str(root)},indent=2),flush=True)

def worker(root):
    m=json.loads((root/'launch.json').read_text());jobs={};started=time.time()
    for arm,cmd in m['commands'].items():
        with (root/f'{arm}.log').open('ab') as log:
            jobs[arm]=subprocess.Popen(cmd,cwd=root/'source',env=env_for(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    m.update(started_utc=utc(),supervisor_pid=os.getpid(),pids={k:p.pid for k,p in jobs.items()});save(root/'launch.json',m)
    next_report=started+3600
    while True:
        codes={k:p.poll() for k,p in jobs.items()}
        save(root/'status.json',{'time_utc':utc(),'phase':'training','pids':m['pids'],'exit_codes':codes})
        if time.time()>=next_report:
            report(root,jobs,f'hour_{int((time.time()-started)//3600)}');next_report+=3600
        if all(code is not None for code in codes.values()):break
        time.sleep(15)
    report(root,jobs,'training_final')
    evaluations={}
    for arm,proc in jobs.items():
        run,rows=metrics(root,arm)
        if proc.returncode!=0 or len(rows)!=m['iterations']:
            evaluations[arm]='skipped: training incomplete';continue
        cmd=[sys.executable,str(root/'source/evaluation/evaluate_communication.py'),'--run-dir',str(run),
            '--episodes',str(m['eval_episodes']),'--modes','hard','sampled','constant','random','--device',m['device']]
        with (root/f'{arm}_evaluation.log').open('ab') as log:
            ep=subprocess.Popen(cmd,cwd=root/'source',env=env_for(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(root/'status.json',{'time_utc':utc(),'phase':'evaluation','arm':arm,'evaluation_pid':ep.pid,'exit_codes':codes})
            evaluations[arm]=ep.wait()
    save(root/'status.json',{'time_utc':utc(),'phase':'complete','exit_codes':codes,'evaluation_exit_codes':evaluations})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--launch',type=Path);group.add_argument('--worker',type=Path)
    p.add_argument('--iterations',type=int,default=800);p.add_argument('--horizon',type=int,default=2048)
    p.add_argument('--device',default='cuda');p.add_argument('--eval-episodes',type=int,default=500)
    p.add_argument('--colocated',type=int,choices=(0,1),default=0)
    p.add_argument('--control', choices=('entropy', 'detached'), default='entropy')
    p.add_argument('--checkpoint-every',type=int,default=25)
    a=p.parse_args()
    if a.prepare:
        if a.iterations<1 or a.horizon<2 or a.eval_episodes<1 or a.checkpoint_every<1:p.error('Positive budgets required')
        prepare(a)
    elif a.worker:worker(a.worker.resolve())
    else:
        root=a.launch.resolve();m=json.loads((root/'launch.json').read_text())
        if not m.get('preflight_passed'):raise RuntimeError('Preflight not passed')
        for name, digest in m['source_sha256'].items():
            if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Archived source changed: {name}')
        if (root/'status.json').exists() or m.get('supervisor_pid'):raise RuntimeError('Already launched')
        if m['device']=='cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; select a CUDA-capable environment or prepare with --device cpu')
        with (root/'supervisor.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,str(root/'source/evaluation/run_entropy_pair.py'),'--worker',str(root)],
                cwd=root/'source',env=env_for(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'experiment_dir':str(root),'supervisor_pid':proc.pid},indent=2))

if __name__=='__main__':main()
