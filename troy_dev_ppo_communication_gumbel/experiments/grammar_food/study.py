"""Namespace-independent fixed sweep: python -m [experiments.]grammar_food.study."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;package=__package__
    protocol=json.loads((root/'protocol.json').read_text());budget=protocol['budget']
    sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}
    manifest=dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=protocol,source_hashes=sources,
                  protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),python=sys.executable)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'protocol.json').write_bytes((root/'protocol.json').read_bytes())
    jobs=[(s,m,seed) for s in protocol['splits'] for m in protocol['methods'] for seed in protocol['final_seeds']]
    start=time.perf_counter();completed=[]
    def one(job):
        split,method,seed=job;name=f"{split['name']}_{method}_{seed}";folder=output/name
        cmd=[sys.executable,'-m',package+'.train','--output',str(folder),'--method',method,'--seed',str(seed),
             '--split',split['name'],'--shift',str(split['shift']),'--steps',str(budget['updates']),
             '--batch-size',str(budget['batch_size']),'--warmup',str(budget['curriculum_initial_updates']),
             '--epochs',str(budget['ppo_epochs']),'--width',str(budget['hidden_width']),
             '--slot-coefficient',str(protocol['regularizers']['learned_slot_permutation_coefficient']),
             '--device',budget['device']]
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,'-m',package+'.evaluate','--run',str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
        for file,digest in sources.items():
            assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest
        return name
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one,job) for job in jobs]):
            completed.append(future.result());row=dict(completed=completed,total=len(jobs),elapsed_seconds=time.perf_counter()-start)
            (output/'progress.json').write_text(json.dumps(row,indent=2));print(f'{len(completed)}/{len(jobs)} {completed[-1]}',flush=True)
    manifest.update(completed=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=len(completed),elapsed_seconds=time.perf_counter()-start)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
