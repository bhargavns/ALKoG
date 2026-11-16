"""Run predeclared final seeds, retaining every final checkpoint and failure."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(root,cache,spec,updates,blocks):
    domain,method,seed=spec;name=f'{domain}_{method}_{seed}';folder=root/name
    start=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/f'{name}.log').open('w') as log:
        cmd=[sys.executable,'-m','experiments.spontaneous_grammar.train','--cache',str(cache/f'{domain}-shift1.pt'),'--output',str(folder),
             '--method',method,'--seed',str(seed),'--updates',str(updates),'--device','cpu']
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
        cmd=[sys.executable,'-m','experiments.spontaneous_grammar.evaluate','--cache',str(cache/f'{domain}-shift1.pt'),'--checkpoint',str(folder/'checkpoint.pt'),
             '--output',str(folder/'evaluation'),'--blocks',str(blocks)]
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    return dict(name=name,seconds=time.perf_counter()-start)

def run(protocol,output,workers=6):
    protocol=Path(protocol);p=json.loads(protocol.read_text());root=Path(output);root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).parent
    for name,expected in p['frozen_source_sha256'].items():
        if digest(source/name)!=expected:raise RuntimeError(f'Frozen source changed: {name}')
    (root/'protocol.json').write_bytes(protocol.read_bytes())
    cache=root/'data';cache.mkdir()
    from .task import build_cache
    for domain in ('attributes','binding'):build_cache(domain,1,cache/f'{domain}-shift1.pt')
    specs=[(domain,method,seed) for method in p['methods'] for domain in ('binding','attributes') for seed in p['seeds']]
    specs += [('binding','invariance',seed) for seed in p['seeds']]
    completed=[];failures=[];start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(workers) as executor:
        futures={executor.submit(execute,root,cache,s,p['updates'],p['evaluation_blocks']):s for s in specs}
        for f in concurrent.futures.as_completed(futures):
            try:completed.append(f.result())
            except Exception as e:failures.append(dict(spec=futures[f],error=repr(e)))
            status=dict(completed=completed,failures=failures,total=len(specs),wall_seconds=time.perf_counter()-start)
            (root/'status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(dict(completed=len(completed),failed=len(failures),total=len(specs),last=completed[-1] if completed else None,wall_seconds=status['wall_seconds'])),flush=True)
    for name,expected in p['frozen_source_sha256'].items():
        if digest(source/name)!=expected:raise RuntimeError(f'Source changed during final runs: {name}')
    if failures:raise RuntimeError(f'{len(failures)} failed runs; all retained in status.json')
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args();run(a.protocol,a.output,a.workers)
