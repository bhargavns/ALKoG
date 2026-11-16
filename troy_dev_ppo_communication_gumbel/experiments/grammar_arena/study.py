"""Fixed five-seed GPU navigation and frozen-protocol arena study."""
import argparse,datetime,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .arena import source_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-study',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();output=args.output.resolve();source=args.source_study.resolve();protocol_path=args.protocol.resolve()
    protocol=json.loads(protocol_path.read_text());output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parent;start=time.perf_counter()
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    checkpoints={}
    for split in protocol['splits']:
        for seed in protocol['seeds']:
            for method in ('modular_curriculum','silent_task'):
                name=f'{split}_{method}_{seed}';file=source/name/'checkpoint.pt'
                checkpoints[name]=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest=dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=protocol,
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),integration_source_hashes=hashes,
                  original_checkpoint_hashes=checkpoints,source_study=str(source),python=sys.executable,**source_identity())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'protocol.json').write_bytes(protocol_path.read_bytes())
    (output/'source').mkdir()
    for f in root.glob('*.py'):(output/'source'/f.name).write_bytes(f.read_bytes())
    completed=[]
    def execute(cmd,log):
        with log.open('w',encoding='utf-8') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    for seed in protocol['seeds']:
        execute([sys.executable,'-m',__package__+'.train_navigation','--seed',str(seed),'--updates',str(protocol['navigation']['updates']),
                 '--environments',str(protocol['navigation']['environments']),'--horizon',str(protocol['navigation']['horizon']),
                 '--device','cuda','--output',str(output/'navigation'/f'seed{seed}')],output/f'navigation-seed{seed}.log')
        print('TRAINED NAVIGATOR',seed,flush=True)
    def one(job):
        split,seed=job;name=f'{split}_seed{seed}'
        execute([sys.executable,'-m',__package__+'.evaluate','--protocol',str(source/f'{split}_modular_curriculum_{seed}'),
                 '--silent',str(source/f'{split}_silent_task_{seed}'),'--navigation',str(output/'navigation'/f'seed{seed}'),
                 '--output',str(output/'transfer'/name),'--repeats',str(protocol['evaluation']['contexts_per_meaning']),
                 '--seed',str(protocol['evaluation']['seed']),'--device','cuda','--conditions',*protocol['evaluation']['conditions']],output/f'{name}.log')
        result=json.loads((output/'transfer'/name/'results.json').read_text())
        return dict(name=name,rows=[r for r in result['rows'] if r['partition']=='heldout_meanings'],seconds=result['total_seconds'])
    jobs=[(split,seed) for split in protocol['splits'] for seed in protocol['seeds']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs]):
            row=future.result();completed.append(row)
            (output/'progress.json').write_text(json.dumps(dict(completed=completed,total=len(jobs),seconds=time.perf_counter()-start),indent=2))
            print('EVALUATED',row['name'],len(completed),'/',len(jobs),flush=True)
    for file,digest in hashes.items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest,file
    for name,digest in checkpoints.items():assert hashlib.sha256((source/name/'checkpoint.pt').read_bytes()).hexdigest()==digest,name
    manifest.update(completed=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,
                    evaluation_jobs=len(completed),all_protocol_checkpoints_unchanged=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
