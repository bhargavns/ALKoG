"""Same PPO/fixed roles with an explicit training-only deployment admission test."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
import torch
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'project'))
from experiments.grammar_food.task import all_facts,held_out
from experiments.grammar_food.evaluate import load
from experiments.grammar_food.train import run

@torch.no_grad()
def training_admission(model,config):
    facts=all_facts(next(model.parameters()).device)
    facts=facts[~held_out(facts,config['split'],config['shift'])]
    assert len(facts)==48
    words=model.speak(facts,True)[0];decoded=torch.stack([p.argmax(-1) for p in model.listen(words)],-1)
    return dict(training_meanings=48,exact=float((decoded==facts).all(-1).float().mean()),heldout_meanings_used=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan_path=ROOT/'protocol-quality-control.json';plan=json.loads(plan_path.read_text());(a.output/'protocol.json').write_bytes(plan_path.read_bytes())
    start=time.perf_counter();rows=[]
    for split,shift in (('context',0),('food_pair',1)):
        for base_seed in range(5):
            folder=a.source/f'{split}_modular_curriculum_{base_seed}';model,c=load(folder)
            initial=training_admission(model,c);attempts=[];accepted=initial['exact']==1.;accepted_folder=folder;training_seed=base_seed
            if not accepted:
                for attempt in range(3):
                    training_seed=1000*(attempt+1)+base_seed
                    candidate=a.output/'attempts'/f'{split}_pipeline{base_seed}_attempt{attempt+1}_rng{training_seed}'
                    model,c=run('modular_curriculum',training_seed,split,shift,1800,512,64,'cuda',candidate,600,2,5.)
                    score=training_admission(model,c);accepted=score['exact']==1.
                    attempts.append(dict(folder=str(candidate.resolve()),initialization_seed=training_seed,admission=score,accepted=accepted,training_seconds=c['seconds']))
                    if accepted:accepted_folder=candidate;break
            row=dict(split=split,pipeline_seed=base_seed,initial_admission=initial,attempts=attempts,accepted=accepted)
            if accepted:
                deployment=a.output/'accepted'/f'{split}_modular_curriculum_{base_seed}';deployment.mkdir(parents=True,exist_ok=False)
                cp=torch.load(accepted_folder/'checkpoint.pt',map_location='cpu',weights_only=False)
                cp['config'].update(initialization_seed=training_seed,pipeline_seed=base_seed,seed=base_seed,
                                    training_only_admission=True,source_checkpoint=str(accepted_folder.resolve()),
                                    admission=training_admission(model,c),new_grammar_architecture=False)
                if attempts:cp['cuda_rng_after_training']=torch.cuda.get_rng_state_all()
                torch.save(cp,deployment/'checkpoint.pt');(deployment/'config.json').write_text(json.dumps(cp['config'],indent=2))
                row.update(deployment=str(deployment.resolve()),deployment_sha256=hashlib.sha256((deployment/'checkpoint.pt').read_bytes()).hexdigest())
            rows.append(row);(a.output/'progress.json').write_text(json.dumps(dict(runs=rows,seconds=time.perf_counter()-start),indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='attempts'}),flush=True)
    result=dict(protocol=plan,runs=rows,seconds=time.perf_counter()-start,all_training_admitted=all(r['accepted'] for r in rows))
    (a.output/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
