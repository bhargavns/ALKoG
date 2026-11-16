"""Independent retained-artifact checks and per-pair order evidence."""
from pathlib import Path
import argparse,collections,csv,hashlib,json
import numpy as np
import torch
from .task import features
from .evaluate import load_models,predictions

def run(study,protocol,transfer,output):
    root=Path(study);out=Path(output);out.mkdir(parents=True,exist_ok=False);spec=json.loads(Path(protocol).read_text());checks=[];examples=[]
    expected={(d,m,s) for d in ('attributes','binding') for m in spec['methods'] for s in spec['seeds']}|{('binding','invariance',s) for s in spec['seeds']}
    found=set();torch.set_num_threads(1)
    for folder in sorted(p for p in root.glob('*_*_*') if p.is_dir()):
        c=json.loads((folder/'config.json').read_text());identity=c['domain'],c['method'],c['seed'];found.add(identity)
        assert c['updates']==6000 and c['game_episodes']==1536000 and c['shift']==1
        for file,sha in spec['frozen_source_sha256'].items():assert hashlib.sha256((folder/'source'/file).read_bytes()).hexdigest()==sha
        curve=json.loads((folder/'curve.json').read_text());assert curve[-1]['update']==6000
        assert all(np.isfinite(row['loss']) for row in curve)
        cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(value).all() for value in cp['models'].values())
        e=json.loads((folder/'evaluation/evaluation.json').read_text())
        assert e['train']['mean']['blocked']==.25 and e['test']['mean']['blocked']==.25
        checks.append(dict(run=folder.name,source_exact=True,final_update=6000,finite=True,blocked_exact_chance=True,checkpoint_sha256=hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest()))
        if c['domain']=='binding' and c['method'] in ('parallel','invariance'):
            with (folder/'evaluation/codebook.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            facts=[tuple(int(row[f'fact{j}']) for j in range(7)) for row in rows];lookup={f:i for i,f in enumerate(facts)}
            words=[tuple(map(int,row['symbols'].split())) for row in rows];pairs=[]
            for i,(f,row) in enumerate(zip(facts,rows)):
                if row['held_out']!='1':continue
                j=lookup[f[3:6]+f[:3]+f[6:]]
                assert rows[j]['held_out']=='0'
                if words[i]!=words[j] and collections.Counter(words[i])==collections.Counter(words[j]):pairs.append((i,j))
            assert len(pairs)==e['binding']['same_bag_different_order_pairs']
            models,_=load_models(folder/'checkpoint.pt');model=models[0];correct=0;sorted_correct=0
            if pairs:
                scene=features(torch.tensor(facts),'binding');candidate=torch.stack([scene[torch.tensor(pair)] for pair in pairs]).repeat_interleave(2,0)
                token=torch.tensor([words[i] for pair in pairs for i in pair]);target=torch.tensor([0,1]).repeat(len(pairs))
                normal=(predictions(model.listener,candidate,token)==target).reshape(-1,2);sorted_out=(predictions(model.listener,candidate,token.sort(-1).values)==target).reshape(-1,2)
                correct=int(normal.all(-1).sum());sorted_correct=int(sorted_out.all(-1).sum())
                with (out/f'{folder.name}-same-bag-pairs.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['heldout_id','training_counterpart_id','heldout_facts','counterpart_facts','heldout_symbols','counterpart_symbols','both_correct_normal','both_correct_sorted'])
                    for k,(i,j) in enumerate(pairs):
                        w.writerow([i,j,str(facts[i]),str(facts[j]),str(words[i]),str(words[j]),int(normal[k].all()),int(sorted_out[k].all())])
                        if c['seed']==0 and c['method']=='invariance' and len(examples)<3 and normal[k].all():examples.append(dict(heldout_facts=facts[i],counterpart_facts=facts[j],heldout_symbols=words[i],counterpart_symbols=words[j]))
            assert correct==e['binding']['same_bag_both_correct'] and sorted_correct==e['binding']['same_bag_both_correct_after_sort']
    assert found==expected,(found^expected)
    transfer=Path(transfer);result=json.loads((transfer/'results.json').read_text());physical=collections.Counter()
    for file in transfer.glob('*-episodes.json'):
        scene_path=file.with_name('-'.join(file.stem.split('-')[:2])+'-scenes.json');scene=json.loads(scene_path.read_text());episodes=json.loads(file.read_text())
        for r in episodes:
            i=r['index'];facts=scene['facts'][i];candidate=scene['candidates'][i][r['food']]
            correct=candidate==facts[:2];risk_gate=0 if scene['swap'][i]==facts[2] else 1;safe=not(facts[3] and r['route']==risk_gate)
            assert correct==r['food_correct'] and safe==r['route_safe'] and (correct and safe)==r['expected_success']
            assert r['success']==bool(correct and safe) or r['execution_error']
            physical['episodes']+=1;physical['execution_errors']+=r['execution_error'];physical['timeouts']+=r['timeout']
    assert physical['episodes']==12800
    result=dict(runs=len(checks),all_65_final_runs_retained=True,source_and_finiteness_checks=checks,order_evidence_replayed_from_csv=True,examples=examples,
                arena_logical_outcomes=dict(physical),arena_audit_scope='Independently recomputes food/route logical outcomes from public scene records. Does not independently audit swept geometry or reproduce trajectories.',frozen_sources_unchanged=all(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==sha for name,sha in spec['frozen_source_sha256'].items()))
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_and_finiteness_checks'}));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--transfer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.study,a.protocol,a.transfer,a.output)
