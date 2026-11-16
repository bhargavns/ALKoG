"""Explicitly exploratory, decision-only hazard counterfactuals after transfer."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .evaluate import load_models,packets
from .task import features,split_mask
from .arena_transfer import new_facts,select
from ..grammar_arena.evaluate import cases

@torch.no_grad()
def run(study,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);rows=[]
    protocol=dict(status='Exploratory after the primary MuJoCo evaluation; no model selection or training',seeds=list(range(5)),conditions=['normal','flip_state','flip_animal'],families=['standard','transformed'],
                  intervention='Re-encode the whole scene with one private hazard fact changed. No token positions or word meanings assigned.',scope='Decision-only expected outcomes. These are NOT additional executed MuJoCo episodes.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for seed in range(5):
        model=load_models(Path(study)/f'attributes_parallel_{seed}/checkpoint.pt')[0][0]
        for family in protocol['families']:
            batch,g,y=cases('food_pair',1,4,9600001,family);actual=new_facts(batch['facts']);held=split_mask(actual,'attributes',1).numpy();normal=None
            for condition in protocol['conditions']:
                sent=actual.clone()
                if condition=='flip_state':sent[:,4]=1-sent[:,4]
                if condition=='flip_animal':sent[:,3]=1-sent[:,3]
                words=packets(model.speaker,features(sent,'attributes'));choice=select(model,batch,words,g,y)
                food=choice['food'];route=choice['route'];correct=food==batch['target'].numpy()
                risk=np.where(batch['swap'].numpy()==batch['facts'][:,2].numpy(),0,1)
                hit=batch['facts'][:,3].numpy().astype(bool)&(route==risk);success=correct&~hit
                if normal is None:normal=choice
                records=[dict(index=i,actual_facts=actual[i].tolist(),sent_facts=sent[i].tolist(),words=words[i].tolist(),food=int(food[i]),route=int(route[i]),food_correct=bool(correct[i]),logical_hazard=bool(hit[i]),logical_success=bool(success[i]),route_changed=bool(route[i]!=normal['route'][i]),food_changed=bool(food[i]!=normal['food'][i])) for i in range(len(words))]
                (out/f'seed{seed}-{family}-{condition}.json').write_text(json.dumps(records))
                for part,mask in [('all',np.ones(len(held),bool)),('heldout',held),('actual_free',batch['facts'][:,3].numpy()==1),('actual_caged',batch['facts'][:,3].numpy()==0)]:
                    subset=[r for r,m in zip(records,mask) if m]
                    rows.append(dict(seed=seed,family=family,condition=condition,partition=part,n=len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in ('food_correct','logical_hazard','logical_success','route_changed','food_changed')}))
    result=dict(protocol=protocol,rows=rows);(out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows[-4:]));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.study,a.output)
