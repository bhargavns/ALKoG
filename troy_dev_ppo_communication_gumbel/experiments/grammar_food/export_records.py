"""Recompute frozen per-episode records and validate aggregate reports independently."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .evaluate import load
from .task import exhaustive_contexts,outcome,held_out,execute_path

@torch.no_grad()
def run(folder):
    folder=Path(folder);model,c=load(folder);batch=exhaustive_contexts();n=len(batch['facts'])
    batch['query_game']=torch.zeros(n,dtype=torch.bool)
    words=model.speak(batch['facts'],True)[0]
    permutation=torch.randperm(n,generator=torch.Generator().manual_seed(82000019+991))
    conditions=('normal','blocked','constant','shuffle','reverse','permute_vocabulary')
    arrays={};reference=json.loads((folder/'evaluation.json').read_text());test=held_out(batch['facts'],c['split'],c['shift'])
    for iv in conditions:
        if iv=='normal':signal=words
        elif iv=='blocked':signal=torch.full_like(words,model.vocab)
        elif iv=='constant':signal=torch.zeros_like(words)
        elif iv=='shuffle':signal=words[permutation]
        elif iv=='reverse':signal=words.flip(-1)
        else:signal=(words+1)%model.vocab
        if model.channel=='silent':signal=words
        a=model.act(batch,True,words_override=signal)
        metrics=outcome(batch,a['food'],a['route'],0 if model.channel=='silent' else 8)
        arrays[f'{iv}_food']=a['food'].numpy().astype('int8');arrays[f'{iv}_route']=a['route'].numpy().astype('int8')
        for k,v in metrics.items():arrays[f'{iv}_{k}']=v.numpy()
        for partition,mask in [('all',torch.ones(n,dtype=torch.bool)),('train_meanings',~test),('heldout_meanings',test)]:
            row=next(r for r in reference['rows'] if r['partition']==partition and r['intervention']==iv)
            for k,v in metrics.items():assert abs(float(v[mask].float().mean())-row[k])<1e-6,(folder.name,iv,partition,k)
    # Scalar execution examples cover every meaning, both endpoint contexts.
    checked=[]
    for meaning in range(64):
        for offset in (0,431):
            i=meaning*432+offset;food=int(arrays['normal_food'][i]);route=int(arrays['normal_route'][i])
            s=execute_path(batch['facts'][i].tolist(),batch['candidates'][i].tolist(),int(batch['swap'][i]),food,route)
            expected=float(arrays['normal_reward'][i])
            if model.channel=='silent':s['reward']+=.008
            assert abs(s['reward']-expected)<1e-6
            assert s['success']==bool(arrays['normal_success'][i])
            assert s['hazard']==bool(arrays['normal_hazard'][i])
            checked.append(dict(index=i,facts=batch['facts'][i].tolist(),words=words[i].tolist(),food=food,route=route,**s))
    arrays.update(words=words.numpy().astype('int16'),heldout=test.numpy())
    np.savez_compressed(folder/'episode-records.npz',**arrays)
    (folder/'independent-execution.json').write_text(json.dumps(checked,indent=2))
    return dict(run=folder.name,episodes=n,conditions=len(conditions),scalar_checks=len(checked),passed=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    print(json.dumps(run(a.run)),flush=True)

if __name__=='__main__':main()
