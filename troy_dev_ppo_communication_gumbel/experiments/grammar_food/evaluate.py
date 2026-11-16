"""Frozen-policy task, semantic, information and intervention evaluation."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
import torch
from .models import Pair
from .task import Sampler,all_facts,held_out,outcome,oracle,ARITIES,execute_path,exhaustive_contexts

def load(folder,device='cpu'):
    checkpoint=torch.load(Path(folder)/'checkpoint.pt',map_location=device,weights_only=False)
    config=checkpoint['config'];model=Pair(config['channel'],config['width']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    return model,config

def entropy(values):
    _,counts=np.unique(values,return_counts=True);p=counts/counts.sum()
    return float(-(p*np.log2(p)).sum())

def mi(a,b):
    return entropy(a)+entropy(b)-entropy(np.array([f'{x},{y}' for x,y in zip(a,b)]))

def rank(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');out=np.empty(len(x),float);j=0
    while j<len(x):
        k=j+1
        while k<len(x) and x[order[k]]==x[order[j]]:k+=1
        out[order[j:k]]=(j+k-1)/2;j=k
    return out

def edit(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a):
        current=[i+1]
        for j,y in enumerate(b):current.append(min(current[-1]+1,previous[j+1]+1,previous[j]+int(x!=y)))
        previous=current
    return previous[-1]

def language(model,config):
    facts=all_facts(next(model.parameters()).device)
    with torch.no_grad():
        words=model.speak(facts,True)[0]
        predictions=torch.stack([p.argmax(-1) for p in model.listen(words)],-1)
    x=facts.cpu().numpy();m=words.cpu().numpy();pred=predictions.cpu().numpy()
    matrix=np.array([[mi(m[:,j],x[:,a]) for a in range(4)] for j in range(m.shape[1])])
    posdis=[]
    for j in range(m.shape[1]):
        h=entropy(m[:,j]);v=np.sort(matrix[j]);posdis.append(float((v[-1]-v[-2])/h) if h>1e-10 else 0.)
    xd=[];md=[]
    for i in range(len(x)):
        for j in range(i):xd.append(np.mean(x[i]!=x[j]));md.append(edit(m[i],m[j]))
    xr,mr=rank(xd),rank(md)
    topographic=float(np.corrcoef(xr,mr)[0,1]) if np.std(mr)>0 else 0.
    test=held_out(facts,config['split'],config['shift']).cpu().numpy();train=~test
    result=dict(unique_messages=len(np.unique(m,axis=0)),topographic_similarity=topographic,
                positional_disentanglement=float(np.mean(posdis)),mutual_information=matrix.tolist(),
                semantic_accuracy_all=float((pred==x).mean()),semantic_exact_all=float((pred==x).all(-1).mean()),
                semantic_exact_train=float((pred[train]==x[train]).all(-1).mean()),
                semantic_exact_heldout=float((pred[test]==x[test]).all(-1).mean()),
                semantic_per_attribute=(pred==x).mean(0).tolist(),codebook=[dict(facts=a.tolist(),words=b.tolist(),decoded=c.tolist(),heldout=bool(t)) for a,b,c,t in zip(x,m,pred,test)])
    if m.shape[1]==4 and model.channel!='silent':
        # Assignment is estimated from TRAIN meanings only, not test labels.
        trainmi=np.array([[mi(m[train,j],x[train,a]) for a in range(4)] for j in range(4)])
        assignment=max(itertools.permutations(range(4)),key=lambda p:sum(trainmi[p[a],a] for a in range(4)))
        edited=[];truth=[];sources=[]
        for i in np.flatnonzero(test):
            row=m[i].copy()
            for a in range(4):
                options=np.flatnonzero(train&(x[:,a]==x[i,a]))
                donor=max(options,key=lambda k:np.count_nonzero(x[k]!=x[i]))
                row[assignment[a]]=m[donor,assignment[a]]
            edited.append(row);truth.append(x[i]);sources.append(int(i))
        with torch.no_grad():
            decoded=torch.stack([p.argmax(-1) for p in model.listen(torch.tensor(np.array(edited),device=facts.device))],-1).cpu().numpy()
        result.update(train_fitted_attribute_to_position=list(assignment),
                      recombination_exact=float((decoded==np.array(truth)).all(-1).mean()),
                      recombination_per_attribute=(decoded==np.array(truth)).mean(0).tolist())
    if model.channel in ('slots','tied_slots'):
        result['sender_role_assignment']=(model.slot_assignment() if model.channel=='tied_slots' else model.sender_slots.softmax(-1)).detach().cpu().tolist()
        result['receiver_role_assignment']=(model.slot_assignment().T if model.channel=='tied_slots' else model.receiver_slots.softmax(-1)).detach().cpu().tolist()
    train_messages={tuple(row) for row in m[train]}
    result['heldout_messages_unseen_in_training']=float(np.mean([tuple(row) not in train_messages for row in m[test]]))
    if m.shape[1]==4 and model.channel!='silent':
        transplant=[];expected=[];which=[]
        for i in np.flatnonzero(test):
            for a,n in enumerate(ARITIES):
                for value in range(n):
                    if value==x[i,a]:continue
                    options=np.flatnonzero(train&(x[:,a]==value))
                    donor=max(options,key=lambda k:np.count_nonzero(x[k]!=x[i]))
                    row=m[i].copy();row[assignment[a]]=m[donor,assignment[a]]
                    target=x[i].copy();target[a]=value
                    transplant.append(row);expected.append(target);which.append(a)
        with torch.no_grad():
            decoded=torch.stack([p.argmax(-1) for p in model.listen(torch.tensor(np.array(transplant),device=facts.device))],-1).cpu().numpy()
        expected=np.array(expected);which=np.array(which)
        result['single_token_edit_exact']=float((decoded==expected).all(-1).mean())
        result['single_token_edit_by_attribute']=[float((decoded[which==a]==expected[which==a]).all(-1).mean()) for a in range(4)]
    return result

@torch.no_grad()
def evaluate(model,config,n=16384,seed=82000019,exhaustive=True):
    device=next(model.parameters()).device
    sampler=Sampler(seed,str(device),config['split'],config['shift'],partition='all')
    # Balanced across all 64 meanings, with independent random public contexts.
    if exhaustive:batch=exhaustive_contexts(device);facts=batch['facts'];n=len(facts)
    else:
        facts=all_facts(device).repeat((n+63)//64,1)[:n]
        batch=sampler.for_facts(facts)
    batch['query_game']=torch.zeros(n,device=device,dtype=torch.bool)
    basewords=model.speak(facts,True)[0]
    partition=held_out(facts,config['split'],config['shift'])
    rows=[];normal=None
    shuffle=torch.randperm(n,generator=torch.Generator(device=device).manual_seed(seed+991),device=device)
    for intervention in ('normal','blocked','constant','shuffle','reverse','permute_vocabulary'):
        if intervention=='normal':words=basewords
        elif intervention=='blocked':words=torch.full_like(basewords,model.vocab)
        elif intervention=='constant':words=torch.zeros_like(basewords)
        elif intervention=='shuffle':words=basewords[shuffle]
        elif intervention=='reverse':words=basewords.flip(-1)
        else:words=(basewords+1)%model.vocab
        if model.channel=='silent':words=basewords
        action=model.act(batch,True,words_override=words)
        metrics=outcome(batch,action['food'],action['route'],0 if model.channel=='silent' else 8)
        if intervention=='normal':normal=(action,metrics)
        for label,mask in [('all',torch.ones(n,device=device,dtype=torch.bool)),('train_meanings',~partition),('heldout_meanings',partition)]:
            row=dict(partition=label,intervention=intervention,episodes=int(mask.sum()))
            for key,value in metrics.items():row[key]=float(value[mask].float().mean())
            danger=facts[:,3].bool()&mask
            row['hazard_given_free']=float(metrics['hazard'][danger].float().mean())
            rows.append(row)
    _,_,best=oracle(batch)
    for label,mask in [('all',torch.ones(n,device=device,dtype=torch.bool)),('train_meanings',~partition),('heldout_meanings',partition)]:
        rows.append(dict(partition=label,intervention='oracle',episodes=int(mask.sum()),**{k:float(v[mask].float().mean()) for k,v in best.items()}))
    action,metrics=normal;examples=[]
    for i in range(min(128,n)):
        example=execute_path(facts[i].tolist(),batch['candidates'][i].tolist(),int(batch['swap'][i]),int(action['food'][i]),int(action['route'][i]))
        assert example['success']==bool(metrics['success'][i])
        assert example['hazard']==bool(metrics['hazard'][i])
        assert abs(example['reward']-float(metrics['reward'][i]))<(0.009 if model.channel=='silent' else 1e-5)
        examples.append(dict(facts=facts[i].tolist(),candidates=batch['candidates'][i].tolist(),target=int(batch['target'][i]),
                             swap=int(batch['swap'][i]),words=basewords[i].tolist(),chosen_food=int(action['food'][i]),
                             chosen_route=int(action['route'][i]),**example))
    # Strong no-message reference allowed to know which target combinations
    # belong to each evaluation stratum. This catches target-prior shortcuts.
    silent_bounds={}
    for label,want_heldout in [('train_meanings',False),('heldout_meanings',True)]:
        weights=torch.zeros(n,4,device=device)
        for hazard in range(4):
            for c in range(4):
                hypothetical=torch.cat((batch['candidates'][:,c],torch.full((n,1),hazard//2,device=device),
                                        torch.full((n,1),hazard%2,device=device)),-1)
                weights[:,c]+=(held_out(hypothetical,config['split'],config['shift'])==want_heldout).float()
        ceiling=weights.max(-1).values/weights.sum(-1)
        mask=partition if want_heldout else ~partition
        silent_bounds[label]=float(ceiling[mask].mean())
    silent_bounds['all']=.25
    return dict(rows=rows,language=language(model,config),examples=examples,silent_food_ceiling=silent_bounds,
                exact_context_enumeration=exhaustive)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--episodes',type=int,default=16384)
    p.add_argument('--sample-contexts',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);m,c=load(a.run)
    result=evaluate(m,c,a.episodes,exhaustive=not a.sample_contexts)
    (a.run/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(config=c,normal=[r for r in result['rows'] if r['intervention']=='normal'],
                          language={k:v for k,v in result['language'].items() if k not in ('codebook','mutual_information')})),flush=True)

if __name__=='__main__':main()
