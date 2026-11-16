"""Frozen-model causal and compositional evaluation; no checkpoint selection."""
from pathlib import Path
import argparse,collections,csv,itertools,json,math
import numpy as np
import torch
from torch import nn
from .task import Game,features,mixed_codes,ARITIES
from .models import Pair,Listener,payload,intervene

def load_models(path,device='cpu'):
    cp=torch.load(path,map_location=device,weights_only=False);c=cp['config']
    models=nn.ModuleList([Pair(c['domain'],c['width'],c['vocab'],c['max_length'],c.get('perceptual_anchoring',False),
        c.get('contextual_sender',False),c.get('position_embeddings',False),c.get('attention_sender',False),c.get('parallel_channel',False)) for _ in range(c['population'])]).to(device)
    models.load_state_dict(cp['models']);models.eval();return models,c

@torch.no_grad()
def packets(speaker,scenes,batch=256):
    return torch.cat([speaker(x)['tokens'] for x in scenes.split(batch)])

@torch.no_grad()
def predictions(listener,candidates,tokens,batch=256):
    return torch.cat([listener(c,tokens=t).argmax(-1) for c,t in zip(candidates.split(batch),tokens.split(batch))])

@torch.no_grad()
def causal(models,batch,vocab=8):
    result={};target=batch['target'];generator=torch.Generator(device=target.device).manual_seed(8100001)
    for i,s in enumerate(models):
        original=packets(s.speaker,batch['private'])
        altered={k:intervene(original,k,vocab,generator) for k in ('normal','blocked','shuffle','sort','reverse','permute_vocabulary')}
        for j,r in enumerate(models):
            result[f'{i}:{j}']={k:float((predictions(r.listener,batch['candidates'],v)==target).float().mean()) for k,v in altered.items()}
    return dict(mean={k:float(np.mean([v[k] for v in result.values()])) for k in next(iter(result.values()))},pairs=result,episodes=len(target))

def edit(a,b):
    last=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,last[j]+1,last[j-1]+(x!=y)))
        last=row
    return last[-1]

def rank(a):
    _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-counts/2)[inverse]

def mutual(a,b):
    a=np.asarray(a);b=np.asarray(b);_,a=np.unique(a,return_inverse=True);_,b=np.unique(b,return_inverse=True)
    joint=np.zeros((a.max()+1,b.max()+1));np.add.at(joint,(a,b),1);joint/=len(a)
    expected=joint.sum(1)[:,None]*joint.sum(0)[None,:];mask=joint>0
    return float((joint[mask]*np.log2(joint[mask]/expected[mask])).sum())

def code_metrics(facts,tokens,vocab=8):
    codes=[tuple(payload(t,vocab,vocab+1)) for t in tokens];counts=collections.Counter(codes)
    rng=np.random.default_rng(8200001);pairs=rng.integers(len(facts),size=(12000,2));a,b=pairs.T
    md=(facts[a]!=facts[b]).sum(-1);ld=np.array([edit(codes[i],codes[j]) for i,j in pairs])
    x,y=rank(md),rank(ld);rho=float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else 0.
    p=np.array(list(counts.values()))/len(codes)
    matrix=np.array([[*c,*([vocab]*(tokens.shape[1]-len(c)))] for c in codes])
    info=np.array([[mutual(matrix[:,j],facts[:,k]) for k in range(facts.shape[1])] for j in range(matrix.shape[1])])
    return dict(unique=len(counts),meaning_count=len(facts),mean_payload=float(np.mean(list(map(len,codes)))),code_entropy_bits=float(-(p*np.log2(p)).sum()),
                topographic_spearman=rho,position_fact_mutual_information_bits=info.tolist()),codes

@torch.no_grad()
def binding_audit(pair,game,tokens):
    facts=game.data['facts'];held=game.data['held'];ids=torch.nonzero(held,as_tuple=True)[0]
    swapped=facts[held].clone();swapped[:,:3]=facts[held,3:6];swapped[:,3:6]=facts[held,:3]
    lookup={tuple(f.tolist()):i for i,f in enumerate(facts)};other=torch.tensor([lookup[tuple(f.tolist())] for f in swapped],device=game.device)
    ids=ids.to(game.device);candidates=torch.stack((game.features[ids],game.features[other]),1).repeat_interleave(2,0)
    words=torch.stack((tokens[ids],tokens[other]),1).flatten(0,1);target=torch.tensor([0,1],device=game.device).repeat(len(ids))
    correct={k:(predictions(pair.listener,candidates,intervene(words,k,8))==target).reshape(-1,2) for k in ('normal','blocked','sort','reverse')}
    code=[tuple(payload(t,8,9)) for t in words]
    matched=torch.tensor([a!=b and sorted(a)==sorted(b) for a,b in zip(code[::2],code[1::2])],device=game.device)
    return dict(pairs=len(ids),accuracy={k:float(v.float().mean()) for k,v in correct.items()},
                both_correct={k:float(v.all(-1).float().mean()) for k,v in correct.items()},
                same_bag_different_order_pairs=int(matched.sum()),same_bag_both_correct=int((correct['normal'].all(-1)&matched).sum()),
                same_bag_both_correct_after_sort=int((correct['sort'].all(-1)&matched).sum()))

def rewrite_fragment(a,b):
    start=0
    while start<min(len(a),len(b)) and a[start]==b[start]:start+=1
    end=0
    while end<min(len(a),len(b))-start and a[len(a)-1-end]==b[len(b)-1-end]:end+=1
    old=a[start:len(a)-end if end else len(a)];new=b[start:len(b)-end if end else len(b)]
    return old,new

def replace_fragment(code,old,new):
    if not old:return None
    for j in range(len(code)-len(old)+1):
        if code[j:j+len(old)]==old:return code[:j]+new+code[j+len(old):]
    return None

@torch.no_grad()
def rewrite_audit(pair,game,codes):
    """Learn local edit rules ONLY from training meanings, apply to heldout pairs."""
    facts=game.data['facts'].tolist();held=game.data['held'].tolist();lookup={tuple(f):i for i,f in enumerate(facts)}
    proposals=collections.defaultdict(collections.Counter);held_pairs=[]
    for i,f in enumerate(facts):
        for d,n in enumerate(ARITIES[game.domain]):
            for value in range(n):
                if value==f[d]:continue
                changed=f.copy();changed[d]=value;j=lookup.get(tuple(changed))
                if j is None:continue
                key=(d,f[d],value)
                if not held[i] and not held[j]:
                    old,new=rewrite_fragment(codes[i],codes[j])
                    if old and old!=new and max(len(old),len(new))<=3:proposals[key][old,new]+=1
                elif held[i] and held[j]:held_pairs.append((i,j,key))
    rules={k:sorted(v.items(),key=lambda x:(-x[1],x[0]))[0] for k,v in proposals.items() if max(v.values())>=3}
    # Fixed pseudorandom cap controls diagnostic cost without choosing easy pairs.
    rng=np.random.default_rng(8300001)
    if len(held_pairs)>4096:held_pairs=[held_pairs[j] for j in rng.choice(len(held_pairs),4096,replace=False)]
    candidates=[];before=[];after=[];applicable=0
    def pack(c):
        if len(c)>8:return None
        return list(c)+([8]+[9]*(7-len(c)) if len(c)<8 else [])
    for i,j,key in held_pairs:
        if key not in rules:continue
        (old,new),support=rules[key];changed=replace_fragment(codes[i],old,new)
        encoded=pack(changed) if changed is not None else None
        if encoded is None:continue
        applicable+=1;candidates.append(torch.stack((game.features[i],game.features[j])))
        before.append(pack(codes[i]));after.append(encoded)
    success=0;initial=0
    if candidates:
        cand=torch.stack(candidates);a=predictions(pair.listener,cand,torch.tensor(before,device=game.device));b=predictions(pair.listener,cand,torch.tensor(after,device=game.device))
        initial=int((a==0).sum());success=int(((a==0)&(b==1)).sum())
    return dict(rules=len(rules),rule_selection='training-only, modal changed contiguous span <=3 symbols, support >=3',heldout_pairs=len(held_pairs),applicable=applicable,
                coverage=applicable/max(1,len(held_pairs)),original_correct=initial,causal_switches=success,conditional_causal_switch=success/max(1,applicable),overall_causal_switch=success/max(1,len(held_pairs)),
                examples=[dict(fact=k[0],old_value=k[1],new_value=k[2],old_symbols=list(v[0][0]),new_symbols=list(v[0][1]),train_support=v[1]) for k,v in sorted(rules.items())[:20]])

def fresh_listener(pair,game,batch,seed,width=96,updates=300):
    """A fixed-budget independent learner; 128 TRAIN meanings, never test labels."""
    torch.manual_seed(8400000+seed);listener=Listener(game.domain,width,8,anchored=True).to(game.device)
    ids=torch.nonzero(~game.data['held'],as_tuple=True)[0].to(game.device);ids=ids[torch.randperm(len(ids),device=game.device)[:128]]
    scenes=game.features[ids];words=packets(pair.speaker,scenes);opt=torch.optim.Adam(listener.parameters(),lr=.001)
    for _ in range(updates):
        chosen=torch.rand(len(ids),4,device=game.device).argsort(-1)[:,:4] if False else torch.randint(len(ids),(256,4),device=game.device)
        # Resample duplicate candidates to avoid contradictory correct labels.
        while (chosen.sort(-1).values.diff(dim=-1)==0).any():
            bad=(chosen.sort(-1).values.diff(dim=-1)==0).any(-1);chosen[bad]=torch.randint(len(ids),(int(bad.sum()),4),device=game.device)
        y=torch.randint(4,(256,),device=game.device);target=chosen[torch.arange(256,device=game.device),y]
        logits=listener(scenes[chosen],tokens=words[target]);loss=nn.functional.cross_entropy(logits,y)
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        message=packets(pair.speaker,batch['private']);score=float((predictions(listener,batch['candidates'],message)==batch['target']).float().mean())
    return dict(train_meanings=128,updates=updates,heldout_success=score,blocked_success=.25)

def evaluate(path,cache,output,blocks=1024,device='cpu',fresh=True):
    torch.set_num_threads(1);models,c=load_models(path,device);game=Game(cache,8500001,device);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    train=game.sample(blocks*4,'train');test=game.sample(blocks*4,'test');result=dict(config=c,train=causal(models,train),test=causal(models,test))
    tokens=packets(models[0].speaker,game.features);cm,codes=code_metrics(game.data['facts'].numpy(),tokens.cpu());result['code']=cm
    result['local_rewrites']=rewrite_audit(models[0],game,codes)
    if game.domain=='binding':
        result['binding']=binding_audit(models[0],game,tokens);result['three_objects']=causal(models,game.sample_longer(blocks//2))
    if fresh:result['fresh_listener']=fresh_listener(models[0],game,test,c['seed'],c['width'])
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    with (out/'codebook.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['meaning_id','held_out',*[f'fact{i}' for i in range(game.data['facts'].shape[1])],'symbols'])
        for i,(facts,held,code) in enumerate(zip(game.data['facts'].tolist(),game.data['held'].tolist(),codes)):w.writerow([i,int(held),*facts,' '.join(map(str,code))])
    print(json.dumps(dict(run=str(path.parent),test=result['test']['mean'],code=cm['unique'],binding=result.get('binding'),rewrites=result['local_rewrites']['overall_causal_switch'])),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--blocks',type=int,default=1024);p.add_argument('--device',default='cpu');p.add_argument('--skip-fresh',action='store_true');a=p.parse_args()
    evaluate(a.checkpoint,a.cache,a.output,a.blocks,a.device,not a.skip_fresh)
