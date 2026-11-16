"""Target-balanced description games grounded in four public grid destinations.

No word labels, word positions or semantic decoder heads are supplied. An
observer privately knows the rewarded room description; a collector sees four
public room descriptions. Every sampled candidate set is reused with EACH of
its four members as the private target, so a no-message policy scores exactly
25% on the paired deterministic evaluation. Grid route execution is supplied,
separate from communication learning, and is not learned motor control.
"""
from pathlib import Path
import itertools,json
import numpy as np
import torch
import torch.nn.functional as F

ARITIES={'attributes':(8,8,4,2,2),'binding':(6,4,2,6,4,2,2)}
FOODS=('apple','berries','grain','mushroom','banana','bread','carrot','nuts')
COLOURS=('red','green','yellow','blue','purple','orange','white','brown')
ENTITIES=('apple','berries','grain','mushroom','lion','tiger')
ROOMS=((1,1),(11,1),(1,11),(11,11));START=(6,6)

def mixed_codes(facts,arities):
    value=torch.zeros(len(facts),dtype=torch.long,device=facts.device)
    for j,n in enumerate(arities):value=value*n+facts[:,j]
    return value

def split_mask(facts,domain,shift=1):
    if domain=='attributes':
        delta=(facts[:,1]-facts[:,0])%8
        return (delta==shift%8)|(delta==(shift+3)%8)
    # Reversing the physical roles of a held-out scene produces a training
    # scene: colour difference1 becomes3. Every primitive is seen in both roles.
    return (facts[:,1]-facts[:,4])%4==shift%4

def valid(facts,domain):
    if domain=='attributes':return torch.ones(len(facts),dtype=torch.bool)
    return ~(facts[:,:3]==facts[:,3:6]).all(-1)

def rectangular_sets(arities,facts,partition):
    """All complete two-variable rectangles with four valid same-split corners."""
    full=torch.cartesian_prod(*(torch.arange(n) for n in arities));codes=mixed_codes(full,arities)
    lookup=torch.full((len(full),),-1,dtype=torch.long);lookup[mixed_codes(facts,arities)]=torch.arange(len(facts))
    strides=[];s=1
    for n in reversed(arities):strides.insert(0,s);s*=n
    train=[];test=[];train_dims=[];test_dims=[]
    for i,j in itertools.combinations(range(len(arities)),2):
        base=codes[(full[:,i]==0)&(full[:,j]==0)]
        pi=torch.tensor(list(itertools.combinations(range(arities[i]),2)))
        pj=torch.tensor(list(itertools.combinations(range(arities[j]),2)))
        b,ii,jj=torch.meshgrid(torch.arange(len(base)),torch.arange(len(pi)),torch.arange(len(pj)),indexing='ij')
        b,ii,jj=b.flatten(),ii.flatten(),jj.flatten()
        corner=torch.stack([base[b]+pi[ii,a]*strides[i]+pj[jj,c]*strides[j] for a,c in ((0,0),(1,0),(0,1),(1,1))],-1)
        ids=lookup[corner];ok=(ids>=0).all(-1);ids=ids[ok]
        held=partition[ids];tr=ids[~held.any(-1)];te=ids[held.all(-1)]
        train.append(tr);test.append(te)
        train_dims.append(torch.tensor([i,j]).expand(len(tr),2));test_dims.append(torch.tensor([i,j]).expand(len(te),2))
    return torch.cat(train),torch.cat(test),torch.cat(train_dims),torch.cat(test_dims)

def binding_features(objects,xy):
    """Objects are [...,number,3], coordinates [...,number,2]; zero padding later."""
    return torch.cat((F.one_hot(objects[...,0],6),F.one_hot(objects[...,1],4),F.one_hot(objects[...,2],2),xy),-1).float()

def features(facts,domain):
    if domain=='attributes':
        return torch.cat([F.one_hot(facts[:,j],n) for j,n in enumerate(ARITIES[domain])],-1).float()[:,None,:]
    objects=facts[:,:6].reshape(-1,2,3);xy=torch.zeros(len(facts),2,2)
    horizontal=facts[:,6]==0
    xy[horizontal,0,0]=-1;xy[horizontal,1,0]=1
    xy[~horizontal,0,1]=-1;xy[~horizontal,1,1]=1
    return F.pad(binding_features(objects,xy),(0,0,0,1))

def single_features(facts):
    coords=torch.tensor([[-1,0],[1,0],[0,-1],[0,1],[0,0]]).float()
    return F.pad(binding_features(facts[:,:3,None].transpose(1,2),coords[facts[:,3]][:,None,:]),(0,0,0,2))

def build_cache(domain,shift,path):
    torch.set_num_threads(1);arity=ARITIES[domain]
    facts=torch.cartesian_prod(*(torch.arange(n) for n in arity));facts=facts[valid(facts,domain)]
    held=split_mask(facts,domain,shift)
    tr,te,td,ed=rectangular_sets(arity,facts,held)
    data=dict(domain=domain,shift=shift,facts=facts,held=held,features=features(facts,domain),train=tr,test=te,train_dims=td,test_dims=ed)
    if domain=='binding':
        one=torch.cartesian_prod(torch.arange(6),torch.arange(4),torch.arange(2),torch.arange(5))
        rect,_,_,_=rectangular_sets((6,4,2,5),one,torch.zeros(len(one),dtype=torch.bool))
        data.update(single_facts=one,single_features=single_features(one),single_rects=rect)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(data,path)
    return data

class Game:
    def __init__(self,cache,seed=0,device='cpu'):
        self.data=torch.load(cache,map_location='cpu',weights_only=False);self.domain=self.data['domain'];self.shift=self.data['shift']
        self.device=device;self.generator=torch.Generator(device=device).manual_seed(seed)
        self.features=self.data['features'].to(device);self.train=self.data['train'].to(device);self.test=self.data['test'].to(device)
        self.curriculum={}
        if self.domain=='attributes':
            dims=self.data['train_dims']
            for active in (2,3,4):self.curriculum[active]=self.train[dims.max(-1).values<active]
        else:self.single=self.data['single_features'].to(device);self.single_rects=self.data['single_rects'].to(device)

    def from_rectangles(self,rectangle_ids,table=None):
        table=self.features if table is None else table
        blocks=len(rectangle_ids);dev=self.device;g=self.generator
        order=torch.rand(blocks,4,generator=g,device=dev).argsort(-1)
        ids=rectangle_ids.gather(1,order)
        candidates=table[ids].repeat_interleave(4,0)
        target=torch.arange(4,device=dev).repeat(blocks)
        goal_ids=ids.flatten();private=table[goal_ids]
        return dict(private=private,candidates=candidates,target=target,goal_ids=goal_ids)

    def sample(self,batch=256,partition='train',curriculum=None):
        assert batch%4==0
        table=self.features
        if curriculum=='single' and self.domain=='binding':rectangles=self.single_rects;table=self.single
        elif curriculum in self.curriculum:rectangles=self.curriculum[curriculum]
        else:rectangles=self.train if partition=='train' else self.test
        ids=torch.randint(len(rectangles),(batch//4,),generator=self.generator,device=self.device)
        return self.from_rectangles(rectangles[ids],table)

    def codebook(self):return self.features

    def sample_longer(self,blocks=256):
        """Novel three-object scenes, balanced four-way targets; no training use."""
        assert self.domain=='binding';g=self.generator;dev=self.device;collected=[]
        arity=(6,4,2)*3+(2,)
        while len(collected)<blocks:
            b=min(256,blocks-len(collected));base=torch.stack([torch.randint(n,(b,),generator=g,device=dev) for n in arity],-1)
            varying=torch.rand(b,10,generator=g,device=dev).argsort(-1)[:,:2]
            alternatives=base.clone()
            for j in range(b):
                for d in varying[j]:alternatives[j,d]=(base[j,d]+torch.randint(1,arity[int(d)],(),generator=g,device=dev))%arity[int(d)]
            corners=[]
            for x,y in ((0,0),(1,0),(0,1),(1,1)):
                row=base.clone()
                for k,on in enumerate((x,y)):
                    if on:row[torch.arange(b),varying[:,k]]=alternatives[torch.arange(b),varying[:,k]]
                corners.append(row)
            fact=torch.stack(corners,1);obj=fact[...,:9].reshape(b,4,3,3)
            distinct=torch.ones(b,dtype=torch.bool,device=dev)
            for i,j in ((0,1),(0,2),(1,2)):distinct&=~(obj[:,:,i]==obj[:,:,j]).all(-1).any(-1)
            for f in fact[distinct]:collected.append(f)
        fact=torch.stack(collected[:blocks]);obj=fact[...,:9].reshape(blocks,4,3,3)
        coords=torch.zeros(blocks,4,3,2,device=dev);values=torch.tensor([-1.,0.,1.],device=dev)
        horizontal=fact[...,9]==0
        coords[...,0]=torch.where(horizontal[:,:,None],values[None,None,:],0.)
        coords[...,1]=torch.where(~horizontal[:,:,None],values[None,None,:],0.)
        feat=binding_features(obj,coords)
        candidates=feat.repeat_interleave(4,0);target=torch.arange(4,device=dev).repeat(blocks)
        private=feat.flatten(0,1)
        return dict(private=private,candidates=candidates,target=target,goal_ids=torch.full((blocks*4,),-1,device=dev))

def grid_execute(room):
    """Supplied public cardinal route; no learned path-planning claim."""
    x,y=START;gx,gy=ROOMS[int(room)];path=[(x,y)]
    while x!=gx:x+=1 if gx>x else -1;path.append((x,y))
    while y!=gy:y+=1 if gy>y else -1;path.append((x,y))
    return path

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--shift',type=int,default=1);a=p.parse_args()
    for domain in ARITIES:
        d=build_cache(domain,a.shift,a.output/f'{domain}-shift{a.shift}.pt')
        print(json.dumps(dict(domain=domain,meanings=len(d['facts']),train_meanings=int((~d['held']).sum()),test_meanings=int(d['held'].sum()),train_rectangles=len(d['train']),test_rectangles=len(d['test']))))
