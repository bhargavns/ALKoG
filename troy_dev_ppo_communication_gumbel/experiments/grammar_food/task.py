"""Sampled food and hazard games, with explicit observation boundaries.

Private facts: food kind, food colour, potentially dangerous animal, free flag.
The observer sees these four attributes. The collector sees a 2x2 rectangle of
food candidates in shuffled public rooms and public animal positions. It gets
no private target/status field. Planning geometry is supplied, not learned.
"""
from collections import deque
import itertools
import torch
import torch.nn.functional as F

ARITIES=(4,4,2,2)
FOODS=('apple','berries','grain','mushroom')
COLOURS=('red','green','yellow','blue')
ANIMALS=('lion','tiger')
START=(1,4)
GOALS=((9,1),(9,3),(9,5),(9,7))
GATES=((5,1),(5,4),(5,8))
WIDTH,HEIGHT=11,10
TIME_COST=.02
BIT_COST=.001

def one_hot_facts(facts):
    return torch.cat([F.one_hot(facts[:,i],n).float() for i,n in enumerate(ARITIES)],-1)

def all_facts(device='cpu'):
    return torch.cartesian_prod(*(torch.arange(n,device=device) for n in ARITIES))

def held_out(facts,split='context',shift=0):
    if split=='context':
        hazard=2*facts[:,2]+facts[:,3]
        return facts[:,1]==(facts[:,0]+hazard+shift)%4
    if split=='food_pair':
        return facts[:,1]==(facts[:,0]+shift)%4
    raise ValueError(split)

def plan(start,goal,blocked,width=WIDTH,height=HEIGHT):
    frontier=deque([tuple(start)]);parents={tuple(start):None};blocked=set(blocked)
    while frontier:
        p=frontier.popleft()
        if p==tuple(goal):
            path=[]
            while p is not None:path.append(p);p=parents[p]
            return path[::-1]
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            q=(p[0]+dx,p[1]+dy)
            if 0<=q[0]<width and 0<=q[1]<height and q not in blocked and q not in parents:
                parents[q]=p;frontier.append(q)
    raise ValueError('Unreachable route')

def standard_paths():
    wall={(5,y) for y in range(HEIGHT)}-set(GATES)
    paths=[]
    for goal in GOALS:
        routes=[]
        for gate in GATES:
            blocked=wall|(set(GATES)-{gate})
            first=plan(START,gate,blocked)
            last=plan(gate,goal,blocked)
            routes.append(first+last[1:])
        paths.append(routes)
    return paths

PATHS=standard_paths()
LENGTHS=tuple(tuple(len(path)-1 for path in paths) for paths in PATHS)
HIT_STEPS=(len(plan(START,GATES[0],{(5,y) for y in range(HEIGHT)}-{GATES[0]}))-1,
           len(plan(START,GATES[1],{(5,y) for y in range(HEIGHT)}-{GATES[1]}))-1)

class Sampler:
    def __init__(self,seed,device='cpu',split='context',shift=0,partition='train'):
        self.device=torch.device(device);self.generator=torch.Generator(device=device).manual_seed(seed)
        self.split=split;self.shift=shift
        facts=all_facts(device);test=held_out(facts,split,shift)
        self.facts=facts if partition=='all' else facts[test if partition=='test' else ~test]
        self.lengths=torch.tensor(LENGTHS,device=device).float()

    def sample(self,n):
        g=self.generator;dev=self.device
        facts=self.facts[torch.randint(len(self.facts),(n,),generator=g,device=dev)]
        return self.for_facts(facts)

    def for_facts(self,facts):
        n=len(facts);g=self.generator;dev=self.device
        # Rectangular candidate sets avoid a target-specific minimal-pair pattern.
        other_kind=(facts[:,0]+torch.randint(1,4,(n,),generator=g,device=dev))%4
        other_colour=(facts[:,1]+torch.randint(1,4,(n,),generator=g,device=dev))%4
        candidates=torch.stack((facts[:,:2],torch.stack((facts[:,0],other_colour),-1),
                                torch.stack((other_kind,facts[:,1]),-1),
                                torch.stack((other_kind,other_colour),-1)),1)
        permutation=torch.rand(n,4,generator=g,device=dev).argsort(-1)
        candidates=candidates.gather(1,permutation[:,:,None].expand(-1,-1,2))
        target=(permutation==0).long().argmax(-1)
        swap=torch.randint(2,(n,),generator=g,device=dev)
        query=torch.randint(4,(n,),generator=g,device=dev)
        return dict(facts=facts,candidates=candidates,target=target,swap=swap,query=query,
                    lengths=self.lengths[None].expand(n,-1,-1))

def exhaustive_contexts(device='cpu'):
    """All 64 meanings x 9 rectangles x 24 room permutations x 2 guard layouts.

    Evaluation only. The training sampler never calls this function.
    """
    conditions=torch.tensor([(k,c,*p,s) for k in (1,2,3) for c in (1,2,3)
                             for p in itertools.permutations(range(4)) for s in (0,1)],device=device)
    facts=all_facts(device).repeat_interleave(len(conditions),0)
    cond=conditions.repeat(64,1);n=len(facts)
    ok=(facts[:,0]+cond[:,0])%4;oc=(facts[:,1]+cond[:,1])%4
    candidates=torch.stack((facts[:,:2],torch.stack((facts[:,0],oc),-1),
                            torch.stack((ok,facts[:,1]),-1),torch.stack((ok,oc),-1)),1)
    perm=cond[:,2:6];candidates=candidates.gather(1,perm[:,:,None].expand(-1,-1,2))
    return dict(facts=facts,candidates=candidates,target=(perm==0).long().argmax(-1),swap=cond[:,6],
                query=torch.zeros(n,dtype=torch.long,device=device),
                lengths=torch.tensor(LENGTHS,device=device).float()[None].expand(n,-1,-1))

def route_risks(facts,swap):
    animals=torch.stack((swap,1-swap),-1)
    guard=(animals==facts[:,2,None])&facts[:,3,None].bool()
    return F.pad(guard,(0,1),value=False)

@torch.no_grad()
def outcome(batch,food,route,bits=8):
    """Only chosen actions receive an environmental outcome; no expected rewards."""
    n=len(food);ids=torch.arange(n,device=food.device)
    risky=route_risks(batch['facts'],batch['swap'])
    hit=risky[ids,route]
    right_food=food==batch['target']
    success=right_food&~hit
    lengths=batch['lengths'][ids,food,route]
    stops=torch.tensor((*HIT_STEPS,0),device=food.device).float()[route]
    steps=torch.where(hit,stops,lengths)
    reward=success.float()-hit.float()-TIME_COST*steps-BIT_COST*bits
    safe_lengths=batch['lengths'][ids,food].masked_fill(risky,1e6)
    best=safe_lengths.min(-1).values
    return dict(reward=reward,success=success,food_correct=right_food,hazard=hit,steps=steps,
                optimal_route=(lengths==best)&~hit,safe_excess_steps=torch.where(hit,0.,lengths-best))

@torch.no_grad()
def oracle(batch,bits=0):
    ids=torch.arange(len(batch['facts']),device=batch['facts'].device)
    lengths=batch['lengths'][ids,batch['target']]
    risky=route_risks(batch['facts'],batch['swap'])
    route=lengths.masked_fill(risky,1e6).argmin(-1)
    return batch['target'],route,outcome(batch,batch['target'],route,bits)

def execute_path(facts,candidates,swap,food,route):
    """Independent cell-by-cell execution for a selected frozen policy action."""
    path=PATHS[food][route];risk_cell=None
    if facts[3]:risk_cell=GATES[0 if facts[2]==swap else 1]
    history=[];total=-8*BIT_COST;hit=False
    true_pair=tuple(facts[:2]);correct_food=tuple(candidates[food])==true_pair
    for t,p in enumerate(path[1:]):
        reward=-TIME_COST;hit=p==risk_cell
        success=t==len(path)-2 and correct_food and not hit
        if hit:reward-=1
        if success:reward+=1
        total+=reward;history.append(dict(t=t+1,position=p,reward=reward))
        if hit:break
    return dict(success=correct_food and not hit,hazard=hit,steps=len(history),
                reward=total,trajectory=history)
