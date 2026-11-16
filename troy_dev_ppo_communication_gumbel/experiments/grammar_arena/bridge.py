"""Integer-only frozen communication, public macro decisions and gate waypoints."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from ..grammar_food.evaluate import load
from .arena import FOOD_X,FOOD_Y,GATE_Y,STEP,FOOD_BONUS,HAZARD_PENALTY,MOVE_COST,public_waypoints

def freeze_protocol(folder):
    model,config=load(folder)
    for p in model.parameters():p.requires_grad_(False)
    return model,config,hashlib.sha256((Path(folder)/'checkpoint.pt').read_bytes()).hexdigest()

def waypoints(food,route):
    return public_waypoints(food,route,include_start=True)

def public_lengths(geometry,yaw):
    rotation=float(geometry['rotation']);scale=float(geometry['scale'])
    angle=rotation-yaw;c,s=np.cos(angle),np.sin(angle)
    matrix=np.array([[c,-s],[s,c]])*scale
    lengths=np.empty((4,3));prefix=np.empty(3)
    for food in range(4):
        for route in range(3):lengths[food,route]=np.abs(np.diff(waypoints(food,route),axis=0)@matrix.T).sum()/STEP
    for route in range(3):
        points=np.array([[-4.8,-.6],[-1.1,GATE_Y[route]],[0,GATE_Y[route]]])
        prefix[route]=np.abs(np.diff(points,axis=0)@matrix.T).sum()/STEP
    return lengths,prefix

@torch.no_grad()
def choose(model,batch,words,geometries,yaws,oracle=False):
    """No private target/status input is read except the explicit oracle arm."""
    if oracle:
        facts=batch['facts'];beliefs=[torch.nn.functional.one_hot(facts[:,j],n).float() for j,n in enumerate((4,4,2,2))]
    else:beliefs=[x.softmax(-1) for x in model.listen(words)]
    cand=batch['candidates'];n=len(cand);ids=torch.arange(n)[:,None]
    food_score=beliefs[0].clamp_min(1e-9).log()[ids,cand[:,:,0]]+beliefs[1].clamp_min(1e-9).log()[ids,cand[:,:,1]]
    food=food_score.argmax(-1)
    animals=torch.stack((batch['swap'],1-batch['swap']),-1)
    danger=torch.nn.functional.pad(beliefs[2].gather(1,animals)*beliefs[3][:,1,None],(0,1))
    lengths,prefix=map(np.stack,zip(*(public_lengths(g,y) for g,y in zip(geometries,yaws))))
    lengths=torch.tensor(lengths).float();prefix=torch.tensor(prefix).float()
    risk=danger;selected_lengths=lengths[torch.arange(n),food]
    utility=FOOD_BONUS*(1-risk)-HAZARD_PENALTY*risk-MOVE_COST*((1-risk)*selected_lengths+risk*prefix)
    route=utility.argmax(-1)
    return dict(food=food.numpy(),route=route.numpy(),beliefs=torch.cat(beliefs,-1).numpy(),estimated_danger=danger.numpy())

@torch.no_grad()
def messages(model,facts,condition,seed=92001):
    words=model.speak(facts,True)[0]
    if condition in ('normal','oracle','silent'):return words
    if condition=='blocked':return torch.full_like(words,model.vocab)
    if condition=='constant':return torch.zeros_like(words)
    if condition=='reverse':return words.flip(-1)
    if condition=='shuffle':return words[torch.randperm(len(words),generator=torch.Generator().manual_seed(seed))]
    if condition=='invert_state':
        changed=facts.clone();changed[:,3]=1-changed[:,3]
        # Only state position changes in the fixed-slot protocol.
        donor=model.speak(changed,True)[0];words[:,3]=donor[:,3];return words
    raise ValueError(condition)
