"""Small cooperative decision processes with explicit information boundaries.

Policies are finite conditional probability tables. Reward optimization sums all
possible private states and messages, not only a sampled observed reward. No
public token has an assigned meaning in a learned method. Geometry is public,
and autonomous path execution is a separate receiver-local planner.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def probabilities(logits, hard=False):
    if hard:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return logits.softmax(-1)


def parameter(*shape):
    return nn.Parameter(torch.randn(*shape)*.25)


def channel(p, mode):
    """Payload 0 is a genuine signal category where allowed; never free padding."""
    if mode == 'constant':
        q=torch.zeros_like(p); q[..., 0]=p.sum(-1); return q
    if mode == 'permute':
        return p.roll(1,-1)
    return p


class PrivateGoal(nn.Module):
    K,V=4,8
    def __init__(self, method='learned'):
        super().__init__(); self.method=method
        self.sender=parameter(self.K,self.V+1)
        self.receiver=parameter(self.V+1,self.K)

    def forward(self, hard=False, intervention='normal'):
        if self.method=='oracle':
            z=self.sender.sum()*0
            return dict(success=z+1,packets=z,utility=z+1)
        s=probabilities(self.sender,hard); d=probabilities(self.receiver,hard)
        if self.method=='continuous':
            s=torch.cat((s[:,:1]*0,probabilities(self.sender[:,1:],hard)),1)
        elif self.method=='silent':
            s=torch.zeros_like(s); s[:,0]=1
        elif self.method in ('fixed','oracle'):
            s=F.one_hot(torch.arange(self.K,device=s.device)+1,self.V+1).float()
            d=torch.full_like(d,1/self.K)
            d[torch.arange(self.K,device=s.device)+1]=0
            d[torch.arange(self.K,device=s.device)+1,torch.arange(self.K,device=s.device)]=1
        sent=1-s[:,0].mean()
        if intervention=='blocked':
            delivered=torch.zeros_like(s); delivered[:,0]=1
        elif intervention=='constant':
            delivered=torch.zeros_like(s); delivered[:,0]=s[:,0]; delivered[:,1]=s[:,1:].sum(-1)
        elif intervention=='permute':
            delivered=torch.cat((s[:,:1],s[:,1:].roll(1,-1)),1)
        else: delivered=s
        correct=(delivered*d.T).sum()/self.K
        if self.method=='oracle': sent=sent*0
        return dict(success=correct,packets=sent,utility=correct-.05*sent)


class RouteSafety(nn.Module):
    V,H,T=8,9,9
    rho=.08
    gates=(0,4,8)
    def __init__(self,method='learned'):
        super().__init__(); self.method=method
        # Last word is the sender's own transmission history, not access to a
        # partner's private neural memory. Both sides know the deterministic cache.
        self.sender=parameter(self.T,2,self.H,self.V+1)
        self.receiver=parameter(self.H,2)

    def forward(self,hard=False,intervention='normal'):
        if self.method=='oracle':
            z=self.sender.sum()*0
            return dict(success=z+1,packets=z,utility=z+1,gate_packets=z,travel_packets=z)
        sender=probabilities(self.sender,hard); decoder=probabilities(self.receiver,hard)
        device=sender.device
        p=torch.zeros(2,self.H,device=device); p[:,0]=.5
        packets=p.sum()*0; gate_packets=packets; travel_packets=packets
        history=torch.arange(self.H,device=device)[:,None].expand(self.H,self.V+1)
        words=torch.arange(self.V+1,device=device)[None,:].expand_as(history)
        for t in range(self.T):
            s=sender[t]
            if self.method=='continuous':
                s=torch.cat((s[:,:,:1]*0,probabilities(self.sender[t,:,:,1:],hard)),-1)
            elif self.method=='silent' or (self.method=='opening' and t>0):
                s=torch.zeros_like(s); s[:,:,0]=1
            elif self.method=='opening':
                s=torch.cat((s[:,:,:1]*0,probabilities(self.sender[t,:,:,1:],hard)),-1)
            elif self.method in ('fixed_event','oracle'):
                s=torch.zeros_like(s)
                for state in range(2):
                    for h in range(self.H):
                        # Default cached route 0; token 1 means route 0, 2 route 1.
                        remembered=0 if h in (0,1) else 1
                        z=state+1 if t in self.gates and state!=remembered else 0
                        s[state,h,z]=1
                decoder=torch.zeros_like(decoder); decoder[:,0]=1
                decoder[2]=torch.tensor([0.,1.],device=device)
            mass=p[:,:,None]*s
            count=mass[:,:,1:].sum()
            if self.method=='oracle': count=count*0
            packets=packets+count
            if t in self.gates: gate_packets=gate_packets+count
            else: travel_packets=travel_packets+count
            actual=words
            if intervention=='constant': actual=torch.where(words>0,1,0)
            if intervention=='permute': actual=torch.where(words>0,words%self.V+1,0)
            if intervention=='blocked' or (intervention=='suppress_updates' and t>0):
                actual=torch.zeros_like(words)
            retained=torch.zeros_like(history) if self.method=='memoryless' or intervention=='reset_memory' else history
            dst=torch.where(actual==0,retained,actual)
            transition=F.one_hot(dst,self.H).float()
            if intervention=='drop25':
                transition=.75*transition+.25*F.one_hot(retained,self.H).float()
            p=mass.reshape(2,-1)@transition.reshape(-1,self.H)
            if t in self.gates: p=p*decoder.T
            if t<self.T-1:
                p=(1-self.rho)*p+self.rho*p.flip(0)
        success=p.sum()
        return dict(success=success,packets=packets,utility=success-.02*packets,
                    gate_packets=gate_packets,travel_packets=travel_packets)


class Scouting(nn.Module):
    K,V=4,8
    costs=(.02,.12,.40)
    def __init__(self,method='learned_cold'):
        super().__init__(); self.method=method
        # First action: commit to one of four goals, SCOUT, or ASK.
        # Observation index 4 means no private target observation.
        self.first=parameter(3,5,6)
        self.sender=parameter(self.K,self.V)
        self.answer=parameter(self.V,self.K)

    def warmup(self,steps=600):
        opt=torch.optim.Adam((self.sender,self.answer),lr=.05)
        for _ in range(steps):
            s=self.sender.softmax(-1); d=self.answer.softmax(-1)
            loss=-(s*d.T).sum()/self.K
            opt.zero_grad(); loss.backward(); opt.step()

    def forward(self,hard=False,intervention='normal'):
        s=probabilities(self.sender,hard); d=probabilities(self.answer,hard)
        if self.method=='fixed':
            s=F.one_hot(torch.arange(4,device=s.device),self.V).float()
            d=torch.full_like(d,1/self.K);d[:self.K]=torch.eye(self.K,device=s.device)
        reply=(channel(s,intervention)*d.T).sum(-1)
        if intervention=='blocked': reply=d[0].expand(self.K)
        first=probabilities(self.first,hard)
        if self.method=='silent':
            first=F.pad(probabilities(self.first[:,:,:5],hard),(0,1))
        success=first.sum()*0; asks=success; scouts=success; cost=success
        contexts=[]
        for c,distance in enumerate(self.costs):
            for visible in (False,True):
                obs=torch.arange(4,device=s.device) if visible else torch.full((4,),4,device=s.device)
                choice=first[c,obs]
                if self.method=='always_ask':
                    choice=torch.zeros_like(choice);choice[:,5]=1
                elif self.method=='scout_only':
                    choice=torch.zeros_like(choice)
                    if visible: choice[torch.arange(4),torch.arange(4)]=1
                    else: choice[:,4]=1
                elif self.method=='fixed':
                    choice=torch.zeros_like(choice)
                    if visible: choice[torch.arange(4),torch.arange(4)]=1
                    elif distance<.08: choice[:,4]=1
                    else: choice[:,5]=1
                elif self.method=='oracle':
                    choice=torch.zeros_like(choice);choice[torch.arange(4),torch.arange(4)]=1
                correct=choice[torch.arange(4),torch.arange(4)]+choice[:,4]+choice[:,5]*reply
                ask=choice[:,5].mean(); scout=choice[:,4].mean()
                val=correct.mean()-.08*ask-distance*scout
                success=success+correct.mean()/6
                asks=asks+ask/6;scouts=scouts+scout/6
                cost=cost+(.08*ask+distance*scout)/6
                contexts.append((correct.mean(),ask,scout,val))
        return dict(success=success,packets=2*asks,utility=success-cost,asks=asks,scouts=scouts,
                    **{f'context_{i}_{k}':v for i,row in enumerate(contexts)
                       for k,v in zip(('success','ask','scout','utility'),row)})


class Complementary(nn.Module):
    K=8
    bit_cost=.02
    def __init__(self,method='dialogue_cold'):
        super().__init__(); self.method=method
        self.query=parameter(8,9)
        self.lookup=parameter(8,8)
        self.bit_encoder=parameter(2,2)
        self.answer=parameter(8,3,2)
        self.broadcast=parameter(256,32)
        self.broadcast_decoder=parameter(32,8,2)
        bits=((torch.arange(256)[:,None]>>torch.arange(8))&1).long()
        self.register_buffer('maps',bits)

    def warmup(self,steps=1000):
        # Learn a referential convention between query words and entity slots.
        # Words start random; no English names or fixed word-to-entity mapping.
        opt=torch.optim.Adam((self.query,self.lookup),lr=.05)
        for t in range(steps):
            q=self.query[:,1:].softmax(-1); l=self.lookup.softmax(-1)
            reward=(q*l.T).sum()/8
            entropy=-(q*q.clamp_min(1e-9).log()).sum(-1).mean()
            loss=-reward-.02*max(0,1-t/500)*entropy
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad(): self.query[:,0]=-4

    def forward(self,hard=False,intervention='normal'):
        maps=self.maps; dev=maps.device
        if self.method=='silent':
            z=self.answer.sum()*0
            return dict(success=z+.5,packets=z,bits=z,utility=z+.5,asks=z)
        if self.method in ('oracle','fixed_broadcast'):
            z=self.answer.sum()*0; bits=0 if self.method=='oracle' else 8
            return dict(success=z+1,packets=z+(0 if bits==0 else 1),bits=z+bits,
                        utility=z+1-self.bit_cost*bits,asks=z)
        if self.method=='broadcast5':
            enc=probabilities(self.broadcast,hard)
            dec=probabilities(self.broadcast_decoder,hard)
            if intervention in ('blocked','constant'): enc=channel(enc,'constant')
            if intervention=='permute': enc=channel(enc,'permute')
            # Correct probabilities [map, word, queried entity].
            correct=torch.where(maps[:,None,:].bool(),dec[None,:,:,1],dec[None,:,:,0])
            success=(enc[:,:,None]*correct).sum()/(256*8)
            return dict(success=success,packets=success*0+1,bits=success*0+5,
                        utility=success-.10,asks=success*0)
        q=probabilities(self.query,hard)
        lookup=probabilities(self.lookup,hard)
        enc=probabilities(self.bit_encoder,hard)
        dec=probabilities(self.answer,hard)
        if self.method=='fixed_dialogue':
            q=F.pad(torch.eye(8,device=dev),(1,0))
            lookup=torch.eye(8,device=dev);enc=torch.eye(2,device=dev)
            dec=torch.zeros_like(dec); dec[:,0,0]=1;dec[:,1,1]=1;dec[:,2]=.5
        # Attention selects an entity fact, then a learned binary word is sent.
        # This is a structured lookup inductive bias, not a grammar claim.
        response=torch.einsum('zi,mib->mzb',lookup,enc[maps])
        if intervention=='remove_query': response=response[:,:1,:].expand_as(response)
        if intervention in ('blocked','constant'): response=channel(response,'constant')
        if intervention=='permute': response=response.roll(1,1)
        # Probability the receiver chooses each map's correct queried bit.
        correct=torch.where(maps[:,None,:,None].bool(),dec[None,None,:,:,1],dec[None,None,:,:,0])
        # [map, query-word, entity, response-word]; decoder slot 2 = no reply.
        answered=torch.einsum('mzb,mzqb->mzq',response,correct[:,:,:,:2])
        no_reply=correct[:,0,:,2]
        success=(q[:,0][None,:]*no_reply).sum()/(256*8)
        success=success+torch.einsum('qz,mzq->',q[:,1:],answered)/(256*8)
        asked=1-q[:,0].mean()
        bits=1+4*asked  # one request-presence flag, then three query bits + one answer bit
        return dict(success=success,packets=2*asked,bits=bits,
                    utility=success-self.bit_cost*bits,asks=asked)


TASKS={
    'private_goal':(PrivateGoal,('learned','continuous','silent','fixed','oracle')),
    'route_safety':(RouteSafety,('learned','memoryless','continuous','opening','silent','fixed_event','oracle')),
    'scouting':(Scouting,('learned_cold','learned_warm','always_ask','silent','scout_only','fixed','oracle')),
    'complementary':(Complementary,('dialogue_cold','dialogue_warm','broadcast5','silent','fixed_dialogue','fixed_broadcast','oracle')),
}


def entropy_regularizer(model):
    terms=[]
    for p in model.parameters():
        if p.ndim>=2:
            v=p.softmax(-1)
            terms.append(-(v*v.clamp_min(1e-9).log()).sum(-1).mean())
    return torch.stack(terms).mean()


def silent_route_ceiling():
    q=(1+(1-2*RouteSafety.rho)**4)/2
    return .5*q*q


def binary_entropy(x):
    return -x*math.log2(x)-(1-x)*math.log2(1-x) if 0<x<1 else 0.


def broadcast_bound(bits=5):
    low,high=0.,.5
    for _ in range(80):
        mid=(low+high)/2
        if binary_entropy(mid)<1-bits/8: low=mid
        else: high=mid
    return 1-(low+high)/2
