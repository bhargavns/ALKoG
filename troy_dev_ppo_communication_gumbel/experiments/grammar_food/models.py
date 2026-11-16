"""Separate neural observer and collector policies with an integer-only wire."""
import torch
from torch import nn
from torch.distributions import Categorical
from .task import ARITIES,one_hot_facts,TIME_COST,HIT_STEPS

class Pair(nn.Module):
    def __init__(self,channel='sequence',width=64):
        super().__init__();self.channel=channel;self.width=width
        self.vocab=256 if channel=='flat' else 4
        self.length=1 if channel=='flat' else 4
        self.fact_embeddings=nn.ModuleList(nn.Embedding(n,16) for n in ARITIES)
        self.sender_initial=nn.Sequential(nn.Linear(64,width),nn.Tanh())
        self.sender_cell=nn.GRUCell(16,width)
        self.sender_embedding=nn.Embedding(self.vocab+1,16)
        self.sender_output=nn.Linear(width,self.vocab)
        self.modular_heads=nn.ModuleList(nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,4)) for _ in ARITIES)
        self.sender_slots=nn.Parameter(torch.randn(4,4)*.20)
        self.receiver_slots=nn.Parameter(torch.randn(4,4)*.20)
        self.receiver_embedding=nn.Embedding(self.vocab+1,16,padding_idx=self.vocab)
        self.receiver_cell=nn.GRUCell(16,width)
        self.interpret=nn.ModuleList(nn.Linear(width,n) for n in ARITIES)
        self.local_interpret=nn.ModuleList(nn.Sequential(nn.Linear(16,32),nn.Tanh(),nn.Linear(32,n)) for n in ARITIES)
        # Critic sees private facts, public scene, query, and episode kind only.
        self.critic=nn.Sequential(nn.Linear(62,width),nn.Tanh(),nn.Linear(width,1))

    def speak(self,facts,greedy=False,tokens=None,statistics=False):
        embedded=[e(facts[:,i]) for i,e in enumerate(self.fact_embeddings)]
        h=self.sender_initial(torch.cat(embedded,-1))
        n=len(facts);dev=facts.device;previous=torch.full((n,),self.vocab,device=dev,dtype=torch.long)
        logp=torch.zeros(n,device=dev);entropy=logp.clone();words=[];distributions=[]
        if self.channel=='silent':
            result=(torch.full((n,self.length),self.vocab,device=dev,dtype=torch.long),logp,entropy)
            return (*result,torch.zeros((),device=dev)) if statistics else result
        for j in range(self.length):
            if self.channel=='modular':logits=self.modular_heads[j](embedded[j])
            elif self.channel in ('slots','tied_slots'):
                assignment=(self.slot_assignment() if self.channel=='tied_slots' else self.sender_slots.softmax(-1))[j]
                local=sum(assignment[a]*embedded[a] for a in range(4))
                logits=self.modular_heads[j](local)
            elif self.channel=='flat':logits=self.sender_output(h)
            else:
                h=self.sender_cell(self.sender_embedding(previous),h)
                logits=self.sender_output(h)
            dist=Categorical(logits=logits)
            distributions.append(dist.probs)
            word=tokens[:,j] if tokens is not None else logits.argmax(-1) if greedy else dist.sample()
            words.append(word);logp=logp+dist.log_prob(word);entropy=entropy+dist.entropy()
            previous=word
        result=(torch.stack(words,1),logp,entropy)
        if not statistics:return result
        p=torch.stack(distributions,1)
        marginal=p.mean(0)
        marginal_h=-(marginal*marginal.clamp_min(1e-9).log()).sum(-1).mean()
        conditional_h=-(p*p.clamp_min(1e-9).log()).sum(-1).mean()
        # Minibatch signalling objective. For autoregressive messages this is a
        # teacher-forced diversity/confidence proxy, not exact sequence MI.
        signalling=marginal_h-conditional_h
        return (*result,signalling)

    def listen(self,words):
        if self.channel in ('modular','slots','tied_slots'):
            embeddings=self.receiver_embedding(words)
            if self.channel=='modular':return [head(embeddings[:,j]) for j,head in enumerate(self.local_interpret)]
            assignment=self.slot_assignment().T if self.channel=='tied_slots' else self.receiver_slots.softmax(-1)
            return [head(torch.einsum('j,nje->ne',assignment[a],embeddings)) for a,head in enumerate(self.local_interpret)]
        h=torch.zeros(len(words),self.width,device=words.device)
        for j in range(words.shape[1]):h=self.receiver_cell(self.receiver_embedding(words[:,j]),h)
        return [head(h) for head in self.interpret]

    def action_distributions(self,batch,words):
        logits=self.listen(words)
        lp=[x.log_softmax(-1) for x in logits]
        cand=batch['candidates'];ids=torch.arange(len(cand),device=cand.device)[:,None]
        food_logits=lp[0][ids,cand[:,:,0]]+lp[1][ids,cand[:,:,1]]
        probabilities=[x.softmax(-1) for x in logits]
        animals=torch.stack((batch['swap'],1-batch['swap']),-1)
        species=probabilities[2].gather(1,animals)
        danger=torch.nn.functional.pad(species*probabilities[3][:,1,None],(0,1))
        # Public lengths for each food and each passage, then environmental cost
        # expectation using the collector's own learned beliefs, not true facts.
        lengths=batch['lengths']
        stops=torch.tensor((*HIT_STEPS,0),device=cand.device).float()
        predicted_steps=(1-danger[:,None,:])*lengths+danger[:,None,:]*stops
        route_logits=25*((1-2*danger[:,None,:])-TIME_COST*predicted_steps)
        padded=[torch.nn.functional.pad(x,(0,4-x.shape[-1]),value=-1e9) for x in logits]
        query_logits=torch.stack(padded,1)[torch.arange(len(cand),device=cand.device),batch['query']]
        return food_logits,route_logits,query_logits,logits

    def act(self,batch,greedy=False,saved=None,words_override=None):
        words,slp,se,signalling=self.speak(batch['facts'],greedy,None if saved is None else saved['words'],True)
        if words_override is not None:words=words_override
        food_logits,routes,query_logits,interpret=self.action_distributions(batch,words)
        fd=Categorical(logits=food_logits)
        food=saved['food'] if saved is not None else food_logits.argmax(-1) if greedy else fd.sample()
        ids=torch.arange(len(food),device=food.device)
        rd=Categorical(logits=routes[ids,food])
        route=saved['route'] if saved is not None else rd.logits.argmax(-1) if greedy else rd.sample()
        qd=Categorical(logits=query_logits)
        answer=saved['answer'] if saved is not None else query_logits.argmax(-1) if greedy else qd.sample()
        query_game=batch['query_game']
        rlp=torch.where(query_game,qd.log_prob(answer),fd.log_prob(food)+rd.log_prob(route))
        re=torch.where(query_game,qd.entropy(),fd.entropy()+rd.entropy())
        return dict(words=words,food=food,route=route,answer=answer,logp=slp+rlp,
                    sender_entropy=se,receiver_entropy=re,interpret=interpret,signalling=signalling)

    def slot_penalty(self):
        if self.channel not in ('slots','tied_slots'):return self.sender_slots.sum()*0
        eye=torch.eye(4,device=self.sender_slots.device)
        if self.channel=='tied_slots':
            a=self.slot_assignment()
            return (a@a.T-eye).square().mean()
        return sum((a.softmax(-1)@a.softmax(-1).T-eye).square().mean() for a in (self.sender_slots,self.receiver_slots))

    def slot_assignment(self):
        # Static learned convention shared by the two policies, never a private
        # per-episode tensor. Alignment is supplied; field order remains learned.
        a=self.sender_slots.exp()
        for _ in range(5):
            a=a/a.sum(-1,keepdim=True)
            a=a/a.sum(0,keepdim=True)
        return a

    def value(self,batch):
        candidate_features=torch.cat([torch.nn.functional.one_hot(batch['candidates'][:,:,i],4).float() for i in (0,1)],-1)
        features=torch.cat((one_hot_facts(batch['facts']),candidate_features.flatten(1),
                            batch['swap'][:,None].float(),torch.nn.functional.one_hot(batch['query'],4).float(),
                            batch['query_game'][:,None].float(),batch['lengths'].flatten(1)/20),-1)
        return self.critic(features).squeeze(-1)
