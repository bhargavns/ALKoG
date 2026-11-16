"""Global recurrent encoder/decoder: no semantic word slots or attribute heads."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SceneEncoder(nn.Module):
    def __init__(self,domain,width):
        super().__init__();self.domain=domain
        dim=24 if domain=='attributes' else 14
        self.elements=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.global_net=nn.Sequential(nn.Linear(width,width),nn.Tanh())
    def forward(self,scene):
        # All objects use the SAME network; swapping their array order has no
        # linguistic effect. Coordinates ground the roles, not word positions.
        mask=(scene.abs().sum(-1)>0).float()
        elements=self.elements(scene)*mask[...,None]
        return self.global_net(elements.sum(-2))

class GroundedEncoder(nn.Module):
    """Fixed isometry of public perceptual features; no token/position roles.

    This is an explicit perceptual-anchoring bias, not a language template.
    Attribute scenes retain their one-hot percepts; spatial scenes retain an
    unordered object-feature map on five public coordinate cells. The word
    encoder, recurrent listener and message meanings remain learned.
    """
    def __init__(self,domain,width):
        super().__init__();self.domain=domain;dim=24 if domain=='attributes' else 65
        if width<dim:raise ValueError('Anchored encoder needs width >= perceptual dimension')
        matrix=torch.linalg.qr(torch.randn(width,dim),mode='reduced').Q
        self.register_buffer('projection',matrix.T)
    def forward(self,scene):
        if self.domain=='attributes':
            raw=scene[...,0,:]
            mean=torch.tensor([1/8]*16+[1/4]*4+[1/2]*4,device=scene.device)
            raw=raw-mean
        else:
            xy=scene[...,-2:];mask=(scene.abs().sum(-1)>0).float()
            # Physical cells left/right/down/up/centre. Array order is irrelevant.
            cell=torch.full(mask.shape,4,device=scene.device,dtype=torch.long)
            cell=torch.where(xy[...,0]<-.5,0,cell);cell=torch.where(xy[...,0]>.5,1,cell)
            cell=torch.where(xy[...,1]<-.5,2,cell);cell=torch.where(xy[...,1]>.5,3,cell)
            per_object=torch.cat((scene[...,:12],torch.ones_like(mask[...,None])),-1)*mask[...,None]
            raw=torch.einsum('...sp,...sd->...pd',F.one_hot(cell,5).float(),per_object).flatten(-2)
        return raw@self.projection

class Speaker(nn.Module):
    def __init__(self,domain,width=96,vocab=8,max_length=8,contextual=True,positions=True,attention=False):
        super().__init__();self.vocab=vocab;self.eos=vocab;self.pad=vocab+1;self.max_length=max_length
        self.contextual=contextual;self.positions=positions;self.attention=attention;self.encoder=SceneEncoder(domain,width);self.embedding=nn.Embedding(vocab+2,32)
        self.position=nn.Embedding(max_length,16) if positions else None
        self.cell=nn.GRUCell(32+(width if contextual else 0)+(16 if positions else 0),width);self.output=nn.Linear(width,vocab+1)
        if attention:
            self.unit=nn.Sequential(nn.Linear(24 if domain=='attributes' else 14,width),nn.Tanh())
            self.query=nn.Linear(width,width);self.key=nn.Linear(width,width)
            
        with torch.no_grad():self.output.bias[self.eos]=-3.

    def forward(self,scene,mode='greedy',temperature=1.,saved=None,min_length=0):
        h=self.encoder(scene);context=h;n=len(scene);dev=scene.device
        previous=self.embedding(torch.full((n,),self.pad,device=dev,dtype=torch.long))
        if self.attention:
            if scene.shape[-1]==24:
                masks=torch.stack([torch.nn.functional.pad(torch.ones(k,device=dev),(sum((8,8,4,2,2)[:j]),24-sum((8,8,4,2,2)[:j+1]))) for j,k in enumerate((8,8,4,2,2))])
                inputs=scene[:,0,None,:]*masks[None,:,:]
            else:inputs=scene
            units=self.unit(inputs);unit_mask=inputs.abs().sum(-1)>0;keys=self.key(units)
        alive=torch.ones(n,device=dev);logp=alive*0;entropy=alive*0;length=alive*0
        tokens=[];wire=[];logits_list=[];masks=[]
        for j in range(self.max_length):
            if self.attention:
                scores=torch.einsum('nud,nd->nu',keys,self.query(h))/h.shape[-1]**.5
                weights=scores.masked_fill(~unit_mask,-1e9).softmax(-1)
                context=torch.einsum('nu,nud->nd',weights,units)
            inputs=[previous]
            if self.contextual:inputs.append(context)
            if self.positions:inputs.append(self.position.weight[j].expand(n,-1))
            h=self.cell(torch.cat(inputs,-1),h)
            logits=self.output(h)
            if j<min_length:logits=torch.cat((logits[:,:self.vocab],torch.full_like(logits[:,self.vocab:],-1e9)),-1)
            dist=Categorical(logits=logits)
            logits_list.append(logits);masks.append(alive)
            if mode=='gumbel':
                soft=F.gumbel_softmax(logits,tau=temperature,hard=False)
                hard=F.one_hot(soft.argmax(-1),self.vocab+1).float()
                # Exact integer one-hot forward values, with the usual biased
                # straight-through gradient. Subtraction is evaluated first.
                value=hard+(soft-soft.detach())
                word=value.argmax(-1)
            else:
                word=(saved[:,j].clamp_max(self.eos) if saved is not None else logits.argmax(-1) if mode=='greedy' else dist.sample())
                value=F.one_hot(word,self.vocab+1).float()
            tokens.append(torch.where(alive.detach().bool(),word,torch.full_like(word,self.pad)))
            wire.append(value)
            logp=logp+alive.detach()*dist.log_prob(word);entropy=entropy+alive.detach()*dist.entropy()
            length=length+alive*(1-value[:,self.eos]);alive=alive*(1-value[:,self.eos])
            previous=value@self.embedding.weight[:self.vocab+1]
        return dict(tokens=torch.stack(tokens,1),wire=torch.stack(wire,1),logp=logp,entropy=entropy,length=length,
                    logits=torch.stack(logits_list,1),alive=torch.stack(masks,1))

    def imitation(self,scene,targets):
        out=self.forward(scene,mode='sample',saved=targets)
        labels=targets.clamp_max(self.eos)
        loss=F.cross_entropy(out['logits'].flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
        mask=(targets!=self.pad).float()
        return (loss*mask).sum()/mask.sum().clamp_min(1)

class Listener(nn.Module):
    def __init__(self,domain,width=96,vocab=8,anchored=False):
        super().__init__();self.vocab=vocab;self.eos=vocab;self.pad=vocab+1
        self.embedding=nn.Embedding(vocab+2,32,padding_idx=self.pad);self.cell=nn.GRUCell(32,width)
        self.candidates=GroundedEncoder(domain,width) if anchored else SceneEncoder(domain,width)
        self.message_projection=nn.Linear(width,width)
        self.candidate_projection=nn.Identity() if anchored else nn.Linear(width,width)

    def message(self,tokens=None,wire=None):
        if wire is None:
            wire=F.one_hot(tokens.clamp_max(self.eos),self.vocab+1).float()
        n,L,_=wire.shape;h=torch.zeros(n,self.cell.hidden_size,device=wire.device);alive=torch.ones(n,device=wire.device)
        for j in range(L):
            embedded=wire[:,j]@self.embedding.weight[:self.vocab+1]
            proposed=self.cell(embedded,h);h=alive[:,None]*proposed+(1-alive[:,None])*h
            alive=alive*(1-wire[:,j,self.eos])
        return self.message_projection(h)

    def forward(self,candidates,tokens=None,wire=None):
        message=F.normalize(self.message(tokens,wire),dim=-1)
        encoded=F.normalize(self.candidate_projection(self.candidates(candidates)),dim=-1)
        return 20.*torch.einsum('nd,ncd->nc',message,encoded)

class ParallelSpeaker(Speaker):
    """Whole-scene -> unassigned parallel token slots, an architectural control.

    Every slot receives the entire scene embedding. There is no fact-specific
    input, target token, slot label or manually assigned lexical alignment.
    Unlike recurrent methods this control has a fixed eight-symbol packet.
    """
    def __init__(self,domain,width=96,vocab=8,max_length=8):
        super().__init__(domain,width,vocab,max_length)
        self.parallel=nn.Linear(width,max_length*vocab)
    def forward(self,scene,mode='greedy',temperature=1.,saved=None,min_length=0):
        h=self.encoder(scene);raw=self.parallel(h).reshape(-1,self.max_length,self.vocab)
        logits=torch.cat((raw,torch.full_like(raw[...,:1],-1e9)),-1)
        dist=Categorical(logits=logits)
        if mode=='gumbel':
            soft=F.gumbel_softmax(logits,tau=temperature,hard=False)
            hard=F.one_hot(soft.argmax(-1),self.vocab+1).float();wire=hard+(soft-soft.detach());words=wire.argmax(-1)
        else:
            words=saved if saved is not None else logits.argmax(-1) if mode=='greedy' else dist.sample()
            wire=F.one_hot(words,self.vocab+1).float()
        n=len(scene)
        return dict(tokens=words,wire=wire,logp=dist.log_prob(words).sum(-1),entropy=dist.entropy().sum(-1),
                    length=torch.full((n,),float(self.max_length),device=scene.device),logits=logits,alive=torch.ones_like(words).float())

class ParallelListener(Listener):
    def __init__(self,domain,width=96,vocab=8,max_length=8):
        super().__init__(domain,width,vocab,True)
        self.sentence=nn.Sequential(nn.Linear(32*max_length,width),nn.Tanh(),nn.Linear(width,width))
    def message(self,tokens=None,wire=None):
        if wire is None:wire=F.one_hot(tokens.clamp_max(self.eos),self.vocab+1).float()
        # Mask at/after the first EOS, so blocked channels contain no payload.
        alive=torch.cat((torch.ones_like(wire[:,:1,0]),(1-wire[:,:,self.eos]).cumprod(1)[:,:-1]),1)
        embedded=(wire@self.embedding.weight[:self.vocab+1])*alive[:,:,None]
        return self.sentence(embedded.flatten(1))

class Pair(nn.Module):
    def __init__(self,domain,width=96,vocab=8,max_length=8,anchored=False,contextual=True,positions=True,attention=False,parallel=False):
        super().__init__();self.domain=domain
        self.speaker=Speaker(domain,width,vocab,max_length,contextual,positions,attention);self.listener=Listener(domain,width,vocab,anchored)
        if parallel:
            self.speaker=ParallelSpeaker(domain,width,vocab,max_length);self.listener=ParallelListener(domain,width,vocab,max_length)
        self.critic_encoder=SceneEncoder(domain,width)
        self.critic=nn.Sequential(nn.Linear(width*2,width),nn.Tanh(),nn.Linear(width,1))
    def value(self,batch):
        # A centralized baseline only; no private activations enter listener.
        private=self.critic_encoder(batch['private']);public=self.critic_encoder(batch['candidates']).mean(1)
        return self.critic(torch.cat((private,public),-1)).squeeze(-1)

def payload(tokens,eos,pad):
    result=[]
    for v in tokens:
        v=int(v)
        if v==eos or v==pad:break
        result.append(v)
    return result

def intervene(tokens,condition,vocab,generator=None):
    result=tokens.clone();eos,pad=vocab,vocab+1
    if condition=='normal':return result
    if condition=='blocked':
        result.fill_(pad);result[:,0]=eos;return result
    if condition=='shuffle':return result[torch.randperm(len(result),device=result.device,generator=generator)]
    if condition=='permute_vocabulary':
        return torch.where(result<vocab,(result+1)%vocab,result)
    for row in result:
        size=int((row<vocab).sum())
        if condition=='sort':row[:size]=row[:size].sort().values
        elif condition=='reverse':row[:size]=row[:size].flip(0)
        else:raise ValueError(condition)
    return result
