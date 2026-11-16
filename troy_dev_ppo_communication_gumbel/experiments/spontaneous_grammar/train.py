"""Compare sampled PPO with role-free differentiable/cultural learning games."""
from pathlib import Path
import argparse,copy,hashlib,json,platform,time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .task import Game
from .models import Pair

METHODS=('ppo','gumbel','compression','curriculum','iterated','population','anchored','reconstruction','attention','parallel','invariance')

@torch.no_grad()
def test_groups(models,game,blocks=256):
    batch=game.sample(blocks*4,'test');success=[];length=[]
    for speaker in models:
        packet=speaker.speaker(batch['private'])
        length.append(float(packet['length'].mean()))
        for listener in models:
            logits=listener.listener(batch['candidates'],tokens=packet['tokens'])
            success.append(float((logits.argmax(-1)==batch['target']).float().mean()))
    return dict(success=float(np.mean(success)),length=float(np.mean(length)),pair_success=success,episodes=len(batch['target']))

def run(output,cache,method,seed=97,updates=2400,batch_size=256,width=96,vocab=8,max_length=8,device='cpu',generations=4):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(seed)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    game=Game(cache,7300000+seed,device);domain=game.domain
    n=3 if method=='population' else 1
    models=nn.ModuleList([Pair(domain,width,vocab,max_length,anchored=method in ('anchored','reconstruction','attention','parallel','invariance'),attention=method=='attention',parallel=method in ('parallel','invariance')).to(device) for _ in range(n)])
    opt=torch.optim.Adam(models.parameters(),lr=.001)
    cost=0. if method=='gumbel' else .02
    curve=[];role_augmentation_episodes=0;teacher=None;teacher_scene=None;teacher_tokens=None;generation=0;imitation_examples=0
    rng=np.random.default_rng(7400000+seed);task_rng=np.random.default_rng(7600000+seed);start=time.perf_counter();previous_generation=-1;single_episodes=0
    # A separate diagnostic generator never supplies training samples.
    diagnostic=Game(cache,7500000+seed,device)
    for step in range(updates):
        if method=='iterated':
            generation=min(generations-1,step//max(1,updates//generations))
            if generation!=previous_generation:
                if generation>0:
                    teacher=copy.deepcopy(models[0].speaker).eval()
                    for p in teacher.parameters():p.requires_grad_(False)
                    train_ids=torch.nonzero(~game.data['held'],as_tuple=True)[0].to(device)
                    ids=train_ids[torch.randperm(len(train_ids),device=device)[:128]]
                    teacher_scene=game.features[ids]
                    with torch.no_grad():teacher_tokens=teacher(teacher_scene)['tokens']
                    torch.save(dict(model=models.state_dict(),generation=generation-1),output/f'generation{generation-1}.pt')
                    models=nn.ModuleList([Pair(domain,width,vocab,max_length).to(device)])
                    opt=torch.optim.Adam(models.parameters(),lr=.001)
                previous_generation=generation
        curriculum=None
        if method=='curriculum':
            if domain=='binding' and step<min(600,updates//3):curriculum='single'
            elif domain=='attributes' and step<min(800,updates//3):curriculum=min(5,2+step//max(1,min(800,updates//3)//4))
        if domain=='binding' and curriculum is None and task_rng.random()<.20:curriculum='single'
        if curriculum=='single':single_episodes+=batch_size
        data=game.sample(batch_size,curriculum=curriculum)
        if method=='population':i,j=map(int,rng.integers(n,size=2))
        else:i=j=0
        sender=models[i].speaker;listener=models[j].listener
        generation_budget=updates//generations if method=='iterated' else updates
        local_step=step%generation_budget if method=='iterated' else step
        warmup=min(800,max(1,generation_budget//3));min_length=max_length if local_step<warmup else 0
        effective_cost=cost*max(0.,(local_step-warmup)/max(1,generation_budget-warmup))
        temperature=max(.5,1.-.5*local_step/max(1,generation_budget-1))
        if method=='ppo':
            with torch.no_grad():
                packet=sender(data['private'],mode='sample',min_length=min_length);dist=Categorical(logits=listener(data['candidates'],tokens=packet['tokens']))
                action=dist.sample();old=packet['logp']+dist.log_prob(action)
                reward=(action==data['target']).float()-.01-effective_cost*packet['length']
                advantage=reward-models[0].value(data);advantage=(advantage-advantage.mean())/advantage.std().clamp_min(1e-6)
            for _ in range(2):
                out=sender(data['private'],mode='sample',saved=packet['tokens'],min_length=min_length);dist=Categorical(logits=listener(data['candidates'],tokens=out['tokens']))
                ratio=(out['logp']+dist.log_prob(action)-old).exp()
                loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()+.5*(models[0].value(data)-reward).square().mean()
                loss-=.005*(out['entropy'].mean()+dist.entropy().mean())
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(models.parameters(),1.);opt.step()
            sampled_success=float((action==data['target']).float().mean());length=float(packet['length'].mean())
        else:
            packet=sender(data['private'],mode='gumbel',temperature=temperature,min_length=min_length)
            logits=listener(data['candidates'],wire=packet['wire'])
            loss=nn.functional.cross_entropy(logits,data['target'])+effective_cost*packet['length'].mean()
            if method in ('reconstruction','attention','parallel','invariance'):
                # A whole-scene perceptual objective; no token labels, word roles
                # or word-to-fact alignment are given to either agent.
                target_scene=nn.functional.normalize(listener.candidates(data['private']),dim=-1)
                message_scene=nn.functional.normalize(listener.message(wire=packet['wire']),dim=-1)
                loss+=2.*(1.-(target_scene*message_scene).sum(-1)).mean()
            if method=='iterated' and teacher is not None:
                local=step%(max(1,updates//generations))
                # Transmission bottleneck: only128 training meanings provide
                # teacher-generated strings. No English or semantic token labels.
                if local<200:
                    ids=torch.randint(len(teacher_scene),(64,),device=device)
                    loss+=sender.imitation(teacher_scene[ids],teacher_tokens[ids]);imitation_examples+=64
            if method=='invariance' and domain=='binding' and curriculum!='single':
                from .task import split_mask,features
                original=game.data['facts'][data['goal_ids'].cpu()].to(device)
                swapped=original.clone();swapped[:,:3]=original[:,3:6];swapped[:,3:6]=original[:,:3]
                allowed=~split_mask(swapped,'binding',game.shift)
                if allowed.any():
                    # Only TRAIN/TRAIN transformations. Preserve object inventory
                    # under physical role reversal, while distinguishing the roles.
                    a=data['private'][allowed];b=features(swapped[allowed].cpu(),'binding').to(device)
                    oa=sender(a,mode='gumbel',temperature=temperature);ob=sender(b,mode='gumbel',temperature=temperature)
                    ha=oa['wire'][...,:vocab].sum(1)/max_length;hb=ob['wire'][...,:vocab].sum(1)/max_length
                    loss+=10.*(ha-hb).square().mean()
                    candidates=torch.stack((a,b),1)
                    loss+=.5*(nn.functional.cross_entropy(listener(candidates,wire=oa['wire']),torch.zeros(len(a),dtype=torch.long,device=device))+nn.functional.cross_entropy(listener(candidates,wire=ob['wire']),torch.ones(len(a),dtype=torch.long,device=device)))
                    role_augmentation_episodes+=2*len(a)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(models.parameters(),1.);opt.step()
            sampled_success=float((logits.argmax(-1)==data['target']).float().mean());length=float(packet['length'].detach().mean())
        if (step+1)%100==0 or step==0 or step==updates-1:
            with torch.no_grad():
                greedy=sender(data['private'],min_length=min_length);glogits=listener(data['candidates'],tokens=greedy['tokens'])
                greedy_success=float((glogits.argmax(-1)==data['target']).float().mean())
            row=dict(update=step+1,generation=generation,seconds=time.perf_counter()-start,loss=float(loss.detach()),
                     sampled_success=sampled_success,greedy_train_success=greedy_success,greedy_length=float(greedy['length'].mean()),sampled_length=length,
                     minimum_training_payload=min_length,effective_length_cost=effective_cost)
            curve.append(row)
            if (step+1)%400==0:print(json.dumps(row),flush=True)
    config=dict(method=method,seed=seed,domain=domain,shift=game.shift,updates=updates,batch_size=batch_size,width=width,vocab=vocab,max_length=max_length,
                device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,python=platform.python_version(),torch=torch.__version__,
                game_episodes=updates*batch_size,imitation_examples=imitation_examples,population=n,generations=generations if method=='iterated' else 1,
                length_coefficient=cost,training_seconds=time.perf_counter()-start,learning='sampled joint PPO' if method=='ppo' else 'straight-through discrete Gumbel referential cross-entropy',
                semantic_word_roles='none',semantic_decoder_heads=False,teacher_word_labels='previous generation only' if method=='iterated' else 'none',
                parameters=sum(p.numel() for p in models.parameters()))
    config.update(role_augmentation_episodes=role_augmentation_episodes,single_object_episodes=single_episodes,perceptual_anchoring=method in ('anchored','reconstruction','attention','parallel','invariance'),
                  candidate_representation='fixed random isometry of public semantic/grid percepts' if method in ('anchored','reconstruction','attention','parallel','invariance') else 'learned global scene encoder',
                  contextual_sender=True,position_embeddings=True,attention_sender=method=='attention',parallel_channel=method in ('parallel','invariance'),scene_reconstruction=method in ('reconstruction','attention','parallel','invariance'),architecture_version=3,eos_warmup=min(800,max(1,(updates//generations if method=='iterated' else updates)//3)))
    torch.save(dict(models=models.state_dict(),optimizer=opt.state_dict(),config=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None,environment_rng=game.generator.get_state()),output/'checkpoint.pt')
    (output/'config.json').write_text(json.dumps(config,indent=2));(output/'curve.json').write_text(json.dumps(curve,indent=2))
    source=output/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
    diagnostic_score=test_groups(models,diagnostic)
    (output/'diagnostic.json').write_text(json.dumps(diagnostic_score,indent=2))
    print(json.dumps(dict(config=config,diagnostic=diagnostic_score)),flush=True)
    return models,config

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=97)
    p.add_argument('--updates',type=int,default=2400);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--width',type=int,default=96);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args();run(a.output,a.cache,a.method,a.seed,a.updates,a.batch_size,a.width,device=a.device)

if __name__=='__main__':main()
