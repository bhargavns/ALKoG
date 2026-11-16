"""Joint neural PPO from sampled choices and rewards; no Gumbel or enumeration."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import torch
from .task import Sampler,outcome,BIT_COST
from .models import Pair

METHODS={'flat_task':('flat',False),'sequence_task':('sequence',False),
         'flat_curriculum':('flat',True),'sequence_curriculum':('sequence',True),
         'sequence_signalling':('sequence',True),'slots_signalling':('slots',True),
         'tied_slots_curriculum':('tied_slots',True),'tied_slots_task':('tied_slots',False),
         'modular_curriculum':('modular',True),'silent_task':('silent',False)}

def run(method,seed,split,shift,steps,batch_size,width,device,output,warmup=400,epochs=2,slot_coefficient=.5):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.set_num_threads(1)
    channel,curriculum=METHODS[method]
    model=Pair(channel,width).to(device)
    env=Sampler(7100000+seed,device,split,shift)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    curve=[];start=time.perf_counter();query_episodes=0
    for t in range(steps):
        data=env.sample(batch_size)
        if curriculum:
            # Broadcast encoder never observes which fact the partner is asked.
            data['query_game']=torch.ones(batch_size,device=device,dtype=torch.bool) if t<warmup else torch.rand(batch_size,device=device,generator=env.generator)<.20
        else:data['query_game']=torch.zeros(batch_size,device=device,dtype=torch.bool)
        query_episodes+=int(data['query_game'].sum())
        with torch.no_grad():
            old=model.act(data)
            task=outcome(data,old['food'],old['route'],0 if channel=='silent' else 8)
            truth=data['facts'].gather(1,data['query'][:,None]).squeeze(-1)
            qr=(old['answer']==truth).float()-(0 if channel=='silent' else 8*BIT_COST)
            reward=torch.where(data['query_game'],qr,task['reward'])
            value=model.value(data)
            advantage=reward-value
            advantage=(advantage-advantage.mean())/advantage.std().clamp_min(1e-6)
        ent=.01*max(0.,1-t/(steps*.7))
        for _ in range(epochs):
            current=model.act(data,saved=old)
            ratio=(current['logp']-old['logp']).exp()
            policy=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
            val=model.value(data)
            loss=policy+.5*(val-reward).square().mean()-ent*current['sender_entropy'].mean()-.001*current['receiver_entropy'].mean()
            if method in ('sequence_signalling','slots_signalling','tied_slots_curriculum','tied_slots_task','modular_curriculum'):
                loss=loss-.10*current['signalling']+slot_coefficient*model.slot_penalty()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if t%50==0 or t==steps-1:
            row=dict(update=t+1,seconds=time.perf_counter()-start,loss=float(loss.detach()),
                     sampled_reward=float(reward.mean()),sampled_food_success=float(task['success'].float().mean()),
                     query_fraction=float(data['query_game'].float().mean()),
                     sender_entropy=float(current['sender_entropy'].mean().detach()))
            curve.append(row)
        if (t+1)%200==0:print(json.dumps(dict(method=method,seed=seed,split=split,**curve[-1])),flush=True)
    config=dict(method=method,channel=channel,seed=seed,split=split,shift=shift,steps=steps,batch_size=batch_size,
                width=width,device=device,warmup=warmup if curriculum else 0,epochs=epochs,
                seconds=time.perf_counter()-start,optimizer='Adam lr .001',sampled_training_episodes=steps*batch_size,
                python=platform.python_version(),torch=torch.__version__,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
                reward='sampled food/path outcome or sampled semantic-query outcome; no expected reward enumeration')
    config.update(signalling_coefficient=.10 if method in ('sequence_signalling','slots_signalling','tied_slots_curriculum','tied_slots_task','modular_curriculum') else 0.,
                  slot_permutation_coefficient=slot_coefficient if channel in ('slots','tied_slots') else 0.,
                  query_mix_after_warmup=.20 if curriculum else 0.)
    config.update(query_episodes=query_episodes,food_episodes=steps*batch_size-query_episodes,
                  stored_parameters=sum(p.numel() for p in model.parameters()),
                  last_update_parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().max()>1e-12))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),
                    environment_rng=env.generator.get_state(),config=config),output/'checkpoint.pt')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'curve.json').write_text(json.dumps(curve,indent=2))
    source=output/'source';source.mkdir()
    hashes={}
    for p in Path(__file__).parent.glob('*.py'):
        (source/p.name).write_bytes(p.read_bytes());hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    return model,config

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--method',choices=list(METHODS),required=True);p.add_argument('--seed',type=int,default=97)
    p.add_argument('--split',choices=('context','food_pair'),default='context');p.add_argument('--shift',type=int,default=2)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--width',type=int,default=64);p.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    p.add_argument('--warmup',type=int,default=400);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--slot-coefficient',type=float,default=.5)
    a=p.parse_args();m,c=run(a.method,a.seed,a.split,a.shift,a.steps,a.batch_size,a.width,a.device,a.output,a.warmup,a.epochs,a.slot_coefficient)
    print(json.dumps(c),flush=True)

if __name__=='__main__':main()
