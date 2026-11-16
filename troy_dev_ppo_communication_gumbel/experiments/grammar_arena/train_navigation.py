"""GPU PPO over real ordered MuJoCo transitions for public waypoint following."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .arena import FoodArena,source_identity

class Navigator(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.actor=nn.Sequential(nn.Linear(3,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,4))
        self.critic=nn.Sequential(nn.Linear(3,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.zeros_(self.actor[-1].bias)
    def forward(self,obs):return Categorical(logits=self.actor(obs)),self.critic(obs).squeeze(-1)

def gae(reward,terminated,done,value,next_value,gamma=.99,lam=.95):
    """Time limits bootstrap their last state but never leak across resets."""
    advantages=torch.zeros_like(reward);running=torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta=reward[t]+gamma*next_value[t]*(1-terminated[t])-value[t]
        running=delta+gamma*lam*(1-done[t])*running;advantages[t]=running
    return advantages,advantages+value

@torch.no_grad()
def evaluate_motor(policy,seed=880001,episodes=256,environments=32,device='cuda'):
    envs=[FoodArena(seed+j) for j in range(environments)];obs=np.stack([e.reset_motor() for e in envs])
    successes=[];steps=[];active=[True]*environments
    while len(successes)<episodes:
        actions=policy(torch.as_tensor(obs,device=device))[0].logits.argmax(-1).cpu().numpy()
        for j,e in enumerate(envs):
            o,r,terminal,timeout,info=e.step(actions[j]);obs[j]=o
            if terminal or timeout:
                if len(successes)<episodes:successes.append(info['reached']);steps.append(e.steps)
                obs[j]=e.reset_motor()
    for e in envs:e.close()
    return dict(success=float(np.mean(successes)),mean_steps=float(np.mean(steps)),episodes=episodes)

def run(output,seed=97,updates=200,environments=32,horizon=128,device='cuda',width=64):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(seed)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    policy=Navigator(width).to(device);opt=torch.optim.Adam(policy.parameters(),lr=.0005)
    envs=[FoodArena(8100000+seed*1000+j) for j in range(environments)]
    obs=np.stack([e.reset_motor() for e in envs]);curve=[];completed=success=0;start=time.perf_counter()
    for update in range(updates):
        states=[];next_states=[];acts=[];oldlp=[];values=[];rewards=[];terminals=[];dones=[]
        for t in range(horizon):
            state=torch.as_tensor(obs,device=device)
            with torch.no_grad():dist,value=policy(state);action=dist.sample();logp=dist.log_prob(action)
            actions=action.cpu().numpy();nobs=[];rr=[];tt=[];dd=[];reset_obs=[]
            for j,e in enumerate(envs):
                no,reward,terminated,truncated,info=e.step(actions[j]);done=terminated or truncated
                nobs.append(no);rr.append(reward);tt.append(terminated);dd.append(done)
                if done:completed+=1;success+=int(info['reached']);reset_obs.append(e.reset_motor())
                else:reset_obs.append(no)
            states.append(state);next_states.append(torch.as_tensor(np.stack(nobs),device=device))
            acts.append(action);oldlp.append(logp);values.append(value)
            rewards.append(torch.tensor(rr,device=device));terminals.append(torch.tensor(tt,device=device,dtype=torch.float32));dones.append(torch.tensor(dd,device=device,dtype=torch.float32))
            obs=np.stack(reset_obs)
        states,next_states,acts,oldlp,values,rewards,terminals,dones=map(torch.stack,(states,next_states,acts,oldlp,values,rewards,terminals,dones))
        with torch.no_grad():
            next_value=policy(next_states.flatten(0,1))[1].reshape(horizon,environments)
            advantage,returns=gae(rewards,terminals,dones,values,next_value)
            advantage=(advantage-advantage.mean())/advantage.std().clamp_min(1e-6)
        flat=states.flatten(0,1);actions=acts.flatten();old=oldlp.flatten();advantage=advantage.flatten();returns=returns.flatten()
        entropy_weight=.01*max(.1,1-update/updates)
        for epoch in range(4):
            for ids in torch.randperm(len(flat),device=device).split(1024):
                dist,pred=policy(flat[ids]);ratio=(dist.log_prob(actions[ids])-old[ids]).exp()
                loss=-torch.minimum(ratio*advantage[ids],ratio.clamp(.8,1.2)*advantage[ids]).mean()+.5*(pred-returns[ids]).square().mean()-entropy_weight*dist.entropy().mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);opt.step()
        if (update+1)%10==0 or update==0:
            row=dict(update=update+1,environment_steps=(update+1)*environments*horizon,seconds=time.perf_counter()-start,
                     sampled_success=success/max(1,completed),completed=completed,loss=float(loss.detach()),
                     steps_per_second=(update+1)*environments*horizon/(time.perf_counter()-start))
            curve.append(row);print(json.dumps(row),flush=True);completed=success=0
    for e in envs:e.close()
    config=dict(seed=seed,updates=updates,environments=environments,horizon=horizon,width=width,device=device,
                gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None,python=platform.python_version(),torch=torch.__version__,
                training_seconds=time.perf_counter()-start,environment_steps=updates*environments*horizon,
                observation='public egocentric waypoint unit direction and distance only; 3 inputs',
                reward='.8*(waypoint distance progress)-.01+2*waypoint reached; no private task reward',
                gamma=.99,gae_lambda=.95,ppo_epochs=4,learning_rate=.0005,minibatch=1024,**source_identity())
    torch.save(dict(model=policy.state_dict(),optimizer=opt.state_dict(),config=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None),output/'checkpoint.pt')
    (output/'config.json').write_text(json.dumps(config,indent=2));(output/'curve.json').write_text(json.dumps(curve,indent=2))
    source=output/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
    return policy,config

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=97)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--environments',type=int,default=32);p.add_argument('--horizon',type=int,default=128);p.add_argument('--device',default='cuda')
    a=p.parse_args();model,config=run(a.output,a.seed,a.updates,a.environments,a.horizon,a.device)
    score=evaluate_motor(model,device=a.device);(a.output/'motor-evaluation.json').write_text(json.dumps(score,indent=2));print(json.dumps(dict(config=config,evaluation=score)),flush=True)

if __name__=='__main__':main()
