"""Reproducible oracle grid study: optional messages, memory, and temporal PPO.

This deliberately uses public geometry and private hazard status. It is not a
visual or private-location communication experiment. See README for supervision.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import deque
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical


ARMS = ('gru', 'graph', 'memoryless', 'continuous', 'periodic', 'random',
        'opening', 'none', 'oracle', 'free', 'expensive', 'auxiliary', 'counterfactual', 'cold')
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Grid:
    """Batched 11x7 kinematic arena. Exactly one possible private status flip.

    Sender observes status, geometry, position, time, and its own send history.
    Receiver observes goal/lion offsets and its pose, never true hazard status.
    Completed lanes remain absorbing until the fixed horizon, including rewards.
    """
    def __init__(self, count, seed, device='cpu', scenario='dynamic'):
        g = torch.Generator(device=device).manual_seed(seed)
        self.device, self.count, self.t = device, count, 0
        self.pos = torch.stack((torch.zeros(count, device=device),
                                torch.randint(1, 6, (count,), generator=g, device=device)), -1).long()
        self.goal = torch.stack((torch.full((count,), 10, device=device), self.pos[:, 1]), -1)
        self.lion = torch.stack((torch.full((count,), 5, device=device), self.pos[:, 1]), -1)
        self.relevant = torch.rand(count, generator=g, device=device) < .5
        self.lion[~self.relevant, 1] = (self.pos[~self.relevant, 1] + 3) % 7
        self.risk = (torch.rand(count, generator=g, device=device) < .5).float()
        self.flip = torch.randint(2, 8, (count,), generator=g, device=device)
        self.changes = torch.rand(count, generator=g, device=device) < .5
        if scenario == 'static':
            self.changes[:] = False
        if scenario == 'late':
            self.flip += 4
        self.rotated = scenario == 'rotated'
        self.done = torch.zeros(count, dtype=torch.bool, device=device)
        self.reached = self.done.clone()
        self.caught = self.done.clone()
        self.last = torch.full((count,), .5, device=device)
        self.age = torch.zeros(count, device=device)

    def observation(self):
        pos, goal, lion = self.pos.float(), self.goal.float(), self.lion.float()
        if self.rotated:
            pos, goal, lion = [torch.stack((6 - x[:, 1], x[:, 0]), -1) for x in (pos, goal, lion)]
        return torch.cat(((goal-pos)/10, (lion-pos)/10, pos/10), -1)

    def sender(self):
        return torch.cat((self.observation(), self.risk[:, None], self.last[:, None],
                          (self.age/24)[:, None], torch.full_like(self.last[:, None], self.t/24)), -1)

    def step(self, action, send, cost):
        active = ~self.done
        moves = torch.tensor(MOVES, device=self.device)[action]
        if self.rotated:
            moves = torch.stack((moves[:, 1], -moves[:, 0]), -1)
        old_dist = (self.goal-self.pos).abs().sum(-1)
        self.pos = torch.where(active[:, None], self.pos+moves, self.pos)
        self.pos[:, 0].clamp_(0, 10)
        self.pos[:, 1].clamp_(0, 6)
        hit = active & (self.pos == self.lion).all(-1) & self.risk.bool()
        reached = active & (self.pos == self.goal).all(-1) & ~hit
        reward = active.float() * (-.1 + .1*(old_dist-(self.goal-self.pos).abs().sum(-1)))
        reward += 5*reached.float()-8*hit.float()-cost*send.float()*active
        self.done |= hit | reached
        self.reached |= reached
        self.caught |= hit
        self.last = torch.where(send.bool() & active, self.risk, self.last)
        self.age = torch.where(send.bool(), 0., self.age+1)
        self.t += 1
        event = self.changes & (self.flip == self.t) & ~self.done
        self.risk = torch.where(event, 1-self.risk, self.risk)
        return reward, active, event


def mlp(i, o, width=64):
    return nn.Sequential(nn.Linear(i, width), nn.Tanh(), nn.Linear(width, o))


class Pair(nn.Module):
    def __init__(self, memory='gru'):
        super().__init__()
        self.memory = memory
        self.encoder = mlp(1, 4, 16)
        self.embedding = nn.Embedding(4, 8)
        self.cell = nn.GRUCell(9, 16)
        self.readout = nn.Linear(16, 1)
        self.decoder = mlp(8, 1, 16)
        self.gate = mlp(10, 1)
        self.actor = nn.Sequential(nn.Linear(7, 128), nn.Tanh(), nn.Linear(128, 128),
                                   nn.Tanh(), nn.Linear(128, 4))
        self.critic = mlp(8, 1)
        self.communication_q = mlp(8, 2)

    def receive(self, hidden, token, send):
        embed = self.embedding(token)
        if self.memory == 'gru':
            h = self.cell(torch.cat((embed*send[:, None], send[:, None]), -1), hidden)
            belief = self.readout(h).sigmoid().squeeze(-1)
        else:
            decoded = self.decoder(embed).sigmoid().squeeze(-1)
            prior = hidden[:, 0] if self.memory == 'graph' else torch.full_like(decoded, .5)
            belief = torch.where(send.bool(), decoded, prior)
            h = hidden.clone()
            h[:, 0] = belief
        return h, belief

    def initial(self, n, device):
        h = torch.zeros(n, 16, device=device)
        if self.memory == 'graph':
            h[:, 0] = .5
        return h


def teacher_data(seed, count=12000):
    """BFS action supervision for spatial warm-up; no communication labels."""
    rng = np.random.default_rng(seed)
    obs, labels = [], []
    for _ in range(count//60):
        goal = (10, int(rng.integers(1, 6)))
        lion = (5, int(rng.integers(7)))
        risk = int(rng.integers(2))
        distance = {goal: 0}
        queue = deque([goal])
        while queue:
            x, y = queue.popleft()
            for dx, dy in MOVES:
                p = (x+dx, y+dy)
                if 0 <= p[0] <= 10 and 0 <= p[1] <= 6 and p not in distance and not (risk and p == lion):
                    distance[p] = distance[(x, y)]+1
                    queue.append(p)
        for p in list(distance):
            if p == goal:
                continue
            ds = [distance.get((p[0]+dx, p[1]+dy), 999) for dx, dy in MOVES]
            obs.append([*(np.subtract(goal, p)/10), *(np.subtract(lion, p)/10), *(np.array(p)/10), risk])
            labels.append(int(np.argmin(ds)))
    return torch.tensor(np.array(obs), dtype=torch.float32), torch.tensor(labels)


def warmup(seed, device, steps=400):
    """Matched functional warm-up: BFS navigation; exact safety-response reward.

    Four utterances are enumerated, with no assigned token semantics. Receiver
    correctness is averaged over send-at-reset followed by eight silent steps.
    This is privileged modular pretraining, NOT communication emerging from
    navigation reward alone. Main comparisons share this warm-up budget.
    """
    torch.manual_seed(seed)
    base = Pair('gru').to(device)
    x, y = teacher_data(seed)
    x, y = x.to(device), y.to(device)
    # Uniform cells heavily overrepresent forward travel. Balance movement
    # classes and emphasize the few cells where private danger changes a route.
    weights = 1/torch.bincount(y, minlength=4).float()[y]
    weights *= torch.where((x[:, 2:4].abs().sum(-1) <= .21) & x[:, 6].bool(), 5., 1.)
    opt = torch.optim.Adam(base.actor.parameters(), lr=.003)
    for _ in range(steps):
        idx = torch.multinomial(weights, 512, replacement=True)
        loss = nn.functional.cross_entropy(base.actor(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    states = {}
    for memory in ('gru', 'graph', 'memoryless'):
        model = copy.deepcopy(base)
        model.memory = memory
        opt = torch.optim.Adam(list(model.encoder.parameters())+list(model.embedding.parameters())+
                               list(model.cell.parameters())+list(model.readout.parameters())+
                               list(model.decoder.parameters()), lr=.003)
        risk = torch.arange(2, device=device).repeat_interleave(4).float()
        tokens = torch.arange(4, device=device).repeat(2)
        for _ in range(steps):
            weights = model.encoder(torch.arange(2, device=device).float()[:, None]).softmax(-1).flatten()
            h = model.initial(8, device)
            reward = 0.
            for t in range(8 if memory != 'memoryless' else 1):
                h, belief = model.receive(h, tokens, torch.full_like(risk, float(t == 0)))
                correct = risk*belief+(1-risk)*(1-belief)
                reward = reward + (weights*correct).sum()/2
            loss = -reward/(8 if memory != 'memoryless' else 1)
            opt.zero_grad(); loss.backward(); opt.step()
        states[memory] = copy.deepcopy(model.state_dict())
    return states


def schedule(arm, env, model, sample=True):
    gate = Bernoulli(logits=model.gate(env.sender()).squeeze(-1))
    if hasattr(env, 'schedule_override'):
        rate, mode = env.schedule_override
        if mode == 'random':
            send = (torch.rand_like(env.risk) < rate).float()
        else:
            phase = env.schedule_phase
            send = (torch.floor((env.t+1)*rate+phase) > torch.floor(env.t*rate+phase)).float()
    elif arm == 'continuous':
        send = torch.ones_like(env.risk)
    elif arm in ('none', 'oracle'):
        send = torch.zeros_like(env.risk)
    elif arm == 'opening':
        send = torch.full_like(env.risk, float(env.t == 0))
    elif arm == 'periodic':
        send = torch.full_like(env.risk, float(env.t % 4 == 0))
    elif arm == 'random':
        send = (torch.rand_like(env.risk) < .25).float()
    else:
        send = gate.sample() if sample else (gate.probs >= .5).float()
    return send, gate


def rollout(model, arm, env, cost, sample=True, intervention='normal', horizon=24, save=False):
    h = model.initial(env.count, env.device)
    logps, values, qs, gates, rewards, masks, beliefs, risks, events = [], [], [], [], [], [], [], [], []
    sends, traces = [], []
    replay_inputs = {k: [] for k in ('obs', 'sender', 'risk', 'last', 'send', 'token', 'action')}
    previous_tokens = torch.zeros(env.count, dtype=torch.long, device=env.device)
    previous_send = torch.zeros_like(env.risk)
    for _ in range(horizon):
        obs, risk = env.observation(), env.risk.clone()
        sender_input, last = env.sender(), env.last.clone()
        send, gate = schedule(arm, env, model, sample)
        msg = Categorical(logits=model.encoder(risk[:, None]))
        token = msg.sample() if sample else msg.logits.argmax(-1)
        delivered, delivered_token = send, token
        if intervention == 'suppress_updates' and env.t > 0:
            delivered = torch.zeros_like(send)
        elif intervention == 'constant':
            delivered_token = torch.zeros_like(token)
        elif intervention == 'shuffle':
            delivered_token = token.roll(1)
        elif intervention == 'delay':
            delivered, delivered_token = previous_send, previous_tokens
        elif intervention == 'drop':
            delivered = send*(torch.rand_like(send) > .25)
        elif intervention == 'suppress_redundant':
            delivered = send * ((risk != env.last) | (env.t == 0)).float()
        h, belief = model.receive(h, delivered_token, delivered)
        if arm == 'oracle':
            belief = risk
        if arm == 'none':
            belief = torch.full_like(belief, .5)
        action_dist = Categorical(logits=model.actor(torch.cat((obs, belief[:, None]), -1)))
        action = action_dist.sample() if sample else action_dist.logits.argmax(-1)
        central = torch.cat((obs, risk[:, None], last[:, None]), -1)
        value = model.critic(central).squeeze(-1)
        q = model.communication_q(central)
        learned_gate = arm not in ('continuous', 'oracle', 'none', 'opening', 'periodic', 'random')
        lp = action_dist.log_prob(action)
        if arm not in ('oracle', 'none'):
            lp = lp + send*msg.log_prob(token)
        if learned_gate:
            lp = lp + gate.log_prob(send)
        for key, item in zip(replay_inputs, (obs, sender_input, risk, last, send, token, action)):
            replay_inputs[key].append(item)
        if save:
            traces.append(dict(t=env.t, position=env.pos.tolist(), risk=risk.tolist(),
                               send=send.tolist(), token=token.tolist(), belief=belief.tolist(),
                               action=action.tolist(), active=(~env.done).tolist()))
        reward, active, event = env.step(action, send, cost)
        logps.append(lp); values.append(value); qs.append(q); gates.append(send)
        rewards.append(reward); masks.append(active); beliefs.append(belief); risks.append(risk)
        events.append(event); sends.append(send*active)
        previous_send, previous_tokens = send, token
    return dict(logp=torch.stack(logps), value=torch.stack(values), q=torch.stack(qs),
                gate=torch.stack(gates), reward=torch.stack(rewards), mask=torch.stack(masks),
                belief=torch.stack(beliefs), risk=torch.stack(risks), event=torch.stack(events),
                send=torch.stack(sends), traces=traces, env=env,
                replay={k: torch.stack(v) for k, v in replay_inputs.items()})


def targets(batch, gamma=.99, lam=.95):
    """Finite-horizon task: final boundary is terminal, not a rollout truncation."""
    r, v, mask = batch['reward'], batch['value'], batch['mask']
    adv = torch.zeros_like(r)
    running = torch.zeros_like(r[0])
    for t in reversed(range(len(r))):
        continuation = mask[t+1].float() if t+1 < len(r) else 0.
        nv = v[t+1] if t+1 < len(r) else 0.
        delta = r[t]+gamma*nv*continuation-v[t]
        running = (delta+gamma*lam*continuation*running)*mask[t]
        adv[t] = running
    return adv, adv+v


def metrics(b):
    active = b['mask']
    env = b['env']
    event_next = torch.zeros_like(active)
    event_next[1:] = b['event'][:-1]
    def rate(x, m):
        return float(x[m].float().mean()) if m.any() else None
    return dict(success=float(env.reached.float().mean()), hazard=float(env.caught.float().mean()),
                timeout=float((~env.done).float().mean()),
                messages=float(b['send'].sum(0).mean()),
                send_rate=rate(b['send'], active), return_mean=float(b['reward'].sum(0).mean()),
                belief_accuracy=rate((b['belief'] >= .5) == b['risk'].bool(), active),
                event_send_rate=rate(b['send'], event_next & active),
                no_event_send_rate=rate(b['send'], ~event_next & active),
                success_relevant=rate(env.reached, env.relevant),
                success_irrelevant=rate(env.reached, ~env.relevant))


def train(model, arm, seed, device, iterations, batch_size, path):
    cost = 0. if arm in ('free', 'oracle') else .2 if arm == 'expensive' else .05
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    curve = []
    model.train()
    for iteration in range(iterations):
        env_seed = seed*1000000+iteration  # same scene stream across arms
        with torch.no_grad():
            old = rollout(model, arm, Grid(batch_size, env_seed, device), cost*min(1., (iteration+1)/25))
            advantage, returns = targets(old)
            valid = old['mask']
            advantage = (advantage-advantage[valid].mean())/advantage[valid].std().clamp_min(1e-5)
            old_logp = old['logp'].clone()
        # Replay full ordered trajectories, including the sampled discrete actions,
        # through a separate deterministic replay function (no resampling).
        data = old['replay']
        for _ in range(3):
            lp, value, q, belief = replay(model, arm, data)
            ratio = (lp-old_logp).exp()
            policy = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)[valid].mean()
            loss = policy+.5*(value[valid]-returns[valid]).square().mean()
            if arm not in ('continuous', 'oracle', 'none', 'opening', 'periodic', 'random'):
                loss -= .01*Bernoulli(logits=model.gate(data['sender']).squeeze(-1)).entropy()[valid].mean()
            if arm == 'auxiliary':
                loss += .1*nn.functional.binary_cross_entropy(belief[valid], old['risk'][valid])
            if arm == 'counterfactual':
                selected = q.gather(-1, old['gate'].long()[..., None]).squeeze(-1)
                loss += .5*(selected[valid]-returns[valid]).square().mean()
                # Q baseline is action-independent and detached; no actor access
                # to private critic inputs at deployment.
                gate_logits = model.gate(data['sender']).squeeze(-1)
                probs = gate_logits.sigmoid()
                cf = (q[..., 1]-q[..., 0]).detach()
                loss -= .05*(probs*cf)[valid].mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            opt.step()
        row = dict(iteration=iteration+1, steps=(iteration+1)*24*batch_size,
                   loss=float(loss.detach()), **metrics(old))
        curve.append(row)
        if (iteration+1) % 25 == 0:
            print(json.dumps(dict(arm=arm, seed=seed, **row)), flush=True)
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), arm=arm, seed=seed,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                    if device == 'cuda' else None), path/'checkpoint.pt')
    (path/'curve.json').write_text(json.dumps(curve, indent=2))
    return cost


def replay(model, arm, data):
    h = model.initial(data['obs'].shape[1], data['obs'].device)
    lps, vs, qs, bs = [], [], [], []
    for t in range(len(data['obs'])):
        obs, risk, send, token, action = [data[k][t] for k in ('obs', 'risk', 'send', 'token', 'action')]
        h, belief = model.receive(h, token, send)
        if arm == 'oracle': belief = risk
        if arm == 'none': belief = torch.full_like(belief, .5)
        dist = Categorical(logits=model.actor(torch.cat((obs, belief[:, None]), -1)))
        lp = dist.log_prob(action)
        if arm not in ('oracle', 'none'):
            lp += send*Categorical(logits=model.encoder(risk[:, None])).log_prob(token)
        if arm not in ('continuous', 'oracle', 'none', 'opening', 'periodic', 'random'):
            lp += Bernoulli(logits=model.gate(data['sender'][t]).squeeze(-1)).log_prob(send)
        central = torch.cat((obs, risk[:, None], data['last'][t, :, None]), -1)
        lps.append(lp); vs.append(model.critic(central).squeeze(-1))
        qs.append(model.communication_q(central)); bs.append(belief)
    return tuple(torch.stack(x) for x in (lps, vs, qs, bs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--iterations', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--evaluation-episodes', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=1500)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = dict(vars(args)); manifest['output'] = str(args.output)
    manifest.update(torch=torch.__version__, python=platform.python_version(),
                    gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output/'source.py').write_text(Path(__file__).read_text())
    results = []
    for seed in args.seeds:
        states = warmup(seed, args.device, args.warmup)
        torch.save(states, args.output/f'warmup_{seed}.pt')
        for arm in args.arms:
            torch.manual_seed(100+seed); random.seed(seed); np.random.seed(seed)
            memory = 'graph' if arm == 'graph' else 'memoryless' if arm == 'memoryless' else 'gru'
            model = Pair(memory).to(args.device); model.load_state_dict(states[memory])
            if arm == 'cold':
                fresh = Pair(memory).to(args.device)
                fresh.actor.load_state_dict(model.actor.state_dict())
                model = fresh
            path = args.output/f'{arm}_{seed}'; path.mkdir()
            start = time.perf_counter()
            with torch.no_grad():
                initial = rollout(model, arm, Grid(256, 810001, args.device), .05, False)
                (path/'initial_evaluation.json').write_text(json.dumps(metrics(initial), indent=2))
            cost = train(model, arm, seed, args.device, args.iterations, args.batch_size, path)
            model.eval()
            evaluation = []
            with torch.no_grad():
                for scenario in ('dynamic', 'static', 'rotated', 'late'):
                    for intervention in ('normal', 'suppress_updates', 'constant', 'shuffle',
                                         'delay', 'drop', 'suppress_redundant'):
                        torch.manual_seed(987654)
                        batch = rollout(model, arm, Grid(args.evaluation_episodes, 900001, args.device,
                                                        scenario), cost, False, intervention,
                                        save=scenario == 'dynamic' and intervention in ('normal', 'suppress_updates'))
                        evaluation.append(dict(scenario=scenario, intervention=intervention, **metrics(batch)))
                        if batch['traces']:
                            # Complete trajectories for first 32 held-out episodes.
                            traces = [{k: v[:32] if isinstance(v, list) else v for k,v in row.items()}
                                      for row in batch['traces']]
                            (path/f'trajectories_{intervention}.json').write_text(json.dumps(traces))
                normal = next(x for x in evaluation if x['scenario'] == 'dynamic' and x['intervention'] == 'normal')
                for mode in ('periodic', 'random'):
                    torch.manual_seed(987654)
                    env = Grid(args.evaluation_episodes, 900001, args.device)
                    env.schedule_override = (normal['send_rate'], mode)
                    env.schedule_phase = torch.rand(env.count, device=args.device)
                    b = rollout(model, arm, env, cost, False)
                    evaluation.append(dict(scenario='dynamic', intervention='matched_'+mode, **metrics(b)))
            row = dict(arm=arm, seed=seed, seconds=time.perf_counter()-start, evaluation=evaluation)
            results.append(row)
            (path/'evaluation.json').write_text(json.dumps(row, indent=2))
            (args.output/'results.json').write_text(json.dumps(results, indent=2))
            print('COMPLETED', arm, seed, round(row['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
