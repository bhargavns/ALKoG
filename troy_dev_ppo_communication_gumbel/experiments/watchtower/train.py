"""Recurrent PPO for the scout and the forager, from the team reward alone.

    python -m experiments.watchtower.train --stage 1 --output results/watchtower/s1_0 --seed 0
    python -m experiments.watchtower.train --stage 1 --output results/watchtower/s1_silent_0 --seed 0 --silent

One update: (A) play 256 whole episodes, sampling every decision, storing what
each agent observed, did and expected; (B) per agent, generalised advantage
estimates from the team reward and its own critic; (C) per agent, clipped PPO
updates of its own network over whole replayed episodes.  The only thing one
agent receives from the other is the integer symbols, as observations.

The forager decides every tick.  The scout decides only on its turns (ticks 0,
5, 10, ...): it looks, updates its memory, and speaks; its reward for a turn is
the team reward of the five ticks that follow.  In stage 3 the forager also
speaks on ticks 2, 7, 12, ...; the scout hears that on its next turn.
"""
from pathlib import Path
import argparse, json, time
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .env import World, SIZE
from .agents import Scout, Forager, STOP, MAX_LEN

SPEAK_F, EVERY = 2, 5


def onehot(h, n=4): return F.one_hot(h + 1, n).float()


def privileged(w): return torch.cat((onehot(w.hunger_f), onehot(w.hunger_s)), -1)


def scout_vec(w):
    return torch.cat((onehot(w.hunger_s), onehot(w.carry), torch.full((w.B, 1), w.t / w.T, device=w.device),
                      w.f_done.float()[:, None], w.s_done.float()[:, None]), -1)


def forager_vec(w, full):
    return torch.cat((w.view(full).flatten(1), F.one_hot(w.pos[:, 0], SIZE).float(), F.one_hot(w.pos[:, 1], SIZE).float(),
                      onehot(w.hunger_f), onehot(w.carry), torch.full((w.B, 1), w.t / w.T, device=w.device)), -1)


def silence(B, device): return torch.full((B, MAX_LEN), STOP, dtype=torch.long, device=device)


@torch.no_grad()
def play(w, S, Fo, cfg, sample=True, controls=(), keep_maps=True):
    """Play one batch of whole episodes.  `controls`: 'block' (forager hears nothing),
    'shuffle' (forager hears another episode's scout), 'block_forager' (scout hears nothing).
    Per-tick records are [T, B, ...]; per-turn scout records are [K, B, ...]."""
    B, T, dev = w.B, w.T, w.device; stage = cfg['stage']
    hS = torch.zeros(B, S.core, device=dev); hF = torch.zeros(B, Fo.core, device=dev)
    heardS = silence(B, dev); freshS = torch.zeros(B, device=dev); embF = None
    tick = {k: [] for k in ('fvec', 'heardF', 'freshF', 'priv', 'alive', 'act', 'lp_act', 'tokF', 'lpF', 'lenF', 'vF', 'r')}
    turn = {k: [] for k in ('smap', 'svec', 'heardS', 'freshS', 'priv', 'alive', 'tokS', 'lpS', 'lenS', 'vS')}
    perm = torch.randperm(B, device=dev)
    for t in range(T):
        alive = ~w.done; full = w.render(); priv = privileged(w)
        lenS = torch.zeros(B, device=dev)
        if t % EVERY == 0:
            smap = w.render(scout=True) if stage == 3 else full; svec = scout_vec(w)
            hS = S.step(hS, smap, svec, heardS, freshS)
            if cfg['silent']: tokS, lpS = silence(B, dev), torch.zeros(B, device=dev)
            else: tokS, lpS, _, lenS = S.mouth(hS, greedy=not sample)
            vS = S.value(torch.cat((hS, priv), -1)).squeeze(1)
            for k, v in (('smap', smap.to(torch.uint8) if keep_maps else smap[:, :1].to(torch.uint8)), ('svec', svec), ('heardS', heardS),
                         ('freshS', freshS), ('priv', priv), ('alive', alive), ('tokS', tokS), ('lpS', lpS), ('lenS', lenS), ('vS', vS)):
                turn[k].append(v)
            heardF = tokS
            if 'block' in controls: heardF = silence(B, dev)
            if 'shuffle' in controls: heardF = tokS[perm]
            embF = Fo.ears(heardF); freshF = torch.ones(B, device=dev); freshS = torch.zeros(B, device=dev)
        else:
            freshF = torch.zeros(B, device=dev)
        fvec = forager_vec(w, full)
        hF = Fo.step(hF, fvec, embF, freshF)
        dist = Categorical(logits=Fo.policy(hF))
        a = dist.sample() if sample else dist.logits.argmax(-1)
        speakF = stage == 3 and t % EVERY == SPEAK_F
        if speakF and not cfg['silent']: tokF, lpF, _, lenF = Fo.mouth(hF, greedy=not sample)
        else: tokF, lpF, lenF = silence(B, dev), torch.zeros(B, device=dev), torch.zeros(B, device=dev)
        vF = Fo.value(torch.cat((hF, priv), -1)).squeeze(1)
        r = w.step(a) - cfg['cost_now'] * (lenS + lenF) * alive.float()
        for k, v in (('fvec', fvec), ('heardF', heardF), ('freshF', freshF), ('priv', priv), ('alive', alive), ('act', a),
                     ('lp_act', dist.log_prob(a)), ('tokF', tokF), ('lpF', lpF), ('lenF', lenF), ('vF', vF), ('r', r)):
            tick[k].append(v)
        if speakF:
            heardS = silence(B, dev) if 'block_forager' in controls else tokF
            freshS = torch.ones(B, device=dev)
    out = {k: torch.stack(v) for k, v in tick.items()} | {'S_' + k: torch.stack(v) for k, v in turn.items()}
    out['stats'] = {k: v.clone() for k, v in w.stats.items()}
    return out


def gae(r, v, alive, gamma=0.99, lam=0.95):
    T = r.shape[0]; adv = torch.zeros_like(r); last = torch.zeros_like(r[0])
    for t in reversed(range(T)):
        nxt_v = v[t + 1] * alive[t + 1] if t + 1 < T else torch.zeros_like(v[t])
        nxt_a = alive[t + 1] if t + 1 < T else torch.zeros_like(v[t])
        delta = r[t] + gamma * nxt_v - v[t]
        last = (delta + gamma * lam * nxt_a * last) * alive[t]
        adv[t] = last
    return adv


def replay(S, Fo, d, idx, stage, silent):
    """Recompute log-probabilities, entropies and values for episodes `idx`, with gradients."""
    T = d['r'].shape[0]; B = len(idx); dev = d['r'].device; zero = torch.zeros(B, device=dev)
    hS = torch.zeros(B, S.core, device=dev); hF = torch.zeros(B, Fo.core, device=dev)
    out = {k: [] for k in ('lp_act', 'ent_act', 'lpF', 'entF', 'vF', 'lpS', 'entS', 'vS')}
    for t in range(T):
        g = lambda k: d[k][t, idx]
        if t % EVERY == 0:
            k = t // EVERY; s = lambda key: d['S_' + key][k, idx]
            hS = S.step(hS, s('smap').float(), s('svec'), s('heardS'), s('freshS'))
            if silent: lpS, entS = zero, zero
            else: _, lpS, entS, _ = S.mouth(hS, forced=s('tokS'))
            out['lpS'].append(lpS); out['entS'].append(entS); out['vS'].append(S.value(torch.cat((hS, s('priv')), -1)).squeeze(1))
            embF = Fo.ears(g('heardF'))
        hF = Fo.step(hF, g('fvec'), embF, g('freshF'))
        dist = Categorical(logits=Fo.policy(hF))
        if stage == 3 and t % EVERY == SPEAK_F and not silent: _, lpF, entF, _ = Fo.mouth(hF, forced=g('tokF'))
        else: lpF, entF = zero, zero
        for k, v in (('lp_act', dist.log_prob(g('act'))), ('ent_act', dist.entropy()), ('lpF', lpF), ('entF', entF),
                     ('vF', Fo.value(torch.cat((hF, g('priv')), -1)).squeeze(1))):
            out[k].append(v)
    return {k: torch.stack(v) for k, v in out.items()}


def clip_loss(lp_new, lp_old, adv, mask, clip=0.2):
    ratio = (lp_new - lp_old).exp()
    l = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    return (l * mask).sum() / mask.sum().clamp_min(1)


def norm(a, mask):
    m = mask.bool()
    if m.sum() < 2: return a
    return (a - a[m].mean()) / (a[m].std() + 1e-8)


def masked_mean(x, m): return (x * m).sum() / m.sum().clamp_min(1)


def run(output, stage=1, seed=0, updates=3000, B=256, lr=3e-4, epochs=3, minibatches=2, cost=0.005, silent=False,
        ent_act=0.01, ent_talk=0.01, curriculum=0.2, cost_from=0.4, device='cpu', log=25):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); gen = torch.Generator(device=device).manual_seed(1000 + seed)
    S, Fo = Scout().to(device), Forager().to(device)
    optS, optF = torch.optim.Adam(S.parameters(), lr=lr), torch.optim.Adam(Fo.parameters(), lr=lr)
    cfg = dict(stage=stage, silent=silent, cost_now=0.)
    curve = []; start = time.perf_counter()
    for u in range(updates):
        frac = u / updates
        cfg['cost_now'] = cost if frac >= cost_from else 0.
        w = World(B, stage, device=device, generator=gen)
        w.reset(single=torch.ones(B, dtype=torch.bool, device=device) if frac < curriculum else None)
        d = play(w, S, Fo, cfg)
        alive = d['alive'].float(); T = alive.shape[0]
        # scout: one decision per turn, rewarded with the team reward of the following EVERY ticks
        aS = d['S_alive'].float(); K = aS.shape[0]
        rS = F.pad(d['r'], (0, 0, 0, K * EVERY - T)).view(K, EVERY, -1).sum(1)
        advS = gae(rS, d['S_vS'], aS); retS = advS + d['S_vS']
        advF = gae(d['r'], d['vF'], alive); retF = advF + d['vF']
        tt = torch.arange(T, device=device)[:, None]
        talkF = alive * ((tt % EVERY) == SPEAK_F).float() * (1. if stage == 3 and not silent else 0.)
        nS, nF, nFt = norm(advS, aS), norm(advF, alive), norm(advF, talkF)
        kt = ent_talk * max(0.1, 1 - frac)
        for _ in range(epochs):
            for idx in torch.randperm(B, device=device).chunk(minibatches):
                o = replay(S, Fo, d, idx, stage, silent)
                a, s_, ft = alive[:, idx], aS[:, idx], talkF[:, idx]
                lossF = clip_loss(o['lp_act'], d['lp_act'][:, idx], nF[:, idx], a) - ent_act * masked_mean(o['ent_act'], a) \
                    + 0.5 * masked_mean((o['vF'] - retF[:, idx]) ** 2, a)
                if ft.sum() > 0:
                    lossF = lossF + clip_loss(o['lpF'], d['lpF'][:, idx], nFt[:, idx], ft) - kt * masked_mean(o['entF'], ft)
                lossS = 0.5 * masked_mean((o['vS'] - retS[:, idx]) ** 2, s_)
                if not silent:
                    lossS = lossS + clip_loss(o['lpS'], d['S_lpS'][:, idx], nS[:, idx], s_) - kt * masked_mean(o['entS'], s_)
                optS.zero_grad(); optF.zero_grad(); (lossS + lossF).backward()
                nn.utils.clip_grad_norm_(S.parameters(), 0.5); nn.utils.clip_grad_norm_(Fo.parameters(), 0.5)
                optS.step(); optF.step()
        if (u + 1) % log == 0 or u == updates - 1:
            st = d['stats']
            row = dict(update=u + 1, ret=float(d['r'].sum(0).mean()), wanted=float(st['wanted'].mean()), other=float(st['other'].mean()),
                       rotten=float(st['rotten'].mean()), lion=float(st['lion'].mean()), tiger=float(st['tiger'].mean()),
                       delivered=float(st['delivered'].mean()), lenS=float(masked_mean(d['S_lenS'], aS)), lenF=float(masked_mean(d['lenF'], talkF)),
                       single=frac < curriculum, cost=cfg['cost_now'], seconds=round(time.perf_counter() - start, 1))
            curve.append(row); print(json.dumps(row), flush=True)
            if (u + 1) % (log * 20) == 0: torch.save(dict(scout=S.state_dict(), forager=Fo.state_dict()), output / 'checkpoint_partial.pt')
    conf = dict(stage=stage, seed=seed, updates=updates, B=B, lr=lr, epochs=epochs, minibatches=minibatches, cost=cost, silent=silent,
                ent_act=ent_act, ent_talk=ent_talk, curriculum=curriculum, cost_from=cost_from, auxiliary_losses='none')
    torch.save(dict(scout=S.state_dict(), forager=Fo.state_dict(), config=conf), output / 'checkpoint.pt')
    (output / 'config.json').write_text(json.dumps(conf, indent=1)); (output / 'curve.json').write_text(json.dumps(curve))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--stage', type=int, default=1); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=3000); p.add_argument('--B', type=int, default=256)
    p.add_argument('--silent', action='store_true'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--ent-talk', type=float, default=0.01)
    p.add_argument('--log', type=int, default=25); p.add_argument('--device', default='cpu')
    a = p.parse_args(); torch.set_num_threads(a.threads)
    run(a.output, a.stage, a.seed, a.updates, a.B, lr=a.lr, silent=a.silent, ent_talk=a.ent_talk, device=a.device, log=a.log)
