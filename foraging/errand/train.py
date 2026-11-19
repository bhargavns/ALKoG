"""Recurrent PPO for the errand world, from the team reward alone (see README.md).

    python -m foraging.errand.train --rules private --output results/errand/A_private_ripe_0 --seed 0 --device cuda

One update: play B whole episodes on the device, then per agent clipped PPO over
whole replayed episodes.  The scout updates its memory every tick but speaks
only at briefings (tick 0 and whenever the forager steps into home); in the
two-way condition the forager speaks first at each briefing.  Only integer
symbols pass between the agents.
"""
from pathlib import Path
import argparse, json, time
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .env import World, SIZE, STAT_KEYS
from .agents import Scout, Forager, STOP, MAX_LEN


def onehot(x, n): return F.one_hot(x + 1, n).float()


def privileged(w):
    return torch.cat((w.fed.float(), w.need.float() / 3), -1)


def scout_vec(w):
    fed = w.fed.float() if w.rules == 'shared' else torch.zeros(w.B, 2, device=w.device)
    return torch.cat((onehot(w.carry, 4), w.carry_ripe.float()[:, None] if w.scout_ripe else torch.zeros(w.B, 1, device=w.device),
                      fed, torch.full((w.B, 1), w.t / w.T, device=w.device), w.stats['items'][:, None] / 3), -1)


def forager_vec(w, full):
    return torch.cat((w.view(full).flatten(1), F.one_hot(w.pos[:, 0], SIZE).float(), F.one_hot(w.pos[:, 1], SIZE).float(),
                      onehot(w.carry, 4), w.carry_ripe.float()[:, None], w.need.float() / 3, w.fed.float(),
                      torch.full((w.B, 1), w.t / w.T, device=w.device)), -1)


def silence(n, device): return torch.full((n, MAX_LEN), STOP, dtype=torch.long, device=device)


def put(base, idx, val):
    """Out-of-place base[idx] = val (keeps gradients)."""
    return base.index_put((idx,), val)


@torch.no_grad()
def play(w, S, Fo, cfg, sample=True, controls=(), keep_maps=True):
    """Whole episodes.  controls: 'block' (forager hears silence), 'shuffle' (forager hears another
    episode's latest briefing), 'block_forager' (scout hears silence).  Records are [T, B, ...]."""
    B, T, dev = w.B, w.T, w.device
    talkF = cfg['forager_talks'] and not cfg['silent']
    hS = torch.zeros(B, S.core, device=dev); hF = torch.zeros(B, Fo.core, device=dev)
    embF = torch.zeros(B, 64, device=dev); lastS = silence(B, dev)
    perm = torch.randperm(B, device=dev)
    keys = ('smap', 'svec', 'fvec', 'priv', 'alive', 'brief', 'heardS', 'tokS', 'lpS', 'lenS', 'vS', 'heardF',
            'tokF', 'lpF', 'lenF', 'act', 'lp_act', 'vF', 'r', 'pos')
    rec = {k: [] for k in keys}
    for t in range(T):
        alive = ~w.done; brief = w.briefing(); bi = brief.nonzero().squeeze(1)
        full = w.render(); smap = w.render(scout=True) if not w.scout_ripe else full
        svec, fvec, priv = scout_vec(w), forager_vec(w, full), privileged(w)
        zeroB = torch.zeros(B, device=dev)
        # 1. the forager speaks first (two-way condition)
        tokF, lpF, lenF = silence(B, dev), zeroB.clone(), zeroB.clone()
        if talkF and len(bi):
            tk, lp, _, ln = Fo.mouth(Fo.speaker_state(hF[bi], fvec[bi]), greedy=not sample)
            tokF[bi] = tk; lpF[bi] = lp; lenF[bi] = ln
        heardS = silence(B, dev) if 'block_forager' in controls else tokF
        embS = torch.zeros(B, 64, device=dev)
        if talkF and len(bi): embS[bi] = S.ears(heardS[bi])
        # 2. the scout looks (every tick) and speaks at briefings
        hS = S.step(hS, smap, svec, embS, brief.float() if talkF else zeroB)
        tokS, lpS, lenS = silence(B, dev), zeroB.clone(), zeroB.clone()
        if not cfg['silent'] and len(bi):
            tk, lp, _, ln = S.mouth(hS[bi], greedy=not sample)
            tokS[bi] = tk; lpS[bi] = lp; lenS[bi] = ln
        lastS[bi] = tokS[bi]
        vS = S.value(torch.cat((hS, priv), -1)).squeeze(1)
        # 3. the forager hears the briefing, looks, remembers and moves
        heardF = tokS
        if 'block' in controls: heardF = silence(B, dev)
        if 'shuffle' in controls: heardF = lastS[perm]
        if len(bi): embF[bi] = Fo.ears(heardF[bi])
        hF = Fo.step(hF, fvec, embF, brief.float())
        dist = Categorical(logits=Fo.policy(hF))
        a = dist.sample() if sample else dist.logits.argmax(-1)
        vF = Fo.value(torch.cat((hF, priv), -1)).squeeze(1)
        pos = w.pos.clone()
        r = w.step(a) - cfg['cost_now'] * (lenS + lenF) * alive.float()
        w.stats['briefings'] += brief.float()
        for k, v in (('smap', smap.to(torch.uint8) if keep_maps else smap[:, :1].to(torch.uint8)), ('svec', svec), ('fvec', fvec),
                     ('priv', priv), ('alive', alive), ('brief', brief), ('heardS', heardS), ('tokS', tokS), ('lpS', lpS),
                     ('lenS', lenS), ('vS', vS), ('heardF', heardF), ('tokF', tokF), ('lpF', lpF), ('lenF', lenF), ('act', a),
                     ('lp_act', dist.log_prob(a)), ('vF', vF), ('r', r), ('pos', pos)):
            rec[k].append(v)
    out = {k: torch.stack(v) for k, v in rec.items()}
    out['final_pos'] = w.pos.clone()
    out['stats'] = {k: v.clone() for k, v in w.stats.items()}
    return out


def gae(r, v, alive, gamma=0.99, lam=0.95):
    T = r.shape[0]; adv = torch.zeros_like(r); last = torch.zeros_like(r[0])
    for t in reversed(range(T)):
        nxt_v = v[t + 1] * alive[t + 1] if t + 1 < T else torch.zeros_like(v[t])
        nxt_a = alive[t + 1] if t + 1 < T else torch.zeros_like(v[t])
        last = (r[t] + gamma * nxt_v - v[t] + gamma * lam * nxt_a * last) * alive[t]
        adv[t] = last
    return adv


def replay(S, Fo, d, idx, cfg):
    """Recompute log-probabilities, entropies and values for episodes `idx`, with gradients.

    Same computation as `replay_stepwise`, arranged for speed: everything that does not depend on the
    agents' memories (map and view encoders, message encoders, word decoders, policy and value heads)
    runs once over all ticks; only the two memory updates run tick by tick."""
    T = d['r'].shape[0]; B = len(idx); dev = d['r'].device
    talkF = cfg['forager_talks'] and not cfg['silent']
    g = lambda k: d[k][:, idx]
    brief = g('brief'); bt, bb = brief.nonzero(as_tuple=True)
    flat = lambda x: x.reshape(T * B, *x.shape[2:])
    seeS = S.see(flat(g('smap')).float()).view(T, B, -1)
    fvec = g('fvec'); seeF = Fo.see(flat(fvec)).view(T, B, -1)
    svec = g('svec'); fresh = brief.float()
    embS = torch.zeros(T, B, 64, device=dev)
    if talkF and len(bt): embS = embS.index_put((bt, bb), S.ears(g('heardS')[bt, bb]))
    embF_new = torch.zeros(T, B, 64, device=dev)
    if len(bt): embF_new = embF_new.index_put((bt, bb), Fo.ears(g('heardF')[bt, bb]))
    hS = torch.zeros(B, S.core, device=dev); hF = torch.zeros(B, Fo.core, device=dev); embF = torch.zeros(B, 64, device=dev)
    zero = torch.zeros(B, device=dev); hSs, hFs, hFprev = [], [], []
    for t in range(T):
        hFprev.append(hF)
        hS = S.memory(torch.cat((seeS[t], svec[t], embS[t], fresh[t, :, None] if talkF else zero[:, None]), -1), hS)
        embF = torch.where(brief[t, :, None], embF_new[t], embF)
        hF = Fo.memory(torch.cat((seeF[t], embF, fresh[t, :, None]), -1), hF)
        hSs.append(hS); hFs.append(hF)
    hS, hF, hFprev = torch.stack(hSs), torch.stack(hFs), torch.stack(hFprev)
    priv = g('priv')
    out = dict(vS=S.value(torch.cat((hS, priv), -1)).squeeze(-1), vF=Fo.value(torch.cat((hF, priv), -1)).squeeze(-1))
    dist = Categorical(logits=Fo.policy(hF))
    out['lp_act'] = dist.log_prob(g('act')); out['ent_act'] = dist.entropy()
    zTB = torch.zeros(T, B, device=dev)
    out['lpS'] = out['entS'] = out['lpF'] = out['entF'] = zTB
    if not cfg['silent'] and len(bt):
        _, lp, en, _ = S.mouth(hS[bt, bb], forced=g('tokS')[bt, bb])
        out['lpS'], out['entS'] = zTB.index_put((bt, bb), lp), zTB.index_put((bt, bb), en)
    if talkF and len(bt):
        _, lp, en, _ = Fo.mouth(Fo.talk(torch.cat((hFprev[bt, bb], seeF[bt, bb]), -1)), forced=g('tokF')[bt, bb])
        out['lpF'], out['entF'] = zTB.index_put((bt, bb), lp), zTB.index_put((bt, bb), en)
    return out


def replay_stepwise(S, Fo, d, idx, cfg):
    """Reference implementation of `replay`, one tick at a time (used by the tests)."""
    T = d['r'].shape[0]; B = len(idx); dev = d['r'].device
    talkF = cfg['forager_talks'] and not cfg['silent']
    hS = torch.zeros(B, S.core, device=dev); hF = torch.zeros(B, Fo.core, device=dev); embF = torch.zeros(B, 64, device=dev)
    out = {k: [] for k in ('lp_act', 'ent_act', 'lpF', 'entF', 'vF', 'lpS', 'entS', 'vS')}
    zero = torch.zeros(B, device=dev)
    for t in range(T):
        g = lambda k: d[k][t, idx]
        brief = g('brief'); bi = brief.nonzero().squeeze(1)
        fvec = g('fvec')
        lpF, entF = zero, zero
        embS = torch.zeros(B, 64, device=dev)
        if talkF and len(bi):
            _, lp, en, _ = Fo.mouth(Fo.speaker_state(hF[bi], fvec[bi]), forced=g('tokF')[bi])
            lpF, entF = put(zero, bi, lp), put(zero, bi, en)
            embS = put(embS, bi, S.ears(g('heardS')[bi]))
        hS = S.step(hS, g('smap').float(), g('svec'), embS, brief.float() if talkF else zero)
        lpS, entS = zero, zero
        if not cfg['silent'] and len(bi):
            _, lp, en, _ = S.mouth(hS[bi], forced=g('tokS')[bi])
            lpS, entS = put(zero, bi, lp), put(zero, bi, en)
        if len(bi): embF = put(embF, bi, Fo.ears(g('heardF')[bi]))
        hF = Fo.step(hF, fvec, embF, brief.float())
        dist = Categorical(logits=Fo.policy(hF)); priv = g('priv')
        for k, v in (('lp_act', dist.log_prob(g('act'))), ('ent_act', dist.entropy()), ('lpF', lpF), ('entF', entF),
                     ('vF', Fo.value(torch.cat((hF, priv), -1)).squeeze(1)), ('lpS', lpS), ('entS', entS),
                     ('vS', S.value(torch.cat((hS, priv), -1)).squeeze(1))):
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


def make_world(cfg, B, device, gen):
    return World(B, cfg['rules'], cfg['phase'], cfg['scout_ripe'], device=device, generator=gen)


def run(output, rules='private', phase='count', scout_ripe=True, forager_talks=False, silent=False, seed=0, updates=1000,
        B=2048, lr=3e-4, epochs=3, chunk=1024, cost=0.0, ent_act=0.01, ent_talk=0.01, curriculum=0.2, cost_from=0.4,
        device='cuda', log=25):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); gen = torch.Generator(device=device).manual_seed(1000 + seed)
    S, Fo = Scout().to(device), Forager().to(device)
    optS, optF = torch.optim.Adam(S.parameters(), lr=lr), torch.optim.Adam(Fo.parameters(), lr=lr)
    cfg = dict(rules=rules, phase=phase, scout_ripe=scout_ripe, forager_talks=forager_talks, silent=silent, cost_now=0.)
    conf = dict(cfg, seed=seed, updates=updates, B=B, lr=lr, epochs=epochs, chunk=chunk, cost=cost, ent_act=ent_act,
                ent_talk=ent_talk, curriculum=curriculum, cost_from=cost_from, auxiliary_losses='none')
    conf.pop('cost_now')
    (output / 'config.json').write_text(json.dumps(conf, indent=1))
    talks = forager_talks and not silent
    curve = []; start = time.perf_counter()
    for u in range(updates):
        frac = u / updates
        cfg['cost_now'] = cost if frac >= cost_from else 0.
        w = make_world(cfg, B, device, gen).reset(max_items=1 if frac < curriculum else 3, held_out='exclude')
        d = play(w, S, Fo, cfg)
        alive = d['alive'].float(); spoke = d['brief'].float()
        advS = gae(d['r'], d['vS'], alive); retS = advS + d['vS']
        advF = gae(d['r'], d['vF'], alive); retF = advF + d['vF']
        nS, nF = norm(advS, spoke), norm(advF, alive)
        kt = ent_talk * max(0.1, 1 - frac)
        for _ in range(epochs):
            for idx in torch.randperm(B, device=device).split(chunk):
                o = replay(S, Fo, d, idx, cfg)
                a, sp = alive[:, idx], spoke[:, idx]
                lossF = clip_loss(o['lp_act'], d['lp_act'][:, idx], nF[:, idx], a) - ent_act * masked_mean(o['ent_act'], a) \
                    + 0.5 * masked_mean((o['vF'] - retF[:, idx]) ** 2, a)
                if talks:
                    lossF = lossF + clip_loss(o['lpF'], d['lpF'][:, idx], nF[:, idx], sp) - kt * masked_mean(o['entF'], sp)
                lossS = 0.5 * masked_mean((o['vS'] - retS[:, idx]) ** 2, a)
                if not silent:
                    lossS = lossS + clip_loss(o['lpS'], d['lpS'][:, idx], nS[:, idx], sp) - kt * masked_mean(o['entS'], sp)
                optS.zero_grad(); optF.zero_grad(); (lossS + lossF).backward()
                nn.utils.clip_grad_norm_(S.parameters(), 0.5); nn.utils.clip_grad_norm_(Fo.parameters(), 0.5)
                optS.step(); optF.step()
        if (u + 1) % log == 0 or u == updates - 1:
            st = d['stats']
            row = dict(update=u + 1, ret=float(d['r'].sum(0).mean()), **{k: float(v.mean()) for k, v in st.items()},
                       lenS=float(masked_mean(d['lenS'], spoke)), lenF=float(masked_mean(d['lenF'], spoke)),
                       curriculum=frac < curriculum, cost=cfg['cost_now'], seconds=round(time.perf_counter() - start, 1))
            curve.append(row); print(json.dumps(row), flush=True)
            torch.save(dict(scout=S.state_dict(), forager=Fo.state_dict(), config=conf, update=u + 1), output / 'checkpoint_partial.pt')
    torch.save(dict(scout=S.state_dict(), forager=Fo.state_dict(), config=conf), output / 'checkpoint.pt')
    (output / 'curve.json').write_text(json.dumps(curve))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--rules', default='private', choices=('fixed', 'private', 'shared'))
    p.add_argument('--phase', default='count', choices=('count', 'described'))
    p.add_argument('--scout-ripe', type=int, default=1); p.add_argument('--forager-talks', action='store_true')
    p.add_argument('--silent', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=1000); p.add_argument('--B', type=int, default=2048)
    p.add_argument('--cost', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--log', type=int, default=25)
    p.add_argument('--device', default='cuda'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--mem-fraction', type=float, default=0.5, help='cap on the share of GPU memory this process may use')
    a = p.parse_args(); torch.set_num_threads(a.threads)
    if a.device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    try:                                   # run below normal priority so the machine stays responsive
        import ctypes; k = ctypes.windll.kernel32; k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    run(a.output, a.rules, a.phase, bool(a.scout_ripe), a.forager_talks, a.silent, a.seed, a.updates, a.B, lr=a.lr, cost=a.cost,
        device=a.device, log=a.log)
