"""Recurrent PPO for the community world: one scout, M foragers (see README.md).

    python -m foraging.community.train --output results/community/X/speak_0 --seed 0
    python -m foraging.community.train --output ... --silent
    python -m foraging.community.train --output ... --turnover 100                 # a new forager every 100 updates
    python -m foraging.community.train --output ... --turnover 100 --generation 400 --bottleneck 512

Episode b belongs to forager b * M // B.  The scout does not know which forager it is talking to.
Forager turnover: every `turnover` updates the oldest forager is replaced by a fresh one (round robin).
Scout generations: every `generation` updates a fresh scout first imitates the old scout's first
briefings, on `bottleneck` fixed maps (0 = fresh maps at every imitation step, i.e. no bottleneck),
then carries on with PPO alongside the foragers.
"""
from pathlib import Path
import argparse, json, time
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from foraging.errand.env import SIZE, STAT_KEYS
from foraging.errand.train import onehot, gae, clip_loss, norm, masked_mean, silence
from .env import World, M
from .agents import Scout, Forager, STOP, MAX_LEN


def privileged(w):
    return torch.cat((w.fed.float(), w.need.float() / 3), -1)


def scout_vec(w):
    fed = w.fed.float() if w.rules == 'shared' else torch.zeros(w.B, 2, device=w.device)
    ripe = w.carry_ripe.float()[:, None] if w.scout_ripe else torch.zeros(w.B, 1, device=w.device)
    return torch.cat((onehot(w.carry, 4), ripe, fed, torch.full((w.B, 1), w.t / w.T, device=w.device),
                      w.stats['items'][:, None] / 3), -1)


def forager_vec(w, full):
    return torch.cat((w.view(full).flatten(1), F.one_hot(w.pos[:, 0], SIZE).float(), F.one_hot(w.pos[:, 1], SIZE).float(),
                      onehot(w.carry, 4), w.carry_ripe.float()[:, None], w.need.float() / 3, w.known_fed(),
                      w.knows.float(), torch.full((w.B, 1), w.t / w.T, device=w.device)), -1)


def blocks(B, device):
    """Slices of the batch that belong to each forager."""
    owner = torch.arange(B) * M // B                      # same rule as World.forager
    edges = torch.searchsorted(owner, torch.arange(M + 1)).tolist()
    return [slice(edges[m], edges[m + 1]) for m in range(M)]


@torch.no_grad()
def first_tokens(S, w, greedy=True):
    """The scout's briefing at tick 0 (what a new scout imitates)."""
    smap = w.render(scout=True) if not w.scout_ripe else w.render()
    h = S.step(torch.zeros(w.B, S.core, device=w.device), smap, scout_vec(w), torch.zeros(w.B, 64, device=w.device),
               torch.zeros(w.B, device=w.device))
    return S.mouth(h, greedy=greedy)[0]


@torch.no_grad()
def play(w, S, Fos, cfg, sample=True, controls=()):
    """Whole episodes.  controls: 'block' (foragers hear silence), 'shuffle' (another episode's briefing)."""
    B, T, dev = w.B, w.T, w.device
    sl = blocks(B, dev)
    hS = torch.zeros(B, S.core, device=dev); hF = torch.zeros(B, Fos[0].core, device=dev)
    embF = torch.zeros(B, 64, device=dev); lastS = silence(B, dev); perm = torch.randperm(B, device=dev)
    keys = ('smap', 'svec', 'fvec', 'priv', 'alive', 'brief', 'tokS', 'lpS', 'lenS', 'vS', 'heardF', 'act', 'lp_act', 'vF', 'r', 'pos')
    rec = {k: [] for k in keys}
    zero64 = torch.zeros(B, 64, device=dev); zeroB = torch.zeros(B, device=dev)
    for t in range(T):
        alive = ~w.done; brief = w.briefing(); bi = brief.nonzero().squeeze(1)
        full = w.render(); smap = w.render(scout=True) if not w.scout_ripe else full
        svec, fvec, priv = scout_vec(w), forager_vec(w, full), privileged(w)
        hS = S.step(hS, smap, svec, zero64, zeroB)
        tokS, lpS, lenS = silence(B, dev), zeroB.clone(), zeroB.clone()
        if not cfg['silent'] and len(bi):
            tk, lp, _, ln = S.mouth(hS[bi], greedy=not sample)
            tokS[bi] = tk; lpS[bi] = lp; lenS[bi] = ln
        lastS[bi] = tokS[bi]
        vS = S.value(torch.cat((hS, priv), -1)).squeeze(1)
        heardF = tokS
        if 'block' in controls: heardF = silence(B, dev)
        if 'shuffle' in controls: heardF = lastS[perm]
        act = torch.zeros(B, dtype=torch.long, device=dev); lp_act = zeroB.clone(); vF = zeroB.clone()
        for m, s in enumerate(sl):
            Fo = Fos[m]; br = brief[s]
            if br.any():
                rows = br.nonzero().squeeze(1)
                e = embF[s].clone(); e[rows] = Fo.ears(heardF[s][rows]); embF[s] = e
            hF[s] = Fo.step(hF[s], fvec[s], embF[s], br.float())
            dist = Categorical(logits=Fo.policy(hF[s]))
            a = dist.sample() if sample else dist.logits.argmax(-1)
            act[s] = a; lp_act[s] = dist.log_prob(a); vF[s] = Fo.value(torch.cat((hF[s], priv[s]), -1)).squeeze(1)
        pos = w.pos.clone()
        r = w.step(act)
        w.stats['briefings'] += brief.float()
        for k, v in (('smap', smap.to(torch.uint8)), ('svec', svec), ('fvec', fvec), ('priv', priv), ('alive', alive),
                     ('brief', brief), ('tokS', tokS), ('lpS', lpS), ('lenS', lenS), ('vS', vS), ('heardF', heardF),
                     ('act', act), ('lp_act', lp_act), ('vF', vF), ('r', r), ('pos', pos)):
            rec[k].append(v)
    out = {k: torch.stack(v) for k, v in rec.items()}
    out['stats'] = {k: v.clone() for k, v in w.stats.items()}
    return out


def replay_scout(S, d, idx, cfg):
    T = d['r'].shape[0]; B = len(idx); dev = d['r'].device
    g = lambda k: d[k][:, idx]
    brief = g('brief'); bt, bb = brief.nonzero(as_tuple=True)
    seeS = S.see(g('smap').reshape(T * B, *d['smap'].shape[2:]).float()).view(T, B, -1)
    svec = g('svec'); h = torch.zeros(B, S.core, device=dev); hs = []
    pad = torch.zeros(B, 64 + 1, device=dev)
    for t in range(T):
        h = S.memory(torch.cat((seeS[t], svec[t], pad), -1), h); hs.append(h)
    h = torch.stack(hs)
    out = dict(vS=S.value(torch.cat((h, g('priv')), -1)).squeeze(-1))
    z = torch.zeros(T, B, device=dev); out['lpS'] = out['entS'] = z
    if not cfg['silent'] and len(bt):
        _, lp, en, _ = S.mouth(h[bt, bb], forced=g('tokS')[bt, bb])
        out['lpS'], out['entS'] = z.index_put((bt, bb), lp), z.index_put((bt, bb), en)
    return out


def replay_forager(Fo, d, idx):
    T = d['r'].shape[0]; B = len(idx); dev = d['r'].device
    g = lambda k: d[k][:, idx]
    brief = g('brief'); bt, bb = brief.nonzero(as_tuple=True)
    fvec = g('fvec'); seeF = Fo.see(fvec.reshape(T * B, -1)).view(T, B, -1)
    new = torch.zeros(T, B, 64, device=dev)
    if len(bt): new = new.index_put((bt, bb), Fo.ears(g('heardF')[bt, bb]))
    h = torch.zeros(B, Fo.core, device=dev); emb = torch.zeros(B, 64, device=dev); hs = []
    for t in range(T):
        emb = torch.where(brief[t, :, None], new[t], emb)
        h = Fo.memory(torch.cat((seeF[t], emb, brief[t, :, None].float()), -1), h); hs.append(h)
    h = torch.stack(hs)
    dist = Categorical(logits=Fo.policy(h))
    return dict(lp_act=dist.log_prob(g('act')), ent_act=dist.entropy(), vF=Fo.value(torch.cat((h, g('priv')), -1)).squeeze(-1))


def imitate(old, w_fn, n_maps, steps, lr, device, bottleneck_maps=None):
    """A fresh scout learns the old scout's first briefings.  Returns the new scout and its token accuracy."""
    new = Scout().to(device); opt = torch.optim.Adam(new.parameters(), lr=lr)
    fixed = bottleneck_maps
    for i in range(steps):
        w = fixed if fixed is not None else w_fn(n_maps)
        target = first_tokens(old, w)
        smap = w.render(scout=True) if not w.scout_ripe else w.render()
        h = new.step(torch.zeros(w.B, new.core, device=device), smap, scout_vec(w), torch.zeros(w.B, 64, device=device),
                     torch.zeros(w.B, device=device))
        _, lp, _, _ = new.mouth(h, forced=target)
        loss = -lp.mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(new.parameters(), 1.0); opt.step()
    w = w_fn(n_maps)                                  # accuracy on fresh maps: how much of the code carried over
    same = (first_tokens(new, w) == first_tokens(old, w)).all(1).float().mean()
    return new, float(same)


def run(output, rules='private', silent=False, seed=0, updates=2000, B=2048, lr=3e-4, epochs=3, chunks=2,
        ent_act=0.01, ent_talk=0.01, curriculum=0.2, turnover=0, generation=0, bottleneck=512, imitation_steps=300,
        device='cuda', log=25):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); gen = torch.Generator(device=device).manual_seed(1000 + seed)
    S = Scout().to(device); Fos = [Forager().to(device) for _ in range(M)]
    optS = torch.optim.Adam(S.parameters(), lr=lr); optF = [torch.optim.Adam(f.parameters(), lr=lr) for f in Fos]
    cfg = dict(rules=rules, silent=silent)
    conf = dict(cfg, seed=seed, updates=updates, B=B, lr=lr, epochs=epochs, chunks=chunks, ent_act=ent_act, ent_talk=ent_talk,
                curriculum=curriculum, turnover=turnover, generation=generation, bottleneck=bottleneck,
                imitation_steps=imitation_steps, foragers=M)
    (output / 'config.json').write_text(json.dumps(conf, indent=1))
    world = lambda n, items: World(n, rules, device=device, generator=gen).reset(max_items=items, held_out='exclude')
    sl = blocks(B, device); born = [0] * M; events = []
    curve, per_forager = [], []; start = time.perf_counter()
    for u in range(updates):
        frac = u / updates; items = 1 if frac < curriculum else 2
        # transmission
        if turnover and u > 0 and u % turnover == 0:
            m = (u // turnover - 1) % M
            Fos[m] = Forager().to(device); optF[m] = torch.optim.Adam(Fos[m].parameters(), lr=lr); born[m] = u
            events.append(dict(update=u, event='new forager', forager=m)); print(json.dumps(events[-1]), flush=True)
        if generation and not silent and u > 0 and u % generation == 0:
            fixed = world(bottleneck, items) if bottleneck else None
            S, kept = imitate(S, lambda n: world(n, items), bottleneck or 512, imitation_steps, 1e-3, device, fixed)
            optS = torch.optim.Adam(S.parameters(), lr=lr)
            events.append(dict(update=u, event='new scout', same_first_briefing=round(kept, 3))); print(json.dumps(events[-1]), flush=True)
        w = world(B, items)
        d = play(w, S, Fos, cfg)
        alive = d['alive'].float(); spoke = d['brief'].float()
        advS = gae(d['r'], d['vS'], alive); retS = advS + d['vS']
        advF = gae(d['r'], d['vF'], alive); retF = advF + d['vF']
        nS = norm(advS, spoke)
        nF = torch.zeros_like(advF)
        for s in sl: nF[:, s] = norm(advF[:, s], alive[:, s])
        kt = ent_talk * max(0.1, 1 - frac)
        for _ in range(epochs):
            parts = [torch.arange(s.start, s.stop, device=device)[torch.randperm(s.stop - s.start, device=device)].chunk(chunks) for s in sl]
            for c in range(chunks):
                pieces = [p[c] for p in parts]; idx = torch.cat(pieces)
                o = replay_scout(S, d, idx, cfg)
                a, sp = alive[:, idx], spoke[:, idx]
                lossS = 0.5 * masked_mean((o['vS'] - retS[:, idx]) ** 2, a)
                if not silent:
                    lossS = lossS + clip_loss(o['lpS'], d['lpS'][:, idx], nS[:, idx], sp) - kt * masked_mean(o['entS'], sp)
                loss = lossS
                for m, ix in enumerate(pieces):
                    of = replay_forager(Fos[m], d, ix); am = alive[:, ix]
                    loss = loss + clip_loss(of['lp_act'], d['lp_act'][:, ix], nF[:, ix], am) - ent_act * masked_mean(of['ent_act'], am) \
                        + 0.5 * masked_mean((of['vF'] - retF[:, ix]) ** 2, am)
                optS.zero_grad(); [o_.zero_grad() for o_ in optF]
                loss.backward()
                nn.utils.clip_grad_norm_(S.parameters(), 0.5); [nn.utils.clip_grad_norm_(f.parameters(), 0.5) for f in Fos]
                optS.step(); [o_.step() for o_ in optF]
        ret = d['r'].sum(0)
        per_forager.append([round(float(ret[s].mean()), 4) for s in sl])
        if (u + 1) % log == 0 or u == updates - 1:
            st = d['stats']
            row = dict(update=u + 1, ret=float(ret.mean()), **{k: float(v.mean()) for k, v in st.items()},
                       ret_by_forager=per_forager[-1], age=[u + 1 - b for b in born],
                       lenS=float(masked_mean(d['lenS'], spoke)), curriculum=frac < curriculum,
                       seconds=round(time.perf_counter() - start, 1))
            curve.append(row); print(json.dumps(row), flush=True)
            torch.save(dict(scout=S.state_dict(), foragers=[f.state_dict() for f in Fos], config=conf, update=u + 1),
                       output / 'checkpoint_partial.pt')
    torch.save(dict(scout=S.state_dict(), foragers=[f.state_dict() for f in Fos], config=conf), output / 'checkpoint.pt')
    (output / 'curve.json').write_text(json.dumps(curve))
    (output / 'per_forager.json').write_text(json.dumps(dict(returns=per_forager, events=events)))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--rules', default='private', choices=('fixed', 'private', 'shared'))
    p.add_argument('--silent', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=2000); p.add_argument('--B', type=int, default=2048)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--log', type=int, default=25)
    p.add_argument('--turnover', type=int, default=0, help='replace one forager every N updates (0 = never)')
    p.add_argument('--generation', type=int, default=0, help='new scout every N updates (0 = never)')
    p.add_argument('--bottleneck', type=int, default=512, help='maps a new scout learns from (0 = fresh maps, no bottleneck)')
    p.add_argument('--imitation-steps', type=int, default=300)
    p.add_argument('--device', default='cuda'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--mem-fraction', type=float, default=0.5, help='cap on the share of GPU memory this process may use')
    a = p.parse_args(); torch.set_num_threads(a.threads)
    if a.device.startswith('cuda'): torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    try:
        import ctypes; k = ctypes.windll.kernel32; k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    run(a.output, a.rules, a.silent, a.seed, a.updates, a.B, a.lr, turnover=a.turnover, generation=a.generation,
        bottleneck=a.bottleneck, imitation_steps=a.imitation_steps, device=a.device, log=a.log)
