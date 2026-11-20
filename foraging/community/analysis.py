"""Tests for the community study (pass lines in README.md).

    python -m foraging.community.analysis --summarise results/community/X
"""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from foraging.errand.env import AWAKE, ASLEEP, TIGER, GLADES, FOODS, FOOD_PER_GLADE
from foraging.errand.words import padded, H, MI, disentanglement, decode, features
from foraging.metrics import spearman, edit
from .env import World, M
from .agents import Scout, Forager, STOP
from .train import play, first_tokens, blocks

N = 4096
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(run):
    cp = torch.load(Path(run) / 'checkpoint.pt', map_location=DEV, weights_only=False)
    S = Scout().to(DEV); S.load_state_dict(cp['scout']); S.eval()
    Fos = []
    for sd in cp['foragers']:
        f = Forager().to(DEV); f.load_state_dict(sd); f.eval(); Fos.append(f)
    return S, Fos, cp['config']


def world(conf, seed=0, n=N, held_out='exclude'):
    return World(n, conf['rules'], device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)).reset(held_out=held_out)


@torch.no_grad()
def returns(S, Fos, conf, controls=(), seed=0, held_out='exclude'):
    torch.manual_seed(seed)
    w = world(conf, seed, held_out=held_out)
    d = play(w, S, Fos, dict(rules=conf['rules'], silent=conf['silent']), controls=controls)
    ret = d['r'].sum(0)
    out = dict(ret=float(ret.mean()), **{k: float(v.mean()) for k, v in d['stats'].items()})
    out['ret_by_forager'] = [round(float(ret[s].mean()), 3) for s in blocks(w.B, DEV)]
    return out


def strip(tokens):
    return [tuple(row[:row.index(STOP)] if STOP in row else row) for row in tokens.tolist()]


def facts(w):
    on, at, st = (x.cpu().numpy() for x in (w.an_on, w.an_type, w.an_state))
    alive, ripe, ft, fg = (x.cpu().numpy() for x in (w.food_alive, w.food_ripe, w.food_type, w.food_glade))
    F = {}
    for g, name in enumerate(GLADES):
        F[f'{name} animal'] = np.where(on[:, g], at[:, g] + 1, 0)
        F[f'{name} state'] = np.where(on[:, g], st[:, g] + 1, 0)
        F[f'{name} risky'] = (on[:, g] & ((st[:, g] == AWAKE) | ((st[:, g] == ASLEEP) & (at[:, g] == TIGER)))).astype(int)
        for f, food in enumerate(FOODS):
            F[f'{name} ripe {food}'] = (alive & ripe & (fg == g) & (ft == f)).any(1).astype(int)
    return F


def scene(w):
    """Per map, the tuple of facts used for topographic similarity."""
    F = facts(w); keys = list(F)
    return [tuple(F[k][b] for k in keys) for b in range(w.B)]


def language(S, conf, n=N, seed=23):
    w = world(conf, seed=seed, n=n)
    msgs = strip(first_tokens(S, w)); P = padded(msgs); F = facts(w)
    X = features(P)
    dec = {}
    for k, y in F.items():
        if len(np.unique(y)) < 2: continue
        acc, base = decode(X, y); dec[k] = dict(accuracy=round(acc, 3), baseline=round(base, 3), gain=round(acc - base, 3))
    sc = scene(w); rng = np.random.default_rng(0); i, j = rng.integers(n, size=5000), rng.integers(n, size=5000)
    topsim = spearman([sum(a != b for a, b in zip(sc[x], sc[y])) for x, y in zip(i, j)], [edit(msgs[x], msgs[y]) for x, y in zip(i, j)])
    units_b = [(P == s).sum(1) for s in range(16)]; units_p = [P[:, k] for k in range(P.shape[1])]
    examples = [dict(scene=w.describe(b), says=' '.join(map(str, msgs[b])) or '(silence)') for b in range(12)]
    return dict(topsim=topsim, bosdis=disentanglement(units_b, F), posdis=disentanglement(units_p, F),
                distinct=len(set(msgs)), mean_length=float(np.mean([len(m) for m in msgs])),
                silent_share=float(np.mean([len(m) == 0 for m in msgs])), decoding=dec, examples=examples)


def learning_speed(run):
    """For each forager that joined mid-run: updates until its return reaches 80% of the others'."""
    p = Path(run) / 'per_forager.json'
    if not p.exists(): return None
    j = json.loads(p.read_text()); R = np.array(j['returns']); out = []
    for e in j['events']:
        if e['event'] != 'new forager': continue
        u0, m = e['update'], e['forager']; others = [k for k in range(R.shape[1]) if k != m]
        need = None
        for u in range(u0, len(R)):
            ref = R[u, others].mean()
            if ref > 0 and R[u, m] >= 0.8 * ref: need = u - u0; break
        out.append(dict(update=u0, forager=m, updates_to_80=need))
    return out


def analyse(run, silent_run=None):
    run = Path(run); S, Fos, conf = load(run)
    res = dict(config=conf, returns={})
    for name, controls in (('normal', ()), ('blocked', ('block',)), ('shuffled', ('shuffle',))):
        res['returns'][name] = returns(S, Fos, conf, controls)
    res['returns']['held_out'] = returns(S, Fos, conf, held_out='only')
    if silent_run:
        sS, sF, sconf = load(silent_run)
        res['returns']['silent_control'] = returns(sS, sF, sconf)
        res['returns']['silent_held_out'] = returns(sS, sF, sconf, held_out='only')
    res['language'] = language(S, conf)
    res['learning_speed'] = learning_speed(run)
    (run / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def verdict(r):
    R, L = r['returns'], r['language']; v = {}
    if 'silent_control' in R:
        gain = R['normal']['ret'] - R['silent_control']['ret']
        lost = min(R['normal']['ret'] - R['blocked']['ret'], R['normal']['ret'] - R['shuffled']['ret'])
        v['1_useful'] = gain >= 0.3 and lost >= 0.5 * gain
        v['3_held_out'] = R['held_out']['ret'] - R['silent_held_out']['ret'] >= 0.3
    d = L['decoding']
    risky = all(d.get(f'{g} risky', {}).get('gain', 0) >= 0.10 for g in GLADES)
    food = sum(d.get(f'{g} ripe {f}', {}).get('gain', 0) >= 0.05 for g in GLADES for f in FOODS)
    v['2_coverage'] = risky and food >= 8
    v['4_structure'] = max(L['posdis'], L['bosdis']) >= 0.15 and (L['topsim'] or 0) >= 0.3
    return v


def speed_summary(ls):
    if not ls: return None
    xs = [e['updates_to_80'] if e['updates_to_80'] is not None else 400 for e in ls]
    h = len(xs) // 2
    return dict(first_half=float(np.median(xs[:h])) if h else None, second_half=float(np.median(xs[h:])), n=len(xs))


def summarise(root):
    root = Path(root); out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / 'checkpoint.pt').exists() and not p.name.startswith('silent')):
        base, seed = run.name.rsplit('_', 1)
        silent = root / f'silent_{seed}'
        r = analyse(run, silent if (silent / 'checkpoint.pt').exists() else None)
        v = verdict(r); R, L = r['returns'], r['language']; sp = speed_summary(r['learning_speed'])
        d = L['decoding']
        risky = ' '.join(f"{g}{d.get(f'{g} risky', {}).get('gain', 0):+.2f}" for g in GLADES)
        out[run.name] = dict(verdict=v, returns=R, language={k: x for k, x in L.items() if k != 'examples'},
                             examples=L['examples'], learning_speed=sp)
        print(f"{run.name:22s} ret {R['normal']['ret']:+.2f} blk {R['blocked']['ret']:+.2f} shuf {R['shuffled']['ret']:+.2f} "
              f"silent {R.get('silent_control', {}).get('ret', float('nan')):+.2f} | held-out {R['held_out']['ret']:+.2f} vs "
              f"{R.get('silent_held_out', {}).get('ret', float('nan')):+.2f} | risky decode {risky} | "
              f"topsim {L['topsim']:.2f} posdis {L['posdis']:.2f} bosdis {L['bosdis']:.2f} len {L['mean_length']:.1f} | "
              f"new-forager speed {sp} | " + ' '.join(f"{k}={'PASS' if x else 'fail'}" for k, x in v.items()), flush=True)
    (root / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--summarise'); p.add_argument('--run'); p.add_argument('--silent-run')
    a = p.parse_args()
    if a.summarise: summarise(a.summarise)
    else:
        r = analyse(a.run, a.silent_run); print(json.dumps({k: r[k] for k in ('returns',)}, indent=1)); print(verdict(r))
