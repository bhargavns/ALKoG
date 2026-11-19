"""Tests for the watchtower study (pass lines in README.md).

    python -m experiments.watchtower.analysis --summarise results/watchtower
"""
from collections import Counter
from pathlib import Path
import argparse, json
import numpy as np
import torch
from .env import World, FOODS, ANIMALS, STATES, GLADES, LION, TIGER, ASLEEP, CAGED, GUARDING
from .agents import Scout, Forager, STOP
from .train import play, scout_vec, silence
from experiments.grammar_pressures.structure import spearman, edit

N = 2048


def load(run):
    cp = torch.load(Path(run) / 'checkpoint.pt', map_location='cpu', weights_only=False)
    S, Fo = Scout(), Forager(); S.load_state_dict(cp['scout']); Fo.load_state_dict(cp['forager']); S.eval(); Fo.eval()
    return S, Fo, cp['config']


def strip(tokens):
    out = []
    for row in tokens.tolist(): out.append(tuple(row[:row.index(STOP)] if STOP in row else row))
    return out


def world(stage, n=N, seed=0):
    return World(n, stage, generator=torch.Generator().manual_seed(seed)).reset()


# ------------------------------------------------------------------ returns
@torch.no_grad()
def returns(S, Fo, conf, controls=(), seed=0, silent=None):
    torch.manual_seed(seed)
    w = world(conf['stage'], seed=seed); kind = (w.hunger_s >= 0).clone()
    d = play(w, S, Fo, dict(stage=conf['stage'], silent=conf['silent'] if silent is None else silent, cost_now=conf['cost']),
             controls=controls, keep_maps=False)
    ret = d['r'].sum(0); st = d['stats']
    out = dict(ret=float(ret.mean()), wanted=float(st['wanted'].mean()), rotten=float(st['rotten'].mean()),
               lion=float(st['lion'].mean()), tiger=float(st['tiger'].mean()), delivered=float(st['delivered'].mean()),
               starved=float(st['starved'].mean()))
    if conf['stage'] == 2:
        out['ret_forager_hungry'] = float(ret[~kind].mean()); out['ret_scout_hungry'] = float(ret[kind].mean())
    return out


# ------------------------------------------------------------------ first utterances
@torch.no_grad()
def first_words(S, w):
    smap = w.render(scout=True)
    h = S.step(torch.zeros(w.B, S.core), smap, scout_vec(w), silence(w.B, 'cpu'), torch.zeros(w.B))
    return strip(S.mouth(h, greedy=True)[0])


def changed(S, w, edit_fn):
    """Fraction of maps whose first utterance changes under `edit_fn` (applied where it returns a mask)."""
    before = first_words(S, w); snap = w.snapshot(); mask = edit_fn(w)
    after = first_words(S, w); w.restore(snap)
    idx = mask.nonzero().squeeze(1).tolist()
    return (float(np.mean([before[i] != after[i] for i in idx])) if idx else None), len(idx)


def swap_type(state):
    def f(w):
        hit = w.pred_on & (w.pred_state == state) & (w.pred_type == LION)
        first = hit & (hit.cumsum(1) == 1)
        w.pred_type = torch.where(first, torch.full_like(w.pred_type, TIGER), w.pred_type)
        return first.any(1)
    return f


def flip_ripe(w):
    hit = w.food_alive & w.food_ripe; first = hit & (hit.cumsum(1) == 1)
    w.food_ripe = w.food_ripe & ~first
    return first.any(1)


def sensitivity(S, stage):
    w = world(stage, seed=11)
    out = {}
    for name, fn in (('lion_to_tiger_asleep', swap_type(ASLEEP)), ('lion_to_tiger_caged', swap_type(CAGED)),
                     ('lion_to_tiger_guarding', swap_type(GUARDING)), ('ripe_to_rotten', flip_ripe)):
        if stage == 3 and name == 'ripe_to_rotten': continue        # the scout cannot see ripeness
        rate, n = changed(S, w, fn); out[name] = dict(changed=rate, n=n)
    return out


# ------------------------------------------------------------------ one occupied glade
def single_glade_maps(stage, reps=4, seed=5):
    configs = [(g, f, r, p, s) for g in range(4) for f in range(3) for r in (True, False)
               for p in (None, LION, TIGER) for s in (range(4) if p is not None else [None])]
    n = len(configs) * reps
    w = World(n, stage, generator=torch.Generator().manual_seed(seed)).reset()
    w.food_alive[:] = False; w.pred_on[:] = False
    for i, (g, f, r, p, s) in enumerate(configs * reps):
        w.food_alive[i, g] = True; w.food_type[i, g] = f; w.food_ripe[i, g] = r
        if p is not None: w.pred_on[i, g] = True; w.pred_type[i, g] = p; w.pred_state[i, g] = s
    w.pred_pos = w._place_predators()
    w.hunger_s[:] = -1; w.s_done[:] = True
    return w, configs * reps


def structure(S, stage):
    w, meta = single_glade_maps(stage)
    msgs = first_words(S, w)
    factors = {'glade': [m[0] for m in meta], 'food': [m[1] for m in meta], 'ripeness': [m[2] for m in meta],
               'predator': [-1 if m[3] is None else m[3] for m in meta], 'state': [m[4] for m in meta]}
    if stage == 3: factors.pop('ripeness')
    rng = np.random.default_rng(0); n = len(msgs)
    i, j = rng.integers(n, size=4000), rng.integers(n, size=4000)
    mdist = [sum(factors[k][a] != factors[k][b] for k in factors) for a, b in zip(i, j)]
    sdist = [edit(msgs[a], msgs[b]) for a, b in zip(i, j)]
    out = dict(topsim=spearman(mdist, sdist), distinct=len(set(msgs)), mean_length=float(np.mean([len(m) for m in msgs])),
               silent_share=float(np.mean([len(m) == 0 for m in msgs])), markers={})
    for k, vals in factors.items():
        use = [x is not None for x in vals]
        best = (0., None, None)
        for v in sorted({x for x in vals if x is not None}, key=str):
            inside = [m for m, x, u in zip(msgs, vals, use) if u and x == v]; outside = [m for m, x, u in zip(msgs, vals, use) if u and x != v]
            for sym in range(16):
                gap = np.mean([sym in m for m in inside]) - np.mean([sym in m for m in outside])
                if gap > best[0]: best = (float(gap), sym, v)
        out['markers'][k] = dict(gap=best[0], symbol=best[1], value=str(best[2]))
    out['examples'] = [dict(map=f"{GLADES[g]}: {'ripe' if r else 'rotten'} {FOODS[f]}" + (f", {ANIMALS[p]} {STATES[s]}" if p is not None else ''),
                            says=' '.join(map(str, msgs[k]))) for k, (g, f, r, p, s) in enumerate(meta[:len(meta) // 4]) if k % 7 == 0][:24]
    return out


# ------------------------------------------------------------------ stage 2: whose hunger
def hunger_marker(S):
    w = world(2, seed=21); h = w._sample_present()
    snap = w.snapshot()
    w.hunger_s[:] = h; w.hunger_f[:] = -1; mine = first_words(S, w); w.restore(snap)
    w.hunger_s[:] = -1; w.hunger_f[:] = h; theirs = first_words(S, w)
    gaps = {sym: np.mean([sym in m for m in mine]) - np.mean([sym in m for m in theirs]) for sym in range(16)}
    sym = max(gaps, key=gaps.get)
    rm = lambda m: set(m) - {sym}
    jac = [len(rm(a) & rm(b)) / max(1, len(rm(a) | rm(b))) for a, b in zip(mine, theirs)]
    per_food = {}
    for f in range(3):
        sel = (h == f).tolist()
        g2 = {s: np.mean([s in m for m, x in zip(mine, sel) if x]) - np.mean([s in m for m, x in zip(mine, sel) if not x]) for s in range(16)}
        b = max(g2, key=g2.get); per_food[FOODS[f]] = dict(symbol=b, gap=float(g2[b]))
    return dict(symbol=sym, in_scout_hungry=float(np.mean([sym in m for m in mine])), in_forager_hungry=float(np.mean([sym in m for m in theirs])),
                jaccard_without_marker=float(np.mean(jac)), identical=float(np.mean([a == b for a, b in zip(mine, theirs)])),
                food_symbols=per_food, examples=[dict(scout_hungry=' '.join(map(str, a)), forager_hungry=' '.join(map(str, b)))
                                                 for a, b in list(zip(mine, theirs))[:8]])


# ------------------------------------------------------------------ stage 3: shared vocabulary
@torch.no_grad()
def shared_vocabulary(S, Fo, conf):
    torch.manual_seed(3)
    w = world(3, seed=31); hs, hf = w.hunger_s.clone(), w.hunger_f.clone()
    d = play(w, S, Fo, dict(stage=3, silent=False, cost_now=conf['cost']), sample=False, keep_maps=False)
    scout, forager = strip(d['S_tokS'][0]), strip(d['tokF'][2])
    def assoc(msgs, hunger):
        res = {}
        for f in range(3):
            sel = (hunger == f).tolist()
            g = {s: np.mean([s in m for m, x in zip(msgs, sel) if x]) - np.mean([s in m for m, x in zip(msgs, sel) if not x]) for s in range(16)}
            b = max(g, key=g.get); res[FOODS[f]] = dict(symbol=b, gap=float(g[b]))
        return res
    a_s, a_f = assoc(scout, hs), assoc(forager, hf)
    same = sum(a_s[f]['symbol'] == a_f[f]['symbol'] and a_s[f]['gap'] > 0.2 and a_f[f]['gap'] > 0.2 for f in FOODS)
    return dict(scout=a_s, forager=a_f, shared_foods=int(same),
                forager_length=float(np.mean([len(m) for m in forager])), scout_length=float(np.mean([len(m) for m in scout])),
                examples=[dict(scout_wants=FOODS[int(hs[i])], forager_wants=FOODS[int(hf[i])], scout_says=' '.join(map(str, scout[i])),
                               forager_says=' '.join(map(str, forager[i]))) for i in range(8)])


# ------------------------------------------------------------------ one run, and the study
def analyse(run, silent_run=None):
    run = Path(run); S, Fo, conf = load(run); stage = conf['stage']
    res = dict(config=conf, returns={})
    for name, controls in (('normal', ()), ('blocked', ('block',)), ('shuffled', ('shuffle',))) + ((('forager_silenced', ('block_forager',)),) if stage == 3 else ()):
        res['returns'][name] = returns(S, Fo, conf, controls)
    if silent_run:
        sS, sF, sconf = load(silent_run); res['returns']['silent_control'] = returns(sS, sF, sconf)
    res['sensitivity'] = sensitivity(S, stage)
    res['structure'] = structure(S, stage)
    if stage == 2: res['hunger_marker'] = hunger_marker(S)
    if stage == 3: res['shared'] = shared_vocabulary(S, Fo, conf)
    (run / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def verdict(r):
    R = r['returns']; stage = r['config']['stage']; v = {}
    if 'silent_control' in R:
        gain = R['normal']['ret'] - R['silent_control']['ret']
        lost = min(R['normal']['ret'] - R['blocked']['ret'], R['normal']['ret'] - R['shuffled']['ret'])
        v['1_useful'] = gain >= 0.2 and lost >= 0.5 * gain
        if stage == 2:
            v['1_useful'] = v['1_useful'] and all(R['normal'][k] - R['silent_control'][k] >= 0.2 for k in ('ret_forager_hungry', 'ret_scout_hungry'))
        if stage == 3:
            v['1_useful'] = gain >= 0.2 and R['normal']['ret'] - R['forager_silenced']['ret'] >= 0.1
    if stage == 1:
        s = r['sensitivity']
        v['2_lion_tiger'] = (s['lion_to_tiger_asleep']['changed'] or 0) >= 0.7 and (s['lion_to_tiger_caged']['changed'] or 1) <= 0.3
        st = r['structure']
        v['3_structure'] = st['topsim'] >= 0.3 and all(m['gap'] >= 0.5 for m in st['markers'].values())
    if stage == 2:
        h = r['hunger_marker']
        v['2_marker'] = h['in_scout_hungry'] >= 0.9 and h['in_forager_hungry'] <= 0.1
        v['3_reuse'] = h['jaccard_without_marker'] >= 0.5
    if stage == 3:
        v['2_shared'] = r['shared']['shared_foods'] >= 2
    v['all'] = all(v.values()) if v else False
    return v


def summarise(root):
    root = Path(root); out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / 'checkpoint.pt').exists() and '_silent' not in p.name):
        stage, seed = run.name.split('_')[0], run.name.rsplit('_', 1)[1]
        silent = root / f'{stage}_silent_{seed}'
        r = analyse(run, silent if (silent / 'checkpoint.pt').exists() else None)
        v = verdict(r); out[run.name] = dict(verdict=v, returns=r['returns'], sensitivity=r['sensitivity'],
                                             structure={k: r['structure'][k] for k in ('topsim', 'distinct', 'mean_length', 'silent_share', 'markers')},
                                             hunger_marker=r.get('hunger_marker'), shared=r.get('shared'))
        R = r['returns']
        print(f"{run.name:8s} ret {R['normal']['ret']:+.2f} blk {R['blocked']['ret']:+.2f} shuf {R['shuffled']['ret']:+.2f} "
              f"silent {R.get('silent_control', {}).get('ret', float('nan')):+.2f} | len {r['structure']['mean_length']:.1f} "
              f"topsim {r['structure']['topsim']:.2f} | " + ' '.join(f"{k}={'PASS' if x else 'fail'}" for k, x in v.items()), flush=True)
    (root / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--summarise'); p.add_argument('--run'); p.add_argument('--silent-run')
    a = p.parse_args()
    if a.summarise: summarise(a.summarise)
    else:
        r = analyse(a.run, a.silent_run); print(json.dumps({k: r[k] for k in ('returns', 'sensitivity')}, indent=1, default=float))
        print(json.dumps({k: v for k, v in r['structure'].items() if k != 'examples'}, indent=1, default=float)); print(verdict(r))
