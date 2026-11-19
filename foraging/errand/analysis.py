"""Tests for the errand study (pass lines in README.md).

    python -m foraging.errand.analysis --summarise results/errand/A
"""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from .env import World, GLADE_ORIGIN, LION, TIGER, ASLEEP, CAGED, FOODS, ANIMALS, STATES, GLADES
from .agents import Scout, Forager, STOP
from .train import play, scout_vec, forager_vec, silence
from foraging.metrics import spearman, edit

N = 4096
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(run):
    cp = torch.load(Path(run) / 'checkpoint.pt', map_location=DEV, weights_only=False)
    S, Fo = Scout().to(DEV), Forager().to(DEV); S.load_state_dict(cp['scout']); Fo.load_state_dict(cp['forager'])
    S.eval(); Fo.eval()
    return S, Fo, cp['config']


def cfg_of(conf, silent=None):
    return dict(rules=conf['rules'], phase=conf['phase'], scout_ripe=conf['scout_ripe'], forager_talks=conf['forager_talks'],
                silent=conf['silent'] if silent is None else silent, cost_now=conf['cost'])


def world(conf, seed=0, n=N, held_out='exclude'):
    return World(n, conf['rules'], conf['phase'], conf['scout_ripe'], device=DEV,
                 generator=torch.Generator(device=DEV).manual_seed(seed)).reset(held_out=held_out)


def strip(tokens):
    return [tuple(row[:row.index(STOP)] if STOP in row else row) for row in tokens.tolist()]


# ------------------------------------------------------------------ returns
@torch.no_grad()
def returns(S, Fo, conf, controls=(), seed=0, held_out='exclude', silent=None):
    torch.manual_seed(seed)
    w = world(conf, seed, held_out=held_out)
    d = play(w, S, Fo, cfg_of(conf, silent), controls=controls, keep_maps=False)
    st = d['stats']
    return dict(ret=float(d['r'].sum(0).mean()), **{k: float(v.mean()) for k, v in st.items()})


# ------------------------------------------------------------------ first briefing
@torch.no_grad()
def first_briefing(S, Fo, w, conf):
    B = w.B; full = w.render(); smap = w.render(scout=True) if not w.scout_ripe else full
    embS = torch.zeros(B, 64, device=DEV); fresh = torch.zeros(B, device=DEV)
    if conf['forager_talks'] and not conf['silent']:
        tokF = Fo.mouth(Fo.speaker_state(torch.zeros(B, Fo.core, device=DEV), forager_vec(w, full)), greedy=True)[0]
        embS = S.ears(tokF); fresh = torch.ones(B, device=DEV)
    h = S.step(torch.zeros(B, S.core, device=DEV), smap, scout_vec(w), embS, fresh)
    return strip(S.mouth(h, greedy=True)[0])


def swap_type(state):
    def f(w):
        hit = w.an_on & (w.an_state == state) & (w.an_type == LION)
        first = hit & (hit.cumsum(1) == 1)
        w.an_type = torch.where(first, torch.full_like(w.an_type, TIGER), w.an_type)
        return first.any(1)
    return f


def sensitivity(S, Fo, conf):
    out = {}
    for name, fn in (('lion_to_tiger_asleep', swap_type(ASLEEP)), ('lion_to_tiger_caged', swap_type(CAGED))):
        w = world(conf, seed=11); before = first_briefing(S, Fo, w, conf); mask = fn(w); after = first_briefing(S, Fo, w, conf)
        idx = mask.nonzero().squeeze(1).tolist()
        out[name] = dict(changed=float(np.mean([before[i] != after[i] for i in idx])) if idx else None, n=len(idx))
    return out


# ------------------------------------------------------------------ does the forager decide?
def glade_of(pos):
    """[..., 2] -> glade index, or -1 outside the glades."""
    r, c = pos[..., 0], pos[..., 1]
    top, bot, lef, rig = r <= 3, r >= 7, c <= 3, c >= 7
    g = torch.full_like(r, -1)
    g = torch.where(top & lef, 0, g); g = torch.where(top & rig, 1, g); g = torch.where(bot & lef, 2, g); g = torch.where(bot & rig, 3, g)
    return g


def first_destination(d):
    g = glade_of(d['pos'])                                     # [T, B]
    hit = g >= 0
    first = hit.float().argmax(0)
    return torch.where(hit.any(0), g.gather(0, first[None]).squeeze(0), torch.full_like(first, -1))


@torch.no_grad()
def forager_decides(S, Fo, conf):
    if conf['rules'] == 'fixed': return None
    w = world(conf, seed=13); snap = w.snapshot()
    danger_before = w.danger()
    d1 = play(w, S, Fo, cfg_of(conf), sample=False, keep_maps=False); dest1 = first_destination(d1)
    w.restore(snap); w.fed = ~w.fed; danger_after = w.danger()
    d2 = play(w, S, Fo, cfg_of(conf), sample=False, keep_maps=False); dest2 = first_destination(d2)
    matters = (danger_before != danger_after).any(1)
    went = (dest1 >= 0) | (dest2 >= 0)
    sel = matters & went
    changed = (dest1 != dest2)[sel].float().mean() if sel.any() else torch.tensor(float('nan'))
    # how often the first destination is a dangerous glade, with the true flags
    gi = dest1.clamp_min(0)
    into_danger = danger_before.gather(1, gi[:, None]).squeeze(1)[dest1 >= 0].float().mean()
    return dict(changed=float(changed), n=int(sel.sum()), first_into_danger=float(into_danger),
                went_out=float((dest1 >= 0).float().mean()))


# ------------------------------------------------------------------ structure
def scene_features(w):
    """Per map: for each glade, the animal (none/lion/tiger), its state, and the food in it (type and, if the
    scout sees it, ripeness).  Returns a list of tuples, one entry per (glade, factor)."""
    feats = []
    on_, at, st = w.an_on.tolist(), w.an_type.tolist(), w.an_state.tolist()
    ft, fr, fg = w.food_type.tolist(), w.food_ripe.tolist(), w.food_glade.tolist()
    for b in range(w.B):
        f = []
        for g in range(4):
            on = on_[b][g]
            f.append(at[b][g] if on else -1); f.append(st[b][g] if on else -1)
            foods = sorted((ft[b][i], fr[b][i] if w.scout_ripe else None) for i in range(len(ft[b])) if fg[b][i] == g)
            f.append(tuple(foods))
        feats.append(tuple(f))
    return feats


def structure(S, Fo, conf, n=2048):
    w = world(conf, seed=17, n=n); msgs = first_briefing(S, Fo, w, conf); feats = scene_features(w)
    rng = np.random.default_rng(0); i, j = rng.integers(n, size=5000), rng.integers(n, size=5000)
    mdist = [sum(a != b for a, b in zip(feats[x], feats[y])) for x, y in zip(i, j)]
    sdist = [edit(msgs[x], msgs[y]) for x, y in zip(i, j)]
    examples = [dict(scene=w.describe(b), says=' '.join(map(str, msgs[b])) or '(silence)') for b in range(10)]
    return dict(topsim=spearman(mdist, sdist), distinct=len(set(msgs)), mean_length=float(np.mean([len(m) for m in msgs])),
                silent_share=float(np.mean([len(m) == 0 for m in msgs])), examples=examples)


# ------------------------------------------------------------------ one run, and the study
def analyse(run, silent_run=None):
    run = Path(run); S, Fo, conf = load(run)
    res = dict(config=conf, returns={})
    for name, controls in (('normal', ()), ('blocked', ('block',)), ('shuffled', ('shuffle',))):
        res['returns'][name] = returns(S, Fo, conf, controls)
    res['returns']['held_out'] = returns(S, Fo, conf, held_out='only')
    if conf['forager_talks']: res['returns']['forager_silenced'] = returns(S, Fo, conf, ('block_forager',))
    if silent_run:
        sS, sF, sconf = load(silent_run)
        res['returns']['silent_control'] = returns(sS, sF, sconf)
        res['returns']['silent_held_out'] = returns(sS, sF, sconf, held_out='only')
    res['sensitivity'] = sensitivity(S, Fo, conf)
    res['forager_decides'] = forager_decides(S, Fo, conf)
    res['structure'] = structure(S, Fo, conf)
    (run / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def verdict(r):
    R = r['returns']; v = {}
    if 'silent_control' in R:
        gain = R['normal']['ret'] - R['silent_control']['ret']
        lost = min(R['normal']['ret'] - R['blocked']['ret'], R['normal']['ret'] - R['shuffled']['ret'])
        v['1_useful'] = gain >= 0.3 and lost >= 0.5 * gain
        v['4_held_out'] = R['held_out']['ret'] - R['silent_held_out']['ret'] >= 0.3
    if r['forager_decides'] is not None:
        v['2_forager_decides'] = (r['forager_decides']['changed'] or 0) >= 0.5
    s = r['sensitivity']
    v['3_animals'] = (s['lion_to_tiger_asleep']['changed'] or 0) >= 0.7 and (s['lion_to_tiger_caged']['changed'] if s['lion_to_tiger_caged']['changed'] is not None else 1) <= 0.3
    v['5_topsim'] = (r['structure']['topsim'] or 0) >= 0.3
    return v


def summarise(root):
    root = Path(root); out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and (p / 'checkpoint.pt').exists() and '_silent' not in p.name):
        base, seed = run.name.rsplit('_', 1)
        silent = root / f'{base}_silent_{seed}'
        r = analyse(run, silent if (silent / 'checkpoint.pt').exists() else None)
        v = verdict(r); out[run.name] = dict(verdict=v, **{k: r[k] for k in ('returns', 'sensitivity', 'forager_decides', 'structure')})
        R = r['returns']; fd = r['forager_decides'] or {}
        print(f"{run.name:24s} ret {R['normal']['ret']:+.2f} blk {R['blocked']['ret']:+.2f} shuf {R['shuffled']['ret']:+.2f} "
              f"silent {R.get('silent_control', {}).get('ret', float('nan')):+.2f} | held-out {R['held_out']['ret']:+.2f} vs "
              f"{R.get('silent_held_out', {}).get('ret', float('nan')):+.2f} | flip {fd.get('changed', float('nan')):.2f} | "
              f"asleep {r['sensitivity']['lion_to_tiger_asleep']['changed'] or 0:.2f} caged {r['sensitivity']['lion_to_tiger_caged']['changed'] or 0:.2f} | "
              f"len {r['structure']['mean_length']:.1f} topsim {r['structure']['topsim']:.2f} | "
              + ' '.join(f"{k}={'PASS' if x else 'fail'}" for k, x in v.items()), flush=True)
    (root / 'summary.json').write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--summarise'); p.add_argument('--run'); p.add_argument('--silent-run')
    a = p.parse_args()
    if a.summarise: summarise(a.summarise)
    else:
        r = analyse(a.run, a.silent_run)
        print(json.dumps({k: r[k] for k in ('returns', 'sensitivity', 'forager_decides')}, indent=1, default=float))
        print(json.dumps({k: v for k, v in r['structure'].items() if k != 'examples'}, indent=1, default=float))
        for e in r['structure']['examples'][:5]: print(e)
        print(verdict(r))
