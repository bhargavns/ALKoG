"""Does the scout's first briefing contain reusable parts?

    python -m foraging.errand.words RUN [RUN ...]

For each run, on fresh maps:
  * per-symbol meaning: how much each symbol's presence tells about each fact of the scene
    (normalised mutual information, 0 = nothing, 1 = determines it);
  * bag-of-symbols and positional disentanglement (Chaabouni et al. 2020): does each symbol,
    or each position, carry one fact rather than several?
  * decoding: how well each fact can be read back from the message, against always guessing
    the most common value;
  * minimal pairs: change one fact on one glade and record which positions of the message change.
"""
from pathlib import Path
import argparse, json
from collections import Counter
import numpy as np
import torch
from . import analysis as A
from .env import LION, TIGER, ASLEEP, AWAKE, CAGED, GLADES
from .agents import VOCAB

END = VOCAB               # padding symbol after the message ends
L = 8


def facts(w):
    """Per map: a dict of discrete facts the scout can see."""
    on, at, st = w.an_on.cpu().numpy(), w.an_type.cpu().numpy(), w.an_state.cpu().numpy()
    ripe, glade = w.food_ripe.cpu().numpy(), w.food_glade.cpu().numpy()
    F = {}
    for g, name in enumerate(GLADES):
        F[f'{name} animal'] = np.where(on[:, g], at[:, g] + 1, 0)                 # 0 none, 1 lion, 2 tiger
        F[f'{name} state'] = np.where(on[:, g], st[:, g] + 1, 0)                  # 0 none, 1 asleep, 2 awake, 3 caged
        risky = on[:, g] & ((st[:, g] == AWAKE) | ((st[:, g] == ASLEEP) & (at[:, g] == TIGER)))
        F[f'{name} risky'] = risky.astype(int)
        F[f'{name} ripe food'] = (ripe & (glade == g)).sum(1)                     # 0, 1, 2
    risky = np.stack([F[f'{n} risky'] for n in GLADES], 1)
    F['number of risky glades'] = risky.sum(1)
    F['any risky tiger'] = np.stack([F[f'{n} risky'] & (F[f'{n} animal'] == 2) for n in GLADES], 1).any(1).astype(int)
    F['any risky lion'] = np.stack([F[f'{n} risky'] & (F[f'{n} animal'] == 1) for n in GLADES], 1).any(1).astype(int)
    return F


def padded(msgs):
    return np.array([list(m[:L]) + [END] * (L - len(m[:L])) for m in msgs])


def H(x):
    _, c = np.unique(x, return_counts=True, axis=0); p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def MI(x, y):
    return H(x) + H(y) - H(np.stack([x, y], 1))


def nmi(x, y):
    h = H(y)
    return MI(x, y) / h if h > 0 else 0.0


def disentanglement(units, F):
    """Mean over units with entropy of (top MI - second MI) / H(unit)."""
    names = list(F); scores, weights = [], []
    for u in units:
        h = H(u)
        if h < 1e-6: continue
        mi = sorted((MI(u, F[n]) for n in names), reverse=True)
        scores.append((mi[0] - mi[1]) / h); weights.append(h)
    return float(np.average(scores, weights=weights)) if scores else 0.0


def decode(X, y, seed=0, steps=300):
    """Multinomial logistic regression on message features; held-out accuracy and majority baseline."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    classes = np.unique(y); yi = np.searchsorted(classes, y)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(yi)
    lin = torch.nn.Linear(X.shape[1], len(classes)); opt = torch.optim.Adam(lin.parameters(), 0.05, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(Xt[tr]), yt[tr]).backward(); opt.step()
    with torch.no_grad(): acc = float((lin(Xt[te]).argmax(1) == yt[te]).float().mean())
    base = Counter(yi[tr]).most_common(1)[0][0]
    return acc, float((yi[te] == base).mean())


def features(P):
    bag = np.stack([(P == s).sum(1) for s in range(VOCAB)], 1)
    pos = np.concatenate([(P[:, j:j + 1] == np.arange(VOCAB + 1)).astype(float) for j in range(L)], 1)
    return np.concatenate([bag, pos], 1)


@torch.no_grad()
def minimal_pairs(S, Fo, conf, n=1000):
    """Which message positions change when one fact on one glade changes (from an empty glade)."""
    out = {}
    edits = [('add awake tiger', TIGER, AWAKE), ('add awake lion', LION, AWAKE), ('add sleeping tiger', TIGER, ASLEEP),
             ('add sleeping lion', LION, ASLEEP), ('add caged tiger', TIGER, CAGED)]
    for g, gname in enumerate(GLADES):
        w = A.world(conf, seed=31, n=n); w.an_on[:, g] = False
        before = padded(A.first_briefing(S, Fo, w, conf))
        for name, at, st in edits:
            w.an_on[:, g] = True; w.an_type[:, g] = at; w.an_state[:, g] = st
            after = padded(A.first_briefing(S, Fo, w, conf))
            out[f'{gname}: {name}'] = dict(any=float((before != after).any(1).mean()),
                                         by_position=[round(float(v), 2) for v in (before != after).mean(0)])
            w.an_on[:, g] = False
    return out


def run_words(run, n=4096):
    S, Fo, conf = A.load(run)
    w = A.world(conf, seed=23, n=n); F = facts(w)
    msgs = A.first_briefing(S, Fo, w, conf); P = padded(msgs)
    present = {s: (P == s).any(1).astype(int) for s in range(VOCAB) if (P == s).any()}
    meaning = {}
    for s, x in present.items():
        ranked = sorted(((nmi(x, F[k]), k) for k in F), reverse=True)[:3]
        meaning[str(s)] = dict(used_in=round(float(x.mean()), 2), tells=[(k, round(v, 2)) for v, k in ranked])
    X = features(P)
    dec = {k: dict(zip(('accuracy', 'baseline'), map(lambda v: round(v, 3), decode(X, F[k])))) for k in F}
    res = dict(run=str(run), maps=n, distinct=len(set(map(tuple, msgs))),
               bosdis=round(disentanglement([(P == s).sum(1) for s in range(VOCAB)], F), 3),
               posdis=round(disentanglement([P[:, j] for j in range(L)], F), 3),
               symbols=meaning, decoding=dec, minimal_pairs=minimal_pairs(S, Fo, conf))
    return res


def show(r):
    print(f"\n=== {r['run']}  distinct {r['distinct']}/{r['maps']}  bosdis {r['bosdis']}  posdis {r['posdis']}")
    print("symbol  used-in  what it tells about (normalised MI)")
    for s, m in sorted(r['symbols'].items(), key=lambda kv: -kv[1]['used_in']):
        print(f"  {s:>3}   {m['used_in']:.2f}    " + '; '.join(f"{k} {v}" for k, v in m['tells']))
    print("decoding from the message: accuracy (always-majority baseline)")
    for k, d in r['decoding'].items(): print(f"  {k:26s} {d['accuracy']:.2f} ({d['baseline']:.2f})")
    print("minimal pairs: share of maps where the message changes, overall and by position 1..8")
    for k, d in r['minimal_pairs'].items(): print(f"  {k:28s} {d['any']:.2f}  {d['by_position']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+'); p.add_argument('--out')
    a = p.parse_args()
    results = [run_words(r) for r in a.runs]
    for r in results: show(r)
    if a.out: Path(a.out).write_text(json.dumps(results, indent=1))
