"""Graded structure measures that do not depend on exact word matches.

    python -m experiments.grammar_pressures.structure RUN_DIR [RUN_DIR ...]

    used      signals needed to cover 95% of everything said
    topsim2/3 Spearman correlation between how many of the contrasted values two
              situations share and the edit distance of what is said about them
              (same contrasted slots only); 1 = similar meanings sound similar
    part2     two-contrast utterances: share whose symbols are exactly predicted
              by "first part depends on one value, second part on the other"
              fitted per slot pair on half the data and tested on the other half
"""
from collections import Counter, defaultdict
import sys, json
from pathlib import Path
from pathlib import Path
import numpy as np
import torch
from .analysis import accuracy
from .analysis import Informant, contexts_for
from .train import load


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float); order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
        for u in np.unique(v): r[v == u] = r[v == u].mean()       # average ranks for ties
        return r
    if len(set(x)) < 2 or len(set(y)) < 2: return float('nan')
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1): prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def measure(run, n=1500, seed=0, state=None):
    """`state`: a generation checkpoint to evaluate instead of the final pair."""
    s, l, w, c = load(run)
    if state is not None:
        cp = torch.load(state, map_location='cpu', weights_only=False); s.load_state_dict(cp['speaker']); l.load_state_dict(cp['listener'])
    inf = Informant(s, l, w, c); rng = np.random.default_rng(seed)
    train = np.nonzero(w.partition == 0)[0]; res = {}; tokens = Counter()
    for k in (1, 2, 3):
        items = contexts_for(w, rng, train, k, n); said = inf.say([x for x, _ in items])
        for m in said: tokens.update(m)
        groups = defaultdict(list)
        for (ctx, slots), m in zip(items, said):
            cur = w.slots(ctx[0]); groups[(int(w.events[ctx[0], 0]), slots)].append((tuple(int(cur[x]) for x in slots), m))
        if k > 1:
            md, sd = [], []
            for g in groups.values():
                for i in range(len(g)):
                    for j in range(i + 1, min(len(g), i + 40)):
                        md.append(sum(x != y for x, y in zip(g[i][0], g[j][0]))); sd.append(edit(g[i][1], g[j][1]))
            res[f'topsim{k}'] = spearman(md, sd)
        if k == 2:
            hit = tot = 0
            for g in groups.values():
                rng.shuffle(g); half = len(g) // 2; best = None
                for a, b in ((0, 1), (1, 0)):       # which value the first / last symbol follows
                    first, last = defaultdict(Counter), defaultdict(Counter)
                    for v, m in g[:half]:
                        if m: first[v[a]][m[0]] += 1; last[v[b]][m[-1]] += 1
                    ok = sum(bool(m) and len(m) <= 2 and v[a] in first and v[b] in last and first[v[a]].most_common(1)[0][0] == m[0]
                             and last[v[b]].most_common(1)[0][0] == m[-1] for v, m in g[half:])
                    best = ok if best is None else max(best, ok)
                hit += best; tot += len(g) - half
            res['part2'] = hit / max(1, tot)
    total = sum(tokens.values()); acc = 0; used = 0
    for _, n_ in tokens.most_common():
        acc += n_; used += 1
        if acc >= 0.95 * total: break
    res['used'] = used; res['acc'] = accuracy(inf, 'train', 256, seed)
    return res


def lineage(run):
    """Structure of every generation's language, founder first."""
    gens = sorted(Path(run).glob('generation*.pt'), key=lambda p: int(p.stem[10:]))
    for p in gens + [None]:
        print(f"  {(p.stem if p else 'final'):14s}", json.dumps({k: round(v, 2) for k, v in measure(run, 800, state=p).items()}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--lineage':
        for run in sys.argv[2:]: print(Path(run).name); lineage(run)
        sys.exit()
    for run in sys.argv[1:]:
        print(Path(run).name.ljust(22), json.dumps({k: round(v, 2) for k, v in measure(run).items()}))
