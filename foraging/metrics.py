"""Measures of message structure shared by the studies."""
import numpy as np


def spearman(x, y):
    """Rank correlation with averaged ranks for ties; nan if either side is constant."""
    def rank(v):
        v = np.asarray(v, float); order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
        for u in np.unique(v): r[v == u] = r[v == u].mean()
        return r
    if len(set(x)) < 2 or len(set(y)) < 2: return float('nan')
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def edit(a, b):
    """Levenshtein distance between two symbol sequences."""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1): prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]
