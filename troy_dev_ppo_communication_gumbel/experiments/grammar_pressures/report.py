"""Per-run table of the grammar measures for one or more result folders.

    python -m experiments.grammar_pressures.report results/grammar_pressures/inventory [name-prefix ...]
"""
from pathlib import Path
import json, sys
import numpy as np
from .world import TEST_PARTITIONS

COLS = ('train', 'held', 'len2', 'cons', 'c_typ', 'c_col', 'c_prd', 'dist', 'AP', 'SA', 'forms', 'overlap', 'concat', 'bag', 'order',
        'h_concat', 'synth', 'own', 'swap', 'rev', 'newc')


def m(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float('nan')


def row(r):
    a, lx, cb, ro = r['accuracy'], r['lexicon'], r['combination'], r['roles']['train'] or {}
    k = lx['word_consistency_by_kind']; d = lx['distinct_forms']; o = lx['class_overlap']
    h = lambda key: m([cb[p].get(key) for p in TEST_PARTITIONS])
    return dict(train=a['train']['normal'], held=m([a[p]['normal'] for p in TEST_PARTITIONS]), len2=cb['train']['mean_length'],
                cons=lx['word_consistency'], c_typ=k.get('type'), c_col=k.get('colour'), c_prd=k.get('pred'), dist=lx['values_distinguished'],
                AP=lx['type_word_shared_A_P'], SA=lx['type_word_shared_S_A'], forms=sum(d.values()), overlap=sum(o.values()),
                concat=cb['train']['concatenative'], bag=cb['train']['same_bag'], order=cb['train']['order_consistency'],
                h_concat=h('concatenative'), synth=h('synthesis_understood'), own=h('speaker_understood'),
                swap=ro.get('accuracy'), rev=ro.get('reversed_string_picks_reversed_roles'), newc=(r.get('newcomer') or {}).get('150', {}).get('train'))


def fmt(x): return '   - ' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:5.2f}' if isinstance(x, float) else f'{x:5d}'


def main(root, prefixes=()):
    root = Path(root); groups = {}
    for p in sorted(root.glob('*/analysis.json')):
        name = p.parent.name.rsplit('_', 1)[0]
        if prefixes and not any(name.startswith(x) for x in prefixes): continue
        groups.setdefault(name, []).append((p.parent.name, row(json.loads(p.read_text()))))
    print(f"{'run':26s}" + ' '.join(f'{c:>7s}' for c in COLS))
    for name, rows in groups.items():
        for run, r in rows: print(f'{run:26s}' + ' '.join(f'{fmt(r[c]):>7s}' for c in COLS))
        print(f"{name + ' MEAN':26s}" + ' '.join(f'{fmt(m([r[c] for _, r in rows])):>7s}' for c in COLS)); print()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
