"""Run the pressure conditions over several seeds and tabulate the analysis.

    python -m experiments.grammar_pressures.sweep --output results/grammar_pressures/sweep --seeds 0 1 2 --workers 6

Every run is kept; nothing is filtered or retried.
"""
from pathlib import Path
import argparse, concurrent.futures, json, os, subprocess, sys
import numpy as np

# name -> extra command-line arguments for train.py
CONDITIONS = {
    # the speaker cannot see the rooms; the collector faces 16 confusable rooms
    'blind_calm':        '--blind --cands 16 --sets 16 --focus 6 --max-len 10',
    'blind_urgent':      '--blind --cands 16 --sets 16 --focus 6 --max-len 10 --cost 0.003 --hazard 0.03',
    # the speaker sees the four rooms it is talking about
    'context_calm':      '',
    'context_urgent':    '--cost 0.01 --hazard 0.03',
    'context_emergency': '--cost 0.01 --hazard 0.3 --curriculum',
    # follow-up 1: is the partial lexicon of context_emergency_2 repeatable?  (10 seeds, 20,000 updates)
    'lead_h10': '--cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9',
    'lead_h20': '--cost 0.01 --hazard 0.2 --curriculum --entropy 0.1 --entropy-until 0.9',
    'lead_h30': '--cost 0.01 --hazard 0.3 --curriculum --entropy 0.1 --entropy-until 0.9',
    # follow-up 2: near and wild misses, misheard symbols, payoff graded by similarity (and its three controls)
    'mixed_plain':        '--cands 8 --far 3 --sets 32 --cost 0.005 --hazard 0.03',
    'mixed_graded':       '--cands 8 --far 3 --sets 32 --cost 0.005 --hazard 0.03 --graded',
    'mixed_noise':        '--cands 8 --far 3 --sets 32 --cost 0.005 --hazard 0.03 --noise 0.1',
    'mixed_graded_noise': '--cands 8 --far 3 --sets 32 --cost 0.005 --hazard 0.03 --graded --noise 0.1',
    # follow-up 3: is the 6-signal inventory what stops a lexicon?  lead_h10 with a larger inventory
    'inv12': '--vocab 12 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9',
    'inv24': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9',
    'inv40': '--vocab 40 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9',
    # follow-up 4: further pressures on top of a 24-signal inventory
    'disp24': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --displaced',
    'life24': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --listener-life 1000',
    'pop24':  '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --pop 3',
    'c8v24':  '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --cands 8 --sets 32',
    # follow-up 5: rooms are only glimpsed (partial observability on the collector's side)
    'key24':      '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --keyhole 0',
    'keydisp24':  '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --keyhole 0 --displaced',
    'keydisp24x': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --keyhole 0.3 --displaced',
    'keydisp24c': '--vocab 24 --keyhole 0 --displaced',
    # follow-up 6: variations around glimpsed rooms + displaced speaker (no urgency unless stated)
    'k_base':   '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.1 --entropy-until 0.9 --curriculum',
    'k_nocur':  '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.1 --entropy-until 0.9',
    'k_c8':     '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.1 --entropy-until 0.9 --curriculum --cands 8 --sets 32',
    'k_h03':    '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.1 --entropy-until 0.9 --curriculum --hazard 0.03',
    'k_ent':    '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.3 --entropy-until 0.9 --curriculum',
    'k_life':   '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.1 --entropy-until 0.9 --curriculum --listener-life 2000',
    'k_pop':    '--vocab 24 --keyhole 0 --displaced --cost 0.01 --entropy 0.1 --entropy-until 0.9 --curriculum --pop 2',
    # follow-up 7: iterated learning (speaker and collector replaced each generation; the new speaker
    # first imitates a limited sample of the old one's utterances, in context).  tight vs loose bottleneck
    'il_k_tight': '--vocab 24 --keyhole 0 --displaced --cost 0.01 --hazard 0.03 --curriculum --entropy 0.1 --entropy-until 0.9 --generations 10 --life 1500 --imitation 600 --bottleneck 200',
    'il_k_loose': '--vocab 24 --keyhole 0 --displaced --cost 0.01 --hazard 0.03 --curriculum --entropy 0.1 --entropy-until 0.9 --generations 10 --life 1500 --imitation 600 --bottleneck 3000',
    'il_l_tight': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --generations 10 --life 1500 --imitation 600 --bottleneck 200',
    'il_l_loose': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --generations 10 --life 1500 --imitation 600 --bottleneck 3000',
    # follow-up 8: sound before meaning.  3,000 updates of babbling (diverse but predictable song,
    # no reference) before the task.  Controls without babbling: k_h03 (kgrid, 20,000) and inv24 (12,000)
    'bab_k': '--vocab 24 --keyhole 0 --displaced --cost 0.01 --hazard 0.03 --curriculum --entropy 0.1 --entropy-until 0.9 --babble 3000',
    'bab_l': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --babble 3000',
    'bab_l_nov': '--vocab 24 --cost 0.01 --hazard 0.1 --curriculum --entropy 0.1 --entropy-until 0.9 --babble 3000 --novelty 1.0',
    # follow-up 9: wanting vs reporting (experiments/grammar_pressures/perspective.py)
    'persp24': 'perspective --vocab 24',
    'persp12': 'perspective --vocab 12',
}
FIRST = ('blind_calm', 'blind_urgent', 'context_calm', 'context_urgent', 'context_emergency')


def one(root, name, seed, updates):
    out = root / f'{name}_{seed}'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with open(root / f'{name}_{seed}.log', 'w') as log:
        args = CONDITIONS[name].split()
        module = args.pop(0) if args and not args[0].startswith('--') else 'train'
        base = [sys.executable, '-m', f'experiments.grammar_pressures.{module}', '--output', str(out), '--seed', str(seed), '--updates', str(updates)]
        subprocess.run(base + args, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
        if module == 'train':       # other games analyse themselves
            subprocess.run([sys.executable, '-m', 'experiments.grammar_pressures.analysis', str(out)], stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
    return name, seed


def row(r):
    """The handful of numbers the README table reports for one run."""
    from .world import TEST_PARTITIONS
    a, lx, cb, bv = r['accuracy'], r['lexicon'], r['combination'], r['brevity']['train']
    m = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else float('nan')
    ent = r.get('entanglement') or {}
    return dict(train=a['train']['normal'], heldout=m([a[p]['normal'] for p in TEST_PARTITIONS]), shuffled=a['train']['shuffled'],
                len_k1=bv['1']['length'], len_k3=bv['3']['length'], word_consistency=lx['word_consistency'],
                word_length=lx['mean_word_length'], values_distinguished=lx['values_distinguished'],
                concatenative=cb['train']['concatenative'],
                synthesis_understood=m([cb[p].get('synthesis_understood') for p in TEST_PARTITIONS]),
                position_purity=ent.get('position_purity', float('nan')), newcomer_150=r['newcomer']['150']['train'])


def summarise(root):
    root = Path(root); table = {}
    for name in CONDITIONS:
        if CONDITIONS[name].startswith('perspective'): continue
        rows = [row(json.loads(p.read_text())) for p in sorted(root.glob(f'{name}_[0-9]*/analysis.json'))]
        if rows: table[name] = dict(seeds=len(rows), **{k: [float(np.mean([x[k] for x in rows])), float(np.min([x[k] for x in rows])), float(np.max([x[k] for x in rows]))] for k in rows[0]})
    (root / 'summary.json').write_text(json.dumps(table, indent=1))
    keys = list(next(iter(table.values())).keys())[1:] if table else []
    lines = ['| condition | seeds | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 2)]
    for name, t in table.items():
        lines.append(f"| {name} | {t['seeds']} | " + ' | '.join(f"{t[k][0]:.2f} ({t[k][1]:.2f}-{t[k][2]:.2f})" for k in keys) + ' |')
    (root / 'summary.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--conditions', nargs='+', default=list(FIRST))
    p.add_argument('--summarise-only', action='store_true'); a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.summarise_only:
        jobs = [(n, s) for s in a.seeds for n in a.conditions if not (a.output / f'{n}_{s}' / 'analysis.json').exists()]
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            for f in concurrent.futures.as_completed([ex.submit(one, a.output, n, s, a.updates) for n, s in jobs]):
                try: print('done', *f.result(), flush=True)
                except Exception as e: print('FAILED', repr(e), flush=True)
    summarise(a.output)
