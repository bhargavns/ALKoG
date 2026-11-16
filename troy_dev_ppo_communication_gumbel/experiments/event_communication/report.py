"""Generate auditable Markdown, CSV and figures from completed study artifacts."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(rows, field):
    values = [x[field] for x in rows if x[field] is not None]
    if not values:
        return 'n/a'
    return f'{statistics.mean(values):.3f} +/- {statistics.stdev(values) if len(values)>1 else 0:.3f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    result_file = a.study/('fresh_results.json' if (a.study/'fresh_results.json').exists() else 'results.json')
    results = json.loads(result_file.read_text())
    manifest = json.loads((a.study/'manifest.json').read_text())
    flat = [dict(arm=x['arm'], seed=x['seed'], seconds=x['seconds'], **e)
            for x in results for e in x['evaluation']]
    for row in flat:
        cost=0. if row['arm'] in ('free','oracle') else .2 if row['arm']=='expensive' else .05
        row['task_return']=row['return_mean']+cost*row['messages']
        row['common_cost_return']=row['task_return']-.05*row['messages']
    with (a.output/'all-evaluations.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    arms = list(dict.fromkeys(x['arm'] for x in results))
    normal = [x for x in flat if x['scenario']=='dynamic' and x['intervention']=='normal']
    lines = ['# Optional communication in ALKoG: experimental report', '',
             '## Scope and main result', '',
             'This is an executed, controlled oracle kinematic-grid study. It tests optional '
             'transmission, persistent safety belief, and independent navigation. It does not '
             'establish visual grounding, physical safety, language-like semantics, or private-location '
             'communication. Read every safety result as a finite held-out measurement.', '',
             f"Completed **{len(results)} runs**, {len(arms)} methods, seeds {manifest['seeds']}. "
             f"Each run uses {manifest['iterations']} updates x 24 steps x {manifest['batch_size']} lanes "
             f"= **{manifest['iterations']*24*manifest['batch_size']:,} allocated lane-steps** "
             f"across {manifest['iterations']*manifest['batch_size']:,} generated episodes. "
             'Lane-steps include absorbing padding after success/death; they are not a count of '
             'active environment interactions. '
             f"There are {manifest['evaluation_episodes']:,} evaluation episodes per condition.", '',
             'All aggregate spreads below are sample standard deviations across training seeds. '
             'They are descriptive, not significance tests. Scores use final checkpoints; no '
             'held-out checkpoint selection was performed.', '',
             '## Normal dynamic evaluation', '',
             '| Method | Success | Hazard failures | Timeouts | Messages/episode | Return at common .05 cost |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        rows = [x for x in normal if x['arm']==arm]
        lines.append('| '+arm+' | '+' | '.join(summarize(rows,k) for k in
                     ('success','hazard','timeout','messages','common_cost_return'))+' |')
    lines += ['', '![Success and message volume](success-messages.png)', '',
              '## Methods and reproducibility', '',
              'The complete method specification is reproduced in [methods.md](methods.md). '
              'It includes information boundaries, rewards, warm-up supervision, all 14 arms, '
              'PPO details, intervention definitions, and limitations.', '',
              f"Runtime: Python {manifest['python']}; PyTorch {manifest['torch']}; GPU {manifest['gpu']}. "
              f"Base commit `{manifest['commit']}`; training source SHA-256 "
              f"`{manifest['source_sha256']}`.", '',
              'Spatial warm-up uses BFS behavior cloning. Communication warm-up maximizes '
              'enumerated safety-response reward, including retention over silence. The cold arm '
              'removes communication warm-up but retains spatial skill. The safety belief bottleneck '
              'is prescribed; public token meanings are learned. This is a modular learning study, '
              'not evidence that all semantic structure emerged from navigation reward alone.', '',
              'Training scenes use matched per-iteration seeds across methods. Each final model '
              'has an optimizer/RNG checkpoint, complete learning curve, 30 evaluation cells and '
              '32 paired trajectory examples. Exact source and warm-up weights are saved alongside '
              'results. Initial diagnostic scores use separate seed 810001.', '',
              '## Development attempts', '',
              '1. Initial 400-update, uniformly sampled spatial warm-up: the oracle receiver '
              'collided in roughly 24% of dynamic development episodes. The movement network '
              'mostly learned forward travel and ignored rare avoidance situations. This was '
              'a navigation failure, so it could not support a communication conclusion.', '',
              '2. Balanced action classes and emphasized hazard-adjacent cells, with 1,500 '
              'warm-up updates: a first oracle check avoided collisions but still timed out '
              'in 10.4% of dynamic episodes. Rotation transfer failed. This motivated a larger '
              'movement network, not tuning the held-out study outcomes.', '',
              '3. Final preflight, separate development seed 123: a two-hidden-layer, '
              '128-wide movement network and 25 PPO updates reached 100% oracle and fact-cache '
              'success and 98.6% GRU/continuous success on the development dynamic evaluation. '
              'The study then fixed architecture and budgets before evaluating seeds 0-2.', '',
              'These attempts have separate source snapshots and logs in the repository output '
              'directory. They are not additional independent replicates of the final study.', '',
              '## Causal message interventions', '',
              'The table reports normal success minus intervened success; positive numbers '
              'indicate harm from the intervention. Attempts remain charged even if delivery is '
              'blocked. Constant payloads preserve timing, so they do not remove all information.', '',
              '| Method | Delete updates | Constant token | Shuffle | 1-step delay | 25% loss | Delete unchanged updates |',
              '|---|---:|---:|---:|---:|---:|---:|']
    interventions = ['suppress_updates','constant','shuffle','delay','drop','suppress_redundant']
    for arm in arms:
        cells=[]
        for iv in interventions:
            delta=[]
            for r in results:
                if r['arm']!=arm: continue
                es = {e['intervention']:e for e in r['evaluation'] if e['scenario']=='dynamic'}
                delta.append(dict(delta=es['normal']['success']-es[iv]['success']))
            cells.append(summarize(delta,'delta'))
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    lines += ['', '## Transmission-rate controls', '',
              'These interventions replace timing in the same trained pair. They target the '
              'normal pair\'s observed transmission rate per active step. Actual episode message '
              'counts can differ because routes, termination times and message histories change. '
              'This is an expected-rate control, not exact per-episode budget matching.', '',
              '| Method | Normal success / messages | Rate-periodic success / messages | Rate-random success / messages |',
              '|---|---:|---:|---:|']
    for arm in arms:
        cells=[]
        for iv in ('normal','matched_periodic','matched_random'):
            rows=[x for x in flat if x['arm']==arm and x['scenario']=='dynamic' and x['intervention']==iv]
            cells.append(summarize(rows,'success')+' / '+summarize(rows,'messages'))
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    lines += ['', '## Memory and event selectivity', '',
              'Graph denotes a one-fact cache with learned decoding, not a complete knowledge '
              'graph. Compare its retention with GRU and memoryless decoding. Event response '
              'is measured at the first observation after a flip; relevant/irrelevant geometry '
              'is the initial corridor relation, not a counterfactual proof of route relevance.', '',
              '| Method | Belief accuracy | Send after event | Send without event | Relevant success | Off-corridor success |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        rows=[x for x in normal if x['arm']==arm]
        lines.append('| '+arm+' | '+' | '.join(summarize(rows,k) for k in
                     ('belief_accuracy','event_send_rate','no_event_send_rate','success_relevant','success_irrelevant'))+' |')
    lines += ['', '## Transfer and long silence', '',
              'Static scenes test whether status can persist without updates. Rotated scenes '
              'rotate coordinates and movement together, while goal direction is outside the '
              'navigation warm-up distribution. Oracle failure on rotation identifies a spatial '
              'generalization limitation. Late scenes delay flips by four steps.', '',
              '| Method | Static success | Static after deleting updates | Rotated success | Late-flip success |',
              '|---|---:|---:|---:|---:|']
    for arm in arms:
        cells=[]
        for scenario,iv in [('static','normal'),('static','suppress_updates'),('rotated','normal'),('late','normal')]:
            rows=[x for x in flat if x['arm']==arm and x['scenario']==scenario and x['intervention']==iv]
            cells.append(summarize(rows,'success'))
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    lines += ['', '![Learning curves](learning-curves.png)', '', '## Per-seed normal results', '',
              '| Method | Seed | Success | Hazard | Messages | Net return | Training + evaluation seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for x in normal:
        lines.append(f"| {x['arm']} | {x['seed']} | {x['success']:.4f} | {x['hazard']:.4f} | "
                     f"{x['messages']:.3f} | {x['return_mean']:.3f} | {x['seconds']:.1f} |")
    lines += ['', '## Interpretation limits', '',
              '- Deterministic evaluation can differ from stochastic training; a gate whose '
              'probability barely exceeds .5 may transmit at every evaluation step.',
              '- Reward permits safe conservative detours without communication. Silence '
              'can be rational; communication usefulness depends on efficiency as well as success.',
              '- The sender sees its own true observations and transmission history; the '
              'receiver never sees true private status except in the explicit oracle condition.',
              '- One binary safety fact and four possible tokens do not test compositionality, '
              'reference, private locations, multiple relations, or general graph learning.',
              '- The counterfactual gate uses fitted unchosen Q estimates, not experimentally '
              'observed alternative rewards. Its success would not validate exact causal value estimation.',
              '- All grid motion is kinematic. This study does not train in MuJoCo, transfer '
              'visual perception, or establish physical safety. Existing MuJoCo baselines were regression-tested.',
              '- Three seeds and a single training budget characterize this implementation; '
              'they do not establish asymptotic superiority or a general impossibility result.', '',
              'Raw data: [all evaluation cells](all-evaluations.csv), '
              '[manifest](manifest.json), [environment lock](requirements-lock.txt).', '']
    (a.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    if (a.study/'final_evaluation_manifest.json').exists():
        final_manifest=json.loads((a.study/'final_evaluation_manifest.json').read_text())
        note=('\nThe original scene seed 900001 overlapped the development checks. Those '
              'scores are retained as development evidence only. All primary tables and figures '
              'in this report use fresh scene seed 1900001, evaluated on CPU after all model '
              'designs and weights were frozen, with 1,024 episodes per condition. No tuning '
              'followed this final evaluation.\n\n')
        report=(a.output/'report.md').read_text(encoding='utf-8')
        report=report.replace('## Normal dynamic evaluation',note+'## Normal dynamic evaluation',1)
        (a.output/'report.md').write_text(report,encoding='utf-8')
        (a.output/'final-evaluation-manifest.json').write_text(json.dumps(final_manifest,indent=2))
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    root=Path(__file__).parent
    (a.output/'methods.md').write_text((root/'README.md').read_text())
    (a.output/'requirements-lock.txt').write_text((root/'requirements-lock.txt').read_text())
    fig,ax=plt.subplots(figsize=(10,6))
    for i,arm in enumerate(arms):
        rows=[x for x in normal if x['arm']==arm]
        ax.scatter([x['messages'] for x in rows], [100*x['success'] for x in rows], label=arm,
                   color=plt.cm.tab20(i),marker=('o','s','^','v','D','P','X')[i%7])
    ax.set(xlabel='Attempted transmissions per episode', ylabel='Goal success (%)', ylim=(-2,102))
    ax.legend(ncol=2, fontsize=8, bbox_to_anchor=(1.01,1)); fig.tight_layout()
    fig.savefig(a.output/'success-messages.png', dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
    for i,arm in enumerate(arms):
        curves=[json.loads((a.study/f"{arm}_{r['seed']}"/'curve.json').read_text())
                for r in results if r['arm']==arm]
        for ax,k in zip(axes,('success','messages')):
            vals=np.array([[row[k] for row in curve] for curve in curves])
            ax.plot(np.arange(vals.shape[1])+1, vals.mean(0), label=arm, alpha=.8,color=plt.cm.tab20(i))
    axes[0].set_ylabel('Stochastic training success'); axes[1].set_ylabel('Messages/episode')
    axes[1].set_xlabel('PPO update'); axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(a.output/'learning-curves.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
