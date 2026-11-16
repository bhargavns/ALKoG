"""Append follow-up experiments and package a standalone HTML report."""
import argparse
import base64
import csv
import json
import statistics
from pathlib import Path
import markdown
from .report import summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.study.parent
    follow=[]
    for seed in (0,1,2):
        follow += json.loads((root/f'event_control_seed{seed}'/'fresh_results.json').read_text())
        for tag in ('value','class'):
            follow.append(json.loads((root/f'event_paired_{tag}_final{seed}'/'fresh_results.json').read_text()))
    flat=[dict(method=r['arm'],seed=r['seed'],**e) for r in follow for e in r['evaluation']]
    with (a.output/'follow-up-evaluations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    normal=[x for x in flat if x['scenario']=='dynamic' and x['intervention']=='normal']
    methods=list(dict.fromkeys(x['method'] for x in normal))
    text=['## Follow-up: sparse achievability and learned counterfactual timing','',
          'These follow-ups respond to the initial study\'s frequent-transmission/silence outcomes. '
          'They are separate experiments, not extra equal-budget arms. There are three seeds for '
          'each method. The event control is prescribed; the other two gates are learned.','',
          '| Follow-up | Success | Hazard | Messages/episode | Net return |',
          '|---|---:|---:|---:|---:|']
    for method in methods:
        rs=[x for x in normal if x['method']==method]
        text.append('| '+method+' | '+' | '.join(summarize(rs,k) for k in
                    ('success','hazard','messages','return_mean'))+' |')
    text += ['', '**Prescribed event control.** Start from the same fact-cache warm-up, then '
             'train the learned message code and navigator for the same 75 PPO updates. The gate '
             'sends at reset and when changed status threatens the initial route before passing '
             'the lion. This timing rule uses task knowledge and is an achievability control, '
             'not evidence of emergent timing. Its learned code/controller warm-start the next two methods.','',
             '**Paired return regression.** Freeze that partner and code. Generate 80 batches '
             'of 256 initial scenes. At decision times cycling from 0 through 9, random sparse '
             'prefix communication creates a range of memory states. Fork each active state '
             'into SEND and SILENT, then allow no further transmissions in either continuation '
             'until the original 24-step deadline. Deterministic navigation and identical '
             'starting states isolate the one-packet return difference, including the .05 cost. '
             'Fit a local sender-only 64-wide gate to these differences with 2,000 Adam '
             'updates, batch size 512, learning rate .001. Rare positive states are oversampled. '
             'Send when predicted difference exceeds zero. This assumes silent continuation '
             'during training, so it is not an optimal multi-packet communication value.','',
             '**Paired usefulness classification.** Use the same fork data and frozen partner, '
             'but predict whether the packet has positive return difference using unweighted '
             'binary cross entropy and uniform state sampling. Same architecture, batch size '
             'and 2,000-update budget. A .5 threshold selects sends. Unlike regression, it '
             'discards the magnitude of benefit and cost. This can favor very sparse '
             'transmission while missing valuable packets.','',
             'Both learned gates use only sender observations and its own transmission history '
             'at deployment. Ground-truth alternative outcomes are available during training '
             'through simulation forks. This is extra supervision and computation; it is '
             'not a sample-efficiency victory over the initial PPO methods. CPU execution '
             'was used for these small follow-ups; the main 42-run study used CUDA.','',
             '| Method | Seed | Labeled active states | Positive states | Extra active fork transitions |',
             '|---|---:|---:|---:|---:|']
    for r in follow:
        if 'states' in r:
            text.append(f"| {r['arm']} | {r['seed']} | {r['states']} | {r['positive_states']} | {r['extra_active_query_steps']} |")
    text += ['', 'Regression and classification reuse identical generated-state/fork data per '
             'seed, but the scripts execute the fork computation separately. The transition '
             'counts above count active simulator steps before termination, including the '
             'terminal transition; absorbing padding is excluded.','',
             '| Method | Delete updates: success | Constant payload: success | Matched periodic: success / messages | Matched random: success / messages |',
             '|---|---:|---:|---:|---:|']
    for method in methods:
        cells=[]
        for iv in ('suppress_updates','constant','matched_periodic','matched_random'):
            rs=[x for x in flat if x['method']==method and x['scenario']=='dynamic' and x['intervention']==iv]
            cells.append(summarize(rs,'success')+(' / '+summarize(rs,'messages') if iv.startswith('matched') else ''))
        text.append('| '+method+' | '+' | '.join(cells)+' |')
    text += ['', 'Constant-payload robustness can mean that timing conveys the warning. '
             'It does not establish that no communication is occurring. Expected-rate '
             'replacement schedules have different realized message counts and histories; '
             'these interventions isolate a timing effect under the frozen pair, not an '
             'optimal independently retrained alternative.','',
             '## Findings and recommendation','',
             'The experiments support a distinction between learning a useful communication '
             'channel and learning its opportunity cost. Standard joint PPO can obtain high '
             'task success while transmitting almost continuously, and a higher message cost '
             'can instead favor silence. Memory alone does not guarantee selective communication.','',
             'The prescribed sparse control establishes that occasional updates are sufficient '
             'for this restricted task. The paired experiments provide a direct learning '
             'mechanism: estimate how much a packet changes the partner\'s resulting return, '
             'then learn a sender-only approximation to that value. Read their success, hazard '
             'and message counts together: minimizing messages and preserving success are '
             'different objectives, and usefulness classification throws away the magnitude '
             'of a missed warning.','',
             'For the next ALKoG iteration, use recurrent or persistent belief memory, independent '
             'navigation, and a gate trained from packet-value comparisons. Prefer return '
             'magnitude over a bare useful/not-useful label when missed updates have unequal '
             'consequences. Periodically refresh comparisons as the partner changes. A learned '
             'critic can amortize the expensive forks, but its unchosen-action estimates need '
             'validation against actual paired outcomes.','',
             'The present data do not justify claims about the full knowledge graph, visual '
             'perception, private location descriptions or physical navigation. Those remain '
             'explicitly unexecuted extensions. This report completes the described oracle '
             'mechanism study and its follow-ups; it is not a claim that every broader research '
             'direction in the preceding theoretical discussion has been implemented.','',
             '## Validation and artifacts','',
             'The existing standalone Gumbel pipeline suite plus the new mechanism tests '
             'passed: 133 tests. Tests cover observation isolation, private status flips, '
             'absorbing terminals, exact recurrent PPO replay, sender gradients, retention '
             'during silence, contextual action selection, and terminal advantage masking. '
             'The test suite uses controlled perception inputs; it does not download or '
             'validate SAM/ResNet visual checkpoints. Additional fork tests check identical '
             'branches under a partner that ignores communication and isolate transmission cost.','',
             'The artifact bundle contains new source, dependency lock, all final checkpoints, '
             'warm-up states, curves, evaluation cells, selected paired trajectories, fork '
             'datasets, development logs and source snapshots. Run configuration and source '
             'hashes accompany the main study. No remote branches were pushed.','',
             '## Relation to prior work','',
             '- [IC3Net](https://arxiv.org/abs/1812.09755) provides a precedent for learned '
             'communication gates.','- [ETCNet](https://arxiv.org/abs/2010.04978) studies '
             'event-triggered communication under bandwidth constraints.','- '
             '[MACC](https://arxiv.org/abs/2006.07200) addresses communication credit assignment '
             'with counterfactual reasoning. The fitted-Q and explicit fork methods in this '
             'report are simplified experimental designs, not reproductions of MACC.','']
    base=(a.output/'report.md').read_text(encoding='utf-8')
    main_results=json.loads((a.study/'results.json').read_text())
    summary=['## Executive findings','',
             'The main study completed 42 runs (14 methods x 3 seeds), followed by 9 runs '
             'covering prescribed sparse timing and two learned counterfactual gates.','']
    for method in methods:
        rs=[x for x in normal if x['method']==method]
        summary.append(f"- **{method}**: success {summarize(rs,'success')}, messages/episode "
                       f"{summarize(rs,'messages')}, hazard rate {summarize(rs,'hazard')}.")
    summary += ['', 'The key result is that counterfactual packet value can train selective '
                'communication in this limited task, with a measurable success/message '
                'tradeoff. Standard PPO and a message penalty did not reliably recover the '
                'desired sparse protocol at the tested budget. These are oracle-state '
                'findings; visual and private-location experiments remain future work.','']
    silent=next(r for r in main_results if r['arm']=='cold')
    summary += ['**Critical negative result:** the cold communication-start arm learned silent '
                'conservative navigation and achieved 100% success across all three seeds, with '
                'zero evaluation transmissions. Its common-cost return (4.895) exceeds both '
                'learned paired-gate variants. Thus this task does **not** establish an overall '
                'communication advantage over the best trained silent policy. The paired '
                'interventions establish a benefit of timing for a particular frozen partner, '
                'which is a narrower claim.','',
                'The next task should strengthen the information asymmetry or route-efficiency '
                'tradeoff, and evaluate against the strongest conservative silent policy. '
                'Private goal/location information is one option. This does not overturn '
                'communication benefits established in a different experimental branch.','']
    report=base.replace('## Scope and main result','\n'.join(summary)+'\n## Scope and main result',1)+'\n'+'\n'.join(text)
    (a.output/'report.md').write_text(report,encoding='utf-8')
    body=markdown.markdown(report,extensions=['tables','fenced_code'])
    for image in ('success-messages.png','learning-curves.png'):
        data=base64.b64encode((a.output/image).read_bytes()).decode()
        body=body.replace('src="'+image+'"','src="data:image/png;base64,'+data+'"')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>ALKoG communication experiments</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#182333;background:#fafbfd}h1,h2{line-height:1.2;color:#124567;margin-top:2em}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;background:white;margin:20px 0}td,th{padding:9px 12px;border:1px solid #d7e0e8;text-align:left;white-space:nowrap}th{background:#eaf1f7}img{max-width:100%;height:auto}code,pre{background:#eef1f4}a{color:#136a9e}li{margin:.4em 0}@media print{body{font-size:10pt}h2{break-after:avoid}table{font-size:7pt;display:table}td,th{white-space:normal;padding:4px}}</style><body>'''+body+'</body></html>'
    (a.output/'report.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    main()
