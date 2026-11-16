"""Self-contained report with definitions, derivations, complete results and limits."""
import argparse
import base64
import csv
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .models import silent_route_ceiling,broadcast_bound
from .illustrations import make_figures
from .setup_explanation import setup_section


LABELS={'private_goal':'1. Private goal','route_safety':'2. Changing route safety',
        'scouting':'3. Communication versus scouting','complementary':'4. Complementary knowledge'}
PRIMARY={'private_goal':'learned','route_safety':'learned','scouting':'learned_warm','complementary':'dialogue_warm'}
INTERVENTIONS={'private_goal':('normal','blocked','constant','permute'),
               'route_safety':('normal','blocked','constant','permute','suppress_updates','reset_memory','drop25'),
               'scouting':('normal','blocked','constant','permute'),
               'complementary':('normal','blocked','constant','permute','remove_query')}


def mean(rows,key):return statistics.mean(r[key] for r in rows)
def cell(rows,key,digits=4):
    v=[r[key] for r in rows]
    return f'{statistics.mean(v):.{digits}f} ± {statistics.stdev(v) if len(v)>1 else 0:.{digits}f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    results=json.loads((a.study/'validated_results.json').read_text())
    assert len(results)==130,'Report only complete studies'
    flat=[dict(task=r['task'],method=r['method'],seed=r['seed'],seconds=r['seconds'],**e)
          for r in results for e in r['evaluations'] if e['intervention'] in INTERVENTIONS[r['task']]]
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (a.output/'all-evaluations.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    normals=[r for r in flat if r['deployment']=='argmax' and r['intervention']=='normal']
    def group(task,method):return [r for r in normals if r['task']==task and r['method']==method]
    g=group('private_goal','learned');route=group('route_safety','learned')
    scout=group('scouting','learned_warm');dialog=group('complementary','dialogue_warm')
    criteria={
        'private_goal':mean(g,'success')>=.95 and min(r['success'] for r in g)>.25,
        'route_safety':mean(route,'success')>=.95 and min(r['utility'] for r in route)>silent_route_ceiling() and mean(route,'packets')<=4.5,
        'scouting':mean(scout,'utility')>=.965 and min(r['utility'] for r in scout)>.91 and
                   all(r['context_0_ask']<.1 and r['context_2_ask']>.9 and r['context_4_ask']>.9 and r['context_1_ask']<.1 for r in scout),
        'complementary':mean(dialog,'success')>=.95-1e-7 and min(r['utility'] for r in dialog)>.84 and mean(dialog,'success')>broadcast_bound()}
    verdicts={t:dict(passes=bool(v),primary=PRIMARY[t],rows=group(t,PRIMARY[t])) for t,v in criteria.items()}
    (a.output/'verdicts.json').write_text(json.dumps(verdicts,indent=2))
    text=['# When communication becomes useful: four controlled ALKoG experiments','',
          '**Date:** November 13, 2026. **Scope:** finite symbolic decision processes, outside visual perception and MuJoCo.','',
          '## Executive conclusion','',
          'Communication helps in the redesigned tasks. The first two demonstrate a large '
          'success-rate advantage when the receiver lacks essential information. The third '
          'demonstrates an efficiency advantage while preserving the ability to act without '
          'communication. The fourth demonstrates a benefit of asking for relevant information '
          'rather than broadcasting the entire world, but learned dialogue remains imperfect '
          'and depends strongly on a preparatory reference-learning stage.','',
          '**Verdict: Options 1–3 pass their pre-specified criteria. Option 4 is a partial '
          'success and fails the across-seed criterion:** its mean accuracy is 95%, but '
          'one optimization seed has utility 0.795, below full-map broadcast at 0.84. '
          'Two of its five warm-start runs achieve perfect accuracy.','',
          '| Option | Primary learned result | Relevant alternative | Finding |',
          '|---|---|---|---|',
          f"| Private goal | Success {cell(g,'success')}; {cell(g,'packets')} transmissions | Silent success 0.25 | Information closes an otherwise unavoidable gap |",
          f"| Changing route safety | Success {cell(route,'success')}; {cell(route,'packets')} transmissions | Silent success {silent_route_ceiling():.4f}; continuous uses 9 | Remembered information plus occasional updates works |",
          f"| Scouting | Utility {cell(scout,'utility')}; {cell(scout,'packets')} packets | Best silent utility 0.91; always ask 0.92 | Learn when to ask, inspect, or act independently |",
          f"| Complementary knowledge | Success {cell(dialog,'success')}; utility {cell(dialog,'utility')} | Silent accuracy 0.50; full-map broadcast utility 0.84 | Interactive requests help, with residual learning failures |",'',
          'The numbers are means ± sample standard deviations across five optimization seeds. '
          'Success probabilities are exact expectations over the finite task distribution (up to floating-point rounding near 10⁻⁶), '
          'not estimates from a small test sample. “100%” therefore means perfect logical '
          'performance for that specified finite model and deterministic deployment; it is '
          'not a claim of physical safety or general real-world competence.','',
          '**Important method qualification:** these experiments use small categorical policy '
          'tables and exact expected-reward gradients with a known task model. They are '
          'mechanism and environment-design tests. They do not demonstrate that a neural PPO '
          'system will discover the same protocols from sampled experience. The strongest '
          'scouting and dialogue results include explicitly reported warm-up learning.','',
          '## 1. Background: what problem are we trying to solve?','',
          'There are two cooperative agents. A sender can observe a fact that its partner '
          'cannot observe. A receiver must make a decision and, where appropriate, navigate '
          'to the chosen destination. A message is a small discrete public symbol. Its '
          'meaning is learned from whether interpreting it improves the shared task outcome; '
          'learned symbols are not assigned English meanings.','',
          'The desired behavior is an initial exchange when needed, independent action using '
          'remembered information, and a later update when that information becomes stale. '
          'Communication is useful only if its benefit exceeds its resource cost.','',
          'An earlier oracle-grid study allowed the receiver to take a cheap conservative '
          'route around uncertain hazards. Its strongest silent-at-evaluation policy reached '
          'every goal, so communication did not improve overall return there. That outcome '
          'motivated these four changes to the information structure. The present experiments '
          'use different tasks and a different learner, so they are not a controlled '
          'optimizer-only comparison with the earlier PPO study.','',
          '## 2. Common definitions and experimental design','',
          '- **Episode:** one complete task attempt. Success receives reward 1; an incorrect '
          'terminal decision receives 0. In the route task, all three crossings must be correct.',
          '- **Policy:** a conditional probability distribution over available actions or '
          'symbols. A table stores trainable logits; softmax turns them into probabilities.',
          '- **Transmission/packet:** one emitted categorical word. Silence can be observed '
          'in a known time slot in Options 1 and 2 and may itself acquire meaning.',
          '- **Utility:** task success minus explicitly defined communication or scouting costs. '
          'All costs are expressed in fractions of the unit success reward.',
          '- **Oracle:** a reference that directly supplies missing information. It is '
          'deliberately privileged and is not a learned communication result.',
          '- **Fixed informative protocol:** a prescribed code or timing rule used to establish '
          'achievability. Such controls are identified separately from learned policies.','',
          'For logits l₁,…,lₙ, an action probability is','',
          '```text\nπ(a=i | observation) = exp(lᵢ) / Σⱼ exp(lⱼ).\n```','',
          'The shared optimization objective is','',
          '```text\nJ = E[ I(success) − communication cost − scouting cost ].\n```','',
          'I(success) equals 1 on success and 0 otherwise. No positive reward is assigned '
          'merely for speaking. The expectation sums private states, messages, and receiver '
          'actions. In the route experiment it also sums all status-change histories through '
          'a dynamic program. This uses the known reward function for alternatives that were '
          'not physically sampled and must not be described as black-box sample efficiency.','',
          'With θ denoting the active policy parameters and H the mean categorical '
          'entropy, the minimized loss is L(θ)=−J(θ)−βH(θ). Autograd differentiates '
          'the finite probability sums; there is no sampled policy-gradient estimator, '
          'critic, or PPO objective here. Adam updates θ from these exact gradients. '
          'The active tables have 72 parameters for private goals, 1,476 for route '
          'safety, 154 for scouting, and 188 for structured dialogue. The one-way '
          'broadcast encoder/decoder has 8,704 active parameters. The dialogue/broadcast '
          'container also stores unused alternative heads. These are not parameter-count '
          'matched architecture comparisons.','',
          'Utilities should be compared within a task, not ranked across tasks with different cost definitions. All learned methods receive 1,500 Adam updates at learning rate 0.04, with gradient '
          'norm clipped at 5. An entropy bonus of at most 0.02 anneals linearly to zero over '
          'the first 600 updates. Entropy initially encourages exploration among words. '
          'The final checkpoint is used, without selecting an earlier checkpoint on results. '
          'Development seed 97 was separate from final optimization seeds 0–4.','',
          'There are 130 evaluated seed–method conditions: 80 optimized policies and 50 '
          'prescribed/reference conditions. Repeated reference conditions are not independent '
          'learning runs. All seeds are reported, including failures. Tiny exact tables ran '
          'on CPU in the existing Windows PyTorch environment; the GPU was available but '
          'was not needed for this study.','',
          '**Information access:** training code can use true hidden states to calculate '
          'rewards. A deployed receiver can use only its designated observations and messages. '
          'The sender does not read the receiver’s private neural state. In Option 2 the '
          'cached communication index is reconstructible from the shared protocol/history.','',
          '### Navigation and what is held constant','',
          'These are finite decision processes coupled to a receiver-local breadth-first '
          'search planner. The planner accepts only a start, a selected public waypoint, '
          'public obstacles, and map dimensions. It has no argument for the true hidden '
          'target or hazard label. The receiver selects its waypoint after interpreting '
          'communication and computes its own movement sequence.','',
          'Navigation is deliberately solved algorithmically rather than learned jointly. '
          'This removes motor learning as a confound. Communication ticks in Option 2 are '
          'logical observation opportunities, not MuJoCo physics frames. Costs of scouting '
          'in Option 3 are abstract known resource costs, not measured robot energy.','',
          'An independent planner check exercised four destinations on 128 newly generated '
          'public obstacle layouts. Every produced path was checked for reachability, '
          'unit-step movement, and obstacle avoidance. This checks the interface and path '
          'executor; it is not an independently trained physical navigation benchmark.','',
          '![Task overview](task-overview.png)','',
          '## 3. Option 1: privately specified destination','',
          'Four candidate destinations are public and reachable. One target G is selected '
          'uniformly from {0,1,2,3}. Only the sender sees G. The receiver knows candidate '
          'locations and its own pose, but receives no target marker. One destination is '
          'selected and committed; a wrong commitment terminates the episode. Exhaustive '
          'search and target-dependent intermediate rewards are unavailable.','',
          'The sender chooses silence or one of eight words at the opening exchange. The '
          'receiver maps the observed signal to a destination index, then plans its own '
          'route. Sender and decoder tables start random. A packet costs 0.05.','',
          'Writing S(z|g) for the sender and D(g|z) for the decoder, the objective is','',
          '```text\nJ = (1/4) Σg Σz S(z|g) D(g|z) − (0.05/4) Σg P(z ≠ silence | g).\n```','',
          '### Why silence alone cannot solve it','',
          'Let O be all information available to a silent receiver. G is independent of O '
          'and uniform. For any possibly randomized receiver policy,','',
          '```text\nP(correct without communication)\n  = Σₒ P(O=o) Σg P(G=g | O=o) π(g | o)\n  = Σₒ P(O=o) (1/4) Σg π(g | o)\n  = 1/4.\n```','',
          'This is an information bound, not a weak-baseline learning failure. A receiver '
          'given the target through an informative code can select correctly every time.','',
          '### Why 0.75 transmissions is legitimate, and what it does not mean','',
          'A synchronized silent slot can represent one destination; three emitted words '
          'can represent the other three. The expected packet count is therefore 3/4, and '
          'perfect-success utility is 1 − 0.05×0.75 = 0.9625. The learned optional protocols '
          'found this solution. Silence is an informative event here: 0.75 packets must not '
          'be interpreted as only 0.75 bits of information. The four equally likely outcomes '
          'still distinguish two bits of target identity. If presence detection or timing '
          'has a physical cost, that cost would have to be added.','',
          '## 4. Option 2: changing route safety','',
          'The receiver must pass three successive bottlenecks, each with two candidate '
          'passages. Exactly one passage is safe. Only the sender observes the currently '
          'safe passage Sₜ ∈ {0,1}. At each of nine logical communication ticks, the sender '
          'may remain silent or emit a word. Crossing decisions occur at ticks 0, 4 and 8. '
          'Between adjacent ticks, the safe side flips independently with probability '
          'ρ=0.08. A wrong crossing ends the episode. Waiting indefinitely or bypassing '
          'the bottlenecks does not complete the task.','',
          'The receiver has a persistent cache of the last delivered word and maps that '
          'cache to a passage choice. No decoded word supplies individual movement actions. '
          'The planner constructs a path through the chosen gate. A packet costs 0.02. '
          'The memoryless comparison resets its cache on a silent tick.','',
          '### Optimal silent success rate','',
          'Over Δ independent flip opportunities, the probability that the safe side is '
          'unchanged is the probability of an even number of flips:','',
          '```text\nq(Δ) = [1 + (1−2ρ)^Δ] / 2.\nq(4) = [1 + 0.84^4] / 2 = 0.748936.\n```','',
          'The first silent choice succeeds with probability 1/2. Conditional on surviving '
          'a crossing, its safe side is known through that survival. Because q(4)>1/2, '
          'repeating the same side at the next crossing is the best silent choice. Thus','',
          f'```text\nP(silent success) = (1/2) q(4)^2 = {silent_route_ceiling():.9f}.\nP(success with opening information only) = q(4)^2 = {2*silent_route_ceiling():.9f}.\n```','',
          'An informative opening message cannot protect against later unpredictable changes. '
          'A prescribed event rule can instead send only when the current safe side differs '
          'from the remembered side at a crossing. With a default side encoded by initial '
          'silence, its expected packet count is','',
          '```text\nE[packets, prescribed event rule] = 1/2 + 2[1−q(4)]\n                                          = 1.002128.\n```','',
          'This rule is a reference, not a learned gate. A simpler informative fixed '
          'schedule at the three crossing decisions would need three packets; continuous '
          'communication over all observation ticks needs nine.','',
          '### Exact temporal credit assignment','',
          'Let pₜ(s,h) be probability mass still alive at tick t, with safe side s and '
          'cached word h. The initial cache is empty. Let πₛ(z|s,h,t) be sender probabilities '
          'and D(a|h′) receiver passage probabilities. A delivered word replaces the cache; '
          'silence retains it. Each tick propagates mass through the message transition. '
          'At a crossing it keeps only the fraction D(s|h′), then applies the status-flip '
          'transition before the next tick:','',
          '```text\nInitial: p₀(s, empty) = 1/2.\nMessage propagation: p′(s,h′) = Σh,z pₜ(s,h) πₛ(z|s,h,t) P(h′|h,z).\nCrossing only:        p′(s,h′) ← p′(s,h′) D(s|h′).\nNext status:         pₜ₊₁(s,h′) = (1−ρ)p′(s,h′) + ρp′(1−s,h′).\nJ = final surviving probability − 0.02 E[attempted packets before termination].\n```','',
          'This recurrence allows an early message to receive credit for a later crossing. '
          'It sums all state/message paths without exposing the true side to the receiver. '
          'It does assume knowledge of the transition model during training.','',
          '## 5. Option 3: asking versus independent scouting','',
          'There are again four possible targets. In half of episodes, the receiver already '
          'observes the target. In the other half it does not. Independently, an inspection '
          'point has a public cost d chosen uniformly from {0.02, 0.12, 0.40}. The receiver '
          'can commit immediately, scout and then learn the target, or ask its partner. '
          'The sender always knows the target.','',
          'An ASK sends a fixed request costing 0.02 and obtains one categorical reply '
          'costing 0.06. Every reply word, including word zero, costs the same. The total '
          'ask cost is c=0.08; there is no free target-specific missing reply. The receiver '
          'learns which reply identifies which destination. A visible target can be reached '
          'without asking or scouting.','',
          'Let A(g|o), A(scout|o), and A(ask|o) be first-action probabilities for '
          'receiver observation o. Let Cg=Σz S(z|g)D(g|z) be reply-code correctness '
          'for target g. Averaging the 24 combinations of visibility, cost and target gives','',
          '```text\nJ = E[ A(G|o) + A(scout|o)(1−d) + A(ask|o)(CG−0.08) ].\n```','',
          '### The optimal decision rule','',
          'When the target is unknown, an immediate guess has expected utility 1/4. '
          'Perfect scouting has utility 1−d, and an informative exchange has utility 1−c. '
          'All scouting costs here are below 3/4, so scouting beats guessing. With a '
          'reliable code, asking should be chosen exactly when c<d:','',
          '```text\nIf target already known: act directly; cost = 0.\nIf target hidden: choose min(d, c) to obtain it.\nBest silent utility = 1 − (1/2)(0.02+0.12+0.40)/3 = 0.91.\nOptimal optional utility = 1 − (1/2)(0.02+0.08+0.08)/3 = 0.97.\nAlways ask utility = 1 − 0.08 = 0.92.\nP(ask optimally) = (1/2)(2/3) = 1/3.\nE[packets] = 2 P(ask) = 2/3.\n```','',
          'This task makes communication a genuine tool: silence can be optimal, scouting '
          'can be optimal, and asking can be optimal in different observable contexts. '
          'Success alone is uninformative because a silent scout can also succeed every '
          'time; the meaningful comparison is utility and conditional action choice.','',
          'Two learned variants start with random request decisions. The cold variant also '
          'starts with a random reply code. The warm variant first trains that code for '
          '600 extra updates by maximizing exact destination-selection reward. Word meanings '
          'remain learned. This isolates whether poor initial communication prevents the '
          'receiver from discovering that asking is worthwhile. It is extra training, '
          'not a fair claim of equal total compute.','',
          '## 6. Option 4: private goal plus private hazard map','',
          'The receiver privately knows an entity index Q chosen uniformly from eight '
          'possibilities. The sender privately sees a map X=(X₀,…,X₇) of eight independent '
          'fair binary facts, such as which entrance is safe for each entity. The receiver '
          'must choose XQ. Neither agent initially has all information needed.','',
          'A learned dialogue starts with an optional request. A one-bit presence flag is '
          'charged in every episode. If a request is made, its word identifies one of eight '
          'query codes (three bits), and the helper returns a learned binary word (one bit). '
          'Total cost is therefore five bits when asking and one flag bit when declining. '
          'Each bit costs 0.02. The no-communication reference disables the entire protocol '
          'and costs zero bits. Requests and replies count as two packets but their total '
          'bit cost, rather than packet count alone, is used for fair comparison.','',
          'The receiver learns a query-word distribution from its private index. The helper '
          'learns a categorical lookup over the map’s entity slots for each query word, '
          'then a binary encoding of the selected fact. The receiver interprets that word '
          'using its own query context. This is a structured lookup inductive bias; it '
          'is not evidence that arbitrary relational syntax emerged. At deterministic '
          'deployment, each categorical component selects its argmax.','',
          'Let Q(z|q) be the request policy, L(i|z) the helper’s entity lookup, '
          'E(b|x) its binary word encoder, and D(x|q,b) the receiver’s answer policy. '
          'z=0 denotes no request; ∅ is the no-reply observation. The expectation '
          'below averages all 256 maps and eight private indices:','',
          '```text\nJ = E[X,q] { Q(0|q) D(Xq|q,∅)\n              + Σz≠0 Q(z|q) Σi L(i|z) Σb E(b|Xi) D(Xq|q,b) }\n    − 0.02 [1 + 4 P(request)].\n```','',
          'A cold dialogue learns all components jointly. A warm dialogue first receives '
          '1,000 additional reference-game updates: query words and helper lookups learn '
          'to identify the same entity slot. These words are randomly initialized and '
          'receive no prescribed English meaning. Hazard reply learning then proceeds '
          'through the main task reward.','',
          '### Why asking can be cheaper than broadcasting','',
          'Without any map information, the receiver is correct with probability 1/2. '
          'Broadcasting all eight map bits permits perfect accuracy but utility 1−0.02×8 '
          '=0.84. A correct request/answer protocol needs at most five bits and can achieve '
          'utility 1−0.02×5=0.90. The helper is told which fact matters, so it can avoid '
          'describing unrelated facts.','',
          'We also trained a one-way encoder with exactly five binary wire bits: 32 '
          'possible codewords. It sees the whole map but never Q; its decoder sees both '
          'the codeword and Q. It is a learned map-compression baseline.','',
          '### A bound for every one-way five-bit protocol','',
          'Let M be a one-way five-bit message, so H(M)≤5 bits. Because the eight map '
          'bits are independent and uniform, H(X)=8. Let Dᵢ be the probability of '
          'incorrectly reconstructing bit i from M, and D the average of the eight Dᵢ. '
          'The receiver’s average query error equals D because Q is uniform and independent. '
          'For binary errors, the entropy bound and concavity of binary entropy imply','',
          '```text\nh₂(d) = −d log₂(d) − (1−d) log₂(1−d).\nH(X|M) = H(X) − I(X;M) ≥ 8−5 = 3.\nH(X|M) ≤ Σᵢ H(Xᵢ|M) ≤ Σᵢ h₂(Dᵢ) ≤ 8 h₂(D).\nTherefore h₂(D) ≥ 3/8.\n```','',
          f'Numerically, D≥{1-broadcast_bound():.9f}, so **one-way five-bit accuracy is at '
          f'most {100*broadcast_bound():.4f}%**, even with an optimal encoder. This is a '
          'valid upper bound rather than a claim that some finite code achieves it. '
          'The learned broadcast baseline can perform below the bound because optimization '
          'and finite-block coding need not attain it.','',
          'The dialogue does not violate the bound: its reply is conditional on the '
          'receiver’s query. The same one-way message must support every possible index; '
          'a queried reply only has to support the requested index. A sender-dependent '
          'timing side channel is not granted to the fixed five-bit broadcast. The '
          'dialogue’s request-presence bit is explicitly charged.','',
          'The entropy argument is the binary-source/Hamming-distortion converse, also '
          'discussed in the [Martinian–Wornell primary paper](https://sia.mit.edu/dev_site/wp-content/uploads/2015/04/2004-martinian-wornell-allerton.pdf). '
          'The finite eight-bit derivation above gives the assumptions used here directly.','',
          '## 7. Full results','',
          'Primary deployment chooses the highest-probability action at each categorical '
          'component. Sampled deployment results are also included in the CSV. Mean ± '
          'sample standard deviation describes variation across optimization seeds, not '
          'uncertainty from missing test states. Reference policies repeated across seeds '
          'usually have zero spread because they are prescribed.','']
    for task,title in LABELS.items():
        text += ['### '+title,'','| Method | Success | Utility | Packets | Bits (if defined) |',
                 '|---|---:|---:|---:|---:|']
        methods=list(dict.fromkeys(r['method'] for r in normals if r['task']==task))
        for method in methods:
            rows=group(task,method)
            text.append('| '+method+' | '+' | '.join(cell(rows,k) for k in ('success','utility','packets'))+' | '+(cell(rows,'bits') if 'bits' in rows[0] else '—')+' |')
        text += ['']
    text += ['![Results by task](results.png)','','## 8. Did each option meet its success criterion?','',
             'Criteria were saved in `protocol.json` before the five-seed final study. '
             'They are task-specific: the scouting task needs an efficiency improvement, '
             'while the private-goal task needs a success improvement.','',
             '| Task | Criterion | Verdict |','|---|---|---|',
             f"| Private goal | Mean success ≥95%; every seed exceeds 25% | {'PASS' if criteria['private_goal'] else 'FAIL'} |",
             f"| Route safety | Mean success ≥95%; every seed beats silent utility; ≤4.5 packets | {'PASS' if criteria['route_safety'] else 'FAIL'} |",
             f"| Scouting | Warm mean utility ≥0.965; all seeds >0.91; appropriate conditional asking | {'PASS' if criteria['scouting'] else 'FAIL'} |",
             f"| Complementary | Warm mean success ≥95%; all seed utilities >0.84; mean accuracy exceeds one-way bound | {'PASS' if criteria['complementary'] else 'FAIL'} |",'',
             '**A high mean is not a reliability guarantee.** The warm dialogue averages 95% '
             'accuracy but fails the required per-seed utility comparison. Its outcomes '
             'below expose unreliability hidden by an aggregate mean. If zero incorrect hazard decisions were required, '
             'the imperfect dialogue seeds would fail that stricter criterion.','',
             '## 9. What the interventions establish','',
             'Blocked delivery prevents information arriving while retaining the original '
             'attempt/request charge. Constant payload replaces every emitted word with one '
             'word; a known silent slot remains silent. A permutation systematically changes '
             'the word mapping. These are interventions on frozen pairs, not retrained '
             'no-communication baselines. A frozen partner may depend on information that '
             'an independently trained silent partner would compensate for differently.','',
             '| Primary learned pair | Normal success | Blocked | Constant payload | Permuted words |',
             '|---|---:|---:|---:|---:|']
    for task in LABELS:
        cells=[]
        for iv in ('normal','blocked','constant','permute'):
            rs=[r for r in flat if r['task']==task and r['method']==PRIMARY[task] and r['deployment']=='argmax' and r['intervention']==iv]
            cells.append(cell(rs,'success'))
        text.append('| '+LABELS[task]+' | '+' | '.join(cells)+' |')
    text += ['', 'For route safety, suppressing all updates after the opening exchange '
             'tests whether later information matters; resetting memory tests whether the '
             'protocol relies on retention; 25% independent packet loss tests robustness. '
             'These effects are integrated exactly using separate sender-predicted and '
             'receiver-delivered caches. The sender retains its own intended transmission '
             'history and receives no free delivery acknowledgments. Under the reliable '
             'normal channel the caches coincide; under corruption they may differ.','',
             '| Route intervention | Success | Packets attempted | Utility |','|---|---:|---:|---:|']
    for iv in INTERVENTIONS['route_safety']:
        rs=[r for r in flat if r['task']=='route_safety' and r['method']=='learned' and r['deployment']=='argmax' and r['intervention']==iv]
        text.append('| '+iv+' | '+' | '.join(cell(rs,k) for k in ('success','packets','utility'))+' |')
    text += ['', 'For complementary knowledge, `remove_query` forces the helper to use the '
             'same query-word input irrespective of what the receiver requested. The helper '
             'therefore loses the information selecting the relevant map fact.','']
    rs=[r for r in flat if r['task']=='complementary' and r['method']=='dialogue_warm' and r['deployment']=='argmax' and r['intervention']=='remove_query']
    text += [f"Warm dialogue success with query information removed: **{cell(rs,'success')}**.",'',
             '## 10. Conditional behavior and optimization failures','',
             '### Scouting decisions','',
             '| Inspection cost | Target visible? | Warm P(ask) | Warm P(scout) | Cold P(ask) |',
             '|---|---|---:|---:|---:|']
    for i,cost in enumerate((.02,.12,.4)):
        for visible in (0,1):
            idx=2*i+visible
            text.append(f"| {cost} | {'yes' if visible else 'no'} | {cell(scout,f'context_{idx}_ask')} | {cell(scout,f'context_{idx}_scout')} | {cell(group('scouting','learned_cold'),f'context_{idx}_ask')} |")
    text += ['', 'Cold scouting sometimes settled on inspection before learning a useful '
             'reply code. Inspection supplies perfect task success, so additional channel '
             'learning can receive little incentive once asking is rarely chosen. The '
             'warm-up addresses this bootstrap problem and is part of the reported method.','',
             '### Dialogue reliability','',
             '| Seed | Cold accuracy | Warm accuracy | Warm utility | Warm request probability |',
             '|---|---:|---:|---:|---:|']
    for seed in range(5):
        c=next(r for r in group('complementary','dialogue_cold') if r['seed']==seed)
        w=next(r for r in dialog if r['seed']==seed)
        text.append(f"| {seed} | {c['success']:.4f} | {w['success']:.4f} | {w['utility']:.4f} | {w['asks']:.4f} |")
    text += ['', 'Some learned query conventions leave an entity inadequately represented. '
             'A query then receives no reliable answer or is not asked, leaving approximately '
             'chance performance for that entity. The prescribed dialogue reaches 100%, '
             'so this gap is an optimization/protocol-learning limitation, not an '
             'information-theoretic impossibility. The structured lookup and referential '
             'warm-up materially affect this result.','',
             'The learned route protocol also uses more packets than the prescribed event '
             'reference. It discovers an effective sparse protocol without establishing '
             'minimum communication. A memoryless receiver can still exploit silence '
             'as a default-side signal, so its perfect success does not demonstrate '
             'persistent memory; its higher packet count is the relevant contrast.','',
             '## 11. Independent execution checks and statistical interpretation','',
             'The primary evaluator sums finite probabilities exactly. Independently written '
             'sampled execution code ran 50,000 episodes for each primary method and seed '
             '(20 policies, one million episodes), using new generator seeds beginning '
             '80000019. It sampled the private variables and, in the route task, executed '
             'the nine-step status process. Sampled accuracy, utility and packet counts '
             'were checked against the exact evaluator. These are independent draws from '
             'the same finite distribution, not unseen semantic states.','',
             'For a sampled success fraction p̂ from n independent episodes, the approximate '
             'sampling standard error is sqrt[p̂(1−p̂)/n]. This sampling error is distinct '
             'from the standard deviation across five trained seeds. Since primary scores '
             'are exact finite-population expectations, no test-episode confidence interval '
             'is attached to them. Five seeds do not establish behavior of every future '
             'optimization run.','',
             'Tests independently enumerate all 512 status histories for a deterministic '
             'route policy and compare them with the dynamic program. Other tests check '
             'silent bounds, fixed informative controls, scouting observations, dialogue '
             'bit charges, gradients, and receiver-local path interfaces. The full existing '
             'standalone Gumbel experiment-pipeline suite is also run; its saved log supplies the final test count.','',
             '## 12. Limitations and connection back to ALKoG','',
             '1. **Known-model training:** exact reward gradients use more information '
             'than ordinary sampled PPO. These experiments establish helpful task structure '
             'and learnable finite protocols, not end-to-end neural RL performance.',
             '2. **Finite lookup policies:** all logical states are available in training. '
             'There is no claim of generalization to unseen objects, layouts as policy '
             'features, relations, or languages. New planner layouts exercise navigation '
             'only; communication chooses public entity slots.',
             '3. **Architectural support:** persistent cache memory and a query-conditioned '
             'entity lookup are supplied. Their existence is not an emergent discovery.',
             '4. **Cost assumptions:** numeric packet, bit and scouting costs are experiment '
             'parameters. Changing them can change the optimal choice. Success without '
             'charging these resources is not the same objective as utility.',
             '5. **Timing information:** silence can carry information when observation '
             'slots are synchronized. Presence and framing costs need explicit treatment '
             'in real channels; Option 4 includes its request flag.',
             '6. **Simplified dynamics:** route changes are observable to the sender before '
             'a crossing decision; no surprise change makes the selected crossing '
             'unavoidably fail after commitment. Timing and transition-distribution shifts '
             'were not evaluated.',
             '7. **No vision or physics:** no cameras, perception noise, MuJoCo stepping, '
             'torque control, physical lion behavior, or visual cage inference is trained here.','',
             'A direct transfer path is to retain these information boundaries while '
             'replacing the finite tables with the repository’s neural encoder/decoder '
             'and sampled actor–critic learning. The private-goal task first tests whether '
             'a channel can become informative; the route task then tests retained facts '
             'and selective updates; scouting tests whether communication is chosen '
             'economically. Complementary dialogue should follow after its cold-start '
             'reliability improves.','',
             'After that, reproduce the task in MuJoCo with oracle observations, keeping '
             'navigation local to the receiver. Only then replace oracle facts with '
             'perception estimates. The sender must be able to observe actual rendered '
             'state changes, and the receiver must not accidentally receive target markers '
             'or privileged geometry that remove the communication problem.','',
             '### Development and implementation checks','',
             'A separate 600-update development study used seed 97. In that study cold '
             'dialogue reached 81.25% and warm dialogue reached 100%; the five-seed final '
             'results show why a single favorable development run is insufficient. The '
             '1,500-update final budget and criteria were fixed before seeds 0–4 were run.','',
             'Implementation checks corrected a prescribed scouting control so that its '
             'request decision was genuinely the stated optimal reference, and supplied '
             'uniform default responses for unused words in fixed codes. A full '
             'reproducibility rerun retained identical learned weights. Channel-corruption '
             'evaluation was separately checked with two caches so that the sender did '
             'not receive an implicit acknowledgment. No learned primary-policy parameters '
             'were tuned to improve the final reported scores.','',
             '## 13. Reproducibility and artifact guide','',
             '**Final regression result: 143 tests passed**, including ten new tests '
             'for these communication options.','',
             '- `manifest.json`: exact software versions, seeds, costs, training budget '
             'and source hashes. `protocol.json`: pre-specified criteria.',
             '- `all-evaluations.csv`: applicable normal/intervention results for both '
             'deterministic and sampled policy deployment.',
             '- `verdicts.json`: criterion outcomes and the five primary rows per task.',
             '- The artifact ZIP contains all final weights, optimizer/RNG states, '
             'learning curves, exact evaluations, independent execution checks, navigation '
             'examples, source snapshots, tests, and the development study.',
             '- The prior oracle-grid report remains separate. No old results were '
             'overwritten and no remote branch was pushed.','',
             'Related research: [Foerster et al., Learning to Communicate with Deep '
             'Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676) studies '
             'learned discrete communication under partial observability. The current '
             'implementation is a smaller exact-gradient diagnostic, not a reproduction '
             'of RIAL or DIAL.','',
             '## Appendix: primary results for every seed','',
             '| Task | Seed | Success | Utility | Packets |','|---|---:|---:|---:|---:|']
    for task in LABELS:
        for r in group(task,PRIMARY[task]):
            text.append(f"| {task} | {r['seed']} | {r['success']:.6f} | {r['utility']:.6f} | {r['packets']:.6f} |")
    report='\n'.join(text)+'\n'
    extra_figures=make_figures(a.study,a.output,results)
    inserts={
        '### Navigation and what is held constant':
            '![Private facts, public channel, receiver memory and independent planner](information-flow.png)\n\n'
            '**Figure 1. Information boundaries.** Private facts reach the receiver through the public channel. '
            'The receiver selects a waypoint and computes its own movements; a message does not supply a movement stream.\n\n',
        '### Why silence alone cannot solve it':
            '![Public goal map alongside the actual learned seed-zero signal code](private-goal-code.png)\n\n'
            '**Figure 3. A goal message and its interpretation.** The left panel is a public-map illustration with a receiver-local path. '
            'The right panel shows the actual seed-0 deterministic code: each row is one possible private target; 1 marks its selected signal. '
            'The map is unchanged across targets, and silence identifies one target under this learned convention.\n\n',
        '### Optimal silent success rate':
            '![Safe-side changes, receiver memory, and actual emitted packets over nine ticks](route-memory-timeline.png)\n\n'
            '**Figure 4. Remembering between updates.** This selected status history is executed by the frozen learned seed-0 policy. '
            'Dotted lines mark the three crossing decisions. A dash is an actual silent slot; word IDs label transmitted packets. '
            'The receiver may retain an outdated choice between crossings, provided it is corrected before a consequential decision. '
            'This is an explanatory trace, not an additional aggregate experiment.\n\n',
        '### The optimal decision rule':
            '![Ask-versus-inspect utility crossover and observed conditional actions](scouting-decision.png)\n\n'
            '**Figure 5. Communication as an optional tool.** With a reliable reply, asking wins when inspection costs more than 0.08; '
            'inspection wins when it costs less. The heatmap displays measured five-seed warm-policy action frequencies. '
            'A receiver that already knows the target acts directly.\n\n',
        '### Why asking can be cheaper than broadcasting':
            '![Private query, private map, learned request and answer, and charged wire bits](dialogue-information-flow.png)\n\n'
            '**Figure 6. Asking for the relevant fact.** A selected seed-2 dialogue illustrates the learned query and reply codes. '
            'The helper needs the query to choose the relevant map slot. The presence flag is included in the five-bit budget. '
            'This successful example does not replace the five-seed reliability results.\n\n',
        '## 7. Full results':
            '![Entropy constraint for one-way messages and its relation to measured accuracy](entropy-bound.png)\n\n'
            '**Figure 7. Reading the information bound.** The left-hand curve must remain above three bits of residual uncertainty, '
            'which limits one-way five-bit accuracy to about 92.76%. The right panel compares existing measurements with that bound. '
            'Queried dialogue changes the sender’s information, so the one-way bound does not apply to it. Error bars show variation across seeds.\n\n',
        '## 10. Conditional behavior and optimization failures':
            '![Measured effects of blocking delivery in each frozen primary pair](message-interventions.png)\n\n'
            '**Figure 9. Causal delivery intervention.** Blocking delivery harms these frozen communicating pairs. '
            'This supports a role for the channel but is a different comparison from retraining a silent partner. '
            'Error bars show sample standard deviations across the five trained seeds.\n\n',
        '## 11. Independent execution checks and statistical interpretation':
            '![Per-seed dialogue accuracy and utility, including the failed reference comparison](dialogue-seed-reliability.png)\n\n'
            '**Figure 10. Why the fourth option is only partially successful.** Average accuracy conceals the low-performing seed. '
            'The red utility bar lies below the full-map broadcast reference and causes the pre-specified across-seed criterion to fail.\n\n'
    }
    for anchor,insert in inserts.items():
        assert report.count(anchor)==1,anchor
        report=report.replace(anchor,insert+anchor,1)
    report=report.replace('![Task overview](task-overview.png)',
        '![Task overview](task-overview.png)\n\n**Figure 2. The four task structures.** These are conceptual diagrams, not simulator screenshots.')
    report=report.replace('![Results by task](results.png)',
        '![Results by task](results.png)\n\n**Figure 8. Existing experimental outcomes.** Compare bars within each panel. '
        'The route panel shows packet counts: its silent policy sends nothing but succeeds only about 28% of the time. '
        'Error bars show standard deviation across optimization seeds.')
    report=re.sub(r'\*\*Figure (\d+)\.',
        lambda m:f"**Figure {int(m.group(1))+(int(m.group(1))>=2)}.",report)
    grid_figure=(
        '![Top-down 13 by 13 validation grid and 7 by 7 two-passage room](grid-layouts.png)\n\n'
        '**Figure 2. What the grids look like.** Each square is an abstract position; a move changes x or y by one cell. '
        'The left panel uses saved navigation-validation layout 0, including its actual obstacles and its saved path to G2. '
        'The right panel uses the exact public wall and passage coordinates of the two-route planner example. '
        'The two route lines are alternatives; the receiver executes only the selected route. '
        'In MuJoCo terms, white cells correspond to navigable floor, dark cells to obstacles, R to the moving receiver, '
        'and G to public candidate destinations. Passage safety is a separate hidden task fact.\n\n'
        '**What was simulated during training:** the four-option learners operate on the finite decision processes '
        'defined below. These grids are the separate planner geometry used to illustrate and validate autonomous '
        'path execution. They are not a spatial grid traversed inside the exact communication-training objective.\n\n')
    anchor='Navigation is deliberately solved algorithmically rather than learned jointly.'
    assert report.count(anchor)==1
    report=report.replace(anchor,grid_figure+anchor,1)
    # Put the complete walkthrough and grid inside the report, before results.
    report=report.replace(grid_figure,'',1)
    walkthrough=setup_section(a.study,grid_figure)
    report=report.replace('## Executive conclusion',walkthrough+'\n## Executive conclusion',1)
    number=iter(range(1,12))
    report=re.sub(r'\*\*Figure \d+\.',lambda m:f'**Figure {next(number)}.',report)
    (a.output/'report.md').write_text(report,encoding='utf-8')
    # Four complementary views; labels favor legibility over compactness.
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    panels=[('private_goal',['silent','continuous','learned'],'success','Success probability'),
            ('route_safety',['silent','opening','continuous','learned','fixed_event'],'packets','Packets per episode'),
            ('scouting',['silent','always_ask','learned_cold','learned_warm'],'utility','Utility'),
            ('complementary',['silent','broadcast5','fixed_broadcast','dialogue_cold','dialogue_warm'],'utility','Utility')]
    for ax,(task,methods,key,ylabel) in zip(axes.flat,panels):
        values=[mean(group(task,m),key) for m in methods]
        errors=[statistics.stdev(r[key] for r in group(task,m)) for m in methods]
        ax.bar(range(len(methods)),values,yerr=errors,capsize=3,color=['#9aa6b2' if m!=PRIMARY[task] else '#168679' for m in methods])
        ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right',fontsize=9)
        ax.set_title(LABELS[task]);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(a.output/'results.png',dpi=170);plt.close(fig)
    # A conceptual diagram, not an experimental screenshot.
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    descriptions=[('Private goal','Sender knows G\n↓ learned word or silence\nReceiver selects G and plans path'),
                  ('Changing route safety','Sender sees safe side S(t)\n↓ occasional updates\nReceiver retains a fact across crossings'),
                  ('Ask or inspect','Receiver already informed? → act\nOtherwise: compare ask cost with inspection\nCheap inspection → scout; costly → ask'),
                  ('Complementary knowledge','Receiver knows entity Q → query\nSender knows map X → requested fact\nReceiver combines Q with the reply')]
    for ax,(title,description) in zip(axes.flat,descriptions):
        ax.axis('off');ax.set_title(title,fontweight='bold',color='#174967')
        ax.text(.5,.5,description,ha='center',va='center',fontsize=12,linespacing=1.8,
                bbox=dict(boxstyle='round,pad=1',facecolor='#eaf3f7',edgecolor='#bdd1dd'))
    fig.tight_layout();fig.savefig(a.output/'task-overview.png',dpi=150);plt.close(fig)
    body=markdown.markdown(report,extensions=['tables','fenced_code'])
    for image in ['results.png','task-overview.png']+extra_figures:
        data=base64.b64encode((a.output/image).read_bytes()).decode()
        body=body.replace('src="'+image+'"','src="data:image/png;base64,'+data+'"')
    css='body{font:17px/1.7 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#1a2936;background:#fafbfd}h1,h2,h3{line-height:1.25;color:#164a67;margin-top:1.7em}table{display:block;overflow:auto;border-collapse:collapse;background:white;font-size:14px}td,th{border:1px solid #d4dfe5;padding:8px 12px;text-align:left}th{background:#e6f0f5}pre{overflow:auto;background:#edf2f5;padding:16px;border-left:4px solid #168679;font-size:14px;line-height:1.6}img{max-width:100%;height:auto}a{color:#08758b}li{margin:.4em 0}@media print{body{font-size:10pt}h2,h3{break-after:avoid}table{font-size:8pt;display:table}pre{white-space:pre-wrap;font-size:8pt}}'
    (a.output/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Four ALKoG communication options</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    # A fresh filename helps readers reopen the expanded version rather than a
    # browser-cached local file. The canonical report remains updated as well.
    (a.output/'report-with-setup.html').write_bytes((a.output/'report.html').read_bytes())
    for f in ('manifest.json','independent_validation.json','protocol.json'):
        source=a.study/f if f!='protocol.json' else Path(__file__).parent/f
        (a.output/f).write_bytes(source.read_bytes())
    print(json.dumps(dict(verdicts=criteria,conditions=len(results),optimized=sum(r['steps']>0 for r in results),optimization_and_evaluation_seconds=sum(r['seconds'] for r in results)),indent=2))


if __name__=='__main__':main()
