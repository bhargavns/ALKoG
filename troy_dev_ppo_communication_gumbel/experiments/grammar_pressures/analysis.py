"""Field-linguist analysis of a frozen emergent language.

Everything is measured from behaviour in controlled contexts, the way one
would elicit a paradigm from a speaker:

    lexicon       put the target among rooms that differ from it in ONE respect
                  (only the agent's type, only the predicate, ...) and record
                  what is said.  If the same short form recurs for a value
                  whatever the rest of the event is, that form is a word.
    word classes  is the word for 'lion' the same whether the lion is the sole
                  participant, the agent or the patient?  Are entity words,
                  property words and predicate words distinct sets of forms?
    combination   in contexts that differ in TWO respects, is the utterance the
                  two words side by side?  In a consistent order?
    productivity  the lexicon and ordering rules are induced from TRAINING
                  events only; utterances are then synthesised for held-out
                  events and given to the listener, and compared with what the
                  speaker itself says there.
    brevity       does utterance length track how much the context demands?
    roles         what is said when the role-reversed event is in the room?
    learnability  how fast does a naive listener acquire the language?
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from .world import PARTITIONS, TEST_PARTITIONS, NC, TYPES, COLOURS, PRED1, PRED2
from .agents import Listener, strip, pack
from .train import load

KIND = dict(pred='pred', S_type='type', A_type='type', P_type='type', S_colour='colour', A_colour='colour', P_colour='colour')


def value_name(valency, slot, value):
    if slot == 'pred': return (PRED1 if valency == 1 else PRED2)[value]
    return (TYPES if slot.endswith('type') else COLOURS)[value]


def make_context(world, rng, i, slots, same_partition=True):
    """Target i plus one distractor per entry of `slots`; a distractor changes
    that slot ('swap' = reverse the roles).  Returns [target, d1, ...] or None."""
    ids = [i]; cur = world.slots(i); sizes = world.slot_sizes(i)
    for slot in slots:
        if slot == 'swap': options = [int(world.swap[i])] if world.swap[i] >= 0 else []
        else: options = [world.edit(i, slot, v) for v in rng.permutation(sizes[slot]) if v != cur[slot]]
        options = [j for j in options if j is not None and j not in ids and world.partition[j] >= 0
                   and (not same_partition or world.partition[j] == world.partition[i])]
        if not options: return None
        ids.append(options[0])
    return ids


class Informant:
    """The trained pair, queried in explicit contexts."""
    def __init__(self, speaker, listener, world, config):
        self.s, self.l, self.w, self.c = speaker, listener, world, config

    @torch.no_grad()
    def say(self, contexts):
        ids = torch.tensor(contexts, device=self.w.device); out = []
        for chunk in ids.split(1024):
            cand = self.w.features[chunk]
            out += strip(self.s(cand[:, 0], cand if self.c['context'] else None)['tokens'], self.s.eos)
        return out

    @torch.no_grad()
    def choose(self, contexts, messages):
        ids = torch.tensor(contexts, device=self.w.device)
        tok = pack(messages, self.l.eos, self.c['max_len'], self.w.device)
        return self.l(self.w.features[ids], tok).argmax(-1).tolist()

    def understand(self, contexts, messages):
        """True where the listener picks the target (index 0) for the given strings."""
        return [p == 0 for p in self.choose(contexts, messages)]


def contexts_for(world, rng, events, pattern, limit):
    """Contexts whose distractors contrast `pattern` distinct slots (1, 2 or 3)."""
    out = []
    for i in rng.permutation(events):
        slots = list(world.slot_sizes(i))
        chosen = list(rng.choice(slots, pattern, replace=False))
        spec = chosen + [chosen[int(rng.integers(len(chosen)))] for _ in range(3 - pattern)]
        ctx = make_context(world, rng, int(i), spec)
        if ctx is not None: out.append((ctx, tuple(sorted(chosen, key=slots.index))))
        if len(out) >= limit: break
    return out


def modal(counter): return counter.most_common(1)[0][0]
def share(counter): return counter.most_common(1)[0][1] / sum(counter.values())
def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def lexicon(inf, rng, events, per_event=2):
    """Words elicited from one-contrast contexts: (valency, slot, value) -> Counter of forms."""
    w = inf.w; ctxs = []; keys = []
    for i in events:
        for slot in w.slot_sizes(i):
            for _ in range(per_event):
                ctx = make_context(w, rng, int(i), [slot] * 3)
                if ctx is None: continue
                ctxs.append(ctx); keys.append((int(w.events[i, 0]), slot, int(w.slots(i)[slot])))
    forms = defaultdict(Counter)
    for key, m in zip(keys, inf.say(ctxs)): forms[key][m] += 1
    return forms


def lexicon_report(forms):
    words = {k: modal(c) for k, c in forms.items()}
    by_kind = defaultdict(list)
    for (val, slot, v), c in forms.items(): by_kind[KIND[slot]].append(share(c))
    def shared(a, b, kind):       # same form for the same value in two roles
        pairs = [(words.get((1 if a == 'S' else 2, f'{a}_{kind}', v)), words.get((1 if b == 'S' else 2, f'{b}_{kind}', v)))
                 for v in range(6 if kind == 'type' else 4)]
        return mean(x == y for x, y in pairs if x is not None and y is not None)
    def distinct(slot, val):      # different values of a slot get different forms
        fs = [f for (vv, s, v), f in words.items() if s == slot and vv == val]
        return len(set(fs)) / max(1, len(fs))
    ent = {words[k] for k in words if KIND[k[1]] == 'type'}; col = {words[k] for k in words if KIND[k[1]] == 'colour'}
    prd = {words[k] for k in words if k[1] == 'pred'}
    return dict(
        word_consistency=mean(share(c) for c in forms.values()),      # 1.0: one form per value whatever the rest of the event is
        word_consistency_by_kind={k: mean(v) for k, v in by_kind.items()},
        mean_word_length=mean(len(f) for f in words.values()),
        values_distinguished=mean(distinct(s, v) for s, v in (('A_type', 2), ('P_type', 2), ('S_type', 1), ('A_colour', 2), ('P_colour', 2), ('S_colour', 1), ('pred', 1), ('pred', 2))),
        type_word_shared_A_P=shared('A', 'P', 'type'), type_word_shared_S_A=shared('S', 'A', 'type'), type_word_shared_S_P=shared('S', 'P', 'type'),
        colour_word_shared_A_P=shared('A', 'P', 'colour'), colour_word_shared_S_A=shared('S', 'A', 'colour'), colour_word_shared_S_P=shared('S', 'P', 'colour'),
        class_overlap=dict(type_colour=len(ent & col), type_pred=len(ent & prd), colour_pred=len(col & prd)),
        distinct_forms=dict(type=len(ent), colour=len(col), pred=len(prd)),
        words={f'{("intr" if k[0] == 1 else "tr")}.{k[1]}={value_name(k[0], k[1], k[2])}': ' '.join(map(str, f)) for k, f in sorted(words.items())})


def combination(inf, rng, events, words, limit=1500, rules=None):
    """Two-contrast contexts: is the utterance word(X) + word(Y)?  `rules` maps a
    slot pair to the order induced from training (used to synthesise utterances)."""
    w = inf.w; items = contexts_for(w, rng, events, 2, limit)
    said = inf.say([c for c, _ in items]); own = inf.understand([c for c, _ in items], said)
    concat = bag = known = predicts = 0; order = defaultdict(Counter); synth_ctx = []; synth_msg = []
    for (ctx, (x, y)), m in zip(items, said):
        i = ctx[0]; val = int(w.events[i, 0]); cur = w.slots(i)
        wx, wy = words.get((val, x, int(cur[x]))), words.get((val, y, int(cur[y])))
        if wx is None or wy is None: continue
        known += 1; bag += sorted(m) == sorted(wx + wy)
        if m == wx + wy: concat += 1; order[(val, x, y)]['xy'] += 1
        elif m == wy + wx: concat += 1; order[(val, x, y)]['yx'] += 1
        if rules is not None and (val, x, y) in rules:
            guess = wx + wy if rules[(val, x, y)] == 'xy' else wy + wx
            synth_ctx.append(ctx); synth_msg.append(guess); predicts += guess == m
    res = dict(contexts=len(items), speaker_understood=mean(own), concatenative=concat / max(1, known), same_bag=bag / max(1, known),
               order_consistency=mean(share(c) for c in order.values()), mean_length=mean(len(m) for m in said))
    if rules is not None:
        res.update(synthesised=len(synth_msg), synthesis_coverage=len(synth_msg) / max(1, len(items)),
                   synthesis_understood=mean(inf.understand(synth_ctx, synth_msg)) if synth_msg else None,
                   synthesis_predicts_speaker=predicts / max(1, len(synth_msg)))
    return res, {k: modal(c) for k, c in order.items()}


def brevity(inf, rng, events, limit=600):
    out = {}
    for k in (1, 2, 3):
        items = contexts_for(inf.w, rng, events, k, limit); said = inf.say([c for c, _ in items])
        out[str(k)] = dict(length=mean(len(m) for m in said), accuracy=mean(inf.understand([c for c, _ in items], said)))
    return out


def roles(inf, rng, events, words, limit=600):
    """The role-reversed twin is in the room (plus two rooms with another predicate)."""
    w = inf.w; ctxs = []
    for i in rng.permutation([e for e in events if w.events[e, 0] == 2]):
        ctx = make_context(w, rng, int(i), ['swap', 'pred', 'pred'], same_partition=False)
        if ctx is not None: ctxs.append(ctx)
        if len(ctxs) >= limit: break
    if not ctxs: return None
    said = inf.say(ctxs); pick = inf.choose(ctxs, said); has = Counter()
    sub = lambda m, f: bool(f) and any(m[q:q + len(f)] == f for q in range(len(m) - len(f) + 1))
    for ctx, m in zip(ctxs, said):
        cur = w.slots(ctx[0])
        for slot in ('A_type', 'P_type', 'A_colour', 'P_colour'): has[slot] += sub(m, words.get((2, slot, int(cur[slot]))))
    # order test: reverse the string; does the listener now choose the role-reversed room?
    rev = inf.choose(ctxs, [m[::-1] for m in said])
    moved = [r == 1 for r, m, p in zip(rev, said, pick) if p == 0 and m != m[::-1]]
    return dict(contexts=len(ctxs), accuracy=mean(p == 0 for p in pick), error_is_reversed_twin=mean(p == 1 for p in pick if p != 0),
                mean_length=mean(len(m) for m in said), mentions={k: v / len(ctxs) for k, v in has.items()},
                reversed_string_picks_reversed_roles=mean(moved), lengths=dict(Counter(len(m) for m in said).most_common(4)))


def entanglement(inf, events):
    """For a context-blind speaker: best single-slot mutual information of each
    symbol position as a share of that position's entropy (1 = one slot per position)."""
    if inf.c['context']: return None
    w = inf.w; tr = [int(i) for i in events if w.events[i, 0] == 2]
    said = inf.say([[i, i, i, i] for i in tr])
    L = Counter(map(len, said)).most_common(1)[0][0]; idx = [k for k, m in enumerate(said) if len(m) == L]
    if L == 0 or len(idx) < 50: return dict(modal_length=L, unique_messages=len(set(said)), events=len(said))
    M = np.array([said[k] for k in idx]); ev = w.events[[tr[k] for k in idx]]
    slots = [ev[:, 1], ev[:, 2] // NC, ev[:, 2] % NC, ev[:, 3] // NC, ev[:, 3] % NC]
    def mi(a, b):
        a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
        j = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(j, (a, b), 1); j /= j.sum()
        e = j.sum(1)[:, None] * j.sum(0)[None]; m = j > 0
        return float((j[m] * np.log2(j[m] / e[m])).sum())
    purity = [max(mi(M[:, p], s) for s in slots) / max(1e-9, mi(M[:, p], M[:, p])) for p in range(L)]
    return dict(modal_length=L, position_purity=float(np.mean(purity)), unique_messages=len(set(said)), events=len(said))


@torch.no_grad()
def accuracy(inf, partition, sets=256, seed=0, mode='normal'):
    w, c = inf.w, inf.c; g = torch.Generator(device=w.device).manual_seed(900 + seed)
    b = w.sample(sets, partition, g, c['n_cand'], tuple(c['focus']))
    tok = inf.s(b['private'], b['candidates'] if c['context'] else None)['tokens']
    if mode == 'shuffle': tok = tok[torch.randperm(len(tok), generator=g, device=tok.device)]
    if mode == 'blocked': tok = torch.full_like(tok, inf.s.eos)
    return float((inf.l(b['candidates'], tok).argmax(-1) == b['target']).float().mean())


def newcomer(inf, budgets=(50, 150, 500), seed=0):
    """A naive listener learns from the frozen speaker on TRAINING events."""
    w, c = inf.w, inf.c; torch.manual_seed(5000 + seed); g = torch.Generator(device=w.device).manual_seed(5000 + seed)
    listener = Listener(c['vocab'], c['width']).to(w.device); opt = torch.optim.Adam(listener.parameters(), lr=c['lr']); out = {}
    probe = Informant(inf.s, listener, w, c)
    for step in range(max(budgets)):
        b = w.sample(c['sets'], 'train', g, c['n_cand'], tuple(c['focus']))
        with torch.no_grad(): tok = inf.s(b['private'], b['candidates'] if c['context'] else None)['tokens']
        loss = F.cross_entropy(listener(b['candidates'], tok), b['target']); opt.zero_grad(); loss.backward(); opt.step()
        if step + 1 in budgets:
            out[str(step + 1)] = dict(train=accuracy(probe, 'train', 128, seed), heldout=mean(accuracy(probe, p, 128, seed) for p in TEST_PARTITIONS))
    return out


def gloss(words_inv, m):
    """Greedy longest-match segmentation of an utterance into known words."""
    out = []; q = 0; longest = max((len(f) for f in words_inv), default=1)
    while q < len(m):
        for n in range(min(longest, len(m) - q), 0, -1):
            if m[q:q + n] in words_inv:
                out.append(f"{''.join(map(str, m[q:q + n]))}:{'/'.join(sorted(words_inv[m[q:q + n]]))}"); q += n; break
        else: out.append(f'{m[q]}:?'); q += 1
    return '  '.join(out)


def analyse(run_dir, device='cpu', fresh=True):
    speaker, listener, world, config = load(run_dir, device); torch.set_num_threads(1)
    inf = Informant(speaker, listener, world, config); seed = config['seed']; rng = np.random.default_rng(100 + seed)
    part = world.partition; train = np.nonzero(part == 0)[0]; held = np.nonzero(part > 0)[0]
    res = dict(config=config)
    res['accuracy'] = {p: dict(normal=accuracy(inf, p, 256, seed), shuffled=accuracy(inf, p, 256, seed, 'shuffle'),
                               blocked=accuracy(inf, p, 256, seed, 'blocked')) for p in PARTITIONS}
    res['brevity'] = {'train': brevity(inf, rng, train), 'heldout': brevity(inf, rng, held)}
    forms = lexicon(inf, rng, train); words = {k: modal(c) for k, c in forms.items()}
    res['lexicon'] = lexicon_report(forms)
    res['combination'] = {}
    res['combination']['train'], rules = combination(inf, rng, train, words)
    for k, name in enumerate(PARTITIONS):
        if name != 'train': res['combination'][name], _ = combination(inf, rng, np.nonzero(part == k)[0], words, 800, rules)
    res['roles'] = {'train': roles(inf, rng, train, words), 'heldout': roles(inf, rng, held, words)}
    res['entanglement'] = entanglement(inf, train)
    if fresh: res['newcomer'] = newcomer(inf, seed=seed)
    inv = defaultdict(set)
    for (val, slot, v), f in words.items():
        if f: inv[f].add(value_name(val, slot, v))
    ex = []
    for pool, k in ((train, 1), (train, 2), (train, 3), (held, 2), (held, 3)):
        items = contexts_for(world, rng, pool, k, 4); said = inf.say([c for c, _ in items])
        for (ctx, slots), m in zip(items, said):
            ex.append(dict(target=world.describe(ctx[0]), partition=PARTITIONS[part[ctx[0]]], rooms_differ_in=list(slots),
                           others=[world.describe(j) for j in ctx[1:]], utterance=' '.join(map(str, m)), gloss=gloss(inv, m)))
    res['examples'] = ex
    (Path(run_dir) / 'analysis.json').write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return res


def brief(r):
    f = lambda x: 'nan' if x is None else f'{x:.2f}'
    a, lx, cb, bv = r['accuracy'], r['lexicon'], r['combination'], r['brevity']['train']
    hc = lambda key: mean(cb[p].get(key) for p in TEST_PARTITIONS)
    s = (f"acc {a['train']['normal']:.2f}/{mean(a[p]['normal'] for p in TEST_PARTITIONS):.2f} | len k1,2,3 {f(bv['1']['length'])},{f(bv['2']['length'])},{f(bv['3']['length'])}"
         f" | word cons {f(lx['word_consistency'])} len {f(lx['mean_word_length'])} dist {f(lx['values_distinguished'])}"
         f" shareAP {f(lx['type_word_shared_A_P'])} SA {f(lx['type_word_shared_S_A'])} | concat {f(cb['train']['concatenative'])} bag {f(cb['train']['same_bag'])}"
         f" order {f(cb['train']['order_consistency'])} | held concat {f(hc('concatenative'))} synth-understood {f(hc('synthesis_understood'))}"
         f" (own {f(hc('speaker_understood'))}) cov {f(hc('synthesis_coverage'))} | swap acc {f(r['roles']['train']['accuracy'])}")
    if r.get('newcomer'): s += f" | newc150 {r['newcomer']['150']['train']:.2f}"
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--no-fresh', action='store_true'); a = p.parse_args()
    r = analyse(a.run, fresh=not a.no_fresh); print(brief(r))
    for e in r['examples'][:14]: print(f"  {e['partition'][:6]:6s} {e['target']:36s} differ:{','.join(e['rooms_differ_in']):24s} -> {e['utterance']:12s} {e['gloss']}")
