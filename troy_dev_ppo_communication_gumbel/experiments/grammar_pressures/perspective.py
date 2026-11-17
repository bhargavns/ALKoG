"""Wanting versus reporting: a first step toward speakers that talk about themselves.

Same event world, same reward-only agents, one new fact about the speaker:
it has an inner state.  Every episode it is in one of two moods.

    report   it has seen an event (as in every other game in this study)
             and the collector should go and LOOK at that room
    want     it wants something with a given property ("a lion",
             "something blue", "something being chased") and the collector
             should FETCH the one room that has it

The collector picks a room and a deed (look / fetch); the team is paid only if
both are right.  The rooms always differ in exactly one respect, in both
moods, so the same property is at stake whether it is seen or wanted.  The
mood is felt by the speaker (an input that is not a percept of the world);
the collector can only learn it from the message.

Nothing tells the agents to mark the mood with its own piece of signal.  A
holistic code (one arbitrary form per mood x property) would do as well on the
trained combinations, provided the inventory allows it.  What is measured:

    marker        is there a symbol whose presence separates want from report,
                  whatever is wanted?  where does it sit (first / last)?
    reuse         with the marker removed, is what is said about a property
                  the same whether it is seen or wanted?
    productivity  some properties are NEVER wanted in training (only seen).
                  Can the speaker express wanting them, and does the
                  collector understand a message built from the trained
                  marker plus the trained report word?  Only a language with
                  a separable mood marker gets these for free.

    python -m experiments.grammar_pressures.perspective --output results/grammar_pressures/perspective/p24_0 --vocab 24 --seed 0
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .world import World, NT, NC, NP1, ELEMENT_DIM, TYPES, COLOURS, PRED1, PRED2, PARTITIONS
from .agents import Speaker, Actor, strip, pack

REPORT, WANT = 0, 1
KIND = dict(pred='pred', S_type='type', A_type='type', P_type='type', S_colour='colour', A_colour='colour', P_colour='colour')
# properties never wanted in training (they are still seen and reported): (kind, valency or 0, value)
HELD = {('colour', 0, 3), ('type', 0, 5), ('pred', 2, 3), ('pred', 1, 3)}      # blue, tiger, 'hides', 'gone'


def prop(world, i, slot):
    """(kind, valency for predicates else 0, value) of event i's slot."""
    k = KIND[slot]; v = int(world.slots(i)[slot])
    return (k, int(world.events[i, 0]) if k == 'pred' else 0, v)


def name(p):
    k, val, v = p
    return (PRED1 if val == 1 else PRED2)[v] if k == 'pred' else (TYPES if k == 'type' else COLOURS)[v]


def mask(slot):
    """Which features of a percept carry one slot (the part of a scene a wish is about)."""
    m = torch.zeros(3, ELEMENT_DIM)
    if slot == 'pred': m[0] = 1; return m
    el = 2 if slot.startswith('P') else 1
    if slot.endswith('type'): m[el, :NT] = 1
    else: m[el, NT:NT + NC] = 1
    m[el, NT + NC:NT + NC + 3] = 1          # its role in the event
    return m


class Game:
    def __init__(self, device='cpu', n_cand=4):
        self.w = World(device); self.n_cand = n_cand; self.device = device
        self.masks = {s: mask(s).to(device) for s in KIND}

    def context(self, rng, i, slot):
        """Event i plus n_cand-1 rooms that differ from it only in `slot`."""
        w = self.w; cur = w.slots(i)[slot]; out = [i]
        for v in rng.permutation(w.slot_sizes(i)[slot]):
            if v == cur: continue
            j = w.edit(i, slot, int(v))
            if j is not None and w.partition[j] == w.partition[i]: out.append(j)
            if len(out) == self.n_cand: return out
        return None

    def episodes(self, rng, n, partition='train', moods=(REPORT, WANT), held=None):
        """n episodes; `held`: None = any property, False = never a held-out
        property in want mood (training), True = only held-out properties wanted."""
        w = self.w; pool = np.nonzero(w.partition == PARTITIONS.index(partition))[0]
        ids, slots, mood = [], [], []
        while len(ids) < n:
            i = int(rng.choice(pool)); slot = str(rng.choice(list(w.slot_sizes(i)))); m = int(rng.choice(moods))
            if m == WANT and held is not None and (prop(w, i, slot) in HELD) != held: continue
            ctx = self.context(rng, i, slot)
            if ctx is None: continue
            ids.append(ctx); slots.append(slot); mood.append(m)
        return self.batch(rng, ids, slots, mood)

    def batch(self, rng, ids, slots, mood):
        w = self.w; dev = self.device
        ids_t = torch.tensor(ids, device=dev); n = len(ids)
        order = torch.tensor(np.array([rng.permutation(self.n_cand) for _ in range(n)]), device=dev)
        rooms = ids_t.gather(1, order); target = (order == 0).float().argmax(1)
        private = w.features[ids_t[:, 0]].clone(); mood_t = torch.tensor(mood, device=dev)
        for k, (s, m) in enumerate(zip(slots, mood)):
            if m == WANT: private[k] = private[k] * self.masks[s]          # a wish is only about the one property
        inner = F.one_hot(mood_t, 2).float()
        return dict(private=private, candidates=w.features[rooms], target=target, mood=mood_t, inner=inner,
                    ids=ids, slots=slots, rooms=rooms)


def step(speaker, actor, b, cost, entropy, baseline):
    out = speaker(b['private'], b['candidates'], mode='sample', inner=b['inner'])
    room, deed = actor(b['candidates'], out['tokens'])
    win = ((room.argmax(-1) == b['target']) & (deed.argmax(-1) == b['mood'])).float()
    reward = win - cost * out['length']
    loss_s = -((reward - baseline) * out['logp']).mean() - entropy * out['entropy'].mean()
    loss_a = F.cross_entropy(room, b['target']) + F.cross_entropy(deed, b['mood'])
    return loss_s, loss_a, float(win.mean()), float(out['length'].mean()), float(reward.mean())


def run(output, seed=0, updates=12000, batch=256, vocab=24, max_len=6, width=64, cost=0.01, entropy=0.1, entropy_until=0.9,
        lr=1e-3, device='cpu', log=250):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    game = Game(device); speaker = Speaker(vocab, max_len, width, inner=2).to(device); actor = Actor(vocab, width).to(device)
    opt_s = torch.optim.Adam(speaker.parameters(), lr=lr); opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    baseline = 0.; curve = []; start = time.perf_counter()
    for t in range(updates):
        b = game.episodes(rng, batch, held=False)
        ent = entropy * max(0., 1. - t / (entropy_until * updates)); lam = cost * min(1., t / (0.3 * updates))
        loss_s, loss_a, win, length, r = step(speaker, actor, b, lam, ent, baseline)
        opt_s.zero_grad(); loss_s.backward(); nn.utils.clip_grad_norm_(speaker.parameters(), 1.); opt_s.step()
        opt_a.zero_grad(); loss_a.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.); opt_a.step()
        baseline = 0.95 * baseline + 0.05 * r
        if (t + 1) % log == 0:
            row = dict(update=t + 1, win=win, length=length, seconds=time.perf_counter() - start); curve.append(row)
            if (t + 1) % (4 * log) == 0: print(json.dumps(row), flush=True)
    config = dict(seed=seed, updates=updates, batch=batch, vocab=vocab, max_len=max_len, width=width, cost=cost, entropy=entropy,
                  entropy_until=entropy_until, lr=lr, held=sorted(map(list, HELD)), auxiliary_losses='none')
    torch.save(dict(speaker=speaker.state_dict(), actor=actor.state_dict(), config=config), output / 'checkpoint.pt')
    (output / 'config.json').write_text(json.dumps(config, indent=1)); (output / 'curve.json').write_text(json.dumps(curve))
    return output


# ------------------------------------------------------------------ analysis
@torch.no_grad()
def say(speaker, b): return strip(speaker(b['private'], b['candidates'], inner=b['inner'])['tokens'], speaker.eos)


@torch.no_grad()
def judge(actor, b, messages, max_len):
    room, deed = actor(b['candidates'], pack(messages, actor.eos, max_len, b['candidates'].device))
    r = (room.argmax(-1) == b['target']); d = (deed.argmax(-1) == b['mood'])
    return (r & d).float().mean().item(), r.float().mean().item(), d.float().mean().item()


def analyse(run_dir, n=3000):
    run_dir = Path(run_dir); cp = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False); c = cp['config']
    speaker = Speaker(c['vocab'], c['max_len'], c['width'], inner=2); speaker.load_state_dict(cp['speaker']); speaker.eval()
    actor = Actor(c['vocab'], c['width']); actor.load_state_dict(cp['actor']); actor.eval()
    game = Game(); w = game.w; rng = np.random.default_rng(1000 + c['seed']); res = dict(config=c)

    # matched pairs: the same rooms and the same property, once seen and once wanted (trained properties only)
    b = game.episodes(rng, n, held=False)
    pair = game.batch(rng, b['ids'], b['slots'], [1 - int(m) for m in b['mood'].tolist()])
    said, said2 = say(speaker, b), say(speaker, pair)
    res['accuracy'] = dict(both=judge(actor, b, said, c['max_len'])[0], room=judge(actor, b, said, c['max_len'])[1], deed=judge(actor, b, said, c['max_len'])[2])
    want = [m for m, k in zip(said, b['mood'].tolist()) if k == WANT] + [m for m, k in zip(said2, pair['mood'].tolist()) if k == WANT]
    rep = [m for m, k in zip(said, b['mood'].tolist()) if k == REPORT] + [m for m, k in zip(said2, pair['mood'].tolist()) if k == REPORT]
    pw = Counter(s for m in want for s in set(m)); pr = Counter(s for m in rep for s in set(m))
    gap = {s: pw[s] / len(want) - pr[s] / len(rep) for s in range(c['vocab'])}
    wm = max(gap, key=gap.get); rm = min(gap, key=gap.get)
    res['marker'] = dict(want_marker=wm, in_want=pw[wm] / len(want), in_report=pr[wm] / len(rep),
                         report_marker=rm, report_marker_in_report=pr[rm] / len(rep), report_marker_in_want=pw[rm] / len(want),
                         want_marker_first=np.mean([m[:1] == (wm,) for m in want if wm in m]) if pw[wm] else None,
                         want_marker_last=np.mean([m[-1:] == (wm,) for m in want if wm in m]) if pw[wm] else None,
                         length_want=float(np.mean([len(m) for m in want])), length_report=float(np.mean([len(m) for m in rep])))
    strip_m = lambda m: tuple(s for s in m if s not in (wm, rm))
    same = [strip_m(a) == strip_m(x) for a, x in zip(said, said2)]
    res['reuse'] = dict(identical_after_marker_removed=float(np.mean(same)),
                        identical_raw=float(np.mean([a == x for a, x in zip(said, said2)])))
    # property words: modal form (marker removed) per property and mood
    forms = defaultdict(Counter)
    for bb, ss in ((b, said), (pair, said2)):
        for k, (i, slot, m) in enumerate(zip(bb['ids'], bb['slots'], ss)):
            forms[(prop(w, i[0], slot), int(bb['mood'][k]))][strip_m(m)] += 1
    props = sorted({p for p, _ in forms})
    word = lambda p, m: forms[(p, m)].most_common(1)[0][0] if (p, m) in forms else None
    # a property word must be non-empty to count as shared across moods
    shared = [word(p, REPORT) == word(p, WANT) for p in props if word(p, REPORT) and word(p, WANT) is not None]
    rw = [word(p, REPORT) for p in props if word(p, REPORT) is not None]
    res['reuse']['report_words_distinct'] = len({x for x in rw if x}) / max(1, len(rw))     # 1 = every property has its own non-empty word
    res['reuse']['property_word_same_in_both_moods'] = float(np.mean(shared)) if shared else None
    res['reuse']['property_word_consistency'] = float(np.mean([c_.most_common(1)[0][1] / sum(c_.values()) for c_ in forms.values()]))
    order = Counter('first' if m[:1] == (wm,) else 'last' if m[-1:] == (wm,) else 'inside' if wm in m else 'none' for m in want)
    res['marker']['positions'] = dict(order)

    # productivity: properties never wanted in training
    h = game.episodes(rng, 1500, moods=(WANT,), held=True)
    own = say(speaker, h)
    res['held_want'] = dict(zip(('both', 'room', 'deed'), judge(actor, h, own, c['max_len'])))
    # build: the report word for that property, with the want marker where it usually goes (removing any report marker)
    hr = game.batch(rng, h['ids'], h['slots'], [REPORT] * len(h['ids'])); rep_said = say(speaker, hr)
    pos = order.most_common(1)[0][0] if order else 'first'
    build = [((wm,) + strip_m(m)) if pos != 'last' else (strip_m(m) + (wm,)) for m in rep_said]
    res['held_want']['built'] = dict(zip(('both', 'room', 'deed'), judge(actor, h, build, c['max_len'])))
    res['held_want']['speaker_uses_marker'] = float(np.mean([wm in m for m in own]))
    res['held_want']['speaker_says_built'] = float(np.mean([a == x for a, x in zip(own, build)]))
    res['examples'] = [dict(event=w.describe(i[0]), about=slot, property=name(prop(w, i[0], slot)),
                            report=' '.join(map(str, a)), want=' '.join(map(str, x)) if mo == REPORT else None)
                       for i, slot, a, x, mo in list(zip(b['ids'], b['slots'], said, said2, b['mood'].tolist()))[:40] if mo == REPORT][:16]
    (run_dir / 'analysis.json').write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return res


def brief(r):
    m, u, h = r['marker'], r['reuse'], r['held_want']; f = lambda x: '  - ' if x is None else f'{x:.2f}'
    return (f"win {r['accuracy']['both']:.2f} (room {r['accuracy']['room']:.2f} deed {r['accuracy']['deed']:.2f}) | "
            f"want-marker {m['want_marker']} in want {m['in_want']:.2f} / report {m['in_report']:.2f}, pos {m['positions']} | "
            f"len want {m['length_want']:.2f} report {m['length_report']:.2f} | same-word {f(u['property_word_same_in_both_moods'])} "
            f"cons {f(u['property_word_consistency'])} distinct {f(u.get('report_words_distinct'))} pairs-identical {f(u['identical_after_marker_removed'])} | "
            f"HELD own {h['both']:.2f} (deed {h['deed']:.2f}) built {h['built']['both']:.2f} uses-marker {h['speaker_uses_marker']:.2f}")


def summarise(root):
    """One line per run and the pass line from the README."""
    for p in sorted(Path(root).glob('*/analysis.json')):
        r = json.loads(p.read_text())
        if 'marker' not in r: continue
        m, u, h = r['marker'], r['reuse'], r['held_want']
        ok = ((u.get('report_words_distinct') or 0) >= 0.8 and m['in_want'] >= 0.9 and m['in_report'] <= 0.1 and (u['property_word_same_in_both_moods'] or 0) >= 0.7 and h['built']['both'] >= 0.7)
        print(f"{p.parent.name:14s} {'PASS' if ok else 'fail'}  {brief(r)}")


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--summarise']: summarise(sys.argv[2]); sys.exit()
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=12000); p.add_argument('--vocab', type=int, default=24); p.add_argument('--max-len', type=int, default=6)
    p.add_argument('--cost', type=float, default=0.01); p.add_argument('--batch', type=int, default=256)
    p.add_argument('--analyse-only', action='store_true'); a = p.parse_args()
    if not a.analyse_only: run(a.output, a.seed, a.updates, a.batch, a.vocab, a.max_len, cost=a.cost)
    r = analyse(a.output); print(brief(r))
    for e in r['examples'][:10]: print(f"  {e['event']:36s} about {e['about']:9s} {e['property']:9s} seen: {e['report']:12s} wanted: {e['want']}")
