"""A world with consequences: the event world, but lions bite and food feeds.

Until now a lion was a one-hot column with no effect on anything.  Here:

    predators   lion, tiger.  Every other type is food.
    danger      a room is dangerous if a predator is free in it:
                  loose(predator); a predator agent of guards / chases / near;
                  a predator patient of near (nearness is mutual)
    tame        a predator present but not free: caged(predator), a predator
                agent of hides, a predator patient of guards / chases / hides
    food        available food: ripe(food); a food participant of a transitive
                event unless it is the patient of guards or hides
    colour      never changes an outcome

The collector has a private need, felt only by itself:

    hungry      +1 for entering a room with available food
    keeper      +1 for entering a room with a tame predator
    either      -1 for entering a dangerous room (whatever else is in it);
                staying home is 0; each symbol costs the team `cost`

The speaker has been in all four rooms and perceives them fully, as an
unordered set (it cannot say "the second room").  It does not know the need.
The collector sees the rooms through fog: who is there (type, colour, role),
not what is happening (the predicate is hidden).  So to be useful a message
must carry hidden states tied to visible identities.

    python -m experiments.grammar_pressures.consequences --output results/grammar_pressures/consequences/cons2_0 --needs 2 --seed 0
    python -m experiments.grammar_pressures.consequences --analyse results/grammar_pressures/consequences/cons2_0
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .world import World, NT, NC, TYPES, COLOURS, PRED1, PRED2, ELEMENT_DIM
from .agents import Speaker, SceneEncoder, strip, pack

PREDATORS = {4, 5}
LOOSE, CAGED, RIPE, GONE = range(4)
GUARDS, CHASES, NEAR, HIDES = range(4)
HUNGRY, KEEPER = 0, 1
NEEDS = ('hungry', 'keeper')
DANGER, FOOD, TAME = 0, 1, 2
STATUS = ('danger', 'food', 'tame')


def flags(v, p, a, b):
    """(danger, food, tame) of one event."""
    ta, tb = a // NC, (b // NC if b >= 0 else -1)
    pa, pb = ta in PREDATORS, tb in PREDATORS
    if v == 1:
        if pa: return (p == LOOSE, False, p == CAGED)
        return (False, p == RIPE, False)
    danger = (pa and p in (GUARDS, CHASES, NEAR)) or (pb and p == NEAR)
    tame = not danger and ((pa and p == HIDES) or (pb and p in (GUARDS, CHASES, HIDES)))
    food = (not pa) or (not pb and p not in (GUARDS, HIDES))
    return (bool(danger), bool(food), bool(tame))


class Consequences:
    def __init__(self, device='cpu', valency1=0.5):
        self.w = w = World(device); self.device = device; self.valency1 = valency1
        self.flags = torch.tensor([flags(*map(int, e)) for e in w.events], dtype=torch.bool, device=device)   # [events, 3]
        d, f, t = self.flags.float().T
        self.value = torch.stack((torch.where(d > 0, -1., f), torch.where(d > 0, -1., t)), 1)               # [events, needs]
        ev = w.events
        # what the collector can see of an event: valency, participants, roles (not the predicate)
        sig = {}; self.sig = torch.tensor([sig.setdefault((int(v), int(a), int(b)), len(sig)) for v, _, a, b in ev], device=device)
        fog = w.features.clone(); fog[:, 0] = 0; self.fog = fog
        train = np.nonzero(w.partition == 0)[0]
        self.pool = [torch.tensor(train[ev[train, 0] == k], device=device) for k in (1, 2)]

    def rooms(self, gen, n, k=4):
        """n scenes of k train events with pairwise different visible signatures."""
        dev = self.device
        def draw(m):
            one = torch.rand(m, k, generator=gen, device=dev) < self.valency1
            i1 = self.pool[0][torch.randint(len(self.pool[0]), (m, k), generator=gen, device=dev)]
            i2 = self.pool[1][torch.randint(len(self.pool[1]), (m, k), generator=gen, device=dev)]
            return torch.where(one, i1, i2)
        ids = draw(n)
        while True:
            s = self.sig[ids]; bad = (s[:, :, None] == s[:, None, :]).sum((1, 2)) > k
            if not bad.any(): return ids
            ids[bad] = draw(int(bad.sum()))

    def batch(self, ids, need):
        """ids [n, k] room events; need [n].  Options: the k rooms then 'stay home'."""
        v = self.value[ids, need[:, None]]                                    # [n, k]
        options = torch.cat((v, torch.zeros(len(ids), 1, device=self.device)), 1)
        return dict(ids=ids, need=need, full=self.w.features[ids], fog=self.fog[ids], options=options)


class RoomsSpeaker(nn.Module):
    """A speaker that perceives a whole set of rooms (order-free) and has no need of its own."""
    def __init__(self, vocab, max_len, width=64):
        super().__init__()
        self.talk = Speaker(vocab, max_len, width)
        self.room = SceneEncoder(width)
        self.mix = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh())
        self.vocab, self.eos, self.max_len = vocab, vocab, max_len

    def forward(self, rooms, mode='greedy', forced=None):
        r = self.room(rooms)                                                   # [n, k, width]
        ctx = self.mix(torch.cat((r.sum(1), r.max(1).values), -1))
        return self.talk(None, mode=mode, forced=forced, ctx=ctx)


class Chooser(nn.Module):
    """The collector: hears the message, feels its need, sees fogged rooms,
    and scores every room plus 'stay home'."""
    def __init__(self, vocab, width=64, emb=32, needs=2):
        super().__init__()
        self.vocab, self.eos = vocab, vocab
        self.embedding = nn.Embedding(vocab + 1, emb); self.gru = nn.GRU(emb, width, batch_first=True)
        self.need = nn.Embedding(needs, width)
        self.h = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, width))
        self.scenes = SceneEncoder(width); self.stay = nn.Linear(width, 1)

    def message(self, tokens):
        length = (tokens != self.eos).long().cumprod(1).sum(1)
        out, _ = self.gru(self.embedding(tokens))
        last = out[torch.arange(len(tokens), device=tokens.device), (length - 1).clamp_min(0)]
        return last * (length > 0).float()[:, None]

    def forward(self, fog, tokens, need):
        h = self.h(torch.cat((self.message(tokens), self.need(need)), -1))
        rooms = torch.einsum('nd,nkd->nk', h, self.scenes(fog)) / h.shape[-1] ** 0.5
        return torch.cat((rooms, self.stay(h)), 1)


def run(output, seed=0, updates=12000, batch=256, vocab=12, max_len=8, width=64, needs=2, cost=0.01, entropy=0.1,
        entropy_until=0.9, lr=1e-3, silent=False, curriculum=False, rooms=0, soft=False, device='cpu', log=250):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(seed); gen = torch.Generator(device=device).manual_seed(1000 + seed)
    game = Consequences(device); s = RoomsSpeaker(vocab, max_len, width).to(device); c = Chooser(vocab, width).to(device)
    opt_s = torch.optim.Adam(s.parameters(), lr=lr); opt_c = torch.optim.Adam(c.parameters(), lr=lr)
    baseline = 0.; curve = []; start = time.perf_counter()
    for t in range(updates):
        need = torch.randint(needs, (batch,), generator=gen, device=device)
        # curriculum: one room at first (enter or stay), then two, three, four
        k = rooms if rooms else min(4, 1 + int(t / (0.15 * updates))) if curriculum else 4
        b = game.batch(game.rooms(gen, batch, k), need)
        out = s(b['full'], mode='sample')
        tok = torch.full_like(out['tokens'], vocab) if silent else out['tokens']
        logits = c(b['fog'], tok, need)
        got = b['options'].gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)
        length = torch.zeros_like(got) if silent else out['length']
        lam = cost * min(1., t / (0.3 * updates)); ent = entropy * max(0., 1. - t / (entropy_until * updates))
        # soft: the collector acts stochastically and the speaker is paid its expected payoff
        earned = (F.softmax(logits.detach(), -1) * b['options']).sum(1) if soft else got
        reward = earned - lam * length
        best = b['options'] == b['options'].max(1, keepdim=True).values
        loss_c = -(F.log_softmax(logits, -1) * best.float() / best.sum(1, keepdim=True)).sum(1).mean()
        opt_c.zero_grad(); loss_c.backward(); nn.utils.clip_grad_norm_(c.parameters(), 1.); opt_c.step()
        if not silent:
            loss_s = -((reward - baseline) * out['logp']).mean() - ent * out['entropy'].mean()
            opt_s.zero_grad(); loss_s.backward(); nn.utils.clip_grad_norm_(s.parameters(), 1.); opt_s.step()
        baseline = 0.95 * baseline + 0.05 * float(reward.mean())
        if (t + 1) % log == 0:
            row = dict(update=t + 1, payoff=float(got.mean()), oracle=float(b['options'].max(1).values.mean()), length=float(length.mean()),
                       seconds=time.perf_counter() - start)
            curve.append(row)
            if (t + 1) % (4 * log) == 0: print(json.dumps(row), flush=True)
    config = dict(seed=seed, updates=updates, batch=batch, vocab=vocab, max_len=max_len, width=width, needs=needs, cost=cost, entropy=entropy,
                  entropy_until=entropy_until, lr=lr, silent=silent, curriculum=curriculum, rooms=rooms, soft=soft, auxiliary_losses='none')
    torch.save(dict(speaker=s.state_dict(), chooser=c.state_dict(), config=config), output / 'checkpoint.pt')
    (output / 'config.json').write_text(json.dumps(config, indent=1)); (output / 'curve.json').write_text(json.dumps(curve))
    return output


# ------------------------------------------------------------------ analysis
def load_pair(run):
    cp = torch.load(Path(run) / 'checkpoint.pt', map_location='cpu', weights_only=False); c = cp['config']
    s = RoomsSpeaker(c['vocab'], c['max_len'], c['width']); s.load_state_dict(cp['speaker']); s.eval()
    ch = Chooser(c['vocab'], c['width']); ch.load_state_dict(cp['chooser']); ch.eval()
    return s, ch, c


@torch.no_grad()
def say(s, game, ids):
    return s(game.w.features[ids])['tokens']


@torch.no_grad()
def payoff(ch, game, ids, need, tok):
    b = game.batch(ids, need)
    return float(b['options'].gather(1, ch(b['fog'], tok, need).argmax(-1, keepdim=True)).mean())


def name_event(w, i): return w.describe(int(i))


@torch.no_grad()
def analyse(run, n=4000, silent_run=None):
    run = Path(run); s, ch, c = load_pair(run); game = Consequences(); w = game.w
    gen = torch.Generator().manual_seed(7); res = dict(config=c)
    ids = game.rooms(gen, n); tok = say(s, game, ids); eos = s.eos
    blocked = torch.full_like(tok, eos); shuffled = tok[torch.randperm(n, generator=gen)]
    res['payoff'] = {}
    for k, need_name in enumerate(NEEDS[:c['needs']] if c['needs'] > 1 else NEEDS[:1]):
        need = torch.full((n,), k, dtype=torch.long)
        oracle = float(game.batch(ids, need)['options'].max(1).values.mean())
        res['payoff'][need_name] = dict(normal=payoff(ch, game, ids, need, tok), blocked=payoff(ch, game, ids, need, blocked),
                                        shuffled=payoff(ch, game, ids, need, shuffled), oracle=oracle)
        if silent_run:
            ss, sch, _ = load_pair(silent_run)
            res['payoff'][need_name]['silent_control'] = payoff(sch, game, ids, need, blocked)
    msgs = strip(tok, eos); res['mean_length'] = float(np.mean([len(m) for m in msgs])); res['distinct_messages'] = len(set(msgs))

    # consequence sensitivity: change one room's predicate; does the message change?
    cons, incons = [], []; examples = []
    for r in range(min(n, 1500)):
        room = int(torch.randint(4, (1,), generator=gen)); i = int(ids[r, room]); v = int(w.events[i, 0])
        for p in range(4):
            if p == int(w.events[i, 1]): continue
            j = w.edit(i, 'pred', p)
            if j is None or w.partition[j] != 0: continue
            new = ids[r].clone(); new[room] = j
            changed = strip(say(s, game, new[None]), eos)[0] != msgs[r]
            consequential = not torch.equal(game.flags[i], game.flags[j])
            (cons if consequential else incons).append(changed)
            if len(examples) < 12 and consequential:
                examples.append(dict(before=name_event(w, i), after=name_event(w, j), message_before=list(msgs[r]),
                                     message_after=list(strip(say(s, game, new[None]), eos)[0])))
    res['sensitivity'] = dict(consequential=float(np.mean(cons)), inconsequential=float(np.mean(incons)), n_cons=len(cons), n_incons=len(incons))

    # status markers: one eventful room among three uneventful ones
    quiet = (~game.flags.any(1)).nonzero().squeeze(1); quiet = quiet[torch.tensor(w.partition[quiet.numpy()] == 0)]
    by = defaultdict(list)
    for st in range(3):
        cand = (game.flags[:, st] & (game.flags.sum(1) == 1)).nonzero().squeeze(1)
        cand = cand[torch.tensor(w.partition[cand.numpy()] == 0)]
        for _ in range(400):
            e = int(cand[torch.randint(len(cand), (1,), generator=gen)])
            for _ in range(20):
                others = quiet[torch.randint(len(quiet), (3,), generator=gen)]
                scene = torch.cat((torch.tensor([e]), others))
                if len(set(game.sig[scene].tolist())) == 4: break
            v, p, a, b = map(int, w.events[e])
            who = a // NC if (st == FOOD and a // NC not in PREDATORS) or (st != FOOD and a // NC in PREDATORS) else b // NC
            by[(st, TYPES[who])].append(strip(say(s, game, scene[None]), eos)[0])
    markers = {}
    for st in range(3):
        mine = [m for (k, _), ms in by.items() if k == st for m in ms]; rest = [m for (k, _), ms in by.items() if k != st for m in ms]
        best = max(range(c['vocab']), key=lambda x: np.mean([x in m for m in mine]) - np.mean([x in m for m in rest]))
        per_type = {t: float(np.mean([best in m for m in ms])) for (k, t), ms in by.items() if k == st}
        markers[STATUS[st]] = dict(symbol=best, in_status=float(np.mean([best in m for m in mine])), in_others=float(np.mean([best in m for m in rest])),
                                   min_over_types=min(per_type.values()), per_type=per_type,
                                   common=[(' '.join(map(str, m)), k) for m, k in Counter(mine).most_common(4)])
    res['markers'] = markers; res['edit_examples'] = examples
    (run / 'analysis.json').write_text(json.dumps(res, indent=1))
    return res


def passes(r, silent_payoff):
    pay = np.mean([v['normal'] for v in r['payoff'].values()])
    a = pay - silent_payoff >= 0.3
    b = r['sensitivity']['consequential'] >= 0.8 and r['sensitivity']['inconsequential'] <= 0.3
    cc = all(m['min_over_types'] >= 0.8 and m['in_others'] <= 0.2 for m in r['markers'].values())
    return dict(a_payoff=bool(a), b_sensitivity=bool(b), c_markers=bool(cc), all=bool(a and b and cc), payoff_gain=float(pay - silent_payoff))


def summarise(root):
    """Analyse every run in `root`, score cons* runs against the pass line, write summary.json."""
    root = Path(root); out = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('cons2_silent')):
        seed = run.name.rsplit('_', 1)[1]; silent = root / f'cons2_silent_{seed}'
        r = analyse(run, silent_run=silent if silent.exists() else None)
        sp = np.mean([v['silent_control'] for v in r['payoff'].values()]) if silent.exists() else float('nan')
        out[run.name] = dict(payoff=r['payoff'], length=r['mean_length'], distinct=r['distinct_messages'], sensitivity=r['sensitivity'],
                             markers={k: {x: v[x] for x in ('symbol', 'in_status', 'in_others', 'min_over_types')} for k, v in r['markers'].items()},
                             pass_line=passes(r, sp))
        pl = out[run.name]['pass_line']; pay = r['payoff']; se = r['sensitivity']; mk = r['markers']
        print(f"{run.name:10s} " + ' '.join(f"{k}: {v['normal']:.2f} (blk {v['blocked']:.2f} sil {v.get('silent_control', float('nan')):.2f} orc {v['oracle']:.2f})" for k, v in pay.items())
              + f" | len {r['mean_length']:.1f} | sens {se['consequential']:.2f}/{se['inconsequential']:.2f} | "
              + ' '.join(f"{k}:{m['symbol']}({m['min_over_types']:.2f}/{m['in_others']:.2f})" for k, m in mk.items())
              + f" | pass a{int(pl['a_payoff'])} b{int(pl['b_sensitivity'])} c{int(pl['c_markers'])}", flush=True)
    (root / 'summary.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output'); p.add_argument('--analyse'); p.add_argument('--silent-run'); p.add_argument('--summarise')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--updates', type=int, default=12000)
    p.add_argument('--vocab', type=int, default=12); p.add_argument('--max-len', type=int, default=8)
    p.add_argument('--needs', type=int, default=2); p.add_argument('--cost', type=float, default=0.01)
    p.add_argument('--silent', action='store_true'); p.add_argument('--device', default='cpu')
    p.add_argument('--entropy', type=float, default=0.1); p.add_argument('--curriculum', action='store_true')
    p.add_argument('--rooms', type=int, default=0, help='fixed number of rooms (diagnostics)')
    p.add_argument('--soft', action='store_true', help='speaker paid the expected payoff of a stochastic collector')
    a = p.parse_args()
    if a.summarise: summarise(a.summarise)
    elif a.analyse:
        r = analyse(a.analyse, silent_run=a.silent_run); print(json.dumps({k: r[k] for k in ('payoff', 'mean_length', 'sensitivity')}, indent=1))
        print(json.dumps({k: {x: v[x] for x in ('symbol', 'in_status', 'in_others', 'min_over_types', 'common')} for k, v in r['markers'].items()}, indent=1))
    else:
        run(a.output, a.seed, a.updates, vocab=a.vocab, max_len=a.max_len, needs=a.needs, cost=a.cost, silent=a.silent,
            entropy=a.entropy, curriculum=a.curriculum, rooms=a.rooms, soft=a.soft, device=a.device)
