"""Training with switchable *environmental* pressures.

There is no auxiliary objective anywhere in this file: no reconstruction term,
no disentanglement or invariance term, no role or position labels.  The only
quantity being optimised is the team's task outcome under the conditions the
world imposes.  Those conditions are the experimental variables:

    cost      every symbol takes one tick, and ticks are charged to the team
    hazard    urgency: after every symbol the danger may arrive (probability h)
              and the collector must act on the part of the message heard so far
    turnover  after a founding period, both agents are replaced by newcomers
              every `life` updates.
              A newcomer speaker first overhears a limited sample of the old
              speaker's utterances about scenes it can also see (`bottleneck`
              events), then both newcomers learn the task by doing it
    context   whether the speaker sees the public rooms it is talking about
              (shared context), and how many respects they differ in (`focus`)
    vocab     size of the signal inventory (articulatory bottleneck)
    n_cand    how many confusable alternatives the collector faces

Two learners are available.  `rf` (default) uses no cross-agent gradient at
all: the speaker does REINFORCE on the binary team outcome and the listener
learns from the revealed outcome of each episode.  `gs` passes a
straight-through gradient through the discrete symbols, as in the earlier
ALKoG recurrent-Gumbel arm; it is kept for comparison and learns poorly here.
"""
from pathlib import Path
import argparse, json, math, time
import torch
from torch import nn
import torch.nn.functional as F
from .world import World
from .agents import Speaker, Listener


def decision_weights(alive, hazard):
    """Probability that the collector acts having heard exactly j+1 symbols.
    alive[:, j] = 1 if symbols 0..j are all payload.  After each symbol the
    danger arrives with probability `hazard`; otherwise the collector acts when
    the speaker stops (or the maximum length is reached)."""
    nxt = torch.cat((alive[:, 1:], torch.zeros_like(alive[:, :1])), 1)
    survive = (1 - hazard) ** torch.arange(alive.shape[1], device=alive.device)
    return survive * (hazard * alive + (1 - hazard) * (alive - nxt))


def gs_step(speaker, listener, batch, n_cand, cost, hazard, tau, train_speaker=True):
    if train_speaker: wire, alive = speaker.relaxed(batch['private'], tau)
    else:
        with torch.no_grad():
            tok = speaker(batch['private'], mode='sample')['tokens']
            wire = F.one_hot(tok, speaker.vocab + 1).float(); alive = (tok != speaker.eos).float().cumprod(1)
    logits = listener.anytime_wire(batch['candidates'], wire)
    ce = F.cross_entropy(logits.flatten(0, 1), batch['target'].repeat_interleave(logits.shape[1]), reduction='none').reshape(alive.shape)
    w = decision_weights(alive, hazard)
    loss = (w * ce).sum(1) + (1 - alive[:, 0]) * math.log(n_cand) + cost * alive.sum(1)
    with torch.no_grad():
        hit = (logits.argmax(-1) == batch['target'][:, None]).float()
        success = (w * hit).sum(1) + (1 - alive[:, 0]) / n_cand
    return loss.mean(), float(success.mean()), float(alive.sum(1).mean())


def rf_step(speaker, listener, batch, n_cand, cost, hazard, entropy, baseline, gen, context):
    out = speaker(batch['private'], batch['candidates'] if context else None, mode='sample')
    every, mask = listener.anytime(batch['candidates'], out['tokens'])
    heard = mask.sum(1)
    if hazard > 0: heard = torch.minimum(heard, torch.empty_like(heard).geometric_(hazard, generator=gen))
    last = (heard.long() - 1).clamp_min(0)
    logits = every[torch.arange(len(last), device=last.device), last]
    correct = torch.where(heard > 0, (logits.argmax(-1) == batch['target']).float(), torch.full_like(heard, 1 / n_cand))
    reward = correct - cost * out['length']
    loss_s = -((reward - baseline) * out['logp']).mean() - entropy * out['entropy'].mean()
    ce = F.cross_entropy(every.flatten(0, 1), batch['target'].repeat_interleave(every.shape[1]), reduction='none').reshape(mask.shape)
    loss_l = (ce * mask).sum() / mask.sum().clamp_min(1)
    return loss_s, loss_l, float(correct.mean()), float(out['length'].mean()), float(reward.mean())


def run(output, seed=0, updates=6000, sets=64, n_cand=4, vocab=6, max_len=6, width=64, learner='rf', context=True,
        cost=0.0, hazard=0.0, generations=0, life=500, bottleneck=250, init=None, imitation=150, childhood=100,
        lr=1e-3, tau=1.0, entropy=0.05, focus=(1, 2, 3), curriculum=False, device='cpu', log=250):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(seed)
    world = World(device); gen = torch.Generator(device=device).manual_seed(1000 + seed)
    make = lambda: (Speaker(vocab, max_len, width).to(device), Listener(vocab, width).to(device))
    speaker, listener = make()
    opt = lambda *m: torch.optim.Adam([p for x in m for p in x.parameters()], lr=lr)
    opt_s, opt_l = opt(speaker), opt(listener)
    # `updates` founding updates (generation 0), then `generations` lifetimes of `life` updates each
    founding = updates
    if init:        # start from an existing founder population instead of training one
        cp = torch.load(Path(init) / 'checkpoint.pt', map_location=device, weights_only=False)
        speaker.load_state_dict(cp['speaker']); listener.load_state_dict(cp['listener']); founding = 0
    total = founding + generations * life
    curve = []; baseline = 1 / n_cand; start = time.perf_counter(); heard_examples = 0
    train_ids = world.pools['train']
    for step in range(total):
        g = 0 if step < founding else 1 + (step - founding) // life
        age = step if g == 0 else (step - founding) % life
        span = founding if g == 0 else life
        if g > 0 and age == 0:
            # ---- a new generation.  The old speaker's utterances for a limited
            # sample of events are all the newcomer ever hears from it.
            with torch.no_grad():
                seen = train_ids[torch.randperm(len(train_ids), generator=gen, device=device)[:bottleneck]]
                heard = speaker(world.features[seen])['tokens']
            torch.save(dict(speaker=speaker.state_dict(), listener=listener.state_dict()), output / f'generation{g - 1}.pt')
            speaker, listener = make(); opt_s, opt_l = opt(speaker), opt(listener)
            for _ in range(imitation):
                pick = torch.randint(len(seen), (128,), generator=gen, device=device)
                loss = speaker.imitate(world.features[seen[pick]], heard[pick])
                opt_s.zero_grad(); loss.backward(); opt_s.step(); heard_examples += 128
        lam = cost * min(1., age / (0.3 * span)) if g == 0 else cost
        # simple scenes first: early on the rooms differ in one respect only, later in up to three
        now = focus if not curriculum or g > 0 else tuple(k for k in focus if k <= 1 + (3 * step) // founding) or focus[:1]
        batch = world.sample(sets, 'train', gen, n_cand, now)
        child = g > 0 and age < childhood           # newcomer listener catches up first
        if learner == 'gs':
            loss, success, length = gs_step(speaker, listener, batch, n_cand, lam, hazard, tau, train_speaker=not child)
            opt_s.zero_grad(); opt_l.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(speaker.parameters()) + list(listener.parameters()), 1.)
            opt_l.step()
            if not child: opt_s.step()
        else:
            # exploration (babbling) fades with age; heirs of a language babble less than its founders
            ent = entropy * (1. if g == 0 else 0.25) * max(0., 1. - age / (0.7 * span))
            loss_s, loss_l, success, length, r = rf_step(speaker, listener, batch, n_cand, lam, hazard, ent, baseline, gen, context)
            if not child:
                opt_s.zero_grad(); loss_s.backward(); nn.utils.clip_grad_norm_(speaker.parameters(), 1.); opt_s.step()
            opt_l.zero_grad(); loss_l.backward(); nn.utils.clip_grad_norm_(listener.parameters(), 1.); opt_l.step()
            baseline = 0.95 * baseline + 0.05 * r
        if (step + 1) % log == 0 or step == total - 1:
            with torch.no_grad():
                ev = world.sample(256, 'train', gen, n_cand, focus); out = speaker(ev['private'], ev['candidates'] if context else None)
                acc = float((listener(ev['candidates'], out['tokens']).argmax(-1) == ev['target']).float().mean())
            row = dict(update=step + 1, generation=g, sampled_success=success, greedy=acc, length=float(out['length'].mean()),
                       cost=lam, seconds=time.perf_counter() - start)
            curve.append(row)
            if (step + 1) % (log * 4) == 0: print(json.dumps(row), flush=True)
    config = dict(seed=seed, updates=updates, sets=sets, n_cand=n_cand, vocab=vocab, max_len=max_len, width=width,
                  learner=learner, context=context, focus=list(focus), curriculum=curriculum, cost=cost, hazard=hazard, generations=generations, life=life, bottleneck=bottleneck, init=str(init) if init else None,
                  imitation=imitation, childhood=childhood, lr=lr, tau=tau, entropy=entropy,
                  game_episodes=total * sets * n_cand, overheard_utterances=heard_examples,
                  seconds=time.perf_counter() - start, torch=torch.__version__,
                  auxiliary_losses='none', semantic_slots='none', role_labels='none')
    torch.save(dict(speaker=speaker.state_dict(), listener=listener.state_dict(), config=config), output / 'checkpoint.pt')
    (output / 'config.json').write_text(json.dumps(config, indent=2)); (output / 'curve.json').write_text(json.dumps(curve))
    return speaker, listener, world, config


def load(path, device='cpu'):
    cp = torch.load(Path(path) / 'checkpoint.pt', map_location=device, weights_only=False); c = cp['config']
    speaker = Speaker(c['vocab'], c['max_len'], c['width']).to(device); listener = Listener(c['vocab'], c['width']).to(device)
    speaker.load_state_dict(cp['speaker']); listener.load_state_dict(cp['listener'])
    return speaker.eval(), listener.eval(), World(device), c


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--updates', type=int, default=6000); p.add_argument('--vocab', type=int, default=6)
    p.add_argument('--cands', type=int, default=4); p.add_argument('--max-len', type=int, default=6)
    p.add_argument('--blind', action='store_true', help='speaker does not see the public rooms')
    p.add_argument('--focus', type=int, nargs='+', default=[1, 2, 3], help='how many respects the rooms may differ in')
    p.add_argument('--curriculum', action='store_true', help='rooms differ in one respect at first, later in up to three')
    p.add_argument('--cost', type=float, default=0.); p.add_argument('--hazard', type=float, default=0.)
    p.add_argument('--generations', type=int, default=0); p.add_argument('--life', type=int, default=500); p.add_argument('--init', default=None); p.add_argument('--bottleneck', type=int, default=250)
    p.add_argument('--imitation', type=int, default=150); p.add_argument('--childhood', type=int, default=100)
    p.add_argument('--learner', choices=('gs', 'rf'), default='rf'); p.add_argument('--width', type=int, default=64)
    p.add_argument('--entropy', type=float, default=0.05); p.add_argument('--sets', type=int, default=64); p.add_argument('--device', default='cpu')
    a = p.parse_args()
    run(a.output, a.seed, a.updates, sets=a.sets, n_cand=a.cands, vocab=a.vocab, max_len=a.max_len, width=a.width, learner=a.learner, context=not a.blind,
        cost=a.cost, hazard=a.hazard, generations=a.generations, life=a.life, init=a.init, bottleneck=a.bottleneck, imitation=a.imitation,
        childhood=a.childhood, entropy=a.entropy, curriculum=a.curriculum, focus=tuple(a.focus), device=a.device)
