"""Explanatory figures and animations of the event-world setup, drawn from real
trained checkpoints (nothing here is a mock-up).

    python -m experiments.grammar_pressures.figures --out results/grammar_pressures/figures

percept.png         what a network actually receives for one event (3 x 21 numbers),
                    plus the keyhole view of the same event
episode.gif         a trained reference-game pair (context_urgent_0) playing real
                    episodes, the collector's belief updated after every symbol,
                    including one episode cut short by the danger clock
perspective.gif     the wanting-versus-reporting pair (persp12_0) on report and want
                    episodes, with room and deed beliefs after every symbol
curve.png           training curve of context_urgent_0
"""
from pathlib import Path
import argparse, io, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .world import World, TYPES, COLOURS, PRED1, PRED2, NT, NC, NP1, NP2, ELEMENT_DIM
from .train import load
from .agents import strip, pack
from .analysis import make_context

INK, MUTED, GRID = '#2C2C2A', '#5F5E5A', '#D3D1C7'
TARGET, OTHER, MARK, DANGER = '#0F6E56', '#B4B2A9', '#D85A30', '#A32D2D'
SURF = '#FCFBF7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.edgecolor': GRID,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})

COLS = list(TYPES) + list(COLOURS) + ['as S', 'as A', 'as P'] + list(PRED1) + list(PRED2)
GROUPS = [('entity type', 0, NT), ('colour', NT, NT + NC), ('role', NT + NC, NT + NC + 3),
          ('intransitive predicate', NT + NC + 3, NT + NC + 3 + NP1), ('transitive predicate', NT + NC + 3 + NP1, ELEMENT_DIM)]


def frame(fig):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert('RGB')


def save_gif(frames, durations, path):
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)


# ------------------------------------------------------------------ percept
def percept_figure(w, out):
    i = w.lookup[(2, 0, 4 * NC + 0, 0 * NC + 1)]          # guards(red lion, green apple)
    rooms = [i, w.edit(i, 'P_type', 1), w.edit(i, 'P_type', 2), w.edit(i, 'P_type', 3)]
    full = w.features[i].numpy(); key = w.view(torch.tensor([rooms]))[0, 0].numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6))
    for ax, data, title in ((axes[0], full, f'What the speaker perceives: {w.describe(i)}'),
                            (axes[1], key, 'Keyhole view of the same event, when the rooms differ only in the patient\'s type')):
        ax.imshow(data, cmap=matplotlib.colors.ListedColormap([SURF, TARGET]), vmin=0, vmax=1, aspect='auto')
        for r in range(3):
            for c in range(ELEMENT_DIM):
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=GRID, lw=.6))
                ax.text(c, r, f'{int(data[r, c])}', ha='center', va='center', fontsize=8, color=SURF if data[r, c] else MUTED)
        ax.set_yticks(range(3), ['element 1: predicate', 'element 2: participant', 'element 3: participant'])
        ax.set_xticks(range(ELEMENT_DIM), COLS, rotation=60, ha='right', fontsize=8)
        for name, a, b in GROUPS[1:]: ax.axvline(a - .5, color=MUTED, lw=1)
        ax.set_title(title, loc='left', fontsize=11, color=INK, pad=24)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    for name, a, b in GROUPS:
        axes[0].text((a + b - 1) / 2, -0.62, name, ha='center', fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(out / 'percept.png', dpi=110); plt.close(fig)


# ------------------------------------------------------------------ reference game
def episode_frames(s, l, w, cfg, ctx, order, danger_after=None):
    """ctx = [target, d1, d2, d3]; order: display order of rooms. Real speaker output, real collector beliefs."""
    ids = torch.tensor([[ctx[k] for k in order]]); tgt = order.index(0)
    rooms = w.features[ids]
    with torch.no_grad():
        tok = s(w.features[[ctx[0]]], rooms if cfg['context'] else None)['tokens']
        msg = strip(tok, s.eos)[0]
        logits, _ = l.anytime(rooms, tok)
        probs = torch.softmax(logits[0], -1).numpy()
    n = len(msg); stop = n if danger_after is None else min(n, danger_after)
    frames, durs = [], []
    for t in range(stop + 1):
        p = np.full(4, .25) if t == 0 else probs[t - 1]
        final = t == stop
        fig = plt.figure(figsize=(10.5, 5.4)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.5, 1], hspace=.45, wspace=.3)
        a0 = fig.add_subplot(gs[0, :]); a0.axis('off')
        a0.text(0, .95, 'Speaker privately knows the rewarded room:', fontsize=10, color=MUTED, va='top')
        a0.text(0, .72, w.describe(ctx[0]), fontsize=14, color=TARGET, va='top', fontweight='bold')
        a0.text(0, .40, 'Message so far:', fontsize=10, color=MUTED, va='top')
        for k in range(n):
            shown = k < t
            a0.add_patch(plt.Rectangle((.17 + k * .065, .06), .05, .3, transform=a0.transAxes, fc=TARGET if shown else SURF, ec=TARGET if shown else GRID, lw=1))
            if shown: a0.text(.195 + k * .065, .21, str(msg[k]), transform=a0.transAxes, ha='center', va='center', color=SURF, fontsize=13, fontweight='bold')
        if danger_after is not None and final:
            a0.text(.62, .21, f'danger arrived after symbol {stop}: act now', transform=a0.transAxes, va='center', color=DANGER, fontsize=11, fontweight='bold')
        elif final:
            a0.text(.17 + n * .065 + .01, .21, 'end', transform=a0.transAxes, va='center', color=MUTED, fontsize=10)
        a1 = fig.add_subplot(gs[1, 0]); a1.axis('off'); a1.set_title('Public rooms (the collector sees these)', loc='left', fontsize=10, color=MUTED)
        for r, eid in enumerate(ids[0].tolist()):
            y = .86 - r * .26; is_t = r == tgt
            a1.text(0, y, f'room {r + 1}', fontsize=10, color=MUTED, va='center')
            a1.text(.2, y, w.describe(eid), fontsize=11, va='center', color=INK, fontweight='bold' if (final and is_t) else 'normal')
        a2 = fig.add_subplot(gs[1, 1])
        choice = int(np.argmax(p))
        colors = [TARGET if (final and r == tgt) else OTHER for r in range(4)]
        a2.barh(range(4), p, color=colors, height=.6)
        a2.set_xlim(0, 1); a2.set_ylim(3.6, -.6); a2.set_yticks(range(4), [f'room {r + 1}' for r in range(4)])
        a2.set_title('Collector\'s belief after this symbol', loc='left', fontsize=10, color=MUTED)
        a2.grid(axis='x', color=GRID, lw=.6); a2.set_axisbelow(True)
        for sp in ('top', 'right', 'left'): a2.spines[sp].set_visible(False)
        for r in range(4):
            inside = p[r] > .8
            a2.text(p[r] - .02 if inside else p[r] + .02, r, f'{p[r]:.2f}', va='center', ha='right' if inside else 'left', fontsize=9, color=SURF if inside else MUTED)
        if final:
            ok = choice == tgt
            a0.text(1, .95, f'collector picks room {choice + 1}: {"right" if ok else "wrong"}', transform=a0.transAxes, ha='right', va='top', fontsize=12, color=TARGET if ok else DANGER, fontweight='bold')
        frames.append(frame(fig)); durs.append(2600 if final else (1300 if t == 0 else 1000))
    return frames, durs, msg


def episode_gif(run, out):
    s, l, w, cfg = load(run); s.eval(); l.eval(); rng = np.random.default_rng(3)
    train = [int(i) for i in np.nonzero(w.partition == 0)[0]]
    specs = [(2, ['P_type'] * 3, None), (2, ['A_colour', 'pred', 'swap'], None), (1, ['S_type', 'S_colour', 'pred'], None), (2, ['A_type', 'P_colour', 'pred'], 1)]
    frames, durs, log = [], [], []
    for val, slots, danger in specs:
        for _ in range(200):
            i = int(rng.choice(train))
            if w.events[i, 0] != val: continue
            ctx = make_context(w, rng, i, slots)
            if ctx: break
        order = list(rng.permutation(4))
        f, d, msg = episode_frames(s, l, w, cfg, ctx, order, danger)
        frames += f; durs += d; log.append(dict(target=w.describe(ctx[0]), rooms=[w.describe(ctx[k]) for k in order], message=list(msg), danger_after=danger))
    save_gif(frames, durs, out / 'episode.gif')
    return log


# ------------------------------------------------------------------ perspective game
def perspective_gif(run, out):
    from .perspective import Game, Speaker, Actor, prop, name, WANT, REPORT, HELD
    cp = torch.load(Path(run) / 'checkpoint.pt', map_location='cpu', weights_only=False); c = cp['config']
    s = Speaker(c['vocab'], c['max_len'], c['width'], inner=2); s.load_state_dict(cp['speaker']); s.eval()
    a = Actor(c['vocab'], c['width']); a.load_state_dict(cp['actor']); a.eval()
    g = Game(); w = g.w; rng = np.random.default_rng(11)
    train = [int(i) for i in np.nonzero(w.partition == 0)[0]]
    picks = [('A_type', REPORT, 4), ('A_type', WANT, 4), ('pred', REPORT, None), ('pred', WANT, None)]   # lion seen / wanted, then a predicate
    frames, durs, log = [], [], []
    base = None
    for slot, mood, typ in picks:
        if mood == WANT and base is not None and base[0] == slot: ctx = base[1]
        else:
            for _ in range(500):
                i = int(rng.choice(train));
                if slot not in w.slot_sizes(i): continue
                if typ is not None and w.slots(i)[slot] != typ: continue
                ctx = g.context(rng, i, slot)
                if ctx: break
            base = (slot, ctx)
        b = g.batch(rng, [ctx], [slot], [mood]); tgt = int(b['target'][0])
        with torch.no_grad():
            tok = s(b['private'], b['candidates'], inner=b['inner'])['tokens']; msg = strip(tok, s.eos)[0]
        rooms = b['rooms'][0].tolist(); wanted = name(prop(w, ctx[0], slot))
        for t in range(len(msg) + 1):
            with torch.no_grad():
                rl, dl = a(b['candidates'], pack([msg[:t]], a.eos, c['max_len']))
            pr, pd = torch.softmax(rl[0], -1).numpy(), torch.softmax(dl[0], -1).numpy()
            final = t == len(msg)
            fig = plt.figure(figsize=(11.5, 5.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], width_ratios=[1.7, 1, .7], hspace=.45, wspace=.4)
            a0 = fig.add_subplot(gs[0, :]); a0.axis('off')
            a0.text(0, .97, 'Inner mood (felt by the speaker, never shown to the collector):', fontsize=10, color=MUTED, va='top')
            a0.text(.5, .97, 'REPORT' if mood == REPORT else 'WANT', fontsize=12, va='top', fontweight='bold', color=TARGET if mood == REPORT else MARK)
            what = f'has seen: {w.describe(ctx[0])}' if mood == REPORT else f'wants something that is: {wanted}   (its percept shows only this)'
            a0.text(0, .70, what, fontsize=12.5, va='top', color=INK)
            a0.text(0, .38, 'Message so far:', fontsize=10, color=MUTED, va='top')
            for k in range(len(msg)):
                shown = k < t; mk = shown and msg[k] == 0
                fc = (MARK if mk else TARGET) if shown else SURF
                a0.add_patch(plt.Rectangle((.17 + k * .065, .04), .05, .3, transform=a0.transAxes, fc=fc, ec=fc if shown else GRID, lw=1))
                if shown: a0.text(.195 + k * .065, .19, str(msg[k]), transform=a0.transAxes, ha='center', va='center', color=SURF, fontsize=13, fontweight='bold')
            a0.text(.62, .19, 'orange = the symbol that marks wanting in this run', transform=a0.transAxes, va='center', fontsize=9, color=MUTED)
            a1 = fig.add_subplot(gs[1, 0]); a1.axis('off'); a1.set_title('Public rooms', loc='left', fontsize=10, color=MUTED)
            for r, eid in enumerate(rooms):
                y = .86 - r * .26
                a1.text(0, y, f'room {r + 1}', fontsize=10, color=MUTED, va='center')
                a1.text(.22, y, w.describe(eid), fontsize=10.5, va='center', color=INK, fontweight='bold' if (final and r == tgt) else 'normal')
            a2 = fig.add_subplot(gs[1, 1])
            a2.barh(range(4), pr, color=[TARGET if (final and r == tgt) else OTHER for r in range(4)], height=.6)
            a2.set_xlim(0, 1); a2.set_ylim(3.6, -.6); a2.set_yticks(range(4), [f'room {r + 1}' for r in range(4)])
            a2.set_title('Which room', loc='left', fontsize=10, color=MUTED)
            a3 = fig.add_subplot(gs[1, 2])
            right = mood
            a3.barh([0, 1], pd, color=[TARGET if (final and k == right) else OTHER for k in range(2)], height=.6)
            a3.set_xlim(0, 1); a3.set_ylim(1.6, -.6); a3.set_yticks([0, 1], ['look', 'fetch'])
            a3.set_title('Which deed', loc='left', fontsize=10, color=MUTED)
            for ax in (a2, a3):
                ax.grid(axis='x', color=GRID, lw=.6); ax.set_axisbelow(True)
                for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
            if final:
                rok, dok = int(np.argmax(pr)) == tgt, int(np.argmax(pd)) == mood
                verdict = 'team paid' if rok and dok else f'not paid: {"room" if not rok else "deed"} wrong' if rok != dok else 'not paid'
                a0.text(1, .97, verdict, transform=a0.transAxes, ha='right', va='top', fontsize=12, fontweight='bold', color=TARGET if rok and dok else DANGER)
                if mood == WANT and prop(w, ctx[0], slot) in HELD:
                    a0.text(1, .80, f'"{wanted}" was never wanted in training', transform=a0.transAxes, ha='right', va='top', fontsize=10, color=MUTED)
            frames.append(frame(fig)); durs.append(2800 if final else (1300 if t == 0 else 1000))
        log.append(dict(mood='report' if mood == REPORT else 'want', about=slot, property=wanted, target=w.describe(ctx[0]), message=list(msg)))
    save_gif(frames, durs, out / 'perspective.gif')
    return log


def curve_figure(run, out):
    curve = json.loads((Path(run) / 'curve.json').read_text())
    u = [r['update'] for r in curve]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title, ylim in ((axes[0], 'greedy', 'Collector picks the rewarded room (chance 0.25)', (0, 1)),
                                 (axes[1], 'length', 'Symbols per message (maximum 6)', (0, 6.2))):
        ax.plot(u, [r[key] for r in curve], color=TARGET, lw=2)
        ax.set_ylim(*ylim); ax.set_title(title, loc='left', fontsize=10.5, color=INK); ax.set_xlabel('training update (256 episodes each)')
        ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axes[0].axhline(.25, color=MUTED, lw=1, ls=(0, (4, 3)))
    fig.tight_layout(); fig.savefig(out / 'curve.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default='results/grammar_pressures/figures')
    p.add_argument('--episode-run', default='results/grammar_pressures/sweep/context_urgent_0')
    p.add_argument('--perspective-run', default='results/grammar_pressures/perspective/persp12_0')
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    percept_figure(World(), out)
    curve_figure(a.episode_run, out)
    print(json.dumps(dict(episode=episode_gif(a.episode_run, out), perspective=perspective_gif(a.perspective_run, out)), indent=1))
