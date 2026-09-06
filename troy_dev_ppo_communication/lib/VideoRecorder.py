import cv2
import numpy as np
import torch

from lib.Grounding import (
    TRIPLE_PAD,
    AnchorMemory,
    assemble_triples_geo,
    deltas_from_anchors,
    perceive_scene_geo,
)

CAM_LABELS = ("front", "left", "back", "right")
ACTION_NAMES = ("forward", "backward", "left", "right")

# overlay colors (BGR)
_BOX = (0, 255, 255)
_REL = (255, 128, 0)
_TEXT = (255, 255, 255)
_HUD_GRAY = 24


class PanoramaRecorder:
    """Streams stitched agent's-eye frames to an mp4.

    Each frame is the 4 body-mounted cameras tiled 2x2
    ([front | left] over [back | right]) with a text HUD strip underneath.
    Per-camera concept boxes and relation labels are drawn when provided
    (i.e. on frames where a perception pass actually ran, so overlays are
    never stale).
    """

    def __init__(self, path, frame_size=512, fps=8, hud_lines_max=3):
        self.path = path
        self.frame_size = frame_size
        self.hud_h = 22 * hud_lines_max + 12
        self.writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (2 * frame_size, 2 * frame_size + self.hud_h),
        )

    def add(self, frames, boxes_per_cam=None, relations_per_cam=None, hud_lines=()):
        """frames: 4 HxWx3 RGB arrays in CAM_LABELS order. boxes_per_cam:
        per-camera {concept_id: (x0,y0,x1,y1)} or None; relations_per_cam:
        per-camera [(src, rel_name, dst)] or None."""
        tiles = []
        for i, frame in enumerate(frames):
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            if boxes_per_cam is not None:
                for cid, (x0, y0, x1, y1) in boxes_per_cam[i].items():
                    cv2.rectangle(img, (x0, y0), (x1, y1), _BOX, 1)
                    cv2.putText(img, f"n{cid}", (x0 + 2, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _BOX, 1)
            if relations_per_cam is not None:
                for j, (src, rel, dst) in enumerate(relations_per_cam[i]):
                    cv2.putText(img, f"n{src} {rel} n{dst}", (6, 18 + 16 * j),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _REL, 1)
            cv2.putText(img, CAM_LABELS[i], (6, self.frame_size - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT, 2)
            tiles.append(img)
        grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
        hud = np.full((self.hud_h, grid.shape[1], 3), _HUD_GRAY, dtype=np.uint8)
        for j, line in enumerate(hud_lines):
            cv2.putText(hud, str(line)[:150], (8, 22 + 22 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1)
        self.writer.write(np.vstack([grid, hud]))

    def close(self):
        self.writer.release()


def record_points(total, n_points=5):
    """Evenly spaced recording points 0, total/4, ..., total (deduplicated)."""
    return sorted({round(total * k / (n_points - 1)) for k in range(n_points)})


def segment_start_iters(total, n_segments):
    """Iterations that BEGIN each of `n_segments` equal segments of a run.

    Iterations are 1-indexed (the training loop is `range(1, iterations+1)`),
    so with total=450 and n_segments=10 the segments are [1..45], [46..90],
    ... [406..450] and this returns their first iterations: 1, 46, ..., 406.
    One recording per segment, taken after that iteration's update.

    Note this deliberately excludes iteration 0 (the untrained policy) and the
    final iteration, neither of which begins a segment. Duplicates collapse
    when n_segments exceeds total.
    """
    if n_segments < 1:
        return []
    seg = total / n_segments
    return sorted({1 + int(round(i * seg)) for i in range(n_segments)})


def format_triples(kg, triples):
    """Human-readable one-liner for a [K,3] triple tensor."""
    parts = []
    for src, rel, dst in triples.tolist():
        if src == TRIPLE_PAD:
            continue
        if rel == TRIPLE_PAD:
            parts.append(f"(n{src})")
        else:
            parts.append(f"(n{src} {kg.relation_names[rel]} n{dst})")
    return " ".join(parts) if parts else "(none)"


@torch.no_grad()
def record_policy_episode(path, env, model, pipeline, kg, k_triples, perceive_every,
                          device, tag="", fps=6, per_frame_fn=None):
    """Roll out one episode with the current policy and write its agent's-eye
    video: stitched panorama per step, perception overlays on the frames where
    perception ran, and a HUD with the action, value, return, and the exact
    triples the policy is seeing. Returns (outcome, episode_return).

    per_frame_fn overrides how perception is run, for callers on a non-SAM path
    (see lib/OracleGrounding). It must return the same per-camera 4-tuple list
    perceive_scene_geo does; triple assembly and the overlays are shared. The
    default keeps the SAM behaviour and ignores the override entirely."""
    rec = PanoramaRecorder(path, frame_size=env.render_size, fps=fps)
    obs, _ = env.reset()

    def perceive():
        if per_frame_fn is None:
            per_frame = perceive_scene_geo(env, pipeline, kg)
        else:
            per_frame = per_frame_fn()
        boxes = [pf[0] for pf in per_frame]
        rels = [pf[3] for pf in per_frame]
        triples, deltas = assemble_triples_geo(kg, per_frame, k=k_triples)
        return boxes, rels, triples, deltas

    mem = AnchorMemory()
    boxes, rels, triples, deltas = perceive()
    triples, anchors, mask = mem.merge(triples, deltas, obs[:2])
    ep_return = 0.0
    outcome = "timeout"
    rec.add(env.render_panorama(), boxes, rels, hud_lines=[
        f"{tag}  step 0  lion_caged={env.lion_caged}",
        f"triples: {format_triples(kg, triples)}",
    ])
    for step in range(1, env.max_steps + 1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        deltas = deltas_from_anchors(anchors, mask, obs)
        action, _, value = model.act(
            obs_t, triples.to(device).unsqueeze(0), deltas.to(device).unsqueeze(0)
        )
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        ep_return += reward
        done = terminated or truncated
        fresh = bool(perceive_every) and step % perceive_every == 0 and not done
        if fresh:
            boxes, rels, triples, deltas = perceive()
            triples, anchors, mask = mem.merge(triples, deltas, obs[:2])
        status = f"lion_caged={env.lion_caged}"
        if done:
            outcome = ("food" if info.get("food_reached")
                       else "death" if info.get("lion_caught")
                       else "water" if info.get("water_reached")
                       else "poison" if info.get("poison_touched") else "timeout")
            status += f"  OUTCOME: {outcome}"
        rec.add(
            env.render_panorama(),
            boxes if fresh else None,
            rels if fresh else None,
            hud_lines=[
                f"{tag}  step {step}  action={ACTION_NAMES[int(action)]}  "
                f"V={float(value):+.2f}  return={ep_return:+.2f}",
                f"triples: {format_triples(kg, triples)}",
                status,
            ],
        )
        if done:
            break
    rec.close()
    return outcome, ep_return


@torch.no_grad()
def record_communication_episode(
    path, env, transmitter, receiver, kg, make_tx_input, perceive_passes,
    device, tag="", fps=6, greedy=False,
):
    """Roll out one two-agent episode and write the transmitter's-eye video.

    The cameras belong to the transmitter, which never moves, so the panorama is
    a fixed viewpoint watching the receiver travel. The HUD carries the emitted
    symbol, the receiver's move, both critics' values, and the running context
    window, which is what makes it possible to eyeball whether a symbol means
    anything consistent.

    `make_tx_input` is the SAME factory the trainer was built with, so the input
    shown here is assembled by the same code that produced it during training --
    stripping, slot count, receiver slot, and per-step anchor refresh included.
    Returns (outcome, episode_return).
    """
    from lib.ReceiverPolicy import SYMBOL_NAMES, build_context, format_context

    rec = PanoramaRecorder(path, frame_size=env.render_size, fps=fps)
    obs, _ = env.reset()

    tx_input = make_tx_input()
    boxes, rels = tx_input.refresh(obs)
    symbols = []
    ep_return = 0.0
    outcome = "timeout"
    rec.add(env.render_panorama(), boxes, rels, hud_lines=[
        f"{tag}  step 0  lion_caged={env.lion_caged}",
        f"triples: {format_triples(kg, tx_input.triples)}",
    ])
    for step in range(1, env.max_steps + 1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        step_deltas = tx_input.deltas(obs, env)

        symbol, _, s_val = transmitter.act(
            obs_t,
            tx_input.triples.to(device).unsqueeze(0),
            step_deltas.to(device).unsqueeze(0),
        )
        sym = int(symbol.item())
        symbols.append(sym)
        context = build_context(symbols, device=device).unsqueeze(0)
        action, _, a_val = receiver.act(context, greedy=greedy)

        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        ep_return += reward
        done = terminated or truncated
        # same cadence as the trainer: SAM only on the opening steps
        fresh = step < perceive_passes and not done
        if fresh:
            boxes, rels = tx_input.refresh(obs)
        status = f"lion_caged={env.lion_caged}"
        if done:
            outcome = ("food" if info.get("food_reached")
                       else "death" if info.get("lion_caught") else "timeout")
            status += f"  OUTCOME: {outcome}"
        rec.add(
            env.render_panorama(),
            boxes if fresh else None,
            rels if fresh else None,
            hud_lines=[
                f"{tag}  step {step}  TX={SYMBOL_NAMES[sym]}  "
                f"RX={ACTION_NAMES[int(action)]}  "
                f"Vtx={float(s_val):+.2f} Vrx={float(a_val):+.2f}  "
                f"return={ep_return:+.2f}",
                f"RX context: {format_context(context[0])}",
                f"triples: {format_triples(kg, tx_input.triples)}",
                status,
            ],
        )
        if done:
            break
    rec.close()
    return outcome, ep_return
