from representation.CommunicationCadence import perception_due
import cv2
import numpy as np
import torch

from representation.Grounding import TRIPLE_PAD

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


def format_triples(vocabulary, triples):
    """Human-readable one-liner for a [K,3] triple tensor."""
    parts = []
    for src, rel, dst in triples.tolist():
        if src == TRIPLE_PAD:
            continue
        if rel == TRIPLE_PAD:
            parts.append(f"(n{src})")
        else:
            parts.append(f"(n{src} {vocabulary.relation_names[rel]} n{dst})")
    return " ".join(parts) if parts else "(none)"


@torch.no_grad()
def record_communication_episode(
    path, env, transmitter, receiver, vocabulary, make_tx_input, perceive_passes,
    device, tag="", fps=6, greedy=False, channel_mode="hard",
):
    """Roll out one two-agent episode and write the transmitter's-eye video.

    The cameras belong to the transmitter: stationary by default, following
    the receiver in co-located mode. The HUD carries the emitted
    symbol, the receiver's move, and its value estimate.

    `make_tx_input` is the SAME factory the trainer was built with, so the input
    shown here is assembled by the same code that produced it during training --
    stripping, slot count, receiver slot, and per-step anchor refresh included.
    Returns (outcome, episode_return).
    """
    from evaluation.CommunicationEvaluation import CommunicationActor, episode_outcome

    rec = PanoramaRecorder(path, frame_size=env.render_size, fps=fps)
    obs, _ = env.reset()

    tx_input = make_tx_input()
    boxes, rels = tx_input.refresh(obs)
    actor = CommunicationActor(transmitter, receiver, device, mode=channel_mode, greedy=greedy)
    ep_return = 0.0
    outcome = "timeout"
    rec.add(env.render_panorama(), boxes, rels, hud_lines=[
        f"{tag}  step 0  lion_caged={env.lion_caged}",
        f"triples: {format_triples(vocabulary, tx_input.triples)}",
    ])
    for step in range(1, env.max_steps + 1):
        action, sym, a_val, message_text = actor.act(
            obs, tx_input.triples, tx_input.deltas(obs, env))
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        done = terminated or truncated
        # Same episode-local cadence as both trainers and held-out evaluation.
        fresh = not done and perception_due(env, step, perceive_passes)
        if fresh:
            boxes, rels = tx_input.refresh(obs)
        status = f"lion_caged={env.lion_caged}"
        if done:
            outcome = episode_outcome(info, truncated)
            status += f"  OUTCOME: {outcome}"
        rec.add(
            env.render_panorama(),
            boxes if fresh else None,
            rels if fresh else None,
            hud_lines=[
                f"{tag}  step {step}  TX={sym}  "
                f"RX={ACTION_NAMES[int(action)]}  "
                f"Vrx={a_val:+.2f} channel={channel_mode}  "
                f"return={ep_return:+.2f}",
                f"RX message: {message_text}",
                f"triples: {format_triples(vocabulary, tx_input.triples)}",
                status,
            ],
        )
        if done:
            break
    rec.close()
    return outcome, ep_return
