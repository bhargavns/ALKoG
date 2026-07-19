import cv2
import numpy as np
import torch

from lib.KGWorldEnv import CAMERA_NAMES
from lib.Perception import relations_from_boxes, union_boxes_by_concept

TRIPLE_PAD = -1

# Objects (orange lion, blue cage, green food) are strongly saturated;
# floor and sky are not. Frames with no saturated pixels skip SAM entirely.
_SATURATION_MIN = 60
_SATURATION_PIXELS = 4


def frame_maybe_has_objects(frame):
    small = frame[::8, ::8].astype(np.int16)
    saturation = small.max(axis=2) - small.min(axis=2)
    return int((saturation > _SATURATION_MIN).sum()) >= _SATURATION_PIXELS


def perceive_scene(env, pipeline, kg, allow_add):
    """Run the full perception stack on the agent's 4-camera panorama.

    For each frame: SAM boxes -> CNN embeddings -> KG concept matching ->
    per-concept union boxes -> geometric relations.

    allow_add=True (phase 1) merges/creates nodes; False (phase 2) matches
    against the frozen graph only.

    Returns list of (concept_boxes, relations, frame) per camera, where
    concept_boxes is {concept_id: box} and relations is
    [(src_concept, relation_name, dst_concept)].
    """
    results = []
    for frame in env.render_panorama():
        if not frame_maybe_has_objects(frame):
            results.append(({}, [], frame))
            continue
        boxes, embeddings = pipeline.detect(frame)
        concept_ids = kg.match_batch(embeddings, allow_add=allow_add)
        concept_boxes = union_boxes_by_concept(concept_ids, boxes)
        relations = relations_from_boxes(concept_boxes, frame.shape[1])
        results.append((concept_boxes, relations, frame))
    return results


def assemble_triples(kg, per_frame, k=4):
    """Build the policy's symbol-triple input from one perception pass.

    Relation triples (src, rel_idx, dst) fill the K slots first, deduplicated
    across frames; remaining slots take unary entries (concept, PAD, PAD) for
    detected concepts not already covered by a relation triple. Unfilled slots
    are all-PAD.

    Returns LongTensor [k, 3].
    """
    relation_triples = set()
    seen_concepts = set()
    for concept_boxes, relations, _ in per_frame:
        for src, rel_name, dst in relations:
            relation_triples.add((src, kg.relation_index(rel_name), dst))
            seen_concepts.update((src, dst))

    detected = set()
    for concept_boxes, _, _ in per_frame:
        for cid in concept_boxes:
            if cid not in seen_concepts:
                detected.add(cid)

    # sorted -> a given triple always occupies the same slot (the MLP input
    # is positional); relation triples take priority over unary detections
    triples = sorted(relation_triples)
    triples += [(cid, TRIPLE_PAD, TRIPLE_PAD) for cid in sorted(detected)]

    triples = triples[:k]
    while len(triples) < k:
        triples.append((TRIPLE_PAD, TRIPLE_PAD, TRIPLE_PAD))
    return torch.tensor(triples, dtype=torch.long)


def perceive_scene_geo(env, pipeline, kg):
    """Phase-2 perception pass with match scores and ground-plane offsets.

    Like perceive_scene(allow_add=False), but additionally records, per
    camera frame:
      - concept_sims:   {concept_id: best cosine similarity among the boxes
                         that matched it} (the retrieval confidence)
      - concept_deltas: {concept_id: (dx, dy)/POS_SCALE} estimated by casting
                        a ray through the union box's bottom-center and
                        intersecting the ground plane (vision-only range)

    Returns list of (concept_boxes, concept_sims, concept_deltas, relations)
    per camera.
    """
    results = []
    for cam_name, frame in zip(CAMERA_NAMES, env.render_panorama()):
        if not frame_maybe_has_objects(frame):
            results.append(({}, {}, {}, []))
            continue
        boxes, embeddings = pipeline.detect(frame)
        ids, sims = kg.match_batch_sims(embeddings)
        concept_boxes = union_boxes_by_concept(ids, boxes)
        concept_sims = {}
        for cid, sim in zip(ids, sims):
            if cid is not None:
                concept_sims[cid] = max(concept_sims.get(cid, -1.0), sim)
        concept_deltas = {
            cid: env.ground_delta(cam_name, (x0 + x1) / 2.0, y1)
            for cid, (x0, y0, x1, y1) in concept_boxes.items()
        }
        relations = relations_from_boxes(concept_boxes, frame.shape[1])
        results.append((concept_boxes, concept_sims, concept_deltas, relations))
    return results


def assemble_triples_geo(kg, per_frame, k=6):
    """Build the phase-2 policy input from one perceive_scene_geo pass.

    Relation triples are deduplicated across frames and, when more than `k`
    exist, ranked by the mean match cosine of their two nodes (best evidence
    across frames). Remaining slots take lone detected concepts as unary
    entries (concept, PAD, PAD), ranked by their match cosine. Chosen entries
    are sorted so a given triple always occupies the same slot (the MLP input
    is positional); unfilled slots are all-PAD.

    Each concept's (dx, dy) comes from the frame where its union box was
    largest (the closest, most reliable view).

    Returns (LongTensor [k, 3], FloatTensor [k, 2, 2]) -- triple indices and
    the agent->src / agent->dst offsets per slot (zeros for PAD).
    """
    best_sim, best_delta, best_area = {}, {}, {}
    for concept_boxes, concept_sims, concept_deltas, _ in per_frame:
        for cid, (x0, y0, x1, y1) in concept_boxes.items():
            area = (x1 - x0) * (y1 - y0)
            if area > best_area.get(cid, -1):
                best_area[cid] = area
                best_delta[cid] = concept_deltas[cid]
            best_sim[cid] = max(best_sim.get(cid, -1.0), concept_sims[cid])

    scored = {}
    for _, concept_sims, _, relations in per_frame:
        for src, rel_name, dst in relations:
            key = (src, kg.relation_index(rel_name), dst)
            score = 0.5 * (concept_sims[src] + concept_sims[dst])
            scored[key] = max(scored.get(key, -1.0), score)

    triples = [key for key, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]
    covered = set()
    for src, _, dst in triples:
        covered.update((src, dst))
    lone = sorted((c for c in best_sim if c not in covered), key=lambda c: -best_sim[c])
    triples += [(cid, TRIPLE_PAD, TRIPLE_PAD) for cid in lone[: k - len(triples)]]
    triples.sort()

    deltas = torch.zeros(k, 2, 2)
    for i, (src, _, dst) in enumerate(triples):
        deltas[i, 0] = torch.as_tensor(best_delta[src], dtype=torch.float32)
        if dst != TRIPLE_PAD:
            deltas[i, 1] = torch.as_tensor(best_delta[dst], dtype=torch.float32)

    while len(triples) < k:
        triples.append((TRIPLE_PAD, TRIPLE_PAD, TRIPLE_PAD))
    return torch.tensor(triples, dtype=torch.long), deltas


class AnchorMemory:
    """Per-episode memory of concept -> world-frame anchor position.

    Objects are static within an episode, so a concept seen at any earlier
    pass keeps a valid anchor even when the current pass misses it (a small
    object at range subtends few pixels and SAM drops it in a large share
    of passes). merge() updates the memory from the current pass's
    detections and back-fills free triple slots with remembered concepts as
    lone (cid, PAD, PAD) entries, so e.g. the food vector stays in the
    policy input from its first successful sighting onward. reset() forgets
    everything at episode boundaries (layouts are re-randomized).
    """

    def __init__(self):
        self._anchors = {}

    def reset(self):
        self._anchors.clear()

    def merge(self, triples, deltas, agent_xy):
        """triples [k,3] / deltas [k,2,2] from the current pass; `agent_xy`
        is the agent's /POS_SCALE position at pass time (first two obs
        dims). Returns (triples [k,3], anchors [k,2,2], mask [k,2,1]) with
        remembered concepts re-inserted into free slots; entries are sorted
        the same way assemble_triples_geo sorts, so a concept occupies the
        same slot whether it came from the pass or from memory. PAD slots
        stay zero.
        """
        k = triples.shape[0]
        agent = torch.as_tensor(agent_xy, dtype=deltas.dtype)

        entries = []
        for i in range(k):
            src, rel, dst = (int(v) for v in triples[i])
            if src == TRIPLE_PAD:
                continue
            self._anchors[src] = (deltas[i, 0] + agent).clone()
            if dst != TRIPLE_PAD:
                self._anchors[dst] = (deltas[i, 1] + agent).clone()
            entries.append((src, rel, dst))

        seen = {n for e in entries for n in (e[0], e[2]) if n != TRIPLE_PAD}
        missing = sorted(set(self._anchors) - seen)
        entries += [(cid, TRIPLE_PAD, TRIPLE_PAD) for cid in missing[: k - len(entries)]]
        entries.sort()

        out = torch.full((k, 3), TRIPLE_PAD, dtype=torch.long)
        anchors = torch.zeros(k, 2, 2, dtype=deltas.dtype)
        for i, (src, rel, dst) in enumerate(entries):
            out[i] = torch.tensor((src, rel, dst))
            anchors[i, 0] = self._anchors[src]
            if dst != TRIPLE_PAD:
                anchors[i, 1] = self._anchors[dst]
        mask = (out[:, [0, 2]] != TRIPLE_PAD).unsqueeze(-1).to(deltas.dtype)
        return out, anchors, mask


def deltas_from_anchors(anchors, mask, obs):
    """Agent-frame (forward, left) offsets to the anchored world positions.

    `obs` is the kinematic env observation [x/8, y/8, cos(yaw), sin(yaw)].
    The world-frame offset is rotated into the agent's frame so that "object
    ahead" produces the same input regardless of the episode's yaw -- the
    action set is egocentric (forward/backward/left/right), so this makes
    the offset->action mapping linear instead of requiring the policy to
    compute the yaw rotation itself.
    """
    o = torch.as_tensor(obs, dtype=anchors.dtype)
    d = (anchors - o[:2]) * mask
    c, s = o[2], o[3]
    return torch.stack(
        [d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], dim=-1
    )


def draw_debug_frame(frame, concept_boxes, relations):
    """Annotate a frame with concept union boxes and relation labels (BGR out)."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    for cid, (x0, y0, x1, y1) in concept_boxes.items():
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 1)
        cv2.putText(
            img, f"c{cid}", (x0 + 2, max(10, y0 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1,
        )
    for i, (src, rel, dst) in enumerate(relations):
        cv2.putText(
            img, f"c{src} {rel} c{dst}", (4, 12 + 12 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 128, 0), 1,
        )
    return img
