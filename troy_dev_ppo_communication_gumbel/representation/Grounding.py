import numpy as np
import torch


# Rows are (source_object_id, relation_id, destination_object_id).
# Valid IDs start at 0; -1 means "no token" and must be masked, not indexed.
# (food_id, -1, -1) represents food alone; (-1, -1, -1) is an unused row.
# Padding keeps the input shape fixed even when the number of detections varies.
TRIPLE_PAD = -1

def assemble_triples_geo(vocabulary, per_frame, k=6):
    """Build the policy input from a visual category-grounding pass.

    Relation triples are deduplicated across frames and, when more than `k`
    exist, ranked by the mean classification confidence of their two nodes (best evidence
    across frames). Remaining slots take lone detected concepts as unary
    entries (concept, PAD, PAD), ranked by their classification confidence. Chosen entries
    are sorted for deterministic assembly; unfilled slots are all-PAD.

    Each concept's (dx, dy) comes from the frame where its union box was
    largest (the closest, most reliable view).

    Returns (LongTensor [k, 3], FloatTensor [k, 2, 2]) -- triple indices and
    the agent->src / agent->dst offsets per slot (zeros for PAD).
    """
    # Each frame supplies boxes, category confidences, world-axis offsets,
    # and relations, all keyed by the shared scene object IDs. Offsets already
    # use the environment's normalized position units; yaw rotation comes later.
    # "sim" is a legacy variable name here: these values are classifier confidences.
    best_sim, best_delta, best_area = {}, {}, {}
    for concept_boxes, concept_sims, concept_deltas, _ in per_frame:
        for cid, (x0, y0, x1, y1) in concept_boxes.items():
            area = (x1 - x0) * (y1 - y0)
            if area > best_area.get(cid, -1):
                best_area[cid] = area
                best_delta[cid] = concept_deltas[cid]
            # Confidence and position can come from different camera views:
            # keep the highest confidence, but use the largest box for position.
            best_sim[cid] = max(best_sim.get(cid, -1.0), concept_sims[cid])

    # A relation seen in several cameras occupies only one row. Its score is
    # the strongest per-camera average of the two endpoint confidences.
    scored = {}
    for _, concept_sims, _, relations in per_frame:
        for src, rel_name, dst in relations:
            key = (src, vocabulary.relation_index(rel_name), dst)
            score = 0.5 * (concept_sims[src] + concept_sims[dst])
            scored[key] = max(scored.get(key, -1.0), score)

    triples = [key for key, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]
    # Relations get slots first. Use spare slots for detected objects that
    # are not already endpoints of a selected relation.
    covered = set()
    for src, _, dst in triples:
        covered.update((src, dst))
    lone = sorted((c for c in best_sim if c not in covered), key=lambda c: -best_sim[c])
    triples += [(cid, TRIPLE_PAD, TRIPLE_PAD) for cid in lone[: k - len(triples)]]
    # Confidence determines selection; IDs determine the final row order.
    triples.sort()

    # Axes: [row, endpoint (source/destination), coordinate (world x/y)].
    # Relations have no position of their own; missing endpoints stay zero.
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

    Scene objects are static within an episode (the moving receiver is handled
    separately by TransmitterInput), so a concept seen at any earlier
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
        remembered concepts re-inserted into free slots. Entries use the same
        ID ordering as assemble_triples_geo; row indices can change when the
        selected entries change. Anchors for PAD endpoints stay zero.
        """
        k = triples.shape[0]
        agent = torch.as_tensor(agent_xy, dtype=deltas.dtype)

        entries = []
        for i in range(k):
            src, rel, dst = (int(v) for v in triples[i])
            if src == TRIPLE_PAD:
                continue
            # Store absolute world positions so transmitter movement does not
            # invalidate remembered locations. New sightings replace old estimates.
            self._anchors[src] = (deltas[i, 0] + agent).clone()
            if dst != TRIPLE_PAD:
                self._anchors[dst] = (deltas[i, 1] + agent).clone()
            entries.append((src, rel, dst))

        # Remember positions, not historical relations. Missing objects return
        # as unary rows only when space remains; current rows are never evicted.
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
        # Columns 0 and 2 are object endpoints; column 1 is a relation ID.
        # Shape [k, 2, 1] broadcasts across each endpoint's x/y coordinates.
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
    # Subtract the current transmitter position, then zero absent endpoints
    # so a padded zero anchor does not look like an object at the world origin.
    d = (anchors - o[:2]) * mask
    c, s = o[2], o[3]
    # Dot with the heading (cos, sin) and left (-sin, cos) unit vectors.
    # The result is forward/left displacement in the same normalized units.
    return torch.stack(
        [d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], dim=-1
    )
