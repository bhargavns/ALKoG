"""Soft semantic category observations and short-term anchor memory."""

import math

import numpy as np
import torch

from lib.Grounding import frame_maybe_has_objects
from lib.KGWorldEnv import CAMERA_NAMES
from lib.ObjectCategories import CATEGORY_TO_ID, NUM_OBJECT_CATEGORIES
from lib.Perception import relations_from_boxes, union_boxes_by_concept

SLOT_FEATURE_DIM = 6
RELATION_FEATURE_DIM = 2  # lion-inside-cage, lion-near-cage


@torch.inference_mode()
def detect_soft_categories_geo(env, pipeline, category_model, min_confidence=0.45):
    """Convert panoramic proposals into one soft observation per category.

    Returns ``(detections, relation_features, debug_frames)``.  A detection is
    ``category_id -> dict(delta, presence, confidence, camera_count)``.  The
    category classifier is soft, while category-wise union boxes provide a
    stable geometry estimate for the initial experiment.
    """
    evidence = {category_id: [] for category_id in range(NUM_OBJECT_CATEGORIES)}
    relation_votes = []
    debug_frames = []

    for cam_name, frame in zip(CAMERA_NAMES, env.render_panorama()):
        if not frame_maybe_has_objects(frame):
            debug_frames.append((frame, [], [], []))
            continue
        boxes, _masks, embeddings, _quality = pipeline.detect_full(frame)
        if not boxes:
            debug_frames.append((frame, [], [], []))
            continue
        probabilities = category_model.probabilities(embeddings)
        confidences, predicted = probabilities.max(dim=-1)
        predicted_ids = []
        kept_boxes = []
        kept_confidences = []
        for box, category_id, confidence in zip(boxes, predicted.tolist(), confidences.tolist()):
            if category_id >= NUM_OBJECT_CATEGORIES or confidence < min_confidence:
                continue
            predicted_ids.append(int(category_id))
            kept_boxes.append(box)
            kept_confidences.append(float(confidence))

        # Construct policy evidence from the complete soft distribution, not
        # only argmax labels.  The best proposals for each category contribute
        # to its presence and geometry, so uncertainty remains visible.
        for category_id in range(NUM_OBJECT_CATEGORIES):
            category_scores_tensor = probabilities[:, category_id]
            confidence = float(category_scores_tensor.max())
            if confidence < min_confidence:
                continue
            selection_threshold = max(min_confidence, 0.7 * confidence)
            selected_indices = torch.nonzero(
                category_scores_tensor >= selection_threshold, as_tuple=False
            ).flatten().tolist()
            selected_boxes = [boxes[index] for index in selected_indices]
            union_box = union_boxes_by_concept(
                [category_id] * len(selected_boxes), selected_boxes
            )[category_id]
            x0, y0, x1, y1 = union_box
            category_scores = [float(category_scores_tensor[index]) for index in selected_indices]
            # A bounded soft-OR: several consistent fragments increase
            # presence without allowing large proposal counts to exceed one.
            presence = 1.0 - math.prod(1.0 - score for score in category_scores)
            delta = env.ground_delta(cam_name, (x0 + x1) / 2.0, y1)
            evidence[category_id].append(
                {
                    "delta": np.asarray(delta, dtype=np.float32),
                    "presence": float(presence),
                    "confidence": confidence,
                    "area": float((x1 - x0) * (y1 - y0)),
                }
            )

        concept_boxes = union_boxes_by_concept(predicted_ids, kept_boxes)
        relations = relations_from_boxes(concept_boxes, frame.shape[1])
        relation_votes.extend(relations)
        debug_frames.append((frame, kept_boxes, predicted_ids, kept_confidences))

    detections = {}
    for category_id, candidates in evidence.items():
        if not candidates:
            continue
        # The largest view usually has the most reliable bottom edge.  Presence
        # and confidence still aggregate across all agreeing cameras.
        best = max(candidates, key=lambda item: item["area"])
        detections[category_id] = {
            "delta": best["delta"],
            "presence": float(1.0 - math.prod(1.0 - c["presence"] for c in candidates)),
            "confidence": float(max(c["confidence"] for c in candidates)),
            "camera_count": len(candidates),
        }

    lion_id = CATEGORY_TO_ID["lion"]
    cage_id = CATEGORY_TO_ID["cage"]
    inside = any(src == lion_id and rel == "inside" and dst == cage_id for src, rel, dst in relation_votes)
    near = any(
        rel == "near" and {src, dst} == {lion_id, cage_id}
        for src, rel, dst in relation_votes
    )
    relations = np.asarray([float(inside), float(near)], dtype=np.float32)
    return detections, relations, debug_frames


class SoftCategoryMemory:
    """Fixed category slots with decaying world-frame anchors.

    Fixed categories make the first experiment deterministic and inspectable.
    Repeated instances of one category require a later instance-slot tracker;
    adding new categories only requires extending ``ObjectCategories.py``.
    """

    def __init__(self, max_age=100, decay=0.98):
        self.max_age = int(max_age)
        self.decay = float(decay)
        self.reset()

    def reset(self):
        self.anchors = {}
        self.relations = np.zeros(RELATION_FEATURE_DIM, dtype=np.float32)
        self.relation_age = 0

    def update(self, detections, relations, agent_xy):
        agent = np.asarray(agent_xy, dtype=np.float32)
        for category_id, detection in detections.items():
            self.anchors[int(category_id)] = {
                "world": agent + np.asarray(detection["delta"], dtype=np.float32),
                "presence": float(detection["presence"]),
                "confidence": float(detection["confidence"]),
                "age": 0,
            }
        self.relations = np.asarray(relations, dtype=np.float32)
        self.relation_age = 0

    def advance(self):
        expired = []
        for category_id, entry in self.anchors.items():
            entry["age"] += 1
            if entry["age"] > self.max_age:
                expired.append(category_id)
        for category_id in expired:
            del self.anchors[category_id]
        self.relation_age += 1

    def features(self, obs):
        """Return ``[C,6]`` slots and two decayed relation features."""
        obs = np.asarray(obs, dtype=np.float32)
        agent = obs[:2]
        cosine, sine = float(obs[2]), float(obs[3])
        slots = np.zeros((NUM_OBJECT_CATEGORIES, SLOT_FEATURE_DIM), dtype=np.float32)
        for category_id, entry in self.anchors.items():
            age = int(entry["age"])
            world_delta = entry["world"] - agent
            forward = world_delta[0] * cosine + world_delta[1] * sine
            left = -world_delta[0] * sine + world_delta[1] * cosine
            decay = self.decay**age
            slots[category_id] = np.asarray(
                [
                    entry["presence"] * decay,
                    entry["confidence"] * decay,
                    float(age == 0),
                    min(age / max(1, self.max_age), 1.0),
                    forward,
                    left,
                ],
                dtype=np.float32,
            )
        relation_decay = self.decay**self.relation_age
        return slots, self.relations * relation_decay
