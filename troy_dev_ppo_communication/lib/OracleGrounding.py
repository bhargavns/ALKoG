"""Ground-truth perception that leaves the KG triple pipeline untouched.

Only the concept-identification step changes. Instead of matching a proposal's
embedding against KG nodes by cosine similarity, the proposal is looked up in
MuJoCo's segmentation image and assigned the body it actually covers. The
output is the same per-camera 4-tuple `perceive_scene_geo` returns, so
`assemble_triples_geo`, `AnchorMemory`, `deltas_from_anchors`, `SymbolTable`,
`SlotAttention` and `PPOTrainer` all run unmodified -- a run isolates
perception error from policy learning without changing the policy at all.

Modes
-----
oracle_id    SAM still proposes the boxes; the oracle only assigns identity.
             Keeps SAM's recall failures (measured on this world: the food is
             proposed in ~31% of frames where it is visible) and removes KG
             matching error. Answers "is concept matching the problem?"
oracle_full  Boxes come from the segmentation image itself, so every visible
             object is found exactly once, with an exact silhouette. Removes
             all perception error. Answers "what can the policy do at the
             perception ceiling?"

Both keep relations geometric: `relations_from_boxes` runs on the resulting
union boxes exactly as in the SAM path, so `inside`/`near` are still inferred
from 2D box overlap rather than read from simulator state. The caged/loose
distinction therefore still has to travel through a (lion, inside, cage)
triple -- the oracle does not hand it to the policy.
"""

import numpy as np
import mujoco
import torch

from lib.Grounding import frame_maybe_has_objects
from lib.KGWorldEnv import CAMERA_NAMES
from lib.Oracle import BACKGROUND, AMBIGUOUS, geom_body_table, label_mask
from lib.ObjectCategories import ID_TO_CATEGORY
from lib.Perception import EMBEDDING_DIM, relations_from_boxes, union_boxes_by_concept
from lib.SymbolicKG import SymbolicKG

ORACLE_MODES = ("oracle_id", "oracle_full")
PERCEPTION_MODES = ("sam", "softcat") + ORACLE_MODES

# Bodies the oracle is allowed to resolve. Deliberately one node per body:
# a real phase-1 KG often splinters the lion into separate caged/loose
# concepts, which leaks the caged/loose bit into node identity. Collapsing
# each body to a single node forces that bit back through the inside relation,
# which is the distinction the task is actually built around.
# "receiver" is in the base set, not the resource set: the receiving agent is
# present in both worlds. Without it here the category head would predict
# `receiver` and body_to_node.get("receiver") would return None, silently
# dropping every receiver detection before it could reach a triple.
_BASE_BODIES = ("food", "lion", "cage", "receiver")
_RESOURCE_BODIES = ("water", "poison")


def oracle_bodies(env):
    """Bodies present for this env configuration."""
    if getattr(env, "resources", False):
        return _BASE_BODIES + _RESOURCE_BODIES
    return _BASE_BODIES


def build_oracle_kg(bodies, device="cuda", seed=0):
    """A KG with exactly one node per body, in `bodies` order.

    Node embeddings are mutually orthogonal one-hots so each detection creates
    its own node; they are never used for matching under the oracle, they only
    give SymbolTable the right node count. Symbols stay randomly initialized
    exactly as in a phase-1 KG -- they carry no information at init either way,
    since PPO trains them from scratch.

    Returns (kg, body_to_node).
    """
    kg = SymbolicKG(device=device, seed=seed)
    body_to_node = {}
    for i, body in enumerate(bodies):
        embedding = torch.zeros(EMBEDDING_DIM, device=device)
        embedding[i] = 1.0
        node_id, created, _sim, _sims = kg.match_or_add(embedding)
        if not created:
            raise RuntimeError(f"oracle KG node collision for body {body!r}")
        body_to_node[body] = node_id
    return kg, body_to_node


def _body_image(seg, table):
    """Per-pixel body index from a raw HxWx2 segmentation image.

    Mirrors label_mask's channel convention (seg[...,0]=object id,
    seg[...,1]=object type). Non-geom and world-body pixels map to 0.
    """
    body_of_geom, body_names, _geom_names = table
    objid = seg[..., 0]
    objtype = seg[..., 1]
    valid = (objid >= 0) & (objtype == int(mujoco.mjtObj.mjOBJ_GEOM))
    safe = np.clip(objid, 0, len(body_of_geom) - 1)
    return np.where(valid, body_of_geom[safe], 0)


def _boxes_from_segmentation(seg, table, bodies, body_to_node, min_pixels=8):
    """Exact union box per visible body, straight from the segmentation."""
    body_of_geom, body_names, _ = table
    body_img = _body_image(seg, table)
    name_to_index = {name: i for i, name in enumerate(body_names)}
    concept_boxes = {}
    for body in bodies:
        index = name_to_index.get(body)
        if index is None:
            continue
        ys, xs = np.nonzero(body_img == index)
        if xs.size < min_pixels:
            continue  # a couple of stray pixels is not a detection
        concept_boxes[body_to_node[body]] = (
            int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        )
    return concept_boxes


def perceive_scene_softcat(
    env, pipeline, kg, category_model, body_to_node, min_confidence=0.45
):
    """Category-classifier perception with the same return shape as the others.

    The classifier's argmax over its softmax is the concept id -- one hard
    symbol per identified object, exactly as a KG node match would give. The
    max probability stands in for the match cosine, so triple ranking in
    assemble_triples_geo works unchanged. Proposals classified as background,
    or below min_confidence, drop out the same way a below-threshold KG match
    does.

    Relations remain geometric: `relations_from_boxes` runs on the per-category
    union boxes, so (lion, inside, cage) is inferred from 2D box containment,
    not read from the classifier or from simulator state.

    Category names and MuJoCo body names coincide (food/lion/cage/water/poison),
    so `body_to_node` from build_oracle_kg indexes both.
    """
    results = []
    for cam_name, frame in zip(CAMERA_NAMES, env.render_panorama()):
        if not frame_maybe_has_objects(frame):
            results.append(({}, {}, {}, []))
            continue
        boxes, _masks, embeddings, _quality = pipeline.detect_full(frame)
        if not boxes:
            results.append(({}, {}, {}, []))
            continue

        probabilities = category_model.probabilities(embeddings)
        confidences, predicted = probabilities.max(dim=-1)

        ids, sims = [], []
        for category_id, confidence in zip(predicted.tolist(), confidences.tolist()):
            name = ID_TO_CATEGORY.get(int(category_id))
            node = body_to_node.get(name)  # background -> None, as a bad match would
            if node is None or confidence < min_confidence:
                ids.append(None)
                sims.append(-1.0)
            else:
                ids.append(node)
                sims.append(float(confidence))

        concept_boxes = union_boxes_by_concept(ids, boxes)
        if not concept_boxes:
            results.append(({}, {}, {}, []))
            continue
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


def perceive_scene_oracle(
    env, pipeline, kg, table, body_to_node, mode="oracle_id", min_coverage=0.5
):
    """Oracle counterpart of perceive_scene_geo, with the same return shape.

    Returns list of (concept_boxes, concept_sims, concept_deltas, relations)
    per camera. concept_sims is 1.0 for every resolved concept: identity is
    certain, so the retrieval-confidence channel carries no doubt.

    `pipeline` is unused in oracle_full mode and may be None there.
    """
    if mode not in ORACLE_MODES:
        raise ValueError(f"mode must be one of {ORACLE_MODES}, got {mode!r}")

    bodies = tuple(body_to_node)
    results = []
    for cam_name, (frame, seg) in zip(
        CAMERA_NAMES, env.render_panorama(with_segmentation=True)
    ):
        if mode == "oracle_full":
            concept_boxes = _boxes_from_segmentation(seg, table, bodies, body_to_node)
        else:
            # keep the SAM early-out so oracle_id sees exactly the frames the
            # real pipeline would bother running on
            if not frame_maybe_has_objects(frame):
                results.append(({}, {}, {}, []))
                continue
            boxes, _embeddings, masks = pipeline.detect(frame)
            ids = []
            for mask in masks:
                body, _geom, _coverage = label_mask(seg, mask, table, min_coverage)
                # BACKGROUND / AMBIGUOUS drop out, mirroring a below-threshold
                # KG match, which also yields no concept
                ids.append(body_to_node.get(body))
            concept_boxes = union_boxes_by_concept(ids, boxes)

        if not concept_boxes:
            results.append(({}, {}, {}, []))
            continue

        concept_sims = {cid: 1.0 for cid in concept_boxes}
        concept_deltas = {
            cid: env.ground_delta(cam_name, (x0 + x1) / 2.0, y1)
            for cid, (x0, y0, x1, y1) in concept_boxes.items()
        }
        relations = relations_from_boxes(concept_boxes, frame.shape[1])
        results.append((concept_boxes, concept_sims, concept_deltas, relations))
    return results
