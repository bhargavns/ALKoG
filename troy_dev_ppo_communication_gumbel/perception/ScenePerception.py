"""Ground visual category predictions into scene boxes, relations and offsets."""
from environment.KGWorldEnv import CAMERA_NAMES
from environment.ObjectCategories import ID_TO_CATEGORY
from perception.Perception import frame_maybe_has_objects, relations_from_boxes, union_boxes_by_concept


def perceive_scene_softcat(
    env, pipeline, vocabulary, category_model, body_to_node, min_confidence=0.45
):
    """Convert visual proposals into per-camera boxes, confidence, offsets, and relations.

    The classifier's argmax over its softmax is the concept id -- one hard
    symbol per identified object, for the symbolic triple encoder. The
    maximum probability ranks triples. Background and low-confidence proposals
    are excluded.

    Relations remain geometric: `relations_from_boxes` runs on the per-category
    union boxes, so (lion, inside, cage) is inferred from 2D box containment,
    not read from the classifier or from simulator state.

    Category names and MuJoCo body names coincide (food/lion/cage/receiver/water/poison),
    so `body_to_node` from build_scene_vocabulary indexes both.
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
