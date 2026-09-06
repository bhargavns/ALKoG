#!/bin/python3
import sys, os
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.CategoryData import draw_category_boxes, label_proposal, save_json
from lib.KGWorldEnv import CAMERA_NAMES, KinematicKGWorldEnv
from lib.ObjectCategories import (
    ALL_CATEGORY_NAMES,
    CATEGORY_TO_ID,
    active_category_names,
)
from lib.Perception import EMBEDDING_DIM, PerceptionPipeline

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

required_arguments = []
optional_arguments = {
    "out": os.path.join(_OUTPUT, "category_dataset.pt"),
    "candidate_dir": os.path.join(_OUTPUT, "candidates_oracle"),
    "scenes": "200",            # distinct layouts; cage state alternates
    "views_per_scene": "2",     # perception passes per layout
    "candidate_images": "40",   # max annotated debug panels written
    "render_size": "512",
    "min_purity": "0.45",       # min fraction of a proposal one body must cover
    "resources": "0",           # 0 = lion/cage/food/receiver; 1 = adds water/poison
    "device": "cuda",
    "seed": "0",
}

USAGE = """
generate_category_dataset.py -- build the supervised proposal-category dataset.

Renders panoramas together with MuJoCo's segmentation image, runs SAM +
ResNet-18 on each frame, and labels every proposal with the object body
covering most of its mask (or background). The result is the training set for
the soft-category head (see train_category_model.py).

resources=0 (default) is the lion/cage/food world all prior KG results were
produced on, plus the stationary receiving agent; only those four categories can
occur. resources=1 adds water and poison. Either way the stored vocabulary is the
full seven-class set, so a head trained on one world loads against the other --
absent categories simply carry no support, and metrics skip them.

The receiving agent is an ordinary object here: it is rendered, proposed by SAM,
and labelled from the segmentation image like any other body. It does not move
and has no reward; the only question this dataset asks of it is whether the
transmitting agent can tell it apart from food, lion, and cage.

Outputs:
    <out>                   torch bundle: embeddings, labels, group_ids, metadata
    <out>.summary.json      class counts and per-category proposal recall
    <candidate_dir>/        annotated proposal panels for eyeballing labels

    Optional:
        out=.../category_dataset.pt candidate_dir=.../candidates_oracle
        scenes=200 views_per_scene=2 candidate_images=40 render_size=512
        min_purity=0.45 resources=0 device=cuda seed=0

    Example Usage:
        generate_category_dataset.py scenes=400 seed=1
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def generate_dataset():
    out_path = g_ArgParse.get("out")
    candidate_dir = g_ArgParse.get("candidate_dir")
    scenes = int(g_ArgParse.get("scenes"))
    views_per_scene = int(g_ArgParse.get("views_per_scene"))
    candidate_images = int(g_ArgParse.get("candidate_images"))
    render_size = int(g_ArgParse.get("render_size"))
    min_purity = float(g_ArgParse.get("min_purity"))
    resources = g_ArgParse.get("resources") == "1"
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))
    active = active_category_names(resources)

    if scenes < 2:
        raise ValueError("scenes must be at least 2 for a grouped train/validation split")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(candidate_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    env = KinematicKGWorldEnv(render_size=render_size, seed=seed, resources=resources)
    print(f"Env resources={resources}; categories that can occur: {active}")
    print("Loading SAM + ResNet-18 (GPU)...")
    pipeline = PerceptionPipeline(device=device) # SAM proposals -> CNN features + mask color histogram per detection  

    embedding_parts, labels, group_ids, metadata, frame_stats = [], [], [], [], []
    group_id = 0
    images_written = 0
    try:
        for scene in range(scenes): # for each "scene" 
            env.fixed_cage_state = bool(scene % 2 == 0) # if set to None, will randomly make the lion caged or not. Otherwise, if true, then lion is caged  
            env.reset(seed=seed + scene)
            for view in range(views_per_scene): # for each view in the scene 
                rgb_frames, oracle_frames = env.render_panorama_with_labels() # returns aligned RGB frames and oracle category masks for each camera 
                for camera_index, (camera_name, frame, oracle) in enumerate(
                    zip(CAMERA_NAMES, rgb_frames, oracle_frames)
                ):
                    boxes, masks, embeddings, quality = pipeline.detect_full(frame) 
                    frame_labels, purities, ious = [], [], []
                    for proposal_index, (box, mask, proposal_quality) in enumerate(
                        zip(boxes, masks, quality)
                    ):
                        label, purity, iou = label_proposal(mask, oracle, min_purity) # ground truth for SAM proposal
                        frame_labels.append(label)
                        purities.append(purity)
                        ious.append(iou)
                        labels.append(label)
                        group_ids.append(group_id)
                        metadata.append(
                            {
                                "scene": scene,
                                "view": view,
                                "camera": camera_name,
                                "camera_index": camera_index,
                                "proposal_index": proposal_index,
                                "box": [int(v) for v in box],
                                "purity": purity,
                                "iou": iou,
                                **proposal_quality,
                            }
                        )
                    if len(embeddings):
                        embedding_parts.append(embeddings.detach().cpu().float())

                    visible = {}
                    for name in active:
                        category_id = CATEGORY_TO_ID[name]
                        pixels = int((oracle == category_id).sum())
                        visible[name] = {
                            "pixels": pixels,
                            "proposal_recalled": bool(
                                pixels > 0 and any(l == category_id for l in frame_labels)
                            ),
                            "best_purity": max(
                                (p for l, p in zip(frame_labels, purities) if l == category_id),
                                default=0.0,
                            ),
                        }
                    frame_stats.append(
                        {
                            "group_id": group_id,
                            "scene": scene,
                            "view": view,
                            "camera": camera_name,
                            "proposal_count": len(boxes),
                            "visible": visible,
                        }
                    )

                    if boxes and images_written < candidate_images:
                        annotated = draw_category_boxes(
                            frame, boxes, frame_labels, purities, prefix="oracle:"
                        )
                        image_path = os.path.join(
                            candidate_dir,
                            f"scene{scene:04d}_view{view:02d}_{camera_name}.png",
                        )
                        if not cv2.imwrite(image_path, annotated):
                            raise RuntimeError(f"Failed to write {image_path}")
                        images_written += 1
                    group_id += 1

                if view + 1 < views_per_scene:
                    for _ in range(3):
                        _o, _r, terminated, truncated, _i = env.step(
                            int(rng.integers(0, env.action_space.n))
                        )
                        if terminated or truncated:
                            break

            if (scene + 1) % 25 == 0:
                print(f"scene {scene + 1}/{scenes}: {len(labels)} proposals so far")
    finally:
        env.close()

    if not embedding_parts:
        raise RuntimeError("SAM produced no accepted proposals; inspect proposal thresholds")
    embeddings = torch.cat(embedding_parts, dim=0)
    if embeddings.shape[0] != len(labels):
        raise RuntimeError(
            f"Embedding/label mismatch: {embeddings.shape[0]} embeddings vs {len(labels)} labels"
        )

    torch.save(
        {
            "format_version": 1,
            "class_names": list(ALL_CATEGORY_NAMES),
            # which object categories this world can actually produce; the
            # trainer requires proposals for exactly these, not all six
            "active_categories": list(active),
            "resources": resources,
            "embedding_dim": EMBEDDING_DIM,
            "embeddings": embeddings,
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "group_ids": torch.as_tensor(group_ids, dtype=torch.long),
            "metadata": metadata,
            "frame_stats": frame_stats,
            "generator_config": {k: g_ArgParse.get(k) for k in optional_arguments},
        },
        out_path,
    )

    visible_counts = {name: 0 for name in active}
    recalled_counts = visible_counts.copy()
    for frame in frame_stats:
        for name, values in frame["visible"].items():
            if values["pixels"] > 0:
                visible_counts[name] += 1
                recalled_counts[name] += int(values["proposal_recalled"])
    summary = {
        "dataset": out_path,
        "proposals": int(len(labels)),
        "frame_groups": int(group_id),
        "candidate_images": images_written,
        "class_counts": {
            name: int(sum(l == class_id for l in labels))
            for class_id, name in enumerate(ALL_CATEGORY_NAMES)
        },
        "proposal_recall": {
            name: recalled_counts[name] / max(1, visible_counts[name])
            for name in visible_counts
        },
    }
    save_json(out_path + ".summary.json", summary)
    print(f"Wrote dataset {out_path} ({len(labels)} proposals, {group_id} frame groups)")
    print(f"Class counts:     {summary['class_counts']}")
    print(f"Proposal recall:  {summary['proposal_recall']}")
    print(f"Candidate panels: {candidate_dir}")


def main(inputArguments):
    initialize(inputArguments)
    generate_dataset()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
