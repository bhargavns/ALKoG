#!/usr/bin/env python3
"""Generate SAM proposals, oracle labels, and candidate debug images."""

import argparse
import os
import sys
from pathlib import Path

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from lib.CategoryData import draw_category_boxes, label_proposal, save_json
from lib.KGWorldEnv import CAMERA_NAMES, KinematicKGWorldEnv
from lib.ObjectCategories import ALL_CATEGORY_NAMES, NUM_OBJECT_CATEGORIES
from lib.Perception import EMBEDDING_DIM, PerceptionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "category_dataset.pt")
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "output" / "candidates_oracle")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--views-per-scene", type=int, default=2)
    parser.add_argument("--candidate-images", type=int, default=40)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--min-purity", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scenes < 2:
        raise ValueError("--scenes must be at least 2 for a grouped train/validation split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env = KinematicKGWorldEnv(render_size=args.render_size, seed=args.seed)
    pipeline = PerceptionPipeline(device=args.device)

    embedding_parts, labels, group_ids, metadata, frame_stats = [], [], [], [], []
    group_id = 0
    images_written = 0
    try:
        for scene in range(args.scenes):
            env.fixed_cage_state = bool(scene % 2 == 0)
            env.reset(seed=args.seed + scene)
            for view in range(args.views_per_scene):
                rgb_frames, oracle_frames = env.render_panorama_with_labels()
                for camera_index, (camera_name, frame, oracle) in enumerate(
                    zip(CAMERA_NAMES, rgb_frames, oracle_frames)
                ):
                    boxes, masks, embeddings, quality = pipeline.detect_full(frame)
                    frame_labels, purities, ious = [], [], []
                    for proposal_index, (box, mask, proposal_quality) in enumerate(
                        zip(boxes, masks, quality)
                    ):
                        label, purity, iou = label_proposal(mask, oracle, args.min_purity)
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
                    for category_id, name in enumerate(ALL_CATEGORY_NAMES[:NUM_OBJECT_CATEGORIES]):
                        pixels = int((oracle == category_id).sum())
                        visible[name] = {
                            "pixels": pixels,
                            "proposal_recalled": bool(
                                pixels > 0 and any(label == category_id for label in frame_labels)
                            ),
                            "best_purity": max(
                                (purity for label, purity in zip(frame_labels, purities) if label == category_id),
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

                    if boxes and images_written < args.candidate_images:
                        annotated = draw_category_boxes(frame, boxes, frame_labels, purities, prefix="oracle:")
                        output_path = args.candidate_dir / (
                            f"scene{scene:04d}_view{view:02d}_{camera_name}.png"
                        )
                        if not cv2.imwrite(str(output_path), annotated):
                            raise RuntimeError(f"Failed to write {output_path}")
                        images_written += 1
                    group_id += 1

                if view + 1 < args.views_per_scene:
                    for _ in range(3):
                        _obs, _reward, terminated, truncated, _info = env.step(
                            int(rng.integers(0, env.action_space.n))
                        )
                        if terminated or truncated:
                            break
    finally:
        env.close()

    if not embedding_parts:
        raise RuntimeError("SAM produced no accepted proposals; inspect proposal thresholds")
    embeddings = torch.cat(embedding_parts, dim=0)
    if embeddings.shape[0] != len(labels):
        raise RuntimeError(
            f"Embedding/label mismatch: {embeddings.shape[0]} embeddings vs {len(labels)} labels"
        )
    bundle = {
        "format_version": 1,
        "class_names": list(ALL_CATEGORY_NAMES),
        "embedding_dim": EMBEDDING_DIM,
        "embeddings": embeddings,
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "group_ids": torch.as_tensor(group_ids, dtype=torch.long),
        "metadata": metadata,
        "frame_stats": frame_stats,
        "generator_config": vars(args) | {"output": str(args.output), "candidate_dir": str(args.candidate_dir)},
    }
    torch.save(bundle, args.output)

    visible_counts = {name: 0 for name in ALL_CATEGORY_NAMES[:NUM_OBJECT_CATEGORIES]}
    recalled_counts = visible_counts.copy()
    for frame in frame_stats:
        for name, values in frame["visible"].items():
            if values["pixels"] > 0:
                visible_counts[name] += 1
                recalled_counts[name] += int(values["proposal_recalled"])
    summary = {
        "dataset": str(args.output),
        "proposals": int(len(labels)),
        "frame_groups": int(group_id),
        "candidate_images": images_written,
        "class_counts": {
            name: int(sum(label == class_id for label in labels))
            for class_id, name in enumerate(ALL_CATEGORY_NAMES)
        },
        "proposal_recall": {
            name: recalled_counts[name] / max(1, visible_counts[name])
            for name in visible_counts
        },
    }
    save_json(args.output.with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
