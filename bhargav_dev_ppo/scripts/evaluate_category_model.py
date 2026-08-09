#!/usr/bin/env python3
"""Evaluate proposal classification and write predicted candidate images."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.CategoryData import (
    classification_metrics,
    draw_category_boxes,
    grouped_split,
    label_proposal,
    proposal_fragmentation,
    save_json,
)
from lib.CategoryModel import load_category_checkpoint
from lib.KGWorldEnv import CAMERA_NAMES, KinematicKGWorldEnv
from lib.ObjectCategories import ALL_CATEGORY_NAMES
from lib.Perception import PerceptionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "output" / "category_dataset.pt")
    parser.add_argument("--model", type=Path, default=ROOT / "output" / "category_model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "category_evaluation")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--live-scenes", type=int, default=20)
    parser.add_argument("--prediction-images", type=int, default=40)
    parser.add_argument("--min-purity", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def predict_embeddings(model, embeddings, device, batch_size):
    predictions, confidences = [], []
    for chunk in embeddings.split(batch_size):
        probabilities = model.probabilities(chunk.to(device))
        confidence, prediction = probabilities.max(dim=-1)
        predictions.append(prediction.cpu())
        confidences.append(confidence.cpu())
    return torch.cat(predictions), torch.cat(confidences)


def save_confusion_plot(matrix, output):
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(ALL_CATEGORY_NAMES)), ALL_CATEGORY_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(ALL_CATEGORY_NAMES)), ALL_CATEGORY_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Oracle")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_category_checkpoint(args.model, device=args.device)
    bundle = torch.load(args.dataset, map_location="cpu", weights_only=False)
    _, validation_indices = grouped_split(
        bundle["group_ids"].numpy(), args.validation_fraction, args.seed
    )
    embeddings = bundle["embeddings"][validation_indices].float()
    targets = bundle["labels"][validation_indices].numpy()
    groups = bundle["group_ids"][validation_indices].numpy()
    predictions, confidences = predict_embeddings(model, embeddings, args.device, args.batch_size)
    metrics = classification_metrics(targets, predictions.numpy())
    metrics["mean_confidence"] = float(confidences.mean())
    metrics["fragmentation"] = proposal_fragmentation(
        groups, targets, predictions.numpy()
    )
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    save_confusion_plot(metrics["confusion_matrix"], args.output_dir / "confusion_matrix.png")

    if args.live_scenes > 0:
        env = KinematicKGWorldEnv(seed=args.seed)
        pipeline = PerceptionPipeline(device=args.device)
        live_targets, live_predictions = [], []
        images_written = 0
        try:
            for scene in range(args.live_scenes):
                env.fixed_cage_state = bool(scene % 2 == 0)
                env.reset(seed=100_000 + args.seed + scene)
                rgb_frames, oracle_frames = env.render_panorama_with_labels()
                for camera_name, frame, oracle in zip(CAMERA_NAMES, rgb_frames, oracle_frames):
                    boxes, masks, frame_embeddings, _ = pipeline.detect_full(frame)
                    if not boxes:
                        continue
                    frame_predictions, frame_confidences = predict_embeddings(
                        model, frame_embeddings.cpu(), args.device, args.batch_size
                    )
                    frame_targets = [
                        label_proposal(mask, oracle, args.min_purity)[0] for mask in masks
                    ]
                    live_targets.extend(frame_targets)
                    live_predictions.extend(frame_predictions.tolist())
                    if images_written < args.prediction_images:
                        annotated = draw_category_boxes(
                            frame,
                            boxes,
                            frame_predictions.tolist(),
                            frame_confidences.tolist(),
                            prefix="pred:",
                        )
                        path = args.output_dir / f"prediction_scene{scene:04d}_{camera_name}.png"
                        if not cv2.imwrite(str(path), annotated):
                            raise RuntimeError(f"Failed to write {path}")
                        images_written += 1
        finally:
            env.close()
        metrics["live_holdout"] = classification_metrics(live_targets, live_predictions)
        metrics["prediction_images"] = images_written

    save_json(args.output_dir / "metrics.json", metrics)
    print(
        f"validation accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}; "
        f"wrote {args.output_dir}"
    )


if __name__ == "__main__":
    main()
