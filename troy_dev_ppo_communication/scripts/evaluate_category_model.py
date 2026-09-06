#!/bin/python3
import sys, os
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
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

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

required_arguments = []
optional_arguments = {
    "dataset": os.path.join(_OUTPUT, "category_dataset.pt"),
    "model": os.path.join(_OUTPUT, "category_model.pt"),
    "out_dir": os.path.join(_OUTPUT, "category_evaluation"),
    "validation_fraction": "0.2",  # must match the value used at training time
    "batch_size": "512",
    "live_scenes": "20",        # 0 = skip the fresh-rollout holdout
    "prediction_images": "40",
    "min_purity": "0.45",
    "resources": "0",         # must match the world the dataset/model came from
    "device": "cuda",
    "seed": "0",
}

USAGE = """
evaluate_category_model.py -- score the soft-category head.

Two evaluations:
  held-out split   the validation half of the stored dataset, re-split with the
                   same grouped split as training (so validation_fraction and
                   seed must match train_category_model.py)
  live holdout     fresh rollouts at unseen seeds, scored against the MuJoCo
                   oracle -- catches overfitting to the stored proposal set

Outputs confusion_matrix.png, metrics.json, and annotated prediction panels.

    Optional:
        dataset=.../category_dataset.pt model=.../category_model.pt
        out_dir=.../category_evaluation validation_fraction=0.2 batch_size=512
        live_scenes=20 prediction_images=40 min_purity=0.45 resources=0
        device=cuda seed=0

    Example Usage:
        evaluate_category_model.py live_scenes=50
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


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
    return output


def evaluate_category_model():
    dataset_path = g_ArgParse.get("dataset")
    model_path = g_ArgParse.get("model")
    out_dir = g_ArgParse.get("out_dir")
    validation_fraction = float(g_ArgParse.get("validation_fraction"))
    batch_size = int(g_ArgParse.get("batch_size"))
    live_scenes = int(g_ArgParse.get("live_scenes"))
    prediction_images = int(g_ArgParse.get("prediction_images"))
    min_purity = float(g_ArgParse.get("min_purity"))
    resources = g_ArgParse.get("resources") == "1"
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))

    os.makedirs(out_dir, exist_ok=True)
    model, checkpoint = load_category_checkpoint(model_path, device=device)
    bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    _, validation_indices = grouped_split(
        bundle["group_ids"].numpy(), validation_fraction, seed
    )
    embeddings = bundle["embeddings"][validation_indices].float()
    targets = bundle["labels"][validation_indices].numpy()
    groups = bundle["group_ids"][validation_indices].numpy()

    predictions, confidences = predict_embeddings(model, embeddings, device, batch_size)
    metrics = classification_metrics(targets, predictions.numpy())
    metrics["mean_confidence"] = float(confidences.mean())
    metrics["fragmentation"] = proposal_fragmentation(groups, targets, predictions.numpy())
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    plot_path = save_confusion_plot(
        metrics["confusion_matrix"], os.path.join(out_dir, "confusion_matrix.png")
    )
    print(f"Stored-split accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")

    if live_scenes > 0:
        # must match the world the dataset was generated in
        env = KinematicKGWorldEnv(seed=seed, resources=resources)
        print("Loading SAM + ResNet-18 (GPU)...")
        pipeline = PerceptionPipeline(device=device)
        live_targets, live_predictions = [], []
        images_written = 0
        try:
            for scene in range(live_scenes):
                env.fixed_cage_state = bool(scene % 2 == 0)
                # offset well past the dataset's seed range so these are unseen
                env.reset(seed=100_000 + seed + scene)
                rgb_frames, oracle_frames = env.render_panorama_with_labels()
                for camera_name, frame, oracle in zip(CAMERA_NAMES, rgb_frames, oracle_frames):
                    boxes, masks, frame_embeddings, _q = pipeline.detect_full(frame)
                    if not boxes:
                        continue
                    frame_predictions, frame_confidences = predict_embeddings(
                        model, frame_embeddings.cpu(), device, batch_size
                    )
                    frame_targets = [
                        label_proposal(mask, oracle, min_purity)[0] for mask in masks
                    ]
                    live_targets.extend(frame_targets)
                    live_predictions.extend(frame_predictions.tolist())
                    if images_written < prediction_images:
                        annotated = draw_category_boxes(
                            frame,
                            boxes,
                            frame_predictions.tolist(),
                            frame_confidences.tolist(),
                            prefix="pred:",
                        )
                        path = os.path.join(
                            out_dir, f"prediction_scene{scene:04d}_{camera_name}.png"
                        )
                        if not cv2.imwrite(path, annotated):
                            raise RuntimeError(f"Failed to write {path}")
                        images_written += 1
        finally:
            env.close()
        metrics["live_holdout"] = classification_metrics(live_targets, live_predictions)
        metrics["prediction_images"] = images_written
        print(
            f"Live holdout accuracy={metrics['live_holdout']['accuracy']:.4f} "
            f"macro-F1={metrics['live_holdout']['macro_f1']:.4f}"
        )

    save_json(os.path.join(out_dir, "metrics.json"), metrics)
    print(f"Wrote {plot_path}")
    print(f"Wrote {os.path.join(out_dir, 'metrics.json')}")


def main(inputArguments):
    initialize(inputArguments)
    evaluate_category_model()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
