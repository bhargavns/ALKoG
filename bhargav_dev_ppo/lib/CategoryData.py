"""Dataset labeling, metrics, and visualization for proposal categories."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.ObjectCategories import (
    ALL_CATEGORY_NAMES,
    BACKGROUND_ID,
    CATEGORY_COLORS,
    NUM_CLASSES,
    NUM_OBJECT_CATEGORIES,
)


def label_proposal(mask, oracle_labels, min_purity=0.45):
    """Assign the object occupying most proposal pixels, or background.

    Returns ``(label, purity, iou)``.  Purity handles proposals that cover one
    small visible cage bar well; IoU remains useful as a stricter diagnostic.
    """
    area = int(mask.sum())
    if area == 0:
        return BACKGROUND_ID, 0.0, 0.0
    best_label, best_purity, best_iou = BACKGROUND_ID, 0.0, 0.0
    for category_id in range(NUM_OBJECT_CATEGORIES):
        target = oracle_labels == category_id
        intersection = int(np.logical_and(mask, target).sum())
        purity = intersection / area
        union = int(np.logical_or(mask, target).sum())
        iou = intersection / union if union else 0.0
        if purity > best_purity:
            best_label, best_purity, best_iou = category_id, purity, iou
    if best_purity < min_purity:
        best_label = BACKGROUND_ID
    return int(best_label), float(best_purity), float(best_iou)


def grouped_split(group_ids, validation_fraction=0.2, seed=0):
    """Split whole rendered frames so proposals from one image cannot leak."""
    groups = np.asarray(group_ids, dtype=np.int64)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * validation_fraction)))
    val_groups = set(int(v) for v in unique[:n_val])
    val = np.array([int(g) in val_groups for g in groups], dtype=bool)
    train = ~val
    if not train.any() or not val.any():
        raise ValueError("Dataset needs at least two rendered-frame groups")
    return np.flatnonzero(train), np.flatnonzero(val)


def confusion_matrix(targets, predictions, num_classes=NUM_CLASSES):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classification_metrics(targets, predictions, num_classes=NUM_CLASSES):
    matrix = confusion_matrix(targets, predictions, num_classes=num_classes)
    rows = {}
    f1_values = []
    for class_id, name in enumerate(ALL_CATEGORY_NAMES):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        support = int(matrix[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)
    accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": rows,
        "confusion_matrix": matrix.tolist(),
    }


def proposal_fragmentation(group_ids, targets, predictions):
    """Count proposal and predicted-category fragments for each visible object."""
    stats = {}
    for group_id in np.unique(group_ids):
        select = np.asarray(group_ids) == group_id
        group_targets = np.asarray(targets)[select]
        group_predictions = np.asarray(predictions)[select]
        for category_id in range(NUM_OBJECT_CATEGORIES):
            proposal_count = int((group_targets == category_id).sum())
            if proposal_count == 0:
                continue
            correct_count = int(
                np.logical_and(group_targets == category_id, group_predictions == category_id).sum()
            )
            key = ALL_CATEGORY_NAMES[category_id]
            stats.setdefault(key, []).append(
                {"group_id": int(group_id), "proposals": proposal_count, "correct": correct_count}
            )
    summary = {}
    for name, values in stats.items():
        summary[name] = {
            "frames": len(values),
            "mean_oracle_fragments": float(np.mean([v["proposals"] for v in values])),
            "mean_correct_fragments": float(np.mean([v["correct"] for v in values])),
        }
    return summary


def draw_category_boxes(frame, boxes, category_ids, confidences, prefix=""):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    for box, category_id, confidence in zip(boxes, category_ids, confidences):
        x0, y0, x1, y1 = (int(v) for v in box)
        rgb = CATEGORY_COLORS[int(category_id)]
        bgr = tuple(reversed(rgb))
        cv2.rectangle(image, (x0, y0), (x1, y1), bgr, 2)
        text = f"{prefix}{ALL_CATEGORY_NAMES[int(category_id)]} {float(confidence):.2f}"
        cv2.putText(
            image,
            text,
            (x0 + 2, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            bgr,
            1,
            cv2.LINE_AA,
        )
    return image


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
