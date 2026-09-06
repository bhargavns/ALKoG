#!/bin/python3
import sys, os
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.CategoryData import classification_metrics, grouped_split, save_json
from lib.CategoryModel import ProposalCategoryHead, save_category_checkpoint
from lib.ObjectCategories import (
    ALL_CATEGORY_NAMES,
    BACKGROUND_NAME,
    CATEGORY_TO_ID,
    NUM_CLASSES,
    OBJECT_CATEGORY_NAMES,
    validate_category_metadata,
)

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

required_arguments = []
optional_arguments = {
    "dataset": os.path.join(_OUTPUT, "category_dataset.pt"),
    "out": os.path.join(_OUTPUT, "category_model.pt"),
    "epochs": "30",
    "batch_size": "256",
    "hidden_dim": "128",
    "lr": "1e-3",
    "weight_decay": "1e-4",
    "validation_fraction": "0.2",  # split is grouped by frame, not by proposal
    "device": "cuda",
    "seed": "0",
}

USAGE = """
train_category_model.py -- train the soft-category head on proposal embeddings.

Fits ProposalCategoryHead over the frozen SAM+ResNet embeddings produced by
generate_category_dataset.py. The train/validation split is grouped by frame,
so proposals from the same rendered frame never straddle the split (proposals
within a frame are highly correlated and would inflate validation scores).

Class weights are inverse-frequency over the training split, because
background proposals vastly outnumber object proposals.

The best macro-F1 epoch is checkpointed.

    Optional:
        dataset=.../category_dataset.pt out=.../category_model.pt
        epochs=30 batch_size=256 hidden_dim=128 lr=1e-3 weight_decay=1e-4
        validation_fraction=0.2 device=cuda seed=0

    Example Usage:
        train_category_model.py epochs=60 lr=5e-4
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
def evaluate(model, embeddings, labels, indices, device, batch_size):
    model.eval()
    predictions = []
    for chunk in torch.as_tensor(indices).split(batch_size):
        logits = model(embeddings[chunk].to(device))
        predictions.append(logits.argmax(dim=-1).cpu())
    predictions = torch.cat(predictions).numpy()
    targets = labels[indices].numpy()
    return classification_metrics(targets, predictions), predictions


def train_category_model():
    dataset_path = g_ArgParse.get("dataset")
    out_path = g_ArgParse.get("out")
    epochs = int(g_ArgParse.get("epochs"))
    batch_size = int(g_ArgParse.get("batch_size"))
    hidden_dim = int(g_ArgParse.get("hidden_dim"))
    lr = float(g_ArgParse.get("lr"))
    weight_decay = float(g_ArgParse.get("weight_decay"))
    validation_fraction = float(g_ArgParse.get("validation_fraction"))
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))

    torch.manual_seed(seed)
    np.random.seed(seed)

    bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    validate_category_metadata(bundle["class_names"])
    embeddings = bundle["embeddings"].float()
    labels = bundle["labels"].long()
    group_ids = bundle["group_ids"].long().numpy()
    if len(embeddings) != len(labels) or len(labels) != len(group_ids):
        raise ValueError("Dataset embeddings, labels, and group IDs must have equal lengths")
    print(f"Loaded {dataset_path}: {len(labels)} proposals, {len(set(group_ids))} frame groups")

    class_counts = torch.bincount(labels, minlength=NUM_CLASSES)
    # Only categories the generating world could produce are required. A
    # resources=0 dataset legitimately has no water/poison; demanding all six
    # object classes would make the base world untrainable. Older bundles
    # without the key fall back to the full object set.
    required = list(bundle.get("active_categories", OBJECT_CATEGORY_NAMES)) + [BACKGROUND_NAME]
    missing = [n for n in required if int(class_counts[CATEGORY_TO_ID[n]]) == 0]
    if missing:
        raise ValueError(
            f"Dataset has no proposals for {missing}. Generate more scenes or "
            "lower min_purity in generate_category_dataset.py."
        )
    print(f"Categories this dataset can contain: {required}")

    train_indices, validation_indices = grouped_split(
        group_ids, validation_fraction=validation_fraction, seed=seed
    )
    train_labels = labels[train_indices]
    train_counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    class_weights = train_labels.numel() / (NUM_CLASSES * train_counts.clamp_min(1.0))
    print(f"Train {len(train_indices)} / validation {len(validation_indices)} proposals")
    print(f"Class counts: {dict(zip(ALL_CATEGORY_NAMES, class_counts.tolist()))}")

    model = ProposalCategoryHead(
        embedding_dim=int(bundle["embedding_dim"]), hidden_dim=hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(embeddings[train_indices], train_labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_embeddings, batch_labels in loader:
            logits = model(batch_embeddings.to(device))
            loss = F.cross_entropy(
                logits, batch_labels.to(device), weight=class_weights.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        metrics, _ = evaluate(
            model, embeddings, labels, validation_indices, device, batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}: loss={row['train_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.4f} "
            f"val_macro_f1={row['validation_macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_category_checkpoint(
                out_path,
                model,
                epoch=epoch,
                metrics=metrics,
                extra={
                    "dataset": dataset_path,
                    "seed": seed,
                    "train_groups": int(len(np.unique(group_ids[train_indices]))),
                    "validation_groups": int(len(np.unique(group_ids[validation_indices]))),
                },
            )

    save_json(out_path + ".history.json", history)
    print(f"Saved best checkpoint to {out_path} (validation macro-F1={best_f1:.4f})")


def main(inputArguments):
    initialize(inputArguments)
    train_category_model()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
