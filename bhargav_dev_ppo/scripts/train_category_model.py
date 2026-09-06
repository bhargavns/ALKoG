#!/usr/bin/env python3
"""Train the supervised proposal-category head on generated embeddings."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib.CategoryData import classification_metrics, grouped_split, save_json
from lib.CategoryModel import ProposalCategoryHead, save_category_checkpoint
from lib.ObjectCategories import ALL_CATEGORY_NAMES, NUM_CLASSES, validate_category_metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "output" / "category_dataset.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "category_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bundle = torch.load(args.dataset, map_location="cpu", weights_only=False)
    validate_category_metadata(bundle["class_names"])
    embeddings = bundle["embeddings"].float()
    labels = bundle["labels"].long()
    group_ids = bundle["group_ids"].long().numpy()
    if len(embeddings) != len(labels) or len(labels) != len(group_ids):
        raise ValueError("Dataset embeddings, labels, and group IDs must have equal lengths")

    class_counts = torch.bincount(labels, minlength=NUM_CLASSES)
    missing = [ALL_CATEGORY_NAMES[i] for i, count in enumerate(class_counts) if int(count) == 0]
    if missing:
        raise ValueError(
            f"Dataset has no proposals for {missing}. Generate more scenes or lower --min-purity."
        )
    train_indices, validation_indices = grouped_split(
        group_ids, validation_fraction=args.validation_fraction, seed=args.seed
    )
    train_labels = labels[train_indices]
    train_counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    class_weights = train_labels.numel() / (NUM_CLASSES * train_counts.clamp_min(1.0))

    model = ProposalCategoryHead(
        embedding_dim=int(bundle["embedding_dim"]), hidden_dim=args.hidden_dim
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(embeddings[train_indices], train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch_embeddings, batch_labels in loader:
            logits = model(batch_embeddings.to(args.device))
            loss = F.cross_entropy(
                logits, batch_labels.to(args.device), weight=class_weights.to(args.device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        metrics, _ = evaluate(
            model, embeddings, labels, validation_indices, args.device, args.batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        print(row)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_category_checkpoint(
                args.output,
                model,
                epoch=epoch,
                metrics=metrics,
                extra={
                    "dataset": str(args.dataset),
                    "seed": args.seed,
                    "train_groups": int(len(np.unique(group_ids[train_indices]))),
                    "validation_groups": int(len(np.unique(group_ids[validation_indices]))),
                },
            )

    save_json(args.output.with_suffix(".history.json"), history)
    print(f"Saved best checkpoint to {args.output} (validation macro-F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
