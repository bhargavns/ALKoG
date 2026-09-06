"""Trainable semantic category head over frozen SAM/ResNet proposals."""

from pathlib import Path

import torch
import torch.nn as nn

from lib.ObjectCategories import ALL_CATEGORY_NAMES, NUM_CLASSES, validate_category_metadata
from lib.Perception import EMBEDDING_DIM


class ProposalCategoryHead(nn.Module):
    """Small classifier used for the first soft-category experiment.

    Keeping the visual encoders frozen makes failures easy to attribute.  A
    later experiment can replace this head's input with a trainable SAM mask
    decoder without changing the fixed semantic vocabulary or PPO slots.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, embeddings, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return self.net(embeddings) / temperature

    def probabilities(self, embeddings, temperature=1.0):
        return self(embeddings, temperature=temperature).softmax(dim=-1)


def save_category_checkpoint(path, model, *, epoch, metrics, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "class_names": list(ALL_CATEGORY_NAMES),
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "extra": dict(extra or {}),
        },
        path,
    )


def load_category_checkpoint(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    validate_category_metadata(blob["class_names"])
    model = ProposalCategoryHead(
        embedding_dim=int(blob["embedding_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        num_classes=len(blob["class_names"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob
