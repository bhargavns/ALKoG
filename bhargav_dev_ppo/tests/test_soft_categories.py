import numpy as np
import pytest
import torch

from lib.CategoryData import classification_metrics, grouped_split, label_proposal
from lib.ObjectCategories import BACKGROUND_ID, CATEGORY_TO_ID, NUM_OBJECT_CATEGORIES
from lib.SoftCategoryGrounding import SoftCategoryMemory
from lib.SoftCategoryPolicy import SoftCategoryActorCritic


def test_label_proposal_uses_object_purity():
    oracle = np.full((8, 8), BACKGROUND_ID, dtype=np.int64)
    oracle[2:6, 2:6] = CATEGORY_TO_ID["cage"]
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    label, purity, iou = label_proposal(mask, oracle)
    assert label == CATEGORY_TO_ID["cage"]
    assert purity == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)


def test_grouped_split_has_no_frame_leakage():
    groups = np.repeat(np.arange(20), 3)
    train, validation = grouped_split(groups, validation_fraction=0.2, seed=7)
    assert set(groups[train]).isdisjoint(set(groups[validation]))


def test_soft_memory_rotates_world_delta_to_agent_frame():
    memory = SoftCategoryMemory(max_age=10)
    memory.update(
        {0: {"delta": np.array([1.0, 0.0]), "presence": 0.9, "confidence": 0.8}},
        np.zeros(2),
        agent_xy=np.zeros(2),
    )
    # Agent faces +y: world +x is to its right, hence left coordinate -1.
    slots, _ = memory.features(np.array([0.0, 0.0, 0.0, 1.0]))
    assert slots[0, 4] == pytest.approx(0.0)
    assert slots[0, 5] == pytest.approx(-1.0)


def test_policy_training_and_inference_shapes():
    model = SoftCategoryActorCritic()
    obs = torch.zeros(4, 4)
    slots = torch.zeros(4, NUM_OBJECT_CATEGORIES, 6)
    slots[:, :, 0] = 1.0
    relations = torch.zeros(4, 2)
    actions = torch.zeros(4, dtype=torch.long)
    logp, entropy, values = model.evaluate(obs, slots, relations, actions)
    assert logp.shape == entropy.shape == values.shape == (4,)
    loss = -(logp.mean() + 0.01 * entropy.mean()) + values.square().mean()
    loss.backward()
    assert model.category_symbols.grad is not None


def test_metrics_perfect_predictions():
    targets = np.arange(6)
    metrics = classification_metrics(targets, targets)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
