"""Visual grounding and category preparation without downloaded model weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from perception.CategoryData import grouped_split, label_proposal
from perception.CategoryModel import ProposalCategoryHead, load_category_checkpoint, save_category_checkpoint
from environment.ObjectCategories import BACKGROUND_ID, CATEGORY_TO_ID, NUM_CLASSES
from perception.Perception import EMBEDDING_DIM, PerceptionPipeline
from representation.SceneVocabulary import build_scene_vocabulary
from perception.ScenePerception import perceive_scene_softcat
from communication.train_communication import create_transmitter_input


def test_visual_grounding_relations_memory_and_live_receiver():
    vocabulary, nodes = build_scene_vocabulary(('food', 'lion', 'cage', 'receiver'))
    frame = np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)
    boxes = [(2, 2, 8, 8), (22, 22, 28, 28), (20, 20, 40, 40),
             (48, 48, 56, 56), (0, 0, 63, 63), (9, 9, 11, 11)]
    categories = [CATEGORY_TO_ID[n] for n in ('food', 'lion', 'cage', 'receiver')]
    probabilities = torch.zeros(6, NUM_CLASSES)
    probabilities[range(4), categories] = 1
    probabilities[4, BACKGROUND_ID] = 1
    probabilities[5] = 1 / NUM_CLASSES  # uncertain proposals are excluded
    pipeline = SimpleNamespace(detect_full=lambda image: (boxes, [], torch.zeros(6, EMBEDDING_DIM), []))
    category = SimpleNamespace(probabilities=lambda features: probabilities)
    receiver = np.array([.8, .2])
    env = SimpleNamespace(
        render_panorama=lambda: [frame] * 4,
        ground_delta=lambda camera, x, y: np.array([x / 64, y / 64]),
        receiver_anchor=lambda: receiver,
    )
    scene = lambda: perceive_scene_softcat(env, pipeline, vocabulary, category, nodes)
    per_frame = scene()
    assert set(per_frame[0][0]) == set(nodes.values())
    assert (nodes['lion'], 'inside', nodes['cage']) in per_frame[0][3]
    inp = create_transmitter_input(
        env, settings=SimpleNamespace(perception='softcat', k_triples=6),
        vocabulary=vocabulary, body_to_node=nodes, pipeline=pipeline, category_model=category)
    obs = np.array([0., 0., 1., 0.])
    inp.reset()
    inp.refresh(obs)
    assert inp.triples.shape == (7, 3)
    assert inp.triples[-1].tolist() == [nodes['receiver'], -1, -1]
    assert nodes['receiver'] not in inp.triples[:-1, [0, 2]]
    torch.testing.assert_close(inp.deltas(obs, env)[-1, 0], torch.tensor([.8, .2]))
    receiver[:] = [.4, .6]
    torch.testing.assert_close(inp.deltas(obs, env)[-1, 0], torch.tensor([.4, .6]))
    # An occluded object remains in anchor memory, but episode reset forgets it.
    env.render_panorama = lambda: [np.zeros_like(frame)] * 4
    inp.refresh(obs)
    assert nodes['food'] in inp.triples[:, 0]
    inp.reset()
    inp.refresh(obs)
    assert torch.all(inp.triples[:-1] == -1)


def test_pipeline_combines_frozen_crop_features_and_mask_colors():
    pipeline = PerceptionPipeline.__new__(PerceptionPipeline)
    pipeline.device = 'cpu'
    frame = np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    cnn = torch.zeros(1, 512)
    cnn[0, 0] = 1
    pipeline.proposer = SimpleNamespace(propose_full=lambda _: ([(0, 0, 8, 8)], [mask], [{'area': 64}]))
    pipeline.encoder = SimpleNamespace(encode=lambda *_: cnn)
    boxes, masks, features, quality = pipeline.detect_full(frame)
    assert features.shape == (1, EMBEDDING_DIM)
    torch.testing.assert_close(features.norm(dim=1), torch.ones(1))
    assert len(boxes) == len(masks) == len(quality) == 1


def test_category_labels_and_group_split():
    labels = np.full((4, 4), CATEGORY_TO_ID['food'])
    mask = np.ones_like(labels, dtype=bool)
    assert label_proposal(mask, labels) == (CATEGORY_TO_ID['food'], 1., 1.)
    assert label_proposal(~mask, labels)[0] == BACKGROUND_ID
    groups = np.repeat(np.arange(8), 3)
    train, val = grouped_split(groups)
    assert not set(groups[train]) & set(groups[val])
    assert len(train) + len(val) == len(groups)


def test_category_checkpoint_roundtrip_and_vocabulary_guard(tmp_path):
    model = ProposalCategoryHead().eval()
    path = tmp_path / 'category.pt'
    save_category_checkpoint(path, model, epoch=1, metrics={})
    restored, blob = load_category_checkpoint(path)
    features = torch.randn(3, EMBEDDING_DIM)
    torch.testing.assert_close(model(features), restored(features))
    blob['class_names'] = ['wrong']
    torch.save(blob, path)
    with pytest.raises(ValueError, match='vocabulary mismatch'):
        load_category_checkpoint(path)
