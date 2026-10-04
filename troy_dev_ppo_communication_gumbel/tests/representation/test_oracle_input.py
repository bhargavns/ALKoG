"""Oracle inputs must be complete, geometric, and independent of rendering."""
import sys
from pathlib import Path
from types import SimpleNamespace


import numpy as np
import pytest
import torch

from environment.KGWorldEnv import CommunicationKGWorldEnv, POS_SCALE
from representation.SceneVocabulary import build_scene_vocabulary
from representation.SceneVocabulary import scene_bodies
from representation.OracleTransmitterInput import OracleTransmitterInput
from representation.Grounding import TRIPLE_PAD


@pytest.mark.parametrize('colocated', [False, True])
@pytest.mark.parametrize('caged', [False, True])
@pytest.mark.parametrize('resources', [False, True])
def test_oracle_complete_exact_and_no_render(monkeypatch, caged, resources, colocated):
    monkeypatch.setattr('environment.KGWorldEnv.mujoco.Renderer',
                        lambda *a, **kw: SimpleNamespace(close=lambda: None))
    env = CommunicationKGWorldEnv(seed=8, fixed_cage_state=caged,
        resources=resources, colocated=colocated, receiver_yaw_mode='shared')
    def forbidden(*a, **kw):
        raise AssertionError('Oracle input must never render')
    monkeypatch.setattr(env, 'render_panorama', forbidden)
    try:
        vocabulary, nodes = build_scene_vocabulary(scene_bodies(env), device='cpu')
        inp = OracleTransmitterInput(vocabulary, env, nodes, 6)
        previous = None
        for episode in range(2):
            obs, _ = env.reset()
            tx_start = env.data.qpos[:2].copy()
            assert env.receiver_yaw == env.data.qpos[2]
            if not colocated:
                assert not np.allclose(env.receiver_xy, tx_start)
            inp.reset()
            inp.refresh(obs)
            assert inp.triples.shape == (7, 3)
            present = set(inp.triples[:, [0, 2]].flatten().tolist()) - {TRIPLE_PAD}
            assert present == set(nodes.values())
            relation_rows = inp.triples[inp.triples[:, 1] != TRIPLE_PAD].tolist()
            assert relation_rows == ([[nodes['lion'], vocabulary.relation_index('inside'), nodes['cage']]] if caged else [])
            current = env.model.body('food').pos[:2].copy()
            if previous is not None:
                assert not np.array_equal(previous, current)
            previous = current
            # Anchors stay exact after movement without another refresh.
            for action in range(4):
                obs, *_ = env.step(action)
                delta = inp.deltas(obs, env)
                if colocated:
                    np.testing.assert_allclose(delta[-1, 0], 0, atol=1e-6)
                else:
                    np.testing.assert_array_equal(env.data.qpos[:2], tx_start)
                rotation = np.array([[obs[2], obs[3]], [-obs[3], obs[2]]])
                for row, triple in enumerate(inp.triples.tolist()):
                    for col, node in enumerate((triple[0], triple[2])):
                        if node == TRIPLE_PAD:
                            assert torch.count_nonzero(delta[row, col]) == 0
                            continue
                        name = next(name for name, i in nodes.items() if i == node)
                        expected = rotation @ ((env.model.body(name).pos[:2] - env.data.qpos[:2]) / POS_SCALE)
                        np.testing.assert_allclose(delta[row, col], expected, atol=2e-7)
    finally:
        env.close()
