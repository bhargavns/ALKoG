"""Use real MuJoCo state/movement, without constructing a graphics context."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from lib.KGWorldEnv import CommunicationKGWorldEnv


@pytest.fixture(autouse=True)
def no_renderer(monkeypatch):
    monkeypatch.setattr('lib.KGWorldEnv.mujoco.Renderer',
                        lambda *args, **kwargs: SimpleNamespace(close=lambda: None))


def test_shared_yaw_preserves_seeded_layout_stream():
    independent = CommunicationKGWorldEnv(seed=21)
    shared = CommunicationKGWorldEnv(seed=21, receiver_yaw_mode='shared')
    try:
        yaws = []
        for _ in range(6):
            obs_i, info_i = independent.reset()
            obs_s, info_s = shared.reset()
            np.testing.assert_array_equal(obs_i, obs_s)
            np.testing.assert_array_equal(independent.model.body_pos, shared.model.body_pos)
            assert info_i == info_s
            assert independent.rng.bit_generator.state == shared.rng.bit_generator.state
            assert shared.receiver_yaw == shared.data.qpos[2]
            assert independent.receiver_yaw != independent.data.qpos[2]
            yaws.append(shared.receiver_yaw)
        assert len(set(yaws)) == len(yaws)  # transmitter orientation remains randomized
    finally:
        independent.close()
        shared.close()


@pytest.mark.parametrize('action', range(4))
def test_shared_actions_follow_transmitter_frame_and_yaw_stays_fixed(action):
    env = CommunicationKGWorldEnv(seed=9, receiver_yaw_mode='shared')
    try:
        obs, _ = env.reset()
        yaw = env.receiver_yaw
        transmitter_pos = env.data.qpos.copy()
        # Move from the center so arena clipping cannot affect the assertion.
        env.model.body('receiver').pos[:2] = 0.
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        expected = (forward, -forward, left, -left)[action] * env.step_size
        next_obs, *_ = env.step(action)
        np.testing.assert_allclose(env.receiver_xy, expected, atol=1e-12)
        np.testing.assert_array_equal(env.data.qpos, transmitter_pos)
        np.testing.assert_array_equal(next_obs, obs)
        assert env.receiver_yaw == yaw
    finally:
        env.close()


def test_independent_default_and_invalid_mode():
    default = CommunicationKGWorldEnv(seed=8)
    explicit = CommunicationKGWorldEnv(seed=8, receiver_yaw_mode='independent')
    try:
        for _ in range(3):
            default.reset()
            explicit.reset()
            assert default.receiver_yaw == explicit.receiver_yaw
            np.testing.assert_array_equal(default.data.qpos, explicit.data.qpos)
    finally:
        default.close()
        explicit.close()
    with pytest.raises(ValueError, match='receiver_yaw_mode'):
        CommunicationKGWorldEnv(receiver_yaw_mode='invalid')
