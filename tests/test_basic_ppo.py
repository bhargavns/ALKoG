"""Smoke tests for the minimal PPO trainer.

These tests intentionally run very short training sessions to validate that the
loop executes end-to-end for both Discrete and Box action spaces.
"""

from __future__ import annotations

import logging

from alkog.config.base import ALKoGConfig
from alkog.training.basic_ppo import train_basic_ppo


def _smoke_cfg(num_envs: int, rollout_length: int) -> ALKoGConfig:
    return ALKoGConfig(
        device="cpu",
        wandb={"enabled": False},
        training={
            "num_envs": num_envs,
            "ppo": {
                "rollout_length": rollout_length,
                "n_epochs": 1,
                "minibatch_size": 32,
                "learning_rate": 3e-4,
            },
        },
    )


def test_basic_ppo_cartpole_smoke() -> None:
    cfg = _smoke_cfg(num_envs=2, rollout_length=32)
    summary = train_basic_ppo(
        cfg=cfg,
        env_id="CartPole-v1",
        total_timesteps=128,
        seed=123,
        log=logging.getLogger("tests.basic_ppo.cartpole"),
    )

    assert summary["global_step"] >= 128
    assert "mean_episode_return_100" in summary


def test_basic_ppo_pendulum_smoke() -> None:
    cfg = _smoke_cfg(num_envs=1, rollout_length=64)
    summary = train_basic_ppo(
        cfg=cfg,
        env_id="Pendulum-v1",
        total_timesteps=64,
        seed=321,
        log=logging.getLogger("tests.basic_ppo.pendulum"),
    )

    assert summary["global_step"] >= 64
    assert "mean_episode_return_100" in summary