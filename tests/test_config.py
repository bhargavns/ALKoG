"""
Tests for the ALKoG config system.

These run without any GPU or Godot dependency — just Python + Pydantic.
"""

import pytest
import yaml

from alkog.config import (
    ALKoGConfig,
    CurriculumPhase,
    EdgeType,
    KGConfig,
    PPOConfig,
    RewardConfig,
)


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

class TestRewardConfig:
    def test_defaults_are_sensible(self):
        r = RewardConfig()
        assert r.task_completion > 0
        assert r.time_step_cost < 0
        assert r.failed_interaction < 0

    def test_positive_reward_must_be_non_negative(self):
        with pytest.raises(ValueError, match="task_completion"):
            RewardConfig(task_completion=-1.0)

    def test_penalty_must_be_non_positive(self):
        with pytest.raises(ValueError, match="time_step_cost"):
            RewardConfig(time_step_cost=0.5)


# ---------------------------------------------------------------------------
# KGConfig
# ---------------------------------------------------------------------------

class TestKGConfig:
    def test_edge_type_enum(self):
        assert EdgeType.SPATIAL == "spatial"
        assert EdgeType.CAUSAL == "causal"

    def test_redundancy_threshold_bounds(self):
        with pytest.raises(Exception):
            KGConfig(redundancy_similarity_threshold=1.5)


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------

class TestPPOConfig:
    def test_gamma_bounds(self):
        with pytest.raises(Exception):
            PPOConfig(gamma=1.5)


# ---------------------------------------------------------------------------
# ALKoGConfig
# ---------------------------------------------------------------------------

class TestALKoGConfig:
    def test_defaults_construct(self):
        cfg = ALKoGConfig()
        assert cfg.project_name == "alkog"
        assert len(cfg.training.curriculum.phases) == 4

    def test_resolved_device_is_string(self):
        cfg = ALKoGConfig(device="cpu")
        assert cfg.resolved_device == "cpu"

    def test_kg_dim_must_match_encoder_output(self):
        with pytest.raises(ValueError, match="node_embedding_dim"):
            ALKoGConfig.model_validate({
                "kg": {"node_embedding_dim": 128},
                "agent": {"visual_encoder": {"output_dim": 256}},
            })

    def test_minibatch_size_validation(self):
        # rollout_length=8, num_envs=1 → total=8; minibatch=64 should fail
        with pytest.raises(ValueError, match="minibatch_size"):
            ALKoGConfig(
                training={
                    "ppo": {"rollout_length": 8, "minibatch_size": 64},
                    "num_envs": 1,
                }
            )

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ALKoGConfig()
        path = tmp_path / "config.yaml"
        cfg.to_yaml(path)
        loaded = ALKoGConfig.from_yaml(path)
        assert loaded.run_name == cfg.run_name
        assert loaded.rewards.task_completion == cfg.rewards.task_completion

    def test_yaml_partial_override(self, tmp_path):
        partial = {"run_name": "my_experiment", "rewards": {"task_completion": 2.0}}
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump(partial))
        cfg = ALKoGConfig.from_yaml(path)
        assert cfg.run_name == "my_experiment"
        assert cfg.rewards.task_completion == 2.0
        # Unset fields still have defaults
        assert cfg.rewards.time_step_cost == -0.01


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class TestCurriculum:
    def test_four_phases(self):
        cfg = ALKoGConfig()
        phases = cfg.training.curriculum.phases
        assert [p.phase_id for p in phases] == [1, 2, 3, 4]

    def test_phase_names(self):
        cfg = ALKoGConfig()
        names = [p.name for p in cfg.training.curriculum.phases]
        assert "solo_exploration" in names
        assert "compositional_reasoning" in names

    def test_partner_phases_have_two_agents(self):
        cfg = ALKoGConfig()
        partner_phases = [p for p in cfg.training.curriculum.phases if p.phase_id >= 3]
        assert all(p.num_agents == 2 for p in partner_phases)
