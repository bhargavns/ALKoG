"""
Root ALKoG configuration.

Usage
-----
# Load from YAML (recommended):
cfg = ALKoGConfig.from_yaml("configs/default.yaml")

# Use defaults entirely:
cfg = ALKoGConfig()

# Override a nested field programmatically:
cfg = ALKoGConfig(rewards=RewardConfig(task_completion=2.0))

# Save current config for reproducibility:
cfg.to_yaml("configs/run_001.yaml")

# Resolve compute device:
device = cfg.resolved_device  # -> "cuda", "mps", or "cpu"
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field, computed_field, model_validator

from alkog.config.agent import AgentConfig
from alkog.config.environment import EnvironmentConfig
from alkog.config.kg import KGConfig
from alkog.config.rewards import RewardConfig
from alkog.config.training import TrainingConfig


class WandBConfig(BaseModel):
    enabled: bool = Field(default=False, description="Set True to enable W&B logging.")
    project: str = Field(default="alkog")
    entity: str | None = Field(default=None, description="Your W&B team/username.")
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="")


class ALKoGConfig(BaseModel):
    # ------------------------------------------------------------------
    # Run identity
    # ------------------------------------------------------------------
    project_name: str = Field(default="alkog")
    run_name: str = Field(default="run_001")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description=(
            "'auto' selects CUDA > MPS > CPU in that priority order. "
            "Override to 'cpu' for debugging on any machine."
        ),
    )

    # ------------------------------------------------------------------
    # Sub-configs
    # ------------------------------------------------------------------
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    kg: KGConfig = Field(default_factory=KGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)

    # ------------------------------------------------------------------
    # Computed fields
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[misc]
    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Cross-config validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def check_minibatch_fits_rollout(self) -> "ALKoGConfig":
        total = self.training.ppo.rollout_length * self.training.num_envs
        mb = self.training.ppo.minibatch_size
        if total < mb:
            raise ValueError(
                f"minibatch_size ({mb}) > total rollout data "
                f"({self.training.ppo.rollout_length} * {self.training.num_envs} = {total}). "
                "Lower minibatch_size or increase rollout_length / num_envs."
            )
        return self

    @model_validator(mode="after")
    def check_kg_dims_consistent(self) -> "ALKoGConfig":
        if self.kg.node_embedding_dim != self.agent.visual_encoder.output_dim:
            raise ValueError(
                f"kg.node_embedding_dim ({self.kg.node_embedding_dim}) must equal "
                f"agent.visual_encoder.output_dim ({self.agent.visual_encoder.output_dim}). "
                "Node embeddings are initialised from visual encoder outputs."
            )
        return self

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ALKoGConfig":
        """Load config from a YAML file, merging with defaults."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        """Serialise the full resolved config to YAML for reproducibility."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)
        )

    def __repr__(self) -> str:
        return (
            f"ALKoGConfig("
            f"run='{self.run_name}', "
            f"device='{self.resolved_device}', "
            f"curriculum_phases={len(self.training.curriculum.phases)}, "
            f"vocab_size={self.agent.token_vocab.vocab_size}"
            f")"
        )
