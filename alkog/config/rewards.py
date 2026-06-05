"""
Reward weights for the ALKoG RL framework.

All values are placeholders from the proposal slide; tune these first
before touching architecture hyperparameters.

Positive signals encourage the agent to:
  - Complete tasks (primary objective)
  - Communicate effectively with a partner
  - Form accurate predictions about interaction outcomes
  - Discover genuinely new, reusable abstractions

Penalties discourage:
  - Wasting time (step cost)
  - Dangerous or wrong interactions
  - Emitting tokens that mislead the partner
  - Redundant nodes (duplicate entries bloat the KG)
  - KG edges that contradict observed physics
"""

from pydantic import BaseModel, Field, model_validator


class RewardConfig(BaseModel):
    # ------------------------------------------------------------------
    # Positive signals
    # ------------------------------------------------------------------
    task_completion: float = Field(
        default=1.0,
        ge=0.0,
        description="Reward when the episode objective is completed.",
    )
    communication_success: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Reward when the partner agent takes the correct action after "
            "receiving an utterance from this agent."
        ),
    )
    predictive_accuracy: float = Field(
        default=0.2,
        ge=0.0,
        description=(
            "Reward when the agent correctly predicts the outcome of "
            "interacting with a bounded object before doing so."
        ),
    )
    novel_abstraction: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Reward when a newly proposed bounding box (KG node) is "
            "successfully reused in at least one future episode. "
            "This is awarded retroactively on first reuse."
        ),
    )

    # ------------------------------------------------------------------
    # Penalties (stored as negative floats for clarity at call sites)
    # ------------------------------------------------------------------
    time_step_cost: float = Field(
        default=-0.01,
        le=0.0,
        description="Applied every timestep; encourages efficiency.",
    )
    failed_interaction: float = Field(
        default=-0.3,
        le=0.0,
        description="Penalty for touching a dangerous object or falling.",
    )
    communication_failure: float = Field(
        default=-0.2,
        le=0.0,
        description="Penalty when the partner takes the wrong action after an utterance.",
    )
    redundant_abstraction: float = Field(
        default=-0.1,
        le=0.0,
        description=(
            "Penalty when a newly proposed node embedding is too similar "
            "to an existing node (cosine similarity above kg.redundancy_threshold)."
        ),
    )
    kg_inconsistency: float = Field(
        default=-0.15,
        le=0.0,
        description=(
            "Penalty when a KG-predicted relationship is contradicted by "
            "an observed interaction outcome."
        ),
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def positive_rewards_are_positive(self) -> "RewardConfig":
        positives = [
            ("task_completion", self.task_completion),
            ("communication_success", self.communication_success),
            ("predictive_accuracy", self.predictive_accuracy),
            ("novel_abstraction", self.novel_abstraction),
        ]
        for name, val in positives:
            if val < 0:
                raise ValueError(f"Reward '{name}' must be non-negative, got {val}")
        return self

    @model_validator(mode="after")
    def penalties_are_negative(self) -> "RewardConfig":
        penalties = [
            ("time_step_cost", self.time_step_cost),
            ("failed_interaction", self.failed_interaction),
            ("communication_failure", self.communication_failure),
            ("redundant_abstraction", self.redundant_abstraction),
            ("kg_inconsistency", self.kg_inconsistency),
        ]
        for name, val in penalties:
            if val > 0:
                raise ValueError(f"Penalty '{name}' must be non-positive, got {val}")
        return self
