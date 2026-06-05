"""
Training configuration: PPO hyperparameters and the 4-phase curriculum.

The curriculum mirrors the proposal exactly:
  Phase 1 — Solo Exploration       (2-3 objects, navigation rewards only)
  Phase 2 — Affordance Discovery   (5-8 objects, interaction rewards)
  Phase 3 — Partner Communication  (5-8 objects, 2 agents, referential games)
  Phase 4 — Compositional Reasoning (8-15 objects, 2 agents, relational tasks)

Phase advancement is gated by task_success_threshold: the rolling mean
success rate over the last eval window must exceed this before the next
phase begins.  This prevents the curriculum from advancing prematurely
if the agent got lucky for a few episodes.

PPO notes:
  - clip_epsilon = 0.2 is the standard starting point.
  - entropy_coef drives exploration; keep it non-trivial in Phases 1-2
    to encourage vocabulary exploration, then lower it in later phases.
  - The GRPO migration path is: swap the PPOTrainer class for a
    GRPOTrainer that wraps the same env interface and reward signals.
    Nothing in this config file needs to change.
"""

from pydantic import BaseModel, Field


class PPOConfig(BaseModel):
    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        description="Adam learning rate for both actor and critic.",
    )
    gamma: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="Discount factor.",
    )
    gae_lambda: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="GAE lambda for advantage estimation.",
    )
    clip_epsilon: float = Field(
        default=0.2,
        gt=0.0,
        description="PPO clipping parameter.",
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Entropy bonus coefficient.  Higher values encourage more "
            "exploration.  Consider starting at 0.05 for Phases 1-2 to "
            "encourage vocabulary exploration."
        ),
    )
    value_loss_coef: float = Field(
        default=0.5,
        gt=0.0,
        description="Weight of the critic loss relative to the actor loss.",
    )
    max_grad_norm: float = Field(
        default=0.5,
        gt=0.0,
        description="Gradient clipping norm.",
    )
    n_epochs: int = Field(
        default=4,
        gt=0,
        description="Number of PPO update epochs per rollout.",
    )
    minibatch_size: int = Field(
        default=64,
        gt=0,
        description="Minibatch size within each PPO epoch.",
    )
    rollout_length: int = Field(
        default=2048,
        gt=0,
        description=(
            "Number of environment steps collected before each PPO update. "
            "Total data per update = rollout_length * num_envs."
        ),
    )
    lr_anneal: bool = Field(
        default=True,
        description="Linearly anneal learning_rate to 0 over total_timesteps.",
    )


class CurriculumPhase(BaseModel):
    name: str
    phase_id: int = Field(ge=1, le=4)

    # Scene complexity
    min_objects: int = Field(gt=0)
    max_objects: int = Field(gt=0)
    num_agents: int = Field(default=1, gt=0)
    max_steps_per_episode: int = Field(gt=0)

    # What task types are active in this phase
    task_types: list[str] = Field(
        description=(
            "Task identifiers the environment will sample from. "
            "Must match keys registered in the task registry."
        ),
    )

    # Advancement gate
    min_episodes_before_advance: int = Field(
        default=500,
        gt=0,
        description="Minimum episodes to run before checking the advancement condition.",
    )
    task_success_threshold: float = Field(
        default=0.65,
        gt=0.0,
        le=1.0,
        description=(
            "Rolling mean task success rate (over the last eval window) that "
            "must be reached before advancing to the next phase."
        ),
    )


class CurriculumConfig(BaseModel):
    phases: list[CurriculumPhase] = Field(
        default_factory=lambda: [
            CurriculumPhase(
                name="solo_exploration",
                phase_id=1,
                min_objects=2,
                max_objects=3,
                num_agents=1,
                max_steps_per_episode=256,
                task_types=["navigate_to_object", "navigate_to_location"],
                min_episodes_before_advance=500,
                task_success_threshold=0.65,
            ),
            CurriculumPhase(
                name="affordance_discovery",
                phase_id=2,
                min_objects=5,
                max_objects=8,
                num_agents=1,
                max_steps_per_episode=512,
                task_types=[
                    "find_pushable_object",
                    "find_interactive_object",
                    "avoid_dangerous_object",
                ],
                min_episodes_before_advance=750,
                task_success_threshold=0.60,
            ),
            CurriculumPhase(
                name="partner_communication",
                phase_id=3,
                min_objects=5,
                max_objects=8,
                num_agents=2,
                max_steps_per_episode=512,
                task_types=[
                    "referential_game_location",
                    "referential_game_identity",
                    "coordinate_navigation",
                ],
                min_episodes_before_advance=1000,
                task_success_threshold=0.55,
            ),
            CurriculumPhase(
                name="compositional_reasoning",
                phase_id=4,
                min_objects=8,
                max_objects=15,
                num_agents=2,
                max_steps_per_episode=1024,
                task_types=[
                    "relational_query",
                    "multi_step_coordination",
                    "causal_chain_task",
                ],
                min_episodes_before_advance=2000,
                task_success_threshold=0.50,
            ),
        ]
    )

    eval_window_episodes: int = Field(
        default=100,
        gt=0,
        description="Number of recent episodes used to compute the rolling success rate.",
    )


class TrainingConfig(BaseModel):
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)

    total_timesteps: int = Field(
        default=10_000_000,
        gt=0,
        description="Total environment steps across all phases.",
    )
    num_envs: int = Field(
        default=8,
        gt=0,
        description="Number of parallel environments.",
    )
    seed: int = Field(default=42)
    checkpoint_interval_steps: int = Field(
        default=50_000,
        gt=0,
        description="Save a checkpoint every N environment steps.",
    )
    eval_interval_steps: int = Field(
        default=10_000,
        gt=0,
        description="Run evaluation and log metrics every N environment steps.",
    )
    checkpoint_dir: str = Field(default="checkpoints/")
    log_dir: str = Field(default="logs/")
