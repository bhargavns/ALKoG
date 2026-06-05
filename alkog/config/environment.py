"""
Environment configuration.

The environment interface is the same whether backed by the mock Python
environment (use_godot=False) or by Godot via WebSocket (use_godot=True).
This makes it trivial to validate the full training pipeline in the mock
before wiring up Godot.

Observation space:
  The raw observation is a rendered RGB frame from Godot (or its mock
  equivalent).  The visual encoder then produces a fixed-size feature
  vector regardless of the original resolution.

  84x84 is the canonical Atari-style resolution and works fine for the
  early curriculum.  Bump to 128x128 or 224x224 in later phases if fine
  visual detail (e.g. distinguishing object textures) becomes important.

Action space:
  Actions are discrete and hierarchically structured.  At each timestep
  the agent first picks an action TYPE, then arguments specific to that
  type.  See alkog/agents/action_space.py (Block 5) for the full schema.

  Action types (from the proposal):
    0 — Movement / Interaction  (direction or object handle)
    1 — Bounding Box Proposal   (x, y, w, h as fractions of scene size)
    2 — KG Edge Proposal        (src_node_id, edge_type, tgt_node_id)
    3 — Utterance               (token sequence)
    4 — No-op
"""

from pydantic import BaseModel, Field


class ObservationConfig(BaseModel):
    width: int = Field(default=84, gt=0, description="Observation frame width in pixels.")
    height: int = Field(default=84, gt=0, description="Observation frame height in pixels.")
    channels: int = Field(default=3, description="RGB = 3.")
    stack_frames: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of consecutive frames stacked into a single observation. "
            "1 = no stacking (agents see a single rendered frame). "
            "Increase to 4 if temporal motion cues become important."
        ),
    )
    normalize: bool = Field(
        default=True,
        description="Normalise pixel values to [0, 1] before passing to the encoder.",
    )


class BoundingBoxConfig(BaseModel):
    min_size_fraction: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Minimum bounding box side length as a fraction of the scene width.",
    )
    max_size_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Maximum bounding box side length as a fraction of the scene width.",
    )
    max_proposals_per_step: int = Field(
        default=1,
        gt=0,
        description=(
            "Maximum number of bounding box proposals the agent can make per timestep. "
            "Keeping this at 1 makes the KG construction cost explicit."
        ),
    )


class GodotBridgeConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=7654, gt=0, lt=65536)
    timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="WebSocket read timeout.  Raise if Godot scene generation is slow.",
    )
    max_reconnect_attempts: int = Field(
        default=5,
        gt=0,
        description="Number of reconnection attempts before raising a fatal error.",
    )


class EnvironmentConfig(BaseModel):
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    bounding_box: BoundingBoxConfig = Field(default_factory=BoundingBoxConfig)
    godot_bridge: GodotBridgeConfig = Field(default_factory=GodotBridgeConfig)

    use_godot: bool = Field(
        default=False,
        description=(
            "False = use the pure-Python mock environment (Block 8). "
            "True  = connect to a running Godot instance via WebSocket (Block 11). "
            "The training loop is identical either way."
        ),
    )

    render_mode: str | None = Field(
        default=None,
        description="Gymnasium render mode: None (headless), 'human', or 'rgb_array'.",
    )

    episode_time_limit_seconds: float | None = Field(
        default=None,
        description=(
            "Real-time episode length cap (seconds).  None = no wall-clock limit. "
            "Useful when debugging to prevent frozen Godot instances from blocking."
        ),
    )
