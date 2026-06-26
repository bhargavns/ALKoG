"""
ALKoG training entry point.

Usage
-----
# With defaults:
python scripts/train.py

# With a custom config:
python scripts/train.py --config configs/default.yaml

# Override individual fields:
python scripts/train.py --config configs/default.yaml --run-name experiment_01
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from alkog.config.base import ALKoGConfig
from alkog.training import train_basic_ppo
from alkog.utils.logging import finish_wandb, get_logger, init_wandb

app = typer.Typer(name="alkog-train", add_completion=False)


@app.command()
def train(
    config: Optional[Path] = typer.Option(  # noqa: UP007
        None,
        "--config",
        "-c",
        help="Path to YAML config file.  Defaults are used for any omitted key.",
    ),
    run_name: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--run-name",
        help="Override the run name from the config.",
    ),
    device: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--device",
        help="Override device: auto | cpu | cuda | mps.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load config, print it, then exit.  Useful for validating YAML.",
    ),
    env_id: str = typer.Option(
        "CartPole-v1",
        "--env-id",
        help="Gymnasium env id used by the basic PPO trainer.",
    ),
    total_timesteps: Optional[int] = typer.Option(  # noqa: UP007
        None,
        "--total-timesteps",
        help=(
            "Override cfg.training.total_timesteps for this run. "
            "Useful for short smoke tests."
        ),
    ),
    seed: Optional[int] = typer.Option(  # noqa: UP007
        None,
        "--seed",
        help="Override cfg.training.seed for this run.",
    ),
) -> None:
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    if config is not None:
        cfg = ALKoGConfig.from_yaml(config)
    else:
        cfg = ALKoGConfig()

    # CLI overrides
    if run_name is not None:
        cfg = cfg.model_copy(update={"run_name": run_name})
    if device is not None:
        cfg = cfg.model_copy(update={"device": device})
    if seed is not None:
        cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"seed": seed})})

    log = get_logger("alkog.train", log_dir=cfg.training.log_dir)
    log.info(f"Config loaded: {cfg}")
    log.info(f"Resolved device: {cfg.resolved_device}")

    if dry_run:
        log.info("Dry run complete — exiting.")
        raise typer.Exit()

    # ------------------------------------------------------------------
    # 2. Save resolved config for reproducibility
    # ------------------------------------------------------------------
    resolved_cfg_path = Path(cfg.training.log_dir) / cfg.run_name / "config.yaml"
    cfg.to_yaml(resolved_cfg_path)
    log.info(f"Resolved config saved to {resolved_cfg_path}")

    # ------------------------------------------------------------------
    # 3. W&B
    # ------------------------------------------------------------------
    init_wandb(cfg)

    # ------------------------------------------------------------------
    # 4. Basic PPO training loop
    # ------------------------------------------------------------------
    run_timesteps = total_timesteps or cfg.training.total_timesteps
    summary = train_basic_ppo(
        cfg=cfg,
        env_id=env_id,
        total_timesteps=run_timesteps,
        seed=cfg.training.seed,
        log=log,
    )
    log.info(f"Training complete: {summary}")

    # ------------------------------------------------------------------
    # 5. Cleanup
    # ------------------------------------------------------------------
    finish_wandb()


if __name__ == "__main__":
    app()
