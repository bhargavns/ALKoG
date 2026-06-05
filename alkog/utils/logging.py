"""
Centralised logging for ALKoG.

Provides:
  - A Rich-enhanced console logger for clean terminal output.
  - A rotating file handler that writes plain-text logs alongside
    each checkpoint directory.
  - A thin W&B wrapper so callers don't import wandb directly
    (making it easy to swap out or disable).

Usage
-----
from alkog.utils.logging import get_logger, init_wandb

log = get_logger(__name__)
log.info("Training started")

# In train.py entry point only:
init_wandb(cfg)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from alkog.config.base import ALKoGConfig

# ---------------------------------------------------------------------------
# Module-level console (shared across all loggers)
# ---------------------------------------------------------------------------
_console = Console(stderr=True)

_LOGGING_FORMAT = "%(message)s"
_DATE_FORMAT = "[%X]"

_root_configured = False


def _configure_root(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """One-time setup of the root logger. Called lazily on first get_logger()."""
    global _root_configured
    if _root_configured:
        return

    handlers: list[logging.Handler] = [
        RichHandler(
            console=_console,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            show_path=False,
        )
    ]

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            Path(log_dir) / "alkog.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format=_LOGGING_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )
    # Silence noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "urllib3", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _root_configured = True


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Return a named logger, configuring the root handler on first call."""
    _configure_root(log_dir=log_dir)
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# W&B wrapper
# ---------------------------------------------------------------------------

_wandb_run: Any = None  # holds the active wandb.Run


def init_wandb(cfg: "ALKoGConfig") -> None:
    """Initialise a W&B run if cfg.wandb.enabled is True."""
    global _wandb_run
    if not cfg.wandb.enabled:
        return
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        get_logger(__name__).warning(
            "wandb not installed.  Install it with `pip install wandb` or "
            "set wandb.enabled: false in your config."
        )
        return

    _wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.run_name,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config=cfg.model_dump(),
    )
    get_logger(__name__).info(f"W&B run initialised: {_wandb_run.url}")


def log_metrics(metrics: dict[str, float], step: int) -> None:
    """Log a dict of scalar metrics.  Works with or without W&B."""
    if _wandb_run is not None:
        _wandb_run.log(metrics, step=step)


def finish_wandb() -> None:
    """Close the W&B run gracefully (call at end of training)."""
    if _wandb_run is not None:
        _wandb_run.finish()
