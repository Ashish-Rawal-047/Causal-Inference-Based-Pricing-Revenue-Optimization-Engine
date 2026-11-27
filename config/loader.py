"""Central configuration loader and logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration with defaults resolved relative to project root."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_directories(cfg: dict[str, Any] | None = None) -> None:
    """Ensure standard output directories exist."""
    cfg = cfg or load_config()
    for key, rel in cfg.get("paths", {}).items():
        resolve_path(rel).mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
