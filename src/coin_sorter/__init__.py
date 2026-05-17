"""Coin sorter computer vision pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

__version__ = "0.1.0"

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load `config.yaml` from disk, with optional local override.

    If `config.local.yaml` exists next to the requested config file, its values
    are deep-merged on top so per-machine overrides do not leak into git.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    local_path = config_path.with_name("config.local.yaml")
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def configure_logging(level: int = logging.INFO) -> None:
    """Initialise root logging with a consistent format across CLIs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
