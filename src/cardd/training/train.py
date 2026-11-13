"""Train a YOLO-seg model from a YAML config."""

from __future__ import annotations

from pathlib import Path


def train(config_path: Path) -> None:
    """Run training using the parameters in ``config_path``."""
    raise NotImplementedError
