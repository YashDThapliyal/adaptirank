"""Strict YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model that rejects undocumented configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    """Configuration shared by every tracked command."""

    experiment: str
    purpose: str
    seed: int = 42
    artifacts_dir: Path = Path("artifacts")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping documents."""

    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return raw


def load_config[ConfigT: StrictModel](path: Path, model: type[ConfigT]) -> ConfigT:
    """Parse a YAML document into a strict typed model."""

    return model.model_validate(load_yaml(path))
