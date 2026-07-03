"""Typed ESCI pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from adaptirank.common.config import RunConfig, StrictModel


def _default_label_grades() -> dict[Literal["E", "S", "C", "I"], int]:
    return {"E": 3, "S": 2, "C": 1, "I": 0}


class SourceFileConfig(StrictModel):
    """One source file from either the tracked fixture or official repository."""

    role: Literal["examples", "products", "sources"]
    filename: str
    url: str | None = None
    fixture_path: Path | None = None
    authoritative_sha256: str | None = None

    @model_validator(mode="after")
    def require_location(self) -> SourceFileConfig:
        if (self.url is None) == (self.fixture_path is None):
            raise ValueError("exactly one of url or fixture_path is required")
        return self


class SourceConfig(StrictModel):
    """Pinned source release and local cache configuration."""

    mode: Literal["fixture", "official"]
    repository: str
    revision: str
    raw_dir: Path
    files: tuple[SourceFileConfig, ...]

    @model_validator(mode="after")
    def require_all_roles(self) -> SourceConfig:
        roles = [item.role for item in self.files]
        if sorted(roles) != ["examples", "products", "sources"]:
            raise ValueError("source files must contain each role exactly once")
        return self


class SamplingConfig(StrictModel):
    """Query-group and broad-catalog development sampling."""

    max_train_queries: int | None = Field(default=None, gt=0)
    max_test_queries: int | None = Field(default=None, gt=0)
    background_products: int | None = Field(default=None, ge=0)


class EsciConfig(StrictModel):
    """Complete M1 pipeline configuration."""

    run: RunConfig
    source: SourceConfig
    variant: Literal["small", "large"]
    product_locale: str = "us"
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    sampling: SamplingConfig
    processed_dir: Path = Path("artifacts/datasets/esci/processed")
    label_grades: dict[Literal["E", "S", "C", "I"], int] = Field(
        default_factory=_default_label_grades
    )

    @property
    def variant_column(self) -> str:
        return "small_version" if self.variant == "small" else "large_version"
