"""Strict M2 retrieval experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from adaptirank.common.config import RunConfig, StrictModel


class BM25FieldSet(StrictModel):
    name: str
    fields: tuple[Literal["title", "description", "brand"], ...]


class BM25Config(StrictModel):
    field_sets: tuple[BM25FieldSet, ...]
    writer_heap_bytes: int = Field(default=512_000_000, ge=50_000_000)


class DenseConfig(StrictModel):
    model_name: str
    model_revision: str
    fields: tuple[Literal["title", "description", "brand"], ...]
    batch_size: int = Field(default=256, gt=0)
    outer_batch_size: int = Field(default=4096, gt=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    nlist: int = Field(default=2048, gt=0)
    nprobe: int = Field(default=32, gt=0)
    training_sample_size: int = Field(default=100_000, gt=0)


class HybridConfig(StrictModel):
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    rrf_k: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def validate_alphas(self) -> HybridConfig:
        if not self.alpha_grid or any(value < 0 or value > 1 for value in self.alpha_grid):
            raise ValueError("alpha_grid must contain values in [0, 1]")
        return self


class RetrievalConfig(StrictModel):
    run: RunConfig
    dataset_dir: Path
    dataset_fingerprint: str
    output_dir: Path = Path("artifacts/retrieval")
    artifact_name: str
    evaluation_splits: tuple[Literal["train", "validation", "test"], ...] = ("validation", "test")
    top_k: tuple[int, ...] = (10, 50, 100, 500)
    max_queries_per_split: int | None = Field(default=None, gt=0)
    bm25: BM25Config
    dense: DenseConfig
    hybrid: HybridConfig

    @model_validator(mode="after")
    def validate_top_k(self) -> RetrievalConfig:
        required = {10, 50, 100, 500}
        if not required.issubset(self.top_k):
            raise ValueError("top_k must include 10, 50, 100, and 500")
        if tuple(sorted(set(self.top_k))) != self.top_k:
            raise ValueError("top_k must be unique and sorted")
        return self
