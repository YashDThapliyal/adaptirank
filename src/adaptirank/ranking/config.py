"""Strict configuration for M3 cross-encoder reranking scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from adaptirank.common.config import RunConfig, StrictModel


class CrossEncoderConfig(StrictModel):
    model_name: str
    model_revision: str
    fields: tuple[Literal["title", "description", "brand"], ...] = ("title", "description", "brand")
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    batch_size: int = Field(default=64, gt=0)
    max_length: int = Field(default=512, gt=0)


class CrossEncoderRunConfig(StrictModel):
    run: RunConfig
    dataset_dir: Path
    dataset_fingerprint: str
    retrieval_output_dir: Path = Path("artifacts/retrieval")
    retrieval_artifact_name: str
    output_dir: Path = Path("artifacts/ranking")
    artifact_name: str
    rank_column: str = "hybrid_rank"
    top_m: int = Field(default=100, gt=0)
    block_queries: int = Field(default=512, gt=0)
    max_queries_per_split: int | None = Field(default=None, gt=0)
    cross_encoder: CrossEncoderConfig
