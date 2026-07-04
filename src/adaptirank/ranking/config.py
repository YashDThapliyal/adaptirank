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
    pair_artifact: Path | None = None
    rank_column: str = "hybrid_rank"
    top_m: int = Field(default=100, gt=0)
    block_queries: int = Field(default=512, gt=0)
    max_queries_per_split: int | None = Field(default=None, gt=0)
    cross_encoder: CrossEncoderConfig


class HandoffSourceRuns(StrictModel):
    m2_bm25: str
    m2_dense: str
    m2_hybrid: str
    m3_bm25: str
    m3_dense: str
    m3_hybrid: str


class HandoffAnalysisConfig(StrictModel):
    run: RunConfig
    dataset_dir: Path
    dataset_fingerprint: str
    retrieval_output_dir: Path = Path("artifacts/retrieval")
    m2_artifact_name: str = "full_scientific"
    m3_artifact_name: str = "m3_three_split"
    top_k: tuple[int, ...] = (10, 50, 100, 500)
    source_runs: HandoffSourceRuns


class RankingFeatureConfig(StrictModel):
    candidate_rank_column: str = "hybrid_rank"
    top_m: int = Field(default=500, gt=0)


class RankingFeatureRunConfig(StrictModel):
    run: RunConfig
    dataset_dir: Path
    dataset_fingerprint: str
    retrieval_output_dir: Path = Path("artifacts/retrieval")
    retrieval_artifact_name: str = "m3_three_split"
    output_dir: Path = Path("artifacts/ranking")
    artifact_name: str = "m3_three_split"
    splits: tuple[Literal["train", "validation", "test"], ...] = (
        "train",
        "validation",
        "test",
    )
    features: RankingFeatureConfig = RankingFeatureConfig()


class PointwiseParams(StrictModel):
    learning_rate: float = Field(gt=0)
    max_iter: int = Field(gt=0)
    max_leaf_nodes: int = Field(gt=1)
    l2_regularization: float = Field(ge=0)


class LambdaMARTParams(StrictModel):
    learning_rate: float = Field(gt=0)
    n_estimators: int = Field(gt=0)
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(gt=0)
    feature_fraction: float = Field(gt=0, le=1)


class LearnedRankingConfig(StrictModel):
    pointwise_grid: tuple[PointwiseParams, ...]
    lambdamart_grid: tuple[LambdaMARTParams, ...]
    early_stopping_rounds: int = Field(default=30, gt=0)
    prediction_batch_size: int = Field(default=250_000, gt=0)
    latency_query_sample: int = Field(default=1_000, gt=0)


class LearnedRankingRunConfig(StrictModel):
    run: RunConfig
    dataset_dir: Path
    dataset_fingerprint: str
    feature_dir: Path
    output_dir: Path = Path("artifacts/ranking")
    artifact_name: str = "m3_three_split"
    ranking: LearnedRankingConfig


class CEUnionRunConfig(StrictModel):
    run: RunConfig
    dataset_fingerprint: str
    retrieval_root: Path
    learned_root: Path
    output_dir: Path
    hybrid_top_m: int = Field(default=100, gt=0)
    lambdamart_top_m: int = Field(default=50, gt=0)


class CEEvaluationRunConfig(StrictModel):
    run: RunConfig
    dataset_dir: Path
    dataset_fingerprint: str
    retrieval_root: Path
    learned_root: Path
    cross_encoder_root: Path
    scores_path: Path
