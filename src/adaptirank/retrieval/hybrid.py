"""Validation-tuned weighted fusion and reciprocal rank fusion."""

from __future__ import annotations

import time
from typing import Any

import polars as pl


def _method_columns(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return frame.select(
        "query_key",
        "product_key",
        pl.col("split").alias(f"{prefix}_split"),
        pl.col("score").alias(f"{prefix}_score"),
        pl.col("rank").alias(f"{prefix}_rank"),
    )


def _union(bm25: pl.DataFrame, dense: pl.DataFrame) -> pl.DataFrame:
    return (
        _method_columns(bm25, "bm25")
        .join(
            _method_columns(dense, "dense"),
            on=["query_key", "product_key"],
            how="full",
            coalesce=True,
        )
        .with_columns(pl.coalesce("bm25_split", "dense_split").alias("split"))
    )


def weighted_fusion(
    bm25: pl.DataFrame, dense: pl.DataFrame, *, alpha: float, top_k: int
) -> tuple[pl.DataFrame, float]:
    """Per-query min-max score fusion with absent method scores set to zero."""

    started = time.perf_counter()
    union = _union(bm25, dense)
    for prefix in ("bm25", "dense"):
        score = pl.col(f"{prefix}_score")
        minimum = score.min().over("query_key")
        maximum = score.max().over("query_key")
        union = union.with_columns(
            pl.when(score.is_null())
            .then(0.0)
            .when(maximum > minimum)
            .then((score - minimum) / (maximum - minimum))
            .otherwise(1.0)
            .alias(f"{prefix}_normalized")
        )
    fused = (
        union.with_columns(
            (alpha * pl.col("bm25_normalized") + (1.0 - alpha) * pl.col("dense_normalized")).alias(
                "score"
            )
        )
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over("query_key")
            .cast(pl.Int32)
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .select(
            "query_key",
            "product_key",
            "split",
            pl.lit("weighted_hybrid").alias("method"),
            "score",
            "rank",
        )
        .sort("query_key", "rank")
    )
    return fused, time.perf_counter() - started


def reciprocal_rank_fusion(
    bm25: pl.DataFrame, dense: pl.DataFrame, *, rrf_k: int, top_k: int
) -> tuple[pl.DataFrame, float]:
    """Fuse method ranks without score calibration."""

    started = time.perf_counter()
    fused = (
        _union(bm25, dense)
        .with_columns(
            (
                pl.when(pl.col("bm25_rank").is_not_null())
                .then(1.0 / (rrf_k + pl.col("bm25_rank")))
                .otherwise(0.0)
                + pl.when(pl.col("dense_rank").is_not_null())
                .then(1.0 / (rrf_k + pl.col("dense_rank")))
                .otherwise(0.0)
            ).alias("score")
        )
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over("query_key")
            .cast(pl.Int32)
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .select(
            "query_key",
            "product_key",
            "split",
            pl.lit("rrf").alias("method"),
            "score",
            "rank",
        )
        .sort("query_key", "rank")
    )
    return fused, time.perf_counter() - started


def hybrid_latencies(
    bm25_latencies: pl.DataFrame,
    dense_latencies: pl.DataFrame,
    *,
    fusion_seconds: float,
) -> pl.DataFrame:
    """Conservative sequential latency: lexical + dense + per-query fusion."""

    joined = bm25_latencies.rename({"latency_ms": "bm25_latency_ms"}).join(
        dense_latencies.rename({"latency_ms": "dense_latency_ms"}), on="query_key", how="inner"
    )
    fusion_ms = fusion_seconds * 1000 / max(joined.height, 1)
    return joined.select(
        "query_key",
        (pl.col("bm25_latency_ms") + pl.col("dense_latency_ms") + fusion_ms).alias("latency_ms"),
    )


def candidate_contract(
    bm25: pl.DataFrame,
    dense: pl.DataFrame,
    weighted: pl.DataFrame,
    rrf: pl.DataFrame,
    relevance: pl.DataFrame,
) -> pl.DataFrame:
    """Wide reusable M3-ready candidate artifact with explicit judgment status."""

    output = _method_columns(bm25, "bm25").join(
        _method_columns(dense, "dense"),
        on=["query_key", "product_key"],
        how="full",
        coalesce=True,
    )
    for frame, prefix in ((weighted, "hybrid"), (rrf, "rrf")):
        output = output.join(
            _method_columns(frame, prefix),
            on=["query_key", "product_key"],
            how="full",
            coalesce=True,
        )
    output = output.with_columns(
        pl.coalesce("hybrid_split", "rrf_split", "bm25_split", "dense_split").alias("split")
    )
    judgments = relevance.select(
        "query_key", "product_key", "esci_label", "relevance_grade"
    ).unique(["query_key", "product_key"])
    return (
        output.join(judgments, on=["query_key", "product_key"], how="left")
        .with_columns(
            pl.when(pl.col("esci_label").is_null())
            .then(pl.lit("unjudged"))
            .otherwise(pl.lit("judged"))
            .alias("judgment_status")
        )
        .select(
            "query_key",
            "product_key",
            "split",
            "bm25_score",
            "bm25_rank",
            "dense_score",
            "dense_rank",
            pl.col("hybrid_score"),
            pl.col("hybrid_rank"),
            pl.col("rrf_score"),
            pl.col("rrf_rank"),
            "esci_label",
            "relevance_grade",
            "judgment_status",
        )
        .sort("query_key", "hybrid_rank", "rrf_rank", nulls_last=True)
    )


def select_validation_alpha(alpha_metrics: dict[float, dict[str, Any]]) -> float:
    """Select alpha on validation primary Recall@100, then NDCG@10, then smaller alpha."""

    if not alpha_metrics:
        raise ValueError("no validation alpha metrics")
    return max(
        alpha_metrics,
        key=lambda alpha: (
            alpha_metrics[alpha]["recall_primary_100"],
            alpha_metrics[alpha]["ndcg_10"],
            -alpha,
        ),
    )
