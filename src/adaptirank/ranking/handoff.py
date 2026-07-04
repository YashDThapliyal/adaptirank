"""Validation and delta analysis for the M3 three-split retrieval handoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from adaptirank.data.provenance import sha256_file

METHOD_PATHS = {
    "bm25": Path("bm25/selected_candidates.parquet"),
    "dense": Path("dense/raw_candidates.parquet"),
    "weighted_hybrid": Path("hybrid/weighted/raw_candidates.parquet"),
    "rrf": Path("hybrid/rrf/raw_candidates.parquet"),
}


def _finite_count(column: str) -> pl.Expr:
    return (pl.col(column).is_nan() | pl.col(column).is_infinite()).sum().alias(f"{column}_invalid")


def validate_contract(path: Path) -> dict[str, Any]:
    """Return contract invariants, raising if a locked M3 requirement is violated."""

    frame = pl.scan_parquet(path)
    required = {
        "query_key",
        "product_key",
        "split",
        "bm25_score",
        "bm25_rank",
        "dense_score",
        "dense_rank",
        "hybrid_score",
        "hybrid_rank",
        "rrf_score",
        "rrf_rank",
        "esci_label",
        "relevance_grade",
        "judgment_status",
    }
    missing = sorted(required - set(frame.collect_schema().names()))
    if missing:
        raise ValueError(f"candidate contract is missing columns: {missing}")
    summary = (
        frame.select(
            pl.len().alias("rows"),
            pl.col("query_key").n_unique().alias("queries"),
            pl.struct("query_key", "product_key").n_unique().alias("unique_pairs"),
            (pl.col("judgment_status") == "judged").sum().alias("judged"),
            (pl.col("judgment_status") == "unjudged").sum().alias("unjudged"),
            (
                (pl.col("judgment_status") == "judged")
                & (pl.col("esci_label").is_null() | pl.col("relevance_grade").is_null())
            )
            .sum()
            .alias("judged_missing_label_or_grade"),
            (
                (pl.col("judgment_status") == "unjudged")
                & (pl.col("esci_label").is_not_null() | pl.col("relevance_grade").is_not_null())
            )
            .sum()
            .alias("unjudged_with_label_or_grade"),
            *(
                _finite_count(column)
                for column in ("bm25_score", "dense_score", "hybrid_score", "rrf_score")
            ),
        )
        .collect()
        .to_dicts()[0]
    )
    by_split = (
        frame.group_by("split")
        .agg(pl.len().alias("rows"), pl.col("query_key").n_unique().alias("queries"))
        .sort("split")
        .collect()
        .to_dicts()
    )
    split_collisions = (
        frame.select("query_key", "split")
        .unique()
        .group_by("query_key")
        .agg(pl.col("split").n_unique().alias("split_count"))
        .filter(pl.col("split_count") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    expected_queries = {"train": 18_799, "validation": 2_089, "test": 8_956}
    observed_queries = {str(row["split"]): int(row["queries"]) for row in by_split}
    failures: list[str] = []
    if summary["rows"] != summary["unique_pairs"]:
        failures.append("duplicate query-product pairs")
    if observed_queries != expected_queries:
        failures.append(f"query counts {observed_queries} != {expected_queries}")
    if split_collisions:
        failures.append(f"{split_collisions} query keys occur in multiple splits")
    for key in (
        "judged_missing_label_or_grade",
        "unjudged_with_label_or_grade",
        "bm25_score_invalid",
        "dense_score_invalid",
        "hybrid_score_invalid",
        "rrf_score_invalid",
    ):
        if summary[key]:
            failures.append(f"{key}={summary[key]}")
    if failures:
        raise ValueError("invalid M3 candidate contract: " + "; ".join(failures))
    return {
        **summary,
        "by_split": by_split,
        "query_split_collisions": split_collisions,
        "sha256": sha256_file(path),
        "path": str(path.resolve()),
    }


def _per_query_overlap(m2_path: Path, m3_path: Path, *, split: str, k: int) -> pl.DataFrame:
    left = (
        pl.scan_parquet(m2_path)
        .filter((pl.col("split") == split) & (pl.col("rank") <= k))
        .select("query_key", "product_key", pl.col("rank").alias("m2_rank"))
    )
    right = (
        pl.scan_parquet(m3_path)
        .filter((pl.col("split") == split) & (pl.col("rank") <= k))
        .select("query_key", "product_key", pl.col("rank").alias("m3_rank"))
    )
    shared = (
        left.join(right, on=["query_key", "product_key"], how="inner")
        .group_by("query_key")
        .agg(
            pl.len().alias("intersection"),
            pl.corr("m2_rank", "m3_rank", method="spearman").alias("spearman"),
        )
    )
    counts = (
        left.group_by("query_key")
        .len(name="m2_count")
        .join(
            right.group_by("query_key").len(name="m3_count"),
            on="query_key",
            how="full",
            coalesce=True,
        )
    )
    return (
        counts.join(shared, on="query_key", how="left")
        .with_columns(pl.col("intersection").fill_null(0))
        .with_columns(
            (
                pl.col("intersection")
                / (pl.col("m2_count") + pl.col("m3_count") - pl.col("intersection"))
            ).alias("jaccard")
        )
        .collect()
    )


def candidate_delta_analysis(
    m2_root: Path, m3_root: Path, *, top_k: tuple[int, ...]
) -> dict[str, Any]:
    """Compare M2 and M3 candidates without writing into either source tree."""

    output: dict[str, Any] = {}
    for method, relative in METHOD_PATHS.items():
        method_result: dict[str, Any] = {}
        for split in ("validation", "test"):
            split_result: dict[str, Any] = {}
            for k in top_k:
                overlap = _per_query_overlap(
                    m2_root / relative, m3_root / relative, split=split, k=k
                )
                jaccard = overlap.get_column("jaccard").to_numpy()
                spearman = overlap.get_column("spearman").drop_nulls().to_numpy()
                spearman = spearman[np.isfinite(spearman)]
                worst = (
                    overlap.sort("jaccard", "query_key")
                    .head(10)
                    .select(
                        "query_key", "m2_count", "m3_count", "intersection", "jaccard", "spearman"
                    )
                )
                split_result[str(k)] = {
                    "queries": overlap.height,
                    "jaccard_mean": float(np.mean(jaccard)),
                    "jaccard_median": float(np.median(jaccard)),
                    "jaccard_p05": float(np.percentile(jaccard, 5)),
                    "spearman_queries": len(spearman),
                    "spearman_mean": float(np.mean(spearman)) if len(spearman) else None,
                    "spearman_median": float(np.median(spearman)) if len(spearman) else None,
                    "worst_jaccard_queries": worst.to_dicts(),
                }
            method_result[split] = split_result
        output[method] = method_result
    return output


def metric_deltas(m2_root: Path, m3_root: Path) -> dict[str, Any]:
    """Return M3 minus M2 aggregate quality metrics for validation and test."""

    metric_paths = {
        "bm25": Path("bm25/title/metrics.json"),
        "dense": Path("dense/metrics.json"),
        "weighted_hybrid": Path("hybrid/weighted/metrics.json"),
        "rrf": Path("hybrid/rrf/metrics.json"),
    }
    metrics = (
        "recall_primary_10",
        "recall_primary_50",
        "recall_primary_100",
        "recall_primary_500",
        "mrr",
        "ndcg_10",
    )
    output: dict[str, Any] = {}
    for method, relative in metric_paths.items():
        m2 = json.loads((m2_root / relative).read_text())["by_split"]
        m3 = json.loads((m3_root / relative).read_text())["by_split"]
        output[method] = {
            split: {name: float(m3[split][name] - m2[split][name]) for name in metrics}
            for split in ("validation", "test")
        }
    return output
