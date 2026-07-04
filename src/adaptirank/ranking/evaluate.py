"""Shared M3 learned-ranking evaluation on judged-aware candidate lists."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from adaptirank.retrieval.base import IndexBuildStats, RetrievalResult
from adaptirank.retrieval.evaluate import PRIMARY_LABELS, evaluate_result


def average_precision_condensed(labels: list[str], relevant_count: int) -> float:
    """Average precision on the condensed judged list, including missed known positives."""

    if relevant_count == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, label in enumerate(labels, start=1):
        if label in PRIMARY_LABELS:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / relevant_count


def evaluate_ranking(
    candidates: pl.DataFrame,
    *,
    method: str,
    queries: pl.DataFrame,
    relevance: pl.DataFrame,
    catalog: pl.DataFrame,
    latencies: pl.DataFrame | None = None,
    top_k: tuple[int, ...] = (5, 10, 50, 100, 500),
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Evaluate one ranking and add MAP to the retrieval-compatible metric contract."""

    if latencies is None:
        latencies = queries.select("query_key").with_columns(pl.lit(None).alias("latency_ms"))
    result = RetrievalResult(
        method=method,
        candidates=candidates,
        query_latencies_ms=latencies,
        build_stats=IndexBuildStats(0.0, 0, catalog.height, {}, {"stage": "ranking"}),
    )
    annotated, per_query, metrics = evaluate_result(
        result,
        queries=queries,
        relevance=relevance,
        catalog=catalog,
        top_k_values=top_k,
    )
    labels: dict[str, list[str]] = defaultdict(list)
    for row in (
        annotated.filter(pl.col("judgment_status") == "judged")
        .sort("query_key", "rank")
        .select("query_key", "esci_label")
        .iter_rows(named=True)
    ):
        labels[str(row["query_key"])].append(str(row["esci_label"]))
    known = {
        str(row["query_key"]): int(row["primary_count"])
        for row in relevance.group_by("query_key")
        .agg(pl.col("esci_label").is_in(PRIMARY_LABELS).sum().alias("primary_count"))
        .iter_rows(named=True)
    }
    average_precision = [
        average_precision_condensed(labels.get(str(key), []), known.get(str(key), 0))
        for key in per_query.get_column("query_key")
    ]
    per_query = per_query.with_columns(pl.Series("average_precision", average_precision))
    for split in metrics["by_split"]:
        values = (
            per_query.filter(pl.col("split") == split).get_column("average_precision").to_numpy()
        )
        metrics["by_split"][split]["map"] = float(np.mean(values)) if len(values) else 0.0
    return annotated, per_query, metrics


def ranked_candidates(frame: pl.DataFrame, score_column: str, rank_column: str) -> pl.DataFrame:
    """Return retrieval-compatible candidates with deterministic tie ordering."""

    ordered = frame.sort(
        ["query_key", score_column, "hybrid_rank", "product_key"],
        descending=[False, True, False, False],
        nulls_last=True,
    ).with_columns(
        pl.col("product_key").cum_count().over("query_key").cast(pl.Int32).alias(rank_column)
    )
    return ordered.select(
        "query_key",
        "product_key",
        "split",
        pl.lit(score_column.removesuffix("_score")).alias("method"),
        pl.col(score_column).alias("score"),
        pl.col(rank_column).alias("rank"),
    )
