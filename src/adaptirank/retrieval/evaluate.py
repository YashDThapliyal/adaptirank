"""Shared judged-aware retrieval evaluation and analysis."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from adaptirank.retrieval.base import RetrievalResult

PRIMARY_LABELS = {"E", "S"}
SENSITIVITY_LABELS = {"E", "S", "C"}
_TOKENS = re.compile(r"\w+", flags=re.UNICODE)


def recall_at_k(retrieved_labels: list[str | None], relevant_count: int, k: int) -> float | None:
    """Recall over known relevant judgments; unjudged entries are never negatives."""

    if relevant_count == 0:
        return None
    hits = sum(label in PRIMARY_LABELS for label in retrieved_labels[:k])
    return hits / relevant_count


def reciprocal_rank_condensed(retrieved_labels: list[str | None]) -> float:
    """MRR on the condensed judged list, ignoring rather than demoting unjudged entries."""

    judged_rank = 0
    for label in retrieved_labels:
        if label is None:
            continue
        judged_rank += 1
        if label in PRIMARY_LABELS:
            return 1.0 / judged_rank
    return 0.0


def ndcg_condensed(retrieved_grades: list[int | None], ideal_grades: list[int], k: int) -> float:
    """NDCG on judged retrieved items only, preserving unknown as unknown."""

    judged = [grade for grade in retrieved_grades if grade is not None][:k]
    ideal = sorted(ideal_grades, reverse=True)[:k]

    def dcg(grades: list[int]) -> float:
        return float(sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades)))

    ideal_dcg = dcg(ideal)
    return dcg(judged) / ideal_dcg if ideal_dcg else 0.0


def annotate_candidates(candidates: pl.DataFrame, relevance: pl.DataFrame) -> pl.DataFrame:
    """Attach raw judgments without coercing absent labels to irrelevant."""

    judgment = relevance.select("query_key", "product_key", "esci_label", "relevance_grade").unique(
        ["query_key", "product_key"]
    )
    return candidates.join(judgment, on=["query_key", "product_key"], how="left").with_columns(
        pl.when(pl.col("esci_label").is_null())
        .then(pl.lit("unjudged"))
        .otherwise(pl.lit("judged"))
        .alias("judgment_status")
    )


def _query_attributes(
    queries: pl.DataFrame, relevance: pl.DataFrame, catalog: pl.DataFrame
) -> pl.DataFrame:
    query_lookup = {str(row["query_key"]): row for row in queries.iter_rows(named=True)}
    labels: dict[str, set[str]] = defaultdict(set)
    overlap: dict[str, float] = defaultdict(float)
    relevant = relevance.filter(pl.col("esci_label").is_in(PRIMARY_LABELS)).join(
        catalog.select("product_key", "title"), on="product_key", how="left"
    )
    for row in relevance.select("query_key", "esci_label").iter_rows(named=True):
        labels[str(row["query_key"])].add(str(row["esci_label"]))
    for row in relevant.select("query_key", "query", "title").iter_rows(named=True):
        query_tokens = set(_TOKENS.findall(str(row["query"] or "").lower()))
        title_tokens = set(_TOKENS.findall(str(row["title"] or "").lower()))
        score = len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0
        key = str(row["query_key"])
        overlap[key] = max(overlap[key], score)
    records: list[dict[str, Any]] = []
    for key, row in query_lookup.items():
        length = len(_TOKENS.findall(str(row["query_text"])))
        length_slice = (
            "short_1_2" if length <= 2 else "medium_3_5" if length <= 5 else "long_6_plus"
        )
        lexical = overlap.get(key, 0.0)
        overlap_slice = "none" if lexical == 0 else "low" if lexical < 0.5 else "high"
        records.append(
            {
                "query_key": key,
                "query_text": row["query_text"],
                "query_length_tokens": length,
                "query_length_slice": length_slice,
                "max_primary_title_overlap": lexical,
                "lexical_overlap_slice": overlap_slice,
                "label_structure": "+".join(sorted(labels.get(key, set()))),
            }
        )
    return pl.DataFrame(records)


def _per_query_metrics(
    annotated: pl.DataFrame,
    queries: pl.DataFrame,
    relevance: pl.DataFrame,
    catalog: pl.DataFrame,
    latencies: pl.DataFrame,
    top_k_values: tuple[int, ...],
) -> pl.DataFrame:
    relevance_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relevance.select(
        "query_key", "product_key", "esci_label", "relevance_grade"
    ).iter_rows(named=True):
        relevance_by_query[str(row["query_key"])].append(row)
    retrieved_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    judged = annotated.filter(pl.col("judgment_status") == "judged").sort("query_key", "rank")
    for row in judged.select(
        "query_key", "product_key", "rank", "esci_label", "relevance_grade"
    ).iter_rows(named=True):
        retrieved_by_query[str(row["query_key"])].append(row)

    hit_aggregations: list[pl.Expr] = []
    for k in top_k_values:
        hit_aggregations.extend(
            [
                ((pl.col("rank") <= k) & pl.col("esci_label").is_in(PRIMARY_LABELS))
                .sum()
                .alias(f"primary_hits_{k}"),
                ((pl.col("rank") <= k) & pl.col("esci_label").is_in(SENSITIVITY_LABELS))
                .sum()
                .alias(f"sensitivity_hits_{k}"),
            ]
        )
    hits = annotated.group_by("query_key").agg(*hit_aggregations)
    records: list[dict[str, Any]] = []
    hit_lookup = {str(row["query_key"]): row for row in hits.iter_rows(named=True)}
    split_lookup = {
        str(row["query_key"]): str(row["benchmark_split"])
        for row in queries.select("query_key", "benchmark_split").iter_rows(named=True)
    }
    for query_key in split_lookup:
        known = relevance_by_query[query_key]
        retrieved = retrieved_by_query.get(query_key, [])
        primary_count = sum(row["esci_label"] in PRIMARY_LABELS for row in known)
        sensitivity_count = sum(row["esci_label"] in SENSITIVITY_LABELS for row in known)
        record: dict[str, Any] = {
            "query_key": query_key,
            "split": split_lookup[query_key],
            "judged_count": len(known),
            "primary_relevant_count": primary_count,
            "sensitivity_relevant_count": sensitivity_count,
            "mrr": reciprocal_rank_condensed([row["esci_label"] for row in retrieved]),
            "ndcg_5": ndcg_condensed(
                [row["relevance_grade"] for row in retrieved],
                [int(row["relevance_grade"]) for row in known],
                5,
            ),
            "ndcg_10": ndcg_condensed(
                [row["relevance_grade"] for row in retrieved],
                [int(row["relevance_grade"]) for row in known],
                10,
            ),
        }
        query_hits = hit_lookup.get(query_key, {})
        for k in top_k_values:
            record[f"recall_primary_{k}"] = (
                int(query_hits.get(f"primary_hits_{k}", 0)) / primary_count
                if primary_count
                else None
            )
            record[f"recall_sensitivity_{k}"] = (
                int(query_hits.get(f"sensitivity_hits_{k}", 0)) / sensitivity_count
                if sensitivity_count
                else None
            )
        records.append(record)
    return (
        pl.DataFrame(records)
        .join(_query_attributes(queries, relevance, catalog), on="query_key", how="left")
        .join(latencies, on="query_key", how="left")
        .sort("split", "query_key")
    )


def _mean_metrics(frame: pl.DataFrame, metric_columns: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for column in metric_columns:
        values = frame.get_column(column).drop_nulls().cast(pl.Float64).to_numpy()
        output[column] = float(np.mean(values)) if len(values) else 0.0
    return output


def _slice_metrics(
    frame: pl.DataFrame, slice_column: str, metric_columns: list[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in frame.get_column(slice_column).drop_nulls().unique().sort().to_list():
        subset = frame.filter(pl.col(slice_column) == value)
        output[str(value)] = {"queries": subset.height, **_mean_metrics(subset, metric_columns)}
    return output


def evaluate_result(
    result: RetrievalResult,
    *,
    queries: pl.DataFrame,
    relevance: pl.DataFrame,
    catalog: pl.DataFrame,
    top_k_values: tuple[int, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Return annotated candidates, raw per-query metrics, and aggregate analysis."""

    annotated = annotate_candidates(result.candidates, relevance)
    per_query = _per_query_metrics(
        annotated,
        queries,
        relevance.filter(pl.col("query_key").is_in(queries.get_column("query_key"))),
        catalog,
        result.query_latencies_ms,
        top_k_values,
    )
    metric_columns = [
        *(f"recall_primary_{k}" for k in top_k_values),
        *(f"recall_sensitivity_{k}" for k in top_k_values),
        "mrr",
        "ndcg_5",
        "ndcg_10",
    ]
    latencies = per_query.get_column("latency_ms").drop_nulls().to_numpy()
    by_split = {
        split: {
            "queries": subset.height,
            **_mean_metrics(subset, metric_columns),
        }
        for split in per_query.get_column("split").unique().sort().to_list()
        if (subset := per_query.filter(pl.col("split") == split)).height
    }
    aggregate: dict[str, Any] = {
        "method": result.method,
        "evaluation_semantics": {
            "primary_relevance": ["E", "S"],
            "sensitivity_relevance": ["E", "S", "C"],
            "unjudged": "unknown; excluded from condensed MRR/NDCG",
        },
        "by_split": by_split,
        "latency": {
            "p50_ms": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
            "p95_ms": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
            "throughput_queries_per_second": (
                1000.0 / float(np.mean(latencies)) if len(latencies) and np.mean(latencies) else 0.0
            ),
        },
        "index": {
            "build_seconds": result.build_stats.build_seconds,
            "size_bytes": result.build_stats.index_size_bytes,
            "document_count": result.build_stats.document_count,
            "artifacts": result.build_stats.artifact_paths,
            "metadata": result.build_stats.metadata,
        },
        "slices": {
            "query_length": _slice_metrics(per_query, "query_length_slice", metric_columns),
            "lexical_overlap": _slice_metrics(per_query, "lexical_overlap_slice", metric_columns),
            "label_structure": _slice_metrics(per_query, "label_structure", metric_columns),
        },
    }
    return annotated, per_query, aggregate


def failure_cases(
    per_query: pl.DataFrame,
    candidates: pl.DataFrame,
    relevance: pl.DataFrame,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Select worst observed test queries with retrieved and missed judged examples."""

    worst = (
        per_query.filter(pl.col("split") == "test")
        .sort("recall_primary_100", "ndcg_10", "query_key")
        .head(limit)
    )
    output: list[dict[str, Any]] = []
    for query in worst.iter_rows(named=True):
        key = query["query_key"]
        top = (
            candidates.filter(pl.col("query_key") == key)
            .sort("rank")
            .head(10)
            .select(
                "product_key",
                "rank",
                "score",
                "judgment_status",
                "esci_label",
                "relevance_grade",
            )
            .to_dicts()
        )
        retrieved_keys = set(
            candidates.filter(pl.col("query_key") == key).get_column("product_key")
        )
        missed = (
            relevance.filter(
                (pl.col("query_key") == key)
                & pl.col("esci_label").is_in(PRIMARY_LABELS)
                & ~pl.col("product_key").is_in(retrieved_keys)
            )
            .select("product_key", "esci_label", "relevance_grade")
            .head(10)
            .to_dicts()
        )
        output.append(
            {
                "query_key": key,
                "query_text": query["query_text"],
                "recall_primary_100": query["recall_primary_100"],
                "ndcg_10": query["ndcg_10"],
                "top_retrieved": top,
                "missed_primary_relevant": missed,
            }
        )
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)
