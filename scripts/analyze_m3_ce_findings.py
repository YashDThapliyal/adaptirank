"""M3 cross-encoder findings from existing artifacts only."""

from __future__ import annotations

import json
import math
import subprocess
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from m3_findings_doc import expanded_findings_doc

from adaptirank.common.paths import project_root
from adaptirank.data.provenance import sha256_file

FP = "dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667"
METHODS = {
    "bm25": "BM25",
    "dense": "Dense",
    "weighted_hybrid": "Weighted Hybrid",
    "rrf": "RRF",
    "pointwise": "Pointwise",
    "lambdamart": "LambdaMART",
    "hybrid_to_cross_encoder": "Hybrid->CE",
    "hybrid_to_lambdamart_to_cross_encoder": "Hybrid->LambdaMART->CE",
}
LABELS = ("E", "S", "C", "I", "UNJUDGED")
METRICS = ("ndcg_5", "ndcg_10", "mrr")
SLICE_COLUMNS = {
    "query_length": "query_length_slice",
    "lexical_overlap": "lexical_overlap_slice",
    "bm25_dense_disagreement": "bm25_dense_disagreement_slice",
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def read_json(path: Path) -> dict[str, Any]:
    data: Any = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("dict[str, Any]", data)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def git(args: list[str], root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except Exception:
        return None


def paths(root: Path) -> dict[str, Path]:
    ranking = root / "artifacts" / "ranking" / FP / "m3_three_split"
    ce = ranking / "cross_encoder"
    return {
        "root": root,
        "ranking": ranking,
        "ce": ce,
        "eval": ce / "evaluation",
        "analysis": ce / "analysis",
        "learned": ranking / "learned",
        "retrieval": root / "artifacts" / "retrieval" / FP / "m3_three_split",
        "dataset": root / "artifacts" / "datasets" / "esci" / "processed" / FP,
        "script": root / "scripts" / "analyze_m3_ce_findings.py",
        "docs": root / "docs",
    }


def label_expr() -> pl.Expr:
    return (
        pl.when(pl.col("esci_label").is_null())
        .then(pl.lit("UNJUDGED"))
        .otherwise(pl.col("esci_label"))
        .alias("label")
    )


def summarize_displacements(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "pct_promoted": None,
            "pct_demoted": None,
            "pct_unchanged": None,
            "pct_abs_ge_10": None,
            "pct_abs_ge_25": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "pct_promoted": float((array < 0).mean()),
        "pct_demoted": float((array > 0).mean()),
        "pct_unchanged": float((array == 0).mean()),
        "pct_abs_ge_10": float((np.abs(array) >= 10).mean()),
        "pct_abs_ge_25": float((np.abs(array) >= 25).mean()),
    }


def mean_metric_delta(
    frame: pl.DataFrame,
    metric: str,
    method: str,
    baseline: str,
) -> float | None:
    method_column = f"{method}__{metric}"
    baseline_column = f"{baseline}__{metric}"
    if method_column not in frame.columns or baseline_column not in frame.columns:
        return None
    values = (frame[method_column] - frame[baseline_column]).drop_nulls().to_numpy()
    if values.size == 0:
        return None
    return float(values.mean())


def count_ordering_violations(frame: pl.LazyFrame | pl.DataFrame) -> int:
    if isinstance(frame, pl.LazyFrame):
        data = frame.select("query_key", "product_key", "score", "rank").collect()
    else:
        data = frame.select("query_key", "product_key", "score", "rank")
    expected = (
        data.sort(["query_key", "score", "product_key"], descending=[False, True, False])
        .with_columns(pl.cum_count("rank").over("query_key").cast(pl.Int32).alias("expected_rank"))
        .filter(pl.col("rank") != pl.col("expected_rank"))
    )
    return expected.height


def load_per_query(p: dict[str, Path]) -> dict[str, pl.DataFrame]:
    per_query_dir = p["learned"] / "analysis" / "per_query"
    frames = {
        method: pl.read_parquet(per_query_dir / f"{method}.parquet")
        for method in ("bm25", "dense", "weighted_hybrid", "rrf", "pointwise", "lambdamart")
    }
    frames["hybrid_to_cross_encoder"] = pl.read_parquet(p["eval"] / "ce_a_per_query.parquet")
    frames["hybrid_to_lambdamart_to_cross_encoder"] = pl.read_parquet(
        p["eval"] / "ce_b_per_query.parquet"
    )
    disagreement = frames["weighted_hybrid"].select(
        "query_key",
        "shared_component_candidates",
        "bm25_dense_spearman",
        "bm25_dense_disagreement_slice",
    )
    for method in ("hybrid_to_cross_encoder", "hybrid_to_lambdamart_to_cross_encoder"):
        frames[method] = frames[method].join(disagreement, on="query_key", how="left")
    return frames


def comparison(learned: dict[str, Any], ce: dict[str, Any]) -> dict[str, Any]:
    output = {
        method: learned["comparison"][method]
        for method in ("bm25", "dense", "weighted_hybrid", "rrf", "pointwise", "lambdamart")
    }
    output["hybrid_to_cross_encoder"] = ce["comparison"]["hybrid_to_cross_encoder"]
    output["hybrid_to_lambdamart_to_cross_encoder"] = ce["comparison"][
        "hybrid_to_lambdamart_to_cross_encoder"
    ]
    return output


def audit(p: dict[str, Path], relevance: pl.DataFrame) -> dict[str, Any]:
    scores = pl.scan_parquet(p["ce"] / "scores.parquet")
    pair_union = pl.scan_parquet(p["ce"] / "pair_union.parquet")
    enriched = pl.scan_parquet(p["ce"] / "scores_enriched.parquet")
    ce_a = pl.scan_parquet(p["eval"] / "ce_a_rankings.parquet")
    ce_b = pl.scan_parquet(p["eval"] / "ce_b_rankings.parquet")
    keys = ["query_key", "product_key", "split"]
    score_rows = scores.select(pl.len()).collect().item()
    union_rows = pair_union.select(pl.len()).collect().item()
    missing = (
        pair_union.join(scores.select(keys), on=keys, how="anti").select(pl.len()).collect().item()
    )
    extra = (
        scores.join(pair_union.select(keys), on=keys, how="anti").select(pl.len()).collect().item()
    )
    duplicates = (
        scores.group_by(keys)
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    invalid = (
        scores.select(
            (
                pl.col("cross_encoder_score").is_null()
                | pl.col("cross_encoder_score").is_nan()
                | pl.col("cross_encoder_score").is_infinite()
            ).sum()
        )
        .collect()
        .item()
    )
    judged = (
        scores.join(
            relevance.select("query_key", "product_key").unique().lazy(),
            on=["query_key", "product_key"],
            how="inner",
        )
        .select(pl.len())
        .collect()
        .item()
    )
    order_a = count_ordering_violations(ce_a)
    order_b = count_ordering_violations(ce_b)
    member_a = (
        ce_a.join(enriched.select(*keys, "in_hybrid_top_100"), on=keys, how="left")
        .filter(~pl.col("in_hybrid_top_100"))
        .select(pl.len())
        .collect()
        .item()
    )
    member_b = (
        ce_b.join(enriched.select(*keys, "in_lambdamart_top_50"), on=keys, how="left")
        .filter(~pl.col("in_lambdamart_top_50"))
        .select(pl.len())
        .collect()
        .item()
    )
    checks = (missing, extra, duplicates, invalid, order_a, order_b, member_a, member_b)
    return {
        "status": "PASS" if all(value == 0 for value in checks) else "FAIL",
        "pair_equality": {
            "union_rows": union_rows,
            "score_rows": score_rows,
            "missing_scores_for_union_pairs": missing,
            "extra_scores_not_in_union": extra,
            "duplicate_score_pairs": duplicates,
            "invalid_score_values": invalid,
        },
        "ordering": {
            "policy": "rank = score DESC, then product_key ASC for ties",
            "ce_a_score_desc_product_key_asc_violations": order_a,
            "ce_b_score_desc_product_key_asc_violations": order_b,
        },
        "membership": {
            "ce_a_non_hybrid_top_100_rows": member_a,
            "ce_b_non_lambdamart_top_50_rows": member_b,
        },
        "unjudged_semantics": {
            "judged_score_rows": judged,
            "unjudged_score_rows": score_rows - judged,
            "policy": "UNJUDGED is preserved separately from I.",
        },
    }


def build_displacement(
    p: dict[str, Path],
    relevance: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    labels = relevance.select("query_key", "product_key", "esci_label").unique(
        ["query_key", "product_key"]
    )
    enriched = pl.scan_parquet(p["ce"] / "scores_enriched.parquet")

    def one(file_name: str, cascade: str, predecessor: str) -> pl.LazyFrame:
        return (
            pl.scan_parquet(p["eval"] / file_name)
            .select(
                "query_key",
                "product_key",
                "split",
                pl.lit(cascade).alias("cascade"),
                pl.col("rank").alias("ce_rank"),
                pl.col("score").alias("ce_score"),
            )
            .join(
                enriched.select(
                    "query_key",
                    "product_key",
                    "split",
                    pl.col(predecessor).alias("predecessor_rank"),
                ),
                on=["query_key", "product_key", "split"],
                how="left",
            )
        )

    frame = (
        pl.concat(
            [
                one("ce_a_rankings.parquet", "ce_a_vs_hybrid", "hybrid_rank"),
                one("ce_b_rankings.parquet", "ce_b_vs_lambdamart", "lambdamart_rank"),
            ]
        )
        .join(labels.lazy(), on=["query_key", "product_key"], how="left")
        .with_columns(label_expr())
        .with_columns((pl.col("ce_rank") - pl.col("predecessor_rank")).alias("rank_displacement"))
        .collect()
    )
    summary = {}
    for cascade in frame["cascade"].unique().sort():
        subset = frame.filter(pl.col("cascade") == cascade)
        by_label = {}
        for label in LABELS:
            values = (
                subset.filter(pl.col("label") == label)["rank_displacement"].drop_nulls().to_list()
            )
            by_label[label] = summarize_displacements(values)
        summary[str(cascade)] = {"by_label": by_label}
    return frame, summary


def displacement_md(summary: dict[str, Any]) -> str:
    lines = [
        "# M3 CE Rank Displacement",
        "",
        "Displacement is CE rank - predecessor rank; negative values are promotions.",
        "UNJUDGED is preserved separately from I.",
        "",
    ]
    for cascade, payload in summary.items():
        lines.extend(
            [
                f"## {cascade}",
                "",
                "| Label | Count | Mean | Median | Promoted | Demoted | abs>=10 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label in LABELS:
            row = payload["by_label"][label]
            lines.append(
                f"| {label} | {row['count']} | {fmt(row['mean'], 3)} | "
                f"{fmt(row['median'], 3)} | {fmt(row['pct_promoted'], 3)} | "
                f"{fmt(row['pct_demoted'], 3)} | {fmt(row['pct_abs_ge_10'], 3)} |"
            )
        lines.append("")
    return "\n".join(lines)


def paired_metrics(per_query: dict[str, pl.DataFrame]) -> pl.DataFrame:
    paired: pl.DataFrame | None = None
    for method, frame in per_query.items():
        columns = [
            column
            for column in ("query_key", "split", "query_text", *METRICS)
            if column in frame.columns
        ]
        renamed = frame.select(columns).rename(
            {metric: f"{method}__{metric}" for metric in METRICS if metric in frame.columns}
        )
        if paired is None:
            paired = renamed
        else:
            paired = paired.join(
                renamed,
                on=["query_key", "split", "query_text"],
                how="full",
                coalesce=True,
            )
    if paired is None:
        raise ValueError("No per-query metrics found.")
    return paired


def query_slices(per_query: dict[str, pl.DataFrame]) -> dict[str, Any]:
    paired = paired_metrics(per_query).join(
        per_query["weighted_hybrid"].select("query_key", "split", *SLICE_COLUMNS.values()),
        on=["query_key", "split"],
        how="left",
    )
    paired = paired.filter(pl.col("split") == "test")
    output: dict[str, Any] = {}
    methods = (
        "weighted_hybrid",
        "pointwise",
        "lambdamart",
        "hybrid_to_cross_encoder",
        "hybrid_to_lambdamart_to_cross_encoder",
    )
    for family, column in SLICE_COLUMNS.items():
        rows = []
        for value in sorted(paired[column].drop_nulls().unique().to_list()):
            subset = paired.filter(pl.col(column) == value)
            row: dict[str, Any] = {"slice": str(value), "split": "test", "queries": subset.height}
            for method in methods:
                for metric in METRICS:
                    metric_mean = subset[f"{method}__{metric}"].mean()
                    if metric_mean is None:
                        row[f"{method}__{metric}"] = None
                    elif isinstance(metric_mean, int | float | Decimal):
                        row[f"{method}__{metric}"] = float(metric_mean)
                    else:
                        raise TypeError(
                            f"Expected numeric mean for {method} {metric}; got {type(metric_mean)}"
                        )
            for metric in METRICS:
                row[f"hybrid_to_cross_encoder_vs_weighted_hybrid__{metric}_delta"] = (
                    mean_metric_delta(subset, metric, "hybrid_to_cross_encoder", "weighted_hybrid")
                )
                row[f"hybrid_to_lambdamart_to_cross_encoder_vs_lambdamart__{metric}_delta"] = (
                    mean_metric_delta(
                        subset,
                        metric,
                        "hybrid_to_lambdamart_to_cross_encoder",
                        "lambdamart",
                    )
                )
                row[f"lambdamart_vs_weighted_hybrid__{metric}_delta"] = mean_metric_delta(
                    subset,
                    metric,
                    "lambdamart",
                    "weighted_hybrid",
                )
                row[f"pointwise_vs_lambdamart__{metric}_delta"] = mean_metric_delta(
                    subset,
                    metric,
                    "pointwise",
                    "lambdamart",
                )
            pointwise_delta = row["pointwise_vs_lambdamart__ndcg_10_delta"]
            row["pointwise_beats_lambdamart_on_ndcg_10"] = pointwise_delta is not None and (
                pointwise_delta > 0
            )
            rows.append(row)
        output[family] = {
            "split": "test",
            "rows": rows,
            "pointwise_beats_lambdamart_slices": [
                row["slice"] for row in rows if row["pointwise_beats_lambdamart_on_ndcg_10"]
            ],
        }
    output["navigational"] = {
        "status": "NOT_IMPLEMENTED",
        "reason": "No defensible navigational intent label exists in saved M3 artifacts.",
    }
    return output


def query_slices_md(summary: dict[str, Any]) -> str:
    lines = ["# M3 Query Slice Summary", "", "All slice results use the test split only.", ""]
    for family, payload in summary.items():
        lines.extend([f"## {family}", ""])
        if payload.get("status"):
            lines.extend([f"{payload['status']}: {payload['reason']}", ""])
            continue
        lines.extend(
            [
                "| Slice | Queries | WH NDCG@5 | WH NDCG@10 | WH MRR | "
                "H->CE dNDCG@10 | H->L->CE dNDCG@10 | LM dNDCG@10 | "
                "PW dNDCG@10 | PW beats LM |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in payload["rows"]:
            lines.append(
                f"| {row['slice']} | {row['queries']} | "
                f"{fmt(row['weighted_hybrid__ndcg_5'])} | "
                f"{fmt(row['weighted_hybrid__ndcg_10'])} | "
                f"{fmt(row['weighted_hybrid__mrr'])} | "
                f"{fmt(row['hybrid_to_cross_encoder_vs_weighted_hybrid__ndcg_10_delta'])} | "
                f"{
                    fmt(row['hybrid_to_lambdamart_to_cross_encoder_vs_lambdamart__ndcg_10_delta'])
                } | "
                f"{fmt(row['lambdamart_vs_weighted_hybrid__ndcg_10_delta'])} | "
                f"{fmt(row['pointwise_vs_lambdamart__ndcg_10_delta'])} | "
                f"{row['pointwise_beats_lambdamart_on_ndcg_10']} |"
            )
        beats = ", ".join(payload["pointwise_beats_lambdamart_slices"]) or "None"
        lines.extend(["", f"Pointwise beats LambdaMART on NDCG@10 in: {beats}.", ""])
    return "\n".join(lines)


def quality_latency(
    learned: dict[str, Any],
    ce: dict[str, Any],
    comp: dict[str, Any],
) -> dict[str, Any]:
    points = {point["method"]: dict(point) for point in learned["quality_latency"]["points"]}
    scoring = ce.get("scoring_stats", {})
    gpu = ce.get("benchmark", {}).get("gpu", {}).get("device_name", "NVIDIA A100-SXM4-40GB")
    roles = {
        "bm25": ("retrieval", 500, None),
        "dense": ("retrieval", 500, None),
        "weighted_hybrid": ("retrieval fusion", 500, None),
        "rrf": ("retrieval fusion", 500, None),
        "pointwise": ("learned reranker", 500, 500),
        "lambdamart": ("learned reranker", 500, 500),
        "hybrid_to_cross_encoder": ("CE reranker", 100, 100),
        "hybrid_to_lambdamart_to_cross_encoder": ("cascade reranker", 500, 50),
    }
    rows = {}
    for method in METHODS:
        point = points.get(method, {})
        role, depth, reranked = roles[method]
        row = {
            "method": METHODS[method],
            "model_role": role,
            "candidate_depth": depth,
            "reranked_depth": reranked,
            "test_ndcg_10": comp[method]["test"].get("ndcg_10"),
            "test_mrr": comp[method]["test"].get("mrr"),
            "hardware": point.get("hardware"),
            "p50_latency_ms": point.get("latency_p50_ms"),
            "p95_latency_ms": point.get("latency_p95_ms"),
            "throughput_queries_per_second": point.get("throughput_queries_per_second"),
            "hardware_mixed": "mixed" in str(point.get("hardware", "")).lower(),
            "notes": "Existing measurements only; stage scope differs by method.",
        }
        if method.startswith("hybrid_to_"):
            row.update(
                {
                    "hardware": gpu,
                    "p50_latency_ms": None,
                    "p95_latency_ms": None,
                    "throughput_pairs_per_second": scoring.get("pairs_per_second"),
                    "hardware_mixed": True,
                    "notes": (
                        "Offline batch scoring throughput only; no online p50/p95 "
                        "request latency was measured."
                    ),
                }
            )
        rows[method] = row
    return {
        "caveat": (
            "Hardware and stage scope differ; CE has imported A100 batch throughput "
            "but no measured online request latency."
        ),
        "methods": rows,
    }


def quality_latency_md(summary: dict[str, Any]) -> str:
    lines = [
        "# M3 Quality-Latency Summary",
        "",
        summary["caveat"],
        "",
        "| Method | Test NDCG@10 | Test MRR | Depth | Reranked | p50 ms | p95 ms | "
        "Throughput | Hardware | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["methods"].values():
        throughput = row.get("throughput_queries_per_second")
        if throughput is None:
            throughput = row.get("throughput_pairs_per_second")
        reranked = "NA" if row["reranked_depth"] is None else str(row["reranked_depth"])
        lines.append(
            f"| {row['method']} | {fmt(row['test_ndcg_10'])} | {fmt(row['test_mrr'])} | "
            f"{row['candidate_depth']} | {reranked} | {fmt(row['p50_latency_ms'], 3)} | "
            f"{fmt(row['p95_latency_ms'], 3)} | {fmt(throughput, 3)} | "
            f"{row['hardware']} | {row['notes']} |"
        )
    return "\n".join(lines)


def final_table(comp: dict[str, Any]) -> list[dict[str, Any]]:
    roles = {
        "bm25": ("retrieval baseline", 500),
        "dense": ("retrieval baseline", 500),
        "weighted_hybrid": ("retrieval fusion", 500),
        "rrf": ("retrieval fusion", 500),
        "pointwise": ("learned reranker", 500),
        "lambdamart": ("learned reranker", 500),
        "hybrid_to_cross_encoder": ("cross-encoder reranker", 100),
        "hybrid_to_lambdamart_to_cross_encoder": ("cascade reranker", 50),
    }
    rows = []
    for method, name in METHODS.items():
        test = comp[method]["test"]
        role, depth = roles[method]
        rows.append(
            {
                "method": name,
                "method_key": method,
                "model_role": role,
                "candidate_depth": depth,
                "queries": test.get("queries"),
                "ndcg_5": test.get("ndcg_5"),
                "ndcg_10": test.get("ndcg_10"),
                "mrr": test.get("mrr"),
                "recall_10": test.get("recall_primary_10"),
                "recall_50": test.get("recall_primary_50"),
                "recall_100": test.get("recall_primary_100"),
                "recall_500": test.get("recall_primary_500"),
            }
        )
    return rows


def final_table_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# M3 Final Comparison Table",
        "",
        "| Method | Role | Queries | NDCG@5 | NDCG@10 | MRR | Recall@10 | Recall@50 | "
        "Recall@100 | Recall@500 | Depth |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['model_role']} | {row['queries']} | "
            f"{fmt(row['ndcg_5'])} | {fmt(row['ndcg_10'])} | {fmt(row['mrr'])} | "
            f"{fmt(row['recall_10'])} | {fmt(row['recall_50'])} | "
            f"{fmt(row['recall_100'])} | {fmt(row['recall_500'])} | "
            f"{row['candidate_depth']} |"
        )
    return "\n".join(lines)


def ablation(p: dict[str, Path]) -> dict[str, Any]:
    scored = pl.scan_parquet(p["ce"] / "scores.parquet").select(
        "query_key",
        "product_key",
        "split",
        pl.lit(1).alias("has_ce_score"),
    )
    rows = []
    for split in ("train", "validation", "test"):
        rankings = pl.scan_parquet(p["learned"] / f"rankings_{split}.parquet").select(
            "query_key",
            "product_key",
            "split",
            "hybrid_rank",
            "lambdamart_rank",
        )
        row = (
            rankings.join(scored, on=["query_key", "product_key", "split"], how="left")
            .select(
                pl.len().alias("candidate_rows"),
                pl.col("has_ce_score").is_not_null().sum().alias("rows_with_ce_score"),
                (pl.col("hybrid_rank") <= 100).sum().alias("hybrid_top_100_rows"),
                (pl.col("has_ce_score").is_not_null() & (pl.col("hybrid_rank") <= 100))
                .sum()
                .alias("hybrid_top_100_rows_with_ce_score"),
                (pl.col("lambdamart_rank") <= 50).sum().alias("lambdamart_top_50_rows"),
                (pl.col("has_ce_score").is_not_null() & (pl.col("lambdamart_rank") <= 50))
                .sum()
                .alias("lambdamart_top_50_rows_with_ce_score"),
            )
            .collect()
            .to_dicts()[0]
        )
        row["split"] = split
        row["full_candidate_ce_coverage"] = row["rows_with_ce_score"] / row["candidate_rows"]
        row["hybrid_top_100_ce_coverage"] = (
            row["hybrid_top_100_rows_with_ce_score"] / row["hybrid_top_100_rows"]
        )
        row["lambdamart_top_50_ce_coverage"] = (
            row["lambdamart_top_50_rows_with_ce_score"] / row["lambdamart_top_50_rows"]
        )
        rows.append(row)
    return {
        "status": "DOCUMENTED_ONLY",
        "coverage_by_split": rows,
        "no_new_inference_needed_for_existing_cascades": all(
            row["hybrid_top_100_ce_coverage"] == 1.0 and row["lambdamart_top_50_ce_coverage"] == 1.0
            for row in rows
        ),
        "full_500_feature_matrix_complete": all(
            row["full_candidate_ce_coverage"] == 1.0 for row in rows
        ),
        "recommendation": "Do not implement in this closeout; define a separate experiment.",
    }


def ablation_md(summary: dict[str, Any]) -> str:
    lines = [
        "# LambdaMART + CE Score Feature Ablation Feasibility",
        "",
        f"Status: {summary['status']}",
        "",
        "| Split | Candidate rows | CE coverage full | Hybrid top 100 | LambdaMART top 50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["coverage_by_split"]:
        lines.append(
            f"| {row['split']} | {row['candidate_rows']} | "
            f"{fmt(row['full_candidate_ce_coverage'], 3)} | "
            f"{fmt(row['hybrid_top_100_ce_coverage'], 3)} | "
            f"{fmt(row['lambdamart_top_50_ce_coverage'], 3)} |"
        )
    lines.extend(
        [
            "",
            "No new inference needed for existing cascades: "
            f"{summary['no_new_inference_needed_for_existing_cascades']}.",
            f"Full 500-candidate matrix complete: {summary['full_500_feature_matrix_complete']}.",
            "",
            f"Recommendation: {summary['recommendation']}",
        ]
    )
    return "\n".join(lines)


def example_entities(p: dict[str, Path]) -> tuple[pl.DataFrame, pl.DataFrame]:
    queries = (
        pl.read_parquet(p["dataset"] / "queries.parquet")
        .select("query_key", "query_text")
        .unique("query_key")
    )
    products = (
        pl.scan_parquet(p["dataset"] / "catalog.parquet")
        .select("product_key", "title", "brand")
        .collect()
        .unique("product_key")
    )
    return queries, products


def score_distribution_by_label(
    p: dict[str, Path], relevance: pl.DataFrame
) -> dict[str, dict[str, Any]]:
    labels = relevance.select("query_key", "product_key", "esci_label").unique(
        ["query_key", "product_key"]
    )
    rows = (
        pl.scan_parquet(p["ce"] / "scores_enriched.parquet")
        .join(labels.lazy(), on=["query_key", "product_key"], how="left")
        .with_columns(label_expr())
        .group_by("label")
        .agg(
            pl.len().alias("count"),
            pl.col("cross_encoder_score").mean().alias("mean"),
            pl.col("cross_encoder_score").median().alias("median"),
            pl.col("cross_encoder_score").quantile(0.25).alias("p25"),
            pl.col("cross_encoder_score").quantile(0.75).alias("p75"),
        )
        .collect()
        .to_dicts()
    )
    mapped = {row["label"]: clean(row) for row in rows}
    empty_row = {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {label: mapped.get(label, {"label": label, **empty_row}) for label in LABELS}


def score_distribution_by_label_md(summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# M3 CE Score Distribution By Label",
        "",
        "UNJUDGED is preserved separately from ESCI I.",
        "",
        "| Label | Count | Mean | Median | p25 | p75 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = summary[label]
        lines.append(
            f"| {label} | {row['count']} | {fmt(row['mean'])} | "
            f"{fmt(row['median'])} | {fmt(row['p25'])} | {fmt(row['p75'])} |"
        )
    return "\n".join(lines)


def example_base(
    p: dict[str, Path], relevance: pl.DataFrame, queries: pl.DataFrame, products: pl.DataFrame
) -> pl.DataFrame:
    labels = relevance.select("query_key", "product_key", "esci_label").unique(
        ["query_key", "product_key"]
    )
    base = (
        pl.scan_parquet(p["ce"] / "scores_enriched.parquet")
        .filter(pl.col("split") == "test")
        .join(labels.lazy(), on=["query_key", "product_key"], how="left")
        .with_columns(label_expr())
        .join(queries.lazy(), on="query_key", how="left")
        .join(products.lazy(), on="product_key", how="left")
        .collect()
    )
    blanks = base.filter(
        pl.col("query_text").is_null() | (pl.col("query_text").str.strip_chars() == "")
    )
    if blanks.height:
        raise ValueError("representative examples have blank query_text")
    return base


def example_rows(
    frame: pl.DataFrame, category: str, delta: float | None = None, limit: int = 12
) -> pl.DataFrame:
    return (
        frame.head(limit)
        .select(
            "query_key",
            "query_text",
            "product_key",
            "title",
            "brand",
            "label",
            "hybrid_rank",
            "lambdamart_rank",
            "cross_encoder_score",
        )
        .with_columns(
            pl.lit(category).alias("category"),
            pl.lit(delta).alias("query_delta_ndcg_10"),
        )
        .select(
            "category",
            "query_key",
            "query_text",
            "product_key",
            "title",
            "brand",
            "label",
            "hybrid_rank",
            "lambdamart_rank",
            "cross_encoder_score",
            "query_delta_ndcg_10",
        )
    )


def keyword_compatibility_examples_base(base: pl.DataFrame) -> pl.DataFrame:
    return base.filter(pl.col("in_hybrid_top_100")).sort(
        ["cross_encoder_score", "query_key", "product_key"],
        descending=[False, False, False],
    )


def win_loss_examples(
    base: pl.DataFrame,
    per_query: dict[str, pl.DataFrame],
    category: str,
    method: str,
    baseline: str,
    *,
    descending: bool,
) -> list[pl.DataFrame]:
    selected = (
        per_query[method]
        .select("query_key", "split", pl.col("ndcg_10").alias("method_ndcg_10"))
        .join(
            per_query[baseline].select(
                "query_key", "split", pl.col("ndcg_10").alias("baseline_ndcg_10")
            ),
            on=["query_key", "split"],
            how="inner",
        )
        .filter(pl.col("split") == "test")
        .with_columns((pl.col("method_ndcg_10") - pl.col("baseline_ndcg_10")).alias("delta"))
        .sort(["delta", "query_key"], descending=[descending, False])
        .head(4)
    )
    output = []
    for row in selected.to_dicts():
        rows = base.filter(pl.col("query_key") == row["query_key"]).sort(
            ["cross_encoder_score", "product_key"], descending=[True, False]
        )
        output.append(example_rows(rows, category, row["delta"], limit=3))
    return output


def build_representative_examples(
    p: dict[str, Path],
    relevance: pl.DataFrame,
    per_query: dict[str, pl.DataFrame],
) -> tuple[pl.DataFrame, dict[str, list[dict[str, Any]]]]:
    queries, products = example_entities(p)
    base = example_base(p, relevance, queries, products)
    frames = []
    frames.extend(
        win_loss_examples(
            base,
            per_query,
            "ce_a_largest_wins_vs_hybrid",
            "hybrid_to_cross_encoder",
            "weighted_hybrid",
            descending=True,
        )
    )
    frames.extend(
        win_loss_examples(
            base,
            per_query,
            "ce_a_largest_losses_vs_hybrid",
            "hybrid_to_cross_encoder",
            "weighted_hybrid",
            descending=False,
        )
    )
    frames.extend(
        win_loss_examples(
            base,
            per_query,
            "ce_b_largest_wins_vs_lambdamart",
            "hybrid_to_lambdamart_to_cross_encoder",
            "lambdamart",
            descending=True,
        )
    )
    frames.extend(
        win_loss_examples(
            base,
            per_query,
            "ce_b_largest_losses_vs_lambdamart",
            "hybrid_to_lambdamart_to_cross_encoder",
            "lambdamart",
            descending=False,
        )
    )
    frames.append(
        example_rows(
            base.filter(pl.col("label") == "I").sort(
                ["cross_encoder_score", "query_key", "product_key"],
                descending=[True, False, False],
            ),
            "high_scoring_I",
        )
    )
    frames.append(
        example_rows(
            base.filter(pl.col("label") == "E").sort(
                ["cross_encoder_score", "query_key", "product_key"],
                descending=[False, False, False],
            ),
            "low_scoring_E",
        )
    )
    inversions = (
        base.filter(pl.col("label").is_in(["S", "C"]))
        .group_by("query_key")
        .agg(
            pl.col("cross_encoder_score").filter(pl.col("label") == "S").max().alias("max_s"),
            pl.col("cross_encoder_score").filter(pl.col("label") == "C").max().alias("max_c"),
        )
        .filter(pl.col("max_s").is_not_null() & pl.col("max_c").is_not_null())
        .with_columns((pl.col("max_c") - pl.col("max_s")).alias("delta"))
        .filter(pl.col("delta") > 0)
        .sort(["delta", "query_key"], descending=[True, False])
        .head(6)
    )
    for row in inversions.to_dicts():
        rows = base.filter(
            (pl.col("query_key") == row["query_key"]) & pl.col("label").is_in(["S", "C"])
        ).sort(["cross_encoder_score", "product_key"], descending=[True, False])
        frames.append(example_rows(rows, "S_C_inversion", row["delta"], limit=2))
    frames.append(
        example_rows(
            keyword_compatibility_examples_base(base),
            "keyword_compatibility_or_product_type_mismatch",
        )
    )
    frame = pl.concat(frames, how="vertical")
    grouped = frame.group_by("category", maintain_order=True).agg(
        pl.struct(pl.all().exclude("category")).alias("rows")
    )
    return frame, clean({category: rows for category, rows in grouped.iter_rows()})


def representative_examples_md(examples: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# M3 CE Representative Examples",
        "",
        "All rows use the test split only. Query text is joined from `queries.parquet`; "
        "product title and brand are joined from `catalog.parquet`.",
        "",
    ]
    for category, rows in examples.items():
        lines.extend(
            [
                f"## {category}",
                "",
                "| Query | Product | Brand | Label | Hybrid | LambdaMART | Score | Delta |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['query_text']} | {row['title']} | {row['brand'] or 'NA'} | "
                f"{row['label']} | {row['hybrid_rank'] or 'NA'} | "
                f"{row['lambdamart_rank'] or 'NA'} | "
                f"{fmt(row['cross_encoder_score'], 4)} | "
                f"{fmt(row['query_delta_ndcg_10'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["method_key"]: row for row in rows}


def delta(rows: dict[str, dict[str, Any]], method: str, base: str, metric: str) -> float | None:
    method_value = rows[method].get(metric)
    base_value = rows[base].get(metric)
    if method_value is None or base_value is None:
        return None
    return float(method_value) - float(base_value)


def _delta_for_min(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float("inf") if value is None else float(value)


def slice_notes(summary: dict[str, Any]) -> str:
    notes = []
    for family, payload in summary.items():
        if payload.get("status"):
            notes.append(f"- {family}: {payload['status']} ({payload['reason']})")
            continue
        if not payload.get("rows"):
            notes.append(f"- {family}: no slice rows available.")
            continue
        beats = ", ".join(payload["pointwise_beats_lambdamart_slices"]) or "none"
        ce_a_key = "hybrid_to_cross_encoder_vs_weighted_hybrid__ndcg_10_delta"
        ce_b_key = "hybrid_to_lambdamart_to_cross_encoder_vs_lambdamart__ndcg_10_delta"
        worst_a = min(payload["rows"], key=lambda row: _delta_for_min(row, ce_a_key))
        worst_b = min(payload["rows"], key=lambda row: _delta_for_min(row, ce_b_key))
        notes.append(
            f"- {family}: Pointwise beats LambdaMART in {beats}; worst Hybrid->CE "
            f"dNDCG@10 is {worst_a['slice']} ({fmt(worst_a[ce_a_key])}); "
            f"worst H->L->CE dNDCG@10 is {worst_b['slice']} ({fmt(worst_b[ce_b_key])})."
        )
    return "\n".join(notes)


def main() -> None:
    start = time.time()
    root = project_root()
    p = paths(root)
    p["analysis"].mkdir(parents=True, exist_ok=True)
    learned = read_json(p["learned"] / "analysis" / "ranking_analysis.json")
    ce = read_json(p["eval"] / "cascade_report.json")
    comp = comparison(learned, ce)
    relevance = pl.read_parquet(p["dataset"] / "relevance.parquet")
    per_query = load_per_query(p)
    correctness = audit(p, relevance)
    write_json(p["analysis"] / "ce_correctness_audit.json", correctness)
    disp_frame, disp_summary = build_displacement(p, relevance)
    disp_frame.write_parquet(p["analysis"] / "rank_displacement.parquet")
    write_json(p["analysis"] / "rank_displacement_summary.json", disp_summary)
    write_text(p["analysis"] / "rank_displacement_summary.md", displacement_md(disp_summary))
    slice_summary = query_slices(per_query)
    write_json(p["analysis"] / "query_slice_summary.json", slice_summary)
    write_text(p["analysis"] / "query_slice_summary.md", query_slices_md(slice_summary))
    latency = quality_latency(learned, ce, comp)
    write_json(p["analysis"] / "quality_latency_summary.json", latency)
    write_text(p["analysis"] / "quality_latency_summary.md", quality_latency_md(latency))
    rows = final_table(comp)
    write_json(p["analysis"] / "final_comparison_table.json", rows)
    write_text(p["analysis"] / "final_comparison_table.md", final_table_md(rows))
    ablate = ablation(p)
    write_json(p["analysis"] / "ce_score_feature_ablation_feasibility.json", ablate)
    write_text(
        p["analysis"] / "ce_score_feature_ablation_feasibility.md",
        ablation_md(ablate),
    )
    score_by_label = score_distribution_by_label(p, relevance)
    write_json(p["analysis"] / "score_distribution_by_label.json", score_by_label)
    write_text(
        p["analysis"] / "score_distribution_by_label.md",
        score_distribution_by_label_md(score_by_label),
    )
    example_frame, representative = build_representative_examples(p, relevance, per_query)
    example_frame.write_parquet(p["analysis"] / "representative_examples.parquet")
    write_json(p["analysis"] / "representative_examples.json", representative)
    write_text(
        p["analysis"] / "representative_examples.md",
        representative_examples_md(representative),
    )
    source = {
        "dataset_fingerprint": FP,
        "paths": {key: str(value) for key, value in p.items() if key != "docs"},
        "model": ce["model"],
        "coverage": ce["coverage"],
        "scoring_stats": ce["scoring_stats"],
        "validation_report": read_json(p["ce"] / "validation_report.json"),
        "score_distribution": read_json(p["ce"] / "score_distribution.json"),
        "score_distribution_by_label": score_by_label,
        "comparison": comp,
        "audit": correctness,
        "displacement_summary": disp_summary,
        "query_slice_summary": slice_summary,
        "quality_latency": latency,
        "feature_ablation": ablate,
        "representative_examples": representative,
    }
    write_json(p["analysis"] / "m3_findings_source.json", source)
    write_text(
        p["docs"] / "m3_findings.md",
        expanded_findings_doc(p, source, rows),
    )
    dirty = git(["status", "--porcelain"], root)
    metadata = {
        "script_path": str(p["script"]),
        "dataset_fingerprint": FP,
        "start_time_unix": start,
        "end_time_unix": time.time(),
        "status": "SUCCESS",
        "git_commit": git(["rev-parse", "HEAD"], root),
        "git_dirty": bool(dirty),
        "input_artifact_shas": {
            "pair_union": sha256_file(p["ce"] / "pair_union.parquet"),
            "scores": sha256_file(p["ce"] / "scores.parquet"),
            "scores_enriched": sha256_file(p["ce"] / "scores_enriched.parquet"),
            "cascade_report": sha256_file(p["eval"] / "cascade_report.json"),
            "ranking_analysis": sha256_file(p["learned"] / "analysis" / "ranking_analysis.json"),
        },
        "output_dir": str(p["analysis"]),
    }
    write_json(p["analysis"] / "run_metadata.json", metadata)
    print(p["analysis"])


if __name__ == "__main__":
    main()
