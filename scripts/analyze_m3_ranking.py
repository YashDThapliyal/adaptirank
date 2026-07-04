"""Run all cheap M3 baselines through one harness and produce slice/latency analysis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.run import ExperimentRun
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.config import LearnedRankingRunConfig
from adaptirank.ranking.evaluate import evaluate_ranking, ranked_candidates
from adaptirank.retrieval.evaluate import write_json


def _retrieval_candidates(path: Path, method: str) -> pl.DataFrame:
    return pl.read_parquet(path).select(
        "query_key", "product_key", "split", pl.lit(method).alias("method"), "score", "rank"
    )


def _mean_slices(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    metrics = (
        "ndcg_5",
        "ndcg_10",
        "mrr",
        "average_precision",
        "recall_primary_10",
        "recall_primary_100",
    )
    return {
        str(value): {
            "queries": subset.height,
            **{
                name: float(np.mean(subset.get_column(name).cast(pl.Float64).to_numpy()))
                for name in metrics
            },
        }
        for value in frame.get_column(column).drop_nulls().unique().sort().to_list()
        if (subset := frame.filter(pl.col(column) == value)).height
    }


def _disagreement(feature_dir: Path) -> pl.DataFrame:
    frames = []
    for split in ("validation", "test"):
        frames.append(
            pl.scan_parquet(feature_dir / f"{split}.parquet").filter(
                pl.col("bm25_rank").is_not_null() & pl.col("dense_rank").is_not_null()
            )
        )
    return (
        pl.concat(frames)
        .group_by("query_key")
        .agg(
            pl.len().alias("shared_component_candidates"),
            pl.corr("bm25_rank", "dense_rank", method="spearman").alias("bm25_dense_spearman"),
        )
        .with_columns(
            pl.when(pl.col("bm25_dense_spearman").is_null())
            .then(pl.lit("no_shared_order"))
            .when(pl.col("bm25_dense_spearman") < 0.25)
            .then(pl.lit("high_disagreement"))
            .when(pl.col("bm25_dense_spearman") < 0.75)
            .then(pl.lit("medium_disagreement"))
            .otherwise(pl.lit("low_disagreement"))
            .alias("bm25_dense_disagreement_slice")
        )
        .collect()
    )


def _heuristic_latency(frame: pl.DataFrame, query_limit: int) -> dict[str, float]:
    values = []
    for key in frame.get_column("query_key").unique(maintain_order=True).head(query_limit):
        block = frame.filter(pl.col("query_key") == key)
        started = time.perf_counter()
        _ = (
            0.80 * block.get_column("hybrid_score").to_numpy()
            + 0.15 * block.get_column("lexical_overlap").to_numpy()
            + 0.05 * block.get_column("brand_match").to_numpy()
        )
        values.append((time.perf_counter() - started) * 1000)
    array = np.asarray(values)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "throughput_queries_per_second": 1000.0 / float(np.mean(array)),
    }


def _svg(points: list[dict[str, Any]]) -> str:
    width, height = 960, 560
    left, top, plot_w, plot_h = 90, 50, 780, 420
    xs = [float(point["latency_p50_ms"]) for point in points if point["latency_p50_ms"] > 0]
    ys = [float(point["test_ndcg_10"]) for point in points]
    x_max = max(xs, default=1.0) * 1.1
    y_min, y_max = min(ys) - 0.005, max(ys) + 0.005
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="480" y="26" text-anchor="middle" font-size="18">'
            "M3 quality-latency map (hardware mixed)</text>"
        ),
        (
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
            f'y2="{top + plot_h}" stroke="black"/>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        (
            f'<text x="{left + plot_w / 2}" y="530" text-anchor="middle">'
            "p50 latency (ms; hardware-labeled)</text>"
        ),
        (
            f'<text x="20" y="{top + plot_h / 2}" '
            f'transform="rotate(-90 20 {top + plot_h / 2})" text-anchor="middle">'
            "Test NDCG@10</text>"
        ),
    ]
    for point in points:
        x_value = float(point["latency_p50_ms"])
        if x_value <= 0:
            continue
        y_value = float(point["test_ndcg_10"])
        x = left + plot_w * x_value / x_max
        y = top + plot_h * (y_max - y_value) / (y_max - y_min)
        label = f"{point['method']} ({point['hardware']})"
        lines.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#2563eb"/>',
                f'<text x="{x + 9:.1f}" y="{y - 7:.1f}" font-size="12">{label}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, LearnedRankingRunConfig)
    root = project_root()
    feature_dir = resolve_project_path(config.feature_dir, root)
    retrieval = root / "artifacts" / "retrieval" / config.dataset_fingerprint / "m3_three_split"
    learned = (
        resolve_project_path(config.output_dir, root)
        / config.dataset_fingerprint
        / config.artifact_name
        / "learned"
    )
    output = learned / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    queries = pl.read_parquet(config.dataset_dir / "queries.parquet").filter(
        pl.col("benchmark_split").is_in(["validation", "test"])
    )
    relevance = pl.read_parquet(config.dataset_dir / "relevance.parquet")
    catalog = pl.read_parquet(config.dataset_dir / "catalog.parquet")
    disagreement = _disagreement(feature_dir)
    retrieval_sources = {
        "bm25": retrieval / "bm25" / "selected_candidates.parquet",
        "dense": retrieval / "dense" / "raw_candidates.parquet",
        "weighted_hybrid": retrieval / "hybrid" / "weighted" / "raw_candidates.parquet",
        "rrf": retrieval / "hybrid" / "rrf" / "raw_candidates.parquet",
    }
    learned_report = json.loads((learned / "report.json").read_text())
    with ExperimentRun(
        experiment="ranking_m3_analysis",
        purpose="m3_shared_harness_and_slice_analysis",
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        comparison = {}
        per_query_by_method = {}
        for method, path in retrieval_sources.items():
            candidates = _retrieval_candidates(path, method).filter(
                pl.col("split").is_in(["validation", "test"])
            )
            _, per_query, metrics = evaluate_ranking(
                candidates,
                method=method,
                queries=queries,
                relevance=relevance,
                catalog=catalog,
            )
            comparison[method] = metrics["by_split"]
            per_query_by_method[method] = per_query
        learned_frames = {
            split: pl.read_parquet(learned / f"rankings_{split}.parquet")
            for split in ("validation", "test")
        }
        learned_all = pl.concat(list(learned_frames.values()))
        for method in ("heuristic", "pointwise", "lambdamart"):
            candidates = ranked_candidates(
                learned_all, f"{method}_score", f"{method}_analysis_rank"
            )
            _, per_query, metrics = evaluate_ranking(
                candidates,
                method=method,
                queries=queries,
                relevance=relevance,
                catalog=catalog,
            )
            comparison[method] = metrics["by_split"]
            per_query_by_method[method] = per_query

        slices = {}
        per_query_dir = output / "per_query"
        per_query_dir.mkdir(exist_ok=True)
        for method, per_query in per_query_by_method.items():
            enriched = per_query.join(disagreement, on="query_key", how="left")
            enriched.write_parquet(per_query_dir / f"{method}.parquet")
            test_enriched = enriched.filter(pl.col("split") == "test")
            slices[method] = {
                "query_length": _mean_slices(test_enriched, "query_length_slice"),
                "lexical_overlap": _mean_slices(test_enriched, "lexical_overlap_slice"),
                "bm25_dense_disagreement": _mean_slices(
                    test_enriched, "bm25_dense_disagreement_slice"
                ),
            }

        hybrid = per_query_by_method["weighted_hybrid"].select(
            "query_key", pl.col("ndcg_10").alias("hybrid_ndcg_10")
        )
        lambda_delta = (
            per_query_by_method["lambdamart"]
            .filter(pl.col("split") == "test")
            .join(hybrid, on="query_key")
            .with_columns((pl.col("ndcg_10") - pl.col("hybrid_ndcg_10")).alias("ndcg_10_delta"))
        )
        examples = {
            "largest_lambdamart_wins": lambda_delta.sort("ndcg_10_delta", descending=True)
            .head(20)
            .select("query_key", "query_text", "ndcg_10_delta", "hybrid_ndcg_10", "ndcg_10")
            .to_dicts(),
            "largest_lambdamart_failures": lambda_delta.sort("ndcg_10_delta")
            .head(20)
            .select("query_key", "query_text", "ndcg_10_delta", "hybrid_ndcg_10", "ndcg_10")
            .to_dicts(),
        }
        latency_sources = {
            method: json.loads((retrieval / relative).read_text())["latency"]
            for method, relative in {
                "bm25": Path("bm25/title/metrics.json"),
                "dense": Path("dense/metrics.json"),
                "weighted_hybrid": Path("hybrid/weighted/metrics.json"),
                "rrf": Path("hybrid/rrf/metrics.json"),
            }.items()
        }
        latency_sources["heuristic"] = _heuristic_latency(
            learned_frames["test"], config.ranking.latency_query_sample
        )
        for method in ("pointwise", "lambdamart"):
            latency_sources[method] = learned_report["comparison"][method]["latency"]
        hardware = {
            "bm25": "local macOS CPU retrieval",
            "dense": "Colab A100 CUDA retrieval",
            "weighted_hybrid": "mixed local CPU + Colab A100",
            "rrf": "mixed local CPU + Colab A100",
            "heuristic": "local macOS CPU score-only",
            "pointwise": "local macOS CPU score-only",
            "lambdamart": "local macOS CPU score-only",
        }
        quality_latency = [
            {
                "method": method,
                "test_ndcg_10": comparison[method]["test"]["ndcg_10"],
                "latency_p50_ms": latency_sources[method]["p50_ms"],
                "latency_p95_ms": latency_sources[method]["p95_ms"],
                "hardware": hardware[method],
            }
            for method in comparison
        ]
        latency_caveat = (
            "hardware and stage scope differ; do not infer a cross-method latency winner"
        )
        report = {
            "dataset_fingerprint": config.dataset_fingerprint,
            "evaluation_harness": "judged-aware condensed MRR/NDCG/MAP; raw-rank recall",
            "comparison": comparison,
            "slices": slices,
            "representative_examples": examples,
            "quality_latency": {
                "points": quality_latency,
                "caveat": latency_caveat,
            },
            "m2_findings_revisited": {
                "dense_below_bm25": comparison["dense"]["test"]["ndcg_10"]
                < comparison["bm25"]["test"]["ndcg_10"],
                "fusion_above_components": comparison["weighted_hybrid"]["test"]["ndcg_10"]
                > max(
                    comparison["dense"]["test"]["ndcg_10"], comparison["bm25"]["test"]["ndcg_10"]
                ),
                "bm25_selected_fields": "title",
                "short_query_lambda_ndcg_10": slices["lambdamart"]["query_length"]["short_1_2"][
                    "ndcg_10"
                ],
                "zero_overlap_lambda_ndcg_10": slices["lambdamart"]["lexical_overlap"]["none"][
                    "ndcg_10"
                ],
            },
        }
        report_path = output / "ranking_analysis.json"
        plot_path = output / "quality_latency.svg"
        write_json(report_path, report)
        plot_path.write_text(_svg(quality_latency))
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("ranking_analysis", report_path)
        run.record_artifact("quality_latency_plot", plot_path)
        run.record_metrics(
            {
                "methods": list(comparison),
                "analysis_sha256": sha256_file(report_path),
                "quality_latency_caveat": latency_caveat,
            }
        )
        print(run.run_dir)


if __name__ == "__main__":
    main()
