"""Evaluate standalone CE and learned LambdaMART-to-CE cascades from shared scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.run import ExperimentRun
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.config import CEEvaluationRunConfig
from adaptirank.ranking.evaluate import evaluate_ranking, ranked_candidates
from adaptirank.retrieval.evaluate import write_json


def _metrics(
    candidates: pl.DataFrame,
    *,
    method: str,
    queries: pl.DataFrame,
    relevance: pl.DataFrame,
    catalog: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame]:
    _, per_query, metrics = evaluate_ranking(
        candidates,
        method=method,
        queries=queries,
        relevance=relevance,
        catalog=catalog,
    )
    return metrics["by_split"], per_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, CEEvaluationRunConfig)
    root = project_root()
    learned = resolve_project_path(config.learned_root, root)
    ce_root = resolve_project_path(config.cross_encoder_root, root)
    scores_path = resolve_project_path(config.scores_path, root)
    union_path = ce_root / "pair_union.parquet"
    union = pl.read_parquet(union_path)
    scores = pl.read_parquet(scores_path)
    if scores.height != union.height:
        raise ValueError(f"CE scores {scores.height} do not cover union {union.height}")
    missing = union.join(
        scores.select("query_key", "product_key"),
        on=["query_key", "product_key"],
        how="anti",
    )
    if not missing.is_empty():
        raise ValueError(f"CE scores miss {missing.height} union pairs")
    scored = union.join(scores, on=["query_key", "product_key", "split"], how="inner")
    invalid = scored.select(
        (
            pl.col("cross_encoder_score").is_null()
            | pl.col("cross_encoder_score").is_nan()
            | pl.col("cross_encoder_score").is_infinite()
        ).sum()
    ).item()
    if invalid:
        raise ValueError(f"CE scores contain {invalid} invalid values")

    ce_a_frame = scored.filter(pl.col("in_hybrid_top_100"))
    ce_b_frame = scored.filter(pl.col("in_lambdamart_top_50"))
    ce_a = ranked_candidates(ce_a_frame, "cross_encoder_score", "ce_a_rank")
    ce_b = ranked_candidates(ce_b_frame, "cross_encoder_score", "ce_b_rank")
    queries = pl.read_parquet(config.dataset_dir / "queries.parquet").filter(
        pl.col("benchmark_split").is_in(["validation", "test"])
    )
    relevance = pl.read_parquet(config.dataset_dir / "relevance.parquet")
    catalog = pl.read_parquet(config.dataset_dir / "catalog.parquet")
    ce_a_eval = ce_a.filter(pl.col("split").is_in(["validation", "test"]))
    ce_b_eval = ce_b.filter(pl.col("split").is_in(["validation", "test"]))
    ce_a_metrics, ce_a_per_query = _metrics(
        ce_a_eval,
        method="hybrid_to_cross_encoder",
        queries=queries,
        relevance=relevance,
        catalog=catalog,
    )
    ce_b_metrics, ce_b_per_query = _metrics(
        ce_b_eval,
        method="hybrid_to_lambdamart_to_cross_encoder",
        queries=queries,
        relevance=relevance,
        catalog=catalog,
    )
    output = ce_root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    ce_a.write_parquet(output / "ce_a_rankings.parquet")
    ce_b.write_parquet(output / "ce_b_rankings.parquet")
    ce_a_per_query.write_parquet(output / "ce_a_per_query.parquet")
    ce_b_per_query.write_parquet(output / "ce_b_per_query.parquet")
    baseline = json.loads((learned / "analysis" / "ranking_analysis.json").read_text())[
        "comparison"
    ]
    scoring_stats_path = ce_root / "scoring_stats.json"
    scoring_stats = (
        json.loads(scoring_stats_path.read_text()) if scoring_stats_path.is_file() else {}
    )
    benchmark_path = ce_root / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else {}
    report = {
        "dataset_fingerprint": config.dataset_fingerprint,
        "model": {
            "name": "cross-encoder/ms-marco-MiniLM-L12-v2",
            "revision": "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
            "role": "pretrained MS MARCO baseline; not e-commerce fine-tuned",
        },
        "coverage": {
            "pair_union_rows": union.height,
            "score_rows": scores.height,
            "missing_pairs": missing.height,
            "invalid_scores": invalid,
            "pair_union_sha256": sha256_file(union_path),
            "scores_sha256": sha256_file(scores_path),
        },
        "comparison": {
            "hybrid": baseline["weighted_hybrid"],
            "hybrid_to_lambdamart": baseline["lambdamart"],
            "hybrid_to_cross_encoder": ce_a_metrics,
            "hybrid_to_lambdamart_to_cross_encoder": ce_b_metrics,
        },
        "scoring_stats": scoring_stats,
        "benchmark": benchmark,
    }
    report_path = output / "cascade_report.json"
    write_json(report_path, report)
    with ExperimentRun(
        experiment=config.run.experiment,
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("cascade_report", report_path)
        run.record_artifact("ce_a_rankings", output / "ce_a_rankings.parquet")
        run.record_artifact("ce_b_rankings", output / "ce_b_rankings.parquet")
        run.record_metrics(report)
        print(run.run_dir)


if __name__ == "__main__":
    main()
