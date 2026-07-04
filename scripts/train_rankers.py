"""Train, validation-select, freeze, and evaluate M3 pointwise/LambdaMART rankers."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# LightGBM/scikit-learn and PyTorch can load competing OpenMP runtimes on macOS. Apply the same
# pre-import guard as retrieval and cross-encoder entry points (ADR-007).
if sys.platform == "darwin":
    for _var in (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from adaptirank.common.config import load_config
from adaptirank.common.ordering import assign_deterministic_rank
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.reproducibility import seed_everything
from adaptirank.common.run import ExperimentRun
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.config import LearnedRankingRunConfig
from adaptirank.ranking.evaluate import evaluate_ranking, ranked_candidates
from adaptirank.ranking.features import FEATURE_COLUMNS
from adaptirank.ranking.models import judged, predict, train_lambdamart, train_pointwise
from adaptirank.retrieval.evaluate import write_json


def _latencies(model: Any, frame: pl.DataFrame, limit: int) -> pl.DataFrame:
    records = []
    for key in frame.get_column("query_key").unique(maintain_order=True).head(limit):
        block = frame.filter(pl.col("query_key") == key)
        started = time.perf_counter()
        predict(model, block)
        records.append({"query_key": key, "latency_ms": (time.perf_counter() - started) * 1000})
    return pl.DataFrame(records)


def _evaluate(
    frame: pl.DataFrame,
    *,
    score_column: str,
    method: str,
    model: Any | None,
    latency_limit: int,
    queries: pl.DataFrame,
    relevance: pl.DataFrame,
    catalog: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    ranked = ranked_candidates(frame, score_column, f"{method}_rank")
    latency = _latencies(model, frame, latency_limit) if model is not None else None
    _, per_query, metrics = evaluate_ranking(
        ranked,
        method=method,
        queries=queries,
        relevance=relevance,
        catalog=catalog,
        latencies=latency,
    )
    return metrics, per_query, ranked


def _selection_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    validation = metrics["by_split"]["validation"]
    return validation["ndcg_10"], validation["mrr"], validation["map"]


def _score_split(
    source: Path,
    destination: Path,
    *,
    pointwise: Any,
    lambdamart: Any,
    batch_size: int,
) -> dict[str, Any]:
    parquet = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    rows = 0
    started = time.perf_counter()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            frame = pl.from_arrow(table)
            assert isinstance(frame, pl.DataFrame)
            pointwise_scores = predict(pointwise, frame)
            lambda_scores = predict(lambdamart, frame)
            heuristic_scores = (
                0.80 * frame.get_column("hybrid_score").to_numpy()
                + 0.15 * frame.get_column("lexical_overlap").to_numpy()
                + 0.05 * frame.get_column("brand_match").to_numpy()
            ).astype(np.float32)
            scored = table.append_column("heuristic_score", pa.array(heuristic_scores))
            scored = scored.append_column("pointwise_score", pa.array(pointwise_scores))
            scored = scored.append_column("lambdamart_score", pa.array(lambda_scores))
            if writer is None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(destination, scored.schema, compression="zstd")
            writer.write_table(scored, row_group_size=batch_size)
            rows += len(batch)
    finally:
        if writer is not None:
            writer.close()
    elapsed = time.perf_counter() - started
    return {
        "rows": rows,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed else 0.0,
        "sha256": sha256_file(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, LearnedRankingRunConfig)
    seed_everything(config.run.seed)
    root = project_root()
    feature_dir = resolve_project_path(config.feature_dir, root)
    out = (
        resolve_project_path(config.output_dir, root)
        / config.dataset_fingerprint
        / config.artifact_name
        / "learned"
    )
    out.mkdir(parents=True, exist_ok=True)
    queries_all = pl.read_parquet(config.dataset_dir / "queries.parquet")
    relevance = pl.read_parquet(config.dataset_dir / "relevance.parquet")
    catalog = pl.read_parquet(config.dataset_dir / "catalog.parquet")

    # The official test feature file is deliberately not opened until both selections are frozen.
    train = judged(pl.read_parquet(feature_dir / "train.parquet"))
    validation_full = pl.read_parquet(feature_dir / "validation.parquet")
    validation = judged(validation_full)
    validation_queries = queries_all.filter(pl.col("benchmark_split") == "validation")

    with ExperimentRun(
        experiment=config.run.experiment,
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        pointwise_trials = []
        pointwise_models = []
        for index, params in enumerate(config.ranking.pointwise_grid):
            model, training_seconds = train_pointwise(train, params.model_dump(), config.run.seed)
            scored = validation_full.with_columns(
                pl.Series("pointwise_score", predict(model, validation_full))
            )
            metrics, _, _ = _evaluate(
                scored,
                score_column="pointwise_score",
                method="pointwise",
                model=None,
                latency_limit=config.ranking.latency_query_sample,
                queries=validation_queries,
                relevance=relevance,
                catalog=catalog,
            )
            pointwise_models.append(model)
            pointwise_trials.append(
                {
                    "trial": index,
                    "params": params.model_dump(),
                    "training_seconds": training_seconds,
                    "validation": metrics["by_split"]["validation"],
                }
            )
        pointwise_index = max(
            range(len(pointwise_trials)),
            key=lambda index: _selection_score(
                {"by_split": {"validation": pointwise_trials[index]["validation"]}}
            ),
        )
        pointwise = pointwise_models[pointwise_index]

        lambda_trials = []
        lambda_models = []
        for index, lambda_params in enumerate(config.ranking.lambdamart_grid):
            model, training_seconds = train_lambdamart(
                train,
                validation,
                lambda_params.model_dump(),
                seed=config.run.seed,
                early_stopping_rounds=config.ranking.early_stopping_rounds,
            )
            scored = validation_full.with_columns(
                pl.Series("lambdamart_score", predict(model, validation_full))
            )
            metrics, _, _ = _evaluate(
                scored,
                score_column="lambdamart_score",
                method="lambdamart",
                model=None,
                latency_limit=config.ranking.latency_query_sample,
                queries=validation_queries,
                relevance=relevance,
                catalog=catalog,
            )
            lambda_models.append(model)
            lambda_trials.append(
                {
                    "trial": index,
                    "params": lambda_params.model_dump(),
                    "training_seconds": training_seconds,
                    "best_iteration": int(model.best_iteration_),
                    "validation": metrics["by_split"]["validation"],
                }
            )
        lambda_index = max(
            range(len(lambda_trials)),
            key=lambda index: _selection_score(
                {"by_split": {"validation": lambda_trials[index]["validation"]}}
            ),
        )
        lambdamart = lambda_models[lambda_index]

        selection = {
            "selection_split": "validation",
            "objective": ["ndcg_10", "mrr", "map"],
            "training_rows": train.height,
            "validation_judged_rows": validation.height,
            "unjudged_training_policy": "excluded from fitting; retained for inference",
            "pointwise": {"selected_trial": pointwise_index, "trials": pointwise_trials},
            "lambdamart": {"selected_trial": lambda_index, "trials": lambda_trials},
            "test_opened_after_freeze": True,
        }
        selection_path = out / "selection.json"
        write_json(selection_path, selection)
        with (out / "pointwise.pkl").open("wb") as handle:
            pickle.dump(pointwise, handle)
        lambdamart.booster_.save_model(str(out / "lambdamart.txt"))

        # Selection is now persisted and frozen; official test data may be opened for final eval.
        split_stats = {}
        rankings = {}
        for split in ("train", "validation", "test"):
            raw_path = out / f"predictions_{split}.parquet"
            split_stats[split] = _score_split(
                feature_dir / f"{split}.parquet",
                raw_path,
                pointwise=pointwise,
                lambdamart=lambdamart,
                batch_size=config.ranking.prediction_batch_size,
            )
            frame = pl.read_parquet(raw_path)
            ranked = frame
            for score_col, rank_col in (
                ("heuristic_score", "heuristic_rank"),
                ("pointwise_score", "pointwise_rank"),
                ("lambdamart_score", "lambdamart_rank"),
            ):
                ranked = assign_deterministic_rank(ranked, score_col=score_col, rank_col=rank_col)
            ranking_path = out / f"rankings_{split}.parquet"
            ranked.write_parquet(ranking_path, compression="zstd")
            rankings[split] = ranked
            split_stats[split]["rankings_sha256"] = sha256_file(ranking_path)
            run.record_artifact(f"rankings_{split}", ranking_path)

        test_queries = queries_all.filter(pl.col("benchmark_split") == "test")
        comparison: dict[str, Any] = {}
        for method in ("heuristic", "pointwise", "lambdamart"):
            validation_metrics, validation_per_query, _ = _evaluate(
                rankings["validation"],
                score_column=f"{method}_score",
                method=method,
                model=pointwise
                if method == "pointwise"
                else lambdamart
                if method == "lambdamart"
                else None,
                latency_limit=config.ranking.latency_query_sample,
                queries=validation_queries,
                relevance=relevance,
                catalog=catalog,
            )
            test_metrics, test_per_query, final_ranked = _evaluate(
                rankings["test"],
                score_column=f"{method}_score",
                method=method,
                model=pointwise
                if method == "pointwise"
                else lambdamart
                if method == "lambdamart"
                else None,
                latency_limit=config.ranking.latency_query_sample,
                queries=test_queries,
                relevance=relevance,
                catalog=catalog,
            )
            method_dir = out / method
            method_dir.mkdir(exist_ok=True)
            pl.concat([validation_per_query, test_per_query]).write_parquet(
                method_dir / "per_query_metrics.parquet"
            )
            final_ranked.write_parquet(method_dir / "test_ranking.parquet")
            comparison[method] = {
                "validation": validation_metrics["by_split"]["validation"],
                "test": test_metrics["by_split"]["test"],
                "latency": test_metrics["latency"],
            }
        feature_importance = {
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, lambdamart.feature_importances_, strict=True)
        }
        report = {
            "dataset_fingerprint": config.dataset_fingerprint,
            "feature_schema_sha256": sha256_file(feature_dir / "feature_schema.json"),
            "selection": selection,
            "split_scoring": split_stats,
            "comparison": comparison,
            "lambdamart_feature_importance": feature_importance,
        }
        report_path = out / "report.json"
        write_json(report_path, report)
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("selection", selection_path)
        run.record_artifact("pointwise_model", out / "pointwise.pkl")
        run.record_artifact("lambdamart_model", out / "lambdamart.txt")
        run.record_artifact("report", report_path)
        run.record_metrics(report)
        print(run.run_dir)


if __name__ == "__main__":
    main()
