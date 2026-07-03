"""Tracked M2 BM25, dense, and hybrid retrieval pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.reproducibility import seed_everything
from adaptirank.common.run import ExperimentRun
from adaptirank.retrieval.base import IndexBuildStats, RetrievalResult
from adaptirank.retrieval.bm25 import BM25Retriever
from adaptirank.retrieval.config import RetrievalConfig
from adaptirank.retrieval.data import load_catalog, load_queries, load_relevance, validate_dataset
from adaptirank.retrieval.dense import DenseRetriever
from adaptirank.retrieval.evaluate import evaluate_result, failure_cases, write_json
from adaptirank.retrieval.hybrid import (
    candidate_contract,
    hybrid_latencies,
    reciprocal_rank_fusion,
    select_validation_alpha,
    weighted_fusion,
)

Method = Literal["bm25", "dense", "hybrid"]


def load_retrieval_config(path: Path) -> RetrievalConfig:
    return load_config(path, RetrievalConfig)


def _root(config: RetrievalConfig) -> Path:
    return (
        resolve_project_path(config.output_dir, project_root())
        / config.dataset_fingerprint
        / config.artifact_name
    )


def _write_method(
    output_dir: Path,
    result: RetrievalResult,
    *,
    queries: pl.DataFrame,
    relevance: pl.DataFrame,
    catalog: pl.DataFrame,
    top_k: tuple[int, ...],
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated, per_query, metrics = evaluate_result(
        result,
        queries=queries,
        relevance=relevance,
        catalog=catalog,
        top_k_values=top_k,
    )
    result.candidates.write_parquet(output_dir / "raw_candidates.parquet")
    result.query_latencies_ms.write_parquet(output_dir / "query_latencies.parquet")
    annotated.write_parquet(output_dir / "candidates.parquet")
    per_query.write_parquet(output_dir / "per_query_metrics.parquet")
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "failure_cases.json",
        failure_cases(per_query, annotated, relevance),
    )
    return metrics, annotated, per_query


def _selection_score(metrics: dict[str, Any]) -> tuple[float, float]:
    validation = metrics["by_split"].get("validation", {})
    return float(validation.get("recall_primary_100", 0.0)), float(validation.get("ndcg_10", 0.0))


def run_bm25(config: RetrievalConfig) -> Path:
    report = validate_dataset(config)
    seed_everything(config.run.seed)
    root = _root(config)
    queries = load_queries(config)
    catalog = load_catalog(config)
    relevance = load_relevance(config)
    max_k = max(config.top_k)
    with ExperimentRun(
        experiment=f"{config.run.experiment}_bm25",
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        field_metrics: dict[str, dict[str, Any]] = {}
        candidate_paths: dict[str, Path] = {}
        latency_paths: dict[str, Path] = {}
        shared_index_dir = root / "bm25" / "shared_index"
        for field_set in config.bm25.field_sets:
            retriever = BM25Retriever(
                fields=field_set.fields,
                writer_heap_bytes=config.bm25.writer_heap_bytes,
            )
            retriever.build(config.dataset_dir / "catalog.parquet", shared_index_dir)
            result = retriever.retrieve(queries, max_k)
            method_dir = root / "bm25" / field_set.name
            metrics, _, _ = _write_method(
                method_dir,
                result,
                queries=queries,
                relevance=relevance,
                catalog=catalog,
                top_k=config.top_k,
            )
            field_metrics[field_set.name] = metrics
            candidate_paths[field_set.name] = method_dir / "raw_candidates.parquet"
            latency_paths[field_set.name] = method_dir / "query_latencies.parquet"
        selected = max(
            field_metrics, key=lambda name: (*_selection_score(field_metrics[name]), name)
        )
        selection = {
            "selected_field_set": selected,
            "selection_split": "validation",
            "objective": ["recall_primary_100", "ndcg_10"],
            "field_sets": {item.name: list(item.fields) for item in config.bm25.field_sets},
        }
        write_json(root / "bm25" / "selection.json", selection)
        pl.read_parquet(candidate_paths[selected]).write_parquet(
            root / "bm25" / "selected_candidates.parquet"
        )
        pl.read_parquet(latency_paths[selected]).write_parquet(
            root / "bm25" / "selected_query_latencies.parquet"
        )
        write_json(root / "bm25" / "comparison.json", field_metrics)
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("bm25_root", root / "bm25")
        run.record_artifact("selected_candidates", root / "bm25" / "selected_candidates.parquet")
        run.record_metrics(
            {
                "dataset_scientific_eligibility": report.get("scientific_eligibility", False),
                "selection": selection,
                "methods": field_metrics,
            }
        )
        return run.run_dir


def _dense_retriever(config: RetrievalConfig) -> DenseRetriever:
    dense = config.dense
    return DenseRetriever(
        model_name=dense.model_name,
        model_revision=dense.model_revision,
        fields=dense.fields,
        batch_size=dense.batch_size,
        outer_batch_size=dense.outer_batch_size,
        device=dense.device,
        nlist=dense.nlist,
        nprobe=dense.nprobe,
        training_sample_size=dense.training_sample_size,
    )


def run_dense(config: RetrievalConfig) -> Path:
    report = validate_dataset(config)
    seed_everything(config.run.seed)
    root = _root(config)
    queries = load_queries(config)
    catalog = load_catalog(config)
    relevance = load_relevance(config)
    with ExperimentRun(
        experiment=f"{config.run.experiment}_dense",
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        retriever = _dense_retriever(config)
        index_dir = root / "dense" / "index"
        retriever.build(config.dataset_dir / "catalog.parquet", index_dir)
        result = retriever.retrieve(queries, max(config.top_k))
        metrics, _, _ = _write_method(
            root / "dense",
            result,
            queries=queries,
            relevance=relevance,
            catalog=catalog,
            top_k=config.top_k,
        )
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("dense_root", root / "dense")
        run.record_artifact("embeddings", index_dir / "product_embeddings.npy")
        run.record_artifact("faiss_index", index_dir / "faiss.index")
        run.record_artifact("candidates", root / "dense" / "candidates.parquet")
        run.record_metrics(
            {
                "dataset_scientific_eligibility": report.get("scientific_eligibility", False),
                "dense": metrics,
            }
        )
        return run.run_dir


def _dummy_stats(catalog_count: int, method: str) -> IndexBuildStats:
    return IndexBuildStats(0.0, 0, catalog_count, {}, {"fusion": method})


def _result(
    method: str,
    candidates: pl.DataFrame,
    latencies: pl.DataFrame,
    catalog_count: int,
) -> RetrievalResult:
    return RetrievalResult(method, candidates, latencies, _dummy_stats(catalog_count, method))


def _report_markdown(comparison: dict[str, Any], selected_alpha: float) -> str:
    lines = [
        "# Retrieval Comparison",
        "",
        f"Selected weighted-fusion alpha on validation: `{selected_alpha}`",
        "",
        "| Method | Test Recall@100 E+S | Test NDCG@10 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in comparison.items():
        test = metrics["by_split"].get("test", {})
        latency = metrics["latency"]
        lines.append(
            f"| {name} | {test.get('recall_primary_100', 0):.6f} | "
            f"{test.get('ndcg_10', 0):.6f} | {latency['p50_ms']:.3f} | "
            f"{latency['p95_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Unjudged products remain unknown. MRR/NDCG use condensed judged lists; recall uses",
            "known relevant judgments and does not relabel missing judgments as irrelevant.",
            "",
        ]
    )
    return "\n".join(lines)


def run_hybrid(config: RetrievalConfig) -> Path:
    report = validate_dataset(config)
    seed_everything(config.run.seed)
    root = _root(config)
    queries = load_queries(config)
    catalog = load_catalog(config)
    relevance = load_relevance(config)
    bm25 = pl.read_parquet(root / "bm25" / "selected_candidates.parquet")
    dense = pl.read_parquet(root / "dense" / "raw_candidates.parquet")
    bm25_latencies = pl.read_parquet(root / "bm25" / "selected_query_latencies.parquet")
    dense_latencies = pl.read_parquet(root / "dense" / "query_latencies.parquet")
    max_k = max(config.top_k)
    validation_keys = queries.filter(pl.col("benchmark_split") == "validation").get_column(
        "query_key"
    )
    with ExperimentRun(
        experiment=f"{config.run.experiment}_hybrid",
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        alpha_metrics: dict[float, dict[str, Any]] = {}
        for alpha in config.hybrid.alpha_grid:
            candidates, seconds = weighted_fusion(
                bm25.filter(pl.col("query_key").is_in(validation_keys)),
                dense.filter(pl.col("query_key").is_in(validation_keys)),
                alpha=alpha,
                top_k=max_k,
            )
            latencies = hybrid_latencies(
                bm25_latencies.filter(pl.col("query_key").is_in(validation_keys)),
                dense_latencies.filter(pl.col("query_key").is_in(validation_keys)),
                fusion_seconds=seconds,
            )
            _, _, metrics = evaluate_result(
                _result(f"weighted_alpha_{alpha}", candidates, latencies, catalog.height),
                queries=queries.filter(pl.col("query_key").is_in(validation_keys)),
                relevance=relevance,
                catalog=catalog,
                top_k_values=config.top_k,
            )
            alpha_metrics[alpha] = metrics["by_split"]["validation"]
        selected_alpha = select_validation_alpha(alpha_metrics)
        weighted_candidates, weighted_seconds = weighted_fusion(
            bm25, dense, alpha=selected_alpha, top_k=max_k
        )
        weighted_latencies = hybrid_latencies(
            bm25_latencies, dense_latencies, fusion_seconds=weighted_seconds
        )
        weighted_metrics, weighted_annotated, _ = _write_method(
            root / "hybrid" / "weighted",
            _result("weighted_hybrid", weighted_candidates, weighted_latencies, catalog.height),
            queries=queries,
            relevance=relevance,
            catalog=catalog,
            top_k=config.top_k,
        )
        rrf_candidates, rrf_seconds = reciprocal_rank_fusion(
            bm25, dense, rrf_k=config.hybrid.rrf_k, top_k=max_k
        )
        rrf_latencies = hybrid_latencies(
            bm25_latencies, dense_latencies, fusion_seconds=rrf_seconds
        )
        rrf_metrics, rrf_annotated, _ = _write_method(
            root / "hybrid" / "rrf",
            _result("rrf", rrf_candidates, rrf_latencies, catalog.height),
            queries=queries,
            relevance=relevance,
            catalog=catalog,
            top_k=config.top_k,
        )
        contract = candidate_contract(bm25, dense, weighted_candidates, rrf_candidates, relevance)
        contract_path = root / "hybrid" / "candidate_contract.parquet"
        contract.write_parquet(contract_path)
        bm25_selection: dict[str, Any] = json.loads((root / "bm25" / "selection.json").read_text())
        selected_bm25 = bm25_selection["selected_field_set"]
        bm25_metrics: dict[str, Any] = json.loads(
            (root / "bm25" / selected_bm25 / "metrics.json").read_text()
        )
        dense_metrics: dict[str, Any] = json.loads((root / "dense" / "metrics.json").read_text())
        comparison = {
            "bm25": bm25_metrics,
            "dense": dense_metrics,
            "weighted_hybrid": weighted_metrics,
            "rrf": rrf_metrics,
        }
        hybrid_root = root / "hybrid"
        tuning = {
            "selection_split": "validation",
            "objective": ["recall_primary_100", "ndcg_10"],
            "selected_alpha": selected_alpha,
            "alpha_metrics": {str(key): value for key, value in alpha_metrics.items()},
            "rrf_k": config.hybrid.rrf_k,
        }
        write_json(hybrid_root / "tuning.json", tuning)
        write_json(hybrid_root / "comparison.json", comparison)
        (hybrid_root / "comparison.md").write_text(_report_markdown(comparison, selected_alpha))
        write_json(
            hybrid_root / "failure_analysis.json",
            {
                "weighted_hybrid": failure_cases(
                    pl.read_parquet(hybrid_root / "weighted" / "per_query_metrics.parquet"),
                    weighted_annotated,
                    relevance,
                ),
                "rrf": failure_cases(
                    pl.read_parquet(hybrid_root / "rrf" / "per_query_metrics.parquet"),
                    rrf_annotated,
                    relevance,
                ),
            },
        )
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("candidate_contract", contract_path)
        run.record_artifact("comparison", hybrid_root / "comparison.json")
        run.record_artifact("failure_analysis", hybrid_root / "failure_analysis.json")
        run.record_metrics(
            {
                "dataset_scientific_eligibility": report.get("scientific_eligibility", False),
                "tuning": tuning,
                "comparison": comparison,
            }
        )
        return run.run_dir


def run_method(config: RetrievalConfig, method: Method) -> Path:
    if method == "bm25":
        return run_bm25(config)
    if method == "dense":
        return run_dense(config)
    return run_hybrid(config)
