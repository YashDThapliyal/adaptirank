"""Retrieval dataset loading and deterministic query limiting."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import polars as pl

from adaptirank.retrieval.config import RetrievalConfig


def validate_dataset(config: RetrievalConfig) -> dict[str, Any]:
    """Validate that retrieval uses the intended scientific or smoke dataset."""

    report_path = config.dataset_dir / "dataset_report.json"
    manifest_path = config.dataset_dir / "manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("dataset report or manifest is missing")
    report: dict[str, Any] = json.loads(report_path.read_text())
    if report.get("dataset_fingerprint") != config.dataset_fingerprint:
        raise ValueError("retrieval config dataset fingerprint does not match dataset report")
    for filename in ("catalog.parquet", "queries.parquet", "relevance.parquet"):
        if not (config.dataset_dir / filename).is_file():
            raise FileNotFoundError(config.dataset_dir / filename)
    return report


def load_queries(config: RetrievalConfig) -> pl.DataFrame:
    queries = pl.read_parquet(config.dataset_dir / "queries.parquet").filter(
        pl.col("benchmark_split").is_in(config.evaluation_splits)
    )
    if config.max_queries_per_split is None:
        return queries.sort("benchmark_split", "query_key")
    limited: list[pl.DataFrame] = []
    for split in config.evaluation_splits:
        frame = queries.filter(pl.col("benchmark_split") == split).with_columns(
            pl.col("query_key")
            .map_elements(
                lambda key: hashlib.sha256(f"{config.run.seed}\0{key}".encode()).hexdigest(),
                return_dtype=pl.String,
            )
            .alias("_sample_order")
        )
        limited.append(
            frame.sort("_sample_order").head(config.max_queries_per_split).drop("_sample_order")
        )
    return pl.concat(limited).sort("benchmark_split", "query_key")


def load_catalog(config: RetrievalConfig) -> pl.DataFrame:
    return pl.read_parquet(config.dataset_dir / "catalog.parquet")


def load_relevance(config: RetrievalConfig) -> pl.DataFrame:
    return pl.read_parquet(config.dataset_dir / "relevance.parquet")
