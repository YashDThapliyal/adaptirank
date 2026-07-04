"""Pretrained cross-encoder reranking scorer with deterministic, resumable scoring.

The cross-encoder scores ``(query_text, product_text)`` pairs for a bounded top-M candidate
set per query. It is a pretrained MS MARCO baseline (not fine-tuned, not e-commerce-native);
its role is a strong reranking signal for the M3 cascade, evaluated separately from the
label-free LambdaMART model.
"""

from __future__ import annotations

import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Fields concatenated (in this fixed order) to build the product side of each pair.
PairField = tuple[str, ...]


def select_device(requested: str) -> str:
    """Resolve CUDA, Apple MPS, or CPU without requiring an accelerator."""

    if requested != "auto":
        return requested
    torch: Any = import_module("torch")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_product_text(row: dict[str, Any], fields: PairField) -> str:
    """Deterministically compose the product side of a pair from configured fields.

    Empty/null fields are skipped; order follows ``fields`` exactly so the same product
    always yields the same text (a prerequisite for reproducible and cacheable scores).
    """

    parts = [str(row[field]).strip() for field in fields if row.get(field) not in (None, "")]
    return " ".join(part for part in parts if part)


class CrossEncoderScorer:
    """Batched, deterministic pretrained cross-encoder over ``(query, product)`` pairs."""

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = select_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model: Any = None

    def load(self) -> Any:
        if self.model is None:
            sentence_transformers: Any = import_module("sentence_transformers")
            self.model = sentence_transformers.CrossEncoder(
                self.model_name,
                revision=self.model_revision,
                device=self.device,
                max_length=self.max_length,
            )
        return self.model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score pairs in fixed input order; identical input yields identical output."""

        if not pairs:
            return np.zeros(0, dtype=np.float32)
        model = self.load()
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def _target_pairs(candidates: pl.DataFrame, rank_column: str, top_m: int) -> pl.DataFrame:
    """Select the top-M candidates per query by the chosen first-stage rank column."""

    if rank_column not in candidates.columns:
        raise KeyError(f"rank column {rank_column!r} not in candidates")
    return (
        candidates.filter(pl.col(rank_column).is_not_null() & (pl.col(rank_column) <= top_m))
        .select("query_key", "product_key", "split", rank_column)
        .unique(["query_key", "product_key"])
        .sort("query_key", rank_column)
    )


def score_top_m(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    catalog: pl.DataFrame,
    scorer: CrossEncoderScorer,
    *,
    fields: PairField,
    top_m: int,
    rank_column: str = "hybrid_rank",
    checkpoint_path: Path | None = None,
    block_queries: int = 512,
) -> pl.DataFrame:
    """Score the top-M candidates per query, resuming from a checkpoint when present.

    Returns one row per scored pair: ``query_key, product_key, split, cross_encoder_score``.
    Scoring proceeds in query blocks; after each block the checkpoint is rewritten atomically,
    so an interrupted run resumes without rescoring completed pairs.
    """

    targets = _target_pairs(candidates, rank_column, top_m)
    return score_pair_frame(
        targets,
        queries,
        catalog,
        scorer,
        fields=fields,
        checkpoint_path=checkpoint_path,
        block_queries=block_queries,
    )


def score_pair_frame(
    targets: pl.DataFrame,
    queries: pl.DataFrame,
    catalog: pl.DataFrame,
    scorer: CrossEncoderScorer,
    *,
    fields: PairField,
    checkpoint_path: Path | None = None,
    block_queries: int = 512,
) -> pl.DataFrame:
    """Score an explicit deduplicated pair frame with the checkpoint/resume contract."""

    required = {"query_key", "product_key", "split"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"explicit pair frame is missing columns: {missing}")
    targets = (
        targets.select(*sorted(required))
        .unique(["query_key", "product_key"])
        .sort("query_key", "product_key")
    )
    query_text = dict(
        zip(
            queries.get_column("query_key").to_list(),
            queries.get_column("query_text").to_list(),
            strict=True,
        )
    )
    catalog_rows = {
        str(row["product_key"]): row
        for row in catalog.select("product_key", *fields).iter_rows(named=True)
    }

    if checkpoint_path is not None and checkpoint_path.is_file():
        done = pl.read_parquet(checkpoint_path)
        if done.height == targets.height:
            missing_pairs = targets.join(
                done.select("query_key", "product_key"),
                on=["query_key", "product_key"],
                how="anti",
            )
            if missing_pairs.is_empty():
                return done.sort("query_key", "cross_encoder_score")

    ordered_queries = targets.get_column("query_key").unique(maintain_order=True).to_list()
    part_dir = (
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".parts")
        if checkpoint_path is not None
        else None
    )
    if part_dir is not None:
        part_dir.mkdir(parents=True, exist_ok=True)
    scored_frames: list[pl.DataFrame] = []
    for start in range(0, len(ordered_queries), block_queries):
        part_path = part_dir / f"part-{start:08d}.parquet" if part_dir is not None else None
        if part_path is not None and part_path.is_file():
            scored_frames.append(pl.read_parquet(part_path))
            continue
        block_keys = ordered_queries[start : start + block_queries]
        block = targets.filter(pl.col("query_key").is_in(block_keys))
        pairs: list[tuple[str, str]] = []
        rows: list[dict[str, Any]] = []
        for item in block.iter_rows(named=True):
            product = catalog_rows.get(str(item["product_key"]))
            if product is None:
                continue
            q = str(query_text.get(item["query_key"], ""))
            pairs.append((q, build_product_text(product, fields)))
            rows.append(
                {
                    "query_key": item["query_key"],
                    "product_key": item["product_key"],
                    "split": item["split"],
                }
            )
        if not rows:
            continue
        scores = scorer.score_pairs(pairs)
        block_scored = pl.DataFrame(rows).with_columns(
            pl.Series("cross_encoder_score", scores, dtype=pl.Float32)
        )
        scored_frames.append(block_scored)
        if part_path is not None:
            tmp = part_path.with_suffix(part_path.suffix + ".tmp")
            block_scored.write_parquet(tmp)
            tmp.replace(part_path)

    if not scored_frames:
        return pl.DataFrame(
            schema={
                "query_key": pl.String,
                "product_key": pl.String,
                "split": pl.String,
                "cross_encoder_score": pl.Float32,
            }
        )
    merged = pl.concat(scored_frames, how="vertical").sort("query_key", "cross_encoder_score")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        merged.write_parquet(tmp)
        tmp.replace(checkpoint_path)
    return merged


def scoring_stats(started: float, pair_count: int, scorer: CrossEncoderScorer) -> dict[str, Any]:
    """Latency/throughput record for a scoring pass (hardware-specific; label accordingly)."""

    elapsed = time.perf_counter() - started
    return {
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
        "device": scorer.device,
        "batch_size": scorer.batch_size,
        "max_length": scorer.max_length,
        "pairs_scored": pair_count,
        "elapsed_seconds": elapsed,
        "pairs_per_second": pair_count / elapsed if elapsed > 0 else 0.0,
    }
