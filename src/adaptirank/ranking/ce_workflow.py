"""Canonical M3 cross-encoder A100 workflow helpers.

This module keeps the full-run Colab notebook thin: filesystem integrity, pair-union
validation, GPU gates, benchmarking, block manifests, and final score checks live here and
are covered by local tests. Heavy model downloads only happen when callers instantiate or
probe the real ``CrossEncoderScorer``.
"""

from __future__ import annotations

import json
import math
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.ordering import verify_pair_uniqueness
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.config import CrossEncoderRunConfig
from adaptirank.ranking.cross_encoder import (
    CrossEncoderScorer,
    PairField,
    build_product_text,
    score_pair_frame,
)

CANONICAL_GIT_COMMIT = "4f327ff86c5a50b11e850620e8b2f8d74311721c"
CANONICAL_DATASET_FINGERPRINT = "dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667"
EXPECTED_CE_UNION_ROWS = 3_156_056
EXPECTED_CE_UNION_SHA256 = "16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d"
EXPECTED_ARCHIVE_SHA256 = "a79bb8ad98b2cdbfb56b6f6680c95ce87ef1dd792a16ac91d95fec563ee67f5f"

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/ranking/cross_encoder_union_m3.yaml"
_PINNED_CONFIG = load_config(_CONFIG_PATH, CrossEncoderRunConfig)
PINNED_CE_MODEL = _PINNED_CONFIG.cross_encoder.model_name
PINNED_CE_REVISION = _PINNED_CONFIG.cross_encoder.model_revision


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary sibling file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_file_sha256(path: Path, expected_sha256: str) -> str:
    """Verify a file SHA-256 and return the observed digest."""

    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path}: observed {observed}, expected {expected_sha256}"
        )
    return observed


def inspect_gpu() -> dict[str, Any]:
    """Return CUDA GPU facts without requiring CUDA to be available."""

    torch: Any = import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                "device_index": device_index,
                "device_name": str(torch.cuda.get_device_name(device_index)),
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1_000_000_000,
            }
        )
    return info


def require_cuda_gpu(*, warn_non_a100: bool = True) -> dict[str, Any]:
    """Require a CUDA runtime for the full CE scorer; warn when the GPU is not A100."""

    info = inspect_gpu()
    if not info["cuda_available"]:
        raise RuntimeError("CUDA GPU is required for the canonical M3 CE full run")
    name = str(info.get("device_name", ""))
    if warn_non_a100 and "A100" not in name.upper():
        print(f"WARNING: canonical run expects A100; observed GPU is {name}")
    return info


def _missing_columns(frame: pl.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(frame.columns))


def verify_ce_union_frame(
    frame: pl.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_CE_UNION_ROWS,
) -> dict[str, Any]:
    """Validate the Hybrid-top-100/LambdaMART-top-50 CE pair union."""

    required = {"query_key", "product_key", "split", "in_hybrid_top_100", "in_lambdamart_top_50"}
    missing = _missing_columns(frame, required)
    if missing:
        raise ValueError(f"CE union is missing columns: {missing}")
    verify_pair_uniqueness(frame)
    if expected_rows is not None and frame.height != expected_rows:
        raise ValueError(f"CE union row count {frame.height} != expected {expected_rows}")
    nulls = frame.select(
        pl.col("query_key").is_null().sum().alias("query_key_nulls"),
        pl.col("product_key").is_null().sum().alias("product_key_nulls"),
        pl.col("split").is_null().sum().alias("split_nulls"),
    ).to_dicts()[0]
    if any(int(value) for value in nulls.values()):
        raise ValueError(f"CE union contains null required keys: {nulls}")
    neither_source = frame.filter(
        ~pl.col("in_hybrid_top_100") & ~pl.col("in_lambdamart_top_50")
    ).height
    if neither_source:
        raise ValueError(f"CE union contains {neither_source} rows from neither source")
    stats = {
        "rows": frame.height,
        "queries": frame.get_column("query_key").n_unique(),
        "unique_pairs": frame.select("query_key", "product_key").n_unique(),
        "hybrid_pairs": int(frame.get_column("in_hybrid_top_100").sum()),
        "lambdamart_pairs": int(frame.get_column("in_lambdamart_top_50").sum()),
        "overlap_pairs": frame.filter(
            pl.col("in_hybrid_top_100") & pl.col("in_lambdamart_top_50")
        ).height,
        "by_split": frame.group_by("split")
        .agg(pl.len().alias("pairs"), pl.col("query_key").n_unique().alias("queries"))
        .sort("split")
        .to_dicts(),
    }
    return stats


def verify_ce_union_files(
    union_path: Path,
    *,
    manifest_path: Path | None = None,
    expected_rows: int | None = EXPECTED_CE_UNION_ROWS,
    expected_sha256: str = EXPECTED_CE_UNION_SHA256,
) -> dict[str, Any]:
    """Verify the persisted union parquet plus optional manifest."""

    observed_sha = verify_file_sha256(union_path, expected_sha256)
    frame = pl.read_parquet(union_path)
    stats = verify_ce_union_frame(frame, expected_rows=expected_rows)
    stats["sha256"] = observed_sha
    stats["path"] = str(union_path.resolve())
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("union_pairs", -1)) != frame.height:
            raise ValueError("CE union manifest union_pairs does not match parquet row count")
        if manifest.get("pair_union_sha256") != observed_sha:
            raise ValueError("CE union manifest pair_union_sha256 does not match parquet")
        stats["manifest_path"] = str(manifest_path.resolve())
    return stats


def _pair_rows(
    targets: pl.DataFrame,
    queries: pl.DataFrame,
    catalog: pl.DataFrame,
    *,
    fields: PairField,
) -> list[tuple[str, str]]:
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
    pairs: list[tuple[str, str]] = []
    for row in targets.iter_rows(named=True):
        product = catalog_rows.get(str(row["product_key"]))
        if product is None:
            continue
        pairs.append(
            (str(query_text.get(row["query_key"], "")), build_product_text(product, fields))
        )
    return pairs


def probe_cross_encoder(
    scorer: CrossEncoderScorer,
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Score a tiny probe set and require finite outputs."""

    probe_pairs = pairs or [
        ("waterproof trail running shoes", "Waterproof trail running shoe with rubber outsole"),
        ("waterproof trail running shoes", "Ceramic coffee mug"),
    ]
    scores = scorer.score_pairs(probe_pairs)
    invalid = int(np.isnan(scores).sum() + np.isinf(scores).sum())
    if invalid:
        raise ValueError(f"cross-encoder probe produced {invalid} invalid scores")
    return {
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
        "device": scorer.device,
        "batch_size": scorer.batch_size,
        "max_length": scorer.max_length,
        "probe_pairs": len(probe_pairs),
        "scores": [float(score) for score in scores],
    }


def deterministic_benchmark_subset(
    pair_frame: pl.DataFrame,
    *,
    n_pairs: int = 4096,
) -> pl.DataFrame:
    """Pick a stable benchmark subset from the canonical pair ordering."""

    required = {"query_key", "product_key", "split"}
    missing = _missing_columns(pair_frame, required)
    if missing:
        raise ValueError(f"benchmark pair frame is missing columns: {missing}")
    return (
        pair_frame.select("query_key", "product_key", "split")
        .sort("split", "query_key", "product_key")
        .head(n_pairs)
    )


def benchmark_batch_sizes(
    subset: pl.DataFrame,
    queries: pl.DataFrame,
    catalog: pl.DataFrame,
    *,
    model_name: str = PINNED_CE_MODEL,
    model_revision: str = PINNED_CE_REVISION,
    fields: PairField = ("title", "description", "brand"),
    batch_sizes: tuple[int, ...] = (128, 256, 384),
    device: str = "cuda",
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Benchmark candidate CE batch sizes on a deterministic subset."""

    pairs = _pair_rows(subset, queries, catalog, fields=fields)
    if not pairs:
        raise ValueError("benchmark subset produced no scoreable pairs")
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        scorer = CrossEncoderScorer(
            model_name=model_name,
            model_revision=model_revision,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        started = time.perf_counter()
        scores = scorer.score_pairs(pairs)
        elapsed = time.perf_counter() - started
        invalid = int(np.isnan(scores).sum() + np.isinf(scores).sum())
        if invalid:
            raise ValueError(f"batch_size={batch_size} produced {invalid} invalid scores")
        results.append(
            {
                "batch_size": batch_size,
                "pairs": len(pairs),
                "elapsed_seconds": elapsed,
                "pairs_per_second": len(pairs) / elapsed if elapsed > 0 else math.inf,
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
            }
        )
    return results


def list_completed_blocks(part_dir: Path) -> list[Path]:
    """Return completed score block parquet files in stable order."""

    if not part_dir.is_dir():
        return []
    return sorted(path for path in part_dir.glob("part-*.parquet") if path.is_file())


def write_block_manifest(
    manifest_path: Path,
    *,
    checkpoint_path: Path,
    part_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a manifest describing checkpoint and block state."""

    blocks = list_completed_blocks(
        part_dir or checkpoint_path.with_suffix(checkpoint_path.suffix + ".parts")
    )
    manifest: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
        "completed_blocks": [
            {
                "path": str(path.resolve()),
                "rows": pl.read_parquet(path, columns=["query_key"]).height,
                "sha256": sha256_file(path),
            }
            for path in blocks
        ],
        "metadata": metadata or {},
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def consolidate_part_blocks(
    part_dir: Path,
    output_path: Path,
    *,
    expected_pairs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Concatenate completed block parquet files into one canonical score parquet."""

    blocks = list_completed_blocks(part_dir)
    if not blocks:
        raise FileNotFoundError(f"no completed CE part blocks found in {part_dir}")
    scores = pl.concat([pl.read_parquet(path) for path in blocks], how="vertical")
    if expected_pairs is not None:
        verify_final_scores(expected_pairs, scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    scores.sort("query_key", "product_key").write_parquet(temporary)
    temporary.replace(output_path)
    return pl.read_parquet(output_path)


def verify_final_scores(pair_frame: pl.DataFrame, scores: pl.DataFrame) -> dict[str, Any]:
    """Verify one finite CE score for exactly every expected pair."""

    required_pairs = {"query_key", "product_key", "split"}
    missing_pairs = _missing_columns(pair_frame, required_pairs)
    if missing_pairs:
        raise ValueError(f"expected pair frame is missing columns: {missing_pairs}")
    required_scores = required_pairs | {"cross_encoder_score"}
    missing_scores = _missing_columns(scores, required_scores)
    if missing_scores:
        raise ValueError(f"score frame is missing columns: {missing_scores}")
    expected = pair_frame.select("query_key", "product_key", "split").unique(
        ["query_key", "product_key"]
    )
    scored = scores.select("query_key", "product_key", "split", "cross_encoder_score")
    verify_pair_uniqueness(expected)
    verify_pair_uniqueness(scored)
    invalid = scored.filter(
        pl.col("cross_encoder_score").is_null()
        | pl.col("cross_encoder_score").is_nan()
        | pl.col("cross_encoder_score").is_infinite()
    ).height
    if invalid:
        raise ValueError(f"score frame contains {invalid} invalid cross_encoder_score values")
    missing = expected.join(
        scored.select("query_key", "product_key"), on=["query_key", "product_key"], how="anti"
    )
    extra = scored.join(
        expected.select("query_key", "product_key"), on=["query_key", "product_key"], how="anti"
    )
    if missing.height or extra.height:
        raise ValueError(f"score pair mismatch: missing={missing.height}, extra={extra.height}")
    split_mismatch = expected.join(
        scored.select("query_key", "product_key", pl.col("split").alias("score_split")),
        on=["query_key", "product_key"],
    ).filter(pl.col("split") != pl.col("score_split"))
    if split_mismatch.height:
        raise ValueError(f"score frame contains {split_mismatch.height} split mismatches")
    score_col = scored.get_column("cross_encoder_score")
    return {
        "rows": scored.height,
        "queries": scored.get_column("query_key").n_unique(),
        "score_min": float(score_col.min()),  # type: ignore[arg-type]
        "score_max": float(score_col.max()),  # type: ignore[arg-type]
        "score_mean": float(score_col.mean()),  # type: ignore[arg-type]
    }


def score_union_with_checkpoints(
    union_frame: pl.DataFrame,
    queries: pl.DataFrame,
    catalog: pl.DataFrame,
    scorer: CrossEncoderScorer,
    *,
    fields: PairField = ("title", "description", "brand"),
    checkpoint_path: Path,
    block_queries: int = 256,
    expected_rows: int | None = EXPECTED_CE_UNION_ROWS,
    manifest_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Score the canonical CE union with validation, resume, final checks, and manifest writing."""

    verify_ce_union_frame(union_frame, expected_rows=expected_rows)
    targets = union_frame.select("query_key", "product_key", "split")
    scores = score_pair_frame(
        targets,
        queries,
        catalog,
        scorer,
        fields=fields,
        checkpoint_path=checkpoint_path,
        block_queries=block_queries,
    )
    stats = verify_final_scores(targets, scores)
    if manifest_path is not None:
        write_block_manifest(
            manifest_path,
            checkpoint_path=checkpoint_path,
            metadata={
                "model_name": scorer.model_name,
                "model_revision": scorer.model_revision,
                "device": scorer.device,
                "batch_size": scorer.batch_size,
                "max_length": scorer.max_length,
                "score_stats": stats,
                **(metadata or {}),
            },
        )
    return scores
