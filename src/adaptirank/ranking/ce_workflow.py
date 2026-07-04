"""Canonical M3 cross-encoder A100 workflow helpers.

This module keeps the full-run Colab notebook thin: filesystem integrity, pair-union
validation, GPU gates, benchmarking, block manifests, and final score checks live here and
are covered by local tests. Heavy model downloads only happen when callers instantiate or
probe the real ``CrossEncoderScorer``.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tarfile
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

M3_CE_RELEASE_REF = "m3-ce-a100-v1"
ARTIFACT_BASE_GIT_COMMIT = "4f327ff86c5a50b11e850620e8b2f8d74311721c"
CANONICAL_DATASET_FINGERPRINT = "dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667"
EXPECTED_CE_UNION_ROWS = 3_156_056
EXPECTED_CE_UNION_SHA256 = "16a43b01f0ba159e5950c1fe7d4363b6c05d7b0c9ffe6c581272379ef9c8488d"
EXPECTED_ARCHIVE_SHA256 = "a79bb8ad98b2cdbfb56b6f6680c95ce87ef1dd792a16ac91d95fec563ee67f5f"

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/ranking/cross_encoder_union_m3.yaml"
_PINNED_CONFIG = load_config(_CONFIG_PATH, CrossEncoderRunConfig)
PINNED_CE_MODEL = _PINNED_CONFIG.cross_encoder.model_name
PINNED_CE_REVISION = _PINNED_CONFIG.cross_encoder.model_revision

SCORES_MINIMAL_COLUMNS: tuple[str, ...] = (
    "query_key",
    "product_key",
    "split",
    "cross_encoder_score",
)
UNION_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "in_hybrid_top_100",
    "in_lambdamart_top_50",
    "hybrid_rank",
    "lambdamart_rank",
)
SCORES_ENRICHED_COLUMNS: tuple[str, ...] = SCORES_MINIMAL_COLUMNS + UNION_MEMBERSHIP_COLUMNS

CE_CANONICAL_ARTIFACTS: tuple[str, ...] = (
    "pair_union.parquet",
    "pair_union_manifest.json",
    "scores.parquet",
    "scores_enriched.parquet",
    "scoring_stats.json",
    "benchmark.json",
    "validation_report.json",
    "score_distribution.json",
    "provenance.json",
    "runtime.json",
    "artifact_manifest.json",
)

CE_LOCAL_TRANSFER_BUNDLE = "m3_ce_local_transfer.tar.gz"
CE_FULL_BUNDLE = "m3_ce_a100_outputs.tar.gz"


def resolve_release_ref(ref: str = M3_CE_RELEASE_REF, repo_root: Path | None = None) -> str:
    """Resolve a release tag or ref to a full git commit SHA."""

    root = repo_root or Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def read_notebook_commit(repo_root: Path | None = None) -> str:
    """Return the release ref commit when available, else current git HEAD."""

    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        return resolve_release_ref(M3_CE_RELEASE_REF, root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()


def atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    """Write parquet through a temporary sibling file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


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

    required = {"query_key", "product_key", "split", *UNION_MEMBERSHIP_COLUMNS}
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


def enrich_scores(union: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    membership = union.select("query_key", "product_key", "split", *UNION_MEMBERSHIP_COLUMNS)
    minimal = scores.select(*SCORES_MINIMAL_COLUMNS)
    enriched = minimal.join(membership, on=["query_key", "product_key", "split"], how="left")
    missing = _missing_columns(enriched, set(SCORES_ENRICHED_COLUMNS))
    if missing:
        raise ValueError(f"enriched scores missing columns: {missing}")
    return enriched.select(*SCORES_ENRICHED_COLUMNS)


def _score_distribution_stats(frame: pl.DataFrame) -> dict[str, Any]:
    column = frame.get_column("cross_encoder_score")
    quantiles = column.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": frame.height,
        "min": float(column.min()),  # type: ignore[arg-type]
        "max": float(column.max()),  # type: ignore[arg-type]
        "mean": float(column.mean()),  # type: ignore[arg-type]
        "std": float(column.std()),  # type: ignore[arg-type]
        "quantiles": {
            "p01": float(quantiles[0]),  # type: ignore[arg-type]
            "p05": float(quantiles[1]),  # type: ignore[arg-type]
            "p25": float(quantiles[2]),  # type: ignore[arg-type]
            "p50": float(quantiles[3]),  # type: ignore[arg-type]
            "p75": float(quantiles[4]),  # type: ignore[arg-type]
            "p95": float(quantiles[5]),  # type: ignore[arg-type]
            "p99": float(quantiles[6]),  # type: ignore[arg-type]
        },
    }


def score_distribution_report(scores: pl.DataFrame) -> dict[str, Any]:
    missing = _missing_columns(scores, set(SCORES_MINIMAL_COLUMNS))
    if missing:
        raise ValueError(f"scores frame is missing columns: {missing}")
    report: dict[str, Any] = {"global": _score_distribution_stats(scores), "by_split": {}}
    for split in scores.get_column("split").unique().sort().to_list():
        report["by_split"][str(split)] = _score_distribution_stats(
            scores.filter(pl.col("split") == split)
        )
    return report


def extended_validation_report(
    union: pl.DataFrame,
    scores: pl.DataFrame,
    *,
    union_sha256: str,
    scores_sha256: str,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    union_stats = verify_ce_union_frame(union, expected_rows=expected_rows)
    score_stats = verify_final_scores(
        union.select("query_key", "product_key", "split"), scores.select(*SCORES_MINIMAL_COLUMNS)
    )
    enriched = enrich_scores(union, scores)
    return {
        "union": {**union_stats, "sha256": union_sha256},
        "scores": {**score_stats, "sha256": scores_sha256, "columns": list(SCORES_MINIMAL_COLUMNS)},
        "enriched": {"rows": enriched.height, "columns": list(SCORES_ENRICHED_COLUMNS)},
        "status": "PASS",
    }


def collect_model_provenance(scorer: CrossEncoderScorer, fields: PairField) -> dict[str, Any]:
    return {
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
        "device": scorer.device,
        "batch_size": scorer.batch_size,
        "max_length": scorer.max_length,
        "pair_fields": list(fields),
        "product_text_policy": (
            "concatenate configured catalog fields in order; skip empty or null values"
        ),
        "role": "pretrained MS MARCO baseline; not e-commerce fine-tuned",
    }


def build_scoring_stats_dict(
    scorer: CrossEncoderScorer,
    pair_count: int,
    elapsed_seconds: float,
    **extra: Any,
) -> dict[str, Any]:
    stats = {
        "model_name": scorer.model_name,
        "model_revision": scorer.model_revision,
        "device": scorer.device,
        "batch_size": scorer.batch_size,
        "max_length": scorer.max_length,
        "pairs_scored": pair_count,
        "elapsed_seconds": elapsed_seconds,
        "pairs_per_second": pair_count / elapsed_seconds if elapsed_seconds > 0 else 0.0,
    }
    stats.update(extra)
    return stats


def build_benchmark_dict(
    trials: list[dict[str, Any]],
    subset_fingerprint: str,
    gpu_info: dict[str, Any],
    selected_batch_size: int,
) -> dict[str, Any]:
    return {
        "subset_fingerprint": subset_fingerprint,
        "gpu": gpu_info,
        "selected_batch_size": selected_batch_size,
        "trials": trials,
    }


def build_provenance_record(
    *,
    notebook_commit: str,
    scoring_code_commit: str,
    artifact_base_commit: str,
    dataset_fingerprint: str,
    union_manifest: dict[str, Any],
    input_archive_sha256: str,
    model_provenance: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "notebook_commit": notebook_commit,
        "scoring_code_commit": scoring_code_commit,
        "artifact_base_commit": artifact_base_commit,
        "dataset_fingerprint": dataset_fingerprint,
        "union_manifest": union_manifest,
        "input_archive_sha256": input_archive_sha256,
        "model": model_provenance,
    }
    record.update(extra)
    return record


_ARTIFACT_PURPOSES = {
    "pair_union.parquet": "canonical CE-A/CE-B deduplicated pair union",
    "pair_union_manifest.json": "union row count and SHA-256 manifest",
    "scores.parquet": "minimal CE scores for cascade evaluation",
    "scores_enriched.parquet": "scores joined with union membership for CE-score ablation",
    "scoring_stats.json": "full-run scoring throughput and model config",
    "benchmark.json": "validation-subset batch-size benchmark",
    "validation_report.json": "union and score invariant validation",
    "score_distribution.json": "global and per-split score distribution",
    "provenance.json": "notebook, code, artifact-base, and model provenance",
    "runtime.json": "phase wall-clock timings",
    "artifact_manifest.json": "durable artifact index with SHA-256 and sizes",
    CE_LOCAL_TRANSFER_BUNDLE: "local import bundle for no-rerun downstream",
    CE_FULL_BUNDLE: "full Drive archive of durable CE outputs",
}


def build_artifact_manifest(drive_root: Path) -> dict[str, Any]:
    search_roots = [drive_root / "final", drive_root / "metadata", drive_root / "checkpoints"]
    artifacts = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            artifacts.append(
                {
                    "path": str(path.relative_to(drive_root)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "purpose": _ARTIFACT_PURPOSES.get(path.name, f"artifact under {root.name}/"),
                }
            )
    return {
        "drive_root": str(drive_root.resolve()),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }


def _audit_scores_parquet(path: Path) -> None:
    frame = pl.read_parquet(path)
    missing = _missing_columns(frame, set(SCORES_MINIMAL_COLUMNS))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    invalid = frame.filter(
        pl.col("cross_encoder_score").is_null()
        | pl.col("cross_encoder_score").is_nan()
        | pl.col("cross_encoder_score").is_infinite()
    ).height
    if invalid:
        raise ValueError(f"{invalid} invalid scores")


def _audit_enriched_parquet(path: Path) -> None:
    frame = pl.read_parquet(path)
    missing = _missing_columns(frame, set(SCORES_ENRICHED_COLUMNS))
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _audit_union_parquet(path: Path) -> None:
    verify_ce_union_frame(pl.read_parquet(path), expected_rows=None)


def _audit_json_object(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")


def run_completeness_audit(
    artifacts: dict[str, Path], *, status: str = "SUCCESS"
) -> dict[str, Any]:
    validators = {
        "pair_union.parquet": _audit_union_parquet,
        "pair_union_manifest.json": _audit_json_object,
        "scores.parquet": _audit_scores_parquet,
        "scores_enriched.parquet": _audit_enriched_parquet,
        "scoring_stats.json": _audit_json_object,
        "benchmark.json": _audit_json_object,
        "validation_report.json": _audit_json_object,
        "score_distribution.json": _audit_json_object,
        "provenance.json": _audit_json_object,
        "runtime.json": _audit_json_object,
        "artifact_manifest.json": _audit_json_object,
    }
    checks: list[dict[str, Any]] = []
    for name in CE_CANONICAL_ARTIFACTS:
        path = artifacts.get(name)
        if path is None or not path.is_file():
            checks.append({"artifact": name, "status": "FAIL", "error": "missing"})
            continue
        try:
            validators.get(name, lambda p: None)(path)
            checks.append(
                {
                    "artifact": name,
                    "status": "PASS",
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        except Exception as exc:
            checks.append(
                {"artifact": name, "status": "FAIL", "path": str(path.resolve()), "error": str(exc)}
            )
    audit_status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    audit = {"status": audit_status, "checks": checks}
    if status == "SUCCESS" and audit_status == "FAIL":
        failed = [item["artifact"] for item in checks if item["status"] == "FAIL"]
        raise ValueError(f"completeness audit failed for: {failed}")
    return audit


NO_RERUN_DOWNSTREAM_STATEMENT = (
    "Import m3_ce_local_transfer.tar.gz with make import-m3-ce-outputs, then run "
    "make rank-m3-ce-evaluate for Hybrid-to-CE and Hybrid-to-LambdaMART-to-CE cascades "
    "without rerunning cross-encoder scoring on A100."
)


def _downstream_check(status: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, **details}
    return payload


def run_downstream_readiness_audit(
    artifacts: dict[str, Path],
    *,
    expected_rows: int | None = EXPECTED_CE_UNION_ROWS,
) -> dict[str, Any]:
    """Verify CE artifacts are sufficient for local no-rerun cascade evaluation."""

    repo_root = Path(__file__).resolve().parents[3]
    checks: dict[str, dict[str, Any]] = {}

    def _fail(name: str, error: str) -> None:
        checks[name] = _downstream_check("FAIL", error=error)

    union_path = artifacts.get("pair_union.parquet")
    scores_path = artifacts.get("scores.parquet")
    enriched_path = artifacts.get("scores_enriched.parquet")
    if (
        union_path is None
        or scores_path is None
        or enriched_path is None
        or not union_path.is_file()
        or not scores_path.is_file()
        or not enriched_path.is_file()
    ):
        for name in (
            "pair_set_completeness",
            "hybrid_to_ce_eval",
            "hybrid_to_lambdamart_to_ce_eval",
            "ce_score_feature_ablation",
            "failure_slice_analysis",
        ):
            if name not in checks:
                _fail(name, "missing pair_union.parquet, scores.parquet, or scores_enriched")
    else:
        try:
            union = pl.read_parquet(union_path)
            scores = pl.read_parquet(scores_path)
            enriched = pl.read_parquet(enriched_path)
            verify_ce_union_frame(union, expected_rows=expected_rows)
            verify_final_scores(
                union.select("query_key", "product_key", "split"),
                scores.select(*SCORES_MINIMAL_COLUMNS),
            )
            if scores.height != union.height:
                raise ValueError(f"scores rows {scores.height} != union rows {union.height}")
            if enriched.height != union.height:
                raise ValueError(f"enriched rows {enriched.height} != union rows {union.height}")
            checks["pair_set_completeness"] = _downstream_check(
                "PASS",
                rows=union.height,
                unique_pairs=union.select("query_key", "product_key").n_unique(),
            )
        except Exception as exc:
            _fail("pair_set_completeness", str(exc))

        try:
            hybrid = union.filter(pl.col("in_hybrid_top_100"))
            if hybrid.is_empty():
                raise ValueError("no hybrid top-100 pairs")
            missing = hybrid.join(
                scores.select("query_key", "product_key"),
                on=["query_key", "product_key"],
                how="anti",
            )
            if not missing.is_empty():
                raise ValueError(f"missing scores for {missing.height} hybrid pairs")
            checks["hybrid_to_ce_eval"] = _downstream_check(
                "PASS",
                pairs=hybrid.height,
                eval_command="make rank-m3-ce-evaluate",
                method="hybrid_to_cross_encoder",
            )
        except Exception as exc:
            _fail("hybrid_to_ce_eval", str(exc))

        try:
            cascade = union.filter(pl.col("in_lambdamart_top_50"))
            if cascade.is_empty():
                raise ValueError("no lambdamart top-50 pairs")
            missing = cascade.join(
                scores.select("query_key", "product_key"),
                on=["query_key", "product_key"],
                how="anti",
            )
            if not missing.is_empty():
                raise ValueError(f"missing scores for {missing.height} lambdamart pairs")
            checks["hybrid_to_lambdamart_to_ce_eval"] = _downstream_check(
                "PASS",
                pairs=cascade.height,
                eval_command="make rank-m3-ce-evaluate",
                method="hybrid_to_lambdamart_to_cross_encoder",
            )
        except Exception as exc:
            _fail("hybrid_to_lambdamart_to_ce_eval", str(exc))

        try:
            missing_cols = _missing_columns(enriched, set(SCORES_ENRICHED_COLUMNS))
            if missing_cols:
                raise ValueError(f"enriched scores missing columns: {missing_cols}")
            checks["ce_score_feature_ablation"] = _downstream_check(
                "PASS",
                columns=list(SCORES_ENRICHED_COLUMNS),
                role="LambdaMART+CE-score ablation via scores_enriched.parquet",
            )
        except Exception as exc:
            _fail("ce_score_feature_ablation", str(exc))

        try:
            required = {"split", "in_hybrid_top_100", "in_lambdamart_top_50", "cross_encoder_score"}
            missing_cols = _missing_columns(enriched, required)
            if missing_cols:
                raise ValueError(f"slice columns missing: {missing_cols}")
            checks["failure_slice_analysis"] = _downstream_check(
                "PASS",
                splits=enriched.get_column("split").unique().sort().to_list(),
                hybrid_pairs=int(enriched.get_column("in_hybrid_top_100").sum()),
                lambdamart_pairs=int(enriched.get_column("in_lambdamart_top_50").sum()),
            )
        except Exception as exc:
            _fail("failure_slice_analysis", str(exc))

    benchmark_path = artifacts.get("benchmark.json")
    scoring_stats_path = artifacts.get("scoring_stats.json")
    try:
        if benchmark_path is None or not benchmark_path.is_file():
            raise FileNotFoundError("benchmark.json missing")
        if scoring_stats_path is None or not scoring_stats_path.is_file():
            raise FileNotFoundError("scoring_stats.json missing")
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        scoring_stats = json.loads(scoring_stats_path.read_text(encoding="utf-8"))
        if not benchmark.get("trials"):
            raise ValueError("benchmark.json has no trials")
        if benchmark.get("selected_batch_size") is None:
            raise ValueError("benchmark.json missing selected_batch_size")
        if scoring_stats.get("pairs_scored") is None:
            raise ValueError("scoring_stats.json missing pairs_scored")
        checks["quality_latency_analysis"] = _downstream_check(
            "PASS",
            selected_batch_size=benchmark.get("selected_batch_size"),
            pairs_scored=scoring_stats.get("pairs_scored"),
            elapsed_seconds=scoring_stats.get("elapsed_seconds"),
            latency_caveat=(
                "CE latency is Colab A100 CUDA; compare quality metrics only across methods"
            ),
        )
    except Exception as exc:
        checks["quality_latency_analysis"] = _downstream_check("FAIL", error=str(exc))

    distribution_path = artifacts.get("score_distribution.json")
    try:
        if distribution_path is None or not distribution_path.is_file():
            raise FileNotFoundError("score_distribution.json missing")
        distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
        if "global" not in distribution or "by_split" not in distribution:
            raise ValueError("score_distribution.json missing global or by_split sections")
        checks["score_distribution_analysis"] = _downstream_check(
            "PASS",
            global_count=distribution["global"].get("count"),
            splits=sorted(distribution["by_split"]),
        )
    except Exception as exc:
        checks["score_distribution_analysis"] = _downstream_check("FAIL", error=str(exc))

    provenance_path = artifacts.get("provenance.json")
    manifest_path = artifacts.get("pair_union_manifest.json")
    try:
        if provenance_path is None or not provenance_path.is_file():
            raise FileNotFoundError("provenance.json missing")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        required_keys = {
            "notebook_commit",
            "scoring_code_commit",
            "artifact_base_commit",
            "dataset_fingerprint",
            "union_manifest",
            "input_archive_sha256",
            "model",
        }
        missing_keys = sorted(required_keys - set(provenance))
        if missing_keys:
            raise ValueError(f"provenance.json missing keys: {missing_keys}")
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError("pair_union_manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("union_pairs", -1)) != int(
            provenance.get("union_manifest", {}).get("union_pairs", -2)
        ):
            raise ValueError("provenance union_manifest does not match pair_union_manifest.json")
        checks["provenance_reconstruction"] = _downstream_check(
            "PASS",
            notebook_commit=provenance.get("notebook_commit"),
            scoring_code_commit=provenance.get("scoring_code_commit"),
            dataset_fingerprint=provenance.get("dataset_fingerprint"),
        )
    except Exception as exc:
        checks["provenance_reconstruction"] = _downstream_check("FAIL", error=str(exc))

    bundle_path = artifacts.get(CE_LOCAL_TRANSFER_BUNDLE)
    try:
        if bundle_path is None or not bundle_path.is_file():
            raise FileNotFoundError(f"{CE_LOCAL_TRANSFER_BUNDLE} missing")
        import_script = repo_root / "scripts" / "import_m3_ce_outputs.py"
        evaluate_config = repo_root / "configs" / "ranking" / "m3_ce_evaluate.yaml"
        if not import_script.is_file() or not evaluate_config.is_file():
            raise FileNotFoundError("local import or evaluate config is missing from repo")
        with tarfile.open(bundle_path, "r:gz") as handle:
            names = {Path(member.name).name for member in handle.getmembers() if member.isfile()}
        missing_members = [name for name in CE_CANONICAL_ARTIFACTS if name not in names]
        if missing_members:
            raise ValueError(f"transfer bundle missing canonical artifacts: {missing_members}")
        checks["local_cascade_eval_import"] = _downstream_check(
            "PASS",
            bundle=str(bundle_path.resolve()),
            import_command="make import-m3-ce-outputs",
            evaluate_command="make rank-m3-ce-evaluate",
            member_count=len(names),
        )
    except Exception as exc:
        checks["local_cascade_eval_import"] = _downstream_check("FAIL", error=str(exc))

    audit_status = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    return {
        "status": audit_status,
        "checks": checks,
        "no_rerun_statement": NO_RERUN_DOWNSTREAM_STATEMENT,
    }


def _create_tar_bundle(bundle_path: Path, members: list[tuple[Path, str]]) -> dict[str, Any]:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as handle:
        for source, arcname in members:
            if not source.is_file():
                raise FileNotFoundError(f"bundle member missing: {source}")
            handle.add(source, arcname=arcname)
    temporary.replace(bundle_path)
    return {
        "path": str(bundle_path.resolve()),
        "sha256": sha256_file(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
    }


def _benchmark_subset_fingerprint(subset: pl.DataFrame) -> str:
    import hashlib

    payload = (
        subset.select("query_key", "product_key", "split")
        .sort("split", "query_key", "product_key")
        .write_csv()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finalize_m3_ce_run(
    *,
    drive_root: Path,
    union_path: Path,
    union_manifest_path: Path,
    checkpoint_path: Path,
    pair_frame: pl.DataFrame,
    scorer: CrossEncoderScorer,
    fields: PairField,
    gpu_info: dict[str, Any],
    benchmark_trials: list[dict[str, Any]],
    benchmark_subset: pl.DataFrame,
    selected_batch_size: int,
    scoring_elapsed_seconds: float,
    notebook_commit: str,
    scoring_code_commit: str,
    artifact_base_commit: str,
    dataset_fingerprint: str,
    union_sha256: str,
    input_archive_sha256: str,
    run_times: dict[str, float] | None = None,
    manifest_path: Path | None = None,
    expected_rows: int | None = EXPECTED_CE_UNION_ROWS,
) -> dict[str, Any]:
    final_dir = drive_root / "final"
    metadata_dir = drive_root / "metadata"
    logs_dir = drive_root / "logs"
    for directory in (final_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    part_dir = checkpoint_path.with_suffix(checkpoint_path.suffix + ".parts")
    scores_path = final_dir / "scores.parquet"
    if part_dir.is_dir() and list_completed_blocks(part_dir):
        consolidated = consolidate_part_blocks(
            part_dir,
            scores_path,
            expected_pairs=pair_frame.select("query_key", "product_key", "split"),
        )
    elif checkpoint_path.is_file():
        frame = pl.read_parquet(checkpoint_path).select(*SCORES_MINIMAL_COLUMNS)
        atomic_write_parquet(scores_path, frame.sort("query_key", "product_key"))
        consolidated = frame
    else:
        raise FileNotFoundError(f"no CE checkpoint or part blocks at {checkpoint_path}")
    scores = consolidated.select(*SCORES_MINIMAL_COLUMNS)
    scores_sha256 = sha256_file(scores_path)
    enriched = enrich_scores(pair_frame, scores)
    enriched_path = final_dir / "scores_enriched.parquet"
    atomic_write_parquet(enriched_path, enriched)
    final_union = final_dir / "pair_union.parquet"
    final_union_manifest = final_dir / "pair_union_manifest.json"
    shutil.copy2(union_path, final_union)
    shutil.copy2(union_manifest_path, final_union_manifest)
    union_manifest = json.loads(union_manifest_path.read_text(encoding="utf-8"))
    scoring_stats = build_scoring_stats_dict(
        scorer,
        scores.height,
        scoring_elapsed_seconds,
        dataset_fingerprint=dataset_fingerprint,
        union_sha256=union_sha256,
        scores_sha256=scores_sha256,
        pairs_by_split=scores.group_by("split").len().sort("split").to_dicts(),
        checkpoint_path=str(checkpoint_path.resolve()),
        parts_dir_preserved=str(part_dir.resolve()) if part_dir.is_dir() else None,
    )
    scoring_stats_path = metadata_dir / "scoring_stats.json"
    atomic_write_json(scoring_stats_path, scoring_stats)
    benchmark = build_benchmark_dict(
        benchmark_trials,
        _benchmark_subset_fingerprint(benchmark_subset),
        gpu_info,
        selected_batch_size,
    )
    benchmark_path = metadata_dir / "benchmark.json"
    atomic_write_json(benchmark_path, benchmark)
    validation_report = extended_validation_report(
        pair_frame, scores, union_sha256=union_sha256, scores_sha256=scores_sha256
    )
    validation_report_path = metadata_dir / "validation_report.json"
    atomic_write_json(validation_report_path, validation_report)
    distribution = score_distribution_report(scores)
    distribution_path = metadata_dir / "score_distribution.json"
    atomic_write_json(distribution_path, distribution)
    model_provenance = collect_model_provenance(scorer, fields)
    provenance = build_provenance_record(
        notebook_commit=notebook_commit,
        scoring_code_commit=scoring_code_commit,
        artifact_base_commit=artifact_base_commit,
        dataset_fingerprint=dataset_fingerprint,
        union_manifest=union_manifest,
        input_archive_sha256=input_archive_sha256,
        model_provenance=model_provenance,
        union_sha256=union_sha256,
        scores_sha256=scores_sha256,
    )
    provenance_path = metadata_dir / "provenance.json"
    atomic_write_json(provenance_path, provenance)
    runtime = {
        "run_times_seconds": run_times or {},
        "scoring_elapsed_seconds": scoring_elapsed_seconds,
        "notebook_commit": notebook_commit,
        "scoring_code_commit": scoring_code_commit,
    }
    runtime_path = metadata_dir / "runtime.json"
    atomic_write_json(runtime_path, runtime)
    manifest = build_artifact_manifest(drive_root)
    manifest_path_out = metadata_dir / "artifact_manifest.json"
    atomic_write_json(manifest_path_out, manifest)
    artifact_map = {
        "pair_union.parquet": final_union,
        "pair_union_manifest.json": final_union_manifest,
        "scores.parquet": scores_path,
        "scores_enriched.parquet": enriched_path,
        "scoring_stats.json": scoring_stats_path,
        "benchmark.json": benchmark_path,
        "validation_report.json": validation_report_path,
        "score_distribution.json": distribution_path,
        "provenance.json": provenance_path,
        "runtime.json": runtime_path,
        "artifact_manifest.json": manifest_path_out,
    }
    audit = run_completeness_audit(artifact_map, status="SUCCESS")
    transfer_members = [
        (artifact_map[name], f"cross_encoder/{name}") for name in CE_CANONICAL_ARTIFACTS
    ]
    transfer_bundle = _create_tar_bundle(final_dir / CE_LOCAL_TRANSFER_BUNDLE, transfer_members)
    artifact_map[CE_LOCAL_TRANSFER_BUNDLE] = final_dir / CE_LOCAL_TRANSFER_BUNDLE
    downstream_readiness_audit = run_downstream_readiness_audit(
        artifact_map, expected_rows=expected_rows
    )
    downstream_audit_path = metadata_dir / "downstream_readiness_audit.json"
    atomic_write_json(downstream_audit_path, downstream_readiness_audit)
    if downstream_readiness_audit["status"] != "PASS":
        failed = [
            name
            for name, payload in downstream_readiness_audit["checks"].items()
            if payload.get("status") != "PASS"
        ]
        raise ValueError(f"downstream readiness audit failed for: {failed}")
    full_members = [(path, f"final/{path.name}") for path in final_dir.iterdir() if path.is_file()]
    full_members.extend(
        (path, f"metadata/{path.name}") for path in metadata_dir.iterdir() if path.is_file()
    )
    if manifest_path is not None and manifest_path.is_file():
        full_members.append((manifest_path, f"checkpoints/{manifest_path.name}"))
    full_bundle = _create_tar_bundle(final_dir / CE_FULL_BUNDLE, full_members)
    return {
        "scores_path": str(scores_path.resolve()),
        "scores_enriched_path": str(enriched_path.resolve()),
        "scores_sha256": scores_sha256,
        "union_sha256": union_sha256,
        "artifact_manifest": manifest,
        "completeness_audit": audit,
        "transfer_bundle": transfer_bundle,
        "full_bundle": full_bundle,
        "downstream_readiness_audit": downstream_readiness_audit,
        "rows": scores.height,
    }


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
    scores = scores.sort("query_key", "product_key")
    atomic_write_parquet(output_path, scores)
    return scores


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
