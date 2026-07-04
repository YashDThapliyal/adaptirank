"""Import durable M3 CE outputs from the local transfer bundle."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import polars as pl

from adaptirank.common.paths import project_root
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.ce_workflow import (
    CANONICAL_DATASET_FINGERPRINT,
    CE_CANONICAL_ARTIFACTS,
    CE_LOCAL_TRANSFER_BUNDLE,
    EXPECTED_CE_UNION_ROWS,
    EXPECTED_CE_UNION_SHA256,
    atomic_write_json,
    atomic_write_parquet,
    run_completeness_audit,
    verify_ce_union_frame,
    verify_file_sha256,
    verify_final_scores,
)


def _extract_bundle(archive: Path, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in CE_CANONICAL_ARTIFACTS:
                continue
            target = destination / name
            handle.extract(member, path=destination.parent)
            source = destination.parent / member.name
            if source != target:
                source.replace(target)
            extracted[name] = target
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive",
        type=Path,
        nargs="?",
        default=project_root() / "artifacts" / "handoffs" / CE_LOCAL_TRANSFER_BUNDLE,
    )
    args = parser.parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"transfer bundle not found: {archive}")

    root = project_root()
    ce_root = (
        root
        / "artifacts"
        / "ranking"
        / CANONICAL_DATASET_FINGERPRINT
        / "m3_three_split"
        / "cross_encoder"
    )
    ce_root.mkdir(parents=True, exist_ok=True)
    staging = ce_root / ".import_staging"
    if staging.exists():
        for path in staging.rglob("*"):
            if path.is_file():
                path.unlink()
    staging.mkdir(parents=True, exist_ok=True)

    extracted = _extract_bundle(archive, staging)
    missing = [name for name in CE_CANONICAL_ARTIFACTS if name not in extracted]
    if missing:
        raise FileNotFoundError(f"bundle missing canonical artifacts: {missing}")

    union = pl.read_parquet(extracted["pair_union.parquet"])
    verify_file_sha256(extracted["pair_union.parquet"], EXPECTED_CE_UNION_SHA256)
    verify_ce_union_frame(union, expected_rows=EXPECTED_CE_UNION_ROWS)
    scores = pl.read_parquet(extracted["scores.parquet"])
    verify_final_scores(
        union.select("query_key", "product_key", "split"),
        scores,
    )
    if scores.height != union.height:
        raise ValueError(f"scores rows {scores.height} != union rows {union.height}")

    for name, source in extracted.items():
        target = ce_root / name
        if name.endswith(".parquet"):
            atomic_write_parquet(target, pl.read_parquet(source))
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            atomic_write_json(target, payload)

    audit = run_completeness_audit({name: ce_root / name for name in CE_CANONICAL_ARTIFACTS})
    import_manifest = {
        "source_archive": str(archive),
        "source_archive_sha256": sha256_file(archive),
        "destination": str(ce_root.resolve()),
        "rows": scores.height,
        "union_sha256": EXPECTED_CE_UNION_SHA256,
        "scores_sha256": sha256_file(ce_root / "scores.parquet"),
        "completeness_audit": audit,
    }
    atomic_write_json(ce_root / "import_manifest.json", import_manifest)
    print(json.dumps(import_manifest, indent=2))
    print()
    print("Next step: make rank-m3-ce-evaluate")
    print(f"CE artifacts installed under: {ce_root}")


if __name__ == "__main__":
    main()
