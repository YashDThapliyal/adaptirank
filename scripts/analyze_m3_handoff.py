"""Validate the M3 candidate handoff and compare it with canonical M2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.run import ExperimentRun
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.config import HandoffAnalysisConfig
from adaptirank.ranking.handoff import candidate_delta_analysis, metric_deltas, validate_contract
from adaptirank.retrieval.evaluate import write_json


def _run_metadata(root: Path, run_id: str) -> dict[str, Any]:
    path = root / "artifacts" / "runs" / run_id / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload: dict[str, Any] = json.loads(path.read_text())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, HandoffAnalysisConfig)
    root = project_root()
    retrieval = resolve_project_path(config.retrieval_output_dir, root) / config.dataset_fingerprint
    m2_root = retrieval / config.m2_artifact_name
    m3_root = retrieval / config.m3_artifact_name
    contract = m3_root / "hybrid" / "candidate_contract.parquet"
    source_runs = config.source_runs.model_dump()
    provenance = {name: _run_metadata(root, run_id) for name, run_id in source_runs.items()}
    for name, metadata in provenance.items():
        if metadata["status"] != "SUCCESS":
            raise ValueError(f"source run {name} is not SUCCESS: {metadata['status']}")
        if metadata["dataset_fingerprint"] != config.dataset_fingerprint:
            raise ValueError(f"source run {name} has the wrong dataset fingerprint")

    with ExperimentRun(
        experiment=config.run.experiment,
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        output_dir = m3_root / "handoff_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_contract(contract)
        deltas = metric_deltas(m2_root, m3_root)
        overlaps = candidate_delta_analysis(m2_root, m3_root, top_k=config.top_k)
        manifest = {
            "dataset_fingerprint": config.dataset_fingerprint,
            "artifact_role": "M3 three-split retrieval handoff",
            "m2_canonical_preserved": True,
            "split_policy": {
                "train": "model fitting only",
                "validation": "selection and early stopping only",
                "test": "final evaluation only",
            },
            "label_policy": {
                "grades": {"E": 3, "S": 2, "C": 1, "I": 0},
                "primary_training": "judged rows only",
                "unjudged": "preserved as null label and null grade; never coerced to I or 0",
            },
            "source_run_ids": source_runs,
            "source_run_provenance": provenance,
            "artifact_fingerprints": {
                "m2_contract_sha256": sha256_file(
                    m2_root / "hybrid" / "candidate_contract.parquet"
                ),
                "m3_bm25_candidates_sha256": sha256_file(
                    m3_root / "bm25" / "selected_candidates.parquet"
                ),
                "m3_dense_candidates_sha256": sha256_file(
                    m3_root / "dense" / "raw_candidates.parquet"
                ),
                "m3_contract_sha256": validation["sha256"],
            },
            "contract_validation": validation,
        }
        manifest_path = output_dir / "manifest.json"
        delta_path = output_dir / "m2_m3_delta_analysis.json"
        write_json(manifest_path, manifest)
        write_json(delta_path, {"metric_deltas_m3_minus_m2": deltas, "candidate_overlap": overlaps})
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("handoff_manifest", manifest_path)
        run.record_artifact("delta_analysis", delta_path)
        run.record_metrics(
            {
                "contract_rows": validation["rows"],
                "contract_queries": validation["queries"],
                "contract_sha256": validation["sha256"],
                "source_run_ids": source_runs,
            }
        )
        print(run.run_dir)


if __name__ == "__main__":
    main()
