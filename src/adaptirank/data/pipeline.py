"""Tracked M1 pipeline entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptirank.common.config import load_config
from adaptirank.common.reproducibility import seed_everything
from adaptirank.common.run import ExperimentRun
from adaptirank.data.config import EsciConfig
from adaptirank.data.esci import DatasetBuildResult, build_dataset
from adaptirank.data.provenance import load_staged_sources, stage_sources


def load_esci_config(path: Path) -> EsciConfig:
    return load_config(path, EsciConfig)


def _run_config(config: EsciConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")


def run_download(config: EsciConfig) -> Path:
    """Stage all configured source files in a provenance-tracked run."""

    with ExperimentRun(
        experiment=f"{config.run.experiment}_download",
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=_run_config(config),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        seed_everything(config.run.seed)
        paths, manifest = stage_sources(config)
        manifest_path = next(iter(paths.values())).parent / "source_manifest.json"
        run.record_artifact("source_manifest", manifest_path)
        run.record_metrics(
            {
                "source_mode": config.source.mode,
                "files": {item["role"]: item["observed_size_bytes"] for item in manifest["files"]},
            }
        )
        return run.run_dir


def run_build(config: EsciConfig) -> tuple[Path, DatasetBuildResult]:
    """Build normalized artifacts only from an already-staged source set."""

    with ExperimentRun(
        experiment=f"{config.run.experiment}_build",
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=_run_config(config),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        seed_everything(config.run.seed)
        paths, manifest = load_staged_sources(config)
        result = build_dataset(config, paths, manifest)
        run.set_dataset_fingerprint(result.fingerprint)
        run.record_artifact("dataset_dir", result.dataset_dir)
        run.record_artifact("dataset_report", result.dataset_dir / "dataset_report.json")
        run.record_metrics(result.report)
        return run.run_dir, result


def run_pipeline(config: EsciConfig) -> tuple[Path, DatasetBuildResult]:
    """Stage and build within one end-to-end tracked verification run."""

    with ExperimentRun(
        experiment=config.run.experiment,
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=_run_config(config),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        seed_everything(config.run.seed)
        paths, manifest = stage_sources(config)
        result = build_dataset(config, paths, manifest)
        run.set_dataset_fingerprint(result.fingerprint)
        run.record_artifact(
            "source_manifest", next(iter(paths.values())).parent / "source_manifest.json"
        )
        run.record_artifact("dataset_dir", result.dataset_dir)
        run.record_artifact("dataset_report", result.dataset_dir / "dataset_report.json")
        run.record_metrics(result.report)
        return run.run_dir, result
