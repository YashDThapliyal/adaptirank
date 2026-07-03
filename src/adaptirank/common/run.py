"""Experiment artifact contract and provenance capture."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import distributions
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict

from adaptirank.common.logging import configure_logging
from adaptirank.common.paths import project_root, resolve_project_path


class RunStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    project: str = "adaptirank"
    experiment: str
    purpose: str
    seed: int
    git_commit: str
    git_dirty: bool
    dataset_fingerprint: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    status: RunStatus
    artifact_locations: dict[str, str]
    error: str | None = None


def _git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "UNAVAILABLE", True


def _environment_text() -> str:
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in distributions())
    header = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        "packages:",
    ]
    return "\n".join([*header, *packages, ""])


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


class ExperimentRun(AbstractContextManager["ExperimentRun"]):
    """Create and finalize one artifact-contract run directory."""

    def __init__(
        self,
        *,
        experiment: str,
        purpose: str,
        seed: int,
        config: dict[str, Any],
        artifacts_dir: Path = Path("artifacts"),
    ) -> None:
        self.root = project_root()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = f"{stamp}-{experiment}-{uuid4().hex[:8]}"
        self.run_dir = resolve_project_path(artifacts_dir, self.root) / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "plots").mkdir()
        (self.run_dir / "checkpoints").mkdir()
        (self.run_dir / "stderr.log").touch()
        (self.run_dir / "metrics.json").write_text("{}\n")
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
        commit, dirty = _git_state(self.root)
        (self.run_dir / "git_commit.txt").write_text(f"{commit}\n")
        (self.run_dir / "environment.txt").write_text(_environment_text())
        self.logger = configure_logging(self.run_dir / "stdout.log")
        self.metadata = RunMetadata(
            run_id=self.run_id,
            experiment=experiment,
            purpose=purpose,
            seed=seed,
            git_commit=commit,
            git_dirty=dirty,
            start_time=datetime.now(UTC),
            status=RunStatus.PARTIAL,
            artifact_locations={"run_dir": str(self.run_dir)},
        )
        self._write_metadata()

    def _write_metadata(self) -> None:
        _atomic_json(self.run_dir / "metadata.json", self.metadata.model_dump(mode="json"))

    def record_metrics(self, metrics: dict[str, Any]) -> None:
        _atomic_json(self.run_dir / "metrics.json", metrics)

    def record_artifact(self, name: str, path: Path) -> None:
        self.metadata.artifact_locations[name] = str(path.resolve())
        self._write_metadata()

    def set_dataset_fingerprint(self, fingerprint: str) -> None:
        self.metadata.dataset_fingerprint = fingerprint
        self._write_metadata()

    def __enter__(self) -> Self:
        self.logger.info("run_started", extra={"run_id": self.run_id})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.metadata.end_time = datetime.now(UTC)
        if exc_value is None:
            self.metadata.status = RunStatus.SUCCESS
            self.logger.info("run_succeeded")
        else:
            self.metadata.status = RunStatus.FAILED
            self.metadata.error = (
                f"{exc_type.__name__}: {exc_value}" if exc_type else str(exc_value)
            )
            assert exc_type is not None
            self.logger.exception("run_failed", exc_info=(exc_type, exc_value, traceback))
        self._write_metadata()
        return False
