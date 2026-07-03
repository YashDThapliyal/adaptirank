import json
from pathlib import Path

import pytest

from adaptirank.common.run import ExperimentRun


def test_run_writes_complete_success_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADAPTIRANK_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "AGENTS.md").touch()
    with ExperimentRun(
        experiment="unit",
        purpose="test",
        seed=42,
        config={"seed": 42},
        artifacts_dir=Path("artifacts"),
    ) as run:
        run.record_metrics({"ok": True})
        run_dir = run.run_dir
    required = {
        "config.yaml",
        "metadata.json",
        "metrics.json",
        "stdout.log",
        "stderr.log",
        "git_commit.txt",
        "environment.txt",
        "plots",
        "checkpoints",
    }
    assert required == {path.name for path in run_dir.iterdir()}
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["project"] == "adaptirank"
    assert metadata["status"] == "SUCCESS"
