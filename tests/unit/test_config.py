from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptirank.common.config import RunConfig, load_config


def test_config_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("experiment: smoke\npurpose: test\nunknown: true\n")
    with pytest.raises(ValidationError):
        load_config(path, RunConfig)


def test_config_loads_canonical_project_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("experiment: smoke\npurpose: test\nseed: 7\n")
    config = load_config(path, RunConfig)
    assert config.experiment == "smoke"
    assert config.seed == 7
