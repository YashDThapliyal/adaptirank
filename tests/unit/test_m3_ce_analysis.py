from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import polars as pl
import pytest


def _module() -> types.ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_m3_ce_findings.py"
    spec = importlib.util.spec_from_file_location("analyze_m3_ce_findings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_displacements_tracks_direction_and_thresholds() -> None:
    mod = _module()
    summary = mod.summarize_displacements([-12, -1, 0, 3, 30])

    assert summary["count"] == 5
    assert summary["median"] == 0.0
    assert summary["pct_promoted"] == 0.4
    assert summary["pct_demoted"] == 0.4
    assert summary["pct_unchanged"] == 0.2
    assert summary["pct_abs_ge_10"] == 0.4
    assert summary["pct_abs_ge_25"] == 0.2


def test_summarize_displacements_empty_is_json_safe() -> None:
    mod = _module()
    summary = mod.summarize_displacements([])

    assert summary["count"] == 0
    assert summary["mean"] is None
    assert summary["pct_promoted"] is None


def test_mean_metric_delta_uses_paired_rows() -> None:
    mod = _module()
    frame = pl.DataFrame({"ce__ndcg_10": [0.8, 0.2], "base__ndcg_10": [0.5, 0.4]})

    mean_delta = mod.mean_metric_delta(frame, "ndcg_10", "ce", "base")
    missing_delta = mod.mean_metric_delta(frame, "mrr", "ce", "base")

    assert mean_delta == pytest.approx(0.05)
    assert missing_delta is None


def test_count_ordering_violations_checks_product_key_ties() -> None:
    mod = _module()
    correct = pl.DataFrame(
        {
            "query_key": ["q1", "q1", "q1"],
            "product_key": ["a", "b", "c"],
            "score": [2.0, 2.0, 1.0],
            "rank": [1, 2, 3],
        }
    )
    swapped_tie = pl.DataFrame(
        {
            "query_key": ["q1", "q1", "q1"],
            "product_key": ["b", "a", "c"],
            "score": [2.0, 2.0, 1.0],
            "rank": [1, 2, 3],
        }
    )

    correct_violations = mod.count_ordering_violations(correct)
    swapped_tie_violations = mod.count_ordering_violations(swapped_tie)

    assert correct_violations == 0
    assert swapped_tie_violations == 2
