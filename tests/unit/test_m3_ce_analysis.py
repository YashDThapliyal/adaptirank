from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import polars as pl
import pytest


def _module() -> types.ModuleType:
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    path = scripts_dir / "analyze_m3_ce_findings.py"
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


def test_delta_returns_none_for_missing_metric_values() -> None:
    mod = _module()
    rows = {
        "ce": {"ndcg_10": 0.7, "mrr": None},
        "base": {"ndcg_10": 0.5, "mrr": 0.4},
    }

    assert mod.delta(rows, "ce", "base", "ndcg_10") == pytest.approx(0.2)
    assert mod.delta(rows, "ce", "base", "mrr") is None


def test_label_expr_preserves_esci_labels_and_marks_unjudged() -> None:
    mod = _module()
    frame = pl.DataFrame({"esci_label": ["E", "S", "C", "I", None]})

    labels = frame.with_columns(mod.label_expr())["label"].to_list()

    assert labels == ["E", "S", "C", "I", "UNJUDGED"]


def test_slice_notes_handles_empty_rows() -> None:
    mod = _module()
    notes = mod.slice_notes(
        {
            "query_length": {
                "split": "test",
                "rows": [],
                "pointwise_beats_lambdamart_slices": [],
            }
        }
    )

    assert notes == "- query_length: no slice rows available."


def test_keyword_compatibility_examples_select_low_ce_scores_in_hybrid_top100() -> None:
    mod = _module()
    frame = pl.DataFrame(
        {
            "query_key": ["q2", "q1", "q1", "q1", "q0"],
            "product_key": ["a", "b", "a", "c", "z"],
            "in_hybrid_top_100": [True, True, True, True, False],
            "cross_encoder_score": [0.2, 0.1, 0.1, 0.9, -10.0],
        }
    )

    selected = mod.keyword_compatibility_examples_base(frame)

    assert selected["query_key"].to_list() == ["q1", "q1", "q2", "q1"]
    assert selected["product_key"].to_list() == ["a", "b", "a", "c"]
    assert selected["cross_encoder_score"].to_list() == [0.1, 0.1, 0.2, 0.9]
