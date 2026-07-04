"""Unit tests for the canonical CE workflow helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from adaptirank.ranking.ce_workflow import (
    consolidate_part_blocks,
    score_union_with_checkpoints,
    verify_ce_union_frame,
    verify_file_sha256,
    verify_final_scores,
    write_block_manifest,
)
from adaptirank.ranking.cross_encoder import CrossEncoderScorer


class _FakeScorer(CrossEncoderScorer):
    def __init__(self) -> None:
        super().__init__(model_name="fake", model_revision="fake", device="cpu", batch_size=2)
        self.scored_pairs = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.scored_pairs += len(pairs)
        return np.array(
            [float(len(query) + len(product)) for query, product in pairs], dtype=np.float32
        )


def _union() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_key": ["q1", "q1", "q2", "q2"],
            "product_key": ["p1", "p2", "p1", "p3"],
            "split": ["train", "train", "validation", "validation"],
            "hybrid_rank": [1, 2, 1, None],
            "lambdamart_rank": [1, None, None, 1],
            "in_hybrid_top_100": [True, True, True, False],
            "in_lambdamart_top_50": [True, False, False, True],
        }
    )


def _queries() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_key": ["q1", "q2"],
            "query_text": ["trail shoe", "water bottle"],
        }
    )


def _catalog() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "product_key": ["p1", "p2", "p3"],
            "title": ["shoe", "sock", "bottle"],
            "description": ["fast", "warm", "steel"],
            "brand": ["A", "B", "C"],
        }
    )


def test_verify_file_sha256_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "payload.txt"
    path.write_text("actual\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_file_sha256(path, "0" * 64)


def test_verify_ce_union_frame_duplicate_detection_precedes_row_count() -> None:
    duplicate = pl.concat([_union(), _union().head(1)], how="vertical")
    with pytest.raises(ValueError, match="pairs must be unique"):
        verify_ce_union_frame(duplicate)


def test_verify_final_scores_pair_equality_and_nan_rejection() -> None:
    pairs = _union().select("query_key", "product_key", "split")
    scores = pairs.with_columns(pl.lit(1.0).cast(pl.Float32).alias("cross_encoder_score"))
    stats = verify_final_scores(pairs, scores)
    assert stats["rows"] == 4

    missing = scores.head(3)
    with pytest.raises(ValueError, match="score pair mismatch"):
        verify_final_scores(pairs, missing)

    nan_scores = scores.with_columns(
        pl.when(pl.col("query_key") == "q1")
        .then(float("nan"))
        .otherwise(pl.col("cross_encoder_score"))
        .alias("cross_encoder_score")
    )
    with pytest.raises(ValueError, match="invalid cross_encoder_score"):
        verify_final_scores(pairs, nan_scores)


def test_write_block_manifest_and_consolidate_part_blocks(tmp_path: Path) -> None:
    pairs = _union().select("query_key", "product_key", "split")
    scores = pairs.with_columns(
        pl.arange(0, pairs.height).cast(pl.Float32).alias("cross_encoder_score")
    )
    part_dir = tmp_path / "scores.parquet.parts"
    part_dir.mkdir()
    scores.head(2).write_parquet(part_dir / "part-00000000.parquet")
    scores.tail(2).write_parquet(part_dir / "part-00000002.parquet")

    final_path = tmp_path / "scores.parquet"
    consolidated = consolidate_part_blocks(part_dir, final_path, expected_pairs=pairs)
    assert consolidated.height == scores.height
    assert final_path.is_file()

    manifest = write_block_manifest(
        tmp_path / "manifest.json",
        checkpoint_path=final_path,
        part_dir=part_dir,
        metadata={"purpose": "unit"},
    )
    assert manifest["checkpoint_exists"] is True
    assert len(manifest["completed_blocks"]) == 2


def test_score_union_with_checkpoints_resumes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scores.parquet"
    manifest = tmp_path / "manifest.json"
    first = _FakeScorer()
    out1 = score_union_with_checkpoints(
        _union(),
        _queries(),
        _catalog(),
        first,
        fields=("title", "description", "brand"),
        checkpoint_path=checkpoint,
        block_queries=1,
        expected_rows=4,
        manifest_path=manifest,
    )
    assert out1.height == 4
    assert checkpoint.is_file()
    assert manifest.is_file()
    assert first.scored_pairs == 4

    second = _FakeScorer()
    out2 = score_union_with_checkpoints(
        _union(),
        _queries(),
        _catalog(),
        second,
        fields=("title", "description", "brand"),
        checkpoint_path=checkpoint,
        block_queries=1,
        expected_rows=4,
    )
    assert out2.height == 4
    assert second.scored_pairs == 0
