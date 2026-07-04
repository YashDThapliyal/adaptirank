"""Unit tests for the cross-encoder scorer (no model download; deterministic fake scorer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from adaptirank.common.config import load_config
from adaptirank.ranking.config import CrossEncoderRunConfig
from adaptirank.ranking.cross_encoder import (
    CrossEncoderScorer,
    build_product_text,
    score_pair_frame,
    score_top_m,
)

_PINNED_CE_REVISION = "7b0235231ca2674cb8ca8f022859a6eba2b1c968"


def test_cross_encoder_configs_are_pinned_and_consistent() -> None:
    smoke = load_config(Path("configs/ranking/cross_encoder_smoke.yaml"), CrossEncoderRunConfig)
    m3 = load_config(Path("configs/ranking/cross_encoder_m3.yaml"), CrossEncoderRunConfig)
    for cfg in (smoke, m3):
        assert cfg.cross_encoder.model_name == "cross-encoder/ms-marco-MiniLM-L12-v2"
        assert cfg.cross_encoder.model_revision == _PINNED_CE_REVISION
    # Smoke is CPU + capped; the full M3 run is uncapped, top-100, auto device (CUDA on Colab).
    assert smoke.cross_encoder.device == "cpu" and smoke.max_queries_per_split == 25
    assert m3.max_queries_per_split is None
    assert m3.top_m == 100
    assert m3.cross_encoder.device == "auto"
    assert m3.retrieval_artifact_name == "m3_three_split"


class _FakeScorer(CrossEncoderScorer):
    """Deterministic scorer that avoids any model load; records how many pairs it scored."""

    def __init__(self) -> None:
        super().__init__(model_name="fake", model_revision="fake", device="cpu")
        self.scored_pairs = 0

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.scored_pairs += len(pairs)
        # Score = length of the product side, so scores are stable and inspectable.
        return np.array([float(len(p[1])) for p in pairs], dtype=np.float32)


def test_build_product_text_is_deterministic_and_skips_empty() -> None:
    row = {"title": "Trail Shoe", "description": None, "brand": "North"}
    assert build_product_text(row, ("title", "description", "brand")) == "Trail Shoe North"
    # Field order is fixed and independent of dict order.
    assert build_product_text(row, ("brand", "title")) == "North Trail Shoe"
    assert build_product_text({"title": "", "brand": "X"}, ("title", "brand")) == "X"


def _fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    contract = pl.DataFrame(
        {
            "query_key": ["q1", "q1", "q1", "q2", "q2"],
            "product_key": ["p1", "p2", "p3", "p1", "p4"],
            "hybrid_rank": [1, 2, 3, 1, 2],
            "split": ["test", "test", "test", "validation", "validation"],
        }
    )
    queries = pl.DataFrame(
        {"query_key": ["q1", "q2"], "query_text": ["trail shoe", "water bottle"]}
    )
    catalog = pl.DataFrame(
        {
            "product_key": ["p1", "p2", "p3", "p4"],
            "title": ["aa", "bbbb", "cccccc", "dd"],
        }
    )
    return contract, queries, catalog


def test_score_top_m_selects_only_top_m_per_query() -> None:
    contract, queries, catalog = _fixture()
    scorer = _FakeScorer()
    out = score_top_m(contract, queries, catalog, scorer, fields=("title",), top_m=2)
    # top_m=2 drops p3 (rank 3 in q1); q1 keeps p1,p2; q2 keeps p1,p4 -> 4 pairs.
    assert out.height == 4
    assert set(out.columns) == {"query_key", "product_key", "split", "cross_encoder_score"}
    assert "p3" not in out.get_column("product_key").to_list()
    # Score equals product-title length under the fake scorer.
    row = out.filter((pl.col("query_key") == "q1") & (pl.col("product_key") == "p2")).row(
        0, named=True
    )
    assert row["cross_encoder_score"] == 4.0


def test_score_pair_frame_accepts_explicit_union() -> None:
    contract, queries, catalog = _fixture()
    targets = contract.select("query_key", "product_key", "split").head(2)
    result = score_pair_frame(targets, queries, catalog, _FakeScorer(), fields=("title",))
    assert result.height == 2
    assert result.select("query_key", "product_key").n_unique() == 2


def test_score_top_m_checkpoint_resumes_without_rescoring(tmp_path: Path) -> None:
    contract, queries, catalog = _fixture()
    ckpt = tmp_path / "scores.parquet"
    first = _FakeScorer()
    out1 = score_top_m(
        contract, queries, catalog, first, fields=("title",), top_m=2, checkpoint_path=ckpt
    )
    assert ckpt.is_file()
    assert first.scored_pairs == 4
    # Resume: everything already scored, so the second scorer scores nothing new.
    second = _FakeScorer()
    out2 = score_top_m(
        contract, queries, catalog, second, fields=("title",), top_m=2, checkpoint_path=ckpt
    )
    assert second.scored_pairs == 0
    assert out2.height == out1.height == 4
