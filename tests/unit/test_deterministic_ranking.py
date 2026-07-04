"""Tests for canonical deterministic ranking: score DESC, product_key ASC."""

from __future__ import annotations

import polars as pl
import pytest

from adaptirank.common.ordering import (
    assign_deterministic_rank,
    canonical_content_fingerprint,
    verify_pair_uniqueness,
)
from adaptirank.ranking.evaluate import ranked_candidates
from adaptirank.retrieval.hybrid import candidate_contract, reciprocal_rank_fusion, weighted_fusion

CONTRACT_COLUMNS = (
    "query_key",
    "product_key",
    "split",
    "bm25_score",
    "bm25_rank",
    "dense_score",
    "dense_rank",
    "hybrid_score",
    "hybrid_rank",
    "rrf_score",
    "rrf_rank",
    "esci_label",
    "relevance_grade",
    "judgment_status",
)


def _candidates(method: str, products: list[str], scores: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_key": ["q"] * len(products),
            "product_key": products,
            "split": ["validation"] * len(products),
            "method": [method] * len(products),
            "score": scores,
            "rank": list(range(1, len(products) + 1)),
        }
    )


def test_verify_pair_uniqueness_rejects_null_product_key() -> None:
    frame = pl.DataFrame({"query_key": ["q"], "product_key": [None]})
    with pytest.raises(ValueError, match="product_key must be non-null"):
        verify_pair_uniqueness(frame)


def test_verify_pair_uniqueness_rejects_duplicate_pairs() -> None:
    frame = pl.DataFrame({"query_key": ["q", "q"], "product_key": ["a", "a"]})
    with pytest.raises(ValueError, match="pairs must be unique"):
        verify_pair_uniqueness(frame)


def test_equal_score_tie_resolves_by_product_key_asc() -> None:
    frame = pl.DataFrame(
        {
            "query_key": ["q", "q", "q"],
            "product_key": ["c", "a", "b"],
            "split": ["test"] * 3,
            "score": [1.0, 1.0, 1.0],
        }
    )
    ranked = assign_deterministic_rank(frame, score_col="score")
    assert ranked.select("product_key", "rank").to_dicts() == [
        {"product_key": "a", "rank": 1},
        {"product_key": "b", "rank": 2},
        {"product_key": "c", "rank": 3},
    ]


def test_stable_top_k_membership_at_tie_boundary() -> None:
    frame = pl.DataFrame(
        {
            "query_key": ["q"] * 4,
            "product_key": ["d", "b", "a", "c"],
            "split": ["test"] * 4,
            "score": [2.0, 1.0, 1.0, 1.0],
        }
    )
    ranked = assign_deterministic_rank(frame, score_col="score")
    top2 = ranked.filter(pl.col("rank") <= 2).get_column("product_key").to_list()
    assert top2 == ["d", "a"]


def test_shuffled_input_yields_identical_ranking() -> None:
    base = pl.DataFrame(
        {
            "query_key": ["q1", "q1", "q2", "q2"],
            "product_key": ["b", "a", "y", "x"],
            "split": ["test"] * 4,
            "score": [0.5, 0.5, 2.0, 1.0],
        }
    )
    shuffled = base.sample(fraction=1.0, shuffle=True, seed=17)
    first = assign_deterministic_rank(base, score_col="score")
    second = assign_deterministic_rank(shuffled, score_col="score")
    assert first.select("query_key", "product_key", "rank").equals(
        second.select("query_key", "product_key", "rank")
    )


def test_persisted_ranking_matches_evaluation_ranking() -> None:
    frame = pl.DataFrame(
        {
            "query_key": ["q", "q", "q"],
            "product_key": ["c", "a", "b"],
            "split": ["test"] * 3,
            "heuristic_score": [0.9, 0.9, 0.8],
        }
    )
    persisted = assign_deterministic_rank(
        frame, score_col="heuristic_score", rank_col="heuristic_rank"
    )
    evaluated = ranked_candidates(frame, "heuristic_score", "heuristic_rank")
    assert persisted.select("query_key", "product_key", "heuristic_rank").equals(
        evaluated.select("query_key", "product_key", pl.col("rank").alias("heuristic_rank"))
    )


def test_repeated_hybrid_builds_have_identical_logical_contract() -> None:
    bm25 = _candidates("bm25", ["a", "b", "d"], [2.0, 1.0, 1.0])
    dense = _candidates("dense", ["b", "c", "d"], [0.9, 0.8, 0.8])
    relevance = pl.DataFrame(
        {
            "query_key": ["q"],
            "product_key": ["a"],
            "esci_label": ["E"],
            "relevance_grade": [3],
        }
    )
    contracts = []
    for seed in (1, 2, 99):
        weighted, _ = weighted_fusion(
            bm25.sample(fraction=1.0, shuffle=True, seed=seed), dense, alpha=0.5, top_k=3
        )
        rrf, _ = reciprocal_rank_fusion(
            bm25.sample(fraction=1.0, shuffle=True, seed=seed + 100),
            dense.sample(fraction=1.0, shuffle=True, seed=seed + 200),
            rrf_k=60,
            top_k=3,
        )
        contracts.append(candidate_contract(bm25, dense, weighted, rrf, relevance))
    fingerprints = [
        canonical_content_fingerprint(contract, columns=CONTRACT_COLUMNS) for contract in contracts
    ]
    assert len(set(fingerprints)) == 1
    assert contracts[0].select(CONTRACT_COLUMNS).equals(contracts[1].select(CONTRACT_COLUMNS))


def test_ranked_candidates_uses_product_key_only_for_equal_scores() -> None:
    frame = pl.DataFrame(
        {
            "query_key": ["q", "q"],
            "product_key": ["b", "a"],
            "split": ["test", "test"],
            "model_score": [1.0, 1.0],
        }
    )
    ranked = ranked_candidates(frame, "model_score", "model_rank")
    assert ranked.get_column("product_key").to_list() == ["a", "b"]
    assert ranked.get_column("rank").to_list() == [1, 2]
