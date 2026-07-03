"""Hand-computable audit of M2 retrieval metric semantics.

Each test pins one documented semantic so that a silent redefinition (e.g. relabelling
unjudged items as irrelevant, switching NDCG to linear gain, or changing the recall
denominator) fails loudly. Fixtures are tiny and computed by hand in the comments.
"""

import math

import pytest

from adaptirank.retrieval.evaluate import (
    PRIMARY_LABELS,
    SENSITIVITY_LABELS,
    ndcg_condensed,
    recall_at_k,
    reciprocal_rank_condensed,
)


def test_relevance_label_sets_are_documented() -> None:
    # Primary relevance = Exact + Substitute; sensitivity additionally counts Complement.
    assert PRIMARY_LABELS == {"E", "S"}
    assert SENSITIVITY_LABELS == {"E", "S", "C"}


# --------------------------------------------------------------------------- recall


def test_unjudged_is_ignored_not_converted_to_irrelevant() -> None:
    labels = [None, "I", None, "E", "S"]
    # denominator = 2 known relevant; hits in first 4 raw slots = E only -> 1/2.
    assert recall_at_k(labels, relevant_count=2, k=4) == 0.5
    # condensed MRR: skip None, I is judged-but-not-primary (rank 1), E is primary at
    # condensed rank 2 -> 1/2.
    assert reciprocal_rank_condensed(labels) == 0.5


def test_recall_denominator_is_total_known_relevant_not_retrieved_count() -> None:
    # Two relevant items retrieved in the top-3, but the query has 4 known relevant.
    labels = ["E", None, "S"]
    assert recall_at_k(labels, relevant_count=4, k=3) == 0.5
    # Denominator does not shrink to the number retrieved (which would give 1.0).


def test_recall_top_k_truncation_uses_raw_positions() -> None:
    # An unjudged item at rank 1 occupies a raw slot and pushes the relevant item out
    # of the top-1 window; recall@1 = 0, recall@2 = 1/1.
    labels = [None, "E"]
    assert recall_at_k(labels, relevant_count=1, k=1) == 0.0
    assert recall_at_k(labels, relevant_count=1, k=2) == 1.0


def test_recall_only_primary_labels_count_as_hits() -> None:
    # C and I are not primary-relevant, so they are never recall hits.
    assert recall_at_k(["C", "I"], relevant_count=1, k=2) == 0.0


def test_recall_zero_relevant_query_returns_none() -> None:
    # Zero known relevant -> undefined recall -> None (excluded from means, not 0.0).
    assert recall_at_k(["E", "S"], relevant_count=0, k=2) is None


# ------------------------------------------------------------------------------- MRR


def test_mrr_condensed_rank_skips_unjudged_entries() -> None:
    # Raw order: unjudged, unjudged, S. Condensed rank of S = 1 -> 1/1.
    assert reciprocal_rank_condensed([None, None, "S"]) == 1.0


def test_mrr_relevance_is_primary_only_and_denominator_is_condensed() -> None:
    # C and I are judged non-primary: they advance the condensed rank but are not hits.
    # Condensed ranks: C=1, I=2, S=3 -> first primary at 3 -> 1/3.
    assert reciprocal_rank_condensed(["C", "I", None, "S", "E"]) == pytest.approx(1 / 3)


def test_mrr_returns_zero_when_no_primary_retrieved() -> None:
    assert reciprocal_rank_condensed(["C", "I", None]) == 0.0
    assert reciprocal_rank_condensed([None, None]) == 0.0


# ------------------------------------------------------------------------------ NDCG


def test_condensed_ndcg_ignores_unknown_entries() -> None:
    retrieved = [None, 0, None, 3, 2]
    assert ndcg_condensed(retrieved, ideal_grades=[3, 2, 0], k=3) > 0
    assert ndcg_condensed([None, None], ideal_grades=[3], k=5) == 0


def test_ndcg_unjudged_removed_so_perfect_judged_order_scores_one() -> None:
    # Unjudged items interleaved with a perfectly ordered judged list -> NDCG 1.0.
    assert ndcg_condensed([None, 3, None, 2], ideal_grades=[3, 2], k=10) == pytest.approx(1.0)


def test_ndcg_gain_mapping_is_exponential_not_linear() -> None:
    # Retrieved order C(grade 1) then E(grade 3); ideal E then C.
    result = ndcg_condensed([1, 3], ideal_grades=[3, 1], k=10)
    log2_3 = math.log2(3)
    expected_exp = (1 + 7 / log2_3) / (7 + 1 / log2_3)  # gain = 2**g - 1
    expected_linear = (1 + 3 / log2_3) / (3 + 1 / log2_3)  # gain = g
    assert result == pytest.approx(expected_exp)
    assert result != pytest.approx(expected_linear)


def test_ndcg_top_k_truncates_both_condensed_and_ideal() -> None:
    # NDCG@1 sees only the first judged item on both sides -> 1.0.
    assert ndcg_condensed([3, 2, 3], ideal_grades=[3, 3, 2], k=1) == pytest.approx(1.0)
    # NDCG@2 exposes the sub-ideal ordering.
    log2_3 = math.log2(3)
    expected_at_2 = (7 + 3 / log2_3) / (7 + 7 / log2_3)
    at_2 = ndcg_condensed([3, 2, 3], ideal_grades=[3, 3, 2], k=2)
    assert at_2 == pytest.approx(expected_at_2)
    assert at_2 < 1.0
