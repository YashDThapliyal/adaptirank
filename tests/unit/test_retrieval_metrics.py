from adaptirank.retrieval.evaluate import (
    ndcg_condensed,
    recall_at_k,
    reciprocal_rank_condensed,
)


def test_unjudged_is_ignored_not_converted_to_irrelevant() -> None:
    labels = [None, "I", None, "E", "S"]
    assert recall_at_k(labels, relevant_count=2, k=4) == 0.5
    assert reciprocal_rank_condensed(labels) == 0.5


def test_condensed_ndcg_ignores_unknown_entries() -> None:
    retrieved = [None, 0, None, 3, 2]
    assert ndcg_condensed(retrieved, ideal_grades=[3, 2, 0], k=3) > 0
    assert ndcg_condensed([None, None], ideal_grades=[3], k=5) == 0
