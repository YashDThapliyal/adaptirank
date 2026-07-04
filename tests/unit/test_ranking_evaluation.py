import polars as pl

from adaptirank.ranking.evaluate import average_precision_condensed, ranked_candidates


def test_average_precision_condensed_counts_missed_known_relevant() -> None:
    assert average_precision_condensed(["I", "E", "S"], relevant_count=4) == (0.5 + 2 / 3) / 4


def test_ranked_candidates_uses_deterministic_tie_break() -> None:
    frame = pl.DataFrame(
        {
            "query_key": ["q", "q"],
            "product_key": ["b", "a"],
            "split": ["test", "test"],
            "hybrid_rank": [2, 1],
            "model_score": [1.0, 1.0],
        }
    )
    ranked = ranked_candidates(frame, "model_score", "model_rank")
    assert ranked.get_column("product_key").to_list() == ["a", "b"]
    assert ranked.get_column("rank").to_list() == [1, 2]
