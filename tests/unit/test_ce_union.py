import polars as pl

from adaptirank.ranking.ce_union import build_ce_union


def test_ce_union_covers_lambda_pairs_below_hybrid_top_100() -> None:
    hybrid = pl.DataFrame(
        {
            "query_key": ["q", "q", "q"],
            "product_key": ["a", "b", "c"],
            "split": ["test", "test", "test"],
            "rank": [1, 100, 101],
        }
    )
    lambdamart = pl.DataFrame(
        {
            "query_key": ["q", "q"],
            "product_key": ["a", "c"],
            "split": ["test", "test"],
            "lambdamart_rank": [2, 1],
        }
    )
    union, stats = build_ce_union(hybrid, lambdamart)
    assert set(union.get_column("product_key")) == {"a", "b", "c"}
    assert stats["lambda_pairs_missing_from_union"] == 0
    assert stats["union_pairs"] == 3
