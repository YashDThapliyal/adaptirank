import polars as pl

from adaptirank.ranking.features import FEATURE_COLUMNS, feature_frame


def test_feature_frame_is_label_free_and_preserves_unjudged() -> None:
    contract = pl.DataFrame(
        {
            "query_key": ["q"],
            "product_key": ["p"],
            "split": ["train"],
            "bm25_score": [None],
            "bm25_rank": [None],
            "dense_score": [0.8],
            "dense_rank": [1],
            "hybrid_score": [0.7],
            "hybrid_rank": [1],
            "rrf_score": [0.1],
            "rrf_rank": [1],
            "esci_label": [None],
            "relevance_grade": [None],
            "judgment_status": ["unjudged"],
        }
    ).lazy()
    queries = pl.DataFrame({"query_key": ["q"], "query_text": ["Acme trail shoe"]}).lazy()
    catalog = pl.DataFrame(
        {"product_key": ["p"], "title": ["Acme road shoe"], "brand": ["Acme"]}
    ).lazy()
    result = feature_frame(contract, queries, catalog, split="train").collect().to_dicts()[0]
    assert result["exact_token_overlap"] == 2
    assert result["brand_match"] == 1
    assert result["bm25_missing"] == 1
    assert result["judgment_status"] == "unjudged"
    assert "relevance_grade" not in FEATURE_COLUMNS
