import polars as pl

from adaptirank.data.judgments import attach_judgments


def test_unjudged_candidates_remain_null_not_irrelevant() -> None:
    candidates = pl.DataFrame(
        {
            "product_locale": ["us", "us"],
            "query_id": ["q1", "q1"],
            "product_id": ["p1", "background"],
        }
    )
    relevance = pl.DataFrame(
        {
            "product_locale": ["us"],
            "query_id": ["q1"],
            "product_id": ["p1"],
            "esci_label": ["E"],
            "relevance_grade": [3],
        }
    )
    result = attach_judgments(candidates, relevance).sort("product_id")
    background = result.filter(pl.col("product_id") == "background").row(0, named=True)
    assert background["judgment_status"] == "unjudged"
    assert background["esci_label"] is None
    assert background["relevance_grade"] is None
