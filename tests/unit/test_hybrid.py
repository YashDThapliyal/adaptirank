import polars as pl

from adaptirank.retrieval.hybrid import (
    candidate_contract,
    reciprocal_rank_fusion,
    select_validation_alpha,
    weighted_fusion,
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


def test_weighted_and_rrf_fusion_return_ranked_union() -> None:
    bm25 = _candidates("bm25", ["a", "b"], [2.0, 1.0])
    dense = _candidates("dense", ["b", "c"], [0.9, 0.8])
    weighted, _ = weighted_fusion(bm25, dense, alpha=0.5, top_k=3)
    rrf, _ = reciprocal_rank_fusion(bm25, dense, rrf_k=60, top_k=3)
    assert set(weighted.get_column("product_key")) == {"a", "b", "c"}
    assert set(rrf.get_column("product_key")) == {"a", "b", "c"}
    assert sorted(weighted.get_column("rank")) == [1, 2, 3]


def test_alpha_selection_uses_validation_objective_only() -> None:
    metrics = {
        0.0: {"recall_primary_100": 0.5, "ndcg_10": 0.8},
        0.5: {"recall_primary_100": 0.7, "ndcg_10": 0.6},
        1.0: {"recall_primary_100": 0.7, "ndcg_10": 0.5},
    }
    assert select_validation_alpha(metrics) == 0.5


def test_candidate_contract_preserves_unjudged_nulls() -> None:
    bm25 = _candidates("bm25", ["a", "b"], [2.0, 1.0])
    dense = _candidates("dense", ["b", "c"], [0.9, 0.8])
    weighted, _ = weighted_fusion(bm25, dense, alpha=0.5, top_k=3)
    rrf, _ = reciprocal_rank_fusion(bm25, dense, rrf_k=60, top_k=3)
    relevance = pl.DataFrame(
        {
            "query_key": ["q"],
            "product_key": ["a"],
            "esci_label": ["E"],
            "relevance_grade": [3],
        }
    )
    contract = candidate_contract(bm25, dense, weighted, rrf, relevance)
    unjudged = contract.filter(pl.col("product_key") == "c").row(0, named=True)
    assert unjudged["judgment_status"] == "unjudged"
    assert unjudged["esci_label"] is None
    assert unjudged["relevance_grade"] is None
