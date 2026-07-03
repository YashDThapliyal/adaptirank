"""Aggregate-level audit of judged-aware evaluation semantics.

A hand-built three-query fixture exercises the split-level means so that the difference
between MRR eligibility (all queries, zero contributes 0.0) and recall eligibility
(zero-relevant queries excluded) is pinned, alongside primary vs sensitivity recall and
unjudged handling.
"""

from typing import Any

import polars as pl
import pytest

from adaptirank.retrieval.base import IndexBuildStats, RetrievalResult
from adaptirank.retrieval.evaluate import evaluate_result

# Catalog: p1..p6. Titles are only used by lexical-overlap slicing, not by the metrics.
_CATALOG = pl.DataFrame(
    {
        "product_key": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "title": ["one", "two", "three", "four", "five", "six"],
    }
)

# Judgments. p5 is a known-relevant item for qA that is never retrieved.
_RELEVANCE = pl.DataFrame(
    {
        "query_key": ["qA", "qA", "qA", "qA", "qA", "qB", "qB", "qV", "qV"],
        "product_key": ["p1", "p2", "p3", "p4", "p5", "p3", "p4", "p1", "p2"],
        "esci_label": ["E", "S", "C", "I", "E", "C", "I", "E", "S"],
        "relevance_grade": [3, 2, 1, 0, 3, 1, 0, 3, 2],
        "query": [
            "qa text",
            "qa text",
            "qa text",
            "qa text",
            "qa text",
            "qb text",
            "qb text",
            "qv text",
            "qv text",
        ],
    }
)

_QUERIES = pl.DataFrame(
    {
        "query_key": ["qA", "qB", "qV"],
        "query_text": ["qa text", "qb text", "qv text"],
        "benchmark_split": ["test", "test", "validation"],
    }
)

# Retrieved candidates (ranked). qA interleaves an unjudged item (p6) at rank 1.
_CANDIDATES = pl.DataFrame(
    {
        "query_key": ["qA", "qA", "qA", "qA", "qA", "qB", "qB", "qV", "qV"],
        "product_key": ["p6", "p3", "p1", "p2", "p4", "p3", "p6", "p1", "p2"],
        "rank": [1, 2, 3, 4, 5, 1, 2, 1, 2],
        "score": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }
)

_LATENCIES = pl.DataFrame({"query_key": ["qA", "qB", "qV"], "latency_ms": [1.0, 1.0, 1.0]})


def _evaluate() -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    result = RetrievalResult(
        method="fixture",
        candidates=_CANDIDATES,
        query_latencies_ms=_LATENCIES,
        build_stats=IndexBuildStats(0.0, 0, _CATALOG.height, {}, {}),
    )
    return evaluate_result(
        result,
        queries=_QUERIES,
        relevance=_RELEVANCE,
        catalog=_CATALOG,
        top_k_values=(2, 4),
    )


def test_unjudged_candidate_is_labelled_unjudged() -> None:
    annotated, _, _ = _evaluate()
    p6 = annotated.filter((pl.col("query_key") == "qA") & (pl.col("product_key") == "p6")).row(
        0, named=True
    )
    assert p6["judgment_status"] == "unjudged"
    assert p6["esci_label"] is None


def test_zero_relevant_query_has_null_recall_but_scores_mrr() -> None:
    _, per_query, _ = _evaluate()
    qb = per_query.filter(pl.col("query_key") == "qB").row(0, named=True)
    assert qb["primary_relevant_count"] == 0
    # Recall is undefined (None) for a query with no primary-relevant judgments...
    assert qb["recall_primary_2"] is None
    assert qb["recall_primary_4"] is None
    # ...but MRR is defined and is 0.0 (no primary item was retrieved).
    assert qb["mrr"] == 0.0
    # Sensitivity recall is defined because C counts: p3 (C) retrieved at rank 1 of 1.
    assert qb["sensitivity_relevant_count"] == 1
    assert qb["recall_sensitivity_2"] == pytest.approx(1.0)


def test_split_level_recall_primary_excludes_zero_relevant_query() -> None:
    _, _, aggregate = _evaluate()
    test_split = aggregate["by_split"]["test"]
    assert test_split["queries"] == 2  # qA and qB are both in the test split
    # qA: unjudged p6 pushes p1/p2 to ranks 3/4, so nothing primary in top-2 -> 0/3.
    # qB: excluded (None). Mean over the one eligible query = 0.0.
    assert test_split["recall_primary_2"] == pytest.approx(0.0)
    # qA top-4 has p1 and p2 -> 2/3; qB excluded. Mean = 2/3.
    assert test_split["recall_primary_4"] == pytest.approx(2 / 3)


def test_split_level_sensitivity_recall_counts_complement() -> None:
    _, _, aggregate = _evaluate()
    test_split = aggregate["by_split"]["test"]
    # qA sensitivity@2: p3 (C) at rank 2 -> 1/4. qB sensitivity@2: 1/1. Mean = 0.625.
    assert test_split["recall_sensitivity_2"] == pytest.approx(0.625)
    # qA sensitivity@4: p3,p1,p2 -> 3/4. qB: 1/1. Mean = 0.875.
    assert test_split["recall_sensitivity_4"] == pytest.approx(0.875)


def test_split_level_mrr_includes_zero_relevant_query_as_zero() -> None:
    _, _, aggregate = _evaluate()
    test_split = aggregate["by_split"]["test"]
    # qA condensed: C,E,S,I -> first primary (E) at condensed rank 2 -> 0.5.
    # qB -> 0.0. MRR denominator is ALL test queries (2), so mean = 0.25.
    assert test_split["mrr"] == pytest.approx(0.25)


def test_splits_are_isolated() -> None:
    _, _, aggregate = _evaluate()
    assert set(aggregate["by_split"]) == {"test", "validation"}
    validation = aggregate["by_split"]["validation"]
    assert validation["queries"] == 1
    # qV retrieves p1(E),p2(S) in perfect order -> all metrics 1.0.
    assert validation["mrr"] == pytest.approx(1.0)
    assert validation["recall_primary_2"] == pytest.approx(1.0)
    assert validation["ndcg_10"] == pytest.approx(1.0)
