"""Judgment attachment with explicit unknown/unjudged semantics."""

from __future__ import annotations

import polars as pl


def attach_judgments(candidates: pl.DataFrame, relevance: pl.DataFrame) -> pl.DataFrame:
    """Left-join judgments without converting unknown labels into negatives."""

    keys = ["product_locale", "query_id", "product_id"]
    required = set(keys)
    if not required.issubset(candidates.columns):
        raise ValueError("candidates lack locale-aware query-product keys")
    judgment_columns = [*keys, "esci_label", "relevance_grade"]
    joined = candidates.join(relevance.select(judgment_columns), on=keys, how="left")
    return joined.with_columns(
        pl.when(pl.col("esci_label").is_null())
        .then(pl.lit("unjudged"))
        .otherwise(pl.lit("judged"))
        .alias("judgment_status")
    )
