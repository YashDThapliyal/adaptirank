"""Construction and validation of the shared CE-A/CE-B scoring union."""

from __future__ import annotations

from typing import Any

import polars as pl


def build_ce_union(
    hybrid: pl.DataFrame,
    lambdamart: pl.DataFrame,
    *,
    hybrid_top_m: int = 100,
    lambdamart_top_m: int = 50,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    hybrid_pairs = (
        hybrid.filter(pl.col("rank") <= hybrid_top_m)
        .select(
            "query_key",
            "product_key",
            "split",
            pl.col("rank").alias("hybrid_rank"),
        )
        .unique(["query_key", "product_key"])
    )
    lambda_pairs = (
        lambdamart.filter(pl.col("lambdamart_rank") <= lambdamart_top_m)
        .select("query_key", "product_key", "split", "lambdamart_rank")
        .unique(["query_key", "product_key"])
    )
    union = (
        hybrid_pairs.join(
            lambda_pairs,
            on=["query_key", "product_key"],
            how="full",
            suffix="_lambda",
            coalesce=True,
        )
        .with_columns(
            pl.coalesce("split", "split_lambda").alias("split"),
            pl.col("hybrid_rank").is_not_null().alias("in_hybrid_top_100"),
            pl.col("lambdamart_rank").is_not_null().alias("in_lambdamart_top_50"),
        )
        .drop("split_lambda")
        .sort("split", "query_key", "product_key")
    )
    missing_lambda = lambda_pairs.join(
        union.select("query_key", "product_key"),
        on=["query_key", "product_key"],
        how="anti",
    ).height
    duplicate_pairs = union.height - union.unique(["query_key", "product_key"]).height
    if missing_lambda or duplicate_pairs:
        raise ValueError(
            f"invalid CE union: missing_lambda={missing_lambda}, duplicate_pairs={duplicate_pairs}"
        )
    stats = {
        "hybrid_top_m": hybrid_top_m,
        "lambdamart_top_m": lambdamart_top_m,
        "hybrid_pairs": hybrid_pairs.height,
        "lambdamart_pairs": lambda_pairs.height,
        "union_pairs": union.height,
        "overlap_pairs": union.filter(
            pl.col("in_hybrid_top_100") & pl.col("in_lambdamart_top_50")
        ).height,
        "lambda_pairs_missing_from_union": missing_lambda,
        "duplicate_pairs": duplicate_pairs,
        "by_split": union.group_by("split")
        .agg(
            pl.len().alias("pairs"),
            pl.col("query_key").n_unique().alias("queries"),
            pl.col("in_hybrid_top_100").sum().alias("hybrid_pairs"),
            pl.col("in_lambdamart_top_50").sum().alias("lambdamart_pairs"),
        )
        .sort("split")
        .to_dicts(),
    }
    return union, stats
