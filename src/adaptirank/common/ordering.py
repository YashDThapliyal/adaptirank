"""Deterministic ranking and canonical ordering utilities."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import polars as pl


def verify_pair_uniqueness(
    frame: pl.DataFrame,
    *,
    query_col: str = "query_key",
    product_col: str = "product_key",
) -> None:
    """Require non-null product keys and unique (query_key, product_key) pairs."""

    null_products = frame.filter(pl.col(product_col).is_null()).height
    if null_products:
        raise ValueError(f"{product_col} must be non-null; found {null_products} null rows")
    duplicate_pairs = frame.height - frame.unique([query_col, product_col]).height
    if duplicate_pairs:
        raise ValueError(
            f"({query_col}, {product_col}) pairs must be unique; found {duplicate_pairs} duplicates"
        )


def assign_deterministic_rank(
    frame: pl.DataFrame,
    *,
    score_col: str,
    rank_col: str = "rank",
    group_col: str = "query_key",
    tie_col: str = "product_key",
    descending: bool = True,
) -> pl.DataFrame:
    """Assign ordinal ranks with canonical tie-breaking: score DESC, product_key ASC."""

    verify_pair_uniqueness(frame, query_col=group_col, product_col=tie_col)
    return (
        frame.sort(
            [group_col, score_col, tie_col],
            descending=[False, descending, False],
            nulls_last=True,
        )
        .with_columns(pl.col(tie_col).cum_count().over(group_col).cast(pl.Int32).alias(rank_col))
        .sort(group_col, rank_col, tie_col)
    )


def canonical_content_fingerprint(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    sort_columns: tuple[str, ...] = ("query_key", "product_key"),
) -> str:
    """Hash logically ordered row content for reproducibility checks."""

    ordered = frame.select(columns).sort(*sort_columns)
    buffer = io.BytesIO()
    ordered.write_csv(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()
