"""Label-free feature construction for M3 learned ranking."""

from __future__ import annotations

from typing import Any

import polars as pl

FEATURE_COLUMNS = (
    "bm25_score",
    "bm25_rank",
    "dense_score",
    "dense_rank",
    "hybrid_score",
    "hybrid_rank",
    "rrf_score",
    "rrf_rank",
    "query_length",
    "title_length",
    "lexical_overlap",
    "exact_token_overlap",
    "brand_match",
    "bm25_missing",
    "dense_missing",
)

FEATURE_DEFINITIONS: dict[str, str] = {
    "bm25_score": "raw selected title-only BM25 score; null when absent from BM25 top 500",
    "bm25_rank": "selected title-only BM25 rank; null when absent from BM25 top 500",
    "dense_score": "raw pinned bi-encoder cosine/IP score; null when absent from dense top 500",
    "dense_rank": "pinned bi-encoder rank; null when absent from dense top 500",
    "hybrid_score": "validation-selected weighted-fusion score (alpha frozen at 0.5)",
    "hybrid_rank": "validation-selected weighted-fusion rank",
    "rrf_score": "reciprocal-rank-fusion score with k=60",
    "rrf_rank": "reciprocal-rank-fusion rank",
    "query_length": "number of unique lower-cased alphanumeric query tokens",
    "title_length": "number of unique lower-cased alphanumeric product-title tokens",
    "lexical_overlap": "Jaccard similarity between unique query and title token sets",
    "exact_token_overlap": "count of exact tokens shared by query and product title",
    "brand_match": "1 when any lower-cased brand token occurs exactly in the query, else 0",
    "bm25_missing": "1 when the candidate is absent from BM25 top 500, else 0",
    "dense_missing": "1 when the candidate is absent from dense top 500, else 0",
}


def _tokens(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .fill_null("")
        .str.to_lowercase()
        .str.extract_all(r"[[:alnum:]_]+")
        .list.unique()
    )


def feature_frame(
    contract: pl.LazyFrame,
    queries: pl.LazyFrame,
    catalog: pl.LazyFrame,
    *,
    split: str,
    rank_column: str = "hybrid_rank",
    top_m: int = 500,
) -> pl.LazyFrame:
    """Build one split without allowing labels into any derived feature."""

    query_features = queries.select("query_key", "query_text").with_columns(
        _tokens("query_text").alias("query_tokens")
    )
    product_features = catalog.select("product_key", "title", "brand").with_columns(
        _tokens("title").alias("title_tokens"),
        _tokens("brand").alias("brand_tokens"),
    )
    candidates = contract.filter(
        (pl.col("split") == split)
        & pl.col(rank_column).is_not_null()
        & (pl.col(rank_column) <= top_m)
    )
    joined = candidates.join(query_features, on="query_key", how="left").join(
        product_features, on="product_key", how="left"
    )
    joined = joined.with_columns(
        pl.col("query_tokens").list.set_intersection("title_tokens").alias("shared_tokens"),
        pl.col("query_tokens").list.set_union("title_tokens").alias("union_tokens"),
        pl.col("query_tokens").list.set_intersection("brand_tokens").alias("shared_brand_tokens"),
    )
    return joined.with_columns(
        pl.col("query_tokens").list.len().cast(pl.Int16).alias("query_length"),
        pl.col("title_tokens").list.len().cast(pl.Int16).alias("title_length"),
        pl.when(pl.col("union_tokens").list.len() > 0)
        .then(pl.col("shared_tokens").list.len() / pl.col("union_tokens").list.len())
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("lexical_overlap"),
        pl.col("shared_tokens").list.len().cast(pl.Int16).alias("exact_token_overlap"),
        (pl.col("shared_brand_tokens").list.len() > 0).cast(pl.Int8).alias("brand_match"),
        pl.col("bm25_score").is_null().cast(pl.Int8).alias("bm25_missing"),
        pl.col("dense_score").is_null().cast(pl.Int8).alias("dense_missing"),
    ).select(
        "query_key",
        "product_key",
        "split",
        *FEATURE_COLUMNS,
        "esci_label",
        "relevance_grade",
        "judgment_status",
    )


def feature_schema() -> dict[str, Any]:
    return {
        "feature_columns": list(FEATURE_COLUMNS),
        "definitions": FEATURE_DEFINITIONS,
        "label_columns": ["esci_label", "relevance_grade", "judgment_status"],
        "category_excluded": "canonical catalog coverage is 0%",
        "cross_encoder_excluded": "primary LambdaMART is label-free and CE-independent",
    }
