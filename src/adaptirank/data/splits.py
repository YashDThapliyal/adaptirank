"""Locale-aware deterministic query-group splitting."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import polars as pl

QueryKey = tuple[str, str]


def stable_key(kind: str, *parts: str) -> str:
    """Create a stable internal key while keeping raw IDs in normalized tables."""

    payload = "\0".join((kind, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def _ordered(keys: Iterable[QueryKey], seed: int) -> list[QueryKey]:
    return sorted(
        set(keys),
        key=lambda key: stable_key("sample", str(seed), key[0], key[1]),
    )


def select_query_groups(
    examples: pl.DataFrame,
    *,
    seed: int,
    max_train: int | None,
    max_test: int | None,
) -> pl.DataFrame:
    """Select complete source query groups without row-level sampling."""

    selected: set[QueryKey] = set()
    for source_split, limit in (("train", max_train), ("test", max_test)):
        keys = [
            (str(row["product_locale"]), str(row["query_id"]))
            for row in examples.filter(pl.col("split") == source_split)
            .select("product_locale", "query_id")
            .unique()
            .iter_rows(named=True)
        ]
        ordered = _ordered(keys, seed)
        selected.update(ordered if limit is None else ordered[:limit])
    key_frame = pl.DataFrame(
        {
            "product_locale": [item[0] for item in selected],
            "query_id": [item[1] for item in selected],
        }
    )
    return examples.join(key_frame, on=["product_locale", "query_id"], how="inner")


def assign_query_splits(
    examples: pl.DataFrame, *, seed: int, validation_fraction: float
) -> pl.DataFrame:
    """Preserve source test and split only source-train locale-aware query groups."""

    groups = examples.select("product_locale", "query_id", "split").unique()
    train_keys = [
        (str(row["product_locale"]), str(row["query_id"]))
        for row in groups.filter(pl.col("split") == "train").iter_rows(named=True)
    ]
    ordered = _ordered(train_keys, seed)
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_keys = set(ordered[:validation_count])
    assignments: list[dict[str, str]] = []
    for row in groups.iter_rows(named=True):
        key = (str(row["product_locale"]), str(row["query_id"]))
        source_split = str(row["split"])
        benchmark_split = (
            "test"
            if source_split == "test"
            else "validation"
            if key in validation_keys
            else "train"
        )
        assignments.append(
            {
                "product_locale": key[0],
                "query_id": key[1],
                "source_split": source_split,
                "benchmark_split": benchmark_split,
                "query_key": stable_key("query", key[0], key[1]),
            }
        )
    return pl.DataFrame(assignments).sort("product_locale", "query_id")


def validate_split_isolation(splits: pl.DataFrame) -> None:
    """Reject duplicate locale-aware query keys or source-test reassignment."""

    if splits.select(pl.struct("product_locale", "query_id").n_unique()).item() != splits.height:
        raise ValueError("query groups overlap across benchmark splits")
    invalid_test = splits.filter(
        (pl.col("source_split") == "test") & (pl.col("benchmark_split") != "test")
    )
    if invalid_test.height:
        raise ValueError("source test query assigned outside benchmark test")
    observed = set(splits.get_column("benchmark_split").to_list())
    if observed != {"train", "validation", "test"}:
        raise ValueError(f"all benchmark splits must be non-empty; observed={sorted(observed)}")
