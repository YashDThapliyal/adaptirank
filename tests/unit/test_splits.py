import polars as pl
import pytest

from adaptirank.data.splits import assign_query_splits, stable_key, validate_split_isolation


def test_locale_is_part_of_query_identity_and_internal_key() -> None:
    examples = pl.DataFrame(
        {
            "product_locale": ["us", "es", "us", "es", "us", "es"],
            "query_id": ["1", "1", "2", "2", "3", "3"],
            "split": ["train", "train", "train", "train", "test", "test"],
        }
    )
    splits = assign_query_splits(examples, seed=42, validation_fraction=0.5)
    validate_split_isolation(splits)
    assert splits.height == 6
    assert stable_key("query", "us", "1") != stable_key("query", "es", "1")


def test_source_test_cannot_be_reassigned() -> None:
    splits = pl.DataFrame(
        {
            "product_locale": ["us", "us", "us"],
            "query_id": ["1", "2", "3"],
            "source_split": ["train", "train", "test"],
            "benchmark_split": ["train", "validation", "train"],
        }
    )
    with pytest.raises(ValueError, match="source test"):
        validate_split_isolation(splits)
