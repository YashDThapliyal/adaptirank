import polars as pl
import pytest

from adaptirank.data.esci import validate_source_tables


def _valid_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    examples = pl.DataFrame(
        {
            "example_id": [1],
            "query": ["query"],
            "query_id": [1],
            "product_id": ["P1"],
            "product_locale": ["us"],
            "esci_label": ["E"],
            "small_version": [1],
            "large_version": [1],
            "split": ["train"],
        }
    )
    products = pl.DataFrame(
        {
            "product_id": ["P1"],
            "product_title": ["Product"],
            "product_description": [None],
            "product_bullet_point": [None],
            "product_brand": [None],
            "product_color": [None],
            "product_locale": ["us"],
        }
    )
    sources = pl.DataFrame({"query_id": [1], "source": ["test"]})
    return examples, products, sources


def test_invalid_label_is_rejected() -> None:
    examples, products, sources = _valid_tables()
    examples = examples.with_columns(pl.lit("X").alias("esci_label"))
    with pytest.raises(ValueError, match="invalid ESCI labels"):
        validate_source_tables(examples, products, sources)


def test_duplicate_locale_product_key_is_rejected() -> None:
    examples, products, sources = _valid_tables()
    products = pl.concat([products, products])
    with pytest.raises(ValueError, match="source product keys"):
        validate_source_tables(examples, products, sources)


def test_missing_required_id_is_rejected() -> None:
    examples, products, sources = _valid_tables()
    examples = examples.with_columns(pl.lit(None).alias("product_id"))
    with pytest.raises(ValueError, match="missing required IDs"):
        validate_source_tables(examples, products, sources)
