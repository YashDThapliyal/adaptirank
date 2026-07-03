"""ESCI source and normalized schema constants."""

from __future__ import annotations

EXAMPLE_COLUMNS = {
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "split",
}

PRODUCT_COLUMNS = {
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "product_locale",
}

SOURCE_COLUMNS = {"query_id", "source"}
VALID_LABELS = {"E", "S", "C", "I"}
VALID_SOURCE_SPLITS = {"train", "test"}
