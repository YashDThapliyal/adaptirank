"""ESCI ingestion, normalization, validation, fingerprinting, and reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.data.config import EsciConfig
from adaptirank.data.provenance import sha256_file
from adaptirank.data.schemas import (
    EXAMPLE_COLUMNS,
    PRODUCT_COLUMNS,
    SOURCE_COLUMNS,
    VALID_LABELS,
    VALID_SOURCE_SPLITS,
)
from adaptirank.data.splits import (
    assign_query_splits,
    select_query_groups,
    stable_key,
    validate_split_isolation,
)


@dataclass(frozen=True)
class DatasetBuildResult:
    fingerprint: str
    dataset_dir: Path
    report: dict[str, Any]


def _read_table(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=10000)
    raise ValueError(f"unsupported source format: {path}")


def _require_columns(frame: pl.DataFrame, required: set[str], role: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{role} is missing required columns: {sorted(missing)}")


def _clean_text(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .replace("", None)
    )


def validate_source_tables(
    examples: pl.DataFrame, products: pl.DataFrame, sources: pl.DataFrame
) -> None:
    _require_columns(examples, EXAMPLE_COLUMNS, "examples")
    _require_columns(products, PRODUCT_COLUMNS, "products")
    _require_columns(sources, SOURCE_COLUMNS, "sources")
    if examples.get_column("example_id").n_unique() != examples.height:
        raise ValueError("duplicate example_id primary keys")
    duplicate_products = (
        products.group_by("product_locale", "product_id").len().filter(pl.col("len") > 1)
    )
    if duplicate_products.height:
        raise ValueError("duplicate (product_locale, product_id) source product keys")
    if sources.get_column("query_id").n_unique() != sources.height:
        raise ValueError("duplicate source query_id keys")
    labels = set(examples.get_column("esci_label").drop_nulls().cast(pl.String).to_list())
    if not labels.issubset(VALID_LABELS):
        raise ValueError(f"invalid ESCI labels: {sorted(labels.difference(VALID_LABELS))}")
    splits = set(examples.get_column("split").drop_nulls().cast(pl.String).to_list())
    if not splits.issubset(VALID_SOURCE_SPLITS):
        raise ValueError(f"invalid source splits: {sorted(splits.difference(VALID_SOURCE_SPLITS))}")
    for role, frame, columns in (
        ("examples", examples, ["example_id", "query_id", "product_id", "product_locale"]),
        ("products", products, ["product_id", "product_locale"]),
    ):
        if frame.select(pl.any_horizontal(pl.col(columns).is_null()).any()).item():
            raise ValueError(f"{role} contains missing required IDs")


def _normalize_raw(
    examples: pl.DataFrame, products: pl.DataFrame, sources: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    examples = examples.with_columns(
        pl.col("example_id").cast(pl.String),
        pl.col("query_id").cast(pl.String),
        pl.col("product_id").cast(pl.String),
        pl.col("product_locale").cast(pl.String).str.to_lowercase(),
        pl.col("esci_label").cast(pl.String),
        pl.col("small_version").cast(pl.Int8),
        pl.col("large_version").cast(pl.Int8),
        pl.col("split").cast(pl.String),
        _clean_text("query").alias("query"),
    )
    products = products.with_columns(
        pl.col("product_id").cast(pl.String),
        pl.col("product_locale").cast(pl.String).str.to_lowercase(),
        *[
            _clean_text(column).alias(column)
            for column in (
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
            )
        ],
    )
    sources = sources.with_columns(
        pl.col("query_id").cast(pl.String),
        _clean_text("source").alias("source"),
    )
    return examples, products, sources


def _internal_keys(frame: pl.DataFrame, kind: str, columns: list[str], output: str) -> pl.Expr:
    return (
        pl.struct(columns)
        .map_elements(
            lambda row: stable_key(kind, *(str(row[column]) for column in columns)),
            return_dtype=pl.String,
        )
        .alias(output)
    )


def _query_consistency(examples: pl.DataFrame) -> None:
    inconsistent = (
        examples.group_by("product_locale", "query_id")
        .agg(pl.col("query").n_unique().alias("query_texts"), pl.col("split").n_unique())
        .filter((pl.col("query_texts") != 1) | (pl.col("split") != 1))
    )
    if inconsistent.height:
        raise ValueError("query group has inconsistent text or source split")


def _fingerprint(config: EsciConfig, source_manifest: dict[str, Any]) -> str:
    semantic = {
        "pipeline_schema": "adaptirank-esci-v1",
        "source_revision": config.source.revision,
        "source_files": [
            {
                "role": item["role"],
                "observed_size_bytes": item["observed_size_bytes"],
                "observed_sha256": item["observed_sha256"],
            }
            for item in source_manifest["files"]
        ],
        "variant": config.variant,
        "product_locale": config.product_locale,
        "validation_fraction": config.validation_fraction,
        "sampling": config.sampling.model_dump(mode="json"),
        "label_grades": config.label_grades,
        "seed": config.run.seed,
    }
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _distribution(frame: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        str(row[column]): int(row["len"])
        for row in frame.group_by(column).len().sort(column).iter_rows(named=True)
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ESCI Dataset Report",
        "",
        f"- Purpose: `{report['purpose']}`",
        f"- Variant: `{report['variant']}`",
        f"- Locale: `{report['product_locale']}`",
        f"- Fingerprint: `{report['dataset_fingerprint']}`",
        f"- Products: {report['counts']['products']}",
        f"- Queries: {report['counts']['queries']}",
        f"- Judgments: {report['counts']['judgments']}",
        f"- Catalog coverage: {report['catalog_coverage']:.6f}",
        "",
        "## Split distribution",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in report["split_distribution"].items())
    lines.extend(["", "## Label distribution", ""])
    lines.extend(f"- {key}: {value}" for key, value in report["label_distribution"].items())
    lines.extend(
        [
            "",
            "Background catalog products are unjudged. Missing labels remain null and are never",
            "interpreted as ESCI `I` or relevance grade 0.",
            "",
        ]
    )
    return "\n".join(lines)


def build_dataset(
    config: EsciConfig,
    paths: dict[str, Path],
    source_manifest: dict[str, Any],
) -> DatasetBuildResult:
    """Build validated normalized ESCI artifacts from staged source files."""

    examples = _read_table(paths["examples"])
    products = _read_table(paths["products"])
    sources = _read_table(paths["sources"])
    validate_source_tables(examples, products, sources)
    examples, products, sources = _normalize_raw(examples, products, sources)

    examples = examples.filter(
        (pl.col("product_locale") == config.product_locale) & (pl.col(config.variant_column) == 1)
    )
    if not examples.height:
        raise ValueError("canonical ESCI filter produced no examples")
    _query_consistency(examples)
    examples = select_query_groups(
        examples,
        seed=config.run.seed,
        max_train=config.sampling.max_train_queries,
        max_test=config.sampling.max_test_queries,
    )
    splits = assign_query_splits(
        examples,
        seed=config.run.seed,
        validation_fraction=config.validation_fraction,
    )
    validate_split_isolation(splits)
    examples = examples.join(
        splits.select("product_locale", "query_id", "benchmark_split", "query_key"),
        on=["product_locale", "query_id"],
        how="inner",
    )

    relevance = (
        examples.join(sources, on="query_id", how="left")
        .with_columns(
            _internal_keys(examples, "product", ["product_locale", "product_id"], "product_key"),
            pl.col("esci_label")
            .replace_strict(config.label_grades, return_dtype=pl.Int8)
            .alias("relevance_grade"),
            pl.lit("judged").alias("judgment_status"),
        )
        .select(
            "example_id",
            "product_locale",
            "query_id",
            "query_key",
            "product_id",
            "product_key",
            "query",
            "esci_label",
            "relevance_grade",
            "judgment_status",
            "small_version",
            "large_version",
            pl.col("split").alias("source_split"),
            "benchmark_split",
            "source",
        )
        .sort("product_locale", "query_id", "example_id")
    )

    locale_products = products.filter(pl.col("product_locale") == config.product_locale)
    judged_keys = relevance.select("product_locale", "product_id").unique()
    judged_products = locale_products.join(
        judged_keys, on=["product_locale", "product_id"], how="inner"
    )
    if judged_products.height != judged_keys.height:
        raise ValueError("one or more judged products are absent from the product catalog")
    if config.sampling.background_products is None:
        catalog = locale_products
    else:
        background = (
            locale_products.join(judged_keys, on=["product_locale", "product_id"], how="anti")
            .sort("product_locale", "product_id")
            .head(config.sampling.background_products)
        )
        catalog = pl.concat([judged_products, background], how="vertical").unique(
            ["product_locale", "product_id"]
        )
    catalog = (
        catalog.with_columns(
            _internal_keys(catalog, "product", ["product_locale", "product_id"], "product_key")
        )
        .select(
            "product_locale",
            "product_id",
            "product_key",
            pl.col("product_title").alias("title"),
            pl.col("product_description").alias("description"),
            pl.col("product_brand").alias("brand"),
            pl.lit(None, dtype=pl.String).alias("category"),
            "product_bullet_point",
            "product_color",
        )
        .sort("product_locale", "product_id")
    )

    queries = (
        relevance.select(
            "product_locale",
            "query_id",
            "query_key",
            pl.col("query").alias("query_text"),
            "source_split",
            "benchmark_split",
            "source",
        )
        .unique()
        .sort("product_locale", "query_id")
    )
    catalog_keys = catalog.select("product_locale", "product_id")
    covered = relevance.join(catalog_keys, on=["product_locale", "product_id"], how="inner").height
    coverage = covered / relevance.height
    if coverage != 1.0:
        raise ValueError(f"catalog coverage must be 1.0, observed {coverage}")

    fingerprint = _fingerprint(config, source_manifest)
    root = project_root()
    dataset_dir = resolve_project_path(config.processed_dir, root) / fingerprint
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "catalog": dataset_dir / "catalog.parquet",
        "queries": dataset_dir / "queries.parquet",
        "relevance": dataset_dir / "relevance.parquet",
        "query_splits": dataset_dir / "query_splits.parquet",
    }
    for name, frame in (
        ("catalog", catalog),
        ("queries", queries),
        ("relevance", relevance),
        ("query_splits", splits),
    ):
        _atomic_parquet(frame, table_paths[name])

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": config.run.purpose,
        "scientific_result_eligible": config.run.purpose == "scientific_benchmark",
        "variant": config.variant,
        "variant_filter": f"{config.variant_column} == 1",
        "product_locale": config.product_locale,
        "dataset_fingerprint": fingerprint,
        "counts": {
            "products": catalog.height,
            "queries": queries.height,
            "judgments": relevance.height,
            "unjudged_background_products": catalog.height - judged_products.height,
        },
        "split_distribution": _distribution(queries, "benchmark_split"),
        "label_distribution": _distribution(relevance, "esci_label"),
        "catalog_coverage": coverage,
        "judgment_semantics": {
            "missing_label": "unknown/unjudged",
            "missing_label_is_irrelevant": False,
            "raw_label_preserved": True,
            "numeric_grade_is_derived": True,
        },
    }
    manifest = {
        "dataset_fingerprint": fingerprint,
        "pipeline_schema": "adaptirank-esci-v1",
        "source_provenance": source_manifest,
        "config": config.model_dump(mode="json"),
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "observed_sha256": sha256_file(path),
            }
            for name, path in table_paths.items()
        },
    }
    _atomic_json(dataset_dir / "manifest.json", manifest)
    _atomic_json(dataset_dir / "dataset_report.json", report)
    report_path = dataset_dir / "dataset_report.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text(_report_markdown(report))
    temporary_report.replace(report_path)
    return DatasetBuildResult(fingerprint, dataset_dir, report)
