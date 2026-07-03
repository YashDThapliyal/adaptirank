import json
from pathlib import Path

import polars as pl

from adaptirank.data.config import EsciConfig
from adaptirank.data.esci import build_dataset
from adaptirank.data.pipeline import load_esci_config
from adaptirank.data.provenance import stage_sources


def _temporary_config(tmp_path: Path) -> EsciConfig:
    config = load_esci_config(Path("configs/data/esci_fixture.yaml"))
    source = config.source.model_copy(update={"raw_dir": tmp_path / "raw"})
    return config.model_copy(update={"source": source, "processed_dir": tmp_path / "processed"})


def test_fixture_pipeline_preserves_contract_and_exact_counts(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    paths, provenance = stage_sources(config)
    result = build_dataset(config, paths, provenance)

    assert result.report["counts"] == {
        "products": 20,
        "queries": 9,
        "judgments": 18,
        "unjudged_background_products": 2,
    }
    assert result.report["split_distribution"] == {
        "test": 3,
        "train": 4,
        "validation": 2,
    }
    assert result.report["label_distribution"] == {"C": 3, "E": 9, "I": 3, "S": 3}
    assert result.report["catalog_coverage"] == 1.0
    assert result.report["scientific_result_eligible"] is False

    relevance = pl.read_parquet(result.dataset_dir / "relevance.parquet")
    assert set(relevance.get_column("esci_label")) == {"E", "S", "C", "I"}
    assert set(relevance.get_column("small_version")) == {1}
    assert set(relevance.get_column("large_version")) == {1}
    assert set(relevance.get_column("judgment_status")) == {"judged"}
    assert relevance.filter(pl.col("esci_label") == "I").get_column(
        "relevance_grade"
    ).to_list() == [0, 0, 0]


def test_provenance_calls_local_hash_observed_not_authoritative(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    _, provenance = stage_sources(config)
    assert provenance["pinned_commit_sha"] == "fixture-v1"
    assert "locally generated" in provenance["checksum_note"]
    for record in provenance["files"]:
        assert len(record["observed_sha256"]) == 64
        assert record["checksum_verification"] == "authoritative_checksum_not_published"
        assert record["authoritative_sha256"] is None
    stored = json.loads((tmp_path / "raw" / "source_manifest.json").read_text())
    assert stored == provenance


def test_large_variant_is_not_selected_by_canonical_fixture(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    paths, provenance = stage_sources(config)
    small_result = build_dataset(config, paths, provenance)
    large_config = config.model_copy(update={"variant": "large"})
    large_result = build_dataset(large_config, paths, provenance)
    assert small_result.report["counts"]["queries"] == 9
    assert large_result.report["counts"]["queries"] == 10
