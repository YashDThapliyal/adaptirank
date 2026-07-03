from pathlib import Path

from adaptirank.data.pipeline import load_esci_config


def test_canonical_and_large_variants_are_explicit() -> None:
    canonical = load_esci_config(Path("configs/data/esci_small_us.yaml"))
    large = load_esci_config(Path("configs/data/esci_large_us.yaml"))
    assert canonical.variant == "small"
    assert canonical.variant_column == "small_version"
    assert canonical.product_locale == "us"
    assert large.variant == "large"
    assert large.variant_column == "large_version"


def test_official_sample_is_integration_verification_with_full_source_urls() -> None:
    config = load_esci_config(Path("configs/data/esci_official_sample.yaml"))
    assert config.run.purpose == "integration_verification"
    assert config.sampling.max_train_queries == 300
    assert config.sampling.max_test_queries == 100
    assert config.sampling.background_products == 10_000
    assert all(item.url and config.source.revision in item.url for item in config.source.files)
