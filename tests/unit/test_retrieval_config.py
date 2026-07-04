from pathlib import Path

from adaptirank.common.config import load_config
from adaptirank.retrieval.config import RetrievalConfig


def test_full_retrieval_config_is_uncapped_and_model_pinned() -> None:
    config = load_config(Path("configs/retrieval/full.yaml"), RetrievalConfig)
    assert config.max_queries_per_split is None
    assert config.top_k == (10, 50, 100, 500)
    assert config.dense.model_revision == "b207367332321f8e44f96e224ef15bc607f4dbf0"
    assert config.dense.device == "auto"


def test_m3_three_split_config_covers_train_and_isolates_artifacts() -> None:
    m3 = load_config(Path("configs/retrieval/m3_three_split.yaml"), RetrievalConfig)
    full = load_config(Path("configs/retrieval/full.yaml"), RetrievalConfig)
    # M3 handoff adds the train split for model fitting; validation/test roles preserved.
    assert m3.evaluation_splits == ("train", "validation", "test")
    # Distinct artifact name so the canonical M2 `full_scientific` artifacts are never overwritten.
    assert m3.artifact_name == "m3_three_split" != full.artifact_name
    # Identical retrieval methodology to the M2 canonical run.
    assert m3.dataset_fingerprint == full.dataset_fingerprint
    assert m3.dense.model_name == full.dense.model_name
    assert m3.dense.model_revision == full.dense.model_revision
    assert m3.dense.fields == full.dense.fields
    assert (m3.dense.nlist, m3.dense.nprobe) == (full.dense.nlist, full.dense.nprobe)
    assert m3.dense.training_sample_size == full.dense.training_sample_size
    assert m3.top_k == full.top_k
    assert m3.hybrid.alpha_grid == full.hybrid.alpha_grid
    assert m3.hybrid.rrf_k == full.hybrid.rrf_k
    m3_fields = tuple(fs.name for fs in m3.bm25.field_sets)
    assert m3_fields == tuple(fs.name for fs in full.bm25.field_sets)
