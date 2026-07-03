from pathlib import Path

from adaptirank.common.config import load_config
from adaptirank.retrieval.config import RetrievalConfig


def test_full_retrieval_config_is_uncapped_and_model_pinned() -> None:
    config = load_config(Path("configs/retrieval/full.yaml"), RetrievalConfig)
    assert config.max_queries_per_split is None
    assert config.top_k == (10, 50, 100, 500)
    assert config.dense.model_revision == "b207367332321f8e44f96e224ef15bc607f4dbf0"
    assert config.dense.device == "auto"
