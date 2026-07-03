from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from adaptirank.retrieval.dense import DenseRetriever, select_device


class _FakeModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 4


class _DeterministicDenseRetriever(DenseRetriever):
    def _load_model(self) -> Any:
        self.model = _FakeModel()
        return self.model

    def _encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    lowered.count("trail") + lowered.count("shoe"),
                    lowered.count("water") + lowered.count("bottle"),
                    lowered.count("phone") + lowered.count("charger"),
                    0.1,
                ],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.stack(rows)


def _retriever() -> _DeterministicDenseRetriever:
    return _DeterministicDenseRetriever(
        model_name="test-only-deterministic-encoder",
        model_revision="test",
        fields=("title", "description", "brand"),
        batch_size=2,
        outer_batch_size=2,
        device="cpu",
        nlist=2,
        nprobe=1,
        training_sample_size=3,
    )


def test_dense_faiss_artifacts_and_cpu_retrieval(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    pl.DataFrame(
        {
            "product_key": ["shoe", "bottle", "charger"],
            "title": ["trail shoe", "water bottle", "phone charger"],
            "description": ["running", "steel", "usb"],
            "brand": ["north", "aqua", "volt"],
        }
    ).write_parquet(catalog_path)
    queries = pl.DataFrame(
        {
            "query_key": ["q1"],
            "query_text": ["trail shoe"],
            "benchmark_split": ["test"],
        }
    )
    retriever = _retriever()
    stats = retriever.build(catalog_path, tmp_path / "dense")
    result = retriever.retrieve(queries, top_k=3)
    assert result.candidates.sort("rank").row(0, named=True)["product_key"] == "shoe"
    assert stats.metadata["index_type"] == "IndexFlatIP"
    assert Path(stats.artifact_paths["embeddings"]).is_file()
    assert Path(stats.artifact_paths["index"]).is_file()

    reloaded = _retriever()
    loaded = reloaded.build(catalog_path, tmp_path / "dense")
    assert loaded.build_seconds == 0
    assert reloaded.retrieve(queries, top_k=3).candidates.height == 3


def test_explicit_cpu_device_never_requires_gpu() -> None:
    assert select_device("cpu") == "cpu"
