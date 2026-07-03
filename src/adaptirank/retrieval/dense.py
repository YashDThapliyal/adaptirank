"""Pretrained Sentence Transformers encoding with a persistent FAISS index."""

from __future__ import annotations

import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from adaptirank.retrieval.base import IndexBuildStats, RetrievalResult, Retriever, directory_size


def select_device(requested: str) -> str:
    """Resolve CUDA, Apple MPS, or CPU without requiring an accelerator."""

    if requested != "auto":
        return requested
    torch: Any = import_module("torch")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compose_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    return "\n".join(str(row[field]) for field in fields if row.get(field))


class DenseRetriever(Retriever):
    """Unfine-tuned dense bi-encoder baseline backed by FAISS."""

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str,
        fields: tuple[str, ...],
        batch_size: int,
        outer_batch_size: int,
        device: str,
        nlist: int,
        nprobe: int,
        training_sample_size: int,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.fields = fields
        self.batch_size = batch_size
        self.outer_batch_size = outer_batch_size
        self.device = select_device(device)
        self.nlist = nlist
        self.nprobe = nprobe
        self.training_sample_size = training_sample_size
        self.model: Any = None
        self.faiss_index: Any = None
        self.product_keys: list[str] = []
        self.stats: IndexBuildStats | None = None

    def _load_model(self) -> Any:
        if self.model is None:
            sentence_transformers: Any = import_module("sentence_transformers")
            self.model = sentence_transformers.SentenceTransformer(
                self.model_name,
                revision=self.model_revision,
                device=self.device,
            )
        return self.model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32, copy=False)

    def build(self, catalog_path: Path, artifact_dir: Path) -> IndexBuildStats:
        # Load PyTorch/Sentence Transformers before FAISS to avoid duplicate OpenMP runtime
        # initialization on macOS. The model is also required for cached-index query encoding.
        self._load_model()
        faiss: Any = import_module("faiss")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = artifact_dir / "product_embeddings.npy"
        keys_path = artifact_dir / "product_keys.parquet"
        index_path = artifact_dir / "faiss.index"
        metadata_path = artifact_dir / "index_metadata.json"
        if all(path.is_file() for path in (embeddings_path, keys_path, index_path, metadata_path)):
            metadata: dict[str, Any] = json.loads(metadata_path.read_text())
            self.product_keys = pl.read_parquet(keys_path).get_column("product_key").to_list()
            self.faiss_index = faiss.read_index(str(index_path))
            if hasattr(self.faiss_index, "nprobe"):
                self.faiss_index.nprobe = self.nprobe
            stats = IndexBuildStats(
                build_seconds=0.0,
                index_size_bytes=directory_size(artifact_dir),
                document_count=len(self.product_keys),
                artifact_paths={
                    "embeddings": str(embeddings_path),
                    "product_keys": str(keys_path),
                    "index": str(index_path),
                    "metadata": str(metadata_path),
                },
                metadata=metadata,
            )
            self.stats = stats
            return stats

        catalog = pl.read_parquet(catalog_path, columns=["product_key", *self.fields])
        self.product_keys = [str(value) for value in catalog.get_column("product_key")]
        pl.DataFrame(
            {"row_id": range(catalog.height), "product_key": self.product_keys}
        ).write_parquet(keys_path)
        model = self._load_model()
        dimension = int(model.get_sentence_embedding_dimension())
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(catalog.height, dimension),
        )
        encode_started = time.perf_counter()
        for start in range(0, catalog.height, self.outer_batch_size):
            batch = catalog.slice(start, self.outer_batch_size)
            texts = [_compose_text(row, self.fields) for row in batch.iter_rows(named=True)]
            embeddings[start : start + len(texts)] = self._encode(texts)
        embeddings.flush()
        embedding_seconds = time.perf_counter() - encode_started

        index_started = time.perf_counter()
        document_count = catalog.height
        effective_nlist = min(self.nlist, max(1, int(document_count**0.5)))
        if document_count < max(10_000, effective_nlist * 39):
            index = faiss.IndexFlatIP(dimension)
            index_type = "IndexFlatIP"
            effective_nlist = 0
        else:
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                effective_nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            sample_size = min(document_count, self.training_sample_size)
            sample_indices = np.linspace(0, document_count - 1, sample_size, dtype=np.int64)
            index.train(np.asarray(embeddings[sample_indices], dtype=np.float32))
            index.nprobe = min(self.nprobe, effective_nlist)
            index_type = "IndexIVFFlat"
        for start in range(0, document_count, self.outer_batch_size):
            index.add(
                np.asarray(embeddings[start : start + self.outer_batch_size], dtype=np.float32)
            )
        faiss.write_index(index, str(index_path))
        index_build_seconds = time.perf_counter() - index_started
        metadata = {
            "engine": "faiss",
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "index_type": index_type,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "fine_tuned": False,
            "device": self.device,
            "embedding_dimension": dimension,
            "embedding_fields": list(self.fields),
            "embedding_seconds": embedding_seconds,
            "index_build_seconds": index_build_seconds,
            "document_count": document_count,
            "nlist": effective_nlist,
            "nprobe": getattr(index, "nprobe", 0),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self.faiss_index = index
        stats = IndexBuildStats(
            build_seconds=embedding_seconds + index_build_seconds,
            index_size_bytes=directory_size(artifact_dir),
            document_count=document_count,
            artifact_paths={
                "embeddings": str(embeddings_path),
                "product_keys": str(keys_path),
                "index": str(index_path),
                "metadata": str(metadata_path),
            },
            metadata=metadata,
        )
        self.stats = stats
        return stats

    def retrieve(self, queries: pl.DataFrame, top_k: int) -> RetrievalResult:
        if self.faiss_index is None or self.stats is None:
            raise RuntimeError("dense index must be built before retrieval")
        candidates: list[dict[str, Any]] = []
        latencies: list[dict[str, Any]] = []
        query_rows = queries.to_dicts()
        for start in range(0, len(query_rows), self.outer_batch_size):
            batch = query_rows[start : start + self.outer_batch_size]
            encode_started = time.perf_counter()
            query_embeddings = self._encode([str(row["query_text"]) for row in batch])
            encode_per_query_ms = (time.perf_counter() - encode_started) * 1000 / len(batch)
            for row, embedding in zip(batch, query_embeddings, strict=True):
                search_started = time.perf_counter()
                scores, indices = self.faiss_index.search(embedding.reshape(1, -1), top_k)
                elapsed_ms = encode_per_query_ms + (time.perf_counter() - search_started) * 1000
                latencies.append({"query_key": row["query_key"], "latency_ms": elapsed_ms})
                for rank, (score, index_id) in enumerate(
                    zip(scores[0], indices[0], strict=True), start=1
                ):
                    if index_id < 0:
                        continue
                    candidates.append(
                        {
                            "query_key": row["query_key"],
                            "product_key": self.product_keys[int(index_id)],
                            "split": row["benchmark_split"],
                            "method": "dense",
                            "score": float(score),
                            "rank": rank,
                        }
                    )
        return RetrievalResult(
            method="dense",
            candidates=pl.DataFrame(candidates),
            query_latencies_ms=pl.DataFrame(latencies),
            build_stats=self.stats,
        )
