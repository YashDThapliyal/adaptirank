"""Persistent Tantivy BM25 retriever."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import polars as pl
import tantivy

from adaptirank.retrieval.base import IndexBuildStats, RetrievalResult, Retriever, directory_size

_TOKENS = re.compile(r"\w+", flags=re.UNICODE)


def _safe_query(text: str) -> str:
    return " ".join(_TOKENS.findall(text.lower()))


class BM25Retriever(Retriever):
    """Tantivy-backed BM25 with configurable indexed query fields."""

    def __init__(self, *, fields: tuple[str, ...], writer_heap_bytes: int = 512_000_000) -> None:
        self.fields = fields
        self.writer_heap_bytes = writer_heap_bytes
        self.index: tantivy.Index | None = None
        self.index_dir: Path | None = None
        self.stats: IndexBuildStats | None = None

    @staticmethod
    def _schema() -> Any:
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("product_key", stored=True)
        builder.add_text_field("title", stored=False)
        builder.add_text_field("description", stored=False)
        builder.add_text_field("brand", stored=False)
        return builder.build()

    def build(self, catalog_path: Path, artifact_dir: Path) -> IndexBuildStats:
        index_dir = artifact_dir / "index"
        metadata_path = artifact_dir / "index_metadata.json"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if index_dir.is_dir() and tantivy.Index.exists(str(index_dir)) and metadata_path.is_file():
            index = tantivy.Index(self._schema(), path=str(index_dir), reuse=True)
            metadata: dict[str, Any] = json.loads(metadata_path.read_text())
            build_seconds = float(metadata["build_seconds"])
            metadata = {**metadata, "cache_reused": True}
        else:
            index_dir.mkdir(parents=True, exist_ok=True)
            index = tantivy.Index(self._schema(), path=str(index_dir), reuse=False)
            writer = index.writer(heap_size=self.writer_heap_bytes)
            document_count = 0
            frame = pl.read_parquet(
                catalog_path,
                columns=["product_key", "title", "description", "brand"],
            )
            for row in frame.iter_rows(named=True):
                writer.add_document(
                    tantivy.Document(
                        product_key=str(row["product_key"]),
                        title=str(row["title"] or ""),
                        description=str(row["description"] or ""),
                        brand=str(row["brand"] or ""),
                    )
                )
                document_count += 1
            writer.commit()
            writer.wait_merging_threads()
            index.reload()
            build_seconds = time.perf_counter() - started
            metadata = {
                "engine": "tantivy",
                "engine_version": getattr(tantivy, "__version__", "unknown"),
                "scoring": "BM25",
                "indexed_fields": ["title", "description", "brand"],
                "document_count": document_count,
                "build_seconds": build_seconds,
                "cache_reused": False,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        index.config_reader("OnCommit")
        self.index = index
        self.index_dir = index_dir
        stats = IndexBuildStats(
            build_seconds=build_seconds,
            index_size_bytes=directory_size(index_dir),
            document_count=int(metadata["document_count"]),
            artifact_paths={"index": str(index_dir), "metadata": str(metadata_path)},
            metadata={**metadata, "query_fields": list(self.fields)},
        )
        self.stats = stats
        return stats

    def retrieve(self, queries: pl.DataFrame, top_k: int) -> RetrievalResult:
        if self.index is None or self.stats is None:
            raise RuntimeError("BM25 index must be built before retrieval")
        searcher = self.index.searcher()
        candidates: list[dict[str, Any]] = []
        latencies: list[dict[str, Any]] = []
        method = f"bm25_{'_'.join(self.fields)}"
        for query in queries.iter_rows(named=True):
            started = time.perf_counter()
            parsed, errors = self.index.parse_query_lenient(
                _safe_query(str(query["query_text"])),
                default_field_names=list(self.fields),
            )
            result = searcher.search(parsed, limit=top_k, count=False)
            elapsed_ms = (time.perf_counter() - started) * 1000
            latencies.append({"query_key": query["query_key"], "latency_ms": elapsed_ms})
            for rank, (score, address) in enumerate(result.hits, start=1):
                document = searcher.doc(address)
                candidates.append(
                    {
                        "query_key": query["query_key"],
                        "product_key": str(document.get_first("product_key")),
                        "split": query["benchmark_split"],
                        "method": method,
                        "score": float(score),
                        "rank": rank,
                        "query_parse_errors": len(errors),
                    }
                )
        return RetrievalResult(
            method=method,
            candidates=pl.DataFrame(candidates),
            query_latencies_ms=pl.DataFrame(latencies),
            build_stats=self.stats,
        )
