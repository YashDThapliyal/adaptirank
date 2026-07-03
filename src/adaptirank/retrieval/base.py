"""Common replaceable retriever interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class IndexBuildStats:
    build_seconds: float
    index_size_bytes: int
    document_count: int
    artifact_paths: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    method: str
    candidates: pl.DataFrame
    query_latencies_ms: pl.DataFrame
    build_stats: IndexBuildStats


class Retriever(ABC):
    """Interface shared by lexical and dense candidate retrievers."""

    @abstractmethod
    def build(self, catalog_path: Path, artifact_dir: Path) -> IndexBuildStats:
        """Build or load a persistent index."""

    @abstractmethod
    def retrieve(self, queries: pl.DataFrame, top_k: int) -> RetrievalResult:
        """Retrieve ranked candidates for locale-aware query keys."""


def directory_size(path: Path) -> int:
    """Return recursive on-disk bytes for a file or directory."""

    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
