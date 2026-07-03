from pathlib import Path

import polars as pl

from adaptirank.retrieval.bm25 import BM25Retriever


def test_persistent_bm25_retrieves_and_reloads(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.parquet"
    pl.DataFrame(
        {
            "product_key": ["shoe", "bottle", "charger"],
            "title": ["waterproof trail shoe", "steel water bottle", "usb charger"],
            "description": ["running grip", "insulated", "phone power"],
            "brand": ["north", "aqua", "volt"],
        }
    ).write_parquet(catalog_path)
    queries = pl.DataFrame(
        {
            "query_key": ["q1"],
            "query_text": ["trail running shoe"],
            "benchmark_split": ["test"],
        }
    )
    first = BM25Retriever(fields=("title", "description"), writer_heap_bytes=50_000_000)
    stats = first.build(catalog_path, tmp_path / "bm25")
    result = first.retrieve(queries, 3)
    assert result.candidates.sort("rank").row(0, named=True)["product_key"] == "shoe"
    assert stats.document_count == 3
    assert stats.index_size_bytes > 0

    reloaded = BM25Retriever(fields=("title",), writer_heap_bytes=50_000_000)
    loaded_stats = reloaded.build(catalog_path, tmp_path / "bm25")
    assert loaded_stats.build_seconds > 0
    assert loaded_stats.metadata["cache_reused"] is True
    assert reloaded.retrieve(queries, 3).candidates.height > 0
