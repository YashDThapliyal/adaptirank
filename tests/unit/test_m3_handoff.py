from pathlib import Path

import polars as pl

from adaptirank.ranking.handoff import validate_contract


def test_validate_contract_preserves_unjudged_and_splits(tmp_path: Path) -> None:
    rows = []
    counts = {"train": 18_799, "validation": 2_089, "test": 8_956}
    for split, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "query_key": f"{split}-{index}",
                    "product_key": "p",
                    "split": split,
                    "bm25_score": 1.0,
                    "bm25_rank": 1,
                    "dense_score": 1.0,
                    "dense_rank": 1,
                    "hybrid_score": 1.0,
                    "hybrid_rank": 1,
                    "rrf_score": 1.0,
                    "rrf_rank": 1,
                    "esci_label": None,
                    "relevance_grade": None,
                    "judgment_status": "unjudged",
                }
            )
    path = tmp_path / "contract.parquet"
    pl.DataFrame(rows).write_parquet(path)
    result = validate_contract(path)
    assert result["queries"] == 29_844
    assert result["unjudged"] == 29_844
    assert result["query_split_collisions"] == 0
