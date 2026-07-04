"""Score top-M candidates with a pretrained cross-encoder for the M3 handoff.

Reads the persisted M3 three-split candidate contract (or a smoke contract), scores the top-M
per query with a pinned cross-encoder, and persists per-pair scores + latency stats under the
experiment artifact contract. Scoring is resumable via an on-disk checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# macOS PyTorch/faiss OpenMP guard is harmless here and keeps parity with retrieval entry points.
if sys.platform == "darwin":
    import os

    for _var in (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.reproducibility import seed_everything
from adaptirank.common.run import ExperimentRun
from adaptirank.ranking.config import CrossEncoderRunConfig
from adaptirank.ranking.cross_encoder import CrossEncoderScorer, score_top_m, scoring_stats
from adaptirank.retrieval.evaluate import write_json


def _limit_queries(contract: pl.DataFrame, queries: pl.DataFrame, per_split: int) -> pl.DataFrame:
    keep = (
        queries.sort("benchmark_split", "query_key")
        .group_by("benchmark_split", maintain_order=True)
        .head(per_split)
        .get_column("query_key")
        .to_list()
    )
    return contract.filter(pl.col("query_key").is_in(keep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, CrossEncoderRunConfig)
    seed_everything(config.run.seed)
    root = project_root()

    contract_path = (
        resolve_project_path(config.retrieval_output_dir, root)
        / config.dataset_fingerprint
        / config.retrieval_artifact_name
        / "hybrid"
        / "candidate_contract.parquet"
    )
    if not contract_path.is_file():
        raise FileNotFoundError(f"candidate contract not found: {contract_path}")
    contract = pl.read_parquet(contract_path)
    queries = pl.read_parquet(config.dataset_dir / "queries.parquet")
    catalog = pl.read_parquet(config.dataset_dir / "catalog.parquet")
    if config.max_queries_per_split is not None:
        contract = _limit_queries(contract, queries, config.max_queries_per_split)

    out_root = (
        resolve_project_path(config.output_dir, root)
        / config.dataset_fingerprint
        / config.artifact_name
        / "cross_encoder"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = out_root / "scores.parquet"

    scorer = CrossEncoderScorer(
        model_name=config.cross_encoder.model_name,
        model_revision=config.cross_encoder.model_revision,
        device=config.cross_encoder.device,
        batch_size=config.cross_encoder.batch_size,
        max_length=config.cross_encoder.max_length,
    )
    with ExperimentRun(
        experiment=f"{config.run.experiment}_cross_encoder",
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        started = time.perf_counter()
        scores = score_top_m(
            contract,
            queries,
            catalog,
            scorer,
            fields=config.cross_encoder.fields,
            top_m=config.top_m,
            rank_column=config.rank_column,
            checkpoint_path=checkpoint,
            block_queries=config.block_queries,
        )
        stats = scoring_stats(started, scores.height, scorer)
        stats.update(
            {
                "top_m": config.top_m,
                "rank_column": config.rank_column,
                "retrieval_artifact_name": config.retrieval_artifact_name,
                "candidate_contract": str(contract_path),
                "pairs_by_split": scores.group_by("split").len().sort("split").to_dicts(),
            }
        )
        write_json(out_root / "scoring_stats.json", stats)
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("cross_encoder_scores", checkpoint)
        run.record_artifact("scoring_stats", out_root / "scoring_stats.json")
        run.record_metrics(stats)
        print(run.run_dir)


if __name__ == "__main__":
    main()
