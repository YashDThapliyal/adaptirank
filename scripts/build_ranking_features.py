"""Build the M3 label-free ranking feature dataset from Hybrid top-500 candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.run import ExperimentRun
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.config import RankingFeatureRunConfig
from adaptirank.ranking.features import feature_frame, feature_schema
from adaptirank.retrieval.evaluate import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, RankingFeatureRunConfig)
    root = project_root()
    contract_path = (
        resolve_project_path(config.retrieval_output_dir, root)
        / config.dataset_fingerprint
        / config.retrieval_artifact_name
        / "hybrid"
        / "candidate_contract.parquet"
    )
    output = (
        resolve_project_path(config.output_dir, root)
        / config.dataset_fingerprint
        / config.artifact_name
        / "features"
    )
    output.mkdir(parents=True, exist_ok=True)
    contract = pl.scan_parquet(contract_path)
    queries = pl.scan_parquet(config.dataset_dir / "queries.parquet")
    catalog = pl.scan_parquet(config.dataset_dir / "catalog.parquet")
    with ExperimentRun(
        experiment=config.run.experiment,
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        split_records = []
        for split in config.splits:
            destination = output / f"{split}.parquet"
            features = feature_frame(
                contract,
                queries,
                catalog,
                split=split,
                rank_column=config.features.candidate_rank_column,
                top_m=config.features.top_m,
            )
            features.sink_parquet(
                destination,
                compression="zstd",
                row_group_size=250_000,
                mkdir=True,
                engine="streaming",
            )
            stats = (
                pl.scan_parquet(destination)
                .select(
                    pl.len().alias("rows"),
                    pl.col("query_key").n_unique().alias("queries"),
                    (pl.col("judgment_status") == "judged").sum().alias("judged"),
                    (pl.col("judgment_status") == "unjudged").sum().alias("unjudged"),
                )
                .collect()
                .to_dicts()[0]
            )
            split_records.append(
                {
                    "split": split,
                    **stats,
                    "path": str(destination),
                    "sha256": sha256_file(destination),
                }
            )
            run.record_artifact(f"features_{split}", destination)
        schema = {
            **feature_schema(),
            "dataset_fingerprint": config.dataset_fingerprint,
            "candidate_contract": str(contract_path.resolve()),
            "candidate_contract_sha256": sha256_file(contract_path),
            "candidate_policy": {
                "rank_column": config.features.candidate_rank_column,
                "top_m": config.features.top_m,
            },
            "split_policy": {
                "train": "fit only",
                "validation": "selection and early stopping only",
                "test": "final evaluation only after freeze",
            },
            "splits": split_records,
        }
        schema_path = output / "feature_schema.json"
        write_json(schema_path, schema)
        run.record_artifact("feature_schema", schema_path)
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_metrics(
            {"splits": split_records, "feature_count": len(feature_schema()["feature_columns"])}
        )
        print(run.run_dir)


if __name__ == "__main__":
    main()
