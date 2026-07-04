"""Build the deduplicated Hybrid-top-100 plus LambdaMART-top-50 CE pair union."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from adaptirank.common.config import load_config
from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.common.run import ExperimentRun
from adaptirank.data.provenance import sha256_file
from adaptirank.ranking.ce_union import build_ce_union
from adaptirank.ranking.config import CEUnionRunConfig
from adaptirank.retrieval.evaluate import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, CEUnionRunConfig)
    root = project_root()
    retrieval = resolve_project_path(config.retrieval_root, root)
    learned = resolve_project_path(config.learned_root, root)
    output = resolve_project_path(config.output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    hybrid = pl.read_parquet(retrieval / "hybrid" / "weighted" / "raw_candidates.parquet")
    lambdamart = pl.concat(
        [
            pl.read_parquet(learned / f"rankings_{split}.parquet")
            for split in ("train", "validation", "test")
        ]
    )
    with ExperimentRun(
        experiment=config.run.experiment,
        purpose=config.run.purpose,
        seed=config.run.seed,
        config=config.model_dump(mode="json"),
        artifacts_dir=config.run.artifacts_dir,
    ) as run:
        union, stats = build_ce_union(
            hybrid,
            lambdamart,
            hybrid_top_m=config.hybrid_top_m,
            lambdamart_top_m=config.lambdamart_top_m,
        )
        union_path = output / "pair_union.parquet"
        union.write_parquet(union_path, compression="zstd")
        manifest = {
            **stats,
            "dataset_fingerprint": config.dataset_fingerprint,
            "pair_union_sha256": sha256_file(union_path),
            "pair_union_path": str(union_path.resolve()),
            "hybrid_source_sha256": sha256_file(
                retrieval / "hybrid" / "weighted" / "raw_candidates.parquet"
            ),
            "lambdamart_ranking_sha256": {
                split: sha256_file(learned / f"rankings_{split}.parquet")
                for split in ("train", "validation", "test")
            },
            "coverage_gate": "PASS",
        }
        manifest_path = output / "pair_union_manifest.json"
        write_json(manifest_path, manifest)
        run.set_dataset_fingerprint(config.dataset_fingerprint)
        run.record_artifact("pair_union", union_path)
        run.record_artifact("pair_union_manifest", manifest_path)
        run.record_metrics(manifest)
        print(run.run_dir)


if __name__ == "__main__":
    main()
