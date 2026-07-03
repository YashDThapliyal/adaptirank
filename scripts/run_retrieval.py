"""Run one M2 retrieval stage through the shared evaluation harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptirank.retrieval.pipeline import load_retrieval_config, run_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), required=True)
    args = parser.parse_args()
    run_dir = run_method(load_retrieval_config(args.config), args.method)
    print(run_dir)


if __name__ == "__main__":
    main()
