"""Run one M2 retrieval stage through the shared evaluation harness."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# macOS ships PyTorch and faiss-cpu with separate OpenMP/BLAS runtimes. Once torch's runtime
# initializes, faiss IVF k-means training deadlocks (the BLAS thread pool collides with torch's
# OpenMP). Constraining every threading layer to a single thread BEFORE torch or faiss is imported
# avoids the deadlock; the dense build runs in seconds. See docs/DECISIONS.md ADR-007. The guard is
# macOS-only so Linux/CUDA hosts, where the two share one runtime, keep default parallelism.
if sys.platform == "darwin":
    for _var in (
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
