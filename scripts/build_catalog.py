"""Build normalized ESCI data from an already-staged source set."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptirank.data.pipeline import load_esci_config, run_build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_dir, result = run_build(load_esci_config(args.config))
    print(run_dir)
    print(result.dataset_dir)


if __name__ == "__main__":
    main()
