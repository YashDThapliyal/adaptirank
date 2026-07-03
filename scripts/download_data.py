"""Stage ESCI source files and record exact observed provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptirank.data.pipeline import load_esci_config, run_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_dir = run_download(load_esci_config(args.config))
    print(run_dir)


if __name__ == "__main__":
    main()
