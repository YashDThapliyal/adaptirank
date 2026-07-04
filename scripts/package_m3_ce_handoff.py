"""Package the bounded M3 cross-encoder inputs for transfer to an A100 Colab runtime."""

from __future__ import annotations

import json
import tarfile

from adaptirank.common.paths import project_root
from adaptirank.data.provenance import sha256_file
from adaptirank.retrieval.evaluate import write_json

FP = "dda38161938e829f2c2fc9b73d40d6cf922a5470c3b45bf176f742ee0ca7c667"


def main() -> None:
    root = project_root()
    output = root / "artifacts" / "handoffs"
    output.mkdir(parents=True, exist_ok=True)
    ce_root = root / "artifacts" / "ranking" / FP / "m3_three_split" / "cross_encoder"
    members = [
        ce_root / "pair_union.parquet",
        ce_root / "pair_union_manifest.json",
        root / "configs" / "ranking" / "cross_encoder_union_m3.yaml",
        root / "configs" / "ranking" / "m3_ce_evaluate.yaml",
        root / "artifacts" / "ranking" / FP / "m3_three_split" / "learned" / "selection.json",
        root / "artifacts" / "ranking" / FP / "m3_three_split" / "learned" / "report.json",
    ]
    missing = [str(path) for path in members if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CE handoff inputs are missing: {missing}")
    archive = output / "m3_ce_a100_input.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in members:
            handle.add(path, arcname=path.relative_to(root))
    manifest = {
        "archive": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "dataset_fingerprint": FP,
        "pair_union_sha256": sha256_file(ce_root / "pair_union.parquet"),
        "pair_union_rows": json.loads((ce_root / "pair_union_manifest.json").read_text())[
            "union_pairs"
        ],
        "dataset_archive_required_separately": "MyDrive/adaptirank/adaptirank_dataset.tar",
        "full_dense_index_not_required": True,
    }
    write_json(output / "m3_ce_a100_input_manifest.json", manifest)
    print(archive)


if __name__ == "__main__":
    main()
