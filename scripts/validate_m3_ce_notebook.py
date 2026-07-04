"""Validate the canonical M3 CE A100 Colab notebook structure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK = Path("notebooks/m3_cross_encoder_a100_runall.ipynb")
EXPECTED_SECTIONS = tuple(range(19))


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(item) for item in source)
    return str(source)


def main() -> None:
    raw = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = raw.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{NOTEBOOK} has no cells")

    text = "\n".join(_cell_source(cell) for cell in cells)
    for section in EXPECTED_SECTIONS:
        if f"## {section}." not in text:
            raise ValueError(f"missing numbered section {section}")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    if not code_cells:
        raise ValueError("notebook has no code cells")
    empty = [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and not _cell_source(cell).strip()
    ]
    if empty:
        raise ValueError(f"empty code cells at indexes: {empty}")

    if "adaptirank" not in text:
        raise ValueError("notebook does not import or reference adaptirank")
    if "adaptirank.ranking.ce_workflow" not in text:
        raise ValueError("notebook does not use adaptirank.ranking.ce_workflow")

    if "M3_CE_RELEASE_REF" not in text:
        raise ValueError("notebook must define M3_CE_RELEASE_REF")
    if "resolve_release_ref" not in text and "checkout release tag" not in text.lower():
        raise ValueError("notebook must checkout release tag or call resolve_release_ref")

    print(f"validated {NOTEBOOK}: {len(cells)} cells, {len(code_cells)} code cells")


if __name__ == "__main__":
    main()
