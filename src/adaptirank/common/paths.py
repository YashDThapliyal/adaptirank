"""Repository-root and artifact path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Find the repository root without relying on the caller's current directory."""

    configured = os.environ.get("ADAPTIRANK_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    cursor = (start or Path(__file__)).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "AGENTS.md").is_file():
            return candidate
    raise RuntimeError("could not locate AdaptiRank repository root")


def resolve_project_path(path: Path, root: Path | None = None) -> Path:
    """Resolve relative configuration paths against the repository root."""

    if path.is_absolute():
        return path
    return (root or project_root()) / path
