"""Source acquisition and exact local provenance."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import requests

from adaptirank.common.paths import project_root, resolve_project_path
from adaptirank.data.config import EsciConfig, SourceFileConfig


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the locally observed SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """Download with a resumable partial file and atomic completion."""

    partial = destination.with_suffix(f"{destination.suffix}.part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def _stage_file(item: SourceFileConfig, destination: Path, *, root: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if item.fixture_path is not None:
        shutil.copy2(resolve_project_path(item.fixture_path, root), destination)
    else:
        assert item.url is not None
        _download(item.url, destination)


def stage_sources(config: EsciConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    """Materialize every configured source and write an observed provenance manifest."""

    root = project_root()
    raw_dir = resolve_project_path(config.source.raw_dir, root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for item in config.source.files:
        destination = raw_dir / item.filename
        _stage_file(item, destination, root=root)
        observed = sha256_file(destination)
        if item.authoritative_sha256 is None:
            verification = "authoritative_checksum_not_published"
        elif observed == item.authoritative_sha256:
            verification = "matched_authoritative_checksum"
        else:
            raise ValueError(f"authoritative checksum mismatch for {item.filename}")
        paths[item.role] = destination
        records.append(
            {
                "role": item.role,
                "filename": item.filename,
                "repository": config.source.repository,
                "revision": config.source.revision,
                "source_url": item.url,
                "fixture_path": str(item.fixture_path) if item.fixture_path else None,
                "observed_size_bytes": destination.stat().st_size,
                "observed_sha256": observed,
                "authoritative_sha256": item.authoritative_sha256,
                "checksum_verification": verification,
            }
        )
    manifest: dict[str, Any] = {
        "source_mode": config.source.mode,
        "repository": config.source.repository,
        "pinned_commit_sha": config.source.revision,
        "checksum_note": (
            "observed_sha256 values are locally generated integrity fingerprints; they are not "
            "authoritative unless checksum_verification says matched_authoritative_checksum"
        ),
        "files": sorted(records, key=lambda item: str(item["role"])),
    }
    manifest_path = raw_dir / "source_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return paths, manifest


def load_staged_sources(config: EsciConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    """Load a previously staged source set without performing network access."""

    raw_dir = resolve_project_path(config.source.raw_dir, project_root())
    paths: dict[str, Path] = {item.role: raw_dir / item.filename for item in config.source.files}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    manifest_path = raw_dir / "source_manifest.json"
    if not manifest_path.is_file():
        missing.append(str(manifest_path))
    if missing:
        raise FileNotFoundError(f"source files are not staged: {missing}")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    return paths, manifest
