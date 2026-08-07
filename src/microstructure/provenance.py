"""Checksums, Git state, and JSON artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    dirty: bool


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git_state(project_root: str | Path) -> GitState:
    """Return the repository commit and dirty state, including unborn repos."""
    root = Path(project_root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else "UNBORN"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return GitState(commit=commit, dirty=bool(status.stdout.strip()))


def runtime_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Atomically write stable, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def provenance_header(
    *,
    project_root: str | Path,
    config_hash: str,
    evidence_tier: str,
    input_manifests: list[str],
) -> dict[str, Any]:
    state = git_state(project_root)
    return {
        "generated_at_utc": utc_now_iso(),
        "evidence_tier": evidence_tier,
        "config_sha256": config_hash,
        "input_manifest_sha256": sorted(input_manifests),
        "git": asdict(state),
        "runtime": runtime_metadata(),
    }
