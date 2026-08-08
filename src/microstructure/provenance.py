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


def git_source_tree_sha256(project_root: str | Path) -> str:
    """Hash exact tracked and non-ignored untracked working-tree bytes.

    The Git revision plus a dirty boolean cannot distinguish two different
    patches on the same commit.  This digest is path-stable, excludes ignored
    raw/run artifacts, and streams file content so provenance remains exact
    without loading the source tree into memory.
    """
    requested_root = Path(project_root).resolve()
    repository = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=requested_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if repository.returncode != 0:
        return hashlib.sha256(b"NOT_A_GIT_WORKTREE").hexdigest()
    repository_root = Path(repository.stdout.strip()).resolve()
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if listing.returncode != 0:
        raise RuntimeError("unable to enumerate Git source-tree files")

    digest = hashlib.sha256()
    relative_paths = sorted(item for item in listing.stdout.split(b"\0") if item)
    for encoded_relative in relative_paths:
        relative = Path(os.fsdecode(encoded_relative))
        path = repository_root / relative
        digest.update(len(encoded_relative).to_bytes(8, "big"))
        digest.update(encoded_relative)
        if path.exists() or path.is_symlink():
            digest.update((path.lstat().st_mode & 0o7777).to_bytes(4, "big"))
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8", errors="surrogateescape")
            digest.update(b"SYMLINK\0")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif path.is_file():
            digest.update(b"FILE\0")
            digest.update(sha256_file(path).encode())
        elif path.exists():
            digest.update(b"OTHER\0")
        else:
            digest.update(b"MISSING\0")
    return digest.hexdigest()


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
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
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
    git = asdict(state)
    git["source_tree_sha256"] = git_source_tree_sha256(project_root)
    return {
        "generated_at_utc": utc_now_iso(),
        "evidence_tier": evidence_tier,
        "config_sha256": config_hash,
        "input_manifest_sha256": sorted(input_manifests),
        "git": git,
        "runtime": runtime_metadata(),
    }
