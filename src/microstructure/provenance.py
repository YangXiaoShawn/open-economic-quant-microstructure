"""Checksums, Git state, and JSON artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    dirty: bool


class ImportOriginError(RuntimeError):
    """Raised when loaded project code does not belong to one source root."""


def _canonical_absolute_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ImportOriginError(f"{label} must be a canonical absolute path")
    return path


def _reject_symlink_path(path: Path, label: str) -> None:
    """Reject a missing path or any symlink component without following it."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ImportOriginError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ImportOriginError(f"{label} contains symlink component: {current}")


def _loaded_module(value: str | ModuleType) -> tuple[str, ModuleType]:
    if isinstance(value, str):
        name = value
        module = sys.modules.get(name)
        if not isinstance(module, ModuleType):
            raise ImportOriginError(f"required project module is not loaded: {name}")
    elif isinstance(value, ModuleType):
        module = value
        name = module.__name__
        if not name or sys.modules.get(name) is not module:
            raise ImportOriginError(
                "loaded project module object is not its canonical sys.modules entry"
            )
    else:
        raise TypeError("project module authorities must be module names or module objects")
    if name != "microstructure" and not name.startswith("microstructure."):
        raise ImportOriginError(f"module is outside the microstructure namespace: {name}")
    return name, module


def assert_project_module_origins(
    project_root: str | Path,
    *modules: str | ModuleType,
) -> None:
    """Prove that loaded project modules come from one unsymlinked checkout.

    The check is intentionally about the code that Python actually loaded, not
    merely the checkout whose bytes were hashed for provenance.  Every module
    and its loaded parent packages must have a canonical ``__file__`` matching
    its name below ``<project_root>/src/microstructure``.  Package search paths
    must contain exactly that one directory, closing mixed/editable namespace
    cases where a clean checkout is hashed while code executes elsewhere.
    """

    root = _canonical_absolute_path(project_root, "project root")
    source_root = root / "src" / "microstructure"
    _reject_symlink_path(source_root, "project source root")
    try:
        source_metadata = source_root.lstat()
    except OSError as error:  # pragma: no cover - covered by path walk above
        raise ImportOriginError("project source root is unavailable") from error
    if not stat.S_ISDIR(source_metadata.st_mode):
        raise ImportOriginError("project source root is not a regular directory")
    try:
        resolved_source_root = source_root.resolve(strict=True)
    except OSError as error:  # pragma: no cover - covered by path walk above
        raise ImportOriginError("project source root cannot be resolved") from error

    requested: dict[str, ModuleType] = {}
    for value in ("microstructure", "microstructure.provenance", *modules):
        name, module = _loaded_module(value)
        previous = requested.setdefault(name, module)
        if previous is not module:
            raise ImportOriginError(f"mixed loaded module objects for {name}")
        components = name.split(".")
        for length in range(1, len(components)):
            parent_name = ".".join(components[:length])
            parent, parent_module = _loaded_module(parent_name)
            previous_parent = requested.setdefault(parent, parent_module)
            if previous_parent is not parent_module:
                raise ImportOriginError(f"mixed loaded module objects for {parent}")

    for name, module in sorted(requested.items()):
        raw_file = getattr(module, "__file__", None)
        if type(raw_file) is not str or not raw_file:
            raise ImportOriginError(f"loaded project module lacks __file__ authority: {name}")
        observed = _canonical_absolute_path(raw_file, f"loaded module {name}")
        _reject_symlink_path(observed, f"loaded module {name}")
        try:
            metadata = observed.lstat()
            resolved = observed.resolve(strict=True)
        except OSError as error:
            raise ImportOriginError(f"loaded project module is unavailable: {name}") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise ImportOriginError(f"loaded project module is not a regular source file: {name}")
        try:
            resolved.relative_to(resolved_source_root)
        except ValueError as error:
            raise ImportOriginError(
                f"loaded project module comes from a foreign source root: {name}"
            ) from error

        relative_parts = name.split(".")[1:]
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            expected = source_root.joinpath(*relative_parts).with_suffix(".py")
        else:
            expected_directory = source_root.joinpath(*relative_parts)
            try:
                search_paths = tuple(package_path)
            except TypeError as error:
                raise ImportOriginError(
                    f"loaded package search path is malformed: {name}"
                ) from error
            if len(search_paths) != 1 or type(search_paths[0]) is not str:
                raise ImportOriginError(f"loaded package has a mixed namespace path: {name}")
            search_path = _canonical_absolute_path(
                search_paths[0], f"loaded package search path {name}"
            )
            _reject_symlink_path(search_path, f"loaded package search path {name}")
            try:
                resolved_search_path = search_path.resolve(strict=True)
            except OSError as error:
                raise ImportOriginError(f"loaded package path is unavailable: {name}") from error
            if resolved_search_path != expected_directory:
                raise ImportOriginError(f"loaded package has a foreign namespace path: {name}")
            expected = expected_directory / "__init__.py"
        if resolved != expected:
            raise ImportOriginError(
                f"loaded project module path does not match its module name: {name}"
            )


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


def strict_git_state(project_root: str | Path) -> GitState:
    """Return Git identity only when both revision and status commands succeed.

    Generic sample workflows intentionally retain ``git_state``'s historical
    non-repository fallback.  Prospective producers use this stricter boundary
    so a failed ``git status`` can never masquerade as a clean worktree.
    """

    root = Path(project_root)
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if revision.returncode != 0 or status.returncode != 0:
        raise RuntimeError("unable to determine strict Git working-tree identity")
    return GitState(commit=revision.stdout.strip(), dirty=bool(status.stdout.strip()))


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
