"""Cross-symbol authority and terminal publication for frozen M8 L2 sessions.

This module deliberately depends on an injected, typed single-symbol capture
producer.  It does not import the exploratory CLI collector and contains no
exchange connection or order-entry path.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from microstructure.data.schemas import SCHEMA_VERSION, get_schema
from microstructure.m8_l2_config import (
    M8_L2_CONFIG_SOURCE_SHA256,
    M8_L2_FREEZE_COMMIT,
    M8_L2_PROTOCOL_SHA256,
    M8L2CaptureLimits,
    M8L2Session,
    M8L2StudyConfig,
    load_m8_l2_config,
)
from microstructure.provenance import (
    ImportOriginError,
    assert_project_module_origins,
    sha256_file,
    utc_now_iso,
    write_json,
)

_SESSION_SCHEMA_VERSION = "m8-live-l2-session-v3"
_CAMPAIGN_SCHEMA_VERSION = "m8-live-l2-campaign-authority-v2"
_RUNTIME_SCHEMA_VERSION = "m8-live-l2-runtime-fingerprint-v1"
_CAMPAIGN_AUTHORITY_NAME = "campaign_authority.json"
_CHECKSUM_NAME = "CHECKSUMS.sha256"
_COMPLETE_MARKER = "_SUCCESS"
_INSUFFICIENT_MARKER = "INSUFFICIENT_DATA"
_COMPLETE_BYTES = b"complete\n"
_INSUFFICIENT_BYTES = b"terminal\n"
_NANOSECONDS_PER_SECOND = 1_000_000_000
_MAX_JSON_AUTHORITY_BYTES = 8 * 1024 * 1024
_MAX_CAMPAIGN_AUTHORITY_BYTES = 16 * 1024
_RUNTIME_DISTRIBUTIONS = (
    "duckdb",
    "numpy",
    "polars",
    "pyarrow",
    "requests",
    "scikit-learn",
    "streamlit",
    "websockets",
)
_SYMBOL_SUMMARY_SCHEMA_VERSION = "m8-binance-l2-symbol-capture-v1"
_REQUIRED_CAPTURE_ARTIFACT_KINDS = frozenset(
    {
        "capture_summary",
        "normalized_data",
        "normalized_manifest",
        "quality_report",
        "raw_journal",
        "raw_journal_manifest",
        "raw_snapshot",
        "raw_snapshot_manifest",
    }
)

SessionStatus = Literal["COMPLETE", "INSUFFICIENT_DATA"]
CaptureStatus = Literal["COMPLETE", "FAILED"]
ReconstructionStatus = Literal["LIVE", "GAPPED", "INVALID", "NOT_STARTED"]


class M8L2CaptureError(RuntimeError):
    """Base class for orchestration, authority, and verification failures."""


class M8L2VerificationError(M8L2CaptureError):
    """Raised when a session bundle is not an exact immutable authority."""


class M8L2CaptureSystemError(M8L2CaptureError):
    """A nonterminal program/I/O failure; captured raw evidence is retained."""

    def __init__(self, message: str, *, evidence_root: Path | None = None) -> None:
        super().__init__(message)
        self.evidence_root = evidence_root


class M8L2DataFailure(M8L2CaptureError):
    """Typed prospective-session failure that may terminalize as insufficient."""

    def __init__(
        self,
        reason_code: str,
        *,
        phase: str,
        message: str,
        partial_result: SymbolCaptureResult | None = None,
    ) -> None:
        _validate_code(reason_code, "reason code")
        _validate_code(phase, "failure phase")
        super().__init__(message)
        self.reason_code = reason_code
        self.phase = phase
        self.partial_result = partial_result


@dataclass(frozen=True, slots=True)
class CaptureSourceIdentity:
    """Clean runtime source identity bound into one prospective session."""

    commit: str
    source_tree_sha256: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class _RuntimeFingerprint:
    """Canonical production interpreter/dependency identity."""

    python_implementation: str
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    dependencies: tuple[tuple[str, str], ...]
    sha256: str

    def payload(self) -> dict[str, object]:
        return _runtime_payload(
            python_implementation=self.python_implementation,
            python_version=self.python_version,
            platform_system=self.platform_system,
            platform_release=self.platform_release,
            platform_machine=self.platform_machine,
            dependencies=self.dependencies,
        )


@dataclass(frozen=True, slots=True)
class _CampaignAuthority:
    """Exact durable campaign identity shared by all four frozen sessions."""

    path: Path
    sha256: str
    raw: bytes
    source: CaptureSourceIdentity
    runtime: _RuntimeFingerprint
    nonce: str
    output_root_path: str
    output_root_device: int
    output_root_inode: int


@dataclass(frozen=True, slots=True)
class ObservedInterval:
    """One continuous interval backed only by OBSERVED reconstructed states."""

    continuity_id: str
    start_received_ns: int
    end_received_ns_exclusive: int

    def __post_init__(self) -> None:
        if not self.continuity_id:
            raise ValueError("observed interval continuity_id must not be empty")
        if self.start_received_ns < 0:
            raise ValueError("observed interval start must be nonnegative")
        if self.end_received_ns_exclusive <= self.start_received_ns:
            raise ValueError("observed interval end must be after start")

    @property
    def duration_ns(self) -> int:
        return self.end_received_ns_exclusive - self.start_received_ns

    def to_dict(self) -> dict[str, object]:
        return {
            "continuity_id": self.continuity_id,
            "start_received_ns": self.start_received_ns,
            "end_received_ns_exclusive": self.end_received_ns_exclusive,
            "duration_ns": self.duration_ns,
            "duration_seconds": self.duration_ns / _NANOSECONDS_PER_SECOND,
        }


@dataclass(frozen=True, slots=True)
class CapturedArtifact:
    """Explicit file coordinate returned by a single-symbol capture producer."""

    path: Path
    kind: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SymbolCaptureResult:
    """Bounded descriptor for one symbol's complete or partial capture evidence."""

    symbol: str
    capture_id: str
    status: CaptureStatus
    completion_reason: str
    reconstruction_status: ReconstructionStatus
    messages: int
    normalized_rows: int
    reconstructed_rows: int
    excluded_rows: int
    continuity_epochs: int
    snapshot_anchors: int
    sequence_gaps: int
    quality_errors: int
    quality_warnings: int
    max_raw_frame_bytes_observed: int
    max_arrow_batch_bytes_observed: int
    first_raw_received_ns: int | None
    last_raw_received_ns: int | None
    valid_observed_intervals: tuple[ObservedInterval, ...]
    artifacts: tuple[CapturedArtifact, ...]
    failure_reason_code: str | None = None
    failure_phase: str | None = None


class CaptureOne(Protocol):
    """Injected adapter boundary for one duration- and boundary-aware capture."""

    async def __call__(
        self,
        *,
        symbol: str,
        scheduled_start_ns: int,
        scheduled_end_ns: int,
        stage_root: Path,
        limits: M8L2CaptureLimits,
        session_id: str,
    ) -> SymbolCaptureResult: ...


class CaptureClock(Protocol):
    def time_ns(self) -> int: ...

    async def sleep(self, seconds: float) -> None: ...


class _SystemClock:
    def time_ns(self) -> int:
        return time.time_ns()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(frozen=True, slots=True)
class M8L2SessionBundle:
    root: Path
    status: SessionStatus
    session_id: str
    session_date: str
    role: str
    manifest_path: Path
    manifest_sha256: str
    checksum_path: Path
    marker_path: Path
    reason_codes: tuple[str, ...]


def _validate_code(value: str, label: str) -> None:
    if not value or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in value
    ):
        raise ValueError(f"{label} must use uppercase letters, digits, and underscores")


def _is_lower_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _stable_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _run_git_text(project_root: Path, arguments: Sequence[str], *, label: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, UnicodeError) as error:
        raise M8L2CaptureSystemError(f"cannot execute Git {label}") from error
    if result.returncode != 0:
        raise M8L2CaptureSystemError(f"Git {label} failed with exit {result.returncode}")
    return result.stdout


def _run_git_bytes(project_root: Path, arguments: Sequence[str], *, label: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot execute Git {label}") from error
    if result.returncode != 0:
        raise M8L2CaptureSystemError(f"Git {label} failed with exit {result.returncode}")
    return result.stdout


def _strict_git_revision(project_root: Path) -> str:
    output = _run_git_text(project_root, ("rev-parse", "HEAD"), label="revision lookup")
    lines = output.splitlines()
    if len(lines) != 1:
        raise M8L2CaptureSystemError("Git revision lookup returned a noncanonical result")
    return lines[0]


def _strict_git_status(project_root: Path) -> str:
    return _run_git_text(
        project_root,
        ("status", "--porcelain=v1", "--untracked-files=normal"),
        label="status lookup",
    )


def _hash_regular_nofollow(path: Path, metadata: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot open Git source file: {path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ):
            raise M8L2CaptureSystemError(f"Git source file changed before hashing: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = path.lstat()
        coordinates = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if coordinates != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or coordinates != (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        ):
            raise M8L2CaptureSystemError(f"Git source file changed while hashing: {path}")
        return digest.hexdigest()
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot hash Git source file: {path}") from error
    finally:
        os.close(descriptor)


def _strict_source_tree_sha256(repository_root: Path, listing: bytes) -> str:
    encoded_paths = [item for item in listing.split(b"\0") if item]
    if len(encoded_paths) != len(set(encoded_paths)):
        raise M8L2CaptureSystemError("Git source-tree listing contains duplicate paths")
    digest = hashlib.sha256()
    for encoded_relative in sorted(encoded_paths):
        relative = Path(os.fsdecode(encoded_relative))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise M8L2CaptureSystemError("Git source-tree listing contains an unsafe path")
        path = repository_root / relative
        digest.update(len(encoded_relative).to_bytes(8, "big"))
        digest.update(encoded_relative)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            digest.update(b"MISSING\0")
            continue
        except OSError as error:
            raise M8L2CaptureSystemError(f"cannot inspect Git source path: {path}") from error
        digest.update((metadata.st_mode & 0o7777).to_bytes(4, "big"))
        if stat.S_ISLNK(metadata.st_mode):
            try:
                target = os.readlink(path).encode("utf-8", errors="surrogateescape")
                repeated = path.lstat()
            except OSError as error:
                raise M8L2CaptureSystemError(f"cannot hash Git source symlink: {path}") from error
            if (metadata.st_dev, metadata.st_ino, metadata.st_mtime_ns, metadata.st_ctime_ns) != (
                repeated.st_dev,
                repeated.st_ino,
                repeated.st_mtime_ns,
                repeated.st_ctime_ns,
            ):
                raise M8L2CaptureSystemError(f"Git source symlink changed while hashing: {path}")
            digest.update(b"SYMLINK\0")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
        elif stat.S_ISREG(metadata.st_mode):
            digest.update(b"FILE\0")
            digest.update(_hash_regular_nofollow(path, metadata).encode())
        else:
            digest.update(b"OTHER\0")
    return digest.hexdigest()


def _source_identity(project_root: Path) -> CaptureSourceIdentity:
    """Take one fail-closed clean Git snapshot without lenient provenance fallbacks."""

    first_revision = _strict_git_revision(project_root)
    first_status = _strict_git_status(project_root)
    if first_status.strip():
        raise M8L2CaptureSystemError(
            "prospective live-L2 capture requires a clean runtime source tree"
        )
    top_level_raw = _run_git_text(
        project_root,
        ("rev-parse", "--show-toplevel"),
        label="repository-root lookup",
    )
    top_level_lines = top_level_raw.splitlines()
    if len(top_level_lines) != 1 or not top_level_lines[0]:
        raise M8L2CaptureSystemError("Git repository-root lookup returned a noncanonical result")
    repository_root = Path(top_level_lines[0]).resolve()
    try:
        project_root.resolve().relative_to(repository_root)
    except ValueError as error:
        raise M8L2CaptureSystemError("capture project root escapes the Git repository") from error
    listing = _run_git_bytes(
        repository_root,
        ("ls-files", "-z", "--cached", "--others", "--exclude-standard"),
        label="source-tree listing",
    )
    source_tree_sha256 = _strict_source_tree_sha256(repository_root, listing)
    second_status = _strict_git_status(project_root)
    second_revision = _strict_git_revision(project_root)
    if second_status.strip() or first_revision != second_revision:
        raise M8L2CaptureSystemError("runtime Git identity changed during source snapshot")
    return CaptureSourceIdentity(
        commit=first_revision,
        source_tree_sha256=source_tree_sha256,
        dirty=False,
    )


def _runtime_payload(
    *,
    python_implementation: str,
    python_version: str,
    platform_system: str,
    platform_release: str,
    platform_machine: str,
    dependencies: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "schema_version": _RUNTIME_SCHEMA_VERSION,
        "python": {
            "implementation": python_implementation,
            "version": python_version,
        },
        "platform": {
            "system": platform_system,
            "release": platform_release,
            "machine": platform_machine,
        },
        "dependencies": dict(dependencies),
    }


def _runtime_fingerprint() -> _RuntimeFingerprint:
    try:
        dependencies = tuple(
            (distribution, importlib_metadata.version(distribution))
            for distribution in _RUNTIME_DISTRIBUTIONS
        )
    except importlib_metadata.PackageNotFoundError as error:
        raise M8L2CaptureSystemError(
            f"production runtime dependency is not installed: {error.name}"
        ) from error
    implementation = platform.python_implementation()
    version = platform.python_version()
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    payload = _runtime_payload(
        python_implementation=implementation,
        python_version=version,
        platform_system=system,
        platform_release=release,
        platform_machine=machine,
        dependencies=dependencies,
    )
    return _RuntimeFingerprint(
        python_implementation=implementation,
        python_version=version,
        platform_system=system,
        platform_release=release,
        platform_machine=machine,
        dependencies=dependencies,
        sha256=_stable_sha256(payload),
    )


def current_m8_l2_runtime_fingerprint_sha256() -> str:
    """Return the canonical production runtime digest used by L2 authorities."""

    return _runtime_fingerprint().sha256


def _runtime_from_recorded(value: object, sha256: object) -> _RuntimeFingerprint:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "python",
        "platform",
        "dependencies",
    }:
        raise M8L2CaptureSystemError("recorded runtime fingerprint payload is not exact")
    if value.get("schema_version") != _RUNTIME_SCHEMA_VERSION:
        raise M8L2CaptureSystemError("recorded runtime fingerprint schema is unsupported")
    python_payload = value.get("python")
    platform_payload = value.get("platform")
    dependencies_payload = value.get("dependencies")
    if not isinstance(python_payload, Mapping) or set(python_payload) != {
        "implementation",
        "version",
    }:
        raise M8L2CaptureSystemError("recorded Python runtime identity is not exact")
    if not isinstance(platform_payload, Mapping) or set(platform_payload) != {
        "system",
        "release",
        "machine",
    }:
        raise M8L2CaptureSystemError("recorded platform runtime identity is not exact")
    if not isinstance(dependencies_payload, Mapping) or set(dependencies_payload) != set(
        _RUNTIME_DISTRIBUTIONS
    ):
        raise M8L2CaptureSystemError("recorded production dependency identity is not exact")
    implementation = python_payload.get("implementation")
    version = python_payload.get("version")
    system = platform_payload.get("system")
    release = platform_payload.get("release")
    machine = platform_payload.get("machine")
    if (
        type(implementation) is not str
        or not implementation
        or type(version) is not str
        or not version
        or type(system) is not str
        or not system
        or type(release) is not str
        or not release
        or type(machine) is not str
        or not machine
    ):
        raise M8L2CaptureSystemError("recorded Python runtime identity is malformed")
    dependencies: list[tuple[str, str]] = []
    for distribution in _RUNTIME_DISTRIBUTIONS:
        dependency_version = dependencies_payload.get(distribution)
        if type(dependency_version) is not str or not dependency_version:
            raise M8L2CaptureSystemError(
                f"recorded dependency version is malformed: {distribution}"
            )
        dependencies.append((distribution, dependency_version))
    canonical_payload = _runtime_payload(
        python_implementation=implementation,
        python_version=version,
        platform_system=system,
        platform_release=release,
        platform_machine=machine,
        dependencies=tuple(dependencies),
    )
    observed_sha256 = _stable_sha256(canonical_payload)
    if type(sha256) is not str or not _is_lower_sha256(sha256) or sha256 != observed_sha256:
        raise M8L2CaptureSystemError("recorded runtime fingerprint digest is invalid")
    if dict(value) != canonical_payload:
        raise M8L2CaptureSystemError("recorded runtime fingerprint payload is not canonical")
    return _RuntimeFingerprint(
        python_implementation=implementation,
        python_version=version,
        platform_system=system,
        platform_release=release,
        platform_machine=machine,
        dependencies=tuple(dependencies),
        sha256=observed_sha256,
    )


def _validate_source_identity(identity: CaptureSourceIdentity) -> None:
    if identity.dirty:
        raise M8L2CaptureSystemError(
            "prospective live-L2 capture requires a clean runtime source tree"
        )
    if len(identity.commit) != 40 or any(
        character not in "0123456789abcdef" for character in identity.commit
    ):
        raise M8L2CaptureSystemError("runtime Git commit must be a lowercase 40-character SHA-1")
    if not _is_lower_sha256(identity.source_tree_sha256):
        raise M8L2CaptureSystemError("runtime source-tree digest is not a lowercase SHA-256")


def _assert_loaded_source_root(project_root: Path) -> None:
    try:
        assert_project_module_origins(project_root, "microstructure.m8_l2_capture")
    except ImportOriginError as error:
        raise M8L2CaptureSystemError(
            "loaded microstructure code does not come from the hashed capture project root"
        ) from error


def _assert_production_capture_origin(project_root: Path, capture_one: CaptureOne) -> None:
    """Bind the real network adapter and every critical dependency to one checkout."""

    capture_type = type(capture_one)
    module_name = getattr(capture_type, "__module__", None)
    if (
        module_name != "microstructure.m8_l2_binance"
        or capture_type.__name__ != "BinanceM8L2Capture"
    ):
        raise M8L2CaptureSystemError(
            "production CaptureOne must be the exact BinanceM8L2Capture adapter"
        )
    adapter_module = sys.modules.get(module_name)
    if adapter_module is None or getattr(adapter_module, "BinanceM8L2Capture", None) is not (
        capture_type
    ):
        raise M8L2CaptureSystemError(
            "production BinanceM8L2Capture class lacks its canonical loaded module"
        )
    try:
        assert_project_module_origins(
            project_root,
            adapter_module,
            "microstructure.data.binance",
            "microstructure.data.book",
            "microstructure.data.quality",
            "microstructure.data.schemas",
            "microstructure.data.storage",
        )
    except ImportOriginError as error:
        raise M8L2CaptureSystemError(
            "production capture adapter has a foreign or mixed import origin"
        ) from error


def _revalidate_runtime_authority(
    *,
    config: M8L2StudyConfig,
    protocol: Path,
    expected_source: CaptureSourceIdentity,
    expected_runtime: _RuntimeFingerprint,
    source_identity_was_injected: bool,
) -> None:
    if not source_identity_was_injected:
        _assert_loaded_source_root(config.path.parent.parent.resolve())
    try:
        reloaded = load_m8_l2_config(config.path)
    except (OSError, ValueError) as error:
        raise M8L2CaptureSystemError(
            "frozen live-L2 config changed during session orchestration"
        ) from error
    if reloaded != config or reloaded.hash != config.hash:
        raise M8L2CaptureSystemError(
            "in-memory live-L2 config differs from its exact frozen byte authority"
        )
    _assert_protocol(protocol)
    if _runtime_fingerprint() != expected_runtime:
        raise M8L2CaptureSystemError(
            "production runtime fingerprint changed during session orchestration"
        )
    if source_identity_was_injected:
        return
    observed = _source_identity(config.path.parent.parent.resolve())
    _validate_source_identity(observed)
    if observed != expected_source:
        raise M8L2CaptureSystemError(
            "runtime commit/source-tree identity changed during session orchestration"
        )


def merge_observed_intervals(intervals: Sequence[ObservedInterval]) -> tuple[ObservedInterval, ...]:
    """Merge overlap only within the same continuity epoch and return time order."""

    by_continuity: dict[str, list[ObservedInterval]] = {}
    for interval in intervals:
        by_continuity.setdefault(interval.continuity_id, []).append(interval)
    merged: list[ObservedInterval] = []
    for continuity_id, values in by_continuity.items():
        ordered = sorted(
            values, key=lambda item: (item.start_received_ns, item.end_received_ns_exclusive)
        )
        current_start: int | None = None
        current_end: int | None = None
        for item in ordered:
            if current_start is None or current_end is None:
                current_start = item.start_received_ns
                current_end = item.end_received_ns_exclusive
            elif item.start_received_ns <= current_end:
                current_end = max(current_end, item.end_received_ns_exclusive)
            else:
                merged.append(ObservedInterval(continuity_id, current_start, current_end))
                current_start = item.start_received_ns
                current_end = item.end_received_ns_exclusive
        if current_start is not None and current_end is not None:
            merged.append(ObservedInterval(continuity_id, current_start, current_end))
    return tuple(
        sorted(
            merged,
            key=lambda item: (
                item.start_received_ns,
                item.end_received_ns_exclusive,
                item.continuity_id,
            ),
        )
    )


def _numeric_union(intervals: Sequence[ObservedInterval]) -> list[tuple[int, int]]:
    ordered = sorted(
        (
            (item.start_received_ns, item.end_received_ns_exclusive)
            for item in merge_observed_intervals(intervals)
        ),
        key=lambda item: item,
    )
    result: list[tuple[int, int]] = []
    for start, end in ordered:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def overlapping_observed_coverage_ns(
    left: Sequence[ObservedInterval], right: Sequence[ObservedInterval]
) -> int:
    """Return exact union-intersection coverage without bridging either feed's gaps."""

    left_union = _numeric_union(left)
    right_union = _numeric_union(right)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left_union) and right_index < len(right_union):
        left_start, left_end = left_union[left_index]
        right_start, right_end = right_union[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _session_identity(
    *,
    config: M8L2StudyConfig,
    session: M8L2Session,
    source: CaptureSourceIdentity,
    campaign_authority_sha256: str,
) -> str:
    if not _is_lower_sha256(campaign_authority_sha256):
        raise M8L2CaptureSystemError("session campaign authority digest is invalid")
    return _stable_sha256(
        {
            "schema_version": _SESSION_SCHEMA_VERSION,
            "config_sha256": config.hash,
            "config_source_sha256": config.source_sha256,
            "protocol_sha256": M8_L2_PROTOCOL_SHA256,
            "protocol_freeze_commit": M8_L2_FREEZE_COMMIT,
            "runtime_commit": source.commit,
            "runtime_source_tree_sha256": source.source_tree_sha256,
            "campaign_authority_sha256": campaign_authority_sha256,
            "date": session.date.isoformat(),
            "role": session.role,
            "scheduled_start_ns": session.start_ns,
            "scheduled_end_ns": session.end_ns,
        }
    )


def _terminal_root(output_root: Path, session: M8L2Session, session_id: str) -> Path:
    return output_root / "sessions" / f"{session.date.isoformat()}-{session.role}-{session_id[:20]}"


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise M8L2CaptureSystemError(f"{label} is not a non-symlink directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _reject_existing_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            continue
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise M8L2CaptureSystemError(f"{label} traverses a symlink: {current}")
        except OSError as error:
            raise M8L2CaptureSystemError(f"cannot inspect {label}: {current}") from error


def _prepare_output_root(output_root: str | Path) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    requested = Path(output_root)
    _reject_existing_symlink_components(requested, label="live-L2 output root")
    requested.mkdir(parents=True, exist_ok=True)
    root = requested.resolve()
    root_identity = _directory_identity(root, label="live-L2 output root")
    sessions = root / "sessions"
    if sessions.is_symlink():
        raise M8L2CaptureSystemError("live-L2 sessions directory must not be a symlink")
    sessions.mkdir(exist_ok=True)
    sessions_identity = _directory_identity(sessions, label="live-L2 sessions directory")
    return root, root_identity, sessions_identity


def _revalidate_output_layout(
    root: Path,
    *,
    root_identity: tuple[int, int],
    sessions_identity: tuple[int, int],
    target: Path,
) -> None:
    if _directory_identity(root, label="live-L2 output root") != root_identity:
        raise M8L2CaptureSystemError("live-L2 output root identity changed")
    sessions = root / "sessions"
    if _directory_identity(sessions, label="live-L2 sessions directory") != sessions_identity:
        raise M8L2CaptureSystemError("live-L2 sessions directory identity changed")
    if target.parent != sessions or target.absolute().parent != sessions.absolute():
        raise M8L2CaptureSystemError("live-L2 target escapes its sessions directory")


def _assert_protocol(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise M8L2CaptureSystemError(f"frozen live-L2 protocol does not exist: {path}")
    observed = sha256_file(path)
    if observed != M8_L2_PROTOCOL_SHA256:
        raise M8L2CaptureSystemError("live-L2 protocol bytes differ from the outcome-blind freeze")
    return observed


def _assert_no_symlink_components(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise M8L2CaptureSystemError(f"{label} escapes its evidence root: {path}") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise M8L2CaptureSystemError(f"cannot inspect {label}: {current}") from error
        if stat.S_ISLNK(mode):
            raise M8L2CaptureSystemError(f"{label} contains a symlink: {current}")


def _assert_regular_inside(path: Path, root: Path, label: str) -> Path:
    absolute_root = root.absolute()
    absolute_path = path.absolute()
    _assert_no_symlink_components(absolute_path, absolute_root, label)
    try:
        mode = absolute_path.lstat().st_mode
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISREG(mode):
        raise M8L2CaptureSystemError(f"{label} is not a regular file: {path}")
    resolved = absolute_path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise M8L2CaptureSystemError(f"{label} escapes its symbol stage: {path}") from error
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.absolute().relative_to(root.absolute()).as_posix()


def _walk_regular_evidence(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Walk evidence without following links and reject every special entry."""

    try:
        root_mode = root.lstat().st_mode
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot inspect session evidence root: {root}") from error
    if not stat.S_ISDIR(root_mode):
        raise M8L2CaptureSystemError(f"session evidence root is not a regular directory: {root}")
    files: list[Path] = []
    directories: list[Path] = [root]
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise M8L2CaptureSystemError(
                f"cannot enumerate session evidence: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise M8L2CaptureSystemError(f"cannot inspect session evidence: {path}") from error
            if stat.S_ISREG(mode):
                files.append(path)
            elif stat.S_ISDIR(mode):
                directories.append(path)
                pending.append(path)
            else:
                raise M8L2CaptureSystemError(
                    f"session evidence contains a non-regular filesystem entry: {path}"
                )
    return (
        tuple(sorted(files, key=lambda item: _relative(item, root))),
        tuple(sorted(directories, key=lambda item: _relative(item, root))),
    )


def _all_regular_files(root: Path) -> tuple[Path, ...]:
    return _walk_regular_evidence(root)[0]


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _decode_canonical_json(raw: bytes, *, label: str) -> object:
    try:
        text = raw.decode("utf-8")

        def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise M8L2CaptureSystemError(f"cannot parse canonical {label}") from error
    if raw != _canonical_json_bytes(value):
        raise M8L2CaptureSystemError(f"{label} is not canonical stable JSON")
    return value


def _read_bounded_regular_nofollow(
    path: Path, *, label: str, maximum_bytes: int = _MAX_JSON_AUTHORITY_BYTES
) -> bytes:
    """Read one stable regular file through a no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise M8L2CaptureSystemError(
            f"cannot open {label} without following links: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum_bytes:
            raise M8L2CaptureSystemError(f"{label} is not a bounded nonempty regular file: {path}")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        current = path.lstat()
        stable_coordinates = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_still_names_descriptor = (
            current.st_dev,
            current.st_ino,
            current.st_size,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) and stat.S_ISREG(current.st_mode)
        if (
            len(raw) > maximum_bytes
            or len(raw) != before.st_size
            or not stable_coordinates
            or not path_still_names_descriptor
        ):
            raise M8L2CaptureSystemError(f"{label} changed during its bounded read: {path}")
        return raw
    except OSError as error:
        raise M8L2CaptureSystemError(f"cannot read {label}: {path}") from error
    finally:
        os.close(descriptor)


def _strict_json(path: Path, *, label: str) -> object:
    """Read a bounded canonical JSON authority with duplicate/NaN rejection."""

    raw = _read_bounded_regular_nofollow(path, label=label)
    return _decode_canonical_json(raw, label=f"{label}: {path}")


def _validate_symbol_result(
    result: SymbolCaptureResult,
    *,
    expected_symbol: str,
    symbol_root: Path,
    session: M8L2Session,
) -> SymbolCaptureResult:
    if result.symbol != expected_symbol:
        raise M8L2CaptureSystemError(
            f"capture result symbol {result.symbol!r} does not match {expected_symbol!r}"
        )
    if not result.capture_id or "/" in result.capture_id or "\\" in result.capture_id:
        raise M8L2CaptureSystemError("single-symbol capture returned an unsafe capture_id")
    for name in (
        "messages",
        "normalized_rows",
        "reconstructed_rows",
        "excluded_rows",
        "continuity_epochs",
        "snapshot_anchors",
        "sequence_gaps",
        "quality_errors",
        "quality_warnings",
        "max_raw_frame_bytes_observed",
        "max_arrow_batch_bytes_observed",
    ):
        value = getattr(result, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise M8L2CaptureSystemError(f"capture result {name} must be a nonnegative integer")
    if result.status == "FAILED":
        if result.failure_reason_code is None or result.failure_phase is None:
            raise M8L2CaptureSystemError("FAILED capture result lacks typed failure coordinates")
        _validate_code(result.failure_reason_code, "capture failure reason")
        _validate_code(result.failure_phase, "capture failure phase")
    elif result.failure_reason_code is not None or result.failure_phase is not None:
        raise M8L2CaptureSystemError("COMPLETE capture result cannot carry a failure")

    if (result.first_raw_received_ns is None) != (result.last_raw_received_ns is None):
        raise M8L2CaptureSystemError("raw receipt bounds must be both null or both present")
    if (
        result.first_raw_received_ns is not None
        and result.last_raw_received_ns is not None
        and not (
            session.start_ns
            <= result.first_raw_received_ns
            <= result.last_raw_received_ns
            < session.end_ns
        )
    ):
        raise M8L2CaptureSystemError(
            "raw receipt bounds fall outside the frozen [start, end) session"
        )
    merged = merge_observed_intervals(result.valid_observed_intervals)
    if tuple(result.valid_observed_intervals) != merged:
        raise M8L2CaptureSystemError(
            "OBSERVED intervals must already be canonical, disjoint, and time ordered"
        )
    continuity_ids = {item.continuity_id for item in merged}
    if len(continuity_ids) > result.continuity_epochs:
        raise M8L2CaptureSystemError(
            "OBSERVED continuity IDs exceed the declared reconstruction epochs"
        )
    for interval in merged:
        if not (
            session.start_ns
            <= interval.start_received_ns
            < interval.end_received_ns_exclusive
            <= session.end_ns
        ):
            raise M8L2CaptureSystemError(
                "OBSERVED interval falls outside the frozen [start, end) session"
            )
        if result.first_raw_received_ns is None or result.last_raw_received_ns is None:
            raise M8L2CaptureSystemError("OBSERVED intervals require raw receipt bounds")
        if (
            interval.start_received_ns < result.first_raw_received_ns
            or interval.end_received_ns_exclusive > result.last_raw_received_ns + 1
        ):
            raise M8L2CaptureSystemError("OBSERVED interval exceeds its raw receipt evidence")

    declared: dict[str, CapturedArtifact] = {}
    for artifact in result.artifacts:
        if not artifact.kind:
            raise M8L2CaptureSystemError("captured artifact kind must not be empty")
        if not _is_lower_sha256(artifact.sha256):
            raise M8L2CaptureSystemError("captured artifact digest is not lowercase SHA-256")
        path = _assert_regular_inside(artifact.path, symbol_root, "captured artifact")
        relative = _relative(path, symbol_root)
        if relative in declared:
            raise M8L2CaptureSystemError(f"duplicate captured artifact coordinate: {relative}")
        if sha256_file(path) != artifact.sha256:
            raise M8L2CaptureSystemError(f"captured artifact digest mismatch: {relative}")
        declared[relative] = artifact
    actual = {_relative(path, symbol_root) for path in _all_regular_files(symbol_root)}
    if actual != set(declared):
        raise M8L2CaptureSystemError(
            "single-symbol capture artifact inventory differs from files on disk "
            f"(missing={sorted(actual - set(declared))}, extra={sorted(set(declared) - actual)})"
        )
    if any(item.kind == "capture_summary" for item in result.artifacts):
        _validate_capture_summary(result, symbol_root=symbol_root, session=session)
    return result


def _gate(
    gates: list[dict[str, object]],
    *,
    gate_id: str,
    passed: bool,
    observed: object,
    required: object,
    symbol: str | None = None,
) -> None:
    entry: dict[str, object] = {
        "gate_id": gate_id,
        "passed": passed,
        "observed": observed,
        "required": required,
    }
    if symbol is not None:
        entry["symbol"] = symbol
    gates.append(entry)


def _symbol_payload(result: SymbolCaptureResult, *, stage_root: Path) -> dict[str, object]:
    intervals = merge_observed_intervals(result.valid_observed_intervals)
    return {
        "symbol": result.symbol,
        "capture_id": result.capture_id,
        "status": result.status,
        "completion_reason": result.completion_reason,
        "reconstruction_status": result.reconstruction_status,
        "messages": result.messages,
        "normalized_rows": result.normalized_rows,
        "reconstructed_rows": result.reconstructed_rows,
        "excluded_rows": result.excluded_rows,
        "continuity_epochs": result.continuity_epochs,
        "snapshot_anchors": result.snapshot_anchors,
        "sequence_gaps": result.sequence_gaps,
        "quality_errors": result.quality_errors,
        "quality_warnings": result.quality_warnings,
        "max_raw_frame_bytes_observed": result.max_raw_frame_bytes_observed,
        "max_arrow_batch_bytes_observed": result.max_arrow_batch_bytes_observed,
        "first_raw_received_ns": result.first_raw_received_ns,
        "last_raw_received_ns": result.last_raw_received_ns,
        "valid_observed_intervals": [item.to_dict() for item in intervals],
        "max_valid_continuity_epoch_seconds": max(
            (item.duration_ns for item in intervals), default=0
        )
        / _NANOSECONDS_PER_SECOND,
        "failure_reason_code": result.failure_reason_code,
        "failure_phase": result.failure_phase,
        "artifacts": [
            {
                "path": _relative(artifact.path, stage_root),
                "kind": artifact.kind,
                "sha256": artifact.sha256,
                "bytes": artifact.path.stat().st_size,
            }
            for artifact in sorted(
                result.artifacts, key=lambda item: _relative(item.path, stage_root)
            )
        ],
    }


def _summary_artifact_map(
    result: SymbolCaptureResult, *, symbol_root: Path
) -> dict[str, CapturedArtifact]:
    return {
        _relative(item.path, symbol_root): item
        for item in result.artifacts
        if item.kind != "capture_summary"
    }


def _require_summary_reference(
    value: object,
    *,
    artifacts: Mapping[str, CapturedArtifact],
    kind: str,
    label: str,
) -> CapturedArtifact:
    if type(value) is not str:
        raise M8L2CaptureSystemError(f"capture summary {label} must be a relative path")
    relative = _safe_checksum_relative(value)
    artifact = artifacts.get(relative)
    if artifact is None or artifact.kind != kind:
        raise M8L2CaptureSystemError(
            f"capture summary {label} does not name a declared {kind} artifact"
        )
    return artifact


def _load_compact_json_line(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")

        def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise M8L2CaptureSystemError(f"cannot parse {label}") from error
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2CaptureSystemError(f"{label} must be a JSON object")
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if raw != canonical:
        raise M8L2CaptureSystemError(f"{label} is not canonical compact JSON")
    return cast(Mapping[str, Any], value)


def _validate_raw_journal(
    result: SymbolCaptureResult,
    *,
    journal: CapturedArtifact,
    artifacts: Mapping[str, CapturedArtifact],
    session: M8L2Session,
) -> None:
    websocket_receipts: list[int] = []
    max_frame_bytes = 0
    anchor_ids: set[str] = set()
    anchor_count = 0
    try:
        with journal.path.open("rb") as handle:
            line_number = 0
            while True:
                line = handle.readline(2 * _MAX_JSON_AUTHORITY_BYTES + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > 2 * _MAX_JSON_AUTHORITY_BYTES or not line.endswith(b"\n"):
                    raise M8L2CaptureSystemError("raw journal contains an oversized/partial line")
                entry = _load_compact_json_line(line[:-1], label=f"raw journal line {line_number}")
                event_kind = entry.get("event_kind")
                if event_kind == "websocket_frame":
                    received = entry.get("received_ts_ns")
                    payload_bytes = entry.get("payload_bytes")
                    payload_base64 = entry.get("payload_base64")
                    payload_sha256 = entry.get("payload_sha256")
                    if (
                        type(received) is not int
                        or not session.start_ns <= received < session.end_ns
                        or type(payload_bytes) is not int
                        or payload_bytes < 0
                        or type(payload_base64) is not str
                        or type(payload_sha256) is not str
                        or not _is_lower_sha256(payload_sha256)
                    ):
                        raise M8L2CaptureSystemError("raw websocket lineage has invalid bounds")
                    try:
                        decoded = base64.b64decode(payload_base64, validate=True)
                    except (ValueError, binascii.Error) as error:
                        raise M8L2CaptureSystemError(
                            "raw websocket lineage has invalid base64 bytes"
                        ) from error
                    if len(decoded) != payload_bytes or hashlib.sha256(decoded).hexdigest() != (
                        payload_sha256
                    ):
                        raise M8L2CaptureSystemError(
                            "raw websocket lineage size/digest differs from preserved bytes"
                        )
                    if websocket_receipts and received < websocket_receipts[-1]:
                        raise M8L2CaptureSystemError("raw websocket receipts are not FIFO ordered")
                    websocket_receipts.append(received)
                    max_frame_bytes = max(max_frame_bytes, payload_bytes)
                elif event_kind == "rest_snapshot_anchor":
                    continuity_id = entry.get("continuity_id")
                    if type(continuity_id) is not str or not continuity_id:
                        raise M8L2CaptureSystemError("raw snapshot anchor lacks continuity ID")
                    if continuity_id in anchor_ids:
                        raise M8L2CaptureSystemError(
                            "raw journal repeats a snapshot anchor continuity ID"
                        )
                    anchor_ids.add(continuity_id)
                    anchor_count += 1
                    raw_snapshot = _require_summary_reference(
                        entry.get("raw_path"),
                        artifacts=artifacts,
                        kind="raw_snapshot",
                        label="raw journal snapshot path",
                    )
                    raw_snapshot_manifest = _require_summary_reference(
                        entry.get("raw_manifest_path"),
                        artifacts=artifacts,
                        kind="raw_snapshot_manifest",
                        label="raw journal snapshot manifest path",
                    )
                    if (
                        entry.get("raw_sha256") != raw_snapshot.sha256
                        or entry.get("raw_manifest_sha256") != raw_snapshot_manifest.sha256
                    ):
                        raise M8L2CaptureSystemError("raw snapshot anchor digest is inconsistent")
                else:
                    raise M8L2CaptureSystemError("raw journal has an unsupported event kind")
    except OSError as error:
        raise M8L2CaptureSystemError("cannot stream the raw journal authority") from error
    if len(websocket_receipts) != result.messages:
        raise M8L2CaptureSystemError("raw journal message count differs from returned claims")
    observed_first = websocket_receipts[0] if websocket_receipts else None
    observed_last = websocket_receipts[-1] if websocket_receipts else None
    if (
        observed_first != result.first_raw_received_ns
        or observed_last != result.last_raw_received_ns
        or max_frame_bytes != result.max_raw_frame_bytes_observed
    ):
        raise M8L2CaptureSystemError("raw journal receipt/frame claims are inconsistent")
    if anchor_count != result.snapshot_anchors or anchor_count != result.continuity_epochs:
        raise M8L2CaptureSystemError("raw journal anchors differ from reconstruction epochs")
    if not {item.continuity_id for item in result.valid_observed_intervals}.issubset(anchor_ids):
        raise M8L2CaptureSystemError("OBSERVED intervals lack a matching raw snapshot anchor")


def _validate_capture_summary(
    result: SymbolCaptureResult, *, symbol_root: Path, session: M8L2Session
) -> None:
    summaries = [item for item in result.artifacts if item.kind == "capture_summary"]
    if len(summaries) != 1:
        raise M8L2CaptureSystemError("capture result must declare exactly one capture summary")
    summary_artifact = summaries[0]
    if _relative(summary_artifact.path, symbol_root) != "quality/capture.summary.json":
        raise M8L2CaptureSystemError("capture summary path is not the canonical coordinate")
    payload = _strict_json(summary_artifact.path, label="symbol capture summary")
    if not isinstance(payload, Mapping) or not all(type(key) is str for key in payload):
        raise M8L2CaptureSystemError("symbol capture summary must be a JSON object")
    summary = cast(Mapping[str, Any], payload)
    expected_claims: dict[str, object] = {
        "schema_version": _SYMBOL_SUMMARY_SCHEMA_VERSION,
        "symbol": result.symbol,
        "capture_id": result.capture_id,
        "capture_status": result.status,
        "completion_reason": result.completion_reason,
        "reconstruction_status": result.reconstruction_status,
        "failure_reason_code": result.failure_reason_code,
        "failure_phase": result.failure_phase,
        "messages": result.messages,
        "normalized_rows": result.normalized_rows,
        "reconstructed_rows": result.reconstructed_rows,
        "excluded_rows": result.excluded_rows,
        "continuity_epochs": result.continuity_epochs,
        "snapshot_anchors": result.snapshot_anchors,
        "sequence_gaps": result.sequence_gaps,
        "quality_errors": result.quality_errors,
        "quality_warnings": result.quality_warnings,
        "max_raw_frame_bytes_observed": result.max_raw_frame_bytes_observed,
        "max_arrow_batch_bytes_observed": result.max_arrow_batch_bytes_observed,
        "first_raw_received_ns": result.first_raw_received_ns,
        "last_raw_received_ns": result.last_raw_received_ns,
        "valid_observed_intervals": [item.to_dict() for item in result.valid_observed_intervals],
        "scheduled_range_ns": {
            "start": session.start_ns,
            "end_exclusive": session.end_ns,
        },
    }
    mismatches = [
        name for name, expected in expected_claims.items() if summary.get(name) != expected
    ]
    if mismatches:
        raise M8L2CaptureSystemError(
            "symbol capture summary differs from returned claims: " + ", ".join(mismatches)
        )

    artifacts = _summary_artifact_map(result, symbol_root=symbol_root)
    inventory_raw = summary.get("artifact_inventory_without_summary")
    if not isinstance(inventory_raw, list):
        raise M8L2CaptureSystemError(
            "symbol capture summary lacks artifact_inventory_without_summary"
        )
    summary_inventory: dict[str, tuple[str, str, int]] = {}
    for index, raw in enumerate(inventory_raw):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "kind", "sha256", "bytes"}:
            raise M8L2CaptureSystemError(
                f"symbol capture summary inventory entry {index} is malformed"
            )
        relative_value = raw.get("path")
        if type(relative_value) is not str:
            raise M8L2CaptureSystemError("symbol capture summary inventory path is invalid")
        relative = _safe_checksum_relative(relative_value)
        kind = raw.get("kind")
        digest = raw.get("sha256")
        size = raw.get("bytes")
        if (
            type(kind) is not str
            or type(digest) is not str
            or not _is_lower_sha256(digest)
            or type(size) is not int
            or size < 0
            or relative in summary_inventory
        ):
            raise M8L2CaptureSystemError("symbol capture summary inventory entry is invalid")
        summary_inventory[relative] = (kind, digest, size)
    expected_inventory = {
        relative: (item.kind, item.sha256, item.path.stat().st_size)
        for relative, item in artifacts.items()
    }
    if summary_inventory != expected_inventory or list(summary_inventory) != sorted(
        summary_inventory
    ):
        raise M8L2CaptureSystemError(
            "symbol capture summary artifact inventory is not exact and canonical"
        )

    raw_journal = _require_summary_reference(
        summary.get("raw_journal"),
        artifacts=artifacts,
        kind="raw_journal",
        label="raw_journal",
    )
    raw_manifest = _require_summary_reference(
        summary.get("raw_journal_manifest"),
        artifacts=artifacts,
        kind="raw_journal_manifest",
        label="raw_journal_manifest",
    )
    if (
        summary.get("raw_journal_sha256") != raw_journal.sha256
        or summary.get("raw_journal_manifest_sha256") != raw_manifest.sha256
    ):
        raise M8L2CaptureSystemError("symbol capture summary raw digests are inconsistent")
    _validate_raw_journal(
        result,
        journal=raw_journal,
        artifacts=artifacts,
        session=session,
    )
    raw_manifest_payload = _strict_json(raw_manifest.path, label="raw journal manifest")
    if not isinstance(raw_manifest_payload, Mapping):
        raise M8L2CaptureSystemError("raw journal manifest must be a JSON object")
    raw_checksum = raw_manifest_payload.get("checksum")
    raw_range = raw_manifest_payload.get("requested_range_ns")
    raw_headers = raw_manifest_payload.get("response_headers")
    if (
        raw_manifest_payload.get("artifact_kind") != "raw_source"
        or not isinstance(raw_checksum, Mapping)
        or raw_checksum.get("algorithm") != "sha256"
        or raw_checksum.get("value") != raw_journal.sha256
        or raw_manifest_payload.get("bytes") != raw_journal.path.stat().st_size
        or raw_manifest_payload.get("path") != raw_journal.path.name
        or raw_range != {"start": session.start_ns, "end_exclusive": session.end_ns}
        or not isinstance(raw_headers, Mapping)
        or raw_headers.get("x-local-message-count") != str(result.messages)
        or raw_headers.get("x-local-snapshot-anchor-count") != str(result.snapshot_anchors)
    ):
        raise M8L2CaptureSystemError("raw journal manifest differs from capture claims")

    datasets_raw = summary.get("normalized_dataset_manifests")
    expected_rows = {
        "book_snapshots": result.snapshot_anchors,
        "depth_deltas": result.normalized_rows,
        "book_observations": result.reconstructed_rows,
        "sequence_gaps": result.sequence_gaps,
    }
    if not isinstance(datasets_raw, Mapping) or set(datasets_raw) != set(expected_rows):
        raise M8L2CaptureSystemError("symbol capture summary normalized manifests are incomplete")
    for dataset, expected_row_count in expected_rows.items():
        raw_entry = datasets_raw[dataset]
        if not isinstance(raw_entry, Mapping):
            raise M8L2CaptureSystemError("normalized dataset summary entry must be an object")
        manifest = _require_summary_reference(
            raw_entry.get("manifest_path"),
            artifacts=artifacts,
            kind="normalized_manifest",
            label=f"normalized_dataset_manifests.{dataset}.manifest_path",
        )
        if raw_entry.get("manifest_sha256") != manifest.sha256:
            raise M8L2CaptureSystemError("normalized dataset manifest digest is inconsistent")
        if raw_entry.get("rows") != expected_row_count:
            raise M8L2CaptureSystemError("normalized dataset row count differs from claims")
        data_path = raw_entry.get("data_path")
        data_sha = raw_entry.get("data_sha256")
        if expected_row_count == 0:
            if data_path is not None or data_sha is not None:
                raise M8L2CaptureSystemError("empty normalized dataset unexpectedly has data")
        else:
            data = _require_summary_reference(
                data_path,
                artifacts=artifacts,
                kind="normalized_data",
                label=f"normalized_dataset_manifests.{dataset}.data_path",
            )
            if data_sha != data.sha256:
                raise M8L2CaptureSystemError("normalized dataset digest is inconsistent")
            _validate_normalized_parquet(
                data.path,
                dataset=dataset,
                expected_rows=expected_row_count,
            )
        normalized_manifest_payload = _strict_json(
            manifest.path, label=f"{dataset} normalized manifest"
        )
        if not isinstance(normalized_manifest_payload, Mapping):
            raise M8L2CaptureSystemError("normalized dataset manifest must be an object")
        if (
            normalized_manifest_payload.get("dataset") != dataset
            or normalized_manifest_payload.get("rows") != expected_row_count
            or normalized_manifest_payload.get("requested_range_ns")
            != {"start": session.start_ns, "end_exclusive": session.end_ns}
        ):
            raise M8L2CaptureSystemError("normalized manifest essentials differ from claims")

    reports_raw = summary.get("quality_reports")
    if not isinstance(reports_raw, Mapping) or set(reports_raw) != {
        "depth_deltas",
        "book_observations",
    }:
        raise M8L2CaptureSystemError("symbol capture summary quality reports are incomplete")
    report_counts = {"errors": 0, "warnings": 0}
    for dataset in ("depth_deltas", "book_observations"):
        report = _require_summary_reference(
            reports_raw[dataset],
            artifacts=artifacts,
            kind="quality_report",
            label=f"quality_reports.{dataset}",
        )
        report_payload = _strict_json(report.path, label=f"{dataset} quality report")
        if not isinstance(report_payload, Mapping) or report_payload.get("dataset") != dataset:
            raise M8L2CaptureSystemError("quality report dataset differs from its coordinate")
        report_summary = report_payload.get("summary")
        if not isinstance(report_summary, Mapping):
            raise M8L2CaptureSystemError("quality report lacks its summary counts")
        for name in report_counts:
            value = report_summary.get(name)
            if type(value) is not int or value < 0:
                raise M8L2CaptureSystemError("quality report count is invalid")
            report_counts[name] += value
    if report_counts != {"errors": result.quality_errors, "warnings": result.quality_warnings}:
        raise M8L2CaptureSystemError("quality report counts differ from returned claims")


def _validate_normalized_parquet(path: Path, *, dataset: str, expected_rows: int) -> None:
    """Bound footer reads before PyArrow metadata/schema validation."""

    try:
        size = path.stat().st_size
        if size < 12:
            raise M8L2CaptureSystemError("normalized data is too short to be Parquet")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise M8L2CaptureSystemError("normalized data is not a regular file")
            if os.pread(descriptor, 4, 0) != b"PAR1":
                raise M8L2CaptureSystemError("normalized data lacks the Parquet header")
            trailer = os.pread(descriptor, 8, size - 8)
            if len(trailer) != 8 or trailer[4:] != b"PAR1":
                raise M8L2CaptureSystemError("normalized data lacks the Parquet trailer")
            footer_bytes = int.from_bytes(trailer[:4], "little")
            if footer_bytes > _MAX_JSON_AUTHORITY_BYTES or footer_bytes + 12 > size:
                raise M8L2CaptureSystemError("normalized Parquet footer exceeds its bound")
        finally:
            os.close(descriptor)
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_rows:
            raise M8L2CaptureSystemError("Parquet footer row count differs from capture claims")
        if parquet.schema_arrow != get_schema(dataset):
            raise M8L2CaptureSystemError("normalized Parquet schema differs from the registry")
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b"schema_version") != SCHEMA_VERSION.encode():
            raise M8L2CaptureSystemError("normalized Parquet schema version is not frozen")
    except M8L2CaptureSystemError:
        raise
    except (OSError, ValueError) as error:
        raise M8L2CaptureSystemError("cannot validate normalized Parquet metadata") from error


def _evaluate_gates(
    results: Mapping[str, SymbolCaptureResult],
    *,
    config: M8L2StudyConfig,
) -> tuple[list[dict[str, object]], list[str], int]:
    limits = config.capture
    gates: list[dict[str, object]] = []
    reasons: list[str] = []
    for symbol in config.study.symbols:
        result = results.get(symbol)
        if result is None:
            reasons.append(f"MISSING_SYMBOL_{symbol}")
            _gate(
                gates,
                gate_id="SYMBOL_RESULT_PRESENT",
                passed=False,
                observed=False,
                required=True,
                symbol=symbol,
            )
            continue
        artifact_kinds = {artifact.kind for artifact in result.artifacts}
        snapshot_artifact_count = sum(
            artifact.kind == "raw_snapshot" for artifact in result.artifacts
        )
        snapshot_manifest_count = sum(
            artifact.kind == "raw_snapshot_manifest" for artifact in result.artifacts
        )
        checks: tuple[tuple[str, bool, object, object], ...] = (
            (
                "CAPTURE_STATUS_COMPLETE",
                (not limits.require_complete_status) or result.status == "COMPLETE",
                result.status,
                "COMPLETE",
            ),
            (
                "SCHEDULED_END_REACHED",
                result.completion_reason == "scheduled_end_reached",
                result.completion_reason,
                "scheduled_end_reached",
            ),
            (
                "RAW_RECEIPT_BOUNDS_PRESENT",
                result.first_raw_received_ns is not None
                and result.last_raw_received_ns is not None,
                {
                    "first": result.first_raw_received_ns,
                    "last": result.last_raw_received_ns,
                },
                "both present inside [start,end)",
            ),
            (
                "MESSAGE_CEILING",
                result.messages <= limits.max_messages_per_symbol,
                result.messages,
                {"maximum": limits.max_messages_per_symbol},
            ),
            (
                "MESSAGE_NORMALIZATION_RECONCILIATION",
                result.messages == result.normalized_rows
                and result.normalized_rows == result.reconstructed_rows + result.excluded_rows,
                {
                    "messages": result.messages,
                    "normalized": result.normalized_rows,
                    "reconstructed": result.reconstructed_rows,
                    "excluded": result.excluded_rows,
                },
                "messages == normalized == reconstructed + excluded",
            ),
            (
                "SNAPSHOT_ANCHOR_RECONCILIATION",
                result.continuity_epochs > 0
                and result.snapshot_anchors == result.continuity_epochs,
                {
                    "epochs": result.continuity_epochs,
                    "snapshot_anchors": result.snapshot_anchors,
                },
                "positive epochs and one anchor per epoch",
            ),
            (
                "LIVE_RECONSTRUCTION",
                (not limits.require_live_reconstruction) or result.reconstruction_status == "LIVE",
                result.reconstruction_status,
                "LIVE",
            ),
            (
                "SEQUENCE_GAPS",
                result.sequence_gaps <= limits.max_sequence_gaps,
                result.sequence_gaps,
                {"maximum": limits.max_sequence_gaps},
            ),
            (
                "QUALITY_ERRORS",
                result.quality_errors <= limits.max_quality_errors,
                result.quality_errors,
                {"maximum": limits.max_quality_errors},
            ),
            (
                "QUALITY_WARNINGS",
                result.quality_warnings <= limits.max_quality_warnings,
                result.quality_warnings,
                {"maximum": limits.max_quality_warnings},
            ),
            (
                "RAW_FRAME_BYTES",
                result.max_raw_frame_bytes_observed <= limits.max_raw_frame_bytes,
                result.max_raw_frame_bytes_observed,
                {"maximum": limits.max_raw_frame_bytes},
            ),
            (
                "ARROW_BATCH_BYTES",
                result.max_arrow_batch_bytes_observed <= limits.max_arrow_batch_bytes,
                result.max_arrow_batch_bytes_observed,
                {"maximum": limits.max_arrow_batch_bytes},
            ),
            (
                "VALID_CONTINUITY_EPOCH",
                max(
                    (
                        item.duration_ns
                        for item in merge_observed_intervals(result.valid_observed_intervals)
                    ),
                    default=0,
                )
                >= limits.min_single_continuity_epoch_seconds * _NANOSECONDS_PER_SECOND,
                max(
                    (
                        item.duration_ns
                        for item in merge_observed_intervals(result.valid_observed_intervals)
                    ),
                    default=0,
                )
                / _NANOSECONDS_PER_SECOND,
                {"minimum_seconds": limits.min_single_continuity_epoch_seconds},
            ),
            (
                "CAPTURE_ARTIFACTS_PRESENT",
                _REQUIRED_CAPTURE_ARTIFACT_KINDS.issubset(artifact_kinds),
                {
                    "kinds": sorted(artifact_kinds),
                    "raw_snapshots": snapshot_artifact_count,
                    "raw_snapshot_manifests": snapshot_manifest_count,
                },
                {
                    "required_kinds": sorted(_REQUIRED_CAPTURE_ARTIFACT_KINDS),
                    "snapshot anchors are cross-bound in the raw journal": True,
                },
            ),
        )
        if result.failure_reason_code is not None:
            reasons.append(result.failure_reason_code)
        for gate_id, passed, observed, required in checks:
            _gate(
                gates,
                gate_id=gate_id,
                passed=passed,
                observed=observed,
                required=required,
                symbol=symbol,
            )
            if not passed:
                reasons.append(f"GATE_{gate_id}_{symbol}")

    overlap_ns = 0
    if all(symbol in results for symbol in config.study.symbols):
        left = results[config.study.symbols[0]].valid_observed_intervals
        right = results[config.study.symbols[1]].valid_observed_intervals
        overlap_ns = overlapping_observed_coverage_ns(left, right)
    overlap_passed = overlap_ns >= limits.min_overlapping_coverage_seconds * _NANOSECONDS_PER_SECOND
    _gate(
        gates,
        gate_id="CROSS_SYMBOL_OBSERVED_OVERLAP",
        passed=overlap_passed,
        observed=overlap_ns / _NANOSECONDS_PER_SECOND,
        required={"minimum_seconds": limits.min_overlapping_coverage_seconds},
    )
    if not overlap_passed:
        reasons.append("GATE_CROSS_SYMBOL_OBSERVED_OVERLAP")
    return gates, sorted(set(reasons)), overlap_ns


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _campaign_payload(
    *,
    config: M8L2StudyConfig,
    source: CaptureSourceIdentity,
    runtime: _RuntimeFingerprint,
    nonce: str,
    root: Path,
    root_identity: tuple[int, int],
) -> dict[str, object]:
    _validate_source_identity(source)
    if not _is_lower_sha256(nonce):
        raise M8L2CaptureSystemError("live-L2 campaign nonce is not 256-bit lowercase hex")
    if runtime != _runtime_from_recorded(runtime.payload(), runtime.sha256):
        raise M8L2CaptureSystemError("live-L2 runtime fingerprint is not canonical")
    return {
        "schema_version": _CAMPAIGN_SCHEMA_VERSION,
        "artifact_kind": "m8_prospective_live_l2_campaign_authority",
        "campaign_nonce": nonce,
        "output_root": {
            "canonical_path": str(root),
            "device": root_identity[0],
            "inode": root_identity[1],
        },
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_sha256": M8_L2_PROTOCOL_SHA256,
        "protocol_freeze_commit": M8_L2_FREEZE_COMMIT,
        "runtime_commit": source.commit,
        "runtime_source_tree_sha256": source.source_tree_sha256,
        "runtime_dirty": False,
        "runtime_fingerprint": runtime.payload(),
        "runtime_fingerprint_sha256": runtime.sha256,
    }


def _recorded_output_root(value: object) -> tuple[str, int, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "canonical_path",
        "device",
        "inode",
    }:
        raise M8L2CaptureSystemError("campaign output-root identity is not exact")
    canonical_path = value.get("canonical_path")
    device = value.get("device")
    inode = value.get("inode")
    if (
        type(canonical_path) is not str
        or not canonical_path
        or "\x00" in canonical_path
        or not PurePosixPath(canonical_path).is_absolute()
        or ".." in PurePosixPath(canonical_path).parts
        or PurePosixPath(canonical_path).as_posix() != canonical_path
    ):
        raise M8L2CaptureSystemError("campaign output-root path is not canonical")
    if type(device) is not int or device < 0 or type(inode) is not int or inode < 0:
        raise M8L2CaptureSystemError("campaign output-root device/inode is malformed")
    return canonical_path, device, inode


def _validate_campaign_bytes(
    raw: bytes,
    *,
    path: Path,
    config: M8L2StudyConfig,
    source: CaptureSourceIdentity,
    expected_runtime: _RuntimeFingerprint | None = None,
    expected_root: Path | None = None,
    expected_root_identity: tuple[int, int] | None = None,
    expected_sha256: str | None = None,
) -> _CampaignAuthority:
    value = _decode_canonical_json(raw, label=f"live-L2 campaign authority: {path}")
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "artifact_kind",
        "campaign_nonce",
        "output_root",
        "config_sha256",
        "config_source_sha256",
        "protocol_sha256",
        "protocol_freeze_commit",
        "runtime_commit",
        "runtime_source_tree_sha256",
        "runtime_dirty",
        "runtime_fingerprint",
        "runtime_fingerprint_sha256",
    }:
        raise M8L2CaptureSystemError("live-L2 campaign authority schema is not exact")
    expected_claims = {
        "schema_version": _CAMPAIGN_SCHEMA_VERSION,
        "artifact_kind": "m8_prospective_live_l2_campaign_authority",
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_sha256": M8_L2_PROTOCOL_SHA256,
        "protocol_freeze_commit": M8_L2_FREEZE_COMMIT,
        "runtime_commit": source.commit,
        "runtime_source_tree_sha256": source.source_tree_sha256,
        "runtime_dirty": False,
    }
    if any(value.get(name) != expected for name, expected in expected_claims.items()):
        raise M8L2CaptureSystemError(
            "live-L2 campaign authority differs from the frozen configuration/source identity"
        )
    nonce = value.get("campaign_nonce")
    if type(nonce) is not str or not _is_lower_sha256(nonce):
        raise M8L2CaptureSystemError("live-L2 campaign nonce is malformed")
    output_path, output_device, output_inode = _recorded_output_root(value.get("output_root"))
    if (expected_root is None) != (expected_root_identity is None):
        raise M8L2CaptureSystemError("campaign root validation coordinates are incomplete")
    if (
        expected_root is not None
        and expected_root_identity is not None
        and (
            output_path != str(expected_root)
            or (output_device, output_inode) != expected_root_identity
        )
    ):
        raise M8L2CaptureSystemError(
            "live-L2 campaign authority differs from the current output-root identity"
        )
    runtime = _runtime_from_recorded(
        value.get("runtime_fingerprint"), value.get("runtime_fingerprint_sha256")
    )
    if expected_runtime is not None and runtime != expected_runtime:
        raise M8L2CaptureSystemError(
            "live-L2 campaign authority differs from the current production runtime"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise M8L2CaptureSystemError("live-L2 campaign authority digest changed")
    return _CampaignAuthority(
        path=path,
        sha256=digest,
        raw=raw,
        source=source,
        runtime=runtime,
        nonce=nonce,
        output_root_path=output_path,
        output_root_device=output_device,
        output_root_inode=output_inode,
    )


def _verify_campaign_authority(
    root: Path,
    *,
    root_identity: tuple[int, int],
    config: M8L2StudyConfig,
    source: CaptureSourceIdentity,
    runtime: _RuntimeFingerprint,
    expected_sha256: str | None = None,
) -> _CampaignAuthority:
    if _directory_identity(root, label="live-L2 output root") != root_identity:
        raise M8L2CaptureSystemError("live-L2 output root identity changed")
    path = root / _CAMPAIGN_AUTHORITY_NAME
    _assert_no_symlink_components(path, root, "live-L2 campaign authority")
    raw = _read_bounded_regular_nofollow(
        path,
        label="live-L2 campaign authority",
        maximum_bytes=_MAX_CAMPAIGN_AUTHORITY_BYTES,
    )
    if _directory_identity(root, label="live-L2 output root") != root_identity:
        raise M8L2CaptureSystemError("live-L2 output root changed during campaign verification")
    return _validate_campaign_bytes(
        raw,
        path=path,
        config=config,
        source=source,
        expected_runtime=runtime,
        expected_root=root,
        expected_root_identity=root_identity,
        expected_sha256=expected_sha256,
    )


def _create_campaign_authority_once(
    root: Path, *, root_identity: tuple[int, int], raw: bytes
) -> None:
    """Atomically link one fsynced authority into place without replacement."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=".campaign-authority-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    root_descriptor = -1
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_descriptor = os.open(root, directory_flags)
        root_metadata = os.fstat(root_descriptor)
        if (root_metadata.st_dev, root_metadata.st_ino) != root_identity:
            raise M8L2CaptureSystemError(
                "live-L2 output root changed before campaign authority creation"
            )
        with suppress(FileExistsError):
            os.link(
                temporary.name,
                _CAMPAIGN_AUTHORITY_NAME,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        # This fsync makes either our new link or a concurrent creator's link
        # durable before either caller may proceed toward a network connection.
        os.fsync(root_descriptor)
        os.unlink(temporary.name, dir_fd=root_descriptor)
        temporary = Path()
        os.fsync(root_descriptor)
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)
        if temporary != Path():
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _ensure_campaign_authority(
    root: Path,
    *,
    root_identity: tuple[int, int],
    config: M8L2StudyConfig,
    source: CaptureSourceIdentity,
    runtime: _RuntimeFingerprint,
) -> _CampaignAuthority:
    path = root / _CAMPAIGN_AUTHORITY_NAME
    try:
        path.lstat()
    except FileNotFoundError:
        _create_campaign_authority_once(
            root,
            root_identity=root_identity,
            raw=_canonical_json_bytes(
                _campaign_payload(
                    config=config,
                    source=source,
                    runtime=runtime,
                    nonce=secrets.token_hex(32),
                    root=root,
                    root_identity=root_identity,
                )
            ),
        )
    except OSError as error:
        raise M8L2CaptureSystemError(
            f"cannot inspect live-L2 campaign authority: {path}"
        ) from error
    return _verify_campaign_authority(
        root,
        root_identity=root_identity,
        config=config,
        source=source,
        runtime=runtime,
    )


def _verify_bundled_campaign_authority(
    path: Path,
    *,
    config: M8L2StudyConfig,
    source: CaptureSourceIdentity,
    expected_sha256: str,
) -> _CampaignAuthority:
    raw = _read_bounded_regular_nofollow(
        path,
        label="bundled live-L2 campaign authority",
        maximum_bytes=_MAX_CAMPAIGN_AUTHORITY_BYTES,
    )
    return _validate_campaign_bytes(
        raw,
        path=path,
        config=config,
        source=source,
        expected_sha256=expected_sha256,
    )


def _fsync_tree(root: Path) -> None:
    files, directories = _walk_regular_evidence(root)
    for path in files:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise M8L2CaptureSystemError(f"cannot fsync non-regular evidence: {path}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in _all_regular_files(root):
        relative = _relative(path, root)
        if relative in exclude:
            continue
        result.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return result


def _write_checksums(stage: Path) -> Path:
    protected = _inventory(
        stage,
        exclude=frozenset({_CHECKSUM_NAME, _COMPLETE_MARKER, _INSUFFICIENT_MARKER}),
    )
    content = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in protected).encode()
    path = stage / _CHECKSUM_NAME
    _write_bytes_durable(path, content)
    return path


def _publish_terminal(
    *,
    stage: Path,
    target: Path,
    config: M8L2StudyConfig,
    protocol: Path,
    session: M8L2Session,
    session_id: str,
    source: CaptureSourceIdentity,
    runtime: _RuntimeFingerprint,
    source_identity_was_injected: bool,
    campaign: _CampaignAuthority,
    status: SessionStatus,
    reason_codes: Sequence[str],
    phase_ledger: Sequence[Mapping[str, object]],
    gates: Sequence[Mapping[str, object]],
    results: Mapping[str, SymbolCaptureResult],
    barrier_released_ns: int | None,
    overlap_ns: int,
    capture_finished_ns: int | None,
    root_identity: tuple[int, int],
    sessions_identity: tuple[int, int],
) -> M8L2SessionBundle:
    observed_campaign = _verify_campaign_authority(
        target.parent.parent,
        root_identity=root_identity,
        config=config,
        source=source,
        runtime=runtime,
        expected_sha256=campaign.sha256,
    )
    if observed_campaign.raw != campaign.raw:
        raise M8L2CaptureSystemError("live-L2 campaign authority bytes changed")
    _verify_bundled_campaign_authority(
        stage / "authority" / _CAMPAIGN_AUTHORITY_NAME,
        config=config,
        source=source,
        expected_sha256=campaign.sha256,
    )
    authority_inventory = _inventory(stage)
    marker_name = _COMPLETE_MARKER if status == "COMPLETE" else _INSUFFICIENT_MARKER
    manifest_payload: dict[str, object] = {
        "schema_version": _SESSION_SCHEMA_VERSION,
        "artifact_kind": "m8_prospective_live_l2_session",
        "generated_at_utc": utc_now_iso(),
        "status": status,
        "session_id": session_id,
        "study_name": config.study.name,
        "protocol_version": config.study.protocol_version,
        "evidence_tier": config.study.evidence_tier,
        "live_trading": False,
        "session": {
            "date": session.date.isoformat(),
            "role": session.role,
            "scheduled_start_ns": session.start_ns,
            "scheduled_end_ns": session.end_ns,
            "scheduled_duration_seconds": (session.end_ns - session.start_ns)
            / _NANOSECONDS_PER_SECOND,
            "barrier_released_ns": barrier_released_ns,
            "barrier_lateness_ns": (
                max(0, barrier_released_ns - session.start_ns)
                if barrier_released_ns is not None
                else None
            ),
            "capture_finished_ns": capture_finished_ns,
        },
        "authority": {
            "campaign_authority_sha256": campaign.sha256,
            "config_sha256": config.hash,
            "config_source_sha256": config.source_sha256,
            "protocol_sha256": M8_L2_PROTOCOL_SHA256,
            "protocol_freeze_commit": M8_L2_FREEZE_COMMIT,
            "runtime_commit": source.commit,
            "runtime_source_tree_sha256": source.source_tree_sha256,
            "runtime_dirty": source.dirty,
            "runtime_fingerprint": campaign.runtime.payload(),
            "runtime_fingerprint_sha256": campaign.runtime.sha256,
        },
        "symbols": {
            symbol: _symbol_payload(result, stage_root=stage)
            for symbol, result in sorted(results.items())
        },
        "cross_symbol_observed_overlap_seconds": overlap_ns / _NANOSECONDS_PER_SECOND,
        "gates": [dict(item) for item in gates],
        "reason_codes": list(sorted(set(reason_codes))),
        "phase_ledger": [dict(item) for item in phase_ledger],
        "artifact_inventory": authority_inventory,
        "terminal_marker": {
            "path": marker_name,
            "bytes": "complete\\n" if status == "COMPLETE" else "terminal\\n",
        },
        "policy": (
            "both symbols share one frozen session authority; only OBSERVED continuity intervals "
            "count toward overlap; failed gates publish no research result"
        ),
    }
    manifest_path = stage / "session_manifest.json"
    write_json(manifest_path, manifest_payload)
    _write_checksums(stage)
    _fsync_tree(stage)
    _revalidate_runtime_authority(
        config=config,
        protocol=protocol,
        expected_source=source,
        expected_runtime=runtime,
        source_identity_was_injected=source_identity_was_injected,
    )
    marker_campaign = _verify_campaign_authority(
        target.parent.parent,
        root_identity=root_identity,
        config=config,
        source=source,
        runtime=runtime,
        expected_sha256=campaign.sha256,
    )
    if marker_campaign.raw != campaign.raw:
        raise M8L2CaptureSystemError(
            "live-L2 campaign authority changed immediately before marker materialization"
        )
    marker_path = stage / marker_name
    _write_bytes_durable(
        marker_path,
        _COMPLETE_BYTES if status == "COMPLETE" else _INSUFFICIENT_BYTES,
    )
    _fsync_directory(stage)
    _revalidate_output_layout(
        target.parent.parent,
        root_identity=root_identity,
        sessions_identity=sessions_identity,
        target=target,
    )
    if target.exists() or target.is_symlink():
        raise M8L2CaptureSystemError(f"refusing to overwrite existing session authority: {target}")
    source_parent = stage.parent
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_parent_descriptor = os.open(source_parent, directory_flags)
    target_parent_descriptor = os.open(target.parent, directory_flags)
    try:
        if (
            os.fstat(source_parent_descriptor).st_dev,
            os.fstat(source_parent_descriptor).st_ino,
        ) != (root_identity):
            raise M8L2CaptureSystemError("live-L2 stage parent identity changed before rename")
        if (
            os.fstat(target_parent_descriptor).st_dev,
            os.fstat(target_parent_descriptor).st_ino,
        ) != sessions_identity:
            raise M8L2CaptureSystemError("live-L2 target parent identity changed before rename")
        _revalidate_runtime_authority(
            config=config,
            protocol=protocol,
            expected_source=source,
            expected_runtime=runtime,
            source_identity_was_injected=source_identity_was_injected,
        )
        final_campaign = _verify_campaign_authority(
            target.parent.parent,
            root_identity=root_identity,
            config=config,
            source=source,
            runtime=runtime,
            expected_sha256=campaign.sha256,
        )
        if final_campaign.raw != campaign.raw:
            raise M8L2CaptureSystemError(
                "live-L2 campaign authority changed immediately before terminal publication"
            )
        os.rename(
            stage.name,
            target.name,
            src_dir_fd=source_parent_descriptor,
            dst_dir_fd=target_parent_descriptor,
        )
        # A cross-directory rename mutates both directory entries.  Durability
        # therefore requires both the old and the new parent directory fsyncs.
        os.fsync(source_parent_descriptor)
        os.fsync(target_parent_descriptor)
    finally:
        os.close(target_parent_descriptor)
        os.close(source_parent_descriptor)
    _revalidate_output_layout(
        target.parent.parent,
        root_identity=root_identity,
        sessions_identity=sessions_identity,
        target=target,
    )
    return verify_m8_l2_session_bundle(target, expected_config=config)


def _retain_system_failure(
    *,
    stage: Path,
    output_root: Path,
    session_id: str,
    session: M8L2Session,
    error: BaseException,
) -> Path:
    evidence_root = stage
    try:
        inventory = _inventory(stage)
        write_json(
            stage / "SYSTEM_FAILURE.json",
            {
                "schema_version": _SESSION_SCHEMA_VERSION,
                "artifact_kind": "m8_live_l2_nonterminal_system_failure",
                "generated_at_utc": utc_now_iso(),
                "terminal": False,
                "research_result": False,
                "session_id": session_id,
                "session_date": session.date.isoformat(),
                "role": session.role,
                "error_type": type(error).__name__,
                "error": str(error)[:2048],
                "preserved_inventory_before_record": inventory,
                "policy": "incomplete evidence only; no terminal marker and no research reuse",
            },
        )
        _fsync_tree(stage)
        incomplete_root = output_root / "incomplete"
        _reject_existing_symlink_components(
            incomplete_root, label="live-L2 incomplete-evidence directory"
        )
        incomplete_root.mkdir(parents=True, exist_ok=True)
        _directory_identity(incomplete_root, label="live-L2 incomplete-evidence directory")
        destination = incomplete_root / f"SYSTEM_FAILURE-{session_id[:20]}-{stage.name[-12:]}"
        if destination.exists():
            raise FileExistsError(destination)
        old_parent = stage.parent
        os.rename(stage, destination)
        _fsync_directory(old_parent)
        _fsync_directory(incomplete_root)
        evidence_root = destination
    except BaseException:
        # The original stage is intentionally never deleted; it may contain the
        # sole copy of already-received raw frames.
        evidence_root = stage
    return evidence_root


def _publish_or_retain_system_failure(
    *,
    stage: Path,
    target: Path,
    output_root: Path,
    config: M8L2StudyConfig,
    protocol: Path,
    session: M8L2Session,
    session_id: str,
    source: CaptureSourceIdentity,
    runtime: _RuntimeFingerprint,
    source_identity_was_injected: bool,
    campaign: _CampaignAuthority,
    status: SessionStatus,
    reason_codes: Sequence[str],
    phase_ledger: Sequence[Mapping[str, object]],
    gates: Sequence[Mapping[str, object]],
    results: Mapping[str, SymbolCaptureResult],
    barrier_released_ns: int | None,
    overlap_ns: int,
    capture_finished_ns: int | None,
    root_identity: tuple[int, int],
    sessions_identity: tuple[int, int],
) -> M8L2SessionBundle:
    """Publish terminally or demote every publication fault to preserved evidence."""

    try:
        return _publish_terminal(
            stage=stage,
            target=target,
            config=config,
            protocol=protocol,
            session=session,
            session_id=session_id,
            source=source,
            runtime=runtime,
            source_identity_was_injected=source_identity_was_injected,
            campaign=campaign,
            status=status,
            reason_codes=reason_codes,
            phase_ledger=phase_ledger,
            gates=gates,
            results=results,
            barrier_released_ns=barrier_released_ns,
            overlap_ns=overlap_ns,
            capture_finished_ns=capture_finished_ns,
            root_identity=root_identity,
            sessions_identity=sessions_identity,
        )
    except BaseException as error:
        # If rename already succeeded but verification failed, demote our newly
        # published directory.  A concurrent pre-existing target is never touched
        # while our private stage still exists.
        evidence_stage = stage if stage.exists() else target
        for marker_name in (_COMPLETE_MARKER, _INSUFFICIENT_MARKER):
            marker = evidence_stage / marker_name
            with suppress(OSError):
                marker.unlink(missing_ok=True)
        evidence_root = _retain_system_failure(
            stage=evidence_stage,
            output_root=output_root,
            session_id=session_id,
            session=session,
            error=error,
        )
        raise M8L2CaptureSystemError(
            f"live-L2 terminal publication failed without a terminal result: "
            f"{type(error).__name__}: {error}",
            evidence_root=evidence_root,
        ) from error


def _data_failure_result(
    *,
    symbol: str,
    reason_code: str,
    phase: str,
    stage_root: Path,
) -> SymbolCaptureResult:
    artifacts = tuple(
        CapturedArtifact(path=path, kind="partial_evidence", sha256=sha256_file(path))
        for path in _all_regular_files(stage_root)
    )
    return SymbolCaptureResult(
        symbol=symbol,
        capture_id=f"{symbol.lower()}-failed",
        status="FAILED",
        completion_reason="capture_failed",
        reconstruction_status="NOT_STARTED",
        messages=0,
        normalized_rows=0,
        reconstructed_rows=0,
        excluded_rows=0,
        continuity_epochs=0,
        snapshot_anchors=0,
        sequence_gaps=0,
        quality_errors=0,
        quality_warnings=0,
        max_raw_frame_bytes_observed=0,
        max_arrow_batch_bytes_observed=0,
        first_raw_received_ns=None,
        last_raw_received_ns=None,
        valid_observed_intervals=(),
        artifacts=artifacts,
        failure_reason_code=reason_code,
        failure_phase=phase,
    )


async def _wait_for_start(clock: CaptureClock, start_ns: int) -> int:
    while True:
        now = clock.time_ns()
        if now >= start_ns:
            return now
        # Recheck wall time at short intervals so suspend/resume and wall-clock
        # adjustments cannot silently convert the absolute UTC boundary into a
        # relative duration.
        await clock.sleep(min(30.0, (start_ns - now) / _NANOSECONDS_PER_SECOND))


async def capture_m8_l2_session(
    config: M8L2StudyConfig,
    session_date: str,
    output_root: str | Path,
    capture_one: CaptureOne,
    *,
    protocol_path: str | Path | None = None,
    clock: CaptureClock | None = None,
    _test_source_identity: CaptureSourceIdentity | None = None,
    _test_allow_injected_capture: bool = False,
) -> M8L2SessionBundle:
    """Capture both symbols under one absolute-time, immutable session authority."""

    try:
        canonical_config = load_m8_l2_config(config.path)
    except (OSError, ValueError) as error:
        raise M8L2CaptureSystemError(
            "configuration is not the frozen live-L2 byte authority"
        ) from error
    if canonical_config != config or canonical_config.hash != config.hash:
        raise M8L2CaptureSystemError(
            "in-memory configuration differs from the frozen live-L2 byte authority"
        )
    session = config.session_for_date(session_date)
    if (
        session.end_ns - session.start_ns
        != config.capture.duration_seconds * _NANOSECONDS_PER_SECOND
    ):
        raise M8L2CaptureSystemError("frozen session duration and capture duration disagree")
    project_root = config.path.parent.parent.resolve()
    requested_protocol = (
        Path(protocol_path)
        if protocol_path is not None
        else project_root / "docs" / "M8_L2_PROTOCOL.md"
    )
    if requested_protocol.is_symlink():
        raise M8L2CaptureSystemError("frozen live-L2 protocol must not be a symlink")
    protocol = requested_protocol.resolve()
    _assert_protocol(protocol)
    if _test_source_identity is not None and clock is None:
        raise M8L2CaptureSystemError(
            "test source identity injection requires an explicitly injected test clock"
        )
    if _test_allow_injected_capture and clock is None:
        raise M8L2CaptureSystemError(
            "test capture injection requires an explicitly injected test clock"
        )
    source_identity_was_injected = _test_source_identity is not None
    if not source_identity_was_injected:
        _assert_loaded_source_root(project_root)
    if not source_identity_was_injected and not _test_allow_injected_capture:
        # These private keyword-only values are unit-test seams for injected
        # collectors.  The production CLI supplies neither and therefore
        # cannot bypass the exact adapter-origin gate.
        _assert_production_capture_origin(project_root, capture_one)
    source = _test_source_identity or _source_identity(project_root)
    _validate_source_identity(source)
    runtime = _runtime_fingerprint()
    _revalidate_runtime_authority(
        config=config,
        protocol=protocol,
        expected_source=source,
        expected_runtime=runtime,
        source_identity_was_injected=source_identity_was_injected,
    )
    root, root_identity, sessions_identity = _prepare_output_root(output_root)
    campaign = _ensure_campaign_authority(
        root,
        root_identity=root_identity,
        config=config,
        source=source,
        runtime=runtime,
    )
    session_id = _session_identity(
        config=config,
        session=session,
        source=source,
        campaign_authority_sha256=campaign.sha256,
    )
    target = _terminal_root(root, session, session_id)
    _revalidate_output_layout(
        root,
        root_identity=root_identity,
        sessions_identity=sessions_identity,
        target=target,
    )
    if target.exists() or target.is_symlink():
        return verify_m8_l2_session_bundle(target, expected_config=config)

    stage = Path(tempfile.mkdtemp(dir=root, prefix=f".m8-l2-{session_id[:12]}-"))
    authority_root = stage / "authority"
    authority_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, authority_root / "m8_l2_capture_study.toml")
    shutil.copyfile(protocol, authority_root / "M8_L2_PROTOCOL.md")
    _write_bytes_durable(
        authority_root / _CAMPAIGN_AUTHORITY_NAME,
        campaign.raw,
    )
    capture_clock = clock or _SystemClock()
    now = capture_clock.time_ns()
    phase_ledger: list[dict[str, object]] = [
        {"phase": "PREFLIGHT", "status": "COMPLETE", "at_ns": now}
    ]
    if now >= session.end_ns:
        phase_ledger.extend(
            [
                {"phase": "BOUNDARY_WAIT", "status": "MISSED", "at_ns": now},
                {"phase": "DUAL_CAPTURE", "status": "NOT_RUN"},
                {"phase": "SESSION_GATES", "status": "FAILED"},
                {"phase": "TERMINALIZATION", "status": "READY"},
            ]
        )
        return _publish_or_retain_system_failure(
            stage=stage,
            target=target,
            output_root=root,
            config=config,
            protocol=protocol,
            session=session,
            session_id=session_id,
            source=source,
            runtime=runtime,
            source_identity_was_injected=source_identity_was_injected,
            campaign=campaign,
            status="INSUFFICIENT_DATA",
            reason_codes=("MISSED_WINDOW",),
            phase_ledger=phase_ledger,
            gates=(),
            results={},
            barrier_released_ns=None,
            overlap_ns=0,
            capture_finished_ns=None,
            root_identity=root_identity,
            sessions_identity=sessions_identity,
        )

    try:
        barrier_ns = await _wait_for_start(capture_clock, session.start_ns)
    except asyncio.CancelledError:
        # Before the declared interval, cancellation is restartable and does not
        # consume or terminalize the future session.
        raise
    if barrier_ns >= session.end_ns:
        try:
            _revalidate_runtime_authority(
                config=config,
                protocol=protocol,
                expected_source=source,
                expected_runtime=runtime,
                source_identity_was_injected=source_identity_was_injected,
            )
            _verify_campaign_authority(
                root,
                root_identity=root_identity,
                config=config,
                source=source,
                runtime=runtime,
                expected_sha256=campaign.sha256,
            )
        except BaseException as error:
            evidence_root = _retain_system_failure(
                stage=stage,
                output_root=root,
                session_id=session_id,
                session=session,
                error=error,
            )
            raise M8L2CaptureSystemError(
                "live-L2 authority changed while waiting for the frozen window",
                evidence_root=evidence_root,
            ) from error
        phase_ledger.extend(
            [
                {"phase": "BOUNDARY_WAIT", "status": "MISSED", "at_ns": barrier_ns},
                {"phase": "DUAL_CAPTURE", "status": "NOT_RUN"},
                {"phase": "SESSION_GATES", "status": "FAILED"},
                {"phase": "TERMINALIZATION", "status": "READY"},
            ]
        )
        return _publish_or_retain_system_failure(
            stage=stage,
            target=target,
            output_root=root,
            config=config,
            protocol=protocol,
            session=session,
            session_id=session_id,
            source=source,
            runtime=runtime,
            source_identity_was_injected=source_identity_was_injected,
            campaign=campaign,
            status="INSUFFICIENT_DATA",
            reason_codes=("MISSED_WINDOW",),
            phase_ledger=phase_ledger,
            gates=(),
            results={},
            barrier_released_ns=barrier_ns,
            overlap_ns=0,
            capture_finished_ns=None,
            root_identity=root_identity,
            sessions_identity=sessions_identity,
        )
    phase_ledger.append({"phase": "BOUNDARY_WAIT", "status": "COMPLETE", "at_ns": barrier_ns})

    try:
        _revalidate_runtime_authority(
            config=config,
            protocol=protocol,
            expected_source=source,
            expected_runtime=runtime,
            source_identity_was_injected=source_identity_was_injected,
        )
        _verify_campaign_authority(
            root,
            root_identity=root_identity,
            config=config,
            source=source,
            runtime=runtime,
            expected_sha256=campaign.sha256,
        )
    except BaseException as error:
        evidence_root = _retain_system_failure(
            stage=stage,
            output_root=root,
            session_id=session_id,
            session=session,
            error=error,
        )
        raise M8L2CaptureSystemError(
            "live-L2 authority changed before dual-symbol capture",
            evidence_root=evidence_root,
        ) from error

    tasks: dict[str, asyncio.Task[SymbolCaptureResult]] = {}
    for symbol in config.study.symbols:
        symbol_root = stage / "symbols" / symbol
        symbol_root.mkdir(parents=True, exist_ok=True)
        tasks[symbol] = asyncio.create_task(
            capture_one(
                symbol=symbol,
                scheduled_start_ns=session.start_ns,
                scheduled_end_ns=session.end_ns,
                stage_root=symbol_root,
                limits=config.capture,
                session_id=session_id,
            ),
            name=f"m8-l2-{session.date.isoformat()}-{symbol}",
        )

    outer_canceled = False
    try:
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)
    except asyncio.CancelledError:
        outer_canceled = True
        for task in tasks.values():
            task.cancel()
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)
    capture_finished_ns = capture_clock.time_ns()

    results: dict[str, SymbolCaptureResult] = {}
    data_reasons: list[str] = []
    system_errors: list[BaseException] = []
    for symbol, outcome in zip(tasks, outcomes, strict=True):
        symbol_root = stage / "symbols" / symbol
        if isinstance(outcome, SymbolCaptureResult):
            try:
                results[symbol] = _validate_symbol_result(
                    outcome,
                    expected_symbol=symbol,
                    symbol_root=symbol_root,
                    session=session,
                )
            except BaseException as error:
                system_errors.append(error)
        elif isinstance(outcome, M8L2DataFailure):
            data_reasons.append(outcome.reason_code)
            partial = outcome.partial_result
            if partial is None:
                partial = _data_failure_result(
                    symbol=symbol,
                    reason_code=outcome.reason_code,
                    phase=outcome.phase,
                    stage_root=symbol_root,
                )
            try:
                results[symbol] = _validate_symbol_result(
                    partial,
                    expected_symbol=symbol,
                    symbol_root=symbol_root,
                    session=session,
                )
            except BaseException as error:
                system_errors.append(error)
        elif isinstance(outcome, asyncio.CancelledError):
            data_reasons.append("CAPTURE_CANCELED")
            results[symbol] = _data_failure_result(
                symbol=symbol,
                reason_code="CAPTURE_CANCELED",
                phase="DUAL_CAPTURE",
                stage_root=symbol_root,
            )
        elif isinstance(outcome, BaseException):
            system_errors.append(outcome)

    if outer_canceled:
        data_reasons.append("CAPTURE_CANCELED")
    if system_errors:
        primary = system_errors[0]
        evidence_root = _retain_system_failure(
            stage=stage,
            output_root=root,
            session_id=session_id,
            session=session,
            error=primary,
        )
        raise M8L2CaptureSystemError(
            f"live-L2 capture ended in a nonterminal system failure: {type(primary).__name__}: {primary}",
            evidence_root=evidence_root,
        ) from primary
    if capture_finished_ns < session.end_ns and any(
        result.status == "COMPLETE" for result in results.values()
    ):
        early_completion_error = M8L2CaptureSystemError(
            "single-symbol capture reported success before the frozen scheduled end"
        )
        evidence_root = _retain_system_failure(
            stage=stage,
            output_root=root,
            session_id=session_id,
            session=session,
            error=early_completion_error,
        )
        raise M8L2CaptureSystemError(
            str(early_completion_error), evidence_root=evidence_root
        ) from early_completion_error

    phase_ledger.append(
        {
            "phase": "DUAL_CAPTURE",
            "status": "FAILED" if data_reasons else "COMPLETE",
            "symbols_completed": sorted(results),
        }
    )
    try:
        _revalidate_runtime_authority(
            config=config,
            protocol=protocol,
            expected_source=source,
            expected_runtime=runtime,
            source_identity_was_injected=source_identity_was_injected,
        )
        _verify_campaign_authority(
            root,
            root_identity=root_identity,
            config=config,
            source=source,
            runtime=runtime,
            expected_sha256=campaign.sha256,
        )
        gates, gate_reasons, overlap_ns = _evaluate_gates(results, config=config)
    except BaseException as error:
        evidence_root = _retain_system_failure(
            stage=stage,
            output_root=root,
            session_id=session_id,
            session=session,
            error=error,
        )
        raise M8L2CaptureSystemError(
            "live-L2 authority/gate evaluation ended in a nonterminal system failure",
            evidence_root=evidence_root,
        ) from error
    all_reasons = sorted(set((*data_reasons, *gate_reasons)))
    status: SessionStatus = "COMPLETE" if not all_reasons else "INSUFFICIENT_DATA"
    phase_ledger.extend(
        [
            {
                "phase": "SESSION_GATES",
                "status": "COMPLETE" if status == "COMPLETE" else "FAILED",
            },
            {"phase": "TERMINALIZATION", "status": "READY"},
        ]
    )
    return _publish_or_retain_system_failure(
        stage=stage,
        target=target,
        output_root=root,
        config=config,
        protocol=protocol,
        session=session,
        session_id=session_id,
        source=source,
        runtime=runtime,
        source_identity_was_injected=source_identity_was_injected,
        campaign=campaign,
        status=status,
        reason_codes=all_reasons,
        phase_ledger=phase_ledger,
        gates=gates,
        results=results,
        barrier_released_ns=barrier_ns,
        overlap_ns=overlap_ns,
        capture_finished_ns=capture_finished_ns,
        root_identity=root_identity,
        sessions_identity=sessions_identity,
    )


def _safe_checksum_relative(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or value != candidate.as_posix()
        or "\n" in value
        or "\r" in value
    ):
        raise M8L2VerificationError(f"unsafe checksum path: {value!r}")
    return value


def _parse_checksums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise M8L2VerificationError(f"cannot read checksum authority: {path}") from error
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise M8L2VerificationError("checksum authority has a malformed line")
        digest = line[:64]
        relative = _safe_checksum_relative(line[66:])
        if not _is_lower_sha256(digest):
            raise M8L2VerificationError("checksum authority contains an invalid digest")
        if relative in result:
            raise M8L2VerificationError(f"checksum authority repeats {relative}")
        result[relative] = digest
    if not result:
        raise M8L2VerificationError("checksum authority must protect at least one file")
    return result


def _manifest_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2VerificationError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _verification_json(path: Path, label: str) -> object:
    try:
        return _strict_json(path, label=label)
    except M8L2CaptureSystemError as error:
        raise M8L2VerificationError(str(error)) from error


def _symbol_result_from_manifest(
    value: object,
    *,
    symbol: str,
    root: Path,
    session: M8L2Session,
) -> SymbolCaptureResult:
    payload = _manifest_object(value, f"symbols.{symbol}")

    def integer(name: str) -> int:
        raw = payload.get(name)
        if type(raw) is not int or raw < 0:
            raise M8L2VerificationError(f"symbols.{symbol}.{name} is invalid")
        return raw

    intervals_raw = payload.get("valid_observed_intervals")
    if not isinstance(intervals_raw, list):
        raise M8L2VerificationError(f"symbols.{symbol} intervals must be an array")
    intervals: list[ObservedInterval] = []
    for index, raw in enumerate(intervals_raw):
        entry = _manifest_object(raw, f"symbols.{symbol}.intervals[{index}]")
        continuity_id = entry.get("continuity_id")
        start = entry.get("start_received_ns")
        end = entry.get("end_received_ns_exclusive")
        if type(continuity_id) is not str or type(start) is not int or type(end) is not int:
            raise M8L2VerificationError("session manifest has a malformed OBSERVED interval")
        try:
            interval = ObservedInterval(continuity_id, start, end)
        except ValueError as error:
            raise M8L2VerificationError(
                "session manifest has an invalid OBSERVED interval"
            ) from error
        if dict(entry) != interval.to_dict():
            raise M8L2VerificationError("session OBSERVED interval derived fields are inconsistent")
        intervals.append(interval)

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise M8L2VerificationError(f"symbols.{symbol}.artifacts must be an array")
    artifacts: list[CapturedArtifact] = []
    for index, raw in enumerate(artifacts_raw):
        entry = _manifest_object(raw, f"symbols.{symbol}.artifacts[{index}]")
        if set(entry) != {"path", "kind", "sha256", "bytes"}:
            raise M8L2VerificationError("session symbol artifact entry is not exact")
        relative_value = entry.get("path")
        kind = entry.get("kind")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if (
            type(relative_value) is not str
            or type(kind) is not str
            or type(digest) is not str
            or not _is_lower_sha256(digest)
            or type(size) is not int
            or size < 0
        ):
            raise M8L2VerificationError("session symbol artifact entry is malformed")
        relative = _safe_checksum_relative(relative_value)
        parts = PurePosixPath(relative).parts
        if len(parts) < 3 or parts[:2] != ("symbols", symbol):
            raise M8L2VerificationError("session symbol artifact escapes its symbol coordinate")
        path = root.joinpath(*parts)
        if path.stat().st_size != size:
            raise M8L2VerificationError("session symbol artifact size differs from its claim")
        artifacts.append(CapturedArtifact(path=path, kind=kind, sha256=digest))

    status = payload.get("status")
    reconstruction = payload.get("reconstruction_status")
    if status not in {"COMPLETE", "FAILED"} or reconstruction not in {
        "LIVE",
        "GAPPED",
        "INVALID",
        "NOT_STARTED",
    }:
        raise M8L2VerificationError("session symbol status is invalid")
    capture_id = payload.get("capture_id")
    completion_reason = payload.get("completion_reason")
    first = payload.get("first_raw_received_ns")
    last = payload.get("last_raw_received_ns")
    failure_reason = payload.get("failure_reason_code")
    failure_phase = payload.get("failure_phase")
    if (
        type(capture_id) is not str
        or type(completion_reason) is not str
        or (first is not None and type(first) is not int)
        or (last is not None and type(last) is not int)
        or (failure_reason is not None and type(failure_reason) is not str)
        or (failure_phase is not None and type(failure_phase) is not str)
    ):
        raise M8L2VerificationError("session symbol scalar claims are invalid")
    result = SymbolCaptureResult(
        symbol=symbol,
        capture_id=capture_id,
        status=cast(CaptureStatus, status),
        completion_reason=completion_reason,
        reconstruction_status=cast(ReconstructionStatus, reconstruction),
        messages=integer("messages"),
        normalized_rows=integer("normalized_rows"),
        reconstructed_rows=integer("reconstructed_rows"),
        excluded_rows=integer("excluded_rows"),
        continuity_epochs=integer("continuity_epochs"),
        snapshot_anchors=integer("snapshot_anchors"),
        sequence_gaps=integer("sequence_gaps"),
        quality_errors=integer("quality_errors"),
        quality_warnings=integer("quality_warnings"),
        max_raw_frame_bytes_observed=integer("max_raw_frame_bytes_observed"),
        max_arrow_batch_bytes_observed=integer("max_arrow_batch_bytes_observed"),
        first_raw_received_ns=first,
        last_raw_received_ns=last,
        valid_observed_intervals=tuple(intervals),
        artifacts=tuple(artifacts),
        failure_reason_code=failure_reason,
        failure_phase=failure_phase,
    )
    try:
        validated = _validate_symbol_result(
            result,
            expected_symbol=symbol,
            symbol_root=root / "symbols" / symbol,
            session=session,
        )
    except M8L2CaptureError as error:
        raise M8L2VerificationError(str(error)) from error
    if dict(payload) != _symbol_payload(validated, stage_root=root):
        raise M8L2VerificationError("session symbol payload is not its canonical claim projection")
    return validated


def verify_m8_l2_session_bundle(
    bundle_dir: str | Path,
    *,
    expected_config: M8L2StudyConfig | None = None,
) -> M8L2SessionBundle:
    """Verify marker bytes, exact physical inventory, checksums, and frozen authority."""

    requested_root = Path(bundle_dir)
    try:
        _reject_existing_symlink_components(requested_root, label="session bundle path")
    except M8L2CaptureSystemError as error:
        raise M8L2VerificationError(str(error)) from error
    root = requested_root.absolute()
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise M8L2VerificationError(f"session bundle is not a regular directory: {root}")
        files, _ = _walk_regular_evidence(root)
    except (OSError, M8L2CaptureSystemError) as error:
        raise M8L2VerificationError(f"invalid session bundle filesystem: {error}") from error
    if root.parent.name != "sessions":
        raise M8L2VerificationError("session bundle parent is not the canonical sessions directory")
    manifest_path = root / "session_manifest.json"
    checksum_path = root / _CHECKSUM_NAME
    regular_paths = {_relative(path, root): path for path in files}
    if "session_manifest.json" not in regular_paths or _CHECKSUM_NAME not in regular_paths:
        raise M8L2VerificationError("session bundle lacks its manifest or checksum authority")
    checksums = _parse_checksums(checksum_path)
    marker_candidates = [
        root / name for name in (_COMPLETE_MARKER, _INSUFFICIENT_MARKER) if name in regular_paths
    ]
    if len(marker_candidates) != 1:
        raise M8L2VerificationError("session bundle must have exactly one terminal marker")
    marker_path = marker_candidates[0]
    expected_marker_bytes = (
        _COMPLETE_BYTES if marker_path.name == _COMPLETE_MARKER else _INSUFFICIENT_BYTES
    )
    if marker_path.read_bytes() != expected_marker_bytes:
        raise M8L2VerificationError("session terminal marker bytes differ from the contract")

    actual = set(regular_paths)
    expected = set(checksums) | {_CHECKSUM_NAME, marker_path.name}
    if actual != expected:
        raise M8L2VerificationError(
            "session physical inventory differs from its checksum authority "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    for relative, digest in checksums.items():
        path = root / relative
        if sha256_file(path) != digest:
            raise M8L2VerificationError(f"checksum mismatch for {relative}")

    payload = _manifest_object(
        _verification_json(manifest_path, "session manifest"), "session manifest"
    )
    if set(payload) != {
        "schema_version",
        "artifact_kind",
        "generated_at_utc",
        "status",
        "session_id",
        "study_name",
        "protocol_version",
        "evidence_tier",
        "live_trading",
        "session",
        "authority",
        "symbols",
        "cross_symbol_observed_overlap_seconds",
        "gates",
        "reason_codes",
        "phase_ledger",
        "artifact_inventory",
        "terminal_marker",
        "policy",
    }:
        raise M8L2VerificationError("session manifest keys differ from the exact schema")
    if payload.get("schema_version") != _SESSION_SCHEMA_VERSION:
        raise M8L2VerificationError("unsupported live-L2 session manifest schema")
    if payload.get("artifact_kind") != "m8_prospective_live_l2_session":
        raise M8L2VerificationError("session artifact kind is invalid")
    status = payload.get("status")
    if status not in {"COMPLETE", "INSUFFICIENT_DATA"}:
        raise M8L2VerificationError("session manifest has an unsupported status")
    expected_marker = _COMPLETE_MARKER if status == "COMPLETE" else _INSUFFICIENT_MARKER
    if marker_path.name != expected_marker:
        raise M8L2VerificationError("terminal marker disagrees with session status")
    session_id = payload.get("session_id")
    if type(session_id) is not str or not _is_lower_sha256(session_id):
        raise M8L2VerificationError("session manifest has an invalid session_id")
    authority = _manifest_object(payload.get("authority"), "session authority")
    if set(authority) != {
        "campaign_authority_sha256",
        "config_sha256",
        "config_source_sha256",
        "protocol_sha256",
        "protocol_freeze_commit",
        "runtime_commit",
        "runtime_source_tree_sha256",
        "runtime_dirty",
        "runtime_fingerprint",
        "runtime_fingerprint_sha256",
    }:
        raise M8L2VerificationError("session authority keys differ from the exact schema")
    config_path = root / "authority" / "m8_l2_capture_study.toml"
    protocol_path = root / "authority" / "M8_L2_PROTOCOL.md"
    campaign_path = root / "authority" / _CAMPAIGN_AUTHORITY_NAME
    try:
        bundled_config = load_m8_l2_config(config_path)
    except (OSError, ValueError) as error:
        raise M8L2VerificationError("bundled live-L2 config is not the frozen authority") from error
    if authority.get("config_source_sha256") != M8_L2_CONFIG_SOURCE_SHA256:
        raise M8L2VerificationError("session config bytes are not the frozen authority")
    if authority.get("config_sha256") != bundled_config.hash:
        raise M8L2VerificationError("session config semantic hash differs from bundled bytes")
    if authority.get("protocol_sha256") != M8_L2_PROTOCOL_SHA256:
        raise M8L2VerificationError("session protocol bytes are not the frozen authority")
    if authority.get("protocol_freeze_commit") != M8_L2_FREEZE_COMMIT:
        raise M8L2VerificationError("session freeze commit is not the declared authority")
    source = CaptureSourceIdentity(
        commit=str(authority.get("runtime_commit")),
        source_tree_sha256=str(authority.get("runtime_source_tree_sha256")),
        dirty=authority.get("runtime_dirty") is not False,
    )
    try:
        _validate_source_identity(source)
    except M8L2CaptureSystemError as error:
        raise M8L2VerificationError(f"invalid runtime source identity: {error}") from error
    campaign_sha256 = authority.get("campaign_authority_sha256")
    if type(campaign_sha256) is not str or not _is_lower_sha256(campaign_sha256):
        raise M8L2VerificationError("session campaign authority digest is invalid")
    try:
        bundled_campaign = _verify_bundled_campaign_authority(
            campaign_path,
            config=bundled_config,
            source=source,
            expected_sha256=campaign_sha256,
        )
    except M8L2CaptureSystemError as error:
        raise M8L2VerificationError(f"invalid bundled campaign authority: {error}") from error
    try:
        manifest_runtime = _runtime_from_recorded(
            authority.get("runtime_fingerprint"),
            authority.get("runtime_fingerprint_sha256"),
        )
    except M8L2CaptureSystemError as error:
        raise M8L2VerificationError(f"invalid session runtime fingerprint: {error}") from error
    if manifest_runtime != bundled_campaign.runtime:
        raise M8L2VerificationError(
            "session runtime fingerprint differs from bundled campaign authority"
        )
    campaign_relative = f"authority/{_CAMPAIGN_AUTHORITY_NAME}"
    if checksums.get(campaign_relative) != campaign_sha256:
        raise M8L2VerificationError(
            "session checksum authority does not bind the campaign authority digest"
        )
    if sha256_file(config_path) != M8_L2_CONFIG_SOURCE_SHA256:
        raise M8L2VerificationError("bundled live-L2 config bytes are corrupt")
    if sha256_file(protocol_path) != M8_L2_PROTOCOL_SHA256:
        raise M8L2VerificationError("bundled live-L2 protocol bytes are corrupt")
    if expected_config is not None:
        try:
            expected_canonical = load_m8_l2_config(expected_config.path)
        except (OSError, ValueError) as error:
            raise M8L2VerificationError("caller config is not backed by frozen bytes") from error
        if expected_canonical != expected_config:
            raise M8L2VerificationError("caller config object differs from its frozen bytes")
        if (
            expected_config.source_sha256 != bundled_config.source_sha256
            or expected_config.hash != bundled_config.hash
        ):
            raise M8L2VerificationError("caller config semantics differ from the session authority")
    if (
        payload.get("study_name") != bundled_config.study.name
        or payload.get("protocol_version") != bundled_config.study.protocol_version
        or payload.get("evidence_tier") != bundled_config.study.evidence_tier
        or payload.get("live_trading") is not False
    ):
        raise M8L2VerificationError("session study claims differ from frozen configuration")

    inventory_raw = payload.get("artifact_inventory")
    if not isinstance(inventory_raw, list):
        raise M8L2VerificationError("session artifact inventory must be an array")
    inventory: dict[str, tuple[str, int]] = {}
    for index, item in enumerate(inventory_raw):
        entry = _manifest_object(item, f"artifact_inventory[{index}]")
        if set(entry) != {"path", "sha256", "bytes"}:
            raise M8L2VerificationError("session artifact inventory entry is not exact")
        relative = _safe_checksum_relative(str(entry.get("path")))
        digest = str(entry.get("sha256"))
        size = entry.get("bytes")
        if not _is_lower_sha256(digest) or isinstance(size, bool) or not isinstance(size, int):
            raise M8L2VerificationError("session artifact inventory entry is malformed")
        if relative in inventory:
            raise M8L2VerificationError(f"session artifact inventory repeats {relative}")
        inventory[relative] = (digest, size)
    if list(inventory) != sorted(inventory):
        raise M8L2VerificationError("session artifact inventory is not canonically ordered")
    expected_inventory = set(checksums) - {"session_manifest.json"}
    if set(inventory) != expected_inventory:
        raise M8L2VerificationError("manifest artifact inventory is not exact")
    for relative, (digest, size) in inventory.items():
        path = root / relative
        if checksums.get(relative) != digest or path.stat().st_size != size:
            raise M8L2VerificationError(f"manifest inventory metadata differs for {relative}")

    reasons_raw = payload.get("reason_codes")
    if not isinstance(reasons_raw, list) or not all(type(item) is str for item in reasons_raw):
        raise M8L2VerificationError("session reason_codes must be a string array")
    reasons = tuple(cast(list[str], reasons_raw))
    if list(reasons) != sorted(set(reasons)):
        raise M8L2VerificationError("session reason codes are not canonical")

    session_payload = _manifest_object(payload.get("session"), "session coordinates")
    if set(session_payload) != {
        "date",
        "role",
        "scheduled_start_ns",
        "scheduled_end_ns",
        "scheduled_duration_seconds",
        "barrier_released_ns",
        "barrier_lateness_ns",
        "capture_finished_ns",
    }:
        raise M8L2VerificationError("session coordinate keys differ from the exact schema")
    session_date = session_payload.get("date")
    role = session_payload.get("role")
    if type(session_date) is not str or type(role) is not str:
        raise M8L2VerificationError("session date/role coordinates are invalid")
    try:
        frozen_session = bundled_config.session_for_date(session_date)
    except ValueError as error:
        raise M8L2VerificationError("session date is not in the frozen calendar") from error
    expected_session_coordinates = {
        "date": frozen_session.date.isoformat(),
        "role": frozen_session.role,
        "scheduled_start_ns": frozen_session.start_ns,
        "scheduled_end_ns": frozen_session.end_ns,
        "scheduled_duration_seconds": (frozen_session.end_ns - frozen_session.start_ns)
        / _NANOSECONDS_PER_SECOND,
    }
    for name, expected_value in expected_session_coordinates.items():
        if session_payload.get(name) != expected_value:
            raise M8L2VerificationError(f"session coordinate {name} differs from the freeze")
    capture_finished = session_payload.get("capture_finished_ns")
    if capture_finished is not None and type(capture_finished) is not int:
        raise M8L2VerificationError("session capture_finished_ns is invalid")

    recomputed_session_id = _session_identity(
        config=bundled_config,
        session=frozen_session,
        source=source,
        campaign_authority_sha256=campaign_sha256,
    )
    if session_id != recomputed_session_id:
        raise M8L2VerificationError("session_id does not derive from the frozen authority")
    expected_name = f"{frozen_session.date.isoformat()}-{frozen_session.role}-{session_id[:20]}"
    if root.name != expected_name:
        raise M8L2VerificationError("session bundle basename differs from its authority")

    symbols_raw = _manifest_object(payload.get("symbols"), "session symbols")
    if set(symbols_raw) != set(bundled_config.study.symbols) and not (
        not symbols_raw and reasons == ("MISSED_WINDOW",)
    ):
        raise M8L2VerificationError("session symbols differ from the exact frozen pair")
    results: dict[str, SymbolCaptureResult] = {}
    for symbol in bundled_config.study.symbols:
        if symbol in symbols_raw:
            results[symbol] = _symbol_result_from_manifest(
                symbols_raw[symbol], symbol=symbol, root=root, session=frozen_session
            )

    gates_raw = payload.get("gates")
    if not isinstance(gates_raw, list) or not all(isinstance(item, Mapping) for item in gates_raw):
        raise M8L2VerificationError("session gates must be an array of objects")
    if reasons == ("MISSED_WINDOW",):
        if status != "INSUFFICIENT_DATA" or results or gates_raw:
            raise M8L2VerificationError("missed-window terminal evidence is inconsistent")
        overlap_ns = 0
    else:
        if set(results) != set(bundled_config.study.symbols):
            raise M8L2VerificationError("non-missed session lacks the exact frozen symbol pair")
        expected_gates, expected_reasons, overlap_ns = _evaluate_gates(
            results, config=bundled_config
        )
        if len(expected_gates) != 29 or [dict(item) for item in gates_raw] != expected_gates:
            raise M8L2VerificationError("session gates are not the exact 29 recomputed gates")
        if list(reasons) != expected_reasons:
            raise M8L2VerificationError("session reasons differ from recomputed gate failures")
        expected_status = "COMPLETE" if not expected_reasons else "INSUFFICIENT_DATA"
        if status != expected_status:
            raise M8L2VerificationError("session status differs from recomputed gates")
    if payload.get("cross_symbol_observed_overlap_seconds") != (
        overlap_ns / _NANOSECONDS_PER_SECOND
    ):
        raise M8L2VerificationError("cross-symbol overlap differs from recomputed intervals")
    if status == "COMPLETE" and (
        type(capture_finished) is not int or capture_finished < frozen_session.end_ns
    ):
        raise M8L2VerificationError("complete session was finalized before scheduled end")
    terminal_marker = _manifest_object(payload.get("terminal_marker"), "terminal marker claim")
    if terminal_marker != {
        "path": expected_marker,
        "bytes": "complete\\n" if status == "COMPLETE" else "terminal\\n",
    }:
        raise M8L2VerificationError("terminal marker claim differs from exact bytes")
    return M8L2SessionBundle(
        root=root,
        status=cast(SessionStatus, status),
        session_id=session_id,
        session_date=session_date,
        role=role,
        manifest_path=manifest_path,
        manifest_sha256=checksums["session_manifest.json"],
        checksum_path=checksum_path,
        marker_path=marker_path,
        reason_codes=reasons,
    )


__all__ = [
    "CaptureClock",
    "CaptureOne",
    "CaptureSourceIdentity",
    "CapturedArtifact",
    "M8L2CaptureError",
    "M8L2CaptureSystemError",
    "M8L2DataFailure",
    "M8L2SessionBundle",
    "M8L2VerificationError",
    "ObservedInterval",
    "SymbolCaptureResult",
    "capture_m8_l2_session",
    "current_m8_l2_runtime_fingerprint_sha256",
    "merge_observed_intervals",
    "overlapping_observed_coverage_ns",
    "verify_m8_l2_session_bundle",
]
