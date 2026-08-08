"""Strict, phase-separated inputs for the frozen prospective M8 L2 study.

The capture verifier remains the authority for a complete session bundle.  This
module adds the narrower research-input boundary: it binds an optional external
manifest/checksum digest, exposes only explicitly inventoried symbol artifacts,
and reopens Parquet through a no-follow directory/file-descriptor chain whenever
payload rows are requested.

Development and held-out entry points are intentionally separate.  A held-out
object cannot be constructed without the digest of an already verified
development lock; this module does not inspect or interpret that lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from microstructure.data.schemas import SCHEMA_VERSION, ensure_schema, get_schema
from microstructure.m8_l2_capture import (
    M8L2SessionBundle,
    M8L2VerificationError,
    verify_m8_l2_session_bundle,
)
from microstructure.m8_l2_config import M8L2StudyConfig
from microstructure.research.l2_multidate import L2ObservedInterval

DevelopmentRole = Literal["train", "validation"]
HeldoutRole = Literal["primary_test", "replication_test"]
L2InputAccessPhase = Literal["development", "heldout_after_lock"]

_CHECKSUM_NAME = "CHECKSUMS.sha256"
_MANIFEST_NAME = "session_manifest.json"
_READ_CHUNK_BYTES = 1 << 20
_MAX_JSON_BYTES = 8 << 20
_MAX_CHECKSUM_BYTES = 8 << 20
_MAX_PARQUET_FILE_BYTES = 2 << 30
_MAX_PARQUET_UNCOMPRESSED_BYTES = 4 << 30
_MAX_COMBINED_UNCOMPRESSED_BYTES = 6 << 30
_MAX_PARQUET_FOOTER_BYTES = 8 << 20
_VERIFIED_INPUT_TOKEN = object()


class M8L2InputError(M8L2VerificationError):
    """Raised when a verified capture is not a safe, phase-correct input."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, label: str) -> None:
    if not _is_sha256(value):
        raise M8L2InputError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class L2SessionFileAuthority:
    """External/discovered digest authority for the two session control files."""

    manifest_sha256: str
    checksums_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.manifest_sha256, "session manifest authority")
        _require_sha256(self.checksums_sha256, "session checksum-file authority")

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "checksums_sha256": self.checksums_sha256,
        }


@dataclass(frozen=True, slots=True)
class L2CampaignRuntimeIdentity:
    """Campaign identity that must compare equal across all four sessions."""

    campaign_authority_sha256: str
    runtime_commit: str
    runtime_source_tree_sha256: str
    runtime_fingerprint_sha256: str
    runtime_dirty: bool

    def __post_init__(self) -> None:
        _require_sha256(self.campaign_authority_sha256, "campaign authority")
        if len(self.runtime_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.runtime_commit
        ):
            raise M8L2InputError("campaign runtime commit must be a lowercase Git SHA-1")
        _require_sha256(self.runtime_source_tree_sha256, "campaign runtime source tree")
        _require_sha256(self.runtime_fingerprint_sha256, "campaign runtime fingerprint")
        if self.runtime_dirty:
            raise M8L2InputError("campaign runtime identity must be clean")

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_authority_sha256": self.campaign_authority_sha256,
            "runtime_commit": self.runtime_commit,
            "runtime_source_tree_sha256": self.runtime_source_tree_sha256,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "runtime_dirty": self.runtime_dirty,
        }


@dataclass(frozen=True, slots=True)
class VerifiedL2Artifact:
    """One exact session-relative artifact coordinate."""

    relative_path: str
    kind: str
    sha256: str
    bytes: int
    rows: int | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        _safe_relative(self.relative_path)
        if not self.kind:
            raise M8L2InputError("artifact kind must not be empty")
        _require_sha256(self.sha256, "artifact digest")
        if self.bytes < 0:
            raise M8L2InputError("artifact byte count must be nonnegative")
        if self.rows is not None and self.rows < 0:
            raise M8L2InputError("artifact row count must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "rows": self.rows,
            "dataset": self.dataset,
        }


@dataclass(frozen=True, slots=True)
class VerifiedL2SymbolInput:
    """Exact research-relevant artifacts and claims for one symbol."""

    symbol: str
    capture_id: str
    normalized_rows: int
    reconstructed_rows: int
    excluded_rows: int
    valid_observed_intervals: tuple[L2ObservedInterval, ...]
    raw_journal: VerifiedL2Artifact
    capture_summary: VerifiedL2Artifact
    depth_deltas: VerifiedL2Artifact
    book_observations: VerifiedL2Artifact


@dataclass(frozen=True, slots=True)
class LoadedL2SymbolFrames:
    """The only payload opened by the research producer for one symbol/session."""

    book_observations: pl.DataFrame
    depth_deltas: pl.DataFrame
    intervals: tuple[L2ObservedInterval, ...]


@dataclass(frozen=True, slots=True)
class _RootIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class VerifiedL2SessionInput:
    """Verified authority with an explicitly phase-authorized payload loader."""

    root: Path
    session_id: str
    session_date: str
    role: str
    config_sha256: str
    config_source_sha256: str
    file_authority: L2SessionFileAuthority
    campaign_identity: L2CampaignRuntimeIdentity
    symbols: Mapping[str, VerifiedL2SymbolInput]
    access_phase: L2InputAccessPhase
    development_lock_sha256: str | None
    _root_identity: _RootIdentity = field(repr=False, compare=False)
    _expected_paths: frozenset[str] = field(repr=False, compare=False)
    _session_start_ns: int = field(repr=False, compare=False)
    _session_end_ns: int = field(repr=False, compare=False)
    _verification_token: object = field(repr=False, compare=False)

    def load_symbol_frames(self, symbol: str) -> LoadedL2SymbolFrames:
        """Load exactly one requested symbol after revalidating immutable evidence."""

        if self._verification_token is not _VERIFIED_INPUT_TOKEN:
            raise M8L2InputError("L2 payload loading requires a verifier-created input object")
        descriptor = self.symbols.get(symbol)
        if descriptor is None:
            raise M8L2InputError(f"symbol {symbol!r} is not in verified session {self.session_id}")
        _assert_inventory(
            self.root,
            expected_root=self._root_identity,
            expected_paths=self._expected_paths,
        )
        _verify_artifact_hash(
            self.root,
            expected_root=self._root_identity,
            artifact=descriptor.raw_journal,
        )
        _verify_artifact_hash(
            self.root,
            expected_root=self._root_identity,
            artifact=descriptor.capture_summary,
        )
        books, books_uncompressed = _load_verified_parquet(
            self.root,
            expected_root=self._root_identity,
            artifact=descriptor.book_observations,
        )
        deltas, deltas_uncompressed = _load_verified_parquet(
            self.root,
            expected_root=self._root_identity,
            artifact=descriptor.depth_deltas,
        )
        if books_uncompressed + deltas_uncompressed > _MAX_COMBINED_UNCOMPRESSED_BYTES:
            raise M8L2InputError("requested symbol Parquet payload exceeds the memory bound")
        _validate_loaded_frames(
            books,
            deltas,
            descriptor=descriptor,
            session_start_ns=self._session_start_ns,
            session_end_ns=self._session_end_ns,
        )
        _assert_inventory(
            self.root,
            expected_root=self._root_identity,
            expected_paths=self._expected_paths,
        )
        return LoadedL2SymbolFrames(
            book_observations=books,
            depth_deltas=deltas,
            intervals=descriptor.valid_observed_intervals,
        )


def _safe_relative(value: str) -> tuple[str, ...]:
    if not value or "\\" in value or "\x00" in value:
        raise M8L2InputError("artifact path is not a safe POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise M8L2InputError("artifact path is not a safe POSIX relative path")
    if pure.as_posix() != value:
        raise M8L2InputError("artifact path is not canonical")
    return pure.parts


def _absolute_without_resolve(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_root(root: Path, expected: _RootIdentity | None = None) -> int:
    """Open every absolute-path directory component without following a link."""

    if not root.is_absolute():
        raise M8L2InputError("session root must be absolute")
    descriptor = os.open(root.anchor, _directory_flags())
    try:
        for part in root.parts[1:]:
            next_descriptor = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise M8L2InputError("session root is not a directory")
        observed = _RootIdentity(metadata.st_dev, metadata.st_ino)
        if expected is not None and observed != expected:
            raise M8L2InputError("session root identity changed after verification")
        return descriptor
    except (OSError, M8L2InputError):
        os.close(descriptor)
        raise


def _open_relative(root_descriptor: int, relative: str) -> tuple[int, int, str]:
    parts = _safe_relative(relative)
    directory = os.dup(root_descriptor)
    try:
        for part in parts[:-1]:
            next_directory = os.open(part, _directory_flags(), dir_fd=directory)
            os.close(directory)
            directory = next_directory
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(parts[-1], flags, dir_fd=directory)
        return directory, descriptor, parts[-1]
    except OSError:
        os.close(directory)
        raise


def _same_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _assert_still_named(
    parent_descriptor: int,
    leaf: str,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    current = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or not _same_metadata(expected, current):
        raise M8L2InputError(f"{label} path changed during its descriptor snapshot")


def _hash_descriptor(descriptor: int, maximum_bytes: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    remaining = maximum_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest(), total


def _read_control_file(
    root: Path,
    *,
    expected_root: _RootIdentity | None,
    relative: str,
    maximum_bytes: int,
) -> tuple[bytes, _RootIdentity]:
    root_descriptor = _open_root(root, expected_root)
    try:
        root_metadata = os.fstat(root_descriptor)
        root_identity = _RootIdentity(root_metadata.st_dev, root_metadata.st_ino)
        parent, descriptor, leaf = _open_relative(root_descriptor, relative)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size < 1
                or before.st_size > maximum_bytes
            ):
                raise M8L2InputError(f"{relative} is not a bounded nonempty regular file")
            chunks: list[bytes] = []
            os.lseek(descriptor, 0, os.SEEK_SET)
            remaining = maximum_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            _assert_still_named(parent, leaf, after, label=relative)
            if len(raw) != before.st_size or not _same_metadata(before, after):
                raise M8L2InputError(f"{relative} changed during its bounded read")
            return raw, root_identity
        finally:
            os.close(descriptor)
            os.close(parent)
    except OSError as error:
        raise M8L2InputError(f"cannot securely read session control file {relative}") from error
    finally:
        os.close(root_descriptor)


def _secure_inventory(root: Path, expected_root: _RootIdentity) -> frozenset[str]:
    root_descriptor = _open_root(root, expected_root)

    def visit(descriptor: int, prefix: tuple[str, ...]) -> list[str]:
        result: list[str] = []
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise M8L2InputError("cannot enumerate explicit session bundle") from error
        for name in names:
            if not name or "/" in name or "\x00" in name:
                raise M8L2InputError("session bundle contains an unsafe directory entry")
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise M8L2InputError("session inventory changed during enumeration") from error
            relative_parts = (*prefix, name)
            if stat.S_ISREG(metadata.st_mode):
                result.append(PurePosixPath(*relative_parts).as_posix())
            elif stat.S_ISDIR(metadata.st_mode):
                try:
                    child = os.open(name, _directory_flags(), dir_fd=descriptor)
                except OSError as error:
                    raise M8L2InputError("cannot securely descend session directory") from error
                try:
                    result.extend(visit(child, relative_parts))
                finally:
                    os.close(child)
            else:
                raise M8L2InputError("session bundle contains a symlink or special file")
        return result

    try:
        return frozenset(visit(root_descriptor, ()))
    finally:
        os.close(root_descriptor)


def _assert_inventory(
    root: Path, *, expected_root: _RootIdentity, expected_paths: frozenset[str]
) -> None:
    observed = _secure_inventory(root, expected_root)
    if observed != expected_paths:
        raise M8L2InputError(
            "session inventory changed after verification "
            f"(missing={sorted(expected_paths - observed)}, "
            f"extra={sorted(observed - expected_paths)})"
        )


def _json_object(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise M8L2InputError(f"cannot parse {label}") from error
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2InputError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _parse_checksums(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines(keepends=True)
    except UnicodeError as error:
        raise M8L2InputError("session checksum authority must be ASCII") from error
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 68 or not line.endswith("\n") or line[64:66] != "  ":
            raise M8L2InputError("session checksum authority has a malformed line")
        digest = line[:64]
        relative = line[66:-1]
        _require_sha256(digest, "session checksum entry")
        _safe_relative(relative)
        if relative in result:
            raise M8L2InputError("session checksum authority repeats a path")
        result[relative] = digest
    if not result or list(result) != sorted(result):
        raise M8L2InputError("session checksum authority is empty or not canonically ordered")
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2InputError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise M8L2InputError(f"{label} must be a nonnegative integer")
    return value


def _artifact_entries(
    symbol_payload: Mapping[str, Any], symbol: str
) -> dict[str, VerifiedL2Artifact]:
    raw = symbol_payload.get("artifacts")
    if not isinstance(raw, list):
        raise M8L2InputError(f"symbol {symbol} lacks its artifact inventory")
    result: dict[str, VerifiedL2Artifact] = {}
    for value in raw:
        entry = _mapping(value, f"{symbol} artifact")
        relative = entry.get("path")
        kind = entry.get("kind")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if (
            type(relative) is not str
            or type(kind) is not str
            or type(digest) is not str
            or type(size) is not int
        ):
            raise M8L2InputError(f"symbol {symbol} has a malformed artifact")
        artifact = VerifiedL2Artifact(
            relative_path=relative,
            kind=kind,
            sha256=digest,
            bytes=size,
        )
        if relative in result:
            raise M8L2InputError(f"symbol {symbol} repeats an artifact coordinate")
        result[relative] = artifact
    return result


def _one_kind(
    artifacts: Mapping[str, VerifiedL2Artifact], *, kind: str, symbol: str
) -> VerifiedL2Artifact:
    matches = [item for item in artifacts.values() if item.kind == kind]
    if len(matches) != 1:
        raise M8L2InputError(f"symbol {symbol} must have exactly one {kind} artifact")
    return matches[0]


def _symbol_relative(symbol: str, value: object) -> str:
    if type(value) is not str:
        raise M8L2InputError(f"symbol {symbol} summary path must be a string")
    parts = _safe_relative(value)
    return PurePosixPath("symbols", symbol, *parts).as_posix()


def _normalized_artifact(
    *,
    symbol: str,
    dataset: str,
    summary: Mapping[str, Any],
    artifacts: Mapping[str, VerifiedL2Artifact],
) -> VerifiedL2Artifact:
    datasets = _mapping(summary.get("normalized_dataset_manifests"), "normalized manifests")
    entry = _mapping(datasets.get(dataset), f"normalized manifest {dataset}")
    rows = _integer(entry.get("rows"), f"{dataset} rows")
    relative = _symbol_relative(symbol, entry.get("data_path"))
    artifact = artifacts.get(relative)
    if artifact is None or artifact.kind != "normalized_data":
        raise M8L2InputError(f"symbol {symbol} {dataset} data is not exactly inventoried")
    if entry.get("data_sha256") != artifact.sha256:
        raise M8L2InputError(f"symbol {symbol} {dataset} digest claims disagree")
    return VerifiedL2Artifact(
        relative_path=artifact.relative_path,
        kind=artifact.kind,
        sha256=artifact.sha256,
        bytes=artifact.bytes,
        rows=rows,
        dataset=dataset,
    )


def _parse_intervals(
    symbol_payload: Mapping[str, Any], *, symbol: str, start_ns: int, end_ns: int
) -> tuple[L2ObservedInterval, ...]:
    raw = symbol_payload.get("valid_observed_intervals")
    if not isinstance(raw, list) or not raw:
        raise M8L2InputError(f"symbol {symbol} lacks valid OBSERVED intervals")
    intervals: list[L2ObservedInterval] = []
    prior_end = -1
    for value in raw:
        entry = _mapping(value, f"symbol {symbol} interval")
        continuity = entry.get("continuity_id")
        interval_start = entry.get("start_received_ns")
        interval_end = entry.get("end_received_ns_exclusive")
        if (
            type(continuity) is not str
            or type(interval_start) is not int
            or type(interval_end) is not int
        ):
            raise M8L2InputError(f"symbol {symbol} has a malformed OBSERVED interval")
        interval = L2ObservedInterval(continuity, interval_start, interval_end)
        if interval.start_received_ns < start_ns or interval.end_received_ns_exclusive > end_ns:
            raise M8L2InputError(f"symbol {symbol} OBSERVED interval escapes its session")
        if interval.start_received_ns < prior_end:
            raise M8L2InputError(f"symbol {symbol} OBSERVED intervals overlap")
        prior_end = interval.end_received_ns_exclusive
        intervals.append(interval)
    return tuple(intervals)


def _symbol_descriptor(
    root: Path,
    *,
    expected_root: _RootIdentity,
    symbol: str,
    payload: Mapping[str, Any],
    start_ns: int,
    end_ns: int,
) -> VerifiedL2SymbolInput:
    artifacts = _artifact_entries(payload, symbol)
    capture_summary = _one_kind(artifacts, kind="capture_summary", symbol=symbol)
    raw_journal = _one_kind(artifacts, kind="raw_journal", symbol=symbol)
    summary_raw, _ = _read_control_file(
        root,
        expected_root=expected_root,
        relative=capture_summary.relative_path,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    if hashlib.sha256(summary_raw).hexdigest() != capture_summary.sha256:
        raise M8L2InputError(f"symbol {symbol} capture summary digest changed")
    summary = _json_object(summary_raw, f"symbol {symbol} capture summary")
    if summary.get("symbol") != symbol or summary.get("capture_id") != payload.get("capture_id"):
        raise M8L2InputError(f"symbol {symbol} capture summary identity differs")
    depth = _normalized_artifact(
        symbol=symbol,
        dataset="depth_deltas",
        summary=summary,
        artifacts=artifacts,
    )
    books = _normalized_artifact(
        symbol=symbol,
        dataset="book_observations",
        summary=summary,
        artifacts=artifacts,
    )
    normalized_rows = _integer(payload.get("normalized_rows"), f"{symbol} normalized_rows")
    reconstructed_rows = _integer(payload.get("reconstructed_rows"), f"{symbol} reconstructed_rows")
    excluded_rows = _integer(payload.get("excluded_rows"), f"{symbol} excluded_rows")
    if (
        depth.rows != normalized_rows
        or books.rows != reconstructed_rows
        or normalized_rows != reconstructed_rows + excluded_rows
    ):
        raise M8L2InputError(f"symbol {symbol} normalized row claims do not reconcile")
    capture_id = payload.get("capture_id")
    if type(capture_id) is not str or not capture_id:
        raise M8L2InputError(f"symbol {symbol} capture_id is invalid")
    intervals = _parse_intervals(payload, symbol=symbol, start_ns=start_ns, end_ns=end_ns)
    return VerifiedL2SymbolInput(
        symbol=symbol,
        capture_id=capture_id,
        normalized_rows=normalized_rows,
        reconstructed_rows=reconstructed_rows,
        excluded_rows=excluded_rows,
        valid_observed_intervals=intervals,
        raw_journal=raw_journal,
        capture_summary=capture_summary,
        depth_deltas=depth,
        book_observations=books,
    )


def _load_verified_parquet(
    root: Path,
    *,
    expected_root: _RootIdentity,
    artifact: VerifiedL2Artifact,
) -> tuple[pl.DataFrame, int]:
    if artifact.dataset not in {"book_observations", "depth_deltas"} or artifact.rows is None:
        raise M8L2InputError("requested artifact is not a research Parquet input")
    if artifact.bytes < 12 or artifact.bytes > _MAX_PARQUET_FILE_BYTES:
        raise M8L2InputError(f"{artifact.dataset} Parquet exceeds its physical byte bound")
    root_descriptor = _open_root(root, expected_root)
    try:
        parent, descriptor, leaf = _open_relative(root_descriptor, artifact.relative_path)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != artifact.bytes:
                raise M8L2InputError(f"{artifact.dataset} is not its claimed regular file")
            if os.pread(descriptor, 4, 0) != b"PAR1":
                raise M8L2InputError(f"{artifact.dataset} lacks its Parquet header")
            trailer = os.pread(descriptor, 8, before.st_size - 8)
            if len(trailer) != 8 or trailer[4:] != b"PAR1":
                raise M8L2InputError(f"{artifact.dataset} lacks its Parquet trailer")
            footer_bytes = int.from_bytes(trailer[:4], "little")
            if footer_bytes > _MAX_PARQUET_FOOTER_BYTES or footer_bytes + 12 > before.st_size:
                raise M8L2InputError(f"{artifact.dataset} Parquet footer exceeds its bound")
            first_digest, first_bytes = _hash_descriptor(descriptor, artifact.bytes)
            if first_digest != artifact.sha256 or first_bytes != artifact.bytes:
                raise M8L2InputError(f"{artifact.dataset} Parquet digest/bytes changed")
            os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                parquet = pq.ParquetFile(handle)
                if parquet.metadata.num_rows != artifact.rows:
                    raise M8L2InputError(f"{artifact.dataset} footer row count differs")
                expected_schema = get_schema(artifact.dataset)
                # Parquet canonicalizes list child names (``item`` -> ``element``)
                # on read, so Arrow's normal structural equality is the registry
                # contract; frozen schema metadata is checked explicitly below.
                if parquet.schema_arrow != expected_schema:
                    raise M8L2InputError(f"{artifact.dataset} Parquet schema differs")
                metadata = parquet.schema_arrow.metadata or {}
                if (
                    metadata.get(b"schema_name") != artifact.dataset.encode()
                    or metadata.get(b"schema_version") != SCHEMA_VERSION.encode()
                ):
                    raise M8L2InputError(f"{artifact.dataset} schema metadata differs")
                uncompressed = sum(
                    parquet.metadata.row_group(index).total_byte_size
                    for index in range(parquet.metadata.num_row_groups)
                )
                if uncompressed > _MAX_PARQUET_UNCOMPRESSED_BYTES:
                    raise M8L2InputError(
                        f"{artifact.dataset} Parquet uncompressed payload exceeds its bound"
                    )
                table = parquet.read()
            ensure_schema(table, artifact.dataset)
            if table.num_rows != artifact.rows:
                raise M8L2InputError(f"{artifact.dataset} materialized row count differs")
            if table.nbytes > _MAX_PARQUET_UNCOMPRESSED_BYTES:
                raise M8L2InputError(
                    f"{artifact.dataset} materialized payload exceeds its memory bound"
                )
            second_digest, second_bytes = _hash_descriptor(descriptor, artifact.bytes)
            after = os.fstat(descriptor)
            _assert_still_named(parent, leaf, after, label=artifact.relative_path)
            if (
                second_digest != artifact.sha256
                or second_bytes != artifact.bytes
                or not _same_metadata(before, after)
            ):
                raise M8L2InputError(f"{artifact.dataset} changed during its descriptor snapshot")
            frame = pl.from_arrow(cast(pa.Table, table))
            if not isinstance(frame, pl.DataFrame):  # pragma: no cover - Arrow table overload
                raise M8L2InputError(f"{artifact.dataset} did not materialize as a frame")
            return frame, uncompressed
        finally:
            os.close(descriptor)
            os.close(parent)
    except M8L2InputError:
        raise
    except (OSError, pa.ArrowException, ValueError) as error:
        raise M8L2InputError(f"cannot securely load {artifact.dataset} Parquet") from error
    finally:
        os.close(root_descriptor)


def _verify_artifact_hash(
    root: Path,
    *,
    expected_root: _RootIdentity,
    artifact: VerifiedL2Artifact,
) -> None:
    """Rebind one requested-symbol lineage artifact through a stable descriptor."""

    root_descriptor = _open_root(root, expected_root)
    try:
        parent, descriptor, leaf = _open_relative(root_descriptor, artifact.relative_path)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != artifact.bytes:
                raise M8L2InputError(f"{artifact.kind} is not its claimed regular file")
            digest, total = _hash_descriptor(descriptor, artifact.bytes)
            after = os.fstat(descriptor)
            _assert_still_named(parent, leaf, after, label=artifact.relative_path)
            if (
                digest != artifact.sha256
                or total != artifact.bytes
                or not _same_metadata(before, after)
            ):
                raise M8L2InputError(
                    f"{artifact.kind} changed during its bounded descriptor snapshot"
                )
        finally:
            os.close(descriptor)
            os.close(parent)
    except M8L2InputError:
        raise
    except OSError as error:
        raise M8L2InputError(f"cannot securely verify {artifact.kind}") from error
    finally:
        os.close(root_descriptor)


def _all_true(frame: pl.DataFrame, expression: pl.Expr) -> bool:
    value = frame.select(expression.all()).item()
    return value is True


def _validate_loaded_frames(
    books: pl.DataFrame,
    deltas: pl.DataFrame,
    *,
    descriptor: VerifiedL2SymbolInput,
    session_start_ns: int,
    session_end_ns: int,
) -> None:
    if books.height != descriptor.reconstructed_rows or deltas.height != descriptor.normalized_rows:
        raise M8L2InputError(f"symbol {descriptor.symbol} materialized rows do not reconcile")
    if deltas.height - books.height != descriptor.excluded_rows:
        raise M8L2InputError(f"symbol {descriptor.symbol} excluded rows do not reconcile")
    for label, frame in (("book_observations", books), ("depth_deltas", deltas)):
        if frame.is_empty():
            raise M8L2InputError(f"symbol {descriptor.symbol} {label} is empty")
        if (
            frame.get_column("symbol").null_count()
            or frame.get_column("continuity_id").null_count()
        ):
            raise M8L2InputError(f"symbol {descriptor.symbol} {label} lacks live identity")
        if frame.get_column("symbol").unique().to_list() != [descriptor.symbol]:
            raise M8L2InputError(f"symbol {descriptor.symbol} {label} contains another symbol")
        if not _all_true(
            frame,
            (pl.col("available_ts_ns") >= session_start_ns)
            & (pl.col("available_ts_ns") < session_end_ns)
            & pl.col("received_ts_ns").is_not_null()
            & (pl.col("available_ts_ns") >= pl.col("received_ts_ns")),
        ):
            raise M8L2InputError(f"symbol {descriptor.symbol} {label} escapes session timing")
    if not _all_true(
        books,
        pl.col("is_valid")
        & (pl.col("sequence_start") <= pl.col("sequence_end"))
        & (pl.col("best_bid") < pl.col("best_ask")),
    ):
        raise M8L2InputError(f"symbol {descriptor.symbol} book observations are invalid")
    if not _all_true(deltas, pl.col("first_update_id") <= pl.col("last_update_id")):
        raise M8L2InputError(f"symbol {descriptor.symbol} depth sequences are invalid")

    delta_keys = deltas.select(
        "continuity_id",
        pl.col("first_update_id").alias("sequence_start"),
        pl.col("last_update_id").alias("sequence_end"),
        "available_ts_ns",
    ).unique()
    reconciled = books.select(
        "continuity_id", "sequence_start", "sequence_end", "available_ts_ns"
    ).join(
        delta_keys,
        on=["continuity_id", "sequence_start", "sequence_end", "available_ts_ns"],
        how="anti",
    )
    if reconciled.height:
        raise M8L2InputError(
            f"symbol {descriptor.symbol} book rows do not reconcile to normalized deltas"
        )

    for interval in descriptor.valid_observed_intervals:
        book_slice = books.filter(
            (pl.col("continuity_id") == interval.continuity_id)
            & (pl.col("available_ts_ns") >= interval.start_received_ns)
            & (pl.col("available_ts_ns") < interval.end_received_ns_exclusive)
        )
        delta_slice = deltas.filter(
            (pl.col("continuity_id") == interval.continuity_id)
            & (pl.col("available_ts_ns") >= interval.start_received_ns)
            & (pl.col("available_ts_ns") < interval.end_received_ns_exclusive)
        )
        if book_slice.is_empty() or delta_slice.is_empty():
            raise M8L2InputError(
                f"symbol {descriptor.symbol} OBSERVED interval lacks reconciled rows"
            )


def _verify_input(
    bundle_dir: str | Path,
    *,
    expected_config: M8L2StudyConfig,
    expected_date: str,
    expected_role: str,
    access_phase: L2InputAccessPhase,
    development_lock_sha256: str | None,
    expected_file_authority: L2SessionFileAuthority | None,
    expected_campaign: L2CampaignRuntimeIdentity | None,
) -> VerifiedL2SessionInput:
    root = _absolute_without_resolve(bundle_dir)
    try:
        bundle: M8L2SessionBundle = verify_m8_l2_session_bundle(
            root, expected_config=expected_config
        )
    except M8L2VerificationError as error:
        raise M8L2InputError(f"session capture authority failed verification: {error}") from error
    if bundle.status != "COMPLETE" or bundle.reason_codes:
        raise M8L2InputError("L2 research inputs require a gate-complete session")
    if bundle.root != root:
        raise M8L2InputError("capture verifier returned a different session root")
    if bundle.session_date != expected_date or bundle.role != expected_role:
        raise M8L2InputError("session date/role differs from the requested frozen coordinate")

    manifest_raw, root_identity = _read_control_file(
        root,
        expected_root=None,
        relative=_MANIFEST_NAME,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    checksums_raw, repeated_root = _read_control_file(
        root,
        expected_root=root_identity,
        relative=_CHECKSUM_NAME,
        maximum_bytes=_MAX_CHECKSUM_BYTES,
    )
    if repeated_root != root_identity:
        raise M8L2InputError("session root changed while binding control authority")
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    checksums_sha256 = hashlib.sha256(checksums_raw).hexdigest()
    discovered_authority = L2SessionFileAuthority(manifest_sha256, checksums_sha256)
    if manifest_sha256 != bundle.manifest_sha256:
        raise M8L2InputError("secure manifest snapshot differs from capture verifier authority")
    if expected_file_authority is not None and discovered_authority != expected_file_authority:
        raise M8L2InputError("session control files differ from external digest authority")
    checksums = _parse_checksums(checksums_raw)
    if checksums.get(_MANIFEST_NAME) != manifest_sha256:
        raise M8L2InputError("session checksum file does not bind its manifest")

    payload = _json_object(manifest_raw, "session manifest")
    if payload.get("status") != "COMPLETE" or payload.get("reason_codes") != []:
        raise M8L2InputError("session manifest is not gate-complete")
    gates = payload.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != 29
        or any(not isinstance(item, Mapping) or item.get("passed") is not True for item in gates)
    ):
        raise M8L2InputError("session does not contain the 29 passing frozen gates")
    session_payload = _mapping(payload.get("session"), "session coordinates")
    if session_payload.get("date") != expected_date or session_payload.get("role") != expected_role:
        raise M8L2InputError("manifest session coordinate differs from its requested role")
    expected_session = expected_config.session_for_date(expected_date)
    if expected_session.role != expected_role:
        raise M8L2InputError("requested role differs from the frozen calendar")
    start_ns = expected_session.start_ns
    end_ns = expected_session.end_ns
    if (
        session_payload.get("scheduled_start_ns") != start_ns
        or session_payload.get("scheduled_end_ns") != end_ns
    ):
        raise M8L2InputError("manifest time bounds differ from the frozen calendar")

    authority = _mapping(payload.get("authority"), "session campaign authority")
    campaign = L2CampaignRuntimeIdentity(
        campaign_authority_sha256=str(authority.get("campaign_authority_sha256")),
        runtime_commit=str(authority.get("runtime_commit")),
        runtime_source_tree_sha256=str(authority.get("runtime_source_tree_sha256")),
        runtime_fingerprint_sha256=str(authority.get("runtime_fingerprint_sha256")),
        runtime_dirty=authority.get("runtime_dirty") is not False,
    )
    if expected_campaign is not None and campaign != expected_campaign:
        raise M8L2InputError("session campaign/runtime identity differs from prior authority")
    config_sha256 = authority.get("config_sha256")
    config_source_sha256 = authority.get("config_source_sha256")
    if (
        config_sha256 != expected_config.hash
        or config_source_sha256 != expected_config.source_sha256
    ):
        raise M8L2InputError("session configuration identity differs from the caller authority")

    symbols_payload = _mapping(payload.get("symbols"), "session symbols")
    if set(symbols_payload) != set(expected_config.study.symbols):
        raise M8L2InputError("session does not contain the exact frozen symbol pair")
    descriptors = {
        symbol: _symbol_descriptor(
            root,
            expected_root=root_identity,
            symbol=symbol,
            payload=_mapping(symbols_payload[symbol], f"symbol {symbol}"),
            start_ns=start_ns,
            end_ns=end_ns,
        )
        for symbol in expected_config.study.symbols
    }

    expected_marker = "_SUCCESS"
    expected_paths = frozenset({*checksums, _CHECKSUM_NAME, expected_marker})
    _assert_inventory(root, expected_root=root_identity, expected_paths=expected_paths)
    return VerifiedL2SessionInput(
        root=root,
        session_id=bundle.session_id,
        session_date=expected_date,
        role=expected_role,
        config_sha256=cast(str, config_sha256),
        config_source_sha256=cast(str, config_source_sha256),
        file_authority=discovered_authority,
        campaign_identity=campaign,
        symbols=MappingProxyType(descriptors),
        access_phase=access_phase,
        development_lock_sha256=development_lock_sha256,
        _root_identity=root_identity,
        _expected_paths=expected_paths,
        _session_start_ns=start_ns,
        _session_end_ns=end_ns,
        _verification_token=_VERIFIED_INPUT_TOKEN,
    )


def verify_m8_l2_development_input(
    bundle_dir: str | Path,
    *,
    expected_config: M8L2StudyConfig,
    expected_date: str,
    expected_role: DevelopmentRole,
    expected_file_authority: L2SessionFileAuthority | None = None,
    expected_campaign: L2CampaignRuntimeIdentity | None = None,
) -> VerifiedL2SessionInput:
    """Verify/load-authorize only the frozen train or validation session."""

    development = {
        session.date.isoformat(): session.role for session in expected_config.sessions[:2]
    }
    if expected_role not in {"train", "validation"} or development.get(expected_date) != (
        expected_role
    ):
        raise M8L2InputError("development input must be the frozen train/validation coordinate")
    return _verify_input(
        bundle_dir,
        expected_config=expected_config,
        expected_date=expected_date,
        expected_role=expected_role,
        access_phase="development",
        development_lock_sha256=None,
        expected_file_authority=expected_file_authority,
        expected_campaign=expected_campaign,
    )


def verify_m8_l2_heldout_input(
    bundle_dir: str | Path,
    *,
    expected_config: M8L2StudyConfig,
    expected_date: str,
    expected_role: HeldoutRole,
    development_lock_sha256: str,
    expected_file_authority: L2SessionFileAuthority | None = None,
    expected_campaign: L2CampaignRuntimeIdentity | None = None,
) -> VerifiedL2SessionInput:
    """Verify/load-authorize one held-out session after a lock digest is known."""

    _require_sha256(development_lock_sha256, "development lock authority")
    heldout = {session.date.isoformat(): session.role for session in expected_config.sessions[2:]}
    if (
        expected_role not in {"primary_test", "replication_test"}
        or heldout.get(expected_date) != expected_role
    ):
        raise M8L2InputError("held-out input must be the frozen primary/replication coordinate")
    return _verify_input(
        bundle_dir,
        expected_config=expected_config,
        expected_date=expected_date,
        expected_role=expected_role,
        access_phase="heldout_after_lock",
        development_lock_sha256=development_lock_sha256,
        expected_file_authority=expected_file_authority,
        expected_campaign=expected_campaign,
    )


__all__ = [
    "L2CampaignRuntimeIdentity",
    "L2SessionFileAuthority",
    "LoadedL2SymbolFrames",
    "M8L2InputError",
    "VerifiedL2Artifact",
    "VerifiedL2SessionInput",
    "VerifiedL2SymbolInput",
    "verify_m8_l2_development_input",
    "verify_m8_l2_heldout_input",
]
