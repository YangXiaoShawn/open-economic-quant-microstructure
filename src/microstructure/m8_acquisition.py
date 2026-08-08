"""Outcome-blind raw acquisition authority for the frozen M8 study.

This module is deliberately unable to normalize an aggregate-trade row.  It
captures the two permitted exchangeInfo responses, authenticates the eight
official daily ZIPs, and inspects ZIP directory metadata only.  The CSV member
is first opened later by :mod:`microstructure.m8_normalization`, after the
analysis lock has become durable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from datetime import date as Date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast
from urllib.parse import parse_qsl, urlparse

from microstructure.data.binance import (
    BinanceHTTPError,
    BinanceMetadataContractError,
    BinancePublicClient,
    BinanceResponseSizeLimitError,
    SymbolMetadata,
)
from microstructure.data.binance_archive import (
    AcquiredDailyArchive,
    ArchiveDownloadLimits,
    BinanceArchiveClient,
    BinanceArchiveContractError,
    BinanceArchiveHTTPError,
    DailyArchiveRequest,
    RawArchiveArtifact,
)
from microstructure.data.evidence_budget import EvidenceBudgetExceeded, RetainedEvidenceBudget
from microstructure.m8_config import M8PeriodRole, M8StudyConfig, load_m8_config
from microstructure.provenance import sha256_file

M8_ACQUISITION_SCHEMA_VERSION = "1.0.0"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_NAME = re.compile(r"^m8-acquisition\.manifest-([0-9a-f]{20})\.json$")
_SOURCE_MANIFEST_NAME = re.compile(r"^.+\.manifest-[0-9a-f]{20}\.json$")
_SAFE_SYMBOL = re.compile(r"^[A-Z0-9]{2,20}$")
_UTC_TIMESTAMP = re.compile(
    r"^(?P<second>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_MAX_EOCD_BYTES = 22 + 65_535
_MAX_ZIP_DIRECTORY_BYTES = 256 * 1024
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_STRUCT = struct.Struct("<4s4H2LH")
_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_LOCAL_FILE_STRUCT = struct.Struct("<4s5H3L2H")
_RAW_ROOT_NAME = "raw"
_MANIFEST_DIRECTORY_NAME = "_manifests"
_ATTEMPT_DIRECTORY_NAME = "_attempts"
_FAILURE_MANIFEST_NAME = "failure.json"
_FAILURE_CHECKSUMS_NAME = "checksums.sha256"
_FAILURE_TERMINAL_NAME = "INSUFFICIENT_DATA"
_FAILURE_TERMINAL_BYTES = b"terminal\n"
_FAILURE_ATTEMPT_NAME = re.compile(r"^m8-acquisition-attempt-([0-9a-f]{20})$")
_RAW_SIDECAR_KEYS = frozenset(
    {
        "artifact_kind",
        "bytes",
        "checksum",
        "downloaded_at_utc",
        "manifest_version",
        "path",
        "requested_range_ns",
        "response_headers",
        "source",
        "source_uri",
        "upstream_checksum_sha256",
    }
)

RetainedArtifactKind = Literal[
    "metadata_body",
    "archive_zip",
    "archive_checksum",
    "rejected_prefix",
    "source_manifest",
]

M8AcquisitionReasonCode = Literal[
    "DECLARED_OBJECT_UNAVAILABLE",
    "METADATA_CONTRACT",
    "CHECKSUM_CONTRACT",
    "ZIP_CONTRACT",
    "RESPONSE_SIZE_LIMIT",
    "TOTAL_EVIDENCE_BUDGET",
]

M8AcquisitionStepKind = Literal["metadata", "archive"]


class M8AcquisitionError(RuntimeError):
    """Raised when an M8 raw acquisition authority cannot be trusted."""


class _M8MetadataResponseContractError(M8AcquisitionError):
    """A deterministic contract defect in an already-retained exchangeInfo body."""


@dataclass(frozen=True, slots=True)
class _RegularFileSnapshot:
    content: bytes
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


class _MetadataProvider(Protocol):
    def fetch_exchange_info(self, *, symbol: str, raw_root: str | Path) -> SymbolMetadata: ...


class _ArchiveProvider(Protocol):
    def acquire(
        self,
        request: DailyArchiveRequest,
        *,
        raw_root: str | Path,
        limits: ArchiveDownloadLimits,
    ) -> AcquiredDailyArchive: ...


@dataclass(frozen=True, slots=True)
class M8RetainedArtifact:
    """One physically retained raw-response body or source sidecar."""

    path: str
    sha256: str
    bytes: int
    kind: RetainedArtifactKind
    source_uri: str
    paired_body_path: str | None


@dataclass(frozen=True, slots=True)
class M8RawSymbolMetadata:
    """Verified exchangeInfo evidence used only for tick/lot normalization."""

    venue: str
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    lot_size: Decimal
    min_price: Decimal
    max_price: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    observed_ts_ns: int
    raw_path: Path
    raw_sha256: str
    raw_bytes: int
    source_uri: str
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest_bytes: int


@dataclass(frozen=True, slots=True)
class M8RawArchiveEntry:
    """Raw-only descriptor for one frozen symbol/date archive."""

    root: Path
    symbol: str
    date: Date
    role: M8PeriodRole
    tick_size: Decimal
    lot_size: Decimal
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    archive_source_uri: str
    archive_source_manifest_path: Path
    archive_source_manifest_sha256: str
    archive_source_manifest_bytes: int
    checksum_path: Path
    checksum_sha256: str
    checksum_bytes: int
    checksum_source_uri: str
    checksum_source_manifest_path: Path
    checksum_source_manifest_sha256: str
    checksum_source_manifest_bytes: int
    upstream_sha256: str
    member_name: str
    declared_uncompressed_bytes: int
    max_compressed_bytes: int
    max_uncompressed_bytes: int
    max_checksum_bytes: int
    transfer_chunk_bytes: int
    max_csv_line_bytes: int
    csv_member_opened: bool = False
    economic_fields_inspected: bool = False


@dataclass(frozen=True, slots=True)
class M8RawArchiveDescriptor:
    """A verified, reconstructable archive reference with no row-reading API."""

    entry: M8RawArchiveEntry

    @property
    def symbol(self) -> str:
        return self.entry.symbol

    @property
    def date(self) -> Date:
        return self.entry.date

    @property
    def role(self) -> M8PeriodRole:
        return self.entry.role

    def reconstruct(self) -> AcquiredDailyArchive:
        """Re-hash raw bytes and reconstruct the authenticated archive handle."""

        _verify_archive_entry_files(self.entry)
        request = DailyArchiveRequest(
            symbol=self.entry.symbol,
            date=self.entry.date,
            tick_size=self.entry.tick_size,
            lot_size=self.entry.lot_size,
        )
        limits = ArchiveDownloadLimits(
            max_compressed_bytes=self.entry.max_compressed_bytes,
            max_uncompressed_bytes=self.entry.max_uncompressed_bytes,
            max_checksum_bytes=self.entry.max_checksum_bytes,
            transfer_chunk_bytes=self.entry.transfer_chunk_bytes,
            max_csv_line_bytes=self.entry.max_csv_line_bytes,
        )
        return AcquiredDailyArchive(
            request=request,
            archive_artifact=RawArchiveArtifact(
                kind="archive_zip",
                path=self.entry.archive_path,
                manifest_path=self.entry.archive_source_manifest_path,
                sha256=self.entry.archive_sha256,
                manifest_sha256=self.entry.archive_source_manifest_sha256,
                bytes=self.entry.archive_bytes,
                source_uri=self.entry.archive_source_uri,
            ),
            checksum_artifact=RawArchiveArtifact(
                kind="archive_checksum",
                path=self.entry.checksum_path,
                manifest_path=self.entry.checksum_source_manifest_path,
                sha256=self.entry.checksum_sha256,
                manifest_sha256=self.entry.checksum_source_manifest_sha256,
                bytes=self.entry.checksum_bytes,
                source_uri=self.entry.checksum_source_uri,
            ),
            upstream_sha256=self.entry.upstream_sha256,
            declared_uncompressed_bytes=self.entry.declared_uncompressed_bytes,
            limits=limits,
            requires_member_open_guard=self.entry.role in {"primary_test", "replication_test"},
        )


@dataclass(frozen=True, slots=True)
class M8AcquisitionManifest:
    """Verified authority for the exact raw evidence of one M8 study."""

    root: Path
    path: Path
    sha256: str
    config_sha256: str
    config_source_sha256: str
    protocol_version: str
    protocol_document_sha256: str
    copied_from_manifest_sha256: str | None
    evidence_set_sha256: str
    symbol_metadata: tuple[M8RawSymbolMetadata, ...]
    archives: tuple[M8RawArchiveEntry, ...]
    retained_artifacts: tuple[M8RetainedArtifact, ...]
    total_raw_evidence_bytes: int
    total_accepted_zip_bytes: int
    config: M8StudyConfig

    @property
    def entries(self) -> tuple[M8RawArchiveEntry, ...]:
        return self.archives

    @property
    def metadata_count(self) -> int:
        return len(self.symbol_metadata)

    @property
    def archive_count(self) -> int:
        return len(self.archives)

    @property
    def content_identity_sha256(self) -> str:
        """Root-independent identity of every semantic claim and raw byte."""

        return self.evidence_set_sha256

    def metadata_for(self, symbol: str) -> M8RawSymbolMetadata:
        for metadata in self.symbol_metadata:
            if metadata.symbol == symbol:
                return metadata
        raise KeyError(f"no verified M8 symbol metadata for {symbol}")

    def archive_descriptor_for(self, symbol: str, date: Date | str) -> M8RawArchiveDescriptor:
        date_text = date.isoformat() if isinstance(date, Date) else date
        for entry in self.archives:
            if entry.symbol == symbol and entry.date.isoformat() == date_text:
                return M8RawArchiveDescriptor(entry)
        raise KeyError(f"no verified M8 raw archive for {symbol}/{date_text}")


@dataclass(frozen=True, slots=True)
class M8AcquisitionResult:
    """Stable CLI-facing result for one completed raw-only acquisition."""

    output_root: Path
    manifest_path: Path
    manifest_sha256: str
    metadata_count: int
    archive_count: int
    total_raw_evidence_bytes: int
    manifest: M8AcquisitionManifest

    @property
    def status(self) -> Literal["ACQUIRED"]:
        return "ACQUIRED"


@dataclass(frozen=True, slots=True)
class M8AcquisitionStep:
    """One declared raw acquisition unit in frozen execution order."""

    kind: M8AcquisitionStepKind
    symbol: str
    date: Date | None
    role: str


@dataclass(frozen=True, slots=True)
class M8AcquisitionFailureManifest:
    """Verified immutable authority for a deterministic raw-only failure."""

    root: Path
    attempt_dir: Path
    path: Path
    sha256: str
    checksums_path: Path
    checksums_sha256: str
    terminal_path: Path
    reason_code: M8AcquisitionReasonCode
    diagnostic: str
    failed_step: M8AcquisitionStep
    completed_steps: tuple[M8AcquisitionStep, ...]
    remaining_steps: tuple[M8AcquisitionStep, ...]
    retained_artifacts: tuple[M8RetainedArtifact, ...]
    retained_inventory_sha256: str
    total_raw_evidence_bytes: int
    config: M8StudyConfig


@dataclass(frozen=True, slots=True)
class M8AcquisitionFailureResult:
    """CLI-facing result for one immutable deterministic acquisition failure."""

    output_root: Path
    attempt_dir: Path
    attempt_manifest_path: Path
    attempt_manifest_sha256: str
    checksums_path: Path
    checksums_sha256: str
    terminal_path: Path
    reason_code: M8AcquisitionReasonCode
    diagnostic: str
    failed_symbol: str
    failed_date: Date | None
    failed_role: str
    completed_count: int
    remaining_count: int
    retained_inventory_sha256: str
    retained_artifact_count: int
    total_raw_evidence_bytes: int
    manifest: M8AcquisitionFailureManifest
    status: Literal["INSUFFICIENT_DATA"] = "INSUFFICIENT_DATA"


M8AcquisitionOutcome = M8AcquisitionResult | M8AcquisitionFailureResult


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise M8AcquisitionError(f"{label} keys differ: missing={missing}, extra={extra}")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8AcquisitionError(f"{label} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise M8AcquisitionError(f"{label} must be a JSON array")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise M8AcquisitionError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise M8AcquisitionError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise M8AcquisitionError(f"{label} must be a boolean")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if _DIGEST.fullmatch(text) is None:
        raise M8AcquisitionError(f"{label} must be one lowercase SHA-256")
    return text


def _decimal(value: object, label: str, *, positive: bool = False) -> Decimal:
    text = _text(value, label)
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise M8AcquisitionError(f"{label} must be a canonical decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0) or format(parsed, "f") != text:
        raise M8AcquisitionError(f"{label} is not a valid canonical decimal")
    return parsed


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_bounded_regular_snapshot(
    path: Path,
    label: str,
    *,
    byte_limit: int,
) -> _RegularFileSnapshot:
    """Read one regular file from one fd and reject path/inode replacement."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise M8AcquisitionError(f"{label} is not a regular file")
            if before.st_size > byte_limit:
                raise M8AcquisitionError(f"{label} exceeds its byte ceiling")
            content = source.read(byte_limit + 1)
            after = os.fstat(source.fileno())
            if len(content) > byte_limit:
                raise M8AcquisitionError(f"{label} exceeds its byte ceiling")
            if len(content) != before.st_size or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise M8AcquisitionError(f"{label} changed while it was read")
            linked = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(linked.st_mode) or (linked.st_dev, linked.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise M8AcquisitionError(f"{label} path changed while it was read")
            return _RegularFileSnapshot(
                content=content,
                device=before.st_dev,
                inode=before.st_ino,
                size=before.st_size,
                modified_ns=before.st_mtime_ns,
                changed_ns=before.st_ctime_ns,
            )
    except M8AcquisitionError:
        raise
    except OSError as exc:
        raise M8AcquisitionError(f"cannot read {label}: {path}") from exc


def _assert_snapshot_path(path: Path, snapshot: _RegularFileSnapshot, label: str) -> None:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise M8AcquisitionError(f"{label} path disappeared after snapshot") from exc
    if not stat.S_ISREG(observed.st_mode) or (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    ) != (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.modified_ns,
        snapshot.changed_ns,
    ):
        raise M8AcquisitionError(f"{label} path changed after snapshot")


def _read_bounded_regular_bytes(path: Path, label: str, *, byte_limit: int) -> bytes:
    return _read_bounded_regular_snapshot(path, label, byte_limit=byte_limit).content


def _parse_json_bytes(content: bytes, label: str) -> Mapping[str, Any]:
    try:

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise M8AcquisitionError(f"{label} contains duplicate key {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise M8AcquisitionError(f"{label} contains forbidden JSON constant {value}")

        parsed = json.loads(
            content,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except M8AcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M8AcquisitionError(f"cannot parse {label}") from exc
    return _object(parsed, label)


def _read_json(
    path: Path, label: str, *, byte_limit: int = _MAX_MANIFEST_BYTES
) -> Mapping[str, Any]:
    try:
        content = _read_bounded_regular_bytes(path, label, byte_limit=byte_limit)
        if len(content) > byte_limit:
            raise M8AcquisitionError(f"{label} exceeds its JSON byte ceiling")
        return _parse_json_bytes(content, label)
    except M8AcquisitionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M8AcquisitionError(f"cannot parse {label}: {path}") from exc


def _protocol_path(config: M8StudyConfig) -> Path:
    path = config.path.parent.parent / "docs" / "M8_MULTIDATE_TRADE_PROTOCOL.md"
    if not path.is_file() or path.is_symlink():
        raise M8AcquisitionError(f"frozen M8 protocol document is missing: {path}")
    return path.resolve()


def _verify_config_sources(config: M8StudyConfig) -> str:
    if not config.path.is_file() or config.path.is_symlink():
        raise M8AcquisitionError("frozen M8 machine specification is missing or symbolic")
    if sha256_file(config.path) != config.source_sha256:
        raise M8AcquisitionError("frozen M8 machine specification bytes changed")
    try:
        reloaded = load_m8_config(config.path)
    except Exception as exc:
        raise M8AcquisitionError("cannot re-validate frozen M8 machine specification") from exc
    if reloaded.hash != config.hash or reloaded.source_sha256 != config.source_sha256:
        raise M8AcquisitionError("in-memory M8 configuration differs from its frozen source")
    return sha256_file(_protocol_path(config))


def _relative(root: Path, path: Path, label: str) -> str:
    root_resolved = root.resolve()
    path_absolute = path.absolute()
    if path_absolute.is_symlink():
        raise M8AcquisitionError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise M8AcquisitionError(f"{label} is missing or escapes acquisition root: {path}")
    relative = resolved.relative_to(root_resolved).as_posix()
    if relative.startswith(f"{_MANIFEST_DIRECTORY_NAME}/"):
        raise M8AcquisitionError(f"{label} incorrectly points into manifest storage")
    return relative


def _declared_file(root: Path, value: object, label: str) -> Path:
    raw = _text(value, label)
    declared = Path(raw)
    if (
        declared.is_absolute()
        or "\\" in raw
        or raw != declared.as_posix()
        or not declared.parts
        or any(part in {"", ".", ".."} for part in declared.parts)
        or declared.parts[0] != _RAW_ROOT_NAME
    ):
        raise M8AcquisitionError(f"{label} must be one canonical relative raw-evidence path")
    current = root.resolve()
    for component in declared.parts:
        current /= component
        if current.is_symlink():
            raise M8AcquisitionError(f"{label} traverses a symbolic link: {current}")
    resolved = current.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise M8AcquisitionError(f"{label} is missing or escapes acquisition root: {raw}")
    return resolved


def _verify_file(path: Path, digest: str, byte_count: int, label: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise M8AcquisitionError(f"{label} is missing or symbolic: {path}")
        observed_bytes = path.stat().st_size
        if observed_bytes != byte_count:
            raise M8AcquisitionError(
                f"{label} byte count changed: expected {byte_count}, observed {observed_bytes}"
            )
        observed_sha = sha256_file(path)
    except M8AcquisitionError:
        raise
    except OSError as exc:
        raise M8AcquisitionError(f"cannot verify {label}: {path}") from exc
    if observed_sha != digest:
        raise M8AcquisitionError(
            f"{label} SHA-256 changed: expected {digest}, observed {observed_sha}"
        )


def _utc_ns(value: object, label: str) -> int:
    text = _text(value, label)
    match = _UTC_TIMESTAMP.fullmatch(text)
    if match is None:
        raise M8AcquisitionError(f"{label} must be a canonical UTC timestamp")
    try:
        second = datetime.strptime(match.group("second"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    except ValueError as exc:
        raise M8AcquisitionError(f"{label} is not a real UTC timestamp") from exc
    fraction = (match.group("fraction") or "").ljust(9, "0")
    seconds = int(second.timestamp())
    result = seconds * 1_000_000_000 + int(fraction or "0")
    if result < 1:
        raise M8AcquisitionError(f"{label} must be after the Unix epoch")
    return result


def _day_bounds_ns(day: Date) -> tuple[int, int]:
    start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 1_000_000_000
    return start, start + 86_400 * 1_000_000_000


def _expected_archive_uri(symbol: str, day: Date, *, checksum: bool = False) -> str:
    name = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    suffix = ".CHECKSUM" if checksum else ""
    return f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{name}{suffix}"


def _verify_exchange_info_uri(value: str, symbol: str) -> None:
    try:
        parsed = urlparse(value)
        port = parsed.port
        query = parse_qsl(
            parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2
        )
    except ValueError as exc:
        raise M8AcquisitionError("exchangeInfo source URI is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != "data-api.binance.vision"
        or parsed.hostname != "data-api.binance.vision"
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/api/v3/exchangeInfo"
        or parsed.params
        or parsed.fragment
        or query != [("symbol", symbol)]
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise M8AcquisitionError(
            f"metadata source URI is not the exact official exchangeInfo request for {symbol}"
        )


def _verify_official_uri(value: str, expected: str, label: str) -> None:
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as exc:
        raise M8AcquisitionError(f"{label} is malformed") from exc
    expected_parsed = urlparse(expected)
    if (
        value != expected
        or parsed.scheme != "https"
        or parsed.netloc != "data.binance.vision"
        or parsed.hostname != "data.binance.vision"
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise M8AcquisitionError(f"{label} is not the exact official Binance archive URI")


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise M8AcquisitionError("metadata contains a non-finite decimal")
    return format(value, "f")


def _expected_raw_path(root: Path, relative: str, observed: Path, label: str) -> None:
    expected = (root / relative).resolve()
    if observed.resolve() != expected:
        raise M8AcquisitionError(f"{label} is not stored at its canonical raw path")


def _source_manifest_path_is_canonical(body: Path, sidecar: Path, label: str) -> None:
    if sidecar.parent != body.parent or _SOURCE_MANIFEST_NAME.fullmatch(sidecar.name) is None:
        raise M8AcquisitionError(f"{label} path is not the canonical content-addressed sidecar")
    if not sidecar.name.startswith(f"{body.name}.manifest-"):
        raise M8AcquisitionError(f"{label} does not name its paired raw body")


def _verify_source_manifest_identity(
    body: Path,
    sidecar: Path,
    payload: Mapping[str, Any],
    label: str,
) -> None:
    try:
        stable = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M8AcquisitionError(f"{label} is not stable JSON") from exc
    identity = hashlib.sha256(stable).hexdigest()
    if sidecar.name != f"{body.name}.manifest-{identity[:20]}.json":
        raise M8AcquisitionError(f"{label} filename is not content-addressed by its claims")


def _verify_source_sidecar(
    sidecar_path: Path,
    *,
    body_path: Path,
    body_sha256: str,
    body_bytes: int,
    expected_source: str,
    expected_source_uri: str,
    expected_range: tuple[int | None, int | None],
    expected_upstream: str | None,
    label: str,
) -> Mapping[str, Any]:
    _source_manifest_path_is_canonical(body_path, sidecar_path, label)
    sidecar = _read_json(sidecar_path, label)
    _verify_source_manifest_identity(body_path, sidecar_path, sidecar, label)
    _require_exact_keys(sidecar, _RAW_SIDECAR_KEYS, label)
    if sidecar["manifest_version"] != "1.0.0" or sidecar["artifact_kind"] != "raw_source":
        raise M8AcquisitionError(f"{label} has unsupported source-manifest identity")
    expected: dict[str, object] = {
        "source": expected_source,
        "source_uri": expected_source_uri,
        "bytes": body_bytes,
        "path": body_path.name,
        "upstream_checksum_sha256": expected_upstream,
    }
    if any(sidecar[key] != value for key, value in expected.items()):
        raise M8AcquisitionError(f"{label} lineage claims do not match its raw body")
    checksum = _object(sidecar["checksum"], f"{label}.checksum")
    _require_exact_keys(checksum, frozenset({"algorithm", "value"}), f"{label}.checksum")
    if checksum["algorithm"] != "sha256" or checksum["value"] != body_sha256:
        raise M8AcquisitionError(f"{label} checksum claim does not match its raw body")
    requested = _object(sidecar["requested_range_ns"], f"{label}.requested_range_ns")
    _require_exact_keys(
        requested,
        frozenset({"start", "end_exclusive"}),
        f"{label}.requested_range_ns",
    )
    if (requested["start"], requested["end_exclusive"]) != expected_range:
        raise M8AcquisitionError(f"{label} requested range is not the frozen range")
    headers = _object(sidecar["response_headers"], f"{label}.response_headers")
    if not all(type(key) is str and type(value) is str for key, value in headers.items()):
        raise M8AcquisitionError(f"{label} response headers must contain only strings")
    _utc_ns(sidecar["downloaded_at_utc"], f"{label}.downloaded_at_utc")
    return sidecar


def _preflight_zip_directory(path: Path) -> tuple[int, int]:
    """Bound EOCD and central-directory bytes before ``zipfile`` sees them."""

    try:
        size = path.stat().st_size
        if size < _EOCD_STRUCT.size:
            raise M8AcquisitionError("M8 archive is too small for a ZIP directory")
        tail_size = min(size, _MAX_EOCD_BYTES)
        with path.open("rb") as source:
            source.seek(size - tail_size)
            tail = source.read(tail_size)
        offset = tail.rfind(_EOCD_SIGNATURE)
        if offset < 0 or len(tail) - offset < _EOCD_STRUCT.size:
            raise M8AcquisitionError("M8 archive ZIP end-of-directory record is missing")
        (
            _signature,
            disk,
            central_disk,
            entries_disk,
            entries_total,
            central_bytes,
            central_offset,
            comment_bytes,
        ) = _EOCD_STRUCT.unpack_from(tail, offset)
        absolute_offset = size - tail_size + offset
        if absolute_offset + _EOCD_STRUCT.size + comment_bytes != size:
            raise M8AcquisitionError("M8 archive ZIP has trailing or malformed directory bytes")
        if disk != 0 or central_disk != 0 or entries_disk != 1 or entries_total != 1:
            raise M8AcquisitionError("M8 archive ZIP must contain one single-disk member")
        if central_bytes < 1 or central_bytes > _MAX_ZIP_DIRECTORY_BYTES:
            raise M8AcquisitionError("M8 archive ZIP central directory exceeds its byte ceiling")
        if central_offset + central_bytes != absolute_offset:
            raise M8AcquisitionError("M8 archive ZIP central-directory bounds are invalid")
        return int(central_offset), int(central_bytes)
    except M8AcquisitionError:
        raise
    except (OSError, struct.error) as exc:
        raise M8AcquisitionError("cannot preflight bounded M8 ZIP directory metadata") from exc


def _verify_local_zip_header(
    path: Path,
    member: zipfile.ZipInfo,
    expected_member: str,
    central_offset: int,
) -> None:
    header_offset = int(member.header_offset)
    if header_offset < 0 or header_offset + _LOCAL_FILE_STRUCT.size > central_offset:
        raise M8AcquisitionError("M8 archive ZIP local-header bounds are invalid")
    try:
        with path.open("rb") as source:
            source.seek(header_offset)
            raw_header = source.read(_LOCAL_FILE_STRUCT.size)
            if len(raw_header) != _LOCAL_FILE_STRUCT.size:
                raise M8AcquisitionError("M8 archive ZIP local header is truncated")
            (
                signature,
                _version,
                flags,
                compression,
                _modified_time,
                _modified_date,
                _crc32,
                _compressed_bytes,
                _expanded_bytes,
                filename_bytes,
                extra_bytes,
            ) = _LOCAL_FILE_STRUCT.unpack(raw_header)
            if signature != _LOCAL_FILE_SIGNATURE:
                raise M8AcquisitionError("M8 archive ZIP local-header signature is invalid")
            metadata_bytes = filename_bytes + extra_bytes
            if metadata_bytes > _MAX_ZIP_DIRECTORY_BYTES:
                raise M8AcquisitionError("M8 archive ZIP local metadata exceeds its byte ceiling")
            if header_offset + _LOCAL_FILE_STRUCT.size + metadata_bytes > central_offset:
                raise M8AcquisitionError("M8 archive ZIP local-header metadata bounds are invalid")
            local_name = source.read(filename_bytes)
    except M8AcquisitionError:
        raise
    except (OSError, struct.error) as exc:
        raise M8AcquisitionError("cannot inspect bounded M8 ZIP local metadata") from exc
    if local_name != expected_member.encode("ascii"):
        raise M8AcquisitionError("M8 archive ZIP local member name is unexpected")
    if flags != member.flag_bits or compression != member.compress_type:
        raise M8AcquisitionError("M8 archive ZIP local and central metadata disagree")


def _zip_directory_member(path: Path, expected_member: str, limit: int) -> int:
    """Inspect central-directory metadata without opening the CSV member."""

    central_offset, _central_bytes = _preflight_zip_directory(path)
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise M8AcquisitionError("M8 archive ZIP must contain exactly one member")
            member = members[0]
            mode = member.external_attr >> 16
            if (
                member.filename != expected_member
                or Path(member.filename).name != member.filename
                or "\\" in member.filename
                or member.is_dir()
                or stat.S_ISLNK(mode)
                or member.flag_bits & 0x1
            ):
                raise M8AcquisitionError("M8 archive ZIP member is unsafe or unexpected")
            if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise M8AcquisitionError("M8 archive ZIP compression method is unsupported")
            if member.file_size < 1 or member.file_size > limit:
                raise M8AcquisitionError("M8 archive declared expansion exceeds its ceiling")
            _verify_local_zip_header(path, member, expected_member, central_offset)
            return int(member.file_size)
    except M8AcquisitionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, ValueError) as exc:
        raise M8AcquisitionError("cannot inspect bounded M8 ZIP directory metadata") from exc


def _parse_checksum_body(path: Path, archive_name: str) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise M8AcquisitionError("cannot read official M8 CHECKSUM response") from exc
    endings = (b"", b"\n", b"\r\n")
    for ending in endings:
        suffix = b"  " + archive_name.encode("ascii") + ending
        if len(content) == 64 + len(suffix) and content[64:] == suffix:
            try:
                digest = content[:64].decode("ascii", errors="strict")
            except UnicodeDecodeError as exc:
                raise M8AcquisitionError(
                    "official M8 CHECKSUM digest is not lowercase ASCII"
                ) from exc
            if _DIGEST.fullmatch(digest) is not None:
                return digest
    raise M8AcquisitionError("official M8 CHECKSUM response has malformed exact bytes")


def _metadata_payload_values(
    path: Path,
    symbol: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, object]:
    """Validate response semantics only after a bounded raw-file identity snapshot.

    I/O, path, and hash failures deliberately remain ``M8AcquisitionError``.
    Only parsing and semantic failures attributable to the exact retained body
    become ``_M8MetadataResponseContractError`` and may terminalize acquisition.
    """

    label = f"{symbol} exchangeInfo body"
    snapshot = _read_bounded_regular_snapshot(path, label, byte_limit=_MAX_METADATA_BYTES)
    if snapshot.size != expected_bytes or hashlib.sha256(snapshot.content).hexdigest() != (
        expected_sha256
    ):
        raise M8AcquisitionError(f"{label} identity changed before semantic validation")
    try:
        payload = _parse_json_bytes(snapshot.content, label)
        symbols = _array(payload.get("symbols"), f"{symbol} exchangeInfo symbols")
        if len(symbols) != 1:
            raise M8AcquisitionError(f"{symbol} exchangeInfo must contain exactly one symbol")
        item = _object(symbols[0], f"{symbol} exchangeInfo symbol")
        if item.get("symbol") != symbol:
            raise M8AcquisitionError(f"{symbol} exchangeInfo returned another symbol")
        filters_raw = _array(item.get("filters"), f"{symbol} exchangeInfo filters")
        filters: dict[str, Mapping[str, Any]] = {}
        for index, raw_filter in enumerate(filters_raw):
            filter_item = _object(raw_filter, f"{symbol} exchangeInfo filters[{index}]")
            filter_type = _text(
                filter_item.get("filterType"),
                f"{symbol} exchangeInfo filters[{index}].filterType",
            )
            if filter_type in filters:
                raise M8AcquisitionError(f"{symbol} exchangeInfo repeats filter {filter_type}")
            filters[filter_type] = filter_item
        price = filters["PRICE_FILTER"]
        lot = filters["LOT_SIZE"]
        result: dict[str, object] = {
            "venue": "binance_spot",
            "symbol": symbol,
            "status": _text(item.get("status"), f"{symbol} exchangeInfo status"),
            "base_asset": _text(item.get("baseAsset"), f"{symbol} exchangeInfo baseAsset"),
            "quote_asset": _text(item.get("quoteAsset"), f"{symbol} exchangeInfo quoteAsset"),
            "tick_size": _decimal(price.get("tickSize"), f"{symbol} tickSize", positive=True),
            "lot_size": _decimal(lot.get("stepSize"), f"{symbol} stepSize", positive=True),
            "min_price": _decimal(price.get("minPrice"), f"{symbol} minPrice"),
            "max_price": _decimal(price.get("maxPrice"), f"{symbol} maxPrice"),
            "min_quantity": _decimal(lot.get("minQty"), f"{symbol} minQty"),
            "max_quantity": _decimal(lot.get("maxQty"), f"{symbol} maxQty"),
        }
        if result["status"] != "TRADING":
            raise M8AcquisitionError(f"{symbol} exchangeInfo status is not TRADING")
    except _M8MetadataResponseContractError:
        raise
    except M8AcquisitionError as exc:
        raise _M8MetadataResponseContractError(str(exc)) from exc
    except KeyError as exc:
        raise _M8MetadataResponseContractError(
            f"{symbol} exchangeInfo lacks a frozen filter"
        ) from exc
    _assert_snapshot_path(path, snapshot, label)
    return result


def _verify_metadata_files(root: Path, metadata: M8RawSymbolMetadata) -> None:
    if _SAFE_SYMBOL.fullmatch(metadata.symbol) is None:
        raise M8AcquisitionError("M8 metadata symbol is unsafe")
    if metadata.venue != "binance_spot":
        raise M8AcquisitionError(f"{metadata.symbol} metadata venue is not binance_spot")
    if metadata.observed_ts_ns < 1:
        raise M8AcquisitionError(f"{metadata.symbol} metadata observation time is invalid")
    _verify_exchange_info_uri(metadata.source_uri, metadata.symbol)
    expected_raw = (
        f"{_RAW_ROOT_NAME}/binance_spot/exchange_info/{metadata.symbol}/{metadata.raw_sha256}.json"
    )
    _expected_raw_path(root, expected_raw, metadata.raw_path, f"{metadata.symbol} metadata body")
    _verify_file(
        metadata.raw_path,
        metadata.raw_sha256,
        metadata.raw_bytes,
        f"{metadata.symbol} metadata body",
    )
    _verify_file(
        metadata.source_manifest_path,
        metadata.source_manifest_sha256,
        metadata.source_manifest_bytes,
        f"{metadata.symbol} metadata source sidecar",
    )
    sidecar = _verify_source_sidecar(
        metadata.source_manifest_path,
        body_path=metadata.raw_path,
        body_sha256=metadata.raw_sha256,
        body_bytes=metadata.raw_bytes,
        expected_source="binance_spot_public_api",
        expected_source_uri=metadata.source_uri,
        expected_range=(None, None),
        expected_upstream=None,
        label=f"{metadata.symbol} metadata source sidecar",
    )
    if (
        _utc_ns(sidecar["downloaded_at_utc"], f"{metadata.symbol} metadata timestamp")
        != metadata.observed_ts_ns
    ):
        raise M8AcquisitionError(f"{metadata.symbol} observed time differs from raw sidecar")
    observed = _metadata_payload_values(
        metadata.raw_path,
        metadata.symbol,
        expected_sha256=metadata.raw_sha256,
        expected_bytes=metadata.raw_bytes,
    )
    expected: dict[str, object] = {
        "venue": metadata.venue,
        "symbol": metadata.symbol,
        "status": metadata.status,
        "base_asset": metadata.base_asset,
        "quote_asset": metadata.quote_asset,
        "tick_size": metadata.tick_size,
        "lot_size": metadata.lot_size,
        "min_price": metadata.min_price,
        "max_price": metadata.max_price,
        "min_quantity": metadata.min_quantity,
        "max_quantity": metadata.max_quantity,
    }
    if observed != expected:
        raise M8AcquisitionError(f"{metadata.symbol} parsed metadata differs from raw exchangeInfo")


def _raw_metadata_from_symbol_metadata(
    root: Path,
    metadata: SymbolMetadata,
    expected_symbol: str,
) -> M8RawSymbolMetadata:
    if metadata.symbol != expected_symbol:
        raise M8AcquisitionError(
            f"metadata provider returned {metadata.symbol!r} for {expected_symbol}"
        )
    raw_path = metadata.source_path.resolve()
    sidecar_path = metadata.source_manifest_path.resolve()
    raw_sha = sha256_file(raw_path)
    sidecar_sha = sha256_file(sidecar_path)
    sidecar = _read_json(sidecar_path, f"{expected_symbol} source sidecar")
    source_uri = _text(sidecar.get("source_uri"), f"{expected_symbol} source URI")
    result = M8RawSymbolMetadata(
        venue=metadata.venue,
        symbol=metadata.symbol,
        status=metadata.status,
        base_asset=metadata.base_asset,
        quote_asset=metadata.quote_asset,
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
        min_price=metadata.min_price,
        max_price=metadata.max_price,
        min_quantity=metadata.min_quantity,
        max_quantity=metadata.max_quantity,
        observed_ts_ns=metadata.observed_ts_ns,
        raw_path=raw_path,
        raw_sha256=raw_sha,
        raw_bytes=raw_path.stat().st_size,
        source_uri=source_uri,
        source_manifest_path=sidecar_path,
        source_manifest_sha256=sidecar_sha,
        source_manifest_bytes=sidecar_path.stat().st_size,
    )
    if metadata.source_artifact_id != raw_sha:
        raise M8AcquisitionError(f"{expected_symbol} metadata artifact ID is not its raw SHA")
    _verify_metadata_files(root, result)
    return result


def _verify_archive_entry_files(entry: M8RawArchiveEntry) -> None:
    label = f"{entry.symbol}/{entry.date.isoformat()}"
    if entry.csv_member_opened or entry.economic_fields_inspected:
        raise M8AcquisitionError(f"{label} raw authority claims an economic-data inspection")
    if entry.tick_size <= 0 or entry.lot_size <= 0:
        raise M8AcquisitionError(f"{label} archive scales must be positive")
    archive_name = f"{entry.symbol}-aggTrades-{entry.date.isoformat()}.zip"
    expected_member = archive_name.removesuffix(".zip") + ".csv"
    if entry.member_name != expected_member:
        raise M8AcquisitionError(f"{label} archive declares an unexpected CSV member")
    archive_uri = _expected_archive_uri(entry.symbol, entry.date)
    checksum_uri = _expected_archive_uri(entry.symbol, entry.date, checksum=True)
    _verify_official_uri(entry.archive_source_uri, archive_uri, f"{label} archive URI")
    _verify_official_uri(entry.checksum_source_uri, checksum_uri, f"{label} checksum URI")
    expected_archive_path = (
        f"{_RAW_ROOT_NAME}/binance_spot/daily_agg_trades_archive/{entry.symbol}/"
        f"{entry.date.isoformat()}/{archive_name}"
    )
    expected_checksum_path = (
        f"{_RAW_ROOT_NAME}/binance_spot/daily_agg_trades_archive_checksums/{entry.symbol}/"
        f"{entry.date.isoformat()}/{archive_name}.CHECKSUM"
    )
    _expected_raw_path(entry.root, expected_archive_path, entry.archive_path, f"{label} ZIP")
    _expected_raw_path(entry.root, expected_checksum_path, entry.checksum_path, f"{label} CHECKSUM")
    _verify_file(entry.archive_path, entry.archive_sha256, entry.archive_bytes, f"{label} ZIP")
    _verify_file(
        entry.archive_source_manifest_path,
        entry.archive_source_manifest_sha256,
        entry.archive_source_manifest_bytes,
        f"{label} ZIP source sidecar",
    )
    _verify_file(
        entry.checksum_path, entry.checksum_sha256, entry.checksum_bytes, f"{label} CHECKSUM"
    )
    _verify_file(
        entry.checksum_source_manifest_path,
        entry.checksum_source_manifest_sha256,
        entry.checksum_source_manifest_bytes,
        f"{label} CHECKSUM source sidecar",
    )
    if entry.archive_bytes < 1 or entry.archive_bytes > entry.max_compressed_bytes:
        raise M8AcquisitionError(f"{label} compressed archive exceeds its hard ceiling")
    if entry.checksum_bytes < 1 or entry.checksum_bytes > entry.max_checksum_bytes:
        raise M8AcquisitionError(f"{label} checksum response exceeds its hard ceiling")
    upstream = _parse_checksum_body(entry.checksum_path, archive_name)
    if upstream != entry.upstream_sha256 or upstream != entry.archive_sha256:
        raise M8AcquisitionError(f"{label} official CHECKSUM does not authenticate the ZIP")
    observed_expanded = _zip_directory_member(
        entry.archive_path,
        expected_member,
        entry.max_uncompressed_bytes,
    )
    if observed_expanded != entry.declared_uncompressed_bytes:
        raise M8AcquisitionError(f"{label} ZIP declared expansion changed")
    day_range = _day_bounds_ns(entry.date)
    _verify_source_sidecar(
        entry.archive_source_manifest_path,
        body_path=entry.archive_path,
        body_sha256=entry.archive_sha256,
        body_bytes=entry.archive_bytes,
        expected_source="binance_spot_daily_aggtrades_archive",
        expected_source_uri=entry.archive_source_uri,
        expected_range=day_range,
        expected_upstream=entry.archive_sha256,
        label=f"{label} ZIP source sidecar",
    )
    _verify_source_sidecar(
        entry.checksum_source_manifest_path,
        body_path=entry.checksum_path,
        body_sha256=entry.checksum_sha256,
        body_bytes=entry.checksum_bytes,
        expected_source="binance_spot_daily_aggtrades_archive_checksum",
        expected_source_uri=entry.checksum_source_uri,
        expected_range=day_range,
        expected_upstream=None,
        label=f"{label} CHECKSUM source sidecar",
    )


def _raw_entry_from_acquired(
    root: Path,
    acquired: AcquiredDailyArchive,
    *,
    role: M8PeriodRole,
    expected_symbol: str,
    expected_date: Date,
    limits: ArchiveDownloadLimits,
) -> M8RawArchiveEntry:
    request = acquired.request
    if request.symbol != expected_symbol or request.date != expected_date:
        raise M8AcquisitionError("archive provider returned a different symbol/date")
    if acquired.limits != limits:
        raise M8AcquisitionError("archive provider returned different byte ceilings")
    archive = acquired.archive_artifact
    checksum = acquired.checksum_artifact
    entry = M8RawArchiveEntry(
        root=root,
        symbol=request.symbol,
        date=request.date,
        role=role,
        tick_size=request.tick_size,
        lot_size=request.lot_size,
        archive_path=archive.path.resolve(),
        archive_sha256=archive.sha256,
        archive_bytes=archive.bytes,
        archive_source_uri=archive.source_uri,
        archive_source_manifest_path=archive.manifest_path.resolve(),
        archive_source_manifest_sha256=archive.manifest_sha256,
        archive_source_manifest_bytes=archive.manifest_path.stat().st_size,
        checksum_path=checksum.path.resolve(),
        checksum_sha256=checksum.sha256,
        checksum_bytes=checksum.bytes,
        checksum_source_uri=checksum.source_uri,
        checksum_source_manifest_path=checksum.manifest_path.resolve(),
        checksum_source_manifest_sha256=checksum.manifest_sha256,
        checksum_source_manifest_bytes=checksum.manifest_path.stat().st_size,
        upstream_sha256=acquired.upstream_sha256,
        member_name=request.member_name,
        declared_uncompressed_bytes=acquired.declared_uncompressed_bytes,
        max_compressed_bytes=limits.max_compressed_bytes,
        max_uncompressed_bytes=limits.max_uncompressed_bytes,
        max_checksum_bytes=limits.max_checksum_bytes,
        transfer_chunk_bytes=limits.transfer_chunk_bytes,
        max_csv_line_bytes=limits.max_csv_line_bytes,
    )
    _verify_archive_entry_files(entry)
    return entry


_METADATA_KEYS = frozenset(
    {
        "base_asset",
        "lot_size",
        "max_price",
        "max_quantity",
        "min_price",
        "min_quantity",
        "observed_ts_ns",
        "quote_asset",
        "raw_bytes",
        "raw_path",
        "raw_sha256",
        "source_manifest_bytes",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_uri",
        "status",
        "symbol",
        "tick_size",
        "venue",
    }
)
_ARCHIVE_KEYS = frozenset(
    {
        "archive_bytes",
        "archive_path",
        "archive_sha256",
        "archive_source_manifest_bytes",
        "archive_source_manifest_path",
        "archive_source_manifest_sha256",
        "archive_source_uri",
        "checksum_bytes",
        "checksum_path",
        "checksum_sha256",
        "checksum_source_manifest_bytes",
        "checksum_source_manifest_path",
        "checksum_source_manifest_sha256",
        "checksum_source_uri",
        "csv_member_opened",
        "date",
        "declared_uncompressed_bytes",
        "economic_fields_inspected",
        "lot_size",
        "max_checksum_bytes",
        "max_compressed_bytes",
        "max_csv_line_bytes",
        "max_uncompressed_bytes",
        "member_name",
        "role",
        "symbol",
        "tick_size",
        "transfer_chunk_bytes",
        "upstream_sha256",
    }
)
_RETAINED_KEYS = frozenset({"bytes", "kind", "paired_body_path", "path", "sha256", "source_uri"})
_TOP_LEVEL_KEYS = frozenset(
    {
        "archives",
        "artifact_kind",
        "config",
        "copied_from_manifest_sha256",
        "evidence_set_sha256",
        "outcome_boundary",
        "retained_artifacts",
        "schema_version",
        "symbol_metadata",
        "totals",
    }
)
_FAILURE_TOP_LEVEL_KEYS = frozenset(
    {
        "artifact_kind",
        "completed",
        "config",
        "diagnostic",
        "failed",
        "outcome_boundary",
        "reason_code",
        "remaining",
        "retained_artifacts",
        "retained_inventory_sha256",
        "schema_version",
        "status",
        "terminal_marker",
        "totals",
    }
)
_FAILURE_STEP_KEYS = frozenset({"date", "kind", "role", "symbol"})
_FAILURE_TOTAL_KEYS = frozenset(
    {
        "completed_count",
        "declared_step_count",
        "remaining_count",
        "retained_artifact_count",
        "total_raw_evidence_bytes",
    }
)
_FAILURE_REASON_CODES: frozenset[str] = frozenset(
    {
        "DECLARED_OBJECT_UNAVAILABLE",
        "METADATA_CONTRACT",
        "CHECKSUM_CONTRACT",
        "ZIP_CONTRACT",
        "RESPONSE_SIZE_LIMIT",
        "TOTAL_EVIDENCE_BUDGET",
    }
)


def _metadata_payload(root: Path, metadata: M8RawSymbolMetadata) -> dict[str, object]:
    return {
        "venue": metadata.venue,
        "symbol": metadata.symbol,
        "status": metadata.status,
        "base_asset": metadata.base_asset,
        "quote_asset": metadata.quote_asset,
        "tick_size": _format_decimal(metadata.tick_size),
        "lot_size": _format_decimal(metadata.lot_size),
        "min_price": _format_decimal(metadata.min_price),
        "max_price": _format_decimal(metadata.max_price),
        "min_quantity": _format_decimal(metadata.min_quantity),
        "max_quantity": _format_decimal(metadata.max_quantity),
        "observed_ts_ns": metadata.observed_ts_ns,
        "raw_path": _relative(root, metadata.raw_path, f"{metadata.symbol} metadata body"),
        "raw_sha256": metadata.raw_sha256,
        "raw_bytes": metadata.raw_bytes,
        "source_uri": metadata.source_uri,
        "source_manifest_path": _relative(
            root,
            metadata.source_manifest_path,
            f"{metadata.symbol} metadata source sidecar",
        ),
        "source_manifest_sha256": metadata.source_manifest_sha256,
        "source_manifest_bytes": metadata.source_manifest_bytes,
    }


def _archive_payload(root: Path, entry: M8RawArchiveEntry) -> dict[str, object]:
    return {
        "symbol": entry.symbol,
        "date": entry.date.isoformat(),
        "role": entry.role,
        "tick_size": _format_decimal(entry.tick_size),
        "lot_size": _format_decimal(entry.lot_size),
        "archive_path": _relative(root, entry.archive_path, f"{entry.symbol}/{entry.date} ZIP"),
        "archive_sha256": entry.archive_sha256,
        "archive_bytes": entry.archive_bytes,
        "archive_source_uri": entry.archive_source_uri,
        "archive_source_manifest_path": _relative(
            root,
            entry.archive_source_manifest_path,
            f"{entry.symbol}/{entry.date} ZIP source sidecar",
        ),
        "archive_source_manifest_sha256": entry.archive_source_manifest_sha256,
        "archive_source_manifest_bytes": entry.archive_source_manifest_bytes,
        "checksum_path": _relative(
            root,
            entry.checksum_path,
            f"{entry.symbol}/{entry.date} CHECKSUM",
        ),
        "checksum_sha256": entry.checksum_sha256,
        "checksum_bytes": entry.checksum_bytes,
        "checksum_source_uri": entry.checksum_source_uri,
        "checksum_source_manifest_path": _relative(
            root,
            entry.checksum_source_manifest_path,
            f"{entry.symbol}/{entry.date} CHECKSUM source sidecar",
        ),
        "checksum_source_manifest_sha256": entry.checksum_source_manifest_sha256,
        "checksum_source_manifest_bytes": entry.checksum_source_manifest_bytes,
        "upstream_sha256": entry.upstream_sha256,
        "member_name": entry.member_name,
        "declared_uncompressed_bytes": entry.declared_uncompressed_bytes,
        "max_compressed_bytes": entry.max_compressed_bytes,
        "max_uncompressed_bytes": entry.max_uncompressed_bytes,
        "max_checksum_bytes": entry.max_checksum_bytes,
        "transfer_chunk_bytes": entry.transfer_chunk_bytes,
        "max_csv_line_bytes": entry.max_csv_line_bytes,
        "csv_member_opened": entry.csv_member_opened,
        "economic_fields_inspected": entry.economic_fields_inspected,
    }


def _retained_payload(artifact: M8RetainedArtifact) -> dict[str, object]:
    return {
        "path": artifact.path,
        "sha256": artifact.sha256,
        "bytes": artifact.bytes,
        "kind": artifact.kind,
        "source_uri": artifact.source_uri,
        "paired_body_path": artifact.paired_body_path,
    }


def _config_payload(config: M8StudyConfig, protocol_sha256: str) -> dict[str, object]:
    return {
        "study_name": config.study.name,
        "protocol_version": config.study.protocol_version,
        "protocol_document_sha256": protocol_sha256,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "evidence_tier": config.study.evidence_tier,
        "source": config.study.source,
        "symbols": list(config.study.symbols),
        "periods": [
            {"date": period.date.isoformat(), "role": period.role} for period in config.periods
        ],
        "byte_limits": {
            "max_archive_compressed_bytes": config.study.max_archive_compressed_bytes,
            "max_archive_uncompressed_bytes": config.study.max_archive_uncompressed_bytes,
            "max_total_download_bytes": config.study.max_total_download_bytes,
            "max_checksum_bytes": 4_096,
            "transfer_chunk_bytes": 64 * 1_024,
            "max_csv_line_bytes": 16 * 1_024,
            "max_metadata_response_bytes": 8 * 1024 * 1024,
        },
    }


def _evidence_payload(
    *,
    config: M8StudyConfig,
    protocol_sha256: str,
    root: Path,
    metadata: Sequence[M8RawSymbolMetadata],
    archives: Sequence[M8RawArchiveEntry],
    retained: Sequence[M8RetainedArtifact],
) -> dict[str, object]:
    total_raw = sum(item.bytes for item in retained)
    total_zip = sum(item.archive_bytes for item in archives)
    return {
        "config": _config_payload(config, protocol_sha256),
        "outcome_boundary": {
            "acquisition_mode": "raw_only",
            "csv_member_opened": False,
            "economic_fields_inspected": False,
            "permitted_zip_inspection": "end_of_central_directory_and_directory_metadata_only",
        },
        "symbol_metadata": [_metadata_payload(root, item) for item in metadata],
        "archives": [_archive_payload(root, item) for item in archives],
        "retained_artifacts": [_retained_payload(item) for item in retained],
        "totals": {
            "metadata_count": len(metadata),
            "archive_count": len(archives),
            "retained_artifact_count": len(retained),
            "total_raw_evidence_bytes": total_raw,
            "total_accepted_zip_bytes": total_zip,
        },
    }


def _manifest_payload(
    *,
    config: M8StudyConfig,
    protocol_sha256: str,
    root: Path,
    metadata: Sequence[M8RawSymbolMetadata],
    archives: Sequence[M8RawArchiveEntry],
    retained: Sequence[M8RetainedArtifact],
    copied_from_manifest_sha256: str | None,
) -> dict[str, object]:
    evidence = _evidence_payload(
        config=config,
        protocol_sha256=protocol_sha256,
        root=root,
        metadata=metadata,
        archives=archives,
        retained=retained,
    )
    evidence_identity = hashlib.sha256(_canonical_json_bytes(evidence)).hexdigest()
    return {
        "schema_version": M8_ACQUISITION_SCHEMA_VERSION,
        "artifact_kind": "m8_raw_acquisition_manifest",
        "copied_from_manifest_sha256": copied_from_manifest_sha256,
        "evidence_set_sha256": evidence_identity,
        **evidence,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Durably flush every regular file and directory without following links."""

    if root.is_symlink() or not root.is_dir():
        raise M8AcquisitionError(f"durability root is missing or symbolic: {root}")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        for directory_name, directory_names, file_names in os.walk(
            root.resolve(),
            topdown=False,
            followlinks=False,
        ):
            directory = Path(directory_name)
            for child_name in directory_names:
                child = directory / child_name
                if child.is_symlink() or not child.is_dir():
                    raise M8AcquisitionError(
                        f"durability tree contains a non-directory or symlink: {child}"
                    )
            for file_name in file_names:
                path = directory / file_name
                if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                    raise M8AcquisitionError(
                        f"durability tree contains a non-regular file or symlink: {path}"
                    )
                descriptor = os.open(path, os.O_RDONLY | nofollow)
                try:
                    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                        raise M8AcquisitionError(
                            f"durability tree entry changed while opened: {path}"
                        )
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            _fsync_directory(directory)
    except M8AcquisitionError:
        raise
    except OSError as exc:
        raise M8AcquisitionError(f"cannot durably flush acquisition tree: {root}") from exc


def _write_manifest_payload(root: Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    encoded = _canonical_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    directory = root / _MANIFEST_DIRECTORY_NAME
    directory.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root)
    destination = directory / f"m8-acquisition.manifest-{digest[:20]}.json"
    if destination.exists():
        observed = _read_bounded_regular_bytes(
            destination,
            "existing M8 acquisition manifest",
            byte_limit=_MAX_MANIFEST_BYTES,
        )
        if destination.is_symlink() or observed != encoded:
            raise M8AcquisitionError(f"immutable M8 acquisition manifest collision: {destination}")
        return destination.resolve(), digest
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory,
        prefix=".m8-acquisition.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, destination)
        _fsync_directory(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve(), digest


def _parse_metadata(root: Path, value: object, expected_symbol: str) -> M8RawSymbolMetadata:
    raw = _object(value, f"symbol_metadata[{expected_symbol}]")
    _require_exact_keys(raw, _METADATA_KEYS, f"symbol_metadata[{expected_symbol}]")
    symbol = _text(raw["symbol"], "metadata symbol")
    if symbol != expected_symbol:
        raise M8AcquisitionError("M8 metadata is outside the frozen symbol order")
    metadata = M8RawSymbolMetadata(
        venue=_text(raw["venue"], f"{symbol}.venue"),
        symbol=symbol,
        status=_text(raw["status"], f"{symbol}.status"),
        base_asset=_text(raw["base_asset"], f"{symbol}.base_asset"),
        quote_asset=_text(raw["quote_asset"], f"{symbol}.quote_asset"),
        tick_size=_decimal(raw["tick_size"], f"{symbol}.tick_size", positive=True),
        lot_size=_decimal(raw["lot_size"], f"{symbol}.lot_size", positive=True),
        min_price=_decimal(raw["min_price"], f"{symbol}.min_price"),
        max_price=_decimal(raw["max_price"], f"{symbol}.max_price"),
        min_quantity=_decimal(raw["min_quantity"], f"{symbol}.min_quantity"),
        max_quantity=_decimal(raw["max_quantity"], f"{symbol}.max_quantity"),
        observed_ts_ns=_integer(raw["observed_ts_ns"], f"{symbol}.observed_ts_ns", minimum=1),
        raw_path=_declared_file(root, raw["raw_path"], f"{symbol} metadata body"),
        raw_sha256=_digest(raw["raw_sha256"], f"{symbol} metadata SHA"),
        raw_bytes=_integer(raw["raw_bytes"], f"{symbol} metadata bytes", minimum=1),
        source_uri=_text(raw["source_uri"], f"{symbol} metadata source URI"),
        source_manifest_path=_declared_file(
            root,
            raw["source_manifest_path"],
            f"{symbol} metadata source sidecar",
        ),
        source_manifest_sha256=_digest(
            raw["source_manifest_sha256"],
            f"{symbol} metadata sidecar SHA",
        ),
        source_manifest_bytes=_integer(
            raw["source_manifest_bytes"],
            f"{symbol} metadata sidecar bytes",
            minimum=1,
        ),
    )
    _verify_metadata_files(root, metadata)
    return metadata


def _parse_archive(
    root: Path,
    value: object,
    *,
    expected_symbol: str,
    expected_date: Date,
    expected_role: M8PeriodRole,
    config: M8StudyConfig,
) -> M8RawArchiveEntry:
    label = f"{expected_symbol}/{expected_date.isoformat()}"
    raw = _object(value, f"archives[{label}]")
    _require_exact_keys(raw, _ARCHIVE_KEYS, f"archives[{label}]")
    symbol = _text(raw["symbol"], f"{label}.symbol")
    try:
        day = Date.fromisoformat(_text(raw["date"], f"{label}.date"))
    except ValueError as exc:
        raise M8AcquisitionError(f"{label}.date is not canonical ISO") from exc
    role = _text(raw["role"], f"{label}.role")
    if (symbol, day, role) != (expected_symbol, expected_date, expected_role):
        raise M8AcquisitionError(f"{label} raw archive is outside the frozen order")
    if role not in {"train", "validation", "primary_test", "replication_test"}:
        raise M8AcquisitionError(f"{label} raw archive role is unsupported")
    entry = M8RawArchiveEntry(
        root=root,
        symbol=symbol,
        date=day,
        role=cast(M8PeriodRole, role),
        tick_size=_decimal(raw["tick_size"], f"{label}.tick_size", positive=True),
        lot_size=_decimal(raw["lot_size"], f"{label}.lot_size", positive=True),
        archive_path=_declared_file(root, raw["archive_path"], f"{label} ZIP"),
        archive_sha256=_digest(raw["archive_sha256"], f"{label} ZIP SHA"),
        archive_bytes=_integer(raw["archive_bytes"], f"{label} ZIP bytes", minimum=1),
        archive_source_uri=_text(raw["archive_source_uri"], f"{label} ZIP URI"),
        archive_source_manifest_path=_declared_file(
            root,
            raw["archive_source_manifest_path"],
            f"{label} ZIP source sidecar",
        ),
        archive_source_manifest_sha256=_digest(
            raw["archive_source_manifest_sha256"],
            f"{label} ZIP source sidecar SHA",
        ),
        archive_source_manifest_bytes=_integer(
            raw["archive_source_manifest_bytes"],
            f"{label} ZIP source sidecar bytes",
            minimum=1,
        ),
        checksum_path=_declared_file(root, raw["checksum_path"], f"{label} CHECKSUM"),
        checksum_sha256=_digest(raw["checksum_sha256"], f"{label} CHECKSUM SHA"),
        checksum_bytes=_integer(raw["checksum_bytes"], f"{label} CHECKSUM bytes", minimum=1),
        checksum_source_uri=_text(raw["checksum_source_uri"], f"{label} CHECKSUM URI"),
        checksum_source_manifest_path=_declared_file(
            root,
            raw["checksum_source_manifest_path"],
            f"{label} CHECKSUM source sidecar",
        ),
        checksum_source_manifest_sha256=_digest(
            raw["checksum_source_manifest_sha256"],
            f"{label} CHECKSUM source sidecar SHA",
        ),
        checksum_source_manifest_bytes=_integer(
            raw["checksum_source_manifest_bytes"],
            f"{label} CHECKSUM source sidecar bytes",
            minimum=1,
        ),
        upstream_sha256=_digest(raw["upstream_sha256"], f"{label} upstream SHA"),
        member_name=_text(raw["member_name"], f"{label} member name"),
        declared_uncompressed_bytes=_integer(
            raw["declared_uncompressed_bytes"],
            f"{label} expanded bytes",
            minimum=1,
        ),
        max_compressed_bytes=_integer(
            raw["max_compressed_bytes"],
            f"{label} max compressed bytes",
            minimum=1,
        ),
        max_uncompressed_bytes=_integer(
            raw["max_uncompressed_bytes"],
            f"{label} max expanded bytes",
            minimum=1,
        ),
        max_checksum_bytes=_integer(
            raw["max_checksum_bytes"],
            f"{label} max checksum bytes",
            minimum=1,
        ),
        transfer_chunk_bytes=_integer(
            raw["transfer_chunk_bytes"],
            f"{label} transfer chunk bytes",
            minimum=1,
        ),
        max_csv_line_bytes=_integer(
            raw["max_csv_line_bytes"],
            f"{label} max CSV line bytes",
            minimum=1,
        ),
        csv_member_opened=_boolean(raw["csv_member_opened"], f"{label}.csv_member_opened"),
        economic_fields_inspected=_boolean(
            raw["economic_fields_inspected"],
            f"{label}.economic_fields_inspected",
        ),
    )
    expected_limits = (
        config.study.max_archive_compressed_bytes,
        config.study.max_archive_uncompressed_bytes,
        4_096,
        64 * 1_024,
        16 * 1_024,
    )
    observed_limits = (
        entry.max_compressed_bytes,
        entry.max_uncompressed_bytes,
        entry.max_checksum_bytes,
        entry.transfer_chunk_bytes,
        entry.max_csv_line_bytes,
    )
    if observed_limits != expected_limits:
        raise M8AcquisitionError(f"{label} archive byte ceilings differ from frozen config")
    _verify_archive_entry_files(entry)
    return entry


def _parse_retained(value: object, index: int) -> M8RetainedArtifact:
    label = f"retained_artifacts[{index}]"
    raw = _object(value, label)
    _require_exact_keys(raw, _RETAINED_KEYS, label)
    kind = _text(raw["kind"], f"{label}.kind")
    allowed = {
        "metadata_body",
        "archive_zip",
        "archive_checksum",
        "rejected_prefix",
        "source_manifest",
    }
    if kind not in allowed:
        raise M8AcquisitionError(f"{label}.kind is unsupported")
    paired_raw = raw["paired_body_path"]
    paired = None if paired_raw is None else _text(paired_raw, f"{label}.paired_body_path")
    return M8RetainedArtifact(
        path=_text(raw["path"], f"{label}.path"),
        sha256=_digest(raw["sha256"], f"{label}.sha256"),
        bytes=_integer(raw["bytes"], f"{label}.bytes", minimum=0),
        kind=cast(RetainedArtifactKind, kind),
        source_uri=_text(raw["source_uri"], f"{label}.source_uri"),
        paired_body_path=paired,
    )


def _walk_raw_files(root: Path) -> tuple[Path, ...]:
    raw_root = root / _RAW_ROOT_NAME
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise M8AcquisitionError("M8 acquisition raw root is missing or symbolic")
    pending = [raw_root]
    files: list[Path] = []
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for item in entries:
                    path = Path(item.path)
                    if item.is_symlink():
                        raise M8AcquisitionError(
                            f"symbolic links are forbidden in raw evidence: {path}"
                        )
                    if item.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif item.is_file(follow_symlinks=False):
                        if (
                            item.name.endswith(".tmp")
                            or item.name.startswith(".download-")
                            or item.name.startswith(".raw-")
                        ):
                            raise M8AcquisitionError(
                                f"unpaired temporary file remains in raw evidence: {path}"
                            )
                        files.append(path.resolve())
                    else:
                        raise M8AcquisitionError(
                            f"non-regular raw evidence entry is forbidden: {path}"
                        )
    except M8AcquisitionError:
        raise
    except OSError as exc:
        raise M8AcquisitionError("cannot enumerate retained M8 raw evidence") from exc
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _is_rejected_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return any(component.endswith("_rejected") for component in relative.parts)


def _historical_metadata_symbol(root: Path, path: Path, source_uri: str) -> str | None:
    relative = path.relative_to(root).parts
    if len(relative) < 5 or relative[:3] != (
        _RAW_ROOT_NAME,
        "binance_spot",
        "exchange_info",
    ):
        return None
    symbol = relative[3]
    try:
        _verify_exchange_info_uri(source_uri, symbol)
    except M8AcquisitionError:
        return None
    return symbol


def _common_sidecar_pair(root: Path, sidecar_path: Path) -> tuple[Path, str]:
    label = f"retained source sidecar {sidecar_path.relative_to(root).as_posix()}"
    sidecar = _read_json(sidecar_path, label)
    _require_exact_keys(sidecar, _RAW_SIDECAR_KEYS, label)
    if sidecar["manifest_version"] != "1.0.0" or sidecar["artifact_kind"] != "raw_source":
        raise M8AcquisitionError(f"{label} identity is unsupported")
    body_name = _text(sidecar["path"], f"{label}.path")
    if Path(body_name).name != body_name:
        raise M8AcquisitionError(f"{label} names a non-basename raw body")
    body_path = (sidecar_path.parent / body_name).resolve()
    if not body_path.is_relative_to((root / _RAW_ROOT_NAME).resolve()) or not body_path.is_file():
        raise M8AcquisitionError(f"{label} has no contained paired raw body")
    checksum = _object(sidecar["checksum"], f"{label}.checksum")
    _require_exact_keys(checksum, frozenset({"algorithm", "value"}), f"{label}.checksum")
    body_sha = _digest(checksum["value"], f"{label}.checksum.value")
    body_bytes = _integer(sidecar["bytes"], f"{label}.bytes", minimum=0)
    if checksum["algorithm"] != "sha256":
        raise M8AcquisitionError(f"{label} uses a non-SHA-256 checksum")
    _verify_file(body_path, body_sha, body_bytes, f"{label} paired raw body")
    requested = _object(sidecar["requested_range_ns"], f"{label}.requested_range_ns")
    _require_exact_keys(
        requested,
        frozenset({"start", "end_exclusive"}),
        f"{label}.requested_range_ns",
    )
    start = requested["start"]
    end = requested["end_exclusive"]
    if (start is None) != (end is None):
        raise M8AcquisitionError(f"{label} requested range is half-null")
    if start is not None:
        start_value = _integer(start, f"{label}.requested_range_ns.start")
        end_value = _integer(end, f"{label}.requested_range_ns.end_exclusive")
        if end_value <= start_value:
            raise M8AcquisitionError(f"{label} requested range is empty")
    upstream = sidecar["upstream_checksum_sha256"]
    if upstream is not None:
        _digest(upstream, f"{label}.upstream_checksum_sha256")
    headers = _object(sidecar["response_headers"], f"{label}.response_headers")
    if not all(type(key) is str and type(value) is str for key, value in headers.items()):
        raise M8AcquisitionError(f"{label} response headers must contain only strings")
    _utc_ns(sidecar["downloaded_at_utc"], f"{label}.downloaded_at_utc")
    _text(sidecar["source"], f"{label}.source")
    source_uri = _text(sidecar["source_uri"], f"{label}.source_uri")
    _source_manifest_path_is_canonical(body_path, sidecar_path, label)
    _verify_source_manifest_identity(body_path, sidecar_path, sidecar, label)
    return body_path, source_uri


def _scan_retained_inventory(
    root: Path,
    metadata: Sequence[M8RawSymbolMetadata],
    archives: Sequence[M8RawArchiveEntry],
) -> tuple[M8RetainedArtifact, ...]:
    """Enumerate every physical raw file and reject unpaired or extra accepted data."""

    files = _walk_raw_files(root)
    accepted_bodies: dict[Path, tuple[RetainedArtifactKind, str]] = {}
    accepted_sidecars: dict[Path, tuple[Path, str]] = {}
    for metadata_item in metadata:
        accepted_bodies[metadata_item.raw_path] = ("metadata_body", metadata_item.source_uri)
        accepted_sidecars[metadata_item.source_manifest_path] = (
            metadata_item.raw_path,
            metadata_item.source_uri,
        )
    for archive_item in archives:
        accepted_bodies[archive_item.archive_path] = (
            "archive_zip",
            archive_item.archive_source_uri,
        )
        accepted_bodies[archive_item.checksum_path] = (
            "archive_checksum",
            archive_item.checksum_source_uri,
        )
        accepted_sidecars[archive_item.archive_source_manifest_path] = (
            archive_item.archive_path,
            archive_item.archive_source_uri,
        )
        accepted_sidecars[archive_item.checksum_source_manifest_path] = (
            archive_item.checksum_path,
            archive_item.checksum_source_uri,
        )
    if len(accepted_bodies) != len(metadata) + 2 * len(archives):
        raise M8AcquisitionError("accepted raw bodies are not physically distinct")
    if len(accepted_sidecars) != len(metadata) + 2 * len(archives):
        raise M8AcquisitionError("accepted source sidecars are not physically distinct")

    sidecar_pairs: dict[Path, tuple[Path, str]] = {}
    references: dict[Path, list[tuple[Path, str]]] = {}
    for path in files:
        if _SOURCE_MANIFEST_NAME.fullmatch(path.name) is None:
            continue
        pair = _common_sidecar_pair(root, path)
        sidecar_pairs[path] = pair
        references.setdefault(pair[0], []).append((path, pair[1]))

    artifacts: list[M8RetainedArtifact] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path in sidecar_pairs:
            body_path, source_uri = sidecar_pairs[path]
            expected = accepted_sidecars.get(path)
            if expected is None:
                accepted_body = accepted_bodies.get(body_path)
                historical_symbol = _historical_metadata_symbol(root, body_path, source_uri)
                if (
                    accepted_body is not None and accepted_body[1] == source_uri
                ) or historical_symbol in {item.symbol for item in metadata}:
                    pass
                elif not _is_rejected_path(root, path) or not _is_rejected_path(root, body_path):
                    raise M8AcquisitionError(f"unexpected accepted source sidecar: {relative}")
            elif expected != (body_path, source_uri):
                raise M8AcquisitionError(f"accepted source sidecar changed pairing: {relative}")
            artifacts.append(
                M8RetainedArtifact(
                    path=relative,
                    sha256=sha256_file(path),
                    bytes=path.stat().st_size,
                    kind="source_manifest",
                    source_uri=source_uri,
                    paired_body_path=body_path.relative_to(root).as_posix(),
                )
            )
            continue

        paired = references.get(path, [])
        if not paired:
            raise M8AcquisitionError(f"unpaired retained raw body: {relative}")
        source_uris = {source_uri for _, source_uri in paired}
        if len(source_uris) != 1:
            raise M8AcquisitionError(f"retained raw body has ambiguous source URIs: {relative}")
        source_uri = next(iter(source_uris))
        accepted = accepted_bodies.get(path)
        kind: RetainedArtifactKind
        if accepted is None:
            if not _is_rejected_path(root, path):
                historical_symbol = _historical_metadata_symbol(root, path, source_uri)
                if historical_symbol not in {item.symbol for item in metadata}:
                    raise M8AcquisitionError(f"unexpected extra accepted raw body: {relative}")
                kind = "metadata_body"
            else:
                kind = "rejected_prefix"
        else:
            kind, expected_uri = accepted
            if source_uri != expected_uri:
                raise M8AcquisitionError(f"accepted raw body changed source URI: {relative}")
        artifacts.append(
            M8RetainedArtifact(
                path=relative,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                kind=kind,
                source_uri=source_uri,
                paired_body_path=None,
            )
        )

    if set(accepted_bodies) - set(files) or set(accepted_sidecars) - set(files):
        raise M8AcquisitionError("accepted M8 raw evidence is missing from retained inventory")
    return tuple(sorted(artifacts, key=lambda item: item.path))


def _failure_body_kind(root: Path, path: Path, source_uri: str) -> RetainedArtifactKind:
    """Classify a partial body without parsing metadata or opening a ZIP member."""

    if _is_rejected_path(root, path):
        return "rejected_prefix"
    parts = path.relative_to(root).parts
    if len(parts) == 5 and parts[:3] == (
        _RAW_ROOT_NAME,
        "binance_spot",
        "exchange_info",
    ):
        symbol = parts[3]
        if _SAFE_SYMBOL.fullmatch(symbol) is None or parts[4] != f"{sha256_file(path)}.json":
            raise M8AcquisitionError("partial metadata body is not content-addressed canonically")
        _verify_exchange_info_uri(source_uri, symbol)
        return "metadata_body"
    if len(parts) == 6 and parts[:2] == (_RAW_ROOT_NAME, "binance_spot"):
        dataset, symbol, date_text, filename = parts[2:]
        if _SAFE_SYMBOL.fullmatch(symbol) is None:
            raise M8AcquisitionError("partial archive body has an unsafe symbol")
        try:
            day = Date.fromisoformat(date_text)
        except ValueError as exc:
            raise M8AcquisitionError("partial archive body has a noncanonical date") from exc
        archive_name = f"{symbol}-aggTrades-{date_text}.zip"
        if dataset == "daily_agg_trades_archive" and filename == archive_name:
            _verify_official_uri(source_uri, _expected_archive_uri(symbol, day), "partial ZIP URI")
            return "archive_zip"
        if (
            dataset == "daily_agg_trades_archive_checksums"
            and filename == f"{archive_name}.CHECKSUM"
        ):
            _verify_official_uri(
                source_uri,
                _expected_archive_uri(symbol, day, checksum=True),
                "partial CHECKSUM URI",
            )
            return "archive_checksum"
    raise M8AcquisitionError(
        f"unexpected accepted partial raw body: {path.relative_to(root).as_posix()}"
    )


def _scan_failure_retained_inventory(root: Path) -> tuple[M8RetainedArtifact, ...]:
    """Enumerate a raw-only partial attempt without interpreting economic bytes."""

    files = _walk_raw_files(root)
    sidecar_pairs: dict[Path, tuple[Path, str]] = {}
    references: dict[Path, list[tuple[Path, str]]] = {}
    for path in files:
        if _SOURCE_MANIFEST_NAME.fullmatch(path.name) is None:
            continue
        pair = _common_sidecar_pair(root, path)
        sidecar_pairs[path] = pair
        references.setdefault(pair[0], []).append((path, pair[1]))

    artifacts: list[M8RetainedArtifact] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path in sidecar_pairs:
            body_path, source_uri = sidecar_pairs[path]
            _failure_body_kind(root, body_path, source_uri)
            artifacts.append(
                M8RetainedArtifact(
                    path=relative,
                    sha256=sha256_file(path),
                    bytes=path.stat().st_size,
                    kind="source_manifest",
                    source_uri=source_uri,
                    paired_body_path=body_path.relative_to(root).as_posix(),
                )
            )
            continue
        paired = references.get(path, [])
        if not paired:
            raise M8AcquisitionError(f"unpaired retained partial body: {relative}")
        source_uris = {source_uri for _, source_uri in paired}
        if len(source_uris) != 1:
            raise M8AcquisitionError(f"partial raw body has ambiguous source URIs: {relative}")
        source_uri = next(iter(source_uris))
        artifacts.append(
            M8RetainedArtifact(
                path=relative,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                kind=_failure_body_kind(root, path, source_uri),
                source_uri=source_uri,
                paired_body_path=None,
            )
        )
    return tuple(sorted(artifacts, key=lambda item: item.path))


def _retained_inventory_sha256(retained: Sequence[M8RetainedArtifact]) -> str:
    payload = {"retained_artifacts": [_retained_payload(item) for item in retained]}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _verify_manifest_config(
    raw: object,
    config: M8StudyConfig,
    protocol_sha256: str,
) -> None:
    observed = _object(raw, "M8 acquisition config binding")
    expected = _config_payload(config, protocol_sha256)
    if observed != expected:
        raise M8AcquisitionError("M8 acquisition manifest is bound to another protocol/config")


def _manifest_evidence_object(payload: Mapping[str, Any]) -> dict[str, object]:
    return {
        "config": payload["config"],
        "outcome_boundary": payload["outcome_boundary"],
        "symbol_metadata": payload["symbol_metadata"],
        "archives": payload["archives"],
        "retained_artifacts": payload["retained_artifacts"],
        "totals": payload["totals"],
    }


def _acquisition_steps(config: M8StudyConfig) -> tuple[M8AcquisitionStep, ...]:
    metadata = tuple(
        M8AcquisitionStep(kind="metadata", symbol=symbol, date=None, role="metadata")
        for symbol in config.study.symbols
    )
    archives = tuple(
        M8AcquisitionStep(
            kind="archive",
            symbol=symbol,
            date=period.date,
            role=period.role,
        )
        for period in config.periods
        for symbol in config.study.symbols
    )
    return metadata + archives


def _step_payload(step: M8AcquisitionStep) -> dict[str, object]:
    return {
        "kind": step.kind,
        "symbol": step.symbol,
        "date": None if step.date is None else step.date.isoformat(),
        "role": step.role,
    }


def _parse_failure_step(value: object, label: str) -> M8AcquisitionStep:
    raw = _object(value, label)
    _require_exact_keys(raw, _FAILURE_STEP_KEYS, label)
    kind = _text(raw["kind"], f"{label}.kind")
    if kind not in {"metadata", "archive"}:
        raise M8AcquisitionError(f"{label}.kind is unsupported")
    symbol = _text(raw["symbol"], f"{label}.symbol")
    if _SAFE_SYMBOL.fullmatch(symbol) is None:
        raise M8AcquisitionError(f"{label}.symbol is unsafe")
    role = _text(raw["role"], f"{label}.role")
    date_raw = raw["date"]
    if kind == "metadata":
        if date_raw is not None or role != "metadata":
            raise M8AcquisitionError(f"{label} metadata coordinates are invalid")
        day = None
    else:
        if type(date_raw) is not str:
            raise M8AcquisitionError(f"{label}.date must be a canonical date")
        try:
            day = Date.fromisoformat(date_raw)
        except ValueError as exc:
            raise M8AcquisitionError(f"{label}.date is invalid") from exc
        if day.isoformat() != date_raw:
            raise M8AcquisitionError(f"{label}.date is noncanonical")
    return M8AcquisitionStep(
        kind=cast(M8AcquisitionStepKind, kind),
        symbol=symbol,
        date=day,
        role=role,
    )


def _failure_payload(
    *,
    config: M8StudyConfig,
    protocol_sha256: str,
    reason_code: M8AcquisitionReasonCode,
    diagnostic: str,
    failed: M8AcquisitionStep,
    completed: Sequence[M8AcquisitionStep],
    remaining: Sequence[M8AcquisitionStep],
    retained: Sequence[M8RetainedArtifact],
) -> dict[str, object]:
    inventory_sha = _retained_inventory_sha256(retained)
    return {
        "schema_version": M8_ACQUISITION_SCHEMA_VERSION,
        "artifact_kind": "m8_raw_acquisition_failure",
        "status": "INSUFFICIENT_DATA",
        "terminal_marker": _FAILURE_TERMINAL_NAME,
        "config": _config_payload(config, protocol_sha256),
        "outcome_boundary": {
            "acquisition_mode": "raw_only",
            "csv_member_opened": False,
            "economic_fields_inspected": False,
            "terminal_before_csv_open": True,
        },
        "reason_code": reason_code,
        "diagnostic": diagnostic,
        "failed": _step_payload(failed),
        "completed": [_step_payload(step) for step in completed],
        "remaining": [_step_payload(step) for step in remaining],
        "retained_artifacts": [_retained_payload(item) for item in retained],
        "retained_inventory_sha256": inventory_sha,
        "totals": {
            "completed_count": len(completed),
            "remaining_count": len(remaining),
            "declared_step_count": len(completed) + 1 + len(remaining),
            "retained_artifact_count": len(retained),
            "total_raw_evidence_bytes": sum(item.bytes for item in retained),
        },
    }


def _failure_checksums_bytes(
    manifest_sha256: str,
    retained: Sequence[M8RetainedArtifact],
) -> bytes:
    marker_sha = hashlib.sha256(_FAILURE_TERMINAL_BYTES).hexdigest()
    lines = [
        f"{manifest_sha256}  {_FAILURE_MANIFEST_NAME}",
        f"{marker_sha}  {_FAILURE_TERMINAL_NAME}",
    ]
    lines.extend(f"{item.sha256}  ../../{item.path}" for item in retained)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_new_durable_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(content)
        sink.flush()
        os.fsync(sink.fileno())
    _fsync_directory(path.parent)


def _publish_failure_authority(
    root: Path,
    *,
    payload: Mapping[str, Any],
    retained: Sequence[M8RetainedArtifact],
    config: M8StudyConfig,
) -> M8AcquisitionFailureManifest:
    encoded = _canonical_json_bytes(payload)
    manifest_sha = hashlib.sha256(encoded).hexdigest()
    attempt_root = root / _ATTEMPT_DIRECTORY_NAME
    attempt_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(root)
    destination = attempt_root / f"m8-acquisition-attempt-{manifest_sha[:20]}"
    if destination.exists():
        return read_m8_acquisition_failure(
            destination / _FAILURE_MANIFEST_NAME,
            expected_sha256=manifest_sha,
            config=config,
        )
    stage = Path(tempfile.mkdtemp(prefix=".m8-acquisition-attempt-", dir=attempt_root))
    try:
        _write_new_durable_file(stage / _FAILURE_MANIFEST_NAME, encoded)
        checksums = _failure_checksums_bytes(manifest_sha, retained)
        _write_new_durable_file(stage / _FAILURE_CHECKSUMS_NAME, checksums)
        _write_new_durable_file(stage / _FAILURE_TERMINAL_NAME, _FAILURE_TERMINAL_BYTES)
        _fsync_tree(stage)
        try:
            os.replace(stage, destination)
        except OSError:
            if not destination.is_dir():
                raise
        _fsync_directory(attempt_root)
        _fsync_directory(root)
        return read_m8_acquisition_failure(
            destination / _FAILURE_MANIFEST_NAME,
            expected_sha256=manifest_sha,
            config=config,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def read_m8_acquisition_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    config: M8StudyConfig,
) -> M8AcquisitionManifest:
    """Verify one explicitly named raw manifest; never discover or open a CSV."""

    expected_digest = _digest(expected_sha256, "expected acquisition manifest SHA-256")
    explicit = Path(path).expanduser().absolute()
    if explicit.is_symlink() or not explicit.is_file():
        raise M8AcquisitionError("explicit M8 acquisition manifest is missing or symbolic")
    name_match = _MANIFEST_NAME.fullmatch(explicit.name)
    if name_match is None or name_match.group(1) != expected_digest[:20]:
        raise M8AcquisitionError("M8 acquisition manifest filename is not content-addressed")
    if explicit.parent.name != _MANIFEST_DIRECTORY_NAME:
        raise M8AcquisitionError("M8 acquisition manifest is outside canonical manifest storage")
    root = explicit.parent.parent
    if root.is_symlink() or not root.is_dir():
        raise M8AcquisitionError("M8 acquisition root is missing or symbolic")
    _validate_root_layout(root)
    manifest_snapshot = _read_bounded_regular_snapshot(
        explicit,
        "M8 acquisition manifest",
        byte_limit=_MAX_MANIFEST_BYTES,
    )
    manifest_bytes = manifest_snapshot.content
    observed_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_digest != expected_digest:
        raise M8AcquisitionError("M8 acquisition manifest bytes disagree with supplied SHA-256")
    payload = _parse_json_bytes(manifest_bytes, "M8 acquisition manifest")
    if manifest_bytes != _canonical_json_bytes(payload):
        raise M8AcquisitionError("M8 acquisition manifest is not canonical JSON")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "M8 acquisition manifest")
    if (
        payload["schema_version"] != M8_ACQUISITION_SCHEMA_VERSION
        or payload["artifact_kind"] != "m8_raw_acquisition_manifest"
    ):
        raise M8AcquisitionError("M8 acquisition manifest identity is unsupported")
    protocol_sha = _verify_config_sources(config)
    _verify_manifest_config(payload["config"], config, protocol_sha)
    boundary = _object(payload["outcome_boundary"], "M8 outcome boundary")
    expected_boundary = {
        "acquisition_mode": "raw_only",
        "csv_member_opened": False,
        "economic_fields_inspected": False,
        "permitted_zip_inspection": "end_of_central_directory_and_directory_metadata_only",
    }
    if boundary != expected_boundary:
        raise M8AcquisitionError("M8 acquisition boundary permits economic-data inspection")
    copied_raw = payload["copied_from_manifest_sha256"]
    copied_from = None if copied_raw is None else _digest(copied_raw, "copied-from manifest SHA")
    evidence_identity = _digest(payload["evidence_set_sha256"], "M8 evidence-set SHA")
    recomputed_identity = hashlib.sha256(
        _canonical_json_bytes(_manifest_evidence_object(payload))
    ).hexdigest()
    if evidence_identity != recomputed_identity:
        raise M8AcquisitionError("M8 acquisition evidence-set identity does not match claims")

    raw_metadata = _array(payload["symbol_metadata"], "M8 symbol metadata")
    if len(raw_metadata) != 2 or len(raw_metadata) != len(config.study.symbols):
        raise M8AcquisitionError("M8 acquisition requires exactly two metadata responses")
    metadata = tuple(
        _parse_metadata(root, value, symbol)
        for value, symbol in zip(raw_metadata, config.study.symbols, strict=True)
    )
    raw_archives = _array(payload["archives"], "M8 raw archives")
    expected_order = tuple(
        (symbol, period.date, period.role)
        for period in config.periods
        for symbol in config.study.symbols
    )
    if len(raw_archives) != 8 or len(raw_archives) != len(expected_order):
        raise M8AcquisitionError("M8 acquisition requires exactly eight raw archives")
    archives = tuple(
        _parse_archive(
            root,
            value,
            expected_symbol=symbol,
            expected_date=day,
            expected_role=role,
            config=config,
        )
        for value, (symbol, day, role) in zip(raw_archives, expected_order, strict=True)
    )
    metadata_by_symbol = {item.symbol: item for item in metadata}
    if any(
        item.tick_size != metadata_by_symbol[item.symbol].tick_size
        or item.lot_size != metadata_by_symbol[item.symbol].lot_size
        for item in archives
    ):
        raise M8AcquisitionError("archive tick/lot scales differ from verified symbol metadata")

    raw_retained = _array(payload["retained_artifacts"], "M8 retained inventory")
    retained = tuple(_parse_retained(value, index) for index, value in enumerate(raw_retained))
    if tuple(item.path for item in retained) != tuple(sorted(item.path for item in retained)):
        raise M8AcquisitionError("M8 retained inventory is not in canonical path order")
    if len({item.path for item in retained}) != len(retained):
        raise M8AcquisitionError("M8 retained inventory repeats a physical path")
    observed_inventory = _scan_retained_inventory(root, metadata, archives)
    if retained != observed_inventory:
        raise M8AcquisitionError("M8 retained inventory differs from physical raw evidence")
    total_raw = sum(item.bytes for item in retained)
    total_zip = sum(item.archive_bytes for item in archives)
    totals = _object(payload["totals"], "M8 acquisition totals")
    expected_totals = {
        "metadata_count": 2,
        "archive_count": 8,
        "retained_artifact_count": len(retained),
        "total_raw_evidence_bytes": total_raw,
        "total_accepted_zip_bytes": total_zip,
    }
    if totals != expected_totals:
        raise M8AcquisitionError("M8 acquisition totals disagree with exact retained inventory")
    if total_raw > config.study.max_total_download_bytes:
        raise M8AcquisitionError("M8 retained raw evidence exceeds frozen total-byte ceiling")
    _assert_snapshot_path(explicit, manifest_snapshot, "M8 acquisition manifest")
    return M8AcquisitionManifest(
        root=root.resolve(),
        path=explicit,
        sha256=expected_digest,
        config_sha256=config.hash,
        config_source_sha256=config.source_sha256,
        protocol_version=config.study.protocol_version,
        protocol_document_sha256=protocol_sha,
        copied_from_manifest_sha256=copied_from,
        evidence_set_sha256=evidence_identity,
        symbol_metadata=metadata,
        archives=archives,
        retained_artifacts=retained,
        total_raw_evidence_bytes=total_raw,
        total_accepted_zip_bytes=total_zip,
        config=config,
    )


def read_m8_acquisition_failure(
    path: str | Path,
    *,
    expected_sha256: str,
    config: M8StudyConfig,
) -> M8AcquisitionFailureManifest:
    """Verify one immutable deterministic acquisition failure without opening CSV."""

    expected_digest = _digest(expected_sha256, "expected acquisition failure SHA-256")
    explicit = Path(path).expanduser().absolute()
    if (
        explicit.name != _FAILURE_MANIFEST_NAME
        or explicit.parent.parent.name != _ATTEMPT_DIRECTORY_NAME
    ):
        raise M8AcquisitionError("M8 acquisition failure is outside canonical attempt storage")
    attempt_match = _FAILURE_ATTEMPT_NAME.fullmatch(explicit.parent.name)
    if attempt_match is None or attempt_match.group(1) != expected_digest[:20]:
        raise M8AcquisitionError("M8 acquisition attempt directory is not content-addressed")
    root = explicit.parent.parent.parent
    if root.is_symlink() or not root.is_dir():
        raise M8AcquisitionError("M8 acquisition failure root is missing or symbolic")
    _validate_root_layout(root)
    manifest_snapshot = _read_bounded_regular_snapshot(
        explicit,
        "M8 acquisition failure manifest",
        byte_limit=_MAX_MANIFEST_BYTES,
    )
    manifest_bytes = manifest_snapshot.content
    observed_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_digest != expected_digest:
        raise M8AcquisitionError("M8 acquisition failure bytes disagree with supplied SHA-256")
    payload = _parse_json_bytes(manifest_bytes, "M8 acquisition failure manifest")
    if manifest_bytes != _canonical_json_bytes(payload):
        raise M8AcquisitionError("M8 acquisition failure manifest is not canonical JSON")
    _require_exact_keys(payload, _FAILURE_TOP_LEVEL_KEYS, "M8 acquisition failure manifest")
    if (
        payload["schema_version"] != M8_ACQUISITION_SCHEMA_VERSION
        or payload["artifact_kind"] != "m8_raw_acquisition_failure"
        or payload["status"] != "INSUFFICIENT_DATA"
        or payload["terminal_marker"] != _FAILURE_TERMINAL_NAME
    ):
        raise M8AcquisitionError("M8 acquisition failure identity is unsupported")
    protocol_sha = _verify_config_sources(config)
    _verify_manifest_config(payload["config"], config, protocol_sha)
    expected_boundary = {
        "acquisition_mode": "raw_only",
        "csv_member_opened": False,
        "economic_fields_inspected": False,
        "terminal_before_csv_open": True,
    }
    if _object(payload["outcome_boundary"], "M8 failure outcome boundary") != expected_boundary:
        raise M8AcquisitionError("M8 acquisition failure crosses the raw-only boundary")
    reason_text = _text(payload["reason_code"], "M8 acquisition failure reason code")
    if reason_text not in _FAILURE_REASON_CODES:
        raise M8AcquisitionError("M8 acquisition failure reason code is unsupported")
    reason_code = cast(M8AcquisitionReasonCode, reason_text)
    diagnostic = _text(payload["diagnostic"], "M8 acquisition failure diagnostic")
    if diagnostic.strip() != diagnostic or "\n" in diagnostic or "\r" in diagnostic:
        raise M8AcquisitionError("M8 acquisition failure diagnostic is noncanonical")

    failed = _parse_failure_step(payload["failed"], "M8 acquisition failed step")
    completed = tuple(
        _parse_failure_step(value, f"M8 acquisition completed[{index}]")
        for index, value in enumerate(_array(payload["completed"], "M8 completed steps"))
    )
    remaining = tuple(
        _parse_failure_step(value, f"M8 acquisition remaining[{index}]")
        for index, value in enumerate(_array(payload["remaining"], "M8 remaining steps"))
    )
    declared_steps = _acquisition_steps(config)
    if (*completed, failed, *remaining) != declared_steps:
        raise M8AcquisitionError("M8 acquisition failure steps do not partition frozen order")

    retained = tuple(
        _parse_retained(value, index)
        for index, value in enumerate(
            _array(payload["retained_artifacts"], "M8 failure retained inventory")
        )
    )
    if tuple(item.path for item in retained) != tuple(sorted(item.path for item in retained)):
        raise M8AcquisitionError("M8 failure retained inventory is not in canonical order")
    if len({item.path for item in retained}) != len(retained):
        raise M8AcquisitionError("M8 failure retained inventory repeats a path")
    observed_inventory = _scan_failure_retained_inventory(root)
    observed_by_path = {item.path: item for item in observed_inventory}
    if any(observed_by_path.get(item.path) != item for item in retained):
        raise M8AcquisitionError("M8 failure inventory differs from retained physical evidence")
    inventory_sha = _digest(
        payload["retained_inventory_sha256"],
        "M8 failure retained-inventory SHA",
    )
    if inventory_sha != _retained_inventory_sha256(retained):
        raise M8AcquisitionError("M8 failure retained-inventory SHA disagrees with claims")
    total_raw = sum(item.bytes for item in retained)
    totals = _object(payload["totals"], "M8 acquisition failure totals")
    _require_exact_keys(totals, _FAILURE_TOTAL_KEYS, "M8 acquisition failure totals")
    expected_totals = {
        "completed_count": len(completed),
        "remaining_count": len(remaining),
        "declared_step_count": len(declared_steps),
        "retained_artifact_count": len(retained),
        "total_raw_evidence_bytes": total_raw,
    }
    if totals != expected_totals:
        raise M8AcquisitionError("M8 acquisition failure totals disagree with inventory")
    if reason_code != "TOTAL_EVIDENCE_BUDGET" and total_raw > config.study.max_total_download_bytes:
        raise M8AcquisitionError("M8 failure retained evidence exceeds frozen byte ceiling")

    terminal_path = explicit.parent / _FAILURE_TERMINAL_NAME
    terminal_snapshot = _read_bounded_regular_snapshot(
        terminal_path,
        "M8 acquisition failure terminal marker",
        byte_limit=len(_FAILURE_TERMINAL_BYTES),
    )
    if terminal_snapshot.content != _FAILURE_TERMINAL_BYTES:
        raise M8AcquisitionError("M8 acquisition failure terminal marker bytes are invalid")
    checksums_path = explicit.parent / _FAILURE_CHECKSUMS_NAME
    checksums_snapshot = _read_bounded_regular_snapshot(
        checksums_path,
        "M8 acquisition failure checksums",
        byte_limit=_MAX_MANIFEST_BYTES,
    )
    checksums_bytes = checksums_snapshot.content
    if checksums_bytes != _failure_checksums_bytes(expected_digest, retained):
        raise M8AcquisitionError("M8 acquisition failure checksum manifest is invalid")
    checksums_sha = hashlib.sha256(checksums_bytes).hexdigest()
    _assert_snapshot_path(explicit, manifest_snapshot, "M8 acquisition failure manifest")
    _assert_snapshot_path(
        checksums_path,
        checksums_snapshot,
        "M8 acquisition failure checksums",
    )
    _assert_snapshot_path(
        terminal_path,
        terminal_snapshot,
        "M8 acquisition failure terminal marker",
    )
    return M8AcquisitionFailureManifest(
        root=root,
        attempt_dir=explicit.parent,
        path=explicit,
        sha256=expected_digest,
        checksums_path=checksums_path,
        checksums_sha256=checksums_sha,
        terminal_path=terminal_path,
        reason_code=reason_code,
        diagnostic=diagnostic,
        failed_step=failed,
        completed_steps=completed,
        remaining_steps=remaining,
        retained_artifacts=retained,
        retained_inventory_sha256=inventory_sha,
        total_raw_evidence_bytes=total_raw,
        config=config,
    )


verify_m8_acquisition_manifest = read_m8_acquisition_manifest
verify_m8_acquisition_failure = read_m8_acquisition_failure


def _validate_root_layout(root: Path) -> None:
    allowed = {_RAW_ROOT_NAME, _MANIFEST_DIRECTORY_NAME, _ATTEMPT_DIRECTORY_NAME}
    try:
        for child in root.iterdir():
            if child.is_symlink():
                raise M8AcquisitionError(f"symbolic link is forbidden in acquisition root: {child}")
            if child.name not in allowed:
                raise M8AcquisitionError(f"unexpected acquisition-root artifact: {child}")
            if not child.is_dir():
                raise M8AcquisitionError(f"acquisition-root component is not a directory: {child}")
        manifest_directory = root / _MANIFEST_DIRECTORY_NAME
        if manifest_directory.exists():
            for child in manifest_directory.iterdir():
                if (
                    child.is_symlink()
                    or not child.is_file()
                    or _MANIFEST_NAME.fullmatch(child.name) is None
                ):
                    raise M8AcquisitionError(
                        f"unpaired or noncanonical acquisition manifest artifact: {child}"
                    )
        attempt_directory = root / _ATTEMPT_DIRECTORY_NAME
        if attempt_directory.exists():
            for attempt in attempt_directory.iterdir():
                if (
                    attempt.is_symlink()
                    or not attempt.is_dir()
                    or _FAILURE_ATTEMPT_NAME.fullmatch(attempt.name) is None
                ):
                    raise M8AcquisitionError(
                        f"unpaired or noncanonical acquisition attempt artifact: {attempt}"
                    )
                observed_names: set[str] = set()
                for child in attempt.iterdir():
                    if child.is_symlink() or not child.is_file():
                        raise M8AcquisitionError(
                            f"acquisition attempt contains a non-regular artifact: {child}"
                        )
                    observed_names.add(child.name)
                expected_names = {
                    _FAILURE_MANIFEST_NAME,
                    _FAILURE_CHECKSUMS_NAME,
                    _FAILURE_TERMINAL_NAME,
                }
                if observed_names != expected_names:
                    raise M8AcquisitionError(
                        f"acquisition attempt file set is noncanonical: {attempt}"
                    )
    except M8AcquisitionError:
        raise
    except OSError as exc:
        raise M8AcquisitionError("cannot validate M8 acquisition-root layout") from exc


def _ordered_supplied_metadata(
    supplied: Mapping[str, SymbolMetadata] | Sequence[SymbolMetadata],
    symbols: tuple[str, ...],
) -> tuple[SymbolMetadata, ...]:
    if isinstance(supplied, Mapping):
        if set(supplied) != set(symbols):
            raise M8AcquisitionError("supplied M8 metadata keys differ from frozen symbols")
        return tuple(supplied[symbol] for symbol in symbols)
    ordered = tuple(supplied)
    if tuple(item.symbol for item in ordered) != symbols:
        raise M8AcquisitionError("supplied M8 metadata is outside frozen symbol order")
    return ordered


def _fetch_provider_metadata(
    provider: _MetadataProvider | Callable[[str, Path], SymbolMetadata],
    symbol: str,
    raw_root: Path,
) -> SymbolMetadata:
    method = getattr(provider, "fetch_exchange_info", None)
    if callable(method):
        result = method(symbol=symbol, raw_root=raw_root)
    elif callable(provider):
        result = provider(symbol, raw_root)
    else:
        raise M8AcquisitionError("metadata_provider has no supported fetch boundary")
    if not isinstance(result, SymbolMetadata):
        raise M8AcquisitionError("metadata_provider returned a non-SymbolMetadata value")
    return result


def _deterministic_failure_reason(exc: BaseException) -> M8AcquisitionReasonCode | None:
    if isinstance(exc, EvidenceBudgetExceeded):
        return "TOTAL_EVIDENCE_BUDGET"
    if isinstance(exc, (BinanceMetadataContractError, _M8MetadataResponseContractError)):
        return "METADATA_CONTRACT"
    if isinstance(exc, BinanceResponseSizeLimitError):
        return "RESPONSE_SIZE_LIMIT"
    if isinstance(exc, BinanceArchiveContractError):
        return cast(M8AcquisitionReasonCode, exc.reason_code)
    if (
        isinstance(exc, (BinanceHTTPError, BinanceArchiveHTTPError))
        and not exc.retry_exhausted
        and exc.status_code in {404, 410}
    ):
        return "DECLARED_OBJECT_UNAVAILABLE"
    return None


def _failure_diagnostic(exc: BaseException) -> str:
    text = " ".join(str(exc).replace("\r", " ").replace("\n", " ").split())
    if not text:
        text = type(exc).__name__
    return f"{type(exc).__name__}: {text}"[:2_000].rstrip()


def _finalize_deterministic_failure(
    *,
    root: Path,
    raw_root: Path,
    config: M8StudyConfig,
    protocol_sha256: str,
    reason_code: M8AcquisitionReasonCode,
    cause: BaseException,
    failed: M8AcquisitionStep,
    completed: Sequence[M8AcquisitionStep],
    remaining: Sequence[M8AcquisitionStep],
    budget: RetainedEvidenceBudget | None,
) -> M8AcquisitionFailureResult:
    if budget is not None and budget.reserved_bytes != 0:
        raise M8AcquisitionError(
            "cannot publish deterministic failure with unfinished evidence reservations"
        ) from cause
    retained = _scan_failure_retained_inventory(root)
    _fsync_tree(raw_root)
    durable_retained = _scan_failure_retained_inventory(root)
    if durable_retained != retained:
        raise M8AcquisitionError(
            "raw inventory changed across failure durability barrier"
        ) from cause
    total_raw = sum(item.bytes for item in retained)
    if reason_code != "TOTAL_EVIDENCE_BUDGET" and total_raw > config.study.max_total_download_bytes:
        raise M8AcquisitionError(
            "deterministic failure evidence exceeds frozen byte ceiling"
        ) from cause
    if budget is not None and budget.used_bytes != total_raw:
        raise M8AcquisitionError(
            "retained-evidence budget disagrees with failure inventory"
        ) from cause
    payload = _failure_payload(
        config=config,
        protocol_sha256=protocol_sha256,
        reason_code=reason_code,
        diagnostic=_failure_diagnostic(cause),
        failed=failed,
        completed=completed,
        remaining=remaining,
        retained=retained,
    )
    authority = _publish_failure_authority(
        root,
        payload=payload,
        retained=retained,
        config=config,
    )
    return M8AcquisitionFailureResult(
        output_root=root.resolve(),
        attempt_dir=authority.attempt_dir,
        attempt_manifest_path=authority.path,
        attempt_manifest_sha256=authority.sha256,
        checksums_path=authority.checksums_path,
        checksums_sha256=authority.checksums_sha256,
        terminal_path=authority.terminal_path,
        reason_code=authority.reason_code,
        diagnostic=authority.diagnostic,
        failed_symbol=authority.failed_step.symbol,
        failed_date=authority.failed_step.date,
        failed_role=authority.failed_step.role,
        completed_count=len(authority.completed_steps),
        remaining_count=len(authority.remaining_steps),
        retained_inventory_sha256=authority.retained_inventory_sha256,
        retained_artifact_count=len(authority.retained_artifacts),
        total_raw_evidence_bytes=authority.total_raw_evidence_bytes,
        manifest=authority,
    )


def acquire_m8_archives(
    config: M8StudyConfig,
    output_root: str | Path,
    *,
    archive_client: _ArchiveProvider | None = None,
    metadata_provider: _MetadataProvider | Callable[[str, Path], SymbolMetadata] | None = None,
    supplied_metadata: Mapping[str, SymbolMetadata] | Sequence[SymbolMetadata] | None = None,
) -> M8AcquisitionOutcome:
    """Acquire the exact 2+8 raw authorities without opening a CSV member."""

    if metadata_provider is not None and supplied_metadata is not None:
        raise M8AcquisitionError("metadata_provider and supplied_metadata are mutually exclusive")
    root = Path(output_root).expanduser().absolute()
    if root.is_symlink():
        raise M8AcquisitionError("M8 acquisition output root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    _validate_root_layout(root)
    raw_root = root / _RAW_ROOT_NAME
    raw_root.mkdir(parents=True, exist_ok=True)
    protocol_sha = _verify_config_sources(config)
    steps = _acquisition_steps(config)
    try:
        budget = RetainedEvidenceBudget(raw_root, config.study.max_total_download_bytes)
    except EvidenceBudgetExceeded as exc:
        return _finalize_deterministic_failure(
            root=root,
            raw_root=raw_root,
            config=config,
            protocol_sha256=protocol_sha,
            reason_code="TOTAL_EVIDENCE_BUDGET",
            cause=exc,
            failed=steps[0],
            completed=(),
            remaining=steps[1:],
            budget=None,
        )
    using_default_archive = archive_client is None
    if archive_client is None:
        archive_provider: _ArchiveProvider = BinanceArchiveClient(retained_evidence_budget=budget)
    else:
        archive_provider = archive_client
    using_budgeted_metadata = supplied_metadata is not None or metadata_provider is None
    if metadata_provider is None and supplied_metadata is None:
        metadata_fetcher: _MetadataProvider | Callable[[str, Path], SymbolMetadata] = (
            BinancePublicClient(retained_evidence_budget=budget)
        )
    elif metadata_provider is not None:
        metadata_fetcher = metadata_provider
    else:
        metadata_fetcher = BinancePublicClient(retained_evidence_budget=budget)

    try:
        ordered_supplied = (
            None
            if supplied_metadata is None
            else _ordered_supplied_metadata(supplied_metadata, config.study.symbols)
        )
        completed_steps: list[M8AcquisitionStep] = []
        metadata_items: list[M8RawSymbolMetadata] = []
        for index, symbol in enumerate(config.study.symbols):
            step = steps[len(completed_steps)]
            try:
                source_item = (
                    _fetch_provider_metadata(metadata_fetcher, symbol, raw_root)
                    if ordered_supplied is None
                    else ordered_supplied[index]
                )
                metadata_item = _raw_metadata_from_symbol_metadata(root, source_item, symbol)
            except Exception as exc:
                reason = _deterministic_failure_reason(exc)
                if reason is None:
                    raise
                return _finalize_deterministic_failure(
                    root=root,
                    raw_root=raw_root,
                    config=config,
                    protocol_sha256=protocol_sha,
                    reason_code=reason,
                    cause=exc,
                    failed=step,
                    completed=completed_steps,
                    remaining=steps[len(completed_steps) + 1 :],
                    budget=(budget if using_default_archive and using_budgeted_metadata else None),
                )
            metadata_items.append(metadata_item)
            completed_steps.append(step)
        metadata = tuple(metadata_items)
        metadata_by_symbol = MappingProxyType({item.symbol: item for item in metadata})
        limits = ArchiveDownloadLimits(
            max_compressed_bytes=config.study.max_archive_compressed_bytes,
            max_uncompressed_bytes=config.study.max_archive_uncompressed_bytes,
        )
        archives: list[M8RawArchiveEntry] = []
        for period in config.periods:
            for symbol in config.study.symbols:
                symbol_metadata = metadata_by_symbol[symbol]
                request = DailyArchiveRequest(
                    symbol=symbol,
                    date=period.date,
                    tick_size=symbol_metadata.tick_size,
                    lot_size=symbol_metadata.lot_size,
                )
                step = steps[len(completed_steps)]
                try:
                    acquired = archive_provider.acquire(
                        request,
                        raw_root=raw_root,
                        limits=limits,
                    )
                except Exception as exc:
                    reason = _deterministic_failure_reason(exc)
                    if reason is None:
                        raise
                    return _finalize_deterministic_failure(
                        root=root,
                        raw_root=raw_root,
                        config=config,
                        protocol_sha256=protocol_sha,
                        reason_code=reason,
                        cause=exc,
                        failed=step,
                        completed=completed_steps,
                        remaining=steps[len(completed_steps) + 1 :],
                        budget=(
                            budget if using_default_archive and using_budgeted_metadata else None
                        ),
                    )
                if not isinstance(acquired, AcquiredDailyArchive):
                    raise M8AcquisitionError(
                        "archive_client returned a non-AcquiredDailyArchive value"
                    )
                archives.append(
                    _raw_entry_from_acquired(
                        root,
                        acquired,
                        role=period.role,
                        expected_symbol=symbol,
                        expected_date=period.date,
                        limits=limits,
                    )
                )
                completed_steps.append(step)
        retained = _scan_retained_inventory(root, metadata, archives)
        total_raw = sum(item.bytes for item in retained)
        if total_raw > config.study.max_total_download_bytes:
            budget_failure = EvidenceBudgetExceeded(
                "retained raw evidence exceeds the frozen total-byte ceiling"
            )
            return _finalize_deterministic_failure(
                root=root,
                raw_root=raw_root,
                config=config,
                protocol_sha256=protocol_sha,
                reason_code="TOTAL_EVIDENCE_BUDGET",
                cause=budget_failure,
                failed=steps[-1],
                completed=steps[:-1],
                remaining=(),
                budget=None,
            )
        if budget.reserved_bytes != 0:
            raise M8AcquisitionError("retained-evidence budget has unfinished reservations")
        if using_default_archive and using_budgeted_metadata and budget.used_bytes != total_raw:
            raise M8AcquisitionError(
                "shared retained-evidence budget disagrees with the physical inventory"
            )
        _fsync_tree(raw_root)
        durable_retained = _scan_retained_inventory(root, metadata, archives)
        if durable_retained != retained:
            raise M8AcquisitionError("raw inventory changed across success durability barrier")
        retained = durable_retained
        payload = _manifest_payload(
            config=config,
            protocol_sha256=protocol_sha,
            root=root,
            metadata=metadata,
            archives=archives,
            retained=retained,
            copied_from_manifest_sha256=None,
        )
        manifest_path, manifest_sha = _write_manifest_payload(root, payload)
        if budget.used_bytes > config.study.max_total_download_bytes:
            raise M8AcquisitionError("manifest publication changed the raw-evidence budget")
        _validate_root_layout(root)
        manifest = read_m8_acquisition_manifest(
            manifest_path,
            expected_sha256=manifest_sha,
            config=config,
        )
    except M8AcquisitionError:
        raise
    except Exception as exc:
        raise M8AcquisitionError(f"M8 raw-only acquisition failed closed: {exc}") from exc
    return M8AcquisitionResult(
        output_root=root.resolve(),
        manifest_path=manifest.path,
        manifest_sha256=manifest.sha256,
        metadata_count=manifest.metadata_count,
        archive_count=manifest.archive_count,
        total_raw_evidence_bytes=manifest.total_raw_evidence_bytes,
        manifest=manifest,
    )


def _atomic_copy_exact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    if source.is_symlink() or not source.is_file():
        raise M8AcquisitionError(f"copy source is missing or symbolic: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as sink:
            shutil.copyfileobj(source_handle, sink, length=1024 * 1024)
            sink.flush()
            os.fsync(sink.fileno())
        _verify_file(temporary, expected_sha256, expected_bytes, "staged raw-evidence copy")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rebased_evidence(
    authority: M8AcquisitionManifest,
    new_root: Path,
) -> tuple[tuple[M8RawSymbolMetadata, ...], tuple[M8RawArchiveEntry, ...]]:
    metadata = tuple(
        replace(
            item,
            raw_path=(new_root / item.raw_path.relative_to(authority.root)).resolve(),
            source_manifest_path=(
                new_root / item.source_manifest_path.relative_to(authority.root)
            ).resolve(),
        )
        for item in authority.symbol_metadata
    )
    archives = tuple(
        replace(
            item,
            root=new_root.resolve(),
            archive_path=(new_root / item.archive_path.relative_to(authority.root)).resolve(),
            archive_source_manifest_path=(
                new_root / item.archive_source_manifest_path.relative_to(authority.root)
            ).resolve(),
            checksum_path=(new_root / item.checksum_path.relative_to(authority.root)).resolve(),
            checksum_source_manifest_path=(
                new_root / item.checksum_source_manifest_path.relative_to(authority.root)
            ).resolve(),
        )
        for item in authority.archives
    )
    return metadata, archives


def _assert_distinct_copy_budget(total_raw_bytes: int, max_total_bytes: int) -> None:
    if total_raw_bytes > max_total_bytes - total_raw_bytes:
        raise M8AcquisitionError(
            "external raw evidence plus its distinct self-contained copy exceeds the frozen "
            "total-byte ceiling"
        )


def copy_m8_acquisition_into(
    authority: M8AcquisitionManifest,
    input_root: str | Path,
) -> M8AcquisitionManifest:
    """Atomically copy the complete raw inventory into a self-contained input root."""

    verified = read_m8_acquisition_manifest(
        authority.path,
        expected_sha256=authority.sha256,
        config=authority.config,
    )
    _assert_distinct_copy_budget(
        verified.total_raw_evidence_bytes,
        verified.config.study.max_total_download_bytes,
    )
    destination = Path(input_root).expanduser().absolute()
    source_root = verified.root.resolve()
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise M8AcquisitionError("self-contained M8 input root must not already exist")
    destination_resolved = destination.resolve(strict=False)
    if (
        destination_resolved == source_root
        or destination_resolved.is_relative_to(source_root)
        or source_root.is_relative_to(destination_resolved)
    ):
        raise M8AcquisitionError("M8 acquisition source and copy roots must not overlap")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.copy-", dir=destination_parent)
    ).resolve()
    try:
        for artifact in verified.retained_artifacts:
            source = verified.root / artifact.path
            target = stage / artifact.path
            _atomic_copy_exact(
                source,
                target,
                expected_sha256=artifact.sha256,
                expected_bytes=artifact.bytes,
            )
        metadata, archives = _rebased_evidence(verified, stage)
        retained = _scan_retained_inventory(stage, metadata, archives)
        if retained != verified.retained_artifacts:
            raise M8AcquisitionError("self-contained copy inventory differs from authority")
        _fsync_tree(stage / _RAW_ROOT_NAME)
        durable_retained = _scan_retained_inventory(stage, metadata, archives)
        if durable_retained != retained:
            raise M8AcquisitionError("copied raw inventory changed across durability barrier")
        retained = durable_retained
        payload = _manifest_payload(
            config=verified.config,
            protocol_sha256=verified.protocol_document_sha256,
            root=stage,
            metadata=metadata,
            archives=archives,
            retained=retained,
            copied_from_manifest_sha256=verified.sha256,
        )
        stage_manifest_path, stage_manifest_sha = _write_manifest_payload(stage, payload)
        _validate_root_layout(stage)
        read_m8_acquisition_manifest(
            stage_manifest_path,
            expected_sha256=stage_manifest_sha,
            config=verified.config,
        )
        _fsync_tree(stage)
        os.replace(stage, destination)
        _fsync_directory(destination_parent)
        final_manifest_path = destination / _MANIFEST_DIRECTORY_NAME / stage_manifest_path.name
        copied = read_m8_acquisition_manifest(
            final_manifest_path,
            expected_sha256=stage_manifest_sha,
            config=verified.config,
        )
        if (
            copied.copied_from_manifest_sha256 != verified.sha256
            or copied.content_identity_sha256 != verified.content_identity_sha256
            or copied.retained_artifacts != verified.retained_artifacts
            or copied.total_raw_evidence_bytes != verified.total_raw_evidence_bytes
            or copied.total_accepted_zip_bytes != verified.total_accepted_zip_bytes
        ):
            raise M8AcquisitionError("self-contained acquisition copy is not source-equivalent")
        return copied
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


__all__ = [
    "M8_ACQUISITION_SCHEMA_VERSION",
    "M8AcquisitionError",
    "M8AcquisitionFailureManifest",
    "M8AcquisitionFailureResult",
    "M8AcquisitionManifest",
    "M8AcquisitionOutcome",
    "M8AcquisitionReasonCode",
    "M8AcquisitionResult",
    "M8AcquisitionStep",
    "M8RawArchiveDescriptor",
    "M8RawArchiveEntry",
    "M8RawSymbolMetadata",
    "M8RetainedArtifact",
    "acquire_m8_archives",
    "copy_m8_acquisition_into",
    "read_m8_acquisition_failure",
    "read_m8_acquisition_manifest",
    "verify_m8_acquisition_failure",
    "verify_m8_acquisition_manifest",
]
