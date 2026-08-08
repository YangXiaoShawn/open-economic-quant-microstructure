"""Immutable input-manifest contract for the frozen M8 trade study.

The verifier intentionally inspects only JSON metadata, exact file hashes, and
Parquet footers/schemas.  It never materializes normalized trade rows.  Economic
row-level validation belongs to the acquisition/quality stage whose immutable
reports and findings are bound here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as Date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import parse_qsl, urlparse
from zipfile import BadZipFile, ZipFile

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from microstructure.data.schemas import SCHEMA_VERSION, get_schema
from microstructure.m8_config import M8PeriodRole, M8StudyConfig
from microstructure.provenance import sha256_file

M8_INPUT_MANIFEST_VERSION = "1.2.0"
_STORAGE_MANIFEST_VERSION = "1.0.0"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP = re.compile(
    r"(?P<second>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?:Z|\+00:00)"
)
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_SYMBOL_METADATA_BYTES = 1 * 1024 * 1024
_MAX_ARCHIVE_CHECKSUM_BYTES = 4_096
_MAX_ZIP_DIRECTORY_BYTES = 256 * 1_024
_MAX_ZIP_TAIL_BYTES = 22 + 65_535
_NS_PER_DAY = 86_400 * 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ZIP_CENTRAL_SIGNATURE = b"PK\x01\x02"
_ZIP_CENTRAL_STRUCT = struct.Struct("<4s6H3L5H2L")
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP_EOCD_STRUCT = struct.Struct("<4s4H2LH")
_ZIP64_EXTRA_FIELD_ID = 0x0001
_ZIP16_SENTINEL = 0xFFFF
_ZIP32_SENTINEL = 0xFFFFFFFF

_TOP_KEYS = frozenset(
    {
        "manifest_version",
        "artifact_kind",
        "config",
        "study",
        "symbol_metadata",
        "total_symbol_metadata_bytes",
        "total_raw_zip_bytes",
        "entries",
    }
)
_CONFIG_KEYS = frozenset({"semantic_sha256", "source_sha256", "protocol_version"})
_STUDY_KEYS = frozenset({"name", "evidence_tier", "source", "symbols", "periods"})
_STUDY_PERIOD_KEYS = frozenset({"date", "role"})
_SYMBOL_METADATA_KEYS = frozenset(
    {
        "symbol",
        "status",
        "tick_size",
        "lot_size",
        "observed_ts_ns",
        "raw_path",
        "raw_sha256",
        "raw_bytes",
        "source_uri",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_manifest_bytes",
    }
)
_ARCHIVE_CHECKSUM_KEYS = frozenset(
    {
        "path",
        "sha256",
        "bytes",
        "source_uri",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_manifest_bytes",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "symbol",
        "date",
        "role",
        "complete",
        "requested_range_ns",
        "rows",
        "trade_id_range",
        "observed_range_ns",
        "scales",
        "raw",
        "normalized",
        "quality",
    }
)
_RANGE_KEYS = frozenset({"start", "end_exclusive"})
_OBSERVED_KEYS = frozenset({"start", "end_inclusive"})
_TRADE_ID_KEYS = frozenset({"first", "last", "contiguous_count"})
_SCALE_KEYS = frozenset({"tick_size", "lot_size"})
_CHECKSUM_KEYS = frozenset({"algorithm", "value"})
_RAW_KEYS = frozenset(
    {
        "zip_path",
        "zip_sha256",
        "zip_bytes",
        "uncompressed_bytes",
        "source_uri",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_manifest_bytes",
        "checksum",
    }
)
_NORMALIZED_KEYS = frozenset(
    {
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "dataset_manifest_bytes",
        "rows",
        "parts",
    }
)
_PART_KEYS = frozenset(
    {
        "data_path",
        "data_sha256",
        "data_bytes",
        "sidecar_path",
        "sidecar_sha256",
        "sidecar_bytes",
        "rows",
        "write_ordinal",
        "observed_range_ns",
    }
)
_QUALITY_KEYS = frozenset(
    {
        "report_path",
        "report_sha256",
        "report_bytes",
        "findings_path",
        "findings_sha256",
        "findings_bytes",
        "errors",
        "warnings",
    }
)
_RAW_SIDECAR_KEYS = frozenset(
    {
        "manifest_version",
        "artifact_kind",
        "source",
        "source_uri",
        "downloaded_at_utc",
        "requested_range_ns",
        "checksum",
        "upstream_checksum_sha256",
        "bytes",
        "path",
        "response_headers",
    }
)
_DATASET_KEYS = frozenset(
    {
        "manifest_version",
        "dataset",
        "schema_version",
        "source",
        "source_uri",
        "downloaded_at_utc",
        "requested_range_ns",
        "artifacts",
        "rows",
    }
)
_DATASET_PART_KEYS = frozenset(
    {
        "data_path",
        "manifest_path",
        "data_sha256",
        "manifest_sha256",
        "rows",
        "write_ordinal",
        "observed_range_ns",
    }
)
_PART_SIDECAR_KEYS = frozenset(
    {
        "manifest_version",
        "artifact_kind",
        "dataset",
        "schema_name",
        "schema_version",
        "venue",
        "symbol",
        "partition_date",
        "write_ordinal",
        "source",
        "source_uri",
        "downloaded_at_utc",
        "requested_range_ns",
        "observed_range_ns",
        "source_checksum_sha256",
        "checksum",
        "rows",
        "bytes",
        "path",
        "transformations",
    }
)


class M8ManifestError(RuntimeError):
    """Raised when M8 input evidence is missing, unsafe, or inconsistent."""


@dataclass(frozen=True, slots=True)
class M8NormalizedPart:
    data_path: Path
    data_sha256: str
    data_bytes: int
    sidecar_path: Path
    sidecar_sha256: str
    sidecar_bytes: int
    rows: int
    write_ordinal: int
    observed_start_ns: int
    observed_end_inclusive_ns: int


@dataclass(frozen=True, slots=True)
class M8SymbolMetadata:
    symbol: str
    status: str
    tick_size: Decimal
    lot_size: Decimal
    observed_ts_ns: int
    raw_path: Path
    raw_sha256: str
    raw_bytes: int
    source_uri: str
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest_bytes: int


@dataclass(frozen=True, slots=True)
class M8ArchiveEntry:
    symbol: str
    date: Date
    role: M8PeriodRole
    complete: bool
    rows: int
    first_trade_id: int
    last_trade_id: int
    observed_start_ns: int
    observed_end_inclusive_ns: int
    tick_size: Decimal
    lot_size: Decimal
    raw_zip_path: Path
    raw_zip_sha256: str
    raw_zip_bytes: int
    raw_uncompressed_bytes: int
    raw_source_uri: str
    raw_source_manifest_path: Path
    raw_source_manifest_sha256: str
    raw_source_manifest_bytes: int
    raw_checksum_path: Path
    raw_checksum_sha256: str
    raw_checksum_bytes: int
    raw_checksum_source_uri: str
    raw_checksum_source_manifest_path: Path
    raw_checksum_source_manifest_sha256: str
    raw_checksum_source_manifest_bytes: int
    normalized_dataset_manifest_path: Path
    normalized_dataset_manifest_sha256: str
    normalized_dataset_manifest_bytes: int
    normalized_parts: tuple[M8NormalizedPart, ...]
    quality_report_path: Path
    quality_report_sha256: str
    quality_report_bytes: int
    quality_findings_path: Path
    quality_findings_sha256: str
    quality_findings_bytes: int
    quality_errors: int
    quality_warnings: int


@dataclass(frozen=True, slots=True)
class M8InputManifest:
    root: Path
    path: Path
    sha256: str
    config_sha256: str
    config_source_sha256: str
    protocol_version: str
    symbol_metadata: tuple[M8SymbolMetadata, ...]
    entries: tuple[M8ArchiveEntry, ...]

    def metadata_for(self, symbol: str) -> M8SymbolMetadata:
        for metadata in self.symbol_metadata:
            if metadata.symbol == symbol:
                return metadata
        raise KeyError(f"no verified M8 symbol metadata for {symbol}")

    @property
    def ordered_part_paths(self) -> Mapping[tuple[str, str], tuple[Path, ...]]:
        """Return verified Parquet paths keyed in frozen period/symbol order."""

        paths = {
            (entry.symbol, entry.date.isoformat()): tuple(
                part.data_path for part in entry.normalized_parts
            )
            for entry in self.entries
        }
        return MappingProxyType(paths)

    def part_paths_for(self, symbol: str, date: Date | str) -> tuple[Path, ...]:
        date_text = date.isoformat() if isinstance(date, Date) else date
        try:
            return self.ordered_part_paths[(symbol, date_text)]
        except KeyError as exc:
            raise KeyError(f"no verified M8 parts for {symbol}/{date_text}") from exc


def _day_bounds_ns(day: Date) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    delta = start - _EPOCH
    start_ns = (delta.days * 86_400 + delta.seconds) * 1_000_000_000
    return start_ns, start_ns + _NS_PER_DAY


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M8ManifestError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise M8ManifestError(f"{label} contains a non-string key")
    return cast(Mapping[str, Any], value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise M8ManifestError(f"{label} must be a JSON array")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise M8ManifestError(f"{label} must be a string")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise M8ManifestError(f"{label} must be an integer")
    result = value
    if minimum is not None and result < minimum:
        raise M8ManifestError(f"{label} must be at least {minimum}")
    return result


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise M8ManifestError(f"{label} must be a boolean")
    return value


def _digest(value: object, label: str) -> str:
    digest = _text(value, label)
    if _DIGEST.fullmatch(digest) is None:
        raise M8ManifestError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _decimal(value: object, label: str) -> Decimal:
    raw = _text(value, label)
    try:
        result = Decimal(raw)
    except InvalidOperation as exc:
        raise M8ManifestError(f"{label} must be a decimal string") from exc
    if not result.is_finite() or result <= 0:
        raise M8ManifestError(f"{label} must be finite and positive")
    return result


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise M8ManifestError(f"{label} keys are invalid ({'; '.join(details)})")


def _parse_date(value: object, label: str) -> Date:
    raw = _text(value, label)
    try:
        parsed = Date.fromisoformat(raw)
    except ValueError as exc:
        raise M8ManifestError(f"{label} must be an ISO UTC date") from exc
    if parsed.isoformat() != raw:
        raise M8ManifestError(f"{label} must use canonical YYYY-MM-DD form")
    return parsed


def _parse_role(value: object, label: str) -> M8PeriodRole:
    raw = _text(value, label)
    allowed = {"train", "validation", "primary_test", "replication_test"}
    if raw not in allowed:
        raise M8ManifestError(f"{label} has an unsupported role: {raw!r}")
    return cast(M8PeriodRole, raw)


def _utc_ns_from_iso(value: object, label: str) -> int:
    raw = _text(value, label)
    matched = _UTC_TIMESTAMP.fullmatch(raw)
    if matched is None:
        raise M8ManifestError(
            f"{label} must be an ISO-8601 UTC timestamp with at most nanosecond precision"
        )
    try:
        second = datetime.strptime(matched.group("second"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    except ValueError as exc:
        raise M8ManifestError(f"{label} is not a valid UTC timestamp") from exc
    fraction = (matched.group("fraction") or "").ljust(9, "0")
    delta = second - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + int(fraction or "0")


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise M8ManifestError(f"cannot stat {label} at {path}: {exc}") from exc
    if size > _MAX_JSON_BYTES:
        raise M8ManifestError(f"{label} exceeds the {_MAX_JSON_BYTES}-byte JSON limit")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise M8ManifestError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as handle:
            parsed = json.load(handle, object_pairs_hook=reject_duplicates)
    except M8ManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M8ManifestError(f"cannot parse {label} at {path}: {exc}") from exc
    return _object(parsed, label)


def _resolve_contained_file(root: Path, value: str | Path, label: str) -> Path:
    declared = Path(value)
    candidate = declared if declared.is_absolute() else root / declared
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise M8ManifestError(f"{label} escapes the M8 input root: {declared}")
    if not resolved.is_file():
        raise M8ManifestError(f"missing {label}: {resolved}")
    return resolved


def _declared_file(root: Path, value: object, label: str) -> Path:
    raw = _text(value, label)
    if Path(raw).is_absolute():
        raise M8ManifestError(f"{label} must be relative to the M8 input root")
    return _resolve_contained_file(root, raw, label)


def _relative_file(root: Path, value: Path, label: str) -> str:
    resolved = _resolve_contained_file(root, value, label)
    return resolved.relative_to(root.resolve()).as_posix()


def _verify_file(path: Path, sha256: str, byte_count: int, label: str) -> None:
    if path.stat().st_size != byte_count:
        raise M8ManifestError(f"{label} byte count does not match")
    observed = sha256_file(path)
    if observed != sha256:
        raise M8ManifestError(f"{label} SHA-256 mismatch: expected {sha256}, observed {observed}")


def _range(value: object, label: str) -> tuple[int, int]:
    raw = _object(value, label)
    _exact_keys(raw, _RANGE_KEYS, label)
    start = _integer(raw["start"], f"{label}.start")
    end = _integer(raw["end_exclusive"], f"{label}.end_exclusive")
    if end <= start:
        raise M8ManifestError(f"{label} must be non-empty")
    return start, end


def _observed_range(value: object, label: str) -> tuple[int, int]:
    raw = _object(value, label)
    _exact_keys(raw, _OBSERVED_KEYS, label)
    start = _integer(raw["start"], f"{label}.start")
    end = _integer(raw["end_inclusive"], f"{label}.end_inclusive")
    if end < start:
        raise M8ManifestError(f"{label} ends before it starts")
    return start, end


def _verify_day_range(observed: tuple[int, int], day_range: tuple[int, int], label: str) -> None:
    if observed[0] < day_range[0] or observed[1] >= day_range[1]:
        raise M8ManifestError(f"{label} falls outside its declared UTC day")


def _expected_entry_order(config: M8StudyConfig) -> tuple[tuple[str, Date, M8PeriodRole], ...]:
    return tuple(
        (symbol, period.date, period.role)
        for period in config.periods
        for symbol in config.study.symbols
    )


def _ordered_symbol_metadata(
    config: M8StudyConfig, values: Sequence[M8SymbolMetadata]
) -> tuple[M8SymbolMetadata, ...]:
    expected_symbols = config.study.symbols
    observed: dict[str, M8SymbolMetadata] = {}
    for metadata in values:
        if metadata.symbol in observed:
            raise M8ManifestError(f"duplicate M8 symbol metadata for {metadata.symbol}")
        if metadata.symbol not in expected_symbols:
            raise M8ManifestError(f"extra M8 symbol metadata for {metadata.symbol}")
        observed[metadata.symbol] = metadata
    missing = [symbol for symbol in expected_symbols if symbol not in observed]
    if missing:
        raise M8ManifestError("missing M8 symbol metadata: " + ", ".join(missing))
    if len(observed) != 2:
        raise M8ManifestError("the frozen M8 study requires exactly two symbol metadata entries")
    return tuple(observed[symbol] for symbol in expected_symbols)


def _ordered_entries(
    config: M8StudyConfig, entries: Sequence[M8ArchiveEntry]
) -> tuple[M8ArchiveEntry, ...]:
    expected = _expected_entry_order(config)
    expected_keys = {(symbol, day): role for symbol, day, role in expected}
    observed: dict[tuple[str, Date], M8ArchiveEntry] = {}
    for entry in entries:
        key = (entry.symbol, entry.date)
        if key in observed:
            raise M8ManifestError(
                f"duplicate M8 archive entry for {entry.symbol}/{entry.date.isoformat()}"
            )
        expected_role = expected_keys.get(key)
        if expected_role is None:
            raise M8ManifestError(
                f"extra M8 archive entry for {entry.symbol}/{entry.date.isoformat()}"
            )
        if entry.role != expected_role:
            raise M8ManifestError(
                f"role mismatch for {entry.symbol}/{entry.date.isoformat()}: "
                f"expected {expected_role}, observed {entry.role}"
            )
        observed[key] = entry
    missing = [
        f"{symbol}/{day.isoformat()}"
        for symbol, day, _role in expected
        if (symbol, day) not in observed
    ]
    if missing:
        raise M8ManifestError("missing M8 archive entries: " + ", ".join(missing))
    if len(observed) != 8:
        raise M8ManifestError("the frozen M8 study requires exactly eight symbol/date entries")
    return tuple(observed[(symbol, day)] for symbol, day, _role in expected)


def _symbol_metadata_payload(root: Path, metadata: M8SymbolMetadata) -> dict[str, object]:
    return {
        "symbol": metadata.symbol,
        "status": metadata.status,
        "tick_size": format(metadata.tick_size, "f"),
        "lot_size": format(metadata.lot_size, "f"),
        "observed_ts_ns": metadata.observed_ts_ns,
        "raw_path": _relative_file(root, metadata.raw_path, "exchangeInfo raw body"),
        "raw_sha256": metadata.raw_sha256,
        "raw_bytes": metadata.raw_bytes,
        "source_uri": metadata.source_uri,
        "source_manifest_path": _relative_file(
            root,
            metadata.source_manifest_path,
            "exchangeInfo source sidecar",
        ),
        "source_manifest_sha256": metadata.source_manifest_sha256,
        "source_manifest_bytes": metadata.source_manifest_bytes,
    }


def _part_payload(root: Path, part: M8NormalizedPart) -> dict[str, object]:
    return {
        "data_path": _relative_file(root, part.data_path, "normalized Parquet part"),
        "data_sha256": part.data_sha256,
        "data_bytes": part.data_bytes,
        "sidecar_path": _relative_file(root, part.sidecar_path, "normalized part sidecar"),
        "sidecar_sha256": part.sidecar_sha256,
        "sidecar_bytes": part.sidecar_bytes,
        "rows": part.rows,
        "write_ordinal": part.write_ordinal,
        "observed_range_ns": {
            "start": part.observed_start_ns,
            "end_inclusive": part.observed_end_inclusive_ns,
        },
    }


def _entry_payload(root: Path, entry: M8ArchiveEntry) -> dict[str, object]:
    day_start, day_end = _day_bounds_ns(entry.date)
    parts = sorted(entry.normalized_parts, key=lambda item: item.write_ordinal)
    return {
        "symbol": entry.symbol,
        "date": entry.date.isoformat(),
        "role": entry.role,
        "complete": entry.complete,
        "requested_range_ns": {"start": day_start, "end_exclusive": day_end},
        "rows": entry.rows,
        "trade_id_range": {
            "first": entry.first_trade_id,
            "last": entry.last_trade_id,
            "contiguous_count": entry.rows,
        },
        "observed_range_ns": {
            "start": entry.observed_start_ns,
            "end_inclusive": entry.observed_end_inclusive_ns,
        },
        "scales": {
            "tick_size": format(entry.tick_size, "f"),
            "lot_size": format(entry.lot_size, "f"),
        },
        "raw": {
            "zip_path": _relative_file(root, entry.raw_zip_path, "raw ZIP"),
            "zip_sha256": entry.raw_zip_sha256,
            "zip_bytes": entry.raw_zip_bytes,
            "uncompressed_bytes": entry.raw_uncompressed_bytes,
            "source_uri": entry.raw_source_uri,
            "source_manifest_path": _relative_file(
                root, entry.raw_source_manifest_path, "raw source sidecar"
            ),
            "source_manifest_sha256": entry.raw_source_manifest_sha256,
            "source_manifest_bytes": entry.raw_source_manifest_bytes,
            "checksum": {
                "path": _relative_file(
                    root,
                    entry.raw_checksum_path,
                    "official archive CHECKSUM",
                ),
                "sha256": entry.raw_checksum_sha256,
                "bytes": entry.raw_checksum_bytes,
                "source_uri": entry.raw_checksum_source_uri,
                "source_manifest_path": _relative_file(
                    root,
                    entry.raw_checksum_source_manifest_path,
                    "official archive CHECKSUM source sidecar",
                ),
                "source_manifest_sha256": entry.raw_checksum_source_manifest_sha256,
                "source_manifest_bytes": entry.raw_checksum_source_manifest_bytes,
            },
        },
        "normalized": {
            "dataset_manifest_path": _relative_file(
                root,
                entry.normalized_dataset_manifest_path,
                "normalized dataset manifest",
            ),
            "dataset_manifest_sha256": entry.normalized_dataset_manifest_sha256,
            "dataset_manifest_bytes": entry.normalized_dataset_manifest_bytes,
            "rows": entry.rows,
            "parts": [_part_payload(root, part) for part in parts],
        },
        "quality": {
            "report_path": _relative_file(root, entry.quality_report_path, "quality report"),
            "report_sha256": entry.quality_report_sha256,
            "report_bytes": entry.quality_report_bytes,
            "findings_path": _relative_file(root, entry.quality_findings_path, "quality findings"),
            "findings_sha256": entry.quality_findings_sha256,
            "findings_bytes": entry.quality_findings_bytes,
            "errors": entry.quality_errors,
            "warnings": entry.quality_warnings,
        },
    }


def _manifest_payload(
    config: M8StudyConfig,
    root: Path,
    entries: Sequence[M8ArchiveEntry],
    symbol_metadata: Sequence[M8SymbolMetadata],
) -> dict[str, object]:
    ordered = _ordered_entries(config, entries)
    ordered_metadata = _ordered_symbol_metadata(config, symbol_metadata)
    entry_payloads = [_entry_payload(root, entry) for entry in ordered]
    return {
        "manifest_version": M8_INPUT_MANIFEST_VERSION,
        "artifact_kind": "m8_multidate_trade_input",
        "config": {
            "semantic_sha256": config.hash,
            "source_sha256": config.source_sha256,
            "protocol_version": config.study.protocol_version,
        },
        "study": {
            "name": config.study.name,
            "evidence_tier": config.study.evidence_tier,
            "source": config.study.source,
            "symbols": list(config.study.symbols),
            "periods": [
                {"date": period.date.isoformat(), "role": period.role} for period in config.periods
            ],
        },
        "symbol_metadata": [
            _symbol_metadata_payload(root, metadata) for metadata in ordered_metadata
        ],
        "total_symbol_metadata_bytes": sum(metadata.raw_bytes for metadata in ordered_metadata),
        "total_raw_zip_bytes": sum(entry.raw_zip_bytes for entry in ordered),
        "entries": entry_payloads,
    }


def _verify_exchange_info_uri(source_uri: str, symbol: str, label: str) -> None:
    try:
        parsed = urlparse(source_uri)
        port = parsed.port
    except ValueError as exc:
        raise M8ManifestError(f"{label} has an invalid network location") from exc
    try:
        query = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=2,
        )
    except ValueError as exc:
        raise M8ManifestError(f"{label} has an invalid query") from exc
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
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in source_uri)
    ):
        raise M8ManifestError(
            f"{label} is not the exact official exchangeInfo request for {symbol}"
        )


def _verify_null_requested_range(value: object, label: str) -> None:
    requested = _object(value, label)
    _exact_keys(requested, _RANGE_KEYS, label)
    if requested["start"] is not None or requested["end_exclusive"] is not None:
        raise M8ManifestError(f"{label} must have null API range bounds")


def _verify_exchange_info_payload(
    raw_path: Path,
    *,
    symbol: str,
    status: str,
    tick_size: Decimal,
    lot_size: Decimal,
    label: str,
) -> None:
    payload = _read_json_object(raw_path, f"{label} raw body")
    symbols = _array(payload.get("symbols"), f"{label} raw body symbols")
    if len(symbols) != 1:
        raise M8ManifestError(f"{label} raw body must contain exactly the requested symbol")
    item = _object(symbols[0], f"{label} raw body symbol")
    if _text(item.get("symbol"), f"{label} raw body symbol name") != symbol:
        raise M8ManifestError(f"{label} raw body returned a different symbol")
    if _text(item.get("status"), f"{label} raw body status") != status:
        raise M8ManifestError(f"{label} raw body status does not match")

    filters: dict[str, Mapping[str, Any]] = {}
    for index, raw_filter in enumerate(_array(item.get("filters"), f"{label} raw body filters")):
        filter_value = _object(raw_filter, f"{label} raw body filters[{index}]")
        filter_type = _text(
            filter_value.get("filterType"),
            f"{label} raw body filters[{index}].filterType",
        )
        if filter_type in filters:
            raise M8ManifestError(f"{label} raw body has duplicate {filter_type} filters")
        filters[filter_type] = filter_value
    try:
        price_filter = filters["PRICE_FILTER"]
        lot_filter = filters["LOT_SIZE"]
    except KeyError as exc:
        raise M8ManifestError(
            f"{label} raw body lacks PRICE_FILTER or LOT_SIZE provenance"
        ) from exc
    source_tick = _decimal(price_filter.get("tickSize"), f"{label} raw PRICE_FILTER.tickSize")
    source_lot = _decimal(lot_filter.get("stepSize"), f"{label} raw LOT_SIZE.stepSize")
    if source_tick != tick_size or source_lot != lot_size:
        raise M8ManifestError(f"{label} declared scales do not match its raw exchangeInfo body")


def _verify_symbol_metadata(
    root: Path,
    value: object,
    expected_symbol: str,
    index: int,
) -> M8SymbolMetadata:
    label = f"symbol_metadata[{index}]"
    metadata = _object(value, label)
    _exact_keys(metadata, _SYMBOL_METADATA_KEYS, label)
    symbol = _text(metadata["symbol"], f"{label}.symbol")
    if symbol != expected_symbol:
        raise M8ManifestError(f"{label} is out of frozen symbol order")
    status = _text(metadata["status"], f"{label}.status")
    if status != "TRADING":
        raise M8ManifestError(f"{label} must prove that {symbol} was TRADING")
    tick_size = _decimal(metadata["tick_size"], f"{label}.tick_size")
    lot_size = _decimal(metadata["lot_size"], f"{label}.lot_size")
    observed_ts_ns = _integer(metadata["observed_ts_ns"], f"{label}.observed_ts_ns", minimum=1)

    raw_path = _declared_file(root, metadata["raw_path"], f"{label} raw body")
    raw_sha = _digest(metadata["raw_sha256"], f"{label} raw SHA")
    raw_bytes = _integer(metadata["raw_bytes"], f"{label} raw bytes", minimum=1)
    if raw_bytes > _MAX_SYMBOL_METADATA_BYTES:
        raise M8ManifestError(
            f"{label} raw body exceeds the {_MAX_SYMBOL_METADATA_BYTES}-byte metadata limit"
        )
    _verify_file(raw_path, raw_sha, raw_bytes, f"{label} raw body")

    source_uri = _text(metadata["source_uri"], f"{label}.source_uri")
    _verify_exchange_info_uri(source_uri, symbol, f"{label}.source_uri")
    source_manifest_path = _declared_file(
        root,
        metadata["source_manifest_path"],
        f"{label} source sidecar",
    )
    source_manifest_sha = _digest(metadata["source_manifest_sha256"], f"{label} source sidecar SHA")
    source_manifest_bytes = _integer(
        metadata["source_manifest_bytes"],
        f"{label} source sidecar bytes",
        minimum=1,
    )
    _verify_file(
        source_manifest_path,
        source_manifest_sha,
        source_manifest_bytes,
        f"{label} source sidecar",
    )
    if raw_path == source_manifest_path:
        raise M8ManifestError(f"{label} raw body and source sidecar must be distinct files")

    sidecar = _read_json_object(source_manifest_path, f"{label} source sidecar")
    _exact_keys(sidecar, _RAW_SIDECAR_KEYS, f"{label} source sidecar")
    expected_claims: dict[str, object] = {
        "manifest_version": _STORAGE_MANIFEST_VERSION,
        "artifact_kind": "raw_source",
        "source": "binance_spot_public_api",
        "source_uri": source_uri,
        "bytes": raw_bytes,
        "path": raw_path.name,
    }
    for key, expected in expected_claims.items():
        if sidecar[key] != expected:
            raise M8ManifestError(f"{label} source sidecar {key} claim does not match")
    _verify_null_requested_range(
        sidecar["requested_range_ns"], f"{label} source sidecar requested range"
    )
    checksum = _object(sidecar["checksum"], f"{label} source sidecar checksum")
    _exact_keys(checksum, _CHECKSUM_KEYS, f"{label} source sidecar checksum")
    if (
        checksum["algorithm"] != "sha256"
        or _digest(checksum["value"], f"{label} source sidecar checksum value") != raw_sha
    ):
        raise M8ManifestError(f"{label} source sidecar checksum does not match")
    if sidecar["upstream_checksum_sha256"] is not None:
        raise M8ManifestError(f"{label} API source sidecar must not claim an upstream checksum")
    downloaded_ts_ns = _utc_ns_from_iso(
        sidecar["downloaded_at_utc"], f"{label} source sidecar download timestamp"
    )
    if downloaded_ts_ns != observed_ts_ns:
        raise M8ManifestError(f"{label} observed timestamp does not match its source sidecar")
    response_headers = _object(
        sidecar["response_headers"], f"{label} source sidecar response headers"
    )
    if not all(type(header_value) is str for header_value in response_headers.values()):
        raise M8ManifestError(f"{label} source sidecar response headers must be strings")

    _verify_exchange_info_payload(
        raw_path,
        symbol=symbol,
        status=status,
        tick_size=tick_size,
        lot_size=lot_size,
        label=label,
    )
    return M8SymbolMetadata(
        symbol=symbol,
        status=status,
        tick_size=tick_size,
        lot_size=lot_size,
        observed_ts_ns=observed_ts_ns,
        raw_path=raw_path,
        raw_sha256=raw_sha,
        raw_bytes=raw_bytes,
        source_uri=source_uri,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha,
        source_manifest_bytes=source_manifest_bytes,
    )


def _verify_official_daily_uri(
    source_uri: str,
    *,
    symbol: str,
    expected_name: str,
    label: str,
) -> None:
    try:
        parsed_uri = urlparse(source_uri)
        port = parsed_uri.port
    except ValueError as exc:
        raise M8ManifestError(f"{label} has an invalid network location") from exc
    expected_uri_path = f"/data/spot/daily/aggTrades/{symbol}/{expected_name}"
    expected_uri = f"https://data.binance.vision{expected_uri_path}"
    if (
        source_uri != expected_uri
        or parsed_uri.scheme != "https"
        or parsed_uri.netloc != "data.binance.vision"
        or parsed_uri.hostname != "data.binance.vision"
        or port is not None
        or parsed_uri.username is not None
        or parsed_uri.password is not None
        or parsed_uri.path != expected_uri_path
        or parsed_uri.params
        or parsed_uri.query
        or parsed_uri.fragment
    ):
        raise M8ManifestError(f"{label} is not the exact official daily archive URI")


def _verify_official_checksum(
    *,
    entry_label: str,
    root: Path,
    value: object,
    symbol: str,
    day: Date,
    day_range: tuple[int, int],
    zip_path: Path,
    zip_sha256: str,
    archive_source_uri: str,
    archive_sidecar_path: Path,
) -> tuple[Path, str, int, str, Path, str, int]:
    label = f"{entry_label} official CHECKSUM"
    checksum = _object(value, label)
    _exact_keys(checksum, _ARCHIVE_CHECKSUM_KEYS, label)
    archive_name = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    checksum_name = f"{archive_name}.CHECKSUM"

    checksum_path = _declared_file(root, checksum["path"], f"{label} raw body")
    if checksum_path.name != checksum_name:
        raise M8ManifestError(f"{label} filename does not match the official archive basename")
    checksum_sha = _digest(checksum["sha256"], f"{label} raw SHA")
    checksum_bytes = _integer(checksum["bytes"], f"{label} raw bytes", minimum=1)
    if checksum_bytes > _MAX_ARCHIVE_CHECKSUM_BYTES:
        raise M8ManifestError(
            f"{label} exceeds the {_MAX_ARCHIVE_CHECKSUM_BYTES}-byte checksum limit"
        )
    _verify_file(checksum_path, checksum_sha, checksum_bytes, f"{label} raw body")
    try:
        checksum_body = checksum_path.read_bytes()
    except OSError as exc:
        raise M8ManifestError(f"cannot read {label} raw body: {exc}") from exc
    expected_line = f"{zip_sha256}  {archive_name}".encode("ascii")
    if checksum_body not in {
        expected_line,
        expected_line + b"\n",
        expected_line + b"\r\n",
    }:
        raise M8ManifestError(f"{label} body must be exactly one ZIP SHA-256/basename line")

    source_uri = _text(checksum["source_uri"], f"{label}.source_uri")
    if source_uri != f"{archive_source_uri}.CHECKSUM":
        raise M8ManifestError(f"{label} URI is not bound to its ZIP URI")
    _verify_official_daily_uri(
        source_uri,
        symbol=symbol,
        expected_name=checksum_name,
        label=f"{label}.source_uri",
    )
    sidecar_path = _declared_file(
        root,
        checksum["source_manifest_path"],
        f"{label} source sidecar",
    )
    sidecar_sha = _digest(checksum["source_manifest_sha256"], f"{label} source sidecar SHA")
    sidecar_bytes = _integer(
        checksum["source_manifest_bytes"],
        f"{label} source sidecar bytes",
        minimum=1,
    )
    _verify_file(sidecar_path, sidecar_sha, sidecar_bytes, f"{label} source sidecar")
    if checksum_path in {zip_path, archive_sidecar_path, sidecar_path} or sidecar_path in {
        zip_path,
        archive_sidecar_path,
    }:
        raise M8ManifestError(f"{label} evidence paths must be distinct")

    sidecar = _read_json_object(sidecar_path, f"{label} source sidecar")
    _exact_keys(sidecar, _RAW_SIDECAR_KEYS, f"{label} source sidecar")
    expected_claims: dict[str, object] = {
        "manifest_version": _STORAGE_MANIFEST_VERSION,
        "artifact_kind": "raw_source",
        "source": "binance_spot_daily_aggtrades_archive_checksum",
        "source_uri": source_uri,
        "bytes": checksum_bytes,
        "path": checksum_name,
    }
    for key, expected in expected_claims.items():
        if sidecar[key] != expected:
            raise M8ManifestError(f"{label} source sidecar {key} claim does not match")
    if _range(sidecar["requested_range_ns"], f"{label} requested range") != day_range:
        raise M8ManifestError(f"{label} source sidecar does not claim the full UTC day")
    sidecar_checksum = _object(sidecar["checksum"], f"{label} source sidecar checksum")
    _exact_keys(sidecar_checksum, _CHECKSUM_KEYS, f"{label} source sidecar checksum")
    if (
        sidecar_checksum["algorithm"] != "sha256"
        or _digest(sidecar_checksum["value"], f"{label} source sidecar checksum value")
        != checksum_sha
    ):
        raise M8ManifestError(f"{label} source sidecar checksum claim does not match")
    if sidecar["upstream_checksum_sha256"] is not None:
        raise M8ManifestError(f"{label} source sidecar must not claim an upstream checksum")
    if _utc_ns_from_iso(sidecar["downloaded_at_utc"], f"{label} download timestamp") < 1:
        raise M8ManifestError(f"{label} download timestamp must be after the Unix epoch")
    response_headers = _object(sidecar["response_headers"], f"{label} response headers")
    if not all(type(header_value) is str for header_value in response_headers.values()):
        raise M8ManifestError(f"{label} response headers must be strings")
    return (
        checksum_path,
        checksum_sha,
        checksum_bytes,
        source_uri,
        sidecar_path,
        sidecar_sha,
        sidecar_bytes,
    )


def _verify_raw_source_sidecar(
    *,
    config: M8StudyConfig,
    entry_label: str,
    day_range: tuple[int, int],
    symbol: str,
    day: Date,
    zip_path: Path,
    zip_sha256: str,
    zip_bytes: int,
    source_uri: str,
    sidecar_path: Path,
) -> None:
    expected_name = f"{symbol}-aggTrades-{day.isoformat()}.zip"
    if zip_path.name != expected_name:
        raise M8ManifestError(f"{entry_label} raw ZIP filename is not the frozen daily archive")
    _verify_official_daily_uri(
        source_uri,
        symbol=symbol,
        expected_name=expected_name,
        label=f"{entry_label} source URI",
    )

    sidecar = _read_json_object(sidecar_path, f"{entry_label} raw source sidecar")
    _exact_keys(sidecar, _RAW_SIDECAR_KEYS, f"{entry_label} raw source sidecar")
    if sidecar["manifest_version"] != _STORAGE_MANIFEST_VERSION:
        raise M8ManifestError(f"{entry_label} raw source sidecar version is unsupported")
    if sidecar["artifact_kind"] != "raw_source":
        raise M8ManifestError(f"{entry_label} raw source sidecar kind is invalid")
    if sidecar["source"] != config.study.source or sidecar["source_uri"] != source_uri:
        raise M8ManifestError(f"{entry_label} raw source sidecar lineage does not match")
    if _range(sidecar["requested_range_ns"], f"{entry_label} raw requested range") != day_range:
        raise M8ManifestError(f"{entry_label} raw source sidecar is not a complete UTC day")
    checksum = _object(sidecar["checksum"], f"{entry_label} raw checksum")
    _exact_keys(checksum, _CHECKSUM_KEYS, f"{entry_label} raw checksum")
    if (
        checksum.get("algorithm") != "sha256"
        or _digest(checksum.get("value"), f"{entry_label} raw checksum value") != zip_sha256
    ):
        raise M8ManifestError(f"{entry_label} raw source checksum claim does not match")
    if (
        _digest(
            sidecar["upstream_checksum_sha256"],
            f"{entry_label} upstream checksum",
        )
        != zip_sha256
    ):
        raise M8ManifestError(f"{entry_label} official upstream checksum does not match")
    if _integer(sidecar["bytes"], f"{entry_label} raw sidecar bytes", minimum=1) != zip_bytes:
        raise M8ManifestError(f"{entry_label} raw source byte claim does not match")
    sidecar_filename = _text(sidecar["path"], f"{entry_label} raw sidecar path")
    if Path(sidecar_filename).name != sidecar_filename or sidecar_filename != expected_name:
        raise M8ManifestError(f"{entry_label} raw source sidecar path does not match")
    if _utc_ns_from_iso(sidecar["downloaded_at_utc"], f"{entry_label} download timestamp") < 1:
        raise M8ManifestError(f"{entry_label} download timestamp must be after the Unix epoch")
    response_headers = _object(sidecar["response_headers"], f"{entry_label} response headers")
    if not all(type(header_value) is str for header_value in response_headers.values()):
        raise M8ManifestError(f"{entry_label} response headers must be strings")


def _verify_zip_archive(
    *,
    config: M8StudyConfig,
    entry_label: str,
    symbol: str,
    day: Date,
    zip_path: Path,
    uncompressed_bytes: int,
) -> None:
    expected_member = f"{symbol}-aggTrades-{day.isoformat()}.csv"
    try:
        _preflight_zip_central_directory(
            zip_path,
            entry_label=entry_label,
            expected_member=expected_member,
            expected_uncompressed_bytes=uncompressed_bytes,
            max_uncompressed_bytes=config.study.max_archive_uncompressed_bytes,
        )
        with ZipFile(zip_path) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise M8ManifestError(f"{entry_label} archive must contain exactly one CSV member")
            member = members[0]
            member_path = Path(member.filename)
            if (
                member.is_dir()
                or member.filename != expected_member
                or member_path.is_absolute()
                or ".." in member_path.parts
            ):
                raise M8ManifestError(f"{entry_label} archive member is not the frozen daily CSV")
            if member.flag_bits & 0x1:
                raise M8ManifestError(f"{entry_label} archive member must not be encrypted")
            if member.file_size != uncompressed_bytes:
                raise M8ManifestError(
                    f"{entry_label} expanded-byte claim does not match ZIP metadata"
                )
            if member.file_size > config.study.max_archive_uncompressed_bytes:
                raise M8ManifestError(
                    f"{entry_label} archive exceeds the frozen expanded-byte ceiling"
                )
    except M8ManifestError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise M8ManifestError(
            f"{entry_label} raw archive is not a valid bounded ZIP: {exc}"
        ) from exc


def _zip_extra_field_ids(extra: bytes, *, entry_label: str) -> tuple[int, ...]:
    """Parse bounded central-directory extra fields without opening a ZIP member."""

    field_ids: list[int] = []
    offset = 0
    while offset < len(extra):
        if len(extra) - offset < 4:
            raise M8ManifestError(
                f"{entry_label} archive central-directory extra field is truncated"
            )
        field_id, field_bytes = struct.unpack_from("<HH", extra, offset)
        offset += 4
        if field_bytes > len(extra) - offset:
            raise M8ManifestError(
                f"{entry_label} archive central-directory extra field exceeds its bounds"
            )
        field_ids.append(field_id)
        offset += field_bytes
    return tuple(field_ids)


def _preflight_zip_central_directory(
    path: Path,
    *,
    entry_label: str,
    expected_member: str,
    expected_uncompressed_bytes: int,
    max_uncompressed_bytes: int,
) -> None:
    """Bound and authenticate the one-entry ZIP directory before ``ZipFile`` parses it.

    Only the EOCD record and the declared central-directory bytes are read.  In
    particular, this preflight never seeks to or decompresses the CSV member.
    ZIP64 is deliberately unsupported because the frozen daily-archive limits
    fit in the classic ZIP fields and accepting ZIP64 would add an unnecessary
    parser surface before the economic-data boundary.
    """

    try:
        size = path.stat().st_size
        if size < _ZIP_EOCD_STRUCT.size:
            raise M8ManifestError(f"{entry_label} archive is too small to contain a ZIP EOCD")
        tail_bytes = min(size, _MAX_ZIP_TAIL_BYTES)
        with path.open("rb") as source:
            source.seek(size - tail_bytes)
            tail = source.read(tail_bytes)
        relative_eocd = tail.rfind(_ZIP_EOCD_SIGNATURE)
        if relative_eocd < 0 or len(tail) - relative_eocd < _ZIP_EOCD_STRUCT.size:
            raise M8ManifestError(f"{entry_label} archive ZIP EOCD is missing or truncated")
        (
            _signature,
            disk_number,
            central_disk,
            entries_on_disk,
            entries_total,
            central_bytes,
            central_offset,
            comment_bytes,
        ) = _ZIP_EOCD_STRUCT.unpack_from(tail, relative_eocd)
        eocd_offset = size - tail_bytes + relative_eocd
        if eocd_offset + _ZIP_EOCD_STRUCT.size + comment_bytes != size:
            raise M8ManifestError(
                f"{entry_label} archive ZIP EOCD has trailing or malformed bounds"
            )
        if (
            disk_number == _ZIP16_SENTINEL
            or central_disk == _ZIP16_SENTINEL
            or entries_on_disk == _ZIP16_SENTINEL
            or entries_total == _ZIP16_SENTINEL
            or central_bytes == _ZIP32_SENTINEL
            or central_offset == _ZIP32_SENTINEL
        ):
            raise M8ManifestError(f"{entry_label} archive ZIP64 is not permitted")
        if disk_number != 0 or central_disk != 0:
            raise M8ManifestError(f"{entry_label} archive must be a single-disk ZIP")
        if entries_on_disk != 1 or entries_total != 1:
            raise M8ManifestError(
                f"{entry_label} archive must contain exactly one central-directory member"
            )
        if central_bytes < _ZIP_CENTRAL_STRUCT.size:
            raise M8ManifestError(
                f"{entry_label} archive central directory is smaller than one entry"
            )
        if central_bytes > _MAX_ZIP_DIRECTORY_BYTES:
            raise M8ManifestError(
                f"{entry_label} archive central directory exceeds its metadata byte ceiling"
            )
        if central_offset + central_bytes != eocd_offset:
            raise M8ManifestError(f"{entry_label} archive central-directory bounds are malformed")

        with path.open("rb") as source:
            source.seek(central_offset)
            directory = source.read(central_bytes)
        if len(directory) != central_bytes:
            raise M8ManifestError(f"{entry_label} archive central directory is truncated")
        (
            signature,
            _version_made,
            _version_needed,
            flags,
            compression,
            _modified_time,
            _modified_date,
            _crc32,
            compressed_bytes,
            uncompressed_bytes,
            filename_bytes,
            extra_bytes,
            member_comment_bytes,
            member_disk,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = _ZIP_CENTRAL_STRUCT.unpack_from(directory)
        if signature != _ZIP_CENTRAL_SIGNATURE:
            raise M8ManifestError(f"{entry_label} archive central-directory signature is invalid")
        if (
            compressed_bytes == _ZIP32_SENTINEL
            or uncompressed_bytes == _ZIP32_SENTINEL
            or local_header_offset == _ZIP32_SENTINEL
            or member_disk == _ZIP16_SENTINEL
        ):
            raise M8ManifestError(f"{entry_label} archive ZIP64 entry is not permitted")
        if member_disk != 0:
            raise M8ManifestError(f"{entry_label} archive member starts on another disk")
        declared_entry_bytes = (
            _ZIP_CENTRAL_STRUCT.size + filename_bytes + extra_bytes + member_comment_bytes
        )
        if declared_entry_bytes != central_bytes:
            raise M8ManifestError(
                f"{entry_label} archive central directory does not contain exactly one entry"
            )
        filename_start = _ZIP_CENTRAL_STRUCT.size
        filename_end = filename_start + filename_bytes
        extra_end = filename_end + extra_bytes
        filename = directory[filename_start:filename_end]
        extra = directory[filename_end:extra_end]
        try:
            decoded_filename = filename.decode("ascii")
        except UnicodeDecodeError as exc:
            raise M8ManifestError(
                f"{entry_label} archive central member name is not ASCII"
            ) from exc
        if decoded_filename != expected_member:
            raise M8ManifestError(
                f"{entry_label} archive central member is not the frozen daily CSV"
            )
        if _ZIP64_EXTRA_FIELD_ID in _zip_extra_field_ids(extra, entry_label=entry_label):
            raise M8ManifestError(f"{entry_label} archive ZIP64 extra field is not permitted")
        if flags & 0x1:
            raise M8ManifestError(f"{entry_label} archive member must not be encrypted")
        if compression not in {0, 8}:
            raise M8ManifestError(f"{entry_label} archive member uses unsupported compression")
        if local_header_offset >= central_offset:
            raise M8ManifestError(
                f"{entry_label} archive local-header offset is outside the payload region"
            )
        if uncompressed_bytes != expected_uncompressed_bytes:
            raise M8ManifestError(
                f"{entry_label} expanded-byte claim does not match ZIP central metadata"
            )
        if uncompressed_bytes < 1 or uncompressed_bytes > max_uncompressed_bytes:
            raise M8ManifestError(f"{entry_label} archive exceeds the frozen expanded-byte ceiling")
    except M8ManifestError:
        raise
    except (OSError, struct.error, ValueError) as exc:
        raise M8ManifestError(
            f"{entry_label} raw archive has invalid bounded ZIP metadata: {exc}"
        ) from exc


def _verify_quality(
    *,
    entry_label: str,
    rows: int,
    quality: Mapping[str, Any],
    root: Path,
) -> tuple[Path, str, int, Path, str, int, int, int]:
    _exact_keys(quality, _QUALITY_KEYS, f"{entry_label}.quality")
    report_sha = _digest(quality["report_sha256"], f"{entry_label} quality report SHA")
    report_bytes = _integer(
        quality["report_bytes"], f"{entry_label} quality report bytes", minimum=1
    )
    report_path = _declared_file(root, quality["report_path"], f"{entry_label} quality report")
    _verify_file(report_path, report_sha, report_bytes, f"{entry_label} quality report")
    findings_sha = _digest(quality["findings_sha256"], f"{entry_label} quality findings SHA")
    findings_bytes = _integer(
        quality["findings_bytes"], f"{entry_label} quality findings bytes", minimum=0
    )
    findings_path = _declared_file(
        root, quality["findings_path"], f"{entry_label} quality findings"
    )
    _verify_file(findings_path, findings_sha, findings_bytes, f"{entry_label} quality findings")
    errors = _integer(quality["errors"], f"{entry_label} quality errors", minimum=0)
    warnings = _integer(quality["warnings"], f"{entry_label} quality warnings", minimum=0)
    if errors != 0 or warnings != 0:
        raise M8ManifestError(f"{entry_label} has quality findings and is not complete evidence")
    if findings_bytes != 0:
        raise M8ManifestError(f"{entry_label} zero-finding JSONL must be empty")

    report = _read_json_object(report_path, f"{entry_label} quality report")
    if report.get("dataset") != "trades":
        raise M8ManifestError(f"{entry_label} quality report dataset is not trades")
    if _integer(report.get("rows_checked"), f"{entry_label} rows checked", minimum=1) != rows:
        raise M8ManifestError(f"{entry_label} quality report row count does not match")
    summary = _object(report.get("summary"), f"{entry_label} quality summary")
    if _integer(summary.get("errors"), f"{entry_label} report errors", minimum=0) != errors:
        raise M8ManifestError(f"{entry_label} quality error count does not match")
    if _integer(summary.get("warnings"), f"{entry_label} report warnings", minimum=0) != warnings:
        raise M8ManifestError(f"{entry_label} quality warning count does not match")
    retained = _array(report.get("findings"), f"{entry_label} retained findings")
    if retained:
        raise M8ManifestError(f"{entry_label} zero-finding quality report retained findings")
    if report.get("mutation_policy") != "observations were not changed or repaired":
        raise M8ManifestError(f"{entry_label} quality report mutation policy is invalid")
    return (
        report_path,
        report_sha,
        report_bytes,
        findings_path,
        findings_sha,
        findings_bytes,
        errors,
        warnings,
    )


def _verify_part(
    *,
    config: M8StudyConfig,
    entry_label: str,
    part_index: int,
    part_value: object,
    dataset_artifact_value: object,
    root: Path,
    dataset_root: Path,
    symbol: str,
    day: Date,
    day_range: tuple[int, int],
    source_uri: str,
    raw_sha256: str,
) -> M8NormalizedPart:
    label = f"{entry_label} normalized part {part_index}"
    part = _object(part_value, label)
    _exact_keys(part, _PART_KEYS, label)
    dataset_artifact = _object(dataset_artifact_value, f"{label} dataset descriptor")
    _exact_keys(dataset_artifact, _DATASET_PART_KEYS, f"{label} dataset descriptor")

    ordinal = _integer(part["write_ordinal"], f"{label}.write_ordinal", minimum=0)
    if (
        ordinal != part_index
        or _integer(
            dataset_artifact["write_ordinal"],
            f"{label} dataset write ordinal",
            minimum=0,
        )
        != ordinal
    ):
        raise M8ManifestError(f"{label} write ordinals are not contiguous and ordered")
    rows = _integer(part["rows"], f"{label}.rows", minimum=1)
    if _integer(dataset_artifact["rows"], f"{label} dataset rows", minimum=1) != rows:
        raise M8ManifestError(f"{label} dataset row count does not match")

    data_sha = _digest(part["data_sha256"], f"{label} data SHA")
    sidecar_sha = _digest(part["sidecar_sha256"], f"{label} sidecar SHA")
    if _digest(dataset_artifact["data_sha256"], f"{label} dataset data SHA") != data_sha:
        raise M8ManifestError(f"{label} dataset data checksum does not match")
    if _digest(dataset_artifact["manifest_sha256"], f"{label} dataset sidecar SHA") != sidecar_sha:
        raise M8ManifestError(f"{label} dataset sidecar checksum does not match")

    data_bytes = _integer(part["data_bytes"], f"{label} data bytes", minimum=1)
    sidecar_bytes = _integer(part["sidecar_bytes"], f"{label} sidecar bytes", minimum=1)
    data_path = _declared_file(root, part["data_path"], f"{label} data")
    sidecar_path = _declared_file(root, part["sidecar_path"], f"{label} sidecar")
    nested_data_path = _declared_file(
        dataset_root,
        dataset_artifact["data_path"],
        f"{label} dataset data path",
    )
    nested_sidecar_path = _declared_file(
        dataset_root,
        dataset_artifact["manifest_path"],
        f"{label} dataset sidecar path",
    )
    if data_path != nested_data_path or sidecar_path != nested_sidecar_path:
        raise M8ManifestError(f"{label} explicit paths do not match the dataset manifest")
    _verify_file(data_path, data_sha, data_bytes, f"{label} data")
    _verify_file(sidecar_path, sidecar_sha, sidecar_bytes, f"{label} sidecar")

    observed = _observed_range(part["observed_range_ns"], f"{label} observed range")
    dataset_observed = _observed_range(
        dataset_artifact["observed_range_ns"], f"{label} dataset observed range"
    )
    if observed != dataset_observed:
        raise M8ManifestError(f"{label} observed ranges do not match")
    _verify_day_range(observed, day_range, f"{label} observed range")

    sidecar = _read_json_object(sidecar_path, f"{label} sidecar")
    _exact_keys(sidecar, _PART_SIDECAR_KEYS, f"{label} sidecar")
    expected_claims: dict[str, object] = {
        "manifest_version": _STORAGE_MANIFEST_VERSION,
        "artifact_kind": "normalized_parquet",
        "dataset": "trades",
        "schema_name": "trades",
        "schema_version": SCHEMA_VERSION,
        "venue": "binance_spot",
        "symbol": symbol,
        "partition_date": day.isoformat(),
        "write_ordinal": ordinal,
        "source": config.study.source,
        "source_uri": source_uri,
        "source_checksum_sha256": raw_sha256,
        "rows": rows,
        "bytes": data_bytes,
    }
    for key, expected in expected_claims.items():
        if sidecar[key] != expected:
            raise M8ManifestError(f"{label} sidecar {key} claim does not match")
    if _range(sidecar["requested_range_ns"], f"{label} sidecar requested range") != day_range:
        raise M8ManifestError(f"{label} sidecar requested range does not match")
    if _observed_range(sidecar["observed_range_ns"], f"{label} sidecar observed") != observed:
        raise M8ManifestError(f"{label} sidecar observed range does not match")
    checksum = _object(sidecar["checksum"], f"{label} sidecar checksum")
    if (
        checksum.get("algorithm") != "sha256"
        or _digest(checksum.get("value"), f"{label} sidecar checksum value") != data_sha
    ):
        raise M8ManifestError(f"{label} sidecar data checksum does not match")
    expected_nested_data = _text(dataset_artifact["data_path"], f"{label} nested data path")
    if sidecar["path"] != expected_nested_data:
        raise M8ManifestError(f"{label} sidecar path claim does not match")

    try:
        parquet = pq.ParquetFile(data_path)
        if parquet.metadata.num_rows != rows:
            raise M8ManifestError(f"{label} Parquet footer row count does not match")
        if not parquet.schema_arrow.equals(get_schema("trades"), check_metadata=True):
            raise M8ManifestError(f"{label} Parquet schema is not normalized trades")
    except M8ManifestError:
        raise
    except (OSError, pa.ArrowException, ValueError) as exc:
        raise M8ManifestError(f"cannot inspect {label} Parquet footer: {exc}") from exc

    return M8NormalizedPart(
        data_path=data_path,
        data_sha256=data_sha,
        data_bytes=data_bytes,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_sha,
        sidecar_bytes=sidecar_bytes,
        rows=rows,
        write_ordinal=ordinal,
        observed_start_ns=observed[0],
        observed_end_inclusive_ns=observed[1],
    )


def _verify_normalized(
    *,
    config: M8StudyConfig,
    entry_label: str,
    normalized: Mapping[str, Any],
    root: Path,
    symbol: str,
    day: Date,
    day_range: tuple[int, int],
    entry_rows: int,
    entry_observed: tuple[int, int],
    source_uri: str,
    raw_sha256: str,
) -> tuple[Path, str, int, tuple[M8NormalizedPart, ...]]:
    _exact_keys(normalized, _NORMALIZED_KEYS, f"{entry_label}.normalized")
    dataset_sha = _digest(
        normalized["dataset_manifest_sha256"], f"{entry_label} dataset manifest SHA"
    )
    dataset_bytes = _integer(
        normalized["dataset_manifest_bytes"],
        f"{entry_label} dataset manifest bytes",
        minimum=1,
    )
    dataset_path = _declared_file(
        root,
        normalized["dataset_manifest_path"],
        f"{entry_label} normalized dataset manifest",
    )
    _verify_file(
        dataset_path,
        dataset_sha,
        dataset_bytes,
        f"{entry_label} normalized dataset manifest",
    )
    if dataset_path.parent.name != "_manifests":
        raise M8ManifestError(f"{entry_label} dataset manifest is not under _manifests")
    dataset_root = dataset_path.parent.parent.resolve()
    if not dataset_root.is_relative_to(root.resolve()):
        raise M8ManifestError(f"{entry_label} dataset root escapes the M8 input root")

    declared_rows = _integer(normalized["rows"], f"{entry_label} normalized rows", minimum=1)
    if declared_rows != entry_rows:
        raise M8ManifestError(f"{entry_label} normalized row count does not match")
    top_parts = _array(normalized["parts"], f"{entry_label} normalized parts")
    if not top_parts:
        raise M8ManifestError(f"{entry_label} must declare at least one normalized part")

    dataset = _read_json_object(dataset_path, f"{entry_label} dataset manifest")
    _exact_keys(dataset, _DATASET_KEYS, f"{entry_label} dataset manifest")
    expected_claims: dict[str, object] = {
        "manifest_version": _STORAGE_MANIFEST_VERSION,
        "dataset": "trades",
        "schema_version": SCHEMA_VERSION,
        "source": config.study.source,
        "source_uri": source_uri,
        "rows": entry_rows,
    }
    for key, expected in expected_claims.items():
        if dataset[key] != expected:
            raise M8ManifestError(f"{entry_label} dataset manifest {key} claim does not match")
    if _range(dataset["requested_range_ns"], f"{entry_label} dataset requested range") != day_range:
        raise M8ManifestError(f"{entry_label} dataset requested range does not match")
    dataset_parts = _array(dataset["artifacts"], f"{entry_label} dataset artifacts")
    if len(dataset_parts) != len(top_parts):
        raise M8ManifestError(f"{entry_label} dataset part count does not match")

    parts = tuple(
        _verify_part(
            config=config,
            entry_label=entry_label,
            part_index=index,
            part_value=top_part,
            dataset_artifact_value=dataset_part,
            root=root,
            dataset_root=dataset_root,
            symbol=symbol,
            day=day,
            day_range=day_range,
            source_uri=source_uri,
            raw_sha256=raw_sha256,
        )
        for index, (top_part, dataset_part) in enumerate(zip(top_parts, dataset_parts, strict=True))
    )
    if len({part.data_path for part in parts}) != len(parts):
        raise M8ManifestError(f"{entry_label} contains duplicate normalized part paths")
    if len({part.sidecar_path for part in parts}) != len(parts):
        raise M8ManifestError(f"{entry_label} contains duplicate normalized sidecar paths")
    if sum(part.rows for part in parts) != entry_rows:
        raise M8ManifestError(f"{entry_label} normalized part rows do not sum to entry rows")
    aggregate_observed = (
        min(part.observed_start_ns for part in parts),
        max(part.observed_end_inclusive_ns for part in parts),
    )
    if aggregate_observed != entry_observed:
        raise M8ManifestError(f"{entry_label} normalized part bounds do not match entry bounds")
    return dataset_path, dataset_sha, dataset_bytes, parts


def _verify_entry(
    config: M8StudyConfig,
    root: Path,
    value: object,
    expected: tuple[str, Date, M8PeriodRole],
    symbol_metadata: M8SymbolMetadata,
    index: int,
) -> M8ArchiveEntry:
    label = f"entries[{index}]"
    entry = _object(value, label)
    _exact_keys(entry, _ENTRY_KEYS, label)
    symbol = _text(entry["symbol"], f"{label}.symbol")
    day = _parse_date(entry["date"], f"{label}.date")
    role = _parse_role(entry["role"], f"{label}.role")
    if (symbol, day, role) != expected:
        raise M8ManifestError(
            f"{label} is out of frozen order or has an unexpected symbol/date/role"
        )
    if not _boolean(entry["complete"], f"{label}.complete"):
        raise M8ManifestError(f"{label} is not a complete full-day archive")
    day_range = _day_bounds_ns(day)
    if _range(entry["requested_range_ns"], f"{label}.requested_range_ns") != day_range:
        raise M8ManifestError(f"{label} does not claim the exact full UTC day")

    rows = _integer(entry["rows"], f"{label}.rows", minimum=1)
    trade_ids = _object(entry["trade_id_range"], f"{label}.trade_id_range")
    _exact_keys(trade_ids, _TRADE_ID_KEYS, f"{label}.trade_id_range")
    first_trade_id = _integer(trade_ids["first"], f"{label}.first_trade_id", minimum=0)
    last_trade_id = _integer(trade_ids["last"], f"{label}.last_trade_id", minimum=0)
    contiguous_count = _integer(
        trade_ids["contiguous_count"], f"{label}.contiguous_count", minimum=1
    )
    if contiguous_count != rows or last_trade_id - first_trade_id + 1 != rows:
        raise M8ManifestError(f"{label} aggregate-trade IDs are not a contiguous row count")

    observed = _observed_range(entry["observed_range_ns"], f"{label}.observed_range_ns")
    _verify_day_range(observed, day_range, f"{label} observed event bounds")
    scales = _object(entry["scales"], f"{label}.scales")
    _exact_keys(scales, _SCALE_KEYS, f"{label}.scales")
    tick_size = _decimal(scales["tick_size"], f"{label}.tick_size")
    lot_size = _decimal(scales["lot_size"], f"{label}.lot_size")
    if (
        symbol_metadata.symbol != symbol
        or tick_size != symbol_metadata.tick_size
        or lot_size != symbol_metadata.lot_size
    ):
        raise M8ManifestError(
            f"{label} scales do not match verified exchangeInfo metadata for {symbol}"
        )

    raw = _object(entry["raw"], f"{label}.raw")
    _exact_keys(raw, _RAW_KEYS, f"{label}.raw")
    raw_zip_path = _declared_file(root, raw["zip_path"], f"{label} raw ZIP")
    raw_zip_sha = _digest(raw["zip_sha256"], f"{label} raw ZIP SHA")
    raw_zip_bytes = _integer(raw["zip_bytes"], f"{label} raw ZIP bytes", minimum=1)
    raw_uncompressed_bytes = _integer(
        raw["uncompressed_bytes"], f"{label} raw expanded bytes", minimum=1
    )
    if raw_zip_bytes > config.study.max_archive_compressed_bytes:
        raise M8ManifestError(f"{label} raw ZIP exceeds the frozen compressed-byte ceiling")
    if raw_uncompressed_bytes > config.study.max_archive_uncompressed_bytes:
        raise M8ManifestError(f"{label} archive exceeds the frozen expanded-byte ceiling")
    _verify_file(raw_zip_path, raw_zip_sha, raw_zip_bytes, f"{label} raw ZIP")
    _verify_zip_archive(
        config=config,
        entry_label=label,
        symbol=symbol,
        day=day,
        zip_path=raw_zip_path,
        uncompressed_bytes=raw_uncompressed_bytes,
    )
    source_uri = _text(raw["source_uri"], f"{label}.source_uri")
    raw_sidecar_path = _declared_file(
        root, raw["source_manifest_path"], f"{label} raw source sidecar"
    )
    raw_sidecar_sha = _digest(raw["source_manifest_sha256"], f"{label} raw source sidecar SHA")
    raw_sidecar_bytes = _integer(
        raw["source_manifest_bytes"], f"{label} raw source sidecar bytes", minimum=1
    )
    _verify_file(
        raw_sidecar_path,
        raw_sidecar_sha,
        raw_sidecar_bytes,
        f"{label} raw source sidecar",
    )
    _verify_raw_source_sidecar(
        config=config,
        entry_label=label,
        day_range=day_range,
        symbol=symbol,
        day=day,
        zip_path=raw_zip_path,
        zip_sha256=raw_zip_sha,
        zip_bytes=raw_zip_bytes,
        source_uri=source_uri,
        sidecar_path=raw_sidecar_path,
    )
    (
        raw_checksum_path,
        raw_checksum_sha,
        raw_checksum_bytes,
        raw_checksum_source_uri,
        raw_checksum_sidecar_path,
        raw_checksum_sidecar_sha,
        raw_checksum_sidecar_bytes,
    ) = _verify_official_checksum(
        entry_label=label,
        root=root,
        value=raw["checksum"],
        symbol=symbol,
        day=day,
        day_range=day_range,
        zip_path=raw_zip_path,
        zip_sha256=raw_zip_sha,
        archive_source_uri=source_uri,
        archive_sidecar_path=raw_sidecar_path,
    )

    normalized = _object(entry["normalized"], f"{label}.normalized")
    dataset_path, dataset_sha, dataset_bytes, parts = _verify_normalized(
        config=config,
        entry_label=label,
        normalized=normalized,
        root=root,
        symbol=symbol,
        day=day,
        day_range=day_range,
        entry_rows=rows,
        entry_observed=observed,
        source_uri=source_uri,
        raw_sha256=raw_zip_sha,
    )
    quality = _object(entry["quality"], f"{label}.quality")
    (
        quality_report_path,
        quality_report_sha,
        quality_report_bytes,
        quality_findings_path,
        quality_findings_sha,
        quality_findings_bytes,
        quality_errors,
        quality_warnings,
    ) = _verify_quality(entry_label=label, rows=rows, quality=quality, root=root)

    return M8ArchiveEntry(
        symbol=symbol,
        date=day,
        role=role,
        complete=True,
        rows=rows,
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        observed_start_ns=observed[0],
        observed_end_inclusive_ns=observed[1],
        tick_size=tick_size,
        lot_size=lot_size,
        raw_zip_path=raw_zip_path,
        raw_zip_sha256=raw_zip_sha,
        raw_zip_bytes=raw_zip_bytes,
        raw_uncompressed_bytes=raw_uncompressed_bytes,
        raw_source_uri=source_uri,
        raw_source_manifest_path=raw_sidecar_path,
        raw_source_manifest_sha256=raw_sidecar_sha,
        raw_source_manifest_bytes=raw_sidecar_bytes,
        raw_checksum_path=raw_checksum_path,
        raw_checksum_sha256=raw_checksum_sha,
        raw_checksum_bytes=raw_checksum_bytes,
        raw_checksum_source_uri=raw_checksum_source_uri,
        raw_checksum_source_manifest_path=raw_checksum_sidecar_path,
        raw_checksum_source_manifest_sha256=raw_checksum_sidecar_sha,
        raw_checksum_source_manifest_bytes=raw_checksum_sidecar_bytes,
        normalized_dataset_manifest_path=dataset_path,
        normalized_dataset_manifest_sha256=dataset_sha,
        normalized_dataset_manifest_bytes=dataset_bytes,
        normalized_parts=parts,
        quality_report_path=quality_report_path,
        quality_report_sha256=quality_report_sha,
        quality_report_bytes=quality_report_bytes,
        quality_findings_path=quality_findings_path,
        quality_findings_sha256=quality_findings_sha,
        quality_findings_bytes=quality_findings_bytes,
        quality_errors=quality_errors,
        quality_warnings=quality_warnings,
    )


def _verify_payload(
    config: M8StudyConfig, root: Path, payload: Mapping[str, Any]
) -> tuple[tuple[M8SymbolMetadata, ...], tuple[M8ArchiveEntry, ...]]:
    _exact_keys(payload, _TOP_KEYS, "M8 input manifest")
    if payload["manifest_version"] != M8_INPUT_MANIFEST_VERSION:
        raise M8ManifestError("unsupported M8 input manifest version")
    if payload["artifact_kind"] != "m8_multidate_trade_input":
        raise M8ManifestError("unexpected M8 input manifest artifact kind")

    config_claim = _object(payload["config"], "M8 input manifest config")
    _exact_keys(config_claim, _CONFIG_KEYS, "M8 input manifest config")
    expected_config = {
        "semantic_sha256": config.hash,
        "source_sha256": config.source_sha256,
        "protocol_version": config.study.protocol_version,
    }
    if dict(config_claim) != expected_config:
        raise M8ManifestError("M8 input manifest is bound to a different frozen configuration")

    study = _object(payload["study"], "M8 input manifest study")
    _exact_keys(study, _STUDY_KEYS, "M8 input manifest study")
    expected_periods = [
        {"date": period.date.isoformat(), "role": period.role} for period in config.periods
    ]
    observed_periods = _array(study["periods"], "M8 input manifest study periods")
    for index, raw_period in enumerate(observed_periods):
        period = _object(raw_period, f"M8 input manifest study periods[{index}]")
        _exact_keys(period, _STUDY_PERIOD_KEYS, f"M8 input manifest study periods[{index}]")
    expected_study: dict[str, object] = {
        "name": config.study.name,
        "evidence_tier": config.study.evidence_tier,
        "source": config.study.source,
        "symbols": list(config.study.symbols),
        "periods": expected_periods,
    }
    if dict(study) != expected_study:
        raise M8ManifestError("M8 input manifest study scope differs from the frozen protocol")

    raw_symbol_metadata = _array(payload["symbol_metadata"], "M8 input manifest symbol metadata")
    if len(raw_symbol_metadata) != len(config.study.symbols) or len(raw_symbol_metadata) != 2:
        raise M8ManifestError("M8 input manifest must contain exactly two symbol metadata entries")
    symbol_metadata = tuple(
        _verify_symbol_metadata(root, value, expected_symbol, index)
        for index, (value, expected_symbol) in enumerate(
            zip(raw_symbol_metadata, config.study.symbols, strict=True)
        )
    )
    if len({metadata.symbol for metadata in symbol_metadata}) != len(symbol_metadata):
        raise M8ManifestError("M8 input manifest contains duplicate symbol metadata")
    if len({metadata.raw_path for metadata in symbol_metadata}) != len(symbol_metadata):
        raise M8ManifestError("M8 input manifest reuses an exchangeInfo raw body")
    if len({metadata.source_manifest_path for metadata in symbol_metadata}) != len(symbol_metadata):
        raise M8ManifestError("M8 input manifest reuses an exchangeInfo source sidecar")
    total_symbol_metadata_bytes = _integer(
        payload["total_symbol_metadata_bytes"],
        "M8 total symbol metadata bytes",
        minimum=1,
    )
    observed_metadata_total = sum(metadata.raw_bytes for metadata in symbol_metadata)
    if total_symbol_metadata_bytes != observed_metadata_total:
        raise M8ManifestError("M8 total symbol metadata byte claim does not match")
    if total_symbol_metadata_bytes > len(symbol_metadata) * _MAX_SYMBOL_METADATA_BYTES:
        raise M8ManifestError("M8 symbol metadata exceeds its bounded total byte ceiling")
    metadata_by_symbol = {metadata.symbol: metadata for metadata in symbol_metadata}

    raw_entries = _array(payload["entries"], "M8 input manifest entries")
    expected_order = _expected_entry_order(config)
    if len(raw_entries) != len(expected_order) or len(raw_entries) != 8:
        raise M8ManifestError("M8 input manifest must contain exactly eight entries")
    entries = tuple(
        _verify_entry(
            config,
            root,
            value,
            expected,
            metadata_by_symbol[expected[0]],
            index,
        )
        for index, (value, expected) in enumerate(zip(raw_entries, expected_order, strict=True))
    )
    if len({(entry.symbol, entry.date) for entry in entries}) != len(entries):
        raise M8ManifestError("M8 input manifest contains duplicate symbol/date entries")
    total_raw_zip_bytes = _integer(
        payload["total_raw_zip_bytes"], "M8 total raw ZIP bytes", minimum=1
    )
    observed_total = sum(entry.raw_zip_bytes for entry in entries)
    if total_raw_zip_bytes != observed_total:
        raise M8ManifestError("M8 total raw ZIP byte claim does not match its entries")
    if total_raw_zip_bytes > config.study.max_total_download_bytes:
        raise M8ManifestError("M8 raw archives exceed the frozen total-download ceiling")
    return symbol_metadata, entries


def _encoded_manifest(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_m8_input_manifest(
    config: M8StudyConfig,
    root: str | Path,
    entries: Sequence[M8ArchiveEntry],
    symbol_metadata: Sequence[M8SymbolMetadata],
    *,
    output_dir: str | Path | None = None,
) -> M8InputManifest:
    """Validate and atomically publish one deterministic content-addressed manifest."""

    input_root = Path(root).resolve()
    if not input_root.is_dir():
        raise M8ManifestError(f"M8 input root does not exist: {input_root}")
    payload = _manifest_payload(config, input_root, entries, symbol_metadata)
    _verify_payload(config, input_root, payload)
    encoded = _encoded_manifest(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    directory = (
        (input_root / "_manifests").resolve()
        if output_dir is None
        else (
            Path(output_dir) if Path(output_dir).is_absolute() else input_root / output_dir
        ).resolve()
    )
    if not directory.is_relative_to(input_root):
        raise M8ManifestError("M8 manifest output directory escapes the input root")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"m8-input.manifest-{digest[:20]}.json"
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise M8ManifestError(f"immutable M8 manifest collision at {destination}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=directory,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return read_m8_input_manifest(
        config,
        input_root,
        destination,
        manifest_sha256=digest,
    )


def read_m8_input_manifest(
    config: M8StudyConfig,
    root: str | Path,
    manifest_path: str | Path,
    *,
    manifest_sha256: str,
) -> M8InputManifest:
    """Read and fully verify a manifest and every declared evidence artifact."""

    input_root = Path(root).resolve()
    if not input_root.is_dir():
        raise M8ManifestError(f"M8 input root does not exist: {input_root}")
    expected_sha = _digest(manifest_sha256, "M8 input manifest SHA")
    path = _resolve_contained_file(input_root, Path(manifest_path), "M8 input manifest")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise M8ManifestError(
            f"M8 input manifest SHA-256 mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    expected_name = f"m8-input.manifest-{expected_sha[:20]}.json"
    if path.name != expected_name:
        raise M8ManifestError("M8 input manifest filename is not content-addressed")
    payload = _read_json_object(path, "M8 input manifest")
    symbol_metadata, entries = _verify_payload(config, input_root, payload)
    return M8InputManifest(
        root=input_root,
        path=path,
        sha256=expected_sha,
        config_sha256=config.hash,
        config_source_sha256=config.source_sha256,
        protocol_version=config.study.protocol_version,
        symbol_metadata=symbol_metadata,
        entries=entries,
    )


def verify_m8_input_manifest(
    config: M8StudyConfig,
    root: str | Path,
    manifest_path: str | Path,
    *,
    manifest_sha256: str,
) -> M8InputManifest:
    """Alias with an explicit verification name for CLI/pipeline call sites."""

    return read_m8_input_manifest(
        config,
        root,
        manifest_path,
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "M8_INPUT_MANIFEST_VERSION",
    "M8ArchiveEntry",
    "M8InputManifest",
    "M8ManifestError",
    "M8NormalizedPart",
    "M8SymbolMetadata",
    "read_m8_input_manifest",
    "verify_m8_input_manifest",
    "write_m8_input_manifest",
]
