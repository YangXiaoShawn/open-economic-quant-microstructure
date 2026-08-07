"""Bounded, manifest-anchored loading of normalized public trade data.

The reader deliberately starts from an immutable ingestion manifest rather than
discovering Parquet files in a directory.  It verifies the caller's ingestion
manifest digest, the referenced normalized-data manifest digest, every declared
part and sidecar digest, and the coverage/evidence claims before returning data.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlsplit

import duckdb
import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from microstructure.config import ProjectConfig, datetime_to_ns
from microstructure.data.quality import IncrementalQualityValidator, ValidationReport
from microstructure.data.schemas import SCHEMA_VERSION, ensure_schema, get_schema
from microstructure.data.storage import MANIFEST_VERSION
from microstructure.provenance import read_json, sha256_file

PublicEvidenceTier = Literal["PUBLIC_SAMPLE_PARTIAL", "FULL_DATA"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NS_PER_SECOND = 1_000_000_000
# Matches the downloader's public-response ceiling.  The reader checks this
# before JSON parsing so a recomputed manifest cannot turn one "page" into an
# unbounded-memory payload by inserting giant unused strings.
_MAX_RAW_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_JSON_STRING_BYTES = 64 * 1024
_JSON_BASE_BYTES = 256 * 1024
_JSON_BYTES_PER_STRUCTURE_TOKEN = 32 * 1024
_PARQUET_BASE_BYTES = 1024 * 1024
_PARQUET_BYTES_PER_TRADE_ROW = 4096
_MAX_PARQUET_PART_ENCODED_BYTES = 64 * 1024 * 1024
_MAX_PARQUET_PART_DECODED_BYTES = 128 * 1024 * 1024

__all__ = [
    "ObservedUtcCoverage",
    "PublicDataError",
    "PublicEvidenceTier",
    "PublicTradeDataset",
    "PublicTrades",
    "SymbolObservedCoverage",
    "VerifiedPublicTradeBatchStream",
    "read_public_trades",
    "verify_public_trade_dataset",
]


class PublicDataError(RuntimeError):
    """Raised when a public normalized input cannot be verified safely."""


@dataclass(frozen=True, slots=True)
class ObservedUtcCoverage:
    """Exact inclusive event-time coverage for a bounded set of rows."""

    start_ns: int
    end_inclusive_ns: int
    start_utc: str
    end_inclusive_utc: str


@dataclass(frozen=True, slots=True)
class SymbolObservedCoverage:
    """Actual rows and event-time coverage for one manifested symbol."""

    symbol: str
    rows: int
    complete_range: bool
    tick_size: Decimal
    lot_size: Decimal
    observed: ObservedUtcCoverage


@dataclass(frozen=True, slots=True)
class PublicTrades:
    """Verified public trades in Arrow and Polars representations."""

    arrow_trades: pa.Table
    polars_trades: pl.DataFrame
    observed: ObservedUtcCoverage
    symbols: tuple[SymbolObservedCoverage, ...]
    evidence_tier: PublicEvidenceTier
    all_requested_ranges_complete: bool
    ingestion_manifest_path: Path
    ingestion_manifest_sha256: str
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    part_paths: tuple[Path, ...]
    raw_artifact_paths: tuple[Path, ...]
    raw_manifest_paths: tuple[Path, ...]
    raw_artifact_sha256s: tuple[str, ...]
    validation: ValidationReport
    row_bound: int
    canonical_order: tuple[str, ...]

    @property
    def rows(self) -> int:
        return cast(int, self.arrow_trades.num_rows)

    @property
    def input_manifest_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({self.ingestion_manifest_sha256, self.dataset_manifest_sha256}))


@dataclass(frozen=True, slots=True)
class _ParquetPartDescriptor:
    """Verified metadata needed to stream one normalized Parquet part."""

    data_path: Path
    data_sha256: str
    sidecar_path: Path
    rows: int
    write_ordinal: int
    venue: str
    symbol: str
    partition_date: str
    observed_start_ns: int
    observed_end_inclusive_ns: int


@dataclass(frozen=True, slots=True)
class _RawPageDescriptor:
    """A bounded raw page descriptor; raw records are deliberately not retained."""

    path: Path
    sha256: str
    symbol: str
    from_id: int | None
    rows: int
    first_id: int | None
    last_id: int | None


@dataclass(frozen=True, slots=True)
class _VerifiedStreamSummary:
    validation: ValidationReport
    observed: ObservedUtcCoverage
    symbols: tuple[SymbolObservedCoverage, ...]


class VerifiedPublicTradeBatchStream(Iterator[pa.RecordBatch]):
    """One fresh, bounded validation operation over a public trade data set.

    The final validation and observed-coverage summary become available only
    after the iterator is exhausted.  Closing early releases DuckDB, SQLite,
    and temporary spill resources without claiming that the data were fully
    validated.  One upstream Parquet pass is audited in immutable physical/write
    order and fed directly into DuckDB's bounded external sort; the resulting
    research batches are deterministic without rereading the source parts.
    """

    def __init__(
        self,
        generator: Generator[pa.RecordBatch, None, _VerifiedStreamSummary],
        *,
        fail_on_error: bool,
    ) -> None:
        self._generator = generator
        self._fail_on_error = fail_on_error
        self._summary: _VerifiedStreamSummary | None = None
        self._closed = False

    def __iter__(self) -> VerifiedPublicTradeBatchStream:
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._closed:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration as stop:
            self._closed = True
            self._summary = cast(_VerifiedStreamSummary, stop.value)
            if self._fail_on_error and self._summary.validation.has_errors:
                raise PublicDataError(
                    "public normalized trades failed quality validation with "
                    f"{self._summary.validation.error_count} error findings"
                ) from None
            raise
        except BaseException:
            self._closed = True
            raise

    @property
    def summary(self) -> _VerifiedStreamSummary:
        if self._summary is None:
            raise RuntimeError("verified public batch stream has not been fully consumed")
        return self._summary

    @property
    def validation(self) -> ValidationReport:
        return self.summary.validation

    def close(self) -> None:
        if not self._closed:
            self._generator.close()
            self._closed = True


class _PhysicalAuditBatchSource(Iterator[pa.RecordBatch]):
    """Adapt an audited generator to Arrow's one-input-pass reader API."""

    def __init__(
        self,
        generator: Generator[pa.RecordBatch, None, _VerifiedStreamSummary],
    ) -> None:
        self._generator = generator
        self._summary: _VerifiedStreamSummary | None = None
        self._closed = False

    def __iter__(self) -> _PhysicalAuditBatchSource:
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._closed:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration as stop:
            self._closed = True
            self._summary = cast(_VerifiedStreamSummary, stop.value)
            raise
        except BaseException:
            self._closed = True
            raise

    @property
    def summary(self) -> _VerifiedStreamSummary:
        if self._summary is None:
            raise RuntimeError("physical public-data audit has not been fully consumed")
        return self._summary

    def close(self) -> None:
        if not self._closed:
            self._generator.close()
            self._closed = True


@dataclass(frozen=True, slots=True)
class PublicTradeDataset:
    """Manifest-verified descriptors for a potentially large public data set.

    Construction reads and hashes manifests, sidecars, Parquet footers, and
    bounded raw API pages, but never materializes normalized Parquet rows.
    Every call to :meth:`iter_verified_batches` creates an independent bounded
    operation whose Arrow batch size and DuckDB sort memory are explicit.
    """

    rows: int
    observed: ObservedUtcCoverage
    symbols: tuple[SymbolObservedCoverage, ...]
    evidence_tier: PublicEvidenceTier
    all_requested_ranges_complete: bool
    ingestion_manifest_path: Path
    ingestion_manifest_sha256: str
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    part_paths: tuple[Path, ...]
    raw_artifact_paths: tuple[Path, ...]
    raw_manifest_paths: tuple[Path, ...]
    raw_artifact_sha256s: tuple[str, ...]
    row_bound: int
    canonical_order: tuple[str, ...]
    _config: ProjectConfig
    _requested_range: tuple[int, int]
    _symbol_claims: Mapping[str, _SymbolClaim]
    _parts: tuple[_ParquetPartDescriptor, ...]
    _raw_pages_by_digest: Mapping[str, _RawPageDescriptor]
    _ordered_raw_pages: Mapping[str, tuple[_RawPageDescriptor, ...]]

    @property
    def input_manifest_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({self.ingestion_manifest_sha256, self.dataset_manifest_sha256}))

    def iter_verified_batches(
        self,
        *,
        batch_rows: int = 65_536,
        memory_limit: str = "256MB",
        temp_directory: str | Path | None = None,
    ) -> VerifiedPublicTradeBatchStream:
        """Return a fresh bounded stream in deterministic canonical order."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        if not isinstance(memory_limit, str) or not memory_limit.strip():
            raise ValueError("memory_limit must be a non-empty DuckDB memory size")
        return VerifiedPublicTradeBatchStream(
            _stream_verified_batches(
                self,
                batch_rows=batch_rows,
                memory_limit=memory_limit,
                temp_directory=temp_directory,
            ),
            fail_on_error=self._config.quality.fail_on_error,
        )

    def validate(
        self,
        *,
        batch_rows: int = 65_536,
        memory_limit: str = "256MB",
        temp_directory: str | Path | None = None,
    ) -> ValidationReport:
        """Validate all rows incrementally without retaining normalized data."""
        stream = self.iter_verified_batches(
            batch_rows=batch_rows,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
        )
        try:
            for _ in stream:
                pass
            return stream.validation
        finally:
            stream.close()


@dataclass(frozen=True, slots=True)
class _SymbolClaim:
    rows: int
    complete_range: bool
    tick_size: Decimal
    lot_size: Decimal
    terminal: _TerminalClaim | None


@dataclass(frozen=True, slots=True)
class _TerminalClaim:
    raw_page_count: int
    stop_reason: str
    last_raw_page_sha256: str
    last_path: str
    last_manifest_path: str
    last_request_uri: str
    last_row_count: int


@dataclass(frozen=True, slots=True)
class _VerifiedRawArtifacts:
    paths: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]
    sha256s: frozenset[str]
    pages_by_digest: Mapping[str, _RawPageDescriptor]
    ordered_pages: Mapping[str, tuple[_RawPageDescriptor, ...]]


@dataclass(frozen=True, slots=True)
class _RawAggregateTrade:
    aggregate_id: int
    first_trade_id: int
    last_trade_id: int
    event_ts_ns: int
    price: Decimal
    quantity: Decimal
    buyer_is_maker: bool


@dataclass(frozen=True, slots=True)
class _AggregatePage:
    path: Path
    manifest_path: Path
    sha256: str
    request_uri: str
    downloaded_at_ns: int
    from_id: int | None
    rows: int
    first_id: int | None
    last_id: int | None
    first_event_ts_ns: int | None
    last_event_ts_ns: int | None


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PublicDataError(f"{label} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PublicDataError(f"{label} must be a JSON array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicDataError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicDataError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise PublicDataError(f"{label} must be at least {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise PublicDataError(f"{label} must be a boolean")
    return value


def _digest(value: object, label: str) -> str:
    digest = _text(value, label).lower()
    if _SHA256.fullmatch(digest) is None:
        raise PublicDataError(f"{label} must be a SHA-256 hex digest")
    return digest


def _positive_decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise PublicDataError(f"{label} must be a non-empty decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise PublicDataError(f"{label} is not a valid decimal") from exc
    if not result.is_finite() or result <= 0:
        raise PublicDataError(f"{label} must be positive and finite")
    return result


def _positive_decimal_number(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, float, int, str)):
        raise PublicDataError(f"{label} must be a decimal number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise PublicDataError(f"{label} is not a valid decimal") from exc
    if not result.is_finite() or result <= 0:
        raise PublicDataError(f"{label} must be positive and finite")
    return result


def _scaled_integer(value: Decimal, quantum: Decimal, label: str) -> int:
    scaled = value / quantum
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise PublicDataError(f"{label} is not aligned to exchangeInfo scale")
    return int(integral)


def _unsigned_integer_text(value: str, label: str, *, minimum: int = 0) -> int:
    if not value.isascii() or not value.isdecimal():
        raise PublicDataError(f"{label} must be an unsigned decimal integer")
    result = int(value)
    if result < minimum:
        raise PublicDataError(f"{label} must be at least {minimum}")
    return result


def _load_json_value(path: Path, label: str) -> object:
    _preflight_json_structure(path, label)
    try:
        return cast(object, read_json(path))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PublicDataError(f"cannot read {label} at {path}: {exc}") from exc


def _preflight_json_structure(path: Path, label: str) -> None:
    """Bound JSON token width and bytes relative to its structural cardinality.

    Manifests legitimately grow with O(parts/pages), so a fixed whole-file cap
    would defeat full-history use.  This streaming preflight instead permits
    bytes proportional to JSON structure while rejecting giant padding strings
    before ``read_json`` allocates them.
    """
    total_bytes = 0
    structure_tokens = 0
    current_string_bytes = 0
    in_string = False
    escaped = False
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(64 * 1024):
                total_bytes += len(chunk)
                for byte in chunk:
                    if in_string:
                        current_string_bytes += 1
                        if current_string_bytes > _MAX_JSON_STRING_BYTES:
                            raise PublicDataError(
                                f"{label} has a JSON string above bounded token size "
                                f"{_MAX_JSON_STRING_BYTES} bytes"
                            )
                        if escaped:
                            escaped = False
                        elif byte == 0x5C:  # backslash
                            escaped = True
                        elif byte == 0x22:  # quote
                            in_string = False
                    elif byte == 0x22:
                        in_string = True
                        current_string_bytes = 0
                    elif byte in {0x7B, 0x7D, 0x5B, 0x5D, 0x2C}:  # {}[],
                        structure_tokens += 1
    except PublicDataError:
        raise
    except OSError as exc:
        raise PublicDataError(f"cannot preflight {label} at {path}: {exc}") from exc
    allowed_bytes = _JSON_BASE_BYTES + (structure_tokens * _JSON_BYTES_PER_STRUCTURE_TOKEN)
    if total_bytes > allowed_bytes:
        raise PublicDataError(
            f"{label} has {total_bytes} bytes above its structural JSON bound {allowed_bytes}"
        )


def _request_query(
    source_uri: str,
    *,
    base_url: str,
    endpoint: str,
    label: str,
) -> dict[str, str]:
    """Parse a raw request URI only when it is rooted at the configured API."""
    try:
        configured = urlsplit(base_url)
        requested = urlsplit(source_uri)
        configured_port = configured.port
        requested_port = requested.port
    except ValueError as exc:
        raise PublicDataError(f"{label} is not a valid URL: {exc}") from exc
    if (
        configured.scheme.lower() not in {"http", "https"}
        or configured.hostname is None
        or configured.username is not None
        or configured.password is not None
        or configured.query
        or configured.fragment
    ):
        raise PublicDataError("configured data.base_url is not a plain HTTP(S) base URL")
    if (
        requested.scheme.lower() != configured.scheme.lower()
        or requested.hostname is None
        or requested.hostname.lower() != configured.hostname.lower()
        or requested_port != configured_port
        or requested.username is not None
        or requested.password is not None
    ):
        raise PublicDataError(f"{label} is not bound to configured data.base_url")
    base_path = configured.path.rstrip("/")
    expected_path = f"{base_path}{endpoint}"
    if requested.path != expected_path:
        raise PublicDataError(f"{label} does not use exact endpoint {expected_path!r}")
    if requested.fragment:
        raise PublicDataError(f"{label} must not contain a fragment")
    try:
        pairs = parse_qsl(requested.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise PublicDataError(f"{label} has a malformed query string") from exc
    query: dict[str, str] = {}
    for key, value in pairs:
        if not key or not value:
            raise PublicDataError(f"{label} query keys and values must not be empty")
        if key in query:
            raise PublicDataError(f"{label} has duplicate query parameter {key!r}")
        query[key] = value
    return query


def _validate_exchange_info_payload(
    path: Path,
    *,
    symbol: str,
    claim: _SymbolClaim,
    label: str,
) -> None:
    payload = _object(_load_json_value(path, f"{label} payload"), f"{label} payload")
    symbols = _array(payload.get("symbols"), f"{label} payload.symbols")
    if len(symbols) != 1:
        raise PublicDataError(f"{label} must contain exactly one exchangeInfo symbol")
    item = _object(symbols[0], f"{label} payload.symbols[0]")
    payload_symbol = _text(item.get("symbol"), f"{label} payload.symbol")
    if payload_symbol != symbol:
        raise PublicDataError(f"{label} payload symbol does not match its request URI")
    status = _text(item.get("status"), f"{label} payload.status")
    if status != "TRADING":
        raise PublicDataError(f"{label} payload status is not TRADING")
    _text(item.get("baseAsset"), f"{label} payload.baseAsset")
    _text(item.get("quoteAsset"), f"{label} payload.quoteAsset")

    filters = _array(item.get("filters"), f"{label} payload.filters")
    selected: dict[str, Mapping[str, Any]] = {}
    for filter_index, raw_filter in enumerate(filters):
        filter_item = _object(raw_filter, f"{label} payload.filters[{filter_index}]")
        filter_type = _text(
            filter_item.get("filterType"),
            f"{label} payload.filters[{filter_index}].filterType",
        )
        if filter_type in {"PRICE_FILTER", "LOT_SIZE"}:
            if filter_type in selected:
                raise PublicDataError(f"{label} payload has duplicate {filter_type}")
            selected[filter_type] = filter_item
    if set(selected) != {"PRICE_FILTER", "LOT_SIZE"}:
        raise PublicDataError(f"{label} payload lacks PRICE_FILTER or LOT_SIZE")
    tick_size = _positive_decimal(
        selected["PRICE_FILTER"].get("tickSize"),
        f"{label} payload.PRICE_FILTER.tickSize",
    )
    lot_size = _positive_decimal(
        selected["LOT_SIZE"].get("stepSize"),
        f"{label} payload.LOT_SIZE.stepSize",
    )
    if tick_size != claim.tick_size or lot_size != claim.lot_size:
        raise PublicDataError(f"{label} payload scales do not match ingestion claim for {symbol}")


def _aggregate_records(
    path: Path,
    label: str,
    *,
    request_limit: int,
    requested_range: tuple[int, int],
) -> dict[int, _RawAggregateTrade]:
    # ``requested_range`` is retained in the signature to bind the cache to
    # the verified request.  Binance may legitimately return a terminal
    # sentinel row at/after endTime; normalized rows, not untouched raw pages,
    # are required to fall inside the requested half-open interval.
    del requested_range
    payload = _array(_load_json_value(path, f"{label} payload"), f"{label} payload")
    if len(payload) > request_limit:
        raise PublicDataError(
            f"{label} contains {len(payload)} rows, above configured page limit {request_limit}"
        )
    records: dict[int, _RawAggregateTrade] = {}
    previous_event_ts_ns: int | None = None
    for record_index, raw_record in enumerate(payload):
        record = _object(raw_record, f"{label} payload[{record_index}]")
        aggregate_id = _integer(
            record.get("a"),
            f"{label} payload[{record_index}].a",
            minimum=0,
        )
        if records and aggregate_id <= next(reversed(records)):
            raise PublicDataError(f"{label} aggregate-trade IDs are not strictly increasing")
        first_trade_id = _integer(
            record.get("f"),
            f"{label} payload[{record_index}].f",
            minimum=0,
        )
        last_trade_id = _integer(
            record.get("l"),
            f"{label} payload[{record_index}].l",
            minimum=first_trade_id,
        )
        event_time_ms = _integer(
            record.get("T"),
            f"{label} payload[{record_index}].T",
            minimum=0,
        )
        event_ts_ns = event_time_ms * 1_000_000
        if previous_event_ts_ns is not None and event_ts_ns < previous_event_ts_ns:
            raise PublicDataError(f"{label} aggregate-trade event times are not nondecreasing")
        previous_event_ts_ns = event_ts_ns
        price = _positive_decimal(record.get("p"), f"{label} payload[{record_index}].p")
        quantity = _positive_decimal(record.get("q"), f"{label} payload[{record_index}].q")
        buyer_is_maker = _boolean(
            record.get("m"),
            f"{label} payload[{record_index}].m",
        )
        records[aggregate_id] = _RawAggregateTrade(
            aggregate_id=aggregate_id,
            first_trade_id=first_trade_id,
            last_trade_id=last_trade_id,
            event_ts_ns=event_ts_ns,
            price=price,
            quantity=quantity,
            buyer_is_maker=buyer_is_maker,
        )
    return records


class _RawPageCache:
    """Load at most one bounded aggregate-trade page at a time."""

    def __init__(
        self,
        pages_by_digest: Mapping[str, _RawPageDescriptor],
        *,
        request_limit: int,
        requested_range: tuple[int, int],
    ) -> None:
        self._pages_by_digest = pages_by_digest
        self._request_limit = request_limit
        self._requested_range = requested_range
        self._digest: str | None = None
        self._records: Mapping[int, _RawAggregateTrade] = MappingProxyType({})

    def record(
        self,
        digest: str,
        *,
        symbol: str,
        trade_id: int,
        label: str,
    ) -> _RawAggregateTrade:
        descriptor = self._pages_by_digest.get(digest)
        if descriptor is None:
            raise PublicDataError(f"{label} references an undeclared or empty raw artifact")
        if descriptor.symbol != symbol:
            raise PublicDataError(
                f"{label} references a raw page for {descriptor.symbol}, not {symbol}"
            )
        if self._digest != digest:
            # Re-hash at use time so verification and consumption are not split
            # by an unnoticed local-file replacement.
            _verify_file(descriptor.path, descriptor.sha256, f"raw aggregate page for {symbol}")
            records = _aggregate_records(
                descriptor.path,
                f"raw aggregate page for {symbol}",
                request_limit=self._request_limit,
                requested_range=self._requested_range,
            )
            _verify_file(descriptor.path, descriptor.sha256, f"raw aggregate page for {symbol}")
            ids = tuple(records)
            if (
                len(records) != descriptor.rows
                or (ids[0] if ids else None) != descriptor.first_id
                or (ids[-1] if ids else None) != descriptor.last_id
            ):
                raise PublicDataError(f"raw aggregate page metadata changed for {symbol}")
            self._digest = digest
            self._records = records
        raw_record = self._records.get(trade_id)
        if raw_record is None:
            raise PublicDataError(
                f"{label} trade_id is absent from its exact raw aggregate-trade page"
            )
        return raw_record


class _ExpectedLineageIndex:
    """Disk-backed inverse lineage for every downloader-selected raw trade."""

    def __init__(self, dataset: PublicTradeDataset) -> None:
        self._connection = sqlite3.connect("")
        self._connection.execute("PRAGMA cache_size = -2048")
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute("PRAGMA journal_mode = OFF")
        self._connection.execute("PRAGMA synchronous = OFF")
        self._connection.execute(
            """
            CREATE TABLE expected_lineage (
                symbol TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (symbol, source_sha256, trade_id)
            ) WITHOUT ROWID
            """
        )
        try:
            self._populate(dataset)
        except BaseException:
            self._connection.close()
            raise

    def _populate(self, dataset: PublicTradeDataset) -> None:
        row_cap = cast(int, dataset._config.data.max_events_per_symbol)
        for symbol in dataset._config.data.symbols:
            expected_count = 0
            for page in dataset._ordered_raw_pages[symbol]:
                if expected_count >= row_cap:
                    break
                _verify_file(page.path, page.sha256, f"raw aggregate page for {symbol}")
                records = _aggregate_records(
                    page.path,
                    f"raw aggregate page for {symbol}",
                    request_limit=dataset._config.data.request_limit,
                    requested_range=dataset._requested_range,
                )
                _verify_file(page.path, page.sha256, f"raw aggregate page for {symbol}")
                ids = tuple(records)
                if (
                    len(records) != page.rows
                    or (ids[0] if ids else None) != page.first_id
                    or (ids[-1] if ids else None) != page.last_id
                ):
                    raise PublicDataError(f"raw aggregate page metadata changed for {symbol}")
                selected: list[tuple[str, str, int]] = []
                for record in records.values():
                    if not (
                        dataset._requested_range[0]
                        <= record.event_ts_ns
                        < dataset._requested_range[1]
                    ):
                        continue
                    if expected_count >= row_cap:
                        break
                    selected.append((symbol, page.sha256, record.aggregate_id))
                    expected_count += 1
                try:
                    self._connection.executemany(
                        """
                        INSERT INTO expected_lineage (symbol, source_sha256, trade_id)
                        VALUES (?, ?, ?)
                        """,
                        selected,
                    )
                except sqlite3.IntegrityError as exc:
                    raise PublicDataError(
                        f"raw aggregate pages contain duplicate selected lineage for {symbol}"
                    ) from exc
            claimed = dataset._symbol_claims[symbol].rows
            if expected_count != claimed:
                raise PublicDataError(
                    f"raw selected rows for {symbol} ({expected_count}) do not match "
                    f"normalized coverage claim ({claimed})"
                )
        self._connection.commit()

    def mark_seen(self, *, symbol: str, source_sha256: str, trade_id: int) -> None:
        cursor = self._connection.execute(
            """
            UPDATE expected_lineage
            SET seen_count = seen_count + 1
            WHERE symbol = ? AND source_sha256 = ? AND trade_id = ?
            """,
            (symbol, source_sha256, trade_id),
        )
        if cursor.rowcount != 1:
            raise PublicDataError(
                "normalized trade is outside the exact downloader-selected raw sequence"
            )

    def mismatch_counts(self) -> tuple[int, int]:
        missing, repeated = self._connection.execute(
            """
            SELECT
                SUM(CASE WHEN seen_count = 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN seen_count > 1 THEN 1 ELSE 0 END)
            FROM expected_lineage
            """
        ).fetchone()
        return int(missing or 0), int(repeated or 0)

    def commit(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


def _validate_normalized_trade_row(
    raw_row: object,
    *,
    row_index: int,
    symbol_claims: Mapping[str, _SymbolClaim],
    raw_cache: _RawPageCache,
) -> None:
    label = f"normalized trade row {row_index}"
    row = _object(raw_row, label)
    symbol = _text(row.get("symbol"), f"{label}.symbol")
    claim = symbol_claims.get(symbol)
    if claim is None:
        raise PublicDataError(f"{label} has an unmanifested symbol {symbol}")
    source_id = _digest(row.get("source_artifact_id"), f"{label}.source_artifact_id")
    trade_id = _integer(row.get("trade_id"), f"{label}.trade_id", minimum=0)
    raw_record = raw_cache.record(
        source_id,
        symbol=symbol,
        trade_id=trade_id,
        label=label,
    )

    price = _positive_decimal_number(row.get("price"), f"{label}.price")
    quantity = _positive_decimal_number(row.get("quantity"), f"{label}.quantity")
    expected_price_ticks = _scaled_integer(raw_record.price, claim.tick_size, f"{label}.price")
    expected_quantity_lots = _scaled_integer(
        raw_record.quantity,
        claim.lot_size,
        f"{label}.quantity",
    )
    expected_aggressor = "sell" if raw_record.buyer_is_maker else "buy"
    mismatches = {
        "venue": row.get("venue") != "binance_spot",
        "event_ts_ns": row.get("event_ts_ns") != raw_record.event_ts_ns,
        "received_ts_ns": row.get("received_ts_ns") is not None,
        "available_ts_ns": row.get("available_ts_ns") != raw_record.event_ts_ns,
        "availability_basis": row.get("availability_basis") != "exchange_event_time_proxy",
        "capture_seq": row.get("capture_seq") is not None,
        "continuity_id": row.get("continuity_id") is not None,
        "first_trade_id": row.get("first_trade_id") != raw_record.first_trade_id,
        "last_trade_id": row.get("last_trade_id") != raw_record.last_trade_id,
        "price_ticks": row.get("price_ticks") != expected_price_ticks,
        "quantity_lots": row.get("quantity_lots") != expected_quantity_lots,
        "price": price != raw_record.price,
        "quantity": quantity != raw_record.quantity,
        "quote_quantity": row.get("quote_quantity")
        != float(raw_record.price) * float(raw_record.quantity),
        "aggressor_side": row.get("aggressor_side") != expected_aggressor,
        "buyer_is_maker": row.get("buyer_is_maker") != raw_record.buyer_is_maker,
    }
    mismatched_fields = sorted(field for field, mismatched in mismatches.items() if mismatched)
    if mismatched_fields:
        raise PublicDataError(
            f"{label} does not match its exact raw aggregate-trade record: "
            + ", ".join(mismatched_fields)
        )


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    _preflight_json_structure(path, label)
    try:
        return _object(read_json(path), label)
    except PublicDataError:
        raise
    except (OSError, ValueError) as exc:
        raise PublicDataError(f"cannot read {label} at {path}: {exc}") from exc


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise PublicDataError(f"missing {label}: {path}")
    try:
        observed = sha256_file(path)
    except OSError as exc:
        raise PublicDataError(f"cannot hash {label} at {path}: {exc}") from exc
    if observed != expected_sha256:
        raise PublicDataError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )


def _declared_file(root: Path, value: object, label: str) -> Path:
    declared = Path(_text(value, label))
    if declared.is_absolute():
        raise PublicDataError(f"{label} must be relative to its manifest root")
    resolved_root = root.resolve()
    candidate = (resolved_root / declared).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise PublicDataError(f"{label} escapes its manifest root: {declared}")
    if not candidate.is_file():
        raise PublicDataError(f"missing {label}: {candidate}")
    return candidate


def _utc_iso_from_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, _NS_PER_SECOND)
    instant = datetime.fromtimestamp(seconds, tz=UTC)
    return f"{instant:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def _utc_ns_from_iso(value: object, label: str) -> int:
    raw = _text(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicDataError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise PublicDataError(f"{label} must include a UTC offset")
    utc = parsed.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc - epoch
    return (
        delta.days * 86_400 * _NS_PER_SECOND
        + delta.seconds * _NS_PER_SECOND
        + delta.microseconds * 1_000
    )


def _coverage(start_ns: int, end_inclusive_ns: int) -> ObservedUtcCoverage:
    if end_inclusive_ns < start_ns:
        raise PublicDataError("observed coverage ends before it starts")
    return ObservedUtcCoverage(
        start_ns=start_ns,
        end_inclusive_ns=end_inclusive_ns,
        start_utc=_utc_iso_from_ns(start_ns),
        end_inclusive_utc=_utc_iso_from_ns(end_inclusive_ns),
    )


def _requested_range(manifest: Mapping[str, Any], label: str) -> tuple[int, int]:
    requested = _object(manifest.get("requested_range_ns"), f"{label}.requested_range_ns")
    start_ns = _integer(requested.get("start"), f"{label}.requested_range_ns.start")
    end_ns = _integer(requested.get("end_exclusive"), f"{label}.requested_range_ns.end_exclusive")
    if end_ns <= start_ns:
        raise PublicDataError(f"{label} requested range must be non-empty")
    return start_ns, end_ns


def _terminal_claim(
    item: Mapping[str, Any],
    *,
    label: str,
    rows: int,
    complete_range: bool,
    requested_range: tuple[int, int],
) -> _TerminalClaim | None:
    top_fields = ("raw_page_count", "stop_reason", "last_raw_page_sha256")
    presence = [item.get(field) is not None for field in top_fields]
    summary_value = item.get("stream_summary")
    if not any(presence) and summary_value is None:
        return None
    if not all(presence) or summary_value is None:
        raise PublicDataError(f"{label} has an incomplete terminal stream claim")
    raw_page_count = _integer(item.get("raw_page_count"), f"{label}.raw_page_count", minimum=1)
    stop_reason = _text(item.get("stop_reason"), f"{label}.stop_reason")
    if stop_reason not in {"event_cap", "range_end", "short_page", "empty_page"}:
        raise PublicDataError(f"{label}.stop_reason is unsupported")
    last_sha = _digest(item.get("last_raw_page_sha256"), f"{label}.last_raw_page_sha256")
    summary = _object(summary_value, f"{label}.stream_summary")
    if (
        _integer(
            summary.get("requested_start_ns"),
            f"{label}.stream_summary.requested_start_ns",
        )
        != requested_range[0]
        or _integer(
            summary.get("requested_end_ns"),
            f"{label}.stream_summary.requested_end_ns",
        )
        != requested_range[1]
        or _integer(
            summary.get("rows_yielded"),
            f"{label}.stream_summary.rows_yielded",
            minimum=1,
        )
        != rows
        or _integer(
            summary.get("raw_page_count"),
            f"{label}.stream_summary.raw_page_count",
            minimum=1,
        )
        != raw_page_count
        or _text(summary.get("stop_reason"), f"{label}.stream_summary.stop_reason") != stop_reason
        or _boolean(
            summary.get("complete_range"),
            f"{label}.stream_summary.complete_range",
        )
        != complete_range
    ):
        raise PublicDataError(f"{label}.stream_summary disagrees with its symbol claim")
    expected_complete = stop_reason != "event_cap" and rows > 0
    if complete_range != expected_complete:
        raise PublicDataError(f"{label} completeness disagrees with terminal stop reason")
    last = _object(summary.get("last_raw_page"), f"{label}.stream_summary.last_raw_page")
    last_page_sha = _digest(last.get("sha256"), f"{label}.stream_summary.last_raw_page.sha256")
    if last_page_sha != last_sha:
        raise PublicDataError(f"{label} last raw page SHA-256 claims disagree")
    return _TerminalClaim(
        raw_page_count=raw_page_count,
        stop_reason=stop_reason,
        last_raw_page_sha256=last_sha,
        last_path=_text(last.get("path"), f"{label}.stream_summary.last_raw_page.path"),
        last_manifest_path=_text(
            last.get("manifest_path"),
            f"{label}.stream_summary.last_raw_page.manifest_path",
        ),
        last_request_uri=_text(
            last.get("request_uri"),
            f"{label}.stream_summary.last_raw_page.request_uri",
        ),
        last_row_count=_integer(
            last.get("row_count"),
            f"{label}.stream_summary.last_raw_page.row_count",
            minimum=0,
        ),
    )


def _validate_ingestion_manifest(
    manifest: Mapping[str, Any],
    config: ProjectConfig,
) -> tuple[
    PublicEvidenceTier,
    bool,
    tuple[int, int],
    dict[str, _SymbolClaim],
    Mapping[str, Any],
]:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise PublicDataError("unsupported ingestion manifest version")
    if manifest.get("artifact_kind") != "ingestion_run":
        raise PublicDataError("manifest is not an ingestion_run artifact")
    if manifest.get("mode") != "binance_rest":
        raise PublicDataError("public trade reader requires a binance_rest ingestion manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("schema_version") != config.data.schema_version
    ):
        raise PublicDataError("ingestion manifest has an unsupported schema version")
    if _text(manifest.get("source"), "ingestion.source") != config.data.source:
        raise PublicDataError("ingestion source does not match configured source")

    effective = _text(manifest.get("evidence_tier"), "ingestion.evidence_tier")
    requested = _text(manifest.get("requested_evidence_tier"), "ingestion.requested_evidence_tier")
    if effective not in {"PUBLIC_SAMPLE_PARTIAL", "FULL_DATA"}:
        raise PublicDataError(f"unsupported public evidence tier: {effective!r}")
    if requested not in {"PUBLIC_SAMPLE_PARTIAL", "FULL_DATA"}:
        raise PublicDataError(f"unsupported requested public evidence tier: {requested!r}")
    if requested != config.run.evidence_tier:
        raise PublicDataError("requested evidence tier does not match configuration")

    all_complete = _boolean(
        manifest.get("all_requested_ranges_complete"),
        "ingestion.all_requested_ranges_complete",
    )
    row_cap = _integer(
        manifest.get("row_cap_per_symbol"), "ingestion.row_cap_per_symbol", minimum=1
    )
    if row_cap != config.data.max_events_per_symbol:
        raise PublicDataError("ingestion row cap does not match configuration")
    requested_range = _requested_range(manifest, "ingestion")
    if config.data.end is None:
        raise PublicDataError("public input configuration requires a bounded end time")
    configured_range = (datetime_to_ns(config.data.start), datetime_to_ns(config.data.end))
    if requested_range != configured_range:
        raise PublicDataError("ingestion requested range does not match configuration")
    raw_symbols = _array(manifest.get("symbols"), "ingestion.symbols")
    if not raw_symbols:
        raise PublicDataError("ingestion.symbols must not be empty")
    symbols: dict[str, _SymbolClaim] = {}
    for index, raw_symbol in enumerate(raw_symbols):
        item = _object(raw_symbol, f"ingestion.symbols[{index}]")
        symbol = _text(item.get("symbol"), f"ingestion.symbols[{index}].symbol")
        if symbol in symbols:
            raise PublicDataError(f"duplicate symbol coverage entry: {symbol}")
        rows = _integer(item.get("rows"), f"ingestion.symbols[{index}].rows", minimum=1)
        if rows > row_cap:
            raise PublicDataError(f"manifested rows for {symbol} exceed row_cap_per_symbol")
        complete = _boolean(
            item.get("complete_range"), f"ingestion.symbols[{index}].complete_range"
        )
        symbols[symbol] = _SymbolClaim(
            rows=rows,
            complete_range=complete,
            tick_size=_positive_decimal(
                item.get("tick_size"), f"ingestion.symbols[{index}].tick_size"
            ),
            lot_size=_positive_decimal(
                item.get("lot_size"), f"ingestion.symbols[{index}].lot_size"
            ),
            terminal=_terminal_claim(
                item,
                label=f"ingestion.symbols[{index}]",
                rows=rows,
                complete_range=complete,
                requested_range=requested_range,
            ),
        )

    if set(symbols) != set(config.data.symbols):
        raise PublicDataError("ingestion symbols do not match configured symbols")

    derived_complete = all(claim.complete_range for claim in symbols.values())
    if all_complete != derived_complete:
        raise PublicDataError("all_requested_ranges_complete disagrees with per-symbol coverage")
    expected_effective = requested if all_complete else "PUBLIC_SAMPLE_PARTIAL"
    if effective == "FULL_DATA" and expected_effective != "FULL_DATA":
        raise PublicDataError("partial or lower-tier coverage cannot be promoted to FULL_DATA")
    if effective != expected_effective:
        raise PublicDataError(
            "effective evidence tier does not match requested tier and manifested coverage"
        )
    legacy_complete = sorted(
        symbol
        for symbol, claim in symbols.items()
        if claim.complete_range and claim.terminal is None
    )
    if legacy_complete:
        raise PublicDataError(
            "complete public coverage lacks terminal stream evidence for symbols: "
            + ", ".join(legacy_complete)
        )

    datasets = _array(manifest.get("normalized_datasets"), "ingestion.normalized_datasets")
    if len(datasets) != 1:
        raise PublicDataError("public trade ingestion must declare exactly one normalized dataset")
    dataset = _object(datasets[0], "ingestion.normalized_datasets[0]")
    if dataset.get("schema_name") != "trades":
        raise PublicDataError("public ingestion normalized dataset must use the trades schema")

    return (
        cast(PublicEvidenceTier, effective),
        all_complete,
        requested_range,
        symbols,
        dataset,
    )


def _validate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    ingestion_source: object,
    requested_range: tuple[int, int],
) -> tuple[int, list[Any], str]:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise PublicDataError("unsupported normalized dataset manifest version")
    if manifest.get("dataset") != "trades":
        raise PublicDataError("referenced normalized manifest is not the trades dataset")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PublicDataError("normalized dataset manifest has an unsupported schema version")
    source = _text(manifest.get("source"), "normalized dataset.source")
    if source != ingestion_source:
        raise PublicDataError("ingestion and normalized dataset sources do not match")
    source_uri = _text(manifest.get("source_uri"), "normalized dataset.source_uri")
    if _requested_range(manifest, "normalized dataset") != requested_range:
        raise PublicDataError("ingestion and normalized dataset requested ranges do not match")
    rows = _integer(manifest.get("rows"), "normalized dataset.rows", minimum=1)
    artifacts = _array(manifest.get("artifacts"), "normalized dataset.artifacts")
    if not artifacts:
        raise PublicDataError("normalized dataset manifest declares no Parquet parts")
    return rows, artifacts, source_uri


def _verify_raw_artifacts(
    ingestion: Mapping[str, Any],
    *,
    bundle_root: Path,
    requested_range: tuple[int, int],
    config: ProjectConfig,
    symbol_claims: Mapping[str, _SymbolClaim],
) -> _VerifiedRawArtifacts:
    entries = _array(ingestion.get("raw_artifacts"), "ingestion.raw_artifacts")
    if not entries:
        raise PublicDataError("public ingestion manifest declares no raw artifacts")
    raw_root = (bundle_root / "raw").resolve()
    paths: list[Path] = []
    manifest_paths: list[Path] = []
    digests: set[str] = set()
    pages_by_digest: dict[str, _RawPageDescriptor] = {}
    ordered_page_descriptors: dict[str, tuple[_RawPageDescriptor, ...]] = {}
    aggregate_pages: dict[str, list[_AggregatePage]] = {symbol: [] for symbol in symbol_claims}
    exchange_info_symbols: set[str] = set()
    seen_paths: dict[Path, bool] = {}
    seen_manifest_paths: set[Path] = set()

    try:
        base_path = urlsplit(config.data.base_url).path.rstrip("/")
    except ValueError as exc:
        raise PublicDataError(f"configured data.base_url is not valid: {exc}") from exc
    aggregate_endpoint = "/api/v3/aggTrades"
    exchange_info_endpoint = "/api/v3/exchangeInfo"

    for index, raw_entry in enumerate(entries):
        label = f"ingestion.raw_artifacts[{index}]"
        entry = _object(raw_entry, label)
        path = _declared_file(bundle_root, entry.get("path"), f"{label}.path")
        manifest_path = _declared_file(
            bundle_root, entry.get("manifest_path"), f"{label}.manifest_path"
        )
        if not path.is_relative_to(raw_root) or not manifest_path.is_relative_to(raw_root):
            raise PublicDataError(f"{label} is outside the ingestion bundle's raw root")
        if manifest_path in seen_manifest_paths:
            raise PublicDataError(f"duplicate declared raw sidecar at index {index}")
        seen_manifest_paths.add(manifest_path)

        digest = _digest(entry.get("sha256"), f"{label}.sha256")
        manifest_digest = _digest(entry.get("manifest_sha256"), f"{label}.manifest_sha256")
        _verify_file(path, digest, f"raw artifact {index}")
        _verify_file(manifest_path, manifest_digest, f"raw artifact sidecar {index}")
        if path.stat().st_size > _MAX_RAW_ARTIFACT_BYTES:
            raise PublicDataError(
                f"raw artifact {index} exceeds bounded JSON size {_MAX_RAW_ARTIFACT_BYTES} bytes"
            )
        sidecar = _load_object(manifest_path, f"raw artifact sidecar {index}")
        if sidecar.get("manifest_version") != MANIFEST_VERSION:
            raise PublicDataError(f"raw artifact sidecar {index} has an unsupported version")
        if sidecar.get("artifact_kind") != "raw_source":
            raise PublicDataError(f"raw artifact sidecar {index} has an unexpected kind")
        if sidecar.get("source") != "binance_spot_public_api":
            raise PublicDataError(f"raw artifact sidecar {index} is not a Binance public source")
        source_uri = _text(sidecar.get("source_uri"), f"raw artifact sidecar {index}.source_uri")
        downloaded_at_ns = _utc_ns_from_iso(
            sidecar.get("downloaded_at_utc"),
            f"raw artifact sidecar {index}.downloaded_at_utc",
        )
        if sidecar.get("path") != path.name or manifest_path.parent != path.parent:
            raise PublicDataError(f"raw artifact sidecar {index} path does not match")
        if (
            _integer(sidecar.get("bytes"), f"raw artifact sidecar {index}.bytes", minimum=1)
            != path.stat().st_size
        ):
            raise PublicDataError(f"raw artifact sidecar {index} byte count does not match")
        checksum = _object(sidecar.get("checksum"), f"raw artifact sidecar {index}.checksum")
        if (
            checksum.get("algorithm") != "sha256"
            or _digest(checksum.get("value"), f"raw artifact sidecar {index}.checksum.value")
            != digest
        ):
            raise PublicDataError(f"raw artifact sidecar {index} checksum does not match")

        raw_requested = _object(
            sidecar.get("requested_range_ns"),
            f"raw artifact sidecar {index}.requested_range_ns",
        )
        raw_start = raw_requested.get("start")
        raw_end = raw_requested.get("end_exclusive")

        requested_path = urlsplit(source_uri).path
        raw_label = f"raw artifact sidecar {index}.source_uri"
        is_empty_aggregate = False
        if requested_path == f"{base_path}{aggregate_endpoint}":
            query = _request_query(
                source_uri,
                base_url=config.data.base_url,
                endpoint=aggregate_endpoint,
                label=raw_label,
            )
            initial_keys = {"symbol", "startTime", "endTime", "limit"}
            continuation_keys = {"symbol", "fromId", "limit"}
            if set(query) == initial_keys:
                expected_start_ms = requested_range[0] // 1_000_000
                expected_end_ms = (requested_range[1] - 1) // 1_000_000
                if (
                    _unsigned_integer_text(query["startTime"], f"{raw_label}.startTime")
                    != expected_start_ms
                    or _unsigned_integer_text(query["endTime"], f"{raw_label}.endTime")
                    != expected_end_ms
                ):
                    raise PublicDataError(
                        f"{raw_label} time query does not match the requested range"
                    )
                from_id: int | None = None
            elif set(query) == continuation_keys:
                from_id = _unsigned_integer_text(query["fromId"], f"{raw_label}.fromId")
            else:
                raise PublicDataError(
                    f"{raw_label} must use exactly the initial-time or fromId query parameters"
                )
            symbol = query["symbol"]
            if symbol not in symbol_claims:
                raise PublicDataError(f"{raw_label} symbol is not a configured symbol")
            limit = _unsigned_integer_text(query["limit"], f"{raw_label}.limit", minimum=1)
            if limit != config.data.request_limit:
                raise PublicDataError(f"{raw_label} limit does not match configuration")
            if (
                raw_start is None
                or raw_end is None
                or (
                    _integer(raw_start, f"raw artifact sidecar {index}.requested_range_ns.start"),
                    _integer(
                        raw_end,
                        f"raw artifact sidecar {index}.requested_range_ns.end_exclusive",
                    ),
                )
                != requested_range
            ):
                raise PublicDataError(
                    f"raw artifact sidecar {index} requested range does not match"
                )
            aggregate_records = _aggregate_records(
                path,
                f"raw aggregate-trade artifact {index}",
                request_limit=config.data.request_limit,
                requested_range=requested_range,
            )
            aggregate_ids = tuple(aggregate_records)
            if from_id is not None and aggregate_ids and aggregate_ids[0] != from_id:
                raise PublicDataError(
                    f"raw aggregate-trade artifact {index} does not begin at requested fromId"
                )
            is_empty_aggregate = not aggregate_records
            if aggregate_records:
                descriptor = _RawPageDescriptor(
                    path=path,
                    sha256=digest,
                    symbol=symbol,
                    from_id=from_id,
                    rows=len(aggregate_records),
                    first_id=aggregate_ids[0],
                    last_id=aggregate_ids[-1],
                )
                existing = pages_by_digest.get(digest)
                if existing is not None and (
                    existing.symbol != descriptor.symbol
                    or existing.from_id != descriptor.from_id
                    or existing.first_id != descriptor.first_id
                    or existing.last_id != descriptor.last_id
                ):
                    raise PublicDataError(
                        f"raw aggregate-trade digest has ambiguous nonempty semantics at index {index}"
                    )
                if existing is None:
                    pages_by_digest[digest] = descriptor
            aggregate_pages[symbol].append(
                _AggregatePage(
                    path=path,
                    manifest_path=manifest_path,
                    sha256=digest,
                    request_uri=source_uri,
                    downloaded_at_ns=downloaded_at_ns,
                    from_id=from_id,
                    rows=len(aggregate_records),
                    first_id=aggregate_ids[0] if aggregate_ids else None,
                    last_id=aggregate_ids[-1] if aggregate_ids else None,
                    first_event_ts_ns=(
                        aggregate_records[aggregate_ids[0]].event_ts_ns if aggregate_ids else None
                    ),
                    last_event_ts_ns=(
                        aggregate_records[aggregate_ids[-1]].event_ts_ns if aggregate_ids else None
                    ),
                )
            )
        elif requested_path == f"{base_path}{exchange_info_endpoint}":
            query = _request_query(
                source_uri,
                base_url=config.data.base_url,
                endpoint=exchange_info_endpoint,
                label=raw_label,
            )
            if set(query) != {"symbol"}:
                raise PublicDataError(f"{raw_label} must contain exactly the symbol parameter")
            symbol = query["symbol"]
            claim = symbol_claims.get(symbol)
            if claim is None:
                raise PublicDataError(f"{raw_label} symbol is not a configured symbol")
            if raw_start is not None or raw_end is not None:
                raise PublicDataError(
                    f"raw artifact sidecar {index} exchangeInfo range must be null"
                )
            _validate_exchange_info_payload(
                path,
                symbol=symbol,
                claim=claim,
                label=f"raw exchangeInfo artifact {index}",
            )
            exchange_info_symbols.add(symbol)
        else:
            raise PublicDataError(f"{raw_label} does not use an allowed Binance public endpoint")

        if path in seen_paths and not (seen_paths[path] and is_empty_aggregate):
            raise PublicDataError(
                f"duplicate raw data path has nonempty or non-aggregate semantics at index {index}"
            )
        seen_paths[path] = is_empty_aggregate

        paths.append(path)
        manifest_paths.append(manifest_path)
        digests.add(digest)

    if exchange_info_symbols != set(symbol_claims):
        missing = sorted(set(symbol_claims).difference(exchange_info_symbols))
        raise PublicDataError(
            "public ingestion lacks exchangeInfo raw artifacts for configured symbols: "
            + ", ".join(missing)
        )
    for symbol, pages in aggregate_pages.items():
        initial_pages = [page for page in pages if page.from_id is None]
        if len(initial_pages) != 1:
            raise PublicDataError(
                f"public ingestion must declare exactly one initial-time aggTrades page for {symbol}"
            )
        previous_last_id = initial_pages[0].last_id
        previous_last_event_ts_ns = initial_pages[0].last_event_ts_ns
        continuation_pages = sorted(
            (page for page in pages if page.from_id is not None),
            key=lambda page: cast(int, page.from_id),
        )
        seen_from_ids: set[int] = set()
        terminal_empty_seen = initial_pages[0].rows == 0
        for page in continuation_pages:
            from_id = cast(int, page.from_id)
            if from_id in seen_from_ids:
                raise PublicDataError(f"raw aggregate-trade pagination repeats fromId for {symbol}")
            seen_from_ids.add(from_id)
            if terminal_empty_seen:
                raise PublicDataError(
                    f"raw aggregate-trade pagination continues after an empty page for {symbol}"
                )
            if previous_last_id is None or from_id != previous_last_id + 1:
                raise PublicDataError(
                    f"raw aggregate-trade pagination is not contiguous for {symbol}"
                )
            if (
                previous_last_event_ts_ns is not None
                and page.first_event_ts_ns is not None
                and page.first_event_ts_ns < previous_last_event_ts_ns
            ):
                raise PublicDataError(
                    f"raw aggregate-trade event time reverses across pages for {symbol}"
                )
            if page.rows == 0:
                terminal_empty_seen = True
            else:
                previous_last_id = page.last_id
                previous_last_event_ts_ns = page.last_event_ts_ns

        ordered_pages = [initial_pages[0], *continuation_pages]
        ordered_page_descriptors[symbol] = tuple(
            _RawPageDescriptor(
                path=page.path,
                sha256=page.sha256,
                symbol=symbol,
                from_id=page.from_id,
                rows=page.rows,
                first_id=page.first_id,
                last_id=page.last_id,
            )
            for page in ordered_pages
        )
        terminal_page = ordered_pages[-1]
        symbol_claim = symbol_claims[symbol]
        if symbol_claim.rows >= cast(int, config.data.max_events_per_symbol):
            derived_stop_reason = "event_cap"
        elif terminal_page.rows == 0:
            derived_stop_reason = "empty_page"
        elif (
            terminal_page.last_event_ts_ns is not None
            and terminal_page.last_event_ts_ns >= requested_range[1]
        ):
            derived_stop_reason = "range_end"
        elif terminal_page.rows < config.data.request_limit:
            derived_stop_reason = "short_page"
        else:
            raise PublicDataError(
                f"raw aggregate-trade chain ends without a terminal condition for {symbol}"
            )
        derived_complete = derived_stop_reason != "event_cap" and symbol_claim.rows > 0
        if symbol_claim.complete_range != derived_complete:
            raise PublicDataError(
                f"manifested completeness disagrees with raw terminal page for {symbol}"
            )
        if derived_complete and terminal_page.downloaded_at_ns < requested_range[1]:
            raise PublicDataError(
                f"complete range for {symbol} ends after its terminal page was downloaded"
            )
        terminal_claim = symbol_claim.terminal
        if terminal_claim is not None and (
            terminal_claim.raw_page_count != len(ordered_pages)
            or terminal_claim.stop_reason != derived_stop_reason
            or terminal_claim.last_raw_page_sha256 != terminal_page.sha256
            or terminal_claim.last_row_count != terminal_page.rows
            or terminal_claim.last_request_uri != terminal_page.request_uri
            or _declared_file(
                bundle_root,
                terminal_claim.last_path,
                f"terminal raw page path for {symbol}",
            )
            != terminal_page.path
            or _declared_file(
                bundle_root,
                terminal_claim.last_manifest_path,
                f"terminal raw page manifest path for {symbol}",
            )
            != terminal_page.manifest_path
        ):
            raise PublicDataError(
                f"terminal stream claim does not match declared raw pages for {symbol}"
            )

    return _VerifiedRawArtifacts(
        paths=tuple(sorted(set(paths))),
        manifest_paths=tuple(sorted(manifest_paths)),
        sha256s=frozenset(digests),
        pages_by_digest=MappingProxyType(pages_by_digest),
        ordered_pages=MappingProxyType(ordered_page_descriptors),
    )


def _verify_part_descriptor(
    raw_artifact: object,
    *,
    index: int,
    write_ordinal: int,
    normalized_root: Path,
    dataset_source: str,
    dataset_source_uri: str,
    requested_range: tuple[int, int],
) -> _ParquetPartDescriptor:
    label = f"normalized dataset.artifacts[{index}]"
    artifact = _object(raw_artifact, label)
    declared_rows = _integer(artifact.get("rows"), f"{label}.rows", minimum=1)
    data_sha = _digest(artifact.get("data_sha256"), f"{label}.data_sha256")
    sidecar_sha = _digest(artifact.get("manifest_sha256"), f"{label}.manifest_sha256")
    data_path = _declared_file(normalized_root, artifact.get("data_path"), f"{label}.data_path")
    sidecar_path = _declared_file(
        normalized_root, artifact.get("manifest_path"), f"{label}.manifest_path"
    )
    _verify_file(data_path, data_sha, f"Parquet part {index}")
    _verify_file(sidecar_path, sidecar_sha, f"Parquet sidecar {index}")

    sidecar = _load_object(sidecar_path, f"Parquet sidecar {index}")
    if sidecar.get("manifest_version") != MANIFEST_VERSION:
        raise PublicDataError(f"Parquet sidecar {index} has an unsupported manifest version")
    if sidecar.get("artifact_kind") != "normalized_parquet":
        raise PublicDataError(f"Parquet sidecar {index} has an unexpected artifact kind")
    if sidecar.get("dataset") != "trades" or sidecar.get("schema_name") != "trades":
        raise PublicDataError(f"Parquet sidecar {index} declares an unexpected schema")
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise PublicDataError(f"Parquet sidecar {index} has an unsupported schema version")
    if sidecar.get("source") != dataset_source:
        raise PublicDataError(f"Parquet sidecar {index} source does not match its dataset")
    if sidecar.get("source_uri") != dataset_source_uri:
        raise PublicDataError(f"Parquet sidecar {index} source URI does not match its dataset")
    if _requested_range(sidecar, f"Parquet sidecar {index}") != requested_range:
        raise PublicDataError(f"Parquet sidecar {index} requested range does not match")
    if _integer(sidecar.get("rows"), f"Parquet sidecar {index}.rows", minimum=1) != declared_rows:
        raise PublicDataError(f"Parquet sidecar {index} row count does not match")
    checksum = _object(sidecar.get("checksum"), f"Parquet sidecar {index}.checksum")
    if (
        checksum.get("algorithm") != "sha256"
        or _digest(checksum.get("value"), f"Parquet sidecar {index}.checksum.value") != data_sha
    ):
        raise PublicDataError(f"Parquet sidecar {index} checksum does not match")
    if sidecar.get("path") != artifact.get("data_path"):
        raise PublicDataError(f"Parquet sidecar {index} data path does not match")
    sidecar_ordinal = sidecar.get("write_ordinal")
    if (
        sidecar_ordinal is not None
        and _integer(sidecar_ordinal, f"Parquet sidecar {index}.write_ordinal", minimum=0)
        != write_ordinal
    ):
        raise PublicDataError(f"Parquet sidecar {index} write ordinal does not match")
    if (
        _integer(sidecar.get("bytes"), f"Parquet sidecar {index}.bytes", minimum=1)
        != data_path.stat().st_size
    ):
        raise PublicDataError(f"Parquet sidecar {index} byte count does not match")
    proportional_parquet_bound = _PARQUET_BASE_BYTES + (
        declared_rows * _PARQUET_BYTES_PER_TRADE_ROW
    )
    encoded_parquet_bound = min(
        proportional_parquet_bound,
        _MAX_PARQUET_PART_ENCODED_BYTES,
    )
    if data_path.stat().st_size > encoded_parquet_bound:
        raise PublicDataError(
            f"Parquet part {index} exceeds bounded encoded bytes for its row count"
        )

    expected_schema = get_schema("trades")
    try:
        parquet = pq.ParquetFile(data_path)
        if parquet.metadata.num_rows != declared_rows:
            raise PublicDataError(f"Parquet part {index} metadata row count does not match")
        uncompressed_bytes = sum(
            parquet.metadata.row_group(row_group).total_byte_size
            for row_group in range(parquet.metadata.num_row_groups)
        )
        decoded_parquet_bound = min(
            proportional_parquet_bound,
            _MAX_PARQUET_PART_DECODED_BYTES,
        )
        if uncompressed_bytes > decoded_parquet_bound:
            raise PublicDataError(
                f"Parquet part {index} exceeds bounded decoded bytes for its row count"
            )
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
            raise PublicDataError(f"Parquet part {index} has an unexpected trades schema")
    except PublicDataError:
        raise
    except (OSError, pa.ArrowException, ValueError) as exc:
        raise PublicDataError(f"cannot inspect Parquet part {index} metadata: {exc}") from exc

    observed = _object(sidecar.get("observed_range_ns"), f"Parquet sidecar {index}.observed")
    observed_start = _integer(observed.get("start"), f"Parquet sidecar {index}.observed.start")
    observed_end = _integer(
        observed.get("end_inclusive"),
        f"Parquet sidecar {index}.observed.end_inclusive",
    )
    if (
        observed_end < observed_start
        or observed_start < requested_range[0]
        or observed_end >= requested_range[1]
    ):
        raise PublicDataError(f"Parquet sidecar {index} observed range is invalid")
    artifact_observed_value = artifact.get("observed_range_ns")
    if artifact_observed_value is not None:
        artifact_observed = _object(
            artifact_observed_value,
            f"normalized dataset.artifacts[{index}].observed_range_ns",
        )
        if (
            _integer(
                artifact_observed.get("start"),
                f"normalized dataset.artifacts[{index}].observed_range_ns.start",
            )
            != observed_start
            or _integer(
                artifact_observed.get("end_inclusive"),
                f"normalized dataset.artifacts[{index}].observed_range_ns.end_inclusive",
            )
            != observed_end
        ):
            raise PublicDataError(
                f"normalized dataset artifact {index} observed range does not match sidecar"
            )
    sidecar_symbol = _text(sidecar.get("symbol"), f"Parquet sidecar {index}.symbol")
    sidecar_venue = _text(sidecar.get("venue"), f"Parquet sidecar {index}.venue")
    partition_date = _text(sidecar.get("partition_date"), f"Parquet sidecar {index}.partition_date")
    try:
        parsed_date = datetime.strptime(partition_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise PublicDataError(f"Parquet sidecar {index} partition date is invalid") from exc
    observed_dates = {
        datetime.fromtimestamp(timestamp // _NS_PER_SECOND, tz=UTC).date()
        for timestamp in (observed_start, observed_end)
    }
    if observed_dates != {parsed_date}:
        raise PublicDataError(f"Parquet sidecar {index} observed range crosses its partition date")
    return _ParquetPartDescriptor(
        data_path=data_path,
        data_sha256=data_sha,
        sidecar_path=sidecar_path,
        rows=declared_rows,
        write_ordinal=write_ordinal,
        venue=sidecar_venue,
        symbol=sidecar_symbol,
        partition_date=partition_date,
        observed_start_ns=observed_start,
        observed_end_inclusive_ns=observed_end,
    )


def verify_public_trade_dataset(
    config: ProjectConfig,
    ingestion_manifest_path: str | Path,
    *,
    ingestion_manifest_sha256: str,
) -> PublicTradeDataset:
    """Verify a public-trade bundle without materializing normalized rows.

    ``ingestion_manifest_sha256`` is required so the path itself cannot silently
    select a different ingestion run.  This pass verifies every manifest,
    sidecar, file digest, Parquet footer/schema, raw-page pagination claim, and
    coverage claim.  It retains only O(parts + raw pages + symbols) descriptors.
    Normalized Parquet row data are read only by ``iter_verified_batches``.
    """
    if config.data.mode != "binance_rest":
        raise PublicDataError("public trade reader requires data.mode='binance_rest'")
    configured_cap = config.data.max_events_per_symbol
    if configured_cap is None or configured_cap < 1:
        raise PublicDataError("public trade reader requires a configured positive row cap")
    max_rows = configured_cap * len(config.data.symbols)
    expected_ingestion_sha = _digest(ingestion_manifest_sha256, "ingestion_manifest_sha256")
    manifest_path = Path(ingestion_manifest_path).resolve()
    if manifest_path.parent.name != "_ingestion_manifests":
        raise PublicDataError(
            "ingestion manifest must remain under its bundle _ingestion_manifests directory"
        )
    _verify_file(manifest_path, expected_ingestion_sha, "ingestion manifest")
    ingestion = _load_object(manifest_path, "ingestion manifest")
    evidence_tier, all_complete, requested_range, symbol_claims, dataset_entry = (
        _validate_ingestion_manifest(ingestion, config)
    )

    bundle_root = manifest_path.parent.parent.resolve()
    verified_raw = _verify_raw_artifacts(
        ingestion,
        bundle_root=bundle_root,
        requested_range=requested_range,
        config=config,
        symbol_claims=symbol_claims,
    )
    dataset_manifest_path = _declared_file(
        bundle_root,
        dataset_entry.get("manifest_path"),
        "ingestion.normalized_datasets[0].manifest_path",
    )
    dataset_manifest_sha = _digest(
        dataset_entry.get("manifest_sha256"),
        "ingestion.normalized_datasets[0].manifest_sha256",
    )
    _verify_file(dataset_manifest_path, dataset_manifest_sha, "normalized dataset manifest")
    if dataset_manifest_path.parent.name != "_manifests":
        raise PublicDataError("normalized dataset manifest is outside its _manifests directory")
    normalized_root = dataset_manifest_path.parent.parent.resolve()
    if normalized_root != (bundle_root / "normalized").resolve():
        raise PublicDataError(
            "normalized dataset is not under the ingestion bundle's normalized root"
        )

    dataset_manifest = _load_object(dataset_manifest_path, "normalized dataset manifest")
    dataset_rows, raw_artifacts, dataset_source_uri = _validate_dataset_manifest(
        dataset_manifest,
        ingestion_source=ingestion.get("source"),
        requested_range=requested_range,
    )
    ingestion_rows = _integer(
        dataset_entry.get("rows"), "ingestion.normalized_datasets[0].rows", minimum=1
    )
    if dataset_rows > max_rows:
        raise PublicDataError(
            f"manifested trades contain {dataset_rows} rows, above required bound {max_rows}"
        )
    if dataset_rows != ingestion_rows or dataset_rows != sum(
        claim.rows for claim in symbol_claims.values()
    ):
        raise PublicDataError("ingestion, symbol, and normalized dataset row counts do not match")

    parts: list[_ParquetPartDescriptor] = []
    declared_part_rows = 0
    seen_data_paths: set[Path] = set()
    seen_sidecar_paths: set[Path] = set()
    declared_symbol_rows: dict[str, int] = {symbol: 0 for symbol in symbol_claims}
    declared_symbol_starts: dict[str, int] = {}
    declared_symbol_ends: dict[str, int] = {}
    ordinal_presence: list[bool] = []
    declared_ordinals: set[int] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _object(raw_artifact, f"normalized dataset.artifacts[{index}]")
        ordinal_value = artifact.get("write_ordinal")
        ordinal_presence.append(ordinal_value is not None)
        write_ordinal = (
            _integer(
                ordinal_value,
                f"normalized dataset.artifacts[{index}].write_ordinal",
                minimum=0,
            )
            if ordinal_value is not None
            else index
        )
        if write_ordinal in declared_ordinals:
            raise PublicDataError(f"duplicate Parquet write ordinal: {write_ordinal}")
        declared_ordinals.add(write_ordinal)
        next_rows = _integer(
            artifact.get("rows"), f"normalized dataset.artifacts[{index}].rows", minimum=1
        )
        if declared_part_rows + next_rows > max_rows:
            raise PublicDataError(f"declared Parquet parts exceed required row bound {max_rows}")
        part = _verify_part_descriptor(
            artifact,
            index=index,
            write_ordinal=write_ordinal,
            normalized_root=normalized_root,
            dataset_source=_text(dataset_manifest.get("source"), "normalized dataset.source"),
            dataset_source_uri=dataset_source_uri,
            requested_range=requested_range,
        )
        if part.data_path in seen_data_paths:
            raise PublicDataError(f"duplicate declared Parquet part: {part.data_path}")
        if part.sidecar_path in seen_sidecar_paths:
            raise PublicDataError(f"duplicate declared Parquet sidecar: {part.sidecar_path}")
        seen_data_paths.add(part.data_path)
        seen_sidecar_paths.add(part.sidecar_path)
        if part.venue != "binance_spot":
            raise PublicDataError(f"Parquet part {index} venue is not binance_spot")
        if part.symbol not in symbol_claims:
            raise PublicDataError(f"Parquet part {index} symbol is not manifested")
        declared_part_rows += part.rows
        declared_symbol_rows[part.symbol] += part.rows
        declared_symbol_starts[part.symbol] = min(
            declared_symbol_starts.get(part.symbol, part.observed_start_ns),
            part.observed_start_ns,
        )
        declared_symbol_ends[part.symbol] = max(
            declared_symbol_ends.get(part.symbol, part.observed_end_inclusive_ns),
            part.observed_end_inclusive_ns,
        )
        parts.append(part)

    if declared_part_rows != dataset_rows:
        raise PublicDataError("declared Parquet part rows do not match dataset rows")
    if any(ordinal_presence) and not all(ordinal_presence):
        raise PublicDataError("normalized dataset mixes present and missing write ordinals")
    if all(ordinal_presence) and declared_ordinals != set(range(len(parts))):
        raise PublicDataError("normalized dataset write ordinals must be contiguous from zero")
    canonical_order = ("venue", "symbol", "available_ts_ns", "event_ts_ns", "trade_id")
    symbol_coverage: list[SymbolObservedCoverage] = []
    for symbol in sorted(symbol_claims):
        claim = symbol_claims[symbol]
        if declared_symbol_rows[symbol] != claim.rows:
            raise PublicDataError(f"declared Parquet rows for {symbol} do not match coverage claim")
        if symbol not in declared_symbol_starts:
            raise PublicDataError(f"declared Parquet parts contain no rows for {symbol}")
        symbol_coverage.append(
            SymbolObservedCoverage(
                symbol=symbol,
                rows=claim.rows,
                complete_range=claim.complete_range,
                tick_size=claim.tick_size,
                lot_size=claim.lot_size,
                observed=_coverage(
                    declared_symbol_starts[symbol],
                    declared_symbol_ends[symbol],
                ),
            )
        )
    return PublicTradeDataset(
        rows=dataset_rows,
        observed=_coverage(
            min(part.observed_start_ns for part in parts),
            max(part.observed_end_inclusive_ns for part in parts),
        ),
        symbols=tuple(symbol_coverage),
        evidence_tier=evidence_tier,
        all_requested_ranges_complete=all_complete,
        ingestion_manifest_path=manifest_path,
        ingestion_manifest_sha256=expected_ingestion_sha,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha,
        part_paths=tuple(part.data_path for part in parts),
        raw_artifact_paths=verified_raw.paths,
        raw_manifest_paths=verified_raw.manifest_paths,
        raw_artifact_sha256s=tuple(sorted(verified_raw.sha256s)),
        row_bound=max_rows,
        canonical_order=canonical_order,
        _config=config,
        _requested_range=requested_range,
        _symbol_claims=MappingProxyType(dict(symbol_claims)),
        _parts=tuple(parts),
        _raw_pages_by_digest=verified_raw.pages_by_digest,
        _ordered_raw_pages=verified_raw.ordered_pages,
    )


def _normalized_batch(
    raw_batch: pa.RecordBatch,
    *,
    label: str,
) -> pa.RecordBatch:
    expected = get_schema("trades")
    try:
        arrays: list[pa.Array] = []
        for name in expected.names:
            field_index = raw_batch.schema.get_field_index(name)
            if field_index < 0:
                raise PublicDataError(f"{label} is missing column {name!r}")
            arrays.append(raw_batch.column(field_index))
        normalized = pa.RecordBatch.from_arrays(arrays, schema=expected)
        ensure_schema(normalized, "trades")
        return normalized
    except PublicDataError:
        raise
    except (pa.ArrowException, ValueError) as exc:
        raise PublicDataError(f"{label} has an unexpected trades schema: {exc}") from exc


def _iter_audited_physical_batches(
    dataset: PublicTradeDataset,
    *,
    batch_rows: int,
) -> Generator[pa.RecordBatch, None, _VerifiedStreamSummary]:
    """Yield each source batch once after immutable physical-order auditing."""
    validator = IncrementalQualityValidator(
        "trades",
        max_spread_bps=dataset._config.quality.max_spread_bps,
        max_silence_ns=dataset._config.quality.max_silence_ms * 1_000_000,
        row_chunk_size=batch_rows,
    )
    validator_finished = False
    lineage_index: _ExpectedLineageIndex | None = None
    raw_cache = _RawPageCache(
        dataset._raw_pages_by_digest,
        request_limit=dataset._config.data.request_limit,
        requested_range=dataset._requested_range,
    )
    actual_symbol_rows: dict[str, int] = {symbol: 0 for symbol in dataset._symbol_claims}
    actual_symbol_starts: dict[str, int] = {}
    actual_symbol_ends: dict[str, int] = {}
    total_rows = 0
    audited_schema = get_schema("trades").append(
        pa.field("__physical_ordinal", pa.int64(), nullable=False)
    )
    try:
        lineage_index = _ExpectedLineageIndex(dataset)
        for part in sorted(dataset._parts, key=lambda item: item.write_ordinal):
            _verify_file(part.data_path, part.data_sha256, "normalized Parquet part")
            part_rows = 0
            part_start: int | None = None
            part_end: int | None = None
            try:
                parquet = pq.ParquetFile(part.data_path)
                physical_batches = parquet.iter_batches(batch_size=batch_rows)
                for raw_batch in physical_batches:
                    if raw_batch.num_rows < 1:
                        continue
                    if raw_batch.num_rows > batch_rows:
                        raise PublicDataError(
                            f"Parquet emitted {raw_batch.num_rows} rows above batch bound "
                            f"{batch_rows}"
                        )
                    normalized = _normalized_batch(
                        raw_batch,
                        label=f"Parquet part {part.data_path}",
                    )
                    rows = cast(list[dict[str, Any]], normalized.to_pylist())
                    for local_index, row in enumerate(rows):
                        row_index = total_rows + local_index
                        symbol = str(row["symbol"])
                        venue = str(row["venue"])
                        timestamp = int(row["event_ts_ns"])
                        if symbol != part.symbol or venue != part.venue:
                            raise PublicDataError(
                                f"normalized row {row_index} does not match its Parquet "
                                "sidecar partition"
                            )
                        actual_date = (
                            datetime.fromtimestamp(timestamp // _NS_PER_SECOND, tz=UTC)
                            .date()
                            .isoformat()
                        )
                        if actual_date != part.partition_date:
                            raise PublicDataError(
                                f"normalized row {row_index} does not match its partition date"
                            )
                        if (
                            not dataset._requested_range[0]
                            <= timestamp
                            < dataset._requested_range[1]
                        ):
                            raise PublicDataError(
                                f"normalized row {row_index} is outside the requested range"
                            )
                        claim = dataset._symbol_claims.get(symbol)
                        if claim is None:
                            raise PublicDataError(
                                f"normalized row {row_index} has an unmanifested symbol {symbol}"
                            )
                        try:
                            tick_size = Decimal(str(row["tick_size"]))
                            lot_size = Decimal(str(row["lot_size"]))
                        except InvalidOperation as exc:
                            raise PublicDataError(
                                f"normalized row {row_index} has invalid scales for {symbol}"
                            ) from exc
                        if tick_size != claim.tick_size or lot_size != claim.lot_size:
                            raise PublicDataError(
                                f"normalized row {row_index} scales do not match manifest for "
                                f"{symbol}"
                            )
                        _validate_normalized_trade_row(
                            row,
                            row_index=row_index,
                            symbol_claims=dataset._symbol_claims,
                            raw_cache=raw_cache,
                        )
                        lineage_index.mark_seen(
                            symbol=symbol,
                            source_sha256=str(row["source_artifact_id"]),
                            trade_id=int(row["trade_id"]),
                        )
                        actual_symbol_rows[symbol] += 1
                        actual_symbol_starts[symbol] = min(
                            actual_symbol_starts.get(symbol, timestamp), timestamp
                        )
                        actual_symbol_ends[symbol] = max(
                            actual_symbol_ends.get(symbol, timestamp), timestamp
                        )
                        part_start = timestamp if part_start is None else min(part_start, timestamp)
                        part_end = timestamp if part_end is None else max(part_end, timestamp)
                    del rows
                    validator.update(normalized)
                    lineage_index.commit()
                    batch_count = normalized.num_rows
                    physical_start = total_rows
                    total_rows += batch_count
                    part_rows += batch_count
                    yield pa.RecordBatch.from_arrays(
                        [
                            *normalized.columns,
                            pa.array(
                                range(physical_start, physical_start + batch_count),
                                type=pa.int64(),
                            ),
                        ],
                        schema=audited_schema,
                    )
            except PublicDataError:
                raise
            except (OSError, pa.ArrowException, ValueError) as exc:
                raise PublicDataError(
                    f"cannot stream normalized Parquet part {part.data_path}: {exc}"
                ) from exc
            if part_rows != part.rows:
                raise PublicDataError(
                    f"materialized rows for Parquet part {part.data_path} do not match sidecar"
                )
            if part_start != part.observed_start_ns or part_end != part.observed_end_inclusive_ns:
                raise PublicDataError(
                    f"materialized coverage for Parquet part {part.data_path} does not match "
                    "sidecar"
                )
            _verify_file(part.data_path, part.data_sha256, "normalized Parquet part")

        validation = validator.finish()
        validator_finished = True
        if total_rows != dataset.rows:
            raise PublicDataError("streamed trades do not match manifested dataset rows")
        symbol_coverage: list[SymbolObservedCoverage] = []
        for symbol in sorted(dataset._symbol_claims):
            claim = dataset._symbol_claims[symbol]
            if actual_symbol_rows[symbol] != claim.rows:
                raise PublicDataError(f"materialized rows for {symbol} do not match coverage claim")
            if symbol not in actual_symbol_starts:
                raise PublicDataError(f"materialized trades contain no rows for {symbol}")
            symbol_coverage.append(
                SymbolObservedCoverage(
                    symbol=symbol,
                    rows=claim.rows,
                    complete_range=claim.complete_range,
                    tick_size=claim.tick_size,
                    lot_size=claim.lot_size,
                    observed=_coverage(actual_symbol_starts[symbol], actual_symbol_ends[symbol]),
                )
            )
        observed = _coverage(min(actual_symbol_starts.values()), max(actual_symbol_ends.values()))
        if observed != dataset.observed or tuple(symbol_coverage) != dataset.symbols:
            raise PublicDataError("materialized coverage does not match verified manifest metadata")
        missing_lineage, repeated_lineage = lineage_index.mismatch_counts()
        if (missing_lineage or repeated_lineage) and not (
            dataset._config.quality.fail_on_error and validation.has_errors
        ):
            raise PublicDataError(
                "normalized rows do not exactly cover downloader-selected raw trades: "
                f"missing={missing_lineage}, repeated={repeated_lineage}"
            )
        return _VerifiedStreamSummary(
            validation=validation,
            observed=observed,
            symbols=tuple(symbol_coverage),
        )
    finally:
        if not validator_finished:
            validator.close()
        if lineage_index is not None:
            lineage_index.close()


def _stream_verified_batches(
    dataset: PublicTradeDataset,
    *,
    batch_rows: int,
    memory_limit: str,
    temp_directory: str | Path | None,
) -> Generator[pa.RecordBatch, None, _VerifiedStreamSummary]:
    """Audit source order, then externally sort a fresh bounded research pass."""
    owned_temp: tempfile.TemporaryDirectory[str] | None = None
    if temp_directory is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="microstructure-public-sort-")
        sort_temp = Path(owned_temp.name).resolve()
    else:
        sort_temp = Path(temp_directory).resolve()
        if not sort_temp.is_dir():
            raise PublicDataError(f"DuckDB temporary directory does not exist: {sort_temp}")

    connection: duckdb.DuckDBPyConnection | None = None
    reader: pa.RecordBatchReader | None = None
    source_reader: pa.RecordBatchReader | None = None
    physical_source: _PhysicalAuditBatchSource | None = None
    field_names = get_schema("trades").names
    previous_order_key: tuple[str, str, int, int, int, str, int] | None = None
    emitted_rows = 0
    quoted_fields = ", ".join(f'"{name}"' for name in field_names)
    order_fields = ", ".join(f'"{name}" ASC' for name in dataset.canonical_order)
    query = (
        f"SELECT {quoted_fields}, __physical_ordinal "
        "FROM verified_source "
        f"ORDER BY {order_fields}, source_artifact_id ASC, __physical_ordinal ASC"
    )
    try:
        connection = duckdb.connect(database=":memory:")
        connection.execute("SET memory_limit = ?", [memory_limit])
        connection.execute("SET temp_directory = ?", [str(sort_temp)])
        connection.execute("SET threads = 1")
        connection.execute("SET preserve_insertion_order = false")

        audited_schema = get_schema("trades").append(
            pa.field("__physical_ordinal", pa.int64(), nullable=False)
        )
        physical_source = _PhysicalAuditBatchSource(
            _iter_audited_physical_batches(dataset, batch_rows=batch_rows)
        )
        source_reader = pa.RecordBatchReader.from_batches(audited_schema, physical_source)
        connection.register("verified_source", source_reader)
        reader = connection.execute(query).to_arrow_reader(batch_size=batch_rows)
        summary: _VerifiedStreamSummary | None = None
        for raw_batch in reader:
            if summary is None:
                summary = physical_source.summary
                if dataset._config.quality.fail_on_error and summary.validation.has_errors:
                    raise PublicDataError(
                        "public normalized trades failed quality validation with "
                        f"{summary.validation.error_count} error findings"
                    )
            if raw_batch.num_rows < 1:
                continue
            if raw_batch.num_rows > batch_rows:
                raise PublicDataError(
                    f"DuckDB emitted {raw_batch.num_rows} rows above batch bound {batch_rows}"
                )
            normalized = _normalized_batch(raw_batch, label="DuckDB canonical output")
            physical_ordinals = cast(
                list[int],
                raw_batch.column(
                    raw_batch.schema.get_field_index("__physical_ordinal")
                ).to_pylist(),
            )
            rows = cast(list[dict[str, Any]], normalized.to_pylist())
            for row, physical_ordinal in zip(rows, physical_ordinals, strict=True):
                order_key = (
                    str(row["venue"]),
                    str(row["symbol"]),
                    int(row["available_ts_ns"]),
                    int(row["event_ts_ns"]),
                    int(row["trade_id"]),
                    str(row["source_artifact_id"]),
                    int(physical_ordinal),
                )
                if previous_order_key is not None and order_key < previous_order_key:
                    raise PublicDataError("DuckDB output violated canonical trade order")
                previous_order_key = order_key
            del rows, physical_ordinals
            emitted_rows += normalized.num_rows
            yield normalized
        if emitted_rows != dataset.rows:
            raise PublicDataError("canonical stream rows do not match manifested dataset rows")
        if summary is None:
            summary = physical_source.summary
        return summary
    except duckdb.Error as exc:
        raise PublicDataError(f"cannot externally sort public trades: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
        if source_reader is not None:
            source_reader.close()
        if physical_source is not None:
            physical_source.close()
        if connection is not None:
            connection.close()
        if owned_temp is not None:
            owned_temp.cleanup()


def read_public_trades(
    config: ProjectConfig,
    ingestion_manifest_path: str | Path,
    *,
    ingestion_manifest_sha256: str,
    materialization_max_rows: int = 100_000,
) -> PublicTrades:
    """Compatibility materializer built on the bounded verified stream.

    ``materialization_max_rows`` is a separate finite safety guard.  It is
    checked against already-verified manifest metadata before DuckDB or PyArrow
    reads any normalized row, regardless of the configured ingestion cap.
    """
    dataset = verify_public_trade_dataset(
        config,
        ingestion_manifest_path,
        ingestion_manifest_sha256=ingestion_manifest_sha256,
    )
    if (
        isinstance(materialization_max_rows, bool)
        or not isinstance(materialization_max_rows, int)
        or materialization_max_rows < 1
    ):
        raise ValueError("materialization_max_rows must be a positive integer")
    if dataset.rows > materialization_max_rows:
        raise PublicDataError(
            f"verified public data has {dataset.rows} rows, above materialization guard "
            f"{materialization_max_rows}"
        )

    stream = dataset.iter_verified_batches(
        batch_rows=min(65_536, materialization_max_rows),
        memory_limit="256MB",
    )
    batches: list[pa.RecordBatch] = []
    try:
        for batch in stream:
            batches.append(batch)
        summary = stream.summary
    finally:
        stream.close()
    try:
        trades = pa.Table.from_batches(batches, schema=get_schema("trades"))
        ensure_schema(trades, "trades")
    except (pa.ArrowException, ValueError) as exc:
        raise PublicDataError(f"cannot materialize verified public trades: {exc}") from exc
    if trades.num_rows != dataset.rows:
        raise PublicDataError("materialized trades do not match manifested dataset rows")
    return PublicTrades(
        arrow_trades=trades,
        polars_trades=cast(pl.DataFrame, pl.from_arrow(trades)),
        observed=summary.observed,
        symbols=summary.symbols,
        evidence_tier=dataset.evidence_tier,
        all_requested_ranges_complete=dataset.all_requested_ranges_complete,
        ingestion_manifest_path=dataset.ingestion_manifest_path,
        ingestion_manifest_sha256=dataset.ingestion_manifest_sha256,
        dataset_manifest_path=dataset.dataset_manifest_path,
        dataset_manifest_sha256=dataset.dataset_manifest_sha256,
        part_paths=dataset.part_paths,
        raw_artifact_paths=dataset.raw_artifact_paths,
        raw_manifest_paths=dataset.raw_manifest_paths,
        raw_artifact_sha256s=dataset.raw_artifact_sha256s,
        validation=summary.validation,
        row_bound=dataset.row_bound,
        canonical_order=dataset.canonical_order,
    )
