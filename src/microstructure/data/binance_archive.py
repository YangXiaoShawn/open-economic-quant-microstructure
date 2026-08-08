"""Bounded acquisition and one-shot normalization of Binance daily trade archives.

The acquisition boundary deliberately stops after authenticating the exact ZIP
bytes and inspecting bounded ZIP metadata.  CSV rows are not opened until a
caller explicitly requests a normalized stream.  That separation lets a study
write an analysis lock before either held-out archive is exposed to research
code.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import stat
import struct
import tempfile
import time
import zipfile
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, cast
from urllib.parse import urlsplit

import pyarrow as pa  # type: ignore[import-untyped]
import requests

from microstructure.data.binance import RetryPolicy
from microstructure.data.evidence_budget import (
    EvidenceBudgetError,
    EvidenceReservation,
    RetainedEvidenceBudget,
)
from microstructure.data.schemas import SCHEMA_VERSION, table_from_records
from microstructure.data.storage import write_source_manifest
from microstructure.provenance import sha256_file, utc_now_iso

_DEFAULT_BASE_URL = "https://data.binance.vision"
_SAFE_SYMBOL = re.compile(r"^[A-Z0-9]{2,20}$")
_SAFE_ARCHIVE_NAME = re.compile(r"^[A-Z0-9]{2,20}-aggTrades-\d{4}-\d{2}-\d{2}\.zip$")
_CHECKSUM_LINE = re.compile(rb"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)(?:\r\n|\n)?")
_UNSIGNED_INTEGER = re.compile(rb"(?:0|[1-9][0-9]*)")
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_STRUCT = struct.Struct("<4s4H2LH")
_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_LOCAL_FILE_STRUCT = struct.Struct("<4s5H3L2H")
_MAX_EOCD_BYTES = 22 + 65_535
_MAX_ZIP_ENTRY_METADATA_BYTES = 256 * 1_024
_MAX_DECIMAL_FIELD_BYTES = 64
_MAX_INT64 = (1 << 63) - 1
_MICROSECOND_ARCHIVE_START = date(2025, 1, 1)


class BinanceArchiveError(RuntimeError):
    """Base failure for immutable Binance archive acquisition or parsing."""


class BinanceArchiveHTTPError(BinanceArchiveError):
    """Raised when a bounded public archive response cannot be acquired."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_exhausted = retry_exhausted


class BinanceArchivePayloadError(BinanceArchiveError):
    """Raised when archive bytes or CSV rows violate their frozen contract."""


ArchiveAcquisitionReasonCode = Literal[
    "CHECKSUM_CONTRACT",
    "ZIP_CONTRACT",
    "RESPONSE_SIZE_LIMIT",
]


class BinanceArchiveContractError(BinanceArchivePayloadError):
    """A deterministic raw archive/checksum contract failure before CSV open."""

    def __init__(self, message: str, *, reason_code: ArchiveAcquisitionReasonCode) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class _RetryableDownloadError(BinanceArchiveHTTPError):
    """Internal marker for one recoverable, already-evidenced HTTP attempt."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class _StreamingResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    url: str

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> object: ...


class _StreamingSession(Protocol):
    def get(self, url: str, *, timeout: float, stream: bool) -> _StreamingResponse: ...


@dataclass(frozen=True, slots=True)
class DailyArchiveRequest:
    """One exact official UTC-day archive plus normalization scales."""

    symbol: str
    date: date
    tick_size: Decimal
    lot_size: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or _SAFE_SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("archive symbol must be uppercase ASCII letters/digits")
        if type(self.date) is not date:
            raise ValueError("archive date must be a datetime.date")
        if not isinstance(self.tick_size, Decimal):
            raise ValueError("tick_size must be a Decimal")
        if (
            not self.tick_size.is_finite()
            or self.tick_size <= 0
            or not math.isfinite(float(self.tick_size))
            or float(self.tick_size) <= 0
        ):
            raise ValueError("tick_size must be a positive finite Decimal")
        if not isinstance(self.lot_size, Decimal):
            raise ValueError("lot_size must be a Decimal")
        if (
            not self.lot_size.is_finite()
            or self.lot_size <= 0
            or not math.isfinite(float(self.lot_size))
            or float(self.lot_size) <= 0
        ):
            raise ValueError("lot_size must be a positive finite Decimal")

    @property
    def archive_name(self) -> str:
        return f"{self.symbol}-aggTrades-{self.date.isoformat()}.zip"

    @property
    def member_name(self) -> str:
        return self.archive_name.removesuffix(".zip") + ".csv"

    @property
    def continuity_id(self) -> str:
        return f"binance_spot:{self.symbol}:{self.date.isoformat()}"


@dataclass(frozen=True, slots=True)
class ArchiveDownloadLimits:
    """Hard transport, expansion, and record-boundary ceilings."""

    max_compressed_bytes: int
    max_uncompressed_bytes: int
    max_checksum_bytes: int = 4_096
    transfer_chunk_bytes: int = 64 * 1_024
    max_csv_line_bytes: int = 16 * 1_024

    def __post_init__(self) -> None:
        values = (
            self.max_compressed_bytes,
            self.max_uncompressed_bytes,
            self.max_checksum_bytes,
            self.transfer_chunk_bytes,
            self.max_csv_line_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
        ):
            raise ValueError("all archive byte limits must be positive integers")


@dataclass(frozen=True, slots=True)
class RawArchiveArtifact:
    """Immutable descriptor for an exact public response body."""

    kind: Literal["archive_zip", "archive_checksum", "rejected_prefix"]
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str
    bytes: int
    source_uri: str


@dataclass(frozen=True, slots=True)
class DailyArchiveSummary:
    """Non-economic coverage facts available only after full stream exhaustion."""

    symbol: str
    date: str
    rows: int
    first_trade_id: int
    last_trade_id: int
    first_event_ts_ns: int
    last_event_ts_ns: int
    compressed_bytes: int
    expanded_bytes: int
    source_archive_sha256: str
    member_name: str
    continuity_id: str


@dataclass(frozen=True, slots=True)
class _ZipDescriptor:
    member_name: str
    declared_uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class _ZipDirectoryBounds:
    offset: int
    bytes: int


@dataclass(frozen=True, slots=True)
class _DownloadedBody:
    temporary_path: Path
    sha256: str
    bytes: int
    response_headers: Mapping[str, str]
    downloaded_at_utc: str
    evidence_reservations: tuple[EvidenceReservation, ...]


@dataclass(frozen=True, slots=True)
class AcquiredDailyArchive:
    """Checksum-authenticated raw bytes whose CSV has not yet been opened."""

    request: DailyArchiveRequest
    archive_artifact: RawArchiveArtifact
    checksum_artifact: RawArchiveArtifact
    upstream_sha256: str
    declared_uncompressed_bytes: int
    limits: ArchiveDownloadLimits
    requires_member_open_guard: bool = False

    def iter_normalized_batches(
        self,
        *,
        batch_rows: int = 65_536,
        before_member_open: Callable[[], None] | None = None,
    ) -> DailyArchiveTradeStream:
        """Create a fresh one-shot normalized stream over the authenticated ZIP.

        ``before_member_open`` is a fail-closed held-out-data guard.  It runs
        after bounded ZIP-directory validation and immediately before the CSV
        member is opened.  A raised exception therefore exposes zero member
        bytes.  Acquisition callers normally leave it unset; prospective
        research pipelines use it to revalidate their durable analysis lock
        at the actual economic-data boundary.

        Handles reconstructed for a frozen held-out role set
        ``requires_member_open_guard``.  Such a stream refuses to advance at
        all unless this callback is supplied; the generic archive adapter and
        development-date handles retain their backward-compatible default.
        """
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        return DailyArchiveTradeStream(
            _stream_normalized_batches(
                self,
                batch_rows=batch_rows,
                before_member_open=before_member_open,
            )
        )


class DailyArchiveTradeStream(Iterator[pa.RecordBatch]):
    """One-shot RecordBatch iterator with a terminal-only coverage summary."""

    def __init__(
        self,
        generator: Generator[pa.RecordBatch, None, DailyArchiveSummary],
    ) -> None:
        self._generator = generator
        self._summary: DailyArchiveSummary | None = None
        self._closed = False

    def __iter__(self) -> DailyArchiveTradeStream:
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._closed:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration as stop:
            self._closed = True
            self._summary = cast(DailyArchiveSummary, stop.value)
            raise
        except BaseException:
            self._closed = True
            raise

    @property
    def summary(self) -> DailyArchiveSummary:
        if self._summary is None:
            raise RuntimeError("archive summary is unavailable before full stream exhaustion")
        return self._summary

    def close(self) -> None:
        if not self._closed:
            self._generator.close()
            self._closed = True


def _day_bounds_ns(value: date) -> tuple[int, int]:
    start = datetime(value.year, value.month, value.day, tzinfo=UTC)
    end = start + timedelta(days=1)
    return int(start.timestamp()) * 1_000_000_000, int(end.timestamp()) * 1_000_000_000


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    allowed = {
        "content-length",
        "content-type",
        "etag",
        "last-modified",
        "retry-after",
    }
    return {str(key): str(value) for key, value in headers.items() if str(key).lower() in allowed}


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return None


def _content_length(headers: Mapping[str, str]) -> int | None:
    raw = _header(headers, "content-length")
    if raw is None:
        return None
    if not raw.isascii() or not raw.isdecimal():
        raise BinanceArchivePayloadError("response Content-Length must be an unsigned integer")
    return int(raw)


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = _header(headers, "retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _validate_base_url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
        raise ValueError("archive base_url must be an HTTPS origin without path/query")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("archive base_url must not contain credentials, query, or fragment")
    return normalized


def _archive_urls(base_url: str, request: DailyArchiveRequest) -> tuple[str, str]:
    archive = f"{base_url}/data/spot/daily/aggTrades/{request.symbol}/{request.archive_name}"
    return archive, f"{archive}.CHECKSUM"


def _release_evidence_reservations(
    reservations: tuple[EvidenceReservation, ...] | list[EvidenceReservation],
) -> None:
    for reservation in reservations:
        if reservation.active:
            reservation.release()


def _commit_evidence_reservations(
    reservations: tuple[EvidenceReservation, ...] | list[EvidenceReservation],
) -> None:
    for reservation in reservations:
        reservation.commit()


def _discard_download(body: _DownloadedBody) -> None:
    body.temporary_path.unlink(missing_ok=True)
    _release_evidence_reservations(body.evidence_reservations)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temp(
    temporary: Path,
    *,
    destination_directory: Path,
    suffix: str,
    destination_name: str | None = None,
    kind: Literal["archive_zip", "archive_checksum", "rejected_prefix"],
    source: str,
    source_uri: str,
    downloaded_at_utc: str,
    request: DailyArchiveRequest,
    sha256: str,
    response_headers: Mapping[str, str],
    upstream_checksum_sha256: str | None,
    retained_evidence_budget: RetainedEvidenceBudget | None = None,
    evidence_reservations: tuple[EvidenceReservation, ...] = (),
) -> RawArchiveArtifact:
    destination = destination_directory / (destination_name or f"{sha256}{suffix}")
    if retained_evidence_budget is None and evidence_reservations:
        raise BinanceArchiveError("download reservations require a retained-evidence budget")
    if retained_evidence_budget is not None:
        retained_evidence_budget.assert_contains(temporary)
        retained_evidence_budget.assert_contains(destination)
    transaction = (
        retained_evidence_budget.write_transaction()
        if retained_evidence_budget is not None
        else nullcontext()
    )
    created_destination = False
    try:
        with transaction:
            destination_directory.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != sha256:
                    raise BinanceArchivePayloadError(f"content-address collision at {destination}")
                temporary.unlink(missing_ok=True)
                _release_evidence_reservations(evidence_reservations)
            else:
                os.replace(temporary, destination)
                _fsync_directory(destination_directory)
                created_destination = True
            start_ns, end_ns = _day_bounds_ns(request.date)
            manifest_path, manifest_sha = write_source_manifest(
                destination,
                source=source,
                source_uri=source_uri,
                downloaded_at_utc=downloaded_at_utc,
                requested_start_ns=start_ns,
                requested_end_ns=end_ns,
                upstream_checksum_sha256=upstream_checksum_sha256,
                response_headers=response_headers,
                retained_evidence_budget=retained_evidence_budget,
            )
            if created_destination:
                _commit_evidence_reservations(evidence_reservations)
            return RawArchiveArtifact(
                kind=kind,
                path=destination,
                manifest_path=manifest_path,
                sha256=sha256,
                manifest_sha256=manifest_sha,
                bytes=destination.stat().st_size,
                source_uri=source_uri,
            )
    except BaseException:
        temporary.unlink(missing_ok=True)
        if created_destination:
            destination.unlink(missing_ok=True)
        _release_evidence_reservations(evidence_reservations)
        raise


def _publish_rejected(
    temporary: Path,
    *,
    raw_root: Path,
    request: DailyArchiveRequest,
    source_uri: str,
    downloaded_at_utc: str,
    response_headers: Mapping[str, str],
    reason: str,
    suffix: str,
    attempt_number: int | None = None,
    retained_evidence_budget: RetainedEvidenceBudget | None = None,
    evidence_reservations: tuple[EvidenceReservation, ...] = (),
) -> RawArchiveArtifact:
    digest = sha256_file(temporary)
    headers = dict(response_headers)
    headers.update(
        {
            "x-local-capture-status": "rejected_bounded_prefix",
            "x-local-rejection-reason": reason,
            "x-local-captured-bytes": str(temporary.stat().st_size),
        }
    )
    if attempt_number is not None:
        headers["x-local-download-attempt"] = str(attempt_number)
    return _publish_temp(
        temporary,
        destination_directory=(
            raw_root
            / "binance_spot"
            / "daily_agg_trades_archive_rejected"
            / request.symbol
            / request.date.isoformat()
        ),
        suffix=suffix,
        kind="rejected_prefix",
        source="binance_spot_daily_aggtrades_archive_rejected",
        source_uri=source_uri,
        downloaded_at_utc=downloaded_at_utc,
        request=request,
        sha256=digest,
        response_headers=headers,
        upstream_checksum_sha256=None,
        retained_evidence_budget=retained_evidence_budget,
        evidence_reservations=evidence_reservations,
    )


def _bounded_download_once(
    session: _StreamingSession,
    *,
    url: str,
    raw_root: Path,
    request: DailyArchiveRequest,
    byte_limit: int,
    chunk_bytes: int,
    timeout_seconds: float,
    rejected_suffix: str,
    attempt_number: int,
    retained_evidence_budget: RetainedEvidenceBudget | None,
) -> _DownloadedBody:
    work = raw_root / "binance_spot" / ".archive_downloads"
    if retained_evidence_budget is not None:
        retained_evidence_budget.assert_contains(work)
    work.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=work, prefix=".download-", suffix=".tmp")
    temporary = Path(temporary_name)
    response: _StreamingResponse | None = None
    downloaded_at = utc_now_iso()
    safe_headers: dict[str, str] = {}
    digest = hashlib.sha256()
    captured = 0
    error: BaseException | None = None
    evidence_reservations: list[EvidenceReservation] = []

    def write_chunk(sink: Any, chunk: bytes) -> None:
        reservation = (
            retained_evidence_budget.reserve(
                len(chunk),
                label=f"raw Binance archive response from {url}",
            )
            if retained_evidence_budget is not None
            else None
        )
        if reservation is not None:
            evidence_reservations.append(reservation)
        try:
            sink.write(chunk)
        except BaseException:
            if reservation is not None:
                evidence_reservations.pop()
                reservation.release()
            raise

    try:
        with os.fdopen(descriptor, "wb") as sink:
            descriptor = -1
            try:
                response = session.get(url, timeout=timeout_seconds, stream=True)
            except requests.RequestException as exc:
                raise _RetryableDownloadError(f"GET {url} failed before a response") from exc
            safe_headers = _safe_headers(response.headers)
            if str(response.url) != url:
                raise BinanceArchiveHTTPError("archive response redirected away from exact URL")
            if response.status_code != 200:
                status_code = response.status_code
                message = f"GET {url} returned HTTP {status_code}"
                if status_code in {408, 418, 429} or 500 <= status_code <= 599:
                    retry_after = (
                        _retry_after_seconds(response.headers)
                        if status_code in {418, 429}
                        else None
                    )
                    raise _RetryableDownloadError(
                        message,
                        retry_after_seconds=retry_after,
                    )
                raise BinanceArchiveHTTPError(message, status_code=status_code)
            declared = _content_length(response.headers)
            if declared is not None and declared > byte_limit:
                raise BinanceArchivePayloadError(
                    f"response Content-Length {declared} exceeds byte ceiling {byte_limit}"
                )
            try:
                chunks = response.iter_content(chunk_size=chunk_bytes)
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise BinanceArchivePayloadError("streaming response emitted non-bytes")
                    if not chunk:
                        continue
                    remaining = byte_limit - captured
                    if len(chunk) > remaining:
                        prefix = chunk[:remaining]
                        if prefix:
                            write_chunk(sink, prefix)
                            digest.update(prefix)
                            captured += len(prefix)
                        raise BinanceArchivePayloadError(
                            f"response body exceeds byte ceiling {byte_limit}"
                        )
                    write_chunk(sink, chunk)
                    digest.update(chunk)
                    captured += len(chunk)
            except requests.RequestException as exc:
                raise _RetryableDownloadError(
                    f"GET {url} body interrupted after {captured} bytes"
                ) from exc
            if declared is not None and captured < declared:
                raise _RetryableDownloadError(
                    f"GET {url} body truncated at {captured} of {declared} Content-Length bytes"
                )
            if declared is not None and captured > declared:
                raise BinanceArchivePayloadError(
                    f"GET {url} body length {captured} exceeds Content-Length {declared}"
                )
            sink.flush()
            os.fsync(sink.fileno())
    except BaseException as exc:
        error = exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if response is not None:
            try:
                response.close()
            except BaseException as exc:
                if error is None:
                    error = BinanceArchiveHTTPError(f"GET {url} response could not be closed")
                    error.__cause__ = exc

    if error is not None:
        try:
            _publish_rejected(
                temporary,
                raw_root=raw_root,
                request=request,
                source_uri=url,
                downloaded_at_utc=downloaded_at,
                response_headers=safe_headers,
                reason=str(error),
                suffix=rejected_suffix,
                attempt_number=attempt_number,
                retained_evidence_budget=retained_evidence_budget,
                evidence_reservations=tuple(evidence_reservations),
            )
        except BaseException as evidence_error:
            temporary.unlink(missing_ok=True)
            _release_evidence_reservations(evidence_reservations)
            if isinstance(evidence_error, EvidenceBudgetError):
                raise
            if not isinstance(evidence_error, Exception):
                raise
            raise BinanceArchiveError(
                f"could not retain rejected download attempt {attempt_number}"
            ) from evidence_error
        raise error
    return _DownloadedBody(
        temporary_path=temporary,
        sha256=digest.hexdigest(),
        bytes=captured,
        response_headers=safe_headers,
        downloaded_at_utc=downloaded_at,
        evidence_reservations=tuple(evidence_reservations),
    )


def _bounded_download(
    session: _StreamingSession,
    *,
    url: str,
    raw_root: Path,
    request: DailyArchiveRequest,
    byte_limit: int,
    chunk_bytes: int,
    timeout_seconds: float,
    rejected_suffix: str,
    retry_policy: RetryPolicy,
    sleep: Callable[[float], None],
    random_value: Callable[[], float],
    retained_evidence_budget: RetainedEvidenceBudget | None,
) -> _DownloadedBody:
    attempts = retry_policy.max_retries + 1
    for attempt_index in range(attempts):
        try:
            return _bounded_download_once(
                session,
                url=url,
                raw_root=raw_root,
                request=request,
                byte_limit=byte_limit,
                chunk_bytes=chunk_bytes,
                timeout_seconds=timeout_seconds,
                rejected_suffix=rejected_suffix,
                attempt_number=attempt_index + 1,
                retained_evidence_budget=retained_evidence_budget,
            )
        except _RetryableDownloadError as error:
            if attempt_index >= retry_policy.max_retries:
                error.add_note(f"exhausted {attempts} bounded download attempts")
                error.retry_exhausted = True
                raise
            if error.retry_after_seconds is not None:
                delay = error.retry_after_seconds
            else:
                exponential_cap = min(
                    retry_policy.max_delay_seconds,
                    retry_policy.base_delay_seconds * (2**attempt_index),
                )
                jitter = random_value()
                if (
                    isinstance(jitter, bool)
                    or not isinstance(jitter, (int, float))
                    or not math.isfinite(jitter)
                    or not 0 <= jitter <= 1
                ):
                    raise ValueError("random_value must return a finite number in [0, 1]") from None
                delay = exponential_cap * jitter
            sleep(delay)
    raise AssertionError("archive retry loop exhausted without a terminal result")


def _read_bounded_file(path: Path, *, byte_limit: int) -> bytes:
    with path.open("rb") as source:
        content = source.read(byte_limit + 1)
    if len(content) > byte_limit:
        raise BinanceArchivePayloadError(f"artifact exceeds read ceiling {byte_limit}")
    return content


def _parse_checksum(content: bytes, *, archive_name: str) -> str:
    match = _CHECKSUM_LINE.fullmatch(content)
    if match is None:
        raise BinanceArchivePayloadError(
            "archive CHECKSUM must be one lowercase SHA-256 and exact basename"
        )
    digest = match.group(1).decode("ascii")
    filename = match.group(2).decode("ascii")
    if filename != archive_name or _SAFE_ARCHIVE_NAME.fullmatch(filename) is None:
        raise BinanceArchivePayloadError("archive CHECKSUM names an unexpected file")
    return digest


def _preflight_eocd_handle(source: BinaryIO, size: int) -> _ZipDirectoryBounds:
    if size < _EOCD_STRUCT.size:
        raise BinanceArchivePayloadError("archive is too small to contain a ZIP directory")
    tail_size = min(size, _MAX_EOCD_BYTES)
    source.seek(size - tail_size)
    tail = source.read(tail_size)
    offset = tail.rfind(_EOCD_SIGNATURE)
    if offset < 0 or len(tail) - offset < _EOCD_STRUCT.size:
        raise BinanceArchivePayloadError("archive ZIP end-of-directory record is missing")
    values = _EOCD_STRUCT.unpack_from(tail, offset)
    _, disk, central_disk, entries_disk, entries_total, central_bytes, central_offset, comment = (
        values
    )
    absolute_offset = size - tail_size + offset
    if absolute_offset + _EOCD_STRUCT.size + comment != size:
        raise BinanceArchivePayloadError("archive ZIP has trailing or malformed directory bytes")
    if disk != 0 or central_disk != 0 or entries_disk != 1 or entries_total != 1:
        raise BinanceArchivePayloadError("archive ZIP must contain exactly one single-disk member")
    if central_bytes == 0 or central_bytes > _MAX_ZIP_ENTRY_METADATA_BYTES:
        raise BinanceArchivePayloadError(
            "archive ZIP central directory exceeds its metadata byte ceiling"
        )
    if central_offset + central_bytes != absolute_offset:
        raise BinanceArchivePayloadError("archive ZIP central-directory bounds are invalid")
    return _ZipDirectoryBounds(offset=central_offset, bytes=central_bytes)


def _validate_local_file_header_handle(
    source: BinaryIO,
    *,
    info: zipfile.ZipInfo,
    expected_member: str,
    central_offset: int,
) -> None:
    header_offset = int(info.header_offset)
    if header_offset < 0 or header_offset + _LOCAL_FILE_STRUCT.size > central_offset:
        raise BinanceArchivePayloadError("archive ZIP local-header bounds are invalid")
    source.seek(header_offset)
    header = source.read(_LOCAL_FILE_STRUCT.size)
    if len(header) != _LOCAL_FILE_STRUCT.size:
        raise BinanceArchivePayloadError("archive ZIP local header is truncated")
    (
        signature,
        _version,
        flags,
        compression,
        _modified_time,
        _modified_date,
        _crc32,
        _compressed_bytes,
        _uncompressed_bytes,
        filename_bytes,
        extra_bytes,
    ) = _LOCAL_FILE_STRUCT.unpack(header)
    if signature != _LOCAL_FILE_SIGNATURE:
        raise BinanceArchivePayloadError("archive ZIP local-header signature is invalid")
    metadata_bytes = filename_bytes + extra_bytes
    if metadata_bytes > _MAX_ZIP_ENTRY_METADATA_BYTES:
        raise BinanceArchivePayloadError(
            "archive ZIP local header exceeds its metadata byte ceiling"
        )
    if header_offset + _LOCAL_FILE_STRUCT.size + metadata_bytes > central_offset:
        raise BinanceArchivePayloadError("archive ZIP local-header bounds are invalid")
    local_name = source.read(filename_bytes)
    if local_name != expected_member.encode("ascii"):
        raise BinanceArchivePayloadError("archive ZIP local member path/name is invalid")
    if flags != info.flag_bits or compression != info.compress_type:
        raise BinanceArchivePayloadError("archive ZIP local and central metadata disagree")


def _validate_zip_structure_handle(
    source: BinaryIO,
    size: int,
    *,
    expected_member: str,
    max_uncompressed_bytes: int,
) -> _ZipDescriptor:
    try:
        directory = _preflight_eocd_handle(source, size)
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) != 1:
                raise BinanceArchivePayloadError("archive ZIP must contain exactly one member")
            info = infos[0]
            mode = info.external_attr >> 16
            if (
                info.filename != expected_member
                or Path(info.filename).name != info.filename
                or "\\" in info.filename
                or info.is_dir()
            ):
                raise BinanceArchivePayloadError("archive ZIP member path/name is invalid")
            if stat.S_ISLNK(mode) or info.flag_bits & 0x1:
                raise BinanceArchivePayloadError(
                    "archive ZIP member must be regular and unencrypted"
                )
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise BinanceArchivePayloadError("archive ZIP uses an unsupported compression type")
            if info.file_size < 1 or info.file_size > max_uncompressed_bytes:
                raise BinanceArchiveContractError(
                    "archive member declared uncompressed bytes outside configured ceiling",
                    reason_code="RESPONSE_SIZE_LIMIT",
                )
            _validate_local_file_header_handle(
                source,
                info=info,
                expected_member=expected_member,
                central_offset=directory.offset,
            )
            return _ZipDescriptor(
                member_name=info.filename,
                declared_uncompressed_bytes=int(info.file_size),
            )
    except BinanceArchivePayloadError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BinanceArchivePayloadError("cannot inspect archive ZIP structure") from exc


def _validate_zip_structure(
    path: Path,
    *,
    expected_member: str,
    max_uncompressed_bytes: int,
) -> _ZipDescriptor:
    with path.open("rb") as source:
        return _validate_zip_structure_handle(
            source,
            os.fstat(source.fileno()).st_size,
            expected_member=expected_member,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )


class BinanceArchiveClient:
    """HTTP boundary for official checksum-authenticated daily Spot archives."""

    def __init__(
        self,
        *,
        session: _StreamingSession | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        retained_evidence_budget: RetainedEvidenceBudget | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.session = (
            session
            if session is not None
            else cast(_StreamingSession, cast(object, requests.Session()))
        )
        self.base_url = _validate_base_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._random_value = random_value
        self.retained_evidence_budget = retained_evidence_budget

    def acquire(
        self,
        request: DailyArchiveRequest,
        *,
        raw_root: str | Path,
        limits: ArchiveDownloadLimits,
    ) -> AcquiredDailyArchive:
        """Acquire exact raw bytes without opening the archive CSV member."""
        destination_root = Path(raw_root).resolve()
        archive_url, checksum_url = _archive_urls(self.base_url, request)
        try:
            checksum_body = _bounded_download(
                self.session,
                url=checksum_url,
                raw_root=destination_root,
                request=request,
                byte_limit=limits.max_checksum_bytes,
                chunk_bytes=min(limits.transfer_chunk_bytes, limits.max_checksum_bytes),
                timeout_seconds=self.timeout_seconds,
                rejected_suffix=".CHECKSUM.rejected",
                retry_policy=self.retry_policy,
                sleep=self._sleep,
                random_value=self._random_value,
                retained_evidence_budget=self.retained_evidence_budget,
            )
        except BinanceArchiveContractError:
            raise
        except BinanceArchivePayloadError as exc:
            raise BinanceArchiveContractError(
                str(exc),
                reason_code="RESPONSE_SIZE_LIMIT",
            ) from exc
        try:
            checksum_directory = (
                destination_root
                / "binance_spot"
                / "daily_agg_trades_archive_checksums"
                / request.symbol
                / request.date.isoformat()
            )
            checksum_destination = checksum_directory / f"{request.archive_name}.CHECKSUM"
            if (
                checksum_destination.exists()
                and sha256_file(checksum_destination) != checksum_body.sha256
            ):
                _publish_rejected(
                    checksum_body.temporary_path,
                    raw_root=destination_root,
                    request=request,
                    source_uri=checksum_url,
                    downloaded_at_utc=checksum_body.downloaded_at_utc,
                    response_headers=checksum_body.response_headers,
                    reason="official CHECKSUM basename already contains different immutable bytes",
                    suffix=".CHECKSUM.rejected",
                    retained_evidence_budget=self.retained_evidence_budget,
                    evidence_reservations=checksum_body.evidence_reservations,
                )
                raise BinanceArchivePayloadError(
                    "official CHECKSUM basename collides with different immutable bytes"
                )
            checksum_artifact = _publish_temp(
                checksum_body.temporary_path,
                destination_directory=checksum_directory,
                suffix=".CHECKSUM",
                destination_name=f"{request.archive_name}.CHECKSUM",
                kind="archive_checksum",
                source="binance_spot_daily_aggtrades_archive_checksum",
                source_uri=checksum_url,
                downloaded_at_utc=checksum_body.downloaded_at_utc,
                request=request,
                sha256=checksum_body.sha256,
                response_headers=checksum_body.response_headers,
                upstream_checksum_sha256=None,
                retained_evidence_budget=self.retained_evidence_budget,
                evidence_reservations=checksum_body.evidence_reservations,
            )
        except BaseException:
            _discard_download(checksum_body)
            raise
        try:
            upstream_sha = _parse_checksum(
                _read_bounded_file(checksum_artifact.path, byte_limit=limits.max_checksum_bytes),
                archive_name=request.archive_name,
            )
        except BinanceArchivePayloadError as exc:
            raise BinanceArchiveContractError(
                str(exc),
                reason_code="CHECKSUM_CONTRACT",
            ) from exc

        try:
            archive_body = _bounded_download(
                self.session,
                url=archive_url,
                raw_root=destination_root,
                request=request,
                byte_limit=limits.max_compressed_bytes,
                chunk_bytes=limits.transfer_chunk_bytes,
                timeout_seconds=self.timeout_seconds,
                rejected_suffix=".zip.rejected",
                retry_policy=self.retry_policy,
                sleep=self._sleep,
                random_value=self._random_value,
                retained_evidence_budget=self.retained_evidence_budget,
            )
        except BinanceArchivePayloadError as exc:
            raise BinanceArchiveContractError(
                str(exc),
                reason_code="RESPONSE_SIZE_LIMIT",
            ) from exc
        try:
            if archive_body.sha256 != upstream_sha:
                _publish_rejected(
                    archive_body.temporary_path,
                    raw_root=destination_root,
                    request=request,
                    source_uri=archive_url,
                    downloaded_at_utc=archive_body.downloaded_at_utc,
                    response_headers=archive_body.response_headers,
                    reason=(
                        f"archive SHA-256 {archive_body.sha256} disagrees with official {upstream_sha}"
                    ),
                    suffix=".zip.rejected",
                    retained_evidence_budget=self.retained_evidence_budget,
                    evidence_reservations=archive_body.evidence_reservations,
                )
                raise BinanceArchiveContractError(
                    "archive SHA-256 disagrees with official CHECKSUM",
                    reason_code="CHECKSUM_CONTRACT",
                )

            archive_directory = (
                destination_root
                / "binance_spot"
                / "daily_agg_trades_archive"
                / request.symbol
                / request.date.isoformat()
            )
            official_destination = archive_directory / request.archive_name
            if (
                official_destination.exists()
                and sha256_file(official_destination) != archive_body.sha256
            ):
                _publish_rejected(
                    archive_body.temporary_path,
                    raw_root=destination_root,
                    request=request,
                    source_uri=archive_url,
                    downloaded_at_utc=archive_body.downloaded_at_utc,
                    response_headers=archive_body.response_headers,
                    reason="official archive basename already contains different immutable bytes",
                    suffix=".zip.rejected",
                    retained_evidence_budget=self.retained_evidence_budget,
                    evidence_reservations=archive_body.evidence_reservations,
                )
                raise BinanceArchivePayloadError(
                    "official archive basename collides with different immutable bytes"
                )
            archive_artifact = _publish_temp(
                archive_body.temporary_path,
                destination_directory=archive_directory,
                suffix=".zip",
                destination_name=request.archive_name,
                kind="archive_zip",
                source="binance_spot_daily_aggtrades_archive",
                source_uri=archive_url,
                downloaded_at_utc=archive_body.downloaded_at_utc,
                request=request,
                sha256=archive_body.sha256,
                response_headers=archive_body.response_headers,
                upstream_checksum_sha256=upstream_sha,
                retained_evidence_budget=self.retained_evidence_budget,
                evidence_reservations=archive_body.evidence_reservations,
            )
        except BaseException:
            _discard_download(archive_body)
            raise
        try:
            zip_descriptor = _validate_zip_structure(
                archive_artifact.path,
                expected_member=request.member_name,
                max_uncompressed_bytes=limits.max_uncompressed_bytes,
            )
        except BinanceArchiveContractError:
            raise
        except BinanceArchivePayloadError as exc:
            if isinstance(exc.__cause__, OSError):
                raise
            raise BinanceArchiveContractError(
                str(exc),
                reason_code="ZIP_CONTRACT",
            ) from exc
        return AcquiredDailyArchive(
            request=request,
            archive_artifact=archive_artifact,
            checksum_artifact=checksum_artifact,
            upstream_sha256=upstream_sha,
            declared_uncompressed_bytes=zip_descriptor.declared_uncompressed_bytes,
            limits=limits,
        )


def _parse_unsigned(value: bytes, *, label: str) -> int:
    if len(value) > 19:
        raise BinanceArchivePayloadError(f"{label} exceeds signed int64")
    if _UNSIGNED_INTEGER.fullmatch(value) is None:
        raise BinanceArchivePayloadError(f"{label} must be a canonical unsigned integer")
    parsed = int(value)
    if parsed > _MAX_INT64:
        raise BinanceArchivePayloadError(f"{label} exceeds signed int64")
    return parsed


def _parse_boolean(value: bytes, *, label: str) -> bool:
    lowered = value.lower()
    if lowered == b"true":
        return True
    if lowered == b"false":
        return False
    raise BinanceArchivePayloadError(f"{label} must be true or false")


def _scaled_decimal(value: bytes, *, quantum: Decimal, label: str) -> tuple[Decimal, int]:
    if len(value) > _MAX_DECIMAL_FIELD_BYTES:
        raise BinanceArchivePayloadError(f"{label} exceeds its field byte ceiling")
    try:
        text = value.decode("ascii")
        decimal = Decimal(text)
        if not decimal.is_finite() or decimal <= 0:
            raise BinanceArchivePayloadError(f"{label} must be positive and finite")
        scaled = decimal / quantum
        integral = scaled.to_integral_value()
    except (UnicodeDecodeError, InvalidOperation, ZeroDivisionError) as exc:
        raise BinanceArchivePayloadError(f"{label} is not a valid decimal") from exc
    if scaled != integral:
        raise BinanceArchivePayloadError(f"{label} is not aligned to declared scale")
    integer = int(integral)
    if integer < 1 or integer > _MAX_INT64:
        raise BinanceArchivePayloadError(f"{label} scaled value is outside signed int64")
    return decimal, integer


def _line_chunks(
    source: Any,
    *,
    max_uncompressed_bytes: int,
    chunk_bytes: int,
    max_line_bytes: int,
) -> Generator[bytes, None, int]:
    pending = b""
    expanded = 0
    while True:
        chunk = source.read(chunk_bytes)
        if not isinstance(chunk, bytes):
            raise BinanceArchivePayloadError("archive member emitted non-bytes")
        if not chunk:
            break
        expanded += len(chunk)
        if expanded > max_uncompressed_bytes:
            raise BinanceArchivePayloadError("archive expansion exceeds configured byte ceiling")
        combined = pending + chunk
        pieces = combined.split(b"\n")
        pending = pieces.pop()
        if len(pending) > max_line_bytes:
            raise BinanceArchivePayloadError("archive CSV line exceeds configured byte ceiling")
        for line in pieces:
            if line.endswith(b"\r"):
                line = line[:-1]
            if len(line) > max_line_bytes:
                raise BinanceArchivePayloadError("archive CSV line exceeds configured byte ceiling")
            yield line
    if pending:
        if pending.endswith(b"\r"):
            pending = pending[:-1]
        if len(pending) > max_line_bytes:
            raise BinanceArchivePayloadError("archive CSV line exceeds configured byte ceiling")
        yield pending
    return expanded


def _record_from_line(
    line: bytes,
    *,
    row_number: int,
    acquired: AcquiredDailyArchive,
    previous_trade_id: int | None,
    previous_event_ts_ns: int | None,
    start_ns: int,
    end_ns: int,
) -> tuple[dict[str, object], int, int]:
    fields = line.split(b",")
    if len(fields) != 8:
        raise BinanceArchivePayloadError(
            f"archive CSV row {row_number} must contain exactly 8 fields"
        )
    aggregate_id = _parse_unsigned(fields[0], label=f"row {row_number} aggregate trade ID")
    first_trade_id = _parse_unsigned(fields[3], label=f"row {row_number} first trade ID")
    last_trade_id = _parse_unsigned(fields[4], label=f"row {row_number} last trade ID")
    if first_trade_id > last_trade_id:
        raise BinanceArchivePayloadError(
            f"archive CSV row {row_number} first trade ID exceeds last trade ID"
        )
    if previous_trade_id is not None and aggregate_id != previous_trade_id + 1:
        raise BinanceArchivePayloadError(
            f"archive aggregate trade IDs are noncontiguous at row {row_number}"
        )
    raw_timestamp = _parse_unsigned(fields[5], label=f"row {row_number} timestamp")
    multiplier = 1_000 if acquired.request.date >= _MICROSECOND_ARCHIVE_START else 1_000_000
    if raw_timestamp > _MAX_INT64 // multiplier:
        raise BinanceArchivePayloadError(f"archive CSV row {row_number} timestamp overflows ns")
    event_ts_ns = raw_timestamp * multiplier
    if not start_ns <= event_ts_ns < end_ns:
        raise BinanceArchivePayloadError(
            f"archive CSV row {row_number} timestamp is outside declared UTC date"
        )
    if previous_event_ts_ns is not None and event_ts_ns < previous_event_ts_ns:
        raise BinanceArchivePayloadError(f"archive event time reverses at row {row_number}")
    price, price_ticks = _scaled_decimal(
        fields[1], quantum=acquired.request.tick_size, label=f"row {row_number} price"
    )
    quantity, quantity_lots = _scaled_decimal(
        fields[2], quantum=acquired.request.lot_size, label=f"row {row_number} quantity"
    )
    buyer_is_maker = _parse_boolean(fields[6], label=f"row {row_number} buyer-maker flag")
    _parse_boolean(fields[7], label=f"row {row_number} best-match flag")
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "venue": "binance_spot",
        "symbol": acquired.request.symbol,
        "event_ts_ns": event_ts_ns,
        "received_ts_ns": None,
        "available_ts_ns": event_ts_ns,
        "availability_basis": "exchange_event_time_proxy",
        "capture_seq": None,
        "continuity_id": acquired.request.continuity_id,
        "trade_id": aggregate_id,
        "first_trade_id": first_trade_id,
        "last_trade_id": last_trade_id,
        "price_ticks": price_ticks,
        "quantity_lots": quantity_lots,
        "tick_size": float(acquired.request.tick_size),
        "lot_size": float(acquired.request.lot_size),
        "price": float(price),
        "quantity": float(quantity),
        "quote_quantity": float(price * quantity),
        "aggressor_side": "sell" if buyer_is_maker else "buy",
        "buyer_is_maker": buyer_is_maker,
        "source_artifact_id": acquired.archive_artifact.sha256,
    }
    return record, aggregate_id, event_ts_ns


def _sha256_open_file(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    source.seek(0)
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _stream_normalized_batches(
    acquired: AcquiredDailyArchive,
    *,
    batch_rows: int,
    before_member_open: Callable[[], None] | None,
) -> Generator[pa.RecordBatch, None, DailyArchiveSummary]:
    if acquired.requires_member_open_guard and before_member_open is None:
        raise BinanceArchivePayloadError("held-out archive requires a member-open authority guard")
    archive_path = acquired.archive_artifact.path
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        raw_descriptor = os.open(archive_path, os.O_RDONLY | nofollow)
    except OSError as exc:
        raise BinanceArchivePayloadError("archive bytes are unavailable after acquisition") from exc
    try:
        archive_source = os.fdopen(raw_descriptor, "rb")
    except BaseException:
        os.close(raw_descriptor)
        raise

    start_ns, end_ns = _day_bounds_ns(acquired.request.date)
    rows = 0
    first_trade_id: int | None = None
    last_trade_id: int | None = None
    first_event_ts_ns: int | None = None
    last_event_ts_ns: int | None = None
    records: list[dict[str, object]] = []
    expanded_bytes = 0
    with archive_source:
        try:
            observed = os.fstat(archive_source.fileno())
            if not stat.S_ISREG(observed.st_mode):
                raise BinanceArchivePayloadError("archive bytes are not a regular file")
            if observed.st_size != acquired.archive_artifact.bytes:
                raise BinanceArchivePayloadError("archive bytes changed after acquisition")
            if _sha256_open_file(archive_source) != acquired.archive_artifact.sha256:
                raise BinanceArchivePayloadError("archive checksum changed after acquisition")
            descriptor = _validate_zip_structure_handle(
                archive_source,
                observed.st_size,
                expected_member=acquired.request.member_name,
                max_uncompressed_bytes=acquired.limits.max_uncompressed_bytes,
            )
            if descriptor.declared_uncompressed_bytes != acquired.declared_uncompressed_bytes:
                raise BinanceArchivePayloadError(
                    "archive uncompressed-size claim changed after acquisition"
                )
            archive_source.seek(0)
            archive_context = zipfile.ZipFile(archive_source)
        except BinanceArchivePayloadError:
            raise
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            raise BinanceArchivePayloadError("cannot stream archive CSV safely") from exc
        with archive_context as archive:
            # The same already-hashed file descriptor backs both ZipFile and
            # member decompression.  The path may change, but the guarded bytes
            # cannot switch inode between authority verification and open.
            if before_member_open is not None:
                before_member_open()
            try:
                with archive.open(descriptor.member_name, "r") as member:
                    lines = _line_chunks(
                        member,
                        max_uncompressed_bytes=acquired.limits.max_uncompressed_bytes,
                        chunk_bytes=acquired.limits.transfer_chunk_bytes,
                        max_line_bytes=acquired.limits.max_csv_line_bytes,
                    )
                    while True:
                        try:
                            line = next(lines)
                        except StopIteration as stop:
                            expanded_bytes = int(stop.value)
                            break
                        row_number = rows + 1
                        record, trade_id, event_ts_ns = _record_from_line(
                            line,
                            row_number=row_number,
                            acquired=acquired,
                            previous_trade_id=last_trade_id,
                            previous_event_ts_ns=last_event_ts_ns,
                            start_ns=start_ns,
                            end_ns=end_ns,
                        )
                        if first_trade_id is None:
                            first_trade_id = trade_id
                            first_event_ts_ns = event_ts_ns
                        last_trade_id = trade_id
                        last_event_ts_ns = event_ts_ns
                        records.append(record)
                        rows += 1
                        if len(records) == batch_rows:
                            table = table_from_records("trades", records)
                            batches = table.to_batches(max_chunksize=batch_rows)
                            if len(batches) != 1 or batches[0].num_rows > batch_rows:
                                raise BinanceArchivePayloadError(
                                    "archive normalizer violated its RecordBatch bound"
                                )
                            yield batches[0]
                            records = []
                    if records:
                        table = table_from_records("trades", records)
                        batches = table.to_batches(max_chunksize=batch_rows)
                        if len(batches) != 1 or batches[0].num_rows > batch_rows:
                            raise BinanceArchivePayloadError(
                                "archive normalizer violated its RecordBatch bound"
                            )
                        yield batches[0]
            except BinanceArchivePayloadError:
                raise
            except (
                OSError,
                RuntimeError,
                UnicodeError,
                ValueError,
                OverflowError,
                zipfile.BadZipFile,
            ) as exc:
                raise BinanceArchivePayloadError("cannot stream archive CSV safely") from exc

    if (
        rows < 1
        or first_trade_id is None
        or last_trade_id is None
        or first_event_ts_ns is None
        or last_event_ts_ns is None
    ):
        raise BinanceArchivePayloadError("archive CSV contains no trade rows")
    if expanded_bytes != acquired.declared_uncompressed_bytes:
        raise BinanceArchivePayloadError(
            "streamed archive bytes disagree with ZIP uncompressed-size claim"
        )
    return DailyArchiveSummary(
        symbol=acquired.request.symbol,
        date=acquired.request.date.isoformat(),
        rows=rows,
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        first_event_ts_ns=first_event_ts_ns,
        last_event_ts_ns=last_event_ts_ns,
        compressed_bytes=acquired.archive_artifact.bytes,
        expanded_bytes=expanded_bytes,
        source_archive_sha256=acquired.archive_artifact.sha256,
        member_name=descriptor.member_name,
        continuity_id=acquired.request.continuity_id,
    )


__all__ = [
    "AcquiredDailyArchive",
    "ArchiveAcquisitionReasonCode",
    "ArchiveDownloadLimits",
    "BinanceArchiveClient",
    "BinanceArchiveContractError",
    "BinanceArchiveError",
    "BinanceArchiveHTTPError",
    "BinanceArchivePayloadError",
    "DailyArchiveRequest",
    "DailyArchiveSummary",
    "DailyArchiveTradeStream",
    "RawArchiveArtifact",
    "RetryPolicy",
]
