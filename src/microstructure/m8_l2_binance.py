"""Production Binance adapter for one frozen prospective M8 L2 symbol.

The cross-symbol/session authority lives in :mod:`microstructure.m8_l2_capture`.
This module owns only the public market-data transport, bounded raw journaling,
snapshot-plus-delta reconstruction, and the exhaustive per-symbol artifact
descriptor returned across that typed boundary.  It has no authenticated or
order-entry path.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import pyarrow as pa  # type: ignore[import-untyped]
from websockets.exceptions import WebSocketException

from microstructure.data.binance import (
    BinanceHTTPError,
    BinanceLiveDepthCollector,
    BinanceMetadataContractError,
    BinancePayloadError,
    BinancePublicClient,
    BinanceResponseSizeLimitError,
    CapturedDepth,
    RawDepthFrame,
    SymbolMetadata,
)
from microstructure.data.book import BookInvariantError, BookSnapshot, IncrementalBookReconstructor
from microstructure.data.quality import IncrementalQualityValidator, ValidationReport
from microstructure.data.schemas import get_schema, table_from_records
from microstructure.data.storage import write_capture_parquet, write_source_manifest
from microstructure.m8_l2_capture import (
    CapturedArtifact,
    M8L2DataFailure,
    ObservedInterval,
    SymbolCaptureResult,
)
from microstructure.m8_l2_config import M8L2CaptureLimits
from microstructure.provenance import read_json, sha256_file, utc_now_iso, write_json

_BATCH_ROWS = 1_024
_VARIABLE_RECORD_OVERHEAD_FACTOR = 8
_MAX_OBSERVED_SILENCE_NS = 5_000_000_000
_DEFAULT_RECEIVER_QUEUE_CAPACITY = 1_024
_SOURCE_URI = "wss://data-stream.binance.vision"


class _Collector(Protocol):
    url: str

    def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]: ...


class _Client(Protocol):
    def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SymbolMetadata: ...

    def fetch_depth_snapshot(
        self,
        *,
        symbol: str,
        raw_root: Path,
        continuity_id: str,
        tick_size: object,
        lot_size: object,
    ) -> BookSnapshot: ...


class _DataCaptureIssue(RuntimeError):
    """Internal deterministic capture failure converted after evidence finalization."""

    def __init__(self, reason_code: str, phase: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.phase = phase


class _ScheduledEndReached(RuntimeError):
    """Stop the collector before it decodes the first out-of-window frame."""


class _LocalEvidenceSystemError(RuntimeError):
    """Keep local journal/storage faults out of public-transport classification."""


async def _to_thread_joined[**P, T](
    function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Make cancellation wait for a public REST worker that may still write raw bytes."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as canceled:
        try:
            await task
        except BaseException as worker_error:
            canceled.add_note(
                f"joined REST worker also failed with {type(worker_error).__name__}: {worker_error}"
            )
        raise


@dataclass(frozen=True, slots=True)
class _PublishedJournal:
    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str


class _RawJournal:
    """Write exact websocket bytes before decode and retain FIFO lineage."""

    def __init__(
        self,
        *,
        root: Path,
        symbol: str,
        source_uri: str,
        scheduled_start_ns: int,
        scheduled_end_ns: int,
        max_frame_bytes: int,
        max_messages: int,
    ) -> None:
        self.root = root
        self.symbol = symbol
        self.source_uri = source_uri
        self.scheduled_start_ns = scheduled_start_ns
        self.scheduled_end_ns = scheduled_end_ns
        self.max_frame_bytes = max_frame_bytes
        self.max_messages = max_messages
        self.directory = root / "raw" / "binance_spot" / "depth_stream" / symbol
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=self.directory,
            prefix=".m8-l2-",
            suffix=".ndjson.tmp",
        )
        self._temporary_path = Path(name)
        self._handle = os.fdopen(descriptor, "wb")
        self._pending: deque[tuple[str, int, int, str]] = deque()
        self.messages = 0
        self.snapshot_anchors = 0
        self.first_received_ns: int | None = None
        self.last_received_ns: int | None = None
        self.max_frame_bytes_observed = 0
        self._closed = False
        self._published: _PublishedJournal | None = None

    @property
    def evidence_path(self) -> Path:
        return self._published.path if self._published is not None else self._temporary_path

    def _write(self, payload: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed raw journal")
        encoded = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._handle.write(encoded)
        self._handle.write(b"\n")

    def append_frame(self, frame: RawDepthFrame) -> None:
        """Persist an in-window frame; the collector calls this before parsing."""

        if not (self.scheduled_start_ns <= frame.received_ts_ns < self.scheduled_end_ns):
            return
        payload_size = len(frame.payload)
        payload_sha256 = hashlib.sha256(frame.payload).hexdigest()
        self._write(
            {
                "capture_seq": frame.capture_seq,
                "continuity_id": frame.continuity_id,
                "event_kind": "websocket_frame",
                "payload_base64": base64.b64encode(frame.payload).decode("ascii"),
                "payload_bytes": payload_size,
                "payload_sha256": payload_sha256,
                "received_ts_ns": frame.received_ts_ns,
                "websocket_message_type": "text" if frame.was_text else "binary",
            }
        )
        self.messages += 1
        self.max_frame_bytes_observed = max(self.max_frame_bytes_observed, payload_size)
        if self.first_received_ns is None:
            self.first_received_ns = frame.received_ts_ns
        self.last_received_ns = frame.received_ts_ns
        self._pending.append(
            (frame.continuity_id, frame.capture_seq, frame.received_ts_ns, payload_sha256)
        )
        if payload_size > self.max_frame_bytes:
            raise _DataCaptureIssue(
                "RAW_FRAME_BYTES_EXCEEDED",
                "RAW_CAPTURE",
                f"raw frame has {payload_size} bytes; maximum is {self.max_frame_bytes}",
            )
        # The ceiling is a failure boundary, never an alternate stopping target.
        if self.messages >= self.max_messages:
            raise _DataCaptureIssue(
                "MESSAGE_SAFETY_CEILING_REACHED",
                "RAW_CAPTURE",
                f"message ceiling {self.max_messages} reached before scheduled end",
            )

    def consume_captured(self, item: CapturedDepth) -> None:
        received_ns = item.delta.received_ts_ns
        capture_seq = item.delta.capture_seq
        if received_ns is None or capture_seq is None:
            raise _DataCaptureIssue(
                "MISSING_RAW_LINEAGE",
                "NORMALIZATION",
                "captured depth delta lacks receipt time or capture sequence",
            )
        identity = (
            item.delta.continuity_id,
            capture_seq,
            received_ns,
            hashlib.sha256(item.raw_payload.encode("utf-8")).hexdigest(),
        )
        if not self._pending or self._pending[0] != identity:
            raise _DataCaptureIssue(
                "RAW_LINEAGE_MISMATCH",
                "NORMALIZATION",
                "normalized depth delta is not the next preserved raw frame",
            )
        self._pending.popleft()
        if item.delta.source_artifact_id != identity[-1]:
            raise _DataCaptureIssue(
                "RAW_LINEAGE_MISMATCH",
                "NORMALIZATION",
                "normalized depth source digest differs from preserved raw bytes",
            )

    def append_snapshot(
        self,
        snapshot: BookSnapshot,
        *,
        raw_path: Path,
        manifest_path: Path,
    ) -> None:
        self._write(
            {
                "continuity_id": snapshot.continuity_id,
                "event_kind": "rest_snapshot_anchor",
                "last_update_id": snapshot.last_update_id,
                "raw_manifest_path": raw_path.parent.joinpath(manifest_path.name)
                .relative_to(self.root)
                .as_posix(),
                "raw_manifest_sha256": sha256_file(manifest_path),
                "raw_path": raw_path.relative_to(self.root).as_posix(),
                "raw_sha256": snapshot.source_artifact_id,
                "received_ts_ns": snapshot.received_ts_ns,
                "snapshot_id": snapshot.snapshot_id,
            }
        )
        self.snapshot_anchors += 1

    def _close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def publish(self, *, capture_status: str, error: BaseException | None) -> _PublishedJournal:
        if self._published is not None:
            return self._published
        self._close()
        digest = sha256_file(self._temporary_path)
        destination = self.directory / f"capture-{digest}.ndjson"
        if destination.exists():
            if sha256_file(destination) != digest:
                raise RuntimeError(f"raw journal content-address collision at {destination}")
            self._temporary_path.unlink(missing_ok=True)
        else:
            os.replace(self._temporary_path, destination)
        headers = {
            "x-local-capture-status": capture_status,
            "x-local-journal-format": "typed-base64-frames-v1",
            "x-local-message-count": str(self.messages),
            "x-local-snapshot-anchor-count": str(self.snapshot_anchors),
            "x-local-scheduled-start-ns": str(self.scheduled_start_ns),
            "x-local-scheduled-end-ns-exclusive": str(self.scheduled_end_ns),
        }
        if error is not None:
            headers["x-local-error-type"] = type(error).__name__
            headers["x-local-error"] = str(error)[:512]
        manifest_path, manifest_sha256 = write_source_manifest(
            destination,
            source="binance_spot_public_live_capture_journal",
            source_uri=self.source_uri,
            downloaded_at_utc=utc_now_iso(),
            requested_start_ns=self.scheduled_start_ns,
            requested_end_ns=self.scheduled_end_ns,
            response_headers=headers,
        )
        self._published = _PublishedJournal(
            path=destination,
            sha256=digest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        return self._published

    def close_without_publish(self) -> None:
        self._close()


class _ArrowSpool:
    """Bounded record buffer backed by an Arrow IPC stream on disk."""

    def __init__(
        self,
        *,
        root: Path,
        schema_name: str,
        max_batch_bytes: int,
        on_batch: Callable[[pa.RecordBatch], None] | None = None,
    ) -> None:
        self.schema_name = schema_name
        self.max_batch_bytes = max_batch_bytes
        self.path = root / f"{schema_name}.arrow"
        self._handle = self.path.open("wb")
        self._writer = pa.ipc.new_stream(self._handle, get_schema(schema_name))
        self._on_batch = on_batch
        self._records: list[Mapping[str, object]] = []
        self._estimated_bytes = 0
        self.rows = 0
        self.max_batch_bytes_observed = 0
        self._closed = False

    def append(self, record: Mapping[str, object], *, estimated_bytes: int) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed Arrow spool")
        if estimated_bytes < 1:
            raise RuntimeError("Arrow record estimate must be positive")
        if estimated_bytes > self.max_batch_bytes:
            raise _DataCaptureIssue(
                "ARROW_BATCH_BYTES_EXCEEDED",
                "NORMALIZATION",
                f"one {self.schema_name} record estimate exceeds the Arrow batch ceiling",
            )
        if self._records and self._estimated_bytes + estimated_bytes > self.max_batch_bytes:
            self._flush()
        self._records.append(record)
        self._estimated_bytes += estimated_bytes
        if len(self._records) >= _BATCH_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        table = table_from_records(self.schema_name, self._records)
        batches = table.to_batches(max_chunksize=_BATCH_ROWS)
        if len(batches) != 1:
            raise RuntimeError(f"failed to build one bounded {self.schema_name} Arrow batch")
        batch = batches[0]
        observed_bytes = max(self._estimated_bytes, batch.nbytes)
        self.max_batch_bytes_observed = max(self.max_batch_bytes_observed, observed_bytes)
        if observed_bytes > self.max_batch_bytes:
            raise _DataCaptureIssue(
                "ARROW_BATCH_BYTES_EXCEEDED",
                "NORMALIZATION",
                f"{self.schema_name} Arrow batch exceeds {self.max_batch_bytes} bytes",
            )
        if self._on_batch is not None:
            self._on_batch(batch)
        self._writer.write_batch(batch)
        self.rows += batch.num_rows
        self._records.clear()
        self._estimated_bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._writer.close()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        if not self._closed:
            raise RuntimeError("Arrow spool must be closed before reading")
        with self.path.open("rb") as handle:
            for batch in pa.ipc.open_stream(handle):
                if batch.num_rows > _BATCH_ROWS or batch.nbytes > self.max_batch_bytes:
                    raise RuntimeError(f"persisted {self.schema_name} batch exceeds its bound")
                yield batch


class _ObservedIntervals:
    """Track only receipt spans confirmed by consecutive OBSERVED states."""

    def __init__(self) -> None:
        self._continuity_id: str | None = None
        self._previous_ns: int | None = None
        self._active_start_ns: int | None = None
        self._active_end_ns: int | None = None
        self._intervals: list[ObservedInterval] = []

    def _finish_active(self) -> None:
        if (
            self._continuity_id is not None
            and self._active_start_ns is not None
            and self._active_end_ns is not None
            and self._active_end_ns > self._active_start_ns
        ):
            self._intervals.append(
                ObservedInterval(
                    continuity_id=self._continuity_id,
                    start_received_ns=self._active_start_ns,
                    end_received_ns_exclusive=self._active_end_ns,
                )
            )
        self._active_start_ns = None
        self._active_end_ns = None

    def break_continuity(self) -> None:
        self._finish_active()
        self._continuity_id = None
        self._previous_ns = None

    def excluded(self, continuity_id: str) -> None:
        if self._continuity_id == continuity_id:
            self._finish_active()
        else:
            self._finish_active()
            self._continuity_id = continuity_id
        self._previous_ns = None

    def observed(self, continuity_id: str, received_ns: int) -> None:
        if self._continuity_id != continuity_id:
            self.break_continuity()
            self._continuity_id = continuity_id
        previous = self._previous_ns
        if (
            previous is not None
            and received_ns > previous
            and received_ns - previous <= _MAX_OBSERVED_SILENCE_NS
        ):
            if self._active_start_ns is None:
                self._active_start_ns = previous
            self._active_end_ns = received_ns + 1
        else:
            self._finish_active()
        self._previous_ns = received_ns

    def finish(self) -> tuple[ObservedInterval, ...]:
        self._finish_active()
        return tuple(self._intervals)


def _snapshot_files(root: Path, snapshot: BookSnapshot) -> tuple[Path, Path]:
    directory = root / "raw" / "binance_spot" / "depth_snapshots" / snapshot.symbol
    raw_path = directory / f"{snapshot.source_artifact_id}.json"
    if raw_path.is_symlink() or not raw_path.is_file():
        raise RuntimeError("snapshot raw response is not a regular preserved artifact")
    if sha256_file(raw_path) != snapshot.source_artifact_id:
        raise RuntimeError("snapshot raw response digest does not match its source identity")
    candidates: list[tuple[Path, Mapping[str, object]]] = []
    for path in directory.glob(f"{raw_path.name}.manifest-*.json"):
        payload = read_json(path)
        checksum = payload.get("checksum") if isinstance(payload, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("path") == raw_path.name
            and isinstance(checksum, dict)
            and checksum.get("value") == snapshot.source_artifact_id
        ):
            candidates.append((path, payload))
    if len(candidates) == 1:
        return raw_path, candidates[0][0]
    seconds, nanoseconds = divmod(snapshot.received_ts_ns, 1_000_000_000)
    expected_downloaded = (
        time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds)) + f".{nanoseconds:09d}Z"
    )
    current = [
        path
        for path, payload in candidates
        if payload.get("downloaded_at_utc") == expected_downloaded
    ]
    if len(current) != 1:
        raise RuntimeError(
            "snapshot does not have one source manifest bound to its receipt timestamp"
        )
    return raw_path, current[0]


def _verify_metadata_files(root: Path, metadata: SymbolMetadata) -> None:
    for label, path in (
        ("exchange metadata raw response", metadata.source_path),
        ("exchange metadata source manifest", metadata.source_manifest_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{label} is not a regular file")
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"{label} escapes the single-symbol capture stage") from error
    if sha256_file(metadata.source_path) != metadata.source_artifact_id:
        raise RuntimeError("exchange metadata digest differs from its raw response")
    manifest = read_json(metadata.source_manifest_path)
    checksum = manifest.get("checksum") if isinstance(manifest, dict) else None
    if not (
        isinstance(manifest, dict)
        and manifest.get("path") == metadata.source_path.name
        and isinstance(checksum, dict)
        and checksum.get("value") == metadata.source_artifact_id
    ):
        raise RuntimeError("exchange metadata source manifest is not bound to its raw response")


def _artifact_kind(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    name = path.name
    if "/depth_stream/" in f"/{relative}" and name.endswith(".ndjson"):
        return "raw_journal"
    if "/depth_stream/" in f"/{relative}" and ".ndjson.manifest-" in name:
        return "raw_journal_manifest"
    if "/depth_snapshots/" in f"/{relative}" and ".json.manifest-" in name:
        return "raw_snapshot_manifest"
    if "/depth_snapshots/" in f"/{relative}" and name.endswith(".json"):
        return "raw_snapshot"
    if "/exchange_info/" in f"/{relative}" and ".json.manifest-" in name:
        return "raw_metadata_manifest"
    if "/exchange_info/" in f"/{relative}" and name.endswith(".json"):
        return "raw_metadata"
    if relative.startswith("normalized/") and name.endswith(".parquet"):
        return "normalized_data"
    if relative.startswith("normalized/") and name.endswith(".json"):
        return "normalized_manifest"
    if relative.startswith("quality/") and name.endswith(".summary.json"):
        return "capture_summary"
    if relative.startswith("quality/") and name.endswith(".json"):
        return "quality_report"
    return "partial_evidence"


def _artifacts(root: Path) -> tuple[CapturedArtifact, ...]:
    result: list[CapturedArtifact] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative_parts = path.relative_to(root).parts
        if relative_parts and relative_parts[0].startswith(".arrow-spool-"):
            continue
        if path.is_symlink():
            raise RuntimeError(f"symbol evidence contains a symlink: {path}")
        if path.is_file():
            result.append(
                CapturedArtifact(
                    path=path,
                    kind=_artifact_kind(path, root),
                    sha256=sha256_file(path),
                )
            )
    return tuple(result)


def _classify_binance_data_error(error: BaseException) -> _DataCaptureIssue | None:
    if isinstance(error, UnicodeDecodeError):
        return _DataCaptureIssue("RAW_UTF8_DECODE_FAILED", "NORMALIZATION", str(error))
    if isinstance(error, BinanceHTTPError):
        if error.status_code in {404, 410} and not error.retry_exhausted:
            return _DataCaptureIssue("DECLARED_OBJECT_UNAVAILABLE", "PUBLIC_METADATA", str(error))
        return _DataCaptureIssue("PUBLIC_TRANSPORT_UNAVAILABLE", "PUBLIC_METADATA", str(error))
    if isinstance(error, (BinanceMetadataContractError, BinanceResponseSizeLimitError)):
        return _DataCaptureIssue("PUBLIC_PAYLOAD_CONTRACT_FAILED", "PUBLIC_METADATA", str(error))
    if isinstance(error, BinancePayloadError) and not error.transient:
        return _DataCaptureIssue("PUBLIC_PAYLOAD_CONTRACT_FAILED", "NORMALIZATION", str(error))
    if isinstance(error, BookInvariantError):
        return _DataCaptureIssue("BOOK_INVARIANT_FAILED", "RECONSTRUCTION", str(error))
    return None


@dataclass(slots=True)
class _CaptureState:
    current_continuity_id: str | None = None
    reconstructor: IncrementalBookReconstructor | None = None
    continuity_epochs: int = 0
    reconstruction_status: Literal["LIVE", "GAPPED", "INVALID", "NOT_STARTED"] = "NOT_STARTED"
    excluded_rows: int = 0
    sequence_gaps: int = 0
    stale_rows: int = 0
    max_queue_depth: int = 0
    max_queue_estimated_bytes: int = 0


class BinanceM8L2Capture:
    """Callable production implementation of the frozen ``CaptureOne`` protocol."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], _Client] | None = None,
        collector_factory: Callable[..., _Collector] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        receiver_queue_capacity: int = _DEFAULT_RECEIVER_QUEUE_CAPACITY,
    ) -> None:
        if receiver_queue_capacity < 1:
            raise ValueError("receiver_queue_capacity must be positive")
        self._client_factory = client_factory or cast(Callable[[], _Client], BinancePublicClient)
        self._collector_factory = collector_factory or cast(
            Callable[..., _Collector], BinanceLiveDepthCollector
        )
        self._clock_ns = clock_ns
        self._receiver_queue_capacity = receiver_queue_capacity

    async def __call__(
        self,
        *,
        symbol: str,
        scheduled_start_ns: int,
        scheduled_end_ns: int,
        stage_root: Path,
        limits: M8L2CaptureLimits,
        session_id: str,
    ) -> SymbolCaptureResult:
        if scheduled_end_ns <= scheduled_start_ns:
            raise ValueError("scheduled_end_ns must be after scheduled_start_ns")
        if limits.max_messages_per_symbol < 1:
            raise ValueError("message ceiling must be positive")
        if limits.max_raw_frame_bytes < 1 or limits.max_arrow_batch_bytes < 1:
            raise ValueError("capture byte ceilings must be positive")
        normalized_symbol = symbol.upper()
        if normalized_symbol not in {"BTCUSDT", "ETHUSDT"}:
            raise ValueError(f"unsupported frozen L2 symbol: {symbol}")
        stage = stage_root.resolve()
        stage.mkdir(parents=True, exist_ok=True)
        if any(stage.iterdir()):
            raise RuntimeError("single-symbol capture stage must start empty")
        capture_id = f"{normalized_symbol.lower()}-{session_id[:20]}"
        queue_capacity = min(
            self._receiver_queue_capacity,
            limits.max_messages_per_symbol,
        )
        queue_byte_budget = limits.max_arrow_batch_bytes
        journal = _RawJournal(
            root=stage,
            symbol=normalized_symbol,
            source_uri=_SOURCE_URI,
            scheduled_start_ns=scheduled_start_ns,
            scheduled_end_ns=scheduled_end_ns,
            max_frame_bytes=limits.max_raw_frame_bytes,
            max_messages=limits.max_messages_per_symbol,
        )
        state = _CaptureState()
        interval_tracker = _ObservedIntervals()
        client = self._client_factory()
        receiver_task: asyncio.Task[None] | None = None
        capture_error: BaseException | None = None
        completion_reason = "capture_failed"
        metadata: SymbolMetadata | None = None
        raw_evidence: _PublishedJournal | None = None
        validation_reports: tuple[ValidationReport, ValidationReport] | None = None

        with tempfile.TemporaryDirectory(dir=stage, prefix=".arrow-spool-") as spool_name:
            spool_root = Path(spool_name)
            delta_validator = IncrementalQualityValidator(
                "depth_deltas", row_chunk_size=_BATCH_ROWS
            )
            observation_validator = IncrementalQualityValidator(
                "book_observations", row_chunk_size=_BATCH_ROWS
            )
            validators_finished = False
            spools = {
                "book_snapshots": _ArrowSpool(
                    root=spool_root,
                    schema_name="book_snapshots",
                    max_batch_bytes=limits.max_arrow_batch_bytes,
                ),
                "depth_deltas": _ArrowSpool(
                    root=spool_root,
                    schema_name="depth_deltas",
                    max_batch_bytes=limits.max_arrow_batch_bytes,
                    on_batch=delta_validator.update,
                ),
                "book_observations": _ArrowSpool(
                    root=spool_root,
                    schema_name="book_observations",
                    max_batch_bytes=limits.max_arrow_batch_bytes,
                    on_batch=observation_validator.update,
                ),
                "sequence_gaps": _ArrowSpool(
                    root=spool_root,
                    schema_name="sequence_gaps",
                    max_batch_bytes=limits.max_arrow_batch_bytes,
                ),
            }
            try:
                if self._clock_ns() >= scheduled_end_ns:
                    raise _DataCaptureIssue(
                        "MISSED_WINDOW", "CAPTURE_START", "scheduled capture window already ended"
                    )
                try:
                    metadata = await _to_thread_joined(
                        client.fetch_exchange_info,
                        symbol=normalized_symbol,
                        raw_root=stage / "raw",
                    )
                except BaseException as error:
                    classified = _classify_binance_data_error(error)
                    if classified is not None:
                        raise classified from error
                    raise
                _verify_metadata_files(stage, metadata)
                if metadata.symbol != normalized_symbol or metadata.venue != "binance_spot":
                    raise _DataCaptureIssue(
                        "METADATA_IDENTITY_MISMATCH",
                        "PUBLIC_METADATA",
                        "exchange metadata venue/symbol differs from the requested feed",
                    )
                if metadata.status != "TRADING":
                    raise _DataCaptureIssue(
                        "SYMBOL_NOT_TRADING",
                        "PUBLIC_METADATA",
                        f"{normalized_symbol} exchange status is {metadata.status!r}",
                    )
                if self._clock_ns() >= scheduled_end_ns:
                    raise _DataCaptureIssue(
                        "METADATA_EXHAUSTED_WINDOW",
                        "PUBLIC_METADATA",
                        "metadata acquisition consumed the scheduled capture window",
                    )

                def preserve_in_window_frame(frame: RawDepthFrame) -> None:
                    if frame.received_ts_ns >= scheduled_end_ns:
                        raise _ScheduledEndReached
                    try:
                        journal.append_frame(frame)
                    except (_DataCaptureIssue, _ScheduledEndReached):
                        raise
                    except OSError as error:
                        raise _LocalEvidenceSystemError("local raw-journal write failed") from error

                collector = self._collector_factory(
                    symbols=(normalized_symbol,),
                    tick_size=metadata.tick_size,
                    lot_size=metadata.lot_size,
                    on_raw_frame=preserve_in_window_frame,
                )
                journal.source_uri = collector.url
                queue: asyncio.Queue[tuple[CapturedDepth, int]] = asyncio.Queue(
                    maxsize=queue_capacity
                )
                queue_condition = asyncio.Condition()
                queued_estimated_bytes = 0

                async def receive() -> None:
                    nonlocal queued_estimated_bytes
                    iterator = collector.stream(max_messages=None).__aiter__()
                    try:
                        while True:
                            remaining_ns = scheduled_end_ns - self._clock_ns()
                            if remaining_ns <= 0:
                                return
                            try:
                                item = await asyncio.wait_for(
                                    anext(iterator),
                                    timeout=remaining_ns / 1_000_000_000,
                                )
                            except _ScheduledEndReached:
                                return
                            except TimeoutError:
                                # Re-read absolute wall time.  Suspend/resume or a
                                # wall-clock adjustment must not turn this into a
                                # relative-duration completion.
                                continue
                            except PermissionError:
                                raise
                            except (OSError, WebSocketException) as error:
                                raise _DataCaptureIssue(
                                    "PUBLIC_STREAM_UNAVAILABLE",
                                    "RAW_CAPTURE",
                                    "public websocket transport was exhausted during the "
                                    "declared frozen window",
                                ) from error
                            except StopAsyncIteration:
                                if self._clock_ns() < scheduled_end_ns:
                                    raise _DataCaptureIssue(
                                        "STREAM_ENDED_EARLY",
                                        "RAW_CAPTURE",
                                        "public depth stream ended before scheduled UTC end",
                                    ) from None
                                return
                            received_ns = item.delta.received_ts_ns
                            if received_ns is None:
                                raise _DataCaptureIssue(
                                    "MISSING_RAW_LINEAGE",
                                    "NORMALIZATION",
                                    "captured message lacks local receipt time",
                                )
                            if received_ns < scheduled_start_ns:
                                continue
                            if received_ns >= scheduled_end_ns:
                                return
                            journal.consume_captured(item)
                            raw_size = len(item.raw_payload.encode("utf-8"))
                            item_estimated_bytes = max(
                                4_096, raw_size * _VARIABLE_RECORD_OVERHEAD_FACTOR
                            )
                            if item_estimated_bytes > queue_byte_budget:
                                raise _DataCaptureIssue(
                                    "RECEIVER_QUEUE_ITEM_BYTES_EXCEEDED",
                                    "RAW_CAPTURE",
                                    "one parsed frame exceeds the bounded receiver queue budget",
                                )
                            async with queue_condition:
                                while (
                                    queued_estimated_bytes + item_estimated_bytes
                                    > queue_byte_budget
                                    or queue.full()
                                ):
                                    remaining_ns = scheduled_end_ns - self._clock_ns()
                                    if remaining_ns <= 0:
                                        raise _DataCaptureIssue(
                                            "PROCESSING_BACKLOG_AT_END",
                                            "RAW_CAPTURE",
                                            "bounded receiver queue remained full at scheduled end",
                                        )
                                    try:
                                        await asyncio.wait_for(
                                            queue_condition.wait(),
                                            timeout=remaining_ns / 1_000_000_000,
                                        )
                                    except TimeoutError:
                                        raise _DataCaptureIssue(
                                            "PROCESSING_BACKLOG_AT_END",
                                            "RAW_CAPTURE",
                                            "bounded receiver queue remained full at scheduled end",
                                        ) from None
                                queue.put_nowait((item, item_estimated_bytes))
                                queued_estimated_bytes += item_estimated_bytes
                            state.max_queue_depth = max(state.max_queue_depth, queue.qsize())
                            state.max_queue_estimated_bytes = max(
                                state.max_queue_estimated_bytes, queued_estimated_bytes
                            )
                    finally:
                        closer = getattr(iterator, "aclose", None)
                        if callable(closer):
                            with suppress(BaseException):
                                await closer()

                receiver_task = asyncio.create_task(
                    receive(), name=f"m8-l2-receiver-{normalized_symbol}"
                )

                async def release_item(queued: tuple[CapturedDepth, int]) -> CapturedDepth:
                    nonlocal queued_estimated_bytes
                    item, estimated_bytes = queued
                    async with queue_condition:
                        queued_estimated_bytes -= estimated_bytes
                        if queued_estimated_bytes < 0:  # pragma: no cover - accounting invariant
                            raise RuntimeError("receiver queue byte accounting became negative")
                        queue_condition.notify_all()
                    return item

                async def next_item() -> CapturedDepth | None:
                    if not queue.empty():
                        return await release_item(queue.get_nowait())
                    if receiver_task is None:  # pragma: no cover - construction invariant
                        raise RuntimeError("receiver task was not initialized")
                    if receiver_task.done():
                        await receiver_task
                        return None
                    get_task = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait(
                        {get_task, receiver_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if get_task in done:
                        return await release_item(get_task.result())
                    get_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await get_task
                    if not queue.empty():
                        return await release_item(queue.get_nowait())
                    await receiver_task
                    return None

                while True:
                    item = await next_item()
                    if item is None:
                        break
                    delta = item.delta
                    received_ns = delta.received_ts_ns
                    if received_ns is None:  # pragma: no cover - receiver checks
                        raise RuntimeError("processed delta has no receipt timestamp")
                    if delta.continuity_id != state.current_continuity_id:
                        interval_tracker.break_continuity()
                        try:
                            snapshot = await _to_thread_joined(
                                client.fetch_depth_snapshot,
                                symbol=normalized_symbol,
                                raw_root=stage / "raw",
                                continuity_id=delta.continuity_id,
                                tick_size=metadata.tick_size,
                                lot_size=metadata.lot_size,
                            )
                        except BaseException as error:
                            classified = _classify_binance_data_error(error)
                            if classified is not None:
                                classified.phase = "SNAPSHOT_ANCHOR"
                                raise classified from error
                            raise
                        if not (scheduled_start_ns <= snapshot.received_ts_ns < scheduled_end_ns):
                            raise _DataCaptureIssue(
                                "SNAPSHOT_OUTSIDE_WINDOW",
                                "SNAPSHOT_ANCHOR",
                                "continuity snapshot was not received inside the frozen window",
                            )
                        raw_snapshot, snapshot_manifest = _snapshot_files(stage, snapshot)
                        journal.append_snapshot(
                            snapshot,
                            raw_path=raw_snapshot,
                            manifest_path=snapshot_manifest,
                        )
                        spools["book_snapshots"].append(
                            snapshot.to_record(),
                            estimated_bytes=max(
                                4_096, 128 * (len(snapshot.bids) + len(snapshot.asks))
                            ),
                        )
                        state.reconstructor = IncrementalBookReconstructor(snapshot)
                        state.current_continuity_id = delta.continuity_id
                        state.continuity_epochs += 1
                        state.reconstruction_status = "LIVE"

                    reconstructor = state.reconstructor
                    if reconstructor is None:  # pragma: no cover - guarded above
                        raise RuntimeError("continuity has no snapshot reconstructor")
                    raw_bytes = len(item.raw_payload.encode("utf-8"))
                    spools["depth_deltas"].append(
                        delta.to_record(),
                        estimated_bytes=max(4_096, raw_bytes * _VARIABLE_RECORD_OVERHEAD_FACTOR),
                    )
                    try:
                        step = reconstructor.update(delta)
                    except BaseException as error:
                        classified = _classify_binance_data_error(error)
                        if classified is not None:
                            raise classified from error
                        raise
                    if step.observation is not None:
                        spools["book_observations"].append(step.observation, estimated_bytes=4_096)
                    else:
                        state.excluded_rows += 1
                    if step.gap is not None:
                        spools["sequence_gaps"].append(step.gap.to_record(), estimated_bytes=2_048)
                        state.sequence_gaps += 1
                    if step.outcome == "OBSERVED":
                        if step.observation is None:  # pragma: no cover - outcome invariant
                            raise RuntimeError("OBSERVED step lacks its book observation")
                        available_ns = step.observation["available_ts_ns"]
                        if isinstance(available_ns, bool) or not isinstance(available_ns, int):
                            raise RuntimeError(
                                "OBSERVED book availability timestamp is not an integer"
                            )
                        interval_tracker.observed(
                            delta.continuity_id,
                            available_ns,
                        )
                    else:
                        interval_tracker.excluded(delta.continuity_id)
                    if step.outcome == "STALE":
                        state.stale_rows += 1
                    state.reconstruction_status = reconstructor.status
                    if step.outcome in {"GAP", "INVALID", "EXCLUDED_AFTER_TERMINAL"}:
                        raise _DataCaptureIssue(
                            "SEQUENCE_CONTINUITY_FAILED",
                            "RECONSTRUCTION",
                            f"{normalized_symbol} reconstruction outcome was {step.outcome}",
                        )

                if journal.messages == 0:
                    raise _DataCaptureIssue(
                        "NO_IN_WINDOW_MESSAGES",
                        "RAW_CAPTURE",
                        "scheduled window contained no accepted depth frames",
                    )
                if self._clock_ns() < scheduled_end_ns:
                    raise _DataCaptureIssue(
                        "SCHEDULED_END_NOT_REACHED",
                        "RAW_CAPTURE",
                        "capture transport stopped before the absolute scheduled UTC end",
                    )
                completion_reason = "scheduled_end_reached"
            except BaseException as error:
                capture_error = _classify_binance_data_error(error) or error
            finally:
                if receiver_task is not None and not receiver_task.done():
                    receiver_task.cancel()
                    with suppress(BaseException):
                        await receiver_task

            finalization_error: BaseException | None = None
            try:
                for spool in spools.values():
                    spool.close()
                validation_reports = (
                    delta_validator.finish(),
                    observation_validator.finish(),
                )
                validators_finished = True
                raw_evidence = journal.publish(
                    capture_status=(
                        "raw_capture_complete" if capture_error is None else "incomplete_capture"
                    ),
                    error=capture_error,
                )
                normalized_root = stage / "normalized" / "captures" / capture_id
                time_columns = {
                    "book_snapshots": "received_ts_ns",
                    "depth_deltas": "event_ts_ns",
                    "book_observations": "event_ts_ns",
                    "sequence_gaps": "detected_ts_ns",
                }
                dataset_manifests: dict[str, dict[str, object]] = {}
                for schema_name, spool in spools.items():
                    stored = write_capture_parquet(
                        spool.iter_batches(),
                        root=normalized_root,
                        dataset=schema_name,
                        schema_name=schema_name,
                        venue="binance_spot",
                        symbol=normalized_symbol,
                        capture_id=capture_id,
                        source="binance_spot_public_live_capture_journal",
                        source_uri=str(raw_evidence.path.relative_to(stage)),
                        source_checksum_sha256=raw_evidence.sha256,
                        requested_start_ns=scheduled_start_ns,
                        requested_end_ns=scheduled_end_ns,
                        time_column=time_columns[schema_name],
                        max_input_batch_rows=_BATCH_ROWS,
                    )
                    if stored.rows != spool.rows:
                        raise RuntimeError(
                            f"stored {schema_name} rows differ from its verified Arrow spool"
                        )
                    dataset_manifests[schema_name] = {
                        "data_path": (
                            str(stored.data_path.relative_to(stage))
                            if stored.data_path is not None
                            else None
                        ),
                        "data_sha256": stored.data_sha256,
                        "manifest_path": str(stored.manifest_path.relative_to(stage)),
                        "manifest_sha256": stored.manifest_sha256,
                        "rows": stored.rows,
                    }

                quality_root = stage / "quality"
                quality_root.mkdir(parents=True, exist_ok=True)
                quality_paths: dict[str, str] = {}
                for report in validation_reports:
                    report_path = quality_root / f"{report.dataset}.validation.json"
                    report.write_json(report_path)
                    quality_paths[report.dataset] = report_path.relative_to(stage).as_posix()
                intervals = interval_tracker.finish()
                quality_errors = sum(item.error_count for item in validation_reports)
                quality_warnings = sum(item.warning_count for item in validation_reports)
                summary_path = quality_root / "capture.summary.json"
                inventory_without_summary = _artifacts(stage)
                summary_payload: dict[str, object] = {
                    "schema_version": "m8-binance-l2-symbol-capture-v1",
                    "generated_at_utc": utc_now_iso(),
                    "capture_id": capture_id,
                    "symbol": normalized_symbol,
                    "capture_status": "COMPLETE" if capture_error is None else "FAILED",
                    "completion_reason": completion_reason,
                    "failure_reason_code": (
                        capture_error.reason_code
                        if isinstance(capture_error, _DataCaptureIssue)
                        else None
                    ),
                    "failure_phase": (
                        capture_error.phase
                        if isinstance(capture_error, _DataCaptureIssue)
                        else None
                    ),
                    "failure_type": (
                        type(capture_error).__name__ if capture_error is not None else None
                    ),
                    "failure_message": (
                        str(capture_error)[:2048] if capture_error is not None else None
                    ),
                    "scheduled_range_ns": {
                        "start": scheduled_start_ns,
                        "end_exclusive": scheduled_end_ns,
                    },
                    "messages": journal.messages,
                    "first_raw_received_ns": journal.first_received_ns,
                    "last_raw_received_ns": journal.last_received_ns,
                    "normalized_rows": spools["depth_deltas"].rows,
                    "reconstructed_rows": spools["book_observations"].rows,
                    "excluded_rows": spools["depth_deltas"].rows - spools["book_observations"].rows,
                    "continuity_epochs": state.continuity_epochs,
                    "snapshot_anchors": journal.snapshot_anchors,
                    "sequence_gaps": spools["sequence_gaps"].rows,
                    "stale_rows": state.stale_rows,
                    "reconstruction_status": state.reconstruction_status,
                    "quality_errors": quality_errors,
                    "quality_warnings": quality_warnings,
                    "quality_reports": quality_paths,
                    "valid_observed_intervals": [item.to_dict() for item in intervals],
                    "raw_journal": raw_evidence.path.relative_to(stage).as_posix(),
                    "raw_journal_sha256": raw_evidence.sha256,
                    "raw_journal_manifest": raw_evidence.manifest_path.relative_to(
                        stage
                    ).as_posix(),
                    "raw_journal_manifest_sha256": raw_evidence.manifest_sha256,
                    "normalized_dataset_manifests": dataset_manifests,
                    "artifact_inventory_without_summary": [
                        {
                            "path": item.path.relative_to(stage).as_posix(),
                            "kind": item.kind,
                            "sha256": item.sha256,
                            "bytes": item.path.stat().st_size,
                        }
                        for item in inventory_without_summary
                    ],
                    "receiver_queue_capacity": queue_capacity,
                    "receiver_queue_estimated_byte_budget": queue_byte_budget,
                    "max_receiver_queue_depth": state.max_queue_depth,
                    "max_receiver_queue_estimated_bytes": state.max_queue_estimated_bytes,
                    "max_raw_frame_bytes_observed": journal.max_frame_bytes_observed,
                    "max_arrow_batch_bytes_observed": max(
                        spool.max_batch_bytes_observed for spool in spools.values()
                    ),
                    "live_trading": False,
                    "policy": (
                        "raw websocket bytes precede parsing; snapshots are fresh per continuity; "
                        "only consecutive OBSERVED receipts with no >5s silence form coverage"
                    ),
                }
                write_json(summary_path, summary_payload)
            except BaseException as error:
                finalization_error = error
            finally:
                if not validators_finished:
                    delta_validator.close()
                    observation_validator.close()
                if raw_evidence is None:
                    with suppress(BaseException):
                        raw_evidence = journal.publish(
                            capture_status="finalization_failure", error=finalization_error
                        )
                journal.close_without_publish()

            if finalization_error is not None:
                if capture_error is not None:
                    finalization_error.add_note(
                        f"capture also failed with {type(capture_error).__name__}: {capture_error}"
                    )
                raise finalization_error

            if validation_reports is None or raw_evidence is None:
                raise RuntimeError("capture finalization omitted required evidence")
            intervals = interval_tracker.finish()
            quality_errors = sum(item.error_count for item in validation_reports)
            quality_warnings = sum(item.warning_count for item in validation_reports)
            result = SymbolCaptureResult(
                symbol=normalized_symbol,
                capture_id=capture_id,
                status="FAILED" if capture_error is not None else "COMPLETE",
                completion_reason=completion_reason,
                reconstruction_status=state.reconstruction_status,
                messages=journal.messages,
                normalized_rows=spools["depth_deltas"].rows,
                reconstructed_rows=spools["book_observations"].rows,
                excluded_rows=spools["depth_deltas"].rows - spools["book_observations"].rows,
                continuity_epochs=state.continuity_epochs,
                snapshot_anchors=journal.snapshot_anchors,
                sequence_gaps=spools["sequence_gaps"].rows,
                quality_errors=quality_errors,
                quality_warnings=quality_warnings,
                max_raw_frame_bytes_observed=journal.max_frame_bytes_observed,
                max_arrow_batch_bytes_observed=max(
                    spool.max_batch_bytes_observed for spool in spools.values()
                ),
                first_raw_received_ns=journal.first_received_ns,
                last_raw_received_ns=journal.last_received_ns,
                valid_observed_intervals=intervals,
                artifacts=_artifacts(stage),
                failure_reason_code=(
                    capture_error.reason_code
                    if isinstance(capture_error, _DataCaptureIssue)
                    else None
                ),
                failure_phase=(
                    capture_error.phase if isinstance(capture_error, _DataCaptureIssue) else None
                ),
            )
            if capture_error is None:
                return result
            if isinstance(capture_error, _DataCaptureIssue):
                raise M8L2DataFailure(
                    capture_error.reason_code,
                    phase=capture_error.phase,
                    message=str(capture_error),
                    partial_result=result,
                ) from capture_error
            raise capture_error


__all__ = ["BinanceM8L2Capture"]
