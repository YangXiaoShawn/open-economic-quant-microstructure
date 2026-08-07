"""Command-line interface for research data, reproduction, and reporting."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure import __version__
from microstructure.config import ProjectConfig, datetime_to_ns, load_config
from microstructure.data.binance import (
    BinanceLiveDepthCollector,
    BinancePublicClient,
    CapturedDepth,
    RawDepthFrame,
)
from microstructure.data.book import BookSnapshot, IncrementalBookReconstructor
from microstructure.data.quality import IncrementalQualityValidator, ValidationReport
from microstructure.data.schemas import get_schema, table_from_records
from microstructure.data.storage import write_capture_parquet, write_source_manifest
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.ingestion import (
    IngestionResult,
    ingest_from_config,
    validate_configured_input,
)
from microstructure.pipeline import reproduce
from microstructure.provenance import read_json, sha256_file, utc_now_iso, write_json
from microstructure.reporting import load_run_bundle, verify_checksums, write_report_set


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _ingestion_payload(result: IngestionResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "evidence_tier": result.evidence_tier,
        "output_root": result.output_root,
        "ingestion_manifest": result.ingestion_manifest_path,
        "ingestion_manifest_sha256": result.ingestion_manifest_sha256,
        "rows": result.rows,
        "datasets": [
            {
                "schema": dataset.schema_name,
                "rows": dataset.rows,
                "manifest": dataset.storage.manifest_path,
                "manifest_sha256": dataset.storage.manifest_sha256,
                "quality_errors": dataset.validation.error_count,
                "quality_warnings": dataset.validation.warning_count,
            }
            for dataset in result.datasets
        ],
        "raw_artifacts": len(result.raw_artifacts),
        "symbols": [
            {
                "symbol": item.symbol,
                "rows": item.rows,
                "complete_range": item.complete_range,
                "tick_size": item.metadata.tick_size,
                "lot_size": item.metadata.lot_size,
            }
            for item in result.symbols
        ],
        "quality": {
            "passed": result.validation.passed,
            "rows_checked": result.validation.rows_checked,
            "errors": result.validation.error_count,
            "warnings": result.validation.warning_count,
        },
    }


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_root = (
        Path(args.output_root).resolve() if args.output_root else config.data.partition_root.parent
    )
    result = ingest_from_config(config, output_root)
    _print_json(_ingestion_payload(result))
    return 0


def _synthetic_tables(config: ProjectConfig) -> dict[str, Any]:
    events = config.data.events_per_symbol
    if events is None:
        raise ValueError("synthetic validation requires data.events_per_symbol")
    generated = generate_synthetic_market(
        symbols=config.data.symbols,
        events_per_symbol=events,
        start_ts_ns=datetime_to_ns(config.data.start),
        seed=config.run.seed,
    )
    return {"trades": generated.trades, "book_observations": generated.book_observations}


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tables = _synthetic_tables(config) if config.data.mode == "synthetic" else None
    summary = validate_configured_input(config, tables=tables)
    _print_json(
        {
            "passed": summary.passed,
            "rows_checked": summary.rows_checked,
            "errors": summary.error_count,
            "warnings": summary.warning_count,
            "reports": [
                {
                    "dataset": report.dataset,
                    "rows_checked": report.rows_checked,
                    "errors": report.error_count,
                    "warnings": report.warning_count,
                }
                for report in summary.reports
            ],
            "mutation_policy": "validation did not repair or replace observations",
        }
    )
    return 0 if summary.passed else 1


def _cmd_reproduce(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = reproduce(
        config,
        Path(args.run_dir),
        ingestion_manifest_path=args.ingestion_manifest,
        ingestion_manifest_sha256=args.ingestion_manifest_sha256,
    )
    bundle = load_run_bundle(output)
    _print_json(
        {
            "run_dir": output,
            "run_id": bundle.run_id,
            "evidence_tier": bundle.evidence_tier,
            "observed_start_utc": bundle.observed_start_utc,
            "observed_end_utc": bundle.observed_end_utc,
            "status": "complete",
        }
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    bundle = load_run_bundle(args.run_dir)
    protected = verify_checksums(args.run_dir)
    _print_json(
        {
            "run_dir": bundle.root,
            "run_id": bundle.run_id,
            "evidence_tier": bundle.evidence_tier,
            "protected_files": protected,
            "integrity": "verified",
        }
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    bundle = load_run_bundle(args.run_dir)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else bundle.root.with_name(f"{bundle.root.name}-reports")
    )
    paths = write_report_set(bundle, output)
    _print_json({"run_id": bundle.run_id, "output_dir": output, **asdict(paths)})
    return 0


_LIVE_BATCH_ROWS = 1_024
_MAX_LIVE_RAW_MESSAGE_BYTES = 1 * 1024 * 1024
_LIVE_BATCH_ESTIMATED_BYTES = 16 * 1024 * 1024
_VARIABLE_RECORD_OVERHEAD_FACTOR = 8


@dataclass(frozen=True, slots=True)
class DepthCaptureResult:
    symbol: str
    messages: int
    continuity_epochs: int
    reconstruction_status: Literal["LIVE", "GAPPED", "INVALID"]
    book_observations: int
    sequence_gaps: int
    stale_events: int
    excluded_messages: int
    final_update_id: int
    quality_errors: int
    quality_warnings: int
    raw_path: Path
    raw_manifest_path: Path
    raw_manifest_sha256: str
    summary_path: Path
    completion_reason: str
    requested_duration_seconds: float | None
    elapsed_monotonic_seconds: float
    receipt_coverage_seconds: float
    max_continuity_epoch_seconds: float


@dataclass(slots=True)
class _DepthCaptureStop:
    reason: str = "not_started"
    elapsed_monotonic_seconds: float = 0.0


@dataclass(slots=True)
class _DepthEpochCoverage:
    continuity_id: str
    snapshot_id: str
    first_received_ns: int
    last_received_ns: int
    messages: int = 0
    book_observations: int = 0
    excluded_messages: int = 0
    sequence_gaps: int = 0
    reconstruction_status: Literal["LIVE", "GAPPED", "INVALID"] = "LIVE"
    final_update_id: int = 0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.last_received_ns - self.first_received_ns) / 1_000_000_000.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "continuity_id": self.continuity_id,
            "snapshot_id": self.snapshot_id,
            "first_received_ns": self.first_received_ns,
            "last_received_ns": self.last_received_ns,
            "duration_seconds": self.duration_seconds,
            "messages": self.messages,
            "book_observations": self.book_observations,
            "excluded_messages": self.excluded_messages,
            "sequence_gaps": self.sequence_gaps,
            "reconstruction_status": self.reconstruction_status,
            "final_update_id": self.final_update_id,
        }


async def _bounded_depth_items(
    collector: BinanceLiveDepthCollector,
    *,
    max_messages: int,
    duration_seconds: float | None,
    stop: _DepthCaptureStop,
) -> AsyncIterator[CapturedDepth]:
    """Yield a live stream until its safety cap or a graceful duration deadline."""

    started = time.monotonic()
    yielded = 0
    iterator = collector.stream(max_messages=max_messages).__aiter__()
    try:
        if duration_seconds is None:
            async for item in iterator:
                yielded += 1
                yield item
            stop.reason = "message_limit" if yielded == max_messages else "stream_ended_early"
            return

        deadline = started + duration_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop.reason = "duration_elapsed"
                return
            try:
                item = await asyncio.wait_for(anext(iterator), timeout=remaining)
            except TimeoutError:
                stop.reason = "duration_elapsed"
                return
            except StopAsyncIteration:
                stop.reason = (
                    "message_safety_ceiling" if yielded == max_messages else "stream_ended_early"
                )
                return
            yielded += 1
            yield item
    finally:
        stop.elapsed_monotonic_seconds = max(0.0, time.monotonic() - started)
        with suppress(BaseException):
            closer = getattr(iterator, "aclose", None)
            if callable(closer):
                await closer()


@dataclass(frozen=True, slots=True)
class _PublishedRawCapture:
    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str


class _ArrowBatchSpool:
    """Bounded record buffer backed by a temporary Arrow IPC stream."""

    def __init__(
        self,
        *,
        root: Path,
        schema_name: str,
        batch_rows: int,
        max_buffer_bytes: int,
        on_batch: Callable[[pa.RecordBatch], None] | None = None,
    ) -> None:
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        if max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be positive")
        self.schema_name = schema_name
        self.batch_rows = batch_rows
        self.max_buffer_bytes = max_buffer_bytes
        self.path = root / f"{schema_name}.arrow"
        self._handle = self.path.open("wb")
        self._writer = pa.ipc.new_stream(self._handle, get_schema(schema_name))
        self._on_batch = on_batch
        self._records: list[Mapping[str, object]] = []
        self.rows = 0
        self.max_buffered_rows = 0
        self.max_buffered_estimated_bytes = 0
        self._buffered_estimated_bytes = 0
        self._closed = False

    def append(self, record: Mapping[str, object], *, estimated_bytes: int) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed Arrow spool")
        if estimated_bytes < 1:
            raise ValueError("estimated_bytes must be positive")
        if estimated_bytes > self.max_buffer_bytes:
            raise RuntimeError(f"one {self.schema_name} record exceeds the bounded batch estimate")
        if (
            self._records
            and self._buffered_estimated_bytes + estimated_bytes > self.max_buffer_bytes
        ):
            self._flush()
        self._records.append(record)
        self._buffered_estimated_bytes += estimated_bytes
        self.max_buffered_rows = max(self.max_buffered_rows, len(self._records))
        self.max_buffered_estimated_bytes = max(
            self.max_buffered_estimated_bytes,
            self._buffered_estimated_bytes,
        )
        if len(self._records) >= self.batch_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        table = table_from_records(self.schema_name, self._records)
        batches = table.to_batches(max_chunksize=self.batch_rows)
        if len(batches) != 1 or batches[0].num_rows > self.batch_rows:
            raise RuntimeError(f"failed to construct one bounded {self.schema_name} batch")
        batch = batches[0]
        if self._on_batch is not None:
            self._on_batch(batch)
        self._writer.write_batch(batch)
        self.rows += batch.num_rows
        self._records.clear()
        self._buffered_estimated_bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._writer.close()
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
        self._closed = True

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        if not self._closed:
            raise RuntimeError("Arrow spool must be closed before it can be read")
        with self.path.open("rb") as handle:
            reader = pa.ipc.open_stream(handle)
            for batch in reader:
                if batch.num_rows > self.batch_rows:
                    raise RuntimeError(
                        f"spooled {self.schema_name} batch exceeds {self.batch_rows} rows"
                    )
                yield batch


class _RawMessageSpool:
    """Incrementally persist an exact, typed live-capture journal."""

    def __init__(self, *, root: Path, symbol: str, source_uri: str) -> None:
        self.root = root
        self.symbol = symbol
        self.source_uri = source_uri
        self.directory = root / "raw" / "binance_spot" / "depth_stream" / symbol
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory,
            prefix=".capture-",
            suffix=".ndjson.tmp",
            text=True,
        )
        self._temporary_path = Path(temporary_name)
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self.messages = 0
        self.snapshot_anchors = 0
        self.first_received_ns: int | None = None
        self.last_received_ns: int | None = None
        self._closed = False
        self.published_path: Path | None = None
        self._last_frame_identity: tuple[str, int, int, str] | None = None

    @property
    def evidence_path(self) -> Path:
        """Return the durable path, or the fsynced temporary path after publish failure."""
        return self.published_path or self._temporary_path

    def _write_event(self, event: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed raw capture")
        json.dump(event, self._handle, sort_keys=True, separators=(",", ":"))
        self._handle.write("\n")

    def append_frame(self, frame: RawDepthFrame) -> int:
        """Journal one exact frame before any UTF-8 or JSON parsing."""
        payload_size = len(frame.payload)
        payload_sha256 = hashlib.sha256(frame.payload).hexdigest()
        self._write_event(
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
        if self.first_received_ns is None:
            self.first_received_ns = frame.received_ts_ns
        self.last_received_ns = frame.received_ts_ns
        self._last_frame_identity = (
            frame.continuity_id,
            frame.capture_seq,
            frame.received_ts_ns,
            payload_sha256,
        )
        # The oversize frame is intentionally journaled before capture fails.
        if payload_size > _MAX_LIVE_RAW_MESSAGE_BYTES:
            raise RuntimeError(
                f"live depth message exceeds {_MAX_LIVE_RAW_MESSAGE_BYTES} raw bytes"
            )
        return payload_size

    def append_captured(self, item: CapturedDepth) -> int:
        """Verify callback lineage, with a fallback for injected legacy collectors."""
        received_ts_ns = item.delta.received_ts_ns
        capture_seq = item.delta.capture_seq
        if received_ts_ns is None or capture_seq is None:
            raise RuntimeError("captured depth messages require receipt time and capture sequence")
        payload = item.raw_payload.encode("utf-8")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        identity = (
            item.delta.continuity_id,
            capture_seq,
            received_ts_ns,
            payload_sha256,
        )
        if self._last_frame_identity != identity:
            self.append_frame(
                RawDepthFrame(
                    payload=payload,
                    was_text=True,
                    received_ts_ns=received_ts_ns,
                    capture_seq=capture_seq,
                    continuity_id=item.delta.continuity_id,
                )
            )
        if item.delta.source_artifact_id != payload_sha256:
            raise RuntimeError("normalized depth delta is not bound to its raw frame SHA-256")
        return len(payload)

    def append_snapshot(self, snapshot: BookSnapshot) -> None:
        """Bind one REST snapshot raw artifact into the capture journal."""
        raw_path = (
            self.root
            / "raw"
            / "binance_spot"
            / "depth_snapshots"
            / snapshot.symbol
            / f"{snapshot.source_artifact_id}.json"
        )
        if not raw_path.is_file() or sha256_file(raw_path) != snapshot.source_artifact_id:
            raise RuntimeError("book snapshot is not bound to its preserved raw response")
        manifest_path: Path | None = None
        for candidate in raw_path.parent.glob(f"{raw_path.name}.manifest-*.json"):
            payload = read_json(candidate)
            if (
                isinstance(payload, dict)
                and payload.get("path") == raw_path.name
                and isinstance(payload.get("checksum"), dict)
                and payload["checksum"].get("value") == snapshot.source_artifact_id
                and (manifest_path is None or candidate.name > manifest_path.name)
            ):
                manifest_path = candidate
        if manifest_path is None:
            raise RuntimeError("book snapshot raw response has no valid source manifest")
        self._write_event(
            {
                "continuity_id": snapshot.continuity_id,
                "event_kind": "rest_snapshot_anchor",
                "last_update_id": snapshot.last_update_id,
                "raw_manifest_path": str(manifest_path),
                "raw_manifest_sha256": sha256_file(manifest_path),
                "raw_path": str(raw_path),
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

    def publish(
        self,
        *,
        status: str,
        error: BaseException | None = None,
    ) -> _PublishedRawCapture:
        self._close()
        if self.published_path is not None:
            destination = self.published_path
            digest = sha256_file(destination)
        else:
            digest = sha256_file(self._temporary_path)
            prefix = "capture" if status == "raw_capture_complete" else "capture-failed"
            destination = self.directory / f"{prefix}-{digest}.ndjson"
            if destination.exists():
                if sha256_file(destination) != digest:
                    raise RuntimeError(f"raw depth capture collision at {destination}")
                self._temporary_path.unlink(missing_ok=True)
            else:
                os.replace(self._temporary_path, destination)
            self.published_path = destination
        response_headers = {
            "x-local-capture-status": status,
            "x-local-journal-format": "typed-base64-frames-v1",
            "x-local-message-count": str(self.messages),
            "x-local-snapshot-anchor-count": str(self.snapshot_anchors),
        }
        if error is not None:
            response_headers["x-local-error-type"] = type(error).__name__
            response_headers["x-local-error"] = str(error)[:512]
        manifest_path, manifest_sha = write_source_manifest(
            destination,
            source="binance_spot_public_live_capture_journal",
            source_uri=self.source_uri,
            downloaded_at_utc=utc_now_iso(),
            requested_start_ns=self.first_received_ns,
            requested_end_ns=(
                self.last_received_ns + 1 if self.last_received_ns is not None else None
            ),
            response_headers=response_headers,
        )
        return _PublishedRawCapture(
            path=destination,
            sha256=digest,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
        )

    def close_without_deleting(self) -> None:
        self._close()


def _status_max(
    current: Literal["LIVE", "GAPPED", "INVALID"],
    observed: Literal["LIVE", "GAPPED", "INVALID"],
) -> Literal["LIVE", "GAPPED", "INVALID"]:
    rank = {"LIVE": 0, "GAPPED": 1, "INVALID": 2}
    return observed if rank[observed] > rank[current] else current


def _failure_record(
    *,
    output_root: Path,
    capture_id: str,
    symbol: str,
    raw_spool: _RawMessageSpool,
    raw_evidence: _PublishedRawCapture | None,
    error: BaseException,
) -> None:
    write_json(
        output_root / "quality" / f"live_depth_capture.{capture_id}.failed.json",
        {
            "generated_at_utc": utc_now_iso(),
            "capture_id": capture_id,
            "capture_status": "FAILED",
            "symbol": symbol,
            "messages_preserved": raw_spool.messages,
            "raw_path": str(
                raw_evidence.path if raw_evidence is not None else raw_spool.evidence_path
            ),
            "raw_manifest": (str(raw_evidence.manifest_path) if raw_evidence is not None else None),
            "error_type": type(error).__name__,
            "error": str(error),
            "completion_manifest_published": False,
        },
    )


async def _capture_depth(
    *,
    symbol: str,
    max_messages: int,
    output_root: Path,
    duration_seconds: float | None = None,
) -> DepthCaptureResult:
    if max_messages < 1:
        raise ValueError("max_messages must be positive")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive when supplied")
    output_root.mkdir(parents=True, exist_ok=True)
    capture_id = f"{symbol.lower()}-{time.time_ns()}"
    client = BinancePublicClient()
    metadata = client.fetch_exchange_info(symbol=symbol, raw_root=output_root / "raw")
    raw_spool_holder: list[_RawMessageSpool] = []

    def preserve_raw_frame(frame: RawDepthFrame) -> None:
        if not raw_spool_holder:  # pragma: no cover - collector cannot run during construction
            raise RuntimeError("raw capture journal is not initialized")
        raw_spool_holder[0].append_frame(frame)

    collector = BinanceLiveDepthCollector(
        symbols=(symbol,),
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
        on_raw_frame=preserve_raw_frame,
    )
    raw_spool = _RawMessageSpool(root=output_root, symbol=symbol, source_uri=collector.url)
    raw_spool_holder.append(raw_spool)
    raw_evidence: _PublishedRawCapture | None = None
    delta_validator = IncrementalQualityValidator(
        "depth_deltas",
        row_chunk_size=_LIVE_BATCH_ROWS,
    )
    observation_validator = IncrementalQualityValidator(
        "book_observations",
        row_chunk_size=_LIVE_BATCH_ROWS,
    )
    validators_finished = False
    with tempfile.TemporaryDirectory(
        dir=output_root,
        prefix=f".{capture_id}.spool-",
    ) as temporary_root_name:
        temporary_root = Path(temporary_root_name)
        spools = {
            "book_snapshots": _ArrowBatchSpool(
                root=temporary_root,
                schema_name="book_snapshots",
                batch_rows=_LIVE_BATCH_ROWS,
                max_buffer_bytes=_LIVE_BATCH_ESTIMATED_BYTES,
            ),
            "depth_deltas": _ArrowBatchSpool(
                root=temporary_root,
                schema_name="depth_deltas",
                batch_rows=_LIVE_BATCH_ROWS,
                max_buffer_bytes=_LIVE_BATCH_ESTIMATED_BYTES,
                on_batch=delta_validator.update,
            ),
            "book_observations": _ArrowBatchSpool(
                root=temporary_root,
                schema_name="book_observations",
                batch_rows=_LIVE_BATCH_ROWS,
                max_buffer_bytes=_LIVE_BATCH_ESTIMATED_BYTES,
                on_batch=observation_validator.update,
            ),
            "sequence_gaps": _ArrowBatchSpool(
                root=temporary_root,
                schema_name="sequence_gaps",
                batch_rows=_LIVE_BATCH_ROWS,
                max_buffer_bytes=_LIVE_BATCH_ESTIMATED_BYTES,
            ),
        }
        current_continuity_id: str | None = None
        reconstructor: IncrementalBookReconstructor | None = None
        continuity_epochs = 0
        status: Literal["LIVE", "GAPPED", "INVALID"] = "LIVE"
        stale_events = 0
        excluded_messages = 0
        final_update_id = 0
        capture_stop = _DepthCaptureStop()
        epoch_coverage: list[_DepthEpochCoverage] = []
        current_epoch_coverage: _DepthEpochCoverage | None = None
        try:
            async for item in _bounded_depth_items(
                collector,
                max_messages=max_messages,
                duration_seconds=duration_seconds,
                stop=capture_stop,
            ):
                raw_message_bytes = raw_spool.append_captured(item)
                if item.delta.continuity_id != current_continuity_id:
                    snapshot = client.fetch_depth_snapshot(
                        symbol=symbol,
                        raw_root=output_root / "raw",
                        continuity_id=item.delta.continuity_id,
                        tick_size=metadata.tick_size,
                        lot_size=metadata.lot_size,
                    )
                    raw_spool.append_snapshot(snapshot)
                    snapshot_estimated_bytes = max(
                        4_096,
                        128 * (len(snapshot.bids) + len(snapshot.asks)),
                    )
                    spools["book_snapshots"].append(
                        snapshot.to_record(),
                        estimated_bytes=snapshot_estimated_bytes,
                    )
                    reconstructor = IncrementalBookReconstructor(snapshot)
                    current_continuity_id = item.delta.continuity_id
                    continuity_epochs += 1
                    received_ts_ns = item.delta.received_ts_ns
                    if received_ts_ns is None:  # pragma: no cover - append_captured validates
                        raise RuntimeError("live depth delta has no receipt timestamp")
                    current_epoch_coverage = _DepthEpochCoverage(
                        continuity_id=item.delta.continuity_id,
                        snapshot_id=snapshot.snapshot_id,
                        first_received_ns=received_ts_ns,
                        last_received_ns=received_ts_ns,
                    )
                    epoch_coverage.append(current_epoch_coverage)
                if reconstructor is None:  # pragma: no cover - guarded by epoch creation
                    raise RuntimeError("live depth epoch has no reconstructor")
                if current_epoch_coverage is None:  # pragma: no cover - guarded by epoch creation
                    raise RuntimeError("live depth epoch has no coverage tracker")
                spools["depth_deltas"].append(
                    item.delta.to_record(),
                    estimated_bytes=max(
                        4_096,
                        raw_message_bytes * _VARIABLE_RECORD_OVERHEAD_FACTOR,
                    ),
                )
                step = reconstructor.update(item.delta)
                if step.observation is not None:
                    spools["book_observations"].append(
                        step.observation,
                        estimated_bytes=4_096,
                    )
                else:
                    excluded_messages += 1
                if step.gap is not None:
                    spools["sequence_gaps"].append(
                        step.gap.to_record(),
                        estimated_bytes=2_048,
                    )
                if step.outcome == "STALE":
                    stale_events += 1
                status = _status_max(status, reconstructor.status)
                final_update_id = reconstructor.final_update_id
                received_ts_ns = item.delta.received_ts_ns
                if received_ts_ns is None:  # pragma: no cover - append_captured validates
                    raise RuntimeError("live depth delta has no receipt timestamp")
                current_epoch_coverage.last_received_ns = received_ts_ns
                current_epoch_coverage.messages += 1
                current_epoch_coverage.book_observations += int(step.observation is not None)
                current_epoch_coverage.excluded_messages += int(step.observation is None)
                current_epoch_coverage.sequence_gaps += int(step.gap is not None)
                current_epoch_coverage.reconstruction_status = reconstructor.status
                current_epoch_coverage.final_update_id = reconstructor.final_update_id

            if duration_seconds is None:
                if raw_spool.messages != max_messages or capture_stop.reason != "message_limit":
                    raise RuntimeError(
                        f"live depth stream ended after {raw_spool.messages} of "
                        f"{max_messages} requested messages"
                    )
            else:
                if capture_stop.reason == "message_safety_ceiling":
                    raise RuntimeError(
                        "live depth message safety ceiling was reached before the requested "
                        "capture duration elapsed"
                    )
                if capture_stop.reason != "duration_elapsed":
                    raise RuntimeError(
                        "live depth stream ended before the requested capture duration "
                        f"({capture_stop.reason})"
                    )
                if raw_spool.messages == 0:
                    raise RuntimeError("duration-bounded live depth capture received no messages")
            if raw_spool.snapshot_anchors != continuity_epochs:
                raise RuntimeError("not every continuity epoch has a raw snapshot anchor")
            for spool in spools.values():
                spool.close()
            if spools["depth_deltas"].rows != raw_spool.messages:
                raise RuntimeError("captured and normalized depth-message counts diverged")
            if spools["book_observations"].rows + excluded_messages != raw_spool.messages:
                raise RuntimeError("not every normalized depth message has an explicit outcome")
            delta_quality = delta_validator.finish()
            observation_quality = observation_validator.finish()
            validators_finished = True
            raw_evidence = raw_spool.publish(status="raw_capture_complete")

            normalized_root = output_root / "normalized" / "captures" / capture_id
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
                    symbol=symbol,
                    capture_id=capture_id,
                    source="binance_spot_public_live_capture_journal",
                    source_uri=str(raw_evidence.path),
                    source_checksum_sha256=raw_evidence.sha256,
                    requested_start_ns=raw_spool.first_received_ns,
                    requested_end_ns=(
                        raw_spool.last_received_ns + 1
                        if raw_spool.last_received_ns is not None
                        else None
                    ),
                    time_column=time_columns[schema_name],
                    max_input_batch_rows=_LIVE_BATCH_ROWS,
                )
                if stored.rows != spool.rows:
                    raise RuntimeError(
                        f"stored {schema_name} row count does not match its verified spool"
                    )
                dataset_manifests[schema_name] = {
                    "data_path": str(stored.data_path) if stored.data_path is not None else None,
                    "data_sha256": stored.data_sha256,
                    "manifest_path": str(stored.manifest_path),
                    "manifest_sha256": stored.manifest_sha256,
                    "rows": stored.rows,
                }

            quality_reports: tuple[ValidationReport, ...] = (
                delta_quality,
                observation_quality,
            )
            quality_errors = sum(report.error_count for report in quality_reports)
            quality_warnings = sum(report.warning_count for report in quality_reports)
            quality_root = output_root / "quality"
            quality_root.mkdir(parents=True, exist_ok=True)
            quality_report_paths: dict[str, str] = {}
            for report in quality_reports:
                report_path = quality_root / f"live_{report.dataset}.{capture_id}.validation.json"
                report.write_json(report_path)
                quality_report_paths[report.dataset] = str(report_path)
            summary_path = quality_root / f"live_depth_capture.{capture_id}.summary.json"
            receipt_coverage_seconds = (
                (raw_spool.last_received_ns - raw_spool.first_received_ns) / 1_000_000_000.0
                if raw_spool.first_received_ns is not None
                and raw_spool.last_received_ns is not None
                else 0.0
            )
            max_continuity_epoch_seconds = max(
                (epoch.duration_seconds for epoch in epoch_coverage),
                default=0.0,
            )
            summary_payload = {
                "generated_at_utc": utc_now_iso(),
                "capture_id": capture_id,
                "capture_status": "COMPLETE",
                "messages": raw_spool.messages,
                "continuity_epochs": continuity_epochs,
                "normalized_messages": spools["depth_deltas"].rows,
                "book_observations": spools["book_observations"].rows,
                "sequence_gaps": spools["sequence_gaps"].rows,
                "stale_events": stale_events,
                "excluded_messages": excluded_messages,
                "reconstruction_status": status,
                "quality_errors": quality_errors,
                "quality_warnings": quality_warnings,
                "completion_reason": capture_stop.reason,
                "requested_duration_seconds": duration_seconds,
                "message_safety_ceiling": max_messages,
                "elapsed_monotonic_seconds": capture_stop.elapsed_monotonic_seconds,
                "receipt_coverage_seconds": receipt_coverage_seconds,
                "continuity_epoch_coverage": [epoch.to_dict() for epoch in epoch_coverage],
                "max_continuity_epoch_seconds": max_continuity_epoch_seconds,
                "quality_reports": quality_report_paths,
                "raw_path": str(raw_evidence.path),
                "raw_manifest": str(raw_evidence.manifest_path),
                "raw_manifest_sha256": raw_evidence.manifest_sha256,
                "normalized_dataset_manifests": dataset_manifests,
                "max_buffered_rows_per_dataset": {
                    name: spool.max_buffered_rows for name, spool in spools.items()
                },
                "max_buffered_estimated_bytes_per_dataset": {
                    name: spool.max_buffered_estimated_bytes for name, spool in spools.items()
                },
                "policy": (
                    "every continuity transition receives a fresh snapshot; every captured "
                    "delta is normalized, and non-observed deltas are counted or gap-audited"
                ),
            }
            # The capture-ID-specific completion marker is authoritative and published last.
            write_json(summary_path, summary_payload)
            with suppress(BaseException):
                write_json(
                    quality_root / "live_depth_capture.summary.json",
                    {
                        **summary_payload,
                        "capture_status": "LATEST_POINTER",
                        "latest_capture_status": summary_payload["capture_status"],
                        "authoritative_summary_path": str(summary_path),
                        "authoritative_summary_sha256": sha256_file(summary_path),
                    },
                )
            return DepthCaptureResult(
                symbol=symbol,
                messages=raw_spool.messages,
                continuity_epochs=continuity_epochs,
                reconstruction_status=status,
                book_observations=spools["book_observations"].rows,
                sequence_gaps=spools["sequence_gaps"].rows,
                stale_events=stale_events,
                excluded_messages=excluded_messages,
                final_update_id=final_update_id,
                quality_errors=quality_errors,
                quality_warnings=quality_warnings,
                raw_path=raw_evidence.path,
                raw_manifest_path=raw_evidence.manifest_path,
                raw_manifest_sha256=raw_evidence.manifest_sha256,
                summary_path=summary_path,
                completion_reason=capture_stop.reason,
                requested_duration_seconds=duration_seconds,
                elapsed_monotonic_seconds=capture_stop.elapsed_monotonic_seconds,
                receipt_coverage_seconds=receipt_coverage_seconds,
                max_continuity_epoch_seconds=max_continuity_epoch_seconds,
            )
        except BaseException as error:
            if raw_evidence is None:
                try:
                    raw_evidence = raw_spool.publish(
                        status="incomplete_capture_failure",
                        error=error,
                    )
                except BaseException:
                    raw_spool.close_without_deleting()
            else:
                with suppress(BaseException):
                    raw_evidence = raw_spool.publish(
                        status="normalization_failure",
                        error=error,
                    )
            with suppress(BaseException):
                _failure_record(
                    output_root=output_root,
                    capture_id=capture_id,
                    symbol=symbol,
                    raw_spool=raw_spool,
                    raw_evidence=raw_evidence,
                    error=error,
                )
            raise
        finally:
            for spool in spools.values():
                with suppress(BaseException):
                    spool.close()
            if not validators_finished:
                delta_validator.close()
                observation_validator.close()


def _cmd_collect_l2(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    result = asyncio.run(
        _capture_depth(
            symbol=str(args.symbol).upper(),
            max_messages=int(args.max_messages),
            output_root=output_root,
            duration_seconds=(
                float(args.duration_seconds) if args.duration_seconds is not None else None
            ),
        )
    )
    _print_json(
        {
            "symbol": result.symbol,
            "messages": result.messages,
            "reconstruction_status": result.reconstruction_status,
            "book_observations": result.book_observations,
            "sequence_gaps": result.sequence_gaps,
            "stale_events": result.stale_events,
            "excluded_messages": result.excluded_messages,
            "final_update_id": result.final_update_id,
            "quality_errors": result.quality_errors,
            "quality_warnings": result.quality_warnings,
            "raw_manifest": result.raw_manifest_path,
            "raw_manifest_sha256": result.raw_manifest_sha256,
            "capture_summary": result.summary_path,
            "completion_reason": result.completion_reason,
            "requested_duration_seconds": result.requested_duration_seconds,
            "elapsed_monotonic_seconds": result.elapsed_monotonic_seconds,
            "receipt_coverage_seconds": result.receipt_coverage_seconds,
            "max_continuity_epoch_seconds": result.max_continuity_epoch_seconds,
            "output_root": output_root,
            "live_trading": False,
        }
    )
    return 0 if result.reconstruction_status == "LIVE" and result.quality_errors == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microstructure",
        description="Research-only event-driven market-microstructure system",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest configured synthetic or public data")
    ingest.add_argument("--config", required=True, type=Path)
    ingest.add_argument("--output-root", type=Path)
    ingest.set_defaults(handler=_cmd_ingest)

    validate = subparsers.add_parser("validate", help="run non-mutating data validation")
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=_cmd_validate)

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="produce or verify an immutable end-to-end sample run"
    )
    reproduce_parser.add_argument("--config", required=True, type=Path)
    reproduce_parser.add_argument("--run-dir", required=True, type=Path)
    reproduce_parser.add_argument(
        "--ingestion-manifest",
        type=Path,
        help="explicit public ingestion manifest; required with its SHA-256 for public mode",
    )
    reproduce_parser.add_argument(
        "--ingestion-manifest-sha256",
        help="SHA-256 of --ingestion-manifest; required for public mode",
    )
    reproduce_parser.set_defaults(handler=_cmd_reproduce)

    verify = subparsers.add_parser("verify", help="verify a frozen run and all checksums")
    verify.add_argument("--run-dir", required=True, type=Path)
    verify.set_defaults(handler=_cmd_verify)

    report = subparsers.add_parser("report", help="render reports from a frozen run")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--output-dir", type=Path)
    report.set_defaults(handler=_cmd_report)

    collect = subparsers.add_parser(
        "collect-l2", help="capture and reconstruct public live L2 data; never place orders"
    )
    collect.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), required=True)
    collect.add_argument("--max-messages", type=int, default=1_000)
    collect.add_argument(
        "--duration-seconds",
        type=float,
        help=(
            "gracefully complete after this wall duration; max-messages remains a safety "
            "ceiling and fails the capture if reached first"
        ),
    )
    collect.add_argument("--output-root", type=Path, default=Path("data"))
    collect.set_defaults(handler=_cmd_collect_l2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_messages", 1) < 1:
        parser.error("--max-messages must be positive")
    duration_seconds = getattr(args, "duration_seconds", None)
    if duration_seconds is not None and duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    handler = args.handler
    try:
        return int(handler(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("operation canceled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
