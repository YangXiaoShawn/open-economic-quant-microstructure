"""Streaming, content-addressed Parquet storage and immutable data manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from microstructure.data.evidence_budget import RetainedEvidenceBudget
from microstructure.data.schemas import SCHEMA_VERSION, ensure_schema, get_schema
from microstructure.provenance import read_json, sha256_file, utc_now_iso, write_json

MANIFEST_VERSION = "1.0.0"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_NS_PER_SECOND = 1_000_000_000


class StorageError(RuntimeError):
    """Raised for an unsafe path or inconsistent immutable artifact."""


@dataclass(frozen=True, slots=True)
class PartitionArtifact:
    dataset: str
    venue: str
    symbol: str
    partition_date: str
    rows: int
    write_ordinal: int
    observed_start_ns: int
    observed_end_inclusive_ns: int
    data_path: Path
    manifest_path: Path
    data_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetWriteResult:
    dataset: str
    schema_version: str
    rows: int
    artifacts: tuple[PartitionArtifact, ...]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CaptureDatasetWriteResult:
    """Constant-descriptor result for one bounded-memory live capture."""

    dataset: str
    schema_version: str
    rows: int
    data_path: Path | None
    data_sha256: str | None
    manifest_path: Path
    manifest_sha256: str


def _safe(value: str, label: str) -> str:
    if not value or _SAFE_COMPONENT.fullmatch(value) is None:
        raise StorageError(f"unsafe {label} path component: {value!r}")
    return value


def _stable_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _partition_date(timestamp_ns: int) -> str:
    seconds = timestamp_ns // _NS_PER_SECOND
    return datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()


def _immutable_json(
    directory: Path,
    stem: str,
    payload: Mapping[str, Any],
    *,
    retained_evidence_budget: RetainedEvidenceBudget | None = None,
) -> tuple[Path, str]:
    identity = _stable_sha(payload)
    destination = directory / f"{stem}-{identity[:20]}.json"
    if retained_evidence_budget is not None:
        retained_evidence_budget.assert_contains(destination)
    transaction = (
        retained_evidence_budget.write_transaction()
        if retained_evidence_budget is not None
        else nullcontext()
    )
    with transaction:
        if destination.exists():
            existing = read_json(destination)
            if existing != dict(payload):
                raise StorageError(f"immutable manifest collision at {destination}")
        else:
            encoded = (
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode()
            reservation = (
                retained_evidence_budget.reserve(
                    len(encoded),
                    label=f"raw source manifest {destination.name}",
                )
                if retained_evidence_budget is not None
                else None
            )
            try:
                write_json(destination, payload)
                if destination.stat().st_size != len(encoded):
                    raise StorageError(
                        f"source manifest byte count changed while writing {destination}"
                    )
                if reservation is not None:
                    reservation.commit()
            except BaseException:
                destination.unlink(missing_ok=True)
                if reservation is not None and reservation.active:
                    reservation.release()
                raise
        return destination, sha256_file(destination)


def write_source_manifest(
    raw_path: str | Path,
    *,
    source: str,
    source_uri: str,
    downloaded_at_utc: str,
    requested_start_ns: int | None,
    requested_end_ns: int | None,
    upstream_checksum_sha256: str | None = None,
    response_headers: Mapping[str, str] | None = None,
    retained_evidence_budget: RetainedEvidenceBudget | None = None,
) -> tuple[Path, str]:
    """Write an immutable sidecar for an untouched raw response or archive."""
    path = Path(raw_path)
    if not path.is_file():
        raise StorageError(f"raw artifact does not exist: {path}")
    checksum = sha256_file(path)
    payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "artifact_kind": "raw_source",
        "source": source,
        "source_uri": source_uri,
        "downloaded_at_utc": downloaded_at_utc,
        "requested_range_ns": {"start": requested_start_ns, "end_exclusive": requested_end_ns},
        "checksum": {"algorithm": "sha256", "value": checksum},
        "upstream_checksum_sha256": upstream_checksum_sha256,
        "bytes": path.stat().st_size,
        "path": path.name,
        "response_headers": dict(sorted((response_headers or {}).items())),
    }
    manifest_path, manifest_sha = _immutable_json(
        path.parent,
        f"{path.name}.manifest",
        payload,
        retained_evidence_budget=retained_evidence_budget,
    )
    return manifest_path, manifest_sha


def _write_parquet_part(
    *,
    table: pa.Table,
    root: Path,
    dataset: str,
    schema_name: str,
    venue: str,
    symbol: str,
    date: str,
    source: str,
    source_uri: str,
    downloaded_at_utc: str,
    source_checksum_sha256: str | None,
    requested_start_ns: int | None,
    requested_end_ns: int | None,
    time_column: str,
    compression: str,
    write_ordinal: int,
) -> PartitionArtifact:
    partition = (
        root
        / _safe(dataset, "dataset")
        / f"schema-{_safe(SCHEMA_VERSION, 'schema version')}"
        / f"venue-{_safe(venue, 'venue')}"
        / f"symbol-{_safe(symbol, 'symbol')}"
        / f"date-{_safe(date, 'date')}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(dir=partition, prefix=".part-", suffix=".parquet.tmp")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        table = table.replace_schema_metadata(get_schema(schema_name).metadata)
        pq.write_table(
            table,
            temporary,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        checksum = sha256_file(temporary)
        destination = partition / f"part-{checksum[:20]}.parquet"
        if destination.exists():
            if sha256_file(destination) != checksum:
                raise StorageError(f"content-address collision at {destination}")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    time_bounds = pc.min_max(table.column(time_column)).as_py()
    if time_bounds is None or time_bounds["min"] is None or time_bounds["max"] is None:
        raise StorageError("cannot manifest a Parquet part without a timestamp range")
    manifest_payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "artifact_kind": "normalized_parquet",
        "dataset": dataset,
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSION,
        "venue": venue,
        "symbol": symbol,
        "partition_date": date,
        "write_ordinal": write_ordinal,
        "source": source,
        "source_uri": source_uri,
        "downloaded_at_utc": downloaded_at_utc,
        "requested_range_ns": {"start": requested_start_ns, "end_exclusive": requested_end_ns},
        "observed_range_ns": {
            "start": int(time_bounds["min"]),
            "end_inclusive": int(time_bounds["max"]),
        },
        "source_checksum_sha256": source_checksum_sha256,
        "checksum": {"algorithm": "sha256", "value": checksum},
        "rows": table.num_rows,
        "bytes": destination.stat().st_size,
        "path": str(destination.relative_to(root)),
        "transformations": [
            "normalized field names and types",
            "UTC epoch-nanosecond timestamp conversion",
            "exact integer tick/lot conversion where scale supplied",
        ],
    }
    manifest_path, manifest_sha = _immutable_json(
        partition, f"part-{checksum[:20]}.manifest", manifest_payload
    )
    return PartitionArtifact(
        dataset=dataset,
        venue=venue,
        symbol=symbol,
        partition_date=date,
        rows=table.num_rows,
        write_ordinal=write_ordinal,
        observed_start_ns=int(time_bounds["min"]),
        observed_end_inclusive_ns=int(time_bounds["max"]),
        data_path=destination,
        manifest_path=manifest_path,
        data_sha256=checksum,
        manifest_sha256=manifest_sha,
    )


def _as_table(batch: pa.RecordBatch | pa.Table) -> pa.Table:
    return batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])


def write_partitioned_parquet(
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    root: str | Path,
    dataset: str,
    schema_name: str,
    source: str,
    source_uri: str = "synthetic://local",
    downloaded_at_utc: str | None = None,
    source_checksum_sha256: str | None = None,
    requested_start_ns: int | None = None,
    requested_end_ns: int | None = None,
    time_column: str = "event_ts_ns",
    max_rows_per_file: int = 250_000,
    max_input_batch_rows: int = 250_000,
    compression: str = "zstd",
) -> DatasetWriteResult:
    """Stream batches into immutable Parquet parts partitioned by venue/symbol/day.

    Each input batch is split only within that bounded batch, so this function
    never requires the complete data set in memory.  Existing content-addressed
    parts are reused rather than overwritten.
    """
    if max_rows_per_file < 1:
        raise ValueError("max_rows_per_file must be positive")
    if max_input_batch_rows < 1:
        raise ValueError("max_input_batch_rows must be positive")
    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    download_time = downloaded_at_utc or utc_now_iso()
    artifacts: list[PartitionArtifact] = []

    for raw_batch in batches:
        table = _as_table(raw_batch)
        if table.num_rows > max_input_batch_rows:
            raise StorageError(
                f"input batch has {table.num_rows} rows, above the bounded-memory limit "
                f"{max_input_batch_rows}"
            )
        ensure_schema(table, schema_name)
        if time_column not in table.column_names:
            raise StorageError(f"partition time column is missing: {time_column}")
        groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        venues = table.column("venue").to_pylist()
        symbols = table.column("symbol").to_pylist()
        timestamps = table.column(time_column).to_pylist()
        for row_index, (venue, symbol, timestamp_ns) in enumerate(
            zip(venues, symbols, timestamps, strict=True)
        ):
            groups[(str(venue), str(symbol), _partition_date(int(timestamp_ns)))].append(row_index)

        for (venue, symbol, date), indices in groups.items():
            for offset in range(0, len(indices), max_rows_per_file):
                selected = indices[offset : offset + max_rows_per_file]
                part = table.take(pa.array(selected, type=pa.int64()))
                artifacts.append(
                    _write_parquet_part(
                        table=part,
                        root=destination_root,
                        dataset=dataset,
                        schema_name=schema_name,
                        venue=venue,
                        symbol=symbol,
                        date=date,
                        source=source,
                        source_uri=source_uri,
                        downloaded_at_utc=download_time,
                        source_checksum_sha256=source_checksum_sha256,
                        requested_start_ns=requested_start_ns,
                        requested_end_ns=requested_end_ns,
                        time_column=time_column,
                        compression=compression,
                        write_ordinal=len(artifacts),
                    )
                )

    artifact_entries = [
        {
            "data_path": str(item.data_path.relative_to(destination_root)),
            "manifest_path": str(item.manifest_path.relative_to(destination_root)),
            "data_sha256": item.data_sha256,
            "manifest_sha256": item.manifest_sha256,
            "rows": item.rows,
            "write_ordinal": item.write_ordinal,
            "observed_range_ns": {
                "start": item.observed_start_ns,
                "end_inclusive": item.observed_end_inclusive_ns,
            },
        }
        for item in artifacts
    ]
    stable_identity: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_uri": source_uri,
        "downloaded_at_utc": download_time,
        "requested_range_ns": {"start": requested_start_ns, "end_exclusive": requested_end_ns},
        "artifacts": artifact_entries,
        "rows": sum(item.rows for item in artifacts),
    }
    manifest_directory = destination_root / "_manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha = _immutable_json(
        manifest_directory, f"{_safe(dataset, 'dataset')}.manifest", stable_identity
    )
    return DatasetWriteResult(
        dataset=dataset,
        schema_version=SCHEMA_VERSION,
        rows=sum(item.rows for item in artifacts),
        artifacts=tuple(artifacts),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
    )


def write_capture_parquet(
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    root: str | Path,
    dataset: str,
    schema_name: str,
    venue: str,
    symbol: str,
    capture_id: str,
    source: str,
    source_uri: str,
    downloaded_at_utc: str | None = None,
    source_checksum_sha256: str | None = None,
    requested_start_ns: int | None = None,
    requested_end_ns: int | None = None,
    time_column: str = "event_ts_ns",
    max_input_batch_rows: int = 16_384,
    compression: str = "zstd",
) -> CaptureDatasetWriteResult:
    """Write one live-capture Parquet artifact from a bounded batch iterator.

    The Parquet writer emits one bounded row group per input batch and retains
    exactly one output descriptor, independent of capture length.  Live capture
    data is partitioned by immutable ``capture_id`` rather than UTC day because
    capture-order quality evidence must not be reordered to satisfy a partition.
    """
    if max_input_batch_rows < 1:
        raise ValueError("max_input_batch_rows must be positive")
    safe_dataset = _safe(dataset, "dataset")
    safe_schema = _safe(SCHEMA_VERSION, "schema version")
    safe_venue = _safe(venue, "venue")
    safe_symbol = _safe(symbol, "symbol")
    safe_capture_id = _safe(capture_id, "capture ID")
    schema = get_schema(schema_name)
    if time_column not in schema.names:
        raise StorageError(f"partition time column is missing: {time_column}")

    destination_root = Path(root)
    partition = (
        destination_root
        / safe_dataset
        / f"schema-{safe_schema}"
        / f"venue-{safe_venue}"
        / f"symbol-{safe_symbol}"
        / f"capture-{safe_capture_id}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=partition,
        prefix=".capture-",
        suffix=".parquet.tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    writer: pq.ParquetWriter | None = None
    rows = 0
    observed_start_ns: int | None = None
    observed_end_ns: int | None = None
    destination: Path | None = None
    checksum: str | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for raw_batch in batches:
            table = _as_table(raw_batch)
            if table.num_rows > max_input_batch_rows:
                raise StorageError(
                    f"input batch has {table.num_rows} rows, above the bounded-memory "
                    f"limit {max_input_batch_rows}"
                )
            ensure_schema(table, schema_name)
            if table.num_rows == 0:
                continue
            if set(table.column("venue").to_pylist()) != {venue}:
                raise StorageError("live capture batch contains an unexpected venue")
            if set(table.column("symbol").to_pylist()) != {symbol}:
                raise StorageError("live capture batch contains an unexpected symbol")
            bounds = pc.min_max(table.column(time_column)).as_py()
            if bounds is None or bounds["min"] is None or bounds["max"] is None:
                raise StorageError("cannot write a live capture batch without timestamps")
            batch_start = int(bounds["min"])
            batch_end = int(bounds["max"])
            observed_start_ns = (
                batch_start if observed_start_ns is None else min(observed_start_ns, batch_start)
            )
            observed_end_ns = (
                batch_end if observed_end_ns is None else max(observed_end_ns, batch_end)
            )
            writer.write_table(table, row_group_size=max_input_batch_rows)
            rows += table.num_rows
        writer.close()
        writer = None
        if rows == 0:
            temporary.unlink()
        else:
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            checksum = sha256_file(temporary)
            destination = partition / f"capture-{checksum[:20]}.parquet"
            if destination.exists():
                if sha256_file(destination) != checksum:
                    raise StorageError(f"content-address collision at {destination}")
                temporary.unlink()
            else:
                os.replace(temporary, destination)
    except BaseException:
        if writer is not None:
            with suppress(BaseException):
                writer.close()
        temporary.unlink(missing_ok=True)
        raise

    download_time = downloaded_at_utc or utc_now_iso()
    data_path = destination
    data_sha256 = checksum
    artifact_entry: dict[str, Any] | None = None
    if data_path is not None and data_sha256 is not None:
        artifact_payload: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "artifact_kind": "normalized_live_capture_parquet",
            "dataset": dataset,
            "schema_name": schema_name,
            "schema_version": SCHEMA_VERSION,
            "venue": venue,
            "symbol": symbol,
            "capture_id": capture_id,
            "source": source,
            "source_uri": source_uri,
            "downloaded_at_utc": download_time,
            "requested_range_ns": {
                "start": requested_start_ns,
                "end_exclusive": requested_end_ns,
            },
            "observed_range_ns": {
                "start": observed_start_ns,
                "end_inclusive": observed_end_ns,
            },
            "source_checksum_sha256": source_checksum_sha256,
            "checksum": {"algorithm": "sha256", "value": data_sha256},
            "rows": rows,
            "bytes": data_path.stat().st_size,
            "path": str(data_path.relative_to(destination_root)),
            "transformations": [
                "normalized field names and types",
                "UTC epoch-nanosecond timestamp conversion",
                "exact integer tick/lot conversion where scale supplied",
            ],
        }
        artifact_manifest_path, artifact_manifest_sha = _immutable_json(
            partition,
            f"capture-{data_sha256[:20]}.manifest",
            artifact_payload,
        )
        artifact_entry = {
            "data_path": str(data_path.relative_to(destination_root)),
            "manifest_path": str(artifact_manifest_path.relative_to(destination_root)),
            "data_sha256": data_sha256,
            "manifest_sha256": artifact_manifest_sha,
            "rows": rows,
            "write_ordinal": 0,
            "observed_range_ns": artifact_payload["observed_range_ns"],
        }

    dataset_payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_uri": source_uri,
        "downloaded_at_utc": download_time,
        "requested_range_ns": {
            "start": requested_start_ns,
            "end_exclusive": requested_end_ns,
        },
        "partitioning": {"kind": "capture_id", "value": capture_id},
        "artifacts": [artifact_entry] if artifact_entry is not None else [],
        "rows": rows,
    }
    manifest_directory = destination_root / "_manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_sha = _immutable_json(
        manifest_directory,
        f"{safe_dataset}.capture-{safe_capture_id}.manifest",
        dataset_payload,
    )
    return CaptureDatasetWriteResult(
        dataset=dataset,
        schema_version=SCHEMA_VERSION,
        rows=rows,
        data_path=data_path,
        data_sha256=data_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
    )


def parquet_paths(result: DatasetWriteResult) -> Sequence[Path]:
    """Return concrete parts in manifest order for Polars/DuckDB consumers."""
    return tuple(item.data_path for item in result.artifacts)
