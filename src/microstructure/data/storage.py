"""Streaming, content-addressed Parquet storage and immutable data manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

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


def _immutable_json(directory: Path, stem: str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    identity = _stable_sha(payload)
    destination = directory / f"{stem}-{identity[:20]}.json"
    if destination.exists():
        existing = read_json(destination)
        if existing != dict(payload):
            raise StorageError(f"immutable manifest collision at {destination}")
    else:
        write_json(destination, payload)
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
    manifest_path, manifest_sha = _immutable_json(path.parent, f"{path.name}.manifest", payload)
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

    times = table.column(time_column).to_pylist()
    manifest_payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "artifact_kind": "normalized_parquet",
        "dataset": dataset,
        "schema_name": schema_name,
        "schema_version": SCHEMA_VERSION,
        "venue": venue,
        "symbol": symbol,
        "partition_date": date,
        "source": source,
        "source_uri": source_uri,
        "downloaded_at_utc": downloaded_at_utc,
        "requested_range_ns": {"start": requested_start_ns, "end_exclusive": requested_end_ns},
        "observed_range_ns": {"start": min(times), "end_inclusive": max(times)},
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
    compression: str = "zstd",
) -> DatasetWriteResult:
    """Stream batches into immutable Parquet parts partitioned by venue/symbol/day.

    Each input batch is split only within that bounded batch, so this function
    never requires the complete data set in memory.  Existing content-addressed
    parts are reused rather than overwritten.
    """
    if max_rows_per_file < 1:
        raise ValueError("max_rows_per_file must be positive")
    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    download_time = downloaded_at_utc or utc_now_iso()
    artifacts: list[PartitionArtifact] = []

    for raw_batch in batches:
        table = _as_table(raw_batch)
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
                    )
                )

    artifact_entries = [
        {
            "data_path": str(item.data_path.relative_to(destination_root)),
            "manifest_path": str(item.manifest_path.relative_to(destination_root)),
            "data_sha256": item.data_sha256,
            "manifest_sha256": item.manifest_sha256,
            "rows": item.rows,
        }
        for item in sorted(artifacts, key=lambda item: str(item.data_path))
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


def parquet_paths(result: DatasetWriteResult) -> Sequence[Path]:
    """Return concrete parts in manifest order for Polars/DuckDB consumers."""
    return tuple(item.data_path for item in result.artifacts)
