"""Streaming acquisition boundary for the frozen M8 multi-date trade study.

The implementation deliberately handles one symbol/day archive at a time.  Raw
bytes are checksum authenticated before their CSV member is opened; normalized
RecordBatches flow once through incremental quality validation and partitioned
Parquet storage.  No complete daily table is retained in memory.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure.data.binance import BinancePublicClient, SymbolMetadata
from microstructure.data.binance_archive import (
    AcquiredDailyArchive,
    ArchiveDownloadLimits,
    BinanceArchiveClient,
    DailyArchiveRequest,
)
from microstructure.data.quality import IncrementalQualityValidator, ValidationReport
from microstructure.data.storage import DatasetWriteResult, write_partitioned_parquet
from microstructure.m8_config import M8Period, M8StudyConfig
from microstructure.m8_manifest import (
    M8ArchiveEntry,
    M8InputManifest,
    M8NormalizedPart,
    M8SymbolMetadata,
    write_m8_input_manifest,
)
from microstructure.provenance import read_json, sha256_file

_NS_PER_DAY = 86_400 * 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_DEFAULT_BATCH_ROWS = 65_536


class M8IngestionError(RuntimeError):
    """Raised when acquisition cannot produce complete frozen M8 evidence."""


class M8DataQualityError(M8IngestionError):
    """Raised after preserving a symbol/day whose quality gate did not pass."""

    def __init__(
        self,
        *,
        symbol: str,
        day: date,
        report: ValidationReport,
        output_root: Path,
    ) -> None:
        super().__init__(
            f"M8 quality gate failed for {symbol}/{day.isoformat()} with "
            f"{report.error_count} errors and {report.warning_count} warnings; "
            f"evidence was preserved under {output_root}"
        )
        self.symbol = symbol
        self.day = day
        self.report = report
        self.output_root = output_root


class ArchiveAcquirer(Protocol):
    """Narrow injectable archive boundary used by offline tests."""

    def acquire(
        self,
        request: DailyArchiveRequest,
        *,
        raw_root: str | Path,
        limits: ArchiveDownloadLimits,
    ) -> AcquiredDailyArchive: ...


class MetadataProvider(Protocol):
    """Narrow injectable public exchange-metadata boundary."""

    def fetch_exchange_info(self, *, symbol: str, raw_root: str | Path) -> SymbolMetadata: ...


@dataclass(frozen=True, slots=True)
class M8IngestionResult:
    """Verified acquisition result; normalized rows deliberately stay on disk."""

    output_root: Path
    input_manifest: M8InputManifest
    entries: tuple[M8ArchiveEntry, ...]
    symbol_metadata: tuple[M8SymbolMetadata, ...]

    @property
    def input_manifest_path(self) -> Path:
        return self.input_manifest.path

    @property
    def input_manifest_sha256(self) -> str:
        return self.input_manifest.sha256

    @property
    def rows(self) -> int:
        return sum(entry.rows for entry in self.entries)


def _day_bounds_ns(day: date) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    start_ns = (start - _EPOCH).days * _NS_PER_DAY
    return start_ns, start_ns + _NS_PER_DAY


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M8IngestionError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _metadata_claim(
    metadata: SymbolMetadata,
    *,
    expected_symbol: str,
    output_root: Path,
) -> M8SymbolMetadata:
    if metadata.venue != "binance_spot" or metadata.symbol != expected_symbol:
        raise M8IngestionError(f"exchange metadata identity disagrees for {expected_symbol}")
    if metadata.status != "TRADING":
        raise M8IngestionError(f"exchange metadata does not mark {expected_symbol} TRADING")
    raw_path = metadata.source_path.resolve()
    sidecar_path = metadata.source_manifest_path.resolve()
    if not raw_path.is_relative_to(output_root) or not sidecar_path.is_relative_to(output_root):
        raise M8IngestionError("exchange metadata evidence must be contained in the M8 root")
    if not raw_path.is_file() or not sidecar_path.is_file():
        raise M8IngestionError(f"exchange metadata evidence is missing for {expected_symbol}")
    raw_sha = sha256_file(raw_path)
    if raw_sha != metadata.source_artifact_id:
        raise M8IngestionError(f"exchange metadata raw checksum disagrees for {expected_symbol}")
    sidecar = _object(read_json(sidecar_path), f"{expected_symbol} metadata sidecar")
    source_uri = sidecar.get("source_uri")
    if type(source_uri) is not str:
        raise M8IngestionError(f"exchange metadata source URI is missing for {expected_symbol}")
    return M8SymbolMetadata(
        symbol=expected_symbol,
        status=metadata.status,
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
        observed_ts_ns=metadata.observed_ts_ns,
        raw_path=raw_path,
        raw_sha256=raw_sha,
        raw_bytes=raw_path.stat().st_size,
        source_uri=source_uri,
        source_manifest_path=sidecar_path,
        source_manifest_sha256=sha256_file(sidecar_path),
        source_manifest_bytes=sidecar_path.stat().st_size,
    )


def _load_symbol_metadata(
    config: M8StudyConfig,
    *,
    output_root: Path,
    metadata_provider: MetadataProvider | None,
    supplied_metadata: Mapping[str, SymbolMetadata] | None,
) -> tuple[tuple[M8SymbolMetadata, ...], dict[str, SymbolMetadata]]:
    if metadata_provider is not None and supplied_metadata is not None:
        raise M8IngestionError("supply either metadata_provider or supplied_metadata, not both")
    provider = metadata_provider
    if provider is None and supplied_metadata is None:
        provider = BinancePublicClient()
    raw: dict[str, SymbolMetadata] = {}
    claims: list[M8SymbolMetadata] = []
    for symbol in config.study.symbols:
        if supplied_metadata is not None:
            try:
                item = supplied_metadata[symbol]
            except KeyError as exc:
                raise M8IngestionError(f"missing supplied metadata for {symbol}") from exc
        else:
            if provider is None:
                raise AssertionError("metadata provider resolution failed")
            item = provider.fetch_exchange_info(symbol=symbol, raw_root=output_root / "raw")
        raw[symbol] = item
        claims.append(_metadata_claim(item, expected_symbol=symbol, output_root=output_root))
    if supplied_metadata is not None and set(supplied_metadata) != set(config.study.symbols):
        raise M8IngestionError("supplied metadata symbols differ from the frozen M8 universe")
    return tuple(claims), raw


def _download_time(acquired: AcquiredDailyArchive) -> str:
    sidecar = _object(
        read_json(acquired.archive_artifact.manifest_path),
        "archive raw source sidecar",
    )
    value = sidecar.get("downloaded_at_utc")
    if type(value) is not str or not value:
        raise M8IngestionError("archive source sidecar lacks downloaded_at_utc")
    return value


def _normalized_parts(storage: DatasetWriteResult) -> tuple[M8NormalizedPart, ...]:
    return tuple(
        M8NormalizedPart(
            data_path=item.data_path.resolve(),
            data_sha256=item.data_sha256,
            data_bytes=item.data_path.stat().st_size,
            sidecar_path=item.manifest_path.resolve(),
            sidecar_sha256=item.manifest_sha256,
            sidecar_bytes=item.manifest_path.stat().st_size,
            rows=item.rows,
            write_ordinal=item.write_ordinal,
            observed_start_ns=item.observed_start_ns,
            observed_end_inclusive_ns=item.observed_end_inclusive_ns,
        )
        for item in storage.artifacts
    )


def _archive_entry(
    config: M8StudyConfig,
    *,
    period: M8Period,
    symbol: str,
    metadata: SymbolMetadata,
    archive_client: ArchiveAcquirer,
    output_root: Path,
    limits: ArchiveDownloadLimits,
    batch_rows: int,
    remaining_compressed_bytes: int,
) -> M8ArchiveEntry:
    request = DailyArchiveRequest(
        symbol=symbol,
        date=period.date,
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
    )
    acquired = archive_client.acquire(request, raw_root=output_root / "raw", limits=limits)
    if acquired.archive_artifact.bytes > remaining_compressed_bytes:
        raise M8IngestionError("M8 archives exceed the frozen total-download ceiling")
    source_uri = acquired.archive_artifact.source_uri
    day_start_ns, day_end_ns = _day_bounds_ns(period.date)
    quality_root = output_root / "quality" / symbol / period.date.isoformat()
    quality_root.mkdir(parents=True, exist_ok=True)
    token = f"{time.time_ns():x}"
    findings_path = quality_root / f"findings-{token}.jsonl"
    stream = acquired.iter_normalized_batches(batch_rows=batch_rows)
    with IncrementalQualityValidator(
        "trades",
        findings_jsonl_path=findings_path,
    ) as validator:

        def validated_batches() -> Iterator[pa.RecordBatch]:
            for batch in stream:
                if batch.num_rows < 1 or batch.num_rows > batch_rows:
                    raise M8IngestionError("archive stream violated its configured row bound")
                validator.update(batch)
                yield batch

        storage = write_partitioned_parquet(
            validated_batches(),
            root=output_root / "normalized" / symbol / period.date.isoformat(),
            dataset="trades",
            schema_name="trades",
            source=config.study.source,
            source_uri=source_uri,
            downloaded_at_utc=_download_time(acquired),
            source_checksum_sha256=acquired.archive_artifact.sha256,
            requested_start_ns=day_start_ns,
            requested_end_ns=day_end_ns,
            max_input_batch_rows=batch_rows,
            max_rows_per_file=250_000,
        )
        report = validator.finish()
    summary = stream.summary
    report_path = quality_root / f"validation-{token}.json"
    report.write_json(report_path)

    if summary.symbol != symbol or summary.date != period.date.isoformat():
        raise M8IngestionError(f"archive summary identity disagrees for {symbol}/{period.date}")
    if summary.source_archive_sha256 != acquired.archive_artifact.sha256:
        raise M8IngestionError(f"archive summary checksum disagrees for {symbol}/{period.date}")
    if summary.expanded_bytes != acquired.declared_uncompressed_bytes:
        raise M8IngestionError(
            f"archive expanded-byte summary disagrees for {symbol}/{period.date}"
        )
    if storage.rows != summary.rows or report.rows_checked != summary.rows:
        raise M8IngestionError(
            f"archive, storage, and quality row counts disagree for {symbol}/{period.date}"
        )
    if summary.last_trade_id - summary.first_trade_id + 1 != summary.rows:
        raise M8IngestionError(f"archive trade IDs are not contiguous for {symbol}/{period.date}")
    if report.error_count or (report.warning_count and not config.quality.allow_quality_warnings):
        raise M8DataQualityError(
            symbol=symbol,
            day=period.date,
            report=report,
            output_root=output_root,
        )

    parts = _normalized_parts(storage)
    if not parts or sum(part.rows for part in parts) != summary.rows:
        raise M8IngestionError(f"normalized part accounting failed for {symbol}/{period.date}")
    return M8ArchiveEntry(
        symbol=symbol,
        date=period.date,
        role=period.role,
        complete=True,
        rows=summary.rows,
        first_trade_id=summary.first_trade_id,
        last_trade_id=summary.last_trade_id,
        observed_start_ns=summary.first_event_ts_ns,
        observed_end_inclusive_ns=summary.last_event_ts_ns,
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
        raw_zip_path=acquired.archive_artifact.path.resolve(),
        raw_zip_sha256=acquired.archive_artifact.sha256,
        raw_zip_bytes=acquired.archive_artifact.bytes,
        raw_uncompressed_bytes=summary.expanded_bytes,
        raw_source_uri=source_uri,
        raw_source_manifest_path=acquired.archive_artifact.manifest_path.resolve(),
        raw_source_manifest_sha256=acquired.archive_artifact.manifest_sha256,
        raw_source_manifest_bytes=acquired.archive_artifact.manifest_path.stat().st_size,
        normalized_dataset_manifest_path=storage.manifest_path.resolve(),
        normalized_dataset_manifest_sha256=storage.manifest_sha256,
        normalized_dataset_manifest_bytes=storage.manifest_path.stat().st_size,
        normalized_parts=parts,
        quality_report_path=report_path.resolve(),
        quality_report_sha256=sha256_file(report_path),
        quality_report_bytes=report_path.stat().st_size,
        quality_findings_path=findings_path.resolve(),
        quality_findings_sha256=sha256_file(findings_path),
        quality_findings_bytes=findings_path.stat().st_size,
        quality_errors=report.error_count,
        quality_warnings=report.warning_count,
    )


def ingest_m8_archives(
    config: M8StudyConfig,
    output_root: str | Path,
    *,
    archive_client: ArchiveAcquirer | None = None,
    metadata_provider: MetadataProvider | None = None,
    supplied_metadata: Mapping[str, SymbolMetadata] | None = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> M8IngestionResult:
    """Acquire, normalize, validate, and manifest all eight frozen archives."""

    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    claims, metadata = _load_symbol_metadata(
        config,
        output_root=destination,
        metadata_provider=metadata_provider,
        supplied_metadata=supplied_metadata,
    )
    client = archive_client if archive_client is not None else BinanceArchiveClient()
    limits = ArchiveDownloadLimits(
        max_compressed_bytes=config.study.max_archive_compressed_bytes,
        max_uncompressed_bytes=config.study.max_archive_uncompressed_bytes,
    )
    entries: list[M8ArchiveEntry] = []
    compressed_bytes = 0
    for period in config.periods:
        for symbol in config.study.symbols:
            entry = _archive_entry(
                config,
                period=period,
                symbol=symbol,
                metadata=metadata[symbol],
                archive_client=client,
                output_root=destination,
                limits=limits,
                batch_rows=batch_rows,
                remaining_compressed_bytes=(
                    config.study.max_total_download_bytes - compressed_bytes
                ),
            )
            entries.append(entry)
            compressed_bytes += entry.raw_zip_bytes
            if compressed_bytes > config.study.max_total_download_bytes:
                raise M8IngestionError("M8 archives exceed the frozen total-download ceiling")
    manifest = write_m8_input_manifest(config, destination, entries, claims)
    return M8IngestionResult(
        output_root=destination,
        input_manifest=manifest,
        entries=manifest.entries,
        symbol_metadata=manifest.symbol_metadata,
    )


__all__ = [
    "ArchiveAcquirer",
    "M8DataQualityError",
    "M8IngestionError",
    "M8IngestionResult",
    "MetadataProvider",
    "ingest_m8_archives",
]
