"""Config-driven, research-only data ingestion boundary.

This module composes the lower-level adapters without hiding provenance or data
quality.  A caller-supplied output root keeps every run bundle isolated from the
configured default data directories, which is useful for atomic pipeline staging.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import requests

from microstructure.config import ProjectConfig, datetime_to_ns
from microstructure.data.binance import (
    BinanceHistoricalTradeDownloader,
    BinancePublicClient,
    RetryPolicy,
    SymbolMetadata,
)
from microstructure.data.quality import ValidationReport, validate_table
from microstructure.data.schemas import table_from_records
from microstructure.data.storage import DatasetWriteResult, write_partitioned_parquet
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.provenance import read_json, sha256_file, utc_now_iso, write_json

SUPPORTED_BINANCE_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})

__all__ = [
    "ConfiguredDataAdapter",
    "DataQualityGateError",
    "IngestionError",
    "IngestionResult",
    "NormalizedDatasetResult",
    "RawArtifactResult",
    "SymbolDownloadResult",
    "ValidationSummary",
    "ingest_from_config",
    "ingest_public_trades",
    "ingest_synthetic",
    "validate_configured_input",
    "validate_only",
]


class IngestionError(RuntimeError):
    """Raised when a configuration cannot safely drive the requested ingestion."""


class DataQualityGateError(IngestionError):
    """Raised after preservation when configured error-level quality findings exist."""

    def __init__(self, summary: ValidationSummary, output_root: Path) -> None:
        super().__init__(
            f"data-quality gate failed with {summary.error_count} error findings; "
            f"preserved bundle is under {output_root}"
        )
        self.summary = summary
        self.output_root = output_root


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    reports: tuple[ValidationReport, ...]
    report_paths: tuple[Path, ...]
    rows_checked: int
    error_count: int
    warning_count: int

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def report_for(self, dataset: str) -> ValidationReport:
        for report in self.reports:
            if report.dataset == dataset:
                return report
        raise KeyError(dataset)


@dataclass(frozen=True, slots=True)
class NormalizedDatasetResult:
    schema_name: str
    rows: int
    table: pa.Table
    validation: ValidationReport
    storage: DatasetWriteResult


@dataclass(frozen=True, slots=True)
class RawArtifactResult:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SymbolDownloadResult:
    symbol: str
    rows: int
    complete_range: bool
    metadata: SymbolMetadata


@dataclass(frozen=True, slots=True)
class IngestionResult:
    mode: str
    evidence_tier: str
    output_root: Path
    datasets: tuple[NormalizedDatasetResult, ...]
    validation: ValidationSummary
    raw_artifacts: tuple[RawArtifactResult, ...]
    symbols: tuple[SymbolDownloadResult, ...]
    ingestion_manifest_path: Path
    ingestion_manifest_sha256: str

    @property
    def rows(self) -> int:
        return sum(dataset.rows for dataset in self.datasets)

    @property
    def manifest_sha256s(self) -> tuple[str, ...]:
        return tuple(dataset.storage.manifest_sha256 for dataset in self.datasets)

    def dataset(self, schema_name: str) -> NormalizedDatasetResult:
        for dataset in self.datasets:
            if dataset.schema_name == schema_name:
                return dataset
        raise KeyError(schema_name)


class ConfiguredDataAdapter(Protocol):
    """Extension boundary for a future venue or institutional normalized adapter."""

    mode: str

    def ingest(
        self,
        config: ProjectConfig,
        output_root: str | Path,
    ) -> IngestionResult: ...


def validate_only(
    tables: Mapping[str, pa.Table],
    config: ProjectConfig,
    *,
    output_root: str | Path | None = None,
) -> ValidationSummary:
    """Return validation-only summaries without repairing or replacing any table."""
    reports: list[ValidationReport] = []
    paths: list[Path] = []
    quality_root = Path(output_root) / "quality" if output_root is not None else None
    for schema_name in sorted(tables):
        report = validate_table(
            tables[schema_name],
            schema_name,
            max_spread_bps=config.quality.max_spread_bps,
            max_silence_ns=config.quality.max_silence_ms * 1_000_000,
        )
        reports.append(report)
        if quality_root is not None:
            path = quality_root / f"{schema_name}.validation.json"
            report.write_json(path)
            paths.append(path)
    return ValidationSummary(
        reports=tuple(reports),
        report_paths=tuple(paths),
        rows_checked=sum(report.rows_checked for report in reports),
        error_count=sum(report.error_count for report in reports),
        warning_count=sum(report.warning_count for report in reports),
    )


def _write_dataset(
    *,
    table: pa.Table,
    schema_name: str,
    config: ProjectConfig,
    output_root: Path,
    requested_start_ns: int,
    requested_end_ns: int | None,
    source_uri: str,
) -> DatasetWriteResult:
    return write_partitioned_parquet(
        table.to_batches(max_chunksize=100_000),
        root=output_root / "normalized",
        dataset=schema_name,
        schema_name=schema_name,
        source=config.data.source,
        source_uri=source_uri,
        requested_start_ns=requested_start_ns,
        requested_end_ns=requested_end_ns,
    )


def _dataset_results(
    tables: Mapping[str, pa.Table],
    stores: Mapping[str, DatasetWriteResult],
    summary: ValidationSummary,
) -> tuple[NormalizedDatasetResult, ...]:
    return tuple(
        NormalizedDatasetResult(
            schema_name=name,
            rows=tables[name].num_rows,
            table=tables[name],
            validation=summary.report_for(name),
            storage=stores[name],
        )
        for name in sorted(tables)
    )


def _quality_gate(config: ProjectConfig, summary: ValidationSummary, output_root: Path) -> None:
    if config.quality.fail_on_error and not summary.passed:
        raise DataQualityGateError(summary, output_root)


def _persist_ingestion_manifest(
    *,
    config: ProjectConfig,
    destination: Path,
    mode: str,
    evidence_tier: str,
    datasets: tuple[NormalizedDatasetResult, ...],
    raw_artifacts: tuple[RawArtifactResult, ...],
    symbol_coverage: list[dict[str, object]],
    row_cap_per_symbol: int | None,
) -> tuple[Path, str]:
    payload: dict[str, object] = {
        "manifest_version": "1.0.0",
        "artifact_kind": "ingestion_run",
        "created_at_utc": utc_now_iso(),
        "mode": mode,
        "evidence_tier": evidence_tier,
        "requested_evidence_tier": config.run.evidence_tier,
        "source": config.data.source,
        "schema_version": config.data.schema_version,
        "requested_range_ns": {
            "start": datetime_to_ns(config.data.start),
            "end_exclusive": (
                datetime_to_ns(config.data.end) if config.data.end is not None else None
            ),
        },
        "row_cap_per_symbol": row_cap_per_symbol,
        "all_requested_ranges_complete": all(
            bool(item["complete_range"]) for item in symbol_coverage
        ),
        "symbols": symbol_coverage,
        "normalized_datasets": [
            {
                "schema_name": item.schema_name,
                "rows": item.rows,
                "manifest_path": str(item.storage.manifest_path.relative_to(destination)),
                "manifest_sha256": item.storage.manifest_sha256,
            }
            for item in datasets
        ],
        "raw_artifacts": [
            {
                "path": str(item.path.relative_to(destination)),
                "sha256": item.sha256,
                "manifest_path": str(item.manifest_path.relative_to(destination)),
                "manifest_sha256": item.manifest_sha256,
            }
            for item in raw_artifacts
        ],
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_dir = destination / "_ingestion_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"ingestion.manifest-{identity[:20]}.json"
    if path.exists():
        if read_json(path) != payload:
            raise IngestionError(f"immutable ingestion-manifest collision at {path}")
    else:
        write_json(path, payload)
    return path, sha256_file(path)


def ingest_synthetic(config: ProjectConfig, output_root: str | Path) -> IngestionResult:
    """Generate, validate, and persist the deterministic offline smoke data set."""
    if config.data.mode != "synthetic":
        raise IngestionError("ingest_synthetic requires data.mode='synthetic'")
    events = config.data.events_per_symbol
    if events is None or events < 1:
        raise IngestionError("synthetic ingestion requires a positive events_per_symbol")
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    start_ns = datetime_to_ns(config.data.start)
    generated = generate_synthetic_market(
        symbols=config.data.symbols,
        events_per_symbol=events,
        start_ts_ns=start_ns,
        seed=config.run.seed,
    )
    tables = {
        "trades": generated.trades,
        "book_observations": generated.book_observations,
    }
    summary = validate_only(tables, config, output_root=destination)
    observed_end_ns = (
        max(max(table.column("event_ts_ns").to_pylist()) for table in tables.values()) + 1
    )
    stores = {
        name: _write_dataset(
            table=table,
            schema_name=name,
            config=config,
            output_root=destination,
            requested_start_ns=start_ns,
            requested_end_ns=observed_end_ns,
            source_uri=f"synthetic://seed/{config.run.seed}",
        )
        for name, table in tables.items()
    }
    dataset_results = _dataset_results(tables, stores, summary)
    manifest_path, manifest_sha = _persist_ingestion_manifest(
        config=config,
        destination=destination,
        mode="synthetic",
        evidence_tier="SYNTHETIC_SMOKE",
        datasets=dataset_results,
        raw_artifacts=(),
        symbol_coverage=[
            {"symbol": symbol, "rows": events, "complete_range": True}
            for symbol in config.data.symbols
        ],
        row_cap_per_symbol=events,
    )
    result = IngestionResult(
        mode="synthetic",
        evidence_tier="SYNTHETIC_SMOKE",
        output_root=destination,
        datasets=dataset_results,
        validation=summary,
        raw_artifacts=(),
        symbols=(),
        ingestion_manifest_path=manifest_path,
        ingestion_manifest_sha256=manifest_sha,
    )
    _quality_gate(config, summary, destination)
    return result


def _raw_artifacts(raw_root: Path) -> tuple[RawArtifactResult, ...]:
    results: list[RawArtifactResult] = []
    if not raw_root.exists():
        return ()
    for raw_path in sorted(raw_root.rglob("*.json")):
        if ".manifest-" in raw_path.name:
            continue
        manifests = sorted(raw_path.parent.glob(f"{raw_path.name}.manifest-*.json"))
        if not manifests:
            raise IngestionError(f"raw artifact has no immutable manifest: {raw_path}")
        for manifest_path in manifests:
            results.append(
                RawArtifactResult(
                    path=raw_path,
                    manifest_path=manifest_path,
                    sha256=sha256_file(raw_path),
                    manifest_sha256=sha256_file(manifest_path),
                )
            )
    return tuple(results)


def ingest_public_trades(
    config: ProjectConfig,
    output_root: str | Path,
    *,
    client: BinancePublicClient | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] | None = None,
    random_value: Callable[[], float] | None = None,
) -> IngestionResult:
    """Download bounded public BTC/ETH aggregate trades with exact symbol scales."""
    if config.data.mode != "binance_rest":
        raise IngestionError("Binance ingestion requires data.mode='binance_rest'")
    unsupported = set(config.data.symbols) - SUPPORTED_BINANCE_SYMBOLS
    if unsupported:
        raise IngestionError(f"unsupported public-sample symbols: {sorted(unsupported)}")
    if config.data.end is None:
        raise IngestionError("bounded Binance ingestion requires data.end")
    row_cap = config.data.max_events_per_symbol
    if row_cap is None or row_cap < 1:
        raise IngestionError("bounded Binance ingestion requires max_events_per_symbol")
    if client is not None and session is not None:
        raise IngestionError("supply either client or session, not both")

    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw_root = destination / "raw"
    if client is None:
        client = BinancePublicClient(
            base_url=config.data.base_url,
            timeout_seconds=config.data.timeout_seconds,
            retry_policy=RetryPolicy(max_retries=config.data.max_retries),
            session=session,
            sleep=sleep if sleep is not None else time.sleep,
            random_value=random_value if random_value is not None else random.random,
        )

    start_ns = datetime_to_ns(config.data.start)
    end_ns = datetime_to_ns(config.data.end)
    tables: list[pa.Table] = []
    symbol_results: list[SymbolDownloadResult] = []
    used_raw_artifacts: list[RawArtifactResult] = []
    for symbol in config.data.symbols:
        metadata = client.fetch_exchange_info(symbol=symbol, raw_root=raw_root)
        used_raw_artifacts.append(
            RawArtifactResult(
                path=metadata.source_path,
                manifest_path=metadata.source_manifest_path,
                sha256=metadata.source_artifact_id,
                manifest_sha256=sha256_file(metadata.source_manifest_path),
            )
        )
        downloader = BinanceHistoricalTradeDownloader(
            client=client,
            raw_root=raw_root,
            request_limit=config.data.request_limit,
            tick_size=metadata.tick_size,
            lot_size=metadata.lot_size,
        )
        downloaded = downloader.download(
            symbol=symbol,
            start_ts_ns=start_ns,
            end_ts_ns=end_ns,
            max_events=row_cap,
        )
        if downloaded.trades.num_rows > row_cap:
            raise IngestionError(f"Binance adapter exceeded row cap for {symbol}")
        if downloaded.trades.num_rows == 0:
            raise IngestionError(
                f"Binance returned no aggregate trades for {symbol}; raw responses were preserved"
            )
        used_raw_artifacts.extend(
            RawArtifactResult(
                path=page.path,
                manifest_path=page.manifest_path,
                sha256=page.sha256,
                manifest_sha256=sha256_file(page.manifest_path),
            )
            for page in downloaded.raw_pages
        )
        tables.append(downloaded.trades)
        symbol_results.append(
            SymbolDownloadResult(
                symbol=symbol,
                rows=downloaded.trades.num_rows,
                complete_range=downloaded.complete_range,
                metadata=metadata,
            )
        )

    trades = pa.concat_tables(tables) if tables else table_from_records("trades", [])
    normalized = {"trades": trades}
    summary = validate_only(normalized, config, output_root=destination)
    stores = {
        "trades": _write_dataset(
            table=trades,
            schema_name="trades",
            config=config,
            output_root=destination,
            requested_start_ns=start_ns,
            requested_end_ns=end_ns,
            source_uri=f"{config.data.base_url}/api/v3/aggTrades",
        )
    }
    dataset_results = _dataset_results(normalized, stores, summary)
    unique_raw_artifacts: dict[tuple[Path, Path], RawArtifactResult] = {}
    for item in used_raw_artifacts:
        unique_raw_artifacts[(item.path, item.manifest_path)] = item
    raw_artifacts = tuple(unique_raw_artifacts.values())
    all_requested_ranges_complete = all(item.complete_range for item in symbol_results)
    effective_evidence_tier = (
        config.run.evidence_tier if all_requested_ranges_complete else "PUBLIC_SAMPLE_PARTIAL"
    )
    manifest_path, manifest_sha = _persist_ingestion_manifest(
        config=config,
        destination=destination,
        mode="binance_rest",
        evidence_tier=effective_evidence_tier,
        datasets=dataset_results,
        raw_artifacts=raw_artifacts,
        symbol_coverage=[
            {
                "symbol": item.symbol,
                "rows": item.rows,
                "complete_range": item.complete_range,
                "tick_size": str(item.metadata.tick_size),
                "lot_size": str(item.metadata.lot_size),
            }
            for item in symbol_results
        ],
        row_cap_per_symbol=row_cap,
    )
    result = IngestionResult(
        mode="binance_rest",
        evidence_tier=effective_evidence_tier,
        output_root=destination,
        datasets=dataset_results,
        validation=summary,
        raw_artifacts=raw_artifacts,
        symbols=tuple(symbol_results),
        ingestion_manifest_path=manifest_path,
        ingestion_manifest_sha256=manifest_sha,
    )
    _quality_gate(config, summary, destination)
    return result


def ingest_from_config(
    config: ProjectConfig,
    output_root: str | Path,
    *,
    client: BinancePublicClient | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] | None = None,
    random_value: Callable[[], float] | None = None,
    adapter: ConfiguredDataAdapter | None = None,
) -> IngestionResult:
    """Dispatch the configured offline/public ingestion mode."""
    if adapter is not None:
        if adapter.mode != config.data.mode:
            raise IngestionError(
                f"adapter mode {adapter.mode!r} does not match config mode {config.data.mode!r}"
            )
        return adapter.ingest(config, output_root)
    if config.data.mode == "synthetic":
        return ingest_synthetic(config, output_root)
    return ingest_public_trades(
        config,
        output_root,
        client=client,
        session=session,
        sleep=sleep,
        random_value=random_value,
    )


def validate_configured_input(
    config: ProjectConfig, *, tables: Mapping[str, pa.Table] | None = None
) -> ValidationSummary:
    """Validate supplied normalized tables or discover them under configured storage."""
    if tables is not None:
        return validate_only(tables, config)

    discovered: dict[str, pa.Table] = {}
    configured_row_limit = (
        config.data.events_per_symbol
        if config.data.mode == "synthetic"
        else config.data.max_events_per_symbol
    )
    if configured_row_limit is None:
        raise IngestionError("configured validation requires a finite per-symbol row limit")
    maximum_rows = configured_row_limit * len(config.data.symbols)
    for schema_name in ("trades", "book_observations", "depth_deltas"):
        dataset_root = config.data.partition_root / schema_name
        paths = sorted(dataset_root.rglob("*.parquet")) if dataset_root.exists() else []
        if paths:
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
            if rows > maximum_rows:
                raise IngestionError(
                    f"{schema_name} contains {rows} rows, above configured validation bound "
                    f"{maximum_rows}; validate bounded partitions separately"
                )
            discovered[schema_name] = pa.concat_tables([pq.read_table(path) for path in paths])
    if not discovered:
        raise IngestionError(
            f"no normalized Parquet inputs found under {config.data.partition_root}"
        )
    return validate_only(discovered, config)
