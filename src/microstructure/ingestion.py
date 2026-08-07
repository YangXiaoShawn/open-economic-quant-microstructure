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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import requests

from microstructure.config import ProjectConfig, datetime_to_ns
from microstructure.data.binance import (
    BinanceHistoricalTradeDownloader,
    BinancePublicClient,
    BinanceTradeStreamSummary,
    RawPage,
    RetryPolicy,
    SymbolMetadata,
)
from microstructure.data.quality import (
    IncrementalQualityValidator,
    ValidationReport,
    validate_batches,
    validate_table,
)
from microstructure.data.storage import DatasetWriteResult, write_partitioned_parquet
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.provenance import read_json, sha256_file, utc_now_iso, write_json

SUPPORTED_BINANCE_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})

__all__ = [
    "ConfiguredDataAdapter",
    "DataAdapterRegistry",
    "DataQualityGateError",
    "IngestionError",
    "IngestionResult",
    "NormalizedDatasetResult",
    "RawArtifactResult",
    "SymbolDownloadResult",
    "ValidationSummary",
    "builtin_data_adapter_registry",
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
    table: pa.Table | None
    validation: ValidationReport
    storage: DatasetWriteResult

    def materialize(self, *, max_rows: int) -> pa.Table:
        """Explicitly load a bounded result, verifying its row claim first.

        Synthetic smoke results remain resident and are returned directly.  A
        streamed public result deliberately carries no table; callers that truly
        need one must opt into a finite bound before any Parquet part is read.
        """
        if max_rows < 0:
            raise ValueError("max_rows must not be negative")
        if self.rows > max_rows:
            raise IngestionError(
                f"dataset {self.schema_name!r} has {self.rows} rows, above "
                f"materialization bound {max_rows}"
            )
        if self.table is not None:
            if self.table.num_rows != self.rows:
                raise IngestionError(
                    f"resident {self.schema_name!r} table disagrees with its row claim"
                )
            return self.table

        artifacts = self.storage.artifacts
        if not artifacts:
            raise IngestionError(f"streamed {self.schema_name!r} dataset has no Parquet artifacts")
        observed_rows = sum(pq.ParquetFile(item.data_path).metadata.num_rows for item in artifacts)
        if observed_rows != self.rows:
            raise IngestionError(
                f"stored {self.schema_name!r} rows {observed_rows} disagree with "
                f"manifested rows {self.rows}"
            )
        # The row-count check above proves this eager compatibility path cannot
        # exceed the caller's explicit bound.
        return pa.concat_tables([pq.read_table(item.data_path) for item in artifacts])


@dataclass(frozen=True, slots=True)
class RawArtifactResult:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SymbolDownloadResult:
    symbol: str
    metadata: SymbolMetadata
    stream_summary: BinanceTradeStreamSummary

    @property
    def rows(self) -> int:
        return self.stream_summary.rows_yielded

    @property
    def complete_range(self) -> bool:
        return self.stream_summary.complete_range

    @property
    def raw_page_count(self) -> int:
        return self.stream_summary.raw_page_count

    @property
    def stop_reason(self) -> str:
        return str(self.stream_summary.stop_reason)

    @property
    def last_raw_page_sha256(self) -> str | None:
        page = self.stream_summary.last_raw_page
        return page.sha256 if page is not None else None


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
    """Adapter contract for a configured, normalized ingestion implementation."""

    @property
    def mode(self) -> str: ...

    def ingest(
        self,
        config: ProjectConfig,
        output_root: str | Path,
    ) -> IngestionResult: ...


class DataAdapterRegistry:
    """Explicit, fail-closed mapping from configuration modes to adapters.

    Registries are deliberately instance-scoped.  Tests and embedding
    applications can inject a registry without mutating process-global state,
    and duplicate registrations require an explicit replacement request.
    """

    __slots__ = ("_adapters",)

    def __init__(self, adapters: Iterable[ConfiguredDataAdapter] = ()) -> None:
        self._adapters: dict[str, ConfiguredDataAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    @property
    def modes(self) -> tuple[str, ...]:
        """Return registered modes in deterministic order."""
        return tuple(sorted(self._adapters))

    def register(
        self,
        adapter: ConfiguredDataAdapter,
        *,
        replace_existing: bool = False,
    ) -> None:
        """Register ``adapter`` and reject accidental mode shadowing."""
        mode = adapter.mode
        if not isinstance(mode, str) or not mode:
            raise IngestionError("adapter mode must be a nonempty string")
        if mode in self._adapters and not replace_existing:
            raise IngestionError(f"data adapter mode {mode!r} is already registered")
        self._adapters[mode] = adapter

    def resolve(self, mode: str) -> ConfiguredDataAdapter:
        """Resolve exactly one mode or fail without selecting a fallback."""
        try:
            return self._adapters[mode]
        except KeyError as exc:
            available = ", ".join(self.modes) if self._adapters else "none"
            raise IngestionError(
                f"no data adapter registered for mode {mode!r}; registered modes: {available}"
            ) from exc


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
    quality_token = f"{time.time_ns():x}" if quality_root is not None else None
    for schema_name in sorted(tables):
        report = validate_table(
            tables[schema_name],
            schema_name,
            max_spread_bps=config.quality.max_spread_bps,
            max_silence_ns=config.quality.max_silence_ms * 1_000_000,
        )
        reports.append(report)
        if quality_root is not None:
            path = quality_root / f"{schema_name}.validation-{quality_token}.json"
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
    quality_artifacts: tuple[Path, ...],
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
        "quality_artifacts": [
            {
                "path": str(path.resolve().relative_to(destination)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in quality_artifacts
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
        quality_artifacts=summary.report_paths,
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


def _raw_page_result(page: RawPage) -> RawArtifactResult:
    """Bind one downloader callback to its immutable raw bytes and sidecar."""
    observed_sha256 = sha256_file(page.path)
    if observed_sha256 != page.sha256:
        raise IngestionError(f"raw page checksum disagrees with callback metadata: {page.path}")
    return RawArtifactResult(
        path=page.path,
        manifest_path=page.manifest_path,
        sha256=observed_sha256,
        manifest_sha256=sha256_file(page.manifest_path),
    )


def _public_trade_batches(
    *,
    config: ProjectConfig,
    client: BinancePublicClient,
    raw_root: Path,
    start_ns: int,
    end_ns: int,
    row_cap: int,
    validator: IncrementalQualityValidator,
    used_raw_artifacts: list[RawArtifactResult],
    symbol_results: list[SymbolDownloadResult],
) -> Iterator[pa.RecordBatch]:
    """Yield each normalized public page once while collecting bounded metadata."""
    for symbol in config.data.symbols:
        metadata = client.fetch_exchange_info(symbol=symbol, raw_root=raw_root)
        metadata_sha256 = sha256_file(metadata.source_path)
        if metadata_sha256 != metadata.source_artifact_id:
            raise IngestionError(f"exchangeInfo checksum disagrees with metadata for {symbol}")
        used_raw_artifacts.append(
            RawArtifactResult(
                path=metadata.source_path,
                manifest_path=metadata.source_manifest_path,
                sha256=metadata_sha256,
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
        captured_page_count = 0
        last_captured_page: RawPage | None = None

        def capture_raw_page(page: RawPage) -> None:
            nonlocal captured_page_count, last_captured_page
            captured_page_count += 1
            last_captured_page = page
            used_raw_artifacts.append(_raw_page_result(page))

        stream = downloader.stream(
            symbol=symbol,
            start_ts_ns=start_ns,
            end_ts_ns=end_ns,
            max_events=row_cap,
            on_raw_page=capture_raw_page,
        )
        observed_rows = 0
        for batch in stream:
            if batch.num_rows < 1 or batch.num_rows > config.data.request_limit:
                raise IngestionError(
                    f"Binance stream yielded an invalid batch size for {symbol}: {batch.num_rows}"
                )
            validator.update(batch)
            observed_rows += batch.num_rows
            yield batch

        terminal = stream.summary
        if terminal.requested_start_ns != start_ns or terminal.requested_end_ns != end_ns:
            raise IngestionError(f"Binance stream requested-range summary disagrees for {symbol}")
        if terminal.rows_yielded != observed_rows:
            raise IngestionError(
                f"Binance stream row summary disagrees with yielded rows for {symbol}"
            )
        if observed_rows > row_cap:
            raise IngestionError(f"Binance adapter exceeded row cap for {symbol}")
        if captured_page_count != terminal.raw_page_count:
            raise IngestionError(
                f"Binance stream raw-page summary disagrees with callbacks for {symbol}"
            )
        if last_captured_page != terminal.last_raw_page:
            raise IngestionError(
                f"Binance stream last-page identity disagrees with callback for {symbol}"
            )
        if observed_rows == 0:
            raise IngestionError(
                f"Binance returned no aggregate trades for {symbol}; raw responses were preserved"
            )
        symbol_results.append(
            SymbolDownloadResult(
                symbol=symbol,
                metadata=metadata,
                stream_summary=terminal,
            )
        )


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
    symbol_results: list[SymbolDownloadResult] = []
    used_raw_artifacts: list[RawArtifactResult] = []
    quality_root = destination / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    quality_token = f"{time.time_ns():x}"
    findings_path = quality_root / f"trades.findings-{quality_token}.jsonl"
    with IncrementalQualityValidator(
        "trades",
        max_spread_bps=config.quality.max_spread_bps,
        max_silence_ns=config.quality.max_silence_ms * 1_000_000,
        findings_jsonl_path=findings_path,
    ) as validator:
        batches = _public_trade_batches(
            config=config,
            client=client,
            raw_root=raw_root,
            start_ns=start_ns,
            end_ns=end_ns,
            row_cap=row_cap,
            validator=validator,
            used_raw_artifacts=used_raw_artifacts,
            symbol_results=symbol_results,
        )
        store = write_partitioned_parquet(
            batches,
            root=destination / "normalized",
            dataset="trades",
            schema_name="trades",
            source=config.data.source,
            source_uri=f"{config.data.base_url}/api/v3/aggTrades",
            requested_start_ns=start_ns,
            requested_end_ns=end_ns,
            max_input_batch_rows=config.data.request_limit,
        )
        report = validator.finish()
    report_path = quality_root / f"trades.validation-{quality_token}.json"
    report.write_json(report_path)
    summary = ValidationSummary(
        reports=(report,),
        report_paths=(report_path,),
        rows_checked=report.rows_checked,
        error_count=report.error_count,
        warning_count=report.warning_count,
    )
    expected_rows = sum(item.rows for item in symbol_results)
    if store.rows != expected_rows or report.rows_checked != expected_rows:
        raise IngestionError("public stream storage, validation, and symbol row counts disagree")
    dataset_results = (
        NormalizedDatasetResult(
            schema_name="trades",
            rows=store.rows,
            table=None,
            validation=report,
            storage=store,
        ),
    )
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
        quality_artifacts=(report_path, findings_path),
        symbol_coverage=[
            {
                "symbol": item.symbol,
                "rows": item.rows,
                "complete_range": item.complete_range,
                "raw_page_count": item.raw_page_count,
                "stop_reason": item.stop_reason,
                "last_raw_page_sha256": item.last_raw_page_sha256,
                "tick_size": str(item.metadata.tick_size),
                "lot_size": str(item.metadata.lot_size),
                "stream_summary": {
                    "requested_start_ns": item.stream_summary.requested_start_ns,
                    "requested_end_ns": item.stream_summary.requested_end_ns,
                    "rows_yielded": item.stream_summary.rows_yielded,
                    "raw_page_count": item.stream_summary.raw_page_count,
                    "stop_reason": str(item.stream_summary.stop_reason),
                    "complete_range": item.stream_summary.complete_range,
                    "last_raw_page": (
                        {
                            "path": str(
                                item.stream_summary.last_raw_page.path.relative_to(destination)
                            ),
                            "manifest_path": str(
                                item.stream_summary.last_raw_page.manifest_path.relative_to(
                                    destination
                                )
                            ),
                            "sha256": item.stream_summary.last_raw_page.sha256,
                            "request_uri": item.stream_summary.last_raw_page.request_uri,
                            "row_count": item.stream_summary.last_raw_page.row_count,
                        }
                        if item.stream_summary.last_raw_page is not None
                        else None
                    ),
                },
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


class _SyntheticDataAdapter:
    """Registry wrapper around the stable synthetic ingestion API."""

    mode = "synthetic"

    def ingest(
        self,
        config: ProjectConfig,
        output_root: str | Path,
    ) -> IngestionResult:
        return ingest_synthetic(config, output_root)


@dataclass(frozen=True, slots=True)
class _BinanceRestDataAdapter:
    """Registry wrapper that carries optional HTTP-boundary test dependencies."""

    client: BinancePublicClient | None = None
    session: requests.Session | None = None
    sleep: Callable[[float], None] | None = None
    random_value: Callable[[], float] | None = None

    @property
    def mode(self) -> str:
        return "binance_rest"

    def ingest(
        self,
        config: ProjectConfig,
        output_root: str | Path,
    ) -> IngestionResult:
        return ingest_public_trades(
            config,
            output_root,
            client=self.client,
            session=self.session,
            sleep=self.sleep,
            random_value=self.random_value,
        )


def builtin_data_adapter_registry(
    *,
    client: BinancePublicClient | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] | None = None,
    random_value: Callable[[], float] | None = None,
) -> DataAdapterRegistry:
    """Build an isolated registry containing the supported built-in adapters.

    A new registry is returned on every call, preventing one test or embedding
    application from changing dispatcher behavior process-wide.  Binance HTTP
    dependencies are captured by its adapter so the dispatcher itself remains
    source-agnostic.
    """
    return DataAdapterRegistry(
        (
            _SyntheticDataAdapter(),
            _BinanceRestDataAdapter(
                client=client,
                session=session,
                sleep=sleep,
                random_value=random_value,
            ),
        )
    )


def ingest_from_config(
    config: ProjectConfig,
    output_root: str | Path,
    *,
    client: BinancePublicClient | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] | None = None,
    random_value: Callable[[], float] | None = None,
    adapter: ConfiguredDataAdapter | None = None,
    registry: DataAdapterRegistry | None = None,
) -> IngestionResult:
    """Resolve and run exactly the adapter named by ``config.data.mode``.

    ``adapter`` preserves the original one-off injection API.  ``registry`` is
    the scalable extension path for configured third-party modes.  The two are
    mutually exclusive, and unresolved modes never fall through to a different
    source implementation.
    """
    if adapter is not None and registry is not None:
        raise IngestionError("supply either adapter or registry, not both")
    if adapter is not None:
        if adapter.mode != config.data.mode:
            raise IngestionError(
                f"adapter mode {adapter.mode!r} does not match config mode {config.data.mode!r}"
            )
        selected_registry = DataAdapterRegistry((adapter,))
    elif registry is not None:
        if any(value is not None for value in (client, session, sleep, random_value)):
            raise IngestionError(
                "HTTP dependency hooks cannot be combined with an explicit registry; "
                "capture them in the registered adapter"
            )
        selected_registry = registry
    else:
        selected_registry = builtin_data_adapter_registry(
            client=client,
            session=session,
            sleep=sleep,
            random_value=random_value,
        )
    selected = selected_registry.resolve(config.data.mode)
    return selected.ingest(config, output_root)


@dataclass(frozen=True, slots=True)
class _DatasetManifestClaim:
    rows_by_path: tuple[tuple[Path, int], ...]
    write_order: tuple[Path, ...] | None


@dataclass(frozen=True, slots=True)
class _ParquetSourceKey:
    path: Path
    venue: str
    symbol: str
    continuity_id: str | None
    identity_start: int
    identity_end: int

    @property
    def group_key(self) -> tuple[str, str, tuple[int, str]]:
        continuity = (0, "") if self.continuity_id is None else (1, self.continuity_id)
        return (self.venue, self.symbol, continuity)


def _manifest_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IngestionError(f"{label} must be an integer >= {minimum}")
    return value


def _manifest_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IngestionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_dataset_artifact_claim(
    *,
    label: str,
    normalized_root: Path,
    dataset_root: Path,
    data_path: Path,
    rows: int,
    raw_artifact: Mapping[str, object],
    schema_name: str,
    schema_version: str,
    write_ordinal: int | None,
) -> None:
    claimed_data_sha = _manifest_sha256(raw_artifact.get("data_sha256"), f"{label}.data_sha256")
    observed_data_sha = sha256_file(data_path)
    if observed_data_sha != claimed_data_sha:
        raise IngestionError(f"{label}.data_sha256 checksum mismatch: {data_path}")

    relative_manifest = raw_artifact.get("manifest_path")
    if not isinstance(relative_manifest, str) or not relative_manifest:
        raise IngestionError(f"{label}.manifest_path must be a nonempty string")
    manifest_path = (normalized_root / relative_manifest).resolve()
    if not manifest_path.is_relative_to(dataset_root) or not manifest_path.is_file():
        raise IngestionError(f"{label}.manifest_path is not a declared dataset sidecar")
    claimed_manifest_sha = _manifest_sha256(
        raw_artifact.get("manifest_sha256"), f"{label}.manifest_sha256"
    )
    if sha256_file(manifest_path) != claimed_manifest_sha:
        raise IngestionError(f"{label}.manifest_sha256 checksum mismatch: {manifest_path}")

    sidecar = read_json(manifest_path)
    if not isinstance(sidecar, dict):
        raise IngestionError(f"{label}.manifest_path is not a JSON object")
    expected_identity = {
        "artifact_kind": "normalized_parquet",
        "dataset": schema_name,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "rows": rows,
        "path": str(data_path.relative_to(normalized_root)),
    }
    for key, expected in expected_identity.items():
        if sidecar.get(key) != expected:
            raise IngestionError(f"{label} sidecar {key!r} claim is inconsistent")
    checksum = sidecar.get("checksum")
    if (
        not isinstance(checksum, dict)
        or checksum.get("algorithm") != "sha256"
        or checksum.get("value") != claimed_data_sha
    ):
        raise IngestionError(f"{label} sidecar checksum claim is inconsistent")
    if write_ordinal is not None and sidecar.get("write_ordinal") != write_ordinal:
        raise IngestionError(f"{label} sidecar write_ordinal claim is inconsistent")


def _load_dataset_manifest_claim(
    path: Path,
    *,
    normalized_root: Path,
    dataset_root: Path,
    schema_name: str,
    schema_version: str,
) -> _DatasetManifestClaim:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise IngestionError(f"normalized dataset manifest is not an object: {path}")
    if payload.get("dataset") != schema_name or payload.get("schema_version") != schema_version:
        raise IngestionError(f"normalized dataset manifest identity mismatch: {path}")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise IngestionError(f"normalized dataset manifest has no artifacts: {path}")

    rows_by_path: list[tuple[Path, int]] = []
    ordinal_paths: list[tuple[int, Path]] = []
    ordinal_presence: set[bool] = set()
    seen: set[Path] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        label = f"{path.name}.artifacts[{index}]"
        if not isinstance(raw_artifact, dict):
            raise IngestionError(f"{label} must be an object")
        relative = raw_artifact.get("data_path")
        if not isinstance(relative, str) or not relative:
            raise IngestionError(f"{label}.data_path must be a nonempty string")
        data_path = (normalized_root / relative).resolve()
        if not data_path.is_relative_to(dataset_root) or not data_path.is_file():
            raise IngestionError(f"{label}.data_path is not a declared dataset Parquet file")
        if data_path in seen:
            raise IngestionError(f"{label}.data_path is duplicated")
        seen.add(data_path)
        rows = _manifest_integer(raw_artifact.get("rows"), f"{label}.rows", minimum=1)
        rows_by_path.append((data_path, rows))
        has_ordinal = "write_ordinal" in raw_artifact
        ordinal_presence.add(has_ordinal)
        write_ordinal: int | None = None
        if has_ordinal:
            write_ordinal = _manifest_integer(
                raw_artifact.get("write_ordinal"),
                f"{label}.write_ordinal",
            )
            ordinal_paths.append(
                (
                    write_ordinal,
                    data_path,
                )
            )
        _verify_dataset_artifact_claim(
            label=label,
            normalized_root=normalized_root,
            dataset_root=dataset_root,
            data_path=data_path,
            rows=rows,
            raw_artifact=raw_artifact,
            schema_name=schema_name,
            schema_version=schema_version,
            write_ordinal=write_ordinal,
        )

    if len(ordinal_presence) != 1:
        raise IngestionError(
            f"normalized dataset manifest mixes ordered and legacy artifacts: {path}"
        )
    declared_rows = _manifest_integer(payload.get("rows"), f"{path.name}.rows", minimum=1)
    if declared_rows != sum(rows for _, rows in rows_by_path):
        raise IngestionError(f"normalized dataset manifest row total is inconsistent: {path}")

    write_order: tuple[Path, ...] | None = None
    if ordinal_paths:
        observed_ordinals = sorted(ordinal for ordinal, _ in ordinal_paths)
        if observed_ordinals != list(range(len(ordinal_paths))):
            raise IngestionError(
                f"normalized dataset manifest write ordinals are not contiguous: {path}"
            )
        write_order = tuple(
            data_path for _, data_path in sorted(ordinal_paths, key=lambda item: item[0])
        )
    return _DatasetManifestClaim(
        rows_by_path=tuple(sorted(rows_by_path, key=lambda item: str(item[0]))),
        write_order=write_order,
    )


def _manifest_write_order(
    *,
    normalized_root: Path,
    dataset_root: Path,
    schema_name: str,
    schema_version: str,
    discovered_paths: tuple[Path, ...],
) -> tuple[Path, ...] | None:
    manifest_root = normalized_root / "_manifests"
    manifest_paths = (
        sorted(manifest_root.glob(f"{schema_name}.manifest-*.json"))
        if manifest_root.exists()
        else []
    )
    if not manifest_paths:
        return None
    claims = [
        _load_dataset_manifest_claim(
            path,
            normalized_root=normalized_root,
            dataset_root=dataset_root,
            schema_name=schema_name,
            schema_version=schema_version,
        )
        for path in manifest_paths
    ]
    expected_paths = frozenset(discovered_paths)
    first_rows = claims[0].rows_by_path
    for claim in claims:
        if frozenset(path for path, _ in claim.rows_by_path) != expected_paths:
            raise IngestionError(
                f"{schema_name} dataset manifests do not cover exactly the discovered parts"
            )
        if claim.rows_by_path != first_rows:
            raise IngestionError(
                f"multiple {schema_name} dataset manifests have ambiguous row claims"
            )
    for data_path, claimed_rows in first_rows:
        observed_rows = pq.ParquetFile(data_path).metadata.num_rows
        if observed_rows != claimed_rows:
            raise IngestionError(
                f"{schema_name} dataset manifest rows disagree with Parquet metadata: {data_path}"
            )
    explicit_orders = {claim.write_order for claim in claims if claim.write_order is not None}
    if len(explicit_orders) > 1:
        raise IngestionError(f"multiple {schema_name} dataset manifests have ambiguous write order")
    return next(iter(explicit_orders)) if explicit_orders else None


def _parquet_column_bounds(
    parquet: pq.ParquetFile,
    column_name: str,
    *,
    label: str,
    allow_all_null: bool = False,
) -> tuple[object | None, object | None]:
    column_index = parquet.schema_arrow.get_field_index(column_name)
    if column_index < 0:
        raise IngestionError(f"{label} is missing ordering column {column_name!r}")
    minima: list[Any] = []
    maxima: list[Any] = []
    total_nulls = 0
    metadata = parquet.metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        statistics = row_group.column(column_index).statistics
        if statistics is None or statistics.null_count is None:
            raise IngestionError(f"{label} lacks bounded statistics for {column_name!r}")
        null_count = int(statistics.null_count)
        total_nulls += null_count
        if statistics.has_min_max:
            minima.append(statistics.min)
            maxima.append(statistics.max)
        elif null_count != row_group.num_rows:
            raise IngestionError(f"{label} lacks min/max statistics for {column_name!r}")
    if total_nulls == metadata.num_rows and allow_all_null:
        return None, None
    if total_nulls != 0 or not minima or not maxima:
        raise IngestionError(f"{label} has ambiguous nulls for ordering column {column_name!r}")
    try:
        return min(minima), max(maxima)
    except TypeError as exc:
        raise IngestionError(f"{label} has incomparable statistics for {column_name!r}") from exc


def _required_text_stat(value: object | None, label: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode()
        except UnicodeDecodeError as exc:
            raise IngestionError(f"{label} is not valid UTF-8") from exc
    if not isinstance(value, str) or not value:
        raise IngestionError(f"{label} must be a nonempty string")
    return value


def _required_int_stat(value: object | None, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IngestionError(f"{label} must be an integer")
    return value


def _parquet_source_key(path: Path, schema_name: str) -> _ParquetSourceKey:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows < 1:
        raise IngestionError(f"cannot order an empty Parquet part: {path}")
    venue_min, venue_max = _parquet_column_bounds(parquet, "venue", label=str(path))
    symbol_min, symbol_max = _parquet_column_bounds(parquet, "symbol", label=str(path))
    venue = _required_text_stat(venue_min, f"{path}.venue")
    symbol = _required_text_stat(symbol_min, f"{path}.symbol")
    if venue != _required_text_stat(venue_max, f"{path}.venue") or symbol != (
        _required_text_stat(symbol_max, f"{path}.symbol")
    ):
        raise IngestionError(f"Parquet part spans multiple venue/symbol keys: {path}")

    continuity_min, continuity_max = _parquet_column_bounds(
        parquet,
        "continuity_id",
        label=str(path),
        allow_all_null=True,
    )
    if continuity_min is None and continuity_max is None:
        continuity_id = None
    else:
        continuity_id = _required_text_stat(continuity_min, f"{path}.continuity_id")
        if continuity_id != _required_text_stat(continuity_max, f"{path}.continuity_id"):
            raise IngestionError(f"Parquet part spans multiple continuity IDs: {path}")

    identity_columns = {
        "trades": ("trade_id", "trade_id"),
        "book_observations": ("sequence_start", "sequence_end"),
        "depth_deltas": ("first_update_id", "last_update_id"),
    }
    start_column, end_column = identity_columns[schema_name]
    identity_start_raw, _ = _parquet_column_bounds(parquet, start_column, label=str(path))
    _, identity_end_raw = _parquet_column_bounds(parquet, end_column, label=str(path))
    identity_start = _required_int_stat(identity_start_raw, f"{path}.{start_column}")
    identity_end = _required_int_stat(identity_end_raw, f"{path}.{end_column}")
    if identity_end < identity_start:
        raise IngestionError(f"Parquet part has an invalid source-identity range: {path}")
    return _ParquetSourceKey(
        path=path,
        venue=venue,
        symbol=symbol,
        continuity_id=continuity_id,
        identity_start=identity_start,
        identity_end=identity_end,
    )


def _legacy_source_order(paths: tuple[Path, ...], schema_name: str) -> tuple[Path, ...]:
    descriptors = [_parquet_source_key(path, schema_name) for path in paths]
    if schema_name == "trades":
        descriptors.sort(
            key=lambda item: (item.venue, item.symbol, item.identity_start, str(item.path))
        )
    else:
        descriptors.sort(key=lambda item: (*item.group_key, item.identity_start, str(item.path)))
    previous_by_group: dict[object, _ParquetSourceKey] = {}
    for descriptor in descriptors:
        group_key: object = (
            (descriptor.venue, descriptor.symbol)
            if schema_name == "trades"
            else descriptor.group_key
        )
        previous = previous_by_group.get(group_key)
        if previous is not None and previous.identity_end >= descriptor.identity_start:
            raise IngestionError(
                f"legacy {schema_name} Parquet source ranges overlap; part order is ambiguous"
            )
        previous_by_group[group_key] = descriptor
    return tuple(item.path for item in descriptors)


def _ordered_parquet_paths(
    *,
    config: ProjectConfig,
    dataset_root: Path,
    schema_name: str,
    discovered_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    normalized_root = config.data.partition_root.resolve()
    manifest_order = _manifest_write_order(
        normalized_root=normalized_root,
        dataset_root=dataset_root.resolve(),
        schema_name=schema_name,
        schema_version=config.data.schema_version,
        discovered_paths=discovered_paths,
    )
    if manifest_order is not None:
        return manifest_order
    return _legacy_source_order(discovered_paths, schema_name)


def validate_configured_input(
    config: ProjectConfig, *, tables: Mapping[str, pa.Table] | None = None
) -> ValidationSummary:
    """Validate supplied normalized tables or discover them under configured storage."""
    if tables is not None:
        return validate_only(tables, config)

    configured_row_limit = (
        config.data.events_per_symbol
        if config.data.mode == "synthetic"
        else config.data.max_events_per_symbol
    )
    if configured_row_limit is None:
        raise IngestionError("configured validation requires a finite per-symbol row limit")
    maximum_rows = configured_row_limit * len(config.data.symbols)
    reports: list[ValidationReport] = []
    for schema_name in ("book_observations", "depth_deltas", "trades"):
        dataset_root = config.data.partition_root / schema_name
        discovered_paths = (
            tuple(sorted(path.resolve() for path in dataset_root.rglob("*.parquet")))
            if dataset_root.exists()
            else ()
        )
        if discovered_paths:
            ordered_paths = _ordered_parquet_paths(
                config=config,
                dataset_root=dataset_root,
                schema_name=schema_name,
                discovered_paths=discovered_paths,
            )
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in ordered_paths)
            if rows > maximum_rows:
                raise IngestionError(
                    f"{schema_name} contains {rows} rows, above configured validation bound "
                    f"{maximum_rows}; validate bounded partitions separately"
                )
            batches = (
                batch
                for path in ordered_paths
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16_384)
            )
            report = validate_batches(
                batches,
                schema_name,
                max_spread_bps=config.quality.max_spread_bps,
                max_silence_ns=config.quality.max_silence_ms * 1_000_000,
            )
            if report.rows_checked != rows:
                raise IngestionError(
                    f"streaming validation checked {report.rows_checked} {schema_name} rows, "
                    f"but Parquet metadata declared {rows}"
                )
            reports.append(report)
    if not reports:
        raise IngestionError(
            f"no normalized Parquet inputs found under {config.data.partition_root}"
        )
    return ValidationSummary(
        reports=tuple(reports),
        report_paths=(),
        rows_checked=sum(report.rows_checked for report in reports),
        error_count=sum(report.error_count for report in reports),
        warning_count=sum(report.warning_count for report in reports),
    )
