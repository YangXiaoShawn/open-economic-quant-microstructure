"""Single-archive, streaming normalization for the locked M8 producer.

This module intentionally has no all-calendar entry point.  A caller receives
one already authenticated :class:`AcquiredDailyArchive` and decides when the
CSV member may be opened.  In particular, the M8 producer supplies a
``before_member_open`` guard for held-out dates so the durable analysis lock is
revalidated at the lowest economic-data boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure.data.binance_archive import (
    AcquiredDailyArchive,
    BinanceArchivePayloadError,
)
from microstructure.data.quality import IncrementalQualityValidator
from microstructure.data.storage import DatasetWriteResult, write_partitioned_parquet
from microstructure.m8_config import M8Period, M8StudyConfig
from microstructure.m8_manifest import M8ArchiveEntry, M8NormalizedPart, M8SymbolMetadata
from microstructure.provenance import read_json, sha256_file

_DEFAULT_BATCH_ROWS = 65_536


class M8NormalizationError(RuntimeError):
    """A system or caller-contract failure while normalizing one archive."""


M8NormalizationFailureKind = Literal[
    "PAYLOAD_OR_CONTINUITY",
    "QUALITY_GATE",
    "POSTWRITE_CONSISTENCY",
]
M8NormalizationEvidenceCompletion = Literal[
    "PARTIAL_STREAM",
    "COMPLETE_DATASET_AND_QUALITY",
]


class M8InsufficientDataError(M8NormalizationError):
    """A deterministic archive/data-quality failure in a declared date."""

    def __init__(
        self,
        symbol: str,
        study_date: str,
        reason: str,
        *,
        failure_kind: M8NormalizationFailureKind | None = None,
        evidence_completion: M8NormalizationEvidenceCompletion | None = None,
        completed_evidence: M8ArchiveEntry | None = None,
    ) -> None:
        super().__init__(f"{symbol}/{study_date}: {reason}")
        self.symbol = symbol
        self.study_date = study_date
        self.reason = reason
        self.failure_kind = failure_kind
        self.evidence_completion = evidence_completion
        self.completed_evidence = completed_evidence


@dataclass(frozen=True, slots=True)
class M8NormalizedArchive:
    """One complete archive entry plus its bounded normalization output."""

    entry: M8ArchiveEntry
    output_root: Path


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M8NormalizationError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _downloaded_at(acquired: AcquiredDailyArchive) -> str:
    try:
        sidecar = _object(
            read_json(acquired.archive_artifact.manifest_path),
            "archive source sidecar",
        )
    except M8NormalizationError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise M8NormalizationError(f"cannot parse archive source sidecar: {exc}") from exc
    value = sidecar.get("downloaded_at_utc")
    if type(value) is not str or not value:
        raise M8NormalizationError("archive source sidecar lacks downloaded_at_utc")
    return value


def _parts(storage: DatasetWriteResult) -> tuple[M8NormalizedPart, ...]:
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


def _validate_contract(
    config: M8StudyConfig,
    period: M8Period,
    metadata: M8SymbolMetadata,
    acquired: AcquiredDailyArchive,
    root: Path,
) -> None:
    request = acquired.request
    if request.symbol != metadata.symbol or request.date != period.date:
        raise M8NormalizationError("acquired archive identity disagrees with requested period")
    if request.tick_size != metadata.tick_size or request.lot_size != metadata.lot_size:
        raise M8NormalizationError("acquired archive scales disagree with frozen metadata")
    if metadata.status != "TRADING" or metadata.symbol not in config.study.symbols:
        raise M8NormalizationError("symbol metadata does not prove a frozen TRADING symbol")
    evidence = (
        acquired.archive_artifact.path.resolve(),
        acquired.archive_artifact.manifest_path.resolve(),
        acquired.checksum_artifact.path.resolve(),
        acquired.checksum_artifact.manifest_path.resolve(),
    )
    if len(set(evidence)) != len(evidence) or any(
        not path.is_relative_to(root) or not path.is_file() for path in evidence
    ):
        raise M8NormalizationError("stage-local raw evidence is missing, reused, or escapes root")
    if acquired.upstream_sha256 != acquired.archive_artifact.sha256:
        raise M8NormalizationError("official checksum does not authenticate the stage-local ZIP")
    for artifact in (acquired.archive_artifact, acquired.checksum_artifact):
        if artifact.path.stat().st_size != artifact.bytes:
            raise M8NormalizationError("stage-local raw artifact byte count changed")
        if sha256_file(artifact.path) != artifact.sha256:
            raise M8NormalizationError("stage-local raw artifact checksum changed")
        if sha256_file(artifact.manifest_path) != artifact.manifest_sha256:
            raise M8NormalizationError("stage-local raw sidecar checksum changed")


def _validate_frozen_period(config: M8StudyConfig, period: M8Period) -> None:
    """Reject a date/role reinterpretation before any economic member can open."""

    if period not in config.periods:
        raise M8NormalizationError("M8 period date/role is not an exact frozen configuration entry")


def normalize_m8_archive(
    config: M8StudyConfig,
    period: M8Period,
    metadata: M8SymbolMetadata,
    acquired: AcquiredDailyArchive,
    input_root: str | Path,
    *,
    output_root: str | Path | None = None,
    before_member_open: Callable[[], None] | None = None,
    batch_rows: int = _DEFAULT_BATCH_ROWS,
) -> M8NormalizedArchive:
    """Stream one authenticated archive through DQ and partitioned Parquet.

    ``input_root`` is the immutable raw-evidence authority used for containment
    checks.  ``output_root`` may isolate normalized and DQ artifacts elsewhere;
    omitting it preserves the legacy co-located layout.  The function never
    retains a full daily trade table.  Archive parsing, incremental quality
    validation, and Parquet writing are bounded by ``batch_rows``.  Any
    deterministic payload/continuity/DQ failure is exposed as
    :class:`M8InsufficientDataError` so the producer can publish a terminal
    failed-result bundle without selecting a replacement date.
    """

    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("batch_rows must be a positive integer")
    raw_root = Path(input_root).resolve()
    if not raw_root.is_dir():
        raise M8NormalizationError(f"stage-local M8 input root is missing: {raw_root}")
    derived_root = raw_root if output_root is None else Path(output_root).resolve()
    if derived_root.exists() and not derived_root.is_dir():
        raise M8NormalizationError(
            f"stage-local M8 normalization output root is not a directory: {derived_root}"
        )
    _validate_frozen_period(config, period)
    if period.role in {"primary_test", "replication_test"} and before_member_open is None:
        raise M8NormalizationError(
            "held-out M8 archives require a lock-revalidation callback before member open"
        )
    _validate_contract(config, period, metadata, acquired, raw_root)
    symbol = metadata.symbol
    study_date = period.date.isoformat()
    base = derived_root / "normalized" / symbol / study_date
    if base.exists():
        raise M8NormalizationError(
            f"refusing to overwrite prior normalized evidence for {symbol}/{study_date}"
        )
    quality_root = derived_root / "quality" / symbol / study_date
    if quality_root.exists():
        raise M8NormalizationError(
            f"refusing to overwrite prior quality evidence for {symbol}/{study_date}"
        )
    quality_root.mkdir(parents=True, exist_ok=False)
    findings_path = quality_root / "findings.jsonl"
    report_path = quality_root / "report.json"
    day_start = datetime(period.date.year, period.date.month, period.date.day, tzinfo=UTC)
    day_start_ns = int(day_start.timestamp()) * 1_000_000_000
    day_end_ns = day_start_ns + 86_400 * 1_000_000_000
    stream = acquired.iter_normalized_batches(
        batch_rows=batch_rows,
        before_member_open=before_member_open,
    )
    try:
        with IncrementalQualityValidator(
            "trades",
            findings_jsonl_path=findings_path,
        ) as validator:

            def validated_batches() -> Iterator[pa.RecordBatch]:
                for batch in stream:
                    if batch.num_rows < 1 or batch.num_rows > batch_rows:
                        raise M8NormalizationError(
                            "archive stream violated its configured row bound"
                        )
                    validator.update(batch)
                    yield batch

            storage = write_partitioned_parquet(
                validated_batches(),
                root=base,
                dataset="trades",
                schema_name="trades",
                source=config.study.source,
                source_uri=acquired.archive_artifact.source_uri,
                downloaded_at_utc=_downloaded_at(acquired),
                source_checksum_sha256=acquired.archive_artifact.sha256,
                requested_start_ns=day_start_ns,
                requested_end_ns=day_end_ns,
                max_input_batch_rows=batch_rows,
                max_rows_per_file=250_000,
            )
            report = validator.finish()
        summary = stream.summary
        report.write_json(report_path)
    except BinanceArchivePayloadError as exc:
        stream.close()
        raise M8InsufficientDataError(
            symbol,
            study_date,
            str(exc),
            failure_kind="PAYLOAD_OR_CONTINUITY",
            evidence_completion="PARTIAL_STREAM",
        ) from exc
    except BaseException:
        stream.close()
        raise

    if summary.symbol != symbol or summary.date != study_date:
        raise M8InsufficientDataError(
            symbol,
            study_date,
            "archive summary identity disagrees",
            failure_kind="POSTWRITE_CONSISTENCY",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
        )
    if summary.source_archive_sha256 != acquired.archive_artifact.sha256:
        raise M8InsufficientDataError(
            symbol,
            study_date,
            "archive summary checksum disagrees",
            failure_kind="POSTWRITE_CONSISTENCY",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
        )
    if summary.expanded_bytes != acquired.declared_uncompressed_bytes:
        raise M8InsufficientDataError(
            symbol,
            study_date,
            "expanded byte count disagrees",
            failure_kind="POSTWRITE_CONSISTENCY",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
        )
    if storage.rows != summary.rows or report.rows_checked != summary.rows:
        raise M8InsufficientDataError(
            symbol,
            study_date,
            "stream/storage/DQ row counts disagree",
            failure_kind="POSTWRITE_CONSISTENCY",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
        )
    if summary.last_trade_id - summary.first_trade_id + 1 != summary.rows:
        raise M8InsufficientDataError(
            symbol,
            study_date,
            "aggregate trade IDs are noncontiguous",
            failure_kind="POSTWRITE_CONSISTENCY",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
        )
    normalized_parts = _parts(storage)
    if not normalized_parts or sum(part.rows for part in normalized_parts) != summary.rows:
        raise M8InsufficientDataError(
            symbol,
            study_date,
            "normalized part accounting failed",
            failure_kind="POSTWRITE_CONSISTENCY",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
        )
    entry = M8ArchiveEntry(
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
        raw_source_uri=acquired.archive_artifact.source_uri,
        raw_source_manifest_path=acquired.archive_artifact.manifest_path.resolve(),
        raw_source_manifest_sha256=acquired.archive_artifact.manifest_sha256,
        raw_source_manifest_bytes=acquired.archive_artifact.manifest_path.stat().st_size,
        raw_checksum_path=acquired.checksum_artifact.path.resolve(),
        raw_checksum_sha256=acquired.checksum_artifact.sha256,
        raw_checksum_bytes=acquired.checksum_artifact.bytes,
        raw_checksum_source_uri=acquired.checksum_artifact.source_uri,
        raw_checksum_source_manifest_path=acquired.checksum_artifact.manifest_path.resolve(),
        raw_checksum_source_manifest_sha256=acquired.checksum_artifact.manifest_sha256,
        raw_checksum_source_manifest_bytes=acquired.checksum_artifact.manifest_path.stat().st_size,
        normalized_dataset_manifest_path=storage.manifest_path.resolve(),
        normalized_dataset_manifest_sha256=storage.manifest_sha256,
        normalized_dataset_manifest_bytes=storage.manifest_path.stat().st_size,
        normalized_parts=normalized_parts,
        quality_report_path=report_path.resolve(),
        quality_report_sha256=sha256_file(report_path),
        quality_report_bytes=report_path.stat().st_size,
        quality_findings_path=findings_path.resolve(),
        quality_findings_sha256=sha256_file(findings_path),
        quality_findings_bytes=findings_path.stat().st_size,
        quality_errors=report.error_count,
        quality_warnings=report.warning_count,
    )
    if report.error_count or (report.warning_count and not config.quality.allow_quality_warnings):
        raise M8InsufficientDataError(
            symbol,
            study_date,
            f"quality gate reported {report.error_count} errors and "
            f"{report.warning_count} warnings",
            failure_kind="QUALITY_GATE",
            evidence_completion="COMPLETE_DATASET_AND_QUALITY",
            completed_evidence=entry,
        )
    return M8NormalizedArchive(entry=entry, output_root=derived_root)


__all__ = [
    "M8InsufficientDataError",
    "M8NormalizationError",
    "M8NormalizationEvidenceCompletion",
    "M8NormalizationFailureKind",
    "M8NormalizedArchive",
    "normalize_m8_archive",
]
