"""One explicit four-day public aggregate-trade exploratory study.

This producer is deliberately separate from the frozen M8 trade and live-L2
authorities.  It reuses their bounded archive, normalization, causal-feature,
and numeric fitted-state primitives without weakening either frozen contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, cast

import polars as pl

from microstructure.config import FeatureConfig, ModelConfig
from microstructure.data.binance import BinancePublicClient, SymbolMetadata
from microstructure.data.binance_archive import (
    AcquiredDailyArchive,
    ArchiveDownloadLimits,
    BinanceArchiveClient,
    DailyArchiveRequest,
)
from microstructure.data.evidence_budget import RetainedEvidenceBudget
from microstructure.m8_config import (
    M8Claims,
    M8Features,
    M8Models,
    M8Period,
    M8Quality,
    M8Study,
    M8StudyConfig,
)
from microstructure.m8_manifest import M8ArchiveEntry, M8SymbolMetadata
from microstructure.m8_normalization import normalize_m8_archive
from microstructure.provenance import (
    git_source_tree_sha256,
    read_json,
    sha256_file,
    strict_git_state,
    utc_now_iso,
)
from microstructure.research.multidate import (
    AnalysisLock,
    LockedSelection,
    evaluate_locked_multidate_tests,
    select_multidate_model,
)
from microstructure.research.trade_only import (
    build_trade_only_research_frame,
    validate_trade_only_temporal_contract,
)

SCHEMA_VERSION = "exploratory-aggtrades-study-v1"
EVIDENCE_TIER = "PUBLIC_ARCHIVE_EXPLORATORY"
EXPECTED_DATES = (
    ("2026-08-05", "train"),
    ("2026-08-06", "validation"),
    ("2026-08-07", "primary_test"),
    ("2026-08-08", "replication_test"),
)
EXPECTED_SYMBOLS = ("BTCUSDT", "ETHUSDT")
PROTOCOL_RELATIVE_PATH = "docs/EXPLORATORY_AGGTRADES_2026_08_05_08.md"
SUCCESS_BYTES = b"complete\n"


class ExploratoryStudyError(RuntimeError):
    """Raised when the exploratory producer cannot preserve its authority."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_bytes(value) + b"\n")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ExploratoryStudyError(f"{label} must be a string-keyed table")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ExploratoryStudyError(f"{label} keys differ from the exploratory contract")


def _load_config(path: Path) -> M8StudyConfig:
    source_path = path.resolve()
    source = source_path.read_bytes()
    try:
        payload = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ExploratoryStudyError("cannot parse exploratory configuration") from error
    root = _mapping(payload, "configuration")
    _exact_keys(root, {"study", "periods", "features", "models", "quality", "claims"}, "config")
    study_raw = _mapping(root["study"], "study")
    periods_raw = root["periods"]
    features_raw = _mapping(root["features"], "features")
    models_raw = _mapping(root["models"], "models")
    quality_raw = _mapping(root["quality"], "quality")
    claims_raw = _mapping(root["claims"], "claims")
    if not isinstance(periods_raw, list) or len(periods_raw) != 4:
        raise ExploratoryStudyError("periods must declare exactly four dates")

    study = M8Study(
        name=str(study_raw["name"]),
        protocol_version=str(study_raw["protocol_version"]),
        evidence_tier=str(study_raw["evidence_tier"]),
        seed=int(study_raw["seed"]),
        source=str(study_raw["source"]),
        symbols=tuple(str(value) for value in cast(Sequence[object], study_raw["symbols"])),
        selection_metric=str(study_raw["selection_metric"]),
        target=str(study_raw["target"]),
        label_horizon_events=int(study_raw["label_horizon_events"]),
        calibration_fraction=float(study_raw["calibration_fraction"]),
        bootstrap_samples=int(study_raw["bootstrap_samples"]),
        bootstrap_block_events=int(study_raw["bootstrap_block_events"]),
        feature_stability_bins=int(study_raw["feature_stability_bins"]),
        max_archive_compressed_bytes=int(study_raw["max_archive_compressed_bytes"]),
        max_archive_uncompressed_bytes=int(study_raw["max_archive_uncompressed_bytes"]),
        max_total_download_bytes=int(study_raw["max_total_download_bytes"]),
    )
    periods = tuple(
        M8Period(
            date=date.fromisoformat(str(_mapping(value, "period")["date"])),
            role=cast(Any, str(_mapping(value, "period")["role"])),
        )
        for value in periods_raw
    )
    features = M8Features(
        trade_windows=tuple(
            int(cast(Any, value)) for value in cast(Sequence[object], features_raw["trade_windows"])
        ),
        volatility_window=int(features_raw["volatility_window"]),
        intensity_window=int(features_raw["intensity_window"]),
        large_trade_quantile=float(features_raw["large_trade_quantile"]),
    )
    models = M8Models(
        logistic_c_values=tuple(
            float(cast(Any, value))
            for value in cast(Sequence[object], models_raw["logistic_c_values"])
        ),
        tree_max_depth_values=tuple(
            int(cast(Any, value))
            for value in cast(Sequence[object], models_raw["tree_max_depth_values"])
        ),
        tree_min_samples_leaf=int(models_raw["tree_min_samples_leaf"]),
    )
    quality = M8Quality(
        fail_on_error=bool(quality_raw["fail_on_error"]),
        require_complete_daily_archive=bool(quality_raw["require_complete_daily_archive"]),
        require_contiguous_trade_ids_within_symbol_date=bool(
            quality_raw["require_contiguous_trade_ids_within_symbol_date"]
        ),
        require_nondecreasing_event_time=bool(quality_raw["require_nondecreasing_event_time"]),
        allow_quality_warnings=bool(quality_raw["allow_quality_warnings"]),
    )
    claims = M8Claims(
        allow_p_values=bool(claims_raw["allow_p_values"]),
        allow_significance_claim=bool(claims_raw["allow_significance_claim"]),
        allow_cross_instrument_pooling=bool(claims_raw["allow_cross_instrument_pooling"]),
        allow_execution_claim=bool(claims_raw["allow_execution_claim"]),
        allow_profitability_claim=bool(claims_raw["allow_profitability_claim"]),
    )
    config = M8StudyConfig(
        path=source_path,
        source_sha256=_sha_bytes(source),
        study=study,
        periods=periods,
        features=features,
        models=models,
        quality=quality,
        claims=claims,
    )
    observed_dates = tuple((item.date.isoformat(), item.role) for item in config.periods)
    if (
        config.study.name != "binance-aggtrades-2026-08-05-08-exploratory"
        or config.study.protocol_version != "1.0.0"
        or config.study.evidence_tier != EVIDENCE_TIER
        or config.study.source != "binance_spot_daily_aggtrades_archive"
        or config.study.symbols != EXPECTED_SYMBOLS
        or observed_dates != EXPECTED_DATES
        or config.study.selection_metric != "log_loss"
        or config.study.target != "future_trade_up"
        or config.study.label_horizon_events != 20
        or not config.quality.fail_on_error
        or not config.quality.allow_quality_warnings
        or any(asdict(config.claims).values())
    ):
        raise ExploratoryStudyError("configuration differs from the declared exploratory study")
    if config.study.max_total_download_bytes < 1:
        raise ExploratoryStudyError("total retained-evidence budget must be positive")
    return config


def _protocol_path(config: M8StudyConfig) -> Path:
    result = config.path.parent.parent / PROTOCOL_RELATIVE_PATH
    if not result.is_file() or result.is_symlink():
        raise ExploratoryStudyError("exploratory protocol document is missing or symbolic")
    return result


def _metadata_authority(value: SymbolMetadata) -> M8SymbolMetadata:
    sidecar = _mapping(read_json(value.source_manifest_path), "exchangeInfo sidecar")
    source_uri = sidecar.get("source_uri")
    if type(source_uri) is not str or not source_uri:
        raise ExploratoryStudyError("exchangeInfo sidecar lacks its source URI")
    return M8SymbolMetadata(
        symbol=value.symbol,
        status=value.status,
        tick_size=value.tick_size,
        lot_size=value.lot_size,
        observed_ts_ns=value.observed_ts_ns,
        raw_path=value.source_path.resolve(),
        raw_sha256=sha256_file(value.source_path),
        raw_bytes=value.source_path.stat().st_size,
        source_uri=source_uri,
        source_manifest_path=value.source_manifest_path.resolve(),
        source_manifest_sha256=sha256_file(value.source_manifest_path),
        source_manifest_bytes=value.source_manifest_path.stat().st_size,
    )


def _feature_config(config: M8StudyConfig) -> FeatureConfig:
    return FeatureConfig(
        trade_windows=config.features.trade_windows,
        volatility_window=config.features.volatility_window,
        intensity_window=config.features.intensity_window,
        label_horizon_events=config.study.label_horizon_events,
        large_trade_quantile=config.features.large_trade_quantile,
    )


def _model_config(config: M8StudyConfig) -> ModelConfig:
    return ModelConfig(
        selection_metric=config.study.selection_metric,
        logistic_c_values=config.models.logistic_c_values,
        tree_max_depth_values=config.models.tree_max_depth_values,
        tree_min_samples_leaf=config.models.tree_min_samples_leaf,
    )


def _feature_columns(config: M8StudyConfig) -> tuple[str, ...]:
    columns = ["log_trade_return_1"]
    for window in config.features.trade_windows:
        columns.extend(
            (
                f"signed_trade_volume_w{window}",
                f"trade_volume_w{window}",
                f"trade_imbalance_w{window}",
            )
        )
    columns.extend(
        (
            f"trade_count_w{config.features.intensity_window}",
            f"trade_intensity_w{config.features.intensity_window}",
            f"realized_volatility_w{config.features.volatility_window}",
        )
    )
    return tuple(dict.fromkeys(columns))


def _evaluation_columns(config: M8StudyConfig) -> tuple[str, ...]:
    return (
        "study_date",
        "study_role",
        "symbol",
        "decision_ts_ns",
        "decision_sequence",
        "decision_trade_id",
        "continuity_id",
        "feature_continuity_id",
        "label_continuity_id",
        "max_feature_source_ts_ns",
        "max_feature_source_trade_id",
        "label_start_ts_ns",
        "label_start_trade_id",
        "label_information_end_ts_ns",
        "label_information_end_trade_id",
        "feature_ready",
        "right_censored",
        config.study.target,
        *_feature_columns(config),
    )


def _entry_payload(entry: M8ArchiveEntry, root: Path) -> dict[str, object]:
    return {
        "symbol": entry.symbol,
        "date": entry.date.isoformat(),
        "role": entry.role,
        "rows": entry.rows,
        "first_trade_id": entry.first_trade_id,
        "last_trade_id": entry.last_trade_id,
        "observed_start_ns": entry.observed_start_ns,
        "observed_end_inclusive_ns": entry.observed_end_inclusive_ns,
        "quality_errors": entry.quality_errors,
        "quality_warnings": entry.quality_warnings,
        "raw_zip": {
            "path": str(entry.raw_zip_path.relative_to(root)),
            "sha256": entry.raw_zip_sha256,
            "bytes": entry.raw_zip_bytes,
        },
        "official_checksum": {
            "path": str(entry.raw_checksum_path.relative_to(root)),
            "sha256": entry.raw_checksum_sha256,
            "bytes": entry.raw_checksum_bytes,
        },
        "normalized_manifest": {
            "path": str(entry.normalized_dataset_manifest_path.relative_to(root)),
            "sha256": entry.normalized_dataset_manifest_sha256,
            "bytes": entry.normalized_dataset_manifest_bytes,
        },
        "parts": [
            {
                "path": str(item.data_path.relative_to(root)),
                "sha256": item.data_sha256,
                "bytes": item.data_bytes,
                "rows": item.rows,
            }
            for item in entry.normalized_parts
        ],
        "quality_report": {
            "path": str(entry.quality_report_path.relative_to(root)),
            "sha256": entry.quality_report_sha256,
            "bytes": entry.quality_report_bytes,
        },
        "quality_findings": {
            "path": str(entry.quality_findings_path.relative_to(root)),
            "sha256": entry.quality_findings_sha256,
            "bytes": entry.quality_findings_bytes,
        },
    }


def _build_evaluation_frame(
    entry: M8ArchiveEntry,
    config: M8StudyConfig,
    data_root: Path,
) -> Path:
    paths = [item.data_path for item in entry.normalized_parts]
    frame = pl.read_parquet(paths, rechunk=False)
    research = build_trade_only_research_frame(frame, _feature_config(config)).with_columns(
        pl.lit(entry.date.isoformat()).alias("study_date"),
        pl.lit(entry.role).alias("study_role"),
    )
    validate_trade_only_temporal_contract(research)
    result = research.select(_evaluation_columns(config))
    output = data_root / "derived" / "research" / entry.symbol / entry.date.isoformat()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "evaluation.parquet"
    result.write_parquet(destination, compression="zstd", statistics=True)
    del frame, research, result
    return destination


def _source_authority(project_root: Path) -> dict[str, object]:
    state = strict_git_state(project_root)
    if state.dirty:
        raise ExploratoryStudyError("exploratory producer requires a clean committed source")
    return {
        "commit": state.commit,
        "dirty": False,
        "source_tree_sha256": git_source_tree_sha256(project_root),
    }


def _raw_manifest(
    config: M8StudyConfig,
    metadata: Mapping[str, M8SymbolMetadata],
    archives: Mapping[tuple[str, str], AcquiredDailyArchive],
    data_root: Path,
) -> tuple[Path, str]:
    raw_root = data_root / "raw"
    payload = {
        "schema_version": "exploratory-aggtrades-raw-v1",
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_sha256": sha256_file(_protocol_path(config)),
        "csv_members_opened": False,
        "metadata": [
            {
                "symbol": symbol,
                "status": item.status,
                "tick_size": str(item.tick_size),
                "lot_size": str(item.lot_size),
                "path": str(item.raw_path.relative_to(raw_root)),
                "sha256": item.raw_sha256,
                "bytes": item.raw_bytes,
                "sidecar_path": str(item.source_manifest_path.relative_to(raw_root)),
                "sidecar_sha256": item.source_manifest_sha256,
                "sidecar_bytes": item.source_manifest_bytes,
            }
            for symbol, item in sorted(metadata.items())
        ],
        "archives": [
            {
                "symbol": symbol,
                "date": day,
                "role": next(item.role for item in config.periods if item.date.isoformat() == day),
                "zip_path": str(value.archive_artifact.path.relative_to(raw_root)),
                "zip_sha256": value.archive_artifact.sha256,
                "zip_bytes": value.archive_artifact.bytes,
                "zip_sidecar_path": str(value.archive_artifact.manifest_path.relative_to(raw_root)),
                "zip_sidecar_sha256": value.archive_artifact.manifest_sha256,
                "checksum_path": str(value.checksum_artifact.path.relative_to(raw_root)),
                "checksum_sha256": value.checksum_artifact.sha256,
                "checksum_bytes": value.checksum_artifact.bytes,
                "checksum_sidecar_path": str(
                    value.checksum_artifact.manifest_path.relative_to(raw_root)
                ),
                "checksum_sidecar_sha256": value.checksum_artifact.manifest_sha256,
                "official_zip_sha256": value.upstream_sha256,
                "declared_uncompressed_bytes": value.declared_uncompressed_bytes,
            }
            for (symbol, day), value in sorted(archives.items())
        ],
    }
    destination = data_root / "raw_manifest.json"
    if destination.exists():
        raise ExploratoryStudyError("raw manifest target already exists")
    _write_json(destination, payload)
    return destination, sha256_file(destination)


def _persist_locks(
    selections: Mapping[str, LockedSelection],
    stage: Path,
    *,
    config: M8StudyConfig,
    raw_manifest_sha256: str,
    source: Mapping[str, object],
) -> tuple[Path, str]:
    child_claims: dict[str, dict[str, object]] = {}
    for symbol, selection in sorted(selections.items()):
        child = stage / "analysis" / "locks" / f"{symbol}.selection.json"
        _atomic_write(child, selection.lock.payload_json.encode("utf-8") + b"\n")
        child_claims[symbol] = {
            "path": str(child.relative_to(stage)),
            "sha256": sha256_file(child),
            "selection_lock_sha256": selection.lock.sha256,
            "fitted_state_sha256": selection.fitted_state.sha256,
            "selected_model": selection.selected_model,
        }
        comparison = stage / "models" / f"{symbol}.validation_candidates.parquet"
        comparison.parent.mkdir(parents=True, exist_ok=True)
        selection.validation_comparison.write_parquet(comparison, compression="zstd")
    payload = {
        "schema_version": "exploratory-aggtrades-analysis-lock-v1",
        "created_at_utc": utc_now_iso(),
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_sha256": sha256_file(_protocol_path(config)),
        "raw_manifest_sha256": raw_manifest_sha256,
        "source": dict(source),
        "children": child_claims,
        "development_dates": ["2026-08-05", "2026-08-06"],
        "heldout_dates": ["2026-08-07", "2026-08-08"],
        "heldout_csv_members_opened_before_lock": False,
    }
    aggregate = stage / "analysis" / "analysis_lock.json"
    _write_json(aggregate, payload)
    digest = sha256_file(aggregate)
    _atomic_write(aggregate.with_suffix(".sha256"), f"{digest}  {aggregate.name}\n".encode())
    return aggregate, digest


def _lock_guard(
    *,
    aggregate_path: Path,
    aggregate_sha256: str,
    config: M8StudyConfig,
    project_root: Path,
    source: Mapping[str, object],
) -> None:
    if sha256_file(config.path) != config.source_sha256:
        raise ExploratoryStudyError("configuration changed before held-out member open")
    if sha256_file(_protocol_path(config)) != cast(
        str, read_json(aggregate_path)["protocol_sha256"]
    ):
        raise ExploratoryStudyError("protocol changed before held-out member open")
    if sha256_file(aggregate_path) != aggregate_sha256:
        raise ExploratoryStudyError("aggregate lock changed before held-out member open")
    if _source_authority(project_root) != dict(source):
        raise ExploratoryStudyError("source identity changed before held-out member open")
    aggregate = _mapping(read_json(aggregate_path), "analysis lock")
    children = _mapping(aggregate["children"], "analysis lock children")
    for claim in children.values():
        child = _mapping(claim, "child lock claim")
        path = aggregate_path.parents[1] / str(child["path"])
        if sha256_file(path) != child["sha256"]:
            raise ExploratoryStudyError("child lock changed before held-out member open")


def _checksums(root: Path) -> tuple[Path, str]:
    paths = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name not in {"CHECKSUMS.sha256", "_SUCCESS"}
    )
    lines = [f"{sha256_file(item)}  {item.relative_to(root).as_posix()}" for item in paths]
    destination = root / "CHECKSUMS.sha256"
    _atomic_write(destination, ("\n".join(lines) + "\n").encode("utf-8"))
    return destination, sha256_file(destination)


def verify_exploratory_run(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir).resolve()
    marker = root / "_SUCCESS"
    if marker.read_bytes() != SUCCESS_BYTES:
        raise ExploratoryStudyError("run success marker is missing or invalid")
    checksums = root / "CHECKSUMS.sha256"
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ExploratoryStudyError(f"run artifact failed checksum verification: {relative}")
    evidence = _mapping(read_json(root / "data" / "input_evidence.json"), "input evidence")
    for item in cast(Sequence[object], evidence["files"]):
        claim = _mapping(item, "input evidence file")
        path = Path(str(claim["absolute_path"]))
        if path.stat().st_size != int(claim["bytes"]) or sha256_file(path) != claim["sha256"]:
            raise ExploratoryStudyError(f"external input changed: {path}")
    manifest = root / "run_manifest.json"
    return {
        "status": "COMPLETE",
        "integrity": "verified",
        "run_dir": str(root),
        "run_manifest": str(manifest),
        "run_manifest_sha256": sha256_file(manifest),
        "checksums": str(checksums),
        "checksums_sha256": sha256_file(checksums),
    }


def run_exploratory_study(
    config_path: str | Path,
    data_root: str | Path,
    run_dir: str | Path,
) -> dict[str, object]:
    config = _load_config(Path(config_path))
    project_root = config.path.parent.parent.resolve()
    source = _source_authority(project_root)
    destination_data = Path(data_root).resolve()
    destination_run = Path(run_dir).resolve()
    if destination_run.exists():
        raise ExploratoryStudyError("run target already exists; immutable runs are not overwritten")
    if (destination_data / "derived").exists() or (destination_data / "raw_manifest.json").exists():
        raise ExploratoryStudyError("derived/input-manifest target already exists")
    destination_data.mkdir(parents=True, exist_ok=True)
    raw_root = destination_data / "raw"
    budget = RetainedEvidenceBudget(raw_root, config.study.max_total_download_bytes)
    metadata_client = BinancePublicClient(retained_evidence_budget=budget)
    archive_client = BinanceArchiveClient(retained_evidence_budget=budget)
    metadata: dict[str, M8SymbolMetadata] = {}
    raw_metadata: dict[str, SymbolMetadata] = {}
    for symbol in config.study.symbols:
        observed = metadata_client.fetch_exchange_info(symbol=symbol, raw_root=raw_root)
        raw_metadata[symbol] = observed
        metadata[symbol] = _metadata_authority(observed)
    limits = ArchiveDownloadLimits(
        max_compressed_bytes=config.study.max_archive_compressed_bytes,
        max_uncompressed_bytes=config.study.max_archive_uncompressed_bytes,
    )
    archives: dict[tuple[str, str], AcquiredDailyArchive] = {}
    for period in config.periods:
        for symbol in config.study.symbols:
            item = raw_metadata[symbol]
            archives[(symbol, period.date.isoformat())] = archive_client.acquire(
                DailyArchiveRequest(
                    symbol=symbol,
                    date=period.date,
                    tick_size=item.tick_size,
                    lot_size=item.lot_size,
                ),
                raw_root=raw_root,
                limits=limits,
            )
    raw_manifest_path, raw_manifest_sha256 = _raw_manifest(
        config, metadata, archives, destination_data
    )
    if sha256_file(config.path) != config.source_sha256:
        raise ExploratoryStudyError("configuration changed during raw acquisition")

    stage = destination_run.parent / f".{destination_run.name}.staging-{os.getpid()}"
    if stage.exists():
        raise ExploratoryStudyError("run staging directory already exists")
    stage.mkdir(parents=True)
    normalized: dict[tuple[str, str], M8ArchiveEntry] = {}
    evaluation_paths: dict[tuple[str, str], Path] = {}
    selections: dict[str, LockedSelection] = {}
    try:
        development = config.periods[:2]
        heldout = config.periods[2:]
        for period in development:
            for symbol in config.study.symbols:
                normalized_result = normalize_m8_archive(
                    config,
                    period,
                    metadata[symbol],
                    archives[(symbol, period.date.isoformat())],
                    raw_root,
                    output_root=destination_data / "derived",
                )
                normalized[(symbol, period.date.isoformat())] = normalized_result.entry
                evaluation_paths[(symbol, period.date.isoformat())] = _build_evaluation_frame(
                    normalized_result.entry, config, destination_data
                )
        for symbol_index, symbol in enumerate(config.study.symbols):
            frames = [
                pl.read_parquet(evaluation_paths[(symbol, period.date.isoformat())])
                for period in development
            ]
            selections[symbol] = select_multidate_model(
                frames,
                _model_config(config),
                feature_columns=_feature_columns(config),
                declared_test_dates=[item.date.isoformat() for item in heldout],
                seed=config.study.seed + symbol_index,
                calibration_bins=config.study.feature_stability_bins,
                target=config.study.target,
                calibration_fraction=config.study.calibration_fraction,
                bootstrap_draws=config.study.bootstrap_samples,
                block_width_events=config.study.bootstrap_block_events,
            )
            del frames
        aggregate_path, aggregate_sha256 = _persist_locks(
            selections,
            stage,
            config=config,
            raw_manifest_sha256=raw_manifest_sha256,
            source=source,
        )
        for period in heldout:
            for symbol in config.study.symbols:
                normalized_result = normalize_m8_archive(
                    config,
                    period,
                    metadata[symbol],
                    archives[(symbol, period.date.isoformat())],
                    raw_root,
                    output_root=destination_data / "derived",
                    before_member_open=lambda: _lock_guard(
                        aggregate_path=aggregate_path,
                        aggregate_sha256=aggregate_sha256,
                        config=config,
                        project_root=project_root,
                        source=source,
                    ),
                )
                normalized[(symbol, period.date.isoformat())] = normalized_result.entry
                evaluation_paths[(symbol, period.date.isoformat())] = _build_evaluation_frame(
                    normalized_result.entry, config, destination_data
                )

        summaries: dict[str, object] = {}
        for symbol in config.study.symbols:
            development_frames = [
                pl.read_parquet(evaluation_paths[(symbol, period.date.isoformat())])
                for period in development
            ]
            test_frames = [
                pl.read_parquet(evaluation_paths[(symbol, period.date.isoformat())])
                for period in heldout
            ]
            locked = AnalysisLock.restore(
                selections[symbol].lock.payload_json,
                selections[symbol].lock.sha256,
            )
            evaluation_result = evaluate_locked_multidate_tests(
                development_frames, test_frames, locked
            )
            model_root = stage / "models" / symbol
            metric_root = stage / "metrics" / symbol
            model_root.mkdir(parents=True, exist_ok=True)
            metric_root.mkdir(parents=True, exist_ok=True)
            evaluation_result.predictions.write_parquet(
                model_root / "selected_and_prior_predictions.parquet", compression="zstd"
            )
            evaluation_result.paired_log_loss.per_date.write_parquet(
                metric_root / "paired_log_loss_by_date.parquet", compression="zstd"
            )
            evaluation_result.feature_stability.write_parquet(
                metric_root / "feature_stability.parquet", compression="zstd"
            )
            aggregate = evaluation_result.paired_log_loss.aggregate
            summaries[symbol] = {
                "selected_model": evaluation_result.selected_model,
                "selection_lock_sha256": evaluation_result.lock_sha256,
                "replication_status": evaluation_result.paired_log_loss.replication_status,
                "equal_date_selected_minus_prior_log_loss": {
                    "point_estimate": aggregate.point_estimate,
                    "ci_low": aggregate.lower,
                    "ci_high": aggregate.upper,
                    "bootstrap_samples": aggregate.n_bootstrap,
                    "blocks": aggregate.n_blocks,
                    "status": aggregate.status,
                },
                "per_date": evaluation_result.paired_log_loss.per_date.to_dicts(),
            }
            del development_frames, test_frames, evaluation_result

        evidence_files: list[dict[str, object]] = []
        for path in sorted(
            item for item in destination_data.rglob("*") if item.is_file() and not item.is_symlink()
        ):
            evidence_files.append(
                {
                    "absolute_path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        _write_json(
            stage / "data" / "input_evidence.json",
            {
                "raw_manifest": str(raw_manifest_path),
                "raw_manifest_sha256": raw_manifest_sha256,
                "files": evidence_files,
            },
        )
        archive_rows = [
            _entry_payload(normalized[(symbol, period.date.isoformat())], destination_data)
            for period in config.periods
            for symbol in config.study.symbols
        ]
        _write_json(stage / "metrics" / "exploratory_summary.json", summaries)
        _write_json(
            stage / "run_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
                "evidence_tier": EVIDENCE_TIER,
                "evidence_scope": "trade_only_complete_public_daily_archives_retrospective",
                "generated_at_utc": utc_now_iso(),
                "config": config.public_dict(),
                "protocol": {
                    "path": PROTOCOL_RELATIVE_PATH,
                    "sha256": sha256_file(_protocol_path(config)),
                },
                "source": source,
                "raw_manifest_sha256": raw_manifest_sha256,
                "analysis_lock_sha256": aggregate_sha256,
                "archive_evidence": archive_rows,
                "symbol_results": summaries,
                "claims": {
                    "p_values_computed": False,
                    "significance_claim_authorized": False,
                    "cross_instrument_pooling": False,
                    "execution": "NOT_RUN",
                    "profitability_claim_authorized": False,
                    "live_l2_claim_authorized": False,
                },
            },
        )
        report_lines = [
            "# August 5-8 aggregate-trade exploratory result",
            "",
            "**PUBLIC_ARCHIVE_EXPLORATORY — retrospective trade-only evidence.**",
            "",
            "This result contains no L2, execution, profitability, capacity, or significance claim.",
            "",
        ]
        for symbol in config.study.symbols:
            summary = cast(Mapping[str, Any], summaries[symbol])
            report_aggregate = cast(
                Mapping[str, Any], summary["equal_date_selected_minus_prior_log_loss"]
            )
            report_lines.extend(
                [
                    f"## {symbol}",
                    "",
                    f"- Selected model: `{summary['selected_model']}`",
                    f"- Equal-date selected-minus-prior log-loss delta: `{report_aggregate['point_estimate']}`",
                    f"- Descriptive 95% interval: `[{report_aggregate['ci_low']}, {report_aggregate['ci_high']}]`",
                    f"- Directional replication status: `{summary['replication_status']}`",
                    "",
                ]
            )
        _atomic_write(stage / "reports" / "technical_report.md", "\n".join(report_lines).encode())
        _checksums(stage)
        _atomic_write(stage / "_SUCCESS", SUCCESS_BYTES)
        destination_run.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, destination_run)
        parent = os.open(destination_run.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return verify_exploratory_run(destination_run)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = (
        verify_exploratory_run(args.run_dir)
        if args.verify_only
        else run_exploratory_study(args.config, args.data_root, args.run_dir)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
