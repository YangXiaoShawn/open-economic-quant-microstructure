"""Atomic producer and verifier for the frozen four-session M8 L2 study.

The development lock is verified before either held-out payload can be opened.
Every session coordinate and both of its control-file digests are supplied by
the caller; this module never discovers a ``latest`` directory.  Publication is
terminal-marker based and never overwrites an existing run.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import polars as pl
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from microstructure.m8_l2_analysis_config import (
    M8L2AnalysisConfig,
    load_m8_l2_analysis_config,
)
from microstructure.m8_l2_capture import (
    M8L2SessionBundle,
    current_m8_l2_runtime_fingerprint_sha256,
    verify_m8_l2_session_bundle,
)
from microstructure.m8_l2_config import (
    M8_L2_PROTOCOL_SHA256,
    M8L2StudyConfig,
    load_m8_l2_config,
)
from microstructure.m8_l2_development import (
    L2DevelopmentLockResult,
    verify_m8_l2_development_lock,
)
from microstructure.m8_l2_inputs import (
    L2CampaignRuntimeIdentity,
    L2SessionFileAuthority,
    VerifiedL2SessionInput,
    verify_m8_l2_development_input,
    verify_m8_l2_heldout_input,
)
from microstructure.provenance import (
    ImportOriginError,
    assert_project_module_origins,
    git_source_tree_sha256,
    runtime_metadata,
    sha256_file,
    strict_git_state,
    utc_now_iso,
)
from microstructure.reporting.l2 import (
    L2ReportData,
    canonical_report_data_sha256,
    render_l2_executive_memo,
    render_l2_model_comparison,
    render_l2_technical_report,
)
from microstructure.research.analysis import RegimeThresholds
from microstructure.research.l2_analysis import (
    L2DescriptiveAnalysis,
    build_l2_descriptive_analysis,
)
from microstructure.research.l2_evaluation import (
    L2EvaluationResult,
    L2ExecutionReference,
    L2HeldoutEndpointFrame,
    LockedL2EndpointState,
    evaluate_locked_l2_endpoints,
    run_locked_l2_market_execution,
)
from microstructure.research.l2_multidate import (
    L2EndpointSpec,
    L2RegimeFit,
    apply_l2_regimes,
    build_l2_endpoint_frames,
    l2_model_feature_columns,
    validate_l2_endpoint_frame,
)
from microstructure.research.multidate import FinalFittedState

M8L2StudyRunStatus = Literal["COMPLETE", "INSUFFICIENT_DATA"]

_SCHEMA_VERSION = "m8-l2-study-run-v2"
_REPORT_INPUT_SCHEMA_VERSION = "m8-l2-report-inputs-v1"
_CHECKSUMS_NAME = "CHECKSUMS.sha256"
_SUCCESS_NAME = "_SUCCESS"
_INSUFFICIENT_NAME = "INSUFFICIENT_DATA"
_SUCCESS_BYTES = b"complete\n"
_INSUFFICIENT_BYTES = b"terminal\n"
_EXPECTED_COORDINATES = (
    ("2026-08-10", "train"),
    ("2026-08-11", "validation"),
    ("2026-08-12", "primary_test"),
    ("2026-08-13", "replication_test"),
)
_GIB = 1024**3
# Fail-closed producer/verifier workspace partition for a 16 GiB host.  The
# categories deliberately sum below the host ceiling even at their limits:
# causal 4 + current raw 2 + evaluation 2 + descriptive 2 + execution 2 =
# 12 GiB, leaving 4 GiB for Python/Polars/runtime and publication overhead.
_MAX_FINAL_RAW_BYTES = 2 * _GIB
_MAX_FINAL_CAUSAL_BYTES = 4 * _GIB
_MAX_CAUSAL_COORDINATE_BYTES = 2 * _GIB
_MAX_EVALUATION_WORKSPACE_BYTES = 2 * _GIB
_MAX_DESCRIPTIVE_WORKSPACE_BYTES = 2 * _GIB
_MAX_EXECUTION_WORKSPACE_BYTES = 2 * _GIB
_CAUSAL_ENDPOINT_ROW_UPPER_BYTES = 4 * 1024
_PREDICTION_ROW_UPPER_BYTES = 1536
_PYTHON_CELL_UPPER_BYTES = 192
_PYTHON_ROW_BASE_UPPER_BYTES = 512
_PYTHON_LEDGER_ROW_UPPER_BYTES = 2048
_POLARS_LEDGER_ROW_UPPER_BYTES = 1024

_L2_VALIDATION_COLUMNS = (
    "study_date",
    "study_role",
    "endpoint_name",
    "endpoint_domain",
    "symbol",
    "continuity_id",
    "observed_interval_id",
    "observed_interval_start_ns",
    "observed_interval_end_ns_exclusive",
    "decision_ts_ns",
    "decision_sequence",
    "feature_cutoff_ts_ns",
    "max_feature_source_ts_ns",
    "max_feature_source_sequence",
    "feature_continuity_id",
    "label_start_ts_ns",
    "label_start_sequence",
    "right_censored",
    "future_mid_return",
    "future_mid_up",
    "label_information_end_ts_ns",
    "label_information_end_sequence",
    "label_continuity_id",
    "ofi_signed_future_mid_markout_bps",
    "sample_id",
)
_DESCRIPTIVE_COLUMNS = (
    "endpoint_horizon_value",
    "endpoint_horizon_unit",
    "spread_bps",
    "depth_total_l1",
    "depth_total_l5",
    "depth_total_l10",
    "queue_imbalance_l1",
    "realized_volatility_w100",
    "bid_quantity",
    "ask_quantity",
    "signed_markout_side_source",
    "liquidity_regime",
    "volatility_regime",
)
_EXECUTION_EVENT_COLUMNS = (
    *_L2_VALIDATION_COLUMNS,
    "best_bid",
    "best_ask",
    "bid_quantity",
    "ask_quantity",
    "mid_price",
    "tick_size",
    "lot_size",
)
_EXECUTION_PREDICTION_COLUMNS = (
    "sample_id",
    "symbol",
    "study_date",
    "study_role",
    "endpoint_name",
    "decision_sequence",
    "selected_probability",
    "is_oos",
    "split",
    "child_lock_sha256",
    "aggregate_lock_sha256",
    "endpoint_impact_ofi_window",
)


class M8L2StudyPipelineError(RuntimeError):
    """Raised when final-study production or verification must fail closed."""


class M8L2StudyRunVerificationError(M8L2StudyPipelineError):
    """Raised when a terminal M8 L2 run differs from its authorities."""


def _frame_bytes(frame: pl.DataFrame) -> int:
    return int(frame.estimated_size("b"))


def _frames_bytes(frames: Sequence[pl.DataFrame]) -> int:
    return sum(_frame_bytes(frame) for frame in frames)


def _require_memory_budget(observed: int, maximum: int, label: str) -> None:
    if observed < 0 or observed > maximum:
        raise M8L2StudyPipelineError(
            f"{label} exceeds the fail-closed memory budget ({observed} > {maximum} bytes)"
        )


def _require_verification_memory_budget(observed: int, maximum: int, label: str) -> None:
    if observed < 0 or observed > maximum:
        raise M8L2StudyRunVerificationError(
            f"{label} exceeds the bounded verifier memory budget ({observed} > {maximum} bytes)"
        )


def _parquet_metadata_bytes(root: Path, artifact: object, label: str) -> tuple[int, int]:
    """Inspect Parquet row-group sizes without opening column payloads."""

    relative = getattr(artifact, "relative_path", None)
    claimed_rows = getattr(artifact, "rows", None)
    if not isinstance(relative, str) or not isinstance(claimed_rows, int):
        raise M8L2StudyPipelineError(f"{label} lacks bounded Parquet metadata authority")
    safe = _safe_relative(relative)
    path = root / safe
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise M8L2StudyPipelineError(f"{label} is not a regular Parquet file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            parquet = pq.ParquetFile(handle)
            metadata = parquet.metadata
            rows = int(metadata.num_rows)
            uncompressed = sum(
                int(metadata.row_group(index).total_byte_size)
                for index in range(metadata.num_row_groups)
            )
            after = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise M8L2StudyPipelineError(f"{label} changed during memory admission")
    except M8L2StudyPipelineError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise M8L2StudyPipelineError(f"cannot inspect {label} memory metadata") from error
    if rows != claimed_rows or rows < 1 or uncompressed < 1:
        raise M8L2StudyPipelineError(f"{label} Parquet metadata differs from its authority")
    return uncompressed, rows


def _preflight_symbol_raw(value: VerifiedL2SessionInput, symbol: str) -> tuple[int, int] | None:
    """Return (uncompressed raw bytes, book rows) before payload materialization.

    Verified production inputs always expose artifact descriptors.  Injected
    test loaders may omit them and are then guarded by the immediate post-load
    check in ``_build_one_symbol_frames``.
    """

    descriptor = value.symbols.get(symbol)
    books = getattr(descriptor, "book_observations", None)
    deltas = getattr(descriptor, "depth_deltas", None)
    if books is None or deltas is None:
        return None
    book_bytes, book_rows = _parquet_metadata_bytes(
        value.root, books, f"{value.session_date} {symbol} book observations"
    )
    delta_bytes, _ = _parquet_metadata_bytes(
        value.root, deltas, f"{value.session_date} {symbol} depth deltas"
    )
    return book_bytes + delta_bytes, book_rows


def _loaded_bytes(value: Any) -> int:
    books = cast(pl.DataFrame, value.book_observations)
    deltas = cast(pl.DataFrame, value.depth_deltas)
    intervals = cast(Sequence[object], value.intervals)
    return _frame_bytes(books) + _frame_bytes(deltas) + len(intervals) * 1024


def _projected_frame_bytes(frame: pl.DataFrame, columns: Sequence[str], label: str) -> int:
    selected = tuple(dict.fromkeys(columns))
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise M8L2StudyPipelineError(f"{label} projection lacks required columns: {missing}")
    return sum(int(frame.get_column(name).estimated_size("b")) for name in selected)


def _project_frame(frame: pl.DataFrame, columns: Sequence[str], label: str) -> pl.DataFrame:
    _projected_frame_bytes(frame, columns, label)
    return frame.select(*tuple(dict.fromkeys(columns)))


def _evaluation_workspace_upper_bytes(
    heldout: Sequence[L2HeldoutEndpointFrame], *, feature_count: int
) -> int:
    if feature_count < 1:
        raise M8L2StudyPipelineError("evaluation memory admission requires model features")
    rows = sum(item.frame.height for item in heldout)
    full_width_sort_and_filter = 2 * _frames_bytes([item.frame for item in heldout])
    child_and_concat = 2 * rows * _PREDICTION_ROW_UPPER_BYTES
    numpy_scratch = rows * (feature_count * 8 + 8 * 8)
    return full_width_sort_and_filter + child_and_concat + numpy_scratch


def _descriptive_projection_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*_L2_VALIDATION_COLUMNS, *_DESCRIPTIVE_COLUMNS, *feature_columns)))


def _descriptive_workspace_upper_bytes(
    causal: Sequence[pl.DataFrame], *, columns: Sequence[str]
) -> int:
    projected = sum(
        _projected_frame_bytes(frame, columns, "descriptive endpoint") for frame in causal
    )
    rows = sum(frame.height for frame in causal)
    # Projected inputs, combined concat, largest grouped/sorted temporary and
    # retained outputs are charged as four full projected equivalents.
    return 4 * projected + rows * 512


def _python_projected_rows_upper_bytes(frame: pl.DataFrame, columns: Sequence[str]) -> int:
    payload = _projected_frame_bytes(frame, columns, "Python-row input")
    return payload + frame.height * (
        _PYTHON_ROW_BASE_UPPER_BYTES + len(tuple(dict.fromkeys(columns))) * _PYTHON_CELL_UPPER_BYTES
    )


def _execution_workspace_upper_bytes(
    event_frame: pl.DataFrame,
    predictions: pl.DataFrame,
) -> int:
    signal_rows = predictions.filter(
        (pl.col("selected_probability") >= 0.55) | (pl.col("selected_probability") <= 0.45)
    ).height
    # One possible forced liquidation row is included in every scenario.
    ledger_rows_per_scenario = signal_rows + 1
    scenario_count = 9
    projected_inputs = _projected_frame_bytes(
        event_frame, _EXECUTION_EVENT_COLUMNS, "execution event"
    ) + _projected_frame_bytes(predictions, _EXECUTION_PREDICTION_COLUMNS, "execution prediction")
    python_inputs = _python_projected_rows_upper_bytes(
        event_frame, _EXECUTION_EVENT_COLUMNS
    ) + _python_projected_rows_upper_bytes(predictions, _EXECUTION_PREDICTION_COLUMNS)
    # simulate_predictions holds order/fill/position dictionaries and an
    # event-aligned equity ledger for only the current scenario.
    current_python_ledgers = (
        4 * ledger_rows_per_scenario + event_frame.height
    ) * _PYTHON_LEDGER_ROW_UPPER_BYTES
    # run_locked_l2_market_execution retains the three Polars ledgers from all
    # nine completed scenarios until its final coordinate concat.
    retained_polars_ledgers = (
        3 * ledger_rows_per_scenario * scenario_count * _POLARS_LEDGER_ROW_UPPER_BYTES
    )
    # Caller projection plus the execution layer's ordered event/prediction
    # projections can coexist; charge three complete projected input sets.
    return 3 * projected_inputs + python_inputs + current_python_ledgers + retained_polars_ledgers


def _assert_final_producer_import_origins(project_root: Path) -> None:
    try:
        assert_project_module_origins(
            project_root,
            "microstructure.m8_l2_pipeline",
            "microstructure.m8_l2_development",
            "microstructure.m8_l2_inputs",
            "microstructure.research.l2_analysis",
            "microstructure.research.l2_evaluation",
            "microstructure.research.l2_multidate",
            "microstructure.research.multidate",
            "microstructure.reporting.l2",
        )
    except ImportOriginError as error:
        raise M8L2StudyPipelineError(
            "final-study producer has a foreign or mixed import origin"
        ) from error


@dataclass(frozen=True, slots=True)
class L2StudySessionAuthority:
    """An explicit session path plus independent control-file digests."""

    bundle_path: Path
    manifest_sha256: str
    checksums_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_path", Path(self.bundle_path).absolute())
        _require_sha256(self.manifest_sha256, "session manifest authority")
        _require_sha256(self.checksums_sha256, "session checksums authority")

    @property
    def file_authority(self) -> L2SessionFileAuthority:
        return L2SessionFileAuthority(self.manifest_sha256, self.checksums_sha256)

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_path": str(self.bundle_path),
            "manifest_sha256": self.manifest_sha256,
            "checksums_sha256": self.checksums_sha256,
        }


@dataclass(frozen=True, slots=True)
class M8L2StudyRunResult:
    """One verified terminal final-study bundle."""

    root: Path
    status: M8L2StudyRunStatus
    manifest_path: Path
    manifest_sha256: str
    checksum_path: Path
    checksum_sha256: str
    marker_path: Path
    reason_codes: tuple[str, ...]

    @property
    def technical_report_path(self) -> Path:
        return self.root / "reports" / "technical_report.md"

    @property
    def executive_memo_path(self) -> Path:
        return self.root / "reports" / "executive_memo.md"

    @property
    def model_comparison_path(self) -> Path:
        return self.root / "reports" / "model_comparison.md"


def reproduce_m8_l2_study(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_session: L2StudySessionAuthority,
    validation_session: L2StudySessionAuthority,
    development_lock_dir: str | Path,
    expected_development_lock_sha256: str,
    primary_session: L2StudySessionAuthority,
    replication_session: L2StudySessionAuthority,
    run_dir: str | Path,
    *,
    expected_existing_manifest_sha256: str | None = None,
    expected_existing_checksums_sha256: str | None = None,
) -> M8L2StudyRunResult:
    """Produce a run, or reuse one only under caller-held output authority."""

    return _reproduce_m8_l2_study(
        capture_config,
        analysis_config,
        train_session,
        validation_session,
        development_lock_dir,
        expected_development_lock_sha256,
        primary_session,
        replication_session,
        run_dir,
        expected_existing_manifest_sha256=expected_existing_manifest_sha256,
        expected_existing_checksums_sha256=expected_existing_checksums_sha256,
    )


def verify_m8_l2_study_run(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_session: L2StudySessionAuthority,
    validation_session: L2StudySessionAuthority,
    development_lock_dir: str | Path,
    expected_development_lock_sha256: str,
    primary_session: L2StudySessionAuthority,
    replication_session: L2StudySessionAuthority,
    run_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_checksums_sha256: str | None = None,
) -> M8L2StudyRunResult:
    """Recursively verify a terminal run and every external authority."""

    return _verify_m8_l2_study_run(
        capture_config,
        analysis_config,
        train_session,
        validation_session,
        development_lock_dir,
        expected_development_lock_sha256,
        primary_session,
        replication_session,
        run_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_checksums_sha256=expected_checksums_sha256,
    )


def load_m8_l2_report_data(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_session: L2StudySessionAuthority,
    validation_session: L2StudySessionAuthority,
    development_lock_dir: str | Path,
    expected_development_lock_sha256: str,
    primary_session: L2StudySessionAuthority,
    replication_session: L2StudySessionAuthority,
    run_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_checksums_sha256: str | None = None,
) -> L2ReportData:
    """Load report inputs only after complete terminal and authority verification."""

    if expected_manifest_sha256 is None or expected_checksums_sha256 is None:
        raise M8L2StudyRunVerificationError(
            "report loading requires caller-held manifest and checksum authorities"
        )

    verified = verify_m8_l2_study_run(
        capture_config,
        analysis_config,
        train_session,
        validation_session,
        development_lock_dir,
        expected_development_lock_sha256,
        primary_session,
        replication_session,
        run_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_checksums_sha256=expected_checksums_sha256,
    )
    data = _load_report_data_snapshot(verified.root)
    confirmed = verify_m8_l2_study_run(
        capture_config,
        analysis_config,
        train_session,
        validation_session,
        development_lock_dir,
        expected_development_lock_sha256,
        primary_session,
        replication_session,
        run_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_checksums_sha256=expected_checksums_sha256,
    )
    if confirmed != verified:
        raise M8L2StudyRunVerificationError(
            "final run authority changed while report inputs were loaded"
        )
    return data


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    commit: str
    source_tree_sha256: str
    dirty: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "source_tree_sha256": self.source_tree_sha256,
            "dirty": self.dirty,
        }


@dataclass(frozen=True, slots=True)
class _SessionSnapshot:
    authority: L2StudySessionAuthority
    bundle: M8L2SessionBundle
    manifest: Mapping[str, Any]
    campaign: L2CampaignRuntimeIdentity


@dataclass(frozen=True, slots=True)
class _LockMaterial:
    result: L2DevelopmentLockResult
    aggregate: Mapping[str, Any]
    campaign: L2CampaignRuntimeIdentity
    source: _SourceIdentity
    states: Mapping[tuple[str, str], LockedL2EndpointState]
    regimes: Mapping[str, L2RegimeFit]
    references: Mapping[str, L2ExecutionReference]
    development_frame_sha256: Mapping[tuple[str, str], str]
    snapshot_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _ParentIdentity:
    device: int
    inode: int


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, label: str) -> None:
    if not _is_sha256(value):
        raise M8L2StudyPipelineError(f"{label} must be a lowercase SHA-256")


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise M8L2StudyPipelineError("M8 L2 authority is not finite canonical JSON") from error


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise M8L2StudyPipelineError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise M8L2StudyPipelineError(f"{label} contains forbidden constant {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except M8L2StudyPipelineError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M8L2StudyPipelineError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise M8L2StudyPipelineError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _safe_relative(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise M8L2StudyPipelineError(f"unsafe M8 L2 relative path {value!r}")
    return value


def _join(root: Path, relative: str) -> Path:
    return root.joinpath(*PurePosixPath(_safe_relative(relative)).parts)


def _reject_symlink_components(path: Path) -> None:
    requested = path.absolute()
    current = Path(requested.anchor)
    for part in requested.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise M8L2StudyPipelineError(f"cannot inspect path component {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise M8L2StudyPipelineError(f"M8 L2 path contains symlink component {current}")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise M8L2StudyPipelineError("artifact escapes the M8 L2 run root") from error


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _read_regular(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = 64 * 1024 * 1024,
    expected_sha256: str | None = None,
) -> bytes:
    try:
        before_path = path.lstat()
    except OSError as error:
        raise M8L2StudyPipelineError(f"cannot stat {label}") from error
    if (
        not stat.S_ISREG(before_path.st_mode)
        or before_path.st_size < 0
        or before_path.st_size > maximum_bytes
    ):
        raise M8L2StudyPipelineError(f"{label} is not a bounded regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise M8L2StudyPipelineError(f"cannot open {label} without following links") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not _same_stat(before_path, before):
            raise M8L2StudyPipelineError(f"{label} changed before its descriptor snapshot")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        after_path = path.lstat()
        if len(raw) > maximum_bytes or len(raw) != before.st_size:
            raise M8L2StudyPipelineError(f"{label} exceeds its bounded snapshot")
        if not _same_stat(before, after) or not _same_stat(after, after_path):
            raise M8L2StudyPipelineError(f"{label} changed during its descriptor snapshot")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise M8L2StudyPipelineError(f"{label} differs from its SHA-256 authority")
        return raw
    finally:
        os.close(descriptor)


def _read_json(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label=label, expected_sha256=expected_sha256)
    return _decode_json(raw, label), raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short M8 L2 artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    raw = _canonical_json_bytes(payload)
    _write_bytes(path, raw)
    return hashlib.sha256(raw).hexdigest()


def _write_parquet(path: Path, frame: pl.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise M8L2StudyPipelineError(f"refusing to overwrite M8 L2 artifact {path}")
    frame.write_parquet(path, compression="zstd", statistics=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return sha256_file(path)


def _copy_exact(source: Path, destination: Path, *, expected_sha256: str | None = None) -> str:
    raw = _read_regular(
        source,
        label=f"authority snapshot {source}",
        maximum_bytes=128 * 1024 * 1024,
        expected_sha256=expected_sha256,
    )
    _write_bytes(destination, raw)
    return hashlib.sha256(raw).hexdigest()


def _parse_checksums(raw: bytes, label: str) -> dict[str, str]:
    try:
        lines = raw.decode("ascii").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise M8L2StudyPipelineError(f"{label} must be ASCII") from error
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 68 or not line.endswith("\n") or line[64:66] != "  ":
            raise M8L2StudyPipelineError(f"{label} has a malformed line")
        digest = line[:64]
        relative = _safe_relative(line[66:-1])
        _require_sha256(digest, f"{label} entry")
        if relative in result:
            raise M8L2StudyPipelineError(f"{label} repeats {relative}")
        result[relative] = digest
    if not result or list(result) != sorted(result):
        raise M8L2StudyPipelineError(f"{label} is empty or not canonically ordered")
    return result


def _frame_sha256(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode())
    for chunk in frame.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3).get_chunks():
        digest.update(chunk.to_numpy().astype("<u8", copy=False).tobytes(order="C"))
    digest.update(str(frame.height).encode())
    return digest.hexdigest()


def _revalidate_configs(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
) -> tuple[M8L2StudyConfig, M8L2AnalysisConfig]:
    try:
        capture = load_m8_l2_config(capture_config.path)
        analysis = load_m8_l2_analysis_config(analysis_config.path)
    except (OSError, ValueError) as error:
        raise M8L2StudyPipelineError("frozen M8 L2 configs cannot be reloaded") from error
    if capture != capture_config or analysis != analysis_config:
        raise M8L2StudyPipelineError("in-memory M8 L2 configs differ from exact frozen bytes")
    coordinates = tuple((item.date.isoformat(), item.role) for item in capture.sessions)
    if coordinates != _EXPECTED_COORDINATES:
        raise M8L2StudyPipelineError("M8 L2 session calendar differs from the freeze")
    if (
        analysis.study.capture_config_source_sha256 != capture.source_sha256
        or analysis.study.capture_protocol_sha256 != M8_L2_PROTOCOL_SHA256
        or analysis.study.symbols != capture.study.symbols
        or analysis.study.seed != capture.study.seed
    ):
        raise M8L2StudyPipelineError("capture and analysis configs do not bind one study")
    return capture, analysis


def _current_source_identity(capture: M8L2StudyConfig) -> _SourceIdentity:
    project_root = capture.path.parent.parent.resolve()
    before = strict_git_state(project_root)
    source_tree_sha256 = git_source_tree_sha256(project_root)
    after = strict_git_state(project_root)
    if before != after:
        raise M8L2StudyPipelineError("Git identity changed during final source snapshot")
    result = _SourceIdentity(
        commit=before.commit,
        source_tree_sha256=source_tree_sha256,
        dirty=before.dirty,
    )
    if result.dirty:
        raise M8L2StudyPipelineError("final M8 L2 production requires a clean Git source tree")
    if len(result.commit) != 40 or any(char not in "0123456789abcdef" for char in result.commit):
        raise M8L2StudyPipelineError("final M8 L2 producer commit is not a lowercase Git SHA-1")
    _require_sha256(result.source_tree_sha256, "final M8 L2 source tree")
    return result


def _campaign_from_manifest(manifest: Mapping[str, Any]) -> L2CampaignRuntimeIdentity:
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping):
        raise M8L2StudyPipelineError("session manifest lacks campaign authority")
    return L2CampaignRuntimeIdentity(
        campaign_authority_sha256=str(authority.get("campaign_authority_sha256")),
        runtime_commit=str(authority.get("runtime_commit")),
        runtime_source_tree_sha256=str(authority.get("runtime_source_tree_sha256")),
        runtime_fingerprint_sha256=str(authority.get("runtime_fingerprint_sha256")),
        runtime_dirty=authority.get("runtime_dirty") is not False,
    )


def _verify_session_authority(
    authority: L2StudySessionAuthority,
    *,
    capture: M8L2StudyConfig,
    expected_date: str,
    expected_role: str,
    expected_campaign: L2CampaignRuntimeIdentity | None,
) -> _SessionSnapshot:
    try:
        first = verify_m8_l2_session_bundle(authority.bundle_path, expected_config=capture)
    except Exception as error:
        raise M8L2StudyPipelineError(
            f"session {expected_date} {expected_role} failed capture verification"
        ) from error
    if first.session_date != expected_date or first.role != expected_role:
        raise M8L2StudyPipelineError("session authority has the wrong frozen coordinate")
    manifest, _ = _read_json(
        first.manifest_path,
        f"{expected_role} session manifest",
        expected_sha256=authority.manifest_sha256,
    )
    _read_regular(
        first.checksum_path,
        label=f"{expected_role} session checksums",
        expected_sha256=authority.checksums_sha256,
    )
    if first.manifest_sha256 != authority.manifest_sha256:
        raise M8L2StudyPipelineError("session capture verifier and manifest authority disagree")
    campaign = _campaign_from_manifest(manifest)
    if expected_campaign is not None and campaign != expected_campaign:
        raise M8L2StudyPipelineError("four-session campaign/source identity changed")
    try:
        second = verify_m8_l2_session_bundle(authority.bundle_path, expected_config=capture)
    except Exception as error:
        raise M8L2StudyPipelineError("session changed during authority snapshot") from error
    if second != first:
        raise M8L2StudyPipelineError("session verifier result changed during authority snapshot")
    return _SessionSnapshot(authority, first, manifest, campaign)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2StudyPipelineError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise M8L2StudyPipelineError(f"{label} must be nonempty text")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M8L2StudyPipelineError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise M8L2StudyPipelineError(
            f"{label} must be finite" + (" and positive" if positive else "")
        )
    return result


def _endpoint_specs(analysis: M8L2AnalysisConfig) -> tuple[L2EndpointSpec, ...]:
    windows = set(analysis.features.rolling_windows)
    result: list[L2EndpointSpec] = []
    for endpoint in analysis.endpoints:
        impact_window = (
            endpoint.horizon_value
            if endpoint.domain == "event"
            else endpoint.nominal_event_block_width
        )
        if impact_window not in windows:
            impact_window = min(windows, key=lambda item: abs(item - impact_window))
        result.append(
            L2EndpointSpec(
                name=endpoint.name,
                domain=endpoint.domain,
                horizon_value=endpoint.horizon_value,
                horizon_unit=endpoint.unit,
                paired_block_events=(
                    endpoint.paired_block_width if endpoint.domain == "event" else None
                ),
                paired_block_milliseconds=(
                    endpoint.paired_block_width if endpoint.domain == "clock" else None
                ),
                impact_ofi_window=impact_window,
            )
        )
    return tuple(result)


def _campaign_from_aggregate(aggregate: Mapping[str, Any]) -> L2CampaignRuntimeIdentity:
    campaign = _mapping(aggregate.get("campaign_identity"), "development campaign identity")
    return L2CampaignRuntimeIdentity(
        campaign_authority_sha256=_string(
            campaign.get("campaign_authority_sha256"), "development campaign SHA-256"
        ),
        runtime_commit=_string(campaign.get("runtime_commit"), "development runtime commit"),
        runtime_source_tree_sha256=_string(
            campaign.get("runtime_source_tree_sha256"), "development runtime source tree"
        ),
        runtime_fingerprint_sha256=_string(
            campaign.get("runtime_fingerprint_sha256"),
            "development runtime fingerprint",
        ),
        runtime_dirty=campaign.get("runtime_dirty") is not False,
    )


def _assert_current_runtime(campaign: L2CampaignRuntimeIdentity) -> None:
    if current_m8_l2_runtime_fingerprint_sha256() != campaign.runtime_fingerprint_sha256:
        raise M8L2StudyPipelineError(
            "final producer runtime differs from the frozen capture campaign"
        )


def _explicit_development_authority(
    aggregate: Mapping[str, Any],
    train: L2StudySessionAuthority,
    validation: L2StudySessionAuthority,
) -> None:
    raw_inputs = aggregate.get("development_inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 2:
        raise M8L2StudyPipelineError("development lock has no exact two-session input set")
    for claim, supplied, coordinate in zip(
        raw_inputs,
        (train, validation),
        _EXPECTED_COORDINATES[:2],
        strict=True,
    ):
        payload = _mapping(claim, "development input claim")
        file_authority = _mapping(payload.get("file_authority"), "development input file authority")
        expected = {
            "date": coordinate[0],
            "role": coordinate[1],
            "manifest_sha256": supplied.manifest_sha256,
            "checksums_sha256": supplied.checksums_sha256,
        }
        observed = {
            "date": payload.get("date"),
            "role": payload.get("role"),
            "manifest_sha256": file_authority.get("manifest_sha256"),
            "checksums_sha256": file_authority.get("checksums_sha256"),
        }
        if observed != expected:
            raise M8L2StudyPipelineError(
                "caller development session authority differs from the aggregate lock"
            )


def _verify_lock_context(
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    train: L2StudySessionAuthority,
    validation: L2StudySessionAuthority,
    lock_dir: str | Path,
    expected_lock_sha256: str,
) -> tuple[L2DevelopmentLockResult, Mapping[str, Any], L2CampaignRuntimeIdentity, _SourceIdentity]:
    _require_sha256(expected_lock_sha256, "development aggregate lock authority")
    try:
        result = verify_m8_l2_development_lock(
            capture,
            analysis,
            train.bundle_path,
            validation.bundle_path,
            lock_dir,
            expected_lock_sha256=expected_lock_sha256,
        )
    except Exception as error:
        raise M8L2StudyPipelineError("development lock failed recursive verification") from error
    aggregate, _ = _read_json(
        result.aggregate_path,
        "aggregate development lock",
        expected_sha256=expected_lock_sha256,
    )
    _explicit_development_authority(aggregate, train, validation)
    campaign = _campaign_from_aggregate(aggregate)
    source = _current_source_identity(capture)
    producer = _mapping(
        aggregate.get("producer_source_identity"), "development producer source identity"
    )
    expected_source = source.to_dict()
    if dict(producer) != expected_source:
        raise M8L2StudyPipelineError("current producer source differs from development lock")
    if (
        campaign.runtime_commit != source.commit
        or campaign.runtime_source_tree_sha256 != source.source_tree_sha256
        or campaign.runtime_dirty
    ):
        raise M8L2StudyPipelineError("development lock and campaign source identities disagree")
    return result, aggregate, campaign, source


def _regime_from_payload(payload: Mapping[str, Any], *, symbol: str) -> L2RegimeFit:
    if (
        payload.get("schema_version") != "m8-l2-regime-thresholds-v1"
        or payload.get("artifact_kind") != "train_only_l2_regime_thresholds"
        or payload.get("symbol") != symbol
        or payload.get("study_date") != "2026-08-10"
        or payload.get("fit_scope") != "train_session_only"
    ):
        raise M8L2StudyPipelineError("train-only regime snapshot has invalid semantics")
    thresholds = _mapping(payload.get("thresholds"), "regime thresholds")
    return L2RegimeFit(
        symbol=symbol,
        study_date="2026-08-10",
        volatility_column=_string(payload.get("volatility_column"), "regime feature"),
        lower_quantile=_finite(payload.get("lower_quantile"), "lower regime quantile"),
        upper_quantile=_finite(payload.get("upper_quantile"), "upper regime quantile"),
        thresholds=RegimeThresholds(
            volatility_low=_finite(thresholds.get("volatility_low"), "volatility low"),
            volatility_high=_finite(thresholds.get("volatility_high"), "volatility high"),
            spread_tight_bps=_finite(thresholds.get("spread_tight_bps"), "tight spread"),
            spread_wide_bps=_finite(thresholds.get("spread_wide_bps"), "wide spread"),
            depth_low=_finite(thresholds.get("depth_low"), "low depth"),
            depth_high=_finite(thresholds.get("depth_high"), "high depth"),
        ),
    )


def _execution_reference_from_payload(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    aggregate_sha256: str,
) -> L2ExecutionReference:
    if (
        payload.get("schema_version") != "m8-l2-execution-reference-v1"
        or payload.get("artifact_kind") != "train_only_execution_reference"
        or payload.get("symbol") != symbol
        or payload.get("fit_date") != "2026-08-10"
        or payload.get("fit_role") != "train"
        or payload.get("reference_price_statistic") != "train_median_mid_price"
        or payload.get("reference_depth_statistic") != "train_q05_min_bid_ask_l1_depth"
    ):
        raise M8L2StudyPipelineError("train-only execution snapshot has invalid semantics")
    return L2ExecutionReference.create(
        symbol=symbol,
        training_date="2026-08-10",
        reference_mid_price=_finite(
            payload.get("reference_mid_price"), "execution reference midpoint", positive=True
        ),
        train_l1_depth_q05=_finite(
            payload.get("reference_l1_depth_q05"), "execution reference depth", positive=True
        ),
        lot_size=_finite(payload.get("lot_size"), "execution reference lot", positive=True),
        reference_quantity=_finite(
            payload.get("reference_quantity"), "execution reference quantity", positive=True
        ),
        aggregate_lock_sha256=aggregate_sha256,
    )


def _load_lock_material(
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    result: L2DevelopmentLockResult,
    aggregate: Mapping[str, Any],
    campaign: L2CampaignRuntimeIdentity,
    source: _SourceIdentity,
) -> _LockMaterial:
    endpoint_specs = {item.name: item for item in _endpoint_specs(analysis)}
    states: dict[tuple[str, str], LockedL2EndpointState] = {}
    regimes: dict[str, L2RegimeFit] = {}
    references: dict[str, L2ExecutionReference] = {}
    development_hashes: dict[tuple[str, str], str] = {}
    snapshot_files: dict[str, Path] = {
        _relative(result.aggregate_path, result.root): result.aggregate_path,
        "development_lock.sha256": result.root / "development_lock.sha256",
    }
    if result.status == "NOT_CREATED":
        development_checksums_path = result.root / _CHECKSUMS_NAME
        development_checksums = _parse_checksums(
            _read_regular(
                development_checksums_path,
                label="NOT_CREATED development checksum authority",
                maximum_bytes=4 << 20,
            ),
            "NOT_CREATED development checksum authority",
        )
        for relative, digest in development_checksums.items():
            path = _join(result.root, relative)
            if _stable_file_sha256_for_pipeline(path, relative) != digest:
                raise M8L2StudyPipelineError(
                    f"NOT_CREATED development snapshot changed for {relative}"
                )
            snapshot_files[relative] = path
        snapshot_files[_CHECKSUMS_NAME] = development_checksums_path
        not_created_marker = result.root / "_NOT_CREATED"
        if (
            _read_regular(
                not_created_marker,
                label="NOT_CREATED development terminal marker",
                maximum_bytes=32,
            )
            != b"not-created\n"
        ):
            raise M8L2StudyPipelineError("NOT_CREATED development marker differs")
        snapshot_files["_NOT_CREATED"] = not_created_marker
        return _LockMaterial(
            result=result,
            aggregate=aggregate,
            campaign=campaign,
            source=source,
            states={},
            regimes={},
            references={},
            development_frame_sha256={},
            snapshot_files=tuple(snapshot_files[key] for key in sorted(snapshot_files)),
        )
    child_by_key = {(item.symbol, item.endpoint): item for item in result.children}
    expected_keys = tuple(
        (symbol, endpoint.name)
        for symbol in capture.study.symbols
        for endpoint in analysis.endpoints
    )
    if tuple(child_by_key) != expected_keys:
        raise M8L2StudyPipelineError("development result has an incomplete child order")
    for symbol, endpoint_name in expected_keys:
        child_claim = child_by_key[(symbol, endpoint_name)]
        child, _ = _read_json(
            child_claim.path,
            f"{symbol} {endpoint_name} child lock",
            expected_sha256=child_claim.sha256,
        )
        state_relative = _safe_relative(
            _string(child.get("final_fitted_state_path"), "fitted-state path")
        )
        state_sha = _string(child.get("final_fitted_state_sha256"), "fitted-state SHA-256")
        _require_sha256(state_sha, "fitted-state SHA-256")
        state_path = _join(result.root, state_relative)
        state_raw = _read_regular(
            state_path,
            label=f"{symbol} {endpoint_name} fitted state",
        )
        if not state_raw.endswith(b"\n"):
            raise M8L2StudyPipelineError("fitted-state snapshot lacks canonical newline")
        try:
            fitted_state = FinalFittedState.restore(state_raw[:-1].decode("ascii"), state_sha)
        except (UnicodeDecodeError, ValueError) as error:
            raise M8L2StudyPipelineError("fitted-state snapshot cannot be restored") from error

        regime_relative = _safe_relative(
            _string(child.get("regime_thresholds_path"), "regime path")
        )
        regime_sha = _string(child.get("regime_thresholds_sha256"), "regime SHA-256")
        _require_sha256(regime_sha, "regime SHA-256")
        regime_path = _join(result.root, regime_relative)
        if symbol not in regimes:
            regime_payload, _ = _read_json(
                regime_path,
                f"{symbol} regime thresholds",
                expected_sha256=regime_sha,
            )
            regimes[symbol] = _regime_from_payload(regime_payload, symbol=symbol)

        execution_relative = _safe_relative(
            _string(child.get("execution_reference_path"), "execution reference path")
        )
        execution_sha = _string(
            child.get("execution_reference_sha256"), "execution reference SHA-256"
        )
        _require_sha256(execution_sha, "execution reference SHA-256")
        execution_path = _join(result.root, execution_relative)
        if symbol not in references:
            execution_payload, _ = _read_json(
                execution_path,
                f"{symbol} execution reference",
                expected_sha256=execution_sha,
            )
            references[symbol] = _execution_reference_from_payload(
                execution_payload,
                symbol=symbol,
                aggregate_sha256=result.aggregate_sha256,
            )

        states[(symbol, endpoint_name)] = LockedL2EndpointState(
            symbol=symbol,
            endpoint=endpoint_specs[endpoint_name],
            child_lock_sha256=child_claim.sha256,
            aggregate_lock_sha256=result.aggregate_sha256,
            regime_thresholds_sha256=regime_sha,
            fitted_state=fitted_state,
        )
        development_sha = _string(
            child.get("development_frame_sha256"), "development-frame SHA-256"
        )
        _require_sha256(development_sha, "development-frame SHA-256")
        development_hashes[(symbol, endpoint_name)] = development_sha
        selection_relative = _safe_relative(
            _string(child.get("selection_lock_path"), "selection-lock path")
        )
        for path in (
            child_claim.path,
            state_path,
            regime_path,
            execution_path,
            _join(result.root, selection_relative),
        ):
            snapshot_files[_relative(path, result.root)] = path
    development_checksums_path = result.root / _CHECKSUMS_NAME
    development_checksums_raw = _read_regular(
        development_checksums_path,
        label="development-lock checksum authority",
        maximum_bytes=4 << 20,
    )
    development_checksums = _parse_checksums(
        development_checksums_raw, "development-lock checksum authority"
    )
    for relative, digest in development_checksums.items():
        path = _join(result.root, relative)
        if _stable_file_sha256_for_pipeline(path, relative) != digest:
            raise M8L2StudyPipelineError(f"development-lock snapshot source changed for {relative}")
        snapshot_files[relative] = path
    snapshot_files[_CHECKSUMS_NAME] = development_checksums_path
    locked_marker = result.root / "_LOCKED"
    if (
        _read_regular(locked_marker, label="development-lock terminal marker", maximum_bytes=32)
        != b"locked\n"
    ):
        raise M8L2StudyPipelineError("development-lock terminal marker differs")
    snapshot_files["_LOCKED"] = locked_marker
    return _LockMaterial(
        result=result,
        aggregate=aggregate,
        campaign=campaign,
        source=source,
        states=states,
        regimes=regimes,
        references=references,
        development_frame_sha256=development_hashes,
        snapshot_files=tuple(snapshot_files[key] for key in sorted(snapshot_files)),
    )


def _stable_file_sha256_for_pipeline(path: Path, label: str) -> str:
    try:
        before = path.lstat()
    except OSError as error:
        raise M8L2StudyPipelineError(f"cannot stat {label}") from error
    if not stat.S_ISREG(before.st_mode):
        raise M8L2StudyPipelineError(f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_stat(before, opened):
            raise M8L2StudyPipelineError(f"{label} changed before hashing")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if not _same_stat(opened, after) or not _same_stat(after, path.lstat()):
            raise M8L2StudyPipelineError(f"{label} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _verify_all_sessions(
    capture: M8L2StudyConfig,
    material: _LockMaterial,
    authorities: Sequence[L2StudySessionAuthority],
) -> tuple[_SessionSnapshot, ...]:
    if len(authorities) != 4:
        raise M8L2StudyPipelineError("the final L2 study requires four explicit sessions")
    snapshots: list[_SessionSnapshot] = []
    for supplied, (expected_date, expected_role) in zip(
        authorities, _EXPECTED_COORDINATES, strict=True
    ):
        snapshots.append(
            _verify_session_authority(
                supplied,
                capture=capture,
                expected_date=expected_date,
                expected_role=expected_role,
                expected_campaign=material.campaign,
            )
        )
    development_incomplete = any(item.bundle.status != "COMPLETE" for item in snapshots[:2])
    if material.result.status == "LOCKED" and development_incomplete:
        raise M8L2StudyPipelineError(
            "train/validation failure cannot be promoted without a valid development lock"
        )
    if material.result.status == "NOT_CREATED" and not development_incomplete:
        raise M8L2StudyPipelineError(
            "NOT_CREATED development authority requires an insufficient development session"
        )
    return tuple(snapshots)


def _reverify_material(
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    train: L2StudySessionAuthority,
    validation: L2StudySessionAuthority,
    lock_dir: str | Path,
    expected_lock_sha256: str,
    expected: _LockMaterial,
) -> None:
    result, aggregate, campaign, source = _verify_lock_context(
        capture,
        analysis,
        train,
        validation,
        lock_dir,
        expected_lock_sha256,
    )
    if (
        result.aggregate_sha256 != expected.result.aggregate_sha256
        or dict(aggregate) != dict(expected.aggregate)
        or campaign != expected.campaign
        or source != expected.source
        or result.children != expected.result.children
        or result.status != expected.result.status
        or result.reason_codes != expected.result.reason_codes
    ):
        raise M8L2StudyPipelineError("development lock/source changed during final production")


def _development_input(
    snapshot: _SessionSnapshot,
    *,
    capture: M8L2StudyConfig,
    campaign: L2CampaignRuntimeIdentity,
) -> VerifiedL2SessionInput:
    role = cast(Literal["train", "validation"], snapshot.bundle.role)
    return verify_m8_l2_development_input(
        snapshot.authority.bundle_path,
        expected_config=capture,
        expected_date=snapshot.bundle.session_date,
        expected_role=role,
        expected_file_authority=snapshot.authority.file_authority,
        expected_campaign=campaign,
    )


def _heldout_input(
    snapshot: _SessionSnapshot,
    *,
    capture: M8L2StudyConfig,
    campaign: L2CampaignRuntimeIdentity,
    lock_sha256: str,
) -> VerifiedL2SessionInput:
    role = cast(Literal["primary_test", "replication_test"], snapshot.bundle.role)
    return verify_m8_l2_heldout_input(
        snapshot.authority.bundle_path,
        expected_config=capture,
        expected_date=snapshot.bundle.session_date,
        expected_role=role,
        development_lock_sha256=lock_sha256,
        expected_file_authority=snapshot.authority.file_authority,
        expected_campaign=campaign,
    )


def _build_one_symbol_frames(
    verified: VerifiedL2SessionInput,
    *,
    symbol: str,
    analysis: M8L2AnalysisConfig,
    material: _LockMaterial,
) -> Mapping[str, pl.DataFrame]:
    admission = _preflight_symbol_raw(verified, symbol)
    if admission is not None:
        _require_memory_budget(
            admission[0],
            _MAX_FINAL_RAW_BYTES,
            f"{verified.session_date} {symbol} raw Parquet admission",
        )
        _require_memory_budget(
            admission[1] * len(analysis.endpoints) * _CAUSAL_ENDPOINT_ROW_UPPER_BYTES,
            _MAX_CAUSAL_COORDINATE_BYTES,
            f"{verified.session_date} {symbol} causal build admission",
        )
    loaded = verified.load_symbol_frames(symbol)
    _require_memory_budget(
        _loaded_bytes(loaded),
        _MAX_FINAL_RAW_BYTES,
        f"{verified.session_date} {symbol} raw materialization",
    )
    if admission is None:
        _require_memory_budget(
            loaded.book_observations.height
            * len(analysis.endpoints)
            * _CAUSAL_ENDPOINT_ROW_UPPER_BYTES,
            _MAX_CAUSAL_COORDINATE_BYTES,
            f"{verified.session_date} {symbol} causal build admission",
        )
    built = dict(
        build_l2_endpoint_frames(
            loaded.book_observations,
            loaded.depth_deltas,
            loaded.intervals,
            study_date=verified.session_date,
            study_role=cast(Any, verified.role),
            feature_windows=analysis.features.rolling_windows,
            volatility_window=analysis.features.volatility_window,
            clock_max_state_age_ms=analysis.features.clock_max_state_age_ms,
            endpoints=_endpoint_specs(analysis),
        )
    )
    _require_memory_budget(
        _frames_bytes(list(built.values())),
        _MAX_CAUSAL_COORDINATE_BYTES,
        f"{verified.session_date} {symbol} causal builder output",
    )
    del loaded
    result: dict[str, pl.DataFrame] = {}
    for endpoint in analysis.endpoints:
        source = built.pop(endpoint.name)
        frame = apply_l2_regimes(source, material.regimes[symbol])
        _require_memory_budget(
            _frames_bytes([*built.values(), *result.values(), source, frame]),
            _MAX_CAUSAL_COORDINATE_BYTES,
            f"{verified.session_date} {symbol} causal output/scratch",
        )
        if (
            l2_model_feature_columns(frame, windows=analysis.features.rolling_windows)
            != analysis.features.model_feature_columns
        ):
            raise M8L2StudyPipelineError("rebuilt causal frame has the wrong model features")
        validate_l2_endpoint_frame(frame)
        result[endpoint.name] = frame
        del source
    _require_memory_budget(
        _frames_bytes(list(result.values())),
        _MAX_CAUSAL_COORDINATE_BYTES,
        f"{verified.session_date} {symbol} causal coordinate output",
    )
    return result


def _build_causal_frames(
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    snapshots: Sequence[_SessionSnapshot],
    material: _LockMaterial,
    *,
    train: L2StudySessionAuthority,
    validation: L2StudySessionAuthority,
    lock_dir: str | Path,
    lock_sha256: str,
) -> tuple[
    dict[tuple[str, str, str, str], pl.DataFrame],
    tuple[L2HeldoutEndpointFrame, ...],
]:
    inputs: list[VerifiedL2SessionInput] = [
        _development_input(snapshots[0], capture=capture, campaign=material.campaign),
        _development_input(snapshots[1], capture=capture, campaign=material.campaign),
    ]
    for heldout_snapshot in snapshots[2:]:
        _reverify_material(
            capture,
            analysis,
            train,
            validation,
            lock_dir,
            lock_sha256,
            material,
        )
        inputs.append(
            _heldout_input(
                heldout_snapshot,
                capture=capture,
                campaign=material.campaign,
                lock_sha256=lock_sha256,
            )
        )

    causal: dict[tuple[str, str, str, str], pl.DataFrame] = {}
    heldout: list[L2HeldoutEndpointFrame] = []
    causal_bytes = 0
    for verified in inputs:
        for symbol in capture.study.symbols:
            # This check is intentionally adjacent to every held-out payload load.
            if verified.role in {"primary_test", "replication_test"}:
                _reverify_material(
                    capture,
                    analysis,
                    train,
                    validation,
                    lock_dir,
                    lock_sha256,
                    material,
                )
            frames = _build_one_symbol_frames(
                verified,
                symbol=symbol,
                analysis=analysis,
                material=material,
            )
            for endpoint in analysis.endpoints:
                frame = frames[endpoint.name]
                key = (verified.session_date, verified.role, symbol, endpoint.name)
                causal[key] = frame
                causal_bytes += _frame_bytes(frame)
                _require_memory_budget(
                    causal_bytes,
                    _MAX_FINAL_CAUSAL_BYTES,
                    "32-frame accumulated causal output",
                )
                if verified.role in {"primary_test", "replication_test"}:
                    heldout.append(
                        L2HeldoutEndpointFrame(
                            symbol=symbol,
                            endpoint_name=endpoint.name,
                            study_date=verified.session_date,
                            study_role=cast(Any, verified.role),
                            frame=frame,
                        )
                    )
            del frames, frame
    expected_count = 4 * len(capture.study.symbols) * len(analysis.endpoints)
    if len(causal) != expected_count or len(heldout) != expected_count // 2:
        raise M8L2StudyPipelineError("rebuilt causal frame set is incomplete")
    for symbol in capture.study.symbols:
        for endpoint in analysis.endpoints:
            train_frame = causal[("2026-08-10", "train", symbol, endpoint.name)]
            validation_frame = causal[("2026-08-11", "validation", symbol, endpoint.name)]
            _require_memory_budget(
                2 * (_frame_bytes(train_frame) + _frame_bytes(validation_frame)),
                _MAX_CAUSAL_COORDINATE_BYTES,
                f"{symbol} {endpoint.name} development-hash concat scratch",
            )
            development = pl.concat(
                [train_frame, validation_frame],
                how="vertical",
            )
            if (
                _frame_sha256(development)
                != material.development_frame_sha256[(symbol, endpoint.name)]
            ):
                raise M8L2StudyPipelineError(
                    "rebuilt train/validation causal frame differs from the locked development hash"
                )
            del development, train_frame, validation_frame
    return causal, tuple(heldout)


def _heldout_availability_reasons(
    heldout: Sequence[L2HeldoutEndpointFrame],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for item in heldout:
        eligible = item.frame.filter(
            pl.col("feature_ready")
            & (~pl.col("right_censored"))
            & pl.col("future_mid_up").is_not_null()
        )
        if eligible.is_empty():
            reasons.append(
                f"NO_ELIGIBLE_LABELS::{item.study_role}::{item.symbol}::{item.endpoint_name}"
            )
    return tuple(sorted(reasons))


def _session_gate_rows(snapshots: Sequence[_SessionSnapshot]) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for item in snapshots:
        symbols = item.manifest.get("symbols")
        symbol_payload = symbols if isinstance(symbols, Mapping) else {}
        rows.append(
            {
                "study_date": item.bundle.session_date,
                "study_role": item.bundle.role,
                "status": item.bundle.status,
                "BTCUSDT_gate": (
                    _mapping(symbol_payload.get("BTCUSDT"), "BTC session claim").get("status")
                    if "BTCUSDT" in symbol_payload
                    else "NOT_AVAILABLE"
                ),
                "ETHUSDT_gate": (
                    _mapping(symbol_payload.get("ETHUSDT"), "ETH session claim").get("status")
                    if "ETHUSDT" in symbol_payload
                    else "NOT_AVAILABLE"
                ),
                "overlap_seconds": item.manifest.get("cross_symbol_observed_overlap_seconds", 0.0),
                "reason_codes": list(item.bundle.reason_codes),
                "manifest_sha256": item.authority.manifest_sha256,
                "checksums_sha256": item.authority.checksums_sha256,
            }
        )
    return tuple(rows)


def _not_created_final_reasons(
    snapshots: Sequence[_SessionSnapshot],
) -> tuple[str, ...]:
    reasons: set[str] = set()
    for index, item in enumerate(snapshots):
        if item.bundle.status == "COMPLETE":
            continue
        prefix = "DEVELOPMENT_SESSION_INSUFFICIENT" if index < 2 else "HELDOUT_SESSION_INSUFFICIENT"
        reasons.update(
            f"{prefix}::{item.bundle.role}::{reason}"
            for reason in (item.bundle.reason_codes or ("SESSION_INSUFFICIENT_DATA",))
        )
    return tuple(sorted(reasons))


def _causal_relative(key: tuple[str, str, str, str]) -> str:
    study_date, role, symbol, endpoint = key
    return f"causal_frames/{study_date}-{role}/{symbol.lower()}/{endpoint}.parquet"


_EVALUATION_PATHS = (
    "evaluation/predictions.parquet",
    "evaluation/predictive_metrics.parquet",
    "evaluation/paired_by_session_regime.parquet",
    "evaluation/equal_session_summary.parquet",
    "evaluation/signed_markout.parquet",
)
_DESCRIPTIVE_PATHS = (
    "descriptive/intraday_liquidity.parquet",
    "descriptive/ofi_return_association.parquet",
    "descriptive/signal_half_life.parquet",
    "descriptive/liquidity_recovery.parquet",
    "descriptive/regime_diagnostics.parquet",
    "descriptive/feature_stability.parquet",
    "descriptive/cross_instrument_stability.parquet",
)
_REPORT_PATHS = (
    "reports/technical_report.md",
    "reports/executive_memo.md",
    "reports/model_comparison.md",
)


def _execution_relative(item: L2HeldoutEndpointFrame, name: str) -> str:
    return (
        f"execution/partitions/{item.study_date}-{item.study_role}/"
        f"{item.symbol.lower()}/{item.endpoint_name}/{name}.parquet"
    )


def _authority_sources(
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    snapshots: Sequence[_SessionSnapshot],
    material: _LockMaterial,
) -> Mapping[str, tuple[Path, str]]:
    project_root = capture.path.parent.parent.resolve()
    protocol = project_root / "docs" / "M8_L2_PROTOCOL.md"
    result: dict[str, tuple[Path, str]] = {
        "authority/m8_l2_capture_study.toml": (capture.path, capture.source_sha256),
        "authority/m8_l2_analysis.toml": (analysis.path, analysis.source_sha256),
        "authority/M8_L2_PROTOCOL.md": (protocol, M8_L2_PROTOCOL_SHA256),
        "authority/campaign_authority.json": (
            snapshots[0].authority.bundle_path / "authority" / "campaign_authority.json",
            material.campaign.campaign_authority_sha256,
        ),
    }
    for snapshot in snapshots:
        prefix = f"authority/sessions/{snapshot.bundle.session_date}-{snapshot.bundle.role}"
        result[f"{prefix}/session_manifest.json"] = (
            snapshot.bundle.manifest_path,
            snapshot.authority.manifest_sha256,
        )
        result[f"{prefix}/CHECKSUMS.sha256"] = (
            snapshot.bundle.checksum_path,
            snapshot.authority.checksums_sha256,
        )
    for source in material.snapshot_files:
        relative = _relative(source, material.result.root)
        result[f"authority/development_lock/{relative}"] = (source, sha256_file(source))
    return dict(sorted(result.items()))


def _artifact_kind(relative: str) -> str:
    if relative.startswith("authority/"):
        return "authority_snapshot"
    if relative.startswith("causal_frames/"):
        return "causal_endpoint_frame"
    if relative.startswith("evaluation/"):
        return "locked_evaluation"
    if relative.startswith("execution/"):
        return "market_scenario"
    if relative.startswith("descriptive/"):
        return "descriptive_analysis"
    if relative.startswith("reports/"):
        return "human_report"
    if relative == "provenance.json":
        return "provenance"
    if relative.startswith("report_inputs"):
        return "report_authority"
    raise M8L2StudyPipelineError(f"cannot classify run artifact {relative}")


def _planned_paths(
    *,
    authority_paths: Sequence[str],
    causal: Mapping[tuple[str, str, str, str], pl.DataFrame],
    heldout: Sequence[L2HeldoutEndpointFrame],
    complete: bool,
) -> tuple[str, ...]:
    paths = set(authority_paths)
    paths.update(_causal_relative(key) for key in causal)
    if complete:
        paths.update(_EVALUATION_PATHS)
        paths.update(_DESCRIPTIVE_PATHS)
        for item in heldout:
            paths.update(
                _execution_relative(item, name) for name in ("orders", "fills", "positions")
            )
        paths.update(("execution/metrics.parquet", "execution/assumptions.parquet"))
    paths.update(
        {
            "provenance.json",
            "report_inputs.json",
            "report_inputs.sha256",
            *_REPORT_PATHS,
        }
    )
    return tuple(sorted(paths))


def _development_authority_claim(material: _LockMaterial) -> dict[str, object]:
    return {
        "status": material.result.status,
        "authority_sha256": material.result.aggregate_sha256,
        "reason_codes": list(material.result.reason_codes),
    }


def _run_identity(
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    material: _LockMaterial,
    snapshots: Sequence[_SessionSnapshot],
) -> str:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "capture_config_source_sha256": capture.source_sha256,
        "analysis_config_source_sha256": analysis.source_sha256,
        "development_lock_sha256": material.result.aggregate_sha256,
        "development_authority": _development_authority_claim(material),
        "campaign": material.campaign.to_dict(),
        "source": material.source.to_dict(),
        "sessions": [
            {
                "date": item.bundle.session_date,
                "role": item.bundle.role,
                "manifest_sha256": item.authority.manifest_sha256,
                "checksums_sha256": item.authority.checksums_sha256,
            }
            for item in snapshots
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _provenance_payload(
    *,
    status: M8L2StudyRunStatus,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    material: _LockMaterial,
    snapshots: Sequence[_SessionSnapshot],
    generated_at_utc: str,
    run_id: str,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": "m8_l2_final_study_provenance",
        "status": status,
        "generated_at_utc": generated_at_utc,
        "run_id": run_id,
        "git": material.source.to_dict(),
        "runtime": runtime_metadata(),
        "inputs": {
            "capture_config_sha256": capture.hash,
            "capture_config_source_sha256": capture.source_sha256,
            "capture_protocol_sha256": M8_L2_PROTOCOL_SHA256,
            "analysis_config_sha256": analysis.hash,
            "analysis_config_source_sha256": analysis.source_sha256,
            "development_lock_sha256": material.result.aggregate_sha256,
            "development_lock_dir": str(material.result.root),
            "development_authority": _development_authority_claim(material),
            "campaign_identity": material.campaign.to_dict(),
            "sessions": [
                {
                    "date": item.bundle.session_date,
                    "role": item.bundle.role,
                    "external_bundle_path": str(item.authority.bundle_path),
                    "session_id": item.bundle.session_id,
                    "status": item.bundle.status,
                    "manifest_sha256": item.authority.manifest_sha256,
                    "checksums_sha256": item.authority.checksums_sha256,
                }
                for item in snapshots
            ],
        },
        "phase_separation": {
            "development_lock_verified_before_heldout_payload": (
                material.result.status == "LOCKED"
            ),
            "development_authority_status": material.result.status,
            "heldout_economic_payload_accessed": (
                material.result.status == "LOCKED"
                and all(item.bundle.status == "COMPLETE" for item in snapshots[2:])
            ),
            "heldout_fit_or_update_allowed": False,
            "model_updated_between_test_dates": False,
            "directory_discovery_used": False,
        },
        "claims": {
            "p_values": False,
            "significance": False,
            "cross_symbol_pooling": False,
            "capacity": False,
            "realized_execution": False,
            "profitability": False,
        },
    }


def _research_payload(capture: M8L2StudyConfig, analysis: M8L2AnalysisConfig) -> dict[str, object]:
    return {
        "question": (
            "Do frozen book-state models reduce future-mid direction log loss versus a "
            "historical prior on both untouched sessions?"
        ),
        "period_start_utc": capture.sessions[0].start.isoformat().replace("+00:00", "Z"),
        "period_end_utc": capture.sessions[-1].end.isoformat().replace("+00:00", "Z"),
        "symbols": list(capture.study.symbols),
        "endpoint_names": [item.name for item in analysis.endpoints],
    }


def _manifest_payload(
    *,
    status: M8L2StudyRunStatus,
    reason_codes: Sequence[str],
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    material: _LockMaterial,
    snapshots: Sequence[_SessionSnapshot],
    generated_at_utc: str,
    run_id: str,
    artifact_paths: Sequence[str],
    tabular_claims: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    research = _research_payload(capture, analysis)
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": "m8_prospective_live_l2_final_study",
        "status": status,
        "reason_codes": list(reason_codes),
        "generated_at_utc": generated_at_utc,
        "run_id": run_id,
        "evidence_tier": "FULL_DATA",
        "effective_evidence_tier": ("FULL_DATA" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
        "live_trading": False,
        "research": research,
        "authority": {
            "capture_config_sha256": capture.hash,
            "capture_config_source_sha256": capture.source_sha256,
            "capture_protocol_sha256": M8_L2_PROTOCOL_SHA256,
            "analysis_config_sha256": analysis.hash,
            "analysis_config_source_sha256": analysis.source_sha256,
            "development_lock_sha256": material.result.aggregate_sha256,
            "development_authority": _development_authority_claim(material),
            "campaign_identity": material.campaign.to_dict(),
            "producer_source_identity": material.source.to_dict(),
        },
        "sessions": [dict(row) for row in _session_gate_rows(snapshots)],
        "artifacts": [
            {"path": relative, "kind": _artifact_kind(relative)} for relative in artifact_paths
        ],
        "tabular_outputs": [
            {"path": relative, **dict(tabular_claims[relative])}
            for relative in sorted(tabular_claims)
        ],
        "evaluation": {
            "status": "COMPLETE" if status == "COMPLETE" else "NOT_RUN",
            "selection_roles": ["train", "validation"],
            "heldout_roles": ["primary_test", "replication_test"],
            "model_refit_after_development_lock": False,
            "p_values_computed": False,
            "cross_symbol_pooling": False,
        },
        "execution": {
            "status": "SCENARIO_ONLY" if status == "COMPLETE" else "NOT_RUN",
            "market_orders_only": True,
            "live_trading": False,
            "realized_execution": False,
            "capacity_claim_authorized": False,
            "profitability_claim_authorized": False,
        },
        "claims": {
            "p_values": False,
            "significance": False,
            "cross_symbol_pooling": False,
            "capacity": False,
            "realized_execution": False,
            "profitability": False,
        },
        "terminal_marker": {
            "path": _SUCCESS_NAME if status == "COMPLETE" else _INSUFFICIENT_NAME,
            "bytes": "complete\\n" if status == "COMPLETE" else "terminal\\n",
        },
    }


def _normalize_nonfinite(frame: pl.DataFrame) -> pl.DataFrame:
    float_columns = [name for name, dtype in frame.schema.items() if dtype.is_float()]
    if not float_columns:
        return frame
    return frame.with_columns(
        *[
            pl.when(pl.col(name).is_finite().fill_null(False))
            .then(pl.col(name))
            .otherwise(None)
            .alias(name)
            for name in float_columns
        ]
    )


def _require_finite_causal(frame: pl.DataFrame, label: str) -> None:
    for name, dtype in frame.schema.items():
        if (
            dtype.is_float()
            and frame.select((~pl.col(name).is_finite()).fill_null(False).any()).item()
        ):
            raise M8L2StudyPipelineError(f"{label} contains non-finite {name}")


def _tabular_claim(frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "frame_sha256": _frame_sha256(frame),
        "columns": frame.columns,
    }


def _write_claimed_frame(
    stage: Path,
    relative: str,
    frame: pl.DataFrame,
    claims: dict[str, Mapping[str, object]],
    *,
    causal: bool = False,
) -> None:
    output = frame
    if causal:
        _require_finite_causal(output, relative)
    else:
        output = _normalize_nonfinite(output)
    claims[relative] = _tabular_claim(output)
    _write_parquet(_join(stage, relative), output)


def _write_complete_tabular_outputs(
    stage: Path,
    *,
    causal: Mapping[tuple[str, str, str, str], pl.DataFrame],
    heldout: Sequence[L2HeldoutEndpointFrame],
    evaluation: L2EvaluationResult,
    descriptive: L2DescriptiveAnalysis,
    references: Mapping[str, L2ExecutionReference],
) -> Mapping[str, Mapping[str, object]]:
    claims: dict[str, Mapping[str, object]] = {}
    for key in sorted(causal):
        _write_claimed_frame(stage, _causal_relative(key), causal[key], claims, causal=True)
    evaluation_frames = {
        _EVALUATION_PATHS[0]: evaluation.predictions,
        _EVALUATION_PATHS[1]: evaluation.predictive_metrics,
        _EVALUATION_PATHS[2]: evaluation.paired_by_session_regime,
        _EVALUATION_PATHS[3]: evaluation.equal_session_summary,
        _EVALUATION_PATHS[4]: evaluation.signed_markout,
    }
    for relative, frame in evaluation_frames.items():
        _write_claimed_frame(stage, relative, frame, claims)
    descriptive_frames = {
        _DESCRIPTIVE_PATHS[0]: descriptive.intraday_liquidity,
        _DESCRIPTIVE_PATHS[1]: descriptive.ofi_return_association,
        _DESCRIPTIVE_PATHS[2]: descriptive.signal_half_life,
        _DESCRIPTIVE_PATHS[3]: descriptive.liquidity_recovery,
        _DESCRIPTIVE_PATHS[4]: descriptive.regime_diagnostics,
        _DESCRIPTIVE_PATHS[5]: descriptive.feature_stability,
        _DESCRIPTIVE_PATHS[6]: descriptive.cross_instrument_stability,
    }
    for relative, frame in descriptive_frames.items():
        _write_claimed_frame(stage, relative, frame, claims)

    metric_frames: list[pl.DataFrame] = []
    assumption_frames: list[pl.DataFrame] = []
    for item in sorted(
        heldout,
        key=lambda value: (value.study_date, value.symbol, value.endpoint_name),
    ):
        coordinate_predictions = (
            evaluation.predictions.lazy()
            .filter(
                (pl.col("study_date") == item.study_date)
                & (pl.col("symbol") == item.symbol)
                & (pl.col("endpoint_name") == item.endpoint_name)
                & (pl.col("study_role") == item.study_role)
            )
            .select(*_EXECUTION_PREDICTION_COLUMNS)
            .collect()
        )
        _require_memory_budget(
            _execution_workspace_upper_bytes(item.frame, coordinate_predictions),
            _MAX_EXECUTION_WORKSPACE_BYTES,
            (
                f"{item.study_date} {item.symbol} {item.endpoint_name} "
                "execution input/Python-row/ledger admission"
            ),
        )
        projected_events = _project_frame(
            item.frame,
            _EXECUTION_EVENT_COLUMNS,
            "execution coordinate events",
        )
        coordinate_evaluation = L2EvaluationResult(
            predictions=coordinate_predictions,
            predictive_metrics=pl.DataFrame(),
            paired_by_session_regime=pl.DataFrame(),
            equal_session_summary=pl.DataFrame(),
            signed_markout=pl.DataFrame(),
        )
        execution_item = L2HeldoutEndpointFrame(
            symbol=item.symbol,
            endpoint_name=item.endpoint_name,
            study_date=item.study_date,
            study_role=item.study_role,
            frame=projected_events,
        )
        execution = run_locked_l2_market_execution(
            coordinate_evaluation,
            (execution_item,),
            (references[item.symbol],),
        )
        execution_outputs = (
            execution.orders,
            execution.fills,
            execution.positions,
            execution.metrics,
            execution.assumptions,
        )
        _require_memory_budget(
            _frames_bytes([coordinate_predictions, projected_events, *execution_outputs]),
            _MAX_EXECUTION_WORKSPACE_BYTES,
            (
                f"{item.study_date} {item.symbol} {item.endpoint_name} "
                "execution projected inputs and retained ledgers"
            ),
        )
        for name, frame in (
            ("orders", execution.orders),
            ("fills", execution.fills),
            ("positions", execution.positions),
        ):
            _write_claimed_frame(stage, _execution_relative(item, name), frame, claims)
        metric_frames.append(execution.metrics)
        assumption_frames.append(execution.assumptions)
        del (
            coordinate_predictions,
            projected_events,
            coordinate_evaluation,
            execution_item,
            execution_outputs,
            execution,
        )
    _require_memory_budget(
        _frames_bytes([*metric_frames, *assumption_frames]) * 2,
        _MAX_EXECUTION_WORKSPACE_BYTES,
        "execution metric/assumption concat admission",
    )
    metrics = _normalize_nonfinite(pl.concat(metric_frames, how="diagonal_relaxed"))
    assumptions = _normalize_nonfinite(pl.concat(assumption_frames, how="diagonal_relaxed"))
    _write_claimed_frame(stage, "execution/metrics.parquet", metrics, claims)
    _write_claimed_frame(stage, "execution/assumptions.parquet", assumptions, claims)
    return claims


def _write_insufficient_causal_outputs(
    stage: Path,
    causal: Mapping[tuple[str, str, str, str], pl.DataFrame],
) -> Mapping[str, Mapping[str, object]]:
    claims: dict[str, Mapping[str, object]] = {}
    for key in sorted(causal):
        _write_claimed_frame(stage, _causal_relative(key), causal[key], claims, causal=True)
    return claims


def _json_rows(frame: pl.DataFrame) -> tuple[Mapping[str, Any], ...]:
    normalized = _normalize_nonfinite(frame)
    rows = tuple(cast(Mapping[str, Any], row) for row in normalized.to_dicts())
    try:
        json.dumps(rows, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise M8L2StudyPipelineError("report metric rows are not strict finite JSON") from error
    return rows


def _hypothesis_payload(
    status: M8L2StudyRunStatus,
    reason_codes: Sequence[str],
    evaluation: L2EvaluationResult | None,
) -> dict[str, object]:
    if status == "INSUFFICIENT_DATA":
        return {
            "status": "INSUFFICIENT_DATA",
            "conclusion": (
                "The frozen study is INSUFFICIENT_DATA and no held-out predictive or "
                f"execution conclusion is authorized. Reasons: {', '.join(reason_codes)}."
            ),
            "directionally_replicated_pairs": 0,
            "declared_pairs": 8,
        }
    assert evaluation is not None
    overall = evaluation.equal_session_summary.filter(pl.col("regime") == "ALL")
    replicated = overall.filter(pl.col("directionally_replicated")).height
    total = overall.height
    return {
        "status": "DESCRIPTIVE_COMPLETE",
        "conclusion": (
            f"Directional improvement replicated on both untouched sessions for {replicated} "
            f"of {total} symbol-endpoint pairs. This is descriptive, not a significance, "
            "capacity, realized-execution, or profitability claim."
        ),
        "directionally_replicated_pairs": replicated,
        "declared_pairs": total,
    }


def _report_data(
    *,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    snapshots: Sequence[_SessionSnapshot],
    status: M8L2StudyRunStatus,
    reason_codes: Sequence[str],
    evaluation: L2EvaluationResult | None,
    execution_metrics: pl.DataFrame | None,
) -> L2ReportData:
    paired_rows: tuple[Mapping[str, Any], ...] = ()
    equal_rows: tuple[Mapping[str, Any], ...] = ()
    predictive_rows: tuple[Mapping[str, Any], ...] = ()
    if evaluation is not None:
        predictive_rows = _json_rows(evaluation.predictive_metrics)
        paired_rows = tuple(
            {**dict(row), "status": row.get("bootstrap_status")}
            for row in _json_rows(evaluation.paired_by_session_regime)
        )
        equal_rows = tuple(
            {**dict(row), "status": row.get("replication_status")}
            for row in _json_rows(evaluation.equal_session_summary)
        )
    return L2ReportData(
        manifest=manifest,
        provenance=provenance,
        session_gates=_session_gate_rows(snapshots),
        hypothesis=_hypothesis_payload(status, reason_codes, evaluation),
        predictive_metrics=predictive_rows,
        paired_metrics=paired_rows,
        equal_session_metrics=equal_rows,
        execution_metrics=(_json_rows(execution_metrics) if execution_metrics is not None else ()),
    )


def _report_snapshot_payload(data: L2ReportData) -> dict[str, object]:
    return {
        "schema_version": _REPORT_INPUT_SCHEMA_VERSION,
        "artifact_kind": "m8_l2_verified_report_inputs",
        "report_data_sha256": canonical_report_data_sha256(data),
        "data": {
            "manifest": dict(data.manifest),
            "provenance": dict(data.provenance),
            "session_gates": [dict(row) for row in data.session_gates],
            "hypothesis": dict(data.hypothesis),
            "predictive_metrics": [dict(row) for row in data.predictive_metrics],
            "paired_metrics": [dict(row) for row in data.paired_metrics],
            "equal_session_metrics": [dict(row) for row in data.equal_session_metrics],
            "execution_metrics": [dict(row) for row in data.execution_metrics],
        },
    }


def _write_report_artifacts(stage: Path, data: L2ReportData) -> None:
    snapshot = _report_snapshot_payload(data)
    snapshot_sha = _write_json(stage / "report_inputs.json", snapshot)
    _write_bytes(stage / "report_inputs.sha256", f"{snapshot_sha}  report_inputs.json\n".encode())
    _write_bytes(
        stage / "reports" / "technical_report.md",
        render_l2_technical_report(data).encode("utf-8"),
    )
    _write_bytes(
        stage / "reports" / "executive_memo.md",
        render_l2_executive_memo(data).encode("utf-8"),
    )
    _write_bytes(
        stage / "reports" / "model_comparison.md",
        render_l2_model_comparison(data).encode("utf-8"),
    )


def _walk_regular(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise M8L2StudyPipelineError("cannot enumerate final-run inventory") from error
        for entry in entries:
            path = Path(entry.path)
            relative = _relative(path, root)
            if entry.is_symlink():
                raise M8L2StudyPipelineError(f"final-run inventory contains symlink {relative}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                result[relative] = path
            else:
                raise M8L2StudyPipelineError(
                    f"final-run inventory contains non-regular entry {relative}"
                )
    return dict(sorted(result.items()))


def _write_checksum_manifest(stage: Path, expected_paths: Sequence[str]) -> str:
    files = _walk_regular(stage)
    expected = set(expected_paths) | {"run_manifest.json"}
    if set(files) != expected:
        raise M8L2StudyPipelineError(
            "preterminal final-run inventory differs from manifest-declared artifacts "
            f"(missing={sorted(expected - set(files))}, extra={sorted(set(files) - expected)})"
        )
    raw = "".join(f"{sha256_file(path)}  {relative}\n" for relative, path in files.items()).encode(
        "ascii"
    )
    _write_bytes(stage / _CHECKSUMS_NAME, raw)
    return hashlib.sha256(raw).hexdigest()


def _parent_identity(path: Path) -> _ParentIdentity:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise M8L2StudyPipelineError("M8 L2 publication parent is not a directory")
    return _ParentIdentity(metadata.st_dev, metadata.st_ino)


def _reserve_stage(target: Path) -> tuple[Path, _ParentIdentity]:
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    if target.exists() or target.is_symlink():
        raise M8L2StudyPipelineError(
            f"M8 L2 run destination already exists and is not reusable: {target}"
        )
    identity = _parent_identity(target.parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    if _parent_identity(stage.parent) != identity:
        shutil.rmtree(stage, ignore_errors=True)
        raise M8L2StudyPipelineError("M8 L2 publication parent changed during stage reservation")
    _fsync_directory(target.parent)
    return stage, identity


def _atomic_rename_no_replace(stage: Path, target: Path) -> None:
    """Use the platform's exclusive directory rename; never fall back to replace."""

    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(stage)
    destination = os.fsencode(target)
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        operation = library.renameatx_np
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(-2, source, -2, destination, 0x00000004)  # RENAME_EXCL
    elif hasattr(library, "renameat2"):
        operation = library.renameat2
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(-100, source, -100, destination, 1)  # RENAME_NOREPLACE
    else:  # pragma: no cover - supported production platforms expose one primitive
        raise M8L2StudyPipelineError(
            "platform lacks an atomic no-replace directory publication primitive"
        )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise M8L2StudyPipelineError("M8 L2 run destination appeared during atomic publication")
        raise M8L2StudyPipelineError(f"atomic M8 L2 publication failed with errno {observed_errno}")


def _publish_stage_no_overwrite(
    stage: Path, target: Path, expected_parent: _ParentIdentity
) -> None:
    _reject_symlink_components(target.parent)
    if _parent_identity(target.parent) != expected_parent or stage.parent != target.parent:
        raise M8L2StudyPipelineError("M8 L2 publication parent identity changed")
    if target.exists() or target.is_symlink():
        raise M8L2StudyPipelineError("M8 L2 run destination appeared during atomic publication")
    _atomic_rename_no_replace(stage, target)
    if _parent_identity(target.parent) != expected_parent:
        raise M8L2StudyPipelineError("M8 L2 publication parent changed after rename")
    _fsync_directory(target.parent)


def _copy_authorities(stage: Path, sources: Mapping[str, tuple[Path, str]]) -> None:
    for relative, (source, expected_sha256) in sources.items():
        observed = _copy_exact(
            source,
            _join(stage, relative),
            expected_sha256=expected_sha256,
        )
        if observed != expected_sha256:
            raise M8L2StudyPipelineError("authority snapshot copy changed its digest")


def _terminal_revalidation(
    *,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    train: L2StudySessionAuthority,
    validation: L2StudySessionAuthority,
    primary: L2StudySessionAuthority,
    replication: L2StudySessionAuthority,
    lock_dir: str | Path,
    lock_sha256: str,
    material: _LockMaterial,
    snapshots: Sequence[_SessionSnapshot],
    require_current_runtime: bool = False,
) -> None:
    reloaded_capture, reloaded_analysis = _revalidate_configs(capture, analysis)
    if reloaded_capture != capture or reloaded_analysis != analysis:
        raise M8L2StudyPipelineError("frozen configs changed during final production")
    _reverify_material(
        capture,
        analysis,
        train,
        validation,
        lock_dir,
        lock_sha256,
        material,
    )
    repeated = _verify_all_sessions(capture, material, (train, validation, primary, replication))
    if tuple(repeated) != tuple(snapshots):
        raise M8L2StudyPipelineError("session authorities changed during final production")
    if require_current_runtime:
        _assert_current_runtime(material.campaign)
        # This is deliberately last: after marker durability on the second
        # producer call, no authority work remains between this loaded-code
        # origin proof and the exclusive terminal-directory rename.
        _assert_final_producer_import_origins(capture.path.parent.parent.resolve())


def _publish_run(
    *,
    status: M8L2StudyRunStatus,
    reason_codes: Sequence[str],
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    train: L2StudySessionAuthority,
    validation: L2StudySessionAuthority,
    primary: L2StudySessionAuthority,
    replication: L2StudySessionAuthority,
    lock_dir: str | Path,
    lock_sha256: str,
    material: _LockMaterial,
    snapshots: Sequence[_SessionSnapshot],
    causal: Mapping[tuple[str, str, str, str], pl.DataFrame],
    heldout: Sequence[L2HeldoutEndpointFrame],
    evaluation: L2EvaluationResult | None,
    descriptive: L2DescriptiveAnalysis | None,
    run_dir: str | Path,
) -> M8L2StudyRunResult:
    target = Path(run_dir).absolute()
    stage, parent_identity = _reserve_stage(target)
    marker = stage / (_SUCCESS_NAME if status == "COMPLETE" else _INSUFFICIENT_NAME)
    published = False
    try:
        sources = _authority_sources(capture, analysis, snapshots, material)
        _copy_authorities(stage, sources)
        if status == "COMPLETE":
            if evaluation is None or descriptive is None:
                raise M8L2StudyPipelineError("complete final run lacks economic outputs")
            tabular_claims = _write_complete_tabular_outputs(
                stage,
                causal=causal,
                heldout=heldout,
                evaluation=evaluation,
                descriptive=descriptive,
                references=material.references,
            )
            execution_metrics = pl.read_parquet(stage / "execution" / "metrics.parquet")
        else:
            tabular_claims = _write_insufficient_causal_outputs(stage, causal)
            execution_metrics = None
        artifact_paths = _planned_paths(
            authority_paths=tuple(sources),
            causal=causal,
            heldout=heldout,
            complete=status == "COMPLETE",
        )
        generated_at = utc_now_iso()
        run_id = _run_identity(capture, analysis, material, snapshots)
        provenance = _provenance_payload(
            status=status,
            capture=capture,
            analysis=analysis,
            material=material,
            snapshots=snapshots,
            generated_at_utc=generated_at,
            run_id=run_id,
        )
        _write_json(stage / "provenance.json", provenance)
        manifest = _manifest_payload(
            status=status,
            reason_codes=reason_codes,
            capture=capture,
            analysis=analysis,
            material=material,
            snapshots=snapshots,
            generated_at_utc=generated_at,
            run_id=run_id,
            artifact_paths=artifact_paths,
            tabular_claims=tabular_claims,
        )
        _write_json(stage / "run_manifest.json", manifest)
        report_data = _report_data(
            manifest=manifest,
            provenance=provenance,
            snapshots=snapshots,
            status=status,
            reason_codes=reason_codes,
            evaluation=evaluation,
            execution_metrics=execution_metrics,
        )
        _write_report_artifacts(stage, report_data)
        _terminal_revalidation(
            capture=capture,
            analysis=analysis,
            train=train,
            validation=validation,
            primary=primary,
            replication=replication,
            lock_dir=lock_dir,
            lock_sha256=lock_sha256,
            material=material,
            snapshots=snapshots,
            require_current_runtime=True,
        )
        _write_checksum_manifest(stage, artifact_paths)
        _fsync_directory(stage)
        _write_bytes(marker, _SUCCESS_BYTES if status == "COMPLETE" else _INSUFFICIENT_BYTES)
        _fsync_directory(stage)
        # Checksumming and durable terminal staging can be materially slower
        # than the earlier authority check.  Revalidate every external input,
        # lock, config, and source identity again at the actual publication
        # boundary so a transient or sustained drift cannot be renamed into a
        # terminal authority.
        _terminal_revalidation(
            capture=capture,
            analysis=analysis,
            train=train,
            validation=validation,
            primary=primary,
            replication=replication,
            lock_dir=lock_dir,
            lock_sha256=lock_sha256,
            material=material,
            snapshots=snapshots,
            require_current_runtime=True,
        )
        _publish_stage_no_overwrite(stage, target, parent_identity)
        published = True
        return _verify_m8_l2_study_run(
            capture,
            analysis,
            train,
            validation,
            lock_dir,
            lock_sha256,
            primary,
            replication,
            target,
            expected_manifest_sha256=sha256_file(target / "run_manifest.json"),
            expected_checksums_sha256=sha256_file(target / _CHECKSUMS_NAME),
        )
    except BaseException:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)
            _fsync_directory(target.parent)
        raise


def _reproduce_m8_l2_study(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_session: L2StudySessionAuthority,
    validation_session: L2StudySessionAuthority,
    development_lock_dir: str | Path,
    expected_development_lock_sha256: str,
    primary_session: L2StudySessionAuthority,
    replication_session: L2StudySessionAuthority,
    run_dir: str | Path,
    *,
    expected_existing_manifest_sha256: str | None,
    expected_existing_checksums_sha256: str | None,
) -> M8L2StudyRunResult:
    _assert_final_producer_import_origins(capture_config.path.parent.parent.resolve())
    capture, analysis = _revalidate_configs(capture_config, analysis_config)
    target = Path(run_dir).absolute()
    lock_result, aggregate, campaign, source = _verify_lock_context(
        capture,
        analysis,
        train_session,
        validation_session,
        development_lock_dir,
        expected_development_lock_sha256,
    )
    _assert_current_runtime(campaign)
    if (expected_existing_manifest_sha256 is None) != (expected_existing_checksums_sha256 is None):
        raise M8L2StudyPipelineError(
            "existing-run manifest and checksum authorities must be supplied together"
        )
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise M8L2StudyPipelineError("existing M8 L2 run target is not a regular directory")
        terminal = [
            name for name in (_SUCCESS_NAME, _INSUFFICIENT_NAME) if (target / name).exists()
        ]
        if len(terminal) != 1:
            raise M8L2StudyPipelineError(
                "existing M8 L2 target is unterminated or has conflicting terminal markers"
            )
        if expected_existing_manifest_sha256 is None or expected_existing_checksums_sha256 is None:
            raise M8L2StudyPipelineError(
                "existing M8 L2 target requires caller-held manifest and checksum authorities"
            )
        return _verify_m8_l2_study_run(
            capture,
            analysis,
            train_session,
            validation_session,
            development_lock_dir,
            expected_development_lock_sha256,
            primary_session,
            replication_session,
            target,
            expected_manifest_sha256=expected_existing_manifest_sha256,
            expected_checksums_sha256=expected_existing_checksums_sha256,
        )
    if expected_existing_manifest_sha256 is not None:
        raise M8L2StudyPipelineError(
            "existing-run authorities were supplied but the target does not exist"
        )

    material = _load_lock_material(capture, analysis, lock_result, aggregate, campaign, source)
    snapshots = _verify_all_sessions(
        capture,
        material,
        (train_session, validation_session, primary_session, replication_session),
    )
    if getattr(getattr(material, "result", None), "status", "LOCKED") == "NOT_CREATED":
        reasons = _not_created_final_reasons(snapshots)
        if not reasons or tuple(material.result.reason_codes) != tuple(
            reason for reason in reasons if reason.startswith("DEVELOPMENT_")
        ):
            raise M8L2StudyPipelineError(
                "NOT_CREATED development reasons differ from the four-session authority"
            )
        return _publish_run(
            status="INSUFFICIENT_DATA",
            reason_codes=reasons,
            capture=capture,
            analysis=analysis,
            train=train_session,
            validation=validation_session,
            primary=primary_session,
            replication=replication_session,
            lock_dir=development_lock_dir,
            lock_sha256=expected_development_lock_sha256,
            material=material,
            snapshots=snapshots,
            causal={},
            heldout=(),
            evaluation=None,
            descriptive=None,
            run_dir=target,
        )
    heldout_failures = [item for item in snapshots[2:] if item.bundle.status != "COMPLETE"]
    if heldout_failures:
        reasons = tuple(
            sorted(
                {
                    f"{item.bundle.role}::{reason}"
                    for item in heldout_failures
                    for reason in (item.bundle.reason_codes or ("SESSION_INSUFFICIENT_DATA",))
                }
            )
        )
        return _publish_run(
            status="INSUFFICIENT_DATA",
            reason_codes=reasons,
            capture=capture,
            analysis=analysis,
            train=train_session,
            validation=validation_session,
            primary=primary_session,
            replication=replication_session,
            lock_dir=development_lock_dir,
            lock_sha256=expected_development_lock_sha256,
            material=material,
            snapshots=snapshots,
            causal={},
            heldout=(),
            evaluation=None,
            descriptive=None,
            run_dir=target,
        )

    causal, heldout = _build_causal_frames(
        capture,
        analysis,
        snapshots,
        material,
        train=train_session,
        validation=validation_session,
        lock_dir=development_lock_dir,
        lock_sha256=expected_development_lock_sha256,
    )
    availability_reasons = _heldout_availability_reasons(heldout)
    if availability_reasons:
        return _publish_run(
            status="INSUFFICIENT_DATA",
            reason_codes=availability_reasons,
            capture=capture,
            analysis=analysis,
            train=train_session,
            validation=validation_session,
            primary=primary_session,
            replication=replication_session,
            lock_dir=development_lock_dir,
            lock_sha256=expected_development_lock_sha256,
            material=material,
            snapshots=snapshots,
            causal=causal,
            heldout=heldout,
            evaluation=None,
            descriptive=None,
            run_dir=target,
        )
    _require_memory_budget(
        _evaluation_workspace_upper_bytes(
            heldout,
            feature_count=len(analysis.features.model_feature_columns),
        ),
        _MAX_EVALUATION_WORKSPACE_BYTES,
        "held-out evaluation child-frame/concat admission",
    )
    evaluation = evaluate_locked_l2_endpoints(
        tuple(material.states[key] for key in sorted(material.states)),
        heldout,
        bootstrap_samples=analysis.bootstrap.samples,
        seed=analysis.study.seed,
        calibration_bins=analysis.calibration.bins,
    )
    evaluation_frames = (
        evaluation.predictions,
        evaluation.predictive_metrics,
        evaluation.paired_by_session_regime,
        evaluation.equal_session_summary,
        evaluation.signed_markout,
    )
    _require_memory_budget(
        _frames_bytes(evaluation_frames),
        _MAX_EVALUATION_WORKSPACE_BYTES,
        "held-out evaluation retained outputs",
    )

    causal_values = tuple(causal[key] for key in sorted(causal))
    descriptive_columns = _descriptive_projection_columns(analysis.features.model_feature_columns)
    _require_memory_budget(
        _descriptive_workspace_upper_bytes(causal_values, columns=descriptive_columns),
        _MAX_DESCRIPTIVE_WORKSPACE_BYTES,
        "descriptive projection/concat/output admission",
    )
    descriptive_inputs = tuple(
        _project_frame(frame, descriptive_columns, "descriptive endpoint")
        for frame in causal_values
    )
    descriptive = build_l2_descriptive_analysis(
        descriptive_inputs,
        feature_columns=analysis.features.model_feature_columns,
        stability_bins=analysis.calibration.bins,
    )
    descriptive_outputs = (
        descriptive.intraday_liquidity,
        descriptive.ofi_return_association,
        descriptive.signal_half_life,
        descriptive.liquidity_recovery,
        descriptive.regime_diagnostics,
        descriptive.feature_stability,
        descriptive.cross_instrument_stability,
    )
    _require_memory_budget(
        _frames_bytes([*descriptive_inputs, *descriptive_outputs]),
        _MAX_DESCRIPTIVE_WORKSPACE_BYTES,
        "descriptive projected inputs and retained outputs",
    )
    del causal_values, descriptive_inputs, descriptive_outputs, evaluation_frames
    return _publish_run(
        status="COMPLETE",
        reason_codes=(),
        capture=capture,
        analysis=analysis,
        train=train_session,
        validation=validation_session,
        primary=primary_session,
        replication=replication_session,
        lock_dir=development_lock_dir,
        lock_sha256=expected_development_lock_sha256,
        material=material,
        snapshots=snapshots,
        causal=causal,
        heldout=heldout,
        evaluation=evaluation,
        descriptive=descriptive,
        run_dir=target,
    )


def _stable_file_sha256(path: Path, label: str) -> str:
    try:
        before_path = path.lstat()
    except OSError as error:
        raise M8L2StudyRunVerificationError(f"cannot stat {label}") from error
    if not stat.S_ISREG(before_path.st_mode):
        raise M8L2StudyRunVerificationError(f"{label} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise M8L2StudyRunVerificationError(f"cannot securely open {label}") from error
    try:
        before = os.fstat(descriptor)
        if not _same_stat(before_path, before):
            raise M8L2StudyRunVerificationError(f"{label} changed before hashing")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_path = path.lstat()
        if not _same_stat(before, after) or not _same_stat(after, after_path):
            raise M8L2StudyRunVerificationError(f"{label} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _report_mapping_tuple(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise M8L2StudyRunVerificationError(f"{label} must be an array")
    return tuple(_mapping(item, f"{label} entry") for item in value)


def _load_report_data_snapshot(root: Path) -> L2ReportData:
    payload, raw = _read_json(root / "report_inputs.json", "L2 report-input snapshot")
    if set(payload) != {
        "schema_version",
        "artifact_kind",
        "report_data_sha256",
        "data",
    } or (
        payload.get("schema_version") != _REPORT_INPUT_SCHEMA_VERSION
        or payload.get("artifact_kind") != "m8_l2_verified_report_inputs"
    ):
        raise M8L2StudyRunVerificationError("L2 report-input snapshot schema differs")
    sidecar = _read_regular(
        root / "report_inputs.sha256",
        label="L2 report-input digest sidecar",
        maximum_bytes=256,
    )
    snapshot_sha = hashlib.sha256(raw).hexdigest()
    if sidecar != f"{snapshot_sha}  report_inputs.json\n".encode("ascii"):
        raise M8L2StudyRunVerificationError("L2 report-input digest sidecar differs")
    raw_data = _mapping(payload.get("data"), "L2 report inputs")
    if set(raw_data) != {
        "manifest",
        "provenance",
        "session_gates",
        "hypothesis",
        "predictive_metrics",
        "paired_metrics",
        "equal_session_metrics",
        "execution_metrics",
    }:
        raise M8L2StudyRunVerificationError("L2 report-input data keys differ")
    data = L2ReportData(
        manifest=_mapping(raw_data.get("manifest"), "report manifest"),
        provenance=_mapping(raw_data.get("provenance"), "report provenance"),
        session_gates=_report_mapping_tuple(raw_data.get("session_gates"), "session gates"),
        hypothesis=_mapping(raw_data.get("hypothesis"), "report hypothesis"),
        predictive_metrics=_report_mapping_tuple(
            raw_data.get("predictive_metrics"), "predictive metrics"
        ),
        paired_metrics=_report_mapping_tuple(raw_data.get("paired_metrics"), "paired metrics"),
        equal_session_metrics=_report_mapping_tuple(
            raw_data.get("equal_session_metrics"), "equal-session metrics"
        ),
        execution_metrics=_report_mapping_tuple(
            raw_data.get("execution_metrics"), "execution metrics"
        ),
    )
    if payload.get("report_data_sha256") != canonical_report_data_sha256(data):
        raise M8L2StudyRunVerificationError("L2 report inputs differ from their canonical digest")
    return data


def _verify_report_artifacts(
    root: Path,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    data = _load_report_data_snapshot(root)
    if dict(data.manifest) != dict(manifest) or dict(data.provenance) != dict(provenance):
        raise M8L2StudyRunVerificationError(
            "report-input snapshot differs from run manifest/provenance"
        )
    expected = {
        "reports/technical_report.md": render_l2_technical_report(data).encode("utf-8"),
        "reports/executive_memo.md": render_l2_executive_memo(data).encode("utf-8"),
        "reports/model_comparison.md": render_l2_model_comparison(data).encode("utf-8"),
    }
    for relative, raw in expected.items():
        if (
            _read_regular(
                _join(root, relative), label=f"rendered report {relative}", maximum_bytes=32 << 20
            )
            != raw
        ):
            raise M8L2StudyRunVerificationError(
                f"rendered report {relative} differs from verified machine artifacts"
            )


def _tabular_claims_from_manifest(
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    raw = manifest.get("tabular_outputs")
    if not isinstance(raw, list):
        raise M8L2StudyRunVerificationError("run manifest tabular_outputs must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        claim = _mapping(item, "tabular output claim")
        if set(claim) != {"path", "rows", "frame_sha256", "columns"}:
            raise M8L2StudyRunVerificationError("tabular output claim keys differ")
        relative = _safe_relative(_string(claim.get("path"), "tabular output path"))
        rows = claim.get("rows")
        columns = claim.get("columns")
        digest = claim.get("frame_sha256")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 0
            or not isinstance(columns, list)
            or not all(type(value) is str for value in columns)
            or type(digest) is not str
            or not _is_sha256(digest)
            or relative in result
        ):
            raise M8L2StudyRunVerificationError("tabular output claim is malformed")
        result[relative] = claim
    if list(result) != sorted(result):
        raise M8L2StudyRunVerificationError("tabular output claims are not canonically ordered")
    return result


def _read_claimed_parquet(
    root: Path,
    relative: str,
    claim: Mapping[str, Any],
) -> pl.DataFrame:
    try:
        frame = pl.read_parquet(_join(root, relative))
    except (OSError, pl.exceptions.PolarsError) as error:
        raise M8L2StudyRunVerificationError(
            f"cannot restore claimed tabular artifact {relative}"
        ) from error
    if (
        frame.height != claim.get("rows")
        or frame.columns != claim.get("columns")
        or _frame_sha256(frame) != claim.get("frame_sha256")
    ):
        raise M8L2StudyRunVerificationError(
            f"tabular artifact {relative} differs from its semantic claim"
        )
    return frame


def _verify_one_causal_output(
    frame: pl.DataFrame,
    *,
    analysis: M8L2AnalysisConfig,
    expected_key: tuple[str, str, str, str],
) -> None:
    validate_l2_endpoint_frame(frame)
    coordinate = frame.select("study_date", "study_role", "symbol", "endpoint_name").unique()
    if coordinate.height != 1:
        raise M8L2StudyRunVerificationError("causal artifact has multiple coordinates")
    row = coordinate.row(0, named=True)
    observed_key = (
        str(row["study_date"]),
        str(row["study_role"]),
        str(row["symbol"]),
        str(row["endpoint_name"]),
    )
    if observed_key != expected_key:
        raise M8L2StudyRunVerificationError("causal artifact path/coordinate differs")
    if (
        l2_model_feature_columns(frame, windows=analysis.features.rolling_windows)
        != analysis.features.model_feature_columns
    ):
        raise M8L2StudyRunVerificationError("causal artifact feature contract differs")
    _require_finite_causal(frame, _causal_relative(expected_key))


def _verify_partition_frame(
    frame: pl.DataFrame,
    *,
    key: tuple[str, str, str, str],
    family: str,
    aggregate_lock_sha256: str,
) -> None:
    study_date, role, symbol, endpoint = key
    required = {
        "scenario_id",
        "scenario_symbol",
        "study_date",
        "study_role",
        "endpoint_name",
        "decision_latency_events",
        "order_latency_events",
        "child_lock_sha256",
        "aggregate_lock_sha256",
    }
    if not required.issubset(frame.columns):
        raise M8L2StudyRunVerificationError(f"execution {family} partition lacks authority columns")
    if frame.is_empty():
        return
    expected_values: Mapping[str, object] = {
        "scenario_symbol": symbol,
        "study_date": study_date,
        "study_role": role,
        "endpoint_name": endpoint,
        "aggregate_lock_sha256": aggregate_lock_sha256,
    }
    for column, expected in expected_values.items():
        if set(frame.get_column(column).unique().to_list()) != {expected}:
            raise M8L2StudyRunVerificationError(f"execution {family} partition coordinate differs")
    if frame.get_column("scenario_id").n_unique() > 9:
        raise M8L2StudyRunVerificationError(f"execution {family} partition has extra scenarios")
    if (
        family == "orders"
        and "order_type" in frame.columns
        and set(frame.get_column("order_type").drop_nulls().unique()).difference({"market"})
    ):
        raise M8L2StudyRunVerificationError("execution orders contain a non-market order")
    if (
        family == "fills"
        and "liquidity" in frame.columns
        and set(frame.get_column("liquidity").drop_nulls().unique()).difference({"taker"})
    ):
        raise M8L2StudyRunVerificationError("execution fills contain non-taker liquidity")


def _verify_tabular_outputs_streaming(
    root: Path,
    claims: Mapping[str, Mapping[str, Any]],
    *,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    material: _LockMaterial,
    status: M8L2StudyRunStatus,
    reasons: Sequence[str],
) -> Mapping[str, pl.DataFrame]:
    if getattr(material.result, "status", "LOCKED") == "NOT_CREATED" and claims:
        raise M8L2StudyRunVerificationError(
            "NOT_CREATED final run must not contain economic tabular outputs"
        )
    expected_causal_keys = tuple(
        (session.date.isoformat(), session.role, symbol, endpoint.name)
        for session in capture.sessions
        for symbol in capture.study.symbols
        for endpoint in analysis.endpoints
    )
    causal_paths = {path for path in claims if path.startswith("causal_frames/")}
    expected_causal_paths = {_causal_relative(key) for key in expected_causal_keys}
    if status == "COMPLETE" and causal_paths != expected_causal_paths:
        raise M8L2StudyRunVerificationError("complete run lacks all 32 causal frames")
    if status == "INSUFFICIENT_DATA" and causal_paths and causal_paths != expected_causal_paths:
        raise M8L2StudyRunVerificationError("insufficient run has a partial causal-frame set")
    if (
        causal_paths
        and status == "INSUFFICIENT_DATA"
        and not all(reason.startswith("NO_ELIGIBLE_LABELS::") for reason in reasons)
    ):
        raise M8L2StudyRunVerificationError(
            "capture-gate insufficient run must not publish economic frames"
        )

    observed_availability_reasons: list[str] = []
    if causal_paths:
        for symbol in capture.study.symbols:
            for endpoint in analysis.endpoints:
                train_key = ("2026-08-10", "train", symbol, endpoint.name)
                validation_key = ("2026-08-11", "validation", symbol, endpoint.name)
                train_path = _causal_relative(train_key)
                validation_path = _causal_relative(validation_key)
                train_frame = _read_claimed_parquet(root, train_path, claims[train_path])
                _require_verification_memory_budget(
                    _frame_bytes(train_frame),
                    _MAX_CAUSAL_COORDINATE_BYTES,
                    "train causal verification coordinate",
                )
                _verify_one_causal_output(train_frame, analysis=analysis, expected_key=train_key)
                validation_frame = _read_claimed_parquet(
                    root, validation_path, claims[validation_path]
                )
                _require_verification_memory_budget(
                    _frame_bytes(train_frame) + _frame_bytes(validation_frame),
                    _MAX_CAUSAL_COORDINATE_BYTES,
                    "development causal verification pair",
                )
                _verify_one_causal_output(
                    validation_frame, analysis=analysis, expected_key=validation_key
                )
                development = pl.concat([train_frame, validation_frame], how="vertical")
                _require_verification_memory_budget(
                    _frame_bytes(train_frame)
                    + _frame_bytes(validation_frame)
                    + _frame_bytes(development),
                    _MAX_CAUSAL_COORDINATE_BYTES,
                    "development causal verification concat",
                )
                if (
                    _frame_sha256(development)
                    != material.development_frame_sha256[(symbol, endpoint.name)]
                ):
                    raise M8L2StudyRunVerificationError(
                        "published development causal frame differs from child lock"
                    )
                del train_frame, validation_frame, development
                for study_date, role in _EXPECTED_COORDINATES[2:]:
                    key = (study_date, role, symbol, endpoint.name)
                    relative = _causal_relative(key)
                    frame = _read_claimed_parquet(root, relative, claims[relative])
                    _require_verification_memory_budget(
                        _frame_bytes(frame),
                        _MAX_CAUSAL_COORDINATE_BYTES,
                        "held-out causal verification coordinate",
                    )
                    _verify_one_causal_output(frame, analysis=analysis, expected_key=key)
                    eligible = frame.filter(
                        pl.col("feature_ready")
                        & (~pl.col("right_censored"))
                        & pl.col("future_mid_up").is_not_null()
                    )
                    if eligible.is_empty():
                        observed_availability_reasons.append(
                            f"NO_ELIGIBLE_LABELS::{role}::{symbol}::{endpoint.name}"
                        )
                    del frame, eligible
    if status == "COMPLETE" and observed_availability_reasons:
        raise M8L2StudyRunVerificationError("complete run contains an empty held-out endpoint")
    if (
        status == "INSUFFICIENT_DATA"
        and causal_paths
        and tuple(sorted(observed_availability_reasons)) != tuple(reasons)
    ):
        raise M8L2StudyRunVerificationError(
            "insufficient reasons differ from published causal availability"
        )

    partition_paths = {path for path in claims if path.startswith("execution/partitions/")}
    expected_partition_paths: dict[str, tuple[tuple[str, str, str, str], str]] = {}
    if status == "COMPLETE":
        for study_date, role in _EXPECTED_COORDINATES[2:]:
            for symbol in capture.study.symbols:
                for endpoint in analysis.endpoints:
                    key = (study_date, role, symbol, endpoint.name)
                    for family in ("orders", "fills", "positions"):
                        relative = (
                            f"execution/partitions/{study_date}-{role}/{symbol.lower()}/"
                            f"{endpoint.name}/{family}.parquet"
                        )
                        expected_partition_paths[relative] = (key, family)
    if partition_paths != set(expected_partition_paths):
        raise M8L2StudyRunVerificationError("execution partition inventory differs")
    for relative in sorted(partition_paths):
        frame = _read_claimed_parquet(root, relative, claims[relative])
        _require_verification_memory_budget(
            _frame_bytes(frame),
            _MAX_EXECUTION_WORKSPACE_BYTES,
            "execution partition verification coordinate",
        )
        key, family = expected_partition_paths[relative]
        _verify_partition_frame(
            frame,
            key=key,
            family=family,
            aggregate_lock_sha256=material.result.aggregate_sha256,
        )
        del frame

    retained_paths = set(claims) - causal_paths - partition_paths
    retained: dict[str, pl.DataFrame] = {}
    retained_bytes = {"evaluation": 0, "descriptive": 0, "execution": 0}
    for relative in sorted(retained_paths):
        frame = _read_claimed_parquet(root, relative, claims[relative])
        category = relative.split("/", 1)[0]
        if category not in retained_bytes:
            raise M8L2StudyRunVerificationError(f"unclassified retained tabular output {relative}")
        retained_bytes[category] += _frame_bytes(frame)
        maximum = {
            "evaluation": _MAX_EVALUATION_WORKSPACE_BYTES,
            "descriptive": _MAX_DESCRIPTIVE_WORKSPACE_BYTES,
            "execution": _MAX_EXECUTION_WORKSPACE_BYTES,
        }[category]
        _require_verification_memory_budget(
            retained_bytes[category], maximum, f"retained {category} verification outputs"
        )
        retained[relative] = frame
    return retained


def _all_false(frame: pl.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        if (
            column not in frame.columns
            or frame.get_column(column).null_count()
            or bool(frame.get_column(column).any())
        ):
            raise M8L2StudyRunVerificationError(f"{label} does not keep {column}=false")


def _require_nullable_unit_interval_columns(
    frame: pl.DataFrame, columns: Sequence[str], label: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise M8L2StudyRunVerificationError(f"{label} lacks ratio columns: {missing}")
    for column in columns:
        dtype = frame.schema[column]
        if not (dtype.is_float() or dtype.is_integer()):
            raise M8L2StudyRunVerificationError(f"{label} {column} is not numeric")
        values = frame.get_column(column).drop_nulls().to_list()
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in values
        ):
            raise M8L2StudyRunVerificationError(
                f"{label} {column} must be null or finite in [0, 1]"
            )


def _verify_complete_semantics(
    frames: Mapping[str, pl.DataFrame],
    *,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    material: _LockMaterial,
) -> None:
    expected_noncausal = (
        set(_EVALUATION_PATHS)
        | set(_DESCRIPTIVE_PATHS)
        | {
            "execution/metrics.parquet",
            "execution/assumptions.parquet",
        }
    )
    if not expected_noncausal.issubset(frames):
        raise M8L2StudyRunVerificationError("complete run lacks declared result tables")
    predictions = frames[_EVALUATION_PATHS[0]]
    required_prediction_columns = {
        "sample_id",
        "symbol",
        "study_date",
        "study_role",
        "endpoint_name",
        "is_oos",
        "split",
        "child_lock_sha256",
        "aggregate_lock_sha256",
        "test_used_for_selection",
        "model_updated_between_test_dates",
        "p_value_computed",
        "significance_claim_authorized",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise M8L2StudyRunVerificationError("prediction artifact lacks lock/OOS boundaries")
    if (
        predictions.is_empty()
        or predictions.get_column("sample_id").n_unique() != predictions.height
        or set(predictions.get_column("study_role").unique())
        != {"primary_test", "replication_test"}
        or set(predictions.get_column("split").unique()) != {"final_test"}
        or not bool(predictions.get_column("is_oos").all())
        or set(predictions.get_column("aggregate_lock_sha256").unique())
        != {material.result.aggregate_sha256}
    ):
        raise M8L2StudyRunVerificationError("prediction artifact is not exact held-out OOS data")
    _all_false(
        predictions,
        (
            "test_used_for_selection",
            "model_updated_between_test_dates",
            "p_value_computed",
            "significance_claim_authorized",
        ),
        "prediction artifact",
    )
    for relative in (_EVALUATION_PATHS[1], _EVALUATION_PATHS[2], _EVALUATION_PATHS[3]):
        frame = frames[relative]
        _all_false(
            frame,
            ("p_value_computed", "significance_claim_authorized"),
            relative,
        )
        if "cross_symbol_pooling" in frame.columns:
            _all_false(frame, ("cross_symbol_pooling",), relative)
    metrics = frames["execution/metrics.parquet"]
    assumptions = frames["execution/assumptions.parquet"]
    expected_scenarios = 2 * len(capture.study.symbols) * len(analysis.endpoints) * 3 * 3
    if metrics.height != expected_scenarios or assumptions.height != expected_scenarios:
        raise M8L2StudyRunVerificationError("execution scenario grid is incomplete")
    if metrics.get_column("scenario_id").n_unique() != expected_scenarios:
        raise M8L2StudyRunVerificationError("execution scenario identities collide")
    _require_nullable_unit_interval_columns(
        metrics,
        ("fill_ratio", "fill_ratio_requested", "partial_fill_order_ratio"),
        "execution metrics",
    )
    _all_false(
        metrics,
        (
            "capacity_claim_authorized",
            "realized_execution_claim_authorized",
            "profitability_claim_authorized",
        ),
        "execution metrics",
    )
    _all_false(
        assumptions,
        (
            "live_trading",
            "capacity_claim_authorized",
            "realized_execution_claim_authorized",
            "profitability_claim_authorized",
        ),
        "execution assumptions",
    )
    if set(metrics.get_column("order_type").unique()) != {"market"} or not bool(
        assumptions.get_column("market_orders_only").all()
    ):
        raise M8L2StudyRunVerificationError("execution output is not market-only")


def _artifact_paths_from_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise M8L2StudyRunVerificationError("run manifest artifacts must be an array")
    result: list[str] = []
    for item in raw:
        claim = _mapping(item, "run artifact claim")
        if set(claim) != {"path", "kind"}:
            raise M8L2StudyRunVerificationError("run artifact claim keys differ")
        relative = _safe_relative(_string(claim.get("path"), "run artifact path"))
        if claim.get("kind") != _artifact_kind(relative):
            raise M8L2StudyRunVerificationError("run artifact kind differs from its path")
        result.append(relative)
    if result != sorted(set(result)):
        raise M8L2StudyRunVerificationError("run artifact paths are duplicate or unordered")
    return tuple(result)


def _reverify_internal_terminal_snapshot(
    root: Path,
    *,
    root_identity: _ParentIdentity,
    initial_files: Mapping[str, Path],
    terminal_name: str,
    terminal_bytes: bytes,
    checksums_raw: bytes,
    checksums: Mapping[str, str],
) -> None:
    """Rebind every internal byte after semantic and external verification.

    The first pass supports semantic parsing.  This second stable-descriptor pass
    occurs at the return boundary so an artifact changed after its semantic read
    cannot inherit the earlier verification result.
    """

    try:
        _reject_symlink_components(root)
        before = root.lstat()
    except (M8L2StudyPipelineError, OSError) as error:
        raise M8L2StudyRunVerificationError(
            "final M8 L2 run path changed before return-boundary verification"
        ) from error
    if (
        not stat.S_ISDIR(before.st_mode)
        or _ParentIdentity(before.st_dev, before.st_ino) != root_identity
    ):
        raise M8L2StudyRunVerificationError(
            "final M8 L2 run directory identity changed before return"
        )
    observed_files = _walk_regular(root)
    if set(observed_files) != set(initial_files):
        raise M8L2StudyRunVerificationError(
            "final-run inventory changed after semantic verification"
        )
    if (
        _read_regular(
            root / terminal_name,
            label="return-boundary terminal marker",
            maximum_bytes=32,
        )
        != terminal_bytes
    ):
        raise M8L2StudyRunVerificationError(
            "final-run terminal marker changed after semantic verification"
        )
    if (
        _read_regular(
            root / _CHECKSUMS_NAME,
            label="return-boundary checksums",
            maximum_bytes=16 << 20,
        )
        != checksums_raw
    ):
        raise M8L2StudyRunVerificationError(
            "final-run checksums changed after semantic verification"
        )
    for relative, expected_digest in checksums.items():
        if _stable_file_sha256(_join(root, relative), f"return-boundary {relative}") != (
            expected_digest
        ):
            raise M8L2StudyRunVerificationError(
                f"final-run artifact changed after semantic verification: {relative}"
            )
    try:
        after = root.lstat()
    except OSError as error:
        raise M8L2StudyRunVerificationError(
            "final M8 L2 run path disappeared before verification returned"
        ) from error
    if (
        not stat.S_ISDIR(after.st_mode)
        or _ParentIdentity(after.st_dev, after.st_ino) != root_identity
    ):
        raise M8L2StudyRunVerificationError(
            "final M8 L2 run directory identity changed at verification return"
        )


def _verify_m8_l2_study_run(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_session: L2StudySessionAuthority,
    validation_session: L2StudySessionAuthority,
    development_lock_dir: str | Path,
    expected_development_lock_sha256: str,
    primary_session: L2StudySessionAuthority,
    replication_session: L2StudySessionAuthority,
    run_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_checksums_sha256: str | None = None,
) -> M8L2StudyRunResult:
    capture, analysis = _revalidate_configs(capture_config, analysis_config)
    if expected_manifest_sha256 is not None:
        _require_sha256(expected_manifest_sha256, "expected final-run manifest")
    if expected_checksums_sha256 is not None:
        _require_sha256(expected_checksums_sha256, "expected final-run checksums")
    lock_result, aggregate, campaign, source = _verify_lock_context(
        capture,
        analysis,
        train_session,
        validation_session,
        development_lock_dir,
        expected_development_lock_sha256,
    )
    material = _load_lock_material(capture, analysis, lock_result, aggregate, campaign, source)
    snapshots = _verify_all_sessions(
        capture,
        material,
        (train_session, validation_session, primary_session, replication_session),
    )
    for snapshot in snapshots[2:]:
        if material.result.status == "LOCKED" and snapshot.bundle.status == "COMPLETE":
            _reverify_material(
                capture,
                analysis,
                train_session,
                validation_session,
                development_lock_dir,
                expected_development_lock_sha256,
                material,
            )
            _heldout_input(
                snapshot,
                capture=capture,
                campaign=material.campaign,
                lock_sha256=expected_development_lock_sha256,
            )

    root = Path(run_dir).absolute()
    try:
        _reject_symlink_components(root)
    except M8L2StudyPipelineError as error:
        raise M8L2StudyRunVerificationError(
            "final M8 L2 run path contains an unsafe component"
        ) from error
    try:
        metadata = root.lstat()
    except OSError as error:
        raise M8L2StudyRunVerificationError("final M8 L2 run directory is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise M8L2StudyRunVerificationError("final M8 L2 run must be a regular directory")
    root_identity = _ParentIdentity(metadata.st_dev, metadata.st_ino)
    files = _walk_regular(root)
    terminal_names = [name for name in (_SUCCESS_NAME, _INSUFFICIENT_NAME) if name in files]
    if len(terminal_names) != 1:
        raise M8L2StudyRunVerificationError("final M8 L2 run requires exactly one terminal marker")
    terminal_name = terminal_names[0]
    marker_path = root / terminal_name
    expected_marker_bytes = (
        _SUCCESS_BYTES if terminal_name == _SUCCESS_NAME else _INSUFFICIENT_BYTES
    )
    if (
        _read_regular(marker_path, label="final M8 L2 terminal marker", maximum_bytes=32)
        != expected_marker_bytes
    ):
        raise M8L2StudyRunVerificationError("final M8 L2 terminal marker bytes differ")
    if _CHECKSUMS_NAME not in files or "run_manifest.json" not in files:
        raise M8L2StudyRunVerificationError("final M8 L2 run lacks control authorities")
    checksums_raw = _read_regular(
        root / _CHECKSUMS_NAME,
        label="final M8 L2 checksums",
        maximum_bytes=16 << 20,
    )
    checksums_sha = hashlib.sha256(checksums_raw).hexdigest()
    if expected_checksums_sha256 is not None and checksums_sha != expected_checksums_sha256:
        raise M8L2StudyRunVerificationError("final-run checksums differ from caller authority")
    checksums = _parse_checksums(checksums_raw, "final M8 L2 checksums")
    expected_inventory = set(checksums) | {_CHECKSUMS_NAME, terminal_name}
    if set(files) != expected_inventory:
        raise M8L2StudyRunVerificationError(
            "final-run physical inventory differs from checksums "
            f"(missing={sorted(expected_inventory - set(files))}, "
            f"extra={sorted(set(files) - expected_inventory)})"
        )
    for relative, expected_digest in checksums.items():
        if _stable_file_sha256(_join(root, relative), relative) != expected_digest:
            raise M8L2StudyRunVerificationError(f"final-run checksum mismatch for {relative}")

    manifest, manifest_raw = _read_json(root / "run_manifest.json", "final-run manifest")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise M8L2StudyRunVerificationError("final-run manifest differs from caller authority")
    if checksums.get("run_manifest.json") != manifest_sha:
        raise M8L2StudyRunVerificationError("final-run checksums do not bind the manifest")
    expected_manifest_keys = {
        "schema_version",
        "artifact_kind",
        "status",
        "reason_codes",
        "generated_at_utc",
        "run_id",
        "evidence_tier",
        "effective_evidence_tier",
        "live_trading",
        "research",
        "authority",
        "sessions",
        "artifacts",
        "tabular_outputs",
        "evaluation",
        "execution",
        "claims",
        "terminal_marker",
    }
    if set(manifest) != expected_manifest_keys:
        raise M8L2StudyRunVerificationError("final-run manifest keys differ")
    status_raw = manifest.get("status")
    if status_raw not in {"COMPLETE", "INSUFFICIENT_DATA"}:
        raise M8L2StudyRunVerificationError("final-run status is unsupported")
    status = cast(M8L2StudyRunStatus, status_raw)
    expected_terminal = _SUCCESS_NAME if status == "COMPLETE" else _INSUFFICIENT_NAME
    expected_terminal_claim = {
        "path": expected_terminal,
        "bytes": "complete\\n" if status == "COMPLETE" else "terminal\\n",
    }
    if (
        terminal_name != expected_terminal
        or manifest.get("terminal_marker") != expected_terminal_claim
    ):
        raise M8L2StudyRunVerificationError("final-run status and terminal marker disagree")
    raw_reasons = manifest.get("reason_codes")
    if not isinstance(raw_reasons, list) or not all(type(item) is str for item in raw_reasons):
        raise M8L2StudyRunVerificationError("final-run reason codes must be strings")
    reasons = tuple(cast(list[str], raw_reasons))
    if list(reasons) != sorted(set(reasons)) or (status == "COMPLETE") == bool(reasons):
        raise M8L2StudyRunVerificationError("final-run reason/status boundary is inconsistent")
    if (
        manifest.get("schema_version") != _SCHEMA_VERSION
        or manifest.get("artifact_kind") != "m8_prospective_live_l2_final_study"
        or manifest.get("evidence_tier") != "FULL_DATA"
        or manifest.get("effective_evidence_tier")
        != ("FULL_DATA" if status == "COMPLETE" else "INSUFFICIENT_DATA")
        or manifest.get("live_trading") is not False
    ):
        raise M8L2StudyRunVerificationError("final-run evidence boundary differs")

    run_id = _run_identity(capture, analysis, material, snapshots)
    if manifest.get("run_id") != run_id:
        raise M8L2StudyRunVerificationError("final-run deterministic identity differs")
    authority = _mapping(manifest.get("authority"), "final-run authority")
    expected_authority = {
        "capture_config_sha256": capture.hash,
        "capture_config_source_sha256": capture.source_sha256,
        "capture_protocol_sha256": M8_L2_PROTOCOL_SHA256,
        "analysis_config_sha256": analysis.hash,
        "analysis_config_source_sha256": analysis.source_sha256,
        "development_lock_sha256": material.result.aggregate_sha256,
        "development_authority": _development_authority_claim(material),
        "campaign_identity": material.campaign.to_dict(),
        "producer_source_identity": material.source.to_dict(),
    }
    if dict(authority) != expected_authority:
        raise M8L2StudyRunVerificationError("final-run authority differs from external evidence")
    if manifest.get("sessions") != [dict(row) for row in _session_gate_rows(snapshots)]:
        raise M8L2StudyRunVerificationError("final-run session claims differ from external bundles")
    if status == "COMPLETE" and any(item.bundle.status != "COMPLETE" for item in snapshots):
        raise M8L2StudyRunVerificationError("complete run contains an insufficient session")
    if status == "INSUFFICIENT_DATA":
        if material.result.status == "NOT_CREATED":
            if reasons != _not_created_final_reasons(snapshots):
                raise M8L2StudyRunVerificationError(
                    "NOT_CREATED final reasons differ from all session authorities"
                )
        else:
            session_reasons = {
                f"{item.bundle.role}::{reason}"
                for item in snapshots[2:]
                if item.bundle.status != "COMPLETE"
                for reason in (item.bundle.reason_codes or ("SESSION_INSUFFICIENT_DATA",))
            }
            if session_reasons:
                if set(reasons) != session_reasons:
                    raise M8L2StudyRunVerificationError(
                        "insufficient reasons differ from held-out capture gates"
                    )
            elif not all(reason.startswith("NO_ELIGIBLE_LABELS::") for reason in reasons):
                raise M8L2StudyRunVerificationError(
                    "insufficient final run lacks a typed data-availability boundary"
                )

    provenance, provenance_raw = _read_json(root / "provenance.json", "final-run provenance")
    if _canonical_json_bytes(cast(Mapping[str, object], provenance)) != provenance_raw:
        raise M8L2StudyRunVerificationError("final-run provenance is not canonical JSON")
    generated_at = _string(manifest.get("generated_at_utc"), "manifest generation time")
    expected_provenance = _provenance_payload(
        status=status,
        capture=capture,
        analysis=analysis,
        material=material,
        snapshots=snapshots,
        generated_at_utc=generated_at,
        run_id=run_id,
    )
    if provenance != expected_provenance:
        raise M8L2StudyRunVerificationError("final-run provenance differs from exact authorities")

    artifact_paths = _artifact_paths_from_manifest(manifest)
    if set(artifact_paths) != set(checksums) - {"run_manifest.json"}:
        raise M8L2StudyRunVerificationError(
            "manifest-declared artifacts differ from checksum inventory"
        )
    authority_sources = _authority_sources(capture, analysis, snapshots, material)
    if set(authority_sources) != {path for path in artifact_paths if path.startswith("authority/")}:
        raise M8L2StudyRunVerificationError("self-contained authority snapshot set differs")
    for relative, (external, expected_digest) in authority_sources.items():
        if (
            _stable_file_sha256(external, f"external {relative}") != expected_digest
            or _stable_file_sha256(_join(root, relative), relative) != expected_digest
        ):
            raise M8L2StudyRunVerificationError(
                f"self-contained authority snapshot changed for {relative}"
            )

    tabular_claims = _tabular_claims_from_manifest(manifest)
    expected_parquets = {
        path
        for path in artifact_paths
        if path.endswith(".parquet") and not path.startswith("authority/")
    }
    if set(tabular_claims) != expected_parquets:
        raise M8L2StudyRunVerificationError(
            "tabular semantic claims differ from declared Parquet artifacts"
        )
    frames = _verify_tabular_outputs_streaming(
        root,
        tabular_claims,
        capture=capture,
        analysis=analysis,
        material=material,
        status=status,
        reasons=reasons,
    )
    if status == "COMPLETE":
        _verify_complete_semantics(
            frames,
            capture=capture,
            analysis=analysis,
            material=material,
        )
    elif any(
        path.startswith(("evaluation/", "execution/", "descriptive/")) for path in tabular_claims
    ):
        raise M8L2StudyRunVerificationError(
            "insufficient final run improperly contains promoted economic results"
        )
    _verify_report_artifacts(root, manifest, provenance)
    report_data = _load_report_data_snapshot(root)
    if tuple(dict(row) for row in report_data.session_gates) != tuple(
        dict(row) for row in _session_gate_rows(snapshots)
    ):
        raise M8L2StudyRunVerificationError("report session gates differ from external evidence")
    _terminal_revalidation(
        capture=capture,
        analysis=analysis,
        train=train_session,
        validation=validation_session,
        primary=primary_session,
        replication=replication_session,
        lock_dir=development_lock_dir,
        lock_sha256=expected_development_lock_sha256,
        material=material,
        snapshots=snapshots,
    )
    _reverify_internal_terminal_snapshot(
        root,
        root_identity=root_identity,
        initial_files=files,
        terminal_name=terminal_name,
        terminal_bytes=expected_marker_bytes,
        checksums_raw=checksums_raw,
        checksums=checksums,
    )
    return M8L2StudyRunResult(
        root=root,
        status=status,
        manifest_path=root / "run_manifest.json",
        manifest_sha256=manifest_sha,
        checksum_path=root / _CHECKSUMS_NAME,
        checksum_sha256=checksums_sha,
        marker_path=marker_path,
        reason_codes=reasons,
    )


__all__ = [
    "L2StudySessionAuthority",
    "M8L2StudyPipelineError",
    "M8L2StudyRunResult",
    "M8L2StudyRunStatus",
    "M8L2StudyRunVerificationError",
    "load_m8_l2_report_data",
    "reproduce_m8_l2_study",
    "verify_m8_l2_study_run",
]
