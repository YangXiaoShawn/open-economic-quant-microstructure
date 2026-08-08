"""Atomic producer for the frozen M8 multi-date trade-only study.

The orchestration in this module is deliberately stricter than the exploratory
public-sample producer.  It verifies one explicitly named, content-addressed
input manifest before reading normalized rows, builds every development date
for both instruments, persists a canonical selection lock, and only then opens
the primary and replication test dates.  The resulting ``FULL_DATA`` label is
scoped to the eight complete trade archives; execution, fills, P&L, capacity,
and statistical-significance claims remain unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from microstructure.config import FeatureConfig, ModelConfig
from microstructure.m8_acquisition import (
    M8AcquisitionError,
    M8AcquisitionManifest,
    M8RawSymbolMetadata,
    copy_m8_acquisition_into,
    read_m8_acquisition_manifest,
)
from microstructure.m8_config import M8PeriodRole, M8StudyConfig
from microstructure.m8_manifest import (
    M8ArchiveEntry,
    M8InputManifest,
    M8ManifestError,
    M8SymbolMetadata,
    verify_m8_input_manifest,
    write_m8_input_manifest,
)
from microstructure.m8_normalization import (
    M8InsufficientDataError,
    M8NormalizationEvidenceCompletion,
    M8NormalizationFailureKind,
    normalize_m8_archive,
)
from microstructure.provenance import (
    git_source_tree_sha256,
    git_state,
    runtime_metadata,
    sha256_file,
    utc_now_iso,
    write_json,
)
from microstructure.reporting import (
    load_run_bundle,
    render_executive_memo,
    render_model_comparison_report,
    render_technical_report,
    verify_checksums,
    write_checksum_manifest,
)
from microstructure.research.analysis import DescriptiveAnalysisError, feature_stability_summary
from microstructure.research.features import ResearchDataError, TemporalLeakageError
from microstructure.research.models import ModelEvaluationError, classification_metrics
from microstructure.research.multidate import (
    AnalysisLock,
    FinalFittedState,
    LockedMultiDateTestResult,
    LockedSelection,
    MultiDateEvaluationError,
    evaluate_locked_multidate_tests,
    select_multidate_model,
)
from microstructure.research.trade_only import (
    build_trade_only_research_frame,
    validate_trade_only_temporal_contract,
)

M8_PIPELINE_SCHEMA_VERSION = "1.0.0"
M8_EVIDENCE_TIER = "FULL_DATA"
M8_EVIDENCE_SCOPE = "trade_only_complete_predeclared_daily_archives"
M8_EXECUTION_EXCLUSION_REASON = (
    "NOT_RUN: aggregate-trade archives contain no contemporaneous bid/ask, depth, "
    "cancellation, queue, or local receipt-time state. Execution, fills, fees-to-alpha "
    "conversion, P&L, capacity, and profitability are outside this trade-only study."
)
M8_NO_SIGNIFICANCE_CAVEAT = (
    "The seeded paired block intervals are descriptive dependence diagnostics. No p-values "
    "are computed, H0 is not rejected, and no statistical-significance claim is authorized."
)
M8_NO_POOLING_CAVEAT = (
    "BTCUSDT and ETHUSDT are evaluated and reported separately; no cross-instrument pooling "
    "or persistent-alpha conclusion is authorized."
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ANALYSIS_LOCK_SCHEMA_VERSION = "m8-analysis-lock-v2"
_MAX_LOCK_JSON_BYTES = 8 * 1024 * 1024
_MAX_LOCK_DIGEST_BYTES = 512
_MAX_FAILURE_INVENTORY_BYTES = 16 * 1024 * 1024
_MAX_FAILURE_INVENTORY_ENTRIES = 50_000
_MAX_FAILURE_FINDING_LINE_BYTES = 1024 * 1024
_LOCK_READ_CHUNK_BYTES = 64 * 1024
_DEVELOPMENT_ROLES = frozenset({"train", "validation"})
_TEST_ROLES = frozenset({"primary_test", "replication_test"})
_TRADE_COLUMNS = (
    "symbol",
    "continuity_id",
    "trade_id",
    "event_ts_ns",
    "received_ts_ns",
    "available_ts_ns",
    "availability_basis",
    "price",
    "quantity",
    "aggressor_side",
)


class M8PipelineError(RuntimeError):
    """Raised when an M8 run cannot be produced without violating its protocol."""


M8RunStatus = Literal["COMPLETE", "INSUFFICIENT_DATA"]

_M8FailureReasonCode = Literal[
    "RAW_ARCHIVE_INTEGRITY",
    "ARCHIVE_PAYLOAD_OR_CONTINUITY",
    "ARCHIVE_QUALITY_GATE",
    "ARCHIVE_POSTWRITE_CONSISTENCY",
    "RESEARCH_FRAME_INSUFFICIENT",
    "MODEL_SELECTION_INSUFFICIENT",
    "FINAL_NORMALIZED_MANIFEST_INCOMPLETE",
    "LOCKED_EVALUATION_INSUFFICIENT",
]
_M8FailureStage = Literal[
    "development_acquisition",
    "development_normalization",
    "development_research",
    "model_selection",
    "held_out_acquisition",
    "held_out_normalization",
    "final_manifest",
    "held_out_research",
    "locked_evaluation",
]
_M8FailedRole = Literal[
    "train",
    "validation",
    "primary_test",
    "replication_test",
    "study",
    "all_test_dates",
]

_NORMALIZATION_REASON_CODES = frozenset(
    {
        "ARCHIVE_PAYLOAD_OR_CONTINUITY",
        "ARCHIVE_QUALITY_GATE",
        "ARCHIVE_POSTWRITE_CONSISTENCY",
    }
)
_FAILURE_STAGE_CONTRACT: Mapping[str, tuple[bool, frozenset[str], frozenset[str]]] = {
    "development_acquisition": (
        False,
        frozenset({"RAW_ARCHIVE_INTEGRITY"}),
        _DEVELOPMENT_ROLES,
    ),
    "development_normalization": (
        False,
        _NORMALIZATION_REASON_CODES,
        _DEVELOPMENT_ROLES,
    ),
    "development_research": (
        False,
        frozenset({"RESEARCH_FRAME_INSUFFICIENT"}),
        _DEVELOPMENT_ROLES,
    ),
    "model_selection": (
        False,
        frozenset({"MODEL_SELECTION_INSUFFICIENT"}),
        frozenset({"validation"}),
    ),
    "held_out_acquisition": (
        True,
        frozenset({"RAW_ARCHIVE_INTEGRITY"}),
        _TEST_ROLES,
    ),
    "held_out_normalization": (True, _NORMALIZATION_REASON_CODES, _TEST_ROLES),
    "final_manifest": (
        True,
        frozenset({"FINAL_NORMALIZED_MANIFEST_INCOMPLETE"}),
        frozenset({"study"}),
    ),
    "held_out_research": (
        True,
        frozenset({"RESEARCH_FRAME_INSUFFICIENT"}),
        _TEST_ROLES,
    ),
    "locked_evaluation": (
        True,
        frozenset({"LOCKED_EVALUATION_INSUFFICIENT"}),
        frozenset({"all_test_dates"}),
    ),
}


@dataclass(frozen=True, slots=True)
class M8RunResult:
    """Verified terminal result of one immutable M8 production attempt."""

    path: Path
    status: M8RunStatus
    raw_manifest_sha256: str
    normalized_manifest_sha256: str | None

    def __fspath__(self) -> str:
        return str(self.path)


@dataclass(frozen=True, slots=True)
class _TypedInsufficientFailure:
    """Stable machine-readable classification plus diagnostic failure text."""

    symbol: str
    study_date: str
    reason: str
    reason_code: _M8FailureReasonCode
    failure_stage: _M8FailureStage
    failed_role: _M8FailedRole
    normalization_failure_kind: M8NormalizationFailureKind | None = None
    normalization_evidence_completion: M8NormalizationEvidenceCompletion | None = None
    normalization_completed_evidence: M8ArchiveEntry | None = None


@dataclass(frozen=True, slots=True)
class _FailureEvidenceItem:
    """One canonical, checksum-bound entry in a failed run's evidence inventory."""

    path: str
    sha256: str
    bytes: int


def _typed_failure(
    error: M8InsufficientDataError,
    *,
    reason_code: _M8FailureReasonCode | None = None,
    failure_stage: _M8FailureStage,
    failed_role: _M8FailedRole,
) -> _TypedInsufficientFailure:
    """Attach an explicit protocol classification without parsing diagnostic text."""

    normalization_stage = failure_stage in {
        "development_normalization",
        "held_out_normalization",
    }
    normalization_kind = error.failure_kind
    evidence_completion = error.evidence_completion
    completed_evidence = error.completed_evidence
    if normalization_stage:
        mapping: Mapping[M8NormalizationFailureKind, _M8FailureReasonCode] = {
            "PAYLOAD_OR_CONTINUITY": "ARCHIVE_PAYLOAD_OR_CONTINUITY",
            "QUALITY_GATE": "ARCHIVE_QUALITY_GATE",
            "POSTWRITE_CONSISTENCY": "ARCHIVE_POSTWRITE_CONSISTENCY",
        }
        if normalization_kind is None or evidence_completion is None:
            raise M8PipelineError("normalization failure lacks its typed evidence state")
        derived_reason = mapping[normalization_kind]
        if reason_code is not None and reason_code != derived_reason:
            raise M8PipelineError("normalization failure reason code was not derived from its type")
        reason_code = derived_reason
        if normalization_kind == "PAYLOAD_OR_CONTINUITY" and (
            evidence_completion != "PARTIAL_STREAM" or completed_evidence is not None
        ):
            raise M8PipelineError("payload failure has an invalid evidence-completion state")
        if normalization_kind == "QUALITY_GATE" and (
            evidence_completion != "COMPLETE_DATASET_AND_QUALITY" or completed_evidence is None
        ):
            raise M8PipelineError("quality-gate failure lacks complete normalization evidence")
        if normalization_kind == "POSTWRITE_CONSISTENCY" and (
            evidence_completion != "COMPLETE_DATASET_AND_QUALITY" or completed_evidence is not None
        ):
            raise M8PipelineError("postwrite failure has an invalid evidence-completion state")
    elif (
        reason_code is None
        or normalization_kind is not None
        or evidence_completion is not None
        or completed_evidence is not None
    ):
        raise M8PipelineError("non-normalization failure has invalid typed evidence metadata")

    return _TypedInsufficientFailure(
        symbol=error.symbol,
        study_date=error.study_date,
        reason=error.reason,
        reason_code=reason_code,
        failure_stage=failure_stage,
        failed_role=failed_role,
        normalization_failure_kind=normalization_kind,
        normalization_evidence_completion=evidence_completion,
        normalization_completed_evidence=completed_evidence,
    )


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    commit: str
    dirty: bool
    source_tree_sha256: str

    def public_dict(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "source_tree_sha256": self.source_tree_sha256,
        }


@dataclass(frozen=True, slots=True)
class _DateArtifacts:
    symbol: str
    study_date: str
    role: M8PeriodRole
    research_path: Path
    evaluation_path: Path
    summary: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _SelectionArtifacts:
    symbol: str
    selection: LockedSelection
    lock_path: Path
    fitted_state_path: Path
    comparison_path: Path


@dataclass(frozen=True, slots=True)
class _EvaluationArtifacts:
    symbol: str
    selected_model: str
    predictions_path: Path
    paired_date_path: Path
    stability_path: Path
    plan_path: Path
    predictive_rows: tuple[Mapping[str, object], ...]
    paired_date_rows: tuple[Mapping[str, object], ...]
    aggregate_row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _BoundedFileSnapshot:
    """One regular-file byte snapshot read and hashed through a single descriptor."""

    content: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _BoundedJsonSnapshot:
    """Strict canonical JSON decoded from one hash-bound file snapshot."""

    content: bytes
    text: str
    payload: Mapping[str, Any]
    sha256: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    clean = _json_safe(payload)
    if not isinstance(clean, (dict, list)):
        raise TypeError("JSON artifact payload must be an object or list")
    write_json(path, clean)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably commit directory-entry changes on POSIX filesystems."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Flush every regular artifact and directory without following symbolic links."""

    resolved_root = root.resolve()
    if root.is_symlink() or not resolved_root.is_dir():
        raise M8PipelineError("M8 staging tree is not a real directory")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for directory_name, directory_names, file_names in os.walk(
        resolved_root,
        topdown=False,
        followlinks=False,
    ):
        directory = Path(directory_name)
        for child_name in directory_names:
            child = directory / child_name
            if child.is_symlink():
                raise M8PipelineError(f"M8 staging tree contains a symlink: {child}")
        for file_name in file_names:
            path = directory / file_name
            if path.is_symlink():
                raise M8PipelineError(f"M8 staging tree contains a symlink: {path}")
            if not stat.S_ISREG(path.lstat().st_mode):
                raise M8PipelineError(f"M8 staging artifact is not a regular file: {path}")
            descriptor = os.open(path, os.O_RDONLY | nofollow)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise M8PipelineError(f"M8 staging artifact is not a regular file: {path}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(directory)


def _create_terminal_marker(path: Path, content: str) -> None:
    """Create, flush, and durably link a terminal marker exactly once."""

    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, content: str, *, trailing_newline: bool = True) -> None:
    encoded = (content.rstrip("\n") + "\n" if trailing_newline else content).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _require_digest(value: str, label: str) -> str:
    if value != value.lower() or _DIGEST.fullmatch(value) is None:
        raise M8PipelineError(f"{label} must be an exact lowercase SHA-256 digest")
    return value


def _read_bounded_regular_snapshot(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> _BoundedFileSnapshot:
    """Read, bound, and hash one non-symlink regular file through the same FD."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("bounded snapshot size must be a positive integer")
    expected = (
        None if expected_sha256 is None else _require_digest(expected_sha256, f"{label} SHA-256")
    )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise M8PipelineError(f"{label} cannot be opened without O_NOFOLLOW support")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise M8PipelineError(f"cannot open {label} as a non-symlink regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise M8PipelineError(f"{label} is not a regular file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise M8PipelineError(f"{label} exceeds its {max_bytes}-byte hard limit")
        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_LOCK_READ_CHUNK_BYTES, max_bytes + 1 - observed_bytes),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise M8PipelineError(f"{label} exceeds its {max_bytes}-byte hard limit")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before or observed_bytes != before.st_size:
            raise M8PipelineError(f"{label} changed while its bounded snapshot was read")
        try:
            linked = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise M8PipelineError(f"{label} path changed while its snapshot was read") from exc
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_dev != before.st_dev
            or linked.st_ino != before.st_ino
        ):
            raise M8PipelineError(f"{label} path changed identity while its snapshot was read")
        content = b"".join(chunks)
        observed_sha256 = hashlib.sha256(content).hexdigest()
        if expected is not None and observed_sha256 != expected:
            raise M8PipelineError(f"{label} bytes do not match the expected SHA-256")
        return _BoundedFileSnapshot(content=content, sha256=observed_sha256)
    finally:
        os.close(descriptor)


def _strict_canonical_json_snapshot(
    snapshot: _BoundedFileSnapshot,
    *,
    label: str,
    ensure_ascii: bool,
) -> _BoundedJsonSnapshot:
    """Decode canonical JSON while rejecting duplicate keys and non-finite constants."""

    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise M8PipelineError(f"{label} is not valid UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise M8PipelineError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise M8PipelineError(f"{label} contains forbidden JSON constant {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except M8PipelineError:
        raise
    except (RecursionError, TypeError, ValueError) as exc:
        raise M8PipelineError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise M8PipelineError(f"{label} must be a JSON object")
    payload = cast(Mapping[str, Any], decoded)
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=ensure_ascii,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise M8PipelineError(f"{label} cannot be represented as canonical JSON") from exc
    if snapshot.content != canonical:
        raise M8PipelineError(f"{label} bytes are not canonical JSON")
    return _BoundedJsonSnapshot(
        content=snapshot.content,
        text=text,
        payload=payload,
        sha256=snapshot.sha256,
    )


def _read_bounded_json_snapshot(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    ensure_ascii: bool,
) -> _BoundedJsonSnapshot:
    return _strict_canonical_json_snapshot(
        _read_bounded_regular_snapshot(
            path,
            label=label,
            max_bytes=_MAX_LOCK_JSON_BYTES,
            expected_sha256=expected_sha256,
        ),
        label=label,
        ensure_ascii=ensure_ascii,
    )


def _read_exact_bounded_text(path: Path, *, label: str, expected_text: str) -> None:
    snapshot = _read_bounded_regular_snapshot(
        path,
        label=label,
        max_bytes=_MAX_LOCK_DIGEST_BYTES,
    )
    try:
        observed = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise M8PipelineError(f"{label} is not valid UTF-8") from exc
    if observed != expected_text:
        raise M8PipelineError(f"{label} bytes are invalid")


def _restore_analysis_lock_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> AnalysisLock:
    digest = _require_digest(expected_sha256, f"{label} SHA-256")
    snapshot = _read_bounded_json_snapshot(
        path,
        label=label,
        expected_sha256=digest,
        ensure_ascii=True,
    )
    try:
        lock = AnalysisLock.restore(snapshot.text, digest)
        if lock.payload() != snapshot.payload:
            raise M8PipelineError(f"{label} decoded inconsistently")
    except (MultiDateEvaluationError, ValueError) as exc:
        raise M8PipelineError(f"{label} is invalid") from exc
    return lock


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise M8PipelineError(f"run artifact escapes the staging directory: {path}") from exc


def _published_file(root: Path, value: object, label: str) -> Path:
    """Resolve one canonical bundle-relative regular file without following aliases."""

    if type(value) is not str:
        raise M8PipelineError(f"{label} must be a bundle-relative path")
    declared = Path(value)
    if (
        declared.is_absolute()
        or "\\" in value
        or value != declared.as_posix()
        or not declared.parts
        or any(part in {"", ".", ".."} for part in declared.parts)
    ):
        raise M8PipelineError(f"{label} is not a canonical bundle-relative path")
    current = root.resolve()
    for component in declared.parts:
        current /= component
        if current.is_symlink():
            raise M8PipelineError(f"{label} traverses a symbolic link")
    resolved = current.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise M8PipelineError(f"{label} is missing or escapes the completed bundle")
    return resolved


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise M8PipelineError(f"cannot read {label} as JSON") from exc
    if not isinstance(payload, Mapping):
        raise M8PipelineError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], payload)


def _decode_failure_inventory(snapshot: _BoundedFileSnapshot) -> list[Any]:
    """Decode the stable JSON-list representation used by failed evidence inventories."""

    label = "INSUFFICIENT_DATA failure evidence inventory"
    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise M8PipelineError(f"{label} is not valid UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise M8PipelineError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise M8PipelineError(f"{label} contains forbidden JSON constant {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except M8PipelineError:
        raise
    except (RecursionError, TypeError, ValueError) as exc:
        raise M8PipelineError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, list):
        raise M8PipelineError(f"{label} must be a JSON array")
    try:
        stable = (json.dumps(decoded, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise M8PipelineError(f"{label} is not stable JSON") from exc
    if snapshot.content != stable:
        raise M8PipelineError(f"{label} is not stable canonical JSON")
    return decoded


def _decode_stable_pretty_json_object(
    snapshot: _BoundedFileSnapshot,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Decode the pretty canonical JSON used by terminal and storage records."""

    try:
        text = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise M8PipelineError(f"{label} is not valid UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise M8PipelineError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise M8PipelineError(f"{label} contains forbidden JSON constant {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except M8PipelineError:
        raise
    except (RecursionError, TypeError, ValueError) as exc:
        raise M8PipelineError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise M8PipelineError(f"{label} must be a JSON object")
    try:
        stable = (json.dumps(decoded, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise M8PipelineError(f"{label} is not stable JSON") from exc
    if snapshot.content != stable:
        raise M8PipelineError(f"{label} is not stable canonical JSON")
    return cast(Mapping[str, Any], decoded)


def _verify_inventory_file(
    root: Path,
    item: _FailureEvidenceItem,
) -> None:
    """Hash one inventoried file through a stable non-following descriptor."""

    label = f"failure evidence inventory artifact {item.path}"
    path = _published_file(root, item.path, label)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise M8PipelineError(f"{label} cannot be opened without O_NOFOLLOW support")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise M8PipelineError(f"cannot open {label} as a regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise M8PipelineError(f"{label} is not a regular file")
        if before.st_size != item.bytes:
            raise M8PipelineError(f"{label} byte count disagrees with its inventory claim")
        digest = hashlib.sha256()
        observed_bytes = 0
        while chunk := os.read(descriptor, _LOCK_READ_CHUNK_BYTES):
            digest.update(chunk)
            observed_bytes += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before or observed_bytes != item.bytes:
            raise M8PipelineError(f"{label} changed while it was verified")
        try:
            linked = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise M8PipelineError(f"{label} path changed while it was verified") from exc
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_dev != before.st_dev
            or linked.st_ino != before.st_ino
        ):
            raise M8PipelineError(f"{label} path changed identity while it was verified")
        if digest.hexdigest() != item.sha256:
            raise M8PipelineError(f"{label} SHA-256 disagrees with its inventory claim")
    finally:
        os.close(descriptor)


def _verify_failure_evidence_inventory(
    target: Path,
) -> dict[str, _FailureEvidenceItem]:
    """Verify a failed bundle's pre-terminal inventory against its exact physical tree."""

    inventory_relative = "data/failure_evidence_inventory.json"
    inventory_path = _published_file(
        target,
        inventory_relative,
        "INSUFFICIENT_DATA failure evidence inventory",
    )
    snapshot = _read_bounded_regular_snapshot(
        inventory_path,
        label="INSUFFICIENT_DATA failure evidence inventory",
        max_bytes=_MAX_FAILURE_INVENTORY_BYTES,
    )
    raw_entries = _decode_failure_inventory(snapshot)
    if not raw_entries or len(raw_entries) > _MAX_FAILURE_INVENTORY_ENTRIES:
        raise M8PipelineError(
            "INSUFFICIENT_DATA failure evidence inventory has an invalid entry count"
        )

    entries: list[_FailureEvidenceItem] = []
    for index, raw_entry in enumerate(raw_entries):
        label = f"failure evidence inventory[{index}]"
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"path", "sha256", "bytes"}:
            raise M8PipelineError(f"{label} has an invalid schema")
        path_value = raw_entry.get("path")
        if type(path_value) is not str:
            raise M8PipelineError(f"{label}.path must be a canonical bundle-relative path")
        declared = Path(path_value)
        if (
            declared.is_absolute()
            or "\\" in path_value
            or path_value != declared.as_posix()
            or not declared.parts
            or any(part in {"", ".", ".."} for part in declared.parts)
        ):
            raise M8PipelineError(f"{label}.path is not a canonical bundle-relative path")
        sha_value = raw_entry.get("sha256")
        if type(sha_value) is not str:
            raise M8PipelineError(f"{label}.sha256 must be one lowercase SHA-256")
        digest = _require_digest(sha_value, f"{label}.sha256")
        byte_value = raw_entry.get("bytes")
        if isinstance(byte_value, bool) or not isinstance(byte_value, int) or byte_value < 0:
            raise M8PipelineError(f"{label}.bytes must be a non-negative integer")
        entries.append(_FailureEvidenceItem(path=path_value, sha256=digest, bytes=byte_value))

    paths = [item.path for item in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise M8PipelineError(
            "INSUFFICIENT_DATA failure evidence inventory paths are duplicate or unordered"
        )

    actual_files: set[str] = set()
    try:
        for path in target.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise M8PipelineError(
                    "INSUFFICIENT_DATA bundle contains a symbolic link outside its inventory"
                )
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise M8PipelineError(
                    "INSUFFICIENT_DATA bundle contains a non-regular inventory artifact"
                )
            actual_files.add(path.relative_to(target).as_posix())
    except OSError as exc:
        raise M8PipelineError("cannot enumerate INSUFFICIENT_DATA evidence inventory") from exc

    terminal_exclusions = {
        inventory_relative,
        "checksums.sha256",
        "INSUFFICIENT_DATA",
    }
    if not terminal_exclusions.issubset(actual_files):
        raise M8PipelineError(
            "INSUFFICIENT_DATA evidence inventory is missing a terminal publication artifact"
        )
    inventoried_paths = set(paths)
    observed_scope = actual_files - terminal_exclusions
    missing = sorted(observed_scope - inventoried_paths)
    extra = sorted(inventoried_paths - observed_scope)
    if missing or extra:
        raise M8PipelineError(
            "INSUFFICIENT_DATA failure evidence inventory differs from the physical bundle: "
            f"missing={missing}, extra={extra}"
        )

    by_path = {item.path: item for item in entries}
    for item in entries:
        _verify_inventory_file(target, item)
    return by_path


def _canonical_inventory_child(value: object, label: str) -> str:
    if type(value) is not str:
        raise M8PipelineError(f"{label} must be a canonical relative path")
    declared = Path(value)
    if (
        declared.is_absolute()
        or "\\" in value
        or value != declared.as_posix()
        or not declared.parts
        or any(part in {"", ".", ".."} for part in declared.parts)
    ):
        raise M8PipelineError(f"{label} is not a canonical relative path")
    return value


def _expected_completed_normalizations(
    config: M8StudyConfig,
    failure: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    declared: list[tuple[str, str, str]] = [
        (symbol, period.date.isoformat(), period.role)
        for period in config.periods
        for symbol in config.study.symbols
    ]
    stage = failure.get("failure_stage")
    if stage == "model_selection":
        return [item for item in declared if item[2] in _DEVELOPMENT_ROLES]
    if stage in {"final_manifest", "held_out_research", "locked_evaluation"}:
        return declared
    failed = (
        failure.get("failed_symbol"),
        failure.get("failed_date"),
        failure.get("failed_role"),
    )
    try:
        failed_index = declared.index(cast(tuple[str, str, str], failed))
    except ValueError as exc:
        raise M8PipelineError(
            "INSUFFICIENT_DATA completed normalization progress has no declared failure boundary"
        ) from exc
    if stage == "development_research":
        return declared[: failed_index + 1]
    if stage in {
        "development_acquisition",
        "development_normalization",
        "held_out_acquisition",
        "held_out_normalization",
    }:
        return declared[:failed_index]
    raise M8PipelineError(
        "INSUFFICIENT_DATA completed normalization progress has an unsupported failure stage"
    )


def _require_inventory_claim(
    inventory: Mapping[str, _FailureEvidenceItem],
    path: str,
    label: str,
    *,
    sha256: str | None = None,
    expected_bytes: int | None = None,
) -> _FailureEvidenceItem:
    item = inventory.get(path)
    if item is None:
        raise M8PipelineError(f"{label} is absent from the failure evidence inventory")
    if sha256 is not None and item.sha256 != sha256:
        raise M8PipelineError(f"{label} SHA-256 differs from the failure evidence inventory")
    if expected_bytes is not None and item.bytes != expected_bytes:
        raise M8PipelineError(f"{label} byte count differs from the failure evidence inventory")
    return item


def _read_inventory_bounded_snapshot(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    relative_path: str,
    label: str,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> _BoundedFileSnapshot:
    """Read exactly the bytes already claimed by a verified failure inventory."""

    item = _require_inventory_claim(
        inventory,
        relative_path,
        label,
        sha256=expected_sha256,
    )
    if item.bytes < 1 or item.bytes > max_bytes:
        raise M8PipelineError(f"{label} exceeds its {max_bytes}-byte hard limit")
    snapshot = _read_bounded_regular_snapshot(
        _published_file(target, relative_path, label),
        label=label,
        max_bytes=max_bytes,
        expected_sha256=item.sha256,
    )
    if len(snapshot.content) != item.bytes:
        raise M8PipelineError(f"{label} byte count differs from the failure inventory")
    return snapshot


def _read_inventory_bounded_json_snapshot(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    relative_path: str,
    label: str,
    expected_sha256: str,
    ensure_ascii: bool,
) -> _BoundedJsonSnapshot:
    """Decode canonical JSON from the same FD-bound bytes named by inventory."""

    return _strict_canonical_json_snapshot(
        _read_inventory_bounded_snapshot(
            target=target,
            inventory=inventory,
            relative_path=relative_path,
            label=label,
            max_bytes=_MAX_LOCK_JSON_BYTES,
            expected_sha256=expected_sha256,
        ),
        label=label,
        ensure_ascii=ensure_ascii,
    )


def _read_inventory_exact_bounded_text(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    relative_path: str,
    label: str,
    expected_text: str,
) -> None:
    snapshot = _read_inventory_bounded_snapshot(
        target=target,
        inventory=inventory,
        relative_path=relative_path,
        label=label,
        max_bytes=_MAX_LOCK_DIGEST_BYTES,
    )
    try:
        observed = snapshot.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise M8PipelineError(f"{label} is not valid UTF-8") from exc
    if observed != expected_text:
        raise M8PipelineError(f"{label} bytes are invalid")


def _read_inventory_json_object(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    relative_path: str,
    label: str,
    max_bytes: int = _MAX_LOCK_JSON_BYTES,
) -> Mapping[str, Any]:
    """Read one semantic JSON record from the same bounded bytes bound by inventory."""

    item = _require_inventory_claim(inventory, relative_path, label)
    if item.bytes < 1 or item.bytes > max_bytes:
        raise M8PipelineError(f"{label} exceeds its JSON byte limit")
    snapshot = _read_bounded_regular_snapshot(
        _published_file(target, relative_path, label),
        label=label,
        max_bytes=max_bytes,
        expected_sha256=item.sha256,
    )
    if len(snapshot.content) != item.bytes:
        raise M8PipelineError(f"{label} byte count differs from the failure inventory")
    return _decode_stable_pretty_json_object(snapshot, label=label)


def _verify_quality_findings_jsonl(
    *,
    target: Path,
    item: _FailureEvidenceItem,
    expected_errors: int,
    expected_warnings: int,
    label: str,
) -> None:
    """Stream and count one inventory-bound findings file without retaining it."""

    path = _published_file(target, item.path, label)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise M8PipelineError(f"{label} cannot be opened without O_NOFOLLOW support")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise M8PipelineError(f"cannot open {label} as a regular file") from exc
    digest = hashlib.sha256()
    observed_bytes = 0
    buffered = b""
    errors = 0
    warnings = 0

    def parse_line(line: bytes) -> str:
        if not line or len(line) > _MAX_FAILURE_FINDING_LINE_BYTES:
            raise M8PipelineError(f"{label} contains an empty or oversized JSON line")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise M8PipelineError(f"{label} repeats JSON key {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise M8PipelineError(f"{label} contains forbidden JSON constant {value}")

        try:
            decoded = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except M8PipelineError:
            raise
        except (RecursionError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise M8PipelineError(f"{label} contains invalid JSONL") from exc
        if not isinstance(decoded, Mapping):
            raise M8PipelineError(f"{label} JSONL entries must be objects")
        try:
            canonical = json.dumps(
                decoded,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (RecursionError, TypeError, ValueError) as exc:
            raise M8PipelineError(f"{label} contains unstable JSONL") from exc
        if line != canonical:
            raise M8PipelineError(f"{label} is not stable canonical JSONL")
        severity = decoded.get("severity")
        if severity not in {"ERROR", "WARNING"}:
            raise M8PipelineError(f"{label} contains an invalid finding severity")
        return cast(str, severity)

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != item.bytes:
            raise M8PipelineError(f"{label} byte count differs from its inventory")
        while chunk := os.read(descriptor, _LOCK_READ_CHUNK_BYTES):
            observed_bytes += len(chunk)
            digest.update(chunk)
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                severity = parse_line(line)
                if severity == "ERROR":
                    errors += 1
                else:
                    warnings += 1
            if len(buffered) > _MAX_FAILURE_FINDING_LINE_BYTES:
                raise M8PipelineError(f"{label} contains an oversized JSON line")
        if buffered:
            raise M8PipelineError(f"{label} must end every JSONL entry with a newline")
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or observed_bytes != item.bytes:
            raise M8PipelineError(f"{label} changed while it was verified")
        linked = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_dev != before.st_dev
            or linked.st_ino != before.st_ino
            or digest.hexdigest() != item.sha256
        ):
            raise M8PipelineError(f"{label} changed identity or SHA-256 while verified")
    except OSError as exc:
        raise M8PipelineError(f"cannot verify {label}") from exc
    finally:
        os.close(descriptor)
    if errors != expected_errors or warnings != expected_warnings:
        raise M8PipelineError(f"{label} severity counts differ from the quality report")


def _verify_failed_normalization_evidence(
    *,
    target: Path,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    failure: Mapping[str, Any],
    inventory: Mapping[str, _FailureEvidenceItem],
) -> None:
    """Verify the stage-derived failed symbol/date evidence contract."""

    reason_code = failure.get("reason_code")
    raw_evidence = failure.get("failed_normalization_evidence")
    reason_contract: Mapping[str, tuple[str, str]] = {
        "ARCHIVE_PAYLOAD_OR_CONTINUITY": ("PAYLOAD_OR_CONTINUITY", "PARTIAL_STREAM"),
        "ARCHIVE_QUALITY_GATE": ("QUALITY_GATE", "COMPLETE_DATASET_AND_QUALITY"),
        "ARCHIVE_POSTWRITE_CONSISTENCY": (
            "POSTWRITE_CONSISTENCY",
            "COMPLETE_DATASET_AND_QUALITY",
        ),
    }
    expected_typed = reason_contract.get(cast(str, reason_code))
    if expected_typed is None:
        if raw_evidence is not None:
            raise M8PipelineError("non-normalization failure claims normalization evidence")
        return
    if not isinstance(raw_evidence, Mapping):
        raise M8PipelineError("normalization failure lacks its typed evidence authority")
    expected_keys = {
        "schema_version",
        "failure_kind",
        "evidence_completion",
        "normalized_prefix",
        "quality_prefix",
        "artifacts",
        "complete_normalization",
    }
    if set(raw_evidence) != expected_keys:
        raise M8PipelineError("failed normalization evidence schema is invalid")
    symbol = failure.get("failed_symbol")
    study_date = failure.get("failed_date")
    role = failure.get("failed_role")
    if type(symbol) is not str or type(study_date) is not str or type(role) is not str:
        raise M8PipelineError("failed normalization identity is malformed")
    normalized_prefix = f"data/normalized_input/normalized/{symbol}/{study_date}"
    quality_prefix = f"data/normalized_input/quality/{symbol}/{study_date}"
    kind, completion = expected_typed
    if (
        raw_evidence.get("schema_version") != "m8-failed-normalization-evidence-v1"
        or raw_evidence.get("failure_kind") != kind
        or raw_evidence.get("evidence_completion") != completion
        or raw_evidence.get("normalized_prefix") != normalized_prefix
        or raw_evidence.get("quality_prefix") != quality_prefix
    ):
        raise M8PipelineError("failed normalization evidence has inconsistent typed claims")

    raw_artifacts = raw_evidence.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise M8PipelineError("failed normalization scoped artifacts must be a JSON array")
    scoped_claims: list[_FailureEvidenceItem] = []
    for index, value in enumerate(raw_artifacts):
        label = f"failed normalization artifacts[{index}]"
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
            raise M8PipelineError(f"{label} schema is invalid")
        path = value.get("path")
        sha = value.get("sha256")
        byte_count = value.get("bytes")
        if type(path) is not str or not (
            path.startswith(f"{normalized_prefix}/") or path.startswith(f"{quality_prefix}/")
        ):
            raise M8PipelineError(f"{label} is outside the frozen failed evidence roots")
        if type(sha) is not str:
            raise M8PipelineError(f"{label} lacks its SHA-256")
        digest = _require_digest(sha, f"{label} SHA-256")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise M8PipelineError(f"{label} has an invalid byte count")
        scoped_claims.append(_FailureEvidenceItem(path, digest, byte_count))
    claim_paths = [item.path for item in scoped_claims]
    if claim_paths != sorted(claim_paths) or len(claim_paths) != len(set(claim_paths)):
        raise M8PipelineError("failed normalization artifact paths are duplicate or unordered")
    observed_scope = {
        path: item
        for path, item in inventory.items()
        if path.startswith(f"{normalized_prefix}/") or path.startswith(f"{quality_prefix}/")
    }
    if set(claim_paths) != set(observed_scope) or any(
        observed_scope[item.path] != item for item in scoped_claims
    ):
        raise M8PipelineError("failed normalization evidence differs from its inventoried tree")

    dataset_paths = [
        path
        for path in observed_scope
        if Path(path).parent.as_posix() == f"{normalized_prefix}/_manifests"
        and re.fullmatch(r"trades\.manifest-[0-9a-f]{20}\.json", Path(path).name)
    ]
    report_path = f"{quality_prefix}/report.json"
    findings_path = f"{quality_prefix}/findings.jsonl"
    complete_claim = raw_evidence.get("complete_normalization")
    if completion == "PARTIAL_STREAM":
        if dataset_paths or report_path in observed_scope or findings_path in observed_scope:
            raise M8PipelineError("partial payload failure contains final normalization evidence")
        if complete_claim is not None:
            raise M8PipelineError("partial payload failure claims a complete normalization")
        allowed_part = re.compile(
            rf"^{re.escape(normalized_prefix)}/trades/schema-1\.0\.0/"
            rf"venue-binance_spot/symbol-{re.escape(symbol)}/date-{re.escape(study_date)}/"
            r"part-[0-9a-f]{20}(?:\.parquet|\.manifest-[0-9a-f]{20}\.json)$"
        )
        if any(allowed_part.fullmatch(path) is None for path in observed_scope):
            raise M8PipelineError("partial payload failure contains a noncanonical artifact")
        return

    if (
        len(dataset_paths) != 1
        or report_path not in observed_scope
        or findings_path not in observed_scope
    ):
        raise M8PipelineError("complete failed normalization lacks dataset and quality evidence")
    if kind == "POSTWRITE_CONSISTENCY":
        if complete_claim is not None:
            raise M8PipelineError("postwrite failure claims an accepted normalization entry")
        _read_inventory_json_object(
            target=target,
            inventory=inventory,
            relative_path=dataset_paths[0],
            label="postwrite failed dataset manifest",
        )
        _read_inventory_json_object(
            target=target,
            inventory=inventory,
            relative_path=report_path,
            label="postwrite failed quality report",
        )
        return

    if not isinstance(complete_claim, Mapping):
        raise M8PipelineError("quality-gate failure lacks its complete normalization claim")
    _verify_normalization_claims(
        target=target,
        config=config,
        raw_manifest=raw_manifest,
        inventory=inventory,
        raw_completed=[complete_claim],
        expected_order=[(symbol, study_date, role)],
        require_quality_gate_failure=True,
    )


def _verify_normalization_claims(
    *,
    target: Path,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    inventory: Mapping[str, _FailureEvidenceItem],
    raw_completed: Sequence[Any],
    expected_order: Sequence[tuple[str, str, str]],
    require_quality_gate_failure: bool,
) -> None:
    """Bind normalization claims to raw authority, parts, and exact DQ evidence."""

    observed_order: list[tuple[str, str, str]] = []
    raw_by_key = {
        (entry.symbol, entry.date.isoformat(), str(entry.role)): entry
        for entry in raw_manifest.archives
    }
    expected_keys = {
        "symbol",
        "date",
        "role",
        "rows",
        "raw_zip_sha256",
        "normalized_dataset_manifest_sha256",
        "quality_errors",
        "quality_warnings",
    }
    for index, raw_claim in enumerate(raw_completed):
        label = f"completed normalizations[{index}]"
        if not isinstance(raw_claim, Mapping) or set(raw_claim) != expected_keys:
            raise M8PipelineError(f"{label} has an invalid schema")
        symbol = raw_claim.get("symbol")
        study_date = raw_claim.get("date")
        role = raw_claim.get("role")
        if type(symbol) is not str or type(study_date) is not str or type(role) is not str:
            raise M8PipelineError(f"{label} has a malformed symbol/date/role identity")
        identity = (symbol, study_date, role)
        observed_order.append(identity)
        raw_entry = raw_by_key.get(identity)
        if raw_entry is None:
            raise M8PipelineError(f"{label} is outside the frozen raw acquisition calendar")

        rows = raw_claim.get("rows")
        errors = raw_claim.get("quality_errors")
        warnings = raw_claim.get("quality_warnings")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 1
            or isinstance(errors, bool)
            or not isinstance(errors, int)
            or errors < 0
            or isinstance(warnings, bool)
            or not isinstance(warnings, int)
            or warnings < 0
        ):
            raise M8PipelineError(f"{label} has invalid row or quality counts")
        raw_sha_value = raw_claim.get("raw_zip_sha256")
        normalized_sha_value = raw_claim.get("normalized_dataset_manifest_sha256")
        if type(raw_sha_value) is not str or type(normalized_sha_value) is not str:
            raise M8PipelineError(f"{label} lacks required SHA-256 claims")
        raw_sha = _require_digest(raw_sha_value, f"{label} raw ZIP SHA-256")
        normalized_sha = _require_digest(
            normalized_sha_value,
            f"{label} normalized manifest SHA-256",
        )
        if raw_sha != raw_entry.archive_sha256:
            raise M8PipelineError(f"{label} raw ZIP SHA-256 differs from acquisition authority")

        archive_root = f"data/normalized_input/normalized/{symbol}/{study_date}"
        dataset_directory = f"{archive_root}/_manifests"
        dataset_matches = [
            item
            for path, item in inventory.items()
            if Path(path).parent.as_posix() == dataset_directory
            and re.fullmatch(r"trades\.manifest-[0-9a-f]{20}\.json", Path(path).name)
            and item.sha256 == normalized_sha
        ]
        if len(dataset_matches) != 1:
            raise M8PipelineError(
                f"{label} normalized dataset manifest is absent or ambiguous in the inventory"
            )
        dataset_item = dataset_matches[0]
        dataset_relative = dataset_item.path
        if dataset_item.bytes > _MAX_LOCK_JSON_BYTES:
            raise M8PipelineError(f"{label} normalized dataset manifest exceeds its JSON limit")
        dataset = _read_inventory_json_object(
            target=target,
            inventory=inventory,
            relative_path=dataset_relative,
            label=f"{label} normalized dataset manifest",
        )
        if (
            dataset.get("dataset") != "trades"
            or dataset.get("source") != config.study.source
            or dataset.get("source_uri") != raw_entry.archive_source_uri
            or dataset.get("rows") != rows
        ):
            raise M8PipelineError(f"{label} normalized dataset manifest claims differ")
        artifacts = dataset.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise M8PipelineError(f"{label} normalized dataset manifest has no artifacts")
        part_rows = 0
        part_paths: set[str] = set()
        for part_index, raw_part in enumerate(artifacts):
            part_label = f"{label} normalized part {part_index}"
            if not isinstance(raw_part, Mapping):
                raise M8PipelineError(f"{part_label} is not a JSON object")
            ordinal = raw_part.get("write_ordinal")
            part_row_count = raw_part.get("rows")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal != part_index
                or isinstance(part_row_count, bool)
                or not isinstance(part_row_count, int)
                or part_row_count < 1
            ):
                raise M8PipelineError(f"{part_label} has invalid row or ordinal claims")
            part_rows += part_row_count
            for path_key, sha_key, kind in (
                ("data_path", "data_sha256", "data"),
                ("manifest_path", "manifest_sha256", "sidecar"),
            ):
                child = _canonical_inventory_child(
                    raw_part.get(path_key),
                    f"{part_label} {kind} path",
                )
                sha_value = raw_part.get(sha_key)
                if type(sha_value) is not str:
                    raise M8PipelineError(f"{part_label} {kind} lacks its SHA-256")
                part_sha = _require_digest(sha_value, f"{part_label} {kind} SHA-256")
                part_relative = f"{archive_root}/{child}"
                if part_relative in part_paths:
                    raise M8PipelineError(f"{label} normalized dataset reuses an artifact path")
                part_paths.add(part_relative)
                _require_inventory_claim(
                    inventory,
                    part_relative,
                    f"{part_label} {kind}",
                    sha256=part_sha,
                )
        if part_rows != rows:
            raise M8PipelineError(f"{label} normalized part rows do not sum to its row claim")

        quality_root = f"data/normalized_input/quality/{symbol}/{study_date}"
        report_relative = f"{quality_root}/report.json"
        findings_relative = f"{quality_root}/findings.jsonl"
        report_item = _require_inventory_claim(
            inventory,
            report_relative,
            f"{label} quality report",
        )
        findings_item = _require_inventory_claim(
            inventory,
            findings_relative,
            f"{label} quality findings",
        )
        if report_item.bytes > _MAX_LOCK_JSON_BYTES:
            raise M8PipelineError(f"{label} quality report exceeds its JSON limit")
        report = _read_inventory_json_object(
            target=target,
            inventory=inventory,
            relative_path=report_relative,
            label=f"{label} quality report",
        )
        summary = report.get("summary")
        if (
            report.get("dataset") != "trades"
            or report.get("rows_checked") != rows
            or not isinstance(summary, Mapping)
            or summary.get("errors") != errors
            or summary.get("warnings") != warnings
        ):
            raise M8PipelineError(f"{label} quality report claims differ")
        gate_failed = errors > 0 or (warnings > 0 and not config.quality.allow_quality_warnings)
        if gate_failed is not require_quality_gate_failure:
            raise M8PipelineError(f"{label} quality counts disagree with its completion state")
        _verify_quality_findings_jsonl(
            target=target,
            item=findings_item,
            expected_errors=errors,
            expected_warnings=warnings,
            label=f"{label} quality findings JSONL",
        )
        if require_quality_gate_failure:
            expected_scope = {
                dataset_relative,
                report_relative,
                findings_relative,
                *part_paths,
            }
            observed_scope = {
                path
                for path in inventory
                if path.startswith(f"{archive_root}/") or path.startswith(f"{quality_root}/")
            }
            if observed_scope != expected_scope:
                raise M8PipelineError(
                    f"{label} complete failed evidence omits or adds scoped artifacts"
                )

    if observed_order != expected_order:
        raise M8PipelineError(
            "INSUFFICIENT_DATA completed normalizations differ from frozen failure progress"
        )


def _verify_completed_normalization_evidence(
    *,
    target: Path,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    failure: Mapping[str, Any],
    inventory: Mapping[str, _FailureEvidenceItem],
) -> None:
    """Bind every successfully completed normalization before the failure boundary."""

    raw_completed = failure.get("completed_normalizations")
    if not isinstance(raw_completed, list):
        raise M8PipelineError("INSUFFICIENT_DATA completed normalizations must be a JSON array")
    _verify_normalization_claims(
        target=target,
        config=config,
        raw_manifest=raw_manifest,
        inventory=inventory,
        raw_completed=raw_completed,
        expected_order=_expected_completed_normalizations(config, failure),
        require_quality_gate_failure=False,
    )


def _bind_inventory_artifact(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    path: Path,
    sha256: str,
    byte_count: int,
    label: str,
) -> str:
    """Bind one semantic path/SHA/size claim to an inventoried same-FD read."""

    try:
        relative = path.relative_to(target.resolve()).as_posix()
    except ValueError as exc:
        raise M8PipelineError(f"{label} escapes the failed bundle") from exc
    relative = _canonical_inventory_child(relative, f"{label} path")
    item = _require_inventory_claim(
        inventory,
        relative,
        label,
        sha256=_require_digest(sha256, f"{label} SHA-256"),
        expected_bytes=byte_count,
    )
    _verify_inventory_file(target, item)
    return relative


def _bind_raw_acquisition_inventory(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    manifest: M8AcquisitionManifest,
    manifest_relative: str,
) -> None:
    """Exact-bind a bundled raw manifest and every retained raw artifact."""

    manifest_item = _require_inventory_claim(
        inventory,
        manifest_relative,
        "INSUFFICIENT_DATA bundled raw acquisition manifest",
        sha256=manifest.sha256,
    )
    if manifest_item.bytes < 1 or manifest_item.bytes > _MAX_LOCK_JSON_BYTES:
        raise M8PipelineError("bundled raw acquisition manifest exceeds its JSON byte limit")
    raw_root = manifest.root
    try:
        raw_prefix = raw_root.relative_to(target.resolve()).as_posix()
    except ValueError as exc:
        raise M8PipelineError("bundled raw acquisition root escapes the failed bundle") from exc
    expected_scope = {manifest_relative}
    for index, artifact in enumerate(manifest.retained_artifacts):
        child = _canonical_inventory_child(
            artifact.path,
            f"bundled raw retained artifact[{index}] path",
        )
        relative = f"{raw_prefix}/{child}"
        expected_scope.add(relative)
        item = _require_inventory_claim(
            inventory,
            relative,
            f"bundled raw retained artifact[{index}]",
            sha256=_require_digest(
                artifact.sha256,
                f"bundled raw retained artifact[{index}] SHA-256",
            ),
            expected_bytes=artifact.bytes,
        )
        _verify_inventory_file(target, item)
    observed_scope = {
        path for path in inventory if path == raw_prefix or path.startswith(f"{raw_prefix}/")
    }
    if observed_scope != expected_scope:
        raise M8PipelineError(
            "bundled raw acquisition paths differ from the failure evidence inventory"
        )


def _bind_final_input_manifest_inventory(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    manifest: M8InputManifest,
) -> None:
    """Exact-bind the optional final manifest and every artifact it names."""

    try:
        manifest_relative = manifest.path.relative_to(target.resolve()).as_posix()
    except ValueError as exc:
        raise M8PipelineError("final normalized manifest escapes the failed bundle") from exc
    manifest_relative = _canonical_inventory_child(
        manifest_relative,
        "INSUFFICIENT_DATA final normalized manifest path",
    )
    _read_inventory_bounded_snapshot(
        target=target,
        inventory=inventory,
        relative_path=manifest_relative,
        label="INSUFFICIENT_DATA final normalized manifest",
        max_bytes=_MAX_LOCK_JSON_BYTES,
        expected_sha256=manifest.sha256,
    )
    for metadata in manifest.symbol_metadata:
        _bind_inventory_artifact(
            target=target,
            inventory=inventory,
            path=metadata.raw_path,
            sha256=metadata.raw_sha256,
            byte_count=metadata.raw_bytes,
            label=f"final manifest {metadata.symbol} metadata body",
        )
        _bind_inventory_artifact(
            target=target,
            inventory=inventory,
            path=metadata.source_manifest_path,
            sha256=metadata.source_manifest_sha256,
            byte_count=metadata.source_manifest_bytes,
            label=f"final manifest {metadata.symbol} metadata source manifest",
        )
    for entry in manifest.entries:
        label = f"final manifest {entry.symbol}/{entry.date.isoformat()}"
        artifact_claims = (
            (entry.raw_zip_path, entry.raw_zip_sha256, entry.raw_zip_bytes, "raw ZIP"),
            (
                entry.raw_source_manifest_path,
                entry.raw_source_manifest_sha256,
                entry.raw_source_manifest_bytes,
                "raw source manifest",
            ),
            (
                entry.raw_checksum_path,
                entry.raw_checksum_sha256,
                entry.raw_checksum_bytes,
                "official checksum",
            ),
            (
                entry.raw_checksum_source_manifest_path,
                entry.raw_checksum_source_manifest_sha256,
                entry.raw_checksum_source_manifest_bytes,
                "checksum source manifest",
            ),
            (
                entry.normalized_dataset_manifest_path,
                entry.normalized_dataset_manifest_sha256,
                entry.normalized_dataset_manifest_bytes,
                "normalized dataset manifest",
            ),
            (
                entry.quality_report_path,
                entry.quality_report_sha256,
                entry.quality_report_bytes,
                "quality report",
            ),
            (
                entry.quality_findings_path,
                entry.quality_findings_sha256,
                entry.quality_findings_bytes,
                "quality findings",
            ),
        )
        for path, digest, byte_count, kind in artifact_claims:
            _bind_inventory_artifact(
                target=target,
                inventory=inventory,
                path=path,
                sha256=digest,
                byte_count=byte_count,
                label=f"{label} {kind}",
            )
        for part in entry.normalized_parts:
            _bind_inventory_artifact(
                target=target,
                inventory=inventory,
                path=part.data_path,
                sha256=part.data_sha256,
                byte_count=part.data_bytes,
                label=f"{label} normalized part {part.write_ordinal}",
            )
            _bind_inventory_artifact(
                target=target,
                inventory=inventory,
                path=part.sidecar_path,
                sha256=part.sidecar_sha256,
                byte_count=part.sidecar_bytes,
                label=f"{label} normalized part {part.write_ordinal} sidecar",
            )


def _restore_fitted_state_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> FinalFittedState:
    digest = _require_digest(expected_sha256, f"{label} SHA-256")
    snapshot = _read_bounded_json_snapshot(
        path,
        label=label,
        expected_sha256=digest,
        ensure_ascii=True,
    )
    try:
        state = FinalFittedState.restore(snapshot.text, digest)
        if state.payload() != snapshot.payload:
            raise M8PipelineError(f"{label} decoded inconsistently")
    except (MultiDateEvaluationError, ValueError) as exc:
        raise M8PipelineError(f"{label} is invalid") from exc
    return state


def _restore_inventory_analysis_lock_file(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> AnalysisLock:
    """Restore a failed-run lock only from its inventory-bound same-FD snapshot."""

    digest = _require_digest(expected_sha256, f"{label} SHA-256")
    snapshot = _read_inventory_bounded_json_snapshot(
        target=target,
        inventory=inventory,
        relative_path=relative_path,
        label=label,
        expected_sha256=digest,
        ensure_ascii=True,
    )
    try:
        lock = AnalysisLock.restore(snapshot.text, digest)
        if lock.payload() != snapshot.payload:
            raise M8PipelineError(f"{label} decoded inconsistently")
    except (MultiDateEvaluationError, ValueError) as exc:
        raise M8PipelineError(f"{label} is invalid") from exc
    return lock


def _restore_inventory_fitted_state_file(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    relative_path: str,
    expected_sha256: str,
    label: str,
) -> FinalFittedState:
    """Restore a failed-run fitted state from inventory-bound same-FD bytes."""

    digest = _require_digest(expected_sha256, f"{label} SHA-256")
    snapshot = _read_inventory_bounded_json_snapshot(
        target=target,
        inventory=inventory,
        relative_path=relative_path,
        label=label,
        expected_sha256=digest,
        ensure_ascii=True,
    )
    try:
        state = FinalFittedState.restore(snapshot.text, digest)
        if state.payload() != snapshot.payload:
            raise M8PipelineError(f"{label} decoded inconsistently")
    except (MultiDateEvaluationError, ValueError) as exc:
        raise M8PipelineError(f"{label} is invalid") from exc
    return state


def _caused_by_system_fault(error: BaseException) -> bool:
    """Distinguish an acquisition integrity rejection from wrapped local I/O faults."""

    observed: BaseException | None = error
    seen: set[int] = set()
    while observed is not None and id(observed) not in seen:
        seen.add(id(observed))
        if observed is not error and isinstance(observed, OSError):
            return True
        observed = observed.__cause__ or observed.__context__
    return False


def _utc_from_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    instant = datetime.fromtimestamp(seconds, tz=UTC)
    return f"{instant:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def _day_bounds_ns(day: date) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    start_ns = int(start.timestamp()) * 1_000_000_000
    return start_ns, start_ns + 86_400 * 1_000_000_000


def _project_root(config: M8StudyConfig) -> Path:
    source = config.path.resolve()
    if not source.is_file():
        raise M8PipelineError(f"M8 machine specification is missing: {source}")
    root = source.parent.parent.resolve()
    if not (root / "pyproject.toml").is_file():
        raise M8PipelineError(
            "M8 configuration must live below the research project root containing pyproject.toml"
        )
    return root


def _verify_config_source(config: M8StudyConfig) -> None:
    observed = sha256_file(config.path)
    if observed != config.source_sha256:
        raise M8PipelineError(
            "M8 machine specification bytes changed after the configuration was loaded"
        )


def _capture_source_identity(project_root: Path) -> _SourceIdentity:
    state = git_state(project_root)
    return _SourceIdentity(
        commit=state.commit,
        dirty=state.dirty,
        source_tree_sha256=git_source_tree_sha256(project_root),
    )


def _protocol_source(project_root: Path) -> Path:
    path = project_root / "docs" / "M8_MULTIDATE_TRADE_PROTOCOL.md"
    if not path.is_file():
        raise M8PipelineError(f"frozen M8 protocol is missing: {path}")
    return path


def _atomic_copy_exact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> None:
    expected = _require_digest(expected_sha256, f"SHA-256 for {source}")
    try:
        before_size = source.stat().st_size
    except OSError as exc:
        raise M8PipelineError(f"cannot stat immutable input artifact: {source}") from exc
    if expected_bytes is not None and before_size != expected_bytes:
        raise M8PipelineError(f"immutable input artifact byte count changed: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            while chunk := input_handle.read(1024 * 1024):
                output_handle.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if digest.hexdigest() != expected:
            raise M8PipelineError(f"immutable input artifact changed while copying: {source}")
        if expected_bytes is not None and copied != expected_bytes:
            raise M8PipelineError(f"immutable input artifact changed size while copying: {source}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if sha256_file(destination) != expected:
        raise M8PipelineError(f"frozen evidence copy failed checksum verification: {destination}")


def _load_final_input_manifest(
    config: M8StudyConfig,
    manifest_path: Path,
    manifest_sha256: str,
) -> M8InputManifest:
    expected = _require_digest(manifest_sha256, "M8 input manifest SHA-256")
    explicit = manifest_path.resolve()
    if explicit.parent.name != "_manifests":
        raise M8PipelineError(
            "M8 input manifest must use the canonical <input_root>/_manifests/ location"
        )
    expected_name = f"m8-input.manifest-{expected[:20]}.json"
    if explicit.name != expected_name:
        raise M8PipelineError("M8 input manifest filename is not content-addressed by the SHA")
    if not explicit.is_file():
        raise M8PipelineError(f"M8 input manifest does not exist: {explicit}")
    if sha256_file(explicit) != expected:
        raise M8PipelineError("M8 input manifest bytes do not match the supplied SHA-256")
    input_root = explicit.parent.parent.resolve()
    try:
        manifest = verify_m8_input_manifest(
            config,
            input_root,
            explicit,
            manifest_sha256=expected,
        )
    except M8ManifestError as exc:
        raise M8PipelineError(f"M8 input manifest verification failed: {exc}") from exc
    if manifest.path != explicit or manifest.sha256 != expected:
        raise M8PipelineError("M8 input verifier returned a different manifest authority")
    _validate_manifest_scope(config, manifest)
    return manifest


def _load_raw_manifest(
    config: M8StudyConfig,
    manifest_path: Path,
    manifest_sha256: str,
) -> M8AcquisitionManifest:
    expected = _require_digest(manifest_sha256, "M8 raw acquisition manifest SHA-256")
    explicit = manifest_path.resolve()
    if explicit.name != f"m8-acquisition.manifest-{expected[:20]}.json":
        raise M8PipelineError(
            "M8 raw acquisition manifest filename is not content-addressed by the SHA"
        )
    if not explicit.is_file() or sha256_file(explicit) != expected:
        raise M8PipelineError("M8 raw acquisition manifest bytes do not match the supplied SHA")
    try:
        manifest = read_m8_acquisition_manifest(
            explicit,
            expected_sha256=expected,
            config=config,
        )
    except Exception as exc:
        raise M8PipelineError(f"M8 raw acquisition manifest verification failed: {exc}") from exc
    if manifest.path.resolve() != explicit or manifest.sha256 != expected:
        raise M8PipelineError("M8 raw verifier returned a different manifest authority")
    if (
        manifest.config_sha256 != config.hash
        or manifest.config_source_sha256 != config.source_sha256
        or manifest.protocol_version != config.study.protocol_version
    ):
        raise M8PipelineError("M8 raw acquisition is bound to another protocol/configuration")
    return manifest


def _validate_manifest_scope(config: M8StudyConfig, manifest: M8InputManifest) -> None:
    if (
        manifest.config_sha256 != config.hash
        or manifest.config_source_sha256 != config.source_sha256
        or manifest.protocol_version != config.study.protocol_version
    ):
        raise M8PipelineError("M8 input is bound to a different configuration or protocol")
    if tuple(metadata.symbol for metadata in manifest.symbol_metadata) != config.study.symbols:
        raise M8PipelineError("M8 input does not contain the exact frozen symbol metadata")
    metadata_by_symbol = {metadata.symbol: metadata for metadata in manifest.symbol_metadata}
    if len(metadata_by_symbol) != len(config.study.symbols) or any(
        metadata.status != "TRADING" for metadata in manifest.symbol_metadata
    ):
        raise M8PipelineError("M8 symbol metadata must prove both instruments were TRADING")
    expected = tuple(
        (symbol, period.date, period.role)
        for period in config.periods
        for symbol in config.study.symbols
    )
    observed = tuple((entry.symbol, entry.date, entry.role) for entry in manifest.entries)
    if observed != expected or len(observed) != 8:
        raise M8PipelineError("M8 input does not contain the exact frozen eight-entry calendar")
    if any(
        not entry.complete
        or entry.quality_errors != 0
        or entry.quality_warnings != 0
        or entry.rows < 1
        for entry in manifest.entries
    ):
        raise M8PipelineError(
            "M8 FULL_DATA requires every declared archive complete and free of errors/warnings"
        )
    if any(
        entry.tick_size != metadata_by_symbol[entry.symbol].tick_size
        or entry.lot_size != metadata_by_symbol[entry.symbol].lot_size
        for entry in manifest.entries
    ):
        raise M8PipelineError("M8 archive scales disagree with verified exchange metadata")
    if (
        sum(entry.raw_zip_bytes for entry in manifest.entries)
        > config.study.max_total_download_bytes
    ):
        raise M8PipelineError("M8 raw archives exceed the frozen total-download ceiling")


def _entry_lookup(manifest: M8InputManifest) -> dict[tuple[str, str], M8ArchiveEntry]:
    return {(entry.symbol, entry.date.isoformat()): entry for entry in manifest.entries}


def _normalized_metadata(metadata: M8RawSymbolMetadata) -> M8SymbolMetadata:
    """Project a verified raw metadata descriptor into the final input schema."""

    return M8SymbolMetadata(
        symbol=metadata.symbol,
        status=metadata.status,
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
        observed_ts_ns=metadata.observed_ts_ns,
        raw_path=metadata.raw_path.resolve(),
        raw_sha256=metadata.raw_sha256,
        raw_bytes=metadata.raw_bytes,
        source_uri=metadata.source_uri,
        source_manifest_path=metadata.source_manifest_path.resolve(),
        source_manifest_sha256=metadata.source_manifest_sha256,
        source_manifest_bytes=metadata.source_manifest_bytes,
    )


def _freeze_protocol_and_config(
    config: M8StudyConfig,
    project_root: Path,
    stage: Path,
) -> dict[str, object]:
    _verify_config_source(config)
    protocol = _protocol_source(project_root)
    protocol_sha = sha256_file(protocol)
    config_destination = stage / "protocol" / "m8_multidate_trade_study.toml"
    protocol_destination = stage / "protocol" / "M8_MULTIDATE_TRADE_PROTOCOL.md"
    _atomic_copy_exact(
        config.path,
        config_destination,
        expected_sha256=config.source_sha256,
        expected_bytes=config.path.stat().st_size,
    )
    _atomic_copy_exact(
        protocol,
        protocol_destination,
        expected_sha256=protocol_sha,
        expected_bytes=protocol.stat().st_size,
    )
    return {
        "machine_spec_path": _relative(config_destination, stage),
        "machine_spec_source_sha256": config.source_sha256,
        "config_semantic_sha256": config.hash,
        "protocol_path": _relative(protocol_destination, stage),
        "protocol_sha256": protocol_sha,
        "protocol_version": config.study.protocol_version,
    }


def _snapshot_input_evidence(
    manifest: M8InputManifest,
    stage: Path,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Index stage-local input evidence without creating another raw copy."""

    manifest_copy = stage / "data" / "m8_input_manifest.json"
    _atomic_copy_exact(
        manifest.path,
        manifest_copy,
        expected_sha256=manifest.sha256,
        expected_bytes=manifest.path.stat().st_size,
    )
    hashes: set[str] = {manifest.sha256}
    metadata_rows: list[dict[str, object]] = []
    for metadata in manifest.symbol_metadata:
        hashes.update((metadata.raw_sha256, metadata.source_manifest_sha256))
        metadata_rows.append(
            {
                "symbol": metadata.symbol,
                "status": metadata.status,
                "tick_size": format(metadata.tick_size, "f"),
                "lot_size": format(metadata.lot_size, "f"),
                "observed_ts_ns": metadata.observed_ts_ns,
                "observed_utc": _utc_from_ns(metadata.observed_ts_ns),
                "source_uri": metadata.source_uri,
                "bundle_raw_path": _relative(metadata.raw_path, stage),
                "raw_sha256": metadata.raw_sha256,
                "raw_bytes": metadata.raw_bytes,
                "bundle_source_manifest_path": _relative(metadata.source_manifest_path, stage),
                "source_manifest_sha256": metadata.source_manifest_sha256,
                "source_manifest_bytes": metadata.source_manifest_bytes,
            }
        )
    entries: list[dict[str, object]] = []
    for entry in manifest.entries:
        part_copies: list[dict[str, object]] = []
        for part in entry.normalized_parts:
            hashes.update((part.sidecar_sha256, part.data_sha256))
            part_copies.append(
                {
                    "write_ordinal": part.write_ordinal,
                    "rows": part.rows,
                    "bundle_data_path": _relative(part.data_path, stage),
                    "data_sha256": part.data_sha256,
                    "data_bytes": part.data_bytes,
                    "sidecar_path": _relative(part.sidecar_path, stage),
                    "sidecar_sha256": part.sidecar_sha256,
                }
            )
        hashes.update(
            (
                entry.raw_zip_sha256,
                entry.raw_source_manifest_sha256,
                entry.raw_checksum_sha256,
                entry.raw_checksum_source_manifest_sha256,
                entry.normalized_dataset_manifest_sha256,
                entry.quality_report_sha256,
                entry.quality_findings_sha256,
            )
        )
        entries.append(
            {
                "symbol": entry.symbol,
                "date": entry.date.isoformat(),
                "role": entry.role,
                "complete": entry.complete,
                "rows": entry.rows,
                "trade_id_range": {
                    "first": entry.first_trade_id,
                    "last": entry.last_trade_id,
                    "contiguous": True,
                },
                "observed_range_ns": {
                    "start": entry.observed_start_ns,
                    "end_inclusive": entry.observed_end_inclusive_ns,
                },
                "raw_archive": {
                    "bundle_path": _relative(entry.raw_zip_path, stage),
                    "sha256": entry.raw_zip_sha256,
                    "bytes": entry.raw_zip_bytes,
                    "uncompressed_bytes": entry.raw_uncompressed_bytes,
                    "source_uri": entry.raw_source_uri,
                    "official_checksum": {
                        "bundle_path": _relative(entry.raw_checksum_path, stage),
                        "sha256": entry.raw_checksum_sha256,
                        "bytes": entry.raw_checksum_bytes,
                        "source_uri": entry.raw_checksum_source_uri,
                        "bundle_source_manifest_path": _relative(
                            entry.raw_checksum_source_manifest_path,
                            stage,
                        ),
                        "source_manifest_sha256": (entry.raw_checksum_source_manifest_sha256),
                        "source_manifest_bytes": entry.raw_checksum_source_manifest_bytes,
                    },
                    "source_manifest_path": _relative(entry.raw_source_manifest_path, stage),
                    "source_manifest_sha256": entry.raw_source_manifest_sha256,
                },
                "normalized_dataset_manifest": {
                    "path": _relative(entry.normalized_dataset_manifest_path, stage),
                    "sha256": entry.normalized_dataset_manifest_sha256,
                    "bytes": entry.normalized_dataset_manifest_bytes,
                },
                "normalized_parts": part_copies,
                "quality": {
                    "report_path": _relative(entry.quality_report_path, stage),
                    "report_sha256": entry.quality_report_sha256,
                    "findings_path": _relative(entry.quality_findings_path, stage),
                    "findings_sha256": entry.quality_findings_sha256,
                    "errors": 0,
                    "warnings": 0,
                },
            }
        )
    snapshot = {
        "schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "artifact_kind": "m8_verified_input_snapshot",
        "manifest_authority": {
            "policy": "explicit path plus caller-supplied lowercase SHA-256; no discovery",
            "bundle_path": _relative(manifest.path, stage),
            "sha256": manifest.sha256,
            "frozen_copy_path": _relative(manifest_copy, stage),
        },
        "input_root": _relative(manifest.root, stage),
        "config_sha256": manifest.config_sha256,
        "config_source_sha256": manifest.config_source_sha256,
        "protocol_version": manifest.protocol_version,
        "evidence_tier": M8_EVIDENCE_TIER,
        "evidence_scope": M8_EVIDENCE_SCOPE,
        "symbol_metadata": metadata_rows,
        "entries": entries,
        "self_contained_bundle_input": True,
        "raw_archives_copied_into_run": True,
        "snapshot_created_additional_raw_evidence_copies": False,
        "normalized_rows_are_stage_local": True,
        "note": (
            "Accepted raw archives, official checksums, source sidecars, and metadata are "
            "self-contained below data/input; normalized parts and DQ evidence are isolated "
            "below data/normalized_input. This snapshot only indexes those authoritative "
            "stage-local bytes and creates no further raw copy."
        ),
    }
    _write_json(stage / "data" / "manifest_snapshot.json", snapshot)
    return snapshot, tuple(sorted(hashes))


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


def _trade_feature_columns(config: M8StudyConfig) -> tuple[str, ...]:
    columns: list[str] = ["log_trade_return_1"]
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


def _read_normalized_date(entry: M8ArchiveEntry) -> pl.DataFrame:
    """Read one already-verified date; callers enforce the analysis-lock phase."""

    paths = tuple(part.data_path for part in entry.normalized_parts)
    if not paths:
        raise M8PipelineError(f"no normalized Parquet parts for {entry.symbol}/{entry.date}")
    try:
        frame = pl.read_parquet(list(paths), columns=list(_TRADE_COLUMNS), rechunk=False)
    except Exception as exc:
        raise M8PipelineError(
            f"cannot read normalized M8 date {entry.symbol}/{entry.date}: {exc}"
        ) from exc
    for part in entry.normalized_parts:
        if part.data_path.stat().st_size != part.data_bytes:
            raise M8PipelineError(
                f"normalized part changed size during research read: {part.data_path}"
            )
        if sha256_file(part.data_path) != part.data_sha256:
            raise M8PipelineError(f"normalized part changed during research read: {part.data_path}")
    _validate_normalized_date(frame, entry)
    return frame


def _validate_normalized_date(frame: pl.DataFrame, entry: M8ArchiveEntry) -> None:
    label = f"{entry.symbol}/{entry.date.isoformat()}"
    if frame.height != entry.rows:
        raise M8PipelineError(f"normalized row count disagrees with manifest for {label}")
    if frame.get_column("symbol").null_count() or set(frame.get_column("symbol").unique()) != {
        entry.symbol
    }:
        raise M8PipelineError(f"normalized symbol scope is invalid for {label}")
    expected_continuity = f"binance_spot:{entry.symbol}:{entry.date.isoformat()}"
    continuity = frame.get_column("continuity_id")
    if continuity.null_count() or set(continuity.unique()) != {expected_continuity}:
        raise M8PipelineError(f"normalized continuity does not reset at the UTC date for {label}")
    if not frame.get_column("trade_id").is_sorted():
        raise M8PipelineError(f"normalized trade IDs are not physically ordered for {label}")
    if frame.filter((pl.col("trade_id").diff() != 1).fill_null(False)).height:
        raise M8PipelineError(f"normalized trade IDs contain a gap for {label}")
    first_id = int(cast(int, frame.get_column("trade_id").min()))
    last_id = int(cast(int, frame.get_column("trade_id").max()))
    if (first_id, last_id) != (entry.first_trade_id, entry.last_trade_id):
        raise M8PipelineError(f"normalized trade-ID bounds disagree with manifest for {label}")
    if not frame.get_column("available_ts_ns").is_sorted():
        raise M8PipelineError(f"normalized availability clock reverses for {label}")
    if frame.filter(pl.col("available_ts_ns") < pl.col("event_ts_ns")).height:
        raise M8PipelineError(f"normalized availability precedes event time for {label}")
    if set(frame.get_column("availability_basis").unique()) != {"exchange_event_time_proxy"}:
        raise M8PipelineError(
            f"historical M8 rows have an unsupported availability basis for {label}"
        )
    if frame.get_column("received_ts_ns").null_count() != frame.height:
        raise M8PipelineError(f"historical M8 rows cannot claim local receipt time for {label}")
    observed = (
        int(cast(int, frame.get_column("event_ts_ns").min())),
        int(cast(int, frame.get_column("event_ts_ns").max())),
    )
    if observed != (entry.observed_start_ns, entry.observed_end_inclusive_ns):
        raise M8PipelineError(f"normalized event-time bounds disagree with manifest for {label}")
    day_start, day_end = _day_bounds_ns(entry.date)
    if observed[0] < day_start or observed[1] >= day_end:
        raise M8PipelineError(f"normalized rows escape their UTC date for {label}")


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
        *_trade_feature_columns(config),
    )


def _date_summary(
    entry: M8ArchiveEntry,
    research: pl.DataFrame,
    temporal_audit: Mapping[str, object],
    target: str,
) -> dict[str, object]:
    eligible = research.filter(pl.col("feature_ready") & (~pl.col("right_censored")))
    positive = eligible.filter(pl.col(target) == 1).height
    negative = eligible.filter(pl.col(target) == 0).height
    return {
        "symbol": entry.symbol,
        "date": entry.date.isoformat(),
        "role": entry.role,
        "source_rows": entry.rows,
        "research_rows": research.height,
        "feature_ready_rows": research.filter(pl.col("feature_ready")).height,
        "right_censored_rows": research.filter(pl.col("right_censored")).height,
        "feature_warmup_excluded_rows": research.filter(~pl.col("feature_ready")).height,
        "eligible_labeled_rows": eligible.height,
        "eligible_positive_rows": positive,
        "eligible_negative_rows": negative,
        "eligible_positive_rate": positive / eligible.height if eligible.height else None,
        "observed_start_ts_ns": entry.observed_start_ns,
        "observed_end_inclusive_ts_ns": entry.observed_end_inclusive_ns,
        "observed_start_utc": _utc_from_ns(entry.observed_start_ns),
        "observed_end_inclusive_utc": _utc_from_ns(entry.observed_end_inclusive_ns),
        "first_trade_id": entry.first_trade_id,
        "last_trade_id": entry.last_trade_id,
        "continuity_id": f"binance_spot:{entry.symbol}:{entry.date.isoformat()}",
        "quality_errors": entry.quality_errors,
        "quality_warnings": entry.quality_warnings,
        "temporal_audit": dict(temporal_audit),
        "source_values_repaired": False,
    }


def _build_date_artifacts(
    entry: M8ArchiveEntry,
    config: M8StudyConfig,
    stage: Path,
) -> _DateArtifacts:
    trades = _read_normalized_date(entry)
    research = build_trade_only_research_frame(trades, _feature_config(config)).with_columns(
        pl.lit(entry.date.isoformat()).alias("study_date"),
        pl.lit(entry.role).alias("study_role"),
        pl.col("label_horizon_trades").alias("label_horizon_events"),
    )
    temporal = validate_trade_only_temporal_contract(research)
    missing = sorted(set(_evaluation_columns(config)).difference(research.columns))
    if missing:
        raise ResearchDataError(f"M8 research frame is missing frozen model columns: {missing}")
    evaluation = research.select(_evaluation_columns(config))
    base = stage / "research" / entry.symbol.lower() / entry.date.isoformat()
    research_path = base / "research_frame.parquet"
    evaluation_path = base / "evaluation_frame.parquet"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research.write_parquet(research_path, compression="zstd", statistics=True)
    evaluation.write_parquet(evaluation_path, compression="zstd", statistics=True)
    summary = _date_summary(entry, research, asdict(temporal), config.study.target)
    _write_json(base / "summary.json", summary)
    return _DateArtifacts(
        symbol=entry.symbol,
        study_date=entry.date.isoformat(),
        role=entry.role,
        research_path=research_path,
        evaluation_path=evaluation_path,
        summary=summary,
    )


def _load_evaluation(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise M8PipelineError(f"cannot reload frozen per-date evaluation frame: {path}") from exc


def _select_symbol(
    symbol: str,
    symbol_index: int,
    development: Sequence[_DateArtifacts],
    config: M8StudyConfig,
    stage: Path,
) -> _SelectionArtifacts:
    frames = tuple(_load_evaluation(item.evaluation_path) for item in development)
    test_dates = tuple(
        period.date.isoformat() for period in config.periods if period.role in _TEST_ROLES
    )
    selection = select_multidate_model(
        frames,
        _model_config(config),
        feature_columns=_trade_feature_columns(config),
        declared_test_dates=test_dates,
        seed=config.study.seed + symbol_index * 100_000,
        calibration_bins=config.study.feature_stability_bins,
        target=config.study.target,
        calibration_fraction=config.study.calibration_fraction,
    )
    model_root = stage / "models" / symbol.lower()
    comparison_path = model_root / "validation_candidate_comparison.parquet"
    comparison = selection.validation_comparison.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(symbol).alias("instrument"),
        pl.lit(config.study.label_horizon_events).alias("horizon_events"),
        pl.lit("validation").alias("split"),
        pl.lit(selection.lock.sha256).alias("selection_lock_sha256"),
        pl.lit(selection.fitted_state.sha256).alias("final_fitted_state_sha256"),
        pl.lit(False).alias("significance_claim_authorized"),
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_parquet(comparison_path, compression="zstd", statistics=True)
    fitted_state_path = model_root / "final_fitted_state.json"
    _atomic_write_text(
        fitted_state_path,
        selection.fitted_state.payload_json,
        trailing_newline=False,
    )
    restored_state = _restore_fitted_state_file(
        fitted_state_path,
        selection.fitted_state.sha256,
        f"persisted {symbol} final fitted state",
    )
    if restored_state != selection.fitted_state:
        raise M8PipelineError(f"persisted final fitted state changed for {symbol}")
    lock_path = stage / "analysis" / "locks" / f"{symbol.lower()}.selection_lock.json"
    _atomic_write_text(lock_path, selection.lock.payload_json, trailing_newline=False)
    _restore_analysis_lock_file(
        lock_path,
        selection.lock.sha256,
        f"persisted {symbol} selection lock",
    )
    _fsync_directory(fitted_state_path.parent)
    _fsync_directory(lock_path.parent)
    return _SelectionArtifacts(
        symbol=symbol,
        selection=selection,
        lock_path=lock_path,
        fitted_state_path=fitted_state_path,
        comparison_path=comparison_path,
    )


def _commit_development_manifest(
    entries: Sequence[M8ArchiveEntry],
    date_artifacts: Mapping[tuple[str, str], _DateArtifacts],
    config: M8StudyConfig,
    stage: Path,
) -> tuple[Path, str]:
    expected = tuple(
        (symbol, period.date.isoformat(), period.role)
        for period in config.periods
        if period.role in _DEVELOPMENT_ROLES
        for symbol in config.study.symbols
    )
    lookup = {(entry.symbol, entry.date.isoformat()): entry for entry in entries}
    if tuple((entry.symbol, entry.date.isoformat(), entry.role) for entry in entries) != expected:
        raise M8PipelineError("development normalization did not produce the exact frozen order")
    payload: dict[str, Any] = {
        "schema_version": "m8-development-evidence-v1",
        "artifact_kind": "immutable_development_normalized_manifest",
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_version": config.study.protocol_version,
        "test_member_opened": False,
        "entries": [],
    }
    rows: list[dict[str, object]] = []
    for symbol, study_date, role in expected:
        entry = lookup[(symbol, study_date)]
        artifacts = date_artifacts[(symbol, study_date)]
        rows.append(
            {
                "symbol": symbol,
                "date": study_date,
                "role": role,
                "rows": entry.rows,
                "raw_zip_sha256": entry.raw_zip_sha256,
                "official_checksum_sha256": entry.raw_checksum_sha256,
                "raw_source_manifest_sha256": entry.raw_source_manifest_sha256,
                "official_checksum_source_manifest_sha256": (
                    entry.raw_checksum_source_manifest_sha256
                ),
                "normalized_dataset_manifest_sha256": (entry.normalized_dataset_manifest_sha256),
                "normalized_parts": [
                    {
                        "write_ordinal": part.write_ordinal,
                        "rows": part.rows,
                        "data_sha256": part.data_sha256,
                        "sidecar_sha256": part.sidecar_sha256,
                    }
                    for part in entry.normalized_parts
                ],
                "quality_report_sha256": entry.quality_report_sha256,
                "quality_findings_sha256": entry.quality_findings_sha256,
                "quality_errors": entry.quality_errors,
                "quality_warnings": entry.quality_warnings,
                "research_frame_path": _relative(artifacts.research_path, stage),
                "research_frame_sha256": sha256_file(artifacts.research_path),
                "evaluation_frame_path": _relative(artifacts.evaluation_path, stage),
                "evaluation_frame_sha256": sha256_file(artifacts.evaluation_path),
            }
        )
    payload["entries"] = rows
    encoded = _canonical_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    path = (
        stage
        / "data"
        / "normalized_input"
        / "_manifests"
        / f"m8-development.manifest-{digest[:20]}.json"
    )
    _atomic_write_bytes(path, encoded)
    if sha256_file(path) != digest:
        raise M8PipelineError("development normalized manifest was not durably persisted")
    return path, digest


def _commit_aggregate_lock(
    selections: Sequence[_SelectionArtifacts],
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    development_manifest_path: Path,
    development_manifest_sha256: str,
    protocol_sha256: str,
    source_identity: _SourceIdentity,
    stage: Path,
) -> tuple[Path, str]:
    ordered = tuple(selections)
    if tuple(item.symbol for item in ordered) != config.study.symbols:
        raise M8PipelineError("selection locks are not in the frozen symbol order")
    for item in ordered:
        _restore_lock(item)
        _fsync_directory(item.fitted_state_path.parent)
        _fsync_directory(item.lock_path.parent)
    payload = {
        "schema_version": _ANALYSIS_LOCK_SCHEMA_VERSION,
        "study": config.study.name,
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "source_identity": source_identity.public_dict(),
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
        "development_manifest_path": _relative(development_manifest_path, stage),
        "development_manifest_sha256": development_manifest_sha256,
        "development_dates": [
            period.date.isoformat()
            for period in config.periods
            if period.role in _DEVELOPMENT_ROLES
        ],
        "declared_test_dates": [
            period.date.isoformat() for period in config.periods if period.role in _TEST_ROLES
        ],
        "test_data_opened_before_lock": False,
        "test_economic_rows_materialized_before_lock": False,
        "test_raw_hashes_and_bounded_zip_metadata_verified_before_lock": True,
        "selection_metric": config.study.selection_metric,
        "target": config.study.target,
        "symbols": [
            {
                "symbol": item.symbol,
                "selected_model": item.selection.selected_model,
                "selection_lock_path": _relative(item.lock_path, stage),
                "selection_lock_sha256": item.selection.lock.sha256,
                "final_fitted_state_path": _relative(item.fitted_state_path, stage),
                "final_fitted_state_sha256": item.selection.fitted_state.sha256,
                "development_frame_sha256": item.selection.development_frame_sha256,
                "validation_comparison_path": _relative(item.comparison_path, stage),
                "validation_comparison_rows": item.selection.validation_comparison.height,
            }
            for item in ordered
        ],
        "final_fit_policy": (
            "fit selected specification and independent historical prior once on development "
            "train+validation before this lock; held-out evaluation restores verified numeric "
            "state and performs prediction without fit or update"
        ),
        "claim_permissions": {
            "p_values": False,
            "significance": False,
            "cross_instrument_pooling": False,
            "execution": False,
            "profitability": False,
        },
    }
    encoded = _canonical_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    lock_path = stage / "analysis" / "analysis_lock.json"
    _atomic_write_bytes(lock_path, encoded)
    _atomic_write_text(
        stage / "analysis" / "analysis_lock.sha256",
        f"{digest}  analysis_lock.json",
    )
    if sha256_file(lock_path) != digest:
        raise M8PipelineError("aggregate M8 analysis lock was not durably persisted")
    return lock_path, digest


def _assert_lock_durable(
    lock_path: Path,
    lock_sha256: str,
    selections: Sequence[_SelectionArtifacts],
) -> Mapping[str, Any]:
    aggregate_snapshot = _read_bounded_json_snapshot(
        lock_path,
        label="M8 aggregate lock",
        expected_sha256=lock_sha256,
        ensure_ascii=False,
    )
    aggregate = aggregate_snapshot.payload
    digest_path = lock_path.with_name("analysis_lock.sha256")
    expected_line = f"{lock_sha256}  analysis_lock.json\n"
    _read_exact_bounded_text(
        digest_path,
        label="M8 aggregate lock digest sidecar",
        expected_text=expected_line,
    )
    try:
        development_relative = aggregate["development_manifest_path"]
        development_sha = aggregate["development_manifest_sha256"]
        aggregate_symbols = aggregate["symbols"]
    except (KeyError, TypeError) as exc:
        raise M8PipelineError("M8 aggregate lock payload is invalid before testing") from exc
    if type(development_relative) is not str or type(development_sha) is not str:
        raise M8PipelineError("M8 aggregate lock development binding is invalid")
    stage = lock_path.parent.parent.resolve()
    development_path = _published_file(
        stage,
        development_relative,
        "M8 development evidence",
    )
    _read_bounded_json_snapshot(
        development_path,
        label="M8 development evidence",
        expected_sha256=development_sha,
        ensure_ascii=False,
    )
    if not isinstance(aggregate_symbols, list) or len(aggregate_symbols) != len(selections):
        raise M8PipelineError("M8 aggregate lock has an invalid symbol-state set")
    for raw_symbol, item in zip(aggregate_symbols, selections, strict=True):
        if not isinstance(raw_symbol, Mapping) or any(
            raw_symbol.get(key) != value
            for key, value in {
                "symbol": item.symbol,
                "selection_lock_path": _relative(item.lock_path, stage),
                "selection_lock_sha256": item.selection.lock.sha256,
                "final_fitted_state_path": _relative(item.fitted_state_path, stage),
                "final_fitted_state_sha256": item.selection.fitted_state.sha256,
                "development_frame_sha256": item.selection.development_frame_sha256,
            }.items()
        ):
            raise M8PipelineError(
                f"M8 aggregate lock state claims changed before testing for {item.symbol}"
            )
        _restore_lock(item)
    return aggregate


def _verify_completed_lock_chain(
    *,
    target: Path,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    source_identity: _SourceIdentity,
    protocol_sha256: str,
    run_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    """Rebuild the complete published lock authority from bundle bytes."""

    provenance_path = provenance.get("selection_lock_path")
    provenance_sha = provenance.get("selection_lock_sha256")
    if type(provenance_sha) is not str:
        raise M8PipelineError("completed provenance lacks the aggregate lock SHA-256")
    aggregate_sha = _require_digest(provenance_sha, "completed aggregate lock SHA-256")

    research = run_manifest.get("research")
    if not isinstance(research, Mapping):
        raise M8PipelineError("completed run manifest lacks its research lock authority")
    run_lock = research.get("analysis_lock")
    if not isinstance(run_lock, Mapping):
        raise M8PipelineError("completed run manifest lacks its aggregate lock claim")
    if run_lock.get("committed_before_test_rows_opened") is not True:
        raise M8PipelineError("completed run does not claim a pre-test durable lock")
    if run_lock.get("path") != provenance_path or run_lock.get("sha256") != aggregate_sha:
        raise M8PipelineError("run manifest and provenance aggregate-lock claims differ")

    aggregate_path = _published_file(target, provenance_path, "completed aggregate lock")
    aggregate_snapshot = _read_bounded_json_snapshot(
        aggregate_path,
        label="completed aggregate lock",
        expected_sha256=aggregate_sha,
        ensure_ascii=False,
    )
    aggregate = aggregate_snapshot.payload
    digest_path = aggregate_path.with_name("analysis_lock.sha256")
    _read_exact_bounded_text(
        digest_path,
        label="completed aggregate lock digest sidecar",
        expected_text=f"{aggregate_sha}  analysis_lock.json\n",
    )
    expected_claims: dict[str, object] = {
        "schema_version": _ANALYSIS_LOCK_SCHEMA_VERSION,
        "study": config.study.name,
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "source_identity": source_identity.public_dict(),
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
        "development_dates": [
            period.date.isoformat()
            for period in config.periods
            if period.role in _DEVELOPMENT_ROLES
        ],
        "declared_test_dates": [
            period.date.isoformat() for period in config.periods if period.role in _TEST_ROLES
        ],
        "test_data_opened_before_lock": False,
        "test_economic_rows_materialized_before_lock": False,
        "selection_metric": config.study.selection_metric,
        "target": config.study.target,
    }
    if any(aggregate.get(key) != value for key, value in expected_claims.items()):
        raise M8PipelineError("completed aggregate lock is bound to different authorities")

    development_sha_value = aggregate.get("development_manifest_sha256")
    if type(development_sha_value) is not str:
        raise M8PipelineError("completed aggregate lock lacks a development-manifest SHA")
    development_sha = _require_digest(
        development_sha_value,
        "completed development manifest SHA-256",
    )
    development_path = _published_file(
        target,
        aggregate.get("development_manifest_path"),
        "completed development manifest",
    )
    development = _read_bounded_json_snapshot(
        development_path,
        label="completed development manifest",
        expected_sha256=development_sha,
        ensure_ascii=False,
    ).payload
    if (
        development.get("schema_version") != "m8-development-evidence-v1"
        or development.get("artifact_kind") != "immutable_development_normalized_manifest"
        or development.get("config_sha256") != config.hash
        or development.get("config_source_sha256") != config.source_sha256
        or development.get("protocol_version") != config.study.protocol_version
        or development.get("test_member_opened") is not False
    ):
        raise M8PipelineError("completed development manifest has different authority claims")
    development_entries = development.get("entries")
    expected_development = [
        (symbol, period.date.isoformat(), period.role)
        for period in config.periods
        if period.role in _DEVELOPMENT_ROLES
        for symbol in config.study.symbols
    ]
    if (
        not isinstance(development_entries, Sequence)
        or isinstance(development_entries, (str, bytes))
        or len(development_entries) != len(expected_development)
    ):
        raise M8PipelineError("completed development manifest has an invalid entry set")
    observed_development: list[tuple[object, object, object]] = []
    for entry in development_entries:
        if not isinstance(entry, Mapping):
            raise M8PipelineError("completed development manifest entry is not an object")
        observed_development.append((entry.get("symbol"), entry.get("date"), entry.get("role")))
    if observed_development != expected_development:
        raise M8PipelineError("completed development manifest is outside the frozen order")

    raw_symbols = aggregate.get("symbols")
    if (
        not isinstance(raw_symbols, Sequence)
        or isinstance(raw_symbols, (str, bytes))
        or len(raw_symbols) != len(config.study.symbols)
    ):
        raise M8PipelineError("completed aggregate lock has an invalid symbol-lock set")
    instruments = research.get("instruments")
    if not isinstance(instruments, Mapping):
        raise M8PipelineError("completed run manifest lacks per-symbol lock claims")
    child_paths: list[Path] = []
    state_paths: list[Path] = []
    fitted_state_claims: list[dict[str, str]] = []
    primary_start_ns = _day_bounds_ns(
        next(period.date for period in config.periods if period.role == "primary_test")
    )[0]
    for raw_symbol, expected_symbol in zip(raw_symbols, config.study.symbols, strict=True):
        if not isinstance(raw_symbol, Mapping) or raw_symbol.get("symbol") != expected_symbol:
            raise M8PipelineError("completed child locks are outside the frozen symbol order")
        child_sha_value = raw_symbol.get("selection_lock_sha256")
        if type(child_sha_value) is not str:
            raise M8PipelineError(f"completed {expected_symbol} lock lacks its SHA-256")
        child_sha = _require_digest(
            child_sha_value,
            f"completed {expected_symbol} selection lock SHA-256",
        )
        child_path = _published_file(
            target,
            raw_symbol.get("selection_lock_path"),
            f"completed {expected_symbol} selection lock",
        )
        child_paths.append(child_path)
        child = _restore_analysis_lock_file(
            child_path,
            child_sha,
            f"completed {expected_symbol} selection lock",
        ).payload()
        state_sha_value = raw_symbol.get("final_fitted_state_sha256")
        if type(state_sha_value) is not str:
            raise M8PipelineError(f"completed {expected_symbol} state lacks its SHA-256")
        state_sha = _require_digest(
            state_sha_value,
            f"completed {expected_symbol} final fitted-state SHA-256",
        )
        state_path = _published_file(
            target,
            raw_symbol.get("final_fitted_state_path"),
            f"completed {expected_symbol} final fitted state",
        )
        state_paths.append(state_path)
        state = _restore_fitted_state_file(
            state_path,
            state_sha,
            f"completed {expected_symbol} final fitted state",
        )
        state_payload = state.payload()
        selected = child.get("selected_candidate")
        instrument = instruments.get(expected_symbol)
        if (
            not isinstance(selected, Mapping)
            or not isinstance(instrument, Mapping)
            or selected.get("name") != raw_symbol.get("selected_model")
            or instrument.get("selected_model") != raw_symbol.get("selected_model")
            or instrument.get("selection_lock_sha256") != child_sha
            or instrument.get("final_fitted_state_path")
            != raw_symbol.get("final_fitted_state_path")
            or instrument.get("final_fitted_state_sha256") != state_sha
            or child.get("development_frame_sha256") != raw_symbol.get("development_frame_sha256")
            or child.get("final_fitted_state_sha256") != state_sha
            or child.get("final_fitted_state") != state_payload
            or state_payload.get("development_frame_sha256")
            != raw_symbol.get("development_frame_sha256")
            or state_payload.get("feature_columns") != list(_trade_feature_columns(config))
            or state_payload.get("target") != config.study.target
            or type(state_payload.get("fit_cutoff_ts_ns")) is not int
            or cast(int, state_payload["fit_cutoff_ts_ns"]) >= primary_start_ns
            or child.get("selection_metric") != config.study.selection_metric
            or child.get("target") != config.study.target
            or child.get("train_dates")
            != [period.date.isoformat() for period in config.periods if period.role == "train"]
            or child.get("validation_date")
            != next(
                period.date.isoformat() for period in config.periods if period.role == "validation"
            )
            or child.get("declared_test_dates") != expected_claims["declared_test_dates"]
            or child.get("test_rows_accessed_during_selection") is not False
        ):
            raise M8PipelineError(
                f"completed {expected_symbol} selection lock claims are inconsistent"
            )
        fitted_state_claims.append(
            {
                "symbol": expected_symbol,
                "path": cast(str, raw_symbol["final_fitted_state_path"]),
                "sha256": state_sha,
            }
        )
    if len(set(child_paths)) != len(child_paths):
        raise M8PipelineError("completed aggregate lock reuses a child selection lock")
    if len(set(state_paths)) != len(state_paths):
        raise M8PipelineError("completed aggregate lock reuses a final fitted-state artifact")
    if (
        provenance.get("final_fitted_states") != fitted_state_claims
        or research.get("final_fitted_states") != fitted_state_claims
    ):
        raise M8PipelineError("completed final fitted-state claims differ across authorities")

    artifacts = run_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise M8PipelineError("completed run manifest lacks its artifact index")
    research_manifest_path = _published_file(
        target,
        artifacts.get("research_manifest"),
        "completed research manifest",
    )
    research_manifest = _read_json_object(research_manifest_path, "completed research manifest")
    if (
        research_manifest.get("selection_lock_path") != provenance_path
        or research_manifest.get("selection_lock_sha256") != aggregate_sha
        or research_manifest.get("final_fitted_states") != fitted_state_claims
    ):
        raise M8PipelineError("completed research manifest names a different aggregate lock")
    return {claim["symbol"]: claim["sha256"] for claim in fitted_state_claims}


def _verify_insufficient_failure_taxonomy(
    failure: Mapping[str, Any],
    config: M8StudyConfig,
) -> bool:
    """Validate the stable failure code/stage/role contract without parsing reason text."""

    if failure.get("schema_version") != "m8-insufficient-data-v1":
        raise M8PipelineError("INSUFFICIENT_DATA failure schema is invalid")
    reason_value = failure.get("reason")
    reason_code_value = failure.get("reason_code")
    stage_value = failure.get("failure_stage")
    role_value = failure.get("failed_role")
    if type(reason_value) is not str or not reason_value.strip():
        raise M8PipelineError("INSUFFICIENT_DATA diagnostic reason is invalid")
    if (
        type(stage_value) is not str
        or type(reason_code_value) is not str
        or type(role_value) is not str
    ):
        raise M8PipelineError("INSUFFICIENT_DATA typed failure fields are malformed")
    stage = stage_value
    reason_code = reason_code_value
    role = role_value
    contract: tuple[bool, frozenset[str], frozenset[str]] | None = _FAILURE_STAGE_CONTRACT.get(
        stage
    )
    if contract is None:
        raise M8PipelineError("INSUFFICIENT_DATA failure stage is not recognized")
    expected_after_lock, expected_reason_codes, allowed_roles = contract
    if reason_code not in expected_reason_codes or role not in allowed_roles:
        raise M8PipelineError("INSUFFICIENT_DATA reason code/stage/role are inconsistent")
    if failure.get("failed_after_analysis_lock") is not expected_after_lock:
        raise M8PipelineError("INSUFFICIENT_DATA failure stage disagrees with lock state")

    symbol = failure.get("failed_symbol")
    study_date = failure.get("failed_date")
    if role in _DEVELOPMENT_ROLES or role in _TEST_ROLES:
        if symbol not in config.study.symbols or not any(
            period.date.isoformat() == study_date and period.role == role
            for period in config.periods
        ):
            raise M8PipelineError("INSUFFICIENT_DATA failed symbol/date/role is undeclared")
    elif stage == "final_manifest":
        if symbol != "STUDY" or study_date != "all_dates" or role != "study":
            raise M8PipelineError("INSUFFICIENT_DATA final-manifest identity is invalid")
    elif stage == "locked_evaluation":
        if (
            symbol not in config.study.symbols
            or study_date != "locked_evaluation"
            or role != "all_test_dates"
        ):
            raise M8PipelineError("INSUFFICIENT_DATA locked-evaluation identity is invalid")
    else:
        raise M8PipelineError("INSUFFICIENT_DATA failure identity is invalid")
    return bool(expected_after_lock)


def _verify_insufficient_lock_chain(
    *,
    target: Path,
    inventory: Mapping[str, _FailureEvidenceItem],
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    bundled_raw_sha256: str,
    source_identity: _SourceIdentity,
    protocol_sha256: str,
    failure: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> None:
    """Rebuild a failed bundle's lock authority, or prove it failed before locking."""

    failed_after_lock = _verify_insufficient_failure_taxonomy(failure, config)
    expected_authorities: Mapping[str, object] = {
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol_sha256,
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "bundled_raw_acquisition_manifest_sha256": bundled_raw_sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
    }
    for key, expected in expected_authorities.items():
        if failure.get(key) != expected or provenance.get(key) != expected:
            raise M8PipelineError(f"INSUFFICIENT_DATA failure/provenance have a different {key}")
    if (
        failure.get("source_identity") != source_identity.public_dict()
        or provenance.get("git") != source_identity.public_dict()
    ):
        raise M8PipelineError("INSUFFICIENT_DATA failure/provenance source claims differ")
    if failure.get("generated_at_utc") != provenance.get("generated_at_utc"):
        raise M8PipelineError("INSUFFICIENT_DATA failure/provenance timestamps differ")
    if (
        run_manifest.get("status") != "INSUFFICIENT_DATA"
        or run_manifest.get("evidence_scope") != M8_EVIDENCE_SCOPE
    ):
        raise M8PipelineError("INSUFFICIENT_DATA run manifest has inconsistent status")
    artifacts = run_manifest.get("artifacts")
    research = run_manifest.get("research")
    data_claims = run_manifest.get("data")
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(research, Mapping)
        or not isinstance(data_claims, Mapping)
    ):
        raise M8PipelineError("INSUFFICIENT_DATA run manifest lacks lock authority sections")
    evidence_claim = failure.get("failed_normalization_evidence")
    evidence_completion = (
        evidence_claim.get("evidence_completion") if isinstance(evidence_claim, Mapping) else None
    )
    if (
        provenance.get("failure_reason_code") != failure.get("reason_code")
        or data_claims.get("failure_reason_code") != failure.get("reason_code")
        or provenance.get("failed_normalization_evidence_completion") != evidence_completion
        or data_claims.get("failed_normalization_evidence_completion") != evidence_completion
    ):
        raise M8PipelineError("INSUFFICIENT_DATA typed failure claims differ across records")
    if artifacts.get("failure") != "failure.json" or artifacts.get(
        "raw_acquisition_manifest"
    ) != provenance.get("bundled_raw_acquisition_manifest_path"):
        raise M8PipelineError("INSUFFICIENT_DATA run artifact claims differ from provenance")

    selection_symbols = failure.get("selection_completed_symbols")
    selection_locks = failure.get("selection_locks")
    if (
        not isinstance(selection_symbols, Sequence)
        or isinstance(selection_symbols, (str, bytes))
        or not isinstance(selection_locks, Sequence)
        or isinstance(selection_locks, (str, bytes))
        or any(type(symbol) is not str for symbol in selection_symbols)
    ):
        raise M8PipelineError("INSUFFICIENT_DATA selection progress is malformed")
    selection_symbols_list = list(selection_symbols)
    raw_fitted_state_claims = failure.get("final_fitted_states")
    if (
        not isinstance(raw_fitted_state_claims, Sequence)
        or isinstance(raw_fitted_state_claims, (str, bytes))
        or provenance.get("final_fitted_states") != raw_fitted_state_claims
        or research.get("final_fitted_states") != raw_fitted_state_claims
    ):
        raise M8PipelineError("INSUFFICIENT_DATA final fitted-state claims are malformed")
    fitted_state_claims = list(raw_fitted_state_claims)
    if len(fitted_state_claims) != len(selection_symbols_list) or any(
        not isinstance(claim, Mapping) or claim.get("symbol") != expected_symbol
        for claim, expected_symbol in zip(
            fitted_state_claims,
            selection_symbols_list,
            strict=True,
        )
    ):
        raise M8PipelineError("INSUFFICIENT_DATA fitted states are outside the frozen order")
    primary_start_ns = _day_bounds_ns(
        next(period.date for period in config.periods if period.role == "primary_test")
    )[0]
    expected_selection_started = failed_after_lock or failure.get("failure_stage") == (
        "model_selection"
    )
    if (
        failure.get("selection_started") is not expected_selection_started
        or failure.get("aggregate_lock_committed") is not failed_after_lock
        or failure.get("selection_completed_symbol_count") != len(selection_symbols_list)
        or selection_symbols_list != list(config.study.symbols[: len(selection_symbols_list)])
        or len(selection_locks) != len(selection_symbols_list)
        or research.get("selection_started") is not expected_selection_started
        or research.get("aggregate_lock_committed") is not failed_after_lock
        or research.get("selection_completed_symbols") != selection_symbols_list
        or research.get("selection_completed_symbol_count") != len(selection_symbols_list)
    ):
        raise M8PipelineError("INSUFFICIENT_DATA selection progress claims are inconsistent")
    if failed_after_lock and selection_symbols_list != list(config.study.symbols):
        raise M8PipelineError("post-lock INSUFFICIENT_DATA lacks both completed selections")
    if not expected_selection_started and selection_symbols_list:
        raise M8PipelineError("pre-selection INSUFFICIENT_DATA claims completed selections")
    if failure.get("failure_stage") == "model_selection" and (
        len(selection_symbols_list) >= len(config.study.symbols)
        or failure.get("failed_symbol") != config.study.symbols[len(selection_symbols_list)]
    ):
        raise M8PipelineError("selection failure progress disagrees with the failed symbol")

    evaluation_symbols = failure.get("endpoint_evaluation_completed_symbols")
    if (
        not isinstance(evaluation_symbols, Sequence)
        or isinstance(evaluation_symbols, (str, bytes))
        or any(type(symbol) is not str for symbol in evaluation_symbols)
    ):
        raise M8PipelineError("INSUFFICIENT_DATA endpoint progress is malformed")
    evaluation_symbols_list = list(evaluation_symbols)
    expected_evaluation_started = failure.get("failure_stage") == "locked_evaluation"
    endpoint_claims: Mapping[str, object] = {
        "endpoint_evaluation_performed": expected_evaluation_started,
        "endpoint_evaluation_started": expected_evaluation_started,
        "endpoint_evaluation_completed": False,
        "endpoint_artifacts_published": False,
        "endpoint_evaluation_completed_symbols": evaluation_symbols_list,
        "endpoint_evaluation_completed_symbol_count": len(evaluation_symbols_list),
    }
    if (
        evaluation_symbols_list != list(config.study.symbols[: len(evaluation_symbols_list)])
        or (not expected_evaluation_started and evaluation_symbols_list)
        or len(evaluation_symbols_list) >= len(config.study.symbols)
        or any(failure.get(key) != value for key, value in endpoint_claims.items())
        or any(research.get(key) != value for key, value in endpoint_claims.items())
    ):
        raise M8PipelineError("INSUFFICIENT_DATA endpoint progress claims are inconsistent")
    if (
        expected_evaluation_started
        and failure.get("failed_symbol") != config.study.symbols[len(evaluation_symbols_list)]
    ):
        raise M8PipelineError("locked-evaluation progress disagrees with the failed symbol")

    if not failed_after_lock:
        if (
            failure.get("analysis_lock") is not None
            or failure.get("analysis_lock_path") is not None
            or failure.get("analysis_lock_sha256") is not None
            or provenance.get("selection_lock_path") is not None
            or provenance.get("selection_lock_sha256") is not None
            or research.get("analysis_lock") is not None
            or "analysis_lock" in artifacts
        ):
            raise M8PipelineError("pre-lock INSUFFICIENT_DATA result contains a lock claim")
        if failure.get("held_out_member_opened") is not False:
            raise M8PipelineError("pre-lock INSUFFICIENT_DATA lacks the held-out-open denial")
        if (target / "analysis" / "analysis_lock.json").exists() or (
            target / "analysis" / "analysis_lock.sha256"
        ).exists():
            raise M8PipelineError("pre-lock INSUFFICIENT_DATA contains an aggregate lock")
        partial_paths: list[Path] = []
        partial_state_paths: list[Path] = []
        for lock_claim, state_claim, expected_symbol in zip(
            selection_locks,
            fitted_state_claims,
            selection_symbols_list,
            strict=True,
        ):
            if (
                not isinstance(lock_claim, Mapping)
                or not isinstance(state_claim, Mapping)
                or lock_claim.get("symbol") != expected_symbol
            ):
                raise M8PipelineError("partial selection locks are outside the frozen order")
            lock_sha_value = lock_claim.get("sha256")
            if type(lock_sha_value) is not str:
                raise M8PipelineError(f"partial {expected_symbol} lock lacks its SHA-256")
            lock_sha = _require_digest(
                lock_sha_value,
                f"partial {expected_symbol} selection lock SHA-256",
            )
            lock_relative = _canonical_inventory_child(
                lock_claim.get("path"),
                f"partial {expected_symbol} selection lock",
            )
            lock_path = _published_file(
                target,
                lock_relative,
                f"partial {expected_symbol} selection lock",
            )
            partial_paths.append(lock_path)
            child = _restore_inventory_analysis_lock_file(
                target=target,
                inventory=inventory,
                relative_path=lock_relative,
                expected_sha256=lock_sha,
                label=f"partial {expected_symbol} selection lock",
            ).payload()
            state_sha_value = state_claim.get("sha256")
            if type(state_sha_value) is not str:
                raise M8PipelineError(f"partial {expected_symbol} state lacks its SHA-256")
            state_sha = _require_digest(
                state_sha_value,
                f"partial {expected_symbol} fitted-state SHA-256",
            )
            if (
                lock_claim.get("final_fitted_state_path") != state_claim.get("path")
                or lock_claim.get("final_fitted_state_sha256") != state_sha
            ):
                raise M8PipelineError(f"partial {expected_symbol} lock and state claims differ")
            state_relative = _canonical_inventory_child(
                state_claim.get("path"),
                f"partial {expected_symbol} final fitted state",
            )
            state_path = _published_file(
                target,
                state_relative,
                f"partial {expected_symbol} final fitted state",
            )
            partial_state_paths.append(state_path)
            state = _restore_inventory_fitted_state_file(
                target=target,
                inventory=inventory,
                relative_path=state_relative,
                expected_sha256=state_sha,
                label=f"partial {expected_symbol} final fitted state",
            )
            state_payload = state.payload()
            if (
                child.get("selection_metric") != config.study.selection_metric
                or child.get("target") != config.study.target
                or child.get("final_fitted_state_sha256") != state_sha
                or child.get("final_fitted_state") != state_payload
                or state_payload.get("development_frame_sha256")
                != child.get("development_frame_sha256")
                or state_payload.get("feature_columns") != list(_trade_feature_columns(config))
                or state_payload.get("target") != config.study.target
                or type(state_payload.get("fit_cutoff_ts_ns")) is not int
                or cast(int, state_payload["fit_cutoff_ts_ns"]) >= primary_start_ns
                or child.get("train_dates")
                != [period.date.isoformat() for period in config.periods if period.role == "train"]
                or child.get("validation_date")
                != next(
                    period.date.isoformat()
                    for period in config.periods
                    if period.role == "validation"
                )
                or child.get("declared_test_dates")
                != [
                    period.date.isoformat()
                    for period in config.periods
                    if period.role in _TEST_ROLES
                ]
                or child.get("test_rows_accessed_during_selection") is not False
            ):
                raise M8PipelineError(
                    f"partial {expected_symbol} selection lock claims are inconsistent"
                )
        if len(set(partial_paths)) != len(partial_paths):
            raise M8PipelineError("pre-lock failure reuses a partial child selection lock")
        if len(set(partial_state_paths)) != len(partial_state_paths):
            raise M8PipelineError("pre-lock failure reuses a partial fitted-state artifact")
        return

    failure_path = failure.get("analysis_lock_path")
    failure_sha_value = failure.get("analysis_lock_sha256")
    provenance_path = provenance.get("selection_lock_path")
    provenance_sha_value = provenance.get("selection_lock_sha256")
    if type(failure_sha_value) is not str or type(provenance_sha_value) is not str:
        raise M8PipelineError("post-lock INSUFFICIENT_DATA lacks aggregate lock digests")
    failure_sha = _require_digest(failure_sha_value, "failed aggregate lock SHA-256")
    provenance_sha = _require_digest(provenance_sha_value, "provenance aggregate lock SHA-256")
    if failure_path != provenance_path or failure_sha != provenance_sha:
        raise M8PipelineError("failure and provenance aggregate-lock claims differ")
    if artifacts.get("analysis_lock") != failure_path:
        raise M8PipelineError("failure and run manifest aggregate-lock paths differ")
    run_lock = research.get("analysis_lock")
    if (
        not isinstance(run_lock, Mapping)
        or run_lock.get("path") != failure_path
        or run_lock.get("sha256") != failure_sha
        or run_lock.get("committed_before_test_rows_opened") is not True
    ):
        raise M8PipelineError("failure run manifest has a different aggregate lock claim")

    aggregate_relative = _canonical_inventory_child(failure_path, "failed aggregate lock")
    aggregate_snapshot = _read_inventory_bounded_json_snapshot(
        target=target,
        inventory=inventory,
        relative_path=aggregate_relative,
        label="failed aggregate lock",
        expected_sha256=failure_sha,
        ensure_ascii=False,
    )
    aggregate = aggregate_snapshot.payload
    digest_relative = Path(aggregate_relative).with_name("analysis_lock.sha256").as_posix()
    _read_inventory_exact_bounded_text(
        target=target,
        inventory=inventory,
        relative_path=digest_relative,
        label="failed aggregate lock digest sidecar",
        expected_text=f"{failure_sha}  analysis_lock.json\n",
    )
    expected_lock_claims: Mapping[str, object] = {
        "schema_version": _ANALYSIS_LOCK_SCHEMA_VERSION,
        "study": config.study.name,
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "source_identity": source_identity.public_dict(),
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
        "development_dates": [
            period.date.isoformat()
            for period in config.periods
            if period.role in _DEVELOPMENT_ROLES
        ],
        "declared_test_dates": [
            period.date.isoformat() for period in config.periods if period.role in _TEST_ROLES
        ],
        "test_data_opened_before_lock": False,
        "test_economic_rows_materialized_before_lock": False,
        "test_raw_hashes_and_bounded_zip_metadata_verified_before_lock": True,
        "selection_metric": config.study.selection_metric,
        "target": config.study.target,
    }
    if any(aggregate.get(key) != value for key, value in expected_lock_claims.items()):
        raise M8PipelineError("failed aggregate lock is bound to different authorities")

    development_sha_value = aggregate.get("development_manifest_sha256")
    if type(development_sha_value) is not str:
        raise M8PipelineError("failed aggregate lock lacks a development-manifest SHA")
    development_sha = _require_digest(
        development_sha_value,
        "failed development manifest SHA-256",
    )
    development_relative = _canonical_inventory_child(
        aggregate.get("development_manifest_path"),
        "failed development manifest",
    )
    development = _read_inventory_bounded_json_snapshot(
        target=target,
        inventory=inventory,
        relative_path=development_relative,
        label="failed development manifest",
        expected_sha256=development_sha,
        ensure_ascii=False,
    ).payload
    if (
        development.get("schema_version") != "m8-development-evidence-v1"
        or development.get("artifact_kind") != "immutable_development_normalized_manifest"
        or development.get("config_sha256") != config.hash
        or development.get("config_source_sha256") != config.source_sha256
        or development.get("protocol_version") != config.study.protocol_version
        or development.get("test_member_opened") is not False
    ):
        raise M8PipelineError("failed development manifest has different authority claims")
    development_entries = development.get("entries")
    expected_development = [
        (symbol, period.date.isoformat(), period.role)
        for period in config.periods
        if period.role in _DEVELOPMENT_ROLES
        for symbol in config.study.symbols
    ]
    if (
        not isinstance(development_entries, Sequence)
        or isinstance(development_entries, (str, bytes))
        or len(development_entries) != len(expected_development)
    ):
        raise M8PipelineError("failed development manifest has an invalid entry set")
    observed_development: list[tuple[object, object, object]] = []
    for entry in development_entries:
        if not isinstance(entry, Mapping):
            raise M8PipelineError("failed development manifest entry is not an object")
        observed_development.append((entry.get("symbol"), entry.get("date"), entry.get("role")))
    if observed_development != expected_development:
        raise M8PipelineError("failed development manifest is outside the frozen order")

    raw_symbols = aggregate.get("symbols")
    failure_locks = selection_locks
    if (
        not isinstance(raw_symbols, Sequence)
        or isinstance(raw_symbols, (str, bytes))
        or not isinstance(failure_locks, Sequence)
        or isinstance(failure_locks, (str, bytes))
        or len(raw_symbols) != len(config.study.symbols)
        or len(failure_locks) != len(config.study.symbols)
    ):
        raise M8PipelineError("failed result has an invalid child-lock set")
    child_paths: list[Path] = []
    state_paths: list[Path] = []
    for raw_symbol, failure_lock, state_claim, expected_symbol in zip(
        raw_symbols,
        failure_locks,
        fitted_state_claims,
        config.study.symbols,
        strict=True,
    ):
        if (
            not isinstance(raw_symbol, Mapping)
            or not isinstance(failure_lock, Mapping)
            or not isinstance(state_claim, Mapping)
            or raw_symbol.get("symbol") != expected_symbol
            or failure_lock.get("symbol") != expected_symbol
        ):
            raise M8PipelineError("failed child locks are outside the frozen symbol order")
        child_path_value = raw_symbol.get("selection_lock_path")
        child_sha_value = raw_symbol.get("selection_lock_sha256")
        if type(child_sha_value) is not str:
            raise M8PipelineError(f"failed {expected_symbol} lock lacks its SHA-256")
        child_sha = _require_digest(
            child_sha_value,
            f"failed {expected_symbol} selection lock SHA-256",
        )
        if failure_lock.get("path") != child_path_value or failure_lock.get("sha256") != child_sha:
            raise M8PipelineError(
                f"failure and aggregate {expected_symbol} child-lock claims differ"
            )
        child_relative = _canonical_inventory_child(
            child_path_value,
            f"failed {expected_symbol} selection lock",
        )
        child_path = _published_file(
            target,
            child_relative,
            f"failed {expected_symbol} selection lock",
        )
        child_paths.append(child_path)
        child = _restore_inventory_analysis_lock_file(
            target=target,
            inventory=inventory,
            relative_path=child_relative,
            expected_sha256=child_sha,
            label=f"failed {expected_symbol} selection lock",
        ).payload()
        state_sha_value = raw_symbol.get("final_fitted_state_sha256")
        if type(state_sha_value) is not str:
            raise M8PipelineError(f"failed {expected_symbol} state lacks its SHA-256")
        state_sha = _require_digest(
            state_sha_value,
            f"failed {expected_symbol} final fitted-state SHA-256",
        )
        if (
            raw_symbol.get("final_fitted_state_path") != state_claim.get("path")
            or state_claim.get("sha256") != state_sha
            or failure_lock.get("final_fitted_state_path") != state_claim.get("path")
            or failure_lock.get("final_fitted_state_sha256") != state_sha
        ):
            raise M8PipelineError(f"failed {expected_symbol} final fitted-state claims differ")
        state_relative = _canonical_inventory_child(
            state_claim.get("path"),
            f"failed {expected_symbol} final fitted state",
        )
        state_path = _published_file(
            target,
            state_relative,
            f"failed {expected_symbol} final fitted state",
        )
        state_paths.append(state_path)
        state = _restore_inventory_fitted_state_file(
            target=target,
            inventory=inventory,
            relative_path=state_relative,
            expected_sha256=state_sha,
            label=f"failed {expected_symbol} final fitted state",
        )
        state_payload = state.payload()
        selected = child.get("selected_candidate")
        if (
            not isinstance(selected, Mapping)
            or selected.get("name") != raw_symbol.get("selected_model")
            or child.get("development_frame_sha256") != raw_symbol.get("development_frame_sha256")
            or child.get("final_fitted_state_sha256") != state_sha
            or child.get("final_fitted_state") != state_payload
            or state_payload.get("development_frame_sha256")
            != raw_symbol.get("development_frame_sha256")
            or state_payload.get("feature_columns") != list(_trade_feature_columns(config))
            or state_payload.get("target") != config.study.target
            or type(state_payload.get("fit_cutoff_ts_ns")) is not int
            or cast(int, state_payload["fit_cutoff_ts_ns"]) >= primary_start_ns
            or child.get("selection_metric") != config.study.selection_metric
            or child.get("target") != config.study.target
            or child.get("train_dates")
            != [period.date.isoformat() for period in config.periods if period.role == "train"]
            or child.get("validation_date")
            != next(
                period.date.isoformat() for period in config.periods if period.role == "validation"
            )
            or child.get("declared_test_dates") != expected_lock_claims["declared_test_dates"]
            or child.get("test_rows_accessed_during_selection") is not False
        ):
            raise M8PipelineError(
                f"failed {expected_symbol} selection lock claims are inconsistent"
            )
    if len(set(child_paths)) != len(child_paths):
        raise M8PipelineError("failed aggregate lock reuses a child selection lock")
    if len(set(state_paths)) != len(state_paths):
        raise M8PipelineError("failed aggregate lock reuses a fitted-state artifact")


def _restore_lock(selection: _SelectionArtifacts) -> AnalysisLock:
    lock = _restore_analysis_lock_file(
        selection.lock_path,
        selection.selection.lock.sha256,
        f"persisted {selection.symbol} selection lock",
    )
    lock_payload = lock.payload()
    state = _restore_fitted_state_file(
        selection.fitted_state_path,
        selection.selection.fitted_state.sha256,
        f"persisted {selection.symbol} final fitted state",
    )
    if (
        state != selection.selection.fitted_state
        or lock_payload.get("final_fitted_state_sha256") != state.sha256
        or lock_payload.get("final_fitted_state") != state.payload()
    ):
        raise M8PipelineError(
            f"persisted lock and final fitted state disagree for {selection.symbol}"
        )
    return lock


def _array_sha256(values: NDArray[np.int64]) -> str:
    normalized = np.asarray(values, dtype="<i8")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _plan_payload(result: LockedMultiDateTestResult) -> dict[str, object]:
    plan = result.plan
    return {
        "contract": (
            "date-local features/labels; train predicts validation; locked selected/prior fit "
            "once on train+validation; primary and replication tests receive no update"
        ),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_rows": int(fold.train_indices.size),
                "validation_rows": int(fold.validation_indices.size),
                "train_indices_sha256": _array_sha256(fold.train_indices),
                "validation_indices_sha256": _array_sha256(fold.validation_indices),
                "train_start_ts_ns": fold.train_start_ts_ns,
                "train_end_ts_ns": fold.train_end_ts_ns,
                "validation_start_ts_ns": fold.validation_start_ts_ns,
                "validation_end_ts_ns": fold.validation_end_ts_ns,
                "purged_rows": fold.purged_rows,
                "embargoed_time_buckets": fold.embargoed_time_buckets,
            }
            for fold in plan.folds
        ],
        "final_train_rows": int(plan.final_train_indices.size),
        "test_rows": int(plan.test_indices.size),
        "final_train_indices_sha256": _array_sha256(plan.final_train_indices),
        "test_indices_sha256": _array_sha256(plan.test_indices),
        "test_start_ts_ns": plan.test_start_ts_ns,
        "test_end_ts_ns": plan.test_end_ts_ns,
        "decision_time_count": plan.decision_time_count,
        "test_used_for_selection": False,
        "model_updated_between_test_dates": False,
    }


def _predictive_metric_rows(
    result: LockedMultiDateTestResult,
    config: M8StudyConfig,
) -> tuple[Mapping[str, object], ...]:
    predictions = result.predictions
    rows: list[Mapping[str, object]] = []
    dates = sorted(str(value) for value in predictions.get_column("study_date").unique())
    for study_date in dates:
        current = predictions.filter(pl.col("study_date") == study_date)
        y_true = current.get_column("y_true").to_numpy().astype(np.int64, copy=False)
        period_start = int(cast(int, current.get_column("decision_ts_ns").min()))
        period_end = int(cast(int, current.get_column("decision_ts_ns").max()))
        study_role = str(current.get_column("study_role")[0])
        symbol = str(current.get_column("symbol")[0])
        specifications = (
            (
                "selected",
                result.selected_model,
                "selected_probability",
                str(current.get_column("selected_fit_status")[0]),
                int(current.get_column("selected_fit_cutoff_ts_ns")[0]),
                "validation_log_loss",
            ),
            (
                "historical_prior",
                "historical_prior",
                "prior_probability",
                str(current.get_column("prior_fit_status")[0]),
                int(current.get_column("prior_fit_cutoff_ts_ns")[0]),
                "predeclared_baseline",
            ),
        )
        for role, model, probability_column, fit_status, cutoff, selected_on in specifications:
            probability = (
                current.get_column(probability_column).to_numpy().astype(np.float64, copy=False)
            )
            metrics = classification_metrics(
                y_true,
                probability,
                calibration_bins=config.study.feature_stability_bins,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "instrument": symbol,
                    "study_date": study_date,
                    "study_role": study_role,
                    "test_phase": str(current.get_column("test_phase")[0]),
                    "model_role": role,
                    "model": model,
                    "locked_selected_model": result.selected_model,
                    "horizon_events": config.study.label_horizon_events,
                    "split": "final_test",
                    "n_obs": current.height,
                    "period_start_ts_ns": period_start,
                    "period_end_ts_ns": period_end,
                    "period_start_utc": _utc_from_ns(period_start),
                    "period_end_utc": _utc_from_ns(period_end),
                    "fit_status": fit_status,
                    "fit_cutoff_ts_ns": cutoff,
                    "selected_on": selected_on,
                    "selection_lock_sha256": result.lock_sha256,
                    "test_used_for_selection": False,
                    "model_updated_between_test_dates": False,
                    "significance_claim_authorized": False,
                    **metrics,
                }
            )
    return tuple(rows)


def _endpoint_status(result: LockedMultiDateTestResult) -> str:
    per_date = result.paired_log_loss.per_date.sort("study_date")
    if result.paired_log_loss.aggregate.status != "ok" or per_date.height < 2:
        return "insufficient_data"
    favorable = [bool(value) for value in per_date.get_column("point_favorable").to_list()]
    if all(favorable):
        return "supported"
    if any(favorable):
        return "mixed"
    return "failed"


def _loss_direction(value: float) -> str:
    if not math.isfinite(value):
        return "insufficient_data"
    if value < 0.0:
        return "favorable"
    if value > 0.0:
        return "unfavorable"
    return "tied"


def _three_period_direction_summary(
    selection: _SelectionArtifacts,
    paired_dates: pl.DataFrame,
) -> dict[str, object]:
    """Publish the frozen validation→primary→replication direction diagnostic."""

    comparison = selection.selection.validation_comparison
    selected = comparison.filter(pl.col("selected_on_validation"))
    prior = comparison.filter(pl.col("requested_model") == "historical_prior")
    primary = paired_dates.filter(pl.col("study_role") == "primary_test")
    replication = paired_dates.filter(pl.col("study_role") == "replication_test")
    if selected.height != 1 or prior.height != 1 or primary.height != 1 or replication.height != 1:
        return {
            "validation_primary_replication_status": "insufficient_data",
            "direction_consistent_across_validation_primary_replication": False,
            "favorable_across_validation_primary_replication": False,
        }

    validation_selected = float(selected.get_column("log_loss")[0])
    validation_prior = float(prior.get_column("log_loss")[0])
    validation_delta = validation_selected - validation_prior
    primary_delta = float(primary.get_column("point_delta")[0])
    replication_delta = float(replication.get_column("point_delta")[0])
    values = (validation_delta, primary_delta, replication_delta)
    directions = tuple(_loss_direction(value) for value in values)
    sufficient = all(direction != "insufficient_data" for direction in directions)
    consistent = sufficient and len(set(directions)) == 1
    favorable = sufficient and all(direction == "favorable" for direction in directions)
    if not sufficient:
        status = "insufficient_data"
    elif favorable:
        status = "supported"
    elif consistent and directions[0] in {"unfavorable", "tied"}:
        status = "failed"
    else:
        status = "mixed"
    return {
        "validation_date": selection.selection.validation_date,
        "validation_selected_log_loss": validation_selected,
        "validation_prior_log_loss": validation_prior,
        "validation_point_delta": validation_delta,
        "validation_direction": directions[0],
        "primary_date": str(primary.get_column("study_date")[0]),
        "primary_point_delta": primary_delta,
        "primary_direction": directions[1],
        "replication_date": str(replication.get_column("study_date")[0]),
        "replication_point_delta": replication_delta,
        "replication_direction": directions[2],
        "direction_consistent_across_validation_primary_replication": consistent,
        "favorable_across_validation_primary_replication": favorable,
        "validation_primary_replication_status": status,
    }


def _training_only_stability(
    development_frames: Sequence[pl.DataFrame],
    test_frames: Sequence[pl.DataFrame],
    *,
    feature_columns: Sequence[str],
    bins: int,
    lock_sha256: str,
) -> pl.DataFrame:
    train_candidates = [
        frame
        for frame in development_frames
        if set(str(value) for value in frame.get_column("study_role").unique()) == {"train"}
    ]
    if len(train_candidates) != 1:
        raise ResearchDataError("feature stability requires exactly one frozen training date")
    reference = train_candidates[0].filter(pl.col("feature_ready") & (~pl.col("right_censored")))
    if reference.is_empty():
        raise ResearchDataError("feature stability training reference is empty")
    outputs: list[pl.DataFrame] = []
    for frame in test_frames:
        comparison = frame.filter(pl.col("feature_ready") & (~pl.col("right_censored")))
        if comparison.is_empty():
            raise ResearchDataError("feature stability test comparison is empty")
        study_date = str(comparison.get_column("study_date")[0])
        role = str(comparison.get_column("study_role")[0])
        outputs.append(
            feature_stability_summary(
                reference,
                comparison,
                feature_columns=feature_columns,
                group_columns=("symbol",),
                bins=bins,
            ).with_columns(
                pl.lit(str(reference.get_column("study_date")[0])).alias("reference_study_date"),
                pl.lit("train_only").alias("reference_role"),
                pl.lit(study_date).alias("comparison_study_date"),
                pl.lit(role).alias("comparison_study_role"),
                pl.lit("primary" if role == "primary_test" else "replication").alias("test_phase"),
                pl.lit(True).alias("reference_bins_fit_without_test"),
                pl.lit(lock_sha256).alias("selection_lock_sha256"),
            )
        )
    return pl.concat(outputs, how="vertical").sort("comparison_study_date", "symbol", "feature")


def _evaluate_symbol(
    symbol: str,
    selection: _SelectionArtifacts,
    development: Sequence[_DateArtifacts],
    tests: Sequence[_DateArtifacts],
    config: M8StudyConfig,
    stage: Path,
) -> _EvaluationArtifacts:
    development_frames = tuple(_load_evaluation(item.evaluation_path) for item in development)
    test_frames = tuple(_load_evaluation(item.evaluation_path) for item in tests)
    restored = _restore_lock(selection)
    result = evaluate_locked_multidate_tests(development_frames, test_frames, restored)
    if result.lock_sha256 != selection.selection.lock.sha256:
        raise M8PipelineError(f"locked evaluation returned a different lock for {symbol}")
    if result.predictions.filter(pl.col("model_updated_between_test_dates")).height:
        raise M8PipelineError(f"locked M8 model updated during test dates for {symbol}")
    expected_dates = tuple(
        period.date.isoformat() for period in config.periods if period.role in _TEST_ROLES
    )
    observed_dates = tuple(
        sorted(str(value) for value in result.predictions["study_date"].unique())
    )
    if observed_dates != expected_dates:
        raise M8PipelineError(f"locked predictions omit or add a test date for {symbol}")

    model_root = stage / "models" / symbol.lower()
    predictions = result.predictions.with_columns(
        pl.lit(symbol).alias("instrument"),
        pl.lit(config.study.label_horizon_events).alias("horizon_events"),
        pl.lit("selected_and_historical_prior_only").alias("prediction_scope"),
        pl.lit(False).alias("execution_claim_authorized"),
        pl.lit(False).alias("profitability_claim_authorized"),
    )
    predictions_path = model_root / "selected_and_prior_test_predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(predictions_path, compression="zstd", statistics=True)
    for item in tests:
        date_path = model_root / item.study_date / "selected_and_prior_predictions.parquet"
        date_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.filter(pl.col("study_date") == item.study_date).write_parquet(
            date_path,
            compression="zstd",
            statistics=True,
        )

    status = _endpoint_status(result)
    paired = result.paired_log_loss.per_date.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(symbol).alias("instrument"),
        pl.lit(result.selected_model).alias("selected_model"),
        pl.lit("historical_prior").alias("baseline"),
        pl.lit(status).alias("instrument_status"),
        pl.lit(result.lock_sha256).alias("selection_lock_sha256"),
        pl.lit(False).alias("p_value_computed"),
        pl.lit(False).alias("significance_claim_authorized"),
    )
    paired_date_path = stage / "metrics" / symbol.lower() / "paired_log_loss_by_date.parquet"
    paired_date_path.parent.mkdir(parents=True, exist_ok=True)
    paired.write_parquet(paired_date_path, compression="zstd", statistics=True)

    aggregate = result.paired_log_loss.aggregate
    direction_summary = _three_period_direction_summary(selection, paired)
    aggregate_row: dict[str, object] = {
        "symbol": symbol,
        "instrument": symbol,
        "selected_model": result.selected_model,
        "baseline": "historical_prior",
        "metric": "log_loss",
        "delta_definition": "selected_model_minus_historical_prior",
        "date_weighting": "equal",
        "point_delta": aggregate.point_estimate,
        "ci_low": aggregate.lower,
        "ci_high": aggregate.upper,
        "n_obs": int(cast(int, paired.get_column("n_obs").sum())),
        "n_dates": paired.height,
        "n_blocks": aggregate.n_blocks,
        "samples": aggregate.n_bootstrap,
        "seed": aggregate.seed,
        "bootstrap_status": aggregate.status,
        "block_width_events": config.study.bootstrap_block_events,
        "status": status,
        "replication_status": result.paired_log_loss.replication_status,
        "directionally_replicated": status == "supported",
        "selection_lock_sha256": result.lock_sha256,
        "p_value": None,
        "p_value_computed": False,
        "h0_rejected": False,
        "significance_claim_authorized": False,
        "cross_instrument_pooling": False,
        "execution_claim_authorized": False,
        "profitability_claim_authorized": False,
        **direction_summary,
    }

    stability = _training_only_stability(
        development_frames,
        test_frames,
        feature_columns=_trade_feature_columns(config),
        bins=config.study.feature_stability_bins,
        lock_sha256=result.lock_sha256,
    )
    stability_path = stage / "analysis" / symbol.lower() / "feature_stability.parquet"
    stability_path.parent.mkdir(parents=True, exist_ok=True)
    stability.write_parquet(stability_path, compression="zstd", statistics=True)
    plan_path = stage / "research" / symbol.lower() / "walk_forward_plan.json"
    _write_json(plan_path, _plan_payload(result))
    return _EvaluationArtifacts(
        symbol=symbol,
        selected_model=result.selected_model,
        predictions_path=predictions_path,
        paired_date_path=paired_date_path,
        stability_path=stability_path,
        plan_path=plan_path,
        predictive_rows=_predictive_metric_rows(result, config),
        paired_date_rows=tuple(cast(Mapping[str, object], row) for row in paired.to_dicts()),
        aggregate_row=aggregate_row,
    )


def _combine_parquet(paths: Sequence[Path], destination: Path) -> None:
    if not paths:
        raise M8PipelineError(f"cannot create empty combined Parquet artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lazy = pl.concat([pl.scan_parquet(path) for path in paths], how="vertical_relaxed")
    lazy.sink_parquet(destination, compression="zstd", statistics=True, maintain_order=True)


def _publish_evaluation_artifacts(
    evaluations: Sequence[_EvaluationArtifacts],
    evaluation_stage: Path,
    stage: Path,
) -> tuple[_EvaluationArtifacts, ...]:
    """Publish endpoint files only after every symbol evaluates successfully."""

    published: list[_EvaluationArtifacts] = []

    def publish(path: Path) -> Path:
        try:
            relative = path.resolve().relative_to(evaluation_stage.resolve())
        except ValueError as exc:
            raise M8PipelineError("evaluation artifact escaped its isolated staging root") from exc
        destination = stage / relative
        if destination.exists():
            raise M8PipelineError(f"endpoint publication would overwrite: {destination}")
        _atomic_copy_exact(
            path,
            destination,
            expected_sha256=sha256_file(path),
            expected_bytes=path.stat().st_size,
        )
        return destination

    for item in evaluations:
        published.append(
            _EvaluationArtifacts(
                symbol=item.symbol,
                selected_model=item.selected_model,
                predictions_path=publish(item.predictions_path),
                paired_date_path=publish(item.paired_date_path),
                stability_path=publish(item.stability_path),
                plan_path=publish(item.plan_path),
                predictive_rows=item.predictive_rows,
                paired_date_rows=item.paired_date_rows,
                aggregate_row=item.aggregate_row,
            )
        )
    shutil.rmtree(evaluation_stage)
    return tuple(published)


def _write_reports(stage: Path) -> None:
    bundle = load_run_bundle(stage, require_complete=False, verify_integrity=False)
    report_root = stage / "reports"
    _atomic_write_text(report_root / "technical_report.md", render_technical_report(bundle))
    _atomic_write_text(report_root / "executive_memo.md", render_executive_memo(bundle))
    _atomic_write_text(
        report_root / "model_comparison.md",
        render_model_comparison_report(bundle),
    )


def _final_fitted_state_claims(
    selections: Sequence[_SelectionArtifacts],
    stage: Path,
) -> list[dict[str, str]]:
    return [
        {
            "symbol": item.symbol,
            "path": _relative(item.fitted_state_path, stage),
            "sha256": item.selection.fitted_state.sha256,
        }
        for item in selections
    ]


def _final_fitted_state_sha256_by_symbol(
    selections: Sequence[_SelectionArtifacts],
) -> dict[str, str]:
    return {item.symbol: item.selection.fitted_state.sha256 for item in selections}


def _run_key(
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    normalized_manifest: M8InputManifest,
    protocol: Mapping[str, object],
    source: _SourceIdentity,
    final_fitted_state_sha256_by_symbol: Mapping[str, str],
) -> tuple[str, dict[str, object]]:
    inputs: dict[str, object] = {
        "pipeline_schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "study": config.study.name,
        "protocol_version": config.study.protocol_version,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
        "normalized_input_manifest_sha256": normalized_manifest.sha256,
        "protocol_sha256": protocol["protocol_sha256"],
        "final_fitted_state_sha256_by_symbol": dict(final_fitted_state_sha256_by_symbol),
        "git": source.public_dict(),
        "seed": config.study.seed,
        "evidence_scope": M8_EVIDENCE_SCOPE,
    }
    return _stable_sha256(inputs), inputs


def _quality_payload(
    manifest: M8InputManifest,
    date_artifacts: Sequence[_DateArtifacts],
    *,
    generated_at_utc: str,
) -> dict[str, object]:
    summaries = [dict(item.summary) for item in date_artifacts]
    return {
        "generated_at_utc": generated_at_utc,
        "dataset": "m8_complete_binance_daily_aggregate_trades",
        "rows_checked": sum(entry.rows for entry in manifest.entries),
        "summary": {"errors": 0, "warnings": 0},
        "all_eight_archives_complete": True,
        "raw_acquisition_manifest_and_hashes_verified_before_economic_reads": True,
        "final_normalized_manifest_verified_before_endpoint_evaluation": True,
        "date_local_continuity_resets_verified": True,
        "aggregate_trade_ids_contiguous_within_each_symbol_date": True,
        "availability_clock_nondecreasing": True,
        "availability_basis": "exchange_event_time_proxy_plus_trade_id_tie_break",
        "local_receipt_time_available": False,
        "per_symbol_date": summaries,
        "source_values_repaired": False,
        "mutation_policy": "questionable observations are never silently repaired",
    }


def _normalization_evidence_claim(entry: M8ArchiveEntry) -> dict[str, object]:
    """Serialize one complete normalization without treating it as accepted study input."""

    return {
        "symbol": entry.symbol,
        "date": entry.date.isoformat(),
        "role": entry.role,
        "rows": entry.rows,
        "raw_zip_sha256": entry.raw_zip_sha256,
        "normalized_dataset_manifest_sha256": entry.normalized_dataset_manifest_sha256,
        "quality_errors": entry.quality_errors,
        "quality_warnings": entry.quality_warnings,
    }


def _failed_normalization_evidence_payload(
    stage: Path,
    failure: _TypedInsufficientFailure,
) -> dict[str, object] | None:
    """Bind the exact failed symbol/date subtree with a typed completion state."""

    kind = failure.normalization_failure_kind
    completion = failure.normalization_evidence_completion
    complete_entry = failure.normalization_completed_evidence
    if kind is None or completion is None:
        if complete_entry is not None:
            raise M8PipelineError("non-normalization failure retained a normalization entry")
        return None

    normalized_prefix = f"data/normalized_input/normalized/{failure.symbol}/{failure.study_date}"
    quality_prefix = f"data/normalized_input/quality/{failure.symbol}/{failure.study_date}"
    scoped_roots = (stage / normalized_prefix, stage / quality_prefix)
    files: list[Path] = []
    for root in scoped_roots:
        if root.is_symlink():
            raise M8PipelineError("failed normalization evidence root is a symbolic link")
        if not root.exists():
            continue
        if not root.is_dir():
            raise M8PipelineError("failed normalization evidence root is not a directory")
        for path in root.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)):
                raise M8PipelineError("failed normalization evidence is not a regular tree")
            if stat.S_ISREG(mode):
                files.append(path)
    artifacts = [
        {
            "path": path.relative_to(stage).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]
    scoped_paths = {cast(str, item["path"]) for item in artifacts}
    dataset_manifests = {
        path
        for path in scoped_paths
        if Path(path).parent.as_posix() == f"{normalized_prefix}/_manifests"
        and re.fullmatch(r"trades\.manifest-[0-9a-f]{20}\.json", Path(path).name)
    }
    report_path = f"{quality_prefix}/report.json"
    findings_path = f"{quality_prefix}/findings.jsonl"
    if completion == "PARTIAL_STREAM":
        if kind != "PAYLOAD_OR_CONTINUITY" or complete_entry is not None:
            raise M8PipelineError("partial failed normalization has an invalid typed state")
        if dataset_manifests or report_path in scoped_paths or findings_path in scoped_paths:
            raise M8PipelineError("partial failed normalization contains final evidence")
        normalization_claim: dict[str, object] | None = None
    else:
        if kind not in {"QUALITY_GATE", "POSTWRITE_CONSISTENCY"}:
            raise M8PipelineError("complete failed normalization has an invalid typed state")
        if len(dataset_manifests) != 1 or not {report_path, findings_path}.issubset(scoped_paths):
            raise M8PipelineError("complete failed normalization lacks final evidence")
        if kind == "QUALITY_GATE":
            if complete_entry is None:
                raise M8PipelineError("quality-gate failure lacks its structured evidence")
            normalization_claim = _normalization_evidence_claim(complete_entry)
        else:
            if complete_entry is not None:
                raise M8PipelineError("postwrite failure unexpectedly claims accepted evidence")
            normalization_claim = None

    return {
        "schema_version": "m8-failed-normalization-evidence-v1",
        "failure_kind": kind,
        "evidence_completion": completion,
        "normalized_prefix": normalized_prefix,
        "quality_prefix": quality_prefix,
        "artifacts": artifacts,
        "complete_normalization": normalization_claim,
    }


def _assert_test_open_authority(
    *,
    config: M8StudyConfig,
    project_root: Path,
    source_identity: _SourceIdentity,
    protocol_sha256: str,
    authority_manifest: M8AcquisitionManifest,
    stage_manifest: M8AcquisitionManifest,
    aggregate_lock_path: Path,
    aggregate_lock_sha256: str,
    selections: Sequence[_SelectionArtifacts],
) -> None:
    """Revalidate all immutable authorities at the actual member-open boundary."""

    aggregate = _assert_lock_durable(
        aggregate_lock_path,
        aggregate_lock_sha256,
        selections,
    )
    expected_lock_claims: dict[str, object] = {
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "raw_acquisition_manifest_sha256": authority_manifest.sha256,
        "raw_evidence_content_identity_sha256": authority_manifest.content_identity_sha256,
        "source_identity": source_identity.public_dict(),
        "test_data_opened_before_lock": False,
        "test_economic_rows_materialized_before_lock": False,
    }
    if any(aggregate.get(key) != value for key, value in expected_lock_claims.items()):
        raise M8PipelineError("aggregate lock authority claims changed before held-out open")
    _verify_config_source(config)
    if sha256_file(_protocol_source(project_root)) != protocol_sha256:
        raise M8PipelineError("frozen M8 protocol changed before held-out member open")
    current_source = _capture_source_identity(project_root)
    if (
        current_source != source_identity
        or current_source.dirty
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", current_source.commit) is None
    ):
        raise M8PipelineError("Git/source-tree identity changed before held-out member open")
    observed_authority = _load_raw_manifest(
        config,
        authority_manifest.path,
        authority_manifest.sha256,
    )
    try:
        observed_stage = read_m8_acquisition_manifest(
            stage_manifest.path,
            expected_sha256=stage_manifest.sha256,
            config=config,
        )
    except Exception as exc:
        raise M8PipelineError(
            f"bundled raw acquisition evidence changed before held-out member open: {exc}"
        ) from exc
    if (
        observed_authority.total_accepted_zip_bytes != observed_stage.total_accepted_zip_bytes
        or observed_authority.total_raw_evidence_bytes != observed_stage.total_raw_evidence_bytes
        or observed_authority.content_identity_sha256 != observed_stage.content_identity_sha256
        or observed_stage.copied_from_manifest_sha256 != authority_manifest.sha256
        or observed_authority.protocol_document_sha256 != protocol_sha256
        or observed_stage.protocol_document_sha256 != protocol_sha256
    ):
        raise M8PipelineError(
            "bundled raw evidence differs from the explicit acquisition authority"
        )


def _publish_insufficient_data(
    *,
    config: M8StudyConfig,
    stage: Path,
    project_root: Path,
    source_identity: _SourceIdentity,
    protocol: Mapping[str, object],
    authority_manifest: M8AcquisitionManifest,
    stage_manifest: M8AcquisitionManifest,
    selections: Sequence[_SelectionArtifacts],
    aggregate_lock_path: Path,
    aggregate_lock_sha256: str,
    completed_entries: Sequence[M8ArchiveEntry],
    failure: _TypedInsufficientFailure,
    generated_at_utc: str,
    final_manifest: M8InputManifest | None = None,
    endpoint_evaluation_started: bool = False,
    completed_evaluation_symbols: Sequence[str] = (),
) -> None:
    """Finalize a terminal, immutable failed-result bundle after lock exposure."""

    _assert_lock_durable(aggregate_lock_path, aggregate_lock_sha256, selections)
    final_fitted_state_claims = _final_fitted_state_claims(selections, stage)
    _verify_config_source(config)
    if sha256_file(_protocol_source(project_root)) != protocol["protocol_sha256"]:
        raise M8PipelineError("protocol changed while preserving insufficient-data evidence")
    if _capture_source_identity(project_root) != source_identity:
        raise M8PipelineError("source identity changed while preserving insufficient-data evidence")
    forbidden = (
        stage / "models" / "predictions.parquet",
        stage / "metrics" / "predictive_metrics.json",
        stage / "metrics" / "hypothesis_evaluation.json",
        stage / "metrics" / "paired_log_loss_by_date.parquet",
        stage / "metrics" / "equal_date_hypothesis.parquet",
    )
    if any(path.exists() for path in forbidden):
        raise M8PipelineError("endpoint artifacts exist in an insufficient-data attempt")
    if any(path.is_file() for path in stage.rglob("*prediction*")):
        raise M8PipelineError("prediction artifacts exist in an insufficient-data attempt")
    declared_order = [
        (symbol, period.date.isoformat(), period.role)
        for period in config.periods
        for symbol in config.study.symbols
    ]
    completed = [
        {
            "symbol": entry.symbol,
            "date": entry.date.isoformat(),
            "role": entry.role,
            "rows": entry.rows,
            "raw_zip_sha256": entry.raw_zip_sha256,
            "normalized_dataset_manifest_sha256": (entry.normalized_dataset_manifest_sha256),
            "quality_errors": entry.quality_errors,
            "quality_warnings": entry.quality_warnings,
        }
        for entry in completed_entries
    ]
    failed_key = next(
        (
            item
            for item in declared_order
            if item[0] == failure.symbol and item[1] == failure.study_date
        ),
        None,
    )
    failed_index = (
        declared_order.index(failed_key) if failed_key is not None else len(declared_order)
    )
    stopped_before = [
        {"symbol": symbol, "date": study_date, "role": role}
        for symbol, study_date, role in declared_order[failed_index + 1 :]
    ]
    failed_normalization_evidence = _failed_normalization_evidence_payload(stage, failure)
    failure_payload = {
        "schema_version": "m8-insufficient-data-v1",
        "status": "INSUFFICIENT_DATA",
        "terminal": True,
        "generated_at_utc": generated_at_utc,
        "reason": failure.reason,
        "reason_code": failure.reason_code,
        "failure_stage": failure.failure_stage,
        "failed_symbol": failure.symbol,
        "failed_date": failure.study_date,
        "failed_role": failure.failed_role,
        "failed_normalization_evidence": failed_normalization_evidence,
        "failed_after_analysis_lock": True,
        "replacement_date_selected": False,
        "reselection_performed": False,
        "endpoint_evaluation_performed": endpoint_evaluation_started,
        "endpoint_evaluation_started": endpoint_evaluation_started,
        "endpoint_evaluation_completed": False,
        "endpoint_artifacts_published": False,
        "endpoint_evaluation_completed_symbols": list(completed_evaluation_symbols),
        "endpoint_evaluation_completed_symbol_count": len(completed_evaluation_symbols),
        "predictions_published": False,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol["protocol_sha256"],
        "raw_acquisition_manifest_sha256": authority_manifest.sha256,
        "bundled_raw_acquisition_manifest_sha256": stage_manifest.sha256,
        "raw_evidence_content_identity_sha256": (authority_manifest.content_identity_sha256),
        "source_identity": source_identity.public_dict(),
        "analysis_lock_path": _relative(aggregate_lock_path, stage),
        "analysis_lock_sha256": aggregate_lock_sha256,
        "aggregate_lock_committed": True,
        "selection_started": True,
        "selection_completed_symbols": [selection.symbol for selection in selections],
        "selection_completed_symbol_count": len(selections),
        "selection_locks": [
            {
                "symbol": selection.symbol,
                "path": _relative(selection.lock_path, stage),
                "sha256": selection.selection.lock.sha256,
                "final_fitted_state_path": _relative(
                    selection.fitted_state_path,
                    stage,
                ),
                "final_fitted_state_sha256": selection.selection.fitted_state.sha256,
            }
            for selection in selections
        ],
        "final_fitted_states": final_fitted_state_claims,
        "completed_normalizations": completed,
        "stopped_before": stopped_before,
        "final_all_date_normalized_manifest": (
            None
            if final_manifest is None
            else {
                "path": _relative(final_manifest.path, stage),
                "sha256": final_manifest.sha256,
            }
        ),
    }
    _write_json(stage / "failure.json", failure_payload)
    resolved = config.public_dict()
    resolved.update(
        {
            "effective_evidence_tier": "INSUFFICIENT_DATA",
            "evidence_scope": M8_EVIDENCE_SCOPE,
            "protocol": dict(protocol),
            "raw_acquisition_manifest_sha256": authority_manifest.sha256,
            "execution_status": "NOT_RUN",
        }
    )
    _write_json(stage / "resolved_config.json", resolved)
    provenance = {
        "generated_at_utc": generated_at_utc,
        "status": "INSUFFICIENT_DATA",
        "failure_reason_code": failure.reason_code,
        "failed_normalization_evidence_completion": (failure.normalization_evidence_completion),
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "raw_acquisition_manifest_path": str(authority_manifest.path.resolve()),
        "raw_acquisition_manifest_sha256": authority_manifest.sha256,
        "bundled_raw_acquisition_manifest_path": _relative(stage_manifest.path, stage),
        "bundled_raw_acquisition_manifest_sha256": stage_manifest.sha256,
        "raw_evidence_content_identity_sha256": authority_manifest.content_identity_sha256,
        "raw_evidence_byte_budget": {
            "external_total": authority_manifest.total_raw_evidence_bytes,
            "bundle_copy_total": stage_manifest.total_raw_evidence_bytes,
            "combined_total": (
                authority_manifest.total_raw_evidence_bytes
                + stage_manifest.total_raw_evidence_bytes
            ),
            "ceiling": config.study.max_total_download_bytes,
        },
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_version": config.study.protocol_version,
        "selection_lock_path": _relative(aggregate_lock_path, stage),
        "selection_lock_sha256": aggregate_lock_sha256,
        "final_fitted_states": final_fitted_state_claims,
        "git": source_identity.public_dict(),
        "runtime": runtime_metadata(),
        "pipeline_schema_version": M8_PIPELINE_SCHEMA_VERSION,
    }
    _write_json(stage / "provenance.json", provenance)
    run_manifest = {
        "schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "run_id": config.study.name,
        "status": "INSUFFICIENT_DATA",
        "evidence_tier": "INSUFFICIENT_DATA",
        "evidence_scope": M8_EVIDENCE_SCOPE,
        "data": {
            "mode": "binance_spot_daily_aggtrades_trade_only",
            "symbols": list(config.study.symbols),
            "all_requested_ranges_complete": False,
            "completed_normalizations": completed,
            "failure_reason_code": failure.reason_code,
            "failed_normalization_evidence_completion": (failure.normalization_evidence_completion),
            "failed_symbol": failure.symbol,
            "failed_date": failure.study_date,
            "stopped_before": stopped_before,
        },
        "artifacts": {
            "failure": "failure.json",
            "analysis_lock": _relative(aggregate_lock_path, stage),
            "raw_acquisition_manifest": _relative(stage_manifest.path, stage),
            "resolved_config": "resolved_config.json",
        },
        "research": {
            "scope": "trade_only",
            "analysis_lock": {
                "path": _relative(aggregate_lock_path, stage),
                "sha256": aggregate_lock_sha256,
                "committed_before_test_rows_opened": True,
            },
            "aggregate_lock_committed": True,
            "selection_started": True,
            "selection_completed_symbols": [selection.symbol for selection in selections],
            "selection_completed_symbol_count": len(selections),
            "final_fitted_states": final_fitted_state_claims,
            "endpoint_status": "insufficient_data",
            "endpoint_evaluation_performed": endpoint_evaluation_started,
            "endpoint_evaluation_started": endpoint_evaluation_started,
            "endpoint_evaluation_completed": False,
            "endpoint_artifacts_published": False,
            "endpoint_evaluation_completed_symbols": list(completed_evaluation_symbols),
            "endpoint_evaluation_completed_symbol_count": len(completed_evaluation_symbols),
            "reselection_performed": False,
            "replacement_date_selected": False,
            "instruments": {
                symbol: {
                    "status": "insufficient_data",
                    "validation_primary_replication_status": "insufficient_data",
                }
                for symbol in config.study.symbols
            },
        },
        "execution_assumptions": {
            "status": "NOT_RUN",
            "reason": M8_EXECUTION_EXCLUSION_REASON,
            "pnl_calculated": False,
            "fills_calculated": False,
            "capacity_calculated": False,
            "profitability_claim_authorized": False,
        },
    }
    _write_json(stage / "run_manifest.json", run_manifest)
    _atomic_write_text(
        stage / "reports" / "insufficient_data.md",
        (
            "# M8 study result: INSUFFICIENT_DATA\n\n"
            f"The predeclared {failure.symbol}/{failure.study_date} archive failed the frozen "
            f"data contract after the analysis lock: {failure.reason}\n\n"
            "No replacement date or reselection occurred. No prediction or endpoint artifact "
            "was published; any staged evaluation work was discarded. No execution, P&L, or "
            "significance result was published.\n"
        ),
    )
    inventory = [
        {
            "path": path.relative_to(stage).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    ]
    _write_json(stage / "data" / "failure_evidence_inventory.json", inventory)
    _fsync_tree(stage)
    write_checksum_manifest(stage)
    _create_terminal_marker(stage / "INSUFFICIENT_DATA", "terminal")


def _publish_prelock_insufficient_data(
    *,
    config: M8StudyConfig,
    stage: Path,
    project_root: Path,
    source_identity: _SourceIdentity,
    protocol: Mapping[str, object],
    authority_manifest: M8AcquisitionManifest,
    stage_manifest: M8AcquisitionManifest,
    completed_entries: Sequence[M8ArchiveEntry],
    failure: _TypedInsufficientFailure,
    generated_at_utc: str,
    partial_selections: Sequence[_SelectionArtifacts] = (),
) -> None:
    """Preserve a deterministic declared-data failure before model locking."""

    _verify_config_source(config)
    if sha256_file(_protocol_source(project_root)) != protocol["protocol_sha256"]:
        raise M8PipelineError("protocol changed while preserving pre-lock failure evidence")
    if _capture_source_identity(project_root) != source_identity:
        raise M8PipelineError("source changed while preserving pre-lock failure evidence")
    declared = [
        (symbol, period.date.isoformat(), period.role)
        for period in config.periods
        for symbol in config.study.symbols
    ]
    failed_key = next(
        (item for item in declared if item[0] == failure.symbol and item[1] == failure.study_date),
        None,
    )
    failed_index = declared.index(failed_key) if failed_key is not None else -1
    completed = [
        {
            "symbol": entry.symbol,
            "date": entry.date.isoformat(),
            "role": entry.role,
            "rows": entry.rows,
            "raw_zip_sha256": entry.raw_zip_sha256,
            "normalized_dataset_manifest_sha256": entry.normalized_dataset_manifest_sha256,
            "quality_errors": entry.quality_errors,
            "quality_warnings": entry.quality_warnings,
        }
        for entry in completed_entries
    ]
    selection_started = failure.failure_stage == "model_selection"
    completed_selection_symbols = [item.symbol for item in partial_selections]
    if (
        completed_selection_symbols and not selection_started
    ) or completed_selection_symbols != list(
        config.study.symbols[: len(completed_selection_symbols)]
    ):
        raise M8PipelineError("pre-lock partial selections are outside the frozen order")
    final_fitted_state_claims = _final_fitted_state_claims(partial_selections, stage)
    stopped_before = [
        {"symbol": symbol, "date": study_date, "role": role}
        for symbol, study_date, role in declared[failed_index + 1 :]
    ]
    failed_normalization_evidence = _failed_normalization_evidence_payload(stage, failure)
    failure_payload = {
        "schema_version": "m8-insufficient-data-v1",
        "status": "INSUFFICIENT_DATA",
        "terminal": True,
        "generated_at_utc": generated_at_utc,
        "reason": failure.reason,
        "reason_code": failure.reason_code,
        "failure_stage": failure.failure_stage,
        "failed_symbol": failure.symbol,
        "failed_date": failure.study_date,
        "failed_role": failure.failed_role,
        "failed_normalization_evidence": failed_normalization_evidence,
        "failed_after_analysis_lock": False,
        "held_out_member_opened": False,
        "analysis_lock": None,
        "analysis_lock_path": None,
        "analysis_lock_sha256": None,
        "aggregate_lock_committed": False,
        "selection_started": selection_started,
        "selection_completed_symbols": completed_selection_symbols,
        "selection_completed_symbol_count": len(completed_selection_symbols),
        "selection_locks": [
            {
                "symbol": selection.symbol,
                "path": _relative(selection.lock_path, stage),
                "sha256": selection.selection.lock.sha256,
                "final_fitted_state_path": _relative(
                    selection.fitted_state_path,
                    stage,
                ),
                "final_fitted_state_sha256": selection.selection.fitted_state.sha256,
            }
            for selection in partial_selections
        ],
        "final_fitted_states": final_fitted_state_claims,
        "replacement_date_selected": False,
        "reselection_performed": False,
        "endpoint_evaluation_performed": False,
        "endpoint_evaluation_started": False,
        "endpoint_evaluation_completed": False,
        "endpoint_artifacts_published": False,
        "endpoint_evaluation_completed_symbols": [],
        "endpoint_evaluation_completed_symbol_count": 0,
        "predictions_published": False,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "protocol_version": config.study.protocol_version,
        "protocol_sha256": protocol["protocol_sha256"],
        "raw_acquisition_manifest_sha256": authority_manifest.sha256,
        "bundled_raw_acquisition_manifest_sha256": stage_manifest.sha256,
        "raw_evidence_content_identity_sha256": (authority_manifest.content_identity_sha256),
        "source_identity": source_identity.public_dict(),
        "completed_normalizations": completed,
        "stopped_before": stopped_before,
        "final_all_date_normalized_manifest": None,
    }
    _write_json(stage / "failure.json", failure_payload)
    resolved = config.public_dict()
    resolved.update(
        {
            "effective_evidence_tier": "INSUFFICIENT_DATA",
            "evidence_scope": M8_EVIDENCE_SCOPE,
            "protocol": dict(protocol),
            "raw_acquisition_manifest_sha256": authority_manifest.sha256,
            "execution_status": "NOT_RUN",
        }
    )
    _write_json(stage / "resolved_config.json", resolved)
    _write_json(
        stage / "provenance.json",
        {
            "generated_at_utc": generated_at_utc,
            "status": "INSUFFICIENT_DATA",
            "failure_reason_code": failure.reason_code,
            "failed_normalization_evidence_completion": (failure.normalization_evidence_completion),
            "config_sha256": config.hash,
            "config_source_sha256": config.source_sha256,
            "raw_acquisition_manifest_path": str(authority_manifest.path.resolve()),
            "raw_acquisition_manifest_sha256": authority_manifest.sha256,
            "bundled_raw_acquisition_manifest_path": _relative(stage_manifest.path, stage),
            "bundled_raw_acquisition_manifest_sha256": stage_manifest.sha256,
            "raw_evidence_content_identity_sha256": (authority_manifest.content_identity_sha256),
            "raw_evidence_byte_budget": {
                "external_total": authority_manifest.total_raw_evidence_bytes,
                "bundle_copy_total": stage_manifest.total_raw_evidence_bytes,
                "combined_total": (
                    authority_manifest.total_raw_evidence_bytes
                    + stage_manifest.total_raw_evidence_bytes
                ),
                "ceiling": config.study.max_total_download_bytes,
            },
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_version": config.study.protocol_version,
            "selection_lock_path": None,
            "selection_lock_sha256": None,
            "final_fitted_states": final_fitted_state_claims,
            "git": source_identity.public_dict(),
            "runtime": runtime_metadata(),
            "pipeline_schema_version": M8_PIPELINE_SCHEMA_VERSION,
        },
    )
    _write_json(
        stage / "run_manifest.json",
        {
            "schema_version": M8_PIPELINE_SCHEMA_VERSION,
            "run_id": config.study.name,
            "status": "INSUFFICIENT_DATA",
            "evidence_tier": "INSUFFICIENT_DATA",
            "evidence_scope": M8_EVIDENCE_SCOPE,
            "data": {
                "all_requested_ranges_complete": False,
                "completed_normalizations": completed,
                "failure_reason_code": failure.reason_code,
                "failed_normalization_evidence_completion": (
                    failure.normalization_evidence_completion
                ),
                "failed_symbol": failure.symbol,
                "failed_date": failure.study_date,
                "stopped_before": stopped_before,
            },
            "artifacts": {
                "failure": "failure.json",
                "raw_acquisition_manifest": _relative(stage_manifest.path, stage),
                "resolved_config": "resolved_config.json",
            },
            "research": {
                "scope": "trade_only",
                "analysis_lock": None,
                "aggregate_lock_committed": False,
                "selection_started": selection_started,
                "selection_completed_symbols": completed_selection_symbols,
                "selection_completed_symbol_count": len(completed_selection_symbols),
                "final_fitted_states": final_fitted_state_claims,
                "endpoint_status": "insufficient_data",
                "endpoint_evaluation_performed": False,
                "endpoint_evaluation_started": False,
                "endpoint_evaluation_completed": False,
                "endpoint_artifacts_published": False,
                "endpoint_evaluation_completed_symbols": [],
                "endpoint_evaluation_completed_symbol_count": 0,
            },
            "execution_assumptions": {
                "status": "NOT_RUN",
                "reason": M8_EXECUTION_EXCLUSION_REASON,
                "pnl_calculated": False,
                "fills_calculated": False,
                "capacity_calculated": False,
                "profitability_claim_authorized": False,
            },
        },
    )
    _atomic_write_text(
        stage / "reports" / "insufficient_data.md",
        (
            "# M8 study result: INSUFFICIENT_DATA\n\n"
            f"The predeclared {failure.symbol}/{failure.study_date} input failed before the "
            f"analysis lock: {failure.reason}\n\n"
            + (
                "Candidate selection began and completed for "
                f"{', '.join(completed_selection_symbols) or 'no symbol'}, but no aggregate "
                "lock was committed. "
                if selection_started
                else "No candidate selection or aggregate lock was created. "
            )
            + (
                "No held-out member, replacement date, prediction, endpoint publication, "
                "execution, P&L, or significance result was produced.\n"
            )
        ),
    )
    inventory = [
        {
            "path": path.relative_to(stage).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    ]
    _write_json(stage / "data" / "failure_evidence_inventory.json", inventory)
    _fsync_tree(stage)
    write_checksum_manifest(stage)
    _create_terminal_marker(stage / "INSUFFICIENT_DATA", "terminal")


def _produce_m8(
    config: M8StudyConfig,
    authority_manifest: M8AcquisitionManifest,
    stage: Path,
    project_root: Path,
    source_identity: _SourceIdentity,
) -> tuple[M8RunStatus, str | None]:
    if stage.exists() and any(stage.iterdir()):
        raise M8PipelineError("M8 staging directory must be empty")
    stage.mkdir(parents=True, exist_ok=True)
    protocol = _freeze_protocol_and_config(config, project_root, stage)
    if authority_manifest.protocol_document_sha256 != protocol["protocol_sha256"]:
        raise M8PipelineError("raw acquisition manifest is bound to another protocol document")
    evidence_root = stage / "data"
    raw_input_root = evidence_root / "input"
    normalized_input_root = evidence_root / "normalized_input"
    planned_copy_bytes = authority_manifest.total_raw_evidence_bytes
    combined_raw_evidence_bytes = authority_manifest.total_raw_evidence_bytes + planned_copy_bytes
    if combined_raw_evidence_bytes > config.study.max_total_download_bytes:
        raise M8PipelineError(
            "external raw evidence plus the distinct bundle copy exceeds the frozen total-byte "
            "ceiling"
        )
    try:
        raw_manifest = copy_m8_acquisition_into(authority_manifest, raw_input_root)
    except Exception as exc:
        raise M8PipelineError(f"cannot freeze raw acquisition evidence into run: {exc}") from exc
    if (
        raw_manifest.copied_from_manifest_sha256 != authority_manifest.sha256
        or raw_manifest.content_identity_sha256 != authority_manifest.content_identity_sha256
        or raw_manifest.total_raw_evidence_bytes != authority_manifest.total_raw_evidence_bytes
    ):
        raise M8PipelineError("bundled raw evidence is not an exact semantic copy of authority")
    generated_at = utc_now_iso()
    metadata = tuple(
        _normalized_metadata(raw_manifest.metadata_for(symbol)) for symbol in config.study.symbols
    )
    metadata_by_symbol = {item.symbol: item for item in metadata}
    entries: list[M8ArchiveEntry] = []

    date_artifacts: dict[tuple[str, str], _DateArtifacts] = {}
    # Phase 1: stream-normalize and research only the two development dates for
    # both instruments.  No held-out CSV member is opened in this loop.
    for period in config.periods:
        if period.role not in _DEVELOPMENT_ROLES:
            continue
        for symbol in config.study.symbols:
            descriptor = raw_manifest.archive_descriptor_for(symbol, period.date)
            try:
                acquired = descriptor.reconstruct()
            except M8AcquisitionError as exc:
                if _caused_by_system_fault(exc):
                    raise
                failure = _typed_failure(
                    M8InsufficientDataError(symbol, period.date.isoformat(), str(exc)),
                    reason_code="RAW_ARCHIVE_INTEGRITY",
                    failure_stage="development_acquisition",
                    failed_role=period.role,
                )
                _publish_prelock_insufficient_data(
                    config=config,
                    stage=stage,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol=protocol,
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    completed_entries=entries,
                    failure=failure,
                    generated_at_utc=generated_at,
                )
                return "INSUFFICIENT_DATA", None
            try:
                normalized = normalize_m8_archive(
                    config,
                    period,
                    metadata_by_symbol[symbol],
                    acquired,
                    raw_input_root,
                    output_root=normalized_input_root,
                )
            except M8InsufficientDataError as exc:
                _publish_prelock_insufficient_data(
                    config=config,
                    stage=stage,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol=protocol,
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    completed_entries=entries,
                    failure=_typed_failure(
                        exc,
                        failure_stage="development_normalization",
                        failed_role=period.role,
                    ),
                    generated_at_utc=generated_at,
                )
                return "INSUFFICIENT_DATA", None
            entry = normalized.entry
            entries.append(entry)
            try:
                date_artifacts[(symbol, period.date.isoformat())] = _build_date_artifacts(
                    entry, config, stage
                )
            except (ResearchDataError, TemporalLeakageError) as exc:
                failure = _typed_failure(
                    M8InsufficientDataError(symbol, period.date.isoformat(), str(exc)),
                    reason_code="RESEARCH_FRAME_INSUFFICIENT",
                    failure_stage="development_research",
                    failed_role=period.role,
                )
                _publish_prelock_insufficient_data(
                    config=config,
                    stage=stage,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol=protocol,
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    completed_entries=entries,
                    failure=failure,
                    generated_at_utc=generated_at,
                )
                return "INSUFFICIENT_DATA", None

    development_manifest_path, development_manifest_sha = _commit_development_manifest(
        entries,
        date_artifacts,
        config,
        stage,
    )
    selections: list[_SelectionArtifacts] = []
    for symbol_index, symbol in enumerate(config.study.symbols):
        development = tuple(
            date_artifacts[(symbol, period.date.isoformat())]
            for period in config.periods
            if period.role in _DEVELOPMENT_ROLES
        )
        try:
            selections.append(_select_symbol(symbol, symbol_index, development, config, stage))
        except (MultiDateEvaluationError, ModelEvaluationError) as exc:
            validation_date = next(
                period.date.isoformat() for period in config.periods if period.role == "validation"
            )
            failure = _typed_failure(
                M8InsufficientDataError(symbol, validation_date, str(exc)),
                reason_code="MODEL_SELECTION_INSUFFICIENT",
                failure_stage="model_selection",
                failed_role="validation",
            )
            _publish_prelock_insufficient_data(
                config=config,
                stage=stage,
                project_root=project_root,
                source_identity=source_identity,
                protocol=protocol,
                authority_manifest=authority_manifest,
                stage_manifest=raw_manifest,
                completed_entries=entries,
                failure=failure,
                generated_at_utc=generated_at,
                partial_selections=selections,
            )
            return "INSUFFICIENT_DATA", None
    final_fitted_state_claims = _final_fitted_state_claims(selections, stage)
    selection_by_symbol = {item.symbol: item for item in selections}
    aggregate_lock_path, aggregate_lock_sha = _commit_aggregate_lock(
        selections,
        config,
        authority_manifest,
        development_manifest_path,
        development_manifest_sha,
        protocol["protocol_sha256"],
        source_identity,
        stage,
    )
    _assert_lock_durable(aggregate_lock_path, aggregate_lock_sha, selections)

    # Phase 2: stream-normalize every held-out date.  The callback executes
    # inside the ZIP reader immediately before each CSV member open.
    for period in config.periods:
        if period.role not in _TEST_ROLES:
            continue
        for symbol in config.study.symbols:
            descriptor = raw_manifest.archive_descriptor_for(symbol, period.date)

            def before_member_open() -> None:
                _assert_test_open_authority(
                    config=config,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol_sha256=cast(str, protocol["protocol_sha256"]),
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    aggregate_lock_path=aggregate_lock_path,
                    aggregate_lock_sha256=aggregate_lock_sha,
                    selections=selections,
                )

            try:
                acquired = descriptor.reconstruct()
            except M8AcquisitionError as exc:
                if _caused_by_system_fault(exc):
                    raise
                failure = _typed_failure(
                    M8InsufficientDataError(symbol, period.date.isoformat(), str(exc)),
                    reason_code="RAW_ARCHIVE_INTEGRITY",
                    failure_stage="held_out_acquisition",
                    failed_role=period.role,
                )
                _publish_insufficient_data(
                    config=config,
                    stage=stage,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol=protocol,
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    selections=selections,
                    aggregate_lock_path=aggregate_lock_path,
                    aggregate_lock_sha256=aggregate_lock_sha,
                    completed_entries=entries,
                    failure=failure,
                    generated_at_utc=generated_at,
                )
                return "INSUFFICIENT_DATA", None
            try:
                normalized = normalize_m8_archive(
                    config,
                    period,
                    metadata_by_symbol[symbol],
                    acquired,
                    raw_input_root,
                    output_root=normalized_input_root,
                    before_member_open=before_member_open,
                )
            except M8InsufficientDataError as exc:
                _publish_insufficient_data(
                    config=config,
                    stage=stage,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol=protocol,
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    selections=selections,
                    aggregate_lock_path=aggregate_lock_path,
                    aggregate_lock_sha256=aggregate_lock_sha,
                    completed_entries=entries,
                    failure=_typed_failure(
                        exc,
                        failure_stage="held_out_normalization",
                        failed_role=period.role,
                    ),
                    generated_at_utc=generated_at,
                )
                return "INSUFFICIENT_DATA", None
            entries.append(normalized.entry)

    # No endpoint artifact is constructed until all eight normalizations pass
    # and the strict legacy all-date manifest has been persisted and reloaded.
    try:
        manifest = write_m8_input_manifest(config, evidence_root, entries, metadata)
        manifest = verify_m8_input_manifest(
            config,
            evidence_root,
            manifest.path,
            manifest_sha256=manifest.sha256,
        )
    except M8ManifestError as exc:
        failure = _typed_failure(
            M8InsufficientDataError("STUDY", "all_dates", str(exc)),
            reason_code="FINAL_NORMALIZED_MANIFEST_INCOMPLETE",
            failure_stage="final_manifest",
            failed_role="study",
        )
        _publish_insufficient_data(
            config=config,
            stage=stage,
            project_root=project_root,
            source_identity=source_identity,
            protocol=protocol,
            authority_manifest=authority_manifest,
            stage_manifest=raw_manifest,
            selections=selections,
            aggregate_lock_path=aggregate_lock_path,
            aggregate_lock_sha256=aggregate_lock_sha,
            completed_entries=entries,
            failure=failure,
            generated_at_utc=generated_at,
        )
        return "INSUFFICIENT_DATA", None
    input_snapshot, input_hashes = _snapshot_input_evidence(manifest, stage)
    lookup = _entry_lookup(manifest)
    for period in config.periods:
        if period.role not in _TEST_ROLES:
            continue
        for symbol in config.study.symbols:
            entry = lookup[(symbol, period.date.isoformat())]
            try:
                date_artifacts[(symbol, period.date.isoformat())] = _build_date_artifacts(
                    entry, config, stage
                )
            except (ResearchDataError, TemporalLeakageError) as exc:
                failure = _typed_failure(
                    M8InsufficientDataError(symbol, period.date.isoformat(), str(exc)),
                    reason_code="RESEARCH_FRAME_INSUFFICIENT",
                    failure_stage="held_out_research",
                    failed_role=period.role,
                )
                _publish_insufficient_data(
                    config=config,
                    stage=stage,
                    project_root=project_root,
                    source_identity=source_identity,
                    protocol=protocol,
                    authority_manifest=authority_manifest,
                    stage_manifest=raw_manifest,
                    selections=selections,
                    aggregate_lock_path=aggregate_lock_path,
                    aggregate_lock_sha256=aggregate_lock_sha,
                    completed_entries=entries,
                    failure=failure,
                    generated_at_utc=generated_at,
                    final_manifest=manifest,
                )
                return "INSUFFICIENT_DATA", manifest.sha256

    _assert_lock_durable(aggregate_lock_path, aggregate_lock_sha, selections)
    evaluation_stage = stage / ".endpoint-staging"
    evaluation_stage.mkdir(parents=True, exist_ok=False)
    staged_evaluations: list[_EvaluationArtifacts] = []
    for selection in selections:
        symbol = selection.symbol
        development = tuple(
            date_artifacts[(symbol, period.date.isoformat())]
            for period in config.periods
            if period.role in _DEVELOPMENT_ROLES
        )
        tests = tuple(
            date_artifacts[(symbol, period.date.isoformat())]
            for period in config.periods
            if period.role in _TEST_ROLES
        )
        try:
            staged_evaluations.append(
                _evaluate_symbol(
                    symbol,
                    selection,
                    development,
                    tests,
                    config,
                    evaluation_stage,
                )
            )
        except (
            MultiDateEvaluationError,
            ModelEvaluationError,
            ResearchDataError,
            TemporalLeakageError,
            DescriptiveAnalysisError,
        ) as exc:
            shutil.rmtree(evaluation_stage)
            failure = _typed_failure(
                M8InsufficientDataError(symbol, "locked_evaluation", str(exc)),
                reason_code="LOCKED_EVALUATION_INSUFFICIENT",
                failure_stage="locked_evaluation",
                failed_role="all_test_dates",
            )
            _publish_insufficient_data(
                config=config,
                stage=stage,
                project_root=project_root,
                source_identity=source_identity,
                protocol=protocol,
                authority_manifest=authority_manifest,
                stage_manifest=raw_manifest,
                selections=selections,
                aggregate_lock_path=aggregate_lock_path,
                aggregate_lock_sha256=aggregate_lock_sha,
                completed_entries=entries,
                failure=failure,
                generated_at_utc=generated_at,
                final_manifest=manifest,
                endpoint_evaluation_started=True,
                completed_evaluation_symbols=tuple(item.symbol for item in staged_evaluations),
            )
            return "INSUFFICIENT_DATA", manifest.sha256
    evaluations = _publish_evaluation_artifacts(staged_evaluations, evaluation_stage, stage)

    ordered_dates = tuple(
        date_artifacts[(symbol, period.date.isoformat())]
        for period in config.periods
        for symbol in config.study.symbols
    )
    _combine_parquet(
        [item.predictions_path for item in evaluations],
        stage / "models" / "predictions.parquet",
    )
    _combine_parquet(
        [item.comparison_path for item in selections],
        stage / "models" / "validation_candidate_comparison.parquet",
    )
    _combine_parquet(
        [item.paired_date_path for item in evaluations],
        stage / "metrics" / "paired_log_loss_by_date.parquet",
    )
    _combine_parquet(
        [item.stability_path for item in evaluations],
        stage / "analysis" / "feature_stability.parquet",
    )
    aggregate_frame = pl.DataFrame(
        [dict(item.aggregate_row) for item in evaluations], infer_schema_length=None
    )
    aggregate_frame.write_parquet(
        stage / "metrics" / "equal_date_hypothesis.parquet",
        compression="zstd",
        statistics=True,
    )
    predictive_rows = [dict(row) for item in evaluations for row in item.predictive_rows]
    _write_json(stage / "metrics" / "predictive_metrics.json", predictive_rows)
    _write_json(stage / "metrics" / "execution_metrics.json", [])
    _write_json(stage / "metrics" / "execution_sensitivity.json", [])
    _write_json(
        stage / "metrics" / "execution_exclusion.json",
        {
            "status": "NOT_RUN",
            "reason": M8_EXECUTION_EXCLUSION_REASON,
            "execution_metrics_rows": 0,
            "execution_sensitivity_rows": 0,
            "fills_calculated": False,
            "pnl_calculated": False,
            "capacity_calculated": False,
            "execution_claim_authorized": False,
            "profitability_claim_authorized": False,
        },
    )
    hypothesis_payload = {
        "schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "evidence_tier": M8_EVIDENCE_TIER,
        "evidence_scope": M8_EVIDENCE_SCOPE,
        "hypotheses": {
            "H0": (
                "The validation-selected model does not reduce untouched-date log loss "
                "relative to the historical-prior classifier."
            ),
            "H1": (
                "The validation-selected model reduces log loss on both primary and "
                "replication dates."
            ),
        },
        "selection_metric": "log_loss",
        "delta_definition": "selected_model_minus_historical_prior",
        "date_weighting": "equal",
        "bootstrap": {
            "method": "paired contiguous-block percentile bootstrap from block sufficient statistics",
            "samples": config.study.bootstrap_samples,
            "block_width_events": config.study.bootstrap_block_events,
            "ci_level": 0.95,
        },
        "per_symbol": [dict(item.aggregate_row) for item in evaluations],
        "per_date": [dict(row) for item in evaluations for row in item.paired_date_rows],
        "per_date_artifact": "metrics/paired_log_loss_by_date.parquet",
        "cross_instrument_conclusion": {
            "status": "not_inferred",
            "pooling_performed": False,
            "persistent_alpha_claim_authorized": False,
            "text": M8_NO_POOLING_CAVEAT,
        },
        "p_values_computed": False,
        "h0_rejected": False,
        "significance_claim_authorized": False,
        "execution_claim_authorized": False,
        "profitability_claim_authorized": False,
        "caveat": M8_NO_SIGNIFICANCE_CAVEAT,
    }
    _write_json(stage / "metrics" / "hypothesis_evaluation.json", hypothesis_payload)
    _write_json(
        stage / "analysis" / "instrument_status.json",
        [
            {
                "symbol": item.symbol,
                "selected_model": item.selected_model,
                "status": item.aggregate_row["status"],
                "replication_status": item.aggregate_row["replication_status"],
                "validation_primary_replication_status": item.aggregate_row[
                    "validation_primary_replication_status"
                ],
                "direction_consistent_across_validation_primary_replication": (
                    item.aggregate_row["direction_consistent_across_validation_primary_replication"]
                ),
                "selection_lock_sha256": item.aggregate_row["selection_lock_sha256"],
                "final_fitted_state_path": _relative(
                    selection_by_symbol[item.symbol].fitted_state_path,
                    stage,
                ),
                "final_fitted_state_sha256": selection_by_symbol[
                    item.symbol
                ].selection.fitted_state.sha256,
            }
            for item in evaluations
        ],
    )

    research_manifest = {
        "schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "artifact_kind": "m8_per_date_research_frames",
        "temporal_contract": (
            "exchange event time is the availability proxy; aggregate-trade ID breaks ties; "
            "features and labels reset at every symbol/date continuity ID"
        ),
        "target": config.study.target,
        "label_horizon_events": config.study.label_horizon_events,
        "feature_columns": list(_trade_feature_columns(config)),
        "selection_lock_path": _relative(aggregate_lock_path, stage),
        "selection_lock_sha256": aggregate_lock_sha,
        "final_fitted_states": final_fitted_state_claims,
        "test_data_opened_before_lock": False,
        "test_economic_rows_materialized_before_lock": False,
        "test_raw_hashes_and_bounded_zip_metadata_verified_before_lock": True,
        "test_normalization_completed_after_lock": True,
        "final_all_date_manifest_committed_before_endpoint_evaluation": True,
        "entries": [
            {
                "symbol": item.symbol,
                "date": item.study_date,
                "role": item.role,
                "research_frame": _relative(item.research_path, stage),
                "research_frame_sha256": sha256_file(item.research_path),
                "evaluation_frame": _relative(item.evaluation_path, stage),
                "evaluation_frame_sha256": sha256_file(item.evaluation_path),
                "summary": _relative(item.research_path.parent / "summary.json", stage),
            }
            for item in ordered_dates
        ],
        "symbol_evaluations": {
            item.symbol: {
                "selected_model": item.selected_model,
                "predictions": _relative(item.predictions_path, stage),
                "paired_date_metrics": _relative(item.paired_date_path, stage),
                "feature_stability": _relative(item.stability_path, stage),
                "walk_forward_plan": _relative(item.plan_path, stage),
                "final_fitted_state_path": _relative(
                    selection_by_symbol[item.symbol].fitted_state_path,
                    stage,
                ),
                "final_fitted_state_sha256": selection_by_symbol[
                    item.symbol
                ].selection.fitted_state.sha256,
                "test_used_for_selection": False,
                "model_updated_between_test_dates": False,
            }
            for item in evaluations
        },
    }
    _write_json(stage / "research" / "manifest.json", research_manifest)
    quality = _quality_payload(manifest, ordered_dates, generated_at_utc=generated_at)
    _write_json(stage / "quality" / "summary.json", quality)
    dashboard_path = stage / "dashboard" / "market_state.parquet"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {key: value for key, value in item.summary.items() if key != "temporal_audit"}
            for item in ordered_dates
        ],
        infer_schema_length=None,
    ).write_parquet(stage / "quality" / "per_date_summary.parquet")
    pl.DataFrame(
        [
            {
                "symbol": item.symbol,
                "date": item.study_date,
                "role": item.role,
                "eligible_labeled_rows": item.summary["eligible_labeled_rows"],
                "positive_rate": item.summary["eligible_positive_rate"],
                "scope": "trade_only_no_book_state",
                "evidence_tier": M8_EVIDENCE_TIER,
            }
            for item in ordered_dates
        ],
        infer_schema_length=None,
    ).write_parquet(dashboard_path)

    resolved = config.public_dict()
    resolved.update(
        {
            "effective_evidence_tier": M8_EVIDENCE_TIER,
            "evidence_scope": M8_EVIDENCE_SCOPE,
            "full_data_boundary": (
                "complete requested trade archives only; not complete market observability"
            ),
            "protocol": protocol,
            "input_manifest_sha256": manifest.sha256,
            "raw_acquisition_manifest_sha256": authority_manifest.sha256,
            "execution_status": "NOT_RUN",
            "claim_permissions": {
                "p_values": False,
                "significance": False,
                "cross_instrument_pooling": False,
                "execution": False,
                "profitability": False,
            },
        }
    )
    _write_json(stage / "resolved_config.json", resolved)

    run_key, run_key_inputs = _run_key(
        config,
        authority_manifest,
        manifest,
        protocol,
        source_identity,
        _final_fitted_state_sha256_by_symbol(selections),
    )
    provenance = {
        "generated_at_utc": generated_at,
        "evidence_tier": M8_EVIDENCE_TIER,
        "evidence_scope": M8_EVIDENCE_SCOPE,
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "input_manifest_sha256": [manifest.sha256],
        "input_data_and_evidence_sha256": sorted(
            {*input_hashes, authority_manifest.sha256, raw_manifest.sha256}
        ),
        "m8_input_manifest_path": _relative(manifest.path, stage),
        "m8_input_manifest_sha256": manifest.sha256,
        "raw_acquisition_manifest_path": str(authority_manifest.path.resolve()),
        "raw_acquisition_manifest_sha256": authority_manifest.sha256,
        "bundled_raw_acquisition_manifest_path": _relative(raw_manifest.path, stage),
        "bundled_raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": authority_manifest.content_identity_sha256,
        "raw_evidence_byte_budget": {
            "external_total": authority_manifest.total_raw_evidence_bytes,
            "bundle_copy_total": raw_manifest.total_raw_evidence_bytes,
            "combined_total": (
                authority_manifest.total_raw_evidence_bytes + raw_manifest.total_raw_evidence_bytes
            ),
            "ceiling": config.study.max_total_download_bytes,
        },
        "manifest_authority": (
            "explicit raw-acquisition path and caller-supplied lowercase SHA-256; no discovery"
        ),
        "protocol_path": protocol["protocol_path"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_version": config.study.protocol_version,
        "selection_lock_path": _relative(aggregate_lock_path, stage),
        "selection_lock_sha256": aggregate_lock_sha,
        "final_fitted_states": final_fitted_state_claims,
        "git": source_identity.public_dict(),
        "runtime": runtime_metadata(),
        "seed": config.study.seed,
        "run_key": run_key,
        "run_key_inputs": run_key_inputs,
        "pipeline_schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "data_availability_clock": "exchange_event_time_proxy_plus_aggregate_trade_id",
        "local_receipt_time_available": False,
        "execution_simulated": False,
        "pnl_calculated": False,
    }
    _write_json(stage / "provenance.json", provenance)

    rows = sum(entry.rows for entry in manifest.entries)
    observed_start = min(entry.observed_start_ns for entry in manifest.entries)
    observed_end = max(entry.observed_end_inclusive_ns for entry in manifest.entries)
    symbol_coverage = []
    for symbol in config.study.symbols:
        entries = [entry for entry in manifest.entries if entry.symbol == symbol]
        symbol_coverage.append(
            {
                "symbol": symbol,
                "rows": sum(entry.rows for entry in entries),
                "complete": True,
                "complete_range": True,
                "observed_start_utc": _utc_from_ns(
                    min(entry.observed_start_ns for entry in entries)
                ),
                "observed_end_inclusive_utc": _utc_from_ns(
                    max(entry.observed_end_inclusive_ns for entry in entries)
                ),
                "requested_dates": [entry.date.isoformat() for entry in entries],
            }
        )
    artifacts = {
        "resolved_config": "resolved_config.json",
        "protocol": cast(str, protocol["protocol_path"]),
        "machine_spec": cast(str, protocol["machine_spec_path"]),
        "raw_acquisition_manifest": _relative(raw_manifest.path, stage),
        "input_manifest_snapshot": "data/m8_input_manifest.json",
        "data_manifest_snapshot": "data/manifest_snapshot.json",
        "quality_summary": "quality/summary.json",
        "quality_by_date": "quality/per_date_summary.parquet",
        "research_manifest": "research/manifest.json",
        "analysis_lock": "analysis/analysis_lock.json",
        "predictions": "models/predictions.parquet",
        "model_comparison_data": "models/validation_candidate_comparison.parquet",
        "predictive_metrics": "metrics/predictive_metrics.json",
        "hypothesis_evaluation": "metrics/hypothesis_evaluation.json",
        "paired_date_metrics": "metrics/paired_log_loss_by_date.parquet",
        "equal_date_metrics": "metrics/equal_date_hypothesis.parquet",
        "feature_stability": "analysis/feature_stability.parquet",
        "instrument_status": "analysis/instrument_status.json",
        "execution_metrics": "metrics/execution_metrics.json",
        "execution_sensitivity": "metrics/execution_sensitivity.json",
        "execution_exclusion": "metrics/execution_exclusion.json",
        "market_state": "dashboard/market_state.parquet",
        "technical_report": "reports/technical_report.md",
        "executive_memo": "reports/executive_memo.md",
        "model_comparison": "reports/model_comparison.md",
    }
    run_manifest = {
        "schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "run_id": config.study.name,
        "run_key": run_key,
        "status": "complete",
        "evidence_tier": M8_EVIDENCE_TIER,
        "evidence_scope": M8_EVIDENCE_SCOPE,
        "data": {
            "mode": "binance_spot_daily_aggtrades_trade_only",
            "source": config.study.source,
            "symbols": list(config.study.symbols),
            "rows": rows,
            "all_requested_ranges_complete": True,
            "requested_dates": [period.date.isoformat() for period in config.periods],
            "observed_start_utc": _utc_from_ns(observed_start),
            "observed_end_utc": _utc_from_ns(observed_end),
            "observed_start_ts_ns": observed_start,
            "observed_end_ts_ns": observed_end,
            "availability_basis": "exchange_event_time_proxy_plus_aggregate_trade_id",
            "local_receipt_time_available": False,
            "symbol_coverage": symbol_coverage,
            "date_coverage": [
                {
                    "symbol": entry.symbol,
                    "date": entry.date.isoformat(),
                    "role": entry.role,
                    "rows": entry.rows,
                    "complete": entry.complete,
                    "quality_errors": entry.quality_errors,
                    "quality_warnings": entry.quality_warnings,
                    "observed_start_utc": _utc_from_ns(entry.observed_start_ns),
                    "observed_end_inclusive_utc": _utc_from_ns(entry.observed_end_inclusive_ns),
                }
                for entry in manifest.entries
            ],
            "full_data_boundary": (
                "all bytes in all predeclared daily trade archives; no book or receipt-time data"
            ),
        },
        "artifacts": artifacts,
        "research": {
            "question": (
                "whether frozen aggregate-trade order-flow features improve next-20-trade "
                "direction log loss over a historical prior on both untouched dates"
            ),
            "scope": "trade_only",
            "target": config.study.target,
            "label_horizon_trades": config.study.label_horizon_events,
            "evaluation_contract": (
                "per-symbol train/validation selection, disk-persisted lock, primary and "
                "replication evaluation with one fixed train+validation fit"
            ),
            "selection_contract": "validation log loss only; test rows never select or update",
            "final_fitted_states": final_fitted_state_claims,
            "analysis_lock": {
                "path": _relative(aggregate_lock_path, stage),
                "sha256": aggregate_lock_sha,
                "committed_before_test_rows_opened": True,
            },
            "hypothesis_evaluation": {
                "artifact": "metrics/hypothesis_evaluation.json",
                "per_date_artifact": "metrics/paired_log_loss_by_date.parquet",
                "equal_date_artifact": "metrics/equal_date_hypothesis.parquet",
                "baseline": "historical_prior",
                "metric": "log_loss",
                "delta_definition": "selected_model_minus_historical_prior",
                "date_weighting": "equal",
                "per_symbol_only": True,
                "cross_instrument_pooling": False,
                "p_values_computed": False,
                "h0_rejected": False,
                "significance_claim_authorized": False,
                "persistent_alpha_claim_authorized": False,
            },
            "instruments": {
                item.symbol: {
                    "selected_model": item.selected_model,
                    "status": item.aggregate_row["status"],
                    "replication_status": item.aggregate_row["replication_status"],
                    "validation_primary_replication_status": item.aggregate_row[
                        "validation_primary_replication_status"
                    ],
                    "direction_consistent_across_validation_primary_replication": (
                        item.aggregate_row[
                            "direction_consistent_across_validation_primary_replication"
                        ]
                    ),
                    "selection_lock_sha256": item.aggregate_row["selection_lock_sha256"],
                    "final_fitted_state_path": _relative(
                        selection_by_symbol[item.symbol].fitted_state_path,
                        stage,
                    ),
                    "final_fitted_state_sha256": selection_by_symbol[
                        item.symbol
                    ].selection.fitted_state.sha256,
                    "model_updated_between_test_dates": False,
                }
                for item in evaluations
            },
        },
        "execution_assumptions": {
            "status": "NOT_RUN",
            "reason": M8_EXECUTION_EXCLUSION_REASON,
            "pnl_calculated": False,
            "fills_calculated": False,
            "capacity_calculated": False,
            "profitability_claim_authorized": False,
        },
        "warnings": [
            (
                "FULL_DATA is narrowly scoped to complete predeclared trade archives and does "
                "not mean full market observability or deployable evidence."
            ),
            "Exchange event time is an availability proxy; no local receipt clock is available.",
            M8_EXECUTION_EXCLUSION_REASON,
            M8_NO_SIGNIFICANCE_CAVEAT,
            M8_NO_POOLING_CAVEAT,
        ],
    }
    _write_json(stage / "run_manifest.json", run_manifest)
    _write_reports(stage)

    # Revalidate every external hash after all economic computation and prove
    # that neither the frozen config nor the exact source tree changed mid-run.
    _verify_config_source(config)
    if sha256_file(_protocol_source(project_root)) != protocol["protocol_sha256"]:
        raise M8PipelineError("frozen M8 protocol changed during production")
    second_manifest = _load_final_input_manifest(config, manifest.path, manifest.sha256)
    if second_manifest.sha256 != manifest.sha256:
        raise M8PipelineError("M8 input identity changed during production")
    second_raw = _load_raw_manifest(
        config,
        authority_manifest.path,
        authority_manifest.sha256,
    )
    if second_raw.sha256 != authority_manifest.sha256:
        raise M8PipelineError("M8 raw acquisition identity changed during production")
    read_m8_acquisition_manifest(
        raw_manifest.path,
        expected_sha256=raw_manifest.sha256,
        config=config,
    )
    if _capture_source_identity(project_root) != source_identity:
        raise M8PipelineError("Git/source-tree identity changed during M8 production")
    _assert_lock_durable(aggregate_lock_path, aggregate_lock_sha, selections)
    if (
        input_snapshot["manifest_authority"]
        != json.loads((stage / "data" / "manifest_snapshot.json").read_text(encoding="utf-8"))[
            "manifest_authority"
        ]
    ):
        raise M8PipelineError("frozen input snapshot changed during production")

    _fsync_tree(stage)
    write_checksum_manifest(stage)
    _create_terminal_marker(stage / "_SUCCESS", "complete")
    load_run_bundle(stage)
    return "COMPLETE", manifest.sha256


def _reuse_completed(
    target: Path,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    project_root: Path,
    source_identity: _SourceIdentity,
) -> M8RunResult:
    if (target / "INSUFFICIENT_DATA").exists():
        raise M8PipelineError("completed M8 target has conflicting terminal markers")
    try:
        success_bytes = (target / "_SUCCESS").read_bytes()
    except OSError as exc:
        raise M8PipelineError("completed M8 terminal marker is unavailable") from exc
    if success_bytes != b"complete\n":
        raise M8PipelineError("completed M8 terminal marker bytes are invalid")
    try:
        bundle = load_run_bundle(target)
    except Exception as exc:
        raise M8PipelineError(f"completed M8 target failed integrity verification: {exc}") from exc
    if bundle.run_id != config.study.name:
        raise M8PipelineError("completed M8 target has a different run ID")
    expected_pairs = {
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
        "protocol_version": config.study.protocol_version,
        "pipeline_schema_version": M8_PIPELINE_SCHEMA_VERSION,
        "evidence_scope": M8_EVIDENCE_SCOPE,
    }
    for key, expected in expected_pairs.items():
        if bundle.provenance.get(key) != expected:
            raise M8PipelineError(f"completed M8 target has a different {key}")
    bundled_raw_path_value = bundle.provenance.get("bundled_raw_acquisition_manifest_path")
    bundled_raw_sha = bundle.provenance.get("bundled_raw_acquisition_manifest_sha256")
    if type(bundled_raw_path_value) is not str or type(bundled_raw_sha) is not str:
        raise M8PipelineError("completed M8 target lacks bundled raw-manifest identity")
    bundled_raw_path = (target / bundled_raw_path_value).resolve()
    if not bundled_raw_path.is_relative_to(target):
        raise M8PipelineError("completed bundled raw-manifest path escapes target")
    try:
        bundled_raw = read_m8_acquisition_manifest(
            bundled_raw_path,
            expected_sha256=bundled_raw_sha,
            config=config,
        )
    except Exception as exc:
        raise M8PipelineError(f"completed bundled raw evidence is invalid: {exc}") from exc
    if (
        bundled_raw.copied_from_manifest_sha256 != raw_manifest.sha256
        or bundled_raw.content_identity_sha256 != raw_manifest.content_identity_sha256
    ):
        raise M8PipelineError("completed bundled raw evidence differs from acquisition authority")
    normalized_sha = bundle.provenance.get("m8_input_manifest_sha256")
    normalized_path_value = bundle.provenance.get("m8_input_manifest_path")
    if type(normalized_sha) is not str or type(normalized_path_value) is not str:
        raise M8PipelineError("completed M8 target lacks normalized-manifest identity")
    normalized_path = (target / normalized_path_value).resolve()
    if not normalized_path.is_relative_to(target):
        raise M8PipelineError("completed M8 normalized-manifest path escapes the bundle")
    normalized_manifest = _load_final_input_manifest(
        config,
        normalized_path,
        normalized_sha,
    )
    bundled_git = bundle.provenance.get("git")
    if not isinstance(bundled_git, Mapping) or dict(bundled_git) != source_identity.public_dict():
        raise M8PipelineError("completed M8 target has a different Git/source-tree identity")
    protocol_sha = sha256_file(_protocol_source(project_root))
    if bundle.provenance.get("protocol_sha256") != protocol_sha:
        raise M8PipelineError("completed M8 target has a different frozen protocol")
    final_fitted_state_sha256_by_symbol = _verify_completed_lock_chain(
        target=target,
        config=config,
        raw_manifest=raw_manifest,
        source_identity=source_identity,
        protocol_sha256=protocol_sha,
        run_manifest=bundle.manifest,
        provenance=bundle.provenance,
    )
    expected_run_key, expected_run_key_inputs = _run_key(
        config,
        raw_manifest,
        normalized_manifest,
        {"protocol_sha256": protocol_sha},
        source_identity,
        final_fitted_state_sha256_by_symbol,
    )
    if (
        bundle.manifest.get("run_key") != expected_run_key
        or bundle.provenance.get("run_key") != expected_run_key
        or bundle.provenance.get("run_key_inputs") != expected_run_key_inputs
    ):
        raise M8PipelineError("completed M8 target has a different deterministic run identity")
    execution = bundle.manifest.get("execution_assumptions")
    if not isinstance(execution, Mapping) or execution.get("status") != "NOT_RUN":
        raise M8PipelineError("completed M8 target improperly claims execution evidence")
    if bundle.manifest.get("evidence_scope") != M8_EVIDENCE_SCOPE:
        raise M8PipelineError("completed FULL_DATA target is not explicitly trade-only")
    return M8RunResult(
        path=target,
        status="COMPLETE",
        raw_manifest_sha256=raw_manifest.sha256,
        normalized_manifest_sha256=normalized_manifest.sha256,
    )


def _reuse_insufficient(
    target: Path,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    project_root: Path,
    source_identity: _SourceIdentity,
) -> M8RunResult:
    if (target / "_SUCCESS").exists():
        raise M8PipelineError("INSUFFICIENT_DATA target has conflicting terminal markers")
    try:
        marker_bytes = _read_bounded_regular_snapshot(
            _published_file(target, "INSUFFICIENT_DATA", "INSUFFICIENT_DATA terminal marker"),
            label="INSUFFICIENT_DATA terminal marker",
            max_bytes=32,
        ).content
    except M8PipelineError as exc:
        raise M8PipelineError("INSUFFICIENT_DATA terminal marker is unavailable") from exc
    if marker_bytes != b"terminal\n":
        raise M8PipelineError("INSUFFICIENT_DATA terminal marker bytes are invalid")
    try:
        verify_checksums(target)
    except Exception as exc:
        raise M8PipelineError(
            f"INSUFFICIENT_DATA target failed integrity verification: {exc}"
        ) from exc
    failure_inventory = _verify_failure_evidence_inventory(target)
    failure = _read_inventory_json_object(
        target=target,
        inventory=failure_inventory,
        relative_path="failure.json",
        label="INSUFFICIENT_DATA failure record",
    )
    provenance = _read_inventory_json_object(
        target=target,
        inventory=failure_inventory,
        relative_path="provenance.json",
        label="INSUFFICIENT_DATA provenance record",
    )
    run_manifest = _read_inventory_json_object(
        target=target,
        inventory=failure_inventory,
        relative_path="run_manifest.json",
        label="INSUFFICIENT_DATA run manifest",
    )
    if failure.get("status") != "INSUFFICIENT_DATA" or failure.get("terminal") is not True:
        raise M8PipelineError("INSUFFICIENT_DATA target lacks a terminal failure record")
    expected = {
        "config_sha256": config.hash,
        "config_source_sha256": config.source_sha256,
        "raw_acquisition_manifest_sha256": raw_manifest.sha256,
        "raw_evidence_content_identity_sha256": raw_manifest.content_identity_sha256,
        "protocol_version": config.study.protocol_version,
        "pipeline_schema_version": M8_PIPELINE_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise M8PipelineError(f"INSUFFICIENT_DATA target has a different {key}")
    if provenance.get("git") != source_identity.public_dict():
        raise M8PipelineError("INSUFFICIENT_DATA target has a different source identity")
    if provenance.get("protocol_sha256") != sha256_file(_protocol_source(project_root)):
        raise M8PipelineError("INSUFFICIENT_DATA target has a different protocol")
    bundled_raw_path_value = provenance.get("bundled_raw_acquisition_manifest_path")
    bundled_raw_sha = provenance.get("bundled_raw_acquisition_manifest_sha256")
    if type(bundled_raw_path_value) is not str or type(bundled_raw_sha) is not str:
        raise M8PipelineError("INSUFFICIENT_DATA target lacks bundled raw-manifest identity")
    bundled_raw_relative = _canonical_inventory_child(
        bundled_raw_path_value,
        "INSUFFICIENT_DATA bundled raw acquisition manifest",
    )
    bundled_raw_digest = _require_digest(
        bundled_raw_sha,
        "INSUFFICIENT_DATA bundled raw acquisition manifest SHA-256",
    )
    _read_inventory_bounded_snapshot(
        target=target,
        inventory=failure_inventory,
        relative_path=bundled_raw_relative,
        label="INSUFFICIENT_DATA bundled raw acquisition manifest",
        max_bytes=_MAX_LOCK_JSON_BYTES,
        expected_sha256=bundled_raw_digest,
    )
    bundled_raw_path = _published_file(
        target,
        bundled_raw_relative,
        "INSUFFICIENT_DATA bundled raw acquisition manifest",
    )
    try:
        bundled_raw = read_m8_acquisition_manifest(
            bundled_raw_path,
            expected_sha256=bundled_raw_digest,
            config=config,
        )
    except Exception as exc:
        raise M8PipelineError(f"INSUFFICIENT_DATA bundled raw evidence is invalid: {exc}") from exc
    if (
        bundled_raw.copied_from_manifest_sha256 != raw_manifest.sha256
        or bundled_raw.content_identity_sha256 != raw_manifest.content_identity_sha256
    ):
        raise M8PipelineError(
            "INSUFFICIENT_DATA bundled raw evidence differs from acquisition authority"
        )
    _bind_raw_acquisition_inventory(
        target=target,
        inventory=failure_inventory,
        manifest=bundled_raw,
        manifest_relative=bundled_raw_relative,
    )
    _verify_insufficient_lock_chain(
        target=target,
        inventory=failure_inventory,
        config=config,
        raw_manifest=raw_manifest,
        bundled_raw_sha256=bundled_raw.sha256,
        source_identity=source_identity,
        protocol_sha256=sha256_file(_protocol_source(project_root)),
        failure=failure,
        provenance=provenance,
        run_manifest=run_manifest,
    )
    _verify_failed_normalization_evidence(
        target=target,
        config=config,
        raw_manifest=raw_manifest,
        failure=failure,
        inventory=failure_inventory,
    )
    _verify_completed_normalization_evidence(
        target=target,
        config=config,
        raw_manifest=raw_manifest,
        failure=failure,
        inventory=failure_inventory,
    )
    forbidden = (
        target / "models" / "predictions.parquet",
        target / "metrics" / "predictive_metrics.json",
        target / "metrics" / "hypothesis_evaluation.json",
        target / "metrics" / "paired_log_loss_by_date.parquet",
        target / "metrics" / "equal_date_hypothesis.parquet",
        target / "_SUCCESS",
    )
    if any(path.exists() for path in forbidden):
        raise M8PipelineError("INSUFFICIENT_DATA target improperly contains endpoint output")
    if any(path.is_file() for path in target.rglob("*prediction*")):
        raise M8PipelineError("INSUFFICIENT_DATA target improperly contains predictions")
    normalized_manifest_sha256: str | None = None
    final_manifest_claim = failure.get("final_all_date_normalized_manifest")
    if final_manifest_claim is not None:
        if not isinstance(final_manifest_claim, Mapping):
            raise M8PipelineError("INSUFFICIENT_DATA final-manifest claim is malformed")
        final_path_value = final_manifest_claim.get("path")
        final_sha = final_manifest_claim.get("sha256")
        if type(final_path_value) is not str or type(final_sha) is not str:
            raise M8PipelineError("INSUFFICIENT_DATA final-manifest identity is malformed")
        final_relative = _canonical_inventory_child(
            final_path_value,
            "INSUFFICIENT_DATA final normalized manifest",
        )
        final_digest = _require_digest(
            final_sha,
            "INSUFFICIENT_DATA final normalized manifest SHA-256",
        )
        _read_inventory_bounded_snapshot(
            target=target,
            inventory=failure_inventory,
            relative_path=final_relative,
            label="INSUFFICIENT_DATA final normalized manifest",
            max_bytes=_MAX_LOCK_JSON_BYTES,
            expected_sha256=final_digest,
        )
        final_manifest = _load_final_input_manifest(
            config,
            _published_file(
                target,
                final_relative,
                "INSUFFICIENT_DATA final normalized manifest",
            ),
            final_digest,
        )
        _bind_final_input_manifest_inventory(
            target=target,
            inventory=failure_inventory,
            manifest=final_manifest,
        )
        normalized_manifest_sha256 = final_manifest.sha256
    if _verify_failure_evidence_inventory(target) != failure_inventory:
        raise M8PipelineError("INSUFFICIENT_DATA evidence identity changed during verification")
    return M8RunResult(
        path=target,
        status="INSUFFICIENT_DATA",
        raw_manifest_sha256=raw_manifest.sha256,
        normalized_manifest_sha256=normalized_manifest_sha256,
    )


def _self_verify_produced_terminal(
    *,
    target: Path,
    expected_status: M8RunStatus,
    expected_normalized_manifest_sha256: str | None,
    config: M8StudyConfig,
    raw_manifest: M8AcquisitionManifest,
    project_root: Path,
    source_identity: _SourceIdentity,
) -> M8RunResult:
    """Apply the external reuse contract to a just-produced terminal tree."""

    if expected_status == "COMPLETE":
        observed = _reuse_completed(
            target,
            config,
            raw_manifest,
            project_root,
            source_identity,
        )
    else:
        observed = _reuse_insufficient(
            target,
            config,
            raw_manifest,
            project_root,
            source_identity,
        )
    if (
        observed.status != expected_status
        or observed.raw_manifest_sha256 != raw_manifest.sha256
        or observed.normalized_manifest_sha256 != expected_normalized_manifest_sha256
    ):
        raise M8PipelineError("M8 producer terminal result disagrees with its reuse verifier")
    return observed


def verify_m8_result(
    path: str | Path,
    config: M8StudyConfig,
    *,
    raw_manifest_path: str | Path,
    raw_manifest_sha256: str,
) -> M8RunResult:
    """Verify one immutable complete or insufficient-data M8 result."""

    project_root = _project_root(config)
    _verify_config_source(config)
    raw_manifest = _load_raw_manifest(
        config,
        Path(raw_manifest_path),
        raw_manifest_sha256,
    )
    source_identity = _capture_source_identity(project_root)
    if (
        source_identity.dirty
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_identity.commit) is None
    ):
        raise M8PipelineError("M8 result verification requires the exact clean committed source")
    target = Path(path).resolve()
    if not target.is_dir():
        raise M8PipelineError(f"M8 result directory does not exist: {target}")
    success_marker = (target / "_SUCCESS").is_file()
    insufficient_marker = (target / "INSUFFICIENT_DATA").is_file()
    if success_marker and insufficient_marker:
        raise M8PipelineError("M8 result has conflicting terminal markers")
    if success_marker:
        return _reuse_completed(target, config, raw_manifest, project_root, source_identity)
    if insufficient_marker:
        return _reuse_insufficient(target, config, raw_manifest, project_root, source_identity)
    raise M8PipelineError("M8 result has no recognized terminal marker")


def reproduce_m8(
    config: M8StudyConfig,
    run_dir: Path,
    *,
    raw_manifest_path: Path,
    raw_manifest_sha256: str,
) -> M8RunResult:
    """Produce or verify one immutable M8 trade-only run bundle.

    The input authority is always the caller's exact manifest path and lowercase
    SHA-256.  No directory scan, newest-file rule, implicit fallback, repair, or
    overwrite is permitted.  A new run is built in a sibling directory and
    atomically renamed only after checksums and ``_SUCCESS`` are complete.
    """

    project_root = _project_root(config)
    _verify_config_source(config)
    raw_manifest = _load_raw_manifest(
        config,
        Path(raw_manifest_path),
        raw_manifest_sha256,
    )
    source_identity = _capture_source_identity(project_root)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_identity.commit):
        raise M8PipelineError(
            "M8 production requires a real committed Git revision before opening any economic rows"
        )
    if source_identity.dirty:
        raise M8PipelineError(
            "M8 production and reuse require a clean Git working tree; commit the exact "
            "source state before opening any economic rows"
        )
    target = Path(run_dir).resolve()
    if target.exists():
        if (target / "_SUCCESS").is_file() and (target / "INSUFFICIENT_DATA").is_file():
            raise M8PipelineError("M8 run target has conflicting terminal markers")
        if target.is_dir() and (target / "_SUCCESS").is_file():
            return _reuse_completed(
                target,
                config,
                raw_manifest,
                project_root,
                source_identity,
            )
        if target.is_dir() and (target / "INSUFFICIENT_DATA").is_file():
            return _reuse_insufficient(
                target,
                config,
                raw_manifest,
                project_root,
                source_identity,
            )
        raise M8PipelineError(
            f"M8 run target already exists but is incomplete or not a directory: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)).resolve()
    try:
        status, normalized_manifest_sha256 = _produce_m8(
            config,
            raw_manifest,
            stage,
            project_root,
            source_identity,
        )
        _self_verify_produced_terminal(
            target=stage,
            expected_status=status,
            expected_normalized_manifest_sha256=normalized_manifest_sha256,
            config=config,
            raw_manifest=raw_manifest,
            project_root=project_root,
            source_identity=source_identity,
        )
        if target.exists():
            raise M8PipelineError(f"M8 run target appeared during production: {target}")
        stage.rename(target)
        _fsync_directory(target.parent)
        return _self_verify_produced_terminal(
            target=target,
            expected_status=status,
            expected_normalized_manifest_sha256=normalized_manifest_sha256,
            config=config,
            raw_manifest=raw_manifest,
            project_root=project_root,
            source_identity=source_identity,
        )
    except M8PipelineError:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    except Exception as exc:
        if stage.exists():
            shutil.rmtree(stage)
        raise M8PipelineError(f"M8 production failed closed: {exc}") from exc


__all__ = [
    "M8_EVIDENCE_SCOPE",
    "M8_EVIDENCE_TIER",
    "M8_EXECUTION_EXCLUSION_REASON",
    "M8_PIPELINE_SCHEMA_VERSION",
    "M8PipelineError",
    "M8RunResult",
    "M8RunStatus",
    "reproduce_m8",
    "verify_m8_result",
]
