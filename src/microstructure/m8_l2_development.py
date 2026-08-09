"""Outcome-blind development authority for the frozen four-session M8 L2 study.

This module is the only stage allowed to fit L2 regime thresholds, select a
model, or fit final model state.  Its public producer accepts exactly the train
and validation session paths.  The held-out sessions are represented only by
their predeclared calendar coordinates and cannot be supplied to this API.

Publication is fail closed.  Complete train/validation sessions produce the
eight-child ``LOCKED`` authority.  A verified insufficient development session
instead produces a checksummed ``NOT_CREATED`` authority without opening any
economic rows.  Both variants reserve without overwrite and write their exact
terminal marker last.  Held-out evaluation may restore only ``LOCKED`` state;
it must never call a fit or update API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import polars as pl
import pyarrow.parquet as pq  # type: ignore[import-untyped]

import microstructure.m8_l2_inputs as _l2_inputs_module
from microstructure.config import ModelConfig
from microstructure.m8_l2_analysis_config import (
    M8L2AnalysisConfig,
    M8L2AnalysisEndpoint,
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
from microstructure.provenance import (
    ImportOriginError,
    assert_project_module_origins,
    git_source_tree_sha256,
    sha256_file,
    strict_git_state,
)
from microstructure.research.l2_multidate import (
    L2EndpointSpec,
    L2ObservedInterval,
    L2RegimeFit,
    apply_l2_regimes,
    build_l2_endpoint_frames,
    fit_l2_regime_thresholds,
    l2_model_feature_columns,
)
from microstructure.research.multidate import (
    AnalysisLock,
    FinalFittedState,
    LockedSelection,
    select_multidate_model,
)

_LOCK_SCHEMA_VERSION = "m8-l2-development-lock-v1"
_NOT_CREATED_SCHEMA_VERSION = "m8-l2-development-not-created-v1"
_CHILD_SCHEMA_VERSION = "m8-l2-development-child-lock-v1"
_REGIME_SCHEMA_VERSION = "m8-l2-regime-thresholds-v1"
_EXECUTION_REFERENCE_SCHEMA_VERSION = "m8-l2-execution-reference-v1"
_INVENTORY_SCHEMA_VERSION = "m8-l2-development-inventory-v1"
_LOCKED_MARKER = "_LOCKED"
_LOCKED_BYTES = b"locked\n"
_NOT_CREATED_MARKER = "_NOT_CREATED"
_NOT_CREATED_BYTES = b"not-created\n"
_CHECKSUMS_NAME = "CHECKSUMS.sha256"
_INVENTORY_NAME = "inventory.json"
_AGGREGATE_NAME = "development_lock.json"
_AGGREGATE_DIGEST_NAME = "development_lock.sha256"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_COMPARISON_BYTES = 64 * 1024 * 1024
_GIB = 1024**3
# The 16 GiB host envelope is split into disjoint producer workspaces.  These
# are admission limits, not telemetry thresholds: metadata/row-count bounds are
# checked before the costly builders and the materialized result is checked
# immediately after every allocation boundary.
_MAX_DEVELOPMENT_RAW_BYTES = 2 * _GIB
_MAX_DEVELOPMENT_CAUSAL_BYTES = 4 * _GIB
_MAX_SELECTION_WORKSPACE_BYTES = 2 * _GIB
_CAUSAL_ENDPOINT_ROW_UPPER_BYTES = 4 * 1024
_SELECTION_NUMPY_VECTOR_COUNT = 12
_EXPECTED_DEVELOPMENT = (("2026-08-10", "train"), ("2026-08-11", "validation"))
_EXPECTED_HELDOUT = (
    ("2026-08-12", "primary_test"),
    ("2026-08-13", "replication_test"),
)
_EARLIEST_LOCK_TIME = datetime(2026, 8, 11, 15, 0, tzinfo=UTC)
_LOCK_DEADLINE = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)


class M8L2DevelopmentError(RuntimeError):
    """Raised when development fitting or lock publication must fail closed."""


@dataclass(frozen=True, slots=True)
class ProducerSourceIdentity:
    commit: str
    source_tree_sha256: str
    dirty: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class L2DevelopmentChildLock:
    symbol: str
    endpoint: str
    path: Path
    sha256: str
    selection_lock_sha256: str
    fitted_state_sha256: str


@dataclass(frozen=True, slots=True)
class L2DevelopmentLockResult:
    root: Path
    aggregate_path: Path
    aggregate_sha256: str
    marker_path: Path
    created_at_utc: str
    children: tuple[L2DevelopmentChildLock, ...]
    status: Literal["LOCKED", "NOT_CREATED"] = "LOCKED"
    reason_codes: tuple[str, ...] = ()


# Neutral public name for the LOCKED | NOT_CREATED union.  The historical
# alias remains available because the pre-capture CLI already exposes lock-
# named arguments and both variants intentionally share that authority slot.
L2DevelopmentAuthorityResult = L2DevelopmentLockResult


class _LoadedSymbolFrames(Protocol):
    book_observations: pl.DataFrame
    depth_deltas: pl.DataFrame
    intervals: tuple[L2ObservedInterval, ...]


class _FileAuthority(Protocol):
    manifest_sha256: str
    checksums_sha256: str

    def to_dict(self) -> dict[str, object]: ...


class _ExpectedFileAuthority(Protocol):
    @property
    def manifest_sha256(self) -> str: ...

    @property
    def checksums_sha256(self) -> str: ...


class _CampaignIdentity(Protocol):
    campaign_authority_sha256: str
    runtime_commit: str
    runtime_source_tree_sha256: str
    runtime_fingerprint_sha256: str
    runtime_dirty: bool

    def to_dict(self) -> dict[str, object]: ...


class _VerifiedInput(Protocol):
    root: Path
    session_id: str
    session_date: str
    role: str
    config_sha256: str
    config_source_sha256: str
    file_authority: _FileAuthority
    campaign_identity: _CampaignIdentity
    symbols: Mapping[str, object]
    access_phase: str
    development_lock_sha256: str | None

    def load_symbol_frames(self, symbol: str) -> _LoadedSymbolFrames: ...


class L2DevelopmentInputVerifier(Protocol):
    def __call__(
        self,
        bundle_dir: str | Path,
        *,
        expected_config: M8L2StudyConfig,
        expected_date: str,
        expected_role: str,
        expected_file_authority: object | None = None,
        expected_campaign: object | None = None,
    ) -> _VerifiedInput: ...


def _frame_bytes(frame: pl.DataFrame) -> int:
    """Return Polars' owned-buffer estimate as an integer byte count."""

    return int(frame.estimated_size("b"))


def _loaded_bytes(value: _LoadedSymbolFrames) -> int:
    # Intervals are tiny fixed dataclasses, but charging one KiB each keeps the
    # admission proof conservative even if a malformed authority has many.
    return (
        _frame_bytes(value.book_observations)
        + _frame_bytes(value.depth_deltas)
        + len(value.intervals) * 1024
    )


def _require_memory_budget(observed: int, maximum: int, label: str) -> None:
    if observed < 0 or observed > maximum:
        raise M8L2DevelopmentError(
            f"{label} exceeds the fail-closed memory budget ({observed} > {maximum} bytes)"
        )


def _parquet_metadata_bytes(root: Path, artifact: object, label: str) -> tuple[int, int]:
    """Read only Parquet metadata, through a no-follow descriptor, before load."""

    relative = getattr(artifact, "relative_path", None)
    claimed_rows = getattr(artifact, "rows", None)
    if not isinstance(relative, str) or not isinstance(claimed_rows, int):
        raise M8L2DevelopmentError(f"{label} lacks bounded Parquet metadata authority")
    safe = _safe_relative(relative)
    path = root / safe
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise M8L2DevelopmentError(f"{label} is not a regular Parquet file")
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
            raise M8L2DevelopmentError(f"{label} changed during memory admission")
    except M8L2DevelopmentError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise M8L2DevelopmentError(f"cannot inspect {label} memory metadata") from error
    if rows != claimed_rows or rows < 1 or uncompressed < 1:
        raise M8L2DevelopmentError(f"{label} Parquet metadata differs from its authority")
    return uncompressed, rows


def _preflight_symbol_raw(value: _VerifiedInput, symbol: str) -> tuple[int, int] | None:
    """Return (raw bytes, book rows) without opening column payloads.

    Production verifier objects always expose the two verified artifact
    descriptors.  The ``None`` branch exists only for injected test loaders;
    those remain protected by the immediate post-load budget check.
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


def _causal_build_upper_bytes(book_rows: int, endpoint_count: int) -> int:
    if book_rows < 0 or endpoint_count < 1:
        raise M8L2DevelopmentError("invalid row count for causal-memory admission")
    return book_rows * endpoint_count * _CAUSAL_ENDPOINT_ROW_UPPER_BYTES


def _selection_workspace_upper_bytes(
    train: pl.DataFrame,
    validation: pl.DataFrame,
    *,
    feature_count: int,
) -> int:
    """Bound selection copies and NumPy arrays before entering the fitter.

    ``select_multidate_model`` materializes a full-width date concat, eligible
    filters/splits, and two sets of float64 feature matrices.  Four full-width
    equivalents plus feature/target/probability vectors is deliberately above
    that live set and therefore rejects before NumPy can exceed its 2 GiB slot.
    """

    if feature_count < 1:
        raise M8L2DevelopmentError("selection memory admission requires model features")
    rows = train.height + validation.height
    full_width_copies = 4 * (_frame_bytes(train) + _frame_bytes(validation))
    numpy_bytes = rows * (feature_count * 8 * 2 + _SELECTION_NUMPY_VECTOR_COUNT * 8)
    return full_width_copies + numpy_bytes


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _producer_source_identity(project_root: Path) -> ProducerSourceIdentity:
    before = strict_git_state(project_root)
    source_tree_sha256 = git_source_tree_sha256(project_root)
    after = strict_git_state(project_root)
    if before != after:
        raise M8L2DevelopmentError("Git identity changed during development source snapshot")
    return ProducerSourceIdentity(
        commit=before.commit,
        source_tree_sha256=source_tree_sha256,
        dirty=before.dirty,
    )


def _load_input_verifier() -> L2DevelopmentInputVerifier:
    return cast(
        L2DevelopmentInputVerifier,
        _l2_inputs_module.verify_m8_l2_development_input,
    )


def _assert_development_import_origins(project_root: Path) -> None:
    try:
        assert_project_module_origins(
            project_root,
            "microstructure.m8_l2_development",
            _l2_inputs_module,
            "microstructure.m8_l2_analysis_config",
            "microstructure.research.l2_multidate",
            "microstructure.research.multidate",
        )
    except ImportOriginError as error:
        raise M8L2DevelopmentError(
            "development producer has a foreign or mixed import origin"
        ) from error


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
        raise M8L2DevelopmentError("development authority is not canonical finite JSON") from error


def _decode_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise M8L2DevelopmentError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise M8L2DevelopmentError(f"{label} contains forbidden constant {value}")

    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except M8L2DevelopmentError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M8L2DevelopmentError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or not all(type(key) is str for key in decoded):
        raise M8L2DevelopmentError(f"{label} must be a JSON object")
    return cast(dict[str, Any], decoded)


def _safe_relative(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or value != candidate.as_posix()
    ):
        raise M8L2DevelopmentError(f"unsafe development-lock relative path {value!r}")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise M8L2DevelopmentError("development artifact escapes its lock root") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short write while publishing development authority")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    raw = _canonical_json_bytes(payload)
    _write_bytes(path, raw)
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise M8L2DevelopmentError(f"cannot stat {label}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise M8L2DevelopmentError(f"{label} must be a regular file")
    if metadata.st_size > maximum:
        raise M8L2DevelopmentError(f"{label} exceeds its bounded verification size")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise M8L2DevelopmentError(f"{label} changed during verification")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > maximum:
            raise M8L2DevelopmentError(f"{label} exceeds its bounded verification size")
        if os.fstat(descriptor).st_size != len(raw):
            raise M8L2DevelopmentError(f"{label} changed during verification")
        return raw
    except OSError as error:
        raise M8L2DevelopmentError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label=label, maximum=_MAX_JSON_BYTES)
    return _decode_json_bytes(raw, label), raw


def _reject_symlink_components(path: Path) -> None:
    requested = path.absolute()
    current = Path(requested.anchor)
    for part in requested.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            continue
        if current.is_symlink():
            raise M8L2DevelopmentError(f"development-lock path contains symlink {current}")


def _walk_regular(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise M8L2DevelopmentError("cannot enumerate development-lock inventory") from error
        for entry in entries:
            path = Path(entry.path)
            relative = _relative(path, root)
            if entry.is_symlink():
                raise M8L2DevelopmentError(
                    f"development-lock inventory contains symlink {relative}"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                result[relative] = path
            else:
                raise M8L2DevelopmentError(
                    f"development-lock inventory contains non-regular entry {relative}"
                )
    return dict(sorted(result.items()))


def _validate_source(identity: ProducerSourceIdentity) -> None:
    if identity.dirty:
        raise M8L2DevelopmentError("L2 development locking requires a clean Git source tree")
    if len(identity.commit) != 40 or any(
        char not in "0123456789abcdef" for char in identity.commit
    ):
        raise M8L2DevelopmentError("producer Git commit is not a lowercase 40-character SHA-1")
    if len(identity.source_tree_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in identity.source_tree_sha256
    ):
        raise M8L2DevelopmentError("producer source-tree identity is not a lowercase SHA-256")


def _revalidate_configs(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
) -> tuple[M8L2StudyConfig, M8L2AnalysisConfig]:
    try:
        capture = load_m8_l2_config(capture_config.path)
        analysis = load_m8_l2_analysis_config(analysis_config.path)
    except (OSError, ValueError) as error:
        raise M8L2DevelopmentError(
            "frozen L2 configuration authority cannot be reloaded"
        ) from error
    if capture != capture_config or analysis != analysis_config:
        raise M8L2DevelopmentError("in-memory L2 configuration differs from exact frozen bytes")
    if (
        analysis.study.capture_config_source_sha256 != capture.source_sha256
        or analysis.study.capture_protocol_sha256 != M8_L2_PROTOCOL_SHA256
        or analysis.study.symbols != capture.study.symbols
        or analysis.study.seed != capture.study.seed
    ):
        raise M8L2DevelopmentError("capture and analysis contracts do not share one frozen study")
    coordinates = tuple((item.date.isoformat(), item.role) for item in capture.sessions)
    if coordinates != (*_EXPECTED_DEVELOPMENT, *_EXPECTED_HELDOUT):
        raise M8L2DevelopmentError("capture calendar differs from the frozen four-session study")
    if (
        analysis.study.training_role != "train"
        or analysis.study.selection_role != "validation"
        or analysis.study.primary_endpoint_role != "primary_test"
        or analysis.study.replication_endpoint_role != "replication_test"
    ):
        raise M8L2DevelopmentError("analysis roles differ from the frozen four-session study")
    return capture, analysis


def _campaign_dict(value: _CampaignIdentity) -> dict[str, object]:
    payload = value.to_dict()
    expected = {
        "campaign_authority_sha256",
        "runtime_commit",
        "runtime_source_tree_sha256",
        "runtime_fingerprint_sha256",
        "runtime_dirty",
    }
    if set(payload) != expected:
        raise M8L2DevelopmentError("development input campaign identity is malformed")
    return payload


def _campaign_from_session_bundle(bundle: M8L2SessionBundle) -> _CampaignIdentity:
    manifest, raw = _read_json(bundle.manifest_path, "development session manifest")
    if hashlib.sha256(raw).hexdigest() != bundle.manifest_sha256:
        raise M8L2DevelopmentError("development session manifest authority changed")
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping):
        raise M8L2DevelopmentError("development session campaign authority is malformed")
    try:
        value = _l2_inputs_module.L2CampaignRuntimeIdentity(
            campaign_authority_sha256=str(authority["campaign_authority_sha256"]),
            runtime_commit=str(authority["runtime_commit"]),
            runtime_source_tree_sha256=str(authority["runtime_source_tree_sha256"]),
            runtime_fingerprint_sha256=str(authority["runtime_fingerprint_sha256"]),
            runtime_dirty=authority.get("runtime_dirty") is not False,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise M8L2DevelopmentError("development session campaign authority is malformed") from error
    return cast(_CampaignIdentity, value)


def _session_file_authority(bundle: M8L2SessionBundle) -> dict[str, object]:
    return {
        "manifest_sha256": bundle.manifest_sha256,
        "checksums_sha256": sha256_file(bundle.checksum_path),
    }


def _verify_expected_session_file_authorities(
    bundles: tuple[M8L2SessionBundle, ...],
    expected: Mapping[str, _ExpectedFileAuthority] | None,
) -> None:
    """Bind control-only session files to caller-held digests without opening Parquet."""

    if expected is None:
        return
    expected_dates = {value[0] for value in _EXPECTED_DEVELOPMENT}
    if set(expected) != expected_dates:
        raise M8L2DevelopmentError(
            "expected development session authority set must contain exactly Aug8 and Aug9"
        )
    for bundle in bundles:
        authority = expected[bundle.session_date]
        if (
            bundle.manifest_sha256 != authority.manifest_sha256
            or sha256_file(bundle.checksum_path) != authority.checksums_sha256
        ):
            raise M8L2DevelopmentError(f"caller-held {bundle.role} session file authority changed")


def _development_session_claim(bundle: M8L2SessionBundle) -> dict[str, object]:
    return {
        "date": bundle.session_date,
        "role": bundle.role,
        "status": bundle.status,
        "session_id": bundle.session_id,
        "file_authority": _session_file_authority(bundle),
        "reason_codes": list(bundle.reason_codes),
    }


def _not_created_reasons(
    bundles: tuple[M8L2SessionBundle, M8L2SessionBundle],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"DEVELOPMENT_SESSION_INSUFFICIENT::{bundle.role}::{reason}"
                for bundle in bundles
                if bundle.status != "COMPLETE"
                for reason in (bundle.reason_codes or ("SESSION_INSUFFICIENT_DATA",))
            }
        )
    )


def _file_authority_dict(value: _FileAuthority) -> dict[str, object]:
    payload = value.to_dict()
    if set(payload) != {"manifest_sha256", "checksums_sha256"}:
        raise M8L2DevelopmentError("development input file authority is malformed")
    return payload


def _verify_input_descriptor(
    value: _VerifiedInput,
    bundle: M8L2SessionBundle,
    *,
    capture: M8L2StudyConfig,
    expected_date: str,
    expected_role: str,
) -> None:
    if (
        value.root.absolute() != bundle.root.absolute()
        or value.session_id != bundle.session_id
        or value.session_date != expected_date
        or value.role != expected_role
        or value.config_sha256 != capture.hash
        or value.config_source_sha256 != capture.source_sha256
        or value.access_phase != "development"
        or value.development_lock_sha256 is not None
        or tuple(value.symbols) != capture.study.symbols
        or value.file_authority.manifest_sha256 != bundle.manifest_sha256
    ):
        raise M8L2DevelopmentError("verified development input descriptor changed its authority")


def _endpoint_specs(analysis: M8L2AnalysisConfig) -> tuple[L2EndpointSpec, ...]:
    result: list[L2EndpointSpec] = []
    windows = set(analysis.features.rolling_windows)
    for item in analysis.endpoints:
        impact_window = (
            item.horizon_value if item.domain == "event" else item.nominal_event_block_width
        )
        if impact_window not in windows:
            impact_window = min(windows, key=lambda value: abs(value - impact_window))
        result.append(
            L2EndpointSpec(
                name=item.name,
                domain=item.domain,
                horizon_value=item.horizon_value,
                horizon_unit=item.unit,
                paired_block_events=item.paired_block_width if item.domain == "event" else None,
                paired_block_milliseconds=(
                    item.paired_block_width if item.domain == "clock" else None
                ),
                impact_ofi_window=impact_window,
            )
        )
    return tuple(result)


def _model_config(capture: M8L2StudyConfig) -> ModelConfig:
    return ModelConfig(
        selection_metric=capture.models.selection_metric,
        logistic_c_values=capture.models.logistic_c_values,
        tree_max_depth_values=capture.models.tree_max_depth_values,
        tree_min_samples_leaf=capture.models.tree_min_samples_leaf,
    )


def _finite_unique(frame: pl.DataFrame, column: str, label: str) -> float:
    if column not in frame.columns:
        raise M8L2DevelopmentError(f"execution reference lacks {column}")
    values = [float(value) for value in frame.get_column(column).drop_nulls().unique().to_list()]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0.0:
        raise M8L2DevelopmentError(f"{label} must be one consistent positive finite value")
    return values[0]


def _execution_reference(
    symbol: str,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    analysis: M8L2AnalysisConfig,
) -> dict[str, object]:
    train_ready = train.filter(pl.col("feature_ready")).drop_nulls(
        ["mid_price", "bid_quantity", "ask_quantity", "tick_size", "lot_size"]
    )
    validation_ready = validation.filter(pl.col("feature_ready")).drop_nulls(
        ["tick_size", "lot_size"]
    )
    if train_ready.is_empty() or validation_ready.is_empty():
        raise M8L2DevelopmentError("execution reference requires feature-ready development rows")
    tick = _finite_unique(train_ready, "tick_size", "train tick size")
    lot = _finite_unique(train_ready, "lot_size", "train lot size")
    if (
        _finite_unique(validation_ready, "tick_size", "validation tick size") != tick
        or _finite_unique(validation_ready, "lot_size", "validation lot size") != lot
    ):
        raise M8L2DevelopmentError("tick/lot size changes between development sessions")
    midpoint = train_ready.get_column("mid_price").median()
    executable_depth = train_ready.select(
        pl.min_horizontal("bid_quantity", "ask_quantity").alias("executable_l1_depth")
    )
    depth_q05 = executable_depth.get_column("executable_l1_depth").quantile(
        0.05, interpolation="linear"
    )
    if not isinstance(midpoint, (int, float)) or not isinstance(depth_q05, (int, float)):
        raise M8L2DevelopmentError("execution reference statistics are unavailable")
    reference_mid = float(midpoint)
    reference_depth = float(depth_q05)
    if not math.isfinite(reference_mid) or reference_mid <= 0.0:
        raise M8L2DevelopmentError("train median mid price is not positive and finite")
    if not math.isfinite(reference_depth) or reference_depth <= 0.0:
        raise M8L2DevelopmentError("train q05 executable L1 depth is not positive and finite")
    raw_quantity = min(
        analysis.execution.order_notional_usd / reference_mid,
        analysis.execution.max_l1_participation * reference_depth,
    )
    quantity_lots = math.floor(raw_quantity / lot + 1e-12)
    reference_quantity = quantity_lots * lot
    if quantity_lots < 1 or not math.isfinite(reference_quantity):
        raise M8L2DevelopmentError("frozen execution reference rounds below one lot")
    return {
        "schema_version": _EXECUTION_REFERENCE_SCHEMA_VERSION,
        "artifact_kind": "train_only_execution_reference",
        "symbol": symbol,
        "fit_date": "2026-08-10",
        "fit_role": "train",
        "reference_price_statistic": analysis.execution.reference_price_statistic,
        "reference_depth_statistic": analysis.execution.reference_depth_statistic,
        "reference_mid_price": reference_mid,
        "reference_l1_depth_q05": reference_depth,
        "tick_size": tick,
        "lot_size": lot,
        "tick_lot_consistency_roles": ["train", "validation"],
        "order_notional_usd": analysis.execution.order_notional_usd,
        "max_l1_participation": analysis.execution.max_l1_participation,
        "unrounded_reference_quantity": raw_quantity,
        "reference_quantity_lots": quantity_lots,
        "reference_quantity": reference_quantity,
        "rounding_policy": "floor_to_whole_lot",
        "quantity_policy": analysis.execution.reference_quantity_policy,
        "train_feature_ready_rows": train_ready.height,
    }


def _regime_payload(fitted: L2RegimeFit, analysis: M8L2AnalysisConfig) -> dict[str, object]:
    return {
        "schema_version": _REGIME_SCHEMA_VERSION,
        "artifact_kind": "train_only_l2_regime_thresholds",
        "analysis_config_source_sha256": analysis.source_sha256,
        **fitted.to_dict(),
    }


def _endpoint_payload(endpoint: M8L2AnalysisEndpoint) -> dict[str, object]:
    return {
        "name": endpoint.name,
        "domain": endpoint.domain,
        "horizon_value": endpoint.horizon_value,
        "horizon_unit": endpoint.unit,
        "paired_block_width": endpoint.paired_block_width,
        "paired_block_unit": endpoint.paired_block_unit,
        "nominal_event_block_width": endpoint.nominal_event_block_width,
    }


def _frame_sha256(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode())
    row_hashes = frame.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3)
    for chunk in row_hashes.get_chunks():
        digest.update(chunk.to_numpy().astype("<u8", copy=False).tobytes(order="C"))
    digest.update(str(frame.height).encode())
    return digest.hexdigest()


def _write_parquet(path: Path, frame: pl.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise M8L2DevelopmentError(f"refusing to overwrite development artifact {path}")
    frame.write_parquet(path, compression="zstd", statistics=True)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return sha256_file(path)


def _child_payload(
    *,
    root: Path,
    symbol: str,
    endpoint: M8L2AnalysisEndpoint,
    selection: LockedSelection,
    selection_path: Path,
    state_path: Path,
    comparison_path: Path,
    comparison_file_sha256: str,
    regime_path: Path,
    regime_sha256: str,
    execution_path: Path,
    execution_sha256: str,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    producer: ProducerSourceIdentity,
    train_input: _VerifiedInput,
    validation_input: _VerifiedInput,
) -> dict[str, object]:
    return {
        "schema_version": _CHILD_SCHEMA_VERSION,
        "artifact_kind": "m8_l2_symbol_endpoint_development_lock",
        "symbol": symbol,
        "endpoint": _endpoint_payload(endpoint),
        "capture_config_sha256": capture.hash,
        "capture_config_source_sha256": capture.source_sha256,
        "analysis_config_sha256": analysis.hash,
        "analysis_config_source_sha256": analysis.source_sha256,
        "producer_source_identity": producer.to_dict(),
        "development_inputs": [
            {
                "date": train_input.session_date,
                "role": train_input.role,
                "session_id": train_input.session_id,
                "file_authority": _file_authority_dict(train_input.file_authority),
            },
            {
                "date": validation_input.session_date,
                "role": validation_input.role,
                "session_id": validation_input.session_id,
                "file_authority": _file_authority_dict(validation_input.file_authority),
            },
        ],
        "campaign_identity": _campaign_dict(train_input.campaign_identity),
        "selection_policy": "eight_candidates_fit_train_only_selected_validation_log_loss",
        "final_fit_policy": (
            "selected_specification_and_independent_historical_prior_fit_once_on_"
            "train_plus_validation_before_aggregate_lock"
        ),
        "candidate_count": selection.validation_comparison.height,
        "selected_model": selection.selected_model,
        "selection_lock_path": _relative(selection_path, root),
        "selection_lock_sha256": selection.lock.sha256,
        "final_fitted_state_path": _relative(state_path, root),
        "final_fitted_state_sha256": selection.fitted_state.sha256,
        "validation_comparison_path": _relative(comparison_path, root),
        "validation_comparison_file_sha256": comparison_file_sha256,
        "validation_comparison_frame_sha256": _frame_sha256(selection.validation_comparison),
        "validation_comparison_rows": selection.validation_comparison.height,
        "regime_thresholds_path": _relative(regime_path, root),
        "regime_thresholds_sha256": regime_sha256,
        "execution_reference_path": _relative(execution_path, root),
        "execution_reference_sha256": execution_sha256,
        "development_frame_sha256": selection.development_frame_sha256,
        "test_rows_accessed": False,
        "heldout_fit_or_update_allowed": False,
    }


def _write_child(
    *,
    root: Path,
    symbol: str,
    endpoint: M8L2AnalysisEndpoint,
    selection: LockedSelection,
    regime_path: Path,
    regime_sha256: str,
    execution_path: Path,
    execution_sha256: str,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    producer: ProducerSourceIdentity,
    train_input: _VerifiedInput,
    validation_input: _VerifiedInput,
) -> L2DevelopmentChildLock:
    base = root / "models" / symbol.lower() / endpoint.name
    selection_path = base / "selection_lock.json"
    state_path = base / "final_fitted_state.json"
    comparison_path = base / "validation_comparison.parquet"
    _write_bytes(selection_path, selection.lock.payload_json.encode("ascii") + b"\n")
    _write_bytes(state_path, selection.fitted_state.payload_json.encode("ascii") + b"\n")
    comparison_file_sha = _write_parquet(comparison_path, selection.validation_comparison)
    child_payload = _child_payload(
        root=root,
        symbol=symbol,
        endpoint=endpoint,
        selection=selection,
        selection_path=selection_path,
        state_path=state_path,
        comparison_path=comparison_path,
        comparison_file_sha256=comparison_file_sha,
        regime_path=regime_path,
        regime_sha256=regime_sha256,
        execution_path=execution_path,
        execution_sha256=execution_sha256,
        capture=capture,
        analysis=analysis,
        producer=producer,
        train_input=train_input,
        validation_input=validation_input,
    )
    child_path = base / "child_lock.json"
    child_sha = _write_json(child_path, child_payload)
    return L2DevelopmentChildLock(
        symbol=symbol,
        endpoint=endpoint.name,
        path=child_path,
        sha256=child_sha,
        selection_lock_sha256=selection.lock.sha256,
        fitted_state_sha256=selection.fitted_state.sha256,
    )


def _reserve_destination(path: Path) -> Path:
    _reject_symlink_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)
    try:
        path.mkdir(mode=0o755)
    except FileExistsError as error:
        raise M8L2DevelopmentError(
            f"development-lock destination already exists; overwrite is forbidden: {path}"
        ) from error
    _fsync_directory(path.parent)
    return path


def _publish_inventory(
    root: Path,
    *,
    artifact_kind: str = "m8_l2_development_lock_inventory",
) -> None:
    files = _walk_regular(root)
    if (
        _LOCKED_MARKER in files
        or _NOT_CREATED_MARKER in files
        or _CHECKSUMS_NAME in files
        or _INVENTORY_NAME in files
    ):
        raise M8L2DevelopmentError("terminal inventory files were published out of order")
    inventory_entries = [
        {
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for relative, path in files.items()
    ]
    _write_json(
        root / _INVENTORY_NAME,
        {
            "schema_version": _INVENTORY_SCHEMA_VERSION,
            "artifact_kind": artifact_kind,
            "files": inventory_entries,
        },
    )
    checksum_files = _walk_regular(root)
    lines = [f"{sha256_file(path)}  {relative}\n" for relative, path in checksum_files.items()]
    _write_bytes(root / _CHECKSUMS_NAME, "".join(lines).encode("ascii"))


def _validate_lock_time(now: datetime) -> str:
    if now.tzinfo is None:
        raise M8L2DevelopmentError("development lock clock must be timezone-aware")
    observed = now.astimezone(UTC)
    if observed < _EARLIEST_LOCK_TIME:
        raise M8L2DevelopmentError("development lock cannot precede validation session completion")
    if observed >= _LOCK_DEADLINE:
        raise M8L2DevelopmentError("development lock deadline has passed")
    return observed.isoformat().replace("+00:00", "Z")


def _validate_runtime_campaign(
    producer: ProducerSourceIdentity,
    train_input: _VerifiedInput,
    validation_input: _VerifiedInput,
) -> None:
    train_campaign = _campaign_dict(train_input.campaign_identity)
    validation_campaign = _campaign_dict(validation_input.campaign_identity)
    if train_campaign != validation_campaign:
        raise M8L2DevelopmentError("development sessions have different campaign identities")
    expected = {
        "runtime_commit": producer.commit,
        "runtime_source_tree_sha256": producer.source_tree_sha256,
        "runtime_dirty": False,
    }
    if any(train_campaign.get(name) != value for name, value in expected.items()):
        raise M8L2DevelopmentError(
            "development producer source differs from the frozen capture campaign"
        )
    if (
        train_campaign.get("runtime_fingerprint_sha256")
        != current_m8_l2_runtime_fingerprint_sha256()
    ):
        raise M8L2DevelopmentError(
            "development producer runtime differs from the frozen capture campaign"
        )


def _validate_not_created_context(
    *,
    capture: M8L2StudyConfig,
    producer: ProducerSourceIdentity,
    bundles: tuple[M8L2SessionBundle, M8L2SessionBundle],
) -> _CampaignIdentity:
    campaigns = tuple(_campaign_from_session_bundle(bundle) for bundle in bundles)
    if _campaign_dict(campaigns[0]) != _campaign_dict(campaigns[1]):
        raise M8L2DevelopmentError("development sessions have different campaign identities")
    campaign = _campaign_dict(campaigns[0])
    if (
        campaign.get("runtime_commit") != producer.commit
        or campaign.get("runtime_source_tree_sha256") != producer.source_tree_sha256
        or campaign.get("runtime_dirty") is not False
    ):
        raise M8L2DevelopmentError(
            "development producer source differs from the frozen capture campaign"
        )
    if campaign["runtime_fingerprint_sha256"] != current_m8_l2_runtime_fingerprint_sha256():
        raise M8L2DevelopmentError(
            "development producer runtime differs from the frozen capture campaign"
        )
    for bundle, coordinate in zip(bundles, _EXPECTED_DEVELOPMENT, strict=True):
        if (bundle.session_date, bundle.role) != coordinate:
            raise M8L2DevelopmentError("development session has the wrong frozen coordinate")
        if bundle.status not in {"COMPLETE", "INSUFFICIENT_DATA"}:
            raise M8L2DevelopmentError("development session has an unsupported terminal status")
    if not _not_created_reasons(bundles):
        raise M8L2DevelopmentError(
            "NOT_CREATED development authority requires an insufficient development session"
        )
    if capture.sessions[0].date.isoformat() != bundles[0].session_date:
        raise M8L2DevelopmentError("development capture calendar changed")
    return campaigns[0]


def _publish_not_created_development(
    *,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    bundles: tuple[M8L2SessionBundle, M8L2SessionBundle],
    lock_dir: str | Path,
    expected_session_file_authorities: Mapping[str, _ExpectedFileAuthority] | None,
) -> L2DevelopmentAuthorityResult:
    project_root = capture.path.parent.parent.resolve()
    producer = _producer_source_identity(project_root)
    _validate_source(producer)
    campaign = _validate_not_created_context(
        capture=capture,
        producer=producer,
        bundles=bundles,
    )
    _validate_lock_time(_utc_now())
    root = _reserve_destination(Path(lock_dir).absolute())
    marker = root / _NOT_CREATED_MARKER
    reasons = _not_created_reasons(bundles)
    try:
        created_at = _validate_lock_time(_utc_now())
        payload: dict[str, object] = {
            "schema_version": _NOT_CREATED_SCHEMA_VERSION,
            "artifact_kind": "m8_l2_development_authority_not_created",
            "status": "NOT_CREATED",
            "study": analysis.study.name,
            "created_at_utc": created_at,
            "must_precede_utc": _LOCK_DEADLINE.isoformat().replace("+00:00", "Z"),
            "capture_config_sha256": capture.hash,
            "capture_config_source_sha256": capture.source_sha256,
            "capture_protocol_sha256": analysis.study.capture_protocol_sha256,
            "analysis_config_sha256": analysis.hash,
            "analysis_config_source_sha256": analysis.source_sha256,
            "producer_source_identity": producer.to_dict(),
            "campaign_identity": _campaign_dict(campaign),
            "development_inputs": [_development_session_claim(bundle) for bundle in bundles],
            "reason_codes": list(reasons),
            "children": [],
            "heldout_declarations": [
                {"date": date_value, "role": role} for date_value, role in _EXPECTED_HELDOUT
            ],
            "heldout_access": {
                "paths_received": False,
                "file_hashes_received": False,
                "row_counts_received": False,
                "economic_rows_opened": False,
                "model_fit_or_update_after_lock": False,
            },
            "claims": {
                "cross_symbol_pooling": False,
                "p_values": False,
                "significance": False,
                "capacity": False,
                "realized_execution": False,
                "profitability": False,
            },
        }
        aggregate_path = root / _AGGREGATE_NAME
        aggregate_sha = _write_json(aggregate_path, payload)
        _write_bytes(
            root / _AGGREGATE_DIGEST_NAME,
            f"{aggregate_sha}  {_AGGREGATE_NAME}\n".encode("ascii"),
        )
        _publish_inventory(
            root,
            artifact_kind="m8_l2_development_not_created_inventory",
        )
        _fsync_directory(root)
        _validate_lock_time(_utc_now())
        final_capture, final_analysis = _revalidate_configs(capture, analysis)
        final_producer = _producer_source_identity(project_root)
        repeated = tuple(
            verify_m8_l2_session_bundle(bundle.root, expected_config=capture) for bundle in bundles
        )
        _verify_expected_session_file_authorities(repeated, expected_session_file_authorities)
        if (
            final_capture != capture
            or final_analysis != analysis
            or final_producer != producer
            or repeated != bundles
        ):
            raise M8L2DevelopmentError(
                "development config/source/session authority changed before publication"
            )
        repeated_campaign = _validate_not_created_context(
            capture=capture,
            producer=producer,
            bundles=repeated,
        )
        if _campaign_dict(repeated_campaign) != _campaign_dict(campaign):
            raise M8L2DevelopmentError("development campaign changed before publication")
        _assert_development_import_origins(project_root)
        _validate_lock_time(_utc_now())
        _write_bytes(marker, _NOT_CREATED_BYTES)
        _fsync_directory(root)
        return L2DevelopmentLockResult(
            root=root,
            aggregate_path=aggregate_path,
            aggregate_sha256=aggregate_sha,
            marker_path=marker,
            created_at_utc=created_at,
            children=(),
            status="NOT_CREATED",
            reason_codes=reasons,
        )
    except Exception:
        if not marker.exists():
            shutil.rmtree(root, ignore_errors=True)
            _fsync_directory(root.parent)
        raise


def lock_m8_l2_development(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_bundle_path: str | Path,
    validation_bundle_path: str | Path,
    lock_dir: str | Path,
    *,
    input_loader: L2DevelopmentInputVerifier | None = None,
    expected_session_file_authorities: Mapping[str, _ExpectedFileAuthority] | None = None,
) -> L2DevelopmentAuthorityResult:
    """Publish LOCKED state or a typed NOT_CREATED development authority."""

    project_root = capture_config.path.parent.parent.resolve()
    _assert_development_import_origins(project_root)
    capture, analysis = _revalidate_configs(capture_config, analysis_config)
    train_requested = Path(train_bundle_path)
    validation_requested = Path(validation_bundle_path)
    if train_requested.absolute() == validation_requested.absolute():
        raise M8L2DevelopmentError("train and validation must be distinct frozen session paths")
    try:
        train_bundle = verify_m8_l2_session_bundle(train_requested, expected_config=capture)
        validation_bundle = verify_m8_l2_session_bundle(
            validation_requested, expected_config=capture
        )
    except Exception as error:
        raise M8L2DevelopmentError("development session bundle verification failed") from error
    bundles = (train_bundle, validation_bundle)
    _verify_expected_session_file_authorities(bundles, expected_session_file_authorities)
    for bundle, (expected_date, expected_role) in zip(bundles, _EXPECTED_DEVELOPMENT, strict=True):
        if bundle.session_date != expected_date or bundle.role != expected_role:
            raise M8L2DevelopmentError(
                f"development input must be COMPLETE {expected_date} {expected_role} "
                "or the matching terminal INSUFFICIENT_DATA authority"
            )
    if any(bundle.status != "COMPLETE" for bundle in bundles):
        return _publish_not_created_development(
            capture=capture,
            analysis=analysis,
            bundles=bundles,
            lock_dir=lock_dir,
            expected_session_file_authorities=expected_session_file_authorities,
        )

    project_root = capture.path.parent.parent.resolve()
    producer = _producer_source_identity(project_root)
    _validate_source(producer)
    verifier = input_loader if input_loader is not None else _load_input_verifier()
    try:
        train_input = verifier(
            train_bundle.root,
            expected_config=capture,
            expected_date="2026-08-10",
            expected_role="train",
        )
        validation_input = verifier(
            validation_bundle.root,
            expected_config=capture,
            expected_date="2026-08-11",
            expected_role="validation",
            expected_campaign=train_input.campaign_identity,
        )
    except Exception as error:
        raise M8L2DevelopmentError("development input authority verification failed") from error
    _verify_input_descriptor(
        train_input,
        train_bundle,
        capture=capture,
        expected_date="2026-08-10",
        expected_role="train",
    )
    _verify_input_descriptor(
        validation_input,
        validation_bundle,
        capture=capture,
        expected_date="2026-08-11",
        expected_role="validation",
    )
    _validate_runtime_campaign(producer, train_input, validation_input)
    # Refuse to begin expensive development fitting outside the declared lock
    # window.  The deadline is checked again immediately before publishing the
    # aggregate authority because fitting all eight candidates can itself cross
    # the held-out boundary.
    _validate_lock_time(_utc_now())

    destination = Path(lock_dir).absolute()
    root = _reserve_destination(destination)
    marker_path = root / _LOCKED_MARKER
    try:
        endpoint_specs = _endpoint_specs(analysis)
        endpoint_config = {item.name: item for item in analysis.endpoints}
        declared_test_dates = tuple(value[0] for value in _EXPECTED_HELDOUT)
        children: list[L2DevelopmentChildLock] = []
        execution_claims: list[dict[str, object]] = []
        regime_claims: list[dict[str, object]] = []
        for symbol in capture.study.symbols:
            train_admission = _preflight_symbol_raw(train_input, symbol)
            validation_admission = _preflight_symbol_raw(validation_input, symbol)
            if train_admission is not None and validation_admission is not None:
                admitted_raw = train_admission[0] + validation_admission[0]
                _require_memory_budget(
                    admitted_raw,
                    _MAX_DEVELOPMENT_RAW_BYTES,
                    f"{symbol} train+validation raw Parquet admission",
                )
                _require_memory_budget(
                    _causal_build_upper_bytes(
                        train_admission[1] + validation_admission[1], len(endpoint_specs)
                    ),
                    _MAX_DEVELOPMENT_CAUSAL_BYTES,
                    f"{symbol} eight-frame causal build admission",
                )
            train_loaded = train_input.load_symbol_frames(symbol)
            _require_memory_budget(
                _loaded_bytes(train_loaded),
                _MAX_DEVELOPMENT_RAW_BYTES,
                f"{symbol} train raw materialization",
            )
            validation_loaded = validation_input.load_symbol_frames(symbol)
            _require_memory_budget(
                _loaded_bytes(train_loaded) + _loaded_bytes(validation_loaded),
                _MAX_DEVELOPMENT_RAW_BYTES,
                f"{symbol} train+validation raw materialization",
            )
            if train_admission is None or validation_admission is None:
                _require_memory_budget(
                    _causal_build_upper_bytes(
                        train_loaded.book_observations.height
                        + validation_loaded.book_observations.height,
                        len(endpoint_specs),
                    ),
                    _MAX_DEVELOPMENT_CAUSAL_BYTES,
                    f"{symbol} eight-frame causal build admission",
                )
            train_frames = build_l2_endpoint_frames(
                train_loaded.book_observations,
                train_loaded.depth_deltas,
                train_loaded.intervals,
                study_date="2026-08-10",
                study_role="train",
                feature_windows=analysis.features.rolling_windows,
                volatility_window=analysis.features.volatility_window,
                clock_max_state_age_ms=analysis.features.clock_max_state_age_ms,
                endpoints=endpoint_specs,
            )
            _require_memory_budget(
                sum(_frame_bytes(frame) for frame in train_frames.values()),
                _MAX_DEVELOPMENT_CAUSAL_BYTES,
                f"{symbol} train causal frames",
            )
            validation_frames = build_l2_endpoint_frames(
                validation_loaded.book_observations,
                validation_loaded.depth_deltas,
                validation_loaded.intervals,
                study_date="2026-08-11",
                study_role="validation",
                feature_windows=analysis.features.rolling_windows,
                volatility_window=analysis.features.volatility_window,
                clock_max_state_age_ms=analysis.features.clock_max_state_age_ms,
                endpoints=endpoint_specs,
            )
            _require_memory_budget(
                sum(_frame_bytes(frame) for frame in train_frames.values())
                + sum(_frame_bytes(frame) for frame in validation_frames.values()),
                _MAX_DEVELOPMENT_CAUSAL_BYTES,
                f"{symbol} eight accumulated causal frames",
            )
            first_endpoint = analysis.endpoints[0].name
            regime = fit_l2_regime_thresholds(
                train_frames[first_endpoint],
                lower_quantile=(
                    analysis.regimes.quantile_numerators[0] / analysis.regimes.quantile_denominator
                ),
                upper_quantile=(
                    analysis.regimes.quantile_numerators[1] / analysis.regimes.quantile_denominator
                ),
                volatility_column=analysis.regimes.feature,
            )
            regime_path = root / "references" / symbol.lower() / "regime_thresholds.json"
            regime_sha = _write_json(regime_path, _regime_payload(regime, analysis))
            train_reference_frame = apply_l2_regimes(train_frames[first_endpoint], regime)
            validation_reference_frame = apply_l2_regimes(validation_frames[first_endpoint], regime)
            execution_path = root / "references" / symbol.lower() / "execution_reference.json"
            execution_sha = _write_json(
                execution_path,
                _execution_reference(
                    symbol,
                    train_reference_frame,
                    validation_reference_frame,
                    analysis,
                ),
            )
            del train_reference_frame, validation_reference_frame
            execution_claims.append(
                {
                    "symbol": symbol,
                    "path": _relative(execution_path, root),
                    "sha256": execution_sha,
                }
            )
            regime_claims.append(
                {
                    "symbol": symbol,
                    "path": _relative(regime_path, root),
                    "sha256": regime_sha,
                }
            )
            for endpoint in analysis.endpoints:
                modeled_train = apply_l2_regimes(train_frames[endpoint.name], regime)
                modeled_validation = apply_l2_regimes(validation_frames[endpoint.name], regime)
                feature_columns = l2_model_feature_columns(
                    modeled_train, windows=analysis.features.rolling_windows
                )
                if feature_columns != analysis.features.model_feature_columns:
                    raise M8L2DevelopmentError(
                        "generated L2 feature ladder differs from frozen analysis contract"
                    )
                if (
                    l2_model_feature_columns(
                        modeled_validation, windows=analysis.features.rolling_windows
                    )
                    != feature_columns
                ):
                    raise M8L2DevelopmentError("validation feature ladder differs from train")
                _require_memory_budget(
                    _selection_workspace_upper_bytes(
                        modeled_train,
                        modeled_validation,
                        feature_count=len(feature_columns),
                    ),
                    _MAX_SELECTION_WORKSPACE_BYTES,
                    f"{symbol} {endpoint.name} selection scratch/NumPy admission",
                )
                selection = select_multidate_model(
                    (modeled_train, modeled_validation),
                    _model_config(capture),
                    feature_columns=feature_columns,
                    declared_test_dates=declared_test_dates,
                    seed=analysis.study.seed,
                    calibration_bins=analysis.calibration.bins,
                    target="future_mid_up",
                    calibration_fraction=capture.models.calibration_fraction,
                    bootstrap_draws=analysis.bootstrap.samples,
                    block_width_events=endpoint.nominal_event_block_width,
                )
                if selection.validation_comparison.height != 8:
                    raise M8L2DevelopmentError(
                        "frozen L2 model ladder must contain exactly eight candidates"
                    )
                children.append(
                    _write_child(
                        root=root,
                        symbol=symbol,
                        endpoint=endpoint_config[endpoint.name],
                        selection=selection,
                        regime_path=regime_path,
                        regime_sha256=regime_sha,
                        execution_path=execution_path,
                        execution_sha256=execution_sha,
                        capture=capture,
                        analysis=analysis,
                        producer=producer,
                        train_input=train_input,
                        validation_input=validation_input,
                    )
                )
                # Child publication retains only canonical JSON/Parquet and
                # compact digests; no endpoint frame or fitted scratch is
                # allowed to leak into the next coordinate.
                del selection, modeled_train, modeled_validation

            # The next symbol begins with no raw/causal locals from this one.
            del train_loaded, validation_loaded, train_frames, validation_frames, regime

        expected_children = tuple(
            (symbol, endpoint.name)
            for symbol in capture.study.symbols
            for endpoint in analysis.endpoints
        )
        if tuple((item.symbol, item.endpoint) for item in children) != expected_children:
            raise M8L2DevelopmentError("development child-lock set is incomplete or reordered")
        created_at = _validate_lock_time(_utc_now())
        aggregate_payload: dict[str, object] = {
            "schema_version": _LOCK_SCHEMA_VERSION,
            "artifact_kind": "m8_l2_outcome_blind_development_lock",
            "study": analysis.study.name,
            "created_at_utc": created_at,
            "must_precede_utc": _LOCK_DEADLINE.isoformat().replace("+00:00", "Z"),
            "capture_config_sha256": capture.hash,
            "capture_config_source_sha256": capture.source_sha256,
            "capture_protocol_sha256": analysis.study.capture_protocol_sha256,
            "analysis_config_sha256": analysis.hash,
            "analysis_config_source_sha256": analysis.source_sha256,
            "producer_source_identity": producer.to_dict(),
            "campaign_identity": _campaign_dict(train_input.campaign_identity),
            "development_inputs": [
                {
                    "date": train_input.session_date,
                    "role": train_input.role,
                    "session_id": train_input.session_id,
                    "file_authority": _file_authority_dict(train_input.file_authority),
                },
                {
                    "date": validation_input.session_date,
                    "role": validation_input.role,
                    "session_id": validation_input.session_id,
                    "file_authority": _file_authority_dict(validation_input.file_authority),
                },
            ],
            "children": [
                {
                    "symbol": item.symbol,
                    "endpoint": item.endpoint,
                    "path": _relative(item.path, root),
                    "sha256": item.sha256,
                    "selection_lock_sha256": item.selection_lock_sha256,
                    "final_fitted_state_sha256": item.fitted_state_sha256,
                }
                for item in children
            ],
            "regime_thresholds": regime_claims,
            "execution_references": execution_claims,
            "fit_policy": (
                "regimes_fit_Aug8_only; candidates_fit_Aug8_only_and_select_Aug9_only; "
                "selected_and_prior_fit_once_on_Aug8_plus_Aug9_before_lock"
            ),
            "heldout_declarations": [
                {"date": date_value, "role": role} for date_value, role in _EXPECTED_HELDOUT
            ],
            "heldout_access": {
                "paths_received": False,
                "file_hashes_received": False,
                "row_counts_received": False,
                "economic_rows_opened": False,
                "model_fit_or_update_after_lock": False,
            },
            "claims": {
                "cross_symbol_pooling": False,
                "p_values": False,
                "significance": False,
                "capacity": False,
                "realized_execution": False,
                "profitability": False,
            },
        }
        aggregate_path = root / _AGGREGATE_NAME
        aggregate_sha = _write_json(aggregate_path, aggregate_payload)
        _write_bytes(
            root / _AGGREGATE_DIGEST_NAME,
            f"{aggregate_sha}  {_AGGREGATE_NAME}\n".encode("ascii"),
        )
        _publish_inventory(root)
        _fsync_directory(root)
        # The terminal marker itself is the durable publication boundary.  A
        # lock that finished hashing just after the deadline must not become an
        # authority merely because its expensive work began in time.
        _validate_lock_time(_utc_now())
        final_capture, final_analysis = _revalidate_configs(capture, analysis)
        final_producer = _producer_source_identity(project_root)
        if final_capture != capture or final_analysis != analysis or final_producer != producer:
            raise M8L2DevelopmentError(
                "development config/source authority changed before lock publication"
            )
        if (
            _campaign_dict(train_input.campaign_identity).get("runtime_fingerprint_sha256")
            != current_m8_l2_runtime_fingerprint_sha256()
        ):
            raise M8L2DevelopmentError("development runtime changed before lock publication")
        repeated_bundles = tuple(
            verify_m8_l2_session_bundle(bundle.root, expected_config=capture) for bundle in bundles
        )
        if repeated_bundles != bundles:
            raise M8L2DevelopmentError(
                "development session authority changed before lock publication"
            )
        _verify_expected_session_file_authorities(
            repeated_bundles,
            expected_session_file_authorities,
        )
        _assert_development_import_origins(project_root)
        _validate_lock_time(_utc_now())
        _write_bytes(marker_path, _LOCKED_BYTES)
        _fsync_directory(root)
        return L2DevelopmentLockResult(
            root=root,
            aggregate_path=aggregate_path,
            aggregate_sha256=aggregate_sha,
            marker_path=marker_path,
            created_at_utc=created_at,
            children=tuple(children),
        )
    except Exception:
        if not marker_path.exists():
            shutil.rmtree(root, ignore_errors=True)
            _fsync_directory(root.parent)
        raise


def _parse_checksums(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise M8L2DevelopmentError("development checksums are not ASCII") from error
    result: dict[str, str] = {}
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n") or len(line) < 67 or line[64:66] != "  ":
            raise M8L2DevelopmentError("development checksum line is malformed")
        digest, relative = line[:64], line[66:-1]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise M8L2DevelopmentError("development checksum digest is malformed")
        relative = _safe_relative(relative)
        if relative in result:
            raise M8L2DevelopmentError("development checksum path is duplicated")
        result[relative] = digest
    if not result or list(result) != sorted(result):
        raise M8L2DevelopmentError("development checksum authority is empty or unordered")
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise M8L2DevelopmentError(f"{label} keys differ from the exact schema")


def _verify_reference_claims(
    root: Path,
    aggregate: Mapping[str, Any],
    analysis: M8L2AnalysisConfig,
) -> None:
    execution = aggregate.get("execution_references")
    regimes = aggregate.get("regime_thresholds")
    if not isinstance(execution, list) or not isinstance(regimes, list):
        raise M8L2DevelopmentError("aggregate reference claims are malformed")
    for label, claims, schema in (
        ("execution", execution, _EXECUTION_REFERENCE_SCHEMA_VERSION),
        ("regime", regimes, _REGIME_SCHEMA_VERSION),
    ):
        if len(claims) != len(analysis.study.symbols):
            raise M8L2DevelopmentError(f"aggregate {label} reference set is incomplete")
        for symbol, raw_claim in zip(analysis.study.symbols, claims, strict=True):
            if not isinstance(raw_claim, Mapping) or set(raw_claim) != {"symbol", "path", "sha256"}:
                raise M8L2DevelopmentError(f"aggregate {label} reference claim is malformed")
            if raw_claim["symbol"] != symbol:
                raise M8L2DevelopmentError(f"aggregate {label} reference order changed")
            relative = _safe_relative(str(raw_claim["path"]))
            payload, raw = _read_json(root / relative, f"{label} reference")
            if hashlib.sha256(raw).hexdigest() != raw_claim["sha256"]:
                raise M8L2DevelopmentError(f"aggregate {label} reference hash changed")
            if payload.get("schema_version") != schema or payload.get("symbol") != symbol:
                raise M8L2DevelopmentError(f"aggregate {label} reference semantics changed")
            if _canonical_json_bytes(cast(Mapping[str, object], payload)) != raw:
                raise M8L2DevelopmentError(f"aggregate {label} reference is not canonical JSON")
            if label == "execution" and (
                payload.get("reference_price_statistic")
                != analysis.execution.reference_price_statistic
                or payload.get("reference_depth_statistic")
                != analysis.execution.reference_depth_statistic
                or payload.get("quantity_policy") != analysis.execution.reference_quantity_policy
            ):
                raise M8L2DevelopmentError("execution reference differs from analysis contract")


def _restore_child(
    root: Path,
    claim: Mapping[str, Any],
    *,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    aggregate: Mapping[str, Any],
) -> L2DevelopmentChildLock:
    _exact_keys(
        claim,
        {
            "symbol",
            "endpoint",
            "path",
            "sha256",
            "selection_lock_sha256",
            "final_fitted_state_sha256",
        },
        "aggregate child claim",
    )
    child_path = root / _safe_relative(str(claim["path"]))
    child, child_raw = _read_json(child_path, "development child lock")
    if hashlib.sha256(child_raw).hexdigest() != claim["sha256"]:
        raise M8L2DevelopmentError("development child-lock SHA-256 changed")
    _exact_keys(
        child,
        {
            "schema_version",
            "artifact_kind",
            "symbol",
            "endpoint",
            "capture_config_sha256",
            "capture_config_source_sha256",
            "analysis_config_sha256",
            "analysis_config_source_sha256",
            "producer_source_identity",
            "development_inputs",
            "campaign_identity",
            "selection_policy",
            "final_fit_policy",
            "candidate_count",
            "selected_model",
            "selection_lock_path",
            "selection_lock_sha256",
            "final_fitted_state_path",
            "final_fitted_state_sha256",
            "validation_comparison_path",
            "validation_comparison_file_sha256",
            "validation_comparison_frame_sha256",
            "validation_comparison_rows",
            "regime_thresholds_path",
            "regime_thresholds_sha256",
            "execution_reference_path",
            "execution_reference_sha256",
            "development_frame_sha256",
            "test_rows_accessed",
            "heldout_fit_or_update_allowed",
        },
        "development child lock",
    )
    endpoint_by_name = {item.name: item for item in analysis.endpoints}
    symbol = str(claim["symbol"])
    endpoint_name = str(claim["endpoint"])
    endpoint = endpoint_by_name.get(endpoint_name)
    if endpoint is None:
        raise M8L2DevelopmentError("development child names an unknown endpoint")
    if (
        child.get("schema_version") != _CHILD_SCHEMA_VERSION
        or child.get("artifact_kind") != "m8_l2_symbol_endpoint_development_lock"
        or child.get("symbol") != symbol
        or child.get("endpoint") != _endpoint_payload(endpoint)
        or child.get("capture_config_sha256") != capture.hash
        or child.get("capture_config_source_sha256") != capture.source_sha256
        or child.get("analysis_config_sha256") != analysis.hash
        or child.get("analysis_config_source_sha256") != analysis.source_sha256
        or child.get("producer_source_identity") != aggregate.get("producer_source_identity")
        or child.get("development_inputs") != aggregate.get("development_inputs")
        or child.get("campaign_identity") != aggregate.get("campaign_identity")
        or child.get("candidate_count") != 8
        or child.get("test_rows_accessed") is not False
        or child.get("heldout_fit_or_update_allowed") is not False
    ):
        raise M8L2DevelopmentError("development child-lock contract changed")
    selection_relative = _safe_relative(str(child["selection_lock_path"]))
    selection_raw = _read_regular(
        root / selection_relative, label="selection lock", maximum=_MAX_JSON_BYTES
    )
    if not selection_raw.endswith(b"\n"):
        raise M8L2DevelopmentError("selection lock lacks canonical newline")
    selection_text = selection_raw[:-1].decode("ascii")
    selection = AnalysisLock.restore(selection_text, str(child["selection_lock_sha256"]))
    if selection.sha256 != claim["selection_lock_sha256"]:
        raise M8L2DevelopmentError("child selection lock differs from aggregate")
    if _canonical_json_bytes(cast(Mapping[str, object], selection.payload())) != selection_raw:
        raise M8L2DevelopmentError("selection lock is not canonical JSON")
    state_relative = _safe_relative(str(child["final_fitted_state_path"]))
    state_raw = _read_regular(root / state_relative, label="fitted state", maximum=_MAX_JSON_BYTES)
    if not state_raw.endswith(b"\n"):
        raise M8L2DevelopmentError("fitted state lacks canonical newline")
    state = FinalFittedState.restore(
        state_raw[:-1].decode("ascii"), str(child["final_fitted_state_sha256"])
    )
    if state.sha256 != claim["final_fitted_state_sha256"]:
        raise M8L2DevelopmentError("child fitted state differs from aggregate")
    selection_payload = selection.payload()
    if (
        selection_payload.get("final_fitted_state_sha256") != state.sha256
        or selection_payload.get("final_fitted_state") != state.payload()
        or selection_payload.get("development_frame_sha256")
        != child.get("development_frame_sha256")
        or selection_payload.get("selected_candidate", {}).get("name")
        != child.get("selected_model")
        or selection_payload.get("test_rows_accessed_during_selection") is not False
        or selection_payload.get("declared_test_dates") != [value[0] for value in _EXPECTED_HELDOUT]
    ):
        raise M8L2DevelopmentError("child selection and fitted-state authorities disagree")
    comparison_relative = _safe_relative(str(child["validation_comparison_path"]))
    comparison_path = root / comparison_relative
    comparison_raw = _read_regular(
        comparison_path, label="validation comparison", maximum=_MAX_COMPARISON_BYTES
    )
    if hashlib.sha256(comparison_raw).hexdigest() != child["validation_comparison_file_sha256"]:
        raise M8L2DevelopmentError("validation comparison file hash changed")
    try:
        comparison = pl.read_parquet(comparison_path)
    except Exception as error:
        raise M8L2DevelopmentError("validation comparison cannot be restored") from error
    if (
        comparison.height != child["validation_comparison_rows"]
        or comparison.height != 8
        or _frame_sha256(comparison) != child["validation_comparison_frame_sha256"]
        or _frame_sha256(comparison) != selection_payload.get("validation_comparison_sha256")
        or comparison.filter(pl.col("selected_on_validation")).height != 1
    ):
        raise M8L2DevelopmentError("validation comparison differs from child selection lock")
    return L2DevelopmentChildLock(
        symbol=symbol,
        endpoint=endpoint_name,
        path=child_path,
        sha256=str(claim["sha256"]),
        selection_lock_sha256=selection.sha256,
        fitted_state_sha256=state.sha256,
    )


def _verify_not_created_aggregate(
    *,
    capture: M8L2StudyConfig,
    analysis: M8L2AnalysisConfig,
    train_bundle_path: str | Path,
    validation_bundle_path: str | Path,
    root: Path,
    marker_path: Path,
    aggregate_path: Path,
    aggregate_sha: str,
    aggregate: Mapping[str, Any],
) -> L2DevelopmentAuthorityResult:
    _exact_keys(
        aggregate,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "study",
            "created_at_utc",
            "must_precede_utc",
            "capture_config_sha256",
            "capture_config_source_sha256",
            "capture_protocol_sha256",
            "analysis_config_sha256",
            "analysis_config_source_sha256",
            "producer_source_identity",
            "campaign_identity",
            "development_inputs",
            "reason_codes",
            "children",
            "heldout_declarations",
            "heldout_access",
            "claims",
        },
        "NOT_CREATED development authority",
    )
    if (
        aggregate.get("schema_version") != _NOT_CREATED_SCHEMA_VERSION
        or aggregate.get("artifact_kind") != "m8_l2_development_authority_not_created"
        or aggregate.get("status") != "NOT_CREATED"
        or aggregate.get("study") != analysis.study.name
        or aggregate.get("must_precede_utc") != _LOCK_DEADLINE.isoformat().replace("+00:00", "Z")
        or aggregate.get("capture_config_sha256") != capture.hash
        or aggregate.get("capture_config_source_sha256") != capture.source_sha256
        or aggregate.get("capture_protocol_sha256") != analysis.study.capture_protocol_sha256
        or aggregate.get("analysis_config_sha256") != analysis.hash
        or aggregate.get("analysis_config_source_sha256") != analysis.source_sha256
        or aggregate.get("children") != []
        or aggregate.get("heldout_declarations")
        != [{"date": value[0], "role": value[1]} for value in _EXPECTED_HELDOUT]
        or aggregate.get("heldout_access")
        != {
            "paths_received": False,
            "file_hashes_received": False,
            "row_counts_received": False,
            "economic_rows_opened": False,
            "model_fit_or_update_after_lock": False,
        }
        or aggregate.get("claims")
        != {
            "cross_symbol_pooling": False,
            "p_values": False,
            "significance": False,
            "capacity": False,
            "realized_execution": False,
            "profitability": False,
        }
    ):
        raise M8L2DevelopmentError("NOT_CREATED development contract changed")
    created_raw = aggregate.get("created_at_utc")
    if type(created_raw) is not str:
        raise M8L2DevelopmentError("NOT_CREATED creation time is malformed")
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise M8L2DevelopmentError("NOT_CREATED creation time is malformed") from error
    if not _EARLIEST_LOCK_TIME <= created < _LOCK_DEADLINE:
        raise M8L2DevelopmentError(
            "NOT_CREATED creation time is outside the frozen development window"
        )
    producer_payload = aggregate.get("producer_source_identity")
    if not isinstance(producer_payload, Mapping):
        raise M8L2DevelopmentError("NOT_CREATED producer identity is malformed")
    producer = _producer_source_identity(capture.path.parent.parent.resolve())
    _validate_source(producer)
    if producer.to_dict() != producer_payload:
        raise M8L2DevelopmentError(
            "current producer source differs from NOT_CREATED development authority"
        )
    try:
        bundles = (
            verify_m8_l2_session_bundle(train_bundle_path, expected_config=capture),
            verify_m8_l2_session_bundle(validation_bundle_path, expected_config=capture),
        )
    except Exception as error:
        raise M8L2DevelopmentError(
            "external NOT_CREATED development sessions no longer verify"
        ) from error
    for bundle, coordinate in zip(bundles, _EXPECTED_DEVELOPMENT, strict=True):
        if (bundle.session_date, bundle.role) != coordinate:
            raise M8L2DevelopmentError("external NOT_CREATED development coordinate changed")
    claims = aggregate.get("development_inputs")
    if claims != [_development_session_claim(bundle) for bundle in bundles]:
        raise M8L2DevelopmentError("external NOT_CREATED development session authority changed")
    campaigns = tuple(_campaign_from_session_bundle(bundle) for bundle in bundles)
    if _campaign_dict(campaigns[0]) != _campaign_dict(campaigns[1]) or _campaign_dict(
        campaigns[0]
    ) != aggregate.get("campaign_identity"):
        raise M8L2DevelopmentError("NOT_CREATED campaign identity changed")
    campaign = _campaign_dict(campaigns[0])
    if (
        campaign.get("runtime_commit") != producer.commit
        or campaign.get("runtime_source_tree_sha256") != producer.source_tree_sha256
        or campaign.get("runtime_dirty") is not False
    ):
        raise M8L2DevelopmentError("NOT_CREATED source and campaign identities disagree")
    reasons = _not_created_reasons(bundles)
    if not reasons or aggregate.get("reason_codes") != list(reasons):
        raise M8L2DevelopmentError("NOT_CREATED typed reasons changed")
    return L2DevelopmentLockResult(
        root=root,
        aggregate_path=aggregate_path,
        aggregate_sha256=aggregate_sha,
        marker_path=marker_path,
        created_at_utc=created_raw,
        children=(),
        status="NOT_CREATED",
        reason_codes=reasons,
    )


def verify_m8_l2_development_lock(
    capture_config: M8L2StudyConfig,
    analysis_config: M8L2AnalysisConfig,
    train_bundle_path: str | Path,
    validation_bundle_path: str | Path,
    lock_dir: str | Path,
    *,
    expected_lock_sha256: str | None = None,
) -> L2DevelopmentAuthorityResult:
    """Strictly restore either development authority and its external sessions."""

    capture, analysis = _revalidate_configs(capture_config, analysis_config)
    root = Path(lock_dir).absolute()
    _reject_symlink_components(root)
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise M8L2DevelopmentError("development lock is not a regular directory")
    except OSError as error:
        raise M8L2DevelopmentError("development lock directory is unavailable") from error
    files = _walk_regular(root)
    terminal_names = [name for name in (_LOCKED_MARKER, _NOT_CREATED_MARKER) if name in files]
    if len(terminal_names) != 1:
        raise M8L2DevelopmentError("development authority requires exactly one terminal marker")
    terminal_name = terminal_names[0]
    marker_path = root / terminal_name
    expected_marker = _LOCKED_BYTES if terminal_name == _LOCKED_MARKER else _NOT_CREATED_BYTES
    if _read_regular(marker_path, label="development marker", maximum=32) != expected_marker:
        raise M8L2DevelopmentError("development authority marker bytes differ")
    if _CHECKSUMS_NAME not in files or _INVENTORY_NAME not in files:
        raise M8L2DevelopmentError("development lock lacks checksum/inventory authority")
    checksums = _parse_checksums(
        _read_regular(root / _CHECKSUMS_NAME, label="development checksums", maximum=1_000_000)
    )
    if set(files) != set(checksums) | {_CHECKSUMS_NAME, terminal_name}:
        raise M8L2DevelopmentError("physical development inventory differs from checksums")
    for relative, expected_digest in checksums.items():
        if sha256_file(root / relative) != expected_digest:
            raise M8L2DevelopmentError(f"development checksum mismatch for {relative}")
    inventory, inventory_raw = _read_json(root / _INVENTORY_NAME, "development inventory")
    _exact_keys(inventory, {"schema_version", "artifact_kind", "files"}, "inventory")
    expected_inventory_kind = (
        "m8_l2_development_lock_inventory"
        if terminal_name == _LOCKED_MARKER
        else "m8_l2_development_not_created_inventory"
    )
    if (
        inventory.get("schema_version") != _INVENTORY_SCHEMA_VERSION
        or inventory.get("artifact_kind") != expected_inventory_kind
        or _canonical_json_bytes(cast(Mapping[str, object], inventory)) != inventory_raw
    ):
        raise M8L2DevelopmentError("development inventory authority changed")
    inventory_entries = inventory.get("files")
    expected_inventory = [
        {
            "path": relative,
            "sha256": checksums[relative],
            "bytes": (root / relative).stat().st_size,
        }
        for relative in checksums
        if relative != _INVENTORY_NAME
    ]
    if inventory_entries != expected_inventory:
        raise M8L2DevelopmentError("development inventory entries are not exact")

    aggregate_path = root / _AGGREGATE_NAME
    aggregate, aggregate_raw = _read_json(aggregate_path, "aggregate development lock")
    aggregate_sha = hashlib.sha256(aggregate_raw).hexdigest()
    if expected_lock_sha256 is not None and aggregate_sha != expected_lock_sha256:
        raise M8L2DevelopmentError("aggregate development lock differs from expected SHA-256")
    if checksums.get(_AGGREGATE_NAME) != aggregate_sha:
        raise M8L2DevelopmentError("checksums do not bind the aggregate development lock")
    digest_raw = _read_regular(
        root / _AGGREGATE_DIGEST_NAME,
        label="aggregate development digest",
        maximum=256,
    )
    if digest_raw != f"{aggregate_sha}  {_AGGREGATE_NAME}\n".encode("ascii"):
        raise M8L2DevelopmentError("aggregate development digest sidecar changed")
    if _canonical_json_bytes(cast(Mapping[str, object], aggregate)) != aggregate_raw:
        raise M8L2DevelopmentError("aggregate development lock is not canonical JSON")
    if terminal_name == _NOT_CREATED_MARKER:
        return _verify_not_created_aggregate(
            capture=capture,
            analysis=analysis,
            train_bundle_path=train_bundle_path,
            validation_bundle_path=validation_bundle_path,
            root=root,
            marker_path=marker_path,
            aggregate_path=aggregate_path,
            aggregate_sha=aggregate_sha,
            aggregate=aggregate,
        )
    _exact_keys(
        aggregate,
        {
            "schema_version",
            "artifact_kind",
            "study",
            "created_at_utc",
            "must_precede_utc",
            "capture_config_sha256",
            "capture_config_source_sha256",
            "capture_protocol_sha256",
            "analysis_config_sha256",
            "analysis_config_source_sha256",
            "producer_source_identity",
            "campaign_identity",
            "development_inputs",
            "children",
            "regime_thresholds",
            "execution_references",
            "fit_policy",
            "heldout_declarations",
            "heldout_access",
            "claims",
        },
        "aggregate development lock",
    )
    if (
        aggregate.get("schema_version") != _LOCK_SCHEMA_VERSION
        or aggregate.get("artifact_kind") != "m8_l2_outcome_blind_development_lock"
        or aggregate.get("study") != analysis.study.name
        or aggregate.get("must_precede_utc") != _LOCK_DEADLINE.isoformat().replace("+00:00", "Z")
        or aggregate.get("capture_config_sha256") != capture.hash
        or aggregate.get("capture_config_source_sha256") != capture.source_sha256
        or aggregate.get("capture_protocol_sha256") != analysis.study.capture_protocol_sha256
        or aggregate.get("analysis_config_sha256") != analysis.hash
        or aggregate.get("analysis_config_source_sha256") != analysis.source_sha256
        or aggregate.get("heldout_declarations")
        != [{"date": value[0], "role": value[1]} for value in _EXPECTED_HELDOUT]
        or aggregate.get("heldout_access")
        != {
            "paths_received": False,
            "file_hashes_received": False,
            "row_counts_received": False,
            "economic_rows_opened": False,
            "model_fit_or_update_after_lock": False,
        }
    ):
        raise M8L2DevelopmentError("aggregate development contract changed")
    created_raw = aggregate.get("created_at_utc")
    if type(created_raw) is not str:
        raise M8L2DevelopmentError("aggregate creation time is malformed")
    try:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise M8L2DevelopmentError("aggregate creation time is malformed") from error
    if not _EARLIEST_LOCK_TIME <= created < _LOCK_DEADLINE:
        raise M8L2DevelopmentError("aggregate creation time is outside the frozen lock window")
    producer_payload = aggregate.get("producer_source_identity")
    if not isinstance(producer_payload, Mapping):
        raise M8L2DevelopmentError("aggregate producer identity is malformed")
    producer = _producer_source_identity(capture.path.parent.parent.resolve())
    _validate_source(producer)
    if producer.to_dict() != producer_payload:
        raise M8L2DevelopmentError("current producer source differs from development lock")

    input_claims = aggregate.get("development_inputs")
    if not isinstance(input_claims, list) or len(input_claims) != 2:
        raise M8L2DevelopmentError("aggregate development-input set is malformed")
    verifier = _load_input_verifier()
    verified_inputs: list[_VerifiedInput] = []
    expected_campaign: object | None = None
    for bundle_path, expected_coordinate, claim in zip(
        (train_bundle_path, validation_bundle_path),
        _EXPECTED_DEVELOPMENT,
        input_claims,
        strict=True,
    ):
        if not isinstance(claim, Mapping) or set(claim) != {
            "date",
            "role",
            "session_id",
            "file_authority",
        }:
            raise M8L2DevelopmentError("aggregate development-input claim is malformed")
        try:
            from microstructure.m8_l2_inputs import L2SessionFileAuthority

            raw_authority = cast(Mapping[str, Any], claim["file_authority"])
            file_authority = L2SessionFileAuthority(
                manifest_sha256=str(raw_authority["manifest_sha256"]),
                checksums_sha256=str(raw_authority["checksums_sha256"]),
            )
            value = verifier(
                bundle_path,
                expected_config=capture,
                expected_date=expected_coordinate[0],
                expected_role=expected_coordinate[1],
                expected_file_authority=file_authority,
                expected_campaign=expected_campaign,
            )
        except Exception as error:
            raise M8L2DevelopmentError(
                "external development-input authority no longer verifies"
            ) from error
        if (
            value.session_id != claim["session_id"]
            or _file_authority_dict(value.file_authority) != claim["file_authority"]
        ):
            raise M8L2DevelopmentError("external development-input claim changed")
        verified_inputs.append(value)
        expected_campaign = value.campaign_identity
    if _campaign_dict(verified_inputs[0].campaign_identity) != aggregate.get(
        "campaign_identity"
    ) or _campaign_dict(verified_inputs[1].campaign_identity) != aggregate.get("campaign_identity"):
        raise M8L2DevelopmentError("external campaign identity differs from development lock")

    children_raw = aggregate.get("children")
    if not isinstance(children_raw, list):
        raise M8L2DevelopmentError("aggregate child-lock set is malformed")
    expected_children = tuple(
        (symbol, endpoint.name)
        for symbol in capture.study.symbols
        for endpoint in analysis.endpoints
    )
    if len(children_raw) != len(expected_children):
        raise M8L2DevelopmentError("aggregate child-lock set is incomplete")
    children: list[L2DevelopmentChildLock] = []
    for expected_child, raw_claim in zip(expected_children, children_raw, strict=True):
        if (
            not isinstance(raw_claim, Mapping)
            or (raw_claim.get("symbol"), raw_claim.get("endpoint")) != expected_child
        ):
            raise M8L2DevelopmentError("aggregate child-lock order changed")
        children.append(
            _restore_child(
                root,
                cast(Mapping[str, Any], raw_claim),
                capture=capture,
                analysis=analysis,
                aggregate=aggregate,
            )
        )
    _verify_reference_claims(root, aggregate, analysis)
    return L2DevelopmentLockResult(
        root=root,
        aggregate_path=aggregate_path,
        aggregate_sha256=aggregate_sha,
        marker_path=marker_path,
        created_at_utc=created_raw,
        children=tuple(children),
    )


__all__ = [
    "L2DevelopmentAuthorityResult",
    "L2DevelopmentChildLock",
    "L2DevelopmentInputVerifier",
    "L2DevelopmentLockResult",
    "M8L2DevelopmentError",
    "ProducerSourceIdentity",
    "lock_m8_l2_development",
    "verify_m8_l2_development_lock",
]
