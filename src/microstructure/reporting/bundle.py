"""Read and verify immutable research run bundles.

Reporting is deliberately downstream of research computation.  A completed run
bundle contains frozen JSON/CSV/Parquet artifacts, a checksum manifest, and an
``_SUCCESS`` marker written only after every other file.  The dashboard and
report renderer use this module rather than importing modeling code.
"""

from __future__ import annotations

import csv
import hmac
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from microstructure.provenance import sha256_file

EvidenceTier = str

EVIDENCE_TIERS = frozenset({"SYNTHETIC_SMOKE", "PUBLIC_SAMPLE_PARTIAL", "FULL_DATA"})
SYNTHETIC_WATERMARK = (
    "SYNTHETIC SMOKE — SOFTWARE VALIDATION ONLY; NOT EMPIRICAL OR INVESTMENT EVIDENCE"
)
PUBLIC_SAMPLE_WATERMARK = (
    "PUBLIC SAMPLE / PARTIAL EVIDENCE — RESULTS ARE SAMPLE-SPECIFIC AND RESEARCH-ONLY"
)
FULL_DATA_WATERMARK = "FULL-DATA RESEARCH RUN — SIMULATED RESULTS ARE NOT LIVE-TRADING PERFORMANCE"

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")
_CHECKSUM_EXCLUSIONS = frozenset({"checksums.sha256", "_SUCCESS", "INSUFFICIENT_DATA"})


class RunBundleError(ValueError):
    """Base error for an unusable research run bundle."""


class IncompleteRunError(RunBundleError):
    """Raised when the producer has not finalized a run bundle."""


class RunBundleValidationError(RunBundleError):
    """Raised when provenance or artifact structure is internally inconsistent."""


class ChecksumMismatchError(RunBundleError):
    """Raised when frozen bundle bytes no longer match their manifest."""


def evidence_watermark(evidence_tier: str) -> str:
    """Return the mandatory, user-visible evidence label for a run."""
    labels = {
        "SYNTHETIC_SMOKE": SYNTHETIC_WATERMARK,
        "PUBLIC_SAMPLE_PARTIAL": PUBLIC_SAMPLE_WATERMARK,
        "FULL_DATA": FULL_DATA_WATERMARK,
    }
    try:
        return labels[evidence_tier]
    except KeyError as error:
        raise RunBundleValidationError(f"unsupported evidence tier {evidence_tier!r}") from error


@dataclass(frozen=True, slots=True)
class RunBundle:
    """Validated, read-only view of a completed run directory."""

    root: Path
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    quality: Mapping[str, Any]
    hypothesis_evaluation: Mapping[str, Any]
    predictive_metrics: tuple[Mapping[str, Any], ...]
    execution_metrics: tuple[Mapping[str, Any], ...]
    execution_sensitivity: tuple[Mapping[str, Any], ...]
    market_state: tuple[Mapping[str, Any], ...]

    @property
    def run_id(self) -> str:
        return cast(str, self.manifest["run_id"])

    @property
    def evidence_tier(self) -> str:
        return cast(str, self.manifest["evidence_tier"])

    @property
    def watermark(self) -> str:
        return evidence_watermark(self.evidence_tier)

    @property
    def data(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.manifest["data"])

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(value) for value in cast(Sequence[Any], self.data["symbols"]))

    @property
    def observed_start_utc(self) -> str:
        return cast(str, self.data["observed_start_utc"])

    @property
    def observed_end_utc(self) -> str:
        return cast(str, self.data["observed_end_utc"])


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RunBundleValidationError(
            f"cannot read valid JSON from {path.name}: {error}"
        ) from error


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise RunBundleValidationError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise RunBundleValidationError(f"{field} must be a UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RunBundleValidationError(f"{field} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise RunBundleValidationError(f"{field} must be explicitly UTC")
    return parsed


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise RunBundleValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_manifest(manifest: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RunBundleValidationError("run_manifest.json requires a non-empty run_id")
    if manifest.get("status") != "complete":
        raise IncompleteRunError("run_manifest.json status must be 'complete'")

    tier = manifest.get("evidence_tier")
    if not isinstance(tier, str) or tier not in EVIDENCE_TIERS:
        raise RunBundleValidationError(f"unsupported evidence tier {tier!r}")
    if provenance.get("evidence_tier") != tier:
        raise RunBundleValidationError("run manifest and provenance evidence tiers do not match")

    data = manifest.get("data")
    if not isinstance(data, Mapping):
        raise RunBundleValidationError("run_manifest.json requires a data object")
    mode = data.get("mode")
    source = data.get("source")
    if not isinstance(mode, str) or not isinstance(source, str):
        raise RunBundleValidationError("data.mode and data.source must be strings")
    synthetic_source = "synthetic" in mode.lower() or "synthetic" in source.lower()
    if tier == "SYNTHETIC_SMOKE" and not synthetic_source:
        raise RunBundleValidationError(
            "SYNTHETIC_SMOKE requires a clearly identified synthetic data source"
        )
    if tier != "SYNTHETIC_SMOKE" and synthetic_source:
        raise RunBundleValidationError(
            "synthetic data cannot be promoted to public-sample or full-data evidence"
        )
    if tier == "FULL_DATA" and data.get("all_requested_ranges_complete") is not True:
        raise RunBundleValidationError(
            "FULL_DATA requires manifested complete coverage for every requested range"
        )

    symbols = data.get("symbols")
    if (
        not isinstance(symbols, Sequence)
        or isinstance(symbols, str)
        or not symbols
        or not all(isinstance(symbol, str) and symbol for symbol in symbols)
    ):
        raise RunBundleValidationError("data.symbols must be a non-empty string list")
    observed_start = _utc(data.get("observed_start_utc"), "data.observed_start_utc")
    observed_end = _utc(data.get("observed_end_utc"), "data.observed_end_utc")
    if observed_end < observed_start:
        raise RunBundleValidationError("observed data period ends before it starts")

    _utc(provenance.get("generated_at_utc"), "provenance.generated_at_utc")
    _require_sha256(provenance.get("config_sha256"), "provenance.config_sha256")
    input_hashes = provenance.get("input_manifest_sha256")
    if not isinstance(input_hashes, list):
        raise RunBundleValidationError("provenance.input_manifest_sha256 must be a list")
    if tier != "SYNTHETIC_SMOKE" and not input_hashes:
        raise RunBundleValidationError(
            "public-sample and full-data runs require at least one input manifest SHA-256"
        )
    for index, digest in enumerate(input_hashes):
        _require_sha256(digest, f"provenance.input_manifest_sha256[{index}]")

    manifest_run_key = manifest.get("run_key")
    provenance_run_key = provenance.get("run_key")
    if manifest_run_key is not None or provenance_run_key is not None:
        manifest_digest = _require_sha256(manifest_run_key, "run_manifest.run_key")
        provenance_digest = _require_sha256(provenance_run_key, "provenance.run_key")
        if not hmac.compare_digest(manifest_digest, provenance_digest):
            raise RunBundleValidationError("run manifest and provenance run keys do not match")

    git = provenance.get("git")
    if not isinstance(git, Mapping):
        raise RunBundleValidationError("provenance.git must be an object")
    commit = git.get("commit")
    if commit != "UNBORN" and (
        not isinstance(commit, str) or _GIT_REVISION.fullmatch(commit) is None
    ):
        raise RunBundleValidationError(
            "provenance.git.commit must be UNBORN or a 40/64-character revision"
        )
    if not isinstance(git.get("dirty"), bool):
        raise RunBundleValidationError("provenance.git.dirty must be boolean")


def _safe_relative_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RunBundleValidationError(f"{field} must be a relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RunBundleValidationError(f"{field} cannot escape the run directory")
    destination = root.joinpath(*relative.parts)
    if not destination.is_relative_to(root):
        raise RunBundleValidationError(f"{field} cannot escape the run directory")
    return destination


def _artifact_path(
    root: Path,
    manifest: Mapping[str, Any],
    name: str,
    fallbacks: Sequence[str],
) -> Path | None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise RunBundleValidationError("run_manifest.json artifacts must be an object")
    declared = artifacts.get(name)
    if declared is not None:
        path = _safe_relative_path(root, declared, f"artifacts.{name}")
        if not path.is_file():
            raise RunBundleValidationError(
                f"declared artifact {name!r} does not exist: {path.relative_to(root)}"
            )
        return path
    for fallback in fallbacks:
        candidate = root / fallback
        if candidate.is_file():
            return candidate
    return None


def _records_from_json(path: Path) -> tuple[Mapping[str, Any], ...]:
    payload = _read_json(path)
    if isinstance(payload, Mapping):
        for key in ("rows", "records"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise RunBundleValidationError(f"{path.name} must contain a list of row objects")
    return tuple(cast(Mapping[str, Any], row) for row in payload)


def _read_records(path: Path | None) -> tuple[Mapping[str, Any], ...]:
    if path is None:
        return ()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _records_from_json(path)
    if suffix == ".csv":
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                return tuple(dict(row) for row in csv.DictReader(handle))
        except OSError as error:
            raise RunBundleValidationError(f"cannot read {path.name}: {error}") from error
    if suffix == ".parquet":
        try:
            import polars as pl

            return tuple(pl.read_parquet(path).to_dicts())
        except Exception as error:
            raise RunBundleValidationError(f"cannot read {path.name}: {error}") from error
    raise RunBundleValidationError(f"unsupported artifact format for {path.name}")


def _read_quality(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise RunBundleValidationError(f"{path.name} must contain a JSON object")
    return cast(Mapping[str, Any], payload)


def _bundle_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RunBundleValidationError(
                f"run bundles may not contain symlinks: {path.relative_to(root)}"
            )
        if path.is_file() and path.relative_to(root).as_posix() not in _CHECKSUM_EXCLUSIONS:
            files.append(path)
    return tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_checksum_manifest(run_dir: str | Path) -> Path:
    """Write deterministic checksums for all current run files.

    Call this only while staging a run, before creating ``_SUCCESS``.  The
    checksum file and completion marker are intentionally excluded.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise IncompleteRunError(f"run directory does not exist: {root}")
    if (root / "_SUCCESS").exists() and (root / "INSUFFICIENT_DATA").exists():
        raise IncompleteRunError("run has conflicting completion and insufficient-data markers")
    if (root / "_SUCCESS").exists():
        raise RunBundleValidationError(
            "cannot rewrite checksums for a completed run; create a new run bundle"
        )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in _bundle_files(root)
    ]
    if not lines:
        raise IncompleteRunError("cannot checksum an empty run directory")
    destination = root / "checksums.sha256"
    _atomic_write_text(destination, "\n".join(lines) + "\n")
    return destination


def verify_checksums(run_dir: str | Path) -> int:
    """Verify checksum coverage and return the number of protected files."""
    root = Path(run_dir).resolve()
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise IncompleteRunError("missing checksums.sha256")
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ChecksumMismatchError(f"cannot read checksums.sha256: {error}") from error
    if not lines:
        raise ChecksumMismatchError("checksums.sha256 is empty")

    declared: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ChecksumMismatchError(f"invalid checksums.sha256 line {line_number}")
        digest, relative_name = match.groups()
        if relative_name in declared:
            raise ChecksumMismatchError(f"duplicate checksum entry: {relative_name}")
        path = _safe_relative_path(root, relative_name, "checksum path")
        if not path.is_file() or path.is_symlink():
            raise ChecksumMismatchError(f"checksummed file is missing: {relative_name}")
        actual = sha256_file(path)
        if not hmac.compare_digest(actual, digest):
            raise ChecksumMismatchError(f"checksum mismatch: {relative_name}")
        declared[relative_name] = digest

    actual_files = {path.relative_to(root).as_posix() for path in _bundle_files(root)}
    declared_files = set(declared)
    missing = sorted(actual_files - declared_files)
    extra = sorted(declared_files - actual_files)
    if missing:
        raise ChecksumMismatchError("files absent from checksum manifest: " + ", ".join(missing))
    if extra:
        raise ChecksumMismatchError("checksum entries without files: " + ", ".join(extra))
    return len(declared)


def load_run_bundle(
    run_dir: str | Path,
    *,
    require_complete: bool = True,
    verify_integrity: bool = True,
) -> RunBundle:
    """Load a run after validating provenance, evidence tier, and integrity."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise IncompleteRunError(f"run directory does not exist: {root}")
    if (root / "_SUCCESS").exists() and (root / "INSUFFICIENT_DATA").exists():
        raise IncompleteRunError("run has conflicting completion and insufficient-data markers")
    if require_complete and not (root / "_SUCCESS").is_file():
        raise IncompleteRunError("missing _SUCCESS completion marker")
    if require_complete and verify_integrity:
        verify_checksums(root)

    manifest_path = root / "run_manifest.json"
    provenance_path = root / "provenance.json"
    if not manifest_path.is_file():
        raise IncompleteRunError("missing run_manifest.json")
    if not provenance_path.is_file():
        raise IncompleteRunError("missing provenance.json")
    manifest = _read_json_object(manifest_path)
    provenance = _read_json_object(provenance_path)
    _validate_manifest(manifest, provenance)

    predictive = _artifact_path(
        root,
        manifest,
        "predictive_metrics",
        (
            "metrics/predictive_metrics.json",
            "metrics/predictive_metrics.csv",
            "metrics/predictive_metrics.parquet",
        ),
    )
    execution = _artifact_path(
        root,
        manifest,
        "execution_metrics",
        (
            "metrics/execution_metrics.json",
            "metrics/execution_metrics.csv",
            "metrics/execution_metrics.parquet",
        ),
    )
    execution_sensitivity = _artifact_path(
        root,
        manifest,
        "execution_sensitivity",
        (
            "metrics/execution_sensitivity.json",
            "metrics/execution_sensitivity.csv",
            "metrics/execution_sensitivity.parquet",
        ),
    )
    market_state = _artifact_path(
        root,
        manifest,
        "market_state",
        (
            "dashboard/market_state.json",
            "dashboard/market_state.csv",
            "dashboard/market_state.parquet",
        ),
    )
    quality_path = _artifact_path(
        root,
        manifest,
        "quality_summary",
        ("quality/summary.json",),
    )
    hypothesis_path = _artifact_path(
        root,
        manifest,
        "hypothesis_evaluation",
        (),
    )
    return RunBundle(
        root=root,
        manifest=manifest,
        provenance=provenance,
        quality=_read_quality(quality_path),
        hypothesis_evaluation=_read_quality(hypothesis_path),
        predictive_metrics=_read_records(predictive),
        execution_metrics=_read_records(execution),
        execution_sensitivity=_read_records(execution_sensitivity),
        market_state=_read_records(market_state),
    )
