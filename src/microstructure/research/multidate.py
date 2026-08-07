"""Locked, date-level evaluation primitives for the public trade-only study.

The module deliberately separates model selection from final-test evaluation.
``select_multidate_model`` accepts development data only and emits a content-
hashed analysis lock.  ``evaluate_locked_multidate_tests`` verifies that lock,
fits only the locked candidate and an independent historical prior once on the
combined train/validation reference period, and then opens all declared test
dates without updating either model.

Date-level uncertainty is computed from fixed 40-event block sufficient
statistics.  Bootstrap draws never materialize resampled event rows and are
generated in bounded chunks, which keeps the procedure usable for multi-million
row studies on a local machine.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from microstructure.config import ModelConfig
from microstructure.research.analysis import feature_stability_summary
from microstructure.research.models import (
    BootstrapResult,
    ModelCandidate,
    SigmoidCalibrator,
    build_model_candidates,
    classification_metrics,
    make_classifier,
)
from microstructure.research.splits import PurgedFold, WalkForwardPlan

DATE_BOOTSTRAP_DRAWS = 2_000
DATE_BOOTSTRAP_BLOCK_EVENTS = 40
_BOOTSTRAP_DRAW_CHUNK = 32
_MAX_BOOTSTRAP_INDEX_ELEMENTS = 1_000_000
_LOCK_SCHEMA_VERSION = "multidate-selection-lock-v1"
_DEVELOPMENT_ROLES = frozenset({"train", "validation"})
_TEST_ROLES = frozenset({"test", "primary_test", "replication_test"})
_ALL_ROLES = _DEVELOPMENT_ROLES | _TEST_ROLES

ReplicationStatus = Literal[
    "replicated",
    "failed_replication",
    "no_primary_improvement",
    "insufficient_replication_dates",
]


class MultiDateEvaluationError(ValueError):
    """Raised when the locked date-level protocol would be violated."""


@dataclass(frozen=True, slots=True)
class AnalysisLock:
    """Canonical JSON selection lock suitable for persistence before testing."""

    payload_json: str
    sha256: str

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> AnalysisLock:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(encoded, hashlib.sha256(encoded.encode()).hexdigest())

    @classmethod
    def restore(cls, payload_json: str, sha256: str) -> AnalysisLock:
        lock = cls(payload_json=payload_json, sha256=sha256.lower())
        lock.payload()
        return lock

    def payload(self) -> dict[str, Any]:
        observed = hashlib.sha256(self.payload_json.encode()).hexdigest()
        if observed != self.sha256:
            raise MultiDateEvaluationError("selection lock payload does not match its SHA-256")
        decoded = json.loads(self.payload_json)
        if not isinstance(decoded, dict):
            raise MultiDateEvaluationError("selection lock payload must be a JSON object")
        if decoded.get("schema_version") != _LOCK_SCHEMA_VERSION:
            raise MultiDateEvaluationError("unsupported selection lock schema version")
        return cast(dict[str, Any], decoded)


@dataclass(frozen=True, slots=True)
class LockedSelection:
    """Validation-only model selection plus its persistable analysis lock."""

    lock: AnalysisLock
    validation_comparison: pl.DataFrame
    selected_candidate: ModelCandidate
    feature_columns: tuple[str, ...]
    target: str
    train_dates: tuple[str, ...]
    validation_date: str
    declared_test_dates: tuple[str, ...]
    development_frame_sha256: str
    fitted_data_cutoff_policy: str

    @property
    def selected_model(self) -> str:
        return self.selected_candidate.name


@dataclass(frozen=True, slots=True)
class PairedDateLogLossResult:
    """Per-date paired loss diagnostics and equal-date-weighted uncertainty."""

    predictions: pl.DataFrame
    per_date: pl.DataFrame
    aggregate: BootstrapResult
    replication_status: ReplicationStatus


@dataclass(frozen=True, slots=True)
class LockedMultiDateTestResult:
    """Final locked predictions and descriptive diagnostics for all test dates."""

    plan: WalkForwardPlan
    predictions: pl.DataFrame
    paired_log_loss: PairedDateLogLossResult
    feature_stability: pl.DataFrame
    selected_model: str
    lock_sha256: str


@dataclass(frozen=True, slots=True)
class _FitMatrices:
    x_base: NDArray[np.float64]
    y_base: NDArray[np.int64]
    x_calibration: NDArray[np.float64]
    y_calibration: NDArray[np.int64]
    x_evaluate: NDArray[np.float64]
    fit_cutoff_ts_ns: int


@dataclass(frozen=True, slots=True)
class _FitOutcome:
    raw_probability: NDArray[np.float64]
    probability: NDArray[np.float64]
    fit_status: str
    fit_cutoff_ts_ns: int
    effective_candidate: ModelCandidate


@dataclass(frozen=True, slots=True)
class _SelectionSpec:
    lock: AnalysisLock
    candidate: ModelCandidate
    feature_columns: tuple[str, ...]
    target: str
    seed: int
    calibration_bins: int
    calibration_fraction: float
    train_dates: tuple[str, ...]
    validation_date: str
    test_dates: tuple[str, ...]
    development_frame_sha256: str
    block_width_events: int
    bootstrap_draws: int


_TEMPORAL_COLUMNS = frozenset(
    {
        "study_date",
        "study_role",
        "symbol",
        "decision_ts_ns",
        "decision_trade_id",
        "decision_sequence",
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
    }
)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MultiDateEvaluationError(f"{label} is missing required columns: {missing}")
    if frame.is_empty():
        raise MultiDateEvaluationError(f"{label} must not be empty")


def _parse_dates(values: Sequence[object], *, label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise MultiDateEvaluationError(f"{label} must contain unique dates")
    try:
        for value in normalized:
            date.fromisoformat(value)
    except ValueError as error:
        raise MultiDateEvaluationError(f"{label} must use ISO YYYY-MM-DD dates") from error
    return tuple(sorted(normalized))


def _timestamp_date(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Datetime("ns", time_zone="UTC")).dt.strftime("%Y-%m-%d")


def _validate_date_local_temporal_contract(frame: pl.DataFrame) -> None:
    _require_columns(frame, tuple(_TEMPORAL_COLUMNS), "multi-date evaluation frame")
    for column in (
        "study_date",
        "study_role",
        "symbol",
        "decision_ts_ns",
        "decision_trade_id",
        "decision_sequence",
        "continuity_id",
        "feature_continuity_id",
        "max_feature_source_ts_ns",
        "max_feature_source_trade_id",
        "label_start_ts_ns",
        "label_start_trade_id",
        "feature_ready",
        "right_censored",
    ):
        if frame.get_column(column).null_count():
            raise MultiDateEvaluationError(f"multi-date column {column!r} must not contain nulls")

    roles = set(str(value) for value in frame.get_column("study_role").unique().to_list())
    if not roles.issubset(_ALL_ROLES):
        raise MultiDateEvaluationError(f"unsupported study roles: {sorted(roles - _ALL_ROLES)}")
    _parse_dates(frame.get_column("study_date").unique().to_list(), label="study_date")

    date_roles = frame.group_by("study_date").agg(pl.col("study_role").n_unique().alias("n"))
    if date_roles.filter(pl.col("n") != 1).height:
        raise MultiDateEvaluationError("each study date must have exactly one study role")
    if frame.get_column("symbol").n_unique() != 1:
        raise MultiDateEvaluationError(
            "multi-date evaluation is per instrument; pool instruments only after reporting"
        )
    duplicate_identity = (
        frame.group_by("symbol", "study_date", "decision_sequence").len().filter(pl.col("len") != 1)
    )
    if duplicate_identity.height:
        raise MultiDateEvaluationError("date-level decision identities must be unique")

    if frame.filter(_timestamp_date("decision_ts_ns") != pl.col("study_date")).height:
        raise MultiDateEvaluationError("decision timestamps must fall inside study_date")
    if frame.filter(_timestamp_date("max_feature_source_ts_ns") != pl.col("study_date")).height:
        raise MultiDateEvaluationError("feature lineage must remain inside study_date")
    if frame.filter(pl.col("max_feature_source_ts_ns") > pl.col("decision_ts_ns")).height:
        raise MultiDateEvaluationError("feature lineage reaches beyond its decision")
    if frame.filter(
        (pl.col("max_feature_source_ts_ns") == pl.col("decision_ts_ns"))
        & (pl.col("max_feature_source_trade_id") > pl.col("decision_trade_id"))
    ).height:
        raise MultiDateEvaluationError("feature lineage reaches beyond its decision boundary")
    if frame.filter(pl.col("feature_continuity_id") != pl.col("continuity_id")).height:
        raise MultiDateEvaluationError("feature lookbacks must remain in decision continuity")

    if frame.filter(
        (pl.col("label_start_ts_ns") != pl.col("decision_ts_ns"))
        | (pl.col("label_start_trade_id") != pl.col("decision_trade_id"))
    ).height:
        raise MultiDateEvaluationError(
            "label start boundary must equal the decision timestamp and trade ID"
        )
    if frame.filter(_timestamp_date("label_start_ts_ns") != pl.col("study_date")).height:
        raise MultiDateEvaluationError("label starts must remain inside study_date")

    labeled = frame.filter(~pl.col("right_censored"))
    for column in (
        "label_continuity_id",
        "label_start_ts_ns",
        "label_start_trade_id",
        "label_information_end_ts_ns",
        "label_information_end_trade_id",
    ):
        if labeled.get_column(column).null_count():
            raise MultiDateEvaluationError(f"uncensored labels require non-null {column}")
    if labeled.filter(pl.col("label_continuity_id") != pl.col("continuity_id")).height:
        raise MultiDateEvaluationError("labels must remain in decision continuity")
    if labeled.filter(
        (pl.col("label_information_end_trade_id") <= pl.col("decision_trade_id"))
        | (pl.col("label_information_end_ts_ns") < pl.col("decision_ts_ns"))
        | (
            (pl.col("label_information_end_ts_ns") == pl.col("decision_ts_ns"))
            & (pl.col("label_information_end_trade_id") <= pl.col("decision_trade_id"))
        )
    ).height:
        raise MultiDateEvaluationError(
            "label information end must be strictly later by timestamp/trade-ID order"
        )
    if labeled.filter(
        _timestamp_date("label_information_end_ts_ns") != pl.col("study_date")
    ).height:
        raise MultiDateEvaluationError("label endpoints must remain inside study_date")

    for column in ("continuity_id", "feature_continuity_id"):
        reused = (
            frame.group_by(column)
            .agg(pl.col("study_date").n_unique().alias("date_count"))
            .filter(pl.col("date_count") != 1)
        )
        if reused.height:
            raise MultiDateEvaluationError(f"{column} cannot span study dates")
    label_reused = (
        labeled.group_by("label_continuity_id")
        .agg(pl.col("study_date").n_unique().alias("date_count"))
        .filter(pl.col("date_count") != 1)
    )
    if label_reused.height:
        raise MultiDateEvaluationError("label_continuity_id cannot span study dates")
    censored = frame.filter(pl.col("right_censored"))
    if censored.filter(
        pl.col("label_information_end_ts_ns").is_not_null()
        | pl.col("label_information_end_trade_id").is_not_null()
        | pl.col("label_continuity_id").is_not_null()
    ).height:
        raise MultiDateEvaluationError("censored label endpoints and continuity must remain null")


def _combine_date_frames(
    frames: Sequence[pl.DataFrame],
    *,
    allowed_roles: frozenset[str],
    label: str,
) -> pl.DataFrame:
    materialized = tuple(frames)
    if not materialized:
        raise MultiDateEvaluationError(f"{label} must include at least one per-date frame")
    normalized: list[tuple[str, pl.DataFrame]] = []
    for index, source in enumerate(materialized):
        _require_columns(source, tuple(_TEMPORAL_COLUMNS), f"{label}[{index}]")
        frame = source.with_columns(
            pl.col("study_date").cast(pl.String),
            pl.col("study_role").cast(pl.String),
        )
        dates = frame.get_column("study_date").unique().to_list()
        roles = frame.get_column("study_role").unique().to_list()
        if len(dates) != 1 or len(roles) != 1:
            raise MultiDateEvaluationError(f"{label}[{index}] must contain exactly one date/role")
        role = str(roles[0])
        if role not in allowed_roles:
            raise MultiDateEvaluationError(f"{label}[{index}] has disallowed role {role!r}")
        ordering = frame.select("decision_ts_ns", "decision_sequence").with_columns(
            pl.col("decision_ts_ns").shift(1).alias("_prior_ts"),
            pl.col("decision_sequence").shift(1).alias("_prior_sequence"),
        )
        if ordering.filter(
            (pl.col("decision_ts_ns") < pl.col("_prior_ts"))
            | (
                (pl.col("decision_ts_ns") == pl.col("_prior_ts"))
                & (pl.col("decision_sequence") < pl.col("_prior_sequence"))
            )
        ).height:
            raise MultiDateEvaluationError(
                f"{label}[{index}] must already be in decision/sequence order"
            )
        normalized.append((str(dates[0]), frame))
    # Per-date inputs are already physically ordered; ordering the small list of
    # frame descriptors avoids a full-width multi-million-row sort/copy.
    combined = pl.concat(
        [frame for _, frame in sorted(normalized, key=lambda item: item[0])],
        how="vertical",
    )
    _validate_date_local_temporal_contract(combined)
    return combined


def _eligible() -> pl.Expr:
    return pl.col("feature_ready") & (~pl.col("right_censored"))


def _frame_sha256(frame: pl.DataFrame) -> str:
    """Hash ordered row identities without serializing a second full frame."""

    digest = hashlib.sha256()
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode())
    row_hashes = frame.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3)
    for chunk in row_hashes.get_chunks():
        values = chunk.to_numpy().astype("<u8", copy=False)
        digest.update(values.tobytes(order="C"))
    digest.update(str(frame.height).encode())
    return digest.hexdigest()


def _chronological_calibration_split(
    train: pl.DataFrame, *, fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0.0 < fraction < 0.5:
        raise MultiDateEvaluationError("calibration_fraction must be between zero and one half")
    times = sorted(int(value) for value in train.get_column("decision_ts_ns").unique().to_list())
    if len(times) < 6:
        return train, train.head(0)
    calibration_count = max(2, math.ceil(len(times) * fraction))
    calibration_start = times[-calibration_count]
    base = train.filter(
        (pl.col("decision_ts_ns") < calibration_start)
        & (pl.col("label_information_end_ts_ns") < calibration_start)
    )
    calibration = train.filter(pl.col("decision_ts_ns") >= calibration_start)
    if base.height < 4 or calibration.height < 8:
        return train, train.head(0)
    return base, calibration


def _fit_matrices(
    train: pl.DataFrame,
    evaluate: pl.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    calibration_fraction: float,
) -> _FitMatrices:
    base, calibration = _chronological_calibration_split(train, fraction=calibration_fraction)
    fit_cutoff = train.get_column("label_information_end_ts_ns").max()
    if fit_cutoff is None:
        raise MultiDateEvaluationError("training data have no observable labels")
    empty_features = np.empty((0, len(features)), dtype=np.float64)
    empty_targets = np.empty(0, dtype=np.int64)
    return _FitMatrices(
        x_base=base.select(features).to_numpy().astype(np.float64, copy=False),
        y_base=base.get_column(target).to_numpy().astype(np.int64, copy=False),
        x_calibration=(
            calibration.select(features).to_numpy().astype(np.float64, copy=False)
            if not calibration.is_empty()
            else empty_features
        ),
        y_calibration=(
            calibration.get_column(target).to_numpy().astype(np.int64, copy=False)
            if not calibration.is_empty()
            else empty_targets
        ),
        x_evaluate=evaluate.select(features).to_numpy().astype(np.float64, copy=False),
        fit_cutoff_ts_ns=int(cast(int, fit_cutoff)),
    )


def _positive_probability(estimator: Any, features: NDArray[np.float64]) -> NDArray[np.float64]:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    if classes.size == 1:
        return np.full(features.shape[0], float(classes[0] == 1), dtype=np.float64)
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise MultiDateEvaluationError("classifier does not expose one binary positive class")
    return np.asarray(probabilities[:, int(positive[0])], dtype=np.float64)


def _fit_candidate(
    candidate: ModelCandidate,
    matrices: _FitMatrices,
    *,
    seed: int,
) -> _FitOutcome:
    effective = candidate
    fit_status = "ok"
    if np.unique(matrices.y_base).size < 2 and candidate.family != "baseline":
        effective = ModelCandidate(f"{candidate.name}__prior_fallback", "baseline")
        fit_status = "single_class_prior_fallback"
    estimator = make_classifier(effective, seed=seed)
    estimator.fit(matrices.x_base, matrices.y_base)
    calibrator = SigmoidCalibrator()
    if matrices.y_calibration.size:
        raw_calibration = _positive_probability(estimator, matrices.x_calibration)
        calibrator.fit(matrices.y_calibration, raw_calibration)
    raw = _positive_probability(estimator, matrices.x_evaluate)
    calibrated = calibrator.transform(raw)
    return _FitOutcome(
        raw_probability=raw,
        probability=calibrated,
        fit_status=f"{fit_status};{calibrator.status}",
        fit_cutoff_ts_ns=matrices.fit_cutoff_ts_ns,
        effective_candidate=effective,
    )


def _candidate_payload(candidate: ModelCandidate) -> dict[str, object]:
    return {
        "name": candidate.name,
        "family": candidate.family,
        "c": candidate.c,
        "max_depth": candidate.max_depth,
        "min_samples_leaf": candidate.min_samples_leaf,
    }


def _candidate_from_payload(payload: Mapping[str, Any]) -> ModelCandidate:
    try:
        return ModelCandidate(
            name=str(payload["name"]),
            family=cast(Any, str(payload["family"])),
            c=float(payload["c"]) if payload.get("c") is not None else None,
            max_depth=(int(payload["max_depth"]) if payload.get("max_depth") is not None else None),
            min_samples_leaf=int(payload["min_samples_leaf"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MultiDateEvaluationError(
            "selection lock has an invalid model specification"
        ) from error


def _validate_feature_contract(
    frame: pl.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
) -> None:
    if not features or len(set(features)) != len(features):
        raise MultiDateEvaluationError("feature_columns must be unique and nonempty")
    missing = sorted(set((*features, target)).difference(frame.columns))
    if missing:
        raise MultiDateEvaluationError(f"model columns are missing: {missing}")
    forbidden = [
        name
        for name in features
        if name.startswith("future_") or name.startswith("label_") or name == "right_censored"
    ]
    if forbidden:
        raise MultiDateEvaluationError(f"label/timing columns cannot be features: {forbidden}")
    eligible_target = frame.filter(_eligible()).get_column(target)
    if eligible_target.null_count():
        raise MultiDateEvaluationError("eligible rows must have a non-null classification target")
    target_values = set(eligible_target.unique().to_list())
    if not target_values.issubset({0, 1}):
        raise MultiDateEvaluationError("classification target must contain only 0 and 1")


def select_multidate_model(
    development_frames: Sequence[pl.DataFrame],
    model_config: ModelConfig,
    *,
    feature_columns: Sequence[str],
    declared_test_dates: Sequence[str],
    seed: int,
    calibration_bins: int,
    target: str = "future_trade_up",
    calibration_fraction: float = 0.2,
) -> LockedSelection:
    """Select on train/validation data without accepting or touching test rows.

    The caller must persist ``result.lock.payload_json`` and ``result.lock.sha256``
    before loading primary or replication test frames.
    """

    if model_config.selection_metric != "log_loss":
        raise MultiDateEvaluationError("the multi-date protocol freezes log_loss selection")
    if calibration_bins < 1:
        raise MultiDateEvaluationError("calibration_bins must be positive")
    development = _combine_date_frames(
        development_frames,
        allowed_roles=_DEVELOPMENT_ROLES,
        label="development_frames",
    )
    roles = {
        str(row["study_date"]): str(row["study_role"])
        for row in development.select("study_date", "study_role").unique().to_dicts()
    }
    train_dates = tuple(sorted(value for value, role in roles.items() if role == "train"))
    validation_dates = tuple(sorted(value for value, role in roles.items() if role == "validation"))
    if not train_dates or len(validation_dates) != 1:
        raise MultiDateEvaluationError(
            "development protocol requires at least one train date and one validation date"
        )
    validation_date = validation_dates[0]
    test_dates = _parse_dates(declared_test_dates, label="declared_test_dates")
    if len(test_dates) < 2:
        raise MultiDateEvaluationError("declare one primary and at least one replication date")
    if max(train_dates) >= validation_date or validation_date >= min(test_dates):
        raise MultiDateEvaluationError("study roles must be strictly chronological by date")

    features = tuple(str(value) for value in feature_columns)
    _validate_feature_contract(development, features=features, target=target)
    eligible = development.filter(_eligible() & pl.col(target).is_not_null())
    train = eligible.filter(pl.col("study_role") == "train")
    validation = eligible.filter(pl.col("study_role") == "validation")
    if train.is_empty() or validation.is_empty():
        raise MultiDateEvaluationError("train and validation must contain eligible labeled rows")
    validation_start = cast(int, validation.get_column("decision_ts_ns").min())
    train_label_end = cast(int, train.get_column("label_information_end_ts_ns").max())
    if train_label_end >= validation_start:
        raise MultiDateEvaluationError("training label information reaches validation")

    matrices = _fit_matrices(
        train,
        validation,
        features=features,
        target=target,
        calibration_fraction=calibration_fraction,
    )
    y_validation = validation.get_column(target).to_numpy().astype(np.int64, copy=False)
    candidates = build_model_candidates(model_config)
    comparison_rows: list[dict[str, object]] = []
    for order, candidate in enumerate(candidates):
        outcome = _fit_candidate(candidate, matrices, seed=seed)
        metrics = classification_metrics(
            y_validation,
            outcome.probability,
            calibration_bins=calibration_bins,
        )
        comparison_rows.append(
            {
                "candidate_order": order,
                "study_date": validation_date,
                "study_role": "validation",
                "requested_model": candidate.name,
                "requested_family": candidate.family,
                "model": outcome.effective_candidate.name,
                "family": outcome.effective_candidate.family,
                "fit_status": outcome.fit_status,
                "fit_cutoff_ts_ns": outcome.fit_cutoff_ts_ns,
                "n_obs": validation.height,
                **metrics,
            }
        )
    comparison = pl.DataFrame(comparison_rows, infer_schema_length=None)
    selectable = comparison.with_columns(
        pl.when(
            (pl.col("requested_family") != "baseline")
            & pl.col("fit_status").str.contains("single_class_prior_fallback")
        )
        .then(float("inf"))
        .otherwise(pl.col("log_loss"))
        .alias("_selection_score")
    ).sort("_selection_score", "candidate_order")
    selected_name = str(selectable.get_column("requested_model")[0])
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    comparison = comparison.with_columns(
        (pl.col("requested_model") == selected_name).alias("selected_on_validation"),
        pl.when(pl.col("requested_model") == selected_name)
        .then(pl.lit("validation_log_loss"))
        .otherwise(None)
        .alias("selected_on"),
        pl.lit(False).alias("test_rows_accessed"),
    ).sort("candidate_order")

    development_sha = _frame_sha256(development)
    comparison_sha = _frame_sha256(comparison)
    fitted_policy = (
        "selection fits each declared candidate on train-role rows only; final evaluation "
        "refits only the locked selected specification and an independent historical prior "
        "once on eligible train+validation rows; neither model updates between test dates"
    )
    lock = AnalysisLock.create(
        {
            "schema_version": _LOCK_SCHEMA_VERSION,
            "selected_candidate": _candidate_payload(selected),
            "declared_candidates": [_candidate_payload(candidate) for candidate in candidates],
            "validation_selection_scores": comparison.select(
                "candidate_order",
                "requested_model",
                "requested_family",
                "model",
                "family",
                "fit_status",
                "n_obs",
                "log_loss",
                "selected_on_validation",
            ).to_dicts(),
            "selection_metric": "log_loss",
            "target": target,
            "feature_columns": list(features),
            "seed": seed,
            "calibration_bins": calibration_bins,
            "calibration_fraction": calibration_fraction,
            "train_dates": list(train_dates),
            "validation_date": validation_date,
            "declared_test_dates": list(test_dates),
            "development_frame_sha256": development_sha,
            "validation_comparison_sha256": comparison_sha,
            "validation_rows": validation.height,
            "validation_start_ts_ns": validation_start,
            "validation_end_ts_ns": int(cast(int, validation.get_column("decision_ts_ns").max())),
            "selection_fit_cutoff_policy": (
                "all fitted labels end before the first validation decision"
            ),
            "final_fit_policy": fitted_policy,
            "test_rows_accessed_during_selection": False,
            "test_update_policy": "fit_once_before_primary_test; no updates through replication",
            "bootstrap": {
                "metric": "selected_minus_historical_prior_log_loss",
                "draws": DATE_BOOTSTRAP_DRAWS,
                "block_width_events": DATE_BOOTSTRAP_BLOCK_EVENTS,
                "date_weighting": "equal",
            },
        }
    )
    return LockedSelection(
        lock=lock,
        validation_comparison=comparison,
        selected_candidate=selected,
        feature_columns=features,
        target=target,
        train_dates=train_dates,
        validation_date=validation_date,
        declared_test_dates=test_dates,
        development_frame_sha256=development_sha,
        fitted_data_cutoff_policy=fitted_policy,
    )


def _indices(indexed: pl.DataFrame, condition: pl.Expr) -> NDArray[np.int64]:
    return (
        indexed.filter(condition)
        .get_column("_research_row_id")
        .to_numpy()
        .astype(np.int64, copy=False)
    )


def build_multidate_walk_forward_plan(frame: pl.DataFrame) -> WalkForwardPlan:
    """Build one date-level development fold and a frozen multi-date test plan."""

    _validate_date_local_temporal_contract(frame)
    if not frame.get_column("decision_ts_ns").is_sorted():
        raise MultiDateEvaluationError("plan frame must be sorted by decision time")
    roles = {
        str(row["study_date"]): str(row["study_role"])
        for row in frame.select("study_date", "study_role").unique().to_dicts()
    }
    train_dates = sorted(value for value, role in roles.items() if role == "train")
    validation_dates = sorted(value for value, role in roles.items() if role == "validation")
    test_dates = sorted(value for value, role in roles.items() if role in _TEST_ROLES)
    if not train_dates or len(validation_dates) != 1 or len(test_dates) < 2:
        raise MultiDateEvaluationError(
            "plan requires train date(s), one validation date, and at least two test dates"
        )
    if max(train_dates) >= validation_dates[0] or validation_dates[0] >= min(test_dates):
        raise MultiDateEvaluationError("plan roles are not strictly chronological")

    indexed = frame.with_row_index("_research_row_id")
    eligible = _eligible()
    train_candidates = indexed.filter(eligible & (pl.col("study_role") == "train"))
    validation_candidates = indexed.filter(eligible & (pl.col("study_role") == "validation"))
    test_candidates = indexed.filter(eligible & pl.col("study_role").is_in(_TEST_ROLES))
    if any(part.is_empty() for part in (train_candidates, validation_candidates, test_candidates)):
        raise MultiDateEvaluationError("one or more date-level plan roles have no eligible rows")
    validation_start = int(cast(int, validation_candidates.get_column("decision_ts_ns").min()))
    test_start = int(cast(int, test_candidates.get_column("decision_ts_ns").min()))
    train = train_candidates.filter(pl.col("label_information_end_ts_ns") < validation_start)
    validation = validation_candidates.filter(pl.col("label_information_end_ts_ns") < test_start)
    final_train = indexed.filter(
        eligible
        & pl.col("study_role").is_in(_DEVELOPMENT_ROLES)
        & (pl.col("label_information_end_ts_ns") < test_start)
    )
    test = test_candidates
    if any(part.is_empty() for part in (train, validation, final_train, test)):
        raise MultiDateEvaluationError("one or more date-level plan partitions are empty")
    train_indices = train.get_column("_research_row_id").to_numpy().astype(np.int64)
    validation_indices = validation.get_column("_research_row_id").to_numpy().astype(np.int64)
    final_train_indices = final_train.get_column("_research_row_id").to_numpy().astype(np.int64)
    test_indices = test.get_column("_research_row_id").to_numpy().astype(np.int64)
    fold = PurgedFold(
        fold_id=0,
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_start_ts_ns=int(cast(int, train.get_column("decision_ts_ns").min())),
        train_end_ts_ns=int(cast(int, train.get_column("decision_ts_ns").max())),
        validation_start_ts_ns=validation_start,
        validation_end_ts_ns=int(cast(int, validation.get_column("decision_ts_ns").max())),
        purged_rows=train_candidates.height - train.height,
        embargoed_time_buckets=0,
    )
    return WalkForwardPlan(
        folds=(fold,),
        final_train_indices=final_train_indices,
        test_indices=test_indices,
        test_start_ts_ns=test_start,
        test_end_ts_ns=int(cast(int, test.get_column("decision_ts_ns").max())),
        decision_time_count=frame.get_column("decision_ts_ns").n_unique(),
    )


def _selection_spec(selection: LockedSelection | AnalysisLock) -> _SelectionSpec:
    lock = selection.lock if isinstance(selection, LockedSelection) else selection
    payload = lock.payload()
    try:
        bootstrap = cast(Mapping[str, Any], payload["bootstrap"])
        spec = _SelectionSpec(
            lock=lock,
            candidate=_candidate_from_payload(
                cast(Mapping[str, Any], payload["selected_candidate"])
            ),
            feature_columns=tuple(str(value) for value in payload["feature_columns"]),
            target=str(payload["target"]),
            seed=int(payload["seed"]),
            calibration_bins=int(payload["calibration_bins"]),
            calibration_fraction=float(payload["calibration_fraction"]),
            train_dates=tuple(str(value) for value in payload["train_dates"]),
            validation_date=str(payload["validation_date"]),
            test_dates=tuple(str(value) for value in payload["declared_test_dates"]),
            development_frame_sha256=str(payload["development_frame_sha256"]),
            block_width_events=int(bootstrap["block_width_events"]),
            bootstrap_draws=int(bootstrap["draws"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MultiDateEvaluationError("selection lock is missing typed protocol fields") from error
    if spec.block_width_events != DATE_BOOTSTRAP_BLOCK_EVENTS:
        raise MultiDateEvaluationError("selection lock changed the frozen 40-event block width")
    if spec.bootstrap_draws != DATE_BOOTSTRAP_DRAWS:
        raise MultiDateEvaluationError("selection lock changed the frozen 2000-draw bootstrap")
    return spec


def _rows_by_indices(frame: pl.DataFrame, indices: NDArray[np.int64]) -> pl.DataFrame:
    return frame.with_row_index("_research_row_id").filter(
        pl.col("_research_row_id").is_in(indices)
    )


def _test_phase_expression(primary_date: str) -> pl.Expr:
    return (
        pl.when(pl.col("study_date") == primary_date)
        .then(pl.lit("primary"))
        .otherwise(pl.lit("replication"))
        .alias("test_phase")
    )


def _block_predictions(predictions: pl.DataFrame, *, block_width_events: int) -> pl.DataFrame:
    if block_width_events < 1:
        raise MultiDateEvaluationError("block_width_events must be positive")
    _require_columns(
        predictions,
        (
            "row_id",
            "study_date",
            "study_role",
            "decision_ts_ns",
            "decision_sequence",
            "y_true",
            "selected_probability",
            "prior_probability",
        ),
        "paired test predictions",
    )
    if predictions.get_column("row_id").n_unique() != predictions.height:
        raise MultiDateEvaluationError("paired predictions require unique row IDs")
    if not set(predictions.get_column("y_true").unique().to_list()).issubset({0, 1}):
        raise MultiDateEvaluationError("paired predictions require binary targets")
    for probability in ("selected_probability", "prior_probability"):
        if (
            predictions.get_column(probability).null_count()
            or predictions.filter(
                (~pl.col(probability).is_finite())
                | (pl.col(probability) < 0.0)
                | (pl.col(probability) > 1.0)
            ).height
        ):
            raise MultiDateEvaluationError(f"{probability} must contain finite probabilities")
    ordering = (
        predictions.select("row_id", "study_date", "decision_ts_ns", "decision_sequence")
        .sort("study_date", "decision_ts_ns", "decision_sequence", "row_id")
        .with_columns(pl.int_range(0, pl.len()).over("study_date").alias("_date_event_index"))
        .with_columns(
            (pl.col("_date_event_index") // block_width_events)
            .cast(pl.Int64)
            .alias("date_block_index")
        )
        .with_columns(
            pl.concat_str(
                "study_date",
                pl.col("date_block_index").cast(pl.String),
                separator=":",
            ).alias("date_block_id"),
            pl.lit(block_width_events, dtype=pl.Int64).alias("date_block_width_events"),
        )
        .drop("_date_event_index")
    )
    return predictions.join(
        ordering.select("row_id", "date_block_index", "date_block_id", "date_block_width_events"),
        on="row_id",
        how="inner",
        validate="1:1",
    )


def _loss_sufficient_statistics(blocked: pl.DataFrame) -> pl.DataFrame:
    clipped = blocked.with_columns(
        pl.col("selected_probability").clip(1e-12, 1.0 - 1e-12).alias("_selected_p"),
        pl.col("prior_probability").clip(1e-12, 1.0 - 1e-12).alias("_prior_p"),
    ).with_columns(
        pl.when(pl.col("y_true") == 1)
        .then(-pl.col("_selected_p").log())
        .otherwise(-(1.0 - pl.col("_selected_p")).log())
        .alias("_selected_loss"),
        pl.when(pl.col("y_true") == 1)
        .then(-pl.col("_prior_p").log())
        .otherwise(-(1.0 - pl.col("_prior_p")).log())
        .alias("_prior_loss"),
    )
    return (
        clipped.with_columns((pl.col("_selected_loss") - pl.col("_prior_loss")).alias("_loss_diff"))
        .group_by("study_date", "study_role", "test_phase", "date_block_index")
        .agg(
            pl.len().alias("event_count"),
            pl.col("_selected_loss").sum().alias("selected_loss_sum"),
            pl.col("_prior_loss").sum().alias("prior_loss_sum"),
            pl.col("_loss_diff").sum().alias("loss_diff_sum"),
        )
        .sort("study_date", "date_block_index")
    )


def _date_draws_from_blocks(
    loss_sums: NDArray[np.float64],
    event_counts: NDArray[np.int64],
    *,
    n_bootstrap: int,
    random: np.random.Generator,
    draw_chunk_size: int,
) -> NDArray[np.float64]:
    blocks = loss_sums.size
    if blocks < 2:
        return np.empty(0, dtype=np.float64)
    bounded_chunk = min(
        draw_chunk_size,
        max(1, _MAX_BOOTSTRAP_INDEX_ELEMENTS // blocks),
    )
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, bounded_chunk):
        stop = min(n_bootstrap, start + bounded_chunk)
        sampled = random.integers(0, blocks, size=(stop - start, blocks), dtype=np.int64)
        sampled_loss = np.take(loss_sums, sampled).sum(axis=1)
        sampled_count = np.take(event_counts, sampled).sum(axis=1)
        draws[start:stop] = sampled_loss / sampled_count
    return draws


def paired_date_log_loss(
    predictions: pl.DataFrame,
    *,
    seed: int,
    n_bootstrap: int = DATE_BOOTSTRAP_DRAWS,
    block_width_events: int = DATE_BOOTSTRAP_BLOCK_EVENTS,
    draw_chunk_size: int = _BOOTSTRAP_DRAW_CHUNK,
) -> PairedDateLogLossResult:
    """Compute per-date and equal-date-weighted selected-minus-prior log loss."""

    if n_bootstrap < 1 or draw_chunk_size < 1:
        raise MultiDateEvaluationError("bootstrap draws and chunk size must be positive")
    blocked = _block_predictions(predictions, block_width_events=block_width_events)
    if "test_phase" not in blocked.columns:
        dates = sorted(str(value) for value in blocked.get_column("study_date").unique())
        blocked = blocked.with_columns(_test_phase_expression(dates[0]))
    blocks = _loss_sufficient_statistics(blocked)
    dates = sorted(str(value) for value in blocks.get_column("study_date").unique())
    random = np.random.default_rng(seed)
    date_rows: list[dict[str, object]] = []
    date_draws: list[NDArray[np.float64]] = []
    all_dates_sufficient = True
    for study_date in dates:
        current = blocks.filter(pl.col("study_date") == study_date)
        loss_sums = current.get_column("loss_diff_sum").to_numpy().astype(np.float64)
        counts = current.get_column("event_count").to_numpy().astype(np.int64)
        selected_sum = float(cast(float, current.get_column("selected_loss_sum").sum()))
        prior_sum = float(cast(float, current.get_column("prior_loss_sum").sum()))
        observations = int(counts.sum())
        point = float(loss_sums.sum() / observations)
        draws = _date_draws_from_blocks(
            loss_sums,
            counts,
            n_bootstrap=n_bootstrap,
            random=random,
            draw_chunk_size=draw_chunk_size,
        )
        sufficient = draws.size == n_bootstrap
        all_dates_sufficient &= sufficient
        if sufficient:
            date_draws.append(draws)
        study_role = str(current.get_column("study_role")[0])
        test_phase = str(current.get_column("test_phase")[0])
        date_rows.append(
            {
                "study_date": study_date,
                "study_role": study_role,
                "test_phase": test_phase,
                "metric": "log_loss",
                "delta_definition": "selected_model_minus_historical_prior",
                "selected_log_loss": selected_sum / observations,
                "prior_log_loss": prior_sum / observations,
                "point_delta": point,
                "ci_low": float(np.quantile(draws, 0.025)) if sufficient else None,
                "ci_high": float(np.quantile(draws, 0.975)) if sufficient else None,
                "n_obs": observations,
                "n_blocks": current.height,
                "bootstrap_samples": n_bootstrap,
                "bootstrap_status": "ok" if sufficient else "insufficient_blocks",
                "block_width_events": block_width_events,
                "date_weight": 1.0 / len(dates),
                "point_favorable": point < 0.0,
                "significance_claim_authorized": False,
            }
        )
    per_date = pl.DataFrame(date_rows, infer_schema_length=None).sort("study_date")
    point_estimate = float(cast(float, per_date.get_column("point_delta").mean()))
    if len(dates) >= 2 and all_dates_sufficient:
        aggregate_draws = np.mean(np.vstack(date_draws), axis=0)
        aggregate = BootstrapResult(
            point_estimate=point_estimate,
            lower=float(np.quantile(aggregate_draws, 0.025)),
            upper=float(np.quantile(aggregate_draws, 0.975)),
            n_bootstrap=n_bootstrap,
            n_blocks=int(per_date.get_column("n_blocks").sum()),
            seed=seed,
            status="ok",
            draws=tuple(float(value) for value in aggregate_draws),
        )
    else:
        aggregate = BootstrapResult(
            point_estimate=point_estimate,
            lower=None,
            upper=None,
            n_bootstrap=n_bootstrap,
            n_blocks=int(per_date.get_column("n_blocks").sum()),
            seed=seed,
            status="insufficient_blocks",
            draws=(),
        )

    primary = per_date.filter(pl.col("test_phase") == "primary")
    replication = per_date.filter(pl.col("test_phase") == "replication")
    if primary.height != 1 or replication.is_empty():
        replication_status: ReplicationStatus = "insufficient_replication_dates"
    elif not bool(primary.get_column("point_favorable")[0]):
        replication_status = "no_primary_improvement"
    elif replication.get_column("point_favorable").all():
        replication_status = "replicated"
    else:
        replication_status = "failed_replication"
    return PairedDateLogLossResult(
        predictions=blocked,
        per_date=per_date,
        aggregate=aggregate,
        replication_status=replication_status,
    )


def reference_only_feature_stability(
    frame: pl.DataFrame,
    plan: WalkForwardPlan,
    *,
    feature_columns: Sequence[str],
    bins: int = 10,
    lock_sha256: str | None = None,
) -> pl.DataFrame:
    """Compare each test date with bins fitted only on final train+validation."""

    features = tuple(str(value) for value in feature_columns)
    reference = _rows_by_indices(frame, plan.final_train_indices).drop("_research_row_id")
    tests = _rows_by_indices(frame, plan.test_indices).drop("_research_row_id")
    reference_dates = ",".join(
        sorted(str(value) for value in reference.get_column("study_date").unique())
    )
    test_dates = sorted(str(value) for value in tests.get_column("study_date").unique())
    rows: list[pl.DataFrame] = []
    for index, study_date in enumerate(test_dates):
        comparison = tests.filter(pl.col("study_date") == study_date)
        rows.append(
            feature_stability_summary(
                reference,
                comparison,
                feature_columns=features,
                group_columns=("symbol",),
                bins=bins,
            ).with_columns(
                pl.lit(reference_dates).alias("reference_dates"),
                pl.lit("train_plus_validation").alias("reference_role"),
                pl.lit(study_date).alias("comparison_study_date"),
                pl.lit("primary" if index == 0 else "replication").alias("test_phase"),
                pl.lit(True).alias("reference_only"),
                pl.lit(lock_sha256).alias("selection_lock_sha256"),
            )
        )
    return pl.concat(rows, how="vertical").sort("comparison_study_date", "symbol", "feature")


def evaluate_locked_multidate_tests(
    development_frames: Sequence[pl.DataFrame],
    test_frames: Sequence[pl.DataFrame],
    selection: LockedSelection | AnalysisLock,
) -> LockedMultiDateTestResult:
    """Open declared tests under a persisted lock, with no between-date update."""

    spec = _selection_spec(selection)
    development = _combine_date_frames(
        development_frames,
        allowed_roles=_DEVELOPMENT_ROLES,
        label="development_frames",
    )
    if _frame_sha256(development) != spec.development_frame_sha256:
        raise MultiDateEvaluationError("development data changed after model selection")
    roles = {
        str(row["study_date"]): str(row["study_role"])
        for row in development.select("study_date", "study_role").unique().to_dicts()
    }
    observed_train = tuple(sorted(value for value, role in roles.items() if role == "train"))
    observed_validation = tuple(
        sorted(value for value, role in roles.items() if role == "validation")
    )
    if observed_train != spec.train_dates or observed_validation != (spec.validation_date,):
        raise MultiDateEvaluationError("development schedule changed after model selection")

    tests = _combine_date_frames(
        test_frames,
        allowed_roles=_TEST_ROLES,
        label="test_frames",
    )
    observed_tests = tuple(sorted(str(value) for value in tests.get_column("study_date").unique()))
    if observed_tests != spec.test_dates:
        raise MultiDateEvaluationError("test dates do not match the persisted selection lock")
    test_roles = {
        str(row["study_date"]): str(row["study_role"])
        for row in tests.select("study_date", "study_role").unique().to_dicts()
    }
    if test_roles[spec.test_dates[0]] not in {"test", "primary_test"}:
        raise MultiDateEvaluationError("the first declared test date must be primary_test")
    if any(
        test_roles[study_date] not in {"test", "replication_test"}
        for study_date in spec.test_dates[1:]
    ):
        raise MultiDateEvaluationError("later declared test dates must be replication_test")
    # The locked schedule guarantees every development date precedes every test
    # date, so concatenation preserves global order without sorting the wide frame.
    combined = pl.concat([development, tests], how="vertical")
    _validate_feature_contract(
        combined,
        features=spec.feature_columns,
        target=spec.target,
    )
    plan = build_multidate_walk_forward_plan(combined)
    final_train = _rows_by_indices(combined, plan.final_train_indices)
    final_test = _rows_by_indices(combined, plan.test_indices)
    matrices = _fit_matrices(
        final_train,
        final_test,
        features=spec.feature_columns,
        target=spec.target,
        calibration_fraction=spec.calibration_fraction,
    )
    selected = _fit_candidate(spec.candidate, matrices, seed=spec.seed)
    historical_prior = ModelCandidate("historical_prior", "baseline")
    prior = _fit_candidate(historical_prior, matrices, seed=spec.seed)
    first_test_decision = int(cast(int, final_test.get_column("decision_ts_ns").min()))
    if (
        selected.fit_cutoff_ts_ns >= first_test_decision
        or prior.fit_cutoff_ts_ns >= first_test_decision
    ):
        raise MultiDateEvaluationError("final fitting information reaches the primary test")

    identity_columns = [
        "_research_row_id",
        "study_date",
        "study_role",
        "symbol",
        "decision_ts_ns",
        "decision_sequence",
        "continuity_id",
    ]
    if "sample_id" in final_test.columns:
        identity_columns.append("sample_id")
    predictions = (
        final_test.select(*identity_columns, spec.target)
        .rename({"_research_row_id": "row_id", spec.target: "y_true"})
        .with_columns(
            pl.Series("selected_raw_probability", selected.raw_probability),
            pl.Series("selected_probability", selected.probability),
            pl.Series("prior_raw_probability", prior.raw_probability),
            pl.Series("prior_probability", prior.probability),
            _test_phase_expression(spec.test_dates[0]),
            pl.lit(spec.candidate.name).alias("selected_model"),
            pl.lit(selected.effective_candidate.name).alias("selected_effective_model"),
            pl.lit(selected.fit_status).alias("selected_fit_status"),
            pl.lit(selected.fit_cutoff_ts_ns).alias("selected_fit_cutoff_ts_ns"),
            pl.lit(prior.fit_status).alias("prior_fit_status"),
            pl.lit(prior.fit_cutoff_ts_ns).alias("prior_fit_cutoff_ts_ns"),
            pl.lit(spec.lock.sha256).alias("selection_lock_sha256"),
            pl.lit(True).alias("is_oos"),
            pl.lit(False).alias("model_updated_between_test_dates"),
        )
    )
    paired = paired_date_log_loss(
        predictions,
        seed=spec.seed + 20_000,
        n_bootstrap=spec.bootstrap_draws,
        block_width_events=spec.block_width_events,
    )
    stability = reference_only_feature_stability(
        combined,
        plan,
        feature_columns=spec.feature_columns,
        lock_sha256=spec.lock.sha256,
    )
    return LockedMultiDateTestResult(
        plan=plan,
        predictions=paired.predictions,
        paired_log_loss=paired,
        feature_stability=stability,
        selected_model=spec.candidate.name,
        lock_sha256=spec.lock.sha256,
    )


__all__ = [
    "DATE_BOOTSTRAP_BLOCK_EVENTS",
    "DATE_BOOTSTRAP_DRAWS",
    "AnalysisLock",
    "LockedMultiDateTestResult",
    "LockedSelection",
    "MultiDateEvaluationError",
    "PairedDateLogLossResult",
    "ReplicationStatus",
    "build_multidate_walk_forward_plan",
    "evaluate_locked_multidate_tests",
    "paired_date_log_loss",
    "reference_only_feature_stability",
    "select_multidate_model",
]
