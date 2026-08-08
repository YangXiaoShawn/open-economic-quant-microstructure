"""Locked, date-level evaluation primitives for the public trade-only study.

The module deliberately separates model selection from final-test evaluation.
``select_multidate_model`` accepts development data only, selects the requested
candidate, fits that candidate and an independent historical prior on the
combined train/validation reference period, and emits their transparent numeric
state in a content-hashed analysis lock.  ``evaluate_locked_multidate_tests``
restores only that verified state and predicts all declared test dates without
calling a fitting or update API.

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
from datetime import date, datetime
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import sklearn  # type: ignore[import-untyped]
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
_LOCK_SCHEMA_VERSION = "multidate-selection-lock-v2"
_FITTED_STATE_SCHEMA_VERSION = "multidate-fitted-model-state-v1"
_FITTED_STATE_ARTIFACT_KIND = "multidate_final_fitted_models"
_FITTED_STATE_SERIALIZATION_FORMAT = "canonical-json-numeric-v1"
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
class FinalFittedState:
    """Canonical, non-executable numeric state for the two final classifiers."""

    payload_json: str
    sha256: str

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalFittedState:
        encoded = _canonical_json_text(payload)
        state = cls(encoded, hashlib.sha256(encoded.encode()).hexdigest())
        state.payload()
        return state

    @classmethod
    def restore(cls, payload_json: str, sha256: str) -> FinalFittedState:
        state = cls(payload_json=payload_json, sha256=sha256.lower())
        state.payload()
        return state

    def payload(self) -> dict[str, Any]:
        observed = hashlib.sha256(self.payload_json.encode()).hexdigest()
        if observed != self.sha256:
            raise MultiDateEvaluationError("fitted-state payload does not match its SHA-256")
        decoded = _decode_json_object(self.payload_json, "fitted-state payload")
        if self.payload_json != _canonical_json_text(decoded):
            raise MultiDateEvaluationError("fitted-state payload is not canonical JSON")
        _validate_final_fitted_state_payload(decoded)
        return decoded

    def predict(
        self,
        role: Literal["selected", "historical_prior"],
        features: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        payload = self.payload()
        models = cast(Mapping[str, Any], payload["models"])
        model = cast(Mapping[str, Any], models[role])
        return _predict_serialized_model(model, features)


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
    fitted_state: FinalFittedState

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
    fitted_state: Mapping[str, object] | None


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
    fitted_state: FinalFittedState


def _canonical_json_text(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise MultiDateEvaluationError("fitted state is not finite canonical JSON") from error


def _decode_json_object(payload_json: str, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MultiDateEvaluationError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise MultiDateEvaluationError(f"{label} contains forbidden constant {value}")

    try:
        decoded = json.loads(
            payload_json,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except MultiDateEvaluationError:
        raise
    except (TypeError, json.JSONDecodeError) as error:
        raise MultiDateEvaluationError(f"{label} is not valid JSON") from error
    if not isinstance(decoded, dict) or not all(type(key) is str for key in decoded):
        raise MultiDateEvaluationError(f"{label} must be a JSON object")
    return cast(dict[str, Any], decoded)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise MultiDateEvaluationError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise MultiDateEvaluationError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultiDateEvaluationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise MultiDateEvaluationError(f"{label} must be >= {minimum}")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiDateEvaluationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultiDateEvaluationError(f"{label} must be finite")
    return result


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise MultiDateEvaluationError(f"{label} must be a nonempty string")
    return value


def _numeric_vector(value: object, label: str, *, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise MultiDateEvaluationError(f"{label} must contain exactly {length} values")
    return [_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _integer_vector(value: object, label: str, *, length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise MultiDateEvaluationError(f"{label} must contain exactly {length} integers")
    return [_strict_int(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _class_vector(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise MultiDateEvaluationError(f"{label} must be a nonempty class array")
    result = [_strict_int(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if result != sorted(set(result)) or not set(result).issubset({0, 1}):
        raise MultiDateEvaluationError(f"{label} must be unique ordered binary classes")
    return result


def _validate_candidate_payload_strict(value: object, label: str) -> ModelCandidate:
    payload = _mapping(value, label)
    _exact_keys(
        payload,
        frozenset({"name", "family", "c", "max_depth", "min_samples_leaf"}),
        label,
    )
    name = _string(payload["name"], f"{label}.name")
    family = _string(payload["family"], f"{label}.family")
    if family not in {"baseline", "logistic", "logistic_l2", "shallow_tree"}:
        raise MultiDateEvaluationError(f"{label} has an unsupported model family")
    raw_c = payload["c"]
    c = None if raw_c is None else _finite_float(raw_c, f"{label}.c")
    raw_depth = payload["max_depth"]
    max_depth = (
        None if raw_depth is None else _strict_int(raw_depth, f"{label}.max_depth", minimum=1)
    )
    min_samples_leaf = _strict_int(
        payload["min_samples_leaf"],
        f"{label}.min_samples_leaf",
        minimum=1,
    )
    if c is not None and c <= 0.0:
        raise MultiDateEvaluationError(f"{label}.c must be positive")
    if family == "logistic_l2" and c is None:
        raise MultiDateEvaluationError(f"{label}.c is required for logistic_l2")
    if family != "logistic_l2" and c is not None:
        raise MultiDateEvaluationError(f"{label}.c is only valid for logistic_l2")
    if family == "shallow_tree" and max_depth is None:
        raise MultiDateEvaluationError(f"{label}.max_depth is required for shallow_tree")
    if family != "shallow_tree" and max_depth is not None:
        raise MultiDateEvaluationError(f"{label}.max_depth is only valid for shallow_tree")
    return ModelCandidate(
        name=name,
        family=cast(Any, family),
        c=c,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )


def _validate_classifier_payload(
    value: object,
    *,
    feature_count: int,
    effective: ModelCandidate,
    label: str,
) -> None:
    payload = _mapping(value, label)
    kind = _string(payload.get("kind"), f"{label}.kind")
    if kind == "prior":
        _exact_keys(payload, frozenset({"kind", "classes", "class_probabilities"}), label)
        classes = _class_vector(payload["classes"], f"{label}.classes")
        probabilities = _numeric_vector(
            payload["class_probabilities"],
            f"{label}.class_probabilities",
            length=len(classes),
        )
        if any(not 0.0 <= item <= 1.0 for item in probabilities) or not math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise MultiDateEvaluationError(f"{label} has invalid class probabilities")
        if effective.family != "baseline":
            raise MultiDateEvaluationError(
                f"{label} prior state disagrees with effective candidate"
            )
        return
    if kind == "logistic":
        _exact_keys(
            payload,
            frozenset(
                {
                    "kind",
                    "classes",
                    "scaler_mean",
                    "scaler_scale",
                    "coefficients",
                    "intercept",
                }
            ),
            label,
        )
        if _class_vector(payload["classes"], f"{label}.classes") != [0, 1]:
            raise MultiDateEvaluationError(f"{label} logistic state requires classes [0, 1]")
        _numeric_vector(payload["scaler_mean"], f"{label}.scaler_mean", length=feature_count)
        scales = _numeric_vector(
            payload["scaler_scale"], f"{label}.scaler_scale", length=feature_count
        )
        if any(item <= 0.0 for item in scales):
            raise MultiDateEvaluationError(f"{label} scaler scales must be positive")
        _numeric_vector(payload["coefficients"], f"{label}.coefficients", length=feature_count)
        _finite_float(payload["intercept"], f"{label}.intercept")
        if effective.family not in {"logistic", "logistic_l2"}:
            raise MultiDateEvaluationError(
                f"{label} logistic state disagrees with effective candidate"
            )
        return
    if kind != "decision_tree":
        raise MultiDateEvaluationError(f"{label} has an unsupported classifier kind")
    _exact_keys(
        payload,
        frozenset(
            {
                "kind",
                "classes",
                "node_count",
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "positive_probability",
            }
        ),
        label,
    )
    if effective.family != "shallow_tree":
        raise MultiDateEvaluationError(f"{label} tree state disagrees with effective candidate")
    _class_vector(payload["classes"], f"{label}.classes")
    node_count = _strict_int(payload["node_count"], f"{label}.node_count", minimum=1)
    left = _integer_vector(payload["children_left"], f"{label}.children_left", length=node_count)
    right = _integer_vector(payload["children_right"], f"{label}.children_right", length=node_count)
    features = _integer_vector(payload["feature"], f"{label}.feature", length=node_count)
    _numeric_vector(payload["threshold"], f"{label}.threshold", length=node_count)
    positive = _numeric_vector(
        payload["positive_probability"],
        f"{label}.positive_probability",
        length=node_count,
    )
    if any(not 0.0 <= item <= 1.0 for item in positive):
        raise MultiDateEvaluationError(f"{label} has an invalid leaf probability")
    for index, (left_child, right_child, feature) in enumerate(
        zip(left, right, features, strict=True)
    ):
        leaf = left_child == -1 and right_child == -1
        if leaf:
            continue
        if not (0 <= left_child < node_count and 0 <= right_child < node_count):
            raise MultiDateEvaluationError(f"{label} node {index} has an invalid child")
        if not 0 <= feature < feature_count:
            raise MultiDateEvaluationError(f"{label} node {index} has an invalid feature")


def _validate_calibrator_payload(value: object, label: str) -> None:
    payload = _mapping(value, label)
    kind = _string(payload.get("kind"), f"{label}.kind")
    if kind == "identity":
        _exact_keys(payload, frozenset({"kind", "status"}), label)
        status = _string(payload["status"], f"{label}.status")
        if not status.startswith("identity_"):
            raise MultiDateEvaluationError(f"{label} identity status is invalid")
        return
    if kind != "sigmoid":
        raise MultiDateEvaluationError(f"{label} has an unsupported calibrator kind")
    _exact_keys(
        payload,
        frozenset({"kind", "status", "classes", "coefficient", "intercept"}),
        label,
    )
    if payload["status"] != "sigmoid":
        raise MultiDateEvaluationError(f"{label} sigmoid status is invalid")
    if _class_vector(payload["classes"], f"{label}.classes") != [0, 1]:
        raise MultiDateEvaluationError(f"{label} sigmoid state requires classes [0, 1]")
    _finite_float(payload["coefficient"], f"{label}.coefficient")
    _finite_float(payload["intercept"], f"{label}.intercept")


def _validate_serialized_model(
    value: object,
    *,
    role: str,
    feature_count: int,
) -> None:
    label = f"fitted state model {role}"
    payload = _mapping(value, label)
    _exact_keys(
        payload,
        frozenset(
            {
                "role",
                "requested_candidate",
                "effective_candidate",
                "fit_status",
                "fit_cutoff_ts_ns",
                "base_fit_rows",
                "calibration_rows",
                "imputer_statistics",
                "classifier",
                "calibrator",
            }
        ),
        label,
    )
    if payload["role"] != role:
        raise MultiDateEvaluationError(f"{label} has a mismatched role")
    _validate_candidate_payload_strict(payload["requested_candidate"], f"{label}.requested")
    effective = _validate_candidate_payload_strict(
        payload["effective_candidate"], f"{label}.effective"
    )
    _string(payload["fit_status"], f"{label}.fit_status")
    _strict_int(payload["fit_cutoff_ts_ns"], f"{label}.fit_cutoff_ts_ns", minimum=1)
    _strict_int(payload["base_fit_rows"], f"{label}.base_fit_rows", minimum=1)
    _strict_int(payload["calibration_rows"], f"{label}.calibration_rows", minimum=0)
    _numeric_vector(
        payload["imputer_statistics"],
        f"{label}.imputer_statistics",
        length=feature_count,
    )
    _validate_classifier_payload(
        payload["classifier"],
        feature_count=feature_count,
        effective=effective,
        label=f"{label}.classifier",
    )
    _validate_calibrator_payload(payload["calibrator"], f"{label}.calibrator")


def _validate_final_fitted_state_payload(payload: Mapping[str, Any]) -> None:
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "artifact_kind",
                "serialization_format",
                "library_versions",
                "feature_columns",
                "target",
                "development_frame_sha256",
                "fit_cutoff_ts_ns",
                "eligible_development_rows",
                "models",
            }
        ),
        "final fitted state",
    )
    if payload["schema_version"] != _FITTED_STATE_SCHEMA_VERSION:
        raise MultiDateEvaluationError("unsupported fitted-state schema version")
    if payload["artifact_kind"] != _FITTED_STATE_ARTIFACT_KIND:
        raise MultiDateEvaluationError("unsupported fitted-state artifact kind")
    if payload["serialization_format"] != _FITTED_STATE_SERIALIZATION_FORMAT:
        raise MultiDateEvaluationError("unsupported fitted-state serialization format")
    versions = _mapping(payload["library_versions"], "fitted-state library versions")
    _exact_keys(versions, frozenset({"numpy", "scikit_learn"}), "fitted-state library versions")
    _string(versions["numpy"], "fitted-state NumPy version")
    _string(versions["scikit_learn"], "fitted-state scikit-learn version")
    features_raw = payload["feature_columns"]
    if not isinstance(features_raw, list) or not features_raw:
        raise MultiDateEvaluationError("fitted-state feature columns must be nonempty")
    features = tuple(_string(item, "fitted-state feature") for item in features_raw)
    if len(set(features)) != len(features):
        raise MultiDateEvaluationError("fitted-state feature columns must be unique")
    _string(payload["target"], "fitted-state target")
    development_sha = _string(payload["development_frame_sha256"], "fitted-state development SHA")
    if len(development_sha) != 64 or any(
        character not in "0123456789abcdef" for character in development_sha
    ):
        raise MultiDateEvaluationError("fitted-state development SHA is invalid")
    cutoff = _strict_int(payload["fit_cutoff_ts_ns"], "fitted-state cutoff", minimum=1)
    _strict_int(
        payload["eligible_development_rows"],
        "fitted-state eligible rows",
        minimum=1,
    )
    models = _mapping(payload["models"], "fitted-state models")
    _exact_keys(models, frozenset({"selected", "historical_prior"}), "fitted-state models")
    for role in ("selected", "historical_prior"):
        _validate_serialized_model(models[role], role=role, feature_count=len(features))
        model = _mapping(models[role], f"fitted-state model {role}")
        if model["fit_cutoff_ts_ns"] != cutoff:
            raise MultiDateEvaluationError("fitted-state model cutoffs disagree")


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
        (pl.col("label_information_end_ts_ns") < pl.col("decision_ts_ns"))
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


def _float_list(values: object) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result = [float(value) for value in array]
    if any(not math.isfinite(value) for value in result):
        raise MultiDateEvaluationError("fitted estimator contains a non-finite parameter")
    return result


def _integer_list(values: object) -> list[int]:
    return [int(value) for value in np.asarray(values).reshape(-1)]


def _classifier_state(estimator: Any, effective: ModelCandidate) -> dict[str, object]:
    model = estimator.named_steps["model"]
    classes = _integer_list(model.classes_)
    if effective.family == "baseline":
        return {
            "kind": "prior",
            "classes": classes,
            "class_probabilities": _float_list(model.class_prior_),
        }
    if effective.family in {"logistic", "logistic_l2"}:
        scaler = estimator.named_steps["scale"]
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        intercept = np.asarray(model.intercept_, dtype=np.float64)
        if coefficients.shape[0] != 1 or intercept.shape != (1,):
            raise MultiDateEvaluationError("final logistic estimator is not binary")
        return {
            "kind": "logistic",
            "classes": classes,
            "scaler_mean": _float_list(scaler.mean_),
            "scaler_scale": _float_list(scaler.scale_),
            "coefficients": _float_list(coefficients[0]),
            "intercept": float(intercept[0]),
        }
    if effective.family != "shallow_tree":
        raise MultiDateEvaluationError("cannot serialize an unsupported final classifier")
    tree = model.tree_
    values = np.asarray(tree.value, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 1 or values.shape[0] != int(tree.node_count):
        raise MultiDateEvaluationError("final decision-tree state has an unsupported shape")
    positive_index = classes.index(1) if 1 in classes else None
    totals = values[:, 0, :].sum(axis=1)
    if np.any(totals <= 0.0):
        raise MultiDateEvaluationError("final decision tree contains an empty node")
    positive_probability = (
        np.zeros(int(tree.node_count), dtype=np.float64)
        if positive_index is None
        else values[:, 0, positive_index] / totals
    )
    return {
        "kind": "decision_tree",
        "classes": classes,
        "node_count": int(tree.node_count),
        "children_left": _integer_list(tree.children_left),
        "children_right": _integer_list(tree.children_right),
        "feature": _integer_list(tree.feature),
        "threshold": _float_list(tree.threshold),
        "positive_probability": _float_list(positive_probability),
    }


def _calibrator_state(calibrator: SigmoidCalibrator) -> dict[str, object]:
    if calibrator.estimator is None:
        return {"kind": "identity", "status": calibrator.status}
    estimator = calibrator.estimator
    classes = _integer_list(estimator.classes_)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    intercept = np.asarray(estimator.intercept_, dtype=np.float64)
    if classes != [0, 1] or coefficients.shape != (1, 1) or intercept.shape != (1,):
        raise MultiDateEvaluationError("final sigmoid calibrator is not binary")
    return {
        "kind": "sigmoid",
        "status": calibrator.status,
        "classes": classes,
        "coefficient": float(coefficients[0, 0]),
        "intercept": float(intercept[0]),
    }


def _serialized_model_state(
    *,
    role: Literal["selected", "historical_prior"],
    requested: ModelCandidate,
    effective: ModelCandidate,
    estimator: Any,
    calibrator: SigmoidCalibrator,
    fit_status: str,
    matrices: _FitMatrices,
) -> dict[str, object]:
    imputer = estimator.named_steps["imputer"]
    return {
        "role": role,
        "requested_candidate": _candidate_payload(requested),
        "effective_candidate": _candidate_payload(effective),
        "fit_status": fit_status,
        "fit_cutoff_ts_ns": matrices.fit_cutoff_ts_ns,
        "base_fit_rows": int(matrices.y_base.size),
        "calibration_rows": int(matrices.y_calibration.size),
        "imputer_statistics": _float_list(imputer.statistics_),
        "classifier": _classifier_state(estimator, effective),
        "calibrator": _calibrator_state(calibrator),
    }


def _stable_sigmoid(value: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(value, dtype=np.float64)
    nonnegative = value >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponential = np.exp(value[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _predict_serialized_model(
    payload: Mapping[str, Any],
    features: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    statistics = np.asarray(payload["imputer_statistics"], dtype=np.float64)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != statistics.size:
        raise MultiDateEvaluationError("test feature matrix disagrees with fitted state")
    imputed = np.where(np.isnan(matrix), statistics.reshape(1, -1), matrix)
    if not np.isfinite(imputed).all():
        raise MultiDateEvaluationError(
            "test feature matrix contains non-finite values after imputation"
        )
    classifier = cast(Mapping[str, Any], payload["classifier"])
    kind = str(classifier["kind"])
    if kind == "prior":
        classes = [int(value) for value in classifier["classes"]]
        probabilities = [float(value) for value in classifier["class_probabilities"]]
        positive = probabilities[classes.index(1)] if 1 in classes else 0.0
        raw = np.full(matrix.shape[0], positive, dtype=np.float64)
    elif kind == "logistic":
        mean = np.asarray(classifier["scaler_mean"], dtype=np.float64)
        scale = np.asarray(classifier["scaler_scale"], dtype=np.float64)
        coefficient = np.asarray(classifier["coefficients"], dtype=np.float64)
        linear = ((imputed - mean) / scale) @ coefficient + float(classifier["intercept"])
        raw = _stable_sigmoid(np.asarray(linear, dtype=np.float64))
    elif kind == "decision_tree":
        left = np.asarray(classifier["children_left"], dtype=np.int64)
        right = np.asarray(classifier["children_right"], dtype=np.int64)
        split_feature = np.asarray(classifier["feature"], dtype=np.int64)
        threshold = np.asarray(classifier["threshold"], dtype=np.float64)
        leaf_probability = np.asarray(classifier["positive_probability"], dtype=np.float64)
        tree_features = np.asarray(imputed, dtype=np.float32)
        raw = np.empty(matrix.shape[0], dtype=np.float64)
        for row_index in range(matrix.shape[0]):
            node = 0
            for _ in range(left.size + 1):
                if left[node] == -1 and right[node] == -1:
                    raw[row_index] = leaf_probability[node]
                    break
                node = (
                    int(left[node])
                    if tree_features[row_index, split_feature[node]] <= threshold[node]
                    else int(right[node])
                )
            else:
                raise MultiDateEvaluationError("fitted decision-tree state contains a cycle")
    else:
        raise MultiDateEvaluationError("fitted state has an unsupported classifier kind")

    calibrator = cast(Mapping[str, Any], payload["calibrator"])
    if calibrator["kind"] == "identity":
        calibrated = np.clip(raw, 1e-12, 1.0 - 1e-12)
    else:
        clipped = np.clip(raw, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped / (1.0 - clipped))
        calibrated = _stable_sigmoid(
            logit * float(calibrator["coefficient"]) + float(calibrator["intercept"])
        )
    raw = np.asarray(raw, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(calibrated).all()
        or np.any((raw < 0.0) | (raw > 1.0))
        or np.any((calibrated < 0.0) | (calibrated > 1.0))
    ):
        raise MultiDateEvaluationError("fitted state produced invalid probabilities")
    return raw, calibrated


def _fit_candidate(
    candidate: ModelCandidate,
    matrices: _FitMatrices,
    *,
    seed: int,
    state_role: Literal["selected", "historical_prior"] | None = None,
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
    if matrices.x_evaluate.shape[0]:
        raw = _positive_probability(estimator, matrices.x_evaluate)
        calibrated = calibrator.transform(raw)
    else:
        raw = np.empty(0, dtype=np.float64)
        calibrated = np.empty(0, dtype=np.float64)
    status = f"{fit_status};{calibrator.status}"
    fitted_state = (
        _serialized_model_state(
            role=state_role,
            requested=candidate,
            effective=effective,
            estimator=estimator,
            calibrator=calibrator,
            fit_status=status,
            matrices=matrices,
        )
        if state_role is not None
        else None
    )
    return _FitOutcome(
        raw_probability=raw,
        probability=calibrated,
        fit_status=status,
        fit_cutoff_ts_ns=matrices.fit_cutoff_ts_ns,
        effective_candidate=effective,
        fitted_state=fitted_state,
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
    bootstrap_draws: int = DATE_BOOTSTRAP_DRAWS,
    block_width_events: int = DATE_BOOTSTRAP_BLOCK_EVENTS,
) -> LockedSelection:
    """Select on train/validation data without accepting or touching test rows.

    The caller must persist ``result.lock.payload_json`` and ``result.lock.sha256``
    before loading primary or replication test frames.
    """

    if model_config.selection_metric != "log_loss":
        raise MultiDateEvaluationError("the multi-date protocol freezes log_loss selection")
    if calibration_bins < 1:
        raise MultiDateEvaluationError("calibration_bins must be positive")
    if bootstrap_draws < 1 or block_width_events < 1:
        raise MultiDateEvaluationError("bootstrap draws and block width must be positive")
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
    final_matrices = _fit_matrices(
        eligible,
        eligible.head(min(1_024, eligible.height)),
        features=features,
        target=target,
        calibration_fraction=calibration_fraction,
    )
    final_selected = _fit_candidate(
        selected,
        final_matrices,
        seed=seed,
        state_role="selected",
    )
    historical_prior = ModelCandidate("historical_prior", "baseline")
    final_prior = _fit_candidate(
        historical_prior,
        final_matrices,
        seed=seed,
        state_role="historical_prior",
    )
    if final_selected.fitted_state is None or final_prior.fitted_state is None:
        raise MultiDateEvaluationError("final development fit did not produce serializable state")
    primary_start = int(
        datetime.fromisoformat(f"{test_dates[0]}T00:00:00+00:00").timestamp() * 1_000_000_000
    )
    if final_matrices.fit_cutoff_ts_ns >= primary_start:
        raise MultiDateEvaluationError("final development fitting information reaches primary test")
    fitted_state = FinalFittedState.create(
        {
            "schema_version": _FITTED_STATE_SCHEMA_VERSION,
            "artifact_kind": _FITTED_STATE_ARTIFACT_KIND,
            "serialization_format": _FITTED_STATE_SERIALIZATION_FORMAT,
            "library_versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "feature_columns": list(features),
            "target": target,
            "development_frame_sha256": development_sha,
            "fit_cutoff_ts_ns": final_matrices.fit_cutoff_ts_ns,
            "eligible_development_rows": eligible.height,
            "models": {
                "selected": dict(final_selected.fitted_state),
                "historical_prior": dict(final_prior.fitted_state),
            },
        }
    )
    serialized_selected = fitted_state.predict("selected", final_matrices.x_evaluate)
    serialized_prior = fitted_state.predict("historical_prior", final_matrices.x_evaluate)
    if not (
        np.allclose(serialized_selected[0], final_selected.raw_probability, rtol=0.0, atol=1e-12)
        and np.allclose(serialized_selected[1], final_selected.probability, rtol=0.0, atol=1e-12)
        and np.allclose(serialized_prior[0], final_prior.raw_probability, rtol=0.0, atol=1e-12)
        and np.allclose(serialized_prior[1], final_prior.probability, rtol=0.0, atol=1e-12)
    ):
        raise MultiDateEvaluationError("serialized final fitted state changes model predictions")
    fitted_policy = (
        "selection fits each declared candidate on train-role rows only; final evaluation "
        "fits only the locked selected specification and an independent historical prior once "
        "on eligible train+validation rows before the lock; test evaluation restores numeric "
        "state and neither model updates between test dates"
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
            "final_fitted_state_sha256": fitted_state.sha256,
            "final_fitted_state": fitted_state.payload(),
            "test_rows_accessed_during_selection": False,
            "test_update_policy": "fit_once_before_primary_test; no updates through replication",
            "bootstrap": {
                "metric": "selected_minus_historical_prior_log_loss",
                "draws": bootstrap_draws,
                "block_width_events": block_width_events,
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
        fitted_state=fitted_state,
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
        fitted_payload = _mapping(payload["final_fitted_state"], "locked final fitted state")
        fitted_state = FinalFittedState.create(cast(Mapping[str, object], fitted_payload))
        claimed_state_sha = str(payload["final_fitted_state_sha256"])
        if fitted_state.sha256 != claimed_state_sha:
            raise MultiDateEvaluationError(
                "selection lock final fitted-state hash does not match its bytes"
            )
        spec = _SelectionSpec(
            lock=lock,
            candidate=_validate_candidate_payload_strict(
                payload["selected_candidate"], "selection lock selected candidate"
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
            fitted_state=fitted_state,
        )
    except MultiDateEvaluationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise MultiDateEvaluationError("selection lock is missing typed protocol fields") from error
    if spec.block_width_events < 1 or spec.bootstrap_draws < 1:
        raise MultiDateEvaluationError("selection lock bootstrap contract must be positive")
    fitted_payload = spec.fitted_state.payload()
    if (
        tuple(fitted_payload["feature_columns"]) != spec.feature_columns
        or fitted_payload["target"] != spec.target
        or fitted_payload["development_frame_sha256"] != spec.development_frame_sha256
    ):
        raise MultiDateEvaluationError("selection lock and fitted-state contracts disagree")
    selected_model = cast(
        Mapping[str, Any], cast(Mapping[str, Any], fitted_payload["models"])["selected"]
    )
    requested = _validate_candidate_payload_strict(
        selected_model["requested_candidate"], "fitted selected requested candidate"
    )
    if requested != spec.candidate:
        raise MultiDateEvaluationError(
            "selection lock candidate differs from fitted selected state"
        )
    if isinstance(selection, LockedSelection) and selection.fitted_state != fitted_state:
        raise MultiDateEvaluationError("in-memory fitted state differs from selection lock")
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
    final_test = _rows_by_indices(combined, plan.test_indices)
    test_matrix = final_test.select(spec.feature_columns).to_numpy().astype(np.float64, copy=False)
    selected_raw, selected_probability = spec.fitted_state.predict("selected", test_matrix)
    prior_raw, prior_probability = spec.fitted_state.predict("historical_prior", test_matrix)
    state_payload = spec.fitted_state.payload()
    state_models = cast(Mapping[str, Any], state_payload["models"])
    selected_state = cast(Mapping[str, Any], state_models["selected"])
    prior_state = cast(Mapping[str, Any], state_models["historical_prior"])
    selected_effective = _validate_candidate_payload_strict(
        selected_state["effective_candidate"], "locked selected effective candidate"
    )
    selected_cutoff = int(selected_state["fit_cutoff_ts_ns"])
    prior_cutoff = int(prior_state["fit_cutoff_ts_ns"])
    first_test_decision = int(cast(int, final_test.get_column("decision_ts_ns").min()))
    if selected_cutoff >= first_test_decision or prior_cutoff >= first_test_decision:
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
            pl.Series("selected_raw_probability", selected_raw),
            pl.Series("selected_probability", selected_probability),
            pl.Series("prior_raw_probability", prior_raw),
            pl.Series("prior_probability", prior_probability),
            _test_phase_expression(spec.test_dates[0]),
            pl.lit(spec.candidate.name).alias("selected_model"),
            pl.lit(selected_effective.name).alias("selected_effective_model"),
            pl.lit(str(selected_state["fit_status"])).alias("selected_fit_status"),
            pl.lit(selected_cutoff).alias("selected_fit_cutoff_ts_ns"),
            pl.lit(str(prior_state["fit_status"])).alias("prior_fit_status"),
            pl.lit(prior_cutoff).alias("prior_fit_cutoff_ts_ns"),
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
    "FinalFittedState",
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
