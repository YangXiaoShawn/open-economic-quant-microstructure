"""Lock-only evaluation and market-order scenarios for the M8 live-L2 study.

This module intentionally exposes no fitting, calibration, selection, or regime-
threshold API.  Its only model operation is restoring numeric development state
through :class:`~microstructure.research.multidate.FinalFittedState` and asking
that state for probabilities on already-built primary/replication endpoint
frames.

All uncertainty is descriptive.  Paired log-loss differences resample frozen-
width overlapping windows locally inside each continuity/OBSERVED interval,
never pool symbols, never compute a p-value, and weight the primary and
replication sessions equally.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from microstructure.config import ExecutionConfig
from microstructure.execution.simulator import simulate_predictions
from microstructure.m8_l2_analysis_config import (
    M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256,
    M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256,
)
from microstructure.research.l2_multidate import (
    L2EndpointSpec,
    L2ResearchError,
    validate_l2_endpoint_frame,
)
from microstructure.research.models import classification_metrics
from microstructure.research.multidate import FinalFittedState

HeldoutRole = Literal["primary_test", "replication_test"]

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HELDOUT_ROLES = frozenset({"primary_test", "replication_test"})
_EXPECTED_EVENT_LATENCIES = (0, 1, 5)
_EXPECTED_REGIMES = tuple(
    f"{volatility}__{liquidity}"
    for volatility in ("low", "medium", "high")
    for liquidity in ("liquid", "normal", "stressed")
)
_ALL_REGIME = "ALL"
_TARGET = "future_mid_up"
_REFERENCE_PRICE_STATISTIC = "train_median_mid_price"
_REFERENCE_DEPTH_STATISTIC = "train_q05_min_bid_ask_l1_depth"
_REFERENCE_SCHEMA_VERSION = "m8-l2-execution-reference-v1"
_FROZEN_ENDPOINTS: Mapping[str, tuple[str, int, str, int, int]] = {
    # domain, horizon, unit, paired-block width, impact OFI window
    "event_20": ("event", 20, "events", 40, 20),
    "event_100": ("event", 100, "events", 200, 100),
    "clock_1000ms": ("clock", 1_000, "milliseconds", 2_000, 20),
    "clock_5000ms": ("clock", 5_000, "milliseconds", 10_000, 100),
}


class L2LockedEvaluationError(ValueError):
    """Raised when lock-only evaluation or scenario replay would be invalid."""


def _sha256(value: str, label: str) -> str:
    if _HEX_SHA256.fullmatch(value) is None:
        raise L2LockedEvaluationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise L2LockedEvaluationError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _require(frame: pl.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise L2LockedEvaluationError(f"{label} is missing required columns: {missing}")
    if frame.is_empty():
        raise L2LockedEvaluationError(f"{label} must not be empty")


def _null_nonfinite(frame: pl.DataFrame) -> pl.DataFrame:
    """Make every tabular output safe for strict ``allow_nan=False`` JSON."""

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


def _candidate_name(model: Mapping[str, Any], label: str) -> str:
    candidate = _mapping(model.get("requested_candidate"), f"{label} requested candidate")
    value = candidate.get("name")
    if not isinstance(value, str) or not value:
        raise L2LockedEvaluationError(f"{label} requested candidate has no name")
    return value


def _validate_frozen_endpoint(endpoint: L2EndpointSpec) -> None:
    expected = _FROZEN_ENDPOINTS.get(endpoint.name)
    if expected is None:
        raise L2LockedEvaluationError(f"unknown frozen M8 L2 endpoint: {endpoint.name!r}")
    domain, horizon, unit, block_width, impact_window = expected
    observed_block = (
        endpoint.paired_block_events
        if endpoint.domain == "event"
        else endpoint.paired_block_milliseconds
    )
    observed = (
        endpoint.domain,
        endpoint.horizon_value,
        endpoint.horizon_unit,
        observed_block,
        endpoint.impact_ofi_window,
    )
    if observed != (domain, horizon, unit, block_width, impact_window):
        raise L2LockedEvaluationError(
            f"endpoint {endpoint.name!r} differs from the frozen M8 L2 endpoint contract"
        )


@dataclass(frozen=True, slots=True)
class LockedL2EndpointState:
    """One externally verified child lock and its restored numeric model state."""

    symbol: str
    endpoint: L2EndpointSpec
    child_lock_sha256: str
    aggregate_lock_sha256: str
    regime_thresholds_sha256: str
    fitted_state: FinalFittedState

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.upper():
            raise L2LockedEvaluationError("locked L2 symbol must be nonempty uppercase text")
        _sha256(self.child_lock_sha256, "child lock SHA-256")
        _sha256(self.aggregate_lock_sha256, "aggregate lock SHA-256")
        _sha256(self.regime_thresholds_sha256, "regime-threshold SHA-256")
        _validate_frozen_endpoint(self.endpoint)
        payload = self.fitted_state.payload()
        if payload.get("target") != _TARGET:
            raise L2LockedEvaluationError("locked L2 fitted state must target future_mid_up")
        features = payload.get("feature_columns")
        if (
            not isinstance(features, list)
            or not features
            or not all(isinstance(value, str) and value for value in features)
            or len(set(features)) != len(features)
        ):
            raise L2LockedEvaluationError("locked L2 fitted-state features are invalid")
        models = _mapping(payload.get("models"), "locked L2 fitted-state models")
        if set(models) != {"selected", "historical_prior"}:
            raise L2LockedEvaluationError(
                "locked L2 fitted state requires selected and historical-prior models"
            )
        _candidate_name(_mapping(models["selected"], "selected model"), "selected model")
        prior_name = _candidate_name(
            _mapping(models["historical_prior"], "historical-prior model"),
            "historical-prior model",
        )
        if prior_name != "historical_prior":
            raise L2LockedEvaluationError("locked baseline must be historical_prior")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(cast(list[str], self.fitted_state.payload()["feature_columns"]))

    @property
    def selected_model(self) -> str:
        models = _mapping(self.fitted_state.payload()["models"], "fitted-state models")
        return _candidate_name(_mapping(models["selected"], "selected model"), "selected model")

    def fit_cutoff(self, role: Literal["selected", "historical_prior"]) -> int:
        models = _mapping(self.fitted_state.payload()["models"], "fitted-state models")
        model = _mapping(models[role], f"{role} model")
        value = model.get("fit_cutoff_ts_ns")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise L2LockedEvaluationError(f"{role} model has an invalid fit cutoff")
        return value


@dataclass(frozen=True, slots=True)
class L2HeldoutEndpointFrame:
    """One verified primary or replication endpoint frame."""

    symbol: str
    endpoint_name: str
    study_date: str
    study_role: HeldoutRole
    frame: pl.DataFrame

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.upper():
            raise L2LockedEvaluationError("held-out L2 symbol must be uppercase")
        if not self.endpoint_name:
            raise L2LockedEvaluationError("held-out endpoint name must not be empty")
        if self.study_role not in _HELDOUT_ROLES:
            raise L2LockedEvaluationError("held-out frame role must be primary or replication")
        try:
            # ISO syntax is also checked by validate_l2_endpoint_frame; this
            # lightweight guard prevents ambiguous mapping keys before that call.
            year, month, day = (int(value) for value in self.study_date.split("-"))
            if year < 1 or not 1 <= month <= 12 or not 1 <= day <= 31:
                raise ValueError
        except (TypeError, ValueError):
            raise L2LockedEvaluationError("held-out study date must use YYYY-MM-DD") from None


@dataclass(frozen=True, slots=True)
class L2EvaluationResult:
    """Frozen lock-only predictions and descriptive endpoint diagnostics."""

    predictions: pl.DataFrame
    predictive_metrics: pl.DataFrame
    paired_by_session_regime: pl.DataFrame
    equal_session_summary: pl.DataFrame
    signed_markout: pl.DataFrame


@dataclass(frozen=True, slots=True)
class L2ExecutionReference:
    """Development-lock-persisted market-scenario sizing authority."""

    symbol: str
    training_date: str
    reference_mid_price: float
    train_l1_depth_q05: float
    lot_size: float
    reference_quantity: float
    reference_sha256: str
    aggregate_lock_sha256: str
    reference_price_statistic: str = _REFERENCE_PRICE_STATISTIC
    reference_depth_statistic: str = _REFERENCE_DEPTH_STATISTIC
    analysis_config_source_sha256: str = M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256
    analysis_config_semantic_sha256: str = M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.upper():
            raise L2LockedEvaluationError("execution-reference symbol must be uppercase")
        try:
            from datetime import date

            date.fromisoformat(self.training_date)
        except (TypeError, ValueError):
            raise L2LockedEvaluationError(
                "execution-reference training date must use YYYY-MM-DD"
            ) from None
        if self.training_date != "2026-08-10":
            raise L2LockedEvaluationError(
                "M8 L2 execution reference must come from the Aug 10 train session"
            )
        _sha256(self.aggregate_lock_sha256, "execution-reference aggregate-lock SHA-256")
        if self.reference_price_statistic != _REFERENCE_PRICE_STATISTIC:
            raise L2LockedEvaluationError("execution reference must use train median midpoint")
        if self.reference_depth_statistic != _REFERENCE_DEPTH_STATISTIC:
            raise L2LockedEvaluationError(
                "execution reference must use train q05 minimum bid/ask L1 depth"
            )
        if self.analysis_config_source_sha256 != M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256:
            raise L2LockedEvaluationError("execution reference has the wrong analysis source hash")
        if self.analysis_config_semantic_sha256 != M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256:
            raise L2LockedEvaluationError(
                "execution reference has the wrong analysis semantic hash"
            )
        values = (
            self.reference_mid_price,
            self.train_l1_depth_q05,
            self.lot_size,
            self.reference_quantity,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise L2LockedEvaluationError("execution-reference values must be finite and positive")
        cap = min(100.0 / self.reference_mid_price, 0.10 * self.train_l1_depth_q05)
        expected = math.floor((cap + self.lot_size * 1e-12) / self.lot_size) * self.lot_size
        if expected <= 0.0 or not math.isclose(
            self.reference_quantity,
            expected,
            rel_tol=0.0,
            abs_tol=max(1e-15, self.lot_size * 1e-10),
        ):
            raise L2LockedEvaluationError(
                "persisted execution quantity disagrees with the frozen development formula"
            )
        observed_sha = hashlib.sha256(_json(self.payload()).encode("utf-8")).hexdigest()
        if _sha256(self.reference_sha256, "execution-reference SHA-256") != observed_sha:
            raise L2LockedEvaluationError("execution-reference payload does not match its SHA-256")

    def payload(self) -> dict[str, object]:
        """Return the complete canonical development-reference hash payload."""

        return {
            "schema_version": _REFERENCE_SCHEMA_VERSION,
            "symbol": self.symbol,
            "training_date": self.training_date,
            "reference_mid_price": float(self.reference_mid_price),
            "train_l1_depth_q05": float(self.train_l1_depth_q05),
            "lot_size": float(self.lot_size),
            "reference_quantity": float(self.reference_quantity),
            "reference_price_statistic": self.reference_price_statistic,
            "reference_depth_statistic": self.reference_depth_statistic,
            "analysis_config_source_sha256": self.analysis_config_source_sha256,
            "analysis_config_semantic_sha256": self.analysis_config_semantic_sha256,
            "aggregate_lock_sha256": self.aggregate_lock_sha256,
        }

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        training_date: str,
        reference_mid_price: float,
        train_l1_depth_q05: float,
        lot_size: float,
        reference_quantity: float,
        aggregate_lock_sha256: str,
    ) -> L2ExecutionReference:
        """Create the canonical persistable reference after development fitting."""

        payload: dict[str, object] = {
            "schema_version": _REFERENCE_SCHEMA_VERSION,
            "symbol": symbol,
            "training_date": training_date,
            "reference_mid_price": float(reference_mid_price),
            "train_l1_depth_q05": float(train_l1_depth_q05),
            "lot_size": float(lot_size),
            "reference_quantity": float(reference_quantity),
            "reference_price_statistic": _REFERENCE_PRICE_STATISTIC,
            "reference_depth_statistic": _REFERENCE_DEPTH_STATISTIC,
            "analysis_config_source_sha256": M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256,
            "analysis_config_semantic_sha256": M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256,
            "aggregate_lock_sha256": aggregate_lock_sha256,
        }
        return cls(
            symbol=symbol,
            training_date=training_date,
            reference_mid_price=reference_mid_price,
            train_l1_depth_q05=train_l1_depth_q05,
            lot_size=lot_size,
            reference_quantity=reference_quantity,
            reference_sha256=hashlib.sha256(_json(payload).encode("utf-8")).hexdigest(),
            aggregate_lock_sha256=aggregate_lock_sha256,
        )


@dataclass(frozen=True, slots=True)
class L2MarketExecutionResult:
    """Market-only latency ledgers and explicitly non-claiming scenario summaries."""

    orders: pl.DataFrame
    fills: pl.DataFrame
    positions: pl.DataFrame
    metrics: pl.DataFrame
    assumptions: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _PairedDelta:
    selected_log_loss: float | None
    prior_log_loss: float | None
    point_delta: float | None
    ci_low: float | None
    ci_high: float | None
    n_obs: int
    n_blocks: int
    samples: int
    seed: int
    status: str
    draws: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _MovingBlockInterval:
    """Bounded sufficient statistics for one interval's overlapping blocks."""

    full: NDArray[np.float64]
    tail: NDArray[np.float64] | None
    blocks_per_draw: int


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\x1f".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _frame_key(value: L2HeldoutEndpointFrame) -> tuple[str, str, HeldoutRole]:
    return value.symbol, value.endpoint_name, value.study_role


def _state_key(value: LockedL2EndpointState) -> tuple[str, str]:
    return value.symbol, value.endpoint.name


def _validate_frame_for_state(
    value: L2HeldoutEndpointFrame,
    state: LockedL2EndpointState,
) -> pl.DataFrame:
    try:
        validate_l2_endpoint_frame(value.frame)
    except L2ResearchError as error:
        raise L2LockedEvaluationError(str(error)) from error
    _require(
        value.frame,
        (
            "feature_ready",
            _TARGET,
            "joint_market_regime",
            "volatility_regime",
            "liquidity_regime",
            "ofi_signed_future_mid_markout_bps",
            "decision_ts_ns",
            "decision_sequence",
            "observed_interval_id",
            "observed_interval_start_ns",
            "observed_interval_end_ns_exclusive",
            *state.feature_columns,
        ),
        "held-out L2 endpoint frame",
    )
    identities = value.frame.select(
        pl.col("symbol").n_unique().alias("symbols"),
        pl.col("endpoint_name").n_unique().alias("endpoints"),
        pl.col("study_date").n_unique().alias("dates"),
        pl.col("study_role").n_unique().alias("roles"),
    ).row(0, named=True)
    if any(int(identities[name]) != 1 for name in identities):
        raise L2LockedEvaluationError("one held-out frame must have one symbol/endpoint/date/role")
    observed = value.frame.select("symbol", "endpoint_name", "study_date", "study_role").row(
        0, named=True
    )
    expected = {
        "symbol": value.symbol,
        "endpoint_name": value.endpoint_name,
        "study_date": value.study_date,
        "study_role": value.study_role,
    }
    if observed != expected:
        raise L2LockedEvaluationError("held-out frame metadata differs from its typed coordinate")
    if value.symbol != state.symbol or value.endpoint_name != state.endpoint.name:
        raise L2LockedEvaluationError("held-out frame coordinate differs from its locked state")
    endpoint_identity = value.frame.select(
        "endpoint_domain", "endpoint_horizon_value", "endpoint_horizon_unit"
    ).unique()
    expected_endpoint_identity = {
        "endpoint_domain": state.endpoint.domain,
        "endpoint_horizon_value": state.endpoint.horizon_value,
        "endpoint_horizon_unit": state.endpoint.horizon_unit,
    }
    if endpoint_identity.height != 1 or endpoint_identity.row(0, named=True) != (
        expected_endpoint_identity
    ):
        raise L2LockedEvaluationError("held-out endpoint semantics differ from its locked state")
    if value.frame.filter(pl.col("continuity_id") != pl.col("observed_interval_id")).height:
        raise L2LockedEvaluationError(
            "L2 research continuity must equal the verified observed-interval identity"
        )
    regimes = set(str(item) for item in value.frame["joint_market_regime"].drop_nulls().unique())
    unknown = regimes.difference(_EXPECTED_REGIMES)
    if unknown:
        raise L2LockedEvaluationError(
            f"held-out frame has unknown train-defined regimes: {sorted(unknown)}"
        )
    block_group = ["study_date", "symbol", "continuity_id", "observed_interval_id"]
    with_event_ordinal = value.frame.sort(
        *block_group, "decision_ts_ns", "decision_sequence"
    ).with_columns(
        (pl.col("decision_sequence").cum_count().over(block_group) - 1)
        .cast(pl.Int64)
        .alias("_endpoint_event_ordinal")
    )
    eligible = with_event_ordinal.filter(
        pl.col("feature_ready") & (~pl.col("right_censored")) & pl.col(_TARGET).is_not_null()
    )
    if eligible.is_empty():
        raise L2LockedEvaluationError("held-out endpoint has no feature-ready labeled rows")
    targets = set(eligible[_TARGET].unique().to_list())
    if not targets.issubset({0, 1}):
        raise L2LockedEvaluationError("held-out binary target must contain only zero and one")
    first_decision = int(cast(int, eligible["decision_ts_ns"].min()))
    if (
        state.fit_cutoff("selected") >= first_decision
        or state.fit_cutoff("historical_prior") >= first_decision
    ):
        raise L2LockedEvaluationError("development fitting information reaches held-out decisions")
    return eligible.sort(
        "study_date",
        "symbol",
        "endpoint_name",
        "decision_ts_ns",
        "decision_sequence",
    )


def _predict_one(
    value: L2HeldoutEndpointFrame,
    state: LockedL2EndpointState,
) -> pl.DataFrame:
    eligible = _validate_frame_for_state(value, state)
    matrix = eligible.select(state.feature_columns).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise L2LockedEvaluationError("held-out fitted-state features must be finite")
    selected_raw, selected_probability = state.fitted_state.predict("selected", matrix)
    prior_raw, prior_probability = state.fitted_state.predict("historical_prior", matrix)
    for label, probability in (
        ("selected raw", selected_raw),
        ("selected calibrated", selected_probability),
        ("prior raw", prior_raw),
        ("prior calibrated", prior_probability),
    ):
        if probability.shape != (eligible.height,) or not np.isfinite(probability).all():
            raise L2LockedEvaluationError(f"{label} predictions are not finite row-aligned data")
        if bool(np.any((probability < 0.0) | (probability > 1.0))):
            raise L2LockedEvaluationError(f"{label} predictions escape the probability interval")
    selected_cutoff = state.fit_cutoff("selected")
    prior_cutoff = state.fit_cutoff("historical_prior")
    identity_columns = (
        "sample_id",
        "symbol",
        "study_date",
        "study_role",
        "endpoint_name",
        "endpoint_domain",
        "endpoint_horizon_value",
        "endpoint_horizon_unit",
        "continuity_id",
        "observed_interval_id",
        "observed_interval_start_ns",
        "observed_interval_end_ns_exclusive",
        "decision_ts_ns",
        "decision_sequence",
        "volatility_regime",
        "liquidity_regime",
        "joint_market_regime",
        "ofi_signed_future_mid_markout_bps",
        "_endpoint_event_ordinal",
    )
    return (
        eligible.select(*identity_columns, pl.col(_TARGET).cast(pl.Int8).alias("y_true"))
        .with_columns(
            pl.Series("selected_raw_probability", selected_raw),
            pl.Series("selected_probability", selected_probability),
            pl.Series("prior_raw_probability", prior_raw),
            pl.Series("prior_probability", prior_probability),
            pl.lit(state.selected_model).alias("selected_model"),
            pl.lit("historical_prior").alias("baseline_model"),
            pl.lit(selected_cutoff, dtype=pl.Int64).alias("selected_fit_cutoff_ts_ns"),
            pl.lit(prior_cutoff, dtype=pl.Int64).alias("prior_fit_cutoff_ts_ns"),
            pl.lit(state.child_lock_sha256).alias("child_lock_sha256"),
            pl.lit(state.aggregate_lock_sha256).alias("aggregate_lock_sha256"),
            pl.lit(state.regime_thresholds_sha256).alias("regime_thresholds_sha256"),
            pl.lit(state.fitted_state.sha256).alias("fitted_state_sha256"),
            pl.lit(state.endpoint.impact_ofi_window, dtype=pl.Int64).alias(
                "endpoint_impact_ofi_window"
            ),
            pl.lit(True).alias("is_oos"),
            pl.lit("final_test").alias("split"),
            pl.lit(False).alias("test_used_for_selection"),
            pl.lit(False).alias("model_updated_between_test_dates"),
            pl.lit(False).alias("p_value_computed"),
            pl.lit(False).alias("significance_claim_authorized"),
        )
        .sort("study_date", "symbol", "endpoint_name", "decision_ts_ns", "decision_sequence")
    )


def _metric_rows(predictions: pl.DataFrame, *, calibration_bins: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    partitions = predictions.partition_by(
        ["symbol", "study_date", "study_role", "endpoint_name"], maintain_order=True
    )
    for frame in partitions:
        y_true = frame["y_true"].to_numpy().astype(np.int64, copy=False)
        specifications = (
            (
                "selected",
                str(frame["selected_model"][0]),
                "selected_probability",
                int(frame["selected_fit_cutoff_ts_ns"][0]),
            ),
            (
                "historical_prior",
                "historical_prior",
                "prior_probability",
                int(frame["prior_fit_cutoff_ts_ns"][0]),
            ),
        )
        for role, model, probability_column, cutoff in specifications:
            probability = frame[probability_column].to_numpy().astype(np.float64, copy=False)
            rows.append(
                {
                    "symbol": str(frame["symbol"][0]),
                    "study_date": str(frame["study_date"][0]),
                    "study_role": str(frame["study_role"][0]),
                    "endpoint_name": str(frame["endpoint_name"][0]),
                    "endpoint_domain": str(frame["endpoint_domain"][0]),
                    "endpoint_horizon_value": int(frame["endpoint_horizon_value"][0]),
                    "endpoint_horizon_unit": str(frame["endpoint_horizon_unit"][0]),
                    "model_role": role,
                    "model": model,
                    "baseline": "historical_prior",
                    "n_obs": frame.height,
                    "period_start_ts_ns": int(cast(int, frame["decision_ts_ns"].min())),
                    "period_end_ts_ns": int(cast(int, frame["decision_ts_ns"].max())),
                    "fit_cutoff_ts_ns": cutoff,
                    "child_lock_sha256": str(frame["child_lock_sha256"][0]),
                    "aggregate_lock_sha256": str(frame["aggregate_lock_sha256"][0]),
                    "regime_thresholds_sha256": str(frame["regime_thresholds_sha256"][0]),
                    "fitted_state_sha256": str(frame["fitted_state_sha256"][0]),
                    "is_oos": True,
                    "test_used_for_selection": False,
                    "model_updated_between_test_dates": False,
                    "p_value": None,
                    "p_value_computed": False,
                    "significance_claim_authorized": False,
                    **classification_metrics(
                        y_true,
                        probability,
                        calibration_bins=calibration_bins,
                    ),
                }
            )
    return rows


def _loss(probability: NDArray[np.float64], target: NDArray[np.int64]) -> NDArray[np.float64]:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.asarray(
        -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)),
        dtype=np.float64,
    )


def _window_sufficient_statistics(
    coordinate: NDArray[np.int64],
    starts: NDArray[np.int64],
    width: int,
    selected_loss: NDArray[np.float64],
    prior_loss: NDArray[np.float64],
    included: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return O(n + candidates) window sums without materializing event rows."""

    if width < 1:
        raise L2LockedEvaluationError("moving-block width must be positive")
    selected_prefix = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(selected_loss * included))
    )
    prior_prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(prior_loss * included)))
    count_prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(included, dtype=np.int64))
    )
    left = np.searchsorted(coordinate, starts, side="left")
    right = np.searchsorted(coordinate, starts + width, side="left")
    return np.column_stack(
        (
            selected_prefix[right] - selected_prefix[left],
            prior_prefix[right] - prior_prefix[left],
            count_prefix[right] - count_prefix[left],
        )
    )


def _moving_block_intervals(
    frame: pl.DataFrame,
    endpoint: L2EndpointSpec,
    selected_loss: NDArray[np.float64],
    prior_loss: NDArray[np.float64],
    included: NDArray[np.bool_],
) -> tuple[tuple[_MovingBlockInterval, ...], int, bool]:
    """Build interval-local overlapping candidates from prefix-sum statistics.

    Event windows use the ordinal assigned before model-row/regime filtering,
    so sparse rows never compress a frozen dependency width.  Clock candidates
    start only at observed decisions and use half-open wall-time windows.  Empty
    regime candidates are discarded, but candidates are never pooled across
    verified observed intervals.
    """

    identity = ["study_date", "symbol", "continuity_id", "observed_interval_id"]
    ordered = frame.with_row_index("_moving_block_row").sort(
        *identity, "decision_ts_ns", "decision_sequence"
    )
    intervals: list[_MovingBlockInterval] = []
    candidate_count = 0
    unsupported_interval = False
    for current in ordered.partition_by(identity, maintain_order=True):
        indices = current["_moving_block_row"].to_numpy().astype(np.int64, copy=False)
        current_selected = selected_loss[indices]
        current_prior = prior_loss[indices]
        current_included = included[indices]
        if not bool(current_included.any()):
            continue
        if endpoint.domain == "event":
            width = endpoint.paired_block_events
            if width is None or width < 1:
                raise L2LockedEvaluationError("event endpoint has no frozen event block width")
            coordinate = current["_endpoint_event_ordinal"].to_numpy().astype(np.int64, copy=False)
            if bool(np.any(coordinate < 0)) or bool(np.any(np.diff(coordinate) <= 0)):
                raise L2LockedEvaluationError(
                    "event moving-block ordinals must be bounded and strictly increasing"
                )
            domain_start = int(coordinate[0])
            domain_end = int(coordinate[-1]) + 1
            domain_span = domain_end - domain_start
            if domain_span < width:
                unsupported_interval = True
                continue
            starts = np.arange(domain_start, domain_end - width + 1, dtype=np.int64)
            full = _window_sufficient_statistics(
                coordinate,
                starts,
                width,
                current_selected,
                current_prior,
                current_included,
            )
            nonempty = full[:, 2] > 0.0
            full = full[nonempty]
            starts = starts[nonempty]
            if full.shape[0] == 0:
                unsupported_interval = True
                continue
            blocks_per_draw = math.ceil(domain_span / width)
            remainder = domain_span % width
            tail = (
                _window_sufficient_statistics(
                    coordinate,
                    starts,
                    remainder,
                    current_selected,
                    current_prior,
                    current_included,
                )
                if remainder
                else None
            )
        else:
            width_ms = endpoint.paired_block_milliseconds
            if width_ms is None or width_ms < 1:
                raise L2LockedEvaluationError("clock endpoint has no frozen wall-time block width")
            width = width_ms * _NANOSECONDS_PER_MILLISECOND
            coordinate = current["decision_ts_ns"].to_numpy().astype(np.int64, copy=False)
            interval_start = int(current["observed_interval_start_ns"][0])
            interval_end = int(current["observed_interval_end_ns_exclusive"][0])
            if (
                interval_start < 0
                or interval_end <= interval_start
                or bool(np.any(coordinate < interval_start))
                or bool(np.any(coordinate >= interval_end))
                or bool(np.any(np.diff(coordinate) < 0))
            ):
                raise L2LockedEvaluationError("clock moving-block coordinates are invalid")
            interval_span = interval_end - interval_start
            legal = coordinate <= interval_end - width
            starts = coordinate[legal]
            if interval_span < width or starts.size == 0:
                unsupported_interval = True
                continue
            full = _window_sufficient_statistics(
                coordinate,
                starts,
                width,
                current_selected,
                current_prior,
                current_included,
            )
            full = full[full[:, 2] > 0.0]
            if full.shape[0] == 0:
                unsupported_interval = True
                continue
            blocks_per_draw = math.ceil(interval_span / width)
            tail = None
        candidate_count += int(full.shape[0])
        intervals.append(
            _MovingBlockInterval(
                full=np.asarray(full, dtype=np.float64),
                tail=(np.asarray(tail, dtype=np.float64) if tail is not None else None),
                blocks_per_draw=blocks_per_draw,
            )
        )
    return tuple(intervals), candidate_count, unsupported_interval


def _paired_delta(
    frame: pl.DataFrame,
    endpoint: L2EndpointSpec,
    *,
    regime: str,
    samples: int,
    seed: int,
) -> _PairedDelta:
    included = (
        np.ones(frame.height, dtype=np.bool_)
        if regime == _ALL_REGIME
        else frame["joint_market_regime"].to_numpy() == regime
    )
    n_obs = int(np.count_nonzero(included))
    if n_obs == 0:
        return _PairedDelta(
            None, None, None, None, None, 0, 0, samples, seed, "empty_regime", np.empty(0)
        )
    target = frame["y_true"].to_numpy().astype(np.int64, copy=False)
    selected_loss = _loss(
        frame["selected_probability"].to_numpy().astype(np.float64, copy=False), target
    )
    prior_loss = _loss(frame["prior_probability"].to_numpy().astype(np.float64, copy=False), target)
    selected_point = float(selected_loss[included].mean())
    prior_point = float(prior_loss[included].mean())
    point = selected_point - prior_point
    intervals, n_blocks, unsupported = _moving_block_intervals(
        frame,
        endpoint,
        selected_loss,
        prior_loss,
        included,
    )
    has_sampling_choice = any(value.full.shape[0] > 1 for value in intervals)
    if unsupported or not intervals or not has_sampling_choice:
        return _PairedDelta(
            selected_point,
            prior_point,
            point,
            None,
            None,
            n_obs,
            n_blocks,
            samples,
            seed,
            "insufficient_blocks",
            np.empty(0),
        )
    random = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    # Resample bounded block sufficient statistics, never per-draw event rows.
    # Each interval has an independent draw and is therefore never pooled with
    # another continuity/OBSERVED interval.
    for draw_index in range(samples):
        totals = np.zeros(3, dtype=np.float64)
        for interval in intervals:
            sampled = random.integers(
                0,
                interval.full.shape[0],
                size=interval.blocks_per_draw,
            )
            if interval.tail is None:
                totals += interval.full[sampled].sum(axis=0)
            else:
                if sampled.size > 1:
                    totals += interval.full[sampled[:-1]].sum(axis=0)
                totals += interval.tail[sampled[-1]]
        if totals[2] <= 0.0:
            raise L2LockedEvaluationError("moving-block draw has no regime observations")
        draws[draw_index] = totals[0] / totals[2] - totals[1] / totals[2]
    return _PairedDelta(
        selected_point,
        prior_point,
        point,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        n_obs,
        n_blocks,
        samples,
        seed,
        "ok",
        draws,
    )


def _diagnostics(
    predictions: pl.DataFrame,
    states: Mapping[tuple[str, str], LockedL2EndpointState],
    *,
    samples: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    per_session_rows: list[dict[str, object]] = []
    markout_rows: list[dict[str, object]] = []
    draws_by_key: dict[tuple[str, str, str, str], _PairedDelta] = {}
    partitions = predictions.partition_by(
        ["symbol", "study_date", "study_role", "endpoint_name"], maintain_order=True
    )
    for raw in partitions:
        symbol = str(raw["symbol"][0])
        study_date = str(raw["study_date"][0])
        study_role = str(raw["study_role"][0])
        endpoint_name = str(raw["endpoint_name"][0])
        state = states[(symbol, endpoint_name)]
        blocked = raw.sort(
            "study_date",
            "symbol",
            "continuity_id",
            "observed_interval_id",
            "decision_ts_ns",
            "decision_sequence",
        )
        for regime in (_ALL_REGIME, *_EXPECTED_REGIMES):
            current = (
                blocked
                if regime == _ALL_REGIME
                else blocked.filter(pl.col("joint_market_regime") == regime)
            )
            row_seed = _stable_seed(seed, symbol, study_date, endpoint_name, regime)
            paired = _paired_delta(
                blocked,
                state.endpoint,
                regime=regime,
                samples=samples,
                seed=row_seed,
            )
            draws_by_key[(symbol, endpoint_name, regime, study_role)] = paired
            block_width: int
            block_unit: str
            if state.endpoint.domain == "event":
                block_width = cast(int, state.endpoint.paired_block_events)
                block_unit = "events"
            else:
                block_width = cast(int, state.endpoint.paired_block_milliseconds)
                block_unit = "milliseconds"
            per_session_rows.append(
                {
                    "symbol": symbol,
                    "study_date": study_date,
                    "study_role": study_role,
                    "endpoint_name": endpoint_name,
                    "endpoint_domain": state.endpoint.domain,
                    "regime": regime,
                    "regime_scope": "overall" if regime == _ALL_REGIME else "train_defined_joint",
                    "selected_model": state.selected_model,
                    "baseline": "historical_prior",
                    "metric": "log_loss",
                    "delta_definition": "selected_minus_historical_prior",
                    "selected_log_loss": paired.selected_log_loss,
                    "prior_log_loss": paired.prior_log_loss,
                    "point_delta": paired.point_delta,
                    "ci_low": paired.ci_low,
                    "ci_high": paired.ci_high,
                    "n_obs": paired.n_obs,
                    "n_blocks": paired.n_blocks,
                    "samples": paired.samples,
                    "seed": paired.seed,
                    "bootstrap_status": paired.status,
                    "block_width": block_width,
                    "block_unit": block_unit,
                    "date_weight": 0.5,
                    "point_favorable": (
                        paired.point_delta < 0.0 if paired.point_delta is not None else False
                    ),
                    "child_lock_sha256": state.child_lock_sha256,
                    "aggregate_lock_sha256": state.aggregate_lock_sha256,
                    "regime_thresholds_sha256": state.regime_thresholds_sha256,
                    "p_value": None,
                    "p_value_computed": False,
                    "h0_rejected": False,
                    "significance_claim_authorized": False,
                    "cross_symbol_pooling": False,
                }
            )
            markout = current["ofi_signed_future_mid_markout_bps"].drop_nulls()
            markout_rows.append(
                {
                    "symbol": symbol,
                    "study_date": study_date,
                    "study_role": study_role,
                    "endpoint_name": endpoint_name,
                    "regime": regime,
                    "n_obs": len(markout),
                    "mean_ofi_signed_future_mid_markout_bps": (
                        float(cast(Any, markout.mean())) if len(markout) else None
                    ),
                    "median_ofi_signed_future_mid_markout_bps": (
                        float(cast(Any, markout.median())) if len(markout) else None
                    ),
                    "positive_fraction": (
                        float(cast(Any, (markout > 0.0).mean())) if len(markout) else None
                    ),
                    "metric": "ofi_signed_future_mid_markout",
                    "descriptive_only": True,
                    "observed_trade_impact": False,
                    "child_lock_sha256": state.child_lock_sha256,
                    "aggregate_lock_sha256": state.aggregate_lock_sha256,
                    "regime_thresholds_sha256": state.regime_thresholds_sha256,
                    "p_value_computed": False,
                    "significance_claim_authorized": False,
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for (symbol, endpoint_name), state in sorted(states.items()):
        for regime in (_ALL_REGIME, *_EXPECTED_REGIMES):
            primary = draws_by_key.get((symbol, endpoint_name, regime, "primary_test"))
            replication = draws_by_key.get((symbol, endpoint_name, regime, "replication_test"))
            if primary is None or replication is None:
                raise L2LockedEvaluationError("paired diagnostics lack a declared held-out role")
            points = (primary.point_delta, replication.point_delta)
            complete_points = all(value is not None for value in points)
            point = (
                0.5 * cast(float, points[0]) + 0.5 * cast(float, points[1])
                if complete_points
                else None
            )
            bootstrap_ok = primary.status == "ok" and replication.status == "ok"
            if bootstrap_ok:
                aggregate_draws = 0.5 * primary.draws + 0.5 * replication.draws
                ci_low = float(np.quantile(aggregate_draws, 0.025))
                ci_high = float(np.quantile(aggregate_draws, 0.975))
                status = "ok"
            else:
                ci_low = None
                ci_high = None
                status = (
                    "empty_regime"
                    if primary.status == "empty_regime" or replication.status == "empty_regime"
                    else "insufficient_blocks"
                )
            primary_favorable = primary.point_delta is not None and primary.point_delta < 0.0
            replication_favorable = (
                replication.point_delta is not None and replication.point_delta < 0.0
            )
            replicated = primary_favorable and replication_favorable
            if not complete_points:
                replication_status = "insufficient_data"
            elif replicated:
                replication_status = "replicated"
            elif not primary_favorable:
                replication_status = "no_primary_improvement"
            else:
                replication_status = "failed_replication"
            aggregate_rows.append(
                {
                    "symbol": symbol,
                    "endpoint_name": endpoint_name,
                    "endpoint_domain": state.endpoint.domain,
                    "regime": regime,
                    "regime_scope": "overall" if regime == _ALL_REGIME else "train_defined_joint",
                    "selected_model": state.selected_model,
                    "baseline": "historical_prior",
                    "metric": "log_loss",
                    "delta_definition": "selected_minus_historical_prior",
                    "date_weighting": "equal_primary_replication",
                    "primary_weight": 0.5,
                    "replication_weight": 0.5,
                    "primary_point_delta": primary.point_delta,
                    "replication_point_delta": replication.point_delta,
                    "point_delta": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_obs": primary.n_obs + replication.n_obs,
                    "n_sessions": 2,
                    "n_blocks": primary.n_blocks + replication.n_blocks,
                    "samples": samples,
                    "bootstrap_status": status,
                    "directionally_replicated": replicated,
                    "replication_status": replication_status,
                    "child_lock_sha256": state.child_lock_sha256,
                    "aggregate_lock_sha256": state.aggregate_lock_sha256,
                    "regime_thresholds_sha256": state.regime_thresholds_sha256,
                    "p_value": None,
                    "p_value_computed": False,
                    "h0_rejected": False,
                    "significance_claim_authorized": False,
                    "cross_symbol_pooling": False,
                }
            )
    return (
        pl.DataFrame(per_session_rows, infer_schema_length=None).sort(
            "symbol", "endpoint_name", "regime", "study_date"
        ),
        pl.DataFrame(aggregate_rows, infer_schema_length=None).sort(
            "symbol", "endpoint_name", "regime"
        ),
        pl.DataFrame(markout_rows, infer_schema_length=None).sort(
            "symbol", "endpoint_name", "regime", "study_date"
        ),
    )


def evaluate_locked_l2_endpoints(
    locked_states: Sequence[LockedL2EndpointState],
    heldout_frames: Sequence[L2HeldoutEndpointFrame],
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 20_260_807,
    calibration_bins: int = 10,
) -> L2EvaluationResult:
    """Evaluate all declared held-out frames without exposing any fitting path."""

    if bootstrap_samples != 2_000:
        raise L2LockedEvaluationError("M8 L2 bootstrap samples are frozen at 2000")
    if calibration_bins != 10:
        raise L2LockedEvaluationError("M8 L2 calibration bins are frozen at 10")
    states = tuple(locked_states)
    frames = tuple(heldout_frames)
    if not states or not frames:
        raise L2LockedEvaluationError("locked states and held-out frames must be nonempty")
    by_state: dict[tuple[str, str], LockedL2EndpointState] = {}
    for state in states:
        key = _state_key(state)
        if key in by_state:
            raise L2LockedEvaluationError(f"duplicate locked L2 endpoint state: {key}")
        by_state[key] = state
    aggregate_hashes = {state.aggregate_lock_sha256 for state in states}
    if len(aggregate_hashes) != 1:
        raise L2LockedEvaluationError("all endpoint states must share one aggregate lock")
    by_frame: dict[tuple[str, str, HeldoutRole], L2HeldoutEndpointFrame] = {}
    for frame in frames:
        frame_key = _frame_key(frame)
        if frame_key in by_frame:
            raise L2LockedEvaluationError(f"duplicate held-out L2 endpoint frame: {frame_key}")
        by_frame[frame_key] = frame
    expected = {
        (symbol, endpoint, cast(HeldoutRole, role))
        for symbol, endpoint in by_state
        for role in ("primary_test", "replication_test")
    }
    if set(by_frame) != expected:
        raise L2LockedEvaluationError(
            "held-out frames differ from the exact primary/replication locked endpoint set"
        )
    predictions = pl.concat(
        [
            _predict_one(by_frame[(symbol, endpoint, role)], state)
            for (symbol, endpoint), state in sorted(by_state.items())
            for role in cast(tuple[HeldoutRole, ...], ("primary_test", "replication_test"))
        ],
        how="vertical_relaxed",
    ).sort("symbol", "endpoint_name", "study_date", "decision_ts_ns", "decision_sequence")
    if predictions["sample_id"].n_unique() != predictions.height:
        raise L2LockedEvaluationError("held-out sample identities collide across endpoint frames")
    metrics = pl.DataFrame(
        _metric_rows(predictions, calibration_bins=calibration_bins), infer_schema_length=None
    ).sort("symbol", "endpoint_name", "study_date", "model_role")
    paired, aggregate, markout = _diagnostics(
        predictions,
        by_state,
        samples=bootstrap_samples,
        seed=seed,
    )
    return L2EvaluationResult(
        predictions=_null_nonfinite(predictions),
        predictive_metrics=_null_nonfinite(metrics),
        paired_by_session_regime=_null_nonfinite(paired),
        equal_session_summary=_null_nonfinite(aggregate),
        signed_markout=_null_nonfinite(markout),
    )


def _metadata_columns(
    frame: pl.DataFrame,
    *,
    scenario_id: str,
    symbol: str,
    study_date: str,
    study_role: str,
    endpoint_name: str,
    decision_latency: int,
    order_latency: int,
    reference: L2ExecutionReference,
    child_lock_sha256: str,
) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(scenario_id).alias("scenario_id"),
        pl.lit(symbol).alias("scenario_symbol"),
        pl.lit(study_date).alias("study_date"),
        pl.lit(study_role).alias("study_role"),
        pl.lit(endpoint_name).alias("endpoint_name"),
        pl.lit(decision_latency, dtype=pl.Int64).alias("decision_latency_events"),
        pl.lit(order_latency, dtype=pl.Int64).alias("order_latency_events"),
        pl.lit(reference.reference_quantity).alias("reference_quantity"),
        pl.lit(reference.training_date).alias("execution_reference_training_date"),
        pl.lit(reference.reference_price_statistic).alias("reference_price_statistic"),
        pl.lit(reference.reference_depth_statistic).alias("reference_depth_statistic"),
        pl.lit(reference.analysis_config_source_sha256).alias("analysis_config_source_sha256"),
        pl.lit(reference.analysis_config_semantic_sha256).alias("analysis_config_semantic_sha256"),
        pl.lit(reference.reference_sha256).alias("execution_reference_sha256"),
        pl.lit(child_lock_sha256).alias("child_lock_sha256"),
        pl.lit(reference.aggregate_lock_sha256).alias("aggregate_lock_sha256"),
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def run_locked_l2_market_execution(
    evaluation: L2EvaluationResult,
    heldout_frames: Sequence[L2HeldoutEndpointFrame],
    references: Sequence[L2ExecutionReference],
    *,
    decision_latency_events: Sequence[int] = _EXPECTED_EVENT_LATENCIES,
    order_latency_events: Sequence[int] = _EXPECTED_EVENT_LATENCIES,
    probability_threshold: float = 0.55,
    taker_fee_bps: float = 4.0,
    inventory_order_multiples: int = 10,
) -> L2MarketExecutionResult:
    """Replay selected OOS predictions over the frozen market-only event grid."""

    decisions = tuple(decision_latency_events)
    orders = tuple(order_latency_events)
    if decisions != _EXPECTED_EVENT_LATENCIES or orders != _EXPECTED_EVENT_LATENCIES:
        raise L2LockedEvaluationError("M8 L2 latency grids are frozen at event counts 0, 1, 5")
    if probability_threshold != 0.55 or taker_fee_bps != 4.0:
        raise L2LockedEvaluationError("M8 L2 probability threshold and taker fee are frozen")
    if inventory_order_multiples != 10:
        raise L2LockedEvaluationError("M8 L2 inventory bound is frozen at ten order multiples")
    _require(
        evaluation.predictions,
        (
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
        ),
        "locked L2 predictions",
    )
    if not bool(evaluation.predictions["is_oos"].all()) or set(
        evaluation.predictions["split"].unique()
    ) != {"final_test"}:
        raise L2LockedEvaluationError("market scenarios require only explicit held-out OOS rows")
    frame_map = {_frame_key(value): value for value in heldout_frames}
    if len(frame_map) != len(tuple(heldout_frames)):
        raise L2LockedEvaluationError("execution frames contain duplicate coordinates")
    reference_map: dict[str, L2ExecutionReference] = {}
    for reference in references:
        if reference.symbol in reference_map:
            raise L2LockedEvaluationError("execution references contain duplicate symbols")
        reference_map[reference.symbol] = reference
    prediction_keys = {
        (str(row["symbol"]), str(row["endpoint_name"]), str(row["study_role"]))
        for row in evaluation.predictions.select("symbol", "endpoint_name", "study_role")
        .unique()
        .to_dicts()
    }
    if set(frame_map) != prediction_keys:
        raise L2LockedEvaluationError("execution frames differ from evaluated held-out coordinates")
    if set(reference_map) != {symbol for symbol, _, _ in prediction_keys}:
        raise L2LockedEvaluationError("execution references differ from evaluated symbols")

    order_frames: list[pl.DataFrame] = []
    fill_frames: list[pl.DataFrame] = []
    position_frames: list[pl.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    assumption_rows: list[dict[str, object]] = []
    for symbol, endpoint_name, role in sorted(prediction_keys):
        coordinate = cast(tuple[str, str, HeldoutRole], (symbol, endpoint_name, role))
        heldout = frame_map[coordinate]
        reference = reference_map[symbol]
        if reference.training_date >= heldout.study_date:
            raise L2LockedEvaluationError(
                "execution reference training date must precede every held-out session"
            )
        current_predictions = evaluation.predictions.filter(
            (pl.col("symbol") == symbol)
            & (pl.col("endpoint_name") == endpoint_name)
            & (pl.col("study_role") == role)
        )
        try:
            validate_l2_endpoint_frame(heldout.frame)
        except L2ResearchError as error:
            raise L2LockedEvaluationError(str(error)) from error
        frame_coordinate = heldout.frame.select(
            "symbol", "study_date", "study_role", "endpoint_name"
        ).unique()
        expected_coordinate = {
            "symbol": symbol,
            "study_date": heldout.study_date,
            "study_role": role,
            "endpoint_name": endpoint_name,
        }
        if frame_coordinate.height != 1 or frame_coordinate.row(0, named=True) != (
            expected_coordinate
        ):
            raise L2LockedEvaluationError(
                "execution event frame differs from its evaluated coordinate"
            )
        if heldout.frame.filter(pl.col("continuity_id") != pl.col("observed_interval_id")).height:
            raise L2LockedEvaluationError(
                "execution continuity must equal the verified observed interval"
            )
        evaluated_identity = current_predictions.select("sample_id", "decision_sequence").sort(
            "decision_sequence"
        )
        replay_identity = (
            heldout.frame.join(evaluated_identity.select("sample_id"), on="sample_id", how="inner")
            .select("sample_id", "decision_sequence")
            .sort("decision_sequence")
        )
        if not replay_identity.equals(evaluated_identity):
            raise L2LockedEvaluationError(
                "execution event frame is not row-identical to evaluated predictions"
            )
        aggregate_hashes = set(
            str(value) for value in current_predictions["aggregate_lock_sha256"].unique()
        )
        child_hashes = set(
            str(value) for value in current_predictions["child_lock_sha256"].unique()
        )
        if aggregate_hashes != {reference.aggregate_lock_sha256} or len(child_hashes) != 1:
            raise L2LockedEvaluationError("execution reference and prediction locks disagree")
        _require(
            heldout.frame,
            (
                "decision_ts_ns",
                "decision_sequence",
                "continuity_id",
                "best_bid",
                "best_ask",
                "bid_quantity",
                "ask_quantity",
                "mid_price",
                "tick_size",
                "lot_size",
            ),
            "L2 execution event frame",
        )
        observed_lots = heldout.frame["lot_size"].drop_nulls().unique().to_list()
        if len(observed_lots) != 1 or not math.isclose(
            float(observed_lots[0]), reference.lot_size, rel_tol=0.0, abs_tol=1e-15
        ):
            raise L2LockedEvaluationError("execution reference lot size differs from held-out data")
        event_columns = [name for name in heldout.frame.columns if name != "event_ts_ns"]
        events = (
            heldout.frame.select(*event_columns)
            .with_columns(pl.col("decision_ts_ns").alias("event_ts_ns"))
            .sort("decision_ts_ns", "decision_sequence")
        )
        simulation_predictions = current_predictions.select(
            "symbol",
            "decision_sequence",
            pl.col("selected_probability").alias("probability"),
            "is_oos",
            "split",
        )
        for decision_latency in decisions:
            for order_latency in orders:
                scenario_id = (
                    f"{symbol}::{heldout.study_date}::{endpoint_name}::"
                    f"d{decision_latency}::o{order_latency}"
                )
                execution_config = ExecutionConfig(
                    decision_latency_events=decision_latency,
                    order_latency_events=order_latency,
                    maker_fee_bps=0.0,
                    taker_fee_bps=taker_fee_bps,
                    half_spread_bps=0.0,
                    slippage_bps_per_unit=0.0,
                    signal_threshold=probability_threshold,
                    max_position_units=(reference.reference_quantity * inventory_order_multiples),
                    order_size_units=reference.reference_quantity,
                    limit_fill_base_probability=0.0,
                    queue_ahead_units=0.0,
                    limit_max_age_events=1,
                    cancel_latency_events=0,
                    liquidate_at_end=True,
                    capacity_multipliers=(1.0,),
                )
                result = simulate_predictions(
                    events,
                    simulation_predictions,
                    execution_config,
                    order_type="market",
                    size_multiplier=1.0,
                    seed=_stable_seed(20_260_807, scenario_id),
                    markout_events=int(current_predictions["endpoint_impact_ofi_window"][0]),
                )
                child_hash = next(iter(child_hashes))
                scenario_orders = _metadata_columns(
                    result.orders,
                    scenario_id=scenario_id,
                    symbol=symbol,
                    study_date=heldout.study_date,
                    study_role=role,
                    endpoint_name=endpoint_name,
                    decision_latency=decision_latency,
                    order_latency=order_latency,
                    reference=reference,
                    child_lock_sha256=child_hash,
                )
                if "order_type" in scenario_orders.columns and set(
                    scenario_orders["order_type"].drop_nulls().unique()
                ).difference({"market"}):
                    raise L2LockedEvaluationError("market-only replay emitted a non-market order")
                scenario_fills = _metadata_columns(
                    result.fills,
                    scenario_id=scenario_id,
                    symbol=symbol,
                    study_date=heldout.study_date,
                    study_role=role,
                    endpoint_name=endpoint_name,
                    decision_latency=decision_latency,
                    order_latency=order_latency,
                    reference=reference,
                    child_lock_sha256=child_hash,
                )
                if "liquidity" in scenario_fills.columns and set(
                    scenario_fills["liquidity"].drop_nulls().unique()
                ).difference({"taker"}):
                    raise L2LockedEvaluationError("market-only replay emitted a maker fill")
                scenario_positions = _metadata_columns(
                    result.positions,
                    scenario_id=scenario_id,
                    symbol=symbol,
                    study_date=heldout.study_date,
                    study_role=role,
                    endpoint_name=endpoint_name,
                    decision_latency=decision_latency,
                    order_latency=order_latency,
                    reference=reference,
                    child_lock_sha256=child_hash,
                )
                order_frames.append(scenario_orders)
                fill_frames.append(scenario_fills)
                position_frames.append(scenario_positions)
                metric_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "symbol": symbol,
                        "study_date": heldout.study_date,
                        "study_role": role,
                        "endpoint_name": endpoint_name,
                        "decision_latency_events": decision_latency,
                        "order_latency_events": order_latency,
                        "order_type": "market",
                        "strategy_orders": result.metrics["strategy_orders"],
                        "strategy_fills": result.metrics["strategy_fills"],
                        "forced_liquidation_fills": result.metrics["forced_liquidation_fills"],
                        "requested_quantity": result.metrics["requested_quantity"],
                        "accepted_quantity": result.metrics["accepted_quantity"],
                        "filled_quantity": result.metrics["filled_quantity"],
                        "fill_ratio": result.metrics["fill_ratio"],
                        "fill_ratio_requested": result.metrics["fill_ratio_requested"],
                        "partial_fill_order_ratio": result.metrics["partial_fill_order_ratio"],
                        "gross_pnl": result.metrics["gross_pnl"],
                        "total_fees": result.metrics["total_fees"],
                        "net_pnl": result.metrics["net_pnl"],
                        "turnover_notional": result.metrics["turnover_notional"],
                        "maximum_drawdown": result.metrics["maximum_drawdown"],
                        "maximum_absolute_inventory": result.metrics["maximum_absolute_inventory"],
                        "forced_liquidation_quantity": result.metrics[
                            "forced_liquidation_quantity"
                        ],
                        "unliquidated_quantity": result.metrics["unliquidated_quantity"],
                        "mean_arrival_cost_bps": result.metrics["mean_arrival_cost_bps"],
                        "mean_post_fill_markout_bps": result.metrics["mean_post_fill_markout_bps"],
                        "reference_quantity": reference.reference_quantity,
                        "execution_reference_training_date": reference.training_date,
                        "reference_price_statistic": reference.reference_price_statistic,
                        "reference_depth_statistic": reference.reference_depth_statistic,
                        "analysis_config_source_sha256": (reference.analysis_config_source_sha256),
                        "analysis_config_semantic_sha256": (
                            reference.analysis_config_semantic_sha256
                        ),
                        "inventory_limit_units": (
                            reference.reference_quantity * inventory_order_multiples
                        ),
                        "execution_reference_sha256": reference.reference_sha256,
                        "child_lock_sha256": child_hash,
                        "aggregate_lock_sha256": reference.aggregate_lock_sha256,
                        "scenario_only": True,
                        "capacity_claim_authorized": False,
                        "realized_execution_claim_authorized": False,
                        "profitability_claim_authorized": False,
                    }
                )
                assumption_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "symbol": symbol,
                        "study_date": heldout.study_date,
                        "study_role": role,
                        "endpoint_name": endpoint_name,
                        "market_orders_only": True,
                        "probability_buy_threshold": probability_threshold,
                        "probability_sell_threshold": 1.0 - probability_threshold,
                        "decision_latency_events": decision_latency,
                        "order_latency_events": order_latency,
                        "taker_fee_bps": taker_fee_bps,
                        "reference_quantity": reference.reference_quantity,
                        "reference_mid_price": reference.reference_mid_price,
                        "train_l1_depth_q05": reference.train_l1_depth_q05,
                        "execution_reference_training_date": reference.training_date,
                        "reference_price_statistic": reference.reference_price_statistic,
                        "reference_depth_statistic": reference.reference_depth_statistic,
                        "analysis_config_source_sha256": (reference.analysis_config_source_sha256),
                        "analysis_config_semantic_sha256": (
                            reference.analysis_config_semantic_sha256
                        ),
                        "max_l1_participation": 0.10,
                        "inventory_order_multiples": inventory_order_multiples,
                        "extra_slippage_bps": 0.0,
                        "l1_fill_policy": "fill_up_to_recorded_l1_depth_cancel_remainder",
                        "scenario_reset_policy": ("per_symbol_session_endpoint_latency_pair"),
                        "end_liquidation": True,
                        "replay_is_exogenous": True,
                        "live_trading": False,
                        "limit_fill_model": "NOT_RUN",
                        "capacity_sensitivity": "NOT_RUN",
                        "execution_reference_sha256": reference.reference_sha256,
                        "child_lock_sha256": child_hash,
                        "aggregate_lock_sha256": reference.aggregate_lock_sha256,
                        "simulator_assumptions_json": _json(result.assumptions),
                        "capacity_claim_authorized": False,
                        "realized_execution_claim_authorized": False,
                        "profitability_claim_authorized": False,
                    }
                )

    concatenated_orders = (
        pl.concat(order_frames, how="diagonal_relaxed") if order_frames else pl.DataFrame()
    )
    concatenated_fills = (
        pl.concat(fill_frames, how="diagonal_relaxed") if fill_frames else pl.DataFrame()
    )
    concatenated_positions = (
        pl.concat(position_frames, how="diagonal_relaxed") if position_frames else pl.DataFrame()
    )
    return L2MarketExecutionResult(
        orders=_null_nonfinite(concatenated_orders),
        fills=_null_nonfinite(concatenated_fills),
        positions=_null_nonfinite(concatenated_positions),
        metrics=_null_nonfinite(
            pl.DataFrame(metric_rows, infer_schema_length=None).sort("scenario_id")
        ),
        assumptions=_null_nonfinite(
            pl.DataFrame(assumption_rows, infer_schema_length=None).sort("scenario_id")
        ),
    )


__all__ = [
    "L2EvaluationResult",
    "L2ExecutionReference",
    "L2HeldoutEndpointFrame",
    "L2LockedEvaluationError",
    "L2MarketExecutionResult",
    "LockedL2EndpointState",
    "evaluate_locked_l2_endpoints",
    "run_locked_l2_market_execution",
]
