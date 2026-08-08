"""Outcome-blind causal frames for the prospective four-session L2 study.

The capture continuity identifier is not sufficient for research: a period of
silence may split one capture epoch into several intervals that are actually
backed by continuously observed book states.  This module therefore resegments
both books and deltas by the verified OBSERVED intervals before any rolling
feature or future label is calculated.

Clock labels are exact-horizon labels.  They use the last state observable at
``t+h`` (never the first state after it), require the target to remain inside
the same verified interval, and censor stale carried-forward states.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

import polars as pl

from microstructure.config import FeatureConfig
from microstructure.research.analysis import RegimeThresholds, assign_market_regimes
from microstructure.research.features import add_future_event_labels, build_research_features

EndpointDomain = Literal["event", "clock"]
StudyRole = Literal["train", "validation", "primary_test", "replication_test"]

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_NANOSECONDS_PER_DAY = 86_400_000_000_000
_ALLOWED_ROLES = frozenset({"train", "validation", "primary_test", "replication_test"})
_REGIME_FEATURES = (
    "volatility_regime_low",
    "volatility_regime_high",
    "liquidity_regime_liquid",
    "liquidity_regime_stressed",
)


class L2ResearchError(ValueError):
    """Raised when a live-L2 input would violate the frozen research contract."""


@dataclass(frozen=True, slots=True)
class L2ObservedInterval:
    """One capture-verified interval of continuously OBSERVED book state."""

    continuity_id: str
    start_received_ns: int
    end_received_ns_exclusive: int

    def __post_init__(self) -> None:
        if not self.continuity_id:
            raise ValueError("L2 interval continuity_id must not be empty")
        if self.start_received_ns < 0 or self.end_received_ns_exclusive <= self.start_received_ns:
            raise ValueError("L2 interval bounds are invalid")


@dataclass(frozen=True, slots=True)
class L2EndpointSpec:
    """One independently selected prediction endpoint."""

    name: str
    domain: EndpointDomain
    horizon_value: int
    horizon_unit: Literal["events", "milliseconds"]
    paired_block_events: int | None
    paired_block_milliseconds: int | None
    impact_ofi_window: int

    def __post_init__(self) -> None:
        if not self.name or self.domain not in {"event", "clock"}:
            raise ValueError("L2 endpoint name/domain is invalid")
        if self.horizon_value < 1 or self.impact_ofi_window < 1:
            raise ValueError("L2 endpoint horizons and OFI window must be positive")
        if self.domain == "event":
            if (
                self.horizon_unit != "events"
                or self.paired_block_events is None
                or self.paired_block_events < 1
                or self.paired_block_milliseconds is not None
            ):
                raise ValueError("event endpoint requires only a positive event block")
        elif (
            self.horizon_unit != "milliseconds"
            or self.paired_block_milliseconds is None
            or self.paired_block_milliseconds < 1
            or self.paired_block_events is not None
        ):
            raise ValueError("clock endpoint requires only a positive wall-time block")

    @property
    def horizon_ns(self) -> int:
        if self.domain != "clock":
            raise ValueError("event endpoints do not have a clock horizon")
        return self.horizon_value * _NANOSECONDS_PER_MILLISECOND


@dataclass(frozen=True, slots=True)
class L2RegimeFit:
    """Train-only regime thresholds plus their explicit fit contract."""

    symbol: str
    study_date: str
    volatility_column: str
    lower_quantile: float
    upper_quantile: float
    thresholds: RegimeThresholds

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "study_date": self.study_date,
            "volatility_column": self.volatility_column,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "thresholds": asdict(self.thresholds),
            "fit_scope": "train_session_only",
        }


def _require(frame: pl.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise L2ResearchError(f"{label} is missing required columns: {missing}")
    if frame.is_empty():
        raise L2ResearchError(f"{label} must not be empty")


def _date_bounds(study_date: str) -> tuple[int, int]:
    try:
        start = datetime.fromisoformat(f"{study_date}T00:00:00+00:00")
    except ValueError as error:
        raise L2ResearchError("study_date must use ISO YYYY-MM-DD") from error
    start_ns = int(start.timestamp()) * 1_000_000_000
    return start_ns, start_ns + _NANOSECONDS_PER_DAY


def _validate_intervals(
    intervals: Sequence[L2ObservedInterval],
    *,
    study_date: str,
) -> tuple[L2ObservedInterval, ...]:
    values = tuple(
        sorted(
            intervals,
            key=lambda item: (
                item.start_received_ns,
                item.end_received_ns_exclusive,
                item.continuity_id,
            ),
        )
    )
    if not values:
        raise L2ResearchError("L2 research requires at least one verified OBSERVED interval")
    date_start, date_end = _date_bounds(study_date)
    prior_end = -1
    for item in values:
        if item.start_received_ns < date_start or item.end_received_ns_exclusive > date_end:
            raise L2ResearchError("OBSERVED interval escapes its frozen study date")
        if item.start_received_ns < prior_end:
            raise L2ResearchError("OBSERVED intervals overlap")
        prior_end = item.end_received_ns_exclusive
    return values


def _segment_one(
    frame: pl.DataFrame,
    intervals: Sequence[L2ObservedInterval],
    *,
    study_date: str,
    time_column: str,
    label: str,
) -> pl.DataFrame:
    _require(frame, ("symbol", "continuity_id", time_column), label)
    outputs: list[pl.DataFrame] = []
    for ordinal, interval in enumerate(intervals):
        research_id = f"{study_date}::{interval.continuity_id}::observed-{ordinal:04d}"
        subset = frame.filter(
            (pl.col("continuity_id") == interval.continuity_id)
            & (pl.col(time_column) >= interval.start_received_ns)
            & (pl.col(time_column) < interval.end_received_ns_exclusive)
        )
        if subset.is_empty():
            raise L2ResearchError(f"{label} has no rows for verified interval {research_id}")
        outputs.append(
            subset.rename({"continuity_id": "capture_continuity_id"}).with_columns(
                pl.lit(research_id).alias("continuity_id"),
                pl.lit(research_id).alias("observed_interval_id"),
                pl.lit(interval.start_received_ns, dtype=pl.Int64).alias(
                    "observed_interval_start_ns"
                ),
                pl.lit(interval.end_received_ns_exclusive, dtype=pl.Int64).alias(
                    "observed_interval_end_ns_exclusive"
                ),
            )
        )
    result = pl.concat(outputs, how="vertical_relaxed")
    if (
        result.height
        != frame.filter(
            pl.any_horizontal(
                [
                    (pl.col("continuity_id") == item.continuity_id)
                    & (pl.col(time_column) >= item.start_received_ns)
                    & (pl.col(time_column) < item.end_received_ns_exclusive)
                    for item in intervals
                ]
            )
        ).height
    ):
        raise L2ResearchError(f"{label} interval segmentation is not one-to-one")
    return result


def segment_l2_inputs(
    book_observations: pl.DataFrame,
    depth_deltas: pl.DataFrame,
    intervals: Sequence[L2ObservedInterval],
    *,
    study_date: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Restrict and re-key books/deltas to verified continuous-observation intervals."""

    verified = _validate_intervals(intervals, study_date=study_date)
    books = _segment_one(
        book_observations,
        verified,
        study_date=study_date,
        time_column="available_ts_ns",
        label="book observations",
    )
    deltas = _segment_one(
        depth_deltas,
        verified,
        study_date=study_date,
        time_column="available_ts_ns",
        label="depth deltas",
    )
    return books, deltas


def _add_volatility_windows(frame: pl.DataFrame, windows: Sequence[int]) -> pl.DataFrame:
    group = ["symbol", "continuity_id"]
    expressions = [
        pl.col("log_mid_return_1")
        .pow(2)
        .rolling_sum(window_size=window, min_samples=1)
        .over(group)
        .sqrt()
        .alias(f"realized_volatility_w{window}")
        for window in sorted(set(windows))
    ]
    return frame.with_columns(expressions)


def _exact_clock_label(
    features: pl.DataFrame,
    endpoint: L2EndpointSpec,
    *,
    max_state_age_ns: int,
) -> pl.DataFrame:
    if endpoint.domain != "clock":
        raise L2ResearchError("exact clock label requires a clock endpoint")
    if max_state_age_ns < 0:
        raise L2ResearchError("clock max state age must be nonnegative")
    keys = ["symbol", "continuity_id"]
    decisions = features.with_columns(
        (pl.col("decision_ts_ns") + endpoint.horizon_ns).alias("clock_target_ts_ns")
    )
    right = features.select(
        *keys,
        pl.col("decision_ts_ns").alias("_target_state_ts_ns"),
        pl.col("decision_sequence").alias("_target_state_sequence"),
        pl.col("mid_price").alias("_target_state_mid"),
    ).sort(["_target_state_ts_ns", *keys, "_target_state_sequence"])
    # ``join_asof(..., strategy="backward")`` selects the last matching row.
    # Sorting equal-time states by sequence therefore makes the exact-target
    # tie-break explicit: the greatest observable sequence at ``t + h`` wins.
    joined = decisions.sort(["clock_target_ts_ns", *keys, "decision_sequence"]).join_asof(
        right,
        left_on="clock_target_ts_ns",
        right_on="_target_state_ts_ns",
        by=keys,
        strategy="backward",
        allow_exact_matches=True,
        check_sortedness=False,
    )
    state_age = pl.col("clock_target_ts_ns") - pl.col("_target_state_ts_ns")
    censored = (
        pl.col("_target_state_ts_ns").is_null()
        | (pl.col("clock_target_ts_ns") >= pl.col("observed_interval_end_ns_exclusive"))
        | (state_age > max_state_age_ns)
        | (pl.col("_target_state_sequence") < pl.col("decision_sequence"))
    )
    return (
        joined.with_columns(
            censored.alias("right_censored"),
            state_age.alias("clock_target_state_age_ns"),
            pl.lit(endpoint.horizon_ns, dtype=pl.Int64).alias("clock_horizon_ns"),
        )
        .with_columns(
            pl.when(~pl.col("right_censored"))
            .then((pl.col("_target_state_mid") / pl.col("mid_price")).log())
            .otherwise(None)
            .alias("future_mid_return"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("clock_target_ts_ns"))
            .otherwise(None)
            .alias("label_information_end_ts_ns"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_state_sequence"))
            .otherwise(None)
            .alias("label_information_end_sequence"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("continuity_id"))
            .otherwise(None)
            .alias("label_continuity_id"),
            pl.col("decision_ts_ns").alias("label_start_ts_ns"),
            pl.col("decision_sequence").alias("label_start_sequence"),
        )
        .with_columns(
            pl.when(pl.col("future_mid_return").is_null())
            .then(None)
            .when(pl.col("future_mid_return") > 0)
            .then(1)
            .when(pl.col("future_mid_return") < 0)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("future_mid_direction"),
            pl.when(pl.col("future_mid_return").is_null())
            .then(None)
            .otherwise((pl.col("future_mid_return") > 0).cast(pl.Int8))
            .alias("future_mid_up"),
        )
        .drop("_target_state_ts_ns", "_target_state_sequence", "_target_state_mid")
        .sort("decision_ts_ns", "decision_sequence")
    )


def _standardize_endpoint(
    frame: pl.DataFrame,
    endpoint: L2EndpointSpec,
    *,
    study_date: str,
    study_role: StudyRole,
) -> pl.DataFrame:
    ofi_column = f"ofi_w{endpoint.impact_ofi_window}"
    _require(frame, (ofi_column, "future_mid_return", "right_censored"), "endpoint frame")
    signed_markout = (
        pl.when(pl.col("right_censored") | pl.col("future_mid_return").is_null())
        .then(None)
        .otherwise(
            pl.when(pl.col(ofi_column) > 0)
            .then(1.0)
            .when(pl.col(ofi_column) < 0)
            .then(-1.0)
            .otherwise(0.0)
            * 10_000.0
            * pl.col("future_mid_return")
        )
    )
    result = frame.with_columns(
        pl.lit(study_date).alias("study_date"),
        pl.lit(study_role).alias("study_role"),
        pl.lit(endpoint.name).alias("endpoint_name"),
        pl.lit(endpoint.domain).alias("endpoint_domain"),
        pl.lit(endpoint.horizon_value, dtype=pl.Int64).alias("endpoint_horizon_value"),
        pl.lit(endpoint.horizon_unit).alias("endpoint_horizon_unit"),
        pl.col("continuity_id").alias("feature_continuity_id"),
        pl.col("decision_sequence").alias("decision_trade_id"),
        pl.col("max_feature_source_sequence").alias("max_feature_source_trade_id"),
        pl.col("label_start_sequence").alias("label_start_trade_id"),
        pl.col("label_information_end_sequence").alias("label_information_end_trade_id"),
        signed_markout.alias("ofi_signed_future_mid_markout_bps"),
        pl.lit(ofi_column).alias("signed_markout_side_source"),
        pl.lit("descriptive_ofi_sign_times_future_mid_log_return").alias("signed_markout_policy"),
    ).with_columns(
        pl.concat_str(
            "study_date",
            "symbol",
            "endpoint_name",
            "continuity_id",
            pl.col("decision_sequence").cast(pl.String),
            separator="::",
        ).alias("sample_id")
    )
    validate_l2_endpoint_frame(result)
    return result


def build_l2_endpoint_frames(
    book_observations: pl.DataFrame,
    depth_deltas: pl.DataFrame,
    intervals: Sequence[L2ObservedInterval],
    *,
    study_date: str,
    study_role: StudyRole,
    feature_windows: Sequence[int],
    volatility_window: int,
    clock_max_state_age_ms: int,
    endpoints: Sequence[L2EndpointSpec],
) -> Mapping[str, pl.DataFrame]:
    """Build every predeclared endpoint from one verified symbol/session capture."""

    if study_role not in _ALLOWED_ROLES:
        raise L2ResearchError(f"unsupported L2 study role {study_role!r}")
    windows = tuple(sorted(set(feature_windows)))
    if not windows or any(window < 1 for window in windows) or volatility_window < 1:
        raise L2ResearchError("L2 rolling windows must be positive")
    specs = tuple(endpoints)
    if not specs or len({item.name for item in specs}) != len(specs):
        raise L2ResearchError("L2 endpoints must be nonempty and uniquely named")
    books, deltas = segment_l2_inputs(
        book_observations,
        depth_deltas,
        intervals,
        study_date=study_date,
    )
    feature_config = FeatureConfig(
        trade_windows=windows,
        volatility_window=volatility_window,
        intensity_window=max(windows),
        label_horizon_events=1,
        large_trade_quantile=0.95,
    )
    features = build_research_features(
        books,
        None,
        feature_config,
        depth_deltas=deltas,
    )
    features = _add_volatility_windows(features, windows)
    result: dict[str, pl.DataFrame] = {}
    max_state_age_ns = clock_max_state_age_ms * _NANOSECONDS_PER_MILLISECOND
    for endpoint in specs:
        labeled = (
            add_future_event_labels(features, endpoint.horizon_value)
            if endpoint.domain == "event"
            else _exact_clock_label(features, endpoint, max_state_age_ns=max_state_age_ns)
        )
        result[endpoint.name] = _standardize_endpoint(
            labeled,
            endpoint,
            study_date=study_date,
            study_role=study_role,
        )
    return result


def validate_l2_endpoint_frame(frame: pl.DataFrame) -> None:
    """Fail closed on date, interval, feature-lineage, or label leakage."""

    required = (
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
    _require(frame, required, "L2 endpoint frame")
    for column in (
        "study_date",
        "study_role",
        "endpoint_name",
        "endpoint_domain",
        "symbol",
        "continuity_id",
        "observed_interval_id",
        "decision_ts_ns",
        "decision_sequence",
        "feature_cutoff_ts_ns",
        "max_feature_source_ts_ns",
        "max_feature_source_sequence",
        "feature_continuity_id",
        "label_start_ts_ns",
        "label_start_sequence",
        "right_censored",
        "sample_id",
    ):
        if frame.get_column(column).null_count():
            raise L2ResearchError(f"L2 endpoint column {column!r} must not contain nulls")
    if frame.get_column("study_date").n_unique() != 1:
        raise L2ResearchError("one L2 endpoint frame must contain exactly one study date")
    study_date = str(frame.get_column("study_date")[0])
    date_start, date_end = _date_bounds(study_date)
    invalid = frame.filter(
        (pl.col("study_role").is_in(list(_ALLOWED_ROLES)).not_())
        | (pl.col("decision_ts_ns") < date_start)
        | (pl.col("decision_ts_ns") >= date_end)
        | (pl.col("decision_ts_ns") < pl.col("observed_interval_start_ns"))
        | (pl.col("decision_ts_ns") >= pl.col("observed_interval_end_ns_exclusive"))
        | (pl.col("feature_cutoff_ts_ns") != pl.col("decision_ts_ns"))
        | (pl.col("max_feature_source_ts_ns") > pl.col("decision_ts_ns"))
        | (
            (pl.col("max_feature_source_ts_ns") == pl.col("decision_ts_ns"))
            & (pl.col("max_feature_source_sequence") > pl.col("decision_sequence"))
        )
        | (pl.col("feature_continuity_id") != pl.col("continuity_id"))
        | (pl.col("label_start_ts_ns") != pl.col("decision_ts_ns"))
        | (pl.col("label_start_sequence") != pl.col("decision_sequence"))
    )
    if invalid.height:
        raise L2ResearchError("L2 endpoint base timing or interval lineage is invalid")
    if frame.get_column("sample_id").n_unique() != frame.height:
        raise L2ResearchError("L2 endpoint sample identities must be unique")
    labeled = frame.filter(~pl.col("right_censored"))
    for column in (
        "future_mid_return",
        "future_mid_up",
        "label_information_end_ts_ns",
        "label_information_end_sequence",
        "label_continuity_id",
        "ofi_signed_future_mid_markout_bps",
    ):
        if labeled.get_column(column).null_count():
            raise L2ResearchError(f"uncensored L2 labels require non-null {column}")
    if labeled.filter(
        (pl.col("label_continuity_id") != pl.col("continuity_id"))
        | (pl.col("label_information_end_ts_ns") < pl.col("decision_ts_ns"))
        | (
            (pl.col("label_information_end_ts_ns") == pl.col("decision_ts_ns"))
            & (pl.col("label_information_end_sequence") <= pl.col("decision_sequence"))
        )
        | (pl.col("label_information_end_ts_ns") >= pl.col("observed_interval_end_ns_exclusive"))
        | (pl.col("label_information_end_ts_ns") >= date_end)
        | (pl.col("label_information_end_sequence") < pl.col("decision_sequence"))
    ).height:
        raise L2ResearchError("L2 labels are not strictly future and interval-local")
    censored = frame.filter(pl.col("right_censored"))
    if censored.filter(
        pl.col("future_mid_return").is_not_null()
        | pl.col("future_mid_up").is_not_null()
        | pl.col("label_information_end_ts_ns").is_not_null()
        | pl.col("label_information_end_sequence").is_not_null()
        | pl.col("label_continuity_id").is_not_null()
        | pl.col("ofi_signed_future_mid_markout_bps").is_not_null()
    ).height:
        raise L2ResearchError("censored L2 labels must not retain future outcomes")


def _finite_quantile(frame: pl.DataFrame, column: str, quantile: float) -> float:
    value = frame.get_column(column).quantile(quantile, interpolation="linear")
    if value is None or not math.isfinite(float(value)):
        raise L2ResearchError(f"cannot fit finite train threshold for {column}")
    return float(value)


def fit_l2_regime_thresholds(
    train_frame: pl.DataFrame,
    *,
    lower_quantile: float,
    upper_quantile: float,
    volatility_column: str,
) -> L2RegimeFit:
    """Fit volatility/liquidity thresholds on the train session only."""

    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise L2ResearchError("regime quantiles must satisfy 0 < lower < upper < 1")
    _require(
        train_frame,
        (
            "symbol",
            "study_date",
            "study_role",
            "feature_ready",
            volatility_column,
            "spread_bps",
            "depth_total_l1",
        ),
        "L2 regime training frame",
    )
    if set(train_frame.get_column("study_role").unique().to_list()) != {"train"}:
        raise L2ResearchError("regime thresholds may be fit only on the train session")
    if train_frame.get_column("symbol").n_unique() != 1:
        raise L2ResearchError("regime thresholds are fit separately per symbol")
    eligible = train_frame.filter(pl.col("feature_ready")).drop_nulls(
        [volatility_column, "spread_bps", "depth_total_l1"]
    )
    if eligible.is_empty():
        raise L2ResearchError("no feature-ready train rows are available for regime fitting")
    symbol = str(eligible.get_column("symbol")[0])
    study_date = str(eligible.get_column("study_date")[0])
    return L2RegimeFit(
        symbol=symbol,
        study_date=study_date,
        volatility_column=volatility_column,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        thresholds=RegimeThresholds(
            volatility_low=_finite_quantile(eligible, volatility_column, lower_quantile),
            volatility_high=_finite_quantile(eligible, volatility_column, upper_quantile),
            spread_tight_bps=_finite_quantile(eligible, "spread_bps", lower_quantile),
            spread_wide_bps=_finite_quantile(eligible, "spread_bps", upper_quantile),
            depth_low=_finite_quantile(eligible, "depth_total_l1", lower_quantile),
            depth_high=_finite_quantile(eligible, "depth_total_l1", upper_quantile),
        ),
    )


def apply_l2_regimes(frame: pl.DataFrame, fitted: L2RegimeFit) -> pl.DataFrame:
    """Apply one symbol's persisted train-only thresholds and numeric dummies."""

    if set(str(value) for value in frame.get_column("symbol").unique().to_list()) != {
        fitted.symbol
    }:
        raise L2ResearchError("regime thresholds and endpoint symbol disagree")
    assigned = assign_market_regimes(
        frame,
        train_thresholds={fitted.symbol: fitted.thresholds},
        volatility_column=fitted.volatility_column,
    )
    return assigned.with_columns(
        (pl.col("volatility_regime") == "low").cast(pl.Float64).alias("volatility_regime_low"),
        (pl.col("volatility_regime") == "high").cast(pl.Float64).alias("volatility_regime_high"),
        (pl.col("liquidity_regime") == "liquid").cast(pl.Float64).alias("liquidity_regime_liquid"),
        (pl.col("liquidity_regime") == "stressed")
        .cast(pl.Float64)
        .alias("liquidity_regime_stressed"),
        pl.lit(fitted.study_date).alias("regime_fit_study_date"),
        pl.lit("train_session_only").alias("regime_fit_scope"),
    )


def l2_model_feature_columns(frame: pl.DataFrame, *, windows: Sequence[int]) -> tuple[str, ...]:
    """Return the exact book-only feature ladder; all-zero trade proxies are forbidden."""

    rolling = tuple(sorted(set(windows)))
    columns = (
        "spread_bps",
        "depth_total_l1",
        "depth_total_l5",
        "depth_total_l10",
        "queue_imbalance_l1",
        "queue_imbalance_l5",
        "queue_imbalance_l10",
        "microprice_deviation_bps",
        "ofi_l1",
        *(f"ofi_w{window}" for window in rolling),
        *(f"cancellation_intensity_w{window}" for window in rolling),
        *(f"realized_volatility_w{window}" for window in rolling),
        *_REGIME_FEATURES,
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise L2ResearchError(f"L2 model feature contract is incomplete: {missing}")
    forbidden = [
        name
        for name in columns
        if name.startswith(("signed_trade_", "trade_volume_", "trade_count_", "trade_intensity_"))
    ]
    if forbidden:
        raise L2ResearchError(f"trade-only proxies cannot enter the book-only model: {forbidden}")
    return tuple(columns)


def dependency_block_expression(endpoint: L2EndpointSpec) -> pl.Expr:
    """Return a continuity-local paired dependency-block identifier expression."""

    if endpoint.domain == "event":
        assert endpoint.paired_block_events is not None
        block = (
            pl.col("decision_sequence")
            .rank("ordinal")
            .over(["study_date", "symbol", "continuity_id"])
            - 1
        ) // endpoint.paired_block_events
    else:
        assert endpoint.paired_block_milliseconds is not None
        block_ns = endpoint.paired_block_milliseconds * _NANOSECONDS_PER_MILLISECOND
        block = (pl.col("decision_ts_ns") - pl.col("observed_interval_start_ns")) // block_ns
    return pl.concat_str(
        "study_date",
        "symbol",
        "continuity_id",
        block.cast(pl.Int64).cast(pl.String),
        separator="::",
    ).alias("bootstrap_block")


__all__ = [
    "EndpointDomain",
    "L2EndpointSpec",
    "L2ObservedInterval",
    "L2RegimeFit",
    "L2ResearchError",
    "StudyRole",
    "apply_l2_regimes",
    "build_l2_endpoint_frames",
    "dependency_block_expression",
    "fit_l2_regime_thresholds",
    "l2_model_feature_columns",
    "segment_l2_inputs",
    "validate_l2_endpoint_frame",
]
