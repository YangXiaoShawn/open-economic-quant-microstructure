"""Reproducible descriptive market-microstructure diagnostics.

No function in this module claims causal or tradable significance.  Thresholds
used for large trades, shocks, recovery, and regimes are supplied explicitly by
the caller and are tagged as train-period inputs; they are never estimated from
the analyzed evaluation sample.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray


class DescriptiveAnalysisError(ValueError):
    """Raised when a descriptive diagnostic lacks a valid input contract."""


@dataclass(frozen=True, slots=True)
class HalfLifeResult:
    """Correlation-decay curve and per-instrument descriptive half-life summary."""

    curve: pl.DataFrame
    summary: pl.DataFrame


@dataclass(frozen=True, slots=True)
class LiquidityShockThresholds:
    """Externally fitted shock and recovery thresholds for one instrument."""

    spread_shock_bps: float
    depth_shock_max: float
    spread_recovery_bps: float
    depth_recovery_min: float
    max_recovery_events: int

    def __post_init__(self) -> None:
        values = (
            self.spread_shock_bps,
            self.depth_shock_max,
            self.spread_recovery_bps,
            self.depth_recovery_min,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise DescriptiveAnalysisError("liquidity thresholds must be finite and nonnegative")
        if self.max_recovery_events < 1:
            raise DescriptiveAnalysisError("max_recovery_events must be positive")


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Externally fitted volatility and liquidity regime boundaries."""

    volatility_low: float
    volatility_high: float
    spread_tight_bps: float
    spread_wide_bps: float
    depth_low: float
    depth_high: float

    def __post_init__(self) -> None:
        values = (
            self.volatility_low,
            self.volatility_high,
            self.spread_tight_bps,
            self.spread_wide_bps,
            self.depth_low,
            self.depth_high,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise DescriptiveAnalysisError("regime thresholds must be finite and nonnegative")
        if self.volatility_low > self.volatility_high:
            raise DescriptiveAnalysisError("volatility_low cannot exceed volatility_high")
        if self.spread_tight_bps > self.spread_wide_bps:
            raise DescriptiveAnalysisError("tight spread cannot exceed wide spread")
        if self.depth_low > self.depth_high:
            raise DescriptiveAnalysisError("low depth cannot exceed high depth")


def _require(frame: pl.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DescriptiveAnalysisError(f"{table} is missing required columns: {missing}")
    if frame.is_empty():
        raise DescriptiveAnalysisError(f"{table} must not be empty")


def _finite_pair(
    frame: pl.DataFrame, left: str, right: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    paired = frame.select(left, right).drop_nulls()
    x = paired.get_column(left).to_numpy().astype(np.float64)
    y = paired.get_column(right).to_numpy().astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _average_rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def intraday_liquidity_summary(
    frame: pl.DataFrame,
    *,
    timestamp_column: str = "decision_ts_ns",
    bucket_minutes: int = 60,
    utc_offset_minutes: int = 0,
    spread_column: str = "spread_bps",
    depth_column: str = "depth_total_l1",
    imbalance_column: str = "queue_imbalance_l1",
) -> pl.DataFrame:
    """Summarize liquidity by fixed minute-of-day buckets."""

    _require(
        frame,
        ("symbol", timestamp_column, spread_column, depth_column, imbalance_column),
        "intraday frame",
    )
    if bucket_minutes < 1 or bucket_minutes > 1_440 or 1_440 % bucket_minutes:
        raise DescriptiveAnalysisError("bucket_minutes must be a positive divisor of 1440")
    minute_ns = 60_000_000_000
    prepared = frame.with_columns(
        (
            ((pl.col(timestamp_column) // minute_ns + utc_offset_minutes) % 1_440)
            // bucket_minutes
            * bucket_minutes
        )
        .cast(pl.Int32)
        .alias("intraday_bucket_start_minute")
    )
    result = (
        prepared.group_by("symbol", "intraday_bucket_start_minute")
        .agg(
            pl.len().alias("n_observations"),
            pl.col(spread_column).mean().alias("mean_spread_bps"),
            pl.col(spread_column).median().alias("median_spread_bps"),
            pl.col(depth_column).mean().alias("mean_depth_l1"),
            pl.col(depth_column).median().alias("median_depth_l1"),
            pl.col(imbalance_column).mean().alias("mean_queue_imbalance_l1"),
        )
        .with_columns(
            (
                (pl.col("intraday_bucket_start_minute") // 60).cast(pl.String).str.pad_start(2, "0")
                + pl.lit(":")
                + (pl.col("intraday_bucket_start_minute") % 60)
                .cast(pl.String)
                .str.pad_start(2, "0")
            ).alias("intraday_bucket_label"),
            pl.lit(utc_offset_minutes, dtype=pl.Int32).alias("utc_offset_minutes"),
            pl.lit("intraday_liquidity_descriptive").alias("analysis_kind"),
            pl.lit(True).alias("descriptive_only"),
        )
        .sort("symbol", "intraday_bucket_start_minute")
    )
    return result


def ofi_future_return_association(
    frame: pl.DataFrame,
    *,
    horizon_return_columns: Mapping[int, str],
    ofi_column: str = "ofi_l1",
    min_observations: int = 3,
) -> pl.DataFrame:
    """Report descriptive OFI/return slopes and rank/linear correlations."""

    if not horizon_return_columns or any(horizon <= 0 for horizon in horizon_return_columns):
        raise DescriptiveAnalysisError("supplied event horizons must be positive")
    _require(frame, ("symbol", ofi_column, *horizon_return_columns.values()), "OFI frame")
    if min_observations < 2:
        raise DescriptiveAnalysisError("min_observations must be at least two")

    rows: list[dict[str, object]] = []
    for symbol_frame in frame.partition_by("symbol", maintain_order=True):
        symbol = str(symbol_frame.get_column("symbol")[0])
        for horizon, return_column in sorted(horizon_return_columns.items()):
            x, y = _finite_pair(symbol_frame, ofi_column, return_column)
            enough = x.size >= min_observations
            variance = float(np.var(x)) if x.size else math.nan
            slope = (
                float(np.mean((x - x.mean()) * (y - y.mean())) / variance)
                if enough and variance > 0
                else math.nan
            )
            intercept = float(y.mean() - slope * x.mean()) if math.isfinite(slope) else math.nan
            rows.append(
                {
                    "symbol": symbol,
                    "horizon_events": horizon,
                    "ofi_column": ofi_column,
                    "return_column": return_column,
                    "n_observations": int(x.size),
                    "pearson_correlation": _pearson(x, y) if enough else math.nan,
                    "spearman_correlation": (
                        _pearson(_average_rank(x), _average_rank(y)) if enough else math.nan
                    ),
                    "ols_slope_return_per_ofi_unit": slope,
                    "ols_intercept": intercept,
                    "mean_future_return": float(y.mean()) if y.size else math.nan,
                    "analysis_status": "ok" if enough else "insufficient_observations",
                    "analysis_kind": "ofi_future_return_descriptive_association",
                    "descriptive_only": True,
                }
            )
    return pl.DataFrame(rows).sort("symbol", "horizon_events")


def estimate_signal_half_life(
    association_curve: pl.DataFrame,
    *,
    correlation_column: str = "pearson_correlation",
) -> HalfLifeResult:
    """Estimate correlation half-life from a caller-supplied horizon curve."""

    _require(
        association_curve,
        ("symbol", "horizon_events", correlation_column, "n_observations"),
        "association curve",
    )
    curve = association_curve.with_columns(
        pl.col(correlation_column).abs().alias("absolute_correlation")
    ).sort("symbol", "horizon_events")
    curve_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for symbol_curve in curve.partition_by("symbol", maintain_order=True):
        symbol = str(symbol_curve.get_column("symbol")[0])
        horizons = symbol_curve.get_column("horizon_events").to_numpy().astype(np.float64)
        correlations = symbol_curve.get_column("absolute_correlation").to_numpy().astype(np.float64)
        finite = np.isfinite(horizons) & np.isfinite(correlations)
        horizons = horizons[finite]
        correlations = correlations[finite]
        if not correlations.size:
            for row in symbol_curve.iter_rows(named=True):
                curve_rows.append(
                    {
                        **row,
                        "normalized_absolute_correlation": None,
                        "half_correlation_threshold": None,
                        "analysis_kind": "signal_decay_curve_descriptive",
                        "descriptive_only": True,
                    }
                )
            summaries.append(
                {
                    "symbol": symbol,
                    "reference_horizon_events": None,
                    "reference_absolute_correlation": None,
                    "half_correlation_threshold": None,
                    "first_crossing_half_life_events": None,
                    "exponential_half_life_events": None,
                    "analysis_status": "no_finite_correlations",
                    "analysis_kind": "signal_half_life_descriptive",
                    "descriptive_only": True,
                }
            )
            continue
        reference = float(correlations[0])
        half_threshold = reference / 2.0
        crossing = horizons[correlations <= half_threshold]
        first_crossing = float(crossing[0]) if crossing.size else None
        positive = correlations > 0
        exponential_half_life: float | None = None
        if positive.sum() >= 2:
            slope = float(np.polyfit(horizons[positive], np.log(correlations[positive]), 1)[0])
            if slope < 0:
                exponential_half_life = math.log(2.0) / -slope
        status = "ok" if reference > 0 else "zero_reference_correlation"
        summaries.append(
            {
                "symbol": symbol,
                "reference_horizon_events": int(horizons[0]),
                "reference_absolute_correlation": reference,
                "half_correlation_threshold": half_threshold,
                "first_crossing_half_life_events": first_crossing,
                "exponential_half_life_events": exponential_half_life,
                "analysis_status": status,
                "analysis_kind": "signal_half_life_descriptive",
                "descriptive_only": True,
            }
        )
        for row in symbol_curve.iter_rows(named=True):
            value = float(row["absolute_correlation"])
            curve_rows.append(
                {
                    **row,
                    "normalized_absolute_correlation": (
                        value / reference if reference > 0 and math.isfinite(value) else None
                    ),
                    "half_correlation_threshold": half_threshold,
                    "analysis_kind": "signal_decay_curve_descriptive",
                    "descriptive_only": True,
                }
            )
    return HalfLifeResult(
        curve=pl.DataFrame(curve_rows).sort("symbol", "horizon_events"),
        summary=pl.DataFrame(summaries).sort("symbol"),
    )


def large_trade_price_impact_summary(
    frame: pl.DataFrame,
    *,
    impact_columns: Mapping[int, str],
    train_quantity_thresholds: Mapping[str, float],
    quantity_column: str = "quantity",
) -> pl.DataFrame:
    """Compare impact above/below externally supplied train-period size cutoffs."""

    if not impact_columns or any(horizon <= 0 for horizon in impact_columns):
        raise DescriptiveAnalysisError("impact horizons must be positive")
    _require(frame, ("symbol", quantity_column, *impact_columns.values()), "trade-impact frame")
    symbols = {str(value) for value in frame.get_column("symbol").unique()}
    missing = sorted(symbols.difference(train_quantity_thresholds))
    if missing:
        raise DescriptiveAnalysisError(f"missing train-period quantity thresholds: {missing}")
    if any(
        not math.isfinite(train_quantity_thresholds[symbol])
        or train_quantity_thresholds[symbol] <= 0
        for symbol in symbols
    ):
        raise DescriptiveAnalysisError("large-trade thresholds must be finite and positive")

    rows: list[dict[str, object]] = []
    for symbol_frame in frame.partition_by("symbol", maintain_order=True):
        symbol = str(symbol_frame.get_column("symbol")[0])
        threshold = train_quantity_thresholds[symbol]
        for horizon, impact_column in sorted(impact_columns.items()):
            for is_large in (False, True):
                subset = (
                    symbol_frame.filter((pl.col(quantity_column) >= threshold) == is_large)
                    .select(impact_column)
                    .drop_nulls()
                )
                values = subset.get_column(impact_column).to_numpy().astype(np.float64)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "symbol": symbol,
                        "horizon_events": horizon,
                        "impact_column": impact_column,
                        "large_trade": is_large,
                        "train_quantity_threshold": threshold,
                        "n_observations": int(values.size),
                        "mean_signed_impact_bps": (
                            float(values.mean()) if values.size else math.nan
                        ),
                        "median_signed_impact_bps": (
                            float(np.median(values)) if values.size else math.nan
                        ),
                        "mean_absolute_impact_bps": (
                            float(np.abs(values).mean()) if values.size else math.nan
                        ),
                        "threshold_source": "caller_supplied_train_period",
                        "analysis_kind": "large_trade_price_impact_descriptive",
                        "descriptive_only": True,
                    }
                )
    return pl.DataFrame(rows).sort("symbol", "horizon_events", "large_trade")


def liquidity_recovery_summary(
    frame: pl.DataFrame,
    *,
    train_thresholds: Mapping[str, LiquidityShockThresholds],
    spread_column: str = "spread_bps",
    depth_column: str = "depth_total_l1",
    time_column: str = "decision_ts_ns",
    sequence_column: str = "decision_sequence",
) -> pl.DataFrame:
    """Track nonoverlapping recovery episodes after threshold-defined shocks."""

    _require(
        frame,
        ("symbol", "continuity_id", spread_column, depth_column, time_column, sequence_column),
        "liquidity-recovery frame",
    )
    symbols = {str(value) for value in frame.get_column("symbol").unique()}
    missing = sorted(symbols.difference(train_thresholds))
    if missing:
        raise DescriptiveAnalysisError(f"missing train-period liquidity thresholds: {missing}")

    episodes: list[dict[str, object]] = []
    for segment in frame.sort(["symbol", "continuity_id", sequence_column]).partition_by(
        ["symbol", "continuity_id"], maintain_order=True
    ):
        symbol = str(segment.get_column("symbol")[0])
        continuity_id = str(segment.get_column("continuity_id")[0])
        threshold = train_thresholds[symbol]
        rows = list(segment.iter_rows(named=True))
        index = 0
        while index < len(rows):
            row = rows[index]
            spread = cast(float, row[spread_column])
            depth = cast(float, row[depth_column])
            spread_shock = spread >= threshold.spread_shock_bps
            depth_shock = depth <= threshold.depth_shock_max
            if not (spread_shock or depth_shock):
                index += 1
                continue
            search_end = min(len(rows) - 1, index + threshold.max_recovery_events)
            recovery_index: int | None = None
            for candidate_index in range(index + 1, search_end + 1):
                candidate = rows[candidate_index]
                if (
                    cast(float, candidate[spread_column]) <= threshold.spread_recovery_bps
                    and cast(float, candidate[depth_column]) >= threshold.depth_recovery_min
                ):
                    recovery_index = candidate_index
                    break
            full_horizon_observed = index + threshold.max_recovery_events < len(rows)
            right_censored = recovery_index is None and not full_horizon_observed
            information_end_index = (
                recovery_index
                if recovery_index is not None
                else index + threshold.max_recovery_events
                if full_horizon_observed
                else None
            )
            shock_ts = cast(int, row[time_column])
            recovery_row = rows[recovery_index] if recovery_index is not None else None
            episodes.append(
                {
                    "symbol": symbol,
                    "continuity_id": continuity_id,
                    "shock_ts_ns": shock_ts,
                    "shock_sequence": cast(int, row[sequence_column]),
                    "shock_spread_bps": spread,
                    "shock_depth_l1": depth,
                    "spread_shock": spread_shock,
                    "depth_shock": depth_shock,
                    "recovered": None if right_censored else recovery_index is not None,
                    "recovery_events": (
                        recovery_index - index if recovery_index is not None else None
                    ),
                    "recovery_time_ns": (
                        cast(int, recovery_row[time_column]) - shock_ts
                        if recovery_row is not None
                        else None
                    ),
                    "recovery_ts_ns": (
                        cast(int, recovery_row[time_column]) if recovery_row is not None else None
                    ),
                    "recovery_right_censored": right_censored,
                    "recovery_censor_reason": (
                        "segment_ends_before_max_horizon" if right_censored else None
                    ),
                    "recovery_information_end_ts_ns": (
                        cast(int, rows[information_end_index][time_column])
                        if information_end_index is not None
                        else None
                    ),
                    "spread_shock_threshold_bps": threshold.spread_shock_bps,
                    "depth_shock_threshold_max": threshold.depth_shock_max,
                    "spread_recovery_threshold_bps": threshold.spread_recovery_bps,
                    "depth_recovery_threshold_min": threshold.depth_recovery_min,
                    "max_recovery_events": threshold.max_recovery_events,
                    "threshold_source": "caller_supplied_train_period",
                    "analysis_kind": "liquidity_recovery_descriptive",
                    "descriptive_only": True,
                }
            )
            index = (recovery_index + 1) if recovery_index is not None else search_end + 1
    if not episodes:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "continuity_id": pl.String,
                "shock_ts_ns": pl.Int64,
                "analysis_kind": pl.String,
                "descriptive_only": pl.Boolean,
            }
        )
    return pl.DataFrame(episodes, infer_schema_length=None).sort("symbol", "shock_ts_ns")


def assign_market_regimes(
    frame: pl.DataFrame,
    *,
    train_thresholds: Mapping[str, RegimeThresholds],
    volatility_column: str,
    spread_column: str = "spread_bps",
    depth_column: str = "depth_total_l1",
) -> pl.DataFrame:
    """Assign regimes using only explicit instrument-specific train thresholds."""

    _require(
        frame,
        ("symbol", volatility_column, spread_column, depth_column),
        "market-regime frame",
    )
    symbols = {str(value) for value in frame.get_column("symbol").unique()}
    missing = sorted(symbols.difference(train_thresholds))
    if missing:
        raise DescriptiveAnalysisError(f"missing train-period regime thresholds: {missing}")
    outputs: list[pl.DataFrame] = []
    for symbol_frame in frame.partition_by("symbol", maintain_order=True):
        symbol = str(symbol_frame.get_column("symbol")[0])
        threshold = train_thresholds[symbol]
        outputs.append(
            symbol_frame.with_columns(
                pl.when(pl.col(volatility_column) <= threshold.volatility_low)
                .then(pl.lit("low"))
                .when(pl.col(volatility_column) >= threshold.volatility_high)
                .then(pl.lit("high"))
                .otherwise(pl.lit("medium"))
                .alias("volatility_regime"),
                pl.when(
                    (pl.col(spread_column) >= threshold.spread_wide_bps)
                    | (pl.col(depth_column) <= threshold.depth_low)
                )
                .then(pl.lit("stressed"))
                .when(
                    (pl.col(spread_column) <= threshold.spread_tight_bps)
                    & (pl.col(depth_column) >= threshold.depth_high)
                )
                .then(pl.lit("liquid"))
                .otherwise(pl.lit("normal"))
                .alias("liquidity_regime"),
                pl.lit(threshold.volatility_low).alias("train_volatility_low"),
                pl.lit(threshold.volatility_high).alias("train_volatility_high"),
                pl.lit(threshold.spread_tight_bps).alias("train_spread_tight_bps"),
                pl.lit(threshold.spread_wide_bps).alias("train_spread_wide_bps"),
                pl.lit(threshold.depth_low).alias("train_depth_low"),
                pl.lit(threshold.depth_high).alias("train_depth_high"),
            ).with_columns(
                (pl.col("volatility_regime") + pl.lit("__") + pl.col("liquidity_regime")).alias(
                    "joint_market_regime"
                ),
                pl.lit("caller_supplied_train_period").alias("regime_threshold_source"),
                pl.lit("volatility_liquidity_regime_descriptive").alias("analysis_kind"),
                pl.lit(True).alias("descriptive_only"),
            )
        )
    return pl.concat(outputs, how="vertical_relaxed")


def regime_outcome_summary(
    regime_frame: pl.DataFrame,
    *,
    outcome_columns: Sequence[str],
) -> pl.DataFrame:
    """Summarize supplied outcomes without using them to define regimes."""

    _require(
        regime_frame,
        ("symbol", "volatility_regime", "liquidity_regime", *outcome_columns),
        "regime outcomes",
    )
    expressions: list[pl.Expr] = [pl.len().alias("n_observations")]
    for column in outcome_columns:
        expressions.extend(
            [
                pl.col(column).mean().alias(f"mean__{column}"),
                pl.col(column).median().alias(f"median__{column}"),
            ]
        )
    return (
        regime_frame.group_by("symbol", "volatility_regime", "liquidity_regime")
        .agg(expressions)
        .with_columns(
            pl.lit("regime_outcomes_descriptive").alias("analysis_kind"),
            pl.lit(True).alias("descriptive_only"),
        )
        .sort("symbol", "volatility_regime", "liquidity_regime")
    )


def cross_instrument_stability_summary(
    effects: pl.DataFrame,
    *,
    value_column: str,
    comparison_columns: Sequence[str] = ("horizon_events",),
    instrument_column: str = "symbol",
) -> pl.DataFrame:
    """Summarize effect direction and dispersion across instruments."""

    _require(effects, (instrument_column, value_column, *comparison_columns), "effects")
    rows: list[dict[str, object]] = []
    partitions = (
        effects.partition_by(list(comparison_columns), maintain_order=True)
        if comparison_columns
        else [effects]
    )
    for partition in partitions:
        by_instrument = partition.group_by(instrument_column).agg(
            pl.col(value_column).mean().alias("_instrument_effect")
        )
        values = (
            by_instrument.get_column("_instrument_effect")
            .drop_nulls()
            .to_numpy()
            .astype(np.float64)
        )
        values = values[np.isfinite(values)]
        nonzero = values[values != 0]
        sign_agreement = (
            max(float((nonzero > 0).mean()), float((nonzero < 0).mean()))
            if nonzero.size
            else math.nan
        )
        row: dict[str, object] = {
            column: partition.get_column(column)[0] for column in comparison_columns
        }
        row.update(
            {
                "value_column": value_column,
                "n_instruments": int(values.size),
                "mean_effect": float(values.mean()) if values.size else math.nan,
                "median_effect": float(np.median(values)) if values.size else math.nan,
                "effect_std": float(values.std(ddof=0)) if values.size else math.nan,
                "minimum_effect": float(values.min()) if values.size else math.nan,
                "maximum_effect": float(values.max()) if values.size else math.nan,
                "sign_agreement_fraction": sign_agreement,
                "analysis_kind": "cross_instrument_stability_descriptive",
                "descriptive_only": True,
            }
        )
        rows.append(row)
    return (
        pl.DataFrame(rows).sort(list(comparison_columns))
        if comparison_columns
        else pl.DataFrame(rows)
    )


def _population_stability_index(
    reference: NDArray[np.float64],
    comparison: NDArray[np.float64],
    *,
    bins: int,
    reference_missing: int,
    comparison_missing: int,
) -> float:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    internal = (
        np.unique(np.quantile(reference, quantiles)[1:-1]) if reference.size else np.array([])
    )
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    reference_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    comparison_counts = np.histogram(comparison, bins=edges)[0].astype(np.float64)
    reference_counts = np.append(reference_counts, reference_missing)
    comparison_counts = np.append(comparison_counts, comparison_missing)
    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * reference_counts.size
    )
    comparison_share = (comparison_counts + epsilon) / (
        comparison_counts.sum() + epsilon * comparison_counts.size
    )
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def _max_cdf_distance(reference: NDArray[np.float64], comparison: NDArray[np.float64]) -> float:
    if not reference.size or not comparison.size:
        return math.nan
    points = np.sort(np.unique(np.concatenate((reference, comparison))))
    reference_cdf = np.searchsorted(np.sort(reference), points, side="right") / reference.size
    comparison_cdf = np.searchsorted(np.sort(comparison), points, side="right") / comparison.size
    return float(np.max(np.abs(reference_cdf - comparison_cdf)))


def feature_stability_summary(
    reference: pl.DataFrame,
    comparison: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    group_columns: Sequence[str] = ("symbol",),
    bins: int = 10,
) -> pl.DataFrame:
    """Compare feature distributions using bins learned from reference only."""

    if not feature_columns:
        raise DescriptiveAnalysisError("feature_columns must not be empty")
    if bins < 2:
        raise DescriptiveAnalysisError("feature stability requires at least two bins")
    _require(reference, (*group_columns, *feature_columns), "reference features")
    _require(comparison, (*group_columns, *feature_columns), "comparison features")

    if group_columns:
        reference_groups = reference.partition_by(list(group_columns), as_dict=True)
        comparison_groups = comparison.partition_by(list(group_columns), as_dict=True)
        missing_groups = sorted(set(reference_groups).difference(comparison_groups), key=str)
        if missing_groups:
            raise DescriptiveAnalysisError(
                f"comparison is missing reference groups: {missing_groups}"
            )
    else:
        reference_groups = {(): reference}
        comparison_groups = {(): comparison}

    rows: list[dict[str, object]] = []
    for key, reference_group in reference_groups.items():
        comparison_group = comparison_groups[key]
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_columns, key_tuple, strict=True))
        for feature in feature_columns:
            reference_series = reference_group.get_column(feature)
            comparison_series = comparison_group.get_column(feature)
            reference_values = reference_series.drop_nulls().to_numpy().astype(np.float64)
            comparison_values = comparison_series.drop_nulls().to_numpy().astype(np.float64)
            reference_values = reference_values[np.isfinite(reference_values)]
            comparison_values = comparison_values[np.isfinite(comparison_values)]
            reference_invalid = reference_group.height - reference_values.size
            comparison_invalid = comparison_group.height - comparison_values.size
            reference_mean = float(reference_values.mean()) if reference_values.size else math.nan
            comparison_mean = (
                float(comparison_values.mean()) if comparison_values.size else math.nan
            )
            reference_std = (
                float(reference_values.std(ddof=0)) if reference_values.size else math.nan
            )
            comparison_std = (
                float(comparison_values.std(ddof=0)) if comparison_values.size else math.nan
            )
            rows.append(
                {
                    **group_values,
                    "feature": feature,
                    "reference_n": int(reference_group.height),
                    "comparison_n": int(comparison_group.height),
                    "reference_missing_rate": reference_invalid / reference_group.height,
                    "comparison_missing_rate": comparison_invalid / comparison_group.height,
                    "reference_mean": reference_mean,
                    "comparison_mean": comparison_mean,
                    "reference_std": reference_std,
                    "comparison_std": comparison_std,
                    "standardized_mean_shift": (
                        (comparison_mean - reference_mean) / reference_std
                        if reference_std > 0 and math.isfinite(comparison_mean)
                        else None
                    ),
                    "variance_ratio": (
                        (comparison_std**2) / (reference_std**2)
                        if reference_std > 0 and math.isfinite(comparison_std)
                        else None
                    ),
                    "population_stability_index": _population_stability_index(
                        reference_values,
                        comparison_values,
                        bins=bins,
                        reference_missing=reference_invalid,
                        comparison_missing=comparison_invalid,
                    ),
                    "max_empirical_cdf_distance": _max_cdf_distance(
                        reference_values, comparison_values
                    ),
                    "reference_constant": bool(reference_std == 0),
                    "bin_source": "reference_period_only",
                    "analysis_kind": "feature_stability_descriptive",
                    "descriptive_only": True,
                }
            )
    return pl.DataFrame(rows).sort([*group_columns, "feature"])


__all__ = [
    "DescriptiveAnalysisError",
    "HalfLifeResult",
    "LiquidityShockThresholds",
    "RegimeThresholds",
    "assign_market_regimes",
    "cross_instrument_stability_summary",
    "estimate_signal_half_life",
    "feature_stability_summary",
    "intraday_liquidity_summary",
    "large_trade_price_impact_summary",
    "liquidity_recovery_summary",
    "ofi_future_return_association",
    "regime_outcome_summary",
]
