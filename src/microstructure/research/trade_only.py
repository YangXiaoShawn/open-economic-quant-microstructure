"""Leakage-safe empirical features and labels from normalized trades alone.

Trade availability time is the decision clock.  Feature windows contain only the
current trade and earlier trades in the same ``(symbol, continuity_id)`` segment.
Future labels advance by trade ID inside that segment and preserve their explicit
information end so purged time-series evaluation can treat overlap correctly.
"""

from __future__ import annotations

import polars as pl

from microstructure.config import FeatureConfig
from microstructure.research.features import (
    ResearchDataError,
    TemporalAudit,
    TemporalLeakageError,
)

_GROUP = ["symbol", "continuity_id"]
_REQUIRED_TRADES = frozenset(
    {
        "symbol",
        "continuity_id",
        "trade_id",
        "available_ts_ns",
        "event_ts_ns",
        "price",
        "quantity",
        "aggressor_side",
    }
)


def _require_columns(frame: pl.DataFrame, required: frozenset[str], table: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ResearchDataError(f"{table} is missing required columns: {missing}")
    if frame.is_empty():
        raise ResearchDataError(f"{table} must not be empty")


def _validate_config(config: FeatureConfig) -> None:
    windows = (*config.trade_windows, config.intensity_window, config.volatility_window)
    if not config.trade_windows or any(window < 1 for window in windows):
        raise ResearchDataError("trade-only feature windows must be nonempty and positive")
    if config.label_horizon_events < 1:
        raise ResearchDataError("trade-only label horizon must be positive")


def _validate_normalized_trades(trades: pl.DataFrame) -> pl.DataFrame:
    _require_columns(trades, _REQUIRED_TRADES, "normalized trades")
    invalid = trades.filter(
        pl.col("symbol").is_null()
        | pl.col("continuity_id").is_null()
        | pl.col("trade_id").is_null()
        | pl.col("available_ts_ns").is_null()
        | pl.col("event_ts_ns").is_null()
        | (pl.col("available_ts_ns") < pl.col("event_ts_ns"))
        | (~pl.col("price").cast(pl.Float64).is_finite())
        | (pl.col("price") <= 0)
        | (~pl.col("quantity").cast(pl.Float64).is_finite())
        | (pl.col("quantity") <= 0)
        | (~pl.col("aggressor_side").cast(pl.String).str.to_lowercase().is_in(["buy", "sell"]))
    )
    if not invalid.is_empty():
        raise ResearchDataError(
            "normalized trades require segment identity, observable timing, positive finite values, "
            "and buy/sell aggressor side"
        )

    duplicates = trades.group_by(*_GROUP, "trade_id").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ResearchDataError("trade IDs must be unique within each continuity segment")

    ordered = trades.sort([*_GROUP, "trade_id"])
    backwards = ordered.filter(
        pl.col("available_ts_ns") < pl.col("available_ts_ns").shift(1).over(_GROUP)
    )
    if not backwards.is_empty():
        raise ResearchDataError(
            "available_ts_ns must be nondecreasing by trade_id within each continuity segment"
        )
    return ordered


def build_trade_only_features(trades: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """Build causal rolling trade features on the availability-time clock."""

    _validate_config(config)
    ordered = _validate_normalized_trades(trades)
    prepared = (
        ordered.with_columns(
            pl.col("event_ts_ns").alias("market_event_ts_ns"),
            pl.col("available_ts_ns").alias("decision_ts_ns"),
            pl.col("available_ts_ns").alias("feature_cutoff_ts_ns"),
            pl.col("trade_id").alias("decision_trade_id"),
            pl.col("trade_id").alias("decision_sequence"),
            pl.col("continuity_id").alias("feature_continuity_id"),
            pl.when(pl.col("aggressor_side").str.to_lowercase() == "buy")
            .then(1.0)
            .otherwise(-1.0)
            .alias("trade_sign"),
            pl.lit(1, dtype=pl.Int64).alias("_trade_observation"),
        )
        .with_columns(
            (pl.col("quantity") * pl.col("trade_sign")).alias("signed_trade_quantity"),
            pl.col("price").shift(1).over(_GROUP).alias("_previous_trade_price"),
            pl.col("available_ts_ns").first().over(_GROUP).alias("_segment_start_ts_ns"),
            pl.col("trade_id").cum_count().over(_GROUP).alias("history_trades"),
            pl.concat_str(
                ["symbol", "continuity_id", pl.col("trade_id").cast(pl.String)],
                separator=":",
            ).alias("sample_id"),
        )
        .with_columns(
            pl.when(pl.col("_previous_trade_price").is_not_null())
            .then((pl.col("price") / pl.col("_previous_trade_price")).log())
            .otherwise(0.0)
            .alias("log_trade_return_1")
        )
    )

    expressions: list[pl.Expr] = []
    windows = sorted(set((*config.trade_windows, config.intensity_window)))
    for window in windows:
        signed = (
            pl.col("signed_trade_quantity")
            .rolling_sum(window_size=window, min_samples=1)
            .over(_GROUP)
        )
        volume = pl.col("quantity").rolling_sum(window_size=window, min_samples=1).over(_GROUP)
        count = (
            pl.col("_trade_observation").rolling_sum(window_size=window, min_samples=1).over(_GROUP)
        )
        window_start = pl.coalesce(
            [
                pl.col("decision_ts_ns").shift(window - 1).over(_GROUP),
                pl.col("_segment_start_ts_ns"),
            ]
        )
        elapsed_seconds = (pl.col("decision_ts_ns") - window_start) / 1_000_000_000.0
        expressions.extend(
            [
                signed.alias(f"signed_trade_volume_w{window}"),
                volume.alias(f"trade_volume_w{window}"),
                pl.when(volume > 0)
                .then(signed / volume)
                .otherwise(None)
                .alias(f"trade_imbalance_w{window}"),
                count.cast(pl.Float64).alias(f"trade_count_w{window}"),
                pl.when(elapsed_seconds > 0)
                .then(count / elapsed_seconds)
                .otherwise(0.0)
                .cast(pl.Float64)
                .alias(f"trade_intensity_w{window}"),
            ]
        )

    volatility_window = config.volatility_window
    expressions.append(
        pl.col("log_trade_return_1")
        .pow(2)
        .rolling_sum(window_size=volatility_window, min_samples=1)
        .over(_GROUP)
        .sqrt()
        .alias(f"realized_volatility_w{volatility_window}")
    )
    warmup = max((*config.trade_windows, config.intensity_window, config.volatility_window))
    return (
        prepared.with_columns(expressions)
        .with_columns(
            (pl.col("history_trades") >= warmup).alias("feature_ready"),
            pl.col("decision_ts_ns").alias("max_feature_source_ts_ns"),
            pl.col("decision_trade_id").alias("max_feature_source_trade_id"),
        )
        .drop("_trade_observation", "_previous_trade_price", "_segment_start_ts_ns")
        .sort(["decision_ts_ns", "symbol", "continuity_id", "decision_trade_id"])
    )


def add_future_trade_labels(frame: pl.DataFrame, horizon_trades: int) -> pl.DataFrame:
    """Attach strictly subsequent trade-price labels without crossing a gap."""

    if horizon_trades < 1:
        raise ResearchDataError("trade label horizon must be at least one trade")
    _require_columns(
        frame,
        frozenset(
            {
                "symbol",
                "continuity_id",
                "decision_ts_ns",
                "decision_trade_id",
                "price",
            }
        ),
        "trade feature frame",
    )
    labeled = (
        frame.sort([*_GROUP, "decision_trade_id"])
        .with_columns(
            pl.col("price").shift(-horizon_trades).over(_GROUP).alias("_target_trade_price"),
            pl.col("decision_ts_ns")
            .shift(-horizon_trades)
            .over(_GROUP)
            .alias("_target_trade_ts_ns"),
            pl.col("decision_trade_id")
            .shift(-horizon_trades)
            .over(_GROUP)
            .alias("_target_trade_id"),
            pl.col("continuity_id")
            .shift(-horizon_trades)
            .over(_GROUP)
            .alias("_target_continuity_id"),
        )
        .with_columns(
            (
                pl.col("_target_trade_price").is_null()
                | (pl.col("_target_trade_id") <= pl.col("decision_trade_id"))
                | (pl.col("_target_trade_ts_ns") < pl.col("decision_ts_ns"))
                | (
                    (pl.col("_target_trade_ts_ns") == pl.col("decision_ts_ns"))
                    & (pl.col("_target_trade_id") <= pl.col("decision_trade_id"))
                )
                | (pl.col("_target_continuity_id") != pl.col("continuity_id"))
            ).alias("right_censored")
        )
        .with_columns(
            pl.when(~pl.col("right_censored"))
            .then((pl.col("_target_trade_price") / pl.col("price")).log())
            .otherwise(None)
            .alias("future_trade_return"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_trade_price"))
            .otherwise(None)
            .alias("future_trade_price"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_trade_ts_ns"))
            .otherwise(None)
            .alias("label_information_end_ts_ns"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_trade_id"))
            .otherwise(None)
            .alias("label_information_end_trade_id"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_continuity_id"))
            .otherwise(None)
            .alias("label_continuity_id"),
            pl.lit(horizon_trades, dtype=pl.Int64).alias("label_horizon_trades"),
            pl.col("decision_ts_ns").alias("label_start_ts_ns"),
            pl.col("decision_trade_id").alias("label_start_trade_id"),
        )
        .with_columns(
            pl.when(pl.col("future_trade_return").is_null())
            .then(None)
            .when(pl.col("future_trade_return") > 0)
            .then(1)
            .when(pl.col("future_trade_return") < 0)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("future_trade_direction"),
            pl.when(pl.col("future_trade_return").is_null())
            .then(None)
            .otherwise((pl.col("future_trade_return") > 0).cast(pl.Int8))
            .alias("future_trade_up"),
        )
        .drop(
            "_target_trade_price",
            "_target_trade_ts_ns",
            "_target_trade_id",
            "_target_continuity_id",
        )
        .sort(["decision_ts_ns", "symbol", "continuity_id", "decision_trade_id"])
    )
    validate_trade_only_temporal_contract(labeled)
    return labeled


def build_trade_only_research_frame(trades: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """Build the complete causal trade-only feature and future-label frame."""

    features = build_trade_only_features(trades, config)
    return add_future_trade_labels(features, config.label_horizon_events)


def validate_trade_only_temporal_contract(frame: pl.DataFrame) -> TemporalAudit:
    """Fail closed when trade feature lineage or label timing is noncausal."""

    required = frozenset(
        {
            "symbol",
            "continuity_id",
            "decision_ts_ns",
            "decision_trade_id",
            "feature_cutoff_ts_ns",
            "feature_continuity_id",
            "max_feature_source_ts_ns",
            "max_feature_source_trade_id",
            "right_censored",
            "future_trade_return",
            "future_trade_price",
            "future_trade_direction",
            "future_trade_up",
            "label_start_ts_ns",
            "label_start_trade_id",
            "label_information_end_ts_ns",
            "label_information_end_trade_id",
            "label_continuity_id",
        }
    )
    _require_columns(frame, required, "trade-only research frame")
    future_feature = frame.filter(
        (pl.col("feature_continuity_id") != pl.col("continuity_id"))
        | (pl.col("max_feature_source_ts_ns") > pl.col("feature_cutoff_ts_ns"))
        | (
            (pl.col("max_feature_source_ts_ns") == pl.col("feature_cutoff_ts_ns"))
            & (pl.col("max_feature_source_trade_id") > pl.col("decision_trade_id"))
        )
    )
    if not future_feature.is_empty():
        raise TemporalLeakageError("trade feature lineage extends beyond its decision cutoff")

    uncensored = ~pl.col("right_censored")
    invalid_label = frame.filter(
        (pl.col("label_start_ts_ns") != pl.col("decision_ts_ns"))
        | (pl.col("label_start_trade_id") != pl.col("decision_trade_id"))
        | (
            uncensored
            & (
                pl.col("future_trade_return").is_null()
                | pl.col("future_trade_price").is_null()
                | pl.col("future_trade_direction").is_null()
                | pl.col("future_trade_up").is_null()
                | pl.col("label_information_end_ts_ns").is_null()
                | pl.col("label_information_end_trade_id").is_null()
                | (pl.col("label_continuity_id") != pl.col("continuity_id"))
                | (pl.col("label_information_end_trade_id") <= pl.col("decision_trade_id"))
                | (pl.col("label_information_end_ts_ns") < pl.col("decision_ts_ns"))
                | (
                    (pl.col("label_information_end_ts_ns") == pl.col("decision_ts_ns"))
                    & (pl.col("label_information_end_trade_id") <= pl.col("decision_trade_id"))
                )
            )
        )
        | (
            pl.col("right_censored")
            & (
                pl.col("future_trade_return").is_not_null()
                | pl.col("future_trade_price").is_not_null()
                | pl.col("future_trade_direction").is_not_null()
                | pl.col("future_trade_up").is_not_null()
                | pl.col("label_information_end_ts_ns").is_not_null()
                | pl.col("label_information_end_trade_id").is_not_null()
                | pl.col("label_continuity_id").is_not_null()
            )
        )
    )
    if not invalid_label.is_empty():
        raise TemporalLeakageError(
            "trade labels are not strictly future, gap-local, and censor-safe"
        )

    censored_rows = frame.filter(pl.col("right_censored")).height
    return TemporalAudit(
        rows=frame.height,
        labeled_rows=frame.height - censored_rows,
        right_censored_rows=censored_rows,
        continuity_segments=frame.select("symbol", "continuity_id").unique().height,
    )


__all__ = [
    "add_future_trade_labels",
    "build_trade_only_features",
    "build_trade_only_research_frame",
    "validate_trade_only_temporal_contract",
]
