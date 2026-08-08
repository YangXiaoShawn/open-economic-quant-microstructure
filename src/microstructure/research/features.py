"""Leakage-safe L1 and trade-flow research features and labels.

The normalized ``available_ts_ns`` column is the information-set clock.  Book
observations at a decision are observable at that decision, while trades from a
separate archive stream are joined strictly before it unless a future adapter
can prove a shared ordering.  Every rolling operation is scoped by
``continuity_id`` so sequence gaps cannot contaminate a new book segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from microstructure.config import FeatureConfig


class ResearchDataError(ValueError):
    """Raised when normalized inputs do not satisfy the research contract."""


class TemporalLeakageError(ResearchDataError):
    """Raised when feature lineage or label timing reaches beyond its cutoff."""


@dataclass(frozen=True, slots=True)
class TemporalAudit:
    """Summary returned after validating a supervised research frame."""

    rows: int
    labeled_rows: int
    right_censored_rows: int
    continuity_segments: int


BOOK_REQUIRED_COLUMNS = frozenset(
    {
        "symbol",
        "event_ts_ns",
        "available_ts_ns",
        "continuity_id",
        "sequence_end",
        "is_valid",
        "best_bid",
        "best_ask",
        "bid_quantity",
        "ask_quantity",
    }
)
TRADE_REQUIRED_COLUMNS = frozenset(
    {
        "symbol",
        "trade_id",
        "available_ts_ns",
        "quantity",
        "aggressor_side",
    }
)

_STATIC_MODEL_FEATURES = (
    "spread_bps",
    "depth_total_l1",
    "depth_total_l5",
    "depth_total_l10",
    "queue_imbalance_l1",
    "queue_imbalance_l5",
    "queue_imbalance_l10",
    "microprice_deviation_bps",
    "ofi_l1",
    "log_mid_return_1",
    "realized_price_impact_bps_1",
    "spread_recovery_bps_1",
    "depth_recovery_l1_1",
)
_MODEL_FEATURE_PREFIXES = (
    "cancellation_intensity_w",
    "ofi_w",
    "signed_trade_volume_w",
    "trade_volume_w",
    "trade_count_w",
    "trade_intensity_w",
    "realized_volatility_w",
)

DEPTH_DELTA_REQUIRED_COLUMNS = frozenset(
    {
        "venue",
        "symbol",
        "event_ts_ns",
        "available_ts_ns",
        "continuity_id",
        "first_update_id",
        "last_update_id",
        "bids",
        "asks",
    }
)


def _require_columns(frame: pl.DataFrame, required: frozenset[str], table: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ResearchDataError(f"{table} is missing required columns: {missing}")


def _assert_normalized_books(books: pl.DataFrame) -> None:
    _require_columns(books, BOOK_REQUIRED_COLUMNS, "book observations")
    if books.is_empty():
        raise ResearchDataError("book observations must not be empty")

    invalid = books.filter(
        (pl.col("available_ts_ns") < pl.col("event_ts_ns"))
        | (pl.col("best_bid") <= 0)
        | (pl.col("best_ask") <= 0)
        | (pl.col("bid_quantity") < 0)
        | (pl.col("ask_quantity") < 0)
        | (pl.col("is_valid") & (pl.col("best_bid") >= pl.col("best_ask")))
    )
    if not invalid.is_empty():
        raise ResearchDataError(
            "book observations contain impossible timing, price, quantity, or valid crossed-book rows"
        )

    duplicates = (
        books.group_by(["symbol", "continuity_id", "sequence_end"]).len().filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ResearchDataError("book sequence keys must be unique within each continuity segment")

    ordered = books.sort(["symbol", "continuity_id", "sequence_end"])
    backwards = ordered.filter(
        pl.col("available_ts_ns")
        < pl.col("available_ts_ns").shift(1).over(["symbol", "continuity_id"])
    )
    if not backwards.is_empty():
        raise ResearchDataError(
            "available_ts_ns must be nondecreasing by sequence within each continuity segment"
        )


def _prepare_trades(trades: pl.DataFrame | None) -> pl.DataFrame | None:
    if trades is None or trades.is_empty():
        return None
    _require_columns(trades, TRADE_REQUIRED_COLUMNS, "trades")
    invalid = trades.filter(
        (pl.col("quantity") < 0)
        | (~pl.col("aggressor_side").str.to_lowercase().is_in(["buy", "sell"]))
    )
    if not invalid.is_empty():
        raise ResearchDataError("trades require nonnegative quantity and buy/sell aggressor_side")

    group = ["symbol", "continuity_id"] if "continuity_id" in trades.columns else ["symbol"]
    identity_columns = ["symbol"]
    if "continuity_id" in trades.columns:
        identity_columns.append("continuity_id")
    return (
        trades.sort([*group, "available_ts_ns", "trade_id"])
        .with_columns(
            pl.when(pl.col("aggressor_side").str.to_lowercase() == "buy")
            .then(1.0)
            .otherwise(-1.0)
            .alias("_trade_sign"),
        )
        .with_columns(
            (pl.col("quantity") * pl.col("_trade_sign")).alias("_signed_quantity"),
            pl.col("quantity").alias("_absolute_quantity"),
            pl.lit(1, dtype=pl.Int64).alias("_trade_observation"),
        )
        .with_columns(
            pl.col("_signed_quantity").cum_sum().over(group).alias("_cum_signed"),
            pl.col("_absolute_quantity").cum_sum().over(group).alias("_cum_volume"),
            pl.col("_trade_observation").cum_sum().over(group).alias("_cum_count"),
        )
        .select(
            *identity_columns,
            pl.col("available_ts_ns").alias("trade_feature_max_source_ts_ns"),
            "_cum_signed",
            "_cum_volume",
            "_cum_count",
        )
    )


def build_cancellation_intensity_features(
    depth_deltas: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (20, 100),
) -> pl.DataFrame:
    """Build causal cancellation-intensity proxies from observable L2 deletes.

    Binance diff-depth encodes a zero quantity as removal of that price level.
    Those deletes are directly observable; an update to a smaller nonzero
    quantity is not classified here because its cancellation/execution split is
    not identifiable from the delta alone. Windows are event-count windows,
    include the current available delta, and reset at every continuity epoch.
    """

    _require_columns(depth_deltas, DEPTH_DELTA_REQUIRED_COLUMNS, "depth deltas")
    if depth_deltas.is_empty():
        raise ResearchDataError("depth deltas must not be empty")
    normalized_windows = tuple(sorted(set(windows)))
    if not normalized_windows or any(isinstance(window, bool) or window < 1 for window in windows):
        raise ResearchDataError("cancellation windows must be positive integers")

    group = ["venue", "symbol", "continuity_id"]
    ordered = depth_deltas.sort([*group, "last_update_id", "first_update_id"])
    invalid = ordered.filter(
        pl.col("continuity_id").is_null()
        | (pl.col("available_ts_ns") < pl.col("event_ts_ns"))
        | (pl.col("first_update_id") > pl.col("last_update_id"))
    )
    if not invalid.is_empty():
        raise ResearchDataError(
            "depth deltas require a continuity ID, observable availability, and valid ranges"
        )
    duplicates = (
        ordered.group_by([*group, "first_update_id", "last_update_id"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        raise ResearchDataError("depth-delta sequence ranges must be unique within an epoch")

    prior_end = pl.col("last_update_id").shift(1).over(group)
    prior_available = pl.col("available_ts_ns").shift(1).over(group)
    broken = ordered.filter(
        prior_end.is_not_null()
        & (
            (pl.col("last_update_id") <= prior_end)
            | (pl.col("first_update_id") > prior_end + 1)
            | (pl.col("available_ts_ns") < prior_available)
        )
    )
    if not broken.is_empty():
        raise ResearchDataError(
            "depth deltas contain a stale/gapped sequence or reversing availability clock"
        )

    per_event = ordered.with_columns(
        (
            pl.col("bids").list.eval(pl.element().struct.field("quantity_lots") == 0).list.sum()
            + pl.col("asks").list.eval(pl.element().struct.field("quantity_lots") == 0).list.sum()
        )
        .fill_null(0)
        .cast(pl.Int64)
        .alias("cancellation_deletes_current"),
        (pl.col("bids").list.len() + pl.col("asks").list.len())
        .cast(pl.Int64)
        .alias("depth_updates_current"),
        pl.col("available_ts_ns").alias("decision_ts_ns"),
        pl.col("available_ts_ns").alias("feature_cutoff_ts_ns"),
        pl.col("available_ts_ns").alias("max_feature_source_ts_ns"),
        pl.col("last_update_id").alias("decision_sequence"),
        pl.col("last_update_id").alias("max_feature_source_sequence"),
        pl.lit("zero_quantity_level_deletes_only").alias("cancellation_observation_policy"),
        pl.lit(False).alias("nonzero_reduction_classified_as_cancellation"),
    )
    expressions: list[pl.Expr] = []
    for window in normalized_windows:
        deletes = (
            pl.col("cancellation_deletes_current")
            .rolling_sum(window_size=window, min_samples=1)
            .over(group)
        )
        updates = (
            pl.col("depth_updates_current")
            .rolling_sum(window_size=window, min_samples=1)
            .over(group)
        )
        expressions.extend(
            [
                deletes.alias(f"cancellation_deletes_w{window}"),
                updates.alias(f"depth_updates_w{window}"),
                pl.when(updates > 0)
                .then(deletes / updates)
                .otherwise(0.0)
                .cast(pl.Float64)
                .alias(f"cancellation_intensity_w{window}"),
            ]
        )
    return per_event.with_columns(expressions).sort(
        ["decision_ts_ns", "venue", "symbol", "continuity_id", "decision_sequence"]
    )


def _join_trade_history(books: pl.DataFrame, trades: pl.DataFrame | None) -> pl.DataFrame:
    if trades is None:
        return books.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("trade_feature_max_source_ts_ns"),
            pl.lit(0.0).alias("_cum_signed"),
            pl.lit(0.0).alias("_cum_volume"),
            pl.lit(0, dtype=pl.Int64).alias("_cum_count"),
        )

    # Equal timestamps are deliberately excluded.  Archive trade and book
    # streams do not share a provable exchange-wide sequence.  When trades carry
    # segment provenance, both their cumulative state and join stay gap-local.
    group = ["symbol", "continuity_id"] if "continuity_id" in trades.columns else ["symbol"]
    joined = books.sort(["feature_cutoff_ts_ns", "symbol", "sequence_end"]).join_asof(
        trades.sort(["trade_feature_max_source_ts_ns", *group]),
        left_on="feature_cutoff_ts_ns",
        right_on="trade_feature_max_source_ts_ns",
        by=group,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    return joined.with_columns(
        pl.col("_cum_signed").fill_null(0.0),
        pl.col("_cum_volume").fill_null(0.0),
        pl.col("_cum_count").fill_null(0),
    )


def _join_cancellation_history(
    features: pl.DataFrame,
    depth_deltas: pl.DataFrame | None,
    config: FeatureConfig,
) -> pl.DataFrame:
    if depth_deltas is None:
        return features
    if "venue" not in features.columns:
        raise ResearchDataError(
            "book observations require venue when cancellation features are requested"
        )
    cancellation = build_cancellation_intensity_features(
        depth_deltas,
        windows=config.trade_windows,
    )
    feature_columns = [
        name
        for name in cancellation.columns
        if name.startswith("cancellation_deletes_w")
        or name.startswith("depth_updates_w")
        or name.startswith("cancellation_intensity_w")
    ]
    keyed = cancellation.select(
        "venue",
        "symbol",
        "continuity_id",
        pl.col("decision_sequence").alias("sequence_end"),
        pl.col("max_feature_source_ts_ns").alias("cancellation_feature_max_source_ts_ns"),
        pl.col("max_feature_source_sequence").alias("cancellation_feature_max_source_sequence"),
        "cancellation_observation_policy",
        "nonzero_reduction_classified_as_cancellation",
        *feature_columns,
    )
    keys = ["venue", "symbol", "continuity_id", "sequence_end"]
    joined = features.join(keyed, on=keys, how="left", validate="1:1")
    missing = joined.filter(pl.col("cancellation_feature_max_source_ts_ns").is_null())
    if not missing.is_empty():
        raise ResearchDataError(
            "supplied depth deltas do not cover every research-eligible book observation"
        )
    future = joined.filter(
        (pl.col("cancellation_feature_max_source_ts_ns") > pl.col("feature_cutoff_ts_ns"))
        | (
            (pl.col("cancellation_feature_max_source_ts_ns") == pl.col("feature_cutoff_ts_ns"))
            & (pl.col("cancellation_feature_max_source_sequence") > pl.col("decision_sequence"))
        )
    )
    if not future.is_empty():
        raise TemporalLeakageError("cancellation feature lineage extends beyond its decision")
    return joined


def build_l1_trade_features(
    book_observations: pl.DataFrame,
    trades: pl.DataFrame | None,
    config: FeatureConfig,
) -> pl.DataFrame:
    """Build causal event-time features from normalized L1 states and trades.

    Invalid book rows are not repaired or used as decisions.  The caller keeps
    the normalized/quality tables as the audit record; this returned table is a
    research-eligible view containing valid states only.
    """

    _assert_normalized_books(book_observations)
    if trades is not None and not trades.is_empty() and "continuity_id" not in trades.columns:
        multi_segment_symbols = (
            book_observations.group_by("symbol")
            .agg(pl.col("continuity_id").n_unique().alias("_continuity_count"))
            .filter(pl.col("_continuity_count") > 1)
            .get_column("symbol")
            .to_list()
        )
        if multi_segment_symbols:
            raise ResearchDataError(
                "trades require continuity_id when book history contains multiple continuity "
                f"segments for symbols: {sorted(str(value) for value in multi_segment_symbols)}"
            )
    prepared_trades = _prepare_trades(trades)
    group = ["symbol", "continuity_id"]
    depth_expressions: list[pl.Expr] = []
    for level in (5, 10):
        bid_column = f"depth_bid_{level}"
        ask_column = f"depth_ask_{level}"
        presence = (
            bid_column in book_observations.columns,
            ask_column in book_observations.columns,
        )
        if presence[0] != presence[1]:
            raise ResearchDataError(
                f"book observations must supply both {bid_column} and {ask_column}"
            )
        if all(presence):
            total = pl.col(bid_column) + pl.col(ask_column)
            depth_expressions.extend(
                [
                    total.alias(f"depth_total_l{level}"),
                    pl.when(total > 0)
                    .then((pl.col(bid_column) - pl.col(ask_column)) / total)
                    .otherwise(None)
                    .alias(f"queue_imbalance_l{level}"),
                ]
            )

    books = (
        book_observations.filter(pl.col("is_valid"))
        .sort(["symbol", "continuity_id", "sequence_end"])
        .with_columns(
            pl.col("event_ts_ns").alias("market_event_ts_ns"),
            pl.col("available_ts_ns").alias("decision_ts_ns"),
            pl.col("available_ts_ns").alias("feature_cutoff_ts_ns"),
            pl.col("sequence_end").alias("decision_sequence"),
            ((pl.col("best_bid") + pl.col("best_ask")) / 2.0).alias("mid_price"),
            (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
            (pl.col("bid_quantity") + pl.col("ask_quantity")).alias("depth_total_l1"),
            *depth_expressions,
        )
        .with_columns(
            pl.col("best_bid").shift(1).over(group).alias("_previous_bid"),
            pl.col("best_ask").shift(1).over(group).alias("_previous_ask"),
            pl.col("bid_quantity").shift(1).over(group).alias("_previous_bid_quantity"),
            pl.col("ask_quantity").shift(1).over(group).alias("_previous_ask_quantity"),
            pl.col("mid_price").shift(1).over(group).alias("_previous_mid"),
            pl.col("spread").shift(1).over(group).alias("_previous_spread"),
            pl.col("depth_total_l1").shift(1).over(group).alias("_previous_depth_l1"),
            pl.col("sequence_end").cum_count().over(group).alias("history_events"),
        )
        .with_columns(
            (10_000.0 * pl.col("spread") / pl.col("mid_price")).alias("spread_bps"),
            pl.when(pl.col("depth_total_l1") > 0)
            .then((pl.col("bid_quantity") - pl.col("ask_quantity")) / pl.col("depth_total_l1"))
            .otherwise(None)
            .alias("queue_imbalance_l1"),
            pl.when(pl.col("depth_total_l1") > 0)
            .then(
                (
                    pl.col("best_ask") * pl.col("bid_quantity")
                    + pl.col("best_bid") * pl.col("ask_quantity")
                )
                / pl.col("depth_total_l1")
            )
            .otherwise(None)
            .alias("causal_microprice"),
            pl.when(pl.col("_previous_mid").is_not_null())
            .then((pl.col("mid_price") / pl.col("_previous_mid")).log())
            .otherwise(0.0)
            .alias("log_mid_return_1"),
            pl.when(pl.col("_previous_mid").is_not_null())
            .then(10_000.0 * (pl.col("mid_price") / pl.col("_previous_mid") - 1.0))
            .otherwise(0.0)
            .alias("realized_price_impact_bps_1"),
            pl.when(pl.col("_previous_spread").is_not_null())
            .then(10_000.0 * (pl.col("_previous_spread") - pl.col("spread")) / pl.col("mid_price"))
            .otherwise(0.0)
            .alias("spread_recovery_bps_1"),
            pl.when(pl.col("_previous_depth_l1").is_not_null())
            .then(pl.col("depth_total_l1") - pl.col("_previous_depth_l1"))
            .otherwise(0.0)
            .alias("depth_recovery_l1_1"),
            pl.when(pl.col("_previous_bid").is_null())
            .then(0.0)
            .otherwise(
                pl.when(pl.col("best_bid") >= pl.col("_previous_bid"))
                .then(pl.col("bid_quantity"))
                .otherwise(0.0)
                - pl.when(pl.col("best_bid") <= pl.col("_previous_bid"))
                .then(pl.col("_previous_bid_quantity"))
                .otherwise(0.0)
                - pl.when(pl.col("best_ask") <= pl.col("_previous_ask"))
                .then(pl.col("ask_quantity"))
                .otherwise(0.0)
                + pl.when(pl.col("best_ask") >= pl.col("_previous_ask"))
                .then(pl.col("_previous_ask_quantity"))
                .otherwise(0.0)
            )
            .alias("ofi_l1"),
        )
        .with_columns(
            (
                10_000.0 * (pl.col("causal_microprice") - pl.col("mid_price")) / pl.col("mid_price")
            ).alias("microprice_deviation_bps"),
            pl.col("feature_cutoff_ts_ns").first().over(group).alias("_segment_start_ts_ns"),
        )
    )

    joined = _join_trade_history(books, prepared_trades).sort(
        ["symbol", "continuity_id", "sequence_end"]
    )
    joined = joined.with_columns(
        pl.when(pl.col("trade_feature_max_source_ts_ns") >= pl.col("_segment_start_ts_ns"))
        .then(pl.col("trade_feature_max_source_ts_ns"))
        .otherwise(None)
        .alias("trade_feature_max_source_ts_ns"),
        pl.col("_cum_signed").first().over(group).alias("_segment_base_signed"),
        pl.col("_cum_volume").first().over(group).alias("_segment_base_volume"),
        pl.col("_cum_count").first().over(group).alias("_segment_base_count"),
    )

    rolling_expressions: list[pl.Expr] = []
    trade_feature_windows = sorted(set((*config.trade_windows, config.intensity_window)))
    for window in trade_feature_windows:
        lag_signed = pl.col("_cum_signed").shift(window).over(group)
        lag_volume = pl.col("_cum_volume").shift(window).over(group)
        lag_count = pl.col("_cum_count").shift(window).over(group)
        lag_time = pl.col("feature_cutoff_ts_ns").shift(window).over(group)
        signed = pl.col("_cum_signed") - pl.coalesce([lag_signed, pl.col("_segment_base_signed")])
        volume = pl.col("_cum_volume") - pl.coalesce([lag_volume, pl.col("_segment_base_volume")])
        count = pl.col("_cum_count") - pl.coalesce([lag_count, pl.col("_segment_base_count")])
        elapsed_seconds = (
            pl.col("feature_cutoff_ts_ns") - pl.coalesce([lag_time, pl.col("_segment_start_ts_ns")])
        ) / 1_000_000_000.0
        rolling_expressions.extend(
            [
                signed.alias(f"signed_trade_volume_w{window}"),
                volume.alias(f"trade_volume_w{window}"),
                count.cast(pl.Float64).alias(f"trade_count_w{window}"),
                pl.when(elapsed_seconds > 0)
                .then(count / elapsed_seconds)
                .otherwise(0.0)
                .cast(pl.Float64)
                .alias(f"trade_intensity_w{window}"),
                pl.col("ofi_l1")
                .rolling_sum(window_size=window, min_samples=1)
                .over(group)
                .alias(f"ofi_w{window}"),
            ]
        )

    volatility_window = config.volatility_window
    rolling_expressions.append(
        pl.col("log_mid_return_1")
        .pow(2)
        .rolling_sum(window_size=volatility_window, min_samples=1)
        .over(group)
        .sqrt()
        .alias(f"realized_volatility_w{volatility_window}")
    )
    warmup = max((*config.trade_windows, config.volatility_window, config.intensity_window))

    return (
        joined.with_columns(rolling_expressions)
        .with_columns(
            (pl.col("history_events") >= warmup).alias("feature_ready"),
            pl.col("feature_cutoff_ts_ns").alias("max_feature_source_ts_ns"),
            pl.col("decision_sequence").alias("max_feature_source_sequence"),
        )
        .drop(
            "_previous_bid",
            "_previous_ask",
            "_previous_bid_quantity",
            "_previous_ask_quantity",
            "_previous_mid",
            "_previous_spread",
            "_previous_depth_l1",
            "_segment_start_ts_ns",
            "_cum_signed",
            "_cum_volume",
            "_cum_count",
            "_segment_base_signed",
            "_segment_base_volume",
            "_segment_base_count",
        )
        .sort(["decision_ts_ns", "symbol", "decision_sequence"])
    )


def add_future_event_labels(frame: pl.DataFrame, horizon_events: int) -> pl.DataFrame:
    """Attach strictly subsequent mid-return/direction labels within each segment."""

    if horizon_events < 1:
        raise ResearchDataError("label horizon must be at least one event")
    _require_columns(
        frame,
        frozenset(
            {
                "symbol",
                "continuity_id",
                "decision_ts_ns",
                "decision_sequence",
                "mid_price",
            }
        ),
        "feature frame",
    )
    group = ["symbol", "continuity_id"]
    labeled = (
        frame.sort(["symbol", "continuity_id", "decision_sequence"])
        .with_columns(
            pl.col("mid_price").shift(-horizon_events).over(group).alias("_target_mid"),
            pl.col("decision_ts_ns").shift(-horizon_events).over(group).alias("_target_ts_ns"),
            pl.col("decision_sequence")
            .shift(-horizon_events)
            .over(group)
            .alias("_target_sequence"),
            pl.col("continuity_id")
            .shift(-horizon_events)
            .over(group)
            .alias("_target_continuity_id"),
        )
        .with_columns(
            (
                pl.col("_target_mid").is_null()
                | (pl.col("_target_sequence") <= pl.col("decision_sequence"))
                | (
                    (pl.col("_target_ts_ns") < pl.col("decision_ts_ns"))
                    | (
                        (pl.col("_target_ts_ns") == pl.col("decision_ts_ns"))
                        & (pl.col("_target_sequence") <= pl.col("decision_sequence"))
                    )
                )
            ).alias("right_censored")
        )
        .with_columns(
            pl.when(~pl.col("right_censored"))
            .then((pl.col("_target_mid") / pl.col("mid_price")).log())
            .otherwise(None)
            .alias("future_mid_return"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_ts_ns"))
            .otherwise(None)
            .alias("label_information_end_ts_ns"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_sequence"))
            .otherwise(None)
            .alias("label_information_end_sequence"),
            pl.when(~pl.col("right_censored"))
            .then(pl.col("_target_continuity_id"))
            .otherwise(None)
            .alias("label_continuity_id"),
            pl.lit(horizon_events, dtype=pl.Int64).alias("label_horizon_events"),
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
        .drop("_target_mid", "_target_ts_ns", "_target_sequence", "_target_continuity_id")
        .sort(["decision_ts_ns", "symbol", "decision_sequence"])
    )
    validate_temporal_contract(labeled)
    return labeled


def build_research_frame(
    book_observations: pl.DataFrame,
    trades: pl.DataFrame | None,
    config: FeatureConfig,
    *,
    depth_deltas: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the complete causal feature/strictly-future-label event frame."""

    features = build_research_features(
        book_observations,
        trades,
        config,
        depth_deltas=depth_deltas,
    )
    return add_future_event_labels(features, config.label_horizon_events)


def build_research_features(
    book_observations: pl.DataFrame,
    trades: pl.DataFrame | None,
    config: FeatureConfig,
    *,
    depth_deltas: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build causal features without opening any future-label horizon.

    Separating this stage lets a caller attach several predeclared event- and
    clock-time labels to the exact same information set.  Cancellation inputs
    remain continuity-local and are joined at the current observable update.
    """

    features = build_l1_trade_features(book_observations, trades, config)
    return _join_cancellation_history(features, depth_deltas, config)


def validate_temporal_contract(frame: pl.DataFrame) -> TemporalAudit:
    """Fail closed when feature lineage or labels violate event-time ordering."""

    required = frozenset(
        {
            "symbol",
            "continuity_id",
            "decision_ts_ns",
            "decision_sequence",
            "feature_cutoff_ts_ns",
            "max_feature_source_ts_ns",
            "max_feature_source_sequence",
            "trade_feature_max_source_ts_ns",
            "right_censored",
            "future_mid_return",
            "future_mid_direction",
            "future_mid_up",
            "label_information_end_ts_ns",
            "label_information_end_sequence",
            "label_continuity_id",
        }
    )
    _require_columns(frame, required, "research frame")

    future_feature = frame.filter(
        (pl.col("max_feature_source_ts_ns") > pl.col("feature_cutoff_ts_ns"))
        | (
            (pl.col("max_feature_source_ts_ns") == pl.col("feature_cutoff_ts_ns"))
            & (pl.col("max_feature_source_sequence") > pl.col("decision_sequence"))
        )
        | (
            pl.col("trade_feature_max_source_ts_ns").is_not_null()
            & (pl.col("trade_feature_max_source_ts_ns") >= pl.col("decision_ts_ns"))
        )
    )
    if not future_feature.is_empty():
        raise TemporalLeakageError("feature lineage extends beyond its decision cutoff")

    if {
        "cancellation_feature_max_source_ts_ns",
        "cancellation_feature_max_source_sequence",
    }.issubset(frame.columns):
        future_cancellation = frame.filter(
            (pl.col("cancellation_feature_max_source_ts_ns") > pl.col("feature_cutoff_ts_ns"))
            | (
                (pl.col("cancellation_feature_max_source_ts_ns") == pl.col("feature_cutoff_ts_ns"))
                & (pl.col("cancellation_feature_max_source_sequence") > pl.col("decision_sequence"))
            )
        )
        if not future_cancellation.is_empty():
            raise TemporalLeakageError(
                "cancellation feature lineage extends beyond its decision cutoff"
            )

    uncensored = ~pl.col("right_censored")
    invalid_label = frame.filter(
        (
            uncensored
            & (
                pl.col("label_information_end_ts_ns").is_null()
                | pl.col("label_information_end_sequence").is_null()
                | (pl.col("label_continuity_id") != pl.col("continuity_id"))
                | (pl.col("label_information_end_ts_ns") < pl.col("decision_ts_ns"))
                | (
                    (pl.col("label_information_end_ts_ns") == pl.col("decision_ts_ns"))
                    & (pl.col("label_information_end_sequence") <= pl.col("decision_sequence"))
                )
            )
        )
        | (
            pl.col("right_censored")
            & (
                pl.col("future_mid_return").is_not_null()
                | pl.col("future_mid_direction").is_not_null()
                | pl.col("future_mid_up").is_not_null()
                | pl.col("label_information_end_ts_ns").is_not_null()
                | pl.col("label_information_end_sequence").is_not_null()
            )
        )
    )
    if not invalid_label.is_empty():
        raise TemporalLeakageError("labels are not strictly future, gap-local, and censor-safe")

    censored_rows = frame.filter(pl.col("right_censored")).height
    return TemporalAudit(
        rows=frame.height,
        labeled_rows=frame.height - censored_rows,
        right_censored_rows=censored_rows,
        continuity_segments=frame.select("symbol", "continuity_id").unique().height,
    )


def model_feature_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    """Return the explicit leakage-safe allowlist present in ``frame``."""

    selected = [name for name in _STATIC_MODEL_FEATURES if name in frame.columns]
    selected.extend(
        name
        for name in frame.columns
        if name.startswith(_MODEL_FEATURE_PREFIXES) and name not in selected
    )
    if not selected:
        raise ResearchDataError("research frame contains no recognized model features")
    return tuple(selected)


__all__ = [
    "ResearchDataError",
    "TemporalAudit",
    "TemporalLeakageError",
    "add_future_event_labels",
    "build_cancellation_intensity_features",
    "build_l1_trade_features",
    "build_research_features",
    "build_research_frame",
    "model_feature_columns",
    "validate_temporal_contract",
]
