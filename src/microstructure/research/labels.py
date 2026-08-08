"""Auxiliary, explicitly future-dependent research labels.

These builders never produce model features.  Each output records the end of
the information interval, right-censoring, and the assumptions needed to
interpret the label.  Clock joins and fill evidence are strict with respect to
the decision/activation time and never cross ``continuity_id`` boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import polars as pl


class AuxiliaryLabelError(ValueError):
    """Raised when an auxiliary label cannot satisfy its temporal contract."""


Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class ClockTimeLabelSpec:
    """Clock horizons and admissible delay to the first observed target state."""

    horizons_ns: tuple[int, ...]
    max_target_staleness_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.horizons_ns or any(value <= 0 for value in self.horizons_ns):
            raise AuxiliaryLabelError("clock horizons must be nonempty and strictly positive")
        if self.max_target_staleness_ns is not None and self.max_target_staleness_ns < 0:
            raise AuxiliaryLabelError("max target staleness must be nonnegative")


@dataclass(frozen=True, slots=True)
class LimitFillAssumptions:
    """Observable-input queue proxy for a hypothetical best-quote limit order."""

    side: Side
    horizon_ns: int
    order_quantity: float
    queue_ahead_fraction: float = 1.0
    activation_latency_ns: int = 0

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise AuxiliaryLabelError("limit side must be buy or sell")
        if self.horizon_ns <= 0:
            raise AuxiliaryLabelError("fill horizon must be positive")
        if not math.isfinite(self.order_quantity) or self.order_quantity <= 0:
            raise AuxiliaryLabelError("order quantity must be finite and positive")
        if not 0.0 <= self.queue_ahead_fraction <= 1.0:
            raise AuxiliaryLabelError("queue_ahead_fraction must be in [0, 1]")
        if self.activation_latency_ns < 0:
            raise AuxiliaryLabelError("activation latency must be nonnegative")


@dataclass(frozen=True, slots=True)
class AdverseSelectionSpec:
    """Post-fill markout horizons and target-state staleness bound."""

    horizons_ns: tuple[int, ...]
    max_target_staleness_ns: int | None = None

    def __post_init__(self) -> None:
        ClockTimeLabelSpec(self.horizons_ns, self.max_target_staleness_ns)


_GROUP_COLUMNS = ("symbol", "continuity_id")


def _require(
    frame: pl.DataFrame,
    columns: Sequence[str],
    table: str,
    *,
    allow_empty: bool = False,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise AuxiliaryLabelError(f"{table} is missing required columns: {missing}")
    if frame.is_empty() and not allow_empty:
        raise AuxiliaryLabelError(f"{table} must not be empty")


def _valid_books(
    books: pl.DataFrame,
    *,
    time_column: str,
    mid_column: str,
) -> pl.DataFrame:
    _require(books, (*_GROUP_COLUMNS, time_column, mid_column), "book states")
    result = books.filter(pl.col("is_valid")) if "is_valid" in books.columns else books
    if result.is_empty():
        raise AuxiliaryLabelError("no valid book states are available")
    invalid = result.filter(
        pl.col("continuity_id").is_null()
        | pl.col(time_column).is_null()
        | pl.col(mid_column).is_null()
        | (pl.col(mid_column) <= 0)
    )
    if not invalid.is_empty():
        raise AuxiliaryLabelError("valid book states require segment, time, and positive mid")
    return result.sort([time_column, *_GROUP_COLUMNS])


def _forward_mid_join(
    decisions: pl.DataFrame,
    books: pl.DataFrame,
    *,
    decision_time_column: str,
    book_time_column: str,
    mid_column: str,
    target_column: str,
    book_identity_column: str | None = None,
) -> pl.DataFrame:
    right_columns: list[pl.Expr | str] = [
        *_GROUP_COLUMNS,
        pl.col(book_time_column).alias("_matched_target_ts_ns"),
        pl.col(mid_column).alias("_matched_target_mid"),
    ]
    if book_identity_column is not None:
        right_columns.append(pl.col(book_identity_column).alias("_matched_target_identity"))
    right = books.select(right_columns).sort(["_matched_target_ts_ns", *_GROUP_COLUMNS])
    return (
        decisions.sort([target_column, *_GROUP_COLUMNS])
        .join_asof(
            right,
            left_on=target_column,
            right_on="_matched_target_ts_ns",
            by=list(_GROUP_COLUMNS),
            strategy="forward",
            allow_exact_matches=True,
            check_sortedness=False,
        )
        .sort([decision_time_column, *_GROUP_COLUMNS])
    )


def build_clock_time_mid_labels(
    decisions: pl.DataFrame,
    spec: ClockTimeLabelSpec,
    *,
    book_states: pl.DataFrame | None = None,
    decision_time_column: str = "decision_ts_ns",
    mid_column: str = "mid_price",
    book_time_column: str | None = None,
    book_mid_column: str | None = None,
    book_identity_column: str | None = None,
) -> pl.DataFrame:
    """Build long-form clock-time return/direction labels within book segments."""

    _require(decisions, (*_GROUP_COLUMNS, decision_time_column, mid_column), "decisions")
    invalid_decisions = decisions.filter(
        pl.col("continuity_id").is_null()
        | pl.col(decision_time_column).is_null()
        | pl.col(mid_column).is_null()
        | (pl.col(mid_column) <= 0)
    )
    if invalid_decisions.height:
        raise AuxiliaryLabelError("decisions require segment, time, and positive current mid")
    target_states = decisions if book_states is None else book_states
    target_time = book_time_column or decision_time_column
    target_mid = book_mid_column or mid_column
    books = _valid_books(target_states, time_column=target_time, mid_column=target_mid)
    if book_identity_column is not None:
        _require(books, (book_identity_column,), "clock-time target states")
        if books.get_column(book_identity_column).null_count():
            raise AuxiliaryLabelError("clock-time target identity must not contain nulls")
    eligible_decisions = (
        decisions.filter(pl.col("is_valid"))
        if book_states is None and "is_valid" in decisions.columns
        else decisions
    )
    outputs: list[pl.DataFrame] = []
    for horizon_ns in sorted(set(spec.horizons_ns)):
        with_target = eligible_decisions.with_columns(
            (pl.col(decision_time_column) + horizon_ns).alias("clock_target_ts_ns")
        )
        joined = _forward_mid_join(
            with_target,
            books,
            decision_time_column=decision_time_column,
            book_time_column=target_time,
            mid_column=target_mid,
            target_column="clock_target_ts_ns",
            book_identity_column=book_identity_column,
        ).with_columns(
            (pl.col("_matched_target_ts_ns") - pl.col("clock_target_ts_ns")).alias(
                "clock_target_staleness_ns"
            )
        )
        censored = pl.col("_matched_target_ts_ns").is_null()
        if spec.max_target_staleness_ns is not None:
            censored = censored | (
                pl.col("clock_target_staleness_ns") > spec.max_target_staleness_ns
            )
        labeled = (
            joined.with_columns(
                censored.alias("clock_right_censored"),
                pl.lit(horizon_ns, dtype=pl.Int64).alias("clock_horizon_ns"),
                pl.lit("clock_time_mid_return", dtype=pl.String).alias("clock_label_kind"),
                pl.lit(
                    "first valid state at or after t+h in the same continuity segment",
                    dtype=pl.String,
                ).alias("clock_label_assumption"),
                pl.lit(True).alias("clock_label_is_descriptive"),
                pl.col(decision_time_column).alias("clock_label_start_ts_ns"),
                pl.when(pl.col("_matched_target_ts_ns").is_null())
                .then(pl.lit("no_same_segment_future_state"))
                .when(censored)
                .then(pl.lit("target_state_too_stale"))
                .otherwise(None)
                .alias("clock_censor_reason"),
            )
            .with_columns(
                pl.when(~pl.col("clock_right_censored"))
                .then((pl.col("_matched_target_mid") / pl.col(mid_column)).log())
                .otherwise(None)
                .alias("clock_future_mid_return"),
                pl.when(~pl.col("clock_right_censored"))
                .then(pl.col("_matched_target_mid"))
                .otherwise(None)
                .alias("clock_target_mid_price"),
                pl.when(~pl.col("clock_right_censored"))
                .then(pl.col("_matched_target_ts_ns"))
                .otherwise(None)
                .alias("clock_label_information_end_ts_ns"),
                pl.when(~pl.col("clock_right_censored"))
                .then(pl.col("clock_target_staleness_ns"))
                .otherwise(None)
                .alias("clock_observed_target_staleness_ns"),
            )
            .with_columns(
                pl.when(pl.col("clock_future_mid_return").is_null())
                .then(None)
                .when(pl.col("clock_future_mid_return") > 0)
                .then(1)
                .when(pl.col("clock_future_mid_return") < 0)
                .then(-1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("clock_future_mid_direction")
            )
        )
        if book_identity_column is not None:
            labeled = labeled.with_columns(
                pl.when(~pl.col("clock_right_censored"))
                .then(pl.col("_matched_target_identity"))
                .otherwise(None)
                .alias("clock_label_information_end_identity")
            )
        outputs.append(labeled)
    drop_columns = ["_matched_target_ts_ns", "_matched_target_mid"]
    if book_identity_column is not None:
        drop_columns.append("_matched_target_identity")
    return (
        pl.concat(outputs, how="diagonal_relaxed")
        .drop(*drop_columns)
        .sort([decision_time_column, "clock_horizon_ns", *_GROUP_COLUMNS])
    )


def _side_expression(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column).cast(pl.String).str.to_lowercase().is_in(["buy", "1", "1.0"]))
        .then(1.0)
        .when(pl.col(column).cast(pl.String).str.to_lowercase().is_in(["sell", "-1", "-1.0"]))
        .then(-1.0)
        .otherwise(None)
    )


def add_event_time_price_impact_labels(
    frame: pl.DataFrame,
    *,
    side_column: str,
    return_column: str = "future_mid_return",
    information_end_column: str = "label_information_end_ts_ns",
    right_censored_column: str = "right_censored",
) -> pl.DataFrame:
    """Side-sign a strictly future event-time return without changing its horizon."""

    _require(
        frame,
        (
            side_column,
            return_column,
            information_end_column,
            right_censored_column,
            "label_horizon_events",
        ),
        "event-time impact frame",
    )
    result = frame.with_columns(_side_expression(side_column).alias("event_impact_side_sign"))
    if result.filter(pl.col("event_impact_side_sign").is_null()).height:
        raise AuxiliaryLabelError("event-time impact side must be buy/sell or +1/-1")
    invalid_timing = result.filter(
        (~pl.col(right_censored_column) & pl.col(information_end_column).is_null())
        | (pl.col(right_censored_column) & pl.col(return_column).is_not_null())
    )
    if invalid_timing.height:
        raise AuxiliaryLabelError("event-time return censoring and information end disagree")
    return result.with_columns(
        pl.when(~pl.col(right_censored_column))
        .then(10_000.0 * pl.col("event_impact_side_sign") * pl.col(return_column))
        .otherwise(None)
        .alias("event_time_signed_price_impact_bps"),
        pl.col(information_end_column).alias("event_impact_label_information_end_ts_ns"),
        pl.col(right_censored_column).alias("event_impact_right_censored"),
        pl.lit("event_time_signed_price_impact").alias("event_impact_label_kind"),
        pl.lit(True).alias("event_impact_label_is_descriptive"),
        pl.lit("aggressor-side sign times strictly future same-segment log-mid return").alias(
            "event_impact_label_assumption"
        ),
    )


def build_clock_time_price_impact_labels(
    decisions: pl.DataFrame,
    spec: ClockTimeLabelSpec,
    *,
    book_states: pl.DataFrame | None = None,
    side_column: str = "trade_sign",
    decision_time_column: str = "decision_ts_ns",
    mid_column: str = "mid_price",
    book_time_column: str | None = None,
    book_mid_column: str | None = None,
) -> pl.DataFrame:
    """Add aggressor-signed clock-time price impact to mid-return labels."""

    _require(decisions, (side_column,), "impact decisions")
    result = build_clock_time_mid_labels(
        decisions,
        spec,
        book_states=book_states,
        decision_time_column=decision_time_column,
        mid_column=mid_column,
        book_time_column=book_time_column,
        book_mid_column=book_mid_column,
    ).with_columns(_side_expression(side_column).alias("clock_impact_side_sign"))
    if result.filter(pl.col("clock_impact_side_sign").is_null()).height:
        raise AuxiliaryLabelError("impact side must be buy/sell or +1/-1")
    return result.with_columns(
        pl.when(~pl.col("clock_right_censored"))
        .then(10_000.0 * pl.col("clock_impact_side_sign") * pl.col("clock_future_mid_return"))
        .otherwise(None)
        .alias("clock_signed_price_impact_bps"),
        pl.lit("clock_time_signed_price_impact").alias("clock_label_kind"),
    )


def build_hypothetical_limit_fill_labels(
    book_states: pl.DataFrame,
    trades: pl.DataFrame,
    assumptions: LimitFillAssumptions,
    *,
    decision_time_column: str = "decision_ts_ns",
) -> pl.DataFrame:
    """Label a conservative best-quote fill proxy from subsequent trade prints.

    The proxy ignores cancellations and hidden liquidity.  Opposing prints
    strictly after activation deplete a fixed fraction of displayed queue before
    reaching the hypothetical order.  Historical printed quantity is never
    reused within one candidate order, but separate decision labels remain
    counterfactual scenarios rather than simultaneously live orders.
    """

    _require(
        book_states,
        (
            *_GROUP_COLUMNS,
            decision_time_column,
            "best_bid",
            "best_ask",
            "bid_quantity",
            "ask_quantity",
        ),
        "limit-fill book states",
    )
    _require(
        trades,
        (*_GROUP_COLUMNS, "available_ts_ns", "price", "quantity", "aggressor_side"),
        "limit-fill trades",
        allow_empty=True,
    )
    invalid_trades = trades.filter(
        (pl.col("quantity") < 0)
        | (~pl.col("aggressor_side").str.to_lowercase().is_in(["buy", "sell"]))
    )
    if invalid_trades.height:
        raise AuxiliaryLabelError("trades require nonnegative quantity and buy/sell side")

    trade_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in trades.sort("available_ts_ns").iter_rows(named=True):
        key = (str(row["symbol"]), str(row["continuity_id"]))
        trade_groups.setdefault(key, []).append(row)

    valid_for_coverage = (
        book_states.filter(pl.col("is_valid")) if "is_valid" in book_states.columns else book_states
    )
    coverage: dict[tuple[str, str], int] = {}
    for row in (
        valid_for_coverage.group_by(list(_GROUP_COLUMNS))
        .agg(pl.col(decision_time_column).max().alias("_coverage_end"))
        .iter_rows(named=True)
    ):
        coverage[(str(row["symbol"]), str(row["continuity_id"]))] = int(row["_coverage_end"])

    output: list[dict[str, object]] = []
    for row in book_states.sort([decision_time_column, *_GROUP_COLUMNS]).iter_rows(named=True):
        symbol = str(row["symbol"])
        continuity_id = str(row["continuity_id"])
        decision_ts = int(row[decision_time_column])
        activation_ts = decision_ts + assumptions.activation_latency_ns
        deadline_ts = activation_ts + assumptions.horizon_ns
        valid_decision = bool(row.get("is_valid", True))
        limit_price = float(row["best_bid"] if assumptions.side == "buy" else row["best_ask"])
        displayed_quantity = float(
            row["bid_quantity"] if assumptions.side == "buy" else row["ask_quantity"]
        )
        queue_ahead = displayed_quantity * assumptions.queue_ahead_fraction
        cumulative_executable = 0.0
        full_fill_ts: int | None = None
        if valid_decision:
            for trade in trade_groups.get((symbol, continuity_id), []):
                trade_ts = cast(int, trade["available_ts_ns"])
                if trade_ts <= activation_ts:
                    continue
                if trade_ts > deadline_ts:
                    break
                trade_side = str(trade["aggressor_side"]).lower()
                trade_price = cast(float, trade["price"])
                marketable = (
                    assumptions.side == "buy"
                    and trade_side == "sell"
                    and trade_price <= limit_price
                ) or (
                    assumptions.side == "sell"
                    and trade_side == "buy"
                    and trade_price >= limit_price
                )
                if not marketable:
                    continue
                cumulative_executable += cast(float, trade["quantity"])
                if cumulative_executable >= queue_ahead + assumptions.order_quantity:
                    full_fill_ts = trade_ts
                    break

        observed_fill = min(
            assumptions.order_quantity, max(0.0, cumulative_executable - queue_ahead)
        )
        segment_covers_deadline = coverage.get((symbol, continuity_id), -1) >= deadline_ts
        full_fill = full_fill_ts is not None
        right_censored = (not valid_decision) or (not full_fill and not segment_covers_deadline)
        information_end = (
            full_fill_ts
            if full_fill
            else deadline_ts
            if segment_covers_deadline and valid_decision
            else None
        )
        base = {
            name: row[name]
            for name in (
                "sample_id",
                "symbol",
                "continuity_id",
                decision_time_column,
                "decision_sequence",
            )
            if name in row
        }
        output.append(
            {
                **base,
                "limit_label_kind": "hypothetical_best_quote_fill_proxy",
                "limit_label_is_descriptive": True,
                "limit_label_assumption": (
                    "trade-print depletion of fixed displayed queue; cancellations, hidden liquidity, "
                    "and endogenous impact ignored"
                ),
                "limit_side": assumptions.side,
                "limit_price": limit_price,
                "limit_order_quantity": assumptions.order_quantity,
                "limit_initial_displayed_quantity": displayed_quantity,
                "limit_initial_queue_ahead": queue_ahead,
                "limit_queue_ahead_fraction": assumptions.queue_ahead_fraction,
                "limit_activation_latency_ns": assumptions.activation_latency_ns,
                "limit_horizon_ns": assumptions.horizon_ns,
                "limit_activation_ts_ns": activation_ts,
                "limit_deadline_ts_ns": deadline_ts,
                "limit_trade_evidence_required": True,
                "limit_equal_time_ordering": "trade_at_activation_excluded",
                "limit_cancellation_handling": "ignored_no_order_level_attribution",
                "limit_observed_executable_quantity": cumulative_executable,
                "limit_observed_fill_before_censoring": observed_fill,
                "limit_right_censored": right_censored,
                "limit_censor_reason": (
                    "invalid_decision_book"
                    if not valid_decision
                    else "segment_ends_before_horizon"
                    if right_censored
                    else None
                ),
                "limit_fill_quantity": None if right_censored else observed_fill,
                "limit_fill_fraction": (
                    None if right_censored else observed_fill / assumptions.order_quantity
                ),
                "limit_full_fill": None if right_censored else full_fill,
                "limit_full_fill_ts_ns": full_fill_ts,
                "limit_label_start_ts_ns": decision_ts,
                "limit_label_information_end_ts_ns": information_end,
            }
        )
    return pl.DataFrame(output).sort([decision_time_column, *_GROUP_COLUMNS])


def build_post_fill_adverse_selection_labels(
    fills: pl.DataFrame,
    book_states: pl.DataFrame,
    spec: AdverseSelectionSpec,
    *,
    fill_time_column: str = "fill_ts_ns",
    fill_price_column: str = "fill_price",
    side_column: str = "side",
    book_time_column: str = "decision_ts_ns",
    mid_column: str = "mid_price",
) -> pl.DataFrame:
    """Build side-aware post-fill markout and adverse-selection labels."""

    _require(
        fills,
        (*_GROUP_COLUMNS, fill_time_column, fill_price_column, side_column),
        "fills",
    )
    books = _valid_books(book_states, time_column=book_time_column, mid_column=mid_column)
    prepared_fills = fills.with_columns(_side_expression(side_column).alias("_fill_side_sign"))
    invalid = prepared_fills.filter(
        pl.col("_fill_side_sign").is_null()
        | (pl.col(fill_price_column) <= 0)
        | pl.col("continuity_id").is_null()
    )
    if invalid.height:
        raise AuxiliaryLabelError("fills require buy/sell side, positive price, and continuity_id")

    outputs: list[pl.DataFrame] = []
    for horizon_ns in sorted(set(spec.horizons_ns)):
        with_target = prepared_fills.with_columns(
            (pl.col(fill_time_column) + horizon_ns).alias("adverse_target_ts_ns")
        )
        joined = _forward_mid_join(
            with_target,
            books,
            decision_time_column=fill_time_column,
            book_time_column=book_time_column,
            mid_column=mid_column,
            target_column="adverse_target_ts_ns",
        ).with_columns(
            (pl.col("_matched_target_ts_ns") - pl.col("adverse_target_ts_ns")).alias(
                "adverse_target_staleness_ns"
            )
        )
        censored = pl.col("_matched_target_ts_ns").is_null()
        if spec.max_target_staleness_ns is not None:
            censored = censored | (
                pl.col("adverse_target_staleness_ns") > spec.max_target_staleness_ns
            )
        outputs.append(
            joined.with_columns(
                censored.alias("adverse_right_censored"),
                pl.lit(horizon_ns, dtype=pl.Int64).alias("adverse_horizon_ns"),
                pl.lit("post_fill_adverse_selection").alias("adverse_label_kind"),
                pl.lit(True).alias("adverse_label_is_descriptive"),
                pl.lit(
                    "first valid same-segment mid at or after fill+h; exogenous book and no own impact"
                ).alias("adverse_label_assumption"),
                pl.col(fill_time_column).alias("adverse_label_start_ts_ns"),
                pl.when(pl.col("_matched_target_ts_ns").is_null())
                .then(pl.lit("no_same_segment_future_state"))
                .when(censored)
                .then(pl.lit("target_state_too_stale"))
                .otherwise(None)
                .alias("adverse_censor_reason"),
            )
            .with_columns(
                pl.when(~pl.col("adverse_right_censored"))
                .then(pl.col("_matched_target_mid"))
                .otherwise(None)
                .alias("adverse_target_mid_price"),
                pl.when(~pl.col("adverse_right_censored"))
                .then(pl.col("_matched_target_ts_ns"))
                .otherwise(None)
                .alias("adverse_label_information_end_ts_ns"),
                pl.when(~pl.col("adverse_right_censored"))
                .then(
                    10_000.0
                    * pl.col("_fill_side_sign")
                    * (pl.col("_matched_target_mid") - pl.col(fill_price_column))
                    / pl.col(fill_price_column)
                )
                .otherwise(None)
                .alias("post_fill_markout_bps"),
            )
            .with_columns(
                (-pl.col("post_fill_markout_bps")).alias("adverse_selection_bps"),
                pl.when(pl.col("post_fill_markout_bps").is_null())
                .then(None)
                .otherwise(pl.col("post_fill_markout_bps") < 0)
                .alias("adverse_selection_indicator"),
            )
        )
    return (
        pl.concat(outputs, how="diagonal_relaxed")
        .drop("_matched_target_ts_ns", "_matched_target_mid", "_fill_side_sign")
        .sort([fill_time_column, "adverse_horizon_ns", *_GROUP_COLUMNS])
    )


__all__ = [
    "AdverseSelectionSpec",
    "AuxiliaryLabelError",
    "ClockTimeLabelSpec",
    "LimitFillAssumptions",
    "add_event_time_price_impact_labels",
    "build_clock_time_mid_labels",
    "build_clock_time_price_impact_labels",
    "build_hypothetical_limit_fill_labels",
    "build_post_fill_adverse_selection_labels",
]
