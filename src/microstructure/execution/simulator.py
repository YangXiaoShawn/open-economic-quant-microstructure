"""Deterministic event-driven simulation with explicit fill assumptions.

The simulator consumes historical market states and out-of-sample predictions.
It never connects to an exchange and it deliberately leaves the replay exogenous:
capacity sweeps expose, but cannot identify, endogenous market impact.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from microstructure.config import ExecutionConfig

OrderType = Literal["market", "limit"]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Serialized tables and accounting metrics from one simulation scenario."""

    orders: pl.DataFrame
    fills: pl.DataFrame
    positions: pl.DataFrame
    metrics: dict[str, Any]
    assumptions: dict[str, Any]


@dataclass(slots=True)
class _ActiveLimit:
    order: dict[str, Any]
    remaining_quantity: float
    queue_ahead: float
    cancel_effective_position: int


def _number(row: dict[str, Any], *names: str, default: float | None = None) -> float:
    for name in names:
        value = row.get(name)
        if value is not None:
            return float(value)
    if default is not None:
        return default
    raise ValueError(f"market event is missing all required columns: {names}")


def _object_float(value: object) -> float:
    return float(cast(Any, value))


def _object_int(value: object) -> int:
    return int(cast(Any, value))


def _integer(row: dict[str, Any], *names: str) -> int:
    for name in names:
        value = row.get(name)
        if value is not None:
            return int(value)
    raise ValueError(f"row is missing all required identifier columns: {names}")


def _mid(row: dict[str, Any]) -> float:
    direct = row.get("mid_price")
    if direct is not None:
        return float(direct)
    bid = _number(row, "best_bid", "bid_price_1")
    ask = _number(row, "best_ask", "ask_price_1")
    return 0.5 * (bid + ask)


def _prediction_probability(row: dict[str, Any]) -> float:
    return _number(row, "probability", "probability_up", "prediction", "y_probability")


def _event_identifier(row: dict[str, Any]) -> int:
    return _integer(
        row,
        "decision_sequence",
        "sequence_end",
        "sample_id",
        "event_index",
        "event_id",
        "row_id",
    )


def _timestamp(row: dict[str, Any]) -> int:
    return _integer(row, "event_ts_ns", "decision_ts_ns", "available_ts_ns")


def _continuity(row: dict[str, Any]) -> str:
    value = row.get("continuity_id")
    return str(value) if value is not None else "__NO_CONTINUITY_ID__"


def _trade_side(row: dict[str, Any]) -> int:
    value = row.get("trade_side", row.get("aggressor_side", row.get("side", 0)))
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "buy":
            return 1
        if normalized == "sell":
            return -1
        return 0
    return _object_int(value) if value is not None else 0


def _displayed_depth(row: dict[str, Any], side: int) -> float:
    if side == 1:
        return _number(
            row, "ask_depth_1", "depth_ask_1", "ask_quantity_1", "ask_quantity", default=0.0
        )
    return _number(row, "bid_depth_1", "depth_bid_1", "bid_quantity_1", "bid_quantity", default=0.0)


def _seeded_uniform(seed: int, order_id: object, event_id: int) -> float:
    payload = f"{seed}:{order_id}:{event_id}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64)


def _round_quantity(quantity: float, row: dict[str, Any]) -> float:
    lot_size = _number(row, "lot_size", default=0.0)
    if lot_size <= 0:
        return quantity
    return math.floor((quantity + lot_size * 1e-12) / lot_size) * lot_size


def _round_price_adversely(price: float, side: int, row: dict[str, Any]) -> float:
    tick_size = _number(row, "tick_size", default=0.0)
    if tick_size <= 0:
        return price
    scaled = price / tick_size
    ticks = math.ceil(scaled - 1e-12) if side == 1 else math.floor(scaled + 1e-12)
    return ticks * tick_size


def _empty_frame(columns: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(schema=columns)


def _portfolio_max_drawdown(equity_rows: list[dict[str, Any]]) -> float:
    """Return zero-capital marked net-equity peak-to-trough drawdown."""
    latest_equity: dict[str, float] = {}
    peak = 0.0
    maximum_drawdown = 0.0
    ordered = sorted(
        equity_rows,
        key=lambda item: (int(item["event_ts_ns"]), int(item["observation_id"])),
    )
    index = 0
    while index < len(ordered):
        timestamp = int(ordered[index]["event_ts_ns"])
        while index < len(ordered) and int(ordered[index]["event_ts_ns"]) == timestamp:
            row = ordered[index]
            latest_equity[str(row["symbol"])] = float(row["net_equity"])
            index += 1
        portfolio_equity = sum(latest_equity.values())
        peak = max(peak, portfolio_equity)
        maximum_drawdown = max(maximum_drawdown, peak - portfolio_equity)
    return maximum_drawdown


def simulate_predictions(
    events: pl.DataFrame,
    predictions: pl.DataFrame,
    config: ExecutionConfig,
    *,
    order_type: OrderType = "market",
    size_multiplier: float = 1.0,
    seed: int = 0,
    markout_events: int = 20,
) -> SimulationResult:
    """Replay a prediction policy with fees, latency, depth, and inventory limits.

    Event rows must contain a symbol, event/sample identifier, timestamp, top of
    book, displayed L1 depth, and—when passive fills are requested—signed trade
    quantity. A positive trade side is buyer initiated. Predictions must be
    explicitly out of sample when an ``is_oos`` column is present.
    """
    if order_type not in {"market", "limit"}:
        raise ValueError(f"unsupported order_type: {order_type}")
    if not math.isfinite(size_multiplier) or size_multiplier <= 0:
        raise ValueError("size_multiplier must be finite and positive")
    if markout_events < 0:
        raise ValueError("markout_events must be nonnegative")
    if config.decision_latency_events < 0 or config.order_latency_events < 0:
        raise ValueError("decision and order latency must be nonnegative")
    if not math.isfinite(config.queue_ahead_units) or config.queue_ahead_units < 0:
        raise ValueError("queue_ahead_units must be finite and nonnegative")
    if events.is_empty():
        raise ValueError("events must not be empty")
    required_prediction_columns = {"symbol", "is_oos", "split"}
    missing_prediction_columns = sorted(required_prediction_columns.difference(predictions.columns))
    if missing_prediction_columns:
        raise ValueError(
            "execution predictions require explicit OOS provenance columns: "
            f"{missing_prediction_columns}"
        )
    if not bool(predictions.get_column("is_oos").fill_null(False).all()):
        raise ValueError("execution simulation rejects non-OOS predictions")
    invalid_splits = predictions.filter(
        ~pl.col("split")
        .cast(pl.String)
        .str.to_lowercase()
        .is_in(["test", "final_test", "holdout", "held_out"])
    )
    if not invalid_splits.is_empty():
        raise ValueError("execution simulation accepts held-out test predictions only")

    event_rows = events.to_dicts()
    prediction_rows = predictions.to_dicts()
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        by_symbol[str(row["symbol"])].append(row)
    for rows in by_symbol.values():
        rows.sort(key=lambda item: (_timestamp(item), _event_identifier(item)))
        identifiers = [_event_identifier(row) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("market event identifiers must be unique within each symbol")

    prediction_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        prediction_by_symbol[str(row["symbol"])].append(row)
        probability = _prediction_probability(row)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("prediction probabilities must be finite and lie in [0, 1]")
    unknown_symbols = sorted(set(prediction_by_symbol).difference(by_symbol))
    if unknown_symbols:
        raise ValueError(f"predictions reference symbols absent from events: {unknown_symbols}")

    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    gross_cash: dict[str, float] = defaultdict(float)
    positions: dict[str, float] = defaultdict(float)
    fees_by_symbol: dict[str, float] = defaultdict(float)
    turnover_by_symbol: dict[str, float] = defaultdict(float)
    maximum_inventory_by_symbol: dict[str, float] = defaultdict(float)
    final_mid: dict[str, float] = {}
    total_fees = 0.0
    maker_fees = 0.0
    taker_fees = 0.0
    turnover_notional = 0.0
    maximum_inventory = 0.0
    forced_liquidation_quantity = 0.0
    unliquidated_quantity = 0.0
    next_order_id = 0
    next_fill_id = 0
    next_equity_observation_id = 0

    def record_equity(row: dict[str, Any], symbol: str) -> None:
        nonlocal next_equity_observation_id
        next_equity_observation_id += 1
        gross_equity = gross_cash[symbol] + positions[symbol] * _mid(row)
        equity_rows.append(
            {
                "observation_id": next_equity_observation_id,
                "symbol": symbol,
                "event_ts_ns": _timestamp(row),
                "net_equity": gross_equity - fees_by_symbol[symbol],
            }
        )

    def record_fill(
        *,
        order: dict[str, Any],
        row: dict[str, Any],
        rows: list[dict[str, Any]],
        position_index: int,
        price: float,
        quantity: float,
        liquidity: Literal["maker", "taker"],
        queue_ahead_before: float | None,
        forced_liquidation: bool = False,
    ) -> None:
        nonlocal total_fees, maker_fees, taker_fees, turnover_notional
        nonlocal maximum_inventory, next_fill_id
        symbol = str(order["symbol"])
        side = _object_int(order["side"])
        notional = price * quantity
        fee_rate_bps = config.maker_fee_bps if liquidity == "maker" else config.taker_fee_bps
        fee = notional * fee_rate_bps / 10_000.0
        gross_cash[symbol] -= side * notional
        positions[symbol] += side * quantity
        total_fees += fee
        turnover_notional += notional
        fees_by_symbol[symbol] += fee
        turnover_by_symbol[symbol] += notional
        if liquidity == "maker":
            maker_fees += fee
        else:
            taker_fees += fee
        maximum_inventory = max(maximum_inventory, abs(positions[symbol]))
        maximum_inventory_by_symbol[symbol] = max(
            maximum_inventory_by_symbol[symbol], abs(positions[symbol])
        )

        requested_markout_position = position_index + markout_events
        markout_available = (
            not forced_liquidation
            and requested_markout_position < len(rows)
            and _continuity(rows[requested_markout_position]) == _continuity(row)
        )
        markout_mid = _mid(rows[requested_markout_position]) if markout_available else None
        post_fill_markout_bps = (
            side * (markout_mid - price) / price * 10_000.0 if markout_mid is not None else None
        )
        decision_mid = _object_float(order["decision_mid_price"])
        arrival_cost_bps = side * (price - decision_mid) / decision_mid * 10_000.0
        next_fill_id += 1
        fill_rows.append(
            {
                "fill_id": next_fill_id,
                "order_id": order["order_id"],
                "sample_id": order["sample_id"],
                "symbol": symbol,
                "event_id": _event_identifier(row),
                "event_ts_ns": _timestamp(row),
                "side": side,
                "price": price,
                "quantity": quantity,
                "notional": notional,
                "liquidity": liquidity,
                "fee": fee,
                "queue_ahead_before": queue_ahead_before,
                "arrival_cost_bps": arrival_cost_bps,
                "post_fill_markout_bps": post_fill_markout_bps,
                "adverse_selection_bps": (
                    -post_fill_markout_bps if post_fill_markout_bps is not None else None
                ),
                "requested_markout_events": markout_events,
                "markout_available": markout_available,
                "forced_liquidation": forced_liquidation,
            }
        )
        gross_equity = gross_cash[symbol] + positions[symbol] * _mid(row)
        position_rows.append(
            {
                "fill_id": next_fill_id,
                "symbol": symbol,
                "event_ts_ns": _timestamp(row),
                "position_units": positions[symbol],
                "gross_cash": gross_cash[symbol],
                "mid_price": _mid(row),
                "gross_equity": gross_equity,
                "symbol_cumulative_fees": fees_by_symbol[symbol],
                "net_equity": gross_equity - fees_by_symbol[symbol],
            }
        )
        record_equity(row, symbol)

    latency_events = config.decision_latency_events + config.order_latency_events
    for symbol in sorted(by_symbol):
        rows = by_symbol[symbol]
        identifier_to_position = {_event_identifier(row): index for index, row in enumerate(rows)}
        scheduled: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for prediction in prediction_by_symbol.get(symbol, []):
            probability = _prediction_probability(prediction)
            if probability >= config.signal_threshold:
                side = 1
            elif probability <= 1.0 - config.signal_threshold:
                side = -1
            else:
                continue
            sample_id = _event_identifier(prediction)
            decision_position = identifier_to_position.get(sample_id)
            if decision_position is None:
                raise ValueError(f"prediction sample {sample_id} has no matching {symbol} event")
            arrival_position = decision_position + latency_events
            next_order_id += 1
            decision_row = rows[decision_position]
            order = {
                "order_id": next_order_id,
                "sample_id": sample_id,
                "symbol": symbol,
                "decision_event_ts_ns": _timestamp(decision_row),
                "decision_position": decision_position,
                "decision_continuity_id": _continuity(decision_row),
                "arrival_position": arrival_position,
                "arrival_event_ts_ns": (
                    _timestamp(rows[arrival_position]) if arrival_position < len(rows) else None
                ),
                "side": side,
                "order_type": order_type,
                "requested_quantity": config.order_size_units * size_multiplier,
                "accepted_quantity": 0.0,
                "filled_quantity": 0.0,
                "decision_mid_price": _mid(decision_row),
                "limit_price": None,
                "status": "scheduled" if arrival_position < len(rows) else "expired_before_arrival",
                "rejection_reason": None,
            }
            order_rows.append(order)
            if arrival_position < len(rows):
                scheduled[arrival_position].append(order)

        active_limits: list[_ActiveLimit] = []
        previous_continuity: str | None = None
        for position_index, row in enumerate(rows):
            current_continuity = _continuity(row)
            if previous_continuity is not None and current_continuity != previous_continuity:
                for active in active_limits:
                    active.order["status"] = (
                        "partially_filled_continuity_gap"
                        if _object_float(active.order["filled_quantity"]) > 0
                        else "canceled_continuity_gap"
                    )
                active_limits.clear()
            previous_continuity = current_continuity

            trade_side = _trade_side(row)
            trade_quantity = abs(_number(row, "trade_quantity", "quantity", default=0.0))
            trade_price = _number(row, "trade_price", "price", default=_mid(row))
            remaining_trade_quantity = trade_quantity

            # Exchange events at this position execute before newly arriving orders.
            for active in list(active_limits):
                order = active.order
                crosses = (
                    _object_int(order["side"]) == 1
                    and trade_side < 0
                    and trade_price <= _object_float(order["limit_price"])
                ) or (
                    _object_int(order["side"]) == -1
                    and trade_side > 0
                    and trade_price >= _object_float(order["limit_price"])
                )
                if crosses and remaining_trade_quantity > 0:
                    queue_before = active.queue_ahead
                    queue_consumed = min(active.queue_ahead, remaining_trade_quantity)
                    active.queue_ahead -= queue_consumed
                    remaining_trade_quantity -= queue_consumed
                    if (
                        remaining_trade_quantity > 0
                        and _seeded_uniform(seed, order["order_id"], _event_identifier(row))
                        <= config.limit_fill_base_probability
                    ):
                        inventory_room = (
                            config.max_position_units - positions[symbol]
                            if _object_int(order["side"]) == 1
                            else config.max_position_units + positions[symbol]
                        )
                        fill_quantity = min(
                            active.remaining_quantity,
                            remaining_trade_quantity,
                            max(0.0, inventory_room),
                        )
                        if fill_quantity > 0:
                            record_fill(
                                order=order,
                                row=row,
                                rows=rows,
                                position_index=position_index,
                                price=_object_float(order["limit_price"]),
                                quantity=fill_quantity,
                                liquidity="maker",
                                queue_ahead_before=queue_before,
                            )
                            active.remaining_quantity -= fill_quantity
                            remaining_trade_quantity -= fill_quantity
                            order["filled_quantity"] = (
                                _object_float(order["filled_quantity"]) + fill_quantity
                            )
                            order["status"] = (
                                (
                                    "filled"
                                    if _object_float(order["filled_quantity"])
                                    >= _object_float(order["requested_quantity"]) - 1e-12
                                    else "inventory_clipped_filled"
                                )
                                if active.remaining_quantity <= 1e-12
                                else "partially_filled"
                            )
                    elif remaining_trade_quantity > 0 and active.queue_ahead <= 1e-12:
                        # A failed fill draw represents unobserved queue ahead; the
                        # printed volume cannot also fill another simulated order.
                        remaining_trade_quantity = 0.0
                if active.remaining_quantity <= 1e-12:
                    active_limits.remove(active)
                    continue
                if position_index >= active.cancel_effective_position:
                    order["status"] = (
                        "partially_filled_expired"
                        if _object_float(order["filled_quantity"]) > 0
                        else "expired"
                    )
                    active_limits.remove(active)

            for order in scheduled.get(position_index, []):
                if _continuity(row) != str(order["decision_continuity_id"]):
                    order["status"] = "canceled_continuity_gap"
                    order["rejection_reason"] = "arrival_crossed_continuity_gap"
                    continue
                side = _object_int(order["side"])
                inventory_room = (
                    config.max_position_units - positions[symbol]
                    if side == 1
                    else config.max_position_units + positions[symbol]
                )
                requested = _object_float(order["requested_quantity"])
                accepted = _round_quantity(min(requested, max(0.0, inventory_room)), row)
                order["accepted_quantity"] = accepted
                if accepted <= 1e-12:
                    order["status"] = "rejected"
                    order["rejection_reason"] = "inventory_limit"
                    continue

                if order_type == "market":
                    price = (
                        _number(row, "best_ask", "ask_price_1")
                        if side == 1
                        else _number(row, "best_bid", "bid_price_1")
                    )
                    displayed = _displayed_depth(row, side)
                    fill_quantity = _round_quantity(min(accepted, max(0.0, displayed)), row)
                    if fill_quantity <= 1e-12:
                        order["status"] = "canceled_no_liquidity"
                        continue
                    depth_ratio = fill_quantity / max(displayed, 1e-12)
                    slippage_bps = config.slippage_bps_per_unit * depth_ratio
                    price = _round_price_adversely(
                        price * (1.0 + side * slippage_bps / 10_000.0), side, row
                    )
                    record_fill(
                        order=order,
                        row=row,
                        rows=rows,
                        position_index=position_index,
                        price=price,
                        quantity=fill_quantity,
                        liquidity="taker",
                        queue_ahead_before=None,
                    )
                    order["filled_quantity"] = fill_quantity
                    order["status"] = (
                        "filled"
                        if fill_quantity >= requested - 1e-12
                        else "partially_filled_canceled"
                    )
                else:
                    limit_price = (
                        _number(row, "best_bid", "bid_price_1")
                        if side == 1
                        else _number(row, "best_ask", "ask_price_1")
                    )
                    order["limit_price"] = limit_price
                    order["status"] = "working"
                    active_limits.append(
                        _ActiveLimit(
                            order=order,
                            remaining_quantity=accepted,
                            queue_ahead=config.queue_ahead_units,
                            cancel_effective_position=(
                                position_index
                                + config.limit_max_age_events
                                + config.cancel_latency_events
                            ),
                        )
                    )

            # Mark open inventory at every replay event, including events with no fill.
            record_equity(row, symbol)

        for active in active_limits:
            active.order["status"] = (
                "partially_filled_end_of_data"
                if _object_float(active.order["filled_quantity"]) > 0
                else "end_of_data"
            )

        final_mid[symbol] = _mid(rows[-1])
        if config.liquidate_at_end and abs(positions[symbol]) > 1e-12:
            final_row = rows[-1]
            side = -1 if positions[symbol] > 0 else 1
            quantity = abs(positions[symbol])
            displayed = _displayed_depth(final_row, side)
            fill_quantity = _round_quantity(min(quantity, max(0.0, displayed)), final_row)
            next_order_id += 1
            liquidation_order = {
                "order_id": next_order_id,
                "sample_id": _event_identifier(final_row),
                "symbol": symbol,
                "decision_event_ts_ns": _timestamp(final_row),
                "decision_position": len(rows) - 1,
                "decision_continuity_id": _continuity(final_row),
                "arrival_position": len(rows) - 1,
                "arrival_event_ts_ns": _timestamp(final_row),
                "side": side,
                "order_type": "market",
                "requested_quantity": quantity,
                "accepted_quantity": fill_quantity,
                "filled_quantity": fill_quantity,
                "decision_mid_price": _mid(final_row),
                "limit_price": None,
                "status": "forced_liquidation",
                "rejection_reason": None,
            }
            order_rows.append(liquidation_order)
            if fill_quantity > 0:
                price = (
                    _number(final_row, "best_bid", "bid_price_1")
                    if side == -1
                    else _number(final_row, "best_ask", "ask_price_1")
                )
                liquidation_depth_ratio = fill_quantity / max(displayed, 1e-12)
                liquidation_slippage_bps = config.slippage_bps_per_unit * liquidation_depth_ratio
                price = _round_price_adversely(
                    price * (1.0 + side * liquidation_slippage_bps / 10_000.0),
                    side,
                    final_row,
                )
                record_fill(
                    order=liquidation_order,
                    row=final_row,
                    rows=rows,
                    position_index=len(rows) - 1,
                    price=price,
                    quantity=fill_quantity,
                    liquidity="taker",
                    queue_ahead_before=None,
                    forced_liquidation=True,
                )
                forced_liquidation_quantity += fill_quantity
        unliquidated_quantity += abs(positions[symbol])

    gross_pnl_by_symbol = {
        symbol: gross_cash[symbol] + positions[symbol] * final_mid[symbol] for symbol in final_mid
    }
    gross_pnl = sum(gross_pnl_by_symbol.values())
    net_pnl = gross_pnl - total_fees
    maximum_drawdown = _portfolio_max_drawdown(equity_rows)
    requested_quantity = sum(
        _object_float(order["requested_quantity"])
        for order in order_rows
        if order["status"] != "forced_liquidation"
    )
    accepted_quantity = sum(
        _object_float(order["accepted_quantity"])
        for order in order_rows
        if order["status"] != "forced_liquidation"
    )
    filled_quantity = sum(
        _object_float(order["filled_quantity"])
        for order in order_rows
        if order["status"] != "forced_liquidation"
    )
    partially_filled = sum(
        1 for order in order_rows if str(order["status"]).startswith("partially_filled")
    )
    strategy_orders = sum(1 for order in order_rows if order["status"] != "forced_liquidation")
    strategy_fills = [fill for fill in fill_rows if not bool(fill["forced_liquidation"])]
    available_markouts = [
        fill for fill in strategy_fills if fill["post_fill_markout_bps"] is not None
    ]
    mean_markout = (
        float(
            np.average(
                [_object_float(fill["post_fill_markout_bps"]) for fill in available_markouts],
                weights=[_object_float(fill["notional"]) for fill in available_markouts],
            )
        )
        if available_markouts
        else None
    )
    mean_arrival_cost = (
        float(
            np.average(
                [_object_float(fill["arrival_cost_bps"]) for fill in strategy_fills],
                weights=[_object_float(fill["notional"]) for fill in strategy_fills],
            )
        )
        if strategy_fills
        else None
    )
    unliquidated_by_symbol = {
        symbol: abs(position) for symbol, position in positions.items() if abs(position) > 1e-12
    }
    net_pnl_by_symbol = {
        symbol: gross_pnl_by_symbol[symbol] - fees_by_symbol[symbol]
        for symbol in gross_pnl_by_symbol
    }
    metrics: dict[str, Any] = {
        "order_type": order_type,
        "size_multiplier": size_multiplier,
        "strategy_orders": strategy_orders,
        "strategy_fills": len(strategy_fills),
        "forced_liquidation_fills": len(fill_rows) - len(strategy_fills),
        "requested_quantity": requested_quantity,
        "accepted_quantity": accepted_quantity,
        "filled_quantity": filled_quantity,
        "fill_ratio": filled_quantity / accepted_quantity if accepted_quantity else None,
        "fill_ratio_requested": (
            filled_quantity / requested_quantity if requested_quantity else None
        ),
        "partial_fill_order_ratio": partially_filled / strategy_orders if strategy_orders else None,
        "gross_pnl": gross_pnl,
        "marked_gross_pnl": gross_pnl,
        "gross_pnl_by_symbol": gross_pnl_by_symbol,
        "maker_fees": maker_fees,
        "taker_fees": taker_fees,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "marked_net_pnl": net_pnl,
        "net_pnl_by_symbol": net_pnl_by_symbol,
        "maximum_drawdown": maximum_drawdown,
        "maximum_drawdown_bps_of_turnover": (
            maximum_drawdown / turnover_notional * 10_000.0 if turnover_notional else None
        ),
        "turnover_notional": turnover_notional,
        "turnover_notional_by_symbol": dict(turnover_by_symbol),
        "gross_edge_bps": gross_pnl / turnover_notional * 10_000.0 if turnover_notional else None,
        "net_edge_bps": net_pnl / turnover_notional * 10_000.0 if turnover_notional else None,
        "mean_arrival_cost_bps": mean_arrival_cost,
        "mean_post_fill_markout_bps": mean_markout,
        "mean_adverse_selection_bps": -mean_markout if mean_markout is not None else None,
        "maximum_absolute_inventory": maximum_inventory,
        "maximum_absolute_inventory_by_symbol": dict(maximum_inventory_by_symbol),
        "forced_liquidation_quantity": forced_liquidation_quantity,
        "unliquidated_quantity": unliquidated_quantity,
        "unliquidated_quantity_by_symbol": unliquidated_by_symbol,
        "unliquidated_valuation": (
            "final_mid_mark_not_realized" if unliquidated_by_symbol else "none"
        ),
    }
    assumptions: dict[str, Any] = {
        "replay_is_exogenous": True,
        "live_trading": False,
        "decision_latency_events": config.decision_latency_events,
        "order_latency_events": config.order_latency_events,
        "maker_fee_bps": config.maker_fee_bps,
        "taker_fee_bps": config.taker_fee_bps,
        "signal_threshold": config.signal_threshold,
        "base_order_size_units": config.order_size_units,
        "scenario_size_multiplier": size_multiplier,
        "half_spread_bps_fallback": config.half_spread_bps,
        "slippage_bps_per_unit_of_displayed_depth": config.slippage_bps_per_unit,
        "spread_source": "observed top of book; configured half-spread fallback is not used",
        "market_depth": "L1 only; residual size is canceled rather than extrapolated",
        "limit_fill_model": (
            "opposing printed volume depletes a fixed queue-ahead proxy; eligible residual volume "
            "fills with a seeded Bernoulli probability"
        ),
        "limit_fill_base_probability": config.limit_fill_base_probability,
        "queue_ahead_units": config.queue_ahead_units,
        "limit_max_age_events": config.limit_max_age_events,
        "cancel_latency_events": config.cancel_latency_events,
        "inventory_limit_units_per_symbol": config.max_position_units,
        "end_liquidation": config.liquidate_at_end,
        "capacity_multipliers": list(config.capacity_multipliers),
        "markout_policy": "censored at end of data or continuity boundary; never shortened",
        "residual_inventory_valuation": "final midpoint mark, explicitly unrealized",
        "multi_instrument_units": "quantity and inventory maps are reported per symbol",
    }

    orders_frame = pl.DataFrame(order_rows) if order_rows else _empty_frame({"order_id": pl.Int64})
    fills_frame = pl.DataFrame(fill_rows) if fill_rows else _empty_frame({"fill_id": pl.Int64})
    positions_frame = (
        pl.DataFrame(position_rows).sort(["event_ts_ns", "symbol", "fill_id"])
        if position_rows
        else _empty_frame({"fill_id": pl.Int64, "position_units": pl.Float64})
    )
    return SimulationResult(
        orders=orders_frame,
        fills=fills_frame,
        positions=positions_frame,
        metrics=metrics,
        assumptions=assumptions,
    )


def run_execution_sensitivity(
    events: pl.DataFrame,
    predictions: pl.DataFrame,
    config: ExecutionConfig,
    *,
    seed: int,
    markout_events: int,
) -> pl.DataFrame:
    """Evaluate market/limit execution over the declared capacity grid."""
    rows: list[dict[str, Any]] = []
    for order_type in cast(tuple[OrderType, ...], ("market", "limit")):
        for multiplier in config.capacity_multipliers:
            result = simulate_predictions(
                events,
                predictions,
                config,
                order_type=order_type,
                size_multiplier=multiplier,
                seed=seed,
                markout_events=markout_events,
            )
            rows.append(result.metrics)
    return pl.DataFrame(rows)
