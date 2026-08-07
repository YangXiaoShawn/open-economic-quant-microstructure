from __future__ import annotations

import polars as pl
import pytest

from microstructure.config import ExecutionConfig
from microstructure.execution import simulate_predictions


def execution_config(**overrides: object) -> ExecutionConfig:
    values: dict[str, object] = {
        "decision_latency_events": 0,
        "order_latency_events": 0,
        "maker_fee_bps": 10.0,
        "taker_fee_bps": 10.0,
        "half_spread_bps": 1.0,
        "slippage_bps_per_unit": 0.0,
        "signal_threshold": 0.6,
        "max_position_units": 10.0,
        "order_size_units": 1.0,
        "limit_fill_base_probability": 1.0,
        "queue_ahead_units": 0.0,
        "limit_max_age_events": 10,
        "cancel_latency_events": 1,
        "liquidate_at_end": False,
        "capacity_multipliers": (1.0,),
    }
    values.update(overrides)
    return ExecutionConfig(**values)  # type: ignore[arg-type]


def event_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "symbol": "BTCUSDT",
        "bid_depth_1": 100.0,
        "ask_depth_1": 100.0,
        "trade_side": 0,
        "trade_quantity": 0.0,
        "trade_price": 101.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_market_round_trip_accounts_for_taker_fees() -> None:
    events = event_frame(
        [
            {"sample_id": 0, "event_ts_ns": 0, "best_bid": 100.0, "best_ask": 102.0},
            {"sample_id": 1, "event_ts_ns": 1, "best_bid": 103.0, "best_ask": 105.0},
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0, 1],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "probability_up": [0.9, 0.1],
            "is_oos": [True, True],
            "split": ["test", "test"],
        }
    )

    result = simulate_predictions(events, predictions, execution_config())

    assert result.metrics["gross_pnl"] == pytest.approx(1.0)
    assert result.metrics["total_fees"] == pytest.approx(0.205)
    assert result.metrics["net_pnl"] == pytest.approx(0.795)
    assert result.metrics["turnover_notional"] == pytest.approx(205.0)
    assert result.metrics["maximum_drawdown"] == pytest.approx(1.102)
    assert "net_equity" in result.positions.columns


def test_maximum_drawdown_marks_inventory_on_intervening_events_without_fills() -> None:
    events = event_frame(
        [
            {"sample_id": 0, "event_ts_ns": 0, "best_bid": 99.0, "best_ask": 101.0},
            {"sample_id": 1, "event_ts_ns": 1, "best_bid": 49.0, "best_ask": 51.0},
            {"sample_id": 2, "event_ts_ns": 2, "best_bid": 100.0, "best_ask": 102.0},
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0, 2],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "probability_up": [0.9, 0.1],
            "is_oos": [True, True],
            "split": ["test", "test"],
        }
    )

    result = simulate_predictions(events, predictions, execution_config())

    assert result.metrics["maximum_drawdown"] == pytest.approx(51.101)


def test_market_order_uses_arrival_state_and_top_depth_only() -> None:
    events = event_frame(
        [
            {"sample_id": 0, "event_ts_ns": 0, "best_bid": 99.0, "best_ask": 101.0},
            {
                "sample_id": 1,
                "event_ts_ns": 1,
                "best_bid": 109.0,
                "best_ask": 111.0,
                "ask_depth_1": 1.5,
            },
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [0.9],
            "is_oos": [True],
            "split": ["test"],
        }
    )
    config = execution_config(order_latency_events=1, order_size_units=2.0)

    result = simulate_predictions(events, predictions, config)

    assert result.fills["price"].to_list() == [111.0]
    assert result.fills["quantity"].to_list() == [1.5]
    assert result.orders["status"].to_list() == ["partially_filled_canceled"]


def test_limit_queue_proxy_produces_partial_then_complete_fill() -> None:
    events = event_frame(
        [
            {
                "sample_id": 0,
                "event_ts_ns": 0,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "trade_price": 101.0,
            },
            {
                "sample_id": 1,
                "event_ts_ns": 1,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "trade_side": -1,
                "trade_quantity": 3.0,
                "trade_price": 100.0,
            },
            {
                "sample_id": 2,
                "event_ts_ns": 2,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "trade_side": -1,
                "trade_quantity": 4.0,
                "trade_price": 100.0,
            },
            {
                "sample_id": 3,
                "event_ts_ns": 3,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "trade_side": -1,
                "trade_quantity": 1.0,
                "trade_price": 100.0,
            },
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [0.9],
            "is_oos": [True],
            "split": ["test"],
        }
    )
    config = execution_config(order_size_units=3.0, queue_ahead_units=5.0)

    result = simulate_predictions(events, predictions, config, order_type="limit", seed=7)

    assert result.fills["quantity"].to_list() == [2.0, 1.0]
    assert result.fills["event_id"].to_list() == [2, 3]
    assert result.orders["status"].to_list() == ["filled"]


def test_equal_position_trade_occurs_before_limit_arrival() -> None:
    events = event_frame(
        [
            {"sample_id": 0, "event_ts_ns": 0, "best_bid": 100.0, "best_ask": 102.0},
            {
                "sample_id": 1,
                "event_ts_ns": 1,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "trade_side": -1,
                "trade_quantity": 10.0,
                "trade_price": 100.0,
            },
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [0.9],
            "is_oos": [True],
            "split": ["test"],
        }
    )
    config = execution_config(order_latency_events=1)

    result = simulate_predictions(events, predictions, config, order_type="limit")

    assert result.fills.is_empty()
    assert result.orders["status"].to_list() == ["end_of_data"]


def test_simulator_rejects_in_sample_predictions() -> None:
    events = event_frame([{"sample_id": 0, "event_ts_ns": 0, "best_bid": 100.0, "best_ask": 102.0}])
    predictions = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [0.9],
            "is_oos": [False],
            "split": ["test"],
        }
    )

    with pytest.raises(ValueError, match="non-OOS"):
        simulate_predictions(events, predictions, execution_config())


def test_inventory_cap_and_partial_end_liquidation_are_explicit() -> None:
    events = event_frame(
        [
            {"sample_id": 0, "event_ts_ns": 0, "best_bid": 99.0, "best_ask": 101.0},
            {"sample_id": 1, "event_ts_ns": 1, "best_bid": 99.0, "best_ask": 101.0},
            {
                "sample_id": 2,
                "event_ts_ns": 2,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "bid_depth_1": 1.0,
            },
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "symbol": ["BTCUSDT"] * 3,
            "probability_up": [0.9, 0.9, 0.9],
            "is_oos": [True] * 3,
            "split": ["test"] * 3,
        }
    )
    config = execution_config(max_position_units=2.0, liquidate_at_end=True)

    result = simulate_predictions(events, predictions, config)

    assert result.orders.filter(pl.col("status") == "rejected").height == 1
    assert result.metrics["maximum_absolute_inventory"] == pytest.approx(2.0)
    assert result.metrics["forced_liquidation_quantity"] == pytest.approx(1.0)
    assert result.metrics["unliquidated_quantity"] == pytest.approx(1.0)
    assert result.metrics["gross_pnl"] == pytest.approx(-1.0)


def test_execution_requires_explicit_held_out_provenance() -> None:
    events = event_frame([{"sample_id": 0, "event_ts_ns": 0, "best_bid": 100.0, "best_ask": 102.0}])
    missing_oos = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [0.9],
            "split": ["test"],
        }
    )
    validation = missing_oos.with_columns(
        pl.lit(True).alias("is_oos"), pl.lit("validation").alias("split")
    )

    with pytest.raises(ValueError, match="explicit OOS"):
        simulate_predictions(events, missing_oos, execution_config())
    with pytest.raises(ValueError, match="held-out test"):
        simulate_predictions(events, validation, execution_config())


def test_invalid_probability_and_negative_latency_or_markout_fail_closed() -> None:
    events = event_frame([{"sample_id": 0, "event_ts_ns": 0, "best_bid": 100.0, "best_ask": 102.0}])
    predictions = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [float("nan")],
            "is_oos": [True],
            "split": ["test"],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        simulate_predictions(events, predictions, execution_config())
    valid = predictions.with_columns(pl.lit(0.9).alias("probability_up"))
    with pytest.raises(ValueError, match="latency"):
        simulate_predictions(events, valid, execution_config(order_latency_events=-1))
    with pytest.raises(ValueError, match="markout"):
        simulate_predictions(events, valid, execution_config(), markout_events=-1)
    with pytest.raises(ValueError, match="queue_ahead_units"):
        simulate_predictions(
            events,
            valid,
            execution_config(queue_ahead_units=-1.0),
            order_type="limit",
        )


def test_one_print_cannot_fill_multiple_passive_orders_beyond_its_volume() -> None:
    events = event_frame(
        [
            {"sample_id": 0, "event_ts_ns": 0, "best_bid": 100.0, "best_ask": 102.0},
            {
                "sample_id": 1,
                "event_ts_ns": 1,
                "best_bid": 100.0,
                "best_ask": 102.0,
                "trade_side": -1,
                "trade_quantity": 1.0,
                "trade_price": 100.0,
            },
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0, 0],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "probability_up": [0.9, 0.9],
            "is_oos": [True, True],
            "split": ["test", "test"],
        }
    )

    result = simulate_predictions(
        events, predictions, execution_config(), order_type="limit", seed=5
    )

    assert result.fills["quantity"].sum() == pytest.approx(1.0)
    assert result.metrics["filled_quantity"] == pytest.approx(1.0)


def test_orders_and_markouts_do_not_cross_continuity_gaps() -> None:
    events = event_frame(
        [
            {
                "sample_id": 0,
                "event_ts_ns": 0,
                "continuity_id": "A",
                "best_bid": 100.0,
                "best_ask": 102.0,
            },
            {
                "sample_id": 1,
                "event_ts_ns": 1,
                "continuity_id": "B",
                "best_bid": 101.0,
                "best_ask": 103.0,
                "trade_side": -1,
                "trade_quantity": 10.0,
                "trade_price": 100.0,
            },
        ]
    )
    predictions = pl.DataFrame(
        {
            "sample_id": [0],
            "symbol": ["BTCUSDT"],
            "probability_up": [0.9],
            "is_oos": [True],
            "split": ["test"],
        }
    )

    delayed = simulate_predictions(
        events,
        predictions,
        execution_config(order_latency_events=1),
        order_type="limit",
    )
    immediate = simulate_predictions(
        events, predictions, execution_config(), order_type="market", markout_events=1
    )

    assert delayed.fills.is_empty()
    assert delayed.orders["status"].to_list() == ["canceled_continuity_gap"]
    assert immediate.fills["markout_available"].to_list() == [False]
    assert immediate.fills["post_fill_markout_bps"].to_list() == [None]
