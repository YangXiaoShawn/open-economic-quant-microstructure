from __future__ import annotations

import math

import polars as pl
import pytest

from microstructure.research.labels import (
    AdverseSelectionSpec,
    ClockTimeLabelSpec,
    LimitFillAssumptions,
    add_event_time_price_impact_labels,
    build_clock_time_mid_labels,
    build_clock_time_price_impact_labels,
    build_hypothetical_limit_fill_labels,
    build_post_fill_adverse_selection_labels,
)

SECOND = 1_000_000_000


def _clock_books() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_id": ["a0", "a1", "a3", "a6", "b7", "b8", "b9"],
            "symbol": ["BTCUSDT"] * 7,
            "continuity_id": ["a", "a", "a", "a", "b", "b", "b"],
            "decision_ts_ns": [value * SECOND for value in (0, 1, 3, 6, 7, 8, 9)],
            "decision_sequence": [1, 2, 3, 4, 10, 11, 12],
            "mid_price": [100.0, 101.0, 103.0, 106.0, 200.0, 201.0, 202.0],
            "trade_sign": [1, -1, 1, -1, 1, -1, 1],
            "is_valid": [True] * 7,
        }
    )


def test_clock_time_labels_use_first_later_state_and_do_not_cross_gaps() -> None:
    labels = build_clock_time_mid_labels(
        _clock_books(),
        ClockTimeLabelSpec(horizons_ns=(2 * SECOND,), max_target_staleness_ns=SECOND),
    )
    at_zero = labels.filter(pl.col("sample_id") == "a0").row(0, named=True)
    assert at_zero["clock_target_ts_ns"] == 2 * SECOND
    assert at_zero["clock_label_information_end_ts_ns"] == 3 * SECOND
    assert at_zero["clock_observed_target_staleness_ns"] == SECOND
    assert at_zero["clock_future_mid_return"] == pytest.approx(math.log(103.0 / 100.0))
    assert at_zero["clock_future_mid_direction"] == 1
    assert at_zero["clock_label_is_descriptive"] is True

    # Segment B has observations after t=6, but they are not valid targets for A.
    at_six = labels.filter(pl.col("sample_id") == "a6").row(0, named=True)
    assert at_six["clock_right_censored"] is True
    assert at_six["clock_censor_reason"] == "no_same_segment_future_state"
    assert at_six["clock_future_mid_return"] is None
    assert at_six["clock_label_information_end_ts_ns"] is None


def test_clock_time_labels_preserve_exact_future_book_identity_when_requested() -> None:
    labels = build_clock_time_mid_labels(
        _clock_books(),
        ClockTimeLabelSpec(horizons_ns=(2 * SECOND,), max_target_staleness_ns=SECOND),
        book_identity_column="decision_sequence",
    )

    at_zero = labels.filter(pl.col("sample_id") == "a0").row(0, named=True)
    assert at_zero["clock_label_information_end_ts_ns"] == 3 * SECOND
    assert at_zero["clock_label_information_end_identity"] == 3

    at_six = labels.filter(pl.col("sample_id") == "a6").row(0, named=True)
    assert at_six["clock_right_censored"] is True
    assert at_six["clock_label_information_end_identity"] is None


def test_clock_price_impact_is_side_signed() -> None:
    decisions = _clock_books().filter(pl.col("sample_id").is_in(["a0", "a1"]))
    labels = build_clock_time_price_impact_labels(
        decisions,
        ClockTimeLabelSpec(horizons_ns=(2 * SECOND,)),
        book_states=_clock_books(),
    )
    buy = labels.filter(pl.col("sample_id") == "a0").row(0, named=True)
    sell = labels.filter(pl.col("sample_id") == "a1").row(0, named=True)
    assert buy["clock_signed_price_impact_bps"] == pytest.approx(10_000.0 * math.log(103.0 / 100.0))
    assert sell["clock_signed_price_impact_bps"] == pytest.approx(
        -10_000.0 * math.log(103.0 / 101.0)
    )
    assert labels.get_column("clock_label_kind").unique().to_list() == [
        "clock_time_signed_price_impact"
    ]


def test_event_time_price_impact_preserves_future_information_end() -> None:
    frame = pl.DataFrame(
        {
            "trade_side": [1, -1, 1],
            "future_mid_return": [0.01, 0.02, None],
            "label_information_end_ts_ns": [20, 21, None],
            "right_censored": [False, False, True],
            "label_horizon_events": [2, 2, 2],
        }
    )

    labeled = add_event_time_price_impact_labels(frame, side_column="trade_side")

    assert labeled["event_time_signed_price_impact_bps"].to_list() == [100.0, -200.0, None]
    assert labeled["event_impact_label_information_end_ts_ns"].to_list() == [20, 21, None]
    assert labeled["event_impact_right_censored"].to_list() == [False, False, True]


def _limit_books() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sample_id": [f"a{index}" for index in range(5)],
            "symbol": ["BTCUSDT"] * 5,
            "continuity_id": ["a"] * 5,
            "decision_ts_ns": [index * SECOND for index in range(5)],
            "decision_sequence": list(range(1, 6)),
            "best_bid": [100.0] * 5,
            "best_ask": [102.0] * 5,
            "bid_quantity": [10.0] * 5,
            "ask_quantity": [10.0] * 5,
            "is_valid": [True] * 5,
        }
    )


def _limit_trades() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "continuity_id": ["a"] * 3,
            # The huge print exactly at t=0 must not fill an order activated at t=0.
            "available_ts_ns": [0, SECOND, 2 * SECOND],
            "price": [100.0, 100.0, 100.0],
            "quantity": [100.0, 3.0, 4.0],
            "aggressor_side": ["sell", "sell", "sell"],
        }
    )


def test_limit_fill_proxy_is_strict_partial_and_explicitly_censored() -> None:
    labels = build_hypothetical_limit_fill_labels(
        _limit_books(),
        _limit_trades(),
        LimitFillAssumptions(
            side="buy",
            horizon_ns=2 * SECOND,
            order_quantity=3.0,
            queue_ahead_fraction=0.5,
        ),
    )
    at_zero = labels.filter(pl.col("sample_id") == "a0").row(0, named=True)
    assert at_zero["limit_initial_queue_ahead"] == 5.0
    assert at_zero["limit_observed_executable_quantity"] == 7.0
    assert at_zero["limit_fill_quantity"] == 2.0
    assert at_zero["limit_fill_fraction"] == pytest.approx(2.0 / 3.0)
    assert at_zero["limit_full_fill"] is False
    assert at_zero["limit_label_information_end_ts_ns"] == 2 * SECOND
    assert at_zero["limit_trade_evidence_required"] is True
    assert at_zero["limit_equal_time_ordering"] == "trade_at_activation_excluded"
    assert "cancellations" in at_zero["limit_label_assumption"]

    at_three = labels.filter(pl.col("sample_id") == "a3").row(0, named=True)
    assert at_three["limit_right_censored"] is True
    assert at_three["limit_censor_reason"] == "segment_ends_before_horizon"
    assert at_three["limit_fill_fraction"] is None
    assert at_three["limit_label_information_end_ts_ns"] is None

    no_trade_labels = build_hypothetical_limit_fill_labels(
        _limit_books(),
        _limit_trades().head(0),
        LimitFillAssumptions(
            side="buy",
            horizon_ns=2 * SECOND,
            order_quantity=3.0,
            queue_ahead_fraction=0.5,
        ),
    )
    no_trade_at_zero = no_trade_labels.filter(pl.col("sample_id") == "a0").row(0, named=True)
    assert no_trade_at_zero["limit_right_censored"] is False
    assert no_trade_at_zero["limit_fill_quantity"] == 0.0


def test_post_fill_adverse_selection_has_side_aware_markout_and_gap_censoring() -> None:
    fills = pl.DataFrame(
        {
            "fill_id": ["buy", "sell", "gap"],
            "symbol": ["BTCUSDT"] * 3,
            "continuity_id": ["a", "a", "a"],
            "fill_ts_ns": [0, SECOND, 6 * SECOND],
            "fill_price": [100.0, 104.0, 106.0],
            "side": ["buy", "sell", "buy"],
        }
    )
    labels = build_post_fill_adverse_selection_labels(
        fills,
        _clock_books(),
        AdverseSelectionSpec(horizons_ns=(2 * SECOND,), max_target_staleness_ns=SECOND),
    )
    buy = labels.filter(pl.col("fill_id") == "buy").row(0, named=True)
    assert buy["post_fill_markout_bps"] == pytest.approx(300.0)
    assert buy["adverse_selection_bps"] == pytest.approx(-300.0)
    assert buy["adverse_selection_indicator"] is False
    assert buy["adverse_label_information_end_ts_ns"] == 3 * SECOND

    sell = labels.filter(pl.col("fill_id") == "sell").row(0, named=True)
    assert sell["post_fill_markout_bps"] == pytest.approx(10_000.0 / 104.0)
    assert sell["adverse_selection_bps"] < 0

    gap = labels.filter(pl.col("fill_id") == "gap").row(0, named=True)
    assert gap["adverse_right_censored"] is True
    assert gap["adverse_censor_reason"] == "no_same_segment_future_state"
    assert gap["post_fill_markout_bps"] is None
    assert gap["adverse_label_information_end_ts_ns"] is None
