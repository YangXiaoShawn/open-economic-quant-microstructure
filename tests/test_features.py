from __future__ import annotations

import math

import polars as pl
import pytest

from microstructure.config import FeatureConfig
from microstructure.research.features import (
    ResearchDataError,
    TemporalLeakageError,
    build_research_features,
    build_research_frame,
    model_feature_columns,
    validate_temporal_contract,
)

SECOND = 1_000_000_000


def _feature_config() -> FeatureConfig:
    return FeatureConfig(
        trade_windows=(2,),
        volatility_window=2,
        intensity_window=2,
        label_horizon_events=2,
        large_trade_quantile=0.9,
    )


def _books() -> pl.DataFrame:
    rows = [
        (0, "segment-a", 1, 100.0, 10.0, 102.0, 10.0),
        (1, "segment-a", 2, 100.0, 12.0, 102.0, 8.0),
        (2, "segment-a", 3, 101.0, 5.0, 102.0, 5.0),
        (3, "segment-a", 4, 101.0, 4.0, 103.0, 6.0),
        (4, "segment-a", 5, 102.0, 8.0, 103.0, 4.0),
        (5, "segment-a", 6, 102.0, 6.0, 104.0, 6.0),
        (6, "segment-b", 20, 110.0, 5.0, 112.0, 5.0),
        (7, "segment-b", 21, 110.0, 7.0, 112.0, 3.0),
        (8, "segment-b", 22, 111.0, 5.0, 112.0, 5.0),
    ]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(rows),
            "event_ts_ns": [row[0] * SECOND for row in rows],
            "available_ts_ns": [row[0] * SECOND for row in rows],
            "continuity_id": [row[1] for row in rows],
            "sequence_end": [row[2] for row in rows],
            "is_valid": [True] * len(rows),
            "best_bid": [row[3] for row in rows],
            "bid_quantity": [row[4] for row in rows],
            "best_ask": [row[5] for row in rows],
            "ask_quantity": [row[6] for row in rows],
        }
    )


def _trades() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "continuity_id": ["segment-a", "segment-a", "segment-a", "segment-b"],
            "trade_id": [1, 2, 3, 4],
            "available_ts_ns": [SECOND // 2, SECOND, 3 * SECOND // 2, 7 * SECOND],
            "quantity": [2.0, 7.0, 3.0, 100.0],
            "aggressor_side": ["buy", "sell", "sell", "buy"],
        }
    )


def test_hand_checked_causal_features_and_strict_trade_tie() -> None:
    frame = build_research_frame(_books(), _trades(), _feature_config())
    at_one = frame.filter(pl.col("decision_ts_ns") == SECOND).row(0, named=True)

    assert at_one["mid_price"] == 101.0
    assert at_one["spread"] == 2.0
    assert at_one["queue_imbalance_l1"] == pytest.approx(0.2)
    assert at_one["causal_microprice"] == pytest.approx(101.2)
    assert at_one["ofi_l1"] == pytest.approx(4.0)
    # The buy at 0.5 seconds is known.  The sell timestamped exactly at this
    # book decision is from a separately ordered archive stream and is excluded.
    assert at_one["signed_trade_volume_w2"] == pytest.approx(2.0)
    assert at_one["trade_feature_max_source_ts_ns"] == SECOND // 2
    assert at_one["trade_feature_max_source_ts_ns"] < at_one["decision_ts_ns"]
    assert at_one["realized_price_impact_bps_1"] == pytest.approx(0.0)
    assert at_one["spread_recovery_bps_1"] == pytest.approx(0.0)
    assert at_one["depth_recovery_l1_1"] == pytest.approx(0.0)


def test_feature_stage_can_be_reused_before_any_label_horizon_is_opened() -> None:
    features = build_research_features(_books(), _trades(), _feature_config())
    labeled = build_research_frame(_books(), _trades(), _feature_config())

    assert "future_mid_return" not in features.columns
    assert features.select("symbol", "continuity_id", "decision_sequence").equals(
        labeled.select("symbol", "continuity_id", "decision_sequence")
    )
    assert (
        features.get_column("max_feature_source_ts_ns").to_list()
        == labeled.get_column("max_feature_source_ts_ns").to_list()
    )


def test_optional_multilevel_depth_and_cancellation_enter_canonical_frame() -> None:
    books = _books().with_columns(
        pl.lit("binance_spot").alias("venue"),
        (pl.col("bid_quantity") + 4.0).alias("depth_bid_5"),
        (pl.col("ask_quantity") + 6.0).alias("depth_ask_5"),
        (pl.col("bid_quantity") + 14.0).alias("depth_bid_10"),
        (pl.col("ask_quantity") + 16.0).alias("depth_ask_10"),
    )
    depth_deltas = pl.DataFrame(
        {
            "venue": ["binance_spot"] * books.height,
            "symbol": ["BTCUSDT"] * books.height,
            "event_ts_ns": books.get_column("event_ts_ns"),
            "available_ts_ns": books.get_column("available_ts_ns"),
            "continuity_id": books.get_column("continuity_id"),
            "first_update_id": books.get_column("sequence_end"),
            "last_update_id": books.get_column("sequence_end"),
            "bids": [
                [{"price_ticks": 10_000, "quantity_lots": 0 if index == 1 else 10}]
                for index in range(books.height)
            ],
            "asks": [[{"price_ticks": 10_200, "quantity_lots": 10}] for _ in range(books.height)],
        }
    )

    frame = build_research_frame(
        books,
        _trades(),
        _feature_config(),
        depth_deltas=depth_deltas,
    )
    at_one = frame.filter(pl.col("decision_ts_ns") == SECOND).row(0, named=True)

    assert at_one["depth_total_l5"] == pytest.approx(30.0)
    assert at_one["queue_imbalance_l5"] == pytest.approx(2.0 / 30.0)
    assert at_one["cancellation_deletes_w2"] == 1
    assert at_one["cancellation_intensity_w2"] == pytest.approx(0.25)
    assert at_one["cancellation_feature_max_source_ts_ns"] == SECOND
    assert "cancellation_intensity_w2" in model_feature_columns(frame)
    validate_temporal_contract(frame)


def test_future_event_label_is_exact_right_censored_and_gap_local() -> None:
    frame = build_research_frame(_books(), _trades(), _feature_config())
    at_two = frame.filter(pl.col("decision_ts_ns") == 2 * SECOND).row(0, named=True)
    assert at_two["future_mid_return"] == pytest.approx(math.log(102.5 / 101.5))
    assert at_two["future_mid_direction"] == 1
    assert at_two["future_mid_up"] == 1
    assert at_two["label_information_end_ts_ns"] == 4 * SECOND

    # The final two rows of segment A may not look into segment B even though
    # later book observations exist globally.
    segment_a_tail = frame.filter(
        (pl.col("continuity_id") == "segment-a") & (pl.col("decision_ts_ns") >= 4 * SECOND)
    )
    assert segment_a_tail.get_column("right_censored").to_list() == [True, True]
    assert segment_a_tail.get_column("future_mid_return").null_count() == 2

    first_b = (
        frame.filter(pl.col("continuity_id") == "segment-b")
        .sort("decision_sequence")
        .row(0, named=True)
    )
    assert first_b["ofi_l1"] == 0.0
    assert first_b["signed_trade_volume_w2"] == 0.0

    audit = validate_temporal_contract(frame)
    assert audit.rows == 9
    assert audit.right_censored_rows == 4
    assert audit.continuity_segments == 2


def test_delayed_old_continuity_trade_cannot_enter_new_segment_features() -> None:
    trades = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "continuity_id": ["segment-b", "segment-a"],
            "trade_id": [20, 21],
            "available_ts_ns": [6 * SECOND + SECOND // 4, 6 * SECOND + SECOND // 2],
            "quantity": [3.0, 100.0],
            "aggressor_side": ["sell", "buy"],
        }
    )
    frame = build_research_frame(_books(), trades, _feature_config())
    second_b = frame.filter(
        (pl.col("continuity_id") == "segment-b") & (pl.col("decision_ts_ns") == 7 * SECOND)
    ).row(0, named=True)

    assert second_b["signed_trade_volume_w2"] == pytest.approx(-3.0)
    assert second_b["trade_volume_w2"] == pytest.approx(3.0)
    assert second_b["trade_feature_max_source_ts_ns"] == 6 * SECOND + SECOND // 4


def test_unsegmented_trades_fail_closed_for_multi_segment_books() -> None:
    trades_without_continuity = _trades().drop("continuity_id")

    with pytest.raises(ResearchDataError, match="require continuity_id"):
        build_research_frame(_books(), trades_without_continuity, _feature_config())


def test_mutating_the_future_cannot_change_past_features() -> None:
    original_books = _books()
    original = build_research_frame(original_books, _trades(), _feature_config())
    mutated_books = original_books.with_columns(
        pl.when(pl.col("available_ts_ns") > 2 * SECOND)
        .then(pl.col("best_bid") + 10_000.0)
        .otherwise(pl.col("best_bid"))
        .alias("best_bid"),
        pl.when(pl.col("available_ts_ns") > 2 * SECOND)
        .then(pl.col("best_ask") + 10_000.0)
        .otherwise(pl.col("best_ask"))
        .alias("best_ask"),
    )
    mutated_trades = pl.concat(
        [
            _trades(),
            pl.DataFrame(
                {
                    "symbol": ["BTCUSDT"],
                    "continuity_id": ["segment-a"],
                    "trade_id": [99],
                    "available_ts_ns": [3 * SECOND],
                    "quantity": [1_000_000.0],
                    "aggressor_side": ["buy"],
                }
            ),
        ]
    )
    mutated = build_research_frame(mutated_books, mutated_trades, _feature_config())
    feature_columns = [
        "mid_price",
        "spread_bps",
        "queue_imbalance_l1",
        "causal_microprice",
        "ofi_l1",
        "signed_trade_volume_w2",
        "trade_volume_w2",
        "realized_volatility_w2",
    ]
    cutoff = pl.col("decision_ts_ns") <= 2 * SECOND
    assert (
        original.filter(cutoff)
        .select(feature_columns)
        .equals(mutated.filter(cutoff).select(feature_columns))
    )


def test_lineage_guard_rejects_deliberate_future_source() -> None:
    frame = build_research_frame(_books(), _trades(), _feature_config())
    leaked = frame.with_columns(
        (pl.col("feature_cutoff_ts_ns") + 1).alias("max_feature_source_ts_ns")
    )
    with pytest.raises(TemporalLeakageError, match="feature lineage"):
        validate_temporal_contract(leaked)
