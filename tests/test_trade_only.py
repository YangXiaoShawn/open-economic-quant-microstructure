from __future__ import annotations

import math

import polars as pl
import pytest

from microstructure.config import FeatureConfig
from microstructure.research.features import TemporalLeakageError
from microstructure.research.trade_only import (
    build_trade_only_research_frame,
    validate_trade_only_temporal_contract,
)

SECOND = 1_000_000_000


def _config() -> FeatureConfig:
    return FeatureConfig(
        trade_windows=(2,),
        volatility_window=2,
        intensity_window=2,
        label_horizon_events=2,
        large_trade_quantile=0.9,
    )


def _trades() -> pl.DataFrame:
    rows = [
        ("segment-a", 1, 0, 100.0, 1.0, "buy"),
        ("segment-a", 2, 1, 101.0, 2.0, "sell"),
        ("segment-a", 3, 2, 99.0, 3.0, "buy"),
        ("segment-a", 4, 3, 102.0, 4.0, "buy"),
        ("segment-a", 5, 4, 104.0, 5.0, "sell"),
        ("segment-b", 10, 5, 200.0, 6.0, "buy"),
        ("segment-b", 11, 6, 201.0, 7.0, "sell"),
        ("segment-b", 12, 7, 199.0, 8.0, "sell"),
    ]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(rows),
            "continuity_id": [row[0] for row in rows],
            "trade_id": [row[1] for row in rows],
            "event_ts_ns": [row[2] * SECOND for row in rows],
            "available_ts_ns": [row[2] * SECOND for row in rows],
            "price": [row[3] for row in rows],
            "quantity": [row[4] for row in rows],
            "aggressor_side": [row[5] for row in rows],
        }
    )


def test_trade_only_features_are_hand_checked_and_causal() -> None:
    frame = build_trade_only_research_frame(_trades(), _config())
    second = frame.filter(pl.col("decision_trade_id") == 2).row(0, named=True)

    assert second["signed_trade_volume_w2"] == pytest.approx(-1.0)
    assert second["trade_volume_w2"] == pytest.approx(3.0)
    assert second["trade_imbalance_w2"] == pytest.approx(-1.0 / 3.0)
    assert second["trade_count_w2"] == 2.0
    assert second["trade_intensity_w2"] == pytest.approx(2.0)
    assert second["log_trade_return_1"] == pytest.approx(math.log(101.0 / 100.0))
    assert second["realized_volatility_w2"] == pytest.approx(abs(math.log(101.0 / 100.0)))
    assert second["max_feature_source_ts_ns"] == second["decision_ts_ns"]
    assert second["max_feature_source_trade_id"] == second["decision_trade_id"]


def test_future_mutation_cannot_change_past_trade_features() -> None:
    original = build_trade_only_research_frame(_trades(), _config())
    mutated_trades = _trades().with_columns(
        pl.when(pl.col("available_ts_ns") > SECOND)
        .then(pl.col("price") * 10.0)
        .otherwise(pl.col("price"))
        .alias("price"),
        pl.when(pl.col("available_ts_ns") > SECOND)
        .then(pl.col("quantity") * 100.0)
        .otherwise(pl.col("quantity"))
        .alias("quantity"),
        pl.when(pl.col("available_ts_ns") > SECOND)
        .then(pl.lit("sell"))
        .otherwise(pl.col("aggressor_side"))
        .alias("aggressor_side"),
    )
    mutated = build_trade_only_research_frame(mutated_trades, _config())
    feature_columns = [
        "signed_trade_volume_w2",
        "trade_volume_w2",
        "trade_imbalance_w2",
        "trade_intensity_w2",
        "log_trade_return_1",
        "realized_volatility_w2",
    ]
    past = pl.col("decision_ts_ns") <= SECOND

    assert (
        original.filter(past)
        .select(feature_columns)
        .equals(mutated.filter(past).select(feature_columns))
    )


def test_future_trade_labels_are_exact_censored_and_continuity_local() -> None:
    frame = build_trade_only_research_frame(_trades(), _config())
    second = frame.filter(pl.col("decision_trade_id") == 2).row(0, named=True)
    assert second["future_trade_return"] == pytest.approx(math.log(102.0 / 101.0))
    assert second["future_trade_price"] == 102.0
    assert second["future_trade_direction"] == 1
    assert second["future_trade_up"] == 1
    assert second["label_information_end_ts_ns"] == 3 * SECOND
    assert second["label_information_end_trade_id"] == 4

    segment_a_tail = frame.filter(
        (pl.col("continuity_id") == "segment-a") & pl.col("decision_trade_id").is_in([4, 5])
    )
    assert segment_a_tail.get_column("right_censored").to_list() == [True, True]
    assert segment_a_tail.get_column("future_trade_return").null_count() == 2

    first_b = frame.filter(pl.col("decision_trade_id") == 10).row(0, named=True)
    assert first_b["signed_trade_volume_w2"] == pytest.approx(6.0)
    assert first_b["trade_imbalance_w2"] == pytest.approx(1.0)
    assert first_b["log_trade_return_1"] == 0.0
    assert first_b["future_trade_price"] == 199.0
    assert first_b["label_information_end_trade_id"] == 12

    audit = validate_trade_only_temporal_contract(frame)
    assert audit.rows == 8
    assert audit.labeled_rows == 4
    assert audit.right_censored_rows == 4
    assert audit.continuity_segments == 2


def test_trade_only_lineage_guard_rejects_future_source() -> None:
    frame = build_trade_only_research_frame(_trades(), _config())
    leaked = frame.with_columns(
        (pl.col("feature_cutoff_ts_ns") + 1).alias("max_feature_source_ts_ns")
    )

    with pytest.raises(TemporalLeakageError, match="feature lineage"):
        validate_trade_only_temporal_contract(leaked)
