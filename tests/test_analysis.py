from __future__ import annotations

import polars as pl
import pytest

from microstructure.research.analysis import (
    LiquidityShockThresholds,
    RegimeThresholds,
    assign_market_regimes,
    cross_instrument_stability_summary,
    estimate_signal_half_life,
    feature_stability_summary,
    intraday_liquidity_summary,
    large_trade_price_impact_summary,
    liquidity_recovery_summary,
    ofi_future_return_association,
    regime_outcome_summary,
)

MINUTE = 60_000_000_000


def test_intraday_liquidity_uses_fixed_reproducible_buckets() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "decision_ts_ns": [0, 30 * MINUTE, 60 * MINUTE, 90 * MINUTE],
            "spread_bps": [2.0, 4.0, 6.0, 8.0],
            "depth_total_l1": [100.0, 80.0, 60.0, 40.0],
            "queue_imbalance_l1": [0.2, 0.0, -0.2, 0.4],
        }
    )
    summary = intraday_liquidity_summary(frame, bucket_minutes=60)
    first = summary.row(0, named=True)
    assert first["intraday_bucket_label"] == "00:00"
    assert first["n_observations"] == 2
    assert first["mean_spread_bps"] == 3.0
    assert first["mean_depth_l1"] == 90.0
    assert summary.get_column("descriptive_only").all()


def test_ofi_association_and_half_life_use_supplied_horizons() -> None:
    ofi = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    orthogonal_noise = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(ofi),
            "ofi_l1": ofi,
            "return_h1": ofi,
            "return_h2": [
                value + noise for value, noise in zip(ofi, orthogonal_noise, strict=True)
            ],
            "return_h4": orthogonal_noise,
        }
    )
    association = ofi_future_return_association(
        frame,
        horizon_return_columns={1: "return_h1", 2: "return_h2", 4: "return_h4"},
    )
    at_one = association.filter(pl.col("horizon_events") == 1).row(0, named=True)
    assert at_one["pearson_correlation"] == pytest.approx(1.0)
    assert at_one["ols_slope_return_per_ofi_unit"] == pytest.approx(1.0)
    assert at_one["descriptive_only"] is True

    half_life = estimate_signal_half_life(association)
    result = half_life.summary.row(0, named=True)
    assert result["reference_horizon_events"] == 1
    assert result["first_crossing_half_life_events"] == 4.0
    assert result["analysis_kind"] == "signal_half_life_descriptive"
    assert half_life.curve.get_column("normalized_absolute_correlation")[0] == pytest.approx(1.0)


def test_large_trade_impact_requires_caller_supplied_train_threshold() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "quantity": [1.0, 2.0, 10.0, 20.0],
            "impact_h2": [1.0, 2.0, 10.0, 20.0],
        }
    )
    summary = large_trade_price_impact_summary(
        frame,
        impact_columns={2: "impact_h2"},
        train_quantity_thresholds={"BTCUSDT": 5.0},
    )
    regular = summary.filter(~pl.col("large_trade")).row(0, named=True)
    large = summary.filter(pl.col("large_trade")).row(0, named=True)
    assert regular["mean_signed_impact_bps"] == 1.5
    assert large["mean_signed_impact_bps"] == 15.0
    assert large["train_quantity_threshold"] == 5.0
    assert large["threshold_source"] == "caller_supplied_train_period"


def test_liquidity_recovery_tracks_one_episode_and_censors_segment_tail() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "continuity_id": ["a"] * 6,
            "decision_ts_ns": list(range(6)),
            "decision_sequence": list(range(1, 7)),
            "spread_bps": [2.0, 10.0, 8.0, 4.0, 3.0, 9.0],
            "depth_total_l1": [100.0, 40.0, 60.0, 90.0, 100.0, 30.0],
        }
    )
    summary = liquidity_recovery_summary(
        frame,
        train_thresholds={
            "BTCUSDT": LiquidityShockThresholds(
                spread_shock_bps=8.0,
                depth_shock_max=50.0,
                spread_recovery_bps=4.0,
                depth_recovery_min=80.0,
                max_recovery_events=3,
            )
        },
    )
    assert summary.height == 2
    recovered = summary.row(0, named=True)
    assert recovered["shock_sequence"] == 2
    assert recovered["recovery_events"] == 2
    assert recovered["recovery_time_ns"] == 2
    assert recovered["recovery_right_censored"] is False
    assert recovered["threshold_source"] == "caller_supplied_train_period"

    tail = summary.row(1, named=True)
    assert tail["shock_sequence"] == 6
    assert tail["recovered"] is None
    assert tail["recovery_right_censored"] is True
    assert tail["recovery_information_end_ts_ns"] is None


def test_liquidity_recovery_infers_late_censor_status_without_row_limit() -> None:
    row_count = 201
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * row_count,
            "continuity_id": ["a"] * row_count,
            "decision_ts_ns": list(range(row_count)),
            "decision_sequence": list(range(row_count)),
            "spread_bps": [10.0 if index % 2 == 0 else 2.0 for index in range(row_count)],
            "depth_total_l1": [40.0 if index % 2 == 0 else 100.0 for index in range(row_count)],
        }
    )

    summary = liquidity_recovery_summary(
        frame,
        train_thresholds={
            "BTCUSDT": LiquidityShockThresholds(
                spread_shock_bps=8.0,
                depth_shock_max=50.0,
                spread_recovery_bps=4.0,
                depth_recovery_min=80.0,
                max_recovery_events=1,
            )
        },
    )

    assert summary.height == 101
    assert summary.get_column("recovery_censor_reason")[-1] == ("segment_ends_before_max_horizon")


def test_regimes_are_assigned_from_supplied_train_boundaries() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "volatility": [0.1, 0.5, 0.9],
            "spread_bps": [1.0, 3.0, 6.0],
            "depth_total_l1": [120.0, 80.0, 40.0],
            "future_return": [0.01, 0.0, -0.02],
        }
    )
    thresholds = {
        "BTCUSDT": RegimeThresholds(
            volatility_low=0.2,
            volatility_high=0.8,
            spread_tight_bps=2.0,
            spread_wide_bps=5.0,
            depth_low=50.0,
            depth_high=100.0,
        )
    }
    regimes = assign_market_regimes(
        frame, train_thresholds=thresholds, volatility_column="volatility"
    )
    assert regimes.get_column("joint_market_regime").to_list() == [
        "low__liquid",
        "medium__normal",
        "high__stressed",
    ]
    assert regimes.get_column("regime_threshold_source").unique().to_list() == [
        "caller_supplied_train_period"
    ]
    outcomes = regime_outcome_summary(regimes, outcome_columns=("future_return",))
    assert outcomes.get_column("n_observations").sum() == 3
    assert outcomes.get_column("descriptive_only").all()


def test_cross_instrument_and_feature_stability_are_descriptive() -> None:
    effects = pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "horizon_events": [1, 1, 2, 2],
            "effect": [0.2, 0.1, -0.1, -0.2],
        }
    )
    cross = cross_instrument_stability_summary(effects, value_column="effect")
    assert cross.get_column("sign_agreement_fraction").to_list() == [1.0, 1.0]
    assert cross.get_column("n_instruments").to_list() == [2, 2]

    reference = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "stable": [0.0, 1.0, 2.0, 3.0],
            "shifted": [0.0, 1.0, 2.0, 3.0],
        }
    )
    comparison = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "stable": [0.0, 1.0, 2.0, 3.0],
            "shifted": [2.0, 3.0, 4.0, 5.0],
        }
    )
    stability = feature_stability_summary(
        reference,
        comparison,
        feature_columns=("stable", "shifted"),
        bins=2,
    )
    stable = stability.filter(pl.col("feature") == "stable").row(0, named=True)
    shifted = stability.filter(pl.col("feature") == "shifted").row(0, named=True)
    assert stable["population_stability_index"] == pytest.approx(0.0)
    assert shifted["population_stability_index"] > 0
    assert shifted["standardized_mean_shift"] > 1.0
    assert shifted["bin_source"] == "reference_period_only"
    assert shifted["descriptive_only"] is True
