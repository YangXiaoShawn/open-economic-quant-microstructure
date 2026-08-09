from __future__ import annotations

from datetime import datetime

import polars as pl

from microstructure.research.l2_analysis import build_l2_descriptive_analysis

SECOND = 1_000_000_000
DATES = (
    ("2026-08-10", "train"),
    ("2026-08-11", "validation"),
    ("2026-08-12", "primary_test"),
    ("2026-08-13", "replication_test"),
)
ENDPOINTS = (
    ("event_20", "event", 20, "events"),
    ("event_100", "event", 100, "events"),
    ("clock_1000ms", "clock", 1_000, "milliseconds"),
    ("clock_5000ms", "clock", 5_000, "milliseconds"),
)
FEATURES = (
    "spread_bps",
    "depth_total_l1",
    "ofi_w20",
    "realized_volatility_w100",
)


def _frame(
    study_date: str,
    study_role: str,
    symbol: str,
    endpoint_name: str,
    domain: str,
    horizon: int,
    unit: str,
) -> pl.DataFrame:
    start = int(datetime.fromisoformat(f"{study_date}T14:00:00+00:00").timestamp()) * SECOND
    continuity = f"{study_date}::{symbol}::observed-0000"
    rows: list[dict[str, object]] = []
    for sequence in range(140):
        decision = start + sequence * SECOND
        censored = sequence >= 120
        label_end = decision + (horizon * SECOND if domain == "event" else horizon * 1_000_000)
        ofi = float((sequence % 9) - 4)
        future_return = ofi * 1e-5 + (0.25e-5 if symbol == "BTCUSDT" else -0.1e-5)
        bid_quantity = 1.0 if sequence % 19 == 0 else 5.0 + (sequence % 7) * 0.2
        ask_quantity = 1.2 if sequence % 23 == 0 else 4.5 + (sequence % 5) * 0.2
        spread = 8.0 if sequence % 17 == 0 else 1.0 + (sequence % 3) * 0.1
        rows.append(
            {
                "study_date": study_date,
                "study_role": study_role,
                "endpoint_name": endpoint_name,
                "endpoint_domain": domain,
                "endpoint_horizon_value": horizon,
                "endpoint_horizon_unit": unit,
                "symbol": symbol,
                "continuity_id": continuity,
                "observed_interval_id": continuity,
                "observed_interval_start_ns": start,
                "observed_interval_end_ns_exclusive": start + 3_600 * SECOND,
                "decision_ts_ns": decision,
                "decision_sequence": sequence,
                "feature_cutoff_ts_ns": decision,
                "max_feature_source_ts_ns": decision,
                "max_feature_source_sequence": sequence,
                "feature_continuity_id": continuity,
                "label_start_ts_ns": decision,
                "label_start_sequence": sequence,
                "right_censored": censored,
                "future_mid_return": None if censored else future_return,
                "future_mid_up": None if censored else int(future_return > 0),
                "label_information_end_ts_ns": None if censored else label_end,
                "label_information_end_sequence": None if censored else sequence + horizon,
                "label_continuity_id": None if censored else continuity,
                "ofi_signed_future_mid_markout_bps": (
                    None
                    if censored
                    else (1.0 if ofi > 0 else -1.0 if ofi < 0 else 0.0) * future_return * 10_000.0
                ),
                "signed_markout_side_source": (
                    "ofi_w20" if endpoint_name in {"event_20", "clock_1000ms"} else "ofi_w100"
                ),
                "sample_id": f"{study_date}::{symbol}::{endpoint_name}::{sequence}",
                "mid_price": 100.0 + sequence * 0.01,
                "bid_quantity": bid_quantity,
                "ask_quantity": ask_quantity,
                "spread_bps": spread,
                "depth_total_l1": bid_quantity + ask_quantity,
                "depth_total_l5": bid_quantity + ask_quantity + 10.0,
                "depth_total_l10": bid_quantity + ask_quantity + 20.0,
                "queue_imbalance_l1": (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity),
                "realized_volatility_w100": 0.001 + (sequence % 13) * 0.0001,
                "ofi_w20": ofi,
                "ofi_w100": ofi * 0.8,
                "volatility_regime": ("low" if sequence % 3 == 0 else "high"),
                "liquidity_regime": ("liquid" if sequence % 4 else "stressed"),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _all_frames() -> list[pl.DataFrame]:
    return [
        _frame(study_date, role, symbol, name, domain, horizon, unit)
        for study_date, role in DATES
        for symbol in ("BTCUSDT", "ETHUSDT")
        for name, domain, horizon, unit in ENDPOINTS
    ]


def test_l2_descriptive_outputs_are_date_symbol_endpoint_explicit() -> None:
    result = build_l2_descriptive_analysis(
        _all_frames(), feature_columns=FEATURES, stability_bins=5
    )

    assert result.intraday_liquidity.height == 8 * 3
    assert result.ofi_return_association.height == 32
    assert result.signal_half_life.height == 32
    assert result.liquidity_recovery.height == 16
    assert result.regime_diagnostics.height > 32
    assert result.feature_stability.height == 2 * 2 * 4 * len(FEATURES)
    assert result.cross_instrument_stability.height == 16
    assert result.cross_instrument_stability.get_column("cross_instrument_pooling").not_().all()
    assert set(result.ofi_return_association.get_column("interpretation").unique()) == {
        "descriptive_book_flow_markout_not_trade_impact"
    }


def test_shock_thresholds_and_stability_reference_are_development_only() -> None:
    frames = _all_frames()
    original = build_l2_descriptive_analysis(frames, feature_columns=FEATURES, stability_bins=5)
    mutated = [
        frame.with_columns(
            pl.when(pl.col("study_role").is_in(["primary_test", "replication_test"]))
            .then(pl.col("spread_bps") * 100.0)
            .otherwise(pl.col("spread_bps"))
            .alias("spread_bps")
        )
        for frame in frames
    ]
    changed = build_l2_descriptive_analysis(mutated, feature_columns=FEATURES, stability_bins=5)

    assert (
        original.liquidity_recovery.select(
            "symbol", "train_spread_q95", "train_executable_depth_q05"
        )
        .unique()
        .sort("symbol")
        .equals(
            changed.liquidity_recovery.select(
                "symbol", "train_spread_q95", "train_executable_depth_q05"
            )
            .unique()
            .sort("symbol")
        )
    )
    assert set(changed.feature_stability.get_column("reference_scope").unique()) == {
        "train_plus_validation_only"
    }
