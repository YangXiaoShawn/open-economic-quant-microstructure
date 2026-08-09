from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from microstructure.config import ModelConfig
from microstructure.m8_l2_analysis_config import (
    M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256,
    M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256,
)
from microstructure.research import l2_evaluation
from microstructure.research.l2_evaluation import (
    L2ExecutionReference,
    L2HeldoutEndpointFrame,
    L2LockedEvaluationError,
    LockedL2EndpointState,
    evaluate_locked_l2_endpoints,
    run_locked_l2_market_execution,
)
from microstructure.research.l2_multidate import L2EndpointSpec
from microstructure.research.multidate import FinalFittedState, select_multidate_model

_AGGREGATE_SHA = "a" * 64
_REGIME_SHA = "b" * 64
_ENDPOINTS = (
    L2EndpointSpec("event_20", "event", 20, "events", 40, None, 20),
    L2EndpointSpec("event_100", "event", 100, "events", 200, None, 100),
    L2EndpointSpec("clock_1000ms", "clock", 1_000, "milliseconds", None, 2_000, 20),
    L2EndpointSpec("clock_5000ms", "clock", 5_000, "milliseconds", None, 10_000, 100),
)


def _date_start_ns(study_date: str) -> int:
    return int(datetime.fromisoformat(f"{study_date}T00:00:00+00:00").timestamp() * 1_000_000_000)


def _development_frame(study_date: str, role: str, *, rows: int = 120) -> pl.DataFrame:
    start = _date_start_ns(study_date) + 1_000_000_000
    continuity = f"{study_date}::development"
    records: list[dict[str, object]] = []
    for index in range(rows):
        decision = start + index * 10_000_000
        sequence = index + 1
        target = index % 2
        records.append(
            {
                "study_date": study_date,
                "study_role": role,
                "symbol": "BTCUSDT",
                "decision_ts_ns": decision,
                "decision_trade_id": sequence,
                "decision_sequence": sequence,
                "continuity_id": continuity,
                "feature_continuity_id": continuity,
                "label_continuity_id": continuity,
                "max_feature_source_ts_ns": decision,
                "max_feature_source_trade_id": sequence,
                "label_start_ts_ns": decision,
                "label_start_trade_id": sequence,
                "label_information_end_ts_ns": decision + 1_000_000,
                "label_information_end_trade_id": sequence + 1,
                "feature_ready": True,
                "right_censored": False,
                "signal": 5.0 if target else -5.0,
                "future_mid_up": target,
            }
        )
    return pl.DataFrame(records, infer_schema_length=None)


@pytest.fixture(scope="module")
def fitted_state() -> FinalFittedState:
    selected = select_multidate_model(
        (
            _development_frame("2026-08-10", "train"),
            _development_frame("2026-08-11", "validation"),
        ),
        ModelConfig(
            selection_metric="log_loss",
            logistic_c_values=(1.0,),
            tree_max_depth_values=(2,),
            tree_min_samples_leaf=4,
        ),
        feature_columns=("signal",),
        target="future_mid_up",
        declared_test_dates=("2026-08-12", "2026-08-13"),
        seed=20260807,
        calibration_bins=10,
    )
    assert selected.selected_model != "historical_prior"
    return selected.fitted_state


def _locked_states(
    fitted_state: FinalFittedState,
    endpoints: tuple[L2EndpointSpec, ...] = _ENDPOINTS,
) -> tuple[LockedL2EndpointState, ...]:
    return tuple(
        LockedL2EndpointState(
            symbol="BTCUSDT",
            endpoint=endpoint,
            child_lock_sha256=(f"{index + 1:064x}"),
            aggregate_lock_sha256=_AGGREGATE_SHA,
            regime_thresholds_sha256=_REGIME_SHA,
            fitted_state=fitted_state,
        )
        for index, endpoint in enumerate(endpoints)
    )


def _heldout_frame(
    endpoint: L2EndpointSpec,
    *,
    study_date: str,
    study_role: str,
    aligned: bool,
    rows_per_interval: int = 50,
) -> pl.DataFrame:
    date_start = _date_start_ns(study_date)
    records: list[dict[str, object]] = []
    for interval_index in range(2):
        interval_start = date_start + (interval_index + 1) * 10_000_000_000
        interval_end = interval_start + 6_000_000_000
        interval_id = f"{study_date}::observed-{interval_index}"
        for local_index in range(rows_per_interval):
            row_index = interval_index * rows_per_interval + local_index
            decision = interval_start + local_index * 100_000_000
            sequence = row_index + 1
            signal = 5.0 if aligned else -5.0
            records.append(
                {
                    "study_date": study_date,
                    "study_role": study_role,
                    "endpoint_name": endpoint.name,
                    "endpoint_domain": endpoint.domain,
                    "endpoint_horizon_value": endpoint.horizon_value,
                    "endpoint_horizon_unit": endpoint.horizon_unit,
                    "symbol": "BTCUSDT",
                    "continuity_id": interval_id,
                    "observed_interval_id": interval_id,
                    "observed_interval_start_ns": interval_start,
                    "observed_interval_end_ns_exclusive": interval_end,
                    "decision_ts_ns": decision,
                    "decision_sequence": sequence,
                    "feature_cutoff_ts_ns": decision,
                    "max_feature_source_ts_ns": decision,
                    "max_feature_source_sequence": sequence,
                    "feature_continuity_id": interval_id,
                    "label_start_ts_ns": decision,
                    "label_start_sequence": sequence,
                    "right_censored": False,
                    "future_mid_return": 0.001,
                    "future_mid_up": 1,
                    "label_information_end_ts_ns": decision + 1_000_000,
                    "label_information_end_sequence": sequence + 1,
                    "label_continuity_id": interval_id,
                    "ofi_signed_future_mid_markout_bps": 1.25,
                    "sample_id": (f"BTCUSDT::{study_date}::{endpoint.name}::{sequence}"),
                    "feature_ready": True,
                    "signal": signal,
                    "volatility_regime": "low",
                    "liquidity_regime": "liquid",
                    "joint_market_regime": "low__liquid",
                    "best_bid": 99.99,
                    "best_ask": 100.01,
                    "bid_quantity": 0.4,
                    "ask_quantity": 0.4,
                    "mid_price": 100.0,
                    "tick_size": 0.01,
                    "lot_size": 0.1,
                }
            )
    return pl.DataFrame(records, infer_schema_length=None)


def _heldout_frames(
    endpoints: tuple[L2EndpointSpec, ...] = _ENDPOINTS,
    *,
    replication_aligned: bool = True,
) -> tuple[L2HeldoutEndpointFrame, ...]:
    values: list[L2HeldoutEndpointFrame] = []
    for endpoint in endpoints:
        for study_date, role, aligned in (
            ("2026-08-12", "primary_test", True),
            ("2026-08-13", "replication_test", replication_aligned),
        ):
            values.append(
                L2HeldoutEndpointFrame(
                    symbol="BTCUSDT",
                    endpoint_name=endpoint.name,
                    study_date=study_date,
                    study_role=role,  # type: ignore[arg-type]
                    frame=_heldout_frame(
                        endpoint,
                        study_date=study_date,
                        study_role=role,
                        aligned=aligned,
                    ),
                )
            )
    return tuple(values)


def _forbid_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("locked evaluation must never fit")

    classifiers: tuple[type[object], ...] = (
        LogisticRegression,
        DecisionTreeClassifier,
        DummyClassifier,
    )
    for classifier in classifiers:
        monkeypatch.setattr(classifier, "fit", fail)


def _assert_strict_json_numbers(frame: pl.DataFrame) -> None:
    for row in frame.to_dicts():
        for value in row.values():
            if isinstance(value, float):
                assert math.isfinite(value)


def test_locked_evaluation_never_fits_and_uses_exact_endpoint_blocks(
    fitted_state: FinalFittedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _locked_states(fitted_state)
    frames = _heldout_frames()
    _forbid_fit(monkeypatch)

    result = evaluate_locked_l2_endpoints(states, frames)

    assert result.predictions.height == 800
    assert result.predictions["sample_id"].n_unique() == result.predictions.height
    assert {
        "selected_raw_probability",
        "selected_probability",
        "prior_raw_probability",
        "prior_probability",
    }.issubset(result.predictions.columns)
    assert set(result.predictions["aggregate_lock_sha256"].unique()) == {_AGGREGATE_SHA}
    assert not bool(result.predictions["test_used_for_selection"].any())
    assert not bool(result.predictions["model_updated_between_test_dates"].any())

    overall = result.paired_by_session_regime.filter(
        (pl.col("study_role") == "primary_test") & (pl.col("regime") == "ALL")
    )
    observed = {
        str(row["endpoint_name"]): (
            int(row["block_width"]),
            str(row["block_unit"]),
            int(row["n_blocks"]),
        )
        for row in overall.to_dicts()
    }
    assert observed == {
        "event_20": (40, "events", 22),
        "event_100": (200, "events", 0),
        "clock_1000ms": (2_000, "milliseconds", 82),
        "clock_5000ms": (10_000, "milliseconds", 0),
    }
    assert set(overall["samples"].unique()) == {2_000}
    assert not bool(result.paired_by_session_regime["p_value_computed"].any())
    assert result.paired_by_session_regime["p_value"].null_count() == (
        result.paired_by_session_regime.height
    )
    assert not bool(result.paired_by_session_regime["cross_symbol_pooling"].any())

    empty_regime = result.paired_by_session_regime.filter(
        (pl.col("endpoint_name") == "event_20") & (pl.col("regime") == "medium__normal")
    )
    assert empty_regime.height == 2
    assert set(empty_regime["bootstrap_status"].unique()) == {"empty_regime"}
    assert set(empty_regime["n_obs"].unique()) == {0}

    summary = result.equal_session_summary.filter(pl.col("regime") == "ALL")
    assert bool(summary["directionally_replicated"].all())
    assert set(summary["replication_status"].unique()) == {"replicated"}
    for row in summary.to_dicts():
        expected_delta = 0.5 * float(row["primary_point_delta"]) + 0.5 * float(
            row["replication_point_delta"]
        )
        assert float(row["point_delta"]) == pytest.approx(expected_delta)
        assert float(row["primary_point_delta"]) < 0.0
        assert float(row["replication_point_delta"]) < 0.0
    markout = result.signed_markout.filter(
        (pl.col("endpoint_name") == "event_20") & (pl.col("regime") == "ALL")
    )
    assert set(markout["mean_ofi_signed_future_mid_markout_bps"].unique()) == {1.25}
    assert bool(markout["descriptive_only"].all())
    assert not bool(markout["observed_trade_impact"].any())
    for frame in (
        result.predictions,
        result.predictive_metrics,
        result.paired_by_session_regime,
        result.equal_session_summary,
        result.signed_markout,
    ):
        _assert_strict_json_numbers(frame)


def _moving_block_probe_frame(*, intervals: int, width: int) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    rows_per_interval = 2 * width + 1
    for interval_index in range(intervals):
        interval_id = f"probe-{interval_index}"
        interval_start = interval_index * 1_000_000_000
        for ordinal in range(rows_per_interval):
            index = interval_index * rows_per_interval + ordinal
            records.append(
                {
                    "study_date": "2026-08-12",
                    "symbol": "BTCUSDT",
                    "continuity_id": interval_id,
                    "observed_interval_id": interval_id,
                    "observed_interval_start_ns": interval_start,
                    "observed_interval_end_ns_exclusive": interval_start + 1_000_000_000,
                    "decision_ts_ns": interval_start + ordinal * 1_000,
                    "decision_sequence": index + 1,
                    "_endpoint_event_ordinal": ordinal,
                    "joint_market_regime": "low__liquid",
                    "y_true": ordinal % 2,
                    "selected_probability": 0.15 + 0.7 * ((index % 7) / 7.0),
                    "prior_probability": 0.25 + 0.5 * ((index % 5) / 5.0),
                }
            )
    return pl.DataFrame(records, infer_schema_length=None)


def _independent_event_moving_block_draws(
    frame: pl.DataFrame,
    *,
    width: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    interval_statistics: list[
        tuple[list[tuple[float, float, int]], list[tuple[float, float, int]] | None, int]
    ] = []
    for current in frame.partition_by("observed_interval_id", maintain_order=True):
        target = current["y_true"].to_numpy().astype(np.int64, copy=False)
        selected_probability = np.clip(
            current["selected_probability"].to_numpy(), 1e-12, 1.0 - 1e-12
        )
        prior_probability = np.clip(current["prior_probability"].to_numpy(), 1e-12, 1.0 - 1e-12)
        selected = -(
            target * np.log(selected_probability) + (1 - target) * np.log1p(-selected_probability)
        )
        prior = -(target * np.log(prior_probability) + (1 - target) * np.log1p(-prior_probability))
        size = current.height
        full = [
            (
                float(selected[start : start + width].sum()),
                float(prior[start : start + width].sum()),
                width,
            )
            for start in range(size - width + 1)
        ]
        remainder = size % width
        tail = (
            [
                (
                    float(selected[start : start + remainder].sum()),
                    float(prior[start : start + remainder].sum()),
                    remainder,
                )
                for start in range(size - width + 1)
            ]
            if remainder
            else None
        )
        interval_statistics.append((full, tail, math.ceil(size / width)))
    for draw_index in range(samples):
        selected_total = 0.0
        prior_total = 0.0
        count_total = 0
        for full, tail, blocks_per_draw in interval_statistics:
            sampled = random.integers(0, len(full), size=blocks_per_draw)
            for block_index, candidate_index in enumerate(sampled):
                statistics = (
                    tail[int(candidate_index)]
                    if tail is not None and block_index == blocks_per_draw - 1
                    else full[int(candidate_index)]
                )
                selected_total += statistics[0]
                prior_total += statistics[1]
                count_total += statistics[2]
        draws[draw_index] = selected_total / count_total - prior_total / count_total
    return draws


def _clock_moving_block_probe_frame(*, intervals: int, width_ms: int) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    rows_per_interval = 2 * width_ms + 1
    for interval_index in range(intervals):
        interval_id = f"clock-probe-{interval_index}"
        interval_start = interval_index * 1_000_000_000
        interval_end = interval_start + rows_per_interval * 1_000_000
        for ordinal in range(rows_per_interval):
            index = interval_index * rows_per_interval + ordinal
            records.append(
                {
                    "study_date": "2026-08-12",
                    "symbol": "BTCUSDT",
                    "continuity_id": interval_id,
                    "observed_interval_id": interval_id,
                    "observed_interval_start_ns": interval_start,
                    "observed_interval_end_ns_exclusive": interval_end,
                    "decision_ts_ns": interval_start + ordinal * 1_000_000,
                    "decision_sequence": index + 1,
                    "_endpoint_event_ordinal": ordinal,
                    "joint_market_regime": "low__liquid",
                    "y_true": ordinal % 2,
                    "selected_probability": 0.15 + 0.7 * ((index % 7) / 7.0),
                    "prior_probability": 0.25 + 0.5 * ((index % 5) / 5.0),
                }
            )
    return pl.DataFrame(records, infer_schema_length=None)


def _independent_clock_moving_block_draws(
    frame: pl.DataFrame,
    *,
    width_ms: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    width_ns = width_ms * 1_000_000
    interval_statistics: list[tuple[list[tuple[float, float, int]], int]] = []
    for current in frame.partition_by("observed_interval_id", maintain_order=True):
        target = current["y_true"].to_numpy().astype(np.int64, copy=False)
        selected_probability = np.clip(
            current["selected_probability"].to_numpy(), 1e-12, 1.0 - 1e-12
        )
        prior_probability = np.clip(current["prior_probability"].to_numpy(), 1e-12, 1.0 - 1e-12)
        selected = -(
            target * np.log(selected_probability) + (1 - target) * np.log1p(-selected_probability)
        )
        prior = -(target * np.log(prior_probability) + (1 - target) * np.log1p(-prior_probability))
        times = current["decision_ts_ns"].to_numpy()
        interval_start = int(current["observed_interval_start_ns"][0])
        interval_end = int(current["observed_interval_end_ns_exclusive"][0])
        candidates: list[tuple[float, float, int]] = []
        for start in times:
            if int(start) + width_ns > interval_end:
                continue
            members = (times >= start) & (times < start + width_ns)
            candidates.append(
                (
                    float(selected[members].sum()),
                    float(prior[members].sum()),
                    int(np.count_nonzero(members)),
                )
            )
        interval_statistics.append(
            (candidates, math.ceil((interval_end - interval_start) / width_ns))
        )
    for draw_index in range(samples):
        selected_total = 0.0
        prior_total = 0.0
        count_total = 0
        for candidates, blocks_per_draw in interval_statistics:
            sampled = random.integers(0, len(candidates), size=blocks_per_draw)
            for candidate_index in sampled:
                statistics = candidates[int(candidate_index)]
                selected_total += statistics[0]
                prior_total += statistics[1]
                count_total += statistics[2]
        draws[draw_index] = selected_total / count_total - prior_total / count_total
    return draws


def test_event_moving_blocks_overlap_truncate_and_draw_locally_by_interval() -> None:
    width = 4
    samples = 31
    seed = 91_337
    endpoint = L2EndpointSpec("probe", "event", 2, "events", width, None, 2)

    one_interval = _moving_block_probe_frame(intervals=1, width=width)
    one = l2_evaluation._paired_delta(
        one_interval,
        endpoint,
        regime="ALL",
        samples=samples,
        seed=seed,
    )
    assert one.status == "ok"
    assert one.n_blocks == width + 2
    np.testing.assert_allclose(
        one.draws,
        _independent_event_moving_block_draws(
            one_interval,
            width=width,
            samples=samples,
            seed=seed,
        ),
        rtol=0.0,
        atol=2e-15,
    )

    sparse_regime = one_interval.with_columns(
        pl.when(pl.col("_endpoint_event_ordinal") % 2 == 0)
        .then(pl.lit("low__liquid"))
        .otherwise(pl.lit("medium__normal"))
        .alias("joint_market_regime")
    )
    sparse = l2_evaluation._paired_delta(
        sparse_regime,
        endpoint,
        regime="low__liquid",
        samples=samples,
        seed=seed,
    )
    assert sparse.n_obs == width + 1
    assert sparse.n_blocks == width + 2

    two_intervals = _moving_block_probe_frame(intervals=2, width=width)
    two = l2_evaluation._paired_delta(
        two_intervals,
        endpoint,
        regime="ALL",
        samples=samples,
        seed=seed,
    )
    assert two.n_blocks == 2 * (width + 2)
    np.testing.assert_allclose(
        two.draws,
        _independent_event_moving_block_draws(
            two_intervals,
            width=width,
            samples=samples,
            seed=seed,
        ),
        rtol=0.0,
        atol=2e-15,
    )


def test_clock_moving_blocks_use_legal_half_open_interval_local_windows() -> None:
    width_ms = 4
    samples = 31
    seed = 29_771
    endpoint = L2EndpointSpec("probe", "clock", 2, "milliseconds", None, width_ms, 2)
    frame = _clock_moving_block_probe_frame(intervals=2, width_ms=width_ms)

    result = l2_evaluation._paired_delta(
        frame,
        endpoint,
        regime="ALL",
        samples=samples,
        seed=seed,
    )

    assert result.status == "ok"
    assert result.n_blocks == 2 * (width_ms + 2)
    np.testing.assert_allclose(
        result.draws,
        _independent_clock_moving_block_draws(
            frame,
            width_ms=width_ms,
            samples=samples,
            seed=seed,
        ),
        rtol=0.0,
        atol=2e-15,
    )


def test_directional_replication_requires_negative_delta_on_both_dates(
    fitted_state: FinalFittedState,
) -> None:
    endpoint = (_ENDPOINTS[0],)
    result = evaluate_locked_l2_endpoints(
        _locked_states(fitted_state, endpoint),
        _heldout_frames(endpoint, replication_aligned=False),
    )
    row = result.equal_session_summary.filter(pl.col("regime") == "ALL").row(0, named=True)
    assert float(row["primary_point_delta"]) < 0.0
    assert float(row["replication_point_delta"]) > 0.0
    assert row["directionally_replicated"] is False
    assert row["replication_status"] == "failed_replication"


def test_reference_is_formula_checked_config_bound_and_payload_hashed() -> None:
    reference = L2ExecutionReference.create(
        symbol="BTCUSDT",
        training_date="2026-08-10",
        reference_mid_price=100.0,
        train_l1_depth_q05=20.0,
        lot_size=0.1,
        reference_quantity=1.0,
        aggregate_lock_sha256=_AGGREGATE_SHA,
    )
    assert reference.reference_price_statistic == "train_median_mid_price"
    assert reference.reference_depth_statistic == "train_q05_min_bid_ask_l1_depth"
    assert reference.analysis_config_source_sha256 == M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256
    assert reference.analysis_config_semantic_sha256 == M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256
    with pytest.raises(L2LockedEvaluationError, match="formula"):
        replace(reference, reference_quantity=0.9)
    with pytest.raises(L2LockedEvaluationError, match="payload"):
        replace(reference, train_l1_depth_q05=21.0)
    with pytest.raises(L2LockedEvaluationError, match="median midpoint"):
        replace(reference, reference_price_statistic="heldout_midpoint")


def test_market_only_execution_grid_reconciles_and_never_authorizes_claims(
    fitted_state: FinalFittedState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = (_ENDPOINTS[0],)
    states = _locked_states(fitted_state, endpoint)
    frames = _heldout_frames(endpoint)
    evaluation = evaluate_locked_l2_endpoints(states, frames)
    reference = L2ExecutionReference.create(
        symbol="BTCUSDT",
        training_date="2026-08-10",
        reference_mid_price=100.0,
        train_l1_depth_q05=20.0,
        lot_size=0.1,
        reference_quantity=1.0,
        aggregate_lock_sha256=_AGGREGATE_SHA,
    )
    _forbid_fit(monkeypatch)

    execution = run_locked_l2_market_execution(evaluation, frames, (reference,))

    assert execution.metrics.height == 18
    assert set(execution.metrics["decision_latency_events"].unique()) == {0, 1, 5}
    assert set(execution.metrics["order_latency_events"].unique()) == {0, 1, 5}
    assert set(execution.orders["order_type"].drop_nulls().unique()) == {"market"}
    assert set(execution.fills["liquidity"].drop_nulls().unique()) == {"taker"}
    assert float(cast(Any, execution.fills["quantity"].max())) <= 0.4 + 1e-12
    assert "partially_filled_canceled" in set(execution.orders["status"].unique())
    assert "canceled_continuity_gap" in set(execution.orders["status"].unique())
    assert "forced_liquidation" in set(execution.orders["status"].unique())
    assert float(cast(Any, execution.metrics["maximum_absolute_inventory"].max())) <= 10.0 + 1e-12
    for column in ("fill_ratio", "fill_ratio_requested", "partial_fill_order_ratio"):
        observed = execution.metrics.get_column(column).drop_nulls()
        assert len(observed) > 0
        assert bool(((observed >= 0.0) & (observed <= 1.0)).all())

    first_order = execution.orders.group_by("scenario_id").agg(
        pl.col("order_id").min().alias("first_order_id")
    )
    assert set(first_order["first_order_id"].unique()) == {1}
    strategy_fills = (
        execution.fills.filter(~pl.col("forced_liquidation"))
        .group_by("scenario_id")
        .agg(
            pl.col("quantity").sum().alias("ledger_filled_quantity"),
            pl.col("fee").sum().alias("ledger_strategy_fees"),
        )
    )
    reconciled = execution.metrics.join(strategy_fills, on="scenario_id", how="left")
    assert bool(
        reconciled.select(
            (pl.col("filled_quantity") - pl.col("ledger_filled_quantity")).abs().max().le(1e-12)
        ).item()
    )
    fee_reconciliation = execution.fills.group_by("scenario_id").agg(
        pl.col("fee").sum().alias("ledger_total_fees")
    )
    reconciled = execution.metrics.join(fee_reconciliation, on="scenario_id", how="left")
    assert bool(
        reconciled.select(
            (pl.col("total_fees") - pl.col("ledger_total_fees")).abs().max().le(1e-12)
        ).item()
    )
    assert bool(
        execution.fills.select(
            (pl.col("fee") - pl.col("notional") * pl.lit(4.0 / 10_000.0)).abs().max().le(1e-12)
        ).item()
    )
    assert bool(
        execution.metrics.select(
            (pl.col("net_pnl") - (pl.col("gross_pnl") - pl.col("total_fees"))).abs().max().le(1e-12)
        ).item()
    )

    for frame in (execution.metrics, execution.assumptions):
        assert not bool(frame["capacity_claim_authorized"].any())
        assert not bool(frame["realized_execution_claim_authorized"].any())
        assert not bool(frame["profitability_claim_authorized"].any())
        assert set(frame["aggregate_lock_sha256"].unique()) == {_AGGREGATE_SHA}
    for frame in (
        execution.orders,
        execution.fills,
        execution.positions,
        execution.metrics,
        execution.assumptions,
    ):
        _assert_strict_json_numbers(frame)
    assert set(execution.assumptions["limit_fill_model"].unique()) == {"NOT_RUN"}
    assert set(execution.assumptions["capacity_sensitivity"].unique()) == {"NOT_RUN"}
    assert set(execution.assumptions["reference_price_statistic"].unique()) == {
        "train_median_mid_price"
    }
    assert set(execution.assumptions["reference_depth_statistic"].unique()) == {
        "train_q05_min_bid_ask_l1_depth"
    }
    with pytest.raises(L2LockedEvaluationError, match="latency grids"):
        run_locked_l2_market_execution(
            evaluation,
            frames,
            (reference,),
            decision_latency_events=(0, 1, 4),
        )


def test_tampered_endpoint_width_and_observed_interval_identity_fail_closed(
    fitted_state: FinalFittedState,
) -> None:
    wrong = L2EndpointSpec("event_20", "event", 20, "events", 41, None, 20)
    with pytest.raises(L2LockedEvaluationError, match="frozen M8 L2 endpoint"):
        _locked_states(fitted_state, (wrong,))

    endpoint = (_ENDPOINTS[0],)
    states = _locked_states(fitted_state, endpoint)
    frames = list(_heldout_frames(endpoint))
    damaged = frames[0].frame.with_columns(
        pl.lit("different-observed-interval").alias("observed_interval_id")
    )
    frames[0] = replace(frames[0], frame=damaged)
    with pytest.raises(L2LockedEvaluationError, match="observed-interval identity"):
        evaluate_locked_l2_endpoints(states, frames)
