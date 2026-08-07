from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from microstructure.config import EvaluationConfig, ModelConfig
from microstructure.research.models import (
    ModelEvaluationError,
    SigmoidCalibrator,
    block_bootstrap_metric,
    build_model_candidates,
    classification_metrics,
    evaluate_model_ladder,
    paired_block_bootstrap_difference,
)
from microstructure.research.splits import expanding_walk_forward_splits


def _evaluation_config() -> EvaluationConfig:
    return EvaluationConfig(
        min_train_events=20,
        validation_events=8,
        test_events=8,
        step_events=8,
        embargo_events=1,
        bootstrap_samples=40,
        calibration_bins=5,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        selection_metric="log_loss",
        logistic_c_values=(1.0,),
        tree_max_depth_values=(2,),
        tree_min_samples_leaf=1,
    )


def _model_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for decision in range(48):
        positive = decision % 2
        for symbol_offset, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
            censored = decision == 47
            rows.append(
                {
                    "symbol": symbol,
                    "decision_ts_ns": decision,
                    "label_information_end_ts_ns": None if censored else decision + 1,
                    "right_censored": censored,
                    "future_mid_up": None if censored else positive,
                    "future_mid_return": None if censored else (0.001 if positive else -0.001),
                    "feature_ready": True,
                    "spread_bps": 2.0 + 0.1 * symbol_offset,
                    "depth_total_l1": 20.0,
                    "queue_imbalance_l1": 0.8 if positive else -0.8,
                    "microprice_deviation_bps": 0.5 if positive else -0.5,
                    "ofi_l1": 2.0 if positive else -2.0,
                    "log_mid_return_1": 0.0001 if positive else -0.0001,
                    "ofi_w2": 3.0 if positive else -3.0,
                    "signed_trade_volume_w2": 4.0 if positive else -4.0,
                    "trade_volume_w2": 4.0,
                    "trade_count_w2": 1.0,
                    "trade_intensity_w2": 2.0,
                    "realized_volatility_w2": 0.001,
                }
            )
    return pl.DataFrame(rows).sort(["decision_ts_ns", "symbol"])


def test_model_ladder_contains_required_transparent_families() -> None:
    candidates = build_model_candidates(_model_config())
    assert [candidate.family for candidate in candidates] == [
        "baseline",
        "logistic",
        "logistic_l2",
        "shallow_tree",
    ]


def test_out_of_time_ladder_is_deterministic_and_test_is_not_selected_on() -> None:
    frame = _model_frame()
    plan = expanding_walk_forward_splits(frame, _evaluation_config())
    first = evaluate_model_ladder(
        frame,
        plan,
        _model_config(),
        seed=7,
        calibration_bins=5,
    )
    second = evaluate_model_ladder(
        frame,
        plan,
        _model_config(),
        seed=7,
        calibration_bins=5,
    )

    assert first.selected_model == second.selected_model
    assert first.comparison.equals(second.comparison)
    assert first.predictions.equals(second.predictions)
    assert first.predictions.get_column("is_oos").all()
    assert first.predictions.filter(
        pl.col("fit_cutoff_ts_ns") >= pl.col("decision_ts_ns")
    ).is_empty()
    assert set(first.comparison.get_column("split")) == {"validation", "test"}
    assert first.comparison.get_column("instrument_scope").unique().to_list() == ["POOLED"]
    assert first.comparison.filter(pl.col("split") == "test").get_column(
        "period_start_ts_ns"
    ).unique().to_list() == [40]
    assert {"sample_id", "decision_sequence"}.issubset(first.predictions.columns)
    selected = first.comparison.filter(pl.col("selected_on_validation"))
    assert selected.get_column("model").unique().to_list() == [first.selected_model]


def test_single_class_fallback_is_labeled_as_prior_and_cannot_win_selection() -> None:
    frame = _model_frame().with_columns(
        pl.when(pl.col("right_censored"))
        .then(None)
        .otherwise(1)
        .cast(pl.Int8)
        .alias("future_mid_up")
    )
    plan = expanding_walk_forward_splits(frame, _evaluation_config())
    result = evaluate_model_ladder(
        frame,
        plan,
        _model_config(),
        seed=7,
        calibration_bins=5,
    )

    fallback = result.comparison.filter(pl.col("requested_family") != "baseline")
    assert fallback.get_column("family").unique().to_list() == ["baseline"]
    assert fallback.get_column("model").str.ends_with("__prior_fallback").all()
    assert fallback.get_column("fit_status").str.contains("single_class_prior_fallback").all()
    assert result.selected_model == "historical_prior"


def test_label_columns_are_rejected_from_feature_allowlist() -> None:
    frame = _model_frame()
    plan = expanding_walk_forward_splits(frame, _evaluation_config())
    with pytest.raises(ModelEvaluationError, match="cannot be model features"):
        evaluate_model_ladder(
            frame,
            plan,
            _model_config(),
            seed=7,
            calibration_bins=5,
            features=("queue_imbalance_l1", "future_mid_return"),
        )


def test_calibration_and_metrics_have_exact_small_values() -> None:
    y_true = np.asarray([0, 1], dtype=np.int64)
    probability = np.asarray([0.1, 0.9], dtype=np.float64)
    metrics = classification_metrics(y_true, probability, calibration_bins=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.01)
    assert metrics["log_loss"] == pytest.approx(-np.log(0.9))
    assert metrics["expected_calibration_error"] == pytest.approx(0.1)

    calibrator = SigmoidCalibrator()
    calibration_y = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    raw = np.linspace(0.2, 0.8, 10, dtype=np.float64)
    calibrator.fit(calibration_y, raw)
    transformed = calibrator.transform(raw)
    assert calibrator.status == "sigmoid"
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))
    assert np.all(np.diff(transformed) > 0)


def test_block_bootstrap_is_seeded_and_paired_identity_is_zero() -> None:
    predictions = pl.DataFrame(
        {
            "row_id": list(range(8)),
            "y_true": [0, 1, 0, 1, 1, 0, 1, 0],
            "probability": [0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.6, 0.4],
            "block": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    first = block_bootstrap_metric(
        predictions,
        metric="brier_score",
        block_column="block",
        n_bootstrap=40,
        seed=11,
    )
    second = block_bootstrap_metric(
        predictions,
        metric="brier_score",
        block_column="block",
        n_bootstrap=40,
        seed=11,
    )
    assert first == second
    assert first.status == "ok"
    assert first.n_blocks == 2

    paired = paired_block_bootstrap_difference(
        predictions,
        predictions,
        metric="brier_score",
        block_column="block",
        n_bootstrap=40,
        seed=11,
    )
    assert paired.point_estimate == 0.0
    assert paired.lower == 0.0
    assert paired.upper == 0.0
    assert set(paired.draws) == {0.0}
