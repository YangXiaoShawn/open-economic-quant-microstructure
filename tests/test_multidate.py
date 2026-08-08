from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

import microstructure.research.multidate as multidate
from microstructure.config import FeatureConfig, ModelConfig
from microstructure.research.models import ModelCandidate, build_model_candidates
from microstructure.research.multidate import (
    DATE_BOOTSTRAP_BLOCK_EVENTS,
    DATE_BOOTSTRAP_DRAWS,
    AnalysisLock,
    FinalFittedState,
    MultiDateEvaluationError,
    build_multidate_walk_forward_plan,
    evaluate_locked_multidate_tests,
    paired_date_log_loss,
    select_multidate_model,
)
from microstructure.research.trade_only import build_trade_only_research_frame


def _model_config() -> ModelConfig:
    return ModelConfig(
        selection_metric="log_loss",
        logistic_c_values=(1.0,),
        tree_max_depth_values=(2,),
        tree_min_samples_leaf=5,
    )


def _date_frame(
    study_date: str,
    study_role: str,
    *,
    invert_target: bool = False,
    rows: int = 121,
) -> pl.DataFrame:
    midnight = datetime.fromisoformat(f"{study_date}T00:00:00+00:00")
    start_ns = int(midnight.timestamp() * 1_000_000_000)
    continuity = f"BTCUSDT:{study_date}"
    records: list[dict[str, object]] = []
    for sequence in range(rows):
        decision_ts_ns = start_ns + sequence * 1_000_000_000
        decision_trade_id = 10_000 + sequence
        censored = sequence == rows - 1
        positive = sequence % 2
        if invert_target:
            positive = 1 - positive
        records.append(
            {
                "study_date": study_date,
                "study_role": study_role,
                "symbol": "BTCUSDT",
                "decision_ts_ns": decision_ts_ns,
                "decision_trade_id": decision_trade_id,
                "decision_sequence": sequence,
                "continuity_id": continuity,
                "feature_continuity_id": continuity,
                "label_continuity_id": None if censored else continuity,
                "max_feature_source_ts_ns": decision_ts_ns,
                "max_feature_source_trade_id": decision_trade_id,
                "label_start_ts_ns": decision_ts_ns,
                "label_start_trade_id": decision_trade_id,
                "label_information_end_ts_ns": (None if censored else decision_ts_ns + 750_000_000),
                "label_information_end_trade_id": (None if censored else decision_trade_id + 1),
                "feature_ready": True,
                "right_censored": censored,
                "future_trade_up": None if censored else positive,
                "signal": 1.0 if sequence % 2 else -1.0,
                "slow_feature": float(sequence) / rows,
                "sample_id": f"BTCUSDT:{study_date}:{sequence}",
            }
        )
    return pl.DataFrame(records).with_columns(
        pl.col("future_trade_up").cast(pl.Int8),
        pl.col("label_start_ts_ns").cast(pl.Int64),
        pl.col("label_start_trade_id").cast(pl.Int64),
        pl.col("label_information_end_ts_ns").cast(pl.Int64),
        pl.col("label_information_end_trade_id").cast(pl.Int64),
    )


def _study_frames() -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    development = [
        _date_frame("2024-01-03", "train"),
        _date_frame("2024-01-04", "validation"),
    ]
    tests = [
        _date_frame("2024-01-05", "primary_test"),
        _date_frame("2024-01-06", "replication_test"),
    ]
    return development, tests


def _tied_millisecond_trade_frame(study_date: str, study_role: str) -> pl.DataFrame:
    timestamp_ns = int(
        datetime.fromisoformat(f"{study_date}T00:00:00.123+00:00").timestamp() * 1_000_000_000
    )
    continuity = f"tied-millisecond:{study_date}"
    trades = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 6,
            "continuity_id": [continuity] * 6,
            "trade_id": list(range(100, 106)),
            "event_ts_ns": [timestamp_ns] * 6,
            "available_ts_ns": [timestamp_ns] * 6,
            "price": [100.0, 101.0, 100.0, 102.0, 101.0, 103.0],
            "quantity": [1.0] * 6,
            "aggressor_side": ["buy", "sell", "buy", "sell", "buy", "sell"],
        }
    )
    config = FeatureConfig(
        trade_windows=(1,),
        volatility_window=1,
        intensity_window=1,
        label_horizon_events=1,
        large_trade_quantile=0.9,
    )
    return build_trade_only_research_frame(trades, config).with_columns(
        pl.lit(study_date).alias("study_date"),
        pl.lit(study_role).alias("study_role"),
    )


def test_two_phase_lock_never_reads_test_rows_and_builds_exact_date_plan() -> None:
    development, tests = _study_frames()
    selection = select_multidate_model(
        development,
        _model_config(),
        feature_columns=("signal", "slow_feature"),
        declared_test_dates=("2024-01-05", "2024-01-06"),
        seed=17,
        calibration_bins=5,
    )

    assert selection.train_dates == ("2024-01-03",)
    assert selection.validation_date == "2024-01-04"
    assert selection.declared_test_dates == ("2024-01-05", "2024-01-06")
    assert selection.validation_comparison.height == 4
    assert selection.validation_comparison.get_column("test_rows_accessed").not_().all()
    assert selection.validation_comparison.filter(pl.col("selected_on_validation")).height == 1
    payload = selection.lock.payload()
    assert payload["test_rows_accessed_during_selection"] is False
    assert payload["test_update_policy"] == (
        "fit_once_before_primary_test; no updates through replication"
    )

    # Deliberately changing every test target cannot change a phase-one selection lock:
    # the API accepts only development frames and declared date identities.
    mutated_tests = [
        _date_frame("2024-01-05", "primary_test", invert_target=True),
        _date_frame("2024-01-06", "replication_test", invert_target=True),
    ]
    repeated = select_multidate_model(
        development,
        _model_config(),
        feature_columns=("signal", "slow_feature"),
        declared_test_dates=("2024-01-05", "2024-01-06"),
        seed=17,
        calibration_bins=5,
    )
    assert repeated.lock == selection.lock
    assert repeated.selected_model == selection.selected_model
    assert (
        not tests[0]
        .get_column("future_trade_up")
        .equals(mutated_tests[0].get_column("future_trade_up"))
    )

    result = evaluate_locked_multidate_tests(development, tests, selection)
    fold = result.plan.folds[0]
    assert fold.train_indices.size == 120
    assert fold.validation_indices.size == 120
    assert result.plan.final_train_indices.size == 240
    assert result.plan.test_indices.size == 240
    assert result.predictions.height == 240
    assert result.predictions.get_column("study_date").unique().sort().to_list() == [
        "2024-01-05",
        "2024-01-06",
    ]
    assert result.predictions.get_column("is_oos").all()
    assert result.predictions.get_column("model_updated_between_test_dates").not_().all()
    assert result.predictions.get_column("selected_fit_cutoff_ts_ns").n_unique() == 1
    assert result.predictions.get_column("prior_fit_cutoff_ts_ns").n_unique() == 1
    assert result.predictions.filter(
        pl.col("selected_fit_cutoff_ts_ns") >= pl.col("decision_ts_ns")
    ).is_empty()

    paired = result.paired_log_loss
    assert paired.per_date.height == 2
    assert paired.per_date.get_column("n_blocks").to_list() == [3, 3]
    assert paired.per_date.get_column("date_weight").to_list() == [0.5, 0.5]
    assert paired.aggregate.status == "ok"
    assert paired.aggregate.n_bootstrap == DATE_BOOTSTRAP_DRAWS
    assert len(paired.aggregate.draws) == DATE_BOOTSTRAP_DRAWS
    assert paired.replication_status == "replicated"
    assert result.feature_stability.height == 4
    assert result.feature_stability.get_column("bin_source").unique().to_list() == [
        "reference_period_only"
    ]
    assert result.feature_stability.get_column("reference_dates").unique().to_list() == [
        "2024-01-03,2024-01-04"
    ]
    assert result.feature_stability.get_column("reference_only").all()


def test_lock_can_be_persisted_and_tampering_is_rejected() -> None:
    development, tests = _study_frames()
    selected = select_multidate_model(
        development,
        _model_config(),
        feature_columns=("signal", "slow_feature"),
        declared_test_dates=("2024-01-05", "2024-01-06"),
        seed=5,
        calibration_bins=5,
    )
    restored = AnalysisLock.restore(selected.lock.payload_json, selected.lock.sha256)
    result = evaluate_locked_multidate_tests(development, tests, restored)
    assert result.lock_sha256 == selected.lock.sha256
    mislabeled = [
        tests[0].with_columns(pl.lit("replication_test").alias("study_role")),
        tests[1].with_columns(pl.lit("primary_test").alias("study_role")),
    ]
    with pytest.raises(MultiDateEvaluationError, match="first declared test date"):
        evaluate_locked_multidate_tests(development, mislabeled, restored)
    with pytest.raises(MultiDateEvaluationError, match="does not match"):
        AnalysisLock.restore(selected.lock.payload_json + " ", selected.lock.sha256)


def test_final_fitted_state_is_development_only_canonical_and_hash_bound() -> None:
    development, _ = _study_frames()
    selection = select_multidate_model(
        development,
        _model_config(),
        feature_columns=("signal", "slow_feature"),
        declared_test_dates=("2024-01-05", "2024-01-06"),
        seed=17,
        calibration_bins=5,
    )

    state = selection.fitted_state
    restored = FinalFittedState.restore(state.payload_json, state.sha256)
    payload = restored.payload()
    primary_start_ns = int(datetime(2024, 1, 5, tzinfo=UTC).timestamp() * 1_000_000_000)
    assert restored == state
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == state.payload_json
    assert payload["serialization_format"] == "canonical-json-numeric-v1"
    assert set(payload["library_versions"]) == {"numpy", "scikit_learn"}
    assert payload["development_frame_sha256"] == selection.development_frame_sha256
    assert payload["fit_cutoff_ts_ns"] < primary_start_ns
    assert selection.lock.payload()["final_fitted_state_sha256"] == state.sha256

    changed = json.loads(state.payload_json)
    changed["library_versions"]["numpy"] = "tampered"
    changed_json = json.dumps(changed, sort_keys=True, separators=(",", ":"))
    with pytest.raises(MultiDateEvaluationError, match="does not match"):
        FinalFittedState.restore(changed_json, state.sha256)

    changed_lock = json.loads(selection.lock.payload_json)
    changed_lock["final_fitted_state"]["library_versions"]["numpy"] = "tampered"
    rewritten = AnalysisLock.create(changed_lock)
    with pytest.raises(MultiDateEvaluationError, match="hash does not match"):
        evaluate_locked_multidate_tests(development, _study_frames()[1], rewritten)


def test_locked_test_evaluation_invokes_no_fit_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development, tests = _study_frames()
    selection = select_multidate_model(
        development,
        _model_config(),
        feature_columns=("signal", "slow_feature"),
        declared_test_dates=("2024-01-05", "2024-01-06"),
        seed=17,
        calibration_bins=5,
    )
    restored = AnalysisLock.restore(selection.lock.payload_json, selection.lock.sha256)

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("test evaluation must not fit or recalibrate")

    monkeypatch.setattr(multidate, "make_classifier", forbidden_fit)
    monkeypatch.setattr(multidate.SigmoidCalibrator, "fit", forbidden_fit)

    result = evaluate_locked_multidate_tests(development, tests, restored)
    assert result.predictions.height == 240


def test_every_candidate_numeric_state_matches_original_sklearn_predictions() -> None:
    development, _ = _study_frames()
    combined = multidate._combine_date_frames(
        development,
        allowed_roles=multidate._DEVELOPMENT_ROLES,
        label="development fixture",
    )
    eligible = combined.filter(multidate._eligible() & pl.col("future_trade_up").is_not_null())
    matrices = multidate._fit_matrices(
        eligible,
        eligible.head(37),
        features=("signal", "slow_feature"),
        target="future_trade_up",
        calibration_fraction=0.2,
    )
    candidates = build_model_candidates(_model_config())
    assert {candidate.family for candidate in candidates} == {
        "baseline",
        "logistic",
        "logistic_l2",
        "shallow_tree",
    }
    for candidate in candidates:
        outcome = multidate._fit_candidate(
            candidate,
            matrices,
            seed=31,
            state_role="selected",
        )
        assert outcome.fitted_state is not None
        raw, calibrated = multidate._predict_serialized_model(
            outcome.fitted_state,
            matrices.x_evaluate,
        )
        assert raw == pytest.approx(outcome.raw_probability, abs=1e-12)
        assert calibrated == pytest.approx(outcome.probability, abs=1e-12)


def test_single_class_fallback_state_preserves_one_class_prior() -> None:
    development, _ = _study_frames()
    development = [
        frame.with_columns(
            pl.when(pl.col("right_censored"))
            .then(None)
            .otherwise(1)
            .cast(pl.Int8)
            .alias("future_trade_up")
        )
        for frame in development
    ]
    combined = multidate._combine_date_frames(
        development,
        allowed_roles=multidate._DEVELOPMENT_ROLES,
        label="single-class development fixture",
    )
    eligible = combined.filter(multidate._eligible() & pl.col("future_trade_up").is_not_null())
    matrices = multidate._fit_matrices(
        eligible,
        eligible.head(11),
        features=("signal", "slow_feature"),
        target="future_trade_up",
        calibration_fraction=0.2,
    )
    outcome = multidate._fit_candidate(
        ModelCandidate("logistic_l2_fixture", "logistic_l2", c=1.0),
        matrices,
        seed=9,
        state_role="selected",
    )
    assert outcome.fitted_state is not None
    classifier = outcome.fitted_state["classifier"]
    assert classifier == {
        "kind": "prior",
        "classes": [1],
        "class_probabilities": [1.0],
    }
    raw, calibrated = multidate._predict_serialized_model(
        outcome.fitted_state,
        matrices.x_evaluate,
    )
    assert raw == pytest.approx(np.ones(matrices.x_evaluate.shape[0]))
    assert calibrated == pytest.approx(np.full(matrices.x_evaluate.shape[0], 1.0 - 1e-12))


@pytest.mark.parametrize("kind", ["label", "continuity"])
def test_date_local_label_and_lookback_lineage_fail_closed(kind: str) -> None:
    development, _ = _study_frames()
    if kind == "label":
        next_date_ns = int(datetime(2024, 1, 4, tzinfo=UTC).timestamp() * 1_000_000_000)
        development[0] = development[0].with_columns(
            pl.when(pl.col("decision_sequence") == 0)
            .then(next_date_ns)
            .otherwise(pl.col("label_information_end_ts_ns"))
            .alias("label_information_end_ts_ns")
        )
        message = "label endpoints"
    else:
        reused = "BTCUSDT:2024-01-03"
        development[1] = development[1].with_columns(
            pl.lit(reused).alias("continuity_id"),
            pl.lit(reused).alias("feature_continuity_id"),
            pl.when(pl.col("right_censored"))
            .then(None)
            .otherwise(pl.lit(reused))
            .alias("label_continuity_id"),
        )
        message = "cannot span study dates"
    with pytest.raises(MultiDateEvaluationError, match=message):
        select_multidate_model(
            development,
            _model_config(),
            feature_columns=("signal", "slow_feature"),
            declared_test_dates=("2024-01-05", "2024-01-06"),
            seed=17,
            calibration_bins=5,
        )


def _tied_millisecond_study() -> pl.DataFrame:
    return pl.concat(
        [
            _tied_millisecond_trade_frame("2024-01-03", "train"),
            _tied_millisecond_trade_frame("2024-01-04", "validation"),
            _tied_millisecond_trade_frame("2024-01-05", "primary_test"),
            _tied_millisecond_trade_frame("2024-01-06", "replication_test"),
        ],
        how="vertical",
    )


def test_real_tied_millisecond_trade_labels_use_trade_id_boundary() -> None:
    study = _tied_millisecond_study()
    tied = study.filter(
        (pl.col("study_date") == "2024-01-05") & (pl.col("decision_trade_id") == 100)
    ).row(0, named=True)

    assert tied["label_start_ts_ns"] == tied["decision_ts_ns"]
    assert tied["label_start_trade_id"] == tied["decision_trade_id"]
    assert tied["label_information_end_ts_ns"] == tied["decision_ts_ns"]
    assert tied["label_information_end_trade_id"] > tied["decision_trade_id"]
    plan = build_multidate_walk_forward_plan(study)
    assert plan.folds[0].train_indices.size == 5
    assert plan.folds[0].validation_indices.size == 5
    assert plan.final_train_indices.size == 10
    assert plan.test_indices.size == 10


@pytest.mark.parametrize("invalid_end_trade_id", [100, 99])
def test_tied_millisecond_same_or_lower_label_end_trade_id_fails(
    invalid_end_trade_id: int,
) -> None:
    invalid = _tied_millisecond_study().with_columns(
        pl.when((pl.col("study_date") == "2024-01-05") & (pl.col("decision_trade_id") == 100))
        .then(invalid_end_trade_id)
        .otherwise(pl.col("label_information_end_trade_id"))
        .alias("label_information_end_trade_id")
    )
    with pytest.raises(MultiDateEvaluationError, match="strictly later"):
        build_multidate_walk_forward_plan(invalid)


def test_tied_millisecond_cross_date_endpoint_and_changed_start_fail() -> None:
    next_date_ns = int(
        datetime.fromisoformat("2024-01-06T00:00:00+00:00").timestamp() * 1_000_000_000
    )
    cross_date = _tied_millisecond_study().with_columns(
        pl.when((pl.col("study_date") == "2024-01-05") & (pl.col("decision_trade_id") == 100))
        .then(next_date_ns)
        .otherwise(pl.col("label_information_end_ts_ns"))
        .alias("label_information_end_ts_ns")
    )
    with pytest.raises(MultiDateEvaluationError, match="label endpoints"):
        build_multidate_walk_forward_plan(cross_date)

    changed_start = _tied_millisecond_study().with_columns(
        pl.when((pl.col("study_date") == "2024-01-05") & (pl.col("decision_trade_id") == 100))
        .then(pl.col("decision_trade_id") + 1)
        .otherwise(pl.col("label_start_trade_id"))
        .alias("label_start_trade_id")
    )
    with pytest.raises(MultiDateEvaluationError, match="label start boundary"):
        build_multidate_walk_forward_plan(changed_start)


def test_later_clock_boundary_may_reuse_the_last_observed_sequence() -> None:
    """Exact clock labels can carry a state forward to t+h without a new update."""

    development, tests = _study_frames()
    frames = [*development, *tests]
    carried = [
        frame.with_columns(
            pl.when(pl.col("right_censored"))
            .then(None)
            .otherwise(pl.col("decision_trade_id"))
            .cast(pl.Int64)
            .alias("label_information_end_trade_id")
        )
        for frame in frames
    ]

    plan = build_multidate_walk_forward_plan(pl.concat(carried, how="vertical"))

    assert plan.final_train_indices.size == 240
    assert plan.test_indices.size == 240


def test_custom_bootstrap_contract_is_persisted_and_used() -> None:
    development, tests = _study_frames()
    selection = select_multidate_model(
        development,
        _model_config(),
        feature_columns=("signal", "slow_feature"),
        declared_test_dates=("2024-01-05", "2024-01-06"),
        seed=17,
        calibration_bins=5,
        bootstrap_draws=37,
        block_width_events=7,
    )

    bootstrap = selection.lock.payload()["bootstrap"]
    assert bootstrap == {
        "block_width_events": 7,
        "date_weighting": "equal",
        "draws": 37,
        "metric": "selected_minus_historical_prior_log_loss",
    }
    result = evaluate_locked_multidate_tests(development, tests, selection)
    assert result.paired_log_loss.aggregate.n_bootstrap == 37
    assert result.paired_log_loss.per_date.get_column("n_blocks").to_list() == [18, 18]
    assert result.predictions.get_column("date_block_width_events").unique().to_list() == [7]


def _paired_prediction_fixture() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, (study_date, count) in enumerate((("2024-01-05", 85), ("2024-01-06", 125))):
        start_ns = int(
            datetime.fromisoformat(f"{study_date}T00:00:00+00:00").timestamp() * 1_000_000_000
        )
        for sequence in range(count):
            y_true = sequence % 2
            selected = 0.75 if y_true else 0.25
            if date_index:
                selected = 0.65 if y_true else 0.35
            rows.append(
                {
                    "row_id": len(rows),
                    "study_date": study_date,
                    "study_role": "primary_test" if date_index == 0 else "replication_test",
                    "test_phase": "primary" if date_index == 0 else "replication",
                    "decision_ts_ns": start_ns + sequence,
                    "decision_sequence": sequence,
                    "y_true": y_true,
                    "selected_probability": selected,
                    "prior_probability": 0.5,
                }
            )
    return pl.DataFrame(rows)


def _naive_equal_date_draws(predictions: pl.DataFrame, *, seed: int, draws: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    per_date: list[np.ndarray] = []
    for study_date in sorted(predictions.get_column("study_date").unique().to_list()):
        current = predictions.filter(pl.col("study_date") == study_date).sort(
            "decision_ts_ns", "decision_sequence", "row_id"
        )
        y_true = current.get_column("y_true").to_numpy().astype(np.int64)
        selected = current.get_column("selected_probability").to_numpy()
        prior = current.get_column("prior_probability").to_numpy()
        selected_loss = -(y_true * np.log(selected) + (1 - y_true) * np.log(1 - selected))
        prior_loss = -(y_true * np.log(prior) + (1 - y_true) * np.log(1 - prior))
        differences = selected_loss - prior_loss
        block_index = np.arange(current.height) // DATE_BOOTSTRAP_BLOCK_EVENTS
        block_count = int(block_index.max()) + 1
        sums = np.asarray([differences[block_index == index].sum() for index in range(block_count)])
        counts = np.asarray(
            [(block_index == index).sum() for index in range(block_count)], dtype=np.int64
        )
        date_draws = np.empty(draws)
        for draw in range(draws):
            sampled = random.integers(0, block_count, size=block_count)
            date_draws[draw] = sums[sampled].sum() / counts[sampled].sum()
        per_date.append(date_draws)
    return np.mean(np.vstack(per_date), axis=0)


def test_block_sufficient_statistic_bootstrap_matches_naive_rows_exactly() -> None:
    predictions = _paired_prediction_fixture()
    seed = 91
    result = paired_date_log_loss(
        predictions,
        seed=seed,
        draw_chunk_size=7,
    )
    naive = _naive_equal_date_draws(
        predictions,
        seed=seed,
        draws=DATE_BOOTSTRAP_DRAWS,
    )

    assert result.aggregate.status == "ok"
    assert result.aggregate.n_bootstrap == 2_000
    assert result.aggregate.n_blocks == 7
    assert np.asarray(result.aggregate.draws) == pytest.approx(naive, abs=1e-15)
    assert result.aggregate.point_estimate == pytest.approx(
        result.per_date.get_column("point_delta").mean()
    )
    pooled = (
        result.per_date.get_column("point_delta") * result.per_date.get_column("n_obs")
    ).sum() / result.per_date.get_column("n_obs").sum()
    assert result.aggregate.point_estimate != pytest.approx(pooled)
    assert result.predictions.get_column("date_block_width_events").unique().to_list() == [40]
