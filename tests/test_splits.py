from __future__ import annotations

import polars as pl
import pytest

from microstructure.config import EvaluationConfig
from microstructure.research.splits import SplitError, expanding_walk_forward_splits


def _config(*, embargo: int = 2) -> EvaluationConfig:
    return EvaluationConfig(
        min_train_events=6,
        validation_events=3,
        test_events=3,
        step_events=3,
        embargo_events=embargo,
        bootstrap_samples=50,
        calibration_bins=5,
    )


def _interval_frame(rows: int = 16) -> pl.DataFrame:
    decision = list(range(rows))
    censored = [index + 2 >= rows for index in decision]
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * rows,
            "decision_ts_ns": decision,
            "label_information_end_ts_ns": [
                None if is_censored else index + 2
                for index, is_censored in zip(decision, censored, strict=True)
            ],
            "right_censored": censored,
        }
    )


def test_expanding_folds_purge_overlap_and_apply_embargo() -> None:
    frame = _interval_frame()
    plan = expanding_walk_forward_splits(frame, _config())

    assert len(plan.folds) == 2
    first = plan.folds[0]
    assert first.validation_start_ts_ns == 6
    assert first.validation_end_ts_ns == 8
    assert first.train_indices.tolist() == [0, 1, 2, 3]
    assert first.validation_indices.tolist() == [6, 7, 8]
    assert first.purged_rows == 2
    assert first.embargoed_time_buckets == 2

    second = plan.folds[1]
    assert second.train_indices.tolist() == list(range(7))
    assert second.validation_indices.tolist() == [9, 10]
    assert set(first.train_indices).issubset(set(second.train_indices))

    for fold in plan.folds:
        train = frame[fold.train_indices]
        validation = frame[fold.validation_indices]
        assert train.get_column("label_information_end_ts_ns").max() < fold.validation_start_ts_ns
        assert validation.get_column("label_information_end_ts_ns").max() < plan.test_start_ts_ns
        assert fold.train_end_ts_ns < fold.validation_start_ts_ns


def test_final_period_is_frozen_and_never_enters_development() -> None:
    plan = expanding_walk_forward_splits(_interval_frame(), _config())
    assert plan.test_start_ts_ns == 13
    assert plan.test_end_ts_ns == 15
    # The final two decisions are censored; only t=13 has a complete two-event target.
    assert plan.test_indices.tolist() == [13]
    development_validation = {
        int(index) for fold in plan.folds for index in fold.validation_indices
    }
    assert development_validation.isdisjoint(set(plan.test_indices))
    assert max(plan.final_train_indices) < min(plan.test_indices)


def test_validation_labels_cannot_end_inside_final_test() -> None:
    frame = _interval_frame(rows=50)
    config = EvaluationConfig(
        min_train_events=20,
        validation_events=10,
        test_events=10,
        step_events=10,
        embargo_events=2,
        bootstrap_samples=50,
        calibration_bins=5,
    )
    plan = expanding_walk_forward_splits(frame, config)

    assert plan.test_start_ts_ns == 40
    last_validation = frame.with_row_index("row_id").filter(
        pl.col("row_id").is_in(plan.folds[-1].validation_indices)
    )
    assert last_validation.get_column("decision_ts_ns").to_list() == list(range(30, 38))
    assert last_validation.get_column("label_information_end_ts_ns").max() < plan.test_start_ts_ns


def test_same_timestamp_instruments_stay_in_the_same_fold() -> None:
    base = _interval_frame()
    eth = base.with_columns(pl.lit("ETHUSDT").alias("symbol"))
    pooled = pl.concat([base, eth]).sort(["decision_ts_ns", "symbol"])
    plan = expanding_walk_forward_splits(pooled, _config())
    first_validation = pooled.with_row_index("row_id").filter(
        pl.col("row_id").is_in(plan.folds[0].validation_indices)
    )
    counts = first_validation.group_by("decision_ts_ns").len().get_column("len")
    assert counts.to_list() == [2, 2, 2]


def test_too_short_sample_fails_closed() -> None:
    with pytest.raises(SplitError, match="need at least"):
        expanding_walk_forward_splits(_interval_frame(rows=10), _config())
