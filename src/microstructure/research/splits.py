"""Purged expanding walk-forward splits for overlapping event labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from microstructure.config import EvaluationConfig


class SplitError(ValueError):
    """Raised when a leakage-safe walk-forward plan cannot be constructed."""


IndexArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PurgedFold:
    """One expanding training window and its strictly later validation window."""

    fold_id: int
    train_indices: IndexArray
    validation_indices: IndexArray
    train_start_ts_ns: int
    train_end_ts_ns: int
    validation_start_ts_ns: int
    validation_end_ts_ns: int
    purged_rows: int
    embargoed_time_buckets: int


@dataclass(frozen=True, slots=True)
class WalkForwardPlan:
    """Development folds plus a frozen, never-used-for-selection final test."""

    folds: tuple[PurgedFold, ...]
    final_train_indices: IndexArray
    test_indices: IndexArray
    test_start_ts_ns: int
    test_end_ts_ns: int
    decision_time_count: int


_REQUIRED = frozenset(
    {
        "decision_ts_ns",
        "label_information_end_ts_ns",
        "right_censored",
    }
)


def _require_contract(frame: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise SplitError(f"research frame is missing split columns: {missing}")
    if frame.is_empty():
        raise SplitError("research frame must not be empty")


def _indices(frame: pl.DataFrame, condition: pl.Expr) -> IndexArray:
    return frame.filter(condition).get_column("_research_row_id").to_numpy().astype(np.int64)


def _eligible(frame: pl.DataFrame) -> pl.Expr:
    ready = pl.col("feature_ready") if "feature_ready" in frame.columns else pl.lit(True)
    return (~pl.col("right_censored")) & ready


def _purged_training_indices(
    indexed: pl.DataFrame,
    *,
    validation_start_ts_ns: int,
    decision_cutoff_ts_ns: int,
) -> tuple[IndexArray, int]:
    candidates = indexed.filter(
        _eligible(indexed) & (pl.col("decision_ts_ns") < validation_start_ts_ns)
    )
    safe = candidates.filter(
        (pl.col("decision_ts_ns") < decision_cutoff_ts_ns)
        & (pl.col("label_information_end_ts_ns") < validation_start_ts_ns)
    )
    return (
        safe.get_column("_research_row_id").to_numpy().astype(np.int64),
        candidates.height - safe.height,
    )


def expanding_walk_forward_splits(
    frame: pl.DataFrame,
    config: EvaluationConfig,
) -> WalkForwardPlan:
    """Create global-time expanding folds and a frozen final-test period.

    Configuration counts refer to unique decision-time buckets, not physical
    rows, so instruments sharing a timestamp always remain in the same split.
    Candidate training decisions are separated by ``embargo_events`` buckets;
    label intervals ending at or beyond the evaluation start are purged as an
    independent second guard. Development labels ending at or beyond the final
    test start are also excluded so model selection cannot observe test outcomes.
    """

    _require_contract(frame)
    if config.min_train_events < 1:
        raise SplitError("min_train_events must be positive")
    if config.validation_events < 1 or config.test_events < 1:
        raise SplitError("validation_events and test_events must be positive")
    if config.step_events < 1 or config.embargo_events < 0:
        raise SplitError("step_events must be positive and embargo_events nonnegative")

    indexed = frame.with_row_index("_research_row_id")
    decision_times = sorted(indexed.get_column("decision_ts_ns").unique().to_list())
    required_times = config.min_train_events + config.validation_events + config.test_events
    if len(decision_times) < required_times:
        raise SplitError(
            f"need at least {required_times} decision-time buckets, got {len(decision_times)}"
        )

    development_end = len(decision_times) - config.test_events
    test_start_position = development_end
    test_start = int(decision_times[test_start_position])
    folds: list[PurgedFold] = []
    validation_start_position = config.min_train_events
    while validation_start_position + config.validation_events <= development_end:
        validation_end_position = validation_start_position + config.validation_events
        validation_start = int(decision_times[validation_start_position])
        validation_end = int(decision_times[validation_end_position - 1])
        decision_cut_position = max(0, validation_start_position - config.embargo_events)
        decision_cutoff = int(decision_times[decision_cut_position])
        train_indices, purged_rows = _purged_training_indices(
            indexed,
            validation_start_ts_ns=validation_start,
            decision_cutoff_ts_ns=decision_cutoff,
        )
        validation_indices = _indices(
            indexed,
            _eligible(indexed)
            & (pl.col("decision_ts_ns") >= validation_start)
            & (pl.col("decision_ts_ns") <= validation_end)
            & (pl.col("label_information_end_ts_ns") < test_start),
        )
        if train_indices.size == 0:
            raise SplitError(f"fold {len(folds)} has no training rows after purge and embargo")
        if validation_indices.size == 0:
            raise SplitError(f"fold {len(folds)} has no labeled validation rows")

        train_times = indexed.filter(pl.col("_research_row_id").is_in(train_indices)).get_column(
            "decision_ts_ns"
        )
        folds.append(
            PurgedFold(
                fold_id=len(folds),
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_start_ts_ns=cast(int, train_times.min()),
                train_end_ts_ns=cast(int, train_times.max()),
                validation_start_ts_ns=validation_start,
                validation_end_ts_ns=validation_end,
                purged_rows=purged_rows,
                embargoed_time_buckets=min(config.embargo_events, validation_start_position),
            )
        )
        validation_start_position += config.step_events

    if not folds:
        raise SplitError("configuration produced no development folds")

    test_end = int(decision_times[-1])
    final_decision_cut_position = max(0, test_start_position - config.embargo_events)
    final_decision_cutoff = int(decision_times[final_decision_cut_position])
    final_train_indices, _ = _purged_training_indices(
        indexed,
        validation_start_ts_ns=test_start,
        decision_cutoff_ts_ns=final_decision_cutoff,
    )
    test_indices = _indices(
        indexed,
        _eligible(indexed)
        & (pl.col("decision_ts_ns") >= test_start)
        & (pl.col("decision_ts_ns") <= test_end),
    )
    if final_train_indices.size == 0 or test_indices.size == 0:
        raise SplitError("final train or labeled test set is empty after temporal filtering")

    return WalkForwardPlan(
        folds=tuple(folds),
        final_train_indices=final_train_indices,
        test_indices=test_indices,
        test_start_ts_ns=test_start,
        test_end_ts_ns=test_end,
        decision_time_count=len(decision_times),
    )


__all__ = [
    "PurgedFold",
    "SplitError",
    "WalkForwardPlan",
    "expanding_walk_forward_splits",
]
