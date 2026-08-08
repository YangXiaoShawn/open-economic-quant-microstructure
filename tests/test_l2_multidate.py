from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest

from microstructure.research.l2_multidate import (
    L2EndpointSpec,
    L2ObservedInterval,
    L2ResearchError,
    apply_l2_regimes,
    build_l2_endpoint_frames,
    dependency_block_expression,
    fit_l2_regime_thresholds,
    l2_model_feature_columns,
    validate_l2_endpoint_frame,
)

MILLISECOND = 1_000_000
SECOND = 1_000_000_000
DATE = "2026-08-08"
DATE_START = int(datetime(2026, 8, 8, tzinfo=UTC).timestamp()) * SECOND


def _endpoints() -> tuple[L2EndpointSpec, ...]:
    return (
        L2EndpointSpec("event_20", "event", 20, "events", 40, None, 20),
        L2EndpointSpec("event_100", "event", 100, "events", 200, None, 100),
        L2EndpointSpec("clock_1000ms", "clock", 1_000, "milliseconds", None, 2_000, 20),
        L2EndpointSpec("clock_5000ms", "clock", 5_000, "milliseconds", None, 10_000, 100),
    )


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame, tuple[L2ObservedInterval, ...]]:
    times = [DATE_START + index * 100 * MILLISECOND for index in range(300)]
    second_start = DATE_START + 60 * SECOND
    times.extend(second_start + index * 100 * MILLISECOND for index in range(300))
    sequences = list(range(1, len(times) + 1))
    midpoint = [100.0 + index * 0.001 + (index // 17) * 0.002 for index in sequences]
    bid = [value - 0.01 for value in midpoint]
    ask = [value + 0.01 for value in midpoint]
    bid_quantity = [4.0 + (index % 11) * 0.1 for index in sequences]
    ask_quantity = [3.0 + (index % 7) * 0.1 for index in sequences]
    books = pl.DataFrame(
        {
            "venue": ["binance_spot"] * len(times),
            "symbol": ["BTCUSDT"] * len(times),
            "event_ts_ns": times,
            "available_ts_ns": times,
            "continuity_id": ["capture-a"] * len(times),
            "sequence_end": sequences,
            "is_valid": [True] * len(times),
            "best_bid": bid,
            "best_ask": ask,
            "bid_quantity": bid_quantity,
            "ask_quantity": ask_quantity,
            "depth_bid_5": [value + 8.0 for value in bid_quantity],
            "depth_ask_5": [value + 7.0 for value in ask_quantity],
            "depth_bid_10": [value + 18.0 for value in bid_quantity],
            "depth_ask_10": [value + 17.0 for value in ask_quantity],
            "tick_size": [0.01] * len(times),
            "lot_size": [0.00001] * len(times),
        }
    )
    deltas = pl.DataFrame(
        {
            "venue": ["binance_spot"] * len(times),
            "symbol": ["BTCUSDT"] * len(times),
            "event_ts_ns": times,
            "available_ts_ns": times,
            "continuity_id": ["capture-a"] * len(times),
            "first_update_id": sequences,
            "last_update_id": sequences,
            "bids": [
                [
                    {
                        "price_ticks": 10_000 + index,
                        "quantity_lots": 0 if index % 13 == 0 else 10,
                    }
                ]
                for index in sequences
            ],
            "asks": [[{"price_ticks": 10_002 + index, "quantity_lots": 10}] for index in sequences],
        }
    )
    intervals = (
        L2ObservedInterval("capture-a", DATE_START, DATE_START + 30 * SECOND),
        L2ObservedInterval("capture-a", second_start, second_start + 30 * SECOND),
    )
    return books, deltas, intervals


def _frames(
    *, books: pl.DataFrame | None = None, deltas: pl.DataFrame | None = None
) -> dict[str, pl.DataFrame]:
    default_books, default_deltas, intervals = _inputs()
    return dict(
        build_l2_endpoint_frames(
            default_books if books is None else books,
            default_deltas if deltas is None else deltas,
            intervals,
            study_date=DATE,
            study_role="train",
            feature_windows=(20, 100),
            volatility_window=100,
            clock_max_state_age_ms=500,
            endpoints=_endpoints(),
        )
    )


def _tied_target_inputs() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    tuple[L2ObservedInterval, ...],
    int,
    int,
    int,
    int,
]:
    books, deltas, intervals = _inputs()
    decision_ts = DATE_START + 20 * SECOND
    target_ts = decision_ts + SECOND
    target_sequences = (
        books.filter(pl.col("available_ts_ns") == target_ts).get_column("sequence_end").to_list()
    )
    assert len(target_sequences) == 1
    lower_sequence = int(target_sequences[0])
    higher_sequence = lower_sequence + 1
    books = books.with_columns(
        pl.when(pl.col("sequence_end") == higher_sequence)
        .then(pl.lit(target_ts))
        .otherwise(pl.col("event_ts_ns"))
        .alias("event_ts_ns"),
        pl.when(pl.col("sequence_end") == higher_sequence)
        .then(pl.lit(target_ts))
        .otherwise(pl.col("available_ts_ns"))
        .alias("available_ts_ns"),
    )
    deltas = deltas.with_columns(
        pl.when(pl.col("last_update_id") == higher_sequence)
        .then(pl.lit(target_ts))
        .otherwise(pl.col("event_ts_ns"))
        .alias("event_ts_ns"),
        pl.when(pl.col("last_update_id") == higher_sequence)
        .then(pl.lit(target_ts))
        .otherwise(pl.col("available_ts_ns"))
        .alias("available_ts_ns"),
    )
    return (
        books,
        deltas,
        intervals,
        decision_ts,
        target_ts,
        lower_sequence,
        higher_sequence,
    )


def _tied_target_frames() -> dict[str, pl.DataFrame]:
    books, deltas, intervals, *_ = _tied_target_inputs()
    endpoints = (
        L2EndpointSpec("event_1", "event", 1, "events", 1, None, 1),
        L2EndpointSpec("clock_1000ms", "clock", 1_000, "milliseconds", None, 2_000, 1),
    )
    return dict(
        build_l2_endpoint_frames(
            books,
            deltas,
            intervals,
            study_date=DATE,
            study_role="train",
            feature_windows=(1,),
            volatility_window=1,
            clock_max_state_age_ms=500,
            endpoints=endpoints,
        )
    )


def test_four_endpoints_are_interval_local_and_exclude_trade_only_features() -> None:
    frames = _frames()
    assert set(frames) == {"event_20", "event_100", "clock_1000ms", "clock_5000ms"}

    event = frames["event_20"]
    assert event.get_column("continuity_id").n_unique() == 2
    assert event.get_column("capture_continuity_id").unique().to_list() == ["capture-a"]
    assert event.group_by("continuity_id").len().get_column("len").to_list() == [300, 300]
    assert event.group_by("continuity_id").agg(
        pl.col("right_censored").sum().alias("censored")
    ).get_column("censored").to_list() == [20, 20]

    regime = fit_l2_regime_thresholds(
        event,
        lower_quantile=1.0 / 3.0,
        upper_quantile=2.0 / 3.0,
        volatility_column="realized_volatility_w100",
    )
    modeled = apply_l2_regimes(event, regime)
    features = l2_model_feature_columns(modeled, windows=(20, 100))
    assert "depth_total_l10" in features
    assert "cancellation_intensity_w100" in features
    assert "volatility_regime_high" in features
    assert not any("trade_" in name for name in features)


def test_exact_clock_target_uses_last_known_state_and_never_looks_forward() -> None:
    books, deltas, _ = _inputs()
    decision_ts = DATE_START + 20 * SECOND
    target_ts = decision_ts + SECOND
    target_sequence = int(
        books.filter(pl.col("available_ts_ns") == target_ts).get_column("sequence_end")[0]
    )
    prior_mid = float(
        books.filter(pl.col("available_ts_ns") == target_ts - 100 * MILLISECOND).get_column(
            "best_bid"
        )[0]
        + 0.01
    )
    books_without_exact_target = books.filter(pl.col("available_ts_ns") != target_ts)
    frame = _frames(books=books_without_exact_target, deltas=deltas)["clock_1000ms"]
    row = frame.filter(pl.col("decision_ts_ns") == decision_ts).row(0, named=True)

    assert row["right_censored"] is False
    assert row["label_information_end_ts_ns"] == target_ts
    assert row["label_information_end_sequence"] == target_sequence - 1
    assert row["clock_target_state_age_ns"] == 100 * MILLISECOND
    current_mid = float(row["mid_price"])
    assert row["future_mid_return"] == pytest.approx(math.log(prior_mid / current_mid))


def test_tied_timestamp_event_label_uses_lexicographic_future_order() -> None:
    *_, target_ts, lower_sequence, higher_sequence = _tied_target_inputs()
    frame = _tied_target_frames()["event_1"]
    row = frame.filter(pl.col("decision_sequence") == lower_sequence).row(0, named=True)

    assert row["decision_ts_ns"] == target_ts
    assert row["right_censored"] is False
    assert row["label_information_end_ts_ns"] == target_ts
    assert row["label_information_end_sequence"] == higher_sequence

    for invalid_sequence in (lower_sequence, lower_sequence - 1):
        corrupted = frame.with_columns(
            pl.when(pl.col("decision_sequence") == lower_sequence)
            .then(pl.lit(invalid_sequence))
            .otherwise(pl.col("label_information_end_sequence"))
            .alias("label_information_end_sequence")
        )
        with pytest.raises(L2ResearchError, match="strictly future"):
            validate_l2_endpoint_frame(corrupted)


def test_exact_clock_target_tie_selects_greatest_observable_sequence() -> None:
    books, _, _, decision_ts, target_ts, _, higher_sequence = _tied_target_inputs()
    frame = _tied_target_frames()["clock_1000ms"]
    row = frame.filter(pl.col("decision_ts_ns") == decision_ts).row(0, named=True)
    expected_mid = float(
        (
            books.filter(pl.col("sequence_end") == higher_sequence).get_column("best_bid")[0]
            + books.filter(pl.col("sequence_end") == higher_sequence).get_column("best_ask")[0]
        )
        / 2.0
    )

    assert row["right_censored"] is False
    assert row["label_information_end_ts_ns"] == target_ts
    assert row["label_information_end_sequence"] == higher_sequence
    assert row["clock_target_state_age_ns"] == 0
    assert row["future_mid_return"] == pytest.approx(math.log(expected_mid / row["mid_price"]))


def test_same_timestamp_higher_sequence_mutation_cannot_change_past_features() -> None:
    books, deltas, intervals, _, target_ts, lower_sequence, higher_sequence = _tied_target_inputs()
    endpoint = L2EndpointSpec("event_1", "event", 1, "events", 1, None, 1)

    def build(candidate: pl.DataFrame) -> pl.DataFrame:
        return build_l2_endpoint_frames(
            candidate,
            deltas,
            intervals,
            study_date=DATE,
            study_role="train",
            feature_windows=(1,),
            volatility_window=1,
            clock_max_state_age_ms=500,
            endpoints=(endpoint,),
        )["event_1"]

    original = build(books)
    mutated = build(
        books.with_columns(
            pl.when(pl.col("sequence_end") == higher_sequence)
            .then(pl.col("best_bid") * 3.0)
            .otherwise(pl.col("best_bid"))
            .alias("best_bid"),
            pl.when(pl.col("sequence_end") == higher_sequence)
            .then(pl.col("best_ask") * 3.0)
            .otherwise(pl.col("best_ask"))
            .alias("best_ask"),
        )
    )
    causal_prefix = (pl.col("decision_ts_ns") < target_ts) | (
        (pl.col("decision_ts_ns") == target_ts) & (pl.col("decision_sequence") <= lower_sequence)
    )
    feature_columns = [
        "sample_id",
        "mid_price",
        "spread_bps",
        "ofi_w1",
        "realized_volatility_w1",
        "max_feature_source_ts_ns",
        "max_feature_source_sequence",
    ]
    assert (
        original.filter(causal_prefix)
        .select(feature_columns)
        .equals(mutated.filter(causal_prefix).select(feature_columns))
    )


def test_clock_target_outside_interval_or_too_stale_is_censored() -> None:
    books, deltas, _ = _inputs()
    near_end = DATE_START + 27 * SECOND
    five_second = _frames(books=books, deltas=deltas)["clock_5000ms"]
    assert five_second.filter(pl.col("decision_ts_ns") == near_end).get_column("right_censored")[0]

    decision_ts = DATE_START + 20 * SECOND
    target_ts = decision_ts + SECOND
    stale_books = books.filter(
        (pl.col("available_ts_ns") <= target_ts - 600 * MILLISECOND)
        | (pl.col("available_ts_ns") > target_ts)
    )
    stale = _frames(books=stale_books, deltas=deltas)["clock_1000ms"]
    row = stale.filter(pl.col("decision_ts_ns") == decision_ts).row(0, named=True)
    assert row["right_censored"] is True
    assert row["future_mid_return"] is None
    assert row["label_information_end_ts_ns"] is None


def test_future_price_mutation_cannot_change_past_l2_features() -> None:
    books, deltas, _ = _inputs()
    cutoff = DATE_START + 20 * SECOND
    original = _frames(books=books, deltas=deltas)["event_20"]
    mutated_books = books.with_columns(
        pl.when(pl.col("available_ts_ns") > cutoff)
        .then(pl.col("best_bid") * 3.0)
        .otherwise(pl.col("best_bid"))
        .alias("best_bid"),
        pl.when(pl.col("available_ts_ns") > cutoff)
        .then(pl.col("best_ask") * 3.0)
        .otherwise(pl.col("best_ask"))
        .alias("best_ask"),
    )
    mutated = _frames(books=mutated_books, deltas=deltas)["event_20"]
    feature_columns = [
        "spread_bps",
        "depth_total_l1",
        "queue_imbalance_l1",
        "microprice_deviation_bps",
        "ofi_w20",
        "cancellation_intensity_w20",
        "realized_volatility_w100",
        "max_feature_source_ts_ns",
    ]
    assert (
        original.filter(pl.col("decision_ts_ns") <= cutoff)
        .select(feature_columns)
        .equals(mutated.filter(pl.col("decision_ts_ns") <= cutoff).select(feature_columns))
    )


def test_train_regimes_reject_validation_fit_and_blocks_are_deterministic() -> None:
    event = _frames()["event_20"]
    with pytest.raises(L2ResearchError, match="train session"):
        fit_l2_regime_thresholds(
            event.with_columns(pl.lit("validation").alias("study_role")),
            lower_quantile=1.0 / 3.0,
            upper_quantile=2.0 / 3.0,
            volatility_column="realized_volatility_w100",
        )

    blocked = event.with_columns(dependency_block_expression(_endpoints()[0]))
    assert blocked.get_column("bootstrap_block").null_count() == 0
    assert blocked.select("sample_id", "bootstrap_block").equals(
        event.with_columns(dependency_block_expression(_endpoints()[0])).select(
            "sample_id", "bootstrap_block"
        )
    )


def test_temporal_validator_rejects_label_crossing_observed_interval() -> None:
    frame = _frames()["event_20"]
    row_id = str(frame.filter(~pl.col("right_censored")).get_column("sample_id")[0])
    corrupted = frame.with_columns(
        pl.when(pl.col("sample_id") == row_id)
        .then(pl.col("observed_interval_end_ns_exclusive"))
        .otherwise(pl.col("label_information_end_ts_ns"))
        .alias("label_information_end_ts_ns")
    )
    with pytest.raises(L2ResearchError, match="interval-local"):
        validate_l2_endpoint_frame(corrupted)


def test_temporal_validator_rejects_same_timestamp_future_feature_sequence() -> None:
    frame = _tied_target_frames()["event_1"]
    *_, lower_sequence, _ = _tied_target_inputs()
    corrupted = frame.with_columns(
        pl.when(pl.col("decision_sequence") == lower_sequence)
        .then(pl.col("decision_sequence") + 1)
        .otherwise(pl.col("max_feature_source_sequence"))
        .alias("max_feature_source_sequence")
    )
    with pytest.raises(L2ResearchError, match="base timing"):
        validate_l2_endpoint_frame(corrupted)
