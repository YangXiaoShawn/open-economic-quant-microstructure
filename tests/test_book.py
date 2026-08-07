from __future__ import annotations

import pytest

from microstructure.data.book import (
    BookInvariantError,
    BookSnapshot,
    DepthDelta,
    IncrementalBookReconstructor,
    reconstruct_snapshot_and_deltas,
)


def _snapshot() -> BookSnapshot:
    return BookSnapshot(
        venue="binance_spot",
        symbol="BTCUSDT",
        snapshot_id="snapshot-fixture",
        request_ts_ns=1_000,
        received_ts_ns=1_100,
        available_ts_ns=1_100,
        continuity_id="session-1",
        last_update_id=100,
        depth_limit=5000,
        bids=((10_000, 10), (9_999, 20), (9_998, 30)),
        asks=((10_001, 15), (10_002, 25), (10_003, 35)),
        tick_size=0.01,
        lot_size=0.001,
        source_artifact_id="snapshot-fixture",
    )


def _delta(
    start: int,
    end: int,
    *,
    bids: tuple[tuple[int, int], ...] = (),
    asks: tuple[tuple[int, int], ...] = (),
    event_ts_ns: int = 2_000,
    previous: int | None = None,
    continuity_id: str = "session-1",
    tick_size: float = 0.01,
    lot_size: float = 0.001,
) -> DepthDelta:
    return DepthDelta(
        venue="binance_spot",
        symbol="BTCUSDT",
        event_ts_ns=event_ts_ns,
        received_ts_ns=event_ts_ns + 100,
        available_ts_ns=event_ts_ns + 100,
        availability_basis="local_receive_time",
        capture_seq=end,
        continuity_id=continuity_id,
        first_update_id=start,
        last_update_id=end,
        previous_update_id=previous,
        bids=bids,
        asks=asks,
        tick_size=tick_size,
        lot_size=lot_size,
        source_artifact_id=f"delta-{start}-{end}",
    )


def test_reconstruction_discards_stale_bridges_snapshot_and_accepts_overlap() -> None:
    result = reconstruct_snapshot_and_deltas(
        _snapshot(),
        [
            _delta(98, 100, bids=((10_000, 999),)),
            _delta(100, 102, bids=((10_000, 0),)),
            _delta(102, 104, bids=((10_000, 12),), asks=((10_001, 18),)),
        ],
    )

    assert result.status == "LIVE"
    assert result.stale_events == 1
    assert result.final_update_id == 104
    assert result.observations.num_rows == 2
    first, second = result.observations.to_pylist()
    assert first["best_bid_ticks"] == 9_999
    assert second["best_bid_ticks"] == 10_000
    assert second["sequence_start"] == 102
    assert result.gaps.num_rows == 0


def test_forward_gap_is_recorded_and_later_events_are_not_applied() -> None:
    result = reconstruct_snapshot_and_deltas(
        _snapshot(),
        [
            _delta(100, 102),
            _delta(105, 106, bids=((10_000, 20),)),
            _delta(107, 108, bids=((10_000, 30),)),
        ],
    )

    assert result.status == "GAPPED"
    assert result.final_update_id == 102
    assert result.observations.num_rows == 1
    [gap] = result.gaps.to_pylist()
    assert gap["expected_sequence"] == 103
    assert gap["missing_start"] == 103
    assert gap["missing_end"] == 104
    assert gap["reason"] == "forward_sequence_gap"


def test_crossed_book_is_emitted_as_invalid_then_epoch_stops() -> None:
    result = reconstruct_snapshot_and_deltas(
        _snapshot(),
        [_delta(100, 101, bids=((10_002, 5),)), _delta(102, 102)],
    )

    assert result.status == "INVALID"
    assert result.observations.num_rows == 1
    assert result.observations.column("is_valid").to_pylist() == [False]
    assert result.gaps.column("reason").to_pylist() == ["crossed_or_locked_book"]
    assert result.final_update_id == 101


def test_sequence_order_not_timestamp_order_and_availability_waits_for_snapshot() -> None:
    result = reconstruct_snapshot_and_deltas(
        _snapshot(),
        [
            _delta(100, 101, event_ts_ns=5_000),
            _delta(102, 102, event_ts_ns=4_000),
        ],
    )

    assert result.status == "LIVE"
    assert result.final_update_id == 102
    assert result.observations.column("event_ts_ns").to_pylist() == [5_000, 4_000]
    assert all(
        value >= _snapshot().available_ts_ns
        for value in result.observations.column("available_ts_ns").to_pylist()
    )


def test_previous_update_id_mismatch_requires_resynchronization() -> None:
    result = reconstruct_snapshot_and_deltas(_snapshot(), [_delta(100, 102, previous=99)])

    assert result.status == "GAPPED"
    assert result.observations.num_rows == 0
    assert result.gaps.column("reason").to_pylist() == ["previous_update_id_mismatch"]


def test_delta_scale_mismatch_fails_closed_before_reinterpretation() -> None:
    with pytest.raises(BookInvariantError, match="scales do not match"):
        reconstruct_snapshot_and_deltas(
            _snapshot(),
            [_delta(100, 101, tick_size=0.1)],
        )


def test_malformed_stale_looking_range_is_invalid_and_later_delta_is_audited() -> None:
    reconstructor = IncrementalBookReconstructor(_snapshot())

    malformed = reconstructor.update(_delta(101, 99))
    excluded = reconstructor.update(_delta(101, 101))

    assert malformed.outcome == "INVALID"
    assert malformed.gap is not None
    assert malformed.gap.reason == "invalid_sequence_range"
    assert reconstructor.stale_events == 0
    assert excluded.outcome == "EXCLUDED_AFTER_TERMINAL"
    assert excluded.gap is not None
    assert excluded.gap.reason == "epoch_already_invalid"
