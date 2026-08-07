from __future__ import annotations

from typing import Any, cast

import polars as pl
import pytest

from microstructure.data.schemas import SCHEMA_VERSION, table_from_records
from microstructure.research.features import (
    ResearchDataError,
    build_cancellation_intensity_features,
    model_feature_columns,
)


def _delta(
    update_id: int,
    *,
    bids: list[tuple[int, int]],
    asks: list[tuple[int, int]],
    continuity_id: str = "epoch-a",
) -> dict[str, Any]:
    event_ts_ns = 1_700_000_000_000_000_000 + update_id * 1_000
    return {
        "schema_version": SCHEMA_VERSION,
        "venue": "binance_spot",
        "symbol": "BTCUSDT",
        "event_ts_ns": event_ts_ns,
        "received_ts_ns": event_ts_ns + 100,
        "available_ts_ns": event_ts_ns + 100,
        "availability_basis": "local_receive_time",
        "capture_seq": update_id,
        "continuity_id": continuity_id,
        "first_update_id": update_id,
        "last_update_id": update_id,
        "previous_update_id": update_id - 1 if update_id > 1 else None,
        "bids": [
            {"price_ticks": price_ticks, "quantity_lots": quantity_lots}
            for price_ticks, quantity_lots in bids
        ],
        "asks": [
            {"price_ticks": price_ticks, "quantity_lots": quantity_lots}
            for price_ticks, quantity_lots in asks
        ],
        "tick_size": 0.01,
        "lot_size": 0.001,
        "source_artifact_id": f"{update_id:064x}",
    }


def _frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    table = table_from_records("depth_deltas", records)
    return cast(pl.DataFrame, pl.from_arrow(table))


def test_cancellation_intensity_counts_only_observable_zero_quantity_deletes() -> None:
    frame = _frame(
        [
            _delta(1, bids=[(10_000, 0)], asks=[(10_002, 5)]),
            _delta(2, bids=[(9_999, 0)], asks=[(10_003, 0)]),
            _delta(3, bids=[(10_000, 4)], asks=[]),
            _delta(
                10,
                bids=[(10_000, 7)],
                asks=[],
                continuity_id="epoch-b",
            ),
        ]
    )

    result = build_cancellation_intensity_features(frame, windows=(2,))
    rows = result.sort(["continuity_id", "decision_sequence"]).to_dicts()

    assert [row["cancellation_deletes_current"] for row in rows] == [1, 2, 0, 0]
    assert [row["depth_updates_current"] for row in rows] == [2, 2, 1, 1]
    assert [row["cancellation_deletes_w2"] for row in rows] == [1, 3, 2, 0]
    assert [row["depth_updates_w2"] for row in rows] == [2, 4, 3, 1]
    assert [row["cancellation_intensity_w2"] for row in rows] == pytest.approx(
        [0.5, 0.75, 2.0 / 3.0, 0.0]
    )
    assert set(result.get_column("cancellation_observation_policy")) == {
        "zero_quantity_level_deletes_only"
    }
    assert not result.get_column("nonzero_reduction_classified_as_cancellation").any()
    assert result.get_column("max_feature_source_ts_ns").equals(
        result.get_column("feature_cutoff_ts_ns")
    )
    assert result.get_column("max_feature_source_sequence").equals(
        result.get_column("decision_sequence")
    )
    assert model_feature_columns(result) == ("cancellation_intensity_w2",)


def test_future_depth_mutation_cannot_change_past_cancellation_features() -> None:
    records = [
        _delta(1, bids=[(10_000, 0)], asks=[]),
        _delta(2, bids=[(10_000, 5)], asks=[]),
        _delta(3, bids=[], asks=[(10_002, 0)]),
        _delta(4, bids=[(9_999, 8)], asks=[]),
    ]
    before = build_cancellation_intensity_features(_frame(records), windows=(3,))
    mutated = [dict(record) for record in records]
    mutated[3] = _delta(4, bids=[(9_999, 0)], asks=[(10_004, 0)])
    after = build_cancellation_intensity_features(_frame(mutated), windows=(3,))
    columns = [
        "decision_sequence",
        "cancellation_deletes_w3",
        "depth_updates_w3",
        "cancellation_intensity_w3",
        "max_feature_source_ts_ns",
        "max_feature_source_sequence",
    ]

    assert (
        before.filter(pl.col("decision_sequence") < 4)
        .select(columns)
        .equals(after.filter(pl.col("decision_sequence") < 4).select(columns))
    )


def test_cancellation_features_fail_closed_on_unsegmented_sequence_gap() -> None:
    frame = _frame(
        [
            _delta(1, bids=[(10_000, 0)], asks=[]),
            _delta(3, bids=[(10_000, 0)], asks=[]),
        ]
    )

    with pytest.raises(ResearchDataError, match="stale/gapped sequence"):
        build_cancellation_intensity_features(frame, windows=(2,))
