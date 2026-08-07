from __future__ import annotations

import pyarrow as pa
import pytest

from microstructure.data.schemas import SCHEMA_VERSION, SchemaError, ensure_schema, get_schema
from microstructure.data.synthetic import generate_synthetic_market


def test_normalized_schemas_publish_temporal_and_numeric_contracts() -> None:
    trades = get_schema("trades")
    books = get_schema("book_observations")

    assert trades.metadata is not None
    assert trades.metadata[b"schema_version"] == SCHEMA_VERSION.encode()
    assert trades.field("event_ts_ns").type == pa.int64()
    assert trades.field("available_ts_ns").type == pa.int64()
    assert trades.field("price_ticks").type == pa.int64()
    assert trades.field("quantity_lots").type == pa.int64()
    assert books.field("best_bid_ticks").type == pa.int64()
    assert books.field("continuity_id").type == pa.string()


def test_schema_registry_fails_closed_on_unknown_name_or_version() -> None:
    with pytest.raises(SchemaError, match="unknown normalized schema"):
        get_schema("mystery")
    with pytest.raises(SchemaError, match="unsupported schema version"):
        get_schema("trades", "2.0.0")


def test_synthetic_generator_is_deterministic_and_explicitly_labelled() -> None:
    kwargs = {
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "events_per_symbol": 20,
        "start_ts_ns": 1_704_153_600_000_000_000,
        "seed": 123,
    }
    first = generate_synthetic_market(**kwargs)
    second = generate_synthetic_market(**kwargs)
    changed = generate_synthetic_market(**{**kwargs, "seed": 124})

    assert first.evidence_tier == "SYNTHETIC_SMOKE"
    assert first.trades.equals(second.trades)
    assert first.book_observations.equals(second.book_observations)
    assert not first.trades.equals(changed.trades)
    assert first.trades.schema.equals(get_schema("trades"), check_metadata=True)
    assert first.book_observations.schema.equals(
        get_schema("book_observations"), check_metadata=True
    )
    assert set(first.trades.column("venue").to_pylist()) == {"synthetic"}
    assert first.trades.num_rows == 40


def test_synthetic_information_clock_never_precedes_event_or_receipt() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=10,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=7,
    )
    for table in (data.trades, data.book_observations):
        for row in table.to_pylist():
            assert row["available_ts_ns"] >= row["event_ts_ns"]
            assert row["available_ts_ns"] >= row["received_ts_ns"]
            assert row["availability_basis"] == "synthetic_receipt"


def test_schema_validation_rejects_mislabeled_row_or_metadata_version() -> None:
    table = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=2,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=8,
    ).trades
    records = table.to_pylist()
    records[0]["schema_version"] = "2.0.0"
    mislabeled = pa.Table.from_pylist(records, schema=table.schema)

    with pytest.raises(SchemaError, match="row schema_version mismatch"):
        ensure_schema(mislabeled, "trades")

    bad_metadata = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), b"schema_version": b"2.0.0"}
    )
    with pytest.raises(SchemaError, match="metadata version mismatch"):
        ensure_schema(bad_metadata, "trades")
