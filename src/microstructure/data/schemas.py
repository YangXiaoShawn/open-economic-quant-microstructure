"""Versioned normalized Arrow schemas and their temporal contract.

All timestamps are signed UTC epoch nanoseconds.  ``available_ts_ns`` is the
earliest time at which a row may enter a research information set.  Archive
rows explicitly identify exchange event time as a proxy; live rows use local
receipt time.  Prices and quantities retain exact integer tick/lot columns and
also expose documented floating convenience columns for research consumers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

SCHEMA_VERSION = "1.0.0"


class SchemaError(ValueError):
    """Raised when a normalized table violates its declared schema."""


def _metadata(name: str) -> dict[bytes, bytes]:
    return {
        b"schema_name": name.encode(),
        b"schema_version": SCHEMA_VERSION.encode(),
        b"timestamp_unit": b"UTC epoch nanoseconds",
        b"temporal_contract": (
            b"available_ts_ns is the information-set clock; event_ts_ns alone is not receipt proof"
        ),
        b"numeric_contract": (
            b"price_ticks and quantity_lots are exact; float columns are convenience units"
        ),
    }


_COMMON_EVENT_FIELDS = [
    pa.field("schema_version", pa.string(), nullable=False),
    pa.field("venue", pa.string(), nullable=False),
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("event_ts_ns", pa.int64(), nullable=False),
    pa.field("received_ts_ns", pa.int64()),
    pa.field("available_ts_ns", pa.int64(), nullable=False),
    pa.field("availability_basis", pa.string(), nullable=False),
    pa.field("capture_seq", pa.int64()),
    pa.field("continuity_id", pa.string()),
]


TRADE_SCHEMA = pa.schema(
    [
        *_COMMON_EVENT_FIELDS,
        pa.field("trade_id", pa.int64(), nullable=False),
        pa.field("first_trade_id", pa.int64()),
        pa.field("last_trade_id", pa.int64()),
        pa.field("price_ticks", pa.int64(), nullable=False),
        pa.field("quantity_lots", pa.int64(), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("quantity", pa.float64(), nullable=False),
        pa.field("quote_quantity", pa.float64(), nullable=False),
        pa.field("aggressor_side", pa.string(), nullable=False),
        pa.field("buyer_is_maker", pa.bool_(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
    ],
    metadata=_metadata("trades"),
)


BOOK_OBSERVATION_SCHEMA = pa.schema(
    [
        *_COMMON_EVENT_FIELDS,
        pa.field("sequence_start", pa.int64(), nullable=False),
        pa.field("sequence_end", pa.int64(), nullable=False),
        pa.field("is_valid", pa.bool_(), nullable=False),
        pa.field("best_bid_ticks", pa.int64(), nullable=False),
        pa.field("best_ask_ticks", pa.int64(), nullable=False),
        pa.field("bid_quantity_lots", pa.int64(), nullable=False),
        pa.field("ask_quantity_lots", pa.int64(), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("best_bid", pa.float64(), nullable=False),
        pa.field("best_ask", pa.float64(), nullable=False),
        pa.field("bid_quantity", pa.float64(), nullable=False),
        pa.field("ask_quantity", pa.float64(), nullable=False),
        pa.field("spread", pa.float64(), nullable=False),
        pa.field("mid_price", pa.float64(), nullable=False),
        pa.field("microprice", pa.float64(), nullable=False),
        pa.field("depth_bid_1", pa.float64(), nullable=False),
        pa.field("depth_ask_1", pa.float64(), nullable=False),
        pa.field("depth_bid_5", pa.float64(), nullable=False),
        pa.field("depth_ask_5", pa.float64(), nullable=False),
        pa.field("depth_bid_10", pa.float64(), nullable=False),
        pa.field("depth_ask_10", pa.float64(), nullable=False),
        pa.field("queue_imbalance_1", pa.float64(), nullable=False),
        pa.field("queue_imbalance_5", pa.float64(), nullable=False),
        pa.field("queue_imbalance_10", pa.float64(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
    ],
    metadata=_metadata("book_observations"),
)


LEVEL_TYPE = pa.struct(
    [
        pa.field("price_ticks", pa.int64(), nullable=False),
        pa.field("quantity_lots", pa.int64(), nullable=False),
    ]
)


DEPTH_DELTA_SCHEMA = pa.schema(
    [
        *_COMMON_EVENT_FIELDS,
        pa.field("first_update_id", pa.int64(), nullable=False),
        pa.field("last_update_id", pa.int64(), nullable=False),
        pa.field("previous_update_id", pa.int64()),
        pa.field("bids", pa.list_(LEVEL_TYPE), nullable=False),
        pa.field("asks", pa.list_(LEVEL_TYPE), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
    ],
    metadata=_metadata("depth_deltas"),
)


BOOK_SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("request_ts_ns", pa.int64(), nullable=False),
        pa.field("received_ts_ns", pa.int64(), nullable=False),
        pa.field("available_ts_ns", pa.int64(), nullable=False),
        pa.field("continuity_id", pa.string(), nullable=False),
        pa.field("last_update_id", pa.int64(), nullable=False),
        pa.field("depth_limit", pa.int32(), nullable=False),
        pa.field("bids", pa.list_(LEVEL_TYPE), nullable=False),
        pa.field("asks", pa.list_(LEVEL_TYPE), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
    ],
    metadata=_metadata("book_snapshots"),
)


SEQUENCE_GAP_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("continuity_id", pa.string(), nullable=False),
        pa.field("expected_sequence", pa.int64(), nullable=False),
        pa.field("observed_sequence_start", pa.int64(), nullable=False),
        pa.field("observed_sequence_end", pa.int64(), nullable=False),
        pa.field("missing_start", pa.int64(), nullable=False),
        pa.field("missing_end", pa.int64(), nullable=False),
        pa.field("detected_ts_ns", pa.int64(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("source_artifact_id", pa.string(), nullable=False),
    ],
    metadata=_metadata("sequence_gaps"),
)


SCHEMAS: Mapping[str, pa.Schema] = {
    "trades": TRADE_SCHEMA,
    "book_observations": BOOK_OBSERVATION_SCHEMA,
    "depth_deltas": DEPTH_DELTA_SCHEMA,
    "book_snapshots": BOOK_SNAPSHOT_SCHEMA,
    "sequence_gaps": SEQUENCE_GAP_SCHEMA,
}


def get_schema(name: str, version: str = SCHEMA_VERSION) -> pa.Schema:
    """Return a schema by stable name and fail closed on unknown versions."""
    if version != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema version {version!r}; expected {SCHEMA_VERSION!r}")
    try:
        return SCHEMAS[name]
    except KeyError as exc:
        raise SchemaError(f"unknown normalized schema: {name!r}") from exc


def table_from_records(name: str, records: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Construct a table using the registry rather than inferred Arrow types."""
    schema = get_schema(name)
    try:
        return pa.Table.from_pylist(list(records), schema=schema)
    except (pa.ArrowException, TypeError, ValueError) as exc:
        raise SchemaError(f"records do not conform to {name} {SCHEMA_VERSION}: {exc}") from exc


def ensure_schema(table: pa.Table | pa.RecordBatch, name: str) -> None:
    """Require exact field order/types/nullability; metadata may be absent on batches."""
    expected = get_schema(name)
    actual = table.schema
    if not actual.equals(expected, check_metadata=False):
        raise SchemaError(f"schema mismatch for {name}: expected {expected}, got {actual}")
    metadata = actual.metadata or {}
    declared_name = metadata.get(b"schema_name")
    declared_version = metadata.get(b"schema_version")
    if declared_name is not None and declared_name != name.encode():
        raise SchemaError(
            f"schema metadata name mismatch: expected {name!r}, got {declared_name.decode()}"
        )
    if declared_version is not None and declared_version != SCHEMA_VERSION.encode():
        raise SchemaError(
            "schema metadata version mismatch: "
            f"expected {SCHEMA_VERSION!r}, got {declared_version.decode()}"
        )
    version_column = table.column(actual.get_field_index("schema_version"))
    observed_versions = set(version_column.to_pylist())
    if observed_versions.difference({SCHEMA_VERSION}):
        raise SchemaError(
            f"row schema_version mismatch: expected only {SCHEMA_VERSION!r}, "
            f"got {sorted(observed_versions)!r}"
        )
