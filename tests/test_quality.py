from __future__ import annotations

from microstructure.data.book import DepthDelta, deltas_table
from microstructure.data.quality import validate_table
from microstructure.data.schemas import table_from_records
from microstructure.data.synthetic import generate_synthetic_market


def test_clean_synthetic_tables_pass_and_validation_does_not_mutate() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=25,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=14,
    )
    trade_before = data.trades.to_pylist()
    book_before = data.book_observations.to_pylist()

    trades = validate_table(data.trades, "trades")
    books = validate_table(data.book_observations, "book_observations")

    assert not trades.has_errors
    assert not books.has_errors
    assert data.trades.to_pylist() == trade_before
    assert data.book_observations.to_pylist() == book_before


def test_trade_quality_reports_duplicate_and_nonpositive_values_without_repair() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=3,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=15,
    )
    records = data.trades.to_pylist()
    records[1]["trade_id"] = records[0]["trade_id"]
    records[1]["price_ticks"] = -1
    records[1]["price"] = -0.01
    records[2]["quantity_lots"] = 0
    records[2]["quantity"] = 0.0
    invalid = table_from_records("trades", records)
    before = invalid.to_pylist()

    report = validate_table(invalid, "trades")
    rule_ids = {finding.rule_id for finding in report.findings}

    assert report.has_errors
    assert "trade.duplicate" in rule_ids
    assert "trade.nonpositive_price" in rule_ids
    assert "trade.nonpositive_quantity" in rule_ids
    assert invalid.to_pylist() == before


def test_trade_identity_and_clock_state_are_scoped_by_venue() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=2,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=18,
    )
    records = data.trades.to_pylist()
    records[1]["venue"] = "second_venue"
    records[1]["trade_id"] = records[0]["trade_id"]
    cross_venue = table_from_records("trades", records)

    report = validate_table(cross_venue, "trades")

    assert "trade.duplicate" not in {finding.rule_id for finding in report.findings}


def test_book_quality_reports_sequence_gap_crossed_book_and_clock_discontinuity() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=3,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=16,
    )
    records = data.book_observations.to_pylist()
    records[1]["sequence_start"] = 4
    records[1]["sequence_end"] = 4
    records[1]["best_bid_ticks"] = records[1]["best_ask_ticks"] + 1
    records[1]["best_bid"] = records[1]["best_ask"] + records[1]["tick_size"]
    records[1]["spread"] = records[1]["best_ask"] - records[1]["best_bid"]
    records[1]["mid_price"] = (records[1]["best_bid"] + records[1]["best_ask"]) / 2
    records[1]["microprice"] = records[1]["mid_price"]
    records[2]["received_ts_ns"] = records[1]["received_ts_ns"] - 1
    records[2]["available_ts_ns"] = max(records[2]["event_ts_ns"], records[2]["received_ts_ns"])
    invalid = table_from_records("book_observations", records)

    report = validate_table(invalid, "book_observations")
    rule_ids = {finding.rule_id for finding in report.findings}

    assert "sequence.missing_range" in rule_ids
    assert "book.crossed_or_locked" in rule_ids
    assert "temporal.receive_clock_reversal" in rule_ids


def test_book_quality_rejects_float_values_inconsistent_with_exact_scales() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=2,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=17,
    )
    records = data.book_observations.to_pylist()
    records[0]["best_bid"] += records[0]["tick_size"] / 2.0
    inconsistent = table_from_records("book_observations", records)

    report = validate_table(inconsistent, "book_observations")

    assert "book.price_scale_mismatch" in {finding.rule_id for finding in report.findings}


def test_zero_depth_quantity_is_valid_delete_but_negative_quantity_is_error() -> None:
    common = {
        "venue": "binance_spot",
        "symbol": "BTCUSDT",
        "event_ts_ns": 1_000,
        "received_ts_ns": 1_100,
        "available_ts_ns": 1_100,
        "availability_basis": "local_receive_time",
        "capture_seq": 1,
        "continuity_id": "session-1",
        "first_update_id": 101,
        "last_update_id": 101,
        "previous_update_id": None,
        "asks": (),
        "tick_size": 0.01,
        "lot_size": 0.001,
        "source_artifact_id": "fixture",
    }
    delete = DepthDelta(bids=((100, 0),), **common)
    negative = DepthDelta(
        bids=((100, -1),),
        **{**common, "event_ts_ns": 2_000, "received_ts_ns": 2_100, "available_ts_ns": 2_100},
    )
    table = deltas_table([delete, negative])

    report = validate_table(table, "depth_deltas")
    quantity_findings = [
        finding for finding in report.findings if finding.rule_id == "depth.negative_quantity"
    ]

    assert len(quantity_findings) == 1
    assert quantity_findings[0].row_index == 1


def test_depth_delta_quality_reports_gap_stale_and_previous_id_mismatch() -> None:
    common = {
        "venue": "binance_spot",
        "symbol": "BTCUSDT",
        "availability_basis": "local_receive_time",
        "continuity_id": "session-1",
        "bids": ((10_000, 1),),
        "asks": (),
        "tick_size": 0.01,
        "lot_size": 0.001,
        "source_artifact_id": "fixture",
    }
    deltas = [
        DepthDelta(
            **common,
            event_ts_ns=1_000,
            received_ts_ns=1_100,
            available_ts_ns=1_100,
            capture_seq=1,
            first_update_id=101,
            last_update_id=102,
            previous_update_id=None,
        ),
        DepthDelta(
            **common,
            event_ts_ns=2_000,
            received_ts_ns=2_100,
            available_ts_ns=2_100,
            capture_seq=2,
            first_update_id=105,
            last_update_id=106,
            previous_update_id=99,
        ),
        DepthDelta(
            **common,
            event_ts_ns=3_000,
            received_ts_ns=3_100,
            available_ts_ns=3_100,
            capture_seq=3,
            first_update_id=104,
            last_update_id=105,
            previous_update_id=None,
        ),
    ]

    report = validate_table(deltas_table(deltas), "depth_deltas")
    rule_ids = {finding.rule_id for finding in report.findings}

    assert "sequence.missing_range" in rule_ids
    assert "sequence.previous_id_mismatch" in rule_ids
    assert "sequence.stale_or_duplicate" in rule_ids
