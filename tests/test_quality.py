from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from microstructure.data.book import DepthDelta, deltas_table
from microstructure.data.quality import (
    IncrementalQualityValidator,
    validate_batches,
    validate_table,
)
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


def test_incremental_trade_state_crosses_batch_boundaries_with_global_indexes() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=3,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=41,
    )
    records = data.trades.to_pylist()
    first_event = int(records[0]["event_ts_ns"])
    first_received = int(records[0]["received_ts_ns"])
    records[1]["trade_id"] = records[0]["trade_id"]
    records[1]["event_ts_ns"] = first_event - 1
    records[1]["received_ts_ns"] = first_received - 1
    records[1]["available_ts_ns"] = first_received
    records[2]["event_ts_ns"] = first_event + 1_000
    records[2]["received_ts_ns"] = first_received + 1_000
    records[2]["available_ts_ns"] = first_received + 1_000
    batches = [table_from_records("trades", [record]) for record in records]

    report = validate_batches(batches, "trades", max_silence_ns=100)
    by_rule = {finding.rule_id: finding for finding in report.findings}

    assert report.rows_checked == 3
    assert by_rule["trade.duplicate"].row_index == 1
    assert by_rule["trade.duplicate"].details["first_row"] == 0
    assert by_rule["temporal.out_of_order_event_time"].row_index == 1
    assert by_rule["temporal.out_of_order_event_time"].details["previous_row"] == 0
    assert by_rule["temporal.receive_clock_reversal"].row_index == 1
    assert by_rule["temporal.long_silence"].row_index == 2


def test_incremental_book_sequence_gap_is_detected_across_batches() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=2,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=42,
    )
    records = data.book_observations.to_pylist()
    records[1]["sequence_start"] = int(records[0]["sequence_end"]) + 2
    records[1]["sequence_end"] = records[1]["sequence_start"]

    report = validate_batches(
        (
            table_from_records("book_observations", [records[0]]),
            table_from_records("book_observations", [records[1]]),
        ),
        "book_observations",
    )
    gaps = [finding for finding in report.findings if finding.rule_id == "sequence.missing_range"]

    assert len(gaps) == 1
    assert gaps[0].row_index == 1
    assert gaps[0].details == {
        "expected_sequence": int(records[0]["sequence_end"]) + 1,
        "observed_start": records[1]["sequence_start"],
        "missing_start": int(records[0]["sequence_end"]) + 1,
        "missing_end": int(records[0]["sequence_end"]) + 1,
    }


def test_incremental_depth_sequence_and_previous_hint_cross_batches() -> None:
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
    first = DepthDelta(
        **common,
        event_ts_ns=1_000,
        received_ts_ns=1_100,
        available_ts_ns=1_100,
        capture_seq=1,
        first_update_id=101,
        last_update_id=102,
        previous_update_id=None,
    )
    second = DepthDelta(
        **common,
        event_ts_ns=2_000,
        received_ts_ns=2_100,
        available_ts_ns=2_100,
        capture_seq=2,
        first_update_id=105,
        last_update_id=106,
        previous_update_id=99,
    )

    report = validate_batches(
        (deltas_table([first]), deltas_table([second])),
        "depth_deltas",
    )
    rules = {finding.rule_id: finding for finding in report.findings}

    assert rules["sequence.missing_range"].row_index == 1
    assert rules["sequence.previous_id_mismatch"].row_index == 1


def test_incremental_state_is_isolated_by_venue_across_batches() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=2,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=43,
    )
    records = data.trades.to_pylist()
    records[1]["venue"] = "second_venue"
    records[1]["trade_id"] = records[0]["trade_id"]
    records[1]["event_ts_ns"] = int(records[0]["event_ts_ns"]) - 1
    records[1]["received_ts_ns"] = int(records[0]["received_ts_ns"]) - 1
    records[1]["available_ts_ns"] = records[0]["available_ts_ns"]

    report = validate_batches(
        (table_from_records("trades", [record]) for record in records),
        "trades",
    )
    rule_ids = {finding.rule_id for finding in report.findings}

    assert "trade.duplicate" not in rule_ids
    assert "temporal.out_of_order_event_time" not in rule_ids
    assert "temporal.receive_clock_reversal" not in rule_ids


class _OneShotBatches:
    def __init__(self, batches: list[pa.Table | pa.RecordBatch]) -> None:
        self._batches = batches
        self.iterations = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("batch iterable was consumed more than once")
        yield from self._batches


def test_validate_batches_consumes_one_shot_iterable_once_and_accepts_record_batches() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=4,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=44,
    )
    batches = _OneShotBatches(list(data.trades.to_batches(max_chunksize=1)))

    report = validate_batches(batches, "trades", row_chunk_size=1)

    assert batches.iterations == 1
    assert report.rows_checked == data.trades.num_rows
    assert not report.has_errors


def test_incremental_bounded_preview_keeps_exact_totals_and_streams_all_jsonl(
    tmp_path: Path,
) -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=12,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=45,
    )
    records = data.trades.to_pylist()
    for record in records:
        record["price_ticks"] = -1
        record["price"] = -float(record["tick_size"])
    invalid = table_from_records("trades", records)
    before = invalid.to_pylist()
    findings_path = tmp_path / "quality" / "findings.jsonl"

    report = validate_batches(
        invalid.to_batches(max_chunksize=2),
        "trades",
        max_findings=3,
        findings_jsonl_path=findings_path,
        row_chunk_size=1,
    )
    summary_path = tmp_path / "quality" / "summary.json"
    report.write_json(summary_path)
    streamed = [json.loads(line) for line in findings_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert invalid.to_pylist() == before
    assert report.error_count == len(records)
    assert report.warning_count == 0
    assert report.has_errors
    assert report.findings_truncated
    assert len(report.findings) == 3
    assert len(streamed) == len(records)
    assert [item["row_index"] for item in streamed] == list(range(len(records)))
    assert summary["summary"] == {"errors": len(records), "warnings": 0}
    assert summary["findings_preview"] == {
        "retained": 3,
        "total": len(records),
        "truncated": True,
    }
    assert summary["findings_jsonl_path"] == str(findings_path.resolve())
    assert len(summary["findings"]) == 3


def test_incremental_findings_publish_atomically_and_preserve_prior_on_abort(
    tmp_path: Path,
) -> None:
    findings_path = tmp_path / "quality" / "findings.jsonl"
    findings_path.parent.mkdir(parents=True)
    findings_path.write_text("prior-complete-evidence\n", encoding="utf-8")
    validator = IncrementalQualityValidator(
        "trades",
        findings_jsonl_path=findings_path,
    )

    validator.close()

    assert findings_path.read_text(encoding="utf-8") == "prior-complete-evidence\n"
    assert not list(findings_path.parent.glob(f".{findings_path.name}.*.tmp"))


def test_incremental_finish_is_idempotent_and_update_after_finish_fails() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=2,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=46,
    )
    validator = IncrementalQualityValidator("trades", max_findings=0)
    validator.update(data.trades)

    first = validator.finish()

    assert validator.finish() is first
    with pytest.raises(RuntimeError, match="already closed"):
        validator.update(data.trades)


def test_validate_batches_matches_unbounded_table_rule_semantics() -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=3,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=47,
    )
    records = data.trades.to_pylist()
    records[1]["trade_id"] = records[0]["trade_id"]
    records[2]["quantity_lots"] = 0
    records[2]["quantity"] = 0.0
    invalid = table_from_records("trades", records)

    legacy = validate_table(invalid, "trades")
    incremental = validate_batches([invalid], "trades", max_findings=None)

    assert incremental.rows_checked == legacy.rows_checked
    assert incremental.error_count == legacy.error_count
    assert incremental.warning_count == legacy.warning_count
    assert incremental.findings == legacy.findings
