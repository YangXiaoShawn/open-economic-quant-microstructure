"""Non-mutating data-quality rules with explicit, machine-readable findings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure.data.schemas import ensure_schema
from microstructure.provenance import utc_now_iso, write_json

Severity = Literal["ERROR", "WARNING"]


@dataclass(frozen=True, slots=True)
class QualityFinding:
    rule_id: str
    severity: Severity
    dataset: str
    row_index: int | None
    symbol: str | None
    event_ts_ns: int | None
    message: str
    details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    dataset: str
    rows_checked: int
    findings: tuple[QualityFinding, ...]

    @property
    def has_errors(self) -> bool:
        return any(item.severity == "ERROR" for item in self.findings)

    @property
    def error_count(self) -> int:
        return sum(item.severity == "ERROR" for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "WARNING" for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": utc_now_iso(),
            "dataset": self.dataset,
            "rows_checked": self.rows_checked,
            "summary": {"errors": self.error_count, "warnings": self.warning_count},
            "findings": [asdict(item) for item in self.findings],
            "mutation_policy": "observations were not changed or repaired",
        }

    def write_json(self, path: str | Path) -> None:
        write_json(path, self.to_dict())


def _finding(
    findings: list[QualityFinding],
    *,
    rule_id: str,
    severity: Severity,
    dataset: str,
    row_index: int | None,
    row: Mapping[str, Any] | None,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    findings.append(
        QualityFinding(
            rule_id=rule_id,
            severity=severity,
            dataset=dataset,
            row_index=row_index,
            symbol=str(row["symbol"]) if row is not None and row.get("symbol") else None,
            event_ts_ns=(
                int(row["event_ts_ns"])
                if row is not None and row.get("event_ts_ns") is not None
                else None
            ),
            message=message,
            details=details or {},
        )
    )


def _validate_event_clocks(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    findings: list[QualityFinding],
    max_silence_ns: int,
) -> None:
    previous: dict[tuple[str, str, str | None], tuple[int, int | None, int]] = {}
    for index, row in enumerate(rows):
        event_ts = int(row["event_ts_ns"])
        available_ts = int(row["available_ts_ns"])
        received = row.get("received_ts_ns")
        received_ts = int(received) if received is not None else None
        if available_ts < event_ts:
            _finding(
                findings,
                rule_id="temporal.available_before_event",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="observation is marked available before its exchange event time",
                details={"event_ts_ns": event_ts, "available_ts_ns": available_ts},
            )
        if received_ts is not None and available_ts < received_ts:
            _finding(
                findings,
                rule_id="temporal.available_before_receipt",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="live observation is marked available before local receipt",
                details={"received_ts_ns": received_ts, "available_ts_ns": available_ts},
            )

        key = (str(row["venue"]), str(row["symbol"]), row.get("continuity_id"))
        prior = previous.get(key)
        if prior is not None:
            previous_event, previous_received, previous_index = prior
            if event_ts < previous_event:
                _finding(
                    findings,
                    rule_id="temporal.out_of_order_event_time",
                    severity="WARNING",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message="exchange timestamps reversed in source/capture order",
                    details={
                        "previous_row": previous_index,
                        "previous_event_ts_ns": previous_event,
                    },
                )
            if event_ts - previous_event > max_silence_ns:
                _finding(
                    findings,
                    rule_id="temporal.long_silence",
                    severity="WARNING",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message="time between events exceeded configured silence threshold",
                    details={"silence_ns": event_ts - previous_event},
                )
            if (
                received_ts is not None
                and previous_received is not None
                and received_ts < previous_received
            ):
                _finding(
                    findings,
                    rule_id="temporal.receive_clock_reversal",
                    severity="WARNING",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message="wall-clock receipt timestamp moved backwards",
                    details={
                        "previous_row": previous_index,
                        "previous_received_ts_ns": previous_received,
                    },
                )
        previous[key] = (event_ts, received_ts, index)


def _validate_trades(
    rows: list[dict[str, Any]], dataset: str, findings: list[QualityFinding]
) -> None:
    identities: dict[tuple[str, str, int], int] = {}
    for index, row in enumerate(rows):
        identity = (str(row["venue"]), str(row["symbol"]), int(row["trade_id"]))
        if identity in identities:
            _finding(
                findings,
                rule_id="trade.duplicate",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="duplicate trade identity was preserved",
                details={"first_row": identities[identity], "trade_id": identity[2]},
            )
        else:
            identities[identity] = index
        if int(row["price_ticks"]) <= 0 or float(row["price"]) <= 0.0:
            _finding(
                findings,
                rule_id="trade.nonpositive_price",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="trade price is zero or negative",
            )
        if int(row["quantity_lots"]) <= 0 or float(row["quantity"]) <= 0.0:
            _finding(
                findings,
                rule_id="trade.nonpositive_quantity",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="trade quantity is zero or negative",
            )
        expected_price = int(row["price_ticks"]) * float(row["tick_size"])
        expected_quantity = int(row["quantity_lots"]) * float(row["lot_size"])
        if abs(expected_price - float(row["price"])) > max(1e-12, abs(expected_price) * 1e-12):
            _finding(
                findings,
                rule_id="trade.price_scale_mismatch",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="floating price does not match exact ticks and tick size",
            )
        if abs(expected_quantity - float(row["quantity"])) > max(
            1e-12, abs(expected_quantity) * 1e-12
        ):
            _finding(
                findings,
                rule_id="trade.quantity_scale_mismatch",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="floating quantity does not match exact lots and lot size",
            )
        if row["aggressor_side"] not in {"buy", "sell"}:
            _finding(
                findings,
                rule_id="trade.invalid_aggressor_side",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="aggressor side is outside the normalized enum",
            )


def _validate_book_sequences(
    rows: list[dict[str, Any]], dataset: str, findings: list[QualityFinding]
) -> None:
    previous_end: dict[tuple[str, str, str | None], int] = {}
    for index, row in enumerate(rows):
        start = int(row["sequence_start"])
        end = int(row["sequence_end"])
        if end < start:
            _finding(
                findings,
                rule_id="sequence.invalid_range",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="sequence range ends before it starts",
            )
            continue
        key = (str(row["venue"]), str(row["symbol"]), row.get("continuity_id"))
        prior = previous_end.get(key)
        if prior is not None:
            expected = prior + 1
            if start > expected:
                _finding(
                    findings,
                    rule_id="sequence.missing_range",
                    severity="ERROR",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message="book sequence has a forward gap",
                    details={
                        "expected_sequence": expected,
                        "observed_start": start,
                        "missing_start": expected,
                        "missing_end": start - 1,
                    },
                )
            elif end <= prior:
                _finding(
                    findings,
                    rule_id="sequence.stale_or_duplicate",
                    severity="WARNING",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message="sequence event is fully stale or duplicated",
                    details={"previous_end": prior},
                )
        previous_end[key] = max(prior if prior is not None else end, end)


def _validate_books(
    rows: list[dict[str, Any]],
    dataset: str,
    findings: list[QualityFinding],
    max_spread_bps: float,
) -> None:
    _validate_book_sequences(rows, dataset, findings)
    for index, row in enumerate(rows):
        bid = float(row["best_bid"])
        ask = float(row["best_ask"])
        bid_quantity = float(row["bid_quantity"])
        ask_quantity = float(row["ask_quantity"])
        mid = float(row["mid_price"])
        tick_size = float(row["tick_size"])
        lot_size = float(row["lot_size"])
        expected_bid = int(row["best_bid_ticks"]) * tick_size
        expected_ask = int(row["best_ask_ticks"]) * tick_size
        expected_bid_quantity = int(row["bid_quantity_lots"]) * lot_size
        expected_ask_quantity = int(row["ask_quantity_lots"]) * lot_size
        if abs(expected_bid - bid) > max(1e-12, abs(expected_bid) * 1e-12) or abs(
            expected_ask - ask
        ) > max(1e-12, abs(expected_ask) * 1e-12):
            _finding(
                findings,
                rule_id="book.price_scale_mismatch",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="floating best prices do not match exact ticks and tick size",
            )
        if abs(expected_bid_quantity - bid_quantity) > max(
            1e-12, abs(expected_bid_quantity) * 1e-12
        ) or abs(expected_ask_quantity - ask_quantity) > max(
            1e-12, abs(expected_ask_quantity) * 1e-12
        ):
            _finding(
                findings,
                rule_id="book.quantity_scale_mismatch",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="floating best quantities do not match exact lots and lot size",
            )
        if bid <= 0.0 or ask <= 0.0:
            _finding(
                findings,
                rule_id="book.nonpositive_price",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="best price is zero or negative",
            )
        if bid_quantity <= 0.0 or ask_quantity <= 0.0:
            _finding(
                findings,
                rule_id="book.nonpositive_quantity",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="best-level quantity is zero or negative",
            )
        if bid >= ask:
            _finding(
                findings,
                rule_id="book.crossed_or_locked",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="best bid is greater than or equal to best ask",
                details={"best_bid": bid, "best_ask": ask},
            )
        if mid > 0.0:
            relative_spread_bps = (ask - bid) / mid * 10_000.0
            if relative_spread_bps > max_spread_bps:
                _finding(
                    findings,
                    rule_id="book.abnormal_spread",
                    severity="WARNING",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message="relative spread exceeded configured threshold",
                    details={
                        "spread_bps": relative_spread_bps,
                        "threshold_bps": max_spread_bps,
                    },
                )
        microprice = float(row["microprice"])
        if not bid <= microprice <= ask:
            _finding(
                findings,
                rule_id="book.microprice_outside_quotes",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="microprice is outside the contemporaneous quotes",
            )
        for side in ("bid", "ask"):
            depth_1 = float(row[f"depth_{side}_1"])
            depth_5 = float(row[f"depth_{side}_5"])
            depth_10 = float(row[f"depth_{side}_10"])
            if not 0.0 < depth_1 <= depth_5 <= depth_10:
                _finding(
                    findings,
                    rule_id="book.nonmonotone_depth",
                    severity="ERROR",
                    dataset=dataset,
                    row_index=index,
                    row=row,
                    message=f"cumulative {side} depth is not positive and monotone",
                )
        if not bool(row["is_valid"]):
            _finding(
                findings,
                rule_id="book.invalid_state",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="reconstructor marked this state invalid",
            )


def _validate_depth_deltas(
    rows: list[dict[str, Any]], dataset: str, findings: list[QualityFinding]
) -> None:
    previous_end: dict[tuple[str, str, str | None], int] = {}
    for index, row in enumerate(rows):
        start = int(row["first_update_id"])
        end = int(row["last_update_id"])
        if end < start:
            _finding(
                findings,
                rule_id="sequence.invalid_range",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="delta sequence range ends before it starts",
            )
        else:
            key = (str(row["venue"]), str(row["symbol"]), row.get("continuity_id"))
            prior = previous_end.get(key)
            if prior is not None:
                expected = prior + 1
                previous_hint = row.get("previous_update_id")
                if previous_hint is not None and int(previous_hint) != prior:
                    _finding(
                        findings,
                        rule_id="sequence.previous_id_mismatch",
                        severity="ERROR",
                        dataset=dataset,
                        row_index=index,
                        row=row,
                        message="delta previous-update hint does not match the prior event",
                        details={"expected_previous": prior, "observed_previous": previous_hint},
                    )
                if start > expected:
                    _finding(
                        findings,
                        rule_id="sequence.missing_range",
                        severity="ERROR",
                        dataset=dataset,
                        row_index=index,
                        row=row,
                        message="depth delta sequence has a forward gap",
                        details={
                            "expected_sequence": expected,
                            "observed_start": start,
                            "missing_start": expected,
                            "missing_end": start - 1,
                        },
                    )
                elif end <= prior:
                    _finding(
                        findings,
                        rule_id="sequence.stale_or_duplicate",
                        severity="WARNING",
                        dataset=dataset,
                        row_index=index,
                        row=row,
                        message="depth delta is fully stale or duplicated",
                        details={"previous_end": prior},
                    )
            previous_end[key] = max(prior if prior is not None else end, end)
        for side in ("bids", "asks"):
            seen_prices: set[int] = set()
            for level in row[side]:
                price_ticks = int(level["price_ticks"])
                quantity_lots = int(level["quantity_lots"])
                if price_ticks <= 0:
                    _finding(
                        findings,
                        rule_id="depth.nonpositive_price",
                        severity="ERROR",
                        dataset=dataset,
                        row_index=index,
                        row=row,
                        message="depth change contains a zero or negative price",
                    )
                # Zero is a documented delete instruction, not bad quantity.
                if quantity_lots < 0:
                    _finding(
                        findings,
                        rule_id="depth.negative_quantity",
                        severity="ERROR",
                        dataset=dataset,
                        row_index=index,
                        row=row,
                        message="depth change contains a negative quantity",
                    )
                if price_ticks in seen_prices:
                    _finding(
                        findings,
                        rule_id="depth.duplicate_price_in_event",
                        severity="WARNING",
                        dataset=dataset,
                        row_index=index,
                        row=row,
                        message="one delta updates the same side/price more than once",
                        details={"side": side, "price_ticks": price_ticks},
                    )
                seen_prices.add(price_ticks)


def validate_table(
    table: pa.Table,
    schema_name: str,
    *,
    max_spread_bps: float = 100.0,
    max_silence_ns: int = 5_000_000_000,
) -> ValidationReport:
    """Validate without sorting, de-duplicating, clipping, or changing rows."""
    ensure_schema(table, schema_name)
    rows = table.to_pylist()
    findings: list[QualityFinding] = []
    if schema_name in {"trades", "book_observations", "depth_deltas"}:
        _validate_event_clocks(
            rows,
            dataset=schema_name,
            findings=findings,
            max_silence_ns=max_silence_ns,
        )
    if schema_name == "trades":
        _validate_trades(rows, schema_name, findings)
    elif schema_name == "book_observations":
        _validate_books(rows, schema_name, findings, max_spread_bps)
    elif schema_name == "depth_deltas":
        _validate_depth_deltas(rows, schema_name, findings)
    return ValidationReport(
        dataset=schema_name,
        rows_checked=table.num_rows,
        findings=tuple(findings),
    )
