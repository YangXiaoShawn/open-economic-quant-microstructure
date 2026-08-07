"""Non-mutating data-quality rules with explicit, machine-readable findings."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure.data.schemas import ensure_schema, get_schema
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
    total_errors: int | None = None
    total_warnings: int | None = None
    findings_jsonl_path: str | None = None

    def __post_init__(self) -> None:
        if self.rows_checked < 0:
            raise ValueError("rows_checked must be non-negative")
        if (self.total_errors is None) != (self.total_warnings is None):
            raise ValueError("total_errors and total_warnings must be supplied together")
        retained_errors = sum(item.severity == "ERROR" for item in self.findings)
        retained_warnings = sum(item.severity == "WARNING" for item in self.findings)
        if self.total_errors is not None and self.total_errors < retained_errors:
            raise ValueError("total_errors cannot be smaller than retained error findings")
        if self.total_warnings is not None and self.total_warnings < retained_warnings:
            raise ValueError("total_warnings cannot be smaller than retained warning findings")

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def error_count(self) -> int:
        if self.total_errors is not None:
            return self.total_errors
        return sum(item.severity == "ERROR" for item in self.findings)

    @property
    def warning_count(self) -> int:
        if self.total_warnings is not None:
            return self.total_warnings
        return sum(item.severity == "WARNING" for item in self.findings)

    @property
    def findings_truncated(self) -> bool:
        """Whether ``findings`` is only an in-memory preview of the full result."""
        return len(self.findings) < self.error_count + self.warning_count

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at_utc": utc_now_iso(),
            "dataset": self.dataset,
            "rows_checked": self.rows_checked,
            "summary": {"errors": self.error_count, "warnings": self.warning_count},
            "findings": [asdict(item) for item in self.findings],
            "mutation_policy": "observations were not changed or repaired",
        }
        if self.findings_truncated:
            payload["findings_preview"] = {
                "retained": len(self.findings),
                "total": self.error_count + self.warning_count,
                "truncated": True,
            }
        if self.findings_jsonl_path is not None:
            payload["findings_jsonl_path"] = self.findings_jsonl_path
        return payload

    def write_json(self, path: str | Path) -> None:
        write_json(path, self.to_dict())


class _FindingTarget(Protocol):
    def append(self, finding: QualityFinding) -> None: ...


class _FindingAccumulator:
    """Count every finding while retaining only a bounded in-memory preview."""

    def __init__(
        self,
        *,
        max_findings: int | None,
        findings_jsonl_path: str | Path | None,
    ) -> None:
        if max_findings is not None and max_findings < 0:
            raise ValueError("max_findings must be non-negative or None")
        self._max_findings = max_findings
        self._findings: list[QualityFinding] = []
        self.error_count = 0
        self.warning_count = 0
        self.path = Path(findings_jsonl_path) if findings_jsonl_path is not None else None
        self._temporary_path: Path | None = None
        self._sink: TextIO | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                text=True,
            )
            self._temporary_path = Path(temporary_name)
            self._sink = os.fdopen(descriptor, "w", encoding="utf-8")

    @property
    def findings(self) -> tuple[QualityFinding, ...]:
        return tuple(self._findings)

    def append(self, finding: QualityFinding) -> None:
        if finding.severity == "ERROR":
            self.error_count += 1
        else:
            self.warning_count += 1
        if self._max_findings is None or len(self._findings) < self._max_findings:
            self._findings.append(finding)
        if self._sink is not None:
            self._sink.write(
                json.dumps(
                    asdict(finding),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            self._sink.write("\n")

    def flush(self) -> None:
        if self._sink is not None:
            self._sink.flush()

    def publish(self) -> None:
        if self._sink is not None:
            self._sink.flush()
            os.fsync(self._sink.fileno())
            self._sink.close()
            self._sink = None
        if self.path is not None and self._temporary_path is not None:
            os.replace(self._temporary_path, self.path)
            self._temporary_path = None

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
            self._temporary_path = None


def _continuity_key(continuity_id: object) -> tuple[int, str]:
    if continuity_id is None:
        return (1, "")
    return (0, str(continuity_id))


class _SpillState:
    """Disk-backed exact state whose RAM use does not grow with row history."""

    def __init__(self) -> None:
        # An empty SQLite filename creates a private temporary on-disk database
        # which is deleted when the connection closes.
        self._connection = sqlite3.connect("")
        self._connection.execute("PRAGMA cache_size = -2048")
        self._connection.execute("PRAGMA temp_store = FILE")
        self._connection.execute("PRAGMA journal_mode = OFF")
        self._connection.execute("PRAGMA synchronous = OFF")
        self._connection.executescript(
            """
            CREATE TABLE event_state (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                continuity_is_null INTEGER NOT NULL,
                continuity_id TEXT NOT NULL,
                event_ts_ns INTEGER NOT NULL,
                received_ts_ns INTEGER,
                row_index INTEGER NOT NULL,
                PRIMARY KEY (venue, symbol, continuity_is_null, continuity_id)
            ) WITHOUT ROWID;
            CREATE TABLE sequence_state (
                sequence_kind TEXT NOT NULL,
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                continuity_is_null INTEGER NOT NULL,
                continuity_id TEXT NOT NULL,
                sequence_end INTEGER NOT NULL,
                PRIMARY KEY (
                    sequence_kind,
                    venue,
                    symbol,
                    continuity_is_null,
                    continuity_id
                )
            ) WITHOUT ROWID;
            CREATE TABLE trade_identity (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                first_row INTEGER NOT NULL,
                PRIMARY KEY (venue, symbol, trade_id)
            ) WITHOUT ROWID;
            """
        )

    def get_event(self, key: tuple[str, str, str | None]) -> tuple[int, int | None, int] | None:
        continuity_is_null, continuity_id = _continuity_key(key[2])
        result = self._connection.execute(
            """
            SELECT event_ts_ns, received_ts_ns, row_index
            FROM event_state
            WHERE venue = ? AND symbol = ?
              AND continuity_is_null = ? AND continuity_id = ?
            """,
            (key[0], key[1], continuity_is_null, continuity_id),
        ).fetchone()
        if result is None:
            return None
        event_ts_ns, received_ts_ns, row_index = result
        return (
            int(event_ts_ns),
            int(received_ts_ns) if received_ts_ns is not None else None,
            int(row_index),
        )

    def set_event(
        self,
        key: tuple[str, str, str | None],
        value: tuple[int, int | None, int],
    ) -> None:
        continuity_is_null, continuity_id = _continuity_key(key[2])
        self._connection.execute(
            """
            INSERT INTO event_state (
                venue, symbol, continuity_is_null, continuity_id,
                event_ts_ns, received_ts_ns, row_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (venue, symbol, continuity_is_null, continuity_id)
            DO UPDATE SET
                event_ts_ns = excluded.event_ts_ns,
                received_ts_ns = excluded.received_ts_ns,
                row_index = excluded.row_index
            """,
            (*key[:2], continuity_is_null, continuity_id, *value),
        )

    def get_sequence(self, kind: str, key: tuple[str, str, str | None]) -> int | None:
        continuity_is_null, continuity_id = _continuity_key(key[2])
        result = self._connection.execute(
            """
            SELECT sequence_end
            FROM sequence_state
            WHERE sequence_kind = ? AND venue = ? AND symbol = ?
              AND continuity_is_null = ? AND continuity_id = ?
            """,
            (kind, key[0], key[1], continuity_is_null, continuity_id),
        ).fetchone()
        return int(result[0]) if result is not None else None

    def set_sequence(
        self,
        kind: str,
        key: tuple[str, str, str | None],
        sequence_end: int,
    ) -> None:
        continuity_is_null, continuity_id = _continuity_key(key[2])
        self._connection.execute(
            """
            INSERT INTO sequence_state (
                sequence_kind, venue, symbol, continuity_is_null,
                continuity_id, sequence_end
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                sequence_kind, venue, symbol, continuity_is_null, continuity_id
            ) DO UPDATE SET sequence_end = excluded.sequence_end
            """,
            (kind, key[0], key[1], continuity_is_null, continuity_id, sequence_end),
        )

    def first_trade_row_or_insert(
        self, identity: tuple[str, str, int], row_index: int
    ) -> int | None:
        try:
            self._connection.execute(
                """
                INSERT INTO trade_identity (venue, symbol, trade_id, first_row)
                VALUES (?, ?, ?, ?)
                """,
                (*identity, row_index),
            )
        except sqlite3.IntegrityError:
            result = self._connection.execute(
                """
                SELECT first_row FROM trade_identity
                WHERE venue = ? AND symbol = ? AND trade_id = ?
                """,
                identity,
            ).fetchone()
            if result is None:  # pragma: no cover - guarded by the primary key
                raise RuntimeError("duplicate identity disappeared from quality state") from None
            return int(result[0])
        return None

    def commit(self) -> None:
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


class _ValidationState(Protocol):
    def get_event(self, key: tuple[str, str, str | None]) -> tuple[int, int | None, int] | None: ...

    def set_event(
        self,
        key: tuple[str, str, str | None],
        value: tuple[int, int | None, int],
    ) -> None: ...

    def get_sequence(self, kind: str, key: tuple[str, str, str | None]) -> int | None: ...

    def set_sequence(
        self,
        kind: str,
        key: tuple[str, str, str | None],
        sequence_end: int,
    ) -> None: ...

    def first_trade_row_or_insert(
        self, identity: tuple[str, str, int], row_index: int
    ) -> int | None: ...


class _MemoryState:
    def __init__(self) -> None:
        self._events: dict[tuple[str, str, str | None], tuple[int, int | None, int]] = {}
        self._sequences: dict[tuple[str, str, str, str | None], int] = {}
        self._trade_identities: dict[tuple[str, str, int], int] = {}

    def get_event(self, key: tuple[str, str, str | None]) -> tuple[int, int | None, int] | None:
        return self._events.get(key)

    def set_event(
        self,
        key: tuple[str, str, str | None],
        value: tuple[int, int | None, int],
    ) -> None:
        self._events[key] = value

    def get_sequence(self, kind: str, key: tuple[str, str, str | None]) -> int | None:
        return self._sequences.get((kind, *key))

    def set_sequence(
        self,
        kind: str,
        key: tuple[str, str, str | None],
        sequence_end: int,
    ) -> None:
        self._sequences[(kind, *key)] = sequence_end

    def first_trade_row_or_insert(
        self, identity: tuple[str, str, int], row_index: int
    ) -> int | None:
        first_row = self._trade_identities.get(identity)
        if first_row is None:
            self._trade_identities[identity] = row_index
        return first_row


def _finding(
    findings: _FindingTarget,
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
    findings: _FindingTarget,
    max_silence_ns: int,
    row_offset: int = 0,
    state: _ValidationState | None = None,
) -> None:
    validation_state = state if state is not None else _MemoryState()
    for local_index, row in enumerate(rows):
        index = row_offset + local_index
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
        prior = validation_state.get_event(key)
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
        validation_state.set_event(key, (event_ts, received_ts, index))


def _validate_trades(
    rows: list[dict[str, Any]],
    dataset: str,
    findings: _FindingTarget,
    *,
    row_offset: int = 0,
    state: _ValidationState | None = None,
) -> None:
    validation_state = state if state is not None else _MemoryState()
    for local_index, row in enumerate(rows):
        index = row_offset + local_index
        identity = (str(row["venue"]), str(row["symbol"]), int(row["trade_id"]))
        first_row = validation_state.first_trade_row_or_insert(identity, index)
        if first_row is not None:
            _finding(
                findings,
                rule_id="trade.duplicate",
                severity="ERROR",
                dataset=dataset,
                row_index=index,
                row=row,
                message="duplicate trade identity was preserved",
                details={"first_row": first_row, "trade_id": identity[2]},
            )
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
    rows: list[dict[str, Any]],
    dataset: str,
    findings: _FindingTarget,
    *,
    row_offset: int = 0,
    state: _ValidationState | None = None,
) -> None:
    validation_state = state if state is not None else _MemoryState()
    for local_index, row in enumerate(rows):
        index = row_offset + local_index
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
        prior = validation_state.get_sequence("book_observations", key)
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
        validation_state.set_sequence(
            "book_observations",
            key,
            max(prior if prior is not None else end, end),
        )


def _validate_books(
    rows: list[dict[str, Any]],
    dataset: str,
    findings: _FindingTarget,
    max_spread_bps: float,
    *,
    row_offset: int = 0,
    state: _ValidationState | None = None,
) -> None:
    _validate_book_sequences(
        rows,
        dataset,
        findings,
        row_offset=row_offset,
        state=state,
    )
    for local_index, row in enumerate(rows):
        index = row_offset + local_index
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
    rows: list[dict[str, Any]],
    dataset: str,
    findings: _FindingTarget,
    *,
    row_offset: int = 0,
    state: _ValidationState | None = None,
) -> None:
    validation_state = state if state is not None else _MemoryState()
    for local_index, row in enumerate(rows):
        index = row_offset + local_index
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
            prior = validation_state.get_sequence("depth_deltas", key)
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
            validation_state.set_sequence(
                "depth_deltas",
                key,
                max(prior if prior is not None else end, end),
            )
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


class IncrementalQualityValidator:
    """Validate normalized Arrow batches without retaining the row history.

    Clock, sequence, and exact trade-identity state spill to a private temporary
    SQLite database. ``findings`` in the final report is a bounded preview, while
    total severity counts remain exact. Supplying ``findings_jsonl_path`` streams
    every finding to JSONL in detection order.
    """

    def __init__(
        self,
        schema_name: str,
        *,
        max_spread_bps: float = 100.0,
        max_silence_ns: int = 5_000_000_000,
        max_findings: int | None = 1_000,
        findings_jsonl_path: str | Path | None = None,
        row_chunk_size: int = 16_384,
    ) -> None:
        get_schema(schema_name)
        if row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")
        self.schema_name = schema_name
        self.max_spread_bps = max_spread_bps
        self.max_silence_ns = max_silence_ns
        self.row_chunk_size = row_chunk_size
        self._state = _SpillState()
        try:
            self._findings = _FindingAccumulator(
                max_findings=max_findings,
                findings_jsonl_path=findings_jsonl_path,
            )
        except BaseException:
            self._state.close()
            raise
        self._rows_checked = 0
        self._closed = False
        self._report: ValidationReport | None = None

    def __enter__(self) -> IncrementalQualityValidator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.close()

    @property
    def rows_checked(self) -> int:
        return self._rows_checked

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("incremental quality validator is already closed")

    def _validate_rows(self, rows: list[dict[str, Any]]) -> None:
        row_offset = self._rows_checked
        if self.schema_name in {"trades", "book_observations", "depth_deltas"}:
            _validate_event_clocks(
                rows,
                dataset=self.schema_name,
                findings=self._findings,
                max_silence_ns=self.max_silence_ns,
                row_offset=row_offset,
                state=self._state,
            )
        if self.schema_name == "trades":
            _validate_trades(
                rows,
                self.schema_name,
                self._findings,
                row_offset=row_offset,
                state=self._state,
            )
        elif self.schema_name == "book_observations":
            _validate_books(
                rows,
                self.schema_name,
                self._findings,
                self.max_spread_bps,
                row_offset=row_offset,
                state=self._state,
            )
        elif self.schema_name == "depth_deltas":
            _validate_depth_deltas(
                rows,
                self.schema_name,
                self._findings,
                row_offset=row_offset,
                state=self._state,
            )
        self._rows_checked += len(rows)

    def update(self, batch: pa.Table | pa.RecordBatch) -> None:
        """Consume one table or record batch without changing its observations."""
        self._require_open()
        if not isinstance(batch, (pa.Table, pa.RecordBatch)):
            raise TypeError("batch must be a pyarrow Table or RecordBatch")
        if batch.num_rows == 0:
            ensure_schema(batch, self.schema_name)
            return
        for start in range(0, batch.num_rows, self.row_chunk_size):
            chunk = batch.slice(start, self.row_chunk_size)
            ensure_schema(chunk, self.schema_name)
            self._validate_rows(chunk.to_pylist())
            self._state.commit()
            self._findings.flush()

    def finish(self) -> ValidationReport:
        """Close spill resources and return the exact-count validation report."""
        if self._report is not None:
            return self._report
        self._require_open()
        self._state.commit()
        self._findings.flush()
        self._findings.publish()
        self._state.close()
        self._closed = True
        jsonl_path = self._findings.path
        self._report = ValidationReport(
            dataset=self.schema_name,
            rows_checked=self._rows_checked,
            findings=self._findings.findings,
            total_errors=self._findings.error_count,
            total_warnings=self._findings.warning_count,
            findings_jsonl_path=str(jsonl_path.resolve()) if jsonl_path is not None else None,
        )
        return self._report

    def close(self) -> None:
        """Release resources without fabricating a report for unfinished input."""
        if self._closed:
            return
        self._findings.close()
        self._state.close()
        self._closed = True


def validate_batches(
    batches: Iterable[pa.Table | pa.RecordBatch],
    schema_name: str,
    *,
    max_spread_bps: float = 100.0,
    max_silence_ns: int = 5_000_000_000,
    max_findings: int | None = 1_000,
    findings_jsonl_path: str | Path | None = None,
    row_chunk_size: int = 16_384,
) -> ValidationReport:
    """Consume an iterable once and validate it with bounded retained state."""
    validator = IncrementalQualityValidator(
        schema_name,
        max_spread_bps=max_spread_bps,
        max_silence_ns=max_silence_ns,
        max_findings=max_findings,
        findings_jsonl_path=findings_jsonl_path,
        row_chunk_size=row_chunk_size,
    )
    try:
        for batch in batches:
            validator.update(batch)
        return validator.finish()
    except BaseException:
        validator.close()
        raise


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
