"""Pure snapshot-plus-delta order-book reconstruction.

Sequence order is authoritative.  Exchange timestamps are never used to sort
updates, so a timestamp reversal is reportable without concealing or inventing
book continuity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure.data.schemas import SCHEMA_VERSION, table_from_records

BookLevel = tuple[int, int]
_MAX_BOOK_LEVELS_PER_SIDE = 10_000


class BookInvariantError(ValueError):
    """Raised when a snapshot or delta contains an impossible book level."""


class _BookCapacityError(BookInvariantError):
    """Raised before retained book state can exceed its hard memory bound."""


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    venue: str
    symbol: str
    snapshot_id: str
    request_ts_ns: int
    received_ts_ns: int
    available_ts_ns: int
    continuity_id: str
    last_update_id: int
    depth_limit: int
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    tick_size: float
    lot_size: float
    source_artifact_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "venue": self.venue,
            "symbol": self.symbol,
            "snapshot_id": self.snapshot_id,
            "request_ts_ns": self.request_ts_ns,
            "received_ts_ns": self.received_ts_ns,
            "available_ts_ns": self.available_ts_ns,
            "continuity_id": self.continuity_id,
            "last_update_id": self.last_update_id,
            "depth_limit": self.depth_limit,
            "bids": [
                {"price_ticks": price, "quantity_lots": quantity} for price, quantity in self.bids
            ],
            "asks": [
                {"price_ticks": price, "quantity_lots": quantity} for price, quantity in self.asks
            ],
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "source_artifact_id": self.source_artifact_id,
        }


@dataclass(frozen=True, slots=True)
class DepthDelta:
    venue: str
    symbol: str
    event_ts_ns: int
    received_ts_ns: int | None
    available_ts_ns: int
    availability_basis: str
    capture_seq: int | None
    continuity_id: str
    first_update_id: int
    last_update_id: int
    previous_update_id: int | None
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    tick_size: float
    lot_size: float
    source_artifact_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "venue": self.venue,
            "symbol": self.symbol,
            "event_ts_ns": self.event_ts_ns,
            "received_ts_ns": self.received_ts_ns,
            "available_ts_ns": self.available_ts_ns,
            "availability_basis": self.availability_basis,
            "capture_seq": self.capture_seq,
            "continuity_id": self.continuity_id,
            "first_update_id": self.first_update_id,
            "last_update_id": self.last_update_id,
            "previous_update_id": self.previous_update_id,
            "bids": [
                {"price_ticks": price, "quantity_lots": quantity} for price, quantity in self.bids
            ],
            "asks": [
                {"price_ticks": price, "quantity_lots": quantity} for price, quantity in self.asks
            ],
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "source_artifact_id": self.source_artifact_id,
        }


@dataclass(frozen=True, slots=True)
class SequenceGap:
    venue: str
    symbol: str
    continuity_id: str
    expected_sequence: int
    observed_sequence_start: int
    observed_sequence_end: int
    missing_start: int
    missing_end: int
    detected_ts_ns: int
    reason: str
    source_artifact_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "venue": self.venue,
            "symbol": self.symbol,
            "continuity_id": self.continuity_id,
            "expected_sequence": self.expected_sequence,
            "observed_sequence_start": self.observed_sequence_start,
            "observed_sequence_end": self.observed_sequence_end,
            "missing_start": self.missing_start,
            "missing_end": self.missing_end,
            "detected_ts_ns": self.detected_ts_ns,
            "reason": self.reason,
            "source_artifact_id": self.source_artifact_id,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    status: Literal["LIVE", "GAPPED", "INVALID"]
    observations: pa.Table
    gaps: pa.Table
    stale_events: int
    final_update_id: int


ReconstructionOutcome = Literal[
    "OBSERVED",
    "STALE",
    "GAP",
    "INVALID",
    "EXCLUDED_AFTER_TERMINAL",
]


@dataclass(frozen=True, slots=True)
class ReconstructionStep:
    """Bounded result of applying one delta to one continuity epoch."""

    outcome: ReconstructionOutcome
    observation: Mapping[str, object] | None
    gap: SequenceGap | None


def _levels_to_book(levels: tuple[BookLevel, ...], side: str) -> dict[int, int]:
    if len(levels) > _MAX_BOOK_LEVELS_PER_SIDE:
        raise _BookCapacityError(
            f"{side} snapshot exceeds {_MAX_BOOK_LEVELS_PER_SIDE} retained levels"
        )
    result: dict[int, int] = {}
    for price, quantity in levels:
        if price <= 0:
            raise BookInvariantError(f"{side} snapshot price must be positive: {price}")
        if quantity <= 0:
            raise BookInvariantError(f"{side} snapshot quantity must be positive: {quantity}")
        if price in result:
            raise BookInvariantError(f"duplicate {side} snapshot price: {price}")
        result[price] = quantity
    if not result:
        raise BookInvariantError(f"{side} snapshot must not be empty")
    return result


def _apply_side(book: dict[int, int], changes: tuple[BookLevel, ...], side: str) -> None:
    for price, quantity in changes:
        if price <= 0:
            raise BookInvariantError(f"{side} delta price must be positive: {price}")
        if quantity < 0:
            raise BookInvariantError(f"{side} delta quantity must not be negative: {quantity}")
        if quantity == 0:
            book.pop(price, None)
        else:
            book[price] = quantity
    if len(book) > _MAX_BOOK_LEVELS_PER_SIDE:
        raise _BookCapacityError(f"{side} book exceeds {_MAX_BOOK_LEVELS_PER_SIDE} retained levels")


def _depth(book: dict[int, int], *, bids: bool, levels: int, lot_size: float) -> float:
    ordered = sorted(book, reverse=bids)[:levels]
    return sum(book[price] for price in ordered) * lot_size


def _queue_imbalance(bid_depth: float, ask_depth: float) -> float:
    total = bid_depth + ask_depth
    return (bid_depth - ask_depth) / total if total > 0.0 else 0.0


def _observation(
    *,
    snapshot: BookSnapshot,
    delta: DepthDelta,
    bids: dict[int, int],
    asks: dict[int, int],
    valid: bool,
) -> dict[str, object]:
    if not bids or not asks:
        raise BookInvariantError("delta removed every price level from one side of the book")
    best_bid_ticks = max(bids)
    best_ask_ticks = min(asks)
    bid_quantity_lots = bids[best_bid_ticks]
    ask_quantity_lots = asks[best_ask_ticks]
    best_bid = best_bid_ticks * snapshot.tick_size
    best_ask = best_ask_ticks * snapshot.tick_size
    bid_quantity = bid_quantity_lots * snapshot.lot_size
    ask_quantity = ask_quantity_lots * snapshot.lot_size
    mid_price = (best_bid + best_ask) / 2.0
    microprice = (best_ask * bid_quantity + best_bid * ask_quantity) / (bid_quantity + ask_quantity)
    depth_bid_1 = _depth(bids, bids=True, levels=1, lot_size=snapshot.lot_size)
    depth_ask_1 = _depth(asks, bids=False, levels=1, lot_size=snapshot.lot_size)
    depth_bid_5 = _depth(bids, bids=True, levels=5, lot_size=snapshot.lot_size)
    depth_ask_5 = _depth(asks, bids=False, levels=5, lot_size=snapshot.lot_size)
    depth_bid_10 = _depth(bids, bids=True, levels=10, lot_size=snapshot.lot_size)
    depth_ask_10 = _depth(asks, bids=False, levels=10, lot_size=snapshot.lot_size)
    return {
        "schema_version": SCHEMA_VERSION,
        "venue": snapshot.venue,
        "symbol": snapshot.symbol,
        "event_ts_ns": delta.event_ts_ns,
        "received_ts_ns": delta.received_ts_ns,
        "available_ts_ns": max(snapshot.available_ts_ns, delta.available_ts_ns),
        "availability_basis": delta.availability_basis,
        "capture_seq": delta.capture_seq,
        "continuity_id": snapshot.continuity_id,
        "sequence_start": delta.first_update_id,
        "sequence_end": delta.last_update_id,
        "is_valid": valid,
        "best_bid_ticks": best_bid_ticks,
        "best_ask_ticks": best_ask_ticks,
        "bid_quantity_lots": bid_quantity_lots,
        "ask_quantity_lots": ask_quantity_lots,
        "tick_size": snapshot.tick_size,
        "lot_size": snapshot.lot_size,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_quantity": bid_quantity,
        "ask_quantity": ask_quantity,
        "spread": best_ask - best_bid,
        "mid_price": mid_price,
        "microprice": microprice,
        "depth_bid_1": depth_bid_1,
        "depth_ask_1": depth_ask_1,
        "depth_bid_5": depth_bid_5,
        "depth_ask_5": depth_ask_5,
        "depth_bid_10": depth_bid_10,
        "depth_ask_10": depth_ask_10,
        "queue_imbalance_1": _queue_imbalance(depth_bid_1, depth_ask_1),
        "queue_imbalance_5": _queue_imbalance(depth_bid_5, depth_ask_5),
        "queue_imbalance_10": _queue_imbalance(depth_bid_10, depth_ask_10),
        "source_artifact_id": delta.source_artifact_id,
    }


def _gap(snapshot: BookSnapshot, delta: DepthDelta, expected: int, reason: str) -> SequenceGap:
    missing_end = max(expected, delta.first_update_id - 1)
    return SequenceGap(
        venue=snapshot.venue,
        symbol=snapshot.symbol,
        continuity_id=snapshot.continuity_id,
        expected_sequence=expected,
        observed_sequence_start=delta.first_update_id,
        observed_sequence_end=delta.last_update_id,
        missing_start=expected,
        missing_end=missing_end,
        detected_ts_ns=delta.available_ts_ns,
        reason=reason,
        source_artifact_id=delta.source_artifact_id,
    )


class IncrementalBookReconstructor:
    """Stateful O(book-depth) snapshot-plus-delta reconstruction.

    The class retains only the current epoch's book.  Every input delta returns
    an explicit outcome; after a gap or invalidation, later deltas receive an
    ``EXCLUDED_AFTER_TERMINAL`` gap record rather than disappearing silently.
    """

    def __init__(self, snapshot: BookSnapshot) -> None:
        if snapshot.available_ts_ns < snapshot.received_ts_ns:
            raise BookInvariantError("snapshot cannot be available before it was received")
        if (
            not math.isfinite(snapshot.tick_size)
            or not math.isfinite(snapshot.lot_size)
            or snapshot.tick_size <= 0
            or snapshot.lot_size <= 0
        ):
            raise BookInvariantError("snapshot tick and lot sizes must be finite and positive")
        if snapshot.depth_limit < 1 or snapshot.depth_limit > _MAX_BOOK_LEVELS_PER_SIDE:
            raise BookInvariantError(
                f"snapshot depth_limit must be within 1..{_MAX_BOOK_LEVELS_PER_SIDE}"
            )
        bids = _levels_to_book(snapshot.bids, "bid")
        asks = _levels_to_book(snapshot.asks, "ask")
        if max(bids) >= min(asks):
            raise BookInvariantError("snapshot is crossed or locked")
        self.snapshot = snapshot
        self._bids = bids
        self._asks = asks
        self._last_update_id = snapshot.last_update_id
        self._stale_events = 0
        self._status: Literal["LIVE", "GAPPED", "INVALID"] = "LIVE"

    @property
    def status(self) -> Literal["LIVE", "GAPPED", "INVALID"]:
        return self._status

    @property
    def stale_events(self) -> int:
        return self._stale_events

    @property
    def final_update_id(self) -> int:
        return self._last_update_id

    def _validate_identity(self, delta: DepthDelta) -> None:
        if delta.venue != self.snapshot.venue or delta.symbol != self.snapshot.symbol:
            raise BookInvariantError("delta venue/symbol does not match snapshot")
        if delta.tick_size != self.snapshot.tick_size or delta.lot_size != self.snapshot.lot_size:
            raise BookInvariantError("delta tick/lot scales do not match snapshot metadata")

    def update(self, delta: DepthDelta) -> ReconstructionStep:
        """Apply exactly one delta and return its explicit reconstruction disposition."""
        self._validate_identity(delta)
        expected = self._last_update_id + 1
        if self._status != "LIVE":
            return ReconstructionStep(
                outcome="EXCLUDED_AFTER_TERMINAL",
                observation=None,
                gap=_gap(
                    self.snapshot,
                    delta,
                    expected,
                    f"epoch_already_{self._status.lower()}",
                ),
            )
        if delta.continuity_id != self.snapshot.continuity_id:
            self._status = "GAPPED"
            return ReconstructionStep(
                outcome="GAP",
                observation=None,
                gap=_gap(self.snapshot, delta, expected, "continuity_id_mismatch"),
            )
        if delta.last_update_id < delta.first_update_id:
            self._status = "INVALID"
            return ReconstructionStep(
                outcome="INVALID",
                observation=None,
                gap=_gap(self.snapshot, delta, expected, "invalid_sequence_range"),
            )
        if delta.last_update_id <= self._last_update_id:
            self._stale_events += 1
            return ReconstructionStep(outcome="STALE", observation=None, gap=None)
        if (
            delta.previous_update_id is not None
            and delta.previous_update_id != self._last_update_id
        ):
            self._status = "GAPPED"
            return ReconstructionStep(
                outcome="GAP",
                observation=None,
                gap=_gap(self.snapshot, delta, expected, "previous_update_id_mismatch"),
            )
        if delta.first_update_id > expected:
            self._status = "GAPPED"
            return ReconstructionStep(
                outcome="GAP",
                observation=None,
                gap=_gap(self.snapshot, delta, expected, "forward_sequence_gap"),
            )
        if delta.last_update_id < expected:
            self._stale_events += 1
            return ReconstructionStep(outcome="STALE", observation=None, gap=None)

        candidate_bids = self._bids.copy()
        candidate_asks = self._asks.copy()
        try:
            _apply_side(candidate_bids, delta.bids, "bid")
            _apply_side(candidate_asks, delta.asks, "ask")
            if not candidate_bids or not candidate_asks:
                raise BookInvariantError("delta emptied one side of the order book")
            crossed = max(candidate_bids) >= min(candidate_asks)
            observation = _observation(
                snapshot=self.snapshot,
                delta=delta,
                bids=candidate_bids,
                asks=candidate_asks,
                valid=not crossed,
            )
        except _BookCapacityError:
            self._status = "INVALID"
            return ReconstructionStep(
                outcome="INVALID",
                observation=None,
                gap=_gap(self.snapshot, delta, expected, "book_level_limit_exceeded"),
            )
        except BookInvariantError:
            self._status = "INVALID"
            return ReconstructionStep(
                outcome="INVALID",
                observation=None,
                gap=_gap(self.snapshot, delta, expected, "invalid_book_level"),
            )

        self._bids = candidate_bids
        self._asks = candidate_asks
        self._last_update_id = delta.last_update_id
        if crossed:
            self._status = "INVALID"
            return ReconstructionStep(
                outcome="INVALID",
                observation=observation,
                gap=_gap(self.snapshot, delta, expected, "crossed_or_locked_book"),
            )
        return ReconstructionStep(outcome="OBSERVED", observation=observation, gap=None)


def reconstruct_snapshot_and_deltas(
    snapshot: BookSnapshot, deltas: tuple[DepthDelta, ...] | list[DepthDelta]
) -> ReconstructionResult:
    """Apply buffered/live deltas until an explicit gap or invariant failure.

    Stale events (``u <= last_update_id``) are counted and ignored.  A usable
    event must cover the next expected update ID, allowing safe overlap.  A
    forward gap invalidates the continuity epoch; later events are not emitted.
    """
    reconstructor = IncrementalBookReconstructor(snapshot)
    observation_records: list[dict[str, object]] = []
    gaps: list[SequenceGap] = []

    for delta in deltas:
        step = reconstructor.update(delta)
        if step.observation is not None:
            observation_records.append(dict(step.observation))
        if step.gap is not None:
            gaps.append(step.gap)
        if step.outcome in {"GAP", "INVALID"}:
            break

    return ReconstructionResult(
        status=reconstructor.status,
        observations=table_from_records("book_observations", observation_records),
        gaps=table_from_records("sequence_gaps", [item.to_record() for item in gaps]),
        stale_events=reconstructor.stale_events,
        final_update_id=reconstructor.final_update_id,
    )


def snapshots_table(snapshots: list[BookSnapshot] | tuple[BookSnapshot, ...]) -> pa.Table:
    return table_from_records("book_snapshots", [snapshot.to_record() for snapshot in snapshots])


def deltas_table(deltas: list[DepthDelta] | tuple[DepthDelta, ...]) -> pa.Table:
    return table_from_records("depth_deltas", [delta.to_record() for delta in deltas])
