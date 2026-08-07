"""Deterministic, explicitly synthetic L1 and trade event generation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure.data.schemas import SCHEMA_VERSION, table_from_records

_NS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True, slots=True)
class SyntheticMarketData:
    """Small synthetic market tables; never evidence of observed market behavior."""

    trades: pa.Table
    book_observations: pa.Table
    evidence_tier: str = "SYNTHETIC_SMOKE"


def _symbol_seed(seed: int, symbol: str) -> int:
    material = f"{seed}:{symbol}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _initial_mid_ticks(symbol: str) -> int:
    if symbol.upper().startswith("BTC"):
        return 3_000_000
    if symbol.upper().startswith("ETH"):
        return 200_000
    return 100_000


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return (bid - ask) / total if total > 0.0 else 0.0


def _symbol_records(
    *,
    symbol: str,
    events: int,
    start_ts_ns: int,
    seed: int,
    event_spacing_ns: int,
    tick_size: float,
    lot_size: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(_symbol_seed(seed, symbol))
    mid_ticks = _initial_mid_ticks(symbol)
    trades: list[dict[str, object]] = []
    books: list[dict[str, object]] = []
    continuity_id = f"synthetic:{symbol}:0"
    source_artifact_id = f"synthetic-v1-seed-{seed}"

    for index in range(events):
        event_ts_ns = start_ts_ns + index * event_spacing_ns
        bid_level_lots = rng.randint(50, 250)
        ask_level_lots = rng.randint(50, 250)
        imbalance_1 = _imbalance(float(bid_level_lots), float(ask_level_lots))

        movement_draw = rng.random()
        upward_probability = 0.50 + 0.20 * imbalance_1
        if movement_draw < upward_probability - 0.10:
            mid_ticks += 1
        elif movement_draw > upward_probability + 0.10:
            mid_ticks -= 1
        mid_ticks = max(mid_ticks, 10)

        spread_ticks = 1 if rng.random() < 0.85 else 2
        best_bid_ticks = mid_ticks - spread_ticks // 2
        best_ask_ticks = best_bid_ticks + spread_ticks

        extra_bid_5 = sum(rng.randint(30, 180) for _ in range(4))
        extra_ask_5 = sum(rng.randint(30, 180) for _ in range(4))
        extra_bid_10 = sum(rng.randint(20, 140) for _ in range(5))
        extra_ask_10 = sum(rng.randint(20, 140) for _ in range(5))
        depth_bid_1_lots = bid_level_lots
        depth_ask_1_lots = ask_level_lots
        depth_bid_5_lots = depth_bid_1_lots + extra_bid_5
        depth_ask_5_lots = depth_ask_1_lots + extra_ask_5
        depth_bid_10_lots = depth_bid_5_lots + extra_bid_10
        depth_ask_10_lots = depth_ask_5_lots + extra_ask_10

        best_bid = best_bid_ticks * tick_size
        best_ask = best_ask_ticks * tick_size
        bid_quantity = bid_level_lots * lot_size
        ask_quantity = ask_level_lots * lot_size
        mid_price = (best_bid + best_ask) / 2.0
        microprice = (best_ask * bid_quantity + best_bid * ask_quantity) / (
            bid_quantity + ask_quantity
        )
        received_ts_ns = event_ts_ns + 100_000

        books.append(
            {
                "schema_version": SCHEMA_VERSION,
                "venue": "synthetic",
                "symbol": symbol,
                "event_ts_ns": event_ts_ns,
                "received_ts_ns": received_ts_ns,
                "available_ts_ns": received_ts_ns,
                "availability_basis": "synthetic_receipt",
                "capture_seq": index * 2,
                "continuity_id": continuity_id,
                "sequence_start": index + 1,
                "sequence_end": index + 1,
                "is_valid": True,
                "best_bid_ticks": best_bid_ticks,
                "best_ask_ticks": best_ask_ticks,
                "bid_quantity_lots": bid_level_lots,
                "ask_quantity_lots": ask_level_lots,
                "tick_size": tick_size,
                "lot_size": lot_size,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_quantity": bid_quantity,
                "ask_quantity": ask_quantity,
                "spread": best_ask - best_bid,
                "mid_price": mid_price,
                "microprice": microprice,
                "depth_bid_1": depth_bid_1_lots * lot_size,
                "depth_ask_1": depth_ask_1_lots * lot_size,
                "depth_bid_5": depth_bid_5_lots * lot_size,
                "depth_ask_5": depth_ask_5_lots * lot_size,
                "depth_bid_10": depth_bid_10_lots * lot_size,
                "depth_ask_10": depth_ask_10_lots * lot_size,
                "queue_imbalance_1": _imbalance(float(depth_bid_1_lots), float(depth_ask_1_lots)),
                "queue_imbalance_5": _imbalance(float(depth_bid_5_lots), float(depth_ask_5_lots)),
                "queue_imbalance_10": _imbalance(
                    float(depth_bid_10_lots), float(depth_ask_10_lots)
                ),
                "source_artifact_id": source_artifact_id,
            }
        )

        buy_probability = 0.50 + 0.25 * imbalance_1
        aggressor_side = "buy" if rng.random() < buy_probability else "sell"
        price_ticks = best_ask_ticks if aggressor_side == "buy" else best_bid_ticks
        quantity_lots = rng.randint(1, 40)
        trade_price = price_ticks * tick_size
        trade_quantity = quantity_lots * lot_size
        trade_event_ts_ns = event_ts_ns + 20_000
        trade_received_ts_ns = event_ts_ns + 150_000
        trades.append(
            {
                "schema_version": SCHEMA_VERSION,
                "venue": "synthetic",
                "symbol": symbol,
                "event_ts_ns": trade_event_ts_ns,
                "received_ts_ns": trade_received_ts_ns,
                "available_ts_ns": trade_received_ts_ns,
                "availability_basis": "synthetic_receipt",
                "capture_seq": index * 2 + 1,
                "continuity_id": continuity_id,
                "trade_id": index + 1,
                "first_trade_id": index + 1,
                "last_trade_id": index + 1,
                "price_ticks": price_ticks,
                "quantity_lots": quantity_lots,
                "tick_size": tick_size,
                "lot_size": lot_size,
                "price": trade_price,
                "quantity": trade_quantity,
                "quote_quantity": trade_price * trade_quantity,
                "aggressor_side": aggressor_side,
                "buyer_is_maker": aggressor_side == "sell",
                "source_artifact_id": source_artifact_id,
            }
        )

    return trades, books


def generate_synthetic_market(
    *,
    symbols: Sequence[str],
    events_per_symbol: int,
    start_ts_ns: int,
    seed: int,
    event_spacing_ns: int = 100 * _NS_PER_MILLISECOND,
    tick_size: float = 0.01,
    lot_size: float = 0.001,
) -> SyntheticMarketData:
    """Generate deterministic bounded tables for smoke tests and demos.

    The generator is intentionally labelled synthetic in every row and result.
    It is not calibrated to Binance and must never be reported as market data.
    """
    if events_per_symbol < 1:
        raise ValueError("events_per_symbol must be positive")
    if event_spacing_ns < 1:
        raise ValueError("event_spacing_ns must be positive")
    if tick_size <= 0.0 or lot_size <= 0.0:
        raise ValueError("tick_size and lot_size must be positive")
    if not symbols:
        raise ValueError("symbols must not be empty")

    trade_records: list[dict[str, object]] = []
    book_records: list[dict[str, object]] = []
    for raw_symbol in symbols:
        symbol = raw_symbol.upper()
        trades, books = _symbol_records(
            symbol=symbol,
            events=events_per_symbol,
            start_ts_ns=start_ts_ns,
            seed=seed,
            event_spacing_ns=event_spacing_ns,
            tick_size=tick_size,
            lot_size=lot_size,
        )
        trade_records.extend(trades)
        book_records.extend(books)
    return SyntheticMarketData(
        trades=table_from_records("trades", trade_records),
        book_observations=table_from_records("book_observations", book_records),
    )


def iter_table_batches(table: pa.Table, batch_size: int = 100_000) -> Iterator[pa.RecordBatch]:
    """Expose bounded RecordBatches for the streaming storage interface."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    yield from table.to_batches(max_chunksize=batch_size)
