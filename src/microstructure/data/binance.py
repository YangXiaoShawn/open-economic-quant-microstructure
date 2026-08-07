"""Public Binance Spot market-data adapters; no authenticated/trading endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pyarrow as pa  # type: ignore[import-untyped]
import requests
import websockets
from websockets.exceptions import WebSocketException

from microstructure.data.book import BookSnapshot, DepthDelta
from microstructure.data.schemas import SCHEMA_VERSION, table_from_records
from microstructure.data.storage import write_source_manifest

_NS_PER_MILLISECOND = 1_000_000
_NS_PER_MICROSECOND = 1_000


class BinanceError(RuntimeError):
    """Base error for public Binance market-data collection."""


class BinanceHTTPError(BinanceError):
    """Raised when a public market-data GET cannot be completed safely."""


class BinancePayloadError(BinanceError):
    """Raised when Binance returns malformed or scale-incompatible data."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.base_delay_seconds < 0.0 or self.max_delay_seconds < 0.0:
            raise ValueError("retry delays must not be negative")


@dataclass(frozen=True, slots=True)
class RawPage:
    path: Path
    manifest_path: Path
    sha256: str
    request_uri: str
    row_count: int


@dataclass(frozen=True, slots=True)
class BinanceDownloadResult:
    trades: pa.Table
    raw_pages: tuple[RawPage, ...]
    requested_start_ns: int
    requested_end_ns: int
    complete_range: bool


@dataclass(frozen=True, slots=True)
class CapturedDepth:
    raw_payload: str
    delta: DepthDelta


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    """Public symbol filters captured before tick/lot normalization."""

    venue: str
    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    lot_size: Decimal
    min_price: Decimal
    max_price: Decimal
    min_quantity: Decimal
    max_quantity: Decimal
    observed_ts_ns: int
    source_artifact_id: str
    source_path: Path
    source_manifest_path: Path


class _WebSocketConnection(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


class _WebSocketContext(Protocol):
    async def __aenter__(self) -> _WebSocketConnection: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None: ...


ConnectFactory = Callable[[str], _WebSocketContext]


def _scaled_integer(value: str | Decimal, quantum: Decimal, label: str) -> int:
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(value)
        scaled = decimal_value / quantum
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise BinancePayloadError(f"invalid {label}: {value!r}") from exc
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise BinancePayloadError(
            f"{label} {decimal_value} is not aligned to configured scale {quantum}"
        )
    return int(integral)


def _event_timestamp_ns(value: int, unit: Literal["ms", "us"]) -> int:
    if value < 0:
        raise BinancePayloadError("event timestamp must not be negative")
    return value * (_NS_PER_MILLISECOND if unit == "ms" else _NS_PER_MICROSECOND)


def _safe_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    allowed = {"etag", "last-modified", "retry-after", "content-type"}
    return {
        str(key): str(value)
        for key, value in headers.items()
        if key.lower() in allowed or key.lower().startswith("x-mbx-used-weight")
    }


class BinancePublicClient:
    """Retrying client for market-data-only GET endpoints."""

    def __init__(
        self,
        *,
        base_url: str = "https://data-api.binance.vision",
        timeout_seconds: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self.session = session or requests.Session()
        self._sleep = sleep
        self._random_value = random_value

    def _request(self, path: str, params: Mapping[str, str | int]) -> requests.Response:
        url = f"{self.base_url}{path}"
        last_error: BaseException | None = None
        for attempt in range(self.retry_policy.max_retries + 1):
            response: requests.Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                last_error = exc
                retryable = True
            else:
                if response.status_code == 200:
                    return response
                retryable = response.status_code in {408, 418, 429} or response.status_code >= 500
                last_error = BinanceHTTPError(
                    f"GET {response.url} returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                if not retryable:
                    raise last_error

            if not retryable or attempt >= self.retry_policy.max_retries:
                break
            retry_after: float | None = None
            if response is not None and response.status_code in {418, 429}:
                raw_retry_after = response.headers.get("Retry-After")
                if raw_retry_after is not None:
                    try:
                        retry_after = max(0.0, float(raw_retry_after))
                    except ValueError:
                        retry_after = None
            exponential_cap = min(
                self.retry_policy.max_delay_seconds,
                self.retry_policy.base_delay_seconds * (2**attempt),
            )
            delay = (
                retry_after if retry_after is not None else exponential_cap * self._random_value()
            )
            self._sleep(delay)
        raise BinanceHTTPError(
            f"public Binance GET failed after {self.retry_policy.max_retries + 1} attempts"
        ) from last_error

    def fetch_exchange_info(self, *, symbol: str, raw_root: str | Path) -> SymbolMetadata:
        """Fetch public symbol status and exact PRICE_FILTER/LOT_SIZE scales."""
        symbol = symbol.upper()
        response = self._request("/api/v3/exchangeInfo", {"symbol": symbol})
        observed_ts_ns = time.time_ns()
        content = bytes(response.content)
        raw_page = _write_raw_response(
            content,
            raw_root=Path(raw_root),
            dataset="exchange_info",
            symbol=symbol,
            request_uri=str(response.url),
            downloaded_at_utc=_iso_from_ns(observed_ts_ns),
            requested_start_ns=None,
            requested_end_ns=None,
            response_headers=_safe_response_headers(response.headers),
        )
        try:
            payload = cast(dict[str, Any], response.json())
            symbols = cast(list[dict[str, Any]], payload["symbols"])
            if len(symbols) != 1 or str(symbols[0]["symbol"]).upper() != symbol:
                raise BinancePayloadError(
                    "exchangeInfo did not return exactly the requested symbol"
                )
            item = symbols[0]
            filters = {
                str(value["filterType"]): value
                for value in cast(list[dict[str, Any]], item["filters"])
            }
            price_filter = filters["PRICE_FILTER"]
            lot_filter = filters["LOT_SIZE"]
            tick_size = Decimal(str(price_filter["tickSize"]))
            lot_size = Decimal(str(lot_filter["stepSize"]))
            if tick_size <= 0 or lot_size <= 0:
                raise BinancePayloadError("exchangeInfo returned a nonpositive tick or lot size")
            return SymbolMetadata(
                venue="binance_spot",
                symbol=symbol,
                status=str(item["status"]),
                base_asset=str(item["baseAsset"]),
                quote_asset=str(item["quoteAsset"]),
                tick_size=tick_size,
                lot_size=lot_size,
                min_price=Decimal(str(price_filter["minPrice"])),
                max_price=Decimal(str(price_filter["maxPrice"])),
                min_quantity=Decimal(str(lot_filter["minQty"])),
                max_quantity=Decimal(str(lot_filter["maxQty"])),
                observed_ts_ns=observed_ts_ns,
                source_artifact_id=raw_page.sha256,
                source_path=raw_page.path,
                source_manifest_path=raw_page.manifest_path,
            )
        except BinancePayloadError:
            raise
        except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
            raise BinancePayloadError("malformed Binance exchangeInfo response") from exc

    def fetch_depth_snapshot(
        self,
        *,
        symbol: str,
        raw_root: str | Path,
        continuity_id: str,
        tick_size: Decimal | None = None,
        lot_size: Decimal | None = None,
        limit: int = 5000,
    ) -> BookSnapshot:
        """Fetch a public REST anchor; it intentionally has no exchange event time."""
        if limit not in {5, 10, 20, 50, 100, 500, 1000, 5000}:
            raise ValueError("unsupported Binance depth snapshot limit")
        symbol = symbol.upper()
        if (tick_size is None) != (lot_size is None):
            raise ValueError("tick_size and lot_size must be supplied together")
        if tick_size is None or lot_size is None:
            metadata = self.fetch_exchange_info(symbol=symbol, raw_root=raw_root)
            tick_size = metadata.tick_size
            lot_size = metadata.lot_size
        request_ts_ns = time.time_ns()
        response = self._request("/api/v3/depth", {"symbol": symbol, "limit": limit})
        received_ts_ns = time.time_ns()
        content = bytes(response.content)
        raw_page = _write_raw_response(
            content,
            raw_root=Path(raw_root),
            dataset="depth_snapshots",
            symbol=symbol,
            request_uri=str(response.url),
            downloaded_at_utc=_iso_from_ns(received_ts_ns),
            requested_start_ns=None,
            requested_end_ns=None,
            response_headers=_safe_response_headers(response.headers),
        )
        try:
            payload = cast(dict[str, Any], response.json())
            bids = tuple(
                (
                    _scaled_integer(str(item[0]), tick_size, "bid price"),
                    _scaled_integer(str(item[1]), lot_size, "bid quantity"),
                )
                for item in payload["bids"]
            )
            asks = tuple(
                (
                    _scaled_integer(str(item[0]), tick_size, "ask price"),
                    _scaled_integer(str(item[1]), lot_size, "ask quantity"),
                )
                for item in payload["asks"]
            )
            last_update_id = int(payload["lastUpdateId"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BinancePayloadError("malformed Binance depth snapshot") from exc
        return BookSnapshot(
            venue="binance_spot",
            symbol=symbol,
            snapshot_id=raw_page.sha256,
            request_ts_ns=request_ts_ns,
            received_ts_ns=received_ts_ns,
            available_ts_ns=received_ts_ns,
            continuity_id=continuity_id,
            last_update_id=last_update_id,
            depth_limit=limit,
            bids=bids,
            asks=asks,
            tick_size=float(tick_size),
            lot_size=float(lot_size),
            source_artifact_id=raw_page.sha256,
        )


class BinanceHistoricalTradeDownloader:
    """Bounded historical aggregate-trade downloader using the public REST API."""

    def __init__(
        self,
        *,
        client: BinancePublicClient,
        raw_root: str | Path,
        request_limit: int = 1000,
        tick_size: Decimal | None = None,
        lot_size: Decimal | None = None,
    ) -> None:
        if not 1 <= request_limit <= 1000:
            raise ValueError("request_limit must be in [1, 1000]")
        self.client = client
        self.raw_root = Path(raw_root)
        self.request_limit = request_limit
        if (tick_size is None) != (lot_size is None):
            raise ValueError("tick_size and lot_size must be supplied together")
        self.tick_size = tick_size
        self.lot_size = lot_size

    def download(
        self,
        *,
        symbol: str,
        start_ts_ns: int,
        end_ts_ns: int,
        max_events: int,
    ) -> BinanceDownloadResult:
        """Download ``[start, end)`` without timestamp-based page-boundary loss.

        The first page uses time bounds.  Later pages advance by inclusive
        aggregate-trade ID (``last_id + 1``), avoiding omissions when many
        trades share a millisecond timestamp.  Raw response bytes are retained
        before normalization.
        """
        if end_ts_ns <= start_ts_ns:
            raise ValueError("end_ts_ns must be after start_ts_ns")
        if max_events < 1:
            raise ValueError("max_events must be positive")
        symbol = symbol.upper()
        tick_size = self.tick_size
        lot_size = self.lot_size
        if tick_size is None or lot_size is None:
            metadata = self.client.fetch_exchange_info(symbol=symbol, raw_root=self.raw_root)
            tick_size = metadata.tick_size
            lot_size = metadata.lot_size
        params: dict[str, str | int] = {
            "symbol": symbol,
            "startTime": start_ts_ns // _NS_PER_MILLISECOND,
            "endTime": (end_ts_ns - 1) // _NS_PER_MILLISECOND,
            "limit": self.request_limit,
        }
        records: list[dict[str, object]] = []
        pages: list[RawPage] = []
        complete_range = True
        previous_last_id: int | None = None

        while True:
            response = self.client._request("/api/v3/aggTrades", params)
            downloaded_ns = time.time_ns()
            content = bytes(response.content)
            raw_page = _write_raw_response(
                content,
                raw_root=self.raw_root,
                dataset="agg_trades",
                symbol=symbol,
                request_uri=str(response.url),
                downloaded_at_utc=_iso_from_ns(downloaded_ns),
                requested_start_ns=start_ts_ns,
                requested_end_ns=end_ts_ns,
                response_headers=_safe_response_headers(response.headers),
            )
            try:
                payload = cast(list[dict[str, Any]], response.json())
                if not isinstance(payload, list):
                    raise TypeError("aggregate-trade response is not a list")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BinancePayloadError("malformed Binance aggregate-trade page") from exc
            pages.append(
                RawPage(
                    path=raw_page.path,
                    manifest_path=raw_page.manifest_path,
                    sha256=raw_page.sha256,
                    request_uri=raw_page.request_uri,
                    row_count=len(payload),
                )
            )
            if not payload:
                break

            page_ids: list[int] = []
            page_reached_end = False
            for item in payload:
                try:
                    aggregate_id = int(item["a"])
                    event_ts_ns = _event_timestamp_ns(int(item["T"]), "ms")
                    price_decimal = Decimal(str(item["p"]))
                    quantity_decimal = Decimal(str(item["q"]))
                    buyer_is_maker = bool(item["m"])
                except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
                    raise BinancePayloadError("malformed aggregate-trade record") from exc
                page_ids.append(aggregate_id)
                if event_ts_ns >= end_ts_ns:
                    page_reached_end = True
                    continue
                if event_ts_ns < start_ts_ns:
                    continue
                price_ticks = _scaled_integer(price_decimal, tick_size, "trade price")
                quantity_lots = _scaled_integer(quantity_decimal, lot_size, "trade quantity")
                price = float(price_decimal)
                quantity = float(quantity_decimal)
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "venue": "binance_spot",
                        "symbol": symbol,
                        "event_ts_ns": event_ts_ns,
                        "received_ts_ns": None,
                        "available_ts_ns": event_ts_ns,
                        "availability_basis": "exchange_event_time_proxy",
                        "capture_seq": None,
                        "continuity_id": None,
                        "trade_id": aggregate_id,
                        "first_trade_id": int(item.get("f", aggregate_id)),
                        "last_trade_id": int(item.get("l", aggregate_id)),
                        "price_ticks": price_ticks,
                        "quantity_lots": quantity_lots,
                        "tick_size": float(tick_size),
                        "lot_size": float(lot_size),
                        "price": price,
                        "quantity": quantity,
                        "quote_quantity": price * quantity,
                        "aggressor_side": "sell" if buyer_is_maker else "buy",
                        "buyer_is_maker": buyer_is_maker,
                        "source_artifact_id": raw_page.sha256,
                    }
                )
                if len(records) >= max_events:
                    complete_range = False
                    break

            if len(records) >= max_events:
                break
            last_id = max(page_ids)
            if previous_last_id is not None and last_id <= previous_last_id:
                raise BinancePayloadError("aggregate-trade pagination did not advance")
            previous_last_id = last_id
            if page_reached_end or len(payload) < self.request_limit:
                break
            params = {"symbol": symbol, "fromId": last_id + 1, "limit": self.request_limit}

        return BinanceDownloadResult(
            trades=table_from_records("trades", records),
            raw_pages=tuple(pages),
            requested_start_ns=start_ts_ns,
            requested_end_ns=end_ts_ns,
            complete_range=complete_range and bool(records),
        )


def _iso_from_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return f"{base}.{nanoseconds:09d}Z"


def _write_raw_response(
    content: bytes,
    *,
    raw_root: Path,
    dataset: str,
    symbol: str,
    request_uri: str,
    downloaded_at_utc: str,
    requested_start_ns: int | None,
    requested_end_ns: int | None,
    response_headers: Mapping[str, str],
) -> RawPage:
    checksum = hashlib.sha256(content).hexdigest()
    directory = raw_root / "binance_spot" / dataset / symbol.upper()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{checksum}.json"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
            raise BinancePayloadError(f"raw content-address collision at {destination}")
    else:
        handle, temporary_name = tempfile.mkstemp(dir=directory, prefix=".raw-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    manifest_path, _ = write_source_manifest(
        destination,
        source="binance_spot_public_api",
        source_uri=request_uri,
        downloaded_at_utc=downloaded_at_utc,
        requested_start_ns=requested_start_ns,
        requested_end_ns=requested_end_ns,
        response_headers=response_headers,
    )
    return RawPage(
        path=destination,
        manifest_path=manifest_path,
        sha256=checksum,
        request_uri=request_uri,
        row_count=0,
    )


def parse_depth_message(
    raw_message: str | bytes,
    *,
    received_ts_ns: int,
    capture_seq: int,
    continuity_id: str,
    tick_size: Decimal = Decimal("0.00000001"),
    lot_size: Decimal = Decimal("0.00000001"),
    timestamp_unit: Literal["ms", "us"] = "us",
) -> DepthDelta:
    """Normalize raw or combined-stream Spot ``U/u`` depth payloads."""
    raw_bytes = raw_message.encode() if isinstance(raw_message, str) else raw_message
    try:
        decoded = json.loads(raw_bytes)
        payload = decoded.get("data", decoded)
        if payload.get("e") != "depthUpdate":
            raise BinancePayloadError(f"unexpected websocket event: {payload.get('e')!r}")
        symbol = str(payload["s"]).upper()
        bids = tuple(
            (
                _scaled_integer(str(item[0]), tick_size, "bid price"),
                _scaled_integer(str(item[1]), lot_size, "bid quantity"),
            )
            for item in payload["b"]
        )
        asks = tuple(
            (
                _scaled_integer(str(item[0]), tick_size, "ask price"),
                _scaled_integer(str(item[1]), lot_size, "ask quantity"),
            )
            for item in payload["a"]
        )
        event_ts_ns = _event_timestamp_ns(int(payload["E"]), timestamp_unit)
        first_update_id = int(payload["U"])
        last_update_id = int(payload["u"])
        previous = payload.get("pu")
    except BinancePayloadError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BinancePayloadError("malformed Binance depth websocket message") from exc
    return DepthDelta(
        venue="binance_spot",
        symbol=symbol,
        event_ts_ns=event_ts_ns,
        received_ts_ns=received_ts_ns,
        available_ts_ns=received_ts_ns,
        availability_basis="local_receive_time",
        capture_seq=capture_seq,
        continuity_id=continuity_id,
        first_update_id=first_update_id,
        last_update_id=last_update_id,
        previous_update_id=int(previous) if previous is not None else None,
        bids=bids,
        asks=asks,
        tick_size=float(tick_size),
        lot_size=float(lot_size),
        source_artifact_id=hashlib.sha256(raw_bytes).hexdigest(),
    )


class BinanceLiveDepthCollector:
    """Optional reconnecting collector for public diff-depth streams."""

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        websocket_base_url: str = "wss://data-stream.binance.vision",
        tick_size: Decimal = Decimal("0.00000001"),
        lot_size: Decimal = Decimal("0.00000001"),
        max_reconnects: int = 5,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        if not symbols:
            raise ValueError("symbols must not be empty")
        self.symbols = tuple(symbol.upper() for symbol in symbols)
        streams = "/".join(f"{symbol.lower()}@depth@100ms" for symbol in self.symbols)
        self.url = f"{websocket_base_url.rstrip('/')}/stream?streams={streams}&timeUnit=MICROSECOND"
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_reconnects = max_reconnects
        self._connect_factory = connect_factory or cast(ConnectFactory, websockets.connect)

    async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
        """Yield exact raw payloads plus normalized deltas across continuity epochs."""
        capture_seq = 0
        yielded = 0
        reconnect = 0
        while reconnect <= self.max_reconnects:
            continuity_id = f"binance-live-{time.time_ns()}-{reconnect}"
            try:
                async with self._connect_factory(self.url) as connection:
                    async for raw in connection:
                        received_ts_ns = time.time_ns()
                        raw_text = raw.decode() if isinstance(raw, bytes) else raw
                        delta = parse_depth_message(
                            raw_text,
                            received_ts_ns=received_ts_ns,
                            capture_seq=capture_seq,
                            continuity_id=continuity_id,
                            tick_size=self.tick_size,
                            lot_size=self.lot_size,
                            timestamp_unit="us",
                        )
                        yield CapturedDepth(raw_payload=raw_text, delta=delta)
                        capture_seq += 1
                        yielded += 1
                        if max_messages is not None and yielded >= max_messages:
                            return
                reconnect += 1
            except (OSError, WebSocketException):
                reconnect += 1
                if reconnect > self.max_reconnects:
                    raise
                await asyncio.sleep(min(30.0, 0.5 * (2 ** (reconnect - 1))))


def depth_deltas_table(captured: tuple[CapturedDepth, ...] | list[CapturedDepth]) -> pa.Table:
    return table_from_records("depth_deltas", [item.delta.to_record() for item in captured])
