"""Public Binance Spot market-data adapters; no authenticated/trading endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pyarrow as pa  # type: ignore[import-untyped]
import requests
import websockets
from websockets.exceptions import WebSocketException

from microstructure.data.book import BookSnapshot, DepthDelta
from microstructure.data.evidence_budget import RetainedEvidenceBudget
from microstructure.data.schemas import SCHEMA_VERSION, get_schema, table_from_records
from microstructure.data.storage import write_source_manifest
from microstructure.provenance import sha256_file

_NS_PER_MILLISECOND = 1_000_000
_NS_PER_MICROSECOND = 1_000


class BinanceError(RuntimeError):
    """Base error for public Binance market-data collection."""


class BinanceHTTPError(BinanceError):
    """Raised when a public market-data GET cannot be completed safely."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_exhausted = retry_exhausted


class BinancePayloadError(BinanceError):
    """Raised when Binance returns malformed or scale-incompatible data."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class BinanceMetadataContractError(BinancePayloadError):
    """A bounded exchangeInfo response violates the declared metadata contract."""


class BinanceResponseSizeLimitError(BinancePayloadError):
    """A public response violates its frozen body-size contract."""


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


class BinanceTradeStreamStopReason(StrEnum):
    """Why a normally exhausted historical-trade stream stopped."""

    EMPTY_PAGE = "empty_page"
    SHORT_PAGE = "short_page"
    RANGE_END = "range_end"
    EVENT_CAP = "event_cap"


@dataclass(frozen=True, slots=True)
class BinanceTradeStreamSummary:
    """Constant-size terminal metadata for a historical-trade stream."""

    requested_start_ns: int
    requested_end_ns: int
    rows_yielded: int
    raw_page_count: int
    stop_reason: BinanceTradeStreamStopReason
    complete_range: bool
    last_raw_page: RawPage | None


@dataclass(frozen=True, slots=True)
class CapturedDepth:
    raw_payload: str
    delta: DepthDelta


@dataclass(frozen=True, slots=True)
class RawDepthFrame:
    """One exact websocket frame timestamped before UTF-8/JSON normalization."""

    payload: bytes
    was_text: bool
    received_ts_ns: int
    capture_seq: int
    continuity_id: str


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
RawDepthFrameCallback = Callable[[RawDepthFrame], None]


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
    allowed = {"content-length", "content-type", "etag", "last-modified", "retry-after"}
    return {
        str(key): str(value)
        for key, value in headers.items()
        if key.lower() in allowed or key.lower().startswith("x-mbx-used-weight")
    }


def _close_response(response: object) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        cast(Callable[[], object], close)()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        max_response_bytes: int = 8 * 1024 * 1024,
        retained_evidence_budget: RetainedEvidenceBudget | None = None,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or RetryPolicy()
        self.session = session or requests.Session()
        self._sleep = sleep
        self._random_value = random_value
        self.max_response_bytes = max_response_bytes
        self.retained_evidence_budget = retained_evidence_budget

    def _session_get(
        self,
        url: str,
        params: Mapping[str, str | int],
        *,
        stream_response: bool,
    ) -> requests.Response:
        if not stream_response:
            return self.session.get(url, params=params, timeout=self.timeout_seconds)
        try:
            return self.session.get(
                url,
                params=params,
                timeout=self.timeout_seconds,
                stream=True,
            )
        except TypeError as exc:
            # Older injected test sessions may implement only the original
            # three-argument boundary.  Production requests.Session accepts
            # ``stream`` and therefore always takes the bounded transport path.
            if "unexpected keyword argument 'stream'" not in str(exc):
                raise
            return self.session.get(url, params=params, timeout=self.timeout_seconds)

    def _request(
        self,
        path: str,
        params: Mapping[str, str | int],
        *,
        stream_response: bool = False,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        last_error: BaseException | None = None
        for attempt in range(self.retry_policy.max_retries + 1):
            response: requests.Response | None = None
            try:
                response = self._session_get(
                    url,
                    params,
                    stream_response=stream_response,
                )
            except requests.RequestException as exc:
                last_error = exc
                retryable = True
            else:
                if response.status_code == 200:
                    return response
                retryable = response.status_code in {408, 418, 429} or response.status_code >= 500
                response_detail = (
                    "response body intentionally not materialized"
                    if stream_response
                    else response.text[:200]
                )
                last_error = BinanceHTTPError(
                    f"GET {response.url} returned HTTP {response.status_code}: {response_detail}",
                    status_code=response.status_code,
                )
                if not retryable:
                    _close_response(response)
                    raise last_error

            if not retryable or attempt >= self.retry_policy.max_retries:
                if response is not None:
                    _close_response(response)
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
            if response is not None:
                _close_response(response)
            self._sleep(delay)
        status_code = last_error.status_code if isinstance(last_error, BinanceHTTPError) else None
        raise BinanceHTTPError(
            f"public Binance GET failed after {self.retry_policy.max_retries + 1} attempts",
            status_code=status_code,
            retry_exhausted=True,
        ) from last_error

    def _bounded_request_body(
        self,
        path: str,
        params: Mapping[str, str | int],
        *,
        raw_root: Path,
        rejected_dataset: str,
        symbol: str,
        requested_start_ns: int | None,
        requested_end_ns: int | None,
        max_response_bytes: int | None = None,
    ) -> tuple[bytes, str, dict[str, str], int]:
        """Read one bounded body, retrying recoverable transport interruptions.

        Every interrupted attempt is persisted before retry.  Payload/size
        violations remain non-retryable because another identical response is
        not evidence of a transient network failure.
        """
        byte_ceiling = max_response_bytes or self.max_response_bytes
        for attempt in range(self.retry_policy.max_retries + 1):
            response = self._request(path, params, stream_response=True)
            observed_ts_ns = time.time_ns()
            request_uri = str(response.url)
            response_headers = _safe_response_headers(response.headers)
            bounded_body = _read_bounded_response_body(
                response,
                max_response_bytes=byte_ceiling,
            )
            if bounded_body.error_message is None:
                return (
                    bounded_body.content,
                    request_uri,
                    response_headers,
                    observed_ts_ns,
                )

            rejected_headers = _rejected_response_headers(response_headers, bounded_body)
            rejected_headers["x-local-body-attempt"] = str(attempt + 1)
            _write_raw_response(
                bounded_body.content,
                raw_root=raw_root,
                dataset=rejected_dataset,
                symbol=symbol,
                request_uri=request_uri,
                downloaded_at_utc=_iso_from_ns(observed_ts_ns),
                requested_start_ns=requested_start_ns,
                requested_end_ns=requested_end_ns,
                response_headers=rejected_headers,
                retained_evidence_budget=self.retained_evidence_budget,
            )
            if not bounded_body.retryable or attempt >= self.retry_policy.max_retries:
                if bounded_body.retryable:
                    raise BinancePayloadError(bounded_body.error_message, transient=True)
                raise BinanceResponseSizeLimitError(bounded_body.error_message)
            exponential_cap = min(
                self.retry_policy.max_delay_seconds,
                self.retry_policy.base_delay_seconds * (2**attempt),
            )
            self._sleep(exponential_cap * self._random_value())
        raise AssertionError("bounded response retry loop exhausted without a terminal result")

    def fetch_exchange_info(self, *, symbol: str, raw_root: str | Path) -> SymbolMetadata:
        """Fetch public symbol status and exact PRICE_FILTER/LOT_SIZE scales."""
        symbol = symbol.upper()
        content, request_uri, response_headers, observed_ts_ns = self._bounded_request_body(
            "/api/v3/exchangeInfo",
            {"symbol": symbol},
            raw_root=Path(raw_root),
            rejected_dataset="exchange_info_rejected",
            symbol=symbol,
            requested_start_ns=None,
            requested_end_ns=None,
        )
        raw_page = _write_raw_response(
            content,
            raw_root=Path(raw_root),
            dataset="exchange_info",
            symbol=symbol,
            request_uri=request_uri,
            downloaded_at_utc=_iso_from_ns(observed_ts_ns),
            requested_start_ns=None,
            requested_end_ns=None,
            response_headers=response_headers,
            retained_evidence_budget=self.retained_evidence_budget,
        )
        try:
            payload = cast(dict[str, Any], json.loads(content))
            symbols = cast(list[dict[str, Any]], payload["symbols"])
            if len(symbols) != 1 or str(symbols[0]["symbol"]).upper() != symbol:
                raise BinanceMetadataContractError(
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
                raise BinanceMetadataContractError(
                    "exchangeInfo returned a nonpositive tick or lot size"
                )
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
        except (BinanceMetadataContractError, BinanceResponseSizeLimitError):
            raise
        except BinancePayloadError as exc:
            if exc.transient:
                raise
            raise BinanceMetadataContractError(str(exc)) from exc
        except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
            raise BinanceMetadataContractError("malformed Binance exchangeInfo response") from exc

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
        content, request_uri, response_headers, received_ts_ns = self._bounded_request_body(
            "/api/v3/depth",
            {"symbol": symbol, "limit": limit},
            raw_root=Path(raw_root),
            rejected_dataset="depth_snapshots_rejected",
            symbol=symbol,
            requested_start_ns=None,
            requested_end_ns=None,
        )
        raw_page = _write_raw_response(
            content,
            raw_root=Path(raw_root),
            dataset="depth_snapshots",
            symbol=symbol,
            request_uri=request_uri,
            downloaded_at_utc=_iso_from_ns(received_ts_ns),
            requested_start_ns=None,
            requested_end_ns=None,
            response_headers=response_headers,
            retained_evidence_budget=self.retained_evidence_budget,
        )
        try:
            payload = cast(dict[str, Any], json.loads(content))
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


@dataclass(frozen=True, slots=True)
class _ParsedAggregateTrade:
    aggregate_id: int
    first_trade_id: int
    last_trade_id: int
    event_ts_ns: int
    price: Decimal
    quantity: Decimal
    buyer_is_maker: bool


def _response_header(headers: Mapping[str, str], name: str) -> str | None:
    normalized_name = name.lower()
    for key, value in headers.items():
        if str(key).lower() == normalized_name:
            return str(value)
    return None


@dataclass(frozen=True, slots=True)
class _BoundedResponseBody:
    content: bytes
    error_message: str | None
    observed_bytes_lower_bound: int
    retryable: bool = False


def _rejected_response_headers(
    response_headers: Mapping[str, str],
    body: _BoundedResponseBody,
) -> dict[str, str]:
    """Describe a bounded rejected response in its immutable raw sidecar."""
    result = dict(response_headers)
    result.update(
        {
            "x-local-capture-status": (
                "rejected_partial_response" if body.content else "rejected_headers_only"
            ),
            "x-local-captured-bytes": str(len(body.content)),
            "x-local-observed-bytes-lower-bound": str(body.observed_bytes_lower_bound),
            "x-local-rejection-reason": body.error_message or "unknown",
        }
    )
    return result


def _read_bounded_response_body(
    response: requests.Response,
    *,
    max_response_bytes: int,
) -> _BoundedResponseBody:
    """Read one response with a hard prefix bound on the production transport.

    A real ``requests.Response`` exposes ``iter_content`` and is consumed a
    chunk at a time.  Injected legacy test responses without that method fall
    back to their already-materialized ``content`` attribute for compatibility.
    """
    headers = response.headers
    declared = _response_header(headers, "content-length")
    try:
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                return _BoundedResponseBody(
                    content=b"",
                    error_message="aggregate-trade response has invalid Content-Length",
                    observed_bytes_lower_bound=0,
                )
            if declared_size < 0:
                return _BoundedResponseBody(
                    content=b"",
                    error_message="aggregate-trade response has invalid Content-Length",
                    observed_bytes_lower_bound=0,
                )
            if declared_size > max_response_bytes:
                return _BoundedResponseBody(
                    content=b"",
                    error_message=(
                        "aggregate-trade response Content-Length exceeded the response-byte ceiling"
                    ),
                    observed_bytes_lower_bound=0,
                )

        iter_content = getattr(response, "iter_content", None)
        if callable(iter_content):
            chunks: list[bytes] = []
            captured_bytes = 0
            chunk_size = min(64 * 1024, max_response_bytes + 1)
            iterator = cast(
                Callable[..., Iterator[object]],
                iter_content,
            )
            try:
                for raw_chunk in iterator(chunk_size=chunk_size):
                    if not raw_chunk:
                        continue
                    if not isinstance(raw_chunk, bytes):
                        return _BoundedResponseBody(
                            content=b"".join(chunks),
                            error_message=(
                                "aggregate-trade response yielded a non-bytes body chunk"
                            ),
                            observed_bytes_lower_bound=captured_bytes,
                        )
                    remaining = max_response_bytes - captured_bytes
                    if len(raw_chunk) > remaining:
                        if remaining:
                            chunks.append(raw_chunk[:remaining])
                        return _BoundedResponseBody(
                            content=b"".join(chunks),
                            error_message=(
                                "aggregate-trade response body exceeded the response-byte ceiling"
                            ),
                            observed_bytes_lower_bound=captured_bytes + len(raw_chunk),
                        )
                    chunks.append(raw_chunk)
                    captured_bytes += len(raw_chunk)
            except requests.RequestException as exc:
                return _BoundedResponseBody(
                    content=b"".join(chunks),
                    error_message=(
                        "aggregate-trade response body was interrupted after "
                        f"{captured_bytes} bytes: {type(exc).__name__}"
                    ),
                    observed_bytes_lower_bound=captured_bytes,
                    retryable=True,
                )
            return _BoundedResponseBody(
                content=b"".join(chunks),
                error_message=None,
                observed_bytes_lower_bound=captured_bytes,
            )

        # Compatibility path for minimal injected responses.  This is not used
        # by requests.Response and therefore is not the production transport.
        content = bytes(response.content)
        if len(content) > max_response_bytes:
            return _BoundedResponseBody(
                content=content[:max_response_bytes],
                error_message=("aggregate-trade response body exceeded the response-byte ceiling"),
                observed_bytes_lower_bound=len(content),
            )
        return _BoundedResponseBody(
            content=content,
            error_message=None,
            observed_bytes_lower_bound=len(content),
        )
    finally:
        _close_response(response)


def _parse_aggregate_trade_page(
    content: bytes,
    *,
    request_limit: int,
) -> list[_ParsedAggregateTrade]:
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinancePayloadError("malformed Binance aggregate-trade page") from exc
    if not isinstance(decoded, list):
        raise BinancePayloadError("malformed Binance aggregate-trade page")
    if len(decoded) > request_limit:
        raise BinancePayloadError("aggregate-trade response exceeded the requested page-size bound")

    parsed: list[_ParsedAggregateTrade] = []
    previous_id: int | None = None
    previous_ts_ns: int | None = None
    for raw_item in decoded:
        if not isinstance(raw_item, dict):
            raise BinancePayloadError("malformed aggregate-trade record")
        item = cast(dict[str, Any], raw_item)
        try:
            aggregate_id = int(item["a"])
            first_trade_id = int(item.get("f", aggregate_id))
            last_trade_id = int(item.get("l", aggregate_id))
            event_ts_ns = _event_timestamp_ns(int(item["T"]), "ms")
            price = Decimal(str(item["p"]))
            quantity = Decimal(str(item["q"]))
            buyer_is_maker = item["m"]
        except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
            raise BinancePayloadError("malformed aggregate-trade record") from exc
        if not isinstance(buyer_is_maker, bool):
            raise BinancePayloadError("malformed aggregate-trade record")
        if aggregate_id < 0 or first_trade_id < 0 or last_trade_id < first_trade_id:
            raise BinancePayloadError("invalid aggregate-trade identifiers")
        if previous_id is not None and aggregate_id <= previous_id:
            raise BinancePayloadError("aggregate-trade page IDs are not strictly increasing")
        if previous_ts_ns is not None and event_ts_ns < previous_ts_ns:
            raise BinancePayloadError("aggregate-trade page event times are not ordered")
        parsed.append(
            _ParsedAggregateTrade(
                aggregate_id=aggregate_id,
                first_trade_id=first_trade_id,
                last_trade_id=last_trade_id,
                event_ts_ns=event_ts_ns,
                price=price,
                quantity=quantity,
                buyer_is_maker=buyer_is_maker,
            )
        )
        previous_id = aggregate_id
        previous_ts_ns = event_ts_ns
    return parsed


class BinanceTradeBatchStream(Iterator[pa.RecordBatch]):
    """Lazy, page-bounded iterator over normalized aggregate trades.

    No HTTP request is made until the first call to :func:`next`.  Each
    nonempty batch comes from one retained raw response and has no more rows
    than the downloader's request limit.  The iterator retains only terminal
    counters and the latest raw-page descriptor; callers that need every raw
    descriptor can process them incrementally with ``on_raw_page``.  Production
    HTTP responses are consumed incrementally and stop after the first chunk
    crossing ``max_response_bytes``.  Minimal injected responses that do not
    implement ``iter_content`` retain a compatibility-only ``content`` fallback.
    """

    def __init__(
        self,
        *,
        client: BinancePublicClient,
        raw_root: Path,
        request_limit: int,
        max_response_bytes: int,
        tick_size: Decimal | None,
        lot_size: Decimal | None,
        symbol: str,
        start_ts_ns: int,
        end_ts_ns: int,
        max_events: int,
        on_raw_page: Callable[[RawPage], None] | None,
    ) -> None:
        if end_ts_ns <= start_ts_ns:
            raise ValueError("end_ts_ns must be after start_ts_ns")
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._client = client
        self._raw_root = raw_root
        self._request_limit = request_limit
        self._max_response_bytes = max_response_bytes
        self._tick_size = tick_size
        self._lot_size = lot_size
        self._symbol = symbol.upper()
        self._start_ts_ns = start_ts_ns
        self._end_ts_ns = end_ts_ns
        self._max_events = max_events
        self._on_raw_page = on_raw_page
        self._params: dict[str, str | int] = {
            "symbol": self._symbol,
            "startTime": start_ts_ns // _NS_PER_MILLISECOND,
            "endTime": (end_ts_ns - 1) // _NS_PER_MILLISECOND,
            "limit": request_limit,
        }
        self._rows_yielded = 0
        self._raw_page_count = 0
        self._last_raw_page: RawPage | None = None
        self._previous_last_id: int | None = None
        self._previous_last_event_ts_ns: int | None = None
        self._summary: BinanceTradeStreamSummary | None = None
        self._failed = False

    def __iter__(self) -> BinanceTradeBatchStream:
        return self

    @property
    def rows_yielded(self) -> int:
        return self._rows_yielded

    @property
    def raw_page_count(self) -> int:
        return self._raw_page_count

    @property
    def last_raw_page(self) -> RawPage | None:
        return self._last_raw_page

    @property
    def summary(self) -> BinanceTradeStreamSummary:
        """Return terminal metadata, failing closed before normal exhaustion."""
        if self._summary is None:
            raise RuntimeError("trade stream summary is unavailable before normal exhaustion")
        return self._summary

    def __next__(self) -> pa.RecordBatch:
        if self._summary is not None:
            raise StopIteration
        if self._failed:
            raise RuntimeError("trade stream cannot resume after a previous failure")
        try:
            return self._next_batch()
        except StopIteration:
            raise
        except Exception:
            self._failed = True
            raise

    def _resolve_scales(self) -> tuple[Decimal, Decimal]:
        if self._tick_size is None or self._lot_size is None:
            metadata = self._client.fetch_exchange_info(
                symbol=self._symbol,
                raw_root=self._raw_root,
            )
            self._tick_size = metadata.tick_size
            self._lot_size = metadata.lot_size
        return self._tick_size, self._lot_size

    def _record_page(self, raw_page: RawPage) -> None:
        self._raw_page_count += 1
        self._last_raw_page = raw_page
        if self._on_raw_page is not None:
            self._on_raw_page(raw_page)

    def _finish(self, reason: BinanceTradeStreamStopReason) -> None:
        self._summary = BinanceTradeStreamSummary(
            requested_start_ns=self._start_ts_ns,
            requested_end_ns=self._end_ts_ns,
            rows_yielded=self._rows_yielded,
            raw_page_count=self._raw_page_count,
            stop_reason=reason,
            complete_range=(
                reason is not BinanceTradeStreamStopReason.EVENT_CAP and self._rows_yielded > 0
            ),
            last_raw_page=self._last_raw_page,
        )

    def _normalize_records(
        self,
        items: list[_ParsedAggregateTrade],
        *,
        source_artifact_id: str,
        tick_size: Decimal,
        lot_size: Decimal,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for item in items:
            price_ticks = _scaled_integer(item.price, tick_size, "trade price")
            quantity_lots = _scaled_integer(item.quantity, lot_size, "trade quantity")
            price = float(item.price)
            quantity = float(item.quantity)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "venue": "binance_spot",
                    "symbol": self._symbol,
                    "event_ts_ns": item.event_ts_ns,
                    "received_ts_ns": None,
                    "available_ts_ns": item.event_ts_ns,
                    "availability_basis": "exchange_event_time_proxy",
                    "capture_seq": None,
                    "continuity_id": None,
                    "trade_id": item.aggregate_id,
                    "first_trade_id": item.first_trade_id,
                    "last_trade_id": item.last_trade_id,
                    "price_ticks": price_ticks,
                    "quantity_lots": quantity_lots,
                    "tick_size": float(tick_size),
                    "lot_size": float(lot_size),
                    "price": price,
                    "quantity": quantity,
                    "quote_quantity": price * quantity,
                    "aggressor_side": "sell" if item.buyer_is_maker else "buy",
                    "buyer_is_maker": item.buyer_is_maker,
                    "source_artifact_id": source_artifact_id,
                }
            )
        return records

    def _next_batch(self) -> pa.RecordBatch:
        tick_size, lot_size = self._resolve_scales()
        while True:
            content, request_uri, response_headers, downloaded_ns = (
                self._client._bounded_request_body(
                    "/api/v3/aggTrades",
                    self._params,
                    raw_root=self._raw_root,
                    rejected_dataset="agg_trades_rejected",
                    symbol=self._symbol,
                    requested_start_ns=self._start_ts_ns,
                    requested_end_ns=self._end_ts_ns,
                    max_response_bytes=self._max_response_bytes,
                )
            )
            raw_page_base = _write_raw_response(
                content,
                raw_root=self._raw_root,
                dataset="agg_trades",
                symbol=self._symbol,
                request_uri=request_uri,
                downloaded_at_utc=_iso_from_ns(downloaded_ns),
                requested_start_ns=self._start_ts_ns,
                requested_end_ns=self._end_ts_ns,
                response_headers=response_headers,
                retained_evidence_budget=self._client.retained_evidence_budget,
            )
            parsed = _parse_aggregate_trade_page(content, request_limit=self._request_limit)
            raw_page = RawPage(
                path=raw_page_base.path,
                manifest_path=raw_page_base.manifest_path,
                sha256=raw_page_base.sha256,
                request_uri=raw_page_base.request_uri,
                row_count=len(parsed),
            )
            self._record_page(raw_page)
            if not parsed:
                self._finish(BinanceTradeStreamStopReason.EMPTY_PAGE)
                raise StopIteration

            first = parsed[0]
            last = parsed[-1]
            if self._previous_last_id is not None and first.aggregate_id <= self._previous_last_id:
                raise BinancePayloadError("aggregate-trade pagination did not advance")
            if (
                self._previous_last_event_ts_ns is not None
                and first.event_ts_ns < self._previous_last_event_ts_ns
            ):
                raise BinancePayloadError("aggregate-trade pages are not time ordered")
            self._previous_last_id = last.aggregate_id
            self._previous_last_event_ts_ns = last.event_ts_ns

            in_range = [
                item for item in parsed if self._start_ts_ns <= item.event_ts_ns < self._end_ts_ns
            ]
            remaining = self._max_events - self._rows_yielded
            selected = in_range[:remaining]
            records = self._normalize_records(
                selected,
                source_artifact_id=raw_page.sha256,
                tick_size=tick_size,
                lot_size=lot_size,
            )
            self._rows_yielded += len(records)

            terminal_reason: BinanceTradeStreamStopReason | None = None
            if len(in_range) >= remaining:
                terminal_reason = BinanceTradeStreamStopReason.EVENT_CAP
            elif last.event_ts_ns >= self._end_ts_ns:
                terminal_reason = BinanceTradeStreamStopReason.RANGE_END
            elif len(parsed) < self._request_limit:
                terminal_reason = BinanceTradeStreamStopReason.SHORT_PAGE
            else:
                self._params = {
                    "symbol": self._symbol,
                    "fromId": last.aggregate_id + 1,
                    "limit": self._request_limit,
                }

            if terminal_reason is not None:
                self._finish(terminal_reason)
            if records:
                table = table_from_records("trades", records)
                batches = table.to_batches(max_chunksize=self._request_limit)
                if len(batches) != 1:
                    raise BinancePayloadError("failed to construct one bounded trade batch")
                return batches[0]
            if terminal_reason is not None:
                raise StopIteration


class BinanceHistoricalTradeDownloader:
    """Historical aggregate-trade downloader with lazy and guarded materialized APIs."""

    def __init__(
        self,
        *,
        client: BinancePublicClient,
        raw_root: str | Path,
        request_limit: int = 1000,
        max_response_bytes: int = 8 * 1024 * 1024,
        materialization_max_rows: int = 100_000,
        tick_size: Decimal | None = None,
        lot_size: Decimal | None = None,
    ) -> None:
        if not 1 <= request_limit <= 1000:
            raise ValueError("request_limit must be in [1, 1000]")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if materialization_max_rows < 1:
            raise ValueError("materialization_max_rows must be positive")
        self.client = client
        self.raw_root = Path(raw_root)
        self.request_limit = request_limit
        self.max_response_bytes = max_response_bytes
        self.materialization_max_rows = materialization_max_rows
        if (tick_size is None) != (lot_size is None):
            raise ValueError("tick_size and lot_size must be supplied together")
        self.tick_size = tick_size
        self.lot_size = lot_size

    def stream(
        self,
        *,
        symbol: str,
        start_ts_ns: int,
        end_ts_ns: int,
        max_events: int,
        on_raw_page: Callable[[RawPage], None] | None = None,
    ) -> BinanceTradeBatchStream:
        """Create a lazy ``[start, end)`` page stream without making an HTTP call."""
        return BinanceTradeBatchStream(
            client=self.client,
            raw_root=self.raw_root,
            request_limit=self.request_limit,
            max_response_bytes=self.max_response_bytes,
            tick_size=self.tick_size,
            lot_size=self.lot_size,
            symbol=symbol,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            max_events=max_events,
            on_raw_page=on_raw_page,
        )

    def download(
        self,
        *,
        symbol: str,
        start_ts_ns: int,
        end_ts_ns: int,
        max_events: int,
    ) -> BinanceDownloadResult:
        """Materialize the lazy stream for backwards-compatible small downloads."""
        if max_events > self.materialization_max_rows:
            raise ValueError(
                "max_events exceeds the guarded download() materialization limit; "
                "consume stream() incrementally for larger histories"
            )
        raw_pages: list[RawPage] = []
        stream = self.stream(
            symbol=symbol,
            start_ts_ns=start_ts_ns,
            end_ts_ns=end_ts_ns,
            max_events=max_events,
            on_raw_page=raw_pages.append,
        )
        batches = list(stream)
        trades = (
            pa.Table.from_batches(batches, schema=get_schema("trades"))
            if batches
            else table_from_records("trades", [])
        )
        return BinanceDownloadResult(
            trades=trades,
            raw_pages=tuple(raw_pages),
            requested_start_ns=start_ts_ns,
            requested_end_ns=end_ts_ns,
            complete_range=stream.summary.complete_range,
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
    retained_evidence_budget: RetainedEvidenceBudget | None = None,
) -> RawPage:
    checksum = hashlib.sha256(content).hexdigest()
    directory = raw_root / "binance_spot" / dataset / symbol.upper()
    destination = directory / f"{checksum}.json"
    if retained_evidence_budget is not None:
        retained_evidence_budget.assert_contains(destination)
    transaction = (
        retained_evidence_budget.write_transaction()
        if retained_evidence_budget is not None
        else nullcontext()
    )
    with transaction:
        directory.mkdir(parents=True, exist_ok=True)
        body_reservation = None
        created_body = False
        if destination.exists():
            if sha256_file(destination) != checksum:
                raise BinancePayloadError(f"raw content-address collision at {destination}")
        else:
            if retained_evidence_budget is not None:
                body_reservation = retained_evidence_budget.reserve(
                    len(content),
                    label=f"raw Binance response {destination.name}",
                )
            try:
                handle, temporary_name = tempfile.mkstemp(
                    dir=directory,
                    prefix=".raw-",
                    suffix=".tmp",
                )
            except BaseException:
                if body_reservation is not None and body_reservation.active:
                    body_reservation.release()
                raise
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, destination)
                _fsync_directory(directory)
                created_body = True
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                if body_reservation is not None and body_reservation.active:
                    body_reservation.release()
                raise
        try:
            manifest_path, _ = write_source_manifest(
                destination,
                source="binance_spot_public_api",
                source_uri=request_uri,
                downloaded_at_utc=downloaded_at_utc,
                requested_start_ns=requested_start_ns,
                requested_end_ns=requested_end_ns,
                response_headers=response_headers,
                retained_evidence_budget=retained_evidence_budget,
            )
            if body_reservation is not None:
                body_reservation.commit()
        except BaseException:
            if created_body:
                destination.unlink(missing_ok=True)
            if body_reservation is not None and body_reservation.active:
                body_reservation.release()
            raise
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
        on_raw_frame: RawDepthFrameCallback | None = None,
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
        self._on_raw_frame = on_raw_frame

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
                        if isinstance(raw, bytes):
                            raw_bytes = raw
                            was_text = False
                        else:
                            raw_bytes = raw.encode("utf-8")
                            was_text = True
                        if self._on_raw_frame is not None:
                            self._on_raw_frame(
                                RawDepthFrame(
                                    payload=raw_bytes,
                                    was_text=was_text,
                                    received_ts_ns=received_ts_ns,
                                    capture_seq=capture_seq,
                                    continuity_id=continuity_id,
                                )
                            )
                        raw_text = raw_bytes.decode("utf-8") if isinstance(raw, bytes) else raw
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
