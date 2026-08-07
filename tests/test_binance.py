from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from microstructure.data.binance import (
    BinanceHistoricalTradeDownloader,
    BinancePublicClient,
    RetryPolicy,
    parse_depth_message,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload, separators=(",", ":")).encode()
        self.text = self.content.decode()
        self.headers = dict(headers or {})
        self.url = "https://data-api.binance.vision/fixture"

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: Mapping[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        response = self.responses.pop(0)
        response.url = f"{url}?{urlencode(params)}"
        return response


def _client(session: FakeSession, *, sleeps: list[float] | None = None) -> BinancePublicClient:
    recorded_sleeps = sleeps if sleeps is not None else []
    return BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=0.5),
        sleep=recorded_sleeps.append,
        random_value=lambda: 1.0,
    )


def test_public_client_honors_retry_after_on_rate_limit() -> None:
    sleeps: list[float] = []
    session = FakeSession(
        [
            FakeResponse(429, {"code": -1003}, headers={"Retry-After": "2"}),
            FakeResponse(200, []),
        ]
    )
    client = _client(session, sleeps=sleeps)

    response = client._request("/api/v3/aggTrades", {"symbol": "BTCUSDT"})

    assert response.status_code == 200
    assert sleeps == [2.0]
    assert len(session.calls) == 2


def test_exchange_info_extracts_exact_public_tick_and_lot_filters(tmp_path: Path) -> None:
    payload = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01000000",
                        "maxPrice": "1000000.00000000",
                        "tickSize": "0.01000000",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.00001000",
                        "maxQty": "9000.00000000",
                        "stepSize": "0.00001000",
                    },
                ],
            }
        ]
    }
    session = FakeSession([FakeResponse(200, payload)])

    metadata = _client(session).fetch_exchange_info(symbol="btcusdt", raw_root=tmp_path)

    assert metadata.symbol == "BTCUSDT"
    assert metadata.status == "TRADING"
    assert metadata.base_asset == "BTC"
    assert metadata.quote_asset == "USDT"
    assert metadata.tick_size == Decimal("0.01000000")
    assert metadata.lot_size == Decimal("0.00001000")
    assert (tmp_path / "binance_spot" / "exchange_info" / "BTCUSDT").is_dir()


def test_historical_trade_downloader_paginates_by_inclusive_id_and_preserves_raw(
    tmp_path: Path,
) -> None:
    page_one = [
        {"a": 1, "p": "100.01", "q": "0.002", "f": 10, "l": 10, "T": 1000, "m": False},
        {"a": 2, "p": "100.02", "q": "0.003", "f": 11, "l": 12, "T": 1000, "m": True},
    ]
    page_two = [{"a": 3, "p": "100.03", "q": "0.004", "f": 13, "l": 13, "T": 1500, "m": False}]
    session = FakeSession([FakeResponse(200, page_one), FakeResponse(200, page_two)])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    result = downloader.download(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )

    assert result.complete_range
    assert result.trades.column("trade_id").to_pylist() == [1, 2, 3]
    assert result.trades.column("aggressor_side").to_pylist() == ["buy", "sell", "buy"]
    assert result.trades.column("price_ticks").to_pylist() == [10_001, 10_002, 10_003]
    assert (
        result.trades.column("availability_basis").to_pylist() == ["exchange_event_time_proxy"] * 3
    )
    assert session.calls[1]["params"] == {"symbol": "BTCUSDT", "fromId": 3, "limit": 2}
    assert len(result.raw_pages) == 2
    assert all(page.path.is_file() and page.manifest_path.is_file() for page in result.raw_pages)


def test_parse_spot_diff_depth_uses_u_ranges_and_zero_as_delete() -> None:
    raw = json.dumps(
        {
            "stream": "btcusdt@depth@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1_700_000_000_123_456,
                "s": "BTCUSDT",
                "U": 101,
                "u": 104,
                "b": [["100.01", "0.00000"], ["100.00", "0.00200"]],
                "a": [["100.02", "0.00300"]],
            },
        }
    )

    delta = parse_depth_message(
        raw,
        received_ts_ns=1_700_000_000_200_000_000,
        capture_seq=9,
        continuity_id="session-1",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
        timestamp_unit="us",
    )

    assert delta.first_update_id == 101
    assert delta.last_update_id == 104
    assert delta.previous_update_id is None  # Spot's documented payload has no pu.
    assert delta.bids == ((10_001, 0), (10_000, 2))
    assert delta.asks == ((10_002, 3),)
    assert delta.event_ts_ns == 1_700_000_000_123_456_000
    assert delta.available_ts_ns == delta.received_ts_ns
