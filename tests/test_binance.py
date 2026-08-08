from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import requests

from microstructure.data.binance import (
    BinanceHistoricalTradeDownloader,
    BinanceHTTPError,
    BinanceLiveDepthCollector,
    BinancePayloadError,
    BinancePublicClient,
    BinanceTradeStreamStopReason,
    RawDepthFrame,
    RawPage,
    RetryPolicy,
    _write_raw_response,
    parse_depth_message,
)
from microstructure.data.evidence_budget import EvidenceBudgetExceeded, RetainedEvidenceBudget


class _FakeWebSocket:
    def __init__(self, frames: list[str | bytes]) -> None:
        self.frames = frames

    async def __aenter__(self) -> _FakeWebSocket:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        async def iterate() -> AsyncIterator[str | bytes]:
            for frame in self.frames:
                yield frame

        return iterate()


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


class ChunkedResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        headers: Mapping[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.url = "https://data-api.binance.vision/fixture"
        self.chunks = chunks
        self.chunks_read = 0
        self.requested_chunk_sizes: list[int] = []
        self.closed = False

    @property
    def content(self) -> bytes:
        raise AssertionError("streaming trade path must not access response.content")

    @property
    def text(self) -> str:
        raise AssertionError("streaming trade path must not access response.text")

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        self.requested_chunk_sizes.append(chunk_size)
        for chunk in self.chunks:
            self.chunks_read += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


class StreamingSession:
    def __init__(self, responses: list[ChunkedResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
        stream: bool = False,
    ) -> ChunkedResponse:
        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "timeout": timeout,
                "stream": stream,
            }
        )
        response = self.responses.pop(0)
        response.url = f"{url}?{urlencode(params)}"
        return response


class InterruptedChunkedResponse(ChunkedResponse):
    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        self.requested_chunk_sizes.append(chunk_size)
        self.chunks_read += 1
        yield self.chunks[0]
        raise requests.ConnectionError("fixture stream interrupted")


def _write_budget_fixture_raw(
    root: Path,
    *,
    budget: RetainedEvidenceBudget | None = None,
) -> RawPage:
    return _write_raw_response(
        b'{"fixture":true}',
        raw_root=root,
        dataset="budget_fixture",
        symbol="BTCUSDT",
        request_uri="https://data-api.binance.vision/fixture",
        downloaded_at_utc="2026-08-07T00:00:00.000000000Z",
        requested_start_ns=1,
        requested_end_ns=2,
        response_headers={"content-type": "application/json"},
        retained_evidence_budget=budget,
    )


def _client(session: FakeSession, *, sleeps: list[float] | None = None) -> BinancePublicClient:
    recorded_sleeps = sleeps if sleeps is not None else []
    return BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=0.5),
        sleep=recorded_sleeps.append,
        random_value=lambda: 1.0,
    )


def test_final_retryable_streaming_response_is_closed() -> None:
    response = ChunkedResponse([], status_code=503)
    session = StreamingSession([response])
    client = BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=0),
    )

    with pytest.raises(BinanceHTTPError, match="public Binance GET failed"):
        client._request("/api/v3/aggTrades", {"symbol": "BTCUSDT"}, stream_response=True)

    assert response.closed


def test_interrupted_stream_preserves_bounded_prefix_before_failure(tmp_path: Path) -> None:
    prefix = b'[{"a":1'
    response = InterruptedChunkedResponse([prefix])
    session = StreamingSession([response])
    downloader = BinanceHistoricalTradeDownloader(
        client=BinancePublicClient(
            session=session,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_retries=0),
        ),
        raw_root=tmp_path,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=0,
        end_ts_ns=2_000_000_000,
        max_events=1,
    )
    with pytest.raises(BinancePayloadError, match="interrupted after 7 bytes"):
        next(stream)

    rejected = list((tmp_path / "binance_spot" / "agg_trades_rejected").rglob("*.json"))
    raw_paths = [path for path in rejected if ".manifest-" not in path.name]
    assert len(raw_paths) == 1
    assert raw_paths[0].read_bytes() == prefix
    assert response.closed


def test_interrupted_stream_retries_after_preserving_failed_attempt(tmp_path: Path) -> None:
    prefix = b'[{"a":1'
    successful = json.dumps([_trade(1)], separators=(",", ":")).encode()
    interrupted = InterruptedChunkedResponse([prefix])
    completed = ChunkedResponse([successful])
    session = StreamingSession([interrupted, completed])
    sleeps: list[float] = []
    downloader = BinanceHistoricalTradeDownloader(
        client=BinancePublicClient(
            session=session,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.25),
            sleep=sleeps.append,
            random_value=lambda: 1.0,
        ),
        raw_root=tmp_path,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=1,
    )

    batch = next(stream)

    assert batch.num_rows == 1
    assert len(session.calls) == 2
    assert sleeps == [0.25]
    assert interrupted.closed and completed.closed
    rejected = _raw_trade_payloads(tmp_path, dataset="agg_trades_rejected")
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == prefix


def test_retained_budget_charges_retry_prefix_and_success_sidecars(tmp_path: Path) -> None:
    prefix = b'[{"a":1'
    successful = json.dumps([_trade(1)], separators=(",", ":")).encode()
    interrupted = InterruptedChunkedResponse([prefix])
    completed = ChunkedResponse([successful])
    session = StreamingSession([interrupted, completed])
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=100_000)
    downloader = BinanceHistoricalTradeDownloader(
        client=BinancePublicClient(
            session=session,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.0),
            sleep=lambda _: None,
            random_value=lambda: 0.0,
            retained_evidence_budget=budget,
        ),
        raw_root=tmp_path,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    batch = next(
        downloader.stream(
            symbol="BTCUSDT",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            max_events=1,
        )
    )

    retained_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert batch.num_rows == 1
    assert any(path.read_bytes() == prefix for path in retained_files)
    assert any(path.read_bytes() == successful for path in retained_files)
    assert len([path for path in retained_files if ".manifest-" in path.name]) == 2
    assert budget.used_bytes == sum(path.stat().st_size for path in retained_files)
    assert budget.reserved_bytes == 0


def test_raw_body_and_sidecar_honor_exact_combined_budget_and_deduplicate(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    reference_page = _write_budget_fixture_raw(reference)
    exact_bytes = reference_page.path.stat().st_size + reference_page.manifest_path.stat().st_size

    root = tmp_path / "bounded"
    budget = RetainedEvidenceBudget(root, limit_bytes=exact_bytes)
    first = _write_budget_fixture_raw(root, budget=budget)
    assert budget.used_bytes == exact_bytes
    assert budget.remaining_bytes == 0

    duplicate = _write_budget_fixture_raw(root, budget=budget)
    assert duplicate.path == first.path
    assert duplicate.manifest_path == first.manifest_path
    assert budget.used_bytes == exact_bytes
    assert budget.reserved_bytes == 0


def test_raw_sidecar_overage_rolls_back_new_body_and_reservation(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    reference_page = _write_budget_fixture_raw(reference)
    exact_bytes = reference_page.path.stat().st_size + reference_page.manifest_path.stat().st_size

    root = tmp_path / "bounded"
    budget = RetainedEvidenceBudget(root, limit_bytes=exact_bytes - 1)

    with pytest.raises(EvidenceBudgetExceeded, match="raw source manifest"):
        _write_budget_fixture_raw(root, budget=budget)

    assert [path for path in root.rglob("*") if path.is_file()] == []
    assert budget.used_bytes == 0
    assert budget.reserved_bytes == 0
    assert budget.remaining_bytes == exact_bytes - 1


def test_raw_temp_creation_failure_releases_body_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=10_000)

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError("fixture temp failure")

    monkeypatch.setattr("microstructure.data.binance.tempfile.mkstemp", fail_mkstemp)

    with pytest.raises(OSError, match="fixture temp failure"):
        _write_budget_fixture_raw(tmp_path, budget=budget)

    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []
    assert budget.used_bytes == 0
    assert budget.reserved_bytes == 0
    assert budget.remaining_bytes == 10_000


def test_exchange_info_response_is_transport_bounded_and_preserved(tmp_path: Path) -> None:
    content = json.dumps(_exchange_info_payload(), separators=(",", ":")).encode()
    response = ChunkedResponse([content])
    session = StreamingSession([response])
    client = BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=0),
        max_response_bytes=len(content) - 1,
    )

    with pytest.raises(BinancePayloadError, match="response body exceeded"):
        client.fetch_exchange_info(symbol="BTCUSDT", raw_root=tmp_path)

    assert session.calls[0]["stream"] is True
    assert response.closed
    rejected = list((tmp_path / "binance_spot" / "exchange_info_rejected").rglob("*.json"))
    raw_paths = [path for path in rejected if ".manifest-" not in path.name]
    assert len(raw_paths) == 1
    assert len(raw_paths[0].read_bytes()) == len(content) - 1


def test_depth_snapshot_response_is_transport_bounded_and_preserved(tmp_path: Path) -> None:
    content = json.dumps(
        {"lastUpdateId": 10, "bids": [["100.00", "0.002"]], "asks": []},
        separators=(",", ":"),
    ).encode()
    response = ChunkedResponse([content])
    session = StreamingSession([response])
    client = BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=0),
        max_response_bytes=len(content) - 1,
    )

    with pytest.raises(BinancePayloadError, match="response body exceeded"):
        client.fetch_depth_snapshot(
            symbol="BTCUSDT",
            raw_root=tmp_path,
            continuity_id="epoch-1",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("0.001"),
            limit=5,
        )

    assert session.calls[0]["stream"] is True
    assert response.closed
    rejected = list((tmp_path / "binance_spot" / "depth_snapshots_rejected").rglob("*.json"))
    raw_paths = [path for path in rejected if ".manifest-" not in path.name]
    assert len(raw_paths) == 1
    assert len(raw_paths[0].read_bytes()) == len(content) - 1


def _trade(
    aggregate_id: int,
    *,
    timestamp_ms: int = 1000,
    buyer_is_maker: bool = False,
) -> dict[str, object]:
    return {
        "a": aggregate_id,
        "p": "100.01",
        "q": "0.002",
        "f": aggregate_id * 10,
        "l": aggregate_id * 10,
        "T": timestamp_ms,
        "m": buyer_is_maker,
    }


def _exchange_info_payload(symbol: str = "BTCUSDT") -> dict[str, object]:
    return {
        "symbols": [
            {
                "symbol": symbol,
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


def _raw_trade_payloads(root: Path, *, dataset: str = "agg_trades") -> list[Path]:
    directory = root / "binance_spot" / dataset / "BTCUSDT"
    return sorted(path for path in directory.glob("*.json") if ".manifest-" not in path.name)


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
    session = FakeSession([FakeResponse(200, _exchange_info_payload())])

    metadata = _client(session).fetch_exchange_info(symbol="btcusdt", raw_root=tmp_path)

    assert metadata.symbol == "BTCUSDT"
    assert metadata.status == "TRADING"
    assert metadata.base_asset == "BTC"
    assert metadata.quote_asset == "USDT"
    assert metadata.tick_size == Decimal("0.01000000")
    assert metadata.lot_size == Decimal("0.00001000")
    assert (tmp_path / "binance_spot" / "exchange_info" / "BTCUSDT").is_dir()


def test_trade_stream_is_lazy_even_when_symbol_metadata_must_be_fetched(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        [
            FakeResponse(200, _exchange_info_payload()),
            FakeResponse(200, [_trade(1)]),
        ]
    )
    pages: list[RawPage] = []
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
    )

    stream = downloader.stream(
        symbol="btcusdt",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
        on_raw_page=pages.append,
    )

    assert session.calls == []
    with pytest.raises(RuntimeError, match="before normal exhaustion"):
        _ = stream.summary

    batch = next(stream)

    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 1
    assert batch.column("trade_id").to_pylist() == [1]
    assert len(session.calls) == 2
    assert len(pages) == 1
    assert stream.last_raw_page == pages[0]
    assert stream.summary.rows_yielded == 1
    assert stream.summary.raw_page_count == 1
    assert stream.summary.stop_reason is BinanceTradeStreamStopReason.SHORT_PAGE
    assert stream.summary.complete_range
    with pytest.raises(StopIteration):
        next(stream)


def test_trade_stream_uses_chunked_transport_without_accessing_response_content(
    tmp_path: Path,
) -> None:
    encoded = json.dumps([_trade(1)], separators=(",", ":")).encode()
    response = ChunkedResponse(
        [encoded[:11], encoded[11:]],
        headers={"Content-Length": str(len(encoded))},
    )
    session = StreamingSession([response])
    client = BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=0),
    )
    downloader = BinanceHistoricalTradeDownloader(
        client=client,
        raw_root=tmp_path,
        request_limit=2,
        max_response_bytes=len(encoded),
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )
    batch = next(stream)

    assert batch.column("trade_id").to_pylist() == [1]
    assert session.calls[0]["stream"] is True
    assert response.chunks_read == 2
    assert response.closed
    assert response.requested_chunk_sizes == [len(encoded) + 1]


def test_trade_stream_truncates_exactly_at_event_cap_before_yield(tmp_path: Path) -> None:
    response = FakeResponse(200, [_trade(1), _trade(2), _trade(3)])
    session = FakeSession([response])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=3,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=2,
    )
    batch = next(stream)

    assert batch.num_rows == 2
    assert batch.num_rows <= downloader.request_limit
    assert batch.column("trade_id").to_pylist() == [1, 2]
    assert stream.last_raw_page is not None
    assert stream.last_raw_page.row_count == 3
    assert stream.summary.rows_yielded == 2
    assert stream.summary.stop_reason is BinanceTradeStreamStopReason.EVENT_CAP
    assert not stream.summary.complete_range
    assert len(session.calls) == 1


def test_materialized_download_rejects_large_request_before_http(tmp_path: Path) -> None:
    session = FakeSession([])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        materialization_max_rows=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    with pytest.raises(ValueError, match=r"consume stream\(\) incrementally"):
        downloader.download(
            symbol="BTCUSDT",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            max_events=3,
        )

    assert session.calls == []


def test_trade_stream_empty_page_has_explicit_incomplete_summary(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(200, [])])
    pages: list[RawPage] = []
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
        on_raw_page=pages.append,
    )

    assert list(stream) == []
    assert stream.summary.rows_yielded == 0
    assert stream.summary.raw_page_count == 1
    assert stream.summary.stop_reason is BinanceTradeStreamStopReason.EMPTY_PAGE
    assert not stream.summary.complete_range
    assert len(pages) == 1
    assert pages[0].row_count == 0


def test_trade_stream_stops_at_exclusive_range_end(tmp_path: Path) -> None:
    session = FakeSession(
        [FakeResponse(200, [_trade(1, timestamp_ms=1000), _trade(2, timestamp_ms=2000)])]
    )
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )
    batches = list(stream)

    assert len(batches) == 1
    assert batches[0].column("trade_id").to_pylist() == [1]
    assert stream.summary.stop_reason is BinanceTradeStreamStopReason.RANGE_END
    assert stream.summary.complete_range


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


def test_historical_trade_downloader_rejects_oversized_page_after_preserving_raw(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "a": index,
            "p": "100.01",
            "q": "0.002",
            "f": index,
            "l": index,
            "T": 1000,
            "m": False,
        }
        for index in range(3)
    ]
    session = FakeSession([FakeResponse(200, payload)])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    with pytest.raises(BinancePayloadError, match="page-size bound"):
        downloader.download(
            symbol="BTCUSDT",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            max_events=10,
        )

    raw_pages = _raw_trade_payloads(tmp_path)
    assert raw_pages


def test_trade_stream_enforces_actual_response_byte_ceiling_and_preserves_raw(
    tmp_path: Path,
) -> None:
    response = FakeResponse(200, [_trade(1)])
    byte_ceiling = len(response.content) - 1
    session = FakeSession([response])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        max_response_bytes=byte_ceiling,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )
    with pytest.raises(BinancePayloadError, match="body exceeded"):
        next(stream)

    raw_pages = _raw_trade_payloads(tmp_path, dataset="agg_trades_rejected")
    assert len(raw_pages) == 1
    assert raw_pages[0].read_bytes() == response.content[:byte_ceiling]


def test_chunked_oversized_body_stops_after_first_crossing_chunk(tmp_path: Path) -> None:
    response = ChunkedResponse([b"abcd", b"efgh", b"must-not-be-read"])
    session = StreamingSession([response])
    client = BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=0),
    )
    downloader = BinanceHistoricalTradeDownloader(
        client=client,
        raw_root=tmp_path,
        request_limit=2,
        max_response_bytes=5,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )
    with pytest.raises(BinancePayloadError, match="body exceeded"):
        next(stream)

    assert response.chunks_read == 2
    assert response.requested_chunk_sizes == [6]
    assert response.closed
    assert session.calls[0]["stream"] is True
    rejected = _raw_trade_payloads(tmp_path, dataset="agg_trades_rejected")
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == b"abcde"
    sidecars = list(rejected[0].parent.glob(f"{rejected[0].name}.manifest-*.json"))
    assert len(sidecars) == 1
    headers = json.loads(sidecars[0].read_text())["response_headers"]
    assert headers["x-local-captured-bytes"] == "5"
    assert headers["x-local-observed-bytes-lower-bound"] == "8"


def test_trade_stream_enforces_declared_response_byte_ceiling_and_preserves_raw(
    tmp_path: Path,
) -> None:
    response = ChunkedResponse([b"must-not-be-read"], headers={"Content-Length": "100000"})
    session = StreamingSession([response])
    client = BinancePublicClient(
        session=session,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_retries=0),
    )
    downloader = BinanceHistoricalTradeDownloader(
        client=client,
        raw_root=tmp_path,
        request_limit=2,
        max_response_bytes=4096,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    with pytest.raises(BinancePayloadError, match="Content-Length exceeded"):
        downloader.download(
            symbol="BTCUSDT",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            max_events=10,
        )

    assert response.chunks_read == 0
    assert response.closed
    assert session.calls[0]["stream"] is True
    raw_pages = _raw_trade_payloads(tmp_path, dataset="agg_trades_rejected")
    assert len(raw_pages) == 1
    assert raw_pages[0].read_bytes() == b""
    sidecars = list(raw_pages[0].parent.glob(f"{raw_pages[0].name}.manifest-*.json"))
    assert len(sidecars) == 1
    sidecar = sidecars[0]
    assert json.loads(sidecar.read_text())["response_headers"]["Content-Length"] == "100000"


def test_trade_stream_rejects_unordered_page_after_preserving_raw(tmp_path: Path) -> None:
    response = FakeResponse(200, [_trade(2), _trade(1)])
    session = FakeSession([response])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )
    with pytest.raises(BinancePayloadError, match="not strictly increasing"):
        next(stream)

    raw_pages = _raw_trade_payloads(tmp_path)
    assert len(raw_pages) == 1
    assert raw_pages[0].read_bytes() == response.content


def test_trade_stream_rejects_nonprogressing_next_page(tmp_path: Path) -> None:
    first_response = FakeResponse(200, [_trade(1), _trade(2)])
    repeated_response = FakeResponse(200, [_trade(2), _trade(3)])
    session = FakeSession([first_response, repeated_response])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )
    stream = downloader.stream(
        symbol="BTCUSDT",
        start_ts_ns=1_000_000_000,
        end_ts_ns=2_000_000_000,
        max_events=10,
    )

    assert next(stream).column("trade_id").to_pylist() == [1, 2]
    assert session.calls[1:] == []
    with pytest.raises(BinancePayloadError, match="did not advance"):
        next(stream)

    assert session.calls[1]["params"] == {"symbol": "BTCUSDT", "fromId": 3, "limit": 2}


def test_trade_stream_preserves_malformed_raw_page_before_parsing_error(tmp_path: Path) -> None:
    response = FakeResponse(200, {"not": "a trade list"})
    session = FakeSession([response])
    downloader = BinanceHistoricalTradeDownloader(
        client=_client(session),
        raw_root=tmp_path,
        request_limit=2,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
    )

    with pytest.raises(BinancePayloadError, match="malformed Binance aggregate-trade page"):
        downloader.download(
            symbol="BTCUSDT",
            start_ts_ns=1_000_000_000,
            end_ts_ns=2_000_000_000,
            max_events=10,
        )

    raw_pages = _raw_trade_payloads(tmp_path)
    assert len(raw_pages) == 1
    assert raw_pages[0].read_bytes() == response.content


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


def test_live_collector_reports_exact_raw_frame_before_utf8_parse_failure() -> None:
    malformed = b"\xffnot-json"
    observed: list[RawDepthFrame] = []
    connection = _FakeWebSocket([malformed])
    collector = BinanceLiveDepthCollector(
        symbols=("BTCUSDT",),
        max_reconnects=0,
        connect_factory=lambda url: connection,
        on_raw_frame=observed.append,
    )

    async def receive_one() -> object:
        return await anext(collector.stream(max_messages=1))

    with pytest.raises(UnicodeDecodeError):
        asyncio.run(receive_one())

    assert len(observed) == 1
    assert observed[0].payload == malformed
    assert observed[0].was_text is False
    assert observed[0].capture_seq == 0
    assert observed[0].continuity_id.startswith("binance-live-")
