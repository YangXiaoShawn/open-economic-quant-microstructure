from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from pathlib import Path

import pytest

from microstructure import m8_l2_binance as adapter_module
from microstructure.data.binance import (
    BinanceHTTPError,
    CapturedDepth,
    RawDepthFrame,
    SymbolMetadata,
)
from microstructure.data.book import BookSnapshot, DepthDelta
from microstructure.data.storage import write_source_manifest
from microstructure.m8_l2_binance import BinanceM8L2Capture
from microstructure.m8_l2_capture import M8L2DataFailure
from microstructure.m8_l2_config import M8L2CaptureLimits
from microstructure.provenance import read_json, sha256_file, utc_now_iso

START_NS = 1_000_000_000
END_NS = 20_000_000_000
SESSION_ID = "a" * 64


def _limits(
    *,
    max_messages: int = 100,
    max_raw_frame_bytes: int = 1_048_576,
    max_arrow_batch_bytes: int = 16_777_216,
) -> M8L2CaptureLimits:
    return M8L2CaptureLimits(
        duration_seconds=19,
        max_messages_per_symbol=max_messages,
        max_raw_frame_bytes=max_raw_frame_bytes,
        max_arrow_batch_bytes=max_arrow_batch_bytes,
        min_overlapping_coverage_seconds=1,
        min_single_continuity_epoch_seconds=1,
        require_complete_status=True,
        require_live_reconstruction=True,
        max_sequence_gaps=0,
        max_quality_errors=0,
        max_quality_warnings=0,
    )


def _raw_artifact(
    raw_root: Path,
    *,
    dataset: str,
    symbol: str,
    payload: bytes,
) -> tuple[Path, Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    path = raw_root / "binance_spot" / dataset / symbol / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    manifest, _ = write_source_manifest(
        path,
        source="binance_spot_public_api",
        source_uri="https://example.invalid/public",
        downloaded_at_utc=utc_now_iso(),
        requested_start_ns=None,
        requested_end_ns=None,
    )
    return path, manifest, digest


def _metadata(raw_root: Path, symbol: str) -> SymbolMetadata:
    payload = json.dumps({"symbol": symbol}, sort_keys=True).encode()
    path, manifest, digest = _raw_artifact(
        raw_root, dataset="exchange_info", symbol=symbol, payload=payload
    )
    return SymbolMetadata(
        venue="binance_spot",
        symbol=symbol,
        status="TRADING",
        base_asset=symbol[:-4],
        quote_asset="USDT",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.001"),
        min_price=Decimal("0.01"),
        max_price=Decimal("1000000"),
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("1000000"),
        observed_ts_ns=START_NS,
        source_artifact_id=digest,
        source_path=path,
        source_manifest_path=manifest,
    )


def _snapshot(
    raw_root: Path,
    symbol: str,
    continuity_id: str,
    *,
    last_update_id: int = 0,
) -> BookSnapshot:
    payload = json.dumps(
        {
            "asks": [["100.02", "0.010"]],
            "bids": [["100.00", "0.010"]],
            "lastUpdateId": last_update_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _, _, digest = _raw_artifact(
        raw_root, dataset="depth_snapshots", symbol=symbol, payload=payload
    )
    return BookSnapshot(
        venue="binance_spot",
        symbol=symbol,
        snapshot_id=digest,
        request_ts_ns=START_NS,
        received_ts_ns=START_NS + 1,
        available_ts_ns=START_NS + 1,
        continuity_id=continuity_id,
        last_update_id=last_update_id,
        depth_limit=5_000,
        bids=((10_000, 10),),
        asks=((10_002, 10),),
        tick_size=0.01,
        lot_size=0.001,
        source_artifact_id=digest,
    )


def _captured(
    symbol: str,
    *,
    sequence: int,
    received_ns: int,
    continuity_id: str = "continuity-1",
    first_update_id: int | None = None,
    previous_update_id: int | None = None,
) -> CapturedDepth:
    payload = json.dumps(
        {
            "continuity": continuity_id,
            "received": received_ns,
            "sequence": sequence,
            "symbol": symbol,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return CapturedDepth(
        raw_payload=payload,
        delta=DepthDelta(
            venue="binance_spot",
            symbol=symbol,
            event_ts_ns=received_ns - 10,
            received_ts_ns=received_ns,
            available_ts_ns=received_ns,
            availability_basis="local_receive_time",
            capture_seq=sequence,
            continuity_id=continuity_id,
            first_update_id=(sequence if first_update_id is None else first_update_id),
            last_update_id=sequence,
            previous_update_id=(sequence - 1 if previous_update_id is None else previous_update_id),
            bids=((10_000, 10 + sequence),),
            asks=(),
            tick_size=0.01,
            lot_size=0.001,
            source_artifact_id=hashlib.sha256(payload.encode()).hexdigest(),
        ),
    )


class _FakeClient:
    def __init__(
        self,
        *,
        snapshot_last_update_id: int = 0,
        snapshot_hook: Callable[[], None] | None = None,
        metadata_hook: Callable[[], None] | None = None,
    ) -> None:
        self.snapshot_last_update_id = snapshot_last_update_id
        self.snapshot_hook = snapshot_hook
        self.metadata_hook = metadata_hook

    def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SymbolMetadata:
        if self.metadata_hook is not None:
            self.metadata_hook()
        return _metadata(raw_root, symbol)

    def fetch_depth_snapshot(self, **kwargs: object) -> BookSnapshot:
        if self.snapshot_hook is not None:
            self.snapshot_hook()
        return _snapshot(
            Path(kwargs["raw_root"]),
            str(kwargs["symbol"]),
            str(kwargs["continuity_id"]),
            last_update_id=self.snapshot_last_update_id,
        )


def _collector_factory(
    items: list[CapturedDepth],
    *,
    on_yield: Callable[[int], None] | None = None,
    event_clock: list[int] | None = None,
) -> Callable[..., object]:
    class FakeCollector:
        url = "wss://example.invalid/depth"

        def __init__(self, **kwargs: object) -> None:
            self.callback = kwargs["on_raw_frame"]

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            assert max_messages is None
            for index, item in enumerate(items):
                received_ns = item.delta.received_ts_ns
                capture_seq = item.delta.capture_seq
                assert received_ns is not None and capture_seq is not None
                if event_clock is not None:
                    event_clock[0] = max(event_clock[0], received_ns)
                callback = cast_callback(self.callback)
                callback(
                    RawDepthFrame(
                        payload=item.raw_payload.encode(),
                        was_text=True,
                        received_ts_ns=received_ns,
                        capture_seq=capture_seq,
                        continuity_id=item.delta.continuity_id,
                    )
                )
                if on_yield is not None:
                    on_yield(index)
                yield item
                await asyncio.sleep(0)

    return FakeCollector


def cast_callback(value: object) -> Callable[[RawDepthFrame], None]:
    assert callable(value)
    return value  # type: ignore[return-value]


def _run(
    root: Path,
    items: list[CapturedDepth],
    *,
    client: _FakeClient | None = None,
    limits: M8L2CaptureLimits | None = None,
    queue_capacity: int = 1_024,
) -> object:
    event_clock = [START_NS]
    adapter = BinanceM8L2Capture(
        client_factory=lambda: client or _FakeClient(),
        collector_factory=_collector_factory(items, event_clock=event_clock),
        clock_ns=lambda: event_clock[0],
        receiver_queue_capacity=queue_capacity,
    )
    return asyncio.run(
        adapter(
            symbol="BTCUSDT",
            scheduled_start_ns=START_NS,
            scheduled_end_ns=END_NS,
            stage_root=root,
            limits=limits or _limits(),
            session_id=SESSION_ID,
        )
    )


def test_exact_end_excludes_future_frame_and_reconciles_all_artifacts(tmp_path: Path) -> None:
    items = [
        _captured("BTCUSDT", sequence=1, received_ns=START_NS + 1_000),
        _captured("BTCUSDT", sequence=2, received_ns=START_NS + 2_000),
        _captured("BTCUSDT", sequence=3, received_ns=END_NS),
    ]

    result = _run(tmp_path, items)

    assert result.status == "COMPLETE"  # type: ignore[union-attr]
    assert result.completion_reason == "scheduled_end_reached"  # type: ignore[union-attr]
    assert result.messages == result.normalized_rows == 2  # type: ignore[union-attr]
    assert result.reconstructed_rows == 2  # type: ignore[union-attr]
    assert result.excluded_rows == 0  # type: ignore[union-attr]
    [interval] = result.valid_observed_intervals  # type: ignore[union-attr]
    assert interval.start_received_ns == START_NS + 1_000
    assert interval.end_received_ns_exclusive == START_NS + 2_001
    actual = {path.resolve() for path in tmp_path.rglob("*") if path.is_file()}
    declared = {item.path.resolve() for item in result.artifacts}  # type: ignore[union-attr]
    assert declared == actual
    assert all(sha256_file(item.path) == item.sha256 for item in result.artifacts)  # type: ignore[union-attr]
    kinds = {item.kind for item in result.artifacts}  # type: ignore[union-attr]
    assert {
        "capture_summary",
        "normalized_data",
        "normalized_manifest",
        "quality_report",
        "raw_journal",
        "raw_journal_manifest",
        "raw_snapshot",
        "raw_snapshot_manifest",
    }.issubset(kinds)
    summary = read_json(tmp_path / "quality" / "capture.summary.json")
    assert summary["first_raw_received_ns"] == START_NS + 1_000
    assert summary["last_raw_received_ns"] == START_NS + 2_000
    expected_inventory = [
        {
            "path": item.path.relative_to(tmp_path).as_posix(),
            "kind": item.kind,
            "sha256": item.sha256,
            "bytes": item.path.stat().st_size,
        }
        for item in result.artifacts  # type: ignore[union-attr]
        if item.kind != "capture_summary"
    ]
    assert summary["artifact_inventory_without_summary"] == expected_inventory
    journal_path = next(
        item.path
        for item in result.artifacts
        if item.kind == "raw_journal"  # type: ignore[union-attr]
    )
    events = [json.loads(line) for line in journal_path.read_text().splitlines()]
    raw_frames = [item for item in events if item["event_kind"] == "websocket_frame"]
    assert len(raw_frames) == 2
    assert all(START_NS <= item["received_ts_ns"] < END_NS for item in raw_frames)


@pytest.mark.parametrize(
    ("items", "snapshot_last_update_id", "expected_reconstructed", "expected_intervals"),
    [
        (
            [
                _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100),
                _captured("BTCUSDT", sequence=2, received_ns=END_NS),
            ],
            0,
            1,
            0,
        ),
        (
            [
                _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100),
                _captured("BTCUSDT", sequence=2, received_ns=START_NS + 200),
                _captured("BTCUSDT", sequence=3, received_ns=END_NS),
            ],
            10,
            0,
            0,
        ),
    ],
)
def test_one_message_and_stale_only_never_fabricate_coverage(
    tmp_path: Path,
    items: list[CapturedDepth],
    snapshot_last_update_id: int,
    expected_reconstructed: int,
    expected_intervals: int,
) -> None:
    result = _run(
        tmp_path,
        items,
        client=_FakeClient(snapshot_last_update_id=snapshot_last_update_id),
    )

    assert result.reconstructed_rows == expected_reconstructed  # type: ignore[union-attr]
    assert len(result.valid_observed_intervals) == expected_intervals  # type: ignore[union-attr]


def test_long_silence_splits_observed_intervals(tmp_path: Path) -> None:
    first = START_NS + 100
    items = [
        _captured("BTCUSDT", sequence=1, received_ns=first),
        _captured("BTCUSDT", sequence=2, received_ns=first + 1_000_000_000),
        _captured("BTCUSDT", sequence=3, received_ns=first + 7_000_000_000),
        _captured("BTCUSDT", sequence=4, received_ns=first + 8_000_000_000),
        _captured("BTCUSDT", sequence=5, received_ns=END_NS),
    ]

    result = _run(tmp_path, items)

    intervals = result.valid_observed_intervals  # type: ignore[union-attr]
    assert len(intervals) == 2
    assert intervals[0].end_received_ns_exclusive < intervals[1].start_received_ns


def test_snapshot_thread_does_not_block_bounded_receiver(tmp_path: Path) -> None:
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    yielded = 0

    def block_snapshot() -> None:
        snapshot_started.set()
        assert release_snapshot.wait(timeout=5)

    def observe_yield(_: int) -> None:
        nonlocal yielded
        yielded += 1

    items = [
        *[
            _captured("BTCUSDT", sequence=index, received_ns=START_NS + index * 1_000)
            for index in range(1, 11)
        ],
        _captured("BTCUSDT", sequence=11, received_ns=END_NS),
    ]
    event_clock = [START_NS]
    adapter = BinanceM8L2Capture(
        client_factory=lambda: _FakeClient(snapshot_hook=block_snapshot),
        collector_factory=_collector_factory(
            items, on_yield=observe_yield, event_clock=event_clock
        ),
        clock_ns=lambda: event_clock[0],
        receiver_queue_capacity=32,
    )

    async def scenario() -> object:
        task = asyncio.create_task(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )
        assert await asyncio.to_thread(snapshot_started.wait, 2)
        for _ in range(100):
            if yielded >= 10:
                break
            await asyncio.sleep(0.001)
        assert yielded >= 10
        release_snapshot.set()
        return await task

    result = asyncio.run(scenario())
    summary = read_json(tmp_path / "quality" / "capture.summary.json")
    assert result.messages == 10  # type: ignore[union-attr]
    assert summary["max_receiver_queue_depth"] > 1
    assert (
        summary["max_receiver_queue_estimated_bytes"]
        <= summary["receiver_queue_estimated_byte_budget"]
    )


def test_metadata_thread_does_not_block_event_loop(tmp_path: Path) -> None:
    metadata_started = threading.Event()
    release_metadata = threading.Event()
    heartbeat = False

    def block_metadata() -> None:
        metadata_started.set()
        assert release_metadata.wait(timeout=5)

    items = [
        _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100),
        _captured("BTCUSDT", sequence=2, received_ns=END_NS),
    ]
    event_clock = [START_NS]
    adapter = BinanceM8L2Capture(
        client_factory=lambda: _FakeClient(metadata_hook=block_metadata),
        collector_factory=_collector_factory(items, event_clock=event_clock),
        clock_ns=lambda: event_clock[0],
    )

    async def scenario() -> None:
        nonlocal heartbeat
        task = asyncio.create_task(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )
        assert await asyncio.to_thread(metadata_started.wait, 2)
        await asyncio.sleep(0)
        heartbeat = True
        release_metadata.set()
        await task

    asyncio.run(scenario())
    assert heartbeat


def test_sequence_gap_is_typed_data_failure_with_exhaustive_partial_artifacts(
    tmp_path: Path,
) -> None:
    items = [
        _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100),
        _captured(
            "BTCUSDT",
            sequence=3,
            received_ns=START_NS + 200,
            first_update_id=3,
            previous_update_id=1,
        ),
        _captured("BTCUSDT", sequence=4, received_ns=END_NS),
    ]

    with pytest.raises(M8L2DataFailure) as raised:
        _run(tmp_path, items)

    assert raised.value.reason_code == "SEQUENCE_CONTINUITY_FAILED"
    partial = raised.value.partial_result
    assert partial is not None
    assert partial.status == "FAILED"
    assert partial.sequence_gaps == 1
    actual = {path.resolve() for path in tmp_path.rglob("*") if path.is_file()}
    assert {item.path.resolve() for item in partial.artifacts} == actual


def test_message_cap_preserves_raw_and_raises_typed_failure(tmp_path: Path) -> None:
    items = [_captured("BTCUSDT", sequence=1, received_ns=START_NS + 100)]

    with pytest.raises(M8L2DataFailure) as raised:
        _run(tmp_path, items, limits=_limits(max_messages=1))

    assert raised.value.reason_code == "MESSAGE_SAFETY_CEILING_REACHED"
    partial = raised.value.partial_result
    assert partial is not None
    assert partial.messages == 1
    assert partial.normalized_rows == 0


def test_preparse_utf8_failure_retains_exact_bytes_and_is_typed(tmp_path: Path) -> None:
    malformed = b"\xffnot-json"

    class MalformedCollector:
        url = "wss://example.invalid/depth"

        def __init__(self, **kwargs: object) -> None:
            self.callback = cast_callback(kwargs["on_raw_frame"])

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            self.callback(
                RawDepthFrame(
                    payload=malformed,
                    was_text=False,
                    received_ts_ns=START_NS + 100,
                    capture_seq=0,
                    continuity_id="parse-failure",
                )
            )
            raise UnicodeDecodeError("utf-8", malformed, 0, 1, "invalid start byte")
            if False:  # pragma: no cover
                yield _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100)

    adapter = BinanceM8L2Capture(
        client_factory=lambda: _FakeClient(),
        collector_factory=MalformedCollector,
        clock_ns=lambda: START_NS,
    )

    with pytest.raises(M8L2DataFailure) as raised:
        asyncio.run(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )

    assert raised.value.reason_code == "RAW_UTF8_DECODE_FAILED"
    partial = raised.value.partial_result
    assert partial is not None
    journal = next(item.path for item in partial.artifacts if item.kind == "raw_journal")
    [event] = [json.loads(line) for line in journal.read_text().splitlines()]
    assert base64.b64decode(event["payload_base64"]) == malformed


def test_queue_byte_budget_is_hard_even_with_large_item_capacity(tmp_path: Path) -> None:
    items = [
        _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100),
        _captured("BTCUSDT", sequence=2, received_ns=START_NS + 200),
        _captured("BTCUSDT", sequence=3, received_ns=END_NS),
    ]

    result = _run(
        tmp_path,
        items,
        limits=_limits(max_arrow_batch_bytes=8_192),
        queue_capacity=10_000,
    )

    summary = read_json(tmp_path / "quality" / "capture.summary.json")
    assert summary["receiver_queue_estimated_byte_budget"] == 8_192
    assert summary["max_receiver_queue_estimated_bytes"] <= 8_192
    assert result.max_arrow_batch_bytes_observed <= 8_192  # type: ignore[union-attr]


def test_permission_error_remains_nonterminal_system_failure(tmp_path: Path) -> None:
    class PermissionClient(_FakeClient):
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SymbolMetadata:
            raise PermissionError("injected local permission failure")

    adapter = BinanceM8L2Capture(
        client_factory=PermissionClient,
        collector_factory=_collector_factory([]),
        clock_ns=lambda: START_NS,
    )

    with pytest.raises(PermissionError, match="injected local permission failure"):
        asyncio.run(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )

    assert not (tmp_path / "INSUFFICIENT_DATA").exists()
    assert (tmp_path / "quality" / "capture.summary.json").is_file()


def test_exhausted_websocket_transport_is_typed_session_data_failure(tmp_path: Path) -> None:
    item = _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100)

    class DisconnectCollector:
        url = "wss://example.invalid/depth"

        def __init__(self, **kwargs: object) -> None:
            self.callback = cast_callback(kwargs["on_raw_frame"])

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            received_ns = item.delta.received_ts_ns
            capture_seq = item.delta.capture_seq
            assert received_ns is not None and capture_seq is not None
            self.callback(
                RawDepthFrame(
                    payload=item.raw_payload.encode(),
                    was_text=True,
                    received_ts_ns=received_ns,
                    capture_seq=capture_seq,
                    continuity_id=item.delta.continuity_id,
                )
            )
            yield item
            raise OSError("websocket retry budget exhausted")

    adapter = BinanceM8L2Capture(
        client_factory=lambda: _FakeClient(),
        collector_factory=DisconnectCollector,
        clock_ns=lambda: START_NS,
    )

    with pytest.raises(M8L2DataFailure) as raised:
        asyncio.run(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )

    assert raised.value.reason_code == "PUBLIC_STREAM_UNAVAILABLE"
    assert raised.value.partial_result is not None
    assert raised.value.partial_result.messages == 1


def test_exhausted_public_rest_transport_is_typed_session_data_failure(
    tmp_path: Path,
) -> None:
    class UnavailableClient(_FakeClient):
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SymbolMetadata:
            raise BinanceHTTPError("public REST retry budget exhausted", retry_exhausted=True)

    adapter = BinanceM8L2Capture(
        client_factory=UnavailableClient,
        collector_factory=_collector_factory([]),
        clock_ns=lambda: START_NS,
    )

    with pytest.raises(M8L2DataFailure) as raised:
        asyncio.run(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )

    assert raised.value.reason_code == "PUBLIC_TRANSPORT_UNAVAILABLE"
    assert raised.value.phase == "PUBLIC_METADATA"


def test_local_raw_journal_oserror_remains_system_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _captured("BTCUSDT", sequence=1, received_ns=START_NS + 100)

    def fail_local_write(*args: object, **kwargs: object) -> None:
        raise OSError("local disk write failed")

    monkeypatch.setattr(adapter_module._RawJournal, "_write", fail_local_write)
    adapter = BinanceM8L2Capture(
        client_factory=lambda: _FakeClient(),
        collector_factory=_collector_factory([item]),
        clock_ns=lambda: START_NS,
    )

    with pytest.raises(RuntimeError, match="local raw-journal write failed") as raised:
        asyncio.run(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )

    assert not isinstance(raised.value, M8L2DataFailure)


def test_cancellation_joins_rest_worker_before_returning(tmp_path: Path) -> None:
    metadata_started = threading.Event()
    release_metadata = threading.Event()

    def block_metadata() -> None:
        metadata_started.set()
        assert release_metadata.wait(timeout=5)

    adapter = BinanceM8L2Capture(
        client_factory=lambda: _FakeClient(metadata_hook=block_metadata),
        collector_factory=_collector_factory([]),
        clock_ns=lambda: START_NS,
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            adapter(
                symbol="BTCUSDT",
                scheduled_start_ns=START_NS,
                scheduled_end_ns=END_NS,
                stage_root=tmp_path,
                limits=_limits(),
                session_id=SESSION_ID,
            )
        )
        assert await asyncio.to_thread(metadata_started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release_metadata.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    before = sorted(
        (path.relative_to(tmp_path).as_posix(), sha256_file(path))
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert before
    # No detached REST worker remains to mutate the retained evidence afterward.
    time.sleep(0.01)
    after = sorted(
        (path.relative_to(tmp_path).as_posix(), sha256_file(path))
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert after == before
