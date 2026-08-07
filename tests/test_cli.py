from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from microstructure import cli as cli_module
from microstructure.cli import main
from microstructure.data.binance import CapturedDepth, RawDepthFrame
from microstructure.data.book import BookSnapshot, DepthDelta
from microstructure.data.storage import write_source_manifest
from microstructure.provenance import read_json, sha256_file, utc_now_iso

PROJECT_ROOT = Path(__file__).parents[1]


def _captured_depth(
    *,
    sequence: int,
    continuity_id: str,
    event_ts_ns: int,
) -> CapturedDepth:
    raw_payload = json.dumps(
        {"continuity_id": continuity_id, "sequence": sequence},
        separators=(",", ":"),
    )
    return CapturedDepth(
        raw_payload=raw_payload,
        delta=DepthDelta(
            venue="binance_spot",
            symbol="BTCUSDT",
            event_ts_ns=event_ts_ns,
            received_ts_ns=event_ts_ns + 100,
            available_ts_ns=event_ts_ns + 100,
            availability_basis="local_receive_time",
            capture_seq=sequence,
            continuity_id=continuity_id,
            first_update_id=sequence,
            last_update_id=sequence,
            previous_update_id=sequence - 1,
            bids=((10_000, 10 + sequence),),
            asks=(),
            tick_size=0.01,
            lot_size=0.001,
            source_artifact_id=hashlib.sha256(raw_payload.encode()).hexdigest(),
        ),
    )


def _preserved_snapshot(
    *,
    output_root: Path,
    continuity_id: str,
    last_update_id: int,
    received_ts_ns: int,
) -> BookSnapshot:
    payload = json.dumps(
        {
            "asks": [["100.02", "0.010"]],
            "bids": [["100.00", "0.010"]],
            "lastUpdateId": last_update_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    raw_path = (
        output_root / "raw" / "binance_spot" / "depth_snapshots" / "BTCUSDT" / f"{digest}.json"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)
    write_source_manifest(
        raw_path,
        source="binance_spot_public_api",
        source_uri="https://example.invalid/api/v3/depth",
        downloaded_at_utc=utc_now_iso(),
        requested_start_ns=None,
        requested_end_ns=None,
    )
    return BookSnapshot(
        venue="binance_spot",
        symbol="BTCUSDT",
        snapshot_id=digest,
        request_ts_ns=received_ts_ns - 100,
        received_ts_ns=received_ts_ns,
        available_ts_ns=received_ts_ns,
        continuity_id=continuity_id,
        last_update_id=last_update_id,
        depth_limit=100,
        bids=((10_000, 10),),
        asks=((10_002, 10),),
        tick_size=0.01,
        lot_size=0.001,
        source_artifact_id=digest,
    )


def test_cli_help_and_version_are_available(capsys: object) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0


def test_validate_command_runs_offline(capsys: object) -> None:
    exit_code = main(["validate", "--config", str(PROJECT_ROOT / "configs" / "smoke.toml")])

    assert exit_code == 0


def test_live_depth_capture_resnapshots_and_preserves_every_reconnect_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = [
        _captured_depth(
            sequence=11,
            continuity_id="epoch-1",
            event_ts_ns=2_000_000_000,
        ),
        _captured_depth(
            sequence=21,
            continuity_id="epoch-2",
            event_ts_ns=3_000_000_000,
        ),
    ]
    snapshot_calls: list[str] = []

    class FakeClient:
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SimpleNamespace:
            assert symbol == "BTCUSDT"
            assert raw_root == tmp_path / "raw"
            return SimpleNamespace(tick_size=Decimal("0.01"), lot_size=Decimal("0.001"))

        def fetch_depth_snapshot(
            self,
            *,
            symbol: str,
            raw_root: Path,
            continuity_id: str,
            tick_size: Decimal,
            lot_size: Decimal,
        ) -> BookSnapshot:
            snapshot_calls.append(continuity_id)
            last_update_id = 10 if continuity_id == "epoch-1" else 20
            received_ts_ns = 1_900_000_000 if continuity_id == "epoch-1" else 2_900_000_000
            return _preserved_snapshot(
                output_root=tmp_path,
                continuity_id=continuity_id,
                last_update_id=last_update_id,
                received_ts_ns=received_ts_ns,
            )

    class FakeCollector:
        url = "wss://example.invalid/stream"

        def __init__(self, **kwargs: object) -> None:
            assert kwargs["symbols"] == ("BTCUSDT",)

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            assert max_messages == 2
            for item in captured:
                yield item

    monkeypatch.setattr(cli_module, "BinancePublicClient", FakeClient)
    monkeypatch.setattr(cli_module, "BinanceLiveDepthCollector", FakeCollector)

    result = asyncio.run(
        cli_module._capture_depth(
            symbol="BTCUSDT",
            max_messages=2,
            output_root=tmp_path,
        )
    )

    assert result.messages == 2
    assert snapshot_calls == ["epoch-1", "epoch-2"]
    assert result.reconstruction_status == "LIVE"
    assert result.book_observations == 2
    assert result.quality_errors == 0
    summary = read_json(result.summary_path)
    assert summary["capture_status"] == "COMPLETE"
    assert summary["continuity_epochs"] == 2
    assert [item["continuity_id"] for item in summary["continuity_epoch_coverage"]] == [
        "epoch-1",
        "epoch-2",
    ]
    assert sum(item["messages"] for item in summary["continuity_epoch_coverage"]) == 2
    assert summary["normalized_messages"] == 2
    assert summary["excluded_messages"] == 0
    assert all(
        entry["rows"] == expected
        for entry, expected in (
            (summary["normalized_dataset_manifests"]["book_snapshots"], 2),
            (summary["normalized_dataset_manifests"]["depth_deltas"], 2),
            (summary["normalized_dataset_manifests"]["book_observations"], 2),
            (summary["normalized_dataset_manifests"]["sequence_gaps"], 0),
        )
    )
    with result.raw_path.open(encoding="utf-8") as handle:
        journal = [json.loads(line) for line in handle]
    assert [event["event_kind"] for event in journal] == [
        "websocket_frame",
        "rest_snapshot_anchor",
        "websocket_frame",
        "rest_snapshot_anchor",
    ]
    assert base64.b64decode(journal[0]["payload_base64"]).decode() == captured[0].raw_payload

    second_result = asyncio.run(
        cli_module._capture_depth(
            symbol="BTCUSDT",
            max_messages=2,
            output_root=tmp_path,
        )
    )
    latest_pointer = read_json(tmp_path / "quality" / "live_depth_capture.summary.json")
    assert result.summary_path.is_file()
    assert second_result.summary_path.is_file()
    assert second_result.summary_path != result.summary_path
    assert latest_pointer["capture_status"] == "LATEST_POINTER"
    assert latest_pointer["authoritative_summary_path"] == str(second_result.summary_path)


def test_live_depth_duration_completes_gracefully_before_message_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _captured_depth(
        sequence=1,
        continuity_id="epoch-duration",
        event_ts_ns=2_000_000_000,
    )

    class FakeClient:
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SimpleNamespace:
            return SimpleNamespace(tick_size=Decimal("0.01"), lot_size=Decimal("0.001"))

        def fetch_depth_snapshot(self, **kwargs: object) -> BookSnapshot:
            return _preserved_snapshot(
                output_root=tmp_path,
                continuity_id=str(kwargs["continuity_id"]),
                last_update_id=0,
                received_ts_ns=1_900_000_000,
            )

    class FakeCollector:
        url = "wss://example.invalid/stream"

        def __init__(self, **kwargs: object) -> None:
            self.callback = kwargs["on_raw_frame"]

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            assert max_messages == 10
            received_ts_ns = item.delta.received_ts_ns
            assert received_ts_ns is not None
            self.callback(
                RawDepthFrame(
                    payload=item.raw_payload.encode(),
                    was_text=True,
                    received_ts_ns=received_ts_ns,
                    capture_seq=1,
                    continuity_id="epoch-duration",
                )
            )
            yield item
            await asyncio.sleep(60)

    monkeypatch.setattr(cli_module, "BinancePublicClient", FakeClient)
    monkeypatch.setattr(cli_module, "BinanceLiveDepthCollector", FakeCollector)

    result = asyncio.run(
        cli_module._capture_depth(
            symbol="BTCUSDT",
            max_messages=10,
            duration_seconds=0.01,
            output_root=tmp_path,
        )
    )

    assert result.messages == 1
    assert result.completion_reason == "duration_elapsed"
    assert result.requested_duration_seconds == 0.01
    assert result.elapsed_monotonic_seconds >= 0.009
    summary = read_json(result.summary_path)
    assert summary["completion_reason"] == "duration_elapsed"
    assert summary["message_safety_ceiling"] == 10
    assert summary["requested_duration_seconds"] == 0.01
    assert summary["max_continuity_epoch_seconds"] == 0.0


def test_live_depth_duration_fails_if_message_safety_ceiling_arrives_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SimpleNamespace:
            return SimpleNamespace(tick_size=Decimal("0.01"), lot_size=Decimal("0.001"))

        def fetch_depth_snapshot(self, **kwargs: object) -> BookSnapshot:
            return _preserved_snapshot(
                output_root=tmp_path,
                continuity_id=str(kwargs["continuity_id"]),
                last_update_id=0,
                received_ts_ns=1_900_000_000,
            )

    class FakeCollector:
        url = "wss://example.invalid/stream"

        def __init__(self, **kwargs: object) -> None:
            self.callback = kwargs["on_raw_frame"]

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            assert max_messages == 2
            for sequence in (1, 2):
                item = _captured_depth(
                    sequence=sequence,
                    continuity_id="epoch-cap",
                    event_ts_ns=2_000_000_000 + sequence,
                )
                received_ts_ns = item.delta.received_ts_ns
                assert received_ts_ns is not None
                self.callback(
                    RawDepthFrame(
                        payload=item.raw_payload.encode(),
                        was_text=True,
                        received_ts_ns=received_ts_ns,
                        capture_seq=sequence,
                        continuity_id="epoch-cap",
                    )
                )
                yield item

    monkeypatch.setattr(cli_module, "BinancePublicClient", FakeClient)
    monkeypatch.setattr(cli_module, "BinanceLiveDepthCollector", FakeCollector)

    with pytest.raises(RuntimeError, match="message safety ceiling"):
        asyncio.run(
            cli_module._capture_depth(
                symbol="BTCUSDT",
                max_messages=2,
                duration_seconds=60.0,
                output_root=tmp_path,
            )
        )

    assert not list((tmp_path / "quality").glob("live_depth_capture.*.summary.json"))
    assert list((tmp_path / "quality").glob("live_depth_capture.*.failed.json"))


def test_live_depth_capture_is_one_pass_and_history_independent_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message_count = 2_050
    yielded = 0
    stream_calls = 0
    snapshot_calls = 0

    class FakeClient:
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SimpleNamespace:
            return SimpleNamespace(tick_size=Decimal("0.01"), lot_size=Decimal("0.001"))

        def fetch_depth_snapshot(self, **kwargs: object) -> BookSnapshot:
            nonlocal snapshot_calls
            snapshot_calls += 1
            return _preserved_snapshot(
                output_root=tmp_path,
                continuity_id=str(kwargs["continuity_id"]),
                last_update_id=0,
                received_ts_ns=1_000_000_000,
            )

    class FakeCollector:
        url = "wss://example.invalid/stream"

        def __init__(self, **kwargs: object) -> None:
            callback = kwargs["on_raw_frame"]
            assert callable(callback)
            self.callback = callback

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            nonlocal stream_calls, yielded
            stream_calls += 1
            assert max_messages == message_count
            for sequence in range(1, message_count + 1):
                item = _captured_depth(
                    sequence=sequence,
                    continuity_id="epoch-1",
                    event_ts_ns=2_000_000_000 + sequence,
                )
                received_ts_ns = item.delta.received_ts_ns
                assert received_ts_ns is not None
                self.callback(
                    RawDepthFrame(
                        payload=item.raw_payload.encode(),
                        was_text=True,
                        received_ts_ns=received_ts_ns,
                        capture_seq=sequence,
                        continuity_id="epoch-1",
                    )
                )
                yielded += 1
                yield item

    monkeypatch.setattr(cli_module, "BinancePublicClient", FakeClient)
    monkeypatch.setattr(cli_module, "BinanceLiveDepthCollector", FakeCollector)

    result = asyncio.run(
        cli_module._capture_depth(
            symbol="BTCUSDT",
            max_messages=message_count,
            output_root=tmp_path,
        )
    )

    assert stream_calls == 1
    assert yielded == message_count
    assert snapshot_calls == 1
    assert result.messages == message_count
    assert result.book_observations == message_count
    summary = read_json(result.summary_path)
    assert max(summary["max_buffered_rows_per_dataset"].values()) <= 1_024
    assert max(summary["max_buffered_estimated_bytes_per_dataset"].values()) <= 16 * 1024 * 1024
    depth_manifest_entry = summary["normalized_dataset_manifests"]["depth_deltas"]
    depth_manifest = read_json(depth_manifest_entry["manifest_path"])
    assert depth_manifest["rows"] == message_count
    assert len(depth_manifest["artifacts"]) == 1
    with result.raw_path.open(encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == message_count + 1


def test_live_depth_failure_atomically_preserves_preparse_frame_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = b"\xffnot-json\n"

    class FakeClient:
        def fetch_exchange_info(self, *, symbol: str, raw_root: Path) -> SimpleNamespace:
            return SimpleNamespace(tick_size=Decimal("0.01"), lot_size=Decimal("0.001"))

    class FakeCollector:
        url = "wss://example.invalid/stream"

        def __init__(self, **kwargs: object) -> None:
            self.callback = kwargs["on_raw_frame"]

        async def stream(self, *, max_messages: int | None = None) -> AsyncIterator[CapturedDepth]:
            self.callback(
                RawDepthFrame(
                    payload=malformed,
                    was_text=False,
                    received_ts_ns=2_000_000_100,
                    capture_seq=0,
                    continuity_id="epoch-parse-failure",
                )
            )
            raise UnicodeDecodeError("utf-8", malformed, 0, 1, "invalid start byte")
            if False:  # pragma: no cover - makes this an async generator
                yield _captured_depth(
                    sequence=1,
                    continuity_id="unreachable",
                    event_ts_ns=1,
                )

    monkeypatch.setattr(cli_module, "BinancePublicClient", FakeClient)
    monkeypatch.setattr(cli_module, "BinanceLiveDepthCollector", FakeCollector)

    with pytest.raises(UnicodeDecodeError):
        asyncio.run(
            cli_module._capture_depth(
                symbol="BTCUSDT",
                max_messages=1,
                output_root=tmp_path,
            )
        )

    assert not (tmp_path / "quality" / "live_depth_capture.summary.json").exists()
    [failed_raw] = list(
        (tmp_path / "raw" / "binance_spot" / "depth_stream" / "BTCUSDT").glob(
            "capture-failed-*.ndjson"
        )
    )
    [event] = [json.loads(line) for line in failed_raw.read_text().splitlines()]
    assert base64.b64decode(event["payload_base64"]) == malformed
    manifests = list(failed_raw.parent.glob(f"{failed_raw.name}.manifest-*.json"))
    assert manifests
    assert any(
        read_json(path)["response_headers"]["x-local-capture-status"]
        == "incomplete_capture_failure"
        for path in manifests
    )
    [failure] = list((tmp_path / "quality").glob("live_depth_capture.*.failed.json"))
    assert read_json(failure)["completion_manifest_published"] is False


def test_raw_journal_publish_recovers_if_manifest_write_fails_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool = cli_module._RawMessageSpool(
        root=tmp_path,
        symbol="BTCUSDT",
        source_uri="wss://example.invalid/stream",
    )
    spool.append_frame(
        RawDepthFrame(
            payload=b"{}",
            was_text=True,
            received_ts_ns=1,
            capture_seq=0,
            continuity_id="epoch-1",
        )
    )
    real_write_manifest = cli_module.write_source_manifest
    attempts = 0

    def flaky_manifest(*args: object, **kwargs: object) -> tuple[Path, str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected manifest failure")
        return real_write_manifest(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_module, "write_source_manifest", flaky_manifest)

    with pytest.raises(RuntimeError, match="injected manifest failure"):
        spool.publish(status="raw_capture_complete")
    renamed_path = spool.evidence_path
    assert renamed_path.is_file()

    evidence = spool.publish(status="incomplete_capture_failure")

    assert evidence.path == renamed_path
    assert evidence.sha256 == sha256_file(renamed_path)
    assert attempts == 2
