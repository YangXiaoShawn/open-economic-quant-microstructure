from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import microstructure.m8_l2_capture as l2_module
from microstructure.data.schemas import SCHEMA_VERSION, get_schema
from microstructure.m8_l2_capture import (
    CapturedArtifact,
    CaptureSourceIdentity,
    M8L2CaptureSystemError,
    M8L2DataFailure,
    M8L2SessionBundle,
    M8L2VerificationError,
    ObservedInterval,
    SymbolCaptureResult,
    capture_m8_l2_session,
    merge_observed_intervals,
    overlapping_observed_coverage_ns,
    verify_m8_l2_session_bundle,
)
from microstructure.m8_l2_config import M8L2Session, M8L2StudyConfig, load_m8_l2_config
from microstructure.provenance import read_json, sha256_file, write_json

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_l2_capture_study.toml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "M8_L2_PROTOCOL.md"
SOURCE = CaptureSourceIdentity(commit="1" * 40, source_tree_sha256="2" * 64, dirty=False)


class FakeClock:
    def __init__(
        self,
        now_ns: int,
        *,
        finish_ns: int | None = None,
        finish_on_call: int = 3,
    ) -> None:
        self.now_ns = now_ns
        self.finish_ns = finish_ns
        self.finish_on_call = finish_on_call
        self.time_calls = 0
        self.sleeps: list[float] = []

    def time_ns(self) -> int:
        self.time_calls += 1
        if self.finish_ns is not None and self.time_calls >= self.finish_on_call:
            self.now_ns = max(self.now_ns, self.finish_ns)
        return self.now_ns

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now_ns += round(seconds * 1_000_000_000)
        await asyncio.sleep(0)


def _artifact(root: Path, name: str = "raw.ndjson") -> CapturedArtifact:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"raw":true}\n', encoding="utf-8")
    return CapturedArtifact(path=path, kind="raw_journal", sha256=sha256_file(path))


def _parquet(path: Path, dataset: str, rows: int) -> None:
    schema = get_schema(dataset)

    def value_for(field: pa.Field) -> object:
        if field.name == "schema_version":
            return SCHEMA_VERSION
        if pa.types.is_string(field.type):
            return "fixture"
        if pa.types.is_integer(field.type):
            return 1
        if pa.types.is_floating(field.type):
            return 1.0
        if pa.types.is_boolean(field.type):
            return True
        if pa.types.is_list(field.type):
            return []
        raise AssertionError(f"unsupported fixture type: {field.type}")

    table = pa.Table.from_arrays(
        [pa.array([value_for(field)] * rows, type=field.type) for field in schema],
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _complete_artifacts(
    root: Path,
    *,
    symbol: str,
    start_ns: int,
    end_ns: int,
    intervals: tuple[ObservedInterval, ...],
    reconstructed_rows: int,
    excluded_rows: int,
) -> tuple[CapturedArtifact, ...]:
    continuity_id = f"{symbol}-epoch-1"
    snapshot_path = root / "raw" / "snapshot.json"
    snapshot_manifest_path = root / "raw" / "snapshot.manifest.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(snapshot_path, {"lastUpdateId": 1})
    write_json(snapshot_manifest_path, {"checksum": sha256_file(snapshot_path)})

    raw_path = root / "raw" / "capture.ndjson"
    first_payload = b"x" * 1024
    last_payload = b"y"
    journal_entries = [
        {
            "continuity_id": continuity_id,
            "event_kind": "rest_snapshot_anchor",
            "raw_manifest_path": snapshot_manifest_path.relative_to(root).as_posix(),
            "raw_manifest_sha256": sha256_file(snapshot_manifest_path),
            "raw_path": snapshot_path.relative_to(root).as_posix(),
            "raw_sha256": sha256_file(snapshot_path),
            "received_ts_ns": start_ns + 1,
            "snapshot_id": f"{symbol}-snapshot",
        },
        {
            "event_kind": "websocket_frame",
            "payload_base64": base64.b64encode(first_payload).decode("ascii"),
            "payload_bytes": len(first_payload),
            "payload_sha256": hashlib.sha256(first_payload).hexdigest(),
            "received_ts_ns": start_ns + 10_000_000_000,
        },
        {
            "event_kind": "websocket_frame",
            "payload_base64": base64.b64encode(last_payload).decode("ascii"),
            "payload_bytes": len(last_payload),
            "payload_sha256": hashlib.sha256(last_payload).hexdigest(),
            "received_ts_ns": end_ns - 10_000_000_001,
        },
    ]
    raw_path.write_bytes(
        b"".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for item in journal_entries
        )
    )
    raw_manifest_path = root / "raw" / "capture.manifest.json"
    write_json(
        raw_manifest_path,
        {
            "artifact_kind": "raw_source",
            "bytes": raw_path.stat().st_size,
            "checksum": {"algorithm": "sha256", "value": sha256_file(raw_path)},
            "path": raw_path.name,
            "requested_range_ns": {"start": start_ns, "end_exclusive": end_ns},
            "response_headers": {
                "x-local-message-count": "2",
                "x-local-snapshot-anchor-count": "1",
            },
        },
    )

    dataset_rows = {
        "book_snapshots": 1,
        "depth_deltas": 2,
        "book_observations": reconstructed_rows,
        "sequence_gaps": 0,
    }
    dataset_summaries: dict[str, dict[str, object]] = {}
    artifact_coordinates: list[tuple[Path, str]] = [
        (raw_path, "raw_journal"),
        (raw_manifest_path, "raw_journal_manifest"),
        (snapshot_path, "raw_snapshot"),
        (snapshot_manifest_path, "raw_snapshot_manifest"),
    ]
    for dataset, rows in dataset_rows.items():
        data_path = root / "normalized" / f"{dataset}.parquet"
        if rows:
            _parquet(data_path, dataset, rows)
            artifact_coordinates.append((data_path, "normalized_data"))
        manifest_path = root / "normalized" / f"{dataset}.manifest.json"
        write_json(
            manifest_path,
            {
                "dataset": dataset,
                "requested_range_ns": {"start": start_ns, "end_exclusive": end_ns},
                "rows": rows,
                "schema_version": SCHEMA_VERSION,
            },
        )
        artifact_coordinates.append((manifest_path, "normalized_manifest"))
        dataset_summaries[dataset] = {
            "data_path": data_path.relative_to(root).as_posix() if rows else None,
            "data_sha256": sha256_file(data_path) if rows else None,
            "manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "rows": rows,
        }

    quality_paths: dict[str, str] = {}
    for dataset in ("depth_deltas", "book_observations"):
        path = root / "quality" / f"{dataset}.validation.json"
        write_json(
            path,
            {
                "dataset": dataset,
                "rows_checked": dataset_rows[dataset],
                "summary": {"errors": 0, "warnings": 0},
            },
        )
        artifact_coordinates.append((path, "quality_report"))
        quality_paths[dataset] = path.relative_to(root).as_posix()

    without_summary = tuple(
        CapturedArtifact(path=path, kind=kind, sha256=sha256_file(path))
        for path, kind in artifact_coordinates
    )
    summary_path = root / "quality" / "capture.summary.json"
    write_json(
        summary_path,
        {
            "schema_version": "m8-binance-l2-symbol-capture-v1",
            "symbol": symbol,
            "capture_id": f"{symbol.lower()}-capture",
            "capture_status": "COMPLETE",
            "completion_reason": "scheduled_end_reached",
            "reconstruction_status": "LIVE",
            "failure_reason_code": None,
            "failure_phase": None,
            "scheduled_range_ns": {"start": start_ns, "end_exclusive": end_ns},
            "messages": 2,
            "normalized_rows": 2,
            "reconstructed_rows": reconstructed_rows,
            "excluded_rows": excluded_rows,
            "continuity_epochs": 1,
            "snapshot_anchors": 1,
            "sequence_gaps": 0,
            "quality_errors": 0,
            "quality_warnings": 0,
            "max_raw_frame_bytes_observed": 1024,
            "max_arrow_batch_bytes_observed": 8192,
            "first_raw_received_ns": start_ns + 10_000_000_000,
            "last_raw_received_ns": end_ns - 10_000_000_001,
            "valid_observed_intervals": [item.to_dict() for item in intervals],
            "raw_journal": raw_path.relative_to(root).as_posix(),
            "raw_journal_sha256": sha256_file(raw_path),
            "raw_journal_manifest": raw_manifest_path.relative_to(root).as_posix(),
            "raw_journal_manifest_sha256": sha256_file(raw_manifest_path),
            "normalized_dataset_manifests": dataset_summaries,
            "quality_reports": quality_paths,
            "artifact_inventory_without_summary": [
                {
                    "path": item.path.relative_to(root).as_posix(),
                    "kind": item.kind,
                    "sha256": item.sha256,
                    "bytes": item.path.stat().st_size,
                }
                for item in sorted(without_summary, key=lambda value: value.path.as_posix())
            ],
        },
    )
    return (
        *without_summary,
        CapturedArtifact(
            path=summary_path,
            kind="capture_summary",
            sha256=sha256_file(summary_path),
        ),
    )


def _complete_result(
    *,
    symbol: str,
    root: Path,
    start_ns: int,
    end_ns: int,
    intervals: tuple[ObservedInterval, ...] | None = None,
) -> SymbolCaptureResult:
    valid = (
        (
            ObservedInterval(
                continuity_id=f"{symbol}-epoch-1",
                start_received_ns=start_ns + 10_000_000_000,
                end_received_ns_exclusive=end_ns - 10_000_000_000,
            ),
        )
        if intervals is None
        else intervals
    )
    reconstructed_rows = 2 if valid else 0
    excluded_rows = 2 - reconstructed_rows
    return SymbolCaptureResult(
        symbol=symbol,
        capture_id=f"{symbol.lower()}-capture",
        status="COMPLETE",
        completion_reason="scheduled_end_reached",
        reconstruction_status="LIVE",
        messages=2,
        normalized_rows=2,
        reconstructed_rows=reconstructed_rows,
        excluded_rows=excluded_rows,
        continuity_epochs=1,
        snapshot_anchors=1,
        sequence_gaps=0,
        quality_errors=0,
        quality_warnings=0,
        max_raw_frame_bytes_observed=1024,
        max_arrow_batch_bytes_observed=8192,
        first_raw_received_ns=start_ns + 10_000_000_000,
        last_raw_received_ns=end_ns - 10_000_000_001,
        valid_observed_intervals=valid,
        artifacts=_complete_artifacts(
            root,
            symbol=symbol,
            start_ns=start_ns,
            end_ns=end_ns,
            intervals=valid,
            reconstructed_rows=reconstructed_rows,
            excluded_rows=excluded_rows,
        ),
    )


def _manifest(bundle_root: Path) -> dict[str, object]:
    value = read_json(bundle_root / "session_manifest.json")
    assert isinstance(value, dict)
    return value


def _refresh_manifest_checksum(bundle_root: Path) -> None:
    checksum_path = bundle_root / "CHECKSUMS.sha256"
    entries: list[str] = []
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        relative = line[66:]
        digest = (
            sha256_file(bundle_root / "session_manifest.json")
            if relative == "session_manifest.json"
            else line[:64]
        )
        entries.append(f"{digest}  {relative}\n")
    checksum_path.write_text("".join(entries), encoding="ascii")


def _complete_bundle(tmp_path: Path) -> tuple[M8L2SessionBundle, M8L2StudyConfig, M8L2Session]:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        return _complete_result(
            symbol=str(kwargs["symbol"]),
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    bundle = asyncio.run(
        capture_m8_l2_session(
            config,
            session.date.isoformat(),
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
            _test_source_identity=SOURCE,
        )
    )
    return bundle, config, session


def test_observed_interval_union_and_cross_symbol_intersection_do_not_bridge_gaps() -> None:
    left = (
        ObservedInterval("left-a", 0, 100),
        ObservedInterval("left-a", 80, 120),
        ObservedInterval("left-b", 200, 400),
    )
    right = (
        ObservedInterval("right-a", 50, 250),
        ObservedInterval("right-b", 300, 500),
    )

    assert merge_observed_intervals(left) == (
        ObservedInterval("left-a", 0, 120),
        ObservedInterval("left-b", 200, 400),
    )
    assert overlapping_observed_coverage_ns(left, right) == 220


def test_dual_capture_uses_one_absolute_barrier_and_publishes_verified_complete_bundle(
    tmp_path: Path,
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    clock = FakeClock(
        session.start_ns - 1_000_000_000,
        finish_ns=session.end_ns,
        finish_on_call=4,
    )
    entered: set[str] = set()
    both_entered = asyncio.Event()
    requests: dict[str, tuple[int, int, str]] = {}

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        symbol = str(kwargs["symbol"])
        entered.add(symbol)
        requests[symbol] = (
            int(kwargs["scheduled_start_ns"]),
            int(kwargs["scheduled_end_ns"]),
            str(kwargs["session_id"]),
        )
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        return _complete_result(
            symbol=symbol,
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    bundle = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=clock,
            _test_source_identity=SOURCE,
        )
    )

    assert bundle.status == "COMPLETE"
    assert bundle.marker_path.name == "_SUCCESS"
    assert bundle.marker_path.read_bytes() == b"complete\n"
    assert set(entered) == {"BTCUSDT", "ETHUSDT"}
    assert clock.sleeps == [1.0]
    assert {value[:2] for value in requests.values()} == {(session.start_ns, session.end_ns)}
    assert len({value[2] for value in requests.values()}) == 1
    manifest = _manifest(bundle.root)
    assert manifest["status"] == "COMPLETE"
    assert manifest["reason_codes"] == []
    assert manifest["cross_symbol_observed_overlap_seconds"] == pytest.approx(3580.0)
    symbols = manifest["symbols"]
    assert isinstance(symbols, dict)
    btc = symbols["BTCUSDT"]
    assert btc["first_raw_received_ns"] == session.start_ns + 10_000_000_000
    assert btc["last_raw_received_ns"] == session.end_ns - 10_000_000_001
    assert btc["valid_observed_intervals"][0]["continuity_id"] == "BTCUSDT-epoch-1"
    assert verify_m8_l2_session_bundle(bundle.root, expected_config=config) == bundle


def test_typed_failure_in_one_symbol_does_not_cancel_the_other_symbol(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    completed: set[str] = set()

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        symbol = str(kwargs["symbol"])
        root = Path(str(kwargs["stage_root"]))
        if symbol == "BTCUSDT":
            _artifact(root, "partial.ndjson")
            await asyncio.sleep(0)
            raise M8L2DataFailure(
                "NETWORK_DISCONNECT",
                phase="DUAL_CAPTURE",
                message="declared feed disconnected",
            )
        await asyncio.sleep(0.01)
        completed.add(symbol)
        return _complete_result(
            symbol=symbol,
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    bundle = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
            _test_source_identity=SOURCE,
        )
    )

    assert completed == {"ETHUSDT"}
    assert bundle.status == "INSUFFICIENT_DATA"
    assert bundle.marker_path.name == "INSUFFICIENT_DATA"
    assert bundle.marker_path.read_bytes() == b"terminal\n"
    assert "NETWORK_DISCONNECT" in bundle.reason_codes
    assert (bundle.root / "symbols" / "BTCUSDT" / "partial.ndjson").is_file()
    assert not (bundle.root / "_SUCCESS").exists()


def test_only_observed_intervals_count_toward_epoch_and_overlap_gates(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        symbol = str(kwargs["symbol"])
        root = Path(str(kwargs["stage_root"]))
        return _complete_result(
            symbol=symbol,
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
            intervals=(),
        )

    bundle = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
            _test_source_identity=SOURCE,
        )
    )

    assert bundle.status == "INSUFFICIENT_DATA"
    assert "GATE_VALID_CONTINUITY_EPOCH_BTCUSDT" in bundle.reason_codes
    assert "GATE_VALID_CONTINUITY_EPOCH_ETHUSDT" in bundle.reason_codes
    assert "GATE_CROSS_SYMBOL_OBSERVED_OVERLAP" in bundle.reason_codes


def test_missed_absolute_window_is_terminal_insufficient_and_never_calls_capture(
    tmp_path: Path,
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    called = False

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    bundle = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.end_ns + 1),
            _test_source_identity=SOURCE,
        )
    )

    assert not called
    assert bundle.status == "INSUFFICIENT_DATA"
    assert bundle.reason_codes == ("MISSED_WINDOW",)
    assert bundle.marker_path.read_bytes() == b"terminal\n"


@pytest.mark.parametrize("error", [PermissionError("disk denied"), RuntimeError("bug")])
def test_system_fault_is_nonterminal_and_preserves_raw_evidence(
    tmp_path: Path, error: BaseException
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    other_finished = False

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        nonlocal other_finished
        symbol = str(kwargs["symbol"])
        root = Path(str(kwargs["stage_root"]))
        _artifact(root, "received-before-system-fault.ndjson")
        if symbol == "BTCUSDT":
            await asyncio.sleep(0)
            raise error
        await asyncio.sleep(0.01)
        other_finished = True
        return _complete_result(
            symbol=symbol,
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    with pytest.raises(M8L2CaptureSystemError) as raised:
        asyncio.run(
            capture_m8_l2_session(
                config,
                "2026-08-08",
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )

    assert other_finished
    evidence_root = raised.value.evidence_root
    assert evidence_root is not None and evidence_root.is_dir()
    assert (evidence_root / "SYSTEM_FAILURE.json").is_file()
    assert list(evidence_root.rglob("received-before-system-fault.ndjson"))
    assert not list(evidence_root.rglob("_SUCCESS"))
    assert not list(evidence_root.rglob("INSUFFICIENT_DATA"))
    record = read_json(evidence_root / "SYSTEM_FAILURE.json")
    assert record["terminal"] is False
    assert record["research_result"] is False
    assert not list((tmp_path / "sessions").glob("*"))


def test_active_cancellation_terminalizes_as_insufficient_and_preserves_partial_raw(
    tmp_path: Path,
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    entered: set[str] = set()
    both_entered = asyncio.Event()

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        symbol = str(kwargs["symbol"])
        root = Path(str(kwargs["stage_root"]))
        _artifact(root, "partial.ndjson")
        entered.add(symbol)
        if len(entered) == 2:
            both_entered.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def scenario() -> object:
        task = asyncio.create_task(
            capture_m8_l2_session(
                config,
                "2026-08-08",
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        task.cancel()
        return await task

    bundle = asyncio.run(scenario())

    assert bundle.status == "INSUFFICIENT_DATA"
    assert "CAPTURE_CANCELED" in bundle.reason_codes
    assert len(list(bundle.root.rglob("partial.ndjson"))) == 2


def test_verifier_rejects_tamper_extra_files_and_bad_marker_and_reuse_is_verify_only(
    tmp_path: Path,
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    calls = 0

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        nonlocal calls
        calls += 1
        return _complete_result(
            symbol=str(kwargs["symbol"]),
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    first = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
            _test_source_identity=SOURCE,
        )
    )
    reused = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.end_ns + 1),
            _test_source_identity=SOURCE,
        )
    )
    assert reused == first
    assert calls == 2

    extra = first.root / "unmanifested.txt"
    extra.write_text("tamper", encoding="utf-8")
    with pytest.raises(M8L2VerificationError, match="physical inventory"):
        verify_m8_l2_session_bundle(first.root)
    extra.unlink()

    first.marker_path.write_bytes(b"COMPLETE\n")
    with pytest.raises(M8L2VerificationError, match="marker bytes"):
        verify_m8_l2_session_bundle(first.root)


def test_capture_result_with_undeclared_file_is_program_fault_not_data_failure(
    tmp_path: Path,
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        root = Path(str(kwargs["stage_root"]))
        result = _complete_result(
            symbol=str(kwargs["symbol"]),
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )
        (root / "undeclared.bin").write_bytes(b"raw")
        return result

    with pytest.raises(M8L2CaptureSystemError) as raised:
        asyncio.run(
            capture_m8_l2_session(
                config,
                "2026-08-08",
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )
    assert raised.value.evidence_root is not None
    assert list(raised.value.evidence_root.rglob("undeclared.bin"))


def test_terminal_publication_io_fault_is_demoted_and_raw_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        return _complete_result(
            symbol=str(kwargs["symbol"]),
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    real_write = l2_module._write_bytes_durable

    def fail_terminal_marker(path: Path, payload: bytes) -> None:
        if path.name == "_SUCCESS":
            raise PermissionError("injected terminal permission failure")
        real_write(path, payload)

    monkeypatch.setattr(l2_module, "_write_bytes_durable", fail_terminal_marker)

    with pytest.raises(M8L2CaptureSystemError) as raised:
        asyncio.run(
            capture_m8_l2_session(
                config,
                "2026-08-08",
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )

    evidence_root = raised.value.evidence_root
    assert evidence_root is not None
    assert (evidence_root / "SYSTEM_FAILURE.json").is_file()
    assert len(list(evidence_root.rglob("capture.ndjson"))) == 2
    assert not (evidence_root / "_SUCCESS").exists()
    assert not (evidence_root / "INSUFFICIENT_DATA").exists()


def test_protocol_drift_during_capture_is_system_failure_and_preserves_raw(
    tmp_path: Path,
) -> None:
    project = tmp_path / "relocated-project"
    config_path = project / "configs" / "m8_l2_capture_study.toml"
    protocol_path = project / "docs" / "M8_L2_PROTOCOL.md"
    config_path.parent.mkdir(parents=True)
    protocol_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG_PATH.read_bytes())
    protocol_path.write_bytes(PROTOCOL_PATH.read_bytes())
    config = load_m8_l2_config(config_path)
    session = config.sessions[0]
    both_finished = 0

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        nonlocal both_finished
        result = _complete_result(
            symbol=str(kwargs["symbol"]),
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )
        both_finished += 1
        if both_finished == 2:
            protocol_path.write_bytes(b"protocol drift\n")
        await asyncio.sleep(0)
        return result

    with pytest.raises(M8L2CaptureSystemError, match="authority/gate") as raised:
        asyncio.run(
            capture_m8_l2_session(
                config,
                "2026-08-08",
                tmp_path / "evidence",
                capture_one,
                protocol_path=protocol_path,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )

    evidence_root = raised.value.evidence_root
    assert evidence_root is not None
    assert (evidence_root / "SYSTEM_FAILURE.json").is_file()
    assert len(list(evidence_root.rglob("capture.ndjson"))) == 2
    assert not (evidence_root / "_SUCCESS").exists()
    assert not (evidence_root / "INSUFFICIENT_DATA").exists()


def test_manifest_is_machine_readable_and_contains_exact_checksum_inventory(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        return _complete_result(
            symbol=str(kwargs["symbol"]),
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    bundle = asyncio.run(
        capture_m8_l2_session(
            config,
            "2026-08-08",
            tmp_path,
            capture_one,
            protocol_path=PROTOCOL_PATH,
            clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
            _test_source_identity=SOURCE,
        )
    )

    payload = json.loads(bundle.manifest_path.read_text())
    checksummed = {
        line[66:] for line in bundle.checksum_path.read_text(encoding="ascii").splitlines()
    }
    inventoried = {item["path"] for item in payload["artifact_inventory"]}
    assert inventoried == checksummed - {"session_manifest.json"}


def test_complete_result_returned_before_scheduled_end_is_nonterminal(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        return _complete_result(
            symbol=str(kwargs["symbol"]),
            root=Path(str(kwargs["stage_root"])),
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )

    with pytest.raises(M8L2CaptureSystemError, match="before the frozen scheduled end") as raised:
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns),
                _test_source_identity=SOURCE,
            )
        )
    assert raised.value.evidence_root is not None
    assert not list((tmp_path / "sessions").glob("*"))


def test_co_lied_text_normalized_artifact_is_rejected_as_system_fault(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        root = Path(str(kwargs["stage_root"]))
        result = _complete_result(
            symbol=str(kwargs["symbol"]),
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )
        target = next(
            item
            for item in result.artifacts
            if item.kind == "normalized_data" and item.path.name == "depth_deltas.parquet"
        )
        target.path.write_text("not parquet but every self-report agrees\n", encoding="utf-8")
        digest = sha256_file(target.path)
        summary_artifact = next(item for item in result.artifacts if item.kind == "capture_summary")
        summary = read_json(summary_artifact.path)
        summary["normalized_dataset_manifests"]["depth_deltas"]["data_sha256"] = digest
        for entry in summary["artifact_inventory_without_summary"]:
            if entry["path"] == target.path.relative_to(root).as_posix():
                entry["sha256"] = digest
                entry["bytes"] = target.path.stat().st_size
        write_json(summary_artifact.path, summary)
        artifacts = tuple(
            CapturedArtifact(
                path=item.path,
                kind=item.kind,
                sha256=(
                    digest
                    if item.path == target.path
                    else sha256_file(item.path)
                    if item.kind == "capture_summary"
                    else item.sha256
                ),
            )
            for item in result.artifacts
        )
        return replace(result, artifacts=artifacts)

    with pytest.raises(M8L2CaptureSystemError, match="Parquet"):
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )


@pytest.mark.parametrize("case", ["summary_claim", "epoch_anchor"])
def test_capture_summary_and_epoch_claim_mismatches_are_nonterminal(
    tmp_path: Path, case: str
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        root = Path(str(kwargs["stage_root"]))
        result = _complete_result(
            symbol=str(kwargs["symbol"]),
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )
        if case == "summary_claim":
            return replace(result, messages=3, normalized_rows=3, reconstructed_rows=3)
        summary_artifact = next(item for item in result.artifacts if item.kind == "capture_summary")
        summary = read_json(summary_artifact.path)
        summary["continuity_epochs"] = 2
        summary["snapshot_anchors"] = 2
        write_json(summary_artifact.path, summary)
        artifacts = tuple(
            replace(item, sha256=sha256_file(item.path)) if item.kind == "capture_summary" else item
            for item in result.artifacts
        )
        return replace(result, continuity_epochs=2, snapshot_anchors=2, artifacts=artifacts)

    with pytest.raises(M8L2CaptureSystemError):
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                tmp_path,
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )


@pytest.mark.parametrize("spoof", ["session", "gate", "source"])
def test_verifier_recomputes_rechecksummed_session_gate_and_source_claims(
    tmp_path: Path, spoof: str
) -> None:
    bundle, _, _ = _complete_bundle(tmp_path)
    payload = _manifest(bundle.root)
    if spoof == "session":
        payload["session"]["scheduled_start_ns"] += 1
    elif spoof == "gate":
        payload["gates"][0]["observed"] = "attacker-controlled"
    else:
        payload["authority"]["runtime_source_tree_sha256"] = "3" * 64
    write_json(bundle.manifest_path, payload)
    _refresh_manifest_checksum(bundle.root)

    with pytest.raises(M8L2VerificationError):
        verify_m8_l2_session_bundle(bundle.root)


def test_output_root_and_sessions_symlinks_are_rejected_before_capture(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    outside = tmp_path / "outside"
    outside.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(outside, target_is_directory=True)

    async def never(**kwargs: object) -> SymbolCaptureResult:
        raise AssertionError(kwargs)

    with pytest.raises(M8L2CaptureSystemError, match="traverses a symlink"):
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                root_link,
                never,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.end_ns + 1),
                _test_source_identity=SOURCE,
            )
        )

    root = tmp_path / "root"
    root.mkdir()
    (root / "sessions").symlink_to(outside, target_is_directory=True)
    with pytest.raises(M8L2CaptureSystemError, match="sessions directory"):
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                root,
                never,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.end_ns + 1),
                _test_source_identity=SOURCE,
            )
        )


def test_fifo_is_rejected_by_producer_and_verifier_without_opening_it(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]

    async def capture_one(**kwargs: object) -> SymbolCaptureResult:
        root = Path(str(kwargs["stage_root"]))
        result = _complete_result(
            symbol=str(kwargs["symbol"]),
            root=root,
            start_ns=session.start_ns,
            end_ns=session.end_ns,
        )
        os.mkfifo(root / "attacker.fifo")
        return result

    with pytest.raises(M8L2CaptureSystemError, match="non-regular filesystem entry"):
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                tmp_path / "producer",
                capture_one,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns, finish_ns=session.end_ns),
                _test_source_identity=SOURCE,
            )
        )

    bundle, _, _ = _complete_bundle(tmp_path / "verifier")
    os.mkfifo(bundle.root / "attacker.fifo")
    with pytest.raises(M8L2VerificationError, match="non-regular filesystem entry"):
        verify_m8_l2_session_bundle(bundle.root)


def test_in_memory_config_forgery_and_production_identity_injection_are_rejected(
    tmp_path: Path,
) -> None:
    config = load_m8_l2_config(CONFIG_PATH)
    session = config.sessions[0]
    forged = replace(
        config,
        capture=replace(
            config.capture,
            max_messages_per_symbol=config.capture.max_messages_per_symbol - 1,
        ),
    )

    async def never(**kwargs: object) -> SymbolCaptureResult:
        raise AssertionError(kwargs)

    with pytest.raises(M8L2CaptureSystemError, match="in-memory configuration"):
        asyncio.run(
            capture_m8_l2_session(
                forged,
                session.date.isoformat(),
                tmp_path / "forged",
                never,
                protocol_path=PROTOCOL_PATH,
                clock=FakeClock(session.start_ns),
                _test_source_identity=SOURCE,
            )
        )
    with pytest.raises(M8L2CaptureSystemError, match="test source identity injection"):
        asyncio.run(
            capture_m8_l2_session(
                config,
                session.date.isoformat(),
                tmp_path / "production",
                never,
                protocol_path=PROTOCOL_PATH,
                _test_source_identity=SOURCE,
            )
        )


def test_terminal_rename_fsyncs_both_old_and_new_parent_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renamed = False
    fsynced_after_rename: set[tuple[int, int]] = set()
    real_rename = l2_module.os.rename
    real_fsync = l2_module.os.fsync

    def observed_rename(source: object, target: object, **kwargs: object) -> None:
        nonlocal renamed
        real_rename(source, target, **kwargs)
        renamed = True

    def observed_fsync(descriptor: int) -> None:
        if renamed:
            metadata = os.fstat(descriptor)
            if stat.S_ISDIR(metadata.st_mode):
                fsynced_after_rename.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(l2_module.os, "rename", observed_rename)
    monkeypatch.setattr(l2_module.os, "fsync", observed_fsync)
    bundle, _, _ = _complete_bundle(tmp_path)

    root_metadata = tmp_path.stat()
    sessions_metadata = (tmp_path / "sessions").stat()
    assert bundle.status == "COMPLETE"
    assert (root_metadata.st_dev, root_metadata.st_ino) in fsynced_after_rename
    assert (sessions_metadata.st_dev, sessions_metadata.st_ino) in fsynced_after_rename
