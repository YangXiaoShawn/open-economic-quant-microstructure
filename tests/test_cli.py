from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from microstructure import cli as cli_module
from microstructure.cli import main
from microstructure.data.binance import CapturedDepth, RawDepthFrame
from microstructure.data.book import BookSnapshot, DepthDelta
from microstructure.data.storage import write_source_manifest
from microstructure.m8_acquisition import M8AcquisitionFailureResult
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


def _mock_m8_config(*, allow_quality_warnings: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        periods=(SimpleNamespace(), SimpleNamespace()),
        study=SimpleNamespace(
            symbols=("BTCUSDT", "ETHUSDT"),
            evidence_tier="FULL_DATA",
        ),
        quality=SimpleNamespace(allow_quality_warnings=allow_quality_warnings),
    )


def test_acquire_m8_command_prints_raw_only_authority_and_propagates_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "study.toml"
    output_root = tmp_path / "m8-data"
    manifest_path = output_root / "_manifests" / "m8-acquisition.manifest-aaaaaaaa.json"
    digest = "a" * 64
    config = _mock_m8_config()
    result = SimpleNamespace(
        output_root=output_root.resolve(),
        manifest_path=manifest_path,
        manifest_sha256=digest,
        metadata_count=2,
        archive_count=8,
        total_raw_evidence_bytes=12_345,
    )
    calls: list[tuple[object, Path]] = []

    def fake_load(path: Path) -> object:
        assert path == config_path
        return config

    def fake_acquire(loaded: object, destination: Path) -> object:
        calls.append((loaded, destination))
        return result

    monkeypatch.setattr(cli_module, "load_m8_config", fake_load)
    monkeypatch.setattr(cli_module, "acquire_m8_archives", fake_acquire)

    exit_code = main(
        [
            "acquire-m8",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert calls == [(config, output_root.resolve())]
    assert json.loads(capsys.readouterr().out) == {
        "archives": 8,
        "csv_members_opened": False,
        "economic_fields_inspected": False,
        "metadata_responses": 2,
        "output_root": str(output_root.resolve()),
        "raw_manifest": str(manifest_path),
        "raw_manifest_sha256": digest,
        "scope": "raw_only",
        "status": "acquired",
        "total_raw_evidence_bytes": 12_345,
    }


def test_acquire_m8_failure_is_reported_with_error_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: _mock_m8_config())

    def fail_acquisition(config: object, root: Path) -> object:
        raise RuntimeError("archive authentication failed")

    monkeypatch.setattr(cli_module, "acquire_m8_archives", fail_acquisition)

    exit_code = main(
        [
            "acquire-m8",
            "--config",
            str(tmp_path / "study.toml"),
            "--output-root",
            str(tmp_path / "data"),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err == "error: archive authentication failed\n"


def test_acquire_m8_deterministic_failure_prints_json_and_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "data"
    attempt_dir = output_root / "_attempts" / f"m8-acquisition-attempt-{'a' * 20}"
    result = M8AcquisitionFailureResult(
        output_root=output_root,
        attempt_dir=attempt_dir,
        attempt_manifest_path=attempt_dir / "failure.json",
        attempt_manifest_sha256="a" * 64,
        checksums_path=attempt_dir / "checksums.sha256",
        checksums_sha256="b" * 64,
        terminal_path=attempt_dir / "INSUFFICIENT_DATA",
        reason_code="DECLARED_OBJECT_UNAVAILABLE",
        diagnostic="BinanceArchiveHTTPError: HTTP 404",
        failed_symbol="BTCUSDT",
        failed_date=date(2024, 1, 3),
        failed_role="train",
        completed_count=2,
        remaining_count=7,
        retained_inventory_sha256="c" * 64,
        retained_artifact_count=6,
        total_raw_evidence_bytes=1_234,
        manifest=None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: _mock_m8_config())
    monkeypatch.setattr(cli_module, "acquire_m8_archives", lambda config, root: result)

    exit_code = main(
        [
            "acquire-m8",
            "--config",
            str(tmp_path / "study.toml"),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "INSUFFICIENT_DATA"
    assert payload["reason_code"] == "DECLARED_OBJECT_UNAVAILABLE"
    assert payload["failed_symbol"] == "BTCUSDT"
    assert payload["failed_date"] == "2024-01-03"
    assert payload["failure_manifest_sha256"] == "a" * 64
    assert payload["retained_inventory_sha256"] == "c" * 64
    assert payload["csv_members_opened"] is False


def test_reproduce_m8_propagates_explicit_manifest_coordinates_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "study.toml"
    run_dir = tmp_path / "run"
    manifest_path = tmp_path / "data" / "_manifests" / "input.json"
    digest = "0123456789abcdef" * 4
    config = _mock_m8_config()
    calls: list[tuple[object, Path, Path, str]] = []

    def fake_reproduce(
        loaded: object,
        destination: Path,
        *,
        raw_manifest_path: Path,
        raw_manifest_sha256: str,
    ) -> SimpleNamespace:
        calls.append((loaded, destination, raw_manifest_path, raw_manifest_sha256))
        return SimpleNamespace(
            path=run_dir,
            status="COMPLETE",
            raw_manifest_sha256=raw_manifest_sha256,
            normalized_manifest_sha256="f" * 64,
        )

    bundle = SimpleNamespace(
        run_id="m8-unit",
        evidence_tier="FULL_DATA",
        observed_start_utc="2024-01-03T00:00:00Z",
        observed_end_utc="2024-01-06T23:59:59Z",
    )
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: config)
    monkeypatch.setattr(cli_module, "reproduce_m8", fake_reproduce)
    monkeypatch.setattr(cli_module, "load_run_bundle", lambda path: bundle)

    exit_code = main(
        [
            "reproduce-m8",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--raw-manifest",
            str(manifest_path),
            "--raw-manifest-sha256",
            digest,
        ]
    )

    assert exit_code == 0
    assert calls == [(config, run_dir, manifest_path, digest)]
    assert json.loads(capsys.readouterr().out) == {
        "evidence_tier": "FULL_DATA",
        "observed_end_utc": "2024-01-06T23:59:59Z",
        "observed_start_utc": "2024-01-03T00:00:00Z",
        "normalized_manifest_sha256": "f" * 64,
        "raw_manifest_sha256": digest,
        "run_dir": str(run_dir),
        "run_id": "m8-unit",
        "status": "COMPLETE",
    }


def test_reproduce_m8_producer_failure_uses_error_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: _mock_m8_config())

    def fail_producer(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("locked production failed")

    monkeypatch.setattr(cli_module, "reproduce_m8", fail_producer)
    exit_code = main(
        [
            "reproduce-m8",
            "--config",
            str(tmp_path / "study.toml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--raw-manifest",
            str(tmp_path / "input.json"),
            "--raw-manifest-sha256",
            "c" * 64,
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err == "error: locked production failed\n"


def test_reproduce_m8_preserves_verified_insufficient_data_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = "d" * 64
    run_dir = tmp_path / "run"
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: _mock_m8_config())
    monkeypatch.setattr(
        cli_module,
        "reproduce_m8",
        lambda *args, **kwargs: SimpleNamespace(
            path=run_dir,
            status="INSUFFICIENT_DATA",
            raw_manifest_sha256=digest,
            normalized_manifest_sha256=None,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_run_bundle",
        lambda path: pytest.fail("an insufficient-data result is not a complete run bundle"),
    )

    exit_code = main(
        [
            "reproduce-m8",
            "--config",
            str(tmp_path / "study.toml"),
            "--run-dir",
            str(run_dir),
            "--raw-manifest",
            str(tmp_path / "raw.json"),
            "--raw-manifest-sha256",
            digest,
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "normalized_manifest_sha256": None,
        "raw_manifest_sha256": digest,
        "run_dir": str(run_dir),
        "status": "INSUFFICIENT_DATA",
    }


def test_verify_m8_accepts_complete_or_insufficient_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = "e" * 64
    run_dir = tmp_path / "run"
    raw_manifest = tmp_path / "raw.json"
    config = _mock_m8_config()
    calls: list[tuple[object, Path, Path, str]] = []

    def fake_verify(
        path: Path,
        observed_config: object,
        *,
        raw_manifest_path: Path,
        raw_manifest_sha256: str,
    ) -> SimpleNamespace:
        calls.append((observed_config, path, raw_manifest_path, raw_manifest_sha256))
        return SimpleNamespace(
            path=run_dir,
            status="INSUFFICIENT_DATA",
            raw_manifest_sha256=digest,
            normalized_manifest_sha256=None,
        )

    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: config)
    monkeypatch.setattr(cli_module, "verify_m8_result", fake_verify)
    monkeypatch.setattr(cli_module, "verify_checksums", lambda path: 17)

    assert (
        main(
            [
                "verify-m8",
                "--config",
                str(tmp_path / "study.toml"),
                "--run-dir",
                str(run_dir),
                "--raw-manifest",
                str(raw_manifest),
                "--raw-manifest-sha256",
                digest,
            ]
        )
        == 0
    )
    assert calls == [(config, run_dir, raw_manifest, digest)]
    assert json.loads(capsys.readouterr().out)["protected_files"] == 17


def test_report_m8_exposes_frozen_insufficient_report_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = "9" * 64
    run_dir = tmp_path / "run"
    report = run_dir / "reports" / "insufficient_data.md"
    report.parent.mkdir(parents=True)
    report.write_text("# INSUFFICIENT_DATA\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: _mock_m8_config())
    monkeypatch.setattr(
        cli_module,
        "verify_m8_result",
        lambda *args, **kwargs: SimpleNamespace(
            path=run_dir,
            status="INSUFFICIENT_DATA",
            raw_manifest_sha256=digest,
            normalized_manifest_sha256=None,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "write_report_set",
        lambda *args, **kwargs: pytest.fail("terminal failure report must not be rewritten"),
    )

    assert (
        main(
            [
                "report-m8",
                "--config",
                str(tmp_path / "study.toml"),
                "--run-dir",
                str(run_dir),
                "--raw-manifest",
                str(tmp_path / "raw.json"),
                "--raw-manifest-sha256",
                digest,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"] == str(report)
    assert payload["reports_regenerated"] is False


@pytest.mark.parametrize(
    "manifest_args",
    [
        [],
        ["--raw-manifest", "input.json"],
        ["--raw-manifest-sha256", "a" * 64],
        ["--raw-manifest", "input.json", "--raw-manifest-sha256", "A" * 64],
        ["--raw-manifest", "input.json", "--raw-manifest-sha256", "a" * 63],
        ["--raw-manifest", "input.json", "--raw-manifest-sha256", "g" * 64],
    ],
)
def test_reproduce_m8_rejects_missing_or_noncanonical_manifest_coordinates(
    manifest_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "reproduce_m8",
        lambda *args, **kwargs: pytest.fail("producer must not run after argument failure"),
    )

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "reproduce-m8",
                "--config",
                str(tmp_path / "study.toml"),
                "--run-dir",
                str(tmp_path / "run"),
                *manifest_args,
            ]
        )

    assert caught.value.code == 2


def test_m8_make_targets_keep_reproduction_explicit_and_checks_offline() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    reproduce_recipe = makefile.split("reproduce-m8:\n", maxsplit=1)[1].split(
        "\nverify-run:", maxsplit=1
    )[0]
    assert 'test -n "$(M8_RAW_MANIFEST)"' in reproduce_recipe
    assert 'test -n "$(M8_RAW_MANIFEST_SHA256)"' in reproduce_recipe
    assert '--raw-manifest "$(M8_RAW_MANIFEST)"' in reproduce_recipe
    assert '--raw-manifest-sha256 "$(M8_RAW_MANIFEST_SHA256)"' in reproduce_recipe
    assert "latest" not in reproduce_recipe.lower()
    for target, following in (("verify-m8-run", "report:"), ("report-m8", "dashboard:")):
        recipe = makefile.split(f"{target}:\n", maxsplit=1)[1].split(f"\n{following}", maxsplit=1)[
            0
        ]
        assert 'test -n "$(M8_RAW_MANIFEST)"' in recipe
        assert 'test -n "$(M8_RAW_MANIFEST_SHA256)"' in recipe
        assert '--raw-manifest "$(M8_RAW_MANIFEST)"' in recipe
        assert '--raw-manifest-sha256 "$(M8_RAW_MANIFEST_SHA256)"' in recipe
    check_dependencies = makefile.split("\ncheck:", maxsplit=1)[1].splitlines()[0]
    assert "download-m8" not in check_dependencies


@pytest.mark.parametrize(("status", "expected_exit"), [("COMPLETE", 0), ("INSUFFICIENT_DATA", 1)])
def test_capture_m8_l2_session_command_uses_frozen_runner_without_source_override(
    status: str,
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "m8-l2.toml"
    output_root = tmp_path / "l2"
    config = object()
    adapter = object()
    target = output_root / "sessions" / "authority"
    bundle = SimpleNamespace(
        root=target,
        status=status,
        session_id="a" * 64,
        session_date="2026-08-08",
        role="train",
        manifest_path=target / "session_manifest.json",
        manifest_sha256="b" * 64,
        checksum_path=target / "CHECKSUMS.sha256",
        marker_path=target / ("_SUCCESS" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
        reason_codes=() if status == "COMPLETE" else ("GATE_VALID_CONTINUITY_EPOCH_BTCUSDT",),
    )
    calls: list[tuple[object, str, Path, object]] = []

    async def fake_capture(
        loaded: object,
        session_date: str,
        root: Path,
        capture_one: object,
        **kwargs: object,
    ) -> object:
        assert kwargs == {}
        calls.append((loaded, session_date, root, capture_one))
        return bundle

    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: config)
    monkeypatch.setattr(cli_module, "BinanceM8L2Capture", lambda: adapter)
    monkeypatch.setattr(cli_module, "capture_m8_l2_session", fake_capture)

    exit_code = main(
        [
            "capture-m8-l2-session",
            "--config",
            str(config_path),
            "--date",
            "2026-08-08",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == expected_exit
    assert calls == [(config, "2026-08-08", output_root.resolve(), adapter)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert payload["session_date"] == "2026-08-08"
    assert payload["session_manifest_sha256"] == "b" * 64
    assert payload["live_trading"] is False


def test_capture_m8_l2_session_system_failure_uses_error_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: object())

    async def fail(*args: object, **kwargs: object) -> object:
        raise OSError("injected capture I/O failure")

    monkeypatch.setattr(cli_module, "capture_m8_l2_session", fail)

    exit_code = main(
        [
            "capture-m8-l2-session",
            "--config",
            str(tmp_path / "m8-l2.toml"),
            "--date",
            "2026-08-08",
            "--output-root",
            str(tmp_path / "l2"),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err == "error: injected capture I/O failure\n"


def test_m8_l2_make_target_is_explicit_and_check_remains_offline() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "M8_L2_CONFIG ?= configs/m8_l2_capture_study.toml" in makefile
    assert "M8_L2_DATA_ROOT ?= data/m8_l2" in makefile
    assert "M8_L2_SESSION_DATE ?=" in makefile
    recipe = makefile.split("capture-m8-l2-session:\n", maxsplit=1)[1].split(
        "\nvalidate-data:", maxsplit=1
    )[0]
    assert 'test -n "$(M8_L2_SESSION_DATE)"' in recipe
    assert '--config "$(M8_L2_CONFIG)"' in recipe
    assert '--date "$(M8_L2_SESSION_DATE)"' in recipe
    assert '--output-root "$(M8_L2_DATA_ROOT)"' in recipe
    check_dependencies = makefile.split("\ncheck:", maxsplit=1)[1].splitlines()[0]
    assert "capture-m8-l2-session" not in check_dependencies


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
