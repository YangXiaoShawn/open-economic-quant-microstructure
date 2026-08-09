from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
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
from microstructure.m8_l2_capture import M8L2VerificationError
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


def test_report_m8_writes_self_contained_insufficient_report_outside_frozen_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = "9" * 64
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "external-report"
    run_dir.mkdir()
    failure = {
        "failed_symbol": "ETHUSDT",
        "failed_date": "2024-01-04",
        "failed_role": "validation",
        "failure_stage": "development_normalization",
        "reason_code": "ARCHIVE_QUALITY_GATE",
        "reason": "53 temporal.long_silence warnings violated the frozen gate",
        "replacement_date_selected": False,
        "reselection_performed": False,
        "config_sha256": "a" * 64,
        "config_source_sha256": "b" * 64,
        "raw_acquisition_manifest_sha256": digest,
        "bundled_raw_acquisition_manifest_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
        "selection_started": False,
        "selection_completed_symbols": [],
        "aggregate_lock_committed": False,
        "held_out_member_opened": False,
        "endpoint_evaluation_started": False,
        "endpoint_evaluation_completed": False,
        "endpoint_evaluation_completed_symbols": [],
        "predictions_published": False,
        "endpoint_artifacts_published": False,
        "completed_normalizations": [{"symbol": "BTCUSDT", "date": "2024-01-03", "role": "train"}],
        "stopped_before": [
            {"symbol": "BTCUSDT", "date": "2024-01-05", "role": "primary_test"},
            {"symbol": "BTCUSDT", "date": "2024-01-06", "role": "replication_test"},
        ],
    }
    provenance = {
        "git": {
            "commit": "e" * 40,
            "dirty": False,
            "source_tree_sha256": "f" * 64,
        }
    }
    run_manifest = {
        "research": {"endpoint_status": "insufficient_data"},
        "execution_assumptions": {
            "status": "NOT_RUN",
            "fills_calculated": False,
            "pnl_calculated": False,
            "capacity_calculated": False,
        },
    }
    for name, payload in (
        ("failure.json", failure),
        ("provenance.json", provenance),
        ("run_manifest.json", run_manifest),
    ):
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    bundled_before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    monkeypatch.setattr(cli_module, "load_m8_config", lambda path: _mock_m8_config())
    verification_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_verify(*args: object, **kwargs: object) -> object:
        verification_calls.append((args, kwargs))
        return SimpleNamespace(
            path=run_dir,
            status="INSUFFICIENT_DATA",
            raw_manifest_sha256=digest,
            normalized_manifest_sha256=None,
        )

    monkeypatch.setattr(cli_module, "verify_m8_result", fake_verify)
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
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    report = output_dir / "insufficient_data.md"
    assert payload["report"] == str(report)
    assert payload["reports_regenerated"] is True
    assert payload["source_bundle_modified"] is False
    assert payload["report_sha256"] == sha256_file(report)
    rendered = report.read_text(encoding="utf-8")
    for expected in (
        "2024-01-03` through `2024-01-06",
        "ETHUSDT",
        "2024-01-04",
        "validation",
        "ARCHIVE_QUALITY_GATE",
        "53 temporal.long_silence warnings",
        "Config semantic SHA-256",
        "Raw acquisition manifest SHA-256",
        "Git commit",
        "Git dirty: `false`",
        "Source-tree SHA-256",
        "Candidate selection started: `false`",
        "Held-out member opened: `false`",
        "Endpoint evaluation started: `false`",
        "Execution status: `NOT_RUN`",
    ):
        assert expected in rendered
    assert {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    } == bundled_before
    assert len(verification_calls) == 2
    assert verification_calls[0] == verification_calls[1]


def test_report_m8_rejects_output_inside_immutable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = "8" * 64
    run_dir = tmp_path / "run"
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

    exit_code = main(
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
            "--output-dir",
            str(run_dir / "reports"),
        ]
    )

    assert exit_code == 2
    assert "outside the immutable run bundle" in capsys.readouterr().err


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
        session_date="2026-08-10",
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
            "2026-08-10",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == expected_exit
    assert calls == [(config, "2026-08-10", output_root.resolve(), adapter)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert payload["session_date"] == "2026-08-10"
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
            "2026-08-10",
            "--output-root",
            str(tmp_path / "l2"),
        ]
    )

    assert exit_code == 2
    assert capsys.readouterr().err == "error: injected capture I/O failure\n"


@pytest.mark.parametrize("status", ["COMPLETE", "INSUFFICIENT_DATA"])
def test_verify_m8_l2_session_accepts_both_terminal_states(
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "m8-l2.toml"
    bundle_dir = tmp_path / "sessions" / ("a" * 64)
    config = object()
    result = SimpleNamespace(
        root=bundle_dir.absolute(),
        status=status,
        session_id="a" * 64,
        session_date="2026-08-10",
        role="train",
        manifest_path=bundle_dir / "session_manifest.json",
        manifest_sha256="b" * 64,
        checksum_path=bundle_dir / "CHECKSUMS.sha256",
        marker_path=bundle_dir / ("_SUCCESS" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
        reason_codes=() if status == "COMPLETE" else ("GATE_CROSS_SYMBOL_OVERLAP",),
    )
    calls: list[tuple[Path, object]] = []

    def fake_verify(path: Path, *, expected_config: object) -> object:
        calls.append((path, expected_config))
        return result

    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: config)
    monkeypatch.setattr(cli_module, "verify_m8_l2_session_bundle", fake_verify)

    exit_code = main(
        [
            "verify-m8-l2-session",
            "--config",
            str(config_path),
            "--bundle-dir",
            str(bundle_dir),
        ]
    )

    assert exit_code == 0
    assert calls == [(bundle_dir, config)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert payload["integrity"] == "verified"
    assert payload["session_id"] == "a" * 64
    assert payload["reason_codes"] == list(result.reason_codes)
    assert payload["live_trading"] is False


@pytest.mark.parametrize(
    "message",
    [
        "checksum mismatch for symbols/BTCUSDT/capture_summary.json",
        "caller config semantics differ from the session authority",
    ],
)
def test_verify_m8_l2_session_rejects_tampering_and_wrong_config(
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    bundle_dir = tmp_path / "sessions" / ("a" * 64)
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: config)

    def fail(path: Path, *, expected_config: object) -> object:
        assert path == bundle_dir
        assert expected_config is config
        raise M8L2VerificationError(message)

    monkeypatch.setattr(cli_module, "verify_m8_l2_session_bundle", fail)

    exit_code = main(
        [
            "verify-m8-l2-session",
            "--config",
            str(tmp_path / "m8-l2.toml"),
            "--bundle-dir",
            str(bundle_dir),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {message}\n"


def test_m8_l2_make_target_is_explicit_and_check_remains_offline() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "M8_L2_CONFIG ?= configs/m8_l2_capture_study.toml" in makefile
    assert "M8_L2_DATA_ROOT ?= data/m8_l2" in makefile
    assert "M8_L2_SESSION_DATE ?=" in makefile
    assert "M8_L2_BUNDLE_DIR ?=" in makefile
    capture_recipe = makefile.split("capture-m8-l2-session:\n", maxsplit=1)[1].split(
        "\nverify-m8-l2-session:", maxsplit=1
    )[0]
    assert 'test -n "$(M8_L2_SESSION_DATE)"' in capture_recipe
    assert '--config "$(M8_L2_CONFIG)"' in capture_recipe
    assert '--date "$(M8_L2_SESSION_DATE)"' in capture_recipe
    assert '--output-root "$(M8_L2_DATA_ROOT)"' in capture_recipe
    verify_recipe = makefile.split("verify-m8-l2-session:\n", maxsplit=1)[1].split(
        "\nvalidate-data:", maxsplit=1
    )[0]
    assert 'test -n "$(M8_L2_BUNDLE_DIR)"' in verify_recipe
    assert '--config "$(M8_L2_CONFIG)"' in verify_recipe
    assert '--bundle-dir "$(M8_L2_BUNDLE_DIR)"' in verify_recipe
    check_dependencies = makefile.split("\ncheck:", maxsplit=1)[1].splitlines()[0]
    assert "capture-m8-l2-session" not in check_dependencies
    assert "verify-m8-l2-session" not in check_dependencies


def test_m8_l2_make_contract_normalizes_only_valid_terminal_exit_one(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/sh\nprintf \'{"status":"INSUFFICIENT_DATA"}\\n\'\nexit "$FAKE_EXIT"\n',
        encoding="ascii",
    )
    fake_python.chmod(0o755)
    base_environment = {**os.environ, "FAKE_EXIT": "1"}
    command = [
        "make",
        "--no-print-directory",
        "capture-m8-l2-session",
        f"PYTHON={fake_python}",
        "M8_L2_SESSION_DATE=2026-08-10",
    ]

    valid_terminal = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid_terminal.returncode == 0
    assert json.loads(valid_terminal.stdout)["status"] == "INSUFFICIENT_DATA"

    system_failure = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**base_environment, "FAKE_EXIT": "2"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert system_failure.returncode != 0

    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    for command_name in (
        "capture-m8-l2-session",
        "lock-m8-l2-development",
        "verify-m8-l2-development-lock",
        "reproduce-m8-l2",
        "verify-m8-l2-run",
        "report-m8-l2",
    ):
        recipe_line = next(
            line for line in makefile.splitlines() if f"microstructure.cli {command_name} " in line
        )
        assert "|| status=$$?" in recipe_line
        assert 'if [ "$$status" -eq 1 ]; then exit 0; fi' in recipe_line
        assert 'exit "$$status"' in recipe_line


def _m8_l2_development_cli_fixture(
    tmp_path: Path,
    *,
    train_status: str = "COMPLETE",
    validation_status: str = "COMPLETE",
) -> tuple[list[str], SimpleNamespace, SimpleNamespace]:
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    train_root.mkdir()
    validation_root.mkdir()
    train_checksums = train_root / "CHECKSUMS.sha256"
    validation_checksums = validation_root / "CHECKSUMS.sha256"
    train_checksums.write_bytes(b"train checksums authority\n")
    validation_checksums.write_bytes(b"validation checksums authority\n")
    train = SimpleNamespace(
        root=train_root.absolute(),
        status=train_status,
        session_id="1" * 64,
        session_date="2026-08-10",
        role="train",
        manifest_path=train_root / "session_manifest.json",
        manifest_sha256="a" * 64,
        checksum_path=train_checksums,
        marker_path=train_root
        / ("_SUCCESS" if train_status == "COMPLETE" else "INSUFFICIENT_DATA"),
        reason_codes=() if train_status == "COMPLETE" else ("GATE_TRAIN",),
    )
    validation = SimpleNamespace(
        root=validation_root.absolute(),
        status=validation_status,
        session_id="2" * 64,
        session_date="2026-08-11",
        role="validation",
        manifest_path=validation_root / "session_manifest.json",
        manifest_sha256="b" * 64,
        checksum_path=validation_checksums,
        marker_path=(
            validation_root
            / ("_SUCCESS" if validation_status == "COMPLETE" else "INSUFFICIENT_DATA")
        ),
        reason_codes=() if validation_status == "COMPLETE" else ("GATE_VALIDATION",),
    )
    arguments = [
        "--capture-config",
        str(tmp_path / "capture.toml"),
        "--analysis-config",
        str(tmp_path / "analysis.toml"),
        "--train-bundle-dir",
        str(train_root),
        "--train-manifest-sha256",
        train.manifest_sha256,
        "--train-checksums-sha256",
        sha256_file(train_checksums),
        "--validation-bundle-dir",
        str(validation_root),
        "--validation-manifest-sha256",
        validation.manifest_sha256,
        "--validation-checksums-sha256",
        sha256_file(validation_checksums),
        "--lock-dir",
        str(tmp_path / "development-lock"),
    ]
    return arguments, train, validation


def test_lock_m8_l2_development_uses_only_explicit_session_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, train, validation = _m8_l2_development_cli_fixture(tmp_path)
    capture_config = object()
    analysis_config = object()
    strict_input = object()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: capture_config)
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: analysis_config)

    def fake_verify(path: Path, *, expected_config: object) -> object:
        assert expected_config is capture_config
        return train if path == train.root else validation

    def fake_strict_input(
        path: Path,
        *,
        expected_config: object,
        expected_date: str,
        expected_role: str,
        expected_file_authority: object,
        expected_campaign: object | None,
    ) -> object:
        calls.append(
            (
                path,
                expected_config,
                expected_date,
                expected_role,
                expected_file_authority,
                expected_campaign,
            )
        )
        return strict_input

    def fake_lock(
        capture: object,
        analysis: object,
        train_path: Path,
        validation_path: Path,
        lock_path: Path,
        *,
        input_loader: object,
        expected_session_file_authorities: object,
    ) -> object:
        assert capture is capture_config
        assert analysis is analysis_config
        assert (train_path, validation_path) == (train.root, validation.root)
        assert callable(input_loader)
        assert expected_session_file_authorities == {
            "2026-08-10": cli_module.L2SessionFileAuthority(
                train.manifest_sha256,
                hashlib.sha256(train.checksum_path.read_bytes()).hexdigest(),
            ),
            "2026-08-11": cli_module.L2SessionFileAuthority(
                validation.manifest_sha256,
                hashlib.sha256(validation.checksum_path.read_bytes()).hexdigest(),
            ),
        }
        assert (
            input_loader(
                train_path,
                expected_config=capture_config,
                expected_date="2026-08-10",
                expected_role="train",
            )
            is strict_input
        )
        return SimpleNamespace(
            root=lock_path,
            aggregate_path=lock_path / "development_lock.json",
            aggregate_sha256="d" * 64,
            marker_path=lock_path / "_LOCKED",
            created_at_utc="2026-08-11T15:00:00Z",
            children=(
                SimpleNamespace(
                    symbol="BTCUSDT",
                    endpoint="event_20",
                    path=lock_path / "BTCUSDT" / "event_20" / "lock.json",
                    sha256="e" * 64,
                    selection_lock_sha256="f" * 64,
                    fitted_state_sha256="0" * 64,
                ),
            ),
        )

    monkeypatch.setattr(cli_module, "verify_m8_l2_session_bundle", fake_verify)
    monkeypatch.setattr(cli_module, "verify_m8_l2_development_input", fake_strict_input)
    monkeypatch.setattr(cli_module, "lock_m8_l2_development", fake_lock)

    assert main(["lock-m8-l2-development", *arguments]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "LOCKED"
    assert payload["development_lock_sha256"] == "d" * 64
    assert payload["heldout_accessed"] is False
    assert payload["live_trading"] is False
    assert calls[0][4].manifest_sha256 == "a" * 64
    assert calls[0][4].checksums_sha256 == sha256_file(train.checksum_path)


def test_lock_m8_l2_development_returns_one_for_verified_insufficient_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, train, validation = _m8_l2_development_cli_fixture(
        tmp_path,
        validation_status="INSUFFICIENT_DATA",
    )
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: object())
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: object())
    monkeypatch.setattr(
        cli_module,
        "verify_m8_l2_session_bundle",
        lambda path, **kwargs: train if path == train.root else validation,
    )
    lock_dir = Path(arguments[arguments.index("--lock-dir") + 1]).absolute()
    monkeypatch.setattr(
        cli_module,
        "lock_m8_l2_development",
        lambda *args, **kwargs: SimpleNamespace(
            root=lock_dir,
            aggregate_path=lock_dir / "development_lock.json",
            aggregate_sha256="d" * 64,
            marker_path=lock_dir / "_NOT_CREATED",
            created_at_utc="2026-08-11T15:00:00Z",
            children=(),
            status="NOT_CREATED",
            reason_codes=("DEVELOPMENT_SESSION_INSUFFICIENT::validation::GATE_VALIDATION",),
        ),
    )

    assert main(["lock-m8-l2-development", *arguments]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_CREATED"
    assert payload["development_lock_sha256"] == "d" * 64
    assert payload["reason_codes"] == [
        "DEVELOPMENT_SESSION_INSUFFICIENT::validation::GATE_VALIDATION"
    ]
    assert payload["heldout_accessed"] is False


def test_verify_m8_l2_development_lock_binds_expected_aggregate_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, train, validation = _m8_l2_development_cli_fixture(tmp_path)
    capture_config = object()
    analysis_config = object()
    expected_lock_sha = "c" * 64
    lock_dir = (tmp_path / "development-lock").absolute()

    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: capture_config)
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: analysis_config)
    monkeypatch.setattr(
        cli_module,
        "verify_m8_l2_session_bundle",
        lambda path, **kwargs: train if path == train.root else validation,
    )

    def fake_verify_lock(*args: object, **kwargs: object) -> object:
        assert args == (capture_config, analysis_config, train.root, validation.root, lock_dir)
        assert kwargs == {"expected_lock_sha256": expected_lock_sha}
        return SimpleNamespace(
            root=lock_dir,
            aggregate_path=lock_dir / "development_lock.json",
            aggregate_sha256=expected_lock_sha,
            marker_path=lock_dir / "_LOCKED",
            created_at_utc="2026-08-11T15:00:00Z",
            children=(),
        )

    monkeypatch.setattr(cli_module, "verify_m8_l2_development_lock", fake_verify_lock)

    assert (
        main(
            [
                "verify-m8-l2-development-lock",
                *arguments,
                "--development-lock-sha256",
                expected_lock_sha,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["integrity"] == "verified"
    assert payload["development_lock_sha256"] == expected_lock_sha


def test_verify_m8_l2_not_created_authority_returns_one_with_verified_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments, train, validation = _m8_l2_development_cli_fixture(
        tmp_path,
        validation_status="INSUFFICIENT_DATA",
    )
    expected_sha = "c" * 64
    lock_dir = Path(arguments[arguments.index("--lock-dir") + 1]).absolute()
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: object())
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: object())
    monkeypatch.setattr(
        cli_module,
        "verify_m8_l2_session_bundle",
        lambda path, **kwargs: train if path == train.root else validation,
    )
    monkeypatch.setattr(
        cli_module,
        "verify_m8_l2_development_lock",
        lambda *args, **kwargs: SimpleNamespace(
            root=lock_dir,
            aggregate_path=lock_dir / "development_lock.json",
            aggregate_sha256=expected_sha,
            marker_path=lock_dir / "_NOT_CREATED",
            created_at_utc="2026-08-11T15:00:00Z",
            children=(),
            status="NOT_CREATED",
            reason_codes=("DEVELOPMENT_SESSION_INSUFFICIENT::validation::GATE_VALIDATION",),
        ),
    )

    assert (
        main(
            [
                "verify-m8-l2-development-lock",
                *arguments,
                "--development-lock-sha256",
                expected_sha,
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "NOT_CREATED"
    assert payload["integrity"] == "verified"
    assert payload["terminal_marker"].endswith("/_NOT_CREATED")


def test_m8_l2_development_cli_rejects_noncanonical_authority_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _ = _m8_l2_development_cli_fixture(tmp_path)
    digest_index = arguments.index("--train-manifest-sha256") + 1
    arguments[digest_index] = "A" * 64
    monkeypatch.setattr(
        cli_module,
        "lock_m8_l2_development",
        lambda *args, **kwargs: pytest.fail("argument failure must precede lock production"),
    )

    with pytest.raises(SystemExit) as caught:
        main(["lock-m8-l2-development", *arguments])

    assert caught.value.code == 2


def test_m8_l2_development_make_targets_are_explicit_and_not_in_check() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    gitignore_lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/m8_l2/**" in gitignore_lines
    lock_recipe = makefile.split("lock-m8-l2-development:\n", maxsplit=1)[1].split(
        "\nverify-m8-l2-development-lock:", maxsplit=1
    )[0]
    verify_recipe = makefile.split("verify-m8-l2-development-lock:\n", maxsplit=1)[1].split(
        "\nvalidate-data:", maxsplit=1
    )[0]
    for variable in (
        "M8_L2_TRAIN_BUNDLE_DIR",
        "M8_L2_TRAIN_MANIFEST_SHA256",
        "M8_L2_TRAIN_CHECKSUMS_SHA256",
        "M8_L2_VALIDATION_BUNDLE_DIR",
        "M8_L2_VALIDATION_MANIFEST_SHA256",
        "M8_L2_VALIDATION_CHECKSUMS_SHA256",
        "M8_L2_DEVELOPMENT_LOCK_DIR",
    ):
        assert f'test -n "$({variable})"' in lock_recipe
        assert f'"$({variable})"' in lock_recipe
        assert f'test -n "$({variable})"' in verify_recipe
        assert f'"$({variable})"' in verify_recipe
    assert 'test -n "$(M8_L2_DEVELOPMENT_LOCK_SHA256)"' in verify_recipe
    assert '--development-lock-sha256 "$(M8_L2_DEVELOPMENT_LOCK_SHA256)"' in verify_recipe
    for recipe in (lock_recipe, verify_recipe):
        assert "latest" not in recipe.lower()
    check_dependencies = makefile.split("\ncheck:", maxsplit=1)[1].splitlines()[0]
    assert "lock-m8-l2-development" not in check_dependencies
    assert "verify-m8-l2-development-lock" not in check_dependencies


def test_m8_l2_final_make_targets_bind_four_sessions_and_stay_offline() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "M8_L2_RUN_DIR ?= artifacts/runs/binance-m8-live-l2" in makefile
    assert "M8_L2_REPORT_DIR ?= artifacts/runs/binance-m8-live-l2-reports" in makefile
    reproduce_recipe = makefile.split("reproduce-m8-l2:\n", maxsplit=1)[1].split(
        "\nverify-m8-l2-run:", maxsplit=1
    )[0]
    verify_recipe = makefile.split("verify-m8-l2-run:\n", maxsplit=1)[1].split(
        "\nreport-m8-l2:", maxsplit=1
    )[0]
    report_recipe = makefile.split("report-m8-l2:\n", maxsplit=1)[1].split(
        "\nvalidate-data:", maxsplit=1
    )[0]
    authority_variables = (
        "M8_L2_TRAIN_BUNDLE_DIR",
        "M8_L2_TRAIN_MANIFEST_SHA256",
        "M8_L2_TRAIN_CHECKSUMS_SHA256",
        "M8_L2_VALIDATION_BUNDLE_DIR",
        "M8_L2_VALIDATION_MANIFEST_SHA256",
        "M8_L2_VALIDATION_CHECKSUMS_SHA256",
        "M8_L2_DEVELOPMENT_LOCK_DIR",
        "M8_L2_DEVELOPMENT_LOCK_SHA256",
        "M8_L2_PRIMARY_BUNDLE_DIR",
        "M8_L2_PRIMARY_MANIFEST_SHA256",
        "M8_L2_PRIMARY_CHECKSUMS_SHA256",
        "M8_L2_REPLICATION_BUNDLE_DIR",
        "M8_L2_REPLICATION_MANIFEST_SHA256",
        "M8_L2_REPLICATION_CHECKSUMS_SHA256",
    )
    for recipe in (reproduce_recipe, verify_recipe, report_recipe):
        for variable in authority_variables:
            assert f'test -n "$({variable})"' in recipe
            assert f'"$({variable})"' in recipe
        for role in ("train", "validation", "primary", "replication"):
            assert f"--{role}-bundle-dir" in recipe
            assert f"--{role}-manifest-sha256" in recipe
            assert f"--{role}-checksums-sha256" in recipe
        assert "--development-lock-dir" in recipe
        assert "--development-lock-sha256" in recipe
        assert ' --run-dir "$(M8_L2_RUN_DIR)"' in recipe
        assert "latest" not in recipe.lower()
    for recipe in (verify_recipe, report_recipe):
        assert 'test -n "$(M8_L2_RUN_MANIFEST_SHA256)"' in recipe
        assert 'test -n "$(M8_L2_RUN_CHECKSUMS_SHA256)"' in recipe
        assert '--run-manifest-sha256 "$(M8_L2_RUN_MANIFEST_SHA256)"' in recipe
        assert '--run-checksums-sha256 "$(M8_L2_RUN_CHECKSUMS_SHA256)"' in recipe
    assert '--output-dir "$(M8_L2_REPORT_DIR)"' in report_recipe
    check_dependencies = makefile.split("\ncheck:", maxsplit=1)[1].splitlines()[0]
    for target in ("reproduce-m8-l2", "verify-m8-l2-run", "report-m8-l2"):
        assert target not in check_dependencies


def _m8_l2_study_cli_arguments(
    tmp_path: Path,
    *,
    include_run_authority: bool,
) -> list[str]:
    arguments = [
        "--capture-config",
        str(tmp_path / "capture.toml"),
        "--analysis-config",
        str(tmp_path / "analysis.toml"),
    ]
    for index, role in enumerate(("train", "validation", "primary", "replication"), start=1):
        arguments.extend(
            [
                f"--{role}-bundle-dir",
                str(tmp_path / role),
                f"--{role}-manifest-sha256",
                format(index, "x") * 64,
                f"--{role}-checksums-sha256",
                format(index + 4, "x") * 64,
            ]
        )
    arguments.extend(
        [
            "--development-lock-dir",
            str(tmp_path / "development-lock"),
            "--development-lock-sha256",
            "9" * 64,
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    if include_run_authority:
        arguments.extend(
            [
                "--run-manifest-sha256",
                "a" * 64,
                "--run-checksums-sha256",
                "b" * 64,
            ]
        )
    return arguments


def _fake_m8_l2_study_result(tmp_path: Path, status: str) -> SimpleNamespace:
    root = (tmp_path / "run").absolute()
    return SimpleNamespace(
        root=root,
        status=status,
        manifest_path=root / "run_manifest.json",
        manifest_sha256="a" * 64,
        checksum_path=root / "CHECKSUMS.sha256",
        checksum_sha256="b" * 64,
        marker_path=root / ("_SUCCESS" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
        reason_codes=() if status == "COMPLETE" else ("PRIMARY_SESSION_NOT_COMPLETE",),
    )


@pytest.mark.parametrize(("status", "expected_exit"), [("COMPLETE", 0), ("INSUFFICIENT_DATA", 1)])
def test_reproduce_m8_l2_binds_all_four_explicit_session_authorities(
    status: str,
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_config = object()
    analysis_config = object()
    result = _fake_m8_l2_study_result(tmp_path, status)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: capture_config)
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: analysis_config)

    def fake_reproduce(*args: object) -> object:
        calls.append(args)
        return result

    monkeypatch.setattr(cli_module, "reproduce_m8_l2_study", fake_reproduce)

    assert (
        main(
            [
                "reproduce-m8-l2",
                *_m8_l2_study_cli_arguments(tmp_path, include_run_authority=False),
            ]
        )
        == expected_exit
    )
    assert len(calls) == 1
    call = calls[0]
    assert call[0:2] == (capture_config, analysis_config)
    for index, (position, role) in enumerate(
        zip((2, 3, 6, 7), ("train", "validation", "primary", "replication"), strict=True),
        start=1,
    ):
        authority = call[position]
        assert isinstance(authority, cli_module.L2StudySessionAuthority)
        assert authority.bundle_path == (tmp_path / role).absolute()
        assert authority.manifest_sha256 == format(index, "x") * 64
        assert authority.checksums_sha256 == format(index + 4, "x") * 64
    assert call[4] == (tmp_path / "development-lock").absolute()
    assert call[5] == "9" * 64
    assert call[8] == (tmp_path / "run").absolute()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert payload["run_manifest_sha256"] == "a" * 64
    assert payload["checksums_sha256"] == "b" * 64
    assert payload["reason_codes"] == list(result.reason_codes)
    assert payload["live_trading"] is False


@pytest.mark.parametrize(("status", "expected_exit"), [("COMPLETE", 0), ("INSUFFICIENT_DATA", 1)])
def test_verify_m8_l2_run_requires_and_forwards_terminal_run_authority(
    status: str,
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_config = object()
    analysis_config = object()
    result = _fake_m8_l2_study_result(tmp_path, status)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: capture_config)
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: analysis_config)

    def fake_verify(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(cli_module, "verify_m8_l2_study_run", fake_verify)

    assert (
        main(
            [
                "verify-m8-l2-run",
                *_m8_l2_study_cli_arguments(tmp_path, include_run_authority=True),
            ]
        )
        == expected_exit
    )
    assert calls[0][1] == {
        "expected_manifest_sha256": "a" * 64,
        "expected_checksums_sha256": "b" * 64,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["integrity"] == "verified"
    assert payload["status"] == status


@pytest.mark.parametrize(("status", "expected_exit"), [("COMPLETE", 0), ("INSUFFICIENT_DATA", 1)])
def test_report_m8_l2_reverifies_then_writes_only_to_external_output(
    status: str,
    expected_exit: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_config = object()
    analysis_config = object()
    result = _fake_m8_l2_study_result(tmp_path, status)
    report_data = object()
    output = (tmp_path / "reports").absolute()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(cli_module, "load_m8_l2_config", lambda path: capture_config)
    monkeypatch.setattr(cli_module, "load_m8_l2_analysis_config", lambda path: analysis_config)

    def fake_verify(*args: object, **kwargs: object) -> object:
        calls.append(("verify", args, kwargs))
        return result

    def fake_load(*args: object, **kwargs: object) -> object:
        calls.append(("load", args, kwargs))
        return report_data

    def fake_write(path: Path, data: object) -> tuple[Path, Path, Path]:
        assert path == output
        assert data is report_data
        return (
            path / "technical_report.md",
            path / "executive_memo.md",
            path / "model_comparison.md",
        )

    monkeypatch.setattr(cli_module, "verify_m8_l2_study_run", fake_verify)
    monkeypatch.setattr(cli_module, "load_m8_l2_report_data", fake_load)
    monkeypatch.setattr(cli_module, "write_l2_report_set", fake_write)
    monkeypatch.setattr(cli_module, "canonical_report_data_sha256", lambda data: "c" * 64)

    assert (
        main(
            [
                "report-m8-l2",
                *_m8_l2_study_cli_arguments(tmp_path, include_run_authority=True),
                "--output-dir",
                str(output),
            ]
        )
        == expected_exit
    )
    assert [call[0] for call in calls] == ["verify", "load"]
    assert calls[0][1:] == calls[1][1:]
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"] == str(output)
    assert payload["technical_report"] == str(output / "technical_report.md")
    assert payload["executive_memo"] == str(output / "executive_memo.md")
    assert payload["model_comparison"] == str(output / "model_comparison.md")
    assert payload["report_inputs_sha256"] == "c" * 64
    assert payload["source_bundle_modified"] is False


@pytest.mark.parametrize(
    ("command", "include_run_authority"),
    [
        ("reproduce-m8-l2", False),
        ("verify-m8-l2-run", True),
        ("report-m8-l2", True),
    ],
)
def test_m8_l2_final_commands_reject_noncanonical_session_authority_before_io(
    command: str,
    include_run_authority: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _m8_l2_study_cli_arguments(
        tmp_path,
        include_run_authority=include_run_authority,
    )
    arguments[arguments.index("--primary-manifest-sha256") + 1] = "A" * 64
    if command == "report-m8-l2":
        arguments.extend(["--output-dir", str(tmp_path / "reports")])
    monkeypatch.setattr(
        cli_module,
        "load_m8_l2_config",
        lambda path: pytest.fail("argument rejection must precede config I/O"),
    )

    with pytest.raises(SystemExit) as caught:
        main([command, *arguments])

    assert caught.value.code == 2


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
