from __future__ import annotations

import gc
import hashlib
import json
import shutil
import weakref
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest

import microstructure.m8_l2_pipeline as pipeline
import microstructure.research.l2_evaluation as l2_evaluation_module
import test_m8_l2_development as development_fixture
from microstructure.m8_l2_analysis_config import load_m8_l2_analysis_config
from microstructure.m8_l2_capture import M8L2SessionBundle
from microstructure.m8_l2_config import load_m8_l2_config
from microstructure.m8_l2_development import lock_m8_l2_development
from microstructure.m8_l2_inputs import L2CampaignRuntimeIdentity
from microstructure.reporting.l2 import L2ReportData

PROJECT_ROOT = Path(__file__).parents[1]
CAPTURE_CONFIG = PROJECT_ROOT / "configs" / "m8_l2_capture_study.toml"
ANALYSIS_CONFIG = PROJECT_ROOT / "configs" / "m8_l2_analysis.toml"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _authority(tmp_path: Path, name: str) -> pipeline.L2StudySessionAuthority:
    return pipeline.L2StudySessionAuthority(
        tmp_path / name,
        _sha(f"manifest:{name}"),
        _sha(f"checksums:{name}"),
    )


def _bundle(
    authority: pipeline.L2StudySessionAuthority,
    *,
    date: str,
    role: str,
    status: str = "COMPLETE",
    reasons: tuple[str, ...] = (),
) -> M8L2SessionBundle:
    return M8L2SessionBundle(
        root=authority.bundle_path,
        status=status,  # type: ignore[arg-type]
        session_id=_sha(f"session:{date}"),
        session_date=date,
        role=role,
        manifest_path=authority.bundle_path / "session_manifest.json",
        manifest_sha256=authority.manifest_sha256,
        checksum_path=authority.bundle_path / "CHECKSUMS.sha256",
        marker_path=authority.bundle_path
        / ("_SUCCESS" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
        reason_codes=reasons,
    )


def test_session_authority_requires_both_independent_digests(tmp_path: Path) -> None:
    value = _authority(tmp_path, "train")
    assert value.bundle_path.is_absolute()
    assert value.file_authority.manifest_sha256 == value.manifest_sha256
    assert value.to_dict()["checksums_sha256"] == value.checksums_sha256

    with pytest.raises(pipeline.M8L2StudyPipelineError, match="lowercase SHA-256"):
        pipeline.L2StudySessionAuthority(tmp_path / "bad", "not-a-sha", "0" * 64)


def test_atomic_stage_is_hidden_and_published_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "run"
    stage, parent = pipeline._reserve_stage(target)
    pipeline._write_bytes(stage / "payload", b"evidence\n")
    pipeline._write_bytes(stage / "_SUCCESS", b"complete\n")
    assert not target.exists()

    pipeline._publish_stage_no_overwrite(stage, target, parent)
    assert (target / "payload").read_bytes() == b"evidence\n"
    assert (target / "_SUCCESS").read_bytes() == b"complete\n"

    with pytest.raises(pipeline.M8L2StudyPipelineError, match="already exists"):
        pipeline._reserve_stage(target)


def test_atomic_rename_exclusively_rejects_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "run"
    stage, parent = pipeline._reserve_stage(target)
    pipeline._write_bytes(stage / "payload", b"candidate\n")
    original = pipeline._atomic_rename_no_replace

    def racing_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner").write_text("other\n", encoding="utf-8")
        original(source, destination)

    monkeypatch.setattr(pipeline, "_atomic_rename_no_replace", racing_rename)
    with pytest.raises(pipeline.M8L2StudyPipelineError, match="appeared"):
        pipeline._publish_stage_no_overwrite(stage, target, parent)
    assert (target / "owner").read_text(encoding="utf-8") == "other\n"
    assert (stage / "payload").read_bytes() == b"candidate\n"
    shutil.rmtree(stage)


def test_stage_rejects_symlinked_publication_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(pipeline.M8L2StudyPipelineError, match="symlink component"):
        pipeline._reserve_stage(linked / "run")


def test_report_snapshot_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    manifest = {
        "evidence_tier": "FULL_DATA",
        "effective_evidence_tier": "INSUFFICIENT_DATA",
        "status": "INSUFFICIENT_DATA",
        "live_trading": False,
        "research": {
            "question": "Does the frozen model improve direction log loss?",
            "period_start_utc": "2026-08-10T14:00:00Z",
            "period_end_utc": "2026-08-13T15:00:00Z",
        },
    }
    provenance = {
        "git": {"commit": "a" * 40, "source_tree_sha256": "b" * 64, "dirty": False},
        "inputs": {
            "capture_config_sha256": "c" * 64,
            "capture_protocol_sha256": "d" * 64,
            "analysis_config_sha256": "e" * 64,
            "development_lock_sha256": "f" * 64,
        },
    }
    data = L2ReportData(
        manifest=manifest,
        provenance=provenance,
        session_gates=(),
        hypothesis={
            "conclusion": "Insufficient evidence for deployment.",
            "directionally_replicated_pairs": 0,
        },
        predictive_metrics=(),
        paired_metrics=(),
        equal_session_metrics=(),
        execution_metrics=(),
    )
    pipeline._write_report_artifacts(tmp_path, data)
    restored = pipeline._load_report_data_snapshot(tmp_path)
    assert restored.manifest == manifest
    assert restored.hypothesis == data.hypothesis

    sidecar = tmp_path / "report_inputs.sha256"
    sidecar.write_text(f"{'0' * 64}  report_inputs.json\n", encoding="ascii")
    with pytest.raises(pipeline.M8L2StudyRunVerificationError, match="sidecar differs"):
        pipeline._load_report_data_snapshot(tmp_path)


def test_pipeline_memory_boundary_and_execution_projection() -> None:
    payload: dict[str, list[object]] = {
        name: [1, 2, 3] for name in pipeline._EXECUTION_PREDICTION_COLUMNS
    }
    for name in (
        "sample_id",
        "symbol",
        "study_date",
        "study_role",
        "endpoint_name",
        "split",
        "child_lock_sha256",
        "aggregate_lock_sha256",
    ):
        payload[name] = [f"{name}-{index}" for index in range(3)]
    payload["selected_probability"] = [0.1, 0.5, 0.9]
    payload["is_oos"] = [True, True, True]
    payload["unused_wide_payload"] = ["x" * 10_000] * 3
    frame = pl.DataFrame(payload)
    projected = pipeline._project_frame(
        frame,
        pipeline._EXECUTION_PREDICTION_COLUMNS,
        "test execution predictions",
    )
    assert projected.columns == list(pipeline._EXECUTION_PREDICTION_COLUMNS)
    assert projected.estimated_size("b") < frame.estimated_size("b")
    observed = int(projected.estimated_size("b"))
    pipeline._require_memory_budget(observed, observed, "inclusive boundary")
    with pytest.raises(pipeline.M8L2StudyPipelineError, match="memory budget"):
        pipeline._require_memory_budget(observed + 1, observed, "overflow")

    event_payload: dict[str, list[object]] = {
        name: [1, 2, 3] for name in pipeline._EXECUTION_EVENT_COLUMNS
    }
    events = pl.DataFrame(event_payload)
    execution_upper = pipeline._execution_workspace_upper_bytes(events, projected)
    signal_rows = 2
    ledger_rows = signal_rows + 1
    projected_inputs = pipeline._projected_frame_bytes(
        events, pipeline._EXECUTION_EVENT_COLUMNS, "events"
    ) + pipeline._projected_frame_bytes(
        projected, pipeline._EXECUTION_PREDICTION_COLUMNS, "predictions"
    )
    python_inputs = pipeline._python_projected_rows_upper_bytes(
        events, pipeline._EXECUTION_EVENT_COLUMNS
    ) + pipeline._python_projected_rows_upper_bytes(
        projected, pipeline._EXECUTION_PREDICTION_COLUMNS
    )
    assert execution_upper == (
        3 * projected_inputs
        + python_inputs
        + (4 * ledger_rows + events.height) * pipeline._PYTHON_LEDGER_ROW_UPPER_BYTES
        + 3 * ledger_rows * 9 * pipeline._POLARS_LEDGER_ROW_UPPER_BYTES
    )


def test_execution_ratio_verifier_requires_nullable_finite_unit_interval() -> None:
    valid = pl.DataFrame(
        {
            "fill_ratio": [0.0, 1.0, None],
            "fill_ratio_requested": [0.25, None, 0.75],
            "partial_fill_order_ratio": [None, 0.5, 1.0],
        }
    )
    pipeline._require_nullable_unit_interval_columns(
        valid,
        ("fill_ratio", "fill_ratio_requested", "partial_fill_order_ratio"),
        "execution metrics",
    )
    with pytest.raises(pipeline.M8L2StudyRunVerificationError, match="lacks ratio columns"):
        pipeline._require_nullable_unit_interval_columns(
            valid.drop("fill_ratio_requested"),
            ("fill_ratio", "fill_ratio_requested", "partial_fill_order_ratio"),
            "execution metrics",
        )
    for invalid in (float("nan"), float("inf"), -0.01, 1.01):
        tampered = valid.with_columns(pl.lit(invalid).alias("fill_ratio"))
        with pytest.raises(
            pipeline.M8L2StudyRunVerificationError,
            match=r"null or finite in \[0, 1\]",
        ):
            pipeline._require_nullable_unit_interval_columns(
                tampered,
                ("fill_ratio", "fill_ratio_requested", "partial_fill_order_ratio"),
                "execution metrics",
            )


def test_streaming_verifier_does_not_accumulate_partition_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = load_m8_l2_config(CAPTURE_CONFIG)
    analysis = load_m8_l2_analysis_config(ANALYSIS_CONFIG)
    causal_keys = tuple(
        (session.date.isoformat(), session.role, symbol, endpoint.name)
        for session in capture.sessions
        for symbol in capture.study.symbols
        for endpoint in analysis.endpoints
    )
    claims: dict[str, dict[str, object]] = {
        pipeline._causal_relative(key): {} for key in causal_keys
    }
    for date, role in pipeline._EXPECTED_COORDINATES[2:]:
        for symbol in capture.study.symbols:
            for endpoint in analysis.endpoints:
                for family in ("orders", "fills", "positions"):
                    relative = (
                        f"execution/partitions/{date}-{role}/{symbol.lower()}/"
                        f"{endpoint.name}/{family}.parquet"
                    )
                    claims[relative] = {}

    references: list[weakref.ReferenceType[pl.DataFrame]] = []
    maximum_live = 0

    def fake_read(root: Path, relative: str, claim: Any) -> pl.DataFrame:
        nonlocal maximum_live
        del root, claim
        gc.collect()
        maximum_live = max(maximum_live, sum(reference() is not None for reference in references))
        if relative.startswith("causal_frames/"):
            frame = pl.DataFrame(
                {"feature_ready": [True], "right_censored": [False], "future_mid_up": [1]}
            )
        else:
            frame = pl.DataFrame({"partition": [relative]})
        references.append(weakref.ref(frame))
        return frame

    monkeypatch.setattr(pipeline, "_read_claimed_parquet", fake_read)
    monkeypatch.setattr(pipeline, "_verify_one_causal_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_verify_partition_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_frame_sha256", lambda frame: "1" * 64)
    material = SimpleNamespace(
        development_frame_sha256={
            (symbol, endpoint.name): "1" * 64
            for symbol in capture.study.symbols
            for endpoint in analysis.endpoints
        },
        result=SimpleNamespace(aggregate_sha256="2" * 64),
    )
    retained = pipeline._verify_tabular_outputs_streaming(
        tmp_path,
        claims,
        capture=capture,
        analysis=analysis,
        material=material,
        status="COMPLETE",
        reasons=(),
    )
    assert retained == {}
    assert maximum_live <= 2


def test_verified_heldout_failure_never_opens_economic_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = load_m8_l2_config(CAPTURE_CONFIG)
    analysis = load_m8_l2_analysis_config(ANALYSIS_CONFIG)
    authorities = tuple(_authority(tmp_path, role) for _, role in pipeline._EXPECTED_COORDINATES)
    campaign = L2CampaignRuntimeIdentity("a" * 64, "b" * 40, "c" * 64, "d" * 64, False)
    snapshots = tuple(
        pipeline._SessionSnapshot(
            authority,
            _bundle(
                authority,
                date=date,
                role=role,
                status="INSUFFICIENT_DATA" if role == "primary_test" else "COMPLETE",
                reasons=("COVERAGE_GATE",) if role == "primary_test" else (),
            ),
            {"symbols": {}, "cross_symbol_observed_overlap_seconds": 0.0},
            campaign,
        )
        for authority, (date, role) in zip(authorities, pipeline._EXPECTED_COORDINATES, strict=True)
    )
    material = object()
    monkeypatch.setattr(
        pipeline,
        "_verify_lock_context",
        lambda *args, **kwargs: (object(), {}, campaign, object()),
    )
    monkeypatch.setattr(pipeline, "_load_lock_material", lambda *args, **kwargs: material)
    monkeypatch.setattr(pipeline, "_verify_all_sessions", lambda *args, **kwargs: snapshots)
    monkeypatch.setattr(
        pipeline,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: "d" * 64,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("held-out economic frames were opened after a gate failure")

    monkeypatch.setattr(pipeline, "_build_causal_frames", forbidden)
    expected = pipeline.M8L2StudyRunResult(
        root=tmp_path / "run",
        status="INSUFFICIENT_DATA",
        manifest_path=tmp_path / "run" / "run_manifest.json",
        manifest_sha256="d" * 64,
        checksum_path=tmp_path / "run" / "CHECKSUMS.sha256",
        checksum_sha256="e" * 64,
        marker_path=tmp_path / "run" / "INSUFFICIENT_DATA",
        reason_codes=("primary_test::COVERAGE_GATE",),
    )

    def fake_publish(**kwargs: Any) -> pipeline.M8L2StudyRunResult:
        assert kwargs["status"] == "INSUFFICIENT_DATA"
        assert kwargs["reason_codes"] == ("primary_test::COVERAGE_GATE",)
        assert kwargs["causal"] == {}
        assert kwargs["heldout"] == ()
        return expected

    monkeypatch.setattr(pipeline, "_publish_run", fake_publish)
    result = pipeline.reproduce_m8_l2_study(
        capture,
        analysis,
        authorities[0],
        authorities[1],
        tmp_path / "lock",
        "f" * 64,
        authorities[2],
        authorities[3],
        tmp_path / "run",
    )
    assert result == expected


def test_not_created_development_publishes_and_verifies_four_session_terminal_without_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = development_fixture._environment(tmp_path, monkeypatch)
    campaign_raw = b'{"artifact_kind":"test-campaign-authority"}\n'
    campaign_sha = hashlib.sha256(campaign_raw).hexdigest()

    def session_bundle(
        path: Path,
        *,
        date: str,
        role: str,
        status: str,
        reasons: tuple[str, ...] = (),
    ) -> M8L2SessionBundle:
        authority_dir = path / "authority"
        authority_dir.mkdir(exist_ok=True)
        (authority_dir / "campaign_authority.json").write_bytes(campaign_raw)
        manifest_payload = {
            "authority": {
                "campaign_authority_sha256": campaign_sha,
                "runtime_commit": development_fixture.COMMIT,
                "runtime_source_tree_sha256": development_fixture.SOURCE_TREE,
                "runtime_fingerprint_sha256": development_fixture.RUNTIME_SHA,
                "runtime_dirty": False,
            },
            "symbols": {},
            "cross_symbol_observed_overlap_seconds": 0.0,
        }
        manifest = path / "session_manifest.json"
        manifest.write_text(
            json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        checksums = path / "CHECKSUMS.sha256"
        checksums.write_text("session control authority\n", encoding="ascii")
        return M8L2SessionBundle(
            root=path.absolute(),
            status=cast(Any, status),
            session_id=_sha(f"session:{date}"),
            session_date=date,
            role=role,
            manifest_path=manifest,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            checksum_path=checksums,
            marker_path=path / ("_SUCCESS" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
            reason_codes=reasons,
        )

    train = session_bundle(
        environment.train_path,
        date="2026-08-10",
        role="train",
        status="INSUFFICIENT_DATA",
        reasons=("GATE_COVERAGE",),
    )
    validation = session_bundle(
        environment.validation_path,
        date="2026-08-11",
        role="validation",
        status="COMPLETE",
    )
    primary_path = train.root.parent / "2026-08-12-primary"
    replication_path = train.root.parent / "2026-08-13-replication"
    primary_path.mkdir()
    replication_path.mkdir()
    primary = session_bundle(
        primary_path,
        date="2026-08-12",
        role="primary_test",
        status="COMPLETE",
    )
    replication = session_bundle(
        replication_path,
        date="2026-08-13",
        role="replication_test",
        status="COMPLETE",
    )
    bundles = {item.root: item for item in (train, validation, primary, replication)}

    def fake_session_verify(path: str | Path, *, expected_config: Any) -> M8L2SessionBundle:
        assert expected_config == environment.capture
        return bundles[Path(path).absolute()]

    monkeypatch.setattr(
        development_fixture.development,
        "verify_m8_l2_session_bundle",
        fake_session_verify,
    )
    monkeypatch.setattr(pipeline, "verify_m8_l2_session_bundle", fake_session_verify)
    monkeypatch.setattr(
        pipeline,
        "_current_source_identity",
        lambda capture: pipeline._SourceIdentity(
            development_fixture.COMMIT,
            development_fixture.SOURCE_TREE,
            False,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: development_fixture.RUNTIME_SHA,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("NOT_CREATED finalization opened economic rows")

    monkeypatch.setattr(pipeline, "_development_input", forbidden)
    monkeypatch.setattr(pipeline, "_heldout_input", forbidden)
    monkeypatch.setattr(pipeline, "_build_causal_frames", forbidden)
    development_authority = lock_m8_l2_development(
        environment.capture,
        environment.analysis,
        train.root,
        validation.root,
        tmp_path / "development-authority",
    )
    assert development_authority.status == "NOT_CREATED"
    authorities = tuple(
        pipeline.L2StudySessionAuthority(
            bundle.root,
            bundle.manifest_sha256,
            hashlib.sha256(bundle.checksum_path.read_bytes()).hexdigest(),
        )
        for bundle in (train, validation, primary, replication)
    )
    run_dir = tmp_path / "final-run"
    result = pipeline.reproduce_m8_l2_study(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        development_authority.root,
        development_authority.aggregate_sha256,
        authorities[2],
        authorities[3],
        run_dir,
    )
    assert result.status == "INSUFFICIENT_DATA"
    assert result.reason_codes == ("DEVELOPMENT_SESSION_INSUFFICIENT::train::GATE_COVERAGE",)
    assert not list(run_dir.rglob("*.parquet"))
    manifest = json.loads(result.manifest_path.read_text(encoding="ascii"))
    assert manifest["authority"]["development_authority"] == {
        "status": "NOT_CREATED",
        "authority_sha256": development_authority.aggregate_sha256,
        "reason_codes": list(development_authority.reason_codes),
    }
    assert manifest["tabular_outputs"] == []

    verified = pipeline.verify_m8_l2_study_run(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        development_authority.root,
        development_authority.aggregate_sha256,
        authorities[2],
        authorities[3],
        run_dir,
        expected_manifest_sha256=result.manifest_sha256,
        expected_checksums_sha256=result.checksum_sha256,
    )
    assert verified == result


def test_final_producer_rejects_runtime_drift_before_material_or_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = load_m8_l2_config(CAPTURE_CONFIG)
    analysis = load_m8_l2_analysis_config(ANALYSIS_CONFIG)
    authorities = tuple(_authority(tmp_path, role) for _, role in pipeline._EXPECTED_COORDINATES)
    campaign = L2CampaignRuntimeIdentity("a" * 64, "b" * 40, "c" * 64, "d" * 64, False)
    monkeypatch.setattr(
        pipeline,
        "_verify_lock_context",
        lambda *args, **kwargs: (object(), {}, campaign, object()),
    )
    monkeypatch.setattr(
        pipeline,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: "e" * 64,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("runtime drift reached lock material or session rows")

    monkeypatch.setattr(pipeline, "_load_lock_material", forbidden)
    monkeypatch.setattr(pipeline, "_verify_all_sessions", forbidden)
    with pytest.raises(pipeline.M8L2StudyPipelineError, match="runtime differs"):
        pipeline.reproduce_m8_l2_study(
            capture,
            analysis,
            authorities[0],
            authorities[1],
            tmp_path / "lock",
            "f" * 64,
            authorities[2],
            authorities[3],
            tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


def test_final_producer_rejects_foreign_origin_before_lock_or_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = load_m8_l2_config(CAPTURE_CONFIG)
    analysis = load_m8_l2_analysis_config(ANALYSIS_CONFIG)
    authorities = tuple(_authority(tmp_path, role) for _, role in pipeline._EXPECTED_COORDINATES)
    foreign = tmp_path / "foreign" / "research" / "l2_evaluation.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# foreign checkout\n", encoding="utf-8")
    lock_calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal lock_calls
        lock_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(l2_evaluation_module, "__file__", str(foreign))
    monkeypatch.setattr(pipeline, "_verify_lock_context", forbidden)
    with pytest.raises(pipeline.M8L2StudyPipelineError, match="foreign or mixed import origin"):
        pipeline.reproduce_m8_l2_study(
            capture,
            analysis,
            authorities[0],
            authorities[1],
            tmp_path / "lock",
            "f" * 64,
            authorities[2],
            authorities[3],
            tmp_path / "run",
        )

    assert lock_calls == 0
    assert not (tmp_path / "run").exists()


def test_complete_producer_verifier_reuse_and_report_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = development_fixture._environment(tmp_path, monkeypatch)
    lock = lock_m8_l2_development(
        environment.capture,
        environment.analysis,
        environment.train_path,
        environment.validation_path,
        tmp_path / "development-lock",
    )
    environment.result = lock
    campaign = environment.train_input.campaign_identity
    authorities = (
        pipeline.L2StudySessionAuthority(
            environment.train_path,
            environment.train_input.file_authority.manifest_sha256,
            environment.train_input.file_authority.checksums_sha256,
        ),
        pipeline.L2StudySessionAuthority(
            environment.validation_path,
            environment.validation_input.file_authority.manifest_sha256,
            environment.validation_input.file_authority.checksums_sha256,
        ),
        _authority(tmp_path, "primary_test"),
        _authority(tmp_path, "replication_test"),
    )
    symbols = MappingProxyType({symbol: object() for symbol in environment.capture.study.symbols})
    primary_input = development_fixture._FakeInput(
        root=authorities[2].bundle_path,
        session_id=_sha("2026-08-12"),
        session_date="2026-08-12",
        role="primary_test",
        config_sha256=environment.capture.hash,
        config_source_sha256=environment.capture.source_sha256,
        file_authority=authorities[2].file_authority,
        campaign_identity=campaign,
        symbols=symbols,
        frames={
            symbol: development_fixture._frames(symbol, "2026-08-12")
            for symbol in environment.capture.study.symbols
        },
        events=environment.events,
        access_phase="heldout_after_lock",
        development_lock_sha256=lock.aggregate_sha256,
    )
    replication_input = development_fixture._FakeInput(
        root=authorities[3].bundle_path,
        session_id=_sha("2026-08-13"),
        session_date="2026-08-13",
        role="replication_test",
        config_sha256=environment.capture.hash,
        config_source_sha256=environment.capture.source_sha256,
        file_authority=authorities[3].file_authority,
        campaign_identity=campaign,
        symbols=symbols,
        frames={
            symbol: development_fixture._frames(symbol, "2026-08-13")
            for symbol in environment.capture.study.symbols
        },
        events=environment.events,
        access_phase="heldout_after_lock",
        development_lock_sha256=lock.aggregate_sha256,
    )
    verified_inputs = {
        "train": environment.train_input,
        "validation": environment.validation_input,
        "primary_test": primary_input,
        "replication_test": replication_input,
    }
    snapshots = tuple(
        pipeline._SessionSnapshot(
            authority,
            _bundle(authority, date=date, role=role),
            {
                "symbols": {
                    symbol: {"status": "COMPLETE"} for symbol in environment.capture.study.symbols
                },
                "cross_symbol_observed_overlap_seconds": 30.0,
            },
            campaign,
        )
        for authority, (date, role) in zip(authorities, pipeline._EXPECTED_COORDINATES, strict=True)
    )

    monkeypatch.setattr(
        pipeline,
        "_current_source_identity",
        lambda capture: pipeline._SourceIdentity(
            development_fixture.COMMIT,
            development_fixture.SOURCE_TREE,
            False,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: development_fixture.RUNTIME_SHA,
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_all_sessions",
        lambda capture, material, supplied: snapshots,
    )
    monkeypatch.setattr(
        pipeline,
        "_development_input",
        lambda snapshot, **kwargs: verified_inputs[snapshot.bundle.role],
    )
    monkeypatch.setattr(
        pipeline,
        "_heldout_input",
        lambda snapshot, **kwargs: verified_inputs[snapshot.bundle.role],
    )

    def authority_sources(
        capture: Any, analysis: Any, supplied: Any, material: Any
    ) -> dict[str, tuple[Path, str]]:
        del supplied
        protocol = PROJECT_ROOT / "docs" / "M8_L2_PROTOCOL.md"
        sources = {
            "authority/m8_l2_capture_study.toml": (capture.path, capture.source_sha256),
            "authority/m8_l2_analysis.toml": (analysis.path, analysis.source_sha256),
            "authority/M8_L2_PROTOCOL.md": (
                protocol,
                pipeline.M8_L2_PROTOCOL_SHA256,
            ),
        }
        for source in material.snapshot_files:
            relative = source.relative_to(material.result.root).as_posix()
            sources[f"authority/development_lock/{relative}"] = (
                source,
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        return dict(sorted(sources.items()))

    monkeypatch.setattr(pipeline, "_authority_sources", authority_sources)
    publication_events: list[str] = []
    producer_terminal_revalidations = 0
    original_terminal_revalidation = pipeline._terminal_revalidation
    original_publish_stage = pipeline._publish_stage_no_overwrite

    def observed_terminal_revalidation(**kwargs: Any) -> None:
        nonlocal producer_terminal_revalidations
        if kwargs.get("require_current_runtime") is True:
            producer_terminal_revalidations += 1
        publication_events.append("revalidate")
        original_terminal_revalidation(**kwargs)

    def observed_publish_stage(stage: Path, target: Path, parent: pipeline._ParentIdentity) -> None:
        assert publication_events[-2:] == ["revalidate", "revalidate"]
        publication_events.append("publish")
        original_publish_stage(stage, target, parent)

    monkeypatch.setattr(pipeline, "_terminal_revalidation", observed_terminal_revalidation)
    monkeypatch.setattr(pipeline, "_publish_stage_no_overwrite", observed_publish_stage)
    run_dir = tmp_path / "final-run"
    result = pipeline.reproduce_m8_l2_study(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        lock.root,
        lock.aggregate_sha256,
        authorities[2],
        authorities[3],
        run_dir,
    )
    assert result.status == "COMPLETE"
    assert result.marker_path.read_bytes() == b"complete\n"
    assert len(list((run_dir / "causal_frames").rglob("*.parquet"))) == 32
    assert len(list((run_dir / "execution" / "partitions").rglob("*.parquet"))) == 48
    manifest = json.loads(result.manifest_path.read_text(encoding="ascii"))
    assert manifest["evaluation"]["model_refit_after_development_lock"] is False
    assert manifest["execution"]["realized_execution"] is False
    assert producer_terminal_revalidations == 2

    memory_failure_dir = tmp_path / "memory-failure-run"
    with monkeypatch.context() as bounded:
        bounded.setattr(pipeline, "_MAX_FINAL_RAW_BYTES", 1)
        with pytest.raises(
            pipeline.M8L2StudyPipelineError,
            match=r"raw materialization.*memory budget",
        ):
            pipeline.reproduce_m8_l2_study(
                environment.capture,
                environment.analysis,
                authorities[0],
                authorities[1],
                lock.root,
                lock.aggregate_sha256,
                authorities[2],
                authorities[3],
                memory_failure_dir,
            )
    assert not memory_failure_dir.exists()
    assert not (memory_failure_dir / "_SUCCESS").exists()
    assert not (memory_failure_dir / "INSUFFICIENT_DATA").exists()

    with pytest.raises(
        pipeline.M8L2StudyPipelineError,
        match="requires caller-held manifest and checksum authorities",
    ):
        pipeline.reproduce_m8_l2_study(
            environment.capture,
            environment.analysis,
            authorities[0],
            authorities[1],
            lock.root,
            lock.aggregate_sha256,
            authorities[2],
            authorities[3],
            run_dir,
        )
    reused = pipeline.reproduce_m8_l2_study(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        lock.root,
        lock.aggregate_sha256,
        authorities[2],
        authorities[3],
        run_dir,
        expected_existing_manifest_sha256=result.manifest_sha256,
        expected_existing_checksums_sha256=result.checksum_sha256,
    )
    assert reused == result

    def forbidden_current_origin(*args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs, "offline verifier inspected current import origin"))

    current_origin = pipeline._assert_final_producer_import_origins
    monkeypatch.setattr(pipeline, "_assert_final_producer_import_origins", forbidden_current_origin)
    report_data = pipeline.load_m8_l2_report_data(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        lock.root,
        lock.aggregate_sha256,
        authorities[2],
        authorities[3],
        run_dir,
        expected_manifest_sha256=result.manifest_sha256,
        expected_checksums_sha256=result.checksum_sha256,
    )
    assert report_data.manifest["status"] == "COMPLETE"
    assert len(report_data.execution_metrics) == 144
    monkeypatch.setattr(pipeline, "_assert_final_producer_import_origins", current_origin)

    original_report = result.technical_report_path.read_bytes()
    mutated_at_return_boundary = False

    def mutate_report_after_semantic_verification(**kwargs: Any) -> None:
        nonlocal mutated_at_return_boundary
        original_terminal_revalidation(**kwargs)
        if kwargs.get("require_current_runtime") is not True:
            result.technical_report_path.write_text("return-boundary drift\n", encoding="utf-8")
            mutated_at_return_boundary = True

    with monkeypatch.context() as return_boundary:
        return_boundary.setattr(
            pipeline,
            "_terminal_revalidation",
            mutate_report_after_semantic_verification,
        )
        with pytest.raises(
            pipeline.M8L2StudyRunVerificationError,
            match="changed after semantic verification",
        ):
            pipeline.verify_m8_l2_study_run(
                environment.capture,
                environment.analysis,
                authorities[0],
                authorities[1],
                lock.root,
                lock.aggregate_sha256,
                authorities[2],
                authorities[3],
                run_dir,
                expected_manifest_sha256=result.manifest_sha256,
                expected_checksums_sha256=result.checksum_sha256,
            )
    assert mutated_at_return_boundary
    result.technical_report_path.write_bytes(original_report)

    result.technical_report_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(pipeline.M8L2StudyPipelineError, match="checksum mismatch"):
        pipeline.verify_m8_l2_study_run(
            environment.capture,
            environment.analysis,
            authorities[0],
            authorities[1],
            lock.root,
            lock.aggregate_sha256,
            authorities[2],
            authorities[3],
            run_dir,
            expected_manifest_sha256=result.manifest_sha256,
            expected_checksums_sha256=result.checksum_sha256,
        )

    insufficient_snapshots = (
        snapshots[0],
        snapshots[1],
        pipeline._SessionSnapshot(
            authorities[2],
            _bundle(
                authorities[2],
                date="2026-08-12",
                role="primary_test",
                status="INSUFFICIENT_DATA",
                reasons=("COVERAGE_GATE",),
            ),
            {
                "symbols": {
                    symbol: {"status": "FAILED"} for symbol in environment.capture.study.symbols
                },
                "cross_symbol_observed_overlap_seconds": 10.0,
            },
            campaign,
        ),
        snapshots[3],
    )
    monkeypatch.setattr(
        pipeline,
        "_verify_all_sessions",
        lambda capture, material, supplied: insufficient_snapshots,
    )
    insufficient_dir = tmp_path / "insufficient-run"
    insufficient = pipeline.reproduce_m8_l2_study(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        lock.root,
        lock.aggregate_sha256,
        authorities[2],
        authorities[3],
        insufficient_dir,
    )
    assert insufficient.status == "INSUFFICIENT_DATA"
    assert insufficient.marker_path.read_bytes() == b"terminal\n"
    assert insufficient.reason_codes == ("primary_test::COVERAGE_GATE",)
    assert not (insufficient_dir / "causal_frames").exists()
    assert not (insufficient_dir / "evaluation").exists()
    insufficient_reports = pipeline.load_m8_l2_report_data(
        environment.capture,
        environment.analysis,
        authorities[0],
        authorities[1],
        lock.root,
        lock.aggregate_sha256,
        authorities[2],
        authorities[3],
        insufficient_dir,
        expected_manifest_sha256=insufficient.manifest_sha256,
        expected_checksums_sha256=insufficient.checksum_sha256,
    )
    assert insufficient_reports.manifest["status"] == "INSUFFICIENT_DATA"
    assert insufficient_reports.execution_metrics == ()
