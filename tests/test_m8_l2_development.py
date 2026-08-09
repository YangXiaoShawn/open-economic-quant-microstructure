from __future__ import annotations

import inspect
import json
import math
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import polars as pl
import pytest

import microstructure.m8_l2_development as development
import microstructure.research.l2_multidate as l2_multidate_module
from microstructure.m8_l2_analysis_config import load_m8_l2_analysis_config
from microstructure.m8_l2_capture import M8L2SessionBundle
from microstructure.m8_l2_config import load_m8_l2_config
from microstructure.m8_l2_development import (
    M8L2DevelopmentError,
    ProducerSourceIdentity,
    lock_m8_l2_development,
    verify_m8_l2_development_lock,
)
from microstructure.m8_l2_inputs import (
    L2CampaignRuntimeIdentity,
    L2SessionFileAuthority,
)
from microstructure.research.l2_multidate import L2ObservedInterval

SECOND = 1_000_000_000
MILLISECOND = 1_000_000
COMMIT = "a" * 40
SOURCE_TREE = "b" * 64
CAMPAIGN_SHA = "c" * 64
RUNTIME_SHA = "d" * 64


@dataclass(frozen=True)
class _Loaded:
    book_observations: pl.DataFrame
    depth_deltas: pl.DataFrame
    intervals: tuple[L2ObservedInterval, ...]


@dataclass
class _FakeInput:
    root: Path
    session_id: str
    session_date: str
    role: str
    config_sha256: str
    config_source_sha256: str
    file_authority: L2SessionFileAuthority
    campaign_identity: L2CampaignRuntimeIdentity
    symbols: MappingProxyType[str, object]
    frames: dict[str, _Loaded]
    events: list[str]
    access_phase: str = "development"
    development_lock_sha256: str | None = None

    def load_symbol_frames(self, symbol: str) -> _Loaded:
        self.events.append(f"load:{self.session_date}:{symbol}")
        return self.frames[symbol]


@dataclass
class _Environment:
    capture: Any
    analysis: Any
    train_path: Path
    validation_path: Path
    train_input: _FakeInput
    validation_input: _FakeInput
    events: list[str]
    result: Any | None = None


def _source() -> ProducerSourceIdentity:
    return ProducerSourceIdentity(COMMIT, SOURCE_TREE, False)


def _frames(symbol: str, study_date: str) -> _Loaded:
    start = int(datetime.fromisoformat(f"{study_date}T14:00:00+00:00").timestamp()) * SECOND
    count = 280
    times = [start + index * 100 * MILLISECOND for index in range(count)]
    sequence = list(range(1, count + 1))
    mids = [
        100.0 + 0.025 * math.sin(index / 5.0) + 0.008 * math.sin(index / 17.0) for index in sequence
    ]
    bid_quantity = [3.0 + (index % 13) * 0.05 for index in sequence]
    ask_quantity = [2.5 + (index % 11) * 0.05 for index in sequence]
    continuity = f"capture:{symbol}:{study_date}"
    books = pl.DataFrame(
        {
            "venue": ["binance_spot"] * count,
            "symbol": [symbol] * count,
            "event_ts_ns": times,
            "available_ts_ns": times,
            "continuity_id": [continuity] * count,
            "sequence_end": sequence,
            "is_valid": [True] * count,
            "best_bid": [value - 0.01 for value in mids],
            "best_ask": [value + 0.01 for value in mids],
            "bid_quantity": bid_quantity,
            "ask_quantity": ask_quantity,
            "depth_bid_5": [value + 8.0 for value in bid_quantity],
            "depth_ask_5": [value + 7.0 for value in ask_quantity],
            "depth_bid_10": [value + 18.0 for value in bid_quantity],
            "depth_ask_10": [value + 17.0 for value in ask_quantity],
            "tick_size": [0.01] * count,
            "lot_size": [0.00001] * count,
        }
    )
    deltas = pl.DataFrame(
        {
            "venue": ["binance_spot"] * count,
            "symbol": [symbol] * count,
            "event_ts_ns": times,
            "available_ts_ns": times,
            "continuity_id": [continuity] * count,
            "first_update_id": sequence,
            "last_update_id": sequence,
            "bids": [
                [
                    {
                        "price_ticks": 10_000 + index,
                        "quantity_lots": 0 if index % 13 == 0 else 10,
                    }
                ]
                for index in sequence
            ],
            "asks": [[{"price_ticks": 10_002 + index, "quantity_lots": 10}] for index in sequence],
        }
    )
    return _Loaded(
        books,
        deltas,
        (L2ObservedInterval(continuity, start, start + 30 * SECOND),),
    )


def _bundle(root: Path, *, date: str, role: str, manifest_sha: str) -> M8L2SessionBundle:
    return M8L2SessionBundle(
        root=root,
        status="COMPLETE",
        session_id=hashlib_sha(date),
        session_date=date,
        role=role,
        manifest_path=root / "session_manifest.json",
        manifest_sha256=manifest_sha,
        checksum_path=root / "CHECKSUMS.sha256",
        marker_path=root / "_SUCCESS",
        reason_codes=(),
    )


def _terminal_bundle(
    root: Path,
    *,
    date: str,
    role: str,
    status: str,
    reasons: tuple[str, ...],
) -> M8L2SessionBundle:
    campaign = {
        "campaign_authority_sha256": CAMPAIGN_SHA,
        "runtime_commit": COMMIT,
        "runtime_source_tree_sha256": SOURCE_TREE,
        "runtime_fingerprint_sha256": RUNTIME_SHA,
        "runtime_dirty": False,
    }
    manifest = root / "session_manifest.json"
    manifest.write_bytes(_canonical({"authority": campaign}))
    checksum = root / "CHECKSUMS.sha256"
    checksum.write_text("session authority\n", encoding="ascii")
    return M8L2SessionBundle(
        root=root,
        status=cast(Any, status),
        session_id=hashlib_sha(date),
        session_date=date,
        role=role,
        manifest_path=manifest,
        manifest_sha256=hashlib_sha_bytes(manifest.read_bytes()),
        checksum_path=checksum,
        marker_path=root / ("_SUCCESS" if status == "COMPLETE" else "INSUFFICIENT_DATA"),
        reason_codes=reasons,
    )


def hashlib_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Environment:
    capture = load_m8_l2_config("configs/m8_l2_capture_study.toml")
    analysis = load_m8_l2_analysis_config("configs/m8_l2_analysis.toml")
    sessions = tmp_path / "sessions"
    train_path = sessions / f"2026-08-10-train-{hashlib_sha('train')[:20]}"
    validation_path = sessions / f"2026-08-11-validation-{hashlib_sha('validation')[:20]}"
    train_path.mkdir(parents=True)
    validation_path.mkdir()
    train_manifest = "1" * 64
    validation_manifest = "2" * 64
    campaign = L2CampaignRuntimeIdentity(CAMPAIGN_SHA, COMMIT, SOURCE_TREE, RUNTIME_SHA, False)
    events: list[str] = []
    symbols = MappingProxyType({symbol: object() for symbol in capture.study.symbols})
    train_input = _FakeInput(
        root=train_path.absolute(),
        session_id=hashlib_sha("2026-08-10"),
        session_date="2026-08-10",
        role="train",
        config_sha256=capture.hash,
        config_source_sha256=capture.source_sha256,
        file_authority=L2SessionFileAuthority(train_manifest, "3" * 64),
        campaign_identity=campaign,
        symbols=symbols,
        frames={symbol: _frames(symbol, "2026-08-10") for symbol in capture.study.symbols},
        events=events,
    )
    validation_input = _FakeInput(
        root=validation_path.absolute(),
        session_id=hashlib_sha("2026-08-11"),
        session_date="2026-08-11",
        role="validation",
        config_sha256=capture.hash,
        config_source_sha256=capture.source_sha256,
        file_authority=L2SessionFileAuthority(validation_manifest, "4" * 64),
        campaign_identity=campaign,
        symbols=symbols,
        frames={symbol: _frames(symbol, "2026-08-11") for symbol in capture.study.symbols},
        events=events,
    )
    bundles = {
        train_path.absolute(): _bundle(
            train_path.absolute(), date="2026-08-10", role="train", manifest_sha=train_manifest
        ),
        validation_path.absolute(): _bundle(
            validation_path.absolute(),
            date="2026-08-11",
            role="validation",
            manifest_sha=validation_manifest,
        ),
    }

    def fake_capture_verify(path: str | Path, *, expected_config: Any) -> M8L2SessionBundle:
        assert expected_config == capture
        return bundles[Path(path).absolute()]

    def fake_input_verify(
        path: str | Path,
        *,
        expected_config: Any,
        expected_date: str,
        expected_role: str,
        expected_file_authority: Any = None,
        expected_campaign: Any = None,
    ) -> _FakeInput:
        assert expected_config == capture
        events.append(f"authority:{expected_date}")
        value = train_input if expected_date == "2026-08-10" else validation_input
        assert Path(path).absolute() == value.root
        assert expected_role == value.role
        if expected_file_authority is not None:
            assert expected_file_authority == value.file_authority
        if expected_campaign is not None:
            assert expected_campaign == campaign
        return value

    monkeypatch.setattr(development, "verify_m8_l2_session_bundle", fake_capture_verify)
    monkeypatch.setattr(development, "_load_input_verifier", lambda: fake_input_verify)
    monkeypatch.setattr(development, "_producer_source_identity", lambda _root: _source())
    monkeypatch.setattr(
        development,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: RUNTIME_SHA,
    )
    monkeypatch.setattr(development, "_utc_now", lambda: datetime(2026, 8, 11, 16, 0, tzinfo=UTC))
    return _Environment(
        capture,
        analysis,
        train_path,
        validation_path,
        train_input,
        validation_input,
        events,
    )


def test_foreign_import_origin_fails_before_bundle_or_economic_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign" / "research" / "l2_multidate.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# foreign checkout\n", encoding="utf-8")
    bundle_calls = 0

    def forbidden_bundle(*args: object, **kwargs: object) -> M8L2SessionBundle:
        nonlocal bundle_calls
        bundle_calls += 1
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(l2_multidate_module, "__file__", str(foreign))
    monkeypatch.setattr(development, "verify_m8_l2_session_bundle", forbidden_bundle)
    target = tmp_path / "development-lock"
    with pytest.raises(M8L2DevelopmentError, match="foreign or mixed import origin"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )

    assert bundle_calls == 0
    assert environment.events == []
    assert not target.exists()


def test_import_origin_drift_before_marker_is_not_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign" / "research" / "l2_multidate.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# drifted checkout\n", encoding="utf-8")
    original_publish_inventory = development._publish_inventory

    def publish_then_drift(root: Path) -> None:
        original_publish_inventory(root)
        monkeypatch.setattr(l2_multidate_module, "__file__", str(foreign))

    monkeypatch.setattr(development, "_publish_inventory", publish_then_drift)
    target = tmp_path / "development-lock"
    with pytest.raises(M8L2DevelopmentError, match="foreign or mixed import origin"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )

    assert environment.events
    assert not target.exists()


def test_development_memory_budget_boundary_and_selection_estimate() -> None:
    frame = pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
            "future_mid_up": [0, 1, 0],
        }
    )
    observed = development._selection_workspace_upper_bytes(frame, frame, feature_count=2)
    assert observed > 2 * (frame.estimated_size("b") * 2)
    development._require_memory_budget(observed, observed, "selection boundary")
    with pytest.raises(M8L2DevelopmentError, match="memory budget"):
        development._require_memory_budget(observed + 1, observed, "selection overflow")
    assert development._causal_build_upper_bytes(60_000, 4) == (
        60_000 * 4 * development._CAUSAL_ENDPOINT_ROW_UPPER_BYTES
    )


def test_development_raw_overflow_is_system_failure_without_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    monkeypatch.setattr(development, "_MAX_DEVELOPMENT_RAW_BYTES", 1)
    target = tmp_path / "development-lock"
    with pytest.raises(M8L2DevelopmentError, match=r"raw materialization.*memory budget"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )
    assert not target.exists()
    assert not (target / "_LOCKED").exists()


def test_insufficient_development_publishes_verified_not_created_without_economic_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    train = _terminal_bundle(
        environment.train_path,
        date="2026-08-10",
        role="train",
        status="INSUFFICIENT_DATA",
        reasons=("GATE_COVERAGE",),
    )
    validation = _terminal_bundle(
        environment.validation_path,
        date="2026-08-11",
        role="validation",
        status="COMPLETE",
        reasons=(),
    )
    bundles = {train.root: train, validation.root: validation}
    monkeypatch.setattr(
        development,
        "verify_m8_l2_session_bundle",
        lambda path, *, expected_config: bundles[Path(path).absolute()],
    )

    def forbidden_loader() -> object:
        raise AssertionError("NOT_CREATED publication opened an economic input loader")

    monkeypatch.setattr(development, "_load_input_verifier", forbidden_loader)
    target = tmp_path / "development-not-created"
    result = lock_m8_l2_development(
        environment.capture,
        environment.analysis,
        train.root,
        validation.root,
        target,
    )

    assert result.status == "NOT_CREATED"
    assert result.children == ()
    assert result.reason_codes == ("DEVELOPMENT_SESSION_INSUFFICIENT::train::GATE_COVERAGE",)
    assert result.marker_path.read_bytes() == b"not-created\n"
    assert not list(result.root.rglob("*.parquet"))
    payload = json.loads(result.aggregate_path.read_text(encoding="ascii"))
    assert payload["status"] == "NOT_CREATED"
    assert payload["children"] == []
    assert payload["heldout_access"]["economic_rows_opened"] is False

    restored = verify_m8_l2_development_lock(
        environment.capture,
        environment.analysis,
        train.root,
        validation.root,
        result.root,
        expected_lock_sha256=result.aggregate_sha256,
    )
    assert restored == result
    assert environment.events == []


def test_not_created_rejects_caller_authority_swap_before_producer_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    train = _terminal_bundle(
        environment.train_path,
        date="2026-08-10",
        role="train",
        status="INSUFFICIENT_DATA",
        reasons=("GATE_COVERAGE",),
    )
    validation = _terminal_bundle(
        environment.validation_path,
        date="2026-08-11",
        role="validation",
        status="COMPLETE",
        reasons=(),
    )
    authorities = {
        train.session_date: L2SessionFileAuthority(
            train.manifest_sha256,
            hashlib_sha_bytes(train.checksum_path.read_bytes()),
        ),
        validation.session_date: L2SessionFileAuthority(
            validation.manifest_sha256,
            hashlib_sha_bytes(validation.checksum_path.read_bytes()),
        ),
    }
    train.checksum_path.write_bytes(b"self-consistent replacement authority\n")
    bundles = {train.root: train, validation.root: validation}
    monkeypatch.setattr(
        development,
        "verify_m8_l2_session_bundle",
        lambda path, *, expected_config: bundles[Path(path).absolute()],
    )
    monkeypatch.setattr(
        development,
        "_load_input_verifier",
        lambda: pytest.fail("NOT_CREATED swap check opened an economic loader"),
    )
    target = tmp_path / "development-not-created"

    with pytest.raises(M8L2DevelopmentError, match="caller-held train session"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            train.root,
            validation.root,
            target,
            expected_session_file_authorities=authorities,
        )

    assert not target.exists()
    assert environment.events == []


def test_not_created_rechecks_caller_authority_at_marker_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    train = _terminal_bundle(
        environment.train_path,
        date="2026-08-10",
        role="train",
        status="INSUFFICIENT_DATA",
        reasons=("GATE_COVERAGE",),
    )
    validation = _terminal_bundle(
        environment.validation_path,
        date="2026-08-11",
        role="validation",
        status="COMPLETE",
        reasons=(),
    )
    bundles = {train.root: train, validation.root: validation}
    monkeypatch.setattr(
        development,
        "verify_m8_l2_session_bundle",
        lambda path, *, expected_config: bundles[Path(path).absolute()],
    )
    authorities = {
        bundle.session_date: L2SessionFileAuthority(
            bundle.manifest_sha256,
            hashlib_sha_bytes(bundle.checksum_path.read_bytes()),
        )
        for bundle in bundles.values()
    }
    original_publish_inventory = development._publish_inventory

    def publish_then_tamper(root: Path, **kwargs: object) -> None:
        original_publish_inventory(root, **kwargs)
        train.checksum_path.write_bytes(b"tampered before terminal marker\n")

    monkeypatch.setattr(development, "_publish_inventory", publish_then_tamper)
    monkeypatch.setattr(
        development,
        "_load_input_verifier",
        lambda: pytest.fail("NOT_CREATED marker check opened an economic loader"),
    )
    target = tmp_path / "development-not-created"

    with pytest.raises(M8L2DevelopmentError, match="caller-held train session"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            train.root,
            validation.root,
            target,
            expected_session_file_authorities=authorities,
        )

    assert not target.exists()
    assert environment.events == []


@pytest.fixture(scope="module")
def locked_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Environment]:
    monkeypatch = pytest.MonkeyPatch()
    environment = _environment(tmp_path_factory.mktemp("l2-development"), monkeypatch)
    original_select = cast(Any, development.__dict__["select_multidate_model"])
    fit_observations: list[bool] = []
    destination = environment.train_path.parent.parent / "development-lock"

    def observed_select(*args: Any, **kwargs: Any) -> Any:
        fit_observations.append((destination / "_LOCKED").exists())
        return original_select(*args, **kwargs)

    monkeypatch.setattr(development, "select_multidate_model", observed_select)
    environment.result = lock_m8_l2_development(
        environment.capture,
        environment.analysis,
        environment.train_path,
        environment.validation_path,
        destination,
    )
    assert fit_observations == [False] * 8
    try:
        yield environment
    finally:
        monkeypatch.undo()


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _rebuild_publication(root: Path) -> None:
    import hashlib

    aggregate_raw = (root / "development_lock.json").read_bytes()
    aggregate_sha = hashlib.sha256(aggregate_raw).hexdigest()
    (root / "development_lock.sha256").write_text(
        f"{aggregate_sha}  development_lock.json\n", encoding="ascii"
    )
    payload_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"_LOCKED", "CHECKSUMS.sha256", "inventory.json"}
    )
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in payload_files
    ]
    inventory = {
        "schema_version": "m8-l2-development-inventory-v1",
        "artifact_kind": "m8_l2_development_lock_inventory",
        "files": entries,
    }
    (root / "inventory.json").write_bytes(_canonical(inventory))
    checksum_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"_LOCKED", "CHECKSUMS.sha256"}
    )
    (root / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="ascii",
    )


def test_development_lock_is_complete_outcome_blind_and_roundtrips(
    locked_environment: _Environment,
) -> None:
    result = locked_environment.result
    assert result is not None
    assert result.marker_path.read_bytes() == b"locked\n"
    assert len(result.children) == 8
    assert locked_environment.events[:2] == ["authority:2026-08-10", "authority:2026-08-11"]
    assert all(event.startswith("authority:") for event in locked_environment.events[:2])
    aggregate = json.loads(result.aggregate_path.read_text(encoding="ascii"))
    assert aggregate["heldout_access"] == {
        "paths_received": False,
        "file_hashes_received": False,
        "row_counts_received": False,
        "economic_rows_opened": False,
        "model_fit_or_update_after_lock": False,
    }
    assert aggregate["heldout_declarations"] == [
        {"date": "2026-08-12", "role": "primary_test"},
        {"date": "2026-08-13", "role": "replication_test"},
    ]
    restored = verify_m8_l2_development_lock(
        locked_environment.capture,
        locked_environment.analysis,
        locked_environment.train_path,
        locked_environment.validation_path,
        result.root,
        expected_lock_sha256=result.aggregate_sha256,
    )
    assert restored.aggregate_sha256 == result.aggregate_sha256
    assert [(item.symbol, item.endpoint) for item in restored.children] == [
        (symbol, endpoint.name)
        for symbol in locked_environment.capture.study.symbols
        for endpoint in locked_environment.analysis.endpoints
    ]


def test_execution_reference_is_train_only_and_matches_frozen_formula(
    locked_environment: _Environment,
) -> None:
    result = locked_environment.result
    assert result is not None
    aggregate = json.loads(result.aggregate_path.read_text(encoding="ascii"))
    for claim in aggregate["execution_references"]:
        payload = json.loads((result.root / claim["path"]).read_text(encoding="ascii"))
        assert payload["fit_date"] == "2026-08-10"
        assert payload["reference_price_statistic"] == "train_median_mid_price"
        assert payload["reference_depth_statistic"] == "train_q05_min_bid_ask_l1_depth"
        unrounded = min(
            100.0 / payload["reference_mid_price"],
            0.10 * payload["reference_l1_depth_q05"],
        )
        assert payload["unrounded_reference_quantity"] == pytest.approx(unrounded)
        assert payload["reference_quantity_lots"] == math.floor(
            unrounded / payload["lot_size"] + 1e-12
        )
        assert payload["reference_quantity"] == pytest.approx(
            payload["reference_quantity_lots"] * payload["lot_size"]
        )


def test_public_api_has_no_heldout_or_test_path_and_refuses_overwrite(
    locked_environment: _Environment,
) -> None:
    parameters = inspect.signature(lock_m8_l2_development).parameters
    assert "test_bundle_path" not in parameters
    assert "primary_bundle_path" not in parameters
    assert "replication_bundle_path" not in parameters
    result = locked_environment.result
    assert result is not None
    loads_before = sum(event.startswith("load:") for event in locked_environment.events)
    with pytest.raises(M8L2DevelopmentError, match="overwrite is forbidden"):
        lock_m8_l2_development(
            locked_environment.capture,
            locked_environment.analysis,
            locked_environment.train_path,
            locked_environment.validation_path,
            result.root,
        )
    assert sum(event.startswith("load:") for event in locked_environment.events) == loads_before


def test_rechecksummed_semantic_tamper_is_rejected(
    locked_environment: _Environment, tmp_path: Path
) -> None:
    result = locked_environment.result
    assert result is not None
    target = tmp_path / "tampered"
    shutil.copytree(result.root, target)
    aggregate_path = target / "development_lock.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="ascii"))
    claim = aggregate["children"][0]
    child_path = target / claim["path"]
    child = json.loads(child_path.read_text(encoding="ascii"))
    child["selected_model"] = "tree_depth_999"
    child_path.write_bytes(_canonical(child))
    claim["sha256"] = hashlib_sha_bytes(child_path.read_bytes())
    aggregate_path.write_bytes(_canonical(aggregate))
    _rebuild_publication(target)
    with pytest.raises(M8L2DevelopmentError, match=r"contract changed|authorities disagree"):
        verify_m8_l2_development_lock(
            locked_environment.capture,
            locked_environment.analysis,
            locked_environment.train_path,
            locked_environment.validation_path,
            target,
        )


def hashlib_sha_bytes(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def test_missing_terminal_marker_is_rejected(
    locked_environment: _Environment, tmp_path: Path
) -> None:
    result = locked_environment.result
    assert result is not None
    target = tmp_path / "no-marker"
    shutil.copytree(result.root, target)
    (target / "_LOCKED").unlink()
    with pytest.raises(M8L2DevelopmentError, match="terminal marker"):
        verify_m8_l2_development_lock(
            locked_environment.capture,
            locked_environment.analysis,
            locked_environment.train_path,
            locked_environment.validation_path,
            target,
        )


def test_late_lock_and_dirty_source_fail_before_economic_rows_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "late", monkeypatch)
    monkeypatch.setattr(development, "_utc_now", lambda: datetime(2026, 8, 12, 14, 0, tzinfo=UTC))
    target = tmp_path / "late-lock"
    with pytest.raises(M8L2DevelopmentError, match="deadline"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )
    assert not any(event.startswith("load:") for event in environment.events)
    assert not target.exists()

    environment.events.clear()
    monkeypatch.setattr(
        development,
        "_producer_source_identity",
        lambda _root: ProducerSourceIdentity(COMMIT, SOURCE_TREE, True),
    )
    monkeypatch.setattr(development, "_utc_now", lambda: datetime(2026, 8, 11, 16, 0, tzinfo=UTC))
    with pytest.raises(M8L2DevelopmentError, match="clean Git"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            tmp_path / "dirty-lock",
        )
    assert environment.events == []


def test_lock_crossing_deadline_during_fit_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "crossing-deadline", monkeypatch)
    observed_times = iter(
        (
            datetime(2026, 8, 11, 16, 0, tzinfo=UTC),
            datetime(2026, 8, 12, 14, 0, tzinfo=UTC),
        )
    )
    monkeypatch.setattr(development, "_utc_now", lambda: next(observed_times))
    target = tmp_path / "crossing-deadline-lock"

    with pytest.raises(M8L2DevelopmentError, match="deadline"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )

    assert any(event.startswith("load:") for event in environment.events)
    assert not target.exists()


def test_lock_crossing_deadline_during_publication_is_not_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "crossing-publication", monkeypatch)
    observed_times = iter(
        (
            datetime(2026, 8, 11, 16, 0, tzinfo=UTC),
            datetime(2026, 8, 12, 13, 59, 59, tzinfo=UTC),
            datetime(2026, 8, 12, 14, 0, tzinfo=UTC),
        )
    )
    monkeypatch.setattr(development, "_utc_now", lambda: next(observed_times))
    target = tmp_path / "crossing-publication-lock"

    with pytest.raises(M8L2DevelopmentError, match="deadline"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )

    assert any(event.startswith("load:") for event in environment.events)
    assert not target.exists()


def test_lock_crossing_deadline_during_final_authority_recheck_is_not_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "crossing-final-authority", monkeypatch)
    observed_times = iter(
        (
            datetime(2026, 8, 11, 16, 0, tzinfo=UTC),
            datetime(2026, 8, 12, 13, 59, 58, tzinfo=UTC),
            datetime(2026, 8, 12, 13, 59, 59, tzinfo=UTC),
            datetime(2026, 8, 12, 14, 0, tzinfo=UTC),
        )
    )
    monkeypatch.setattr(development, "_utc_now", lambda: next(observed_times))
    target = tmp_path / "crossing-final-authority-lock"

    with pytest.raises(M8L2DevelopmentError, match="deadline"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )

    assert any(event.startswith("load:") for event in environment.events)
    assert not target.exists()


def test_source_drift_during_fitting_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "source-drift", monkeypatch)
    identities = iter(
        (
            _source(),
            ProducerSourceIdentity("3" * 40, "4" * 64, False),
        )
    )
    monkeypatch.setattr(
        development,
        "_producer_source_identity",
        lambda _root: next(identities),
    )
    target = tmp_path / "source-drift-lock"

    with pytest.raises(M8L2DevelopmentError, match="changed before lock publication"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )

    assert any(event.startswith("load:") for event in environment.events)
    assert not target.exists()


def test_runtime_drift_fails_before_rows_and_before_terminal_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    early = _environment(tmp_path / "runtime-early", monkeypatch)
    monkeypatch.setattr(
        development,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: "e" * 64,
    )
    with pytest.raises(M8L2DevelopmentError, match="runtime differs"):
        lock_m8_l2_development(
            early.capture,
            early.analysis,
            early.train_path,
            early.validation_path,
            tmp_path / "runtime-early-lock",
        )
    assert not any(event.startswith("load:") for event in early.events)

    late = _environment(tmp_path / "runtime-late", monkeypatch)
    observed = iter((RUNTIME_SHA, "f" * 64))
    monkeypatch.setattr(
        development,
        "current_m8_l2_runtime_fingerprint_sha256",
        lambda: next(observed),
    )
    target = tmp_path / "runtime-late-lock"
    with pytest.raises(M8L2DevelopmentError, match="runtime changed"):
        lock_m8_l2_development(
            late.capture,
            late.analysis,
            late.train_path,
            late.validation_path,
            target,
        )
    assert any(event.startswith("load:") for event in late.events)
    assert not target.exists()


def test_system_fault_removes_unlocked_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "fault", monkeypatch)
    original_write_json = development._write_json

    def fail_regime(path: Path, payload: Any) -> str:
        if path.name == "regime_thresholds.json":
            raise OSError("injected durable-write failure")
        return original_write_json(path, payload)

    monkeypatch.setattr(development, "_write_json", fail_regime)
    target = tmp_path / "fault-lock"
    with pytest.raises(OSError, match="injected"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            target,
        )
    assert not target.exists()
    assert all(not event.startswith("load:2026-08-12") for event in environment.events)


def test_wrong_development_coordinates_and_campaign_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path / "wrong", monkeypatch)
    with pytest.raises(M8L2DevelopmentError, match="COMPLETE 2026-08-10 train"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.validation_path,
            environment.train_path,
            tmp_path / "reversed",
        )
    assert environment.events == []

    environment.validation_input.campaign_identity = L2CampaignRuntimeIdentity(
        "e" * 64, COMMIT, SOURCE_TREE, RUNTIME_SHA, False
    )
    with pytest.raises(M8L2DevelopmentError, match="campaign"):
        lock_m8_l2_development(
            environment.capture,
            environment.analysis,
            environment.train_path,
            environment.validation_path,
            tmp_path / "campaign-mismatch",
        )
    assert not any(event.startswith("load:") for event in environment.events)
