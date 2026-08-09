from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import microstructure.m8_l2_inputs as input_module
import test_m8_l2_capture as capture_fixtures
from microstructure.data.schemas import SCHEMA_VERSION, get_schema
from microstructure.m8_l2_config import M8L2Session, M8L2StudyConfig, load_m8_l2_config
from microstructure.m8_l2_inputs import (
    L2CampaignRuntimeIdentity,
    L2SessionFileAuthority,
    M8L2InputError,
    verify_m8_l2_development_input,
    verify_m8_l2_heldout_input,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_l2_capture_study.toml"


def _base_record(field: pa.Field) -> object:
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
    raise AssertionError(f"unsupported fixture field: {field}")


def _valid_parquet(
    path: Path,
    dataset: str,
    rows: int,
    *,
    session: M8L2Session,
    corrupt_continuity: bool = False,
) -> None:
    schema = get_schema(dataset)
    symbol = next(part for part in path.parts if part in {"BTCUSDT", "ETHUSDT"})
    continuity = f"{symbol}-epoch-1"
    times = [session.start_ns + 10_000_000_000, session.end_ns - 10_000_000_001]
    records: list[dict[str, object]] = []
    for index in range(rows):
        record = {field.name: _base_record(field) for field in schema}
        available = times[min(index, len(times) - 1)]
        record.update(
            {
                "venue": "binance_spot",
                "symbol": symbol,
                "event_ts_ns": available,
                "received_ts_ns": available,
                "available_ts_ns": available,
                "availability_basis": "local_receipt",
                "capture_seq": index + 1,
                "continuity_id": continuity,
                "source_artifact_id": "a" * 64,
                "tick_size": 0.01,
                "lot_size": 0.001,
            }
        )
        if dataset == "depth_deltas":
            record.update(
                {
                    "first_update_id": index + 1,
                    "last_update_id": index + 1,
                    "previous_update_id": index if index else None,
                    "bids": [],
                    "asks": [],
                }
            )
        elif dataset == "book_observations":
            record.update(
                {
                    "continuity_id": "wrong-epoch" if corrupt_continuity else continuity,
                    "sequence_start": index + 1,
                    "sequence_end": index + 1,
                    "is_valid": True,
                    "best_bid_ticks": 10_000,
                    "best_ask_ticks": 10_001,
                    "bid_quantity_lots": 2_000,
                    "ask_quantity_lots": 2_000,
                    "best_bid": 100.0,
                    "best_ask": 100.01,
                    "bid_quantity": 2.0,
                    "ask_quantity": 2.0,
                    "spread": 0.01,
                    "mid_price": 100.005,
                    "microprice": 100.005,
                    "depth_bid_1": 2.0,
                    "depth_ask_1": 2.0,
                    "depth_bid_5": 2.0,
                    "depth_ask_5": 2.0,
                    "depth_bid_10": 2.0,
                    "depth_ask_10": 2.0,
                    "queue_imbalance_1": 0.0,
                    "queue_imbalance_5": 0.0,
                    "queue_imbalance_10": 0.0,
                }
            )
        elif dataset == "book_snapshots":
            record.update(
                {
                    "snapshot_id": f"{symbol}-snapshot",
                    "request_ts_ns": session.start_ns,
                    "received_ts_ns": session.start_ns + 1,
                    "available_ts_ns": session.start_ns + 1,
                    "last_update_id": 0,
                    "depth_limit": 100,
                    "bids": [],
                    "asks": [],
                }
            )
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)


class _CaptureFactory:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: M8L2StudyConfig,
        *,
        corrupt_continuity: bool = False,
    ) -> None:
        self.config = config
        self.session = config.sessions[0]

        def write(path: Path, dataset: str, rows: int) -> None:
            _valid_parquet(
                path,
                dataset,
                rows,
                session=self.session,
                corrupt_continuity=corrupt_continuity,
            )

        monkeypatch.setattr(capture_fixtures, "_parquet", write)

    def capture(self, output_root: Path, session_index: int):
        self.session = self.config.sessions[session_index]
        return capture_fixtures._capture_complete_session(
            output_root,
            config=self.config,
            session=self.session,
        )


@pytest.fixture
def config() -> M8L2StudyConfig:
    return load_m8_l2_config(CONFIG_PATH)


def test_development_input_binds_authority_and_loads_only_requested_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 0)
    verified = verify_m8_l2_development_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
    )

    assert verified.root == bundle.root
    assert verified.access_phase == "development"
    assert verified.development_lock_sha256 is None
    assert verified.file_authority.manifest_sha256 == bundle.manifest_sha256
    assert verified.file_authority.to_dict() == {
        "manifest_sha256": bundle.manifest_sha256,
        "checksums_sha256": verified.file_authority.checksums_sha256,
    }
    assert verified.campaign_identity.to_dict()["runtime_dirty"] is False
    assert set(verified.symbols) == {"BTCUSDT", "ETHUSDT"}
    assert verified.symbols["BTCUSDT"].depth_deltas.rows == 2
    assert verified.symbols["BTCUSDT"].book_observations.rows == 2

    opened: list[str] = []
    original = input_module._load_verified_parquet
    original_lineage = input_module._verify_artifact_hash

    def recording_loader(*args: object, **kwargs: object):
        artifact = kwargs["artifact"]
        assert isinstance(artifact, input_module.VerifiedL2Artifact)
        opened.append(artifact.relative_path)
        return original(*args, **kwargs)

    def recording_lineage(*args: object, **kwargs: object) -> None:
        artifact = kwargs["artifact"]
        assert isinstance(artifact, input_module.VerifiedL2Artifact)
        opened.append(artifact.relative_path)
        original_lineage(*args, **kwargs)

    monkeypatch.setattr(input_module, "_load_verified_parquet", recording_loader)
    monkeypatch.setattr(input_module, "_verify_artifact_hash", recording_lineage)
    loaded = verified.load_symbol_frames("BTCUSDT")
    assert loaded.book_observations.height == 2
    assert loaded.depth_deltas.height == 2
    assert loaded.intervals == verified.symbols["BTCUSDT"].valid_observed_intervals
    assert len(opened) == 4
    assert all(path.startswith("symbols/BTCUSDT/") for path in opened)
    assert not any("ETHUSDT" in path for path in opened)

    rebound = verify_m8_l2_development_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
        expected_file_authority=verified.file_authority,
        expected_campaign=verified.campaign_identity,
    )
    assert rebound.file_authority == verified.file_authority


def test_development_coordinates_are_hard_separated_before_bundle_open(
    tmp_path: Path,
    config: M8L2StudyConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object):
        nonlocal called
        called = True
        raise AssertionError("verifier must not open a held-out coordinate through dev API")

    monkeypatch.setattr(input_module, "verify_m8_l2_session_bundle", forbidden)
    with pytest.raises(M8L2InputError, match="development input"):
        verify_m8_l2_development_input(
            tmp_path,
            expected_config=config,
            expected_date="2026-08-12",
            expected_role=cast(input_module.DevelopmentRole, "primary_test"),
        )
    assert called is False


def test_heldout_input_requires_and_binds_development_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    with pytest.raises(M8L2InputError, match="development lock authority"):
        verify_m8_l2_heldout_input(
            tmp_path,
            expected_config=config,
            expected_date="2026-08-12",
            expected_role="primary_test",
            development_lock_sha256="not-a-digest",
        )

    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 2)
    lock_sha256 = "d" * 64
    verified = verify_m8_l2_heldout_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-12",
        expected_role="primary_test",
        development_lock_sha256=lock_sha256,
    )
    assert verified.access_phase == "heldout_after_lock"
    assert verified.development_lock_sha256 == lock_sha256
    assert verified.load_symbol_frames("ETHUSDT").book_observations.height == 2


def test_cross_session_campaign_identity_must_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    train = factory.capture(tmp_path, 0)
    train_input = verify_m8_l2_development_input(
        train.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
    )
    validation = factory.capture(tmp_path, 1)
    matching = verify_m8_l2_development_input(
        validation.root,
        expected_config=config,
        expected_date="2026-08-11",
        expected_role="validation",
        expected_campaign=train_input.campaign_identity,
    )
    assert matching.campaign_identity == train_input.campaign_identity

    wrong = replace(
        train_input.campaign_identity,
        runtime_source_tree_sha256="3" * 64,
    )
    with pytest.raises(M8L2InputError, match="campaign/runtime identity"):
        verify_m8_l2_development_input(
            validation.root,
            expected_config=config,
            expected_date="2026-08-11",
            expected_role="validation",
            expected_campaign=wrong,
        )


def test_external_file_authority_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 0)
    wrong = L2SessionFileAuthority("0" * 64, "1" * 64)
    with pytest.raises(M8L2InputError, match="external digest authority"):
        verify_m8_l2_development_input(
            bundle.root,
            expected_config=config,
            expected_date="2026-08-10",
            expected_role="train",
            expected_file_authority=wrong,
        )


def test_insufficient_extra_file_and_symlink_roots_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 0)

    original_verifier = input_module.verify_m8_l2_session_bundle
    monkeypatch.setattr(
        input_module,
        "verify_m8_l2_session_bundle",
        lambda *args, **kwargs: replace(
            bundle, status="INSUFFICIENT_DATA", reason_codes=("NO_MESSAGES",)
        ),
    )
    with pytest.raises(M8L2InputError, match="gate-complete"):
        verify_m8_l2_development_input(
            bundle.root,
            expected_config=config,
            expected_date="2026-08-10",
            expected_role="train",
        )
    monkeypatch.setattr(input_module, "verify_m8_l2_session_bundle", original_verifier)

    extra = bundle.root / "unmanifested.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(M8L2InputError, match="capture authority failed verification"):
        verify_m8_l2_development_input(
            bundle.root,
            expected_config=config,
            expected_date="2026-08-10",
            expected_role="train",
        )
    extra.unlink()

    alias = tmp_path / "session-alias"
    alias.symlink_to(bundle.root, target_is_directory=True)
    with pytest.raises(M8L2InputError, match="capture authority failed verification"):
        verify_m8_l2_development_input(
            alias,
            expected_config=config,
            expected_date="2026-08-10",
            expected_role="train",
        )


def test_payload_tamper_after_verification_is_detected_before_rows_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 0)
    verified = verify_m8_l2_development_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
    )
    target = bundle.root / verified.symbols["BTCUSDT"].book_observations.relative_path
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(M8L2InputError, match=r"Parquet|changed|claimed"):
        verified.load_symbol_frames("BTCUSDT")


def test_loader_rejects_an_object_not_created_by_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 0)
    verified = verify_m8_l2_development_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
    )
    forged = replace(verified, _verification_token=object())
    with pytest.raises(M8L2InputError, match="verifier-created"):
        forged.load_symbol_frames("BTCUSDT")


def test_descriptor_snapshot_detects_atomic_path_replacement_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config)
    bundle = factory.capture(tmp_path, 0)
    verified = verify_m8_l2_development_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
    )
    target = bundle.root / verified.symbols["BTCUSDT"].book_observations.relative_path
    replacement = tmp_path / "replacement.parquet"
    shutil.copy2(target, replacement)
    original_hash = input_module._hash_descriptor
    replaced = False

    target_inode = target.stat().st_ino

    def replace_after_hash(descriptor: int, maximum_bytes: int) -> tuple[str, int]:
        nonlocal replaced
        result = original_hash(descriptor, maximum_bytes)
        if not replaced and os.fstat(descriptor).st_ino == target_inode:
            os.replace(replacement, target)
            replaced = True
        return result

    monkeypatch.setattr(input_module, "_hash_descriptor", replace_after_hash)
    with pytest.raises(M8L2InputError, match="path changed"):
        verified.load_symbol_frames("BTCUSDT")
    assert replaced is True


def test_loaded_row_continuity_must_reconcile_to_verified_intervals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: M8L2StudyConfig,
) -> None:
    factory = _CaptureFactory(monkeypatch, config, corrupt_continuity=True)
    bundle = factory.capture(tmp_path, 0)
    verified = verify_m8_l2_development_input(
        bundle.root,
        expected_config=config,
        expected_date="2026-08-10",
        expected_role="train",
    )
    with pytest.raises(M8L2InputError, match=r"reconcile|OBSERVED interval"):
        verified.load_symbol_frames("BTCUSDT")


def test_campaign_identity_constructor_rejects_dirty_runtime() -> None:
    with pytest.raises(M8L2InputError, match="must be clean"):
        L2CampaignRuntimeIdentity(
            campaign_authority_sha256="a" * 64,
            runtime_commit="b" * 40,
            runtime_source_tree_sha256="c" * 64,
            runtime_fingerprint_sha256="d" * 64,
            runtime_dirty=True,
        )
