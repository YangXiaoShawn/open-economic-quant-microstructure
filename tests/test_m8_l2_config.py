from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from microstructure.m8_l2_config import (
    M8_L2_CONFIG_SOURCE_SHA256,
    M8_L2_FREEZE_COMMIT,
    M8_L2_PROTOCOL_SHA256,
    M8L2ConfigError,
    load_m8_l2_config,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_l2_capture_study.toml"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "M8_L2_PROTOCOL.md"


def _mutated(tmp_path: Path, old: str, new: str) -> Path:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "mutated.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def test_frozen_live_l2_config_is_typed_and_bound_to_exact_bytes(tmp_path: Path) -> None:
    config = load_m8_l2_config(CONFIG_PATH)

    assert config.source_sha256 == M8_L2_CONFIG_SOURCE_SHA256
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == M8_L2_CONFIG_SOURCE_SHA256
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == M8_L2_PROTOCOL_SHA256
    assert M8_L2_FREEZE_COMMIT == "6db6c8cf81b726069d1833672864e0554976b985"
    assert config.study.symbols == ("BTCUSDT", "ETHUSDT")
    assert config.study.stream_interval_ms == 100
    assert config.capture.duration_seconds == 3600
    assert config.capture.max_messages_per_symbol == 60_000
    assert config.sessions[0].start_ns == 1_786_370_400_000_000_000
    assert config.sessions[0].end_ns - config.sessions[0].start_ns == 3_600_000_000_000
    assert [item.role for item in config.sessions] == [
        "train",
        "validation",
        "primary_test",
        "replication_test",
    ]
    assert len(config.hash) == 64
    assert config.public_dict()["source_sha256"] == M8_L2_CONFIG_SOURCE_SHA256

    relocated = tmp_path / "same-bytes.toml"
    relocated.write_bytes(CONFIG_PATH.read_bytes())
    assert load_m8_l2_config(relocated).hash == config.hash


def test_live_l2_config_objects_are_immutable() -> None:
    config = load_m8_l2_config(CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.capture.duration_seconds = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'symbols = ["BTCUSDT", "ETHUSDT"]',
            'symbols = ["ETHUSDT", "BTCUSDT"]',
            r"study\.symbols is frozen",
        ),
        ("stream_interval_ms = 100", "stream_interval_ms = 1000", "stream_interval_ms is frozen"),
        ('date = "2026-08-10"', 'date = "2026-08-14"', "session calendar/order is frozen"),
        ('start_utc = "14:00:00"', 'start_utc = "14:00:01"', "session calendar/order is frozen"),
        ("duration_seconds = 3600", "duration_seconds = 3599", "capture contract is frozen"),
        (
            "min_overlapping_coverage_seconds = 3300",
            "min_overlapping_coverage_seconds = 1",
            "capture contract is frozen",
        ),
        ("max_quality_warnings = 0", "max_quality_warnings = 1", "capture contract is frozen"),
        ("depth_levels = [1, 5, 10]", "depth_levels = [1, 5]", "features contract is frozen"),
        ("bootstrap_samples = 2000", "bootstrap_samples = 1999", "models contract is frozen"),
        ("market_orders_only = true", "market_orders_only = false", "execution contract is frozen"),
        ("allow_p_values = false", "allow_p_values = true", "claims contract is frozen"),
    ],
)
def test_every_frozen_dimension_fails_closed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _mutated(tmp_path, old, new)

    with pytest.raises(M8L2ConfigError, match=message):
        load_m8_l2_config(path)


def test_formatting_only_byte_drift_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "commented.toml"
    path.write_text("# outcome-blind but different bytes\n" + CONFIG_PATH.read_text())

    with pytest.raises(M8L2ConfigError, match="configuration bytes do not match"):
        load_m8_l2_config(path)


def test_unknown_key_and_bool_as_integer_fail_before_byte_authority(tmp_path: Path) -> None:
    unknown = _mutated(
        tmp_path,
        'name = "binance-m8-live-l2-study-v2"',
        'name = "binance-m8-live-l2-study-v2"\nunreviewed = true',
    )
    with pytest.raises(M8L2ConfigError, match=r"unknown=.*unreviewed"):
        load_m8_l2_config(unknown)

    invalid_type = _mutated(tmp_path, "duration_seconds = 3600", "duration_seconds = true")
    with pytest.raises(M8L2ConfigError, match="duration_seconds must be an integer"):
        load_m8_l2_config(invalid_type)


def test_only_frozen_dates_can_be_resolved() -> None:
    config = load_m8_l2_config(CONFIG_PATH)

    assert config.session_for_date("2026-08-12").role == "primary_test"
    with pytest.raises(M8L2ConfigError, match="not a frozen"):
        config.session_for_date("2026-08-14")
