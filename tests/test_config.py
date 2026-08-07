from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest

from microstructure.config import ConfigError, datetime_to_ns, load_config

PROJECT_ROOT = Path(__file__).parents[1]


def test_smoke_config_is_typed_and_stably_hashed() -> None:
    first = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    second = load_config(PROJECT_ROOT / "configs" / "smoke.toml")

    assert first.hash == second.hash
    assert len(first.hash) == 64
    assert first.data.symbols == ("BTCUSDT", "ETHUSDT")
    assert first.data.start.tzinfo == UTC
    assert first.data.partition_root == PROJECT_ROOT / "data" / "normalized"
    assert first.evaluation.embargo_events >= first.features.label_horizon_events


def test_datetime_to_ns_rejects_naive_values() -> None:
    from datetime import datetime

    with pytest.raises(ConfigError, match="timezone aware"):
        datetime_to_ns(datetime(2024, 1, 1))


def test_config_rejects_unimplemented_schema_version(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "unsupported-schema.toml"
    path.write_text(
        source.replace('schema_version = "1.0.0"', 'schema_version = "2.0.0"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"unsupported data\.schema_version"):
        load_config(path)


def test_config_accepts_extensible_adapter_mode_identifier(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "third-party-mode.toml"
    path.write_text(
        source.replace('mode = "synthetic"', 'mode = "fixture_vendor.v1"'),
        encoding="utf-8",
    )

    assert load_config(path).data.mode == "fixture_vendor.v1"


def test_config_rejects_unsafe_adapter_mode_identifier(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs" / "smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "unsafe-mode.toml"
    path.write_text(
        source.replace('mode = "synthetic"', 'mode = "Fixture Vendor"'),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="lowercase adapter identifier"):
        load_config(path)


@pytest.mark.parametrize(
    ("line", "replacement", "message"),
    [
        (
            'end = "2024-01-02T00:10:00Z"',
            "",
            r"requires a bounded data\.end",
        ),
        (
            "max_events_per_symbol = 5000",
            "max_events_per_symbol = 0",
            "requires positive data.max_events_per_symbol",
        ),
    ],
)
def test_public_config_fails_fast_on_unbounded_inputs(
    tmp_path: Path, line: str, replacement: str, message: str
) -> None:
    source = (PROJECT_ROOT / "configs" / "public_sample.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid-public.toml"
    path.write_text(source.replace(line, replacement), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)
