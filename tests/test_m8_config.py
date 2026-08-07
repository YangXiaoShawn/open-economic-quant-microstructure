from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from microstructure.m8_config import M8ConfigError, load_m8_config

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_multidate_trade_study.toml"


def _mutated_config(tmp_path: Path, old: str, new: str) -> Path:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "m8-invalid.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def test_frozen_m8_config_is_typed_hashed_and_json_safe() -> None:
    config = load_m8_config(CONFIG_PATH)

    assert config.path == CONFIG_PATH.resolve()
    assert config.study.protocol_version == "1.0.2"
    assert config.study.source == "binance_spot_daily_aggtrades_archive"
    assert config.study.symbols == ("BTCUSDT", "ETHUSDT")
    assert tuple((period.date, period.role) for period in config.periods) == (
        (date(2024, 1, 3), "train"),
        (date(2024, 1, 4), "validation"),
        (date(2024, 1, 5), "primary_test"),
        (date(2024, 1, 6), "replication_test"),
    )
    assert config.source_sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert len(config.hash) == 64
    public = config.public_dict()
    assert public["config_sha256"] == config.hash
    assert public["source_sha256"] == config.source_sha256
    assert json.loads(json.dumps(public))["periods"][3]["role"] == "replication_test"


def test_semantic_hash_ignores_path_comments_and_formatting(tmp_path: Path) -> None:
    original = load_m8_config(CONFIG_PATH)
    relocated_path = tmp_path / "relocated.toml"
    relocated_path.write_text(
        "# formatting-only comment\n" + CONFIG_PATH.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    relocated = load_m8_config(relocated_path)

    assert relocated.hash == original.hash
    assert relocated.source_sha256 != original.source_sha256
    assert relocated.path != original.path


def test_config_objects_are_immutable() -> None:
    config = load_m8_config(CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.study.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'source = "binance_spot_daily_aggtrades_archive"',
            'source = "binance_spot_rest"',
            r"study\.source is frozen",
        ),
        (
            'symbols = ["BTCUSDT", "ETHUSDT"]',
            'symbols = ["ETHUSDT", "BTCUSDT"]',
            r"study\.symbols is frozen",
        ),
        (
            'symbols = ["BTCUSDT", "ETHUSDT"]',
            'symbols = ["BTCUSDT", "BTCUSDT"]',
            r"study\.symbols is frozen|must be unique",
        ),
        (
            "max_archive_compressed_bytes = 268435456",
            "max_archive_compressed_bytes = 268435457",
            "max_archive_compressed_bytes is frozen",
        ),
        (
            "max_archive_uncompressed_bytes = 2147483648",
            "max_archive_uncompressed_bytes = 0",
            "max_archive_uncompressed_bytes is frozen",
        ),
        (
            "max_total_download_bytes = 8589934592",
            "max_total_download_bytes = 8589934593",
            "max_total_download_bytes is frozen",
        ),
        (
            "trade_windows = [5, 20, 100]",
            "trade_windows = [5, 20, 200]",
            r"features\.trade_windows is frozen",
        ),
        (
            "large_trade_quantile = 0.95",
            "large_trade_quantile = 0.90",
            "large_trade_quantile is frozen",
        ),
        (
            "logistic_c_values = [0.1, 1.0, 10.0]",
            "logistic_c_values = [0.1, 1.0, 100.0]",
            r"models\.logistic_c_values is frozen",
        ),
        (
            "tree_max_depth_values = [2, 4, 6]",
            "tree_max_depth_values = [2, 4, 8]",
            "tree_max_depth_values is frozen",
        ),
        (
            "allow_significance_claim = false",
            "allow_significance_claim = true",
            r"claims\.allow_significance_claim is frozen",
        ),
        (
            "allow_quality_warnings = false",
            "allow_quality_warnings = true",
            r"quality\.allow_quality_warnings is frozen",
        ),
    ],
)
def test_frozen_study_dimensions_fail_closed(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _mutated_config(tmp_path, old, new)

    with pytest.raises(M8ConfigError, match=message):
        load_m8_config(path)


def test_period_roles_must_have_the_exact_frozen_order(tmp_path: Path) -> None:
    path = _mutated_config(tmp_path, 'role = "validation"', 'role = "primary_test"')

    with pytest.raises(M8ConfigError, match="period date/role order is frozen"):
        load_m8_config(path)


def test_period_dates_must_be_unique(tmp_path: Path) -> None:
    path = _mutated_config(tmp_path, 'date = "2024-01-04"', 'date = "2024-01-03"')

    with pytest.raises(M8ConfigError, match="period dates must be unique"):
        load_m8_config(path)


def test_period_dates_cannot_be_reordered_or_replaced(tmp_path: Path) -> None:
    path = _mutated_config(tmp_path, 'date = "2024-01-06"', 'date = "2024-01-07"')

    with pytest.raises(M8ConfigError, match="period date/role order is frozen"):
        load_m8_config(path)


def test_unknown_or_missing_fields_fail_closed(tmp_path: Path) -> None:
    path = _mutated_config(
        tmp_path,
        'target = "future_trade_up"',
        'target = "future_trade_up"\nunreviewed_option = true',
    )

    with pytest.raises(M8ConfigError, match=r"study keys.*unknown=unreviewed_option"):
        load_m8_config(path)


def test_invalid_types_do_not_coerce_bool_to_integer(tmp_path: Path) -> None:
    path = _mutated_config(tmp_path, "bootstrap_samples = 2000", "bootstrap_samples = true")

    with pytest.raises(M8ConfigError, match=r"study\.bootstrap_samples must be an integer"):
        load_m8_config(path)


def test_invalid_toml_is_wrapped_as_m8_config_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("[study\n", encoding="utf-8")

    with pytest.raises(M8ConfigError, match="cannot parse M8 TOML"):
        load_m8_config(path)
