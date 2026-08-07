"""Fail-closed parser for the frozen M8 multi-date trade study.

This module intentionally does not reuse the exploratory sample configuration.
Protocol version 1.0.2 is a fixed, outcome-blind study contract: changing any
date, role, source, feature, model, safety ceiling, or claim permission requires
a new protocol version and corresponding parser review.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Any, Literal, cast

M8PeriodRole = Literal["train", "validation", "primary_test", "replication_test"]

_TOP_LEVEL_KEYS = frozenset({"study", "periods", "features", "models", "quality", "claims"})
_STUDY_KEYS = frozenset(
    {
        "name",
        "protocol_version",
        "evidence_tier",
        "seed",
        "source",
        "symbols",
        "selection_metric",
        "target",
        "label_horizon_events",
        "calibration_fraction",
        "bootstrap_samples",
        "bootstrap_block_events",
        "feature_stability_bins",
        "max_archive_compressed_bytes",
        "max_archive_uncompressed_bytes",
        "max_total_download_bytes",
    }
)
_PERIOD_KEYS = frozenset({"date", "role"})
_FEATURE_KEYS = frozenset(
    {"trade_windows", "volatility_window", "intensity_window", "large_trade_quantile"}
)
_MODEL_KEYS = frozenset({"logistic_c_values", "tree_max_depth_values", "tree_min_samples_leaf"})
_QUALITY_KEYS = frozenset(
    {
        "fail_on_error",
        "require_complete_daily_archive",
        "require_contiguous_trade_ids_within_symbol_date",
        "require_nondecreasing_event_time",
        "allow_quality_warnings",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "allow_p_values",
        "allow_significance_claim",
        "allow_cross_instrument_pooling",
        "allow_execution_claim",
        "allow_profitability_claim",
    }
)

_FROZEN_PERIODS: tuple[tuple[str, M8PeriodRole], ...] = (
    ("2024-01-03", "train"),
    ("2024-01-04", "validation"),
    ("2024-01-05", "primary_test"),
    ("2024-01-06", "replication_test"),
)
_FROZEN_SYMBOLS = ("BTCUSDT", "ETHUSDT")


class M8ConfigError(ValueError):
    """Raised when an M8 study file violates its frozen protocol contract."""


@dataclass(frozen=True, slots=True)
class M8Study:
    name: str
    protocol_version: str
    evidence_tier: str
    seed: int
    source: str
    symbols: tuple[str, ...]
    selection_metric: str
    target: str
    label_horizon_events: int
    calibration_fraction: float
    bootstrap_samples: int
    bootstrap_block_events: int
    feature_stability_bins: int
    max_archive_compressed_bytes: int
    max_archive_uncompressed_bytes: int
    max_total_download_bytes: int


@dataclass(frozen=True, slots=True)
class M8Period:
    date: Date
    role: M8PeriodRole


@dataclass(frozen=True, slots=True)
class M8Features:
    trade_windows: tuple[int, ...]
    volatility_window: int
    intensity_window: int
    large_trade_quantile: float


@dataclass(frozen=True, slots=True)
class M8Models:
    logistic_c_values: tuple[float, ...]
    tree_max_depth_values: tuple[int, ...]
    tree_min_samples_leaf: int


@dataclass(frozen=True, slots=True)
class M8Quality:
    fail_on_error: bool
    require_complete_daily_archive: bool
    require_contiguous_trade_ids_within_symbol_date: bool
    require_nondecreasing_event_time: bool
    allow_quality_warnings: bool


@dataclass(frozen=True, slots=True)
class M8Claims:
    allow_p_values: bool
    allow_significance_claim: bool
    allow_cross_instrument_pooling: bool
    allow_execution_claim: bool
    allow_profitability_claim: bool


@dataclass(frozen=True, slots=True)
class M8StudyConfig:
    """Typed, immutable representation of the frozen M8 study specification."""

    path: Path
    source_sha256: str
    study: M8Study
    periods: tuple[M8Period, ...]
    features: M8Features
    models: M8Models
    quality: M8Quality
    claims: M8Claims

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "study": {
                "name": self.study.name,
                "protocol_version": self.study.protocol_version,
                "evidence_tier": self.study.evidence_tier,
                "seed": self.study.seed,
                "source": self.study.source,
                "symbols": list(self.study.symbols),
                "selection_metric": self.study.selection_metric,
                "target": self.study.target,
                "label_horizon_events": self.study.label_horizon_events,
                "calibration_fraction": self.study.calibration_fraction,
                "bootstrap_samples": self.study.bootstrap_samples,
                "bootstrap_block_events": self.study.bootstrap_block_events,
                "feature_stability_bins": self.study.feature_stability_bins,
                "max_archive_compressed_bytes": self.study.max_archive_compressed_bytes,
                "max_archive_uncompressed_bytes": self.study.max_archive_uncompressed_bytes,
                "max_total_download_bytes": self.study.max_total_download_bytes,
            },
            "periods": [
                {"date": period.date.isoformat(), "role": period.role} for period in self.periods
            ],
            "features": {
                "trade_windows": list(self.features.trade_windows),
                "volatility_window": self.features.volatility_window,
                "intensity_window": self.features.intensity_window,
                "large_trade_quantile": self.features.large_trade_quantile,
            },
            "models": {
                "logistic_c_values": list(self.models.logistic_c_values),
                "tree_max_depth_values": list(self.models.tree_max_depth_values),
                "tree_min_samples_leaf": self.models.tree_min_samples_leaf,
            },
            "quality": {
                "fail_on_error": self.quality.fail_on_error,
                "require_complete_daily_archive": self.quality.require_complete_daily_archive,
                "require_contiguous_trade_ids_within_symbol_date": (
                    self.quality.require_contiguous_trade_ids_within_symbol_date
                ),
                "require_nondecreasing_event_time": self.quality.require_nondecreasing_event_time,
                "allow_quality_warnings": self.quality.allow_quality_warnings,
            },
            "claims": {
                "allow_p_values": self.claims.allow_p_values,
                "allow_significance_claim": self.claims.allow_significance_claim,
                "allow_cross_instrument_pooling": self.claims.allow_cross_instrument_pooling,
                "allow_execution_claim": self.claims.allow_execution_claim,
                "allow_profitability_claim": self.claims.allow_profitability_claim,
            },
        }

    @property
    def hash(self) -> str:
        """Return a location- and formatting-independent semantic SHA-256."""

        encoded = json.dumps(
            self._semantic_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def public_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation with both semantic and byte hashes."""

        return {
            "path": str(self.path),
            "config_sha256": self.hash,
            "source_sha256": self.source_sha256,
            **self._semantic_payload(),
        }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M8ConfigError(f"{label} must be a TOML table")
    if not all(isinstance(key, str) for key in value):
        raise M8ConfigError(f"{label} contains a non-string key")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise M8ConfigError(f"{label} keys do not match the frozen contract ({'; '.join(details)})")


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise M8ConfigError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise M8ConfigError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise M8ConfigError(f"{label} must be a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise M8ConfigError(f"{label} must be a finite number")
    return result


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise M8ConfigError(f"{label} must be a boolean")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise M8ConfigError(f"{label} must be an array")
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label)))


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label))
    )


def _number_tuple(value: object, label: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label))
    )


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise M8ConfigError(f"{label} is frozen at {expected!r}, observed {observed!r}")


def _parse_study(raw: object) -> M8Study:
    table = _mapping(raw, "study")
    _exact_keys(table, _STUDY_KEYS, "study")
    study = M8Study(
        name=_text(table["name"], "study.name"),
        protocol_version=_text(table["protocol_version"], "study.protocol_version"),
        evidence_tier=_text(table["evidence_tier"], "study.evidence_tier"),
        seed=_integer(table["seed"], "study.seed"),
        source=_text(table["source"], "study.source"),
        symbols=_text_tuple(table["symbols"], "study.symbols"),
        selection_metric=_text(table["selection_metric"], "study.selection_metric"),
        target=_text(table["target"], "study.target"),
        label_horizon_events=_integer(table["label_horizon_events"], "study.label_horizon_events"),
        calibration_fraction=_number(table["calibration_fraction"], "study.calibration_fraction"),
        bootstrap_samples=_integer(table["bootstrap_samples"], "study.bootstrap_samples"),
        bootstrap_block_events=_integer(
            table["bootstrap_block_events"], "study.bootstrap_block_events"
        ),
        feature_stability_bins=_integer(
            table["feature_stability_bins"], "study.feature_stability_bins"
        ),
        max_archive_compressed_bytes=_integer(
            table["max_archive_compressed_bytes"], "study.max_archive_compressed_bytes"
        ),
        max_archive_uncompressed_bytes=_integer(
            table["max_archive_uncompressed_bytes"], "study.max_archive_uncompressed_bytes"
        ),
        max_total_download_bytes=_integer(
            table["max_total_download_bytes"], "study.max_total_download_bytes"
        ),
    )
    if len(set(study.symbols)) != len(study.symbols):
        raise M8ConfigError("study.symbols must be unique")

    expected: dict[str, object] = {
        "name": "binance-m8-multidate-trades",
        "protocol_version": "1.0.2",
        "evidence_tier": "FULL_DATA",
        "seed": 20260807,
        "source": "binance_spot_daily_aggtrades_archive",
        "symbols": _FROZEN_SYMBOLS,
        "selection_metric": "log_loss",
        "target": "future_trade_up",
        "label_horizon_events": 20,
        "calibration_fraction": 0.20,
        "bootstrap_samples": 2000,
        "bootstrap_block_events": 40,
        "feature_stability_bins": 10,
        "max_archive_compressed_bytes": 268_435_456,
        "max_archive_uncompressed_bytes": 2_147_483_648,
        "max_total_download_bytes": 8_589_934_592,
    }
    for field, frozen in expected.items():
        _require_equal(getattr(study, field), frozen, f"study.{field}")
    if not (
        study.max_archive_compressed_bytes
        < study.max_archive_uncompressed_bytes
        < study.max_total_download_bytes
    ):
        raise M8ConfigError("study byte ceilings must increase from compressed to total")
    return study


def _parse_periods(raw: object) -> tuple[M8Period, ...]:
    items = _list(raw, "periods")
    periods: list[M8Period] = []
    for index, item in enumerate(items):
        table = _mapping(item, f"periods[{index}]")
        _exact_keys(table, _PERIOD_KEYS, f"periods[{index}]")
        raw_date = _text(table["date"], f"periods[{index}].date")
        try:
            parsed_date = Date.fromisoformat(raw_date)
        except ValueError as exc:
            raise M8ConfigError(f"periods[{index}].date must be an ISO UTC date") from exc
        if parsed_date.isoformat() != raw_date:
            raise M8ConfigError(f"periods[{index}].date must use canonical YYYY-MM-DD form")
        raw_role = _text(table["role"], f"periods[{index}].role")
        if raw_role not in {"train", "validation", "primary_test", "replication_test"}:
            raise M8ConfigError(f"periods[{index}].role is unsupported: {raw_role!r}")
        periods.append(M8Period(date=parsed_date, role=cast(M8PeriodRole, raw_role)))

    if len({period.date for period in periods}) != len(periods):
        raise M8ConfigError("period dates must be unique")
    observed = tuple((period.date.isoformat(), period.role) for period in periods)
    _require_equal(observed, _FROZEN_PERIODS, "period date/role order")
    return tuple(periods)


def _parse_features(raw: object) -> M8Features:
    table = _mapping(raw, "features")
    _exact_keys(table, _FEATURE_KEYS, "features")
    features = M8Features(
        trade_windows=_integer_tuple(table["trade_windows"], "features.trade_windows"),
        volatility_window=_integer(table["volatility_window"], "features.volatility_window"),
        intensity_window=_integer(table["intensity_window"], "features.intensity_window"),
        large_trade_quantile=_number(
            table["large_trade_quantile"], "features.large_trade_quantile"
        ),
    )
    expected: dict[str, object] = {
        "trade_windows": (5, 20, 100),
        "volatility_window": 100,
        "intensity_window": 50,
        "large_trade_quantile": 0.95,
    }
    for field, frozen in expected.items():
        _require_equal(getattr(features, field), frozen, f"features.{field}")
    return features


def _parse_models(raw: object) -> M8Models:
    table = _mapping(raw, "models")
    _exact_keys(table, _MODEL_KEYS, "models")
    models = M8Models(
        logistic_c_values=_number_tuple(table["logistic_c_values"], "models.logistic_c_values"),
        tree_max_depth_values=_integer_tuple(
            table["tree_max_depth_values"], "models.tree_max_depth_values"
        ),
        tree_min_samples_leaf=_integer(
            table["tree_min_samples_leaf"], "models.tree_min_samples_leaf"
        ),
    )
    expected: dict[str, object] = {
        "logistic_c_values": (0.1, 1.0, 10.0),
        "tree_max_depth_values": (2, 4, 6),
        "tree_min_samples_leaf": 40,
    }
    for field, frozen in expected.items():
        _require_equal(getattr(models, field), frozen, f"models.{field}")
    return models


def _parse_quality(raw: object) -> M8Quality:
    table = _mapping(raw, "quality")
    _exact_keys(table, _QUALITY_KEYS, "quality")
    quality = M8Quality(
        fail_on_error=_boolean(table["fail_on_error"], "quality.fail_on_error"),
        require_complete_daily_archive=_boolean(
            table["require_complete_daily_archive"],
            "quality.require_complete_daily_archive",
        ),
        require_contiguous_trade_ids_within_symbol_date=_boolean(
            table["require_contiguous_trade_ids_within_symbol_date"],
            "quality.require_contiguous_trade_ids_within_symbol_date",
        ),
        require_nondecreasing_event_time=_boolean(
            table["require_nondecreasing_event_time"],
            "quality.require_nondecreasing_event_time",
        ),
        allow_quality_warnings=_boolean(
            table["allow_quality_warnings"], "quality.allow_quality_warnings"
        ),
    )
    expected = {
        "fail_on_error": True,
        "require_complete_daily_archive": True,
        "require_contiguous_trade_ids_within_symbol_date": True,
        "require_nondecreasing_event_time": True,
        "allow_quality_warnings": False,
    }
    for field, frozen in expected.items():
        _require_equal(getattr(quality, field), frozen, f"quality.{field}")
    return quality


def _parse_claims(raw: object) -> M8Claims:
    table = _mapping(raw, "claims")
    _exact_keys(table, _CLAIM_KEYS, "claims")
    claims = M8Claims(
        allow_p_values=_boolean(table["allow_p_values"], "claims.allow_p_values"),
        allow_significance_claim=_boolean(
            table["allow_significance_claim"], "claims.allow_significance_claim"
        ),
        allow_cross_instrument_pooling=_boolean(
            table["allow_cross_instrument_pooling"],
            "claims.allow_cross_instrument_pooling",
        ),
        allow_execution_claim=_boolean(
            table["allow_execution_claim"], "claims.allow_execution_claim"
        ),
        allow_profitability_claim=_boolean(
            table["allow_profitability_claim"], "claims.allow_profitability_claim"
        ),
    )
    for field in _CLAIM_KEYS:
        _require_equal(getattr(claims, field), False, f"claims.{field}")
    return claims


def load_m8_config(path: str | Path) -> M8StudyConfig:
    """Load and validate the exact M8 protocol-v1.0.2 machine specification."""

    config_path = Path(path).resolve()
    source_bytes = config_path.read_bytes()
    try:
        decoded = source_bytes.decode("utf-8")
        raw = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise M8ConfigError(f"cannot parse M8 TOML configuration: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, _TOP_LEVEL_KEYS, "configuration")
    return M8StudyConfig(
        path=config_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        study=_parse_study(root["study"]),
        periods=_parse_periods(root["periods"]),
        features=_parse_features(root["features"]),
        models=_parse_models(root["models"]),
        quality=_parse_quality(root["quality"]),
        claims=_parse_claims(root["claims"]),
    )


__all__ = [
    "M8Claims",
    "M8ConfigError",
    "M8Features",
    "M8Models",
    "M8Period",
    "M8PeriodRole",
    "M8Quality",
    "M8Study",
    "M8StudyConfig",
    "load_m8_config",
]
