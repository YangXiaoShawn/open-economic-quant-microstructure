"""Fail-closed parser for the frozen prospective M8 live-L2 study."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Literal, cast

M8L2SessionRole = Literal["train", "validation", "primary_test", "replication_test"]

M8_L2_FREEZE_COMMIT = "6db6c8cf81b726069d1833672864e0554976b985"
M8_L2_CONFIG_SOURCE_SHA256 = "b1bf3b4e2820e24e4555bfeb9cb0957f9a0bcdef62039f7d92360e0a97d0dd39"
M8_L2_PROTOCOL_SHA256 = "4c77a2099a4cabd049d10e0f8264d3b4c66704d8e87cbaf0c817fd085f4bbd83"

_TOP_LEVEL_KEYS = frozenset(
    {"study", "sessions", "capture", "features", "models", "execution", "claims"}
)
_STUDY_KEYS = frozenset(
    {
        "name",
        "protocol_version",
        "evidence_tier",
        "seed",
        "source",
        "symbols",
        "stream_interval_ms",
    }
)
_SESSION_KEYS = frozenset({"date", "start_utc", "end_utc", "role"})
_CAPTURE_KEYS = frozenset(
    {
        "duration_seconds",
        "max_messages_per_symbol",
        "max_raw_frame_bytes",
        "max_arrow_batch_bytes",
        "min_overlapping_coverage_seconds",
        "min_single_continuity_epoch_seconds",
        "require_complete_status",
        "require_live_reconstruction",
        "max_sequence_gaps",
        "max_quality_errors",
        "max_quality_warnings",
    }
)
_FEATURE_KEYS = frozenset(
    {
        "depth_levels",
        "event_horizons",
        "clock_horizons_ms",
        "include_spread",
        "include_depth",
        "include_ofi",
        "include_queue_imbalance",
        "include_microprice",
        "include_cancellation_intensity",
        "include_realized_volatility",
        "include_reference_fit_regimes",
    }
)
_MODEL_KEYS = frozenset(
    {
        "selection_metric",
        "logistic_c_values",
        "tree_max_depth_values",
        "tree_min_samples_leaf",
        "calibration_fraction",
        "bootstrap_samples",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "market_orders_only",
        "taker_fee_bps",
        "decision_latency_events",
        "order_latency_events",
        "liquidate_at_end",
        "allow_limit_fill_claim",
        "allow_capacity_claim",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "allow_p_values",
        "allow_significance_claim",
        "allow_realized_execution_claim",
        "allow_profitability_claim",
    }
)

_FROZEN_SESSIONS: tuple[tuple[str, str, str, M8L2SessionRole], ...] = (
    ("2026-08-10", "14:00:00", "15:00:00", "train"),
    ("2026-08-11", "14:00:00", "15:00:00", "validation"),
    ("2026-08-12", "14:00:00", "15:00:00", "primary_test"),
    ("2026-08-13", "14:00:00", "15:00:00", "replication_test"),
)


class M8L2ConfigError(ValueError):
    """Raised when the live-L2 configuration differs from its frozen contract."""


@dataclass(frozen=True, slots=True)
class M8L2Study:
    name: str
    protocol_version: str
    evidence_tier: str
    seed: int
    source: str
    symbols: tuple[str, ...]
    stream_interval_ms: int


@dataclass(frozen=True, slots=True)
class M8L2Session:
    date: date
    start_utc: time
    end_utc: time
    role: M8L2SessionRole

    @property
    def start(self) -> datetime:
        return datetime.combine(self.date, self.start_utc, tzinfo=UTC)

    @property
    def end(self) -> datetime:
        return datetime.combine(self.date, self.end_utc, tzinfo=UTC)

    @property
    def start_ns(self) -> int:
        return int(self.start.timestamp()) * 1_000_000_000

    @property
    def end_ns(self) -> int:
        return int(self.end.timestamp()) * 1_000_000_000


@dataclass(frozen=True, slots=True)
class M8L2CaptureLimits:
    duration_seconds: int
    max_messages_per_symbol: int
    max_raw_frame_bytes: int
    max_arrow_batch_bytes: int
    min_overlapping_coverage_seconds: int
    min_single_continuity_epoch_seconds: int
    require_complete_status: bool
    require_live_reconstruction: bool
    max_sequence_gaps: int
    max_quality_errors: int
    max_quality_warnings: int


@dataclass(frozen=True, slots=True)
class M8L2Features:
    depth_levels: tuple[int, ...]
    event_horizons: tuple[int, ...]
    clock_horizons_ms: tuple[int, ...]
    include_spread: bool
    include_depth: bool
    include_ofi: bool
    include_queue_imbalance: bool
    include_microprice: bool
    include_cancellation_intensity: bool
    include_realized_volatility: bool
    include_reference_fit_regimes: bool


@dataclass(frozen=True, slots=True)
class M8L2Models:
    selection_metric: str
    logistic_c_values: tuple[float, ...]
    tree_max_depth_values: tuple[int, ...]
    tree_min_samples_leaf: int
    calibration_fraction: float
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class M8L2Execution:
    market_orders_only: bool
    taker_fee_bps: float
    decision_latency_events: tuple[int, ...]
    order_latency_events: tuple[int, ...]
    liquidate_at_end: bool
    allow_limit_fill_claim: bool
    allow_capacity_claim: bool


@dataclass(frozen=True, slots=True)
class M8L2Claims:
    allow_p_values: bool
    allow_significance_claim: bool
    allow_realized_execution_claim: bool
    allow_profitability_claim: bool


@dataclass(frozen=True, slots=True)
class M8L2StudyConfig:
    path: Path
    source_sha256: str
    study: M8L2Study
    sessions: tuple[M8L2Session, ...]
    capture: M8L2CaptureLimits
    features: M8L2Features
    models: M8L2Models
    execution: M8L2Execution
    claims: M8L2Claims

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "study": {
                "name": self.study.name,
                "protocol_version": self.study.protocol_version,
                "evidence_tier": self.study.evidence_tier,
                "seed": self.study.seed,
                "source": self.study.source,
                "symbols": list(self.study.symbols),
                "stream_interval_ms": self.study.stream_interval_ms,
            },
            "sessions": [
                {
                    "date": item.date.isoformat(),
                    "start_utc": item.start_utc.isoformat(),
                    "end_utc": item.end_utc.isoformat(),
                    "role": item.role,
                }
                for item in self.sessions
            ],
            "capture": {
                name: getattr(self.capture, name) for name in self.capture.__dataclass_fields__
            },
            "features": {
                "depth_levels": list(self.features.depth_levels),
                "event_horizons": list(self.features.event_horizons),
                "clock_horizons_ms": list(self.features.clock_horizons_ms),
                **{
                    name: getattr(self.features, name)
                    for name in self.features.__dataclass_fields__
                    if name.startswith("include_")
                },
            },
            "models": {
                "selection_metric": self.models.selection_metric,
                "logistic_c_values": list(self.models.logistic_c_values),
                "tree_max_depth_values": list(self.models.tree_max_depth_values),
                "tree_min_samples_leaf": self.models.tree_min_samples_leaf,
                "calibration_fraction": self.models.calibration_fraction,
                "bootstrap_samples": self.models.bootstrap_samples,
            },
            "execution": {
                "market_orders_only": self.execution.market_orders_only,
                "taker_fee_bps": self.execution.taker_fee_bps,
                "decision_latency_events": list(self.execution.decision_latency_events),
                "order_latency_events": list(self.execution.order_latency_events),
                "liquidate_at_end": self.execution.liquidate_at_end,
                "allow_limit_fill_claim": self.execution.allow_limit_fill_claim,
                "allow_capacity_claim": self.execution.allow_capacity_claim,
            },
            "claims": {
                name: getattr(self.claims, name) for name in self.claims.__dataclass_fields__
            },
        }

    @property
    def hash(self) -> str:
        encoded = json.dumps(
            self._semantic_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def public_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "config_sha256": self.hash,
            "source_sha256": self.source_sha256,
            **self._semantic_payload(),
        }

    def session_for_date(self, value: str | date) -> M8L2Session:
        requested = date.fromisoformat(value) if isinstance(value, str) else value
        matches = [item for item in self.sessions if item.date == requested]
        if len(matches) != 1:
            raise M8L2ConfigError(f"date is not a frozen live-L2 session: {requested.isoformat()}")
        return matches[0]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2ConfigError(f"{label} must be a TOML table with string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise M8L2ConfigError(
            f"{label} keys differ (missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)})"
        )


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise M8L2ConfigError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise M8L2ConfigError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise M8L2ConfigError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise M8L2ConfigError(f"{label} must be a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise M8L2ConfigError(f"{label} must be a finite number")
    return result


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise M8L2ConfigError(f"{label} must be a boolean")
    return value


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label))
    )


def _number_tuple(value: object, label: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label))
    )


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label)))


def _frozen(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise M8L2ConfigError(f"{label} is frozen at {expected!r}, observed {observed!r}")


def _parse_clock(value: object, label: str) -> time:
    raw = _text(value, label)
    try:
        parsed = time.fromisoformat(raw)
    except ValueError as error:
        raise M8L2ConfigError(f"{label} must use HH:MM:SS") from error
    if parsed.tzinfo is not None or parsed.microsecond or parsed.isoformat() != raw:
        raise M8L2ConfigError(f"{label} must use canonical UTC HH:MM:SS")
    return parsed


def _parse_study(raw: object) -> M8L2Study:
    table = _mapping(raw, "study")
    _exact_keys(table, _STUDY_KEYS, "study")
    result = M8L2Study(
        name=_text(table["name"], "study.name"),
        protocol_version=_text(table["protocol_version"], "study.protocol_version"),
        evidence_tier=_text(table["evidence_tier"], "study.evidence_tier"),
        seed=_integer(table["seed"], "study.seed"),
        source=_text(table["source"], "study.source"),
        symbols=tuple(item.upper() for item in _text_tuple(table["symbols"], "study.symbols")),
        stream_interval_ms=_integer(table["stream_interval_ms"], "study.stream_interval_ms"),
    )
    expected: dict[str, object] = {
        "name": "binance-m8-live-l2-study-v2",
        "protocol_version": "2.0.0",
        "evidence_tier": "FULL_DATA",
        "seed": 20260807,
        "source": "binance_spot_live_diff_depth_100ms",
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "stream_interval_ms": 100,
    }
    for name, value in expected.items():
        _frozen(getattr(result, name), value, f"study.{name}")
    return result


def _parse_sessions(raw: object) -> tuple[M8L2Session, ...]:
    result: list[M8L2Session] = []
    for index, item in enumerate(_list(raw, "sessions")):
        table = _mapping(item, f"sessions[{index}]")
        _exact_keys(table, _SESSION_KEYS, f"sessions[{index}]")
        raw_date = _text(table["date"], f"sessions[{index}].date")
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as error:
            raise M8L2ConfigError(f"sessions[{index}].date must use YYYY-MM-DD") from error
        if parsed_date.isoformat() != raw_date:
            raise M8L2ConfigError(f"sessions[{index}].date must use canonical YYYY-MM-DD")
        role = _text(table["role"], f"sessions[{index}].role")
        if role not in {"train", "validation", "primary_test", "replication_test"}:
            raise M8L2ConfigError(f"sessions[{index}].role is unsupported")
        session = M8L2Session(
            date=parsed_date,
            start_utc=_parse_clock(table["start_utc"], f"sessions[{index}].start_utc"),
            end_utc=_parse_clock(table["end_utc"], f"sessions[{index}].end_utc"),
            role=cast(M8L2SessionRole, role),
        )
        if session.end <= session.start:
            raise M8L2ConfigError(f"sessions[{index}] end must be after start on the same UTC date")
        result.append(session)
    observed = tuple(
        (item.date.isoformat(), item.start_utc.isoformat(), item.end_utc.isoformat(), item.role)
        for item in result
    )
    _frozen(observed, _FROZEN_SESSIONS, "session calendar/order")
    return tuple(result)


def _parse_capture(raw: object) -> M8L2CaptureLimits:
    table = _mapping(raw, "capture")
    _exact_keys(table, _CAPTURE_KEYS, "capture")
    result = M8L2CaptureLimits(
        duration_seconds=_integer(table["duration_seconds"], "capture.duration_seconds"),
        max_messages_per_symbol=_integer(
            table["max_messages_per_symbol"], "capture.max_messages_per_symbol"
        ),
        max_raw_frame_bytes=_integer(table["max_raw_frame_bytes"], "capture.max_raw_frame_bytes"),
        max_arrow_batch_bytes=_integer(
            table["max_arrow_batch_bytes"], "capture.max_arrow_batch_bytes"
        ),
        min_overlapping_coverage_seconds=_integer(
            table["min_overlapping_coverage_seconds"],
            "capture.min_overlapping_coverage_seconds",
        ),
        min_single_continuity_epoch_seconds=_integer(
            table["min_single_continuity_epoch_seconds"],
            "capture.min_single_continuity_epoch_seconds",
        ),
        require_complete_status=_boolean(
            table["require_complete_status"], "capture.require_complete_status"
        ),
        require_live_reconstruction=_boolean(
            table["require_live_reconstruction"], "capture.require_live_reconstruction"
        ),
        max_sequence_gaps=_integer(table["max_sequence_gaps"], "capture.max_sequence_gaps"),
        max_quality_errors=_integer(table["max_quality_errors"], "capture.max_quality_errors"),
        max_quality_warnings=_integer(
            table["max_quality_warnings"], "capture.max_quality_warnings"
        ),
    )
    expected = M8L2CaptureLimits(
        duration_seconds=3600,
        max_messages_per_symbol=60000,
        max_raw_frame_bytes=1048576,
        max_arrow_batch_bytes=16777216,
        min_overlapping_coverage_seconds=3300,
        min_single_continuity_epoch_seconds=1800,
        require_complete_status=True,
        require_live_reconstruction=True,
        max_sequence_gaps=0,
        max_quality_errors=0,
        max_quality_warnings=0,
    )
    _frozen(result, expected, "capture contract")
    return result


def _parse_features(raw: object) -> M8L2Features:
    table = _mapping(raw, "features")
    _exact_keys(table, _FEATURE_KEYS, "features")
    result = M8L2Features(
        depth_levels=_int_tuple(table["depth_levels"], "features.depth_levels"),
        event_horizons=_int_tuple(table["event_horizons"], "features.event_horizons"),
        clock_horizons_ms=_int_tuple(table["clock_horizons_ms"], "features.clock_horizons_ms"),
        include_spread=_boolean(table["include_spread"], "features.include_spread"),
        include_depth=_boolean(table["include_depth"], "features.include_depth"),
        include_ofi=_boolean(table["include_ofi"], "features.include_ofi"),
        include_queue_imbalance=_boolean(
            table["include_queue_imbalance"], "features.include_queue_imbalance"
        ),
        include_microprice=_boolean(table["include_microprice"], "features.include_microprice"),
        include_cancellation_intensity=_boolean(
            table["include_cancellation_intensity"], "features.include_cancellation_intensity"
        ),
        include_realized_volatility=_boolean(
            table["include_realized_volatility"], "features.include_realized_volatility"
        ),
        include_reference_fit_regimes=_boolean(
            table["include_reference_fit_regimes"], "features.include_reference_fit_regimes"
        ),
    )
    expected = M8L2Features(
        depth_levels=(1, 5, 10),
        event_horizons=(20, 100),
        clock_horizons_ms=(1000, 5000),
        include_spread=True,
        include_depth=True,
        include_ofi=True,
        include_queue_imbalance=True,
        include_microprice=True,
        include_cancellation_intensity=True,
        include_realized_volatility=True,
        include_reference_fit_regimes=True,
    )
    _frozen(result, expected, "features contract")
    return result


def _parse_models(raw: object) -> M8L2Models:
    table = _mapping(raw, "models")
    _exact_keys(table, _MODEL_KEYS, "models")
    result = M8L2Models(
        selection_metric=_text(table["selection_metric"], "models.selection_metric"),
        logistic_c_values=_number_tuple(table["logistic_c_values"], "models.logistic_c_values"),
        tree_max_depth_values=_int_tuple(
            table["tree_max_depth_values"], "models.tree_max_depth_values"
        ),
        tree_min_samples_leaf=_integer(
            table["tree_min_samples_leaf"], "models.tree_min_samples_leaf"
        ),
        calibration_fraction=_number(table["calibration_fraction"], "models.calibration_fraction"),
        bootstrap_samples=_integer(table["bootstrap_samples"], "models.bootstrap_samples"),
    )
    expected = M8L2Models(
        selection_metric="log_loss",
        logistic_c_values=(0.1, 1.0, 10.0),
        tree_max_depth_values=(2, 4, 6),
        tree_min_samples_leaf=40,
        calibration_fraction=0.20,
        bootstrap_samples=2000,
    )
    _frozen(result, expected, "models contract")
    return result


def _parse_execution(raw: object) -> M8L2Execution:
    table = _mapping(raw, "execution")
    _exact_keys(table, _EXECUTION_KEYS, "execution")
    result = M8L2Execution(
        market_orders_only=_boolean(table["market_orders_only"], "execution.market_orders_only"),
        taker_fee_bps=_number(table["taker_fee_bps"], "execution.taker_fee_bps"),
        decision_latency_events=_int_tuple(
            table["decision_latency_events"], "execution.decision_latency_events"
        ),
        order_latency_events=_int_tuple(
            table["order_latency_events"], "execution.order_latency_events"
        ),
        liquidate_at_end=_boolean(table["liquidate_at_end"], "execution.liquidate_at_end"),
        allow_limit_fill_claim=_boolean(
            table["allow_limit_fill_claim"], "execution.allow_limit_fill_claim"
        ),
        allow_capacity_claim=_boolean(
            table["allow_capacity_claim"], "execution.allow_capacity_claim"
        ),
    )
    expected = M8L2Execution(
        market_orders_only=True,
        taker_fee_bps=4.0,
        decision_latency_events=(0, 1, 5),
        order_latency_events=(0, 1, 5),
        liquidate_at_end=True,
        allow_limit_fill_claim=False,
        allow_capacity_claim=False,
    )
    _frozen(result, expected, "execution contract")
    return result


def _parse_claims(raw: object) -> M8L2Claims:
    table = _mapping(raw, "claims")
    _exact_keys(table, _CLAIM_KEYS, "claims")
    result = M8L2Claims(
        allow_p_values=_boolean(table["allow_p_values"], "claims.allow_p_values"),
        allow_significance_claim=_boolean(
            table["allow_significance_claim"], "claims.allow_significance_claim"
        ),
        allow_realized_execution_claim=_boolean(
            table["allow_realized_execution_claim"], "claims.allow_realized_execution_claim"
        ),
        allow_profitability_claim=_boolean(
            table["allow_profitability_claim"], "claims.allow_profitability_claim"
        ),
    )
    _frozen(result, M8L2Claims(False, False, False, False), "claims contract")
    return result


def load_m8_l2_config(path: str | Path) -> M8L2StudyConfig:
    """Load the exact outcome-blind protocol-v1.0.0 live-L2 configuration."""

    config_path = Path(path).resolve()
    source = config_path.read_bytes()
    try:
        raw = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise M8L2ConfigError(f"cannot parse M8 live-L2 TOML: {error}") from error
    root = _mapping(raw, "configuration")
    _exact_keys(root, _TOP_LEVEL_KEYS, "configuration")
    result = M8L2StudyConfig(
        path=config_path,
        source_sha256=hashlib.sha256(source).hexdigest(),
        study=_parse_study(root["study"]),
        sessions=_parse_sessions(root["sessions"]),
        capture=_parse_capture(root["capture"]),
        features=_parse_features(root["features"]),
        models=_parse_models(root["models"]),
        execution=_parse_execution(root["execution"]),
        claims=_parse_claims(root["claims"]),
    )
    if result.source_sha256 != M8_L2_CONFIG_SOURCE_SHA256:
        raise M8L2ConfigError(
            "configuration bytes do not match the outcome-blind freeze "
            f"{M8_L2_CONFIG_SOURCE_SHA256}"
        )
    return result


__all__ = [
    "M8_L2_CONFIG_SOURCE_SHA256",
    "M8_L2_FREEZE_COMMIT",
    "M8_L2_PROTOCOL_SHA256",
    "M8L2CaptureLimits",
    "M8L2Claims",
    "M8L2ConfigError",
    "M8L2Execution",
    "M8L2Features",
    "M8L2Models",
    "M8L2Session",
    "M8L2SessionRole",
    "M8L2Study",
    "M8L2StudyConfig",
    "load_m8_l2_config",
]
