"""Typed project configuration with deterministic hashing."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from microstructure.data.schemas import SCHEMA_VERSION


class ConfigError(ValueError):
    """Raised when a project configuration is internally inconsistent."""


EvidenceTier = Literal["SYNTHETIC_SMOKE", "PUBLIC_SAMPLE_PARTIAL", "FULL_DATA"]
# Adapter modes are intentionally open-ended: ingestion owns the fail-closed
# registry, while configuration validates a stable identifier that third-party
# adapters can use.  Built-in modes retain their source-specific checks below.
DataMode = str
_DATA_MODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,63}")


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ConfigError(f"timestamp must include a UTC offset: {value!r}")
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    evidence_tier: EvidenceTier
    seed: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    mode: DataMode
    source: str
    symbols: tuple[str, ...]
    start: datetime
    end: datetime | None
    events_per_symbol: int | None
    max_events_per_symbol: int | None
    raw_root: Path
    partition_root: Path
    schema_version: str
    base_url: str
    request_limit: int
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class QualityConfig:
    max_spread_bps: float
    max_silence_ms: int
    fail_on_error: bool


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    trade_windows: tuple[int, ...]
    volatility_window: int
    intensity_window: int
    label_horizon_events: int
    large_trade_quantile: float


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    min_train_events: int
    validation_events: int
    test_events: int
    step_events: int
    embargo_events: int
    bootstrap_samples: int
    calibration_bins: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    selection_metric: str
    logistic_c_values: tuple[float, ...]
    tree_max_depth_values: tuple[int, ...]
    tree_min_samples_leaf: int


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    decision_latency_events: int
    order_latency_events: int
    maker_fee_bps: float
    taker_fee_bps: float
    half_spread_bps: float
    slippage_bps_per_unit: float
    signal_threshold: float
    max_position_units: float
    order_size_units: float
    limit_fill_base_probability: float
    queue_ahead_units: float
    limit_max_age_events: int
    cancel_latency_events: int
    liquidate_at_end: bool
    capacity_multipliers: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    path: Path
    project_root: Path
    run: RunConfig
    data: DataConfig
    quality: QualityConfig
    features: FeatureConfig
    evaluation: EvaluationConfig
    models: ModelConfig
    execution: ExecutionConfig
    canonical: Mapping[str, Any]

    @property
    def hash(self) -> str:
        """Return a stable SHA-256 hash of the source configuration."""
        payload = json.dumps(self.canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation with resolved paths and timestamps."""
        result = asdict(self)
        result.pop("canonical")
        result["path"] = str(self.path)
        result["project_root"] = str(self.project_root)
        data = cast(dict[str, Any], result["data"])
        data["start"] = self.data.start.isoformat().replace("+00:00", "Z")
        data["end"] = self.data.end.isoformat().replace("+00:00", "Z") if self.data.end else None
        data["raw_root"] = str(self.data.raw_root)
        data["partition_root"] = str(self.data.partition_root)
        return result


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing TOML section [{name}]")
    return cast(Mapping[str, Any], value)


def _resolve(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project TOML configuration."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    project_root = config_path.parent.parent.resolve()
    run_raw = _section(raw, "run")
    data_raw = _section(raw, "data")
    quality_raw = _section(raw, "quality")
    feature_raw = _section(raw, "features")
    evaluation_raw = _section(raw, "evaluation")
    model_raw = _section(raw, "models")
    execution_raw = _section(raw, "execution")

    evidence_tier = str(run_raw["evidence_tier"])
    if evidence_tier not in {"SYNTHETIC_SMOKE", "PUBLIC_SAMPLE_PARTIAL", "FULL_DATA"}:
        raise ConfigError(f"unsupported evidence tier: {evidence_tier}")
    mode = str(data_raw["mode"])
    if _DATA_MODE_PATTERN.fullmatch(mode) is None:
        raise ConfigError(
            "data.mode must be a lowercase adapter identifier containing only "
            "letters, digits, underscores, dots, or hyphens"
        )

    start = _utc_datetime(str(data_raw["start"]))
    end_value = data_raw.get("end")
    end = _utc_datetime(str(end_value)) if end_value is not None else None
    if end is not None and end <= start:
        raise ConfigError("data.end must be after data.start")

    run = RunConfig(
        name=str(run_raw["name"]),
        evidence_tier=cast(EvidenceTier, evidence_tier),
        seed=int(run_raw["seed"]),
    )
    data = DataConfig(
        mode=mode,
        source=str(data_raw["source"]),
        symbols=tuple(str(symbol).upper() for symbol in data_raw["symbols"]),
        start=start,
        end=end,
        events_per_symbol=(
            int(data_raw["events_per_symbol"])
            if data_raw.get("events_per_symbol") is not None
            else None
        ),
        max_events_per_symbol=(
            int(data_raw["max_events_per_symbol"])
            if data_raw.get("max_events_per_symbol") is not None
            else None
        ),
        raw_root=_resolve(project_root, str(data_raw.get("raw_root", "data/raw"))),
        partition_root=_resolve(project_root, str(data_raw["partition_root"])),
        schema_version=str(data_raw["schema_version"]),
        base_url=str(data_raw.get("base_url", "https://data-api.binance.vision")).rstrip("/"),
        request_limit=int(data_raw.get("request_limit", 1000)),
        timeout_seconds=float(data_raw.get("timeout_seconds", 30.0)),
        max_retries=int(data_raw.get("max_retries", 5)),
    )
    quality = QualityConfig(
        max_spread_bps=float(quality_raw["max_spread_bps"]),
        max_silence_ms=int(quality_raw["max_silence_ms"]),
        fail_on_error=bool(quality_raw["fail_on_error"]),
    )
    features = FeatureConfig(
        trade_windows=tuple(int(window) for window in feature_raw["trade_windows"]),
        volatility_window=int(feature_raw["volatility_window"]),
        intensity_window=int(feature_raw["intensity_window"]),
        label_horizon_events=int(feature_raw["label_horizon_events"]),
        large_trade_quantile=float(feature_raw["large_trade_quantile"]),
    )
    evaluation = EvaluationConfig(
        min_train_events=int(evaluation_raw["min_train_events"]),
        validation_events=int(evaluation_raw["validation_events"]),
        test_events=int(evaluation_raw["test_events"]),
        step_events=int(evaluation_raw["step_events"]),
        embargo_events=int(evaluation_raw["embargo_events"]),
        bootstrap_samples=int(evaluation_raw["bootstrap_samples"]),
        calibration_bins=int(evaluation_raw["calibration_bins"]),
    )
    models = ModelConfig(
        selection_metric=str(model_raw["selection_metric"]),
        logistic_c_values=tuple(float(value) for value in model_raw["logistic_c_values"]),
        tree_max_depth_values=tuple(int(value) for value in model_raw["tree_max_depth_values"]),
        tree_min_samples_leaf=int(model_raw["tree_min_samples_leaf"]),
    )
    execution = ExecutionConfig(
        decision_latency_events=int(execution_raw["decision_latency_events"]),
        order_latency_events=int(execution_raw["order_latency_events"]),
        maker_fee_bps=float(execution_raw["maker_fee_bps"]),
        taker_fee_bps=float(execution_raw["taker_fee_bps"]),
        half_spread_bps=float(execution_raw["half_spread_bps"]),
        slippage_bps_per_unit=float(execution_raw["slippage_bps_per_unit"]),
        signal_threshold=float(execution_raw["signal_threshold"]),
        max_position_units=float(execution_raw["max_position_units"]),
        order_size_units=float(execution_raw["order_size_units"]),
        limit_fill_base_probability=float(execution_raw["limit_fill_base_probability"]),
        queue_ahead_units=float(execution_raw["queue_ahead_units"]),
        limit_max_age_events=int(execution_raw["limit_max_age_events"]),
        cancel_latency_events=int(execution_raw["cancel_latency_events"]),
        liquidate_at_end=bool(execution_raw["liquidate_at_end"]),
        capacity_multipliers=tuple(float(value) for value in execution_raw["capacity_multipliers"]),
    )

    if not data.symbols:
        raise ConfigError("data.symbols must not be empty")
    if data.mode == "synthetic" and (data.events_per_symbol is None or data.events_per_symbol < 1):
        raise ConfigError("synthetic mode requires positive data.events_per_symbol")
    if data.mode == "synthetic" and run.evidence_tier != "SYNTHETIC_SMOKE":
        raise ConfigError("synthetic inputs must use the SYNTHETIC_SMOKE evidence tier")
    if data.mode == "binance_rest" and run.evidence_tier == "SYNTHETIC_SMOKE":
        raise ConfigError("public inputs cannot use the SYNTHETIC_SMOKE evidence tier")
    if data.mode == "binance_rest" and data.end is None:
        raise ConfigError("binance_rest mode requires a bounded data.end")
    if data.mode == "binance_rest" and (
        data.max_events_per_symbol is None or data.max_events_per_symbol < 1
    ):
        raise ConfigError("binance_rest mode requires positive data.max_events_per_symbol")
    if data.schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported data.schema_version {data.schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    if not all(window > 1 for window in features.trade_windows):
        raise ConfigError("all feature trade windows must exceed one event")
    if features.label_horizon_events < 1:
        raise ConfigError("label_horizon_events must be positive")
    if evaluation.embargo_events < features.label_horizon_events:
        raise ConfigError("embargo_events must cover label_horizon_events")
    if not 0.5 < execution.signal_threshold < 1.0:
        raise ConfigError("signal_threshold must be between 0.5 and 1.0")
    if not 0.0 <= execution.limit_fill_base_probability <= 1.0:
        raise ConfigError("limit_fill_base_probability must be in [0, 1]")
    if execution.limit_max_age_events < 1 or execution.cancel_latency_events < 0:
        raise ConfigError("limit order age must be positive and cancel latency nonnegative")
    if execution.decision_latency_events < 0 or execution.order_latency_events < 0:
        raise ConfigError("decision and order latency must be nonnegative")
    if execution.max_position_units <= 0 or execution.order_size_units <= 0:
        raise ConfigError("execution position and order sizes must be positive")
    execution_floats = (
        execution.maker_fee_bps,
        execution.taker_fee_bps,
        execution.half_spread_bps,
        execution.slippage_bps_per_unit,
        execution.max_position_units,
        execution.order_size_units,
        execution.queue_ahead_units,
    )
    if not all(math.isfinite(value) for value in execution_floats):
        raise ConfigError("execution numeric assumptions must be finite")
    if execution.queue_ahead_units < 0:
        raise ConfigError("execution queue_ahead_units must be nonnegative")
    if execution.half_spread_bps < 0 or execution.slippage_bps_per_unit < 0:
        raise ConfigError("execution spread and slippage assumptions must be nonnegative")
    if not execution.capacity_multipliers or not all(
        math.isfinite(value) and value > 0 for value in execution.capacity_multipliers
    ):
        raise ConfigError("execution capacity_multipliers must be finite and positive")
    if not 0.0 < features.large_trade_quantile < 1.0:
        raise ConfigError("large_trade_quantile must lie strictly between zero and one")

    return ProjectConfig(
        path=config_path,
        project_root=project_root,
        run=run,
        data=data,
        quality=quality,
        features=features,
        evaluation=evaluation,
        models=models,
        execution=execution,
        canonical=raw,
    )


def datetime_to_ns(value: datetime) -> int:
    """Convert an aware UTC datetime to integer epoch nanoseconds."""
    if value.tzinfo is None:
        raise ConfigError("datetime must be timezone aware")
    return int(value.timestamp() * 1_000_000_000)
