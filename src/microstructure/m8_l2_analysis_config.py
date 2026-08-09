"""Exact, outcome-blind contract for the frozen M8 live-L2 analysis.

The capture configuration controls what is observed.  This independent file
controls how already-verified session bundles may be analysed.  The authority
loader requires both the frozen semantics and the exact reviewed TOML bytes;
the semantic-hash helper exists only for provenance comparisons and does not
authorize a differently encoded configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

M8L2AnalysisRole = Literal["train", "validation", "primary_test", "replication_test"]
M8L2EndpointDomain = Literal["event", "clock"]
M8L2EndpointUnit = Literal["events", "milliseconds"]

M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256 = (
    "0d786d5f4109bb5bf773a6197df3fa861c9b7eb61c16c957bd49fb56147fd7d8"
)
M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256 = (
    "17c91f64765f35195ab03a4caac93d8ff9c5f009c16e785fd84ebd9569d6f84b"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "study",
        "features",
        "endpoints",
        "regimes",
        "calibration",
        "bootstrap",
        "signed_impact",
        "execution",
        "claims",
    }
)
_STUDY_KEYS = frozenset(
    {
        "name",
        "protocol_version",
        "seed",
        "source",
        "capture_config_source_sha256",
        "capture_protocol_sha256",
        "symbols",
        "training_role",
        "selection_role",
        "primary_endpoint_role",
        "replication_endpoint_role",
    }
)
_FEATURE_KEYS = frozenset(
    {
        "decision_scope",
        "flat_direction_policy",
        "rolling_windows",
        "volatility_window",
        "model_feature_columns",
        "clock_max_state_age_ms",
        "clock_target_policy",
        "clock_label_information_end",
        "clock_record_target_sequence",
        "clock_censor_if_no_eligible_state",
    }
)
_ENDPOINT_KEYS = frozenset(
    {
        "name",
        "domain",
        "horizon_value",
        "unit",
        "paired_block_width",
        "paired_block_unit",
        "nominal_event_block_width",
    }
)
_REGIME_KEYS = frozenset({"fit_role", "feature", "quantile_numerators", "quantile_denominator"})
_CALIBRATION_KEYS = frozenset({"bins"})
_BOOTSTRAP_KEYS = frozenset({"method", "samples"})
_SIGNED_IMPACT_KEYS = frozenset({"metric", "side_rule", "price_rule"})
_EXECUTION_KEYS = frozenset(
    {
        "market_orders_only",
        "probability_threshold",
        "symmetric_probability_thresholds",
        "order_notional_usd",
        "max_l1_participation",
        "inventory_order_multiples",
        "reference_price_fit_role",
        "reference_depth_fit_role",
        "reference_price_statistic",
        "reference_depth_statistic",
        "reference_quantity_policy",
        "l1_fill_policy",
        "scenario_reset_policy",
        "extra_slippage_bps",
        "liquidate_at_end",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "allow_capacity_claim",
        "allow_realized_execution_claim",
        "allow_profitability_claim",
    }
)

_CAPTURE_CONFIG_SOURCE_SHA256 = "b1bf3b4e2820e24e4555bfeb9cb0957f9a0bcdef62039f7d92360e0a97d0dd39"
_CAPTURE_PROTOCOL_SHA256 = "4c77a2099a4cabd049d10e0f8264d3b4c66704d8e87cbaf0c817fd085f4bbd83"
_MODEL_FEATURE_COLUMNS = (
    "spread_bps",
    "depth_total_l1",
    "depth_total_l5",
    "depth_total_l10",
    "queue_imbalance_l1",
    "queue_imbalance_l5",
    "queue_imbalance_l10",
    "microprice_deviation_bps",
    "ofi_l1",
    "ofi_w20",
    "ofi_w100",
    "cancellation_intensity_w20",
    "cancellation_intensity_w100",
    "realized_volatility_w20",
    "realized_volatility_w100",
    "volatility_regime_low",
    "volatility_regime_high",
    "liquidity_regime_liquid",
    "liquidity_regime_stressed",
)


class M8L2AnalysisConfigError(ValueError):
    """Raised when an analysis file differs from the frozen contract."""


@dataclass(frozen=True, slots=True)
class M8L2AnalysisStudy:
    name: str
    protocol_version: str
    seed: int
    source: str
    capture_config_source_sha256: str
    capture_protocol_sha256: str
    symbols: tuple[str, ...]
    training_role: M8L2AnalysisRole
    selection_role: M8L2AnalysisRole
    primary_endpoint_role: M8L2AnalysisRole
    replication_endpoint_role: M8L2AnalysisRole


@dataclass(frozen=True, slots=True)
class M8L2AnalysisFeatures:
    decision_scope: str
    flat_direction_policy: str
    rolling_windows: tuple[int, ...]
    volatility_window: int
    model_feature_columns: tuple[str, ...]
    clock_max_state_age_ms: int
    clock_target_policy: str
    clock_label_information_end: str
    clock_record_target_sequence: bool
    clock_censor_if_no_eligible_state: bool


@dataclass(frozen=True, slots=True)
class M8L2AnalysisEndpoint:
    name: str
    domain: M8L2EndpointDomain
    horizon_value: int
    unit: M8L2EndpointUnit
    paired_block_width: int
    paired_block_unit: M8L2EndpointUnit
    nominal_event_block_width: int


@dataclass(frozen=True, slots=True)
class M8L2AnalysisRegimes:
    fit_role: M8L2AnalysisRole
    feature: str
    quantile_numerators: tuple[int, ...]
    quantile_denominator: int


@dataclass(frozen=True, slots=True)
class M8L2AnalysisCalibration:
    bins: int


@dataclass(frozen=True, slots=True)
class M8L2AnalysisBootstrap:
    method: str
    samples: int


@dataclass(frozen=True, slots=True)
class M8L2AnalysisSignedImpact:
    metric: str
    side_rule: str
    price_rule: str


@dataclass(frozen=True, slots=True)
class M8L2AnalysisExecution:
    market_orders_only: bool
    probability_threshold: float
    symmetric_probability_thresholds: bool
    order_notional_usd: float
    max_l1_participation: float
    inventory_order_multiples: int
    reference_price_fit_role: M8L2AnalysisRole
    reference_depth_fit_role: M8L2AnalysisRole
    reference_price_statistic: str
    reference_depth_statistic: str
    reference_quantity_policy: str
    l1_fill_policy: str
    scenario_reset_policy: str
    extra_slippage_bps: float
    liquidate_at_end: bool


@dataclass(frozen=True, slots=True)
class M8L2AnalysisClaims:
    allow_capacity_claim: bool
    allow_realized_execution_claim: bool
    allow_profitability_claim: bool


@dataclass(frozen=True, slots=True)
class M8L2AnalysisConfig:
    """Typed analysis contract, with separate semantic and exact-byte identities."""

    path: Path
    source_sha256: str
    study: M8L2AnalysisStudy
    features: M8L2AnalysisFeatures
    endpoints: tuple[M8L2AnalysisEndpoint, ...]
    regimes: M8L2AnalysisRegimes
    calibration: M8L2AnalysisCalibration
    bootstrap: M8L2AnalysisBootstrap
    signed_impact: M8L2AnalysisSignedImpact
    execution: M8L2AnalysisExecution
    claims: M8L2AnalysisClaims

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "study": {name: getattr(self.study, name) for name in self.study.__dataclass_fields__},
            "features": {
                "decision_scope": self.features.decision_scope,
                "flat_direction_policy": self.features.flat_direction_policy,
                "rolling_windows": list(self.features.rolling_windows),
                "volatility_window": self.features.volatility_window,
                "model_feature_columns": list(self.features.model_feature_columns),
                "clock_max_state_age_ms": self.features.clock_max_state_age_ms,
                "clock_target_policy": self.features.clock_target_policy,
                "clock_label_information_end": self.features.clock_label_information_end,
                "clock_record_target_sequence": self.features.clock_record_target_sequence,
                "clock_censor_if_no_eligible_state": (
                    self.features.clock_censor_if_no_eligible_state
                ),
            },
            "endpoints": [
                {name: getattr(endpoint, name) for name in endpoint.__dataclass_fields__}
                for endpoint in self.endpoints
            ],
            "regimes": {
                "fit_role": self.regimes.fit_role,
                "feature": self.regimes.feature,
                "quantile_numerators": list(self.regimes.quantile_numerators),
                "quantile_denominator": self.regimes.quantile_denominator,
            },
            "calibration": {"bins": self.calibration.bins},
            "bootstrap": {
                name: getattr(self.bootstrap, name) for name in self.bootstrap.__dataclass_fields__
            },
            "signed_impact": {
                "metric": self.signed_impact.metric,
                "side_rule": self.signed_impact.side_rule,
                "price_rule": self.signed_impact.price_rule,
            },
            "execution": {
                name: getattr(self.execution, name) for name in self.execution.__dataclass_fields__
            },
            "claims": {
                name: getattr(self.claims, name) for name in self.claims.__dataclass_fields__
            },
        }

    @property
    def semantic_sha256(self) -> str:
        encoded = json.dumps(
            self._semantic_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def hash(self) -> str:
        """Compatibility alias for the formatting-independent semantic identity."""

        return self.semantic_sha256

    def public_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "config_sha256": self.semantic_sha256,
            "semantic_sha256": self.semantic_sha256,
            "source_sha256": self.source_sha256,
            **self._semantic_payload(),
        }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise M8L2AnalysisConfigError(f"{label} must be a TOML table with string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise M8L2AnalysisConfigError(
            f"{label} keys differ (missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)})"
        )


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise M8L2AnalysisConfigError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise M8L2AnalysisConfigError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise M8L2AnalysisConfigError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise M8L2AnalysisConfigError(f"{label} must be a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise M8L2AnalysisConfigError(f"{label} must be a finite number")
    return result


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise M8L2AnalysisConfigError(f"{label} must be a boolean")
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label)))


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(_list(value, label))
    )


def _frozen(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise M8L2AnalysisConfigError(f"{label} is frozen at {expected!r}, observed {observed!r}")


def _role(value: object, label: str) -> M8L2AnalysisRole:
    observed = _text(value, label)
    if observed not in {"train", "validation", "primary_test", "replication_test"}:
        raise M8L2AnalysisConfigError(f"{label} is not a supported frozen-session role")
    return cast(M8L2AnalysisRole, observed)


def _parse_study(raw: object) -> M8L2AnalysisStudy:
    table = _mapping(raw, "study")
    _exact_keys(table, _STUDY_KEYS, "study")
    result = M8L2AnalysisStudy(
        name=_text(table["name"], "study.name"),
        protocol_version=_text(table["protocol_version"], "study.protocol_version"),
        seed=_integer(table["seed"], "study.seed"),
        source=_text(table["source"], "study.source"),
        capture_config_source_sha256=_text(
            table["capture_config_source_sha256"], "study.capture_config_source_sha256"
        ),
        capture_protocol_sha256=_text(
            table["capture_protocol_sha256"], "study.capture_protocol_sha256"
        ),
        symbols=_text_tuple(table["symbols"], "study.symbols"),
        training_role=_role(table["training_role"], "study.training_role"),
        selection_role=_role(table["selection_role"], "study.selection_role"),
        primary_endpoint_role=_role(table["primary_endpoint_role"], "study.primary_endpoint_role"),
        replication_endpoint_role=_role(
            table["replication_endpoint_role"], "study.replication_endpoint_role"
        ),
    )
    expected = M8L2AnalysisStudy(
        name="binance-m8-live-l2-analysis-v2",
        protocol_version="2.0.0",
        seed=20260807,
        source="verified_m8_l2_session_bundles",
        capture_config_source_sha256=_CAPTURE_CONFIG_SOURCE_SHA256,
        capture_protocol_sha256=_CAPTURE_PROTOCOL_SHA256,
        symbols=("BTCUSDT", "ETHUSDT"),
        training_role="train",
        selection_role="validation",
        primary_endpoint_role="primary_test",
        replication_endpoint_role="replication_test",
    )
    _frozen(result, expected, "study contract")
    return result


def _parse_features(raw: object) -> M8L2AnalysisFeatures:
    table = _mapping(raw, "features")
    _exact_keys(table, _FEATURE_KEYS, "features")
    result = M8L2AnalysisFeatures(
        decision_scope=_text(table["decision_scope"], "features.decision_scope"),
        flat_direction_policy=_text(
            table["flat_direction_policy"], "features.flat_direction_policy"
        ),
        rolling_windows=_integer_tuple(table["rolling_windows"], "features.rolling_windows"),
        volatility_window=_integer(table["volatility_window"], "features.volatility_window"),
        model_feature_columns=_text_tuple(
            table["model_feature_columns"], "features.model_feature_columns"
        ),
        clock_max_state_age_ms=_integer(
            table["clock_max_state_age_ms"],
            "features.clock_max_state_age_ms",
        ),
        clock_target_policy=_text(table["clock_target_policy"], "features.clock_target_policy"),
        clock_label_information_end=_text(
            table["clock_label_information_end"], "features.clock_label_information_end"
        ),
        clock_record_target_sequence=_boolean(
            table["clock_record_target_sequence"], "features.clock_record_target_sequence"
        ),
        clock_censor_if_no_eligible_state=_boolean(
            table["clock_censor_if_no_eligible_state"],
            "features.clock_censor_if_no_eligible_state",
        ),
    )
    if any(value <= 0 for value in (*result.rolling_windows, result.volatility_window)):
        raise M8L2AnalysisConfigError("feature windows must be positive")
    if result.clock_max_state_age_ms < 0:
        raise M8L2AnalysisConfigError("features.clock_max_state_age_ms must be nonnegative")
    _frozen(
        result,
        M8L2AnalysisFeatures(
            decision_scope="per_symbol_verified_observed_intervals",
            flat_direction_policy="flat_is_non_up",
            rolling_windows=(20, 100),
            volatility_window=100,
            model_feature_columns=_MODEL_FEATURE_COLUMNS,
            clock_max_state_age_ms=500,
            clock_target_policy="exact_target_locf_same_valid_observed_interval",
            clock_label_information_end="exact_target",
            clock_record_target_sequence=True,
            clock_censor_if_no_eligible_state=True,
        ),
        "features contract",
    )
    return result


def _parse_endpoints(raw: object) -> tuple[M8L2AnalysisEndpoint, ...]:
    result: list[M8L2AnalysisEndpoint] = []
    for index, item in enumerate(_list(raw, "endpoints")):
        table = _mapping(item, f"endpoints[{index}]")
        _exact_keys(table, _ENDPOINT_KEYS, f"endpoints[{index}]")
        raw_domain = _text(table["domain"], f"endpoints[{index}].domain")
        if raw_domain not in {"event", "clock"}:
            raise M8L2AnalysisConfigError(f"endpoints[{index}].domain is unsupported")
        raw_unit = _text(table["unit"], f"endpoints[{index}].unit")
        if raw_unit not in {"events", "milliseconds"}:
            raise M8L2AnalysisConfigError(f"endpoints[{index}].unit is unsupported")
        raw_block_unit = _text(table["paired_block_unit"], f"endpoints[{index}].paired_block_unit")
        if raw_block_unit not in {"events", "milliseconds"}:
            raise M8L2AnalysisConfigError(f"endpoints[{index}].paired_block_unit is unsupported")
        endpoint = M8L2AnalysisEndpoint(
            name=_text(table["name"], f"endpoints[{index}].name"),
            domain=cast(M8L2EndpointDomain, raw_domain),
            horizon_value=_integer(table["horizon_value"], f"endpoints[{index}].horizon_value"),
            unit=cast(M8L2EndpointUnit, raw_unit),
            paired_block_width=_integer(
                table["paired_block_width"], f"endpoints[{index}].paired_block_width"
            ),
            paired_block_unit=cast(M8L2EndpointUnit, raw_block_unit),
            nominal_event_block_width=_integer(
                table["nominal_event_block_width"],
                f"endpoints[{index}].nominal_event_block_width",
            ),
        )
        if (
            endpoint.horizon_value <= 0
            or endpoint.paired_block_width <= 0
            or endpoint.nominal_event_block_width <= 0
        ):
            raise M8L2AnalysisConfigError("endpoint horizons and block widths must be positive")
        expected_unit = "events" if endpoint.domain == "event" else "milliseconds"
        if endpoint.unit != expected_unit or endpoint.paired_block_unit != expected_unit:
            raise M8L2AnalysisConfigError(
                f"endpoints[{index}] units do not match its endpoint domain"
            )
        result.append(endpoint)
    expected = (
        M8L2AnalysisEndpoint("event_20", "event", 20, "events", 40, "events", 40),
        M8L2AnalysisEndpoint("event_100", "event", 100, "events", 200, "events", 200),
        M8L2AnalysisEndpoint(
            "clock_1000ms", "clock", 1000, "milliseconds", 2000, "milliseconds", 20
        ),
        M8L2AnalysisEndpoint(
            "clock_5000ms", "clock", 5000, "milliseconds", 10000, "milliseconds", 100
        ),
    )
    _frozen(tuple(result), expected, "endpoint order/contract")
    return tuple(result)


def _parse_regimes(raw: object) -> M8L2AnalysisRegimes:
    table = _mapping(raw, "regimes")
    _exact_keys(table, _REGIME_KEYS, "regimes")
    result = M8L2AnalysisRegimes(
        fit_role=_role(table["fit_role"], "regimes.fit_role"),
        feature=_text(table["feature"], "regimes.feature"),
        quantile_numerators=_integer_tuple(
            table["quantile_numerators"], "regimes.quantile_numerators"
        ),
        quantile_denominator=_integer(
            table["quantile_denominator"], "regimes.quantile_denominator"
        ),
    )
    if result.quantile_denominator <= 0 or any(
        value <= 0 or value >= result.quantile_denominator for value in result.quantile_numerators
    ):
        raise M8L2AnalysisConfigError("regime quantiles must lie strictly between zero and one")
    _frozen(
        result,
        M8L2AnalysisRegimes("train", "realized_volatility_w100", (1, 2), 3),
        "regimes contract",
    )
    return result


def _parse_calibration(raw: object) -> M8L2AnalysisCalibration:
    table = _mapping(raw, "calibration")
    _exact_keys(table, _CALIBRATION_KEYS, "calibration")
    result = M8L2AnalysisCalibration(bins=_integer(table["bins"], "calibration.bins"))
    if result.bins < 2:
        raise M8L2AnalysisConfigError("calibration.bins must be at least two")
    _frozen(result, M8L2AnalysisCalibration(10), "calibration contract")
    return result


def _parse_bootstrap(raw: object) -> M8L2AnalysisBootstrap:
    table = _mapping(raw, "bootstrap")
    _exact_keys(table, _BOOTSTRAP_KEYS, "bootstrap")
    result = M8L2AnalysisBootstrap(
        method=_text(table["method"], "bootstrap.method"),
        samples=_integer(table["samples"], "bootstrap.samples"),
    )
    if result.samples <= 0:
        raise M8L2AnalysisConfigError("bootstrap.samples must be positive")
    _frozen(result, M8L2AnalysisBootstrap("paired_moving_block", 2000), "bootstrap contract")
    return result


def _parse_signed_impact(raw: object) -> M8L2AnalysisSignedImpact:
    table = _mapping(raw, "signed_impact")
    _exact_keys(table, _SIGNED_IMPACT_KEYS, "signed_impact")
    result = M8L2AnalysisSignedImpact(
        metric=_text(table["metric"], "signed_impact.metric"),
        side_rule=_text(table["side_rule"], "signed_impact.side_rule"),
        price_rule=_text(table["price_rule"], "signed_impact.price_rule"),
    )
    _frozen(
        result,
        M8L2AnalysisSignedImpact(
            "ofi_signed_future_mid_markout",
            "sign_of_horizon_matched_ofi",
            "ofi_sign_times_future_log_mid_return_bps",
        ),
        "signed-impact contract",
    )
    return result


def _parse_execution(raw: object) -> M8L2AnalysisExecution:
    table = _mapping(raw, "execution")
    _exact_keys(table, _EXECUTION_KEYS, "execution")
    result = M8L2AnalysisExecution(
        market_orders_only=_boolean(table["market_orders_only"], "execution.market_orders_only"),
        probability_threshold=_number(
            table["probability_threshold"], "execution.probability_threshold"
        ),
        symmetric_probability_thresholds=_boolean(
            table["symmetric_probability_thresholds"],
            "execution.symmetric_probability_thresholds",
        ),
        order_notional_usd=_number(table["order_notional_usd"], "execution.order_notional_usd"),
        max_l1_participation=_number(
            table["max_l1_participation"], "execution.max_l1_participation"
        ),
        inventory_order_multiples=_integer(
            table["inventory_order_multiples"], "execution.inventory_order_multiples"
        ),
        reference_price_fit_role=_role(
            table["reference_price_fit_role"], "execution.reference_price_fit_role"
        ),
        reference_depth_fit_role=_role(
            table["reference_depth_fit_role"], "execution.reference_depth_fit_role"
        ),
        reference_price_statistic=_text(
            table["reference_price_statistic"], "execution.reference_price_statistic"
        ),
        reference_depth_statistic=_text(
            table["reference_depth_statistic"], "execution.reference_depth_statistic"
        ),
        reference_quantity_policy=_text(
            table["reference_quantity_policy"], "execution.reference_quantity_policy"
        ),
        l1_fill_policy=_text(table["l1_fill_policy"], "execution.l1_fill_policy"),
        scenario_reset_policy=_text(
            table["scenario_reset_policy"], "execution.scenario_reset_policy"
        ),
        extra_slippage_bps=_number(table["extra_slippage_bps"], "execution.extra_slippage_bps"),
        liquidate_at_end=_boolean(table["liquidate_at_end"], "execution.liquidate_at_end"),
    )
    if not 0.5 < result.probability_threshold < 1.0:
        raise M8L2AnalysisConfigError("execution.probability_threshold must be between 0.5 and 1")
    if result.order_notional_usd <= 0:
        raise M8L2AnalysisConfigError("execution.order_notional_usd must be positive")
    if not 0.0 < result.max_l1_participation <= 1.0:
        raise M8L2AnalysisConfigError("execution.max_l1_participation must be in (0, 1]")
    if result.inventory_order_multiples <= 0:
        raise M8L2AnalysisConfigError("execution.inventory_order_multiples must be positive")
    if result.extra_slippage_bps < 0:
        raise M8L2AnalysisConfigError("execution.extra_slippage_bps must be nonnegative")
    expected = M8L2AnalysisExecution(
        market_orders_only=True,
        probability_threshold=0.55,
        symmetric_probability_thresholds=True,
        order_notional_usd=100.0,
        max_l1_participation=0.10,
        inventory_order_multiples=10,
        reference_price_fit_role="train",
        reference_depth_fit_role="train",
        reference_price_statistic="train_median_mid_price",
        reference_depth_statistic="train_q05_min_bid_ask_l1_depth",
        reference_quantity_policy=("min_100usd_and_10pct_train_q05_l1_depth_rounded_down_to_lot"),
        l1_fill_policy="fill_up_to_recorded_l1_depth_cancel_remainder",
        scenario_reset_policy="per_symbol_session_endpoint_latency_pair",
        extra_slippage_bps=0.0,
        liquidate_at_end=True,
    )
    _frozen(result, expected, "execution contract")
    return result


def _parse_claims(raw: object) -> M8L2AnalysisClaims:
    table = _mapping(raw, "claims")
    _exact_keys(table, _CLAIM_KEYS, "claims")
    result = M8L2AnalysisClaims(
        allow_capacity_claim=_boolean(table["allow_capacity_claim"], "claims.allow_capacity_claim"),
        allow_realized_execution_claim=_boolean(
            table["allow_realized_execution_claim"], "claims.allow_realized_execution_claim"
        ),
        allow_profitability_claim=_boolean(
            table["allow_profitability_claim"], "claims.allow_profitability_claim"
        ),
    )
    _frozen(result, M8L2AnalysisClaims(False, False, False), "claims contract")
    return result


def _parse_source(path: Path, source: bytes) -> M8L2AnalysisConfig:
    try:
        raw = tomllib.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise M8L2AnalysisConfigError(f"cannot parse M8 live-L2 analysis TOML: {error}") from error
    root = _mapping(raw, "configuration")
    _exact_keys(root, _TOP_LEVEL_KEYS, "configuration")
    return M8L2AnalysisConfig(
        path=path,
        source_sha256=hashlib.sha256(source).hexdigest(),
        study=_parse_study(root["study"]),
        features=_parse_features(root["features"]),
        endpoints=_parse_endpoints(root["endpoints"]),
        regimes=_parse_regimes(root["regimes"]),
        calibration=_parse_calibration(root["calibration"]),
        bootstrap=_parse_bootstrap(root["bootstrap"]),
        signed_impact=_parse_signed_impact(root["signed_impact"]),
        execution=_parse_execution(root["execution"]),
        claims=_parse_claims(root["claims"]),
    )


def semantic_hash_m8_l2_analysis_config(path: str | Path) -> str:
    """Hash validated semantics; this does not authorize non-frozen source bytes."""

    config_path = Path(path).resolve()
    return _parse_source(config_path, config_path.read_bytes()).semantic_sha256


def load_m8_l2_analysis_config(path: str | Path) -> M8L2AnalysisConfig:
    """Load only the exact reviewed, outcome-blind M8 L2 analysis contract."""

    config_path = Path(path).resolve()
    result = _parse_source(config_path, config_path.read_bytes())
    if result.semantic_sha256 != M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256:
        raise M8L2AnalysisConfigError(
            "configuration semantics do not match the code-bound outcome-blind freeze "
            f"{M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256}"
        )
    if result.source_sha256 != M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256:
        raise M8L2AnalysisConfigError(
            "configuration bytes do not match the outcome-blind freeze "
            f"{M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256}"
        )
    return result


__all__ = [
    "M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256",
    "M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256",
    "M8L2AnalysisBootstrap",
    "M8L2AnalysisCalibration",
    "M8L2AnalysisClaims",
    "M8L2AnalysisConfig",
    "M8L2AnalysisConfigError",
    "M8L2AnalysisEndpoint",
    "M8L2AnalysisExecution",
    "M8L2AnalysisFeatures",
    "M8L2AnalysisRegimes",
    "M8L2AnalysisRole",
    "M8L2AnalysisSignedImpact",
    "M8L2AnalysisStudy",
    "M8L2EndpointDomain",
    "M8L2EndpointUnit",
    "load_m8_l2_analysis_config",
    "semantic_hash_m8_l2_analysis_config",
]
