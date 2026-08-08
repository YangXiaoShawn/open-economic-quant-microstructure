from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from microstructure.m8_l2_analysis_config import (
    M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256,
    M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256,
    M8L2AnalysisConfigError,
    load_m8_l2_analysis_config,
    semantic_hash_m8_l2_analysis_config,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_l2_analysis.toml"


def _write_mutation(tmp_path: Path, old: str, new: str) -> Path:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in source
    path = tmp_path / "mutated.toml"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return path


def _changed_assignment(value: str) -> str:
    if value == "true":
        return "false"
    if value == "false":
        return "true"
    if value.startswith('"'):
        assert value.endswith('"')
        return value[:-1] + '-changed"'
    if value.startswith("["):
        assert value.endswith("]")
        addition = '"CHANGED"' if '"' in value else "999"
        return value[:-1] + f", {addition}]"
    if "." in value:
        return str(float(value) + 0.01)
    return str(int(value) + 1)


def _all_field_mutations() -> list[tuple[int, str]]:
    lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    mutations: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if not line or line.startswith("["):
            continue
        key, separator, value = line.partition(" = ")
        assert separator and key and value
        changed = list(lines)
        changed[index] = f"{key} = {_changed_assignment(value)}"
        mutations.append((index + 1, "\n".join(changed) + "\n"))
    return mutations


def test_frozen_analysis_contract_binds_exact_bytes_and_all_rules() -> None:
    config = load_m8_l2_analysis_config(CONFIG_PATH)

    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == (
        M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256
    )
    assert config.source_sha256 == M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256
    assert config.semantic_sha256 == M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256
    assert config.hash == M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256
    assert config.study.symbols == ("BTCUSDT", "ETHUSDT")
    assert config.study.training_role == "train"
    assert config.study.selection_role == "validation"
    assert config.study.primary_endpoint_role == "primary_test"
    assert config.study.replication_endpoint_role == "replication_test"

    assert config.features.decision_scope == "per_symbol_verified_observed_intervals"
    assert config.features.flat_direction_policy == "flat_is_non_up"
    assert config.features.rolling_windows == (20, 100)
    assert config.features.volatility_window == 100
    assert config.features.model_feature_columns == (
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
    assert config.features.clock_max_state_age_ms == 500
    assert config.features.clock_target_policy == ("exact_target_locf_same_valid_observed_interval")
    assert config.features.clock_label_information_end == "exact_target"
    assert config.features.clock_record_target_sequence is True
    assert config.features.clock_censor_if_no_eligible_state is True

    assert [endpoint.name for endpoint in config.endpoints] == [
        "event_20",
        "event_100",
        "clock_1000ms",
        "clock_5000ms",
    ]
    assert [(endpoint.domain, endpoint.horizon_value) for endpoint in config.endpoints] == [
        ("event", 20),
        ("event", 100),
        ("clock", 1000),
        ("clock", 5000),
    ]
    assert [endpoint.paired_block_width for endpoint in config.endpoints] == [40, 200, 2000, 10000]
    assert [endpoint.paired_block_unit for endpoint in config.endpoints] == [
        "events",
        "events",
        "milliseconds",
        "milliseconds",
    ]
    assert [endpoint.nominal_event_block_width for endpoint in config.endpoints] == [
        40,
        200,
        20,
        100,
    ]

    assert config.regimes.fit_role == "train"
    assert config.regimes.feature == "realized_volatility_w100"
    assert config.regimes.quantile_numerators == (1, 2)
    assert config.regimes.quantile_denominator == 3
    assert config.calibration.bins == 10
    assert config.bootstrap.method == "paired_moving_block"
    assert config.bootstrap.samples == 2000
    assert config.signed_impact.metric == "ofi_signed_future_mid_markout"
    assert config.signed_impact.side_rule == "sign_of_horizon_matched_ofi"
    assert config.signed_impact.price_rule == "ofi_sign_times_future_log_mid_return_bps"

    assert config.execution.market_orders_only is True
    assert config.execution.probability_threshold == 0.55
    assert config.execution.symmetric_probability_thresholds is True
    assert config.execution.order_notional_usd == 100.0
    assert config.execution.max_l1_participation == 0.10
    assert config.execution.inventory_order_multiples == 10
    assert config.execution.reference_price_fit_role == "train"
    assert config.execution.reference_depth_fit_role == "train"
    assert config.execution.reference_price_statistic == "train_median_mid_price"
    assert config.execution.reference_depth_statistic == "train_q05_min_bid_ask_l1_depth"
    assert config.execution.reference_quantity_policy == (
        "min_100usd_and_10pct_train_q05_l1_depth_rounded_down_to_lot"
    )
    assert config.execution.l1_fill_policy == "fill_up_to_recorded_l1_depth_cancel_remainder"
    assert config.execution.scenario_reset_policy == ("per_symbol_session_endpoint_latency_pair")
    assert config.execution.extra_slippage_bps == 0.0
    assert config.execution.liquidate_at_end is True
    assert config.claims.allow_capacity_claim is False
    assert config.claims.allow_realized_execution_claim is False
    assert config.claims.allow_profitability_claim is False

    public = config.public_dict()
    assert public["semantic_sha256"] == M8_L2_ANALYSIS_CONFIG_SEMANTIC_SHA256
    assert public["source_sha256"] == M8_L2_ANALYSIS_CONFIG_SOURCE_SHA256


def test_analysis_config_objects_are_immutable() -> None:
    config = load_m8_l2_analysis_config(CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.execution.probability_threshold = 0.99  # type: ignore[misc]


def test_semantic_hash_ignores_formatting_but_authority_loader_does_not(tmp_path: Path) -> None:
    canonical_hash = semantic_hash_m8_l2_analysis_config(CONFIG_PATH)
    formatted = tmp_path / "formatted.toml"
    formatted.write_text(
        "# Formatting is not part of the semantic identity.\n\n"
        + CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert semantic_hash_m8_l2_analysis_config(formatted) == canonical_hash
    with pytest.raises(M8L2AnalysisConfigError, match="configuration bytes do not match"):
        load_m8_l2_analysis_config(formatted)


@pytest.mark.parametrize(
    ("line_number", "mutated_source"),
    _all_field_mutations(),
    ids=lambda value: f"line-{value}" if isinstance(value, int) else None,
)
def test_changing_any_declared_field_fails_closed(
    line_number: int, mutated_source: str, tmp_path: Path
) -> None:
    path = tmp_path / f"changed-line-{line_number}.toml"
    path.write_text(mutated_source, encoding="utf-8")

    with pytest.raises(M8L2AnalysisConfigError):
        load_m8_l2_analysis_config(path)


def test_extra_and_missing_keys_fail_closed(tmp_path: Path) -> None:
    extra = _write_mutation(
        tmp_path,
        "[calibration]\nbins = 10",
        "[calibration]\nbins = 10\nunreviewed = true",
    )
    with pytest.raises(M8L2AnalysisConfigError, match=r"calibration keys differ.*unreviewed"):
        load_m8_l2_analysis_config(extra)

    missing = _write_mutation(tmp_path, "bins = 10\n", "")
    with pytest.raises(M8L2AnalysisConfigError, match=r"calibration keys differ.*bins"):
        load_m8_l2_analysis_config(missing)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "probability_threshold = 0.55",
            "probability_threshold = nan",
            "must be a finite number",
        ),
        (
            "max_l1_participation = 0.10",
            "max_l1_participation = -0.10",
            r"must be in \(0, 1\]",
        ),
        (
            "clock_max_state_age_ms = 500",
            "clock_max_state_age_ms = -1",
            "must be nonnegative",
        ),
        (
            "samples = 2000",
            "samples = true",
            "must be an integer",
        ),
    ],
)
def test_illegal_numeric_values_fail_before_source_authority(
    old: str, new: str, message: str, tmp_path: Path
) -> None:
    path = _write_mutation(tmp_path, old, new)

    with pytest.raises(M8L2AnalysisConfigError, match=message):
        load_m8_l2_analysis_config(path)
