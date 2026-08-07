"""Atomic, reproducible end-to-end producer for the synthetic research slice.

This module orchestrates existing data, research, model, execution, and reporting
APIs.  It does not contain an exchange connection or an order-entry path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from microstructure.config import ProjectConfig, datetime_to_ns
from microstructure.data.quality import ValidationReport, validate_table
from microstructure.data.storage import DatasetWriteResult, write_partitioned_parquet
from microstructure.data.synthetic import generate_synthetic_market, iter_table_batches
from microstructure.execution import run_execution_sensitivity, simulate_predictions
from microstructure.provenance import (
    git_source_tree_sha256,
    git_state,
    provenance_header,
    sha256_file,
    write_json,
)
from microstructure.reporting import (
    load_run_bundle,
    render_executive_memo,
    render_model_comparison,
    render_technical_report,
    write_checksum_manifest,
)
from microstructure.research.analysis import (
    LiquidityShockThresholds,
    RegimeThresholds,
    assign_market_regimes,
    cross_instrument_stability_summary,
    estimate_signal_half_life,
    feature_stability_summary,
    intraday_liquidity_summary,
    large_trade_price_impact_summary,
    liquidity_recovery_summary,
    ofi_future_return_association,
    regime_outcome_summary,
)
from microstructure.research.features import (
    add_future_event_labels,
    build_research_frame,
    validate_temporal_contract,
)
from microstructure.research.labels import add_event_time_price_impact_labels
from microstructure.research.models import (
    block_bootstrap_metric,
    classification_metrics,
    evaluate_model_ladder,
)
from microstructure.research.splits import WalkForwardPlan, expanding_walk_forward_splits

PIPELINE_SCHEMA_VERSION = "1.0.0"
_SYNTHETIC_EVIDENCE = "SYNTHETIC_SMOKE"


class PipelineError(RuntimeError):
    """Raised when a run cannot be produced without violating its contracts."""


def _utc_from_ns(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    clean = _json_safe(payload)
    if not isinstance(clean, (dict, list)):
        raise TypeError("JSON artifact payload must be an object or list")
    write_json(path, clean)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content.rstrip() + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _mapping_columns_as_json(frame: pl.DataFrame) -> pl.DataFrame:
    """Make metric-map columns portable to Parquet without inventing map entries."""
    rows = frame.to_dicts()
    mapping_columns = {
        key for row in rows for key, value in row.items() if isinstance(value, Mapping)
    }
    if not mapping_columns:
        return frame
    serialized: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key, value in row.items():
            if key in mapping_columns:
                output[f"{key}_json"] = json.dumps(
                    _json_safe(value), sort_keys=True, separators=(",", ":")
                )
            else:
                output[key] = value
        serialized.append(output)
    return pl.DataFrame(serialized)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _dataset_snapshot(result: DatasetWriteResult, run_root: Path) -> dict[str, Any]:
    return {
        "dataset": result.dataset,
        "schema_version": result.schema_version,
        "rows": result.rows,
        "manifest_path": _relative(result.manifest_path, run_root),
        "manifest_sha256": result.manifest_sha256,
        "partitions": [
            {
                "venue": artifact.venue,
                "symbol": artifact.symbol,
                "partition_date": artifact.partition_date,
                "rows": artifact.rows,
                "data_path": _relative(artifact.data_path, run_root),
                "data_sha256": artifact.data_sha256,
                "manifest_path": _relative(artifact.manifest_path, run_root),
                "manifest_sha256": artifact.manifest_sha256,
            }
            for artifact in result.artifacts
        ],
    }


def _input_identity_hashes(results: Sequence[DatasetWriteResult]) -> list[str]:
    """Hash timestamp-free manifest identities and their immutable data parts."""
    identities: list[str] = []
    for result in results:
        partitions = sorted(
            (
                {
                    "venue": artifact.venue,
                    "symbol": artifact.symbol,
                    "partition_date": artifact.partition_date,
                    "rows": artifact.rows,
                    "data_sha256": artifact.data_sha256,
                }
                for artifact in result.artifacts
            ),
            key=lambda item: (
                str(item["venue"]),
                str(item["symbol"]),
                str(item["partition_date"]),
                str(item["data_sha256"]),
            ),
        )
        identities.append(
            _stable_sha256(
                {
                    "artifact_kind": "normalized_dataset_manifest_identity",
                    "dataset": result.dataset,
                    "schema_version": result.schema_version,
                    "rows": result.rows,
                    "partitions": partitions,
                }
            )
        )
        identities.extend(artifact.data_sha256 for artifact in result.artifacts)
    return sorted(identities)


def _run_identity(
    *,
    config_sha256: str,
    input_identity_sha256: Sequence[str],
    git: Mapping[str, Any],
    seed: int,
) -> tuple[str, dict[str, Any]]:
    inputs = {
        "config_sha256": config_sha256,
        "input_manifest_identity_or_data_sha256": sorted(input_identity_sha256),
        "git": {
            "commit": str(git.get("commit", "UNKNOWN")),
            "dirty": bool(git.get("dirty", False)),
            "source_tree_sha256": str(git.get("source_tree_sha256", "UNKNOWN")),
        },
        "seed": seed,
    }
    return _stable_sha256(inputs), inputs


def _quality_payload(
    reports: Sequence[ValidationReport], *, generated_at_utc: str
) -> dict[str, Any]:
    findings = [asdict(finding) for report in reports for finding in report.findings]
    errors = sum(report.error_count for report in reports)
    warnings = sum(report.warning_count for report in reports)
    return {
        "generated_at_utc": generated_at_utc,
        "dataset": "synthetic_normalized_market",
        "rows_checked": sum(report.rows_checked for report in reports),
        "summary": {"errors": errors, "warnings": warnings},
        "reports": [
            {
                "dataset": report.dataset,
                "rows_checked": report.rows_checked,
                "errors": report.error_count,
                "warnings": report.warning_count,
            }
            for report in reports
        ],
        "findings": findings,
        "mutation_policy": "observations were not changed or repaired",
    }


def _serialize_plan(plan: WalkForwardPlan) -> dict[str, Any]:
    return {
        "contract": (
            "global decision-time buckets; feature_ready and uncensored rows only; "
            "training labels end strictly before evaluation; configured embargo applied"
        ),
        "index_basis": (
            "zero-based row positions in research/evaluation_frame.parquet; this exact "
            "feature-ready frame is passed to splitting and model evaluation"
        ),
        "decision_time_count": plan.decision_time_count,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_indices": [int(value) for value in fold.train_indices.tolist()],
                "validation_indices": [int(value) for value in fold.validation_indices.tolist()],
                "train_start_ts_ns": fold.train_start_ts_ns,
                "train_end_ts_ns": fold.train_end_ts_ns,
                "validation_start_ts_ns": fold.validation_start_ts_ns,
                "validation_end_ts_ns": fold.validation_end_ts_ns,
                "purged_rows": fold.purged_rows,
                "embargoed_time_buckets": fold.embargoed_time_buckets,
            }
            for fold in plan.folds
        ],
        "final_train_indices": [int(value) for value in plan.final_train_indices.tolist()],
        "test_indices": [int(value) for value in plan.test_indices.tolist()],
        "test_start_ts_ns": plan.test_start_ts_ns,
        "test_end_ts_ns": plan.test_end_ts_ns,
    }


def _bootstrap_comparison(
    comparison: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    horizon_events: int,
) -> tuple[list[dict[str, Any]], pl.DataFrame]:
    block_width = max(2, 2 * horizon_events)
    blocked = (
        predictions.with_columns(
            (pl.col("decision_ts_ns").rank(method="dense").cast(pl.Int64) - 1).alias(
                "_bootstrap_time_rank"
            )
        )
        .with_columns(
            (pl.col("_bootstrap_time_rank") // block_width).cast(pl.String).alias("bootstrap_block")
        )
        .drop("_bootstrap_time_rank")
    )
    rows: list[dict[str, Any]] = []
    for row in comparison.to_dicts():
        serialized = dict(row)
        if row["split"] == "test":
            selected = blocked.filter(
                (pl.col("split") == "test") & (pl.col("model") == str(row["model"]))
            )
            interval = block_bootstrap_metric(
                selected,
                metric=metric,
                block_column="bootstrap_block",
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            serialized[f"{metric}_ci_low"] = interval.lower
            serialized[f"{metric}_ci_high"] = interval.upper
            serialized["bootstrap_status"] = interval.status
            serialized["bootstrap_blocks"] = interval.n_blocks
            serialized["bootstrap_samples"] = interval.n_bootstrap
            serialized["bootstrap_block_width_events"] = block_width
            serialized["bootstrap_block_policy"] = (
                "pooled_dense_decision_time_clusters_2x_label_horizon"
            )
            serialized["bootstrap_limitation"] = (
                "fixed clusters approximate serial and contemporaneous dependence; "
                "they do not establish asymptotic coverage"
            )
        rows.append(serialized)
    return rows, blocked


def _execution_events(
    research_frame: pl.DataFrame, trades: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build an availability-time replay frame without joining future trades.

    The current simulator accepts one qualifying trade per market-state event.
    For the synthetic fixture there is exactly one trade in each book interval.
    If a future input has more, this adapter selects the latest eligible trade and
    records the conservative omission count rather than double-counting volume.
    """
    trade_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades.sort(["symbol", "continuity_id", "available_ts_ns", "trade_id"]).to_dicts():
        key = (str(trade["symbol"]), str(trade["continuity_id"]))
        trade_groups[key].append(trade)

    pointers: dict[tuple[str, str], int] = defaultdict(int)
    prior_decision: dict[tuple[str, str], int] = {}
    omitted_trades = 0
    output: list[dict[str, Any]] = []
    ordered = research_frame.sort(["symbol", "continuity_id", "decision_sequence"])
    for research_row in ordered.to_dicts():
        symbol = str(research_row["symbol"])
        continuity = str(research_row["continuity_id"])
        key = (symbol, continuity)
        decision_ts = int(research_row["decision_ts_ns"])
        lower_bound = prior_decision.get(key, -1)
        candidates = trade_groups.get(key, [])
        pointer = pointers[key]
        eligible: list[dict[str, Any]] = []
        while (
            pointer < len(candidates) and int(candidates[pointer]["available_ts_ns"]) <= decision_ts
        ):
            if int(candidates[pointer]["available_ts_ns"]) > lower_bound:
                eligible.append(candidates[pointer])
            pointer += 1
        pointers[key] = pointer
        prior_decision[key] = decision_ts
        omitted_trades += max(0, len(eligible) - 1)
        selected_trade: dict[str, Any] | None = eligible[-1] if eligible else None
        trade_side = 0
        trade_quantity = 0.0
        trade_price = float(research_row["mid_price"])
        if selected_trade is not None:
            trade_side = 1 if str(selected_trade["aggressor_side"]).lower() == "buy" else -1
            trade_quantity = float(selected_trade["quantity"])
            trade_price = float(selected_trade["price"])

        output.append(
            {
                "symbol": symbol,
                "continuity_id": continuity,
                "sample_id": int(research_row["decision_sequence"]),
                "decision_sequence": int(research_row["decision_sequence"]),
                # The execution clock is availability/decision time, not exchange event time.
                "event_ts_ns": decision_ts,
                "decision_ts_ns": decision_ts,
                "market_event_ts_ns": int(research_row["market_event_ts_ns"]),
                "best_bid": float(research_row["best_bid"]),
                "best_ask": float(research_row["best_ask"]),
                "mid_price": float(research_row["mid_price"]),
                "bid_quantity": float(research_row["bid_quantity"]),
                "ask_quantity": float(research_row["ask_quantity"]),
                "depth_bid_1": float(research_row["depth_bid_1"]),
                "depth_ask_1": float(research_row["depth_ask_1"]),
                "tick_size": float(research_row["tick_size"]),
                "lot_size": float(research_row["lot_size"]),
                "trade_side": trade_side,
                "trade_quantity": trade_quantity,
                "trade_price": trade_price,
            }
        )
    return pl.DataFrame(output), {
        "clock": "decision_ts_ns derived from available_ts_ns",
        "join_rule": (
            "latest trade with previous_decision_ts < trade.available_ts_ns <= decision_ts_ns "
            "within symbol and continuity_id"
        ),
        "multiple_trades_policy": "latest eligible trade retained; earlier interval trades omitted",
        "omitted_eligible_trades": omitted_trades,
        "trade_only_queue_limitation": True,
    }


def _indexed_plan_rows(evaluation: pl.DataFrame, indices: NDArray[np.int64]) -> pl.DataFrame:
    return evaluation.with_row_index("_evaluation_index").filter(
        pl.col("_evaluation_index").is_in(indices.tolist())
    )


def _symbol_quantiles(
    frame: pl.DataFrame,
    column: str,
    quantiles: Sequence[float],
    *,
    positive_only: bool = False,
) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    for partition in frame.partition_by("symbol", maintain_order=True):
        symbol = str(partition.get_column("symbol")[0])
        values = partition.get_column(column).drop_nulls().to_numpy().astype(np.float64)
        values = values[np.isfinite(values)]
        if positive_only:
            values = values[values > 0]
        if not values.size:
            raise PipelineError(f"training partition {symbol} has no finite values for {column}")
        result[symbol] = tuple(float(np.quantile(values, value)) for value in quantiles)
    return result


def _regime_model_performance(
    predictions: pl.DataFrame,
    regimes: pl.DataFrame,
    *,
    calibration_bins: int,
) -> pl.DataFrame:
    regime_keys = regimes.select(
        pl.col("_evaluation_index").alias("row_id"),
        "symbol",
        "volatility_regime",
        "liquidity_regime",
        "joint_market_regime",
    )
    joined = predictions.join(
        regime_keys,
        on=["row_id", "symbol"],
        how="inner",
        validate="m:1",
    )
    group_columns = (
        "model",
        "family",
        "split",
        "fold_id",
        "symbol",
        "volatility_regime",
        "liquidity_regime",
        "joint_market_regime",
    )
    rows: list[dict[str, Any]] = []
    for partition in joined.partition_by(list(group_columns), maintain_order=True):
        y_true = partition.get_column("y_true").to_numpy().astype(np.int64)
        probability = partition.get_column("probability").to_numpy().astype(np.float64)
        start = int(cast(int, partition.get_column("decision_ts_ns").min()))
        end = int(cast(int, partition.get_column("decision_ts_ns").max()))
        row = {column: partition.get_column(column)[0] for column in group_columns}
        row.update(
            {
                "n_obs": partition.height,
                "period_start_utc": _utc_from_ns(start),
                "period_end_utc": _utc_from_ns(end),
                **classification_metrics(
                    y_true,
                    probability,
                    calibration_bins=calibration_bins,
                ),
                "threshold_source": "caller_supplied_final_training_period",
                "analysis_kind": "regime_model_performance_descriptive",
                "descriptive_only": True,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows).sort(list(group_columns))


def _write_descriptive_analysis(
    *,
    research: pl.DataFrame,
    evaluation: pl.DataFrame,
    plan: WalkForwardPlan,
    execution_events: pl.DataFrame,
    model_predictions: pl.DataFrame,
    feature_columns: Sequence[str],
    config: ProjectConfig,
    stage: Path,
    generated_at_utc: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Persist predeclared descriptive diagnostics with train-only thresholds."""
    indexed = evaluation.with_row_index("_evaluation_index")
    train = _indexed_plan_rows(evaluation, plan.final_train_indices)
    test = _indexed_plan_rows(evaluation, plan.test_indices)
    configured_horizon = config.features.label_horizon_events
    horizons = sorted({1, max(1, configured_horizon // 2), configured_horizon})
    keys = ["symbol", "continuity_id", "decision_sequence"]
    labeled = indexed
    return_columns: dict[int, str] = {}
    for horizon in horizons:
        return_column = f"future_mid_return_h{horizon}"
        variant = (
            add_future_event_labels(research, horizon)
            .filter(pl.col("feature_ready"))
            .select(*keys, pl.col("future_mid_return").alias(return_column))
        )
        labeled = labeled.join(variant, on=keys, how="left", validate="1:1")
        return_columns[horizon] = return_column
    labeled = labeled.sort("_evaluation_index")
    if labeled.height != evaluation.height:
        raise PipelineError("descriptive labels do not align one-to-one with evaluation rows")

    intraday = intraday_liquidity_summary(labeled, bucket_minutes=60)
    association = ofi_future_return_association(
        labeled,
        horizon_return_columns=return_columns,
        ofi_column="ofi_l1",
    )
    half_life = estimate_signal_half_life(association)
    cross_instrument = cross_instrument_stability_summary(
        association,
        value_column="ols_slope_return_per_ofi_unit",
    )

    volatility_column = f"realized_volatility_w{config.features.volatility_window}"
    volatility_quantiles = _symbol_quantiles(train, volatility_column, (1 / 3, 2 / 3))
    spread_quantiles = _symbol_quantiles(train, "spread_bps", (1 / 3, 2 / 3))
    depth_quantiles = _symbol_quantiles(train, "depth_total_l1", (1 / 3, 2 / 3))
    regime_thresholds = {
        symbol: RegimeThresholds(
            volatility_low=volatility_quantiles[symbol][0],
            volatility_high=volatility_quantiles[symbol][1],
            spread_tight_bps=spread_quantiles[symbol][0],
            spread_wide_bps=spread_quantiles[symbol][1],
            depth_low=depth_quantiles[symbol][0],
            depth_high=depth_quantiles[symbol][1],
        )
        for symbol in volatility_quantiles
    }
    regimes = assign_market_regimes(
        labeled,
        train_thresholds=regime_thresholds,
        volatility_column=volatility_column,
    )
    held_out_regimes = regimes.filter(pl.col("_evaluation_index").is_in(plan.test_indices.tolist()))
    regime_outcomes = regime_outcome_summary(
        held_out_regimes,
        outcome_columns=("future_mid_return", "future_mid_up"),
    )
    regime_model_performance = _regime_model_performance(
        model_predictions.filter(pl.col("split") == "test"),
        held_out_regimes,
        calibration_bins=config.evaluation.calibration_bins,
    )

    recovery_spread = _symbol_quantiles(train, "spread_bps", (0.5, 0.9))
    recovery_depth = _symbol_quantiles(train, "depth_total_l1", (0.1, 0.5))
    recovery_thresholds = {
        symbol: LiquidityShockThresholds(
            spread_shock_bps=recovery_spread[symbol][1],
            depth_shock_max=recovery_depth[symbol][0],
            spread_recovery_bps=recovery_spread[symbol][0],
            depth_recovery_min=recovery_depth[symbol][1],
            max_recovery_events=max(2, configured_horizon),
        )
        for symbol in recovery_spread
    }
    recovery = liquidity_recovery_summary(
        labeled,
        train_thresholds=recovery_thresholds,
    )
    stability = feature_stability_summary(
        train,
        test,
        feature_columns=feature_columns,
    )

    impact_input = (
        labeled.join(
            execution_events.select(*keys, "trade_side", "trade_quantity"),
            on=keys,
            how="left",
            validate="1:1",
        )
        .filter((pl.col("trade_quantity") > 0) & (pl.col("trade_side") != 0))
        .with_columns(pl.col(f"future_mid_return_h{configured_horizon}").alias("future_mid_return"))
    )
    event_impact = add_event_time_price_impact_labels(
        impact_input,
        side_column="trade_side",
    )
    impact_train = event_impact.filter(
        pl.col("_evaluation_index").is_in(plan.final_train_indices.tolist())
    )
    quantity_thresholds = {
        symbol: values[0]
        for symbol, values in _symbol_quantiles(
            impact_train,
            "trade_quantity",
            (config.features.large_trade_quantile,),
            positive_only=True,
        ).items()
    }
    large_trade_impact = large_trade_price_impact_summary(
        event_impact,
        impact_columns={configured_horizon: "event_time_signed_price_impact_bps"},
        train_quantity_thresholds=quantity_thresholds,
        quantity_column="trade_quantity",
    )

    frames = {
        "intraday_liquidity": intraday,
        "ofi_future_return": association,
        "signal_decay_curve": half_life.curve,
        "signal_half_life": half_life.summary,
        "event_time_impact_labels": event_impact,
        "large_trade_price_impact": large_trade_impact,
        "liquidity_recovery": recovery,
        "market_regimes": regimes,
        "regime_outcomes": regime_outcomes,
        "regime_model_performance": regime_model_performance,
        "cross_instrument_stability": cross_instrument,
        "feature_stability": stability,
    }
    analysis_root = stage / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for name, frame in frames.items():
        relative = f"analysis/{name}.parquet"
        frame.write_parquet(stage / relative)
        artifacts[f"analysis_{name}"] = relative
    manifest = {
        "generated_at_utc": generated_at_utc,
        "evidence_tier": _SYNTHETIC_EVIDENCE,
        "descriptive_only": True,
        "economic_claim_authorized": False,
        "threshold_source": "final_training_period_only",
        "training_rows": train.height,
        "test_rows": test.height,
        "event_horizons": horizons,
        "artifacts": {
            name: {"path": artifacts[f"analysis_{name}"], "rows": frame.height}
            for name, frame in frames.items()
        },
        "limitations": [
            "Synthetic diagnostics validate computation only.",
            "Trade-side impact uses the latest strictly observable interval trade.",
            "Regime and shock thresholds are derived only from the final training rows.",
            "Regime outcome and model-performance summaries are restricted to the held-out test.",
        ],
    }
    _write_json(analysis_root / "manifest.json", manifest)
    artifacts["analysis_manifest"] = "analysis/manifest.json"
    return artifacts, manifest


def _assert_execution_alignment(events: pl.DataFrame, predictions: pl.DataFrame) -> None:
    if predictions.is_empty():
        raise PipelineError("selected model produced no held-out test predictions")
    event_keys = {
        (str(row["symbol"]), int(row["decision_sequence"])): (
            str(row["continuity_id"]),
            int(row["decision_ts_ns"]),
        )
        for row in events.select(
            "symbol", "decision_sequence", "continuity_id", "decision_ts_ns"
        ).to_dicts()
    }
    for row in predictions.to_dicts():
        key = (str(row["symbol"]), int(row["decision_sequence"]))
        expected = event_keys.get(key)
        observed = (str(row["continuity_id"]), int(row["decision_ts_ns"]))
        if expected is None or expected != observed:
            raise PipelineError(
                "selected prediction does not match its causal execution event and continuity"
            )
        if row.get("split") != "test" or row.get("is_oos") is not True:
            raise PipelineError("execution requires validation-selected held-out test predictions")


def _execution_metric_row(
    metrics: Mapping[str, Any],
    *,
    model: str,
    family: str,
    order_label: str,
    horizon_events: int,
    period_start_utc: str,
    period_end_utc: str,
    n_obs: int,
) -> dict[str, Any]:
    turnover = float(metrics.get("turnover_notional", 0.0))
    fees = float(metrics.get("total_fees", 0.0))
    return {
        "model": model if order_label == "market" else f"{model} [limit execution]",
        "predictive_model": model,
        "family": family,
        "instrument": "POOLED",
        "instrument_scope": "POOLED",
        "horizon_events": horizon_events,
        "split": "test",
        "n_obs": n_obs,
        "period_start_utc": period_start_utc,
        "period_end_utc": period_end_utc,
        "order_type": order_label,
        "gross_bps": metrics.get("gross_edge_bps"),
        "fees_bps": fees / turnover * 10_000.0 if turnover else None,
        "net_bps": metrics.get("net_edge_bps"),
        "fill_rate": metrics.get("fill_ratio"),
        "turnover": metrics.get("turnover_notional"),
        "max_drawdown": metrics.get("maximum_drawdown"),
        "max_drawdown_bps_of_turnover": metrics.get("maximum_drawdown_bps_of_turnover"),
        "mean_adverse_selection_bps": metrics.get("mean_adverse_selection_bps"),
        "maximum_absolute_inventory_by_symbol": metrics.get("maximum_absolute_inventory_by_symbol"),
        "selected_on": "validation",
        "evidence_tier": _SYNTHETIC_EVIDENCE,
    }


def _produce_synthetic(config: ProjectConfig, stage: Path) -> None:
    if config.data.mode != "synthetic":
        raise PipelineError("synthetic producer requires data.mode='synthetic'")
    if config.data.events_per_symbol is None:
        raise PipelineError("synthetic configuration requires events_per_symbol")

    generated = provenance_header(
        project_root=config.project_root,
        config_hash=config.hash,
        evidence_tier=_SYNTHETIC_EVIDENCE,
        input_manifests=[],
    )
    generated_at = cast(str, generated["generated_at_utc"])
    start_ns = datetime_to_ns(config.data.start)
    synthetic = generate_synthetic_market(
        symbols=config.data.symbols,
        events_per_symbol=config.data.events_per_symbol,
        start_ts_ns=start_ns,
        seed=config.run.seed,
    )

    normalized_root = stage / "data" / "normalized"
    requested_end_ns = datetime_to_ns(config.data.end) if config.data.end else None
    trades_written = write_partitioned_parquet(
        iter_table_batches(synthetic.trades),
        root=normalized_root,
        dataset="trades",
        schema_name="trades",
        source=config.data.source,
        source_uri="synthetic://local/deterministic-v1",
        downloaded_at_utc=generated_at,
        requested_start_ns=start_ns,
        requested_end_ns=requested_end_ns,
    )
    books_written = write_partitioned_parquet(
        iter_table_batches(synthetic.book_observations),
        root=normalized_root,
        dataset="book_observations",
        schema_name="book_observations",
        source=config.data.source,
        source_uri="synthetic://local/deterministic-v1",
        downloaded_at_utc=generated_at,
        requested_start_ns=start_ns,
        requested_end_ns=requested_end_ns,
    )

    quality_reports = (
        validate_table(
            synthetic.trades,
            "trades",
            max_spread_bps=config.quality.max_spread_bps,
            max_silence_ns=config.quality.max_silence_ms * 1_000_000,
        ),
        validate_table(
            synthetic.book_observations,
            "book_observations",
            max_spread_bps=config.quality.max_spread_bps,
            max_silence_ns=config.quality.max_silence_ms * 1_000_000,
        ),
    )
    quality = _quality_payload(quality_reports, generated_at_utc=generated_at)
    _write_json(stage / "quality" / "summary.json", quality)
    if config.quality.fail_on_error and any(report.has_errors for report in quality_reports):
        raise PipelineError("synthetic normalized data failed configured quality gates")

    trades = cast(pl.DataFrame, pl.from_arrow(synthetic.trades))
    books = cast(pl.DataFrame, pl.from_arrow(synthetic.book_observations))
    research = build_research_frame(books, trades, config.features)
    temporal_audit = validate_temporal_contract(research)
    research_path = stage / "research" / "research_frame.parquet"
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research.write_parquet(research_path)

    evaluation = research.filter(pl.col("feature_ready"))
    if evaluation.is_empty():
        raise PipelineError("causal feature construction produced no feature-ready rows")
    evaluation_path = stage / "research" / "evaluation_frame.parquet"
    evaluation.write_parquet(evaluation_path)

    plan = expanding_walk_forward_splits(evaluation, config.evaluation)
    _write_json(stage / "research" / "folds.json", _serialize_plan(plan))
    ladder = evaluate_model_ladder(
        evaluation,
        plan,
        config.models,
        seed=config.run.seed,
        calibration_bins=config.evaluation.calibration_bins,
    )
    predictive_rows, predictions = _bootstrap_comparison(
        ladder.comparison,
        ladder.predictions,
        metric=ladder.selection_metric,
        n_bootstrap=config.evaluation.bootstrap_samples,
        seed=config.run.seed + 10_000,
        horizon_events=config.features.label_horizon_events,
    )
    _write_json(stage / "metrics" / "predictive_metrics.json", predictive_rows)
    predictions_path = stage / "models" / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(predictions_path)
    selected_predictions = predictions.filter(
        (pl.col("split") == "test") & (pl.col("model") == ladder.selected_model)
    )
    selected_path = stage / "models" / "selected_test_predictions.parquet"
    selected_predictions.write_parquet(selected_path)

    execution_events, adapter_assumptions = _execution_events(research, trades)
    analysis_artifacts, analysis_manifest = _write_descriptive_analysis(
        research=research,
        evaluation=evaluation,
        plan=plan,
        execution_events=execution_events,
        model_predictions=predictions,
        feature_columns=ladder.feature_columns,
        config=config,
        stage=stage,
        generated_at_utc=generated_at,
    )
    _assert_execution_alignment(execution_events, selected_predictions)
    execution_events_path = stage / "execution" / "events.parquet"
    execution_events_path.parent.mkdir(parents=True, exist_ok=True)
    execution_events.write_parquet(execution_events_path)
    market_result = simulate_predictions(
        execution_events,
        selected_predictions,
        config.execution,
        order_type="market",
        seed=config.run.seed,
        markout_events=config.features.label_horizon_events,
    )
    limit_result = simulate_predictions(
        execution_events,
        selected_predictions,
        config.execution,
        order_type="limit",
        seed=config.run.seed,
        markout_events=config.features.label_horizon_events,
    )
    sensitivity = run_execution_sensitivity(
        execution_events,
        selected_predictions,
        config.execution,
        seed=config.run.seed,
        markout_events=config.features.label_horizon_events,
    )
    sensitivity_parquet = _mapping_columns_as_json(sensitivity)
    for name, frame in (
        ("market_orders", market_result.orders),
        ("market_fills", market_result.fills),
        ("market_positions", market_result.positions),
        ("limit_orders", limit_result.orders),
        ("limit_fills", limit_result.fills),
        ("limit_positions", limit_result.positions),
        ("capacity_sensitivity", sensitivity_parquet),
    ):
        frame.write_parquet(stage / "execution" / f"{name}.parquet")

    selected_comparison = ladder.comparison.filter(
        (pl.col("split") == "test") & (pl.col("model") == ladder.selected_model)
    )
    if selected_comparison.height != 1:
        raise PipelineError("selected model must have exactly one final-test comparison row")
    selected_metric = selected_comparison.to_dicts()[0]
    family = str(selected_metric["family"])
    period_start = str(selected_metric["period_start_utc"])
    period_end = str(selected_metric["period_end_utc"])
    execution_rows = [
        _execution_metric_row(
            market_result.metrics,
            model=ladder.selected_model,
            family=family,
            order_label="market",
            horizon_events=config.features.label_horizon_events,
            period_start_utc=period_start,
            period_end_utc=period_end,
            n_obs=selected_predictions.height,
        ),
        _execution_metric_row(
            limit_result.metrics,
            model=ladder.selected_model,
            family=family,
            order_label="limit",
            horizon_events=config.features.label_horizon_events,
            period_start_utc=period_start,
            period_end_utc=period_end,
            n_obs=selected_predictions.height,
        ),
    ]
    _write_json(stage / "metrics" / "execution_metrics.json", execution_rows)
    _write_json(stage / "metrics" / "execution_sensitivity.json", sensitivity.to_dicts())

    market_columns = [
        "symbol",
        "decision_ts_ns",
        "mid_price",
        "spread_bps",
        "depth_total_l1",
        "queue_imbalance_l1",
        "ofi_l1",
        "feature_ready",
    ]
    volatility_columns = [
        name for name in research.columns if name.startswith("realized_volatility_w")
    ]
    market_state = research.select(
        *market_columns,
        *volatility_columns,
        pl.lit(_SYNTHETIC_EVIDENCE).alias("evidence_tier"),
    ).sort(["decision_ts_ns", "symbol"])
    market_state_path = stage / "dashboard" / "market_state.parquet"
    market_state_path.parent.mkdir(parents=True, exist_ok=True)
    market_state.write_parquet(market_state_path)

    available_values = [
        int(cast(int, synthetic.trades.column("available_ts_ns").to_pylist()[0])),
        int(cast(int, synthetic.book_observations.column("available_ts_ns").to_pylist()[0])),
    ]
    observed_start_ns = min(available_values)
    observed_end_ns = max(
        int(cast(int, synthetic.trades.column("available_ts_ns").to_pylist()[-1])),
        int(cast(int, synthetic.book_observations.column("available_ts_ns").to_pylist()[-1])),
    )
    data_snapshot = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "source": config.data.source,
        "source_uri": "synthetic://local/deterministic-v1",
        "evidence_tier": _SYNTHETIC_EVIDENCE,
        "requested_period_utc": {
            "start": config.data.start.isoformat().replace("+00:00", "Z"),
            "end": config.data.end.isoformat().replace("+00:00", "Z") if config.data.end else None,
        },
        "observed_period_utc": {
            "start": _utc_from_ns(observed_start_ns),
            "end": _utc_from_ns(observed_end_ns),
        },
        "datasets": [
            _dataset_snapshot(trades_written, stage),
            _dataset_snapshot(books_written, stage),
        ],
    }
    _write_json(stage / "data" / "manifest_snapshot.json", data_snapshot)

    input_hashes = sorted([trades_written.manifest_sha256, books_written.manifest_sha256])
    input_identity_hashes = _input_identity_hashes((trades_written, books_written))
    git_metadata = cast(Mapping[str, Any], generated["git"])
    run_key, run_key_inputs = _run_identity(
        config_sha256=config.hash,
        input_identity_sha256=input_identity_hashes,
        git=git_metadata,
        seed=config.run.seed,
    )
    generated["input_manifest_sha256"] = input_hashes
    generated.update(
        {
            "run_key": run_key,
            "run_key_inputs": run_key_inputs,
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "seed": config.run.seed,
            "requested_evidence_tier": config.run.evidence_tier,
            "effective_evidence_tier": _SYNTHETIC_EVIDENCE,
            "observed_start_utc": _utc_from_ns(observed_start_ns),
            "observed_end_utc": _utc_from_ns(observed_end_ns),
        }
    )
    _write_json(stage / "provenance.json", generated)
    resolved_config = config.public_dict()
    resolved_config["effective_evidence_tier"] = _SYNTHETIC_EVIDENCE
    _write_json(stage / "resolved_config.json", resolved_config)

    run_manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": config.run.name,
        "run_key": run_key,
        "status": "complete",
        "evidence_tier": _SYNTHETIC_EVIDENCE,
        "data": {
            "mode": "synthetic",
            "source": config.data.source,
            "symbols": list(config.data.symbols),
            "observed_start_utc": _utc_from_ns(observed_start_ns),
            "observed_end_utc": _utc_from_ns(observed_end_ns),
            "observed_start_ts_ns": observed_start_ns,
            "observed_end_ts_ns": observed_end_ns,
        },
        "artifacts": {
            "resolved_config": "resolved_config.json",
            "data_manifest_snapshot": "data/manifest_snapshot.json",
            "quality_summary": "quality/summary.json",
            "research_frame": "research/research_frame.parquet",
            "evaluation_frame": "research/evaluation_frame.parquet",
            "folds": "research/folds.json",
            "predictions": "models/predictions.parquet",
            "selected_test_predictions": "models/selected_test_predictions.parquet",
            "predictive_metrics": "metrics/predictive_metrics.json",
            "execution_metrics": "metrics/execution_metrics.json",
            "execution_sensitivity": "metrics/execution_sensitivity.json",
            "market_state": "dashboard/market_state.parquet",
            "technical_report": "reports/technical_report.md",
            "executive_memo": "reports/executive_memo.md",
            "model_comparison": "reports/model_comparison.md",
            **analysis_artifacts,
        },
        "research": {
            **asdict(temporal_audit),
            "feature_ready_rows": evaluation.height,
            "evaluation_rows": evaluation.height,
            "evaluation_frame": "research/evaluation_frame.parquet",
            "fold_index_basis": "zero-based rows of the persisted evaluation frame",
            "selected_model": ladder.selected_model,
            "selection_metric": ladder.selection_metric,
            "feature_columns": list(ladder.feature_columns),
            "model_candidates": sorted(
                str(value) for value in ladder.comparison.get_column("model").unique()
            ),
            "descriptive_analysis": {
                "manifest": analysis_artifacts["analysis_manifest"],
                "descriptive_only": analysis_manifest["descriptive_only"],
                "threshold_source": analysis_manifest["threshold_source"],
                "event_horizons": analysis_manifest["event_horizons"],
            },
            "test_start_utc": _utc_from_ns(plan.test_start_ts_ns),
            "test_end_utc": _utc_from_ns(plan.test_end_ts_ns),
        },
        "execution_assumptions": {
            "market": market_result.assumptions,
            "limit": limit_result.assumptions,
            "event_adapter": adapter_assumptions,
        },
        "warnings": [
            "Synthetic output validates software behavior only and is not market evidence.",
            "Execution is exogenous simulation; no live order path exists.",
            (
                f"Requested evidence tier {config.run.evidence_tier} was overridden by "
                "SYNTHETIC_SMOKE because the source is synthetic."
                if config.run.evidence_tier != _SYNTHETIC_EVIDENCE
                else "Synthetic evidence tier was preserved."
            ),
        ],
    }
    _write_json(stage / "run_manifest.json", run_manifest)

    provisional = load_run_bundle(stage, require_complete=False, verify_integrity=False)
    _atomic_write_text(
        stage / "reports" / "technical_report.md",
        render_technical_report(provisional),
    )
    _atomic_write_text(
        stage / "reports" / "executive_memo.md",
        render_executive_memo(provisional),
    )
    _atomic_write_text(
        stage / "reports" / "model_comparison.md",
        render_model_comparison(provisional),
    )

    write_checksum_manifest(stage)
    success_descriptor = os.open(stage / "_SUCCESS", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(success_descriptor, "w", encoding="utf-8") as success:
        success.write("complete\n")
    load_run_bundle(stage)


def _produce(
    config: ProjectConfig,
    stage: Path,
    *,
    ingestion_manifest_path: str | Path | None,
    ingestion_manifest_sha256: str | None,
) -> None:
    if config.data.mode == "synthetic":
        _produce_synthetic(config, stage)
        return
    if config.data.mode != "binance_rest":
        raise PipelineError(
            f"no research-run producer is registered for data mode {config.data.mode!r}"
        )
    if ingestion_manifest_path is None or ingestion_manifest_sha256 is None:
        raise PipelineError(
            "public reproduction requires an explicit ingestion manifest path and SHA-256"
        )
    from microstructure.public_pipeline import produce_public_trade_run

    produce_public_trade_run(
        config,
        stage,
        ingestion_manifest_path=ingestion_manifest_path,
        ingestion_manifest_sha256=ingestion_manifest_sha256,
    )


def reproduce(
    config: ProjectConfig,
    run_dir: Path,
    *,
    ingestion_manifest_path: str | Path | None = None,
    ingestion_manifest_sha256: str | None = None,
) -> Path:
    """Produce or verify one immutable research run bundle.

    A verified completed target is reused without changing a byte. An existing
    incomplete target is never repaired or overwritten. New output is built in a
    sibling staging directory, verified, and atomically renamed into place.
    Public-data runs require an explicit content-hashed ingestion-manifest anchor;
    synthetic runs reject one so their input contract cannot be confused.
    """
    anchored = ingestion_manifest_path is not None or ingestion_manifest_sha256 is not None
    normalized_ingestion_sha256 = (
        ingestion_manifest_sha256.lower() if ingestion_manifest_sha256 is not None else None
    )
    if (ingestion_manifest_path is None) != (ingestion_manifest_sha256 is None):
        raise PipelineError("ingestion manifest path and SHA-256 must be supplied together")
    if config.data.mode == "synthetic" and anchored:
        raise PipelineError("synthetic reproduction does not accept a public input manifest")
    if config.data.mode == "binance_rest" and not anchored:
        raise PipelineError(
            "public reproduction requires an explicit ingestion manifest path and SHA-256"
        )
    if config.data.mode == "binance_rest":
        if ingestion_manifest_path is None or ingestion_manifest_sha256 is None:
            raise PipelineError("public ingestion anchor is incomplete")
        public_manifest_path = Path(ingestion_manifest_path).resolve()
        if not public_manifest_path.is_file():
            raise PipelineError(f"public ingestion manifest does not exist: {public_manifest_path}")
        observed_manifest_sha = sha256_file(public_manifest_path)
        if observed_manifest_sha != normalized_ingestion_sha256:
            raise PipelineError("public ingestion manifest bytes do not match the supplied SHA-256")

    target = run_dir.resolve()
    if target.exists():
        if target.is_dir() and (target / "_SUCCESS").is_file():
            bundle = load_run_bundle(target)
            if bundle.provenance.get("config_sha256") != config.hash:
                raise PipelineError(
                    "completed run target was produced from a different configuration"
                )
            if bundle.run_id != config.run.name:
                raise PipelineError(
                    "completed run target has a different run ID than the configuration"
                )
            current_state = git_state(config.project_root)
            current_git = {
                "commit": current_state.commit,
                "dirty": current_state.dirty,
                "source_tree_sha256": git_source_tree_sha256(config.project_root),
            }
            bundled_git = bundle.provenance.get("git")
            if not isinstance(bundled_git, Mapping) or any(
                bundled_git.get(key) != value for key, value in current_git.items()
            ):
                raise PipelineError(
                    "completed run target was produced from a different Git/source-tree state"
                )
            if (
                config.data.mode == "binance_rest"
                and bundle.provenance.get("ingestion_manifest_sha256")
                != normalized_ingestion_sha256
            ):
                raise PipelineError(
                    "completed run target was produced from a different ingestion manifest"
                )
            return target
        raise PipelineError(
            f"run target already exists but is not a verified completed bundle: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)).resolve()
    try:
        _produce(
            config,
            stage,
            ingestion_manifest_path=ingestion_manifest_path,
            ingestion_manifest_sha256=normalized_ingestion_sha256,
        )
        if target.exists():
            raise PipelineError(f"run target appeared during production: {target}")
        stage.rename(target)
        load_run_bundle(target)
        return target
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


__all__ = ["PipelineError", "reproduce"]
