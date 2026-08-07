"""Frozen public aggregate-trade research-run producer.

This producer is deliberately narrower than the synthetic vertical slice.  It
loads one explicitly manifested, bounded public trade ingestion; proves trade-ID
and availability-clock continuity before deriving research epochs; evaluates
each instrument independently; and publishes predictive diagnostics only.  It
never invokes the execution simulator or calculates P&L.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from microstructure.config import ProjectConfig
from microstructure.provenance import (
    provenance_header,
    read_json,
    sha256_file,
    write_json,
)
from microstructure.public_data import PublicTrades, read_public_trades
from microstructure.reporting import (
    load_run_bundle,
    render_executive_memo,
    render_model_comparison,
    render_technical_report,
    write_checksum_manifest,
)
from microstructure.research.analysis import (
    feature_stability_summary,
    ofi_future_return_association,
)
from microstructure.research.models import (
    block_bootstrap_metric,
    evaluate_model_ladder,
    paired_block_bootstrap_difference,
)
from microstructure.research.splits import WalkForwardPlan, expanding_walk_forward_splits
from microstructure.research.trade_only import (
    build_trade_only_research_frame,
    validate_trade_only_temporal_contract,
)

PUBLIC_PIPELINE_SCHEMA_VERSION = "1.0.0"
PUBLIC_EVIDENCE_TIER = "PUBLIC_SAMPLE_PARTIAL"
EXECUTION_EXCLUSION_REASON = (
    "Not run: aggregate-trade history has no contemporaneous quotes, depth, queue state, "
    "or local receipt clock, so execution, fees-to-alpha conversion, P&L, fills, and capacity "
    "would be unsupported claims."
)
HYPOTHESIS_BASELINE = "historical_prior"
HYPOTHESIS_BLOCK_POLICY = "fixed_contiguous_2x_label_horizon"
HYPOTHESIS_CAVEAT = (
    "Exploratory paired percentile interval on a bounded retrospective sample; it is a "
    "dependence diagnostic, not a p-value, confirmatory significance test, or basis for "
    "rejecting H0."
)
CROSS_INSTRUMENT_CONCLUSION = (
    "No pooled or cross-instrument conclusion is inferred: BTCUSDT and ETHUSDT are separate "
    "sample-specific diagnostics. They cannot support a persistent-alpha claim, regardless "
    "of their point-estimate directions."
)

__all__ = [
    "EXECUTION_EXCLUSION_REASON",
    "PUBLIC_EVIDENCE_TIER",
    "PUBLIC_PIPELINE_SCHEMA_VERSION",
    "PublicPipelineError",
    "produce_public_trade_run",
]


class PublicPipelineError(RuntimeError):
    """Raised when the frozen public run cannot be produced honestly."""


@dataclass(frozen=True, slots=True)
class _SymbolResult:
    symbol: str
    research: pl.DataFrame
    evaluation: pl.DataFrame
    plan: WalkForwardPlan
    predictions: pl.DataFrame
    comparison: pl.DataFrame
    selected_predictions: pl.DataFrame
    selected_model: str
    feature_columns: tuple[str, ...]
    temporal_audit: Mapping[str, Any]
    hypothesis_evaluation: Mapping[str, Any]


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


def _freeze_protocol(config: ProjectConfig, stage: Path) -> tuple[str, str]:
    source = config.project_root / "docs" / "PUBLIC_TRADE_PROTOCOL.md"
    if not source.is_file():
        raise PublicPipelineError(f"frozen public trade protocol is missing: {source}")
    before = sha256_file(source)
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise PublicPipelineError(f"cannot read frozen public trade protocol: {source}") from exc
    after = sha256_file(source)
    if before != after or hashlib.sha256(content).hexdigest() != before:
        raise PublicPipelineError("public trade protocol changed while it was being frozen")

    relative = "protocol/PUBLIC_TRADE_PROTOCOL.md"
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    if sha256_file(destination) != before:
        raise PublicPipelineError("frozen protocol copy does not match its source SHA-256")
    return relative, before


def _utc_from_ns(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    instant = datetime.fromtimestamp(seconds, tz=UTC)
    return f"{instant:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return cast(Mapping[str, Any], value)


def _project_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _verified_manifest_object(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    before = sha256_file(path)
    if before != expected_sha256:
        raise PublicPipelineError(f"input manifest changed after verified loading: {path}")
    payload = _mapping(read_json(path))
    after = sha256_file(path)
    if after != expected_sha256 or after != before:
        raise PublicPipelineError(f"input manifest changed while capturing lineage: {path}")
    if payload is None:
        raise PublicPipelineError(f"verified input manifest is not a JSON object: {path}")
    return payload


def _input_lineage(public: PublicTrades, config: ProjectConfig) -> dict[str, Any]:
    ingestion = _verified_manifest_object(
        public.ingestion_manifest_path, public.ingestion_manifest_sha256
    )
    dataset = _verified_manifest_object(
        public.dataset_manifest_path, public.dataset_manifest_sha256
    )

    manifest_hashes = set(public.input_manifest_sha256s)
    data_hashes = set(public.raw_artifact_sha256s)
    raw_entries = ingestion.get("raw_artifacts")
    if not isinstance(raw_entries, list):
        raise PublicPipelineError("verified ingestion manifest has no raw artifact array")
    for entry_value in raw_entries:
        entry = _mapping(entry_value)
        if entry is None:
            raise PublicPipelineError("verified raw artifact entry is not an object")
        raw_sha = entry.get("sha256")
        raw_manifest_sha = entry.get("manifest_sha256")
        if isinstance(raw_sha, str):
            data_hashes.add(raw_sha)
            raw_path = public.ingestion_manifest_path.parent.parent / str(entry.get("path"))
            if sha256_file(raw_path) != raw_sha:
                raise PublicPipelineError(f"raw input changed while capturing lineage: {raw_path}")
        if isinstance(raw_manifest_sha, str):
            manifest_hashes.add(raw_manifest_sha)
            raw_manifest_path = public.ingestion_manifest_path.parent.parent / str(
                entry.get("manifest_path")
            )
            if sha256_file(raw_manifest_path) != raw_manifest_sha:
                raise PublicPipelineError(
                    f"raw manifest changed while capturing lineage: {raw_manifest_path}"
                )

    part_entries = dataset.get("artifacts")
    if not isinstance(part_entries, list):
        raise PublicPipelineError("verified dataset manifest has no artifact array")
    for entry_value in part_entries:
        entry = _mapping(entry_value)
        if entry is None:
            raise PublicPipelineError("verified normalized artifact entry is not an object")
        data_sha = entry.get("data_sha256")
        part_manifest_sha = entry.get("manifest_sha256")
        if isinstance(data_sha, str):
            data_hashes.add(data_sha)
            data_path = public.dataset_manifest_path.parent.parent / str(entry.get("data_path"))
            if sha256_file(data_path) != data_sha:
                raise PublicPipelineError(
                    f"normalized part changed while capturing lineage: {data_path}"
                )
        if isinstance(part_manifest_sha, str):
            manifest_hashes.add(part_manifest_sha)
            part_manifest_path = public.dataset_manifest_path.parent.parent / str(
                entry.get("manifest_path")
            )
            if sha256_file(part_manifest_path) != part_manifest_sha:
                raise PublicPipelineError(
                    "normalized part manifest changed while capturing lineage: "
                    f"{part_manifest_path}"
                )

    return {
        "ingestion_manifest": {
            "absolute_path": str(public.ingestion_manifest_path.resolve()),
            "project_relative_or_absolute_path": _project_path(
                public.ingestion_manifest_path, config.project_root
            ),
            "sha256": public.ingestion_manifest_sha256,
        },
        "normalized_dataset_manifest": {
            "absolute_path": str(public.dataset_manifest_path.resolve()),
            "project_relative_or_absolute_path": _project_path(
                public.dataset_manifest_path, config.project_root
            ),
            "sha256": public.dataset_manifest_sha256,
        },
        "normalized_parts": [
            _project_path(path, config.project_root) for path in public.part_paths
        ],
        "raw_artifacts": [
            _project_path(path, config.project_root) for path in public.raw_artifact_paths
        ],
        "raw_manifests": [
            _project_path(path, config.project_root) for path in public.raw_manifest_paths
        ],
        "manifest_sha256": sorted(manifest_hashes),
        "data_sha256": sorted(data_hashes),
    }


def _derive_verified_continuity(
    public: PublicTrades, config: ProjectConfig
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if public.rows > public.row_bound:
        raise PublicPipelineError("materialized public rows exceed the verified configured bound")
    raw = public.polars_trades
    expected_symbols = set(config.data.symbols)
    if set(str(value) for value in raw.get_column("symbol").unique()) != expected_symbols:
        raise PublicPipelineError("materialized public symbols do not match the configuration")
    if raw.get_column("continuity_id").null_count() != raw.height:
        raise PublicPipelineError(
            "archive normalized rows must retain null source continuity before derivation"
        )

    derived_frames: list[pl.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for symbol in config.data.symbols:
        source = raw.filter(pl.col("symbol") == symbol).sort("trade_id")
        if source.is_empty():
            raise PublicPipelineError(f"verified public input has no rows for {symbol}")
        ids = source.get_column("trade_id").to_numpy().astype(np.int64)
        available = source.get_column("available_ts_ns").to_numpy().astype(np.int64)
        event = source.get_column("event_ts_ns").to_numpy().astype(np.int64)
        if np.any(np.diff(ids) != 1):
            raise PublicPipelineError(f"aggregate trade IDs are not contiguous by one for {symbol}")
        if np.any(np.diff(available) < 0):
            raise PublicPipelineError(
                f"availability clock reverses in aggregate-trade-ID order for {symbol}"
            )
        if np.any(available < event):
            raise PublicPipelineError(f"availability precedes event time for {symbol}")
        bases = set(str(value) for value in source.get_column("availability_basis").unique())
        if bases != {"exchange_event_time_proxy"}:
            raise PublicPipelineError(
                f"historical public trades require the exchange-event-time proxy for {symbol}"
            )
        if source.get_column("received_ts_ns").null_count() != source.height:
            raise PublicPipelineError(
                f"historical public trades cannot claim local receipt time for {symbol}"
            )

        continuity_id = (
            f"public-aggtrade:{public.ingestion_manifest_sha256[:16]}:{symbol}:"
            f"{int(ids[0])}-{int(ids[-1])}"
        )
        derived_frames.append(source.with_columns(pl.lit(continuity_id).alias("continuity_id")))
        audits.append(
            {
                "symbol": symbol,
                "rows": source.height,
                "first_trade_id": int(ids[0]),
                "last_trade_id": int(ids[-1]),
                "trade_id_step": 1,
                "trade_ids_contiguous": True,
                "availability_clock_nondecreasing": True,
                "tied_availability_rows": int(np.count_nonzero(np.diff(available) == 0)),
                "availability_basis": "exchange_event_time_proxy",
                "local_receipt_time_available": False,
                "derived_continuity_id": continuity_id,
                "derivation_timing": "assigned only after ID and clock verification",
                "source_rows_mutated": False,
            }
        )
    return pl.concat(derived_frames).sort(["symbol", "trade_id"]), audits


def _trade_feature_columns(config: ProjectConfig, frame: pl.DataFrame) -> tuple[str, ...]:
    columns: list[str] = ["log_trade_return_1"]
    for window in config.features.trade_windows:
        columns.extend(
            [
                f"signed_trade_volume_w{window}",
                f"trade_volume_w{window}",
                f"trade_imbalance_w{window}",
            ]
        )
    intensity = config.features.intensity_window
    columns.extend([f"trade_count_w{intensity}", f"trade_intensity_w{intensity}"])
    columns.append(f"realized_volatility_w{config.features.volatility_window}")
    selected = tuple(dict.fromkeys(columns))
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise PublicPipelineError(f"declared trade-only model features are missing: {missing}")
    return selected


def _serialize_plan(plan: WalkForwardPlan) -> dict[str, Any]:
    return {
        "contract": (
            "single-symbol decision-time buckets; expanding training; feature-ready and "
            "uncensored evaluation; labels ending at or beyond evaluation are purged; "
            "configured embargo applied"
        ),
        "index_basis": "zero-based row positions in this symbol's evaluation_frame.parquet",
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
        "test_used_for_selection": False,
    }


def _add_fixed_blocks(predictions: pl.DataFrame, *, block_width: int) -> pl.DataFrame:
    group = ["split", "fold_id"]
    return (
        predictions.with_columns(
            pl.col("decision_sequence").min().over(group).alias("_block_origin_trade_id")
        )
        .with_columns(
            ((pl.col("decision_sequence") - pl.col("_block_origin_trade_id")) // block_width)
            .cast(pl.Int64)
            .alias("_block_index")
        )
        .with_columns(
            pl.concat_str(
                [
                    "symbol",
                    "split",
                    pl.col("fold_id").cast(pl.String),
                    pl.col("_block_index").cast(pl.String),
                ],
                separator=":",
            ).alias("bootstrap_block"),
            (pl.col("_block_origin_trade_id") + pl.col("_block_index") * block_width).alias(
                "bootstrap_block_start_trade_id"
            ),
        )
        .with_columns(
            (pl.col("bootstrap_block_start_trade_id") + block_width - 1).alias(
                "bootstrap_block_end_trade_id"
            ),
            pl.lit(block_width, dtype=pl.Int64).alias("bootstrap_block_width_trades"),
            pl.lit("fixed_contiguous_2x_label_horizon").alias("bootstrap_block_policy"),
        )
        .drop("_block_origin_trade_id", "_block_index")
    )


def _bootstrap_comparison(
    comparison: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    symbol: str,
    selected_model: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
    horizon: int,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    block_width = 2 * horizon
    for index, source_row in enumerate(
        comparison.sort(["split", "fold_id", "requested_model"]).to_dicts()
    ):
        row = dict(source_row)
        requested_model = str(row.get("requested_model", row["model"]))
        row["selected_on_validation"] = requested_model == selected_model
        row["selected_on"] = "validation" if requested_model == selected_model else None
        row["selection_contract"] = "validation folds only; final test opened after selection"
        row["test_used_for_selection"] = False
        row["evidence_tier"] = PUBLIC_EVIDENCE_TIER
        row["execution_evaluated"] = False
        row[f"{metric}_ci_low"] = None
        row[f"{metric}_ci_high"] = None
        row["bootstrap_status"] = None
        row["bootstrap_blocks"] = None
        row["bootstrap_samples"] = None
        row["bootstrap_seed"] = None
        row["bootstrap_block_width_trades"] = None
        row["bootstrap_block_policy"] = None
        row["bootstrap_limitation"] = None
        if row["split"] == "test":
            evaluated = predictions.filter(
                (pl.col("split") == "test")
                & (pl.col("model") == str(row["model"]))
                & (pl.col("requested_model") == requested_model)
            )
            interval_seed = seed + index
            interval = block_bootstrap_metric(
                evaluated,
                metric=metric,
                block_column="bootstrap_block",
                n_bootstrap=n_bootstrap,
                seed=interval_seed,
            )
            row[f"{metric}_ci_low"] = interval.lower
            row[f"{metric}_ci_high"] = interval.upper
            row["bootstrap_status"] = interval.status
            row["bootstrap_blocks"] = interval.n_blocks
            row["bootstrap_samples"] = interval.n_bootstrap
            row["bootstrap_seed"] = interval_seed
            row["bootstrap_block_width_trades"] = block_width
            row["bootstrap_block_policy"] = "fixed_contiguous_2x_label_horizon"
            row["bootstrap_limitation"] = (
                "percentile dependence diagnostic on fixed trade blocks; not a p-value or "
                "confirmatory coverage guarantee"
            )
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(symbol).alias("instrument"),
        pl.lit(symbol).alias("instrument_scope"),
    )


def _metric_delta_direction(metric: str) -> tuple[str, int]:
    if metric in {"log_loss", "brier_score", "expected_calibration_error"}:
        return "negative_selected_minus_prior_is_favorable", -1
    if metric in {"accuracy", "balanced_accuracy", "roc_auc", "pr_auc"}:
        return "positive_selected_minus_prior_is_favorable", 1
    raise PublicPipelineError(f"unsupported paired hypothesis metric: {metric}")


def _paired_hypothesis_evaluation(
    predictions: pl.DataFrame,
    *,
    symbol: str,
    selected_model: str,
    metric: str,
    n_bootstrap: int,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    selected = predictions.filter(
        (pl.col("split") == "test") & (pl.col("requested_model") == selected_model)
    )
    baseline = predictions.filter(
        (pl.col("split") == "test") & (pl.col("requested_model") == HYPOTHESIS_BASELINE)
    )
    if selected.is_empty():
        raise PublicPipelineError(
            f"validation-selected test predictions are missing for paired test: {symbol}"
        )
    if baseline.is_empty():
        raise PublicPipelineError(f"historical-prior test predictions are missing for {symbol}")

    identity_columns = [
        "row_id",
        "y_true",
        "bootstrap_block",
        "bootstrap_block_start_trade_id",
        "bootstrap_block_end_trade_id",
        "bootstrap_block_width_trades",
        "bootstrap_block_policy",
    ]
    for name, frame in (("selected", selected), ("historical_prior", baseline)):
        missing = sorted(set(identity_columns).difference(frame.columns))
        if missing:
            raise PublicPipelineError(
                f"{name} predictions lack paired-bootstrap identity columns: {missing}"
            )
        if frame.get_column("row_id").n_unique() != frame.height:
            raise PublicPipelineError(f"{name} predictions contain duplicate row IDs for {symbol}")

    selected_identity = selected.select(identity_columns).sort("row_id")
    baseline_identity = baseline.select(identity_columns).sort("row_id")
    if not selected_identity.equals(baseline_identity):
        raise PublicPipelineError(
            f"selected and historical-prior predictions do not share identical rows/blocks for {symbol}"
        )

    block_width = 2 * horizon
    policies = selected.get_column("bootstrap_block_policy").unique().to_list()
    widths = selected.get_column("bootstrap_block_width_trades").unique().to_list()
    if policies != [HYPOTHESIS_BLOCK_POLICY] or widths != [block_width]:
        raise PublicPipelineError(
            f"paired hypothesis blocks do not implement the frozen 2x-horizon policy for {symbol}"
        )

    interval = paired_block_bootstrap_difference(
        selected,
        baseline,
        metric=metric,
        block_column="bootstrap_block",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    favorable_direction, favorable_sign = _metric_delta_direction(metric)
    signed_point = favorable_sign * interval.point_estimate
    if not math.isfinite(signed_point):
        point_assessment = "unavailable"
        point_favorable: bool | None = None
    elif signed_point > 0:
        point_assessment = "favorable_point_only"
        point_favorable = True
    elif signed_point < 0:
        point_assessment = "unfavorable_point"
        point_favorable = False
    else:
        point_assessment = "point_tie"
        point_favorable = False

    if interval.lower is None or interval.upper is None:
        interval_relation = "unavailable"
    elif interval.lower <= 0.0 <= interval.upper:
        interval_relation = "includes_zero"
    elif favorable_sign * interval.lower > 0.0 and favorable_sign * interval.upper > 0.0:
        interval_relation = "entirely_favorable"
    else:
        interval_relation = "entirely_unfavorable"

    return {
        "symbol": symbol,
        "selected_model": selected_model,
        "baseline": HYPOTHESIS_BASELINE,
        "metric": metric,
        "delta_definition": "selected_model_minus_historical_prior",
        "point_delta": interval.point_estimate,
        "ci_level": 0.95,
        "ci_low": interval.lower,
        "ci_high": interval.upper,
        "n_obs": selected.height,
        "n_blocks": interval.n_blocks,
        "samples": interval.n_bootstrap,
        "seed": interval.seed,
        "status": interval.status,
        "block_column": "bootstrap_block",
        "block_policy": HYPOTHESIS_BLOCK_POLICY,
        "block_width_trades": block_width,
        "paired_row_ids_identical": True,
        "paired_blocks_identical": True,
        "favorable_direction": favorable_direction,
        "point_favorable": point_favorable,
        "point_assessment": point_assessment,
        "interval_relation_to_zero": interval_relation,
        "exploratory": True,
        "significance_claim_authorized": False,
        "h0_rejection_authorized": False,
        "caveat": HYPOTHESIS_CAVEAT,
        "cross_instrument_conclusion": CROSS_INSTRUMENT_CONCLUSION,
    }


def _attach_paired_hypothesis_to_comparison(
    comparison: pl.DataFrame,
    hypothesis: Mapping[str, Any],
) -> pl.DataFrame:
    selected_model = str(hypothesis["selected_model"])
    rows: list[dict[str, Any]] = []
    attached = 0
    for source in comparison.to_dicts():
        row = dict(source)
        selected_test = row["split"] == "test" and row.get("requested_model") == selected_model
        paired_fields = {
            "paired_baseline": None,
            "paired_metric": None,
            "paired_metric_delta": None,
            "paired_metric_delta_ci_low": None,
            "paired_metric_delta_ci_high": None,
            "paired_n_obs": None,
            "paired_bootstrap_blocks": None,
            "paired_bootstrap_samples": None,
            "paired_bootstrap_seed": None,
            "paired_bootstrap_status": None,
            "paired_bootstrap_block_policy": None,
            "paired_favorable_direction": None,
            "paired_point_favorable": None,
            "paired_exploratory": None,
            "paired_significance_claim_authorized": None,
        }
        if selected_test:
            attached += 1
            paired_fields.update(
                {
                    "paired_baseline": hypothesis["baseline"],
                    "paired_metric": hypothesis["metric"],
                    "paired_metric_delta": hypothesis["point_delta"],
                    "paired_metric_delta_ci_low": hypothesis["ci_low"],
                    "paired_metric_delta_ci_high": hypothesis["ci_high"],
                    "paired_n_obs": hypothesis["n_obs"],
                    "paired_bootstrap_blocks": hypothesis["n_blocks"],
                    "paired_bootstrap_samples": hypothesis["samples"],
                    "paired_bootstrap_seed": hypothesis["seed"],
                    "paired_bootstrap_status": hypothesis["status"],
                    "paired_bootstrap_block_policy": hypothesis["block_policy"],
                    "paired_favorable_direction": hypothesis["favorable_direction"],
                    "paired_point_favorable": hypothesis["point_favorable"],
                    "paired_exploratory": hypothesis["exploratory"],
                    "paired_significance_claim_authorized": hypothesis[
                        "significance_claim_authorized"
                    ],
                }
            )
        row.update(paired_fields)
        rows.append(row)
    if attached != 1:
        raise PublicPipelineError(
            "paired hypothesis metadata must attach to exactly one selected test comparison row"
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _rows_by_plan(
    evaluation: pl.DataFrame, indices: np.ndarray[Any, np.dtype[np.int64]]
) -> pl.DataFrame:
    indexed = evaluation.with_row_index("_research_row_id")
    return indexed.filter(pl.col("_research_row_id").is_in(indices)).drop("_research_row_id")


def _evaluate_symbol(
    trades: pl.DataFrame,
    *,
    config: ProjectConfig,
    symbol_index: int,
) -> _SymbolResult:
    symbol = str(trades.get_column("symbol")[0])
    research = build_trade_only_research_frame(trades, config.features).with_columns(
        pl.col("label_horizon_trades").alias("label_horizon_events")
    )
    temporal = validate_trade_only_temporal_contract(research)
    evaluation = research.filter(pl.col("feature_ready"))
    if evaluation.is_empty():
        raise PublicPipelineError(f"trade-only feature construction produced no rows for {symbol}")
    features = _trade_feature_columns(config, evaluation)
    plan = expanding_walk_forward_splits(evaluation, config.evaluation)
    ladder = evaluate_model_ladder(
        evaluation,
        plan,
        config.models,
        seed=config.run.seed + symbol_index * 100_000,
        calibration_bins=config.evaluation.calibration_bins,
        target="future_trade_up",
        features=features,
    )
    block_width = 2 * config.features.label_horizon_events
    predictions = _add_fixed_blocks(ladder.predictions, block_width=block_width)
    comparison = _bootstrap_comparison(
        ladder.comparison,
        predictions,
        symbol=symbol,
        selected_model=ladder.selected_model,
        metric=ladder.selection_metric,
        n_bootstrap=config.evaluation.bootstrap_samples,
        seed=config.run.seed + symbol_index * 100_000 + 10_000,
        horizon=config.features.label_horizon_events,
    )
    selected = predictions.filter(
        (pl.col("split") == "test") & (pl.col("requested_model") == ladder.selected_model)
    )
    if selected.is_empty() or selected.get_column("requested_model").n_unique() != 1:
        raise PublicPipelineError(f"validation-selected test predictions are missing for {symbol}")
    hypothesis = _paired_hypothesis_evaluation(
        predictions,
        symbol=symbol,
        selected_model=ladder.selected_model,
        metric=ladder.selection_metric,
        n_bootstrap=config.evaluation.bootstrap_samples,
        seed=config.run.seed + symbol_index * 100_000 + 20_000,
        horizon=config.features.label_horizon_events,
    )
    comparison = _attach_paired_hypothesis_to_comparison(comparison, hypothesis)
    return _SymbolResult(
        symbol=symbol,
        research=research,
        evaluation=evaluation,
        plan=plan,
        predictions=predictions,
        comparison=comparison,
        selected_predictions=selected,
        selected_model=ladder.selected_model,
        feature_columns=features,
        temporal_audit=asdict(temporal),
        hypothesis_evaluation=hypothesis,
    )


def _trade_summary(
    trades: pl.DataFrame, continuity_audits: Sequence[Mapping[str, Any]]
) -> pl.DataFrame:
    continuity = {
        str(row["symbol"]): str(row["derived_continuity_id"]) for row in continuity_audits
    }
    rows: list[dict[str, Any]] = []
    for frame in trades.partition_by("symbol", maintain_order=True):
        symbol = str(frame.get_column("symbol")[0])
        buy = frame.filter(pl.col("aggressor_side") == "buy")
        sell = frame.filter(pl.col("aggressor_side") == "sell")
        rows.append(
            {
                "symbol": symbol,
                "rows": frame.height,
                "first_trade_id": int(cast(int, frame.get_column("trade_id").min())),
                "last_trade_id": int(cast(int, frame.get_column("trade_id").max())),
                "observed_start_utc": _utc_from_ns(
                    int(cast(int, frame.get_column("event_ts_ns").min()))
                ),
                "observed_end_utc": _utc_from_ns(
                    int(cast(int, frame.get_column("event_ts_ns").max()))
                ),
                "total_quantity": float(cast(float, frame.get_column("quantity").sum())),
                "total_quote_quantity": float(
                    cast(float, frame.get_column("quote_quantity").sum())
                ),
                "buy_rows": buy.height,
                "sell_rows": sell.height,
                "buy_quantity": float(cast(float, buy.get_column("quantity").sum())),
                "sell_quantity": float(cast(float, sell.get_column("quantity").sum())),
                "derived_continuity_id": continuity[symbol],
                "availability_basis": "exchange_event_time_proxy",
                "analysis_kind": "public_trade_sample_summary_descriptive",
                "descriptive_only": True,
            }
        )
    return pl.DataFrame(rows).sort("symbol")


def _write_analyses(
    *,
    trades: pl.DataFrame,
    symbol_results: Sequence[_SymbolResult],
    continuity_audits: Sequence[Mapping[str, Any]],
    config: ProjectConfig,
    stage: Path,
    generated_at_utc: str,
) -> dict[str, str]:
    summary = _trade_summary(trades, continuity_audits)
    stability_frames: list[pl.DataFrame] = []
    flow_frames: list[pl.DataFrame] = []
    flow_feature = f"trade_imbalance_w{max(config.features.trade_windows)}"
    for result in symbol_results:
        train = _rows_by_plan(result.evaluation, result.plan.final_train_indices)
        test = _rows_by_plan(result.evaluation, result.plan.test_indices)
        stability_frames.append(
            feature_stability_summary(
                train,
                test,
                feature_columns=result.feature_columns,
                group_columns=("symbol",),
            ).with_columns(
                pl.lit("final_training_period").alias("reference_period"),
                pl.lit("untouched_final_test").alias("comparison_period"),
                pl.lit(PUBLIC_EVIDENCE_TIER).alias("evidence_tier"),
            )
        )
        labeled = result.evaluation.filter(~pl.col("right_censored"))
        flow_frames.append(
            ofi_future_return_association(
                labeled,
                horizon_return_columns={
                    config.features.label_horizon_events: "future_trade_return"
                },
                ofi_column=flow_feature,
                min_observations=3,
            ).with_columns(
                pl.lit("all_feature_ready_labeled_rows").alias("analysis_scope"),
                pl.lit(PUBLIC_EVIDENCE_TIER).alias("evidence_tier"),
            )
        )
    stability = pl.concat(stability_frames)
    flow_return = pl.concat(flow_frames)
    artifacts = {
        "trade_summary": "analysis/trade_summary.parquet",
        "feature_stability": "analysis/feature_stability.parquet",
        "flow_return_analysis": "analysis/flow_return_analysis.parquet",
    }
    for name, frame in (
        ("trade_summary", summary),
        ("feature_stability", stability),
        ("flow_return_analysis", flow_return),
    ):
        path = stage / artifacts[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    _write_json(
        stage / "analysis" / "manifest.json",
        {
            "generated_at_utc": generated_at_utc,
            "evidence_tier": PUBLIC_EVIDENCE_TIER,
            "descriptive_only": True,
            "economic_claim_authorized": False,
            "execution_claim_authorized": False,
            "artifacts": {
                "trade_summary": {"path": artifacts["trade_summary"], "rows": summary.height},
                "feature_stability": {
                    "path": artifacts["feature_stability"],
                    "rows": stability.height,
                    "reference": "final training period",
                    "comparison": "untouched final test",
                },
                "flow_return_analysis": {
                    "path": artifacts["flow_return_analysis"],
                    "rows": flow_return.height,
                    "scope": "all feature-ready labeled rows",
                },
            },
            "limitations": [
                "Retrospective bounded sample; no confirmatory significance claim.",
                "Exchange event time is an availability proxy, not local receipt evidence.",
                "Trade-only observations cannot support book or execution analysis.",
            ],
        },
    )
    return {**artifacts, "analysis_manifest": "analysis/manifest.json"}


def _quality_payload(
    public: PublicTrades,
    continuity_audits: Sequence[Mapping[str, Any]],
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc,
        "dataset": "manifested_public_aggregate_trades",
        "rows_checked": public.validation.rows_checked,
        "summary": {
            "errors": public.validation.error_count,
            "warnings": public.validation.warning_count,
        },
        "normalized_schema_validation": {
            "dataset": public.validation.dataset,
            "rows_checked": public.validation.rows_checked,
            "errors": public.validation.error_count,
            "warnings": public.validation.warning_count,
            "findings": [asdict(finding) for finding in public.validation.findings],
        },
        "aggregate_trade_continuity": list(continuity_audits),
        "row_bound": public.row_bound,
        "row_bound_respected": public.rows <= public.row_bound,
        "canonical_source_order": list(public.canonical_order),
        "mutation_policy": (
            "verified normalized observations were not repaired or overwritten; continuity "
            "exists only in persisted derived research frames"
        ),
    }


def _hypothesis_number(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return f"{number:.6f}" if math.isfinite(number) else "N/A"
    return str(value)


def _render_hypothesis_report_section(stage: Path) -> str:
    artifact = stage / "metrics" / "hypothesis_evaluation.json"
    payload = _mapping(read_json(artifact))
    if payload is None:
        raise PublicPipelineError("serialized hypothesis evaluation must be a JSON object")
    rows = payload.get("per_symbol")
    if not isinstance(rows, list) or not rows:
        raise PublicPipelineError("serialized hypothesis evaluation has no per-symbol rows")

    lines = [
        "## Paired H0/H1 diagnostic",
        "",
        (
            "The frozen comparison is validation-selected test predictions versus the "
            "historical-prior baseline on identical `row_id` and fixed 2x-horizon blocks. "
            "For Δ log-loss (selected minus historical prior), a negative value favors the "
            "selected model."
        ),
        "",
        (
            "| Symbol | Selected model | Baseline | Metric | Point Δ | Paired 95% interval | "
            "Observations | Blocks | Samples | Seed | Status |"
        ),
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for value in rows:
        row = _mapping(value)
        if row is None:
            raise PublicPipelineError("serialized per-symbol hypothesis row is not an object")
        interval = (
            f"[{_hypothesis_number(row.get('ci_low'))}, {_hypothesis_number(row.get('ci_high'))}]"
        )
        cells = (
            row.get("symbol"),
            row.get("selected_model"),
            row.get("baseline"),
            row.get("metric"),
            _hypothesis_number(row.get("point_delta")),
            interval,
            row.get("n_obs"),
            row.get("n_blocks"),
            row.get("samples"),
            row.get("seed"),
            row.get("status"),
        )
        lines.append(
            "| "
            + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells)
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "These paired percentile intervals are dependence diagnostics, not p-values "
                "or confirmatory significance intervals; H0 is not rejected by this "
                "exploratory design."
            ),
            "",
            (
                "No cross-instrument estimate was pooled. Mixed directions, if present, "
                "cannot support persistent alpha; even matching directions would remain "
                "bounded, sample-specific diagnostics."
            ),
        ]
    )
    return "\n".join(lines)


def _report_set(stage: Path) -> None:
    bundle = load_run_bundle(stage, require_complete=False, verify_integrity=False)
    hypothesis_section = _render_hypothesis_report_section(stage)
    _atomic_write_text(
        stage / "reports" / "technical_report.md",
        render_technical_report(bundle),
    )
    _atomic_write_text(
        stage / "reports" / "executive_memo.md",
        render_executive_memo(bundle),
    )
    _atomic_write_text(
        stage / "reports" / "model_comparison.md",
        render_model_comparison(bundle)
        + "\nExecution status: `NOT_RUN`. Reason: "
        + EXECUTION_EXCLUSION_REASON
        + "\n\n"
        + hypothesis_section
        + "\n",
    )


def produce_public_trade_run(
    config: ProjectConfig,
    stage: Path,
    *,
    ingestion_manifest_path: str | Path,
    ingestion_manifest_sha256: str,
) -> None:
    """Populate an empty atomic stage with one verified public trade-only run.

    The caller owns staging-directory creation, cleanup on failure, and final
    rename.  This function writes ``_SUCCESS`` only after every artifact, report,
    and checksum is durable in the supplied stage.
    """

    destination = stage.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise PublicPipelineError("public run stage must be empty")
    destination.mkdir(parents=True, exist_ok=True)

    if config.data.mode != "binance_rest":
        raise PublicPipelineError("public trade producer requires data.mode='binance_rest'")
    explicit_manifest_path = Path(ingestion_manifest_path).resolve()
    public = read_public_trades(
        config,
        explicit_manifest_path,
        ingestion_manifest_sha256=ingestion_manifest_sha256,
    )
    if public.evidence_tier == "FULL_DATA" and not public.all_requested_ranges_complete:
        raise PublicPipelineError("reader evidence tier contradicts manifested completeness")
    derived_trades, continuity_audits = _derive_verified_continuity(public, config)
    if public.validation.has_errors:
        raise PublicPipelineError("verified public normalized input has quality errors")

    generated = provenance_header(
        project_root=config.project_root,
        config_hash=config.hash,
        evidence_tier=PUBLIC_EVIDENCE_TIER,
        input_manifests=[],
    )
    generated_at = cast(str, generated["generated_at_utc"])
    lineage = _input_lineage(public, config)
    manifest_hashes = cast(list[str], lineage["manifest_sha256"])
    data_hashes = cast(list[str], lineage["data_sha256"])
    protocol_path, protocol_sha256 = _freeze_protocol(config, destination)

    symbol_results: list[_SymbolResult] = []
    for symbol_index, symbol in enumerate(config.data.symbols):
        symbol_results.append(
            _evaluate_symbol(
                derived_trades.filter(pl.col("symbol") == symbol),
                config=config,
                symbol_index=symbol_index,
            )
        )

    research_root = destination / "research"
    model_root = destination / "models"
    combined_research: list[pl.DataFrame] = []
    combined_evaluation: list[pl.DataFrame] = []
    combined_predictions: list[pl.DataFrame] = []
    combined_comparison: list[pl.DataFrame] = []
    combined_selected: list[pl.DataFrame] = []
    symbol_manifest: dict[str, Any] = {}
    for result in symbol_results:
        slug = result.symbol.lower()
        symbol_research = research_root / slug
        symbol_models = model_root / slug
        symbol_research.mkdir(parents=True, exist_ok=True)
        symbol_models.mkdir(parents=True, exist_ok=True)
        result.research.write_parquet(symbol_research / "research_frame.parquet")
        result.evaluation.write_parquet(symbol_research / "evaluation_frame.parquet")
        _write_json(symbol_research / "folds.json", _serialize_plan(result.plan))
        result.predictions.write_parquet(symbol_models / "predictions.parquet")
        result.comparison.write_parquet(symbol_models / "comparison.parquet")
        result.selected_predictions.write_parquet(
            symbol_models / "selected_test_predictions.parquet"
        )
        combined_research.append(result.research)
        combined_evaluation.append(result.evaluation)
        combined_predictions.append(result.predictions)
        combined_comparison.append(result.comparison)
        combined_selected.append(result.selected_predictions)
        symbol_manifest[result.symbol] = {
            "research_frame": f"research/{slug}/research_frame.parquet",
            "evaluation_frame": f"research/{slug}/evaluation_frame.parquet",
            "folds": f"research/{slug}/folds.json",
            "predictions": f"models/{slug}/predictions.parquet",
            "comparison": f"models/{slug}/comparison.parquet",
            "selected_test_predictions": f"models/{slug}/selected_test_predictions.parquet",
            "selected_model": result.selected_model,
            "selection_metric": config.models.selection_metric,
            "selection_source": "validation folds only",
            "test_used_for_selection": False,
            "feature_columns": list(result.feature_columns),
            "temporal_audit": dict(result.temporal_audit),
            "paired_hypothesis_evaluation": dict(result.hypothesis_evaluation),
            "test_start_utc": _utc_from_ns(result.plan.test_start_ts_ns),
            "test_end_utc": _utc_from_ns(result.plan.test_end_ts_ns),
        }

    research = pl.concat(combined_research)
    evaluation = pl.concat(combined_evaluation)
    predictions = pl.concat(combined_predictions)
    comparison = pl.concat(combined_comparison)
    selected_predictions = pl.concat(combined_selected)
    research.write_parquet(research_root / "research_frame.parquet")
    evaluation.write_parquet(research_root / "evaluation_frame.parquet")
    predictions.write_parquet(model_root / "predictions.parquet")
    comparison.write_parquet(model_root / "comparison.parquet")
    selected_predictions.write_parquet(model_root / "selected_test_predictions.parquet")

    metrics_root = destination / "metrics"
    _write_json(metrics_root / "predictive_metrics.json", comparison.to_dicts())
    hypothesis_rows = [dict(result.hypothesis_evaluation) for result in symbol_results]
    hypothesis_payload = {
        "schema_version": PUBLIC_PIPELINE_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "evidence_tier": PUBLIC_EVIDENCE_TIER,
        "hypotheses": {
            "H0": (
                "The validation-selected model does not improve held-out selection-metric "
                "performance over the historical-prior classifier."
            ),
            "H1_exploratory": (
                "Causal aggregate-trade features improve held-out selection-metric "
                "performance relative to the historical prior."
            ),
        },
        "comparison_contract": (
            "per-symbol validation-selected final-test predictions minus historical-prior "
            "predictions on identical row_id and fixed dependency block"
        ),
        "selection_metric": config.models.selection_metric,
        "delta_definition": "selected_model_minus_historical_prior",
        "bootstrap": {
            "method": "paired fixed-block percentile bootstrap",
            "ci_level": 0.95,
            "samples": config.evaluation.bootstrap_samples,
            "seed_policy": "run_seed + symbol_index*100000 + 20000",
            "block_policy": HYPOTHESIS_BLOCK_POLICY,
            "block_width_trades": 2 * config.features.label_horizon_events,
        },
        "per_symbol": hypothesis_rows,
        "cross_instrument_conclusion": {
            "status": "not_inferred",
            "pooling_performed": False,
            "persistent_alpha_claim_authorized": False,
            "text": CROSS_INSTRUMENT_CONCLUSION,
        },
        "exploratory": True,
        "significance_claim_authorized": False,
        "caveat": HYPOTHESIS_CAVEAT,
    }
    _write_json(metrics_root / "hypothesis_evaluation.json", hypothesis_payload)
    _write_json(metrics_root / "execution_metrics.json", [])
    _write_json(metrics_root / "execution_sensitivity.json", [])
    _write_json(
        metrics_root / "execution_exclusion.json",
        {
            "status": "NOT_RUN",
            "reason": EXECUTION_EXCLUSION_REASON,
            "execution_metrics_rows": 0,
            "execution_sensitivity_rows": 0,
            "pnl_calculated": False,
            "profitability_claim_authorized": False,
        },
    )

    analysis_artifacts = _write_analyses(
        trades=derived_trades,
        symbol_results=symbol_results,
        continuity_audits=continuity_audits,
        config=config,
        stage=destination,
        generated_at_utc=generated_at,
    )
    quality = _quality_payload(public, continuity_audits, generated_at_utc=generated_at)
    _write_json(destination / "quality" / "summary.json", quality)

    market_state = evaluation.select(
        "symbol",
        "decision_ts_ns",
        "decision_trade_id",
        "price",
        "quantity",
        "aggressor_side",
        f"trade_imbalance_w{max(config.features.trade_windows)}",
        f"realized_volatility_w{config.features.volatility_window}",
        pl.lit(PUBLIC_EVIDENCE_TIER).alias("evidence_tier"),
        pl.lit("trade_only_no_book_state").alias("market_state_scope"),
    ).sort(["decision_ts_ns", "symbol", "decision_trade_id"])
    dashboard_path = destination / "dashboard" / "market_state.parquet"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    market_state.write_parquet(dashboard_path)

    data_snapshot = {
        "schema_version": PUBLIC_PIPELINE_SCHEMA_VERSION,
        "mode": "binance_rest_trade_only",
        "source": config.data.source,
        "evidence_tier": PUBLIC_EVIDENCE_TIER,
        "reader_effective_evidence_tier": public.evidence_tier,
        "configured_requested_evidence_tier": config.run.evidence_tier,
        "producer_evidence_policy": (
            "always PUBLIC_SAMPLE_PARTIAL for this retrospective exploratory protocol"
        ),
        "requested_period_utc": {
            "start": config.data.start.isoformat().replace("+00:00", "Z"),
            "end": config.data.end.isoformat().replace("+00:00", "Z") if config.data.end else None,
        },
        "observed_period_utc": {
            "start": public.observed.start_utc,
            "end_inclusive": public.observed.end_inclusive_utc,
        },
        "rows": public.rows,
        "row_bound": public.row_bound,
        "all_requested_ranges_complete": public.all_requested_ranges_complete,
        "canonical_order": list(public.canonical_order),
        "manifest_authority": {
            "policy": "explicit path and caller-supplied SHA-256; no directory discovery",
            "absolute_path": str(public.ingestion_manifest_path.resolve()),
            "project_relative_or_absolute_path": _project_path(
                public.ingestion_manifest_path, config.project_root
            ),
            "sha256": public.ingestion_manifest_sha256,
        },
        "lineage": lineage,
        "symbols": [
            {
                "symbol": item.symbol,
                "rows": item.rows,
                "complete_range": item.complete_range,
                "tick_size": str(item.tick_size),
                "lot_size": str(item.lot_size),
                "observed_start_utc": item.observed.start_utc,
                "observed_end_inclusive_utc": item.observed.end_inclusive_utc,
            }
            for item in public.symbols
        ],
        "transformation": (
            "source normalized rows unchanged; one derived continuity epoch per symbol was "
            "assigned only after contiguous aggregate-ID and nonreversing-clock checks"
        ),
    }
    _write_json(destination / "data" / "manifest_snapshot.json", data_snapshot)

    resolved_config = config.public_dict()
    resolved_config["effective_evidence_tier"] = PUBLIC_EVIDENCE_TIER
    resolved_config["reader_effective_evidence_tier"] = public.evidence_tier
    resolved_config["evidence_policy"] = (
        "retrospective public trade protocol cannot be promoted above PUBLIC_SAMPLE_PARTIAL"
    )
    resolved_config["protocol"] = {
        "path": protocol_path,
        "sha256": protocol_sha256,
    }
    _write_json(destination / "resolved_config.json", resolved_config)

    git_metadata = cast(Mapping[str, Any], generated["git"])
    run_key_inputs = {
        "config_sha256": config.hash,
        "input_manifest_sha256": manifest_hashes,
        "input_data_sha256": data_hashes,
        "protocol_sha256": protocol_sha256,
        "git": {
            "commit": str(git_metadata.get("commit", "UNKNOWN")),
            "dirty": bool(git_metadata.get("dirty", False)),
            "source_tree_sha256": str(git_metadata.get("source_tree_sha256", "UNKNOWN")),
        },
        "seed": config.run.seed,
        "protocol": "public_aggregate_trade_exploratory_v1",
    }
    run_key = _stable_sha256(run_key_inputs)
    generated.update(
        {
            "evidence_tier": PUBLIC_EVIDENCE_TIER,
            "requested_evidence_tier": config.run.evidence_tier,
            "effective_evidence_tier": PUBLIC_EVIDENCE_TIER,
            "reader_effective_evidence_tier": public.evidence_tier,
            "input_manifest_sha256": manifest_hashes,
            "input_data_sha256": data_hashes,
            "ingestion_manifest_path": _project_path(
                public.ingestion_manifest_path, config.project_root
            ),
            "ingestion_manifest_absolute_path": str(public.ingestion_manifest_path.resolve()),
            "ingestion_manifest_sha256": public.ingestion_manifest_sha256,
            "ingestion_manifest_authority": (
                "explicit path and caller-supplied SHA-256; no directory discovery"
            ),
            "protocol_path": protocol_path,
            "protocol_sha256": protocol_sha256,
            "run_key": run_key,
            "run_key_inputs": run_key_inputs,
            "pipeline_schema_version": PUBLIC_PIPELINE_SCHEMA_VERSION,
            "seed": config.run.seed,
            "observed_start_utc": public.observed.start_utc,
            "observed_end_utc": public.observed.end_inclusive_utc,
            "data_availability_clock": "exchange_event_time_proxy",
            "local_receipt_time_available": False,
            "execution_simulated": False,
        }
    )
    _write_json(destination / "provenance.json", generated)

    artifacts = {
        "resolved_config": "resolved_config.json",
        "protocol": protocol_path,
        "data_manifest_snapshot": "data/manifest_snapshot.json",
        "quality_summary": "quality/summary.json",
        "research_frame": "research/research_frame.parquet",
        "evaluation_frame": "research/evaluation_frame.parquet",
        "predictions": "models/predictions.parquet",
        "selected_test_predictions": "models/selected_test_predictions.parquet",
        "model_comparison_data": "models/comparison.parquet",
        "predictive_metrics": "metrics/predictive_metrics.json",
        "hypothesis_evaluation": "metrics/hypothesis_evaluation.json",
        "execution_metrics": "metrics/execution_metrics.json",
        "execution_sensitivity": "metrics/execution_sensitivity.json",
        "execution_exclusion": "metrics/execution_exclusion.json",
        "market_state": "dashboard/market_state.parquet",
        "technical_report": "reports/technical_report.md",
        "executive_memo": "reports/executive_memo.md",
        "model_comparison": "reports/model_comparison.md",
        **analysis_artifacts,
    }
    run_manifest = {
        "schema_version": PUBLIC_PIPELINE_SCHEMA_VERSION,
        "run_id": config.run.name,
        "run_key": run_key,
        "status": "complete",
        "evidence_tier": PUBLIC_EVIDENCE_TIER,
        "data": {
            "mode": "binance_rest_trade_only",
            "source": config.data.source,
            "symbols": list(config.data.symbols),
            "rows": public.rows,
            "row_bound": public.row_bound,
            "all_requested_ranges_complete": public.all_requested_ranges_complete,
            "reader_effective_evidence_tier": public.evidence_tier,
            "configured_requested_evidence_tier": config.run.evidence_tier,
            "observed_start_utc": public.observed.start_utc,
            "observed_end_utc": public.observed.end_inclusive_utc,
            "observed_start_ts_ns": public.observed.start_ns,
            "observed_end_ts_ns": public.observed.end_inclusive_ns,
            "availability_basis": "exchange_event_time_proxy",
            "local_receipt_time_available": False,
            "symbol_coverage": [
                {
                    "symbol": item.symbol,
                    "rows": item.rows,
                    "complete_range": item.complete_range,
                    "observed_start_utc": item.observed.start_utc,
                    "observed_end_inclusive_utc": item.observed.end_inclusive_utc,
                }
                for item in public.symbols
            ],
        },
        "artifacts": artifacts,
        "research": {
            "question": (
                "whether recent observable aggregate-trade direction and size contain "
                "out-of-time information about future trade-price direction"
            ),
            "target": "future_trade_up",
            "label_horizon_trades": config.features.label_horizon_events,
            "evaluation_contract": "separate per-symbol expanding purged walk-forward",
            "selection_contract": "validation folds only; final test never used for selection",
            "bootstrap_contract": {
                "samples": config.evaluation.bootstrap_samples,
                "seeded": True,
                "block_width_trades": 2 * config.features.label_horizon_events,
                "block_policy": "fixed_contiguous_2x_label_horizon",
                "status": "dependence diagnostic, not confirmatory significance",
            },
            "hypothesis_evaluation": {
                "artifact": "metrics/hypothesis_evaluation.json",
                "hypotheses": ["H0", "H1_exploratory"],
                "baseline": HYPOTHESIS_BASELINE,
                "metric": config.models.selection_metric,
                "delta_definition": "selected_model_minus_historical_prior",
                "paired_on": ["row_id", "bootstrap_block"],
                "per_symbol_only": True,
                "cross_instrument_pooling": False,
                "persistent_alpha_claim_authorized": False,
                "exploratory": True,
                "significance_claim_authorized": False,
                "caveat": HYPOTHESIS_CAVEAT,
                "cross_instrument_conclusion": CROSS_INSTRUMENT_CONCLUSION,
            },
            "symbols": symbol_manifest,
            "descriptive_analysis": {
                "manifest": analysis_artifacts["analysis_manifest"],
                "descriptive_only": True,
                "economic_claim_authorized": False,
            },
        },
        "execution_assumptions": {
            "status": "NOT_RUN",
            "reason": EXECUTION_EXCLUSION_REASON,
            "pnl_calculated": False,
            "fills_calculated": False,
            "capacity_calculated": False,
            "profitability_claim_authorized": False,
        },
        "warnings": [
            "PUBLIC_SAMPLE_PARTIAL: results are bounded, retrospective, and sample-specific.",
            "Exchange event time is an availability proxy and not local receipt evidence.",
            EXECUTION_EXCLUSION_REASON,
            "Bootstrap intervals are dependence diagnostics and not significance claims.",
            CROSS_INSTRUMENT_CONCLUSION,
        ],
    }
    _write_json(destination / "run_manifest.json", run_manifest)
    _report_set(destination)
    if _input_lineage(public, config) != lineage:
        raise PublicPipelineError("external input lineage changed while producing the run")
    write_checksum_manifest(destination)
    descriptor = os.open(destination / "_SUCCESS", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as success:
        success.write("complete\n")
    load_run_bundle(destination)
