"""Transparent model ladder, calibration, metrics, and dependent bootstrap."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin  # type: ignore[import-untyped]
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from microstructure.config import ModelConfig
from microstructure.research.features import model_feature_columns
from microstructure.research.splits import WalkForwardPlan


class ModelEvaluationError(ValueError):
    """Raised when a model evaluation would be invalid or underidentified."""


ModelFamily = Literal["baseline", "logistic", "logistic_l2", "shallow_tree"]


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """One predeclared member of the transparent classification ladder."""

    name: str
    family: ModelFamily
    c: float | None = None
    max_depth: int | None = None
    min_samples_leaf: int = 1


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A deterministic block-bootstrap percentile interval."""

    point_estimate: float
    lower: float | None
    upper: float | None
    n_bootstrap: int
    n_blocks: int
    seed: int
    status: Literal["ok", "insufficient_blocks"]
    draws: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ModelLadderResult:
    """Out-of-time predictions and fold/final-test comparison rows."""

    predictions: pl.DataFrame
    comparison: pl.DataFrame
    selected_model: str
    feature_columns: tuple[str, ...]
    selection_metric: str


@dataclass(slots=True)
class SigmoidCalibrator:
    """Platt-style calibration fitted only on a chronological calibration tail."""

    estimator: LogisticRegression | None = None
    status: str = "identity_not_fitted"

    def fit(self, y_true: NDArray[np.int64], raw_probability: NDArray[np.float64]) -> None:
        if y_true.size < 8 or np.unique(y_true).size < 2:
            self.status = "identity_insufficient_calibration_data"
            return
        transformed = _logit(raw_probability).reshape(-1, 1)
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000)
        estimator.fit(transformed, y_true)
        self.estimator = estimator
        self.status = "sigmoid"

    def transform(self, raw_probability: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.estimator is None:
            return np.asarray(np.clip(raw_probability, 1e-12, 1.0 - 1e-12), dtype=np.float64)
        probability = self.estimator.predict_proba(_logit(raw_probability).reshape(-1, 1))[:, 1]
        return np.asarray(probability, dtype=np.float64)


def _logit(probability: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_model_candidates(config: ModelConfig) -> tuple[ModelCandidate, ...]:
    """Expand the typed configuration into a stable, auditable model ladder."""

    candidates: list[ModelCandidate] = [ModelCandidate("historical_prior", "baseline")]
    candidates.append(ModelCandidate(name="logistic_unpenalized", family="logistic"))
    candidates.extend(
        ModelCandidate(name=f"logistic_l2_c_{value:g}", family="logistic_l2", c=value)
        for value in config.logistic_c_values
    )
    candidates.extend(
        ModelCandidate(
            name=f"tree_depth_{depth}",
            family="shallow_tree",
            max_depth=depth,
            min_samples_leaf=config.tree_min_samples_leaf,
        )
        for depth in config.tree_max_depth_values
    )
    return tuple(candidates)


def make_classifier(candidate: ModelCandidate, *, seed: int) -> Pipeline:
    """Construct a CPU-light classifier with train-only preprocessing."""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if candidate.family == "baseline":
        model: ClassifierMixin = DummyClassifier(strategy="prior")
        return Pipeline([("imputer", imputer), ("model", model)])
    if candidate.family in {"logistic", "logistic_l2"}:
        if candidate.family == "logistic_l2" and (candidate.c is None or candidate.c <= 0):
            raise ModelEvaluationError("regularized logistic C must be positive")
        model = LogisticRegression(
            C=np.inf if candidate.family == "logistic" else candidate.c,
            solver="lbfgs",
            max_iter=2_000,
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("scale", StandardScaler()), ("model", model)])
    if candidate.family == "shallow_tree":
        if candidate.max_depth is None or candidate.max_depth < 1:
            raise ModelEvaluationError("tree max_depth must be positive")
        model = DecisionTreeClassifier(
            max_depth=candidate.max_depth,
            min_samples_leaf=candidate.min_samples_leaf,
            random_state=seed,
        )
        return Pipeline([("imputer", imputer), ("model", model)])
    raise ModelEvaluationError(f"unsupported model family: {candidate.family}")


def expected_calibration_error(
    y_true: NDArray[np.int64],
    probability: NDArray[np.float64],
    *,
    bins: int,
) -> float:
    """Return fixed-width expected calibration error."""

    if bins < 1:
        raise ModelEvaluationError("calibration bins must be positive")
    if y_true.size == 0:
        return math.nan
    probability = np.clip(probability, 0.0, 1.0)
    assignments = np.digitize(probability, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    result = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(probability[mask].mean())
            )
    return result


def classification_metrics(
    y_true: NDArray[np.int64],
    probability: NDArray[np.float64],
    *,
    calibration_bins: int,
) -> dict[str, float]:
    """Compute proper scoring, discrimination, and calibration metrics."""

    if y_true.size == 0 or y_true.shape != probability.shape:
        raise ModelEvaluationError("metric inputs must be equally sized and nonempty")
    if not np.isin(y_true, [0, 1]).all():
        raise ModelEvaluationError("classification target must contain only 0 and 1")
    probability = np.clip(probability.astype(np.float64), 1e-12, 1.0 - 1e-12)
    prediction = (probability >= 0.5).astype(np.int64)
    two_classes = np.unique(y_true).size == 2
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, prediction)) if two_classes else math.nan
        ),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if two_classes else math.nan,
        "pr_auc": (
            float(average_precision_score(y_true, probability)) if two_classes else math.nan
        ),
        "expected_calibration_error": expected_calibration_error(
            y_true, probability, bins=calibration_bins
        ),
        "positive_rate": float(y_true.mean()),
    }


def _positive_probability(
    estimator: Pipeline, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    if classes.size == 1:
        return np.full(features.shape[0], float(classes[0] == 1), dtype=np.float64)
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise ModelEvaluationError("classifier does not expose a binary positive class")
    return np.asarray(probabilities[:, int(positive[0])], dtype=np.float64)


def _rows(frame: pl.DataFrame, indices: NDArray[np.int64]) -> pl.DataFrame:
    return frame.filter(pl.col("_research_row_id").is_in(indices))


def _chronological_calibration_split(
    train: pl.DataFrame,
    *,
    fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0.0 < fraction < 0.5:
        raise ModelEvaluationError("calibration_fraction must be between zero and one half")
    times = sorted(train.get_column("decision_ts_ns").unique().to_list())
    if len(times) < 6:
        return train, train.head(0)
    calibration_count = max(2, math.ceil(len(times) * fraction))
    calibration_start = int(times[-calibration_count])
    base = train.filter(
        (pl.col("decision_ts_ns") < calibration_start)
        & (pl.col("label_information_end_ts_ns") < calibration_start)
    )
    calibration = train.filter(pl.col("decision_ts_ns") >= calibration_start)
    if base.height < 4 or calibration.height < 8:
        return train, train.head(0)
    return base, calibration


def _fit_candidate(
    candidate: ModelCandidate,
    train: pl.DataFrame,
    evaluate: pl.DataFrame,
    *,
    features: tuple[str, ...],
    target: str,
    seed: int,
    calibration_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, int, ModelCandidate]:
    base, calibration = _chronological_calibration_split(train, fraction=calibration_fraction)
    x_base = base.select(features).to_numpy().astype(np.float64)
    y_base = base.get_column(target).to_numpy().astype(np.int64)
    fit_status = "ok"
    effective_candidate = candidate
    if np.unique(y_base).size < 2 and candidate.family != "baseline":
        effective_candidate = ModelCandidate(
            name=f"{candidate.name}__prior_fallback",
            family="baseline",
        )
        fit_status = "single_class_prior_fallback"
    estimator = make_classifier(effective_candidate, seed=seed)
    estimator.fit(x_base, y_base)

    calibrator = SigmoidCalibrator()
    if not calibration.is_empty():
        x_calibration = calibration.select(features).to_numpy().astype(np.float64)
        y_calibration = calibration.get_column(target).to_numpy().astype(np.int64)
        calibrator.fit(y_calibration, _positive_probability(estimator, x_calibration))
    x_evaluate = evaluate.select(features).to_numpy().astype(np.float64)
    raw_probability = _positive_probability(estimator, x_evaluate)
    probability = calibrator.transform(raw_probability)

    fitting_rows = pl.concat([base, calibration]) if not calibration.is_empty() else base
    fit_cutoff_value = fitting_rows.get_column("label_information_end_ts_ns").max()
    if fit_cutoff_value is None:
        raise ModelEvaluationError("training rows have no observable labels")
    status = f"{fit_status};{calibrator.status}"
    return raw_probability, probability, status, cast(int, fit_cutoff_value), effective_candidate


def _prediction_rows(
    evaluated: pl.DataFrame,
    *,
    candidate: ModelCandidate,
    requested_candidate: ModelCandidate,
    fold_id: int,
    split: Literal["validation", "test"],
    target: str,
    raw_probability: NDArray[np.float64],
    probability: NDArray[np.float64],
    fit_cutoff_ts_ns: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, row in enumerate(evaluated.iter_rows(named=True)):
        decision_ts_ns = int(row["decision_ts_ns"])
        symbol = str(row.get("symbol", "UNKNOWN"))
        decision_sequence = int(row.get("decision_sequence", row["_research_row_id"]))
        sample_id = str(row.get("sample_id", f"{symbol}:{decision_ts_ns}:{decision_sequence}"))
        if fit_cutoff_ts_ns >= decision_ts_ns:
            raise ModelEvaluationError("model fitting information reaches the evaluation decision")
        result.append(
            {
                "row_id": int(row["_research_row_id"]),
                "sample_id": sample_id,
                "symbol": symbol,
                "instrument": symbol,
                "decision_ts_ns": decision_ts_ns,
                "decision_sequence": decision_sequence,
                "continuity_id": str(row.get("continuity_id", "UNKNOWN")),
                "fold_id": fold_id,
                "split": split,
                "model": candidate.name,
                "family": candidate.family,
                "requested_model": requested_candidate.name,
                "requested_family": requested_candidate.family,
                "y_true": int(row[target]),
                "raw_probability": float(raw_probability[index]),
                "probability": float(probability[index]),
                "predicted_class": int(probability[index] >= 0.5),
                "fit_cutoff_ts_ns": fit_cutoff_ts_ns,
                "is_oos": True,
            }
        )
    return result


def _metric_row(
    *,
    candidate: ModelCandidate,
    requested_candidate: ModelCandidate,
    fold_id: int,
    split: Literal["validation", "test"],
    y_true: NDArray[np.int64],
    probability: NDArray[np.float64],
    calibration_bins: int,
    fit_status: str,
    evaluated: pl.DataFrame,
    horizon_events: int | None,
) -> dict[str, object]:
    metrics = classification_metrics(y_true, probability, calibration_bins=calibration_bins)
    period_start = cast(int, evaluated.get_column("decision_ts_ns").min())
    period_end = cast(int, evaluated.get_column("decision_ts_ns").max())
    instruments = sorted(str(value) for value in evaluated.get_column("symbol").unique())
    instrument_scope = instruments[0] if len(instruments) == 1 else "POOLED"
    return {
        "model": candidate.name,
        "family": candidate.family,
        "requested_model": requested_candidate.name,
        "requested_family": requested_candidate.family,
        "symbol": instrument_scope,
        "instrument": instrument_scope,
        "instrument_scope": instrument_scope,
        "horizon_events": horizon_events,
        "fold_id": fold_id,
        "split": split,
        "period_start_ts_ns": period_start,
        "period_end_ts_ns": period_end,
        "period_start_utc": _ns_to_utc(period_start),
        "period_end_utc": _ns_to_utc(period_end),
        "n_obs": int(y_true.size),
        "fit_status": fit_status,
        **metrics,
    }


def _ns_to_utc(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _metric_direction(metric: str) -> Literal["min", "max"]:
    if metric in {"log_loss", "brier_score", "expected_calibration_error"}:
        return "min"
    if metric in {"accuracy", "balanced_accuracy", "roc_auc", "pr_auc"}:
        return "max"
    raise ModelEvaluationError(f"unsupported selection metric: {metric}")


def _select_model(
    comparison_rows: list[dict[str, object]],
    candidates: tuple[ModelCandidate, ...],
    metric: str,
) -> str:
    direction = _metric_direction(metric)
    scores: list[tuple[float, int, str]] = []
    for order, candidate in enumerate(candidates):
        candidate_rows = [
            row
            for row in comparison_rows
            if row["split"] == "validation"
            and row.get("requested_model", row["model"]) == candidate.name
        ]
        used_fallback = any(row["model"] != candidate.name for row in candidate_rows)
        values = (
            [
                cast(float, row[metric])
                for row in candidate_rows
                if math.isfinite(cast(float, row[metric]))
            ]
            if not used_fallback
            else []
        )
        score = float(np.mean(values)) if values else math.nan
        sortable = score if direction == "min" else -score
        if not math.isfinite(sortable):
            sortable = math.inf
        scores.append((sortable, order, candidate.name))
    return min(scores)[2]


def evaluate_model_ladder(
    frame: pl.DataFrame,
    plan: WalkForwardPlan,
    model_config: ModelConfig,
    *,
    seed: int,
    calibration_bins: int,
    target: str = "future_mid_up",
    features: tuple[str, ...] | None = None,
    calibration_fraction: float = 0.2,
) -> ModelLadderResult:
    """Evaluate every model OOT, select on validation, then open final test once."""

    if target not in frame.columns:
        raise ModelEvaluationError(f"target column not found: {target}")
    selected_features = features or model_feature_columns(frame)
    missing_features = sorted(set(selected_features).difference(frame.columns))
    if missing_features:
        raise ModelEvaluationError(f"feature columns not found: {missing_features}")
    forbidden = [
        name
        for name in selected_features
        if name.startswith("future_") or name.startswith("label_") or name == "right_censored"
    ]
    if forbidden:
        raise ModelEvaluationError(f"label/timing columns cannot be model features: {forbidden}")

    indexed = frame.with_row_index("_research_row_id")
    candidates = build_model_candidates(model_config)
    horizon: int | None = None
    if "label_horizon_events" in frame.columns:
        horizon_values = frame.get_column("label_horizon_events").drop_nulls().unique().to_list()
        if len(horizon_values) == 1:
            horizon = int(horizon_values[0])
    predictions: list[dict[str, object]] = []
    comparison: list[dict[str, object]] = []

    for fold in plan.folds:
        train = _rows(indexed, fold.train_indices).filter(pl.col(target).is_not_null())
        validation = _rows(indexed, fold.validation_indices).filter(pl.col(target).is_not_null())
        for candidate in candidates:
            raw, calibrated, fit_status, fit_cutoff, effective_candidate = _fit_candidate(
                candidate,
                train,
                validation,
                features=selected_features,
                target=target,
                seed=seed,
                calibration_fraction=calibration_fraction,
            )
            y_validation = validation.get_column(target).to_numpy().astype(np.int64)
            predictions.extend(
                _prediction_rows(
                    validation,
                    candidate=effective_candidate,
                    requested_candidate=candidate,
                    fold_id=fold.fold_id,
                    split="validation",
                    target=target,
                    raw_probability=raw,
                    probability=calibrated,
                    fit_cutoff_ts_ns=fit_cutoff,
                )
            )
            comparison.append(
                _metric_row(
                    candidate=effective_candidate,
                    requested_candidate=candidate,
                    fold_id=fold.fold_id,
                    split="validation",
                    y_true=y_validation,
                    probability=calibrated,
                    calibration_bins=calibration_bins,
                    fit_status=fit_status,
                    evaluated=validation,
                    horizon_events=horizon,
                )
            )

    selected_model = _select_model(comparison, candidates, model_config.selection_metric)

    final_train = _rows(indexed, plan.final_train_indices).filter(pl.col(target).is_not_null())
    final_test = _rows(indexed, plan.test_indices).filter(pl.col(target).is_not_null())
    for candidate in candidates:
        raw, calibrated, fit_status, fit_cutoff, effective_candidate = _fit_candidate(
            candidate,
            final_train,
            final_test,
            features=selected_features,
            target=target,
            seed=seed,
            calibration_fraction=calibration_fraction,
        )
        y_test = final_test.get_column(target).to_numpy().astype(np.int64)
        predictions.extend(
            _prediction_rows(
                final_test,
                candidate=effective_candidate,
                requested_candidate=candidate,
                fold_id=-1,
                split="test",
                target=target,
                raw_probability=raw,
                probability=calibrated,
                fit_cutoff_ts_ns=fit_cutoff,
            )
        )
        comparison.append(
            _metric_row(
                candidate=effective_candidate,
                requested_candidate=candidate,
                fold_id=-1,
                split="test",
                y_true=y_test,
                probability=calibrated,
                calibration_bins=calibration_bins,
                fit_status=fit_status,
                evaluated=final_test,
                horizon_events=horizon,
            )
        )

    comparison_frame = pl.DataFrame(comparison).with_columns(
        (pl.col("model") == selected_model).alias("selected_on_validation"),
        pl.when(pl.col("model") == selected_model)
        .then(pl.lit("validation"))
        .otherwise(None)
        .alias("selected_on"),
    )
    prediction_frame = (
        pl.DataFrame(predictions)
        .sort(["split", "fold_id", "model", "decision_ts_ns", "symbol"])
        .with_columns(pl.lit(horizon, dtype=pl.Int64).alias("horizon_events"))
    )
    return ModelLadderResult(
        predictions=prediction_frame,
        comparison=comparison_frame,
        selected_model=selected_model,
        feature_columns=selected_features,
        selection_metric=model_config.selection_metric,
    )


def _metric_from_arrays(
    y_true: NDArray[np.int64],
    probability: NDArray[np.float64],
    metric: str,
) -> float:
    metrics = classification_metrics(y_true, probability, calibration_bins=10)
    if metric not in metrics:
        raise ModelEvaluationError(f"unsupported bootstrap metric: {metric}")
    return metrics[metric]


def _bootstrap_arrays(
    predictions: pl.DataFrame,
    *,
    block_column: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.object_], list[object]]:
    required = {"y_true", "probability", block_column}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ModelEvaluationError(f"bootstrap predictions missing columns: {missing}")
    y_true = predictions.get_column("y_true").to_numpy().astype(np.int64)
    probability = predictions.get_column("probability").to_numpy().astype(np.float64)
    blocks = predictions.get_column(block_column).to_numpy().astype(object)
    unique_blocks = list(dict.fromkeys(blocks.tolist()))
    return y_true, probability, blocks, unique_blocks


def block_bootstrap_metric(
    predictions: pl.DataFrame,
    *,
    metric: str,
    block_column: str,
    n_bootstrap: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap complete dependency blocks rather than overlapping events."""

    if n_bootstrap < 1:
        raise ModelEvaluationError("n_bootstrap must be positive")
    y_true, probability, blocks, unique_blocks = _bootstrap_arrays(
        predictions, block_column=block_column
    )
    point = _metric_from_arrays(y_true, probability, metric)
    if len(unique_blocks) < 2:
        return BootstrapResult(
            point_estimate=point,
            lower=None,
            upper=None,
            n_bootstrap=n_bootstrap,
            n_blocks=len(unique_blocks),
            seed=seed,
            status="insufficient_blocks",
            draws=(),
        )

    indices_by_block = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    random = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled_positions = random.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        sampled_blocks = [unique_blocks[int(position)] for position in sampled_positions]
        sampled_indices = np.concatenate([indices_by_block[block] for block in sampled_blocks])
        draws.append(
            _metric_from_arrays(y_true[sampled_indices], probability[sampled_indices], metric)
        )
    finite = np.asarray([draw for draw in draws if math.isfinite(draw)], dtype=np.float64)
    lower = float(np.quantile(finite, 0.025)) if finite.size else None
    upper = float(np.quantile(finite, 0.975)) if finite.size else None
    return BootstrapResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        n_bootstrap=n_bootstrap,
        n_blocks=len(unique_blocks),
        seed=seed,
        status="ok",
        draws=tuple(draws),
    )


def paired_block_bootstrap_difference(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    metric: str,
    block_column: str,
    n_bootstrap: int,
    seed: int,
) -> BootstrapResult:
    """Return a paired left-minus-right metric interval using common blocks."""

    required = {"row_id", "y_true", "probability", block_column}
    for name, frame in (("left", left), ("right", right)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ModelEvaluationError(f"{name} predictions missing columns: {missing}")
    paired = left.select(
        "row_id",
        "y_true",
        block_column,
        pl.col("probability").alias("left_probability"),
    ).join(
        right.select(
            "row_id",
            pl.col("y_true").alias("right_y_true"),
            pl.col("probability").alias("right_probability"),
        ),
        on="row_id",
        how="inner",
        validate="1:1",
    )
    if paired.height != left.height or paired.height != right.height:
        raise ModelEvaluationError("paired predictions must contain identical unique row IDs")
    if paired.filter(pl.col("y_true") != pl.col("right_y_true")).height:
        raise ModelEvaluationError("paired predictions disagree on target values")

    y_true = paired.get_column("y_true").to_numpy().astype(np.int64)
    left_probability = paired.get_column("left_probability").to_numpy().astype(np.float64)
    right_probability = paired.get_column("right_probability").to_numpy().astype(np.float64)
    blocks = paired.get_column(block_column).to_numpy().astype(object)
    unique_blocks = list(dict.fromkeys(blocks.tolist()))
    point = _metric_from_arrays(y_true, left_probability, metric) - _metric_from_arrays(
        y_true, right_probability, metric
    )
    if len(unique_blocks) < 2:
        return BootstrapResult(
            point, None, None, n_bootstrap, len(unique_blocks), seed, "insufficient_blocks", ()
        )
    indices_by_block = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    random = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        sampled_positions = random.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        sampled_blocks = [unique_blocks[int(position)] for position in sampled_positions]
        sampled_indices = np.concatenate([indices_by_block[block] for block in sampled_blocks])
        draws.append(
            _metric_from_arrays(y_true[sampled_indices], left_probability[sampled_indices], metric)
            - _metric_from_arrays(
                y_true[sampled_indices], right_probability[sampled_indices], metric
            )
        )
    finite = np.asarray([draw for draw in draws if math.isfinite(draw)], dtype=np.float64)
    return BootstrapResult(
        point_estimate=point,
        lower=float(np.quantile(finite, 0.025)) if finite.size else None,
        upper=float(np.quantile(finite, 0.975)) if finite.size else None,
        n_bootstrap=n_bootstrap,
        n_blocks=len(unique_blocks),
        seed=seed,
        status="ok",
        draws=tuple(draws),
    )


__all__ = [
    "BootstrapResult",
    "ModelCandidate",
    "ModelEvaluationError",
    "ModelLadderResult",
    "SigmoidCalibrator",
    "block_bootstrap_metric",
    "build_model_candidates",
    "classification_metrics",
    "evaluate_model_ladder",
    "expected_calibration_error",
    "make_classifier",
    "paired_block_bootstrap_difference",
]
