"""Reproducible descriptive economics for the frozen live-L2 study.

These functions run only after endpoint labels are opened.  They never select a
model or refit a threshold used by prediction.  Shock thresholds and stability
bins are fitted on the declared development reference, then applied unchanged
to held-out sessions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from microstructure.research.analysis import feature_stability_summary
from microstructure.research.l2_multidate import L2ResearchError, validate_l2_endpoint_frame

_NANOSECONDS_PER_MINUTE = 60_000_000_000
_DEVELOPMENT_ROLES = frozenset({"train", "validation"})
_TEST_ROLES = ("primary_test", "replication_test")


@dataclass(frozen=True, slots=True)
class L2DescriptiveAnalysis:
    """Machine-readable L2 economic analyses, kept separate from model scores."""

    intraday_liquidity: pl.DataFrame
    ofi_return_association: pl.DataFrame
    signal_half_life: pl.DataFrame
    liquidity_recovery: pl.DataFrame
    regime_diagnostics: pl.DataFrame
    feature_stability: pl.DataFrame
    cross_instrument_stability: pl.DataFrame


def _combined(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    values = tuple(frames)
    if not values:
        raise L2ResearchError("L2 descriptive analysis requires endpoint frames")
    for frame in values:
        validate_l2_endpoint_frame(frame)
    combined = pl.concat(values, how="diagonal_relaxed")
    keys = ["study_date", "symbol", "endpoint_name", "sample_id"]
    if combined.select(keys).unique().height != combined.height:
        raise L2ResearchError("L2 descriptive endpoint identities are not unique")
    return combined


def _canonical_books(combined: pl.DataFrame) -> pl.DataFrame:
    endpoint_names = sorted(str(value) for value in combined.get_column("endpoint_name").unique())
    chosen = "event_20" if "event_20" in endpoint_names else endpoint_names[0]
    books = combined.filter(pl.col("endpoint_name") == chosen)
    keys = ["study_date", "symbol", "continuity_id", "decision_sequence"]
    if books.select(keys).unique().height != books.height:
        raise L2ResearchError("canonical L2 book observations are not unique")
    return books


def _intraday_liquidity(books: pl.DataFrame) -> pl.DataFrame:
    return (
        books.with_columns(
            ((pl.col("decision_ts_ns") // _NANOSECONDS_PER_MINUTE) % (24 * 60))
            .cast(pl.Int32)
            .alias("utc_minute_of_day")
        )
        .group_by("study_date", "study_role", "symbol", "utc_minute_of_day")
        .agg(
            pl.len().alias("n_observations"),
            pl.col("spread_bps").mean().alias("mean_spread_bps"),
            pl.col("spread_bps").median().alias("median_spread_bps"),
            pl.col("depth_total_l1").mean().alias("mean_depth_l1"),
            pl.col("depth_total_l5").mean().alias("mean_depth_l5"),
            pl.col("depth_total_l10").mean().alias("mean_depth_l10"),
            pl.col("queue_imbalance_l1").mean().alias("mean_queue_imbalance_l1"),
            pl.col("realized_volatility_w100").mean().alias("mean_realized_volatility_w100"),
        )
        .sort("study_date", "symbol", "utc_minute_of_day")
    )


def _finite_correlation(x: pl.Series, y: pl.Series) -> float | None:
    pairs = (
        pl.DataFrame({"x": x, "y": y})
        .drop_nulls()
        .filter(pl.col("x").is_finite() & pl.col("y").is_finite())
    )
    if (
        pairs.height < 3
        or pairs.get_column("x").n_unique() < 2
        or pairs.get_column("y").n_unique() < 2
    ):
        return None
    value = pairs.select(pl.corr("x", "y")).item()
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _ofi_association(combined: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for key, frame in combined.filter(~pl.col("right_censored")).group_by(
        "study_date", "study_role", "symbol", "endpoint_name", maintain_order=True
    ):
        study_date, study_role, symbol, endpoint_name = (str(value) for value in key)
        side_source = str(frame.get_column("signed_markout_side_source")[0])
        rows.append(
            {
                "study_date": study_date,
                "study_role": study_role,
                "symbol": symbol,
                "endpoint_name": endpoint_name,
                "side_source": side_source,
                "n_observations": frame.height,
                "ofi_return_correlation": _finite_correlation(
                    frame.get_column(side_source), frame.get_column("future_mid_return")
                ),
                "mean_ofi_signed_future_mid_markout_bps": frame.get_column(
                    "ofi_signed_future_mid_markout_bps"
                ).mean(),
                "positive_direction_rate": frame.get_column("future_mid_up").mean(),
                "interpretation": "descriptive_book_flow_markout_not_trade_impact",
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "study_date", "symbol", "endpoint_name"
    )


def _signal_half_life(association: pl.DataFrame, combined: pl.DataFrame) -> pl.DataFrame:
    endpoint_contract = (
        combined.select(
            "endpoint_name", "endpoint_domain", "endpoint_horizon_value", "endpoint_horizon_unit"
        )
        .unique()
        .sort("endpoint_domain", "endpoint_horizon_value")
    )
    enriched = association.join(endpoint_contract, on="endpoint_name", how="left")
    rows: list[dict[str, object]] = []
    for key, frame in enriched.group_by(
        "study_date", "study_role", "symbol", "endpoint_domain", maintain_order=True
    ):
        study_date, study_role, symbol, domain = (str(value) for value in key)
        ordered = frame.sort("endpoint_horizon_value")
        correlations = ordered.get_column("ofi_return_correlation").to_list()
        baseline = (
            abs(float(correlations[0])) if correlations and correlations[0] is not None else None
        )
        crossed: int | None = None
        if baseline is not None and baseline > 0:
            for horizon, correlation in zip(
                ordered.get_column("endpoint_horizon_value").to_list(),
                correlations,
                strict=True,
            ):
                if correlation is not None and abs(float(correlation)) <= 0.5 * baseline:
                    crossed = int(horizon)
                    break
        for row in ordered.to_dicts():
            correlation = row["ofi_return_correlation"]
            association_ratio = None
            if correlation is not None and baseline is not None and baseline != 0.0:
                association_ratio = abs(float(correlation)) / baseline
            rows.append(
                {
                    "study_date": study_date,
                    "study_role": study_role,
                    "symbol": symbol,
                    "endpoint_domain": domain,
                    "endpoint_name": row["endpoint_name"],
                    "horizon_value": row["endpoint_horizon_value"],
                    "horizon_unit": row["endpoint_horizon_unit"],
                    "ofi_return_correlation": correlation,
                    "absolute_association_ratio_to_shortest": association_ratio,
                    "half_life_crossed_at_horizon": crossed,
                    "half_life_status": (
                        "crossed_within_declared_horizons"
                        if crossed is not None
                        else "not_crossed_or_unidentified"
                    ),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "study_date", "symbol", "endpoint_domain", "horizon_value"
    )


def _training_shock_thresholds(books: pl.DataFrame) -> pl.DataFrame:
    train = books.filter(pl.col("study_role") == "train").with_columns(
        pl.min_horizontal("bid_quantity", "ask_quantity").alias("executable_l1_depth")
    )
    symbols = set(str(value) for value in books.get_column("symbol").unique())
    if set(str(value) for value in train.get_column("symbol").unique()) != symbols:
        raise L2ResearchError("liquidity shock thresholds require train rows for every symbol")
    return train.group_by("symbol").agg(
        pl.col("spread_bps").quantile(0.95, interpolation="linear").alias("train_spread_q95"),
        pl.col("executable_l1_depth")
        .quantile(0.05, interpolation="linear")
        .alias("train_executable_depth_q05"),
    )


def _liquidity_recovery(books: pl.DataFrame) -> pl.DataFrame:
    thresholds = _training_shock_thresholds(books)
    joined = books.with_columns(
        pl.min_horizontal("bid_quantity", "ask_quantity").alias("executable_l1_depth")
    ).join(thresholds, on="symbol", how="left")
    group = ["study_date", "symbol", "continuity_id"]
    rows: list[pl.DataFrame] = []
    for horizon in (20, 100):
        rows.append(
            joined.with_columns(
                pl.col("spread_bps").shift(-horizon).over(group).alias("future_spread_bps"),
                pl.col("executable_l1_depth")
                .shift(-horizon)
                .over(group)
                .alias("future_executable_l1_depth"),
            )
            .filter(
                (pl.col("spread_bps") >= pl.col("train_spread_q95"))
                | (pl.col("executable_l1_depth") <= pl.col("train_executable_depth_q05"))
            )
            .drop_nulls(["future_spread_bps", "future_executable_l1_depth"])
            .with_columns(
                pl.lit(horizon).alias("recovery_horizon_events"),
                (pl.col("future_spread_bps") - pl.col("spread_bps")).alias("spread_change_bps"),
                (pl.col("future_executable_l1_depth") - pl.col("executable_l1_depth")).alias(
                    "executable_depth_change"
                ),
            )
            .group_by("study_date", "study_role", "symbol", "recovery_horizon_events")
            .agg(
                pl.len().alias("n_shocks"),
                pl.col("spread_change_bps").mean().alias("mean_spread_change_bps"),
                pl.col("executable_depth_change").mean().alias("mean_executable_depth_change"),
                pl.col("train_spread_q95").first(),
                pl.col("train_executable_depth_q05").first(),
            )
        )
    return pl.concat(rows, how="vertical_relaxed").sort(
        "study_date", "symbol", "recovery_horizon_events"
    )


def _regime_diagnostics(combined: pl.DataFrame) -> pl.DataFrame:
    required = {"volatility_regime", "liquidity_regime"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise L2ResearchError(f"regime diagnostics are missing columns: {missing}")
    return (
        combined.filter(~pl.col("right_censored"))
        .group_by(
            "study_date",
            "study_role",
            "symbol",
            "endpoint_name",
            "volatility_regime",
            "liquidity_regime",
        )
        .agg(
            pl.len().alias("n_observations"),
            pl.col("future_mid_up").mean().alias("positive_direction_rate"),
            pl.col("future_mid_return").mean().alias("mean_future_mid_return"),
            pl.col("ofi_signed_future_mid_markout_bps")
            .mean()
            .alias("mean_ofi_signed_future_mid_markout_bps"),
        )
        .sort("study_date", "symbol", "endpoint_name", "volatility_regime", "liquidity_regime")
    )


def _feature_stability(
    combined: pl.DataFrame, *, feature_columns: tuple[str, ...], bins: int
) -> pl.DataFrame:
    reference = combined.filter(pl.col("study_role").is_in(list(_DEVELOPMENT_ROLES)))
    if reference.is_empty():
        raise L2ResearchError("feature stability requires development rows")
    outputs: list[pl.DataFrame] = []
    for role in _TEST_ROLES:
        comparison = combined.filter(pl.col("study_role") == role)
        if comparison.is_empty():
            raise L2ResearchError(f"feature stability requires {role} rows")
        outputs.append(
            feature_stability_summary(
                reference,
                comparison,
                feature_columns=feature_columns,
                group_columns=("symbol", "endpoint_name"),
                bins=bins,
            ).with_columns(
                pl.lit(role).alias("comparison_role"),
                pl.lit("train_plus_validation_only").alias("reference_scope"),
            )
        )
    return pl.concat(outputs, how="vertical_relaxed").sort(
        "comparison_role", "symbol", "endpoint_name", "feature"
    )


def _cross_instrument_stability(association: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for key, frame in association.group_by(
        "study_date", "study_role", "endpoint_name", maintain_order=True
    ):
        study_date, study_role, endpoint_name = (str(value) for value in key)
        by_symbol: dict[str, float | None] = {
            str(row["symbol"]): (
                None
                if row["ofi_return_correlation"] is None
                else float(row["ofi_return_correlation"])
            )
            for row in frame.to_dicts()
        }
        btc = by_symbol.get("BTCUSDT")
        eth = by_symbol.get("ETHUSDT")
        same_direction = None
        if btc is not None and eth is not None and btc != 0.0 and eth != 0.0:
            same_direction = math.copysign(1.0, btc) == math.copysign(1.0, eth)
        rows.append(
            {
                "study_date": study_date,
                "study_role": study_role,
                "endpoint_name": endpoint_name,
                "btc_ofi_return_correlation": btc,
                "eth_ofi_return_correlation": eth,
                "both_observed": btc is not None and eth is not None,
                "same_direction": same_direction,
                "cross_instrument_pooling": False,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("study_date", "endpoint_name")


def build_l2_descriptive_analysis(
    endpoint_frames: Sequence[pl.DataFrame],
    *,
    feature_columns: Sequence[str],
    stability_bins: int,
) -> L2DescriptiveAnalysis:
    """Build all frozen descriptive outputs without fitting a predictive model."""

    features = tuple(str(value) for value in feature_columns)
    if not features or stability_bins < 2:
        raise L2ResearchError("descriptive feature columns and stability bins are required")
    combined = _combined(endpoint_frames)
    books = _canonical_books(combined)
    association = _ofi_association(combined)
    return L2DescriptiveAnalysis(
        intraday_liquidity=_intraday_liquidity(books),
        ofi_return_association=association,
        signal_half_life=_signal_half_life(association, combined),
        liquidity_recovery=_liquidity_recovery(books),
        regime_diagnostics=_regime_diagnostics(combined),
        feature_stability=_feature_stability(
            combined, feature_columns=features, bins=stability_bins
        ),
        cross_instrument_stability=_cross_instrument_stability(association),
    )


__all__ = ["L2DescriptiveAnalysis", "build_l2_descriptive_analysis"]
