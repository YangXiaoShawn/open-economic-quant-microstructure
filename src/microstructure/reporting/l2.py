"""Deterministic human-readable reports for the prospective live-L2 bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


class L2ReportError(ValueError):
    """Raised when machine artifacts cannot support an honest L2 report."""


@dataclass(frozen=True, slots=True)
class L2ReportData:
    """Verified machine artifacts consumed by all three report surfaces."""

    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    session_gates: tuple[Mapping[str, Any], ...]
    hypothesis: Mapping[str, Any]
    predictive_metrics: tuple[Mapping[str, Any], ...]
    paired_metrics: tuple[Mapping[str, Any], ...]
    equal_session_metrics: tuple[Mapping[str, Any], ...]
    execution_metrics: tuple[Mapping[str, Any], ...]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise L2ReportError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise L2ReportError(f"{label} must be nonempty text")
    return value


def _number(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        observed = float(cast(Any, value))
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(observed):
        return "N/A"
    return f"{observed:.6f}"


def _ratio(value: object, label: str) -> str:
    if value is None:
        return "N/A"
    try:
        observed = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise L2ReportError(f"{label} must be a finite ratio") from error
    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
        raise L2ReportError(f"{label} must lie in [0, 1]")
    return f"{observed:.6f}"


def _bool(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "N/A"


def _session_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Date | Role | Status | BTC gate | ETH gate | Overlap seconds |\n"
        "| --- | --- | --- | --- | --- | ---: |"
    )
    body = [
        "| {date} | {role} | {status} | {btc} | {eth} | {overlap} |".format(
            date=row.get("study_date", "N/A"),
            role=row.get("study_role", "N/A"),
            status=row.get("status", "N/A"),
            btc=row.get("BTCUSDT_gate", row.get("btc_gate", "N/A")),
            eth=row.get("ETHUSDT_gate", row.get("eth_gate", "N/A")),
            overlap=_number(row.get("overlap_seconds")),
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def _predictive_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Symbol | Endpoint | Session | Model | N | Log loss | Prior | Delta | Brier | ECE |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [
        "| {symbol} | {endpoint} | {date} | {model} | {n} | {loss} | {prior} | {delta} | {brier} | {ece} |".format(
            symbol=row.get("symbol", "N/A"),
            endpoint=row.get("endpoint_name", "N/A"),
            date=row.get("study_date", row.get("study_role", "N/A")),
            model=row.get("selected_model", row.get("model", "N/A")),
            n=row.get("n_obs", "N/A"),
            loss=_number(row.get("selected_log_loss", row.get("log_loss"))),
            prior=_number(row.get("prior_log_loss")),
            delta=_number(row.get("point_delta", row.get("delta_log_loss"))),
            brier=_number(row.get("selected_brier_score", row.get("brier_score"))),
            ece=_number(
                row.get(
                    "selected_expected_calibration_error", row.get("expected_calibration_error")
                )
            ),
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def _paired_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Symbol | Endpoint | Session/regime | N | Blocks | Δ log loss | 95% low | 95% high | Status |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )

    def scope(row: Mapping[str, Any]) -> str:
        session = row.get("study_date", "equal-session")
        regime = row.get("regime", "N/A")
        return f"{session} / {regime}"

    body = [
        "| {symbol} | {endpoint} | {scope} | {n} | {blocks} | {delta} | {low} | {high} | {status} |".format(
            symbol=row.get("symbol", "N/A"),
            endpoint=row.get("endpoint_name", "N/A"),
            scope=scope(row),
            n=row.get("n_obs", "N/A"),
            blocks=row.get("n_blocks", "N/A"),
            delta=_number(row.get("point_delta")),
            low=_number(row.get("ci_low", row.get("lower"))),
            high=_number(row.get("ci_high", row.get("upper"))),
            status=row.get("status", "N/A"),
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def _execution_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Symbol | Endpoint | Session | Decision/order latency | Orders | Fill ratio | Turnover | Marked net P&L | Residual inventory |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [
        "| {symbol} | {endpoint} | {date} | {decision}/{order} events | {orders} | {fill} | {turnover} | {pnl} | {residual} |".format(
            symbol=row.get("symbol", "N/A"),
            endpoint=row.get("endpoint_name", "N/A"),
            date=row.get("study_date", "N/A"),
            decision=row.get("decision_latency_events", "N/A"),
            order=row.get("order_latency_events", "N/A"),
            orders=row.get("strategy_orders", "N/A"),
            fill=_ratio(row.get("fill_ratio"), "execution fill ratio"),
            turnover=_number(row.get("turnover_notional")),
            pnl=_number(row.get("marked_net_pnl", row.get("net_pnl"))),
            residual=_number(row.get("unliquidated_quantity")),
        )
        for row in rows
    ]
    return "\n".join([header, *body])


def _authority(
    data: L2ReportData,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    research = _mapping(data.manifest.get("research"), "run manifest research")
    git = _mapping(data.provenance.get("git"), "provenance Git")
    inputs = _mapping(data.provenance.get("inputs"), "provenance inputs")
    if data.manifest.get("evidence_tier") != "FULL_DATA":
        raise L2ReportError("live-L2 reports require FULL_DATA session scope")
    status = data.manifest.get("status")
    effective_tier = data.manifest.get("effective_evidence_tier")
    expected_tier = "FULL_DATA" if status == "COMPLETE" else "INSUFFICIENT_DATA"
    if status not in {"COMPLETE", "INSUFFICIENT_DATA"} or effective_tier != expected_tier:
        raise L2ReportError("live-L2 report status and effective evidence tier disagree")
    if data.manifest.get("live_trading") is not False:
        raise L2ReportError("live-L2 research reports must state live_trading=false")
    return research, git, inputs


def _evidence_banner(data: L2ReportData) -> str:
    if data.manifest.get("status") == "COMPLETE":
        return (
            "FULL-DATA PUBLIC L2 RESEARCH — RESEARCH/SIMULATION ONLY; "
            "NOT LIVE TRADING OR REALIZED PERFORMANCE"
        )
    return (
        "INSUFFICIENT_DATA — NO HELD-OUT, EXECUTION, ECONOMIC, "
        "SIGNIFICANCE, OR PROFITABILITY CONCLUSION"
    )


def render_l2_technical_report(data: L2ReportData) -> str:
    research, git, inputs = _authority(data)
    conclusion = _text(data.hypothesis.get("conclusion"), "hypothesis conclusion")
    return f"""# M8 prospective live-L2 technical report

> {_evidence_banner(data)}

## Research question

{_text(research.get("question"), "research question")}

## Immutable authority

- Capture period: `{research.get("period_start_utc")}` through `{research.get("period_end_utc")}`
- Capture config SHA-256: `{inputs.get("capture_config_sha256")}`
- Capture protocol SHA-256: `{inputs.get("capture_protocol_sha256")}`
- Analysis contract SHA-256: `{inputs.get("analysis_config_sha256")}`
- Development aggregate lock SHA-256: `{inputs.get("development_lock_sha256")}`
- Git commit: `{git.get("commit")}`; source-tree SHA-256: `{git.get("source_tree_sha256")}`; dirty: `{git.get("dirty")}`
- Test update policy: fit once on Aug 8-9, then no refit or recalibration on Aug 10-11.

## Session data-quality gates

{_session_table(data.session_gates)}

## Held-out predictive quality

{_predictive_table(data.predictive_metrics)}

## Paired dependence-aware diagnostics

{_paired_table(data.paired_metrics)}

Equal-session summaries are reported separately and never pooled across symbols. P-values, H0 rejection, statistical-significance claims, and persistent-alpha claims are not authorized.

## Market-order scenarios

{_execution_table(data.execution_metrics)}

These are exogenous historical replays at recorded L1 quotes with frozen fees, event latency, displayed-depth caps, inventory limits, and end liquidation. They are not realized execution; no capacity or profitability claim is authorized.

## Outcome

{conclusion}

## Limitations

- Public Binance depth data are exchange-specific and contain no authenticated account or order-entry path.
- Book-only data do not identify true queue priority, hidden liquidity, trade aggressor depletion, endogenous impact, or limit-fill probability.
- OFI-signed future-mid markout is a descriptive book-flow measure, not observed trade impact or a causal effect.
- Four fixed one-hour sessions cannot establish persistence outside the declared dates, instruments, or market regimes.
- All confidence intervals are seeded descriptive block-bootstrap diagnostics; multiple-testing and generalizability remain material limitations.
"""


def render_l2_executive_memo(data: L2ReportData) -> str:
    research, git, inputs = _authority(data)
    conclusion = _text(data.hypothesis.get("conclusion"), "hypothesis conclusion")
    replicated = [
        row
        for row in data.equal_session_metrics
        if row.get("regime") == "ALL" and row.get("directionally_replicated") is True
    ]
    declared_replicated = data.hypothesis.get("directionally_replicated_pairs")
    if declared_replicated != len(replicated):
        raise L2ReportError(
            "hypothesis replicated-pair count differs from overall equal-session metrics"
        )
    return f"""# Investment committee memo — prospective live-L2 study

> RESEARCH/SIMULATION ONLY — NO LIVE ORDERS, REALIZED EXECUTION, SIGNIFICANCE, CAPACITY, OR PROFITABILITY CLAIM

**Evidence tier.** {_evidence_banner(data)}.

**Decision.** Do not interpret this four-session study as deployment evidence. It is a predeclared test of whether book-state models improve direction log loss over a historical prior and whether that direction repeats on both untouched sessions.

**Evidence boundary.** The study covers `{research.get("period_start_utc")}` through `{research.get("period_end_utc")}` for BTCUSDT and ETHUSDT. The exact capture/analysis inputs are bound by `{inputs.get("capture_config_sha256")}` and `{inputs.get("analysis_config_sha256")}`; the development lock is `{inputs.get("development_lock_sha256")}`. Code identity is `{git.get("commit")}` with source tree `{git.get("source_tree_sha256")}`. Primary and replication predictions restore that lock without update or refit.

**Result.** {conclusion}

Directionally replicated symbol/endpoint pairs: **{len(replicated)}**. This count is descriptive and is not a multiple-testing-adjusted discovery claim.

**Economic interpretation.** Predictive scoring and the 3x3 market-order scenario grid are reported separately. Scenario P&L is a marked replay under recorded L1 depth, 4 bps taker fees, frozen event latency, partial fills, inventory limits, and end liquidation. It is not realized or deployable performance.

**Recommendation.** Preserve the result—including null, adverse, or insufficient outcomes—without date replacement. Any next study requires a new preregistered authority and broader independent dates.
"""


def render_l2_model_comparison(data: L2ReportData) -> str:
    research, git, inputs = _authority(data)
    return f"""# M8 live-L2 model comparison

> {_evidence_banner(data)}; NO CROSS-SYMBOL POOLING OR SIGNIFICANCE CLAIM

Period: `{research.get("period_start_utc")}` through `{research.get("period_end_utc")}`. Capture config: `{inputs.get("capture_config_sha256")}`. Analysis config: `{inputs.get("analysis_config_sha256")}`. Development lock: `{inputs.get("development_lock_sha256")}`. Git commit: `{git.get("commit")}`; source tree: `{git.get("source_tree_sha256")}`.

{_predictive_table(data.predictive_metrics)}

## Selected-minus-prior paired diagnostics

{_paired_table(data.equal_session_metrics)}

Every endpoint was selected on the validation session only. The primary and replication sessions use the same persisted numeric fitted state without update.
"""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_l2_report_set(output_dir: str | Path, data: L2ReportData) -> tuple[Path, Path, Path]:
    """Write all L2 reports from the same verified machine-artifact view."""

    root = Path(output_dir)
    technical = root / "technical_report.md"
    memo = root / "executive_memo.md"
    comparison = root / "model_comparison.md"
    _atomic_text(technical, render_l2_technical_report(data))
    _atomic_text(memo, render_l2_executive_memo(data))
    _atomic_text(comparison, render_l2_model_comparison(data))
    return technical, memo, comparison


def canonical_report_data_sha256(data: L2ReportData) -> str:
    """Bind the exact machine inputs used to render all report prose."""

    payload = {
        "manifest": dict(data.manifest),
        "provenance": dict(data.provenance),
        "session_gates": [dict(row) for row in data.session_gates],
        "hypothesis": dict(data.hypothesis),
        "predictive_metrics": [dict(row) for row in data.predictive_metrics],
        "paired_metrics": [dict(row) for row in data.paired_metrics],
        "equal_session_metrics": [dict(row) for row in data.equal_session_metrics],
        "execution_metrics": [dict(row) for row in data.execution_metrics],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "L2ReportData",
    "L2ReportError",
    "canonical_report_data_sha256",
    "render_l2_executive_memo",
    "render_l2_model_comparison",
    "render_l2_technical_report",
    "write_l2_report_set",
]
