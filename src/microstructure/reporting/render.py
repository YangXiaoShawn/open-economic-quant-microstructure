"""Pure Markdown rendering from a validated, frozen run bundle."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microstructure.reporting.bundle import RunBundle
from microstructure.reporting.tables import comparison_rows, render_model_comparison

_DEFAULT_RESEARCH_QUESTION = (
    "When do order-flow imbalance, liquidity, and observable market state predict\n"
    "short-horizon price movement, and how much apparent value survives the separately\n"
    "specified execution model?"
)


def _display(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mapping_table(values: Mapping[str, Any]) -> str:
    if not values:
        return "No serialized values were supplied."
    lines = ["| Field | Serialized value |", "| --- | --- |"]
    for key in sorted(values):
        lines.append(f"| {_display(key)} | {_display(values[key])} |")
    return "\n".join(lines)


def _provenance_table(bundle: RunBundle) -> str:
    git = bundle.provenance.get("git", {})
    if not isinstance(git, Mapping):
        git = {}
    input_hashes = bundle.provenance.get("input_manifest_sha256", [])
    values = {
        "Run ID": bundle.run_id,
        "Evidence tier": bundle.evidence_tier,
        "Symbols": ", ".join(bundle.symbols),
        "Observed start (UTC)": bundle.observed_start_utc,
        "Observed end (UTC)": bundle.observed_end_utc,
        "Configuration SHA-256": bundle.provenance.get("config_sha256"),
        "Input manifest SHA-256": input_hashes,
        "Git commit": git.get("commit"),
        "Git dirty at run time": git.get("dirty"),
        "Seed": bundle.provenance.get("seed"),
        "Runtime metadata": bundle.provenance.get("runtime"),
        "Generated at (UTC)": bundle.provenance.get("generated_at_utc"),
    }
    return _mapping_table(values)


def _sensitivity_table(bundle: RunBundle) -> str:
    if not bundle.execution_sensitivity:
        return "No execution-sensitivity rows were serialized."
    columns = (
        "order_type",
        "size_multiplier",
        "net_pnl",
        "net_edge_bps",
        "fill_ratio",
        "turnover_notional",
        "maximum_drawdown",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in bundle.execution_sensitivity:
        lines.append("| " + " | ".join(_display(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _research_question(bundle: RunBundle) -> str:
    research = bundle.manifest.get("research")
    if isinstance(research, Mapping):
        question = research.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return _DEFAULT_RESEARCH_QUESTION


def _hypothesis_number(value: object) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return _display(value)


def _paired_hypothesis_section(bundle: RunBundle) -> str:
    payload = bundle.hypothesis_evaluation
    if not payload:
        return ""
    if payload.get("evidence_scope") == "trade_only_complete_predeclared_daily_archives":
        return _m8_multidate_hypothesis_section(payload)
    values = payload.get("per_symbol")
    if not isinstance(values, list) or not values:
        return (
            "## Paired H0/H1 diagnostic\n\n"
            "The declared hypothesis artifact contains no per-symbol rows."
        )

    rows = [value for value in values if isinstance(value, Mapping)]
    metric = payload.get("selection_metric")
    if metric == "log_loss":
        direction_text = (
            "For Δ log-loss (selected minus historical prior), a negative value favors the "
            "selected model."
        )
    else:
        directions = sorted(
            {
                str(row.get("favorable_direction"))
                for row in rows
                if row.get("favorable_direction") is not None
            }
        )
        direction_text = (
            "The serialized favorable direction is "
            + (", ".join(f"`{value}`" for value in directions) if directions else "N/A")
            + "."
        )

    lines = [
        "## Paired H0/H1 diagnostic",
        "",
        (
            "The frozen comparison is validation-selected test predictions versus the "
            "historical-prior baseline on identical `row_id` and fixed 2x-horizon blocks. "
            + direction_text
        ),
        "",
        (
            "| Symbol | Selected model | Baseline | Metric | Point Δ | Paired 95% interval | "
            "Observations | Blocks | Samples | Seed | Status |"
        ),
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
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
        lines.append("| " + " | ".join(_display(cell) for cell in cells) + " |")

    caveat = payload.get("caveat")
    cross = payload.get("cross_instrument_conclusion")
    cross_text: object = None
    if isinstance(cross, Mapping):
        cross_text = cross.get("text")
    if not isinstance(cross_text, str) or not cross_text.strip():
        cross_text = (
            "No cross-instrument estimate was pooled. Mixed directions cannot support "
            "persistent alpha; even matching directions remain sample-specific."
        )
    lines.extend(
        [
            "",
            (
                "These paired percentile intervals are dependence diagnostics, not p-values "
                "or confirmatory significance intervals; H0 is not rejected by this "
                "exploratory design."
            ),
            _display(caveat) if isinstance(caveat, str) and caveat.strip() else "",
            "",
            (
                "No cross-instrument estimate was pooled. Mixed directions, if present, "
                "cannot support persistent alpha; even matching directions would remain "
                "bounded, sample-specific diagnostics."
            ),
            _display(cross_text),
        ]
    )
    return "\n".join(lines)


def _m8_multidate_hypothesis_section(payload: Mapping[str, Any]) -> str:
    """Render the frozen M8 component dates and equal-date endpoint verbatim."""

    raw_dates = payload.get("per_date")
    raw_symbols = payload.get("per_symbol")
    date_rows = (
        [row for row in raw_dates if isinstance(row, Mapping)]
        if isinstance(raw_dates, list)
        else []
    )
    symbol_rows = (
        [row for row in raw_symbols if isinstance(row, Mapping)]
        if isinstance(raw_symbols, list)
        else []
    )
    lines = [
        "## M8 predeclared multi-date endpoint",
        "",
        (
            "`FULL_DATA` here means only that every byte of all eight predeclared "
            "Binance daily aggregate-trade archives was verified and included. This is a "
            "trade-only study; it does not mean full market observability, order-book "
            "evidence, execution evidence, or deployable performance."
        ),
        "",
        "### Untouched-date components",
        "",
        (
            "| Symbol | UTC date | Frozen role | Locked model | Baseline | Selected log loss | "
            "Prior log loss | Point Δ | Paired 95% interval | N | Blocks | Status |"
        ),
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in sorted(
        date_rows,
        key=lambda value: (str(value.get("symbol", "")), str(value.get("study_date", ""))),
    ):
        interval = (
            f"[{_hypothesis_number(row.get('ci_low'))}, {_hypothesis_number(row.get('ci_high'))}]"
        )
        date_cells = (
            row.get("symbol"),
            row.get("study_date"),
            row.get("study_role"),
            row.get("selected_model"),
            row.get("baseline"),
            _hypothesis_number(row.get("selected_log_loss")),
            _hypothesis_number(row.get("prior_log_loss")),
            _hypothesis_number(row.get("point_delta")),
            interval,
            row.get("n_obs"),
            row.get("n_blocks"),
            row.get("bootstrap_status"),
        )
        lines.append("| " + " | ".join(_display(cell) for cell in date_cells) + " |")
    if not date_rows:
        lines.append(
            "| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | missing |"
        )

    lines.extend(
        [
            "",
            "### Equal-date-weighted endpoint",
            "",
            (
                "| Symbol | Locked model | Baseline | Point Δ | Paired 95% interval | "
                "Dates | N | Blocks | Status | Replication status |"
            ),
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in sorted(symbol_rows, key=lambda value: str(value.get("symbol", ""))):
        interval = (
            f"[{_hypothesis_number(row.get('ci_low'))}, {_hypothesis_number(row.get('ci_high'))}]"
        )
        aggregate_cells = (
            row.get("symbol"),
            row.get("selected_model"),
            row.get("baseline"),
            _hypothesis_number(row.get("point_delta")),
            interval,
            row.get("n_dates"),
            row.get("n_obs"),
            row.get("n_blocks"),
            row.get("status"),
            row.get("replication_status"),
        )
        lines.append("| " + " | ".join(_display(cell) for cell in aggregate_cells) + " |")
    if not symbol_rows:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | insufficient_data | N/A |")

    direction_rows = [row for row in symbol_rows if "validation_primary_replication_status" in row]
    if direction_rows:
        lines.extend(
            [
                "",
                "### Validation → primary → replication direction consistency",
                "",
                (
                    "| Symbol | Validation date | Validation Δ | Primary date | Primary Δ | "
                    "Replication date | Replication Δ | Same direction | All favorable | Status |"
                ),
                "| --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | --- |",
            ]
        )
        for row in sorted(direction_rows, key=lambda value: str(value.get("symbol", ""))):
            direction_cells = (
                row.get("symbol"),
                row.get("validation_date"),
                _hypothesis_number(row.get("validation_point_delta")),
                row.get("primary_date"),
                _hypothesis_number(row.get("primary_point_delta")),
                row.get("replication_date"),
                _hypothesis_number(row.get("replication_point_delta")),
                row.get("direction_consistent_across_validation_primary_replication"),
                row.get("favorable_across_validation_primary_replication"),
                row.get("validation_primary_replication_status"),
            )
            lines.append("| " + " | ".join(_display(cell) for cell in direction_cells) + " |")

    caveat = payload.get("caveat")
    cross = payload.get("cross_instrument_conclusion")
    cross_text = cross.get("text") if isinstance(cross, Mapping) else None
    lines.extend(
        [
            "",
            (
                "Negative Δ log loss favors the locked selected model. The two date rows "
                "are mandatory components of each equal-date estimate; neither date nor "
                "instrument is omitted because of its direction."
            ),
            (
                _display(caveat)
                if isinstance(caveat, str) and caveat.strip()
                else (
                    "Intervals are descriptive paired block-bootstrap diagnostics. No "
                    "p-value, H0 rejection, or significance claim is authorized."
                )
            ),
            (
                _display(cross_text)
                if isinstance(cross_text, str) and cross_text.strip()
                else "No cross-instrument estimate or persistent-alpha conclusion is inferred."
            ),
            "No execution, P&L, capacity, or profitability claim is authorized.",
        ]
    )
    return "\n".join(lines)


def _symbol_coverage_table(bundle: RunBundle) -> str:
    coverage = bundle.data.get("symbol_coverage")
    if not isinstance(coverage, list):
        return ""

    columns = (
        "Symbol",
        "Rows",
        "Observed start (UTC)",
        "Observed end (UTC, inclusive)",
        "Requested range complete",
    )
    lines = [
        "### Per-symbol observed coverage",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for item in coverage:
        if isinstance(item, Mapping):
            observed_end = item.get("observed_end_inclusive_utc", item.get("observed_end_utc"))
            complete = item.get("complete_range", item.get("complete"))
            values = (
                item.get("symbol"),
                item.get("rows"),
                item.get("observed_start_utc"),
                observed_end,
                complete,
            )
        else:
            values = (item, None, None, None, None)
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    if not coverage:
        lines.append("| N/A | N/A | N/A | N/A | N/A |")
    return "\n".join(lines)


def _date_coverage_table(bundle: RunBundle) -> str:
    coverage = bundle.data.get("date_coverage")
    if not isinstance(coverage, list):
        return ""
    lines = [
        "### Per-date observed coverage",
        "",
        (
            "| Symbol | UTC date | Frozen role | Rows | Observed start (UTC) | "
            "Observed end (UTC, inclusive) | Complete | DQ errors | DQ warnings |"
        ),
        "| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for item in coverage:
        if not isinstance(item, Mapping):
            continue
        cells = (
            item.get("symbol"),
            item.get("date"),
            item.get("role"),
            item.get("rows"),
            item.get("observed_start_utc"),
            item.get("observed_end_inclusive_utc"),
            item.get("complete"),
            item.get("quality_errors"),
            item.get("quality_warnings"),
        )
        lines.append("| " + " | ".join(_display(cell) for cell in cells) + " |")
    if len(lines) == 3:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    return "\n".join(lines)


def _execution_not_run(assumptions: Mapping[str, Any]) -> bool:
    return assumptions.get("status") == "NOT_RUN"


def _execution_not_run_reason(assumptions: Mapping[str, Any]) -> str:
    reason = assumptions.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return "No reason was serialized."


def _executive_summary(bundle: RunBundle) -> str:
    row_count = len(comparison_rows(bundle))
    if bundle.evidence_tier == "SYNTHETIC_SMOKE":
        return (
            "This run is an offline software smoke test. Its serialized values may be useful "
            "for checking data flow, metric accounting, and presentation, but they do not "
            "measure a market relationship, tradable edge, or statistical significance. "
            f"The bundle contains {row_count} held-out comparison row(s)."
        )
    if bundle.evidence_tier == "PUBLIC_SAMPLE_PARTIAL":
        return (
            "This is a bounded public-data sample, not a full empirical study. The table "
            "reports serialized held-out diagnostics without extrapolating beyond the stated "
            f"UTC interval. The bundle contains {row_count} comparison row(s)."
        )
    research = bundle.manifest.get("research")
    trade_only = isinstance(research, Mapping) and research.get("scope") == "trade_only"
    if trade_only:
        return (
            "This full-data label is narrowly scoped to every byte of the predeclared "
            "trade archives, not to full market observability. The run is exchange-specific "
            "and contains no order-book, execution, P&L, capacity, or profitability evidence. "
            f"The bundle contains {row_count} held-out comparison row(s)."
        )
    return (
        "This bundle is labeled as a full-data research run. Results remain simulated, "
        "exchange-specific, and conditional on the recorded execution assumptions. "
        f"The bundle contains {row_count} held-out comparison row(s)."
    )


def render_technical_report(bundle: RunBundle) -> str:
    """Render a deterministic technical report without calculating new statistics."""
    model_table = render_model_comparison(bundle).rstrip()
    hypothesis_section = _paired_hypothesis_section(bundle)
    hypothesis_text = f"\n\n{hypothesis_section}" if hypothesis_section else ""
    quality = _mapping_table(bundle.quality)
    coverage_table = _symbol_coverage_table(bundle)
    date_coverage_table = _date_coverage_table(bundle)
    coverage_sections = [value for value in (coverage_table, date_coverage_table) if value]
    coverage_text = "\n\n" + "\n\n".join(coverage_sections) + "\n\n" if coverage_sections else " "
    execution_assumptions = bundle.manifest.get("execution_assumptions", {})
    if not isinstance(execution_assumptions, Mapping):
        execution_assumptions = {"serialized_value": execution_assumptions}
    execution_text = _mapping_table(execution_assumptions)
    if _execution_not_run(execution_assumptions):
        exclusion_reason = _display(_execution_not_run_reason(execution_assumptions))
        model_execution_text = (
            "Only predictive diagnostics are serialized for this run. Execution simulation, "
            "fills, and P&L are absent by design, so predictive metrics cannot be interpreted "
            "as executable or profitable performance."
        )
        execution_scope_text = (
            "**Execution simulation and P&L were not run for this bundle.** "
            f"Reason: {exclusion_reason}\n\n"
            "No fill, fee, slippage, latency, queue, inventory, liquidation, capacity, "
            "turnover, or profitability result is available from this run."
        )
        sensitivity_text = (
            "Execution sensitivity was not run, and no scenario rows are available. "
            f"Reason: {exclusion_reason}"
        )
        economic_text = (
            "No capital recommendation is made by this renderer. This run contains no "
            "execution simulation or P&L, so its predictive diagnostics provide no evidence "
            "of fillability, net returns, deployable capacity, or profitability. Predictive "
            "outputs remain bounded by the declared evidence tier and observed interval."
        )
        execution_limitation = (
            "- Execution simulation and P&L were not run; no execution-performance or "
            "profitability inference is available."
        )
    else:
        model_execution_text = (
            "Predictive metrics and execution results remain distinct serialized inputs. A\n"
            "model metric does not establish that a fill was possible or that its signal\n"
            "survives fees and latency."
        )
        execution_scope_text = (
            "Any fill probability, queue position, partial-fill behavior, fee, slippage,\n"
            "latency, inventory, liquidation, or capacity result is conditional on these\n"
            "assumptions and on the observability of the source data."
        )
        sensitivity_text = (
            f"{_sensitivity_table(bundle)}\n\n"
            "This scenario grid changes declared execution size assumptions only. It is not\n"
            "an estimate of endogenous market impact or deployable capacity."
        )
        economic_text = (
            "No capital recommendation is made by this renderer. Synthetic outputs have no\n"
            "empirical interpretation. Public-sample outputs are interval-specific. Full-data\n"
            "outputs still require robustness across instruments, regimes, costs, latency,\n"
            "fill specifications, and alternative periods before an investment conclusion."
        )
        execution_limitation = (
            "- Approximate fills and simulated P&L are not realized execution performance."
        )
    return f"""# Technical report — {bundle.run_id}

> **{bundle.watermark}**

This document was rendered only from a checksum-verified frozen run bundle. It
does not retrain a model, recompute a statistic, or infer a missing value.

## Run identity and provenance

{_provenance_table(bundle)}

## Executive summary

{_executive_summary(bundle)}

## Research question

{_research_question(bundle)}

## Data lineage and quality

The report covers only `{bundle.observed_start_utc}` through
`{bundle.observed_end_utc}` in UTC for {", ".join(bundle.symbols)}. Data source and
mode are `{_display(bundle.data.get("source"))}` and
`{_display(bundle.data.get("mode"))}`.{coverage_text}The frozen quality summary is:

{quality}

Questionable observations are findings, not silently repaired inputs. Consult
the run's data manifests and transformation records before interpreting any row.

## Temporal design and evaluation

Feature observability, future-label boundaries, fold definitions, purging, and
embargo decisions belong to the serialized research artifacts. This reporting
layer does not reconstruct them. A model row is eligible for the comparison
table only when its serialized split is `test`, `final_test`, `holdout`, or
`held_out`; validation rows are not promoted as final evidence.

## Model comparison

{model_table}{hypothesis_text}

{model_execution_text}

## Execution assumptions

{execution_text}

{execution_scope_text}

## Execution sensitivity

{sensitivity_text}

## Economic interpretation

{economic_text}

## Limitations

- Exchange behavior and public data coverage may not generalize to other venues.
- Exchange event time does not prove local receipt or decision-time availability.
- Trade-only data cannot identify true queue position or cancellation dynamics.
{execution_limitation}
- Small or selected periods can exaggerate stability; multiple comparisons raise
  false-discovery risk.
- Checksums establish artifact integrity, not economic validity.

## Reproduction record

Use the exact resolved configuration and input-manifest hashes shown above. The
Git dirty flag describes the code state at run time; `UNBORN` means that no commit
was available and therefore weakens code-level reproducibility. Verify
`checksums.sha256` before using the bundle.
"""


def render_executive_memo(bundle: RunBundle) -> str:
    """Render a compact, explicitly two-page investment-committee-style memo."""
    evidence_note = _executive_summary(bundle)
    comparison_count = len(comparison_rows(bundle))
    sensitivity_count = len(bundle.execution_sensitivity)
    git = bundle.provenance.get("git", {})
    if not isinstance(git, Mapping):
        git = {}
    input_hashes = bundle.provenance.get("input_manifest_sha256", [])
    hypothesis_section = _paired_hypothesis_section(bundle)
    hypothesis_text = f"\n\n{hypothesis_section}" if hypothesis_section else ""
    execution_assumptions = bundle.manifest.get("execution_assumptions", {})
    if not isinstance(execution_assumptions, Mapping):
        execution_assumptions = {"serialized_value": execution_assumptions}
    if _execution_not_run(execution_assumptions):
        exclusion_reason = _display(_execution_not_run_reason(execution_assumptions))
        artifact_inventory = (
            f"It contains {comparison_count} held-out comparison row(s). Execution "
            "simulation, fills, execution sensitivity, and P&L were not run. "
            f"Reason: {exclusion_reason} No absent execution metric should be interpreted "
            "as zero."
        )
        decision_framing = (
            "**Decision framing.** This bundle can assess only the serialized predictive "
            "diagnostics. It cannot assess fillability, fees, spread, latency, adverse "
            "selection, inventory, capacity, net returns, or profitability because no "
            "execution model or P&L calculation was run."
        )
        material_risks = (
            "**Material risks.** Public exchange data may omit receipt-time information and "
            "matching-engine state. Trade-only observations cannot reveal spread, depth, "
            "cancellations, queue priority, or fillability. Clock gaps, sequence gaps, "
            "regime concentration, and repeated model searches can create optimistic "
            "predictive diagnostics."
        )
        kill_criteria = (
            "**Kill criteria.** Stop escalation if the apparent effect disappears on the "
            "untouched test period; changes sign across instruments or regimes without an "
            "economic explanation; fails checksum, timing, or leakage controls; or depends "
            "on repeated sample or model selection. Predictive diagnostics alone cannot "
            "clear an execution or capital gate."
        )
        next_evidence = (
            "**Next evidence requested.** Reproduce the same frozen configuration, then test "
            "a predeclared adjacent period and both default instruments. Publish data-quality "
            "exceptions, fold boundaries, calibration, and dependence-aware intervals. "
            "Acquire appropriate quote or order-book data before any separate execution, "
            "fill, cost, capacity, or P&L study. Record failed hypotheses as carefully as "
            "favorable ones."
        )
    else:
        artifact_inventory = (
            f"It contains\n{comparison_count} held-out comparison row(s) and "
            f"{sensitivity_count} execution-\nsensitivity scenario row(s). Those rows are "
            "diagnostics, not a\nclaim that a signal is stable, executable, or profitable."
        )
        decision_framing = (
            "**Decision framing.** Predictive quality, execution assumptions, and simulated\n"
            "strategy outcomes must be assessed separately. A higher classification score is\n"
            "not sufficient: the apparent edge must remain after fees, spread, latency,\n"
            "partial fills, adverse selection, liquidation, and inventory constraints. Any\n"
            "missing metric is reported as unavailable rather than zero."
        )
        material_risks = (
            "**Material risks.** Public exchange data may omit receipt-time information and\n"
            "matching-engine state. Trade-only observations cannot reveal true queue priority.\n"
            "Clock gaps, sequence gaps, regime concentration, cost assumptions, and repeated\n"
            "model searches can all create optimistic results. Simulated fills do not prove\n"
            "capacity or operational executability."
        )
        kill_criteria = (
            "**Kill criteria.** Stop escalation if the apparent effect disappears on the\n"
            "untouched test period; changes sign across instruments or regimes without an\n"
            "economic explanation; fails checksum, timing, or leakage controls; depends on an\n"
            "implausibly favorable fee, latency, or fill assumption; or cannot beat the\n"
            "declared baseline after costs with uncertainty reported."
        )
        next_evidence = (
            "**Next evidence requested.** Reproduce the same frozen configuration, then test a\n"
            "predeclared adjacent period and both default instruments. Publish data-quality\n"
            "exceptions, fold boundaries, calibration, bootstrap intervals, gross-to-net\n"
            "attribution, fill/latency sensitivity, turnover, inventory, and liquidation\n"
            "effects. Record failed hypotheses as carefully as favorable ones."
        )
    return f"""# Research review memo — {bundle.run_id}

> **{bundle.watermark}**

## Page 1 — Decision and evidence

**Recommendation:** Continue research only; authorize no capital deployment and
no live-order connection on the basis of this run.

**Evidence boundary.** {evidence_note}

The observed interval is `{bundle.observed_start_utc}` through
`{bundle.observed_end_utc}` for {", ".join(bundle.symbols)}. The run records
configuration `{bundle.provenance.get("config_sha256")}`, input-manifest hashes
`{_display(input_hashes)}`, and Git commit `{_display(git.get("commit"))}` with
dirty flag `{_display(git.get("dirty"))}`. {artifact_inventory}

{decision_framing}{hypothesis_text}

**Current conclusion.** The defensible decision is to preserve the run as
reproducible evidence and use it to choose the next falsification test. It is not
to extrapolate beyond the recorded venue, instruments, UTC period, or evidence
tier.

<div style="page-break-after: always;"></div>

## Page 2 — Risks, kill criteria, and next work

{material_risks}

{kill_criteria}

{next_evidence}

**Governance.** Any later memo must retain the evidence banner, actual UTC data
period, configuration and input hashes, and Git state. A changed assumption is a
new run, not a revision of this one. Live trading remains outside project scope.
"""


@dataclass(frozen=True, slots=True)
class ReportPaths:
    technical_report: Path
    executive_memo: Path
    model_comparison: Path


def render_model_comparison_report(bundle: RunBundle) -> str:
    """Render the comparison table together with its frozen hypothesis evidence."""

    content = render_model_comparison(bundle).rstrip()
    hypothesis_section = _paired_hypothesis_section(bundle)
    if hypothesis_section:
        content += "\n\n" + hypothesis_section
    return content + "\n"


def _atomic_write(path: Path, content: str) -> None:
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


def write_report_set(bundle: RunBundle, output_dir: str | Path) -> ReportPaths:
    """Atomically write a deterministic report set outside the frozen input bundle."""
    output = Path(output_dir).resolve()
    if output == bundle.root or output.is_relative_to(bundle.root):
        raise ValueError("report output cannot mutate the frozen input run bundle")
    technical = output / "technical_report.md"
    memo = output / "executive_memo.md"
    comparison = output / "model_comparison.md"
    _atomic_write(technical, render_technical_report(bundle))
    _atomic_write(memo, render_executive_memo(bundle))
    _atomic_write(comparison, render_model_comparison_report(bundle))
    return ReportPaths(
        technical_report=technical,
        executive_memo=memo,
        model_comparison=comparison,
    )
