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
    return (
        "This bundle is labeled as a full-data research run. Results remain simulated, "
        "exchange-specific, and conditional on the recorded execution assumptions. "
        f"The bundle contains {row_count} held-out comparison row(s)."
    )


def render_technical_report(bundle: RunBundle) -> str:
    """Render a deterministic technical report without calculating new statistics."""
    model_table = render_model_comparison(bundle).rstrip()
    quality = _mapping_table(bundle.quality)
    execution_assumptions = bundle.manifest.get("execution_assumptions", {})
    if not isinstance(execution_assumptions, Mapping):
        execution_assumptions = {"serialized_value": execution_assumptions}
    execution_text = _mapping_table(execution_assumptions)
    return f"""# Technical report — {bundle.run_id}

> **{bundle.watermark}**

This document was rendered only from a checksum-verified frozen run bundle. It
does not retrain a model, recompute a statistic, or infer a missing value.

## Run identity and provenance

{_provenance_table(bundle)}

## Executive summary

{_executive_summary(bundle)}

## Research question

When do order-flow imbalance, liquidity, and observable market state predict
short-horizon price movement, and how much apparent value survives the separately
specified execution model?

## Data lineage and quality

The report covers only `{bundle.observed_start_utc}` through
`{bundle.observed_end_utc}` in UTC for {", ".join(bundle.symbols)}. Data source and
mode are `{_display(bundle.data.get("source"))}` and
`{_display(bundle.data.get("mode"))}`. The frozen quality summary is:

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

{model_table}

Predictive metrics and execution results remain distinct serialized inputs. A
model metric does not establish that a fill was possible or that its signal
survives fees and latency.

## Execution assumptions

{execution_text}

Any fill probability, queue position, partial-fill behavior, fee, slippage,
latency, inventory, liquidation, or capacity result is conditional on these
assumptions and on the observability of the source data.

## Execution sensitivity

{_sensitivity_table(bundle)}

This scenario grid changes declared execution size assumptions only. It is not
an estimate of endogenous market impact or deployable capacity.

## Economic interpretation

No capital recommendation is made by this renderer. Synthetic outputs have no
empirical interpretation. Public-sample outputs are interval-specific. Full-data
outputs still require robustness across instruments, regimes, costs, latency,
fill specifications, and alternative periods before an investment conclusion.

## Limitations

- Exchange behavior and public data coverage may not generalize to other venues.
- Exchange event time does not prove local receipt or decision-time availability.
- Trade-only data cannot identify true queue position or cancellation dynamics.
- Approximate fills and simulated P&L are not realized execution performance.
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
dirty flag `{_display(git.get("dirty"))}`. It contains
{comparison_count} held-out comparison row(s) and {sensitivity_count} execution-
sensitivity scenario row(s). Those rows are diagnostics, not a
claim that a signal is stable, executable, or profitable.

**Decision framing.** Predictive quality, execution assumptions, and simulated
strategy outcomes must be assessed separately. A higher classification score is
not sufficient: the apparent edge must remain after fees, spread, latency,
partial fills, adverse selection, liquidation, and inventory constraints. Any
missing metric is reported as unavailable rather than zero.

**Current conclusion.** The defensible decision is to preserve the run as
reproducible evidence and use it to choose the next falsification test. It is not
to extrapolate beyond the recorded venue, instruments, UTC period, or evidence
tier.

<div style="page-break-after: always;"></div>

## Page 2 — Risks, kill criteria, and next work

**Material risks.** Public exchange data may omit receipt-time information and
matching-engine state. Trade-only observations cannot reveal true queue priority.
Clock gaps, sequence gaps, regime concentration, cost assumptions, and repeated
model searches can all create optimistic results. Simulated fills do not prove
capacity or operational executability.

**Kill criteria.** Stop escalation if the apparent effect disappears on the
untouched test period; changes sign across instruments or regimes without an
economic explanation; fails checksum, timing, or leakage controls; depends on an
implausibly favorable fee, latency, or fill assumption; or cannot beat the
declared baseline after costs with uncertainty reported.

**Next evidence requested.** Reproduce the same frozen configuration, then test a
predeclared adjacent period and both default instruments. Publish data-quality
exceptions, fold boundaries, calibration, bootstrap intervals, gross-to-net
attribution, fill/latency sensitivity, turnover, inventory, and liquidation
effects. Record failed hypotheses as carefully as favorable ones.

**Governance.** Any later memo must retain the evidence banner, actual UTC data
period, configuration and input hashes, and Git state. A changed assumption is a
new run, not a revision of this one. Live trading remains outside project scope.
"""


@dataclass(frozen=True, slots=True)
class ReportPaths:
    technical_report: Path
    executive_memo: Path
    model_comparison: Path


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
    _atomic_write(comparison, render_model_comparison(bundle))
    return ReportPaths(
        technical_report=technical,
        executive_memo=memo,
        model_comparison=comparison,
    )
