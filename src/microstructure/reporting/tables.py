"""Deterministic, presentation-only model comparison tables."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from microstructure.reporting.bundle import RunBundle

_JOIN_KEYS = ("instrument", "model", "horizon_events", "split", "study_date")
_TEST_SPLITS = frozenset({"test", "final_test", "holdout", "held_out"})


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def _canonical_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["instrument"] = _first(row, ("instrument", "symbol"), "N/A")
    result["model"] = _first(row, ("model", "model_name"), "N/A")
    result["horizon_events"] = _first(
        row, ("horizon_events", "label_horizon_events", "horizon"), "N/A"
    )
    # Missing split provenance is not evidence that a row is held out.  Keep the
    # presentation layer fail-closed so an unsplit metric cannot be promoted to
    # final-test evidence merely by being serialized.
    result["split"] = str(_first(row, ("split", "evaluation_split"), "unknown"))
    # A multi-date held-out study can legitimately emit the same model/split
    # combination more than once.  Preserve the declared study date in the
    # presentation join so a replication period cannot overwrite the primary
    # test period.  Legacy single-period rows retain one explicit sentinel and
    # therefore keep their previous join behaviour.
    result["study_date"] = str(_first(row, ("study_date", "period_date", "date"), "N/A"))
    result["n_obs"] = _first(row, ("n_obs", "observations", "sample_size"))
    result["period_start_utc"] = _first(row, ("period_start_utc", "test_start_utc", "start_utc"))
    result["period_end_utc"] = _first(row, ("period_end_utc", "test_end_utc", "end_utc"))
    result["brier_score"] = _first(row, ("brier_score", "brier"))
    result["expected_calibration_error"] = _first(row, ("expected_calibration_error", "ece"))
    result["fees_bps"] = _first(row, ("fees_bps", "cost_bps", "costs_bps"))
    result["max_drawdown"] = _first(row, ("max_drawdown", "max_drawdown_units"))
    return result


def _join_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "N/A")) for key in _JOIN_KEYS)


def comparison_rows(bundle: RunBundle) -> tuple[Mapping[str, Any], ...]:
    """Join predictive and execution results without recomputing any metric."""
    joined: dict[tuple[str, ...], dict[str, Any]] = {}
    for source_row in bundle.predictive_metrics:
        row = _canonical_row(source_row)
        if row["split"].lower() not in _TEST_SPLITS:
            continue
        joined[_join_key(row)] = row
    for source_row in bundle.execution_metrics:
        row = _canonical_row(source_row)
        if row["split"].lower() not in _TEST_SPLITS:
            continue
        key = _join_key(row)
        if key in joined:
            joined[key].update({name: value for name, value in row.items() if value is not None})
        else:
            joined[key] = row
    return tuple(joined[key] for key in sorted(joined))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_plain(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_integer(value: Any) -> str:
    number = _number(value)
    return f"{int(number):,}" if number is not None and number.is_integer() else "N/A"


def _format_estimate(
    row: Mapping[str, Any], key: str, *, decimals: int, percent: bool = False
) -> str:
    estimate = _number(row.get(key))
    if estimate is None:
        return "N/A"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    rendered = f"{estimate * scale:.{decimals}f}{suffix}"
    lower = _number(row.get(f"{key}_ci_low"))
    upper = _number(row.get(f"{key}_ci_high"))
    if lower is not None and upper is not None:
        rendered += f" [{lower * scale:.{decimals}f}, {upper * scale:.{decimals}f}]{suffix}"
    return rendered


def _period(row: Mapping[str, Any]) -> str:
    start = _format_plain(row.get("period_start_utc"))
    end = _format_plain(row.get("period_end_utc"))
    if start == "N/A" and end == "N/A":
        return "N/A"
    return f"{start} → {end}"


def render_model_comparison(bundle: RunBundle) -> str:
    """Render a Markdown table from serialized held-out metrics only."""
    git = bundle.provenance.get("git", {})
    if not isinstance(git, Mapping):
        git = {}
    input_hashes = bundle.provenance.get("input_manifest_sha256", [])
    input_hash_text = (
        ", ".join(str(value) for value in input_hashes)
        if isinstance(input_hashes, Sequence) and not isinstance(input_hashes, str)
        else _format_plain(input_hashes)
    )
    lines = [
        f"> **{bundle.watermark}**",
        "",
        (
            f"Run `{bundle.run_id}`; observed UTC period "
            f"`{bundle.observed_start_utc}` to `{bundle.observed_end_utc}`."
        ),
        "",
        f"Configuration SHA-256: `{_format_plain(bundle.provenance.get('config_sha256'))}`.",
        f"Input manifest SHA-256: `{input_hash_text or 'none'}`.",
        (
            f"Git commit: `{_format_plain(git.get('commit'))}`; dirty at run time: "
            f"`{str(git.get('dirty')).lower()}`."
        ),
        "",
    ]
    rows = comparison_rows(bundle)
    if not rows:
        lines.extend(
            [
                "No held-out model-comparison rows were serialized in this run bundle.",
                "",
            ]
        )
        return "\n".join(lines)

    headers = (
        "Instrument",
        "Horizon",
        "Model",
        "Split",
        "N",
        "Test period (UTC)",
        "ROC-AUC",
        "PR-AUC",
        "Log loss",
        "Brier",
        "ECE",
        "Gross bps",
        "Fees bps",
        "Net bps",
        "Fill rate",
        "Turnover",
        "Max drawdown",
        "Selected on",
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = (
            _format_plain(row.get("instrument")),
            _format_plain(row.get("horizon_events")),
            _format_plain(row.get("model")),
            _format_plain(row.get("split")),
            _format_integer(row.get("n_obs")),
            _period(row),
            _format_estimate(row, "roc_auc", decimals=4),
            _format_estimate(row, "pr_auc", decimals=4),
            _format_estimate(row, "log_loss", decimals=4),
            _format_estimate(row, "brier_score", decimals=4),
            _format_estimate(row, "expected_calibration_error", decimals=4),
            _format_estimate(row, "gross_bps", decimals=3),
            _format_estimate(row, "fees_bps", decimals=3),
            _format_estimate(row, "net_bps", decimals=3),
            _format_estimate(row, "fill_rate", decimals=1, percent=True),
            _format_estimate(row, "turnover", decimals=3),
            _format_estimate(row, "max_drawdown", decimals=3),
            _format_plain(row.get("selected_on")),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            (
                "`N/A` means the producer did not serialize a comparable metric; "
                "it is never interpreted as zero. Confidence intervals, when supplied, "
                "are shown in brackets."
            ),
            "",
        ]
    )
    return "\n".join(lines)
