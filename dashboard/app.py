"""Read-only Streamlit dashboard for a completed microstructure run bundle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from microstructure.provenance import sha256_file
from microstructure.reporting import RunBundle, RunBundleError, load_run_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "artifacts" / "runs" / "sample-smoke"


def _argument_run_dir(arguments: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir")
    parsed, _ = parser.parse_known_args(arguments)
    configured = parsed.run_dir or os.environ.get("MICROSTRUCTURE_RUN_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_RUN_DIR


def _integrity_key(run_dir: Path) -> str:
    checksum_path = run_dir / "checksums.sha256"
    return sha256_file(checksum_path) if checksum_path.is_file() else "missing"


@st.cache_resource(show_spinner=False)
def _cached_bundle(run_dir: str, integrity_key: str) -> RunBundle:
    del integrity_key  # It is part of Streamlit's cache key.
    return load_run_bundle(run_dir)


def _rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _show_rows(rows: Sequence[Mapping[str, Any]], empty_message: str) -> None:
    if rows:
        st.dataframe(_rows(rows), hide_index=True)
    else:
        st.info(empty_message)


def _show_overview(bundle: RunBundle) -> None:
    st.subheader("Evidence and lineage")
    columns = st.columns(4)
    columns[0].metric("Run", bundle.run_id)
    columns[1].metric("Evidence", bundle.evidence_tier)
    columns[2].metric("Symbols", len(bundle.symbols))
    columns[3].metric(
        "Git state",
        "dirty" if bool(cast_mapping(bundle.provenance.get("git")).get("dirty")) else "clean",
    )
    st.markdown(
        f"**Observed UTC period:** `{bundle.observed_start_utc}` → `{bundle.observed_end_utc}`"
    )
    st.markdown(f"**Instruments:** {', '.join(bundle.symbols)}")
    st.caption(
        "The dashboard reads serialized artifacts only. It does not download data, "
        "train models, or simulate orders."
    )


def cast_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _show_quality(bundle: RunBundle) -> None:
    st.subheader("Non-mutating validation findings")
    if bundle.quality:
        st.json(dict(bundle.quality), expanded=True)
    else:
        st.info("No quality summary was serialized in this completed run bundle.")
    st.caption("Findings are displayed as recorded; this app does not repair observations.")


def _show_market_state(bundle: RunBundle) -> None:
    st.subheader("Market-state aggregates")
    _show_rows(
        bundle.market_state,
        "No dashboard-safe market-state aggregate was serialized for this run.",
    )
    st.caption(
        "Only bounded aggregates are loaded here; the dashboard never scans external raw data."
    )


def _show_predictions(bundle: RunBundle) -> None:
    st.subheader("Serialized predictive diagnostics")
    _show_rows(
        bundle.predictive_metrics,
        "No predictive metric rows were serialized for this run.",
    )
    st.caption(
        "Predictive metrics do not establish fillability or performance after execution costs."
    )


def _show_execution(bundle: RunBundle) -> None:
    st.subheader("Serialized simulated performance")
    _show_rows(
        bundle.execution_metrics,
        "No execution or simulated-performance rows were serialized for this run.",
    )
    st.markdown("#### Execution sensitivity grid")
    _show_rows(
        bundle.execution_sensitivity,
        "No execution-sensitivity rows were serialized for this run.",
    )
    assumptions = bundle.manifest.get("execution_assumptions")
    if isinstance(assumptions, Mapping) and assumptions:
        st.markdown("#### Recorded execution assumptions")
        st.json(dict(assumptions), expanded=False)
    st.caption(
        "Fees, latency, fills, adverse selection, inventory, and liquidation are model "
        "assumptions—not realized trading outcomes."
    )


def _show_reproducibility(bundle: RunBundle) -> None:
    st.subheader("Frozen provenance")
    st.markdown(f"**Run directory:** `{bundle.root}`")
    st.markdown(f"**Configuration SHA-256:** `{bundle.provenance.get('config_sha256', 'N/A')}`")
    st.markdown("#### Run manifest")
    st.code(json.dumps(bundle.manifest, indent=2, sort_keys=True), language="json")
    st.markdown("#### Provenance")
    st.code(json.dumps(bundle.provenance, indent=2, sort_keys=True), language="json")
    st.caption(
        "The completion marker and checksum manifest were verified before these values loaded."
    )


def render_dashboard(bundle: RunBundle) -> None:
    """Render a verified bundle without changing it."""
    st.title("Order Flow to Price Impact")
    if bundle.evidence_tier in {"SYNTHETIC_SMOKE", "PUBLIC_SAMPLE_PARTIAL"}:
        st.warning(bundle.watermark)
    else:
        st.info(bundle.watermark)

    labels = (
        "Overview",
        "Data Quality",
        "Market State",
        "Predictions",
        "Simulated Performance",
        "Reproducibility & Limitations",
    )
    tabs = st.tabs(labels)
    with tabs[0]:
        _show_overview(bundle)
    with tabs[1]:
        _show_quality(bundle)
    with tabs[2]:
        _show_market_state(bundle)
    with tabs[3]:
        _show_predictions(bundle)
    with tabs[4]:
        _show_execution(bundle)
    with tabs[5]:
        _show_reproducibility(bundle)


def main(arguments: Sequence[str] | None = None) -> None:
    st.set_page_config(page_title="Microstructure Research", layout="wide")
    run_dir = _argument_run_dir(sys.argv[1:] if arguments is None else arguments).resolve()
    try:
        bundle = _cached_bundle(str(run_dir), _integrity_key(run_dir))
    except RunBundleError as error:
        st.title("Order Flow to Price Impact")
        st.error(f"Run bundle is incomplete or invalid: {error}")
        st.caption(
            "Select a directory containing run_manifest.json, provenance.json, "
            "checksums.sha256, and the final _SUCCESS marker."
        )
        st.stop()
    render_dashboard(bundle)


if __name__ == "__main__":
    main()
