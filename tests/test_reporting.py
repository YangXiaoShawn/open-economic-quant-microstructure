from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from microstructure.reporting import (
    ChecksumMismatchError,
    IncompleteRunError,
    RunBundleValidationError,
    comparison_rows,
    load_run_bundle,
    render_model_comparison,
    render_technical_report,
    write_checksum_manifest,
    write_report_set,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_bundle(
    root: Path,
    *,
    evidence_tier: str = "SYNTHETIC_SMOKE",
    data_mode: str = "synthetic",
    data_source: str = "synthetic_fixture_v1",
    finalize: bool = True,
) -> Path:
    manifest = {
        "artifacts": {
            "execution_metrics": "metrics/execution_metrics.json",
            "execution_sensitivity": "metrics/execution_sensitivity.json",
            "market_state": "dashboard/market_state.json",
            "predictive_metrics": "metrics/predictive_metrics.json",
            "quality_summary": "quality/summary.json",
        },
        "data": {
            "mode": data_mode,
            "observed_end_utc": "2024-01-02T00:10:00Z",
            "observed_start_utc": "2024-01-02T00:00:00Z",
            "source": data_source,
            "symbols": ["BTCUSDT", "ETHUSDT"],
        },
        "evidence_tier": evidence_tier,
        "execution_assumptions": {
            "decision_latency_events": 1,
            "taker_fee_bps": 4.0,
        },
        "run_id": "unit-smoke",
        "schema_version": "1.0.0",
        "status": "complete",
    }
    provenance = {
        "config_sha256": "a" * 64,
        "evidence_tier": evidence_tier,
        "generated_at_utc": "2026-08-07T12:00:00Z",
        "git": {"commit": "UNBORN", "dirty": True},
        "input_manifest_sha256": ["b" * 64],
        "runtime": {"machine": "test", "python": "3.12.0"},
        "seed": 7,
    }
    predictive = [
        {
            "brier_score": 0.25,
            "expected_calibration_error": 0.1,
            "horizon_events": 20,
            "instrument": "BTCUSDT",
            "log_loss": 0.6931,
            "model": "majority",
            "n_obs": 200,
            "period_end_utc": "2024-01-02T00:10:00Z",
            "period_start_utc": "2024-01-02T00:08:00Z",
            "pr_auc": 0.5,
            "roc_auc": 0.5,
            "roc_auc_ci_high": 0.55,
            "roc_auc_ci_low": 0.45,
            "selected_on": "validation:log_loss",
            "split": "test",
        },
        {
            "instrument": "BTCUSDT",
            "model": "validation-only-model",
            "roc_auc": 0.99,
            "split": "validation",
        },
    ]
    execution = [
        {
            "fees_bps": 4.0,
            "fill_rate": 0.6,
            "gross_bps": 1.0,
            "horizon_events": 20,
            "instrument": "BTCUSDT",
            "max_drawdown": -2.0,
            "model": "majority",
            "net_bps": -3.0,
            "split": "test",
            "turnover": 8.0,
        }
    ]
    _write_json(root / "run_manifest.json", manifest)
    _write_json(root / "provenance.json", provenance)
    _write_json(root / "metrics" / "predictive_metrics.json", predictive)
    _write_json(root / "metrics" / "execution_metrics.json", execution)
    _write_json(
        root / "metrics" / "execution_sensitivity.json",
        [
            {
                "order_type": "market",
                "size_multiplier": 1.0,
                "net_pnl": -0.5,
                "net_edge_bps": -3.0,
                "fill_ratio": 1.0,
                "turnover_notional": 8.0,
                "maximum_drawdown": 0.5,
            }
        ],
    )
    _write_json(root / "dashboard" / "market_state.json", [{"spread_bps": 2.0}])
    _write_json(root / "quality" / "summary.json", {"error_count": 0, "warning_count": 1})
    if finalize:
        write_checksum_manifest(root)
        (root / "_SUCCESS").write_text("", encoding="utf-8")
    return root


def test_rendering_is_deterministic_held_out_only_and_watermarked(tmp_path: Path) -> None:
    bundle = load_run_bundle(_build_bundle(tmp_path / "run"))

    first = render_technical_report(bundle)
    second = render_technical_report(bundle)
    table = render_model_comparison(bundle)

    assert first == second
    assert "SYNTHETIC SMOKE — SOFTWARE VALIDATION ONLY" in first
    assert "2024-01-02T00:00:00Z" in first
    assert "Configuration SHA-256" in first
    assert "UNBORN" in first
    assert "Runtime metadata" in first
    assert "Seed" in first
    assert "0.5000 [0.4500, 0.5500]" in table
    assert "0.2500" in table
    assert "0.1000" in table
    assert "-3.000" in table
    assert "Configuration SHA-256" in table
    assert "Input manifest SHA-256" in table
    assert "Git commit" in table
    assert "Execution sensitivity" in first
    assert "size_multiplier" in first
    assert len(bundle.execution_sensitivity) == 1
    assert "validation-only-model" not in table
    assert len(comparison_rows(bundle)) == 1


def test_report_set_is_written_outside_frozen_bundle(tmp_path: Path) -> None:
    run_dir = _build_bundle(tmp_path / "run")
    checksum_before = (run_dir / "checksums.sha256").read_bytes()
    bundle = load_run_bundle(run_dir)

    paths = write_report_set(bundle, tmp_path / "published")

    assert paths.technical_report.is_file()
    assert paths.executive_memo.is_file()
    assert paths.model_comparison.is_file()
    memo = paths.executive_memo.read_text(encoding="utf-8")
    comparison = paths.model_comparison.read_text(encoding="utf-8")
    assert "a" * 64 in memo
    assert "b" * 64 in memo
    assert "UNBORN" in memo
    assert "a" * 64 in comparison
    assert "b" * 64 in comparison
    assert "UNBORN" in comparison
    assert (run_dir / "checksums.sha256").read_bytes() == checksum_before
    assert load_run_bundle(run_dir).run_id == "unit-smoke"


def test_incomplete_and_tampered_bundles_fail_clearly(tmp_path: Path) -> None:
    incomplete = _build_bundle(tmp_path / "incomplete", finalize=False)
    with pytest.raises(IncompleteRunError, match="_SUCCESS"):
        load_run_bundle(incomplete)

    tampered = _build_bundle(tmp_path / "tampered")
    (tampered / "quality" / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        load_run_bundle(tampered)


def test_synthetic_source_cannot_be_promoted_to_public_evidence(tmp_path: Path) -> None:
    run_dir = _build_bundle(
        tmp_path / "laundered",
        evidence_tier="PUBLIC_SAMPLE_PARTIAL",
        data_mode="synthetic",
        data_source="synthetic_fixture_v1",
    )
    with pytest.raises(RunBundleValidationError, match="cannot be promoted"):
        load_run_bundle(run_dir)


def test_full_data_requires_manifested_complete_coverage(tmp_path: Path) -> None:
    run_dir = _build_bundle(
        tmp_path / "incomplete_full_data",
        evidence_tier="FULL_DATA",
        data_mode="binance_rest",
        data_source="binance_spot_public_rest",
    )

    with pytest.raises(RunBundleValidationError, match="complete coverage"):
        load_run_bundle(run_dir)


def test_canonical_documents_make_no_unrun_performance_claim() -> None:
    technical = (PROJECT_ROOT / "reports" / "technical_report.md").read_text(encoding="utf-8")
    comparison = (PROJECT_ROOT / "reports" / "model_comparison.md").read_text(encoding="utf-8")
    memo = (PROJECT_ROOT / "reports" / "executive_memo.md").read_text(encoding="utf-8")
    resume = (PROJECT_ROOT / "portfolio" / "resume_bullets.md").read_text(encoding="utf-8")

    assert "SOURCE-CONTROLLED TEMPLATE" in technical
    assert "manually copied empirical or synthetic performance results" in technical
    assert "no copied model or execution numbers" in comparison
    assert memo.count("page-break-after: always") == 1
    assert "Authorize no capital" in memo
    assert "## Research-focused" in resume
    assert "## Quant-trading-focused" in resume
    assert "## Data-engineering-focused" in resume
