from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from microstructure.reporting import (
    ChecksumMismatchError,
    IncompleteRunError,
    RunBundleValidationError,
    comparison_rows,
    load_run_bundle,
    render_executive_memo,
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
        {
            "instrument": "BTCUSDT",
            "model": "unsplit-model",
            "roc_auc": 0.999,
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
    assert "unsplit-model" not in table
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


def test_checksum_manifest_fsyncs_file_before_replace_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "durable-checksums"
    root.mkdir()
    (root / "artifact.txt").write_text("evidence\n", encoding="utf-8")
    events: list[str] = []
    original_fsync = os.fsync
    original_replace = os.replace

    def fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        events.append("fsync_directory" if stat.S_ISDIR(mode) else "fsync_file")
        original_fsync(descriptor)

    def replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    write_checksum_manifest(root)
    assert events == ["fsync_file", "replace", "fsync_directory"]


def test_complete_bundle_rejects_conflicting_insufficient_marker(tmp_path: Path) -> None:
    complete = _build_bundle(tmp_path / "conflicting-terminal")
    (complete / "INSUFFICIENT_DATA").write_text("terminal\n", encoding="utf-8")
    with pytest.raises(IncompleteRunError, match="conflicting"):
        load_run_bundle(complete)


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


def test_public_bundle_requires_manifested_input_identity(tmp_path: Path) -> None:
    run_dir = _build_bundle(
        tmp_path / "missing-input",
        evidence_tier="PUBLIC_SAMPLE_PARTIAL",
        data_mode="binance_rest",
        data_source="binance_spot_public_rest",
        finalize=False,
    )
    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_manifest_sha256"] = []
    _write_json(provenance_path, provenance)
    write_checksum_manifest(run_dir)
    (run_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    with pytest.raises(RunBundleValidationError, match="at least one input manifest"):
        load_run_bundle(run_dir)


def test_public_trade_only_report_uses_manifest_scope_and_excludes_execution(
    tmp_path: Path,
) -> None:
    run_dir = _build_bundle(
        tmp_path / "public-trade-only",
        evidence_tier="PUBLIC_SAMPLE_PARTIAL",
        data_mode="binance_rest_trade_only",
        data_source="binance_spot_public_rest",
        finalize=False,
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    question = "Do signed aggregate trades predict future trade-price direction?"
    exclusion_reason = "Trade-only data contain no quotes, depth, or queue state."
    manifest["research"] = {"question": question}
    manifest["data"]["symbol_coverage"] = [
        {
            "symbol": "BTCUSDT",
            "rows": 5_000,
            "observed_start_utc": "2024-01-02T00:00:00Z",
            "observed_end_inclusive_utc": "2024-01-02T00:01:53.619000Z",
            "complete_range": False,
        },
        {
            "symbol": "ETHUSDT",
            "rows": 5_000,
            "observed_start_utc": "2024-01-02T00:00:00Z",
            "observed_end_inclusive_utc": "2024-01-02T00:06:49.796000Z",
            "complete_range": False,
        },
    ]
    manifest["execution_assumptions"] = {
        "status": "NOT_RUN",
        "reason": exclusion_reason,
        "pnl_calculated": False,
        "fills_calculated": False,
    }
    manifest["artifacts"]["hypothesis_evaluation"] = "metrics/hypothesis_evaluation.json"
    _write_json(manifest_path, manifest)
    _write_json(
        run_dir / "metrics" / "hypothesis_evaluation.json",
        {
            "selection_metric": "log_loss",
            "caveat": ("Exploratory paired interval only; no significance claim is authorized."),
            "cross_instrument_conclusion": {
                "status": "not_inferred",
                "text": "BTCUSDT and ETHUSDT are not pooled.",
            },
            "per_symbol": [
                {
                    "symbol": "BTCUSDT",
                    "selected_model": "logistic_l2_c1",
                    "baseline": "historical_prior",
                    "metric": "log_loss",
                    "point_delta": -0.012345,
                    "ci_low": -0.02,
                    "ci_high": 0.003,
                    "n_obs": 400,
                    "n_blocks": 10,
                    "samples": 500,
                    "seed": 20280807,
                    "status": "ok",
                    "favorable_direction": "negative_selected_minus_prior_is_favorable",
                },
                {
                    "symbol": "ETHUSDT",
                    "selected_model": "shallow_tree_d2_leaf25",
                    "baseline": "historical_prior",
                    "metric": "log_loss",
                    "point_delta": 0.006789,
                    "ci_low": -0.004,
                    "ci_high": 0.018,
                    "n_obs": 400,
                    "n_blocks": 10,
                    "samples": 500,
                    "seed": 20380807,
                    "status": "ok",
                    "favorable_direction": "negative_selected_minus_prior_is_favorable",
                },
            ],
        },
    )
    _write_json(run_dir / "metrics" / "execution_metrics.json", [])
    _write_json(run_dir / "metrics" / "execution_sensitivity.json", [])
    write_checksum_manifest(run_dir)
    (run_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    bundle = load_run_bundle(run_dir)
    technical = render_technical_report(bundle)
    memo = render_executive_memo(bundle)
    published = write_report_set(bundle, tmp_path / "published-public-trade-only")

    assert bundle.execution_metrics == ()
    assert bundle.execution_sensitivity == ()
    assert len(bundle.hypothesis_evaluation["per_symbol"]) == 2
    assert question in technical
    assert "### Per-symbol observed coverage" in technical
    assert "| BTCUSDT | 5000 |" in technical
    assert "2024-01-02T00:01:53.619000Z" in technical
    assert "| ETHUSDT | 5000 |" in technical
    assert "2024-01-02T00:06:49.796000Z" in technical
    assert (
        "| BTCUSDT | 5000 | 2024-01-02T00:00:00Z | 2024-01-02T00:01:53.619000Z | false |"
    ) in technical
    assert (
        "| ETHUSDT | 5000 | 2024-01-02T00:00:00Z | 2024-01-02T00:06:49.796000Z | false |"
    ) in technical
    assert "Execution simulation and P&L were not run" in technical
    assert "Execution sensitivity was not run" in technical
    assert exclusion_reason in technical
    assert "conditional on these assumptions" not in technical
    assert "This scenario grid changes" not in technical
    assert "Execution simulation, fills, execution sensitivity, and P&L were not run" in memo
    assert exclusion_reason in memo
    assert "simulated strategy outcomes" not in memo
    assert "Simulated fills do not prove" not in memo
    for rendered in (
        technical,
        memo,
        published.technical_report.read_text(encoding="utf-8"),
        published.executive_memo.read_text(encoding="utf-8"),
        published.model_comparison.read_text(encoding="utf-8"),
    ):
        assert "Paired H0/H1 diagnostic" in rendered
        assert "negative value favors the selected model" in rendered
        assert "BTCUSDT" in rendered
        assert "-0.012345" in rendered
        assert "ETHUSDT" in rendered
        assert "0.006789" in rendered
        assert "not p-values or confirmatory significance intervals" in rendered
        assert "cannot support persistent alpha" in rendered


def test_m8_full_archive_report_preserves_date_components_and_narrow_scope(
    tmp_path: Path,
) -> None:
    run_dir = _build_bundle(
        tmp_path / "m8-trade-only",
        evidence_tier="FULL_DATA",
        data_mode="binance_spot_daily_aggtrades_trade_only",
        data_source="binance_spot_daily_aggtrades_archive",
        finalize=False,
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data"]["all_requested_ranges_complete"] = True
    manifest["data"]["date_coverage"] = [
        {
            "symbol": "BTCUSDT",
            "date": "2024-01-05",
            "role": "primary_test",
            "rows": 100,
            "observed_start_utc": "2024-01-05T00:00:00Z",
            "observed_end_inclusive_utc": "2024-01-05T23:59:59.999000Z",
            "complete": True,
            "quality_errors": 0,
            "quality_warnings": 0,
        },
        {
            "symbol": "BTCUSDT",
            "date": "2024-01-06",
            "role": "replication_test",
            "rows": 120,
            "observed_start_utc": "2024-01-06T00:00:00Z",
            "observed_end_inclusive_utc": "2024-01-06T23:59:59.999000Z",
            "complete": True,
            "quality_errors": 0,
            "quality_warnings": 0,
        },
    ]
    manifest["research"] = {
        "scope": "trade_only",
        "question": "Does the locked trade-only model improve next-20-trade log loss?",
    }
    manifest["execution_assumptions"] = {
        "status": "NOT_RUN",
        "reason": "No contemporaneous order book or local receipt clock.",
    }
    manifest["artifacts"]["hypothesis_evaluation"] = "metrics/hypothesis_evaluation.json"
    _write_json(manifest_path, manifest)
    _write_json(run_dir / "metrics" / "execution_metrics.json", [])
    _write_json(run_dir / "metrics" / "execution_sensitivity.json", [])
    _write_json(
        run_dir / "metrics" / "predictive_metrics.json",
        [
            {
                "instrument": "BTCUSDT",
                "model": "tree_depth_2",
                "horizon_events": 20,
                "split": "final_test",
                "study_date": "2024-01-05",
                "n_obs": 100,
                "period_start_utc": "2024-01-05T00:00:00Z",
                "period_end_utc": "2024-01-05T23:59:59.999000Z",
                "log_loss": 0.61,
            },
            {
                "instrument": "BTCUSDT",
                "model": "tree_depth_2",
                "horizon_events": 20,
                "split": "final_test",
                "study_date": "2024-01-06",
                "n_obs": 120,
                "period_start_utc": "2024-01-06T00:00:00Z",
                "period_end_utc": "2024-01-06T23:59:59.999000Z",
                "log_loss": 0.65,
            },
        ],
    )
    _write_json(
        run_dir / "metrics" / "hypothesis_evaluation.json",
        {
            "schema_version": "1.0.0",
            "evidence_scope": "trade_only_complete_predeclared_daily_archives",
            "selection_metric": "log_loss",
            "caveat": "No p-values, H0 rejection, or significance claim is authorized.",
            "cross_instrument_conclusion": {
                "text": "BTCUSDT and ETHUSDT remain separate; no pooling is authorized."
            },
            "per_date": [
                {
                    "symbol": "BTCUSDT",
                    "study_date": "2024-01-05",
                    "study_role": "primary_test",
                    "selected_model": "tree_depth_2",
                    "baseline": "historical_prior",
                    "selected_log_loss": 0.61,
                    "prior_log_loss": 0.64,
                    "point_delta": -0.03,
                    "ci_low": -0.05,
                    "ci_high": -0.01,
                    "n_obs": 100,
                    "n_blocks": 3,
                    "bootstrap_status": "ok",
                },
                {
                    "symbol": "BTCUSDT",
                    "study_date": "2024-01-06",
                    "study_role": "replication_test",
                    "selected_model": "tree_depth_2",
                    "baseline": "historical_prior",
                    "selected_log_loss": 0.65,
                    "prior_log_loss": 0.64,
                    "point_delta": 0.01,
                    "ci_low": -0.02,
                    "ci_high": 0.04,
                    "n_obs": 120,
                    "n_blocks": 3,
                    "bootstrap_status": "ok",
                },
            ],
            "per_symbol": [
                {
                    "symbol": "BTCUSDT",
                    "selected_model": "tree_depth_2",
                    "baseline": "historical_prior",
                    "point_delta": -0.01,
                    "ci_low": -0.03,
                    "ci_high": 0.02,
                    "n_dates": 2,
                    "n_obs": 220,
                    "n_blocks": 6,
                    "status": "mixed",
                    "replication_status": "mixed",
                    "validation_date": "2024-01-04",
                    "validation_point_delta": -0.02,
                    "primary_date": "2024-01-05",
                    "primary_point_delta": -0.03,
                    "replication_date": "2024-01-06",
                    "replication_point_delta": 0.01,
                    "direction_consistent_across_validation_primary_replication": False,
                    "favorable_across_validation_primary_replication": False,
                    "validation_primary_replication_status": "mixed",
                }
            ],
        },
    )
    write_checksum_manifest(run_dir)
    (run_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    bundle = load_run_bundle(run_dir)
    technical = render_technical_report(bundle)
    memo = render_executive_memo(bundle)
    published = write_report_set(bundle, tmp_path / "m8-published")
    held_out_rows = comparison_rows(bundle)

    assert len(held_out_rows) == 2
    assert {row["study_date"] for row in held_out_rows} == {"2024-01-05", "2024-01-06"}
    assert "### Per-date observed coverage" in technical
    assert "M8 predeclared multi-date endpoint" in technical
    assert "2024-01-05" in technical and "2024-01-06" in technical
    assert "-0.030000" in technical and "0.010000" in technical
    assert "Equal-date-weighted endpoint" in technical
    assert "Validation → primary → replication direction consistency" in technical
    assert "2024-01-04" in technical
    assert "mixed" in technical
    assert "not to full market observability" in technical
    assert "contains no order-book, execution, P&L, capacity" in memo
    for path in (
        published.technical_report,
        published.executive_memo,
        published.model_comparison,
    ):
        regenerated = path.read_text(encoding="utf-8")
        assert "M8 predeclared multi-date endpoint" in regenerated
        assert "2024-01-05" in regenerated and "2024-01-06" in regenerated
        assert "No p-values, H0 rejection, or significance claim is authorized" in regenerated


def test_manifest_and_provenance_run_keys_must_match(tmp_path: Path) -> None:
    run_dir = _build_bundle(tmp_path / "mismatched-run-key", finalize=False)
    manifest_path = run_dir / "run_manifest.json"
    provenance_path = run_dir / "provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    manifest["run_key"] = "c" * 64
    provenance["run_key"] = "d" * 64
    _write_json(manifest_path, manifest)
    _write_json(provenance_path, provenance)
    write_checksum_manifest(run_dir)
    (run_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")

    with pytest.raises(RunBundleValidationError, match="run keys do not match"):
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
