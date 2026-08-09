from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from microstructure.reporting.l2 import (
    L2ReportData,
    L2ReportError,
    canonical_report_data_sha256,
    render_l2_executive_memo,
    render_l2_model_comparison,
    render_l2_technical_report,
    write_l2_report_set,
)


def _data() -> L2ReportData:
    manifest = {
        "status": "COMPLETE",
        "evidence_tier": "FULL_DATA",
        "effective_evidence_tier": "FULL_DATA",
        "live_trading": False,
        "research": {
            "question": "Do causal L2 states improve future-mid direction log loss?",
            "period_start_utc": "2026-08-10T14:00:00Z",
            "period_end_utc": "2026-08-13T15:00:00Z",
        },
    }
    provenance = {
        "git": {"commit": "a" * 40, "source_tree_sha256": "b" * 64, "dirty": False},
        "inputs": {
            "capture_config_sha256": "c" * 64,
            "capture_protocol_sha256": "d" * 64,
            "analysis_config_sha256": "e" * 64,
            "development_lock_sha256": "f" * 64,
        },
    }
    session_gates = tuple(
        {
            "study_date": f"2026-08-{day:02d}",
            "study_role": role,
            "status": "COMPLETE",
            "BTCUSDT_gate": "passed",
            "ETHUSDT_gate": "passed",
            "overlap_seconds": 3_590.0,
        }
        for day, role in (
            (8, "train"),
            (9, "validation"),
            (10, "primary_test"),
            (11, "replication_test"),
        )
    )
    predictive = (
        {
            "symbol": "BTCUSDT",
            "endpoint_name": "event_20",
            "study_date": "2026-08-12",
            "selected_model": "logistic_l2_c_1",
            "n_obs": 400,
            "selected_log_loss": 0.65,
            "prior_log_loss": 0.69,
            "point_delta": -0.04,
            "selected_brier_score": 0.23,
            "selected_expected_calibration_error": 0.02,
        },
    )
    paired = (
        {
            "symbol": "BTCUSDT",
            "endpoint_name": "event_20",
            "study_date": "2026-08-12",
            "n_obs": 400,
            "n_blocks": 10,
            "point_delta": -0.04,
            "ci_low": -0.08,
            "ci_high": 0.01,
            "status": "ok",
            "regime": "ALL",
        },
    )
    equal = (
        {
            **{key: value for key, value in paired[0].items() if key != "study_date"},
            "directionally_replicated": True,
        },
    )
    execution = (
        {
            "symbol": "BTCUSDT",
            "endpoint_name": "event_20",
            "study_date": "2026-08-12",
            "decision_latency_events": 0,
            "order_latency_events": 1,
            "strategy_orders": 20,
            "fill_ratio": 0.8,
            "turnover_notional": 1_000.0,
            "marked_net_pnl": -2.0,
            "unliquidated_quantity": 0.0,
        },
    )
    return L2ReportData(
        manifest=manifest,
        provenance=provenance,
        session_gates=session_gates,
        hypothesis={
            "conclusion": "The endpoint improved on primary and replication sessions.",
            "directionally_replicated_pairs": 1,
        },
        predictive_metrics=predictive,
        paired_metrics=paired,
        equal_session_metrics=equal,
        execution_metrics=execution,
    )


def test_l2_reports_are_artifact_driven_and_keep_claim_boundaries() -> None:
    data = _data()
    technical = render_l2_technical_report(data)
    memo = render_l2_executive_memo(data)
    comparison = render_l2_model_comparison(data)

    for report in (technical, memo, comparison):
        assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in report
        assert "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff" in report
        assert "no refit" in report.lower() or "without update" in report.lower()
    assert "not realized execution" in technical.lower()
    assert "no capacity or profitability claim" in technical.lower()
    assert "Directionally replicated symbol/endpoint pairs: **1**" in memo
    assert "2026-08-12 / ALL" in technical
    assert "equal-session / ALL" in comparison
    assert "0.650000" in comparison
    assert len(canonical_report_data_sha256(data)) == 64


def test_l2_report_set_is_deterministic_and_complete(tmp_path: Path) -> None:
    paths = write_l2_report_set(tmp_path, _data())
    first = [path.read_bytes() for path in paths]
    repeated = write_l2_report_set(tmp_path, _data())

    assert paths == repeated
    assert [path.read_bytes() for path in repeated] == first
    assert {path.name for path in paths} == {
        "technical_report.md",
        "executive_memo.md",
        "model_comparison.md",
    }


def test_l2_reports_reject_promoted_or_underspecified_authority() -> None:
    data = _data()
    with pytest.raises(L2ReportError, match="FULL_DATA"):
        render_l2_technical_report(
            replace(
                data,
                manifest={**data.manifest, "evidence_tier": "PUBLIC_SAMPLE_PARTIAL"},
            )
        )

    with pytest.raises(L2ReportError, match="conclusion"):
        render_l2_executive_memo(replace(data, hypothesis={}))


def test_l2_report_counts_only_overall_pairs_and_labels_insufficient_data() -> None:
    data = _data()
    duplicated_regime = {
        **data.equal_session_metrics[0],
        "regime": "HIGH_SPREAD__HIGH_VOLATILITY",
        "directionally_replicated": True,
    }
    memo = render_l2_executive_memo(
        replace(data, equal_session_metrics=(*data.equal_session_metrics, duplicated_regime))
    )
    assert "Directionally replicated symbol/endpoint pairs: **1**" in memo

    insufficient_manifest = {
        **data.manifest,
        "status": "INSUFFICIENT_DATA",
        "effective_evidence_tier": "INSUFFICIENT_DATA",
    }
    insufficient = replace(
        data,
        manifest=insufficient_manifest,
        hypothesis={
            "conclusion": "The frozen study is INSUFFICIENT_DATA.",
            "directionally_replicated_pairs": 0,
        },
        predictive_metrics=(),
        paired_metrics=(),
        equal_session_metrics=(),
        execution_metrics=(),
    )
    technical = render_l2_technical_report(insufficient)
    assert "INSUFFICIENT_DATA" in technical
    assert "FULL-DATA PUBLIC L2 RESEARCH" not in technical


def test_l2_report_rejects_replicated_pair_count_mismatch() -> None:
    with pytest.raises(L2ReportError, match="replicated-pair count"):
        render_l2_executive_memo(
            replace(
                _data(),
                hypothesis={
                    "conclusion": "Mismatch.",
                    "directionally_replicated_pairs": 2,
                },
            )
        )


def test_l2_report_rejects_invalid_execution_fill_ratio() -> None:
    data = _data()
    invalid = ({**data.execution_metrics[0], "fill_ratio": 1.01},)
    with pytest.raises(L2ReportError, match=r"\[0, 1\]"):
        render_l2_technical_report(replace(data, execution_metrics=invalid))
