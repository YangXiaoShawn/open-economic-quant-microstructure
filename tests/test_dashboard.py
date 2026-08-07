from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from microstructure.reporting import write_checksum_manifest

PROJECT_ROOT = Path(__file__).parents[1]
APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _dashboard_bundle(root: Path) -> Path:
    _write_json(
        root / "run_manifest.json",
        {
            "artifacts": {
                "execution_metrics": "metrics/execution_metrics.json",
                "execution_sensitivity": "metrics/execution_sensitivity.json",
                "market_state": "dashboard/market_state.json",
                "predictive_metrics": "metrics/predictive_metrics.json",
                "quality_summary": "quality/summary.json",
            },
            "data": {
                "mode": "synthetic",
                "observed_end_utc": "2024-01-02T00:10:00Z",
                "observed_start_utc": "2024-01-02T00:00:00Z",
                "source": "synthetic_fixture_v1",
                "symbols": ["BTCUSDT", "ETHUSDT"],
            },
            "evidence_tier": "SYNTHETIC_SMOKE",
            "execution_assumptions": {"taker_fee_bps": 4.0},
            "run_id": "dashboard-smoke",
            "status": "complete",
        },
    )
    _write_json(
        root / "provenance.json",
        {
            "config_sha256": "c" * 64,
            "evidence_tier": "SYNTHETIC_SMOKE",
            "generated_at_utc": "2026-08-07T12:00:00Z",
            "git": {"commit": "UNBORN", "dirty": True},
            "input_manifest_sha256": ["d" * 64],
        },
    )
    _write_json(root / "metrics" / "predictive_metrics.json", [{"model": "baseline"}])
    _write_json(root / "metrics" / "execution_metrics.json", [{"net_bps": -1.0}])
    _write_json(
        root / "metrics" / "execution_sensitivity.json",
        [{"order_type": "market", "size_multiplier": 1.0, "net_pnl": -1.0}],
    )
    _write_json(root / "dashboard" / "market_state.json", [{"spread_bps": 2.0}])
    _write_json(root / "quality" / "summary.json", {"error_count": 0})
    write_checksum_manifest(root)
    (root / "_SUCCESS").write_text("", encoding="utf-8")
    return root


def test_dashboard_rejects_an_incomplete_run(tmp_path: Path, monkeypatch: Any) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    monkeypatch.setenv("MICROSTRUCTURE_RUN_DIR", str(incomplete))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.error
    assert "incomplete or invalid" in app.error[0].value
    assert "_SUCCESS" in app.caption[0].value


def test_dashboard_is_read_only_and_displays_all_evidence_tabs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _dashboard_bundle(tmp_path / "complete")
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("MICROSTRUCTURE_RUN_DIR", str(run_dir))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.warning
    assert "SYNTHETIC SMOKE" in app.warning[0].value
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Data Quality",
        "Market State",
        "Predictions",
        "Simulated Performance",
        "Reproducibility & Limitations",
    ]
    assert any("Execution sensitivity grid" in item.value for item in app.markdown)
    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
