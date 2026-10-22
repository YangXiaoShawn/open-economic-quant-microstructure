"""Verify Microstructure Maker queue-position and fee sensitivity table against the 144 real scenarios and oeq-core."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE_SRC = ROOT.parent / "packages" / "oeq-core" / "src"
if CORE_SRC.exists() and str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from oeq_core import (  # noqa: E402
    ClaimBoundary,
    EvidenceGrade,
    build_run_manifest,
    turnover_penalized_net_return,
    validate_claim_boundary,
    verify_manifest_checksums,
)


class TestMicrostructureMakerFeeSensitivity(unittest.TestCase):
    def test_maker_queue_fee_table_matches_144_real_scenarios(self) -> None:
        csv_path = ROOT / "reports" / "maker_queue_fee_sensitivity.csv"
        json_path = ROOT / "reports" / "microstructure_144_scenarios.json"
        with csv_path.open("r", encoding="utf-8") as fh:
            rows = {r["execution_tier"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(rows), 6)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        gross_edges = sorted(float(s["gross_edge_bps"]) for s in payload["scenarios"])
        self.assertEqual(len(gross_edges), 144)
        gross_pos = sum(1 for x in gross_edges if x > 0.0)
        self.assertEqual(gross_pos, 110)
        max_gross = max(gross_edges)
        med_gross = (gross_edges[71] + gross_edges[72]) / 2.0

        for tier, row in rows.items():
            hurdle = float(row["effective_hurdle_bps"])
            expected_net_pos = sum(1 for x in gross_edges if x > hurdle)
            self.assertEqual(int(row["scenarios_gross_positive"]), gross_pos)
            self.assertEqual(int(row["scenarios_net_positive"]), expected_net_pos, msg=tier)
            self.assertAlmostEqual(float(row["best_scenario_net_edge_bps"]), max_gross - hurdle, places=3, msg=tier)
            self.assertAlmostEqual(float(row["median_scenario_net_edge_bps"]), med_gross - hurdle, places=3, msg=tier)

        net_ret = turnover_penalized_net_return(gross_return=max_gross / 10_000.0, turnover=1.0, cost_bps=0.45)
        self.assertAlmostEqual(net_ret * 10_000.0, float(rows["Maker 0bp (Front-of-Queue)"]["best_scenario_net_edge_bps"]), places=3)

    def test_claim_boundary_and_manifest(self) -> None:
        boundary = ClaimBoundary(
            project_slug="microstructure",
            grade=EvidenceGrade.OOS_PREDICTIVE_COMPARISON,
            allows_causal_language=False,
            allows_live_profitability_claim=False,
        )
        self.assertEqual(validate_claim_boundary(boundary), [])
        manifest = build_run_manifest(
            project_slug="microstructure",
            root=ROOT,
            files=[
                ROOT / "reports" / "maker_queue_fee_sensitivity.csv",
                ROOT / "reports" / "microstructure_144_scenarios.json",
                ROOT / "reports" / "technical_report.md",
            ],
            evidence_grade=boundary.grade.value,
        )
        self.assertEqual(verify_manifest_checksums(ROOT, manifest), [])


if __name__ == "__main__":
    unittest.main()
