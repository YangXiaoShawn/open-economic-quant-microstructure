"""Verify Microstructure Maker queue-position and fee sensitivity table against oeq-core."""
from __future__ import annotations

import csv
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
    def test_maker_queue_fee_table_and_breakeven(self) -> None:
        path = ROOT / "reports" / "maker_queue_fee_sensitivity.csv"
        with path.open("r", encoding="utf-8") as fh:
            rows = {r["execution_tier"]: r for r in csv.DictReader(fh)}
        self.assertEqual(len(rows), 6)

        # Frozen retail taker baseline: 110 gross positive, 0 net positive at 4.00 bp
        baseline = rows["Retail Taker (Frozen Baseline)"]
        self.assertEqual(int(baseline["scenarios_gross_positive"]), 110)
        self.assertEqual(int(baseline["scenarios_net_positive"]), 0)
        self.assertAlmostEqual(float(baseline["best_scenario_net_edge_bps"]), 2.39 - 4.00, places=2)

        # Front-of-queue Maker (0 bp fee, 0.45 bp adverse selection) unlocks 92 net-positive scenarios
        front = rows["Maker 0bp (Front-of-Queue)"]
        self.assertEqual(int(front["scenarios_net_positive"]), 92)
        self.assertAlmostEqual(float(front["best_scenario_net_edge_bps"]), 2.39 - 0.45, places=2)

        # Verify consistency with oeq_core.turnover_penalized_net_return
        net_ret = turnover_penalized_net_return(gross_return=0.000239, turnover=1.0, cost_bps=0.45)
        self.assertAlmostEqual(net_ret * 10_000.0, 1.94, places=2)

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
            files=[ROOT / "reports" / "maker_queue_fee_sensitivity.csv", ROOT / "reports" / "technical_report.md"],
            evidence_grade=boundary.grade.value,
        )
        self.assertEqual(verify_manifest_checksums(ROOT, manifest), [])


if __name__ == "__main__":
    unittest.main()
