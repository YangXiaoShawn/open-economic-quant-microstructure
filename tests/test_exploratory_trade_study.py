from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from microstructure.exploratory_trade_study import (
    ExploratoryStudyError,
    _checksums,
    _load_config,
    verify_exploratory_run,
)
from microstructure.provenance import sha256_file


def test_exploratory_config_freezes_four_date_roles() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _load_config(root / "configs" / "exploratory_aggtrades_2026-08-05_08.toml")

    assert tuple((item.date.isoformat(), item.role) for item in config.periods) == (
        ("2026-08-05", "train"),
        ("2026-08-06", "validation"),
        ("2026-08-07", "primary_test"),
        ("2026-08-08", "replication_test"),
    )
    assert config.study.evidence_tier == "PUBLIC_ARCHIVE_EXPLORATORY"
    assert config.quality.allow_quality_warnings is True
    assert not any(asdict(config.claims).values())


def test_exploratory_verifier_rejects_external_input_tamper(tmp_path: Path) -> None:
    external = tmp_path / "external.bin"
    external.write_bytes(b"input\n")
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run / "data" / "input_evidence.json").write_text(
        json.dumps(
            {
                "raw_manifest": str(external),
                "raw_manifest_sha256": sha256_file(external),
                "files": [
                    {
                        "absolute_path": str(external),
                        "sha256": sha256_file(external),
                        "bytes": external.stat().st_size,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _checksums(run)
    (run / "_SUCCESS").write_bytes(b"complete\n")

    assert verify_exploratory_run(run)["integrity"] == "verified"
    external.write_bytes(b"tampered\n")
    with pytest.raises(ExploratoryStudyError, match="external input changed"):
        verify_exploratory_run(run)
