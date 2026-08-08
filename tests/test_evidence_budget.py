from __future__ import annotations

from pathlib import Path

import pytest

from microstructure.data.evidence_budget import (
    EvidenceBudgetError,
    EvidenceBudgetExceeded,
    EvidenceBudgetStateError,
    RetainedEvidenceBudget,
)
from microstructure.data.storage import write_source_manifest


def _write_fixture_source_manifest(
    raw_path: Path,
    *,
    budget: RetainedEvidenceBudget | None = None,
) -> Path:
    manifest_path, _ = write_source_manifest(
        raw_path,
        source="fixture",
        source_uri="https://example.test/raw.bin",
        downloaded_at_utc="2026-08-07T00:00:00Z",
        requested_start_ns=1,
        requested_end_ns=2,
        response_headers={"content-type": "application/octet-stream"},
        retained_evidence_budget=budget,
    )
    return manifest_path


def test_counts_preexisting_regular_files_and_honors_exact_boundary(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    (root / "nested").mkdir(parents=True)
    (root / "first.bin").write_bytes(b"abc")
    (root / "nested" / "second.bin").write_bytes(b"de")
    budget = RetainedEvidenceBudget(root, limit_bytes=8)

    assert budget.used_bytes == 5
    assert budget.reserved_bytes == 0
    assert budget.remaining_bytes == 3

    reservation = budget.reserve(3, label="boundary fixture")
    assert budget.remaining_bytes == 0
    reservation.commit()

    assert budget.used_bytes == 8
    assert budget.reserved_bytes == 0
    with pytest.raises(EvidenceBudgetExceeded, match="boundary overflow"):
        budget.reserve(1, label="boundary overflow")


def test_outstanding_reservations_prevent_oversubscription_and_release(tmp_path: Path) -> None:
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=7)
    first = budget.reserve(5)

    with pytest.raises(EvidenceBudgetExceeded):
        budget.reserve(3)

    first.release()
    second = budget.reserve(7)
    second.commit()
    assert budget.used_bytes == 7
    with pytest.raises(EvidenceBudgetStateError, match="already committed"):
        second.release()


def test_context_manager_releases_uncommitted_reservation(tmp_path: Path) -> None:
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=4)

    with pytest.raises(RuntimeError, match="fixture"), budget.reserve(4):
        raise RuntimeError("fixture")

    assert budget.used_bytes == 0
    assert budget.reserved_bytes == 0
    assert budget.remaining_bytes == 4


def test_scan_does_not_follow_or_charge_symlinks(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "large.bin").write_bytes(b"x" * 100)
    root = tmp_path / "raw"
    root.mkdir()
    (root / "link").symlink_to(external, target_is_directory=True)

    budget = RetainedEvidenceBudget(root, limit_bytes=0)

    assert budget.used_bytes == 0
    with pytest.raises(EvidenceBudgetError, match="traverses a symlink"):
        budget.assert_contains(root / "link" / "new.bin")


def test_rejects_preexisting_overage_and_outside_target(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "existing.bin").write_bytes(b"abcd")

    with pytest.raises(EvidenceBudgetExceeded, match="preexisting"):
        RetainedEvidenceBudget(root, limit_bytes=3)

    budget = RetainedEvidenceBudget(root, limit_bytes=4)
    with pytest.raises(EvidenceBudgetError, match="outside budget root"):
        budget.assert_contains(tmp_path / "elsewhere.bin")


def test_source_manifest_charges_exact_bytes_and_reuses_without_double_charge(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_raw = reference_root / "raw.bin"
    reference_raw.write_bytes(b"abc")
    reference_manifest = _write_fixture_source_manifest(reference_raw)
    exact_total = reference_raw.stat().st_size + reference_manifest.stat().st_size

    root = tmp_path / "bounded"
    root.mkdir()
    raw = root / "raw.bin"
    raw.write_bytes(b"abc")
    budget = RetainedEvidenceBudget(root, limit_bytes=exact_total)

    manifest = _write_fixture_source_manifest(raw, budget=budget)
    assert budget.used_bytes == exact_total
    assert budget.remaining_bytes == 0
    assert budget.used_bytes == sum(
        path.stat().st_size for path in root.iterdir() if path.is_file()
    )

    duplicate = _write_fixture_source_manifest(raw, budget=budget)
    assert duplicate == manifest
    assert budget.used_bytes == exact_total


def test_source_manifest_budget_failure_writes_no_sidecar_and_releases_reservation(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_raw = reference_root / "raw.bin"
    reference_raw.write_bytes(b"abc")
    reference_manifest = _write_fixture_source_manifest(reference_raw)
    exact_total = reference_raw.stat().st_size + reference_manifest.stat().st_size

    root = tmp_path / "bounded"
    root.mkdir()
    raw = root / "raw.bin"
    raw.write_bytes(b"abc")
    budget = RetainedEvidenceBudget(root, limit_bytes=exact_total - 1)

    with pytest.raises(EvidenceBudgetExceeded, match="raw source manifest"):
        _write_fixture_source_manifest(raw, budget=budget)

    assert list(root.iterdir()) == [raw]
    assert budget.used_bytes == len(b"abc")
    assert budget.reserved_bytes == 0
