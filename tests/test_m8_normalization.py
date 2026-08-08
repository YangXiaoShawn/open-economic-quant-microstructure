from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from microstructure.data.binance_archive import (
    AcquiredDailyArchive,
    ArchiveDownloadLimits,
    DailyArchiveRequest,
    RawArchiveArtifact,
)
from microstructure.data.quality import ValidationReport
from microstructure.m8_config import M8Period, load_m8_config
from microstructure.m8_manifest import M8SymbolMetadata
from microstructure.m8_normalization import (
    M8InsufficientDataError,
    M8NormalizationError,
    normalize_m8_archive,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    role: str,
    gap_at: int | None = None,
) -> tuple[object, M8Period, M8SymbolMetadata, AcquiredDailyArchive, Path]:
    project_root = Path(__file__).resolve().parents[1]
    config = load_m8_config(project_root / "configs" / "m8_multidate_trade_study.toml")
    study_date = date(2024, 1, 5) if role == "primary_test" else date(2024, 1, 3)
    period = M8Period(date=study_date, role=role)  # type: ignore[arg-type]
    root = tmp_path / "input"
    raw = root / "raw"
    raw.mkdir(parents=True)
    request = DailyArchiveRequest(
        symbol="BTCUSDT",
        date=study_date,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.0001"),
    )
    start_ms = (
        int(datetime(study_date.year, study_date.month, study_date.day, tzinfo=UTC).timestamp())
        * 1_000
    )
    rows: list[str] = []
    for index in range(180):
        aggregate_id = 1000 + index + (1 if gap_at is not None and index >= gap_at else 0)
        price = Decimal("100") + Decimal(index % 17) / Decimal("100")
        rows.append(
            f"{aggregate_id},{price:.2f},0.5000,{2000 + index},{2000 + index},"
            f"{start_ms + index},{'true' if index % 2 else 'false'},true\n"
        )
    archive_path = raw / request.archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(request.member_name, "".join(rows).encode("ascii"))
    archive_sha = _sha(archive_path)
    archive_sidecar = raw / "archive.source.json"
    archive_sidecar.write_text(
        json.dumps({"downloaded_at_utc": "2026-08-07T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    checksum_path = raw / f"{request.archive_name}.CHECKSUM"
    checksum_path.write_text(f"{archive_sha}  {request.archive_name}\n", encoding="ascii")
    checksum_sidecar = raw / "checksum.source.json"
    checksum_sidecar.write_text("{}\n", encoding="utf-8")
    archive_artifact = RawArchiveArtifact(
        kind="archive_zip",
        path=archive_path,
        manifest_path=archive_sidecar,
        sha256=archive_sha,
        manifest_sha256=_sha(archive_sidecar),
        bytes=archive_path.stat().st_size,
        source_uri=(
            f"https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/{request.archive_name}"
        ),
    )
    checksum_artifact = RawArchiveArtifact(
        kind="archive_checksum",
        path=checksum_path,
        manifest_path=checksum_sidecar,
        sha256=_sha(checksum_path),
        manifest_sha256=_sha(checksum_sidecar),
        bytes=checksum_path.stat().st_size,
        source_uri=f"{archive_artifact.source_uri}.CHECKSUM",
    )
    with zipfile.ZipFile(archive_path) as archive:
        expanded = archive.getinfo(request.member_name).file_size
    acquired = AcquiredDailyArchive(
        request=request,
        archive_artifact=archive_artifact,
        checksum_artifact=checksum_artifact,
        upstream_sha256=archive_sha,
        declared_uncompressed_bytes=expanded,
        limits=ArchiveDownloadLimits(
            max_compressed_bytes=10_000_000,
            max_uncompressed_bytes=10_000_000,
        ),
    )
    metadata = M8SymbolMetadata(
        symbol="BTCUSDT",
        status="TRADING",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.0001"),
        observed_ts_ns=1,
        raw_path=raw / "unused-metadata.json",
        raw_sha256="0" * 64,
        raw_bytes=1,
        source_uri="https://api.binance.com/api/v3/exchangeInfo?symbol=BTCUSDT",
        source_manifest_path=raw / "unused-metadata-sidecar.json",
        source_manifest_sha256="1" * 64,
        source_manifest_bytes=1,
    )
    return config, period, metadata, acquired, root


def test_held_out_archive_requires_guard_before_stream_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, period, metadata, acquired, root = _fixture(tmp_path, role="primary_test")
    opened = 0
    original = cast(Any, zipfile.ZipFile.open)

    def spy(self: zipfile.ZipFile, *args: object, **kwargs: object) -> Any:
        nonlocal opened
        opened += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    with pytest.raises(M8NormalizationError, match="lock-revalidation callback"):
        normalize_m8_archive(config, period, metadata, acquired, root)  # type: ignore[arg-type]
    assert opened == 0


def test_primary_test_cannot_be_spoofed_as_train_to_bypass_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, frozen_period, metadata, acquired, root = _fixture(
        tmp_path,
        role="primary_test",
    )
    spoofed_period = M8Period(date=frozen_period.date, role="train")
    opened = 0
    original = cast(Any, zipfile.ZipFile.open)

    def spy(self: zipfile.ZipFile, *args: object, **kwargs: object) -> Any:
        nonlocal opened
        opened += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    with pytest.raises(M8NormalizationError, match="exact frozen configuration entry"):
        normalize_m8_archive(
            config,  # type: ignore[arg-type]
            spoofed_period,
            metadata,
            acquired,
            root,
        )
    assert opened == 0


def test_guard_runs_immediately_before_held_out_member_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, period, metadata, acquired, root = _fixture(tmp_path, role="primary_test")
    events: list[str] = []
    original = cast(Any, zipfile.ZipFile.open)

    def spy(self: zipfile.ZipFile, *args: object, **kwargs: object) -> Any:
        events.append("open")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    result = normalize_m8_archive(
        config,  # type: ignore[arg-type]
        period,
        metadata,
        acquired,
        root,
        before_member_open=lambda: events.append("guard"),
        batch_rows=32,
    )
    assert events == ["guard", "open"]
    assert result.entry.rows == 180
    assert len(result.entry.normalized_parts) >= 1


def test_output_root_isolates_derived_evidence_from_raw_authority(tmp_path: Path) -> None:
    config, period, metadata, acquired, raw_root = _fixture(tmp_path, role="train")
    raw_before = {
        path.relative_to(raw_root).as_posix(): _sha(path)
        for path in raw_root.rglob("*")
        if path.is_file()
    }
    derived_root = tmp_path / "normalized_input"

    result = normalize_m8_archive(
        config,  # type: ignore[arg-type]
        period,
        metadata,
        acquired,
        raw_root,
        output_root=derived_root,
        batch_rows=32,
    )

    assert result.output_root == derived_root.resolve()
    assert result.entry.normalized_dataset_manifest_path.is_relative_to(derived_root.resolve())
    assert result.entry.quality_report_path.is_relative_to(derived_root.resolve())
    assert all(
        part.data_path.is_relative_to(derived_root.resolve())
        and part.sidecar_path.is_relative_to(derived_root.resolve())
        for part in result.entry.normalized_parts
    )
    assert raw_before == {
        path.relative_to(raw_root).as_posix(): _sha(path)
        for path in raw_root.rglob("*")
        if path.is_file()
    }
    assert not (raw_root / "normalized").exists()
    assert not (raw_root / "quality").exists()


def test_guard_failure_propagates_without_open_or_data_reclassification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, period, metadata, acquired, root = _fixture(tmp_path, role="primary_test")
    opened = 0
    original = cast(Any, zipfile.ZipFile.open)

    def spy(self: zipfile.ZipFile, *args: object, **kwargs: object) -> Any:
        nonlocal opened
        opened += 1
        return original(self, *args, **kwargs)

    class GuardFailure(RuntimeError):
        pass

    def rejected() -> None:
        raise GuardFailure("lock changed")

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    with pytest.raises(GuardFailure, match="lock changed"):
        normalize_m8_archive(
            config,  # type: ignore[arg-type]
            period,
            metadata,
            acquired,
            root,
            before_member_open=rejected,
        )
    assert opened == 0


def test_gap_after_lock_is_typed_insufficient_data(tmp_path: Path) -> None:
    config, period, metadata, acquired, root = _fixture(
        tmp_path,
        role="primary_test",
        gap_at=17,
    )
    guarded = False

    def guard() -> None:
        nonlocal guarded
        guarded = True

    with pytest.raises(M8InsufficientDataError, match="noncontiguous") as raised:
        normalize_m8_archive(
            config,  # type: ignore[arg-type]
            period,
            metadata,
            acquired,
            root,
            before_member_open=guard,
            batch_rows=8,
        )
    assert guarded is True
    assert raised.value.failure_kind == "PAYLOAD_OR_CONTINUITY"
    assert raised.value.evidence_completion == "PARTIAL_STREAM"
    assert raised.value.completed_evidence is None


@pytest.mark.parametrize(
    "failure",
    [OSError("disk full"), ValueError("writer implementation failure")],
)
def test_parquet_system_failures_are_not_reclassified_as_insufficient_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    config, period, metadata, acquired, root = _fixture(tmp_path, role="train")

    def fail_write(*_args: object, **_kwargs: object) -> Any:
        raise failure

    import microstructure.m8_normalization as normalization_module

    monkeypatch.setattr(normalization_module, "write_partitioned_parquet", fail_write)
    with pytest.raises(type(failure), match=str(failure)) as raised:
        normalize_m8_archive(
            config,  # type: ignore[arg-type]
            period,
            metadata,
            acquired,
            root,
        )
    assert not isinstance(raised.value, M8InsufficientDataError)


def test_quality_report_permission_failure_is_not_insufficient_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, period, metadata, acquired, root = _fixture(tmp_path, role="train")

    def fail_report_write(self: ValidationReport, path: str | Path) -> None:
        del self, path
        raise PermissionError("quality report is read-only")

    monkeypatch.setattr(ValidationReport, "write_json", fail_report_write)
    with pytest.raises(PermissionError, match="quality report is read-only") as raised:
        normalize_m8_archive(
            config,  # type: ignore[arg-type]
            period,
            metadata,
            acquired,
            root,
            batch_rows=32,
        )
    assert not isinstance(raised.value, M8InsufficientDataError)
