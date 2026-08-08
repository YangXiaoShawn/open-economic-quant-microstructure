from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import microstructure.m8_pipeline as m8_pipeline
import microstructure.research.multidate as multidate
from microstructure.data.storage import write_source_manifest
from microstructure.m8_acquisition import (
    M8AcquisitionError,
    M8AcquisitionManifest,
    M8RawArchiveDescriptor,
    M8RawArchiveEntry,
    M8RawSymbolMetadata,
    M8RetainedArtifact,
)
from microstructure.m8_config import M8StudyConfig, load_m8_config
from microstructure.provenance import sha256_file
from microstructure.reporting import load_run_bundle, verify_checksums, write_checksum_manifest
from microstructure.research.multidate import (
    AnalysisLock,
    FinalFittedState,
    MultiDateEvaluationError,
)


@dataclass(frozen=True)
class _Harness:
    config: M8StudyConfig
    authority: M8AcquisitionManifest
    run_dir: Path
    source_identity: m8_pipeline._SourceIdentity
    member_opens: list[tuple[str, str]]
    stage_manifests: list[M8AcquisitionManifest]


def _write_bytes(path: Path, content: bytes) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path), path.stat().st_size


def _observed_ns(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1e9)


@pytest.mark.parametrize(
    "content",
    [
        b'{"key":1,"key":2}',
        b'{"key":NaN}',
        b'{ "key":1}',
        b"\xff",
    ],
    ids=["duplicate-key", "nan", "noncanonical", "invalid-utf8"],
)
def test_bounded_json_snapshot_rejects_ambiguous_or_noncanonical_input(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "lock.json"
    path.write_bytes(content)
    with pytest.raises(m8_pipeline.M8PipelineError):
        m8_pipeline._read_bounded_json_snapshot(
            path,
            label="fixture lock",
            expected_sha256=hashlib.sha256(content).hexdigest(),
            ensure_ascii=False,
        )


def _metadata(root: Path, symbol: str) -> M8RawSymbolMetadata:
    downloaded = "2026-08-07T12:00:00.000000000Z"
    uri = f"https://data-api.binance.vision/api/v3/exchangeInfo?symbol={symbol}"
    payload = {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "baseAsset": symbol.removesuffix("USDT"),
                "quoteAsset": "USDT",
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01",
                        "maxPrice": "1000000.00",
                        "tickSize": "0.01",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.0001",
                        "maxQty": "100000.0000",
                        "stepSize": "0.0001",
                    },
                ],
            }
        ]
    }
    raw_body = (json.dumps(payload, sort_keys=True) + "\n").encode()
    raw_digest = hashlib.sha256(raw_body).hexdigest()
    raw_path = root / "raw" / "binance_spot" / "exchange_info" / symbol / f"{raw_digest}.json"
    raw_sha, raw_bytes = _write_bytes(
        raw_path,
        raw_body,
    )
    sidecar_path, sidecar_sha = write_source_manifest(
        raw_path,
        source="binance_spot_public_api",
        source_uri=uri,
        downloaded_at_utc=downloaded,
        requested_start_ns=None,
        requested_end_ns=None,
        response_headers={"content-type": "application/json"},
    )
    return M8RawSymbolMetadata(
        venue="binance_spot",
        symbol=symbol,
        status="TRADING",
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.0001"),
        min_price=Decimal("0.01"),
        max_price=Decimal("1000000.00"),
        min_quantity=Decimal("0.0001"),
        max_quantity=Decimal("100000.0000"),
        observed_ts_ns=_observed_ns(downloaded),
        raw_path=raw_path.resolve(),
        raw_sha256=raw_sha,
        raw_bytes=raw_bytes,
        source_uri=uri,
        source_manifest_path=sidecar_path.resolve(),
        source_manifest_sha256=sidecar_sha,
        source_manifest_bytes=sidecar_path.stat().st_size,
    )


def _archive(
    root: Path,
    config: M8StudyConfig,
    *,
    symbol: str,
    study_date: date,
    role: str,
    period_index: int,
    symbol_index: int,
    gap_at: int | None,
    silence_at: int | None,
) -> M8RawArchiveEntry:
    archive_name = f"{symbol}-aggTrades-{study_date.isoformat()}.zip"
    member_name = archive_name.removesuffix(".zip") + ".csv"
    archive_uri = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{archive_name}"
    archive_path = (
        root
        / "raw"
        / "binance_spot"
        / "daily_agg_trades_archive"
        / symbol
        / study_date.isoformat()
        / archive_name
    )
    start_ms = int(
        datetime(study_date.year, study_date.month, study_date.day, tzinfo=UTC).timestamp() * 1_000
    )
    lines: list[str] = []
    first_id = 10_000_000 * (symbol_index + 1) + period_index * 1_000
    for index in range(180):
        aggregate_id = first_id + index + (1 if gap_at is not None and index >= gap_at else 0)
        price = 100.0 + math.sin(index / 6.0 + period_index * 0.4 + symbol_index * 0.7)
        price += 0.12 * math.sin(index / 17.0 + period_index * 0.4)
        event_offset_ms = index // 3 + (
            6_000 if silence_at is not None and index >= silence_at else 0
        )
        lines.append(
            f"{aggregate_id},{price:.2f},{0.5 + (index % 11) * 0.03:.4f},"
            f"{aggregate_id},{aggregate_id},{start_ms + event_offset_ms},"
            f"{'true' if index % 3 else 'false'},true\n"
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, "".join(lines).encode("ascii"))
    archive_sha = sha256_file(archive_path)
    checksum_path = (
        root
        / "raw"
        / "binance_spot"
        / "daily_agg_trades_archive_checksums"
        / symbol
        / study_date.isoformat()
        / f"{archive_name}.CHECKSUM"
    )
    checksum_sha, checksum_bytes = _write_bytes(
        checksum_path,
        f"{archive_sha}  {archive_name}\n".encode("ascii"),
    )
    start_ns = start_ms * 1_000_000
    end_ns = start_ns + 86_400 * 1_000_000_000
    archive_sidecar, archive_sidecar_sha = write_source_manifest(
        archive_path,
        source=config.study.source,
        source_uri=archive_uri,
        downloaded_at_utc="2026-08-07T12:00:00Z",
        requested_start_ns=start_ns,
        requested_end_ns=end_ns,
        upstream_checksum_sha256=archive_sha,
        response_headers={"content-type": "application/zip"},
    )
    checksum_sidecar, checksum_sidecar_sha = write_source_manifest(
        checksum_path,
        source="binance_spot_daily_aggtrades_archive_checksum",
        source_uri=f"{archive_uri}.CHECKSUM",
        downloaded_at_utc="2026-08-07T12:00:00Z",
        requested_start_ns=start_ns,
        requested_end_ns=end_ns,
        response_headers={"content-type": "text/plain"},
    )
    with zipfile.ZipFile(archive_path) as archive:
        expanded = archive.getinfo(member_name).file_size
    return M8RawArchiveEntry(
        root=root.resolve(),
        symbol=symbol,
        date=study_date,
        role=cast(Any, role),
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.0001"),
        archive_path=archive_path.resolve(),
        archive_sha256=archive_sha,
        archive_bytes=archive_path.stat().st_size,
        archive_source_uri=archive_uri,
        archive_source_manifest_path=archive_sidecar.resolve(),
        archive_source_manifest_sha256=archive_sidecar_sha,
        archive_source_manifest_bytes=archive_sidecar.stat().st_size,
        checksum_path=checksum_path.resolve(),
        checksum_sha256=checksum_sha,
        checksum_bytes=checksum_bytes,
        checksum_source_uri=f"{archive_uri}.CHECKSUM",
        checksum_source_manifest_path=checksum_sidecar.resolve(),
        checksum_source_manifest_sha256=checksum_sidecar_sha,
        checksum_source_manifest_bytes=checksum_sidecar.stat().st_size,
        upstream_sha256=archive_sha,
        member_name=member_name,
        declared_uncompressed_bytes=expanded,
        max_compressed_bytes=config.study.max_archive_compressed_bytes,
        max_uncompressed_bytes=config.study.max_archive_uncompressed_bytes,
        max_checksum_bytes=4_096,
        transfer_chunk_bytes=64 * 1_024,
        max_csv_line_bytes=16 * 1_024,
    )


def _retained(
    root: Path, metadata: tuple[M8RawSymbolMetadata, ...], archives: tuple[M8RawArchiveEntry, ...]
) -> tuple[M8RetainedArtifact, ...]:
    rows: list[M8RetainedArtifact] = []

    def add(path: Path, kind: str, uri: str, paired: Path | None = None) -> None:
        rows.append(
            M8RetainedArtifact(
                path=path.resolve().relative_to(root.resolve()).as_posix(),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                kind=cast(Any, kind),
                source_uri=uri,
                paired_body_path=(
                    None
                    if paired is None
                    else paired.resolve().relative_to(root.resolve()).as_posix()
                ),
            )
        )

    for metadata_item in metadata:
        add(metadata_item.raw_path, "metadata_body", metadata_item.source_uri)
        add(
            metadata_item.source_manifest_path,
            "source_manifest",
            metadata_item.source_uri,
            metadata_item.raw_path,
        )
    for archive_item in archives:
        add(archive_item.archive_path, "archive_zip", archive_item.archive_source_uri)
        add(
            archive_item.archive_source_manifest_path,
            "source_manifest",
            archive_item.archive_source_uri,
            archive_item.archive_path,
        )
        add(
            archive_item.checksum_path,
            "archive_checksum",
            archive_item.checksum_source_uri,
        )
        add(
            archive_item.checksum_source_manifest_path,
            "source_manifest",
            archive_item.checksum_source_uri,
            archive_item.checksum_path,
        )
    return tuple(sorted(rows, key=lambda item: item.path))


def _authority(
    tmp_path: Path,
    config: M8StudyConfig,
    *,
    gaps: set[tuple[str, str]] | None = None,
    silences: set[tuple[str, str]] | None = None,
) -> M8AcquisitionManifest:
    root = (tmp_path / "authority").resolve()
    root.mkdir(parents=True)
    metadata = tuple(_metadata(root, symbol) for symbol in config.study.symbols)
    archives = tuple(
        _archive(
            root,
            config,
            symbol=symbol,
            study_date=period.date,
            role=period.role,
            period_index=period_index,
            symbol_index=symbol_index,
            gap_at=(17 if gaps and (symbol, period.date.isoformat()) in gaps else None),
            silence_at=(90 if silences and (symbol, period.date.isoformat()) in silences else None),
        )
        for period_index, period in enumerate(config.periods)
        for symbol_index, symbol in enumerate(config.study.symbols)
    )
    retained = _retained(root, metadata, archives)
    identity = hashlib.sha256(
        "".join(f"{item.path}:{item.sha256}:{item.bytes}\n" for item in retained).encode()
    ).hexdigest()
    body = (json.dumps({"fixture": identity}, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    manifest_path = root / "_manifests" / f"m8-acquisition.manifest-{digest[:20]}.json"
    _write_bytes(manifest_path, body)
    protocol_path = Path(__file__).resolve().parents[1] / "docs" / "M8_MULTIDATE_TRADE_PROTOCOL.md"
    return M8AcquisitionManifest(
        root=root,
        path=manifest_path.resolve(),
        sha256=digest,
        config_sha256=config.hash,
        config_source_sha256=config.source_sha256,
        protocol_version=config.study.protocol_version,
        protocol_document_sha256=sha256_file(protocol_path),
        copied_from_manifest_sha256=None,
        evidence_set_sha256=identity,
        symbol_metadata=metadata,
        archives=archives,
        retained_artifacts=retained,
        total_raw_evidence_bytes=sum(item.bytes for item in retained),
        total_accepted_zip_bytes=sum(item.archive_bytes for item in archives),
        config=config,
    )


def _install_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    authority: M8AcquisitionManifest,
    source_identity: m8_pipeline._SourceIdentity,
    member_opens: list[tuple[str, str]],
    stage_manifests: list[M8AcquisitionManifest],
) -> None:
    by_path: dict[Path, M8AcquisitionManifest] = {authority.path.resolve(): authority}

    def read_manifest(
        path: str | Path,
        *,
        expected_sha256: str,
        config: M8StudyConfig,
    ) -> M8AcquisitionManifest:
        resolved = Path(path).resolve()
        observed = by_path.get(resolved)
        if observed is None:
            for staged in stage_manifests:
                if staged.sha256 != expected_sha256:
                    continue
                destination = resolved.parent.parent

                def move(
                    staged_path: Path,
                    source_root: Path = staged.root,
                    target_root: Path = destination,
                ) -> Path:
                    relative = staged_path.resolve().relative_to(source_root.resolve())
                    return (target_root / relative).resolve()

                observed = replace(
                    staged,
                    root=destination,
                    path=resolved,
                    symbol_metadata=tuple(
                        replace(
                            item,
                            raw_path=move(item.raw_path),
                            source_manifest_path=move(item.source_manifest_path),
                        )
                        for item in staged.symbol_metadata
                    ),
                    archives=tuple(
                        replace(
                            item,
                            root=destination,
                            archive_path=move(item.archive_path),
                            archive_source_manifest_path=move(item.archive_source_manifest_path),
                            checksum_path=move(item.checksum_path),
                            checksum_source_manifest_path=move(item.checksum_source_manifest_path),
                        )
                        for item in staged.archives
                    ),
                )
                by_path[resolved] = observed
                break
        if observed is None:
            raise AssertionError(f"unexpected acquisition manifest path: {resolved}")
        assert observed.sha256 == expected_sha256
        assert observed.config_sha256 == config.hash
        if observed.copied_from_manifest_sha256 is not None:
            expected_files = {item.path for item in observed.retained_artifacts} | {
                observed.path.relative_to(observed.root).as_posix()
            }
            actual_files = {
                item.relative_to(observed.root).as_posix()
                for item in observed.root.rglob("*")
                if item.is_file()
            }
            if actual_files != expected_files:
                raise M8AcquisitionError("unexpected acquisition-root artifact")
        return observed

    def rebase(path: Path, destination: Path) -> Path:
        return (destination / path.resolve().relative_to(authority.root.resolve())).resolve()

    def copy_manifest(
        observed: M8AcquisitionManifest,
        destination_root: str | Path,
    ) -> M8AcquisitionManifest:
        assert observed is authority
        destination = Path(destination_root).resolve()
        shutil.copytree(authority.root, destination)
        copied_authority_path = rebase(authority.path, destination)
        copied_authority_path.unlink()
        copied_body = (
            json.dumps(
                {"fixture": authority.content_identity_sha256, "copied_from": authority.sha256},
                sort_keys=True,
            )
            + "\n"
        ).encode()
        copied_sha = hashlib.sha256(copied_body).hexdigest()
        copied_path = (
            destination / "_manifests" / (f"m8-acquisition.manifest-{copied_sha[:20]}.json")
        )
        _write_bytes(copied_path, copied_body)
        copied_metadata = tuple(
            replace(
                item,
                raw_path=rebase(item.raw_path, destination),
                source_manifest_path=rebase(item.source_manifest_path, destination),
            )
            for item in authority.symbol_metadata
        )
        copied_archives = tuple(
            replace(
                item,
                root=destination,
                archive_path=rebase(item.archive_path, destination),
                archive_source_manifest_path=rebase(item.archive_source_manifest_path, destination),
                checksum_path=rebase(item.checksum_path, destination),
                checksum_source_manifest_path=rebase(
                    item.checksum_source_manifest_path, destination
                ),
            )
            for item in authority.archives
        )
        copied = replace(
            authority,
            root=destination,
            path=copied_path,
            sha256=copied_sha,
            copied_from_manifest_sha256=authority.sha256,
            symbol_metadata=copied_metadata,
            archives=copied_archives,
        )
        by_path[copied.path.resolve()] = copied
        stage_manifests.append(copied)
        return copied

    original_open = cast(Any, zipfile.ZipFile.open)

    def open_spy(self: zipfile.ZipFile, name: object, *args: object, **kwargs: object) -> Any:
        member_name = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        filename = Path(member_name).name
        if filename.endswith(".csv"):
            symbol, _, study_date = filename.removesuffix(".csv").partition("-aggTrades-")
            member_opens.append((symbol, study_date))
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(m8_pipeline, "read_m8_acquisition_manifest", read_manifest)
    monkeypatch.setattr(m8_pipeline, "copy_m8_acquisition_into", copy_manifest)
    monkeypatch.setattr(m8_pipeline, "_capture_source_identity", lambda _root: source_identity)
    monkeypatch.setattr(zipfile.ZipFile, "open", open_spy)


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gaps: set[tuple[str, str]] | None = None,
    silences: set[tuple[str, str]] | None = None,
) -> _Harness:
    project_root = Path(__file__).resolve().parents[1]
    config = load_m8_config(project_root / "configs" / "m8_multidate_trade_study.toml")
    authority = _authority(tmp_path, config, gaps=gaps, silences=silences)
    source = m8_pipeline._SourceIdentity(
        commit="0" * 40,
        dirty=False,
        source_tree_sha256="1" * 64,
    )
    opens: list[tuple[str, str]] = []
    staged: list[M8AcquisitionManifest] = []
    _install_boundaries(monkeypatch, authority, source, opens, staged)
    return _Harness(
        config=config,
        authority=authority,
        run_dir=(tmp_path / "run").resolve(),
        source_identity=source,
        member_opens=opens,
        stage_manifests=staged,
    )


def _refresh_failure_inventory(run_dir: Path, *changed_paths: Path) -> None:
    inventory_path = run_dir / "data" / "failure_evidence_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    by_path = {item["path"]: item for item in inventory}
    for path in changed_paths:
        relative = path.relative_to(run_dir).as_posix()
        by_path[relative]["sha256"] = sha256_file(path)
        by_path[relative]["bytes"] = path.stat().st_size
    m8_pipeline._write_json(inventory_path, inventory)
    write_checksum_manifest(run_dir)


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Harness]:
    tmp_path = tmp_path_factory.mktemp("m8-pipeline-v2")
    patcher = pytest.MonkeyPatch()
    harness = _harness(tmp_path, patcher)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == "COMPLETE"
    assert result.path == harness.run_dir
    try:
        yield harness
    finally:
        patcher.undo()


def test_first_held_out_member_opens_only_after_both_durable_locks(
    completed: _Harness,
) -> None:
    assert completed.member_opens == [
        ("BTCUSDT", "2024-01-03"),
        ("ETHUSDT", "2024-01-03"),
        ("BTCUSDT", "2024-01-04"),
        ("ETHUSDT", "2024-01-04"),
        ("BTCUSDT", "2024-01-05"),
        ("ETHUSDT", "2024-01-05"),
        ("BTCUSDT", "2024-01-06"),
        ("ETHUSDT", "2024-01-06"),
    ]
    aggregate_path = completed.run_dir / "analysis" / "analysis_lock.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["source_identity"] == completed.source_identity.public_dict()
    assert aggregate["raw_acquisition_manifest_sha256"] == completed.authority.sha256
    assert aggregate["development_manifest_sha256"] == sha256_file(
        completed.run_dir / aggregate["development_manifest_path"]
    )
    assert (completed.run_dir / "analysis" / "analysis_lock.sha256").read_text() == (
        f"{sha256_file(aggregate_path)}  analysis_lock.json\n"
    )
    for symbol in completed.config.study.symbols:
        assert (
            completed.run_dir / "analysis" / "locks" / f"{symbol.lower()}.selection_lock.json"
        ).is_file()


def test_final_fitted_state_is_bound_across_complete_bundle_authorities(
    completed: _Harness,
) -> None:
    aggregate = json.loads(
        (completed.run_dir / "analysis" / "analysis_lock.json").read_text(encoding="utf-8")
    )
    claims = [
        {
            "symbol": item["symbol"],
            "path": item["final_fitted_state_path"],
            "sha256": item["final_fitted_state_sha256"],
        }
        for item in aggregate["symbols"]
    ]
    primary_start_ns = int(datetime(2024, 1, 5, tzinfo=UTC).timestamp() * 1_000_000_000)
    for aggregate_symbol, claim in zip(
        aggregate["symbols"],
        claims,
        strict=True,
    ):
        state_path = completed.run_dir / claim["path"]
        state = FinalFittedState.restore(
            state_path.read_text(encoding="utf-8"),
            claim["sha256"],
        )
        child_path = completed.run_dir / aggregate_symbol["selection_lock_path"]
        child = AnalysisLock.restore(
            child_path.read_text(encoding="utf-8"),
            aggregate_symbol["selection_lock_sha256"],
        ).payload()
        assert sha256_file(state_path) == claim["sha256"]
        assert child["final_fitted_state_sha256"] == claim["sha256"]
        assert child["final_fitted_state"] == state.payload()
        assert state.payload()["fit_cutoff_ts_ns"] < primary_start_ns

    provenance = json.loads((completed.run_dir / "provenance.json").read_text())
    research_manifest = json.loads((completed.run_dir / "research" / "manifest.json").read_text())
    run_manifest = json.loads((completed.run_dir / "run_manifest.json").read_text())
    expected_sha_by_symbol = {claim["symbol"]: claim["sha256"] for claim in claims}
    assert provenance["final_fitted_states"] == claims
    assert research_manifest["final_fitted_states"] == claims
    assert run_manifest["research"]["final_fitted_states"] == claims
    assert provenance["run_key_inputs"]["final_fitted_state_sha256_by_symbol"] == (
        expected_sha_by_symbol
    )


def test_no_model_or_calibrator_fit_occurs_after_held_out_authority_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    fit_counts = {"classifier": 0, "calibrator": 0}
    boundary_snapshots: list[tuple[int, int]] = []
    original_classifier = multidate.make_classifier
    original_calibrator_fit = multidate.SigmoidCalibrator.fit
    original_boundary = m8_pipeline._assert_test_open_authority

    def classifier_spy(*args: Any, **kwargs: Any) -> Any:
        fit_counts["classifier"] += 1
        return original_classifier(*args, **kwargs)

    def calibrator_fit_spy(*args: Any, **kwargs: Any) -> Any:
        fit_counts["calibrator"] += 1
        return original_calibrator_fit(*args, **kwargs)

    def boundary_spy(**kwargs: Any) -> None:
        original_boundary(**kwargs)
        for selection in kwargs["selections"]:
            assert sha256_file(selection.fitted_state_path) == (
                selection.selection.fitted_state.sha256
            )
        boundary_snapshots.append((fit_counts["classifier"], fit_counts["calibrator"]))

    monkeypatch.setattr(multidate, "make_classifier", classifier_spy)
    monkeypatch.setattr(multidate.SigmoidCalibrator, "fit", calibrator_fit_spy)
    monkeypatch.setattr(m8_pipeline, "_assert_test_open_authority", boundary_spy)

    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == "COMPLETE"
    assert boundary_snapshots
    assert boundary_snapshots[0][0] > 0
    assert boundary_snapshots[0][1] > 0
    assert all(snapshot == boundary_snapshots[0] for snapshot in boundary_snapshots)
    assert (fit_counts["classifier"], fit_counts["calibrator"]) == boundary_snapshots[0]


def test_complete_bundle_has_final_manifest_then_endpoints_and_narrow_claims(
    completed: _Harness,
) -> None:
    bundle = load_run_bundle(completed.run_dir)
    assert bundle.evidence_tier == "FULL_DATA"
    assert bundle.data["all_requested_ranges_complete"] is True
    assert len(bundle.hypothesis_evaluation["per_date"]) == 4
    assert len(bundle.hypothesis_evaluation["per_symbol"]) == 2
    assert bundle.manifest["execution_assumptions"]["status"] == "NOT_RUN"
    assert bundle.hypothesis_evaluation["p_values_computed"] is False
    assert bundle.hypothesis_evaluation["cross_instrument_conclusion"]["pooling_performed"] is False
    final_path = completed.run_dir / bundle.provenance["m8_input_manifest_path"]
    assert final_path.is_file()
    research = json.loads((completed.run_dir / "research" / "manifest.json").read_text())
    assert research["final_all_date_manifest_committed_before_endpoint_evaluation"] is True
    assert len(research["entries"]) == 8
    predictions = completed.run_dir / "models" / "predictions.parquet"
    assert predictions.is_file()


def test_self_contained_snapshot_has_no_duplicate_raw_evidence(completed: _Harness) -> None:
    snapshot = json.loads(
        (completed.run_dir / "data" / "manifest_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["self_contained_bundle_input"] is True
    assert snapshot["snapshot_created_additional_raw_evidence_copies"] is False
    assert "external" not in json.dumps(snapshot).lower()
    provenance = json.loads((completed.run_dir / "provenance.json").read_text())
    budget = provenance["raw_evidence_byte_budget"]
    assert budget["external_total"] == completed.authority.total_raw_evidence_bytes
    assert budget["bundle_copy_total"] == completed.authority.total_raw_evidence_bytes
    assert budget["combined_total"] == 2 * completed.authority.total_raw_evidence_bytes
    staged = completed.stage_manifests[0]
    physical = sum(
        (completed.run_dir / "data" / "input" / item.path).stat().st_size
        for item in staged.retained_artifacts
    )
    assert physical == staged.total_raw_evidence_bytes
    assert len({item.path for item in staged.retained_artifacts}) == len(staged.retained_artifacts)


def test_complete_reuse_changes_no_bytes_or_member_opens(completed: _Harness) -> None:
    before = {
        path.relative_to(completed.run_dir).as_posix(): sha256_file(path)
        for path in completed.run_dir.rglob("*")
        if path.is_file()
    }
    open_count = len(completed.member_opens)
    result = m8_pipeline.reproduce_m8(
        completed.config,
        completed.run_dir,
        raw_manifest_path=completed.authority.path,
        raw_manifest_sha256=completed.authority.sha256,
    )
    assert result.status == "COMPLETE"
    assert len(completed.member_opens) == open_count
    assert before == {
        path.relative_to(completed.run_dir).as_posix(): sha256_file(path)
        for path in completed.run_dir.rglob("*")
        if path.is_file()
    }


def test_complete_reuse_rejects_mutated_success_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    (result.path / "_SUCCESS").write_bytes(b"tampered terminal bytes\n")
    with pytest.raises(m8_pipeline.M8PipelineError, match="terminal marker bytes"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


@pytest.mark.parametrize(
    "artifact",
    ["aggregate", "digest_sidecar", "child_lock", "fitted_state", "development_manifest"],
)
def test_complete_reuse_rejects_rechecksummed_lock_chain_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    aggregate_path = result.path / "analysis" / "analysis_lock.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if artifact == "aggregate":
        aggregate["source_identity"] = {
            "commit": "f" * 40,
            "dirty": False,
            "source_tree_sha256": "e" * 64,
        }
        aggregate_path.write_text(
            json.dumps(aggregate, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    elif artifact == "digest_sidecar":
        (result.path / "analysis" / "analysis_lock.sha256").write_text(
            f"{'0' * 64}  analysis_lock.json\n",
            encoding="utf-8",
        )
    elif artifact == "child_lock":
        child = result.path / aggregate["symbols"][0]["selection_lock_path"]
        child.write_bytes(child.read_bytes() + b"\n")
    elif artifact == "fitted_state":
        state = result.path / aggregate["symbols"][0]["final_fitted_state_path"]
        state.write_bytes(state.read_bytes() + b"\n")
    else:
        development = result.path / aggregate["development_manifest_path"]
        development.write_bytes(development.read_bytes() + b"\n")

    (result.path / "_SUCCESS").unlink()
    write_checksum_manifest(result.path)
    (result.path / "_SUCCESS").write_bytes(b"complete\n")
    with pytest.raises(m8_pipeline.M8PipelineError, match="completed"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_complete_verifier_bounds_fitted_state_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    aggregate = json.loads(
        (result.path / "analysis" / "analysis_lock.json").read_text(encoding="utf-8")
    )
    state = result.path / aggregate["symbols"][0]["final_fitted_state_path"]
    state.write_bytes(b"x" * (m8_pipeline._MAX_LOCK_JSON_BYTES + 1))
    (result.path / "_SUCCESS").unlink()
    write_checksum_manifest(result.path)
    (result.path / "_SUCCESS").write_bytes(b"complete\n")
    with pytest.raises(m8_pipeline.M8PipelineError, match="hard limit"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_postlock_inventory_binding_rejects_temporary_rechecksummed_lock_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-05")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure_path = result.path / "failure.json"
    provenance_path = result.path / "provenance.json"
    run_manifest_path = result.path / "run_manifest.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    aggregate_path = result.path / failure["analysis_lock_path"]
    sidecar_path = aggregate_path.with_name("analysis_lock.sha256")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["unclaimed_rechecksummed_extension"] = True
    replacement_bytes = json.dumps(
        aggregate,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    replacement_sha = hashlib.sha256(replacement_bytes).hexdigest()
    failure["analysis_lock_sha256"] = replacement_sha
    provenance["selection_lock_sha256"] = replacement_sha
    run_manifest["research"]["analysis_lock"]["sha256"] = replacement_sha
    m8_pipeline._write_json(failure_path, failure)
    m8_pipeline._write_json(provenance_path, provenance)
    m8_pipeline._write_json(run_manifest_path, run_manifest)
    _refresh_failure_inventory(result.path, failure_path, provenance_path, run_manifest_path)

    saved_aggregate = tmp_path / "saved-analysis-lock.json"
    saved_sidecar = tmp_path / "saved-analysis-lock.sha256"
    replacement_aggregate = tmp_path / "replacement-analysis-lock.json"
    replacement_sidecar = tmp_path / "replacement-analysis-lock.sha256"
    shutil.copy2(aggregate_path, saved_aggregate)
    shutil.copy2(sidecar_path, saved_sidecar)
    replacement_aggregate.write_bytes(replacement_bytes)
    replacement_sidecar.write_text(
        f"{replacement_sha}  analysis_lock.json\n",
        encoding="utf-8",
    )
    original_verify_chain = m8_pipeline._verify_insufficient_lock_chain
    swapped = False

    def swap_during_semantic_verification(*args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        replacement_aggregate.replace(aggregate_path)
        replacement_sidecar.replace(sidecar_path)
        swapped = True
        try:
            original_verify_chain(*args, **kwargs)
        finally:
            saved_aggregate.replace(aggregate_path)
            saved_sidecar.replace(sidecar_path)

    monkeypatch.setattr(
        m8_pipeline,
        "_verify_insufficient_lock_chain",
        swap_during_semantic_verification,
    )
    with pytest.raises(m8_pipeline.M8PipelineError, match="failure evidence inventory"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert swapped is True


def test_descriptor_integrity_error_is_terminal_insufficient_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)

    def fail_integrity(_descriptor: M8RawArchiveDescriptor) -> Any:
        raise M8AcquisitionError("fixture raw integrity failure")

    monkeypatch.setattr(M8RawArchiveDescriptor, "reconstruct", fail_integrity)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == "INSUFFICIENT_DATA"
    assert (result.path / "INSUFFICIENT_DATA").read_bytes() == b"terminal\n"
    assert harness.member_opens == []


@pytest.mark.parametrize("fault", ["permission", "wrapped_permission", "assertion"])
def test_descriptor_system_fault_does_not_publish_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)

    def fail_system(_descriptor: M8RawArchiveDescriptor) -> Any:
        if fault == "assertion":
            raise AssertionError("fixture programmer fault")
        if fault == "permission":
            raise PermissionError("fixture permission fault")
        try:
            raise PermissionError("fixture wrapped permission fault")
        except PermissionError as exc:
            raise M8AcquisitionError("cannot verify fixture raw evidence") from exc

    monkeypatch.setattr(M8RawArchiveDescriptor, "reconstruct", fail_system)
    with pytest.raises(m8_pipeline.M8PipelineError, match="failed closed"):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert not harness.run_dir.exists()
    assert not tuple(tmp_path.glob(".run.staging-*"))
    assert harness.member_opens == []


@pytest.mark.parametrize("commit,dirty", [("UNBORN", False), ("0" * 40, True)])
def test_dirty_or_unborn_source_opens_zero_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit: str,
    dirty: bool,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    bad_source = replace(harness.source_identity, commit=commit, dirty=dirty)
    monkeypatch.setattr(m8_pipeline, "_capture_source_identity", lambda _root: bad_source)
    with pytest.raises(m8_pipeline.M8PipelineError):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert harness.member_opens == []


def test_missing_lock_opens_zero_held_out_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original = m8_pipeline._commit_aggregate_lock

    def commit_then_remove(*args: Any, **kwargs: Any) -> tuple[Path, str]:
        path, digest = original(*args, **kwargs)
        path.unlink()
        return path, digest

    monkeypatch.setattr(m8_pipeline, "_commit_aggregate_lock", commit_then_remove)
    with pytest.raises(m8_pipeline.M8PipelineError, match="aggregate lock"):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert all(study_date in {"2024-01-03", "2024-01-04"} for _, study_date in harness.member_opens)


def test_aggregate_path_replacement_during_same_fd_snapshot_opens_zero_held_out_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original_commit = m8_pipeline._commit_aggregate_lock
    original_read = m8_pipeline.os.read
    armed: dict[str, Any] = {}

    def commit_then_arm(*args: Any, **kwargs: Any) -> tuple[Path, str]:
        path, digest = original_commit(*args, **kwargs)
        replacement = path.with_name(".analysis_lock.replacement.json")
        shutil.copy2(path, replacement)
        armed.update(
            {
                "path": path,
                "replacement": replacement,
                "inode": path.stat().st_ino,
                "swapped": False,
            }
        )
        return path, digest

    def replace_after_first_read(descriptor: int, count: int) -> bytes:
        chunk = original_read(descriptor, count)
        if (
            armed
            and not armed["swapped"]
            and m8_pipeline.os.fstat(descriptor).st_ino == armed["inode"]
        ):
            path = cast(Path, armed["path"])
            original = path.with_name(".analysis_lock.original.json")
            path.rename(original)
            cast(Path, armed["replacement"]).rename(path)
            armed["swapped"] = True
        return chunk

    monkeypatch.setattr(m8_pipeline, "_commit_aggregate_lock", commit_then_arm)
    monkeypatch.setattr(m8_pipeline.os, "read", replace_after_first_read)
    with pytest.raises(m8_pipeline.M8PipelineError, match="changed"):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert armed["swapped"] is True
    assert all(study_date in {"2024-01-03", "2024-01-04"} for _, study_date in harness.member_opens)
    assert not harness.run_dir.exists()


@pytest.mark.parametrize(
    "artifact",
    ["aggregate", "digest_sidecar", "child_lock", "fitted_state", "development_manifest"],
)
def test_lock_boundary_rejects_symlink_artifacts_before_held_out_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original = m8_pipeline._commit_aggregate_lock

    def commit_then_symlink(*args: Any, **kwargs: Any) -> tuple[Path, str]:
        path, digest = original(*args, **kwargs)
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        stage = path.parent.parent
        targets = {
            "aggregate": path,
            "digest_sidecar": path.with_name("analysis_lock.sha256"),
            "child_lock": stage / aggregate["symbols"][0]["selection_lock_path"],
            "fitted_state": stage / aggregate["symbols"][0]["final_fitted_state_path"],
            "development_manifest": stage / aggregate["development_manifest_path"],
        }
        target = targets[artifact]
        real = target.with_name(f".{target.name}.real")
        target.rename(real)
        target.symlink_to(real.name)
        return path, digest

    monkeypatch.setattr(m8_pipeline, "_commit_aggregate_lock", commit_then_symlink)
    with pytest.raises(m8_pipeline.M8PipelineError):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert all(study_date in {"2024-01-03", "2024-01-04"} for _, study_date in harness.member_opens)
    assert not harness.run_dir.exists()


@pytest.mark.parametrize(
    "artifact",
    ["aggregate", "digest_sidecar", "child_lock", "fitted_state", "development_manifest"],
)
def test_lock_boundary_rejects_oversized_artifacts_before_held_out_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original = m8_pipeline._commit_aggregate_lock

    def commit_then_oversize(*args: Any, **kwargs: Any) -> tuple[Path, str]:
        path, digest = original(*args, **kwargs)
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        stage = path.parent.parent
        targets = {
            "aggregate": path,
            "digest_sidecar": path.with_name("analysis_lock.sha256"),
            "child_lock": stage / aggregate["symbols"][0]["selection_lock_path"],
            "fitted_state": stage / aggregate["symbols"][0]["final_fitted_state_path"],
            "development_manifest": stage / aggregate["development_manifest_path"],
        }
        target = targets[artifact]
        limit = (
            m8_pipeline._MAX_LOCK_DIGEST_BYTES
            if artifact == "digest_sidecar"
            else m8_pipeline._MAX_LOCK_JSON_BYTES
        )
        target.write_bytes(b"x" * (limit + 1))
        return path, digest

    monkeypatch.setattr(m8_pipeline, "_commit_aggregate_lock", commit_then_oversize)
    with pytest.raises(m8_pipeline.M8PipelineError, match="hard limit"):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert all(study_date in {"2024-01-03", "2024-01-04"} for _, study_date in harness.member_opens)
    assert not harness.run_dir.exists()


def test_same_size_different_stage_authority_opens_zero_held_out_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original = m8_pipeline._commit_aggregate_lock

    def commit_then_tamper(*args: Any, **kwargs: Any) -> tuple[Path, str]:
        result = original(*args, **kwargs)
        staged = harness.stage_manifests[0]
        harness.stage_manifests[0] = replace(staged, evidence_set_sha256="f" * 64)
        # The reader closure resolves by path. Replace its visible object through
        # the pipeline argument as well; exact totals remain unchanged.
        monkeypatch.setattr(
            m8_pipeline,
            "read_m8_acquisition_manifest",
            lambda path, *, expected_sha256, config: (
                harness.stage_manifests[0]
                if Path(path).resolve() == staged.path.resolve()
                else harness.authority
            ),
        )
        return result

    monkeypatch.setattr(m8_pipeline, "_commit_aggregate_lock", commit_then_tamper)
    with pytest.raises(m8_pipeline.M8PipelineError, match="differs"):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert all(study_date in {"2024-01-03", "2024-01-04"} for _, study_date in harness.member_opens)


def test_prelock_gap_publishes_terminal_insufficient_without_held_out_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-03")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == "INSUFFICIENT_DATA"
    assert result.normalized_manifest_sha256 is None
    failure = json.loads((result.path / "failure.json").read_text())
    assert failure["failed_after_analysis_lock"] is False
    assert failure["reason_code"] == "ARCHIVE_PAYLOAD_OR_CONTINUITY"
    assert failure["failure_stage"] == "development_normalization"
    assert failure["failed_role"] == "train"
    assert failure["failed_normalization_evidence"] == {
        "schema_version": "m8-failed-normalization-evidence-v1",
        "failure_kind": "PAYLOAD_OR_CONTINUITY",
        "evidence_completion": "PARTIAL_STREAM",
        "normalized_prefix": "data/normalized_input/normalized/BTCUSDT/2024-01-03",
        "quality_prefix": "data/normalized_input/quality/BTCUSDT/2024-01-03",
        "artifacts": [],
        "complete_normalization": None,
    }
    assert failure["analysis_lock"] is None
    assert failure["analysis_lock_path"] is None
    assert failure["analysis_lock_sha256"] is None
    assert failure["aggregate_lock_committed"] is False
    assert failure["selection_started"] is False
    assert failure["selection_completed_symbols"] == []
    assert failure["endpoint_evaluation_started"] is False
    assert failure["endpoint_evaluation_completed"] is False
    assert failure["endpoint_artifacts_published"] is False
    assert failure["held_out_member_opened"] is False
    assert not (result.path / "analysis" / "analysis_lock.json").exists()
    assert not tuple(result.path.rglob("*prediction*"))
    assert harness.member_opens == [("BTCUSDT", "2024-01-03")]
    assert verify_checksums(result.path) > 0


def test_early_payload_failure_preserves_canonical_partial_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-03")})
    original_normalize = m8_pipeline.normalize_m8_archive

    def normalize_in_small_batches(*args: Any, **kwargs: Any) -> Any:
        kwargs["batch_rows"] = 8
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(m8_pipeline, "normalize_m8_archive", normalize_in_small_batches)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )

    failure = json.loads((result.path / "failure.json").read_text(encoding="utf-8"))
    evidence = failure["failed_normalization_evidence"]
    assert evidence["evidence_completion"] == "PARTIAL_STREAM"
    assert evidence["artifacts"]
    assert all(
        item["path"].endswith(".parquet") or ".manifest-" in item["path"]
        for item in evidence["artifacts"]
    )
    assert not tuple((result.path / evidence["normalized_prefix"] / "_manifests").glob("*.json"))
    assert not (result.path / evidence["quality_prefix"] / "report.json").exists()
    assert (
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        ).status
        == "INSUFFICIENT_DATA"
    )


def test_second_symbol_warning_preserves_separated_evidence_and_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        silences={("ETHUSDT", "2024-01-03")},
    )
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )

    assert result.status == "INSUFFICIENT_DATA"
    assert result.normalized_manifest_sha256 is None
    failure = json.loads((result.path / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "development_normalization"
    assert failure["failed_symbol"] == "ETHUSDT"
    assert failure["failed_date"] == "2024-01-03"
    assert failure["failed_role"] == "train"
    assert failure["reason_code"] == "ARCHIVE_QUALITY_GATE"
    assert failure["held_out_member_opened"] is False
    failed_evidence = failure["failed_normalization_evidence"]
    assert failed_evidence["failure_kind"] == "QUALITY_GATE"
    assert failed_evidence["evidence_completion"] == "COMPLETE_DATASET_AND_QUALITY"
    assert failed_evidence["complete_normalization"]["quality_warnings"] == 1
    assert [
        (item["symbol"], item["date"], item["role"]) for item in failure["completed_normalizations"]
    ] == [("BTCUSDT", "2024-01-03", "train")]
    assert harness.member_opens == [
        ("BTCUSDT", "2024-01-03"),
        ("ETHUSDT", "2024-01-03"),
    ]

    raw_root = result.path / "data" / "input"
    staged = harness.stage_manifests[0]
    expected_raw_files = {item.path for item in staged.retained_artifacts} | {
        staged.path.relative_to(staged.root).as_posix()
    }
    assert {
        path.relative_to(raw_root).as_posix() for path in raw_root.rglob("*") if path.is_file()
    } == expected_raw_files
    assert {path.name for path in raw_root.iterdir()} == {"_manifests", "raw"}
    assert not (raw_root / "normalized").exists()
    assert not (raw_root / "quality").exists()

    derived_root = result.path / "data" / "normalized_input"
    btc_reports = tuple((derived_root / "quality" / "BTCUSDT").rglob("report.json"))
    eth_reports = tuple((derived_root / "quality" / "ETHUSDT").rglob("report.json"))
    assert len(btc_reports) == len(eth_reports) == 1
    assert json.loads(btc_reports[0].read_text(encoding="utf-8"))["summary"] == {
        "errors": 0,
        "warnings": 0,
    }
    assert json.loads(eth_reports[0].read_text(encoding="utf-8"))["summary"] == {
        "errors": 0,
        "warnings": 1,
    }
    assert tuple((derived_root / "normalized" / "BTCUSDT").rglob("*.parquet"))
    assert tuple((derived_root / "normalized" / "ETHUSDT").rglob("*.parquet"))
    inventory = json.loads(
        (result.path / "data" / "failure_evidence_inventory.json").read_text(encoding="utf-8")
    )
    assert any(
        item["path"].startswith("data/normalized_input/quality/ETHUSDT/") for item in inventory
    )

    verified = m8_pipeline.verify_m8_result(
        result.path,
        harness.config,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert verified.status == "INSUFFICIENT_DATA"
    member_opens_before_reuse = len(harness.member_opens)
    reused = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert reused == verified
    assert len(harness.member_opens) == member_opens_before_reuse


def test_quality_failure_cannot_drop_failed_evidence_and_rechecksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        silences={("ETHUSDT", "2024-01-03")},
    )
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    shutil.rmtree(
        result.path / "data" / "normalized_input" / "normalized" / "ETHUSDT" / "2024-01-03"
    )
    shutil.rmtree(result.path / "data" / "normalized_input" / "quality" / "ETHUSDT" / "2024-01-03")
    terminal_exclusions = {
        "data/failure_evidence_inventory.json",
        "checksums.sha256",
        "INSUFFICIENT_DATA",
    }
    rebuilt_inventory = [
        {
            "path": path.relative_to(result.path).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(result.path.rglob("*"))
        if path.is_file() and path.relative_to(result.path).as_posix() not in terminal_exclusions
    ]
    m8_pipeline._write_json(
        result.path / "data" / "failure_evidence_inventory.json",
        rebuilt_inventory,
    )
    write_checksum_manifest(result.path)

    with pytest.raises(m8_pipeline.M8PipelineError, match=r"failed normalization|quality-gate"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_inventory_semantic_read_rejects_post_hash_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        silences={("ETHUSDT", "2024-01-03")},
    )
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    report_relative = "data/normalized_input/quality/BTCUSDT/2024-01-03/report.json"
    original_verify = m8_pipeline._verify_inventory_file
    swapped = False

    def verify_then_swap(root: Path, item: m8_pipeline._FailureEvidenceItem) -> None:
        nonlocal swapped
        original_verify(root, item)
        if item.path == report_relative and not swapped:
            report_path = root / report_relative
            report_path.write_bytes(report_path.read_bytes() + b" ")
            swapped = True

    monkeypatch.setattr(m8_pipeline, "_verify_inventory_file", verify_then_swap)
    with pytest.raises(m8_pipeline.M8PipelineError, match=r"SHA-256|inventory|canonical"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert swapped is True


def test_bundled_raw_retained_artifact_rejects_rechecksummed_inventory_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-03")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    staged = harness.stage_manifests[0]
    retained = staged.retained_artifacts[0]
    retained_path = result.path / "data" / "input" / retained.path
    retained_path.write_bytes(retained_path.read_bytes() + b"\n")
    _refresh_failure_inventory(result.path, retained_path)

    with pytest.raises(m8_pipeline.M8PipelineError, match="failure evidence inventory"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_final_manifest_part_claim_must_match_failure_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original_evaluate = m8_pipeline._evaluate_symbol

    def fail_second(symbol: str, *args: Any, **kwargs: Any) -> Any:
        if symbol == "ETHUSDT":
            raise MultiDateEvaluationError("fixture evaluation insufficiency")
        return original_evaluate(symbol, *args, **kwargs)

    monkeypatch.setattr(m8_pipeline, "_evaluate_symbol", fail_second)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    original_load = m8_pipeline._load_final_input_manifest

    def load_with_forged_part_claim(*args: Any, **kwargs: Any) -> Any:
        manifest = original_load(*args, **kwargs)
        first_entry = manifest.entries[0]
        forged_part = replace(first_entry.normalized_parts[0], data_sha256="f" * 64)
        forged_entry = replace(
            first_entry,
            normalized_parts=(forged_part, *first_entry.normalized_parts[1:]),
        )
        return replace(manifest, entries=(forged_entry, *manifest.entries[1:]))

    monkeypatch.setattr(m8_pipeline, "_load_final_input_manifest", load_with_forged_part_claim)
    with pytest.raises(m8_pipeline.M8PipelineError, match="failure evidence inventory"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_entry",
        "extra_entry",
        "undeclared_file",
        "path_escape",
        "wrong_sha",
        "wrong_bytes",
        "duplicate_path",
    ],
)
def test_failure_inventory_rejects_rechecksummed_physical_or_schema_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        gaps={("ETHUSDT", "2024-01-03")},
    )
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    inventory_path = result.path / "data" / "failure_evidence_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert isinstance(inventory, list) and inventory

    if mutation == "missing_entry":
        inventory.pop(0)
    elif mutation == "extra_entry":
        inventory.append(
            {
                "path": "zzzz-missing-evidence.bin",
                "sha256": "0" * 64,
                "bytes": 1,
            }
        )
    elif mutation == "undeclared_file":
        (result.path / "undeclared-evidence.bin").write_bytes(b"undeclared\n")
    elif mutation == "path_escape":
        inventory[0]["path"] = "../escaped-evidence.bin"
    elif mutation == "wrong_sha":
        inventory[0]["sha256"] = "0" * 64
    elif mutation == "wrong_bytes":
        inventory[0]["bytes"] += 1
    else:
        inventory.append(dict(inventory[0]))
    inventory.sort(key=lambda item: item["path"])
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksum_manifest(result.path)

    with pytest.raises(m8_pipeline.M8PipelineError, match="inventory"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_failure_inventory_rejects_rechecksummed_completed_evidence_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        gaps={("ETHUSDT", "2024-01-03")},
    )
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure_path = result.path / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert len(failure["completed_normalizations"]) == 1
    failure["completed_normalizations"][0]["normalized_dataset_manifest_sha256"] = "0" * 64
    failure_path.write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inventory_path = result.path / "data" / "failure_evidence_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    failure_entry = next(item for item in inventory if item["path"] == "failure.json")
    failure_entry["sha256"] = sha256_file(failure_path)
    failure_entry["bytes"] = failure_path.stat().st_size
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksum_manifest(result.path)

    with pytest.raises(m8_pipeline.M8PipelineError, match=r"completed|normalized"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_producer_rejects_invalid_terminal_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-03")})
    original_publish = m8_pipeline._publish_prelock_insufficient_data

    def publish_then_tamper(*args: Any, **kwargs: Any) -> None:
        original_publish(*args, **kwargs)
        stage = cast(Path, kwargs["stage"])
        inventory_path = stage / "data" / "failure_evidence_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory[0]["sha256"] = "0" * 64
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksum_manifest(stage)

    monkeypatch.setattr(
        m8_pipeline,
        "_publish_prelock_insufficient_data",
        publish_then_tamper,
    )
    with pytest.raises(m8_pipeline.M8PipelineError, match="inventory"):
        m8_pipeline.reproduce_m8(
            harness.config,
            harness.run_dir,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert not harness.run_dir.exists()
    assert not tuple(harness.run_dir.parent.glob(f".{harness.run_dir.name}.staging-*"))


@pytest.mark.parametrize("complete", [False, True])
def test_producer_self_verifies_stage_and_published_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    harness = _harness(
        tmp_path,
        monkeypatch,
        gaps=None if complete else {("BTCUSDT", "2024-01-03")},
    )
    function_name = "_reuse_completed" if complete else "_reuse_insufficient"
    original_reuse = cast(Any, getattr(m8_pipeline, function_name))
    verified_paths: list[Path] = []

    def reuse_spy(target: Path, *args: Any, **kwargs: Any) -> Any:
        verified_paths.append(target.resolve())
        return original_reuse(target, *args, **kwargs)

    monkeypatch.setattr(m8_pipeline, function_name, reuse_spy)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )

    assert result.status == ("COMPLETE" if complete else "INSUFFICIENT_DATA")
    assert len(verified_paths) == 2
    assert verified_paths[0] != harness.run_dir
    assert verified_paths[0].parent == harness.run_dir.parent
    assert verified_paths[1] == harness.run_dir


def test_postlock_gap_preserves_lock_and_stops_before_later_dates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-05")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == "INSUFFICIENT_DATA"
    failure = json.loads((result.path / "failure.json").read_text())
    assert failure["failed_after_analysis_lock"] is True
    assert failure["reason_code"] == "ARCHIVE_PAYLOAD_OR_CONTINUITY"
    assert failure["failure_stage"] == "held_out_normalization"
    assert failure["failed_role"] == "primary_test"
    assert failure["aggregate_lock_committed"] is True
    assert failure["selection_completed_symbols"] == list(harness.config.study.symbols)
    assert failure["endpoint_evaluation_started"] is False
    assert failure["endpoint_evaluation_completed"] is False
    assert failure["endpoint_artifacts_published"] is False
    assert (result.path / failure["analysis_lock_path"]).is_file()
    assert failure["final_all_date_normalized_manifest"] is None
    assert ("ETHUSDT", "2024-01-05") not in harness.member_opens
    assert not tuple(result.path.rglob("*prediction*"))
    before = len(harness.member_opens)
    reused = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert reused.status == "INSUFFICIENT_DATA"
    assert len(harness.member_opens) == before


def test_second_symbol_evaluation_failure_publishes_no_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original = m8_pipeline._evaluate_symbol

    def fail_second(symbol: str, *args: Any, **kwargs: Any) -> Any:
        if symbol == "ETHUSDT":
            raise MultiDateEvaluationError("fixture evaluation insufficiency")
        return original(symbol, *args, **kwargs)

    monkeypatch.setattr(m8_pipeline, "_evaluate_symbol", fail_second)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == "INSUFFICIENT_DATA"
    assert result.normalized_manifest_sha256 is not None
    assert not tuple(result.path.rglob("*prediction*"))
    assert not (result.path / ".endpoint-staging").exists()
    failure = json.loads((result.path / "failure.json").read_text())
    assert failure["final_all_date_normalized_manifest"]["sha256"] == (
        result.normalized_manifest_sha256
    )
    assert failure["reason_code"] == "LOCKED_EVALUATION_INSUFFICIENT"
    assert failure["failure_stage"] == "locked_evaluation"
    assert failure["failed_role"] == "all_test_dates"
    assert failure["endpoint_evaluation_started"] is True
    assert failure["endpoint_evaluation_completed"] is False
    assert failure["endpoint_artifacts_published"] is False
    assert failure["endpoint_evaluation_completed_symbols"] == ["BTCUSDT"]
    assert failure["endpoint_evaluation_completed_symbol_count"] == 1


@pytest.mark.parametrize(
    "artifact",
    [
        "aggregate",
        "digest_sidecar",
        "child_lock",
        "fitted_state",
        "development_manifest",
        "failure_claim",
        "provenance_claim",
    ],
)
def test_postlock_failure_rejects_rechecksummed_lock_chain_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-05")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure_path = result.path / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    aggregate_path = result.path / failure["analysis_lock_path"]
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if artifact == "aggregate":
        aggregate_path.write_bytes(aggregate_path.read_bytes() + b"\n")
    elif artifact == "digest_sidecar":
        (result.path / "analysis" / "analysis_lock.sha256").write_text(
            f"{'0' * 64}  analysis_lock.json\n",
            encoding="utf-8",
        )
    elif artifact == "child_lock":
        child = result.path / aggregate["symbols"][0]["selection_lock_path"]
        child.write_bytes(child.read_bytes() + b"\n")
    elif artifact == "fitted_state":
        state = result.path / aggregate["symbols"][0]["final_fitted_state_path"]
        state.write_bytes(state.read_bytes() + b"\n")
    elif artifact == "development_manifest":
        development = result.path / aggregate["development_manifest_path"]
        development.write_bytes(development.read_bytes() + b"\n")
    elif artifact == "failure_claim":
        failure["analysis_lock_sha256"] = "0" * 64
        failure_path.write_text(
            json.dumps(failure, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        provenance_path = result.path / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["selection_lock_sha256"] = "0" * 64
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    write_checksum_manifest(result.path)
    with pytest.raises(m8_pipeline.M8PipelineError):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    assert not tuple(result.path.rglob("*prediction*"))


def test_insufficient_verifier_bounds_child_lock_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-05")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure = json.loads((result.path / "failure.json").read_text(encoding="utf-8"))
    aggregate = json.loads(
        (result.path / failure["analysis_lock_path"]).read_text(encoding="utf-8")
    )
    child = result.path / aggregate["symbols"][0]["selection_lock_path"]
    child.write_bytes(b"x" * (m8_pipeline._MAX_LOCK_JSON_BYTES + 1))
    _refresh_failure_inventory(result.path, child)
    with pytest.raises(m8_pipeline.M8PipelineError, match=r"hard limit|failure evidence inventory"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason_code", "UNRECOGNIZED"),
        ("failure_stage", "model_selection"),
        ("failed_role", "replication_test"),
    ],
)
def test_failure_rejects_rechecksummed_typed_identity_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-05")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure_path = result.path / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure[field] = value
    failure_path.write_text(
        json.dumps(failure, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_failure_inventory(result.path, failure_path)
    with pytest.raises(m8_pipeline.M8PipelineError, match=r"failure|reason|stage|role"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_prelock_failure_rejects_rechecksummed_aggregate_lock_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-03")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure_path = result.path / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["analysis_lock_sha256"] = "0" * 64
    failure_path.write_text(
        json.dumps(failure, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_failure_inventory(result.path, failure_path)
    with pytest.raises(m8_pipeline.M8PipelineError, match=r"pre-lock.*lock claim"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


@pytest.mark.parametrize("artifact", ["child_lock", "fitted_state"])
def test_partial_selection_failure_preserves_and_verifies_completed_child_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    original = m8_pipeline._select_symbol

    def fail_second(symbol: str, *args: Any, **kwargs: Any) -> Any:
        if symbol == "ETHUSDT":
            raise MultiDateEvaluationError("fixture second-symbol selection insufficiency")
        return original(symbol, *args, **kwargs)

    monkeypatch.setattr(m8_pipeline, "_select_symbol", fail_second)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    failure = json.loads((result.path / "failure.json").read_text(encoding="utf-8"))
    assert failure["failure_stage"] == "model_selection"
    assert failure["selection_started"] is True
    assert failure["selection_completed_symbols"] == ["BTCUSDT"]
    assert failure["selection_completed_symbol_count"] == 1
    assert failure["aggregate_lock_committed"] is False
    assert failure["held_out_member_opened"] is False
    assert len(failure["selection_locks"]) == 1
    child_path = result.path / failure["selection_locks"][0]["path"]
    assert child_path.is_file()
    assert not (result.path / "analysis" / "analysis_lock.json").exists()
    assert not tuple(result.path.rglob("*prediction*"))
    assert (
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        ).status
        == "INSUFFICIENT_DATA"
    )

    if artifact == "child_lock":
        child_path.write_bytes(child_path.read_bytes() + b"\n")
        changed_path = child_path
    else:
        state_path = result.path / failure["final_fitted_states"][0]["path"]
        state_path.write_bytes(state_path.read_bytes() + b"\n")
        changed_path = state_path
    _refresh_failure_inventory(result.path, changed_path)
    with pytest.raises(m8_pipeline.M8PipelineError, match="partial BTCUSDT"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


@pytest.mark.parametrize("complete", [False, True])
def test_terminal_publication_flushes_tree_before_checksums_and_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
) -> None:
    gaps = None if complete else {("BTCUSDT", "2024-01-03")}
    harness = _harness(tmp_path, monkeypatch, gaps=gaps)
    events: list[str] = []
    original_tree = m8_pipeline._fsync_tree
    original_checksums = cast(Any, m8_pipeline).write_checksum_manifest
    original_marker = m8_pipeline._create_terminal_marker

    def flush_tree(path: Path) -> None:
        events.append("tree")
        original_tree(path)

    def checksums(path: str | Path) -> Path:
        assert events[-1] == "tree"
        events.append("checksums")
        return cast(Path, original_checksums(path))

    def marker(path: Path, content: str) -> None:
        assert events[-1] == "checksums"
        events.append("marker")
        original_marker(path, content)

    monkeypatch.setattr(m8_pipeline, "_fsync_tree", flush_tree)
    monkeypatch.setattr(m8_pipeline, "write_checksum_manifest", checksums)
    monkeypatch.setattr(m8_pipeline, "_create_terminal_marker", marker)
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    assert result.status == ("COMPLETE" if complete else "INSUFFICIENT_DATA")
    assert events == ["tree", "checksums", "marker"]


def test_failure_marker_is_exact_and_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, gaps={("BTCUSDT", "2024-01-03")})
    result = m8_pipeline.reproduce_m8(
        harness.config,
        harness.run_dir,
        raw_manifest_path=harness.authority.path,
        raw_manifest_sha256=harness.authority.sha256,
    )
    marker = result.path / "INSUFFICIENT_DATA"
    assert marker.read_bytes() == b"terminal\n"
    marker.write_text("changed\n", encoding="utf-8")
    with pytest.raises(m8_pipeline.M8PipelineError, match="marker bytes"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )
    marker.write_bytes(b"terminal\n")
    (result.path / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    with pytest.raises(m8_pipeline.M8PipelineError, match="conflicting"):
        m8_pipeline.verify_m8_result(
            result.path,
            harness.config,
            raw_manifest_path=harness.authority.path,
            raw_manifest_sha256=harness.authority.sha256,
        )


def test_every_complete_bundle_byte_is_checksum_protected(completed: _Harness) -> None:
    protected = verify_checksums(completed.run_dir)
    actual = [
        path
        for path in completed.run_dir.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "_SUCCESS"}
    ]
    assert protected == len(actual)
