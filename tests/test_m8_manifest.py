from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from microstructure.data.storage import write_partitioned_parquet, write_source_manifest
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.m8_config import M8StudyConfig, load_m8_config
from microstructure.m8_manifest import (
    M8ArchiveEntry,
    M8InputManifest,
    M8ManifestError,
    M8NormalizedPart,
    M8SymbolMetadata,
    read_m8_input_manifest,
    verify_m8_input_manifest,
    write_m8_input_manifest,
)
from microstructure.provenance import sha256_file, write_json

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_multidate_trade_study.toml"
_NS_PER_DAY = 86_400 * 1_000_000_000


@dataclass(slots=True)
class _Fixture:
    config: M8StudyConfig
    root: Path
    symbol_metadata: tuple[M8SymbolMetadata, ...]
    entries: tuple[M8ArchiveEntry, ...]
    manifest: M8InputManifest


def _day_start_ns(day: date) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return (start - epoch).days * _NS_PER_DAY


def _replace_column(table: pa.Table, name: str, values: list[str]) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _entry_artifacts(
    root: Path,
    config: M8StudyConfig,
    *,
    symbol: str,
    day: date,
    role: str,
    seed: int,
    tick_size: Decimal,
    lot_size: Decimal,
) -> M8ArchiveEntry:
    day_text = day.isoformat()
    start_ns = _day_start_ns(day)
    end_ns = start_ns + _NS_PER_DAY
    raw_dir = root / "raw" / symbol / day_text
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{symbol}-aggTrades-{day_text}.zip"
    csv_bytes = (
        "aggregate_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,"
        "buyer_is_maker,best_match\n"
        f"1,100.00,1.0,1,1,{start_ns // 1_000_000 + 1},false,true\n"
        f"2,100.01,2.0,2,2,{start_ns // 1_000_000 + 2},true,true\n"
        f"3,100.02,1.5,3,3,{start_ns // 1_000_000 + 3},false,true\n"
    ).encode()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-aggTrades-{day_text}.csv", csv_bytes)
    zip_sha = sha256_file(zip_path)
    source_uri = (
        "https://data.binance.vision/data/spot/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{day_text}.zip"
    )
    raw_sidecar_path, raw_sidecar_sha = write_source_manifest(
        zip_path,
        source=config.study.source,
        source_uri=source_uri,
        downloaded_at_utc="2026-08-07T12:00:00Z",
        requested_start_ns=start_ns,
        requested_end_ns=end_ns,
        upstream_checksum_sha256=zip_sha,
    )
    checksum_dir = root / "raw" / "checksums" / symbol / day_text
    checksum_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = checksum_dir / f"{zip_path.name}.CHECKSUM"
    checksum_path.write_bytes(f"{zip_sha}  {zip_path.name}\n".encode("ascii"))
    checksum_sha = sha256_file(checksum_path)
    checksum_source_uri = f"{source_uri}.CHECKSUM"
    checksum_sidecar_path, checksum_sidecar_sha = write_source_manifest(
        checksum_path,
        source="binance_spot_daily_aggtrades_archive_checksum",
        source_uri=checksum_source_uri,
        downloaded_at_utc="2026-08-07T11:59:59.123456789Z",
        requested_start_ns=start_ns,
        requested_end_ns=end_ns,
    )

    generated = generate_synthetic_market(
        symbols=(symbol,),
        events_per_symbol=3,
        start_ts_ns=start_ns + 1_000_000,
        seed=seed,
    )
    trades = _replace_column(generated.trades, "venue", ["binance_spot"] * 3)
    trades = _replace_column(trades, "source_artifact_id", [zip_sha] * 3)
    normalized_root = root / "normalized" / symbol / day_text
    dataset = write_partitioned_parquet(
        [trades],
        root=normalized_root,
        dataset="trades",
        schema_name="trades",
        source=config.study.source,
        source_uri=source_uri,
        downloaded_at_utc="2026-08-07T12:00:00Z",
        source_checksum_sha256=zip_sha,
        requested_start_ns=start_ns,
        requested_end_ns=end_ns,
    )
    parts = tuple(
        M8NormalizedPart(
            data_path=artifact.data_path,
            data_sha256=artifact.data_sha256,
            data_bytes=artifact.data_path.stat().st_size,
            sidecar_path=artifact.manifest_path,
            sidecar_sha256=artifact.manifest_sha256,
            sidecar_bytes=artifact.manifest_path.stat().st_size,
            rows=artifact.rows,
            write_ordinal=artifact.write_ordinal,
            observed_start_ns=artifact.observed_start_ns,
            observed_end_inclusive_ns=artifact.observed_end_inclusive_ns,
        )
        for artifact in dataset.artifacts
    )

    quality_dir = root / "quality" / symbol / day_text
    quality_dir.mkdir(parents=True, exist_ok=True)
    findings_path = quality_dir / "findings.jsonl"
    findings_path.write_bytes(b"")
    report_path = quality_dir / "validation.json"
    write_json(
        report_path,
        {
            "generated_at_utc": "2026-08-07T12:00:00Z",
            "dataset": "trades",
            "rows_checked": 3,
            "summary": {"errors": 0, "warnings": 0},
            "findings": [],
            "mutation_policy": "observations were not changed or repaired",
            "findings_jsonl_path": str(findings_path.relative_to(root)),
        },
    )
    observed_start = min(part.observed_start_ns for part in parts)
    observed_end = max(part.observed_end_inclusive_ns for part in parts)
    return M8ArchiveEntry(
        symbol=symbol,
        date=day,
        role=role,  # type: ignore[arg-type]
        complete=True,
        rows=3,
        first_trade_id=1,
        last_trade_id=3,
        observed_start_ns=observed_start,
        observed_end_inclusive_ns=observed_end,
        tick_size=tick_size,
        lot_size=lot_size,
        raw_zip_path=zip_path,
        raw_zip_sha256=zip_sha,
        raw_zip_bytes=zip_path.stat().st_size,
        raw_uncompressed_bytes=len(csv_bytes),
        raw_source_uri=source_uri,
        raw_source_manifest_path=raw_sidecar_path,
        raw_source_manifest_sha256=raw_sidecar_sha,
        raw_source_manifest_bytes=raw_sidecar_path.stat().st_size,
        raw_checksum_path=checksum_path,
        raw_checksum_sha256=checksum_sha,
        raw_checksum_bytes=checksum_path.stat().st_size,
        raw_checksum_source_uri=checksum_source_uri,
        raw_checksum_source_manifest_path=checksum_sidecar_path,
        raw_checksum_source_manifest_sha256=checksum_sidecar_sha,
        raw_checksum_source_manifest_bytes=checksum_sidecar_path.stat().st_size,
        normalized_dataset_manifest_path=dataset.manifest_path,
        normalized_dataset_manifest_sha256=dataset.manifest_sha256,
        normalized_dataset_manifest_bytes=dataset.manifest_path.stat().st_size,
        normalized_parts=parts,
        quality_report_path=report_path,
        quality_report_sha256=sha256_file(report_path),
        quality_report_bytes=report_path.stat().st_size,
        quality_findings_path=findings_path,
        quality_findings_sha256=sha256_file(findings_path),
        quality_findings_bytes=0,
        quality_errors=0,
        quality_warnings=0,
    )


def _build_symbol_metadata(root: Path, config: M8StudyConfig) -> tuple[M8SymbolMetadata, ...]:
    scales = {
        "BTCUSDT": (Decimal("0.01000000"), Decimal("0.00001000")),
        "ETHUSDT": (Decimal("0.01000000"), Decimal("0.00010000")),
    }
    observed_ts_ns = _day_start_ns(date(2026, 8, 7)) + 43_200_123_456_789
    metadata: list[M8SymbolMetadata] = []
    for symbol in config.study.symbols:
        tick_size, lot_size = scales[symbol]
        raw_payload = {
            "timezone": "UTC",
            "serverTime": 1_786_104_000_123,
            "symbols": [
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "filters": [
                        {
                            "filterType": "PRICE_FILTER",
                            "minPrice": "0.00000001",
                            "maxPrice": "1000000.00000000",
                            "tickSize": format(tick_size, "f"),
                        },
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": format(lot_size, "f"),
                            "maxQty": "100000.00000000",
                            "stepSize": format(lot_size, "f"),
                        },
                    ],
                }
            ],
        }
        raw_bytes = json.dumps(
            raw_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        raw_dir = root / "raw" / "binance_spot" / "exchange_info" / symbol
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{raw_sha}.json"
        raw_path.write_bytes(raw_bytes)
        source_uri = f"https://data-api.binance.vision/api/v3/exchangeInfo?symbol={symbol}"
        source_manifest_path, source_manifest_sha = write_source_manifest(
            raw_path,
            source="binance_spot_public_api",
            source_uri=source_uri,
            downloaded_at_utc="2026-08-07T12:00:00.123456789Z",
            requested_start_ns=None,
            requested_end_ns=None,
            response_headers={"content-type": "application/json"},
        )
        metadata.append(
            M8SymbolMetadata(
                symbol=symbol,
                status="TRADING",
                tick_size=tick_size,
                lot_size=lot_size,
                observed_ts_ns=observed_ts_ns,
                raw_path=raw_path,
                raw_sha256=raw_sha,
                raw_bytes=len(raw_bytes),
                source_uri=source_uri,
                source_manifest_path=source_manifest_path,
                source_manifest_sha256=source_manifest_sha,
                source_manifest_bytes=source_manifest_path.stat().st_size,
            )
        )
    return tuple(metadata)


def _build_entries(
    root: Path,
    config: M8StudyConfig,
    symbol_metadata: tuple[M8SymbolMetadata, ...],
) -> tuple[M8ArchiveEntry, ...]:
    metadata_by_symbol = {metadata.symbol: metadata for metadata in symbol_metadata}
    entries: list[M8ArchiveEntry] = []
    seed = 100
    for period in config.periods:
        for symbol in config.study.symbols:
            scale = metadata_by_symbol[symbol]
            entries.append(
                _entry_artifacts(
                    root,
                    config,
                    symbol=symbol,
                    day=period.date,
                    role=period.role,
                    seed=seed,
                    tick_size=scale.tick_size,
                    lot_size=scale.lot_size,
                )
            )
            seed += 1
    return tuple(entries)


@pytest.fixture
def manifest_fixture(tmp_path: Path) -> _Fixture:
    config = load_m8_config(CONFIG_PATH)
    root = tmp_path / "m8-input"
    root.mkdir()
    symbol_metadata = _build_symbol_metadata(root, config)
    entries = _build_entries(root, config, symbol_metadata)
    manifest = write_m8_input_manifest(
        config,
        root,
        tuple(reversed(entries)),
        tuple(reversed(symbol_metadata)),
    )
    return _Fixture(
        config=config,
        root=root,
        symbol_metadata=symbol_metadata,
        entries=entries,
        manifest=manifest,
    )


def _write_readdressed_manifest(root: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path = root / "_manifests" / f"m8-input.manifest-{digest[:20]}.json"
    path.write_bytes(encoded)
    return path, digest


def _top_payload(fixture: _Fixture) -> dict[str, Any]:
    value = json.loads(fixture.manifest.path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rebind_symbol_metadata_artifacts(
    fixture: _Fixture,
    payload: dict[str, Any],
    *,
    index: int = 0,
    raw_payload: dict[str, Any] | None = None,
    sidecar_payload: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    metadata = payload["symbol_metadata"][index]
    raw_path = fixture.root / metadata["raw_path"]
    if raw_payload is not None:
        write_json(raw_path, raw_payload)
        metadata["raw_sha256"] = sha256_file(raw_path)
        metadata["raw_bytes"] = raw_path.stat().st_size

    sidecar_path = fixture.root / metadata["source_manifest_path"]
    if sidecar_payload is None:
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        sidecar_payload = loaded
    if raw_payload is not None:
        sidecar_payload["checksum"]["value"] = metadata["raw_sha256"]
        sidecar_payload["bytes"] = metadata["raw_bytes"]
    write_json(sidecar_path, sidecar_payload)
    metadata["source_manifest_sha256"] = sha256_file(sidecar_path)
    metadata["source_manifest_bytes"] = sidecar_path.stat().st_size
    payload["total_symbol_metadata_bytes"] = sum(
        item["raw_bytes"] for item in payload["symbol_metadata"]
    )
    return _write_readdressed_manifest(fixture.root, payload)


def _rebind_archive_checksum_artifacts(
    fixture: _Fixture,
    payload: dict[str, Any],
    *,
    index: int = 0,
    checksum_body: bytes | None = None,
    sidecar_payload: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    checksum = payload["entries"][index]["raw"]["checksum"]
    checksum_path = fixture.root / checksum["path"]
    if checksum_body is not None:
        checksum_path.write_bytes(checksum_body)
        checksum["sha256"] = sha256_file(checksum_path)
        checksum["bytes"] = checksum_path.stat().st_size

    sidecar_path = fixture.root / checksum["source_manifest_path"]
    if sidecar_payload is None:
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        sidecar_payload = loaded
    if checksum_body is not None:
        sidecar_payload["checksum"]["value"] = checksum["sha256"]
        sidecar_payload["bytes"] = checksum["bytes"]
    write_json(sidecar_path, sidecar_payload)
    checksum["source_manifest_sha256"] = sha256_file(sidecar_path)
    checksum["source_manifest_bytes"] = sidecar_path.stat().st_size
    return _write_readdressed_manifest(fixture.root, payload)


def _rebind_raw_zip_bytes_for_preflight(
    fixture: _Fixture,
    payload: dict[str, Any],
    content: bytes,
    *,
    index: int = 0,
) -> tuple[Path, str]:
    raw = payload["entries"][index]["raw"]
    zip_path = fixture.root / raw["zip_path"]
    zip_path.write_bytes(content)
    raw["zip_sha256"] = sha256_file(zip_path)
    raw["zip_bytes"] = zip_path.stat().st_size
    payload["total_raw_zip_bytes"] = sum(entry["raw"]["zip_bytes"] for entry in payload["entries"])
    return _write_readdressed_manifest(fixture.root, payload)


def test_manifest_round_trip_is_deterministic_complete_and_ordered(
    manifest_fixture: _Fixture,
) -> None:
    fixture = manifest_fixture
    manifest = fixture.manifest

    assert manifest.path.name == f"m8-input.manifest-{manifest.sha256[:20]}.json"
    assert sha256_file(manifest.path) == manifest.sha256
    assert manifest.config_sha256 == fixture.config.hash
    assert manifest.config_source_sha256 == fixture.config.source_sha256
    assert manifest.protocol_version == "1.0.2"
    assert [metadata.symbol for metadata in manifest.symbol_metadata] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    assert manifest.metadata_for("BTCUSDT").tick_size == Decimal("0.01000000")
    assert len(manifest.entries) == 8
    assert [(entry.date.isoformat(), entry.symbol, entry.role) for entry in manifest.entries] == [
        (period.date.isoformat(), symbol, period.role)
        for period in fixture.config.periods
        for symbol in fixture.config.study.symbols
    ]
    first = manifest.entries[0]
    assert first.raw_checksum_source_uri == f"{first.raw_source_uri}.CHECKSUM"
    assert first.raw_checksum_path.read_bytes() == (
        f"{first.raw_zip_sha256}  {first.raw_zip_path.name}\n".encode("ascii")
    )
    assert manifest.part_paths_for(first.symbol, first.date) == tuple(
        part.data_path for part in first.normalized_parts
    )
    assert next(iter(manifest.ordered_part_paths)) == ("BTCUSDT", "2024-01-03")

    repeated = write_m8_input_manifest(
        fixture.config,
        fixture.root,
        fixture.entries,
        fixture.symbol_metadata,
    )
    assert repeated.path == manifest.path
    assert repeated.sha256 == manifest.sha256
    assert len(list((fixture.root / "_manifests").glob("m8-input.manifest-*.json"))) == 1


def test_reader_verifies_only_parquet_footer_not_trade_rows(
    manifest_fixture: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_rows_are_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("manifest verification must not read Parquet rows")

    monkeypatch.setattr(pq, "read_table", fail_if_rows_are_read)

    verified = verify_m8_input_manifest(
        manifest_fixture.config,
        manifest_fixture.root,
        manifest_fixture.manifest.path,
        manifest_sha256=manifest_fixture.manifest.sha256,
    )

    assert len(verified.entries) == 8


@pytest.mark.parametrize(
    "case",
    [
        "multiple_members",
        "zip64_eocd",
        "zip64_entry",
        "oversized_directory",
        "malformed_directory_bounds",
    ],
)
def test_reader_rejects_unsafe_zip_directory_before_zipfile_parser(
    manifest_fixture: _Fixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    payload = _top_payload(manifest_fixture)
    entry = manifest_fixture.manifest.entries[0]
    content = bytearray(entry.raw_zip_path.read_bytes())
    eocd = content.rfind(b"PK\x05\x06")
    assert eocd >= 0
    central_offset = struct.unpack_from("<L", content, eocd + 16)[0]
    assert content[central_offset : central_offset + 4] == b"PK\x01\x02"

    if case == "multiple_members":
        struct.pack_into("<H", content, eocd + 8, 2)
        struct.pack_into("<H", content, eocd + 10, 2)
    elif case == "zip64_eocd":
        struct.pack_into("<H", content, eocd + 10, 0xFFFF)
    elif case == "zip64_entry":
        struct.pack_into("<L", content, central_offset + 20, 0xFFFFFFFF)
    elif case == "oversized_directory":
        struct.pack_into("<L", content, eocd + 12, 256 * 1_024 + 1)
    else:
        struct.pack_into("<L", content, eocd + 16, central_offset + 1)

    path, manifest_sha = _rebind_raw_zip_bytes_for_preflight(
        manifest_fixture,
        payload,
        bytes(content),
    )
    parser_called = False

    def unexpected_zipfile_parser(*_args: object, **_kwargs: object) -> None:
        nonlocal parser_called
        parser_called = True
        raise AssertionError("unsafe ZIP metadata reached ZipFile")

    import microstructure.m8_manifest as manifest_module

    monkeypatch.setattr(manifest_module, "ZipFile", unexpected_zipfile_parser)
    with pytest.raises(M8ManifestError, match=r"ZIP64|central|directory"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=manifest_sha,
        )
    assert parser_called is False


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra", "misrole"])
def test_writer_rejects_missing_duplicate_extra_and_misrole(tmp_path: Path, case: str) -> None:
    config = load_m8_config(CONFIG_PATH)
    root = tmp_path / "m8-input"
    root.mkdir()
    symbol_metadata = _build_symbol_metadata(root, config)
    entries = list(_build_entries(root, config, symbol_metadata))
    if case == "missing":
        entries.pop()
        message = "missing M8 archive entries"
    elif case == "duplicate":
        entries[1] = entries[0]
        message = "duplicate M8 archive entry"
    elif case == "extra":
        entries.append(replace(entries[0], date=date(2024, 1, 7)))
        message = "extra M8 archive entry"
    else:
        entries[0] = replace(entries[0], role="validation")
        message = "role mismatch"

    with pytest.raises(M8ManifestError, match=message):
        write_m8_input_manifest(config, root, entries, symbol_metadata)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("incomplete", "not a complete full-day archive"),
        ("zero_rows", "rows must be at least 1"),
        ("id_gap", "IDs are not a contiguous row count"),
        ("zero_tick", "tick_size must be finite and positive"),
        ("negative_lot", "lot_size must be finite and positive"),
    ],
)
def test_writer_rejects_invalid_complete_id_and_scale_claims(
    tmp_path: Path, case: str, message: str
) -> None:
    config = load_m8_config(CONFIG_PATH)
    root = tmp_path / "m8-input"
    root.mkdir()
    symbol_metadata = _build_symbol_metadata(root, config)
    entries = list(_build_entries(root, config, symbol_metadata))
    if case == "incomplete":
        entries[0] = replace(entries[0], complete=False)
    elif case == "zero_rows":
        entries[0] = replace(entries[0], rows=0)
    elif case == "id_gap":
        entries[0] = replace(entries[0], last_trade_id=4)
    elif case == "zero_tick":
        entries[0] = replace(entries[0], tick_size=Decimal("0"))
    else:
        entries[0] = replace(entries[0], lot_size=Decimal("-0.1"))

    with pytest.raises(M8ManifestError, match=message):
        write_m8_input_manifest(config, root, entries, symbol_metadata)


def test_writer_rejects_event_bounds_outside_declared_day(tmp_path: Path) -> None:
    config = load_m8_config(CONFIG_PATH)
    root = tmp_path / "m8-input"
    root.mkdir()
    symbol_metadata = _build_symbol_metadata(root, config)
    entries = list(_build_entries(root, config, symbol_metadata))
    entries[0] = replace(entries[0], observed_start_ns=_day_start_ns(entries[0].date) - 1)

    with pytest.raises(M8ManifestError, match="outside its declared UTC day"):
        write_m8_input_manifest(config, root, entries, symbol_metadata)


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra"])
def test_writer_rejects_incomplete_or_ambiguous_symbol_metadata(tmp_path: Path, case: str) -> None:
    config = load_m8_config(CONFIG_PATH)
    root = tmp_path / "m8-input"
    root.mkdir()
    symbol_metadata = list(_build_symbol_metadata(root, config))
    entries = _build_entries(root, config, tuple(symbol_metadata))
    if case == "missing":
        symbol_metadata.pop()
        message = "missing M8 symbol metadata"
    elif case == "duplicate":
        symbol_metadata[1] = symbol_metadata[0]
        message = "duplicate M8 symbol metadata"
    else:
        symbol_metadata.append(replace(symbol_metadata[0], symbol="SOLUSDT"))
        message = "extra M8 symbol metadata"

    with pytest.raises(M8ManifestError, match=message):
        write_m8_input_manifest(config, root, entries, symbol_metadata)


def test_writer_binds_every_entry_scale_to_exchange_info(tmp_path: Path) -> None:
    config = load_m8_config(CONFIG_PATH)
    root = tmp_path / "m8-input"
    root.mkdir()
    symbol_metadata = _build_symbol_metadata(root, config)
    entries = list(_build_entries(root, config, symbol_metadata))
    entries[0] = replace(entries[0], tick_size=Decimal("0.02"))

    with pytest.raises(M8ManifestError, match="scales do not match verified exchangeInfo"):
        write_m8_input_manifest(config, root, entries, symbol_metadata)


@pytest.mark.parametrize(
    "source_uri",
    [
        "http://data-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT",
        "https://evil.example/api/v3/exchangeInfo?symbol=BTCUSDT",
        ("https://data-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT&symbol=BTCUSDT"),
        ("https://data-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT&limit=1"),
        "https://\ndata-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT",
        "https://data-api.binance.vision:/api/v3/exchangeInfo?symbol=BTCUSDT",
        "https://[data-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT",
    ],
)
def test_writer_rejects_nonexact_exchange_info_uri(
    manifest_fixture: _Fixture, source_uri: str
) -> None:
    symbol_metadata = list(manifest_fixture.symbol_metadata)
    symbol_metadata[0] = replace(symbol_metadata[0], source_uri=source_uri)

    with pytest.raises(
        M8ManifestError,
        match=r"exact official exchangeInfo request|invalid network location",
    ):
        write_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            manifest_fixture.entries,
            symbol_metadata,
        )


@pytest.mark.parametrize(
    "artifact",
    [
        "raw_zip",
        "raw_sidecar",
        "raw_checksum",
        "raw_checksum_sidecar",
        "dataset_manifest",
        "part",
        "part_sidecar",
        "quality_report",
        "quality_findings",
    ],
)
def test_reader_rejects_any_artifact_tamper(manifest_fixture: _Fixture, artifact: str) -> None:
    entry = manifest_fixture.manifest.entries[0]
    paths = {
        "raw_zip": entry.raw_zip_path,
        "raw_sidecar": entry.raw_source_manifest_path,
        "raw_checksum": entry.raw_checksum_path,
        "raw_checksum_sidecar": entry.raw_checksum_source_manifest_path,
        "dataset_manifest": entry.normalized_dataset_manifest_path,
        "part": entry.normalized_parts[0].data_path,
        "part_sidecar": entry.normalized_parts[0].sidecar_path,
        "quality_report": entry.quality_report_path,
        "quality_findings": entry.quality_findings_path,
    }
    with paths[artifact].open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(M8ManifestError, match=r"byte count does not match|SHA-256 mismatch"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            manifest_fixture.manifest.path,
            manifest_sha256=manifest_fixture.manifest.sha256,
        )


@pytest.mark.parametrize(
    "body_kind",
    [
        "wrong_digest",
        "uppercase_digest",
        "wrong_basename",
        "carriage_return_only",
        "extra_line",
    ],
)
def test_reader_rejects_rebound_official_checksum_body_forgery(
    manifest_fixture: _Fixture, body_kind: str
) -> None:
    payload = _top_payload(manifest_fixture)
    entry = manifest_fixture.manifest.entries[0]
    digest = entry.raw_zip_sha256
    basename = entry.raw_zip_path.name
    if body_kind == "wrong_digest":
        body = f"{'0' * 64}  {basename}\n".encode("ascii")
    elif body_kind == "uppercase_digest":
        body = f"{digest.upper()}  {basename}\n".encode("ascii")
    elif body_kind == "wrong_basename":
        body = f"{digest}  ETHUSDT-aggTrades-2024-01-03.zip\n".encode("ascii")
    elif body_kind == "carriage_return_only":
        body = f"{digest}  {basename}\r".encode("ascii")
    else:
        body = f"{digest}  {basename}\nextra\n".encode("ascii")
    path, manifest_sha = _rebind_archive_checksum_artifacts(
        manifest_fixture,
        payload,
        checksum_body=body,
    )

    with pytest.raises(M8ManifestError, match="body must be exactly one ZIP SHA-256"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=manifest_sha,
        )


@pytest.mark.parametrize("line_ending", [b"", b"\n", b"\r\n"])
def test_reader_preserves_and_accepts_official_single_line_checksum_endings(
    manifest_fixture: _Fixture, line_ending: bytes
) -> None:
    payload = _top_payload(manifest_fixture)
    entry = manifest_fixture.manifest.entries[0]
    checksum_body = (
        f"{entry.raw_zip_sha256}  {entry.raw_zip_path.name}".encode("ascii") + line_ending
    )
    path, manifest_sha = _rebind_archive_checksum_artifacts(
        manifest_fixture,
        payload,
        checksum_body=checksum_body,
    )

    verified = read_m8_input_manifest(
        manifest_fixture.config,
        manifest_fixture.root,
        path,
        manifest_sha256=manifest_sha,
    )

    assert verified.entries[0].raw_checksum_path.read_bytes() == checksum_body


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("source", "source sidecar source claim does not match"),
        ("uri", "source sidecar source_uri claim does not match"),
        ("range", "does not claim the full UTC day"),
        ("checksum", "source sidecar checksum claim does not match"),
        ("upstream", "must not claim an upstream checksum"),
        ("path", "source sidecar path claim does not match"),
    ],
)
def test_reader_rejects_rebound_official_checksum_sidecar_forgery(
    manifest_fixture: _Fixture, case: str, message: str
) -> None:
    payload = _top_payload(manifest_fixture)
    entry = manifest_fixture.manifest.entries[0]
    sidecar = json.loads(entry.raw_checksum_source_manifest_path.read_text(encoding="utf-8"))
    if case == "source":
        sidecar["source"] = "untrusted_source"
    elif case == "uri":
        sidecar["source_uri"] = entry.raw_source_uri
    elif case == "range":
        sidecar["requested_range_ns"]["end_exclusive"] -= 1
    elif case == "checksum":
        sidecar["checksum"]["value"] = "0" * 64
    elif case == "upstream":
        sidecar["upstream_checksum_sha256"] = entry.raw_zip_sha256
    else:
        sidecar["path"] = entry.raw_zip_path.name
    path, manifest_sha = _rebind_archive_checksum_artifacts(
        manifest_fixture,
        payload,
        sidecar_payload=sidecar,
    )

    with pytest.raises(M8ManifestError, match=message):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=manifest_sha,
        )


def test_writer_rejects_checksum_uri_not_bound_to_zip(manifest_fixture: _Fixture) -> None:
    entries = list(manifest_fixture.entries)
    entries[0] = replace(
        entries[0],
        raw_checksum_source_uri=(
            "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
            "BTCUSDT-aggTrades-2024-01-04.zip.CHECKSUM"
        ),
    )

    with pytest.raises(M8ManifestError, match="URI is not bound to its ZIP URI"):
        write_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            entries,
            manifest_fixture.symbol_metadata,
        )


@pytest.mark.parametrize(
    "source_uri",
    [
        (
            "https://\ndata.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
            "BTCUSDT-aggTrades-2024-01-03.zip"
        ),
        (
            "https://data.binance.vision:/data/spot/daily/aggTrades/BTCUSDT/"
            "BTCUSDT-aggTrades-2024-01-03.zip"
        ),
        (
            "https://[data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
            "BTCUSDT-aggTrades-2024-01-03.zip"
        ),
    ],
)
def test_writer_rejects_noncanonical_archive_uri(
    manifest_fixture: _Fixture, source_uri: str
) -> None:
    entries = list(manifest_fixture.entries)
    entries[0] = replace(entries[0], raw_source_uri=source_uri)

    with pytest.raises(
        M8ManifestError,
        match=r"exact official daily archive URI|invalid network location",
    ):
        write_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            entries,
            manifest_fixture.symbol_metadata,
        )


def test_writer_enforces_official_checksum_byte_ceiling(manifest_fixture: _Fixture) -> None:
    entries = list(manifest_fixture.entries)
    entries[0] = replace(entries[0], raw_checksum_bytes=4_097)

    with pytest.raises(M8ManifestError, match="4096-byte checksum limit"):
        write_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            entries,
            manifest_fixture.symbol_metadata,
        )


@pytest.mark.parametrize("artifact", ["raw", "source_sidecar"])
def test_reader_rejects_exchange_info_artifact_tamper(
    manifest_fixture: _Fixture, artifact: str
) -> None:
    metadata = manifest_fixture.manifest.symbol_metadata[0]
    path = metadata.raw_path if artifact == "raw" else metadata.source_manifest_path
    with path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(M8ManifestError, match=r"byte count does not match|SHA-256 mismatch"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            manifest_fixture.manifest.path,
            manifest_sha256=manifest_fixture.manifest.sha256,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("symbol", "returned a different symbol"),
        ("status", "raw body status does not match"),
        ("tick", "declared scales do not match"),
        ("missing_lot", "lacks PRICE_FILTER or LOT_SIZE provenance"),
    ],
)
def test_reader_rejects_rebound_exchange_info_semantic_forgery(
    manifest_fixture: _Fixture, case: str, message: str
) -> None:
    payload = _top_payload(manifest_fixture)
    raw_path = manifest_fixture.manifest.symbol_metadata[0].raw_path
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    symbol = raw_payload["symbols"][0]
    if case == "symbol":
        symbol["symbol"] = "ETHUSDT"
    elif case == "status":
        symbol["status"] = "BREAK"
    elif case == "tick":
        symbol["filters"][0]["tickSize"] = "0.02000000"
    else:
        symbol["filters"] = [symbol["filters"][0]]
    path, digest = _rebind_symbol_metadata_artifacts(
        manifest_fixture,
        payload,
        raw_payload=raw_payload,
    )

    with pytest.raises(M8ManifestError, match=message):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("timestamp", "observed timestamp does not match"),
        ("range", "must have null API range bounds"),
        ("checksum", "source sidecar checksum does not match"),
        ("source", "source sidecar source claim does not match"),
    ],
)
def test_reader_rejects_rebound_exchange_info_sidecar_forgery(
    manifest_fixture: _Fixture, case: str, message: str
) -> None:
    payload = _top_payload(manifest_fixture)
    sidecar_path = manifest_fixture.manifest.symbol_metadata[0].source_manifest_path
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if case == "timestamp":
        sidecar["downloaded_at_utc"] = "2026-08-07T12:00:00.123456788Z"
    elif case == "range":
        sidecar["requested_range_ns"]["start"] = 1
    elif case == "checksum":
        sidecar["checksum"]["value"] = "0" * 64
    else:
        sidecar["source"] = "untrusted_source"
    path, digest = _rebind_symbol_metadata_artifacts(
        manifest_fixture,
        payload,
        sidecar_payload=sidecar,
    )

    with pytest.raises(M8ManifestError, match=message):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


def test_reader_rejects_manifest_path_traversal_even_when_readdressed(
    manifest_fixture: _Fixture, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside")
    payload = _top_payload(manifest_fixture)
    payload["entries"][0]["raw"]["zip_path"] = "../../outside.zip"
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="escapes the M8 input root"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


def test_reader_rejects_exchange_info_path_traversal_even_when_readdressed(
    manifest_fixture: _Fixture, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text('{"symbols": []}', encoding="utf-8")
    payload = _top_payload(manifest_fixture)
    payload["symbol_metadata"][0]["raw_path"] = "../outside.json"
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="escapes the M8 input root"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


@pytest.mark.parametrize("field", ["path", "source_manifest_path"])
def test_reader_rejects_official_checksum_path_traversal_even_when_readdressed(
    manifest_fixture: _Fixture, tmp_path: Path, field: str
) -> None:
    outside = tmp_path / "outside.CHECKSUM"
    outside.write_text("0" * 64 + "  outside.zip\n", encoding="ascii")
    payload = _top_payload(manifest_fixture)
    payload["entries"][0]["raw"]["checksum"][field] = "../outside.CHECKSUM"
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="escapes the M8 input root"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_reader_requires_exact_official_checksum_descriptor_keys(
    manifest_fixture: _Fixture, case: str
) -> None:
    payload = _top_payload(manifest_fixture)
    checksum = payload["entries"][0]["raw"]["checksum"]
    if case == "missing":
        checksum.pop("source_manifest_sha256")
    else:
        checksum["unbound_claim"] = True
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="official CHECKSUM keys are invalid"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


def test_reader_rejects_pre_checksum_manifest_shape(manifest_fixture: _Fixture) -> None:
    payload = _top_payload(manifest_fixture)
    payload["manifest_version"] = "1.1.0"
    payload["entries"][0]["raw"].pop("checksum")
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="unsupported M8 input manifest version"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


@pytest.mark.parametrize("case", ["order", "total", "version", "extra_key"])
def test_reader_rejects_symbol_metadata_schema_and_aggregate_tamper(
    manifest_fixture: _Fixture, case: str
) -> None:
    payload = _top_payload(manifest_fixture)
    if case == "order":
        payload["symbol_metadata"].reverse()
        message = "out of frozen symbol order"
    elif case == "total":
        payload["total_symbol_metadata_bytes"] += 1
        message = "total symbol metadata byte claim does not match"
    elif case == "version":
        payload["manifest_version"] = "1.0.0"
        message = "unsupported M8 input manifest version"
    else:
        payload["symbol_metadata"][0]["unbound_claim"] = True
        message = r"symbol_metadata\[0\] keys are invalid"
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match=message):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


def test_reader_rejects_rebound_dataset_row_and_ordinal_forgery(
    manifest_fixture: _Fixture,
) -> None:
    payload = _top_payload(manifest_fixture)
    normalized = payload["entries"][0]["normalized"]
    dataset_path = manifest_fixture.root / normalized["dataset_manifest_path"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["artifacts"][0]["write_ordinal"] = 1
    write_json(dataset_path, dataset)
    normalized["dataset_manifest_sha256"] = sha256_file(dataset_path)
    normalized["dataset_manifest_bytes"] = dataset_path.stat().st_size
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="write ordinals are not contiguous"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


def test_reader_rejects_rebound_part_sidecar_semantic_forgery(
    manifest_fixture: _Fixture,
) -> None:
    payload = _top_payload(manifest_fixture)
    normalized = payload["entries"][0]["normalized"]
    part = normalized["parts"][0]
    sidecar_path = manifest_fixture.root / part["sidecar_path"]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["symbol"] = "ETHUSDT"
    write_json(sidecar_path, sidecar)
    rebound_sidecar_sha = sha256_file(sidecar_path)
    part["sidecar_sha256"] = rebound_sidecar_sha
    part["sidecar_bytes"] = sidecar_path.stat().st_size

    dataset_path = manifest_fixture.root / normalized["dataset_manifest_path"]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["artifacts"][0]["manifest_sha256"] = rebound_sidecar_sha
    write_json(dataset_path, dataset)
    normalized["dataset_manifest_sha256"] = sha256_file(dataset_path)
    normalized["dataset_manifest_bytes"] = dataset_path.stat().st_size
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="sidecar symbol claim does not match"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )


def test_reader_rejects_config_binding_and_content_address_tamper(
    manifest_fixture: _Fixture,
) -> None:
    payload = _top_payload(manifest_fixture)
    payload["config"]["semantic_sha256"] = "0" * 64
    path, digest = _write_readdressed_manifest(manifest_fixture.root, payload)

    with pytest.raises(M8ManifestError, match="different frozen configuration"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            path,
            manifest_sha256=digest,
        )

    copied = manifest_fixture.root / "_manifests" / "not-content-addressed.json"
    copied.write_bytes(manifest_fixture.manifest.path.read_bytes())
    with pytest.raises(M8ManifestError, match="filename is not content-addressed"):
        read_m8_input_manifest(
            manifest_fixture.config,
            manifest_fixture.root,
            copied,
            manifest_sha256=manifest_fixture.manifest.sha256,
        )
