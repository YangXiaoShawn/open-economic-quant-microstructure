from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from microstructure.data.binance import SymbolMetadata
from microstructure.data.binance_archive import (
    AcquiredDailyArchive,
    ArchiveDownloadLimits,
    DailyArchiveRequest,
    RawArchiveArtifact,
)
from microstructure.data.storage import write_source_manifest
from microstructure.m8_config import M8StudyConfig, load_m8_config
from microstructure.m8_ingestion import (
    M8DataQualityError,
    M8IngestionError,
    ingest_m8_archives,
)
from microstructure.m8_manifest import read_m8_input_manifest
from microstructure.provenance import sha256_file

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "m8_multidate_trade_study.toml"
_NS_PER_DAY = 86_400 * 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _day_start_ns(day: date) -> int:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return (start - _EPOCH).days * _NS_PER_DAY


def _metadata(root: Path, symbol: str) -> SymbolMetadata:
    tick = Decimal("0.01")
    lot = Decimal("0.00001") if symbol == "BTCUSDT" else Decimal("0.0001")
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
                        "maxPrice": "10000000.00",
                        "tickSize": format(tick, "f"),
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": format(lot, "f"),
                        "maxQty": "100000.0",
                        "stepSize": format(lot, "f"),
                    },
                ],
            }
        ]
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    raw_path = root / "raw" / "binance_spot" / "exchange_info" / symbol / f"{digest}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(body)
    source_uri = f"https://data-api.binance.vision/api/v3/exchangeInfo?symbol={symbol}"
    sidecar_path, _ = write_source_manifest(
        raw_path,
        source="binance_spot_public_api",
        source_uri=source_uri,
        downloaded_at_utc="2026-08-07T20:00:00Z",
        requested_start_ns=None,
        requested_end_ns=None,
    )
    return SymbolMetadata(
        venue="binance_spot",
        symbol=symbol,
        status="TRADING",
        base_asset=symbol.removesuffix("USDT"),
        quote_asset="USDT",
        tick_size=tick,
        lot_size=lot,
        min_price=Decimal("0.01"),
        max_price=Decimal("10000000.00"),
        min_quantity=lot,
        max_quantity=Decimal("100000.0"),
        observed_ts_ns=1_786_132_800_000_000_000,
        source_artifact_id=digest,
        source_path=raw_path,
        source_manifest_path=sidecar_path,
    )


class _ArchiveClient:
    def __init__(self, *, warning_on: tuple[str, str] | None = None) -> None:
        self.warning_on = warning_on
        self.calls: list[tuple[str, str, int, int]] = []

    def acquire(
        self,
        request: DailyArchiveRequest,
        *,
        raw_root: str | Path,
        limits: ArchiveDownloadLimits,
    ) -> AcquiredDailyArchive:
        self.calls.append(
            (
                request.symbol,
                request.date.isoformat(),
                limits.max_compressed_bytes,
                limits.max_uncompressed_bytes,
            )
        )
        root = Path(raw_root)
        day_text = request.date.isoformat()
        start_ms = _day_start_ns(request.date) // 1_000_000
        base_id = request.date.toordinal() * 10 + (0 if request.symbol == "BTCUSDT" else 5)
        timestamps = [start_ms + 1, start_ms + 2, start_ms + 3, start_ms + 4]
        if self.warning_on == (request.symbol, day_text):
            timestamps[-1] += 6_000
        rows = [
            (
                f"{base_id + index},42000.{index:02d},0.00100,{(base_id + index) * 2},"
                f"{(base_id + index) * 2},{timestamps[index]},"
                f"{'true' if index % 2 else 'false'},true"
            )
            for index in range(4)
        ]
        csv_bytes = ("\n".join(rows) + "\n").encode()
        archive_dir = root / "binance_spot" / "daily_agg_trades_archive" / request.symbol / day_text
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / request.archive_name
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(request.member_name, csv_bytes)
        archive_sha = sha256_file(archive_path)
        source_uri = (
            "https://data.binance.vision/data/spot/daily/aggTrades/"
            f"{request.symbol}/{request.archive_name}"
        )
        archive_sidecar, archive_sidecar_sha = write_source_manifest(
            archive_path,
            source="binance_spot_daily_aggtrades_archive",
            source_uri=source_uri,
            downloaded_at_utc="2026-08-07T20:00:00Z",
            requested_start_ns=_day_start_ns(request.date),
            requested_end_ns=_day_start_ns(request.date) + _NS_PER_DAY,
            upstream_checksum_sha256=archive_sha,
        )
        checksum_path = archive_dir / f"{request.archive_name}.CHECKSUM"
        checksum_path.write_text(f"{archive_sha}  {request.archive_name}\n", encoding="ascii")
        checksum_uri = source_uri + ".CHECKSUM"
        checksum_sidecar, checksum_sidecar_sha = write_source_manifest(
            checksum_path,
            source="binance_spot_daily_aggtrades_archive_checksum",
            source_uri=checksum_uri,
            downloaded_at_utc="2026-08-07T20:00:00Z",
            requested_start_ns=_day_start_ns(request.date),
            requested_end_ns=_day_start_ns(request.date) + _NS_PER_DAY,
        )
        return AcquiredDailyArchive(
            request=request,
            archive_artifact=RawArchiveArtifact(
                kind="archive_zip",
                path=archive_path,
                manifest_path=archive_sidecar,
                sha256=archive_sha,
                manifest_sha256=archive_sidecar_sha,
                bytes=archive_path.stat().st_size,
                source_uri=source_uri,
            ),
            checksum_artifact=RawArchiveArtifact(
                kind="archive_checksum",
                path=checksum_path,
                manifest_path=checksum_sidecar,
                sha256=sha256_file(checksum_path),
                manifest_sha256=checksum_sidecar_sha,
                bytes=checksum_path.stat().st_size,
                source_uri=checksum_uri,
            ),
            upstream_sha256=archive_sha,
            declared_uncompressed_bytes=len(csv_bytes),
            limits=limits,
        )


def _config() -> M8StudyConfig:
    return load_m8_config(CONFIG_PATH)


def _supplied_metadata(root: Path, config: M8StudyConfig) -> dict[str, SymbolMetadata]:
    return {symbol: _metadata(root, symbol) for symbol in config.study.symbols}


def test_ingest_m8_streams_all_eight_archives_and_writes_verified_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    root = tmp_path / "m8"
    root.mkdir()
    metadata = _supplied_metadata(root, config)
    client = _ArchiveClient()

    import pyarrow as pa  # type: ignore[import-untyped]

    monkeypatch.setattr(
        pa,
        "concat_tables",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("M8 ingestion must not concatenate full archives")
        ),
    )
    result = ingest_m8_archives(
        config,
        root,
        archive_client=client,
        supplied_metadata=metadata,
        batch_rows=2,
    )

    assert result.rows == 32
    assert len(result.entries) == 8
    assert len(result.symbol_metadata) == 2
    assert len(client.calls) == 8
    assert [(entry.date.isoformat(), entry.symbol) for entry in result.entries] == [
        (period.date.isoformat(), symbol)
        for period in config.periods
        for symbol in config.study.symbols
    ]
    assert all(entry.quality_errors == entry.quality_warnings == 0 for entry in result.entries)
    assert all(len(entry.normalized_parts) == 2 for entry in result.entries)
    assert result.input_manifest_path.parent.name == "_manifests"
    verified = read_m8_input_manifest(
        config,
        root,
        result.input_manifest_path,
        manifest_sha256=result.input_manifest_sha256,
    )
    assert verified.sha256 == result.input_manifest_sha256
    assert verified.metadata_for("BTCUSDT").tick_size == Decimal("0.01")


def test_ingest_m8_preserves_evidence_and_fails_before_manifest_on_warning(
    tmp_path: Path,
) -> None:
    config = _config()
    root = tmp_path / "m8"
    root.mkdir()
    metadata = _supplied_metadata(root, config)
    first = config.periods[0]
    client = _ArchiveClient(warning_on=(config.study.symbols[0], first.date.isoformat()))

    with pytest.raises(M8DataQualityError, match="0 errors and 1 warnings"):
        ingest_m8_archives(
            config,
            root,
            archive_client=client,
            supplied_metadata=metadata,
            batch_rows=2,
        )

    assert list(root.rglob("*.zip"))
    assert list(root.rglob("*.parquet"))
    assert list(root.rglob("validation-*.json"))
    assert not list((root / "_manifests").glob("m8-input.manifest-*.json"))


def test_ingest_m8_rejects_metadata_outside_input_root(tmp_path: Path) -> None:
    config = _config()
    root = tmp_path / "m8"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    supplied = _supplied_metadata(external, config)

    with pytest.raises(M8IngestionError, match="contained"):
        ingest_m8_archives(
            config,
            root,
            archive_client=_ArchiveClient(),
            supplied_metadata=supplied,
        )


def test_ingest_m8_enforces_total_archive_ceiling_after_preserving_raw(
    tmp_path: Path,
) -> None:
    config = _config()
    constrained = replace(config, study=replace(config.study, max_total_download_bytes=1))
    root = tmp_path / "m8"
    root.mkdir()
    metadata = _supplied_metadata(root, constrained)

    with pytest.raises(M8IngestionError, match="total-download ceiling"):
        ingest_m8_archives(
            constrained,
            root,
            archive_client=_ArchiveClient(),
            supplied_metadata=metadata,
            batch_rows=2,
        )

    assert len(list(root.rglob("*.zip"))) == 1
    assert not list(root.rglob("*.parquet"))
    assert not list((root / "_manifests").glob("m8-input.manifest-*.json"))
