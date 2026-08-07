from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import microstructure.public_data as public_data
from microstructure.config import ProjectConfig, load_config
from microstructure.data.schemas import table_from_records
from microstructure.data.storage import (
    DatasetWriteResult,
    write_partitioned_parquet,
    write_source_manifest,
)
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.provenance import read_json, sha256_file, write_json
from microstructure.public_data import (
    PublicDataError,
    read_public_trades,
    verify_public_trade_dataset,
)

START_NS = 1_704_153_600_000_000_000
END_NS = START_NS + 1_000_000_000
PROJECT_ROOT = Path(__file__).parents[1]


@dataclass(frozen=True, slots=True)
class ManifestedFixture:
    config: ProjectConfig
    ingestion_path: Path
    ingestion_sha256: str
    dataset: DatasetWriteResult
    table: pa.Table
    aggregate_paths: dict[str, Path]
    exchange_info_paths: dict[str, Path]
    raw_manifest_paths: dict[Path, Path]


def _config(root: Path, evidence_tier: str) -> ProjectConfig:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    return replace(
        base,
        run=replace(base.run, evidence_tier=evidence_tier),
        data=replace(
            base.data,
            start=datetime.fromtimestamp(START_NS / 1_000_000_000, tz=UTC),
            end=datetime.fromtimestamp(END_NS / 1_000_000_000, tz=UTC),
            max_events_per_symbol=3,
            partition_root=root / "normalized",
            raw_root=root / "raw",
        ),
    )


def _as_public_trades(
    table: pa.Table,
    source_artifact_ids: dict[str, str],
    *,
    invalid_quality: bool,
    cross_symbol_lineage: bool,
    normalized_scale_mismatch: bool,
) -> pa.Table:
    records = table.to_pylist()
    for record in records:
        symbol = str(record["symbol"])
        lineage_symbol = "ETHUSDT" if cross_symbol_lineage and symbol == "BTCUSDT" else symbol
        price = float(Decimal(int(record["price_ticks"])) * Decimal("0.01"))
        quantity = float(Decimal(int(record["quantity_lots"])) * Decimal("0.001"))
        event_ts_ns = int(record["event_ts_ns"]) // 1_000_000 * 1_000_000
        record.update(
            {
                "venue": "binance_spot",
                "event_ts_ns": event_ts_ns,
                "received_ts_ns": None,
                "available_ts_ns": event_ts_ns,
                "availability_basis": "exchange_event_time_proxy",
                "capture_seq": None,
                "continuity_id": None,
                "source_artifact_id": source_artifact_ids[lineage_symbol],
                "price": price,
                "quantity": quantity,
                "quote_quantity": price * quantity,
            }
        )
    if invalid_quality:
        records[1] = dict(records[0])
    if normalized_scale_mismatch:
        records[1]["tick_size"] = 0.02
    return table_from_records("trades", records)


def _fixture(
    root: Path,
    *,
    stored_schema: str = "trades",
    requested_tier: str = "PUBLIC_SAMPLE_PARTIAL",
    effective_tier: str = "PUBLIC_SAMPLE_PARTIAL",
    complete: bool = False,
    invalid_quality: bool = False,
    cross_symbol_lineage: bool = False,
    raw_trade_id_offset: bool = False,
    raw_price_mismatch: bool = False,
    normalized_scale_mismatch: bool = False,
    physical_clock_reversal: bool = False,
    raw_terminal_sentinel: bool = False,
    drop_last_normalized_per_symbol: bool = False,
) -> ManifestedFixture:
    config = _config(root, requested_tier)
    generated = generate_synthetic_market(
        symbols=("BTCUSDT", "ETHUSDT"),
        events_per_symbol=3,
        start_ts_ns=START_NS,
        seed=17,
    )
    raw_entries: list[dict[str, object]] = []
    aggregate_paths: dict[str, Path] = {}
    exchange_info_paths: dict[str, Path] = {}
    raw_manifest_paths: dict[Path, Path] = {}
    aggregate_digests: dict[str, str] = {}
    generated_records = generated.trades.to_pylist()
    for symbol in config.data.symbols:
        exchange_info_path = (
            root / "raw" / "binance_spot" / "exchange_info" / symbol / "fixture.json"
        )
        write_json(
            exchange_info_path,
            {
                "symbols": [
                    {
                        "symbol": symbol,
                        "status": "TRADING",
                        "baseAsset": symbol.removesuffix("USDT"),
                        "quoteAsset": "USDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.00100000"},
                        ],
                    }
                ]
            },
        )
        exchange_manifest_path, exchange_manifest_sha = write_source_manifest(
            exchange_info_path,
            source="binance_spot_public_api",
            source_uri=f"{config.data.base_url}/api/v3/exchangeInfo?symbol={symbol}",
            downloaded_at_utc="2026-08-07T12:00:00Z",
            requested_start_ns=None,
            requested_end_ns=None,
        )
        exchange_info_paths[symbol] = exchange_info_path
        raw_manifest_paths[exchange_info_path] = exchange_manifest_path
        raw_entries.append(
            {
                "path": str(exchange_info_path.relative_to(root)),
                "sha256": sha256_file(exchange_info_path),
                "manifest_path": str(exchange_manifest_path.relative_to(root)),
                "manifest_sha256": exchange_manifest_sha,
            }
        )

        symbol_records = [record for record in generated_records if str(record["symbol"]) == symbol]
        aggregate_path = root / "raw" / "binance_spot" / "agg_trades" / symbol / "fixture.json"
        raw_payload = [
            {
                "a": int(record["trade_id"])
                + (1_000_000 if raw_trade_id_offset and symbol == "BTCUSDT" else 0),
                "f": record["first_trade_id"],
                "l": record["last_trade_id"],
                "p": str(
                    Decimal(int(record["price_ticks"])) * Decimal("0.01")
                    + (
                        Decimal("0.01")
                        if raw_price_mismatch
                        and symbol == "BTCUSDT"
                        and record["trade_id"] == symbol_records[1]["trade_id"]
                        else Decimal(0)
                    )
                ),
                "q": str(Decimal(int(record["quantity_lots"])) * Decimal("0.001")),
                "T": int(record["event_ts_ns"]) // 1_000_000,
                "m": record["buyer_is_maker"],
            }
            for record in symbol_records
        ]
        if raw_terminal_sentinel and symbol == "BTCUSDT":
            final = dict(raw_payload[-1])
            final["a"] = int(final["a"]) + 1
            final["f"] = int(final["l"]) + 1
            final["l"] = int(final["l"]) + 1
            final["T"] = END_NS // 1_000_000
            raw_payload.append(final)
        write_json(aggregate_path, raw_payload)
        aggregate_manifest_path, aggregate_manifest_sha = write_source_manifest(
            aggregate_path,
            source="binance_spot_public_api",
            source_uri=(
                f"{config.data.base_url}/api/v3/aggTrades?symbol={symbol}"
                f"&startTime={START_NS // 1_000_000}"
                f"&endTime={(END_NS - 1) // 1_000_000}"
                f"&limit={config.data.request_limit}"
            ),
            downloaded_at_utc="2026-08-07T12:00:00Z",
            requested_start_ns=START_NS,
            requested_end_ns=END_NS,
        )
        aggregate_paths[symbol] = aggregate_path
        raw_manifest_paths[aggregate_path] = aggregate_manifest_path
        aggregate_digest = sha256_file(aggregate_path)
        aggregate_digests[symbol] = aggregate_digest
        raw_entries.append(
            {
                "path": str(aggregate_path.relative_to(root)),
                "sha256": aggregate_digest,
                "manifest_path": str(aggregate_manifest_path.relative_to(root)),
                "manifest_sha256": aggregate_manifest_sha,
            }
        )

    table = (
        _as_public_trades(
            generated.trades,
            aggregate_digests,
            invalid_quality=invalid_quality,
            cross_symbol_lineage=cross_symbol_lineage,
            normalized_scale_mismatch=normalized_scale_mismatch,
        )
        if stored_schema == "trades"
        else generated.book_observations
    )
    if physical_clock_reversal:
        records = table.to_pylist()
        records[1], records[2] = records[2], records[1]
        table = table_from_records("trades", records)
    if drop_last_normalized_per_symbol:
        records = table.to_pylist()
        retained: list[dict[str, Any]] = []
        for symbol in config.data.symbols:
            symbol_records = [record for record in records if str(record["symbol"]) == symbol]
            retained.extend(symbol_records[:-1])
        table = table_from_records("trades", retained)
    dataset = write_partitioned_parquet(
        table.to_batches(max_chunksize=3),
        root=root / "normalized",
        dataset="trades",
        schema_name=stored_schema,
        source="binance_spot_rest",
        source_uri="https://data-api.binance.vision/api/v3/aggTrades",
        downloaded_at_utc="2026-08-07T12:00:00Z",
        requested_start_ns=START_NS,
        requested_end_ns=END_NS,
        max_rows_per_file=2,
    )
    counts: dict[str, int] = {}
    for symbol in table.column("symbol").to_pylist():
        counts[str(symbol)] = counts.get(str(symbol), 0) + 1
    payload: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "artifact_kind": "ingestion_run",
        "created_at_utc": "2026-08-07T12:00:00Z",
        "mode": "binance_rest",
        "evidence_tier": effective_tier,
        "requested_evidence_tier": requested_tier,
        "source": config.data.source,
        "schema_version": "1.0.0",
        "requested_range_ns": {"start": START_NS, "end_exclusive": END_NS},
        "row_cap_per_symbol": config.data.max_events_per_symbol,
        "all_requested_ranges_complete": complete,
        "symbols": [
            {
                "symbol": symbol,
                "rows": rows,
                "complete_range": complete,
                "tick_size": "0.01",
                "lot_size": "0.001",
                **(
                    {
                        "raw_page_count": 1,
                        "stop_reason": "short_page",
                        "last_raw_page_sha256": aggregate_digests[symbol],
                        "stream_summary": {
                            "requested_start_ns": START_NS,
                            "requested_end_ns": END_NS,
                            "rows_yielded": rows,
                            "raw_page_count": 1,
                            "stop_reason": "short_page",
                            "complete_range": complete,
                            "last_raw_page": {
                                "path": str(aggregate_paths[symbol].relative_to(root)),
                                "manifest_path": str(
                                    raw_manifest_paths[aggregate_paths[symbol]].relative_to(root)
                                ),
                                "sha256": aggregate_digests[symbol],
                                "request_uri": read_json(
                                    raw_manifest_paths[aggregate_paths[symbol]]
                                )["source_uri"],
                                "row_count": 3,
                            },
                        },
                    }
                    if drop_last_normalized_per_symbol
                    else {}
                ),
            }
            for symbol, rows in sorted(counts.items())
        ],
        "normalized_datasets": [
            {
                "schema_name": "trades",
                "rows": table.num_rows,
                "manifest_path": str(dataset.manifest_path.relative_to(root)),
                "manifest_sha256": dataset.manifest_sha256,
            }
        ],
        "raw_artifacts": raw_entries,
    }
    ingestion_path = root / "_ingestion_manifests" / "ingestion.manifest-fixture.json"
    write_json(ingestion_path, payload)
    return ManifestedFixture(
        config=config,
        ingestion_path=ingestion_path,
        ingestion_sha256=sha256_file(ingestion_path),
        dataset=dataset,
        table=table,
        aggregate_paths=aggregate_paths,
        exchange_info_paths=exchange_info_paths,
        raw_manifest_paths=raw_manifest_paths,
    )


def _rewrite_ingestion(fixture: ManifestedFixture, payload: dict[str, Any]) -> str:
    write_json(fixture.ingestion_path, payload)
    return sha256_file(fixture.ingestion_path)


def _raw_entry(payload: dict[str, Any], fixture: ManifestedFixture, path: Path) -> dict[str, Any]:
    relative_path = str(path.relative_to(fixture.ingestion_path.parent.parent))
    for raw_entry in payload["raw_artifacts"]:
        if raw_entry["path"] == relative_path:
            return raw_entry
    raise AssertionError(f"fixture raw artifact is not manifested: {path}")


def _rewrite_raw_sidecar_uri(
    fixture: ManifestedFixture,
    path: Path,
    source_uri: str,
) -> str:
    sidecar_path = fixture.raw_manifest_paths[path]
    sidecar = read_json(sidecar_path)
    sidecar["source_uri"] = source_uri
    write_json(sidecar_path, sidecar)
    ingestion = read_json(fixture.ingestion_path)
    entry = _raw_entry(ingestion, fixture, path)
    entry["manifest_sha256"] = sha256_file(sidecar_path)
    return _rewrite_ingestion(fixture, ingestion)


def _rewrite_raw_payload(
    fixture: ManifestedFixture,
    path: Path,
    payload: dict[str, Any] | list[Any],
) -> str:
    write_json(path, payload)
    raw_sha = sha256_file(path)
    sidecar_path = fixture.raw_manifest_paths[path]
    sidecar = read_json(sidecar_path)
    sidecar["bytes"] = path.stat().st_size
    sidecar["checksum"]["value"] = raw_sha
    write_json(sidecar_path, sidecar)
    ingestion = read_json(fixture.ingestion_path)
    entry = _raw_entry(ingestion, fixture, path)
    entry["sha256"] = raw_sha
    entry["manifest_sha256"] = sha256_file(sidecar_path)
    return _rewrite_ingestion(fixture, ingestion)


def test_reads_only_manifested_parts_and_returns_arrow_polars_utc_coverage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    undeclared = tmp_path / "normalized" / "undeclared.parquet"
    undeclared.write_bytes(b"this file must never be discovered")

    result = read_public_trades(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    assert result.rows == 6
    assert result.arrow_trades.equals(fixture.table)
    assert result.polars_trades.height == 6
    assert result.validation.error_count == 0
    assert result.row_bound == 6
    assert result.evidence_tier == "PUBLIC_SAMPLE_PARTIAL"
    assert not result.all_requested_ranges_complete
    assert result.observed.start_ns == START_NS
    assert result.observed.end_inclusive_ns == START_NS + 200_000_000
    assert result.observed.start_utc == "2024-01-02T00:00:00.000000000Z"
    assert result.observed.end_inclusive_utc == "2024-01-02T00:00:00.200000000Z"
    assert {item.symbol: item.rows for item in result.symbols} == {
        "BTCUSDT": 3,
        "ETHUSDT": 3,
    }
    assert set(result.part_paths) == {item.data_path for item in fixture.dataset.artifacts}
    expected_raw_paths = set(fixture.aggregate_paths.values()) | set(
        fixture.exchange_info_paths.values()
    )
    assert set(result.raw_artifact_paths) == expected_raw_paths
    assert set(result.raw_manifest_paths) == set(fixture.raw_manifest_paths.values())
    assert set(result.raw_artifact_sha256s) == {sha256_file(path) for path in expected_raw_paths}
    assert result.canonical_order == (
        "venue",
        "symbol",
        "available_ts_ns",
        "event_ts_ns",
        "trade_id",
    )
    assert undeclared not in result.part_paths


def test_requires_matching_ingestion_and_normalized_manifest_hashes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(PublicDataError, match="ingestion manifest SHA-256 mismatch"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256="0" * 64,
        )

    payload = read_json(fixture.ingestion_path)
    payload["normalized_datasets"][0]["manifest_sha256"] = "0" * 64
    rewritten_sha = _rewrite_ingestion(fixture, payload)
    with pytest.raises(PublicDataError, match="normalized dataset manifest SHA-256 mismatch"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=rewritten_sha,
        )


@pytest.mark.parametrize(
    ("requested_tier", "complete"),
    [("PUBLIC_SAMPLE_PARTIAL", True), ("FULL_DATA", False)],
)
def test_rejects_partial_manifest_promoted_to_full_data(
    tmp_path: Path, requested_tier: str, complete: bool
) -> None:
    fixture = _fixture(
        tmp_path,
        requested_tier=requested_tier,
        effective_tier="FULL_DATA",
        complete=complete,
    )

    with pytest.raises(PublicDataError, match="cannot be promoted to FULL_DATA"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=fixture.ingestion_sha256,
        )


def test_rejects_missing_or_tampered_declared_parts(tmp_path: Path) -> None:
    missing_fixture = _fixture(tmp_path / "missing")
    missing_fixture.dataset.artifacts[0].data_path.unlink()
    with pytest.raises(PublicDataError, match=r"missing normalized dataset\.artifacts"):
        read_public_trades(
            missing_fixture.config,
            missing_fixture.ingestion_path,
            ingestion_manifest_sha256=missing_fixture.ingestion_sha256,
        )

    tampered_fixture = _fixture(tmp_path / "tampered")
    part = tampered_fixture.dataset.artifacts[0].data_path
    part.write_bytes(part.read_bytes() + b"tampered")
    with pytest.raises(PublicDataError, match=r"Parquet part \d+ SHA-256 mismatch"):
        read_public_trades(
            tampered_fixture.config,
            tampered_fixture.ingestion_path,
            ingestion_manifest_sha256=tampered_fixture.ingestion_sha256,
        )


def test_rejects_unexpected_normalized_schema(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, stored_schema="book_observations")

    with pytest.raises(PublicDataError, match="declares an unexpected schema"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=fixture.ingestion_sha256,
        )


def test_enforces_required_row_bound_before_part_loading(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dataset_payload = read_json(fixture.dataset.manifest_path)
    dataset_payload["rows"] = 7
    write_json(fixture.dataset.manifest_path, dataset_payload)
    ingestion_payload = read_json(fixture.ingestion_path)
    ingestion_payload["normalized_datasets"][0]["rows"] = 7
    ingestion_payload["normalized_datasets"][0]["manifest_sha256"] = sha256_file(
        fixture.dataset.manifest_path
    )
    ingestion_sha = _rewrite_ingestion(fixture, ingestion_payload)
    fixture.dataset.artifacts[0].data_path.write_bytes(b"also tampered")

    with pytest.raises(PublicDataError, match="above required bound 6"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_rejects_cross_manifest_coverage_disagreement(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = read_json(fixture.ingestion_path)
    payload["requested_range_ns"]["start"] = START_NS + 1
    rewritten_sha = _rewrite_ingestion(fixture, payload)

    with pytest.raises(PublicDataError, match="requested range does not match configuration"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=rewritten_sha,
        )


@pytest.mark.parametrize("mismatch", ["source", "symbols", "range"])
def test_rejects_manifest_that_does_not_match_config(tmp_path: Path, mismatch: str) -> None:
    fixture = _fixture(tmp_path)
    config = fixture.config
    if mismatch == "source":
        config = replace(config, data=replace(config.data, source="different_public_source"))
    elif mismatch == "symbols":
        config = replace(config, data=replace(config.data, symbols=("BTCUSDT",)))
    else:
        config = replace(
            config,
            data=replace(
                config.data,
                start=datetime.fromtimestamp((START_NS - 1_000_000_000) / 1e9, tz=UTC),
            ),
        )

    with pytest.raises(PublicDataError, match=r"do(?:es)? not match"):
        read_public_trades(
            config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=fixture.ingestion_sha256,
        )


def test_rejects_tampered_raw_bytes_and_undeclared_row_lineage(tmp_path: Path) -> None:
    tampered = _fixture(tmp_path / "tampered-raw")
    tampered_path = tampered.aggregate_paths["BTCUSDT"]
    tampered_path.write_bytes(tampered_path.read_bytes() + b"tampered")
    with pytest.raises(PublicDataError, match=r"raw artifact \d+ SHA-256 mismatch"):
        read_public_trades(
            tampered.config,
            tampered.ingestion_path,
            ingestion_manifest_sha256=tampered.ingestion_sha256,
        )

    lineage = _fixture(tmp_path / "lineage")
    replacement_raw = lineage.ingestion_path.parent.parent / "raw" / "replacement.json"
    original_payload = read_json(lineage.aggregate_paths["BTCUSDT"])
    for record in original_payload:
        record["a"] += 1_000_000
    write_json(replacement_raw, original_payload)
    replacement_manifest, replacement_manifest_sha = write_source_manifest(
        replacement_raw,
        source="binance_spot_public_api",
        source_uri=(
            f"{lineage.config.data.base_url}/api/v3/aggTrades?symbol=BTCUSDT"
            f"&startTime={START_NS // 1_000_000}"
            f"&endTime={(END_NS - 1) // 1_000_000}"
            f"&limit={lineage.config.data.request_limit}"
        ),
        downloaded_at_utc="2026-08-07T12:00:00Z",
        requested_start_ns=START_NS,
        requested_end_ns=END_NS,
    )
    payload = read_json(lineage.ingestion_path)
    entry = _raw_entry(payload, lineage, lineage.aggregate_paths["BTCUSDT"])
    entry.update(
        {
            "path": str(replacement_raw.relative_to(tmp_path / "lineage")),
            "sha256": sha256_file(replacement_raw),
            "manifest_path": str(replacement_manifest.relative_to(tmp_path / "lineage")),
            "manifest_sha256": replacement_manifest_sha,
        }
    )
    ingestion_sha = _rewrite_ingestion(lineage, payload)
    with pytest.raises(PublicDataError, match="references an undeclared or empty raw artifact"):
        read_public_trades(
            lineage.config,
            lineage.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("wrong_scheme", "not bound to configured data.base_url"),
        ("wrong_host", "not bound to configured data.base_url"),
        ("wrong_path", "does not use an allowed Binance public endpoint"),
        ("extra_query", "initial-time or fromId query parameters"),
        ("missing_time", "initial-time or fromId query parameters"),
        ("wrong_symbol", "exactly one initial-time aggTrades page"),
    ],
)
def test_rejects_unbound_or_malformed_aggregate_trade_uri(
    tmp_path: Path,
    case: str,
    expected: str,
) -> None:
    fixture = _fixture(tmp_path)
    base = fixture.config.data.base_url
    query = (
        f"symbol=BTCUSDT&startTime={START_NS // 1_000_000}"
        f"&endTime={(END_NS - 1) // 1_000_000}"
        f"&limit={fixture.config.data.request_limit}"
    )
    uris = {
        "wrong_scheme": f"http://data-api.binance.vision/api/v3/aggTrades?{query}",
        "wrong_host": f"https://evil.example/api/v3/aggTrades?{query}",
        "wrong_path": f"{base}/api/v3/aggTrades/extra?{query}",
        "extra_query": f"{base}/api/v3/aggTrades?{query}&unexpected=1",
        "missing_time": (
            f"{base}/api/v3/aggTrades?symbol=BTCUSDT"
            f"&startTime={START_NS // 1_000_000}"
            f"&limit={fixture.config.data.request_limit}"
        ),
        "wrong_symbol": f"{base}/api/v3/aggTrades?{query.replace('BTCUSDT', 'ETHUSDT')}",
    }
    ingestion_sha = _rewrite_raw_sidecar_uri(
        fixture,
        fixture.aggregate_paths["BTCUSDT"],
        uris[case],
    )

    with pytest.raises(PublicDataError, match=expected):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_rejects_exchange_info_query_and_requires_every_configured_symbol(
    tmp_path: Path,
) -> None:
    malformed = _fixture(tmp_path / "query")
    malformed_sha = _rewrite_raw_sidecar_uri(
        malformed,
        malformed.exchange_info_paths["BTCUSDT"],
        (f"{malformed.config.data.base_url}/api/v3/exchangeInfo?symbol=BTCUSDT&unexpected=1"),
    )
    with pytest.raises(PublicDataError, match="exactly the symbol parameter"):
        read_public_trades(
            malformed.config,
            malformed.ingestion_path,
            ingestion_manifest_sha256=malformed_sha,
        )

    missing = _fixture(tmp_path / "missing")
    ingestion = read_json(missing.ingestion_path)
    missing_relative = str(
        missing.exchange_info_paths["BTCUSDT"].relative_to(missing.ingestion_path.parent.parent)
    )
    ingestion["raw_artifacts"] = [
        entry for entry in ingestion["raw_artifacts"] if entry["path"] != missing_relative
    ]
    missing_sha = _rewrite_ingestion(missing, ingestion)
    with pytest.raises(PublicDataError, match=r"lacks exchangeInfo.*BTCUSDT"):
        read_public_trades(
            missing.config,
            missing.ingestion_path,
            ingestion_manifest_sha256=missing_sha,
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "expected"),
    [
        ({"cross_symbol_lineage": True}, "raw page for ETHUSDT, not BTCUSDT"),
        ({"raw_trade_id_offset": True}, "trade_id is absent from its exact raw"),
        ({"raw_price_mismatch": True}, "exact raw aggregate-trade record: price"),
    ],
)
def test_rejects_cross_symbol_and_exact_raw_record_mismatches(
    tmp_path: Path,
    fixture_kwargs: dict[str, bool],
    expected: str,
) -> None:
    fixture = _fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(PublicDataError, match=expected):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=fixture.ingestion_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("tick_size", "0.02", "payload scales do not match ingestion claim"),
        ("status", "BREAK", "payload status is not TRADING"),
        ("symbol", "ETHUSDT", "payload symbol does not match its request URI"),
    ],
)
def test_rejects_semantically_tampered_exchange_info_payload(
    tmp_path: Path,
    field: str,
    value: str,
    expected: str,
) -> None:
    fixture = _fixture(tmp_path)
    metadata_path = fixture.exchange_info_paths["BTCUSDT"]
    payload = read_json(metadata_path)
    item = payload["symbols"][0]
    if field == "tick_size":
        price_filter = next(
            raw_filter
            for raw_filter in item["filters"]
            if raw_filter["filterType"] == "PRICE_FILTER"
        )
        price_filter["tickSize"] = value
    else:
        item[field] = value
    ingestion_sha = _rewrite_raw_payload(fixture, metadata_path, payload)

    with pytest.raises(PublicDataError, match=expected):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_rejects_raw_aggregate_event_time_reversal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    aggregate_path = fixture.aggregate_paths["BTCUSDT"]
    payload = read_json(aggregate_path)
    payload[0]["T"] = int(payload[1]["T"]) + 1
    ingestion_sha = _rewrite_raw_payload(fixture, aggregate_path, payload)

    with pytest.raises(PublicDataError, match="event times are not nondecreasing"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_allows_shared_empty_terminal_page_with_distinct_symbol_sidecars(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    bundle_root = fixture.ingestion_path.parent.parent
    empty_path = bundle_root / "raw" / "binance_spot" / "agg_trades" / "shared-empty.json"
    write_json(empty_path, [])
    empty_sha = sha256_file(empty_path)
    ingestion = read_json(fixture.ingestion_path)
    for symbol in fixture.config.data.symbols:
        symbol_rows = [row for row in fixture.table.to_pylist() if str(row["symbol"]) == symbol]
        from_id = max(int(row["trade_id"]) for row in symbol_rows) + 1
        sidecar_path, sidecar_sha = write_source_manifest(
            empty_path,
            source="binance_spot_public_api",
            source_uri=(
                f"{fixture.config.data.base_url}/api/v3/aggTrades?symbol={symbol}"
                f"&fromId={from_id}&limit={fixture.config.data.request_limit}"
            ),
            downloaded_at_utc="2026-08-07T12:00:01Z",
            requested_start_ns=START_NS,
            requested_end_ns=END_NS,
        )
        ingestion["raw_artifacts"].append(
            {
                "path": str(empty_path.relative_to(bundle_root)),
                "sha256": empty_sha,
                "manifest_path": str(sidecar_path.relative_to(bundle_root)),
                "manifest_sha256": sidecar_sha,
            }
        )
    ingestion_sha = _rewrite_ingestion(fixture, ingestion)

    result = read_public_trades(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=ingestion_sha,
    )

    assert result.rows == fixture.table.num_rows
    assert empty_path in result.raw_artifact_paths


def test_rejects_manifest_scale_mismatch_and_fresh_quality_errors(tmp_path: Path) -> None:
    scale = _fixture(tmp_path / "scale")
    payload = read_json(scale.ingestion_path)
    payload["symbols"][0]["tick_size"] = "0.02"
    ingestion_sha = _rewrite_ingestion(scale, payload)
    with pytest.raises(PublicDataError, match="payload scales do not match ingestion claim"):
        read_public_trades(
            scale.config,
            scale.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )

    invalid = _fixture(tmp_path / "quality", invalid_quality=True)
    with pytest.raises(PublicDataError, match="failed quality validation"):
        read_public_trades(
            invalid.config,
            invalid.ingestion_path,
            ingestion_manifest_sha256=invalid.ingestion_sha256,
        )


def test_canonical_order_is_independent_of_manifest_part_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dataset_payload = read_json(fixture.dataset.manifest_path)
    dataset_payload["artifacts"] = list(reversed(dataset_payload["artifacts"]))
    write_json(fixture.dataset.manifest_path, dataset_payload)
    ingestion_payload = read_json(fixture.ingestion_path)
    ingestion_payload["normalized_datasets"][0]["manifest_sha256"] = sha256_file(
        fixture.dataset.manifest_path
    )
    ingestion_sha = _rewrite_ingestion(fixture, ingestion_payload)

    result = read_public_trades(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=ingestion_sha,
    )

    assert result.arrow_trades.equals(fixture.table)


def test_verify_only_never_reads_normalized_parquet_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def _forbidden_read(*args: object, **kwargs: object) -> pa.Table:
        del args, kwargs
        raise AssertionError("verify-only must not read normalized rows")

    monkeypatch.setattr(pq.ParquetFile, "read", _forbidden_read)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    assert dataset.rows == 6
    assert dataset.part_paths == tuple(item.data_path for item in fixture.dataset.artifacts)


def test_verified_batch_stream_is_bounded_fresh_and_canonical(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    first = dataset.iter_verified_batches(
        batch_rows=2,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )
    first_batches = list(first)
    first_table = pa.Table.from_batches(first_batches, schema=fixture.table.schema)
    assert first.validation.error_count == 0
    assert [batch.num_rows for batch in first_batches] == [2, 2, 2]
    assert first_table.equals(fixture.table)

    second = dataset.iter_verified_batches(
        batch_rows=1,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )
    second_batches = list(second)
    assert all(batch.num_rows == 1 for batch in second_batches)
    assert pa.Table.from_batches(second_batches, schema=fixture.table.schema).equals(first_table)


def test_verified_stream_scans_each_parquet_part_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )
    original = pq.ParquetFile.iter_batches
    calls = 0

    def _counted_iter_batches(
        self: pq.ParquetFile,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", _counted_iter_batches)
    stream = dataset.iter_verified_batches(
        batch_rows=2,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )

    assert sum(batch.num_rows for batch in stream) == dataset.rows
    assert calls == len(dataset.part_paths)


def test_materialization_guard_is_checked_before_any_normalized_row_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)

    def _forbidden_stream(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("materialization guard must run before normalized row reading")

    monkeypatch.setattr(
        public_data.PublicTradeDataset,
        "iter_verified_batches",
        _forbidden_stream,
    )
    with pytest.raises(PublicDataError, match="above materialization guard 5"):
        read_public_trades(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=fixture.ingestion_sha256,
            materialization_max_rows=5,
        )


def test_incremental_dq_detects_duplicate_across_one_row_batches(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, invalid_quality=True)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    stream = dataset.iter_verified_batches(
        batch_rows=1,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )
    with pytest.raises(PublicDataError, match="failed quality validation"):
        list(stream)


def test_physical_order_dq_survives_external_canonical_sort_across_parts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, physical_clock_reversal=True)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    report = dataset.validate(
        batch_rows=1,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )

    assert any(item.rule_id == "temporal.out_of_order_event_time" for item in report.findings)


@pytest.mark.parametrize(
    ("fixture_kwargs", "expected"),
    [
        ({"raw_price_mismatch": True}, "exact raw aggregate-trade record: price"),
        ({"normalized_scale_mismatch": True}, "scales do not match manifest"),
    ],
)
def test_cross_batch_lineage_and_scale_failures_are_detected(
    tmp_path: Path,
    fixture_kwargs: dict[str, bool],
    expected: str,
) -> None:
    fixture = _fixture(tmp_path, **fixture_kwargs)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )
    stream = dataset.iter_verified_batches(
        batch_rows=1,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )

    with pytest.raises(PublicDataError, match=expected):
        list(stream)


def test_stream_rehashes_parquet_after_descriptor_verification(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )
    part = fixture.dataset.artifacts[0].data_path
    part.write_bytes(part.read_bytes() + b"changed-after-verification")

    stream = dataset.iter_verified_batches(
        batch_rows=2,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )
    with pytest.raises(PublicDataError, match="normalized Parquet part SHA-256 mismatch"):
        list(stream)


def test_raw_terminal_sentinel_outside_range_is_preserved_but_not_normalized(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, raw_terminal_sentinel=True)

    result = read_public_trades(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    assert result.rows == 6
    assert max(result.arrow_trades.column("event_ts_ns").to_pylist()) < END_NS


def test_rejects_oversized_raw_json_before_payload_parsing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    metadata_path = fixture.exchange_info_paths["BTCUSDT"]
    payload = read_json(metadata_path)
    payload["unused_padding"] = "x" * (8 * 1024 * 1024)
    ingestion_sha = _rewrite_raw_payload(fixture, metadata_path, payload)

    with pytest.raises(PublicDataError, match="exceeds bounded JSON size"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_rejects_giant_manifest_string_before_json_materialization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    dataset_manifest = read_json(fixture.dataset.manifest_path)
    dataset_manifest["unused_padding"] = "x" * (64 * 1024 + 1)
    write_json(fixture.dataset.manifest_path, dataset_manifest)
    ingestion = read_json(fixture.ingestion_path)
    ingestion["normalized_datasets"][0]["manifest_sha256"] = sha256_file(
        fixture.dataset.manifest_path
    )
    ingestion_sha = _rewrite_ingestion(fixture, ingestion)

    with pytest.raises(PublicDataError, match="JSON string above bounded token size"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_inverse_lineage_rejects_dropped_selected_raw_trades(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        complete=True,
        drop_last_normalized_per_symbol=True,
    )
    dataset = verify_public_trade_dataset(
        fixture.config,
        fixture.ingestion_path,
        ingestion_manifest_sha256=fixture.ingestion_sha256,
    )

    stream = dataset.iter_verified_batches(
        batch_rows=1,
        memory_limit="64MB",
        temp_directory=tmp_path,
    )
    with pytest.raises(PublicDataError, match=r"raw selected rows.*do not match"):
        list(stream)


def test_terminal_stream_summary_is_bound_to_top_level_claim(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        complete=True,
        drop_last_normalized_per_symbol=True,
    )
    ingestion = read_json(fixture.ingestion_path)
    ingestion["symbols"][0]["stream_summary"]["raw_page_count"] = 2
    ingestion_sha = _rewrite_ingestion(fixture, ingestion)

    with pytest.raises(PublicDataError, match="stream_summary disagrees"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_terminal_stream_summary_is_bound_to_declared_page_chain(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        complete=True,
        drop_last_normalized_per_symbol=True,
    )
    ingestion = read_json(fixture.ingestion_path)
    ingestion["symbols"][0]["raw_page_count"] = 2
    ingestion["symbols"][0]["stream_summary"]["raw_page_count"] = 2
    ingestion_sha = _rewrite_ingestion(fixture, ingestion)

    with pytest.raises(PublicDataError, match="does not match declared raw pages"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )


def test_complete_legacy_claim_requires_terminal_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, complete=True)

    with pytest.raises(PublicDataError, match="lacks terminal stream evidence"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=fixture.ingestion_sha256,
        )


def test_complete_range_cannot_end_after_terminal_page_download(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        complete=True,
        drop_last_normalized_per_symbol=True,
    )
    ingestion = read_json(fixture.ingestion_path)
    for path in fixture.aggregate_paths.values():
        sidecar_path = fixture.raw_manifest_paths[path]
        sidecar = read_json(sidecar_path)
        sidecar["downloaded_at_utc"] = "2023-01-01T00:00:00Z"
        write_json(sidecar_path, sidecar)
        entry = _raw_entry(ingestion, fixture, path)
        entry["manifest_sha256"] = sha256_file(sidecar_path)
    ingestion_sha = _rewrite_ingestion(fixture, ingestion)

    with pytest.raises(PublicDataError, match="ends after its terminal page was downloaded"):
        verify_public_trade_dataset(
            fixture.config,
            fixture.ingestion_path,
            ingestion_manifest_sha256=ingestion_sha,
        )
