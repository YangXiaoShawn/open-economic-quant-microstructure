from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import microstructure.ingestion as ingestion_module
from microstructure.config import ProjectConfig, load_config
from microstructure.data.binance import BinanceHistoricalTradeDownloader
from microstructure.data.quality import validate_table
from microstructure.data.schemas import table_from_records
from microstructure.data.storage import DatasetWriteResult
from microstructure.ingestion import (
    DataAdapterRegistry,
    DataQualityGateError,
    IngestionError,
    IngestionResult,
    builtin_data_adapter_registry,
    ingest_from_config,
    ingest_public_trades,
    ingest_synthetic,
    validate_configured_input,
)
from microstructure.provenance import read_json, sha256_file, write_json

PROJECT_ROOT = Path(__file__).parents[1]


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload, separators=(",", ":")).encode()
        self.text = self.content.decode()
        self.headers = dict(headers or {})
        self.url = "https://data-api.binance.vision/fixture"

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: Mapping[str, object], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        response = self.responses.pop(0)
        response.url = f"{url}?{urlencode(params)}"
        return response


def _metadata(symbol: str, *, lot_size: str) -> dict[str, object]:
    base = "BTC" if symbol == "BTCUSDT" else "ETH"
    return {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "baseAsset": base,
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
                        "minQty": lot_size,
                        "maxQty": "9000.0",
                        "stepSize": lot_size,
                    },
                ],
            }
        ]
    }


def _trades(symbol: str, start_ms: int, quantity: str) -> list[dict[str, object]]:
    offset = 0 if symbol == "BTCUSDT" else 100
    return [
        {
            "a": offset + 1,
            "p": "100.01",
            "q": quantity,
            "f": offset + 10,
            "l": offset + 10,
            "T": start_ms,
            "m": False,
        },
        {
            "a": offset + 2,
            "p": "100.02",
            "q": quantity,
            "f": offset + 11,
            "l": offset + 11,
            "T": start_ms + 1,
            "m": True,
        },
    ]


def test_synthetic_ingestion_validates_and_stages_immutable_bundle(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            events_per_symbol=8,
            partition_root=tmp_path / "normalized",
        ),
    )

    result = ingest_synthetic(config, tmp_path)

    assert result.mode == "synthetic"
    assert result.evidence_tier == "SYNTHETIC_SMOKE"
    assert result.validation.passed
    assert result.dataset("trades").rows == 16
    assert result.dataset("book_observations").rows == 16
    trades = result.dataset("trades")
    assert trades.table is not None
    assert trades.materialize(max_rows=16) is trades.table
    assert result.rows == 32
    assert result.raw_artifacts == ()
    assert result.ingestion_manifest_path.is_file()
    assert result.ingestion_manifest_sha256 == sha256_file(result.ingestion_manifest_path)
    assert all(dataset.storage.manifest_path.is_file() for dataset in result.datasets)
    assert all(path.is_file() for path in result.validation.report_paths)
    assert validate_configured_input(config).passed


def test_dispatcher_uses_synthetic_mode_and_caller_output_root(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    config = replace(base, data=replace(base.data, events_per_symbol=2))

    result = ingest_from_config(config, tmp_path)

    assert result.mode == "synthetic"
    assert result.output_root == tmp_path.resolve()
    assert (tmp_path / "normalized").is_dir()


def test_dispatcher_accepts_a_mode_checked_external_adapter(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")

    class FixtureAdapter:
        mode = "synthetic"

        def ingest(self, config: ProjectConfig, output_root: str | Path) -> IngestionResult:
            assert config is base
            return ingest_synthetic(base, output_root)

    result = ingest_from_config(base, tmp_path, adapter=FixtureAdapter())

    assert result.mode == "synthetic"


def test_registry_dispatches_a_configured_third_party_adapter(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    config = replace(base, data=replace(base.data, mode="fixture_vendor"))
    calls: list[tuple[ProjectConfig, Path]] = []

    class FixtureVendorAdapter:
        mode = "fixture_vendor"

        def ingest(self, config: ProjectConfig, output_root: str | Path) -> IngestionResult:
            destination = Path(output_root)
            calls.append((config, destination))
            synthetic = ingest_synthetic(base, destination)
            return replace(synthetic, mode=self.mode)

    registry = DataAdapterRegistry()
    registry.register(FixtureVendorAdapter())

    result = ingest_from_config(config, tmp_path, registry=registry)

    assert result.mode == "fixture_vendor"
    assert calls == [(config, tmp_path)]


def test_registry_rejects_unknown_and_duplicate_modes_without_fallback(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    unknown = replace(base, data=replace(base.data, mode="unregistered_vendor"))
    registry = builtin_data_adapter_registry()

    with pytest.raises(IngestionError, match=r"no data adapter registered.*unregistered_vendor"):
        ingest_from_config(unknown, tmp_path, registry=registry)

    class DuplicateSyntheticAdapter:
        mode = "synthetic"

        def ingest(self, config: ProjectConfig, output_root: str | Path) -> IngestionResult:
            raise AssertionError("duplicate adapter must never be selected")

    with pytest.raises(IngestionError, match="already registered"):
        registry.register(DuplicateSyntheticAdapter())

    assert registry.modes == ("binance_rest", "synthetic")


def test_validation_only_reports_errors_without_mutating_input(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    generated = ingest_synthetic(
        replace(base, data=replace(base.data, symbols=("BTCUSDT",), events_per_symbol=2)),
        tmp_path,
    )
    records = generated.dataset("trades").table.to_pylist()
    records[1]["trade_id"] = records[0]["trade_id"]
    invalid = table_from_records("trades", records)
    before = invalid.to_pylist()

    summary = validate_configured_input(base, tables={"trades": invalid})

    assert not summary.passed
    assert summary.error_count >= 1
    assert invalid.to_pylist() == before


def test_discovered_validation_streams_parquet_without_legacy_eager_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            events_per_symbol=4,
            partition_root=tmp_path / "normalized",
        ),
    )
    generated = ingest_synthetic(config, tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("discovered validation must not materialize whole Parquet files")

    monkeypatch.setattr(ingestion_module.pq, "read_table", forbidden)
    monkeypatch.setattr(ingestion_module.pa, "concat_tables", forbidden)

    summary = validate_configured_input(config)

    assert summary.passed
    assert summary.rows_checked == generated.rows
    assert tuple(report.dataset for report in summary.reports) == (
        "book_observations",
        "trades",
    )
    assert summary.report_paths == ()


def test_discovered_validation_rejects_metadata_rows_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    stored_config = replace(
        base,
        data=replace(
            base.data,
            events_per_symbol=4,
            partition_root=tmp_path / "normalized",
        ),
    )
    ingest_synthetic(stored_config, tmp_path)
    bounded_config = replace(
        stored_config,
        data=replace(stored_config.data, events_per_symbol=1),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("row data must not be read after the metadata guard fails")

    monkeypatch.setattr(ingestion_module, "validate_batches", forbidden)

    with pytest.raises(IngestionError, match="above configured validation bound"):
        validate_configured_input(bounded_config)


def test_public_ingestion_uses_exchange_scales_retries_caps_and_preserves_raw(
    tmp_path: Path,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    config = replace(
        base,
        run=replace(base.run, evidence_tier="FULL_DATA"),
        data=replace(
            base.data,
            max_events_per_symbol=2,
            request_limit=2,
            max_retries=1,
            partition_root=tmp_path / "normalized",
        ),
    )
    start_ms = int(config.data.start.timestamp() * 1000)
    session = FakeSession(
        [
            FakeResponse(429, {"code": -1003}, headers={"Retry-After": "1"}),
            FakeResponse(200, _metadata("BTCUSDT", lot_size="0.00001")),
            FakeResponse(200, _trades("BTCUSDT", start_ms, "0.00002")),
            FakeResponse(200, _metadata("ETHUSDT", lot_size="0.0001")),
            FakeResponse(200, _trades("ETHUSDT", start_ms, "0.0002")),
        ]
    )
    sleeps: list[float] = []

    result = ingest_from_config(
        config,
        tmp_path,
        session=session,  # type: ignore[arg-type]
        sleep=sleeps.append,
        random_value=lambda: 1.0,
    )

    assert sleeps == [1.0]
    assert result.mode == "binance_rest"
    assert result.evidence_tier == "PUBLIC_SAMPLE_PARTIAL"
    assert result.validation.passed
    assert result.dataset("trades").rows == 4
    assert {item.symbol: item.rows for item in result.symbols} == {
        "BTCUSDT": 2,
        "ETHUSDT": 2,
    }
    assert all(not item.complete_range for item in result.symbols)
    assert [item.metadata.lot_size for item in result.symbols] == [
        Decimal("0.00001"),
        Decimal("0.0001"),
    ]
    dataset = result.dataset("trades")
    assert dataset.table is None
    with pytest.raises(IngestionError, match="materialization bound 3"):
        dataset.materialize(max_rows=3)
    rows = dataset.materialize(max_rows=4).to_pylist()
    assert [row["quantity_lots"] for row in rows] == [2, 2, 2, 2]
    assert len(result.raw_artifacts) == 4
    assert all(
        item.path.is_file() and item.manifest_path.is_file() for item in result.raw_artifacts
    )
    assert len(session.calls) == 5
    manifest = read_json(result.ingestion_manifest_path)
    assert manifest["all_requested_ranges_complete"] is False
    assert manifest["requested_evidence_tier"] == "FULL_DATA"
    assert manifest["evidence_tier"] == "PUBLIC_SAMPLE_PARTIAL"
    assert manifest["row_cap_per_symbol"] == 2
    assert {item["symbol"] for item in manifest["symbols"]} == {"BTCUSDT", "ETHUSDT"}
    assert all(item["raw_page_count"] == 1 for item in manifest["symbols"])
    assert all(item["stop_reason"] == "event_cap" for item in manifest["symbols"])
    assert all(item["last_raw_page_sha256"] for item in manifest["symbols"])
    assert len(manifest["raw_artifacts"]) == 4
    by_symbol = {item.symbol: item for item in result.symbols}
    for item in manifest["symbols"]:
        observed = by_symbol[item["symbol"]]
        terminal = observed.stream_summary
        stream_claim = item["stream_summary"]
        assert stream_claim == {
            "requested_start_ns": terminal.requested_start_ns,
            "requested_end_ns": terminal.requested_end_ns,
            "rows_yielded": terminal.rows_yielded,
            "raw_page_count": terminal.raw_page_count,
            "stop_reason": str(terminal.stop_reason),
            "complete_range": terminal.complete_range,
            "last_raw_page": {
                "path": str(terminal.last_raw_page.path.relative_to(result.output_root)),
                "manifest_path": str(
                    terminal.last_raw_page.manifest_path.relative_to(result.output_root)
                ),
                "sha256": terminal.last_raw_page.sha256,
                "request_uri": terminal.last_raw_page.request_uri,
                "row_count": terminal.last_raw_page.row_count,
            },
        }
    assert len(result.raw_artifacts) == len(result.symbols) + sum(
        item.stream_summary.raw_page_count for item in result.symbols
    )


def test_public_ingestion_is_one_shot_and_avoids_legacy_materializers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            max_events_per_symbol=2,
            request_limit=2,
            partition_root=tmp_path / "normalized",
        ),
    )
    start_ms = int(config.data.start.timestamp() * 1000)
    session = FakeSession(
        [
            FakeResponse(200, _metadata("BTCUSDT", lot_size="0.00001")),
            FakeResponse(200, _trades("BTCUSDT", start_ms, "0.00002")),
            FakeResponse(200, _metadata("ETHUSDT", lot_size="0.0001")),
            FakeResponse(200, _trades("ETHUSDT", start_ms, "0.0002")),
        ]
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("legacy eager path must not be called")

    original_stream = BinanceHistoricalTradeDownloader.stream
    stream_iterations: list[int] = []

    class OneShotStream:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.iterations = 0
            stream_iterations.append(0)
            self.index = len(stream_iterations) - 1

        def __iter__(self) -> OneShotStream:
            self.iterations += 1
            stream_iterations[self.index] = self.iterations
            if self.iterations != 1:
                raise AssertionError("stream was iterated more than once")
            return self

        def __next__(self) -> Any:
            return next(self.inner)

        @property
        def summary(self) -> Any:
            return self.inner.summary

    def one_shot_stream(
        downloader: BinanceHistoricalTradeDownloader, **kwargs: Any
    ) -> OneShotStream:
        return OneShotStream(original_stream(downloader, **kwargs))

    original_write = ingestion_module.write_partitioned_parquet
    write_calls: list[dict[str, Any]] = []

    def counted_write(batches: Iterable[Any], **kwargs: Any) -> DatasetWriteResult:
        write_calls.append(dict(kwargs))
        return original_write(batches, **kwargs)

    monkeypatch.setattr(BinanceHistoricalTradeDownloader, "download", forbidden)
    monkeypatch.setattr(BinanceHistoricalTradeDownloader, "stream", one_shot_stream)
    monkeypatch.setattr(ingestion_module, "validate_table", forbidden)
    monkeypatch.setattr(ingestion_module.pa, "concat_tables", forbidden)
    monkeypatch.setattr(ingestion_module, "write_partitioned_parquet", counted_write)

    result = ingest_public_trades(
        config,
        tmp_path,
        session=session,  # type: ignore[arg-type]
    )

    assert result.dataset("trades").table is None
    assert stream_iterations == [1, 1]
    assert len(write_calls) == 1
    assert write_calls[0]["max_input_batch_rows"] == config.data.request_limit
    assert len(result.dataset("trades").storage.artifacts) == 2


def test_public_incremental_quality_matches_eager_rules_across_pages(
    tmp_path: Path,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            symbols=("BTCUSDT",),
            max_events_per_symbol=2,
            request_limit=1,
            partition_root=tmp_path / "normalized",
        ),
    )
    start_ms = int(config.data.start.timestamp() * 1000)
    trades = _trades("BTCUSDT", start_ms, "0.00002")
    trades[1]["T"] = start_ms + config.quality.max_silence_ms + 1
    session = FakeSession(
        [
            FakeResponse(200, _metadata("BTCUSDT", lot_size="0.00001")),
            FakeResponse(200, [trades[0]]),
            FakeResponse(200, [trades[1]]),
        ]
    )

    result = ingest_public_trades(
        config,
        tmp_path,
        session=session,  # type: ignore[arg-type]
    )
    materialized = result.dataset("trades").materialize(max_rows=2)
    eager = validate_table(
        materialized,
        "trades",
        max_spread_bps=config.quality.max_spread_bps,
        max_silence_ns=config.quality.max_silence_ms * 1_000_000,
    )
    incremental = result.validation.report_for("trades")
    write_order = tuple(item.data_path for item in result.dataset("trades").storage.artifacts)
    assert write_order != tuple(sorted(write_order))
    discovered = validate_configured_input(config).report_for("trades")
    result.dataset("trades").storage.manifest_path.unlink()
    legacy_discovered = validate_configured_input(config).report_for("trades")

    assert incremental.rows_checked == eager.rows_checked == 2
    assert incremental.error_count == eager.error_count == 0
    assert incremental.warning_count == eager.warning_count == 1
    assert incremental.findings == eager.findings
    assert discovered.findings == incremental.findings
    assert legacy_discovered.findings == incremental.findings
    assert incremental.findings[0].rule_id == "temporal.long_silence"
    assert incremental.findings[0].row_index == 1


def test_public_quality_gate_preserves_raw_normalized_and_manifest_evidence(
    tmp_path: Path,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            symbols=("BTCUSDT",),
            max_events_per_symbol=2,
            request_limit=2,
            partition_root=tmp_path / "normalized",
        ),
    )
    start_ms = int(config.data.start.timestamp() * 1000)
    session = FakeSession(
        [
            FakeResponse(200, _metadata("BTCUSDT", lot_size="0.00001")),
            FakeResponse(200, _trades("BTCUSDT", start_ms, "0.00000")),
        ]
    )

    with pytest.raises(DataQualityGateError) as captured:
        ingest_public_trades(
            config,
            tmp_path,
            session=session,  # type: ignore[arg-type]
        )

    assert captured.value.summary.error_count == 2
    assert list((tmp_path / "raw").rglob("*.json"))
    parquet_paths = list((tmp_path / "normalized").rglob("*.parquet"))
    assert parquet_paths
    manifest_paths = list((tmp_path / "_ingestion_manifests").glob("*.json"))
    assert len(manifest_paths) == 1
    manifest = read_json(manifest_paths[0])
    assert manifest["normalized_datasets"][0]["rows"] == 2
    assert len(manifest["raw_artifacts"]) == 2
    quality_paths = list((tmp_path / "quality").glob("trades.validation-*.json"))
    assert len(quality_paths) == 1
    quality = read_json(quality_paths[0])
    assert quality["summary"] == {"errors": 2, "warnings": 0}
    findings_paths = list((tmp_path / "quality").glob("trades.findings-*.jsonl"))
    assert len(findings_paths) == 1
    findings = findings_paths[0].read_text().splitlines()
    assert len(findings) == 2
    quality_claims = manifest["quality_artifacts"]
    assert len(quality_claims) == 2
    for claim in quality_claims:
        path = tmp_path / claim["path"]
        assert claim["sha256"] == sha256_file(path)
        assert claim["bytes"] == path.stat().st_size


def test_discovered_validation_rejects_conflicting_manifest_write_orders(
    tmp_path: Path,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            symbols=("BTCUSDT",),
            max_events_per_symbol=2,
            request_limit=1,
            partition_root=tmp_path / "normalized",
        ),
    )
    start_ms = int(config.data.start.timestamp() * 1000)
    trades = _trades("BTCUSDT", start_ms, "0.00002")
    session = FakeSession(
        [
            FakeResponse(200, _metadata("BTCUSDT", lot_size="0.00001")),
            FakeResponse(200, [trades[0]]),
            FakeResponse(200, [trades[1]]),
        ]
    )
    result = ingest_public_trades(
        config,
        tmp_path,
        session=session,  # type: ignore[arg-type]
    )
    storage = result.dataset("trades").storage
    payload = read_json(storage.manifest_path)
    payload["artifacts"] = list(reversed(payload["artifacts"]))
    for ordinal, artifact in enumerate(payload["artifacts"]):
        artifact["write_ordinal"] = ordinal
    write_json(storage.manifest_path.parent / "trades.manifest-conflict.json", payload)

    with pytest.raises(IngestionError, match="write_ordinal"):
        validate_configured_input(config)


def test_discovered_validation_rejects_same_row_count_parquet_tampering(
    tmp_path: Path,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "smoke.toml")
    config = replace(
        base,
        data=replace(
            base.data,
            events_per_symbol=4,
            partition_root=tmp_path / "normalized",
        ),
    )
    generated = ingest_synthetic(config, tmp_path)
    artifact = generated.dataset("trades").storage.artifacts[0]
    table = pq.read_table(artifact.data_path)
    prices = table.column("price")
    replacement = pa.chunked_array(
        [pa.array([float(prices[0].as_py()) + 1.0, *prices.slice(1).to_pylist()])],
        type=pa.float64(),
    )
    tampered = table.set_column(table.schema.get_field_index("price"), "price", replacement)
    pq.write_table(tampered, artifact.data_path)

    with pytest.raises(IngestionError, match="data_sha256 checksum mismatch"):
        validate_configured_input(config)


def test_public_ingestion_requires_bounded_supported_universe(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    missing_cap = replace(base, data=replace(base.data, max_events_per_symbol=None))
    unsupported = replace(base, data=replace(base.data, symbols=("BNBUSDT",)))

    with pytest.raises(IngestionError, match="max_events_per_symbol"):
        ingest_public_trades(missing_cap, tmp_path)
    with pytest.raises(IngestionError, match="unsupported public-sample symbols"):
        ingest_public_trades(unsupported, tmp_path)


def test_public_ingestion_rejects_empty_requested_coverage_but_preserves_raw(
    tmp_path: Path,
) -> None:
    base = load_config(PROJECT_ROOT / "configs" / "public_sample.toml")
    config = replace(base, data=replace(base.data, symbols=("BTCUSDT",)))
    session = FakeSession(
        [
            FakeResponse(200, _metadata("BTCUSDT", lot_size="0.00001")),
            FakeResponse(200, []),
        ]
    )

    with pytest.raises(IngestionError, match="no aggregate trades"):
        ingest_public_trades(config, tmp_path, session=session)  # type: ignore[arg-type]

    assert list((tmp_path / "raw").rglob("*.json"))
