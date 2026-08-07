from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from microstructure.config import ProjectConfig, load_config
from microstructure.data.schemas import table_from_records
from microstructure.ingestion import (
    IngestionError,
    IngestionResult,
    ingest_from_config,
    ingest_public_trades,
    ingest_synthetic,
    validate_configured_input,
)
from microstructure.provenance import read_json, sha256_file

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

    result = ingest_public_trades(
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
    rows = result.dataset("trades").table.to_pylist()
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
    assert len(manifest["raw_artifacts"]) == 4


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
