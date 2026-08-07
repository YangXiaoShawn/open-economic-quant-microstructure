from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from microstructure.data.schemas import get_schema
from microstructure.data.storage import (
    StorageError,
    write_partitioned_parquet,
    write_source_manifest,
)
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.provenance import read_json, sha256_file


def test_partitioned_parquet_is_streamed_content_addressed_and_manifested(
    tmp_path: Path,
) -> None:
    start_ns = 1_704_153_600_000_000_000
    data = generate_synthetic_market(
        symbols=("BTCUSDT", "ETHUSDT"),
        events_per_symbol=7,
        start_ts_ns=start_ns,
        seed=11,
    )
    result = write_partitioned_parquet(
        data.trades.to_batches(max_chunksize=4),
        root=tmp_path,
        dataset="trades",
        schema_name="trades",
        source="synthetic_v1",
        requested_start_ns=start_ns,
        requested_end_ns=start_ns + 1_000_000_000,
        max_rows_per_file=3,
        downloaded_at_utc="2026-08-07T12:00:00Z",
    )

    assert result.rows == 14
    assert result.artifacts
    assert result.manifest_path.is_file()
    assert result.manifest_sha256 == sha256_file(result.manifest_path)
    assert {item.symbol for item in result.artifacts} == {"BTCUSDT", "ETHUSDT"}
    for artifact in result.artifacts:
        assert artifact.data_path.is_file()
        assert artifact.rows <= 3
        assert "symbol-" + artifact.symbol in str(artifact.data_path)
        assert artifact.data_sha256 == sha256_file(artifact.data_path)
        assert pq.read_schema(artifact.data_path).equals(get_schema("trades"), check_metadata=True)
        manifest = read_json(artifact.manifest_path)
        assert manifest["source"] == "synthetic_v1"
        assert manifest["downloaded_at_utc"] == "2026-08-07T12:00:00Z"
        assert manifest["schema_version"] == "1.0.0"
        assert manifest["checksum"]["value"] == artifact.data_sha256
        assert manifest["requested_range_ns"]["start"] == start_ns
        assert manifest["write_ordinal"] == artifact.write_ordinal
        assert manifest["observed_range_ns"] == {
            "start": artifact.observed_start_ns,
            "end_inclusive": artifact.observed_end_inclusive_ns,
        }

    dataset_manifest = read_json(result.manifest_path)
    assert [item["write_ordinal"] for item in dataset_manifest["artifacts"]] == list(
        range(len(result.artifacts))
    )


def test_same_normalized_content_reuses_immutable_parquet(tmp_path: Path) -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=3,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=99,
    )
    kwargs = {
        "root": tmp_path,
        "dataset": "trades",
        "schema_name": "trades",
        "source": "synthetic_v1",
        "downloaded_at_utc": "2026-08-07T12:00:00Z",
    }
    first = write_partitioned_parquet([data.trades], **kwargs)
    second = write_partitioned_parquet([data.trades], **kwargs)

    assert [item.data_path for item in first.artifacts] == [
        item.data_path for item in second.artifacts
    ]
    assert first.manifest_path == second.manifest_path
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_raw_source_manifest_contains_required_lineage_and_checksum(tmp_path: Path) -> None:
    raw = tmp_path / "page.json"
    raw.write_bytes(b'[{"a":1}]')

    manifest_path, manifest_sha = write_source_manifest(
        raw,
        source="binance_spot_public_api",
        source_uri="https://data-api.binance.vision/api/v3/aggTrades?symbol=BTCUSDT",
        downloaded_at_utc="2026-08-07T12:00:00Z",
        requested_start_ns=100,
        requested_end_ns=200,
        response_headers={"ETag": "abc", "X-MBX-USED-WEIGHT-1M": "4"},
    )
    manifest = read_json(manifest_path)

    assert manifest_sha == sha256_file(manifest_path)
    assert manifest["artifact_kind"] == "raw_source"
    assert manifest["checksum"]["value"] == sha256_file(raw)
    assert manifest["requested_range_ns"] == {"start": 100, "end_exclusive": 200}
    assert manifest["response_headers"]["ETag"] == "abc"


def test_partition_writer_rejects_oversized_input_before_materializing_it(
    tmp_path: Path,
) -> None:
    data = generate_synthetic_market(
        symbols=("BTCUSDT",),
        events_per_symbol=3,
        start_ts_ns=1_704_153_600_000_000_000,
        seed=101,
    )

    with pytest.raises(StorageError, match="above the bounded-memory limit 2"):
        write_partitioned_parquet(
            iter((data.trades,)),
            root=tmp_path,
            dataset="trades",
            schema_name="trades",
            source="synthetic_v1",
            max_input_batch_rows=2,
        )

    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.manifest-*.json"))
