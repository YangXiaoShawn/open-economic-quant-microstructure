from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import microstructure.m8_acquisition as m8_acquisition
from microstructure.data.binance import (
    BinanceHTTPError,
    BinanceMetadataContractError,
    BinancePublicClient,
    BinanceResponseSizeLimitError,
    RetryPolicy,
)
from microstructure.data.binance_archive import (
    BinanceArchiveClient,
    BinanceArchiveContractError,
    BinanceArchiveHTTPError,
    BinanceArchivePayloadError,
)
from microstructure.data.evidence_budget import EvidenceBudgetExceeded, RetainedEvidenceBudget
from microstructure.m8_acquisition import (
    M8AcquisitionError,
    M8AcquisitionFailureResult,
    M8AcquisitionResult,
    acquire_m8_archives,
    copy_m8_acquisition_into,
    read_m8_acquisition_failure,
    read_m8_acquisition_manifest,
    verify_m8_acquisition_manifest,
)
from microstructure.m8_config import M8StudyConfig, load_m8_config

_ORIGINAL_ZIPFILE_OPEN = zipfile.ZipFile.open


@dataclass
class _FakeResponse:
    url: str
    content: bytes
    status_code: int = 200

    @property
    def headers(self) -> Mapping[str, str]:
        return {
            "Content-Length": str(len(self.content)),
            "Content-Type": "application/octet-stream",
        }

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def close(self) -> None:
        return None


class _MetadataSession:
    def __init__(self, bodies: Mapping[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        params: Mapping[str, str | int],
        timeout: float,
        stream: bool = False,
    ) -> _FakeResponse:
        del timeout, stream
        symbol = str(params["symbol"])
        request_uri = f"{url}?symbol={symbol}"
        self.calls.append(request_uri)
        return _FakeResponse(request_uri, self.bodies[symbol])


class _ArchiveSession:
    def __init__(self, bodies: Mapping[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

    def get(self, url: str, *, timeout: float, stream: bool) -> _FakeResponse:
        del timeout, stream
        self.calls.append(url)
        return _FakeResponse(url, self.bodies[url])


class _StatusArchiveSession:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.calls: list[str] = []

    def get(self, url: str, *, timeout: float, stream: bool) -> _FakeResponse:
        del timeout, stream
        self.calls.append(url)
        return _FakeResponse(url, b"declared object unavailable", self.status_code)


def _exchange_info(symbol: str) -> bytes:
    base_asset = symbol.removesuffix("USDT")
    return json.dumps(
        {
            "timezone": "UTC",
            "serverTime": 1_704_067_200_000,
            "symbols": [
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "baseAsset": base_asset,
                    "quoteAsset": "USDT",
                    "filters": [
                        {
                            "filterType": "PRICE_FILTER",
                            "minPrice": "0.01000000",
                            "maxPrice": "1000000.00000000",
                            "tickSize": "0.01000000",
                        },
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.00010000",
                            "maxQty": "100000.00000000",
                            "stepSize": "0.00010000",
                        },
                    ],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _zip_bytes(member_name: str, marker: str) -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            member_name,
            (
                "agg_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,buyer_maker\n"
                f"1,100.00,0.1000,1,1,1704067200000,{marker}\n"
            ).encode(),
        )
    return destination.getvalue()


def _archive_responses(config: M8StudyConfig) -> dict[str, bytes]:
    responses: dict[str, bytes] = {}
    for period in config.periods:
        for symbol in config.study.symbols:
            archive_name = f"{symbol}-aggTrades-{period.date.isoformat()}.zip"
            member_name = archive_name.removesuffix(".zip") + ".csv"
            archive_uri = (
                f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{archive_name}"
            )
            archive_body = _zip_bytes(member_name, f"{symbol}-{period.role}")
            digest = hashlib.sha256(archive_body).hexdigest()
            responses[archive_uri] = archive_body
            responses[f"{archive_uri}.CHECKSUM"] = f"{digest}  {archive_name}\n".encode()
    return responses


def _acquire_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> M8AcquisitionResult:
    config = load_m8_config(Path(__file__).parents[1] / "configs/m8_multidate_trade_study.toml")
    metadata_session = _MetadataSession(
        {symbol: _exchange_info(symbol) for symbol in config.study.symbols}
    )
    archive_session = _ArchiveSession(_archive_responses(config))
    client_budget_ids: list[int] = []

    def public_client_factory(**kwargs: Any) -> BinancePublicClient:
        client_budget_ids.append(id(kwargs["retained_evidence_budget"]))
        return BinancePublicClient(
            session=metadata_session,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_retries=0),
            **kwargs,
        )

    def archive_client_factory(**kwargs: Any) -> BinanceArchiveClient:
        client_budget_ids.append(id(kwargs["retained_evidence_budget"]))
        return BinanceArchiveClient(
            session=archive_session,
            retry_policy=RetryPolicy(max_retries=0),
            **kwargs,
        )

    monkeypatch.setattr(m8_acquisition, "BinancePublicClient", public_client_factory)
    monkeypatch.setattr(m8_acquisition, "BinanceArchiveClient", archive_client_factory)
    member_open_calls: list[str] = []

    def forbidden_member_open(*args: object, **kwargs: object) -> object:
        del args, kwargs
        member_open_calls.append("opened")
        raise AssertionError("raw acquisition must never open a ZIP member")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_member_open)
    result = acquire_m8_archives(config, tmp_path / "authority")
    assert member_open_calls == []
    assert len(client_budget_ids) == 2
    assert len(set(client_budget_ids)) == 1
    assert len(metadata_session.calls) == 2
    assert len(archive_session.calls) == 16
    return result


def _install_status_clients(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int,
) -> tuple[M8StudyConfig, _StatusArchiveSession]:
    config = load_m8_config(Path(__file__).parents[1] / "configs/m8_multidate_trade_study.toml")
    metadata_session = _MetadataSession(
        {symbol: _exchange_info(symbol) for symbol in config.study.symbols}
    )
    archive_session = _StatusArchiveSession(status_code)

    def public_client_factory(**kwargs: Any) -> BinancePublicClient:
        return BinancePublicClient(
            session=metadata_session,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_retries=0),
            **kwargs,
        )

    def archive_client_factory(**kwargs: Any) -> BinanceArchiveClient:
        return BinanceArchiveClient(
            session=archive_session,
            retry_policy=RetryPolicy(max_retries=0),
            **kwargs,
        )

    monkeypatch.setattr(m8_acquisition, "BinancePublicClient", public_client_factory)
    monkeypatch.setattr(m8_acquisition, "BinanceArchiveClient", archive_client_factory)
    return config, archive_session


def _install_metadata_body_clients(
    monkeypatch: pytest.MonkeyPatch,
    bodies: Mapping[str, bytes],
) -> tuple[M8StudyConfig, _MetadataSession, _ArchiveSession]:
    config = load_m8_config(Path(__file__).parents[1] / "configs/m8_multidate_trade_study.toml")
    metadata_session = _MetadataSession(bodies)
    archive_session = _ArchiveSession(_archive_responses(config))

    def public_client_factory(**kwargs: Any) -> BinancePublicClient:
        return BinancePublicClient(
            session=metadata_session,  # type: ignore[arg-type]
            retry_policy=RetryPolicy(max_retries=0),
            **kwargs,
        )

    def archive_client_factory(**kwargs: Any) -> BinanceArchiveClient:
        return BinanceArchiveClient(
            session=archive_session,
            retry_policy=RetryPolicy(max_retries=0),
            **kwargs,
        )

    monkeypatch.setattr(m8_acquisition, "BinancePublicClient", public_client_factory)
    monkeypatch.setattr(m8_acquisition, "BinanceArchiveClient", archive_client_factory)
    return config, metadata_session, archive_session


def test_raw_only_acquisition_is_complete_content_addressed_and_budget_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _acquire_fixture(tmp_path, monkeypatch)

    assert result.metadata_count == 2
    assert result.archive_count == 8
    assert result.manifest_path.name == (
        f"m8-acquisition.manifest-{result.manifest_sha256[:20]}.json"
    )
    assert result.manifest.copied_from_manifest_sha256 is None
    assert result.manifest.content_identity_sha256 == result.manifest.evidence_set_sha256
    assert all(not item.csv_member_opened for item in result.manifest.archives)
    assert all(not item.economic_fields_inspected for item in result.manifest.archives)
    assert len(result.manifest.retained_artifacts) == 36
    assert sum(item.bytes for item in result.manifest.retained_artifacts) == (
        result.total_raw_evidence_bytes
    )
    raw_budget = RetainedEvidenceBudget(
        result.output_root / "raw",
        result.manifest.config.study.max_total_download_bytes,
    )
    assert raw_budget.used_bytes == result.total_raw_evidence_bytes
    assert result.manifest_path.stat().st_size not in {
        item.bytes for item in result.manifest.retained_artifacts
    } or result.manifest_path.relative_to(result.output_root).as_posix() not in {
        item.path for item in result.manifest.retained_artifacts
    }
    reconstructed = result.manifest.archive_descriptor_for("BTCUSDT", "2024-01-05").reconstruct()
    assert reconstructed.request.symbol == "BTCUSDT"
    assert reconstructed.request.date.isoformat() == "2024-01-05"
    assert reconstructed.declared_uncompressed_bytes > 0
    assert verify_m8_acquisition_manifest is read_m8_acquisition_manifest


def test_reader_rejects_raw_tamper_without_opening_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _acquire_fixture(tmp_path, monkeypatch)
    checksum = result.manifest.archives[0].checksum_path
    checksum.write_bytes(checksum.read_bytes() + b"x")

    with pytest.raises(M8AcquisitionError, match="byte count changed"):
        read_m8_acquisition_manifest(
            result.manifest_path,
            expected_sha256=result.manifest_sha256,
            config=result.manifest.config,
        )


def test_held_out_descriptor_requires_guard_and_runs_it_immediately_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _acquire_fixture(tmp_path, monkeypatch)
    held_out = result.manifest.archive_descriptor_for("BTCUSDT", "2024-01-05").reconstruct()
    development = result.manifest.archive_descriptor_for("BTCUSDT", "2024-01-03").reconstruct()
    assert held_out.requires_member_open_guard is True
    assert development.requires_member_open_guard is False
    events: list[str] = []

    def tracked_open(
        archive: zipfile.ZipFile,
        name: str | zipfile.ZipInfo,
        mode: str = "r",
        pwd: bytes | None = None,
        *,
        force_zip64: bool = False,
    ) -> Any:
        events.append("open")
        return _ORIGINAL_ZIPFILE_OPEN(
            archive,
            name,
            mode=mode,
            pwd=pwd,
            force_zip64=force_zip64,
        )

    monkeypatch.setattr(zipfile.ZipFile, "open", tracked_open)
    unguarded = held_out.iter_normalized_batches(batch_rows=1)
    with pytest.raises(BinanceArchivePayloadError, match="requires a member-open authority guard"):
        next(unguarded)
    assert events == []

    guarded = held_out.iter_normalized_batches(
        batch_rows=1,
        before_member_open=lambda: events.append("guard"),
    )
    with pytest.raises(BinanceArchivePayloadError):
        next(guarded)
    assert events == ["guard", "open"]


def test_copy_is_self_contained_source_equivalent_and_has_no_extra_raw_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _acquire_fixture(tmp_path, monkeypatch)
    copied = copy_m8_acquisition_into(result.manifest, tmp_path / "bundle-input")

    assert copied.copied_from_manifest_sha256 == result.manifest_sha256
    assert copied.content_identity_sha256 == result.manifest.content_identity_sha256
    assert copied.retained_artifacts == result.manifest.retained_artifacts
    assert copied.total_raw_evidence_bytes == result.total_raw_evidence_bytes
    assert copied.total_accepted_zip_bytes == result.manifest.total_accepted_zip_bytes
    copied_raw_files = tuple(path for path in (copied.root / "raw").rglob("*") if path.is_file())
    assert len(copied_raw_files) == len(copied.retained_artifacts)
    assert all(path.resolve().is_relative_to(copied.root) for path in copied_raw_files)
    assert len(tuple((copied.root / "_manifests").glob("*.json"))) == 1
    assert copied.path.name != result.manifest_path.name
    for source_item, copied_item in zip(
        result.manifest.archives,
        copied.archives,
        strict=True,
    ):
        assert source_item.archive_path != copied_item.archive_path
        assert source_item.archive_sha256 == copied_item.archive_sha256


def test_distinct_copy_budget_has_exact_boundary() -> None:
    m8_acquisition._assert_distinct_copy_budget(5, 10)
    with pytest.raises(M8AcquisitionError, match="distinct self-contained copy"):
        m8_acquisition._assert_distinct_copy_budget(6, 10)


def test_repeated_acquisition_excludes_manifest_indexes_from_raw_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _acquire_fixture(tmp_path, monkeypatch)
    second = acquire_m8_archives(first.manifest.config, first.output_root)

    raw_files = tuple(path for path in (second.output_root / "raw").rglob("*") if path.is_file())
    assert second.total_raw_evidence_bytes == sum(path.stat().st_size for path in raw_files)
    assert second.total_raw_evidence_bytes == sum(
        item.bytes for item in second.manifest.retained_artifacts
    )
    manifest_paths = {
        path.relative_to(second.output_root).as_posix()
        for path in (second.output_root / "_manifests").glob("*.json")
    }
    assert len(manifest_paths) == 2
    assert manifest_paths.isdisjoint({item.path for item in second.manifest.retained_artifacts})
    assert second.total_raw_evidence_bytes >= first.total_raw_evidence_bytes


def test_zip_preflight_rejects_many_entries_before_zipfile_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "many.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("first.csv", b"1")
        archive.writestr("second.csv", b"2")
    parser_calls: list[str] = []

    def forbidden_parser(*args: object, **kwargs: object) -> object:
        del args, kwargs
        parser_calls.append("called")
        raise AssertionError("unbounded ZIP parser must not run")

    monkeypatch.setattr(m8_acquisition.zipfile, "ZipFile", forbidden_parser)
    with pytest.raises(M8AcquisitionError, match="one single-disk member"):
        m8_acquisition._zip_directory_member(destination, "first.csv", 1024)
    assert parser_calls == []


def test_zip_preflight_rejects_oversized_directory_claim_before_zipfile_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "oversized-directory.zip"
    destination.write_bytes(_zip_bytes("expected.csv", "marker"))
    body = bytearray(destination.read_bytes())
    eocd_offset = body.rfind(m8_acquisition._EOCD_SIGNATURE)
    values = list(m8_acquisition._EOCD_STRUCT.unpack_from(body, eocd_offset))
    values[5] = m8_acquisition._MAX_ZIP_DIRECTORY_BYTES + 1
    m8_acquisition._EOCD_STRUCT.pack_into(body, eocd_offset, *values)
    destination.write_bytes(body)
    parser_calls: list[str] = []

    def forbidden_parser(*args: object, **kwargs: object) -> object:
        del args, kwargs
        parser_calls.append("called")
        raise AssertionError("unbounded ZIP parser must not run")

    monkeypatch.setattr(m8_acquisition.zipfile, "ZipFile", forbidden_parser)
    with pytest.raises(M8AcquisitionError, match="central directory exceeds"):
        m8_acquisition._zip_directory_member(destination, "expected.csv", 1024)
    assert parser_calls == []


def test_zip_preflight_rejects_malformed_eocd_before_zipfile_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "malformed-eocd.zip"
    body = bytearray(_zip_bytes("expected.csv", "marker"))
    eocd_offset = body.rfind(m8_acquisition._EOCD_SIGNATURE)
    assert eocd_offset >= 0
    body[eocd_offset : eocd_offset + 4] = b"NOPE"
    destination.write_bytes(body)
    parser_calls: list[str] = []

    def forbidden_parser(*args: object, **kwargs: object) -> object:
        del args, kwargs
        parser_calls.append("called")
        raise AssertionError("unbounded ZIP parser must not run")

    monkeypatch.setattr(m8_acquisition.zipfile, "ZipFile", forbidden_parser)
    with pytest.raises(M8AcquisitionError, match="end-of-directory record is missing"):
        m8_acquisition._zip_directory_member(destination, "expected.csv", 1024)
    assert parser_calls == []


def test_exact_404_publishes_verified_raw_only_failure_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, archive_session = _install_status_clients(monkeypatch, status_code=404)
    member_opens: list[str] = []

    def forbidden_member_open(*args: object, **kwargs: object) -> object:
        del args, kwargs
        member_opens.append("opened")
        raise AssertionError("deterministic acquisition failure must not open a CSV member")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_member_open)
    result = acquire_m8_archives(config, tmp_path / "authority")

    assert isinstance(result, M8AcquisitionFailureResult)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.reason_code == "DECLARED_OBJECT_UNAVAILABLE"
    assert result.failed_symbol == "BTCUSDT"
    assert result.failed_date is not None
    assert result.failed_date.isoformat() == "2024-01-03"
    assert result.failed_role == "train"
    assert result.completed_count == 2
    assert result.remaining_count == 7
    assert result.retained_artifact_count == 6
    assert result.terminal_path.read_bytes() == b"terminal\n"
    assert result.attempt_dir.name.endswith(result.attempt_manifest_sha256[:20])
    assert member_opens == []
    assert len(archive_session.calls) == 1

    verified = read_m8_acquisition_failure(
        result.attempt_manifest_path,
        expected_sha256=result.attempt_manifest_sha256,
        config=config,
    )
    assert verified.reason_code == result.reason_code
    assert verified.retained_inventory_sha256 == result.retained_inventory_sha256
    assert verified.total_raw_evidence_bytes == result.total_raw_evidence_bytes
    assert verified.checksums_sha256 == result.checksums_sha256


def test_nontrading_exchange_info_publishes_verified_metadata_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_m8_config(Path(__file__).parents[1] / "configs/m8_multidate_trade_study.toml")
    bodies = {symbol: _exchange_info(symbol) for symbol in config.study.symbols}
    payload = json.loads(bodies["BTCUSDT"])
    payload["symbols"][0]["status"] = "HALT"
    bodies["BTCUSDT"] = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    config, metadata_session, archive_session = _install_metadata_body_clients(
        monkeypatch,
        bodies,
    )
    member_opens: list[str] = []

    def forbidden_member_open(*args: object, **kwargs: object) -> object:
        del args, kwargs
        member_opens.append("opened")
        raise AssertionError("metadata failure must not open a CSV member")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_member_open)
    result = acquire_m8_archives(config, tmp_path / "authority")

    assert isinstance(result, M8AcquisitionFailureResult)
    assert result.reason_code == "METADATA_CONTRACT"
    assert result.failed_symbol == "BTCUSDT"
    assert result.failed_date is None
    assert result.failed_role == "metadata"
    assert result.completed_count == 0
    assert result.remaining_count == 9
    assert result.retained_artifact_count == 2
    assert result.terminal_path.read_bytes() == b"terminal\n"
    assert metadata_session.calls == [
        "https://data-api.binance.vision/api/v3/exchangeInfo?symbol=BTCUSDT"
    ]
    assert archive_session.calls == []
    assert member_opens == []

    verified = read_m8_acquisition_failure(
        result.attempt_manifest_path,
        expected_sha256=result.attempt_manifest_sha256,
        config=config,
    )
    assert verified.reason_code == "METADATA_CONTRACT"
    assert verified.retained_artifacts == result.manifest.retained_artifacts


@pytest.mark.parametrize(
    ("filter_type", "field", "value"),
    [
        ("PRICE_FILTER", "minPrice", "NaN"),
        ("LOT_SIZE", "maxQty", "Infinity"),
        ("PRICE_FILTER", "tickSize", "not-a-decimal"),
    ],
)
def test_invalid_exchange_info_filter_number_is_terminal_metadata_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filter_type: str,
    field: str,
    value: str,
) -> None:
    config = load_m8_config(Path(__file__).parents[1] / "configs/m8_multidate_trade_study.toml")
    bodies = {symbol: _exchange_info(symbol) for symbol in config.study.symbols}
    payload = json.loads(bodies["BTCUSDT"])
    filters = payload["symbols"][0]["filters"]
    next(item for item in filters if item["filterType"] == filter_type)[field] = value
    bodies["BTCUSDT"] = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    config, _, archive_session = _install_metadata_body_clients(monkeypatch, bodies)

    result = acquire_m8_archives(config, tmp_path / "authority")

    assert isinstance(result, M8AcquisitionFailureResult)
    assert result.reason_code == "METADATA_CONTRACT"
    assert result.failed_role == "metadata"
    assert archive_session.calls == []
    verified = read_m8_acquisition_failure(
        result.attempt_manifest_path,
        expected_sha256=result.attempt_manifest_sha256,
        config=config,
    )
    assert verified.reason_code == "METADATA_CONTRACT"


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("metadata permission denied"),
        OSError("metadata disk fault"),
        RuntimeError("metadata implementation fault"),
        M8AcquisitionError("metadata hash identity fault"),
    ],
)
def test_metadata_verification_system_or_integrity_fault_never_terminalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    config = load_m8_config(Path(__file__).parents[1] / "configs/m8_multidate_trade_study.toml")
    bodies = {symbol: _exchange_info(symbol) for symbol in config.study.symbols}
    config, _, archive_session = _install_metadata_body_clients(monkeypatch, bodies)

    def fail_verification(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise failure

    monkeypatch.setattr(m8_acquisition, "_verify_metadata_files", fail_verification)
    output_root = tmp_path / "authority"
    with pytest.raises(M8AcquisitionError, match=str(failure)):
        acquire_m8_archives(config, output_root)

    assert archive_session.calls == []
    assert not (output_root / "_attempts").exists()
    assert not list(output_root.rglob("INSUFFICIENT_DATA"))


def test_earlier_failure_attempt_remains_verifiable_after_later_raw_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _install_status_clients(monkeypatch, status_code=404)
    output_root = tmp_path / "authority"
    first = acquire_m8_archives(config, output_root)
    second = acquire_m8_archives(config, output_root)
    assert isinstance(first, M8AcquisitionFailureResult)
    assert isinstance(second, M8AcquisitionFailureResult)

    verified_first = read_m8_acquisition_failure(
        first.attempt_manifest_path,
        expected_sha256=first.attempt_manifest_sha256,
        config=config,
    )
    assert verified_first.retained_artifacts == first.manifest.retained_artifacts
    assert verified_first.reason_code == "DECLARED_OBJECT_UNAVAILABLE"


def test_retry_exhausted_http_failure_remains_retryable_and_publishes_no_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, archive_session = _install_status_clients(monkeypatch, status_code=503)
    output_root = tmp_path / "authority"

    with pytest.raises(M8AcquisitionError, match="failed closed"):
        acquire_m8_archives(config, output_root)

    assert len(archive_session.calls) == 1
    assert not (output_root / "_attempts").exists()
    assert not list(output_root.rglob("INSUFFICIENT_DATA"))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BinanceHTTPError("missing", status_code=404), "DECLARED_OBJECT_UNAVAILABLE"),
        (BinanceArchiveHTTPError("gone", status_code=410), "DECLARED_OBJECT_UNAVAILABLE"),
        (BinanceMetadataContractError("bad metadata"), "METADATA_CONTRACT"),
        (BinanceResponseSizeLimitError("too large"), "RESPONSE_SIZE_LIMIT"),
        (
            BinanceArchiveContractError("bad checksum", reason_code="CHECKSUM_CONTRACT"),
            "CHECKSUM_CONTRACT",
        ),
        (
            BinanceArchiveContractError("bad zip", reason_code="ZIP_CONTRACT"),
            "ZIP_CONTRACT",
        ),
        (EvidenceBudgetExceeded("budget"), "TOTAL_EVIDENCE_BUDGET"),
        (BinanceHTTPError("retry", status_code=503, retry_exhausted=True), None),
        (PermissionError("denied"), None),
        (OSError("disk fault"), None),
    ],
)
def test_acquisition_failure_classification_is_typed_and_stable(
    error: BaseException,
    expected: str | None,
) -> None:
    assert m8_acquisition._deterministic_failure_reason(error) == expected


def test_success_manifest_is_published_only_after_recursive_raw_durability_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_fsync_tree = m8_acquisition._fsync_tree
    original_write_manifest = m8_acquisition._write_manifest_payload

    def tracked_fsync_tree(path: Path) -> None:
        events.append(f"barrier:{path.name}")
        original_fsync_tree(path)

    def tracked_write_manifest(*args: object, **kwargs: object) -> tuple[Path, str]:
        events.append("publish:manifest")
        return original_write_manifest(*args, **kwargs)

    monkeypatch.setattr(m8_acquisition, "_fsync_tree", tracked_fsync_tree)
    monkeypatch.setattr(m8_acquisition, "_write_manifest_payload", tracked_write_manifest)
    _acquire_fixture(tmp_path, monkeypatch)

    assert events.index("barrier:raw") < events.index("publish:manifest")


def test_failure_attempt_is_published_only_after_recursive_raw_durability_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _install_status_clients(monkeypatch, status_code=404)
    events: list[str] = []
    original_fsync_tree = m8_acquisition._fsync_tree
    original_publish = m8_acquisition._publish_failure_authority

    def tracked_fsync_tree(path: Path) -> None:
        events.append(f"barrier:{path.name}")
        original_fsync_tree(path)

    def tracked_publish(*args: object, **kwargs: object) -> object:
        events.append("publish:failure")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(m8_acquisition, "_fsync_tree", tracked_fsync_tree)
    monkeypatch.setattr(m8_acquisition, "_publish_failure_authority", tracked_publish)
    result = acquire_m8_archives(config, tmp_path / "authority")

    assert isinstance(result, M8AcquisitionFailureResult)
    assert events.index("barrier:raw") < events.index("publish:failure")


def test_failure_terminal_is_not_published_when_raw_durability_barrier_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _install_status_clients(monkeypatch, status_code=404)
    output_root = tmp_path / "authority"

    def fail_raw_barrier(path: Path) -> None:
        assert path.name == "raw"
        raise OSError("injected raw fsync failure")

    monkeypatch.setattr(m8_acquisition, "_fsync_tree", fail_raw_barrier)
    with pytest.raises(M8AcquisitionError, match="failed closed"):
        acquire_m8_archives(config, output_root)

    assert not (output_root / "_attempts").exists()
    assert not list(output_root.rglob("INSUFFICIENT_DATA"))


def test_self_contained_copy_flushes_raw_tree_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _acquire_fixture(tmp_path, monkeypatch)
    events: list[str] = []
    original_fsync_tree = m8_acquisition._fsync_tree
    original_write_manifest = m8_acquisition._write_manifest_payload

    def tracked_fsync_tree(path: Path) -> None:
        events.append(f"barrier:{path.name}")
        original_fsync_tree(path)

    def tracked_write_manifest(*args: object, **kwargs: object) -> tuple[Path, str]:
        events.append("publish:manifest")
        return original_write_manifest(*args, **kwargs)

    monkeypatch.setattr(m8_acquisition, "_fsync_tree", tracked_fsync_tree)
    monkeypatch.setattr(m8_acquisition, "_write_manifest_payload", tracked_write_manifest)
    copy_m8_acquisition_into(result.manifest, tmp_path / "copy")

    assert events.index("barrier:raw") < events.index("publish:manifest")


def test_manifest_snapshot_rejects_path_swap_after_same_fd_hash_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _acquire_fixture(tmp_path, monkeypatch)
    original = result.manifest_path.read_bytes()
    replacement_payload = json.loads(original)
    replacement_payload["evidence_set_sha256"] = "0" * 64
    replacement = m8_acquisition._canonical_json_bytes(replacement_payload)
    assert len(replacement) == len(original)
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_bytes(replacement)
    original_snapshot = m8_acquisition._read_bounded_regular_snapshot
    swapped = False

    def swapping_snapshot(
        path: Path,
        label: str,
        *,
        byte_limit: int,
    ) -> object:
        nonlocal swapped
        snapshot = original_snapshot(path, label, byte_limit=byte_limit)
        if path == result.manifest_path and not swapped:
            os.replace(replacement_path, result.manifest_path)
            swapped = True
        return snapshot

    monkeypatch.setattr(m8_acquisition, "_read_bounded_regular_snapshot", swapping_snapshot)
    with pytest.raises(M8AcquisitionError, match="path changed after snapshot"):
        read_m8_acquisition_manifest(
            result.manifest_path,
            expected_sha256=result.manifest_sha256,
            config=result.manifest.config,
        )
