from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import requests

from microstructure.data.binance_archive import (
    ArchiveDownloadLimits,
    BinanceArchiveClient,
    BinanceArchiveHTTPError,
    BinanceArchivePayloadError,
    DailyArchiveRequest,
    RetryPolicy,
)
from microstructure.data.evidence_budget import EvidenceBudgetExceeded, RetainedEvidenceBudget
from microstructure.data.schemas import ensure_schema
from microstructure.provenance import sha256_file

BASE_URL = "https://fixtures.invalid"
DAY = date(2024, 1, 3)
START_MS = 1_704_240_000_000


class _Response:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        headers: Mapping[str, str] | None = None,
        status_code: int = 200,
        interrupt_after: int | None = None,
        final_url: str | None = None,
    ) -> None:
        self.chunks = chunks
        self.headers = dict(headers or {})
        self.status_code = status_code
        self.interrupt_after = interrupt_after
        self.final_url = final_url
        self.url = ""
        self.closed = False
        self.chunk_sizes: list[int] = []
        self.chunks_read = 0

    @property
    def content(self) -> bytes:
        raise AssertionError("archive acquisition must not access response.content")

    @property
    def text(self) -> str:
        raise AssertionError("archive acquisition must not access response.text")

    def iter_content(self, *, chunk_size: int) -> Iterator[bytes]:
        self.chunk_sizes.append(chunk_size)
        for index, chunk in enumerate(self.chunks):
            if self.interrupt_after is not None and index == self.interrupt_after:
                raise requests.ConnectionError("fixture interrupted")
            self.chunks_read += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: list[_Response | requests.RequestException]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, timeout: float, stream: bool) -> _Response:
        self.calls.append({"url": url, "timeout": timeout, "stream": stream})
        response = self.responses.pop(0)
        if isinstance(response, requests.RequestException):
            raise response
        response.url = response.final_url or url
        return response


def _request(*, archive_date: date = DAY) -> DailyArchiveRequest:
    return DailyArchiveRequest(
        symbol="BTCUSDT",
        date=archive_date,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.0001"),
    )


def _limits(
    *,
    compressed: int = 1_000_000,
    uncompressed: int = 1_000_000,
    checksum: int = 4_096,
    chunk: int = 17,
    line: int = 16_384,
) -> ArchiveDownloadLimits:
    return ArchiveDownloadLimits(
        max_compressed_bytes=compressed,
        max_uncompressed_bytes=uncompressed,
        max_checksum_bytes=checksum,
        transfer_chunk_bytes=chunk,
        max_csv_line_bytes=line,
    )


def _row(
    aggregate_id: int,
    *,
    timestamp_ms: int,
    price: str = "42000.01",
    quantity: str = "0.0010",
    first_trade_id: int | None = None,
    last_trade_id: int | None = None,
    buyer_is_maker: str = "true",
    best_match: str = "true",
) -> bytes:
    first = aggregate_id * 2 if first_trade_id is None else first_trade_id
    last = first if last_trade_id is None else last_trade_id
    return (
        f"{aggregate_id},{price},{quantity},{first},{last},{timestamp_ms},"
        f"{buyer_is_maker},{best_match}"
    ).encode()


def _zip_bytes(
    content: bytes,
    *,
    member: str | None = None,
    extra_member: bool = False,
) -> bytes:
    request = _request()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member or request.member_name, content)
        if extra_member:
            archive.writestr("unexpected.csv", b"x")
    return buffer.getvalue()


def _responses(
    archive: bytes,
    *,
    expected_sha: str | None = None,
    checksum_body: bytes | None = None,
    archive_chunks: list[bytes] | None = None,
    archive_headers: Mapping[str, str] | None = None,
) -> tuple[_Session, _Response, _Response]:
    request = _request()
    official = expected_sha or hashlib.sha256(archive).hexdigest()
    checksum = checksum_body or f"{official}  {request.archive_name}\n".encode()
    checksum_response = _Response([checksum], headers={"Content-Length": str(len(checksum))})
    chunks = (
        archive_chunks
        if archive_chunks is not None
        else [archive[index : index + 11] for index in range(0, len(archive), 11)]
    )
    headers = dict(
        archive_headers if archive_headers is not None else {"Content-Length": str(len(archive))}
    )
    archive_response = _Response(chunks, headers=headers)
    return _Session([checksum_response, archive_response]), checksum_response, archive_response


def _acquire(
    tmp_path: Path,
    csv_content: bytes,
    *,
    member: str | None = None,
    extra_member: bool = False,
    limits: ArchiveDownloadLimits | None = None,
):
    archive = _zip_bytes(csv_content, member=member, extra_member=extra_member)
    session, _, _ = _responses(archive)
    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
    ).acquire(_request(), raw_root=tmp_path, limits=limits or _limits())
    return acquired, session, archive


def _retained_regular_file_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def test_acquire_is_bounded_content_addressed_and_does_not_open_csv(tmp_path: Path) -> None:
    csv_content = (
        b"\n".join(
            [
                _row(10, timestamp_ms=START_MS),
                _row(11, timestamp_ms=START_MS + 1, buyer_is_maker="false"),
                _row(12, timestamp_ms=START_MS + 2),
            ]
        )
        + b"\n"
    )
    archive = _zip_bytes(csv_content)
    session, checksum_response, archive_response = _responses(archive)
    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        timeout_seconds=7.5,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits(chunk=13))

    digest = hashlib.sha256(archive).hexdigest()
    assert acquired.archive_artifact.path.name == _request().archive_name
    assert acquired.archive_artifact.sha256 == digest == acquired.upstream_sha256
    assert acquired.archive_artifact.bytes == len(archive)
    assert acquired.declared_uncompressed_bytes == len(csv_content)
    assert acquired.archive_artifact.path.read_bytes() == archive
    assert acquired.checksum_artifact.path.name == f"{_request().archive_name}.CHECKSUM"
    checksum_manifest = json.loads(acquired.checksum_artifact.manifest_path.read_text())
    assert checksum_manifest["source"] == "binance_spot_daily_aggtrades_archive_checksum"
    assert checksum_manifest["path"] == f"{_request().archive_name}.CHECKSUM"
    assert sha256_file(acquired.archive_artifact.manifest_path) == (
        acquired.archive_artifact.manifest_sha256
    )
    manifest = json.loads(acquired.archive_artifact.manifest_path.read_text())
    assert manifest["source"] == "binance_spot_daily_aggtrades_archive"
    assert manifest["path"] == _request().archive_name
    assert manifest["upstream_checksum_sha256"] == digest
    assert manifest["requested_range_ns"] == {
        "start": 1_704_240_000_000_000_000,
        "end_exclusive": 1_704_326_400_000_000_000,
    }
    assert [call["stream"] for call in session.calls] == [True, True]
    assert [call["timeout"] for call in session.calls] == [7.5, 7.5]
    assert checksum_response.closed and archive_response.closed
    assert archive_response.chunk_sizes == [13]
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_shared_evidence_budget_charges_every_success_artifact_exactly_once(
    tmp_path: Path,
) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS) + b"\n")
    session, _, _ = _responses(archive)
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=1_000_000)

    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retained_evidence_budget=budget,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert acquired.archive_artifact.path.is_file()
    assert acquired.checksum_artifact.path.is_file()
    assert acquired.archive_artifact.manifest_path.is_file()
    assert acquired.checksum_artifact.manifest_path.is_file()
    assert budget.reserved_bytes == 0
    assert budget.used_bytes == _retained_regular_file_bytes(tmp_path)


def test_budgeted_content_reuse_releases_download_reservations_without_double_charge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "microstructure.data.binance_archive.utc_now_iso",
        lambda: "2026-08-07T00:00:00Z",
    )
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS) + b"\n")
    first_session, _, _ = _responses(archive)
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=1_000_000)
    first = BinanceArchiveClient(
        session=first_session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retained_evidence_budget=budget,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())
    first_used = budget.used_bytes
    first_files = sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()
    )

    second_session, _, _ = _responses(archive)
    second = BinanceArchiveClient(
        session=second_session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retained_evidence_budget=budget,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert second.archive_artifact.path == first.archive_artifact.path
    assert second.checksum_artifact.path == first.checksum_artifact.path
    assert second.archive_artifact.manifest_path == first.archive_artifact.manifest_path
    assert second.checksum_artifact.manifest_path == first.checksum_artifact.manifest_path
    assert budget.used_bytes == first_used == _retained_regular_file_bytes(tmp_path)
    assert budget.reserved_bytes == 0
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()) == (
        first_files
    )
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_budgeted_retries_charge_one_duplicate_prefix_and_every_distinct_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "microstructure.data.binance_archive.utc_now_iso",
        lambda: "2026-08-07T00:00:00Z",
    )
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS) + b"\n")
    _, checksum_response, archive_response = _responses(archive)
    first_interrupted = _Response(
        [archive[:12], archive[12:]],
        headers={"Content-Length": str(len(archive))},
        interrupt_after=1,
    )
    second_interrupted = _Response(
        [archive[:12], archive[12:]],
        headers={"Content-Length": str(len(archive))},
        interrupt_after=1,
    )
    session = _Session([checksum_response, first_interrupted, second_interrupted, archive_response])
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=1_000_000)

    BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=0.0),
        sleep=lambda _: None,
        random_value=lambda: 0.0,
        retained_evidence_budget=budget,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert len(list(tmp_path.rglob("*.zip.rejected"))) == 1
    rejected_manifests = list(tmp_path.rglob("*.zip.rejected.manifest-*.json"))
    assert len(rejected_manifests) == 2
    assert {
        json.loads(path.read_text())["response_headers"]["x-local-download-attempt"]
        for path in rejected_manifests
    } == {"1", "2"}
    assert budget.used_bytes == _retained_regular_file_bytes(tmp_path)
    assert budget.reserved_bytes == 0
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_sidecar_budget_failure_rolls_back_new_body_and_all_reservations(
    tmp_path: Path,
) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS) + b"\n")
    session, checksum_response, archive_response = _responses(archive)
    checksum_bytes = sum(len(chunk) for chunk in checksum_response.chunks)
    budget = RetainedEvidenceBudget(tmp_path, limit_bytes=checksum_bytes)

    with pytest.raises(EvidenceBudgetExceeded, match="raw source manifest"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
            retained_evidence_budget=budget,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert checksum_response.closed
    assert archive_response.chunks_read == 0
    assert budget.used_bytes == budget.reserved_bytes == 0
    assert _retained_regular_file_bytes(tmp_path) == 0
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_normalized_stream_is_one_shot_bounded_and_terminally_summarized(
    tmp_path: Path,
) -> None:
    csv_content = (
        b"\n".join(
            [
                _row(40, timestamp_ms=START_MS, quantity="0.0010"),
                _row(41, timestamp_ms=START_MS, buyer_is_maker="false"),
                _row(42, timestamp_ms=START_MS + 2, price="42000.02"),
            ]
        )
        + b"\n"
    )
    acquired, _, archive = _acquire(tmp_path, csv_content)
    stream = acquired.iter_normalized_batches(batch_rows=2)
    assert iter(stream) is stream
    with pytest.raises(RuntimeError, match="before full stream exhaustion"):
        _ = stream.summary

    batches = list(stream)
    assert [batch.num_rows for batch in batches] == [2, 1]
    for batch in batches:
        ensure_schema(batch, "trades")
    table = pa.Table.from_batches(batches)
    assert table.column("trade_id").to_pylist() == [40, 41, 42]
    assert table.column("event_ts_ns").to_pylist() == [
        START_MS * 1_000_000,
        START_MS * 1_000_000,
        (START_MS + 2) * 1_000_000,
    ]
    assert table.column("price_ticks").to_pylist() == [4_200_001, 4_200_001, 4_200_002]
    assert table.column("quantity_lots").to_pylist() == [10, 10, 10]
    assert table.column("aggressor_side").to_pylist() == ["sell", "buy", "sell"]
    assert set(table.column("continuity_id").to_pylist()) == {"binance_spot:BTCUSDT:2024-01-03"}
    assert set(table.column("source_artifact_id").to_pylist()) == {acquired.archive_artifact.sha256}
    summary = stream.summary
    assert summary.rows == 3
    assert summary.first_trade_id == 40
    assert summary.last_trade_id == 42
    assert summary.first_event_ts_ns == START_MS * 1_000_000
    assert summary.last_event_ts_ns == (START_MS + 2) * 1_000_000
    assert summary.expanded_bytes == len(csv_content)
    assert summary.compressed_bytes == len(archive)
    with pytest.raises(StopIteration):
        next(stream)


def test_member_stream_hash_directory_and_rows_are_bound_to_one_open_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    official_csv = _row(1, timestamp_ms=START_MS, price="42000.01") + b"\n"
    malicious_csv = _row(1, timestamp_ms=START_MS, price="99999.99") + b"\n"
    assert len(official_csv) == len(malicious_csv)
    acquired, _, _ = _acquire(tmp_path, official_csv)
    acquired = replace(acquired, requires_member_open_guard=True)
    official_path = acquired.archive_artifact.path
    official_sha = acquired.archive_artifact.sha256
    malicious_path = tmp_path / "malicious.zip"
    malicious_path.write_bytes(_zip_bytes(malicious_csv))
    official_backup = tmp_path / "official-backup.zip"
    malicious_backup = tmp_path / "malicious-backup.zip"
    original_zipfile = zipfile.ZipFile
    swapped = False

    def swapping_zipfile(source: object, *args: object, **kwargs: object) -> zipfile.ZipFile:
        nonlocal swapped
        if not swapped:
            os.replace(official_path, official_backup)
            os.replace(malicious_path, official_path)
            swapped = True
        return original_zipfile(source, *args, **kwargs)  # type: ignore[arg-type]

    def restore_and_verify_authority() -> None:
        os.replace(official_path, malicious_backup)
        os.replace(official_backup, official_path)
        assert sha256_file(official_path) == official_sha

    monkeypatch.setattr(zipfile, "ZipFile", swapping_zipfile)
    batches = list(
        acquired.iter_normalized_batches(
            batch_rows=1,
            before_member_open=restore_and_verify_authority,
        )
    )

    table = pa.Table.from_batches(batches)
    assert swapped is True
    assert table.column("price").to_pylist() == [42000.01]
    assert sha256_file(official_path) == official_sha


def test_early_close_cannot_fabricate_summary(tmp_path: Path) -> None:
    content = b"\n".join([_row(1, timestamp_ms=START_MS), _row(2, timestamp_ms=START_MS + 1)])
    acquired, _, _ = _acquire(tmp_path, content)
    stream = acquired.iter_normalized_batches(batch_rows=1)
    assert next(stream).num_rows == 1
    stream.close()
    with pytest.raises(RuntimeError, match="before full stream exhaustion"):
        _ = stream.summary
    with pytest.raises(StopIteration):
        next(stream)


def test_archive_body_ceiling_preserves_only_bounded_prefix(tmp_path: Path) -> None:
    csv_content = _row(1, timestamp_ms=START_MS) + b"\n"
    archive = _zip_bytes(csv_content)
    limit = max(1, len(archive) // 2)
    session, checksum_response, archive_response = _responses(
        archive,
        archive_chunks=[archive[:limit], archive[limit:]],
        archive_headers={},
    )
    client = BinanceArchiveClient(session=session, base_url=BASE_URL)  # type: ignore[arg-type]

    with pytest.raises(BinanceArchivePayloadError, match="exceeds byte ceiling"):
        client.acquire(
            _request(),
            raw_root=tmp_path,
            limits=_limits(compressed=limit, chunk=limit),
        )

    rejected = list(tmp_path.rglob("*.zip.rejected"))
    assert len(rejected) == 1
    assert rejected[0].stat().st_size == limit
    assert rejected[0].read_bytes() == archive[:limit]
    assert checksum_response.closed and archive_response.closed
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_content_length_ceiling_rejects_without_reading_body(tmp_path: Path) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    session, _, archive_response = _responses(
        archive,
        archive_headers={"Content-Length": str(len(archive) + 1)},
    )
    with pytest.raises(BinanceArchivePayloadError, match="Content-Length"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits(compressed=len(archive)))
    assert archive_response.chunks_read == 0
    rejected = list(tmp_path.rglob("*.zip.rejected"))
    assert len(rejected) == 1 and rejected[0].stat().st_size == 0


def test_interrupted_archive_stream_preserves_prefix_and_closes(tmp_path: Path) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    official = hashlib.sha256(archive).hexdigest()
    checksum = f"{official}  {_request().archive_name}\n".encode()
    checksum_response = _Response([checksum], headers={"Content-Length": str(len(checksum))})
    archive_response = _Response([archive[:12], archive[12:]], interrupt_after=1)
    session = _Session([checksum_response, archive_response])

    with pytest.raises(BinanceArchiveHTTPError, match="interrupted after 12 bytes"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
            retry_policy=RetryPolicy(max_retries=0),
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    rejected = list(tmp_path.rglob("*.zip.rejected"))
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == archive[:12]
    assert archive_response.closed


def test_retryable_503_is_evidenced_then_succeeds(tmp_path: Path) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    _, checksum_response, archive_response = _responses(archive)
    unavailable = _Response([], status_code=503)
    session = _Session([unavailable, checksum_response, archive_response])
    delays: list[float] = []

    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retry_policy=RetryPolicy(
            max_retries=1,
            base_delay_seconds=4.0,
            max_delay_seconds=1.0,
        ),
        sleep=delays.append,
        random_value=lambda: 0.5,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert acquired.archive_artifact.path.read_bytes() == archive
    assert len(session.calls) == 3
    assert unavailable.closed and checksum_response.closed and archive_response.closed
    assert delays == [0.5]
    rejected = list(tmp_path.rglob("*.CHECKSUM.rejected"))
    manifests = list(tmp_path.rglob("*.CHECKSUM.rejected.manifest-*.json"))
    assert len(rejected) == len(manifests) == 1
    evidence = json.loads(manifests[0].read_text())
    assert evidence["response_headers"]["x-local-download-attempt"] == "1"
    assert "HTTP 503" in evidence["response_headers"]["x-local-rejection-reason"]
    assert not session.responses


def test_mid_body_interruption_retries_after_retaining_prefix_and_sidecar(
    tmp_path: Path,
) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    _, checksum_response, archive_response = _responses(archive)
    interrupted = _Response(
        [archive[:12], archive[12:]],
        headers={"Content-Length": str(len(archive))},
        interrupt_after=1,
    )
    session = _Session([checksum_response, interrupted, archive_response])

    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.0),
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    rejected = list(tmp_path.rglob("*.zip.rejected"))
    manifests = list(tmp_path.rglob("*.zip.rejected.manifest-*.json"))
    assert len(rejected) == len(manifests) == 1
    assert rejected[0].read_bytes() == archive[:12]
    assert acquired.archive_artifact.path.read_bytes() == archive
    assert interrupted.closed and archive_response.closed
    assert not session.responses


def test_429_honors_retry_after_and_closes_before_sleep(tmp_path: Path) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    _, checksum_response, archive_response = _responses(archive)
    throttled = _Response([], status_code=429, headers={"Retry-After": "2.75"})
    session = _Session([throttled, checksum_response, archive_response])
    delays: list[float] = []

    def record_sleep(delay: float) -> None:
        assert throttled.closed
        delays.append(delay)

    def unexpected_random() -> float:
        raise AssertionError("valid Retry-After must bypass jitter")

    BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retry_policy=RetryPolicy(max_retries=1),
        sleep=record_sleep,
        random_value=unexpected_random,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert delays == [2.75]
    assert not session.responses


def test_content_length_truncation_is_evidenced_and_retried(tmp_path: Path) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    _, checksum_response, archive_response = _responses(archive)
    truncated = _Response(
        [archive],
        headers={"Content-Length": str(len(archive) + 7)},
    )
    session = _Session([checksum_response, truncated, archive_response])

    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.0),
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    rejected = list(tmp_path.rglob("*.zip.rejected"))
    assert len(rejected) == 1 and rejected[0].read_bytes() == archive
    assert acquired.archive_artifact.path.read_bytes() == archive
    assert truncated.closed
    assert not session.responses


@pytest.mark.parametrize("failure_kind", ["404", "redirect"])
def test_nonretryable_http_failures_stop_after_one_attempt(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    if failure_kind == "404":
        failure = _Response([], status_code=404)
        message = "HTTP 404"
    else:
        failure = _Response([], final_url="https://redirect.invalid/archive")
        message = "redirected"
    session = _Session([failure])
    delays: list[float] = []

    with pytest.raises(BinanceArchiveHTTPError, match=message):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
            retry_policy=RetryPolicy(max_retries=3),
            sleep=delays.append,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert len(session.calls) == 1
    assert not delays
    assert failure.closed
    manifests = list(tmp_path.rglob("*.CHECKSUM.rejected.manifest-*.json"))
    assert len(manifests) == 1


def test_retry_attempt_exhaustion_retains_evidence_for_every_connect_failure(
    tmp_path: Path,
) -> None:
    session = _Session(
        [
            requests.ConnectionError("connect one"),
            requests.Timeout("connect two"),
            requests.ConnectionError("connect three"),
        ]
    )
    delays: list[float] = []

    with pytest.raises(BinanceArchiveHTTPError, match="failed before a response") as raised:
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
            retry_policy=RetryPolicy(
                max_retries=2,
                base_delay_seconds=2.0,
                max_delay_seconds=3.0,
            ),
            sleep=delays.append,
            random_value=lambda: 0.5,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert delays == [1.0, 1.5]
    assert len(session.calls) == 3
    assert any("exhausted 3" in note for note in raised.value.__notes__)
    rejected = list(tmp_path.rglob("*.CHECKSUM.rejected"))
    manifests = list(tmp_path.rglob("*.CHECKSUM.rejected.manifest-*.json"))
    assert len(rejected) == 1
    assert len(manifests) == 3
    attempts = sorted(
        json.loads(path.read_text())["response_headers"]["x-local-download-attempt"]
        for path in manifests
    )
    assert attempts == ["1", "2", "3"]
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_official_checksum_mismatch_never_publishes_trusted_zip(tmp_path: Path) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    session, _, archive_response = _responses(archive, expected_sha="0" * 64)

    with pytest.raises(BinanceArchivePayloadError, match="official CHECKSUM"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert not list((tmp_path / "binance_spot" / "daily_agg_trades_archive").rglob("*.zip"))
    rejected = list(tmp_path.rglob("*.zip.rejected"))
    assert len(rejected) == 1 and rejected[0].read_bytes() == archive
    assert archive_response.closed


def test_official_checksum_basename_is_immutable_across_conflicting_acquisitions(
    tmp_path: Path,
) -> None:
    first_archive = _zip_bytes(_row(1, timestamp_ms=START_MS))
    first_session, _, _ = _responses(first_archive)
    first = BinanceArchiveClient(
        session=first_session,  # type: ignore[arg-type]
        base_url=BASE_URL,
    ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    second_archive = _zip_bytes(_row(2, timestamp_ms=START_MS + 1))
    second_session, _, _ = _responses(second_archive)
    with pytest.raises(BinanceArchivePayloadError, match="official CHECKSUM basename collides"):
        BinanceArchiveClient(
            session=second_session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())

    assert first.archive_artifact.path.name == _request().archive_name
    assert first.archive_artifact.path.read_bytes() == first_archive
    assert len(second_session.calls) == 1
    rejected = list(tmp_path.rglob("*.CHECKSUM.rejected"))
    expected_checksum = (
        f"{hashlib.sha256(second_archive).hexdigest()}  {_request().archive_name}\n".encode()
    )
    assert len(rejected) == 1 and rejected[0].read_bytes() == expected_checksum
    assert not list(tmp_path.rglob(".download-*.tmp"))


@pytest.mark.parametrize(
    "checksum",
    [
        b"not-a-checksum\n",
        f"{'0' * 64} {_request().archive_name}\n".encode(),
        f"{'0' * 64}  WRONG.zip\n".encode(),
        f"{'A' * 64}  {_request().archive_name}\n".encode(),
        f"{'0' * 64}  {_request().archive_name}\r".encode(),
    ],
)
def test_malformed_checksum_fails_before_archive_request(tmp_path: Path, checksum: bytes) -> None:
    response = _Response([checksum], headers={"Content-Length": str(len(checksum))})
    session = _Session([response])
    with pytest.raises(BinanceArchivePayloadError, match="CHECKSUM"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())
    assert len(session.calls) == 1
    assert response.closed
    assert len(list(tmp_path.rglob("*.CHECKSUM"))) == 1


@pytest.mark.parametrize(
    ("member", "extra", "message"),
    [
        ("wrong.csv", False, "member path/name"),
        ("../BTCUSDT-aggTrades-2024-01-03.csv", False, "member path/name"),
        (None, True, "exactly one"),
    ],
)
def test_zip_member_contract_is_fail_closed(
    tmp_path: Path,
    member: str | None,
    extra: bool,
    message: str,
) -> None:
    archive = _zip_bytes(_row(1, timestamp_ms=START_MS), member=member, extra_member=extra)
    session, _, _ = _responses(archive)
    with pytest.raises(BinanceArchivePayloadError, match=message):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())
    trusted = list(tmp_path.rglob("daily_agg_trades_archive/*/*/*.zip"))
    assert len(trusted) == 1
    assert trusted[0].read_bytes() == archive


def test_zip_local_and_central_member_names_must_agree(tmp_path: Path) -> None:
    archive = bytearray(_zip_bytes(_row(1, timestamp_ms=START_MS)))
    expected_name = _request().member_name.encode("ascii")
    local_name_offset = archive.find(expected_name)
    central_name_offset = archive.find(expected_name, local_name_offset + len(expected_name))
    assert local_name_offset >= 0 and central_name_offset > local_name_offset
    archive[local_name_offset : local_name_offset + len(expected_name)] = b"x" * len(expected_name)
    response_bytes = bytes(archive)
    session, _, _ = _responses(response_bytes)

    with pytest.raises(BinanceArchivePayloadError, match="local member path/name"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(_request(), raw_root=tmp_path, limits=_limits())


def test_declared_uncompressed_size_ceiling_fails_before_csv_open(tmp_path: Path) -> None:
    content = _row(1, timestamp_ms=START_MS) + b"\n"
    archive = _zip_bytes(content)
    session, _, _ = _responses(archive)
    with pytest.raises(BinanceArchivePayloadError, match="uncompressed bytes"):
        BinanceArchiveClient(
            session=session,  # type: ignore[arg-type]
            base_url=BASE_URL,
        ).acquire(
            _request(),
            raw_root=tmp_path,
            limits=_limits(uncompressed=len(content) - 1),
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"1,2,3\n", "exactly 8 fields"),
        (
            _row(1, timestamp_ms=START_MS) + b"\n" + _row(3, timestamp_ms=START_MS + 1),
            "noncontiguous",
        ),
        (
            _row(1, timestamp_ms=START_MS + 2) + b"\n" + _row(2, timestamp_ms=START_MS + 1),
            "time reverses",
        ),
        (_row(1, timestamp_ms=START_MS - 1), "outside declared UTC date"),
        (_row(1, timestamp_ms=START_MS, price="42000.001"), "not aligned"),
        (_row(1, timestamp_ms=START_MS, quantity="0"), "positive and finite"),
        (_row(1, timestamp_ms=START_MS, buyer_is_maker="yes"), "true or false"),
        (_row(1, timestamp_ms=START_MS, first_trade_id=9, last_trade_id=8), "exceeds"),
    ],
)
def test_csv_shape_type_date_sequence_and_scale_contracts(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    acquired, _, _ = _acquire(tmp_path, content)
    with pytest.raises(BinanceArchivePayloadError, match=message):
        list(acquired.iter_normalized_batches(batch_rows=2))
    assert acquired.archive_artifact.path.is_file()
    assert not list(tmp_path.rglob(".download-*.tmp"))


def test_csv_line_ceiling_is_enforced_during_expansion(tmp_path: Path) -> None:
    content = _row(1, timestamp_ms=START_MS)
    acquired, _, _ = _acquire(tmp_path, content, limits=_limits(line=8))
    with pytest.raises(BinanceArchivePayloadError, match="CSV line exceeds"):
        list(acquired.iter_normalized_batches())


def test_archive_tampering_is_detected_before_csv_open(tmp_path: Path) -> None:
    acquired, _, _ = _acquire(tmp_path, _row(1, timestamp_ms=START_MS))
    with acquired.archive_artifact.path.open("ab") as sink:
        sink.write(b"tamper")
    with pytest.raises(BinanceArchivePayloadError, match="bytes changed"):
        next(acquired.iter_normalized_batches())


def test_request_and_limit_validation() -> None:
    with pytest.raises(ValueError, match="uppercase"):
        DailyArchiveRequest("../btc", DAY, Decimal("0.01"), Decimal("0.001"))
    with pytest.raises(ValueError, match="tick_size"):
        DailyArchiveRequest("BTCUSDT", DAY, Decimal("0"), Decimal("0.001"))
    with pytest.raises(ValueError, match="byte limits"):
        _limits(compressed=0)
    with pytest.raises(ValueError, match="HTTPS origin"):
        BinanceArchiveClient(base_url="http://data.binance.vision/path")


def test_microsecond_archive_timestamp_policy_from_2025(tmp_path: Path) -> None:
    archive_date = date(2025, 1, 1)
    request = _request(archive_date=archive_date)
    timestamp_us = 1_735_689_600_000_001
    content = _row(1, timestamp_ms=timestamp_us)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
        archive_file.writestr(request.member_name, content)
    archive = buffer.getvalue()
    official = hashlib.sha256(archive).hexdigest()
    checksum = f"{official}  {request.archive_name}\n".encode()
    session = _Session(
        [
            _Response([checksum], headers={"Content-Length": str(len(checksum))}),
            _Response([archive], headers={"Content-Length": str(len(archive))}),
        ]
    )
    acquired = BinanceArchiveClient(
        session=session,  # type: ignore[arg-type]
        base_url=BASE_URL,
    ).acquire(request, raw_root=tmp_path, limits=_limits())
    table = pa.Table.from_batches(list(acquired.iter_normalized_batches()))
    assert table.column("event_ts_ns").to_pylist() == [timestamp_us * 1_000]
