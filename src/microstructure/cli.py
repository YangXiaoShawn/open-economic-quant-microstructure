"""Command-line interface for research data, reproduction, and reporting."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa  # type: ignore[import-untyped]

from microstructure import __version__
from microstructure.config import ProjectConfig, datetime_to_ns, load_config
from microstructure.data.binance import (
    BinanceLiveDepthCollector,
    BinancePublicClient,
    CapturedDepth,
    depth_deltas_table,
)
from microstructure.data.book import (
    ReconstructionResult,
    deltas_table,
    reconstruct_snapshot_and_deltas,
    snapshots_table,
)
from microstructure.data.quality import validate_table
from microstructure.data.storage import write_partitioned_parquet, write_source_manifest
from microstructure.data.synthetic import generate_synthetic_market
from microstructure.ingestion import (
    IngestionResult,
    ingest_from_config,
    validate_configured_input,
)
from microstructure.pipeline import reproduce
from microstructure.provenance import utc_now_iso, write_json
from microstructure.reporting import load_run_bundle, verify_checksums, write_report_set


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _ingestion_payload(result: IngestionResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "evidence_tier": result.evidence_tier,
        "output_root": result.output_root,
        "ingestion_manifest": result.ingestion_manifest_path,
        "ingestion_manifest_sha256": result.ingestion_manifest_sha256,
        "rows": result.rows,
        "datasets": [
            {
                "schema": dataset.schema_name,
                "rows": dataset.rows,
                "manifest": dataset.storage.manifest_path,
                "manifest_sha256": dataset.storage.manifest_sha256,
                "quality_errors": dataset.validation.error_count,
                "quality_warnings": dataset.validation.warning_count,
            }
            for dataset in result.datasets
        ],
        "raw_artifacts": len(result.raw_artifacts),
        "symbols": [
            {
                "symbol": item.symbol,
                "rows": item.rows,
                "complete_range": item.complete_range,
                "tick_size": item.metadata.tick_size,
                "lot_size": item.metadata.lot_size,
            }
            for item in result.symbols
        ],
        "quality": {
            "passed": result.validation.passed,
            "rows_checked": result.validation.rows_checked,
            "errors": result.validation.error_count,
            "warnings": result.validation.warning_count,
        },
    }


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_root = (
        Path(args.output_root).resolve() if args.output_root else config.data.partition_root.parent
    )
    result = ingest_from_config(config, output_root)
    _print_json(_ingestion_payload(result))
    return 0


def _synthetic_tables(config: ProjectConfig) -> dict[str, Any]:
    events = config.data.events_per_symbol
    if events is None:
        raise ValueError("synthetic validation requires data.events_per_symbol")
    generated = generate_synthetic_market(
        symbols=config.data.symbols,
        events_per_symbol=events,
        start_ts_ns=datetime_to_ns(config.data.start),
        seed=config.run.seed,
    )
    return {"trades": generated.trades, "book_observations": generated.book_observations}


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    tables = _synthetic_tables(config) if config.data.mode == "synthetic" else None
    summary = validate_configured_input(config, tables=tables)
    _print_json(
        {
            "passed": summary.passed,
            "rows_checked": summary.rows_checked,
            "errors": summary.error_count,
            "warnings": summary.warning_count,
            "reports": [
                {
                    "dataset": report.dataset,
                    "rows_checked": report.rows_checked,
                    "errors": report.error_count,
                    "warnings": report.warning_count,
                }
                for report in summary.reports
            ],
            "mutation_policy": "validation did not repair or replace observations",
        }
    )
    return 0 if summary.passed else 1


def _cmd_reproduce(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = reproduce(config, Path(args.run_dir))
    bundle = load_run_bundle(output)
    _print_json(
        {
            "run_dir": output,
            "run_id": bundle.run_id,
            "evidence_tier": bundle.evidence_tier,
            "observed_start_utc": bundle.observed_start_utc,
            "observed_end_utc": bundle.observed_end_utc,
            "status": "complete",
        }
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    bundle = load_run_bundle(args.run_dir)
    protected = verify_checksums(args.run_dir)
    _print_json(
        {
            "run_dir": bundle.root,
            "run_id": bundle.run_id,
            "evidence_tier": bundle.evidence_tier,
            "protected_files": protected,
            "integrity": "verified",
        }
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    bundle = load_run_bundle(args.run_dir)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else bundle.root.with_name(f"{bundle.root.name}-reports")
    )
    paths = write_report_set(bundle, output)
    _print_json({"run_id": bundle.run_id, "output_dir": output, **asdict(paths)})
    return 0


def _atomic_raw_messages(path: Path, captured: Sequence[CapturedDepth]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for item in captured:
                handle.write(item.raw_payload.rstrip("\n") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


async def _capture_depth(
    *,
    symbol: str,
    max_messages: int,
    output_root: Path,
) -> tuple[list[CapturedDepth], ReconstructionResult, str, str]:
    client = BinancePublicClient()
    metadata = client.fetch_exchange_info(symbol=symbol, raw_root=output_root / "raw")
    collector = BinanceLiveDepthCollector(
        symbols=(symbol,),
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
    )
    stream = collector.stream(max_messages=max_messages)
    first = await anext(stream)
    captured = [first]
    snapshot = client.fetch_depth_snapshot(
        symbol=symbol,
        raw_root=output_root / "raw",
        continuity_id=first.delta.continuity_id,
        tick_size=metadata.tick_size,
        lot_size=metadata.lot_size,
    )
    snapshots = {snapshot.continuity_id: snapshot}
    async for item in stream:
        captured.append(item)
        if item.delta.continuity_id not in snapshots:
            snapshots[item.delta.continuity_id] = client.fetch_depth_snapshot(
                symbol=symbol,
                raw_root=output_root / "raw",
                continuity_id=item.delta.continuity_id,
                tick_size=metadata.tick_size,
                lot_size=metadata.lot_size,
            )

    first_received_ns = first.delta.received_ts_ns
    last_received_ns = captured[-1].delta.received_ts_ns
    if first_received_ns is None or last_received_ns is None:
        raise RuntimeError("captured depth messages must carry receipt timestamps")
    raw_path = (
        output_root
        / "raw"
        / "binance_spot"
        / "depth_stream"
        / symbol
        / (f"{first_received_ns}-{last_received_ns}.ndjson")
    )
    _atomic_raw_messages(raw_path, captured)
    raw_manifest, raw_manifest_sha = write_source_manifest(
        raw_path,
        source="binance_spot_public_websocket",
        source_uri=collector.url,
        downloaded_at_utc=utc_now_iso(),
        requested_start_ns=first_received_ns,
        requested_end_ns=last_received_ns + 1,
    )

    deltas = [item.delta for item in captured]
    reconstructions = [
        reconstruct_snapshot_and_deltas(
            epoch_snapshot,
            [delta for delta in deltas if delta.continuity_id == continuity_id],
        )
        for continuity_id, epoch_snapshot in snapshots.items()
    ]
    status: Literal["LIVE", "GAPPED", "INVALID"] = (
        "INVALID"
        if any(item.status == "INVALID" for item in reconstructions)
        else "GAPPED"
        if any(item.status == "GAPPED" for item in reconstructions)
        else "LIVE"
    )
    reconstruction = ReconstructionResult(
        status=status,
        observations=pa.concat_tables([item.observations for item in reconstructions]),
        gaps=pa.concat_tables([item.gaps for item in reconstructions]),
        stale_events=sum(item.stale_events for item in reconstructions),
        final_update_id=reconstructions[-1].final_update_id,
    )
    normalized_root = output_root / "normalized"
    tables = {
        "book_snapshots": (snapshots_table(list(snapshots.values())), "received_ts_ns"),
        "depth_deltas": (deltas_table(deltas), "event_ts_ns"),
        "book_observations": (reconstruction.observations, "event_ts_ns"),
        "sequence_gaps": (reconstruction.gaps, "detected_ts_ns"),
    }
    for schema_name, (table, time_column) in tables.items():
        write_partitioned_parquet(
            table.to_batches(max_chunksize=100_000),
            root=normalized_root,
            dataset=schema_name,
            schema_name=schema_name,
            source="binance_spot_live_depth",
            source_uri=collector.url,
            source_checksum_sha256=None,
            time_column=time_column,
        )
    delta_quality = validate_table(depth_deltas_table(captured), "depth_deltas")
    quality_reports = [delta_quality]
    if reconstruction.observations.num_rows:
        quality_reports.append(validate_table(reconstruction.observations, "book_observations"))
    quality_root = output_root / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    for report in quality_reports:
        report.write_json(quality_root / f"live_{report.dataset}.validation.json")
    write_json(
        quality_root / "live_depth_capture.summary.json",
        {
            "generated_at_utc": utc_now_iso(),
            "messages": len(captured),
            "continuity_epochs": len(snapshots),
            "normalized_messages": len(deltas),
            "excluded_messages": 0,
            "reconstruction_status": reconstruction.status,
            "quality_errors": sum(report.error_count for report in quality_reports),
            "quality_warnings": sum(report.warning_count for report in quality_reports),
            "policy": "every reconnect epoch receives a fresh snapshot; no captured delta is dropped",
        },
    )
    return captured, reconstruction, str(raw_manifest), raw_manifest_sha


def _cmd_collect_l2(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    captured, reconstruction, raw_manifest, raw_manifest_sha = asyncio.run(
        _capture_depth(
            symbol=str(args.symbol).upper(),
            max_messages=int(args.max_messages),
            output_root=output_root,
        )
    )
    _print_json(
        {
            "symbol": str(args.symbol).upper(),
            "messages": len(captured),
            "reconstruction_status": reconstruction.status,
            "book_observations": reconstruction.observations.num_rows,
            "sequence_gaps": reconstruction.gaps.num_rows,
            "stale_events": reconstruction.stale_events,
            "final_update_id": reconstruction.final_update_id,
            "raw_manifest": raw_manifest,
            "raw_manifest_sha256": raw_manifest_sha,
            "output_root": output_root,
            "live_trading": False,
        }
    )
    return 0 if reconstruction.status == "LIVE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microstructure",
        description="Research-only event-driven market-microstructure system",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest configured synthetic or public data")
    ingest.add_argument("--config", required=True, type=Path)
    ingest.add_argument("--output-root", type=Path)
    ingest.set_defaults(handler=_cmd_ingest)

    validate = subparsers.add_parser("validate", help="run non-mutating data validation")
    validate.add_argument("--config", required=True, type=Path)
    validate.set_defaults(handler=_cmd_validate)

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="produce or verify an immutable end-to-end sample run"
    )
    reproduce_parser.add_argument("--config", required=True, type=Path)
    reproduce_parser.add_argument("--run-dir", required=True, type=Path)
    reproduce_parser.set_defaults(handler=_cmd_reproduce)

    verify = subparsers.add_parser("verify", help="verify a frozen run and all checksums")
    verify.add_argument("--run-dir", required=True, type=Path)
    verify.set_defaults(handler=_cmd_verify)

    report = subparsers.add_parser("report", help="render reports from a frozen run")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--output-dir", type=Path)
    report.set_defaults(handler=_cmd_report)

    collect = subparsers.add_parser(
        "collect-l2", help="capture and reconstruct public live L2 data; never place orders"
    )
    collect.add_argument("--symbol", choices=("BTCUSDT", "ETHUSDT"), required=True)
    collect.add_argument("--max-messages", type=int, default=1_000)
    collect.add_argument("--output-root", type=Path, default=Path("data"))
    collect.set_defaults(handler=_cmd_collect_l2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_messages", 1) < 1:
        parser.error("--max-messages must be positive")
    handler = args.handler
    try:
        return int(handler(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("operation canceled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
