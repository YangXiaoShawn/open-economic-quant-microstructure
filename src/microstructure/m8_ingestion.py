"""Compatibility tombstone for the retired pre-lock M8 ingestion workflow.

The original entry point normalized all eight declared archive members before
the analysis lock existed.  Keeping that behavior callable would create a
direct held-out-data bypass.  Raw acquisition now lives in
``microstructure.m8_acquisition``; lock-gated, one-archive normalization lives
in ``microstructure.m8_normalization``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from microstructure.data.binance import SymbolMetadata
from microstructure.m8_config import M8StudyConfig


class M8IngestionError(RuntimeError):
    """Raised whenever the retired all-calendar ingestion API is invoked."""


def ingest_m8_archives(
    config: M8StudyConfig,
    output_root: str | Path,
    *,
    archive_client: object | None = None,
    metadata_provider: object | None = None,
    supplied_metadata: Mapping[str, SymbolMetadata] | None = None,
    batch_rows: int = 65_536,
) -> None:
    """Fail closed instead of opening held-out economic data before its lock."""

    del (
        config,
        output_root,
        archive_client,
        metadata_provider,
        supplied_metadata,
        batch_rows,
    )
    raise M8IngestionError(
        "ingest_m8_archives is retired because it opens held-out economic data "
        "before the analysis lock; use acquire_m8_archives followed by reproduce_m8"
    )


__all__ = ["M8IngestionError", "ingest_m8_archives"]
