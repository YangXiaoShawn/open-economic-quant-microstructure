"""Regression test for the retired unsafe all-calendar M8 entry point."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from microstructure.m8_config import M8StudyConfig
from microstructure.m8_ingestion import M8IngestionError, ingest_m8_archives


def test_legacy_all_calendar_ingestion_fails_before_any_adapter_call(tmp_path: Path) -> None:
    class ForbiddenAdapter:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"retired ingestion touched adapter attribute {name!r}")

    config = cast(M8StudyConfig, SimpleNamespace())
    with pytest.raises(M8IngestionError, match="before the analysis lock"):
        ingest_m8_archives(
            config,
            tmp_path,
            archive_client=ForbiddenAdapter(),
            metadata_provider=ForbiddenAdapter(),
        )
