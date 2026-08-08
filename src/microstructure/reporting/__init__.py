"""Frozen-bundle reporting and verification interfaces."""

from microstructure.reporting.bundle import (
    ChecksumMismatchError,
    IncompleteRunError,
    RunBundle,
    RunBundleError,
    RunBundleValidationError,
    evidence_watermark,
    load_run_bundle,
    verify_checksums,
    write_checksum_manifest,
)
from microstructure.reporting.render import (
    ReportPaths,
    render_executive_memo,
    render_model_comparison_report,
    render_technical_report,
    write_report_set,
)
from microstructure.reporting.tables import comparison_rows, render_model_comparison

__all__ = [
    "ChecksumMismatchError",
    "IncompleteRunError",
    "ReportPaths",
    "RunBundle",
    "RunBundleError",
    "RunBundleValidationError",
    "comparison_rows",
    "evidence_watermark",
    "load_run_bundle",
    "render_executive_memo",
    "render_model_comparison",
    "render_model_comparison_report",
    "render_technical_report",
    "verify_checksums",
    "write_checksum_manifest",
    "write_report_set",
]
