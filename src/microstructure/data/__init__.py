"""Market-data ingestion, normalization, storage, and validation primitives.

The package deliberately contains no authenticated or order-entry API.  Binance
support is restricted to public market-data endpoints.
"""

from microstructure.data.book import (
    BookSnapshot,
    DepthDelta,
    ReconstructionResult,
    reconstruct_snapshot_and_deltas,
)
from microstructure.data.quality import QualityFinding, ValidationReport, validate_table
from microstructure.data.schemas import SCHEMA_VERSION, get_schema, table_from_records
from microstructure.data.storage import DatasetWriteResult, write_partitioned_parquet
from microstructure.data.synthetic import SyntheticMarketData, generate_synthetic_market

__all__ = [
    "SCHEMA_VERSION",
    "BookSnapshot",
    "DatasetWriteResult",
    "DepthDelta",
    "QualityFinding",
    "ReconstructionResult",
    "SyntheticMarketData",
    "ValidationReport",
    "generate_synthetic_market",
    "get_schema",
    "reconstruct_snapshot_and_deltas",
    "table_from_records",
    "validate_table",
    "write_partitioned_parquet",
]
