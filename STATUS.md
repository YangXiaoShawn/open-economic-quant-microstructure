# Status

Last updated: 2026-08-07 UTC

## Current evidence

The portfolio-quality software vertical slice is complete. Its model and
execution output is labeled `SYNTHETIC_SMOKE` and supports only a software
reproducibility claim.

A credential-free `PUBLIC_SAMPLE_PARTIAL` ingestion was also completed for the
fixed 2024-01-02 UTC configuration: 5,000 aggregate trades each for BTCUSDT and
ETHUSDT. Both symbol ranges reached the configured cap and are explicitly
incomplete. The normalized 10,000 rows produced zero current validation errors
and warnings. This supports a bounded data-pipeline observation only—not a market
signal, statistical-significance, profitability, or capacity claim.

## Milestones

| Milestone | Status | Evidence |
| --- | --- | --- |
| M0 Repository contract | Complete | Required files, Python packaging, locked environment, typed configuration and provenance |
| M1 Event data foundation | Complete | Public/synthetic adapters, exact schemas, partitioned Parquet, immutable raw/normalized manifests |
| M2 Book and quality controls | Complete | Gap-safe snapshot/delta replay and non-mutating quality rules covered by tests |
| M3 Leakage-safe dataset | Complete | Causal features, future labels, censoring, continuity isolation, lineage audit |
| M4 Model evaluation | Complete | Purged walk-forward prior/unpenalized/L2/tree ladder with calibration and block bootstrap |
| M5 Execution research | Complete | OOS-only costs, two-stage latency, market/limit partial fills, queue proxy, inventory, liquidation and sensitivity |
| M6 Reproducible vertical slice | Complete | Atomic CLI producer, immutable checksum bundle, deterministic run key, idempotent verification |
| M7 Research communication | Complete | Code-generated report/memo/table, read-only six-tab dashboard, limitations and portfolio material |
| M8 Broader empirical study | Deferred | Requires multi-date public trades and contiguous locally captured L2 epochs under a frozen protocol |

## Verification ledger

- 2026-08-07: `make lint` passed across source, tests, and dashboard.
- 2026-08-07: `make typecheck` passed in strict mode across 25 package modules.
- 2026-08-07: `make test` passed all 90 offline tests on Python 3.12.13.
- 2026-08-07: pipeline tests independently reproduced two semantically identical
  run bundles, verified their checksums, rejected corruption, and preserved an
  incomplete target without repair.
- 2026-08-07: `make download-sample` succeeded after network permission was
  granted; raw public responses and metadata were manifested and kept outside
  Git. `validate-public-data` passed 10,000 normalized rows with zero findings.
- 2026-08-07: the canonical `make reproduce-sample`, `make verify-run`, generated
  report, and dashboard bundle are part of final acceptance. Generated artifacts
  remain ignored rather than committed.

## Empirical claim register

- Supported: the fixed, capped public REST sample can be acquired, normalized
  with exchange-provided scales, content-hashed, partitioned, and validated.
- Not supported: order-flow predictability, effect half-life, cross-instrument
  stability, economic profitability, live fill probability, or deployable
  capacity.
- Failed hypotheses: none can be evaluated from the synthetic run or the bounded
  trade-only ingestion. Treating the absence of a test as a null result would be
  incorrect.

## Immediate next milestone

Freeze a multi-date empirical protocol before looking at outcomes. Capture
continuous public L2 epochs locally for book-dependent claims, preserve adjacent
untouched dates for testing, require at least two temporal bootstrap blocks, and
publish both supported and failed hypotheses across instruments, regimes, and
execution sensitivities.
