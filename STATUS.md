# Status

Last updated: 2026-08-08 UTC

## Current evidence

The portfolio-quality software vertical slice is complete. Its model and
execution output is labeled `SYNTHETIC_SMOKE` and supports only a software
reproducibility claim.

A credential-free `PUBLIC_SAMPLE_PARTIAL` ingestion and a frozen trade-only
research protocol were also completed for the
fixed 2024-01-02 UTC configuration: 5,000 aggregate trades each for BTCUSDT and
ETHUSDT. Both symbol ranges reached the configured cap and are explicitly
incomplete. The normalized 10,000 rows produced zero current validation errors
and warnings. This supports a bounded data-pipeline observation only—not a market
signal, statistical-significance, profitability, or capacity claim. The public
producer uses per-symbol purged folds and paired model-minus-prior uncertainty;
execution and P&L are explicitly `NOT_RUN` because no contemporaneous book is
present.

The prospective M8 implementation now has a raw-only acquisition authority,
strict retained-evidence byte accounting, bounded ZIP-directory verification,
development-only normalization, durable per-symbol and aggregate analysis
locks, immediate held-out member-open revalidation, and immutable complete or
`INSUFFICIENT_DATA` terminal bundles. The frozen raw-only acquisition has now
completed: its manifest binds two exchange-metadata responses, eight official
ZIP/CHECKSUM pairs, and 36 retained artifacts totaling 117,897,562 bytes. Every
acquisition entry records `csv_member_opened=false` and
`economic_fields_inspected=false`, proving the raw-only acquisition phase did
not inspect economic rows. A later clean-source attempt opened development
training members only and stopped at the frozen DQ gate before selection,
locks, or held-out access. That source-tagged terminal is retained but not
promoted because it exposed a raw/derived bundle-layout verification defect.
The corrected producer requires a fresh clean-source, self-verified terminal;
its authoritative status and any metrics live only in the generated checksum
bundle and are not copied back into this source-controlled file.

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
| M8 Broader empirical study | In progress | Raw-only authority acquired and verified; canonical trade terminal/resource evidence is generated post-commit, while four prospective local-L2 sessions remain required |

## Verification ledger

- 2026-08-07: `make lint` passed across source, tests, and dashboard.
- 2026-08-08: final `make check` passed Ruff, strict mypy across 40 package
  modules, all 574 offline tests on Python 3.12.13, and a fresh current-source
  synthetic bundle. `ruff format --check` also passed across all 92 Python files.
- 2026-08-07: `make check` passed Ruff, strict mypy, all tests, and a fresh
  current-source synthetic bundle produced in an isolated temporary target.
- 2026-08-07: adversarial M8 tests proved that raw acquisition opens no CSV
  member, unsafe ZIP metadata is rejected before the standard ZIP parser, both
  symbol locks precede the first held-out open, tampered/missing authorities
  expose zero held-out rows, and deterministic data failures publish no endpoint.
- 2026-08-08: raw-only M8 acquisition published and independently reverified
  `data/m8/_manifests/m8-acquisition.manifest-04d5c01f3810b6a300ec.json`
  (SHA-256 `04d5c01f3810b6a300ec0f9317052f254b2bec5d89bc0dfefd18cd71ad6582e6`).
  Verification opened no CSV member and found no extra, missing, symlinked, or
  unmanifested raw artifact.
- 2026-08-08: the frozen dual-symbol L2 session core and Binance adapter passed
  82 joint tests. A real adapter-produced mock bundle exposed 19 artifacts that
  passed strict raw-journal, snapshot, Parquet footer/schema, quality, manifest,
  absolute-time, overlap, and 29-gate reconciliation.
- 2026-08-07: pipeline tests independently reproduced two semantically identical
  run bundles, verified their checksums, rejected corruption, and preserved an
  incomplete target without repair.
- 2026-08-07: `make download-sample` succeeded after network permission was
  granted; raw public responses and metadata were manifested and kept outside
  Git. `validate-public-data` passed 10,000 normalized rows with zero findings.
- 2026-08-07: clean-commit canonical synthetic and public bundles were produced,
  checksum-verified, and independently re-rendered into technical report, IC
  memo, and model table. Generated artifacts remain ignored rather than
  committed.

## Empirical claim register

- Supported: the fixed, capped public REST sample can be acquired, normalized
  with exchange-provided scales, content-hashed, partitioned, and validated.
- Exploratory diagnostic: the public producer persists each symbol's selected
  model, paired held-out model-minus-prior loss, fixed-block interval and status
  in a checksum-protected artifact. Exact metrics are read by generated reports
  and are not manually duplicated in this status file; instruments are not
  pooled.
- Not supported: confirmatory order-flow predictability, statistical
  significance, effect half-life, cross-date/instrument stability, economic
  profitability, live fill probability, or deployable capacity. Execution is
  `NOT_RUN` for the public trade-only input.
- Failed/unsupported hypothesis disclosure is generated per symbol. The single
  capped date cannot test persistence or book/liquidity hypotheses, and
  synthetic output cannot evaluate a market hypothesis.

## Post-commit evidence sequence

From a clean committed source tree, run `make reproduce-m8`,
`make verify-m8-run`, and `make report-m8` with those exact coordinates. Measure
peak resident memory on the complete eight-archive run and publish either the
complete trade-only evidence bundle or the protocol-declared
`INSUFFICIENT_DATA` result. Retain the current single-date bundle unchanged.
Use the acquired manifest path and SHA above; do not rediscover a "latest"
input. Separately collect the frozen simultaneous public L2 sessions required
for book-dependent claims; they cannot be inferred from the trade archives.
