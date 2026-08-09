# Status

Last updated: 2026-08-09 UTC

> **Replacement-campaign source-freeze snapshot.** This tracked file records the
> state before the v2 four-date L2 campaign. It must not be edited between the
> first Aug 10 capture and the Aug 13 terminal. During that interval, current
> status is authoritative only in `data/m8_l2/campaign_authority.json` and the
> immutable per-session bundles; after the campaign, generated final-run
> provenance is the authority for results.

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

The frozen trade-only M8 study has reached its canonical terminal state. The
raw-only authority binds two exchange-metadata responses, eight official
ZIP/CHECKSUM pairs, and 36 retained artifacts totaling 117,897,562 bytes; every
acquisition entry records `csv_member_opened=false` and
`economic_fields_inspected=false`. The corrected clean-source producer at commit
`88060613abe211cd8e80a3499678fca830f8ba2d` published the checksummed
`artifacts/runs/binance-m8-multidate` bundle with status
`INSUFFICIENT_DATA`. BTCUSDT training normalization completed with 2,071,461
rows, zero errors, and zero warnings. ETHUSDT training normalization completed
with 987,297 rows, zero errors, and 53 warnings, triggering the predeclared
quality gate. Selection never started; no development lock or prediction was
published; no validation or held-out archive member was opened; execution and
P&L are `NOT_RUN`. No date or policy was replaced. Its report command revalidates
the terminal/raw authority and writes a fresh external report without modifying
the canonical bundle.

The live-L2 campaign remains the active evidence milestone. The superseded v1
campaign is preserved, not repaired: Aug 8 is `MISSED_WINDOW`, Aug 9 is a
verified complete session, and its development authority is `NOT_CREATED`.
Before observing any v2 session, the user reset the active prospective calendar
to Aug 10 train, Aug 11 validation, Aug 12 primary test, and Aug 13 replication
test. V1 evidence is not an input to v2. The v2 capture config, protocol, and
analysis authority are separately hash-bound. The campaign-authority mechanism
requires all four dates to share one outcome-blind nonce, canonical output-root
filesystem identity, clean commit/source/import origin, and hashed
Python/platform/production-dependency fingerprint. Strict capture and session
verification, verified lazy inputs, observed-interval endpoint frames,
Aug 10/11 `LOCKED | NOT_CREATED` development authority, no-refit Aug 12/13 evaluation, dependency-aware
diagnostics, market-only scenarios, descriptive analyses, and artifact-driven
L2 reporting are integrated in a tested end-to-end producer and CLI/Make path.
Development and final production have explicit 8 GiB and 12 GiB fail-closed
live-memory envelopes respectively; crossing any admission or post-allocation
bound is a system failure and cannot publish a research terminal.
The atomic final bundle is either `COMPLETE` or `INSUFFICIENT_DATA`; it embeds
exact config/protocol, campaign, session-control, and development authority
snapshots, while its verifier also revalidates every external authority. Reports
are regenerated only into an external directory, leaving the immutable run
untouched. The software path was complete at this remaining-campaign source
freeze; the real v2 session-control evidence was not. No v2 session bundle or
L2 empirical metric existed when these tracked bytes were frozen.

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
| M8 Broader empirical study | In progress | Trade-only branch closed with a canonical `INSUFFICIENT_DATA` terminal; complete frozen L2 software path verified offline; four prospective local-L2 session terminals and the resulting empirical terminal remain required |

## Verification ledger

- 2026-08-08: the final L2 session, input, development-lock, total-producer,
  recursive-verifier, CLI/Make, and external-report paths passed their focused
  offline suites. A consolidated repository gate is run separately before source
  freeze; its volatile test count is not hardcoded in this market-evidence file.
- 2026-08-08: every path listed by the canonical trade M8
  `checksums.sha256` reverified. Its manifest/failure/provenance agree on
  `INSUFFICIENT_DATA`, the clean source identity above, zero selection, zero
  held-out access, and `NOT_RUN` execution.
- 2026-08-07: `make lint` passed across source, tests, and dashboard.
- 2026-08-08: the pre-L2 baseline `make check` passed Ruff, strict mypy across 40 package
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
- Inconclusive by design: the full-archive trade M8 hypothesis was not evaluated
  because ETHUSDT training data failed the frozen warning gate. The terminal is
  evidence of data insufficiency, not evidence that the hypothesis failed.
- Failed/unsupported hypothesis disclosure is generated per symbol for runs
  that reach evaluation. The single capped date cannot test persistence or
  book/liquidity hypotheses, and synthetic output cannot evaluate a market
  hypothesis.

## Next evidence sequence

Retain the canonical trade-only `INSUFFICIENT_DATA` bundle and its source-tagged
predecessor unchanged; do not relax the warning gate, replace a date, or rerun
until a favorable outcome appears. Use the completed L2 producer and explicit
authority interfaces with ignored targets
`artifacts/runs/binance-m8-l2-development-lock` and
`artifacts/runs/binance-m8-live-l2` for the development authority and final run,
respectively. Then capture the exact simultaneous Aug 10--13 BTCUSDT/ETHUSDT
sessions under one new clean v2 campaign authority. After Aug 11, durably publish
either the fitted `LOCKED` authority or the control-only `NOT_CREATED` authority
before Aug 12. A valid `NOT_CREATED` result exits 1 but does not cancel the Aug
12/13 captures; the final producer uses all four control authorities without
opening economic frames. After Aug 13, produce and recursively verify the immutable
final bundle, render reports externally, measure peak RSS, and complete the
clean-room/resource audit. A failed/missed date terminalizes the declared
campaign rather than selecting a substitute. The
frozen facts are in `docs/M8_L2_ANALYSIS_CONTRACT.md`; generated bundles remain
the only authority for any eventual metrics.
