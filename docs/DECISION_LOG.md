# Decision log

## 2026-08-07 — Correct the M8 freeze claim to outcome-blind

- **Discovery:** A parallel feasibility audit read official Jan 3–5 archive
  availability, checksum, byte/row count, aggregate-trade-ID boundary, and
  timestamp-boundary metadata before commit `a34ba13`. It did not inspect or
  retain economic fields, class balance, features, labels, or model results;
  Jan 6 was not inspected.
- **Correction:** Protocol 1.0.1 describes the study as outcome-blind rather
  than claiming it was fully acquisition/inspection-blind. The calendar and all
  roles remain unchanged because they were selected mechanically before those
  metadata were reported.
- **Constraint:** No economic field or model outcome from any declared date may
  be inspected until the ingestion, analysis lock, and final-test gates are
  implemented. Coverage-only facts cannot be used to replace a date.

## 2026-08-07 — Freeze the M8 multi-date trade calendar before acquisition

- **Decision:** Reserve the complete 2024-01-03 and 2024-01-04 UTC Binance Spot
  daily aggregate-trade archives for training and validation, and reserve the
  adjacent 2024-01-05 and 2024-01-06 archives as primary and replication tests.
  Both BTCUSDT and ETHUSDT are mandatory. The inspected 2024-01-02 sample is
  excluded from all study estimates.
- **Reason:** Adjacent dates chosen mechanically before acquisition provide an
  auditable barrier against outcome-based date selection. Full daily archives
  remove the current row-cap truncation while remaining feasible on local disk
  with streaming normalization.
- **Evaluation:** Model selection uses validation log loss only. A locked model
  is then evaluated without refit on both untouched dates, using paired
  40-trade-block loss differences and explicit direction-replication status.
- **Claim boundary:** This is a complete-data trade-only study. Execution, book,
  fill, P&L, capacity, significance, and cross-instrument pooling remain
  unauthorized. M8 still requires separately frozen continuous real L2 evidence.
- **Failure policy:** Missing, oversized, corrupt, noncontiguous, or otherwise
  invalid declared data produces `INSUFFICIENT_DATA`; dates are never replaced.

Material choices are appended; prior entries are not rewritten to hide reversals.

## 2026-08-07 — Initialize the empty `Microstructure` directory

- **Context:** The requested `microstructure` folder exists as `Microstructure`
  and contains no files or Git metadata.
- **Decision:** Treat the capitalized directory as the target and initialize a
  clean Python research repository there.
- **Consequence:** There is no existing user work to merge or preserve inside the
  target; all new paths are scoped to this directory.

## 2026-08-07 — Separate offline software evidence from market evidence

- **Context:** A new user must reproduce a small run without a large download,
  while the project must not invent empirical findings.
- **Decision:** Make the default smoke/reproduction data a deterministic,
  explicitly synthetic fixture generated from declared rules. Keep public
  Binance ingestion as a separate, credential-free command and never describe
  fixture metrics as empirical market results.
- **Consequence:** The vertical slice can be tested offline. Economic conclusions
  remain deliberately unavailable until a manifested public-data run is made.

## 2026-08-07 — Use trade events for the initial research slice

- **Context:** Historical trades are broadly public and compact; historical full
  depth is less consistently public. The objective explicitly permits trades or
  a small book sample for the first slice.
- **Decision:** Build the first model dataset from signed trades and event-time
  liquidity proxies. Implement and test L2 snapshot/delta reconstruction as a
  separate adapter path, but do not pretend trade-only data identify queue fills.
- **Consequence:** Initial fill probability and queue position are declared
  assumptions. Later L2 runs can replace them through the same interfaces.

## 2026-08-07 — Prefer integer event ordering plus UTC nanoseconds

- **Context:** Exchange timestamps may tie and floating timestamps can obscure
  exact temporal order.
- **Decision:** Normalize `event_id`, `sequence`, `event_ts_ns`, and
  `received_ts_ns`. Stable event ordering is `(event_ts_ns, sequence, event_id)`;
  receipt time is retained for latency realism.
- **Consequence:** Features and labels can state observable cutoffs explicitly;
  downstream modules must not reorder tied events arbitrarily.

## 2026-08-07 — Make validation non-mutating

- **Context:** Silent repair can erase evidence of feed or clock problems.
- **Decision:** Validators emit typed findings and summaries but never mutate raw
  or normalized observations. Any later filtering writes a new derived dataset
  and records exclusion counts.
- **Consequence:** A run can fail on fatal findings or continue on declared
  warnings without altering source evidence.

## 2026-08-07 — Evaluate event-time models with purged walk-forward folds

- **Context:** Random splits leak regime information and overlapping future-label
  horizons leak outcomes across adjacent partitions.
- **Decision:** Use expanding-window, time-ordered folds. Remove training rows
  whose label end reaches the validation/test boundary and support a configurable
  embargo. Reserve the last fold for final comparison, not model selection.
- **Consequence:** Small fixtures may yield wide uncertainty; that is preferable
  to optimistic metrics.

## 2026-08-07 — Do not infer historical Spot depth from trade archives

- **Context:** Binance's credential-free Spot archive exposes trades and
  aggregate trades, but not a complete historical Level-2 feed suitable for book
  reconstruction. REST snapshots are current anchors rather than historical
  depth observations.
- **Decision:** Use fixed, bounded public REST aggregate trades for historical
  ingestion. Keep Level-2 research behind an optional public live diff-depth
  collector that begins with a locally received snapshot and creates a new epoch
  after every reconnect or gap.
- **Consequence:** Historical trade research and book research are different
  evidence paths. No queue, cancellation, or depth claim is inferred from the
  downloaded trade sample.

## 2026-08-07 — Derive numerical scales from exchange metadata

- **Context:** Fixed decimal precision can misrepresent instruments and can
  change across venue rules. Binance aggregate-trade timestamps also changed
  archive units for newer files, while REST uses its documented request units.
- **Decision:** Fetch `PRICE_FILTER.tickSize` and `LOT_SIZE.stepSize` from public
  `exchangeInfo` before normalization, retain integer ticks/lots plus the scale,
  and make timestamp-unit conversion explicit at the adapter boundary.
- **Consequence:** Exact source values are reproducible without assuming a
  universal `1e-8` quantum. Low-level fallback scales are not used by the sample
  workflow.

## 2026-08-07 — Freeze results as immutable atomic bundles

- **Context:** A dashboard or report must never read half-written results, and a
  repeated command must not silently revise earlier evidence.
- **Decision:** Produce into a sibling staging directory, serialize all inputs,
  folds, metrics, ledgers, assumptions and reports, checksum every file, create
  `_SUCCESS` last, verify the bundle, then atomically rename it. A completed
  target is reusable only if verification passes; incomplete or corrupted
  targets are rejected rather than repaired.
- **Consequence:** Changed configuration or assumptions require a new run. The
  deterministic semantic run key is based on config, immutable input identity,
  Git state and seed, excluding wall-clock generation time.

## 2026-08-07 — Keep execution conditional and out-of-sample

- **Context:** Archived public events cannot identify a strategy's true queue
  position, endogenous impact, or colocated latency. Replaying in-sample
  probabilities would further overstate execution evidence.
- **Decision:** Accept only validation-selected, held-out test predictions in the
  execution simulator. Treat market depth, limit-fill Bernoulli behavior, queue
  ahead, latency, fees, liquidation and capacity as serialized scenario inputs;
  use common keyed randomness for comparable sensitivities.
- **Consequence:** Predictive and execution tables remain separate. Simulated
  results are conditional diagnostics, not realized or deployable performance.

## 2026-08-07 — Cap and label the first public sample

- **Context:** A local Apple Silicon workflow needs a small, auditable public
  acquisition rather than an open-ended download.
- **Decision:** Request the fixed interval beginning 2024-01-02 00:00 UTC for
  BTCUSDT and ETHUSDT, capped at 5,000 aggregate trades per symbol, and label the
  acquisition `PUBLIC_SAMPLE_PARTIAL`.
- **Consequence:** The cap was reached for both symbols. All 10,000 normalized
  rows passed the current validators, but both requested ranges are recorded as
  incomplete and no economic hypothesis is promoted from this sample.

## 2026-08-07 — Freeze the trade-only public study before model comparison

- **Context:** The already acquired public sample contains aggregate trades but
  no contemporaneous order-book state. Reusing book features or the execution
  simulator would turn missing observations into assumptions and overstate the
  evidence.
- **Decision:** Freeze a retrospective, explicitly exploratory trade-only
  protocol with per-symbol causal features, strictly later trade labels,
  purged walk-forward folds, validation-only model selection, and a paired
  fixed-block comparison against the historical prior. Require the caller to
  select the ingestion manifest by both path and SHA-256; never scan for a
  "latest" input. Serialize execution artifacts as `NOT_RUN` with a reason.
- **Consequence:** BTCUSDT and ETHUSDT are reported separately, including failed
  or contradictory outcomes. The run cannot be interpreted as a book, fill,
  P&L, statistical-significance, or persistent-alpha study.

## 2026-08-07 — Make page/batch streaming the default public ingestion path

- **Context:** A configurable row cap made the first 10,000-row sample safe, but
  the original downloader, validator, and ingestion composition retained the
  entire acquisition in memory. A cap on observations is not a RAM contract and
  does not satisfy the larger-history requirement.
- **Decision:** Expose a lazy, page-bounded Binance iterator; feed each page once
  through an incremental validator with disk-backed exact identity state and
  then into a batch-bounded Parquet writer. Keep eager materialization only as an
  explicit compatibility operation with its own finite row guard.
- **Consequence:** Public acquisition memory is bounded by response/batch and
  validator preview limits rather than total history length. Manifest and part
  metadata may still grow with page count; downstream full-history readers must
  likewise use verified batch scans instead of eager Arrow/Polars copies.

## 2026-08-07 — Verify public history before exposing a bounded canonical stream

- **Context:** Hashing only an ingestion manifest does not prove that normalized
  rows still match the declared raw pages, source symbol, exchange metadata, or
  physical arrival order. Sorting first can also conceal source-order defects.
- **Decision:** Bind the explicit ingestion path+SHA to configuration, raw URI
  semantics and exact aggregate-trade fields; require inverse raw-to-normalized
  coverage; run physical-order incremental DQ; and feed that single verified
  Arrow stream into a memory-limited DuckDB external sort. Keep eager loading as
  a separately guarded compatibility API.
- **Consequence:** The current 10,000-row input can be materialized for the
  predeclared small study, while larger consumers can iterate bounded canonical
  batches without trusting a global in-memory sort.

## 2026-08-07 — Make quality evidence atomic and ingestion-manifested

- **Context:** Reusing a fixed findings filename could truncate a previous
  complete JSONL before a rerun succeeded, leaving an older summary pointing at
  partial/new evidence.
- **Decision:** Stream findings into a same-directory temporary file, fsync and
  atomically publish only on validator completion, use a distinct per-ingestion
  quality filename, and bind report/findings paths, byte counts and SHA-256 into
  the immutable ingestion manifest.
- **Consequence:** A failed rerun cannot damage previously published findings;
  complete quality evidence is independently integrity-checkable.

## 2026-08-07 — Treat the exact source tree as part of run identity

- **Context:** A Git commit plus `dirty=true` cannot distinguish different local
  patches. It also allowed an old clean bundle to satisfy `make smoke` after the
  implementation changed.
- **Decision:** Hash the bytes, paths and modes of every tracked and non-ignored
  untracked source file. Include that digest with commit/dirty state in
  provenance and both synthetic/public run keys; refuse completed-target reuse
  when any component differs. Run the `make check` smoke leg in a fresh temporary
  target.
- **Consequence:** Ignored raw/run artifacts do not perturb identity, but changed
  code, configuration, tests or documentation requires a new immutable bundle.

## 2026-08-07 — Journal live depth before parsing and reconstruct incrementally

- **Context:** Retaining every WebSocket payload, delta and reconstructed book
  row in Python made `collect-l2 --max-messages` an in-memory history limit. A
  malformed frame could also fail before its original bytes were preserved.
- **Decision:** Emit exact raw frames to a typed capture journal before decoding,
  record each reconnect snapshot anchor, enforce per-frame/batch byte ceilings,
  run a bounded-state incremental book reconstructor and incremental DQ, and
  stream capture-scoped Parquet output. Publish one immutable capture-ID summary
  last; keep the fixed summary filename only as a latest-pointer.
- **Consequence:** Capture memory is bounded by book depth and batch limits rather
  than message count; every epoch is resnapshotted, every message is reconciled
  to an observation/exclusion, and parse/sequence failure leaves explicit raw
  evidence without fabricating a completed capture.
