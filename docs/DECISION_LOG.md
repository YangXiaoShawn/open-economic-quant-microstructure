# Decision log

## 2026-08-09 — Supersede the incomplete Aug 8–11 campaign with a new prospective v2 calendar

- **Observed boundary:** The v1 Aug 8 session is an immutable `MISSED_WINDOW`,
  the Aug 9 validation session is immutable and complete, and the v1 development
  authority is `NOT_CREATED`. Those facts and files remain preserved.
- **User decision:** Abandon v1 as the active empirical study and, before any of
  the new dates is observed, freeze Aug 10 train, Aug 11 validation, Aug 12
  primary test, and Aug 13 replication test at 14:00–15:00 UTC.
- **Integrity rule:** This is a new campaign/version and storage/source
  authority, not a replacement bundle inside v1. V1 observations are not used
  for v2 training, selection, evaluation, or reporting. Once Aug 10 begins, no
  v2 date, threshold, feature, model, or interpretation rule may be changed.
- **Consequence:** A complete v2 train and validation may create the eight-state
  `LOCKED` development authority before Aug 12; otherwise v2 terminates through
  its existing `NOT_CREATED`/`INSUFFICIENT_DATA` branches without substitution.

## 2026-08-08 — Preserve the missed first L2 window as control evidence only

- **Observed fact:** The declared Aug 8 14:00--15:00 UTC train window ended
  before a clean committed producer authority was available. No capture command
  ran, `data/m8_l2` remained absent, and no raw or economic field was opened.
- **Decision:** Never backfill, replace, or infer that session. Once the release
  candidate has a clean source authority, invoke the already-tested late-start
  path solely to publish `INSUFFICIENT_DATA / MISSED_WINDOW`; it must not call a
  symbol capture adapter or network endpoint. Verify and retain that terminal,
  then continue Aug 9--11 on their declared windows under the same campaign
  authority.
- **Consequence:** The four-date study cannot promote a fitted development lock,
  predictive metric, descriptive result, or execution scenario. Its honest final
  outcome is necessarily aggregate `INSUFFICIENT_DATA`, backed by one missed
  control terminal plus the remaining declared session authorities.

## 2026-08-08 — Make development insufficiency a positive authority

- **Problem:** A valid `INSUFFICIENT_DATA` terminal on Aug 8 or Aug 9 forbids
  fitting, but a missing development lock cannot authorize the final four-date
  terminal and is indistinguishable from an interrupted workflow.
- **Decision:** The Aug 9 command always atomically publishes exactly one
  development authority. `LOCKED` contains the eight fitted child states and
  uses `_LOCKED` bytes `locked\n`. `NOT_CREATED` contains only typed,
  recursively verified session-control reasons, uses `_NOT_CREATED` bytes
  `not-created\n`, and must not load any economic frame. Both statuses expose the
  same canonical authority path and SHA fields; a valid `NOT_CREATED` command
  or verification exits 1 rather than masquerading as a system error.
- **Consequence:** Aug 10/11 are still captured on schedule. After Aug 11 the
  one final producer verifies all four session controls and the development
  authority. A `NOT_CREATED` branch publishes aggregate `INSUFFICIENT_DATA`,
  reports the union of development and held-out reasons, and contains no
  Parquet, model, prediction, descriptive, or execution artifact.

## 2026-08-08 — Complete the explicit-authority L2 terminal producer

- **Decision:** Expose one operational path through
  `lock-m8-l2-development`, `verify-m8-l2-development-lock`,
  `reproduce-m8-l2`, `verify-m8-l2-run`, and `report-m8-l2`. Every stage takes
  explicit bundle paths and independently supplied manifest/checksum SHA-256
  authorities; development and final verification additionally require the
  exact lock and run-control digests. No command discovers a "latest" input.
- **Terminal semantics:** After both held-out bundles' base authorities are
  verified, any non-`COMPLETE` held-out session publishes aggregate
  `INSUFFICIENT_DATA` without opening either held-out economic frame. If all
  sessions are complete but an endpoint has no eligible held-out labels, the
  same typed terminal is published without predictive, descriptive, or execution
  promotion. Otherwise the producer restores the eight locked states without
  refit, writes all declared evaluation/descriptive/market-scenario artifacts,
  snapshots its external authorities, checksums the exact inventory, writes
  `_SUCCESS` last, and immediately performs recursive verification. The exact
  final marker bytes are `complete\n` for `_SUCCESS` and `terminal\n` for
  `INSUFFICIENT_DATA`.
- **Reporting:** Both the canonical trade-M8 failure report and live-L2 reports
  are rendered only after verification into a directory outside the immutable
  run. The generated report-input snapshot is checksummed and re-rendered for
  equality; report commands state that the source bundle was not modified.

## 2026-08-08 — Bind one outcome-blind L2 campaign to its runtime and storage root

- **Decision:** The first predeclared capture creates one random 256-bit,
  outcome-blind campaign nonce and binds all four sessions to one canonical
  output-root path plus its filesystem device/inode identity. Moving to a
  different root or replacing that directory is rejected before capture rather
  than treated as a continuation of the campaign.
- **Runtime authority:** Bind the clean commit/source-tree identity, loaded
  package/module origin, Python/platform identity, and the exact versions of the
  eight production dependencies. Persist the canonical runtime payload and its
  SHA-256 and revalidate it throughout orchestration and later input/final-run
  verification.
- **Reason:** A commit hash alone does not prove that all dates used the same
  interpreter, dependency environment, imported source tree, physical evidence
  root, or prospectively chosen campaign instance.

## 2026-08-08 — Accept the frozen trade-only M8 data-insufficiency terminal

- **Observed evidence:** The corrected clean-source run at commit
  `88060613abe211cd8e80a3499678fca830f8ba2d` normalized 2,071,461 BTCUSDT
  training rows with no findings, then normalized 987,297 ETHUSDT training rows
  with zero errors and 53 `temporal.long_silence` warnings. That violated the
  predeclared zero-warning gate.
- **Decision:** Treat `artifacts/runs/binance-m8-multidate` as the canonical
  `INSUFFICIENT_DATA` terminal. Preserve its complete failed-normalization
  evidence and the earlier noncanonical layout-defect terminal unchanged. Do not
  replace the date, relax the warning rule, or reinterpret insufficiency as a
  failed economic hypothesis.
- **Boundary proved by the terminal:** Selection did not start; no analysis lock,
  fitted state, prediction, endpoint, held-out member, execution, P&L, capacity,
  or significance result was produced. The trade-only study is closed; live-L2
  evidence remains a separate prospective campaign.

## 2026-08-08 — Freeze the complete live-L2 analysis contract before capture

- **Authority:** The exact analysis TOML has source SHA-256
  `71edf7eeb9d5e935a18b0d8e354dc29b5b1132ace8eccd577730572d2caa8617`
  and semantic SHA-256
  `eeb9ac23ff275f26de57533a317d8165a89e99a14e86404a667cc69f6477bdac`.
  It binds capture-config SHA-256
  `491b14727a3e8bad907d1ad64072f6ebc14e407f98a5c31fca7b0a9e6801e758`
  and capture-protocol SHA-256
  `fe6d4aea5af3e9c529486b7e108afefeba623bf5ad3cc0743c0426a5e62e1fa7`.
  Every declared TOML field is rendered without amendment in
  `docs/M8_L2_ANALYSIS_CONTRACT.md` and enforced by a fail-closed loader.
- **Development boundary:** Fit volatility regimes on Aug 8 only; select each of
  the two-symbol by four-endpoint candidates on Aug 9; persist final fitted model,
  prior, preprocessing, calibration, regimes, and execution reference in eight
  child locks plus one aggregate lock before either held-out session is exposed.
- **Held-out and claims boundary:** Aug 10/11 restore locked state without fit,
  refit, recalibration, threshold change, or regime update. Evaluation uses the
  four declared horizons and 2,000-draw paired moving blocks. Execution is a
  market-order scenario only. Capacity, realized execution, and profitability
  claims remain forbidden irrespective of the result.

## 2026-08-08 — Keep raw authority separate from derived M8 failure evidence

- **Discovery:** The first clean-source development attempt stopped at its
  frozen data-quality gate before selection, locks, or held-out access, but its
  terminal bundle could not be reused: normalized and quality artifacts had
  been written inside the directory that the raw-acquisition verifier correctly
  requires to be an exact raw-only authority.
- **Decision:** Preserve the bundled raw authority unchanged below
  `data/input`; write normalized and DQ artifacts below
  `data/normalized_input`; root the final cross-stage manifest at `data`.
  `INSUFFICIENT_DATA` inventories must exactly match the physical regular-file
  tree and bind completed or failed normalization evidence. The producer runs
  the external reuse verifier before and after atomic publication.
- **Evidence policy:** The original source-tagged attempt is retained rather
  than repaired or overwritten. It is not canonical research evidence, and its
  failure opened no declared held-out member. A fresh terminal requires a new
  clean committed source identity; the frozen dates and quality policy do not
  change.

## 2026-08-08 — Lock transparent final fitted state before held-out access

- **Decision:** Fit and calibrate the validation-selected model and an independent
  historical prior once on train plus validation. Serialize canonical numeric
  preprocessing, estimator, calibration, feature-order, cutoff, and fallback
  state into each child lock; bind both state hashes into the aggregate lock.
- **Held-out rule:** Primary and replication prediction restore those numeric
  states. No classifier/calibrator fit, refit, recalibration, or online update is
  permitted after the aggregate lock becomes durable.
- **Reason:** A lock that committed only a future refit policy did not freeze the
  actual model used on untouched data.

## 2026-08-08 — Separate deterministic acquisition insufficiency from system faults

- **Decision:** Declared-object absence and authenticated metadata/CHECKSUM/ZIP,
  response-size, or total-evidence-budget violations produce a typed immutable
  raw-only `INSUFFICIENT_DATA` authority. Transient-network exhaustion,
  permission/local-I/O errors, collisions, and program faults do not terminalize.
- **Reason:** Deterministic missing/invalid declared evidence consumes the frozen
  study, while an operational failure must remain safely retryable and cannot be
  recorded as an empirical outcome.

## 2026-08-08 — Define prospective L2 coverage by observed book-state intervals

- **Decision:** A frozen session counts only consecutive receipt-time intervals
  backed by `OBSERVED` reconstructed states within one continuity epoch. Stale,
  excluded, gapped, invalid, and silent intervals are not bridged. Cross-symbol
  coverage is the intersection of each symbol's interval union.
- **Publication:** Both symbols share one absolute UTC barrier and one atomic
  terminal authority. Failed data/gates publish `INSUFFICIENT_DATA`; system
  faults preserve nonterminal raw evidence. No failed date is replaced.
- **Reason:** Raw first/last websocket timestamps can hide reconnects and silent
  holes and therefore cannot prove usable simultaneous book coverage.

## 2026-08-07 — Enforce raw-only acquisition and lock-before-open execution

- **Decision:** Split M8 into an immutable raw authority and a one-way research
  producer. Acquisition authenticates exchange metadata, eight ZIPs, eight
  official CHECKSUM responses, and bounded ZIP directory metadata, but cannot
  open a CSV member. The producer normalizes only train/validation, persists two
  symbol locks and an aggregate lock, then revalidates every committed identity
  immediately before each held-out member is opened.
- **Reason:** Merely delaying a later Parquet scan would not preserve the
  prospective boundary if held-out economic rows had already been decompressed.
  The first decompressed member byte is therefore the enforced boundary.
- **Failure policy:** Deterministic data insufficiency before or after locking is
  an immutable terminal result with no replacement date, endpoint prediction,
  execution result, or profitability/significance claim. Unexpected system
  failures publish no partial research target.
- **Protocol effect:** This is operational hardening only. It changes no date,
  feature, candidate, endpoint, hypothesis, estimand, or interpretation rule.

## 2026-08-07 — Count every retained raw-evidence byte and physical copy

- **Decision:** Use one reservation ledger for accepted responses, retry/error
  prefixes, CHECKSUM bodies, exchange metadata, and every source sidecar. Raw
  manifests enumerate the exact physical inventory. A self-contained run copy
  is a distinct retained copy and must fit under the same frozen total ceiling
  before its first byte is written.
- **Reason:** Per-response limits alone do not bound accumulated retries,
  sidecars, or duplicated evidence on a 16 GB local machine. Post-hoc counting
  could leave an oversized partial publication.
- **Consequence:** Raw response publication is fail-closed and rollback-safe;
  normalized Parquet and machine-generated manifest indexes remain outside the
  raw-response byte ceiling, as declared by the protocol.

## 2026-08-07 — Freeze future live-L2 sessions before observation

- **Decision:** Reserve 14:00–15:00 UTC on 2026-08-08 through 2026-08-11 for
  concurrent BTCUSDT/ETHUSDT development, validation, primary-test, and
  replication-test captures.
- **Acceptance:** Both symbols need at least 3,300 seconds of overlapping
  receipt-time coverage, a 1,800-second continuous valid epoch, zero sequence
  gaps, zero DQ findings, exact row reconciliation, and complete immutable
  capture evidence.
- **Reason:** Fixed future sessions prevent outcome-based window choice and make
  disconnects or missing data visible failures rather than hidden replacements.
- **Claim boundary:** The first protocol is book-only. It permits future-mid
  prediction and market-order scenarios, but not limit-fill, realized execution,
  capacity, significance, or profitability claims.

## 2026-08-07 — Record the Jan 6 coverage-metadata race

- **Discovery:** Before a stop message reached the parallel feasibility audit,
  it completed the same official archive availability, checksum, byte/row
  count, aggregate-trade-ID boundary, and timestamp-boundary checks for Jan 6.
  It did not inspect or retain price, quantity, maker direction, class balance,
  features, labels, or model results.
- **Correction:** Protocol 1.0.2 records coverage-only inspection for all four
  dates. The calendar, roles, hypotheses, model grid, and interpretation rules
  were already frozen and remain unchanged.
- **Boundary:** This metadata helps verify feasibility and completeness only. It
  is not economic evidence and cannot justify changing or dropping a date.

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
