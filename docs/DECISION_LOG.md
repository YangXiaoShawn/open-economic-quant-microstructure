# Decision log

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
