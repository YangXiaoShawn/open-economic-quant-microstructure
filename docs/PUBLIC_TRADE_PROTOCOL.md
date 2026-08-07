# Public aggregate-trade exploratory protocol

## Evidence status

This protocol governs the first real-data research run built from the fixed,
capped Binance Spot aggregate-trade sample already acquired for 2024-01-02. It
is **retrospective and exploratory**, not a preregistered confirmatory study. The
data availability, per-symbol coverage, and class balance were inspected before
this document was frozen; model comparison and held-out results were not.

Every output must retain the `PUBLIC_SAMPLE_PARTIAL` evidence tier. The run may
support a sample-specific data and predictability diagnostic, but it cannot
support a claim about persistent alpha, statistical significance, execution,
profitability, or capacity.

## Question and hypotheses

The narrow question is whether recently observed aggregate-trade direction and
size contain out-of-time information about the sign of the trade price 20
aggregate trades later.

- **H0:** the transparent model ladder does not improve held-out log loss over a
  historical-prior classifier in this capped sample.
- **H1 (exploratory):** causal signed-volume and trade-imbalance features improve
  held-out log loss relative to that prior.

All tested model rows are published. The final test is not used for feature,
hyperparameter, calibration, or model selection. A favorable point estimate is
not called significant; the block bootstrap is a dependence diagnostic, not a
confirmatory p-value procedure.

## Data and coverage policy

- Instruments are BTCUSDT and ETHUSDT, evaluated separately because the fixed
  5,000-row caps produce different observed clock-time endpoints.
- The exact ingestion manifest and normalized part hashes are inputs to the run.
- Internal aggregate-trade IDs must be unique and step by one within each symbol;
  the availability clock must not reverse. Only after those checks may the
  derived research view assign one continuity epoch per symbol. Raw normalized
  rows remain unchanged.
- Exchange event time is the only historical availability proxy. No local
  receipt-time or colocated-latency claim is allowed.
- Tied exchange timestamps retain aggregate-trade-ID ordering and remain in the
  same time split.

## Causal feature and label contract

At decision trade `i`, features may use trade `i` and earlier trades from the
same verified continuity epoch:

- signed trade volume and absolute volume over 5, 20, and 100 trades;
- signed-volume imbalance over the same windows;
- trade count and event-time intensity;
- one-trade log return;
- realized trade-price volatility over 100 trades.

The target is `1` when the trade price at `i + 20` is above the price at `i`, and
`0` otherwise. The target trade ID and availability timestamp are serialized.
Segment tails are right-censored. Feature-ready rows require the full longest
lookback.

## Evaluation

- Each instrument receives its own expanding time-ordered walk-forward plan.
- Configuration: 1,200 initial decision-time buckets, 400 validation buckets,
  400 final-test buckets, 400-bucket steps, and a 20-bucket embargo.
- Label information ending at or after an evaluation boundary is purged.
- The model ladder is historical prior, unpenalized logistic regression, the
  declared L2 grid, and the declared shallow-tree grid.
- Selection metric is validation log loss. Calibration is trained only from the
  chronological training/calibration region.
- The primary H0/H1 diagnostic is the paired difference in held-out log loss:
  validation-selected model minus historical prior on identical `row_id`
  observations. It uses the same seeded resample draw for both models within
  each fixed, contiguous 40-trade block (twice the label horizon), separately
  by instrument. Five hundred draws, the seed, row count, block count, point
  difference, and percentile interval are serialized. Marginal per-model
  intervals are secondary and are never compared as a substitute for the
  paired loss difference.

## Explicit exclusions

There is no contemporaneous bid/ask, depth, cancellation, queue, or local
receipt-time history in this dataset. Therefore this run does not calculate:

- order-book imbalance, microprice, spread, or liquidity recovery;
- limit-fill probability or queue position;
- market/limit execution, fees-to-alpha conversion, P&L, or capacity.

Those analyses require continuous snapshot-plus-delta L2 epochs collected and
validated separately.

## Promotion criteria

This exploratory run cannot be promoted to `FULL_DATA`. A later confirmatory
study must freeze its protocol before model outcomes are inspected, use multiple
complete nontruncated dates, preserve adjacent untouched dates for final testing,
report per-date and cross-instrument stability, and add continuous L2 evidence
before making book-dependent or execution claims.
