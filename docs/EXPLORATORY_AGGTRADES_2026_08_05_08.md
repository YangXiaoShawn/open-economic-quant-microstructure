# August 5–8 public aggregate-trade exploratory protocol

## Scope

This retrospective, trade-only study uses the complete official Binance Spot
daily `aggTrades` archives for BTCUSDT and ETHUSDT on 2026-08-05 through
2026-08-08. It is independent of the frozen live-L2 campaign. It contains no
book depth, spread, queue, cancellation, local receipt time, or executable
order evidence.

The date roles are fixed before any archive CSV member is opened:

| UTC date | Role |
| --- | --- |
| 2026-08-05 | Train |
| 2026-08-06 | Validation and model selection |
| 2026-08-07 | Primary test |
| 2026-08-08 | Replication test |

The evidence tier is `PUBLIC_ARCHIVE_EXPLORATORY`. Results cannot be described
as confirmatory, statistically significant, persistent alpha, executable P&L,
or an L2 finding.

## Data and quality

Each ZIP must match the exchange-published `.CHECKSUM`, contain exactly its
declared CSV member, remain inside bounded compressed and expanded byte limits,
and preserve its raw response and source sidecar. Normalization is streamed to
partitioned Parquet. Aggregate-trade IDs must be contiguous and event time must
not reverse within each symbol/date.

Quality warnings are retained and reported but do not stop this exploratory
run; any error, checksum failure, archive-contract failure, ID gap, or temporal
violation stops the run. No observation is repaired or replaced.

## Features, target, and evaluation

Features use only the current and prior trades within one UTC-day continuity
segment: one-trade return; signed volume, total volume, and imbalance over 5,
20, and 100 trades; 50-trade count and intensity; and 100-trade realized
volatility. The target is whether trade price 20 aggregate trades later is
higher. Segment tails are censored.

The candidate ladder is the historical prior, unpenalized logistic regression,
the declared L2-regularized logistic grid, and the declared shallow-tree grid.
Only August 5–6 may select and fit the final selected/prior states. Both symbol
locks and one aggregate lock must be persisted before either August 7 or August
8 CSV member opens. No refit is allowed afterward.

Selected-minus-prior log-loss differences are reported separately for both test
dates and with equal-date weighting. Seeded 40-trade paired block intervals are
descriptive dependence diagnostics only; no p-values or significance claim are
authorized. BTCUSDT and ETHUSDT are never pooled.

## Explicit exclusions

Execution, fills, fees, capacity, queue position, market impact, profitability,
and live-L2 conclusions are `NOT_RUN` or unauthorized. A favorable point
estimate is evidence only about these four retrospective trade archives.
