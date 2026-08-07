# Methodology and limitations

## Evidence tiers

- `SYNTHETIC_SMOKE` verifies deterministic software behavior only. Its values are
  neither market observations nor investment evidence.
- `PUBLIC_SAMPLE_PARTIAL` describes a fixed, bounded public-data interval. It may
  support interval-specific research observations but not broad generalization.
- `FULL_DATA` is reserved for a manifested empirical study meeting its declared
  coverage and acceptance tests. It still represents research and simulation,
  not realized live performance.

Evidence tier is derived from data manifests. A synthetic source cannot be
promoted by changing a report label.

## Time and observability

All reported intervals are UTC. Stable event ordering uses the normalized event
timestamp plus sequence and event identifiers; tied timestamps are not reordered
arbitrarily. Exchange event time is not automatically equivalent to local receipt
time. A feature at decision event `t` may contain only values observable at or
before `t`; its label begins strictly after `t`. Decision and order latency are
separate assumptions.

Time-ordered evaluation is necessary but not sufficient. When label intervals
overlap a fold boundary, affected training rows must be purged. Configured embargo
separates adjacent folds. The final test period is not used for model selection,
feature selection, hyperparameter tuning, threshold choice, or probability
calibration.

## Data lineage and quality

External raw data is immutable and excluded from Git. Each download or fixture
requires a source, retrieval time or deterministic generation rule, requested and
observed period, schema version, row count, and checksum. Normalization and later
exclusions create new artifacts rather than rewriting raw observations.

Validation covers duplicates, out-of-order timestamps, missing sequence ranges,
crossed books, nonpositive price or quantity, abnormal spread, long silence, and
clock discontinuities. A warning does not prove an observation is harmless. A
fatal gap can require abandoning an affected reconstruction segment rather than
interpolating it.

Public endpoints can change schema, retention, throttling, or geographic
availability. Download success does not establish completeness. Exchange
maintenance, symbol-rule changes, clock behavior, delistings, and missing markets
can bias a selected sample.

## Market-state and feature measurement

Order-flow imbalance, signed volume, intensity, spread, depth, queue imbalance,
microprice, volatility, price impact, liquidity recovery, and regime features are
conditional on the event types actually observed. Trade signing can be wrong.
Displayed depth can be cancelled before execution. Aggregated or trade-only data
cannot identify hidden orders, matching-engine priority, or individual queue
position. Cancellation intensity is unavailable unless the feed exposes enough
book history to measure it defensibly.

Feature windows create serial dependence, and overlapping future labels reduce
effective sample size. Intraday and volatility regimes may be unbalanced. A
relationship can reflect a common response to news rather than a causal effect of
order flow on price.

## Statistical modeling

Simple historical or majority baselines anchor the model ladder. Linear and
regularized models provide interpretable comparisons; a tree model tests bounded
nonlinearity. More complex time-series or point-process models require a stated
economic reason and evidence that simpler models leave meaningful structure.

ROC-AUC alone can obscure calibration and class imbalance. Classification reports
should include log loss, Brier score, precision-recall diagnostics where useful,
class rates, and calibration. Regression reports require scale-aware errors and a
baseline comparison. Bootstrap confidence intervals must respect temporal
dependence. Repeated instruments, horizons, regimes, features, thresholds, and
models create multiplicity; an unadjusted favorable result is exploratory.

Model importance is not structural causality. Feature rankings can be unstable
under correlation or regime shift. A selected model may decay after the observed
period, and cryptocurrency venue behavior may not transfer to equities, futures,
or fragmented markets.

## Execution and fills

Predictive metrics are not execution results. Simulated economics depend on maker
and taker fees, half-spread and slippage, decision and order latency, order size,
fill probability, queue proxy, partial fills, adverse selection, inventory cap,
liquidation, and capacity assumptions. Each assumption must be serialized and
sensitivity-tested.

A queue proxy is not true priority. A fill inferred from subsequent traded volume
can be optimistic when cancellations, hidden liquidity, competing orders, and
matching rules are unknown. Limit-order simulations can suffer severe adverse
selection; market-order simulations can understate impact. Forced end-of-period
liquidation may dominate a short sample. Capacity extrapolation from public top-of-
book data is especially uncertain.

Annualized return or Sharpe-like statistics are inappropriate for synthetic or
very short runs. Simulated P&L excludes operational failures, exchange outages,
funding and financing where omitted, taxes, custody, counterparty risk, and live
model drift. The project contains no order-entry path and is not a deployment
system.

## Reporting and generalizability

Generated reports read serialized artifacts; they do not recalculate statistics.
Every surface shows evidence tier, observed UTC interval, configuration hash,
input-manifest hashes, Git commit or `UNBORN`, and dirty state. Missing values are
`N/A`, never zero. Checksums demonstrate byte integrity, not correctness of the
economic design.

Results from BTCUSDT and ETHUSDT on one venue cannot be assumed to apply to other
symbols, venues, asset classes, tick sizes, participant mixes, or regulatory
settings. Robustness requires predeclared adjacent periods, cross-instrument and
regime comparisons, alternative defensible execution assumptions, and careful
documentation of results that fail.
