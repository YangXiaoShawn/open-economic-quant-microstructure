# Methodology and limitations

> **Historical calendar note (2026-08-23):** Any Aug 8–11 live-L2 schedule in
> this document is superseded by the v2 Aug 10–13 protocol recorded in the
> README and `docs/M8_L2_ANALYSIS_CONTRACT.md`. It is retained as research
> history, not current campaign authority.

## Evidence tiers

- `SYNTHETIC_SMOKE` verifies deterministic software behavior only. Its values are
  neither market observations nor investment evidence.
- `PUBLIC_SAMPLE_PARTIAL` describes a fixed, bounded public-data interval. It may
  support interval-specific research observations but not broad generalization.
- `FULL_DATA` is reserved for a manifested empirical study meeting its declared
  coverage and acceptance tests. It still represents research and simulation,
  not realized live performance.
- `INSUFFICIENT_DATA` is a terminal evidence status, not a lower-quality set of
  model estimates. It means a predeclared input or quality gate prevented the
  required evaluation; absent predictions and execution fields remain absent
  rather than being imputed, rerun on replacement dates, or reported as zeros.

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

The canonical full-archive trade M8 result illustrates this boundary. BTCUSDT
training normalization completed on 2,071,461 rows with no findings. ETHUSDT
training normalization completed on 987,297 rows with zero errors but 53 long-
silence warnings, violating the frozen zero-warning gate. Selection did not
start, neither held-out date was opened, and execution was not run. This supports
the conclusion that the declared trade study was data-insufficient; it neither
supports nor refutes the economic hypothesis. Relaxing the warning rule or
choosing another date after seeing that terminal would invalidate the protocol.

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

The frozen live-L2 extension narrows execution further. It permits market-order
scenarios only, with threshold, reference notional/depth, lot rounding, L1 fill
cap, inventory, liquidation, fee, and decision/order latency rules fixed before
held-out access. Latencies in this campaign are event counts, not milliseconds.
Recorded L1 limits the scenario fill and any residual is cancelled; no deeper
walk, hidden liquidity, endogenous reaction, or true impact is modeled. The
frozen zero-extra-slippage setting is one transparent scenario, not evidence
that slippage is zero. Capacity, realized execution, limit-fill, queue-priority,
and profitability claims remain forbidden.

## Prospective live-L2 boundary

The Aug 8--11 BTCUSDT/ETHUSDT sessions must share one clean capture runtime
identity and pass the exact simultaneous observed-interval gates. Raw websocket
receipt time is available, but it is internet-path receipt time—not a colocated
clock or matching-engine acknowledgment. A long nominal session cannot conceal
gaps: features and labels are limited to verified observed intervals, and clock
targets are censored if no sufficiently fresh same-interval state exists.

The campaign authority also binds an outcome-blind nonce, the one canonical
output-root path and filesystem identity, the loaded package/module origin, and
a hashed Python/platform/production-dependency fingerprint. These controls make
environment and storage substitution visible, but they do not make public-
internet latency colocated or prove that the exchange feed was complete.

Regime thresholds are fit on Aug 8 only. When both development sessions are
complete, model selection and calibration use Aug 8/9 only and must be committed
in eight symbol-by-endpoint child locks plus one aggregate `LOCKED` authority
before Aug 10/11 frames are exposed. If either development session is
insufficient, a control-only `NOT_CREATED` authority is committed instead and no
economic frame from any session is opened. Held-out evaluation restores only a
`LOCKED` state without refit. Paired moving-block intervals
partially address serial dependence but do not prove independence, solve all
overlapping-horizon dependence, correct every model/horizon/regime comparison,
or authorize a p-value. Directional agreement across two adjacent one-hour
sessions is still a narrow venue- and period-specific result.

At the pre-capture source freeze, these were software and governance controls,
not book evidence: no declared L2 session bundle or downstream L2 metric had
been promoted. This tracked file is intentionally unchanged during the four-day
campaign. The exact field-level authority is
`docs/M8_L2_ANALYSIS_CONTRACT.md`; any eventual numbers must come from a verified
generated bundle rather than this source-controlled limitations file.

The completed final producer takes four explicit session path/manifest/checksum
authorities plus the development-authority path and SHA. A development or
held-out session failure
or no eligible label produces a checksummed `INSUFFICIENT_DATA` terminal with no
promoted evaluation or execution, rather than an opportunistic retry. A complete
run copies exact control authorities into a self-contained snapshot but still
revalidates their external originals. Reports are re-rendered from a checksummed
report-input snapshot into a separate directory; report generation cannot mutate
the immutable empirical bundle. These are integrity and governance guarantees,
not proof that the economic design or market conclusion is correct.

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
