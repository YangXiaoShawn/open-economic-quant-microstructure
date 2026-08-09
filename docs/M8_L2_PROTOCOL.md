# M8 prospective live-L2 protocol — replacement campaign v2

## Scope and frozen calendar

This replacement protocol was fixed before any v2 live session occurred. Both symbols
must be captured concurrently from Binance Spot diff-depth at the requested
100 ms stream interval:

| UTC session | Role |
| --- | --- |
| 2026-08-10 14:00–15:00 | Train |
| 2026-08-11 14:00–15:00 | Validation |
| 2026-08-12 14:00–15:00 | Primary test |
| 2026-08-13 14:00–15:00 | Replication test |

The dates are the four consecutive UTC dates beginning after the v2 reset on
2026-08-09, and the common hour was chosen before observing those sessions. Missing,
quiet, volatile, disconnected, corrupt, or unfavorable sessions are never
replaced. They produce an explicit `INSUFFICIENT_DATA` status. The superseded
Aug 8–11 campaign and its evidence remain immutable and are not inputs to this
study. The exact config
bytes in `configs/m8_l2_capture_study.toml`, this protocol, their hashes, and the
freeze Git commit must enter every resulting bundle.

This is a research-only public market-data capture. It authenticates to no
account and has no order-entry path.

## Capture and continuity acceptance

Each symbol receives its own snapshot and diff-depth stream, started as close to
the common boundary as the public network allows. Raw websocket bytes must be
journaled before UTF-8/JSON parsing. Every continuity epoch begins from a fresh
REST snapshot and may use only buffered deltas satisfying Binance `U/u`
bridging. Gaps, malformed ranges, crossed books, or scale mismatches terminate
that epoch; they are recorded rather than repaired.

A session is usable only when both symbols satisfy all of the following:

- capture-specific completion status is `COMPLETE` and reconstruction ends
  `LIVE`;
- requested duration is 3,600 seconds and overlapping receipt-time coverage is
  at least 3,300 seconds;
- at least one valid continuity epoch spans 1,800 seconds;
- sequence gaps and quality errors/warnings are all zero;
- no raw frame exceeds 1 MiB and no Arrow batch estimate exceeds 16 MiB;
- message, normalized-row, reconstructed-row, and excluded-row reconciliation
  is exact;
- every epoch has a raw snapshot anchor and every raw/normalized/quality file is
  checksum-manifested.

The 60,000-message ceiling is a safety bound, not a stopping target. A
duration-aware graceful stop must publish completion evidence; SIGINT/cancellation
or hitting the message ceiling early is not a complete one-hour session.

## Causal dataset

Continuity is never bridged across reconnects or gaps. At each decision book
event, features may use only locally received snapshot/delta state available at
or before that event. Frozen features are absolute/relative spread, L1/L5/L10
depth, OFI, L1/L5/L10 queue imbalance, microprice displacement, observable
zero-quantity cancellation intensity, short realized volatility, and regimes
whose thresholds are fit on the training session only.

Labels begin strictly after the decision event and are censored at continuity
boundaries. Report future mid-price direction/return and signed price impact at
20/100-event and 1/5-second horizons. Limit-fill and adverse-selection labels are
not authorized without contemporaneous trade prints that prove depletion; this
book-only protocol does not infer them from cancellation alone.

## Evaluation and hypotheses

Per symbol, train on Aug 10, select the fixed prior/logistic/L2/tree ladder by Aug
11 log loss, lock the specification, then evaluate it without refit on Aug 12 and
Aug 13. Probability calibration and regime thresholds use development data only.
All declared horizons and model comparison rows are published; no final period
is used for selection.

The primary book hypothesis is that observable OFI, imbalance, microprice, and
liquidity state reduce future-mid-direction log loss relative to a historical
prior on both untouched sessions. Report paired dependency-block differences by
symbol, session, horizon, and train-defined regime, plus equal-session-weighted
stability. A result is directionally replicated only if both untouched sessions
favor the selected model. No p-value, H0 rejection, cross-symbol pooled alpha,
or significance claim is authorized.

## Execution boundary

Only scenario-based market-order evaluation is permitted: recorded best quotes,
frozen taker fee, decision/order latency grids, inventory bounds, and end-of-run
liquidation. It may show whether a predictive markout survives those serialized
assumptions, but it is not realized execution. Limit fills, true queue priority,
hidden liquidity, endogenous impact, and deployable capacity remain unsupported.
Every report must keep predictive scores, simulated execution assumptions, and
scenario P&L separate and must state that profitability is not established.

## Immutable outputs

The study requires capture-specific raw journals and snapshots, capture and
normalized manifests, DQ/exclusion summaries, causal frames, analysis lock,
frozen predictions/comparisons, regime/stability diagnostics, scenario execution
ledgers, generated reports, config/protocol/input/Git provenance, checksums, and
`_SUCCESS` written last. A missing session or failed gate may produce a failure
bundle, but never a promoted result bundle.
