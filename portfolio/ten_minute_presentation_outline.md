# Ten-minute presentation outline

> **Historical calendar note (2026-08-23):** Any Aug 8–11 live-L2 schedule in
> this outline is superseded by the v2 Aug 10–13 protocol recorded in the
> README and `docs/M8_L2_ANALYSIS_CONTRACT.md`. It is retained as research
> history, not current campaign authority.

## 0:00–0:50 — The question and the trap

- Ask when order-flow imbalance and liquidity predict a strictly future move.
- Then ask whether the effect survives costs and uncertain execution.
- State the central trap: predictability, executability, and profitability are
  different claims.

## 0:50–1:50 — Evidence hierarchy

- `SYNTHETIC_SMOKE`: software behavior only.
- `PUBLIC_SAMPLE_PARTIAL`: fixed interval, limited inference.
- `FULL_DATA`: broader manifested study, still simulated and venue-specific.
- `INSUFFICIENT_DATA`: a declared gate prevented evaluation; no missing estimate
  is replaced with zero or a substitute date.
- Pre-capture source-freeze status: synthetic vertical slice and bounded public trade
  study completed at their stated tiers; full-archive trade M8 stopped on 53
  ETHUSDT training warnings before selection or held-out access; no performance
  claim and no promoted L2 result.

## 1:50–3:05 — Event-driven data architecture

- Public trades and optional Level-2 snapshot/delta adapters.
- UTC nanoseconds plus sequence/event identifiers establish stable order.
- Streaming normalization and partitioned Parquet keep memory bounded.
- Manifests retain source, actual period, schema version, row counts, and hashes.

## 3:05–4:15 — Data quality and leakage controls

- Detect duplicates, ordering and sequence gaps, crossed books, invalid values,
  abnormal spreads, silence, and clock discontinuities without silent repair.
- Features use observations available through decision event `t`; labels start
  after `t`.
- Use expanding walk-forward folds with purge and embargo for overlapping labels.

## 4:15–5:30 — Economic features and model ladder

- Spread, depth, OFI, queue imbalance, microprice, signed volume, intensity,
  volatility, impact, recovery, and regimes when observable.
- Compare historical/majority, linear or logistic, regularized, and tree models.
- Select on validation only; report calibration and uncertainty on held-out data.

## 5:30–6:55 — From prediction to execution

- Keep model metrics and execution artifacts separate.
- Record maker/taker fees, decision and order latency, market versus limit fills,
  queue proxy, partial fills, adverse selection, inventory cap, and liquidation.
- Show gross-to-net and fee/fill/latency/size sensitivity rather than one favored
  P&L number.

## 6:55–8:05 — Reproducibility and reporting

- One frozen run bundle records resolved config, input hashes, actual UTC period,
  seed, runtime, Git commit or `UNBORN`, and dirty state.
- Checksums plus exact `_SUCCESS` or typed `INSUFFICIENT_DATA` markers prevent
  readers from treating partial output as a terminal bundle.
- Technical report, model table, IC memo, and Streamlit app read frozen artifacts;
  they do not retrain or backfill missing metrics.
- The four-date L2 path adds an outcome-blind campaign/runtime/storage identity,
  explicit session/lock/run digests, recursive verification, and external report
  rendering that never mutates the empirical bundle.

## 8:05–9:10 — What would count as evidence?

- Stable held-out effect across BTCUSDT and ETHUSDT, adjacent periods, and regimes.
- Calibration and uncertainty, not ROC-AUC alone.
- Net economics robust to defensible fees, latency, fills, and liquidation.
- Transparent failures and multiplicity-aware interpretation.
- For frozen L2: one clean campaign identity, valid simultaneous observed
  intervals, and an Aug 8/9 `LOCKED | NOT_CREATED` development authority that
  predates all Aug 10/11 access; only `LOCKED` permits economic-frame access.

## 9:10–10:00 — Limitations and next experiment

- Public event time, queue visibility, hidden liquidity, and venue generalization
  remain limitations.
- Next: use the completed frozen L2 producer to capture only the Aug 8--11
  simultaneous sessions, then run all four horizons without refit or a
  replacement date and complete peak-memory/clean-room verification. Market-only
  scenarios are not realized execution; capacity and profitability claims
  remain forbidden.
- Close with the governance boundary: research and simulation only; no live-order
  path.
