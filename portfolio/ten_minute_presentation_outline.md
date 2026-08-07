# Ten-minute presentation outline

## 0:00–0:50 — The question and the trap

- Ask when order-flow imbalance and liquidity predict a strictly future move.
- Then ask whether the effect survives costs and uncertain execution.
- State the central trap: predictability, executability, and profitability are
  different claims.

## 0:50–1:50 — Evidence hierarchy

- `SYNTHETIC_SMOKE`: software behavior only.
- `PUBLIC_SAMPLE_PARTIAL`: fixed interval, limited inference.
- `FULL_DATA`: broader manifested study, still simulated and venue-specific.
- Current canonical status: synthetic vertical slice completed for software
  validation; bounded public trade ingestion completed; no performance claim.

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
- Checksums plus `_SUCCESS` prevent dashboards from reading partial output.
- Technical report, model table, IC memo, and Streamlit app read frozen artifacts;
  they do not retrain or backfill missing metrics.

## 8:05–9:10 — What would count as evidence?

- Stable held-out effect across BTCUSDT and ETHUSDT, adjacent periods, and regimes.
- Calibration and uncertainty, not ROC-AUC alone.
- Net economics robust to defensible fees, latency, fills, and liquidation.
- Transparent failures and multiplicity-aware interpretation.

## 9:10–10:00 — Limitations and next experiment

- Public event time, queue visibility, hidden liquidity, and venue generalization
  remain limitations.
- Next: freeze a multi-date public protocol, capture continuous local book epochs,
  and test predeclared adjacent periods without changing the specification.
- Close with the governance boundary: research and simulation only; no live-order
  path.
