# Project plan

## Objective

Answer, with defensible event-time evidence, when order-flow imbalance and
liquidity conditions predict short-horizon price movement, and how much apparent
value remains after fees, latency, fill uncertainty, adverse selection, and
inventory constraints. The system is research-only and has no live-order path.

## Delivery strategy

The first deliverable is one narrow, fully reproducible vertical slice. Broader
market coverage and more sophisticated models follow only after its contracts,
timing, and execution accounting are tested.

| Milestone | State | Depends on | Deliverable | Acceptance evidence |
|---|---|---|---|---|
| M0 Repository contract | Complete | None | Instructions, plan, decisions, status, packaging | Required files exist; Python 3.12 environment installed; config/provenance tests, Ruff, and mypy pass |
| M1 Event data foundation | Complete | M0 | Binance trade adapter, optional L2 collector, schemas, manifest/checksum, partitioned Parquet | Mocked retries/metadata/pagination; UTC/schema and content-addressed storage tests pass |
| M2 Book and quality controls | Complete | M1 | Snapshot/delta replay, sequence validation, quality findings | Gap, overlap, crossed-book, duplicate, ordering, invalid value, spread, silence, and clock tests pass |
| M3 Leakage-safe dataset | Complete | M1-M2 | Trade-flow/book features and future labels | Causal lineage, future-label, censoring, and gap-isolation tests pass |
| M4 Model evaluation | Complete | M3 | Baseline, logistic/regularized linear, tree; purged walk-forward evaluation | Prior/unpenalized/L2/tree ladder, calibration, bootstrap, purge, and OOT tests pass |
| M5 Execution research | Complete | M3-M4 | Market/limit fills, latency, partial fills, costs, inventory, liquidation, sensitivity | Deterministic accounting, OOS, volume-conservation, continuity, partial-fill, and liquidation tests pass |
| M6 Reproducible vertical slice | Complete | M1-M5 | One CLI-driven sample run and machine-readable artifacts | Atomic producer and corruption/idempotence tests pass; canonical smoke requires no network |
| M7 Research communication | Complete | M6 | Generated technical report/table, IC memo, limitations, dashboard, portfolio material | Render/dashboard tests read only verified artifacts and visibly label evidence tier |
| M8 Broader empirical study | In progress | M6-M7 | Frozen clean-commit single-date trade-only study complete; next add multi-day BTCUSDT/ETHUSDT and local-L2 stability tests | Explicit input path+SHA and paired held-out comparison are frozen; next require untouched dates/regimes/failed hypotheses |

## First vertical slice

```text
small documented event fixture
  -> normalized partitioned Parquet + manifest
  -> explicit data-quality findings
  -> causal trade-flow / liquidity-state features
  -> strictly future return and direction labels
  -> purged walk-forward baseline, logistic, and tree models
  -> latency- and cost-aware simulation
  -> JSON/CSV/Markdown run artifacts and dashboard inputs
```

The offline fixture is an explicitly synthetic smoke test for software
reproducibility, not empirical market evidence. `make download-sample` adds a
small public-data path without making the test suite depend on network access.

## Dependencies

- Python 3.12; Polars/PyArrow for bounded-memory transformation and Parquet.
- DuckDB for partition inspection and aggregation.
- scikit-learn for transparent, CPU-friendly models and calibration.
- pytest, Ruff, and mypy for verification.
- Streamlit for a local read-only research dashboard.
- Public Binance market-data endpoints only; no API key or account connection.

## Risks and mitigations

| Risk | Consequence | Mitigation / acceptance test |
|---|---|---|
| Snapshot/delta mismatch or sequence gap | Corrupt book state and false imbalance | Halt the affected segment, emit a finding, require a new snapshot; replay gap tests |
| Exchange timestamp is not local observability | Optimistic latency and feature timing | Retain event and receipt times; model configurable decision/order latency |
| Trade-only fixture cannot identify queue dynamics | Fill estimates are assumption-driven | Label fill results as proxy-based; keep book model interface separate |
| Overlapping horizons leak across folds | Inflated validation estimates | Purge by label end time plus optional embargo; boundary tests |
| Small/nonstationary sample | Unstable or meaningless inference | Report uncertainty and evidence tier; defer economic claims until M8 |
| Class imbalance/calibration drift | Misleading probabilities | Persist class rates, Brier/log loss, calibration diagnostics by fold/regime |
| Fee/latency/fill assumptions dominate P&L | Fragile strategy result | Publish assumption grid and sensitivity; never collapse it into model quality |
| Public endpoint/schema changes | Broken ingestion | Version adapter/schema, capture response metadata/checksum, contract tests |
| Memory/disk pressure | Local run failure | Byte-bounded response streams, disk-backed incremental DQ, batch Parquet, verified spill-capable scans, explicit eager guards; disclose metadata growth by page/part |
| Multiple testing | False discoveries | Predeclare core hypotheses, report all tested variants, use adjusted interpretation |

## Acceptance test matrix

1. **Reproduction:** a clean environment follows `README.md`, runs
   `make reproduce-sample`, and receives a run directory with provenance,
   validation, folds, metrics, trades, sensitivities, and reports.
2. **Data:** normalization is deterministic; timestamps are UTC; manifest hashes
   match bytes on disk; Parquet is partitioned by source/symbol/date.
3. **Book:** snapshot-plus-delta replay enforces side sorting, nonnegative depth,
   sequence continuity, and uncrossed top of book.
4. **Timing:** deliberately shifted future values cause leakage tests to fail;
   label intervals never enter feature windows or training folds.
5. **Models:** historical/majority, logistic or regularized linear, and tree models
   run on identical time folds without selecting on the final test set.
6. **Execution:** fees, two latency components, fill probability/queue proxy,
   partial fills, adverse selection, inventory cap, liquidation, turnover, and
   size/capacity sensitivity are represented and unit tested.
7. **Reports:** model table and technical report are rendered from serialized
   results; each carries data interval, config hash, manifests, and Git state.
8. **Claims:** synthetic and smoke outputs contain an explicit non-empirical
   banner; no unverified profitability or significance claim is emitted.

## Definition of done

The objective's ten acceptance criteria have reproducible evidence for M0-M7:

1. `README.md` provides a clean setup and reproduction path.
2. The canonical workflow is deterministic synthetic data and needs no download.
3. Snapshot/delta sequence, stale/overlap/gap, crossed-book, and invariant tests pass.
4. Feature/label lineage, strict-before joins, censoring, and deliberate leakage tests pass.
5. Unpenalized and regularized logistic models plus a shallow tree share frozen OOT folds.
6. Execution records fees, two latency stages, queue/fill uncertainty, partial fills,
   adverse selection, inventory, liquidation, turnover, and size sensitivity.
7. Reports are rendered from serialized, checksum-verified outputs.
8. Run provenance records actual UTC coverage, config/input hashes, seed, runtime,
   Git revision, dirty state, and exact tracked/non-ignored source-tree digest.
9. Synthetic watermarks and claim checks prevent an unverified profitability claim.
10. Limitations, evidence promotion rules, and the absence of evaluable failed
    hypotheses are documented.

Passing M0-M7 completes the portfolio-quality system and first vertical slice.
M8 remains the next empirical milestone: the downloaded public trade sample
validates ingestion only and is intentionally insufficient for economic claims.
