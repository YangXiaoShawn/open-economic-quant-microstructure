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
| M8 Broader empirical study | In progress | M6-M7 | Closed trade-only study plus complete software for the replacement four-date live-L2 capture/analysis campaign | Canonical trade and superseded L2-v1 authorities remain immutable; capture Aug 10--13 under one new campaign authority, publish the Aug 10/11 `LOCKED` or `NOT_CREATED` development authority before Aug 12, and publish a recursively verified result or protocol-declared insufficiency |

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
| Memory/disk pressure | Local run failure | Byte-bounded response streams, hard retained-evidence reservations, disk-backed incremental DQ, batch Parquet, verified spill-capable scans, explicit eager guards; measure peak RSS on any successful full L2 production before promotion |
| Multiple testing | False discoveries | Predeclare core hypotheses, report all tested variants, use adjusted interpretation |
| Missed or invalid prospective L2 session | Outcome-driven replacement or false continuity | One frozen calendar and campaign identity; atomic `INSUFFICIENT_DATA`; no replacement date; require observed-interval and cross-symbol coverage gates |
| Source/runtime/root changes across the four captures | Sessions are not one prospective campaign | Root campaign authority binds config/protocol, outcome-blind nonce, canonical output-root filesystem identity, clean source/import origin, and hashed Python/platform/dependency runtime; reject drift before network access |
| Held-out L2 economic access without a development authority | Selection leakage | Verify Aug 10/11 control authorities and first persist either eight child locks plus one aggregate `LOCKED` authority or a control-only `NOT_CREATED` authority; only the `LOCKED` branch may expose Aug 12/13 frames, and its evaluator has no fit path |

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
M8 remains the active empirical milestone. Its prospective calendar,
hypotheses, selection rule, untouched tests, uncertainty, failure policy, and
lock-before-open boundary are frozen in
`docs/M8_MULTIDATE_TRADE_PROTOCOL.md`. The raw-only acquirer and terminal
producer are implemented and adversarially tested. All eight full daily archives
and their official evidence are bound by one verified raw-only manifest, without
opening a CSV member during acquisition. The subsequent clean-commit economic
run reached a verified `INSUFFICIENT_DATA` terminal at the ETHUSDT training DQ
gate: no selection or held-out member access occurred, so the trade hypothesis
was not evaluated and the date/policy cannot be replaced.

Book claims now depend on the separately frozen live-L2 campaign. The immutable
capture rules are in `docs/M8_L2_PROTOCOL.md`; the exhaustive downstream rules
and their exact source/semantic hashes are in
`docs/M8_L2_ANALYSIS_CONTRACT.md`. Remaining acceptance evidence is: the four
exact session terminals sharing one clean campaign authority; a durable
`LOCKED | NOT_CREATED` development authority before held-out access; unchanged
no-refit evaluation on the `LOCKED` branch;
generated descriptive/predictive/execution artifacts or an honest
`INSUFFICIENT_DATA` terminal; peak-RSS evidence below the local ceiling; and
clean-room reproduction. The producer, recursive verifier, CLI/Make interfaces,
self-contained authority snapshots, and external non-mutating report renderer
are implemented and covered by offline tests. The existing public sample remains
exploratory only.
