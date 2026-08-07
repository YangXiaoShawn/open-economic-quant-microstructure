# Repository instructions

These instructions apply to the entire repository.

## Mission and safety boundary

Build a reproducible research and simulation system for short-horizon market
microstructure. The repository must never place live orders, authenticate to a
trading account, or imply that a simulated result is executable profit.

## Research integrity

- Never invent observations, performance, statistical significance, or a data
  source. Label synthetic, fixture, smoke-test, partial, and full-data results.
- Keep predictive quality, execution assumptions, and strategy results separate.
- Every generated result must include the configuration hash, input manifest
  hashes, UTC data interval, code version or explicit `UNBORN`, and dirty state.
- Preserve raw observations. Put transformations in normalized or derived data
  and log exclusions; do not silently repair suspect events.
- Treat a timestamp at decision time `t` as unavailable unless its event and
  receipt ordering prove it was observable at `t`. Features use information at
  or before `t`; labels begin strictly after `t`.
- Do not tune against the final test period. Use time-ordered splits, purge
  overlapping label horizons, and embargo adjacent folds when configured.

## Engineering conventions

- Target Python 3.12 and a local Apple Silicon machine with 16 GB RAM.
- Keep core logic in `src/microstructure`; notebooks may call but not duplicate it.
- Prefer Polars lazy/streaming scans and partitioned Parquet. DuckDB may query
  partitions without loading the full data set.
- New data sources implement the adapter interfaces; exchange-specific fields do
  not leak into normalized research modules.
- Store timestamps as UTC epoch nanoseconds and prices/quantities as decimal-safe
  integer ticks/lots where the adapter supplies metadata; floating research
  columns must document their units.
- Randomized procedures require an explicit seed.
- External raw data belongs under ignored `data/raw`; only small, documented test
  fixtures belong in Git.

## Verification

Run focused tests after each meaningful phase and `make check` before handoff.
Tests must cover sequence gaps and book invariants, temporal leakage, purged
splits, deterministic simulations, partial fills, fees, and latency. A test may
not call the public internet; mock adapters at the HTTP boundary.

Primary commands:

```text
make setup
make download-sample
make validate-data
make smoke
make test
make reproduce-sample
make report
make dashboard
```

## Documentation discipline

Record material assumptions and reversals in `docs/DECISION_LOG.md`. Keep
`STATUS.md` honest and current. Update `docs/PROJECT_PLAN.md` acceptance evidence
when a milestone moves state. Do not manually paste model metrics into prose;
reports must read machine-generated run artifacts.
