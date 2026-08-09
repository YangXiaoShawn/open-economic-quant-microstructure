# M8 live-L2 analysis contract

## Authority and status

This document is a human-readable, field-complete rendering of
`configs/m8_l2_analysis.toml`. It does not amend or supersede those bytes. The
strict loader rejects a changed, missing, or additional field.

| Authority | SHA-256 |
| --- | --- |
| Analysis TOML source bytes | `0d786d5f4109bb5bf773a6197df3fa861c9b7eb61c16c957bd49fb56147fd7d8` |
| Analysis TOML canonical semantics | `17c91f64765f35195ab03a4caac93d8ff9c5f009c16e785fd84ebd9569d6f84b` |
| Bound capture-config source bytes | `b1bf3b4e2820e24e4555bfeb9cb0957f9a0bcdef62039f7d92360e0a97d0dd39` |
| Bound capture-protocol source bytes | `4c77a2099a4cabd049d10e0f8264d3b4c66704d8e87cbaf0c817fd085f4bbd83` |

The calendar, capture gates, model candidate grid, fee, and decision/order
latency event-count grid remain authoritative in the bound capture config and
`docs/M8_L2_PROTOCOL.md`; they are not silently restated as analysis-config
fields here. At freeze time, this contract contained no observed session data or
model outcome. It therefore authorizes no empirical conclusion by itself.

## Operational enforcement

The implemented producer consumes four explicit session bundle paths, manifest
SHA-256 values, and checksum-file SHA-256 values. Development locking additionally
binds its aggregate SHA, and final verification/reporting additionally binds the
run manifest and checksum-file SHA. The campaign authority ties every date to one
outcome-blind nonce, canonical filesystem root identity, clean source/import
origin, and hashed Python/platform/production-dependency fingerprint. A final
bundle snapshots these authorities for self-contained audit while recursively
revalidating the external originals. Report re-rendering occurs outside the
immutable run and leaves it unchanged. These enforcement facts implement the
frozen contract; they are not additional TOML fields or empirical outcomes.

## Study fields

| TOML path | Frozen value |
| --- | --- |
| `study.name` | `binance-m8-live-l2-analysis` |
| `study.protocol_version` | `1.0.0` |
| `study.seed` | `20260807` |
| `study.source` | `verified_m8_l2_session_bundles` |
| `study.capture_config_source_sha256` | `491b14727a3e8bad907d1ad64072f6ebc14e407f98a5c31fca7b0a9e6801e758` |
| `study.capture_protocol_sha256` | `fe6d4aea5af3e9c529486b7e108afefeba623bf5ad3cc0743c0426a5e62e1fa7` |
| `study.symbols` | `BTCUSDT`, `ETHUSDT` |
| `study.training_role` | `train` |
| `study.selection_role` | `validation` |
| `study.primary_endpoint_role` | `primary_test` |
| `study.replication_endpoint_role` | `replication_test` |

Only checksum-verified session bundles with the exact bound capture authorities
are eligible. All per-symbol decisions operate inside verified observed
continuity intervals; reconnects, gaps, excluded state, and invalid intervals
are never bridged.

## Feature and label fields

| TOML path | Frozen value |
| --- | --- |
| `features.decision_scope` | `per_symbol_verified_observed_intervals` |
| `features.flat_direction_policy` | `flat_is_non_up` |
| `features.rolling_windows` | `20`, `100` |
| `features.volatility_window` | `100` |
| `features.clock_max_state_age_ms` | `500` |
| `features.clock_target_policy` | `exact_target_locf_same_valid_observed_interval` |
| `features.clock_label_information_end` | `exact_target` |
| `features.clock_record_target_sequence` | `true` |
| `features.clock_censor_if_no_eligible_state` | `true` |

The exact ordered `features.model_feature_columns` value is:

1. `spread_bps`
2. `depth_total_l1`
3. `depth_total_l5`
4. `depth_total_l10`
5. `queue_imbalance_l1`
6. `queue_imbalance_l5`
7. `queue_imbalance_l10`
8. `microprice_deviation_bps`
9. `ofi_l1`
10. `ofi_w20`
11. `ofi_w100`
12. `cancellation_intensity_w20`
13. `cancellation_intensity_w100`
14. `realized_volatility_w20`
15. `realized_volatility_w100`
16. `volatility_regime_low`
17. `volatility_regime_high`
18. `liquidity_regime_liquid`
19. `liquidity_regime_stressed`

For a clock endpoint, the target is the exact horizon time. The eligible future
book is the most recent state at or before that target, must be no more than 500
ms old, and must belong to the same valid observed interval. Otherwise the label
is censored. Its information end remains the exact target, and the chosen target
sequence is recorded. A zero future return is assigned to the non-up class.

## Endpoint fields

| `endpoints` row | `domain` | `horizon_value` | `unit` | `paired_block_width` | `paired_block_unit` | `nominal_event_block_width` |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `event_20` | `event` | 20 | `events` | 40 | `events` | 40 |
| `event_100` | `event` | 100 | `events` | 200 | `events` | 200 |
| `clock_1000ms` | `clock` | 1000 | `milliseconds` | 2000 | `milliseconds` | 20 |
| `clock_5000ms` | `clock` | 5000 | `milliseconds` | 10000 | `milliseconds` | 100 |

These four rows are the complete `[[endpoints]]` array; each table column maps
one-for-one to a TOML field. Dependency blocks cannot cross a verified observed
interval.

## Regime, calibration, bootstrap, and signed-impact fields

| TOML path | Frozen value |
| --- | --- |
| `regimes.fit_role` | `train` |
| `regimes.feature` | `realized_volatility_w100` |
| `regimes.quantile_numerators` | `1`, `2` |
| `regimes.quantile_denominator` | `3` |
| `calibration.bins` | `10` |
| `bootstrap.method` | `paired_moving_block` |
| `bootstrap.samples` | `2000` |
| `signed_impact.metric` | `ofi_signed_future_mid_markout` |
| `signed_impact.side_rule` | `sign_of_horizon_matched_ofi` |
| `signed_impact.price_rule` | `ofi_sign_times_future_log_mid_return_bps` |

Regime cutoffs are the training-role one-third and two-thirds quantiles of
`realized_volatility_w100`. They are fit once and then applied without update.
Here “horizon-matched OFI” means decision-time observable rolling OFI matched to
the endpoint window, never OFI measured during or after the label interval:
`event_20` and `clock_1000ms` use `ofi_w20`; `event_100` and `clock_5000ms` use
`ofi_w100`.
Those columns come from the causal decision feature frame, and their maximum
source timestamp is validated not to exceed the decision timestamp. The signed-
impact estimand multiplies that decision-time OFI sign by the strictly future
log-mid return in basis points. The 2,000-draw paired moving-block design uses
each endpoint's frozen block width; it does not authorize a p-value or a cross-
symbol pooled significance claim.

## Execution fields

| TOML path | Frozen value |
| --- | --- |
| `execution.market_orders_only` | `true` |
| `execution.probability_threshold` | `0.55` |
| `execution.symmetric_probability_thresholds` | `true` |
| `execution.order_notional_usd` | `100.0` |
| `execution.max_l1_participation` | `0.10` |
| `execution.inventory_order_multiples` | `10` |
| `execution.reference_price_fit_role` | `train` |
| `execution.reference_depth_fit_role` | `train` |
| `execution.reference_price_statistic` | `train_median_mid_price` |
| `execution.reference_depth_statistic` | `train_q05_min_bid_ask_l1_depth` |
| `execution.reference_quantity_policy` | `min_100usd_and_10pct_train_q05_l1_depth_rounded_down_to_lot` |
| `execution.l1_fill_policy` | `fill_up_to_recorded_l1_depth_cancel_remainder` |
| `execution.scenario_reset_policy` | `per_symbol_session_endpoint_latency_pair` |
| `execution.extra_slippage_bps` | `0.0` |
| `execution.liquidate_at_end` | `true` |

The symmetric threshold means long above `0.55`, short below `0.45`, and no
order otherwise. Reference price and depth are fitted on training data only. The
reference quantity is the smaller of USD 100 at the training median mid and 10%
of the training fifth percentile of minimum bid/ask L1 depth, rounded down to the
exchange lot. Recorded L1 depth caps each fill; any residual is cancelled.
Inventory is capped at ten reference-order multiples. State resets for every
symbol, session, endpoint, and latency pair, and end inventory is liquidated.
Latency values and taker fee are consumed from the separately bound capture
config; latency is measured in events, not milliseconds. Zero extra slippage is
a frozen scenario assumption, not a claim that endogenous impact is zero.

## Claim fields

| TOML path | Frozen value |
| --- | --- |
| `claims.allow_capacity_claim` | `false` |
| `claims.allow_realized_execution_claim` | `false` |
| `claims.allow_profitability_claim` | `false` |

Accordingly, eventual reports may describe predictive diagnostics and
serialized market-order scenarios only. They cannot call those scenarios
realized fills, deployable capacity, or evidence of profitability. Any missing
or invalid declared session may instead produce `INSUFFICIENT_DATA`; that
terminal is not replaced and does not count as a test of the economic
hypothesis unless the required evaluation actually occurred.
