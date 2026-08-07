# Research protocol

## Question and estimands

The primary question is not whether a classifier can fit event data. It is
whether an order-flow or liquidity signal predicts a strictly future price state
out of time, whether that relationship is stable across symbols and regimes, and
whether its scale exceeds execution frictions under declared—not inferred—fill
assumptions.

The primary predictive estimands are:

1. the change in future mid-price direction probability associated with a
   one-standard-deviation change in causal order-flow imbalance;
2. the conditional future log-mid return across predeclared event horizons;
3. the decay of that relationship as the horizon increases;
4. the interaction of order flow with contemporaneous spread, depth, volatility,
   and trade intensity.

Economic evaluation is a separate conditional exercise: given frozen OOS
predictions and a declared execution model, measure gross edge, explicit fees,
arrival cost, post-fill adverse selection, fill fraction, turnover, inventory,
and marked or liquidated net P&L. It is not an estimate of deployable capacity.

## Predeclared core hypotheses

- **H1 — Order-flow direction:** positive trailing signed flow and L1 OFI are
  associated with positive strictly future mid returns, and vice versa.
- **H2 — Decay:** predictive association is strongest at short horizons and
  decays rather than monotonically increasing with horizon.
- **H3 — Liquidity interaction:** a given flow shock has larger price impact when
  displayed depth is low or relative spread/volatility is high.
- **H4 — Recovery:** after a large signed trade or spread/depth shock, liquidity
  recovery time varies with the pre-shock volatility/liquidity regime.
- **H5 — Stability:** effect direction is not assumed transferable; BTCUSDT and
  ETHUSDT are reported separately before any pooled conclusion.

Rejecting or failing to support a hypothesis is a valid result and belongs in
the generated report. Synthetic smoke data cannot support or reject any market
hypothesis; it can only verify that the estimands and controls are computed.

## Descriptive analysis contract

The run producer persists intraday liquidity, OFI/return association, signal
decay and half-life, event-time signed impact, large-trade impact, liquidity
recovery, market regimes, model diagnostics by regime, cross-instrument effect
stability, and train-versus-test feature stability. Large-trade, shock, recovery,
and regime thresholds are fitted on the final training rows only and serialized.
These outputs carry `descriptive_only=true`; synthetic values cannot support or
reject the hypotheses above.

## Information set

A decision sample is keyed by `(available_ts_ns, sequence, event_id)` within a
continuous segment. A source timestamp is only an availability proxy unless a
local receipt timestamp exists. Features use observations whose availability key
is no later than the decision key. Labels start after the decision and end at
their persisted information-end key. No feature, label, or fold crosses a known
feed gap.

Tied observations from different streams have no assumed causal ordering unless
the source supplies one. As-of joins therefore use the last strictly observable
state. Missing future targets are right-censored; they are never filled with the
last observation.

## Evaluation protocol

- Splits follow global UTC/event order with expanding training windows.
- Training observations whose label interval overlaps the next evaluation
  boundary are purged; a configured embargo adds separation.
- Imputation, scaling, regime thresholds, feature selection, calibration, and
  hyperparameter choice use training/calibration/validation data only.
- The final held-out period is evaluated once after model choice is frozen.
- Baseline, unpenalized/regularized linear, and shallow tree models share the same
  features, folds, and final test observations.
- Classification reporting includes class support, log loss, Brier score,
  ROC-AUC when defined, accuracy/balanced accuracy, and calibration diagnostics.
- Return models, when supported, report MAE and rank correlation without treating
  statistical fit as tradable value.
- Dependent uncertainty uses contiguous event or UTC-day blocks. Fewer than two
  independent blocks produces an `insufficient_blocks` result rather than a
  confidence interval.

## Multiple testing and model selection

The core family is the two symbols × predeclared horizons × core OFI association.
Exploratory regimes, feature variants, model variants, and sensitivity grids are
reported as exploratory and are not promoted to confirmatory evidence. Where
p-values are later added for a full-data study, false-discovery-rate adjustment
is applied within the declared family and both raw and adjusted values are kept.

Models are selected on aggregate validation log loss (classification) or MAE
(regression). Prefer the simpler model when performance is statistically
indistinguishable under the predeclared rule. The final test cannot change the
model, signal threshold, calibration, size, latency, or fee assumption.

The frozen trade-only public protocol is narrower and is specified separately
in `docs/PUBLIC_TRADE_PROTOCOL.md`. It evaluates each symbol independently and
uses the same held-out rows and identical contiguous bootstrap draws for the
validation-selected model-minus-historical-prior log-loss difference. Marginal
model intervals are not substituted for that paired estimand. Because the input
has no contemporaneous book, execution, fill, P&L, and capacity artifacts are
serialized as `NOT_RUN`, not zero.

## Execution scenarios

Base market orders use the book observable at order-arrival time after separate
decision and order latencies. Available L1 depth caps fills; missing deeper depth
is not extrapolated. Passive orders join behind a declared queue proxy, fill only
against eligible opposing printed flow, can fill partially, and remain exposed
during cancel latency. Maker/taker fees, inventory caps, and end liquidation are
explicit.

Latency, queue position, cancellation ordering, and endogenous impact are not
identified by archived exchange data. They are scenario inputs and must be shown
as sensitivity axes. A strategy that works only under the most favorable fill or
latency scenario fails the economic robustness test.

## Promotion criteria for empirical claims

No market conclusion may appear in “main findings” until a manifested public or
institutional data run has:

- at least two non-overlapping UTC dates per reported confidence interval;
- a frozen final test period not used for selection;
- acceptable sequence/data-quality coverage for every book-based feature;
- results for both default instruments or an explicit single-instrument scope;
- cost/latency/fill sensitivity and failed-hypothesis disclosure;
- a clean evidence label, config hash, input checksums, and Git state.
