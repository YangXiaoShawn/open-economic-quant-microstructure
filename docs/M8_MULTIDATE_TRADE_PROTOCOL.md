# M8 prospective multi-date aggregate-trade protocol

## Freeze and scope

This protocol is outcome-blind: it is frozen before inspecting any declared
date's price, quantity, aggressor side, class balance, feature, label, model, or
economic result. During a parallel feasibility audit before the first protocol
commit, the official Jan 3–5 archive availability, checksums, byte/row counts,
aggregate-trade-ID boundaries, and timestamp boundaries were inspected; Jan 6
was not inspected. Those coverage-only facts were not supplied when the dates
were chosen, and the mechanically selected calendar below was not changed after
they became known. The previously inspected 2024-01-02 sample is
protocol-development evidence only and is excluded from every fit, threshold,
validation score, and test result below.

The study is a prospective stability test of the trade-only hypothesis. It is
not an order-book, execution, profitability, or capacity study. Even a favorable
result cannot answer the repository's book-dependent or trading-cost questions;
those require separately frozen, contemporaneous, continuous L2 evidence.

The machine-readable specification is
`configs/m8_multidate_trade_study.toml`. Its exact bytes, Git commit, and SHA-256
must be copied into the final study bundle. Changing a date, role, feature,
model, endpoint, or interpretation rule creates a new protocol version and may
not overwrite this study.

## Data calendar and completeness

Use the complete Binance Spot daily aggregate-trade archive for both BTCUSDT and
ETHUSDT on each UTC date:

| UTC date | Frozen role | Permitted use |
| --- | --- | --- |
| 2024-01-03 | Train | Feature construction, fitting, train-only thresholds |
| 2024-01-04 | Validation | Candidate selection only |
| 2024-01-05 | Primary test | Open once after the selected specification is locked |
| 2024-01-06 | Replication test | Open with the same locked specification; no refit |

The dates were chosen mechanically as the four adjacent dates immediately after
the inspected 2024-01-02 development sample, not because of observed economic
outcomes. Pre-freeze coverage-only inspection does not authorize any calendar
change. Do not replace a quiet, volatile, missing, inconvenient, or unfavorable
date.

All eight symbol/date archives must be present, checksum-verified, and complete
for `[00:00:00Z, 24:00:00Z)`. A missing archive, truncated response, failed
checksum, row cap, noncontiguous aggregate-trade ID inside a symbol/date, or data
quality error makes the study `INSUFFICIENT_DATA`. Such a failure is reported;
it is not repaired by choosing another date. Raw archives remain immutable and
uncommitted. Normalized data is partitioned by venue, symbol, and UTC date.

Archive download and CSV normalization must be streaming and byte-bounded. The
compressed, expanded, and total byte ceilings in the machine-readable protocol
are hard safety limits, not sampling rules. Crossing one fails closed before a
research result is produced.

## Timing and continuity

Each symbol/date is a separate continuity segment. Features reset at its first
trade, labels are censored at its final trade, and neither lookbacks nor labels
cross midnight or a sequence gap. Exchange event time is only an availability
proxy; aggregate-trade ID breaks tied timestamps. No local receipt-time claim is
permitted.

At decision trade `i`, a feature may use `i` and earlier trades from the same
verified segment. The target is one only when the price at `i + 20` is greater
than the decision price. The target trade ID and information-end timestamp are
serialized. The longest 100-trade lookback and 20-trade label tail determine
feature readiness and right censoring.

## Frozen features, candidates, and selection

The feature set is fixed before acquisition:

- one-trade log return;
- signed quantity, absolute quantity, and signed-volume imbalance over 5, 20,
  and 100 trades;
- trade count and event-time intensity over 50 trades;
- realized trade-price volatility over 100 trades.

Evaluate, separately for each symbol, the historical-prior classifier,
unpenalized logistic regression, L2 logistic regression with
`C in {0.1, 1, 10}`, and shallow decision trees with depth in `{2, 4, 6}` and
minimum leaf size 40. Median imputation, standardization where applicable, and
sigmoid calibration are learned only from chronologically earlier rows.

The 2024-01-03 fit predicts 2024-01-04. Mean validation log loss selects one
candidate per symbol; stable candidate order is the tie breaker. After selection,
the selected candidate is refit once on 2024-01-03 plus 2024-01-04 using the
same chronological calibration rule. The selected specification and its hash
are written to an analysis lock before either test date is evaluated. The same
locked fit predicts both test dates; there is no update between primary and
replication tests. All candidate validation rows and all locked-model test rows
are published.

## Hypotheses and estimands

For each symbol:

- **H0:** the validation-selected model does not reduce held-out log loss versus
  the historical-prior classifier on the untouched dates.
- **H1:** the validation-selected model reduces held-out log loss versus the
  prior on both the primary and replication dates.

The primary estimands are selected-model minus prior log loss for each
symbol/date and the equal-date-weighted mean across the two test dates for each
symbol. Negative values favor the selected model. A result is
`directionally_replicated` only when both date-level point differences are
negative. Otherwise it is `mixed`, `failed`, or `insufficient_data` according to
the serialized observations. This status is descriptive and is not a
significance decision.

Uncertainty uses paired, contiguous 40-trade blocks, resetting at every UTC date.
The same resampled blocks are used for selected and prior predictions. Report
2,000 seeded percentile draws for each date and an equal-date-weighted paired
draw for each symbol. Every interval must contain observations from the two
nonoverlapping test dates before the aggregate is emitted.

No p-values are computed and no H0 rejection or statistical-significance claim
is authorized. The two symbol hypotheses are not pooled. Candidate, feature,
date, and regime diagnostics beyond the endpoints above are secondary and are
reported without selective omission.

## Stability and failed-result reporting

The run must publish, by symbol and date:

- row counts, UTC bounds, class balance, and all exclusions/censoring;
- prior and selected-model proper scores and paired loss differences;
- feature distribution stability using bins fitted on the training date only;
- validation, primary-test, and replication-test direction consistency;
- every declared model candidate's validation score;
- explicit `supported`, `mixed`, `failed`, or `insufficient_data` status.

No date or instrument may disappear because its result is unfavorable. Any
aggregate row must be accompanied by its component date rows.

## Explicit exclusions and promotion boundary

Aggregate trades contain no contemporaneous bid/ask, depth, cancellation,
queue, or local receipt clock. Therefore this study must serialize execution,
fills, P&L, fees-to-alpha conversion, and capacity as `NOT_RUN`. It cannot
promote a book, fill, latency, execution, or profitability claim.

`FULL_DATA` means only that every byte of every predeclared daily trade archive
was verified and included for this narrowly defined trade-only study. It does
not mean full market observability, external validity, or deployable evidence.
The overall M8 milestone remains incomplete until a separately frozen protocol
has at least two nonoverlapping, continuous, contemporaneous L2 capture periods
per reported interval and connects their gap-safe book states to causal research
and execution artifacts.

## Required immutable outputs

The atomic final bundle must contain the frozen protocol and machine spec,
analysis lock, acquisition and normalized manifests, per-date quality summary,
research/evaluation frames, predictions, candidate comparison, paired hypothesis
artifact, feature stability, generated report/memo/table, resolved configuration,
Git/source-tree provenance, checksum manifest, and `_SUCCESS` written last.
Corruption, input relocation without matching bytes, a dirty/different source
identity, or any incomplete date must fail verification and reuse.
