# M8 prospective multi-date aggregate-trade protocol

## Freeze and scope

This protocol is outcome-blind: it is frozen before inspecting any declared
date's price, quantity, aggressor side, class balance, feature, label, model, or
economic result. During a parallel feasibility audit before the first protocol
commit, the official Jan 3–5 archive availability, checksums, byte/row counts,
aggregate-trade-ID boundaries, and timestamp boundaries were inspected. A
stop-message race then allowed the same coverage-only checks for Jan 6 after the
outcome-blind freeze; no economic field or result was read. These coverage-only
facts were not supplied when the dates were chosen, and the mechanically
selected calendar below was not changed after they became known. The previously
inspected 2024-01-02 sample is
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

### Protocol-boundary clarification

The acquisition/locking rules below are an operational hardening of the
existing outcome-blind protocol, not an outcome-driven amendment. They change
no date, role, feature, label, candidate, hypothesis, estimand, or interpretation
rule. No declared economic outcome was opened to motivate this clarification:
in particular, no declared price, quantity, buyer-maker value, class balance,
feature, label, fit, score, or test result has been inspected. The historical
coverage-only disclosure above remains the complete exception. An
implementation that cannot enforce the hardened boundary does not execute this
protocol.

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

Acquisition of all eight archives is **raw-only**. It may verify the exact
official `CHECKSUM` response, response lengths and hashes, and bounded ZIP
end-of-central-directory/central-directory metadata for exactly one
expected-name member and its declared sizes. It must not open, extract, or read
the CSV member; stream decompressed member bytes; parse even its header; expose
an economic field; or derive row, ID, timestamp, or class-balance coverage.
Those operations constitute economic-data opening and belong to the staged
normalization boundary below. Public `exchangeInfo` may be parsed only for the
declared symbol's status and exact tick/lot filters, with its exact response and
response sidecar preserved.

Archive transfer and later CSV normalization must both be streaming and
byte-bounded. The byte ceilings in the machine-readable protocol are hard
safety limits, not sampling rules:

- `max_archive_compressed_bytes` bounds each archive ZIP response body while it
  is transferred; an asserted `Content-Length` does not replace streamed byte
  accounting.
- `max_archive_uncompressed_bytes` bounds both the central-directory-declared
  size and the actual expanded bytes of each sole CSV member. The limit is
  rechecked while the member is normalized.
- `max_total_download_bytes` bounds the total immutable raw-evidence bytes
  accepted for this study. The total includes every retained ZIP body, official
  `CHECKSUM` body, raw-response sidecar, `exchangeInfo` metadata body and
  sidecar, and every retained rejected-response prefix and its rejection
  sidecar. A failed attempt or retry does not reset this total. A
  content-addressed file is counted once if it is physically retained once;
  distinct retained copies are counted separately. Derived normalized files,
  quality reports, and aggregate manifest/checksum indexes are not raw-response
  evidence and do not enter this ceiling.

The smaller `CHECKSUM`, metadata, and rejected-prefix responses also have fixed
per-response bounds in the adapter and remain subject to the hard total above.
Before accepting another raw artifact, the acquisition layer must reserve and
account for its bounded body and sidecar; no published raw acquisition manifest
may exceed the total. Crossing any per-response, expanded, or total limit fails
closed before a research result is produced. Every accepted and rejected raw
artifact is immutable, byte-counted, hashed, and enumerated by the raw
acquisition manifest.

## Prospective materialization and lock boundary

The study executes in the following one-way order:

1. Acquire and authenticate all eight raw ZIPs, all eight official `CHECKSUM`
   responses, and the exact symbol-metadata responses. Publish the immutable raw
   acquisition manifest without opening any CSV member.
2. Open, stream-normalize, and quality-check only the train and validation CSV
   members. Publish an immutable development normalized manifest that binds
   every normalized part, sidecar, quality artifact, and its raw source.
3. Select and refit using only those development rows. Persist one lock per
   symbol, then an aggregate lock that commits the exact bytes and SHA-256 of
   both symbol locks. Each symbol lock commits its selected specification,
   development-frame identity, and deterministic final-fit policy; the aggregate
   lock also commits the frozen
   protocol/config, raw acquisition manifest, development normalized manifest,
   and clean real Git revision. Close and `fsync` every lock and digest file and
   `fsync` its containing directory. The aggregate lock is not durable until
   all child locks and their directories are durable.
4. Immediately before the first decompressed CSV byte of every primary or
   replication archive is read, re-read and re-hash the exact durable aggregate
   lock and all identities it commits. Only then may that member be
   stream-normalized and quality-checked. Both untouched dates use the same
   locked fit and transformation state; there is no reselection, refit,
   recalibration, threshold change, feature change, or model update after the
   lock, including between primary and replication.
5. On success, publish a final normalized manifest covering all eight declared
   symbol/dates. Only that final manifest may authorize endpoint evaluation and
   final-bundle publication.

For this boundary, "opened" means the first decompressed byte of a CSV member,
not a later Parquet scan. Inspecting ZIP directory metadata is not opening the
member. Merely writing a lock path is not persistence: exact bytes, hashes,
file descriptors, and containing directories must meet the durability rule
above before a held-out member-open callback can succeed.

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

If primary or replication normalization, completeness validation, or data
quality fails after the aggregate lock is durable, the run stops at the first
deterministic failure and publishes immutable `INSUFFICIENT_DATA` terminal
evidence. That evidence binds the same protocol/config, clean Git revision, raw
acquisition manifest, development normalized manifest, per-symbol locks, and
aggregate lock; it records the failing symbol/date/role, typed reason, retained
partial evidence hashes, and which later members remained unopened. It contains
no endpoint result. Its checksum manifest and terminal marker are written last,
and publication uses a new atomic target followed by directory durability.

The same run identity may only verify and reuse that terminal evidence. It may
not overwrite or delete it, reopen candidate selection, substitute an input,
replace a date, relax quality, or continue with a partial aggregate. A source
fix has a different clean Git revision and therefore a different run identity;
it still cannot erase the original failed evidence. A failure before locking is
also reported as `INSUFFICIENT_DATA`, but it cannot create a test-evaluation
bundle or claim that held-out data was opened under a lock.

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
per-symbol and aggregate analysis locks, raw acquisition manifest, development
and final normalized manifests, exact official `CHECKSUM` responses and
sidecars, exact exchange metadata responses and sidecars, per-date quality
summary, research/evaluation frames, predictions, candidate comparison, paired
hypothesis artifact, feature stability, generated report/memo/table, resolved
configuration, and a clean real Git revision/source-tree identity. The final
manifest binds all of those exact bytes and hashes. The checksum manifest and
`_SUCCESS` are written last only after atomic publication and directory
durability. Corruption, input relocation without matching bytes, a dirty,
unborn, synthetic, or different source identity, or any incomplete date must
fail verification and reuse.
