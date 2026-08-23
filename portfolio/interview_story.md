# Interview story: Order Flow to Price Impact

> **Historical calendar note (2026-08-23):** Any Aug 8–11 live-L2 schedule in
> this narrative is superseded by the v2 Aug 10–13 protocol recorded in the
> README and `docs/M8_L2_ANALYSIS_CONTRACT.md`. It is retained as research
> history, not current campaign authority.

## One-sentence version

I designed a reproducibility-first market-microstructure research system that
keeps leakage-safe prediction, execution assumptions, and simulated performance
separate—and refuses to turn synthetic smoke output into an alpha claim.

## The problem

Short-horizon market prediction is unusually easy to overstate. Events are
serially dependent, labels overlap, exchange time is not necessarily receipt
time, and an accurate forecast can still be untradeable after spread, fees,
latency, queue uncertainty, adverse selection, and inventory liquidation. Public
data also varies in depth and quality, while the target machine has 16 GB of RAM
and no paid data feed.

The research question therefore had two parts: under what observable conditions
does order flow predict a future price change, and how much of that relationship
survives a separately specified execution model?

## My approach

I organized the project around immutable event and run contracts rather than a
large notebook. Raw observations remain unchanged. Normalized timestamps and
sequence identifiers make ordering explicit. Features stop at decision event
`t`; labels begin after `t`. Time-ordered folds purge overlapping label horizons
and apply an embargo before held-out evaluation.

The model ladder begins with a historical or majority baseline, then transparent
linear and regularized alternatives, followed by bounded nonlinear models. The
execution layer records fees, two latency components, fill and queue proxies,
partial fills, adverse selection, inventory limits, liquidation, turnover, and
size sensitivity independently of predictive metrics.

Each run freezes its resolved configuration, input checksums, actual UTC period,
random seed, runtime, and Git state. Reports and the read-only Streamlit dashboard
consume that bundle; they cannot retrain a preferred model. Synthetic input
forces a prominent software-test watermark everywhere.

## A difficult design choice

The initial compact data path may contain trades without complete historical
depth. It would have been easy to present a precise-looking queue simulator, but
the data cannot identify true queue priority. I treated queue position and fills
as explicit assumptions, kept the interface replaceable by later Level-2 data,
and made sensitivity—not a single fill estimate—the relevant output.

That decision reduced the apparent sophistication of the first result while
making the inference more honest.

## Verification strategy

The offline smoke path is deterministic and network-free. Focused tests cover
bundle completion, checksum integrity, evidence-tier consistency, synthetic
watermarks, held-out-only comparison tables, deterministic rendering, and clear
dashboard failures for incomplete runs. Research tests separately target event
ordering, leakage, purged folds, fee accounting, latency, partial fills, and
inventory constraints.

## What the empirical work actually produced

The capped public trade sample remained exploratory and explicitly skipped
execution because it had no contemporaneous book. The predeclared full-archive
trade study then produced a useful negative operational result: BTCUSDT training
data passed, while ETHUSDT training data produced 53 long-silence warnings
against a zero-warning gate. The pipeline published a checksummed
`INSUFFICIENT_DATA` terminal before selection and before either held-out date was
opened. I did not relax the rule, substitute a date, or describe the absence of
a model result as a failed market hypothesis.

For the book extension I froze four simultaneous BTCUSDT/ETHUSDT sessions and a
field-complete analysis contract before capture. The implementation binds all
dates to one outcome-blind campaign, clean source/import/runtime identity, and
canonical storage root; limits features and labels to verified observed
intervals; locks the Aug 8/9 development state before Aug 10/11; and provides
no-refit evaluation and market-only scenarios. The final producer recursively
verifies explicit path and digest authorities, snapshots them into an immutable
terminal, and renders reports externally without changing the run. Those are
software controls, not empirical L2 evidence: at the pre-capture source freeze,
no declared L2 result had been promoted. Tracked source remains unchanged during
the four-session campaign; immutable session/final bundles carry live status.

## Evidence boundary

No empirical economic result is claimed merely because a pipeline ran. A
synthetic smoke run proves only that the software contracts and accounting
execute as intended. The bounded public trade run supports exploratory,
interval-specific diagnostics but no execution or broad claim. The canonical
full-archive result supports only data insufficiency because evaluation never
began. Generalization still requires valid adjacent periods, both instruments,
regimes, uncertainty, and transparent failed hypotheses.

## What I would do next

I would operate the completed producer on the already-frozen Aug 8--11 sessions
without changing their calendar or analysis contract, then complete the
clean-room and peak-memory audits on the resulting terminal. Aug 8/9 choices
must be durably locked before Aug 10/11 data is exposed. I would accept a missed
or invalid session as `INSUFFICIENT_DATA`, not search for a replacement. Only
after stable no-refit evidence would I consider point-process models; the
criterion for complexity would be out-of-time economic evidence, not an improved
in-sample score.

## Likely follow-up questions

**Why not random cross-validation?** It mixes regimes and leaks information across
overlapping future horizons. Walk-forward folds better match deployment order.

**Why report calibration?** Thresholded execution decisions consume
probabilities, so ranking alone is insufficient. Miscalibration changes trade
frequency, inventory, and cost exposure.

**What would make you stop?** Leakage, an invalid book reconstruction, failure on
the untouched period, instability without an economic explanation, or economics
that require implausible fills or latency.

**What makes this portfolio-quality?** The result is auditable: assumptions,
failures, provenance, and evidence boundaries are first-class artifacts rather
than caveats added after seeing performance.
