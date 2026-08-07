# Interview story: Order Flow to Price Impact

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

## Evidence boundary

No empirical economic result is claimed until a manifested public research run
meets the promotion protocol. A synthetic smoke run proves only that the software
contracts and accounting execute as intended. The current bounded public trade
ingestion supports a pipeline-quality observation only; generalization requires
adjacent periods, both instruments, regimes, uncertainty, and failed hypotheses.

## What I would do next

I would freeze the core OFI and liquidity hypotheses over multiple dates, capture
continuous BTCUSDT/ETHUSDT depth epochs where book claims are required, inspect
every quality exception, and run the same specification on an adjacent untouched
period. Only then would I consider point-process models. The criterion for
complexity would be stable out-of-time economic evidence, not an improved
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
