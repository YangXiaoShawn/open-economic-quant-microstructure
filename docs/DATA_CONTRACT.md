# Data contract and lineage

## Scope

The normalized layer separates exchange-specific acquisition from research
logic. An adapter may add a new venue, but it must produce the same versioned
event contracts, preserve original bytes, and declare how observation time is
approximated. No normalized table is evidence that an event was available to a
real colocated strategy unless local receipt time was actually captured.

Current schema version: `1.0.0`.

## Clocks and ordering

All timestamps are signed UTC epoch nanoseconds:

- `event_ts_ns`: timestamp supplied by the market-data source;
- `received_ts_ns`: local wall-clock receipt when captured live, otherwise null;
- `available_ts_ns`: earliest time the pipeline permits the row to enter an
  information set;
- `availability_basis`: explicit reason, such as `local_receive_time`,
  `exchange_event_time_proxy`, or `synthetic_receipt`;
- `capture_seq`: local arrival ordering when a collector supplies it;
- `continuity_id`: a feed epoch that cannot be crossed by rolling features,
  labels, open orders, or markouts.

Within a continuous book epoch, sequence IDs—not exchange timestamps—are the
authoritative reconstruction order. Research ordering is stable on availability
time, sequence/capture order, and event identity. Separate trade and book streams
with equal timestamps have no assumed common ordering, so cross-stream joins are
strictly prior unless a future source proves a shared sequence.

## Exact numerical representation

Adapters retain integer `price_ticks`/`quantity_lots` and the corresponding
`tick_size`/`lot_size`. Floating `price` and `quantity` columns are convenience
units and must agree with the exact representation. Binance symbol scales come
from public `exchangeInfo` filters for each download; fixed `1e-8` scale defaults
exist only as explicit low-level fallbacks and are not the configured sample
path.

Derived ratios, log returns, volatility, probabilities, and P&L use `Float64`
with their units named or documented. Execution quantities are rounded down to
the observable lot size; slippage-adjusted prices round adversely to the tick.

## Normalized tables

### Trades

Identity is `(venue, symbol, trade_id)`. Required economic fields include exact
and floating price/quantity, quote quantity, first/last constituent trade IDs,
buyer-maker flag, normalized aggressor side, timestamps, and source artifact ID.
A `buy` aggressor lifts the ask; a `sell` aggressor hits the bid.

### Depth deltas

Each event contains `first_update_id`, `last_update_id`, optional previous update
ID, and bid/ask lists of exact `(price_ticks, quantity_lots)` changes. Quantity
zero is a delete instruction; a negative quantity is invalid.

### Book snapshots

A snapshot carries its request/receipt/availability times, last update ID,
depth limit, exact levels, scale metadata, source artifact ID, and a new
continuity ID. Binance REST snapshots do not provide an exchange event timestamp;
the local receipt time is the anchor availability time.

### Book observations

Reconstruction emits best bid/ask, L1 quantities, cumulative depth at 1/5/10
levels, spread, mid, microprice, queue imbalance, sequence range, validity flag,
and full lineage. A crossed/locked or emptied book terminates the epoch rather
than being silently repaired.

### Sequence gaps

A gap row records expected and observed ranges, the missing inclusive range,
detection time, continuity ID, source artifact, and reason. After a forward gap,
the book is not live again until a new snapshot starts a new epoch.

## Binance acquisition semantics

The configured historical adapter uses only credential-free public market-data
REST endpoints. It starts aggregate-trade pagination with a fixed UTC interval,
then advances by aggregate trade ID so trades sharing a timestamp are not lost.
HTTP 408/418/429/5xx and connection failures use bounded exponential backoff;
`Retry-After` is honored for rate-limit responses. Every exact response body is
content-addressed and manifested before normalization.

The optional live collector uses the market-data-only WebSocket endpoint and
diff-depth `U/u` updates. A correct local book buffers updates, fetches a public
snapshot, discards stale events, requires the first usable event to cover
`lastUpdateId + 1`, and then validates continuity for every event. Reconnection
starts a new continuity epoch. These rules follow Binance's official
[Spot WebSocket stream guide](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~).
Public REST behavior and rate-limit headers are described in the official
[Spot REST documentation](https://developers.binance.com/en/docs/products/spot/rest-api).

Every historical download requires a finite per-symbol row cap. The low-level
downloader cannot be invoked without one, and validation refuses to concatenate
more rows than the configured bound. Large studies must be decomposed into
bounded date/symbol partitions rather than increasing one in-memory request
without limit. A `ConfiguredDataAdapter` protocol is the normalized extension
boundary for later venues or institutional sources; its mode must match the
resolved configuration before dispatch.

## Storage and manifests

Normalized Parquet is content-addressed and partitioned beneath:

```text
<root>/<dataset>/schema-<version>/venue-<venue>/symbol-<symbol>/date-YYYY-MM-DD/
```

Writers consume bounded Arrow record batches; they do not require all history in
memory. Each part has an immutable sidecar with source URI, download/creation
time, requested and observed ranges, row/byte counts, schema version,
transformations, input checksum, and Parquet checksum. A dataset manifest lists
all parts and manifest hashes. Run bundles snapshot these manifest hashes and
protect every included file with `checksums.sha256`.

Each config-driven ingestion also writes a content-named immutable ingestion
manifest linking only the exact raw responses used by that invocation to the
normalized dataset manifests. It records the row cap and each symbol's
`complete_range` state, so capped coverage cannot survive merely as terminal
output or be mistaken for the entire requested interval.

External raw and normalized data are ignored by Git. Small deterministic test
fixtures may be committed only under tests and must state whether they are
synthetic or sampled public observations.

## Quality policy

Validators never sort, deduplicate, clip, interpolate, or rewrite observations.
They emit typed findings for duplicate trades, timestamp/order reversals,
availability/receipt contradictions, sequence gaps or stale updates, crossed
books, invalid price/quantity, scale mismatches, abnormal spread, nonmonotone
depth, long silence, and receipt-clock reversal. A downstream research view may
exclude a row, but it must preserve the normalized source and report the
exclusion count/reason.
