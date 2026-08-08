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
HTTP 408/418/429/5xx, connection failures, and recoverable interruptions while
streaming an HTTP 200 body use bounded exponential backoff; `Retry-After` is
honored for rate-limit responses. Response bodies have a byte ceiling. Every
accepted exact body is content-addressed and manifested before normalization;
an interrupted or oversized response preserves a bounded rejected prefix and an
explicit rejection sidecar before retry/failure.

The optional live collector uses the market-data-only WebSocket endpoint and
diff-depth `U/u` updates. A correct local book buffers updates, fetches a public
snapshot, discards stale events, requires the first usable event to cover
`lastUpdateId + 1`, and then validates continuity for every event. Reconnection
starts a new continuity epoch. These rules follow Binance's official
[Spot WebSocket stream guide](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-streams/~).
Public REST behavior and rate-limit headers are described in the official
[Spot REST documentation](https://developers.binance.com/en/docs/products/spot/rest-api).

Live capture persists each WebSocket frame before UTF-8/JSON parsing in a typed
base64 journal containing exact bytes, local receipt time, capture sequence and
continuity epoch. Each reconnect snapshot's raw body and sidecar are journaled as
an anchor. A frame above 1 MiB fails only after its evidence is preserved;
normalized Arrow spools flush before an estimated 16 MiB batch ceiling. A
capture-ID-scoped normalized root produces one streaming Parquet descriptor per
nonempty table, and a capture-ID completion summary is atomically published
last. The fixed summary name is only a latest-pointer, never the sole completion
record. Historical REST output retains the date-partition layout described
below.

Every historical download requires a finite per-symbol event cap, but that cap
is not treated as a RAM budget. The default path lazily yields at most one REST
page per Arrow batch, updates exact disk-backed quality state, and feeds the
Parquet writer once. Eager compatibility materialization has a separate hard
row guard. A `ConfiguredDataAdapter` protocol/registry is the normalized
extension boundary for later venues or institutional sources; its mode must
match the resolved configuration before dispatch.

### Daily-archive acquisition boundary

A daily-archive acquisition object authenticates raw transport evidence; it is
not a normalized-data object. Acquisition may hash exact bodies, authenticate
the official `CHECKSUM`, and inspect bounded ZIP
end-of-central-directory/central-directory metadata for member name, count, and
declared compressed/expanded sizes. It must not open or extract the CSV member,
read decompressed member bytes, parse a header or row, expose economic fields,
or compute row/ID/timestamp coverage. The first decompressed member byte is the
economic-data-open boundary, and the archive API must expose a fail-closed guard
immediately before that byte can be read.

For the prospective M8 study, all declared ZIP, `CHECKSUM`, and exchange-metadata
responses are acquired and placed in an immutable raw acquisition manifest
before any member is opened. Only train and validation members are then
stream-normalized and quality-checked. Their immutable development normalized
manifest is an input to selection. One per-symbol analysis lock and an aggregate
lock committing both child-lock hashes are closed and `fsync`ed, along with
their digest files and containing directories. Each child lock binds the
selected specification, development-frame identity, feature order, imputer and
scaler parameters, selected-estimator state, independent historical-prior state,
calibration state, fit cutoffs, and the canonical numeric fitted-state SHA-256.
Those states are fit on development data and durably closed before held-out
member access. The aggregate lock binds the
protocol/config, raw acquisition manifest, development normalized manifest, and
clean real Git revision. Every primary/replication member-open guard must
re-read and re-hash that exact durable lock before allowing decompression. Test
normalization may construct held-out features, but prediction restores the
locked numeric state; evaluation cannot reselect, fit, refit, recalibrate, or
update it.

Declared-object 404/410 responses, invalid exchange-metadata semantics, official
`CHECKSUM` or ZIP structural violations, frozen per-response size violations,
and retained-evidence budget exhaustion are typed deterministic insufficiency
and publish an immutable raw-only `INSUFFICIENT_DATA` authority. Retry exhaustion,
connection interruption, permission/local-I/O errors, collisions, and program
faults remain nonterminal system failures so they cannot masquerade as a market
or data result.

The M8 byte limits have distinct meanings. The compressed ceiling applies to
each archive response while streaming. The expanded ceiling applies to both the
declared and actually streamed bytes of each CSV member. The total-download
ceiling is the hard sum of immutable raw-response evidence retained for the
study: ZIP bodies, official `CHECKSUM` bodies, every response sidecar,
`exchangeInfo` bodies and sidecars, and all retained rejected-response prefixes
and rejection sidecars. Retried attempts still consume the total. A physically
retained content-addressed file is counted once even if referenced more than
once; separate retained copies are counted separately. Normalized Parquet,
quality outputs, and derived aggregate indexes are outside this raw-evidence
sum. Type-specific bounded `CHECKSUM`, metadata, and rejection responses remain
subject to the same total. Accounting is enforced while bytes are retained, not
after all downloads finish, and the raw acquisition manifest records every
accepted/rejected artifact and the final exact total.

If held-out normalization, continuity/completeness validation, or quality fails
after lock durability, the run publishes an atomic, checksum-protected,
immutable `INSUFFICIENT_DATA` terminal artifact under the same run identity. It
binds the raw and development manifests, per-symbol and aggregate locks, clean
Git identity, typed failure, partial-evidence hashes, and unopened remainder;
it contains no endpoint result. That identity may subsequently verify/reuse the
failure but may not overwrite it, return to selection, substitute inputs, or
publish a partial successful bundle. A source change creates a new clean Git
identity rather than rewriting prior evidence.

## Storage and manifests

Normalized Parquet is content-addressed and partitioned beneath:

```text
<root>/<dataset>/schema-<version>/venue-<venue>/symbol-<symbol>/date-YYYY-MM-DD/
```

Writers reject oversized input batches before row conversion and consume
bounded Arrow record batches; they do not require all event rows in memory.
Each part has an immutable sidecar with source URI, download/creation
time, requested and observed ranges, row/byte counts, schema version,
transformations, input checksum, write ordinal, and Parquet checksum. A dataset
manifest lists all parts, sidecars, hashes, row counts, observed bounds and write
order. Validation checks those bytes before scanning rows. Run bundles snapshot
these manifest hashes and protect every included file with `checksums.sha256`.

Each config-driven ingestion also writes a content-named immutable ingestion
manifest linking only the exact raw responses used by that invocation to the
normalized dataset manifests. It records the row cap and each symbol's
`complete_range` state, so capped coverage cannot survive merely as terminal
output or be mistaken for the entire requested interval. Quality reports and
full findings JSONL are atomically published under per-run names and their
SHA-256/byte counts are bound into the same ingestion manifest.

The public research reader verifies metadata, URI/symbol semantics, exact raw
record membership, inverse raw-to-normalized coverage, scales, parts and
sidecars before exposing rows. Its batch API performs physical-order incremental
quality checks while a single upstream Arrow stream is staged through a
memory-limited, spill-capable DuckDB canonical sort. The legacy eager reader has
an independent finite materialization guard checked before any Parquet row read.

For M8 specifically, the raw acquisition manifest and normalized manifests are
different immutable stages and must not be collapsed into one post hoc record.
A self-contained run preserves the acquisition authority as an exact raw-only
copy below `data/input`; normalization, Parquet, and DQ evidence live separately
below `data/normalized_input`.  The final all-date manifest is rooted at `data`
so it may bind both trees without making derived artifacts members of the raw
authority.  A failed run inventories every pre-terminal regular file, binds any
completed and failed normalization evidence, and applies the same verifier both
before and after atomic publication.
A successful final bundle binds the raw acquisition manifest, final normalized
manifest for all eight declared symbol/dates, official `CHECKSUM` and metadata
bodies plus their sidecars, per-symbol and aggregate lock hashes, and the clean
real Git revision. Verification rejects a dirty, unborn, synthetic, or changed
source identity and rechecks all bound artifact bytes before reuse.

External raw and normalized data are ignored by Git. Small deterministic test
fixtures may be committed only under tests and must state whether they are
synthetic or sampled public observations.

### Frozen dual-symbol live-L2 session authority

Each prospective L2 date uses one absolute UTC start/end barrier and one
cross-symbol authority for BTCUSDT and ETHUSDT. The two single-symbol producers
run concurrently and preserve raw frames before parsing. Coverage is computed
only from intervals backed by consecutive `OBSERVED` reconstructed book states;
raw socket first/last span, stale updates, excluded messages, gaps, and silent
holes do not create coverage. The session gate uses the exact union/intersection
of those intervals, requires one sufficiently long valid continuity epoch,
reconciles raw/normalized/reconstructed/excluded rows and snapshot anchors, and
enforces the frozen gap, error, warning, frame-byte, and Arrow-batch limits.

All per-symbol raw journals, snapshots, normalized data, manifests, quality
reports, and capture summaries must form an exhaustive regular-file inventory.
A passing dual-symbol session is published atomically with checksums and exact
`_SUCCESS` bytes; a typed capture/gate failure publishes an immutable
`INSUFFICIENT_DATA` authority. Permission, local-I/O, source/config/protocol
drift, or program faults retain nonterminal raw evidence but may not publish a
research terminal. Neither terminal authorizes live trading.

## Quality policy

Validators never sort, deduplicate, clip, interpolate, or rewrite observations.
They emit typed findings for duplicate trades, timestamp/order reversals,
availability/receipt contradictions, sequence gaps or stale updates, crossed
books, invalid price/quantity, scale mismatches, abnormal spread, nonmonotone
depth, long silence, and receipt-clock reversal. A downstream research view may
exclude a row, but it must preserve the normalized source and report the
exclusion count/reason.

Incremental validators retain only a bounded in-memory finding preview; exact
duplicate/clock/sequence state spills to SQLite and complete findings stream to
JSONL. A failed validation closes and removes only its temporary sink, leaving
any previously published evidence untouched.
