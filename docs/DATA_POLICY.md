# Data policy

## Public repository boundary

The public repository and its Hugging Face mirror contain only Git-tracked
source code, configuration, tests, documentation, and small placeholder files.
They do not contain exchange observations or generated research bundles.

The following local paths are excluded from every public package:

- `data/raw/`, `data/normalized/`, `data/derived/`, `data/models/`, and
  `data/quality/`;
- `data/m8/`, `data/m8_l2/`, and `data/_ingestion_manifests/`;
- `artifacts/runs/`;
- local environments, caches, bytecode, credentials, and dashboard secrets.

Empty `.gitkeep` placeholders may exist in the standalone GitHub repository but
are omitted from the Hugging Face publication mirror where practical.

## Provider terms

The research adapters use credential-free public Binance market-data endpoints
and official archive metadata. Public availability is not treated as a grant to
redistribute provider data. Users who reproduce a public-data study obtain the
inputs independently and remain responsible for the provider's current terms,
rate limits, and permitted uses.

## Research outputs

Source-controlled prose may summarize bounded aggregate validation counts and
protocol terminals. Raw events, normalized rows, derived features, fitted model
states, account-like records, and generated run bundles are not redistributed.
Synthetic fixtures and smoke outputs are labeled `SYNTHETIC_SMOKE` and support
software verification only.

The public website and Space may encode low-dimensional aggregate values already
stated in tracked README or status prose—for example row counts, warning counts,
and a declared terminal state—when they link back to that source and preserve its
evidence label. They do not publish additional model or execution metrics.

## Trading boundary

The repository has no authenticated exchange client, account connection, or
order-entry path. Nothing in the public package is investment advice or evidence
of executable profit.
