from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import polars as pl
import pytest

import microstructure.public_pipeline as public_pipeline
from microstructure.cli import main
from microstructure.config import ProjectConfig, load_config
from microstructure.ingestion import IngestionResult, ingest_public_trades
from microstructure.pipeline import PipelineError, reproduce
from microstructure.provenance import sha256_file
from microstructure.public_data import PublicDataError, PublicTrades, read_public_trades
from microstructure.public_pipeline import (
    EXECUTION_EXCLUSION_REASON,
    PublicPipelineError,
    produce_public_trade_run,
)
from microstructure.reporting import load_run_bundle, verify_checksums
from microstructure.research.models import paired_block_bootstrap_difference


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.status_code = 200
        self._payload = payload
        self.content = json.dumps(payload, separators=(",", ":")).encode()
        self.text = self.content.decode()
        self.headers: dict[str, str] = {}
        self.url = "https://data-api.binance.vision/fixture"

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: Mapping[str, object], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        response = self.responses.pop(0)
        response.url = f"{url}?{urlencode(params)}"
        return response


@dataclass(frozen=True, slots=True)
class _CompletedFixture:
    config: ProjectConfig
    ingestion: IngestionResult
    public: PublicTrades
    run_dir: Path


@dataclass(frozen=True, slots=True)
class _UnifiedFixture:
    source: _CompletedFixture
    run_dir: Path


def _metadata(symbol: str) -> dict[str, object]:
    base = "BTC" if symbol == "BTCUSDT" else "ETH"
    return {
        "symbols": [
            {
                "symbol": symbol,
                "status": "TRADING",
                "baseAsset": base,
                "quoteAsset": "USDT",
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01",
                        "maxPrice": "1000000.00",
                        "tickSize": "0.01",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "10000.000",
                        "stepSize": "0.001",
                    },
                ],
            }
        ]
    }


def _aggregate_trades(symbol: str, *, start_ms: int, rows: int) -> list[dict[str, object]]:
    id_start = 1_000_000 if symbol == "BTCUSDT" else 2_000_000
    price_start = 10_000 if symbol == "BTCUSDT" else 20_000
    price_pattern = (0, 1, 3, 2, 5, 1, 4, 2)
    result: list[dict[str, object]] = []
    for index in range(rows):
        trade_id = id_start + index
        price_ticks = price_start + price_pattern[index % len(price_pattern)]
        quantity_lots = 1 + index % 5
        result.append(
            {
                "a": trade_id,
                "p": f"{price_ticks / 100:.2f}",
                "q": f"{quantity_lots / 1000:.3f}",
                "f": 10_000_000 + index,
                "l": 10_000_000 + index,
                "T": start_ms + index,
                "m": bool(index % 2),
            }
        )
    return result


def _write_config(project_root: Path) -> ProjectConfig:
    config_path = project_root / "configs" / "public-pipeline-fixture.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
[run]
name = "public-trade-fixture"
evidence_tier = "PUBLIC_SAMPLE_PARTIAL"
seed = 20260807

[data]
mode = "binance_rest"
source = "binance_spot_rest"
symbols = ["BTCUSDT", "ETHUSDT"]
start = "2024-01-02T00:00:00Z"
end = "2024-01-02T00:01:00Z"
max_events_per_symbol = 80
raw_root = "data/raw"
partition_root = "data/normalized"
schema_version = "1.0.0"
base_url = "https://data-api.binance.vision"
request_limit = 100
timeout_seconds = 5.0
max_retries = 0

[quality]
max_spread_bps = 100.0
max_silence_ms = 5000
fail_on_error = true

[features]
trade_windows = [2, 4]
volatility_window = 4
intensity_window = 3
label_horizon_events = 2
large_trade_quantile = 0.90

[evaluation]
min_train_events = 24
validation_events = 12
test_events = 12
step_events = 12
embargo_events = 2
bootstrap_samples = 12
calibration_bins = 5

[models]
selection_metric = "log_loss"
logistic_c_values = [1.0]
tree_max_depth_values = [2]
tree_min_samples_leaf = 2

[execution]
decision_latency_events = 1
order_latency_events = 1
maker_fee_bps = 1.0
taker_fee_bps = 4.0
half_spread_bps = 1.0
slippage_bps_per_unit = 0.20
signal_threshold = 0.52
max_position_units = 1.0
order_size_units = 0.1
limit_fill_base_probability = 0.55
queue_ahead_units = 0.1
limit_max_age_events = 5
cancel_latency_events = 1
liquidate_at_end = true
capacity_multipliers = [0.5, 1.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    protocol = project_root / "docs" / "PUBLIC_TRADE_PROTOCOL.md"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(
        "# Frozen public aggregate-trade fixture protocol\n\n"
        "Retrospective PUBLIC_SAMPLE_PARTIAL predictive diagnostics only. "
        "No execution or P&L.\n",
        encoding="utf-8",
    )
    return load_config(config_path)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def completed_public_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> _CompletedFixture:
    project_root = tmp_path_factory.mktemp("public-pipeline-project")
    config = _write_config(project_root)
    start_ms = int(config.data.start.timestamp() * 1000)
    session = _FakeSession(
        [
            _FakeResponse(_metadata("BTCUSDT")),
            _FakeResponse(_aggregate_trades("BTCUSDT", start_ms=start_ms, rows=80)),
            _FakeResponse(_metadata("ETHUSDT")),
            _FakeResponse(_aggregate_trades("ETHUSDT", start_ms=start_ms, rows=80)),
        ]
    )
    ingestion = ingest_public_trades(
        config,
        project_root / "data",
        session=session,  # type: ignore[arg-type]
    )
    assert len(session.calls) == 4
    public = read_public_trades(
        config,
        ingestion.ingestion_manifest_path,
        ingestion_manifest_sha256=ingestion.ingestion_manifest_sha256,
    )

    # A newer-looking decoy proves that the producer never discovers or selects
    # directory contents: the explicit path+digest pair remains the only authority.
    decoy = (
        ingestion.ingestion_manifest_path.parent / "ingestion.manifest-zzzzzzzzzzzzzzzzzzzz.json"
    )
    decoy.write_text("not a manifest\n", encoding="utf-8")
    run_dir = project_root / "artifacts" / "staging-public-run"
    produce_public_trade_run(
        config,
        run_dir,
        ingestion_manifest_path=ingestion.ingestion_manifest_path,
        ingestion_manifest_sha256=ingestion.ingestion_manifest_sha256,
    )
    return _CompletedFixture(config, ingestion, public, run_dir)


@pytest.fixture(scope="module")
def unified_public_bundle(
    completed_public_bundle: _CompletedFixture,
    tmp_path_factory: pytest.TempPathFactory,
) -> _UnifiedFixture:
    source = completed_public_bundle
    target = tmp_path_factory.mktemp("unified-public-pipeline") / "public-run"
    output = reproduce(
        source.config,
        target,
        ingestion_manifest_path=source.ingestion.ingestion_manifest_path,
        ingestion_manifest_sha256=source.ingestion.ingestion_manifest_sha256,
    )
    return _UnifiedFixture(source=source, run_dir=output)


def test_public_run_is_manifest_anchored_partial_and_provenanced(
    completed_public_bundle: _CompletedFixture,
) -> None:
    fixture = completed_public_bundle
    bundle = load_run_bundle(fixture.run_dir)
    provenance = _json(fixture.run_dir / "provenance.json")
    snapshot = _json(fixture.run_dir / "data" / "manifest_snapshot.json")

    assert bundle.evidence_tier == "PUBLIC_SAMPLE_PARTIAL"
    assert bundle.manifest["data"]["rows"] == 160
    assert bundle.manifest["data"]["row_bound"] == 160
    assert bundle.manifest["data"]["all_requested_ranges_complete"] is False
    assert bundle.manifest["data"]["reader_effective_evidence_tier"] == ("PUBLIC_SAMPLE_PARTIAL")
    assert {
        item["symbol"]: item["rows"] for item in bundle.manifest["data"]["symbol_coverage"]
    } == {
        "BTCUSDT": 80,
        "ETHUSDT": 80,
    }
    assert provenance["config_sha256"] == fixture.config.hash
    assert provenance["ingestion_manifest_sha256"] == (fixture.ingestion.ingestion_manifest_sha256)
    assert provenance["ingestion_manifest_absolute_path"] == str(
        fixture.ingestion.ingestion_manifest_path.resolve()
    )
    assert fixture.ingestion.ingestion_manifest_sha256 in provenance["input_manifest_sha256"]
    assert provenance["input_data_sha256"]
    assert set(provenance["git"]) == {"commit", "dirty", "source_tree_sha256"}
    assert len(provenance["git"]["source_tree_sha256"]) == 64
    assert provenance["execution_simulated"] is False
    assert snapshot["manifest_authority"]["policy"].endswith("no directory discovery")
    assert snapshot["manifest_authority"]["sha256"] == (fixture.ingestion.ingestion_manifest_sha256)
    assert fixture.public.polars_trades.get_column("continuity_id").null_count() == 160
    assert bundle.quality["summary"] == {"errors": 0, "warnings": 0}
    assert all(
        audit["trade_ids_contiguous"]
        and audit["availability_clock_nondecreasing"]
        and not audit["source_rows_mutated"]
        for audit in bundle.quality["aggregate_trade_continuity"]
    )

    protocol_path = fixture.run_dir / "protocol" / "PUBLIC_TRADE_PROTOCOL.md"
    assert protocol_path.is_file()
    assert provenance["protocol_sha256"] == sha256_file(protocol_path)
    assert provenance["protocol_sha256"] == sha256_file(
        fixture.config.project_root / "docs" / "PUBLIC_TRADE_PROTOCOL.md"
    )
    assert provenance["run_key_inputs"]["protocol_sha256"] == provenance["protocol_sha256"]
    assert bundle.manifest["artifacts"]["protocol"] == ("protocol/PUBLIC_TRADE_PROTOCOL.md")


def test_per_symbol_folds_are_purged_oos_and_test_never_selects(
    completed_public_bundle: _CompletedFixture,
) -> None:
    run_dir = completed_public_bundle.run_dir
    manifest = _json(run_dir / "run_manifest.json")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        slug = symbol.lower()
        evaluation = pl.read_parquet(run_dir / "research" / slug / "evaluation_frame.parquet")
        folds = _json(run_dir / "research" / slug / "folds.json")
        predictions = pl.read_parquet(run_dir / "models" / slug / "predictions.parquet")
        comparison = pl.read_parquet(run_dir / "models" / slug / "comparison.parquet")
        selected = pl.read_parquet(run_dir / "models" / slug / "selected_test_predictions.parquet")

        assert evaluation.get_column("feature_ready").all()
        assert evaluation.get_column("label_horizon_events").unique().to_list() == [2]
        assert set(evaluation.get_column("continuity_id").drop_nulls().unique())
        assert all(
            not name.startswith(("future_", "label_"))
            for name in manifest["research"]["symbols"][symbol]["feature_columns"]
        )

        # Regression guard: serialized fold count is exactly the evaluated plan,
        # not a nested/quadratically duplicated list.
        validation_fold_ids = set(
            comparison.filter(pl.col("split") == "validation").get_column("fold_id").unique()
        )
        assert len(folds["folds"]) == len(validation_fold_ids)
        assert [fold["fold_id"] for fold in folds["folds"]] == list(range(len(folds["folds"])))

        final_train = {int(index) for index in folds["final_train_indices"]}
        final_test = {int(index) for index in folds["test_indices"]}
        assert final_train
        assert final_test
        assert final_train.isdisjoint(final_test)
        indexed = evaluation.with_row_index("_research_row_id")
        train_rows = indexed.filter(pl.col("_research_row_id").is_in(final_train))
        test_rows = indexed.filter(pl.col("_research_row_id").is_in(final_test))
        train_label_end = train_rows.get_column("label_information_end_ts_ns").max()
        test_decision_start = test_rows.get_column("decision_ts_ns").min()
        assert isinstance(train_label_end, int)
        assert isinstance(test_decision_start, int)
        assert train_label_end < test_decision_start

        test_predictions = predictions.filter(pl.col("split") == "test")
        assert test_predictions.get_column("is_oos").all()
        assert test_predictions.filter(
            pl.col("fit_cutoff_ts_ns") >= pl.col("decision_ts_ns")
        ).is_empty()
        assert test_predictions.get_column("bootstrap_block_width_trades").unique().to_list() == [4]
        assert test_predictions.filter(
            (pl.col("decision_sequence") < pl.col("bootstrap_block_start_trade_id"))
            | (pl.col("decision_sequence") > pl.col("bootstrap_block_end_trade_id"))
        ).is_empty()
        assert comparison.get_column("test_used_for_selection").not_().all()
        assert set(
            comparison.filter(pl.col("selected_on_validation"))
            .get_column("selected_on")
            .drop_nulls()
        ) == {"validation"}
        assert set(
            comparison.filter(pl.col("split") == "test")
            .get_column("bootstrap_block_policy")
            .drop_nulls()
        ) == {"fixed_contiguous_2x_label_horizon"}
        assert selected.get_column("split").unique().to_list() == ["test"]
        assert selected.get_column("requested_model").unique().to_list() == [
            manifest["research"]["symbols"][symbol]["selected_model"]
        ]


def test_paired_hypothesis_uses_identical_test_rows_and_blocks(
    completed_public_bundle: _CompletedFixture,
) -> None:
    fixture = completed_public_bundle
    run_dir = fixture.run_dir
    manifest = _json(run_dir / "run_manifest.json")
    payload = _json(run_dir / "metrics" / "hypothesis_evaluation.json")

    assert manifest["artifacts"]["hypothesis_evaluation"] == ("metrics/hypothesis_evaluation.json")
    assert manifest["research"]["hypothesis_evaluation"] == {
        "artifact": "metrics/hypothesis_evaluation.json",
        "baseline": "historical_prior",
        "caveat": payload["caveat"],
        "cross_instrument_conclusion": payload["cross_instrument_conclusion"]["text"],
        "cross_instrument_pooling": False,
        "delta_definition": "selected_model_minus_historical_prior",
        "exploratory": True,
        "hypotheses": ["H0", "H1_exploratory"],
        "metric": "log_loss",
        "paired_on": ["row_id", "bootstrap_block"],
        "per_symbol_only": True,
        "persistent_alpha_claim_authorized": False,
        "significance_claim_authorized": False,
    }
    assert payload["bootstrap"] == {
        "block_policy": "fixed_contiguous_2x_label_horizon",
        "block_width_trades": 4,
        "ci_level": 0.95,
        "method": "paired fixed-block percentile bootstrap",
        "samples": fixture.config.evaluation.bootstrap_samples,
        "seed_policy": "run_seed + symbol_index*100000 + 20000",
    }
    assert payload["cross_instrument_conclusion"]["status"] == "not_inferred"
    assert payload["cross_instrument_conclusion"]["pooling_performed"] is False
    assert payload["cross_instrument_conclusion"]["persistent_alpha_claim_authorized"] is False

    rows = {row["symbol"]: row for row in payload["per_symbol"]}
    assert set(rows) == {"BTCUSDT", "ETHUSDT"}
    identity_columns = [
        "row_id",
        "y_true",
        "bootstrap_block",
        "bootstrap_block_start_trade_id",
        "bootstrap_block_end_trade_id",
        "bootstrap_block_width_trades",
        "bootstrap_block_policy",
    ]
    for symbol_index, symbol in enumerate(fixture.config.data.symbols):
        slug = symbol.lower()
        predictions = pl.read_parquet(run_dir / "models" / slug / "predictions.parquet")
        comparison = pl.read_parquet(run_dir / "models" / slug / "comparison.parquet")
        row = rows[symbol]
        selected = predictions.filter(
            (pl.col("split") == "test") & (pl.col("requested_model") == row["selected_model"])
        )
        prior = predictions.filter(
            (pl.col("split") == "test") & (pl.col("requested_model") == "historical_prior")
        )

        assert (
            selected.select(identity_columns)
            .sort("row_id")
            .equals(prior.select(identity_columns).sort("row_id"))
        )
        assert selected.get_column("row_id").n_unique() == selected.height
        assert selected.get_column("bootstrap_block").n_unique() == row["n_blocks"]

        direct = paired_block_bootstrap_difference(
            selected,
            prior,
            metric="log_loss",
            block_column="bootstrap_block",
            n_bootstrap=fixture.config.evaluation.bootstrap_samples,
            seed=fixture.config.run.seed + symbol_index * 100_000 + 20_000,
        )

        def _log_loss(frame: pl.DataFrame) -> float:
            losses = []
            for truth, probability in frame.select("y_true", "probability").iter_rows():
                probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
                truth = int(truth)
                losses.append(
                    -(truth * math.log(probability) + (1 - truth) * math.log(1 - probability))
                )
            return sum(losses) / len(losses)

        manual_delta = _log_loss(selected) - _log_loss(prior)
        assert row["point_delta"] == pytest.approx(manual_delta)
        assert row["point_delta"] == pytest.approx(direct.point_estimate)
        assert row["ci_low"] == pytest.approx(direct.lower)
        assert row["ci_high"] == pytest.approx(direct.upper)
        assert row["n_obs"] == selected.height == prior.height
        assert row["n_blocks"] == direct.n_blocks
        assert row["samples"] == direct.n_bootstrap
        assert row["seed"] == direct.seed
        assert row["status"] == direct.status
        assert row["baseline"] == "historical_prior"
        assert row["block_policy"] == "fixed_contiguous_2x_label_horizon"
        assert row["favorable_direction"] == "negative_selected_minus_prior_is_favorable"
        assert row["point_assessment"] in {
            "favorable_point_only",
            "unfavorable_point",
            "point_tie",
            "unavailable",
        }
        assert row["exploratory"] is True
        assert row["significance_claim_authorized"] is False
        assert row["h0_rejection_authorized"] is False

        attached = comparison.filter(pl.col("paired_metric").is_not_null())
        assert attached.height == 1
        attached_row = attached.row(0, named=True)
        assert attached_row["split"] == "test"
        assert attached_row["requested_model"] == row["selected_model"]
        assert attached_row["paired_baseline"] == "historical_prior"
        assert attached_row["paired_metric_delta"] == pytest.approx(row["point_delta"])
        assert attached_row["paired_metric_delta_ci_low"] == pytest.approx(row["ci_low"])
        assert attached_row["paired_metric_delta_ci_high"] == pytest.approx(row["ci_high"])
        assert comparison.filter(pl.col("paired_metric").is_null()).height == (
            comparison.height - 1
        )

    for report_name in (
        "technical_report.md",
        "executive_memo.md",
        "model_comparison.md",
    ):
        report = (run_dir / "reports" / report_name).read_text(encoding="utf-8")
        assert "Paired H0/H1 diagnostic" in report
        assert "negative value favors the selected model" in report
        assert "not p-values or confirmatory significance intervals" in report
        assert "cannot support persistent alpha" in report
        assert "BTCUSDT" in report
        assert "ETHUSDT" in report


def test_trade_only_outputs_have_analysis_reports_integrity_and_no_execution_claim(
    completed_public_bundle: _CompletedFixture,
) -> None:
    run_dir = completed_public_bundle.run_dir
    bundle = load_run_bundle(run_dir)
    exclusion = _json(run_dir / "metrics" / "execution_exclusion.json")
    analysis = _json(run_dir / "analysis" / "manifest.json")

    assert bundle.execution_metrics == ()
    assert bundle.execution_sensitivity == ()
    assert exclusion == {
        "execution_metrics_rows": 0,
        "execution_sensitivity_rows": 0,
        "pnl_calculated": False,
        "profitability_claim_authorized": False,
        "reason": EXECUTION_EXCLUSION_REASON,
        "status": "NOT_RUN",
    }
    assert bundle.manifest["execution_assumptions"]["status"] == "NOT_RUN"
    assert bundle.manifest["execution_assumptions"]["pnl_calculated"] is False
    assert not (run_dir / "execution").exists()
    assert set(analysis["artifacts"]) == {
        "trade_summary",
        "feature_stability",
        "flow_return_analysis",
    }
    assert analysis["descriptive_only"] is True
    assert analysis["economic_claim_authorized"] is False
    assert pl.read_parquet(run_dir / "analysis" / "trade_summary.parquet").height == 2
    assert pl.read_parquet(run_dir / "analysis" / "feature_stability.parquet").height > 0
    assert pl.read_parquet(run_dir / "analysis" / "flow_return_analysis.parquet").height == 2

    for name in ("technical_report.md", "executive_memo.md", "model_comparison.md"):
        rendered = (run_dir / "reports" / name).read_text(encoding="utf-8")
        assert "PUBLIC SAMPLE / PARTIAL EVIDENCE" in rendered
        assert EXECUTION_EXCLUSION_REASON in rendered
    protected = verify_checksums(run_dir)
    checksum_lines = (run_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert protected == len(checksum_lines)
    assert all("_SUCCESS" not in line for line in checksum_lines)
    protected_paths = [run_dir / line.split("  ", 1)[1] for line in checksum_lines]
    assert (run_dir / "_SUCCESS").stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in protected_paths
    )


def test_explicit_manifest_digest_fails_closed_before_success(
    completed_public_bundle: _CompletedFixture, tmp_path: Path
) -> None:
    fixture = completed_public_bundle
    stage = tmp_path / "bad-digest-stage"

    with pytest.raises(PublicDataError, match="ingestion manifest SHA-256 mismatch"):
        produce_public_trade_run(
            fixture.config,
            stage,
            ingestion_manifest_path=fixture.ingestion.ingestion_manifest_path,
            ingestion_manifest_sha256="0" * 64,
        )

    assert stage.is_dir()
    assert not (stage / "_SUCCESS").exists()
    assert not list(stage.iterdir())


def test_trade_id_gap_is_rejected_before_continuity_derivation(
    completed_public_bundle: _CompletedFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = completed_public_bundle
    btc_ids = fixture.public.polars_trades.filter(pl.col("symbol") == "BTCUSDT").get_column(
        "trade_id"
    )
    pivot = int(btc_ids[40])
    gapped = fixture.public.polars_trades.with_columns(
        pl.when((pl.col("symbol") == "BTCUSDT") & (pl.col("trade_id") >= pivot))
        .then(pl.col("trade_id") + 1)
        .otherwise(pl.col("trade_id"))
        .alias("trade_id")
    )
    supplied = replace(fixture.public, polars_trades=gapped)

    def _reader(
        config: ProjectConfig,
        path: str | Path,
        *,
        ingestion_manifest_sha256: str,
    ) -> PublicTrades:
        del config, path, ingestion_manifest_sha256
        return supplied

    monkeypatch.setattr(public_pipeline, "read_public_trades", _reader)
    with pytest.raises(PublicPipelineError, match="not contiguous"):
        produce_public_trade_run(
            fixture.config,
            tmp_path / "gap-stage",
            ingestion_manifest_path=fixture.ingestion.ingestion_manifest_path,
            ingestion_manifest_sha256=fixture.ingestion.ingestion_manifest_sha256,
        )


def test_availability_reversal_is_rejected_before_continuity_derivation(
    completed_public_bundle: _CompletedFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = completed_public_bundle
    btc_ids = fixture.public.polars_trades.filter(pl.col("symbol") == "BTCUSDT").get_column(
        "trade_id"
    )
    pivot = int(btc_ids[40])
    reversed_clock = fixture.public.polars_trades.with_columns(
        pl.when((pl.col("symbol") == "BTCUSDT") & (pl.col("trade_id") == pivot))
        .then(pl.col("available_ts_ns") + 10_000_000)
        .otherwise(pl.col("available_ts_ns"))
        .alias("available_ts_ns")
    )
    supplied = replace(fixture.public, polars_trades=reversed_clock)

    def _reader(
        config: ProjectConfig,
        path: str | Path,
        *,
        ingestion_manifest_sha256: str,
    ) -> PublicTrades:
        del config, path, ingestion_manifest_sha256
        return supplied

    monkeypatch.setattr(public_pipeline, "read_public_trades", _reader)
    with pytest.raises(PublicPipelineError, match="availability clock reverses"):
        produce_public_trade_run(
            fixture.config,
            tmp_path / "clock-stage",
            ingestion_manifest_path=fixture.ingestion.ingestion_manifest_path,
            ingestion_manifest_sha256=fixture.ingestion.ingestion_manifest_sha256,
        )


def test_unified_reproduce_is_atomic_idempotent_relocatable_and_manifest_bound(
    unified_public_bundle: _UnifiedFixture,
    tmp_path: Path,
) -> None:
    fixture = unified_public_bundle
    source = fixture.source
    target = fixture.run_dir
    checksum_before = (target / "checksums.sha256").read_bytes()
    success_mtime = (target / "_SUCCESS").stat().st_mtime_ns
    external_manifest_before = source.ingestion.ingestion_manifest_path.read_bytes()

    assert (
        reproduce(
            source.config,
            target,
            ingestion_manifest_path=source.ingestion.ingestion_manifest_path,
            ingestion_manifest_sha256=source.ingestion.ingestion_manifest_sha256.upper(),
        )
        == target
    )
    assert (target / "checksums.sha256").read_bytes() == checksum_before
    assert (target / "_SUCCESS").stat().st_mtime_ns == success_mtime

    relocated = tmp_path / "relocated" / "_ingestion_manifests" / "ingestion.json"
    relocated.parent.mkdir(parents=True)
    shutil.copyfile(source.ingestion.ingestion_manifest_path, relocated)
    assert (
        reproduce(
            source.config,
            target,
            ingestion_manifest_path=relocated,
            ingestion_manifest_sha256=source.ingestion.ingestion_manifest_sha256,
        )
        == target
    )

    changed = tmp_path / "changed-ingestion.json"
    changed.write_bytes(external_manifest_before + b"\n")
    with pytest.raises(PipelineError, match="different ingestion manifest"):
        reproduce(
            source.config,
            target,
            ingestion_manifest_path=changed,
            ingestion_manifest_sha256=sha256_file(changed),
        )

    assert source.ingestion.ingestion_manifest_path.read_bytes() == external_manifest_before
    assert (target / "checksums.sha256").read_bytes() == checksum_before
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_cli_reuses_verified_public_run_and_reports_scope(
    unified_public_bundle: _UnifiedFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = unified_public_bundle
    source = fixture.source

    exit_code = main(
        [
            "reproduce",
            "--config",
            str(source.config.path),
            "--run-dir",
            str(fixture.run_dir),
            "--ingestion-manifest",
            str(source.ingestion.ingestion_manifest_path),
            "--ingestion-manifest-sha256",
            source.ingestion.ingestion_manifest_sha256,
        ]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "evidence_tier": "PUBLIC_SAMPLE_PARTIAL",
        "observed_end_utc": load_run_bundle(fixture.run_dir).observed_end_utc,
        "observed_start_utc": load_run_bundle(fixture.run_dir).observed_start_utc,
        "run_dir": str(fixture.run_dir),
        "run_id": source.config.run.name,
        "status": "complete",
    }
