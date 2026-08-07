from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from microstructure.config import ProjectConfig, load_config
from microstructure.pipeline import PipelineError, reproduce
from microstructure.reporting import ChecksumMismatchError, load_run_bundle


def _config(project_root: Path) -> ProjectConfig:
    config_path = project_root / "configs" / "pipeline-smoke.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
[run]
name = "pipeline-smoke"
evidence_tier = "SYNTHETIC_SMOKE"
seed = 20260807

[data]
mode = "synthetic"
source = "synthetic_pipeline_fixture_v1"
symbols = ["BTCUSDT", "ETHUSDT"]
start = "2024-01-02T00:00:00Z"
events_per_symbol = 72
partition_root = "data/normalized"
schema_version = "1.0.0"

[quality]
max_spread_bps = 100.0
max_silence_ms = 5000
fail_on_error = true

[features]
trade_windows = [2, 4]
volatility_window = 4
intensity_window = 3
label_horizon_events = 2
large_trade_quantile = 0.95

[evaluation]
min_train_events = 24
validation_events = 12
test_events = 12
step_events = 12
embargo_events = 2
bootstrap_samples = 8
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
max_position_units = 0.01
order_size_units = 0.002
limit_fill_base_probability = 0.55
queue_ahead_units = 0.001
limit_max_age_events = 5
cancel_latency_events = 1
liquidate_at_end = true
capacity_multipliers = [0.5, 1.0]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)


@pytest.fixture(scope="module")
def completed_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ProjectConfig, Path]:
    project_root = tmp_path_factory.mktemp("pipeline-project")
    config = _config(project_root)
    run_dir = project_root / "artifacts" / "runs" / "pipeline-smoke"
    return config, reproduce(config, run_dir)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reproduce_builds_verified_honestly_labeled_vertical_slice(
    completed_bundle: tuple[ProjectConfig, Path],
) -> None:
    config, run_dir = completed_bundle
    bundle = load_run_bundle(run_dir)

    assert bundle.evidence_tier == "SYNTHETIC_SMOKE"
    assert bundle.manifest["data"]["mode"] == "synthetic"
    assert bundle.provenance["requested_evidence_tier"] == "SYNTHETIC_SMOKE"
    assert bundle.provenance["effective_evidence_tier"] == "SYNTHETIC_SMOKE"
    assert bundle.manifest["run_key"] == bundle.provenance["run_key"]
    assert len(str(bundle.manifest["run_key"])) == 64
    assert len(bundle.provenance["input_manifest_sha256"]) == 2
    assert bundle.observed_start_utc < bundle.observed_end_utc
    assert bundle.quality["summary"]["errors"] == 0
    assert bundle.manifest["research"]["feature_ready_rows"] > 0

    normalized_parts = sorted((run_dir / "data" / "normalized").rglob("*.parquet"))
    normalized_manifests = sorted((run_dir / "data" / "normalized").rglob("*.json"))
    assert normalized_parts
    assert normalized_manifests
    assert (run_dir / "research" / "research_frame.parquet").is_file()
    evaluation = pl.read_parquet(run_dir / "research" / "evaluation_frame.parquet")
    assert evaluation.get_column("feature_ready").all()
    assert evaluation.height == bundle.manifest["research"]["evaluation_rows"]
    folds = _read_json(run_dir / "research" / "folds.json")
    assert isinstance(folds, dict)
    assert folds["index_basis"].startswith("zero-based row positions")
    recorded_indices = [
        int(index)
        for fold in folds["folds"]
        for key in ("train_indices", "validation_indices")
        for index in fold[key]
    ]
    recorded_indices.extend(int(index) for index in folds["final_train_indices"])
    recorded_indices.extend(int(index) for index in folds["test_indices"])
    assert recorded_indices
    assert min(recorded_indices) >= 0
    assert max(recorded_indices) < evaluation.height
    assert (run_dir / "research" / "folds.json").is_file()
    analysis_manifest = _read_json(run_dir / "analysis" / "manifest.json")
    assert isinstance(analysis_manifest, dict)
    assert analysis_manifest["descriptive_only"] is True
    assert analysis_manifest["economic_claim_authorized"] is False
    assert analysis_manifest["threshold_source"] == "final_training_period_only"
    expected_analysis = {
        "intraday_liquidity",
        "ofi_future_return",
        "signal_decay_curve",
        "signal_half_life",
        "event_time_impact_labels",
        "large_trade_price_impact",
        "liquidity_recovery",
        "market_regimes",
        "regime_outcomes",
        "regime_model_performance",
        "cross_instrument_stability",
        "feature_stability",
    }
    assert set(analysis_manifest["artifacts"]) == expected_analysis
    assert all((run_dir / "analysis" / f"{name}.parquet").is_file() for name in expected_analysis)
    regime_performance = pl.read_parquet(run_dir / "analysis/regime_model_performance.parquet")
    assert regime_performance.get_column("split").unique().to_list() == ["test"]
    assert set(regime_performance.get_column("threshold_source")) == {
        "caller_supplied_final_training_period"
    }

    families = {str(row["family"]) for row in bundle.predictive_metrics}
    assert families == {"baseline", "logistic", "logistic_l2", "shallow_tree"}
    assert {str(row["split"]) for row in bundle.predictive_metrics} == {
        "validation",
        "test",
    }
    test_metric_rows = [row for row in bundle.predictive_metrics if row["split"] == "test"]
    assert {int(row["bootstrap_block_width_events"]) for row in test_metric_rows} == {4}
    assert {str(row["bootstrap_block_policy"]) for row in test_metric_rows} == {
        "pooled_dense_decision_time_clusters_2x_label_horizon"
    }
    all_predictions = pl.read_parquet(run_dir / "models" / "predictions.parquet")
    assert (
        all_predictions.group_by("decision_ts_ns")
        .agg(pl.col("bootstrap_block").n_unique().alias("block_count"))
        .filter(pl.col("block_count") != 1)
        .is_empty()
    )
    selected = pl.read_parquet(run_dir / "models" / "selected_test_predictions.parquet")
    assert selected.get_column("model").n_unique() == 1
    assert selected.get_column("split").unique().to_list() == ["test"]
    assert selected.get_column("is_oos").all()
    assert selected.get_column("continuity_id").null_count() == 0

    execution_events = pl.read_parquet(run_dir / "execution" / "events.parquet")
    assert execution_events.filter(pl.col("event_ts_ns") != pl.col("decision_ts_ns")).is_empty()
    assert {str(row["order_type"]) for row in bundle.execution_metrics} == {
        "market",
        "limit",
    }
    sensitivity = _read_json(run_dir / "metrics" / "execution_sensitivity.json")
    assert isinstance(sensitivity, list)
    assert {str(row["order_type"]) for row in sensitivity} == {"market", "limit"}
    assert {float(row["size_multiplier"]) for row in sensitivity} == {0.5, 1.0}

    technical = (run_dir / "reports" / "technical_report.md").read_text(encoding="utf-8")
    memo = (run_dir / "reports" / "executive_memo.md").read_text(encoding="utf-8")
    table = (run_dir / "reports" / "model_comparison.md").read_text(encoding="utf-8")
    for rendered in (technical, memo, table):
        assert "SYNTHETIC SMOKE" in rendered
    assert "NOT EMPIRICAL OR INVESTMENT EVIDENCE" in technical
    assert "authorize no capital deployment" in memo
    assert "No capital recommendation is made" in technical

    checksums_before = (run_dir / "checksums.sha256").read_bytes()
    success_mtime = (run_dir / "_SUCCESS").stat().st_mtime_ns
    assert reproduce(config, run_dir) == run_dir
    assert (run_dir / "checksums.sha256").read_bytes() == checksums_before
    assert (run_dir / "_SUCCESS").stat().st_mtime_ns == success_mtime
    assert not list(run_dir.parent.glob(".pipeline-smoke.staging-*"))


def test_checksum_corruption_is_rejected_without_overwrite(
    completed_bundle: tuple[ProjectConfig, Path], tmp_path: Path
) -> None:
    config, source = completed_bundle
    corrupted = tmp_path / "corrupted-run"
    shutil.copytree(source, corrupted)
    metrics_path = corrupted / "metrics" / "execution_metrics.json"
    original = metrics_path.read_bytes()
    metrics_path.write_bytes(original + b"\n")

    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        reproduce(config, corrupted)
    assert metrics_path.read_bytes() == original + b"\n"


def test_independent_runs_have_deterministic_semantic_metrics(
    completed_bundle: tuple[ProjectConfig, Path],
) -> None:
    config, first = completed_bundle
    second = reproduce(config, first.parent / "pipeline-smoke-repeat")

    assert (
        _read_json(first / "run_manifest.json")["run_key"]
        == _read_json(second / "run_manifest.json")["run_key"]
    )
    assert (
        _read_json(first / "provenance.json")["run_key_inputs"]
        == _read_json(second / "provenance.json")["run_key_inputs"]
    )

    for relative in (
        "metrics/predictive_metrics.json",
        "metrics/execution_metrics.json",
        "metrics/execution_sensitivity.json",
    ):
        assert _read_json(first / relative) == _read_json(second / relative)


def test_existing_incomplete_target_is_not_repaired_or_overwritten(
    completed_bundle: tuple[ProjectConfig, Path], tmp_path: Path
) -> None:
    config, _ = completed_bundle
    target = tmp_path / "incomplete"
    target.mkdir()
    marker = target / "producer-failed.txt"
    marker.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="not a verified completed bundle"):
        reproduce(config, target)
    assert marker.read_text(encoding="utf-8") == "preserve me\n"


def test_completed_target_from_different_config_is_not_reused(
    completed_bundle: tuple[ProjectConfig, Path], tmp_path: Path
) -> None:
    config, target = completed_bundle
    alternate_path = tmp_path / "alternate.toml"
    alternate_path.write_text(
        config.path.read_text(encoding="utf-8").replace("seed = 20260807", "seed = 20260808"),
        encoding="utf-8",
    )
    alternate = load_config(alternate_path)

    with pytest.raises(PipelineError, match="different configuration"):
        reproduce(alternate, target)


def test_completed_target_from_different_source_tree_is_not_reused(
    completed_bundle: tuple[ProjectConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, target = completed_bundle
    monkeypatch.setattr(
        "microstructure.pipeline.git_source_tree_sha256",
        lambda project_root: "f" * 64,
    )

    with pytest.raises(PipelineError, match="different Git/source-tree state"):
        reproduce(config, target)


def test_synthetic_reproduction_rejects_public_manifest_anchor(
    completed_bundle: tuple[ProjectConfig, Path], tmp_path: Path
) -> None:
    config, _ = completed_bundle

    with pytest.raises(PipelineError, match="does not accept a public input manifest"):
        reproduce(
            config,
            tmp_path / "wrongly-anchored",
            ingestion_manifest_path=tmp_path / "ingestion.json",
            ingestion_manifest_sha256="a" * 64,
        )


def test_public_reproduction_requires_and_hashes_explicit_manifest_anchor(
    tmp_path: Path,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "public_sample.toml")
    target = tmp_path / "public-run"

    with pytest.raises(PipelineError, match="requires an explicit ingestion manifest"):
        reproduce(config, target)

    manifest = tmp_path / "ingestion.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PipelineError, match="bytes do not match"):
        reproduce(
            config,
            target,
            ingestion_manifest_path=manifest,
            ingestion_manifest_sha256="a" * 64,
        )
