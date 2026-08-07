"""Leakage-safe research datasets, time splits, and transparent models."""

from microstructure.research.features import (
    ResearchDataError,
    TemporalAudit,
    TemporalLeakageError,
    add_future_event_labels,
    build_l1_trade_features,
    build_research_frame,
    model_feature_columns,
    validate_temporal_contract,
)
from microstructure.research.models import (
    BootstrapResult,
    ModelLadderResult,
    block_bootstrap_metric,
    classification_metrics,
    evaluate_model_ladder,
    paired_block_bootstrap_difference,
)
from microstructure.research.splits import (
    PurgedFold,
    SplitError,
    WalkForwardPlan,
    expanding_walk_forward_splits,
)

__all__ = [
    "BootstrapResult",
    "ModelLadderResult",
    "PurgedFold",
    "ResearchDataError",
    "SplitError",
    "TemporalAudit",
    "TemporalLeakageError",
    "WalkForwardPlan",
    "add_future_event_labels",
    "block_bootstrap_metric",
    "build_l1_trade_features",
    "build_research_frame",
    "classification_metrics",
    "evaluate_model_ladder",
    "expanding_walk_forward_splits",
    "model_feature_columns",
    "paired_block_bootstrap_difference",
    "validate_temporal_contract",
]
