"""Leakage-safe research datasets, time splits, and transparent models."""

from microstructure.research.features import (
    ResearchDataError,
    TemporalAudit,
    TemporalLeakageError,
    add_future_event_labels,
    build_l1_trade_features,
    build_research_features,
    build_research_frame,
    model_feature_columns,
    validate_temporal_contract,
)
from microstructure.research.l2_analysis import (
    L2DescriptiveAnalysis,
    build_l2_descriptive_analysis,
)
from microstructure.research.l2_multidate import (
    L2EndpointSpec,
    L2ObservedInterval,
    L2RegimeFit,
    L2ResearchError,
    apply_l2_regimes,
    build_l2_endpoint_frames,
    dependency_block_expression,
    fit_l2_regime_thresholds,
    l2_model_feature_columns,
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
    "L2DescriptiveAnalysis",
    "L2EndpointSpec",
    "L2ObservedInterval",
    "L2RegimeFit",
    "L2ResearchError",
    "ModelLadderResult",
    "PurgedFold",
    "ResearchDataError",
    "SplitError",
    "TemporalAudit",
    "TemporalLeakageError",
    "WalkForwardPlan",
    "add_future_event_labels",
    "apply_l2_regimes",
    "block_bootstrap_metric",
    "build_l1_trade_features",
    "build_l2_descriptive_analysis",
    "build_l2_endpoint_frames",
    "build_research_features",
    "build_research_frame",
    "classification_metrics",
    "dependency_block_expression",
    "evaluate_model_ladder",
    "expanding_walk_forward_splits",
    "fit_l2_regime_thresholds",
    "l2_model_feature_columns",
    "model_feature_columns",
    "paired_block_bootstrap_difference",
    "validate_temporal_contract",
]
