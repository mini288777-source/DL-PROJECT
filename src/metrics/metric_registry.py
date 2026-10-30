"""
Canonical Metric Registry and Specification.
Single Source of Truth for all evaluation metrics across the repository.
Prevents naming collisions, ambiguous formulations, and metric confusion.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    display_name: str
    definition: str
    formula: str
    unit: str
    aggregation: str
    positive_direction: bool  # True if higher is better, False if lower is better
    scope: str                # e.g., 'stage1', 'stage2', 'end_to_end', 'global'
    source_function: str


METRIC_REGISTRY: Dict[str, MetricDefinition] = {
    "MACRO_F1": MetricDefinition(
        metric_id="MACRO_F1",
        display_name="Macro-Averaged F1-Score",
        definition="Unweighted mean of F1-scores across all evaluated classes, giving equal weight to minority classes.",
        formula="mean(F1_c for c in classes)",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="global",
        source_function="sklearn.metrics.f1_score(average='macro', zero_division=0)",
    ),
    "END_TO_END_MACRO_F1": MetricDefinition(
        metric_id="END_TO_END_MACRO_F1",
        display_name="End-to-End Hierarchical Macro-F1",
        definition="Macro-F1 evaluated on the end-to-end multi-class predictions across all 7 canonical classes.",
        formula="mean(F1_c for c in canonical_7_classes)",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="end_to_end",
        source_function="src.models.evaluation.compute_intent_metrics",
    ),
    "EMERGENCY_RECALL": MetricDefinition(
        metric_id="EMERGENCY_RECALL",
        display_name="End-to-End Emergency Recall",
        definition="Fraction of ground-truth emergency utterances correctly classified as emergency.",
        formula="TP_emergency / (TP_emergency + FN_emergency)",
        unit="percentage [0, 100]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="end_to_end",
        source_function="src.models.evaluation.compute_intent_metrics",
    ),
    "EMERGENCY_PRECISION_STAGE1": MetricDefinition(
        metric_id="EMERGENCY_PRECISION_STAGE1",
        display_name="Stage-1 Emergency Precision",
        definition="Precision of the binary Stage-1 Emergency Gate in identifying emergency utterances.",
        formula="TP_s1 / (TP_s1 + FP_s1)",
        unit="percentage [0, 100]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="stage1",
        source_function="src.metrics.stage1_precision",
    ),
    "EMERGENCY_PRECISION_END_TO_END": MetricDefinition(
        metric_id="EMERGENCY_PRECISION_END_TO_END",
        display_name="End-to-End Emergency Precision",
        definition="Precision of final emergency predictions across the full uncurated test distribution (~95% negative).",
        formula="TP_emergency / (TP_emergency + FP_emergency)",
        unit="percentage [0, 100]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="end_to_end",
        source_function="src.metrics.end_to_end_precision",
    ),
    "FER": MetricDefinition(
        metric_id="FER",
        display_name="False-Emergency Rate (FER)",
        definition="Fraction of general speech / non-emergency utterances incorrectly routed as an emergency.",
        formula="FP_emergency / Total_General_Speech = 1 - Specificity_s1",
        unit="percentage [0, 100]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better
        scope="stage1",
        source_function="src.metrics.false_emergency_rate",
    ),
    "IPER": MetricDefinition(
        metric_id="IPER",
        display_name="Intent Preservation Rate (IPER)",
        definition="Fraction of emergency utterances correctly routed and preserved as ANY valid emergency intent.",
        formula="preserved_emergency_utterances / total_emergency_utterances",
        unit="percentage [0, 100]",
        aggregation="mean_over_conditions",
        positive_direction=True,  # Higher is better
        scope="end_to_end",
        source_function="src.metrics.compute_iper",
    ),
    "IPER_ERROR": MetricDefinition(
        metric_id="IPER_ERROR",
        display_name="Intent Preservation Error Rate",
        definition="Complement of Intent Preservation Rate; fraction of emergency utterances lost to general speech or failed.",
        formula="1.0 - IPER",
        unit="percentage [0, 100]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better
        scope="end_to_end",
        source_function="src.metrics.compute_iper_error",
    ),
    "STAGE2_MACRO_F1": MetricDefinition(
        metric_id="STAGE2_MACRO_F1",
        display_name="Stage-2 Subtype Macro-F1",
        definition="Macro-F1 of the conditional emergency subtype router evaluated strictly on confirmed emergency samples.",
        formula="mean(F1_sub for sub in feasible_emergency_classes)",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="stage2",
        source_function="src.models.evaluation.compute_stage2_metrics",
    ),
    "STAGE2_BALANCED_ACCURACY": MetricDefinition(
        metric_id="STAGE2_BALANCED_ACCURACY",
        display_name="Stage-2 Balanced Accuracy",
        definition="Macro-averaged recall across feasible emergency classes at Stage 2.",
        formula="mean(Recall_c for c in feasible_emergency_classes)",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=True,
        scope="stage2",
        source_function="sklearn.metrics.balanced_accuracy_score",
    ),
    "EMERGENCY_FORGETTING": MetricDefinition(
        metric_id="EMERGENCY_FORGETTING",
        display_name="Emergency Recall Catastrophic Forgetting",
        definition="Degradation in emergency class recall between initial calibration baseline (B0) and continual adaptation state (Bt). Positive value indicates catastrophic forgetting.",
        formula="Emergency_Recall(B0) - Emergency_Recall(Bt)",
        unit="percentage [-100, 100]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better (<=0 means retention or improvement)
        scope="continual",
        source_function="src.continual.replay.compute_emergency_forgetting",
    ),
    "EXPECTED_CALIBRATION_ERROR": MetricDefinition(
        metric_id="EXPECTED_CALIBRATION_ERROR",
        display_name="Expected Calibration Error (ECE)",
        definition="Binned weighted difference between model confidence and observed accuracy across probability bins.",
        formula="sum(|B_m|/N * |acc(B_m) - conf(B_m)|)",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better
        scope="calibration",
        source_function="src.continual.drift.compute_expected_calibration_error",
    ),
    "MAXIMUM_CALIBRATION_ERROR": MetricDefinition(
        metric_id="MAXIMUM_CALIBRATION_ERROR",
        display_name="Maximum Calibration Error (MCE)",
        definition="Worst-case bin calibration error between confidence and empirical accuracy.",
        formula="max(|acc(B_m) - conf(B_m)|)",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better
        scope="calibration",
        source_function="src.continual.drift.compute_maximum_calibration_error",
    ),
    "BRIER_SCORE": MetricDefinition(
        metric_id="BRIER_SCORE",
        display_name="Multi-Class Brier Score",
        definition="Mean squared error of multi-class predicted probability vector against one-hot outcome indicator.",
        formula="1/N * sum_i sum_k (p_ik - y_ik)^2",
        unit="scalar [0, 2]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better
        scope="calibration",
        source_function="src.continual.drift.compute_brier_score",
    ),
    "REPRESENTATION_MMD": MetricDefinition(
        metric_id="REPRESENTATION_MMD",
        display_name="Representation Maximum Mean Discrepancy",
        definition="Kernel two-sample Maximum Mean Discrepancy (MMD) between reference and current latent representations.",
        formula="MMD^2(P, Q) via RBF kernel with median bandwidth heuristic",
        unit="scalar [0, 2]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better (closer distributions)
        scope="drift",
        source_function="src.continual.drift.compute_mmd_rbf",
    ),
    "PREDICTION_TVD": MetricDefinition(
        metric_id="PREDICTION_TVD",
        display_name="Prediction Total Variation Distance",
        definition="Total variation distance between reference and current empirical class prediction distributions.",
        formula="0.5 * sum_c |P(Y=c) - Q(Y=c)|",
        unit="scalar [0, 1]",
        aggregation="mean_over_conditions",
        positive_direction=False,  # Lower is better
        scope="drift",
        source_function="src.continual.drift.compute_prediction_tvd",
    ),
}


def get_metric(metric_id: str) -> MetricDefinition:
    """Retrieve canonical metric specification by ID."""
    clean_id = metric_id.upper().strip()
    if clean_id not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric ID: '{metric_id}'. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[clean_id]


def list_metrics() -> List[str]:
    """List all registered canonical metric IDs."""
    return list(METRIC_REGISTRY.keys())
