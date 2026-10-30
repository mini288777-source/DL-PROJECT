"""
Phase 3 Evaluation and Metrics Utilities.

Computes multi-class intent classification metrics:
Accuracy, Macro-F1, Weighted-F1, Per-Class Precision & Recall, and Confusion Matrices.
Preserves recording-level prediction traces for future IPER computation (Phase 6).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    all_classes: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics across intent classes.

    Args:
        y_true: Ground truth intent strings.
        y_pred: Predicted intent strings.
        all_classes: Full canonical taxonomy of intent classes.

    Returns:
        Dictionary of computed metrics and per-class breakdowns.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=all_classes, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, labels=all_classes, average="macro", zero_division=0))

    # Per-class precision and recall
    prec_per_class = precision_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)

    per_class_metrics = {}
    for i, c in enumerate(all_classes):
        per_class_metrics[c] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(sum(1 for yt in y_true if yt == c)),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": all_classes,
        "test_sample_count": len(y_true),
    }
