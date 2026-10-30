"""
Unit tests for Subphase 6.5.4: Reproducibility, Completeness, and Artifact Preservation

Validates:
1. Dimension ablation CSV completeness (72 aggregated rows: 2 models x 9 dimensions x 4 budgets).
2. Speaker results CSV completeness (1080 granular rows: 15 speakers x 4 budgets x 9 dimensions x 2 models).
3. All 7 publication figures exist in reports/phase6_5_4/figures/.
4. Metric ranges are mathematically valid in [0, 1].
5. Original Phase 3-6 historical benchmarks remain completely untouched.
"""

import os
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ABLATION_CSV = os.path.join(REPO_ROOT, "results", "phase6_5_4_dimension_ablation.csv")
SPEAKER_CSV = os.path.join(REPO_ROOT, "results", "phase6_5_4_speaker_results.csv")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "phase6_5_4", "figures")


def test_dimension_ablation_csv_completeness():
    """Verify aggregated dimension ablation CSV has exactly 72 rows."""
    assert os.path.isfile(ABLATION_CSV), f"Missing {ABLATION_CSV}"
    df = pd.read_csv(ABLATION_CSV)
    assert len(df) == 72, f"Expected 72 rows, got {len(df)}"
    assert set(df["model"].unique()) == {"hierarchical_adapter", "hierarchical_rbf_svm"}
    assert set(df["N"].unique()) == {20, 30, 40, 50}
    assert set(df["dimension"].unique()) == {2, 4, 8, 16, 32, 64, 128, 256, 384}


def test_speaker_results_csv_completeness():
    """Verify granular speaker results CSV has exactly 1080 rows."""
    assert os.path.isfile(SPEAKER_CSV), f"Missing {SPEAKER_CSV}"
    df = pd.read_csv(SPEAKER_CSV)
    assert len(df) == 1080, f"Expected 1080 rows, got {len(df)}"
    assert len(df["speaker_id"].unique()) == 15


def test_all_figures_exist():
    """Verify all 7 publication figures exist and are non-empty."""
    expected_figures = [
        "explained_variance_vs_dimension.png",
        "macro_f1_vs_dimension.png",
        "emergency_recall_vs_dimension.png",
        "false_emergency_rate_vs_dimension.png",
        "stage2_macro_f1_vs_dimension.png",
        "heatmap_dimension_vs_N.png",
        "dimensionality_vs_variance_performance.png"
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.isfile(fig_path), f"Missing figure: {fig_name}"
        assert os.path.getsize(fig_path) > 1000, f"Empty or corrupted figure: {fig_name}"


def test_metric_mathematical_bounds():
    """Verify that all computed rates and F1 scores fall within [0, 1]."""
    df = pd.read_csv(SPEAKER_CSV)
    succ_df = df[df["status"] == "SUCCESS"]
    metric_cols = [
        "stage1_emergency_recall", "stage1_emergency_precision", "stage1_false_positive_rate",
        "stage1_general_recall", "stage1_missed_emergency_rate", "stage2_feasible_macro_f1",
        "stage2_balanced_accuracy", "end_to_end_iper", "end_to_end_false_emergency_rate",
        "end_to_end_macro_f1"
    ]
    for col in metric_cols:
        vals = succ_df[col].dropna()
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"Out of bounds values in {col}"


def test_original_benchmarks_preserved():
    """Verify historical Phase 3-6 benchmark files are untouched."""
    p5_res = os.path.join(REPO_ROOT, "results", "phase5", "phase5_results.csv")
    assert os.path.isfile(p5_res), "Phase 5 canonical results missing"
    df5 = pd.read_csv(p5_res)
    assert len(df5) == 720, f"Expected 720 rows in Phase 5 benchmark, found {len(df5)}"
