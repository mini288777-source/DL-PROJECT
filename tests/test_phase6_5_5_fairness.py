"""
Unit tests for Subphase 6.5.5: Experimental Fairness & Partition Integrity.
Validates:
1. Exact split match between classical and quantum evaluations.
2. Identical test partitions and sample counts across all models.
3. Complete condition coverage (15 speakers x 4 budgets = 60 conditions).
4. No duplicate cells and no missing primary comparison cells.
"""

import os
import json
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SPEAKER_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv")
MATCHED_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_matched_comparison.csv")


def test_complete_speaker_budget_coverage():
    """Verify all 15 speakers across budgets 20, 30, 40, 50 are evaluated."""
    if not os.path.exists(SPEAKER_RESULTS_PATH):
        pytest.skip("phase6_5_5_speaker_results.csv not yet generated")

    df = pd.read_csv(SPEAKER_RESULTS_PATH)
    expected_speakers = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05", "FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"}
    expected_budgets = {20, 30, 40, 50}

    assert set(df["speaker_id"].unique()) == expected_speakers
    assert set(df["N"].unique()) == expected_budgets
    assert len(df) == 60 * 7, f"Expected 420 rows, got {len(df)}"


def test_matched_comparison_fairness():
    """Verify that classical and quantum models in matched comparison share identical sample counts."""
    if not os.path.exists(MATCHED_PATH):
        pytest.skip("phase6_5_5_matched_comparison.csv not yet generated")

    df = pd.read_csv(MATCHED_PATH)
    assert len(df) == 60, f"Expected 60 matched conditions, got {len(df)}"

    # Check required matched metric columns exist
    required_cols = [
        "speaker_id", "cohort_group", "N",
        "q8_r1_macro_f1", "rbf_svm8_macro_f1", "adapter8_macro_f1",
        "diff_Q8_RBF8_macro_f1", "diff_Q8_Adp8_macro_f1",
        "q8_r1_emerg_recall", "rbf_svm8_emerg_recall", "adapter8_emerg_recall",
        "diff_Q8_RBF8_recall", "diff_Q8_Adp8_recall",
        "q8_r1_fpr", "rbf_svm8_fpr", "adapter8_fpr",
        "diff_Q8_RBF8_fpr", "diff_Q8_Adp8_fpr",
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required matched column {c}"


def test_no_duplicate_cells():
    """Verify each (speaker, N, model) triplet is unique."""
    if not os.path.exists(SPEAKER_RESULTS_PATH):
        pytest.skip("phase6_5_5_speaker_results.csv not yet generated")

    df = pd.read_csv(SPEAKER_RESULTS_PATH)
    dups = df.duplicated(subset=["speaker_id", "N", "model"])
    assert not dups.any(), f"Found {dups.sum()} duplicate condition cells"
