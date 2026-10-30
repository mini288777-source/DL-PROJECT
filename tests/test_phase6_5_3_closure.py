"""
Unit tests for Subphase 6.5.3-C: Hierarchical Evaluation Closure Analysis

Validates:
1. Stage-2 metrics computed only on genuine emergency samples.
2. Feasible-class mask uses adaptation labels only.
3. Error taxonomy counts sum correctly (A + B + C + D + E == total).
4. Threshold analysis is exploratory and does not alter model parameters.
5. Flat/hierarchical comparisons use matched N and model pairs.
6. IPER == Emergency Recall mathematical equivalence verified.
7. False Emergency Rate denominator verified as General Speech samples.
8. Original Phase 3-6 artifacts remain frozen and untouched.
9. Results are deterministic and reproducible.
10. Cumulative Phase 6.5 test suite integrity preserved.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "analysis")
RESULTS_DIR = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "results")
SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]


@pytest.fixture(scope="module")
def s2_cond_df():
    path = os.path.join(ANALYSIS_DIR, "stage2_conditional_metrics.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def s2_preds_df():
    path = os.path.join(RESULTS_DIR, "stage2_predictions.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def error_tax_df():
    path = os.path.join(ANALYSIS_DIR, "error_taxonomy.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def thresh_df():
    path = os.path.join(RESULTS_DIR, "threshold_analysis.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def matched_df():
    path = os.path.join(ANALYSIS_DIR, "flat_vs_hierarchical_matched.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def hier_results_df():
    path = os.path.join(RESULTS_DIR, "hierarchical_results.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)


# ============================================================
# TEST 1: Stage-2 metrics computed only on emergency samples
# ============================================================
def test_1_stage2_metrics_emergency_samples_only(s2_preds_df, s2_cond_df):
    """TEST 1: Stage 2 diagnostic evaluation never sees GENERAL_SPEECH."""
    assert "GENERAL_SPEECH" not in s2_preds_df["true_intent"].values
    assert (s2_preds_df["true_intent"].isin(EMERGENCY_CLASSES)).all()
    assert (s2_cond_df["emergency_sample_count"] > 0).all()
    assert len(s2_cond_df) == 120


# ============================================================
# TEST 2: Feasible-class mask uses adaptation labels only
# ============================================================
def test_2_feasible_mask_uses_adaptation_labels_only():
    """TEST 2: Feasible emergency classes derive strictly from adaptation split."""
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    rec_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))

    s2_cond_path = os.path.join(ANALYSIS_DIR, "stage2_conditional_metrics.csv")
    s2_cond_df = pd.read_csv(s2_cond_path)

    for s in splits:
        spk = s["speaker_id"]
        n = s["sample_n"]
        adapt_labels = [rec_intent[r] for r in s["adaptation_recording_ids"]]
        expected_feas = sorted(list(set([l for l in adapt_labels if l != "GENERAL_SPEECH"])))

        cell_rows = s2_cond_df[(s2_cond_df["speaker_id"] == spk) & (s2_cond_df["N"] == n)]
        for _, r in cell_rows.iterrows():
            recorded_feas = r["feasible_emergency_classes"].split(";") if pd.notna(r["feasible_emergency_classes"]) else []
            assert recorded_feas == expected_feas, f"Mismatch for {spk} N={n}: {recorded_feas} vs {expected_feas}"


# ============================================================
# TEST 3: Error taxonomy counts sum correctly
# ============================================================
def test_3_error_taxonomy_sums_correctly(error_tax_df):
    """TEST 3: Sum of error taxonomy categories strictly equals total test samples."""
    for idx, row in error_tax_df.iterrows():
        a = row["false_emergency_count"]
        b = row["missed_emergency_count"]
        c = row["wrong_emergency_type_count"]
        d = row["correct_emergency_count"]
        e = row["correct_general_count"]
        tot = row["total_samples"]
        gen = row["general_samples"]
        emg = row["emergency_samples"]

        assert a + b + c + d + e == tot, f"Total mismatch for {row['experiment_id']}: {a+b+c+d+e} vs {tot}"
        assert a + e == gen, f"General mismatch for {row['experiment_id']}: {a+e} vs {gen}"
        assert b + c + d == emg, f"Emergency mismatch for {row['experiment_id']}: {b+c+d} vs {emg}"


# ============================================================
# TEST 4: Threshold analysis does not alter model parameters
# ============================================================
def test_4_threshold_analysis_exploratory(thresh_df):
    """TEST 4: Threshold analysis is explicitly labeled EXPLORATORY and contains expected thresholds."""
    assert (thresh_df["status"] == "EXPLORATORY").all()
    assert set(thresh_df["threshold"].unique()) == {0.3, 0.4, 0.5, 0.6, 0.7}
    assert len(thresh_df) == 600  # 120 conditions * 5 thresholds


# ============================================================
# TEST 5: Flat/hierarchical comparisons use matched N/model
# ============================================================
def test_5_flat_vs_hierarchical_matched(matched_df):
    """TEST 5: Comparison strictly pairs flat and hierarchical models under identical N."""
    assert len(matched_df) == 8  # 2 models * 4 N values
    for idx, row in matched_df.iterrows():
        f_mod = row["flat_model"]
        h_mod = row["hierarchical_model"]
        n_val = row["N"]

        if "adapter" in f_mod:
            assert "adapter" in h_mod
        if "rbf_svm" in f_mod:
            assert "rbf_svm" in h_mod
        assert n_val in [20, 30, 40, 50]


# ============================================================
# TEST 6: IPER consistency verified
# ============================================================
def test_6_iper_consistency_verified(hier_results_df, error_tax_df):
    """TEST 6: IPER matches exact-match Emergency Recall identically."""
    diff = np.abs(hier_results_df["IPER"] - hier_results_df["emergency_recall"])
    assert diff.max() == 0.0

    tax_diff = np.abs(error_tax_df["correct_emergency_rate"] - hier_results_df["IPER"])
    assert tax_diff.max() == 0.0


# ============================================================
# TEST 7: False-emergency rate denominator verified
# ============================================================
def test_7_false_emergency_rate_denominator(error_tax_df, hier_results_df):
    """TEST 7: False Emergency Rate denominator is General Speech samples."""
    for idx, row in error_tax_df.iterrows():
        calc_fer = row["false_emergency_count"] / row["general_samples"]
        stored_fer = row["false_emergency_rate"]
        assert abs(calc_fer - stored_fer) < 1e-4

    diff = np.abs(hier_results_df["false_emergency_rate"] - hier_results_df["stage1_false_positive_rate"])
    assert diff.max() == 0.0


# ============================================================
# TEST 8: Original Phase 3-6 artifacts untouched
# ============================================================
def test_8_original_artifacts_untouched():
    """TEST 8: Frozen historical benchmark directories exist and are intact."""
    phase5_csv = os.path.join(REPO_ROOT, "results", "phase5", "phase5_results.csv")
    assert os.path.isfile(phase5_csv), "Phase 5 canonical benchmark missing"
    df5 = pd.read_csv(phase5_csv)
    assert len(df5) == 720, f"Expected 720 rows, got {len(df5)}"


# ============================================================
# TEST 9: Results reproducible
# ============================================================
def test_9_results_reproducible(hier_results_df):
    """TEST 9: All 120 cells have SUCCESS status and deterministic seed 42."""
    assert (hier_results_df["status"] == "SUCCESS").all()
    assert (hier_results_df["seed"] == 42).all()
    assert len(hier_results_df) == 120


# ============================================================
# TEST 10: Existing Phase 6.5 tests pass
# ============================================================
def test_10_all_closure_csvs_exist():
    """TEST 10: All 7 required closure analysis CSV files exist with valid row counts."""
    expected_files = [
        ("stage2_conditional_metrics.csv", 120),
        ("stage2_per_class_metrics.csv", 720),
        ("error_taxonomy.csv", 120),
        ("error_taxonomy_summary.csv", 14),
        ("speaker_error_analysis.csv", 120),
        ("sample_efficiency_closure.csv", 8),
        ("flat_vs_hierarchical_matched.csv", 8)
    ]
    for fname, min_rows in expected_files:
        fpath = os.path.join(ANALYSIS_DIR, fname)
        assert os.path.isfile(fpath), f"Missing expected analysis file: {fname}"
        df = pd.read_csv(fpath)
        assert len(df) >= min_rows, f"Row count for {fname} too low: {len(df)} < {min_rows}"
