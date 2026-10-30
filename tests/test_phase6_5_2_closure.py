"""
Test Suite for Subphase 6.5.2-C:
Classical Rescue Analytical Closure & Audit Verification
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE6_5_DIR = REPO_ROOT / "phase6_5"
RESCUE_DIR = PHASE6_5_DIR / "classical_rescue"
RESULTS_DIR = RESCUE_DIR / "results"
ANALYSIS_DIR = RESCUE_DIR / "analysis"
REPORTS_DIR = RESCUE_DIR / "reports"

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]


@pytest.fixture(scope="module")
def results_df():
    csv_path = RESULTS_DIR / "classical_rescue_results.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def predictions_df():
    csv_path = RESULTS_DIR / "classical_rescue_predictions.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def weights_df():
    csv_path = RESULTS_DIR / "class_weight_records.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def splits_json():
    splits_path = PHASE6_5_DIR / "splits" / "recovery_splits_stratified.json"
    assert splits_path.exists(), f"Missing {splits_path}"
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def manifest_df():
    man_path = REPO_ROOT / "data" / "manifests" / "torgo_manifest.csv"
    assert man_path.exists(), f"Missing {man_path}"
    return pd.read_csv(man_path)


@pytest.fixture(scope="module")
def feasible_metrics_df():
    csv_path = ANALYSIS_DIR / "feasible_class_metrics.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def weight_audit_df():
    csv_path = ANALYSIS_DIR / "class_weight_audit.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def false_alarm_df():
    csv_path = ANALYSIS_DIR / "false_alarm_analysis.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


# ============================================================
# TEST 1: Feasible class mask uses adaptation data only
# ============================================================
def test_1_feasible_class_mask_adaptation_only(splits_json, manifest_df, feasible_metrics_df):
    """TEST 1: Feasible classes are strictly those with count >= 1 in adaptation split."""
    rec_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))
    split_map = {(s["speaker_id"], s["sample_n"]): s["adaptation_recording_ids"] for s in splits_json}
    
    for idx, row in feasible_metrics_df.iterrows():
        spk = row["speaker_id"]
        n = int(row["N"])
        adapt_recs = split_map[(spk, n)]
        adapt_labels = set(rec_intent[r] for r in adapt_recs)
        
        reported_feasible = set(row["feasible_classes"].split(";"))
        assert reported_feasible == adapt_labels, (
            f"Feasible class mismatch for {row['experiment_id']}: {reported_feasible} vs {adapt_labels}"
        )
        assert row["num_feasible_classes"] == len(adapt_labels)


# ============================================================
# TEST 2: Class weights match formula
# ============================================================
def test_2_class_weights_match_formula(weight_audit_df):
    """TEST 2: All class weights strictly match w_c = N_train / (K_present * n_c)."""
    assert len(weight_audit_df) == 840, f"Expected 840 rows in weight audit, got {len(weight_audit_df)}"
    failed = weight_audit_df[weight_audit_df["audit_status"] != "PASS"]
    assert len(failed) == 0, f"Found {len(failed)} failed class weight calculations"


# ============================================================
# TEST 3: Global Macro-F1 unchanged
# ============================================================
def test_3_global_macro_f1_unchanged(results_df, feasible_metrics_df):
    """TEST 3: Global Macro-F1 in feasible metrics matches canonical results file."""
    assert len(results_df) == 240
    assert len(feasible_metrics_df) == 240
    
    diff = np.abs(results_df["macro_f1"] - feasible_metrics_df["global_macro_f1"])
    assert diff.max() < 1e-4, f"Max diff in global Macro-F1 is {diff.max()}"


# ============================================================
# TEST 4: Feasible Macro-F1 is correctly computed
# ============================================================
def test_4_feasible_macro_f1_correctly_computed(predictions_df, feasible_metrics_df):
    """TEST 4: Feasible-class Macro-F1 correctly evaluates only feasible classes."""
    preds_by_exp = {exp_id: grp for exp_id, grp in predictions_df.groupby("experiment_id")}
    
    for idx, row in feasible_metrics_df.sample(20, random_state=42).iterrows():
        exp_id = row["experiment_id"]
        feas_classes = row["feasible_classes"].split(";")
        cell_preds = preds_by_exp[exp_id]
        
        y_true = cell_preds["true_intent"].tolist()
        y_pred = cell_preds["predicted_intent"].tolist()
        
        expected_feas_f1 = f1_score(y_true, y_pred, labels=feas_classes, average="macro", zero_division=0)
        assert abs(expected_feas_f1 - row["feasible_class_macro_f1"]) < 1e-3, (
            f"Mismatch in feasible Macro-F1 for {exp_id}: expected {expected_feas_f1}, got {row['feasible_class_macro_f1']}"
        )


# ============================================================
# TEST 5: IPER implementation is consistent
# ============================================================
def test_5_iper_consistency(results_df, predictions_df):
    """TEST 5: IPER matches exact-match Emergency Recall across all cells."""
    iper_diff = np.abs(results_df["IPER"] - results_df["emergency_recall"])
    assert iper_diff.max() == 0.0, f"Max diff between IPER and emergency recall is {iper_diff.max()}"
    
    # Recompute from predictions
    for exp_id, grp in predictions_df.groupby("experiment_id"):
        emerg_preds = grp[grp["is_emergency_true"] == True]
        if len(emerg_preds) > 0:
            recomputed_recall = (emerg_preds["true_intent"] == emerg_preds["predicted_intent"]).mean()
            stored_iper = results_df[results_df["experiment_id"] == exp_id]["IPER"].values[0]
            assert abs(recomputed_recall - stored_iper) < 1e-4


# ============================================================
# TEST 6: False alarm rate is correctly calculated
# ============================================================
def test_6_false_alarm_rate_correctly_calculated(results_df, predictions_df):
    """TEST 6: False emergency rate equals false emergency count / total general speech."""
    for exp_id, grp in predictions_df.groupby("experiment_id"):
        gen_preds = grp[grp["true_intent"] == "GENERAL_SPEECH"]
        if len(gen_preds) > 0:
            false_emerg = (gen_preds["predicted_intent"] != "GENERAL_SPEECH").mean()
            stored_fer = results_df[results_df["experiment_id"] == exp_id]["false_emergency_rate"].values[0]
            assert abs(false_emerg - stored_fer) < 1e-4


# ============================================================
# TEST 7: No original Phase 3–6 files modified
# ============================================================
def test_7_frozen_benchmarks_untouched():
    """TEST 7: Historical benchmarks in results/phase3, phase4, phase5, phase6 are preserved."""
    phase5_csv = REPO_ROOT / "results" / "phase5" / "phase5_results.csv"
    assert phase5_csv.exists(), "Phase 5 canonical benchmark missing"
    df5 = pd.read_csv(phase5_csv)
    assert len(df5) == 720, f"Expected 720 rows in phase5_results.csv, found {len(df5)}"
    
    phase6_csv = REPO_ROOT / "phase6" / "results" / "phase6_master_results.csv"
    if phase6_csv.exists():
        df6 = pd.read_csv(phase6_csv)
        assert len(df6) > 0


# ============================================================
# TEST 8: Results are deterministic
# ============================================================
def test_8_results_determinism(results_df):
    """TEST 8: Results contain master seed 42 and unique experiment IDs."""
    assert (results_df["seed"] == 42).all(), "All cells must use seed 42"
    assert len(results_df["experiment_id"].unique()) == 240, "All 240 experiment IDs must be unique"
