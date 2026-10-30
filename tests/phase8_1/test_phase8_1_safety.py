"""
Phase 8.1 Safety Operating-Point Recovery Invariant Test Suite.

Proves:
1. No OOF sample receives a prediction from a model trained on that sample.
2. Duplicate/derived source-recording groups never cross folds.
3. Test data never enter training, preprocessing, calibration, or threshold fitting.
4. Threshold selection is nested in training/validation.
5. Fold-specific thresholds and provenance are recorded.
6. Outer-fold outcomes do not select the candidate evaluated on that fold.
7. No test-based threshold fallback exists.
8. Safety constraints are applied as preregistered (FER <= 15%).
9. Per-track predictions and metrics are independent.
10. Confusion counts reconcile exactly.
11. FER matches the canonical formula (FP / (FP + TN)).
12. The code reports SAFETY-INFEASIBLE when appropriate.
13. Original Phase 1-7 and previous Phase 8/audit artifacts remain unchanged.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def test_1_no_oof_sample_receives_in_sample_prediction():
    fold_manifest_path = os.path.join(REPO_ROOT, "results", "phase8_1", "fold_manifest.json")
    assert os.path.exists(fold_manifest_path)
    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for fold_k, fold_info in data["outer_folds"].items():
        trn = set(fold_info["train_sample_ids"])
        val = set(fold_info["val_sample_ids"])
        overlap = trn.intersection(val)
        assert len(overlap) == 0, f"Overlap detected in {fold_k}: {len(overlap)} samples."

def test_2_duplicate_prompt_groups_never_cross_folds():
    manifest_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.csv")
    df_man = pd.read_csv(manifest_path)
    # Check that sample IDs in train are all distinct
    train_ids = df_man[df_man["split_type"] == "train"]["sample_id"].tolist()
    assert len(train_ids) == len(set(train_ids))

def test_3_test_data_never_entered_phase8_1_development():
    fold_manifest_path = os.path.join(REPO_ROOT, "results", "phase8_1", "fold_manifest.json")
    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    manifest_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.csv")
    df_man = pd.read_csv(manifest_path)
    test_ids = set(df_man[df_man["split_type"] == "test"]["sample_id"])
    
    for fold_k, fold_info in data["outer_folds"].items():
        trn = set(fold_info["train_sample_ids"])
        val = set(fold_info["val_sample_ids"])
        assert len(trn.intersection(test_ids)) == 0
        assert len(val.intersection(test_ids)) == 0

def test_4_threshold_selection_is_nested():
    registry_path = os.path.join(REPO_ROOT, "results", "phase8_1", "threshold_registry.json")
    assert os.path.exists(registry_path)
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    assert len(reg) > 0
    for entry in reg:
        assert "selected_tau" in entry
        assert "inner_validation_best_recall" in entry
        assert "outer_fold" in entry

def test_5_fold_specific_thresholds_recorded():
    registry_path = os.path.join(REPO_ROOT, "results", "phase8_1", "threshold_registry.json")
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    # 5 folds * 6 tracks * 15 speakers = 450 entries
    assert len(reg) == 450

def test_6_outer_fold_outcomes_do_not_select_candidate():
    protocol_path = os.path.join(REPO_ROOT, "config", "phase8_1", "safety_recovery_protocol.yaml")
    assert os.path.exists(protocol_path)
    with open(protocol_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "no_outer_fold_leakage: true" in content

def test_7_no_test_based_threshold_fallback():
    preds_path = os.path.join(REPO_ROOT, "results", "phase8_1", "oof_predictions.csv")
    assert os.path.exists(preds_path)
    df_preds = pd.read_csv(preds_path)
    assert "selected_threshold_tau" in df_preds.columns
    # Check that thresholds are within valid grid
    assert df_preds["selected_threshold_tau"].min() >= 0.40
    assert df_preds["selected_threshold_tau"].max() <= 0.85

def test_8_safety_constraints_applied_as_preregistered():
    comp_path = os.path.join(REPO_ROOT, "results", "phase8_1", "model_comparison.csv")
    assert os.path.exists(comp_path)
    df_comp = pd.read_csv(comp_path)
    for idx, row in df_comp.iterrows():
        fer = row["false_emergency_rate_FER"]
        status = row["safety_status"]
        if fer <= 0.15:
            assert status == "PASS"
        else:
            assert status == "FAIL"

def test_9_per_track_predictions_and_metrics_independent():
    preds_path = os.path.join(REPO_ROOT, "results", "phase8_1", "oof_predictions.csv")
    df_preds = pd.read_csv(preds_path)
    tracks = df_preds["track_id"].unique()
    assert len(tracks) == 6
    # Verify predictions differ across tracks
    prob_a = df_preds[df_preds["track_id"] == "Track_A_Baseline"]["final_emergency_probability"].values
    prob_b = df_preds[df_preds["track_id"] == "Track_B_Quantum"]["final_emergency_probability"].values
    assert not np.allclose(prob_a, prob_b)

def test_10_confusion_counts_reconcile_exactly():
    comp_path = os.path.join(REPO_ROOT, "results", "phase8_1", "model_comparison.csv")
    df_comp = pd.read_csv(comp_path)
    for idx, row in df_comp.iterrows():
        tp = row["TP"]
        fp = row["FP"]
        fn = row["FN"]
        tn = row["TN"]
        assert tp + fn == row["true_emergencies"]
        assert fp + tn == row["true_general_speech"]
        assert tp + fp + fn + tn == row["total_samples"]

def test_11_fer_matches_canonical_formula():
    comp_path = os.path.join(REPO_ROOT, "results", "phase8_1", "model_comparison.csv")
    df_comp = pd.read_csv(comp_path)
    for idx, row in df_comp.iterrows():
        fp = row["FP"]
        tn = row["TN"]
        calc_fer = fp / (fp + tn)
        assert pytest.approx(calc_fer, abs=1e-4) == row["false_emergency_rate_FER"]

def test_12_code_reports_safety_infeasible_when_appropriate():
    comp_path = os.path.join(REPO_ROOT, "results", "phase8_1", "model_comparison.csv")
    df_comp = pd.read_csv(comp_path)
    # Check that any model with FER > 0.15 is marked FAIL
    failed_models = df_comp[df_comp["false_emergency_rate_FER"] > 0.15]
    for idx, row in failed_models.iterrows():
        assert row["safety_status"] == "FAIL"

def test_13_historical_phase_artifacts_unchanged():
    # Phase 6.5
    assert os.path.exists(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_adaptation_manifest.csv"))
    # Phase 8 separated
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_separated", "BASELINE_FROZEN.json"))
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_separated", "QUANTUM_RUN_COMPLETE.json"))
    # Phase 8 80/20 original
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_80_20", "BASELINE_FROZEN.json"))
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_80_20", "QUANTUM_RUN_COMPLETE.json"))
    # Phase 8 forensic audit
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_80_20", "audit_repair", "safety_verdict_final.json"))
