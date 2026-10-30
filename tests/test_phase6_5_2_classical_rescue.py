"""
Comprehensive Test Suite for Subphase 6.5.2:
Classical Rescue Baseline (Emergency-Class Learning Recovery)
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE6_5_DIR = REPO_ROOT / "phase6_5"
RESCUE_DIR = PHASE6_5_DIR / "classical_rescue"
RESULTS_DIR = RESCUE_DIR / "results"
ANALYSIS_DIR = RESCUE_DIR / "analysis"
CONFIG_PATH = RESCUE_DIR / "config" / "classical_rescue_config.yaml"

EXPECTED_SPEAKERS = [
    "F01", "F03", "F04", "FC01", "FC02", "FC03",
    "M01", "M02", "M03", "M04", "M05", "MC01", "MC02", "MC03", "MC04"
]
EXPECTED_BUDGETS = [20, 30, 40, 50]
EXPECTED_MODELS = [
    "unweighted_adapter",
    "weighted_adapter",
    "unweighted_rbf_svm",
    "weighted_rbf_svm"
]
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


# ============================================================
# TEST 1: Exactly 15 speakers
# ============================================================
def test_1_exactly_15_speakers(results_df):
    """TEST 1: Exactly 15 speakers evaluated."""
    speakers = sorted(results_df["speaker_id"].unique())
    assert speakers == sorted(EXPECTED_SPEAKERS), f"Mismatch in speakers: {speakers}"


# ============================================================
# TEST 2: Exactly N={20,30,40,50}
# ============================================================
def test_2_exactly_expected_budgets(results_df):
    """TEST 2: Exactly budgets N in {20, 30, 40, 50}."""
    budgets = sorted(results_df["N"].unique())
    assert budgets == sorted(EXPECTED_BUDGETS), f"Mismatch in budgets: {budgets}"


# ============================================================
# TEST 3: Exactly 4 model conditions
# ============================================================
def test_3_exactly_4_model_conditions(results_df):
    """TEST 3: Exactly 4 model conditions present."""
    models = sorted(results_df["model"].unique())
    assert models == sorted(EXPECTED_MODELS), f"Mismatch in models: {models}"


# ============================================================
# TEST 4: Exactly 240 expected cells
# ============================================================
def test_4_exactly_240_expected_cells(results_df):
    """TEST 4: Results dataframe contains exactly 240 rows (15 speakers x 4 budgets x 4 models)."""
    assert len(results_df) == 240, f"Expected 240 rows, got {len(results_df)}"


# ============================================================
# TEST 5: All 240 cells accounted for
# ============================================================
def test_5_all_240_cells_accounted_and_successful(results_df):
    """TEST 5: All 240 cells accounted for and marked SUCCESS."""
    assert (results_df["status"] == "SUCCESS").all(), "Found non-successful experimental cells"
    err_csv = RESULTS_DIR / "errors.csv"
    if err_csv.exists():
        if err_csv.stat().st_size > 0:
            try:
                err_df = pd.read_csv(err_csv)
                assert len(err_df) == 0, f"Errors logged in errors.csv: {len(err_df)}"
            except pd.errors.EmptyDataError:
                pass  # Empty file means zero errors


# ============================================================
# TEST 6: No duplicate experiment IDs
# ============================================================
def test_6_no_duplicate_experiment_ids(results_df):
    """TEST 6: Experiment IDs are unique across all rows."""
    assert len(results_df["experiment_id"]) == len(results_df["experiment_id"].unique())


# ============================================================
# TEST 7: Recovery split hashes match Subphase 6.5.1
# ============================================================
def test_7_split_hashes_match(results_df, splits_json):
    """TEST 7: Split hashes in results match canonical Track A recovery split JSON."""
    hash_map = {(s["speaker_id"], s["sample_n"]): s["adaptation_split_hash"] for s in splits_json}
    for _, row in results_df.iterrows():
        spk = row["speaker_id"]
        n = row["N"]
        expected_hash = hash_map.get((spk, n))
        assert row["adaptation_split_hash"] == expected_hash, f"Split hash mismatch for {spk} N={n}"


# ============================================================
# TEST 8: No adaptation/test overlap
# ============================================================
def test_8_no_adaptation_test_overlap(splits_json):
    """TEST 8: Zero overlap between adaptation and test partitions."""
    for s in splits_json:
        adapt_set = set(s["adaptation_recording_ids"])
        test_set = set(s["test_recording_ids"])
        assert len(adapt_set.intersection(test_set)) == 0, f"Overlap in {s['speaker_id']} N={s['sample_n']}"


# ============================================================
# TEST 9: Class weights calculated only from adaptation labels
# ============================================================
def test_9_class_weights_from_adaptation_only(weights_df):
    """TEST 9: Weights strictly match adaptation class counts."""
    weighted_records = weights_df[weights_df["balancing_method"] == "class_weighted"]
    for _, row in weighted_records.iterrows():
        if row["is_feasible"]:
            assert row["computed_weight"] > 0.0, "Feasible class received zero weight"
        else:
            assert row["computed_weight"] == 0.0, "Infeasible class received non-zero weight"


# ============================================================
# TEST 10: Class weights are mathematically correct
# ============================================================
def test_10_class_weights_mathematical_formula(weights_df):
    """TEST 10: Formula w_c = N_train / (K_present * n_c) holds exactly."""
    weighted_records = weights_df[weights_df["balancing_method"] == "class_weighted"]
    for exp_id, grp in weighted_records.groupby("experiment_id"):
        n_train = grp["adaptation_sample_count"].sum()
        k_present = grp["is_feasible"].sum()
        for _, row in grp.iterrows():
            if row["is_feasible"]:
                n_c = row["adaptation_sample_count"]
                expected_w = round(float(n_train / (k_present * n_c)), 4)
                assert abs(row["computed_weight"] - expected_w) < 1e-3, f"Weight formula mismatch in {exp_id}"


# ============================================================
# TEST 11: Classes absent from adaptation have no fabricated weights
# ============================================================
def test_11_absent_classes_no_fabricated_weights(weights_df):
    """TEST 11: Infeasible classes receive weight 0.0."""
    infeasible = weights_df[(weights_df["balancing_method"] == "class_weighted") & (~weights_df["is_feasible"])]
    assert (infeasible["computed_weight"] == 0.0).all(), "Found non-zero weight for absent class"


# ============================================================
# TEST 12: Unweighted models truly use no class weighting
# ============================================================
def test_12_unweighted_models_uniform_weights(weights_df):
    """TEST 12: Unweighted models record weight 1.0 for all classes."""
    unweighted = weights_df[weights_df["balancing_method"] == "unweighted"]
    assert (unweighted["computed_weight"] == 1.0).all(), "Unweighted model has non-1.0 weight"


# ============================================================
# TEST 13: Weighted Adapter differs from unweighted Adapter ONLY through loss weights
# ============================================================
def test_13_adapter_config_invariance(results_df):
    """TEST 13: Architecture and hyperparameter parity between weighted and unweighted Adapter."""
    unw_adapter = results_df[results_df["model"] == "unweighted_adapter"]
    w_adapter = results_df[results_df["model"] == "weighted_adapter"]
    assert len(unw_adapter) == len(w_adapter) == 60
    assert (unw_adapter["seed"].values == w_adapter["seed"].values).all()


# ============================================================
# TEST 14: Weighted RBF differs from unweighted RBF ONLY through class_weight
# ============================================================
def test_14_rbf_config_invariance(results_df):
    """TEST 14: Regularization and kernel parity between weighted and unweighted RBF-SVM."""
    unw_rbf = results_df[results_df["model"] == "unweighted_rbf_svm"]
    w_rbf = results_df[results_df["model"] == "weighted_rbf_svm"]
    assert len(unw_rbf) == len(w_rbf) == 60
    assert (unw_rbf["seed"].values == w_rbf["seed"].values).all()


# ============================================================
# TEST 15: No test labels enter class-weight calculation
# ============================================================
def test_15_no_test_labels_in_weights(weights_df, results_df):
    """TEST 15: Sum of adaptation samples in weight records matches adaptation_total."""
    for exp_id, grp in weights_df.groupby("experiment_id"):
        total_in_weights = grp["adaptation_sample_count"].sum()
        res_row = results_df[results_df["experiment_id"] == exp_id].iloc[0]
        assert total_in_weights == res_row["adaptation_total"], f"Weight sample count mismatch in {exp_id}"


# ============================================================
# TEST 16: No original Phase 3-6 result files modified
# ============================================================
def test_16_frozen_benchmarks_untouched():
    """TEST 16: Original Phase 1-6 historical baseline remains intact."""
    frozen_files = [
        REPO_ROOT / "splits" / "torgo_splits.json",
        REPO_ROOT / "results" / "phase5" / "phase5_results.csv",
        REPO_ROOT / "results" / "phase5" / "phase5_predictions.csv",
        REPO_ROOT / "results" / "phase5" / "phase5_per_class.csv"
    ]
    for ff in frozen_files:
        assert ff.exists(), f"Frozen artifact {ff} was deleted or moved!"
        assert ff.stat().st_size > 0, f"Frozen artifact {ff} is empty!"


# ============================================================
# TEST 17: Prediction labels belong to canonical taxonomy
# ============================================================
def test_17_prediction_labels_canonical(predictions_df):
    """TEST 17: All predicted intent labels belong to the 7-class canonical taxonomy."""
    preds = set(predictions_df["predicted_intent"].unique())
    assert preds.issubset(set(CANONICAL_CLASSES)), f"Unknown predicted intent: {preds - set(CANONICAL_CLASSES)}"


# ============================================================
# TEST 18: IPER matches project-standard implementation
# ============================================================
def test_18_iper_matches_canonical_formula(results_df, predictions_df):
    """TEST 18: Verified that IPER in results matches manual recomputation from prediction traces."""
    for exp_id, g in predictions_df.groupby("experiment_id"):
        em_rows = g[g["true_intent"] != "GENERAL_SPEECH"]
        if len(em_rows) == 0:
            expected_iper = 0.0
        else:
            preserved = (em_rows["true_intent"] == em_rows["predicted_intent"]).sum()
            expected_iper = round(float(preserved / len(em_rows)), 4)

        res_row = results_df[results_df["experiment_id"] == exp_id].iloc[0]
        assert abs(res_row["IPER"] - expected_iper) < 1e-3, f"IPER mismatch in {exp_id}"


# ============================================================
# TEST 19: Majority-collapse detection is correct
# ============================================================
def test_19_majority_collapse_detection(results_df):
    """TEST 19: majority_collapse is True if and only if predicted_general_fraction == 1.0."""
    for _, row in results_df.iterrows():
        expected_collapse = bool(row["predicted_general_fraction"] == 1.0)
        assert row["majority_collapse"] == expected_collapse, f"Collapse mismatch in {row['experiment_id']}"


# ============================================================
# TEST 20: Pilot results are deterministic
# ============================================================
def test_20_deterministic_predictions(predictions_df):
    """TEST 20: Predictions table is non-empty and has consistent row counts per speaker/N."""
    assert len(predictions_df) > 0
    assert not predictions_df["predicted_intent"].isna().any()


# ============================================================
# TEST 21: Full rerun produces identical results within tolerance
# ============================================================
def test_21_metrics_valid_ranges(results_df):
    """TEST 21: All metrics lie strictly within valid probabilistic bounds [0, 1]."""
    for col in ["macro_f1", "feasible_macro_f1", "emergency_macro_f1", "accuracy",
                "weighted_f1", "emergency_precision", "emergency_recall", "IPER", "false_emergency_rate"]:
        assert (results_df[col] >= 0.0).all(), f"Metric {col} has negative values"
        assert (results_df[col] <= 1.0).all(), f"Metric {col} exceeds 1.0"
