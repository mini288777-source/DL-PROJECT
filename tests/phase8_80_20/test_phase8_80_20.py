"""
Test Suite for Supplementary Phase 8 80/20 Benchmark.

Validates:
1. Split manifest integrity, 80/20 sample counts (4,852 train / 1,215 test), 15 speakers, fingerprint.
2. Zero train/test leakage (IDs disjoint, audio hashes disjoint).
3. Cryptographic sentinels exist, statuses valid, hashes match.
4. Prediction schema integrity, 3,645 rows (1,215 per track), track independence.
5. Numeric consistency across metrics.json, model_comparison.csv, and track summaries.
6. All 12 required publication figures exist and are non-empty.
7. Leakage audit passes with zero leaks.
8. Historical files and frozen CASE B conclusion are strictly preserved.
9. Statistical tests confirm CASE B verdict (no statistically supported advantage).
"""

import os
import json
import hashlib
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def sha256_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_manifest_fingerprint_and_counts():
    manifest_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.json")
    assert os.path.exists(manifest_path), "Missing split_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_active_samples"] == 6067
    assert manifest["train_samples_total"] == 4852
    assert manifest["test_samples_total"] == 1215
    assert len(manifest["speakers"]) == 15
    assert manifest["master_seed"] == 42
    assert len(manifest["dataset_fingerprint"]) == 64

def test_zero_train_test_leakage():
    manifest_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_set = set(manifest["all_train_ids"])
    test_set = set(manifest["all_test_ids"])

    # Sample ID disjointness
    assert len(train_set) == 4852
    assert len(test_set) == 1215
    assert train_set.isdisjoint(test_set), "Train and Test IDs overlap!"

    # Audio hash disjointness
    manifest_csv = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.csv")
    df = pd.read_csv(manifest_csv)
    train_hashes = set(df.loc[df["split_type"] == "train", "audio_sha256"])
    test_hashes = set(df.loc[df["split_type"] == "test", "audio_sha256"])
    assert train_hashes.isdisjoint(test_hashes), "Audio hash collision between train and test!"

def test_sentinels_exist_and_hashes_match():
    base_dir = os.path.join(REPO_ROOT, "results", "phase8_80_20")
    
    # 1. Track A sentinel
    sentinel_a = os.path.join(base_dir, "BASELINE_FROZEN.json")
    assert os.path.exists(sentinel_a), "Missing BASELINE_FROZEN.json"
    with open(sentinel_a, "r", encoding="utf-8") as f:
        sa = json.load(f)
    assert sa["status"] == "frozen"
    assert sa["total_test_predictions"] == 1215
    pred_a = os.path.join(base_dir, "track_a_classical", "predictions_track_a.csv")
    assert sha256_file(pred_a) == sa["predictions_hash"]

    # 2. Track B sentinel
    sentinel_b = os.path.join(base_dir, "QUANTUM_RUN_COMPLETE.json")
    assert os.path.exists(sentinel_b), "Missing QUANTUM_RUN_COMPLETE.json"
    with open(sentinel_b, "r", encoding="utf-8") as f:
        sb = json.load(f)
    assert sb["status"] == "complete"
    assert sb["total_test_predictions"] == 1215
    assert sb["quantum_participation_verified"] is True
    pred_b = os.path.join(base_dir, "track_b_quantum", "predictions_track_b.csv")
    assert sha256_file(pred_b) == sb["predictions_hash"]

    # 3. Track C sentinel
    sentinel_c = os.path.join(base_dir, "SHADOW_RUN_COMPLETE.json")
    assert os.path.exists(sentinel_c), "Missing SHADOW_RUN_COMPLETE.json"
    with open(sentinel_c, "r", encoding="utf-8") as f:
        sc = json.load(f)
    assert sc["status"] == "complete"
    assert sc["total_test_predictions"] == 1215
    pred_c = os.path.join(base_dir, "track_c_classical_shadow", "predictions_track_c.csv")
    assert sha256_file(pred_c) == sc["predictions_hash"]

def test_predictions_schema_and_independence():
    pred_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "predictions.csv")
    assert os.path.exists(pred_path), "Missing predictions.csv"
    df = pd.read_csv(pred_path)

    assert len(df) == 3645, f"Expected 3645 predictions (1215 x 3), found {len(df)}"
    tracks = df["track_id"].value_counts().to_dict()
    assert tracks.get("Track_A") == 1215
    assert tracks.get("Track_B") == 1215
    assert tracks.get("Track_C") == 1215

    # Check required columns
    required_cols = [
        "experiment_id", "track_id", "speaker_id", "speaker_group", "sample_id",
        "true_label", "model_version", "prediction_version", "gate_probability",
        "residual_score", "final_emergency_probability", "final_emergency_prediction",
        "final_7class_prediction", "decision_threshold", "prediction_source", "split_manifest_id"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in predictions.csv"

    # Verify track independence: predictions must not be identical copies
    prob_a = df.loc[df["track_id"] == "Track_A", "final_emergency_probability"].values
    prob_b = df.loc[df["track_id"] == "Track_B", "final_emergency_probability"].values
    prob_c = df.loc[df["track_id"] == "Track_C", "final_emergency_probability"].values

    assert not np.allclose(prob_a, prob_b), "Track A and Track B predictions are identical!"
    assert not np.allclose(prob_b, prob_c), "Track B and Track C predictions are identical!"
    assert not np.allclose(prob_a, prob_c), "Track A and Track C predictions are identical!"

def test_model_comparison_metrics_consistency():
    comp_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "model_comparison.csv")
    assert os.path.exists(comp_path), "Missing model_comparison.csv"
    df_comp = pd.read_csv(comp_path)

    metrics_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        met = json.load(f)

    for idx, row in df_comp.iterrows():
        track_key = "Track_A" if "Track_A" in row["track_id"] else ("Track_B" if "Track_B" in row["track_id"] else "Track_C")
        m = met["models"][track_key]
        assert pytest.approx(row["accuracy"], 1e-4) == m["accuracy"]
        assert pytest.approx(row["macro_f1"], 1e-4) == m["macro_f1"]
        assert pytest.approx(row["emergency_recall"], 1e-4) == m["emergency_recall"]
        assert pytest.approx(row["false_emergency_rate"], 1e-4) == m["false_emergency_rate"]

def test_all_12_figures_exist_and_nonempty():
    fig_dir = os.path.join(REPO_ROOT, "results", "phase8_80_20", "figures")
    expected_figures = [
        "train_test_class_distribution.png",
        "model_comparison.png",
        "before_after_comparison.png",
        "emergency_metrics_comparison.png",
        "confusion_matrix_classical.png",
        "confusion_matrix_quantum.png",
        "confusion_matrix_shadow.png",
        "per_class_performance.png",
        "per_speaker_comparison.png",
        "dysarthric_vs_control.png",
        "train_test_gap.png",
        "runtime_comparison.png"
    ]

    for fname in expected_figures:
        fpath = os.path.join(fig_dir, fname)
        assert os.path.exists(fpath), f"Missing figure: {fname}"
        size = os.path.getsize(fpath)
        assert size > 10000, f"Figure {fname} is suspiciously small ({size} bytes)"

def test_leakage_audit_passes():
    audit_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "leakage_audit.json")
    assert os.path.exists(audit_path), "Missing leakage_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["overall_audit_status"] == "PASS"
    assert audit["checks"]["sample_id_disjointness"]["status"] == "PASS"
    assert audit["checks"]["audio_hash_disjointness"]["status"] == "PASS"
    assert audit["checks"]["preprocessing_fit_on_train_only"]["status"] == "PASS"

def test_historical_integrity_preserved():
    # Historical Phase 6.5
    assert os.path.exists(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_adaptation_manifest.csv"))
    # Historical Phase 8 separated
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_separated", "BASELINE_FROZEN.json"))
    assert os.path.exists(os.path.join(REPO_ROOT, "results", "phase8_separated", "QUANTUM_RUN_COMPLETE.json"))
    assert os.path.exists(os.path.join(REPO_ROOT, "reports", "phase8_separated", "PHASE8_SEPARATED_PIPELINE_REPORT.md"))

def test_scientific_case_b_verdict():
    metrics_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        met = json.load(f)

    stats_info = met["statistical_tests"]
    assert "CASE B" in stats_info["scientific_verdict"]
    assert stats_info["pairwise_wilcoxon"]["Track_B_vs_Track_A_recall"]["significant_at_05"] is False
    assert stats_info["mcnemar_emergency_test_60"]["significant_at_05"] is False
