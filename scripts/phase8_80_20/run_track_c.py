"""
Track C: Matched Classical Shadow Independent Pipeline Execution for Phase 8 80/20 Benchmark.

Strict Protocol Guarantees:
- Enforces prerequisite: requires valid QUANTUM_RUN_COMPLETE.json sentinel before execution.
- Isolated Track C loader and preprocessing (PCA 384->8) fitted strictly on Track C training IDs.
- Deterministic random seed 42.
- 8-D input capacity-matched MLP residual module.
- Evaluates strictly on locked 20% test IDs (1,215 samples).
- Emits predictions with full provenance.
- Outputs written to results/phase8_80_20/track_c_classical_shadow/ and results/phase8_80_20/.
- Writes SHADOW_RUN_COMPLETE.json sentinel upon verified completion.
"""

import sys
import os
import time
import json
import hashlib
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.phase8_separated.phase8_separated_models import (
    RepairedTrackCShadow,
    compute_repaired_phase8_metrics,
)
from src.embeddings.cache import load_embedding_file
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS

def sha256_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def evaluate_safety_gate(metrics: Dict[str, Any]) -> tuple[bool, str]:
    fer = metrics["false_emergency_rate"]
    rec = metrics["emergency_recall"]
    if fer > 0.15:
        return False, f"FER_SPIKE_VETO (FER={fer:.3f} > 0.15)"
    if rec < 0.05:
        return False, f"RECALL_COLLAPSE_VETO (Recall={rec:.3f} < 0.05)"
    return True, "PASS"

def main():
    print("=" * 80)
    print("PHASE 8 80/20 BENCHMARK — TRACK C: MATCHED CLASSICAL SHADOW")
    print("=" * 80)

    results_dir = os.path.join(REPO_ROOT, "results", "phase8_80_20")
    track_dir = os.path.join(results_dir, "track_c_classical_shadow")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)

    # 1. Enforce sequential execution order & sentinel verification
    sentinel_b_path = os.path.join(results_dir, "QUANTUM_RUN_COMPLETE.json")
    if not os.path.exists(sentinel_b_path):
        print(f"ERROR: Track B must complete and freeze before Track C starts. Missing: {sentinel_b_path}")
        sys.exit(1)

    with open(sentinel_b_path, "r", encoding="utf-8") as f:
        sentinel_b = json.load(f)

    if sentinel_b.get("status") != "complete":
        print(f"ERROR: Invalid Track B sentinel status: {sentinel_b.get('status')}")
        sys.exit(1)

    pred_b_path = os.path.join(results_dir, "track_b_quantum", "predictions_track_b.csv")
    if sha256_file(pred_b_path) != sentinel_b.get("predictions_hash"):
        print("ERROR: Track B predictions hash mismatch! Historical integrity violated.")
        sys.exit(1)
    print(f"Verified Track B quantum sentinel (hash={sentinel_b.get('predictions_hash')[:12]}...). Proceeding with Track C.")

    # 2. Load locked split manifest
    manifest_json = os.path.join(results_dir, "split_manifest.json")
    manifest_csv = os.path.join(results_dir, "split_manifest.csv")
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    df_manifest = pd.read_csv(manifest_csv)
    label_map = dict(zip(df_manifest["sample_id"], df_manifest["intent"]))

    # 3. Load Whisper embeddings
    base_torgo_dir = os.path.join(REPO_ROOT, "embeddings", "torgo")
    needed_rids = set(manifest["all_train_ids"] + manifest["all_test_ids"])
    embedding_cache: Dict[str, np.ndarray] = {}
    print(f"Loading Whisper embeddings for {len(needed_rids)} samples...")
    t_load = time.perf_counter()
    for rid in sorted(list(needed_rids)):
        embedding_cache[rid] = load_embedding_file(rid, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=384)
    print(f"Loaded {len(embedding_cache)} embeddings in {time.perf_counter() - t_load:.2f}s.")

    all_speaker_results = []
    all_predictions = []

    t_start = time.perf_counter()

    for idx, spk_info in enumerate(manifest["speakers"]):
        spk = spk_info["speaker_id"]
        spk_group = spk_info["speaker_group"]
        train_ids = spk_info["train_ids"]
        test_ids = spk_info["test_ids"]

        X_train = np.array([embedding_cache[r] for r in train_ids], dtype=np.float32)
        y_train = np.array([label_map[r] for r in train_ids])
        X_test = np.array([embedding_cache[r] for r in test_ids], dtype=np.float32)
        y_test = np.array([label_map[r] for r in test_ids])

        # Verify zero leakage
        assert set(train_ids).isdisjoint(set(test_ids)), f"Leakage detected in speaker {spk}!"

        # Train Track C Matched Classical Shadow model independently
        t0 = time.perf_counter()
        model = RepairedTrackCShadow(alpha=0.5, random_seed=42)
        model.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0

        # Predict detailed on locked test set
        t0 = time.perf_counter()
        pred_7cl, is_em, gate_probs, z_res, final_probs = model.predict_detailed(X_test)
        t_eval = time.perf_counter() - t0

        # Speaker level metrics
        met = compute_repaired_phase8_metrics(y_test, pred_7cl, is_em, track_id="Track_C_Classical_Shadow")
        gate_pass, gate_msg = evaluate_safety_gate(met)
        spk_acc = float(accuracy_score(y_test, pred_7cl))
        intent_pres_rate = float(np.mean(pred_7cl[y_test == GENERAL_SPEECH_CLASS] == GENERAL_SPEECH_CLASS)) if sum(y_test == GENERAL_SPEECH_CLASS) > 0 else 1.0

        all_speaker_results.append({
            "speaker_id": spk,
            "speaker_group": spk_group,
            "track": "Track_C_Classical_Shadow",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "emergency_test_samples": met["true_emergencies"],
            "emergency_recall": met["emergency_recall"],
            "emergency_precision": met["emergency_precision"],
            "false_emergency_rate": met["false_emergency_rate"],
            "accuracy": spk_acc,
            "macro_f1": met["macro_f1"],
            "intent_preservation_rate": intent_pres_rate,
            "locked_threshold": float(model.locked_threshold),
            "train_time_sec": t_fit,
            "eval_time_sec": t_eval,
            "safety_gate_passed": gate_pass,
            "safety_gate_status": gate_msg
        })

        for i, rid in enumerate(test_ids):
            all_predictions.append({
                "experiment_id": "phase8_80_20",
                "track_id": "Track_C",
                "speaker_id": spk,
                "speaker_group": spk_group,
                "sample_id": rid,
                "true_label": y_test[i],
                "model_version": "1.0-shadow",
                "prediction_version": "1.0-80_20",
                "gate_probability": float(gate_probs[i]),
                "residual_score": float(z_res[i]),
                "final_emergency_probability": float(final_probs[i]),
                "final_emergency_prediction": bool(is_em[i]),
                "final_7class_prediction": pred_7cl[i],
                "decision_threshold": float(model.locked_threshold),
                "prediction_source": "track_c_classical_shadow",
                "split_manifest_id": f"PH8_80_20_{spk}"
            })

        print(f"[{idx+1:02d}/15] Spk: {spk:4s} ({spk_group:10s}) | Train: {len(X_train):3d} | Test: {len(X_test):3d} | "
              f"Rec: {met['emergency_recall']:.3f} | Prec: {met['emergency_precision']:.3f} | "
              f"FER: {met['false_emergency_rate']:.3f} | F1: {met['macro_f1']:.3f} | Thresh: {model.locked_threshold:.2f} | [{gate_msg}]", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nTrack C training & evaluation completed in {elapsed:.2f}s across 15 speakers.")

    # Convert to DataFrames
    df_spk = pd.DataFrame(all_speaker_results)
    df_pred = pd.DataFrame(all_predictions)

    # Compute Global Benchmark Metrics across all 1,215 locked test predictions
    y_true_all = df_pred["true_label"].values
    y_pred_7cl_all = df_pred["final_7class_prediction"].values
    is_em_pred_all = df_pred["final_emergency_prediction"].values.astype(bool)

    global_met = compute_repaired_phase8_metrics(y_true_all, y_pred_7cl_all, is_em_pred_all, track_id="Track_C_Classical_Shadow")
    global_acc = float(accuracy_score(y_true_all, y_pred_7cl_all))
    global_fer = global_met["false_emergency_rate"]
    global_intent_pres = float(np.mean(y_pred_7cl_all[y_true_all == GENERAL_SPEECH_CLASS] == GENERAL_SPEECH_CLASS))

    # Dysarthric vs Control aggregates
    dys_mask = df_pred["speaker_group"] == "dysarthric"
    ctrl_mask = df_pred["speaker_group"] == "control"

    dys_met = compute_repaired_phase8_metrics(y_true_all[dys_mask], y_pred_7cl_all[dys_mask], is_em_pred_all[dys_mask], track_id="Track_C_Dysarthric")
    dys_acc = float(accuracy_score(y_true_all[dys_mask], y_pred_7cl_all[dys_mask]))
    ctrl_met = compute_repaired_phase8_metrics(y_true_all[ctrl_mask], y_pred_7cl_all[ctrl_mask], is_em_pred_all[ctrl_mask], track_id="Track_C_Control")
    ctrl_acc = float(accuracy_score(y_true_all[ctrl_mask], y_pred_7cl_all[ctrl_mask]))

    # Per-class metrics
    class_report = classification_report(y_true_all, y_pred_7cl_all, labels=list(CANONICAL_CLASSES), output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_7cl_all, labels=list(CANONICAL_CLASSES)).tolist()

    # Save CSV and JSON artifacts
    res_spk_csv = os.path.join(track_dir, "per_speaker_track_c.csv")
    df_spk.to_csv(res_spk_csv, index=False)

    pred_csv = os.path.join(track_dir, "predictions_track_c.csv")
    df_pred.to_csv(pred_csv, index=False)

    global_summary = {
        "track_id": "Track_C_Classical_Shadow",
        "benchmark": "Phase8_80_20",
        "shadow_configuration": {
            "input_dimension": 8,
            "hidden_dimension": 16,
            "activation": "GELU",
            "trainable_parameters": 161, # (8*16 + 16) + (16*1 + 1)
            "matched_to": "Track B 8-Qubit VQC"
        },
        "total_train_samples": sum(df_spk["train_samples"]),
        "total_test_samples": len(df_pred),
        "total_emergency_test_samples": global_met["true_emergencies"],
        "accuracy": global_acc,
        "macro_f1": global_met["macro_f1"],
        "emergency_recall": global_met["emergency_recall"],
        "emergency_precision": global_met["emergency_precision"],
        "false_emergency_rate": global_fer,
        "intent_preservation_rate": global_intent_pres,
        "dysarthric_accuracy": dys_acc,
        "dysarthric_macro_f1": dys_met["macro_f1"],
        "dysarthric_emergency_recall": dys_met["emergency_recall"],
        "dysarthric_emergency_precision": dys_met["emergency_precision"],
        "dysarthric_fer": dys_met["false_emergency_rate"],
        "control_accuracy": ctrl_acc,
        "control_macro_f1": ctrl_met["macro_f1"],
        "control_emergency_recall": ctrl_met["emergency_recall"],
        "control_emergency_precision": ctrl_met["emergency_precision"],
        "control_fer": ctrl_met["false_emergency_rate"],
        "total_train_time_sec": float(df_spk["train_time_sec"].sum()),
        "total_eval_time_sec": float(df_spk["eval_time_sec"].sum()),
        "per_class_report": class_report,
        "confusion_matrix": cm,
        "canonical_classes": list(CANONICAL_CLASSES)
    }

    metrics_json = os.path.join(track_dir, "metrics_track_c.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    df_summary = pd.DataFrame([{
        "track": "Track_C_Classical_Shadow",
        "accuracy": global_acc,
        "macro_f1": global_met["macro_f1"],
        "emergency_recall": global_met["emergency_recall"],
        "emergency_precision": global_met["emergency_precision"],
        "false_emergency_rate": global_fer,
        "intent_preservation_rate": global_intent_pres,
        "train_samples": sum(df_spk["train_samples"]),
        "test_samples": len(df_pred),
        "train_time_sec": float(df_spk["train_time_sec"].sum()),
        "eval_time_sec": float(df_spk["eval_time_sec"].sum())
    }])
    summary_csv = os.path.join(track_dir, "results_track_c.csv")
    df_summary.to_csv(summary_csv, index=False)

    # Sentinel SHADOW_RUN_COMPLETE.json
    pred_hash = sha256_file(pred_csv)
    res_hash = sha256_file(summary_csv)
    spk_hash = sha256_file(res_spk_csv)

    sentinel_data = {
        "track": "Track_C_Classical_Shadow",
        "status": "complete",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_fingerprint": manifest["dataset_fingerprint"],
        "total_test_predictions": len(df_pred),
        "global_metrics": {
            "accuracy": global_acc,
            "macro_f1": global_met["macro_f1"],
            "emergency_recall": global_met["emergency_recall"],
            "emergency_precision": global_met["emergency_precision"],
            "false_emergency_rate": global_fer,
            "intent_preservation_rate": global_intent_pres
        },
        "predictions_hash": pred_hash,
        "results_hash": res_hash,
        "per_speaker_hash": spk_hash
    }

    sentinel_path_track = os.path.join(track_dir, "SHADOW_RUN_COMPLETE.json")
    sentinel_path_top = os.path.join(results_dir, "SHADOW_RUN_COMPLETE.json")
    with open(sentinel_path_track, "w", encoding="utf-8") as f:
        json.dump(sentinel_data, f, indent=2)
    with open(sentinel_path_top, "w", encoding="utf-8") as f:
        json.dump(sentinel_data, f, indent=2)

    print("\n" + "=" * 80)
    print("TRACK C MATCHED CLASSICAL SHADOW COMPLETED AND FROZEN")
    print(f"Global Accuracy:          {global_acc:.4f}")
    print(f"Global Macro-F1:          {global_met['macro_f1']:.4f}")
    print(f"Global Emergency Recall:  {global_met['emergency_recall']:.4f} ({int(global_met['emergency_recall']*60)}/60)")
    print(f"Global Emergency Prec:    {global_met['emergency_precision']:.4f}")
    print(f"Global FER:               {global_fer:.4f}")
    print(f"Intent Preservation Rate: {global_intent_pres:.4f}")
    print(f"Dysarthric Recall:        {dys_met['emergency_recall']:.4f} | Control Recall: {ctrl_met['emergency_recall']:.4f}")
    print(f"Sentinel written to:      {sentinel_path_top}")
    print("=" * 80)

if __name__ == "__main__":
    main()
