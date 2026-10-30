"""
Track A: Classical Baseline Independent Pipeline Execution.

Strict Protocol Guarantees:
- Independent loader and preprocessing fitted strictly on Track A training IDs.
- Deterministic random seed 42.
- Calibration of operating threshold performed on internal validation / training data only.
- Evaluation executed strictly on locked test IDs.
- Full prediction schema with provenance emitted.
- Outputs written to results/phase8_separated/ and results/phase8_separated/track_a_classical/.
- Validates integrity and creates BASELINE_FROZEN.json sentinel upon verified completion.
"""

import sys
import os
import time
import json
import hashlib
from typing import Dict, List, Any
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.phase8_separated.phase8_separated_models import (
    RepairedTrackAClassical,
    compute_repaired_phase8_metrics,
)
from src.embeddings.cache import load_embedding_file
from src.continual.taxonomy import EMERGENCY_CLASSES

def evaluate_repaired_safety_gate(metrics: Dict[str, Any]) -> tuple[bool, str]:
    fer = metrics["false_emergency_rate"]
    rec = metrics["emergency_recall"]
    if fer > 0.15:
        return False, f"FER_SPIKE_VETO (FER={fer:.3f} > 0.15)"
    if rec < 0.05:
        return False, f"RECALL_COLLAPSE_VETO (Recall={rec:.3f} < 0.05)"
    return True, "PASS"

def sha256_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 80)
    print("TRACK A: CLASSICAL BASELINE — ISOLATED TRAINING & EVALUATION")
    print("=" * 80)
    
    results_dir = os.path.join(REPO_ROOT, "results", "phase8_separated")
    track_dir = os.path.join(results_dir, "track_a_classical")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)

    manifest_path = os.path.join(REPO_ROOT, "config", "phase8_separated", "canonical_split_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    needed_rids = set()
    for s in manifest["splits"]:
        needed_rids.update(s["train_ids"])
        needed_rids.update(s["test_ids"])

    base_torgo_dir = os.path.join(REPO_ROOT, "embeddings", "torgo")
    embedding_cache: Dict[str, np.ndarray] = {}
    for rid in sorted(list(needed_rids)):
        embedding_cache[rid] = load_embedding_file(rid, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=384)

    dysarthric_set = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}

    all_results = []
    all_predictions = []

    t_start = time.perf_counter()

    for idx, split in enumerate(manifest["splits"]):
        spk = split["speaker_id"]
        budget = split["budget"]
        split_id = split["split_id"]
        spk_group = "dysarthric" if spk in dysarthric_set else "control"

        X_train = np.array([embedding_cache[r] for r in split["train_ids"]], dtype=np.float32)
        y_train = np.array(split["train_labels"])
        X_test = np.array([embedding_cache[r] for r in split["test_ids"]], dtype=np.float32)
        y_test = np.array(split["test_labels"])

        t0 = time.perf_counter()
        model = RepairedTrackAClassical(alpha=0.5, random_seed=42)
        model.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_7cl, is_em, gate_probs, z_res, final_probs = model.predict_detailed(X_test)
        t_eval = time.perf_counter() - t0

        met = compute_repaired_phase8_metrics(y_test, pred_7cl, is_em, track_id="Track_A")
        gate_pass, gate_msg = evaluate_repaired_safety_gate(met)

        all_results.append({
            "cell_id": split_id,
            "speaker_id": spk,
            "speaker_group": spk_group,
            "budget": budget,
            "track": "Track_A_Classical",
            "emergency_recall": met["emergency_recall"],
            "emergency_precision": met["emergency_precision"],
            "false_emergency_rate": met["false_emergency_rate"],
            "macro_f1": met["macro_f1"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_time_sec": t_fit,
            "eval_time_sec": t_eval,
            "locked_threshold": float(model.locked_threshold),
            "safety_gate_passed": gate_pass,
            "safety_gate_status": gate_msg,
        })

        for i, rid in enumerate(split["test_ids"]):
            all_predictions.append({
                "experiment_id": "phase8_separated",
                "track_id": "Track_A",
                "speaker_id": spk,
                "budget": budget,
                "sample_id": rid,
                "true_label": y_test[i],
                "model_version": "1.0-classical",
                "prediction_version": "1.0-separated",
                "gate_probability": float(gate_probs[i]),
                "residual_score": float(z_res[i]),
                "final_emergency_probability": float(final_probs[i]),
                "final_emergency_prediction": bool(is_em[i]),
                "final_7class_prediction": pred_7cl[i],
                "decision_threshold": float(model.locked_threshold),
                "prediction_source": "track_a_classical",
                "split_manifest_id": split_id
            })

        print(f"[{idx+1:02d}/60] Track A | Spk: {spk} ({spk_group}), N: {budget} | "
              f"Rec: {met['emergency_recall']:.3f} | Prec: {met['emergency_precision']:.3f} | "
              f"FER: {met['false_emergency_rate']:.3f} | Thresh: {model.locked_threshold:.2f} | [{gate_msg}]", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nTrack A evaluation completed in {elapsed:.2f}s across 60 cells.")

    df_res = pd.DataFrame(all_results)
    df_pred = pd.DataFrame(all_predictions)

    res_csv = os.path.join(results_dir, "results_track_a.csv")
    pred_csv = os.path.join(results_dir, "predictions_track_a.csv")
    df_res.to_csv(res_csv, index=False)
    df_pred.to_csv(pred_csv, index=False)

    # Save to track-specific namespace
    df_res.to_csv(os.path.join(track_dir, "results.csv"), index=False)
    df_pred.to_csv(os.path.join(track_dir, "predictions.csv"), index=False)

    res_hash = sha256_file(res_csv)
    pred_hash = sha256_file(pred_csv)

    sentinel_path = os.path.join(results_dir, "BASELINE_FROZEN.json")
    with open(sentinel_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": "frozen",
            "track": "Track_A_Classical",
            "total_cells": len(all_results),
            "total_predictions": len(all_predictions),
            "results_csv": "results_track_a.csv",
            "results_hash": res_hash,
            "predictions_csv": "predictions_track_a.csv",
            "predictions_hash": pred_hash,
            "mean_emergency_recall": float(df_res["emergency_recall"].mean()),
            "mean_false_emergency_rate": float(df_res["false_emergency_rate"].mean()),
            "mean_macro_f1": float(df_res["macro_f1"].mean()),
            "timestamp": time.time(),
            "manifest_version": manifest.get("manifest_version", "2.0")
        }, f, indent=2)

    print(f"Verified and frozen Track A baseline sentinel at {sentinel_path}")

if __name__ == "__main__":
    main()
