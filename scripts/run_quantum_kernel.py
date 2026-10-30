"""
Master Runner for Phase 4 Quantum Kernel Personalization Experiments.

Executes:
1. Primary Quantum Configuration (15 speakers x 4 budgets = 60 conditions, q=4, reps=1).
2. Qubit Count Ablation (q in {2, 4, 6, 8} across all 60 conditions).
3. Kernel matrix validation, persistent caching, and quality diagnostics.
4. Per-recording prediction logging for future IPER analysis.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.evaluation import compute_intent_metrics
from src.models.quantum_kernel import QSVCClassifier
from src.utils.seeding import set_seed

ALL_CLASSES = [
    "SOS_HELP",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "LOCATION",
    "CRITICAL_RESPONSE",
    "GENERAL_SPEECH",
]


def load_config(config_path: str = "configs/phase4_quantum.yaml") -> Dict[str, Any]:
    """Load YAML experiment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(
    config: Dict[str, Any],
    pilot_mode: bool = False,
    run_ablation: bool = True,
) -> None:
    """Run full Phase 4 quantum kernel benchmark."""
    print("=" * 70)
    print("PHASE 4: QUANTUM KERNEL FEW-SHOT PERSONALIZATION EXPERIMENT")
    print("=" * 70)

    paths = config["paths"]
    primary_cfg = config["primary"]
    ablation_qubits = config["ablation"]["qubits"] if run_ablation else [primary_cfg["qubits"]]

    # Ensure output directories exist
    os.makedirs(paths["cache_dir"], exist_ok=True)
    os.makedirs(paths["confusion_dir"], exist_ok=True)
    os.makedirs(paths["figures_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(paths["curves_csv"]), exist_ok=True)

    # 1. Load splits and manifest
    print(f"\nLoading splits from {paths['splits_json']}...")
    with open(paths["splits_json"], "r", encoding="utf-8") as f:
        splits_list = json.load(f)

    manifest_df = pd.read_csv(paths["manifest_csv"])
    rec_to_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))
    rec_to_transcript = dict(zip(manifest_df["recording_id"], manifest_df["transcript"]))

    # 2. Select target cohort
    if pilot_mode:
        target_speakers = ["F01", "FC01"]
        print(f"PILOT MODE: Running on representative cohort: {target_speakers}")
    else:
        target_speakers = sorted(list(set(entry["speaker_id"] for entry in splits_list)))
        print(f"FULL SWEEP: Running on complete 15-speaker cohort ({len(target_speakers)} speakers)")

    sample_sizes = config["protocol"]["sample_sizes"]
    whisper_dir = paths["whisper_dir"]

    # Pre-index splits: (speaker_id, sample_n) -> entry
    splits_index = {}
    for entry in splits_list:
        key = (entry["speaker_id"], entry["sample_n"])
        splits_index[key] = entry

    # Tracking lists
    curve_records: List[Dict[str, Any]] = []
    prediction_records: List[Dict[str, Any]] = []
    kernel_quality_records: List[Dict[str, Any]] = []

    total_conditions = len(target_speakers) * len(sample_sizes) * len(ablation_qubits)
    condition_idx = 0
    start_all = time.time()

    print(f"Total experimental conditions to evaluate: {total_conditions}\n")

    for q in ablation_qubits:
        print(f"\n--- EXECUTING QUBIT COUNT: {q} QUBITS (PCA={q}) ---")
        for sp in target_speakers:
            for n_val in sample_sizes:
                condition_idx += 1
                run_id = f"phase4_{sp}_qsvc_q{q}_N{n_val}_s{primary_cfg['seed']}"
                print(f"[{condition_idx}/{total_conditions}] Running {run_id} ...", end=" ", flush=True)

                split_entry = splits_index.get((sp, n_val))
                if split_entry is None:
                    print("SKIPPED (Split entry not found)")
                    continue

                adapt_rids = split_entry["adaptation_recording_ids"]
                test_rids = split_entry["test_recording_ids"]

                # Load embeddings
                X_train = np.stack(
                    [np.load(os.path.join(whisper_dir, f"{rid}.npy")) for rid in adapt_rids]
                )
                y_train = [rec_to_intent[rid] for rid in adapt_rids]

                X_test = np.stack(
                    [np.load(os.path.join(whisper_dir, f"{rid}.npy")) for rid in test_rids]
                )
                y_test = [rec_to_intent[rid] for rid in test_rids]

                # Check single-class training condition
                unique_train_intents = sorted(list(set(y_train)))
                is_single_class = len(unique_train_intents) <= 1

                t_cond_start = time.time()

                # Set unified seed
                set_seed(primary_cfg["seed"])

                # Instantiate QSVC
                clf = QSVCClassifier(
                    qubit_count=q,
                    reps=primary_cfg["reps"],
                    entanglement=primary_cfg["entanglement"],
                    C=primary_cfg["C"],
                    whiten=config["pca"]["whiten"],
                    seed=primary_cfg["seed"],
                    cache_dir=paths["cache_dir"],
                    all_classes=ALL_CLASSES,
                )

                try:
                    # Fit
                    clf.fit(X_train, y_train, speaker_id=sp, sample_n=n_val)

                    # Predict
                    preds = clf.predict(X_test, speaker_id=sp, sample_n=n_val)

                    # Evaluate
                    metrics = compute_intent_metrics(y_test, preds, ALL_CLASSES)

                    total_cond_time = time.time() - t_cond_start
                    status = "SUCCESS" if not is_single_class else "SINGLE_CLASS"
                    failure_reason = "NONE" if not is_single_class else "SINGLE_CLASS_ADAPTATION_PRIOR"

                    props = clf.get_circuit_properties()

                    # Save confusion matrix
                    cm_path = os.path.join(paths["confusion_dir"], f"{run_id}.json")
                    with open(cm_path, "w", encoding="utf-8") as f:
                        json.dump(metrics["confusion_matrix"], f)

                    # Log per-recording predictions
                    for rid, yt, yp in zip(test_rids, y_test, preds):
                        prediction_records.append({
                            "run_id": run_id,
                            "speaker_id": sp,
                            "recording_id": rid,
                            "sample_n": n_val,
                            "qubit_count": q,
                            "true_intent": yt,
                            "predicted_intent": str(yp),
                            "split": "test",
                            "reference_transcript": rec_to_transcript.get(rid, ""),
                        })

                    # Log kernel quality diagnostics if kernel was computed
                    if clf.K_train is not None:
                        k_diag = clf.kernel_engine.validate_kernel_matrix(clf.K_train, is_training=True)
                        kernel_quality_records.append({
                            "speaker_id": sp,
                            "sample_n": n_val,
                            "qubits": q,
                            "kernel_symmetry_error": k_diag["symmetry_error"],
                            "min_eigenvalue_raw": k_diag["min_eigenvalue"],
                            "min_eigenvalue_used": max(k_diag["min_eigenvalue"], 0.0),
                            "condition_number": k_diag["condition_number"],
                            "trace": k_diag["trace"],
                            "mean_off_diagonal": k_diag["mean_off_diagonal"],
                            "std_off_diagonal": k_diag["std_off_diagonal"],
                            "duplicate_count": 0,
                            "psd_projection_applied": False,
                        })

                    # Record curve entry
                    curve_records.append({
                        "run_id": run_id,
                        "speaker_id": sp,
                        "method": "qsvc",
                        "sample_n": n_val,
                        "qubit_count": q,
                        "pca_dim": q,
                        "seed": primary_cfg["seed"],
                        "true_test_count": len(y_test),
                        "feature_map": "zz",
                        "reps": primary_cfg["reps"],
                        "entanglement": primary_cfg["entanglement"],
                        "C": primary_cfg["C"],
                        "accuracy": round(metrics["accuracy"], 4),
                        "macro_f1": round(metrics["macro_f1"], 4),
                        "weighted_f1": round(metrics["weighted_f1"], 4),
                        "precision_macro": round(metrics["precision_macro"], 4),
                        "recall_macro": round(metrics["recall_macro"], 4),
                        "adaptation_runtime_sec": round(clf.adaptation_time_sec, 4),
                        "kernel_train_runtime_sec": round(clf.kernel_train_time_sec, 4),
                        "kernel_test_runtime_sec": round(clf.kernel_test_time_sec, 4),
                        "total_runtime_sec": round(total_cond_time, 4),
                        "inference_latency_sec": round(clf.inference_time_sec, 4),
                        "trainable_quantum_parameters": props["trainable_quantum_parameters"],
                        "circuit_depth": props["circuit_depth"],
                        "one_qubit_gate_count": props["one_qubit_gate_count"],
                        "two_qubit_gate_count": props["two_qubit_gate_count"],
                        "kernel_train_size": len(adapt_rids),
                        "kernel_test_size": len(test_rids),
                        "single_class_training": is_single_class,
                        "status": status,
                        "failure_reason": failure_reason,
                    })

                    print(f"DONE (Acc: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, Time: {total_cond_time:.2f}s)")

                except Exception as e:
                    print(f"FAILED: {e}")
                    curve_records.append({
                        "run_id": run_id,
                        "speaker_id": sp,
                        "method": "qsvc",
                        "sample_n": n_val,
                        "qubit_count": q,
                        "pca_dim": q,
                        "seed": primary_cfg["seed"],
                        "true_test_count": len(y_test),
                        "feature_map": "zz",
                        "reps": primary_cfg["reps"],
                        "entanglement": primary_cfg["entanglement"],
                        "C": primary_cfg["C"],
                        "accuracy": 0.0,
                        "macro_f1": 0.0,
                        "weighted_f1": 0.0,
                        "precision_macro": 0.0,
                        "recall_macro": 0.0,
                        "adaptation_runtime_sec": 0.0,
                        "kernel_train_runtime_sec": 0.0,
                        "kernel_test_runtime_sec": 0.0,
                        "total_runtime_sec": round(time.time() - t_cond_start, 4),
                        "inference_latency_sec": 0.0,
                        "trainable_quantum_parameters": 0,
                        "circuit_depth": 0,
                        "one_qubit_gate_count": 0,
                        "two_qubit_gate_count": 0,
                        "kernel_train_size": len(adapt_rids),
                        "kernel_test_size": len(test_rids),
                        "single_class_training": is_single_class,
                        "status": "FAILED",
                        "failure_reason": str(e),
                    })

    # Save output artifacts
    print("\nSaving output datasets...")

    curves_df = pd.DataFrame(curve_records)
    curves_df.to_csv(paths["curves_csv"], index=False)
    print(f"Saved quantum curves: {paths['curves_csv']} ({len(curves_df)} rows)")

    preds_df = pd.DataFrame(prediction_records)
    preds_df.to_csv(paths["predictions_csv"], index=False)
    print(f"Saved quantum predictions: {paths['predictions_csv']} ({len(preds_df)} rows)")

    if kernel_quality_records:
        kq_df = pd.DataFrame(kernel_quality_records)
        kq_df.to_csv(paths["kernel_quality_csv"], index=False)
        print(f"Saved kernel quality report: {paths['kernel_quality_csv']} ({len(kq_df)} rows)")

    total_elapsed = time.time() - start_all
    print(f"\nAll Phase 4 quantum conditions completed in {total_elapsed:.2f}s ({total_elapsed/60.0:.2f}m).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 Quantum Kernel Experiments")
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode on 2 speakers only")
    parser.add_argument("--no-ablation", action="store_true", help="Run only primary q=4 configuration")
    args = parser.parse_args()

    cfg = load_config()
    run_experiment(cfg, pilot_mode=args.pilot, run_ablation=not args.no_ablation)
