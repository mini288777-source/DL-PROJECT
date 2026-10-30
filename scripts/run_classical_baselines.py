"""
Phase 3 Classical Baselines & Few-Shot Sample Efficiency Experiment Runner.

Executes:
1. Baseline A: Trainable Classical Adapter Head on Frozen Whisper Embeddings
2. Baseline B: Classical RBF-Kernel SVM on Frozen Whisper Embeddings

Supports:
- Pilot mode (--pilot: 1 dysarthric + 1 control speaker)
- Full 15-speaker sweep (120 conditions)
- Per-recording prediction logging for downstream IPER analysis
- Comprehensive metrics logging and run logging
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.cache import load_embedding_file
from src.models.classical_adapter import ClassicalAdapterClassifier
from src.models.evaluation import compute_intent_metrics
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.utils.seeding import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 3 Classical Baselines")
    parser.add_argument("--config", type=str, default="configs/phase3_baselines.yaml",
                        help="Path to Phase 3 YAML configuration")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot experiment (1 dysarthric F01 + 1 control FC01)")
    parser.add_argument("--speakers", type=str, default=None,
                        help="Comma-separated speaker IDs (overrides config)")
    parser.add_argument("--sample-sizes", type=str, default=None,
                        help="Comma-separated sample Ns (overrides config)")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods ('adapter', 'rbf_svm')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides config)")
    return parser.parse_args()


def load_cohort_data(
    manifest_path: str,
    splits_path: str,
    whisper_dir: str,
    target_speakers: List[str],
    sample_sizes: List[int],
    expected_dim: int = 384,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Preload manifest rows, split index mappings, and cached embeddings for target speakers.
    """
    df_manifest = pd.read_csv(manifest_path).set_index("recording_id", drop=False)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_json = json.load(f)

    # Build split lookup: (speaker_id, sample_n) -> entry
    split_lookup = {}
    needed_recording_ids = set()

    for entry in splits_json:
        spk = entry["speaker_id"]
        n = entry["sample_n"]
        if spk in target_speakers and n in sample_sizes:
            split_lookup[(spk, n)] = entry
            needed_recording_ids.update(entry["adaptation_recording_ids"])
            needed_recording_ids.update(entry["test_recording_ids"])

    # Load only required embeddings from disk
    embedding_cache: Dict[str, np.ndarray] = {}
    for rid in needed_recording_ids:
        arr = load_embedding_file(rid, feature_type="whisper", base_dir=os.path.dirname(whisper_dir), expected_dim=expected_dim)
        embedding_cache[rid] = arr

    return df_manifest, split_lookup, embedding_cache


def run_experiment(
    config_path: str,
    pilot_mode: bool = False,
    override_speakers: Optional[List[str]] = None,
    override_sample_sizes: Optional[List[int]] = None,
    override_methods: Optional[List[str]] = None,
    override_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    start_total_time = time.perf_counter()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load intent taxonomy
    taxonomy_path = cfg["data"]["intent_taxonomy_path"]
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    all_classes = [item["intent_id"] for item in taxonomy["intents"]]

    # Resolve parameters
    if pilot_mode:
        target_speakers = ["F01", "FC01"]  # 1 dysarthric, 1 control
        print("\n>>> RUNNING IN PILOT MODE (Speakers: F01 [dysarthric], FC01 [control]) <<<")
    elif override_speakers:
        target_speakers = override_speakers
    else:
        target_speakers = cfg["cohort"]["speakers"]

    sample_sizes = override_sample_sizes or cfg["cohort"]["sample_sizes"]
    methods = override_methods or cfg["methods"]
    seed = cfg["cohort"]["seed"]
    output_dir = override_output_dir or cfg["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    whisper_dir = cfg["data"]["whisper_cache_dir"]
    manifest_path = cfg["data"]["manifest_path"]
    splits_path = cfg["data"]["splits_json_path"]
    expected_dim = cfg["data"]["embedding_dim"]

    print("=" * 70)
    print("PHASE 3: CLASSICAL BASELINES EXPERIMENT EXECUTION")
    print(f"Target Speakers ({len(target_speakers)}): {target_speakers}")
    print(f"Sample Sizes:    {sample_sizes}")
    print(f"Methods:         {methods}")
    print(f"Global Seed:     {seed}")
    print(f"Output Directory:{output_dir}")
    print("=" * 70)

    # Preload data
    print("Loading manifest, split definitions, and cached Whisper embeddings...")
    df_manifest, split_lookup, embedding_cache = load_cohort_data(
        manifest_path=manifest_path,
        splits_path=splits_path,
        whisper_dir=whisper_dir,
        target_speakers=target_speakers,
        sample_sizes=sample_sizes,
        expected_dim=expected_dim,
    )
    print(f"Loaded {len(embedding_cache)} embedding vectors across target cohort.")

    # Storage structures for outputs
    baseline_curve_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    run_log_rows: List[Dict[str, Any]] = []
    model_config_rows: List[Dict[str, Any]] = []

    total_conditions = len(target_speakers) * len(methods) * len(sample_sizes)
    current_condition = 0
    num_success = 0
    num_failed = 0

    adapter_cfg = cfg["adapter"]
    rbf_cfg = cfg["rbf_svm"]

    # Main experimental loop
    for spk in target_speakers:
        for n in sample_sizes:
            split_entry = split_lookup.get((spk, n))
            if split_entry is None:
                print(f"[WARN] No split found for {spk} N={n}, skipping.")
                continue

            ad_rids = split_entry["adaptation_recording_ids"]
            test_rids = split_entry["test_recording_ids"]

            # Load adaptation vectors and labels
            X_train = np.vstack([embedding_cache[rid] for rid in ad_rids])
            y_train = [str(df_manifest.loc[rid, "intent"]) for rid in ad_rids]

            # Load held-out test vectors and labels
            X_test = np.vstack([embedding_cache[rid] for rid in test_rids])
            y_test = [str(df_manifest.loc[rid, "intent"]) for rid in test_rids]

            for method in methods:
                current_condition += 1
                cond_start_time = time.perf_counter()
                run_id = f"phase3_{spk}_{method}_N{n}_s{seed}"
                timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

                print(f"[{current_condition}/{total_conditions}] Running {spk} | {method:7s} | N={n:2d} ... ", end="")

                try:
                    # Deterministic seeding before every model run
                    set_seed(seed)

                    if method == "adapter":
                        clf = ClassicalAdapterClassifier(
                            input_dim=adapter_cfg["input_dim"],
                            hidden_dim=adapter_cfg["hidden_dim"],
                            num_classes=len(all_classes),
                            dropout=adapter_cfg["dropout"],
                            learning_rate=adapter_cfg["learning_rate"],
                            weight_decay=adapter_cfg["weight_decay"],
                            epochs=adapter_cfg["epochs"],
                            batch_size=adapter_cfg["batch_size"],
                            class_weight=adapter_cfg["class_weight"],
                            seed=seed,
                            all_classes=all_classes,
                        )
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        param_count = clf.get_parameter_count()
                        ad_time = clf.adaptation_time_sec
                        inf_time = clf.inference_time_sec
                        status_note = "FALLBACK_SINGLE_CLASS" if clf.is_single_class_fallback else "STANDARD"

                    elif method == "rbf_svm":
                        clf = ClassicalRBFSVMClassifier(
                            C=rbf_cfg["C"],
                            gamma=rbf_cfg["gamma"],
                            class_weight=rbf_cfg["class_weight"],
                            standardize=rbf_cfg["standardize"],
                            probability=rbf_cfg["probability"],
                            seed=seed,
                            all_classes=all_classes,
                        )
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        param_count = clf.support_vector_count
                        ad_time = clf.adaptation_time_sec
                        inf_time = clf.inference_time_sec
                        status_note = "FALLBACK_SINGLE_CLASS" if clf.is_single_class_fallback else "STANDARD"

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # Compute classification metrics
                    metrics = compute_intent_metrics(y_true=y_test, y_pred=y_pred, all_classes=all_classes)
                    cond_runtime = time.perf_counter() - cond_start_time

                    # Save machine-readable confusion matrix
                    cm_filename = f"cm_{spk}_{method}_N{n}.json"
                    cm_filepath = os.path.join(cm_dir, cm_filename)
                    with open(cm_filepath, "w", encoding="utf-8") as f:
                        json.dump({
                            "run_id": run_id,
                            "speaker_id": spk,
                            "method": method,
                            "sample_n": n,
                            "confusion_matrix": metrics["confusion_matrix"],
                            "classes": all_classes,
                            "per_class": metrics["per_class"],
                        }, f, indent=2)

                    # Baseline curve summary row
                    baseline_curve_rows.append({
                        "speaker_id": spk,
                        "method": method,
                        "sample_n": n,
                        "seed": seed,
                        "train_count": len(y_train),
                        "test_count": len(y_test),
                        "accuracy": round(metrics["accuracy"], 4),
                        "macro_f1": round(metrics["macro_f1"], 4),
                        "weighted_f1": round(metrics["weighted_f1"], 4),
                        "precision_macro": round(metrics["precision_macro"], 4),
                        "recall_macro": round(metrics["recall_macro"], 4),
                        "adaptation_time_sec": round(ad_time, 4),
                        "inference_time_sec": round(inf_time, 4),
                        "parameter_count": param_count,
                        "feature_type": "whisper",
                        "embedding_dim": expected_dim,
                        "wer": "NOT_APPLICABLE_IN_PHASE_3",
                        "iper": "NOT_APPLICABLE_IN_PHASE_3",
                        "status_note": status_note,
                        "run_id": run_id,
                        "timestamp": timestamp_str,
                    })

                    # Per-recording predictions (crucial for future IPER in Phase 6)
                    for rid, yt, yp in zip(test_rids, y_test, y_pred):
                        ref_transcript = str(df_manifest.loc[rid, "transcript"]) if "transcript" in df_manifest.columns else ""
                        prediction_rows.append({
                            "run_id": run_id,
                            "speaker_id": spk,
                            "recording_id": rid,
                            "method": method,
                            "sample_n": n,
                            "true_intent": yt,
                            "predicted_intent": yp,
                            "split": "test",
                            "seed": seed,
                            "reference_transcript": ref_transcript,
                        })

                    # Run log row
                    run_log_rows.append({
                        "run_id": run_id,
                        "speaker_id": spk,
                        "method": method,
                        "sample_n": n,
                        "status": "SUCCESS",
                        "failure_reason": "NONE" if status_note == "STANDARD" else status_note,
                        "runtime_sec": round(cond_runtime, 4),
                        "timestamp": timestamp_str,
                    })

                    num_success += 1
                    print(f"SUCCESS (Acc: {metrics['accuracy']:.3f}, Macro-F1: {metrics['macro_f1']:.3f}, time: {cond_runtime:.2f}s)")

                except Exception as e:
                    num_failed += 1
                    cond_runtime = time.perf_counter() - cond_start_time
                    err_msg = str(e)
                    print(f"FAILED ({err_msg})")

                    run_log_rows.append({
                        "run_id": run_id,
                        "speaker_id": spk,
                        "method": method,
                        "sample_n": n,
                        "status": "FAILED",
                        "failure_reason": err_msg,
                        "runtime_sec": round(cond_runtime, 4),
                        "timestamp": timestamp_str,
                    })

    # Save output CSVs
    df_curves = pd.DataFrame(baseline_curve_rows)
    df_curves_path = os.path.join(output_dir, "baseline_curves.csv")
    df_curves.to_csv(df_curves_path, index=False)

    df_preds = pd.DataFrame(prediction_rows)
    df_preds_path = os.path.join(output_dir, "predictions.csv")
    df_preds.to_csv(df_preds_path, index=False)

    df_logs = pd.DataFrame(run_log_rows)
    df_logs_path = os.path.join(output_dir, "run_log.csv")
    df_logs.to_csv(df_logs_path, index=False)

    # Save model configuration accounting
    adapter_sample_clf = ClassicalAdapterClassifier(
        input_dim=adapter_cfg["input_dim"],
        hidden_dim=adapter_cfg["hidden_dim"],
        num_classes=len(all_classes),
        dropout=adapter_cfg["dropout"],
    )
    rbf_sample_clf = ClassicalRBFSVMClassifier(
        C=rbf_cfg["C"],
        gamma=rbf_cfg["gamma"],
        class_weight=rbf_cfg["class_weight"],
    )

    model_config_rows = [
        {
            "method": "adapter",
            "architecture": adapter_sample_clf.get_model_info()["architecture"],
            "trainable_parameters": adapter_sample_clf.get_parameter_count(),
            "epochs": adapter_cfg["epochs"],
            "learning_rate": adapter_cfg["learning_rate"],
            "optimizer": adapter_cfg["optimizer"],
            "weight_decay": adapter_cfg["weight_decay"],
            "batch_size": adapter_cfg["batch_size"],
            "class_weight": adapter_cfg["class_weight"],
            "seed": seed,
            "average_adaptation_time_sec": round(float(df_curves[df_curves["method"] == "adapter"]["adaptation_time_sec"].mean()), 4) if len(df_curves) > 0 else 0.0,
        },
        {
            "method": "rbf_svm",
            "architecture": "sklearn.svm.SVC(kernel='rbf')",
            "trainable_parameters": "Support Vectors (Data-Dependent)",
            "epochs": "N/A (Convex Quadratic Program)",
            "learning_rate": "N/A",
            "optimizer": "SMO / libsvm",
            "weight_decay": f"C={rbf_cfg['C']}",
            "batch_size": "Full Adaptation Batch",
            "class_weight": rbf_cfg["class_weight"],
            "seed": seed,
            "average_adaptation_time_sec": round(float(df_curves[df_curves["method"] == "rbf_svm"]["adaptation_time_sec"].mean()), 4) if len(df_curves) > 0 else 0.0,
        }
    ]
    pd.DataFrame(model_config_rows).to_csv("reports/phase3_model_configuration.csv", index=False)

    total_time = time.perf_counter() - start_total_time
    print("\n" + "=" * 70)
    print("PHASE 3 EXECUTION SUMMARY")
    print(f"Total Conditions: {total_conditions}")
    print(f"Successful:       {num_success}")
    print(f"Failed:           {num_failed}")
    print(f"Total Runtime:    {total_time:.2f} s")
    print(f"Curves saved:     {df_curves_path}")
    print(f"Predictions saved:{df_preds_path}")
    print(f"Run log saved:    {df_logs_path}")
    print("=" * 70)

    return {
        "total_conditions": total_conditions,
        "success": num_success,
        "failed": num_failed,
        "curves_path": df_curves_path,
        "predictions_path": df_preds_path,
        "run_log_path": df_logs_path,
        "total_time_sec": total_time,
    }


if __name__ == "__main__":
    args = parse_args()
    spks = [s.strip() for s in args.speakers.split(",")] if args.speakers else None
    ns = [int(s.strip()) for s in args.sample_sizes.split(",")] if args.sample_sizes else None
    meths = [s.strip() for s in args.methods.split(",")] if args.methods else None

    run_experiment(
        config_path=args.config,
        pilot_mode=args.pilot,
        override_speakers=spks,
        override_sample_sizes=ns,
        override_methods=meths,
        override_output_dir=args.output_dir,
    )
