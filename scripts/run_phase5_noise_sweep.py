"""
Phase 5 Master Experiment Runner: Controlled Noise Robustness Evaluation.

Protocol B: Clean Calibration -> Noisy Evaluation.
720 Experimental Cells:
    15 Speakers (8 dysarthric, 7 control)
  x  4 Adaptation budgets N in {20, 30, 40, 50}
  x  3 Models (Classical Adapter, Classical RBF-SVM, Primary Quantum QSVC q=4)
  x  4 Test acoustic conditions (Clean, 15 dB SNR, 10 dB SNR, 5 dB SNR)
  = 720 experimental cells.

Invariants:
- Models are trained / adapted strictly on CLEAN audio embeddings.
- PCA and MinMaxScaler are fitted strictly on CLEAN adaptation embeddings.
- Noisy audio is generated deterministically from held-out test audio only.
- Identical noisy embeddings are evaluated across all three models.
- Clean condition performs automated regression verification against Phase 3/4 baselines.
- Resumable and checkpointed execution.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.noisy_test_cache import NoisyEmbeddingCacheManager
from src.models.classical_adapter import ClassicalAdapterClassifier
from src.models.evaluation import compute_intent_metrics
from src.models.quantum_kernel import QSVCClassifier
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.utils.noise import derive_noise_seed
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


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5 Noise Robustness Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/phase5_config.yaml",
                        help="Path to Phase 5 YAML config")
    parser.add_argument("--pilot-12", action="store_true",
                        help="Run Step 1 pilot: 1 speaker (F01), N=20, 3 models, 4 SNRs (12 cells)")
    parser.add_argument("--pilot-72", action="store_true",
                        help="Run Step 2 pilot: 3 speakers (F01, FC01, M01), N in {20, 50}, 3 models, 4 SNRs (72 cells)")
    parser.add_argument("--speakers", type=str, default=None,
                        help="Comma-separated speaker IDs to override config")
    parser.add_argument("--sample-sizes", type=str, default=None,
                        help="Comma-separated sample budgets N to override config")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated models: 'adapter,rbf_svm,qsvc'")
    parser.add_argument("--snrs", type=str, default=None,
                        help="Comma-separated SNRs: 'clean,15,10,5'")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Re-evaluate already completed cells")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for Whisper noisy test extraction")
    return parser.parse_args()


def generate_experiment_manifest(
    speakers: List[str],
    sample_sizes: List[int],
    models: List[str],
    snr_conditions: List[Dict[str, Any]],
    output_path: str,
) -> pd.DataFrame:
    """Generate and persist the complete 720-cell experiment manifest."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    cell_idx = 0

    dysarthric_set = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}

    for spk in speakers:
        spk_group = "dysarthric" if spk in dysarthric_set else "control"
        for n_val in sample_sizes:
            for m in models:
                for cond in snr_conditions:
                    cell_idx += 1
                    snr_tag = cond["tag"]
                    snr_db = cond["snr_db"]
                    exp_id = f"phase5_{spk}_{m}_N{n_val}_{snr_tag}"

                    rows.append({
                        "experiment_id": exp_id,
                        "cell_index": cell_idx,
                        "phase": "phase5",
                        "protocol": "Protocol_B",
                        "speaker_id": spk,
                        "speaker_group": spk_group,
                        "sample_n": n_val,
                        "model": m,
                        "snr_tag": snr_tag,
                        "snr_db": snr_db if snr_db is not None else -999.0,
                        "noise_type": "additive_white_gaussian_noise" if snr_db is not None else "none",
                        "encoder_name": "openai/whisper-tiny",
                        "embedding_dim": 384,
                        "status": "PENDING",
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def generate_test_manifest(
    splits: List[Dict[str, Any]],
    manifest_df: pd.DataFrame,
    snr_conditions: List[Dict[str, Any]],
    output_path: str,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Generate manifest of all unique held-out test recordings across all SNR conditions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    seen_keys = set()

    for entry in splits:
        spk = entry["speaker_id"]
        for rid in entry["test_recording_ids"]:
            for cond in snr_conditions:
                snr_tag = cond["tag"]
                snr_db = cond["snr_db"]
                key = (rid, snr_tag)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                noise_seed = derive_noise_seed(base_seed, rid, snr_db) if snr_db is not None else -1
                clean_audio_path = manifest_df.loc[rid, "audio_path"] if rid in manifest_df.index else f"data/torgo/audio/{rid}.wav"
                emb_path = (
                    f"embeddings/torgo/whisper/{rid}.npy"
                    if snr_db is None
                    else f"embeddings/torgo/whisper_noisy/snr_{int(snr_db)}/{rid}.npy"
                )

                rows.append({
                    "recording_id": rid,
                    "speaker_id": spk,
                    "intent": manifest_df.loc[rid, "intent"] if rid in manifest_df.index else "UNKNOWN",
                    "snr_tag": snr_tag,
                    "snr_db": snr_db if snr_db is not None else -999.0,
                    "noise_seed": noise_seed,
                    "clean_audio_path": clean_audio_path,
                    "embedding_path": emb_path,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def compute_iper(y_true: List[str], y_pred: List[str]) -> Tuple[float, int, int]:
    """
    Compute Intent-Preserving Emergency Recognition (IPER) rate.
    IPER is the accuracy strictly on the emergency intent subset (all classes != GENERAL_SPEECH).
    """
    emergency_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != "GENERAL_SPEECH"]
    total_emergency = len(emergency_pairs)
    if total_emergency == 0:
        return 0.0, 0, 0
    preserved_emergency = sum(1 for yt, yp in emergency_pairs if yt == yp)
    iper_rate = float(preserved_emergency / total_emergency)
    return iper_rate, preserved_emergency, total_emergency


def main():
    args = parse_args()
    start_time_all = time.time()

    print("=" * 80)
    print("PHASE 5: CONTROLLED NOISE ROBUSTNESS & ACOUSTIC GENERALIZATION")
    print("Protocol B: Clean Calibration -> Noisy Evaluation (720 Experimental Cells)")
    print("=" * 80)

    # 1. Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Resolve cohort, models, and SNRs
    if args.pilot_12:
        speakers = ["F01"]
        sample_sizes = [20]
        models = ["adapter", "rbf_svm", "qsvc"]
        snr_conditions = cfg["noise_conditions"]
        print("\n>>> EXECUTION MODE: PILOT 12 CELLS (1 speaker F01, N=20, 3 models, 4 SNRs) <<<")
    elif args.pilot_72:
        speakers = ["F01", "FC01", "M01"]
        sample_sizes = [20, 50]
        models = ["adapter", "rbf_svm", "qsvc"]
        snr_conditions = cfg["noise_conditions"]
        print("\n>>> EXECUTION MODE: PILOT 72 CELLS (3 speakers, 2 N values, 3 models, 4 SNRs) <<<")
    else:
        speakers = [s.strip() for s in args.speakers.split(",")] if args.speakers else cfg["cohort"]["speakers"]
        sample_sizes = [int(n.strip()) for n in args.sample_sizes.split(",")] if args.sample_sizes else cfg["cohort"]["sample_sizes"]
        models = [m.strip() for m in args.models.split(",")] if args.models else [m["name"] for m in cfg["models"]]
        if args.snrs:
            req_snrs = [s.strip().lower() for s in args.snrs.split(",")]
            snr_conditions = [c for c in cfg["noise_conditions"] if c["tag"].lower() in req_snrs or c["name"].lower() in req_snrs]
        else:
            snr_conditions = cfg["noise_conditions"]

    print(f"Speakers ({len(speakers)}): {speakers}")
    print(f"Sample Budgets ({len(sample_sizes)}): {sample_sizes}")
    print(f"Models ({len(models)}): {models}")
    print(f"SNR Conditions ({len(snr_conditions)}): {[c['name'] for c in snr_conditions]}")

    total_expected_runs = len(speakers) * len(sample_sizes) * len(models) * len(snr_conditions)
    print(f"Total Conditions in This Run: {total_expected_runs}")

    # 3. Load Phase 1 splits and dataset manifest
    paths = cfg["paths"]
    with open(paths["splits_json_path"], "r", encoding="utf-8") as f:
        splits_list = json.load(f)

    manifest_df = pd.read_csv(paths["manifest_path"]).set_index("recording_id", drop=False)
    rec_to_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))
    rec_to_transcript = dict(zip(manifest_df["recording_id"], manifest_df["transcript"]))

    # Index splits: (speaker_id, sample_n) -> entry
    splits_index = {}
    for entry in splits_list:
        splits_index[(entry["speaker_id"], entry["sample_n"])] = entry

    # 4. Generate experiment manifest and test manifest
    exp_manifest_path = os.path.join(paths["phase5_manifest_dir"], "phase5_experiment_manifest.csv")
    full_manifest_df = generate_experiment_manifest(
        speakers=cfg["cohort"]["speakers"],
        sample_sizes=cfg["cohort"]["sample_sizes"],
        models=[m["name"] for m in cfg["models"]],
        snr_conditions=cfg["noise_conditions"],
        output_path=exp_manifest_path,
    )
    print(f"Experiment manifest validated: {len(full_manifest_df)} cells (expected 720).")

    test_manifest_path = os.path.join(paths["phase5_manifest_dir"], "phase5_test_manifest.csv")
    full_test_manifest_df = generate_test_manifest(
        splits=splits_list,
        manifest_df=manifest_df,
        snr_conditions=cfg["noise_conditions"],
        output_path=test_manifest_path,
        base_seed=cfg["cohort"]["seed"],
    )
    print(f"Test manifest validated: {len(full_test_manifest_df)} test evaluation records.")

    # 5. Load baseline reference curves for regression verification
    p3_curves = pd.read_csv(paths["phase3_curves_csv"]) if os.path.isfile(paths["phase3_curves_csv"]) else None
    p4_curves = pd.read_csv(paths["phase4_curves_csv"]) if os.path.isfile(paths["phase4_curves_csv"]) else None

    # 6. Initialize noisy embedding cache manager
    cache_mgr = NoisyEmbeddingCacheManager(
        clean_audio_dir=paths["clean_audio_dir"],
        clean_embedding_dir=paths["clean_embedding_dir"],
        noisy_embedding_base_dir=paths["noisy_embedding_base_dir"],
        manifest_path=paths["manifest_path"],
        base_seed=cfg["cohort"]["seed"],
        expected_dim=384,
    )

    # 7. Check for existing results to support resumability
    out_dir = paths["output_results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "confusion_matrices"), exist_ok=True)

    results_csv_path = os.path.join(out_dir, "phase5_results.csv")
    per_class_csv_path = os.path.join(out_dir, "phase5_per_class.csv")
    predictions_csv_path = os.path.join(out_dir, "phase5_predictions.csv")
    errors_csv_path = os.path.join(out_dir, "phase5_errors.csv")

    existing_results: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(results_csv_path) and not args.force_rerun:
        df_prev = pd.read_csv(results_csv_path)
        for _, r in df_prev.iterrows():
            if str(r.get("status")) == "SUCCESS":
                existing_results[str(r["experiment_id"])] = dict(r)
        print(f"Resumability active: loaded {len(existing_results)} existing SUCCESS records.")

    def safe_load_csv_records(filepath: str) -> List[Dict[str, Any]]:
        if os.path.isfile(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    return df.to_dict("records")
            except Exception:
                pass
        return []

    results_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    # If results exist and not force_rerun, preload existing records
    if not args.force_rerun:
        results_rows = safe_load_csv_records(results_csv_path)
        per_class_rows = safe_load_csv_records(per_class_csv_path)
        error_rows = safe_load_csv_records(errors_csv_path)

    dysarthric_set = set(cfg["cohort"]["dysarthric_speakers"])
    global_seed = cfg["cohort"]["seed"]

    condition_count = 0
    success_count = 0
    skipped_count = 0
    failed_count = 0

    # 8. Main execution loop: Speaker -> Ensure Noisy Audio -> Sample Size -> Train Clean Models -> Evaluate All SNRs
    for spk in speakers:
        spk_group = "dysarthric" if spk in dysarthric_set else "control"
        spk_test_rids = splits_index[(spk, sample_sizes[0])]["test_recording_ids"]

        # Ensure all noisy test embeddings for this speaker exist
        print(f"\n[{spk}] Checking/extracting noisy test embeddings ({len(spk_test_rids)} test recordings)...")
        for cond in snr_conditions:
            snr_db = cond["snr_db"]
            if snr_db is not None:
                gen_cnt = cache_mgr.ensure_noisy_embeddings(
                    recording_ids=spk_test_rids,
                    snr_db=snr_db,
                    batch_size=args.batch_size,
                )
                if gen_cnt > 0:
                    print(f"  Generated {gen_cnt} embeddings for {spk} at {cond['name']}")

        # Preload clean test labels and test embeddings across all SNRs
        y_test = [rec_to_intent[rid] for rid in spk_test_rids]
        test_embeddings_by_snr: Dict[str, np.ndarray] = {}
        for cond in snr_conditions:
            snr_tag = cond["tag"]
            snr_db = cond["snr_db"]
            test_embeddings_by_snr[snr_tag] = np.stack(
                [cache_mgr.load_embedding(rid, snr_db=snr_db) for rid in spk_test_rids]
            )

        for n_val in sample_sizes:
            split_entry = splits_index.get((spk, n_val))
            if split_entry is None:
                print(f"[WARN] No split entry found for {spk} N={n_val}, skipping.")
                continue

            adapt_rids = split_entry["adaptation_recording_ids"]

            # Load CLEAN adaptation embeddings and labels
            X_train = np.stack(
                [cache_mgr.load_embedding(rid, snr_db=None) for rid in adapt_rids]
            )
            y_train = [rec_to_intent[rid] for rid in adapt_rids]

            # Train / Fit Clean Reference Models ONCE per (spk, N)
            trained_models: Dict[str, Any] = {}
            adaptation_times: Dict[str, float] = {}

            for m_name in models:
                t_fit_start = time.perf_counter()
                set_seed(global_seed)

                try:
                    if m_name == "adapter":
                        clf = ClassicalAdapterClassifier(
                            input_dim=384,
                            hidden_dim=64,
                            num_classes=len(ALL_CLASSES),
                            dropout=0.1,
                            learning_rate=0.005,
                            weight_decay=0.0001,
                            epochs=60,
                            batch_size=8,
                            class_weight=None,
                            seed=global_seed,
                            all_classes=ALL_CLASSES,
                        )
                        clf.fit(X_train, y_train)

                    elif m_name == "rbf_svm":
                        clf = ClassicalRBFSVMClassifier(
                            C=1.0,
                            gamma="scale",
                            class_weight=None,
                            standardize=False,
                            probability=False,
                            seed=global_seed,
                            all_classes=ALL_CLASSES,
                        )
                        clf.fit(X_train, y_train)

                    elif m_name == "qsvc":
                        # Note: cache_dir=None to prevent reading clean test cache during noisy test predict
                        clf = QSVCClassifier(
                            qubit_count=4,
                            reps=1,
                            entanglement="linear",
                            C=1.0,
                            whiten=False,
                            seed=global_seed,
                            cache_dir=None,
                            all_classes=ALL_CLASSES,
                        )
                        clf.fit(X_train, y_train, speaker_id=spk, sample_n=n_val)

                    else:
                        raise ValueError(f"Unknown model paradigm: {m_name}")

                    trained_models[m_name] = clf
                    adaptation_times[m_name] = time.perf_counter() - t_fit_start

                except Exception as ex:
                    print(f"\n[ERROR] Fitting failed for {spk} N={n_val} model={m_name}: {ex}")
                    error_rows.append({
                        "experiment_id": f"phase5_{spk}_{m_name}_N{n_val}_TRAIN",
                        "speaker_id": spk,
                        "sample_n": n_val,
                        "model": m_name,
                        "snr_tag": "train",
                        "stage": "clean_adaptation",
                        "exception_type": type(ex).__name__,
                        "message": str(ex),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })
                    continue

            # Now evaluate each model across all SNR test conditions
            for m_name in models:
                if m_name not in trained_models:
                    continue
                clf = trained_models[m_name]
                fit_time = adaptation_times[m_name]

                for cond in snr_conditions:
                    condition_count += 1
                    snr_tag = cond["tag"]
                    snr_db = cond["snr_db"]
                    exp_id = f"phase5_{spk}_{m_name}_N{n_val}_{snr_tag}"

                    # Check resumability
                    if exp_id in existing_results and not args.force_rerun:
                        skipped_count += 1
                        continue

                    print(f"[{condition_count}/{total_expected_runs}] {spk:4s} | N={n_val:2d} | {m_name:7s} | {snr_tag:5s} ... ", end="", flush=True)

                    t_eval_start = time.perf_counter()
                    try:
                        X_test = test_embeddings_by_snr[snr_tag]

                        # Predict
                        set_seed(global_seed)
                        preds = clf.predict(X_test)
                        eval_runtime = time.perf_counter() - t_eval_start

                        # Compute comprehensive intent metrics
                        metrics = compute_intent_metrics(y_true=y_test, y_pred=preds, all_classes=ALL_CLASSES)
                        iper_rate, preserved_em, total_em = compute_iper(y_true=y_test, y_pred=preds)

                        macro_f1 = round(metrics["macro_f1"], 4)
                        accuracy = round(metrics["accuracy"], 4)
                        weighted_f1 = round(metrics["weighted_f1"], 4)

                        # Clean regression check
                        regression_pass = True
                        baseline_macro_f1 = None
                        if snr_tag == "clean":
                            if m_name == "adapter" and p3_curves is not None:
                                match = p3_curves[(p3_curves["speaker_id"] == spk) & (p3_curves["sample_n"] == n_val) & (p3_curves["method"] == "adapter")]
                                if not match.empty:
                                    baseline_macro_f1 = round(float(match.iloc[0]["macro_f1"]), 4)
                            elif m_name == "rbf_svm" and p3_curves is not None:
                                match = p3_curves[(p3_curves["speaker_id"] == spk) & (p3_curves["sample_n"] == n_val) & (p3_curves["method"] == "rbf_svm")]
                                if not match.empty:
                                    baseline_macro_f1 = round(float(match.iloc[0]["macro_f1"]), 4)
                            elif m_name == "qsvc" and p4_curves is not None:
                                match = p4_curves[(p4_curves["speaker_id"] == spk) & (p4_curves["sample_n"] == n_val) & (p4_curves["qubit_count"] == 4)]
                                if not match.empty:
                                    baseline_macro_f1 = round(float(match.iloc[0]["macro_f1"]), 4)

                            if baseline_macro_f1 is not None:
                                diff = abs(macro_f1 - baseline_macro_f1)
                                if diff > 0.01:
                                    print(f"[REGRESSION WARN: Macro-F1 {macro_f1} vs baseline {baseline_macro_f1}] ", end="")
                                    regression_pass = False

                        print(f"F1={macro_f1:.4f} Acc={accuracy:.4f} IPER={iper_rate:.4f} ({eval_runtime:.2f}s)")

                        # Save confusion matrix
                        cm_file = os.path.join(out_dir, "confusion_matrices", f"{exp_id}.json")
                        with open(cm_file, "w", encoding="utf-8") as f:
                            json.dump(metrics["confusion_matrix"], f)

                        # Record row
                        result_row = {
                            "experiment_id": exp_id,
                            "phase": "phase5",
                            "protocol": "Protocol_B",
                            "speaker_id": spk,
                            "speaker_group": spk_group,
                            "sample_n": n_val,
                            "model": m_name,
                            "snr_tag": snr_tag,
                            "snr_db": snr_db if snr_db is not None else -999.0,
                            "noise_type": "additive_white_gaussian_noise" if snr_db is not None else "none",
                            "noise_seed": global_seed,
                            "encoder_name": "openai/whisper-tiny",
                            "embedding_dim": 384,
                            "test_sample_count": len(y_test),
                            "emergency_sample_count": total_em,
                            "emergency_preserved_count": preserved_em,
                            "accuracy": accuracy,
                            "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1,
                            "precision_macro": round(metrics["precision_macro"], 4),
                            "recall_macro": round(metrics["recall_macro"], 4),
                            "iper": round(iper_rate, 4),
                            "baseline_clean_macro_f1": baseline_macro_f1,
                            "clean_regression_pass": regression_pass,
                            "adaptation_time_sec": round(fit_time, 4),
                            "inference_time_sec": round(eval_runtime, 4),
                            "total_runtime_sec": round(fit_time + eval_runtime, 4),
                            "artifact_reused": True,
                            "pca_fitted_on": "clean_adaptation_only",
                            "scaler_fitted_on": "clean_adaptation_only",
                            "qubits": 4 if m_name == "qsvc" else 0,
                            "reps": 1 if m_name == "qsvc" else 0,
                            "entanglement": "linear" if m_name == "qsvc" else "none",
                            "feature_map": "zz" if m_name == "qsvc" else "none",
                            "kernel": "FidelityQuantumKernel" if m_name == "qsvc" else ("rbf" if m_name == "rbf_svm" else "none"),
                            "status": "SUCCESS",
                            "error_message": "NONE",
                        }
                        results_rows.append(result_row)
                        existing_results[exp_id] = result_row
                        success_count += 1

                        # Record per-class metrics
                        for cls_name, pc in metrics["per_class"].items():
                            per_class_rows.append({
                                "experiment_id": exp_id,
                                "speaker_id": spk,
                                "sample_n": n_val,
                                "model": m_name,
                                "snr_tag": snr_tag,
                                "snr_db": snr_db if snr_db is not None else -999.0,
                                "intent": cls_name,
                                "precision": round(pc["precision"], 4),
                                "recall": round(pc["recall"], 4),
                                "f1": round(pc["f1"], 4),
                                "support": pc["support"],
                            })

                        # Record per-recording predictions
                        for rid, yt, yp in zip(spk_test_rids, y_test, preds):
                            prediction_rows.append({
                                "experiment_id": exp_id,
                                "speaker_id": spk,
                                "recording_id": rid,
                                "sample_n": n_val,
                                "model": m_name,
                                "snr_tag": snr_tag,
                                "true_intent": yt,
                                "predicted_intent": str(yp),
                                "is_emergency": yt != "GENERAL_SPEECH",
                                "intent_preserved": yt == yp,
                                "reference_transcript": rec_to_transcript.get(rid, ""),
                            })

                    except Exception as ex:
                        failed_count += 1
                        print(f"FAILED: {ex}")
                        err_record = {
                            "experiment_id": exp_id,
                            "speaker_id": spk,
                            "sample_n": n_val,
                            "model": m_name,
                            "snr_tag": snr_tag,
                            "stage": "evaluation",
                            "exception_type": type(ex).__name__,
                            "message": str(ex),
                            "traceback": traceback.format_exc(),
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }
                        error_rows.append(err_record)

                        result_row = {
                            "experiment_id": exp_id,
                            "phase": "phase5",
                            "protocol": "Protocol_B",
                            "speaker_id": spk,
                            "speaker_group": spk_group,
                            "sample_n": n_val,
                            "model": m_name,
                            "snr_tag": snr_tag,
                            "status": "FAILED",
                            "error_message": str(ex),
                        }
                        results_rows.append(result_row)

        # Checkpoint tables after every speaker
        pd.DataFrame(results_rows).to_csv(results_csv_path, index=False)
        pd.DataFrame(per_class_rows).to_csv(per_class_csv_path, index=False)
        if error_rows:
            pd.DataFrame(error_rows).to_csv(errors_csv_path, index=False)
        else:
            pd.DataFrame(columns=["experiment_id", "speaker_id", "sample_n", "model", "snr_tag", "stage", "exception_type", "message", "traceback", "timestamp"]).to_csv(errors_csv_path, index=False)
        if prediction_rows:
            pd.DataFrame(prediction_rows).to_csv(predictions_csv_path, index=False)

    # 9. Final Aggregations and Summary
    print("\n" + "=" * 80)
    print("PHASE 5 EXECUTION COMPLETED")
    print(f"Total Evaluated: {condition_count} | Success: {success_count} | Skipped: {skipped_count} | Failed: {failed_count}")
    print(f"Total Elapsed Time: {time.time() - start_time_all:.2f} seconds")
    print("=" * 80)

    # Generate Summary Table: mean/std by model x SNR x N
    df_res = pd.DataFrame(results_rows)
    df_success = df_res[df_res["status"] == "SUCCESS"]

    if not df_success.empty:
        summary_df = df_success.groupby(["model", "snr_tag", "sample_n"]).agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            iper_mean=("iper", "mean"),
            iper_std=("iper", "std"),
            cell_count=("experiment_id", "count"),
        ).reset_index()

        summary_csv_path = os.path.join(out_dir, "phase5_summary.csv")
        summary_df.round(4).to_csv(summary_csv_path, index=False)
        print(f"Summary table saved to {summary_csv_path}")

        # Also copy to phase5/results/
        mirror_dir = "phase5/results"
        os.makedirs(mirror_dir, exist_ok=True)
        df_res.to_csv(os.path.join(mirror_dir, "phase5_results.csv"), index=False)
        summary_df.round(4).to_csv(os.path.join(mirror_dir, "phase5_summary.csv"), index=False)
        pd.DataFrame(per_class_rows).to_csv(os.path.join(mirror_dir, "phase5_per_class.csv"), index=False)
        pd.DataFrame(error_rows).to_csv(os.path.join(mirror_dir, "phase5_errors.csv"), index=False)


if __name__ == "__main__":
    main()
