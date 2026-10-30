"""
Subphase 6.5.5 — Quantum Kernel Recovery & Optimization Experiment Runner
Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid
         Quantum-Classical Deep Learning for Reliable Emergency Communication

Execution Track: Phase 6.5 Recovery Protocol
Splits: Track B (phase6_5/splits/recovery_splits_hierarchical_ready.json)
Cohort: 15 TORGO speakers (8 dysarthric, 7 control)
Calibration budgets: N in {20, 30, 40, 50}

Models Evaluated:
1. Primary Matched Comparison (d=8):
   - QSVC Q8-R1: d=8, 8 qubits, ZZFeatureMap, linear entanglement, reps=1, FidelityQuantumKernel, C=1.0
   - Weighted Adapter d=8: Classical MLP adapter, input_dim=8
   - Weighted RBF-SVM d=8: Classical RBF kernel SVM, input_dim=8, C=1.0
2. Secondary Quantum Ablations:
   - QSVC Q4-R1: d=4, 4 qubits, ZZFeatureMap, linear entanglement, reps=1, FidelityQuantumKernel, C=1.0
   - QSVC Q8-R2: d=8, 8 qubits, ZZFeatureMap, linear entanglement, reps=2, FidelityQuantumKernel, C=1.0
3. Matched Classical Compact Baselines (d=4):
   - Weighted Adapter d=4
   - Weighted RBF-SVM d=4

Architecture: Hierarchical Two-Stage
   Stage 1: Binary Emergency Gate (GENERAL_SPEECH vs EMERGENCY)
   Stage 2: Conditional Emergency Subtype Router (Feasible emergency classes only)
   End-to-End: Gated Hierarchical Routing
"""

import os
import sys
import time
import json
import warnings
import argparse
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.quantum.pca_pipeline import TrainOnlyPCAPipeline
from src.quantum.fidelity_kernel import QuantumFidelityKernelEngine
from src.models.classical_adapter import ClassicalAdapterClassifier
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.embeddings.cache import load_embedding_file
from src.utils.seeding import set_seed

SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")
BASE_TORGO_DIR = os.path.join(REPO_ROOT, "embeddings", "torgo")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
KERNEL_CACHE_DIR = os.path.join(RESULTS_DIR, "phase6_5_5", "kernel_cache")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "phase6_5_5", "figures")

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]
DYSARTHRIC_SET = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Subphase 6.5.5 Quantum Recovery Experiment")
    parser.add_argument("--pilot", action="store_true", help="Run 6-condition pilot (3 speakers x 2 N)")
    return parser.parse_args()


def compute_iper(y_true: List[str], y_pred: List[str]) -> Tuple[float, int, int]:
    emergency_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != "GENERAL_SPEECH"]
    total_emergency = len(emergency_pairs)
    if total_emergency == 0:
        return 0.0, 0, 0
    preserved_emergency = sum(1 for yt, yp in emergency_pairs if yt == yp)
    return float(preserved_emergency / total_emergency), preserved_emergency, total_emergency


def run_experiment(pilot: bool = False):
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(KERNEL_CACHE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("============================================================")
    print(" SUBPHASE 6.5.5 — QUANTUM KERNEL RECOVERY & OPTIMIZATION")
    print(f" Mode: {'PILOT (3 speakers x 2 budgets = 6 conditions)' if pilot else 'FULL EXPERIMENT (15 speakers x 4 budgets = 60 conditions)'}")
    print(f" Master Seed: 42")
    print(f" Kernel Cache: {KERNEL_CACHE_DIR}")
    print("============================================================")

    # 1. Load splits & manifest
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        all_splits = json.load(f)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    rec_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))

    if pilot:
        target_splits = [s for s in all_splits if s["speaker_id"] in ["F01", "M01", "FC01"] and s["sample_n"] in [20, 40]]
    else:
        target_splits = all_splits

    print(f"Loaded {len(target_splits)} conditions to evaluate.")

    # 2. Pre-cache embeddings
    print("Pre-caching Whisper embeddings...")
    unique_recs = set()
    for s in target_splits:
        unique_recs.update(s["adaptation_recording_ids"])
        unique_recs.update(s["test_recording_ids"])

    embedding_cache = {}
    for r in sorted(list(unique_recs)):
        emb = load_embedding_file(r, feature_type="whisper", base_dir=BASE_TORGO_DIR, expected_dim=384)
        embedding_cache[r] = emb.astype(np.float32)
    print(f"Loaded {len(embedding_cache)} unique embeddings.")

    # Model specifications to run
    # Format: (model_key, model_family, dim, qubit_count, reps, C)
    model_specs = [
        ("qsvc_q8_r1", "quantum", 8, 8, 1, 1.0),
        ("hierarchical_adapter_d8", "adapter", 8, 0, 0, 0.0),
        ("hierarchical_rbf_svm_d8", "rbf_svm", 8, 0, 0, 1.0),
        ("qsvc_q4_r1", "quantum", 4, 4, 1, 1.0),
        ("qsvc_q8_r2", "quantum", 8, 8, 2, 1.0),
        ("hierarchical_adapter_d4", "adapter", 4, 0, 0, 0.0),
        ("hierarchical_rbf_svm_d4", "rbf_svm", 4, 0, 0, 1.0),
    ]

    all_speaker_results = []
    kernel_diagnostics_records = []
    runtime_records = []

    t_start_all = time.time()
    total_evals = len(target_splits) * len(model_specs)
    completed_evals = 0

    # Quantum engines (reusable across conditions)
    q_engines = {
        (4, 1): QuantumFidelityKernelEngine(num_qubits=4, reps=1, entanglement="linear"),
        (8, 1): QuantumFidelityKernelEngine(num_qubits=8, reps=1, entanglement="linear"),
        (8, 2): QuantumFidelityKernelEngine(num_qubits=8, reps=2, entanglement="linear"),
    }

    for split_idx, s in enumerate(target_splits):
        t_cond_start = time.perf_counter()
        spk = s["speaker_id"]
        n = s["sample_n"]
        cohort = "dysarthric" if spk in DYSARTHRIC_SET else "control"

        adapt_recs = s["adaptation_recording_ids"]
        test_recs = s["test_recording_ids"]

        X_adapt_raw = np.array([embedding_cache[r] for r in adapt_recs])
        X_test_raw = np.array([embedding_cache[r] for r in test_recs])

        y_adapt = [rec_intent[r] for r in adapt_recs]
        y_test = [rec_intent[r] for r in test_recs]

        # Stage 1 targets
        y_adapt_s1 = ["EMERGENCY" if l != "GENERAL_SPEECH" else "GENERAL_SPEECH" for l in y_adapt]
        y_test_s1_bin = [1 if l != "GENERAL_SPEECH" else 0 for l in y_test]

        # Stage 2 adaptation targets (emergency only)
        emerg_adapt_idx = [i for i, l in enumerate(y_adapt) if l != "GENERAL_SPEECH"]
        y_emerg_adapt = [y_adapt[i] for i in emerg_adapt_idx]
        feasible_emergency_classes = sorted(list(set(y_emerg_adapt)))
        stage2_num_classes = len(feasible_emergency_classes)
        is_single_class = (stage2_num_classes == 1)
        single_class_name = feasible_emergency_classes[0] if is_single_class else None

        # Precompute PCA & Quantum Scalers for d=4 and d=8
        pca_pipelines = {}
        for d in [4, 8]:
            pipe = TrainOnlyPCAPipeline(n_components=d, random_state=42, feature_range=(0.0, np.pi))
            pipe.fit(X_adapt_raw)
            pca_pipelines[d] = pipe

        # Precompute or load Quantum Kernels
        cached_kernels = {}
        for (q, reps), engine in q_engines.items():
            pipe = pca_pipelines[q]
            X_adapt_q = pipe.transform(X_adapt_raw)
            X_test_q = pipe.transform(X_test_raw)

            cache_prefix = f"{spk}_N{n}_q{q}_r{reps}"
            train_k_path = os.path.join(KERNEL_CACHE_DIR, f"{cache_prefix}_train.npy")
            test_k_path = os.path.join(KERNEL_CACHE_DIR, f"{cache_prefix}_test.npy")
            meta_k_path = os.path.join(KERNEL_CACHE_DIR, f"{cache_prefix}_meta.json")

            t_k_train0 = time.perf_counter()
            if os.path.exists(train_k_path) and os.path.exists(test_k_path):
                K_train = np.load(train_k_path)
                K_test = np.load(test_k_path)
                t_k_train = 0.0
                t_k_test = 0.0
            else:
                K_train = engine.evaluate(X_adapt_q)
                t_k_train = time.perf_counter() - t_k_train0

                t_k_test0 = time.perf_counter()
                K_test = engine.evaluate(X_test_q, X_adapt_q)
                t_k_test = time.perf_counter() - t_k_test0

                # Save matrices
                np.save(train_k_path, K_train.astype(np.float64))
                np.save(test_k_path, K_test.astype(np.float64))

            diag = engine.validate_kernel_matrix(K_train)
            diag_record = {
                "speaker_id": spk,
                "cohort_group": cohort,
                "N": n,
                "qubit_count": q,
                "reps": reps,
                "train_shape": list(K_train.shape),
                "test_shape": list(K_test.shape),
                "is_finite": diag["is_finite"],
                "symmetry_error": diag["symmetry_error"],
                "diagonal_error": diag["diagonal_error"],
                "min_eigenvalue": diag["min_eigenvalue"],
                "max_eigenvalue": diag["max_eigenvalue"],
                "condition_number": diag["condition_number"],
                "trace": diag["trace"],
                "mean_off_diagonal": diag["mean_off_diagonal"],
                "std_off_diagonal": diag["std_off_diagonal"],
                "pca_cumulative_variance": pipe.cumulative_explained_variance_,
                "psd_status": "STRICTLY_PSD" if diag["min_eigenvalue"] >= -1e-8 else "NON_PSD",
                "kernel_train_time_sec": t_k_train,
                "kernel_test_time_sec": t_k_test,
            }
            kernel_diagnostics_records.append(diag_record)

            with open(meta_k_path, "w", encoding="utf-8") as f:
                json.dump(diag_record, f, indent=2)

            cached_kernels[(q, reps)] = {
                "K_train": K_train,
                "K_test": K_test,
                "t_train": t_k_train,
                "t_test": t_k_test,
                "diag": diag
            }

        # Run all model specifications
        for (m_key, m_family, d, q, reps, C_val) in model_specs:
            t_train_start = time.perf_counter()
            t_inf_start = 0.0

            unique_s1_classes = sorted(list(set(y_adapt_s1)))
            is_single_class_s1 = (len(unique_s1_classes) == 1)
            single_s1_class = unique_s1_classes[0] if is_single_class_s1 else None

            # ---------------- STAGE 1 TRAINING & INFERENCE ----------------
            if m_family == "quantum":
                k_data = cached_kernels[(q, reps)]
                K_train = k_data["K_train"]
                K_test = k_data["K_test"]

                if is_single_class_s1:
                    s1_model = None
                    train_time = k_data["t_train"]
                    if single_s1_class == "EMERGENCY":
                        p_emerg = np.ones(len(X_test_raw), dtype=np.float64)
                    else:
                        p_emerg = np.zeros(len(X_test_raw), dtype=np.float64)
                    inf_time = k_data["t_test"]
                else:
                    s1_model = SVC(kernel="precomputed", C=C_val, class_weight="balanced", probability=True, random_state=42)
                    s1_model.fit(K_train, y_adapt_s1)
                    train_time = time.perf_counter() - t_train_start + k_data["t_train"]

                    t_inf_0 = time.perf_counter()
                    probs_s1 = s1_model.predict_proba(K_test)
                    emerg_col = s1_model.classes_.tolist().index("EMERGENCY")
                    p_emerg = probs_s1[:, emerg_col]
                    inf_time = time.perf_counter() - t_inf_0 + k_data["t_test"]

            elif m_family == "adapter":
                pipe = pca_pipelines[d]
                X_adapt_d = pipe.pca.transform(X_adapt_raw)
                X_test_d = pipe.pca.transform(X_test_raw)

                s1_model = ClassicalAdapterClassifier(
                    input_dim=d,
                    hidden_dim=min(64, max(16, d * 2)),
                    num_classes=2,
                    dropout=0.1,
                    learning_rate=0.001,
                    weight_decay=0.01,
                    epochs=50,
                    batch_size=16,
                    class_weight="balanced",
                    seed=42,
                    all_classes=["GENERAL_SPEECH", "EMERGENCY"]
                )
                s1_model.fit(X_adapt_d, y_adapt_s1)
                train_time = time.perf_counter() - t_train_start

                t_inf_0 = time.perf_counter()
                probs_s1 = s1_model.predict_proba(X_test_d)
                emerg_col = s1_model.classes_.index("EMERGENCY")
                p_emerg = probs_s1[:, emerg_col]
                inf_time = time.perf_counter() - t_inf_0

            elif m_family == "rbf_svm":
                pipe = pca_pipelines[d]
                X_adapt_d = pipe.pca.transform(X_adapt_raw)
                X_test_d = pipe.pca.transform(X_test_raw)

                s1_model = ClassicalRBFSVMClassifier(
                    C=C_val,
                    gamma="scale",
                    class_weight="balanced",
                    standardize=False,
                    probability=True,
                    seed=42,
                    all_classes=["GENERAL_SPEECH", "EMERGENCY"]
                )
                s1_model.fit(X_adapt_d, y_adapt_s1)
                train_time = time.perf_counter() - t_train_start

                t_inf_0 = time.perf_counter()
                probs_s1 = s1_model.predict_proba(X_test_d)
                emerg_col = s1_model.classes_.index("EMERGENCY")
                p_emerg = probs_s1[:, emerg_col]
                inf_time = time.perf_counter() - t_inf_0

            # Stage 1 binary predictions at primary threshold 0.50
            s1_pred_bin = [1 if p >= 0.50 else 0 for p in p_emerg]
            s1_rec = float(recall_score(y_test_s1_bin, s1_pred_bin, pos_label=1, zero_division=0))
            s1_prec = float(precision_score(y_test_s1_bin, s1_pred_bin, pos_label=1, zero_division=0))
            s1_spec = float(recall_score(y_test_s1_bin, s1_pred_bin, pos_label=0, zero_division=0))
            s1_fpr = 1.0 - s1_spec
            s1_mer = 1.0 - s1_rec

            # ---------------- STAGE 2 TRAINING ----------------
            if is_single_class:
                s2_model = None
            else:
                if m_family == "quantum":
                    K_emerg_train = K_train[np.ix_(emerg_adapt_idx, emerg_adapt_idx)]
                    s2_model = SVC(kernel="precomputed", C=C_val, class_weight="balanced", random_state=42)
                    s2_model.fit(K_emerg_train, y_emerg_adapt)
                elif m_family == "adapter":
                    X_emerg_adapt_d = X_adapt_d[emerg_adapt_idx]
                    s2_model = ClassicalAdapterClassifier(
                        input_dim=d,
                        hidden_dim=min(64, max(16, d * 2)),
                        num_classes=stage2_num_classes,
                        dropout=0.1,
                        learning_rate=0.001,
                        weight_decay=0.01,
                        epochs=50,
                        batch_size=16,
                        class_weight="balanced",
                        seed=42,
                        all_classes=feasible_emergency_classes
                    )
                    s2_model.fit(X_emerg_adapt_d, y_emerg_adapt)
                elif m_family == "rbf_svm":
                    X_emerg_adapt_d = X_adapt_d[emerg_adapt_idx]
                    s2_model = ClassicalRBFSVMClassifier(
                        C=C_val,
                        gamma="scale",
                        class_weight="balanced",
                        standardize=False,
                        probability=True,
                        seed=42,
                        all_classes=feasible_emergency_classes
                    )
                    s2_model.fit(X_emerg_adapt_d, y_emerg_adapt)

            # ---------------- STAGE 2 DIAGNOSTIC EVALUATION ----------------
            true_emerg_idx = [i for i, l in enumerate(y_test) if l != "GENERAL_SPEECH"]
            if len(true_emerg_idx) > 0:
                y_true_emerg = [y_test[i] for i in true_emerg_idx]
                if is_single_class:
                    s2_diag_preds = [single_class_name] * len(true_emerg_idx)
                else:
                    if m_family == "quantum":
                        K_emerg_diag = K_test[np.ix_(true_emerg_idx, emerg_adapt_idx)]
                        s2_diag_preds = s2_model.predict(K_emerg_diag)
                    else:
                        X_emerg_diag = X_test_d[true_emerg_idx]
                        s2_diag_preds = s2_model.predict(X_emerg_diag)

                s2_feas_f1 = float(f1_score(y_true_emerg, s2_diag_preds, labels=feasible_emergency_classes, average="macro", zero_division=0))
                s2_all6_f1 = float(f1_score(y_true_emerg, s2_diag_preds, labels=EMERGENCY_CLASSES, average="macro", zero_division=0))
                s2_bal_acc = float(balanced_accuracy_score(y_true_emerg, s2_diag_preds))
                s2_w_f1 = float(f1_score(y_true_emerg, s2_diag_preds, labels=feasible_emergency_classes, average="weighted", zero_division=0))
            else:
                s2_feas_f1 = 0.0
                s2_all6_f1 = 0.0
                s2_bal_acc = 0.0
                s2_w_f1 = 0.0

            # ---------------- END-TO-END ROUTING ----------------
            gated_emerg_idx = [i for i, b in enumerate(s1_pred_bin) if b == 1]
            if len(gated_emerg_idx) > 0:
                if is_single_class:
                    s2_gated_preds = [single_class_name] * len(gated_emerg_idx)
                else:
                    if m_family == "quantum":
                        K_emerg_gated = K_test[np.ix_(gated_emerg_idx, emerg_adapt_idx)]
                        s2_gated_preds = s2_model.predict(K_emerg_gated)
                    else:
                        X_emerg_gated = X_test_d[gated_emerg_idx]
                        s2_gated_preds = s2_model.predict(X_emerg_gated)
            else:
                s2_gated_preds = []

            e2e_preds = []
            gated_pointer = 0
            for b in s1_pred_bin:
                if b == 0:
                    e2e_preds.append("GENERAL_SPEECH")
                else:
                    e2e_preds.append(s2_gated_preds[gated_pointer])
                    gated_pointer += 1

            # End-to-End Metrics
            iper, n_pres, n_tot = compute_iper(y_test, e2e_preds)
            e2e_macro_f1 = float(f1_score(y_test, e2e_preds, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
            e2e_bin_preds = [0 if p == "GENERAL_SPEECH" else 1 for p in e2e_preds]
            e2e_emerg_rec = float(recall_score(y_test_s1_bin, e2e_bin_preds, pos_label=1, zero_division=0))
            e2e_emerg_prec = float(precision_score(y_test_s1_bin, e2e_bin_preds, pos_label=1, zero_division=0))
            e2e_spec = float(recall_score(y_test_s1_bin, e2e_bin_preds, pos_label=0, zero_division=0))
            e2e_fpr = 1.0 - e2e_spec
            e2e_mer = 1.0 - e2e_emerg_rec

            result_row = {
                "speaker_id": spk,
                "cohort_group": cohort,
                "N": n,
                "model": m_key,
                "model_family": m_family,
                "dimension": d,
                "qubit_count": q,
                "reps": reps,
                "C": C_val,
                "sample_count": len(y_test),
                "feasible_emergency_classes_count": stage2_num_classes,
                "stage1_emergency_recall": round(s1_rec, 4),
                "stage1_emergency_precision": round(s1_prec, 4),
                "stage1_false_positive_rate": round(s1_fpr, 4),
                "stage1_general_recall": round(s1_spec, 4),
                "stage1_missed_emergency_rate": round(s1_mer, 4),
                "stage2_feasible_macro_f1": round(s2_feas_f1, 4),
                "stage2_all6_macro_f1": round(s2_all6_f1, 4),
                "stage2_balanced_accuracy": round(s2_bal_acc, 4),
                "stage2_weighted_f1": round(s2_w_f1, 4),
                "end_to_end_iper": round(iper, 4),
                "end_to_end_emergency_precision": round(e2e_emerg_prec, 4),
                "end_to_end_emergency_recall": round(e2e_emerg_rec, 4),
                "end_to_end_false_emergency_rate": round(e2e_fpr, 4),
                "end_to_end_missed_emergency_rate": round(e2e_mer, 4),
                "end_to_end_macro_f1": round(e2e_macro_f1, 4),
                "train_time_sec": round(train_time, 4),
                "inference_time_sec": round(inf_time, 4),
            }
            all_speaker_results.append(result_row)
            completed_evals += 1

        print(f"Condition [{split_idx+1:02d}/{len(target_splits):02d}] {spk} N={n:02d} completed in {time.perf_counter()-t_cond_start:.2f}s.")

    # 3. Export DataFrames
    all_results_df = pd.DataFrame(all_speaker_results)
    speaker_results_path = os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv")
    all_results_df.to_csv(speaker_results_path, index=False)
    print(f"\nSaved {speaker_results_path} ({len(all_results_df)} rows).")

    # Quantum Conditions CSV
    quantum_df = all_results_df[all_results_df["model_family"] == "quantum"].copy()
    quantum_path = os.path.join(RESULTS_DIR, "phase6_5_5_quantum_conditions.csv")
    quantum_df.to_csv(quantum_path, index=False)
    print(f"Saved {quantum_path} ({len(quantum_df)} rows).")

    # Classical Conditions CSV
    classical_df = all_results_df[all_results_df["model_family"] != "quantum"].copy()
    classical_path = os.path.join(RESULTS_DIR, "phase6_5_5_classical_conditions.csv")
    classical_df.to_csv(classical_path, index=False)
    print(f"Saved {classical_path} ({len(classical_df)} rows).")

    # Kernel Diagnostics CSV
    kernel_diag_df = pd.DataFrame(kernel_diagnostics_records)
    kernel_diag_path = os.path.join(RESULTS_DIR, "phase6_5_5_kernel_diagnostics.csv")
    kernel_diag_df.to_csv(kernel_diag_path, index=False)
    print(f"Saved {kernel_diag_path} ({len(kernel_diag_df)} rows).")

    # Runtime Accounting CSV
    runtime_records = []
    for m_key in all_results_df["model"].unique():
        sub = all_results_df[all_results_df["model"] == m_key]
        row_0 = sub.iloc[0]
        q_val = int(row_0["qubit_count"])
        reps_val = int(row_0["reps"])
        hilbert_dim = 2**q_val if q_val > 0 else 0

        if row_0["model_family"] == "quantum":
            eng = q_engines[(q_val, reps_val)]
            props = eng.get_circuit_properties()
            c_depth = props["circuit_depth"]
            cx_gates = props["two_qubit_gate_count"]
            single_gates = props["one_qubit_gate_count"]
            tot_gates = props["total_gate_count"]
        else:
            c_depth = 0
            cx_gates = 0
            single_gates = 0
            tot_gates = 0

        runtime_records.append({
            "model": m_key,
            "model_family": row_0["model_family"],
            "dimension": int(row_0["dimension"]),
            "qubit_count": q_val,
            "reps": reps_val,
            "hilbert_dim": hilbert_dim,
            "circuit_depth": c_depth,
            "two_qubit_gate_count": cx_gates,
            "one_qubit_gate_count": single_gates,
            "total_gate_count": tot_gates,
            "mean_train_time_sec": round(float(sub["train_time_sec"].mean()), 4),
            "std_train_time_sec": round(float(sub["train_time_sec"].std()), 4),
            "mean_inf_time_sec": round(float(sub["inference_time_sec"].mean()), 4),
            "std_inf_time_sec": round(float(sub["inference_time_sec"].std()), 4),
            "total_wall_time_sec": round(float(sub["train_time_sec"].sum() + sub["inference_time_sec"].sum()), 2),
        })

    runtime_df = pd.DataFrame(runtime_records)
    runtime_path = os.path.join(RESULTS_DIR, "phase6_5_5_runtime.csv")
    runtime_df.to_csv(runtime_path, index=False)
    print(f"Saved {runtime_path} ({len(runtime_df)} rows).")

    # 4. Matched Comparison CSV
    matched_records = []
    condition_keys = target_splits
    for s in condition_keys:
        spk = s["speaker_id"]
        n = s["sample_n"]
        cohort = "dysarthric" if spk in DYSARTHRIC_SET else "control"

        c_sub = all_results_df[(all_results_df["speaker_id"] == spk) & (all_results_df["N"] == n)]
        row_dict = {
            "speaker_id": spk,
            "cohort_group": cohort,
            "N": n,
        }

        # Pull per-model metrics
        for m_key in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8", "qsvc_q4_r1", "qsvc_q8_r2", "hierarchical_rbf_svm_d4", "hierarchical_adapter_d4"]:
            m_row = c_sub[c_sub["model"] == m_key].iloc[0]
            prefix = m_key.replace("hierarchical_", "").replace("_d8", "8").replace("_d4", "4").replace("qsvc_", "")
            row_dict[f"{prefix}_macro_f1"] = m_row["end_to_end_macro_f1"]
            row_dict[f"{prefix}_emerg_recall"] = m_row["end_to_end_emergency_recall"]
            row_dict[f"{prefix}_fpr"] = m_row["end_to_end_false_emergency_rate"]
            row_dict[f"{prefix}_iper"] = m_row["end_to_end_iper"]
            row_dict[f"{prefix}_s2_bal_acc"] = m_row["stage2_balanced_accuracy"]
            row_dict[f"{prefix}_s2_feas_f1"] = m_row["stage2_feasible_macro_f1"]

        # Differences: Q8 vs RBF8 and Q8 vs Adp8
        row_dict["diff_Q8_RBF8_macro_f1"] = round(row_dict["q8_r1_macro_f1"] - row_dict["rbf_svm8_macro_f1"], 4)
        row_dict["diff_Q8_Adp8_macro_f1"] = round(row_dict["q8_r1_macro_f1"] - row_dict["adapter8_macro_f1"], 4)
        row_dict["diff_Q8_RBF8_recall"] = round(row_dict["q8_r1_emerg_recall"] - row_dict["rbf_svm8_emerg_recall"], 4)
        row_dict["diff_Q8_Adp8_recall"] = round(row_dict["q8_r1_emerg_recall"] - row_dict["adapter8_emerg_recall"], 4)
        row_dict["diff_Q8_RBF8_fpr"] = round(row_dict["q8_r1_fpr"] - row_dict["rbf_svm8_fpr"], 4)
        row_dict["diff_Q8_Adp8_fpr"] = round(row_dict["q8_r1_fpr"] - row_dict["adapter8_fpr"], 4)
        row_dict["diff_Q8_RBF8_iper"] = round(row_dict["q8_r1_iper"] - row_dict["rbf_svm8_iper"], 4)
        row_dict["diff_Q8_Adp8_iper"] = round(row_dict["q8_r1_iper"] - row_dict["adapter8_iper"], 4)

        matched_records.append(row_dict)

    matched_df = pd.DataFrame(matched_records)
    matched_path = os.path.join(RESULTS_DIR, "phase6_5_5_matched_comparison.csv")
    matched_df.to_csv(matched_path, index=False)
    print(f"Saved {matched_path} ({len(matched_df)} rows).")

    # 5. Statistical Hypothesis Testing
    stat_records = []
    planned_comparisons = [
        # (ID, metric_col, model_A_col, model_B_col, label)
        ("P1", "end_to_end_macro_f1", "qsvc_q8_r1", "hierarchical_rbf_svm_d8", "Primary: Q8-R1 vs RBF-SVM d=8 on Macro-F1"),
        ("P2", "end_to_end_emergency_recall", "qsvc_q8_r1", "hierarchical_rbf_svm_d8", "Primary: Q8-R1 vs RBF-SVM d=8 on Emergency Recall"),
        ("P3", "end_to_end_false_emergency_rate", "qsvc_q8_r1", "hierarchical_rbf_svm_d8", "Primary: Q8-R1 vs RBF-SVM d=8 on False Emergency Rate"),
        ("P4", "end_to_end_iper", "qsvc_q8_r1", "hierarchical_rbf_svm_d8", "Primary: Q8-R1 vs RBF-SVM d=8 on IPER"),
        ("S1", "end_to_end_macro_f1", "qsvc_q8_r1", "hierarchical_adapter_d8", "Secondary: Q8-R1 vs Adapter d=8 on Macro-F1"),
        ("S2", "end_to_end_emergency_recall", "qsvc_q8_r1", "hierarchical_adapter_d8", "Secondary: Q8-R1 vs Adapter d=8 on Emergency Recall"),
        ("S3", "end_to_end_false_emergency_rate", "qsvc_q8_r1", "hierarchical_adapter_d8", "Secondary: Q8-R1 vs Adapter d=8 on False Emergency Rate"),
        ("S4", "end_to_end_iper", "qsvc_q8_r1", "hierarchical_adapter_d8", "Secondary: Q8-R1 vs Adapter d=8 on IPER"),
        ("S5", "stage2_balanced_accuracy", "qsvc_q8_r1", "hierarchical_rbf_svm_d8", "Secondary: Q8-R1 vs RBF-SVM d=8 on Stage-2 Bal Acc"),
        ("S6", "stage2_balanced_accuracy", "qsvc_q8_r1", "hierarchical_adapter_d8", "Secondary: Q8-R1 vs Adapter d=8 on Stage-2 Bal Acc"),
        ("A1", "end_to_end_macro_f1", "qsvc_q8_r1", "qsvc_q4_r1", "Ablation: Q8-R1 vs Q4-R1 on Macro-F1"),
        ("A2", "end_to_end_macro_f1", "qsvc_q8_r1", "qsvc_q8_r2", "Ablation: Q8-R1 vs Q8-R2 on Macro-F1"),
    ]

    for comp_id, metric, mA, mB, comp_desc in planned_comparisons:
        sub_A = all_results_df[all_results_df["model"] == mA].sort_values(["speaker_id", "N"])[metric].values
        sub_B = all_results_df[all_results_df["model"] == mB].sort_values(["speaker_id", "N"])[metric].values

        diff = sub_A - sub_B
        n_pairs = len(diff)
        mean_diff = float(np.mean(diff))
        med_diff = float(np.median(diff))
        std_diff = float(np.std(diff, ddof=1)) if n_pairs > 1 else 0.0

        # 95% Confidence Interval (t-based)
        if n_pairs > 1 and std_diff > 1e-12:
            se = std_diff / np.sqrt(n_pairs)
            ci_low = mean_diff - 1.96 * se
            ci_high = mean_diff + 1.96 * se
            cohen_dz = float(mean_diff / std_diff)
        else:
            ci_low = mean_diff
            ci_high = mean_diff
            cohen_dz = 0.0

        # Wilcoxon signed-rank test
        if np.all(diff == 0.0):
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                res_w = stats.wilcoxon(diff, zero_method="wilcox")
                w_stat, p_val = float(res_w.statistic), float(res_w.pvalue)
            except Exception:
                w_stat, p_val = 0.0, 1.0

        direction = "QUANTUM > CLASSICAL" if mean_diff > 0 else ("CLASSICAL > QUANTUM" if mean_diff < 0 else "IDENTICAL")

        stat_records.append({
            "comparison_id": comp_id,
            "description": comp_desc,
            "metric": metric,
            "model_A": mA,
            "model_B": mB,
            "n_pairs": n_pairs,
            "mean_A": round(float(np.mean(sub_A)), 4),
            "mean_B": round(float(np.mean(sub_B)), 4),
            "mean_diff": round(mean_diff, 4),
            "median_diff": round(med_diff, 4),
            "std_diff": round(std_diff, 4),
            "ci_95_lower": round(ci_low, 4),
            "ci_95_upper": round(ci_high, 4),
            "cohens_dz": round(cohen_dz, 4),
            "wilcoxon_stat": round(w_stat, 2),
            "raw_p_value": round(p_val, 6),
            "direction": direction
        })

    # Holm-Bonferroni correction across primary planned comparisons (P1-P4)
    stat_df = pd.DataFrame(stat_records)
    prim_mask = stat_df["comparison_id"].str.startswith("P")
    prim_pvals = stat_df.loc[prim_mask, "raw_p_value"].values
    m_prim = len(prim_pvals)

    sorted_indices = np.argsort(prim_pvals)
    adj_pvals = np.zeros(m_prim)
    for rank, idx in enumerate(sorted_indices):
        mult = m_prim - rank
        adj_pvals[idx] = min(1.0, prim_pvals[idx] * mult)

    # Enforce monotonicity
    for i in range(1, m_prim):
        adj_pvals[sorted_indices[i]] = max(adj_pvals[sorted_indices[i]], adj_pvals[sorted_indices[i-1]])

    stat_df["p_value_adjusted_holm"] = stat_df["raw_p_value"]
    stat_df.loc[prim_mask, "p_value_adjusted_holm"] = np.round(adj_pvals, 6)

    # Classify conclusion
    conclusions = []
    for _, row in stat_df.iterrows():
        p_val = row["p_value_adjusted_holm"]
        m_diff = row["mean_diff"]
        if p_val < 0.05 and m_diff > 0.0:
            conclusions.append("QUANTUM_ADVANTAGE")
        elif p_val < 0.05 and m_diff < 0.0:
            conclusions.append("CLASSICAL_SUPERIOR")
        else:
            conclusions.append("NO_STATISTICAL_DIFFERENCE")
    stat_df["conclusion"] = conclusions

    stat_path = os.path.join(RESULTS_DIR, "phase6_5_5_statistical_tests.csv")
    stat_df.to_csv(stat_path, index=False)
    print(f"Saved {stat_path} ({len(stat_df)} comparisons).")

    # 6. Generate Publication Figures
    generate_figures(all_results_df, stat_df, kernel_diag_df, runtime_df)

    print(f"\n============================================================")
    print(f" SUBPHASE 6.5.5 EXECUTION COMPLETE")
    print(f" Total Conditions: {len(target_splits)}")
    print(f" Total Model Evaluations: {len(all_results_df)}")
    print(f" Total Elapsed Time: {time.time()-t_start_all:.2f}s")
    print(f"============================================================")


def generate_figures(results_df: pd.DataFrame, stat_df: pd.DataFrame, diag_df: pd.DataFrame, runtime_df: pd.DataFrame):
    """Generate all 9 required publication-quality figures."""
    print("\n--- Generating Publication-Quality Figures ---")

    colors = {
        "qsvc_q8_r1": "#7c3aed",       # Purple
        "hierarchical_rbf_svm_d8": "#2563eb",  # Blue
        "hierarchical_adapter_d8": "#059669",  # Green
        "qsvc_q4_r1": "#db2777",       # Pink
        "qsvc_q8_r2": "#d97706",       # Amber
        "hierarchical_rbf_svm_d4": "#64748b",  # Slate
        "hierarchical_adapter_d4": "#0891b2",  # Cyan
    }

    labels = {
        "qsvc_q8_r1": "QSVC Q8-R1 (d=8, q=8, reps=1)",
        "hierarchical_rbf_svm_d8": "Classical RBF-SVM (d=8)",
        "hierarchical_adapter_d8": "Classical Adapter (d=8)",
        "qsvc_q4_r1": "QSVC Q4-R1 (d=4, q=4, reps=1)",
        "qsvc_q8_r2": "QSVC Q8-R2 (d=8, q=8, reps=2)",
        "hierarchical_rbf_svm_d4": "Classical RBF-SVM (d=4)",
        "hierarchical_adapter_d4": "Classical Adapter (d=4)",
    }

    # FIGURE 1: Matched End-to-End Macro-F1 across N
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for m in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8", "qsvc_q4_r1", "qsvc_q8_r2"]:
        sub = results_df[results_df["model"] == m].groupby("N")["end_to_end_macro_f1"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(sub["N"], sub["mean"], yerr=sub["sem"], fmt="-o", label=labels[m], color=colors[m], capsize=4, lw=2)
    ax.set_title("FIGURE 1: Matched End-to-End Macro-F1 across Calibration Budgets", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibration Budget (N)", fontsize=11)
    ax.set_ylabel("Macro-F1 (7 Canonical Classes)", fontsize=11)
    ax.set_xticks([20, 30, 40, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "matched_macro_f1.png")
    plt.savefig(f1_path)
    plt.close()
    print(f"Saved {f1_path}")

    # FIGURE 2: Emergency Recall Comparison (Primary d=8)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for m in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8"]:
        sub = results_df[results_df["model"] == m].groupby("N")["end_to_end_emergency_recall"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(sub["N"], sub["mean"] * 100.0, yerr=sub["sem"] * 100.0, fmt="-s", label=labels[m], color=colors[m], capsize=4, lw=2.2)
    ax.set_title("FIGURE 2: End-to-End Emergency Recall (Primary Matched d=8)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibration Budget (N)", fontsize=11)
    ax.set_ylabel("Emergency Recall (%)", fontsize=11)
    ax.set_xticks([20, 30, 40, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "emergency_recall_comparison.png")
    plt.savefig(f2_path)
    plt.close()
    print(f"Saved {f2_path}")

    # FIGURE 3: False-Emergency Rate Comparison
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for m in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8"]:
        sub = results_df[results_df["model"] == m].groupby("N")["end_to_end_false_emergency_rate"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(sub["N"], sub["mean"] * 100.0, yerr=sub["sem"] * 100.0, fmt="-^", label=labels[m], color=colors[m], capsize=4, lw=2.2)
    ax.set_title("FIGURE 3: End-to-End False-Emergency Rate (Primary Matched d=8)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibration Budget (N)", fontsize=11)
    ax.set_ylabel("False-Emergency Rate / FPR (%)", fontsize=11)
    ax.set_xticks([20, 30, 40, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "false_emergency_rate_comparison.png")
    plt.savefig(f3_path)
    plt.close()
    print(f"Saved {f3_path}")

    # FIGURE 4: Stage-2 Feasible-Class Macro-F1
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for m in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8", "qsvc_q4_r1"]:
        sub = results_df[results_df["model"] == m].groupby("N")["stage2_feasible_macro_f1"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(sub["N"], sub["mean"], yerr=sub["sem"], fmt="-d", label=labels[m], color=colors[m], capsize=4, lw=2)
    ax.set_title("FIGURE 4: Stage-2 Conditional Emergency Subtype Macro-F1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibration Budget (N)", fontsize=11)
    ax.set_ylabel("Stage-2 Feasible-Class Macro-F1", fontsize=11)
    ax.set_xticks([20, 30, 40, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "stage2_macro_f1_comparison.png")
    plt.savefig(f4_path)
    plt.close()
    print(f"Saved {f4_path}")

    # FIGURE 5: Stage-2 Balanced Accuracy
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for m in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8", "qsvc_q4_r1"]:
        sub = results_df[results_df["model"] == m].groupby("N")["stage2_balanced_accuracy"].agg(["mean", "sem"]).reset_index()
        ax.errorbar(sub["N"], sub["mean"] * 100.0, yerr=sub["sem"] * 100.0, fmt="-o", label=labels[m], color=colors[m], capsize=4, lw=2)
    ax.set_title("FIGURE 5: Stage-2 Subtype Balanced Accuracy across Budgets", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibration Budget (N)", fontsize=11)
    ax.set_ylabel("Stage-2 Balanced Accuracy (%)", fontsize=11)
    ax.set_xticks([20, 30, 40, 50])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    f5_path = os.path.join(FIGURES_DIR, "stage2_balanced_accuracy_comparison.png")
    plt.savefig(f5_path)
    plt.close()
    print(f"Saved {f5_path}")

    # FIGURE 6: Per-N Sample Efficiency Bar Chart (Macro-F1)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    budgets = [20, 30, 40, 50]
    bar_width = 0.16
    eval_models = ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8", "qsvc_q4_r1", "qsvc_q8_r2"]
    x = np.arange(len(budgets))
    for i, m in enumerate(eval_models):
        means = [results_df[(results_df["model"] == m) & (results_df["N"] == n)]["end_to_end_macro_f1"].mean() for n in budgets]
        ax.bar(x + (i - 2) * bar_width, means, bar_width, label=labels[m], color=colors[m], alpha=0.9)
    ax.set_title("FIGURE 6: Per-N End-to-End Macro-F1 Sample Efficiency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Calibration Budget (N)", fontsize=11)
    ax.set_ylabel("Mean End-to-End Macro-F1", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={b}" for b in budgets])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=8.5)
    plt.tight_layout()
    f6_path = os.path.join(FIGURES_DIR, "per_n_sample_efficiency.png")
    plt.savefig(f6_path)
    plt.close()
    print(f"Saved {f6_path}")

    # FIGURE 7: Quantum Kernel Eigenvalue / PSD Diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    q8_diags = diag_df[diag_df["qubit_count"] == 8]
    ax1.hist(q8_diags["min_eigenvalue"], bins=15, color="#7c3aed", edgecolor="black", alpha=0.8)
    ax1.axvline(0.0, color="red", linestyle="--", label="PSD Bound (lambda_min >= 0)")
    ax1.set_title("Distribution of Minimum Eigenvalue (q=8)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Minimum Eigenvalue", fontsize=10)
    ax1.set_ylabel("Count of Adaptation Kernels", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.scatter(diag_df["pca_cumulative_variance"] * 100.0, diag_df["condition_number"], c=diag_df["qubit_count"], cmap="viridis", alpha=0.8, edgecolors="k")
    ax2.set_title("Kernel Condition Number vs Retained PCA Variance", fontsize=11, fontweight="bold")
    ax2.set_xlabel("PCA Retained Variance (%)", fontsize=10)
    ax2.set_ylabel("Kernel Condition Number", fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    f7_path = os.path.join(FIGURES_DIR, "quantum_kernel_eigenvalues.png")
    plt.savefig(f7_path)
    plt.close()
    print(f"Saved {f7_path}")

    # FIGURE 8: Runtime Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    r_sub = runtime_df.sort_values("mean_train_time_sec")
    y_pos = np.arange(len(r_sub))
    ax.barh(y_pos, r_sub["mean_train_time_sec"], color="#6366f1", alpha=0.85, label="Mean Adaptation Training Time")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(r_sub["model"])
    ax.set_xlabel("Mean Training Runtime per Condition (seconds)", fontsize=10)
    ax.set_title("FIGURE 8: Computational Runtime per Condition by Model Family", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for i, v in enumerate(r_sub["mean_train_time_sec"]):
        ax.text(v + 0.05, i, f"{v:.3f}s", va="center", fontsize=9)
    plt.tight_layout()
    f8_path = os.path.join(FIGURES_DIR, "runtime_comparison.png")
    plt.savefig(f8_path)
    plt.close()
    print(f"Saved {f8_path}")

    # FIGURE 9: Speaker-Level Comparison for Primary d=8 Models
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    spks = sorted(results_df["speaker_id"].unique())
    x = np.arange(len(spks))
    width = 0.25
    q8_means = [results_df[(results_df["model"] == "qsvc_q8_r1") & (results_df["speaker_id"] == s)]["end_to_end_macro_f1"].mean() for s in spks]
    rbf8_means = [results_df[(results_df["model"] == "hierarchical_rbf_svm_d8") & (results_df["speaker_id"] == s)]["end_to_end_macro_f1"].mean() for s in spks]
    adp8_means = [results_df[(results_df["model"] == "hierarchical_adapter_d8") & (results_df["speaker_id"] == s)]["end_to_end_macro_f1"].mean() for s in spks]

    ax.bar(x - width, q8_means, width, label="QSVC Q8-R1", color="#7c3aed", alpha=0.9)
    ax.bar(x, rbf8_means, width, label="RBF-SVM d=8", color="#2563eb", alpha=0.9)
    ax.bar(x + width, adp8_means, width, label="Adapter d=8", color="#059669", alpha=0.9)
    ax.set_title("FIGURE 9: Per-Speaker End-to-End Macro-F1 (Matched d=8 Models)", fontsize=12, fontweight="bold")
    ax.set_xlabel("TORGO Speaker ID", fontsize=11)
    ax.set_ylabel("Mean End-to-End Macro-F1", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(spks, rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    f9_path = os.path.join(FIGURES_DIR, "speaker_level_comparison.png")
    plt.savefig(f9_path)
    plt.close()
    print(f"Saved {f9_path}")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(pilot=args.pilot)
