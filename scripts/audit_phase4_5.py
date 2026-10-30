"""
Audit and hardening analysis script for Phase 4.5.
Conducts comprehensive verification of:
1. 240 conditions integrity
2. Prediction distribution and majority-class collapse analysis
3. Single-class conditions verification
4. Kernel matrix numerical quality and Mercer condition revalidation
5. Classical vs Quantum three-way fairness audit
6. Prediction lineage and IPER readiness
7. Minority-intent diagnostic feasibility study
8. Artifact SHA-256 fingerprinting
"""

import os
import sys
import glob
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

WORKSPACE_DIR = Path("d:/DL")
RESULTS_DIR = WORKSPACE_DIR / "results"
PHASE4_DIR = RESULTS_DIR / "phase4"
REPORTS_DIR = WORKSPACE_DIR / "reports"
SPLITS_JSON = WORKSPACE_DIR / "splits" / "torgo_splits.json"
MANIFEST_CSV = WORKSPACE_DIR / "data" / "manifests" / "torgo_manifest.csv"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_conditions_and_predictions():
    curves_path = PHASE4_DIR / "quantum_curves.csv"
    preds_path = PHASE4_DIR / "quantum_predictions.csv"
    
    assert curves_path.exists(), f"Missing {curves_path}"
    assert preds_path.exists(), f"Missing {preds_path}"
    
    df_curves = pd.read_csv(curves_path)
    df_preds = pd.read_csv(preds_path)
    
    print(f"[1] Loaded {len(df_curves)} rows from quantum_curves.csv")
    print(f"[1] Loaded {len(df_preds)} rows from quantum_predictions.csv")
    
    speakers = sorted(df_curves["speaker_id"].unique())
    n_values = sorted(df_curves["sample_n"].unique())
    qubit_values = sorted(df_curves["qubit_count"].unique())
    
    print(f"Speakers ({len(speakers)}): {speakers}")
    print(f"N values ({len(n_values)}): {n_values}")
    print(f"Qubit values ({len(qubit_values)}): {qubit_values}")
    
    # 1. Condition Integrity Table
    integrity_rows = []
    expected_conditions = 15 * 4 * 4 # 240
    
    kernel_cache_dir = PHASE4_DIR / "kernel_cache"
    
    for spk in speakers:
        for n in n_values:
            for q in qubit_values:
                cond_id = f"phase4_{spk}_qsvc_q{q}_N{n}_s42"
                match = df_curves[(df_curves["speaker_id"] == spk) & (df_curves["sample_n"] == n) & (df_curves["qubit_count"] == q)]
                if len(match) == 1:
                    status = match.iloc[0]["status"]
                    has_metrics = not pd.isna(match.iloc[0]["macro_f1"])
                    is_single_class = bool(match.iloc[0]["single_class_training"])
                else:
                    status = "MISSING"
                    has_metrics = False
                    is_single_class = False
                    
                preds_match = df_preds[(df_preds["speaker_id"] == spk) & (df_preds["sample_n"] == n) & (df_preds["qubit_count"] == q)]
                pred_count = len(preds_match)
                
                # Check cache files
                train_k = kernel_cache_dir / f"{spk}_N{n}_q{q}_r1_train.npy"
                test_k = kernel_cache_dir / f"{spk}_N{n}_q{q}_r1_test.npy"
                cache_present = train_k.exists() and test_k.exists()
                
                integrity_rows.append({
                    "speaker_id": spk,
                    "N": n,
                    "qubits": q,
                    "condition_id": cond_id,
                    "status": status,
                    "single_class_training": is_single_class,
                    "prediction_count": pred_count,
                    "kernel_cache_present": cache_present,
                    "metrics_present": has_metrics
                })
                
    df_integrity = pd.DataFrame(integrity_rows)
    df_integrity.to_csv(REPORTS_DIR / "phase4_5_condition_integrity.csv", index=False)
    print(f"[+] Saved reports/phase4_5_condition_integrity.csv ({len(df_integrity)} rows)")
    
    # 2. Prediction distribution
    total_preds = len(df_preds)
    pred_counts = df_preds["predicted_intent"].value_counts()
    print("\n--- OVERALL PREDICTION DISTRIBUTION ---")
    for k, v in pred_counts.items():
        print(f"  {k:20s}: {v:6d} ({v/total_preds*100:6.2f}%)")
        
    true_counts = df_preds["true_intent"].value_counts()
    print("\n--- OVERALL TRUE LABEL DISTRIBUTION IN TEST ---")
    for k, v in true_counts.items():
        print(f"  {k:20s}: {v:6d} ({v/total_preds*100:6.2f}%)")
        
    # Per-speaker, per-N, per-qubit distribution
    dist_rows = []
    all_classes = sorted(list(set(df_preds["predicted_intent"].unique()) | set(df_preds["true_intent"].unique())))
    
    for (spk, n, q), group in df_preds.groupby(["speaker_id", "sample_n", "qubit_count"]):
        n_group = len(group)
        p_counts = group["predicted_intent"].value_counts().to_dict()
        gen_speech_p = p_counts.get("GENERAL_SPEECH", 0) / n_group * 100.0
        row = {
            "speaker_id": spk,
            "N": n,
            "qubits": q,
            "total_test_samples": n_group,
            "pct_general_speech_pred": gen_speech_p
        }
        for cls_name in all_classes:
            row[f"pred_{cls_name}"] = p_counts.get(cls_name, 0)
        dist_rows.append(row)
        
    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(REPORTS_DIR / "phase4_5_prediction_distribution.csv", index=False)
    print(f"[+] Saved reports/phase4_5_prediction_distribution.csv ({len(df_dist)} rows)")
    
    # 3. Lineage and IPER readiness check
    print("\n--- LINEAGE & IPER READINESS AUDIT ---")
    manifest_df = pd.read_csv(MANIFEST_CSV)
    manifest_rec_ids = set(manifest_df["recording_id"].unique())
    pred_rec_ids = set(df_preds["recording_id"].unique())
    
    missing_in_manifest = pred_rec_ids - manifest_rec_ids
    print(f"Prediction recordings missing in manifest: {len(missing_in_manifest)}")
    assert len(missing_in_manifest) == 0, f"Found orphan predictions: {missing_in_manifest}"
    
    # Check required fields for IPER
    iper_required = ["recording_id", "speaker_id", "sample_n", "true_intent", "predicted_intent", "reference_transcript", "split"]
    missing_iper_cols = [c for c in iper_required if c not in df_preds.columns]
    print(f"Missing IPER columns: {missing_iper_cols}")
    
    # 4. Single-Class Audit
    single_class_rows = df_curves[df_curves["single_class_training"] == True]
    print(f"\n--- SINGLE-CLASS AUDIT ---")
    print(f"Number of single-class conditions in Phase 4: {len(single_class_rows)}")
    for _, r in single_class_rows.iterrows():
        print(f"  {r['speaker_id']} N={r['sample_n']} q={r['qubit_count']}: status={r['status']}, notes={r['failure_reason']}")
        
    return df_curves, df_preds, df_integrity

def audit_kernel_matrices():
    kernel_cache_dir = PHASE4_DIR / "kernel_cache"
    train_kernels = sorted(glob.glob(str(kernel_cache_dir / "*_r1_train.npy")))
    print(f"\n--- KERNEL REVALIDATION AUDIT ---")
    print(f"Found {len(train_kernels)} train kernel cache files.")
    
    reval_rows = []
    max_sym_err = 0.0
    max_diag_err = 0.0
    min_eig_overall = 1.0
    all_psd = True
    
    for fpath in train_kernels:
        fname = Path(fpath).name
        # format: {spk}_N{n}_q{q}_r1_train.npy
        parts = fname.replace("_r1_train.npy", "").split("_")
        spk = parts[0]
        n_val = int(parts[1][1:])
        q_val = int(parts[2][1:])
        
        K = np.load(fpath)
        N = K.shape[0]
        
        # Symmetry
        sym_err = float(np.max(np.abs(K - K.T)))
        diag_err = float(np.max(np.abs(np.diag(K) - 1.0)))
        
        # Eigenvalues
        eigs = np.linalg.eigvalsh(K)
        min_eig = float(np.min(eigs))
        max_eig = float(np.max(eigs))
        trace = float(np.trace(K))
        cond_num = float(max_eig / max(min_eig, 1e-15))
        
        # Off-diagonal elements
        mask = ~np.eye(N, dtype=bool)
        off_diag = K[mask]
        mean_off = float(np.mean(off_diag)) if len(off_diag) > 0 else 1.0
        
        is_psd = min_eig >= -1e-10
        if not is_psd:
            all_psd = False
            
        max_sym_err = max(max_sym_err, sym_err)
        max_diag_err = max(max_diag_err, diag_err)
        min_eig_overall = min(min_eig_overall, min_eig)
        
        reval_rows.append({
            "speaker_id": spk,
            "N": n_val,
            "qubits": q_val,
            "matrix_dim": N,
            "sym_error": sym_err,
            "diag_error": diag_err,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "trace": trace,
            "condition_number": cond_num,
            "mean_off_diagonal": mean_off,
            "is_psd": is_psd
        })
        
    print(f"Max symmetry error across all kernels: {max_sym_err:.2e}")
    print(f"Max diagonal error across all kernels: {max_diag_err:.2e}")
    print(f"Min eigenvalue across all kernels:     {min_eig_overall:.6f}")
    print(f"All matrices strictly positive semi-definite: {all_psd}")
    
    return pd.DataFrame(reval_rows)

def audit_three_way_fairness():
    phase3_curves_path = RESULTS_DIR / "phase3" / "baseline_curves.csv"
    phase4_curves_path = PHASE4_DIR / "quantum_curves.csv"
    
    if not phase3_curves_path.exists():
        print(f"[!] Warning: {phase3_curves_path} not found.")
        return None
        
    df_p3 = pd.read_csv(phase3_curves_path)
    df_p4 = pd.read_csv(phase4_curves_path)
    
    df_q4 = df_p4[df_p4["qubit_count"] == 4]
    
    methods_p3 = sorted(df_p3["method"].unique())
    print(f"\n--- THREE-WAY FAIRNESS AUDIT ---")
    print(f"Phase 3 models: {methods_p3}")
    
    # Check alignment for each speaker and N
    speakers = sorted(df_q4["speaker_id"].unique())
    n_values = sorted(df_q4["sample_n"].unique())
    
    comparison_checks = []
    for spk in speakers:
        for n in n_values:
            p3_adapter = df_p3[(df_p3["speaker_id"] == spk) & (df_p3["sample_n"] == n) & (df_p3["method"] == "adapter")]
            p3_rbf = df_p3[(df_p3["speaker_id"] == spk) & (df_p3["sample_n"] == n) & (df_p3["method"] == "rbf_svm")]
            p4_qsvc = df_q4[(df_q4["speaker_id"] == spk) & (df_q4["sample_n"] == n)]
            
            comparison_checks.append({
                "speaker_id": spk,
                "N": n,
                "has_adapter": len(p3_adapter) == 1,
                "has_rbf": len(p3_rbf) == 1,
                "has_qsvc": len(p4_qsvc) == 1,
                "adapter_f1": p3_adapter.iloc[0]["macro_f1"] if len(p3_adapter) == 1 else None,
                "rbf_f1": p3_rbf.iloc[0]["macro_f1"] if len(p3_rbf) == 1 else None,
                "qsvc_f1": p4_qsvc.iloc[0]["macro_f1"] if len(p4_qsvc) == 1 else None,
            })
            
    df_comp = pd.DataFrame(comparison_checks)
    print(f"Checked {len(df_comp)} speaker-N conditions across all 3 models.")
    all_aligned = bool((df_comp["has_adapter"] & df_comp["has_rbf"] & df_comp["has_qsvc"]).all())
    print(f"All 60 speaker-N conditions perfectly present across Adapter, RBF, and QSVC: {all_aligned}")
    
    return df_comp

def feasibility_diagnostic_study():
    print(f"\n--- MINORITY-INTENT DIAGNOSTIC FEASIBILITY STUDY ---")
    with open(SPLITS_JSON, "r") as f:
        splits_list = json.load(f)
        
    print(f"Loaded {len(splits_list)} splits from {SPLITS_JSON}.")
    manifest_df = pd.read_csv(MANIFEST_CSV).set_index("recording_id")
    
    feasibility_data = []
    for s_dict in splits_list:
        spk = s_dict["speaker_id"]
        n_val = s_dict["sample_n"]
        train_ids = s_dict["adaptation_recording_ids"]
        test_ids = s_dict["test_recording_ids"]
        
        train_intents = [manifest_df.loc[rid]["intent"] for rid in train_ids if rid in manifest_df.index]
        test_intents = [manifest_df.loc[rid]["intent"] for rid in test_ids if rid in manifest_df.index]
        
        t_counts = Counter(train_intents)
        te_counts = Counter(test_intents)
        
        minority_intents_train = {k: v for k, v in t_counts.items() if k != "GENERAL_SPEECH"}
        total_minority_train = sum(minority_intents_train.values())
        
        minority_intents_test = {k: v for k, v in te_counts.items() if k != "GENERAL_SPEECH"}
        total_minority_test = sum(minority_intents_test.values())
        
        feasibility_data.append({
            "speaker_id": spk,
            "N": n_val,
            "train_size": len(train_intents),
            "test_size": len(test_intents),
            "train_minority_count": total_minority_train,
            "train_minority_breakdown": str(minority_intents_train),
            "test_minority_count": total_minority_test,
            "test_minority_breakdown": str(minority_intents_test),
            "train_unique_classes": len(t_counts),
            "test_unique_classes": len(te_counts)
        })
        
    df_feas = pd.DataFrame(feasibility_data)
    print("\nSummary of adaptation minority counts across 60 splits:")
    print(df_feas.groupby("N")[["train_minority_count", "train_unique_classes"]].describe())
    
    print("\nSplits with >= 3 distinct classes in train:")
    multiclass_splits = df_feas[df_feas["train_unique_classes"] >= 3]
    print(multiclass_splits[["speaker_id", "N", "train_unique_classes", "train_minority_breakdown"]])
    
    return df_feas

def fingerprint_artifacts():
    print(f"\n--- ARTIFACT CHECKSUM FINGERPRINTING ---")
    files_to_hash = [
        PHASE4_DIR / "quantum_curves.csv",
        PHASE4_DIR / "quantum_predictions.csv",
        PHASE4_DIR / "quantum_aggregated.csv",
        PHASE4_DIR / "ablation_table.csv",
        PHASE4_DIR / "classical_vs_quantum_comparison.csv",
        WORKSPACE_DIR / "configs" / "phase4_quantum.yaml",
        REPORTS_DIR / "phase4_complete_results.md",
        REPORTS_DIR / "phase4_kernel_validation.md",
        REPORTS_DIR / "phase4_ablation_report.md"
    ]
    
    hashes = []
    for fp in files_to_hash:
        if fp.exists():
            h = compute_sha256(fp)
            size = fp.stat().st_size
            rel = fp.relative_to(WORKSPACE_DIR)
            hashes.append({
                "file": str(rel).replace("\\", "/"),
                "size_bytes": size,
                "sha256": h
            })
            print(f"  {str(rel):45s} ({size:10d} B): {h}")
        else:
            print(f"  {str(fp):45s}: NOT FOUND")
            
    return pd.DataFrame(hashes)

if __name__ == "__main__":
    df_curves, df_preds, df_integrity = audit_conditions_and_predictions()
    df_reval = audit_kernel_matrices()
    df_comp = audit_three_way_fairness()
    df_feas = feasibility_diagnostic_study()
    df_hashes = fingerprint_artifacts()
    print("\nPhase 4.5 audit data collection completed successfully.")
