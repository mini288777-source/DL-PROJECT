"""
Subphase 6.5.4: Representation Space & PCA Dimension Audit
Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid
         Quantum-Classical Deep Learning for Reliable Emergency Communication

Executes:
1. Canonical Representation Space Audit (Whisper-tiny 384-D, float32, unit-L2 normalized)
2. Train-Only PCA Variance Extraction across candidate dimensions d in {2, 4, 8, 16, 32, 64, 128, 256, 384}
3. Hierarchical Two-Stage Evaluation (Weighted Adapter & Weighted RBF-SVM) across dimensions & budgets N in {20, 30, 40, 50}
4. Master ablation tables:
   - results/phase6_5_4_pca_variance.csv
   - results/phase6_5_4_dimension_ablation.csv
   - results/phase6_5_4_speaker_results.csv
5. Publication figures under reports/phase6_5_4/figures/
"""

import os
import sys
import time
import json
import hashlib
import argparse
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.classical_adapter import ClassicalAdapterClassifier
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.embeddings.cache import load_embedding_file
from src.utils.seeding import set_seed

SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")
WHISPER_DIR = os.path.join(REPO_ROOT, "embeddings", "torgo", "whisper")
BASE_TORGO_DIR = os.path.join(REPO_ROOT, "embeddings", "torgo")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "phase6_5_4", "figures")

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]
DYSARTHRIC_SET = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}
CANDIDATE_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 384]


def compute_iper(y_true: List[str], y_pred: List[str]) -> Tuple[float, int, int]:
    emergency_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != "GENERAL_SPEECH"]
    total_emergency = len(emergency_pairs)
    if total_emergency == 0:
        return 0.0, 0, 0
    preserved_emergency = sum(1 for yt, yp in emergency_pairs if yt == yp)
    return float(preserved_emergency / total_emergency), preserved_emergency, total_emergency


def run_dimension_audit(pilot: bool = False):
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("============================================================")
    print(" SUBPHASE 6.5.4 — REPRESENTATION SPACE & PCA DIMENSION AUDIT")
    print(f" Mode: {'PILOT (3 speakers x 2 N)' if pilot else 'FULL EXPERIMENT (15 speakers x 4 N)'}")
    print(f" Candidate Dimensions: {CANDIDATE_DIMS}")
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

    # 3. Compute PCA explained variance records
    print("\n--- Computing Train-Only PCA Explained Variance Across Dimensions ---")
    pca_variance_records = []
    
    for s in target_splits:
        spk = s["speaker_id"]
        n = s["sample_n"]
        cohort = "dysarthric" if spk in DYSARTHRIC_SET else "control"
        
        adapt_recs = s["adaptation_recording_ids"]
        X_adapt = np.array([embedding_cache[r] for r in adapt_recs])
        max_rank = min(X_adapt.shape[0], X_adapt.shape[1])
        
        # Fit full rank PCA on adaptation data
        pca_full = PCA(n_components=max_rank, random_state=42).fit(X_adapt)
        cum_var_full = np.cumsum(pca_full.explained_variance_ratio_)
        
        for d in CANDIDATE_DIMS:
            if d == 384:
                ev_cum = 1.0
                status = "CANONICAL_FULL_DIM"
            elif d <= max_rank:
                ev_cum = float(cum_var_full[d - 1])
                status = "FEASIBLE_PCA"
            else:
                # d exceeds sample budget rank N
                ev_cum = 1.0  # Captures 100% of available rank-N subspace
                status = "RANK_DEFICIENT_D_EXCEEDS_N"

            pca_variance_records.append({
                "speaker_id": spk,
                "cohort_group": cohort,
                "N": n,
                "dimension": d,
                "effective_rank": min(d, max_rank),
                "sample_budget": n,
                "explained_variance_ratio": round(ev_cum, 6),
                "explained_variance_pct": round(ev_cum * 100.0, 2),
                "status": status
            })

    pca_var_df = pd.DataFrame(pca_variance_records)
    pca_var_path = os.path.join(RESULTS_DIR, "phase6_5_4_pca_variance.csv")
    pca_var_df.to_csv(pca_var_path, index=False)
    print(f"Saved {pca_var_path} ({len(pca_var_df)} rows).")

    # 4. Dimension Ablation Experiment
    print("\n--- Running Hierarchical Models Across Dimensionalities ---")
    speaker_results_records = []
    models = ["hierarchical_adapter", "hierarchical_rbf_svm"]

    t0_all = time.time()
    total_runs = len(target_splits) * len(CANDIDATE_DIMS) * len(models)
    completed_runs = 0

    for s in target_splits:
        spk = s["speaker_id"]
        n = s["sample_n"]
        cohort = "dysarthric" if spk in DYSARTHRIC_SET else "control"

        adapt_recs = s["adaptation_recording_ids"]
        test_recs = s["test_recording_ids"]

        X_adapt_raw = np.array([embedding_cache[r] for r in adapt_recs])
        X_test_raw = np.array([embedding_cache[r] for r in test_recs])

        y_adapt = [rec_intent[r] for r in adapt_recs]
        y_test = [rec_intent[r] for r in test_recs]

        # Stage 1 binary targets: 0 = GENERAL_SPEECH, 1 = EMERGENCY
        y_adapt_s1 = [0 if l == "GENERAL_SPEECH" else 1 for l in y_adapt]
        y_test_s1 = [0 if l == "GENERAL_SPEECH" else 1 for l in y_test]
        y_adapt_s1_str = ["GENERAL_SPEECH" if b == 0 else "EMERGENCY" for b in y_adapt_s1]

        # Stage 2 adaptation subset: emergency only
        emerg_adapt_idx = [i for i, b in enumerate(y_adapt_s1) if b == 1]
        y_emerg_adapt = [y_adapt[i] for i in emerg_adapt_idx]
        feasible_emergency_classes = sorted(list(set(y_emerg_adapt)))
        stage2_num_classes = len(feasible_emergency_classes)

        # Precompute PCA models for valid dimensions on adaptation data
        pca_models = {}
        for d in CANDIDATE_DIMS:
            if d < 384 and d <= len(X_adapt_raw):
                pca_obj = PCA(n_components=d, random_state=42)
                pca_obj.fit(X_adapt_raw)
                pca_models[d] = pca_obj

        for d in CANDIDATE_DIMS:
            is_rank_deficient = (d < 384 and d > len(X_adapt_raw))

            if is_rank_deficient:
                # Document rank-deficient status across both models
                for m_name in models:
                    speaker_results_records.append({
                        "speaker_id": spk,
                        "cohort_group": cohort,
                        "N": n,
                        "model": m_name,
                        "dimension": d,
                        "status": "RANK_DEFICIENT_D_EXCEEDS_N",
                        "explained_variance_pct": 100.0,
                        "stage1_emergency_recall": np.nan,
                        "stage1_emergency_precision": np.nan,
                        "stage1_false_positive_rate": np.nan,
                        "stage1_general_recall": np.nan,
                        "stage1_missed_emergency_rate": np.nan,
                        "stage2_feasible_macro_f1": np.nan,
                        "stage2_all6_macro_f1": np.nan,
                        "stage2_balanced_accuracy": np.nan,
                        "stage2_weighted_f1": np.nan,
                        "end_to_end_iper": np.nan,
                        "end_to_end_emergency_precision": np.nan,
                        "end_to_end_emergency_recall": np.nan,
                        "end_to_end_false_emergency_rate": np.nan,
                        "end_to_end_missed_emergency_rate": np.nan,
                        "end_to_end_macro_f1": np.nan
                    })
                    completed_runs += 1
                continue

            # Apply representation transformation
            if d == 384:
                X_adapt_d = X_adapt_raw
                X_test_d = X_test_raw
                ev_pct = 100.0
            else:
                pca_obj = pca_models[d]
                X_adapt_d = pca_obj.transform(X_adapt_raw)
                X_test_d = pca_obj.transform(X_test_raw)
                ev_pct = round(float(np.sum(pca_obj.explained_variance_ratio_)) * 100.0, 2)

            # Stage 2 emergency slice
            X_emerg_adapt_d = X_adapt_d[emerg_adapt_idx]

            for m_name in models:
                t_run0 = time.perf_counter()
                
                # ---------------- STAGE 1 TRAINING ----------------
                if m_name == "hierarchical_adapter":
                    stage1_model = ClassicalAdapterClassifier(
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
                    stage1_model.fit(X_adapt_d, y_adapt_s1_str)
                    probs_all = stage1_model.predict_proba(X_test_d)
                    emerg_col = stage1_model.classes_.index("EMERGENCY")
                    p_emerg = probs_all[:, emerg_col]
                else:
                    stage1_model = ClassicalRBFSVMClassifier(
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        standardize=False,
                        probability=True,
                        seed=42,
                        all_classes=["GENERAL_SPEECH", "EMERGENCY"]
                    )
                    stage1_model.fit(X_adapt_d, y_adapt_s1_str)
                    probs_all = stage1_model.predict_proba(X_test_d)
                    emerg_col = stage1_model.classes_.index("EMERGENCY")
                    p_emerg = probs_all[:, emerg_col]

                # Decision at threshold 0.50
                s1_pred_bin = [1 if p >= 0.50 else 0 for p in p_emerg]
                s1_rec = float(recall_score(y_test_s1, s1_pred_bin, pos_label=1, zero_division=0))
                s1_prec = float(precision_score(y_test_s1, s1_pred_bin, pos_label=1, zero_division=0))
                s1_spec = float(recall_score(y_test_s1, s1_pred_bin, pos_label=0, zero_division=0))
                s1_fpr = 1.0 - s1_spec
                s1_mer = 1.0 - s1_rec

                # ---------------- STAGE 2 TRAINING ----------------
                is_single_class = (stage2_num_classes == 1)
                single_class_name = feasible_emergency_classes[0] if is_single_class else None

                if is_single_class:
                    stage2_model = None
                else:
                    if m_name == "hierarchical_adapter":
                        stage2_model = ClassicalAdapterClassifier(
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
                        stage2_model.fit(X_emerg_adapt_d, y_emerg_adapt)
                    else:
                        stage2_model = ClassicalRBFSVMClassifier(
                            C=1.0,
                            gamma="scale",
                            class_weight="balanced",
                            standardize=False,
                            probability=True,
                            seed=42,
                            all_classes=feasible_emergency_classes
                        )
                        stage2_model.fit(X_emerg_adapt_d, y_emerg_adapt)

                # ---------------- STAGE 2 DIAGNOSTIC EVALUATION ----------------
                true_emerg_idx = [i for i, l in enumerate(y_test) if l != "GENERAL_SPEECH"]
                if len(true_emerg_idx) > 0:
                    X_true_emerg_d = X_test_d[true_emerg_idx]
                    y_true_emerg = [y_test[i] for i in true_emerg_idx]
                    
                    if is_single_class:
                        s2_diag_preds = [single_class_name] * len(X_true_emerg_d)
                    else:
                        s2_diag_preds = stage2_model.predict(X_true_emerg_d)

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
                routed_idx = [i for i, b in enumerate(s1_pred_bin) if b == 1]
                if len(routed_idx) > 0:
                    X_routed_d = X_test_d[routed_idx]
                    if is_single_class:
                        s2_routed_preds = [single_class_name] * len(X_routed_d)
                    else:
                        s2_routed_preds = stage2_model.predict(X_routed_d)
                else:
                    s2_routed_preds = []

                routed_map = dict(zip(routed_idx, s2_routed_preds))
                final_preds = [routed_map[i] if s1_pred_bin[i] == 1 else "GENERAL_SPEECH" for i in range(len(y_test))]

                # End-to-end metrics
                e2e_macro_f1 = float(f1_score(y_test, final_preds, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
                iper, corr_emg, tot_emg = compute_iper(y_test, final_preds)

                pred_emerg_total = sum(1 for yp in final_preds if yp in EMERGENCY_CLASSES)
                e2e_prec = float(corr_emg / pred_emerg_total) if pred_emerg_total > 0 else 0.0
                e2e_rec = iper

                gen_total = sum(1 for yt in y_test if yt == "GENERAL_SPEECH")
                false_emerg = sum(1 for yt, yp in zip(y_test, final_preds) if yt == "GENERAL_SPEECH" and yp in EMERGENCY_CLASSES)
                e2e_fer = float(false_emerg / gen_total) if gen_total > 0 else 0.0
                e2e_mer = 1.0 - e2e_rec

                speaker_results_records.append({
                    "speaker_id": spk,
                    "cohort_group": cohort,
                    "N": n,
                    "model": m_name,
                    "dimension": d,
                    "status": "SUCCESS",
                    "explained_variance_pct": ev_pct,
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
                    "end_to_end_emergency_precision": round(e2e_prec, 4),
                    "end_to_end_emergency_recall": round(e2e_rec, 4),
                    "end_to_end_false_emergency_rate": round(e2e_fer, 4),
                    "end_to_end_missed_emergency_rate": round(e2e_mer, 4),
                    "end_to_end_macro_f1": round(e2e_macro_f1, 4)
                })
                completed_runs += 1

        print(f"Completed {spk} N={n} across all dimensions ({completed_runs}/{total_runs}).")

    print(f"\nExecution finished in {time.time() - t0_all:.2f} s.")

    # 5. Save Granular Speaker Results
    spk_df = pd.DataFrame(speaker_results_records)
    spk_path = os.path.join(RESULTS_DIR, "phase6_5_4_speaker_results.csv")
    spk_df.to_csv(spk_path, index=False)
    print(f"Saved {spk_path} ({len(spk_df)} rows).")

    # 6. Aggregate Dimension Ablation Table
    ablation_records = []
    for (m_name, d_val, n_val), grp in spk_df.groupby(["model", "dimension", "N"]):
        status = grp["status"].iloc[0]
        if status == "SUCCESS":
            ablation_records.append({
                "model": m_name,
                "dimension": d_val,
                "N": n_val,
                "status": "SUCCESS",
                "mean_explained_variance_pct": round(grp["explained_variance_pct"].mean(), 2),
                "stage1_emergency_recall": round(grp["stage1_emergency_recall"].mean(), 4),
                "stage1_emergency_precision": round(grp["stage1_emergency_precision"].mean(), 4),
                "stage1_false_emergency_rate": round(grp["stage1_false_positive_rate"].mean(), 4),
                "stage1_general_speech_recall": round(grp["stage1_general_recall"].mean(), 4),
                "stage1_missed_emergency_rate": round(grp["stage1_missed_emergency_rate"].mean(), 4),
                "stage2_feasible_macro_f1": round(grp["stage2_feasible_macro_f1"].mean(), 4),
                "stage2_all6_macro_f1": round(grp["stage2_all6_macro_f1"].mean(), 4),
                "stage2_balanced_accuracy": round(grp["stage2_balanced_accuracy"].mean(), 4),
                "stage2_weighted_f1": round(grp["stage2_weighted_f1"].mean(), 4),
                "end_to_end_iper": round(grp["end_to_end_iper"].mean(), 4),
                "end_to_end_emergency_precision": round(grp["end_to_end_emergency_precision"].mean(), 4),
                "end_to_end_emergency_recall": round(grp["end_to_end_emergency_recall"].mean(), 4),
                "end_to_end_false_emergency_rate": round(grp["end_to_end_false_emergency_rate"].mean(), 4),
                "end_to_end_missed_emergency_rate": round(grp["end_to_end_missed_emergency_rate"].mean(), 4),
                "end_to_end_macro_f1": round(grp["end_to_end_macro_f1"].mean(), 4)
            })
        else:
            ablation_records.append({
                "model": m_name,
                "dimension": d_val,
                "N": n_val,
                "status": "RANK_DEFICIENT_D_EXCEEDS_N",
                "mean_explained_variance_pct": 100.0,
                "stage1_emergency_recall": np.nan,
                "stage1_emergency_precision": np.nan,
                "stage1_false_emergency_rate": np.nan,
                "stage1_general_speech_recall": np.nan,
                "stage1_missed_emergency_rate": np.nan,
                "stage2_feasible_macro_f1": np.nan,
                "stage2_all6_macro_f1": np.nan,
                "stage2_balanced_accuracy": np.nan,
                "stage2_weighted_f1": np.nan,
                "end_to_end_iper": np.nan,
                "end_to_end_emergency_precision": np.nan,
                "end_to_end_emergency_recall": np.nan,
                "end_to_end_false_emergency_rate": np.nan,
                "end_to_end_missed_emergency_rate": np.nan,
                "end_to_end_macro_f1": np.nan
            })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_path = os.path.join(RESULTS_DIR, "phase6_5_4_dimension_ablation.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print(f"Saved {ablation_path} ({len(ablation_df)} rows).")

    # 7. Generate Publication Figures
    print("\n--- Generating Publication Figures ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Explained variance vs PCA dimension
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for n_val in sorted(pca_var_df["N"].unique()):
        sub_var = pca_var_df[pca_var_df["N"] == n_val]
        mean_var = sub_var.groupby("dimension")["explained_variance_pct"].mean()
        valid_dims = [d for d in CANDIDATE_DIMS if (d <= n_val or d == 384) and d in mean_var.index]
        ax.plot([str(d) for d in valid_dims], [mean_var[d] for d in valid_dims],
                marker="o", linewidth=2.0, markersize=7, label=f"N={n_val}")
    ax.set_title("Train-Only PCA Cumulative Explained Variance vs. Dimension", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Representation Dimension (d)", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
    ax.set_ylim([50, 105])
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "explained_variance_vs_dimension.png"))
    plt.close(fig)

    # Figure 2: End-to-end Macro-F1 vs PCA dimension
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)
    for ax_curr, m_name, title in [(ax1, "hierarchical_adapter", "Hierarchical Adapter"),
                                   (ax2, "hierarchical_rbf_svm", "Hierarchical RBF-SVM")]:
        sub_m = ablation_df[(ablation_df["model"] == m_name) & (ablation_df["status"] == "SUCCESS")]
        for n_val in sorted(sub_m["N"].unique()):
            sub_n = sub_m[sub_m["N"] == n_val].sort_values("dimension")
            ax_curr.plot([str(d) for d in sub_n["dimension"]], sub_n["end_to_end_macro_f1"],
                         marker="s" if "rbf" in m_name else "o", linewidth=2.0, markersize=6, label=f"N={n_val}")
        ax_curr.set_title(title, fontsize=12, fontweight="bold")
        ax_curr.set_xlabel("Representation Dimension (d)", fontsize=11)
        if ax_curr == ax1:
            ax_curr.set_ylabel("End-to-End Macro-F1", fontsize=11)
        ax_curr.legend(frameon=True, fontsize=9)
    fig.suptitle("End-to-End Macro-F1 across Retained Dimensionality", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "macro_f1_vs_dimension.png"))
    plt.close(fig)

    # Figure 3: Emergency recall vs PCA dimension
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)
    for ax_curr, m_name, title in [(ax1, "hierarchical_adapter", "Hierarchical Adapter"),
                                   (ax2, "hierarchical_rbf_svm", "Hierarchical RBF-SVM")]:
        sub_m = ablation_df[(ablation_df["model"] == m_name) & (ablation_df["status"] == "SUCCESS")]
        for n_val in sorted(sub_m["N"].unique()):
            sub_n = sub_m[sub_m["N"] == n_val].sort_values("dimension")
            ax_curr.plot([str(d) for d in sub_n["dimension"]], sub_n["end_to_end_iper"] * 100,
                         marker="o", linewidth=2.0, markersize=6, label=f"N={n_val}")
        ax_curr.set_title(title, fontsize=12, fontweight="bold")
        ax_curr.set_xlabel("Representation Dimension (d)", fontsize=11)
        if ax_curr == ax1:
            ax_curr.set_ylabel("Emergency Recall / IPER (%)", fontsize=11)
        ax_curr.legend(frameon=True, fontsize=9)
    fig.suptitle("Emergency Recall (IPER) across Representation Dimensionality", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "emergency_recall_vs_dimension.png"))
    plt.close(fig)

    # Figure 4: False-emergency rate vs PCA dimension
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)
    for ax_curr, m_name, title in [(ax1, "hierarchical_adapter", "Hierarchical Adapter"),
                                   (ax2, "hierarchical_rbf_svm", "Hierarchical RBF-SVM")]:
        sub_m = ablation_df[(ablation_df["model"] == m_name) & (ablation_df["status"] == "SUCCESS")]
        for n_val in sorted(sub_m["N"].unique()):
            sub_n = sub_m[sub_m["N"] == n_val].sort_values("dimension")
            ax_curr.plot([str(d) for d in sub_n["dimension"]], sub_n["end_to_end_false_emergency_rate"] * 100,
                         marker="^", linewidth=2.0, markersize=6, label=f"N={n_val}")
        ax_curr.set_title(title, fontsize=12, fontweight="bold")
        ax_curr.set_xlabel("Representation Dimension (d)", fontsize=11)
        if ax_curr == ax1:
            ax_curr.set_ylabel("False Emergency Rate (%)", fontsize=11)
        ax_curr.legend(frameon=True, fontsize=9)
    fig.suptitle("False Emergency Rate across Representation Dimensionality", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "false_emergency_rate_vs_dimension.png"))
    plt.close(fig)

    # Figure 5: Stage-2 feasible-class Macro-F1 vs PCA dimension
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for m_name in models:
        sub_m = ablation_df[(ablation_df["model"] == m_name) & (ablation_df["status"] == "SUCCESS")]
        mean_d = sub_m.groupby("dimension")["stage2_feasible_macro_f1"].mean()
        dims = sorted(mean_d.index)
        label = "Hierarchical Adapter" if "adapter" in m_name else "Hierarchical RBF-SVM"
        color = "#1f77b4" if "adapter" in m_name else "#d62728"
        ax.plot([str(d) for d in dims], [mean_d[d] for d in dims],
                marker="o", linewidth=2.2, markersize=7, label=label, color=color)
    ax.set_title("Stage-2 Conditional Subtype Feasible Macro-F1 vs. Dimension", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Representation Dimension (d)", fontsize=11)
    ax.set_ylabel("Feasible Macro-F1 (Evaluated on Emergency Speech)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "stage2_macro_f1_vs_dimension.png"))
    plt.close(fig)

    # Figure 6: Heatmap: PCA dimension x N for primary end-to-end metric (Macro-F1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    for ax_curr, m_name, title in [(ax1, "hierarchical_adapter", "Adapter: Macro-F1 Heatmap"),
                                   (ax2, "hierarchical_rbf_svm", "RBF-SVM: Macro-F1 Heatmap")]:
        sub_m = ablation_df[(ablation_df["model"] == m_name) & (ablation_df["status"] == "SUCCESS")]
        pivot = sub_m.pivot(index="N", columns="dimension", values="end_to_end_macro_f1")
        # Ensure dimensions ordered
        cols = [c for c in CANDIDATE_DIMS if c in pivot.columns]
        pivot = pivot[cols]
        
        im = ax_curr.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        ax_curr.set_title(title, fontsize=12, fontweight="bold")
        ax_curr.set_xticks(range(len(cols)))
        ax_curr.set_xticklabels([str(c) for c in cols], fontsize=9)
        ax_curr.set_yticks(range(len(pivot.index)))
        ax_curr.set_yticklabels([f"N={n}" for n in pivot.index], fontsize=10)
        ax_curr.set_xlabel("Dimension (d)", fontsize=10)
        ax_curr.set_ylabel("Budget (N)", fontsize=10)
        
        # Annotate values
        for i in range(len(pivot.index)):
            for j in range(len(cols)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    ax_curr.text(j, i, f"{val:.3f}", ha="center", va="center",
                                 color="black" if val < 0.10 else "white", fontsize=8.5, fontweight="bold")
        fig.colorbar(im, ax=ax_curr, fraction=0.046, pad=0.04)
        
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "heatmap_dimension_vs_N.png"))
    plt.close(fig)

    # Figure 7: Dimensionality vs explained variance and downstream performance
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)
    sub_succ = ablation_df[ablation_df["status"] == "SUCCESS"]
    dim_grp = sub_succ.groupby("dimension").mean(numeric_only=True)
    dims_sorted = sorted(dim_grp.index)
    
    color = "#1f77b4"
    ax1.set_xlabel("Representation Dimension (d)", fontsize=11)
    ax1.set_ylabel("Cumulative Explained Variance (%)", color=color, fontsize=11)
    ax1.plot([str(d) for d in dims_sorted], [dim_grp.loc[d, "mean_explained_variance_pct"] for d in dims_sorted],
             color=color, marker="o", linewidth=2.2, markersize=7, label="Explained Variance (%)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim([50, 105])

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("End-to-End Macro-F1", color=color2, fontsize=11)
    ax2.plot([str(d) for d in dims_sorted], [dim_grp.loc[d, "end_to_end_macro_f1"] for d in dims_sorted],
             color=color2, marker="s", linewidth=2.2, markersize=7, label="End-to-End Macro-F1")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Representation Dimensionality vs. Explained Variance and Downstream Macro-F1",
                  fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dimensionality_vs_variance_performance.png"))
    plt.close(fig)

    print("All 7 figures generated successfully in reports/phase6_5_4/figures/.")
    print("=== Subphase 6.5.4 Dimension Ablation Finished Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 3-speaker pilot")
    args = parser.parse_args()
    run_dimension_audit(pilot=args.pilot)
