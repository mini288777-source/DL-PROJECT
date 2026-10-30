"""
Subphase 6.5.3-C: Hierarchical Evaluation Closure Analysis Script
Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid
         Quantum-Classical Deep Learning for Reliable Emergency Communication

Performs:
1. Stage-2 Conditional Emergency Analysis (on genuine emergency test utterances only)
   -> phase6_5/hierarchical/analysis/stage2_conditional_metrics.csv
   -> phase6_5/hierarchical/analysis/stage2_per_class_metrics.csv
2. End-to-End Error Taxonomy (A: False Emergency, B: Missed Emergency, C: Wrong Type, D: Correct Emergency, E: Correct General)
   -> phase6_5/hierarchical/analysis/error_taxonomy.csv
   -> phase6_5/hierarchical/analysis/error_taxonomy_summary.csv
3. Exploratory Secondary Threshold Analysis
   -> phase6_5/hierarchical/results/threshold_analysis.csv (enriched)
   -> phase6_5/hierarchical/analysis/threshold_analysis_summary.csv
4. Speaker-Level Error Analysis
   -> phase6_5/hierarchical/analysis/speaker_error_analysis.csv
5. Sample-Efficiency Closure
   -> phase6_5/hierarchical/analysis/sample_efficiency_closure.csv
6. Matched Flat vs Hierarchical Comparison
   -> phase6_5/hierarchical/analysis/flat_vs_hierarchical_matched.csv
7. Publication Figures Generation
   -> phase6_5/hierarchical/figures/
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "results")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "analysis")
FIGURES_DIR = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "figures")
REPORTS_DIR = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "reports")
SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")
FLAT_RESULTS_PATH = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "results", "classical_rescue_results.csv")

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]


def run_closure():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=== Starting Subphase 6.5.3-C Hierarchical Closure Analysis ===")

    # 1. Load data
    results_df = pd.read_csv(os.path.join(RESULTS_DIR, "hierarchical_results.csv"))
    hier_preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "hierarchical_predictions.csv"))
    s1_preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "stage1_predictions.csv"))
    s2_preds_df = pd.read_csv(os.path.join(RESULTS_DIR, "stage2_predictions.csv"))
    thresh_df = pd.read_csv(os.path.join(RESULTS_DIR, "threshold_analysis.csv"))

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_track_b = json.load(f)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    rec_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))

    # Map (speaker_id, sample_n) -> feasible emergency classes from adaptation set ONLY
    split_feasible_map = {}
    for s in splits_track_b:
        adapt_labels = [rec_intent[r] for r in s["adaptation_recording_ids"]]
        feas_emerg = sorted(list(set([l for l in adapt_labels if l != "GENERAL_SPEECH"])))
        split_feasible_map[(s["speaker_id"], s["sample_n"])] = feas_emerg

    # ============================================================
    # 1. STAGE-2 CONDITIONAL EMERGENCY ANALYSIS
    # ============================================================
    print("\n--- 1. Computing Stage-2 Conditional Emergency Metrics ---")
    s2_cond_records = []
    s2_per_class_records = []

    s2_grouped = s2_preds_df.groupby("experiment_id")

    for idx, row in results_df.iterrows():
        exp_id = row["experiment_id"]
        spk = row["speaker_id"]
        n = int(row["N"])
        model = row["model"]
        cohort = row["cohort_group"]

        feas_emerg = split_feasible_map[(spk, n)]
        missing_emerg = [c for c in EMERGENCY_CLASSES if c not in feas_emerg]
        stage2_k = len(feas_emerg)

        cell_s2 = s2_grouped.get_group(exp_id) if exp_id in s2_grouped.groups else pd.DataFrame()
        n_emerg = len(cell_s2)

        if n_emerg > 0:
            y_true = cell_s2["true_intent"].tolist()
            y_pred = cell_s2["stage2_predicted_intent"].tolist()

            # Ensure only emergency utterances
            assert all(yt in EMERGENCY_CLASSES for yt in y_true), f"Non-emergency found in stage2 for {exp_id}"

            # Macro-F1 across all 6 emergency classes
            macro_f1_all6 = float(f1_score(y_true, y_pred, labels=EMERGENCY_CLASSES, average="macro", zero_division=0))
            # Feasible Macro-F1 across adaptation feasible emergency classes only
            feas_macro_f1 = float(f1_score(y_true, y_pred, labels=feas_emerg, average="macro", zero_division=0))
            # Weighted F1
            weighted_f1 = float(f1_score(y_true, y_pred, labels=feas_emerg, average="weighted", zero_division=0))
            # Balanced Accuracy
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))

            # Per-class metrics
            for ec in EMERGENCY_CLASSES:
                is_feas = (ec in feas_emerg)
                support = sum(1 for yt in y_true if yt == ec)
                pred_cnt = sum(1 for yp in y_pred if yp == ec)
                
                # Binary metrics for this specific emergency class
                y_true_binary = [1 if yt == ec else 0 for yt in y_true]
                y_pred_binary = [1 if yp == ec else 0 for yp in y_pred]
                
                prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
                f1_val = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

                s2_per_class_records.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "cohort_group": cohort,
                    "N": n,
                    "model": model,
                    "emergency_class": ec,
                    "is_feasible": is_feas,
                    "support": support,
                    "predicted_count": pred_cnt,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1_val, 4)
                })
        else:
            macro_f1_all6 = 0.0
            feas_macro_f1 = 0.0
            weighted_f1 = 0.0
            bal_acc = 0.0

        s2_cond_records.append({
            "experiment_id": exp_id,
            "speaker_id": spk,
            "cohort_group": cohort,
            "N": n,
            "model": model,
            "stage2_num_classes": stage2_k,
            "feasible_emergency_classes": ";".join(feas_emerg),
            "missing_emergency_classes": ";".join(missing_emerg),
            "emergency_sample_count": n_emerg,
            "emergency_subtype_macro_f1": round(macro_f1_all6, 4),
            "feasible_macro_f1": round(feas_macro_f1, 4),
            "emergency_subtype_weighted_f1": round(weighted_f1, 4),
            "balanced_accuracy": round(bal_acc, 4)
        })

    s2_cond_df = pd.DataFrame(s2_cond_records)
    s2_per_class_df = pd.DataFrame(s2_per_class_records)

    s2_cond_path = os.path.join(ANALYSIS_DIR, "stage2_conditional_metrics.csv")
    s2_per_class_path = os.path.join(ANALYSIS_DIR, "stage2_per_class_metrics.csv")

    s2_cond_df.to_csv(s2_cond_path, index=False)
    s2_per_class_df.to_csv(s2_per_class_path, index=False)
    print(f"Saved {s2_cond_path} ({len(s2_cond_df)} rows).")
    print(f"Saved {s2_per_class_path} ({len(s2_per_class_df)} rows).")

    # ============================================================
    # 2. END-TO-END ERROR TAXONOMY
    # ============================================================
    print("\n--- 2. Computing End-to-End Error Taxonomy ---")
    tax_records = []
    hier_grouped = hier_preds_df.groupby("experiment_id")

    for idx, row in results_df.iterrows():
        exp_id = row["experiment_id"]
        spk = row["speaker_id"]
        n = int(row["N"])
        model = row["model"]
        cohort = row["cohort_group"]

        cell_preds = hier_grouped.get_group(exp_id)
        total_samples = len(cell_preds)

        y_true = cell_preds["true_intent"].tolist()
        y_pred = cell_preds["final_pred"].tolist()

        # Classify each prediction into A, B, C, D, E
        # A: GENERAL_SPEECH -> EMERGENCY (False Emergency)
        # B: EMERGENCY -> GENERAL_SPEECH (Missed Emergency)
        # C: EMERGENCY -> WRONG_EMERGENCY_TYPE (Wrong Type)
        # D: EMERGENCY -> CORRECT_EMERGENCY (Correct Emergency)
        # E: GENERAL_SPEECH -> GENERAL_SPEECH (Correct General)
        a_false_emerg = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "GENERAL_SPEECH" and yp in EMERGENCY_CLASSES)
        b_missed_emerg = sum(1 for yt, yp in zip(y_true, y_pred) if yt in EMERGENCY_CLASSES and yp == "GENERAL_SPEECH")
        c_wrong_type = sum(1 for yt, yp in zip(y_true, y_pred) if yt in EMERGENCY_CLASSES and yp in EMERGENCY_CLASSES and yt != yp)
        d_correct_emerg = sum(1 for yt, yp in zip(y_true, y_pred) if yt in EMERGENCY_CLASSES and yp == yt)
        e_correct_gen = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "GENERAL_SPEECH" and yp == "GENERAL_SPEECH")

        gen_total = a_false_emerg + e_correct_gen
        emerg_total = b_missed_emerg + c_wrong_type + d_correct_emerg

        assert a_false_emerg + b_missed_emerg + c_wrong_type + d_correct_emerg + e_correct_gen == total_samples, \
            f"Taxonomy sum mismatch for {exp_id}"
        assert gen_total == sum(1 for yt in y_true if yt == "GENERAL_SPEECH")
        assert emerg_total == sum(1 for yt in y_true if yt in EMERGENCY_CLASSES)

        # Rates relative to class totals
        fer = float(a_false_emerg / gen_total) if gen_total > 0 else 0.0
        mer = float(b_missed_emerg / emerg_total) if emerg_total > 0 else 0.0
        wtr = float(c_wrong_type / emerg_total) if emerg_total > 0 else 0.0
        cer = float(d_correct_emerg / emerg_total) if emerg_total > 0 else 0.0  # IPER
        cgr = float(e_correct_gen / gen_total) if gen_total > 0 else 0.0

        tax_records.append({
            "experiment_id": exp_id,
            "speaker_id": spk,
            "cohort_group": cohort,
            "N": n,
            "model": model,
            "total_samples": total_samples,
            "general_samples": gen_total,
            "emergency_samples": emerg_total,
            "false_emergency_count": a_false_emerg,
            "missed_emergency_count": b_missed_emerg,
            "wrong_emergency_type_count": c_wrong_type,
            "correct_emergency_count": d_correct_emerg,
            "correct_general_count": e_correct_gen,
            "false_emergency_rate": round(fer, 4),
            "missed_emergency_rate": round(mer, 4),
            "wrong_emergency_type_rate": round(wtr, 4),
            "correct_emergency_rate": round(cer, 4),  # Exactly IPER
            "correct_general_rate": round(cgr, 4),
            "false_emergency_pct_total": round(a_false_emerg / total_samples, 4),
            "missed_emergency_pct_total": round(b_missed_emerg / total_samples, 4),
            "wrong_emergency_type_pct_total": round(c_wrong_type / total_samples, 4),
            "correct_emergency_pct_total": round(d_correct_emerg / total_samples, 4),
            "correct_general_pct_total": round(e_correct_gen / total_samples, 4)
        })

    tax_df = pd.DataFrame(tax_records)
    tax_path = os.path.join(ANALYSIS_DIR, "error_taxonomy.csv")
    tax_df.to_csv(tax_path, index=False)
    print(f"Saved {tax_path} ({len(tax_df)} rows).")

    # Aggregate Error Taxonomy Summary
    tax_summary_records = []
    # By Model and N
    for (m, n_val), grp in tax_df.groupby(["model", "N"]):
        tot = grp["total_samples"].sum()
        gen = grp["general_samples"].sum()
        emg = grp["emergency_samples"].sum()
        a = grp["false_emergency_count"].sum()
        b = grp["missed_emergency_count"].sum()
        c = grp["wrong_emergency_type_count"].sum()
        d = grp["correct_emergency_count"].sum()
        e = grp["correct_general_count"].sum()
        tax_summary_records.append({
            "grouping": "model_x_N",
            "model": m,
            "N": n_val,
            "cohort_group": "all",
            "total_samples": tot,
            "general_samples": gen,
            "emergency_samples": emg,
            "false_emergency_count": a,
            "missed_emergency_count": b,
            "wrong_emergency_type_count": c,
            "correct_emergency_count": d,
            "correct_general_count": e,
            "false_emergency_rate": round(a / gen, 4) if gen > 0 else 0.0,
            "missed_emergency_rate": round(b / emg, 4) if emg > 0 else 0.0,
            "wrong_emergency_type_rate": round(c / emg, 4) if emg > 0 else 0.0,
            "correct_emergency_rate": round(d / emg, 4) if emg > 0 else 0.0,
            "correct_general_rate": round(e / gen, 4) if gen > 0 else 0.0
        })

    # By Model and Cohort
    for (m, ch), grp in tax_df.groupby(["model", "cohort_group"]):
        tot = grp["total_samples"].sum()
        gen = grp["general_samples"].sum()
        emg = grp["emergency_samples"].sum()
        a = grp["false_emergency_count"].sum()
        b = grp["missed_emergency_count"].sum()
        c = grp["wrong_emergency_type_count"].sum()
        d = grp["correct_emergency_count"].sum()
        e = grp["correct_general_count"].sum()
        tax_summary_records.append({
            "grouping": "model_x_cohort",
            "model": m,
            "N": "all",
            "cohort_group": ch,
            "total_samples": tot,
            "general_samples": gen,
            "emergency_samples": emg,
            "false_emergency_count": a,
            "missed_emergency_count": b,
            "wrong_emergency_type_count": c,
            "correct_emergency_count": d,
            "correct_general_count": e,
            "false_emergency_rate": round(a / gen, 4) if gen > 0 else 0.0,
            "missed_emergency_rate": round(b / emg, 4) if emg > 0 else 0.0,
            "wrong_emergency_type_rate": round(c / emg, 4) if emg > 0 else 0.0,
            "correct_emergency_rate": round(d / emg, 4) if emg > 0 else 0.0,
            "correct_general_rate": round(e / gen, 4) if gen > 0 else 0.0
        })

    # Overall by Model
    for m, grp in tax_df.groupby("model"):
        tot = grp["total_samples"].sum()
        gen = grp["general_samples"].sum()
        emg = grp["emergency_samples"].sum()
        a = grp["false_emergency_count"].sum()
        b = grp["missed_emergency_count"].sum()
        c = grp["wrong_emergency_type_count"].sum()
        d = grp["correct_emergency_count"].sum()
        e = grp["correct_general_count"].sum()
        tax_summary_records.append({
            "grouping": "model_overall",
            "model": m,
            "N": "all",
            "cohort_group": "all",
            "total_samples": tot,
            "general_samples": gen,
            "emergency_samples": emg,
            "false_emergency_count": a,
            "missed_emergency_count": b,
            "wrong_emergency_type_count": c,
            "correct_emergency_count": d,
            "correct_general_count": e,
            "false_emergency_rate": round(a / gen, 4) if gen > 0 else 0.0,
            "missed_emergency_rate": round(b / emg, 4) if emg > 0 else 0.0,
            "wrong_emergency_type_rate": round(c / emg, 4) if emg > 0 else 0.0,
            "correct_emergency_rate": round(d / emg, 4) if emg > 0 else 0.0,
            "correct_general_rate": round(e / gen, 4) if gen > 0 else 0.0
        })

    tax_summary_df = pd.DataFrame(tax_summary_records)
    tax_summary_path = os.path.join(ANALYSIS_DIR, "error_taxonomy_summary.csv")
    tax_summary_df.to_csv(tax_summary_path, index=False)
    print(f"Saved {tax_summary_path} ({len(tax_summary_df)} rows).")

    # ============================================================
    # 3. SECONDARY THRESHOLD ANALYSIS (EXPLORATORY)
    # ============================================================
    print("\n--- 3. Enriching Exploratory Threshold Analysis ---")
    tax_map = {row["experiment_id"]: row for idx, row in tax_df.iterrows()}

    enriched_thresh_records = []
    for idx, row in thresh_df.iterrows():
        exp_id = row["experiment_id"]
        tax_info = tax_map[exp_id]
        n_gen = tax_info["general_samples"]
        n_emg = tax_info["emergency_samples"]

        th = float(row["threshold"])
        s1_rec = float(row["stage1_recall"])
        s1_prec = float(row["stage1_precision"])
        s1_fpr = float(row["stage1_fpr"])
        fer = float(row["false_emergency_rate"])
        iper = float(row["end_to_end_iper"])
        macro_f1 = float(row["end_to_end_macro_f1"])

        # Reconstructed counts at threshold th
        pred_emerg_total = (fer * n_gen) + (s1_rec * n_emg)
        correct_emerg_total = iper * n_emg

        e2e_prec = float(correct_emerg_total / pred_emerg_total) if pred_emerg_total > 0 else 0.0
        e2e_rec = iper  # Exact mathematical equivalence
        e2e_f1 = float(2 * e2e_prec * e2e_rec / (e2e_prec + e2e_rec)) if (e2e_prec + e2e_rec) > 0 else 0.0
        gen_recall = 1.0 - fer

        enriched_thresh_records.append({
            "experiment_id": exp_id,
            "speaker_id": row["speaker_id"],
            "cohort_group": row["cohort_group"],
            "N": int(row["N"]),
            "model": row["model"],
            "threshold": th,
            "stage1_recall": round(s1_rec, 4),
            "stage1_precision": round(s1_prec, 4),
            "stage1_fpr": round(s1_fpr, 4),
            "emergency_precision": round(e2e_prec, 4),
            "emergency_recall": round(e2e_rec, 4),
            "emergency_f1": round(e2e_f1, 4),
            "FPR": round(s1_fpr, 4),
            "false_emergency_rate": round(fer, 4),
            "general_speech_recall": round(gen_recall, 4),
            "IPER": round(iper, 4),
            "end_to_end_macro_f1": round(macro_f1, 4),
            "status": "EXPLORATORY"
        })

    enriched_thresh_df = pd.DataFrame(enriched_thresh_records)
    thresh_csv_path = os.path.join(RESULTS_DIR, "threshold_analysis.csv")
    enriched_thresh_df.to_csv(thresh_csv_path, index=False)
    print(f"Updated {thresh_csv_path} with complete schema ({len(enriched_thresh_df)} rows).")

    # Aggregate threshold analysis across models & thresholds
    thresh_summary = enriched_thresh_df.groupby(["model", "threshold"])[[
        "stage1_recall", "stage1_precision", "stage1_fpr",
        "emergency_precision", "emergency_recall", "emergency_f1",
        "false_emergency_rate", "general_speech_recall", "IPER", "end_to_end_macro_f1"
    ]].mean().reset_index()
    thresh_summary["status"] = "EXPLORATORY"
    thresh_summary_path = os.path.join(ANALYSIS_DIR, "threshold_analysis_summary.csv")
    thresh_summary.to_csv(thresh_summary_path, index=False)
    print(f"Saved {thresh_summary_path}.")

    # ============================================================
    # 4. SPEAKER-LEVEL ERROR ANALYSIS
    # ============================================================
    print("\n--- 4. Computing Speaker-Level Error Analysis ---")
    spk_error_records = []

    res_map = {row["experiment_id"]: row for idx, row in results_df.iterrows()}
    s2_cond_map = {row["experiment_id"]: row for idx, row in s2_cond_df.iterrows()}

    for idx, row in tax_df.iterrows():
        exp_id = row["experiment_id"]
        res_info = res_map[exp_id]
        s2_info = s2_cond_map[exp_id]

        spk_error_records.append({
            "experiment_id": exp_id,
            "speaker_id": row["speaker_id"],
            "cohort_group": row["cohort_group"],
            "N": row["N"],
            "model": row["model"],
            # Stage 1
            "stage1_emergency_recall": res_info["stage1_emergency_recall"],
            "stage1_emergency_precision": res_info["stage1_emergency_precision"],
            "stage1_fpr": res_info["stage1_false_positive_rate"],
            # Stage 2
            "stage2_subtype_macro_f1": s2_info["emergency_subtype_macro_f1"],
            "stage2_feasible_macro_f1": s2_info["feasible_macro_f1"],
            "stage2_num_classes": s2_info["stage2_num_classes"],
            # End-to-end
            "end_to_end_iper": res_info["IPER"],
            "end_to_end_macro_f1": res_info["macro_f1"],
            "false_emergency_rate": row["false_emergency_rate"],
            "missed_emergency_rate": row["missed_emergency_rate"],
            "wrong_emergency_type_rate": row["wrong_emergency_type_rate"],
            "correct_emergency_rate": row["correct_emergency_rate"]
        })

    spk_error_df = pd.DataFrame(spk_error_records)
    spk_error_path = os.path.join(ANALYSIS_DIR, "speaker_error_analysis.csv")
    spk_error_df.to_csv(spk_error_path, index=False)
    print(f"Saved {spk_error_path} ({len(spk_error_df)} rows).")

    # ============================================================
    # 5. SAMPLE-EFFICIENCY CLOSURE
    # ============================================================
    print("\n--- 5. Generating Sample-Efficiency Closure Table ---")
    sample_eff_records = []

    for (m, n_val), grp in spk_error_df.groupby(["model", "N"]):
        sample_eff_records.append({
            "model": m,
            "N": n_val,
            "stage1_recall": round(grp["stage1_emergency_recall"].mean(), 4),
            "stage1_precision": round(grp["stage1_emergency_precision"].mean(), 4),
            "stage1_fpr": round(grp["stage1_fpr"].mean(), 4),
            "stage2_macro_f1": round(grp["stage2_subtype_macro_f1"].mean(), 4),
            "stage2_feasible_macro_f1": round(grp["stage2_feasible_macro_f1"].mean(), 4),
            "end_to_end_macro_f1": round(grp["end_to_end_macro_f1"].mean(), 4),
            "IPER": round(grp["end_to_end_iper"].mean(), 4),
            "false_emergency_rate": round(grp["false_emergency_rate"].mean(), 4),
            "missed_emergency_rate": round(grp["missed_emergency_rate"].mean(), 4)
        })

    sample_eff_df = pd.DataFrame(sample_eff_records)
    sample_eff_path = os.path.join(ANALYSIS_DIR, "sample_efficiency_closure.csv")
    sample_eff_df.to_csv(sample_eff_path, index=False)
    # Also update standard sample_efficiency.csv
    sample_eff_df.to_csv(os.path.join(ANALYSIS_DIR, "sample_efficiency.csv"), index=False)
    print(f"Saved {sample_eff_path}.")

    # ============================================================
    # 6. FLAT VS HIERARCHICAL MATCHED COMPARISON
    # ============================================================
    print("\n--- 6. Computing Matched Flat vs Hierarchical Comparison ---")
    flat_df = pd.read_csv(FLAT_RESULTS_PATH)
    flat_weighted = flat_df[flat_df["balancing_method"] == "class_weighted"].copy()

    # Model mapping:
    # weighted_adapter <-> hierarchical_adapter
    # weighted_rbf_svm <-> hierarchical_rbf_svm
    model_pairs = [
        ("weighted_adapter", "hierarchical_adapter"),
        ("weighted_rbf_svm", "hierarchical_rbf_svm")
    ]

    matched_records = []
    for flat_m, hier_m in model_pairs:
        for n_val in [20, 30, 40, 50]:
            f_cell = flat_weighted[(flat_weighted["model"] == flat_m) & (flat_weighted["N"] == n_val)]
            h_cell = results_df[(results_df["model"] == hier_m) & (results_df["N"] == n_val)]

            f_fer = float(f_cell["false_emergency_rate"].mean())
            h_fer = float(h_cell["false_emergency_rate"].mean())
            # Reduction in false emergency rate in percentage points
            reduction_fer_pp = round((f_fer - h_fer) * 100.0, 2)

            f_iper = float(f_cell["IPER"].mean())
            h_iper = float(h_cell["IPER"].mean())
            delta_iper_pp = round((h_iper - f_iper) * 100.0, 2)

            f_mf1 = float(f_cell["macro_f1"].mean())
            h_mf1 = float(h_cell["macro_f1"].mean())
            delta_mf1 = round(h_mf1 - f_mf1, 4)

            f_prec = float(f_cell["emergency_precision"].mean())
            h_prec = float(h_cell["emergency_precision"].mean())
            delta_prec_pp = round((h_prec - f_prec) * 100.0, 2)

            f_rec = float(f_cell["emergency_recall"].mean())
            h_rec = float(h_cell["emergency_recall"].mean())
            delta_rec_pp = round((h_rec - f_rec) * 100.0, 2)

            matched_records.append({
                "flat_model": flat_m,
                "hierarchical_model": hier_m,
                "N": n_val,
                "flat_fer": round(f_fer, 4),
                "hier_fer": round(h_fer, 4),
                "reduction_fer_pp": reduction_fer_pp,
                "flat_iper": round(f_iper, 4),
                "hier_iper": round(h_iper, 4),
                "delta_iper_pp": delta_iper_pp,
                "flat_macro_f1": round(f_mf1, 4),
                "hier_macro_f1": round(h_mf1, 4),
                "delta_macro_f1": delta_mf1,
                "flat_emergency_precision": round(f_prec, 4),
                "hier_emergency_precision": round(h_prec, 4),
                "delta_precision_pp": delta_prec_pp,
                "flat_emergency_recall": round(f_rec, 4),
                "hier_emergency_recall": round(h_rec, 4),
                "delta_recall_pp": delta_rec_pp
            })

    matched_df = pd.DataFrame(matched_records)
    matched_path = os.path.join(ANALYSIS_DIR, "flat_vs_hierarchical_matched.csv")
    matched_df.to_csv(matched_path, index=False)
    print(f"Saved {matched_path} ({len(matched_df)} rows).")

    # ============================================================
    # 7. PUBLICATION FIGURES GENERATION
    # ============================================================
    print("\n--- 7. Generating Diagnostic & Publication Figures ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Precision vs Recall across thresholds (Exploratory)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for m in ["hierarchical_adapter", "hierarchical_rbf_svm"]:
        sub = thresh_summary[thresh_summary["model"] == m]
        label = "Hierarchical Adapter" if "adapter" in m else "Hierarchical RBF-SVM"
        color = "#1f77b4" if "adapter" in m else "#d62728"
        marker = "o" if "adapter" in m else "s"
        ax.plot(sub["emergency_recall"] * 100, sub["emergency_precision"] * 100,
                marker=marker, linewidth=2.2, markersize=8, label=label, color=color)
        for _, r in sub.iterrows():
            ax.annotate(f"θ={r['threshold']:.2f}",
                        (r["emergency_recall"] * 100, r["emergency_precision"] * 100),
                        textcoords="offset points", xytext=(6, -4), fontsize=9)
    ax.set_title("Exploratory Precision vs. Recall across Operating Thresholds", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Emergency Recall / IPER (%)", fontsize=11)
    ax.set_ylabel("Emergency Precision (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "precision_recall_threshold.png"))
    plt.close(fig)

    # Figure 2: False Alarm Rate vs Recall across thresholds (Exploratory)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for m in ["hierarchical_adapter", "hierarchical_rbf_svm"]:
        sub = thresh_summary[thresh_summary["model"] == m]
        label = "Hierarchical Adapter" if "adapter" in m else "Hierarchical RBF-SVM"
        color = "#1f77b4" if "adapter" in m else "#d62728"
        marker = "o" if "adapter" in m else "s"
        ax.plot(sub["false_emergency_rate"] * 100, sub["emergency_recall"] * 100,
                marker=marker, linewidth=2.2, markersize=8, label=label, color=color)
        for _, r in sub.iterrows():
            ax.annotate(f"θ={r['threshold']:.2f}",
                        (r["false_emergency_rate"] * 100, r["emergency_recall"] * 100),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_title("Exploratory Trade-off: False Emergency Rate vs. Recall", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Emergency Rate / FPR on General Speech (%)", fontsize=11)
    ax.set_ylabel("Emergency Recall / IPER (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "false_alarm_recall_threshold.png"))
    plt.close(fig)

    # Figure 3: Stage 2 Conditional Macro-F1 across N
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    n_vals = [20, 30, 40, 50]
    for m in ["hierarchical_adapter", "hierarchical_rbf_svm"]:
        sub = sample_eff_df[sample_eff_df["model"] == m]
        label = "Hierarchical Adapter" if "adapter" in m else "Hierarchical RBF-SVM"
        color = "#1f77b4" if "adapter" in m else "#d62728"
        marker = "o" if "adapter" in m else "s"
        ax.plot(sub["N"], sub["stage2_feasible_macro_f1"],
                marker=marker, linewidth=2.2, markersize=8, label=f"{label} (Feasible)", color=color)
        ax.plot(sub["N"], sub["stage2_macro_f1"],
                marker=marker, linewidth=1.5, linestyle="--", markersize=6,
                label=f"{label} (All 6 Classes)", color=color, alpha=0.6)
    ax.set_title("Stage-2 Conditional Emergency Intent Macro-F1", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Adaptation Sample Budget (N)", fontsize=11)
    ax.set_ylabel("Macro-F1 (Evaluated on Ground-Truth Emergency Speech)", fontsize=11)
    ax.set_xticks(n_vals)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "stage2_conditional_macro_f1.png"))
    plt.close(fig)

    # Figure 4: Error Taxonomy Distribution by Model & Budget
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)
    models = ["hierarchical_adapter", "hierarchical_rbf_svm"]
    titles = ["Hierarchical Adapter Error Composition", "Hierarchical RBF-SVM Error Composition"]
    
    for idx, (m, ax_t) in enumerate(zip(models, axes)):
        sub = tax_summary_df[(tax_summary_df["grouping"] == "model_x_N") & (tax_summary_df["model"] == m)]
        x = np.arange(len(sub))
        width = 0.55
        
        c_gen = sub["correct_general_count"] / sub["total_samples"] * 100
        f_emg = sub["false_emergency_count"] / sub["total_samples"] * 100
        m_emg = sub["missed_emergency_count"] / sub["total_samples"] * 100
        w_emg = sub["wrong_emergency_type_count"] / sub["total_samples"] * 100
        c_emg = sub["correct_emergency_count"] / sub["total_samples"] * 100
        
        p1 = ax_t.bar(x, c_gen, width, label="Correct General", color="#2ca02c", alpha=0.85)
        p2 = ax_t.bar(x, f_emg, width, bottom=c_gen, label="False Emergency", color="#d62728", alpha=0.85)
        p3 = ax_t.bar(x, m_emg, width, bottom=c_gen+f_emg, label="Missed Emergency", color="#ff7f0e", alpha=0.85)
        p4 = ax_t.bar(x, w_emg, width, bottom=c_gen+f_emg+m_emg, label="Wrong Emergency Type", color="#9467bd", alpha=0.85)
        p5 = ax_t.bar(x, c_emg, width, bottom=c_gen+f_emg+m_emg+w_emg, label="Correct Emergency", color="#1f77b4", alpha=0.85)
        
        ax_t.set_title(titles[idx], fontsize=12, fontweight="bold")
        ax_t.set_xticks(x)
        ax_t.set_xticklabels([f"N={n}" for n in sub["N"]])
        ax_t.set_xlabel("Sample Budget (N)", fontsize=11)
        if idx == 0:
            ax_t.set_ylabel("% of Total Test Samples", fontsize=11)
            ax_t.legend(loc="upper right", fontsize=8.5, frameon=True)
            
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "error_taxonomy.png"))
    plt.close(fig)

    # Figure 5: Matched Flat vs Hierarchical Comparison (Reduction in False Alarms)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # Left: False Emergency Rate comparison
    x = np.arange(4)
    width = 0.35
    
    # RBF-SVM
    rbf_matched = matched_df[matched_df["hierarchical_model"] == "hierarchical_rbf_svm"]
    ax1.bar(x - width/2, rbf_matched["flat_fer"] * 100, width, label="Flat RBF-SVM", color="#ff9896")
    ax1.bar(x + width/2, rbf_matched["hier_fer"] * 100, width, label="Hierarchical RBF-SVM", color="#d62728")
    ax1.set_title("RBF-SVM: False Emergency Rate", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={n}" for n in [20, 30, 40, 50]])
    ax1.set_ylabel("False Emergency Rate (%)", fontsize=11)
    ax1.legend(frameon=True, fontsize=9)
    
    # Right: Adapter
    ada_matched = matched_df[matched_df["hierarchical_model"] == "hierarchical_adapter"]
    ax2.bar(x - width/2, ada_matched["flat_fer"] * 100, width, label="Flat Adapter", color="#aec7e8")
    ax2.bar(x + width/2, ada_matched["hier_fer"] * 100, width, label="Hierarchical Adapter", color="#1f77b4")
    ax2.set_title("Adapter: False Emergency Rate", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={n}" for n in [20, 30, 40, 50]])
    ax2.set_ylabel("False Emergency Rate (%)", fontsize=11)
    ax2.legend(frameon=True, fontsize=9)
    
    fig.suptitle("Matched Flat vs. Hierarchical False-Alarm Reduction", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "flat_vs_hierarchical_matched.png"))
    plt.close(fig)

    print("All figures successfully created in phase6_5/hierarchical/figures/.")

    # ============================================================
    # 8. VERIFICATION AUDIT & SUMMARY CHECKS
    # ============================================================
    print("\n--- 8. Verification Audit ---")
    # Check IPER == Emergency Recall
    iper_rec_diff = np.abs(results_df["IPER"] - results_df["emergency_recall"]).max()
    print(f"Max difference between IPER and Emergency Recall: {iper_rec_diff:.6f}")
    assert iper_rec_diff == 0.0, "IPER and Emergency Recall must match identically!"

    # Check False Emergency Rate == FPR over General Speech
    fer_fpr_diff = np.abs(results_df["false_emergency_rate"] - results_df["stage1_false_positive_rate"]).max()
    print(f"Max difference between False Emergency Rate and Stage 1 FPR: {fer_fpr_diff:.6f}")
    assert fer_fpr_diff == 0.0, "False Emergency Rate and Stage 1 FPR must match identically!"

    print("\n=== Subphase 6.5.3-C Analytical Closure Finished Successfully! ===")


if __name__ == "__main__":
    run_closure()
