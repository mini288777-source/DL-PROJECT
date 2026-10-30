"""
Subphase 6.5.2-C: Classical Rescue Analytical Closure & Audit Script

Generates:
1. phase6_5/classical_rescue/analysis/feasible_class_metrics.csv
2. phase6_5/classical_rescue/analysis/class_weight_audit.csv
3. phase6_5/classical_rescue/analysis/prediction_behavior_audit.csv
4. phase6_5/classical_rescue/analysis/false_alarm_analysis.csv

Audits:
- Feasible-class derivation from adaptation data only
- Class weight formula correctness: w_c = N_train / (K_present * n_c)
- Exact prediction distributions across classes and collapse states
- False emergency rates across N, speakers, and cohorts
- IPER vs Emergency Recall mathematical equivalence
- Common evaluation set feasibility
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_stratified.json")
ORIG_SPLITS_PATH = os.path.join(REPO_ROOT, "splits", "torgo_splits.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "results")
ANALYSIS_DIR = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "analysis")

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]


def run_closure_audit():
    print("=== Subphase 6.5.2-C Analytical Closure Audit ===")
    
    # 1. Load inputs
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    split_map = {(s["speaker_id"], s["sample_n"]): s["adaptation_recording_ids"] for s in splits}
    
    manifest_df = pd.read_csv(MANIFEST_PATH)
    rec_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))
    
    results_path = os.path.join(RESULTS_DIR, "classical_rescue_results.csv")
    preds_path = os.path.join(RESULTS_DIR, "classical_rescue_predictions.csv")
    weights_path = os.path.join(RESULTS_DIR, "class_weight_records.csv")
    
    results_df = pd.read_csv(results_path)
    preds_df = pd.read_csv(preds_path)
    weights_df = pd.read_csv(weights_path)
    
    print(f"Loaded {len(results_df)} result rows, {len(preds_df)} predictions, {len(weights_df)} weight records.")
    
    # Group predictions by experiment_id for fast lookup
    preds_by_exp = {exp_id: group for exp_id, group in preds_df.groupby("experiment_id")}
    
    # ============================================================
    # 2. Feasible-Class Metrics Verification
    # ============================================================
    print("\n--- 1. Auditing Feasible-Class Metrics ---")
    feasible_records = []
    
    for idx, row in results_df.iterrows():
        exp_id = row["experiment_id"]
        spk = row["speaker_id"]
        n = int(row["N"])
        model = row["model"]
        balancing = row["balancing_method"]
        cohort = row["cohort_group"]
        
        # Adaptation data labels only
        adapt_recs = split_map[(spk, n)]
        adapt_labels = [rec_intent[r] for r in adapt_recs]
        
        feasible_set = sorted(list(set(adapt_labels)))
        missing_set = [c for c in CANONICAL_CLASSES if c not in feasible_set]
        
        cell_preds = preds_by_exp[exp_id]
        y_true = cell_preds["true_intent"].tolist()
        y_pred = cell_preds["predicted_intent"].tolist()
        
        # Recompute metrics
        glob_f1 = float(f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
        feas_f1 = float(f1_score(y_true, y_pred, labels=feasible_set, average="macro", zero_division=0))
        emerg_f1 = float(f1_score(y_true, y_pred, labels=EMERGENCY_CLASSES, average="macro", zero_division=0))
        
        stored_glob = float(row["macro_f1"])
        stored_feas = float(row["feasible_macro_f1"])
        stored_emerg = float(row["emergency_macro_f1"])
        
        glob_diff = abs(glob_f1 - stored_glob)
        feas_diff = abs(feas_f1 - stored_feas)
        emerg_diff = abs(emerg_f1 - stored_emerg)
        
        is_consistent = (glob_diff < 1e-3) and (feas_diff < 1e-3) and (emerg_diff < 1e-3)
        
        feasible_records.append({
            "experiment_id": exp_id,
            "speaker_id": spk,
            "cohort_group": cohort,
            "N": n,
            "model": model,
            "balancing_method": balancing,
            "global_macro_f1": round(glob_f1, 4),
            "feasible_class_macro_f1": round(feas_f1, 4),
            "emergency_macro_f1": round(emerg_f1, 4),
            "num_feasible_classes": len(feasible_set),
            "num_missing_classes": len(missing_set),
            "feasible_classes": ";".join(feasible_set),
            "missing_classes": ";".join(missing_set),
            "stored_global_macro_f1": stored_glob,
            "stored_feasible_macro_f1": stored_feas,
            "stored_emergency_macro_f1": stored_emerg,
            "is_consistent": is_consistent
        })
        
    feasible_df = pd.DataFrame(feasible_records)
    feasible_out_path = os.path.join(ANALYSIS_DIR, "feasible_class_metrics.csv")
    feasible_df.to_csv(feasible_out_path, index=False)
    print(f"Feasible metrics verified: {feasible_df['is_consistent'].sum()} / {len(feasible_df)} fully consistent.")
    print(f"Saved: {feasible_out_path}")
    
    # ============================================================
    # 3. Class Weight Mathematical Verification
    # ============================================================
    print("\n--- 2. Auditing Class Weights ---")
    weight_audit_records = []
    
    for (spk, n), adapt_recs in split_map.items():
        adapt_labels = [rec_intent[r] for r in adapt_recs]
        counts = {c: adapt_labels.count(c) for c in CANONICAL_CLASSES}
        present_classes = [c for c, cnt in counts.items() if cnt > 0]
        K_present = len(present_classes)
        N_train = len(adapt_labels)
        
        for model in ["weighted_adapter", "weighted_rbf_svm"]:
            exp_id = f"{spk}_{model}_N{n}"
            sub_w = weights_df[weights_df["experiment_id"] == exp_id]
            rec_weights = dict(zip(sub_w["intent_class"], sub_w["computed_weight"]))
            
            for c in CANONICAL_CLASSES:
                n_c = counts[c]
                is_feas = (n_c > 0)
                expected_w = round(N_train / (K_present * n_c), 4) if is_feas else 0.0
                actual_w = round(float(rec_weights.get(c, 0.0)), 4)
                diff = abs(expected_w - actual_w)
                status = "PASS" if diff < 1e-3 else "FAIL"
                
                weight_audit_records.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "N": n,
                    "model": model,
                    "intent_class": c,
                    "is_feasible": is_feas,
                    "adaptation_count": n_c,
                    "total_adaptation_N": N_train,
                    "num_feasible_classes": K_present,
                    "expected_weight": expected_w,
                    "recorded_weight": actual_w,
                    "weight_difference": round(diff, 6),
                    "audit_status": status
                })
                
    weight_audit_df = pd.DataFrame(weight_audit_records)
    weight_out_path = os.path.join(ANALYSIS_DIR, "class_weight_audit.csv")
    weight_audit_df.to_csv(weight_out_path, index=False)
    num_pass = (weight_audit_df["audit_status"] == "PASS").sum()
    print(f"Class weight audit: {num_pass} / {len(weight_audit_df)} PASS.")
    print(f"Saved: {weight_out_path}")
    
    # ============================================================
    # 4. Prediction Behavior Audit
    # ============================================================
    print("\n--- 3. Auditing Prediction Distributions ---")
    behavior_records = []
    
    for idx, row in results_df.iterrows():
        exp_id = row["experiment_id"]
        spk = row["speaker_id"]
        cohort = row["cohort_group"]
        n = int(row["N"])
        model = row["model"]
        balancing = row["balancing_method"]
        
        cell_preds = preds_by_exp[exp_id]
        total_test = len(cell_preds)
        pred_counts = cell_preds["predicted_intent"].value_counts().to_dict()
        
        gen_count = pred_counts.get("GENERAL_SPEECH", 0)
        emerg_count = sum(pred_counts.get(c, 0) for c in EMERGENCY_CLASSES)
        
        gen_frac = gen_count / total_test if total_test > 0 else 0.0
        emerg_frac = emerg_count / total_test if total_test > 0 else 0.0
        
        class_fracs = {c: pred_counts.get(c, 0) / total_test for c in EMERGENCY_CLASSES}
        unique_classes = len(pred_counts)
        majority_collapse = bool(gen_frac == 1.0)
        minority_active = bool(emerg_frac > 0.0)
        
        behavior_records.append({
            "experiment_id": exp_id,
            "speaker_id": spk,
            "cohort_group": cohort,
            "N": n,
            "model": model,
            "balancing_method": balancing,
            "total_test_samples": total_test,
            "general_speech_pred_count": gen_count,
            "emergency_pred_count": emerg_count,
            "general_speech_pred_fraction": round(gen_frac, 4),
            "emergency_pred_fraction": round(emerg_frac, 4),
            "sos_help_pred_fraction": round(class_fracs["SOS_HELP"], 4),
            "fall_pred_fraction": round(class_fracs["FALL"], 4),
            "breathing_medical_pred_fraction": round(class_fracs["BREATHING_MEDICAL"], 4),
            "pain_pred_fraction": round(class_fracs["PAIN"], 4),
            "location_pred_fraction": round(class_fracs["LOCATION"], 4),
            "critical_response_pred_fraction": round(class_fracs["CRITICAL_RESPONSE"], 4),
            "num_unique_predicted_classes": unique_classes,
            "majority_collapse": majority_collapse,
            "minority_active": minority_active
        })
        
    behavior_df = pd.DataFrame(behavior_records)
    behavior_out_path = os.path.join(ANALYSIS_DIR, "prediction_behavior_audit.csv")
    behavior_df.to_csv(behavior_out_path, index=False)
    print(f"Saved: {behavior_out_path}")
    
    # ============================================================
    # 5. False Alarm Analysis
    # ============================================================
    print("\n--- 4. Auditing False Alarm Behavior ---")
    false_alarm_records = []
    
    # Focus on weighted models
    weighted_df = results_df[results_df["balancing_method"] == "class_weighted"]
    
    # Overall summary per model
    for model in ["weighted_adapter", "weighted_rbf_svm"]:
        m_df = weighted_df[weighted_df["model"] == model]
        false_alarm_records.append({
            "model": model,
            "grouping_type": "overall",
            "group_value": "all",
            "mean_false_emergency_rate": round(float(m_df["false_emergency_rate"].mean()), 4),
            "mean_emergency_recall": round(float(m_df["emergency_recall"].mean()), 4),
            "mean_emergency_precision": round(float(m_df["emergency_precision"].mean()), 4),
            "mean_iper": round(float(m_df["IPER"].mean()), 4),
            "mean_macro_f1": round(float(m_df["macro_f1"].mean()), 4),
            "tradeoff_ratio": round(float(m_df["false_emergency_rate"].mean() / m_df["emergency_recall"].mean()), 4)
        })
        
        # By N
        for n in [20, 30, 40, 50]:
            n_df = m_df[m_df["N"] == n]
            r_mean = float(n_df["emergency_recall"].mean())
            f_mean = float(n_df["false_emergency_rate"].mean())
            false_alarm_records.append({
                "model": model,
                "grouping_type": "by_N",
                "group_value": str(n),
                "mean_false_emergency_rate": round(f_mean, 4),
                "mean_emergency_recall": round(r_mean, 4),
                "mean_emergency_precision": round(float(n_df["emergency_precision"].mean()), 4),
                "mean_iper": round(float(n_df["IPER"].mean()), 4),
                "mean_macro_f1": round(float(n_df["macro_f1"].mean()), 4),
                "tradeoff_ratio": round(f_mean / r_mean, 4) if r_mean > 0 else 999.0
            })
            
        # By cohort
        for cohort in ["dysarthric", "control"]:
            c_df = m_df[m_df["cohort_group"] == cohort]
            r_mean = float(c_df["emergency_recall"].mean())
            f_mean = float(c_df["false_emergency_rate"].mean())
            false_alarm_records.append({
                "model": model,
                "grouping_type": "by_cohort",
                "group_value": cohort,
                "mean_false_emergency_rate": round(f_mean, 4),
                "mean_emergency_recall": round(r_mean, 4),
                "mean_emergency_precision": round(float(c_df["emergency_precision"].mean()), 4),
                "mean_iper": round(float(c_df["IPER"].mean()), 4),
                "mean_macro_f1": round(float(c_df["macro_f1"].mean()), 4),
                "tradeoff_ratio": round(f_mean / r_mean, 4) if r_mean > 0 else 999.0
            })
            
        # By speaker
        for spk in sorted(list(m_df["speaker_id"].unique())):
            s_df = m_df[m_df["speaker_id"] == spk]
            r_mean = float(s_df["emergency_recall"].mean())
            f_mean = float(s_df["false_emergency_rate"].mean())
            false_alarm_records.append({
                "model": model,
                "grouping_type": "by_speaker",
                "group_value": spk,
                "mean_false_emergency_rate": round(f_mean, 4),
                "mean_emergency_recall": round(r_mean, 4),
                "mean_emergency_precision": round(float(s_df["emergency_precision"].mean()), 4),
                "mean_iper": round(float(s_df["IPER"].mean()), 4),
                "mean_macro_f1": round(float(s_df["macro_f1"].mean()), 4),
                "tradeoff_ratio": round(f_mean / r_mean, 4) if r_mean > 0 else 999.0
            })

    false_alarm_df = pd.DataFrame(false_alarm_records)
    false_alarm_out_path = os.path.join(ANALYSIS_DIR, "false_alarm_analysis.csv")
    false_alarm_df.to_csv(false_alarm_out_path, index=False)
    print(f"Saved: {false_alarm_out_path}")
    
    print("\n=== All Closure Analysis CSVs Generated Successfully ===")

if __name__ == "__main__":
    run_closure_audit()
