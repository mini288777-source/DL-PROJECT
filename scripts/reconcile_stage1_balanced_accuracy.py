"""
reconcile_stage1_balanced_accuracy.py
=====================================
Performs independent condition-level forensic recomputation of Stage-1 metrics:
- TP, TN, FP, FN
- Sensitivity / Emergency Recall
- Specificity
- False-Emergency Rate (FER)
- Balanced Accuracy = (Recall + Specificity) / 2
Directly validates against:
phase6_5/hierarchical/results/stage1_predictions.csv
Outputs:
results/final_repair/stage1_balanced_accuracy_reconciliation.csv
"""

import os
import pandas as pd

def run_reconciliation():
    pred_path = "phase6_5/hierarchical/results/stage1_predictions.csv"
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing Stage 1 predictions file: {pred_path}")
    
    df = pd.read_csv(pred_path)
    
    # Check columns
    assert "true_binary" in df.columns and "pred_binary" in df.columns
    
    records = []
    
    # 1. Condition-level recomputation
    for (spk, n, m), grp in df.groupby(["speaker_id", "N", "model"]):
        cohort = grp["cohort_group"].iloc[0] if "cohort_group" in grp.columns else "unknown"
        y_true = grp["true_binary"].values
        y_pred = grp["pred_binary"].values
        
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        bal_acc = float((rec + spec) / 2.0)
        fer = float(1.0 - spec)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        
        records.append({
            "record_type": "CONDITION_LEVEL",
            "speaker_id": spk,
            "cohort_group": cohort,
            "N": n,
            "model": m,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "stage1_emergency_recall": round(rec, 4),
            "stage1_specificity": round(spec, 4),
            "stage1_false_emergency_rate": round(fer, 4),
            "stage1_emergency_precision": round(prec, 4),
            "stage1_balanced_accuracy": round(bal_acc, 4),
            "source_artifact": "phase6_5/hierarchical/results/stage1_predictions.csv"
        })
        
    res_df = pd.DataFrame(records)
    
    # 2. Add group-level summary rows
    summary_rows = []
    for (m, n), grp in res_df.groupby(["model", "N"]):
        summary_rows.append({
            "record_type": "GROUP_SUMMARY_MEAN",
            "speaker_id": "ALL_15_SPEAKERS",
            "cohort_group": "FULL_COHORT",
            "N": n,
            "model": m,
            "TP": int(grp["TP"].sum()),
            "TN": int(grp["TN"].sum()),
            "FP": int(grp["FP"].sum()),
            "FN": int(grp["FN"].sum()),
            "stage1_emergency_recall": round(float(grp["stage1_emergency_recall"].mean()), 4),
            "stage1_specificity": round(float(grp["stage1_specificity"].mean()), 4),
            "stage1_false_emergency_rate": round(float(grp["stage1_false_emergency_rate"].mean()), 4),
            "stage1_emergency_precision": round(float(grp["stage1_emergency_precision"].mean()), 4),
            "stage1_balanced_accuracy": round(float(grp["stage1_balanced_accuracy"].mean()), 4),
            "source_artifact": "recomputed_from_stage1_predictions"
        })
        
    full_df = pd.concat([res_df, pd.DataFrame(summary_rows)], ignore_index=True)
    out_path = "results/final_repair/stage1_balanced_accuracy_reconciliation.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"Saved Stage 1 reconciliation to: {out_path}")
    print("\nSummary Group Means:")
    print(pd.DataFrame(summary_rows)[["model", "N", "stage1_emergency_recall", "stage1_specificity", "stage1_false_emergency_rate", "stage1_balanced_accuracy"]])

if __name__ == "__main__":
    run_reconciliation()
