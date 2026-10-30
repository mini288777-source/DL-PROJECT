"""
Automated Quality Gate Validator for Resegregated Model Evaluation.
Tests all criteria from Section 13 of Master Prompt.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(r"D:\DL\reports\resegregated_evaluation")
RESULTS_DIR = BASE_DIR / "results"
SPLITS_DIR = BASE_DIR / "splits"
MANIFEST_DIR = BASE_DIR / "manifests"
FIGURES_DIR = BASE_DIR / "figures"

TAXONOMY = [
    "GENERAL_SPEECH", "CRITICAL_RESPONSE", "FALL",
    "BREATHING_MEDICAL", "PAIN", "SOS_HELP", "LOCATION"
]

EXPECTED_SUPPORT = {
    "GENERAL_SPEECH": 5219,
    "CRITICAL_RESPONSE": 35,
    "FALL": 21,
    "BREATHING_MEDICAL": 18,
    "PAIN": 13,
    "SOS_HELP": 11,
    "LOCATION": 0
}

def check(condition, message):
    if condition:
        print(f" [PASS] {message}")
        return True
    else:
        print(f"![FAIL] {message}")
        return False

def main():
    print("=" * 80)
    print("EXECUTING STAGE A AUTOMATIC QUALITY GATE CHECKS")
    print("=" * 80)
    
    passed_all = True
    
    # 1. Manifest Checks
    split_file = SPLITS_DIR / "torgo_resegregated_v1.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
        
    conditions = split_data if isinstance(split_data, list) else split_data.get("conditions", [])
    passed_all &= check(len(conditions) == 60, f"Manifest has exactly 60 conditions (found {len(conditions)})")
    
    # Check test samples per condition
    all_test_ids_per_budget = {20: set(), 30: set(), 40: set(), 50: set()}
    for c_info in conditions:
        b = c_info.get("sample_budget", c_info.get("sample_budget_n"))
        test_ids = c_info.get("test_ids", c_info.get("evaluation_utterance_ids", []))
        all_test_ids_per_budget[b].update(test_ids)
        
    for b, s in all_test_ids_per_budget.items():
        passed_all &= check(len(s) == 5317, f"Budget N={b} test set has exactly 5,317 unique items (found {len(s)})")
        
    # Check calibration/test recording overlap
    overlap_found = False
    for c_info in conditions:
        cal = set(c_info.get("train_ids", c_info.get("calibration_utterance_ids", [])))
        ev = set(c_info.get("test_ids", c_info.get("evaluation_utterance_ids", [])))
        if cal.intersection(ev):
            overlap_found = True
            break
    passed_all &= check(not overlap_found, "Zero recording overlap between calibration and evaluation sets")
    
    # 2. Prediction Ledger Checks
    pred_file = RESULTS_DIR / "predictions_resegregated_evaluation.csv"
    print(f"\nLoading prediction ledger: {pred_file}...")
    df_preds = pd.read_csv(pred_file)
    passed_all &= check(len(df_preds) == 191412, f"Prediction ledger has exactly 191,412 rows (found {len(df_preds)})")
    
    models = df_preds["model_id"].unique()
    passed_all &= check(len(models) == 9, f"Exactly 9 model architectures evaluated (found {len(models)})")
    
    budgets = df_preds["sample_budget"].unique()
    passed_all &= check(sorted(budgets) == [20, 30, 40, 50], f"Sample budgets are exactly [20, 30, 40, 50] (found {sorted(budgets)})")
    
    true_labels = set(df_preds["true_label"].unique())
    pred_labels = set(df_preds["predicted_label"].unique())
    passed_all &= check(true_labels.issubset(TAXONOMY), "True labels belong strictly to defined 7-class taxonomy")
    passed_all &= check(pred_labels.issubset(TAXONOMY), "Predicted labels belong strictly to defined 7-class taxonomy")
    
    # 3. Confusion Matrix Checks
    cm_dir = RESULTS_DIR / "confusion_matrices"
    all_cms_valid = True
    cm_files = list(cm_dir.glob("confusion_matrix_*.csv"))
    passed_all &= check(len(cm_files) >= 36, f"At least 36 budget-specific confusion matrices present (found {len(cm_files)})")
    
    for cm_path in cm_files:
        cm_df = pd.read_csv(cm_path, index_col=0)
        # Check total
        matrix_vals = cm_df.values.astype(int)
        total = matrix_vals.sum()
        if total != 5317:
            all_cms_valid = False
            print(f"Matrix {cm_path.name} sum = {total} (expected 5317)")
            
        # Check row sums (true class support)
        row_sums = matrix_vals.sum(axis=1)
        expected_rows = [EXPECTED_SUPPORT[c] for c in TAXONOMY]
        if list(row_sums) != expected_rows:
            all_cms_valid = False
            print(f"Matrix {cm_path.name} row sums mismatch: {list(row_sums)} vs {expected_rows}")
            
    passed_all &= check(all_cms_valid, "All 36 confusion matrices sum to 5,317 with exact expected class supports")
    
    # 4. Metric Consistency Checks
    df_pooled = pd.read_csv(RESULTS_DIR / "model_comparison_pooled_by_budget.csv")
    passed_all &= check(len(df_pooled) == 36, f"Pooled table contains exactly 36 records (9 models x 4 budgets)")
    
    # Check QSVC Q8 at N=50
    q8_n50 = df_pooled[(df_pooled["Model Identifier"] == "QSVC_Q8_R1") & (df_pooled["Calibration Budget N"] == 50)].iloc[0]
    rbf_n50 = df_pooled[(df_pooled["Model Identifier"] == "Matched_RBF_SVM_d8") & (df_pooled["Calibration Budget N"] == 50)].iloc[0]
    
    passed_all &= check(abs(q8_n50["Pooled Overall Accuracy (%)"] - 63.72) < 0.05, f"QSVC Q8-R1 N=50 pooled accuracy is 63.72% (found {q8_n50['Pooled Overall Accuracy (%)']})")
    passed_all &= check(abs(rbf_n50["Pooled Overall Accuracy (%)"] - 63.51) < 0.05, f"Matched RBF-SVM N=50 pooled accuracy is 63.51% (found {rbf_n50['Pooled Overall Accuracy (%)']})")
    passed_all &= check(abs(q8_n50["Pooled 7-Class Macro-F1"] - 0.1159) < 0.001, f"QSVC Q8-R1 N=50 pooled 7-class Macro-F1 is 0.1159 (found {q8_n50['Pooled 7-Class Macro-F1']})")
    passed_all &= check(abs(rbf_n50["Pooled 7-Class Macro-F1"] - 0.1145) < 0.001, f"Matched RBF-SVM N=50 pooled 7-class Macro-F1 is 0.1145 (found {rbf_n50['Pooled 7-Class Macro-F1']})")
    
    # 5. Condition-Level Summary Checks
    df_cond = pd.read_csv(RESULTS_DIR / "condition_level_results.csv")
    passed_all &= check(len(df_cond) == 60, f"Condition results file has exactly 60 rows (found {len(df_cond)})")
    
    df_cond_table = pd.read_csv(RESULTS_DIR / "model_comparison_condition_level.csv")
    q8_cond = df_cond_table[df_cond_table["Model Name"].str.contains("QSVC Q8")].iloc[0]
    rbf_cond = df_cond_table[df_cond_table["Model Name"].str.contains("Matched Classical RBF")].iloc[0]
    
    passed_all &= check(abs(q8_cond["Mean Condition Accuracy (%)"] - 72.35) < 0.05, f"QSVC Q8 condition-mean accuracy is 72.35% (found {q8_cond['Mean Condition Accuracy (%)']})")
    passed_all &= check(abs(rbf_cond["Mean Condition Accuracy (%)"] - 71.75) < 0.05, f"RBF condition-mean accuracy is 71.75% (found {rbf_cond['Mean Condition Accuracy (%)']})")
    passed_all &= check(abs(q8_cond["Mean Condition 7-Class Macro-F1"] - 0.1100) < 0.001, f"QSVC Q8 condition-mean Macro-F1 is 0.1100 (found {q8_cond['Mean Condition 7-Class Macro-F1']})")
    passed_all &= check(abs(rbf_cond["Mean Condition 7-Class Macro-F1"] - 0.1072) < 0.001, f"RBF condition-mean Macro-F1 is 0.1072 (found {rbf_cond['Mean Condition 7-Class Macro-F1']})")
    
    # 6. Sample-Budget Scaling Checks
    df_budget = pd.read_csv(RESULTS_DIR / "sample_budget_condition_means.csv")
    passed_all &= check(len(df_budget) == 4, f"Sample budget scaling table has 4 budgets (found {len(df_budget)})")
    b20 = df_budget[df_budget["Calibration_Budget_N"] == 20].iloc[0]
    b50 = df_budget[df_budget["Calibration_Budget_N"] == 50].iloc[0]
    passed_all &= check(abs(b20["QSVC_Macro_F1_Mean"] - 0.0767) < 0.001, f"QSVC N=20 Macro-F1 is 0.0767 (found {b20['QSVC_Macro_F1_Mean']})")
    passed_all &= check(abs(b50["QSVC_Macro_F1_Mean"] - 0.1266) < 0.001, f"QSVC N=50 Macro-F1 is 0.1266 (found {b50['QSVC_Macro_F1_Mean']})")
    passed_all &= check(abs(b20["QSVC_FER_Mean"] - 50.15) < 0.05, f"QSVC N=20 FER is 50.15% (found {b20['QSVC_FER_Mean']}%)")
    passed_all &= check(abs(b50["QSVC_FER_Mean"] - 13.64) < 0.05, f"QSVC N=50 FER is 13.64% (found {b50['QSVC_FER_Mean']}%)")
    
    # 7. Balanced Accuracy Definitions Checks
    passed_all &= check(abs(q8_cond["Mean Condition Stage-1 Binary Balanced Accuracy"] - 0.5014) < 0.001, "Stage-1 Binary Balanced Accuracy is ~0.5014")
    passed_all &= check(abs(q8_n50["Pooled Supported 6-Class Balanced Accuracy"] - 0.1582) < 0.001, "Multiclass Supported Balanced Accuracy is ~0.1582")
    passed_all &= check(abs(q8_n50["Pooled Canonical 7-Class Macro Recall"] - 0.1356) < 0.001, "Canonical 7-Class Macro Recall is ~0.1356")
    
    # 8. Statistical Comparisons Checks
    df_stat = pd.read_csv(RESULTS_DIR / "statistical_comparisons.csv")
    stat_row = df_stat[df_stat["Comparison Pair"].str.contains("QSVC Q8-R1 vs")].iloc[0]
    passed_all &= check(abs(stat_row["Unadjusted p-value"] - 0.5698) < 0.001, f"Primary paired Wilcoxon p-value is 0.5698 (found {stat_row['Unadjusted p-value']})")
    passed_all &= check(stat_row["Holm-Bonferroni Adjusted p"] == 1.0, f"Holm-Bonferroni adjusted p-value is 1.0000 (found {stat_row['Holm-Bonferroni Adjusted p']})")
    passed_all &= check("Parity" in stat_row["Scientific Decision"] or "CASE B" in stat_row["Scientific Decision"], "Scientific decision ratifies CASE B Parity")
    
    # 9. Figures Check
    all_figs = [f"fig{i}" for i in range(1, 13)]
    missing_figs = []
    for f in all_figs:
        pngs = list(FIGURES_DIR.glob(f"{f}_*.png"))
        svgs = list(FIGURES_DIR.glob(f"{f}_*.svg"))
        if not pngs or not svgs:
            missing_figs.append(f)
    passed_all &= check(len(missing_figs) == 0, f"All 12 figures exist in both PNG and SVG formats (missing: {missing_figs})")
    
    # 10. Document Generation Check
    docx_file = BASE_DIR / "RESEGREGATED_MODEL_COMPARISON_REPORT.docx"
    pdf_file = BASE_DIR / "RESEGREGATED_MODEL_COMPARISON_REPORT.pdf"
    md_file = BASE_DIR / "RESEGREGATED_MODEL_COMPARISON_REPORT.md"
    passed_all &= check(docx_file.exists() and docx_file.stat().st_size > 500000, f"DOCX report exists ({docx_file.stat().st_size} bytes)")
    passed_all &= check(pdf_file.exists() and pdf_file.stat().st_size > 500000, f"PDF report exists ({pdf_file.stat().st_size} bytes)")
    passed_all &= check(md_file.exists() and md_file.stat().st_size > 20000, f"MD report exists ({md_file.stat().st_size} bytes)")
    
    print("=" * 80)
    if passed_all:
        print("STAGE A QUALITY GATE: *** ALL CHECKS PASSED (READY FOR STAGE B) ***")
    else:
        print("STAGE A QUALITY GATE: !!! FAILURES DETECTED !!!")
    print("=" * 80)
    return passed_all

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
