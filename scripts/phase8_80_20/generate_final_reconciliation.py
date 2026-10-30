"""
Generate final reconciled metrics, safety verdict JSON, and safety verdict figure
for Phase 8 80/20 Benchmark Audit.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS

def main():
    repair_dir = os.path.join(REPO_ROOT, "results", "phase8_80_20", "audit_repair")
    fig_dir = os.path.join(repair_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load predictions
    df_a = pd.read_csv(os.path.join(REPO_ROOT, "results", "phase8_80_20", "track_a_classical", "predictions_track_a.csv"))
    df_b = pd.read_csv(os.path.join(REPO_ROOT, "results", "phase8_80_20", "track_b_quantum", "predictions_track_b.csv"))
    df_c = pd.read_csv(os.path.join(REPO_ROOT, "results", "phase8_80_20", "track_c_classical_shadow", "predictions_track_c.csv"))
    df_rep = pd.read_csv(os.path.join(repair_dir, "repaired_predictions.csv"))

    y_true = df_b["true_label"].values
    classes_with_support = [c for c in CANONICAL_CLASSES if np.sum(y_true == c) > 0]
    is_true_emergency = np.isin(y_true, list(EMERGENCY_CLASSES))
    is_true_general = (y_true == GENERAL_SPEECH_CLASS)

    total_samples = len(y_true)
    true_em = int(np.sum(is_true_emergency))
    true_gen = int(np.sum(is_true_general))

    configs = [
        ("Track_A", "Original (In-Sample Threshold)", "in_sample", df_a["final_emergency_prediction"].values, df_a["final_7class_prediction"].values),
        ("Track_A", "Repaired (Out-of-Fold Threshold)", "out_of_fold", df_rep[df_rep["track_id"] == "Track_A"]["final_emergency_prediction_repaired"].values, df_rep[df_rep["track_id"] == "Track_A"]["final_7class_prediction_repaired"].values),
        ("Track_B", "Original (In-Sample Threshold)", "in_sample", df_b["final_emergency_prediction"].values, df_b["final_7class_prediction"].values),
        ("Track_B", "Repaired (Out-of-Fold Threshold)", "out_of_fold", df_rep[df_rep["track_id"] == "Track_B"]["final_emergency_prediction_repaired"].values, df_rep[df_rep["track_id"] == "Track_B"]["final_7class_prediction_repaired"].values),
        ("Track_C", "Original (In-Sample Threshold)", "in_sample", df_c["final_emergency_prediction"].values, df_c["final_7class_prediction"].values),
        ("Track_C", "Repaired (Out-of-Fold Threshold)", "out_of_fold", df_rep[df_rep["track_id"] == "Track_C"]["final_emergency_prediction_repaired"].values, df_rep[df_rep["track_id"] == "Track_C"]["final_7class_prediction_repaired"].values),
    ]

    reconciled_rows = []
    for track_id, op_cond, cal_method, is_em_pred, y_pred_7cl in configs:
        is_em_pred = is_em_pred.astype(bool)
        tp = int(np.sum(is_true_emergency & is_em_pred))
        fp = int(np.sum(is_true_general & is_em_pred))
        fn = int(np.sum(is_true_emergency & ~is_em_pred))
        tn = int(np.sum(is_true_general & ~is_em_pred))

        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        fer = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        iper = rec
        gen_pres = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0

        f1_7cl = float(f1_score(y_true, y_pred_7cl, labels=list(CANONICAL_CLASSES), average="macro", zero_division=0))
        f1_6supp = float(f1_score(y_true, y_pred_7cl, labels=classes_with_support, average="macro", zero_division=0))
        f1_bin = float(f1_score(is_true_emergency, is_em_pred, average="binary", zero_division=0))
        acc = float(accuracy_score(y_true, y_pred_7cl))

        safety_status = "PASS" if fer <= 0.15 else "FAIL"

        reconciled_rows.append({
            "track_id": track_id,
            "operating_condition": op_cond,
            "threshold_calibration_method": cal_method,
            "total_test_samples": total_samples,
            "true_emergencies": true_em,
            "true_general_speech": true_gen,
            "true_positives_TP": tp,
            "false_positives_FP": fp,
            "false_negatives_FN": fn,
            "true_negatives_TN": tn,
            "total_predicted_alerts": tp + fp,
            "emergency_recall": rec,
            "emergency_precision": prec,
            "false_emergency_rate_FER": fer,
            "fer_safety_bound": 0.15,
            "fer_safety_status": safety_status,
            "canonical_7class_macro_f1": f1_7cl,
            "supported_6class_macro_f1": f1_6supp,
            "binary_emergency_f1": f1_bin,
            "canonical_iper_emergency_preservation": iper,
            "general_speech_preservation": gen_pres,
            "accuracy": acc
        })

    df_final_reconciled = pd.DataFrame(reconciled_rows)
    df_final_reconciled.to_csv(os.path.join(repair_dir, "metric_reconciliation_final.csv"), index=False)
    print("Saved metric_reconciliation_final.csv successfully.")

    # 2. Perform statistical tests
    correct_a = (df_a["final_7class_prediction"].values == y_true)
    correct_b = (df_b["final_7class_prediction"].values == y_true)
    b_only_acc = int(np.sum(correct_b & ~correct_a))
    a_only_acc = int(np.sum(~correct_b & correct_a))
    res_acc = stats.binomtest(b_only_acc, b_only_acc + a_only_acc, p=0.5)

    em_corr_a = (df_a["final_emergency_prediction"].values[is_true_emergency] == True)
    em_corr_b = (df_b["final_emergency_prediction"].values[is_true_emergency] == True)
    b_only_rec_orig = int(np.sum(em_corr_b & ~em_corr_a))
    a_only_rec_orig = int(np.sum(~em_corr_b & em_corr_a))
    res_rec_orig = stats.binomtest(b_only_rec_orig, b_only_rec_orig + a_only_rec_orig, p=0.5)

    rep_a_em = df_rep[df_rep["track_id"] == "Track_A"]["final_emergency_prediction_repaired"].values.astype(bool)
    rep_b_em = df_rep[df_rep["track_id"] == "Track_B"]["final_emergency_prediction_repaired"].values.astype(bool)
    em_corr_a_oof = (rep_a_em[is_true_emergency] == True)
    em_corr_b_oof = (rep_b_em[is_true_emergency] == True)
    b_only_rec_oof = int(np.sum(em_corr_b_oof & ~em_corr_a_oof))
    a_only_rec_oof = int(np.sum(~em_corr_b_oof & em_corr_a_oof))
    res_rec_oof = stats.binomtest(b_only_rec_oof, b_only_rec_oof + a_only_rec_oof, p=0.5)

    # 3. Create safety_verdict_final.json
    safety_verdict = {
        "benchmark": "Phase 8 Supplementary 80/20 Benchmark",
        "timestamp": "2026-10-10T12:00:00Z",
        "safety_bound_specification": {
            "metric": "False Emergency Rate (FER)",
            "definition": "FP / (FP + TN) = FP / Total_General_Speech",
            "upper_bound_limit": 0.15,
            "safety_bound_percentage": "15.0%"
        },
        "locked_test_safety_evaluations": {
            "oof_calibrated_policy": {
                "description": "Training-only 5-fold Stratified Out-of-Fold calibration (unbiased, zero test leakage)",
                "Track_A_Classical": {
                    "FP": 317,
                    "TN": 838,
                    "total_general_speech": 1155,
                    "FER": 0.27445887445887446,
                    "FER_percentage": "27.45%",
                    "safety_status": "FAIL",
                    "reason": "FER (27.45%) exceeds safety limit of 15.0%"
                },
                "Track_B_Quantum": {
                    "FP": 301,
                    "TN": 854,
                    "total_general_speech": 1155,
                    "FER": 0.2606060606060606,
                    "FER_percentage": "26.06%",
                    "safety_status": "FAIL",
                    "reason": "FER (26.06%) exceeds safety limit of 15.0%"
                },
                "Track_C_Matched_Shadow": {
                    "FP": 300,
                    "TN": 855,
                    "total_general_speech": 1155,
                    "FER": 0.2597402597402597,
                    "FER_percentage": "25.97%",
                    "safety_status": "FAIL",
                    "reason": "FER (25.97%) exceeds safety limit of 15.0%"
                },
                "collective_verdict": "FAILED — All three models fail the locked-test FER requirement under the OOF-calibrated policy. No model may be labeled as safety-passing."
            },
            "in_sample_calibrated_policy_historical": {
                "description": "Historical in-sample calibration on training probabilities (subject to optimism bias)",
                "Track_A_Classical": {"FER": 0.12294372294372294, "safety_status": "PASS_IN_SAMPLE_ONLY"},
                "Track_B_Quantum": {"FER": 0.11861471861471862, "safety_status": "PASS_IN_SAMPLE_ONLY"},
                "Track_C_Matched_Shadow": {"FER": 0.1523809523809524, "safety_status": "FAIL"}
            }
        },
        "overall_safety_status": {
            "locked_test_fer_limit": "FAILED",
            "classical_safety_status": "FAIL",
            "quantum_safety_status": "FAIL",
            "matched_shadow_safety_status": "FAIL"
        },
        "statistical_interpretation": {
            "hypothesis_test_conclusion": "no statistically significant quantum advantage was demonstrated",
            "accuracy_mcnemar_test": {
                "discordant_pairs": {"quantum_only_correct": b_only_acc, "classical_only_correct": a_only_acc},
                "p_value": float(res_acc.pvalue),
                "is_significant": bool(res_acc.pvalue < 0.05)
            },
            "emergency_recall_mcnemar_test_in_sample": {
                "discordant_pairs": {"quantum_only_detected": b_only_rec_orig, "classical_only_detected": a_only_rec_orig},
                "p_value": float(res_rec_orig.pvalue),
                "is_significant": bool(res_rec_orig.pvalue < 0.05)
            },
            "emergency_recall_mcnemar_test_oof": {
                "discordant_pairs": {"quantum_only_detected": b_only_rec_oof, "classical_only_detected": a_only_rec_oof},
                "p_value": float(res_rec_oof.pvalue),
                "is_significant": bool(res_rec_oof.pvalue < 0.05)
            },
            "equivalence_language_guidance": "A non-significant test of difference (p > 0.05) does not prove statistical equivalence. In the absence of a prespecified equivalence margin and a formal Two One-Sided Tests (TOST) design, equivalence cannot be asserted; rather, the data demonstrates that no statistically significant quantum advantage was observed."
        },
        "protocol_governance": {
            "test_set_lock_policy": "STRICTLY LOCKED — No further threshold tuning or hyperparameter adjustments may be evaluated on the locked test set.",
            "future_work_requirement": "Any further threshold development must use training/validation data only (e.g. nested cross-validation), and any future safety claim requires an entirely independent confirmation dataset."
        }
    }

    with open(os.path.join(repair_dir, "safety_verdict_final.json"), "w", encoding="utf-8") as f:
        json.dump(safety_verdict, f, indent=2)
    print("Saved safety_verdict_final.json successfully.")

    # 4. Generate figure for FER safety comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    tracks = ["Classical Baseline\n(Track A)", "Quantum Residual\n(Track B)", "Matched Shadow\n(Track C)"]
    in_sample_fer = [12.29, 11.86, 15.24]
    oof_fer = [27.45, 26.06, 25.97]

    x = np.arange(len(tracks))
    w = 0.35

    plt.bar(x - w/2, in_sample_fer, w, label="Original In-Sample $\\tau$ (Optimism Bias)", color="#90caf9", edgecolor="#1565c0", hatch="//")
    plt.bar(x + w/2, oof_fer, w, label="Repaired Out-of-Fold $\\tau$ (Unbiased)", color="#ef5350", edgecolor="#b71c1c")

    plt.axhline(15.0, color="#d32f2f", linestyle="--", linewidth=2, label="Defined Safety Bound (FER $\\leq$ 15.0%)")

    for i in range(len(tracks)):
        plt.text(x[i] - w/2, in_sample_fer[i] + 0.8, f"{in_sample_fer[i]:.2f}%", ha="center", fontsize=9, fontweight="bold")
        plt.text(x[i] + w/2, oof_fer[i] + 0.8, f"{oof_fer[i]:.2f}%", ha="center", fontsize=9, fontweight="bold", color="#b71c1c")

    plt.xticks(x, tracks, fontsize=10)
    plt.ylabel("False Emergency Rate (FER %)", fontsize=11)
    plt.ylim(0, 35)
    plt.title("Safety Verdict Audit: Locked-Test FER vs. 15% Safety Bound\n(All Models Fail Locked-Test Safety Requirement under Unbiased OOF Calibration)", fontsize=11, fontweight="bold")

    plt.annotate("SAFETY AUDIT VERDICT: FAILED\nUnder unbiased OOF calibration, all models exceed the 15% FER limit on the locked test set.\nNo candidate is safety-passing for deployment.",
                 xy=(0.5, 0.82), xycoords="axes fraction", ha="center", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#c62828", lw=1.5))

    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "safety_verdict_comparison.png"))
    plt.close()
    print("Generated safety_verdict_comparison.png successfully.")

if __name__ == "__main__":
    main()
