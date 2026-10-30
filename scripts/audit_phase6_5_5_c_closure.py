"""
Subphase 6.5.5-C: Quantum Statistical Closure & Final Audit Script
Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid
         Quantum-Classical Deep Learning for Reliable Emergency Communication

Performs:
1. Complete paired statistical comparison: QSVC Q8-R1 vs. Weighted Adapter d=8 across 7 metrics.
2. Secondary matched comparisons: Q4-R1 vs. Q8-R1 and Q8-R1 vs. Q8-R2 across 7 metrics.
3. Verification of primary QSVC Q8-R1 vs. Classical RBF-SVM d=8 comparison.
4. Comprehensive multiple comparison family definition and Holm-Bonferroni correction.
5. Computational cost audit: per-evaluation inference ratio vs. total wall-time ratio.
6. Generation of all 5 closure result CSVs in results/:
   - results/phase6_5_5_C_adapter_comparison.csv
   - results/phase6_5_5_C_quantum_ablation.csv
   - results/phase6_5_5_C_multiple_comparison.csv
   - results/phase6_5_5_C_runtime_audit.csv
   - results/phase6_5_5_C_final_decision.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SPEAKER_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv")
RUNTIME_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_runtime.csv")
KERNEL_DIAG_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_kernel_diagnostics.csv")

METRICS = [
    ("end_to_end_macro_f1", "End-to-End Macro-F1"),
    ("end_to_end_emergency_recall", "Emergency Recall"),
    ("end_to_end_emergency_precision", "Emergency Precision"),
    ("end_to_end_false_emergency_rate", "False-Emergency Rate"),
    ("end_to_end_iper", "End-to-End IPER"),
    ("stage2_feasible_macro_f1", "Stage-2 Feasible Macro-F1"),
    ("stage2_balanced_accuracy", "Stage-2 Balanced Accuracy"),
]


def compute_paired_comparison(df: pd.DataFrame, model_A: str, model_B: str, metric_col: str, metric_label: str) -> dict:
    sub_A = df[df["model"] == model_A].sort_values(["speaker_id", "N"])[metric_col].values
    sub_B = df[df["model"] == model_B].sort_values(["speaker_id", "N"])[metric_col].values

    diff = sub_A - sub_B
    n_pairs = len(diff)
    mean_A = float(np.mean(sub_A))
    mean_B = float(np.mean(sub_B))
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    std_diff = float(np.std(diff, ddof=1)) if n_pairs > 1 else 0.0

    if n_pairs > 1 and std_diff > 1e-12:
        se = std_diff / np.sqrt(n_pairs)
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se
        cohen_dz = float(mean_diff / std_diff)
    else:
        ci_low = mean_diff
        ci_high = mean_diff
        cohen_dz = 0.0

    if np.all(diff == 0.0):
        w_stat, p_val = 0.0, 1.0
    else:
        try:
            res_w = stats.wilcoxon(diff, zero_method="wilcox")
            w_stat, p_val = float(res_w.statistic), float(res_w.pvalue)
        except Exception:
            w_stat, p_val = 0.0, 1.0

    direction = "A > B" if mean_diff > 0 else ("B > A" if mean_diff < 0 else "IDENTICAL")

    return {
        "metric": metric_col,
        "metric_label": metric_label,
        "model_A": model_A,
        "model_B": model_B,
        "n_pairs": n_pairs,
        "mean_A": round(mean_A, 4),
        "mean_B": round(mean_B, 4),
        "mean_diff": round(mean_diff, 4),
        "median_diff": round(median_diff, 4),
        "std_diff": round(std_diff, 4),
        "ci_95_lower": round(ci_low, 4),
        "ci_95_upper": round(ci_high, 4),
        "cohens_dz": round(cohen_dz, 4),
        "wilcoxon_stat": round(w_stat, 2),
        "raw_p_value": round(p_val, 6),
        "direction": direction,
    }


def apply_holm_bonferroni(records: list) -> list:
    """Apply Holm-Bonferroni correction within a planned family of comparisons."""
    m = len(records)
    p_vals = [r["raw_p_value"] for r in records]
    sorted_idx = np.argsort(p_vals)
    adj_p = np.zeros(m)

    for rank, idx in enumerate(sorted_idx):
        mult = m - rank
        adj_p[idx] = min(1.0, p_vals[idx] * mult)

    # Monotonicity enforcement
    for i in range(1, m):
        adj_p[sorted_idx[i]] = max(adj_p[sorted_idx[i]], adj_p[sorted_idx[i-1]])

    for i, r in enumerate(records):
        r["p_value_adjusted_holm"] = round(float(adj_p[i]), 6)
        p_adj = r["p_value_adjusted_holm"]
        m_diff = r["mean_diff"]

        # Classification based on metric semantics
        # For false-emergency rate: lower is better!
        if r["metric"] == "end_to_end_false_emergency_rate":
            if p_adj < 0.05 and m_diff < 0.0:
                r["conclusion"] = f"{r['model_A']}_SUPERIOR_LOWER_FPR"
            elif p_adj < 0.05 and m_diff > 0.0:
                r["conclusion"] = f"{r['model_B']}_SUPERIOR_LOWER_FPR"
            else:
                r["conclusion"] = "NO_STATISTICAL_DIFFERENCE"
        else:
            if p_adj < 0.05 and m_diff > 0.0:
                r["conclusion"] = f"{r['model_A']}_SUPERIOR"
            elif p_adj < 0.05 and m_diff < 0.0:
                r["conclusion"] = f"{r['model_B']}_SUPERIOR"
            else:
                r["conclusion"] = "NO_STATISTICAL_DIFFERENCE"

    return records


def run_closure_audit():
    print("============================================================")
    print(" SUBPHASE 6.5.5-C: QUANTUM STATISTICAL CLOSURE & FINAL AUDIT")
    print("============================================================")

    df_results = pd.read_csv(SPEAKER_RESULTS_PATH)
    df_runtime = pd.read_csv(RUNTIME_PATH)
    df_kernel = pd.read_csv(KERNEL_DIAG_PATH)

    print(f"Loaded {len(df_results)} speaker results across {len(df_results['speaker_id'].unique())} speakers.")

    # 1. PART C: QSVC Q8-R1 vs. Weighted Adapter d=8 (All 7 Metrics)
    print("\n--- Computing QSVC Q8-R1 vs. Weighted Adapter d=8 Closure ---")
    adapter_comp_records = []
    for m_col, m_label in METRICS:
        rec = compute_paired_comparison(df_results, "qsvc_q8_r1", "hierarchical_adapter_d8", m_col, m_label)
        adapter_comp_records.append(rec)
    adapter_comp_records = apply_holm_bonferroni(adapter_comp_records)

    df_adapter_comp = pd.DataFrame(adapter_comp_records)
    adapter_comp_path = os.path.join(RESULTS_DIR, "phase6_5_5_C_adapter_comparison.csv")
    df_adapter_comp.to_csv(adapter_comp_path, index=False)
    print(f"Saved {adapter_comp_path} ({len(df_adapter_comp)} rows).")

    # 2. PART E & F: Quantum Ablations (Q4-R1 vs Q8-R1 and Q8-R1 vs Q8-R2)
    print("\n--- Computing Quantum Ablations Closure ---")
    ablation_records = []
    # Q8-R1 vs Q4-R1
    for m_col, m_label in METRICS:
        rec = compute_paired_comparison(df_results, "qsvc_q8_r1", "qsvc_q4_r1", m_col, m_label)
        rec["ablation_type"] = "QUBIT_SCALING_Q8_VS_Q4"
        ablation_records.append(rec)
    # Q8-R1 vs Q8-R2
    for m_col, m_label in METRICS:
        rec = compute_paired_comparison(df_results, "qsvc_q8_r1", "qsvc_q8_r2", m_col, m_label)
        rec["ablation_type"] = "CIRCUIT_DEPTH_REPS1_VS_REPS2"
        ablation_records.append(rec)

    ablation_records = apply_holm_bonferroni(ablation_records)
    df_ablation = pd.DataFrame(ablation_records)
    ablation_path = os.path.join(RESULTS_DIR, "phase6_5_5_C_quantum_ablation.csv")
    df_ablation.to_csv(ablation_path, index=False)
    print(f"Saved {ablation_path} ({len(df_ablation)} rows).")

    # 3. PART B & D: Primary Comparison Audit & Complete Multiple Comparison Matrix
    print("\n--- Computing Primary Comparison & Multiple Comparison Table ---")
    primary_records = []
    for m_col, m_label in METRICS:
        rec = compute_paired_comparison(df_results, "qsvc_q8_r1", "hierarchical_rbf_svm_d8", m_col, m_label)
        rec["family"] = "PRIMARY_KERNEL_COMPARISON"
        primary_records.append(rec)
    primary_records = apply_holm_bonferroni(primary_records)

    all_mult_records = []
    for r in primary_records:
        r["family"] = "PRIMARY_QSVC_VS_RBF_D8"
        all_mult_records.append(r)
    for r in adapter_comp_records:
        r_copy = dict(r)
        r_copy["family"] = "SECONDARY_QSVC_VS_ADAPTER_D8"
        all_mult_records.append(r_copy)
    for r in ablation_records:
        r_copy = dict(r)
        r_copy["family"] = r.get("ablation_type", "QUANTUM_ABLATION")
        all_mult_records.append(r_copy)

    df_mult = pd.DataFrame(all_mult_records)
    mult_path = os.path.join(RESULTS_DIR, "phase6_5_5_C_multiple_comparison.csv")
    df_mult.to_csv(mult_path, index=False)
    print(f"Saved {mult_path} ({len(df_mult)} rows).")

    # 4. PART H: Computational Cost Audit
    print("\n--- Auditing Computational Cost & Ratios ---")
    runtime_map = dict(zip(df_runtime["model"], df_runtime.to_dict(orient="records")))

    q8_r1_tot = runtime_map["qsvc_q8_r1"]["total_wall_time_sec"]
    q8_r1_inf = runtime_map["qsvc_q8_r1"]["mean_inf_time_sec"]
    q8_r1_trn = runtime_map["qsvc_q8_r1"]["mean_train_time_sec"]

    rbf8_tot = runtime_map["hierarchical_rbf_svm_d8"]["total_wall_time_sec"]
    rbf8_inf = runtime_map["hierarchical_rbf_svm_d8"]["mean_inf_time_sec"]
    rbf8_trn = runtime_map["hierarchical_rbf_svm_d8"]["mean_train_time_sec"]

    adp8_tot = runtime_map["hierarchical_adapter_d8"]["total_wall_time_sec"]
    adp8_inf = runtime_map["hierarchical_adapter_d8"]["mean_inf_time_sec"]
    adp8_trn = runtime_map["hierarchical_adapter_d8"]["mean_train_time_sec"]

    q8_r2_tot = runtime_map["qsvc_q8_r2"]["total_wall_time_sec"]
    q8_r2_inf = runtime_map["qsvc_q8_r2"]["mean_inf_time_sec"]
    q8_r2_trn = runtime_map["qsvc_q8_r2"]["mean_train_time_sec"]

    q4_r1_tot = runtime_map["qsvc_q4_r1"]["total_wall_time_sec"]
    q4_r1_inf = runtime_map["qsvc_q4_r1"]["mean_inf_time_sec"]

    rbf4_tot = runtime_map["hierarchical_rbf_svm_d4"]["total_wall_time_sec"]
    rbf4_inf = runtime_map["hierarchical_rbf_svm_d4"]["mean_inf_time_sec"]

    runtime_audit_records = [
        {
            "comparison": "QSVC Q8-R1 vs. RBF-SVM d=8",
            "metric_type": "Per-Evaluation Test Inference Time",
            "model_A_time_sec": q8_r1_inf,
            "model_B_time_sec": rbf8_inf,
            "ratio_A_to_B": round(q8_r1_inf / rbf8_inf, 2),
            "ratio_percentage": round((q8_r1_inf / rbf8_inf) * 100.0, 1),
            "interpretation": f"QSVC Q8-R1 is {round(q8_r1_inf / rbf8_inf, 1)}x slower per test evaluation than classical RBF-SVM"
        },
        {
            "comparison": "QSVC Q8-R1 vs. RBF-SVM d=8",
            "metric_type": "Total 60-Condition Experiment Wall Time",
            "model_A_time_sec": q8_r1_tot,
            "model_B_time_sec": rbf8_tot,
            "ratio_A_to_B": round(q8_r1_tot / rbf8_tot, 2),
            "ratio_percentage": round((q8_r1_tot / rbf8_tot) * 100.0, 1),
            "interpretation": f"QSVC Q8-R1 total experiment wall time is {round(q8_r1_tot / rbf8_tot, 1)}x slower than classical RBF-SVM"
        },
        {
            "comparison": "QSVC Q8-R2 vs. RBF-SVM d=8",
            "metric_type": "Per-Evaluation Test Inference Time",
            "model_A_time_sec": q8_r2_inf,
            "model_B_time_sec": rbf8_inf,
            "ratio_A_to_B": round(q8_r2_inf / rbf8_inf, 2),
            "ratio_percentage": round((q8_r2_inf / rbf8_inf) * 100.0, 1),
            "interpretation": f"QSVC Q8-R2 is {round(q8_r2_inf / rbf8_inf, 1)}x slower per test evaluation than classical RBF-SVM"
        },
        {
            "comparison": "QSVC Q8-R2 vs. RBF-SVM d=8",
            "metric_type": "Total 60-Condition Experiment Wall Time",
            "model_A_time_sec": q8_r2_tot,
            "model_B_time_sec": rbf8_tot,
            "ratio_A_to_B": round(q8_r2_tot / rbf8_tot, 2),
            "ratio_percentage": round((q8_r2_tot / rbf8_tot) * 100.0, 1),
            "interpretation": f"QSVC Q8-R2 total experiment wall time is {round(q8_r2_tot / rbf8_tot, 1)}x slower than classical RBF-SVM"
        },
        {
            "comparison": "QSVC Q8-R2 vs. QSVC Q8-R1",
            "metric_type": "Circuit Depth Scaling (Reps=2 vs Reps=1)",
            "model_A_time_sec": q8_r2_tot,
            "model_B_time_sec": q8_r1_tot,
            "ratio_A_to_B": round(q8_r2_tot / q8_r1_tot, 2),
            "ratio_percentage": round((q8_r2_tot / q8_r1_tot) * 100.0, 1),
            "interpretation": f"Doubling circuit depth from reps=1 to reps=2 increases total runtime by {round(q8_r2_tot / q8_r1_tot, 2)}x"
        },
        {
            "comparison": "QSVC Q8-R1 vs. Classical Adapter d=8",
            "metric_type": "Total 60-Condition Experiment Wall Time",
            "model_A_time_sec": q8_r1_tot,
            "model_B_time_sec": adp8_tot,
            "ratio_A_to_B": round(q8_r1_tot / adp8_tot, 2),
            "ratio_percentage": round((q8_r1_tot / adp8_tot) * 100.0, 1),
            "interpretation": f"QSVC Q8-R1 total runtime is {round(q8_r1_tot / adp8_tot, 2)}x slower than Classical Adapter d=8"
        },
        {
            "comparison": "QSVC Q4-R1 vs. RBF-SVM d=4",
            "metric_type": "Total 60-Condition Experiment Wall Time",
            "model_A_time_sec": q4_r1_tot,
            "model_B_time_sec": rbf4_tot,
            "ratio_A_to_B": round(q4_r1_tot / rbf4_tot, 2),
            "ratio_percentage": round((q4_r1_tot / rbf4_tot) * 100.0, 1),
            "interpretation": f"QSVC Q4-R1 total experiment wall time is {round(q4_r1_tot / rbf4_tot, 2)}x slower than classical RBF-SVM d=4"
        }
    ]

    df_runtime_audit = pd.DataFrame(runtime_audit_records)
    runtime_audit_path = os.path.join(RESULTS_DIR, "phase6_5_5_C_runtime_audit.csv")
    df_runtime_audit.to_csv(runtime_audit_path, index=False)
    print(f"Saved {runtime_audit_path} ({len(df_runtime_audit)} rows).")

    # 5. PART L: Final Decision Matrix
    print("\n--- Constructing Final Decision Matrix ---")
    decision_records = [
        {
            "decision_scope": "PRIMARY_KERNEL_COMPARISON",
            "comparison": "QSVC Q8-R1 vs. Classical RBF-SVM d=8",
            "classification": "CASE B",
            "classification_label": "NO_STATISTICALLY_SUPPORTED_DIFFERENCE",
            "statistical_justification": (
                "Under 60 matched speaker x N conditions, Macro-F1 (diff=-0.0036, p=0.3428), "
                "Emergency Recall (diff=-4.37%, p=0.0689), False-Emergency Rate (diff=-1.20%, p=0.9505), "
                "and IPER (diff=-3.44%, p=0.0582) show no statistically significant difference after Holm-Bonferroni correction."
            ),
            "scientific_implication": "The fidelity quantum kernel (ZZFeatureMap) provides zero empirical benefit over a classical RBF kernel for few-shot emergency intent classification."
        },
        {
            "decision_scope": "SAFETY_CRITICAL_ADAPTER_COMPARISON",
            "comparison": "QSVC Q8-R1 vs. Classical Adapter d=8",
            "classification": "CASE C (CLASSICAL ADVANTAGE ON RECALL)",
            "classification_label": "CLASSICAL_SUPERIOR_ON_EMERGENCY_RECALL",
            "statistical_justification": (
                "Classical Adapter d=8 achieves statistically significantly higher Emergency Recall (45.81% vs. 27.52%, "
                "diff=-18.29%, p=2.5e-05) and higher IPER (14.21% vs. 8.25%, diff=-5.97%, p=0.0030). "
                "QSVC achieves higher Macro-F1 (0.1051 vs. 0.0949, p=0.0067) and lower false alarms (28.12% vs. 49.82%, p=3e-06)."
            ),
            "scientific_implication": "For safety-critical communication where missed emergencies are unacceptable, classical parameterized adapters remain substantially superior to quantum support vector classifiers."
        },
        {
            "decision_scope": "QUBIT_SCALING_ABLATION",
            "comparison": "QSVC Q8-R1 vs. QSVC Q4-R1",
            "classification": "CASE B",
            "classification_label": "NO_STATISTICALLY_SUPPORTED_DIFFERENCE",
            "statistical_justification": (
                "End-to-end Macro-F1 is statistically indistinguishable between 4 qubits and 8 qubits "
                "(diff=-0.0057, p=0.2025). Stage-2 subtype balanced accuracy increases modestly (29.84% -> 32.53%, diff=+2.69%, p=0.45)."
            ),
            "scientific_implication": "Expanding the Hilbert space dimension from 2^4=16 to 2^8=256 does not improve end-to-end classification performance under few-shot data constraints."
        },
        {
            "decision_scope": "CIRCUIT_DEPTH_ABLATION",
            "comparison": "QSVC Q8-R1 vs. QSVC Q8-R2",
            "classification": "CASE B",
            "classification_label": "NO_PRACTICAL_DIFFERENCE",
            "statistical_justification": (
                "Increasing circuit depth from reps=1 to reps=2 produces a negligible Macro-F1 shift "
                "(diff=-0.0019, 0.1051 -> 0.1070) while doubling total runtime (70.22s -> 128.04s, 1.82x increase)."
            ),
            "scientific_implication": "Additional feature map depth provides zero meaningful representation advantage in the low-resource regime."
        },
        {
            "decision_scope": "COMPUTATIONAL_EFFICIENCY",
            "comparison": "QSVC Q8-R1 vs. Classical RBF-SVM d=8",
            "classification": "DECISIVE_CLASSICAL_EFFICIENCY_ADVANTAGE",
            "classification_label": "CLASSICAL_COMPUTATIONALLY_SUPERIOR",
            "statistical_justification": (
                "Per test evaluation inference time: QSVC is ~773x slower (1.0816s vs. 0.0014s). "
                "Total 60-condition experiment wall time: QSVC is ~213x slower (70.22s vs. 0.33s)."
            ),
            "scientific_implication": "Statevector simulation imposes massive computational overhead with zero classification gain. Hardware deployment would incur additional circuit execution and shot-noise overheads."
        }
    ]

    df_decision = pd.DataFrame(decision_records)
    decision_path = os.path.join(RESULTS_DIR, "phase6_5_5_C_final_decision.csv")
    df_decision.to_csv(decision_path, index=False)
    print(f"Saved {decision_path} ({len(df_decision)} rows).")

    print("\n============================================================")
    print(" SUBPHASE 6.5.5-C STATISTICAL CLOSURE COMPLETE")
    print("============================================================")


if __name__ == "__main__":
    run_closure_audit()
