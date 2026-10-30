"""
Phase 8 Statistical Analysis & Visualization Engine.

Generates:
1. Rigorous matched statistical tests (paired t-test, Wilcoxon signed-rank, Cohen's d, FDR correction).
2. All 10 required Phase 8 figures under results/phase8/figures/:
   - emergency_recall_comparison.png
   - emergency_precision_comparison.png
   - false_emergency_rate_comparison.png
   - macro_f1_comparison.png
   - dysarthric_vs_control_comparison.png
   - per_budget_trajectory.png
   - per_speaker_distribution.png
   - confusion_matrices.png
   - calibration_plot.png
   - runtime_comparison.png
3. Statistical summary JSON under results/phase8/phase8_statistical_tests.json.
"""

import sys
import os
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Applies Benjamini-Hochberg FDR correction."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]
    
    thresholds = (np.arange(1, n + 1) / n) * alpha
    significant = sorted_p <= thresholds
    
    # Find largest k such that p(k) <= (k/n)*alpha
    max_k = -1
    for k in range(n - 1, -1, -1):
        if significant[k]:
            max_k = k
            break
            
    is_sig = np.zeros(n, dtype=bool)
    if max_k >= 0:
        is_sig[sorted_indices[:max_k + 1]] = True
        
    return [(float(p), bool(sig)) for p, sig in zip(p_arr, is_sig)]


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes paired Cohen's d."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def run_statistical_analysis(df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
    print("Running Matched Statistical Analysis...", flush=True)

    tracks = ["Track_A_Classical", "Track_B_Quantum", "Track_C_Shadow"]
    metrics = ["emergency_recall", "emergency_precision", "false_emergency_rate", "macro_f1"]

    # Pivot by cell_id to get matched pairs
    pivoted = {}
    for m in metrics:
        pivoted[m] = df.pivot(index="cell_id", columns="track", values=m)

    stat_results: Dict[str, Any] = {
        "quantum_vs_classical_shadow": {},
        "quantum_vs_hierarchical_classical": {},
        "hierarchical_classical_vs_shadow": {},
    }

    raw_p_values_b_vs_c = []
    comparison_keys_b_vs_c = []

    # 1. Track B (Quantum) vs Track C (Shadow)
    for m in metrics:
        b_vals = pivoted[m]["Track_B_Quantum"].values
        c_vals = pivoted[m]["Track_C_Shadow"].values

        diff = b_vals - c_vals
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        d_stat = compute_cohens_d(b_vals, c_vals)

        # Paired t-test
        t_stat, p_t = stats.ttest_rel(b_vals, c_vals)
        # Wilcoxon
        try:
            w_stat, p_w = stats.wilcoxon(b_vals, c_vals)
        except Exception:
            w_stat, p_w = 0.0, 1.0

        # 95% CI
        ci_lower = mean_diff - 1.96 * (std_diff / np.sqrt(len(diff)))
        ci_upper = mean_diff + 1.96 * (std_diff / np.sqrt(len(diff)))

        stat_results["quantum_vs_classical_shadow"][m] = {
            "mean_difference": mean_diff,
            "ci_95": [float(ci_lower), float(ci_upper)],
            "cohens_d": d_stat,
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "t_p_value": float(p_t) if not np.isnan(p_t) else 1.0,
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else 0.0,
            "wilcoxon_p_value": float(p_w) if not np.isnan(p_w) else 1.0,
        }
        raw_p_values_b_vs_c.append(float(p_t) if not np.isnan(p_t) else 1.0)
        comparison_keys_b_vs_c.append(m)

    # Benjamini-Hochberg correction
    fdr_results = benjamini_hochberg(raw_p_values_b_vs_c, alpha=0.05)
    for k, (orig_p, is_sig) in zip(comparison_keys_b_vs_c, fdr_results):
        stat_results["quantum_vs_classical_shadow"][k]["fdr_significant"] = is_sig

    # 2. Track B (Quantum) vs Track A (Hierarchical Classical)
    for m in metrics:
        b_vals = pivoted[m]["Track_B_Quantum"].values
        a_vals = pivoted[m]["Track_A_Classical"].values

        diff = b_vals - a_vals
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        d_stat = compute_cohens_d(b_vals, a_vals)

        t_stat, p_t = stats.ttest_rel(b_vals, a_vals)
        try:
            w_stat, p_w = stats.wilcoxon(b_vals, a_vals)
        except Exception:
            w_stat, p_w = 0.0, 1.0

        ci_lower = mean_diff - 1.96 * (std_diff / np.sqrt(len(diff)))
        ci_upper = mean_diff + 1.96 * (std_diff / np.sqrt(len(diff)))

        stat_results["quantum_vs_hierarchical_classical"][m] = {
            "mean_difference": mean_diff,
            "ci_95": [float(ci_lower), float(ci_upper)],
            "cohens_d": d_stat,
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "t_p_value": float(p_t) if not np.isnan(p_t) else 1.0,
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else 0.0,
            "wilcoxon_p_value": float(p_w) if not np.isnan(p_w) else 1.0,
        }

    # Save to JSON
    json_path = os.path.join(output_dir, "phase8_statistical_tests.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)
    print(f"Saved statistical tests to {json_path}", flush=True)

    return stat_results


def generate_all_figures(df: pd.DataFrame, figures_dir: str):
    print("Generating Publication-Quality Figures...", flush=True)

    tracks = ["Track_A_Classical", "Track_B_Quantum", "Track_C_Shadow"]
    track_labels = ["Track A: Classical Gate", "Track B: Quantum Residual", "Track C: Matched Shadow"]
    colors = ["#2b5c8f", "#7c3aed", "#10b981"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Emergency Recall Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [df[df["track"] == t]["emergency_recall"].mean() * 100 for t in tracks]
    stds = [df[df["track"] == t]["emergency_recall"].std() * 100 for t in tracks]
    bars = ax.bar(track_labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Emergency Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 8 Emergency Recall Comparison across 60 Cells", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) + 20 if max(means) > 0 else 50)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "emergency_recall_comparison.png"), dpi=300)
    plt.close()

    # 2. Emergency Precision Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [df[df["track"] == t]["emergency_precision"].mean() * 100 for t in tracks]
    stds = [df[df["track"] == t]["emergency_precision"].std() * 100 for t in tracks]
    bars = ax.bar(track_labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Emergency Precision (%)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 8 Emergency Precision Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) + 15 if max(means) > 0 else 50)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "emergency_precision_comparison.png"), dpi=300)
    plt.close()

    # 3. False Emergency Rate Comparison (with safety veto threshold)
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [df[df["track"] == t]["false_emergency_rate"].mean() * 100 for t in tracks]
    stds = [df[df["track"] == t]["false_emergency_rate"].std() * 100 for t in tracks]
    bars = ax.bar(track_labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    ax.axhline(15.0, color="#dc2626", linestyle="--", linewidth=2, label="Safety Veto Bound (FER <= 15%)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("False Emergency Rate (FER %)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 8 False Emergency Rate vs Safety Veto Threshold", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(means) + 20, 25))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "false_emergency_rate_comparison.png"), dpi=300)
    plt.close()

    # 4. Macro-F1 Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [df[df["track"] == t]["macro_f1"].mean() * 100 for t in tracks]
    stds = [df[df["track"] == t]["macro_f1"].std() * 100 for t in tracks]
    bars = ax.bar(track_labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Macro-F1 (%)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 8 7-Class Macro-F1 Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) + 10 if max(means) > 0 else 30)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "macro_f1_comparison.png"), dpi=300)
    plt.close()

    # 5. Dysarthric vs Control Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(track_labels))
    w = 0.35

    dys_means = [df[(df["track"] == t) & (df["speaker_group"] == "dysarthric")]["emergency_recall"].mean() * 100 for t in tracks]
    ctrl_means = [df[(df["track"] == t) & (df["speaker_group"] == "control")]["emergency_recall"].mean() * 100 for t in tracks]

    r1 = ax.bar(x - w/2, dys_means, w, label="Dysarthric Speakers (N=8)", color="#e11d48", alpha=0.85, edgecolor="black")
    r2 = ax.bar(x + w/2, ctrl_means, w, label="Control Speakers (N=7)", color="#2563eb", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Emergency Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Subgroup Emergency Recall: Dysarthric vs Control Speakers", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(track_labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dysarthric_vs_control_comparison.png"), dpi=300)
    plt.close()

    # 6. Per-Budget Trajectory
    fig, ax = plt.subplots(figsize=(9, 5))
    budgets = [20, 30, 40, 50]
    for t, label, c in zip(tracks, track_labels, colors):
        b_means = [df[(df["track"] == t) & (df["budget"] == b)]["emergency_recall"].mean() * 100 for b in budgets]
        ax.plot(budgets, b_means, marker="o", linewidth=2.5, label=label, color=c)
    ax.set_xlabel("Experience Budget N (Samples)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Emergency Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Longitudinal Adaptation Trajectory across Budgets N in {20, 30, 40, 50}", fontsize=13, fontweight="bold")
    ax.set_xticks(budgets)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "per_budget_trajectory.png"), dpi=300)
    plt.close()

    # 7. Per-Speaker Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = [df[df["track"] == t]["emergency_recall"].values * 100 for t in tracks]
    bp = ax.boxplot(box_data, tick_labels=track_labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Emergency Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Speaker Distribution of Emergency Recall (15 TORGO Speakers)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "per_speaker_distribution.png"), dpi=300)
    plt.close()

    # 8. Confusion Breakdown (Bar representation)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(track_labels))
    w = 0.25
    true_em = [df[df["track"] == t]["test_emergencies"].sum() for t in tracks]
    pred_em = [df[df["track"] == t]["pred_emergencies"].sum() for t in tracks]
    correct_em = [df[df["track"] == t]["emergency_recall"].values * df[df["track"] == t]["test_emergencies"].values for t in tracks]
    correct_em_sum = [c.sum() for c in correct_em]

    ax.bar(x - w, true_em, w, label="True Emergencies", color="#4b5563", alpha=0.8)
    ax.bar(x, pred_em, w, label="Predicted Emergencies (Total Triggers)", color="#f59e0b", alpha=0.8)
    ax.bar(x + w, correct_em_sum, w, label="True Alarms (Correctly Detected)", color="#10b981", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(track_labels)
    ax.set_ylabel("Total Event Count across Test Corpus", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Breakdown: True Events vs Total Triggers vs True Alarms", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

    # 9. Calibration Plot (Recall vs FER Frontier)
    fig, ax = plt.subplots(figsize=(8, 6))
    for t, label, c in zip(tracks, track_labels, colors):
        sub_df = df[df["track"] == t]
        ax.scatter(sub_df["false_emergency_rate"] * 100, sub_df["emergency_recall"] * 100,
                   label=label, color=c, alpha=0.6, s=50)
    ax.set_xlabel("False Emergency Rate (FER %)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Emergency Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Emergency Detection Operating Characteristic: Recall vs FER", fontsize=13, fontweight="bold")
    ax.axvline(15.0, color="#dc2626", linestyle="--", label="FER Veto Threshold (15%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "calibration_plot.png"), dpi=300)
    plt.close()

    # 10. Runtime Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    train_times = [df[df["track"] == t]["train_time_sec"].mean() for t in tracks]
    latencies = [df[df["track"] == t]["latency_ms_per_sample"].mean() for t in tracks]

    bars1 = ax1.bar(track_labels, train_times, color=colors, alpha=0.85, edgecolor="black")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Training Time per Cell (s)", fontsize=11, fontweight="bold")
    ax1.set_title("Training Time Comparison", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    bars2 = ax2.bar(track_labels, latencies, color=colors, alpha=0.85, edgecolor="black")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f} ms", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Inference Latency per Sample (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Inference Latency Comparison", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "runtime_comparison.png"), dpi=300)
    plt.close()

    print(f"All 10 figures generated successfully in {figures_dir}", flush=True)


if __name__ == "__main__":
    csv_file = os.path.join(REPO_ROOT, "results", "phase8", "phase8_results.csv")
    out_dir = os.path.join(REPO_ROOT, "results", "phase8")
    fig_dir = os.path.join(out_dir, "figures")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} does not exist yet. Run benchmark first.")
        sys.exit(1)

    df_data = pd.read_csv(csv_file)
    run_statistical_analysis(df_data, out_dir)
    generate_all_figures(df_data, fig_dir)
