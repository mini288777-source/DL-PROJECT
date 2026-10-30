"""
Phase 6.5.6: Final Fair Comparison, Evidence Synthesis & Research Closure
Script to generate all 8 authoritative synthesis figures in reports/phase6_5_6/figures/:

FIGURE 1: Original benchmark vs recovery diagnostic progression
FIGURE 2: Emergency support improvement from original to recovery protocol
FIGURE 3: False-emergency reduction: flat vs hierarchical
FIGURE 4: Representation dimensionality vs Stage-2 performance
FIGURE 5: Matched Q8 vs RBF performance
FIGURE 6: Emergency recall: QSVC vs RBF vs Adapter
FIGURE 7: Quantum computational cost comparison
FIGURE 8: Final evidence chain / research workflow

STRICTLY GENERATED FROM SAVED RESULTS ONLY. NO NEW MODEL TRAINING.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "phase6_5_6", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality styling
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_figure_1():
    """FIGURE 1: Original benchmark vs recovery diagnostic progression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: End-to-End Macro-F1 across Phases
    phases = ["Phase 3-4\n(Orig RBF/QSVC)", "Phase 6.5.2\n(Weighted Flat)", "Phase 6.5.3\n(Hierarchical d=4)", "Phase 6.5.5\n(Hier RBF d=8)", "Phase 6.5.5\n(QSVC Q8-R1)"]
    macro_f1 = [0.1390, 0.1285, 0.1064, 0.1087, 0.1051]
    colors = ["#94a3b8", "#f59e0b", "#3b82f6", "#2563eb", "#7c3aed"]

    bars1 = ax1.bar(phases, macro_f1, color=colors, width=0.55, edgecolor="black", alpha=0.9)
    ax1.set_ylabel("End-to-End Macro-F1")
    ax1.set_title("(A) End-to-End Macro-F1 Progression Across Phases", fontweight="bold")
    ax1.set_ylim(0, 0.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars1, macro_f1):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax1.axhline(0.1390, color="gray", linestyle=":", label="Original Majority-Collapse Macro-F1 (~0.139)")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Emergency Recall vs False-Emergency Rate Tradeoff
    fer = [0.0, 96.82, 15.06, 29.32, 28.12]
    recall = [0.0, 58.42, 25.14, 31.89, 27.52]
    labels = ["Phase 3 (Collapse)", "Phase 6.5.2 (Flat Rescue)", "Phase 6.5.3 (Hierarchical)", "Phase 6.5.5 (RBF d=8)", "Phase 6.5.5 (QSVC Q8)"]

    for x, y, c, lbl in zip(fer, recall, colors, labels):
        ax2.scatter(x, y, color=c, s=140, edgecolor="black", label=lbl, zorder=5)

    ax2.set_xlabel("False-Emergency Rate (FER %)")
    ax2.set_ylabel("Emergency Recall (%)")
    ax2.set_title("(B) Emergency Recall vs. False-Alarm Tradeoff", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 75)

    # Annotate points
    ax2.annotate("Original Collapse\n(0% recall, 0% FER)", (0, 0), textcoords="offset points", xytext=(15, 10),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    ax2.annotate("Flat Rescue\n(High Recall, Extreme FER)", (96.82, 58.42), textcoords="offset points", xytext=(-90, -25),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    ax2.annotate("Hierarchical Gate\n(FER Reduced ~82 pp)", (15.06, 25.14), textcoords="offset points", xytext=(20, 5),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    ax2.annotate("Q8-R1 vs RBF d=8\n(Statistically Matched)", (28.12, 27.52), textcoords="offset points", xytext=(-40, 20),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)

    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8)

    plt.suptitle("FIGURE 1: Evolution from Original Benchmark Collapse to Recovery Closure", fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "fig1_original_vs_recovery_progression.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_2():
    """FIGURE 2: Emergency support improvement from original to recovery protocol."""
    feas_path = os.path.join(REPO_ROOT, "phase6_5", "analysis", "original_vs_recovery_feasibility.csv")
    df = pd.read_csv(feas_path)
    grp = df.groupby("sample_n")[["orig_total_emergency", "rec_total_emergency", "orig_num_emergency_classes", "rec_num_emergency_classes"]].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    budgets = grp["sample_n"].values
    x = np.arange(len(budgets))
    w = 0.35

    # Panel A: Total Emergency Samples in Calibration Split
    ax1.bar(x - w/2, grp["orig_total_emergency"], width=w, label="Original Random Protocol (Track A)", color="#94a3b8", edgecolor="black")
    ax1.bar(x + w/2, grp["rec_total_emergency"], width=w, label="Recovered Stratified Protocol (Track B)", color="#10b981", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={b}" for b in budgets])
    ax1.set_ylabel("Mean Emergency Samples in Calibration Split")
    ax1.set_title("(A) Calibration Emergency Support Count", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    for i in range(len(budgets)):
        orig_v = grp["orig_total_emergency"].iloc[i]
        rec_v = grp["rec_total_emergency"].iloc[i]
        ax1.text(x[i] - w/2, orig_v + 0.3, f"{orig_v:.1f}", ha="center", va="bottom", fontsize=8)
        ax1.text(x[i] + w/2, rec_v + 0.3, f"{rec_v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel B: Number of Represented Emergency Classes (out of 6)
    ax2.bar(x - w/2, grp["orig_num_emergency_classes"], width=w, label="Original Track A", color="#94a3b8", edgecolor="black")
    ax2.bar(x + w/2, grp["rec_num_emergency_classes"], width=w, label="Recovered Track B", color="#3b82f6", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={b}" for b in budgets])
    ax2.set_ylabel("Mean Number of Emergency Classes Present")
    ax2.set_title("(B) Emergency Class Representation (max 6)", fontweight="bold")
    ax2.set_ylim(0, 6.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    for i in range(len(budgets)):
        orig_c = grp["orig_num_emergency_classes"].iloc[i]
        rec_c = grp["rec_num_emergency_classes"].iloc[i]
        ax2.text(x[i] - w/2, orig_c + 0.15, f"{orig_c:.1f}", ha="center", va="bottom", fontsize=8)
        ax2.text(x[i] + w/2, rec_c + 0.15, f"{rec_c:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("FIGURE 2: Calibration Emergency Support: Original vs. Stratified Recovery Protocol", fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "fig2_emergency_support_recovery.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_3():
    """FIGURE 3: False-emergency reduction: flat vs hierarchical."""
    matched_path = os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "analysis", "flat_vs_hierarchical_matched.csv")
    df = pd.read_csv(matched_path)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Filter for RBF and Adapter
    rbf_df = df[df["flat_model"] == "weighted_rbf_svm"].sort_values("N")
    adp_df = df[df["flat_model"] == "weighted_adapter"].sort_values("N")

    n_vals = rbf_df["N"].values
    x = np.arange(len(n_vals))
    w = 0.2

    ax.bar(x - 1.5*w, rbf_df["flat_fer"] * 100, width=w, label="RBF-SVM: Flat Cost-Sensitive", color="#ef4444", edgecolor="black", alpha=0.85)
    ax.bar(x - 0.5*w, rbf_df["hier_fer"] * 100, width=w, label="RBF-SVM: Hierarchical Gated", color="#3b82f6", edgecolor="black", alpha=0.9)
    ax.bar(x + 0.5*w, adp_df["flat_fer"] * 100, width=w, label="Adapter: Flat Cost-Sensitive", color="#f97316", edgecolor="black", alpha=0.85)
    ax.bar(x + 1.5*w, adp_df["hier_fer"] * 100, width=w, label="Adapter: Hierarchical Gated", color="#10b981", edgecolor="black", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in n_vals])
    ax.set_ylabel("False-Emergency Rate (FER %)")
    ax.set_title("FIGURE 3: Mitigation of False-Emergency Rate: Flat vs. Hierarchical Routing", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    # Highlight N=30 RBF reduction
    n30_idx = 1
    flat_val = rbf_df["flat_fer"].iloc[n30_idx] * 100
    hier_val = rbf_df["hier_fer"].iloc[n30_idx] * 100
    red_val = rbf_df["reduction_fer_pp"].iloc[n30_idx]
    ax.annotate(f"N=30 RBF Reduction:\n{flat_val:.1f}% -> {hier_val:.1f}%\n(-{red_val:.1f} pp)",
                (x[n30_idx] - 0.5*w, hier_val), textcoords="offset points", xytext=(15, 40),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontweight="bold", fontsize=8)

    out_path = os.path.join(FIGURES_DIR, "fig3_fer_reduction_flat_vs_hierarchical.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_4():
    """FIGURE 4: Representation dimensionality vs Stage-2 performance."""
    dim_path = os.path.join(REPO_ROOT, "results", "phase6_5_4_dimension_ablation.csv")
    df = pd.read_csv(dim_path)

    # Group by dimension
    grp = df.groupby("dimension")[["mean_explained_variance_pct", "stage2_feasible_macro_f1", "stage2_balanced_accuracy", "end_to_end_macro_f1"]].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    dims = grp["dimension"].values
    x = np.arange(len(dims))

    color1 = "#2563eb"
    ax1.set_xlabel("PCA Dimension (d)")
    ax1.set_ylabel("Cumulative Explained Variance (%)", color=color1)
    line1 = ax1.plot(x, grp["mean_explained_variance_pct"], marker="o", color=color1, lw=2.5, label="Cumulative Variance (%)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"d={d}" for d in dims])
    ax1.set_ylim(60, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for downstream metrics
    ax2 = ax1.twinx()
    color2 = "#7c3aed"
    color3 = "#059669"
    line2 = ax2.plot(x, grp["stage2_balanced_accuracy"] * 100, marker="s", color=color2, lw=2, linestyle="--", label="Stage-2 Balanced Acc (%)")
    line3 = ax2.plot(x, grp["stage2_feasible_macro_f1"] * 100, marker="^", color=color3, lw=2, linestyle="-.", label="Stage-2 Feasible Macro-F1 (x100)")
    ax2.set_ylabel("Stage-2 Performance (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(10, 45)

    # Highlight d=8 candidate
    d8_idx = np.where(dims == 8)[0][0]
    ax1.axvline(d8_idx, color="#dc2626", linestyle=":", lw=2, label="Selected Quantum Parity Candidate (d=8)")

    lines = line1 + line2 + line3 + [plt.Line2D([0], [0], color="#dc2626", linestyle=":", lw=2)]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)

    plt.title("FIGURE 4: PCA Dimensionality Audit: Explained Variance vs. Stage-2 Routing", fontweight="bold")
    out_path = os.path.join(FIGURES_DIR, "fig4_representation_dimensionality.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_5():
    """FIGURE 5: Matched Q8 vs RBF performance."""
    results_path = os.path.join(REPO_ROOT, "results", "phase6_5_5_speaker_results.csv")
    df = pd.read_csv(results_path)

    q8_df = df[df["model"] == "qsvc_q8_r1"].sort_values(["speaker_id", "N"])
    rbf_df = df[df["model"] == "hierarchical_rbf_svm_d8"].sort_values(["speaker_id", "N"])

    metrics = [
        ("end_to_end_macro_f1", "Macro-F1", 1.0, "Score"),
        ("end_to_end_emergency_recall", "Emergency Recall", 100.0, "Recall (%)"),
        ("end_to_end_false_emergency_rate", "False-Emergency Rate", 100.0, "FER (%)"),
        ("end_to_end_iper", "End-to-End IPER", 100.0, "IPER (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for idx, (col, title, scale, ylab) in enumerate(metrics):
        ax = axes[idx]
        q_vals = q8_df[col].values * scale
        r_vals = rbf_df[col].values * scale

        ax.scatter(r_vals, q_vals, color="#7c3aed", alpha=0.7, edgecolor="black", s=50)

        # 45-degree line
        min_v = min(np.min(q_vals), np.min(r_vals))
        max_v = max(np.max(q_vals), np.max(r_vals))
        ax.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", lw=1.5, label="Parity Line (Q8 = RBF)")

        ax.set_xlabel(f"Classical RBF-SVM d=8 {ylab}")
        ax.set_ylabel(f"QSVC Q8-R1 {ylab}")
        ax.set_title(f"Matched {title} (60 Pairs)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("FIGURE 5: Paired Comparison: QSVC Q8-R1 vs. Classical RBF-SVM d=8 Across 60 Conditions", fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_matched_q8_vs_rbf.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_6():
    """FIGURE 6: Emergency recall: QSVC vs RBF vs Adapter."""
    results_path = os.path.join(REPO_ROOT, "results", "phase6_5_5_speaker_results.csv")
    df = pd.read_csv(results_path)

    models = ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8"]
    labels = ["QSVC Q8-R1", "Classical RBF-SVM d=8", "Classical Adapter d=8"]
    colors = ["#7c3aed", "#2563eb", "#059669"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Mean Emergency Recall by Calibration Budget N
    for m, lbl, c in zip(models, labels, colors):
        sub = df[df["model"] == m]
        by_n = sub.groupby("N")["end_to_end_emergency_recall"].mean() * 100
        ax1.plot(by_n.index, by_n.values, marker="o", lw=2.2, color=c, label=lbl)

    ax1.set_xlabel("Calibration Budget (N)")
    ax1.set_ylabel("Emergency Recall (%)")
    ax1.set_title("(A) Emergency Recall vs. Calibration Budget N", fontweight="bold")
    ax1.set_xticks([20, 30, 40, 50])
    ax1.set_ylim(15, 60)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Panel B: Boxplot of Emergency Recall Distribution
    data_rec = [df[df["model"] == m]["end_to_end_emergency_recall"].values * 100 for m in models]
    bp = ax2.boxplot(data_rec, tick_labels=labels, patch_artist=True, medianprops=dict(color="black", lw=2))

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax2.set_ylabel("Emergency Recall (%)")
    ax2.set_title("(B) Emergency Recall Distribution Across 60 Conditions", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate Adapter advantage
    ax2.annotate("Adapter Advantage:\nDelta = +18.29 pp (p = 0.000125)",
                 xy=(3, 45.81), xytext=(2.1, 75),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                 fontweight="bold", fontsize=8)

    plt.suptitle("FIGURE 6: Emergency Recall Comparison: QSVC Q8-R1 vs. Classical Baselines", fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "fig6_emergency_recall_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_7():
    """FIGURE 7: Quantum computational cost comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ["RBF-SVM d=8", "Adapter d=8", "QSVC Q4-R1", "QSVC Q8-R1", "QSVC Q8-R2"]
    colors = ["#2563eb", "#059669", "#8b5cf6", "#7c3aed", "#6d28d9"]

    # Panel A: Test-Set Evaluation Latency per Utterance (seconds/eval)
    eval_latency = [0.0014, 0.0006, 0.4194, 1.0816, 1.9723]
    y_pos = np.arange(len(models))

    ax1.barh(y_pos, eval_latency, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models)
    ax1.set_xlabel("Inference Latency per Evaluation (seconds)")
    ax1.set_title("(A) Measured Per-Evaluation Inference Latency", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(eval_latency):
        ax1.text(v + 0.03, i, f"{v:.4f} s", va="center", fontsize=8, fontweight="bold")

    ax1.text(0.5, 0.15, "QSVC Q8-R1 is ~772.6x slower\nper test evaluation than RBF",
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#ef4444"), fontsize=9)

    # Panel B: Total Wall-Clock Experiment Time (seconds across 60 conditions)
    total_time = [0.33, 15.79, 25.16, 70.22, 128.04]
    ax2.barh(y_pos, total_time, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models)
    ax2.set_xlabel("Total Wall-Clock Time for 60 Conditions (seconds)")
    ax2.set_title("(B) Total Wall-Clock Runtime (60 Conditions)", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(total_time):
        ax2.text(v + 1.5, i, f"{v:.2f} s", va="center", fontsize=8, fontweight="bold")

    ax2.text(0.5, 0.15, "QSVC Q8-R1 is ~212.8x slower\noverall across 60 conditions",
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#ef4444"), fontsize=9)

    plt.suptitle("FIGURE 7: Computational Cost Audit: Classical vs. Quantum Statevector Simulation", fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "fig7_quantum_computational_cost.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_figure_8():
    """FIGURE 8: Final evidence chain / research workflow diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Draw boxes representing each major research phase
    stages = [
        {"id": "Phase 3-6", "title": "ORIGINAL BENCHMARK", "desc": "Random few-shot protocol\nFlat 7-class taxonomy\nMajority-class collapse (~95.1%)\nRBF & QSVC predictions identical", "color": "#fee2e2", "edge": "#ef4444", "x": 0.05, "y": 0.55},
        {"id": "Phase 6.5.0", "title": "DIAGNOSTIC AUDIT", "desc": "Audited 42k predictions\nUncovered data support failure\n7/60 conditions single-class\nFormulation issue identified", "color": "#fef3c7", "edge": "#f59e0b", "x": 0.23, "y": 0.55},
        {"id": "Phase 6.5.1", "title": "DATA RECOVERY", "desc": "Speaker-stratified splits (Track B)\nPrompt-isolated calibration\nEmergency support: 1.1 -> 11.3\nEmergency classes: 1.0 -> 5.2", "color": "#dcfce7", "edge": "#10b981", "x": 0.41, "y": 0.55},
        {"id": "Phase 6.5.2", "title": "CLASSICAL RESCUE", "desc": "Cost-sensitive class weighting\nEscaped majority collapse\nEmergency recall: 0% -> 58%\nFalse alarms exploded (FER 96%)", "color": "#e0e7ff", "edge": "#6366f1", "x": 0.59, "y": 0.55},
        {"id": "Phase 6.5.3", "title": "HIERARCHICAL RECOGNITION", "desc": "2-stage gated architecture\nStage 1: Binary triage gate\nStage 2: Conditional routing\nFER reduced: 96% -> 15%", "color": "#dbeafe", "edge": "#3b82f6", "x": 0.77, "y": 0.55},

        {"id": "Phase 6.5.4", "title": "REPRESENTATION AUDIT", "desc": "Discovered PCA plotting artifact\nEmpirical PCA-4 = 80.7% variance\nd=8 retains 89.7% variance\nRelieved Stage-2 bottleneck", "color": "#ede9fe", "edge": "#8b5cf6", "x": 0.23, "y": 0.12},
        {"id": "Phase 6.5.5", "title": "QUANTUM RECOVERY", "desc": "180 quantum conditions executed\nMatched Q8-R1 vs RBF d=8\nZZFeatureMap & FidelityKernel\nIdentical empirical behavior", "color": "#f3e8ff", "edge": "#a855f7", "x": 0.41, "y": 0.12},
        {"id": "Phase 6.5.5-C", "title": "STATISTICAL CLOSURE", "desc": "60 matched experimental units\nWilcoxon tests & Holm correction\nRuntime audit (~773x per eval)\nConfirmed CASE B (Null Result)", "color": "#fae8ff", "edge": "#d946ef", "x": 0.59, "y": 0.12},
        {"id": "Phase 6.5.6", "title": "FINAL EVIDENCE SYNTHESIS", "desc": "Authoritative research closure\nReconciled metric reporting\nNegative-control contribution\nStatus: NOT READY for deployment", "color": "#f1f5f9", "edge": "#475569", "x": 0.77, "y": 0.12},
    ]

    box_w, box_h = 0.15, 0.32

    for s in stages:
        rect = patches.FancyBboxPatch((s["x"], s["y"]), box_w, box_h,
                                      boxstyle="round,pad=0.015",
                                      facecolor=s["color"],
                                      edgecolor=s["edge"],
                                      linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(s["x"] + box_w/2, s["y"] + box_h - 0.04, s["id"], ha="center", va="top", fontweight="bold", fontsize=9, color=s["edge"])
        ax.text(s["x"] + box_w/2, s["y"] + box_h - 0.09, s["title"], ha="center", va="top", fontweight="bold", fontsize=8.5)
        ax.text(s["x"] + box_w/2, s["y"] + box_h - 0.15, s["desc"], ha="center", va="top", fontsize=7.2, linespacing=1.2)

    # Draw arrows between top stages
    for i in range(4):
        x_start = stages[i]["x"] + box_w
        x_end = stages[i+1]["x"]
        y_pos = stages[i]["y"] + box_h / 2
        ax.annotate("", xy=(x_end, y_pos), xytext=(x_start, y_pos),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#475569"), zorder=3)

    # Draw arrow down from Phase 6.5.3 to Phase 6.5.4
    ax.annotate("", xy=(stages[5]["x"] + box_w/2, stages[5]["y"] + box_h),
                xytext=(stages[4]["x"] + box_w/2, stages[4]["y"]),
                arrowprops=dict(arrowstyle="->", lw=2, color="#475569", connectionstyle="angle,angleA=0,angleB=90,rad=10"), zorder=3)

    # Draw arrows between bottom stages
    for i in range(5, 8):
        x_start = stages[i]["x"] + box_w
        x_end = stages[i+1]["x"]
        y_pos = stages[i]["y"] + box_h / 2
        ax.annotate("", xy=(x_end, y_pos), xytext=(x_start, y_pos),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#475569"), zorder=3)

    ax.set_title("FIGURE 8: Complete Research Evidence Chain and Methodology Progression", fontweight="bold", fontsize=13, y=0.98)
    out_path = os.path.join(FIGURES_DIR, "fig8_evidence_chain_workflow.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    print("All 8 Phase 6.5.6 synthesis figures generated successfully.")
