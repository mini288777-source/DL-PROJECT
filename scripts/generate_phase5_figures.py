"""
Phase 5 Figure Generation Script.

Generates 5 publication-quality scientific figures based strictly on empirical
results in results/phase5/phase5_results.csv:
1. Figure 1: Macro-F1 vs SNR (Degradation curve across models with error bands)
2. Figure 2: Macro-F1 vs Adaptation Budget N (Sample efficiency under acoustic noise)
3. Figure 3: Sample Size x Noise Robustness Heatmap (3-panel comparative grid)
4. Figure 4: Intent-Preserving Emergency Recognition (IPER) vs SNR
5. Figure 5: Multi-Condition Model Comparison (Fair 3-way evaluation across all 720 cells)
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14


MODEL_COLORS = {
    "adapter": "#1f77b4",       # Blue
    "rbf_svm": "#2ca02c",       # Green
    "qsvc": "#d62728",          # Crimson / Quantum
}

MODEL_LABELS = {
    "adapter": "Classical Adapter",
    "rbf_svm": "Classical RBF-SVM",
    "qsvc": "Quantum QSVC (q=4)",
}

SNR_ORDER = ["clean", "snr15", "snr10", "snr5"]
SNR_LABELS = {
    "clean": "Clean",
    "snr15": "15 dB",
    "snr10": "10 dB",
    "snr5": "5 dB",
}


def load_results(results_csv: str = "results/phase5/phase5_results.csv") -> pd.DataFrame:
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    df = pd.read_csv(results_csv)
    return df[df["status"] == "SUCCESS"].copy()


def plot_figure_1_macro_f1_vs_snr(df: pd.DataFrame, out_dir: str):
    """Figure 1: Macro-F1 vs SNR across models with error shading."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for m in ["adapter", "rbf_svm", "qsvc"]:
        sub = df[df["model"] == m]
        if sub.empty:
            continue

        means = []
        stds = []
        for snr in SNR_ORDER:
            snr_sub = sub[sub["snr_tag"] == snr]["macro_f1"]
            means.append(snr_sub.mean() if not snr_sub.empty else np.nan)
            stds.append(snr_sub.std() if not snr_sub.empty else 0.0)

        x_coords = np.arange(len(SNR_ORDER))
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(x_coords, means, marker="o", linewidth=2.2, label=MODEL_LABELS[m], color=MODEL_COLORS[m])
        ax.fill_between(x_coords, means - stds, means + stds, alpha=0.18, color=MODEL_COLORS[m])

    ax.set_xticks(range(len(SNR_ORDER)))
    ax.set_xticklabels([SNR_LABELS[s] for s in SNR_ORDER])
    ax.set_xlabel("Test Acoustic Condition (Acoustic Noise Severity ->)")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Figure 1: Macro-F1 Degradation vs Test SNR (Clean Calibration -> Noisy Evaluation)")
    ax.set_ylim([0.0, max(0.4, (df['macro_f1'].max() + 0.05) if not df.empty else 0.4)])
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "macro_f1_vs_snr.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {out_path}")


def plot_figure_2_macro_f1_vs_sample_size(df: pd.DataFrame, out_dir: str):
    """Figure 2: Macro-F1 vs Adaptation Budget N across SNRs."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True, dpi=300)
    budgets = [20, 30, 40, 50]

    for idx, snr in enumerate(SNR_ORDER):
        ax = axes[idx]
        snr_df = df[df["snr_tag"] == snr]

        for m in ["adapter", "rbf_svm", "qsvc"]:
            m_df = snr_df[snr_df["model"] == m]
            if m_df.empty:
                continue

            means = [m_df[m_df["sample_n"] == n]["macro_f1"].mean() for n in budgets]
            stds = [m_df[m_df["sample_n"] == n]["macro_f1"].std() for n in budgets]

            ax.errorbar(budgets, means, yerr=stds, fmt="-o", capsize=4, linewidth=1.8,
                        label=MODEL_LABELS[m], color=MODEL_COLORS[m])

        ax.set_title(f"Test Condition: {SNR_LABELS[snr]}")
        ax.set_xlabel("Adaptation Budget N (Clean Samples)")
        if idx == 0:
            ax.set_ylabel("Macro-F1 Score")
        ax.set_xticks(budgets)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle("Figure 2: Sample Efficiency (Macro-F1 vs N) Across Noise Conditions", fontsize=14, y=1.03)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "macro_f1_vs_sample_size.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2: {out_path}")


def plot_figure_3_heatmaps(df: pd.DataFrame, out_dir: str):
    """Figure 3: Sample Size x Noise Robustness Heatmap (3 comparative panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, dpi=300)
    budgets = [20, 30, 40, 50]

    vmax = max(0.25, df["macro_f1"].max() if not df.empty else 0.25)
    vmin = min(0.05, df["macro_f1"].min() if not df.empty else 0.05)

    for idx, m in enumerate(["adapter", "rbf_svm", "qsvc"]):
        ax = axes[idx]
        sub = df[df["model"] == m]

        grid = np.zeros((len(budgets), len(SNR_ORDER)))
        for i, n in enumerate(budgets):
            for j, snr in enumerate(SNR_ORDER):
                val = sub[(sub["sample_n"] == n) & (sub["snr_tag"] == snr)]["macro_f1"].mean()
                grid[i, j] = val if not np.isnan(val) else 0.0

        sns.heatmap(
            grid,
            annot=True,
            fmt=".3f",
            cmap="Blues" if m == "adapter" else ("Greens" if m == "rbf_svm" else "Reds"),
            xticklabels=[SNR_LABELS[s] for s in SNR_ORDER],
            yticklabels=budgets,
            cbar=idx == 2,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("Test Acoustic Condition")
        if idx == 0:
            ax.set_ylabel("Adaptation Budget N")

    fig.suptitle("Figure 3: Sample Size x Noise Robustness Heatmap (Macro-F1 Grid)", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "noise_sample_efficiency_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 3: {out_path}")


def plot_figure_4_iper_vs_snr(df: pd.DataFrame, out_dir: str):
    """Figure 4: Intent-Preserving Emergency Recognition (IPER) vs SNR."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    for m in ["adapter", "rbf_svm", "qsvc"]:
        sub = df[df["model"] == m]
        if sub.empty:
            continue

        means = []
        stds = []
        for snr in SNR_ORDER:
            snr_sub = sub[sub["snr_tag"] == snr]["iper"]
            means.append(snr_sub.mean() if not snr_sub.empty else np.nan)
            stds.append(snr_sub.std() if not snr_sub.empty else 0.0)

        x_coords = np.arange(len(SNR_ORDER))
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(x_coords, means, marker="s", linewidth=2.2, label=MODEL_LABELS[m], color=MODEL_COLORS[m])
        ax.fill_between(x_coords, np.maximum(0, means - stds), means + stds, alpha=0.18, color=MODEL_COLORS[m])

    ax.set_xticks(range(len(SNR_ORDER)))
    ax.set_xticklabels([SNR_LABELS[s] for s in SNR_ORDER])
    ax.set_xlabel("Test Acoustic Condition (Acoustic Noise Severity ->)")
    ax.set_ylabel("Emergency Intent Preservation Rate (IPER)")
    ax.set_title("Figure 4: IPER Degradation Under Acoustic Noise")
    ax.set_ylim([0.0, max(0.3, (df['iper'].max() + 0.05) if not df.empty else 0.3)])
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "iper_vs_snr.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 4: {out_path}")


def plot_figure_5_model_comparison(df: pd.DataFrame, out_dir: str):
    """Figure 5: Multi-Condition Controlled Model Comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Group by model and SNR_tag
    plot_df = df.copy()
    plot_df["snr_label"] = plot_df["snr_tag"].map(SNR_LABELS)
    plot_df["model_label"] = plot_df["model"].map(MODEL_LABELS)

    sns.barplot(
        data=plot_df,
        x="snr_label",
        y="macro_f1",
        hue="model_label",
        order=[SNR_LABELS[s] for s in SNR_ORDER],
        palette=[MODEL_COLORS["adapter"], MODEL_COLORS["rbf_svm"], MODEL_COLORS["qsvc"]],
        errorbar="se",
        capsize=0.1,
        ax=ax,
    )

    ax.set_xlabel("Test Acoustic Condition")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Figure 5: Three-Way Controlled Model Comparison Across Full Noise Sweep")
    ax.set_ylim([0.0, max(0.3, (df['macro_f1'].max() + 0.05) if not df.empty else 0.3)])
    ax.legend(title="Paradigm", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 5: {out_path}")


def main():
    results_csv = "results/phase5/phase5_results.csv"
    out_dir = "results/phase5/figures"
    os.makedirs(out_dir, exist_ok=True)

    df = load_results(results_csv)
    print(f"Loaded {len(df)} successful condition rows for figure generation.")

    plot_figure_1_macro_f1_vs_snr(df, out_dir)
    plot_figure_2_macro_f1_vs_sample_size(df, out_dir)
    plot_figure_3_heatmaps(df, out_dir)
    plot_figure_4_iper_vs_snr(df, out_dir)
    plot_figure_5_model_comparison(df, out_dir)

    # Mirror to phase5/figures/
    mirror_dir = "phase5/figures"
    os.makedirs(mirror_dir, exist_ok=True)
    for fig_name in [
        "macro_f1_vs_snr.png",
        "macro_f1_vs_sample_size.png",
        "noise_sample_efficiency_heatmap.png",
        "iper_vs_snr.png",
        "model_comparison.png",
    ]:
        src = os.path.join(out_dir, fig_name)
        dst = os.path.join(mirror_dir, fig_name)
        if os.path.isfile(src):
            import shutil
            shutil.copyfile(src, dst)
    print(f"All 5 figures generated and mirrored to {mirror_dir}.")


if __name__ == "__main__":
    main()
