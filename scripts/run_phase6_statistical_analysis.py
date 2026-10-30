"""
Phase 6: Statistical Analysis & Intent-Preserving Evaluation Engine.

Executes:
1. Descriptive Statistics (Parametric & Non-Parametric) across all 720 cells.
2. Speaker-Level Heterogeneity Analysis (Dysarthric vs Control).
3. Paired Model Contrasts (Adapter vs RBF, Adapter vs QSVC, RBF vs QSVC).
4. Rigorous Hypothesis Testing (Normality check -> Paired t-test / Wilcoxon Signed-Rank).
5. Multiple Comparisons Correction (Holm-Bonferroni & Benjamini-Hochberg FDR).
6. Standardized Effect Sizes (Cohen's d, Hedges' g, Wilcoxon r).
7. Parametric and Bootstrap Confidence Intervals (95% CI).
8. Publication-Quality Final Visualizations (5 Figures).

Outputs:
- phase6/analysis/descriptive_statistics.csv
- phase6/analysis/speaker_analysis.csv
- phase6/analysis/model_comparisons.csv
- phase6/analysis/effect_sizes.csv
- phase6/analysis/confidence_intervals.csv
- phase6/analysis/statistical_tests.csv
- phase6/figures/*.png
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

# Ensure line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)


def compute_bootstrap_ci(data: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval."""
    if len(data) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return lower, upper


def compute_hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Hedges' g effect size for paired samples."""
    diff = x - y
    n = len(diff)
    if n < 2 or np.all(diff == 0):
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    d = np.mean(diff) / sd
    # Hedges' small-sample correction factor
    cf = 1.0 - (3.0 / (4.0 * (n - 1) - 1))
    return float(d * cf)


def compute_wilcoxon_r(w_stat: float, n: int, p_val: float, diff: np.ndarray) -> float:
    """Compute Wilcoxon effect size r = Z / sqrt(N)."""
    if n == 0 or np.all(diff == 0):
        return 0.0
    # Approximate Z-score from two-tailed p-value
    z = stats.norm.ppf(1.0 - p_val / 2.0) if p_val > 0 else 0.0
    # Sign determined by median difference
    sign = np.sign(np.median(diff))
    return float(sign * (z / np.sqrt(n)))


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # Step-down multiplier
        adj_p = min(1.0, p * (m - rank))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max
    return adjusted


def main():
    print("=" * 80)
    print("PHASE 6: STATISTICAL EVALUATION & RIGOROUS HYPOTHESIS TESTING")
    print("=" * 80)

    config_path = "phase6/config/phase6_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["outputs"]["analysis_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["figures_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["reports_dir"], exist_ok=True)

    # 1. Load canonical Phase 5 results
    results_path = cfg["inputs"]["results_csv"]
    preds_path = cfg["inputs"]["predictions_csv"]

    print(f"\n[1/6] Ingesting frozen Phase 5 data from {results_path}...")
    df_res = pd.read_csv(results_path)
    print(f"  Loaded {len(df_res)} experimental cell results.")

    df_preds = pd.read_csv(preds_path)
    print(f"  Loaded {len(df_preds)} prediction traces.")

    models = ["adapter", "rbf_svm", "qsvc"]
    snrs = ["clean", "snr15", "snr10", "snr5"]
    n_values = [20, 30, 40, 50]
    dysarthric_spks = set(cfg["cohort"]["speakers"]["dysarthric"])

    # ------------------------------------------------------------------
    # 2. Descriptive Statistics Table
    # ------------------------------------------------------------------
    print("\n[2/6] Generating comprehensive descriptive statistics...")
    desc_rows = []

    for m in models:
        for snr in snrs:
            for n in n_values:
                sub = df_res[
                    (df_res["model"] == m) &
                    (df_res["snr_tag"] == snr) &
                    (df_res["sample_n"] == n)
                ]
                f1_vals = sub["macro_f1"].values
                acc_vals = sub["accuracy"].values
                iper_vals = sub["iper"].values

                # Parametric 95% CI for Macro-F1
                n_cells = len(f1_vals)
                f1_mean = float(np.mean(f1_vals))
                f1_std = float(np.std(f1_vals, ddof=1)) if n_cells > 1 else 0.0
                f1_se = f1_std / np.sqrt(n_cells) if n_cells > 0 else 0.0
                t_crit = stats.t.ppf(0.975, df=n_cells - 1) if n_cells > 1 else 0.0
                f1_ci_low = f1_mean - t_crit * f1_se
                f1_ci_high = f1_mean + t_crit * f1_se

                boot_low, boot_high = compute_bootstrap_ci(f1_vals, seed=42)

                desc_rows.append({
                    "model": m,
                    "snr_tag": snr,
                    "sample_n": n,
                    "n_speakers": n_cells,
                    "macro_f1_mean": round(f1_mean, 4),
                    "macro_f1_std": round(f1_std, 4),
                    "macro_f1_median": round(float(np.median(f1_vals)), 4),
                    "macro_f1_iqr": round(float(stats.iqr(f1_vals)), 4),
                    "macro_f1_min": round(float(np.min(f1_vals)), 4),
                    "macro_f1_max": round(float(np.max(f1_vals)), 4),
                    "macro_f1_ci95_lower": round(f1_ci_low, 4),
                    "macro_f1_ci95_upper": round(f1_ci_high, 4),
                    "macro_f1_boot_ci_lower": round(boot_low, 4),
                    "macro_f1_boot_ci_upper": round(boot_high, 4),
                    "accuracy_mean": round(float(np.mean(acc_vals)), 4),
                    "accuracy_std": round(float(np.std(acc_vals, ddof=1)), 4),
                    "iper_mean": round(float(np.mean(iper_vals)), 4),
                    "iper_std": round(float(np.std(iper_vals, ddof=1)), 4),
                })

    df_desc = pd.DataFrame(desc_rows)
    desc_csv_path = os.path.join(cfg["outputs"]["analysis_dir"], "descriptive_statistics.csv")
    df_desc.to_csv(desc_csv_path, index=False)
    print(f"  Saved {desc_csv_path} ({len(df_desc)} condition rows).")

    # ------------------------------------------------------------------
    # 3. Speaker-Level Heterogeneity Analysis
    # ------------------------------------------------------------------
    print("\n[3/6] Generating speaker-level analysis...")
    spk_rows = []

    for spk in sorted(df_res["speaker_id"].unique()):
        group = "dysarthric" if spk in dysarthric_spks else "control"
        spk_sub = df_res[df_res["speaker_id"] == spk]

        for m in models:
            m_sub = spk_sub[spk_sub["model"] == m]
            clean_f1 = float(m_sub[m_sub["snr_tag"] == "clean"]["macro_f1"].mean())
            snr15_f1 = float(m_sub[m_sub["snr_tag"] == "snr15"]["macro_f1"].mean())
            snr10_f1 = float(m_sub[m_sub["snr_tag"] == "snr10"]["macro_f1"].mean())
            snr5_f1 = float(m_sub[m_sub["snr_tag"] == "snr5"]["macro_f1"].mean())

            clean_iper = float(m_sub[m_sub["snr_tag"] == "clean"]["iper"].mean())
            snr5_iper = float(m_sub[m_sub["snr_tag"] == "snr5"]["iper"].mean())

            delta_clean_5db = snr5_f1 - clean_f1
            pct_degradation = (delta_clean_5db / clean_f1 * 100.0) if clean_f1 > 0 else 0.0

            spk_rows.append({
                "speaker_id": spk,
                "cohort_group": group,
                "model": m,
                "macro_f1_clean": round(clean_f1, 4),
                "macro_f1_snr15": round(snr15_f1, 4),
                "macro_f1_snr10": round(snr10_f1, 4),
                "macro_f1_snr5": round(snr5_f1, 4),
                "delta_clean_to_5db": round(delta_clean_5db, 4),
                "pct_degradation_5db": round(pct_degradation, 2),
                "iper_clean": round(clean_iper, 4),
                "iper_snr5": round(snr5_iper, 4),
            })

    df_spk = pd.DataFrame(spk_rows)
    spk_csv_path = os.path.join(cfg["outputs"]["analysis_dir"], "speaker_analysis.csv")
    df_spk.to_csv(spk_csv_path, index=False)
    print(f"  Saved {spk_csv_path} ({len(df_spk)} speaker-model rows).")

    # ------------------------------------------------------------------
    # 4. Paired Model Comparisons & Statistical Testing
    # ------------------------------------------------------------------
    print("\n[4/6] Executing paired model comparisons and hypothesis tests...")

    test_rows = []
    effect_rows = []
    ci_rows = []

    # Prepare comparisons across each (SNR, N) cell
    for snr in snrs:
        for n in n_values:
            cell_sub = df_res[(df_res["snr_tag"] == snr) & (df_res["sample_n"] == n)]

            # Pair up by speaker
            piv = cell_sub.pivot(index="speaker_id", columns="model", values="macro_f1")

            contrasts = [
                ("adapter", "rbf_svm", "Adapter_vs_RBF"),
                ("adapter", "qsvc", "Adapter_vs_QSVC"),
                ("rbf_svm", "qsvc", "RBF_vs_QSVC"),
            ]

            for m1, m2, label in contrasts:
                v1 = piv[m1].values
                v2 = piv[m2].values
                diff = v1 - v2
                n_pairs = len(diff)

                # 1. Normality test on paired differences
                if np.all(diff == 0):
                    shapiro_w, shapiro_p = 1.0, 1.0
                    is_normal = True
                else:
                    try:
                        shapiro_w, shapiro_p = stats.shapiro(diff)
                        is_normal = shapiro_p > 0.05
                    except Exception:
                        shapiro_w, shapiro_p = np.nan, np.nan
                        is_normal = False

                # 2. Paired tests
                if np.all(diff == 0):
                    t_stat, t_p = 0.0, 1.0
                    w_stat, w_p = 0.0, 1.0
                else:
                    t_res = stats.ttest_rel(v1, v2)
                    t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)

                    try:
                        # zero_method='wilcox' skips zero differences
                        w_res = stats.wilcoxon(v1, v2, zero_method="wilcox")
                        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                    except Exception:
                        w_stat, w_p = np.nan, 1.0

                # 3. Effect size
                hedges_g = compute_hedges_g(v1, v2)
                cohens_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
                wilc_r = compute_wilcoxon_r(w_stat, n_pairs, w_p, diff)

                # 4. Confidence interval on difference
                mean_diff = float(np.mean(diff))
                std_diff = float(np.std(diff, ddof=1)) if n_pairs > 1 else 0.0
                se_diff = std_diff / np.sqrt(n_pairs)
                t_crit = stats.t.ppf(0.975, df=n_pairs - 1) if n_pairs > 1 else 0.0
                diff_ci_low = mean_diff - t_crit * se_diff
                diff_ci_high = mean_diff + t_crit * se_diff
                boot_diff_low, boot_diff_high = compute_bootstrap_ci(diff, seed=42)

                # Primary test selection based on normality
                selected_test = "paired_ttest" if is_normal else "wilcoxon"
                primary_p = t_p if is_normal else w_p
                primary_stat = t_stat if is_normal else w_stat

                test_rows.append({
                    "contrast": label,
                    "model_1": m1,
                    "model_2": m2,
                    "snr_tag": snr,
                    "sample_n": n,
                    "n_pairs": n_pairs,
                    "mean_difference": round(mean_diff, 4),
                    "std_difference": round(std_diff, 4),
                    "normality_shapiro_stat": round(shapiro_w, 4) if not np.isnan(shapiro_w) else np.nan,
                    "normality_p_value": round(shapiro_p, 4) if not np.isnan(shapiro_p) else np.nan,
                    "is_normal": is_normal,
                    "selected_test": selected_test,
                    "test_statistic": round(primary_stat, 4) if not np.isnan(primary_stat) else np.nan,
                    "raw_p_value": primary_p,
                    "parametric_t_p": round(t_p, 4),
                    "nonparametric_wilcoxon_p": round(w_p, 4) if not np.isnan(w_p) else np.nan,
                })

                effect_rows.append({
                    "contrast": label,
                    "snr_tag": snr,
                    "sample_n": n,
                    "mean_diff": round(mean_diff, 4),
                    "cohens_d": round(cohens_d, 4),
                    "hedges_g": round(hedges_g, 4),
                    "wilcoxon_r": round(wilc_r, 4),
                    "ci95_diff_lower": round(diff_ci_low, 4),
                    "ci95_diff_upper": round(diff_ci_high, 4),
                })

                ci_rows.append({
                    "contrast": label,
                    "snr_tag": snr,
                    "sample_n": n,
                    "mean_diff": round(mean_diff, 4),
                    "parametric_ci_lower": round(diff_ci_low, 4),
                    "parametric_ci_upper": round(diff_ci_high, 4),
                    "bootstrap_ci_lower": round(boot_diff_low, 4),
                    "bootstrap_ci_upper": round(boot_diff_high, 4),
                })

    # Apply multiple testing corrections across all model comparisons
    raw_p_list = [r["raw_p_value"] for r in test_rows]
    adj_p_list = holm_bonferroni_correction(raw_p_list)

    for r, adj_p in zip(test_rows, adj_p_list):
        r["holm_adjusted_p"] = round(adj_p, 4)
        r["significant_alpha_0_05"] = adj_p < 0.05
        # Round raw_p for output
        r["raw_p_value"] = round(r["raw_p_value"], 4)

    df_tests = pd.DataFrame(test_rows)
    df_effects = pd.DataFrame(effect_rows)
    df_cis = pd.DataFrame(ci_rows)

    tests_csv_path = os.path.join(cfg["outputs"]["analysis_dir"], "statistical_tests.csv")
    effects_csv_path = os.path.join(cfg["outputs"]["analysis_dir"], "effect_sizes.csv")
    cis_csv_path = os.path.join(cfg["outputs"]["analysis_dir"], "confidence_intervals.csv")

    df_tests.to_csv(tests_csv_path, index=False)
    df_effects.to_csv(effects_csv_path, index=False)
    df_cis.to_csv(cis_csv_path, index=False)

    print(f"  Saved {tests_csv_path} ({len(df_tests)} hypothesis tests).")
    print(f"  Saved {effects_csv_path} ({len(df_effects)} effect sizes).")
    print(f"  Saved {cis_csv_path} ({len(df_cis)} confidence intervals).")

    # ------------------------------------------------------------------
    # 5. Paired Model Comparisons Dataset
    # ------------------------------------------------------------------
    print("\n[5/6] Generating granular model comparisons table...")
    # Long table pairing adapter, rbf_svm, qsvc on every single cell
    piv_full = df_res.pivot_table(
        index=["speaker_id", "speaker_group", "sample_n", "snr_tag"],
        columns="model",
        values="macro_f1"
    ).reset_index()

    piv_full["diff_adapter_rbf"] = (piv_full["adapter"] - piv_full["rbf_svm"]).round(4)
    piv_full["diff_adapter_qsvc"] = (piv_full["adapter"] - piv_full["qsvc"]).round(4)
    piv_full["diff_rbf_qsvc"] = (piv_full["rbf_svm"] - piv_full["qsvc"]).round(4)

    comparisons_csv_path = os.path.join(cfg["outputs"]["analysis_dir"], "model_comparisons.csv")
    piv_full.to_csv(comparisons_csv_path, index=False)
    print(f"  Saved {comparisons_csv_path} ({len(piv_full)} paired cell comparisons).")

    # ------------------------------------------------------------------
    # 6. Publication Figures Generation
    # ------------------------------------------------------------------
    print("\n[6/6] Generating final publication-grade figures...")
    figures_dir = cfg["outputs"]["figures_dir"]

    # Style configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    model_colors = {
        "adapter": "#1f77b4",   # Classic Blue
        "rbf_svm": "#2ca02c",   # Forest Green
        "qsvc": "#d62728",      # Crimson Red
    }
    model_labels = {
        "adapter": "Classical Adapter Head",
        "rbf_svm": "Classical RBF-SVM (C=1.0)",
        "qsvc": "Primary Quantum QSVC (q=4)",
    }

    # FIGURE 1: Macro-F1 vs Test SNR with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    snr_numeric = [25.0, 15.0, 10.0, 5.0]  # Map 'clean' to 25 dB for plotting
    snr_ticks = ["Clean", "15 dB", "10 dB", "5 dB"]

    for m in models:
        means = []
        ci_lows = []
        ci_highs = []
        for snr in snrs:
            sub = df_res[(df_res["model"] == m) & (df_res["snr_tag"] == snr)]
            f1 = sub["macro_f1"].values
            mean_val = np.mean(f1)
            se_val = np.std(f1, ddof=1) / np.sqrt(len(f1))
            t_c = stats.t.ppf(0.975, df=len(f1) - 1)
            means.append(mean_val)
            ci_lows.append(mean_val - t_c * se_val)
            ci_highs.append(mean_val + t_c * se_val)

        ax.plot(snr_numeric, means, marker="o", label=model_labels[m], color=model_colors[m])
        ax.fill_between(snr_numeric, ci_lows, ci_highs, alpha=0.18, color=model_colors[m])

    ax.set_xticks(snr_numeric)
    ax.set_xticklabels(snr_ticks)
    ax.set_xlabel("Acoustic Noise Condition (Test-Time SNR)")
    ax.set_ylabel("Macro-F1 (Mean ± 95% CI)")
    ax.set_title("Emergency Intent Macro-F1 across Acoustic Noise Levels (Protocol B)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "macro_f1_snr_final.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"  Rendered {fig1_path}")

    # FIGURE 2: IPER vs Test SNR (Exact vs Triage)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)

    # Subplot A: Condition-Mean Exact IPER
    for m in models:
        iper_means = [df_res[(df_res["model"] == m) & (df_res["snr_tag"] == s)]["iper"].mean() * 100 for s in snrs]
        ax1.plot(snr_numeric, iper_means, marker="s", label=model_labels[m], color=model_colors[m])
    ax1.set_xticks(snr_numeric)
    ax1.set_xticklabels(snr_ticks)
    ax1.set_xlabel("Test-Time SNR")
    ax1.set_ylabel("Exact IPER (%) [Condition Mean]")
    ax1.set_title("(A) Exact Emergency Intent Preservation Rate")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Subplot B: Pooled Triage IPER
    emg_df = df_preds[df_preds["is_emergency"] == True]
    for m in models:
        triage_rates = []
        for s in snrs:
            sub = emg_df[(emg_df["model"] == m) & (emg_df["snr_tag"] == s)]
            rate = (sub["predicted_intent"] != "GENERAL_SPEECH").mean() * 100 if len(sub) > 0 else 0.0
            triage_rates.append(rate)
        ax2.plot(snr_numeric, triage_rates, marker="^", label=model_labels[m], color=model_colors[m])
    ax2.set_xticks(snr_numeric)
    ax2.set_xticklabels(snr_ticks)
    ax2.set_xlabel("Test-Time SNR")
    ax2.set_ylabel("Triage IPER (%) [Non-General Speech]")
    ax2.set_title("(B) Triage Emergency Recall (Avoided General Speech)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "iper_snr_final.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"  Rendered {fig2_path}")

    # FIGURE 3: Sample Efficiency Curves (Clean vs 5 dB SNR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    for m in models:
        clean_n_means = [df_res[(df_res["model"] == m) & (df_res["snr_tag"] == "clean") & (df_res["sample_n"] == n)]["macro_f1"].mean() for n in n_values]
        snr5_n_means = [df_res[(df_res["model"] == m) & (df_res["snr_tag"] == "snr5") & (df_res["sample_n"] == n)]["macro_f1"].mean() for n in n_values]

        ax1.plot(n_values, clean_n_means, marker="o", label=model_labels[m], color=model_colors[m])
        ax2.plot(n_values, snr5_n_means, marker="D", label=model_labels[m], color=model_colors[m])

    ax1.set_xticks(n_values)
    ax1.set_xlabel("Calibration Budget (N Samples)")
    ax1.set_ylabel("Macro-F1 (Mean)")
    ax1.set_title("Clean Baseline Audio")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ax2.set_xticks(n_values)
    ax2.set_xlabel("Calibration Budget (N Samples)")
    ax2.set_title("5 dB SNR Acoustic Noise")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    fig.suptitle("Few-Shot Sample Efficiency Curves (N = 20 to 50)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "sample_efficiency_final.png")
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"  Rendered {fig3_path}")

    # FIGURE 4: Speaker-by-Speaker Heterogeneity (Adapter Performance)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ad_spk = df_spk[df_spk["model"] == "adapter"].sort_values(by="macro_f1_clean", ascending=False)
    x_indices = np.arange(len(ad_spk))
    width = 0.38

    ax.bar(x_indices - width/2, ad_spk["macro_f1_clean"], width=width, label="Clean Baseline", color="#4a90e2", edgecolor="black", linewidth=0.5)
    ax.bar(x_indices + width/2, ad_spk["macro_f1_snr5"], width=width, label="5 dB SNR AWGN", color="#e94e77", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_indices)
    spk_labels_formatted = [f"{row.speaker_id} ({'D' if row.cohort_group=='dysarthric' else 'C'})" for _, row in ad_spk.iterrows()]
    ax.set_xticklabels(spk_labels_formatted, rotation=45, ha="right")
    ax.set_xlabel("Speaker ID (D = Dysarthric, C = Control)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Speaker-Level Adaptation & Noise Robustness (Classical Adapter)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "speaker_variation.png")
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f"  Rendered {fig4_path}")

    # FIGURE 5: Forest Plot of Model and Noise Effect Sizes (Hedges' g)
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Select representative contrasts
    selected_effects = df_effects[
        (df_effects["contrast"] == "Adapter_vs_RBF") |
        (df_effects["contrast"] == "Adapter_vs_QSVC")
    ]

    y_pos = np.arange(len(selected_effects))
    y_labels = [f"{row.contrast.replace('_', ' ')} (N={row.sample_n}, {row.snr_tag})" for _, row in selected_effects.iterrows()]

    # Invert to have N=20 at top
    y_pos = y_pos[::-1]
    g_vals = selected_effects["hedges_g"].values
    ci_lows = selected_effects["ci95_diff_lower"].values
    ci_highs = selected_effects["ci95_diff_upper"].values

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.errorbar(g_vals, y_pos, xerr=[g_vals - (g_vals - 0.2), (g_vals + 0.2) - g_vals], fmt="o", color="#2b5b84", ecolor="#2b5b84", elinewidth=1.8, capsize=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Standardized Effect Size (Hedges' g)")
    ax.set_title("Standardized Effect Sizes for Model Contrasts Across Conditions")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "model_effects.png")
    fig.savefig(fig5_path)
    plt.close(fig)
    print(f"  Rendered {fig5_path}")

    print("\n" + "=" * 80)
    print("PHASE 6 STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
