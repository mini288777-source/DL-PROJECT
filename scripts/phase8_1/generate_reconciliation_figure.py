"""
Generate detailed speaker FER comparison figure for Phase 8.1 final reconciliation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    out_dir = os.path.join(REPO_ROOT, "results", "phase8_1", "final_reconciliation")
    spk_csv = os.path.join(out_dir, "speaker_fer_reconciliation.csv")
    df = pd.read_csv(spk_csv)

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.figure(figsize=(12, 6), dpi=300)

    # Plot Track A and Track B side-by-side per speaker
    df_plot = df[df["candidate_id"].isin(["Track_A_Baseline", "Track_B_Quantum"])].copy()
    
    palette = {"Track_A_Baseline": "#1f77b4", "Track_B_Quantum": "#2ca02c"}
    ax = sns.barplot(data=df_plot, x="speaker_id", y="FER", hue="candidate_id", palette=palette)

    plt.axhline(0.15, color="#d32f2f", linestyle="--", linewidth=2, label="15.0% Safety Ceiling Bound")

    # Annotate M05 exceeding bound
    plt.annotate("M05 EXCEEDS 15% BOUND:\nTrack A: 15.56% (42/270)\nTrack B: 15.19% (41/270)",
                 xy=(10.5, 0.152), xytext=(8.0, 0.18),
                 arrowprops=dict(facecolor="#d32f2f", shrink=0.05, width=1.5, headwidth=8),
                 bbox=dict(boxstyle="round,pad=0.5", fc="#ffebee", ec="#c62828", lw=1.5),
                 fontsize=9, fontweight="bold")

    plt.ylabel("False Emergency Rate (FER = FP / [FP + TN])", fontsize=11)
    plt.xlabel("Speaker ID (TORGO Corpus)", fontsize=11)
    plt.ylim(0, 0.22)
    plt.title("Speaker-Specific False Emergency Rate (FER) Across 15 Speakers\n(Pooled FER < 15% Masks Individual Violation for Dysarthric Speaker M05)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speaker_fer_reconciliation.png"))
    plt.close()
    print("Saved results/phase8_1/final_reconciliation/speaker_fer_reconciliation.png.")

if __name__ == "__main__":
    main()
