# Phase 6: Statistical Analysis & Intent-Preserving Evaluation Engine

This directory contains the frozen analytical code, experimental outputs, statistical tests, effect size measurements, confidence intervals, figures, and comprehensive scientific reports for **Phase 6** of the research project:

> *"Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication"*

---

## 1. Directory Structure

```
phase6/
├── README.md                               # This governance and navigational guide
├── config/
│   └── phase6_config.yaml                  # Frozen statistical and evaluation configuration
├── analysis/
│   ├── descriptive_statistics.csv          # 48 conditions (Mean, STD, Median, IQR, Min, Max, 95% CIs)
│   ├── speaker_analysis.csv                # 45 speaker-model rows (Dysarthric vs Control, degradation)
│   ├── model_comparisons.csv               # 240 paired speaker-level contrasts (Adapter, RBF, QSVC)
│   ├── effect_sizes.csv                    # 48 contrasts (Cohen's d, Hedges' g, Wilcoxon r, 95% CI)
│   ├── confidence_intervals.csv            # 48 parametric & bootstrap 95% confidence intervals
│   └── statistical_tests.csv               # 48 hypothesis tests (Shapiro-Wilk, Paired t/Wilcoxon, Holm p)
├── figures/
│   ├── macro_f1_snr_final.png              # Primary metric vs Test SNR degradation
│   ├── iper_snr_final.png                  # Emergency Intent Preservation vs Test SNR
│   ├── sample_efficiency_final.png         # Macro-F1 across adaptation budgets N in {20, 30, 40, 50}
│   ├── speaker_variation.png               # Speaker-level acoustic degradation profiles
│   └── model_effects.png                   # Paired contrast distributions and effect sizes
└── reports/
    ├── phase6_analysis_report.md           # Master scientific and statistical findings report
    └── phase6_reproducibility_report.md    # Computational environment, seeds, and verification audit
```

---

## 2. Research Questions Addressed

- **RQ6.1 (Emergency Intent Preservation):** Under what conditions and to what degree is emergency intent preserved?
  - *Finding:* Exact Intent-Preserving Emergency Recognition (IPER) was $1.56\% \pm 3.77\%$ on Clean, dropping to $0.98\% \pm 3.53\%$ at 5 dB SNR for the Classical Adapter. Non-linear margin classifiers (RBF-SVM and QSVC) exhibited 100% majority-class collapse ($0.0\%$ IPER across all conditions) due to extreme class imbalance (~95% General Speech).
- **RQ6.2 (Noise Robustness):** How does recognition performance degrade under acoustic noise?
  - *Finding:* Classical Adapter Macro-F1 declined monotonically from $0.1450 \pm 0.0181$ (Clean) to $0.1367 \pm 0.0125$ (5 dB SNR). The decline was driven primarily by speaker `FC02` ($-28.94\%$ relative drop).
- **RQ6.3 (Sample Efficiency):** Does increasing calibration budget from $N=20$ to $N=50$ improve recognition?
  - *Finding:* In the Adapter, Macro-F1 was $0.1418 \pm 0.0157$ at $N=20$ and $0.1385 \pm 0.0116$ at $N=50$. Performance plateaued rapidly; increasing $N$ under extreme class imbalance did not provide statistically significant improvements.
- **RQ6.4 (Quantum Advantage Hypothesis):** Does the 4-qubit ZZ-feature map QSVC outperform matched classical baselines?
  - *Finding:* Under the evaluated Phase 5/6 configuration, the 4-qubit QSVC produced **numerically identical** predictions and metrics to the Classical RBF-SVM across all 81,840 evaluated test instances ($p = 1.000$, Hedges' $g = 0.000$). No quantum advantage was observed.
- **RQ6.5 (Speaker Heterogeneity):** How consistent are the observed effects across speakers?
  - *Finding:* Substantial speaker-level variance was observed. Dysarthric speakers showed moderate stability across noise, while control speaker `FC02` exhibited acute sensitivity to acoustic perturbation.

---

## 3. Statistical Methodology

1. **Normality Assessment:** Shapiro-Wilk test applied to paired differences across all 15 speakers for each condition. For non-normal distributions ($p < 0.05$), non-parametric Wilcoxon signed-rank tests were applied.
2. **Multiple Comparisons Correction:** Holm-Bonferroni step-down correction applied across the family of hypothesis tests.
3. **Effect Sizes:** Standardized Hedges' $g$ (small-sample corrected) and Cohen's $d$ computed for all paired contrasts.
4. **Confidence Intervals:** Both parametric Student's $t$ (95%) and non-parametric empirical bootstrap (2,000 resamples, 95% BCa percentile) intervals reported.
