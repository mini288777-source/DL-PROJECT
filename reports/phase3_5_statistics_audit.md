# Statistical Aggregation & Speaker-Level Variation Audit

**Date:** 2026-09-30  
**Phase:** 3.5 — Research Validity, Consistency & Phase 4 Readiness Gate  
**Target File:** `reports/phase3_5_statistics_audit.md`  

---

## 1. Executive Summary

This report reviews the statistical aggregation methodology used in Phase 3 and establishes the aggregation rules for subsequent comparative phases.

* **Primary Experimental Unit:** The **individual speaker personalization task** under sample budget $N$.
* **Primary Aggregation Mode:** **Macro-averaging across speakers** ($M = 15$ speakers per method-budget condition).
* **Dispersion Metrics:** Empirical standard deviation ($s$) and $95\%$ Student's $t$-distribution Confidence Intervals ($CI_{95}$).
* **Methodological Finding:** Sample-weighted pooling (merging all 21,268 test predictions into a single global confusion matrix) is **statistically invalid** as a primary metric for personalized assistive speech because it allows speakers with hundreds of test utterances (e.g., MC01 with 608 test samples) to dominate speakers with severe dysarthria who provided fewer samples (e.g., F01 with 68 test samples).

---

## 2. Statistical Formulations

### 2.1 Speaker-Level Aggregation
For each method $m \in \{\text{adapter}, \text{rbf\_svm}\}$ and each sample budget $N \in \{20, 30, 40, 50\}$, let $y_{s, N}$ denote the performance metric (e.g., Macro-F1) obtained on the held-out test set of speaker $s \in \{1, \dots, M\}$, where $M = 15$.

1. **Cohort Mean:**
$$\bar{y}_{m, N} = \frac{1}{M} \sum_{s=1}^M y_{s, N}$$

2. **Sample Standard Deviation:**
$$s_{m, N} = \sqrt{\frac{1}{M - 1} \sum_{s=1}^M (y_{s, N} - \bar{y}_{m, N})^2}$$

3. **Standard Error of the Mean (SEM):**
$$\text{SEM}_{m, N} = \frac{s_{m, N}}{\sqrt{M}}$$

4. **95% Confidence Interval:**
$$CI_{95\%} = \bar{y}_{m, N} \pm t_{0.025, M-1} \cdot \text{SEM}_{m, N}$$
For $M = 15$, degrees of freedom $\nu = 14$, and the critical value is $t_{0.025, 14} \approx 2.145$.

---

## 3. Comparison of Aggregation Schemes

To illustrate why macro-averaging across speakers is scientifically necessary, consider the sample counts across the 15 TORGO speakers:

| Speaker ID | Pathology Group | Held-Out Test Samples | Weight in Sample-Pooled Metric | Weight in Speaker-Macro Metric |
|---|---|---|---|---|
| **MC01** | Control | 608 | **14.3%** | **6.67%** |
| **MC04** | Control | 496 | **11.7%** | **6.67%** |
| **M05** | Dysarthric (Severe) | 479 | **11.3%** | **6.67%** |
| **FC03** | Control | 490 | **11.5%** | **6.67%** |
| ... | ... | ... | ... | ... |
| **F01** | Dysarthric (Severe) | 68 | **1.6%** | **6.67%** |
| **F03** | Dysarthric (Moderate)| 127 | **3.0%** | **6.67%** |
| **TOTAL** | — | **4,254 / run** | **100.0%** | **100.0%** |

Under sample-weighted pooling:
* A high score on MC01 (control speaker, clear speech, 608 samples) would heavily inflate the reported metric.
* Severe pathology patients (e.g., F01, who produce fewer recordings due to severe physical fatigue) would represent only $1.6\%$ of the evaluation.
* **Speaker-macro averaging ensures that every speaker's assistive personalization outcome is valued equally ($1/15 = 6.67\%$).**

---

## 4. Phase 3 Statistical Baseline Summary

From `results/phase3/classical_sample_efficiency_aggregated.csv`:

| Method | Budget ($N$) | Mean Macro-F1 | Std Dev ($s$) | 95% Confidence Interval | Mean Accuracy |
|---|---|---|---|---|---|
| **Classical Adapter** | 20 | 0.1466 | 0.0413 | $[0.1237, 0.1695]$ | $93.74\% \pm 1.25\%$ |
| **Classical Adapter** | 30 | 0.1453 | 0.0255 | $[0.1312, 0.1594]$ | $93.78\% \pm 1.15\%$ |
| **Classical Adapter** | 40 | 0.1472 | 0.0418 | $[0.1241, 0.1704]$ | $93.63\% \pm 1.21\%$ |
| **Classical Adapter** | 50 | 0.1410 | 0.0168 | $[0.1317, 0.1503]$ | $93.84\% \pm 1.07\%$ |
| **Classical RBF-SVM** | 20 | 0.1390 | 0.0026 | $[0.1376, 0.1405]$ | $94.12\% \pm 0.81\%$ |
| **Classical RBF-SVM** | 30 | 0.1390 | 0.0026 | $[0.1376, 0.1405]$ | $94.12\% \pm 0.81\%$ |
| **Classical RBF-SVM** | 40 | 0.1390 | 0.0026 | $[0.1376, 0.1405]$ | $94.12\% \pm 0.81\%$ |
| **Classical RBF-SVM** | 50 | 0.1390 | 0.0026 | $[0.1376, 0.1405]$ | $94.12\% \pm 0.81\%$ |

## 5. Audit Conclusion

**STATISTICAL METHODOLOGY STATUS: PASS**  
The statistical reporting pipeline correctly prioritizes speaker-level macro aggregation and reports both central tendencies and confidence intervals.
