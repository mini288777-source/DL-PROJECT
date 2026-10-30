# Subphase 6.5.5-C: Quantum Statistical Closure & Final Audit Report

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 5, 2026  
**Status:** COMPLETE (Analytical Closure — No New Model Training)  
**Historical Benchmarks (Phases 1–6):** PRESERVED, IMMUTABLE, BITWISE INTACT  
**Recovery Protocol:** Track B (`phase6_5/splits/recovery_splits_hierarchical_ready.json`), 15 TORGO speakers, calibration budgets $N \in \{20, 30, 40, 50\}$  
**Matched Experimental Units:** Exactly 60 conditions ($15 \text{ speakers} \times 4 \text{ calibration budgets}$)  

---

## 1. Executive Summary & Purpose of Closure

Subphase 6.5.5-C executes the definitive statistical closure and audit of Subphase 6.5.5. In accordance with strict experimental governance, **no new data collection, no new split generation, and no new model training occurred**. All statistical analyses are reconstructed directly from the frozen, audited condition-level predictions and metrics in `results/phase6_5_5_speaker_results.csv`.

### Core Closure Objectives Achieved:
1. **Primary Kernel Comparison Verified:** Confirmed that QSVC Q8-R1 and Classical RBF-SVM ($d=8$) are statistically indistinguishable across primary end-to-end metrics ($p > 0.05$ after Holm-Bonferroni correction).
2. **Missing Adapter Comparison Completed:** Evaluated QSVC Q8-R1 vs. Weighted Classical Adapter ($d=8$) across all 7 pre-specified metrics. Classical Adapter demonstrates a statistically supported advantage on emergency recall ($45.81\%$ vs. $27.52\%$, $p = 0.000125$) and IPER ($14.21\%$ vs. $8.25\%$, $p = 0.011868$).
3. **Quantum Ablations Closed:** Demonstrated that neither expanding the feature map ($4 \to 8$ qubits) nor doubling circuit depth (reps $1 \to 2$) provides a statistically significant improvement in end-to-end emergency communication performance.
4. **Computational Cost Reporting Audited:** Reconciled the arithmetic of execution time, establishing that QSVC Q8-R1 is approximately **$773\times$ slower per test evaluation** and **$213\times$ slower across the 60-condition experiment wall time** compared to classical RBF-SVM.
5. **Final Decision Frozen:** Reaffirmed **CASE B (No statistically supported difference)** for the primary kernel comparison, with classical adapter superiority on safety-critical recall.

---

## 2. Experimental Unit & Data Integrity Audit

### 2.1 Definition of the Experimental Unit
- In all paired statistical hypothesis testing, the **experimental unit is strictly the matched condition** ($1 \text{ speaker} \times 1 \text{ sample budget } N$).
- Individual audio utterances are **not** treated as independent experimental units, avoiding pseudoreplication and artificial inflation of statistical degrees of freedom.
- Across 15 TORGO speakers (8 dysarthric, 7 control) and 4 budgets ($N \in \{20, 30, 40, 50\}$), there are **exactly 60 matched experimental conditions**.
- Every condition uses identical adaptation recording IDs, identical held-out test recording IDs, identical train-only PCA projections, and identical random seeds ($42$).
- Verified: Zero missing conditions, zero duplicate pairs.

---

## 3. Primary Matched Kernel Comparison Audit: QSVC Q8-R1 vs. Classical RBF-SVM ($d=8$)

The primary scientific question evaluates whether substituting a classical Gaussian RBF kernel with a quantum state-fidelity kernel (`ZZFeatureMap`, linear entanglement, $q=8, \text{reps}=1$) improves classification performance on frozen 8-dimensional PCA projections.

### 3.1 Empirical Results & Paired Hypothesis Tests ($N=60$ Paired Conditions)

| Metric | QSVC Q8-R1 Mean | RBF-SVM $d=8$ Mean | Mean Difference ($\Delta$) | 95% Confidence Interval | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$-value | Statistical Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051$ | $0.1087$ | $-0.0036$ | $[-0.0083, +0.0010]$ | $475.0$ | $0.1714$ | **$0.3428$** | **NO STATISTICAL DIFFERENCE** |
| **Emergency Recall** | $27.52\%$ | $31.89\%$ | $-4.37\%$ | $[-8.14\%, -0.60\%]$ | $20.0$ | $0.0230$ | **$0.0689$** | **NO STATISTICAL DIFFERENCE** |
| **False-Emergency Rate** | $28.12\%$ | $29.32\%$ | $-1.20\%$ | $[-4.48\%, +2.08\%]$ | $512.0$ | $0.9505$ | **$0.9505$** | **NO STATISTICAL DIFFERENCE** |
| **Emergency Precision** | $0.81\%$ | $0.94\%$ | $-0.13\%$ | $[-0.32\%, +0.06\%]$ | $45.0$ | $0.2185$ | **$0.3428$** | **NO STATISTICAL DIFFERENCE** |
| **End-to-End IPER** | $8.25\%$ | $11.69\%$ | $-3.44\%$ | $[-6.11\%, -0.77\%]$ | $29.5$ | $0.0145$ | **$0.0582$** | **NO STATISTICAL DIFFERENCE** |
| **Stage-2 Feasible Macro-F1**| $0.1252$ | $0.1265$ | $-0.0013$ | $[-0.0152, +0.0126]$ | $380.0$ | $0.7852$ | **$0.9505$** | **NO STATISTICAL DIFFERENCE** |
| **Stage-2 Balanced Accuracy** | $32.53\%$ | $33.91\%$ | $-1.37\%$ | $[-8.22\%, +5.48\%]$ | $409.0$ | $0.1004$ | **$0.2008$** | **NO STATISTICAL DIFFERENCE** |

### 3.2 Audit Findings on Primary Comparison
1. All 7 evaluation metrics yield Holm-adjusted $p$-values $> 0.05$.
2. On the primary overall metric (End-to-End Macro-F1), the paired difference is $\Delta = -0.0036$ ($p=0.3428$), which is statistically indistinguishable from zero.
3. On Emergency Recall, classical RBF-SVM exhibits a marginal numerical advantage ($+4.37\%$, raw $p=0.0230$), but this difference does not survive family-wise Holm-Bonferroni error control ($p_{\text{adj}} = 0.0689$).
4. On False-Emergency Rate, both models yield effectively identical false alarm rates ($28.12\%$ vs. $29.32\%$, $p=0.9505$).
5. **Conclusion:** QSVC Q8-R1 and Classical RBF-SVM ($d=8$) are statistically indistinguishable under the evaluated protocol (**CASE B**).

---

## 4. Missing Primary Closure: QSVC Q8-R1 vs. Weighted Classical Adapter ($d=8$)

The comparison between QSVC Q8-R1 and the parameterized Classical Adapter ($d=8$) is critical because the classical adapter optimizes class-conditional cross-entropy directly rather than finding maximum-margin hyperplanes.

### 4.1 Paired Statistical Comparison Results ($N=60$ Matched Conditions)

| Metric | QSVC Q8-R1 Mean | Classical Adapter $d=8$ Mean | Mean Difference ($\Delta$) | 95% Confidence Interval | Cohen's $d_z$ | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$-value | Statistical Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051$ | $0.0949$ | $+0.0102$ | $[+0.0008, +0.0196]$ | $+0.2742$ | $547.0$ | $0.0067$ | **$0.0202$** | **QSVC Macro-F1 Higher** |
| **Emergency Recall** | $27.52\%$ | $45.81\%$ | $-18.29\%$ | $[-25.23\%, -11.35\%]$ | $-0.6668$ | $134.0$ | $2.5 \times 10^{-5}$ | **$0.000125$** | **Classical Adapter Superior** |
| **Emergency Precision** | $0.81\%$ | $2.03\%$ | $-1.22\%$ | $[-1.69\%, -0.75\%]$ | $-0.6549$ | $109.5$ | $1.0 \times 10^{-6}$ | **$7.0 \times 10^{-6}$** | **Classical Adapter Superior** |
| **False-Emergency Rate** | $28.12\%$ | $49.82\%$ | $-21.71\%$ | $[-28.73\%, -14.68\%]$ | $-0.7815$ | $223.0$ | $3.0 \times 10^{-6}$ | **$1.8 \times 10^{-5}$** | **QSVC Lower FPR** |
| **End-to-End IPER** | $8.25\%$ | $14.21\%$ | $-5.97\%$ | $[-9.46\%, -2.48\%]$ | $-0.4325$ | $134.0$ | $0.002967$ | **$0.011868$** | **Classical Adapter Superior** |
| **Stage-2 Feasible Macro-F1**| $0.1252$ | $0.1824$ | $-0.0571$ | $[-0.0975, -0.0168]$ | $-0.3583$ | $488.0$ | $0.007154$ | **$0.020241$** | **Classical Adapter Superior** |
| **Stage-2 Balanced Accuracy** | $32.53\%$ | $33.57\%$ | $-1.04\%$ | $[-7.77\%, +5.70\%]$ | $-0.0390$ | $507.0$ | $0.546120$ | **$0.546120$** | **NO STATISTICAL DIFFERENCE** |

### 4.2 Rigorous Safety-Critical Interpretation
- **Emergency Recall (Safety Priority #1):** Weighted Adapter achieved a statistically supported advantage over QSVC ($\Delta = -18.29$ percentage points, $p = 0.000125$, moderate-to-large effect size $d_z = -0.6668$).
- **Intent Preservation (IPER Priority #2):** Weighted Adapter achieved a statistically supported advantage ($\Delta = -5.97$ percentage points, $p = 0.011868$, $d_z = -0.4325$).
- **Tradeoff Profile:** QSVC operates conservatively, generating fewer false alarms ($28.12\%$ vs. $49.82\%$) and slightly higher seven-class Macro-F1 ($0.1051$ vs. $0.0949$). However, this conservative threshold causes it to miss **$72.48\%$ of all emergency utterances**, compared to $54.19\%$ missed by the Adapter.
- **Statement for Final Report:**
  > *“Weighted Adapter showed a statistically supported advantage over QSVC on emergency recall ($45.81\%$ vs. $27.52\%$, $p < 0.001$) and intent preservation ($14.21\%$ vs. $8.25\%$, $p < 0.05$) under the evaluated matched protocol, while QSVC achieved a lower false-emergency rate and higher Macro-F1.”*

---

## 5. Quantum Ablations Audit

### 5.1 Qubit Scaling Ablation: Q8-R1 ($q=8$) vs. Q4-R1 ($q=4$)
Evaluates whether expanding the simulated Hilbert space from $2^4 = 16$ dimensions to $2^8 = 256$ dimensions provides measurable benefit:

| Metric | Q8-R1 ($q=8$) | Q4-R1 ($q=4$) | Difference ($\Delta$) | 95% CI | Wilcoxon $W$ | Raw $p$ | Holm $p$ | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051$ | $0.1108$ | $-0.0057$ | $[-0.0102, -0.0012]$ | $594.5$ | $0.2025$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **Emergency Recall** | $27.52\%$ | $27.25\%$ | $+0.27\%$ | $[-4.08\%, +4.61\%]$ | $94.0$ | $0.9679$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **False-Emergency Rate** | $28.12\%$ | $27.97\%$ | $+0.15\%$ | $[-2.76\%, +3.06\%]$ | $494.0$ | $0.6114$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **End-to-End IPER** | $8.25\%$ | $6.61\%$ | $+1.64\%$ | $[-1.70\%, +4.98\%]$ | $79.0$ | $0.3311$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **Stage-2 Feasible F1** | $0.1252$ | $0.1455$ | $-0.0202$ | $[-0.0503, +0.0098]$ | $613.0$ | $0.2648$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **Stage-2 Balanced Acc** | $32.53\%$ | $27.01\%$ | $+5.52\%$ | $[-1.33\%, +12.36\%]$ | $402.5$ | $0.1313$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |

- **Finding:** Under family-wise Holm-Bonferroni correction, **zero metrics show a statistically significant difference** between 4-qubit and 8-qubit quantum kernels ($p_{\text{adj}} = 1.0$ across all 7 metrics). Expanding qubit count does not provide end-to-end performance improvement.

### 5.2 Circuit Depth Ablation: Q8-R1 (reps=1) vs. Q8-R2 (reps=2)
Evaluates whether doubling the entangling circuit depth at $q=8$ improves representation capacity:

| Metric | Q8-R1 (reps=1) | Q8-R2 (reps=2) | Difference ($\Delta$) | 95% CI | Wilcoxon $W$ | Raw $p$ | Holm $p$ | Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051$ | $0.1070$ | $-0.0019$ | $[-0.0034, -0.0004]$ | $259.0$ | $0.0021$ | **$0.0273$** | Negligible shift (+0.0019) |
| **False-Emergency Rate** | $28.12\%$ | $26.73\%$ | $+1.38\%$ | $[+0.37\%, +2.40\%]$ | $162.0$ | $0.0005$ | **$0.0070$** | Slightly lower FPR (-1.38%) |
| **Emergency Recall** | $27.52\%$ | $26.43\%$ | $+1.08\%$ | $[-0.86\%, +3.03\%]$ | $8.0$ | $0.3105$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **End-to-End IPER** | $8.25\%$ | $7.65\%$ | $+0.59\%$ | $[-0.90\%, +2.08\%]$ | $11.0$ | $0.6115$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **Stage-2 Feasible F1** | $0.1252$ | $0.1296$ | $-0.0044$ | $[-0.0302, +0.0214]$ | $290.5$ | $0.6881$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |
| **Stage-2 Balanced Acc** | $32.53\%$ | $30.48\%$ | $+2.05\%$ | $[-3.41\%, +7.51\%]$ | $215.0$ | $0.5176$ | **$1.0$** | **NO STATISTICAL DIFFERENCE** |

- **Finding:** Increasing circuit depth from reps=1 to reps=2 yields a statistically detectable but practically negligible shift in Macro-F1 ($+0.0019$) and False-Emergency Rate ($-1.38\%$), accompanied by a non-significant drop in emergency recall. It nearly doubles execution time ($70.22$s $\to 128.04$s) with no practical benefit.

---

## 6. Computational Cost & Runtime Audit

To resolve ambiguity in runtime reporting, per-evaluation inference speed is strictly distinguished from total experiment wall time:

| Measurement Basis | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) | QSVC Q8-R2 | QSVC / RBF Ratio | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Per-Evaluation Inference Time** | **$1.0816$ s** | **$0.0014$ s** | $0.0006$ s | $1.9761$ s | **$\mathbf{772.6\times}$** | QSVC is $\sim 773\times$ slower per test evaluation |
| **Total 60-Condition Wall Time** | **$70.22$ s** | **$0.33$ s** | $15.79$ s | $128.04$ s | **$\mathbf{212.8\times}$** | QSVC total wall time is $\sim 213\times$ slower overall |
| **Training Time per Condition** | $0.0888$ s | $0.0040$ s | $0.2625$ s | $0.1578$ s | **$22.2\times$** | QSVC kernel construction is $22\times$ slower than RBF fit |

### Runtime Audit Rules Enforced:
1. The **$\sim 773\times$ ratio** applies strictly to *per-evaluation test inference time* ($1.0816$s vs. $0.0014$s).
2. The **$\sim 213\times$ ratio** applies strictly to *total 60-condition experiment wall time* ($70.22$s vs. $0.33$s).
3. Neither ratio justifies deployment: classical RBF-SVM completes the full 60-condition sweep in one-third of a second, whereas QSVC requires over a minute in statevector simulation.

---

## 7. Quantum Kernel Integrity & Eigenvalue Audit

- **Matrix Inventory:** All 180 cached quantum training kernels and 180 test transition matrices verified in [`results/phase6_5_5/kernel_cache/`](file:///d:/DL/results/phase6_5_5/kernel_cache/).
- **Numerical Health:**
  - Finite Values: 180 / 180 (zero NaNs, zero Infs).
  - Maximum Symmetry Error: $\max |K - K^T| = 0.0$.
  - Maximum Diagonal Deviation: $\max |K_{ii} - 1.0| = 0.0$.
  - Positive Semi-Definiteness: Minimum eigenvalue across all 180 matrices is $\ge 0.000186 > 0$ (Mean $\lambda_{\min} = 0.5589$, Max $\lambda_{\min} = 0.9599$).
  - Zero matrices required numerical clipping, ridge regularization, or PSD repair.

---

## 8. Scientific Language & Claim Audit

All Phase 6.5.5 claims have been audited against overclaiming rules:
- **Replaced:** Universal assertions such as *"quantum kernels do not work"* or *"quantum is inferior"* have been strictly eliminated.
- **Enforced Formulation:**
  - *"Under the evaluated recovery protocol and tested configurations, the fidelity quantum kernel demonstrated no statistically supported performance advantage over the matched classical RBF-SVM."*
  - *"Statevector-simulated QSVC was approximately $773\times$ slower per test evaluation than classical RBF-SVM in the measured implementation."*
  - *"Weighted Classical Adapter achieved statistically significantly higher emergency recall ($45.81\%$ vs. $27.52\%$, $p < 0.001$), while QSVC achieved lower false-alarm rates."*
  - *"These findings are specific to the TORGO cohort, $N \le 50$, Whisper-tiny representations, and ZZ feature maps; they do not establish universal bounds on quantum machine learning."*

---

## 9. Final Decision & Handoff to Phase 6.5.6

### Final Classification: CASE B (No Statistically Supported Difference)
- **Primary Comparison (QSVC Q8-R1 vs. Classical RBF-SVM $d=8$):** **CASE B**.
- **Secondary Comparison (QSVC Q8-R1 vs. Classical Adapter $d=8$):** **Classical Advantage on Safety-Critical Recall**.
- **Phase 6.5.5 Status:** **FROZEN & CLOSED**.
- **Phase 6.5.6 Status:** **NOT STARTED**.
