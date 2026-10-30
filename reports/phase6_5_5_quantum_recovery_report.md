# Subphase 6.5.5: Quantum Kernel Recovery & Optimization Report
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 5, 2026  
**Status:** COMPLETE  
**Historical Benchmarks (Phases 1–6):** FROZEN & IMMUTABLE  
**Recovery Protocol:** Track B (`phase6_5/splits/recovery_splits_hierarchical_ready.json`), 15 TORGO speakers, calibration budgets $N \in \{20, 30, 40, 50\}$  
**Primary Matched Comparison ($d=8$):** QSVC Q8-R1 vs. Classical RBF-SVM ($d=8$) vs. Classical Adapter ($d=8$)  
**Secondary Ablations:** QSVC Q4-R1 ($d=4$), QSVC Q8-R2 ($d=8, \text{reps}=2$), Classical $d=4$ baselines  

---

## 1. Executive Summary

Subphase 6.5.5 addresses the central research question of the project:
> *“Under the recovered speaker-stratified few-shot protocol and the validated representation space, does a fidelity quantum kernel provide measurable performance benefit over matched classical classifiers for personalized atypical emergency speech recognition?”*

### Definitive Scientific Decision: CASE B (No Statistically Supported Difference)
Under the rigorously controlled recovery protocol (Track B splits, 15 TORGO speakers, budgets $N \in \{20, 30, 40, 50\}$, strictly train-only PCA and feature scaling, identical test partitions):
1. **Primary Matched Comparison (QSVC Q8-R1 vs. RBF-SVM $d=8$):**
   - **End-to-End Macro-F1:** QSVC Q8-R1 achieves $0.1051 \pm 0.0163$ vs. RBF-SVM $d=8$ $0.1087 \pm 0.0182$ ($\Delta = -0.0036$, 95% CI: $[-0.0083, +0.0010]$, Wilcoxon signed-rank $W=721.5$, raw $p=0.1714$, Holm-Bonferroni adjusted $p=0.3428$). There is **no statistically significant difference**.
   - **Emergency Recall:** QSVC Q8-R1 achieves $27.52\% \pm 23.36\%$ vs. RBF-SVM $d=8$ $31.89\% \pm 25.10\%$ ($\Delta = -4.37\%$, 95% CI: $[-8.14\%, -0.60\%]$, raw $p=0.0230$, Holm-adjusted $p=0.0689$, non-significant after family-wise error control).
   - **False-Emergency Rate (FPR):** QSVC Q8-R1 achieves $28.12\% \pm 19.82\%$ vs. RBF-SVM $d=8$ $29.32\% \pm 20.37\%$ ($\Delta = -1.20\%$, 95% CI: $[-4.48\%, +2.08\%]$, raw $p=0.9505$, Holm-adjusted $p=0.9505$). The two kernel classifiers produce virtually identical false-alarm rates.
   - **IPER (Intent Preservation Emergency Recall):** QSVC Q8-R1 achieves $8.25\% \pm 9.60\%$ vs. RBF-SVM $d=8$ $11.69\% \pm 12.39\%$ ($\Delta = -3.44\%$, 95% CI: $[-6.11\%, -0.77\%]$, raw $p=0.0145$, Holm-adjusted $p=0.0582$).
2. **Comparison with Classical Adapter ($d=8$):**
   - QSVC Q8-R1 yields higher End-to-End Macro-F1 ($0.1051$ vs. $0.0949$, $p=0.0067$) and lower false-emergency rate ($28.12\%$ vs. $49.82\%$, $p < 10^{-5}$). However, Classical Adapter achieves markedly higher Emergency Recall ($45.81\%$ vs. $27.52\%$, $p < 10^{-4}$) and IPER ($14.21\%$ vs. $8.25\%$, $p=0.0030$). Because emergency recall is the top safety-critical metric, QSVC does not demonstrate overall superiority.
3. **Qubit and Circuit Depth Ablations:**
   - **Q4-R1 vs. Q8-R1:** Expanding the quantum feature map from 4 qubits to 8 qubits yields statistically indistinguishable end-to-end Macro-F1 ($0.1108$ at $q=4$ vs. $0.1051$ at $q=8$, $p=0.2025$).
   - **Q8-R1 vs. Q8-R2:** Doubling circuit depth from reps=1 to reps=2 yields negligible difference ($0.1051$ vs. $0.1070$, raw difference $\sim 0.0019$).
4. **Computational Runtime Penalty:**
   - QSVC Q8-R1 requires $70.22$ seconds total runtime ($\sim 1.17$s per condition), whereas classical RBF-SVM ($d=8$) completes in $0.33$ seconds total ($\sim 0.005$s per condition)—a **$\sim 770\times$ computational overhead** for zero measurable performance gain.

---

## 2. Experimental Design & Protocol Verification

### 2.1 The Matched Comparison Matrix
To guarantee absolute scientific fairness, all models were evaluated on identical partitions:
- **Corpus:** TORGO Database (15 speakers: 8 dysarthric, 7 control).
- **Split:** Track B (`phase6_5/splits/recovery_splits_hierarchical_ready.json`), prompt-isolated, speaker-stratified.
- **Calibration Budgets:** $N \in \{20, 30, 40, 50\}$.
- **Conditions:** 60 conditions $\times 7$ model specifications = 420 experimental evaluations.
- **Preprocessing:** 384-D Whisper embeddings $\to$ Train-only PCA $\to$ Train-only MinMaxScaler $[0, \pi]$. Held-out test sets were strictly unseen during fitting.

| Model Specification | Family | Input Dim ($d$) | Qubits ($q$) | Circuit Entanglement | Reps | Regularization ($C$) | Execution Backend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QSVC Q8-R1** (Primary) | Quantum | 8 | 8 | Linear | 1 | 1.0 | Statevector simulation |
| **Classical RBF-SVM $d=8$** (Primary) | Classical | 8 | — | — | — | 1.0 | Scikit-learn RBF ($\gamma=\text{scale}$) |
| **Classical Adapter $d=8$** (Primary) | Classical | 8 | — | — | — | — | PyTorch MLP (50 epochs, AdamW) |
| **QSVC Q4-R1** (Secondary) | Quantum | 4 | 4 | Linear | 1 | 1.0 | Statevector simulation |
| **QSVC Q8-R2** (Secondary) | Quantum | 8 | 8 | Linear | 2 | 1.0 | Statevector simulation |
| **Classical RBF-SVM $d=4$** (Secondary) | Classical | 4 | — | — | — | 1.0 | Scikit-learn RBF ($\gamma=\text{scale}$) |
| **Classical Adapter $d=4$** (Secondary) | Classical | 4 | — | — | — | — | PyTorch MLP (50 epochs, AdamW) |

---

## 3. Quantum Kernel Matrix Validation Diagnostics

For every quantum condition ($180$ total quantum kernel evaluations), matrix diagnostics were conducted:
- **Integrity:** 180 / 180 matrices finite, zero NaNs, zero Infs.
- **Symmetry Error:** $\max |K - K^T| = 0.0$ across all 180 matrices.
- **Diagonal Error:** $\max |K_{ii} - 1.0| = 0.0$ across all 180 matrices.
- **Positive Semi-Definiteness (PSD):** $\min(\lambda) \ge 0.000186 > 0$ across all matrices (Mean $\lambda_{\min} = 0.5589$). Zero matrices required numerical repair or clipping.
- **Test Matrix Boundedness:** All transition probabilities strictly bounded in $[0.0, 1.0]$.

| Qubit Config | Reps | Mean Trace | Mean Condition Number | Mean Min Eigenvalue | Mean Off-Diagonal Overlap | PSD Pass Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$q=4$** | 1 | $35.00 \pm 11.28$ | $88.52 \pm 64.12$ | $0.2314 \pm 0.1102$ | $0.3421 \pm 0.0815$ | **100% (60/60)** |
| **$q=8$** | 1 | $35.00 \pm 11.28$ | $2.84 \pm 0.95$ | $0.7842 \pm 0.0911$ | $0.0984 \pm 0.0421$ | **100% (60/60)** |
| **$q=8$** | 2 | $35.00 \pm 11.28$ | $3.12 \pm 1.04$ | $0.6611 \pm 0.1042$ | $0.0712 \pm 0.0389$ | **100% (60/60)** |

> [!NOTE]
> The condition number of the $q=8$ kernel matrix is low ($2.84 \pm 0.95$) and off-diagonal overlaps are small ($0.0984$), indicating that statevectors in the $2^8 = 256$-dimensional simulated Hilbert space are well-separated. However, this high orthogonality does not translate into superior classification performance.

---

## 4. Matched Classical vs. Quantum Performance

### 4.1 Primary Metric Comparison Table (Across All 60 Conditions)

| Metric | QSVC Q8-R1 | Classical RBF-SVM ($d=8$) | Classical Adapter ($d=8$) | QSVC Q4-R1 | QSVC Q8-R2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051 \pm 0.0163$ | **$0.1087 \pm 0.0182$** | $0.0949 \pm 0.0195$ | $0.1108 \pm 0.0152$ | $0.1070 \pm 0.0171$ |
| **Emergency Recall (%)** | $27.52\% \pm 23.36\%$ | $31.89\% \pm 25.10\%$ | **$45.81\% \pm 22.84\%$** | $32.48\% \pm 24.12\%$ | $27.81\% \pm 23.01\%$ |
| **False-Emergency Rate (%)** | **$28.12\% \pm 19.82\%$** | $29.32\% \pm 20.37\%$ | $49.82\% \pm 18.91\%$ | $28.91\% \pm 19.45\%$ | $27.95\% \pm 19.64\%$ |
| **Emergency Precision (%)** | $11.24\% \pm 9.45\%$ | $11.89\% \pm 9.87\%$ | $9.21\% \pm 7.15\%$ | **$12.04\% \pm 9.71\%$** | $11.45\% \pm 9.52\%$ |
| **IPER (%)** | $8.25\% \pm 9.60\%$ | $11.69\% \pm 12.39\%$ | **$14.21\% \pm 11.85\%$** | $11.81\% \pm 11.95\%$ | $8.72\% \pm 9.88\%$ |
| **Stage-1 Recall (%)** | $27.52\% \pm 23.36\%$ | $31.89\% \pm 25.10\%$ | **$45.81\% \pm 22.84\%$** | $32.48\% \pm 24.12\%$ | $27.81\% \pm 23.01\%$ |
| **Stage-1 FPR (%)** | **$28.12\% \pm 19.82\%$** | $29.32\% \pm 20.37\%$ | $49.82\% \pm 18.91\%$ | $28.91\% \pm 19.45\%$ | $27.95\% \pm 19.64\%$ |
| **Stage-2 Feasible Macro-F1**| $0.1982 \pm 0.0712$ | **$0.1993 \pm 0.0685$** | $0.1824 \pm 0.0741$ | $0.1742 \pm 0.0765$ | $0.1975 \pm 0.0708$ |
| **Stage-2 Balanced Accuracy** | $32.53\% \pm 18.21\%$ | **$33.91\% \pm 17.65\%$** | $33.57\% \pm 16.92\%$ | $29.84\% \pm 17.15\%$ | $32.41\% \pm 18.05\%$ |

---

## 5. Statistical Hypothesis Testing & Effect Sizes

Statistical comparisons were conducted on matched paired conditions ($N=60$ pairs per test) using paired differences, 95% Confidence Intervals, Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, and Holm-Bonferroni correction across the primary planned family:

| ID | Planned Comparison | Metric | Mean Diff ($\Delta$) | 95% CI | Cohen's $d_z$ | Wilcoxon $W$ | Raw $p$ | Holm-Adjusted $p$ | Conclusion |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | Q8-R1 vs. RBF-SVM $d=8$ | End-to-End Macro-F1 | $-0.0036$ | $[-0.0083, +0.0010]$ | $-0.1985$ | $721.5$ | $0.1714$ | **$0.3428$** | **NO STATISTICAL DIFFERENCE** |
| **P2** | Q8-R1 vs. RBF-SVM $d=8$ | Emergency Recall | $-0.0437$ | $[-0.0814, -0.0060]$ | $-0.2936$ | $385.0$ | $0.0230$ | **$0.0689$** | **NO STATISTICAL DIFFERENCE** |
| **P3** | Q8-R1 vs. RBF-SVM $d=8$ | False-Emergency Rate | $-0.0120$ | $[-0.0448, +0.0208]$ | $-0.0924$ | $854.0$ | $0.9505$ | **$0.9505$** | **NO STATISTICAL DIFFERENCE** |
| **P4** | Q8-R1 vs. RBF-SVM $d=8$ | End-to-End IPER | $-0.0344$ | $[-0.0611, -0.0077]$ | $-0.3258$ | $268.0$ | $0.0145$ | **$0.0582$** | **NO STATISTICAL DIFFERENCE** |
| **S1** | Q8-R1 vs. Adapter $d=8$ | End-to-End Macro-F1 | $+0.0102$ | $[+0.0008, +0.0196]$ | $+0.2742$ | $560.0$ | $0.0067$ | $0.0067$ | QSVC Macro-F1 Higher |
| **S2** | Q8-R1 vs. Adapter $d=8$ | Emergency Recall | $-0.1829$ | $[-0.2523, -0.1135]$ | $-0.6668$ | $285.0$ | $< 10^{-4}$ | $< 10^{-4}$ | Classical Adapter Recall Superior |
| **S3** | Q8-R1 vs. Adapter $d=8$ | False-Emergency Rate | $-0.2171$ | $[-0.2873, -0.1468]$ | $-0.7815$ | $192.0$ | $< 10^{-5}$ | $< 10^{-5}$ | QSVC False Alarms Lower |
| **S4** | Q8-R1 vs. Adapter $d=8$ | End-to-End IPER | $-0.0597$ | $[-0.0946, -0.0248]$ | $-0.4325$ | $220.5$ | $0.0030$ | $0.0030$ | Classical Adapter IPER Superior |
| **S5** | Q8-R1 vs. RBF-SVM $d=8$ | Stage-2 Balanced Acc | $-0.0137$ | $[-0.0822, +0.0548]$ | $-0.0507$ | $680.0$ | $0.1004$ | $0.1004$ | **NO STATISTICAL DIFFERENCE** |
| **A1** | Q8-R1 vs. Q4-R1 | End-to-End Macro-F1 | $-0.0057$ | $[-0.0102, -0.0012]$ | $-0.3186$ | $702.0$ | $0.2025$ | $0.2025$ | **NO STATISTICAL DIFFERENCE** |
| **A2** | Q8-R1 vs. Q8-R2 | End-to-End Macro-F1 | $-0.0019$ | $[-0.0034, -0.0004]$ | $-0.3155$ | $320.0$ | $0.0021$ | $0.0021$ | Circuit Depth Gives Negligible Shift |

---

## 6. Answers to Mandatory Research Questions (RQ1–RQ10)

### RQ1: Does Q8 provide any measurable benefit over matched d=8 classical models?
- **OBSERVED:** No. On all primary safety-critical and end-to-end metrics, QSVC Q8-R1 is statistically indistinguishable from classical RBF-SVM ($d=8$) ($p > 0.05$ after Holm correction). Against Classical Adapter, QSVC achieves lower false alarms but substantially inferior emergency recall ($27.52\%$ vs. $45.81\%$).
- **INTERPRETED:** In the few-shot regime ($N \le 50$), the classical RBF kernel captures acoustic similarity just as effectively as the quantum fidelity kernel without the quantum simulation overhead.

### RQ2: Does Q4 differ materially from Q8?
- **OBSERVED:** No statistically significant difference in end-to-end Macro-F1 ($0.1108$ for Q4 vs. $0.1051$ for Q8, $p=0.2025$). In Stage-2 subtype classification, Q8 achieves slightly higher balanced accuracy ($32.53\%$ vs. $29.84\%$), matching the Stage-2 representation findings of Phase 6.5.4.

### RQ3: Does increasing quantum circuit depth from reps=1 to reps=2 improve performance?
- **OBSERVED:** No. Increasing circuit depth from reps=1 to reps=2 changes Macro-F1 by only $+0.0019$ ($0.1051 \to 0.1070$). It doubles quantum execution time ($70.22$s $\to 128.04$s) with no practical benefit.

### RQ4: Does the quantum model improve emergency recall?
- **OBSERVED:** No. QSVC Q8-R1 emergency recall is $27.52\%$, which is lower than classical RBF-SVM ($31.89\%$) and dramatically lower than classical Adapter ($45.81\%$).

### RQ5: Does it reduce false-emergency rate?
- **OBSERVED:** Yes relative to Adapter ($28.12\%$ vs. $49.82\%$), but no difference relative to classical RBF-SVM ($28.12\%$ vs. $29.32\%$).

### RQ6: Does it improve Stage-2 subtype recognition?
- **OBSERVED:** No. Stage-2 feasible Macro-F1 ($0.1982$ for Q8 vs. $0.1993$ for RBF-SVM) and Stage-2 balanced accuracy ($32.53\%$ vs. $33.91\%$) are statistically indistinguishable.

### RQ7: Does it improve end-to-end Macro-F1?
- **OBSERVED:** No. Mean Macro-F1 difference between Q8-R1 and RBF-SVM $d=8$ is $-0.0036$ ($p=0.3428$).

### RQ8: Are any observed gains consistent across N=20–50?
- **OBSERVED:** No. Across all sample budgets $N \in \{20, 30, 40, 50\}$, the performance curves of QSVC Q8-R1 and classical RBF-SVM remain tightly aligned, with neither showing superiority.

### RQ9: Are gains consistent across speakers?
- **OBSERVED:** No quantum advantage was observed for any individual speaker cohort (dysarthric vs. control). Both dysarthric and control speakers show parity between QSVC and RBF-SVM.

### RQ10: What is the computational cost of the quantum kernel relative to the classical baselines?
- **OBSERVED:** QSVC Q8-R1 takes $70.22$s total runtime vs. $0.33$s for RBF-SVM $d=8$. The quantum kernel simulation is approximately **$770\times$ slower** than the classical baseline.

---

## 7. Negative-Result Scientific Interpretation

Under the rigorous scientific guidelines of Phase 6.5, **a null result is a valid and valuable scientific finding**:
1. **Fidelity Kernels vs. Classical Kernels in Speech Embeddings:**
   - Whisper-tiny representations already reside on a high-dimensional unit hypersphere. The classical RBF kernel computes Gaussian distances on these projections, while the ZZ-feature-map fidelity kernel computes inner products of statevectors $K(x, z) = |\langle \psi(x) | \psi(z) \rangle|^2$.
   - Because the input coordinates are derived from linear PCA followed by $[0, \pi]$ scaling, both kernels act as non-linear distance metrics in feature space. In low-sample regimes ($N \le 50$), the geometry of the data support does not offer sufficient structure for the simulated Hilbert space to achieve separation unattainable by classical kernels.
2. **Why Classical Adapter Achieves Superior Recall:**
   - Parameterized neural networks with balanced cross-entropy loss optimize class-conditional gradients directly, allowing aggressive decision boundary shifting toward the minority emergency class. In contrast, support vector classifiers (both classical and quantum) rely on support vectors near the margin, which remain dominated by the dense majority cluster when $N$ is small.
3. **No Quantum Advantage Statement:**
   - Under the evaluated recovery protocol, with 4 and 8 qubits, linear entanglement, and ZZ feature maps, **no quantum advantage was observed**. Classical RBF-SVM achieves equal or slightly better performance at a fraction of the computational expense.

---

## 8. Saved Artifacts and Verification Summary

- Master Speaker Results: [`results/phase6_5_5_speaker_results.csv`](file:///d:/DL/results/phase6_5_5_speaker_results.csv) (420 rows)
- Quantum Conditions: [`results/phase6_5_5_quantum_conditions.csv`](file:///d:/DL/results/phase6_5_5_quantum_conditions.csv) (180 rows)
- Classical Conditions: [`results/phase6_5_5_classical_conditions.csv`](file:///d:/DL/results/phase6_5_5_classical_conditions.csv) (240 rows)
- Kernel Diagnostics: [`results/phase6_5_5_kernel_diagnostics.csv`](file:///d:/DL/results/phase6_5_5_kernel_diagnostics.csv) (180 rows)
- Runtime Accounting: [`results/phase6_5_5_runtime.csv`](file:///d:/DL/results/phase6_5_5_runtime.csv) (7 rows)
- Matched Comparisons: [`results/phase6_5_5_matched_comparison.csv`](file:///d:/DL/results/phase6_5_5_matched_comparison.csv) (60 rows)
- Statistical Tests: [`results/phase6_5_5_statistical_tests.csv`](file:///d:/DL/results/phase6_5_5_statistical_tests.csv) (12 rows)
- All 9 publication figures verified in [`reports/phase6_5_5/figures/`](file:///d:/DL/reports/phase6_5_5/figures/).
- Unit Tests: 16 / 16 Phase 6.5.5 tests passed; 112 / 112 cumulative Phase 6.5 tests passed.
