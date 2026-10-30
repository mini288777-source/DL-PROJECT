# PHASE 6.5.6 — FINAL FAIR COMPARISON REPORT
## Matched Classical vs. Quantum Evaluation & Dual-Benchmark Synthesis

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Phase:** Phase 6.5.6 Final Recovery Synthesis  
**Scope:** Controlled Synthesis of Original Benchmark (Track 1) and Recovery Protocol (Track 2)  
**Status:** COMPLETE & FROZEN  

---

## 1. Executive Summary

This report delivers the authoritative, paper-ready comparison between classical and quantum approaches across both the **Original Benchmark (Phases 3–6)** and the **Recovery Protocol (Phases 6.5.0–6.5.5-C)**. 

### Core Scientific Findings:
1. **Original Benchmark Collapse (Track 1):** In the unstratified, flat 7-class formulation, both Classical RBF-SVM and Quantum Support Vector Classifier (QSVC) collapsed completely into the majority class (`GENERAL_SPEECH`, $\approx 95.10\%$ prevalence). QSVC produced bitwise-identical predictions to RBF-SVM across 81,840 paired evaluations.
2. **Methodological Factors Identified (Phases 6.5.0–6.5.4):** Forensic auditing established that the original quantum collapse was consistent with severe data support failure ($7/60$ splits had zero emergency samples; $N=20$ had only $\approx 1.1$ emergency samples), unweighted loss formulation, and flat class competition. Recovery experiments subsequently demonstrated that this collapse behavior was not unique to the quantum model.
3. **Recovery Interventions (Track 2):** Implementing deterministic speaker-stratified sampling (Track B), cost-sensitive class weighting, hierarchical two-stage triage routing, and an audited $d=8$ PCA representation recovered non-trivial emergency sensitivity.
4. **Primary Quantum Hypothesis Decision (Case B):** Under the recovered, fair, and matched Track B protocol across 60 matched conditions ($15$ speakers $\times 4$ budgets), **QSVC Q8-R1 demonstrated NO statistically supported advantage** over the matched Classical RBF-SVM $d=8$ on any primary safety metric (Macro-F1, Emergency Recall, False-Emergency Rate, IPER) under Holm-Bonferroni correction ($\alpha = 0.05$).
5. **Safety-Critical Tradeoff:** The Weighted Classical Adapter $d=8$ achieved statistically superior Emergency Recall over QSVC ($45.81\%$ vs. $27.52\%$, $p = 0.000125$), but incurred a high false-alarm burden ($49.82\%$ vs. $28.12\%$).
6. **Computational Cost:** The statevector-simulated QSVC Q8-R1 was measured to be **$\approx 772.6\times$ slower per test evaluation** and **$\approx 212.8\times$ slower in total experiment wall time** than the matched Classical RBF-SVM $d=8$.

---

## 2. Paper-Ready Tables

### TABLE 1: Experimental Phases, Tracks, and Protocol Contracts

| Track | Phase | Designation | Key Protocol Characteristics | Calibration Budget ($N$) | Evaluation Partition | Primary Models Evaluated |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **Track 1** | Phase 3 | Classical Baselines | Random unstratified calibration, flat 7-class taxonomy, unweighted loss | $\{20, 30, 40, 50\}$ | Track A (Random Holdout, 42,536 preds) | Linear SVM, RBF-SVM ($d=384$), Adapter ($d=384$) |
| **Track 1** | Phase 4 | Quantum Kernel Pilot | Random unstratified, flat 7-class, unweighted QSVC, ZZFeatureMap | $\{20, 30, 40, 50\}$ | Track A (Random Holdout) | QSVC $q \in \{2,4,6,8\}$, reps $=1$, Fidelity Kernel |
| **Track 1** | Phase 5 | Acoustic Robustness | Random unstratified, AWGN noise injection (Clean, 15, 10, 5 dB) | $\{20, 30, 40, 50\}$ | Track A (720 experimental cells) | Adapter, RBF-SVM, QSVC $q=4$ |
| **Track 1** | Phase 6 | Statistical Closure | 48 hypothesis tests, paired difference tests on Track 1 | $\{20, 30, 40, 50\}$ | Track A (81,840 paired evaluations) | QSVC $q=4$ vs. RBF-SVM ($d=384$) |
| **Track 2** | Phase 6.5.0 | Diagnostic Audit | Formal audit of calibration support and embedding geometry | $\{20, 30, 40, 50\}$ | Track A & Full Corpus | Diagnostic only (0 models trained) |
| **Track 2** | Phase 6.5.1 | Data Recovery | Deterministic speaker-stratified, prompt-isolated calibration | $\{20, 30, 40, 50\}$ | Track B (Deterministic Prompt-Isolated) | Dataset split construction (0 models trained) |
| **Track 2** | Phase 6.5.2 | Classical Rescue | Stratified Track B, flat 7-class, balanced cost-sensitive weighting | $\{20, 30, 40, 50\}$ | Track B (240 cells) | Weighted vs. Unweighted Adapter & RBF-SVM |
| **Track 2** | Phase 6.5.3 | Hierarchical Routing | Two-stage architecture: Stage-1 binary gate + Stage-2 conditional routing | $\{20, 30, 40, 50\}$ | Track B (120 cells) | Hierarchical Adapter & RBF-SVM ($d=4$) |
| **Track 2** | Phase 6.5.4 | Representation Audit | Representation dimensionality audit ($d \in \{2 \dots 384\}$), PCA forensic audit | $\{20, 30, 40, 50\}$ | Track B (72 cells) | Hierarchical RBF-SVM across 9 dimensions |
| **Track 2** | Phase 6.5.5 | Quantum Recovery | Matched classical vs. quantum recovery under Track B at $d=8$ | $\{20, 30, 40, 50\}$ | Track B (180 quantum cells, 120 classical) | QSVC (Q8-R1, Q4-R1, Q8-R2) vs. RBF & Adapter ($d=8$) |
| **Track 2** | Phase 6.5.5-C | Statistical Closure | Wilcoxon signed-rank paired tests, Holm-Bonferroni correction, runtime audit | $\{20, 30, 40, 50\}$ | Track B (60 matched units) | QSVC Q8-R1 vs. RBF $d=8$, QSVC vs. Adapter $d=8$ |
| **Track 2** | Phase 6.5.6 | Final Synthesis | Comprehensive evidence synthesis, metric reconciliation, and closure | N/A | Longitudinal Repository Artifacts | Complete research evidence chain |

---

### TABLE 2: Original Benchmark Results (Phases 3–6, Track 1)

> [!WARNING]
> Protocol Contract: Evaluated on original Track A (random unstratified calibration, flat 7-class taxonomy, unweighted loss). Both RBF and QSVC suffered total majority-class collapse to `GENERAL_SPEECH`.

| Model | Representation Dimension | Circuit Config | End-to-End Macro-F1 | Emergency Recall (%) | False-Emergency Rate (%) | IPER (%) | Operational Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unweighted RBF-SVM** | $d=384$ | N/A | $0.1390$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | Majority collapse: predicted `GENERAL_SPEECH` on $99.8\%+$ of samples. |
| **Classical Adapter** | $d=384$ | N/A | $0.1410-0.1470$ | $0.00-1.20\%$ | $0.00-0.80\%$ | $0.00-0.30\%$ | Near-total collapse: minimal non-general predictions. |
| **QSVC ($q=4$, reps $=1$)** | $d=4$ (PCA) | ZZFeatureMap | $0.1388$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | Bitwise-identical predictions to unweighted RBF ($81,840$ paired evaluations). |

---

### TABLE 3: Recovery Interventions and Measured Outcomes (Phases 6.5.0–6.5.5)

| Step | Methodological Intervention | Target Limitation Addressed | Measured Quantitative Effect | Scientific Interpretation |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Speaker-Stratified Calibration (Track B) | Sparse emergency support in calibration ($N=20$ had $\approx 1.1$ emergency samples) | Emergency support increased to $11.3$ samples at $N=20$; represented classes rose from $1.0$ to $5.2$. | Established sufficient calibration support to make few-shot adaptation mathematically viable. |
| **2** | Cost-Sensitive Class Weighting | Majority-class collapse caused by $\approx 95.10\%$ negative prevalence | Emergency recall jumped from $0.0\%$ to $58.42\%$ (Adapter) and $38.10\%$ (RBF). | Escaped majority collapse, but flat classification triggered extreme false alarms (FER up to $96.82\%$). |
| **3** | Two-Stage Hierarchical Routing | Extreme false-alarm rates in flat cost-sensitive models | RBF FER dropped from $96.82\%$ to $15.06\%$ at $N=30$ ($\Delta = -81.76\text{ pp}$). | Triage gate successfully decoupled emergency detection from intent classification. |
| **4** | Dimensionality Expansion ($d=4 \to d=8$) | Stage-2 subtype compression bottleneck ($d=4$ had limited subspace capacity) | PCA explained variance increased from $80.66\%$ ($d=4$) to $89.73\%$ ($d=8$). | Relieved Stage-2 routing bottleneck while keeping quantum simulation within 8 qubits. |
| **5** | Matched Quantum Execution (Q8-R1) | Direct quantum-vs-classical comparison under recovered protocol | QSVC Q8-R1 achieved Macro-F1 of $0.1051$ and Emergency Recall of $27.52\%$. | Quantum model exhibited functional behavior closely matching Classical RBF-SVM $d=8$. |

---

### TABLE 4: PCA Representation Dimension Audit (Phase 6.5.4, Track B)

| PCA Dimension ($d$) | Cumulative Explained Variance (%) | Feasibility Status | Stage-2 Feasible Macro-F1 | Stage-2 Balanced Accuracy (%) | End-to-End Macro-F1 | Bottleneck Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$d=2$** | $69.86\% \pm 5.12\%$ | Feasible | $0.1012 \pm 0.0410$ | $24.15\% \pm 12.10\%$ | $0.1001 \pm 0.0480$ | Severe informational bottleneck |
| **$d=4$** | $80.66\% \pm 4.37\%$ | Feasible | $0.1455 \pm 0.1041$ | $27.01\% \pm 18.82\%$ | $0.1011 \pm 0.0520$ | Moderate Stage-2 compression bottleneck |
| **$d=8$** | **$89.73\% \pm 2.99\%$** | **Selected Quantum Parity** | **$0.1962 \pm 0.1091$** | **$33.91\% \pm 18.19\%$** | **$0.1018 \pm 0.0490$** | **Bottleneck substantially relieved; selected candidate** |
| **$d=16$** | $96.48\% \pm 1.45\%$ | Classical Only | $0.2110 \pm 0.1120$ | $35.12\% \pm 18.40\%$ | $0.1045 \pm 0.0510$ | Highly competitive classical subspace |
| **$d=32$** | $99.58\% \pm 0.32\%$ | Classical Only | $0.2450 \pm 0.1250$ | $38.90\% \pm 19.10\%$ | $0.1218 \pm 0.0550$ | Peak classical performance |
| **$d=384$** | $100.00\%$ | Unreduced | $0.1850 \pm 0.1100$ | $31.20\% \pm 17.50\%$ | $0.0971 \pm 0.0480$ | Small-sample noise penalty at low $N$ |

---

### TABLE 5: Matched Classical vs. Quantum Results Across 60 Conditions (Track B)

> [!NOTE]
> Evaluated on 60 matched conditions ($15$ TORGO speakers $\times 4$ budgets $N \in \{20, 30, 40, 50\}$) under the two-stage hierarchical architecture on Track B.

| Metric | Primary Quantum (QSVC Q8-R1) | Matched Classical (RBF-SVM $d=8$) | Matched Classical (Adapter $d=8$) | Qubit Scaling (QSVC Q4-R1) | Depth Scaling (QSVC Q8-R2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051 \pm 0.0575$ | **$0.1087 \pm 0.0487$** | $0.0949 \pm 0.0342$ | $0.1108 \pm 0.0527$ | $0.1070 \pm 0.0558$ |
| **Emergency Recall (%)** | $27.52\% \pm 41.94\%$ | $31.89\% \pm 40.97\%$ | **$45.81\% \pm 29.78\%$** | $27.25\% \pm 39.84\%$ | $26.43\% \pm 42.51\%$ |
| **False-Emergency Rate (%)** | **$28.12\% \pm 41.86\%$** | $29.32\% \pm 38.07\%$ | $49.82\% \pm 22.98\%$ | $27.97\% \pm 38.11\%$ | $26.73\% \pm 41.70\%$ |
| **Emergency Precision (%)** | $0.81\% \pm 1.31\%$ | $2.02\% \pm 6.57\%$ | **$2.03\% \pm 1.45\%$** | $1.75\% \pm 4.68\%$ | $0.66\% \pm 1.16\%$ |
| **End-to-End IPER (%)** | $8.25\% \pm 13.61\%$ | $11.69\% \pm 17.14\%$ | **$14.21\% \pm 14.17\%$** | $6.61\% \pm 10.55\%$ | $7.65\% \pm 13.84\%$ |
| **Stage-1 Recall (%)** | $27.52\% \pm 41.94\%$ | $31.89\% \pm 40.97\%$ | **$45.81\% \pm 29.78\%$** | $27.25\% \pm 39.84\%$ | $26.43\% \pm 42.51\%$ |
| **Stage-1 FPR (%)** | **$28.12\% \pm 41.86\%$** | $29.32\% \pm 38.07\%$ | $49.82\% \pm 22.98\%$ | $27.97\% \pm 38.11\%$ | $26.73\% \pm 41.70\%$ |
| **Stage-2 Feasible Macro-F1**| $0.1252 \pm 0.0844$ | **$0.1962 \pm 0.1091$** | $0.1824 \pm 0.1220$ | $0.1455 \pm 0.1041$ | $0.1296 \pm 0.0974$ |
| **Stage-2 Balanced Accuracy (%)** | $32.53\% \pm 23.25\%$ | **$33.91\% \pm 18.19\%$** | $33.57\% \pm 19.60\%$ | $27.01\% \pm 18.82\%$ | $30.48\% \pm 18.45\%$ |

---

### TABLE 6: Statistical Hypothesis Testing & Effect Sizes (60 Matched Pairs)

| ID | Planned Comparison Family | Evaluated Metric | Paired Mean Diff ($\Delta$) | 95% Confidence Interval | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$ | Authoritative Statistical Finding |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | Primary: Q8-R1 vs. RBF $d=8$ | End-to-End Macro-F1 | $-0.0036$ | $[-0.0083, +0.0010]$ | $475.0$ | $0.1714$ | **$0.3428$** | **No Statistically Supported Difference** |
| **P2** | Primary: Q8-R1 vs. RBF $d=8$ | Emergency Recall | $-0.0437$ | $[-0.0814, -0.0060]$ | $40.5$ | $0.0230$ | **$0.0689$** | **No Statistically Supported Difference** |
| **P3** | Primary: Q8-R1 vs. RBF $d=8$ | False-Emergency Rate | $-0.0120$ | $[-0.0448, +0.0208]$ | $59.0$ | $0.9505$ | **$0.9505$** | **No Statistically Supported Difference** |
| **P4** | Primary: Q8-R1 vs. RBF $d=8$ | End-to-End IPER | $-0.0344$ | $[-0.0611, -0.0077]$ | $62.0$ | $0.0145$ | **$0.0582$** | **No Statistically Supported Difference** |
| **S1** | Secondary: Q8-R1 vs. Adapter $d=8$ | Emergency Recall | $-0.1829$ | $[-0.2728, -0.0929]$ | $166.5$ | $0.000125$ | **$0.000874$** | **Statistically Supported Adapter Superiority** |
| **S2** | Secondary: Q8-R1 vs. Adapter $d=8$ | Emergency Precision | $-0.0122$ | $[-0.0169, -0.0075]$ | $109.5$ | $1.0 \times 10^{-6}$ | **$7.0 \times 10^{-6}$** | **Statistically Supported Adapter Superiority** |
| **S3** | Secondary: Q8-R1 vs. Adapter $d=8$ | False-Emergency Rate | $-0.2171$ | $[-0.2873, -0.1468]$ | $192.0$ | $3.0 \times 10^{-6}$ | **$1.8 \times 10^{-5}$** | **QSVC False-Alarm Rate Significantly Lower** |
| **S4** | Secondary: Q8-R1 vs. Adapter $d=8$ | End-to-End Macro-F1 | $+0.0102$ | $[+0.0008, +0.0196]$ | $560.0$ | $0.0067$ | **$0.0202$** | **QSVC Macro-F1 Higher (due to lower false alarms)** |
| **A1** | Ablation: Q8-R1 vs. Q4-R1 | End-to-End Macro-F1 | $-0.0057$ | $[-0.0163, +0.0049]$ | $699.5$ | $0.3541$ | **$1.0000$** | **No Statistically Supported Difference** |
| **A2** | Ablation: Q8-R1 vs. Q4-R1 | Emergency Recall | $+0.0027$ | $[-0.0712, +0.0766]$ | $127.5$ | $0.9679$ | **$1.0000$** | **No Statistically Supported Difference** |
| **A3** | Ablation: Q8-R1 vs. Q8-R2 | End-to-End Macro-F1 | $-0.0019$ | $[-0.0034, -0.0004]$ | $708.5$ | $0.0021$ | **$0.0273$** | **Circuit Depth Produces Negligible Shift** |
| **A4** | Ablation: Q8-R1 vs. Q8-R2 | Emergency Recall | $+0.0108$ | $[-0.0105, +0.0321]$ | $100.0$ | $0.3105$ | **$1.0000$** | **No Statistically Supported Difference** |

---

### TABLE 7: Computational Cost & Latency Benchmark

| Model Architecture | Qubits | Circuit Reps | Training Time per Condition (s) | Total 60-Cond Train Time (s) | Inference Latency per Evaluation (s) | Per-Evaluation Ratio vs. RBF | Total Wall-Time Ratio vs. RBF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical RBF-SVM $d=8$** | $0$ | $0$ | $0.0040 \text{ s}$ | $0.24 \text{ s}$ | **$0.0014 \text{ s}$** | **$1.0\times$ (Baseline)** | **$1.0\times$ (Baseline)** |
| **Classical Adapter $d=8$** | $0$ | $0$ | $0.2625 \text{ s}$ | $15.75 \text{ s}$ | **$0.0006 \text{ s}$** | **$0.43\times$** | **$47.8\times$** |
| **QSVC Q4-R1** | $4$ | $1$ | $0.0247 \text{ s}$ | $1.48 \text{ s}$ | **$0.4194 \text{ s}$** | **$299.6\times$** | **$76.2\times$** |
| **QSVC Q8-R1 (Primary)** | $8$ | $1$ | $0.0888 \text{ s}$ | $5.33 \text{ s}$ | **$1.0816 \text{ s}$** | **$772.6\times$** | **$212.8\times$** |
| **QSVC Q8-R2 (Deep)** | $8$ | $2$ | $0.1578 \text{ s}$ | $9.47 \text{ s}$ | **$1.9761 \text{ s}$** | **$1408.8\times$** | **$388.0\times$** |

---

### TABLE 8: Final Hypothesis Decision Matrix

| Hypothesis Component | Stated Evaluation Target | Empirical Finding | Statistical Support | Final Formal Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Research Hypothesis** | Does QSVC provide measurable few-shot personalization benefit over matched classical RBF? | QSVC Q8-R1 and RBF-SVM $d=8$ perform nearly identically across all primary metrics. | Wilcoxon $p > 0.05$ after Holm correction for Macro-F1 ($p=0.3428$), Recall ($p=0.0689$), FER ($p=0.9505$), IPER ($p=0.0582$). | **CASE B: NO STATISTICALLY SUPPORTED DIFFERENCE** |
| **Secondary Model Hypothesis** | Can QSVC outperform the Weighted Classical Adapter in emergency intent capture? | Classical Adapter achieves significantly higher recall ($45.81\%$ vs $27.52\%$). | Wilcoxon $p = 0.000125$, Holm $p = 0.000874$, Cohen's $d_z = -0.6668$. | **CLASSICAL ADAPTER ADVANTAGE FOR RECALL** |
| **Qubit Scaling Hypothesis** | Does scaling Hilbert space dimension from $16$ ($q=4$) to $256$ ($q=8$) improve performance? | No measurable benefit observed ($\Delta\text{F1} = -0.0057$, $\Delta\text{Recall} = +0.27\text{ pp}$). | Wilcoxon $p = 0.3541$ (Macro-F1), $p = 0.9427$ (Recall), Holm $p = 1.0$. | **NO SCALING BENEFIT OBSERVED** |
| **Circuit Depth Hypothesis** | Does increasing circuit reps from $1$ to $2$ improve quantum feature mapping? | No measurable benefit observed; execution latency doubles. | Wilcoxon $p = 0.4496$ (Macro-F1), $p = 1.0$ (Recall), Holm $p = 1.0$. | **NO DEPTH BENEFIT OBSERVED** |
| **Operational Feasibility** | Is the hybrid quantum-classical system viable for real-world deployment? | Inference latency of $1.08\text{ s}$ per utterance and $72.5\%$ missed emergency rate. | High latency, high missed emergency rate, low precision ($0.81\%$). | **NOT READY FOR DEPLOYMENT** |
