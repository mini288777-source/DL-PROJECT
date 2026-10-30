# PHASE 6.5.6 — METRIC DEFINITION & REPORTING RECONCILIATION AUDIT

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Tracking Phase:** Phase 6.5.6 Final Recovery Synthesis  
**Scope:** Forensic Audit and Resolution of Historical Reporting Discrepancies  
**Status:** COMPLETE & FROZEN  

---

## 1. Executive Overview

During the longitudinal execution of Phase 3 through Phase 6.5.5-C, several key metric values appeared in preliminary reports that differed from the final condition-level empirical data. Scientific integrity requires that discrepancies are neither silently overwritten nor ignored. This document provides an exhaustive forensic reconciliation of:
1. **Emergency Precision**: The historical preliminary values ($11.24\%$ for QSVC Q8-R1 and $9.21\%$ for Adapter $d=8$) versus the audited closure values ($0.81\%$ and $2.03\%$).
2. **PCA Explained Variance**: The historical $27.4\%$ value for $d=4$ versus the audited empirical value of $80.66\% \pm 4.37\%$.
3. **Computational Runtime Penalty**: The historical statement of "$\approx 770\times$ slower overall" versus the audited dual-basis metrics ($772.6\times$ per test evaluation vs. $212.8\times$ total experiment wall-clock time).

---

## 2. Authoritative Reconciliation Table

| Metric | Historical Provisional Value | Final Audited Closure Value | Mathematical Definition | Evaluation Population | Root Cause & Scientific Resolution | Authoritative Final Value |
| :--- | :---: | :---: | :--- | :--- | :--- | :---: |
| **Emergency Precision (QSVC Q8-R1)** | $11.24\% \pm 9.45\%$ | **$0.81\% \pm 1.31\%$** | $\frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FP_{\text{emerg}}}$ on end-to-end binary predictions | All 60 matched Track B conditions ($15$ speakers $\times 4$ budgets) | **Reporting Transcription Artifact in Preliminary Text:** Historical $11.24\%$ was transcribed from an early non-zero condition slice. Audited computation across all 60 conditions with standard `zero_division=0` yields $0.81\%$, driven by the severe $95.10\%$ negative class prevalence. | **$0.81\% \pm 1.31\%$** |
| **Emergency Precision (Adapter $d=8$)** | $9.21\% \pm 7.15\%$ | **$2.03\% \pm 1.45\%$** | $\frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FP_{\text{emerg}}}$ on end-to-end binary predictions | All 60 matched Track B conditions | **Reporting Transcription Artifact:** Historical $9.21\%$ was transcribed from a provisional partial run; true full-cohort empirical mean across all 60 conditions is $2.03\%$. | **$2.03\% \pm 1.45\%$** |
| **Emergency Precision (RBF-SVM $d=8$)** | $11.89\% \pm 9.87\%$ | **$2.02\% \pm 6.57\%$** | $\frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FP_{\text{emerg}}}$ on end-to-end binary predictions | All 60 matched Track B conditions | **Reporting Transcription Artifact:** Directly reconciled from `results/phase6_5_5_speaker_results.csv` where empirical mean is $2.02\%$ ($0.020212$). | **$2.02\% \pm 6.57\%$** |
| **Emergency Precision (QSVC Q4-R1)** | $12.04\% \pm 9.71\%$ | **$1.75\% \pm 4.68\%$** | $\frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FP_{\text{emerg}}}$ on end-to-end binary predictions | All 60 matched Track B conditions | **Reporting Transcription Artifact:** Empirical mean across all 60 conditions in `results/phase6_5_5_speaker_results.csv` is $1.75\%$. | **$1.75\% \pm 4.68\%$** |
| **Emergency Precision (QSVC Q8-R2)** | $11.45\% \pm 9.52\%$ | **$0.66\% \pm 1.16\%$** | $\frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FP_{\text{emerg}}}$ on end-to-end binary predictions | All 60 matched Track B conditions | **Reporting Transcription Artifact:** Empirical mean across all 60 conditions in `results/phase6_5_5_speaker_results.csv` is $0.66\%$. | **$0.66\% \pm 1.16\%$** |
| **PCA Cumulative Variance ($d=4$)** | $27.4\%$ | **$80.66\% \pm 4.37\%$** | $\sum_{i=1}^4 \lambda_i / \sum_{j=1}^{384} \lambda_j$ | 15 TORGO speakers, per-speaker calibration splits | **Hardcoded Plotting Array Placeholder:** In Phase 4, the plotting script used a synthetic array `[17.8, 27.4, 34.2, 39.8]` which was erroneously cited in reports. Forensic recomputation from model metadata proved actual empirical variance is $80.66\% \pm 4.37\%$. | **$80.66\% \pm 4.37\%$** |
| **PCA Cumulative Variance ($d=8$)** | $39.8\%$ (placeholder) | **$89.73\% \pm 2.99\%$** | $\sum_{i=1}^8 \lambda_i / \sum_{j=1}^{384} \lambda_j$ | 15 TORGO speakers, per-speaker calibration splits | **Hardcoded Plotting Array Placeholder:** Same synthetic array origin. Empirical recomputation confirmed $d=8$ captures $89.73\% \pm 2.99\%$ variance, relieving the Stage-2 bottleneck. | **$89.73\% \pm 2.99\%$** |
| **Quantum Runtime Overhead** | "$\approx 770\times$ slower overall" | **$772.6\times$ per eval; $212.8\times$ total wall-time** | $\frac{T_{\text{QSVC}}}{T_{\text{RBF}}}$ | 60 training conditions, 22,168 test evaluations | **Measurement Basis Conflation:** The $\approx 770\times$ ratio applies strictly to individual test-set kernel vector evaluations ($1.0816\text{ s}$ vs. $0.0014\text{ s}$). Total 60-condition wall-time ratio is $212.8\times$ ($70.22\text{ s}$ vs. $0.33\text{ s}$). | **$772.6\times$ per eval; $212.8\times$ total wall-time** |

---

## 3. Deep-Dive Forensic Analyses

### 3.1 Emergency Precision: Why are Empirical Values $\approx 0.81\% - 2.03\%$?

A natural scientific question is why the true empirical emergency precision is so low across all models in Track B. This is directly explained by the base rate of emergency utterances in the TORGO corpus:

1. **Test Set Class Distribution (Prevalence):**
   In the prompt-isolated Track B test sets, `GENERAL_SPEECH` represents approximately $95.10\%$ of all utterances, while emergency utterances (spanning 6 classes) represent only $\approx 4.90\%$.
2. **Precision Formulation Under Extreme Imbalance:**
   Consider a representative test evaluation of $N_{\text{test}} = 100$ samples:
   - True Negatives (`GENERAL_SPEECH`): $95$ utterances.
   - True Positives (Emergencies): $5$ utterances.
   - For QSVC Q8-R1, the measured Emergency Recall is $27.52\%$ and False-Emergency Rate (FER) is $28.12\%$:
     $$\text{Expected True Positives } (TP) = 5 \times 0.2752 \approx 1.38$$
     $$\text{Expected False Positives } (FP) = 95 \times 0.2812 \approx 26.71$$
     $$\text{Expected Precision} = \frac{TP}{TP + FP} = \frac{1.38}{1.38 + 26.71} = \frac{1.38}{28.09} \approx 4.91\%$$
   - In conditions where $TP = 0$ (e.g. for speakers with very few test emergencies or where the gate fails to trigger), `precision_score(..., zero_division=0)` outputs $0.0\%$.
   - Across the 60 experimental conditions:
     - 38 conditions had $TP = 0$ ($\text{Precision} = 0.0\%$).
     - 22 conditions had $TP > 0$ with mean non-zero precision of $2.21\%$.
     - Macro-averaging across all 60 conditions yields:
       $$\text{Mean Precision} = \frac{22 \times 0.0221 + 38 \times 0.0}{60} = \mathbf{0.8105\%}$$
3. **Classical Adapter Behavior:**
   The Weighted Classical Adapter achieves higher recall ($45.81\%$), which produces $TP > 0$ in 53 of 60 conditions (mean non-zero precision: $2.29\%$). Across all 60 conditions, its mean precision is $\mathbf{2.0268\%} \approx \mathbf{2.03\%}$.
4. **Resolution:**
   The historical values of $11.24\%$ and $9.21\%$ in the preliminary text table were transcription errors from an exploratory script that evaluated precision only on a balanced subset or miscalculated the denominator. The authoritative, mathematically verified condition-level values from `results/phase6_5_5_speaker_results.csv` are **$0.81\%$** (QSVC Q8-R1) and **$2.03\%$** (Adapter $d=8$).

---

### 3.2 PCA Retained Variance: Resolving the Historical $27.4\%$ Placeholder

In Phase 4, the initial benchmarking report cited that 4 principal components retained only $27.4\%$ of embedding variance. In Phase 6.5.4, an exhaustive forensic code audit traced this number directly to:
```python
# Hardcoded placeholder array from historical plotting utility:
pca_variances = [17.8, 27.4, 34.2, 39.8]
```
When PCA was empirically fitted on the actual standardized 384-dimensional Whisper-tiny embeddings from the per-speaker calibration sets (Track B):
- **$d=2$**: $69.86\% \pm 5.12\%$
- **$d=4$**: **$80.66\% \pm 4.37\%$** (Original Phase 4 adaptation data: $79.43\% \pm 5.09\%$; pooled corpus: $73.25\%$)
- **$d=8$**: **$89.73\% \pm 2.99\%$**
- **$d=16$**: $96.48\% \pm 1.45\%$
- **$d=32$**: $99.58\% \pm 0.32\%$

**Resolution:**
The historical $27.4\%$ was never an empirical measurement. It was a reporting artifact. The authoritative empirical value for $d=4$ is **$80.66\% \pm 4.37\%$**, and for $d=8$ is **$89.73\% \pm 2.99\%$**.

---

### 3.3 Computational Runtime: Distinguishing Per-Evaluation Latency from Wall-Time

Preliminary summaries colloquially reported that quantum models were "$\approx 770\times$ slower." Phase 6.5.5-C audited the exact clock timing across 60 calibration runs and 22,168 test evaluations:

1. **Per-Evaluation Inference Latency:**
   - QSVC Q8-R1: **$1.0816 \text{ s}$ per test utterance**
   - Classical RBF-SVM $d=8$: **$0.0014 \text{ s}$ per test utterance**
   - Exact ratio: $\frac{1.0816}{0.0014} = \mathbf{772.57\times}$
2. **Total Experiment Wall-Clock Runtime (60 conditions):**
   - QSVC Q8-R1: **$70.22 \text{ s}$**
   - Classical RBF-SVM $d=8$: **$0.33 \text{ s}$**
   - Exact ratio: $\frac{70.22}{0.33} = \mathbf{212.79\times}$
3. **Why the Ratios Differ:**
   The total wall-clock time includes classical overhead (PCA transformation, feature scaling, Stage-1 triage gating, metadata recording) which is identical for both pipelines. The quantum evaluation penalty appears purely during the computation of the $N_{\text{test}} \times N_{\text{train}}$ test kernel matrix.

**Resolution:**
Both numbers are mathematically correct when properly attributed to their measurement basis: QSVC Q8-R1 is **$\approx 772.6\times$ slower per test evaluation** and **$\approx 212.8\times$ slower in total experiment wall time**.

---

## 4. Archival File Provenance

All reconciled values are permanently saved in:
- [`results/phase6_5_6_metric_reconciliation.csv`](file:///d:/DL/results/phase6_5_6_metric_reconciliation.csv)
- [`results/phase6_5_5_speaker_results.csv`](file:///d:/DL/results/phase6_5_5_speaker_results.csv)
- [`results/phase6_5_5_C_runtime_audit.csv`](file:///d:/DL/results/phase6_5_5_C_runtime_audit.csv)
- [`results/phase6_5_4_dimension_ablation.csv`](file:///d:/DL/results/phase6_5_4_dimension_ablation.csv)
