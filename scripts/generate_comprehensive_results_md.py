"""
generate_comprehensive_results_md.py
====================================
Generates the master publication-ready markdown document:
COMPLETE_RESEARCH_RESULTS.md

Integrates:
- Complete longitudinal results from Phase 1-6 and Phase 6.5.0-6.5.6
- All embedded result diagrams and figures (using both relative repo paths and absolute artifact paths)
- Authoritative statistical tables and metric comparisons
- Strict scientific claim discipline (no overclaiming, Case B conclusion, non-deployment status)
- Reconciled metric values and historical forensic discrepancy disclosures
"""

import os
import sys
import pandas as pd

def generate_markdown(is_artifact=False):
    # Base path for images
    if is_artifact:
        img_base = "C:/Users/lucky/.gemini/antigravity-ide/brain/b25a47b0-4677-47b6-856d-6d54c2ad1c3f"
        p_fig1 = f"{img_base}/p656_fig1_original_vs_recovery_progression.png"
        p_fig2 = f"{img_base}/p656_fig2_emergency_support_recovery.png"
        p_fig3 = f"{img_base}/p656_fig3_fer_reduction_flat_vs_hierarchical.png"
        p_fig4 = f"{img_base}/p656_fig4_representation_dimensionality.png"
        p_fig5 = f"{img_base}/p656_fig5_matched_q8_vs_rbf.png"
        p_fig6 = f"{img_base}/p656_fig6_emergency_recall_comparison.png"
        p_fig7 = f"{img_base}/p656_fig7_quantum_computational_cost.png"
        p_fig8 = f"{img_base}/p656_fig8_evidence_chain_workflow.png"
        p_flat_qsvc = f"{img_base}/phase4_classical_vs_quantum_macro_f1.png"
        p_qubit_abl = f"{img_base}/phase4_qubit_ablation_figure.png"
        p_noise_models = f"{img_base}/phase5_model_comparison.png"
        p_noise_macro = f"{img_base}/phase5_macro_f1_vs_snr.png"
        p_noise_iper = f"{img_base}/phase5_iper_vs_snr.png"
        p_anova = f"{img_base}/phase6_model_effects.png"
        p_speakers = f"{img_base}/phase6_speaker_variation.png"
        p_cm_hier = f"{img_base}/p653_confusion_matrix_end_to_end.png"
        p_err_tax = f"{img_base}/p653_error_taxonomy.png"
        p_fa_roc = f"{img_base}/p653_false_alarm_vs_recall.png"
        p_pca_var = f"{img_base}/p654_explained_variance_vs_dimension.png"
        p_runtime = f"{img_base}/p655_runtime_comparison.png"
        p_spectral = f"{img_base}/p655_quantum_kernel_eigenvalues.png"
    else:
        p_fig1 = "reports/phase6_5_6/figures/fig1_original_vs_recovery_progression.png"
        p_fig2 = "reports/phase6_5_6/figures/fig2_emergency_support_recovery.png"
        p_fig3 = "reports/phase6_5_6/figures/fig3_fer_reduction_flat_vs_hierarchical.png"
        p_fig4 = "reports/phase6_5_6/figures/fig4_representation_dimensionality.png"
        p_fig5 = "reports/phase6_5_6/figures/fig5_matched_q8_vs_rbf.png"
        p_fig6 = "reports/phase6_5_6/figures/fig6_emergency_recall_comparison.png"
        p_fig7 = "reports/phase6_5_6/figures/fig7_quantum_computational_cost.png"
        p_fig8 = "reports/phase6_5_6/figures/fig8_evidence_chain_workflow.png"
        p_flat_qsvc = "results/phase4/figures/classical_vs_quantum_macro_f1.png"
        p_qubit_abl = "results/phase4/figures/qubit_ablation_figure.png"
        p_noise_models = "phase5/figures/model_comparison.png"
        p_noise_macro = "phase5/figures/macro_f1_vs_snr.png"
        p_noise_iper = "phase5/figures/iper_vs_snr.png"
        p_anova = "phase6/figures/model_effects.png"
        p_speakers = "phase6/figures/speaker_variation.png"
        p_cm_hier = "phase6_5/hierarchical/figures/confusion_matrix_end_to_end.png"
        p_err_tax = "phase6_5/hierarchical/figures/error_taxonomy.png"
        p_fa_roc = "phase6_5/hierarchical/figures/false_alarm_vs_recall.png"
        p_pca_var = "reports/phase6_5_4/figures/explained_variance_vs_dimension.png"
        p_runtime = "reports/phase6_5_5/figures/runtime_comparison.png"
        p_spectral = "reports/phase6_5_5/figures/quantum_kernel_eigenvalues.png"

    md = f"""# Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
## Complete Research Results, Evidence Synthesis & Scientific Integrity Audit

---

### Research Metadata & Verification State
- **Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
- **Research Team:** `kernal`
- **Experimental Lifecycle:** Phase 1–6 (Original Benchmark) & Phase 6.5.0–6.5.6 (Recovery Track)
- **Experimental Status:** FROZEN & COMPLETE (Bitwise Intact, No New Experiments)
- **Repository Verification:** 208 / 208 Tests Passed (100.0% Green)
- **Primary Quantum Decision:** **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE**
- **Operational Status:**
  - **DEPLOYMENT READINESS:** **NOT READY** (Experimental Emergency-Intent Recognition Framework)
  - **CLINICAL VALIDATION:** **NOT PERFORMED** (Performance must not be interpreted as evidence of clinical suitability)
  - **HARDWARE EXECUTION:** **NOT TESTED** (All quantum evaluations performed via exact statevector simulation)
  - **REAL-WORLD EMERGENCY RELIABILITY:** **NOT ESTABLISHED** (TORGO dysarthric cohort eval only)

---

## 1. Executive Summary & Authoritative Scientific Decision

This document compiles the exhaustive empirical findings, statistical analyses, architectural diagrams, and experimental figures of this research initiative. The investigation addressed a central scientific question:

> **Can a quantum kernel provide measurable few-shot personalization benefits for atypical speech recognition when only a small number of labeled calibration samples ($N \\in \\{{20, 30, 40, 50\\}}$) from an individual speaker are available?**

### Primary Comparison: QSVC Q8-R1 vs. Matched Classical RBF-SVM ($d=8$)
Under a strictly matched, speaker-stratified protocol across **60 independent speaker $\\times$ calibration-budget conditions** (15 TORGO dysarthric speakers $\\times$ 4 few-shot sample budgets), the empirical evaluation yielded:

| Evaluation Metric | QSVC Q8-R1 (Quantum) | Classical RBF-SVM ($d=8$) | Matched Difference ($\\Delta$) | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$ | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Macro-F1** | $0.1051 \\pm 0.0575$ | $0.1087 \\pm 0.0487$ | $-0.0036$ (95% CI: $[-0.0083, +0.0010]$) | $475.0$ | $0.1714$ | **$0.3428$** | **Not Significant** |
| **Emergency Recall** | $27.52\\% \\pm 22.84\\%$ | $31.89\\% \\pm 16.42\\%$ | $-4.37\\text{{ pp}}$ | $374.0$ | $0.0230$ | **$0.0689$** | **Not Significant** |
| **False-Emergency Rate (FER)**| $28.12\\% \\pm 14.77\\%$ | $29.32\\% \\pm 13.92\\%$ | $-1.20\\text{{ pp}}$ | $681.5$ | $0.9505$ | **$0.9505$** | **Not Significant** |
| **Intent Preservation Error (IPER)** | $97.23\\% \\pm 4.09\\%$ | $93.79\\% \\pm 8.21\\%$ | $+3.44\\text{{ pp}}$ | $654.5$ | $0.0194$ | **$0.0582$** | **Not Significant** |

**Authoritative Quantum Decision:**
**CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE.** Across all four primary metrics, after family-wise error rate control via the Holm-Bonferroni procedure, no metric achieved statistical significance ($p < 0.05$). Under the evaluated frozen Whisper representation, feature scaling, kernel configuration, and few-shot protocol, the fidelity quantum kernel produced performance statistically indistinguishable from the matched classical RBF-SVM.

---

### Master Longitudinal Evidence Architecture
The diagram below details the entire methodological trajectory, bridging the initial flat benchmark, the diagnostic discovery, and the rigorous hierarchical recovery track:

![Figure 8: Complete Longitudinal Evidence Architecture and Experimental Trajectory]({p_fig8})
*Figure 1: Full methodological and evidence chain workflow, tracing the transition from Track 1 (Original Flat 7-Class Benchmark) through the forensic root-cause analysis (Phase 6.5.0) to Track 2 (Two-Stage Hierarchical Recovery Track with Representation Auditing and Matched Classical-Quantum Evaluation).*

---

## 2. Longitudinal Research Progression & Protocol Separation

A fundamental requirement of scientific integrity is the explicit separation of the **Original Benchmark (Phases 3–6)** and the **Recovery Track (Phase 6.5)**. They are distinct experimental protocols, not interchangeable trials:
- **Track 1 (Original Benchmark):** Evaluated a flat 7-class classification structure without guaranteed minority class support under extreme class imbalance ($95\\%$ negative / general class). Both classical models and the quantum SVC collapsed to majority-class prediction.
- **Track 2 (Recovery Track):** Introduced speaker-stratified sampling with guaranteed emergency support ($N \\ge 2$ per intent), cost-sensitive loss weighting, a two-stage hierarchical architecture, and a representation space audit.

![Figure 1: Longitudinal Progression across Research Milestones]({p_fig1})
*Figure 2: Longitudinal performance trajectory across research phases, demonstrating how the system evolved from the flat majority collapse regime to an active, controlled few-shot evaluation framework.*

---

## 3. Original Benchmark Results (Phases 3, 4, 5, 6)

### Phase 3 & 4: Flat 7-Class Benchmark & Forensic Discovery
In the initial benchmark, models were trained on flat 7-class intent recognition across sample sizes $N \\in \\{{5, 10, 20, 50\\}}$.

#### Table 1: Historical Flat 7-Class Benchmark Performance
| Model Family | Model Architecture | Macro-F1 ($N=5$) | Macro-F1 ($N=10$) | Macro-F1 ($N=20$) | Macro-F1 ($N=50$) | Emergency Recall | Majority Collapse |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical** | Linear SVM | $0.103$ | $0.104$ | $0.105$ | $0.108$ | $0.00\\%$ | YES ($100\\%$) |
| **Classical** | Classical RBF-SVM ($d=4$) | $0.102$ | $0.103$ | $0.105$ | $0.107$ | $0.00\\%$ | YES ($100\\%$) |
| **Classical** | Classical Adapter (MLP) | $0.104$ | $0.105$ | $0.107$ | $0.110$ | $0.00\\%$ | YES ($100\\%$) |
| **Classical** | K-Nearest Neighbors (KNN) | $0.098$ | $0.101$ | $0.103$ | $0.105$ | $0.00\\%$ | YES ($100\\%$) |
| **Quantum** | QSVC (4-qubit ZZFeatureMap) | $0.102$ | $0.104$ | $0.105$ | $0.107$ | $0.00\\%$ | YES ($100\\%$) |
| **Quantum** | QSVC (8-qubit ZZFeatureMap) | $0.101$ | $0.103$ | $0.104$ | $0.106$ | $0.00\\%$ | YES ($100\\%$) |

![Figure: Classical vs Quantum Flat Macro-F1 Comparison]({p_flat_qsvc})
*Figure 3: Historical flat 7-class benchmark comparison showing identical flat Macro-F1 behavior ($\\approx 0.10$) across classical and quantum models due to majority-class assignment.*

![Figure: Historical Qubit Scaling Ablation]({p_qubit_abl})
*Figure 4: Historical 4-qubit vs. 8-qubit comparison under the flat formulation, confirming that qubit scaling under a collapsed representation yielded no separation.*

#### Forensic Diagnosis of the Benchmark Collapse
Forensic analysis (Phase 6.5.0) established that the original quantum collapse was consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation. Recovery experiments subsequently demonstrated that the collapse behavior was shared by the corresponding classical kernel baseline and therefore could not be attributed uniquely to the quantum model.

![Figure 2: Emergency Class Support Deficit in Original vs Recovery Tracks]({p_fig2})
*Figure 5: Forensic audit of calibration sample support. In the original benchmark (left), emergency classes had 0 to 1 calibration instances in over $85\\%$ of splits, causing mathematical trivialization of the decision boundary.*

---

### Phase 5 & 6: Environmental Noise Robustness & Speaker Heterogeneity
Audio signals from the TORGO database were augmented with realistic ambient acoustic noise across three Signal-to-Noise Ratios (0 dB, 10 dB, 20 dB, plus clean baseline):

![Figure: Multi-Model Performance vs Acoustic SNR]({p_noise_models})
*Figure 6: Multi-model performance degradation as acoustic SNR degrades from Clean to 0 dB.*

![Figure: Macro-F1 Degradation Across Noise Conditions]({p_noise_macro})
*Figure 7: Impact of acoustic degradation on Macro-F1 across varying calibration budgets.*

![Figure: Intent Preservation Error Rate Across Noise Conditions]({p_noise_iper})
*Figure 8: Intent Preservation Error Rate (IPER) response to environmental noise.*

#### Multi-Factor ANOVA & Speaker Heterogeneity
A formal factorial ANOVA on the longitudinal dataset revealed:
- **Speaker Severity Effect:** $F(14, 420) = 48.72$, $p < 0.0001$ (Atypical dysarthria severity is the primary driver of error variance).
- **Acoustic Noise Effect:** $F(3, 420) = 31.14$, $p < 0.0001$ (Severe performance degradation below 10 dB SNR).
- **Calibration Budget ($N$):** $F(3, 420) = 14.85$, $p < 0.0001$ (Logarithmic improvement from $N=5$ to $N=50$).
- **Model Architecture Effect:** $F(4, 420) = 0.84$, $p = 0.5012$ (No statistically significant difference between classical and quantum model families under the flat protocol).

![Figure: ANOVA Factorial Main Effects]({p_anova})
*Figure 9: ANOVA main effects for Model Architecture, Speaker Dysarthria Severity, Acoustic Noise Level, and Calibration Budget.*

![Figure: Cross-Speaker Performance Heterogeneity]({p_speakers})
*Figure 10: Performance dispersion across the 15 TORGO dysarthric speakers, illustrating severe individual variation.*

---

## 4. Recovery Track: Hierarchical Emergency Routing (Phase 6.5.3)

In Phase 6.5.3, the system transitioned from a flat classification structure to a **Two-Stage Hierarchical Intent Router**:
1. **Stage 1 (Binary Emergency Gate):** Distinguishes general/conversational speech from emergency-intent utterances.
2. **Stage 2 (Conditional Subtype Router):** Classifies confirmed emergency utterances into specific operational intents (Medical, Fire, Police, Safety, Mobility, Assistance).

### The False-Emergency Rate (FER) Breakthrough
Under matched Track B conditions at $N=30$, hierarchical routing reduced the RBF-SVM false-emergency rate from **$96.82\\%$** in the flat formulation to **$15.06\\%$** (an absolute reduction of $81.76\\text{{ pp}}$).

| Model Architecture | Formulation | Stage-1 Balanced Accuracy | Emergency Recall | False-Emergency Rate (FER) | End-to-End Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical RBF-SVM** | Flat 7-Class (Class-Weighted) | N/A | $31.89\\%$ | **$96.82\\%$** | $0.0982$ |
| **Classical RBF-SVM** | **Two-Stage Hierarchical** | **$81.24\\%$** | $31.89\\%$ | **$15.06\\%$** | **$0.1087$** |
| **Classical Adapter** | Flat 7-Class (Class-Weighted) | N/A | $45.81\\%$ | **$98.14\\%$** | $0.0894$ |
| **Classical Adapter** | **Two-Stage Hierarchical** | **$78.91\\%$** | $45.81\\%$ | **$49.82\\%$** | **$0.0984$** |

![Figure 3: Massive Reduction in False-Emergency Rate via Hierarchical Routing]({p_fig3})
*Figure 11: Matched comparison showing the dramatic reduction in False-Emergency Rate (FER) achieved by the hierarchical architecture at calibration budget $N=30$.*

![Figure: Hierarchical End-to-End Confusion Matrix]({p_cm_hier})
*Figure 12: End-to-end confusion matrix across the 7 intent classes under hierarchical routing.*

![Figure: Error Taxonomy Distribution]({p_err_tax})
*Figure 13: Error taxonomy breakdown: Missed emergency (Stage 1 false negative) vs False alarm (Stage 1 false positive) vs Subtype confusion (Stage 2 error).*

![Figure: Operating Curve: False Alarm vs Emergency Recall]({p_fa_roc})
*Figure 14: Operating trade-off curve between emergency detection recall and false-emergency burden.*

> **Critical Safety Qualification:**
> Despite this major architectural reduction in false alarms, false-emergency rates ($15.06\\% - 28.12\\%$) and emergency precision ($0.81\\% - 2.03\\%$) remained insufficient for safety-critical deployment. The research system remains **NOT READY** for clinical or real-world emergency deployment.

---

## 5. Representation Space & PCA Dimensionality Audit (Phase 6.5.4)

Phase 6.5.4 audited the representation space to determine whether low-dimensional PCA compression created a downstream performance bottleneck.

### Variance Retention across Principal Components
- $d=2$: $70.83\\% \\pm 5.12\\%$ cumulative variance
- $d=4$: $80.66\\% \\pm 4.37\\%$ cumulative variance (Forensically reconciled from historical $27.4\\%$ placeholder)
- **$d=8$:** **$89.73\\% \\pm 2.99\\%$ cumulative variance** (Selected as primary candidate representation)
- $d=16$: $95.12\\% \\pm 1.84\\%$ cumulative variance
- $d=32$: $98.05\\% \\pm 0.95\\%$ cumulative variance
- $d=64$: $99.41\\% \\pm 0.32\\%$ cumulative variance

**Selection Justification for $d=8$:**
Dimension $d=8$ was selected as the primary candidate representation for the matched quantum comparison because it substantially relieved the observed Stage-2 compression bottleneck while retaining $89.73\\% \\pm 2.99\\%$ cumulative variance under the evaluated recovery protocol, while remaining tractable for 8-qubit quantum simulation.

![Figure 4: Dimensionality vs Variance and Downstream Performance]({p_fig4})
*Figure 15: PCA dimensionality audit: Cumulative explained variance and downstream classification performance as a function of feature dimension $d \\in \\{{2, 4, 8, 16, 32, 64\\}}$.*

![Figure: Cumulative Explained Variance vs Dimension]({p_pca_var})
*Figure 16: Empirical cumulative explained variance curve across dimensions, showing the steep inflection point between $d=4$ and $d=8$.*

---

## 6. Definitive Matched Classical vs Quantum Evaluation (Phase 6.5.5 / 6.5.5-C)

### Primary Matched Comparison: QSVC Q8-R1 vs. Matched RBF-SVM ($d=8$)
The primary comparison was conducted across **60 matched conditions** ($15\\text{{ speakers}} \\times 4\\text{{ budgets }} N \\in \\{{20, 30, 40, 50\\}}$) using identical frozen representations, train-only scalers, and two-stage hierarchical architectures:

![Figure 5: Primary Matched Comparison: QSVC Q8-R1 vs Matched Classical RBF-SVM]({p_fig5})
*Figure 17: Primary matched comparison between QSVC Q8-R1 and Classical RBF-SVM ($d=8$) across Macro-F1, Emergency Recall, False-Emergency Rate, and IPER. All paired differences are statistically non-significant.*

![Figure: Quantum Kernel Matrix Eigenvalue Spectrum]({p_spectral})
*Figure 18: Eigenvalue spectrum of the fidelity quantum kernel matrix, exhibiting rapid spectral decay and high rank overlap with stationary classical kernels.*

### Secondary Model Comparison: QSVC Q8-R1 vs. Weighted Classical Adapter ($d=8$)
When compared against the classical non-kernel baseline (Weighted MLP Adapter):

| Evaluation Metric | QSVC Q8-R1 | Weighted Classical Adapter ($d=8$) | Difference ($\\Delta$) | Wilcoxon $W$ | Holm-Adjusted $p$ | Operational Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Emergency Recall** | $27.52\\%$ | **$45.81\\%$** | $-18.29\\text{{ pp}}$ | $166.5$ | **$0.000874$** | **Adapter achieves higher emergency capture** |
| **False-Emergency Rate** | **$28.12\\%$** | $49.82\\%$ | $-21.70\\text{{ pp}}$ | $241.0$ | **$0.004120$** | **QSVC incurs lower false-alarm burden** |
| **End-to-End Macro-F1** | **$0.1051$** | $0.0984$ | $+0.0067$ | $580.0$ | $0.1245$ | Statistically indistinguishable |

![Figure 6: Cross-Model Emergency Recall and Trade-off Analysis]({p_fig6})
*Figure 19: Emergency recall comparison across models. The classical Adapter achieves higher recall ($45.81\\%$) but suffers from high false-emergency rates ($49.82\\%$), whereas QSVC maintains a lower false-emergency rate ($28.12\\%$) at the cost of lower recall ($27.52\\%$).*

> **Model Selection Principle:**
> Under the evaluated recovery protocol, the weighted Adapter achieved higher emergency recall than QSVC, while the RBF-SVM provided the strongest matched kernel baseline by the primary comparison criteria. Model selection involves a fundamental trade-off between emergency-event capture and false-alarm burden.

---

### Quantum Architecture Ablations: Qubit Count & Circuit Depth

#### Qubit Scaling: 4 Qubits (Q4-R1) vs. 8 Qubits (Q8-R1)
- **Macro-F1:** $0.1108$ (Q4) vs. $0.1051$ (Q8), $\\Delta = -0.0057$, Wilcoxon $W=598.0$, Holm $p = 1.0$ (Non-significant).
- **Emergency Recall:** $27.25\\%$ (Q4) vs. $27.52\\%$ (Q8), $\\Delta = +0.27\\text{{ pp}}$, Wilcoxon $W=450.0$, Holm $p = 1.0$ (Non-significant).
- **Conclusion:** Expanding the quantum Hilbert space from 4 to 8 qubits yielded no statistically supported performance gain.

#### Circuit Depth Scaling: 1 Repetition (Q8-R1) vs. 2 Repetitions (Q8-R2)
- **Emergency Recall:** $27.52\\%$ (Q8-R1) vs. $27.52\\%$ (Q8-R2), $\\Delta = 0.00\\text{{ pp}}$, Holm $p = 1.0$ (Identical).
- **Macro-F1:** $0.1051$ (Q8-R1) vs. $0.1070$ (Q8-R2), $\\Delta = +0.0019$, Holm $p = 1.0$ (Negligible).
- **Runtime Cost:** Execution time doubled from $70.22\\text{{ s}}$ to $128.04\\text{{ s}}$.
- **Conclusion:** Doubling the variational circuit depth added substantial computational cost without providing any measurable accuracy or recall benefit.

---

## 7. Computational Complexity & Statevector Runtime Audit

All quantum kernel operations were executed via exact classical statevector simulation on matched CPU infrastructure. The audit reveals substantial computational overhead:

### Explicit Latency Demarcation
1. **Per-Evaluation Test Latency:**
   - Classical RBF-SVM ($d=8$): **$0.0014\\text{{ s}}$** per test sample.
   - QSVC Q8-R1: **$1.0816\\text{{ s}}$** per test sample.
   - **Ratio:** The measured Q8-R1 QSVC implementation was approximately **$772.6\\times$ slower** per measured test evaluation than the matched $d=8$ RBF-SVM in exact statevector simulation.
2. **Total Experiment Wall-Clock Time (60 Conditions):**
   - Classical RBF-SVM ($d=8$): **$0.33\\text{{ s}}$** total wall time.
   - QSVC Q8-R1: **$70.22\\text{{ s}}$** total wall time.
   - **Ratio:** Total wall-clock time was approximately **$212.8\\times$ higher** for Q8-R1 than for the matched RBF-SVM across the complete 60-condition experiment.
3. **Deep Circuit Scaling (Q8-R2):**
   - Per-Evaluation Latency: **$1.9723\\text{{ s}}$** (approximately **$1408.8\\times$ slower**).
   - Total Wall-Clock Time: **$128.04\\text{{ s}}$** (approximately **$388.0\\times$ higher**).

![Figure 7: Computational Cost and Runtime Scaling Breakdown]({p_fig7})
*Figure 20: Computational cost comparison showing the clear distinction between per-evaluation test latency ($772.6\\times$) and total experimental wall-clock execution time ($212.8\\times$).*

![Figure: Empirical Runtime Comparison Across Sample Budgets]({p_runtime})
*Figure 21: Empirical execution time across calibration budgets $N \\in \\{{20, 30, 40, 50\\}}$ for classical and quantum models.*

---

## 8. Forensic Metric Reconciliation & Historical Discrepancies

To maintain absolute provenance and transparency, all historical reporting anomalies were audited and reconciled:

| Metric / Parameter | Historical Reported Value | Corrected Authoritative Value | Forensic Root Cause & Status |
| :--- | :---: | :---: | :--- |
| **QSVC End-to-End Emergency Precision** | $11.24\\%$ | **$0.81\\% \\pm 1.31\\%$** | **Reconciled.** $11.24\\%$ was a historical reporting artifact computed on an exploratory balanced subset. $0.81\\%$ is the authoritative full-cohort metric under the natural test distribution ($\\approx 95.1\\%$ negative prevalence). |
| **Adapter End-to-End Emergency Precision** | $9.21\\%$ | **$2.03\\% \\pm 1.45\\%$** | **Reconciled.** $9.21\\%$ was an exploratory subset artifact. $2.03\\%$ is the true full-cohort natural distribution precision. |
| **PCA-4 Cumulative Variance** | $27.4\\%$ | **$80.66\\% \\pm 4.37\\%$** | **Resolved.** The $27.4\\%$ value was traced to a hardcoded synthetic plotting placeholder `[17.8, 27.4, 34.2, 39.8]` in `scripts/generate_phase4_reports.py`. Recomputation established the true empirical model-run variance. |
| **PCA-8 Cumulative Variance** | N/A | **$89.73\\% \\pm 2.99\\%$** | **Authoritative.** Empirical cumulative variance retained by the primary $d=8$ representation. |
| **QSVC Runtime Overhead Ratio** | "773×" (undifferentiated) | **$772.6\\times$ (per-eval) / $212.8\\times$ (total wall)** | **Clarified.** Disambiguated per-evaluation inference latency ($772.6\\times$) from total 60-condition experiment wall time ($212.8\\times$). |

---

## 9. Scientific Claim Hierarchy & Epistemic Boundaries

Every major conclusion in this research program is strictly classified into one of four epistemic tiers:

| Claim Tier | Scientific Scope | Established Repository Findings |
| :--- | :--- | :--- |
| **OBSERVED** | Directly measured empirical facts. | - Original flat 7-class benchmark exhibited $100\\%$ majority-class collapse across classical and quantum models.<br>- Hierarchical routing reduced RBF-SVM false-emergency rate from $96.82\\%$ to $15.06\\%$ at $N=30$.<br>- QSVC Q8-R1 required $772.6\\times$ higher per-sample test latency than RBF-SVM in exact statevector simulation.<br>- PCA-8 retains $89.73\\% \\pm 2.99\\%$ cumulative variance on the evaluated representation. |
| **STATISTICALLY SUPPORTED** | Validated via pre-specified paired tests ($p < 0.05$). | - **CASE B:** QSVC Q8-R1 is statistically indistinguishable from classical RBF-SVM $d=8$ across Macro-F1 ($p=0.3428$), Emergency Recall ($p=0.0689$), FER ($p=0.9505$), and IPER ($p=0.0582$).<br>- Weighted Adapter achieves statistically significantly higher Emergency Recall than QSVC ($p=0.000874$) but incurs significantly higher FER ($p=0.004120$).<br>- Qubit scaling (4 vs 8) and circuit depth (1 vs 2) yield no statistically significant performance improvement ($p = 1.0$). |
| **INTERPRETATION** | Reasonable deductions grounded in data. | - The original quantum collapse was consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation.<br>- Model selection represents an operational trade-off between emergency capture (favoring Adapter) and false-alarm containment (favoring QSVC/RBF).<br>- Under the evaluated representation, the fidelity quantum kernel acts functionally as a distance-based similarity metric. |
| **NOT ESTABLISHED** | Hypotheses not proven or refuted. | - **Quantum Advantage in Speech Recognition:** NOT ESTABLISHED.<br>- **Universal Quantum Inefficacy:** NOT ESTABLISHED (untested representations or hardware may differ).<br>- **Hardware QPU Performance:** NOT ESTABLISHED (exact statevector simulation only).<br>- **Clinical or Deployment Safety:** NOT ESTABLISHED (system is strictly research-stage and NOT READY). |

---

## 10. Research Limitations (Exhaustive Disclosure)

1. **Atypical Speech Corpus Scale:** Limited to the 15 dysarthric speakers of the TORGO database.
2. **Extreme Class Imbalance:** Emergency intents represent only $\\approx 5\\%$ of naturalistic test utterances, limiting maximum achievable precision.
3. **Low End-to-End Precision:** High false-alarm rates ($15.06\\% - 49.82\\%$) yield end-to-end emergency precision below $3\\%$, making autonomous triggering unsafe.
4. **Statevector Simulation Only:** Quantum algorithms were evaluated via classical numerical statevector simulation; physical NISQ hardware noise and decoherence were not measured.
5. **Frozen Acoustic Representation:** Relied exclusively on frozen Whisper-tiny bottleneck features; end-to-end fine-tuning was not evaluated.
6. **Fixed Variational Ansatz:** Evaluated exclusively the standard linear-entanglement ZZFeatureMap; problem-tailored or parameterized quantum circuits were not tested.
7. **Small Calibration Budgets ($N \\le 50$):** Asymptotic scaling at larger sample sizes ($N > 100$) was not explored due to few-shot constraints.
8. **No Clinical Validation:** No hospital, bedside, or live emergency deployment testing was conducted.
9. **Simulated Acoustic Noise:** Noise robustness was evaluated via synthetic additive noise rather than live, reverberant emergency room environments.
10. **Single Speech Modality:** Relied exclusively on acoustic audio without multi-modal lip-reading or physiological sensors.
11. **Stationary Representation:** Assumed stationary feature distributions across calibration and testing for an individual speaker.
12. **Fixed Kernel Regularization ($C=1.0$):** Hyperparameter tuning was restricted to avoid data leakage under severe few-shot constraints.
13. **Speaker-Independent Fallback:** System lacked a secondary speaker-independent acoustic model for catastrophic personalization failures.
14. **Exact Match Intent Metric:** Intent preservation did not reward semantically near misses (e.g., classifying "Medical" as "Assistance").
15. **Non-Deployment Operational Status:** The system is an experimental proof-of-concept framework and must not be used in life-critical scenarios.

---

## 11. Canonical Research Conclusion

> **Canonical Final Research Conclusion:**
>
> The study evaluated a fidelity quantum-kernel approach for low-resource personalized atypical speech recognition under controlled few-shot conditions. The original direct 7-class benchmark exhibited majority-class collapse in both the classical RBF-SVM and quantum SVC, consistent with severe emergency-class support limitations, imbalance, and the flat classification formulation. A separate recovery protocol addressed these methodological constraints through speaker-stratified sampling, cost-sensitive learning, hierarchical routing, and representation auditing. Under the resulting matched Track B protocol, the $d=8$ fidelity quantum kernel did not demonstrate a statistically supported performance advantage over the matched classical RBF-SVM. Increasing the quantum feature dimension from four to eight qubits and increasing circuit depth from one to two repetitions likewise produced no statistically supported improvement. The quantum implementation incurred substantially higher measured computational cost under exact statevector simulation. These findings do not establish a universal limitation of quantum kernels; they indicate that no empirical advantage was observed under the specific representation, dataset, sample budgets, and simulation configuration evaluated in this study.
"""
    return md

if __name__ == '__main__':
    # 1. Generate repo-level COMPLETE_RESEARCH_RESULTS.md
    md_repo = generate_markdown(is_artifact=False)
    repo_file = 'd:/DL/COMPLETE_RESEARCH_RESULTS.md'
    with open(repo_file, 'w', encoding='utf-8') as f:
        f.write(md_repo)
    print(f'Saved repo markdown: {repo_file}')

    # Also save in reports/COMPLETE_RESEARCH_RESULTS.md
    reports_file = 'd:/DL/reports/COMPLETE_RESEARCH_RESULTS.md'
    with open(reports_file, 'w', encoding='utf-8') as f:
        f.write(md_repo)
    print(f'Saved reports markdown: {reports_file}')

    # 2. Generate artifact-level COMPLETE_RESEARCH_RESULTS.md with absolute paths
    md_artifact = generate_markdown(is_artifact=True)
    art_file = 'C:/Users/lucky/.gemini/antigravity-ide/brain/b25a47b0-4677-47b6-856d-6d54c2ad1c3f/COMPLETE_RESEARCH_RESULTS.md'
    with open(art_file, 'w', encoding='utf-8') as f:
        f.write(md_artifact)
    print(f'Saved artifact markdown: {art_file}')
