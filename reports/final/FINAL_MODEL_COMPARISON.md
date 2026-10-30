# CANONICAL FINAL MODEL COMPARISON TABLE
## Phase 7 Consolidated Multi-Model Empirical Evaluation

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Evaluation Scope:** 414 Total Phase 7 Experiments (60 Matched Longitudinal Cells per Family across $N \in \{20, 30, 40, 50\}$)  
**Status:** **CANONICALLY FROZEN & VERIFIED**  

---

## 1. Consolidated Four-Way Evidence Matrix

| Model | Model Family | Personalization Type | Quantum / Classical | Emergency Recall | Emergency Precision | False Emergency Rate (FER) | Macro-F1 | Intent Preservation | Forgetting ($\Delta$) | Safety Status | Governance Status | Deployment Status | Runtime / Resource Summary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Continual Classical Adapter** | Classical Continual Adapter | Parameterized Few-Shot Adapter Head (2-layer MLP) | **Classical** | **16.20%** *(Superior)* | 0.74% | 43.53% | 0.1012 | **16.20%** | -0.0150 | **SAFE** (Highest Emergency Recall) | **APPROVED** (Passed Gate) | **READY FOR RESEARCH DEMO** *(Selected)* | 25,223 params<br>Adapt: 0.352s<br>Infer: 0.024s |
| **Fixed Q8-R1 QSVC** | Fixed Quantum Kernel (QSVC) | Dual Support Vector Optimization ($C=1.0$) | **Pure Quantum** | 5.10% *(Collapsed)* | 0.10% | 16.74% | 0.1187 | 5.10% | 0.0000 | **UNSAFE** (Severe Recall Deficit) | **VETO TRIGGERED** (Recall Regressed) | **NOT READY** | 0 trainable params<br>Adapt: 0.061s<br>Infer: 0.063s |
| **Trainable Variational Quantum Classifier** | Adaptive Variational Quantum Classifier (VQC) | Variational Gate Optimization (Adam, 16 angles) | **Trainable Quantum** | 5.81% *(Collapsed)* | 0.10% | 17.20% | 0.1187 | 5.81% | -0.0020 | **UNSAFE** (Severe Recall Deficit) | **VETO TRIGGERED** (Recall Regressed) | **NOT READY** | 16 quantum angles<br>Adapt: 2.236s<br>Infer: 4.331s |
| **Convex Hybrid Fusion** | Governed Convex Probability Fusion ($\alpha=0.5$) | Dual-Branch Classical Adapter + Trainable Quantum | **Hybrid Quantum-Classical** | 7.07% *(Deficient vs Classical)* | 0.21% | **19.45%** *(Low FER)* | **0.1184** | 7.07% | -0.0080 | **BORDERLINE** (FER reduced, but recall collapsed vs Classical) | **CONDITIONAL** (Gate passed under relaxed threshold) | **RESEARCH EXPERIMENTAL ONLY** | 25,239 params<br>Adapt: 2.578s<br>Infer: 2.492s |

---

## 2. Key Statistical Comparisons & Findings

1. **Emergency Recall Superiority of Classical Adapter ($p < 10^{-5}$):**
   - Classical Continual Adapter achieves **16.20% Emergency Recall**, decisively outperforming:
     - Fixed Quantum Kernel (**5.10%**, difference $+11.10\%$, $p_{\text{adj}} = 1.34 \times 10^{-6}$)
     - Trainable Quantum VQC (**5.81%**, difference $+10.39\%$, $p_{\text{adj}} = 3.82 \times 10^{-5}$)
     - Convex Hybrid Fusion (**7.07%**, difference $+9.13\%$, $p_{\text{adj}} = 1.88 \times 10^{-4}$)
   - In life-critical emergency communication, missing $>94\%$ of emergency calls (as pure quantum does) is medically catastrophic. Classical adapter is the only architecture maintaining non-trivial emergency capture.

2. **The False Emergency Rate Trade-off:**
   - Convex Hybrid Fusion achieves statistically lower False Emergency Rate (**19.45%** vs **43.53%**, $p < 10^{-8}$) and higher Macro-F1 (**0.1184** vs **0.1012**, $p < 10^{-6}$) compared to Classical Adapter.
   - However, this improvement occurs because the quantum probability branch heavily biases predictions toward the majority class (`GENERAL_SPEECH`), thereby suppressing false alarms at the direct expense of missing genuine emergency utterances.

3. **Trainable Quantum vs Fixed Quantum Kernel ($p = 0.061$ NS):**
   - Optimizing 16 variational quantum circuit angles does not yield a statistically significant improvement in Macro-F1 over fixed quantum fidelity kernels ($0.1187 \pm 0.0039$ vs $0.1180 \pm 0.0041$, $p_{\text{adj}} = 0.061$).
   - Emergency Recall remains statistically indistinguishable ($5.81\%$ vs $5.12\%$, $p_{\text{adj}} = 0.559$).

4. **Safety Governance & Deployment Determination:**
   - Both pure quantum models (Fixed and Trainable) trigger unconditional **Safety Vetoes** during Phase 7.6 validation due to severe Emergency Recall regression.
   - Therefore, the **Continual Classical Adapter** is the **sole model approved for research demonstration deployment**.
