# FINAL RESEARCH QUESTIONS & HYPOTHESIS LEDGER
## Complete Evidence-Based Research Ledger Across All Completed Phases

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Governing Standard:** Pre-registered Protocol & Multi-Metric Safety Evaluation  
**Status:** CANONICALLY FROZEN & VERIFIED  

---

## 1. Summary Ledger Table

| RQ ID | Research Question | Phase | Intervention | Control Baseline | Primary Metric | Empirical Result | Statistical Evidence | Safety Status | Final Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RQ7.1** | Can continual personalization adapt to atypical speech without catastrophic forgetting? | 7.4 / 7.5 | Balanced Class Replay Memory (Capacity: 100) | Natural sequential adaptation (no replay buffer) | Forgetting Delta ($\Delta \le 0.015$) & Macro-F1 | Macro-F1 increased (+0.0460); FER reduced (-0.3509); Forgetting bounded at $\Delta = -0.0150$. | Paired t-test $p < 0.0001$; FDR corrected $q < 0.0001$. | **SAFE** | **SUPPORTED** |
| **RQ7.2** | Can calibrated confidence reliably route ambiguous utterances to human review and abstention? | 7.2 / 7.3 | Temperature scaling calibration & entropy thresholding | Uncalibrated raw softmax argmax | Expected Calibration Error (ECE) & Abstention Accuracy | Degraded acoustic speech ($<0.30$) and low confidence ($<0.60$) 100% routed to human verification/abstention. | ECE reduced from 0.28 to 0.04; 0 unverified leaks across 60 evaluation sessions. | **SAFE** | **SUPPORTED** |
| **RQ7.3** | Does continual fixed quantum kernel personalization outperform matched classical personalization? | 7.8 / 7.12 | Fixed Q8-R1 QSVC ($C=1.0$) with precomputed quantum fidelity kernel | Matched Continual Classical Adapter (25,223 params) | Emergency Recall & Macro-F1 | Fixed quantum emergency recall collapsed to **5.10%** (vs **16.20%** classical). | Paired t-test $p_{\text{adj}} = 1.34 \times 10^{-6}$ (Classical decisively superior on Emergency Recall). | **UNSAFE** (Veto Triggered) | **NOT SUPPORTED** |
| **RQ7.4** | Does adaptive variational quantum representation outperform fixed quantum kernels in continual learning? | 7.9 / 7.12 | Trainable Variational Quantum Classifier (16 Adam rotation angles) | Fixed Q8-R1 QSVC ($C=1.0$) | Macro-F1 & Emergency Recall | Macro-F1: 0.1187 vs 0.1180; Emergency Recall: **5.81%** vs **5.10%**. | Paired t-test $p_{\text{adj}} = 0.061$ (NS for Macro-F1); $p_{\text{adj}} = 0.559$ (NS for Recall). | **UNSAFE** (Veto Triggered) | **NOT SUPPORTED** |
| **RQ7.5** | Does hybrid classical + quantum fusion outperform both standalone baselines? | 7.10 / 7.12 | Convex probability fusion ($\alpha=0.5$ between Adapter and Quantum VQC) | Standalone Classical Adapter and Standalone Quantum VQC | Macro-F1, False Emergency Rate, Emergency Recall | Higher Macro-F1 (0.1184 vs 0.1012, $p < 10^{-6}$) and lower FER (19.45% vs 43.53%, $p < 10^{-8}$) than Classical; but Emergency Recall collapsed (**7.07%** vs **16.20%**, $p < 10^{-3}$). Did not beat pure Quantum on Macro-F1 ($p=0.275$). | Statistically significant F1 and FER gains over Classical; statistically significant recall loss. | **BORDERLINE** (Conditional) | **PARTIALLY SUPPORTED** |
| **RQ7.6** | Can an autonomous continual self-improvement loop operate safely without unverified learning? | 7.11 | Mandatory Human-in-the-Loop Verification Barrier (`ContinualSelfImprovementEngine`) | Autonomous pseudo-labeling / unverified self-training | Unverified experience leak count & candidate governance decision rate | 0 unverified experiences admitted to training; 4 unsafe candidates vetoed; 56 held; 0 regressions deployed. | 100% verification barrier compliance across 60 longitudinal cycles. | **SAFE** | **SUPPORTED** |
| **RQ7.7** | Can multi-metric safety governance prevent silent degradation in atypical emergency communication? | 7.6 / 7.13 / 7.14 | Phase 7.6 Multi-Metric Validation Gate and Unconditional Safety Veto | Single-metric (Macro-F1 only) promotion policy | Interception rate under adversarial and natural distribution shifts | 100% of candidate regressions, degraded acoustic inputs, and adversarial injections intercepted. | 48/48 stress tests safely handled via abstention; 6/6 failure injection attacks blocked. | **SAFE** | **SUPPORTED** |

---

## 2. Definitive Scientific Interpretation

1. **Confirmation of Historical Case B Across Continual Learning:**
   - In static Phase 4/5/6 benchmarks, fixed quantum fidelity kernels demonstrated no measurable advantage over classical RBF-SVM or classical Adapter baselines.
   - In Phase 7 continual learning, both fixed quantum kernels (Phase 7.8) and trainable variational quantum representations (Phase 7.9) fail to provide any advantage in life-critical Emergency Recall.
   - Quantum kernel methods suffer from persistent soft-margin collapse toward majority non-emergency speech under the severe class imbalance characteristic of real-world atypical communication datasets.

2. **The Clinical Danger of Macro-F1 Optimization Alone:**
   - Pure quantum and hybrid models achieve higher aggregate Macro-F1 (~0.118 vs ~0.101) and lower False Emergency Rates (~17-19% vs ~44%) than the Classical Adapter.
   - However, they achieve this by suppressing emergency predictions across the board, missing over 92% to 94% of genuine emergencies.
   - For an assistive emergency system, an unrecognized life-threatening event (false negative) is catastrophic. Therefore, classical personalization is scientifically and clinically superior for emergency capture.
