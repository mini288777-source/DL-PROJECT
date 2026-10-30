# FINAL PROJECT PLAN & LIFECYCLE ROADMAP
## Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication

---

### Project Metadata
- **Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
- **Research Team:** `kernal`
- **Experimental Status:** **EXPERIMENTAL RESEARCH COMPLETE & FROZEN**
- **Repository Verification:** 208 / 208 Tests Passing (100.0% Green)
- **Primary Quantum Decision:** **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE**
- **Operational Status:**
  - **DEPLOYMENT READINESS:** **NOT READY** (Experimental research-stage framework only)
  - **CLINICAL VALIDATION:** **NOT PERFORMED** (No clinical or bedside trials conducted)
  - **HARDWARE EXECUTION:** **NOT TESTED** (All quantum operations evaluated via exact classical statevector simulation)
  - **REAL-WORLD RELIABILITY:** **NOT ESTABLISHED** (TORGO dysarthric cohort evaluation only)

---

## 1. Experimental Research Lifecycle (Phases 1 – 6.5.6: COMPLETE)

```
[Phase 1: Foundation] -> [Phase 2: Whisper Embeddings] -> [Phase 3: Classical Baselines]
                                                                    |
[Phase 6: Statistics] <- [Phase 5: Noise Sweep]        <- [Phase 4: Quantum Baseline]
        |
        v
[Phase 6.5.0: Forensic Diagnostic of Collapse]
        |
        v
[Phase 6.5.1: Stratified Few-Shot Splits] -> [Phase 6.5.2: Classical Cost-Sensitive Rescue]
                                                                    |
[Phase 6.5.5: Quantum Recovery & Closure] <- [Phase 6.5.4: PCA Audit] <- [Phase 6.5.3: Hierarchical Router]
        |
        v
[Phase 6.5.6: Final Evidence Synthesis & Claim Harmonization] (COMPLETE)
```

### Phase 1: Foundation, Environment & Audio Verification
- **Objective:** Curate the 15-speaker TORGO atypical speech corpus (8 dysarthric, 7 control), verify 6,067 recordings at 16 kHz mono, and establish directory contracts.
- **Status:** **COMPLETE & FROZEN**.

### Phase 2: Feature Representation & Bottleneck Embeddings
- **Objective:** Extract frozen Whisper-tiny acoustic embeddings (384-dimensional bottleneck representations) with unit-L2 normalization.
- **Status:** **COMPLETE & FROZEN**.

### Phase 3: Classical Few-Shot Baselines (Track 1)
- **Objective:** Evaluate flat 7-class classical classifiers (Linear SVM, RBF-SVM, Adapter MLP, KNN) across few-shot calibration budgets $N \in \{20, 30, 40, 50\}$.
- **Finding:** Observed universal majority-class collapse to `GENERAL_SPEECH` due to extreme class imbalance (~95.10% negative) and unconstrained calibration sampling.
- **Status:** **COMPLETE & FROZEN**.

### Phase 4: Quantum Kernel Baseline (Track 1)
- **Objective:** Evaluate 4-qubit and 8-qubit QSVC classifiers with `ZZFeatureMap` (linear entanglement, reps=1, exact statevector simulation).
- **Finding:** Quantum model exhibited identical collapse behavior to classical baselines ($\text{Macro-F1} \approx 0.10$, $\text{Emergency Recall} = 0.00\%$).
- **Status:** **COMPLETE & FROZEN**.

### Phase 5: Controlled Synthetic Noise Perturbation
- **Objective:** Subject clean audio embeddings to controlled synthetic additive white Gaussian noise (AWGN) at 15, 10, and 5 dB SNR, plus clean baseline (720 experimental cells).
- **Finding:** Performance degraded monotonically below 10 dB SNR; classical and quantum models exhibited matched degradation slopes.
- **Status:** **COMPLETE & FROZEN**.

### Phase 6: Factorial Statistical Analysis & Longitudinal Audit
- **Objective:** Perform formal hypothesis testing (Holm-Bonferroni correction, Cohen's $d$, Wilcoxon signed-rank tests) across Phase 3–5 results.
- **Finding:** Confirmed that under the flat formulation, classical vs. quantum model differences were statistically non-significant ($p > 0.05$).
- **Status:** **COMPLETE & FROZEN**.

---

## 2. Recovery Track (Phase 6.5: COMPLETE)

### Subphase 6.5.0: Forensic Root-Cause Diagnostic
- **Discovery:** Proved that the Phase 3–4 collapse was consistent with severe minority-class support limitations (0–1 emergency samples in 85%+ of calibration splits) and the flat classification formulation, and was shared equally by classical and quantum models.
- **Status:** **COMPLETE & FROZEN**.

### Subphase 6.5.1: Speaker-Stratified Prompt-Isolated Protocol (Track B)
- **Objective:** Generate reproducible few-shot splits guaranteeing $\ge 2$ calibration instances per feasible emergency intent across budgets $N \in \{20, 30, 40, 50\}$.
- **Status:** **COMPLETE & FROZEN**.

### Subphase 6.5.2 & 6.5.2-C: Classical Rescue & Cost-Sensitive Learning
- **Objective:** Rescue models from majority collapse via cost-sensitive class-weighted loss and balanced sampling.
- **Finding:** Emergency recall recovered ($31.89\%$ for RBF, $45.81\%$ for Adapter), but flat formulation incurred catastrophic False-Emergency Rates ($96.82\%$ for RBF at $N=30$).
- **Status:** **COMPLETE & FROZEN**.

### Subphase 6.5.3 & 6.5.3-C: Two-Stage Hierarchical Intent Recognition
- **Architecture:** Stage 1 Binary Emergency Gate (`GENERAL_SPEECH` vs `EMERGENCY`) + Stage 2 Conditional Subtype Router.
- **Finding:** Reduced RBF-SVM False-Emergency Rate from $96.82\%$ to $15.06\%$ at $N=30$ ($-81.76\text{ pp}$ reduction), establishing the structural necessity of hierarchical gating.
- **Status:** **COMPLETE & FROZEN**.

### Subphase 6.5.4: Representation Space & PCA Dimensionality Audit
- **Finding:** Identified that $d \le 4$ created a Stage-2 subtype compression bottleneck; $d=8$ relieved the bottleneck while retaining $89.73\% \pm 2.99\%$ cumulative explained variance (forensically superseding the historical $27.4\%$ plotting placeholder). Selected $d=8$ as primary representation.
- **Status:** **COMPLETE & FROZEN**.

### Subphase 6.5.5 & 6.5.5-C: Definitive Matched Classical vs. Quantum Evaluation
- **Evaluation:** Matched 60 conditions ($15\text{ speakers} \times 4\text{ budgets}$) comparing QSVC Q8-R1 vs. Matched Classical RBF-SVM ($d=8$).
- **Primary Finding:** Across Macro-F1 ($\Delta = -0.0036$, Holm $p=0.3428$), Emergency Recall ($\Delta = -4.37\text{ pp}$, Holm $p=0.0689$), FER ($\Delta = -1.20\text{ pp}$, Holm $p=0.9505$), and IPER ($\Delta = -3.44\text{ pp}$, Holm $p=0.0582$), all paired differences are non-significant.
- **Primary Quantum Decision:** **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE**.
- **Status:** **COMPLETE & FROZEN**.

### Subphase 6.5.6: Final Evidence Synthesis, Metric Reconciliation & Claim Harmonization
- **Deliverable:** Longitudinal artifact provenance inventory, metric reconciliation, and de-biased scientific claim auditing.
- **Status:** **COMPLETE & FROZEN**.

---

## 3. Phase 7: Final Research Deliverables (ACTIVE & UPCOMING)

No additional scientific experiments, model retraining, or data collection will be performed. Phase 7 focuses exclusively on presentation, reporting, academic defense, and dissemination.

### Subphase 7.1: Final Scientific Report & Manuscript Sections
- Complete, peer-review-ready technical report detailing the dual-track methodology, forensic diagnostics, and Case B findings.

### Subphase 7.2: Review 2 Presentation Slide Deck (PPT)
- Visual presentation deck structured for the doctoral/academic review committee, clearly communicating the valid null result and architectural findings.

### Subphase 7.3: Publication-Quality Architecture Diagrams
- High-resolution SVG/PNG diagrams of the Two-Stage Hierarchical Router, Whisper embedding extraction, and ZZFeatureMap quantum kernel circuit.

### Subphase 7.4: Methodological Flowcharts
- Step-by-step decision trees and evidence chain workflows illustrating Track 1 vs. Track 2 protocol separation.

### Subphase 7.5: Master Results & Reconciliation Tables
- Machine-readable and formatted tables covering matched comparisons, ablations, runtime overhead, and metric reconciliations.

### Subphase 7.6: Defense & Review Q&A Preparation Dossier
- Comprehensive defense preparation guide addressing anticipated hostile and skeptical review questions regarding quantum advantage, few-shot bounds, and clinical safety.

### Subphase 7.7: Open Reproducibility Package
- Sealed, deterministic package with single-command verification scripts and seed-locked execution logs.

### Subphase 7.8: Final Repository Integrity Audit & Tagging
- Complete file manifest audit, clean-up of temporary scratch files, and git tagging for archival permanence.

---

## 4. Operational & Epistemic Commitments

1. **Null-Result Integrity:** The project embraces Case B as an authoritative, scientifically valuable result. No post-hoc manipulation or hyperparameter fishing will be conducted to manufacture a synthetic quantum advantage.
2. **Strict Protocol Separation:** Track 1 (Original Benchmark) and Track 2 (Recovery Track) are distinct experimental protocols and will never be conflated as a single continuous trial.
3. **Safety Disclaimers:** All publications, slides, and reports must prominently display:
   - `DEPLOYMENT READINESS: NOT READY`
   - `CLINICAL VALIDATION: NOT PERFORMED`
   - `HARDWARE EXECUTION: NOT TESTED (Exact Statevector Simulation Only)`
   - `REAL-WORLD RELIABILITY: NOT ESTABLISHED`
