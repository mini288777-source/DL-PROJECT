# Phase 6.5 — Emergency Recognition Recovery Plan
## Canonical Master Specification (Subphases 6.5.1 – 6.5.8)

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Parent Track:** Phase 6.5 — Emergency Recognition Recovery  
**Baseline Status:** Phases 3–6 Frozen and Intact in `results/phase5/` and `phase6/`  
**Execution Mode:** Sequential Phase-Gated Recovery  

---

## Executive Summary & Guiding Scientific Principles

Phase 6.5.0 established conclusively that the primary root cause of the benchmark collapse (0.000 emergency recall and 0.000 IPER across all kernel models) is **severe adaptation minority-class absence**:
1. At $N \in \{20, 30, 40, 50\}$, random adaptation subsets contain between 0 and 3 emergency samples across all 6 emergency classes combined (with `SOS_HELP` having zero support in 59/60 cells and `LOCATION` in 60/60 cells).
2. Unweighted soft-margin C-SVM ($C=1.0$) trivially assigns minority samples to slack variables $\xi_i=1$, producing constant majority-class predictors regardless of kernel geometry.
3. The original direct 7-class few-shot formulation is mathematically and empirically **NOT FEASIBLE** under unstratified random sampling.

The recovery plan outlined below enforces strict phase gates. **No step jumps ahead.** The objective is **not** to manufacture an artificial quantum advantage, but to restore genuine discriminative capacity under a fair protocol and evaluate whether quantum kernels offer any statistically verifiable advantage over well-tuned classical baselines.

---

## Master Architecture of Recovery Track (6.5.1 – 6.5.8)

```
Subphase 6.5.0: Forensic Diagnostic & Root-Cause Audit [COMPLETED]
       │
       ▼
Subphase 6.5.1: Stratified Split Protocol & Data Feasibility Engineering
       │
       ▼
Subphase 6.5.2: Classical Rescue Baseline (Class-Weighted & Resampled SVM)
       │
       ▼
Subphase 6.5.3: Hierarchical Intent Formulation (Binary Detector + Subtype Router)
       │
       ▼
Subphase 6.5.4: Representation Space Ablation & PCA Geometry Audit
       │
       ▼
Subphase 6.5.5: Quantum Kernel Optimization & Expressivity Tuning
       │
       ▼
Subphase 6.5.6: Head-to-Head Rigorous Statistical Benchmark (Fair Comparison)
       │
       ▼
Subphase 6.5.7: Environmental Robustness Re-evaluation (Protocol B Noise Sweep)
       │
       ▼
Subphase 6.5.8: Phase 6.5 Synthesis, Final Reports & Canonical Closure
```

---

## Detailed Specifications per Subphase

### Subphase 6.5.1 — Stratified Split Protocol & Data Feasibility Engineering

- **Primary Research Question:** Can we construct speaker-specific adaptation splits with guaranteed emergency-class support without causing data leakage or violating speech independence?
- **Objectives:**
  1. Audit full speaker audio pools in TORGO to determine exact counts per emergency class per speaker.
  2. Implement an adaptive stratification algorithm:
     - For each speaker, guarantee a minimum of $k \ge 1$ (target $k=2$ or $3$ where corpus permits) adaptation instances for every available emergency class.
     - Isolate held-out test sets preserving all true speaker emergency test utterances (disjoint session/recording IDs).
  3. Formalize a multi-tier budget:
     - **Tier 1 (Uniform Few-Shot):** Balanced subset (e.g., $N=20, 30, 40, 50$ with controlled minority representation).
     - **Tier 2 (Emergency Enriched):** Fixed ratio calibration sets (e.g., 50% Emergency, 50% General Speech).
  4. Ensure strict backward isolation: store all splits in `phase6_5/splits/torgo_stratified_splits.json`. Do **not** overwrite `splits/torgo_splits.json`.
- **Inputs:** `data/manifests/torgo_intent_mapping.csv`, `data/manifests/clean_manifest.csv`, `splits/torgo_splits.json`.
- **Outputs:**
  - `phase6_5/splits/torgo_stratified_splits.json`
  - `phase6_5/reports/subphase_6_5_1_split_audit.md`
  - `phase6_5/analysis/stratified_class_support.csv`
- **Acceptance Gate for 6.5.1:**
  - Emergency class support $>0$ for at least 4 emergency classes in $\ge 80\%$ of adaptation cells where corpus allows.
  - Zero leakage between adaptation and test sets verified by recording ID hash collision checks.

---

### Subphase 6.5.2 — Classical Rescue Baseline (Class-Weighted & Resampled SVM)

- **Primary Research Question:** Does addressing data imbalance and SVM soft-margin cost asymmetry recover non-trivial emergency recognition using purely classical models?
- **Objectives:**
  1. Train RBF-SVM using `class_weight="balanced"` on the new stratified splits across all 15 speakers at $N \in \{20, 30, 40, 50\}$.
  2. Implement cost-sensitive regularization grid search: $C \in \{0.1, 1.0, 10.0, 100.0\}$.
  3. Test classical minority oversampling/SMOTE on Whisper 384-D embeddings within the adaptation set.
  4. Measure emergency recall, emergency precision, Macro-F1, and IPER.
- **Inputs:** Stratified splits from 6.5.1, Whisper embeddings from `data/embeddings/whisper_clean/`.
- **Outputs:**
  - `phase6_5/results/classical_rescue_results.csv`
  - `phase6_5/results/classical_rescue_predictions.csv`
  - `phase6_5/reports/subphase_6_5_2_classical_rescue.md`
- **Acceptance Gate for 6.5.2:**
  - Classical RBF-SVM Emergency Recall $> 0.15$ on held-out test sets.
  - IPER $> 10\%$ for at least one budget $N$.
  - Prediction diversity verified (GENERAL_SPEECH prediction rate $< 95\%$).

---

### Subphase 6.5.3 — Hierarchical Intent Formulation (Binary Detector + Subtype Router)

- **Primary Research Question:** Does decoupling the emergency-vs-general detection from fine-grained emergency subtype classification improve decision boundaries under extreme few-shot constraints?
- **Architecture:**
  - **Stage 1 (Binary Gate):** Classify utterance as `EMERGENCY` vs `GENERAL_SPEECH`. (Aggregates all 6 emergency classes into a single positive class, raising adaptation positive count from 1–2 up to 6–12).
  - **Stage 2 (Subtype Router):** Conditional 6-class classifier activated only when Stage 1 flags `EMERGENCY`.
- **Methodology:**
  1. Train Stage 1 binary RBF-SVM and threshold-tuned classifier.
  2. Train Stage 2 multi-class classifier using emergency-only support.
  3. Compare End-to-End Hierarchical Macro-F1 vs Flat 7-Class Classical Rescue.
- **Outputs:**
  - `phase6_5/results/hierarchical_baseline_results.csv`
  - `phase6_5/reports/subphase_6_5_3_hierarchical_evaluation.md`
- **Acceptance Gate for 6.5.3:**
  - Stage 1 Emergency Detection Recall $\ge 0.50$.
  - End-to-End IPER statistically superior to flat 7-class baseline ($p < 0.05$).

---

### Subphase 6.5.4 — Representation Space Ablation & PCA Geometry Audit

- **Primary Research Question:** Where does discriminative emergency information live in Whisper latent space, and how much is lost across PCA dimensions $d \in \{2, 4, 6, 8, 16, 32, 64, 384\}$?
- **Objectives:**
  1. Compute Fisher Discriminant Ratio (FDR) and Davies-Bouldin index for emergency vs general speech across latent dimensions.
  2. Train linear and kernel probes across dimensions $d \in \{2, 4, 6, 8, 16, 32, 64, 384\}$.
  3. Test whether supervised dimensionality reduction (LDA, Supervised UMAP, or Linear Projection) preserves emergency separation better than unsupervised PCA for low-qubit quantum inputs ($d=4, 6, 8$).
- **Outputs:**
  - `phase6_5/analysis/representation_geometry_metrics.csv`
  - `phase6_5/reports/subphase_6_5_4_representation_ablation.md`
- **Acceptance Gate for 6.5.4:**
  - Definitive dimension vs emergency-retention tradeoff curve established.
  - Optimal input projection method selected for Quantum Kernel evaluation in 6.5.5.

---

### Subphase 6.5.5 — Quantum Kernel Optimization & Expressivity Tuning

- **Primary Research Question:** Can a properly regularized, class-weighted Quantum Support Vector Classifier (QSVC) with an optimized feature map learn emergency decision boundaries on stratified few-shot data?
- **Parameter Sweep & Hypotheses:**
  1. **Class Weighting:** Enforce `class_weight="balanced"` in QSVC QP solver.
  2. **Qubit Allocation:** Evaluate $q=4, 6, 8$ matching top representations from 6.5.4.
  3. **Feature Map Architectures:**
     - Pauli Feature Map ($ZZ$ vs $Z$ vs Chebyshev / Haar-like rotations).
     - Entanglement topologies: `linear`, `circular`, `full`.
     - Repetitions: `reps` $\in \{1, 2, 3\}$.
  4. **Kernel Regularization:** Cost parameter $C \in \{0.1, 1.0, 10.0, 100.0\}$.
  5. **Metric Tracking:** Kernel target alignment (KTA), kernel matrix condition number, training margin width, test emergency recall.
- **Outputs:**
  - `phase6_5/results/quantum_optimization_results.csv`
  - `phase6_5/reports/subphase_6_5_5_quantum_tuning.md`
- **Acceptance Gate for 6.5.5:**
  - QSVC avoids trivial collapse (non-zero emergency predictions).
  - QSVC training convergence with finite non-trivial support vectors verified.

---

### Subphase 6.5.6 — Head-to-Head Rigorous Statistical Benchmark (Fair Comparison)

- **Primary Research Question:** Under identical data, splits, pre-processing, and class-weighting, does QSVC demonstrate an empirical advantage over RBF-SVM or Classical Adapter?
- **Experimental Protocol:**
  - Fully paired design across all 15 speakers, 4 budgets ($N=20, 30, 40, 50$), and 3 seeds where applicable.
  - Primary Endpoints:
    - Emergency Macro-F1 ($F1_{\text{emerg}}$)
    - Intent Preservation Rate (IPER)
    - Mean Emergency Recall
    - Overall Balanced Accuracy
  - Statistical Validation:
    - Paired Wilcoxon Signed-Rank Test across all matched speaker-budget cells.
    - Hodges-Lehmann median effect size estimator.
    - Holm-Bonferroni Family-Wise Error Rate (FWER) correction across hypotheses.
- **Possible Scientific Verdicts:**
  - **Verdict A (Quantum Advantage):** QSVC significantly outperforms RBF-SVM ($p < 0.05$, positive effect size).
  - **Verdict B (Classical Parity/Superiority):** RBF-SVM matches or exceeds QSVC; both recover emergency recognition.
  - **Verdict C (Classical Only):** RBF-SVM recovers emergency recognition, but QSVC fails or suffers from barren/flat geometry.
  - **Verdict D (Persistent Intractability):** Neither model achieves acceptable emergency recognition without external training data.
- **Outputs:**
  - `phase6_5/results/phase6_5_benchmark_results.csv`
  - `phase6_5/analysis/phase6_5_statistical_tests.csv`
  - `phase6_5/reports/subphase_6_5_6_fair_comparison.md`

---

### Subphase 6.5.7 — Environmental Robustness Re-evaluation (Protocol B Noise Sweep)

- **Condition for Execution:** Triggered **only** if at least one model demonstrates meaningful emergency recovery (Mean Emergency Recall $\ge 0.15$ in 6.5.6). If no model succeeds on clean data, noise sweeps are cancelled to prevent meaningless compute expenditure.
- **Methodology:**
  - Evaluate recovered models under SNR levels: $\{+20\text{ dB}, +10\text{ dB}, 0\text{ dB}, -5\text{ dB}\}$.
  - Noise conditions: Babble and Emergency Vehicle Sirens.
  - Compute Robustness Degradation Index (RDI) and Noise Graceful Degradation Slope ($\Delta F1 / \Delta\text{SNR}$).
- **Outputs:**
  - `phase6_5/results/noise_robustness_recovered.csv`
  - `phase6_5/reports/subphase_6_5_7_noise_robustness.md`

---

### Subphase 6.5.8 — Phase 6.5 Synthesis, Final Reports & Canonical Closure

- **Deliverables:**
  1. Master Phase 6.5 Final Synthesis Report (`phase6_5/reports/PHASE6_5_FINAL_SYNTHESIS_REPORT.md`).
  2. Publication-ready comparison figures (Clean F1 recovery, IPER delta, Confusion matrices).
  3. Formal update to repository `README.md` and Paper Draft outlining the Phase 6.5 discoveries.
  4. Final reproducibility hash freeze of all Phase 6.5 code, configs, and seeds.

---

## Strict Constraints & Prohibitions

1. **Frozen Baseline Protection:** Under no circumstances shall files in `results/phase5/` or `phase6/` be modified, renamed, or overwritten.
2. **Sequential Phase Gating:** Subphases must be executed strictly in order (6.5.1 $\rightarrow$ 6.5.2 $\rightarrow$ ... $\rightarrow$ 6.5.8). No model training or tuning may occur before 6.5.1 splits are validated.
3. **Honest Reporting:** No hyperparameter cherry-picking or metric masking. If quantum models do not outperform classical baselines, this must be documented clearly as an empirical scientific contribution.
4. **Reproducibility Guarantee:** All code must run deterministically via fixed seeds in the virtual environment.

---
*Signed and Approved for Execution: ML Research & Reproducibility Audit Team*
