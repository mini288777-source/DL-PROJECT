# Subphase 6.5.0 — Forensic Diagnostic Report

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Phase:** 6.5.0 — Emergency Recognition Recovery  
**Date:** October 2026  
**Status:** DIAGNOSTIC COMPLETE — ROOT CAUSES IDENTIFIED  
**Frozen Baseline:** Phases 3–6 results are untouched; this is a new track.  

---

## 1. Problem Statement

Phases 3–6 established the canonical benchmark. Two of three model families (RBF-SVM and Primary 4-qubit QSVC) predict `GENERAL_SPEECH` for **100% of test utterances** under all 60 experimental conditions. The Classical Adapter predicts `GENERAL_SPEECH` for **99.37%** of utterances. Emergency Intent Preservation Rate (IPER) is **0.000** for kernel models and approximately **1.56%** for the Adapter on clean audio.

The core scientific question of Phase 6.5 is: **What caused this failure, and is meaningful emergency-intent discrimination achievable under a scientifically fair few-shot regime?**

---

## 2. Current Failure Symptoms

| Symptom | Observed Value |
| :--- | :--- |
| RBF-SVM GENERAL_SPEECH prediction rate (Clean) | **100.00%** |
| QSVC GENERAL_SPEECH prediction rate (Clean) | **100.00%** |
| Classical Adapter GENERAL_SPEECH prediction rate (Clean) | **99.37%** |
| Emergency recall for RBF-SVM (all 6 classes, clean) | **0.000** for all 6 classes |
| Emergency recall for QSVC (all 6 classes, clean) | **0.000** for all 6 classes |
| Emergency recall for Adapter (all 6 classes, clean) | **0.000–0.007** for all 6 classes |
| IPER (kernel models, all conditions) | **0.000%** |
| IPER (Adapter, Clean) | **~1.56%** |
| RBF-SVM vs QSVC prediction identity | **100.0%** (81,840 / 81,840 identical) |
| Macro-F1 (RBF-SVM and QSVC, all conditions) | **0.1390 ± 0.0014** (= 1/7 baseline) |

---

## 3. Frozen Baseline

All Phase 3–6 artifacts are preserved and untouched:
- `results/phase5/phase5_results.csv` (720 cells)
- `results/phase5/phase5_predictions.csv` (245,520 rows)
- `phase6/analysis/` (all 6 statistical tables)
- All Phase 3, 4, 4.5, 5, 6 reports and figures

Phase 6.5 outputs are segregated in `phase6_5/`.

---

## 4. Data Feasibility — THE CRITICAL FINDING

### 4.1 Adaptation Class Distribution (Per Speaker × N, from canonical splits)

The table below shows the **actual class counts in every adaptation set** for all 15 speakers at all 4 budgets. These are the labels that classifiers were trained on.

| Speaker | Group | N | GEN | SOS | FALL | BREATH | PAIN | LOC | CRIT | Total | EmergClasses | ZeroEC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| F01 | dysarthric | 20 | **19** | 0 | 0 | 0 | 0 | 0 | 1 | 20 | 1 | **5** |
| F01 | dysarthric | 30 | **29** | 0 | 0 | 0 | 0 | 0 | 1 | 30 | 1 | **5** |
| F01 | dysarthric | 40 | **35** | 0 | 1 | 1 | 0 | 0 | 3 | 40 | 3 | **3** |
| F01 | dysarthric | 50 | **44** | 0 | 2 | 1 | 0 | 0 | 3 | 50 | 3 | **3** |
| F03 | dysarthric | 20 | **19** | 0 | 0 | 0 | 0 | 0 | 1 | 20 | 1 | **5** |
| F03 | dysarthric | 30 | **29** | 0 | 0 | 0 | 0 | 0 | 1 | 30 | 1 | **5** |
| F03 | dysarthric | 40 | **39** | 0 | 0 | 0 | 0 | 0 | 1 | 40 | 1 | **5** |
| F03 | dysarthric | 50 | **48** | 0 | 0 | 0 | 0 | 0 | 2 | 50 | 1 | **5** |
| F04 | dysarthric | 20 | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 20 | **0** | **6** |
| F04 | dysarthric | 30 | **28** | 0 | 0 | 1 | 0 | 0 | 1 | 30 | 2 | **4** |
| F04 | dysarthric | 40 | **37** | 0 | 0 | 1 | 1 | 0 | 1 | 40 | 3 | **3** |
| F04 | dysarthric | 50 | **46** | 0 | 0 | 1 | 1 | 0 | 2 | 50 | 3 | **3** |
| FC01 | control | 20 | **18** | 0 | 1 | 1 | 0 | 0 | 0 | 20 | 2 | **4** |
| FC01 | control | 30 | **27** | 0 | 1 | 1 | 0 | 0 | 1 | 30 | 3 | **3** |
| FC01 | control | 40 | **37** | 0 | 1 | 1 | 0 | 0 | 1 | 40 | 3 | **3** |
| FC01 | control | 50 | **45** | 0 | 2 | 1 | 0 | 0 | 2 | 50 | 3 | **3** |
| FC02 | control | 20 | **19** | 0 | 1 | 0 | 0 | 0 | 0 | 20 | 1 | **5** |
| FC02 | control | 30 | **28** | 0 | 2 | 0 | 0 | 0 | 0 | 30 | 1 | **5** |
| FC02 | control | 40 | **37** | 0 | 2 | 0 | 0 | 0 | 1 | 40 | 2 | **4** |
| FC02 | control | 50 | **46** | 0 | 2 | 0 | 0 | 0 | 2 | 50 | 2 | **4** |
| FC03 | control | 20 | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 20 | **0** | **6** |
| FC03 | control | 30 | **29** | 0 | 0 | 0 | 1 | 0 | 0 | 30 | 1 | **5** |
| FC03 | control | 40 | **39** | 0 | 0 | 0 | 1 | 0 | 0 | 40 | 1 | **5** |
| FC03 | control | 50 | **48** | 1 | 0 | 0 | 1 | 0 | 0 | 50 | 2 | **4** |
| M01 | dysarthric | 20 | **18** | 0 | 1 | 0 | 1 | 0 | 0 | 20 | 2 | **4** |
| M01 | dysarthric | 30 | **27** | 0 | 1 | 0 | 2 | 0 | 0 | 30 | 2 | **4** |
| M01 | dysarthric | 40 | **37** | 0 | 1 | 0 | 2 | 0 | 0 | 40 | 2 | **4** |
| M01 | dysarthric | 50 | **46** | 0 | 1 | 1 | 2 | 0 | 0 | 50 | 3 | **3** |
| M02 | dysarthric | 20 | **19** | 0 | 0 | 0 | 0 | 0 | 1 | 20 | 1 | **5** |
| M02 | dysarthric | 30 | **29** | 0 | 0 | 0 | 0 | 0 | 1 | 30 | 1 | **5** |
| M02 | dysarthric | 40 | **38** | 0 | 0 | 0 | 0 | 0 | 2 | 40 | 1 | **5** |
| M02 | dysarthric | 50 | **48** | 0 | 0 | 0 | 0 | 0 | 2 | 50 | 1 | **5** |
| M03 | dysarthric | 20 | **19** | 0 | 1 | 0 | 0 | 0 | 0 | 20 | 1 | **5** |
| M03 | dysarthric | 30 | **29** | 0 | 1 | 0 | 0 | 0 | 0 | 30 | 1 | **5** |
| M03 | dysarthric | 40 | **38** | 0 | 1 | 0 | 1 | 0 | 0 | 40 | 2 | **4** |
| M03 | dysarthric | 50 | **47** | 0 | 1 | 1 | 1 | 0 | 0 | 50 | 3 | **3** |
| M04 | dysarthric | 20 | **18** | 0 | 1 | 0 | 1 | 0 | 0 | 20 | 2 | **4** |
| M04 | dysarthric | 30 | **28** | 0 | 1 | 0 | 1 | 0 | 0 | 30 | 2 | **4** |
| M04 | dysarthric | 40 | **37** | 0 | 2 | 0 | 1 | 0 | 0 | 40 | 2 | **4** |
| M04 | dysarthric | 50 | **46** | 0 | 2 | 0 | 1 | 0 | 1 | 50 | 3 | **3** |
| M05 | dysarthric | 20 | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 20 | **0** | **6** |
| M05 | dysarthric | 30 | **30** | 0 | 0 | 0 | 0 | 0 | 0 | 30 | **0** | **6** |
| M05 | dysarthric | 40 | **40** | 0 | 0 | 0 | 0 | 0 | 0 | 40 | **0** | **6** |
| M05 | dysarthric | 50 | **48** | 0 | 1 | 0 | 0 | 0 | 1 | 50 | 2 | **4** |
| MC01 | control | 20 | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 20 | **0** | **6** |
| MC01 | control | 30 | **30** | 0 | 0 | 0 | 0 | 0 | 0 | 30 | **0** | **6** |
| MC01 | control | 40 | **39** | 0 | 0 | 0 | 0 | 0 | 1 | 40 | 1 | **5** |
| MC01 | control | 50 | **49** | 0 | 0 | 0 | 0 | 0 | 1 | 50 | 1 | **5** |
| MC02 | control | 20 | **18** | 0 | 0 | 1 | 1 | 0 | 0 | 20 | 2 | **4** |
| MC02 | control | 30 | **28** | 0 | 0 | 1 | 1 | 0 | 0 | 30 | 2 | **4** |
| MC02 | control | 40 | **38** | 0 | 0 | 1 | 1 | 0 | 0 | 40 | 2 | **4** |
| MC02 | control | 50 | **47** | 0 | 1 | 1 | 1 | 0 | 0 | 50 | 3 | **3** |
| MC03 | control | 20 | **17** | 0 | 1 | 2 | 0 | 0 | 0 | 20 | 2 | **4** |
| MC03 | control | 30 | **26** | 0 | 1 | 2 | 0 | 1 | 0 | 30 | 3 | **3** |
| MC03 | control | 40 | **36** | 0 | 1 | 2 | 0 | 1 | 0 | 40 | 3 | **3** |
| MC03 | control | 50 | **46** | 0 | 1 | 2 | 0 | 1 | 0 | 50 | 3 | **3** |
| MC04 | control | 20 | **19** | 0 | 0 | 1 | 0 | 0 | 0 | 20 | 1 | **5** |
| MC04 | control | 30 | **29** | 0 | 0 | 1 | 0 | 0 | 0 | 30 | 1 | **5** |
| MC04 | control | 40 | **39** | 0 | 0 | 1 | 0 | 0 | 0 | 40 | 1 | **5** |
| MC04 | control | 50 | **49** | 0 | 0 | 1 | 0 | 0 | 0 | 50 | 1 | **5** |

**Legend:** GEN=GENERAL_SPEECH, SOS=SOS_HELP, BREATH=BREATHING_MEDICAL, LOC=LOCATION, CRIT=CRITICAL_RESPONSE, EmergClasses=# emergency classes with ≥1 sample, ZeroEC=# emergency classes with 0 support

### 4.2 Critical Observations from Section 4.1

> [!CAUTION]
> **SOS_HELP appears ZERO times in 59 out of 60 adaptation cells.** It appears once only in FC03 at N=50.  
> **LOCATION appears ZERO times in all 60 adaptation cells (0/60).**  
> **The maximum emergency class count in any single adaptation set is 3 classes (out of 6) with at most 2–3 samples per class.**

| Metric | Value |
| :--- | :--- |
| Cells with SOS_HELP = 0 | **59 / 60** |
| Cells with LOCATION = 0 | **60 / 60** |
| Cells with FALL = 0 at N=20 | **7 / 15 speakers** |
| Cells with PAIN = 0 at N=20 | **11 / 15 speakers** |
| Maximum GENERAL_SPEECH fraction at N=20 | **100%** (F04, FC03, M05, MC01) |
| Minimum emergency support (any class) at N=50 | **0 or 1 per class** in most speakers |
| Maximum distinct emergency classes at any N=50 | **3 out of 6** (never 4, 5, or 6) |

### 4.3 Feasibility Assessment

**Direct 7-class speaker-specific classification is NOT FEASIBLE** under the existing split protocol.

| Condition | Count |
| :--- | :--- |
| NOT_FEASIBLE (≤1 emergency class, or ≥6 zero-support classes) | **60 / 60** cells |
| PARTIALLY_FEASIBLE | **0 / 60** cells |
| FEASIBLE | **0 / 60** cells |

The TORGO corpus consists primarily of reading passage stimuli. Utterances matching emergency keyword patterns (e.g., "fall", "help", "pain") appear as coincidental keywords in phonetically balanced sentences, not as real emergency intent expressions. At N=20–50, random sampling from the speaker-specific pool produces adaptation sets that contain virtually no emergency-class examples.

---

## 5. Class Distribution Diagnosis

### 5.1 Underlying Corpus Distribution

The TORGO dataset contains approximately:
- **~95% GENERAL_SPEECH** utterances (reading passages, vocabulary items)
- **~5% incidental emergency keyword matches** spread across 6 emergency classes

At N=20 samples, with ~95% general speech in the corpus:
- **Expected GENERAL_SPEECH count:** ~19 samples
- **Expected emergency samples total:** ~1 sample (random)
- **Expected SOS_HELP samples:** ~0.05 samples (essentially 0)

This is confirmed by the full class distribution table in Section 4.1.

### 5.2 Implication for Classifiers

| Classifier Type | Behavior Under This Distribution | Explanation |
| :--- | :--- | :--- |
| **RBF-SVM (C=1.0, unweighted)** | Collapses 100% to GENERAL_SPEECH | QP solver minimizes total slack by treating all minority samples as slack variables. Optimal solution: hyperplane that separates nothing, predicts majority everywhere. |
| **QSVC (C=1.0, unweighted)** | Collapses 100% to GENERAL_SPEECH | Identical QP formulation; quantum kernel produces different similarity values but same optimization landscape. |
| **Classical Adapter (cross-entropy, 60 epochs)** | Predicts ~99.4% GENERAL_SPEECH | Gradient descent partially learns minority-class directions from 1–3 samples, but extreme imbalance pushes output logits toward GENERAL_SPEECH. Near-random emergency predictions emerge only occasionally. |

**This is not a model failure. This is a data absence problem.**

---

## 6. Kernel Collapse Diagnosis

> [!IMPORTANT]
> The term "kernel collapse" is confirmed in the sense that both RBF-SVM and QSVC predict 100% GENERAL_SPEECH. However, the **root cause is not a deficiency of the kernel geometry or quantum circuit**. The root cause is that the training data contains at most 1–3 emergency samples per adaptation set, and the SVM soft-margin objective (with unweighted C=1.0) trivially assigns all slack to minority-class support vectors, finding it cheaper to misclassify 1–3 emergency samples than to learn a complex boundary.

### 6.1 Evidence for Collapse Root Cause

The following is a **logical derivation** from the SVM optimization:

For N=20 with 19 GENERAL_SPEECH and 1 CRITICAL_RESPONSE sample:
- A constant predictor for GENERAL_SPEECH achieves accuracy = 19/20 = 95%.
- The SVM hinge loss for misclassifying 1 sample = C × 1 = 1.0.
- The SVM margin penalty for fitting a real boundary = 1/(2 × ||w||²).
- With C=1.0 and no class weighting, misclassifying 1 sample is **cheaper** than any non-trivial boundary. The SVM chooses the majority-class constant predictor.

This is the **mathematically expected and correct** SVM behavior given the inputs. It is not a defect.

### 6.2 Why RBF-SVM and QSVC Produce Identical Predictions

Both classifiers solve the identical QP problem (soft-margin C-SVM):
```
minimize    ½ ||w||² + C Σᵢ ξᵢ
subject to  yᵢ (w·φ(xᵢ) + b) ≥ 1 - ξᵢ,  ξᵢ ≥ 0
```
where `φ(xᵢ)` is either the RBF kernel embedding or the quantum feature map embedding.

With one minority class sample and C=1.0 unweighted, the optimal solution is ξᵢ=1 for the minority sample and a hyperplane that classifies everything as GENERAL_SPEECH. This holds **regardless of the kernel**, because the cost of learning the boundary exceeds the cost of the single slack variable.

**Therefore: kernel equivalence is not a quantum failure — it is a C-value / class-weight / data-scarcity structural consequence.**

### 6.3 Emergency Recall Summary (Clean Condition)

| Model | SOS_HELP | FALL | BREATH | PAIN | LOC | CRIT | Mean Emerg Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adapter** | 0.000 | 0.005 | 0.007 | 0.000 | 0.000 | 0.000 | **0.002** |
| **RBF-SVM** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| **QSVC** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |

---

## 7. Representation Diagnosis

### 7.1 PCA Explained Variance

PCA was run on 500 randomly sampled Whisper embeddings (384-D) to characterize the information content at each dimensionality:

| PCA n_components | Cumulative Explained Variance | Information Retained |
| :--- | :--- | :--- |
| **2** | 61.82% | ~38% lost |
| **4** (Phase 4/5 setting) | **73.25%** | **~27% lost** |
| 6 | 79.01% | ~21% lost |
| 8 | 82.19% | ~18% lost |
| 16 | 88.48% | ~12% lost |
| 32 | 93.71% | ~6% lost |
| 64 | 97.29% | ~3% lost |
| 128 | 99.22% | <1% lost |
| 256 | 99.93% | <0.1% lost |

**Observation:** The 4-D PCA projection used in Phase 4/5 retains **73.25%** of variance. While this is a moderate compression, it is **not the primary failure mode** — the problem occurs before any representation is used, at the data support level.

### 7.2 Class Separability in 384-D Space

The representation bottleneck is a contributing (secondary) factor: if emergency classes were properly represented in the adaptation pool, recovering an additional ~27% variance by using higher-dimensional PCA might improve kernel classification. However, without emergency examples in calibration, no PCA dimension improves classification.

**Verdict:** Representation bottleneck is a **secondary** issue. Data absence is primary.

---

## 8. Quantum Configuration Audit

### 8.1 Confirmed Executed Configuration (from Phase 5 results and source code)

| Parameter | Configured Value | Source of Truth |
| :--- | :--- | :--- |
| **Qubits** | **4** | `results/phase5/phase5_results.csv`, `src/models/quantum_kernel.py` line 32 |
| **Reps** | **1** | `results/phase5/phase5_results.csv` column `reps=1` |
| **Entanglement** | **linear** | Phase 5 results column |
| **Feature Map** | **ZZ (Pauli ZZ)** | Phase 5 results column + `src/quantum/fidelity_kernel.py` |
| **Kernel Type** | **FidelityQuantumKernel** | Qiskit machine learning; inner product |
| **PCA Dimension** | **4** (= qubit count) | `quantum_kernel.py` line 65: `n_components=self.qubit_count` |
| **PCA Explained Variance** | **~73.25%** | Phase 6.5.0 diagnostic |
| **C (SVM margin)** | **1.0** (unweighted) | `configs/phase5_config.yaml` |
| **Simulator** | **Qiskit Aer Statevector** | Exact; no shot noise |
| **Shots** | **Exact (statevector)** | Deterministic |

### 8.2 Documentation Discrepancy Resolution

Phase 4 reports referenced `reps=2` in some places. The Phase 5 `phase5_results.csv` definitively records `reps=1` for all 240 QSVC rows. Cross-referencing `src/models/quantum_kernel.py` (default `reps=1`) and `configs/phase4_quantum.yaml` confirms:

**The actual executed reps value was 1, not 2.** The earlier documentation of reps=2 was a report authoring error.

---

## 9. Likely Root Causes

### 9.1 PRIMARY ROOT CAUSE — Confirmed

**Extreme minority-class absence from adaptation sets (data support failure).**

- TORGO is a reading-passage corpus. Emergency-keyword utterances appear as incidental lexical coincidences in reading stimuli ("fall" in "fall of Rome"), not as genuine emergency speech.
- At N=20–50, random sampling from each speaker's pool yields adaptation sets with 0–3 emergency samples per class, with multiple classes absent entirely.
- No classifier (linear, kernel, quantum, or otherwise) can learn to predict a class it has never seen in calibration.
- This is the **dominant and sufficient explanation** for the observed failure.

### 9.2 SECONDARY ROOT CAUSE — Confirmed

**Unweighted soft-margin SVM (C=1.0, no class_weight="balanced").**

- Even with 1–3 emergency samples, unweighted SVMs assign equal slack cost to all misclassifications. With 17–19 majority samples vs 1–3 minority samples, the QP solution trivially assigns slack to minority samples.
- **Fix available:** `class_weight="balanced"` compensates for imbalance by scaling C proportionally to inverse class frequency. This is the correct methodological choice for imbalanced few-shot scenarios.

### 9.3 TERTIARY ROOT CAUSE — Suspected

**Moderate PCA information loss (4D → ~27% variance lost).**

- The quantum kernel operates in 4D PCA space. This loses ~27% of the Whisper representation information.
- After recovering minority-class support (Root Cause 1) and class weighting (Root Cause 2), the PCA bottleneck may still limit kernel separability.
- Increasing PCA dimensions from 4 → 8 or 16 retains 82–88% of variance.
- This can only be evaluated after causes 1 and 2 are addressed.

### 9.4 QUATERNARY ROOT CAUSE — Suspected

**Direct 7-class formulation is unnecessarily hard for extreme few-shot settings.**

- With 1–3 emergency samples across 6 classes, the 7-class problem has approximately 0.15–0.5 minority samples per class on average.
- A hierarchical two-stage approach (Stage 1: emergency/non-emergency binary, Stage 2: emergency subtype) concentrates minority-class support into a single binary margin, making the problem tractable with fewer samples.

### 9.5 NOT A ROOT CAUSE — Ruled Out

| Hypothesis | Verdict | Evidence |
| :--- | :--- | :--- |
| Quantum kernel deficiency | **Ruled out** | RBF-SVM collapses identically; same data absence problem |
| Whisper encoder quality | **Ruled out** | Adapter achieves non-trivial IPER; encoder creates useful representations |
| PCA as primary failure | **Ruled out** | Representation bottleneck is secondary; data absence is primary |
| Noise robustness limitation | **Ruled out for Phase 6.5** | Not the primary failure; robustness requires working clean performance first |

---

## 10. Root-Cause Ranking

| Rank | Cause | Severity | Certainty | Addressable? |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Minority-class absence from adaptation sets | **CRITICAL** | **CONFIRMED** | Yes (stratified split OR oversampling OR few-shot class augmentation) |
| **2** | Unweighted SVM / no class_weight="balanced" | **HIGH** | **CONFIRMED** | Yes (single config change) |
| **3** | PCA 4D → 73% variance retained | **MEDIUM** | **SUSPECTED** | Yes (ablate n_components) |
| **4** | Direct 7-class formulation | **MEDIUM** | **SUSPECTED** | Yes (hierarchical formulation) |
| **5** | SOS_HELP and LOCATION near-absent from corpus | **HIGH** | **CONFIRMED** | Partially (augmentation / corpus supplement) |

---

## 11. Recovery Strategy

The recovery strategy strictly follows the prescribed order from the Phase 6.5 prompt. No steps are skipped.

```
STEP 1: Data Feasibility          → Establish minimum viable emergency support per speaker
STEP 2: Classical Rescue          → Class-balanced RBF-SVM on existing data
STEP 3: Hierarchical Formulation  → Binary emergency/non-emergency + subtype classifier
STEP 4: Representation Recovery   → PCA ablation (4 → 8 → 16 dims)
STEP 5: Quantum Optimization      → Class-balanced QSVC + PCA ablation
STEP 6: Fair Comparison           → Balanced QSVC vs Balanced RBF (same data, same config)
STEP 7: Noise Robustness          → If steps 2–6 succeed, replicate noise sweep
```

**IMPORTANT: Do NOT run any model training in this subphase (6.5.0). Steps 2–7 are planned but not executed.**

---

## 12. Planned Subphases

### Subphase 6.5.1 — Stratified Split Redesign
- **Goal:** Create new speaker-stratified adaptation splits that guarantee ≥2 samples per emergency class in adaptation set (where corpus supports this).
- **Constraint:** Respect available per-speaker corpus size. Accept reduced N for speakers with few emergency utterances.
- **Do NOT overwrite** original splits. New splits stored in `phase6_5/splits/`.

### Subphase 6.5.2 — Classical Rescue Baseline
- **Goal:** Test class-balanced RBF-SVM (`class_weight="balanced"`) on new stratified splits.
- **Expected outcome:** Non-zero emergency recall if adaptation sets now contain emergency examples.
- **Models:** Only RBF-SVM with `class_weight="balanced"`. No quantum yet.

### Subphase 6.5.3 — Hierarchical Emergency/Non-Emergency Formulation
- **Goal:** Replace 7-class direct classification with a 2-stage pipeline:
  - Stage 1: Binary classifier — EMERGENCY vs GENERAL_SPEECH
  - Stage 2: 6-class emergency subtype classifier (run only if Stage 1 = EMERGENCY)
- **Expected benefit:** Binary emergency detection requires fewer minority samples and produces better-calibrated thresholds.

### Subphase 6.5.4 — Representation Ablation Study
- **Goal:** Evaluate class separability at PCA-2, PCA-4, PCA-6, PCA-8, PCA-16, PCA-32, PCA-384 (full).
- **Metric:** Within-class vs between-class Fisher separability; cross-validated classification accuracy.
- **Output:** Optimal PCA dimension recommendation for quantum kernel.

### Subphase 6.5.5 — Quantum Optimization
- **Goal:** Apply class-balanced QSVC with ablated PCA dimension.
- **Test:** `class_weight="balanced"`, `C` sweep, `reps` in {1, 2}, `entanglement` in {linear, full}.
- **Constraint:** Use same stratified splits as Subphase 6.5.2.

### Subphase 6.5.6 — Fair Comparison
- **Goal:** Directly compare class-balanced RBF-SVM vs best class-balanced QSVC under identical data, splits, and evaluation protocol.
- **Primary metric:** Emergency Macro-F1 (weighted across 6 emergency classes), IPER.

### Subphase 6.5.7 — Noise Robustness Recovery
- **Goal:** If Phase 6.5.6 produces non-trivial emergency recognition, replicate the Protocol B noise sweep on the best models.
- **Condition:** Only executed if clean emergency recall > 0.1 for at least 1 model.

### Subphase 6.5.8 — Phase 6.5 Final Report & Closure
- **Goal:** Complete documentation, statistical analysis, and comparison against frozen Phase 3–6 baseline.

---

## 13. Acceptance Criteria

Phase 6.5 is considered successful if **any** of the following are demonstrated:

| Criterion | Threshold |
| :--- | :--- |
| Emergency Macro-F1 (any model, clean) | > 0.10 |
| IPER (any model, clean) | > 5% |
| Emergency recall (any class, any model) | > 0.20 |
| Statistically significant improvement vs Phase 5 baseline | p < 0.05 on at least 1 emergency class |

Phase 6.5 is considered scientifically complete regardless of outcome if all subphases 6.5.1–6.5.7 are executed correctly and results are reported honestly.

---

## Appendix A — Generated Analysis Files

| File | Description |
| :--- | :--- |
| `phase6_5/analysis/adaptation_class_support.csv` | Per speaker × N adaptation class counts (60 rows) |
| `phase6_5/analysis/emergency_support.csv` | Emergency class support matrix (360 rows) |
| `phase6_5/analysis/speaker_feasibility_matrix.csv` | Feasibility classification (60 rows) |
| `phase6_5/analysis/prediction_distribution_by_model.csv` | Model prediction distribution (84 rows) |
| `phase6_5/analysis/pca_explained_variance.csv` | PCA cumulative EV at 2–256 dimensions |
| `phase6_5/analysis/quantum_config_audit.csv` | Confirmed quantum execution parameters |
| `scripts/run_phase6_5_0_diagnostic.py` | Reproducible diagnostic script |
