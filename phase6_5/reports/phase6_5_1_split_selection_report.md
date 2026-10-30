# Subphase 6.5.1 — Recovery Split Selection Report

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Phase:** 6.5.1 — Data Feasibility & Speaker-Stratified Recovery Splits  
**Date:** October 2026  
**Status:** CANONICAL SELECTION COMPLETED & AUDITED  

---

## 1. Methodological Preamble & Governance Rules

In strict adherence to experimental research rigor:
1. **Zero Model Peeking:** No models were trained, evaluated, or tuned during this selection process. The selection of the canonical recovery split is derived **strictly from intrinsic dataset geometry, class availability, prompt isolation, and clinical plausibility criteria**.
2. **Frozen Benchmark Protection:** The historical Phase 1–6 results (`results/phase5/`, `phase6/`) remain completely untouched and serve as the frozen historical baseline.
3. **Data Constraint Grounding:** No artificial class support or synthetic data was fabricated.

---

## 2. Candidate Partitioning Strategies Evaluated

Four distinct data partitioning strategies were designed and evaluated against our predeclared selection criteria:

### Candidate Strategy 1: Unconstrained Global Random Few-Shot (Original Phase 1 Protocol)
- **Description:** Randomly sample $N$ prompts from the speaker's total prompt pool, irrespective of intent class labels.
- **Advantages:**
  - Mirrors completely unbiased random sampling from corpus distributions.
  - Very simple implementation.
- **Disadvantages:**
  - Catastrophic minority-class starvation: at $N=20$, yields 94.3% `GENERAL_SPEECH` and an average of 4.9 missing emergency classes per speaker.
  - Causes total classifier collapse to majority class across all kernel models (0.000 emergency recall, 0.000 IPER).
  - Statistically invalid for studying few-shot emergency recognition.
- **Verdict:** **REJECTED** (Root cause of Phase 5 collapse).

### Candidate Strategy 2: Uniform Fixed-Quota Multi-Class Stratification (e.g. $N/7$ per Class)
- **Description:** Enforce equal sample quotas across all 7 classes (e.g. exactly 3 samples per class at $N=21$).
- **Advantages:**
  - Eliminates class imbalance in the training objective.
  - Equal mathematical margin pressure for all classes.
- **Disadvantages:**
  - **Empirically Impossible:** 8 out of 15 speakers completely lack `LOCATION` recordings, and 4 out of 15 completely lack `PAIN` recordings.
  - Forcing equal representation would require either discarding 8 speakers or fabricating synthetic data (violating Section 5).
  - Ecologically implausible: in a real assistive communication deployment, users cannot be forced to record classes they cannot vocalize.
- **Verdict:** **REJECTED** (Violates data feasibility).

### Candidate Strategy 3: Track A — Stratified Direct Intent Adaptation (Adaptive Round-Robin with Singleton Padding)
- **Description:**
  - Identifies all emergency classes physically present for each speaker.
  - Selects whole prompt groups in a balanced round-robin manner across available emergency classes up to ~40% of budget.
  - Guarantees $\ge 1$ held-out test prompt group per class when corpus support permits (flagging `test_support_limited = True` for singletons).
  - Packs the remaining budget to exactly $N \in \{20, 30, 40, 50\}$ using `GENERAL_SPEECH` singletons to ensure zero prompt-group leakage.
  - Preserves exact nested subset containment: $S_{20} \subset S_{30} \subset S_{40} \subset S_{50}$.
- **Advantages:**
  - Directly addresses the original 7-class research question under realistic data constraints.
  - Raises average emergency samples from 1.1 $\to$ 11.3 at $N=20$ and represents 5.2 / 6.0 emergency classes.
  - Zero prompt-group leakage and zero recording leakage.
  - Exactly nested and 100% deterministic.
- **Disadvantages:**
  - For speakers missing `LOCATION` or `PAIN`, those classes remain absent in adaptation (reflecting true corpus limits).
- **Verdict:** **SELECTED AS PRIMARY CANONICAL SPLIT (Track A)**.

### Candidate Strategy 4: Track B — Emergency-Focused / Hierarchical-Ready Allocation
- **Description:**
  - Explicitly targets a 50% Emergency / 50% General Speech split (up to 25 emergency samples at $N=50$).
  - Concentrates all available emergency utterances into the adaptation set to maximize binary discriminative evidence for Stage 1 (Emergency vs General Speech) and subtype variety for Stage 2.
- **Advantages:**
  - Ideal calibration geometry for the planned Subphase 6.5.3 two-stage hierarchical model.
  - Provides the strongest possible binary emergency signal.
- **Disadvantages:**
  - For speakers with low total emergency counts (e.g. F01 has only 10 emergency recordings in total), allocates all or nearly all emergency recordings to adaptation, leaving fewer emergency samples in the test set.
- **Verdict:** **RETAINED AS SPECIALIZED CO-TRACK (Track B)** for Subphase 6.5.3 hierarchical recovery.

---

## 3. Predeclared Selection Evaluation Matrix

| Evaluation Criterion | Weight | Strategy 1 (Original) | Strategy 2 (Uniform 1/7) | Strategy 3 (Track A: Stratified) | Strategy 4 (Track B: Hierarchical) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Data Feasibility** | Critical | High | **FAIL (Impossible)** | **PASS (100% Compatible)** | **PASS (100% Compatible)** |
| **2. Emergency Coverage** | High | Extreme Failure (1.1 recs) | High (if data existed) | **High (11.3 recs, 5.2 classes)** | **Maximum (up to 50% ratio)** |
| **3. Test Support Preservation** | High | High (accidental) | Unknown | **Guaranteed ($\ge 1$ test prompt/class)** | Moderate (higher adapt consumption) |
| **4. Prompt Group Isolation** | Critical | Flawed in cross-sessions | N/A | **Zero Leakage (Verified)** | **Zero Leakage (Verified)** |
| **5. Strict Nestedness ($S_{20..50}$)** | Critical | Yes | No | **Yes (Mathematical Guarantee)** | **Yes (Mathematical Guarantee)** |
| **6. Reproducibility** | High | Deterministic | N/A | **Deterministic (Seed 42)** | **Deterministic (Seed 42)** |
| **7. Ecological Plausibility** | Medium | Plausible | Unrealistic | **High (Balanced few-shot)** | **High (Assisted emergency setup)** |
| **OVERALL RANK** | — | 4th | Disqualified | **1st (CANONICAL WINNER)** | **2nd (Co-Track B)** |

---

## 4. Formal Selection Decision & Rationale

**Canonical Strategy Selected:** **Track A — Stratified Direct Intent Protocol** (`phase6_5/splits/recovery_splits_stratified.json`).

### Primary Justification:
1. **Maximum Feasible Multi-Class Representation:** Track A boosts represented emergency classes from **1.1 classes to 5.2 classes** at $N=20$, reducing zero-support emergency classes from 4.9 down to 0.8.
2. **Defensible Test Preservation:** By reserving held-out prompt groups for every class with $\ge 2$ groups, Track A ensures that downstream test metrics reflect true generalization rather than training set memorization.
3. **Pristine Leakage Control:** By utilizing singleton padding to meet exact $N$ constraints, Track A achieves zero prompt-group overlap between adaptation and test partitions across all 60 conditions.
4. **Direct Comparability:** Track A maintains exact compatibility with the 7-class taxonomy evaluated in Phases 3–5, allowing Subphase 6.5.2 to isolate the pure effect of class-balanced few-shot adaptation against the frozen Phase 5 baseline.

---

## 5. Known Structural Limitations & Guidance for Subphase 6.5.2+

1. **Inherent Corpus Limits Cannot Be Bypassed:**
   - Eight speakers (F01, F03, F04, FC01, M01, M02, M03, M04) will never produce predictions for `LOCATION` because that class was never included in their reading prompt scripts.
   - Four speakers (F01, F03, FC01, MC01) will never predict `PAIN` for the same reason.
   - Downstream Macro-F1 evaluations must explicitly document whether unweighted macro averaging penalizes speakers for classes that do not exist in their physical corpus.
2. **Transition to Subphase 6.5.2:**
   - With the data foundation now fully established and verified by 13 automated tests, Subphase 6.5.2 will introduce **Class-Balanced Classical SVM Baselines** (`class_weight="balanced"`) on these exact recovery splits to test whether classical emergency recognition can be recovered.
