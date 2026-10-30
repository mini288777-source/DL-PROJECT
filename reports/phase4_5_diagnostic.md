# Phase 4.5 Controlled Minority-Intent Diagnostic Feasibility Study

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Diagnostic Status:** **NOT FEASIBLE (DATA-CONSTRAINED)**  
**Classification:** Secondary Methodological Audit  

---

## 1. Diagnostic Purpose & Scope

The purpose of this diagnostic was to investigate whether the observed empirical equivalence between Classical RBF-SVM and Quantum Kernel QSVC ($0.1390 \pm 0.0007$) is solely an artifact of majority-class margin collapse under the unweighted $C=1.0$ hinge loss, or whether meaningful performance divergence emerges when models are trained on balanced minority-intent representations.

### Mandatory Experimental Constraints:
1. **No Synthetic Data:** Generating synthetic samples via SMOTE, generative adversarial models, or Gaussian noise jittering is strictly prohibited.
2. **No Data Fabrication or Relabeling:** Merging distinct clinical emergency intents into an artificial monolithic class or inventing intent labels is prohibited.
3. **No Test-Set Contamination:** Pulling emergency recordings from the frozen held-out test splits to construct a balanced adaptation pool is strictly prohibited.
4. **Primary Result Priority:** This diagnostic is strictly secondary; the natural TORGO few-shot experiment remains the primary research finding.

---

## 2. Empirical Feasibility Analysis Across the 15-Speaker Cohort

We audited the adaptation pools across all 60 canonical splits in `splits/torgo_splits.json` using the largest adaptation budget ($N=50$):

| Speaker ID | Group | `GENERAL_SPEECH` Samples | Minority Emergency Samples | Breakdown of Available Emergency Samples in Adaptation Pool |
|---|---|---|---|---|
| **F01** | Dysarthric (F) | 44 | 6 | `CRITICAL_RESPONSE`: 3, `FALL`: 2, `BREATHING_MEDICAL`: 1 |
| **F03** | Dysarthric (F) | 48 | 2 | `CRITICAL_RESPONSE`: 2 |
| **F04** | Dysarthric (F) | 46 | 4 | `CRITICAL_RESPONSE`: 2, `PAIN`: 1, `BREATHING_MEDICAL`: 1 |
| **FC01** | Control (F) | 45 | 5 | `CRITICAL_RESPONSE`: 2, `FALL`: 2, `BREATHING_MEDICAL`: 1 |
| **FC02** | Control (F) | 46 | 4 | `CRITICAL_RESPONSE`: 2, `FALL`: 2 |
| **FC03** | Control (F) | 48 | 2 | `SOS_HELP`: 1, `PAIN`: 1 |
| **M01** | Dysarthric (M) | 46 | 4 | `PAIN`: 2, `FALL`: 1, `BREATHING_MEDICAL`: 1 |
| **M02** | Dysarthric (M) | 48 | 2 | `CRITICAL_RESPONSE`: 2 |
| **M03** | Dysarthric (M) | 47 | 3 | `BREATHING_MEDICAL`: 1, `FALL`: 1, `PAIN`: 1 |
| **M04** | Dysarthric (M) | 46 | 4 | `FALL`: 2, `CRITICAL_RESPONSE`: 1, `PAIN`: 1 |
| **M05** | Dysarthric (M) | 48 | 2 | `CRITICAL_RESPONSE`: 1, `FALL`: 1 |
| **MC01** | Control (M) | 49 | 1 | `CRITICAL_RESPONSE`: 1 |
| **MC02** | Control (M) | 47 | 3 | `BREATHING_MEDICAL`: 1, `FALL`: 1, `PAIN`: 1 |
| **MC03** | Control (M) | 46 | 4 | `BREATHING_MEDICAL`: 2, `FALL`: 1, `LOCATION`: 1 |
| **MC04** | Control (M) | 49 | 1 | `BREATHING_MEDICAL`: 1 |

---

## 3. Evaluation of Candidate Diagnostic Formulations

### 3.1 Option A: Seven-Class Balanced Diagnostic
* **Requirement:** At least $k \ge 1$ samples for all 7 intent classes per speaker.
* **Finding:** **IMPOSSIBLE**. In every speaker split, between 3 and 6 of the emergency classes have exactly **0 samples** in the adaptation pool.
  - `LOCATION` is absent in 14 / 15 speakers.
  - `SOS_HELP` is absent in 14 / 15 speakers.
  - `PAIN` is absent in 9 / 15 speakers.
* **Verdict:** **NOT FEASIBLE**.

### 3.2 Option B: Common Emergency Sub-Class Diagnostic
* **Requirement:** Identify at least one common emergency intent present across all 15 speakers to perform a balanced binary (`GENERAL_SPEECH` vs. `EMERGENCY_INTENT`) comparison.
* **Finding:** **IMPOSSIBLE**. Zero emergency classes exist in all 15 adaptation sets:
  - `CRITICAL_RESPONSE`: Present in 9 speakers, absent in 6 (`FC03`, `M01`, `M03`, `MC02`, `MC03`, `MC04`).
  - `BREATHING_MEDICAL`: Present in 8 speakers, absent in 7 (`F03`, `FC02`, `FC03`, `M02`, `M04`, `M05`, `MC01`).
  - `FALL`: Present in 9 speakers, absent in 6 (`F03`, `F04`, `FC03`, `M02`, `MC01`, `MC04`).
* **Verdict:** **NOT FEASIBLE**.

### 3.3 Option C: Subsetting to Speakers with $\ge 2$ Classes
* **Requirement:** Filter cohort down to a tiny subset of speakers (e.g. F01) and subsample `GENERAL_SPEECH` to match minority counts.
* **Finding:** In F01, total emergency samples $= 6$. Subsampling `GENERAL_SPEECH` to 6 samples yields an adaptation set of $N=12$.
  - Training an SVM on $N=12$ with 4 classes (each with 1–3 samples) severely lacks statistical degrees of freedom.
  - Discarding 14 out of 15 speakers violates cohort validity and destroys generalizability across severity levels.
* **Verdict:** **METHODOLOGICALLY FLAWED & NON-DEFENSIBLE**.

---

## 4. Formal Decision & Scientific Rationale

```
==================================================
DIAGNOSTIC STATUS: NOT FEASIBLE
==================================================
```

### Rationale:
In accordance with Section 19 of the project protocol:
> *"If the diagnostic is not statistically or data-wise feasible: mark: NOT FEASIBLE and document why. Do NOT consider infeasibility a failure."*

Attempting to force a balanced diagnostic on this dataset would require violating foundational research ethics:
1. Synthesizing artificial data (violating data fidelity).
2. Leaking test emergency samples into training (violating split isolation).
3. Inventing synthetic labels (violating clinical taxonomy).

Acknowledging and thoroughly documenting the data constraint preserves scientific honesty. The natural-distribution experiment conducted in Phase 4 remains the definitive, uncompromised empirical result.
