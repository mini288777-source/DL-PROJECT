# Phase 4.5 Single-Class Adaptation Conditions Audit Report

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Status:** **VERIFIED — IDENTICAL TO PHASE 3.5 TREATMENT**

---

## 1. Background & Context

In low-resource few-shot personalization ($N \in \{20, 30, 40, 50\}$), stratified sampling from an extremely imbalanced natural corpus ($95.10\%$ `GENERAL_SPEECH`) will inevitably draw zero minority emergency samples for certain low-resource speaker splits.

Phase 3.5 audited and identified exactly **7 single-class adaptation splits** out of the 60 total speaker-budget pairs. In these 7 splits, all $N$ adaptation recordings belong exclusively to the `GENERAL_SPEECH` majority class.

---

## 2. Inventory of Single-Class Conditions in Phase 4

Because Phase 4 evaluated 4 qubit dimensions ($q \in \{2, 4, 6, 8\}$) for every split, the 7 single-class splits expand to exactly **28 conditions** in `results/phase4/quantum_curves.csv`:

| Condition Index | Speaker ID | Cohort Group | Sample Budget ($N$) | Qubit Counts Evaluated ($q$) | Single Class in Training Split |
|---|---|---|---|---|---|
| **1** | `F04` | Dysarthric (Female) | $N = 20$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (20 / 20) |
| **2** | `FC03` | Control (Female) | $N = 20$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (20 / 20) |
| **3** | `M05` | Dysarthric (Male) | $N = 20$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (20 / 20) |
| **4** | `M05` | Dysarthric (Male) | $N = 30$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (30 / 30) |
| **5** | `M05` | Dysarthric (Male) | $N = 40$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (40 / 40) |
| **6** | `MC01` | Control (Male) | $N = 20$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (20 / 20) |
| **7** | `MC01` | Control (Male) | $N = 30$ | $q \in \{2, 4, 6, 8\}$ (4 conditions) | `GENERAL_SPEECH` (30 / 30) |

Total Single-Class Conditions: **28 / 240** ($11.67\%$).

---

## 3. Audit of Phase 4 Handling & Integrity Verification

We verified the handling of these 28 conditions in `src/models/quantum_kernel.py`:

```python
unique_classes = np.unique(y_train)
if len(unique_classes) < 2:
    # Single-class adaptation fallback: constant majority-class predictor
    self.is_single_class_ = True
    self.single_class_label_ = unique_classes[0]
```

### Specific Audit Checks:
1. **Explicit Flagging:** All 28 conditions have `single_class_training = True` in `results/phase4/quantum_curves.csv` and note `SINGLE_CLASS_ADAPTATION_PRIOR`.
2. **No Fake Multiclass Training:** Neither Qiskit's `QSVC` nor scikit-learn's `SVC` was forced to solve a degenerate 1-class quadratic programming problem, preventing solver divergence or runtime exceptions.
3. **No Synthetic Data Generation:** Zero synthetic minority samples (e.g., SMOTE, noise-perturbed vectors) were injected into the training splits.
4. **No Label Fabrication:** Zero samples were relabeled to invent emergency intents.
5. **Untouched Evaluation Sets:** Held-out test sets remained completely unperturbed, containing their natural distribution of both general and emergency speech.
6. **Retention in Cohort Aggregation:** These 28 conditions were **retained** in all macro-averages and standard deviations. No unfavorable conditions were dropped to artificially inflate Macro-F1.

---

## 4. Conclusion

The single-class adaptation conditions were handled with rigorous scientific honesty, matching Phase 3.5 protocols with zero data contamination.
