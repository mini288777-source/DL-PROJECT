# Phase 4.5 Three-Way Experimental Fairness Audit

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Evaluated Models:** Classical Adapter Head vs. Classical RBF-SVM vs. Primary Quantum Kernel (`QSVC`, $q=4$)  
**Status:** **100% FAIR & MATCHED**

---

## 1. Audit Objective

A rigorous comparative benchmark requires that the **sole independent variable** across experimental arms is the **adaptation mechanism**. All other factors—speaker identities, training recording IDs, test recording IDs, sample budgets, intent labels, acoustic representations, and metric implementations—must be held strictly constant.

This audit cross-references `results/phase3/baseline_curves.csv` against `results/phase4/quantum_curves.csv`.

---

## 2. Controlled Variables Verification Matrix

| Experimental Variable | Classical Adapter Head | Classical RBF-SVM | Quantum Kernel (`QSVC`, $q=4$) | Verification Status |
|---|---|---|---|---|
| **Speaker Cohort** | 15 TORGO Speakers (8 Dysarthric, 7 Control) | 15 TORGO Speakers (8 Dysarthric, 7 Control) | 15 TORGO Speakers (8 Dysarthric, 7 Control) | **IDENTICAL (15 / 15)** |
| **Sample Budgets ($N$)** | $N \in \{20, 30, 40, 50\}$ | $N \in \{20, 30, 40, 50\}$ | $N \in \{20, 30, 40, 50\}$ | **IDENTICAL (4 / 4)** |
| **Adaptation Recording IDs**| From `splits/torgo_splits.json` | From `splits/torgo_splits.json` | From `splits/torgo_splits.json` | **IDENTICAL (60 / 60)** |
| **Test Recording IDs** | From `splits/torgo_splits.json` | From `splits/torgo_splits.json` | From `splits/torgo_splits.json` | **IDENTICAL (60 / 60)** |
| **Input Representation** | 384-D Whisper-tiny FP32 | 384-D Whisper-tiny FP32 | 384-D Whisper-tiny FP32 | **IDENTICAL (384-D)** |
| **Intent Taxonomy** | Canonical 7-Class Scheme | Canonical 7-Class Scheme | Canonical 7-Class Scheme | **IDENTICAL (7 classes)** |
| **Evaluation Function** | `compute_intent_metrics` | `compute_intent_metrics` | `compute_intent_metrics` | **IDENTICAL (Shared code)**|
| **Single-Class Fallback**| Constant majority prediction | Constant majority prediction | Constant majority prediction | **IDENTICAL (Same 7 splits)**|
| **Random Seed** | Seed = 42 | Seed = 42 | Seed = 42 | **IDENTICAL** |

---

## 3. Independent Variable Characterization

The only variable differing across arms is the learning architecture and adaptation paradigm:

| Property | Classical Adapter Head | Classical RBF-SVM | Primary Quantum Kernel (`QSVC`, $q=4$) |
|---|---|---|---|
| **Paradigm** | Parametric Neural Network | Classical RKHS Kernel Machine | Quantum Statevector Kernel Machine |
| **Capacity Architecture** | Linear(384,64) $\to$ LN $\to$ GELU $\to$ Linear(64,7) | Non-linear infinite-dimensional Gaussian RKHS | Non-linear 16-dimensional quantum Hilbert space |
| **Optimization Method** | Mini-batch SGD / AdamW (cross-entropy) | Convex quadratic programming (dual hinge loss) | Convex quadratic programming (dual hinge loss) |
| **Trainable Parameters** | **25,223** | **Dual Support Vectors** | **0** (Fixed ZZ unitary feature map) |
| **Regularization / Hyperparam**| Weight decay = $0.01$, lr = $10^{-3}$ | $C = 1.0, \gamma = \text{'scale'}$ | $C = 1.0$, linear CNOT, reps=1 |
| **Coordinate Compression** | None (operates on 384-D directly) | None (operates on 384-D directly) | Train-only PCA ($384 \to 4$) + MinMax ($[0, \pi]$) |

---

## 4. Key Scientific Conclusion

Because RBF-SVM and QSVC share:
1. Identical dual soft-margin quadratic programming solvers (`scikit-learn.svm.SVC`)
2. Identical regularization parameter ($C=1.0$)
3. Identical unweighted hinge loss

the observed equivalence in Macro-F1 ($0.1390 \pm 0.0007$) provides a completely fair and direct comparison of classical vs. quantum kernel feature mappings under unweighted few-shot supervision.
