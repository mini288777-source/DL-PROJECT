# Phase 4.5 Complete Results & Research Hardening Report

**Date:** 2026-09-30  
**Phase:** 4.5 — Phase 4 Closure, Research Hardening & Phase 5 Gate  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Sample Budgets:** $N \in \{20, 30, 40, 50\}$ labeled samples per speaker  
**Primary Configuration:** 4 Qubits, 1 Repetition, Linear Entanglement, ZZ Feature Map, Fidelity Quantum Kernel, QSVC ($C=1.0$)  
**Status:** **PHASE 4.5 COMPLETE — READY FOR PHASE 5**  

---

## 1. Phase 4 Closure Summary

Phase 4.5 successfully conducted a comprehensive audit, research hardening, and integrity verification across all artifacts, source code, numerical results, and scientific documentation generated during Phase 4.

Every empirical finding has been independently verified against raw CSVs and persistent kernel files. The frozen Phase-3.5 experimental contract was strictly preserved, zero information leakage was detected, all 240 conditions were accounted for without omission, and scientific language was refined to maintain uncompromised academic defensibility.

---

## 2. Remaining Work Identified & Completed

| Domain | Issue / Item Identified | Remediation / Action Completed in Phase 4.5 | Validation Artifact |
|---|---|---|---|
| **Prediction Lineage** | Explicit `method` and `N` column names in predictions | Added `method="qsvc"` and `N` columns to `quantum_predictions.csv` after archiving pre-audit original | `tests/test_phase4_5_lineage.py` |
| **Leakage Verification** | Need formal test for train-only PCA/MinMax isolation | Created unit test asserting zero parameter shifts during test transformation | `tests/test_phase4_5_no_transform_leakage.py` |
| **Scientific Wording** | Overstated phrases ("bit-for-bit exact", "mathematical proof") | Replaced with rigorous wording ("numerically equivalent within tolerance", "analytical explanation") | `reports/phase4_5_null_result_statement.md` |
| **Ablation Visualization**| Dedicated qubit count ablation figure required | Added `qubit_ablation_figure.png` to visualization pipeline and artifacts | `results/phase4/figures/qubit_ablation_figure.png` |
| **Diagnostic Feasibility**| Controlled minority-intent feasibility assessment | Evaluated adaptation pools; confirmed infeasibility without synthetic data | `reports/phase4_5_diagnostic.md` |
| **Artifact Freezing** | Need cryptographic fingerprints of all canonical outputs | Generated SHA-256 registry for all canonical Phase 4 outputs | `reports/PHASE4_FROZEN_ARTIFACTS.md` |
| **Phase 5 Contract** | Need formal specification for noise sweep order of operations | Authored input contract and experimental grid design doc | `configs/phase5_contract.yaml` |
| **Noise Infrastructure** | Need verification of noise generator without running sweep | Implemented 5-case unit test validating SNR accuracy and determinism | `tests/test_phase5_input_contract.py` |

---

## 3. Comprehensive Audit Findings

1. **Condition Accounting:** Exactly 240 conditions ($15\text{ speakers} \times 4\text{ budgets} \times 4\text{ qubit counts}$) evaluated with a **100.0% success rate (0 failures)**.
2. **Prediction Instances:** Exactly **85,072 prediction records** logged with zero orphan instances.
3. **Mercer Kernel Quality:** All 212 multi-class training matrices in `results/phase4/kernel_cache/` verified to be strictly symmetric ($\max |K - K^T| < 10^{-14}$), unit diagonal ($\text{diag}(K) \equiv 1.0$), and positive semi-definite ($\lambda_{\min} \ge -10^{-12}$).
4. **Single-Class Conditions:** Exactly 28 conditions ($7\text{ splits} \times 4\text{ qubit counts}$) flagged with `single_class_training=True` and handled via constant prior fallback without inventing emergency labels.
5. **Experimental Fairness:** Classical Adapter, Classical RBF-SVM, and Primary Quantum QSVC ($q=4$) evaluated on identical splits, recordings, and metrics.

---

## 4. Primary Experiment & Ablation Verification

### 4.1 Primary Quantum Performance ($q=4$, reps=1, linear)

| Sample Budget ($N$) | Cohort Macro-F1 | 95% Confidence Interval | Cohort Accuracy | Weighted-F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **$N = 20$** | **0.1390** | $[0.1383, 0.1398]$ | **94.77%** | 0.9224 | 0.1354 | 0.1429 |
| **$N = 30$** | **0.1390** | $[0.1383, 0.1398]$ | **94.77%** | 0.9224 | 0.1354 | 0.1429 |
| **$N = 40$** | **0.1390** | $[0.1383, 0.1398]$ | **94.77%** | 0.9224 | 0.1354 | 0.1429 |
| **$N = 50$** | **0.1390** | $[0.1383, 0.1398]$ | **94.77%** | 0.9224 | 0.1354 | 0.1429 |

### 4.2 Qubit Count Ablation & Structural Capacity

| Qubits ($q$) | PCA Dim ($d$) | Hilbert Dim ($2^d$) | Cumulative PCA Variance | $N=50$ Macro-F1 | $N=50$ Accuracy | Total Gates | Circuit Depth | Trainable Params |
|---|---|---|---|---|---|---|---|---|
| **$q = 2$** | 2 | 4 | $17.8\% \pm 3.1\%$ | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | 7 | 5 | **0** |
| **$q = 4$ (PRIMARY)**| 4 | 16 | $27.4\% \pm 3.9\%$ | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | 17 | 11 | **0** |
| **$q = 6$** | 6 | 64 | $34.2\% \pm 4.2\%$ | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | 27 | 17 | **0** |
| **$q = 8$** | 8 | 256 | $39.8\% \pm 4.5\%$ | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | 37 | 23 | **0** |

---

## 5. Three-Way Comparative Analysis

| Adaptation Strategy | Budget ($N$) | Macro-F1 ($CI_{95}$) | Accuracy ($CI_{95}$) | Trainable Parameters | Capacity Model |
|---|---|---|---|---|---|
| **Classical Adapter Head** | 20 | **$0.1466 \pm 0.0127$** | $93.87\% \pm 1.90\%$ | 25,223 | Parametric Neural MLP |
| **Classical Adapter Head** | 30 | **$0.1453 \pm 0.0078$** | $93.53\% \pm 1.25\%$ | 25,223 | Parametric Neural MLP |
| **Classical Adapter Head** | 40 | **$0.1472 \pm 0.0129$** | $93.25\% \pm 1.53\%$ | 25,223 | Parametric Neural MLP |
| **Classical Adapter Head** | 50 | **$0.1410 \pm 0.0051$** | $94.14\% \pm 0.85\%$ | 25,223 | Parametric Neural MLP |
| **Classical RBF-SVM** | 20 | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | Dual Support Vectors | Classical RKHS Kernel |
| **Classical RBF-SVM** | 50 | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | Dual Support Vectors | Classical RKHS Kernel |
| **Quantum Kernel (QSVC $q=4$)** | 20 | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | **0** | Quantum Hilbert Space |
| **Quantum Kernel (QSVC $q=4$)** | 50 | $0.1390 \pm 0.0007$ | $94.77\% \pm 0.99\%$ | **0** | Quantum Hilbert Space |

---

## 6. Null-Result Statement & Class Imbalance Status

1. **Condition-Specific Null Result:** Under the evaluated unweighted soft-margin protocol ($C=1.0$), the quantum kernel does not demonstrate an empirical Macro-F1 advantage over classical RBF-SVM on clean speech.
2. **Margin Collapse Mechanism:** In dual soft-margin SVM optimization, the single minority training sample's dual weight is bounded by $C=1.0$, which is overwhelmed by 19–47 majority samples, causing both classical and quantum kernel decision boundaries to default to `GENERAL_SPEECH`.
3. **Controlled Diagnostic Status:** Formally designated as **NOT FEASIBLE** due to natural corpus constraints (3 to 6 emergency classes have zero samples in every speaker's adaptation split; no emergency class exists across all 15 speakers).

---

## 7. Artifact Integrity & SHA-256 Fingerprints

All canonical Phase 4 research artifacts have been cryptographically fingerprinted in `reports/PHASE4_FROZEN_ARTIFACTS.md`:
* `results/phase4/quantum_curves.csv`: `67a2a506a4fd624543470668a4b6ac33dc574165cebd681e43b043a6a3d3da65`
* `results/phase4/quantum_predictions.csv`: `5a4e497ac46d1bd41deb8691807f56e1afa30d91440e40bc091160c86d96da16`
* `results/phase4/quantum_aggregated.csv`: `436c1695ce461c577ecc6696e65e160845bd0813a3a3e99521d2fb3e018ecf4e`
* `results/phase4/ablation_table.csv`: `7ce2838c9b2481081d980478ae49e0ecad13de9abad64af0563c780c651c9bb9`
* `results/phase4/classical_vs_quantum_comparison.csv`: `9fba72141f83933b151e6dae786f8a9c4df3273deb0c94209fad8b6673616c85`

---

## 8. Phase 5 Input Contract & Infrastructure Validation

* **Specification:** Frozen in `configs/phase5_contract.yaml` and `reports/phase5_experimental_contract.md`.
* **Execution Grid:** 3 models $\times$ 4 sample budgets ($20, 30, 40, 50$) $\times$ 4 noise levels (Clean, $15\text{ dB}$, $10\text{ dB}$, $5\text{ dB}$) $\times$ 15 speakers = **720 runs**.
* **Split Preservation Rule:** Noise applied strictly after split assignment; train/test identities locked.
* **Toy Infrastructure Test:** Verified in `tests/test_phase5_input_contract.py` (5 / 5 tests pass).
* **Hard Stop Enforcement:** Phase 5 has NOT been executed.

---

## 9. Final Readiness Decision

```
==================================================
PHASE 4.5 COMPLETE
PHASE 5 READY
==================================================
```
All 33 unit, integration, and reproducibility tests are passing.
