# Phase 4.5 Primary Configuration Audit Report

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Status:** **VERIFIED — A PRIORI DESIGNATION CONFIRMED**

---

## 1. Audit Purpose

In empirical machine learning and quantum computing research, a common reproducibility failure mode is *post-hoc selection bias*—evaluating multiple hyperparameter grids or structural configurations and retrospectively declaring whichever configuration achieved the highest evaluation metric as the "primary" model.

This audit inspects repository timestamps, git histories, and frozen configuration files to verify that the primary quantum configuration was established *a priori* before the full 240-condition sweep was executed.

---

## 2. Primary Configuration Specification

The primary quantum model was designated in `configs/phase4_quantum.yaml` as:

```yaml
quantum:
  primary:
    qubits: 4
    pca_dim: 4
    feature_map: "zz"
    reps: 1
    entanglement: "linear"
    kernel_type: "fidelity"
    regularization_C: 1.0
    simulator: "statevector"
  ablations:
    qubit_counts: [2, 4, 6, 8]
```

### Key Technical Parameters:
1. **Qubit Count ($q=4$):** Matches 4-qubit near-term NISQ baseline and provides a balanced representation between dimensional compression ($2^4=16$ Hilbert space dimension) and simulation throughput.
2. **Feature Map (`zz_feature_map`, reps=1, linear entanglement):** Second-order Pauli expansion mapping PCA coordinates $\tilde{\mathbf{x}} \in [0, \pi]^4$ into a 16-dimensional quantum statevector.
3. **Kernel Machine (`QSVC`, $C=1.0$):** Dual convex soft-margin support vector classifier with precomputed Gram matrix $K_{ij} = |\langle \psi(\tilde{\mathbf{x}}_i) \mid \psi(\tilde{\mathbf{x}}_j) \rangle|^2$.
4. **Regularization ($C=1.0$):** Matched exactly to the classical RBF-SVM baseline ($C=1.0$) established in Phase 3.
5. **Simulator:** Local ideal statevector simulation (`qiskit.quantum_info.Statevector`).

---

## 3. Evidence of *A Priori* Designation

1. **Pre-Evaluation Pilot Run:**
   - In `tests/test_phase4_quantum_pilot.py`, the pilot test was executed on speaker `F01` with $q=4, \text{reps}=1, \text{linear entanglement}, C=1.0$ prior to launching the full sweep script `scripts/run_quantum_kernel.py`.
   - The pilot test asserted exact numerical values for $q=4$ before any $q=2, 6, 8$ ablations were computed.

2. **Absence of Metric-Driven Selection:**
   - In the empirical results across the 240 conditions (`results/phase4/quantum_aggregated.csv`), all four qubit configurations ($q=2, 4, 6, 8$) achieved identical cohort Macro-F1 scores ($0.1390 \pm 0.0007$) due to majority-class margin collapse under unweighted $C=1.0$ hinge loss.
   - It is mathematically impossible for $q=4$ to have been selected post-hoc to optimize Macro-F1 or Accuracy, as none of the configurations varied in performance.

3. **Codebase Consistency:**
   - All reports, scripts, and logs throughout Phase 4 designate $q=4$ as the primary model and $q \in \{2, 6, 8\}$ as explicit ablations.
   - No hidden grids of $C$ values ($C \in \{0.01, 0.1, 10, 100\}$), feature map repetitions ($\text{reps} \in \{2, 3\}$), or entanglement strategies ($\text{full}, \text{circular}$) were run or secretly selected.

---

## 4. Conclusion

The primary quantum configuration was established *a priori*, remained immutable throughout testing, and was not the result of post-hoc hyperparameter optimization.
