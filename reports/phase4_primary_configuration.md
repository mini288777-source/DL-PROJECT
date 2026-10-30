# Phase 4 Primary Quantum Configuration Specification

**Date:** 2026-09-30  
**Phase:** 4 — Quantum Kernel Few-Shot Personalization  
**Status:** **FROZEN PRIOR TO EXPERIMENTATION**  
**Target File:** `reports/phase4_primary_configuration.md`  

---

## 1. Primary Quantum Configuration Definition

To prevent post-hoc hyperparameter selection or p-hacking, the primary quantum configuration for this benchmark is **formally frozen *a priori***.

| Parameter | Frozen Specification | Rationale & Scientific Justification |
|---|---|---|
| **Input Representation** | Frozen 384-D Whisper-tiny embedding | Identical to Phase 3 Adapter and RBF-SVM baselines |
| **Dimensionality Reduction** | Train-only PCA ($384 \to 4$) | Matches qubit count; strictly zero test-set leakage |
| **Pre-Quantum Scaling** | MinMax to $[0, \pi]$ (fitted train-only) | Maps coordinates into phase gate rotation period $[0, \pi]$ |
| **Qubit Count ($d$)** | **4 Qubits** | Balances representational capacity with simulation speed |
| **Feature Map** | ZZ second-order Pauli-Z feature map | Introduces two-body non-linear quantum entanglement |
| **Repetitions (`reps`)** | **1** | Minimal circuit depth ($d=11$), avoiding barren plateaus |
| **Entanglement Strategy** | **Linear** (nearest neighbor) | Nearest-neighbor CNOT connectivity ($6$ two-qubit gates) |
| **Quantum Kernel** | **FidelityQuantumKernel** | State-fidelity overlap $K(\mathbf{x}, \mathbf{y}) = \|\langle \psi(\mathbf{x}) \mid \psi(\mathbf{y}) \rangle\|^2$ |
| **Classifier** | **QSVC** (C-SVM with precomputed kernel) | Dual convex quadratic program, exactly matching classical RBF |
| **Regularization ($C$)** | **$C = 1.0$** | Predetermined; matched to Classical RBF baseline |
| **Simulator Backend** | Local Statevector Simulator | Zero shot noise; exact mathematical fidelity overlap |
| **Random Seed** | **42** | Global seed across Python, NumPy, Qiskit |

---

## 2. Circuit Architecture Details (Primary: $q=4, \text{reps}=1$)

* **Total Qubits:** 4
* **Circuit Depth:** 11
* **One-Qubit Gates:** 11 (4 Hadamard gates $H$, 7 Phase gates $P(2x_i)$)
* **Two-Qubit Gates:** 6 CNOT gates (`cx`)
* **Total Gates:** 17
* **Trainable Quantum Parameters:** **0** (strictly non-variational fixed feature map)
* **Classical Regularization Parameters:** 1 ($C=1.0$)

---

## 3. Transparency & Non-Cheating Declaration

This configuration was chosen and documented **before running any 15-speaker sweep**. The primary curve reported in the final research paper will be this exact 4-qubit, 1-rep linear ZZ configuration, regardless of whether it underperforms, matches, or exceeds the classical baselines.
