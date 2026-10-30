# Subphase 6.5.5-C: Computational Runtime & Complexity Audit

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 5, 2026  
**Status:** COMPLETE (Analytical Audit Only)  

---

## 1. Executive Summary of Runtime Audit

In prior Phase 6.5.5 summaries, the computational cost penalty of quantum kernel simulation was informally summarized as *"$\sim 770\times$ slower"*. This audit clarifies the exact mathematical basis of that figure and separates:
1. **Per-evaluation test inference latency:** $\mathbf{772.6\times}$ slower for QSVC Q8-R1 relative to Classical RBF-SVM ($d=8$).
2. **Total 60-condition experiment wall time:** $\mathbf{212.8\times}$ slower for QSVC Q8-R1 relative to Classical RBF-SVM ($d=8$).
3. **Training / kernel construction time:** $\mathbf{22.2\times}$ slower for QSVC Q8-R1 ($0.0888$s vs. $0.0040$s).

---

## 2. Granular Runtime Accounting by Model Family

All empirical measurements were recorded during the full 60-condition sweep across 15 TORGO speakers ($N \in \{20, 30, 40, 50\}$) on an AMD/Intel x86_64 architecture running exact statevector simulation:

| Model Specification | Family | Dimension ($d$) | Qubits ($q$) | Circuit Depth | CNOT Gates | Mean Train Time (s) | Mean Inf Time (s) | Total Wall Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical RBF-SVM ($d=8$)** | Classical | 8 | — | — | — | $0.0040 \pm 0.0008$ | $0.0014 \pm 0.0009$ | **$0.33$ s** |
| **Classical Adapter ($d=8$)** | Classical | 8 | — | — | — | $0.2625 \pm 0.1049$ | $0.0006 \pm 0.0002$ | **$15.79$ s** |
| **Classical RBF-SVM ($d=4$)** | Classical | 4 | — | — | — | $0.0038 \pm 0.0007$ | $0.0014 \pm 0.0009$ | **$0.31$ s** |
| **Classical Adapter ($d=4$)** | Classical | 4 | — | — | — | $0.2247 \pm 0.0890$ | $0.0004 \pm 0.0002$ | **$13.51$ s** |
| **QSVC Q4-R1** | Quantum | 4 | 4 | 11 | 6 | $0.0247 \pm 0.0179$ | $0.4194 \pm 0.4570$ | **$26.64$ s** |
| **QSVC Q8-R1** | Quantum | 8 | 8 | 23 | 14 | $0.0888 \pm 0.0690$ | $1.0816 \pm 1.0408$ | **$70.22$ s** |
| **QSVC Q8-R2** | Quantum | 8 | 8 | 31 | 28 | $0.1578 \pm 0.1255$ | $1.9761 \pm 1.8272$ | **$128.04$ s** |

---

## 3. Computational Ratios & Exact Arithmetic Verification

### 3.1 Primary Comparison: QSVC Q8-R1 vs. Classical RBF-SVM ($d=8$)
- **Per-Evaluation Inference Latency:**
  $$\text{Ratio} = \frac{1.0816 \text{ s}}{0.0014 \text{ s}} = 772.57 \approx \mathbf{772.6\times}$$
  *Interpretation:* Evaluating kernel overlaps for a test utterance against adaptation support vectors using 8-qubit statevectors takes $\sim 773\times$ longer than computing Euclidean distances in scikit-learn's optimized C-backend.
- **Total Experiment Wall Time:**
  $$\text{Ratio} = \frac{70.22 \text{ s}}{0.33 \text{ s}} = 212.79 \approx \mathbf{212.8\times}$$
  *Interpretation:* Across the entire 60-condition sweep, QSVC required $70.22$ seconds compared to $0.33$ seconds for classical RBF-SVM.

### 3.2 Circuit Depth Scaling: QSVC Q8-R2 vs. QSVC Q8-R1
- **Depth and Gate Scaling:**
  - Circuit depth increases from $23$ to $31$ ($1.35\times$).
  - Two-qubit CNOT gate count doubles from $14$ to $28$ ($2.0\times$).
  - One-qubit gate count doubles from $23$ to $46$ ($2.0\times$).
- **Runtime Scaling:**
  $$\text{Ratio} = \frac{128.04 \text{ s}}{70.22 \text{ s}} = 1.823 \approx \mathbf{1.82\times}$$
  *Interpretation:* Doubling entangling repetitions nearly doubles total execution time while producing negligible classification change ($\Delta \text{Macro-F1} = +0.0019$).

### 3.3 Comparison against Classical Adapter ($d=8$)
- **Total Wall Time Ratio:**
  $$\text{Ratio} = \frac{70.22 \text{ s}}{15.79 \text{ s}} = 4.447 \approx \mathbf{4.45\times}$$
  *Interpretation:* Even though the Classical Adapter undergoes 50 epochs of AdamW gradient descent per condition, it completes in under 16 seconds—$4.45\times$ faster than QSVC Q8-R1.

---

## 4. Hardware Deployment & Scaling Implications

1. **Statevector Simulation vs. Physical QPU Execution:**
   - In statevector simulation, kernel entries $K_{ij} = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$ are computed via exact linear algebra on complex statevectors of dimension $2^q = 256$.
   - On a physical Quantum Processing Unit (QPU), computing each entry requires running a swap-test or compute-uncompute circuit over $S$ shots (typically $S \in [10^3, 10^4]$).
   - For an adaptation set of $N=50$ and a test set of $M=300$, computing the test kernel matrix requires $M \times N = 15,000$ circuit executions per condition.
   - At $15,000 \text{ circuits} \times 10^3 \text{ shots} = 1.5 \times 10^7$ shots per speaker condition, physical QPU execution would be astronomically expensive with zero expected accuracy gain.
2. **Scientific Conclusion on Efficiency:**
   - Under both simulated and physical execution paradigms, quantum kernel classifiers impose substantial computational overhead without compensating performance benefit over classical kernel baselines.
