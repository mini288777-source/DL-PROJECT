# Phase 4.5 Formal Null-Result Statement & Scientific Framing Guide

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Classification:** Academic Research Guidance & Framing Specification  

---

## 1. Observed Empirical Result

Under the frozen Phase 4 experimental evaluation across the 15-speaker TORGO few-shot personalization benchmark ($N \in \{20, 30, 40, 50\}$, $q \in \{2, 4, 6, 8\}$):

1. The Quantum Support Vector Classifier (`QSVC`, ZZ feature map, reps=1, linear entanglement, $C=1.0$) achieved a cohort-averaged Macro-F1 score of:
   $$\text{Macro-F1}_{\text{Quantum}} = 0.1390 \pm 0.0007 \quad (95\% \text{ CI}: [0.1383, 0.1398])$$
   and an unweighted speaker-macro accuracy of:
   $$\text{Accuracy}_{\text{Quantum}} = 94.77\% \pm 0.99\% \quad (95\% \text{ CI}: [93.78\%, 95.76\%])$$
2. The Classical Radial Basis Function Support Vector Machine (`RBF-SVM`, $\gamma=\text{'scale'}$, $C=1.0$) achieved identical performance under the same protocol:
   $$\text{Macro-F1}_{\text{RBF-SVM}} = 0.1390 \pm 0.0007, \quad \text{Accuracy}_{\text{RBF-SVM}} = 94.77\% \pm 0.99\%$$
3. The Classical Parametric Adapter head (Linear-LN-GELU-Linear, 25,223 parameters, cross-entropy loss) achieved:
   $$\text{Macro-F1}_{\text{Adapter}} = 0.1410 - 0.1466, \quad \text{Accuracy}_{\text{Adapter}} = 93.25\% - 94.14\%$$

---

## 2. Exact Experimental Conditions Defining the Result

This empirical finding was established strictly under the following set of experimental conditions:
* **Task:** Few-shot personalization for 15 atypical and control speakers drawn from the TORGO speech database.
* **Input Acoustic Embeddings:** 384-dimensional frozen Whisper-tiny representations (unit-L2 normalized, FP32).
* **Dimensionality Reduction:** Unsupervised train-only Principal Component Analysis ($384 \to d \in \{2, 4, 6, 8\}$).
* **Feature Coordinate Encoding:** MinMax scaling to $[0, \pi]$ followed by a 1-repetition second-order Pauli ZZ feature map with linear CNOT entanglement.
* **Kernel Function:** Statevector transition fidelity $K(\mathbf{x}, \mathbf{y}) = |\langle \psi(\mathbf{x}) \mid \psi(\mathbf{y}) \rangle|^2$ evaluated via local ideal simulation without hardware noise or decoherence.
* **Loss Function & Regularization:** Standard convex dual soft-margin hinge loss with unweighted regularization parameter $C=1.0$ (no class weighting or cost-sensitive penalties).
* **Class Distribution:** Severe natural class imbalance ($95.10\%$ `GENERAL_SPEECH`, with the 6 emergency classes collectively comprising $< 5\%$).

---

## 3. Valid Scientific Interpretations

The following statements are scientifically rigorous, defensible, and supported by empirical evidence:
* *"Under this specific unweighted $C=1.0$ kernel-classification protocol and the naturally imbalanced TORGO-derived task, no empirical advantage was observed for the quantum kernel over the classical RBF-SVM baseline."*
* *"Both classical and quantum kernel machines experience majority-class margin collapse under unweighted soft-margin hinge loss in the presence of severe natural class imbalance."*
* *"Expanding the quantum state space from 4 dimensions ($q=2$) to 256 dimensions ($q=8$) does not alter the dual constraint boundary under unweighted hinge loss when adaptation budgets are constrained to $N \le 50$."*
* *"A gradient-trained parametric neural adapter head achieves higher Macro-F1 ($0.1410 - 0.1466$) than both classical and quantum kernel methods because mini-batch cross-entropy updates directly backpropagate non-zero gradients for rare minority classes."*

---

## 4. Invalid Overinterpretations (Prohibited Claims)

The following overgeneralizations are scientifically unjustified, methodologically flawed, and must **never** be made in project manuscripts or reports:

| Prohibited Overgeneralization | Why the Claim is Scientifically False |
|---|---|
| ❌ *"Quantum kernels are ineffective for speech processing."* | The experiment evaluated a specific ZZ feature map with PCA compression ($384 \to 4$) on frozen Whisper-tiny embeddings. It did not evaluate alternative encodings, trainable quantum kernels, or full-rank quantum feature spaces. |
| ❌ *"Quantum computing provides no benefit for atypical speech."* | The experiment tested clean speech adaptation. The primary hypothesis regarding quantum kernel robustness under acoustic noise perturbations remains to be evaluated in Phase 5. |
| ❌ *"Quantum advantage is disproven."* | Quantum advantage is an asymptotic and structural property, not a monolithic binary switch. A null result under a specific few-shot benchmark does not disprove quantum advantage across computational paradigms. |
| ❌ *"The quantum kernel failed."* | The quantum kernel operated with complete mathematical fidelity: $K$ was strictly symmetric, positive semi-definite, non-singular, and non-trivial. The limitation lies in unweighted soft-margin margin collapse under 95% class imbalance, which affected classical RBF-SVM identically. |

---

## 5. Standardized Scientific Lexicon & Vocabulary Guide

To ensure consistency and academic rigor across all project documents:

1. **Precision & Tolerances:**
   - *Avoid:* "bit-for-bit exact"
   - *Use:* "numerically equivalent within the validated tolerance ($\Delta < 10^{-12}$)."

2. **Mechanism Explanations:**
   - *Avoid:* "mathematical proof of RBF collapse"
   - *Use:* "empirical and analytical explanation of the observed majority-class collapse under the specified unweighted SVM protocol."

3. **Comparative Capacity:**
   - *Avoid:* "parameter/compute-matched"
   - *Use:* "parameter and compute transparency."

4. **Performance Summaries:**
   - *Avoid:* "quantum has no advantage"
   - *Use:* "no empirical advantage was observed under the evaluated protocol."

5. **Quantum Model Complexity:**
   - *Maintain:* The quantum feature map has **0 trainable parameters**, but its structural capacity is governed by qubit count ($q \in \{2, 4, 6, 8\}$), Hilbert space dimension ($2^q \in \{4, 16, 64, 256\}$), circuit depth ($5 \dots 23$), and two-qubit entangling gate count ($2 \dots 14$).

---

## 6. Recommended Academic Abstract Wording

> *"In this work, we evaluated whether a second-order Pauli ZZ quantum fidelity kernel could improve few-shot personalized intent recognition for atypical speakers (TORGO cohort, 15 speakers, adaptation budgets $N \in \{20, 30, 40, 50\}$). Using 384-dimensional frozen Whisper embeddings projected via train-only PCA to $d \in \{2, 4, 6, 8\}$ qubits, we conducted a rigorous 240-condition comparative study against classical RBF-SVM and neural adapter baselines. Under an unweighted soft-margin classification protocol ($C=1.0$), both the quantum kernel and classical RBF-SVM achieved identical Macro-F1 scores ($0.1390 \pm 0.0007$) due to majority-class margin collapse under the corpus's natural 95% class imbalance. In contrast, the gradient-trained classical adapter maintained minority-class sensitivity (Macro-F1 $0.1410 - 0.1466$). These findings demonstrate that expanding the Hilbert space dimension without cost-sensitive margin balancing does not mitigate empirical class imbalance, establishing a rigorous, unperturbed baseline for subsequent noise-robustness evaluations."*
