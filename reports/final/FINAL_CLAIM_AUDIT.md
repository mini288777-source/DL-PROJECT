# FINAL SCIENTIFIC CLAIM CORRECTIONS DIRECTORY
## Detailed Catalogue of Systematic Text Harmonizations and Wording De-Biasing

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Scope:** Text-Level Corrections, Scoping, and Scientific Accuracy Adjustments  
**Status:** COMPLETE & FROZEN  

---

## 1. Overview

This document provides a granular reference guide to the nine critical scientific wording corrections applied across the repository. It explicitly details the flawed or overclaiming phrase, the scientific rationale for why the phrasing required revision, and the exact authoritative phrasing adopted.

---

## 2. Granular Correction Entries

### CORRECTION 1: Causal Attribution of Original Benchmark Collapse
- **Problematic Original Formulation:**  
  *"The apparent collapse of quantum personalization in the original benchmark was an artifact of few-shot data starvation and formulation flaws."*
- **Scientific Flaw:**  
  The word "artifact" in this context and the strong causal assertion ("was an artifact of") imply a proven counterfactual causal proof that cannot be formally claimed from observational few-shot evaluations on a single cohort.
- **Authoritative Corrected Formulation:**  
  *"The original quantum collapse was consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation identified during forensic analysis. Recovery experiments subsequently demonstrated that the collapse behavior was not unique to the quantum model."*
- **Alternative Acceptable Formulation:**  
  *"The original benchmark exhibited collapse behavior under severe few-shot class-support and imbalance constraints. Recovery experiments showed that this behavior was shared by the corresponding classical kernel baseline and therefore could not be attributed uniquely to the quantum model."*
- **Prohibited Words in Context:** Do not use "proved", "caused by", or "uniquely caused by".

---

### CORRECTION 2: Broad Kernel Functional Equivalence
- **Problematic Original Formulation:**  
  *"The fidelity quantum kernel behaves functionally as a stationary distance-based similarity metric analogous to a classical RBF kernel."*
- **Scientific Flaw:**  
  This constitutes an overgeneralization. In general Hilbert space theory, fidelity quantum kernels with non-linear feature maps can induce non-stationary, non-Euclidean geometries. The observation that it behaved analogously to an RBF kernel in this study is strictly an empirical outcome conditioned on the frozen Whisper embeddings, PCA scaling, and few-shot sampling.
- **Authoritative Corrected Formulation:**  
  *"Under the evaluated frozen Whisper representation, feature scaling, kernel configuration, and few-shot protocol, the fidelity quantum kernel produced performance statistically indistinguishable from the matched classical RBF-SVM."*
- **Concise Form:**  
  *"Under the evaluated configuration, QSVC produced performance statistically indistinguishable from the matched RBF-SVM."*

---

### CORRECTION 3: Representation Dimensionality Optimality
- **Problematic Original Formulation:**  
  *"d=8 is the optimal dimension." / "d=8 universally solves representation capacity."*
- **Scientific Flaw:**  
  The choice of $d=8$ was constrained by the computational feasibility of simulating up to 8 qubits on classical hardware without exponential slowdown. Higher classical dimensions ($d=16, 32$) actually achieved higher raw downstream performance.
- **Authoritative Corrected Formulation:**  
  *"d=8 was selected as the primary candidate representation for the matched quantum comparison because it substantially relieved the observed Stage-2 compression bottleneck while retaining 89.73% ± 2.99% cumulative variance under the evaluated recovery protocol."*
- **Prohibited Words in Context:** Do not use "universally optimal", "best representation", or "optimal dimension" without explicit scoping to the evaluation protocol.

---

### CORRECTION 4: PCA and the "Curse of Dimensionality"
- **Problematic Original Formulation:**  
  *"PCA dimensionality reduction solved the curse of dimensionality for few-shot speech recognition."*
- **Scientific Flaw:**  
  Linear PCA does not theoretically solve the curse of dimensionality. It merely projects features onto principal axes of maximum variance, which in this dataset reduced small-sample covariance estimation noise.
- **Authoritative Corrected Formulation:**  
  *"Reduced-dimensional representations were associated with improved downstream performance for some classical models under the evaluated few-shot conditions."*

---

### CORRECTION 5: Operational Efficacy of Hierarchical Routing
- **Problematic Original Formulation:**  
  *"Hierarchical routing solves the emergency speech recognition problem and enables reliable emergency communication."*
- **Scientific Flaw:**  
  While hierarchical routing dramatically reduced the False-Emergency Rate ($\Delta = -81.76\text{ pp}$ at $N=30$), the system's emergency precision remained extremely low ($\le 2.03\%$), and more than $54\%$ of true emergency utterances were missed. The system is therefore fundamentally not "reliable" or "solved."
- **Authoritative Corrected Formulation:**  
  *"Hierarchical routing substantially reduced false-emergency rates under the evaluated matched recovery conditions, but substantial missed-emergency and low-precision limitations remained."*

---

### CORRECTION 6: Computational Cost & Runtime Distinction
- **Problematic Original Formulation:**  
  *"Quantum computing is 770× slower overall than classical SVM."*
- **Scientific Flaw:**  
  This conflates two fundamentally different operational metrics: single-sample test inference latency versus total experimental wall-clock duration across 60 conditions.
- **Authoritative Corrected Formulation:**  
  - **Per-Evaluation Inference Latency:** *"The measured Q8-R1 QSVC implementation was approximately 772.6× slower per measured test evaluation than the matched d=8 RBF-SVM in exact statevector simulation (1.0816 s vs. 0.0014 s)."*
  - **Total Experiment Wall Time:** *"Across the complete 60-condition experiment, total wall-clock time was approximately 212.8× higher for Q8-R1 than for the matched RBF-SVM (70.22 s vs. 0.33 s)."*
  - **Q8-R2 Circuit Depth Scaling:** Approximately $1408.8\times$ slower per evaluation ($1.9761\text{ s}$) and $388.0\times$ slower in total wall-clock runtime ($128.04\text{ s}$).

---

### CORRECTION 7: Universal Negative Claims Against Quantum Learning
- **Problematic Original Formulation:**  
  *"Quantum machine learning provides no benefit and quantum kernels cannot assist atypical speech."*
- **Scientific Flaw:**  
  A null result under a specific set of constraints (fixed ZZFeatureMap, linear entanglement, frozen Whisper-tiny front-end, exact statevector simulation on CPU, $N \le 50$) does not prove a universal impossibility theorem for all quantum machine learning algorithms.
- **Authoritative Corrected Formulation:**  
  *"No statistically supported quantum advantage was observed under the evaluated configuration."*
- **Extended Form:**  
  *"Under the evaluated speaker-stratified few-shot protocol, frozen Whisper representation, and exact statevector fidelity-kernel configuration, no statistically supported quantum advantage was observed over the matched classical RBF-SVM. The present study does not establish a universal limitation of quantum kernels."*

---

### CORRECTION 8: Universal Classical Model Superiority
- **Problematic Original Formulation:**  
  *"The classical Adapter is universally superior to quantum and RBF models."*
- **Scientific Flaw:**  
  The Classical Adapter achieved superior Emergency Recall ($45.81\%$), but suffered an unacceptably high false-emergency rate ($49.82\%$). It represents a different operating point on the precision-recall trade-off curve, not a universally superior classifier.
- **Authoritative Corrected Formulation:**  
  *"Under the evaluated recovery protocol, the weighted Adapter achieved higher emergency recall than QSVC, while the RBF-SVM provided the strongest matched kernel baseline by the primary comparison criteria. Model selection involves an explicit trade-off between emergency-event capture and false-alarm burden."*

---

### CORRECTION 9: Operational Terminology & Clinical Readiness
- **Problematic Original Formulation:**  
  Referring to the experimental software pipeline as a *"Reliable Emergency Communication System"* or claiming *"deployment readiness."*
- **Scientific Flaw:**  
  The project title is a frozen research grant title. However, referring to the experimental implementation as a viable, safe, or deployable emergency system is false and dangerous.
- **Authoritative Corrected Formulation:**  
  When describing the software or experimental pipeline, use:  
  *"experimental emergency-intent recognition framework"* or *"research-stage emergency speech recognition system"*.  
  The operational status is explicitly frozen as:  
  **DEPLOYMENT READINESS: NOT READY**  
  **CLINICAL VALIDATION: NOT ESTABLISHED**  
  **REAL-WORLD EMERGENCY RELIABILITY: NOT ESTABLISHED**
