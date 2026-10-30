# FINAL RESEARCH WORDING & CANONICAL PHRASING GUIDE
## Authoritative Lexicon for Papers, Defense Slides, Review 2 & Scholarly Communication

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Scope:** Canonical Statements, Lexical Constraints, and Approved Scientific Formulations  
**Status:** COMPLETE & FROZEN  

---

## 1. Canonical Statements

### 1.1 Canonical Primary Hypothesis Statement
> **Canonical Phrasing:**  
> "Under the evaluated speaker-stratified few-shot protocol, frozen Whisper representation, and exact statevector fidelity-kernel configuration, no statistically supported quantum advantage was observed over the matched classical RBF-SVM."  
> **Formal Classification:** CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE.

*Extended Form for Discussions:*  
"The result indicates that under the studied few-shot conditions ($N \le 50$) and frozen Whisper embeddings, the fidelity quantum kernel performed statistically indistinguishably from the classical RBF-SVM, while incurring a measured $\approx 772.6\times$ inference latency penalty. These findings do not establish a universal limitation of quantum kernels; they indicate that no empirical advantage was observed under the specific representation, dataset, sample budgets, and simulation configuration evaluated in this study."

---

### 1.2 Canonical Scientific Contribution Statement
> **Canonical Phrasing:**  
> "The study contributes a controlled, reproducible evaluation framework for few-shot atypical speech recognition that explicitly audits data support, class imbalance, hierarchical routing, representation dimensionality, and matched classical-versus-quantum kernel performance. Under the evaluated configuration, the study finds no statistically supported quantum advantage and documents the computational trade-off of exact statevector QSVC."

---

### 1.3 Canonical Final Research Conclusion
> **Canonical Phrasing:**  
> "The study evaluated a fidelity quantum-kernel approach for low-resource personalized atypical speech recognition under controlled few-shot conditions. The original direct 7-class benchmark exhibited majority-class collapse in both the classical RBF-SVM and quantum SVC, consistent with severe emergency-class support limitations, imbalance, and the flat classification formulation. A separate recovery protocol addressed these methodological constraints through speaker-stratified sampling, cost-sensitive learning, hierarchical routing, and representation auditing. Under the resulting matched Track B protocol, the $d=8$ fidelity quantum kernel did not demonstrate a statistically supported performance advantage over the matched classical RBF-SVM. Increasing the quantum feature dimension from four to eight qubits and increasing circuit depth from one to two repetitions likewise produced no statistically supported improvement. The quantum implementation incurred substantially higher measured computational cost under exact statevector simulation. These findings do not establish a universal limitation of quantum kernels; they indicate that no empirical advantage was observed under the specific representation, dataset, sample budgets, and simulation configuration evaluated in this study."

---

## 2. Approved Lexical Dictionary & Typology

To maintain scientific integrity in all presentations and papers, enforce the following word substitutions:

| Concept / Domain | PROHIBITED Phrasing | REQUIRED / APPROVED Scientific Phrasing |
| :--- | :--- | :--- |
| **Benchmark Collapse Cause** | "caused by data starvation" / "artifact of formulation flaws" | "consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation" |
| **Quantum vs. RBF Behavior** | "quantum kernels act as RBF kernels" / "functionally equivalent to RBF" | "produced performance statistically indistinguishable from the matched classical RBF-SVM under the evaluated configuration" |
| **Representation Selection** | "d=8 is optimal" / "best dimension" | "d=8 was selected as the primary candidate representation because it substantially relieved the observed Stage-2 compression bottleneck while retaining 89.73% ± 2.99% variance" |
| **Curse of Dimensionality** | "PCA solved the curse of dimensionality" | "reduced-dimensional representations were associated with improved downstream performance for some classical models under small-sample constraints" |
| **Hierarchical Efficacy** | "hierarchical routing solves emergency recognition / makes system reliable" | "hierarchical routing substantially reduced false-emergency rates under matched recovery conditions, but substantial missed-emergency and low-precision limitations remained" |
| **Quantum Latency** | "quantum is 770× slower overall" | "QSVC Q8-R1 was approximately 772.6× slower per measured test evaluation and 212.8× slower in total experiment wall time" |
| **Quantum Generalization** | "quantum ML is useless / cannot help" | "no statistically supported quantum advantage was observed under the evaluated configuration" |
| **Classical Model Superiority**| "the Adapter is universally superior" | "the weighted Adapter achieved higher emergency recall than QSVC, while RBF-SVM provided the strongest matched kernel baseline; model selection involves a trade-off between recall and false alarms" |
| **Operational Readiness** | "reliable emergency communication system" / "deployment ready" | "experimental emergency-intent recognition framework / research-stage system (Status: NOT READY)" |
| **Experimental Tracks** | "Phase 6.5 proved Phase 3 was wrong" / combining tracks in single test | "Track 1 and Track 2 represent distinct experimental formulations; synthesis between them is strictly qualitative and attributional" |

---

## 3. Defense Slide & Paper Guidelines

### For Paper Submission & Review 2:
1. **Never apologize for a null result:** Frame Case B as a rigorous scientific contribution that protects the literature from false positive claims.
2. **Emphasize the Methodological Audit:** The forensic discovery of calibration starvation ($7/60$ single-class splits) and the PCA placeholder artifact demonstrates high experimental vigilance.
3. **Present the Two Tracks Separately:** Keep Track 1 (Original Benchmark) and Track 2 (Recovery Protocol) clearly partitioned in tables and figures.
4. **State Deployment Status Transparently:** Always explicitly state that the system is **NOT READY** for clinical deployment due to low precision ($\le 2.03\%$) and missed emergency rates ($>54\%$).
5. **Acknowledge Simulation Boundaries:** Explicitly note that tests used exact classical statevector simulation on CPU without physical quantum hardware.
