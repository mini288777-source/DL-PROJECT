# PHASE 6.5.6 — EXECUTIVE SUMMARY: FINAL FAIR COMPARISON & EVIDENCE SYNTHESIS

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Team:** kernal  
**Phase:** Phase 6.5.6 Final Recovery Synthesis & Research Closure  
**Status:** COMPLETE & FROZEN  

---

### Problem
Atypical (dysarthric) speech recognition in safety-critical emergency communication faces three compounding barriers: extreme natural class imbalance ($\approx 95.10\%$ non-emergency speech), severe calibration data scarcity ($N \le 50$ samples per speaker to avoid patient vocal fatigue), and high acoustic heterogeneity across motor-speech etiologies.

---

### Original Benchmark Failure
In the initial unstratified, flat 7-class benchmark (Phases 3–6), both Classical RBF-SVM and Quantum SVC (QSVC) collapsed completely into the majority class (`GENERAL_SPEECH`). Predicting the majority produced $\approx 95\%$ nominal accuracy but exactly $0.0\%$ emergency recall and precision. Across 81,840 paired evaluations, QSVC predictions were bitwise-identical to RBF-SVM, yielding no observable quantum advantage.

---

### Recovery Interventions
Diagnostic audits (Phases 6.5.0–6.5.4) revealed that the collapse was driven by protocol flaws: $7/60$ calibration splits contained zero emergency samples, losses were unweighted, and classes competed in a single flat softmax/SVM. We recovered functionality through:
1. **Speaker-Stratified Calibration (Track B):** Increased calibration emergency support from $\approx 1.1$ to $11.3$ samples at $N=20$.
2. **Cost-Sensitive Class Weighting:** Penalized majority errors, recovering emergency recall to $58.42\%$ (Adapter).
3. **Hierarchical Two-Stage Routing:** Decoupled binary emergency triage from intent classification, reducing the false-alarm explosion by $\approx 82$ percentage points at $N=30$.

---

### Representation Finding
Forensic code auditing traced the historical claim that $d=4$ PCA captured only $27.4\%$ variance to a hardcoded plotting script artifact. True empirical variance for $d=4$ is $80.66\% \pm 4.37\%$. Expanding dimensionality to $d=8$ captured $89.73\% \pm 2.99\%$ variance and relieved the Stage-2 routing bottleneck, establishing $d=8$ as the primary candidate for fair quantum comparison.

---

### Quantum Result
Under the recovered, leak-free Track B protocol across 60 matched experimental conditions ($15$ speakers $\times 4$ budgets), QSVC Q8-R1 (8 qubits, ZZFeatureMap, linear entanglement, FidelityQuantumKernel) achieved an End-to-End Macro-F1 of $0.1051 \pm 0.0575$ and Emergency Recall of $27.52\% \pm 41.94\%$. All 180 evaluated quantum kernel matrices were verified to be strictly positive semi-definite (eigenvalues $\ge 0.000186$).

---

### Classical Comparison
- **QSVC Q8-R1 vs. Classical RBF-SVM $d=8$:** RBF-SVM achieved Macro-F1 of $0.1087 \pm 0.0487$ and Emergency Recall of $31.89\% \pm 40.97\%$. Functional behavior between the two kernel classifiers was nearly identical.
- **QSVC Q8-R1 vs. Classical Adapter $d=8$:** The Weighted Adapter achieved significantly higher Emergency Recall ($45.81\%$ vs. $27.52\%$, paired difference $\Delta = +18.29\text{ pp}$), but suffered a higher false-emergency rate ($49.82\%$ vs. $28.12\%$).

---

### Statistical Conclusion
Rigorous Wilcoxon signed-rank testing with Holm-Bonferroni family-wise error rate control ($\alpha = 0.05$) across all 60 matched units confirmed **CASE B: NO STATISTICALLY SUPPORTED DIFFERENCE** between QSVC Q8-R1 and Classical RBF-SVM $d=8$ (Macro-F1 $p=0.3428$; Recall $p=0.0689$; False-Emergency Rate $p=0.9505$; IPER $p=0.0582$). Qubit scaling ($q=4 \to q=8$, $p=1.0$) and circuit depth scaling (reps $1 \to 2$, $p=1.0$) also yielded no statistically supported performance improvements.

---

### Runtime Conclusion
Statevector-simulated QSVC incurred massive computational overhead relative to classical baselines:
- **Per-Evaluation Inference Latency:** QSVC Q8-R1 took $1.0816\text{ s}$ per test utterance vs. $0.0014\text{ s}$ for RBF-SVM (**$\approx 772.6\times$ slower**).
- **Total Experiment Wall-Time (60 conditions):** QSVC Q8-R1 required $70.22\text{ s}$ vs. $0.33\text{ s}$ for RBF-SVM (**$\approx 212.8\times$ slower**).

---

### Most Important Contribution
A rigorous **methodological negative control**: this study demonstrates that the original quantum collapse was consistent with severe few-shot data starvation, class imbalance, and the flat classification formulation identified during forensic analysis. Recovery experiments subsequently demonstrated that the collapse behavior was not unique to the quantum model. Under the resulting matched protocol, the fidelity quantum kernel produced performance statistically indistinguishable from the matched classical RBF-SVM across all primary safety metrics while imposing a measured $\approx 772.6\times$ per-evaluation latency penalty.

---

### Main Limitations
Evaluation constrained to $N \le 50$; natural $\approx 95.10\%$ negative class prevalence in evaluated test distributions; sparse samples for `LOCATION` and `PAIN`; frozen Whisper-tiny representations; simulated statevector execution without physical QPU testing; and small cohort size ($15$ TORGO speakers) without clinical validation.

---

### Final Hypothesis Decision
**UNDER THE EVALUATED PROTOCOL: NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE WAS OBSERVED (CASE B).**  
Under the evaluated speaker-stratified few-shot protocol, frozen Whisper representation, and exact statevector fidelity-kernel configuration, no statistically supported quantum advantage was observed over the matched classical RBF-SVM. The result does not establish that quantum kernels cannot provide advantages under other representations, datasets, hardware platforms, or optimization regimes.

---

### Deployment Readiness
**NOT READY**  
Due to high missed-emergency rates ($\ge 54.19\%$), extreme false alarms, low emergency precision ($\le 2.03\%$), and high inference latency, the system is fundamentally not viable for clinical or real-world emergency deployment.
