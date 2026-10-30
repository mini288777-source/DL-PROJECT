# PHASE 8 FAILURE-MODE FORENSIC AUDIT
## Quantitative Diagnostic of Observed Failure Mechanisms Across Phases 1–7

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Status:** **CANONICALLY FROZEN FOR PHASE 8 FOUNDATION**  
**Date:** October 2026  

---

## 1. Executive Forensic Summary

Before introducing any new architectural, algorithmic, or training intervention in Phase 8, this document explicitly quantifies the 10 failure mechanisms observed across Phase 4, Phase 5, Phase 6.5, and Phase 7.

The central empirical reality established by previous phases is that **quantum kernels and variational circuits failed in Phase 4–7 not because quantum circuits cannot evaluate inner products, but because they were tasked with global 7-class classification under severe class imbalance (~95% background speech)**. This led to catastrophic soft-margin and gradient collapse onto the majority non-emergency class.

Every proposed Phase 8 intervention must map directly and traceably to one of these ten quantified failure modes.

---

## 2. Quantified Failure Modes

### Failure Mode 1: Extreme Majority Class Dominance (`GENERAL_SPEECH` ~95.2%)
- **Empirical Evidence:** In the 15-speaker TORGO evaluation corpus (16,554 total utterances), non-emergency speech (`GENERAL_SPEECH`) constitutes **95.2%** of all test tokens. Emergency intents are fractured across 6 rare minority classes (`CRITICAL_RESPONSE`, `FALL`, `BREATHING_MEDICAL`, `PAIN`, `SOS_HELP`, `LOCATION`), averaging $<0.8\%$ prevalence each.
- **Observed Failure:** Standard unweighted loss functions and unweighted soft-margin SVM formulations ($C=1.0$) maximize global accuracy ($94.8\%$) by collapsing the decision boundary, predicting `GENERAL_SPEECH` for $100\%$ of inputs.
- **Phase 8 Requirement:** Binary emergency triage separation must be decoupled from fine-grained emergency subtype classification to eliminate majority-class gradient domination.

### Failure Mode 2: Emergency Recall Degradation Under Sequential Intake
- **Empirical Evidence:** In Phase 7.4 (longitudinal natural accumulation across 15 speakers, 60 cells), unconstrained sequential adaptation caused Classical Emergency Recall to decline from **17.26% down to 9.97%** ($p = 0.0382$).
- **Observed Failure:** Because ~95% of incoming longitudinal speech is general conversation, continuous gradient updates without protected minority retention systematically overwrite emergency representations.
- **Phase 8 Requirement:** Protected emergency quota replay and asymmetric emergency loss weighting during personalization.

### Failure Mode 3: Dysarthric Speaker Emergency Recall Collapse
- **Empirical Evidence:** In severe dysarthric speakers (`M01`, `F01`, `M04`), phonetic distortion, hypernasality, and involuntary articulatory pauses cause acoustic feature vectors to drift into generic noise clusters. In Phase 7.8, Fixed Quantum Recall on dysarthric speakers collapsed to **3.84%** (versus 6.54% on control speakers).
- **Observed Failure:** Symmetrical distance metrics in Hilbert space fail to separate distorted atypical vowel prolongations from standard conversational pauses.
- **Phase 8 Requirement:** Speaker-specific acoustic centering and specialized emergency subspace projection.

### Failure Mode 4: The False-Emergency vs. Recall Trade-Off
- **Empirical Evidence:**
  - Continual Classical Adapter: Emergency Recall = **16.20%**, False Emergency Rate (FER) = **43.53%**.
  - Convex Hybrid Fusion: Emergency Recall = **7.07%**, FER = **19.45%**.
  - Fixed Quantum Kernel: Emergency Recall = **5.10%**, FER = **16.74%**.
- **Observed Failure:** Reducing FER via probability damping or conservative margins in Phase 7.10 directly crushed emergency recall by $>56\%$. The system achieved low false alarm rates solely by refusing to predict emergencies.
- **Phase 8 Requirement:** Hierarchical two-stage architecture where Stage 1 maximizes emergency recall via a high-sensitivity classical gate, and Stage 2 refines subtypes and suppresses false alarms without dropping true emergencies.

### Failure Mode 5: Fixed Quantum Kernel Structural Inflexibility
- **Empirical Evidence:** The Phase 4/5/7.8 Fixed Q8-R1 QSVC ($ZZ\text{FeatureMap}$, $q=8$, $1$ rep, linear entanglement) yielded identical predictions across all 81,840 noise test instances as a classical RBF-SVM with $C=1.0$ ($0.1390$ Macro-F1, $0.00\%$ IPER).
- **Observed Failure:** With zero trainable circuit parameters, the fixed quantum kernel matrix $\mathbf{K}_{ij} = |\langle \psi(x_i)|\psi(x_j)\rangle|^2$ cannot adapt its metric tensor to the atypical speaker's distorted geometry. Support vectors are monopolized by majority-class exemplars.
- **Phase 8 Requirement:** Trainable variational parameters specifically dedicated to learning an atypical emergency subspace, rather than static fidelity mapping.

### Failure Mode 6: Trainable Quantum Representation Optimization Collapse
- **Empirical Evidence:** In Phase 7.9, optimizing 16 variational rotation angles via Adam across 60 cells yielded Macro-F1 of **0.1187**, which was statistically indistinguishable from the fixed quantum kernel ($0.1180$, $p_{\text{adj}} = 0.061$). Emergency Recall remained stagnant at **5.81%** ($p_{\text{adj}} = 0.559$).
- **Observed Failure:** When trained end-to-end on a flat 7-class Cross-Entropy objective, the quantum circuit gradients for the 6 minority classes were swamped by the majority class gradients by a factor of $20:1$, freezing rotation angles near their initial values.
- **Phase 8 Requirement:** The quantum circuit must operate ONLY on emergency-candidate samples (Stage 2 conditional refinement) where classes are balanced, completely removing `GENERAL_SPEECH` from the quantum circuit's loss gradient.

### Failure Mode 7: Hybrid Convex Probability Fusion Attenuation
- **Empirical Evidence:** In Phase 7.10, convex probability fusion $P_{\text{hybrid}} = \alpha P_{\text{classical}} + (1-\alpha) P_{\text{quantum}}$ with $\alpha=0.5$ reduced Classical Emergency Recall from **16.20% down to 7.07%**.
- **Observed Failure:** Because the quantum branch was overconfident in `GENERAL_SPEECH` ($P_{\text{quantum}}(\text{GEN}) \approx 0.95$), linear probability averaging dragged the classical branch's emergency probabilities below the argmax decision threshold.
- **Phase 8 Requirement:** Residual feature-level or latent gating integration, rather than naive output probability averaging.

### Failure Mode 8: Replay Memory Capacity & Allocation Skew
- **Empirical Evidence:** In Phase 7.5, balanced replay memory with capacity 100 maintained forgetting at $\Delta = -0.0150$. However, across budgets $N=20 \to 50$, the number of verified emergency exemplars per speaker was frequently $\le 2$ samples per specific emergency subtype.
- **Observed Failure:** Extreme sample scarcity within individual emergency subtypes caused high parameter variance and overfitting during personalization.
- **Phase 8 Requirement:** Protected emergency exemplar oversampling and class-balanced minority memory slots.

### Failure Mode 9: Low-Resource Calibration Instability
- **Empirical Evidence:** Temperature scaling calibration in Phase 7.2 exhibited high estimator variance when validation sets contained fewer than 3 emergency samples.
- **Observed Failure:** Over-cautious temperature optimization led to widespread score suppression, pushing valid emergency intents into the low-confidence abstention zone ($<0.60$).
- **Phase 8 Requirement:** Prior-regularized Bayesian temperature scaling or pooled validation anchors.

### Failure Mode 10: Classical Simulation Constraints & Latency Overhead
- **Empirical Evidence:**
  - Classical Adapter: 25,223 parameters, **0.024s** inference latency.
  - Trainable Quantum VQC: 16 parameters, **4.331s** inference latency (180× slower).
- **Observed Failure:** Simulating full 384-dimensional quantum statevectors is computationally impossible ($\mathcal{O}(2^{384})$). Compressing 384-D Whisper embeddings down to 4 or 8 components via linear PCA discards $>50\%$ of feature variance.
- **Phase 8 Requirement:** Targeted compact emergency projection: Train a classical bottleneck encoder ($384 \to 8$) explicitly optimized for emergency intent separation before quantum circuit encoding, preserving life-critical variance within an 8-qubit budget.

---

## 3. Mapping Proposed Phase 8 Interventions to Failure Forensics

| Intervention Ladder Step | Proposed Scientific Mechanism | Target Failure Mode Mapped | Expected Failure Resolution |
| :--- | :--- | :--- | :--- |
| **Track A: Strengthened Classical Baseline** | Hierarchical 2-Stage Architecture (Stage 1: Binary Emergency Detector; Stage 2: 6-Class Emergency Classifier) + Class-Weighted Focal Loss | Failure Modes 1, 2, 4 | Decouples emergency detection from majority dominance; prevents recall collapse. |
| **Track B: Quantum Residual Subtype Module** | 8-Qubit Trainable Quantum Feature Map operating conditionally ONLY on Stage 1 Emergency Positives to refine subtype classification | Failure Modes 5, 6, 7, 10 | Completely isolates quantum gradients from majority class; operates on balanced minority space. |
| **Track C: Matched Classical Shadow Model** | Identical 2-Stage pipeline where Stage 2 is a parameter-matched Classical MLP operating on the same 8-D bottleneck representation | Fair Comparison Standard | Directly isolates whether the quantum Hilbert representation adds measurable benefit over classical nonlinear mapping. |

This forensic audit forms the immutable scientific foundation for Phase 8.
