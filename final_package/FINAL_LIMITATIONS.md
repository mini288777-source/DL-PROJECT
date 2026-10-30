# EMPIRICAL LIMITATIONS & SCIENTIFIC BOUNDARIES

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 2026  

---

## 1. Acoustic & Population Limitations

1. **Atypical Dysarthric Speech Degradation:**
   - Dysarthric speech exhibits severe phonetic distortion, hypernasality, temporal variability, and irregular phonation.
   - Even with modern foundation encoders (`openai/whisper-tiny`), baseline recognition capacity remains low across severe speakers (`M01`, `F01`, `M04`), where acoustic variance is orders of magnitude higher than neurotypical speech.

2. **Extreme Class Imbalance:**
   - In longitudinal assistive corpora, non-emergency conversational utterances (`GENERAL_SPEECH`) represent $>95\%$ of all acoustic samples.
   - Emergency utterances (`SOS_HELP`, `FALL`, `PAIN`, `BREATHING_MEDICAL`) are inherently rare. This extreme skew induces severe majority-class collapse in unweighted optimization objectives and kernel machines.

3. **Low Absolute Emergency Recall (16.20% Classical / 5.10% Quantum):**
   - While the Classical Adapter achieves three times the recall of quantum baselines, an absolute recall of 16.20% means that ~84% of emergency utterances are not captured on the first forward pass.
   - This necessitates operational human verification, audio prompts, and confirmation dialogues rather than autonomous emergency dispatch.

---

## 2. Quantum Computing & Algorithmic Limitations

1. **Analytical Statevector Simulation (No Physical QPU Noise):**
   - All quantum circuits were evaluated using PennyLane's `default.qubit` exact statevector simulator.
   - While this eliminated physical gate noise, decoherence, and shot noise (providing the most mathematically optimistic setting for quantum algorithms), physical NISQ hardware would introduce additional gate infidelity and readout errors.

2. **Barren Plateaus & Expressibility in Low Qubit Regimes:**
   - Due to the classical feature dimension of Whisper embeddings (384-D), PCA dimensionality reduction ($384 \to 8$ or $384 \to 4$) was required to simulate quantum circuits within tractable classical memory.
   - Compressing 384 dimensions into 4 or 8 components discards $>50\%$ of the acoustic feature variance, placing quantum circuits at an unavoidable information-theoretic disadvantage relative to classical adapters that operate directly on all 384 dimensions.

3. **Inference Latency Overhead:**
   - Quantum kernel matrix construction scales quadratically with dataset size $\mathcal{O}(N^2)$, and variational gradient optimization on statevectors required $>4.3$ seconds per inference.
   - This latency is incompatible with real-time edge deployment on assistive embedded devices, whereas the classical adapter executes in $0.024$ seconds.

---

## 3. Dataset & Cohort Scope

1. **TORGO Database Scale:**
   - Evaluations were conducted across 15 speakers (8 dysarthric, 7 control) comprising 16,554 total utterances.
   - While TORGO is the benchmark gold standard for dysarthric speech research, larger cohorts across diverse etiologies (e.g., ALS, Parkinson's, traumatic brain injury) will be needed for broader generalizability.

2. **Controlled Protocol Constraints:**
   - Evaluations followed strict pre-registered Protocols (Protocol B clean-to-noisy and longitudinal budget increments $N \in \{20, 30, 40, 50\}$). Performance under unconstrained conversational settings remains subject to ongoing investigation.

---

## 4. Clinical & Regulatory Disclaimer

- **RESEARCH PROTOTYPE ONLY:**
  - This system has **NOT** undergone medical clinical trials, institutional review board (IRB) patient validation, or regulatory clearance (FDA 510(k), CE-MDR).
  - It must **NEVER** be deployed as an autonomous, unmonitored medical dispatch system.
