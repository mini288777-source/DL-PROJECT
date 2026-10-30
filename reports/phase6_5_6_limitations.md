# PHASE 6.5.6 — EXPLICIT RESEARCH LIMITATIONS & OPERATIONAL BOUNDARIES

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Team:** kernal  
**Scope:** Exhaustive Audit of Experimental, Methodological, Clinical, and Computational Constraints  
**Status:** COMPLETE & FROZEN  

---

## 1. Operational Readiness & Certification Status

To prevent scientific overclaiming and ensure absolute transparency regarding deployment viability, the operational status of the evaluated models is formally declared as follows:

| Assessment Dimension | Official Operational Designation | Detailed Scientific Status |
| :--- | :---: | :--- |
| **System Deployment Readiness** | **NOT READY** | The system fails safety-critical deployment criteria: emergency precision is $\le 2.03\%$, missed emergency rate is $\ge 54.19\%$, and false-emergency rate is $\ge 28.12\%$. |
| **Clinical Validation** | **NOT ESTABLISHED** | No clinical trials, hospital evaluations, or assistive technology patient certifications have been conducted. |
| **Real-World Emergency Reliability** | **NOT ESTABLISHED** | The system has not been tested in live emergency scenarios, unconstrained acoustic environments, or during acute medical crises. |
| **Hardware Quantum Advantage** | **NOT ESTABLISHED** | All quantum experiments were performed via exact statevector simulation on classical CPUs; no physical QPU hardware was evaluated. |
| **Generalization Beyond TORGO** | **NOT ESTABLISHED** | Evaluation is strictly bounded to the 15 dysarthric speakers of the TORGO database; performance on other etiologies (e.g. ALS, stroke, Parkinson's) is unknown. |

---

## 2. Exhaustive Listing of the 15 Core Limitations

### 1. Calibration Sample Budget Constrained to $N \le 50$
Personalization was evaluated strictly under low-resource calibration budgets $N \in \{20, 30, 40, 50\}$ utterances per speaker. While this reflects realistic clinical constraints on patient recording fatigue, it severely limits the empirical capacity of both classical and quantum classifiers to learn complex multi-class decision boundaries.

### 2. Single-Corpus Centered Evaluation (TORGO Database)
All experiments were conducted exclusively on the TORGO database of dysarthric speech. TORGO is an established research benchmark, but its recording protocol (head-mounted microphones and structured prompt elicitation) does not reflect noisy real-world ambient conditions.

### 3. Limited Common Emergency Classes
Of the six non-general emergency classes defined in the taxonomy, only four (`CRITICAL_RESPONSE`, `FALL`, `BREATHING_MEDICAL`, and `SOS_HELP`) possessed adequate speaker representation. Two classes—`LOCATION` and `PAIN`—suffered from severe corpus-level data scarcity, appearing in very few utterances across the 15 speakers.

### 4. Severe Intrinsic Class Imbalance
The TORGO dataset exhibits naturalistic, extreme class skew: `GENERAL_SPEECH` constitutes approximately $95.10\%$ of all utterances, while emergency utterances represent only $\approx 4.90\%$. This skew severely depresses precision metrics and induces strong priors toward majority prediction.

### 5. Low Emergency Prevalence at Test Time
In realistic assistive settings, true emergency calls occur infrequently. Under an emergency base rate of $\approx 5\%$, even a strong false-positive rate of $10\%$ results in more false alarms than true emergencies, creating an inherent precision ceiling ($\le 2.03\%$ across all evaluated models).

### 6. Synthetic AWGN as the Sole Noise Stressor
The original robustness analysis (Phase 5) evaluated acoustic degradation strictly using Additive White Gaussian Noise (AWGN) at Clean, 15 dB, 10 dB, and 5 dB SNR. Real-world emergency environments involve non-stationary, reverberant, and multi-speaker background noise, which were not tested.

### 7. Frozen Upstream Acoustic Encoder (Whisper-Tiny)
Acoustic embeddings were extracted from a frozen, pre-trained Whisper-tiny encoder ($384$ dimensions) without end-to-end speech fine-tuning. The downstream classifiers were therefore entirely dependent on the acoustic separability learned by Whisper on standard speech.

### 8. Simulated Quantum Statevector Execution
All quantum kernel computations were performed using classical statevector linear algebra ($K_{ij} = |\langle\psi(x_i)|\psi(x_j)\rangle|^2$) via Qiskit. No physical quantum circuit sampling noise (shot noise), gate infidelities, or decoherence effects were modeled.

### 9. Absence of Physical Quantum Hardware Execution
No physical Quantum Processing Unit (QPU) was utilized. Consequently, claims regarding quantum speedup, physical circuit latency, quantum error mitigation, or NISQ hardware resilience cannot be made.

### 10. Lack of Clinical Validation
The research protocol did not include patient-in-the-loop clinical validation, clinician oversight, speech-language pathologist (SLP) error auditing, or user experience (UX) evaluations with dysarthric individuals.

### 11. Absence of Real-World Emergency User Trials
The models were evaluated offline on pre-recorded audio prompts. No trials were conducted with actual emergency dispatchers, 911/112 telecommunicators, or active personal emergency response systems (PERS).

### 12. Lack of Safety Certification
The pipeline has not undergone regulatory safety auditing (e.g. FDA, CE Medical Device, ISO 13485) required for life-critical assistive medical technology.

### 13. Protocol Discontinuity Between Original and Recovery Tracks
The recovery interventions (speaker stratification, cost-sensitive weighting, hierarchical routing, and PCA dimensionality expansion) represent a modified experimental protocol (Track B) distinct from the original benchmark (Track A). The two tracks cannot be analyzed as a single matched statistical continuum.

### 14. Severe Speaker-Level Acoustic Heterogeneity
TORGO speakers display wide variation in dysarthria severity (ranging from mild to severe cerebral palsy and ALS). Model performance varied widely across individuals (Macro-F1 ranged from $0.04$ to $0.22$), indicating that a universal few-shot architecture does not serve all severity levels equally.

### 15. Small Cohort Size Relative to Clinical Scale
The entire experimental corpus comprises $15$ speakers ($8$ dysarthric, $7$ control). While standard for atypical speech research due to data collection barriers, $15$ subjects represent a small sample size for establishing clinical efficacy across diverse populations.
