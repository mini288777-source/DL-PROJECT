# Research Limitations and Intended Use

## Project: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
**Document Reference:** `docs/application/RESEARCH_LIMITATIONS_AND_INTENDED_USE.md`  
**Classification:** Scientific Governance, Risk Assessment & Regulatory Scope  
**Effective Date:** 2026-10-11  

---

## 1. Intended Use and Scope Boundaries

### 1.1 Non-Clinical Research Demonstration
This application is exclusively an **experimental research prototype and technical demonstration platform**. It is designed for offline benchmarking, acoustic feature inspection, and architectural analysis of classical and quantum neural adapters on atypical speech representations.

### 1.2 Prohibited Use Cases
- **NOT A MEDICAL DEVICE:** This software is not certified, cleared, or approved as a medical device by the United States Food and Drug Administration (FDA), European Medicines Agency (CE-MDR), Health Canada, or any other global medical device regulatory authority.
- **NOT AN EMERGENCY RESPONSE TOOL:** This software must **never** be deployed, configured, or relied upon for life-critical, acute emergency, or urgent safety communications. It possesses zero integration with emergency dispatch systems (911/CAD, PSAP), hospital alerting systems, or clinical monitors.
- **NOT FOR CLINICAL DIAGNOSIS OR CARE:** It is not intended for patient monitoring, medical triage, diagnostic assessment of dysarthria severity, or acute assistive communication in unmonitored environments.

---

## 2. Core Scientific Findings & Evidence Summary

### 2.1 The "No Quantum Advantage" Conclusion
Throughout rigorous empirical evaluations across Phases 3, 4, 5, 6, 7, and 8, the scientific conclusion remains unequivocal:
> **There is no statistically supported quantum advantage under the tested protocol.**

- In head-to-head comparisons on identical dysarthric acoustic representations, hybrid quantum-classical networks (incorporating parameterized quantum circuits via PennyLane/Qiskit) demonstrated parity or slight deficit relative to purely classical feed-forward neural networks of comparable or smaller parameter capacity.
- Quantum simulation added significant computational overhead and latency without yielding statistically significant improvements in Macro-F1, Emergency Recall, or Frame Error Rate (FER).

### 2.2 Phase 8.1 Safety Recovery Metrics
In offline evaluations conducted during Phase 8.1 (Safety Recovery Protocol):
- **Emergency Class Recall:** Ranged between **10.13% and 19.83%**.
- **Emergency Class Precision:** Ranged between **7.45% and 8.98%**.
- **Clinical Meaning:** Over 80% of simulated emergency utterances were missed (false negatives), and over 90% of emergency alerts were spurious (false positives). These error rates confirm that current models are completely unsuitable for real-world life-critical use.

### 2.3 Phase 7.5 Personalization Baseline (Speaker F01)
For the served Phase 7.5 continual learning adapter on speaker `F01`:
- Reported Frame Error Rate (FER): **20.31%**.
- While personalized few-shot adaptation demonstrated measurable reduction in catastrophic forgetting compared to unadapted baselines, substantial acoustic variance and misclassifications persist.

---

## 3. Class Imbalance & Evaluation Metric Semantics

### 3.1 The Fallacy of Overall Accuracy
In atypical and dysarthric speech datasets (such as TORGO), speech distributions are drastically imbalanced:
- The vast majority of spoken tokens represent everyday conversational words (`GENERAL_SPEECH`).
- Urgent emergency intents (`SOS_HELP`, `BREATHING_MEDICAL`, `FALL`, `PAIN`, `CRITICAL_RESPONSE`) constitute a tiny fraction ($< 5\%$) of total recording hours.
- **Critical Risk:** A naive model that classifies 100% of inputs as `GENERAL_SPEECH` can achieve $> 95\%$ overall accuracy while failing 100% of emergency life-safety events. Therefore, **overall accuracy is rejected as a valid safety metric**.

### 3.2 Authoritative Metric Definitions

| Metric Name | Mathematical Definition | Clinical / Research Interpretation |
| :--- | :--- | :--- |
| **Emergency Recall** | $\frac{\text{TP}_{\text{emerg}}}{\text{TP}_{\text{emerg}} + \text{FN}_{\text{emerg}}}$ | Proportion of actual emergency calls successfully detected. In life safety, low recall means patients in danger receive no help. |
| **Emergency Precision** | $\frac{\text{TP}_{\text{emerg}}}{\text{TP}_{\text{emerg}} + \text{FP}_{\text{emerg}}}$ | Proportion of emergency alarms that are genuine. Low precision causes alarm fatigue, wasted first-responder resources, and user abandonment. |
| **Macro-F1** | $\frac{1}{C} \sum_{c=1}^C F1_c$ | Unweighted average of F1 scores across all 7 taxonomy classes, treating minority emergency classes equally with majority general speech. |
| **Frame Error Rate (FER)** | $\frac{\text{Incorrect Frames}}{\text{Total Frames}}$ | Proportion of acoustic frame tokens misclassified across temporal sequences. |
| **Abstention Rate** | $\frac{\text{Samples with } \max(P) < \tau}{\text{Total Samples}}$ | Fraction of inputs where model confidence is insufficient, prompting a safe `ABSTAIN` request for clarification. |

---

## 4. Operating Threshold ($\tau = 0.6000$) Policy & Notice

### 4.1 Provenance of Demonstration Cutoff
The serving application defaults to a decision cutoff of $\tau = 0.6000$:
> **Notice: $\tau = 0.6000$ is an uncalibrated demonstration default — threshold provenance is unverified for this served model.**

- This value is not derived from ROC convex hull analysis, risk-sensitive cost-matrix optimization, or prospective clinical calibration.
- It is supplied purely to demonstrate the frontend abstention state machine and visual UI feedback.

### 4.2 Decoupling from the 15% FER Research Constraint
Research reports reference an internal target ceiling of $15\%$ False Escalation Rate (FER). 
- The 15% FER limit is an **internal exploratory optimization bound** utilized in offline training sweeps.
- It does **not** represent a regulatory standard, a patient-safety threshold, or proof of clinical efficacy.
- The runtime parameter $\tau = 0.6000$ is completely decoupled from the 15% FER research ceiling.

---

## 5. Gate Governance Status: Gates C and D Blocked

The project enforces strict phase-gate governance. Under no circumstances may application settings or user actions alter gate status to `PASS`.

### 5.1 Gate C (Controlled User Pilot) — BLOCKED
Gate C authorizes testing with human participants in controlled environments. It remains **STRICTLY BLOCKED** due to outstanding blockers:
1. Absence of Institutional Review Board (IRB) or Independent Ethics Committee (IEC) approval.
2. Lack of signed Data Use Agreements (DUAs) for multi-institution dysarthric cohorts.
3. Absence of formal prospective clinical oversight and qualified medical supervision.
4. Incomplete risk management documentation under ISO 14971 standards.

### 5.2 Gate D (Clinical & Emergency Deployment) — BLOCKED
Gate D authorizes deployment in live healthcare, residential, or emergency communication workflows. It remains **STRICTLY BLOCKED** due to:
1. High emergency false-negative rate ($> 80\%$) and false-positive rate ($> 90\%$).
2. Total absence of FDA 510(k) clearance or De Novo medical device authorization.
3. Zero integration with PSAP / CAD / 911 technical standards (NENA i3, NG911).
4. Failure to satisfy clinical safety tolerances for life-critical assistive technology.
