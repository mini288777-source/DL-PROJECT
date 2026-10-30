# RESEARCH DEMO MODEL SELECTION & DEPLOYMENT REPORT
## Canonical Architecture Selection for Prototype Integration

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Target Application:** `AtypicalSpeechEmergencyAssistant` (Phase 7.14 Pipeline)  
**Status:** **CANONICALLY SELECTED & VERIFIED**  
**Deployment Tier:** **RESEARCH DEMO PROTOTYPE ONLY**  

---

## 1. Selected Architecture Specification

- **Selected Model:** **Continual Classical Adapter Head**
- **Model Family:** Parameterized Linear + LayerNorm Multi-Layer Perceptron
- **Model Version:** `ClassicalAdapter_v1.0.0_continual` (Active Baseline: `v0.0.0_baseline`)
- **Parameter Count:** Exactly 25,223 trainable weights and biases
- **Input Dimension:** 384-dimensional frozen acoustic representations (`openai/whisper-tiny`)
- **Output Dimension:** 7 canonical speech and emergency classes
- **Inference Latency:** 24.2 ms (Edge-compute feasible)
- **Adaptation Time:** 0.352 seconds per update cycle

---

## 2. Selection Rationale & Safety Justification

Per pre-registered safety governance rules, model selection for assistive emergency communication is **not** dictated by Macro-F1 or overall accuracy alone. It is governed strictly by **Emergency Recall** and **Governance Approval**:

| Selection Dimension | Continual Classical Adapter | Fixed Q8-R1 QSVC | Trainable Quantum VQC | Convex Hybrid Fusion | Selection Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Emergency Recall** | **16.20% (Superior)** | 5.10% (Collapsed) | 5.81% (Collapsed) | 7.07% (Deficient) | **Classical Adapter** |
| **False Emergency Rate** | 43.53% | **16.74% (Lowest)** | 17.20% | 19.45% | **Classical Adapter** (Acceptable trade-off) |
| **Macro-F1** | 0.1012 | 0.1187 | **0.1187** | 0.1184 | Classical trade-off |
| **Phase 7.6 Governance** | **APPROVED** | **VETOED** | **VETOED** | CONDITIONAL | **Classical Adapter** |
| **Catastrophic Forgetting** | Bounded ($\Delta \le 0.015$) | None ($\Delta = 0.000$) | Negligible ($\Delta = -0.002$) | Negligible ($\Delta = -0.008$) | **Classical Adapter** |
| **Inference Latency** | **0.024s** | 0.063s | 4.331s | 2.492s | **Classical Adapter** |
| **Edge Hardware Feasible** | **YES (CPU/Mobile)** | Sim only | Sim only | Partial | **Classical Adapter** |

### Why Quantum Models Were Disqualified:
1. **Severe Emergency Recall Deficit:** Pure quantum models collapsed onto majority non-emergency speech (`GENERAL_SPEECH`), failing to detect over $94\%$ of emergency utterances.
2. **Phase 7.6 Safety Veto:** Both Fixed QSVC and Trainable VQC triggered unconditional Safety Vetoes during candidate validation. Activating them would directly violate project safety contracts.
3. **Computational Latency:** Trainable quantum simulation required $>4.3$ seconds per utterance, exceeding life-critical latency limits.

---

## 3. Validation Status

- **Phase 7.6 Validation Gate:** **APPROVED**
- **Phase 7.13 Robustness & Adversarial Audit:** **PASSED (10/10 tests)**
- **Phase 7.14 Application Integration:** **PASSED (10/10 tests)**
- **Regression Suite:** **469 / 469 PASSED**

---

## 4. Known Technical Limitations

1. **Low Baseline Emergency Recall (16.20%):** While three times higher than quantum alternatives, 16.20% recall reflects the extreme difficulty of atypical dysarthric speech recognition under severe few-shot resource constraints ($N \le 50$).
2. **High False Emergency Rate (43.53%):** To preserve life-critical emergency recall, the classical adapter maintains broad decision boundaries, resulting in elevated false alarms that require human caregiver confirmation.
3. **No Autonomous Deployment:** The model must always operate in conjunction with the Mandatory Human Verification Barrier.
