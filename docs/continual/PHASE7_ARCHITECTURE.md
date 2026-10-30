# PHASE 7 ARCHITECTURE SPECIFICATION
## Closed-Loop Continual Learning & Verified Personalization Framework

---

### 1. Architectural Overview & Design Philosophy

Phase 7 establishes a controlled, verified continual-learning architecture for low-resource personalized atypical speech recognition. The fundamental problem addressed in this extension is:

> *Static few-shot models calibrated on a tiny initial sample budget ($N \in \{20, 30, 50\}$) remain bounded by their initial data starvation. Can accumulated, verified user interactions over time safely expand speaker-specific support without suffering from confirmation bias, catastrophic forgetting, or runaway false alarms?*

To prevent catastrophic failure modes common to naïve continual learning, the architecture implements a **strict closed-loop firewall**:

```
                         [ USER AUDIO INPUT ]
                                  │
                                  ▼
                [ FROZEN WHISPER-TINY ENCODER (384-D) ]
                                  │
                                  ▼
               [ HIERARCHICAL PERSONALIZATION ENGINE ]
                     ├── Stage 1: Emergency Gate
                     └── Stage 2: Subtype Router
                                  │
                                  ▼
               [ PREDICTION + CALIBRATED CONFIDENCE ]
                                  │
                                  ▼
                 [ ELIGIBILITY & VERIFICATION GATE ]
             (Novelty / Cleanness / User Confirmation)
                                  │
                                  ▼
               [ IMMUTABLE EXPERIENCE REPLAY BUFFER ]
                                  │
                                  ▼
                [ CANDIDATE MODEL OFFLINE UPDATE ]
                                  │
                                  ▼
                 [ PROTECTED MULTI-METRIC GATE ]
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
             [ PROMOTION AS ACTIVE ]    [ REJECTION / ROLLBACK ]
```

---

### 2. Core Architectural Subsystems

#### Subsystem A: Frozen Representation Invariant
- **Model:** `openai/whisper-tiny`
- **Output:** 384-dimensional bottleneck mean-pooled embedding.
- **Contract:** Strictly frozen, immutable, evaluation-mode only. No gradient backpropagation into the Whisper backbone is permitted.
- **Normalization:** Train-only unit-L2 normalization.

#### Subsystem B: Active Personalization Layer
- The inference engine operates the verified Two-Stage Hierarchical structure established in Phase 6.5.3:
  - **Stage 1 (Binary Emergency Gate):** Distinguishes `GENERAL_SPEECH` from `EMERGENCY`.
  - **Stage 2 (Conditional Subtype Router):** Routes confirmed emergency utterances among feasible emergency intents.
- **Model Families Evaluated in Future Tracks:**
  - Classical Adapter (MLP)
  - Classical Support Vector Classifier (RBF Kernel)
  - Quantum Support Vector Classifier (QSVC)
  - Trainable Quantum Variational Classifier (Future Track E)

#### Subsystem C: Uncertainty Quantification & Confidence Assessment
- For every inference query, the system extracts:
  - Raw class logits / decision function distances.
  - Softmax calibrated probability $\hat{p} \in [0.0, 1.0]$.
  - Margin uncertainty: difference between top-1 and top-2 predicted class probabilities.
  - Distance from nearest calibration support vector.

#### Subsystem D: The Anti-Self-Training Firewall (Eligibility Gate)
The system strictly enforces the principle:
$$\text{Prediction} \neq \text{Ground Truth}$$

A sample is **strictly disqualified** from automatic admission to experience memory if:
1. Confidence $\hat{p}$ is below the high-precision eligibility threshold.
2. Prediction conflicts with recent temporal stability checks.
3. Acoustic SNR falls below the quality cutoff ($< 10\text{ dB}$).
4. Representation distance reveals extreme out-of-distribution drift.

**Admission Pathways:**
- `AUTO_ELIGIBLE`: High-confidence non-emergency samples in uncontentious feature domains.
- `USER_CONFIRMED`: Explicit feedback provided by the speaker or caregiver (e.g., confirming "Yes, I requested assistance" or dismissing a false alarm).
- `QUARANTINED`: Suspicious, ambiguous, or noisy samples isolated for offline clinical review.

#### Subsystem E: Experience Replay Memory
- Implements a speaker-specific replay buffer that stores verified experience records.
- Enforces class balancing to prevent emergency class dilution by ubiquitous general conversational speech (~95% background prevalence).
- Retains initial seed adaptation samples ($N_{\text{initial}}$) permanently to anchor baseline generalization.

#### Subsystem F: Protected Validation & Candidate Promotion Gate
- Updating a model occurs strictly **offline** in an isolated sandbox.
- Before a candidate model $M_{k+1}$ can replace the active model $M_k$, it must undergo rigorous benchmarking against:
  1. A permanently sequestered, protected validation set $D_{\text{val}}$ never seen in experience memory.
  2. The current active model $M_k$ in paired testing.
- **Gate Criteria:** The candidate is **REJECTED** if:
  - Emergency Recall degrades on $D_{\text{val}}$.
  - False-Emergency Rate increases beyond acceptable bounds.
  - End-to-End Macro-F1 degrades.

---

### 3. Future Experimental Tracks (Roadmap)

| Track | Family | Description | Future Phase |
| :--- | :--- | :--- | :--- |
| **Track A** | Classical | Fixed classical baseline (Frozen Phase 6.5 Adapter/RBF) | Phase 7.2 |
| **Track B** | Classical | Continual classical personalization via verified replay | Phase 7.3 |
| **Track C** | Quantum | Fixed QSVC baseline (Frozen Phase 6.5 Q8-R1) | Phase 7.4 |
| **Track D** | Quantum | Continual QSVC (Support vector set expansion) | Phase 7.5 |
| **Track E** | Quantum | Trainable quantum representation (Parameterized VQC) | Phase 7.6 |
| **Track F** | Hybrid | Classical-quantum co-adaptation | Phase 7.7 |

*Note: Phase 7.0 executes zero experimental runs. All tracks are future comparative milestones.*
