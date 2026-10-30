# Gate C: Confirmatory Evaluation Preregistration Protocol (Draft)

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Identifier:** PREREG-PHASE-C-2026-V1  
**Status:** **DRAFT / UNFROZEN / PENDING INSTITUTIONAL APPROVAL**  
**Lead Authors:** Independent Clinical-ML Validation Lead, Lead Biostatistician  
**Target Registry:** Open Science Framework (OSF) / ClinicalML Registry  

---

## 1. Study Rationale & Primary Research Questions

### 1.1 Study Context
Atypical speech resulting from motor speech disorders (dysarthria) poses profound challenges for standard automatic speech recognition (ASR). This confirmatory study evaluates whether low-resource personalized adapter models can reliably recognize emergency communication intents from an untouched, external atypical speech cohort.

### 1.2 Explicit Non-Interventional Scope
> [!IMPORTANT]
> This evaluation is strictly non-interventional and diagnostic. No predictions will be used to trigger alarms, inform emergency dispatch, or guide clinical care. This study does NOT assess or establish clinical safety or regulatory efficacy.

### 1.3 Hypotheses
1. **Primary Hypothesis $H_1$:** On an independent, untouched dysarthric speech cohort, the personalized classical adapter achieves an Emergency Intent Recall $\ge 70.0\%$ while maintaining a False Emergency Rate (FER) $\le 15.0\%$ when operated at a threshold calibrated strictly on development data.
2. **Null Hypothesis $H_0$:** The personalized adapter fails to achieve Emergency Intent Recall $\ge 70.0\%$ or incurs FER $> 15.0\%$.
3. **Exploratory Hypothesis $H_2$ (Quantum vs Classical):** A hybrid quantum-classical neural network architecture provides no statistically significant advantage in Emergency Recall or Macro-F1 over an identically parameterized classical architecture (consistent with Phase 6.5.6 findings).

---

## 2. Frozen System Specification

Before any confirmatory data are unlocked, the evaluation system components are cryptographically frozen:

```mermaid
flowchart LR
    A[Acoustic Audio: 16 kHz Mono] --> B[Feature Extraction: 80-bin Mel]
    B --> C[Frozen Pretrained Encoder: wav2vec2-base]
    C --> D[Frozen Classifier Adapter: Phase 7.5 Classical]
    D --> E[Decision Logic: Prespecified tau]
    E --> F[Class Intent Output & Audit Log]
```

### 2.1 Acoustic Preprocessing Pipeline
- **Sampling Rate:** Exactly 16,000 Hz, single channel (mono), 16-bit linear PCM.
- **Acoustic Representation:** 80-dimensional log-mel filterbank energies computed over 25ms Hamming windows with a 10ms frame shift.
- **Normalization:** Utterance-level cepstral mean and variance normalization (CMVN).

### 2.2 Model Architecture & Weights
- **Base Acoustic Encoder:** Pretrained self-supervised acoustic encoder (`facebook/wav2vec2-base`), locked frozen weights.
- **Personalized Adapter:** Lightweight speaker-adapted bottleneck linear projection layer.
- **Candidate Checkpoint Registry Status:**
  - *Phase 7.5 Served Adapter:* Materialized on disk at `models/continual/phase7_5/<SPEAKER>/classical_adapter/<VERSION>/model_checkpoint.pt`.
  - *Phase 8.1 / Phase 8.2 Candidate Checkpoints:* Currently **UNMATERIALIZED** on disk (returning HTTP 503 `MODEL_NOT_READY`).
  - *Preregistration Rule:* Confirmatory evaluation may ONLY evaluate materialized checkpoints whose SHA-256 checksums are registered in this protocol prior to execution.

### 2.3 Intent Class Taxonomy (10 Classes)
Referenced from immutable config `config/taxonomy.yaml` (SHA-256: `3664d8689df0f30bc987cd4de5ddeb80f11f1e3b5fb672243e902f6959e92bec`):
- **Emergency Classes ($C_{emergency} = 4$):**
  1. `medical_emergency` (e.g., "help", "cannot breathe", "chest pain")
  2. `fire_emergency` (e.g., "fire", "smoke", "burning")
  3. `security_emergency` (e.g., "intruder", "break in", "danger")
  4. `accident_emergency` (e.g., "fallen", "can't get up", "trapped")
- **Non-Emergency Classes ($C_{non\_emergency} = 6$):**
  5. `call_nurse`
  6. `pain_report`
  7. `bathroom_request`
  8. `water_request`
  9. `general_communication`
  10. `control_command`

---

## 3. Prespecified Decision Policy & Threshold Discipline

### 3.1 Uncalibrated Score Disclosure
- The raw output score $s(x) \in [0, 1]$ represents an uncalibrated classifier score, NOT a true posterior probability.
- Softmax outputs shall be labeled as **"uncalibrated model score"** in all evaluation tables and logs.

### 3.2 Decision Threshold $\tau$ Rules
- The decision policy classifies an utterance $x$ as an Emergency if:
  $$\hat{Y}_{emergency} = \begin{cases} 1 & \text{if } \max_{c \in C_{emergency}} s_c(x) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$
- **Demonstration Default:** The Phase 7.5 demonstration default $\tau = 0.6000$ remains **`THRESHOLD_PROVENANCE_UNVERIFIED`** for safety-critical classification.
- **Strict Protocol Constraint:** The operational threshold $\tau$ MUST be selected exclusively on an untouched development split of the external cohort ($D_{dev}$). **Under no circumstances may $\tau$ be selected, tuned, or modified using the locked confirmatory evaluation split ($D_{confirmatory}$).**

---

## 4. Prespecified Primary and Secondary Endpoints

### 4.1 Primary Endpoints
1. **Emergency Intent Recall (Sensitivity):**
   $$\text{Recall}_{em} = \frac{TP_{em}}{TP_{em} + FN_{em}}$$
2. **False Emergency Rate (FER / False Alarm Rate):**
   $$\text{FER} = \frac{FP_{em}}{N_{non\_emergency}}$$

### 4.2 Secondary Endpoints
1. **Emergency Precision (Positive Predictive Value):**
   $$\text{Precision}_{em} = \frac{TP_{em}}{TP_{em} + FP_{em}}$$
2. **False Negative Rate (Miss Rate):**
   $$\text{FNR}_{em} = 1 - \text{Recall}_{em} = \frac{FN_{em}}{TP_{em} + FN_{em}}$$
3. **Macro-Averaged F1 Score (10 Classes):**
   $$\text{Macro-F1} = \frac{1}{10} \sum_{c=1}^{10} F1_c$$
4. **Intent Recognition Accuracy vs Transcription WER:**
   - Word Error Rate (WER) and Character Error Rate (CER) will be evaluated separately from Intent Classification Accuracy.
   - Acceptable transcription quality will NOT be construed as proof of emergency intent recognition.

---

## 5. Preregistration Integrity Safeguards

1. **Immutable Cryptographic Hash:** Prior to evaluation, the final YAML representation of this protocol will be hashed with SHA-256 and committed to the repository ledger.
2. **Single-Pass Evaluation:** The offline runner (`src/evaluation/confirmatory_runner.py`) enforces a single-pass execution. Any rerun, parameter adjustment, or post-hoc threshold change will be flagged in the public audit report as a protocol deviation.
3. **Negative Result Reporting:** If the system achieves Emergency Recall $< 70.0\%$ or FER $> 15.0\%$, the study will publish a formal negative result statement without post-hoc rationalization.
