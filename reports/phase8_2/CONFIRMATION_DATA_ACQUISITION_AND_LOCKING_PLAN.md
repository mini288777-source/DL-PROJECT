# PHASE 8.2 PROSPECTIVE CONFIRMATION DATA ACQUISITION AND LOCKING PLAN

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document:** `reports/phase8_2/CONFIRMATION_DATA_ACQUISITION_AND_LOCKING_PLAN.md`  
**Protocol Reference:** [`config/phase8_2/confirmation_protocol.yaml`](file:///d:/DL/config/phase8_2/confirmation_protocol.yaml)  
**Governance Status:** **FROZEN PROSPECTIVE READINESS SPECIFICATION — DATA ACQUISITION PENDING**

---

## 1. Purpose and Governance Authority

Because all speech data currently in the repository (the TORGO corpus) have been utilized in historical model development, threshold selection, or diagnostic benchmarking, **no independent confirmation dataset is available**.

Under Section 4 of `PHASE8_2_INDEPENDENT_CONFIRMATORY_EVALUATION.md`, this document establishes the binding, prospective data-acquisition and locking protocol required before any future confirmatory evaluation can be executed.

---

## 2. Eleven-Point Prospective Acquisition & Locking Specification

### 2.1 Target Population and Inclusion/Exclusion Criteria
- **Inclusion Criteria**:
  - Adult individuals (aged $\ge 18$) with medically diagnosed dysarthria resulting from cerebral palsy, amyotrophic lateral sclerosis (ALS), Parkinson's disease, or stroke.
  - Certified clinical speech-language pathologist (SLP) assessment documenting dysarthria severity (mild, moderate, severe) using Frenchay Dysarthria Assessment (FDA) or equivalent clinical scale.
  - Matched control cohort of neurotypical adult speakers with no history of motor speech disorder.
- **Exclusion Criteria**:
  - Individuals who contributed speech samples to the TORGO database or any public acoustic database previously utilized in model training.
  - Profound cognitive impairment preventing informed consent or structured phrase elicitation.
  - Environmental noise floor exceeding 45 dBA during recording.

### 2.2 Speaker, Session, and Recording Identifiers
- Every speaker will be assigned an anonymized unique identifier (`CONF_SPK_XXX`).
- Identifier format must strictly encode cohort:
  - `CONF_DY_XXX` (Dysarthric speaker)
  - `CONF_CT_XXX` (Control speaker)
- Recording session metadata must record microphone channel (headset condenser vs. acoustic array), acoustic environment, session date, and utterance sequence index (`CONF_SPK_XXX_S01_U0001`).

### 2.3 Consent, Authorization, Governance, and Secure Storage
- Institutional Review Board (IRB) / Research Ethics Committee (REC) protocol approval must be obtained prior to participant recruitment.
- Written informed consent must explicitly permit secondary evaluation of speech recognition and machine learning algorithms for emergency communication.
- Audio files and associated metadata must be stored in encrypted, access-restricted institutional repositories with role-based access control (RBAC).

### 2.4 Annotation Taxonomy, Ground Truth, Adjudication, and Quality Control
- **Canonical 7-Class Taxonomy**:
  1. `GENERAL_SPEECH` (Non-emergency baseline)
  2. `CRITICAL_RESPONSE` (Binary emergency intent)
  3. `FALL` (Binary emergency intent)
  4. `BREATHING_MEDICAL` (Binary emergency intent)
  5. `PAIN` (Binary emergency intent)
  6. `SOS_HELP` (Binary emergency intent)
  7. `LOCATION` (Binary emergency intent)
- **Annotation Protocol**:
  - Independent double-annotation by two certified SLPs blinded to model architectures.
  - Disagreements adjudicated by a third senior clinical specialist.
  - Inter-rater reliability (Cohen's Kappa $\kappa$) must exceed $0.85$ before an utterance is admitted to ground truth.

### 2.5 Emergency and Non-Emergency Class Support & Power Planning
- **Support Sizing Rule**: Sample sizes must not be arbitrarily guessed. Numeric targets are designated **`TO BE DETERMINED`** pending formal power calculations.
- **Power Planning Methodology**:
  - Precision target: Estimate False Emergency Rate (FER) with a 95% Wilson score confidence interval margin of error no greater than $\pm 2.0\%$ ($\Delta \le 0.02$).
  - Given the safety boundary $\text{FER} \le 0.1500$, the required non-emergency sample size is determined by:
    $$N_{\text{non-emergency}} \ge \frac{z_{1 - \alpha/2}^2 \cdot p(1 - p)}{\Delta^2} = \frac{1.96^2 \cdot 0.15 \cdot 0.85}{0.02^2} \approx 1,225 \text{ non-emergency samples per cohort}$$
  - Emergency Recall power calculation: Powering detection of emergency recall at $80\%$ with $\alpha = 0.05$ against a null hypothesis of $10\%$ requires a formal prospective clinical sample size determination (`TO BE DETERMINED`).

### 2.6 Duplicate and Derivative-Recording Checks
- Prospective audio recordings must undergo automated acoustic hashing (MD5, SHA-256) and acoustic fingerprinting (Chromaprint / Dejavu).
- Acoustic fingerprints must be cross-referenced against all 6,067 TORGO recordings to detect and reject accidental duplicate recordings or re-encoded derivatives.

### 2.7 Complete Physical and Logical Isolation from Historical Development Data
- Confirmation data will reside on a distinct, read-only volume completely isolated from `data/torgo/` and `results/phase8_1/`.
- Training pipelines, gradient backpropagation scripts, and PCA fitting routines will be programmatically forbidden from accessing the confirmation volume.

### 2.8 Label Custody and Locking until Protocol Freeze
- Ground truth intent labels will be encrypted with a public-key lock held by an independent audit committee.
- Feature extraction and model inference will be executed strictly on unlabelled audio.
- Ground truth labels will only be unblinded after model inference predictions have been recorded, hashed, and committed to git.

### 2.9 Preregistration of Models, Thresholds, Metrics, and Multiplicity Corrections
- The following candidate checkpoints and thresholds from Phase 8.1 are preregistered and frozen:
  1. **Track A (Classical baseline)**: Stage 1 Gate + 384-D Classical Residual; $\tau = 0.1500$.
  2. **INT-1-CALIB (Platt calibration)**: Stage 1 Gate + Platt Scaled Residual; $\tau = 0.1500$.
  3. **INT-2-COST (Cost-sensitive loss)**: Stage 1 Gate + Cost Weighted Residual; $\tau = 0.1500$.
  4. **INT-3-SAMP (Balanced sampling)**: Stage 1 Gate + Balanced Residual; $\tau = 0.1500$.
  5. **Track B (Quantum residual)**: Stage 1 Gate + 8-Qubit VQC Residual; $\tau = 0.1500$.
  6. **Track C (Matched classical shadow)**: Stage 1 Gate + Matched Classical MLP; $\tau = 0.1500$.
- **Fixed Safety Metric**: $\text{FER} = \frac{\text{FP}}{\text{FP} + \text{TN}} \le 0.1500$. Post-hoc threshold relaxation is strictly prohibited.
- **Multiplicity Correction**: Holm-Bonferroni correction applied across all pairwise candidate comparisons.

### 2.10 Data Access and Immutable Audit Logging
- Every file access to confirmation recordings will be logged with cryptographic timestamps.
- Zero interactive exploratory analysis of confirmation data will be permitted prior to formal code-locked execution.

### 2.11 Stop Criteria
Evaluation must immediately halt and abort if any of the following occur:
1. Detection of acoustic hash collisions with historical TORGO recordings.
2. Inter-annotator disagreement exceeding 15% of samples.
3. Total emergency positive samples falling below 50 utterances across all dysarthric speakers.
4. Any attempt to modify candidate model weights, PCA scalers, or decision thresholds after reading confirmation features.

---

## 3. Implementation Readiness

The prospective protocol configuration file is locked in [`config/phase8_2/confirmation_protocol.yaml`](file:///d:/DL/config/phase8_2/confirmation_protocol.yaml). All preregistered criteria are frozen and awaiting external data acquisition.
