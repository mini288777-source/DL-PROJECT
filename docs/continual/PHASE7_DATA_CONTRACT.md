# PHASE 7 DATA CONTRACT & EXPERIENCE SCHEMA
## Experience Memory Schema, Eligibility Criteria & Replay Protocol

---

### 1. The Experience Record Contract

Every speech sample acquired during user interaction or simulation is tracked as an immutable, typed **Experience Record**. This contract defines the schema and operational invariants governing experience ingestion.

#### Experience Record Field Specifications

| Field Name | Type | Nullable | Validation Constraints / Enum Values | Description |
| :--- | :--- | :---: | :--- | :--- |
| `sample_id` | String | NO | Valid UUIDv4 or SHA-256 string | Unique identifier for the speech instance. |
| `speaker_id` | String | NO | Canonical TORGO speaker ID (`F01`, `M01`, etc.) | Speaker identity partition. |
| `audio_reference` | String | NO | Valid relative path to `.wav` file | Audio recording location (16 kHz mono). |
| `embedding_reference` | String | NO | Valid relative path to `.npy` file | Frozen 384-D Whisper bottleneck embedding. |
| `predicted_intent` | String | NO | Canonical 7-class taxonomy ID | Intent assigned by the predicting model. |
| `confirmed_intent` | String | YES | Canonical 7-class taxonomy ID or `None` | Verified true intent if feedback acquired. |
| `prediction_confidence`| Float | NO | Range: $[0.0, 1.0]$ | Calibrated prediction probability. |
| `confirmation_source` | String | NO | `user_explicit`, `clinician_verified`, `oracle_eval`, `unconfirmed` | Origin of the confirmation signal. |
| `timestamp` | String | NO | ISO 8601 UTC format (`YYYY-MM-DDTHH:MM:SSZ`) | Moment of recording ingestion. |
| `model_version` | String | NO | Semantic version string (`v0.1.0`, etc.) | Model version that performed inference. |
| `representation_version`| String | NO | Hash string | Fingerprint of encoder and feature scaling. |
| `learning_status` | String | NO | Enumerated Status (see Section 2) | Current stage in the eligibility lifecycle. |
| `data_source` | String | NO | `torgo_stream`, `held_out_stream`, `interactive_sim` | Experimental partition of origin. |
| `quality_score` | Float | NO | Range: $[0.0, 1.0]$ | Acoustic SNR / signal integrity metric. |
| `novelty_score` | Float | NO | Range: $[0.0, 1.0]$ | Mahalanobis / Euclidean distance to support. |
| `environment_metadata` | Dict | NO | JSON-serializable dictionary | SNR level, background noise, channel info. |

---

### 2. Learning Status State Machine

Every record transitions strictly through controlled states:

```
[ UNREVIEWED ]
      │
      ├── (High confidence non-emergency) ──────────────► [ AUTO_ELIGIBLE ]
      │                                                          │
      ├── (User/caregiver explicit feedback) ───────────► [ USER_CONFIRMED ]
      │                                                          │
      ├── (Degraded audio / out-of-distribution) ───────► [ QUARANTINED ]
      │                                                          │
      └── (Explicit misclassification feedback) ────────► [ REJECTED ]
                                                                 │
                                                                 ▼
                                                    [ USED_FOR_TRAINING ]
                                                                 OR
                                                    [ VALIDATION_ONLY ]
```

#### Status Definitions
1. `UNREVIEWED`: Initial state immediately following inference. Ineligible for model updates.
2. `AUTO_ELIGIBLE`: High-confidence non-emergency prediction meeting strict cleanness criteria.
3. `USER_CONFIRMED`: Explicit confirmation received from speaker or caregiver. Highest training weight.
4. `QUARANTINED`: Borderline or noisy sample held in quarantine; cannot enter training or validation.
5. `REJECTED`: Disconfirmed prediction or corrupt audio. Permanently excluded from training.
6. `USED_FOR_TRAINING`: Admitted to the candidate training replay buffer for offline model calibration.
7. `VALIDATION_ONLY`: Sequestered into the protected validation split; never seen during training.

---

### 3. Replay Buffer Balancing & Protection Contract

Because natural atypical speech exhibits severe class imbalance (~95.10% general speech vs. ~4.90% emergency speech), naïve experience accumulation will rapidly swamp minority emergency intents.

**Balancing Invariants:**
1. **Initial Seed Anchoring:** The initial $N_{\text{initial}}$ calibration instances from Phase 6.5 are permanently retained in replay memory and cannot be evicted.
2. **Quota-Based Replay:** Ingestion into `USED_FOR_TRAINING` enforces a maximum ratio of $4:1$ general-to-emergency utterances in the candidate adaptation dataset.
3. **Emergency Support Floor:** Continual training is triggered only when $\ge 2$ verified instances exist for each target emergency intent.

---

### 4. Phase 7.0 Prohibition Notice

**STRICT ENFORCEMENT:**
Phase 7.0 defines the schema and contract only. No automated experience collector, active learning loop, or candidate dataset assembly engine is active in Phase 7.0.
