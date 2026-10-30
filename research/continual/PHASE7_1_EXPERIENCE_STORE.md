# PHASE 7.1 — EXPERIENCE STORE & PERSONALIZATION DATA LAYER
## Research Data Layer Specification, Schemas, Eligibility & Invariants

---

> **CRITICAL SCIENTIFIC NOTICE:**
> Phase 7.1 introduces the data layer only. No model training or experimental performance claim is made.

---

### 1. Purpose
The purpose of Phase 7.1 is to establish a rigorous, type-safe, auditable data layer for longitudinal speech personalization. It provides the infrastructure to ingest speech interactions, record predictions and confidence scores, capture user or clinician confirmations, evaluate eligibility under a strict anti-self-training firewall, and maintain speaker-isolated experience stores for future replay.

---

### 2. Scope
- **Included:**
  - Experience Record data model and serialization.
  - Multi-state verification lifecycle (`UNVERIFIED`, `USER_VERIFIED`, `SYSTEM_VERIFIED`, `REJECTED`, `INVALID`).
  - Learning eligibility engine with standardized reason codes.
  - Speaker-scoped storage and strict cross-speaker isolation.
  - Research session grouping with chronological sequencing.
  - Deterministic duplicate protection (sample ID and SHA-256 fingerprint).
  - In-memory and persistent file-backed (JSON Lines) storage backends.
  - Diagnostic class and speaker imbalance reporting.
- **Excluded:**
  - Zero model training or fine-tuning.
  - Zero weight updates or parameter adaptation.
  - Zero changes to frozen Phase 1–6.5 artifacts or splits.
  - Zero modification to the frozen CASE B quantum conclusion.
  - Zero automated promotion from unverified prediction to ground truth.

---

### 3. Experience Record Schema
The canonical Experience Record schema (`src/continual/schema.py`) encapsulates all required fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `sample_id` / `experience_id` | str | Globally unique identifier for the interaction. |
| `speaker_id` | str | Speaker identifier (e.g., TORGO `F01`, `M01`). |
| `session_id` | str | Identifier of the research interaction session. |
| `timestamp` | str | ISO 8601 UTC timestamp of speech interaction. |
| `schema_version` | str | Schema specification version (default: `1.0.0`). |
| `audio_reference` | str | File path or URI to source audio recording. |
| `embedding_reference` | str | Path to frozen 384-D Whisper-tiny embedding. |
| `predicted_intent` | str | Canonical 7-class intent predicted by active model. |
| `prediction_confidence` | float | Calibrated prediction confidence in $[0.0, 1.0]$. |
| `confirmed_intent` | Optional[str] | Verified ground-truth intent if confirmation acquired. |
| `confirmation_source` | str | Source of confirmation (`user_explicit`, `clinician_verified`, etc.). |
| `verification_status` | str | State in verification lifecycle. |
| `learning_status` | str | Canonical status matching Phase 7.0 contract. |
| `learning_eligible` | bool | Boolean gate indicating authorization for training. |
| `eligibility_reason` | str | Structured, explainable eligibility code. |
| `quality_score` | float | Acoustic cleanness / SNR score in $[0.0, 1.0]$. |
| `novelty_score` | float | Representation distance from calibration support. |
| `data_quality_status` | str | Status flag: `CLEAN`, `DEGRADED`, or `QUARANTINED`. |
| `model_version` | str | Model identifier that generated the original prediction. |
| `representation_version` | str | Encoder specification hash (`whisper_tiny_384_unit_L2`). |
| `audit_trail` | list | List of state transition records preserving provenance. |

---

### 4. Verification States
Verification is explicitly modeled as a state machine:
- `UNVERIFIED`: Default initial state. Predictions are NOT ground truth.
- `USER_VERIFIED`: Confirmed directly by the speaker or caregiver.
- `SYSTEM_VERIFIED`: Verified against trusted oracle or held-out ground truth in benchmark simulations.
- `REJECTED`: Disconfirmed or explicitly rejected by user/clinician.
- `INVALID`: Malformed, corrupt, or unparseable record.

**Invariant:** `UNVERIFIED != GROUND TRUTH`.

---

### 5. Eligibility Rules
A record is authorized for future candidate adaptation (`learning_eligible = True`) **ONLY IF**:
1. It passes all schema and taxonomic validations.
2. It has an unambiguous speaker association (`speaker_id` non-empty).
3. It references a valid model version.
4. Its verification status is `USER_VERIFIED` or `SYSTEM_VERIFIED`.
5. Its confirmed intent is one of the 7 canonical classes.
6. It is not quarantined (`data_quality_status != "QUARANTINED"`).
7. Its acoustic quality is acceptable (`quality_score >= 0.5` and not `DEGRADED`).

**Anti-Self-Training Invariant:**
Prediction confidence alone—regardless of magnitude (even $\hat{p} = 1.0$)—NEVER confers learning eligibility.

---

### 6. Speaker & Session Model
- **Speaker Isolation:** The store maintains strict partitioning by speaker. Querying speaker $A$ will never return records belonging to speaker $B$. Omitting `speaker_id` in scoped methods raises `SpeakerIsolationError`.
- **Session Tracking:** `PersonalizationSession` groups utterances into chronological sequences. Attempting to add an utterance from a different speaker into an existing session raises `SessionSpeakerMismatchError`.

---

### 7. Persistence Model
- **`InMemoryExperienceStore`:** High-speed, deterministic in-memory store for unit tests and synthetic simulation loops.
- **`PersistentExperienceStore`:** File-backed storage persisting records as appendable JSON Lines files partitioned by speaker:
  `data/continual/experiences/{speaker_id}.jsonl`
  All file writes are performed atomically via temporary files to avoid partial or corrupted writes.

---

### 8. Duplicate Handling
`DuplicateDetector` provides deterministic duplicate prevention using:
1. `sample_id` uniqueness tracking.
2. Cryptographic SHA-256 fingerprinting of `(sample_id, speaker_id, session_id, audio_reference, timestamp, model_version)`.
- **Policies:**
  - `REJECT`: Raises `DuplicateRecordError`.
  - `IDEMPOTENT_IGNORE`: Returns existing record idempotently without duplication.
  - `VERSION`: Appends version suffix for intentional re-evaluations.

---

### 9. Provenance Rules
When an interaction is verified, its original state is strictly preserved:
- Original `predicted_intent` is preserved.
- Original `prediction_confidence` is preserved.
- Original `model_version` is preserved.
- Original `timestamp` is preserved.
- A new `AuditEntry` is appended to `audit_trail` recording the transition timestamp, previous status, new status, confirmed intent, and verifier ID.

---

### 10. Diagnostic Statistics Available
The store exposes non-invasive data-layer diagnostics:
- Total, verified, eligible, unverified, and rejected record counts.
- Frequency distribution across all 7 canonical classes (predicted vs. verified).
- Counts per speaker.
- 2D cross-tabulation matrix: `speaker x canonical_class`.
- Verification status breakdown.

---

### 11. Security & Privacy Baseline
- Audio files are referenced by relative path or URI; raw audio waveforms are never logged.
- Personally identifiable information is omitted from debug output.
- All storage is local and self-contained; zero network telemetry or cloud data transfers.

---

### 12. Explicit Exclusions
- No training routines or optimizers exist in this module.
- No dataset rebalancing, synthetic generation, oversampling, or undersampling is performed.
- No automatic model updates or parameter mutations are possible.

---

### 13. Testing Status
- Dedicated Phase 7.1 test suite: `tests/test_phase7_1_experience_store.py` (19 / 19 PASSED).
- Phase 7 boundary tests: `tests/test_phase7_boundary.py` (10 / 10 PASSED).
- Pre-existing regression suite: 225 / 225 PASSED.
- Total active tests: 254 / 254 PASSED.
