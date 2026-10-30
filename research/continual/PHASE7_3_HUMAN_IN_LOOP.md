# PHASE 7.3 — HUMAN-IN-THE-LOOP ACTIVE LEARNING
## Verification Workflow, Review Queue, Ground-Truth Protection & Safe Experience Enrichment

---

> **CRITICAL SCIENTIFIC NOTICES:**
> 1. **Human verification, not model confidence, establishes ground truth.** Neither high confidence ($\hat{p} \approx 1.0$), nor low entropy, nor model consensus can ever automatically convert an unverified prediction into ground truth or grant learning eligibility.
> 2. **Phase 7.3 does not train or update any recognition model.** No weights are updated, no classifiers are adapted, and no models are promoted or fine-tuned in this phase.
> 3. **The frozen CASE B conclusion remains immutable:** No statistically supported quantum advantage was observed under the evaluated fixed-kernel protocol.

---

### 1. Purpose
Phase 7.3 implements a safe, type-safe, and fully auditable human-in-the-loop active-learning layer. In safety-critical emergency communication for individuals with dysarthric speech, autonomous self-training based on unverified model predictions leads to catastrophic confirmation bias and margin collapse.

Phase 7.3 provides the infrastructure for users, caregivers, clinicians, or research evaluators to:
1. Inspect speech recognition predictions alongside uncertainty and confidence signals,
2. Accept predictions when verified to be accurate,
3. Correct erroneous predictions with valid canonical taxonomy labels,
4. Reject ambiguous, low-quality, or corrupted interactions,
5. Provide an explicit, cryptographically traceable ground-truth label.

The results seamlessly enrich the **Experience Store** (Phase 7.1) and pass through the deterministic **Eligibility Engine** (Phase 7.2) without bypassing either.

---

### 2. End-to-End Review Architecture

The complete human-in-the-loop decision and verification pipeline operates strictly unidirectionally:

```
[ Model Prediction ]
        │
        ▼
[ Confidence & Uncertainty ] (Phase 7.2)
        │
        ▼
[ Decision & Suggested Action ] (Phase 7.2)
        │
        ▼
[ Human Review Request ] ─────► [ Review Queue (Priority: HIGH / MED / LOW) ]
        │
        ▼
[ Human Verifier Action ] (ACCEPT / CORRECT / REJECT / INVALID)
        │
        ▼
[ Auditable Review Response ]
        │
        ▼
[ Verification Mutation in Store ] (confirmed_intent, verifier_id, audit_trail)
        │
        ▼
[ Multi-Stage Eligibility Gate ] (Phase 7.2)
        │
        ▼
[ Authorized Experience Record ] (learning_eligible: True / False)
```

---

### 3. Review Lifecycle States
A review request traverses deterministic lifecycle states (`ReviewState`):
- `PENDING_REVIEW`: Enqueued and awaiting human inspection.
- `ACCEPTED`: Human verified that the predicted label is accurate.
- `CORRECTED`: Human supplied a different canonical label as ground truth.
- `REJECTED`: Human indicated the sample must be excluded from learning evidence.
- `EXPIRED`: Review request exceeded configured retention window (`timeout_seconds`).

---

### 4. Human Review Actions & Behaviors

#### A. ACCEPT Action
- **Semantics:** The human reviewer confirms that `predicted_label` accurately reflects the speaker's intent.
- **State Transition:** Request transitions to `ACCEPTED`.
- **Ground Truth Assignment:** `confirmed_intent = predicted_label`.
- **Provenance:** `predicted_intent` remains strictly immutable. `confirmation_source = "user_explicit"`.
- **Eligibility Gate:** Triggers `EligibilityEngine.evaluate_eligibility(record)`. If all 12 stages pass, record is granted `ELIGIBLE_USER_CONFIRMED`.

#### B. CORRECT Action
- **Semantics:** The human reviewer identifies that the model misclassified the utterance and provides the true intent label.
- **Validation:** Replacement label is validated against the 7 canonical classes (`GENERAL_SPEECH`, `CRITICAL_RESPONSE`, `FALL`, `BREATHING_MEDICAL`, `PAIN`, `SOS_HELP`, `LOCATION`). Arbitrary strings are strictly rejected.
- **State Transition:** Request transitions to `CORRECTED`.
- **Ground Truth Assignment:** `confirmed_intent = verified_label`.
- **Provenance Preservation:** `predicted_intent` remains untouched (e.g. `FALL`), preserving the discrepancy (`FALL` $\to$ `PAIN`) as valuable error telemetry.
- **Eligibility Gate:** Re-evaluates eligibility. If valid, granted `ELIGIBLE_USER_CONFIRMED`.

#### C. REJECT Action
- **Semantics:** The reviewer indicates the utterance is unrepresentative, compromised by acoustic artifacts, or should not be learned.
- **State Transition:** Request transitions to `REJECTED`.
- **Record Impact:** `verification_status = REJECTED`, `learning_status = REJECTED`.
- **Eligibility:** Permanently barred from learning (`learning_eligible = False`, `eligibility_reason = INELIGIBLE_REJECTED`).

#### D. INVALID Action
- **Semantics:** Speech sample is unintelligible, clipped, or corrupted.
- **Record Impact:** `verification_status = INVALID`, `learning_eligible = False`.

---

### 5. Ground-Truth Safety Invariants
1. **Model Confidence Never Verifies:** Neither high confidence ($\hat{p} = 0.9999$), nor decision margin, nor low entropy can ever set `confirmed_intent` or mark a record as `USER_VERIFIED`.
2. **Explicit Verification Source Required:** Ground truth requires an authorized human reviewer (`user_explicit`, `clinician_verified`, or `oracle_ground_truth`).
3. **Canonical Taxonomy Integrity:** No non-canonical labels are permitted into the learning memory.

---

### 6. Active-Learning Review Queue & Deterministic Prioritization
Review items are scheduled through the `ReviewQueue` abstraction (`InMemoryReviewQueue` and `PersistentReviewQueue`).

#### Priority Levels (`ReviewPriority`)
Scheduling is governed by deterministic rules configured in `config/continual/review_policy.yaml`:
1. **HIGH Priority:**
   - Any utterance triggering `ActionDecision.ABSTAIN`.
   - Emergency intents with confidence $< 0.85$.
   - Any utterance bearing `HIGH_EMERGENCY_RISK`, `HIGH_UNCERTAINTY`, or `LOW_CONFIDENCE` tags.
2. **MEDIUM Priority:**
   - Ambiguous utterances triggering `ActionDecision.CONFIRM`.
   - Moderate confidence predictions ($0.65 \le \hat{p} < 0.85$).
3. **LOW Priority:**
   - Confident general conversational speech ($\hat{p} \ge 0.70$) with `ActionDecision.ACCEPT`.

Queue retrieval (`get_next_review`) sorts deterministically by priority rank ascending (`HIGH=0`, `MEDIUM=1`, `LOW=2`) and secondarily by chronological request timestamp.

---

### 7. Idempotency & Conflict Prevention
1. **Idempotent Resubmission:** If an identical review response (`review_action`, `verified_label`) is submitted multiple times, the service returns the existing finalized state idempotently without creating duplicate entries or mutating records.
2. **Conflicting Action Rejection:** If a second action contradicts an already finalized review (e.g. attempting to submit `REJECT` after `ACCEPT`), a `ReviewConflictError` is raised to prevent race conditions and preserve audit integrity.

---

### 8. Speaker & Session Isolation
- Every review request and response embeds `speaker_id` and `session_id`.
- The `HumanReviewService` asserts that the `ExperienceRecord.speaker_id` exactly matches `ReviewRequest.speaker_id`.
- Review queues support speaker-scoped slicing (`get_by_speaker`), ensuring reviewers for Speaker A cannot view or alter records for Speaker B.

---

### 9. Multi-Stage Eligibility Integration
Human verification is a **necessary**, but **not sufficient**, condition for learning eligibility:
- A user-verified sample with severe acoustic degradation (`quality_score < 0.50` or `data_quality_status == "DEGRADED"`) transitions to `USER_VERIFIED` but remains `learning_eligible = False` with `INELIGIBLE_LOW_QUALITY`.
- A quarantined sample confirmed by a user remains `INELIGIBLE_QUARANTINED`.
- This firewall guarantees that human error or degraded audio cannot corrupt future continual adaptation sets.

---

### 10. Verification & Test Suite Summary
- Dedicated Phase 7.3 tests: `tests/test_phase7_3_human_in_the_loop.py` (19 / 19 PASSED).
- Phase 7.2 tests: `tests/test_phase7_2_confidence_engine.py` (18 / 18 PASSED).
- Phase 7.1 tests: `tests/test_phase7_1_experience_store.py` (19 / 19 PASSED).
- Phase 7 Boundary tests: `tests/test_phase7_boundary.py` (10 / 10 PASSED).
- Existing regression tests (Phases 1–6.5): 225 / 225 PASSED.
- Cumulative repository tests: 291 / 291 PASSED (100% green).

---

### 11. Explicit Exclusions
- **No Model Training:** Zero parameter updates, gradient backpropagation, or SVM refitting.
- **No Self-Training:** Autonomous label assignment without human interaction is strictly forbidden.
- **No Data Rebalancing:** No oversampling or artificial synthesis of emergency intents.
- **No Phase 7.4 Initiation:** Continual model adaptation remains strictly blocked until future phases.
