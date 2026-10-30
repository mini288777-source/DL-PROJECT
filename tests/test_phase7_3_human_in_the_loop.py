"""
Phase 7.3 Human-in-the-Loop Active Learning Test Suite.
Verifies:
- Review request and response lifecycles and validation.
- ACCEPT, CORRECT, REJECT, and INVALID human review actions.
- Absolute ground-truth protection (confidence alone NEVER establishes verification).
- Strict canonical taxonomy validation (rejecting arbitrary text or non-canonical labels).
- Idempotent submission handling and conflicting duplicate prevention.
- Speaker and session isolation in review processing.
- Integration with ExperienceStore and deterministic EligibilityEngine.
- Deterministic review prioritization (HIGH, MEDIUM, LOW).
- Full provenance and audit trail retention.
"""

import os
import shutil
import tempfile
import pytest

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import CANONICAL_CLASSES
from src.continual.decision import (
    ActionDecision,
    ConfidenceDecision,
    PrioritizationTag,
)
from src.continual.store import (
    InMemoryExperienceStore,
    SpeakerIsolationError,
    RecordNotFoundError,
)
from src.continual.review import (
    ReviewRequest,
    ReviewResponse,
    ReviewState,
    ReviewAction,
    ReviewPriority,
    VerifierType,
    ReviewValidationError,
    ReviewConflictError,
    ReviewNotFoundError,
    determine_review_priority,
    validate_review_request,
    validate_review_response,
    load_review_policy,
)
from src.continual.queue import (
    InMemoryReviewQueue,
    PersistentReviewQueue,
    HumanReviewService,
)


@pytest.fixture
def clean_record() -> ExperienceRecord:
    """Fixture providing a valid unverified ExperienceRecord."""
    return ExperienceRecord(
        sample_id="exp_test_001",
        speaker_id="F01",
        session_id="session_01",
        audio_reference="data/torgo/audio/F01_session1_001.wav",
        embedding_reference="embeddings/torgo/whisper/F01_001.npy",
        predicted_intent="FALL",
        prediction_confidence=0.72,
        model_version="whisper_adapter_v1.0.0",
        quality_score=0.95,
        verification_status=VerificationStatus.UNVERIFIED.value,
        confirmed_intent=None,
        learning_eligible=False,
    )


@pytest.fixture
def clean_decision() -> ConfidenceDecision:
    """Fixture providing a valid ConfidenceDecision."""
    return ConfidenceDecision(
        decision_id="dec_test_001",
        experience_id="exp_test_001",
        speaker_id="F01",
        session_id="session_01",
        model_version="whisper_adapter_v1.0.0",
        predicted_label="FALL",
        raw_confidence=0.72,
        confidence_type="SOFTMAX_PROBABILITY",
        confidence_status="VALID",
        confidence_band="MEDIUM_CONFIDENCE",
        decision=ActionDecision.CONFIRM.value,
        decision_reason="EMERGENCY_INTENT_CONFIRMATION_REQUIRED",
        verification_required=True,
        learning_eligible=False,
        prioritization_tags=[PrioritizationTag.HIGH_EMERGENCY_RISK.value],
    )


# ==============================================================================
# 1. REVIEW REQUEST CREATION & VALIDATION
# ==============================================================================

def test_valid_review_request_created(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    assert req.review_id == "rev_exp_test_001"
    assert req.experience_id == "exp_test_001"
    assert req.speaker_id == "F01"
    assert req.predicted_label == "FALL"
    assert req.confidence == 0.72
    assert req.status == ReviewState.PENDING_REVIEW.value
    assert req.is_pending is True
    assert req.priority == ReviewPriority.HIGH.value  # Emergency risk triggers HIGH


def test_invalid_experience_id_rejected():
    queue = InMemoryReviewQueue()
    req = ReviewRequest(
        review_id="rev_bad",
        experience_id="",
        speaker_id="F01",
        session_id="sess_01",
        predicted_label="GENERAL_SPEECH",
        confidence=0.90,
    )
    with pytest.raises(ReviewValidationError, match="experience_id"):
        queue.enqueue(req)


def test_invalid_speaker_id_rejected():
    queue = InMemoryReviewQueue()
    req = ReviewRequest(
        review_id="rev_bad",
        experience_id="exp_123",
        speaker_id="",
        session_id="sess_01",
        predicted_label="GENERAL_SPEECH",
        confidence=0.90,
    )
    with pytest.raises(ReviewValidationError, match="speaker_id"):
        queue.enqueue(req)


def test_invalid_predicted_label_rejected():
    queue = InMemoryReviewQueue()
    req = ReviewRequest(
        review_id="rev_bad",
        experience_id="exp_123",
        speaker_id="F01",
        session_id="sess_01",
        predicted_label="NON_CANONICAL_INTENT",
        confidence=0.90,
    )
    with pytest.raises(ReviewValidationError, match="taxonomy"):
        queue.enqueue(req)


# ==============================================================================
# 2. ACCEPT ACTION TESTS
# ==============================================================================

def test_accept_action_updates_verification_and_eligibility(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    resp = ReviewResponse(
        response_id="resp_001",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.ACCEPT.value,
        verifier_type=VerifierType.USER.value,
        verifier_id="user_alice",
        comment="Utterance matches my intended emergency call",
    )

    updated_req, updated_rec = service.process_review(resp)

    # Review request state updated
    assert updated_req.status == ReviewState.ACCEPTED.value
    assert updated_req.verified_label == "FALL"
    assert updated_req.is_pending is False

    # Experience record verification updated
    assert updated_rec.verification_status == VerificationStatus.USER_VERIFIED.value
    assert updated_rec.confirmed_intent == "FALL"
    assert updated_rec.confirmation_source == VerifierType.USER.value
    assert updated_rec.verifier_id == "user_alice"

    # Original prediction strictly preserved
    assert updated_rec.predicted_intent == "FALL"

    # Eligibility recomputed via Phase 7.2 rule engine
    assert updated_rec.learning_eligible is True
    assert updated_rec.eligibility_reason == EligibilityReason.ELIGIBLE_USER_CONFIRMED.value


# ==============================================================================
# 3. CORRECT ACTION TESTS
# ==============================================================================

def test_correct_action_preserves_prediction_and_updates_ground_truth(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)  # predicted_intent is "FALL"
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    resp = ReviewResponse(
        response_id="resp_002",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.CORRECT.value,
        verified_label="PAIN",  # User corrects prediction to PAIN
        verifier_type=VerifierType.CLINICIAN.value,
        verifier_id="dr_smith",
        comment="Speaker clearly vocalized severe abdominal pain",
    )

    updated_req, updated_rec = service.process_review(resp)

    # Review request state updated
    assert updated_req.status == ReviewState.CORRECTED.value
    assert updated_req.verified_label == "PAIN"

    # CRITICAL: Original prediction remains "FALL"
    assert updated_rec.predicted_intent == "FALL"
    # Ground truth intent becomes "PAIN"
    assert updated_rec.confirmed_intent == "PAIN"
    assert updated_rec.verification_status == VerificationStatus.USER_VERIFIED.value

    # Eligibility recomputed
    assert updated_rec.learning_eligible is True
    assert updated_rec.eligibility_reason == EligibilityReason.ELIGIBLE_USER_CONFIRMED.value


def test_correct_action_with_invalid_taxonomy_rejected(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    resp = ReviewResponse(
        response_id="resp_bad",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.CORRECT.value,
        verified_label="ARBITRARY_CUSTOM_INTENT",  # Invalid!
    )

    with pytest.raises(ReviewValidationError, match="canonical taxonomy"):
        service.process_review(resp)


# ==============================================================================
# 4. REJECT ACTION TESTS
# ==============================================================================

def test_reject_action_disqualifies_record_from_learning(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    resp = ReviewResponse(
        response_id="resp_003",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.REJECT.value,
        reason="Acoustic background noise too severe for reliable interpretation",
    )

    updated_req, updated_rec = service.process_review(resp)

    assert updated_req.status == ReviewState.REJECTED.value
    assert updated_req.verified_label is None

    # Record rejected
    assert updated_rec.verification_status == VerificationStatus.REJECTED.value
    assert updated_rec.learning_status == LearningStatus.REJECTED.value
    assert updated_rec.learning_eligible is False
    assert updated_rec.eligibility_reason == EligibilityReason.INELIGIBLE_REJECTED.value


# ==============================================================================
# 5. GROUND TRUTH SAFETY (CRITICAL INVARIANT)
# ==============================================================================

def test_confidence_and_decision_alone_never_create_verification(clean_record):
    """
    CRITICAL SAFETY INVARIANT:
    Neither 1.0 confidence, nor low entropy, nor ACCEPT decision can ever
    verify an ExperienceRecord without human review.
    """
    store = InMemoryExperienceStore()
    ultra_confident_record = ExperienceRecord(
        sample_id="exp_ultra_confident",
        speaker_id="F01",
        audio_reference="data/audio/f01.wav",
        embedding_reference="embeddings/f01.npy",
        predicted_intent="SOS_HELP",
        prediction_confidence=0.9999,  # Extremely confident
        verification_status=VerificationStatus.UNVERIFIED.value,
        confirmed_intent=None,
    )
    store.create(ultra_confident_record)

    rec = store.read("exp_ultra_confident")
    assert rec.verification_status == VerificationStatus.UNVERIFIED.value
    assert rec.confirmed_intent is None
    assert rec.learning_eligible is False
    assert rec.eligibility_reason == EligibilityReason.INELIGIBLE_UNVERIFIED.value


# ==============================================================================
# 6. IDEMPOTENCY & CONFLICT PREVENTION
# ==============================================================================

def test_idempotent_duplicate_review_submission(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    resp = ReviewResponse(
        response_id="resp_dup_1",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.ACCEPT.value,
    )

    # First submission
    service.process_review(resp)

    # Duplicate submission with identical action & response
    dup_resp = ReviewResponse(
        response_id="resp_dup_2",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.ACCEPT.value,
    )
    req2, rec2 = service.process_review(dup_resp)
    assert req2.status == ReviewState.ACCEPTED.value
    assert rec2.verification_status == VerificationStatus.USER_VERIFIED.value


def test_conflicting_second_review_action_raises_error(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    # First: ACCEPT
    resp1 = ReviewResponse(
        response_id="resp_first",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.ACCEPT.value,
    )
    service.process_review(resp1)

    # Second: Attempt to overwrite with REJECT
    resp2 = ReviewResponse(
        response_id="resp_conflict",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.REJECT.value,
    )
    with pytest.raises(ReviewConflictError, match="already finalized"):
        service.process_review(resp2)


# ==============================================================================
# 7. SPEAKER & SESSION ISOLATION
# ==============================================================================

def test_speaker_isolation_enforced(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)  # Speaker F01

    # Record for Speaker M01
    record_b = ExperienceRecord(
        sample_id="exp_speaker_b",
        speaker_id="M01",
        audio_reference="data/audio/m01.wav",
        embedding_reference="embeddings/m01.npy",
        predicted_intent="PAIN",
        prediction_confidence=0.80,
    )
    store.create(record_b)

    service = HumanReviewService(queue=queue, store=store)
    req_a = service.create_review_request(record=clean_record, decision=clean_decision)

    # Attempt to use ReviewRequest for F01 against Record of M01
    forged_resp = ReviewResponse(
        response_id="resp_forged",
        review_id=req_a.review_id,
        experience_id="exp_speaker_b",  # Points to M01!
        review_action=ReviewAction.ACCEPT.value,
    )

    with pytest.raises((ReviewValidationError, SpeakerIsolationError)):
        service.process_review(forged_resp)


# ==============================================================================
# 8. ACTIVE LEARNING QUEUE & DETERMINISTIC PRIORITIZATION
# ==============================================================================

def test_deterministic_priority_calculation():
    # 1. Abstention -> HIGH
    p1 = determine_review_priority("GENERAL_SPEECH", 0.40, decision=ActionDecision.ABSTAIN.value)
    assert p1 == ReviewPriority.HIGH.value

    # 2. Emergency with moderate confidence -> HIGH
    p2 = determine_review_priority("FALL", 0.75, decision=ActionDecision.CONFIRM.value)
    assert p2 == ReviewPriority.HIGH.value

    # 3. Low confidence tag -> HIGH
    p3 = determine_review_priority("GENERAL_SPEECH", 0.60, prioritization_tags=[PrioritizationTag.LOW_CONFIDENCE.value])
    assert p3 == ReviewPriority.HIGH.value

    # 4. Standard accepted general speech -> LOW
    p4 = determine_review_priority("GENERAL_SPEECH", 0.90, decision=ActionDecision.ACCEPT.value)
    assert p4 == ReviewPriority.LOW.value


def test_queue_ordering_by_priority_and_time():
    queue = InMemoryReviewQueue()

    req_low = ReviewRequest(
        review_id="rev_low",
        experience_id="exp_l",
        speaker_id="F01",
        session_id="s1",
        predicted_label="GENERAL_SPEECH",
        confidence=0.95,
        priority=ReviewPriority.LOW.value,
        requested_at="2026-10-06T10:00:00Z",
    )
    req_high = ReviewRequest(
        review_id="rev_high",
        experience_id="exp_h",
        speaker_id="F01",
        session_id="s1",
        predicted_label="SOS_HELP",
        confidence=0.55,
        priority=ReviewPriority.HIGH.value,
        requested_at="2026-10-06T10:05:00Z",
    )
    req_med = ReviewRequest(
        review_id="rev_med",
        experience_id="exp_m",
        speaker_id="F01",
        session_id="s1",
        predicted_label="BREATHING_MEDICAL",
        confidence=0.75,
        priority=ReviewPriority.MEDIUM.value,
        requested_at="2026-10-06T10:02:00Z",
    )

    queue.enqueue(req_low)
    queue.enqueue(req_high)
    queue.enqueue(req_med)

    # Next review must be the HIGH priority item, despite being submitted last
    next_rev = queue.get_next_review()
    assert next_rev.review_id == "rev_high"

    # Query by priority
    highs = queue.get_by_priority(ReviewPriority.HIGH.value)
    assert len(highs) == 1
    assert highs[0].review_id == "rev_high"

    counts = queue.count_by_priority()
    assert counts[ReviewPriority.HIGH.value] == 1
    assert counts[ReviewPriority.MEDIUM.value] == 1
    assert counts[ReviewPriority.LOW.value] == 1
    assert queue.count_pending() == 3


# ==============================================================================
# 9. PERSISTENT REVIEW QUEUE ROUNDTRIP
# ==============================================================================

def test_persistent_review_queue_roundtrip():
    temp_dir = tempfile.mkdtemp(prefix="phase7_3_test_queue_")
    try:
        pqueue = PersistentReviewQueue(storage_dir=temp_dir)
        req = ReviewRequest(
            review_id="rev_persist_1",
            experience_id="exp_p1",
            speaker_id="F03",
            session_id="s_p",
            predicted_label="LOCATION",
            confidence=0.88,
            priority=ReviewPriority.MEDIUM.value,
        )
        pqueue.enqueue(req)

        resp = ReviewResponse(
            response_id="resp_p1",
            review_id="rev_persist_1",
            experience_id="exp_p1",
            review_action=ReviewAction.ACCEPT.value,
        )
        pqueue.mark_reviewed("rev_persist_1", resp)

        # Reload from fresh instance
        reloaded = PersistentReviewQueue(storage_dir=temp_dir)
        reloaded_req = reloaded.get_review("rev_persist_1")
        assert reloaded_req is not None
        assert reloaded_req.status == ReviewState.ACCEPTED.value
        assert reloaded_req.verified_label == "LOCATION"
        assert reloaded.count_pending() == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# 10. PROVENANCE, AUDIT & MULTI-STAGE GATING INTEGRATION
# ==============================================================================

def test_provenance_and_audit_trail_preserved(clean_record, clean_decision):
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    store.create(clean_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=clean_record, decision=clean_decision)
    resp = ReviewResponse(
        response_id="resp_provenance",
        review_id=req.review_id,
        experience_id=clean_record.sample_id,
        review_action=ReviewAction.ACCEPT.value,
        verifier_type=VerifierType.USER.value,
        verifier_id="user_bob",
        comment="Explicit audit confirmation",
    )
    updated_req, updated_rec = service.process_review(resp)

    # Provenance assertions
    assert updated_rec.model_version == clean_record.model_version
    assert updated_rec.representation_version == clean_record.representation_version
    assert updated_rec.predicted_intent == "FALL"
    assert updated_rec.confirmed_intent == "FALL"
    assert updated_req.review_policy_version == "v1.0.0_research_draft"
    assert updated_req.resolved_at != ""

    # Audit trail assertions
    assert len(updated_rec.audit_trail) >= 1
    last_audit = updated_rec.audit_trail[-1]
    assert last_audit["to_status"] == VerificationStatus.USER_VERIFIED.value
    assert last_audit["verifier_id"] == "user_bob"
    assert "Explicit audit confirmation" in last_audit["note"]


def test_low_quality_record_remains_ineligible_even_if_user_verified():
    """
    Multi-stage gate test:
    A degraded acoustic sample (quality_score < 0.50) confirmed by user
    transitions to USER_VERIFIED, but MUST REMAIN INELIGIBLE for learning.
    """
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    degraded_record = ExperienceRecord(
        sample_id="exp_degraded_001",
        speaker_id="M02",
        audio_reference="data/audio/m02_bad.wav",
        embedding_reference="embeddings/m02_bad.npy",
        predicted_intent="PAIN",
        prediction_confidence=0.85,
        quality_score=0.20,  # Below 0.50 threshold!
        data_quality_status="DEGRADED",
    )
    store.create(degraded_record)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=degraded_record)
    resp = ReviewResponse(
        response_id="resp_deg",
        review_id=req.review_id,
        experience_id="exp_degraded_001",
        review_action=ReviewAction.ACCEPT.value,
        verifier_type=VerifierType.USER.value,
    )
    _, updated_rec = service.process_review(resp)

    # Verification status updated
    assert updated_rec.verification_status == VerificationStatus.USER_VERIFIED.value
    assert updated_rec.confirmed_intent == "PAIN"

    # But eligibility is STRICTLY DENIED due to low acoustic quality
    assert updated_rec.learning_eligible is False
    assert updated_rec.eligibility_reason == EligibilityReason.INELIGIBLE_LOW_QUALITY.value


def test_invalid_action_marks_record_invalid():
    queue = InMemoryReviewQueue()
    store = InMemoryExperienceStore()
    rec = ExperienceRecord(
        sample_id="exp_corrupt",
        speaker_id="F01",
        audio_reference="data/audio/f01_corrupt.wav",
        predicted_intent="GENERAL_SPEECH",
        prediction_confidence=0.60,
    )
    store.create(rec)
    service = HumanReviewService(queue=queue, store=store)

    req = service.create_review_request(record=rec)
    resp = ReviewResponse(
        response_id="resp_inv",
        review_id=req.review_id,
        experience_id="exp_corrupt",
        review_action=ReviewAction.INVALID.value,
        reason="Garbled recording, unidentifiable speech",
    )
    updated_req, updated_rec = service.process_review(resp)

    assert updated_req.status == ReviewState.REJECTED.value
    assert updated_rec.verification_status == VerificationStatus.INVALID.value
    assert updated_rec.learning_eligible is False


def test_speaker_scoped_queue_retrieval():
    queue = InMemoryReviewQueue()
    req_f01 = ReviewRequest(
        review_id="rev_f01",
        experience_id="exp_f01",
        speaker_id="F01",
        session_id="s1",
        predicted_label="FALL",
        confidence=0.80,
    )
    req_m01 = ReviewRequest(
        review_id="rev_m01",
        experience_id="exp_m01",
        speaker_id="M01",
        session_id="s1",
        predicted_label="PAIN",
        confidence=0.80,
    )
    queue.enqueue(req_f01)
    queue.enqueue(req_m01)

    f01_reviews = queue.get_by_speaker("F01")
    assert len(f01_reviews) == 1
    assert f01_reviews[0].speaker_id == "F01"

    m01_reviews = queue.get_by_speaker("M01")
    assert len(m01_reviews) == 1
    assert m01_reviews[0].speaker_id == "M01"

    # Next review scoped to F01
    next_f01 = queue.get_next_review(speaker_id="F01")
    assert next_f01.review_id == "rev_f01"
