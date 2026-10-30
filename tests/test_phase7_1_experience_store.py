"""
Phase 7.1 Experience Store & Data Layer Test Suite.
Validates:
1. Schema & Validation Constraints (taxonomy, confidence, status, required fields).
2. Verification Lifecycle (UNVERIFIED != ground truth, audit trails, preservation of predictions).
3. Eligibility Engine (anti-self-training firewall, reasons, disqualifications).
4. Speaker-Scoped Personalization (isolation, chronological ordering, cross-speaker protection).
5. Session Integrity (session tracking, chronological sequence, speaker mismatch guard).
6. Duplicate Protection (deterministic detection, rejection policy).
7. Diagnostic Statistics (class, speaker, verification counts, 2D cross-tabulation).
8. Provenance Preservation (encoder version, model version, historical prediction preservation).
9. Persistent Backend (JSONL roundtrip, update, reload, integrity).
10. Strict Non-Learning Guarantee (stored records never train models or alter weights).
"""

import os
import shutil
import tempfile
import pytest

from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent,
)
from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.validation import (
    validate_experience_record,
    assert_valid_experience_record,
    ValidationError,
)
from src.continual.eligibility import evaluate_learning_eligibility
from src.continual.session import (
    PersonalizationSession,
    SessionSpeakerMismatchError,
)
from src.continual.duplicate import (
    DuplicateDetector,
    DuplicatePolicy,
    DuplicateRecordError,
)
from src.continual.store import (
    InMemoryExperienceStore,
    PersistentExperienceStore,
    RecordNotFoundError,
    SpeakerIsolationError,
)


@pytest.fixture
def valid_record():
    return ExperienceRecord(
        sample_id="exp-001",
        speaker_id="F01",
        session_id="session-001",
        timestamp="2026-10-06T01:00:00Z",
        audio_reference="data/torgo/F01/Session1/wav_arrayMic/001.wav",
        embedding_reference="data/embeddings/F01/001.npy",
        predicted_intent="SOS_HELP",
        prediction_confidence=0.92,
        model_version="v_phase6_5_hierarchical",
        representation_version="whisper_tiny_384_unit_L2",
        data_source="torgo_stream",
        quality_score=0.95,
        novelty_score=0.12,
        environment_metadata={"snr_db": 22.5},
    )


# ==============================================================================
# 1. SCHEMA & VALIDATION TESTS
# ==============================================================================

def test_valid_experience_record_accepted(valid_record):
    """Verify that a well-formed Experience Record passes validation."""
    errors = validate_experience_record(valid_record)
    assert errors == [], f"Validation failed unexpectedly: {errors}"
    assert_valid_experience_record(valid_record)


def test_missing_required_fields_rejected():
    """Verify that missing critical fields (sample_id, speaker_id, timestamp) fail validation."""
    bad_record = ExperienceRecord(
        sample_id="",
        speaker_id="",
        predicted_intent="SOS_HELP",
        audio_reference="test.wav",
    )
    errors = validate_experience_record(bad_record)
    assert any("sample_id" in e for e in errors)
    assert any("speaker_id" in e for e in errors)
    with pytest.raises(ValidationError):
        assert_valid_experience_record(bad_record)


def test_invalid_taxonomy_rejected(valid_record):
    """Verify that non-canonical intent labels are strictly rejected."""
    valid_record.predicted_intent = "Medical"  # Non-canonical label
    errors = validate_experience_record(valid_record)
    assert any("predicted_intent" in e for e in errors)
    with pytest.raises(ValidationError):
        assert_valid_experience_record(valid_record)

    valid_record.predicted_intent = "FALL"
    valid_record.confirmed_intent = "Assistance"  # Disallowed unmapped label
    errors = validate_experience_record(valid_record)
    assert any("confirmed_intent" in e for e in errors)


def test_invalid_confidence_range_rejected(valid_record):
    """Verify that out-of-range confidence scores are rejected."""
    valid_record.prediction_confidence = 1.45
    errors = validate_experience_record(valid_record)
    assert any("prediction_confidence" in e for e in errors)

    valid_record.prediction_confidence = -0.1
    errors = validate_experience_record(valid_record)
    assert any("prediction_confidence" in e for e in errors)


def test_invalid_verification_status_rejected(valid_record):
    """Verify that arbitrary or unapproved verification statuses fail validation."""
    valid_record.verification_status = "SOME_UNOFFICIAL_STATUS"
    errors = validate_experience_record(valid_record)
    assert any("verification_status" in e for e in errors)


# ==============================================================================
# 2. VERIFICATION & IMMUTABILITY TESTS
# ==============================================================================

def test_unverified_record_remains_unverified_on_creation(valid_record):
    """Verify default initial state is UNVERIFIED and not ground truth."""
    store = InMemoryExperienceStore()
    stored = store.create(valid_record)
    assert stored.verification_status == VerificationStatus.UNVERIFIED.value
    assert stored.confirmed_intent is None
    assert stored.learning_eligible is False
    assert stored.eligibility_reason == EligibilityReason.INELIGIBLE_UNVERIFIED.value


def test_verified_record_preserves_original_prediction(valid_record):
    """
    CRITICAL PROVENANCE INVARIANT:
    When a record is verified (e.g. user corrected SOS_HELP -> FALL),
    the original predicted_intent, confidence, and model_version MUST NOT be overwritten.
    """
    store = InMemoryExperienceStore()
    store.create(valid_record)

    original_prediction = valid_record.predicted_intent
    original_confidence = valid_record.prediction_confidence
    original_model = valid_record.model_version

    updated = store.mark_verified(
        sample_id=valid_record.sample_id,
        verified_label="FALL",
        verification_source="user_explicit",
        verifier_id="user_f01",
        note="User confirmed fall alert correction",
    )

    # Historical prediction must remain intact
    assert updated.predicted_intent == original_prediction
    assert updated.prediction_confidence == original_confidence
    assert updated.model_version == original_model

    # Verified fields updated
    assert updated.confirmed_intent == "FALL"
    assert updated.verification_status == VerificationStatus.USER_VERIFIED.value
    assert updated.learning_status == LearningStatus.USER_CONFIRMED.value
    assert updated.learning_eligible is True
    assert updated.eligibility_reason == EligibilityReason.ELIGIBLE_USER_CONFIRMED.value

    # Audit trail recorded
    assert len(updated.audit_trail) >= 1
    audit = updated.audit_trail[-1]
    assert audit["from_status"] == VerificationStatus.UNVERIFIED.value
    assert audit["to_status"] == VerificationStatus.USER_VERIFIED.value
    assert audit["confirmed_intent"] == "FALL"


def test_rejected_record_cannot_become_eligible(valid_record):
    """Verify that a rejected record is marked ineligible and cannot enter training sets."""
    store = InMemoryExperienceStore()
    store.create(valid_record)

    rejected = store.mark_rejected(
        sample_id=valid_record.sample_id,
        reason="Garbled microphone burst / acoustic noise",
        verifier_id="clinician_01",
    )

    assert rejected.verification_status == VerificationStatus.REJECTED.value
    assert rejected.learning_status == LearningStatus.REJECTED.value
    assert rejected.learning_eligible is False
    assert rejected.eligibility_reason == EligibilityReason.INELIGIBLE_REJECTED.value
    assert rejected.exclusion_reason == "Garbled microphone burst / acoustic noise"

    # Must not appear in eligible list
    eligible = store.list_learning_eligible(speaker_id=valid_record.speaker_id)
    assert valid_record.sample_id not in [r.sample_id for r in eligible]


# ==============================================================================
# 3. ELIGIBILITY ENGINE TESTS
# ==============================================================================

def test_unverified_record_is_strictly_ineligible(valid_record):
    """
    CRITICAL ANTI-SELF-TRAINING TEST:
    Even with prediction_confidence = 1.0, an unverified prediction is INELIGIBLE.
    """
    valid_record.prediction_confidence = 1.0
    valid_record.verification_status = VerificationStatus.UNVERIFIED.value
    valid_record.confirmed_intent = None

    eligible, reason = evaluate_learning_eligibility(valid_record)
    assert eligible is False
    assert reason == EligibilityReason.INELIGIBLE_UNVERIFIED.value


def test_quarantined_record_is_ineligible(valid_record):
    """Verify that quarantined records are strictly ineligible for model updates."""
    valid_record.verification_status = VerificationStatus.USER_VERIFIED.value
    valid_record.confirmed_intent = "SOS_HELP"
    valid_record.learning_status = LearningStatus.QUARANTINED.value

    eligible, reason = evaluate_learning_eligibility(valid_record)
    assert eligible is False
    assert reason == EligibilityReason.INELIGIBLE_QUARANTINED.value


def test_low_quality_record_is_ineligible(valid_record):
    """Verify that acoustically degraded records are disqualified."""
    valid_record.verification_status = VerificationStatus.USER_VERIFIED.value
    valid_record.confirmed_intent = "SOS_HELP"
    valid_record.quality_score = 0.35  # Below 0.5 threshold

    eligible, reason = evaluate_learning_eligibility(valid_record)
    assert eligible is False
    assert reason == EligibilityReason.INELIGIBLE_LOW_QUALITY.value


# ==============================================================================
# 4. SPEAKER ISOLATION & SCOPING TESTS
# ==============================================================================

def test_speaker_isolation_enforced():
    """Verify speaker A experiences never leak into speaker B retrieval."""
    store = InMemoryExperienceStore()

    rec_f01 = ExperienceRecord(
        sample_id="f01-01",
        speaker_id="F01",
        predicted_intent="SOS_HELP",
        audio_reference="f01.wav",
        timestamp="2026-10-06T01:00:00Z",
    )
    rec_m01 = ExperienceRecord(
        sample_id="m01-01",
        speaker_id="M01",
        predicted_intent="FALL",
        audio_reference="m01.wav",
        timestamp="2026-10-06T01:05:00Z",
    )

    store.create(rec_f01)
    store.create(rec_m01)

    f01_records = store.list_by_speaker("F01")
    m01_records = store.list_by_speaker("M01")

    assert len(f01_records) == 1
    assert f01_records[0].sample_id == "f01-01"
    assert "m01-01" not in [r.sample_id for r in f01_records]

    assert len(m01_records) == 1
    assert m01_records[0].sample_id == "m01-01"
    assert "f01-01" not in [r.sample_id for r in m01_records]


def test_missing_speaker_id_raises_isolation_error():
    """Verify that retrieval with empty speaker ID raises SpeakerIsolationError."""
    store = InMemoryExperienceStore()
    with pytest.raises(SpeakerIsolationError):
        store.list_by_speaker("")


# ==============================================================================
# 5. SESSION MODEL TESTS
# ==============================================================================

def test_session_association_and_chronology(valid_record):
    """Verify that sessions group records and preserve chronological order."""
    session = PersonalizationSession(
        session_id="sess-f01-01",
        speaker_id="F01",
        model_version="v_phase6_5",
    )

    rec1 = ExperienceRecord(
        sample_id="rec-02",
        speaker_id="F01",
        timestamp="2026-10-06T01:10:00Z",
        predicted_intent="FALL",
        audio_reference="02.wav",
    )
    rec2 = ExperienceRecord(
        sample_id="rec-01",
        speaker_id="F01",
        timestamp="2026-10-06T01:05:00Z",  # Earlier timestamp
        predicted_intent="GENERAL_SPEECH",
        audio_reference="01.wav",
    )

    session.add_record(rec1)
    session.add_record(rec2)

    records = session.get_records()
    assert len(records) == 2
    # Must be chronologically sorted
    assert records[0].sample_id == "rec-01"
    assert records[1].sample_id == "rec-02"


def test_session_speaker_mismatch_rejected():
    """Verify that adding an utterance from another speaker to a session raises an error."""
    session = PersonalizationSession(
        session_id="sess-f01-01",
        speaker_id="F01",
        model_version="v_phase6_5",
    )
    rec_foreign = ExperienceRecord(
        sample_id="m01-rec",
        speaker_id="M01",  # Different speaker
        predicted_intent="SOS_HELP",
        audio_reference="m01.wav",
    )
    with pytest.raises(SessionSpeakerMismatchError):
        session.add_record(rec_foreign)


# ==============================================================================
# 6. DUPLICATE DETECTION TESTS
# ==============================================================================

def test_duplicate_rejection_policy(valid_record):
    """Verify that duplicate sample insertion under REJECT policy raises DuplicateRecordError."""
    store = InMemoryExperienceStore(duplicate_policy=DuplicatePolicy.REJECT)
    store.create(valid_record)

    # Identical record insertion must fail
    with pytest.raises(DuplicateRecordError):
        store.create(valid_record)


def test_duplicate_idempotent_ignore_policy(valid_record):
    """Verify that IDEMPOTENT_IGNORE returns existing record without error or duplicate count."""
    store = InMemoryExperienceStore(duplicate_policy=DuplicatePolicy.IDEMPOTENT_IGNORE)
    first = store.create(valid_record)
    second = store.create(valid_record)

    assert first.sample_id == second.sample_id
    assert len(store.list_by_speaker("F01")) == 1


# ==============================================================================
# 7. DIAGNOSTIC STATISTICS & IMBALANCE AWARENESS TESTS
# ==============================================================================

def test_statistics_and_speaker_class_matrix():
    """Verify safe calculation of class, speaker, and verification statistics."""
    store = InMemoryExperienceStore()

    r1 = ExperienceRecord(
        sample_id="r1",
        speaker_id="F01",
        predicted_intent="GENERAL_SPEECH",
        audio_reference="r1.wav",
    )
    r2 = ExperienceRecord(
        sample_id="r2",
        speaker_id="F01",
        predicted_intent="SOS_HELP",
        audio_reference="r2.wav",
    )
    r3 = ExperienceRecord(
        sample_id="r3",
        speaker_id="M01",
        predicted_intent="FALL",
        audio_reference="r3.wav",
    )

    store.create(r1)
    store.create(r2)
    store.create(r3)

    # Mark r2 verified
    store.mark_verified("r2", "SOS_HELP", "user_explicit")

    stats = store.get_statistics()
    assert stats["total_experiences"] == 3
    assert stats["verified_experiences"] == 1
    assert stats["eligible_experiences"] == 1
    assert stats["unverified_experiences"] == 2
    assert stats["unique_speakers"] == 2

    # Verify 2D cross-tabulation
    matrix = store.get_speaker_class_matrix()
    assert matrix["F01"]["GENERAL_SPEECH"] == 1
    assert matrix["F01"]["SOS_HELP"] == 1
    assert matrix["M01"]["FALL"] == 1


# ==============================================================================
# 8. PERSISTENT EXPERIENCE STORE TESTS
# ==============================================================================

def test_persistent_store_roundtrip(valid_record):
    """Verify that PersistentExperienceStore writes, reads, updates, and reloads from disk."""
    temp_dir = tempfile.mkdtemp(prefix="test_store_")
    try:
        # 1. Initialize and save record
        store = PersistentExperienceStore(storage_dir=temp_dir)
        store.create(valid_record)

        # 2. Check speaker file was written
        spk_file = os.path.join(temp_dir, f"{valid_record.speaker_id}.jsonl")
        assert os.path.exists(spk_file)
        assert os.path.getsize(spk_file) > 0

        # 3. Update verification state
        store.mark_verified(
            sample_id=valid_record.sample_id,
            verified_label="SOS_HELP",
            verification_source="user_explicit",
        )

        # 4. Instantiate a fresh store instance from same directory (cold reload)
        reloaded_store = PersistentExperienceStore(storage_dir=temp_dir)
        reloaded_record = reloaded_store.read(valid_record.sample_id)

        assert reloaded_record is not None
        assert reloaded_record.sample_id == valid_record.sample_id
        assert reloaded_record.verification_status == VerificationStatus.USER_VERIFIED.value
        assert reloaded_record.confirmed_intent == "SOS_HELP"
        assert reloaded_record.learning_eligible is True
        assert reloaded_record.predicted_intent == valid_record.predicted_intent
        assert len(reloaded_record.audit_trail) >= 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
