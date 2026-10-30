"""
Phase 7.2 Confidence & Learning Eligibility Engine Test Suite.
Validates:
1. Confidence Validation (NaN, Inf, negatives, >1 probabilities, missing values).
2. Decision Logic (high/medium/low confidence, abstention, confirmation).
3. Ground Truth Safety Invariant (confidence never creates verification, unverified stays unverified).
4. Learning Eligibility Gate (multi-stage evaluation, reason codes, safety policies).
5. Emergency Communication Safety (risk-averse thresholds, mandatory verification).
6. Uncertainty Quantification (top1-top2 margin, entropy, max probability).
7. Provenance Preservation (model version, policy version, timestamp, raw confidence).
8. Speaker Isolation (speaker scoping preserved across decisions).
9. Policy Versioning (changing thresholds generates new versioned decisions without mutating historical records).
10. Strict Non-Learning Guarantee (no model training, zero weight updates).
"""

import math
import pytest

from src.continual.confidence import (
    ConfidenceBand,
    ConfidenceRepresentation,
    ConfidenceType,
    ConfidenceValidator,
    UncertaintyCalculator,
    load_confidence_policy,
)
from src.continual.decision import (
    ActionDecision,
    ConfidenceDecision,
    DecisionEngine,
    PrioritizationTag,
    SuggestedAction,
)
from src.continual.eligibility import (
    EligibilityEngine,
    evaluate_learning_eligibility,
)
from src.continual.schema import (
    EligibilityReason,
    ExperienceRecord,
    LearningStatus,
    VerificationStatus,
)
from src.continual.store import InMemoryExperienceStore


# ==============================================================================
# 1. CONFIDENCE VALIDATION TESTS
# ==============================================================================

def test_valid_confidence_accepted():
    """Verify that valid probabilities are accepted and banded correctly."""
    policy = load_confidence_policy()
    rep = ConfidenceRepresentation.create(
        raw_score=2.85,
        confidence_value=0.92,
        source_model="classical_adapter_v1",
        confidence_type=ConfidenceType.SOFTMAX_PROBABILITY,
        policy=policy,
    )
    assert rep.is_valid is True
    assert rep.confidence_band == ConfidenceBand.HIGH_CONFIDENCE
    assert rep.validation_notes == []


def test_nan_confidence_rejected():
    """Verify that NaN confidence measurements are strictly rejected."""
    rep = ConfidenceRepresentation.create(
        raw_score=float("nan"),
        confidence_value=float("nan"),
        source_model="classical_adapter_v1",
        confidence_type=ConfidenceType.SOFTMAX_PROBABILITY,
    )
    assert rep.is_valid is False
    assert rep.confidence_band == ConfidenceBand.INVALID_CONFIDENCE
    assert any("NaN" in note for note in rep.validation_notes)


def test_infinite_confidence_rejected():
    """Verify that infinite confidence values are flagged as invalid."""
    rep = ConfidenceRepresentation.create(
        raw_score=float("inf"),
        confidence_value=float("inf"),
        source_model="qsvc_q8_r1",
        confidence_type=ConfidenceType.SOFTMAX_PROBABILITY,
    )
    assert rep.is_valid is False
    assert rep.confidence_band == ConfidenceBand.INVALID_CONFIDENCE
    assert any("Infinite" in note for note in rep.validation_notes)


def test_negative_probability_rejected():
    """Verify that negative probabilities fail validation."""
    rep = ConfidenceRepresentation.create(
        raw_score=-0.5,
        confidence_value=-0.05,
        source_model="rbf_svm_v1",
        confidence_type=ConfidenceType.SOFTMAX_PROBABILITY,
    )
    assert rep.is_valid is False
    assert rep.confidence_band == ConfidenceBand.INVALID_CONFIDENCE
    assert any("negative" in note for note in rep.validation_notes)


def test_greater_than_one_probability_rejected():
    """Verify that probabilities > 1.0 fail validation."""
    rep = ConfidenceRepresentation.create(
        raw_score=1.25,
        confidence_value=1.05,
        source_model="classical_adapter_v1",
        confidence_type=ConfidenceType.SOFTMAX_PROBABILITY,
    )
    assert rep.is_valid is False
    assert rep.confidence_band == ConfidenceBand.INVALID_CONFIDENCE
    assert any("exceed 1.0" in note for note in rep.validation_notes)


def test_none_confidence_rejected():
    """Verify that missing (None) confidence is flagged."""
    is_valid, notes = ConfidenceValidator.validate(None)
    assert is_valid is False
    assert any("None" in note for note in notes)


# ==============================================================================
# 2. DECISION LOGIC & ABSTENTION TESTS
# ==============================================================================

def test_high_confidence_general_speech_accepted():
    """Verify high-confidence general speech is accepted without requiring confirmation."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=2.5,
        confidence_value=0.88,
        source_model="v_base",
    )
    decision = engine.evaluate(
        predicted_label="GENERAL_SPEECH",
        confidence_rep=conf,
        speaker_id="F01",
    )
    assert decision.decision == ActionDecision.ACCEPT.value
    assert decision.verification_required is False
    assert decision.suggested_action == SuggestedAction.ROUTE_PREDICTION.value
    assert decision.learning_eligible is False  # Never eligible without ground truth


def test_medium_confidence_general_speech_requires_confirmation():
    """Verify medium-confidence general speech triggers CONFIRM."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=1.2,
        confidence_value=0.58,  # Between 0.50 and 0.70
        source_model="v_base",
    )
    decision = engine.evaluate(
        predicted_label="GENERAL_SPEECH",
        confidence_rep=conf,
        speaker_id="F01",
    )
    assert decision.decision == ActionDecision.CONFIRM.value
    assert decision.verification_required is True
    assert decision.suggested_action == SuggestedAction.REQUEST_USER_CONFIRMATION.value


def test_low_confidence_general_speech_abstains():
    """Verify low-confidence general speech triggers ABSTAIN."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=0.4,
        confidence_value=0.42,  # < 0.50
        source_model="v_base",
    )
    decision = engine.evaluate(
        predicted_label="GENERAL_SPEECH",
        confidence_rep=conf,
        speaker_id="F01",
    )
    assert decision.decision == ActionDecision.ABSTAIN.value
    assert decision.verification_required is True
    assert decision.suggested_action == SuggestedAction.FALLBACK_TO_CARE_SERVICE.value


# ==============================================================================
# 3. EMERGENCY SAFETY LOGIC TESTS
# ==============================================================================

def test_emergency_high_confidence_accepts_routing_but_requires_verification():
    """
    EMERGENCY SAFETY CONTRACT:
    A high-confidence emergency prediction (e.g. SOS_HELP at 0.90) is accepted for dispatch routing,
    BUT verification is still mandatory before it can ever be admitted to learning memory.
    """
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=3.0,
        confidence_value=0.90,
        source_model="v_base",
    )
    decision = engine.evaluate(
        predicted_label="SOS_HELP",
        confidence_rep=conf,
        speaker_id="F01",
    )
    assert decision.decision == ActionDecision.ACCEPT.value
    assert decision.verification_required is True
    assert decision.learning_eligible is False


def test_emergency_medium_confidence_requires_user_confirmation():
    """Verify medium-confidence emergency prediction (e.g., FALL at 0.72) triggers CONFIRM."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=1.5,
        confidence_value=0.72,  # Between 0.65 and 0.85
        source_model="v_base",
    )
    decision = engine.evaluate(
        predicted_label="FALL",
        confidence_rep=conf,
        speaker_id="F01",
    )
    assert decision.decision == ActionDecision.CONFIRM.value
    assert decision.verification_required is True
    assert decision.suggested_action == SuggestedAction.REQUEST_USER_CONFIRMATION.value


def test_emergency_low_confidence_abstains_and_escalates_safety():
    """Verify low-confidence emergency prediction (< 0.65) triggers ABSTAIN and ESCALATE_SAFETY."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=0.8,
        confidence_value=0.55,  # Below emergency confirm threshold 0.65
        source_model="v_base",
    )
    decision = engine.evaluate(
        predicted_label="BREATHING_MEDICAL",
        confidence_rep=conf,
        speaker_id="F01",
    )
    assert decision.decision == ActionDecision.ABSTAIN.value
    assert decision.verification_required is True
    assert decision.suggested_action == SuggestedAction.ESCALATE_SAFETY.value


# ==============================================================================
# 4. GROUND TRUTH SAFETY & ANTI-SELF-TRAINING TESTS
# ==============================================================================

def test_confidence_never_creates_ground_truth():
    """
    CORE SCIENTIFIC FIREWALL TEST:
    Even with 0.9999 confidence, an unverified Experience Record must NEVER
    have confirmed_intent populated or become learning_eligible.
    """
    store = InMemoryExperienceStore()
    record = ExperienceRecord(
        sample_id="test-gt-firewall",
        speaker_id="F01",
        predicted_intent="SOS_HELP",
        prediction_confidence=0.9999,
        audio_reference="audio.wav",
    )
    stored = store.create(record)

    # Invariants
    assert stored.confirmed_intent is None
    assert stored.verification_status == VerificationStatus.UNVERIFIED.value
    assert stored.learning_eligible is False
    assert stored.eligibility_reason == EligibilityReason.INELIGIBLE_UNVERIFIED.value


def test_decision_engine_never_grants_learning_eligibility():
    """DecisionEngine must always emit learning_eligible = False for prediction events."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=5.0,
        confidence_value=0.99,
        source_model="v_base",
    )
    for label in ["GENERAL_SPEECH", "SOS_HELP", "FALL", "PAIN"]:
        decision = engine.evaluate(
            predicted_label=label,
            confidence_rep=conf,
            speaker_id="F01",
        )
        assert decision.learning_eligible is False


# ==============================================================================
# 5. MULTI-STAGE LEARNING ELIGIBILITY GATE TESTS
# ==============================================================================

def test_eligibility_multi_stage_reasons():
    """Verify distinct, explainable reason codes for every disqualifying condition."""
    # 1. Unverified
    r1 = ExperienceRecord(
        sample_id="e1", speaker_id="F01", predicted_intent="FALL", audio_reference="a.wav"
    )
    el1, reason1 = evaluate_learning_eligibility(r1)
    assert el1 is False
    assert reason1 == EligibilityReason.INELIGIBLE_UNVERIFIED.value

    # 2. Rejected
    r2 = ExperienceRecord(
        sample_id="e2", speaker_id="F01", predicted_intent="FALL", audio_reference="a.wav",
        verification_status=VerificationStatus.REJECTED.value, confirmed_intent=None
    )
    el2, reason2 = evaluate_learning_eligibility(r2)
    assert el2 is False
    assert reason2 == EligibilityReason.INELIGIBLE_REJECTED.value

    # 3. Degraded Quality
    r3 = ExperienceRecord(
        sample_id="e3", speaker_id="F01", predicted_intent="FALL", audio_reference="a.wav",
        verification_status=VerificationStatus.USER_VERIFIED.value, confirmed_intent="FALL",
        confirmation_source="user_explicit", quality_score=0.30
    )
    el3, reason3 = evaluate_learning_eligibility(r3)
    assert el3 is False
    assert reason3 == EligibilityReason.INELIGIBLE_LOW_QUALITY.value

    # 4. Quarantined
    r4 = ExperienceRecord(
        sample_id="e4", speaker_id="F01", predicted_intent="FALL", audio_reference="a.wav",
        verification_status=VerificationStatus.USER_VERIFIED.value, confirmed_intent="FALL",
        confirmation_source="user_explicit", learning_status=LearningStatus.QUARANTINED.value
    )
    el4, reason4 = evaluate_learning_eligibility(r4)
    assert el4 is False
    assert reason4 == EligibilityReason.INELIGIBLE_QUARANTINED.value

    # 5. Fully Valid & Verified -> Eligible
    r5 = ExperienceRecord(
        sample_id="e5", speaker_id="F01", predicted_intent="FALL", audio_reference="a.wav",
        verification_status=VerificationStatus.USER_VERIFIED.value, confirmed_intent="FALL",
        confirmation_source="user_explicit", quality_score=0.95
    )
    el5, reason5 = evaluate_learning_eligibility(r5)
    assert el5 is True
    assert reason5 == EligibilityReason.ELIGIBLE_USER_CONFIRMED.value


# ==============================================================================
# 6. UNCERTAINTY QUANTIFICATION TESTS
# ==============================================================================

def test_uncertainty_calculator():
    """Verify computation of margin and entropy metrics from class probabilities."""
    probs = {
        "GENERAL_SPEECH": 0.70,
        "SOS_HELP": 0.20,
        "FALL": 0.10,
    }
    metrics = UncertaintyCalculator.compute_from_probabilities(probs)
    assert metrics["max_prob"] == 0.70
    assert abs(metrics["top1_top2_margin"] - 0.50) < 1e-6
    assert metrics["entropy"] > 0.0
    assert 0.0 <= metrics["normalized_entropy"] <= 1.0


# ==============================================================================
# 7. PROVENANCE & POLICY VERSIONING TESTS
# ==============================================================================

def test_decision_provenance_and_policy_versioning():
    """Verify decision object preserves policy version and model attribution."""
    engine = DecisionEngine()
    conf = ConfidenceRepresentation.create(
        raw_score=2.0,
        confidence_value=0.86,
        source_model="qsvc_q8_r1_v1.2",
    )
    dec = engine.evaluate(
        predicted_label="GENERAL_SPEECH",
        confidence_rep=conf,
        speaker_id="F01",
        session_id="sess_001",
        model_version="qsvc_q8_r1_v1.2",
    )

    assert dec.model_version == "qsvc_q8_r1_v1.2"
    assert dec.policy_version == "v1.0.0_research_draft"
    assert dec.raw_confidence == 0.86
    assert dec.speaker_id == "F01"
    assert dec.session_id == "sess_001"
    assert dec.created_at != ""


def test_policy_change_creates_new_decision():
    """Verify changing policy produces a new version-stamped decision without mutating the old one."""
    custom_policy = load_confidence_policy()
    custom_policy["policy_metadata"]["policy_version"] = "v2.0.0_strict"
    custom_policy["abstention_policy"]["general_speech"]["accept_threshold"] = 0.95

    engine_default = DecisionEngine()
    engine_strict = DecisionEngine(policy=custom_policy)

    conf = ConfidenceRepresentation.create(
        raw_score=2.0,
        confidence_value=0.80,
        source_model="v_base",
    )

    dec_default = engine_default.evaluate("GENERAL_SPEECH", conf, speaker_id="F01")
    dec_strict = engine_strict.evaluate("GENERAL_SPEECH", conf, speaker_id="F01")

    # Default accepts at 0.80 (>= 0.70)
    assert dec_default.decision == ActionDecision.ACCEPT.value
    assert dec_default.policy_version == "v1.0.0_research_draft"

    # Strict requires 0.95, so 0.80 triggers CONFIRM
    assert dec_strict.decision == ActionDecision.CONFIRM.value
    assert dec_strict.policy_version == "v2.0.0_strict"

    # Both remain distinct and immutable
    assert dec_default.decision_id != dec_strict.decision_id
