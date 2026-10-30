"""
Deterministic Learning Eligibility Engine (Phase 7.2).
Implements the multi-stage research learning gate:
Experience
    ↓
1. Schema valid?
    ↓
2. Provenance valid?
    ↓
3. Speaker valid?
    ↓
4. Prediction valid?
    ↓
5. Verification valid?
    ↓
6. Confidence policy satisfied?
    ↓
7. Safety policy satisfied?
    ↓
ELIGIBLE / INELIGIBLE (with explicit reason code)

CRITICAL INVARIANTS:
- A prediction is NOT ground truth.
- Confidence alone NEVER creates ground truth or confers learning eligibility.
- Unverified, degraded, quarantined, or rejected records are strictly barred from learning.
"""

import math
from typing import Any, Dict, Optional, Tuple
from src.continual.confidence import load_confidence_policy
from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import is_valid_intent, is_emergency_intent
from src.continual.validation import validate_experience_record


class EligibilityEngine:
    """
    Deterministic rule engine that checks whether an ExperienceRecord
    is authorized to participate in future personalization model updates.
    """

    @staticmethod
    def evaluate_eligibility(
        record: ExperienceRecord,
        policy: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate record against strict multi-stage eligibility contracts.
        Returns:
            (is_eligible: bool, reason_code: str)
        """
        cfg = policy or load_confidence_policy()

        # 1. Structural and schema validation
        val_errors = validate_experience_record(record)
        if val_errors:
            if any("sample_id" in e or "timestamp" in e or "session_id" in e for e in val_errors):
                return False, EligibilityReason.INELIGIBLE_SCHEMA_ERROR.value
            if any("predicted_intent" in e or "confirmed_intent" in e for e in val_errors):
                return False, EligibilityReason.INELIGIBLE_INVALID_TAXONOMY.value
            return False, EligibilityReason.INELIGIBLE_INVALID_SCHEMA.value

        # 2. Provenance validation
        if not record.audio_reference and not record.embedding_reference:
            return False, EligibilityReason.INELIGIBLE_INVALID_PROVENANCE.value
        if record.provenance_status == "FAILED":
            return False, EligibilityReason.INELIGIBLE_INVALID_PROVENANCE.value

        # 3. Speaker association invariant
        if not record.speaker_id or not record.speaker_id.strip():
            return False, EligibilityReason.INELIGIBLE_MISSING_SPEAKER.value

        # 4. Model version reference
        if not record.model_version or not record.model_version.strip():
            return False, EligibilityReason.INELIGIBLE_INVALID_MODEL_VERSION.value

        # 5. Quarantine check
        if (
            record.learning_status == LearningStatus.QUARANTINED.value
            or record.data_quality_status == "QUARANTINED"
        ):
            return False, EligibilityReason.INELIGIBLE_QUARANTINED.value

        # 6. Rejection check
        if (
            record.verification_status == VerificationStatus.REJECTED.value
            or record.learning_status == LearningStatus.REJECTED.value
        ):
            return False, EligibilityReason.INELIGIBLE_REJECTED.value

        # 7. Acoustic and data quality check
        min_quality = cfg.get("safety_invariants", {}).get("min_acoustic_quality_score", 0.50)
        if (
            record.data_quality_status == "DEGRADED"
            or record.quality_score < min_quality
        ):
            return False, EligibilityReason.INELIGIBLE_LOW_QUALITY.value

        # 8. Prediction taxonomic check
        if not is_valid_intent(record.predicted_intent):
            return False, EligibilityReason.INELIGIBLE_INVALID_TAXONOMY.value

        # 9. Verification validity check
        # CRITICAL INVARIANT: High prediction confidence alone CANNOT grant eligibility.
        if record.verification_status == VerificationStatus.UNVERIFIED.value:
            return False, EligibilityReason.INELIGIBLE_UNVERIFIED.value

        if record.verification_status == VerificationStatus.INVALID.value:
            return False, EligibilityReason.INELIGIBLE_INVALID_SCHEMA.value

        # Ground truth verification requirement
        if record.confirmed_intent is None:
            return False, EligibilityReason.INELIGIBLE_UNVERIFIED.value

        if not is_valid_intent(record.confirmed_intent):
            return False, EligibilityReason.INELIGIBLE_INVALID_TAXONOMY.value

        # 10. Confidence mathematical validity check
        if (
            record.prediction_confidence is None
            or math.isnan(record.prediction_confidence)
            or math.isinf(record.prediction_confidence)
            or record.prediction_confidence < 0.0
            or record.prediction_confidence > 1.0
        ):
            return False, EligibilityReason.INELIGIBLE_LOW_CONFIDENCE.value

        # 11. Safety policy check for emergency communication
        # If confirmed intent is emergency, ensure confirmation source is verified
        is_emergency = is_emergency_intent(record.confirmed_intent)
        emergency_requires_confirmation = cfg.get("safety_invariants", {}).get(
            "emergency_requires_user_confirmation", True
        )
        if is_emergency and emergency_requires_confirmation:
            valid_sources = {"user_explicit", "clinician_verified", "oracle_ground_truth", "oracle_eval"}
            if record.confirmation_source not in valid_sources:
                return False, EligibilityReason.INELIGIBLE_SAFETY_POLICY.value

        # 12. Explicit verification state approval
        if record.verification_status == VerificationStatus.USER_VERIFIED.value:
            return True, EligibilityReason.ELIGIBLE_USER_CONFIRMED.value

        if record.verification_status == VerificationStatus.SYSTEM_VERIFIED.value:
            return True, EligibilityReason.ELIGIBLE_SYSTEM_VERIFIED.value

        return False, EligibilityReason.INELIGIBLE_UNVERIFIED.value


def evaluate_learning_eligibility(
    record: ExperienceRecord, policy: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """Convenience functional interface for eligibility evaluation."""
    return EligibilityEngine.evaluate_eligibility(record, policy=policy)
