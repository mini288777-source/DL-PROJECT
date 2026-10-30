"""
Data Quality Validation for Continual Personalization Records (Phase 7.1).
Ensures every Experience Record meets strict semantic, taxonomic, and structural constraints
before persistence.
"""

from datetime import datetime
from typing import List
from src.continual.schema import ExperienceRecord, VerificationStatus, LearningStatus
from src.continual.taxonomy import is_valid_intent, CANONICAL_CLASSES


class ValidationError(Exception):
    """Raised when an Experience Record fails data quality validation."""
    def __init__(self, errors: List[str]):
        super().__init__(f"Experience Record validation failed with {len(errors)} error(s): {'; '.join(errors)}")
        self.errors = errors


def validate_experience_record(record: ExperienceRecord) -> List[str]:
    """
    Validate all structural, taxonomic, and scientific constraints on an ExperienceRecord.
    Returns a list of error descriptions (empty list if perfectly valid).
    """
    errors: List[str] = []

    # 1. Identity validation
    if not record.sample_id or not isinstance(record.sample_id, str) or not record.sample_id.strip():
        errors.append("sample_id must be a non-empty string")

    if not record.speaker_id or not isinstance(record.speaker_id, str) or not record.speaker_id.strip():
        errors.append("speaker_id must be a non-empty string")

    if not record.session_id or not isinstance(record.session_id, str) or not record.session_id.strip():
        errors.append("session_id must be a non-empty string")

    # 2. Timestamp validation (ISO 8601)
    if not record.timestamp or not isinstance(record.timestamp, str):
        errors.append("timestamp must be a valid ISO 8601 string")
    else:
        try:
            # Check ISO format compatibility
            datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            errors.append(f"timestamp '{record.timestamp}' is not a valid ISO 8601 string")

    # 3. Model version reference
    if not record.model_version or not isinstance(record.model_version, str) or not record.model_version.strip():
        errors.append("model_version must be a non-empty string")

    # 4. Taxonomic validation of predicted_intent
    if not record.predicted_intent or not is_valid_intent(record.predicted_intent):
        errors.append(
            f"predicted_intent '{record.predicted_intent}' is invalid; must be one of {CANONICAL_CLASSES}"
        )

    # 5. Taxonomic validation of confirmed_intent (if provided)
    if record.confirmed_intent is not None:
        if not is_valid_intent(record.confirmed_intent):
            errors.append(
                f"confirmed_intent '{record.confirmed_intent}' is invalid; must be one of {CANONICAL_CLASSES}"
            )

    # 6. Confidence range [0.0, 1.0]
    if not isinstance(record.prediction_confidence, (int, float)) or not (0.0 <= record.prediction_confidence <= 1.0):
        errors.append(f"prediction_confidence must be a float in [0.0, 1.0], got {record.prediction_confidence}")

    if record.emergency_probability is not None:
        if not isinstance(record.emergency_probability, (int, float)) or not (0.0 <= record.emergency_probability <= 1.0):
            errors.append(f"emergency_probability must be a float in [0.0, 1.0], got {record.emergency_probability}")

    # 7. Quality and novelty score ranges [0.0, 1.0]
    if not isinstance(record.quality_score, (int, float)) or not (0.0 <= record.quality_score <= 1.0):
        errors.append(f"quality_score must be a float in [0.0, 1.0], got {record.quality_score}")

    if not isinstance(record.novelty_score, (int, float)) or not (0.0 <= record.novelty_score <= 1.0):
        errors.append(f"novelty_score must be a float in [0.0, 1.0], got {record.novelty_score}")

    # 8. Status enum validity
    valid_verif = {s.value for s in VerificationStatus}
    if record.verification_status not in valid_verif:
        errors.append(f"verification_status '{record.verification_status}' invalid; must be in {valid_verif}")

    valid_learning = {s.value for s in LearningStatus}
    if record.learning_status not in valid_learning:
        errors.append(f"learning_status '{record.learning_status}' invalid; must be in {valid_learning}")

    # 9. Provenance reference check (must have audio or embedding reference)
    if not record.audio_reference and not record.embedding_reference:
        errors.append("Record must have at least one valid audio_reference or embedding_reference")

    return errors


def assert_valid_experience_record(record: ExperienceRecord) -> None:
    """Validate record and raise ValidationError if any errors exist."""
    errors = validate_experience_record(record)
    if errors:
        raise ValidationError(errors)
