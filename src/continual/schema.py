"""
Experience Record Schema & Verification Data Model (Phase 7.1).
Defines strongly-typed data structures, verification states, learning statuses,
and audit trails for speaker-specific continual learning experience.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    """Explicit verification state of a speech interaction."""
    UNVERIFIED = "UNVERIFIED"
    USER_VERIFIED = "USER_VERIFIED"
    SYSTEM_VERIFIED = "SYSTEM_VERIFIED"  # E.g., verified test oracle in simulated continual loop
    REJECTED = "REJECTED"
    INVALID = "INVALID"


class LearningStatus(str, Enum):
    """Canonical learning status matching Phase 7.0 research protocol."""
    UNREVIEWED = "UNREVIEWED"
    AUTO_ELIGIBLE = "AUTO_ELIGIBLE"
    USER_CONFIRMED = "USER_CONFIRMED"
    REJECTED = "REJECTED"
    QUARANTINED = "QUARANTINED"
    USED_FOR_TRAINING = "USED_FOR_TRAINING"
    VALIDATION_ONLY = "VALIDATION_ONLY"


class EligibilityReason(str, Enum):
    """Standardized reasons for learning eligibility or ineligibility."""
    ELIGIBLE = "ELIGIBLE"
    ELIGIBLE_USER_CONFIRMED = "ELIGIBLE_USER_CONFIRMED"
    ELIGIBLE_SYSTEM_VERIFIED = "ELIGIBLE_SYSTEM_VERIFIED"
    INELIGIBLE_UNVERIFIED = "INELIGIBLE_UNVERIFIED"
    INELIGIBLE_REJECTED = "INELIGIBLE_REJECTED"
    INELIGIBLE_REJECTED_SAMPLE = "INELIGIBLE_REJECTED_SAMPLE"
    INELIGIBLE_INVALID_SCHEMA = "INELIGIBLE_INVALID_SCHEMA"
    INELIGIBLE_SCHEMA_ERROR = "INELIGIBLE_SCHEMA_ERROR"
    INELIGIBLE_INVALID_TAXONOMY = "INELIGIBLE_INVALID_TAXONOMY"
    INELIGIBLE_INVALID_PROVENANCE = "INELIGIBLE_INVALID_PROVENANCE"
    INELIGIBLE_LOW_QUALITY = "INELIGIBLE_LOW_QUALITY"
    INELIGIBLE_LOW_CONFIDENCE = "INELIGIBLE_LOW_CONFIDENCE"
    INELIGIBLE_SAFETY_POLICY = "INELIGIBLE_SAFETY_POLICY"
    INELIGIBLE_MISSING_SPEAKER = "INELIGIBLE_MISSING_SPEAKER"
    INELIGIBLE_QUARANTINED = "INELIGIBLE_QUARANTINED"
    INELIGIBLE_INVALID_MODEL_VERSION = "INELIGIBLE_INVALID_MODEL_VERSION"
    INELIGIBLE_SELF_PREDICTION_PROHIBITED = "INELIGIBLE_SELF_PREDICTION_PROHIBITED"


@dataclass
class AuditEntry:
    """Record of state mutation or verification transition for complete auditability."""
    transition_timestamp: str
    from_status: str
    to_status: str
    from_learning_status: str
    to_learning_status: str
    confirmed_intent: Optional[str]
    verifier_id: Optional[str]
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        return cls(**data)


@dataclass
class ExperienceRecord:
    """
    Canonical Experience Record for Continual Personalization.
    Encapsulates raw utterance provenance, model prediction, verification state,
    eligibility status, and cryptographic integrity.
    """
    # Primary Identity
    sample_id: str
    speaker_id: str
    session_id: str = "session_default"
    timestamp: str = ""
    schema_version: str = "1.0.0"

    # Input Provenance
    audio_reference: str = ""
    embedding_reference: str = ""
    data_source: str = "torgo_stream"  # torgo_stream, held_out_test, interactive_eval
    preprocessing_version: str = "whisper_tiny_unit_L2"
    representation_version: str = "whisper_tiny_384_unit_L2"

    # Speech Metadata
    duration: Optional[float] = None
    sample_rate: Optional[int] = 16000
    channel_count: Optional[int] = 1

    # Prediction Layer
    predicted_intent: str = ""
    prediction_confidence: float = 0.0
    emergency_probability: Optional[float] = None
    model_version: str = "v0.0.0"
    inference_mode: str = "hierarchical"

    # Verification Layer
    verification_status: str = VerificationStatus.UNVERIFIED.value
    confirmed_intent: Optional[str] = None
    confirmation_source: str = "unconfirmed"  # user_explicit, clinician_verified, oracle_ground_truth
    verification_timestamp: Optional[str] = None
    verifier_id: Optional[str] = None

    # Learning Eligibility Layer
    learning_status: str = LearningStatus.UNREVIEWED.value
    learning_eligible: bool = False
    eligibility_reason: str = EligibilityReason.INELIGIBLE_UNVERIFIED.value
    exclusion_reason: Optional[str] = None

    # Quality and Environment
    quality_score: float = 1.0  # [0.0, 1.0] acoustic cleanness / SNR metric
    novelty_score: float = 0.0  # [0.0, 1.0] representation distance from support
    data_quality_status: str = "CLEAN"  # CLEAN, DEGRADED, QUARANTINED
    provenance_status: str = "VERIFIED"
    environment_metadata: Dict[str, Any] = field(default_factory=dict)

    # Cryptographic Lineage & Audit
    integrity_hash: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        now_iso = datetime.now(timezone.utc).isoformat()
        if not self.timestamp:
            self.timestamp = now_iso
        if not self.created_at:
            self.created_at = self.timestamp
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.integrity_hash is None:
            self.integrity_hash = self.compute_integrity_hash()

    @property
    def experience_id(self) -> str:
        """Alias for sample_id."""
        return self.sample_id

    @property
    def predicted_label(self) -> str:
        """Alias for predicted_intent."""
        return self.predicted_intent

    @property
    def verified_label(self) -> Optional[str]:
        """Alias for confirmed_intent."""
        return self.confirmed_intent

    @property
    def verification_source(self) -> str:
        """Alias for confirmation_source."""
        return self.confirmation_source

    @property
    def source_type(self) -> str:
        """Alias for data_source."""
        return self.data_source

    @property
    def encoder_version(self) -> str:
        """Alias for representation_version."""
        return self.representation_version

    def compute_integrity_hash(self) -> str:
        """Compute SHA-256 fingerprint over core provenance and audio reference."""
        payload = f"{self.sample_id}:{self.speaker_id}:{self.session_id}:{self.audio_reference}:{self.timestamp}:{self.model_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def add_audit_entry(
        self,
        from_status: str,
        to_status: str,
        from_learning_status: str,
        to_learning_status: str,
        confirmed_intent: Optional[str],
        verifier_id: Optional[str],
        note: str,
    ) -> None:
        """Append an audit entry recording state changes."""
        entry = AuditEntry(
            transition_timestamp=datetime.now(timezone.utc).isoformat(),
            from_status=from_status,
            to_status=to_status,
            from_learning_status=from_learning_status,
            to_learning_status=to_learning_status,
            confirmed_intent=confirmed_intent,
            verifier_id=verifier_id,
            note=note,
        )
        self.audit_trail.append(entry.to_dict())
        self.updated_at = entry.transition_timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceRecord":
        """Instantiate record from dictionary, preserving all fields."""
        d = dict(data)
        return cls(**d)
