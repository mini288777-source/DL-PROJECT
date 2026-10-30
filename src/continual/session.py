"""
Session Model for Continual Personalization (Phase 7.1).
Groups speaker-specific speech interactions into chronologically ordered research sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.continual.schema import ExperienceRecord, VerificationStatus


class SessionSpeakerMismatchError(Exception):
    """Raised when an interaction from one speaker is added to another speaker's session."""
    pass


@dataclass
class PersonalizationSession:
    """
    Research session grouping speech interactions for an individual speaker.
    Tracks chronological sequence, active model version, and session provenance.
    """
    session_id: str
    speaker_id: str
    model_version: str
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    records: List[ExperienceRecord] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def add_record(self, record: ExperienceRecord) -> None:
        """
        Add an experience record to the session.
        Guarantees speaker isolation and chronological ordering.
        """
        if record.speaker_id != self.speaker_id:
            raise SessionSpeakerMismatchError(
                f"Cannot add record for speaker '{record.speaker_id}' to session '{self.session_id}' "
                f"owned by speaker '{self.speaker_id}'"
            )

        # Enforce session association
        record.session_id = self.session_id
        if not record.model_version or record.model_version == "v0.0.0":
            record.model_version = self.model_version

        self.records.append(record)
        # Keep chronologically sorted
        self.records.sort(key=lambda r: r.timestamp)

    def get_records(self) -> List[ExperienceRecord]:
        """Return all records in chronological order."""
        return list(self.records)

    def get_verified_records(self) -> List[ExperienceRecord]:
        """Return all verified records in session."""
        return [
            r for r in self.records
            if r.verification_status in (
                VerificationStatus.USER_VERIFIED.value,
                VerificationStatus.SYSTEM_VERIFIED.value,
            )
        ]

    def get_eligible_records(self) -> List[ExperienceRecord]:
        """Return all learning-eligible records in session."""
        return [r for r in self.records if r.learning_eligible]

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to serializable dictionary."""
        return {
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalizationSession":
        """Instantiate session from dictionary."""
        records = [ExperienceRecord.from_dict(r) for r in data.get("records", [])]
        return cls(
            session_id=data["session_id"],
            speaker_id=data["speaker_id"],
            model_version=data["model_version"],
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
            records=records,
        )
