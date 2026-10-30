"""
Experience Store Interface & Implementations (Phase 7.1).
Provides abstract ExperienceStore, InMemoryExperienceStore, and PersistentExperienceStore.
Guarantees speaker isolation, audit logging, duplicate prevention, and imbalance awareness.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.continual.duplicate import DuplicateDetector, DuplicatePolicy, DuplicateRecordError
from src.continual.eligibility import evaluate_learning_eligibility
from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.session import PersonalizationSession
from src.continual.taxonomy import is_valid_intent, CANONICAL_CLASSES
from src.continual.validation import assert_valid_experience_record


class RecordNotFoundError(Exception):
    """Raised when an Experience Record is not found by ID."""
    pass


class SpeakerIsolationError(Exception):
    """Raised when attempting an operation that breaches speaker scoping."""
    pass


class ExperienceStore(ABC):
    """Abstract interface defining all Experience Store operations."""

    @abstractmethod
    def create(self, record: ExperienceRecord) -> ExperienceRecord:
        """Validate, register, and store a new speech experience."""
        pass

    @abstractmethod
    def read(self, sample_id: str) -> Optional[ExperienceRecord]:
        """Retrieve an experience record by sample ID."""
        pass

    @abstractmethod
    def update(self, record: ExperienceRecord) -> ExperienceRecord:
        """Update an existing record with validation and audit logging."""
        pass

    @abstractmethod
    def mark_verified(
        self,
        sample_id: str,
        verified_label: str,
        verification_source: str,
        verifier_id: Optional[str] = None,
        note: str = "User / clinical verification received",
    ) -> ExperienceRecord:
        """Transition record to verified state, re-evaluating eligibility without overwriting prediction."""
        pass

    @abstractmethod
    def mark_rejected(
        self,
        sample_id: str,
        reason: str,
        verifier_id: Optional[str] = None,
    ) -> ExperienceRecord:
        """Transition record to rejected state, disqualifying it from future learning."""
        pass

    @abstractmethod
    def check_eligibility(self, sample_id: str) -> Tuple[bool, str]:
        """Check if an existing experience record is eligible for learning."""
        pass

    @abstractmethod
    def list_by_speaker(self, speaker_id: str) -> List[ExperienceRecord]:
        """Retrieve all records for a specific speaker in chronological order."""
        pass

    @abstractmethod
    def list_by_session(self, session_id: str) -> List[ExperienceRecord]:
        """Retrieve all records for a specific session."""
        pass

    @abstractmethod
    def list_verified(self, speaker_id: Optional[str] = None) -> List[ExperienceRecord]:
        """List all verified records (optionally scoped to a speaker)."""
        pass

    @abstractmethod
    def list_learning_eligible(self, speaker_id: Optional[str] = None) -> List[ExperienceRecord]:
        """List all records authorized for candidate adaptation sets."""
        pass

    @abstractmethod
    def count_by_class(
        self, speaker_id: Optional[str] = None, verified_only: bool = False
    ) -> Dict[str, int]:
        """Count records per canonical class."""
        pass

    @abstractmethod
    def count_by_speaker() -> Dict[str, int]:
        """Count total records per speaker."""
        pass

    @abstractmethod
    def count_by_verification_status(
        self, speaker_id: Optional[str] = None
    ) -> Dict[str, int]:
        """Count records by verification status."""
        pass

    @abstractmethod
    def get_speaker_class_matrix(self) -> Dict[str, Dict[str, int]]:
        """Return 2D cross-tabulation of speaker x class counts."""
        pass

    @abstractmethod
    def get_statistics(self, speaker_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute comprehensive diagnostic statistics for research audit."""
        pass

    @abstractmethod
    def check_duplicate(self, record: ExperienceRecord) -> Tuple[bool, Optional[str]]:
        """Check if a record is a duplicate."""
        pass

    @abstractmethod
    def archive(self, sample_id: str) -> ExperienceRecord:
        """Archive a record."""
        pass


class InMemoryExperienceStore(ExperienceStore):
    """
    In-memory, deterministic implementation of ExperienceStore.
    Ideal for unit testing, fast simulated streams, and hermetic environments.
    """

    def __init__(self, duplicate_policy: DuplicatePolicy = DuplicatePolicy.REJECT):
        self.duplicate_policy = duplicate_policy
        self.detector = DuplicateDetector(policy=duplicate_policy)
        # Store primary index: sample_id -> ExperienceRecord
        self._records: Dict[str, ExperienceRecord] = {}
        # Speaker index: speaker_id -> List[sample_id]
        self._by_speaker: Dict[str, List[str]] = defaultdict(list)
        # Session index: session_id -> List[sample_id]
        self._by_session: Dict[str, List[str]] = defaultdict(list)

    def create(self, record: ExperienceRecord) -> ExperienceRecord:
        """Validate and insert record into memory store."""
        # 1. Structural validation
        assert_valid_experience_record(record)

        # 2. Duplicate detection
        is_dup, existing_id = self.detector.check_duplicate(record)
        if is_dup:
            if self.duplicate_policy == DuplicatePolicy.REJECT:
                raise DuplicateRecordError(
                    f"Record with ID '{record.sample_id}' or duplicate hash matches existing '{existing_id}'"
                )
            elif self.duplicate_policy == DuplicatePolicy.IDEMPOTENT_IGNORE:
                return self._records[existing_id]  # type: ignore

        # 3. Evaluate eligibility state
        eligible, reason = evaluate_learning_eligibility(record)
        record.learning_eligible = eligible
        record.eligibility_reason = reason

        # 4. Register
        self.detector.register(record)
        self._records[record.sample_id] = record
        self._by_speaker[record.speaker_id].append(record.sample_id)
        self._by_session[record.session_id].append(record.sample_id)
        return record

    def read(self, sample_id: str) -> Optional[ExperienceRecord]:
        return self._records.get(sample_id)

    def update(self, record: ExperienceRecord) -> ExperienceRecord:
        if record.sample_id not in self._records:
            raise RecordNotFoundError(f"Cannot update non-existent record '{record.sample_id}'")

        assert_valid_experience_record(record)
        eligible, reason = evaluate_learning_eligibility(record)
        record.learning_eligible = eligible
        record.eligibility_reason = reason
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._records[record.sample_id] = record
        return record

    def mark_verified(
        self,
        sample_id: str,
        verified_label: str,
        verification_source: str,
        verifier_id: Optional[str] = None,
        note: str = "User / clinical verification received",
    ) -> ExperienceRecord:
        record = self.read(sample_id)
        if not record:
            raise RecordNotFoundError(f"Record '{sample_id}' not found for verification")

        if not is_valid_intent(verified_label):
            raise ValueError(
                f"Cannot verify with invalid intent '{verified_label}'. Must be one of {CANONICAL_CLASSES}"
            )

        from_verif = record.verification_status
        from_learn = record.learning_status

        # Determine target verification status based on source
        if verification_source in ("user_explicit", "clinician_verified"):
            to_verif = VerificationStatus.USER_VERIFIED.value
            to_learn = LearningStatus.USER_CONFIRMED.value
        elif verification_source in ("oracle_ground_truth", "oracle_eval"):
            to_verif = VerificationStatus.SYSTEM_VERIFIED.value
            to_learn = LearningStatus.USER_CONFIRMED.value
        else:
            to_verif = VerificationStatus.USER_VERIFIED.value
            to_learn = LearningStatus.USER_CONFIRMED.value

        # Record audit trail entry
        record.add_audit_entry(
            from_status=from_verif,
            to_status=to_verif,
            from_learning_status=from_learn,
            to_learning_status=to_learn,
            confirmed_intent=verified_label,
            verifier_id=verifier_id,
            note=note,
        )

        record.verification_status = to_verif
        record.learning_status = to_learn
        record.confirmed_intent = verified_label
        record.confirmation_source = verification_source
        record.verifier_id = verifier_id
        record.verification_timestamp = datetime.now(timezone.utc).isoformat()

        # Re-evaluate eligibility
        eligible, reason = evaluate_learning_eligibility(record)
        record.learning_eligible = eligible
        record.eligibility_reason = reason

        return self.update(record)

    def mark_rejected(
        self,
        sample_id: str,
        reason: str,
        verifier_id: Optional[str] = None,
    ) -> ExperienceRecord:
        record = self.read(sample_id)
        if not record:
            raise RecordNotFoundError(f"Record '{sample_id}' not found for rejection")

        from_verif = record.verification_status
        from_learn = record.learning_status

        record.add_audit_entry(
            from_status=from_verif,
            to_status=VerificationStatus.REJECTED.value,
            from_learning_status=from_learn,
            to_learning_status=LearningStatus.REJECTED.value,
            confirmed_intent=None,
            verifier_id=verifier_id,
            note=f"Rejected: {reason}",
        )

        record.verification_status = VerificationStatus.REJECTED.value
        record.learning_status = LearningStatus.REJECTED.value
        record.learning_eligible = False
        record.eligibility_reason = EligibilityReason.INELIGIBLE_REJECTED.value
        record.exclusion_reason = reason
        record.verifier_id = verifier_id
        record.verification_timestamp = datetime.now(timezone.utc).isoformat()

        return self.update(record)

    def check_eligibility(self, sample_id: str) -> Tuple[bool, str]:
        record = self.read(sample_id)
        if not record:
            raise RecordNotFoundError(f"Record '{sample_id}' not found")
        return evaluate_learning_eligibility(record)

    def list_by_speaker(self, speaker_id: str) -> List[ExperienceRecord]:
        """Return speaker-scoped records sorted chronologically."""
        if not speaker_id or not speaker_id.strip():
            raise SpeakerIsolationError("speaker_id must be provided to retrieve speaker records")
        ids = self._by_speaker.get(speaker_id, [])
        records = [self._records[i] for i in ids]
        records.sort(key=lambda r: r.timestamp)
        return records

    def list_by_session(self, session_id: str) -> List[ExperienceRecord]:
        ids = self._by_session.get(session_id, [])
        records = [self._records[i] for i in ids]
        records.sort(key=lambda r: r.timestamp)
        return records

    def list_verified(self, speaker_id: Optional[str] = None) -> List[ExperienceRecord]:
        all_recs = self.list_by_speaker(speaker_id) if speaker_id else list(self._records.values())
        return [
            r for r in all_recs
            if r.verification_status in (
                VerificationStatus.USER_VERIFIED.value,
                VerificationStatus.SYSTEM_VERIFIED.value,
            )
        ]

    def list_learning_eligible(self, speaker_id: Optional[str] = None) -> List[ExperienceRecord]:
        all_recs = self.list_by_speaker(speaker_id) if speaker_id else list(self._records.values())
        return [r for r in all_recs if r.learning_eligible]

    def count_by_class(
        self, speaker_id: Optional[str] = None, verified_only: bool = False
    ) -> Dict[str, int]:
        records = self.list_by_speaker(speaker_id) if speaker_id else list(self._records.values())
        counts: Dict[str, int] = {c: 0 for c in CANONICAL_CLASSES}
        for r in records:
            if verified_only:
                if (
                    r.verification_status in (
                        VerificationStatus.USER_VERIFIED.value,
                        VerificationStatus.SYSTEM_VERIFIED.value,
                    )
                    and r.confirmed_intent
                ):
                    counts[r.confirmed_intent] = counts.get(r.confirmed_intent, 0) + 1
            else:
                counts[r.predicted_intent] = counts.get(r.predicted_intent, 0) + 1
        return counts

    def count_by_speaker(self) -> Dict[str, int]:
        return {spk: len(ids) for spk, ids in self._by_speaker.items()}

    def count_by_verification_status(
        self, speaker_id: Optional[str] = None
    ) -> Dict[str, int]:
        records = self.list_by_speaker(speaker_id) if speaker_id else list(self._records.values())
        counts = {s.value: 0 for s in VerificationStatus}
        for r in records:
            counts[r.verification_status] = counts.get(r.verification_status, 0) + 1
        return counts

    def get_speaker_class_matrix(self) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {}
        for spk in sorted(self._by_speaker.keys()):
            matrix[spk] = self.count_by_class(speaker_id=spk, verified_only=False)
        return matrix

    def get_statistics(self, speaker_id: Optional[str] = None) -> Dict[str, Any]:
        records = self.list_by_speaker(speaker_id) if speaker_id else list(self._records.values())
        total = len(records)
        verified = sum(
            1 for r in records
            if r.verification_status in (
                VerificationStatus.USER_VERIFIED.value,
                VerificationStatus.SYSTEM_VERIFIED.value,
            )
        )
        eligible = sum(1 for r in records if r.learning_eligible)
        unverified = sum(1 for r in records if r.verification_status == VerificationStatus.UNVERIFIED.value)
        rejected = sum(1 for r in records if r.verification_status == VerificationStatus.REJECTED.value)

        return {
            "total_experiences": total,
            "verified_experiences": verified,
            "eligible_experiences": eligible,
            "unverified_experiences": unverified,
            "rejected_experiences": rejected,
            "counts_by_class_predicted": self.count_by_class(speaker_id=speaker_id, verified_only=False),
            "counts_by_class_verified": self.count_by_class(speaker_id=speaker_id, verified_only=True),
            "counts_by_verification_status": self.count_by_verification_status(speaker_id=speaker_id),
            "unique_speakers": len(self._by_speaker) if speaker_id is None else 1,
        }

    def check_duplicate(self, record: ExperienceRecord) -> Tuple[bool, Optional[str]]:
        return self.detector.check_duplicate(record)

    def archive(self, sample_id: str) -> ExperienceRecord:
        record = self.read(sample_id)
        if not record:
            raise RecordNotFoundError(f"Record '{sample_id}' not found for archiving")
        record.learning_status = "ARCHIVED"
        record.learning_eligible = False
        record.eligibility_reason = "ARCHIVED"
        return self.update(record)


class PersistentExperienceStore(InMemoryExperienceStore):
    """
    File-backed, persistent implementation of ExperienceStore.
    Persists speaker-partitioned JSON Lines files:
        {storage_dir}/experiences/{speaker_id}.jsonl
    Safe, appendable, portable, git-friendly, zero external database dependencies.
    """

    def __init__(
        self,
        storage_dir: str = "data/continual/experiences",
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.REJECT,
    ):
        super().__init__(duplicate_policy=duplicate_policy)
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load_all_speakers()

    def _get_speaker_file(self, speaker_id: str) -> str:
        return os.path.join(self.storage_dir, f"{speaker_id}.jsonl")

    def _load_all_speakers(self) -> None:
        """Load existing records from disk upon initialization."""
        if not os.path.exists(self.storage_dir):
            return

        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".jsonl"):
                fpath = os.path.join(self.storage_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            rec = ExperienceRecord.from_dict(data)
                            # Direct insertion into memory index without re-triggering file write
                            self.detector.register(rec)
                            self._records[rec.sample_id] = rec
                            self._by_speaker[rec.speaker_id].append(rec.sample_id)
                            self._by_session[rec.session_id].append(rec.sample_id)

    def _persist_speaker(self, speaker_id: str) -> None:
        """Rewrite speaker's partition atomically."""
        fpath = self._get_speaker_file(speaker_id)
        temp_fpath = f"{fpath}.tmp"
        records = self.list_by_speaker(speaker_id)

        with open(temp_fpath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Atomic replacement
        if os.path.exists(fpath):
            os.replace(temp_fpath, fpath)
        else:
            os.rename(temp_fpath, fpath)

    def create(self, record: ExperienceRecord) -> ExperienceRecord:
        saved = super().create(record)
        self._persist_speaker(saved.speaker_id)
        return saved

    def update(self, record: ExperienceRecord) -> ExperienceRecord:
        updated = super().update(record)
        self._persist_speaker(updated.speaker_id)
        return updated
