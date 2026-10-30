"""
Duplicate Protection Engine (Phase 7.1).
Provides deterministic duplicate detection by sample ID and cryptographic payload hash.
Prevents accidental double-insertion and corpus contamination.
"""

from enum import Enum
from typing import Dict, Optional, Set, Tuple
from src.continual.schema import ExperienceRecord


class DuplicatePolicy(str, Enum):
    """Behavior when a duplicate record is encountered during ingestion."""
    REJECT = "REJECT"                    # Raise DuplicateRecordError
    IDEMPOTENT_IGNORE = "IDEMPOTENT_IGNORE"  # Return existing record without modification
    VERSION = "VERSION"                  # Append incremented version suffix to sample_id


class DuplicateRecordError(Exception):
    """Raised when an Experience Record is already present in the store under REJECT policy."""
    pass


class DuplicateDetector:
    """
    Deterministic tracker of observed sample IDs and cryptographic integrity fingerprints.
    """

    def __init__(self, policy: DuplicatePolicy = DuplicatePolicy.REJECT):
        self.policy = policy
        self._sample_ids: Set[str] = set()
        self._integrity_hashes: Dict[str, str] = {}  # hash -> sample_id

    def check_duplicate(self, record: ExperienceRecord) -> Tuple[bool, Optional[str]]:
        """
        Check if record is already registered.
        Returns:
            (is_duplicate: bool, existing_sample_id: Optional[str])
        """
        if record.sample_id in self._sample_ids:
            return True, record.sample_id

        if record.integrity_hash and record.integrity_hash in self._integrity_hashes:
            return True, self._integrity_hashes[record.integrity_hash]

        return False, None

    def register(self, record: ExperienceRecord) -> None:
        """Register a new record into duplicate detection indices."""
        self._sample_ids.add(record.sample_id)
        if record.integrity_hash:
            self._integrity_hashes[record.integrity_hash] = record.sample_id

    def unregister(self, sample_id: str, integrity_hash: Optional[str] = None) -> None:
        """Remove a record from duplicate detection indices."""
        self._sample_ids.discard(sample_id)
        if integrity_hash:
            self._integrity_hashes.pop(integrity_hash, None)
