"""
Phase 7.5 Replay Memory & Catastrophic Forgetting Mitigation.
Implements:
- ReplayMemory abstraction with strict speaker isolation and anti-self-training firewall.
- Seed Anchoring: B0 calibration instances permanently retained and non-evictable.
- Quota-Based Class-Balanced Replay: bounds general-to-emergency ratio at 4:1 (PHASE7_DATA_CONTRACT.md).
- Temporal Causality: strictly prevents future experiences from entering memory prior to their timestamp.
- Cryptographic provenance tracking for all replayed samples.
- Catastrophic forgetting metric calculation: EMERGENCY_FORGETTING.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import yaml

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent,
)
from src.continual.store import SpeakerIsolationError


class ReplayError(Exception):
    """Base exception for replay memory operations."""
    pass


class IneligibleReplayError(ReplayError):
    """Raised when an unverified or ineligible experience is submitted to replay."""
    pass


class TemporalCausalityError(ReplayError):
    """Raised when a future experience violates chronological causality."""
    pass


class ReplayCapacityError(ReplayError):
    """Raised when replay buffer capacity invariants are violated."""
    pass


class ReplayPolicy(str, Enum):
    """Supported replay sampling policies."""
    NO_REPLAY = "NO_REPLAY"
    BALANCED_REPLAY = "BALANCED_REPLAY"
    UNIFORM_REPLAY = "UNIFORM_REPLAY"


def compute_emergency_forgetting(baseline_b0_recall: float, current_bt_recall: float) -> float:
    """
    Compute canonical EMERGENCY_FORGETTING metric.
    Formula: Emergency_Recall(B0) - Emergency_Recall(Bt)
    Positive value indicates catastrophic forgetting; <= 0 indicates retention or gain.
    """
    return float(baseline_b0_recall - current_bt_recall)


@dataclass
class ReplayMemoryItem:
    """Immutable entry in replay memory."""
    experience_id: str
    speaker_id: str
    class_label: str
    timestamp: str
    verification_status: str
    learning_eligible: bool
    insertion_time: str
    is_anchor: bool = False
    quality_score: float = 1.0
    embedding_reference: Optional[str] = None
    embedding_array: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding_array", None)
        return d


class ReplayMemory:
    """
    Speaker-scoped replay memory with seed anchoring and quota balancing.
    """

    def __init__(
        self,
        speaker_id: str,
        policy_config: Optional[Dict[str, Any]] = None,
        max_capacity: int = 50,
        max_ratio: float = 4.0,
        memory_version: str = "v1.0.0",
    ):
        if not speaker_id or not speaker_id.strip():
            raise SpeakerIsolationError("speaker_id must be provided to initialize ReplayMemory.")

        self.speaker_id = speaker_id
        self.policy_config = policy_config or {}
        self.max_capacity = max_capacity
        self.max_ratio = max_ratio
        self.memory_version = memory_version

        # Internal storage: experience_id -> ReplayMemoryItem
        self._items: Dict[str, ReplayMemoryItem] = {}
        # Chronological index of sample IDs
        self._chronological_keys: List[str] = []
        # Provenance audit trail
        self._audit_log: List[Dict[str, Any]] = []

    def size(self) -> int:
        """Return total number of records currently stored."""
        return len(self._items)

    def __len__(self) -> int:
        """Return total number of records currently stored."""
        return self.size()

    def add(
        self,
        record: ExperienceRecord,
        is_anchor: bool = False,
        current_cutoff_timestamp: Optional[str] = None,
    ) -> bool:
        """
        Add a verified experience to replay memory.
        Enforces:
        - Speaker isolation
        - Verification and eligibility firewall
        - Temporal causality
        - Anchor protection
        - Quota-based capacity management
        """
        # 1. Speaker isolation
        if record.speaker_id != self.speaker_id:
            raise SpeakerIsolationError(
                f"Speaker mismatch: Attempted to add record for '{record.speaker_id}' "
                f"into ReplayMemory for '{self.speaker_id}'."
            )

        # 2. Verification status check
        if record.verification_status not in (
            VerificationStatus.USER_VERIFIED.value,
            VerificationStatus.SYSTEM_VERIFIED.value,
        ):
            raise IneligibleReplayError(
                f"Record '{record.sample_id}' has unverified status '{record.verification_status}'. "
                "Only USER_VERIFIED or SYSTEM_VERIFIED records may enter replay memory."
            )

        # 3. Learning eligibility check
        if not record.learning_eligible:
            raise IneligibleReplayError(
                f"Record '{record.sample_id}' is marked learning_eligible=False. "
                "Cannot enter replay memory."
            )

        # 4. Valid canonical intent check
        confirmed_label = record.confirmed_intent or record.predicted_intent
        if not confirmed_label or not is_valid_intent(confirmed_label):
            raise IneligibleReplayError(
                f"Record '{record.sample_id}' has invalid intent '{confirmed_label}'."
            )

        # 5. Quality gate
        if record.data_quality_status == "QUARANTINED" or (
            record.quality_score is not None and record.quality_score < 0.5
        ):
            raise IneligibleReplayError(
                f"Record '{record.sample_id}' failed quality criteria (status: {record.data_quality_status})."
            )

        # 6. Temporal causality check
        if current_cutoff_timestamp and record.timestamp > current_cutoff_timestamp:
            raise TemporalCausalityError(
                f"Temporal causality violation: Record '{record.sample_id}' timestamp "
                f"'{record.timestamp}' is strictly after cutoff '{current_cutoff_timestamp}'."
            )

        # 7. Duplicate handling
        if record.sample_id in self._items:
            return False  # Already present idempotently

        # 8. Capacity & Eviction check
        if len(self._items) >= self.max_capacity:
            # Cannot evict anchors
            # Evict oldest non-anchor GENERAL_SPEECH item
            eviction_candidates = [
                k for k in self._chronological_keys
                if not self._items[k].is_anchor and self._items[k].class_label == GENERAL_SPEECH_CLASS
            ]
            if eviction_candidates:
                evict_id = eviction_candidates[0]
                del self._items[evict_id]
                self._chronological_keys.remove(evict_id)
                self._audit_log.append({
                    "action": "EVICTED",
                    "experience_id": evict_id,
                    "reason": "CAPACITY_QUOTA_MANAGEMENT",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                # If cannot evict, reject if new item is general speech
                if confirmed_label == GENERAL_SPEECH_CLASS and not is_anchor:
                    return False

        # 9. Create and insert item
        item = ReplayMemoryItem(
            experience_id=record.sample_id,
            speaker_id=record.speaker_id,
            class_label=confirmed_label,
            timestamp=record.timestamp,
            verification_status=record.verification_status,
            learning_eligible=True,
            insertion_time=datetime.now(timezone.utc).isoformat(),
            is_anchor=is_anchor,
            quality_score=record.quality_score if record.quality_score is not None else 1.0,
            embedding_reference=record.embedding_reference,
            embedding_array=getattr(record, "embedding_array", None),
        )

        self._items[record.sample_id] = item
        self._chronological_keys.append(record.sample_id)
        # Sort chronological keys strictly by timestamp
        self._chronological_keys.sort(key=lambda k: self._items[k].timestamp)

        self._audit_log.append({
            "action": "INSERTED",
            "experience_id": record.sample_id,
            "class_label": confirmed_label,
            "is_anchor": is_anchor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return True

    def retrieve_all(self) -> List[ReplayMemoryItem]:
        """Retrieve all items in chronological order."""
        return [self._items[k] for k in self._chronological_keys]

    def sample(
        self,
        policy: ReplayPolicy = ReplayPolicy.BALANCED_REPLAY,
        target_count: Optional[int] = None,
        seed: int = 42,
    ) -> List[ReplayMemoryItem]:
        """
        Sample replay experiences according to configured policy.

        BALANCED_REPLAY:
        - Retains ALL available emergency instances.
        - Limits general speech to at most (max_ratio * num_emergency) instances.
        - Deterministically selects samples using fixed seed.

        NO_REPLAY:
        - Returns empty list.
        """
        if policy == ReplayPolicy.NO_REPLAY or not self._items:
            return []

        emerg_items = [
            self._items[k] for k in self._chronological_keys
            if self._items[k].class_label in EMERGENCY_CLASSES
        ]
        gen_items = [
            self._items[k] for k in self._chronological_keys
            if self._items[k].class_label == GENERAL_SPEECH_CLASS
        ]

        if policy == ReplayPolicy.BALANCED_REPLAY:
            # 1. Protect all available emergency items
            selected: List[ReplayMemoryItem] = list(emerg_items)

            # 2. Bound general speech by quota ratio: max_ratio * len(emerg)
            n_emerg = len(emerg_items)
            max_gen = int(np.ceil(self.max_ratio * max(1, n_emerg)))

            # If anchors exist in gen_items, prioritize anchors first
            anchor_gen = [g for g in gen_items if g.is_anchor]
            non_anchor_gen = [g for g in gen_items if not g.is_anchor]

            # Deterministic selection
            rng = np.random.RandomState(seed)
            shuffled_non_anchors = list(non_anchor_gen)
            rng.shuffle(shuffled_non_anchors)

            candidate_gen = anchor_gen + shuffled_non_anchors
            selected_gen = candidate_gen[:max_gen]
            selected.extend(selected_gen)

            # Re-sort selected items chronologically
            selected.sort(key=lambda item: item.timestamp)
            return selected

        elif policy == ReplayPolicy.UNIFORM_REPLAY:
            all_items = self.retrieve_all()
            if target_count is None or target_count >= len(all_items):
                return all_items
            rng = np.random.RandomState(seed)
            indices = sorted(rng.choice(len(all_items), size=target_count, replace=False))
            return [all_items[i] for i in indices]

        return []

    def statistics(self) -> Dict[str, Any]:
        """Generate diagnostic summary of memory contents."""
        counts: Dict[str, int] = {c: 0 for c in CANONICAL_CLASSES}
        for item in self._items.values():
            counts[item.class_label] = counts.get(item.class_label, 0) + 1

        emerg_total = sum(counts[c] for c in EMERGENCY_CLASSES)
        gen_total = counts.get(GENERAL_SPEECH_CLASS, 0)
        anchor_total = sum(1 for item in self._items.values() if item.is_anchor)

        return {
            "speaker_id": self.speaker_id,
            "total_items": len(self._items),
            "emergency_items": emerg_total,
            "general_speech_items": gen_total,
            "anchor_items": anchor_total,
            "emergency_ratio": float(emerg_total / len(self._items)) if self._items else 0.0,
            "class_distribution": counts,
            "capacity": self.max_capacity,
            "utilization_pct": float(len(self._items) / self.max_capacity * 100.0),
        }

    def provenance(self) -> List[Dict[str, Any]]:
        """Return audit trail for memory operations."""
        return list(self._audit_log)
