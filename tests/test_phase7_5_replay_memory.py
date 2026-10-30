"""
Phase 7.5 Replay Memory & Catastrophic Forgetting Test Suite.
Verifies:
1. Replay Memory Ingestion & Eligibility:
   - Only verified (USER_VERIFIED or SYSTEM_VERIFIED) & learning_eligible experiences enter memory.
   - UNVERIFIED, REJECTED, and quarantined experiences are strictly blocked.
   - Idempotent duplicate insertion.
2. Speaker Isolation:
   - Speaker A cannot receive Speaker B replay; cross-speaker insertion raises SpeakerIsolationError.
3. Temporal Causality:
   - Future experiences (timestamp > current cutoff) are strictly blocked from memory.
4. Quota Balancing & Anchor Retention:
   - Seed anchors are permanently retained and non-evictable.
   - Emergency classes are strictly protected from eviction.
   - 4:1 max general-to-emergency ratio enforced in balanced sampling.
5. Deterministic Sampling & Reproducibility:
   - Sampling with fixed seed produces identical replay batches.
6. Provenance & Audit:
   - Insertions, evictions, and sampling events are tracked in audit trail.
7. Forgetting Metric:
   - EMERGENCY_FORGETTING calculates performance delta accurately.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.store import InMemoryExperienceStore, SpeakerIsolationError
from src.continual.replay import (
    ReplayMemory,
    ReplayPolicy,
    ReplayMemoryItem,
    ReplayError,
    IneligibleReplayError,
    TemporalCausalityError,
    ReplayCapacityError,
    compute_emergency_forgetting,
)


@pytest.fixture
def mock_embeddings():
    """Deterministic 384-D mock embeddings."""
    rng = np.random.RandomState(42)
    embs = {}
    for i in range(50):
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        embs[f"sample_{i:03d}"] = vec
    return embs


def create_record(
    sample_id: str,
    speaker_id: str = "F01",
    intent: str = "SOS_HELP",
    verified: bool = True,
    eligible: bool = True,
    timestamp: str = "2026-01-01T10:00:00Z",
    quality_score: float = 0.9,
    quarantined: bool = False,
) -> ExperienceRecord:
    rec = ExperienceRecord(
        sample_id=sample_id,
        speaker_id=speaker_id,
        session_id="session_01",
        audio_reference=f"audio/{sample_id}.wav",
        embedding_reference=f"embeddings/{sample_id}.npy",
        predicted_intent=intent,
        prediction_confidence=0.88,
        model_version="whisper_adapter_v1.0.0",
        quality_score=quality_score,
        timestamp=timestamp,
        verification_status=VerificationStatus.USER_VERIFIED.value if verified else VerificationStatus.UNVERIFIED.value,
        confirmed_intent=intent if verified else None,
        learning_eligible=eligible,
        eligibility_reason=EligibilityReason.ELIGIBLE_USER_CONFIRMED.value if (verified and eligible) else EligibilityReason.INELIGIBLE_UNVERIFIED.value,
        data_quality_status="QUARANTINED" if quarantined else "CLEAN",
    )
    return rec


# ==============================================================================
# 1. ELIGIBILITY & INGESTION FIREWALL
# ==============================================================================

def test_unverified_record_rejected_from_replay():
    """Unverified record cannot enter ReplayMemory."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", verified=False, eligible=False)
    with pytest.raises(IneligibleReplayError):
        mem.add(rec)


def test_ineligible_flag_rejected_from_replay():
    """Record marked learning_eligible=False cannot enter ReplayMemory."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", verified=True, eligible=False)
    with pytest.raises(IneligibleReplayError):
        mem.add(rec)


def test_rejected_status_rejected_from_replay():
    """Record with verification_status=REJECTED cannot enter ReplayMemory."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", verified=False, eligible=False)
    rec.verification_status = VerificationStatus.REJECTED.value
    with pytest.raises(IneligibleReplayError):
        mem.add(rec)


def test_quarantined_record_rejected_from_replay():
    """Quarantined record cannot enter ReplayMemory."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", verified=True, eligible=True, quarantined=True)
    with pytest.raises(IneligibleReplayError):
        mem.add(rec)


def test_verified_eligible_admitted_to_replay():
    """Clean, verified, eligible record successfully added."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", verified=True, eligible=True)
    assert mem.add(rec) is True
    assert mem.size() == 1


def test_idempotent_duplicate_addition():
    """Adding an already present record returns False without duplication."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", verified=True, eligible=True)
    mem.add(rec)
    assert mem.add(rec) is False
    assert mem.size() == 1


# ==============================================================================
# 2. SPEAKER ISOLATION
# ==============================================================================

def test_speaker_mismatch_raises_isolation_error():
    """Attempting to add speaker B's record into speaker A's memory raises SpeakerIsolationError."""
    mem = ReplayMemory(speaker_id="F01")
    rec_m01 = create_record("s1", speaker_id="M01", verified=True, eligible=True)
    with pytest.raises(SpeakerIsolationError):
        mem.add(rec_m01)


def test_empty_speaker_id_raises_isolation_error():
    """Initializing ReplayMemory with empty speaker ID raises SpeakerIsolationError."""
    with pytest.raises(SpeakerIsolationError):
        ReplayMemory(speaker_id="")


# ==============================================================================
# 3. TEMPORAL CAUSALITY
# ==============================================================================

def test_future_experience_rejected_by_cutoff():
    """Record with timestamp after current cutoff raises TemporalCausalityError."""
    mem = ReplayMemory(speaker_id="F01")
    future_rec = create_record("s1", timestamp="2026-01-05T12:00:00Z")
    current_cutoff = "2026-01-02T00:00:00Z"
    with pytest.raises(TemporalCausalityError):
        mem.add(future_rec, current_cutoff_timestamp=current_cutoff)


def test_past_experience_admitted_by_cutoff():
    """Record with timestamp on or before cutoff is admitted."""
    mem = ReplayMemory(speaker_id="F01")
    past_rec = create_record("s1", timestamp="2026-01-01T10:00:00Z")
    current_cutoff = "2026-01-02T00:00:00Z"
    assert mem.add(past_rec, current_cutoff_timestamp=current_cutoff) is True


# ==============================================================================
# 4. SEED ANCHORING & QUOTA BALANCING
# ==============================================================================

def test_seed_anchors_non_evictable():
    """Seed anchor records are marked is_anchor=True and never evicted."""
    mem = ReplayMemory(speaker_id="F01", max_capacity=3)
    r1 = create_record("anchor_1", intent="SOS_HELP", timestamp="2026-01-01T01:00:00Z")
    r2 = create_record("anchor_2", intent="GENERAL_SPEECH", timestamp="2026-01-01T02:00:00Z")
    r3 = create_record("gen_1", intent="GENERAL_SPEECH", timestamp="2026-01-01T03:00:00Z")
    mem.add(r1, is_anchor=True)
    mem.add(r2, is_anchor=True)
    mem.add(r3, is_anchor=False)
    assert mem.size() == 3

    # Add a new emergency record when full -> non-anchor gen_1 should be evicted, anchors preserved
    r4 = create_record("emerg_2", intent="PAIN", timestamp="2026-01-01T04:00:00Z")
    mem.add(r4, is_anchor=False)
    assert mem.size() == 3
    stored_ids = [item.experience_id for item in mem.retrieve_all()]
    assert "anchor_1" in stored_ids
    assert "anchor_2" in stored_ids
    assert "emerg_2" in stored_ids
    assert "gen_1" not in stored_ids


def test_quota_balanced_sampling():
    """Balanced sampling enforces max 4:1 general-to-emergency ratio."""
    mem = ReplayMemory(speaker_id="F01", max_capacity=50, max_ratio=4.0)
    # 2 emergency records
    mem.add(create_record("e1", intent="SOS_HELP", timestamp="2026-01-01T01:00:00Z"))
    mem.add(create_record("e2", intent="FALL", timestamp="2026-01-01T02:00:00Z"))
    # 20 general speech records
    for i in range(20):
        mem.add(create_record(f"g_{i}", intent="GENERAL_SPEECH", timestamp=f"2026-01-01T{10+i:02d}:00:00Z"))

    sampled = mem.sample(policy=ReplayPolicy.BALANCED_REPLAY, seed=42)
    emerg_sampled = [s for s in sampled if s.class_label in EMERGENCY_CLASSES]
    gen_sampled = [s for s in sampled if s.class_label == GENERAL_SPEECH_CLASS]

    # All emergency items preserved
    assert len(emerg_sampled) == 2
    # General speech capped at 4 * 2 = 8
    assert len(gen_sampled) <= 8
    assert len(sampled) == 10


def test_no_replay_returns_empty():
    """NO_REPLAY policy returns empty list."""
    mem = ReplayMemory(speaker_id="F01")
    mem.add(create_record("e1", intent="SOS_HELP"))
    assert mem.sample(policy=ReplayPolicy.NO_REPLAY) == []


# ==============================================================================
# 5. DETERMINISTIC REPRODUCIBILITY & PROVENANCE
# ==============================================================================

def test_deterministic_sampling_seed():
    """Same seed yields identical sampled items."""
    mem = ReplayMemory(speaker_id="F01", max_ratio=2.0)
    mem.add(create_record("e1", intent="SOS_HELP", timestamp="2026-01-01T01:00:00Z"))
    for i in range(10):
        mem.add(create_record(f"g_{i}", intent="GENERAL_SPEECH", timestamp=f"2026-01-01T{10+i:02d}:00:00Z"))

    s1 = [x.experience_id for x in mem.sample(policy=ReplayPolicy.BALANCED_REPLAY, seed=123)]
    s2 = [x.experience_id for x in mem.sample(policy=ReplayPolicy.BALANCED_REPLAY, seed=123)]
    assert s1 == s2


def test_audit_log_provenance():
    """Memory maintains audit log of insertions and evictions."""
    mem = ReplayMemory(speaker_id="F01")
    rec = create_record("s1", intent="SOS_HELP")
    mem.add(rec, is_anchor=True)
    audit = mem.provenance()
    assert len(audit) == 1
    assert audit[0]["action"] == "INSERTED"
    assert audit[0]["experience_id"] == "s1"
    assert audit[0]["is_anchor"] is True


# ==============================================================================
# 6. FORGETTING METRIC COMPUTATION
# ==============================================================================

def test_emergency_forgetting_calculation():
    """EMERGENCY_FORGETTING calculates positive value for degradation, negative for gain."""
    # B0 recall = 0.20, Bt recall = 0.10 -> Forgetting = +0.10 (lost 10% emergency recall)
    assert pytest.approx(compute_emergency_forgetting(0.20, 0.10)) == 0.10
    # B0 recall = 0.20, Bt recall = 0.25 -> Forgetting = -0.05 (improved emergency recall)
    assert pytest.approx(compute_emergency_forgetting(0.20, 0.25)) == -0.05
    # B0 recall = 0.20, Bt recall = 0.20 -> Forgetting = 0.0 (exact retention)
    assert pytest.approx(compute_emergency_forgetting(0.20, 0.20)) == 0.0
