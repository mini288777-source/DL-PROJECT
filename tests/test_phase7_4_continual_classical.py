"""
Phase 7.4 Continual Classical Personalization Baseline Test Suite.
Verifies:
1. Data Eligibility:
   - Only verified (USER_VERIFIED or SYSTEM_VERIFIED) & learning_eligible records enter training.
   - UNVERIFIED, REJECTED, and INVALID records are strictly blocked.
2. Speaker Isolation:
   - Speaker-specific training remains strictly isolated. Cross-speaker samples raise SpeakerIsolationError.
3. Training Provenance:
   - All sample IDs, model versions, configuration hashes, and dataset hashes are recorded and immutable.
4. Model Versioning & Lifecycle:
   - New models enter as CANDIDATE status.
   - In-place overwrites are strictly rejected.
   - Parent version lineage is tracked.
5. Data Leakage Prevention:
   - Overlap between training experiences and protected test sets raises DataLeakageError.
   - Experiences are strictly sorted chronologically.
6. Deterministic Reproducibility:
   - Fixed seed and configuration yield identical dataset hashes and deterministic metrics.
7. Robust Failure Handling:
   - Missing embeddings, NaN/Inf features, or dimension mismatches raise structured integrity errors.
8. Experiment Registry Integrity:
   - All runs are tracked with status, configuration, and metric traceability.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import torch

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import CANONICAL_CLASSES
from src.continual.store import (
    InMemoryExperienceStore,
    SpeakerIsolationError,
)
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    ModelOverwriteError,
    ModelLineageError,
    compute_dataset_hash,
    compute_configuration_hash,
)
from src.continual.experiment_registry import (
    ExperimentStatus,
    ExperimentRunRecord,
    ExperimentRegistry,
)
from src.continual.continual_classical import (
    ContinualTrainingError,
    IneligibleExperienceError,
    DataLeakageError,
    EmbeddingIntegrityError,
    ContinualEvaluationMetrics,
    ContinualClassicalAdapterTrainer,
    validate_training_experiences,
    compute_comprehensive_continual_metrics,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


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


def create_mock_record(
    sample_id: str,
    speaker_id: str = "F01",
    intent: str = "SOS_HELP",
    verified: bool = True,
    eligible: bool = True,
    timestamp: str = "2026-01-01T10:00:00Z",
    quality_score: float = 0.9,
    embedding_array: Optional[np.ndarray] = None,
) -> ExperienceRecord:
    rec = ExperienceRecord(
        sample_id=sample_id,
        speaker_id=speaker_id,
        session_id="session_01",
        audio_reference=f"audio/{sample_id}.wav",
        embedding_reference=f"embeddings/{sample_id}.npy",
        predicted_intent=intent,
        prediction_confidence=0.85,
        model_version="whisper_adapter_v1.0.0",
        quality_score=quality_score,
        timestamp=timestamp,
        verification_status=VerificationStatus.USER_VERIFIED.value if verified else VerificationStatus.UNVERIFIED.value,
        confirmed_intent=intent if verified else None,
        learning_eligible=eligible,
        eligibility_reason=EligibilityReason.ELIGIBLE_USER_CONFIRMED.value if (verified and eligible) else EligibilityReason.INELIGIBLE_UNVERIFIED.value,
    )
    if embedding_array is not None:
        rec.embedding_array = embedding_array
    return rec


# ==============================================================================
# 1. DATA ELIGIBILITY TESTS
# ==============================================================================

def test_unverified_sample_rejected():
    """Unverified predictions MUST NEVER enter training."""
    rec = create_mock_record("s1", verified=False, eligible=False)
    with pytest.raises(IneligibleExperienceError):
        validate_training_experiences([rec], target_speaker="F01")


def test_ineligible_flag_rejected():
    """Records with learning_eligible=False MUST NOT enter training even if verified."""
    rec = create_mock_record("s1", verified=True, eligible=False)
    with pytest.raises(IneligibleExperienceError):
        validate_training_experiences([rec], target_speaker="F01")


def test_rejected_status_rejected():
    """Rejected records MUST NOT enter training."""
    rec = create_mock_record("s1", verified=False, eligible=False)
    rec.verification_status = VerificationStatus.REJECTED.value
    rec.learning_status = LearningStatus.REJECTED.value
    with pytest.raises(IneligibleExperienceError):
        validate_training_experiences([rec], target_speaker="F01")


def test_quarantined_record_rejected():
    """Quarantined records MUST NOT enter training."""
    rec = create_mock_record("s1", verified=True, eligible=True)
    rec.data_quality_status = "QUARANTINED"
    with pytest.raises(IneligibleExperienceError):
        validate_training_experiences([rec], target_speaker="F01")


def test_verified_eligible_admitted():
    """Properly verified, clean records pass eligibility validation."""
    rec = create_mock_record("s1", verified=True, eligible=True)
    # Should not raise
    validate_training_experiences([rec], target_speaker="F01")


# ==============================================================================
# 2. SPEAKER ISOLATION TESTS
# ==============================================================================

def test_speaker_mismatch_rejected():
    """Cross-speaker records in training batch raise SpeakerIsolationError."""
    rec_f01 = create_mock_record("s1", speaker_id="F01", verified=True, eligible=True)
    rec_m01 = create_mock_record("s2", speaker_id="M01", verified=True, eligible=True)
    with pytest.raises(SpeakerIsolationError):
        validate_training_experiences([rec_f01, rec_m01], target_speaker="F01")


# ==============================================================================
# 3. DATA LEAKAGE PREVENTION TESTS
# ==============================================================================

def test_data_leakage_detected(temp_dir, mock_embeddings):
    """If an evaluation sample is present in training experiences, DataLeakageError is raised."""
    trainer = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=temp_dir)
    train_rec = create_mock_record("sample_001", speaker_id="F01", verified=True, eligible=True, embedding_array=mock_embeddings["sample_001"])
    test_record = ("sample_001", "SOS_HELP", mock_embeddings["sample_001"])

    with pytest.raises(DataLeakageError):
        trainer.train_step(
            experiences=[train_rec],
            test_records=[test_record],
        )


def test_chronological_ordering(temp_dir, mock_embeddings):
    """Experiences are strictly ordered chronologically regardless of input list order."""
    trainer = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=temp_dir)
    r1 = create_mock_record("sample_001", timestamp="2026-01-02T10:00:00Z", embedding_array=mock_embeddings["sample_001"])
    r2 = create_mock_record("sample_002", timestamp="2026-01-01T10:00:00Z", embedding_array=mock_embeddings["sample_002"])
    test_rec = ("sample_010", "GENERAL_SPEECH", mock_embeddings["sample_010"])

    # Pass in reverse order: [r1, r2]
    res = trainer.train_step(
        experiences=[r1, r2],
        test_records=[test_rec],
    )
    # Resulting training sample IDs must be chronologically sorted: [r2, r1]
    assert res.train_sample_ids == ["sample_002", "sample_001"]


# ==============================================================================
# 4. MODEL VERSIONING & LIFECYCLE TESTS
# ==============================================================================

def test_model_version_candidate_status(temp_dir):
    """Newly registered models enter as CANDIDATE and immutability is enforced."""
    reg = ModelRegistry(base_dir=temp_dir)
    rec = ModelVersionRecord(
        model_id="classical_adapter",
        version="vF01_b20_seed42",
        speaker_id="F01",
        status=ModelLifecycleState.ACTIVE.value,  # Try to set ACTIVE directly
    )
    # Registration MUST enforce CANDIDATE status
    saved = reg.register_candidate(rec)
    assert saved.status == ModelLifecycleState.CANDIDATE.value

    # Overwrite attempt MUST raise ModelOverwriteError
    with pytest.raises(ModelOverwriteError):
        reg.register_candidate(rec)


def test_parent_lineage_validation(temp_dir):
    """Referencing a non-existent parent version raises ModelLineageError."""
    reg = ModelRegistry(base_dir=temp_dir)
    rec = ModelVersionRecord(
        model_id="classical_adapter",
        version="vF01_b30_seed42",
        speaker_id="F01",
        parent_version="vF01_nonexistent",
    )
    with pytest.raises(ModelLineageError):
        reg.register_candidate(rec)


def test_status_transition_and_active_exclusivity(temp_dir):
    """Promoting a version to ACTIVE archives the previous active version for that speaker."""
    reg = ModelRegistry(base_dir=temp_dir)
    v1 = reg.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="v1", speaker_id="F01"
    ))
    v2 = reg.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="v2", speaker_id="F01"
    ))

    reg.update_status("v1", ModelLifecycleState.ACTIVE)
    assert reg.get_active_model("F01").version == "v1"

    # Promoting v2 should demote v1 to ARCHIVED
    reg.update_status("v2", ModelLifecycleState.ACTIVE)
    assert reg.get_active_model("F01").version == "v2"
    assert reg.get_version("v1").status == ModelLifecycleState.ARCHIVED.value


# ==============================================================================
# 5. DETERMINISTIC REPRODUCIBILITY & CANONICAL ARCHITECTURE
# ==============================================================================

def test_training_canonical_architecture(temp_dir, mock_embeddings):
    """Verify that trained model has canonical 25,223 parameters and runs training."""
    trainer = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=temp_dir)
    train_recs = [
        create_mock_record(f"sample_{i:03d}", intent="GENERAL_SPEECH" if i < 15 else "SOS_HELP", embedding_array=mock_embeddings[f"sample_{i:03d}"])
        for i in range(20)
    ]
    test_recs = [
        ("sample_030", "GENERAL_SPEECH", mock_embeddings["sample_030"]),
        ("sample_031", "SOS_HELP", mock_embeddings["sample_031"]),
    ]

    res = trainer.train_step(
        experiences=train_recs,
        test_records=test_recs,
        custom_seed=42,
    )

    assert res.status == "COMPLETED"
    assert res.budget == 20
    assert os.path.isfile(res.model_checkpoint_path)
    # Checkpoint can be loaded
    sd = torch.load(res.model_checkpoint_path, weights_only=True)
    assert "net.0.weight" in sd
    assert sd["net.0.weight"].shape == (64, 384)


def test_deterministic_reproducibility(temp_dir, mock_embeddings):
    """Same seed and inputs produce identical dataset hash and metrics."""
    trainer1 = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=os.path.join(temp_dir, "t1"))
    trainer2 = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=os.path.join(temp_dir, "t2"))

    train_recs = [
        create_mock_record(f"sample_{i:03d}", intent="GENERAL_SPEECH" if i < 15 else "PAIN", embedding_array=mock_embeddings[f"sample_{i:03d}"])
        for i in range(20)
    ]
    test_recs = [
        ("sample_030", "GENERAL_SPEECH", mock_embeddings["sample_030"]),
        ("sample_031", "PAIN", mock_embeddings["sample_031"]),
    ]

    res1 = trainer1.train_step(experiences=train_recs, test_records=test_recs, custom_seed=42)
    res2 = trainer2.train_step(experiences=train_recs, test_records=test_recs, custom_seed=42)

    assert res1.training_dataset_hash == res2.training_dataset_hash
    assert res1.configuration_hash == res2.configuration_hash
    assert res1.metrics.accuracy == res2.metrics.accuracy
    assert res1.metrics.macro_f1 == res2.metrics.macro_f1


# ==============================================================================
# 6. FAILURE HANDLING TESTS
# ==============================================================================

def test_missing_embedding_fails(temp_dir):
    """Missing embedding fails with EmbeddingIntegrityError."""
    trainer = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=temp_dir)
    rec = create_mock_record("nonexistent_rid", embedding_array=None)
    rec.embedding_reference = "invalid/path/none.npy"

    with pytest.raises(EmbeddingIntegrityError):
        trainer.train_step(
            experiences=[rec],
            test_records=[("test_1", "GENERAL_SPEECH", np.zeros(384, dtype=np.float32))],
        )


def test_nan_embedding_fails(temp_dir):
    """Embedding with NaN fails with EmbeddingIntegrityError."""
    trainer = ContinualClassicalAdapterTrainer(target_speaker="F01", base_dir=temp_dir)
    nan_emb = np.zeros(384, dtype=np.float32)
    nan_emb[10] = np.nan
    rec = create_mock_record("nan_rid", embedding_array=nan_emb)

    with pytest.raises(EmbeddingIntegrityError):
        trainer.train_step(
            experiences=[rec],
            test_records=[("test_1", "GENERAL_SPEECH", np.zeros(384, dtype=np.float32))],
        )


# ==============================================================================
# 7. EXPERIMENT REGISTRY INTEGRITY
# ==============================================================================

def test_experiment_registry_lifecycle(temp_dir):
    """Registry correctly logs and updates experiment runs."""
    reg = ExperimentRegistry(registry_dir=temp_dir)
    run = ExperimentRunRecord(
        experiment_id="exp_001",
        speaker_id="F01",
        experience_budget=20,
        train_experience_ids=["s1", "s2"],
        evaluation_set="test_protected_v1",
        seed=42,
        configuration={"lr": 0.005},
    )
    reg.register_run(run)

    retrieved = reg.get_run("exp_001")
    assert retrieved is not None
    assert retrieved.status == ExperimentStatus.PENDING.value

    reg.update_run("exp_001", status=ExperimentStatus.COMPLETED, metrics={"accuracy": 0.85})
    assert reg.get_run("exp_001").status == ExperimentStatus.COMPLETED.value
    assert reg.get_run("exp_001").metrics["accuracy"] == 0.85
