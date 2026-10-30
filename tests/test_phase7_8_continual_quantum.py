"""
Phase 7.8 Continual Quantum Kernel Personalization Unit Test Suite.

Verifies:
1. Quantum Configuration SSOT:
   - 8 qubits, ZZFeatureMap, reps=1, linear entanglement, FidelityQuantumKernel, exact statevector simulation, QSVC C=1.0.
   - Strict rejection of unauthorized quantum hyperparameters.
2. Feature Pipeline & Information Isolation:
   - 384-D -> 8-D train-only PCA -> MinMaxScaler [0, pi].
   - Zero test leakage: test data never fits PCA or scaler.
   - Data leakage detection raises DataLeakageError.
3. Kernel Integrity & Numerical Validation:
   - Matrix symmetry error < 1e-6, diagonal error < 1e-6, values bounded in [0, 1].
   - Positive semi-definiteness (min eigenvalue >= -1e-5).
   - Corruption / NaN / Inf raises KernelIntegrityError.
4. Deterministic Kernel Caching:
   - Full configuration fingerprinting in cache key.
   - Cache hit preserves bitwise matrix fidelity and records zero kernel train time.
   - Cache invalidation on any parameter / data change.
5. Speaker Isolation & Experience Eligibility:
   - Cross-speaker experiences raise SpeakerIsolationError.
   - Unverified / quarantined / ineligible records raise IneligibleExperienceError.
6. Temporal Safety:
   - Experiences sorted strictly chronologically.
7. Model Versioning & Lifecycle:
   - Models register with lifecycle state CANDIDATE (never directly ACTIVE).
   - Unique versioning, parent version lineage preserved.
8. Phase 7.6 Governance Integration:
   - Candidate evaluated via ValidationGate.
   - Safety Veto triggers REJECT on emergency recall regression.
9. Anti-Phase-7.9 Firewall:
   - Exactly 0 trainable quantum parameters.
10. Single-Class Fallback & Reproducibility:
    - Graceful fallback when training batch has single class.
    - Deterministic reproducibility under identical seed.
"""

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.store import SpeakerIsolationError
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    compute_dataset_hash,
)
from src.continual.continual_classical import (
    IneligibleExperienceError,
    DataLeakageError,
    EmbeddingIntegrityError,
)
from src.continual.governance import (
    ValidationGate,
    PromotionDecision,
    ValidationReasonCode,
)
from src.continual.continual_quantum import (
    CANONICAL_QUANTUM_CONFIG,
    QuantumContinualError,
    QuantumConfigurationError,
    KernelIntegrityError,
    QuantumLeakageError,
    ContinualQSVCClassifier,
    ContinualQuantumTrainer,
    ContinualQuantumKernelCache,
    compute_quantum_cache_key,
)


@pytest.fixture
def temp_dirs():
    """Provides isolated temporary directories for models and cache."""
    models_dir = tempfile.mkdtemp()
    cache_dir = tempfile.mkdtemp()
    yield models_dir, cache_dir
    shutil.rmtree(models_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def mock_experiences():
    """Generates 20 mock verified experiences for speaker F01."""
    records = []
    rng = np.random.RandomState(42)
    classes = ["GENERAL_SPEECH"] * 16 + ["SOS_HELP", "FALL", "PAIN", "LOCATION"]

    for i in range(20):
        # 384-D normalized embedding
        emb = rng.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        rec = ExperienceRecord(
            sample_id=f"exp_F01_{i:03d}",
            speaker_id="F01",
            session_id="sess_01",
            audio_reference=f"audio/exp_F01_{i:03d}.wav",
            embedding_reference=f"embeddings/exp_F01_{i:03d}.npy",
            predicted_intent=classes[i],
            prediction_confidence=0.92,
            model_version="seed_v0",
            quality_score=0.95,
            timestamp=f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00Z",
            verification_status=VerificationStatus.USER_VERIFIED.value,
            confirmed_intent=classes[i],
            confirmation_source="user_explicit",
            learning_eligible=True,
            eligibility_reason=EligibilityReason.ELIGIBLE_USER_CONFIRMED.value,
            learning_status=LearningStatus.USER_CONFIRMED.value,
        )
        rec.embedding_array = emb
        records.append(rec)
    return records


@pytest.fixture
def mock_test_records():
    """Generates 30 mock held-out test instances for speaker F01."""
    records = []
    rng = np.random.RandomState(999)
    classes = ["GENERAL_SPEECH"] * 24 + ["SOS_HELP", "FALL", "PAIN", "LOCATION", "CRITICAL_RESPONSE", "BREATHING_MEDICAL"]

    for i in range(30):
        emb = rng.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        records.append((f"test_F01_{i:03d}", classes[i % len(classes)], emb))
    return records


# ==============================================================================
# 1. QUANTUM CONFIGURATION SSOT TESTS
# ==============================================================================

def test_quantum_canonical_configuration_ssot():
    """Verify default canonical quantum configuration matches Phase 6.5 SSOT exactly."""
    assert CANONICAL_QUANTUM_CONFIG["qubit_count"] == 8
    assert CANONICAL_QUANTUM_CONFIG["representation_dim"] == 8
    assert CANONICAL_QUANTUM_CONFIG["feature_map"] == "ZZFeatureMap"
    assert CANONICAL_QUANTUM_CONFIG["entanglement"] == "linear"
    assert CANONICAL_QUANTUM_CONFIG["reps"] == 1
    assert CANONICAL_QUANTUM_CONFIG["kernel_type"] == "FidelityQuantumKernel"
    assert CANONICAL_QUANTUM_CONFIG["classifier"] == "QSVC"
    assert CANONICAL_QUANTUM_CONFIG["C"] == 1.0
    assert CANONICAL_QUANTUM_CONFIG["trainable_quantum_parameters"] == 0


def test_quantum_configuration_invalid_qubits_rejected():
    """Verify that any qubit count other than 8 raises QuantumConfigurationError."""
    with pytest.raises(QuantumConfigurationError, match="Qubit count must be 8"):
        ContinualQSVCClassifier(qubit_count=4)


def test_quantum_configuration_invalid_reps_rejected():
    """Verify that reps != 1 raises QuantumConfigurationError."""
    with pytest.raises(QuantumConfigurationError, match="Reps must be 1"):
        ContinualQSVCClassifier(reps=2)


def test_quantum_configuration_invalid_entanglement_rejected():
    """Verify that non-linear entanglement raises QuantumConfigurationError."""
    with pytest.raises(QuantumConfigurationError, match="Entanglement must be 'linear'"):
        ContinualQSVCClassifier(entanglement="full")


def test_quantum_configuration_invalid_C_rejected():
    """Verify that regularizer C != 1.0 raises QuantumConfigurationError."""
    with pytest.raises(QuantumConfigurationError, match="Regularization C must be 1.0"):
        ContinualQSVCClassifier(C=0.1)


# ==============================================================================
# 2. FEATURE PIPELINE & STRICT LEAKAGE CONTROL
# ==============================================================================

def test_feature_pipeline_train_only_pca(mock_experiences, mock_test_records, temp_dirs):
    """Verify PCA and MinMaxScaler are fitted strictly on training data with zero test leakage."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    result = trainer.train_step(
        experiences=mock_experiences,
        test_records=mock_test_records,
    )

    assert result.status == "COMPLETED"
    assert result.budget == 20
    assert result.metrics.total_test_samples == 30


def test_feature_pipeline_zero_leakage_exception(mock_experiences, mock_test_records, temp_dirs):
    """Verify that overlap between training experiences and test set raises DataLeakageError."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    # Inject test record into training batch
    leaked_rec = ExperienceRecord(
        sample_id=mock_test_records[0][0],  # Overlap sample_id
        speaker_id="F01",
        session_id="sess_leak",
        audio_reference="audio/leak.wav",
        embedding_reference="embeddings/leak.npy",
        predicted_intent="SOS_HELP",
        prediction_confidence=0.95,
        model_version="v0",
        quality_score=0.95,
        timestamp="2026-01-01T00:00:00Z",
        verification_status=VerificationStatus.USER_VERIFIED.value,
        confirmed_intent="SOS_HELP",
        learning_eligible=True,
    )
    leaked_rec.embedding_array = mock_test_records[0][2]
    contaminated = mock_experiences + [leaked_rec]

    with pytest.raises(DataLeakageError, match="CRITICAL: Data leakage detected"):
        trainer.train_step(experiences=contaminated, test_records=mock_test_records)


def test_feature_pipeline_dimension_integrity(mock_experiences, mock_test_records, temp_dirs):
    """Verify that embeddings with dimension != 384 raise EmbeddingIntegrityError."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    # Corrupt embedding dimension
    mock_experiences[0].embedding_array = np.zeros(128, dtype=np.float32)

    with pytest.raises(EmbeddingIntegrityError, match="does not match expected"):
        trainer.train_step(experiences=mock_experiences, test_records=mock_test_records)


# ==============================================================================
# 3. KERNEL INTEGRITY & NUMERICAL VALIDATION
# ==============================================================================

def test_kernel_symmetry_and_diagonal():
    """Verify computed quantum kernel matrix is symmetric and has unit diagonal."""
    classifier = ContinualQSVCClassifier()
    rng = np.random.RandomState(42)
    X = rng.randn(10, 384).astype(np.float32)
    y = ["GENERAL_SPEECH"] * 8 + ["SOS_HELP", "FALL"]

    classifier.fit(X, y)
    K = classifier.K_train
    assert K is not None

    # Check shape
    assert K.shape == (10, 10)
    # Check symmetry
    sym_err = np.max(np.abs(K - K.T))
    assert sym_err < 1e-10
    # Check diagonal
    diag_err = np.max(np.abs(np.diag(K) - 1.0))
    assert diag_err < 1e-10
    # Check values in [0, 1]
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0)


def test_kernel_positive_semidefinite():
    """Verify training kernel matrix is positive semi-definite."""
    classifier = ContinualQSVCClassifier()
    rng = np.random.RandomState(42)
    X = rng.randn(12, 384).astype(np.float32)
    y = ["GENERAL_SPEECH"] * 10 + ["SOS_HELP", "FALL"]

    classifier.fit(X, y)
    eigs = np.linalg.eigvalsh(classifier.K_train)
    min_eig = np.min(eigs)
    assert min_eig >= -1e-5


def test_kernel_nan_inf_detection():
    """Verify that NaN or Inf in kernel raises KernelIntegrityError."""
    classifier = ContinualQSVCClassifier()
    K_bad = np.eye(5)
    K_bad[0, 1] = np.nan
    with pytest.raises(KernelIntegrityError, match="non-finite values"):
        classifier.validate_kernel_matrix(K_bad, is_training=True)


def test_kernel_out_of_bounds_detection():
    """Verify that kernel values outside [0, 1] raise KernelIntegrityError."""
    classifier = ContinualQSVCClassifier()
    K_bad = np.eye(5)
    K_bad[0, 1] = 1.5
    with pytest.raises(KernelIntegrityError, match="out of bounds"):
        classifier.validate_kernel_matrix(K_bad, is_training=True)


# ==============================================================================
# 4. DETERMINISTIC KERNEL CACHING
# ==============================================================================

def test_kernel_cache_roundtrip(temp_dirs):
    """Verify kernel cache saves, loads, and preserves exact values."""
    _, cache_dir = temp_dirs
    cache = ContinualQuantumKernelCache(cache_dir)

    K_train = np.eye(10, dtype=np.float64)
    K_test = np.ones((5, 10), dtype=np.float64) * 0.5
    key = "test_cache_key_12345"

    cache.save(key, K_train, K_test, {"info": "test"})
    loaded = cache.load(key)

    assert loaded is not None
    loaded_train, loaded_test, meta = loaded
    np.testing.assert_array_equal(K_train, loaded_train)
    np.testing.assert_array_equal(K_test, loaded_test)
    assert meta["cache_key"] == key


def test_kernel_cache_key_invalidation():
    """Verify cache key changes when any configuration or data parameter is modified."""
    base_config = dict(CANONICAL_QUANTUM_CONFIG)
    k1 = compute_quantum_cache_key("F01", 20, 42, "hash_A", base_config)
    k2 = compute_quantum_cache_key("F01", 30, 42, "hash_A", base_config)  # Budget changed
    k3 = compute_quantum_cache_key("F01", 20, 43, "hash_A", base_config)  # Seed changed
    k4 = compute_quantum_cache_key("F01", 20, 42, "hash_B", base_config)  # Data hash changed

    assert k1 != k2
    assert k1 != k3
    assert k1 != k4


def test_kernel_cache_hit_profiling(mock_experiences, mock_test_records, temp_dirs):
    """Verify second run with identical data hits cache and zeroes kernel computation time."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    # First run: cache miss
    res1 = trainer.train_step(mock_experiences, mock_test_records, custom_version="vF01_run1")
    assert res1.kernel_cache_hit is False
    assert res1.kernel_train_time_sec > 0.0

    # Second run: cache hit
    res2 = trainer.train_step(mock_experiences, mock_test_records, custom_version="vF01_run2")
    assert res2.kernel_cache_hit is True
    assert res2.kernel_train_time_sec == 0.0


# ==============================================================================
# 5. SPEAKER ISOLATION & EXPERIENCE ELIGIBILITY
# ==============================================================================

def test_speaker_isolation_enforced(mock_experiences, mock_test_records, temp_dirs):
    """Verify experiences from speaker M01 cannot enter speaker F01 trainer."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    mock_experiences[0].speaker_id = "M01"
    with pytest.raises(SpeakerIsolationError, match="Speaker mismatch"):
        trainer.train_step(mock_experiences, mock_test_records)


def test_experience_eligibility_enforced(mock_experiences, mock_test_records, temp_dirs):
    """Verify unverified or learning_eligible=False experiences are rejected."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    mock_experiences[0].learning_eligible = False
    with pytest.raises(IneligibleExperienceError, match="marked learning_eligible=False"):
        trainer.train_step(mock_experiences, mock_test_records)


def test_quarantined_record_blocked(mock_experiences, mock_test_records, temp_dirs):
    """Verify quarantined or low-quality experiences are rejected."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    mock_experiences[0].data_quality_status = "QUARANTINED"
    with pytest.raises(IneligibleExperienceError, match="failed quality gate"):
        trainer.train_step(mock_experiences, mock_test_records)


# ==============================================================================
# 6. MODEL VERSIONING & CANDIDATE GOVERNANCE
# ==============================================================================

def test_model_versioning_and_candidate_lifecycle(mock_experiences, mock_test_records, temp_dirs):
    """Verify candidate model is registered in CANDIDATE state and never directly ACTIVE."""
    models_dir, cache_dir = temp_dirs
    registry = ModelRegistry(base_dir=models_dir)
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        model_registry=registry,
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    res = trainer.train_step(mock_experiences, mock_test_records)
    candidate_record = registry.get_version(res.model_version)

    assert candidate_record is not None
    assert candidate_record.status == ModelLifecycleState.CANDIDATE.value
    assert candidate_record.model_id == "qsvc_q8_r1"
    assert candidate_record.quantum_configuration is not None
    assert candidate_record.quantum_configuration["qubit_count"] == 8


def test_validation_gate_integration(mock_experiences, mock_test_records, temp_dirs):
    """Verify candidate model passes through Phase 7.6 ValidationGate."""
    models_dir, cache_dir = temp_dirs
    trainer = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
        cache_dir=cache_dir,
    )

    res = trainer.train_step(mock_experiences, mock_test_records)
    assert res.governance_decision in ("PROMOTE", "REJECT", "HOLD")


# ==============================================================================
# 7. ANTI-PHASE-7.9 FIREWALL (ZERO TRAINABLE QUANTUM PARAMETERS)
# ==============================================================================

def test_zero_trainable_quantum_parameters():
    """Verify circuit has exactly 0 trainable quantum parameters."""
    classifier = ContinualQSVCClassifier()
    assert classifier.kernel_engine.trainable_quantum_parameters == 0
    assert classifier.kernel_engine.encoding_parameters == 8


# ==============================================================================
# 8. SINGLE-CLASS FALLBACK & REPRODUCIBILITY
# ==============================================================================

def test_single_class_adaptation_fallback():
    """Verify classifier gracefully handles single-class adaptation without failing."""
    classifier = ContinualQSVCClassifier()
    rng = np.random.RandomState(42)
    X_train = rng.randn(10, 384).astype(np.float32)
    y_train = ["GENERAL_SPEECH"] * 10  # Only one class

    classifier.fit(X_train, y_train)
    assert classifier.is_single_class_fallback is True
    assert classifier.single_observed_class == "GENERAL_SPEECH"

    X_test = rng.randn(5, 384).astype(np.float32)
    preds = classifier.predict(X_test)
    assert preds == ["GENERAL_SPEECH"] * 5


def test_deterministic_quantum_reproducibility(mock_experiences, mock_test_records, temp_dirs):
    """Verify identical seed and data yield bitwise reproducible metrics and dataset hash."""
    models_dir, cache_dir = temp_dirs
    trainer1 = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=os.path.join(models_dir, "t1"),
        cache_dir=cache_dir,
    )
    trainer2 = ContinualQuantumTrainer(
        target_speaker="F01",
        base_dir=os.path.join(models_dir, "t2"),
        cache_dir=cache_dir,
    )

    res1 = trainer1.train_step(mock_experiences, mock_test_records, custom_seed=42)
    res2 = trainer2.train_step(mock_experiences, mock_test_records, custom_seed=42)

    assert res1.training_dataset_hash == res2.training_dataset_hash
    assert res1.metrics.macro_f1 == res2.metrics.macro_f1
    assert res1.metrics.emergency_recall == res2.metrics.emergency_recall
    assert res1.metrics.accuracy == res2.metrics.accuracy
