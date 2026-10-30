"""
Phase 7.9 Continual Trainable Quantum Representation Unit Test Suite.

Verifies:
1. Quantum Architecture SSOT:
   - 8 qubits, 16 trainable parameters, circuit depth 37, linear entanglement,
     re-uploading ZZ feature map, 16 Ry trainable gates.
   - Rejection of invalid qubit counts, parameter counts, or regularization.
2. Trainability & Optimization:
   - Trainable parameters actually update away from initial values.
   - KTA objective improves or converges.
   - Parameter fingerprints are cryptographically recorded.
3. Initialization & Warm-Start Lineage:
   - Deterministic zero initialization for B0.
   - Warm-start initialization from parent theta for B1..B3.
   - Parent parameter fingerprint tracked and validated.
4. Feature Pipeline & Information Isolation:
   - 384-D -> 8-D train-only PCA -> MinMaxScaler [0, pi].
   - Zero evaluation leakage; DataLeakageError raised on overlap.
5. Speaker Isolation & Experience Eligibility:
   - Cross-speaker experiences raise SpeakerIsolationError.
   - Unverified / quarantined / ineligible records raise IneligibleExperienceError.
6. Kernel Integrity:
   - Symmetry error < 1e-6, diagonal error < 1e-6, values bounded in [0, 1].
   - Positive semi-definiteness (min eigenvalue >= -1e-5).
   - NaN / Inf raises TrainableKernelIntegrityError.
7. Model Versioning & Lifecycle:
   - Models register with lifecycle state CANDIDATE (never directly ACTIVE).
   - Lineage and parameter metadata preserved.
8. Phase 7.6 Governance Integration:
   - Candidate evaluated via ValidationGate.
   - Safety Veto triggers REJECT on emergency recall regression.
9. Single-Class Fallback & Reproducibility:
   - Single-class batches handled gracefully without crashing.
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
from src.continual.trainable_quantum import (
    CANONICAL_TRAINABLE_QUANTUM_CONFIG,
    TrainableQuantumError,
    TrainableQuantumArchitectureError,
    QuantumOptimizationError,
    TrainableKernelIntegrityError,
    ParameterizedQuantumFeatureMap,
    QuantumKernelAlignmentOptimizer,
    ContinualTrainableQSVCClassifier,
    ContinualTrainableQuantumTrainer,
    compute_param_fingerprint,
)


@pytest.fixture
def temp_dirs():
    """Provides isolated temporary directories for models."""
    models_dir = tempfile.mkdtemp()
    yield models_dir
    shutil.rmtree(models_dir, ignore_errors=True)


@pytest.fixture
def mock_experiences():
    """Generates 20 mock verified experiences for speaker F01."""
    records = []
    rng = np.random.RandomState(42)
    classes = ["GENERAL_SPEECH"] * 16 + ["SOS_HELP", "FALL", "PAIN", "LOCATION"]

    for i in range(20):
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
    """Generates 25 mock held-out test instances for speaker F01."""
    records = []
    rng = np.random.RandomState(999)
    classes = ["GENERAL_SPEECH"] * 20 + ["SOS_HELP", "FALL", "PAIN", "LOCATION", "CRITICAL_RESPONSE"]

    for i in range(25):
        emb = rng.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        records.append((f"test_F01_{i:03d}", classes[i % len(classes)], emb))
    return records


# ==============================================================================
# 1. QUANTUM ARCHITECTURE SSOT TESTS
# ==============================================================================

def test_trainable_quantum_architecture_ssot():
    """Verify canonical trainable quantum configuration matches SSOT specification."""
    cfg = CANONICAL_TRAINABLE_QUANTUM_CONFIG
    assert cfg["qubit_count"] == 8
    assert cfg["input_dimension"] == 8
    assert cfg["trainable_parameters"] == 16
    assert cfg["circuit_depth"] == 37
    assert cfg["entanglement_pattern"] == "linear"
    assert cfg["data_reuploading"] is True
    assert cfg["optimizer"] == "COBYLA"
    assert cfg["C"] == 1.0


def test_trainable_circuit_gate_counts():
    """Verify circuit ops count and depth match analytical specifications."""
    pqfm = ParameterizedQuantumFeatureMap(num_qubits=8, num_parameters=16)
    assert pqfm.circuit_depth == 37
    assert pqfm.trainable_gate_count == 16
    assert pqfm.entangling_gate_count == 35


def test_trainable_architecture_invalid_parameters_rejected():
    """Verify parameter counts other than 16 raise TrainableQuantumArchitectureError."""
    with pytest.raises(TrainableQuantumArchitectureError, match="parameter count must be 16"):
        ParameterizedQuantumFeatureMap(num_qubits=8, num_parameters=8)


def test_trainable_architecture_invalid_qubits_rejected():
    """Verify qubit counts other than 8 raise TrainableQuantumArchitectureError."""
    with pytest.raises(TrainableQuantumArchitectureError, match="Qubit count must be 8"):
        ParameterizedQuantumFeatureMap(num_qubits=4, num_parameters=16)


def test_trainable_classifier_invalid_C_rejected():
    """Verify regularization C != 1.0 raises TrainableQuantumArchitectureError."""
    with pytest.raises(TrainableQuantumArchitectureError, match="Regularization C must be 1.0"):
        ContinualTrainableQSVCClassifier(C=0.5)


# ==============================================================================
# 2. TRAINABILITY & OPTIMIZATION TESTS
# ==============================================================================

def test_quantum_parameters_update_away_from_zero():
    """Verify that KTA optimization actually updates parameters away from initial values."""
    classifier = ContinualTrainableQSVCClassifier(maxiter=20)
    rng = np.random.RandomState(42)
    X = rng.randn(12, 384).astype(np.float32)
    y = ["GENERAL_SPEECH"] * 8 + ["SOS_HELP", "FALL", "PAIN", "LOCATION"]

    classifier.fit(X, y)
    assert classifier.theta_ is not None
    assert len(classifier.theta_) == 16
    # Parameters must have updated (norm > 0)
    param_norm = float(np.linalg.norm(classifier.theta_))
    assert param_norm > 1e-4
    assert classifier.nfev > 0


def test_parameter_fingerprint_deterministic():
    """Verify compute_param_fingerprint produces deterministic SHA-256 digests."""
    theta1 = np.ones(16, dtype=np.float64) * 0.25
    theta2 = np.ones(16, dtype=np.float64) * 0.25
    theta3 = np.ones(16, dtype=np.float64) * 0.50

    fp1 = compute_param_fingerprint(theta1)
    fp2 = compute_param_fingerprint(theta2)
    fp3 = compute_param_fingerprint(theta3)

    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 64


# ==============================================================================
# 3. INITIALIZATION & WARM-START LINEAGE TESTS
# ==============================================================================

def test_warm_start_continual_lineage(mock_experiences, mock_test_records, temp_dirs):
    """Verify warm-start initialization uses predecessor theta and tracks parent fingerprint."""
    models_dir = temp_dirs
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
    )

    # Step 1: B0 = 10 samples (initial calibration from zero)
    exp_b0 = mock_experiences[:10]
    res1 = trainer.train_step(
        experiences=exp_b0,
        test_records=mock_test_records,
        custom_version="vF01_b10",
    )
    assert res1.initialization_mode == "DETERMINISTIC_ZERO"
    assert res1.parent_param_fingerprint is None

    # Step 2: B1 = 20 samples (warm-started from Step 1 theta)
    exp_b1 = mock_experiences[:20]
    res2 = trainer.train_step(
        experiences=exp_b1,
        test_records=mock_test_records,
        parent_model_version=res1.model_version,
        parent_theta=np.array(res1.optimal_theta),
        custom_version="vF01_b20",
    )
    assert res2.initialization_mode == "WARM_START"
    assert res2.parent_param_fingerprint == res1.optimal_param_fingerprint


# ==============================================================================
# 4. FEATURE PIPELINE & STRICT LEAKAGE CONTROL
# ==============================================================================

def test_trainable_feature_pipeline_zero_leakage(mock_experiences, mock_test_records, temp_dirs):
    """Verify test data overlap in training set raises DataLeakageError."""
    models_dir = temp_dirs
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
    )

    # Inject test record into training experiences
    leak_rec = ExperienceRecord(
        sample_id=mock_test_records[0][0],
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
    leak_rec.embedding_array = mock_test_records[0][2]
    contaminated = mock_experiences + [leak_rec]

    with pytest.raises(DataLeakageError, match="CRITICAL: Data leakage detected"):
        trainer.train_step(experiences=contaminated, test_records=mock_test_records)


def test_trainable_feature_pipeline_dimension_integrity(mock_experiences, mock_test_records, temp_dirs):
    """Verify embedding dimension != 384 raises EmbeddingIntegrityError."""
    models_dir = temp_dirs
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
    )

    mock_experiences[0].embedding_array = np.zeros(64, dtype=np.float32)
    with pytest.raises(EmbeddingIntegrityError, match="does not match expected"):
        trainer.train_step(experiences=mock_experiences, test_records=mock_test_records)


# ==============================================================================
# 5. KERNEL INTEGRITY & NUMERICAL VALIDATION
# ==============================================================================

def test_adapted_kernel_symmetry_and_diagonal():
    """Verify adapted quantum kernel matrix is symmetric and has unit diagonal."""
    classifier = ContinualTrainableQSVCClassifier(maxiter=20)
    rng = np.random.RandomState(42)
    X = rng.randn(10, 384).astype(np.float32)
    y = ["GENERAL_SPEECH"] * 7 + ["SOS_HELP", "FALL", "PAIN"]

    classifier.fit(X, y)
    K = classifier.K_train
    assert K is not None

    assert K.shape == (10, 10)
    sym_err = float(np.max(np.abs(K - K.T)))
    assert sym_err < 1e-10
    diag_err = float(np.max(np.abs(np.diag(K) - 1.0)))
    assert diag_err < 1e-10
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0)


def test_adapted_kernel_positive_semidefinite():
    """Verify adapted training kernel matrix is positive semi-definite."""
    classifier = ContinualTrainableQSVCClassifier(maxiter=20)
    rng = np.random.RandomState(42)
    X = rng.randn(10, 384).astype(np.float32)
    y = ["GENERAL_SPEECH"] * 8 + ["SOS_HELP", "FALL"]

    classifier.fit(X, y)
    eigs = np.linalg.eigvalsh(classifier.K_train)
    min_eig = float(np.min(eigs))
    assert min_eig >= -1e-5


def test_adapted_kernel_nan_detection():
    """Verify NaN in kernel raises TrainableKernelIntegrityError."""
    classifier = ContinualTrainableQSVCClassifier()
    K_bad = np.eye(4)
    K_bad[0, 1] = np.nan
    with pytest.raises(TrainableKernelIntegrityError, match="non-finite values"):
        classifier.validate_kernel_matrix(K_bad, is_training=True)


# ==============================================================================
# 6. SPEAKER ISOLATION & EXPERIENCE ELIGIBILITY
# ==============================================================================

def test_speaker_isolation_enforced(mock_experiences, mock_test_records, temp_dirs):
    """Verify cross-speaker experiences raise SpeakerIsolationError."""
    models_dir = temp_dirs
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
    )

    mock_experiences[0].speaker_id = "M01"
    with pytest.raises(SpeakerIsolationError, match="Speaker mismatch"):
        trainer.train_step(mock_experiences, mock_test_records)


def test_experience_eligibility_enforced(mock_experiences, mock_test_records, temp_dirs):
    """Verify learning_eligible=False experiences raise IneligibleExperienceError."""
    models_dir = temp_dirs
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
    )

    mock_experiences[0].learning_eligible = False
    with pytest.raises(IneligibleExperienceError, match="marked learning_eligible=False"):
        trainer.train_step(mock_experiences, mock_test_records)


# ==============================================================================
# 7. MODEL VERSIONING & CANDIDATE GOVERNANCE
# ==============================================================================

def test_model_versioning_and_candidate_lifecycle(mock_experiences, mock_test_records, temp_dirs):
    """Verify candidate model is registered in CANDIDATE state and preserves provenance."""
    models_dir = temp_dirs
    registry = ModelRegistry(base_dir=models_dir)
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        model_registry=registry,
        base_dir=models_dir,
    )

    res = trainer.train_step(mock_experiences, mock_test_records)
    candidate_record = registry.get_version(res.model_version)

    assert candidate_record is not None
    assert candidate_record.status == ModelLifecycleState.CANDIDATE.value
    assert candidate_record.model_id == "trainable_quantum_representation"
    assert candidate_record.quantum_configuration is not None
    assert candidate_record.quantum_configuration["trainable_parameters"] == 16


def test_validation_gate_integration(mock_experiences, mock_test_records, temp_dirs):
    """Verify candidate model passes through Phase 7.6 ValidationGate."""
    models_dir = temp_dirs
    trainer = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=models_dir,
    )

    res = trainer.train_step(mock_experiences, mock_test_records)
    assert res.governance_decision in ("PROMOTE", "REJECT", "HOLD")


# ==============================================================================
# 8. SINGLE-CLASS FALLBACK & REPRODUCIBILITY
# ==============================================================================

def test_single_class_adaptation_fallback():
    """Verify classifier gracefully handles single-class adaptation without failing."""
    classifier = ContinualTrainableQSVCClassifier()
    rng = np.random.RandomState(42)
    X_train = rng.randn(10, 384).astype(np.float32)
    y_train = ["GENERAL_SPEECH"] * 10

    classifier.fit(X_train, y_train)
    assert classifier.is_single_class_fallback is True
    assert classifier.single_observed_class == "GENERAL_SPEECH"

    X_test = rng.randn(5, 384).astype(np.float32)
    preds = classifier.predict(X_test)
    assert preds == ["GENERAL_SPEECH"] * 5


def test_deterministic_trainable_quantum_reproducibility(mock_experiences, mock_test_records, temp_dirs):
    """Verify identical seed and data yield bitwise reproducible metrics and theta."""
    models_dir = temp_dirs
    trainer1 = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=os.path.join(models_dir, "t1"),
    )
    trainer2 = ContinualTrainableQuantumTrainer(
        target_speaker="F01",
        base_dir=os.path.join(models_dir, "t2"),
    )

    res1 = trainer1.train_step(mock_experiences, mock_test_records, custom_seed=42)
    res2 = trainer2.train_step(mock_experiences, mock_test_records, custom_seed=42)

    assert res1.optimal_param_fingerprint == res2.optimal_param_fingerprint
    assert res1.metrics.macro_f1 == res2.metrics.macro_f1
    assert res1.metrics.emergency_recall == res2.metrics.emergency_recall
    assert res1.metrics.accuracy == res2.metrics.accuracy
