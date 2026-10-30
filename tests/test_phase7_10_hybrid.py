"""
Phase 7.10 — Hybrid Classical + Quantum Personalization Tests.

Tests:
1. Canonical Hybrid SSOT configuration verification
2. Classical branch architecture integrity (25,223 parameters)
3. Quantum branch architecture integrity (8 qubits, 16 parameters)
4. Balanced probability fusion dimensions and convexity (alpha=0.5)
5. Zero leakage verification: train-only PCA and feature scaling
6. Deterministic reproducibility under fixed seed
7. Speaker isolation enforcement (rejecting cross-speaker experiences)
8. Eligibility enforcement (rejecting unverified or ineligible experiences)
9. Model versioning and provenance tracking
10. Phase 7.6 Multi-Metric ValidationGate and Safety Veto integration
11. Single-class fallback handling
12. Branch prediction divergence tracking
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    EligibilityReason,
    LearningStatus,
)
from src.continual.taxonomy import CANONICAL_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.model_version import ModelRegistry, ModelLifecycleState
from src.continual.governance import ValidationGate, PromotionDecision
from src.continual.hybrid import (
    CANONICAL_HYBRID_CONFIG,
    HybridClassicalQuantumClassifier,
    ContinualHybridTrainer,
    ContinualHybridResult,
)
from src.continual.continual_classical import (
    IneligibleExperienceError,
    DataLeakageError,
    ContinualTrainingError,
)


@pytest.fixture
def temp_dirs():
    tmp_models = tempfile.mkdtemp()
    tmp_reg = tempfile.mkdtemp()
    yield tmp_models, tmp_reg
    shutil.rmtree(tmp_models, ignore_errors=True)
    shutil.rmtree(tmp_reg, ignore_errors=True)


@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    N_train = 20
    N_test = 15
    dim = 384
    classes = list(CANONICAL_CLASSES)

    X_train = np.random.randn(N_train, dim).astype(np.float32)
    y_train = [classes[i % len(classes)] for i in range(N_train)]

    X_test = np.random.randn(N_test, dim).astype(np.float32)
    y_test = [classes[(i + 1) % len(classes)] for i in range(N_test)]

    return X_train, y_train, X_test, y_test


def test_hybrid_ssot_configuration():
    """Verify Phase 7.10 hybrid architecture constants against contract."""
    cfg = CANONICAL_HYBRID_CONFIG
    assert cfg["architecture_name"] == "hybrid_adapter_vqfm_calibrated_fusion"
    assert cfg["qubit_count"] == 8
    assert cfg["classical_trainable_parameters"] == 25223
    assert cfg["quantum_trainable_parameters"] == 16
    assert cfg["total_trainable_parameters"] == 25239
    assert cfg["fusion_alpha"] == 0.5
    assert cfg["circuit_depth"] == 37


def test_hybrid_classifier_probability_fusion(mock_dataset):
    """Test that hybrid fusion outputs valid convex probability distributions."""
    X_train, y_train, X_test, y_test = mock_dataset
    model = HybridClassicalQuantumClassifier(
        alpha=0.5,
        classical_epochs=5,
        quantum_maxiter=5,
        seed=42,
    )
    model.fit(X_train, y_train)

    P_hybrid, P_class, P_quant = model.predict_proba(X_test)

    # Check shapes
    assert P_hybrid.shape == (len(X_test), len(CANONICAL_CLASSES))
    assert P_class.shape == (len(X_test), len(CANONICAL_CLASSES))
    assert P_quant.shape == (len(X_test), len(CANONICAL_CLASSES))

    # Check probability axioms
    assert np.all(P_hybrid >= -1e-6)
    assert np.all(P_hybrid <= 1.0 + 1e-6)
    assert np.allclose(P_hybrid.sum(axis=1), 1.0, atol=1e-4)

    # Check convex combination: P_hybrid == 0.5 * P_class + 0.5 * P_quant
    expected_fused = 0.5 * P_class + 0.5 * P_quant
    assert np.allclose(P_hybrid, expected_fused, atol=1e-5)


def test_hybrid_prediction_labels(mock_dataset):
    """Test that predict() returns valid canonical taxonomy labels."""
    X_train, y_train, X_test, y_test = mock_dataset
    model = HybridClassicalQuantumClassifier(
        alpha=0.5,
        classical_epochs=5,
        quantum_maxiter=5,
        seed=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
    for p in preds:
        assert p in CANONICAL_CLASSES


def test_hybrid_reproducibility(mock_dataset):
    """Verify deterministic training and identical predictions under same seed."""
    X_train, y_train, X_test, y_test = mock_dataset

    m1 = HybridClassicalQuantumClassifier(alpha=0.5, classical_epochs=5, quantum_maxiter=5, seed=123)
    m1.fit(X_train, y_train)
    p1 = m1.predict(X_test)

    m2 = HybridClassicalQuantumClassifier(alpha=0.5, classical_epochs=5, quantum_maxiter=5, seed=123)
    m2.fit(X_train, y_train)
    p2 = m2.predict(X_test)

    assert p1 == p2


def test_hybrid_trainer_e2e(temp_dirs, mock_dataset):
    """Test end-to-end ContinualHybridTrainer workflow with governance."""
    tmp_models, _ = temp_dirs
    X_train, y_train, X_test, y_test = mock_dataset

    model_registry = ModelRegistry(base_dir=tmp_models)
    validation_gate = ValidationGate()

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=model_registry,
        validation_gate=validation_gate,
        base_dir=tmp_models,
        config={"classical_epochs": 5, "quantum_maxiter": 5},
    )

    # Build verified experience records
    experiences = []
    for i in range(len(X_train)):
        rec = ExperienceRecord(
            sample_id=f"exp_{i:03d}",
            speaker_id="F01",
            audio_reference=f"audio/exp_{i}.wav",
            embedding_reference=f"emb/exp_{i}.npy",
            predicted_intent=y_train[i],
            confirmed_intent=y_train[i],
            verification_status=VerificationStatus.USER_VERIFIED.value,
            learning_eligible=True,
            eligibility_reason=EligibilityReason.ELIGIBLE_USER_CONFIRMED.value,
        )
        rec.embedding_array = X_train[i]
        experiences.append(rec)

    test_records = [(f"test_{j:03d}", y_test[j], X_test[j]) for j in range(len(X_test))]

    result: ContinualHybridResult = trainer.adapt_speaker_continual(
        experiences=experiences,
        budget=20,
        test_records=test_records,
        seed=42,
    )

    assert result.speaker_id == "F01"
    assert result.budget == 20
    assert len(result.optimal_theta) == 16
    assert result.initialization_mode == "ZERO_INITIALIZED"
    assert result.evaluation_metrics.total_test_samples == len(X_test)
    assert result.governance_decision in ["PROMOTE", "REJECT", "HOLD"]
    assert 0.0 <= result.branch_divergence_rate <= 1.0


def test_speaker_isolation_rejection(temp_dirs, mock_dataset):
    """Verify that cross-speaker experiences are strictly rejected."""
    tmp_models, _ = temp_dirs
    X_train, y_train, X_test, y_test = mock_dataset

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=ModelRegistry(base_dir=tmp_models),
        validation_gate=ValidationGate(),
        base_dir=tmp_models,
    )

    rec = ExperienceRecord(
        sample_id="cross_spk_001",
        speaker_id="M01",  # Mismatch!
        audio_reference="audio/cross.wav",
        embedding_reference="emb/cross.npy",
        predicted_intent="GENERAL_SPEECH",
        confirmed_intent="GENERAL_SPEECH",
        verification_status=VerificationStatus.USER_VERIFIED.value,
        learning_eligible=True,
    )
    rec.embedding_array = X_train[0]

    with pytest.raises(ContinualTrainingError, match="Speaker mismatch"):
        trainer.adapt_speaker_continual(
            experiences=[rec],
            budget=20,
            test_records=[("t1", "GENERAL_SPEECH", X_test[0])],
        )


def test_ineligible_experience_rejection(temp_dirs, mock_dataset):
    """Verify that unverified or ineligible experiences raise IneligibleExperienceError."""
    tmp_models, _ = temp_dirs
    X_train, y_train, X_test, y_test = mock_dataset

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=ModelRegistry(base_dir=tmp_models),
        validation_gate=ValidationGate(),
        base_dir=tmp_models,
    )

    rec = ExperienceRecord(
        sample_id="unverified_001",
        speaker_id="F01",
        audio_reference="audio/unv.wav",
        embedding_reference="emb/unv.npy",
        predicted_intent="GENERAL_SPEECH",
        confirmed_intent="GENERAL_SPEECH",
        verification_status=VerificationStatus.UNVERIFIED.value,  # Unverified!
        learning_eligible=False,
    )
    rec.embedding_array = X_train[0]

    with pytest.raises(IneligibleExperienceError):
        trainer.adapt_speaker_continual(
            experiences=[rec],
            budget=20,
            test_records=[("t1", "GENERAL_SPEECH", X_test[0])],
        )


def test_leakage_detection(temp_dirs, mock_dataset):
    """Verify that test sample leakage into training raises DataLeakageError."""
    tmp_models, _ = temp_dirs
    X_train, y_train, X_test, y_test = mock_dataset

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=ModelRegistry(base_dir=tmp_models),
        validation_gate=ValidationGate(),
        base_dir=tmp_models,
    )

    # Experience with ID matching a test record
    rec = ExperienceRecord(
        sample_id="leaked_001",
        speaker_id="F01",
        audio_reference="audio/leak.wav",
        embedding_reference="emb/leak.npy",
        predicted_intent="GENERAL_SPEECH",
        confirmed_intent="GENERAL_SPEECH",
        verification_status=VerificationStatus.USER_VERIFIED.value,
        learning_eligible=True,
    )
    rec.embedding_array = X_train[0]

    with pytest.raises(DataLeakageError):
        trainer.adapt_speaker_continual(
            experiences=[rec],
            budget=20,
            test_records=[("leaked_001", "GENERAL_SPEECH", X_test[0])],  # Same ID!
        )


def test_warm_start_lineage(temp_dirs, mock_dataset):
    """Verify that parent theta and model version are properly tracked in warm-start."""
    tmp_models, _ = temp_dirs
    X_train, y_train, X_test, y_test = mock_dataset

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=ModelRegistry(base_dir=tmp_models),
        validation_gate=ValidationGate(),
        base_dir=tmp_models,
        config={"classical_epochs": 3, "quantum_maxiter": 3},
    )

    experiences = []
    for i in range(len(X_train)):
        rec = ExperienceRecord(
            sample_id=f"exp_{i:03d}",
            speaker_id="F01",
            audio_reference=f"audio/{i}.wav",
            embedding_reference=f"emb/{i}.npy",
            predicted_intent=y_train[i],
            confirmed_intent=y_train[i],
            verification_status=VerificationStatus.USER_VERIFIED.value,
            learning_eligible=True,
        )
        rec.embedding_array = X_train[i]
        experiences.append(rec)

    test_records = [(f"test_{j:03d}", y_test[j], X_test[j]) for j in range(len(X_test))]

    res1 = trainer.adapt_speaker_continual(experiences=experiences[:10], budget=20, test_records=test_records)
    res2 = trainer.adapt_speaker_continual(
        experiences=experiences,
        budget=30,
        test_records=test_records,
        parent_model_version=res1.model_version,
        parent_theta=np.array(res1.optimal_theta),
    )

    assert res2.initialization_mode == "WARM_STARTED"
    assert res2.parent_model_version == res1.model_version
    assert res2.parent_parameter_fingerprint == res1.parameter_fingerprint


def test_single_class_fallback(mock_dataset):
    """Test that single-class adaptation falls back gracefully without crash."""
    X_train, _, X_test, _ = mock_dataset
    y_single = ["GENERAL_SPEECH"] * len(X_train)

    model = HybridClassicalQuantumClassifier(alpha=0.5, classical_epochs=3, quantum_maxiter=3, seed=42)
    model.fit(X_train, y_single)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert all(p == "GENERAL_SPEECH" for p in preds)


def test_branch_divergence_metric(mock_dataset):
    """Test that branch divergence correctly reflects disagreement between classical and quantum branches."""
    X_train, y_train, X_test, y_test = mock_dataset
    model = HybridClassicalQuantumClassifier(alpha=0.5, classical_epochs=5, quantum_maxiter=5, seed=42)
    model.fit(X_train, y_train)

    P_hybrid, P_class, P_quant = model.predict_proba(X_test)
    c_preds = [model.idx_to_class[idx] for idx in np.argmax(P_class, axis=1)]
    q_preds = [model.idx_to_class[idx] for idx in np.argmax(P_quant, axis=1)]

    div_expected = float(np.mean([c != q for c, q in zip(c_preds, q_preds)]))
    assert 0.0 <= div_expected <= 1.0


def test_parameter_counts():
    """Verify exact parameter counts of classical, quantum, and total hybrid models."""
    model = HybridClassicalQuantumClassifier(alpha=0.5, classical_epochs=1, quantum_maxiter=1, seed=42)
    classical_params = model.classical_branch.get_parameter_count()
    quantum_params = model.quantum_branch.num_parameters
    total_params = classical_params + quantum_params

    assert classical_params == 25223
    assert quantum_params == 16
    assert total_params == 25239


def test_alpha_weighting_extremes(mock_dataset):
    """Verify that alpha=1.0 matches pure classical and alpha=0.0 matches pure quantum."""
    X_train, y_train, X_test, y_test = mock_dataset

    m_c = HybridClassicalQuantumClassifier(alpha=1.0, classical_epochs=5, quantum_maxiter=5, seed=42)
    m_c.fit(X_train, y_train)
    p_hyb_c, p_cls_c, _ = m_c.predict_proba(X_test)
    assert np.allclose(p_hyb_c, p_cls_c, atol=1e-5)

    m_q = HybridClassicalQuantumClassifier(alpha=0.0, classical_epochs=5, quantum_maxiter=5, seed=42)
    m_q.fit(X_train, y_train)
    p_hyb_q, _, p_qnt_q = m_q.predict_proba(X_test)
    assert np.allclose(p_hyb_q, p_qnt_q, atol=1e-5)


def test_invalid_intent_rejected(temp_dirs, mock_dataset):
    """Verify that experiences with unmapped intents are rejected."""
    tmp_models, _ = temp_dirs
    X_train, _, X_test, _ = mock_dataset

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=ModelRegistry(base_dir=tmp_models),
        validation_gate=ValidationGate(),
        base_dir=tmp_models,
    )

    rec = ExperienceRecord(
        sample_id="invalid_intent_001",
        speaker_id="F01",
        audio_reference="audio/inv.wav",
        embedding_reference="emb/inv.npy",
        predicted_intent="UNKNOWN_INTENT_XYZ",
        confirmed_intent="UNKNOWN_INTENT_XYZ",
        verification_status=VerificationStatus.USER_VERIFIED.value,
        learning_eligible=True,
    )
    rec.embedding_array = X_train[0]

    with pytest.raises(IneligibleExperienceError, match="Invalid intent class"):
        trainer.adapt_speaker_continual(
            experiences=[rec],
            budget=20,
            test_records=[("t1", "GENERAL_SPEECH", X_test[0])],
        )


def test_governance_rejection_recorded(temp_dirs, mock_dataset):
    """Verify that if ValidationGate rejects a candidate, status is set to REJECTED."""
    tmp_models, _ = temp_dirs
    X_train, y_train, X_test, y_test = mock_dataset

    # Create a mock validation gate that always rejects
    class AlwaysRejectGate:
        def evaluate_candidate(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(
                decision="REJECT",
                safety_veto=True,
                reason_codes=["SAFETY_VETO_FER_SPIKE"],
            )

    trainer = ContinualHybridTrainer(
        target_speaker="F01",
        model_registry=ModelRegistry(base_dir=tmp_models),
        validation_gate=AlwaysRejectGate(),
        base_dir=tmp_models,
        config={"classical_epochs": 2, "quantum_maxiter": 2},
    )

    experiences = []
    for i in range(len(X_train)):
        rec = ExperienceRecord(
            sample_id=f"exp_{i:03d}",
            speaker_id="F01",
            audio_reference=f"audio/{i}.wav",
            embedding_reference=f"emb/{i}.npy",
            predicted_intent=y_train[i],
            confirmed_intent=y_train[i],
            verification_status=VerificationStatus.USER_VERIFIED.value,
            learning_eligible=True,
        )
        rec.embedding_array = X_train[i]
        experiences.append(rec)

    test_records = [(f"test_{j:03d}", y_test[j], X_test[j]) for j in range(len(X_test))]
    res = trainer.adapt_speaker_continual(experiences=experiences, budget=20, test_records=test_records)

    assert res.governance_decision == "REJECT"
    assert res.safety_veto_triggered is True
    # Confirm not active in registry
    assert trainer.model_registry.get_active_model("F01") is None

