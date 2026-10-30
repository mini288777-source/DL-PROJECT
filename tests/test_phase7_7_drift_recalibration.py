"""
Unit Tests for Phase 7.7: Speaker Drift Detection & Recalibration Engine.
Verifies all requirements from Section 38 of the Phase 7.7 contract:
- Drift Schema & Immutability
- Speaker Isolation & Boundary Enforcement
- Temporal Safety & Causality
- Distribution / Representation Drift (MMD + Cosine)
- Prediction-Distribution Drift (TVD)
- Confidence Drift (KS-test + Delta Mean)
- Calibration Drift (ECE + Brier)
- Recalibration Trigger Policy
- Controlled Recalibration & Provenance
- Phase 7.6 Validation Gate Integration & Safety Veto
- Reproducibility & Historical Immutability
"""

import copy
import os
import shutil
import tempfile
import pytest
import numpy as np

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    ModelVersionError,
)
from src.continual.governance import (
    ValidationGate,
    ValidationDecisionRecord,
    PromotionDecision,
    ValidationReasonCode,
    load_governance_policy,
)
from src.continual.drift import (
    DriftType,
    DriftStatus,
    RecalibrationTriggerState,
    DriftError,
    SpeakerIsolationError,
    TemporalSafetyError,
    DriftDataError,
    DriftWindow,
    DriftEvent,
    RecalibrationCandidate,
    compute_mmd_rbf,
    compute_centroid_cosine_distance,
    compute_prediction_tvd,
    compute_confidence_ks,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_brier_score,
    TemperatureScalingRecalibrator,
    SpeakerDriftDetector,
    ControlledRecalibrationPipeline,
    load_drift_policy,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    # 20 samples of 384-D embeddings
    return np.random.randn(20, 384).astype(np.float32)


@pytest.fixture
def reference_window(sample_embeddings):
    return DriftWindow(
        window_id="win_ref_F01",
        speaker_id="F01",
        sample_ids=[f"s_ref_{i}" for i in range(20)],
        embeddings=sample_embeddings,
        predicted_classes=[GENERAL_SPEECH_CLASS] * 18 + ["FALL", "PAIN"],
        confidences=[0.90] * 15 + [0.80] * 5,
        model_version="v_base_F01",
        true_labels=[GENERAL_SPEECH_CLASS] * 18 + ["FALL", "PAIN"],
        start_timestamp="2026-10-01T10:00:00Z",
        end_timestamp="2026-10-01T12:00:00Z",
    )


# ==============================================================================
# 1. DRIFT SCHEMA & IMMUTABILITY TESTS
# ==============================================================================

def test_drift_window_schema_validation(sample_embeddings):
    """DriftWindow must reject mismatched sample counts or empty speaker."""
    with pytest.raises(DriftDataError, match="valid, non-empty speaker_id"):
        DriftWindow(
            window_id="win1",
            speaker_id="",
            sample_ids=["s1"],
            embeddings=sample_embeddings[:1],
            predicted_classes=["GENERAL_SPEECH"],
            confidences=[0.9],
            model_version="v1",
        )

    with pytest.raises(DriftDataError, match="Embedding count"):
        DriftWindow(
            window_id="win1",
            speaker_id="F01",
            sample_ids=["s1", "s2"],
            embeddings=sample_embeddings[:1],  # 1 embedding for 2 sample_ids
            predicted_classes=["GENERAL_SPEECH", "FALL"],
            confidences=[0.9, 0.8],
            model_version="v1",
        )


def test_drift_event_immutability():
    """DriftEvent attributes cannot be modified in-place."""
    event = DriftEvent(
        drift_event_id="ev_001",
        speaker_id="F01",
        model_version="v1",
        reference_window_id="win_ref",
        current_window_id="win_curr",
        drift_type=DriftType.REPRESENTATION_DRIFT.value,
        drift_method="mmd_rbf",
        statistic=0.22,
        threshold=0.15,
        status=DriftStatus.DRIFT_DETECTED.value,
        policy_version="v1.0.0_frozen",
    )
    with pytest.raises(AttributeError, match="DriftEvent is immutable"):
        event.statistic = 0.50

    with pytest.raises(AttributeError, match="DriftEvent is immutable"):
        event.status = DriftStatus.NO_DRIFT.value


# ==============================================================================
# 2. SPEAKER ISOLATION TESTS
# ==============================================================================

def test_speaker_isolation_enforcement(reference_window, sample_embeddings):
    """Cross-speaker comparison must raise SpeakerIsolationError."""
    detector = SpeakerDriftDetector()
    detector.set_reference_window(reference_window)

    curr_window_other_speaker = DriftWindow(
        window_id="win_curr_M01",
        speaker_id="M01",  # Different speaker
        sample_ids=[f"s_curr_{i}" for i in range(20)],
        embeddings=sample_embeddings,
        predicted_classes=[GENERAL_SPEECH_CLASS] * 20,
        confidences=[0.90] * 20,
        model_version="v_base_M01",
        start_timestamp="2026-10-02T10:00:00Z",
    )

    with pytest.raises(SpeakerIsolationError, match="Speaker isolation violated"):
        detector.evaluate_drift(current_window=curr_window_other_speaker, reference_window=reference_window)


def test_speaker_references_isolated(reference_window, sample_embeddings):
    """Setting reference for Speaker A does not overwrite or bleed into Speaker B."""
    detector = SpeakerDriftDetector()
    detector.set_reference_window(reference_window)

    ref_m01 = DriftWindow(
        window_id="win_ref_M01",
        speaker_id="M01",
        sample_ids=[f"s_m_{i}" for i in range(15)],
        embeddings=sample_embeddings[:15],
        predicted_classes=[GENERAL_SPEECH_CLASS] * 15,
        confidences=[0.85] * 15,
        model_version="v_base_M01",
    )
    detector.set_reference_window(ref_m01)

    assert detector.get_reference_window("F01").speaker_id == "F01"
    assert detector.get_reference_window("M01").speaker_id == "M01"
    assert detector.get_reference_window("F01").window_id == "win_ref_F01"
    assert detector.get_reference_window("M01").window_id == "win_ref_M01"


# ==============================================================================
# 3. TEMPORAL SAFETY TESTS
# ==============================================================================

def test_temporal_safety_enforcement(reference_window, sample_embeddings):
    """Reference window with timestamp after current window violates temporal causality."""
    detector = SpeakerDriftDetector()
    detector.set_reference_window(reference_window)

    # Past window evaluated against future reference
    past_window = DriftWindow(
        window_id="win_past",
        speaker_id="F01",
        sample_ids=[f"s_past_{i}" for i in range(20)],
        embeddings=sample_embeddings,
        predicted_classes=[GENERAL_SPEECH_CLASS] * 20,
        confidences=[0.90] * 20,
        model_version="v_base_F01",
        start_timestamp="2026-09-01T10:00:00Z",  # Earlier than reference start (2026-10-01)
    )

    with pytest.raises(TemporalSafetyError, match="Temporal ordering violated"):
        detector.evaluate_drift(current_window=past_window, reference_window=reference_window)


# ==============================================================================
# 4. DISTRIBUTION / REPRESENTATION DRIFT TESTS
# ==============================================================================

def test_representation_drift_identical_distributions(sample_embeddings):
    """Identical embeddings must yield near-zero MMD and NO_DRIFT."""
    mmd = compute_mmd_rbf(sample_embeddings, sample_embeddings)
    assert mmd < 1e-4

    cos_dist = compute_centroid_cosine_distance(sample_embeddings, sample_embeddings)
    assert cos_dist < 1e-5


def test_representation_drift_synthetic_shift(sample_embeddings):
    """Significantly shifted distribution triggers DRIFT_DETECTED."""
    shifted_embeddings = sample_embeddings + 5.0  # Large distribution shift
    mmd = compute_mmd_rbf(sample_embeddings, shifted_embeddings)
    assert mmd > 0.15

    cos_dist = compute_centroid_cosine_distance(sample_embeddings, shifted_embeddings)
    assert cos_dist > 0.10


def test_insufficient_samples_returns_insufficient_data(reference_window, sample_embeddings):
    """Current window with fewer samples than policy minimum returns INSUFFICIENT_DATA."""
    detector = SpeakerDriftDetector()
    detector.set_reference_window(reference_window)

    tiny_window = DriftWindow(
        window_id="win_tiny",
        speaker_id="F01",
        sample_ids=[f"s_{i}" for i in range(5)],  # 5 samples < min_current_samples (10)
        embeddings=sample_embeddings[:5],
        predicted_classes=[GENERAL_SPEECH_CLASS] * 5,
        confidences=[0.9] * 5,
        model_version="v_base_F01",
        start_timestamp="2026-10-02T10:00:00Z",
    )

    events = detector.evaluate_drift(tiny_window)
    assert len(events) == 1
    assert events[0].status == DriftStatus.INSUFFICIENT_DATA.value


# ==============================================================================
# 5. PREDICTION DISTRIBUTION DRIFT TESTS
# ==============================================================================

def test_prediction_drift_tvd():
    """Prediction TVD computes exact total variation distance."""
    preds_ref = [GENERAL_SPEECH_CLASS] * 10
    preds_curr = [GENERAL_SPEECH_CLASS] * 5 + ["FALL"] * 5

    tvd, deltas = compute_prediction_tvd(preds_ref, preds_curr, CANONICAL_CLASSES)
    # Ref: 100% GENERAL_SPEECH. Curr: 50% GENERAL_SPEECH, 50% FALL.
    # TVD = 0.5 * (|1.0 - 0.5| + |0.0 - 0.5|) = 0.5
    assert pytest.approx(tvd, 1e-4) == 0.5
    assert pytest.approx(deltas[GENERAL_SPEECH_CLASS], 1e-4) == -0.5
    assert pytest.approx(deltas["FALL"], 1e-4) == 0.5


# ==============================================================================
# 6. CONFIDENCE DRIFT TESTS
# ==============================================================================

def test_confidence_drift_ks():
    """Confidence distribution shift detected via two-sample KS test."""
    conf_ref = [0.95] * 20
    conf_curr = [0.55] * 20  # Drastic drop in confidence

    ks_stat, p_val, delta_mean = compute_confidence_ks(conf_ref, conf_curr)
    assert ks_stat == 1.0
    assert p_val < 0.001
    assert pytest.approx(delta_mean, 1e-3) == -0.40


# ==============================================================================
# 7. CALIBRATION METRICS (ECE & BRIER SCORE) TESTS
# ==============================================================================

def test_expected_calibration_error_perfect():
    """Perfect confidence calibration yields ECE = 0."""
    # If confidence is 1.0 and all are correct, ECE is 0
    confidences = [1.0] * 10
    accuracies = [1] * 10
    ece = compute_expected_calibration_error(confidences, accuracies, num_bins=10)
    assert pytest.approx(ece, 1e-4) == 0.0


def test_expected_calibration_error_overconfident():
    """Overconfident model yields high ECE."""
    # Confidence is 1.0 but accuracy is 0.0
    confidences = [1.0] * 10
    accuracies = [0] * 10
    ece = compute_expected_calibration_error(confidences, accuracies, num_bins=10)
    assert pytest.approx(ece, 1e-4) == 1.0


def test_brier_score_computation():
    """Brier score computes mean squared error on probabilities."""
    probs = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
    ])
    true_indices = [0, 1]
    # Sample 1: (0.8 - 1)^2 + (0.2 - 0)^2 = 0.04 + 0.04 = 0.08
    # Sample 2: (0.1 - 0)^2 + (0.9 - 1)^2 = 0.01 + 0.01 = 0.02
    # Mean: (0.08 + 0.02) / 2 = 0.05
    brier = compute_brier_score(probs, true_indices)
    assert pytest.approx(brier, 1e-4) == 0.05


# ==============================================================================
# 8. RECALIBRATION TRIGGER POLICY TESTS
# ==============================================================================

def test_recalibration_trigger_policy_rules():
    """Trigger policy correctly resolves NO_RECALIBRATION, RECOMMENDED, and REQUIRED."""
    detector = SpeakerDriftDetector()

    # Case 1: No drift
    ev_no_drift = [
        DriftEvent(
            drift_event_id="e1",
            speaker_id="F01",
            model_version="v1",
            reference_window_id="w0",
            current_window_id="w1",
            drift_type=DriftType.REPRESENTATION_DRIFT.value,
            drift_method="mmd",
            statistic=0.05,
            threshold=0.15,
            status=DriftStatus.NO_DRIFT.value,
            policy_version="v1.0.0_frozen",
        )
    ]
    state, reasons = detector.assess_recalibration_trigger(ev_no_drift)
    assert state == RecalibrationTriggerState.NO_RECALIBRATION

    # Case 2: Calibration degradation -> RECALIBRATION_REQUIRED
    ev_cal_degraded = [
        DriftEvent(
            drift_event_id="e2",
            speaker_id="F01",
            model_version="v1",
            reference_window_id="w0",
            current_window_id="w1",
            drift_type=DriftType.CALIBRATION_DRIFT.value,
            drift_method="ece_delta",
            statistic=0.12,
            threshold=0.08,
            status=DriftStatus.DRIFT_DETECTED.value,
            policy_version="v1.0.0_frozen",
        )
    ]
    state, reasons = detector.assess_recalibration_trigger(ev_cal_degraded)
    assert state == RecalibrationTriggerState.RECALIBRATION_REQUIRED

    # Case 3: Compound representation + confidence drift -> RECALIBRATION_REQUIRED
    ev_compound = [
        DriftEvent(
            drift_event_id="e3a",
            speaker_id="F01",
            model_version="v1",
            reference_window_id="w0",
            current_window_id="w1",
            drift_type=DriftType.REPRESENTATION_DRIFT.value,
            drift_method="mmd",
            statistic=0.20,
            threshold=0.15,
            status=DriftStatus.DRIFT_DETECTED.value,
            policy_version="v1.0.0_frozen",
        ),
        DriftEvent(
            drift_event_id="e3b",
            speaker_id="F01",
            model_version="v1",
            reference_window_id="w0",
            current_window_id="w1",
            drift_type=DriftType.CONFIDENCE_DRIFT.value,
            drift_method="ks",
            statistic=0.40,
            threshold=0.30,
            status=DriftStatus.DRIFT_DETECTED.value,
            policy_version="v1.0.0_frozen",
        ),
    ]
    state, reasons = detector.assess_recalibration_trigger(ev_compound)
    assert state == RecalibrationTriggerState.RECALIBRATION_REQUIRED

    # Case 4: Transient single-facet drift -> RECALIBRATION_RECOMMENDED
    ev_transient = [
        DriftEvent(
            drift_event_id="e4",
            speaker_id="F01",
            model_version="v1",
            reference_window_id="w0",
            current_window_id="w1",
            drift_type=DriftType.PREDICTION_DRIFT.value,
            drift_method="tvd",
            statistic=0.30,
            threshold=0.25,
            status=DriftStatus.DRIFT_DETECTED.value,
            policy_version="v1.0.0_frozen",
        )
    ]
    state, reasons = detector.assess_recalibration_trigger(ev_transient)
    assert state == RecalibrationTriggerState.RECALIBRATION_RECOMMENDED


# ==============================================================================
# 9. TEMPERATURE SCALING & CANDIDATE GENERATION TESTS
# ==============================================================================

def test_temperature_scaling_optimization():
    """Temperature scaling fits a positive scalar T that scales probabilities."""
    recalibrator = TemperatureScalingRecalibrator()
    np.random.seed(42)
    # Logits: overconfident
    logits = np.random.randn(20, 7) * 4.0
    labels = [CANONICAL_CLASSES[i % 7] for i in range(20)]

    best_t = recalibrator.fit(logits, labels, CANONICAL_CLASSES)
    assert best_t > 0.0

    probs, confs = recalibrator.apply(logits, best_t)
    assert probs.shape == (20, 7)
    assert len(confs) == 20
    assert np.all(probs >= 0.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_recalibration_candidate_creation(temp_dir):
    """Candidate generation registers as CANDIDATE and preserves provenance."""
    registry = ModelRegistry(temp_dir)
    parent_rec = ModelVersionRecord(
        model_id="classical_adapter",
        version="v_parent_F01",
        speaker_id="F01",
        training_sample_ids=["s1", "s2"],
        training_dataset_hash="hash123",
        status=ModelLifecycleState.ACTIVE.value,
    )
    registry.register_candidate(parent_rec)
    registry.update_status("v_parent_F01", ModelLifecycleState.ACTIVE)

    val_gate = ValidationGate()
    detector = SpeakerDriftDetector()
    pipeline = ControlledRecalibrationPipeline(model_registry=registry, validation_gate=val_gate, drift_detector=detector)

    cal_samples = [
        {"sample_id": f"cal_{i}", "confirmed_intent": GENERAL_SPEECH_CLASS}
        for i in range(10)
    ]
    cal_logits = np.random.randn(10, 7)

    recal_cand, cand_rec = pipeline.generate_candidate(
        speaker_id="F01",
        parent_model_version="v_parent_F01",
        calibration_samples=cal_samples,
        calibration_logits=cal_logits,
    )

    # Invariants:
    assert cand_rec.status == ModelLifecycleState.CANDIDATE.value
    assert recal_cand.promotion_status == ModelLifecycleState.CANDIDATE.value
    assert recal_cand.temperature > 0.0
    assert recal_cand.parent_model_version == "v_parent_F01"
    assert registry.get_version(cand_rec.version).status == ModelLifecycleState.CANDIDATE.value
    # Active model must still be parent! Zero auto-promotion
    assert registry.get_active_model("F01").version == "v_parent_F01"


# ==============================================================================
# 10. PHASE 7.6 VALIDATION GATE INTEGRATION & SAFETY VETO TESTS
# ==============================================================================

def test_recalibration_passes_through_validation_gate(temp_dir):
    """Recalibration candidate must be evaluated by ValidationGate, obeying Safety Veto."""
    registry = ModelRegistry(temp_dir)
    parent_rec = ModelVersionRecord(
        model_id="classical_adapter",
        version="v_base_F01",
        speaker_id="F01",
        training_sample_ids=["s_tr_1", "s_tr_2"],
        training_dataset_hash="hash_tr",
        status=ModelLifecycleState.ACTIVE.value,
    )
    registry.register_candidate(parent_rec)
    registry.update_status("v_base_F01", ModelLifecycleState.ACTIVE)

    val_gate = ValidationGate()
    detector = SpeakerDriftDetector()
    pipeline = ControlledRecalibrationPipeline(model_registry=registry, validation_gate=val_gate, drift_detector=detector)

    # 1. Validation dataset with 12 samples (10 GENERAL_SPEECH, 2 FALL)
    val_samples = [
        {"sample_id": f"val_{i}", "confirmed_intent": GENERAL_SPEECH_CLASS, "timestamp": "2026-10-02T10:00:00Z"}
        for i in range(10)
    ] + [
        {"sample_id": "val_emerg_1", "confirmed_intent": "FALL", "timestamp": "2026-10-02T10:00:00Z"},
        {"sample_id": "val_emerg_2", "confirmed_intent": "FALL", "timestamp": "2026-10-02T10:00:00Z"},
    ]

    # Create candidate
    cal_samples = [{"sample_id": f"cal_{i}", "confirmed_intent": GENERAL_SPEECH_CLASS} for i in range(10)]
    cal_logits = np.ones((10, 7))
    recal_cand, cand_rec = pipeline.generate_candidate(
        speaker_id="F01",
        parent_model_version="v_base_F01",
        calibration_samples=cal_samples,
        calibration_logits=cal_logits,
    )

    # Candidate predict function that regresses emergency recall to 0 (Unsafe!)
    def unsafe_candidate_predict_fn(X: np.ndarray):
        return [GENERAL_SPEECH_CLASS] * len(val_samples)

    # Parent predict function that correctly predicts emergency
    def safe_parent_predict_fn(X: np.ndarray):
        return [GENERAL_SPEECH_CLASS] * 10 + ["FALL", "FALL"]

    decision_rec, promoted_rec = pipeline.evaluate_and_govern(
        candidate_record=cand_rec,
        candidate_predict_fn=unsafe_candidate_predict_fn,
        val_samples=val_samples,
        val_dataset_id="val_ds_F01",
        parent_record=parent_rec,
        parent_predict_fn=safe_parent_predict_fn,
    )

    # Safety Veto MUST trigger on emergency recall regression
    assert decision_rec.decision == PromotionDecision.REJECT.value
    assert decision_rec.safety_veto is True
    assert ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value in decision_rec.reason_codes
    assert promoted_rec is None
    # Active model must remain the safe parent!
    assert registry.get_active_model("F01").version == "v_base_F01"


# ==============================================================================
# 11. REPRODUCIBILITY & HISTORICAL IMMUTABILITY
# ==============================================================================

def test_drift_detection_reproducibility(reference_window, sample_embeddings):
    """Identical windows and policy produce identical drift events and statistics."""
    detector1 = SpeakerDriftDetector()
    detector1.set_reference_window(reference_window)

    curr_window = DriftWindow(
        window_id="win_curr",
        speaker_id="F01",
        sample_ids=[f"s_curr_{i}" for i in range(20)],
        embeddings=sample_embeddings + 0.5,
        predicted_classes=[GENERAL_SPEECH_CLASS] * 16 + ["FALL"] * 4,
        confidences=[0.75] * 20,
        model_version="v_base_F01",
        start_timestamp="2026-10-02T10:00:00Z",
    )

    events1 = detector1.evaluate_drift(curr_window)

    detector2 = SpeakerDriftDetector()
    detector2.set_reference_window(reference_window)
    events2 = detector2.evaluate_drift(curr_window)

    assert len(events1) == len(events2)
    for e1, e2 in zip(events1, events2):
        assert e1.drift_type == e2.drift_type
        assert pytest.approx(e1.statistic, 1e-6) == e2.statistic
        assert e1.status == e2.status


def test_calibration_data_anti_leakage(temp_dir):
    """Validation Gate rejects candidate if validation data overlaps training/calibration data."""
    registry = ModelRegistry(temp_dir)
    parent_rec = ModelVersionRecord(
        model_id="classical_adapter",
        version="v_base_F01",
        speaker_id="F01",
        training_sample_ids=["overlap_sample_1", "s2"],
        training_dataset_hash="hash_tr",
        status=ModelLifecycleState.ACTIVE.value,
    )
    registry.register_candidate(parent_rec)
    registry.update_status("v_base_F01", ModelLifecycleState.ACTIVE)

    val_gate = ValidationGate()
    pipeline = ControlledRecalibrationPipeline(model_registry=registry, validation_gate=val_gate)

    cal_samples = [{"sample_id": "cal_1", "confirmed_intent": GENERAL_SPEECH_CLASS}]
    recal_cand, cand_rec = pipeline.generate_candidate(
        speaker_id="F01",
        parent_model_version="v_base_F01",
        calibration_samples=cal_samples,
        calibration_logits=np.zeros((1, 7)),
    )

    # Validation set intentionally contains "overlap_sample_1" from training!
    contaminated_val = [
        ("overlap_sample_1", GENERAL_SPEECH_CLASS, np.zeros(384, dtype=np.float32)),
        ("val_2", GENERAL_SPEECH_CLASS, np.zeros(384, dtype=np.float32)),
    ]

    decision_rec, promoted_rec = pipeline.evaluate_and_govern(
        candidate_record=cand_rec,
        candidate_predict_fn=lambda X: [GENERAL_SPEECH_CLASS] * len(contaminated_val),
        val_samples=contaminated_val,
        val_dataset_id="contaminated_val",
    )

    assert decision_rec.decision == PromotionDecision.REJECT.value
    assert ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value in decision_rec.reason_codes
    assert promoted_rec is None


def test_successful_safe_recalibration_promotion(temp_dir):
    """When recalibration improves calibration without regressing emergency metrics, it can be promoted."""
    registry = ModelRegistry(temp_dir)
    parent_rec = ModelVersionRecord(
        model_id="classical_adapter",
        version="v_base_F01",
        speaker_id="F01",
        training_sample_ids=["tr1", "tr2"],
        training_dataset_hash="hash_tr",
        status=ModelLifecycleState.ACTIVE.value,
    )
    registry.register_candidate(parent_rec)
    registry.update_status("v_base_F01", ModelLifecycleState.ACTIVE)

    val_gate = ValidationGate()
    pipeline = ControlledRecalibrationPipeline(model_registry=registry, validation_gate=val_gate)

    cal_samples = [{"sample_id": "cal_1", "confirmed_intent": GENERAL_SPEECH_CLASS}]
    recal_cand, cand_rec = pipeline.generate_candidate(
        speaker_id="F01",
        parent_model_version="v_base_F01",
        calibration_samples=cal_samples,
        calibration_logits=np.zeros((1, 7)),
    )

    # Protected validation set: 10 general speech, 2 emergency
    clean_val = [
        (f"clean_val_{i}", GENERAL_SPEECH_CLASS, np.zeros(384, dtype=np.float32))
        for i in range(10)
    ] + [
        ("clean_val_e1", "FALL", np.zeros(384, dtype=np.float32)),
        ("clean_val_e2", "FALL", np.zeros(384, dtype=np.float32)),
    ]

    # Predict fn where both parent and candidate achieve 100% emergency recall and zero FER
    def accurate_predict_fn(X: np.ndarray):
        return [GENERAL_SPEECH_CLASS] * 10 + ["FALL", "FALL"]

    decision_rec, promoted_rec = pipeline.evaluate_and_govern(
        candidate_record=cand_rec,
        candidate_predict_fn=accurate_predict_fn,
        val_samples=clean_val,
        val_dataset_id="clean_val_set",
        parent_record=parent_rec,
        parent_predict_fn=accurate_predict_fn,
    )

    assert decision_rec.decision == PromotionDecision.PROMOTE.value
    assert promoted_rec is not None
    assert promoted_rec.status == ModelLifecycleState.ACTIVE.value
    assert registry.get_active_model("F01").version == cand_rec.version


def test_policy_versioning_and_immutability():
    """Policy change creates new version; historical drift event remains unchanged."""
    detector_v1 = SpeakerDriftDetector()
    assert detector_v1.policy_version == "v1.0.0_frozen"

    custom_policy = copy.deepcopy(detector_v1.policy)
    custom_policy["policy_metadata"]["policy_version"] = "v2.0.0_updated"
    custom_policy["drift_methods_and_thresholds"]["representation_drift"]["statistic_threshold"] = 0.50

    detector_v2 = SpeakerDriftDetector(policy=custom_policy)
    assert detector_v2.policy_version == "v2.0.0_updated"
    assert detector_v1.policy_version == "v1.0.0_frozen"  # Unchanged

