"""
Tests for Phase 7.14 - Application Integration Pipeline
Verifies end-to-end integration:
- Acoustic inference & confidence estimation
- Degraded acoustic quality abstention
- Malformed input sanitation
- User confirmation, correction, and rejection
- Immutable experience creation & replay buffering
- Zero unverified learning enforcement
- Governed adaptation cycle & safety veto enforcement
- Active model pointer management & rollback
- Speaker isolation and metadata status reporting
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.continual.taxonomy import CANONICAL_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.review import ReviewAction
from src.continual.governance import PromotionDecision
from src.continual.self_improvement import UnverifiedLearningAttemptError
from src.continual.app_pipeline import (
    AtypicalEmergencyAppPipeline,
    AppInferenceResponse,
    AppAdaptationResponse
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_app_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_val_data():
    rng = np.random.RandomState(42)
    val = []
    # 4 emergency samples
    for i in range(2):
        val.append((f"val_sos_{i}", "SOS_HELP", rng.randn(384).astype(np.float32)))
        val.append((f"val_fall_{i}", "FALL", rng.randn(384).astype(np.float32)))
    # 8 general speech samples
    for i in range(8):
        val.append((f"val_gen_{i}", GENERAL_SPEECH_CLASS, rng.randn(384).astype(np.float32)))
    return val


def test_pipeline_initialization_and_config(temp_app_dir):
    """Verify application pipeline initializes with SSOT configuration and research disclaimer."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    assert pipeline.selected_family == "continual_classical_adapter"
    assert pipeline.confidence_threshold == 0.60
    assert pipeline.high_confidence_threshold == 0.85
    assert pipeline.min_acoustic_quality == 0.30
    assert pipeline.config["app_metadata"]["deployment_status"] == "RESEARCH DEMO ONLY"
    assert "NOT CLINICALLY VALIDATED" in pipeline.config["app_metadata"]["clinical_disclaimer"]


def test_process_audio_inference_valid(temp_app_dir):
    """Verify inference pipeline processes 384-d acoustic embedding and returns structured response."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    rng = np.random.RandomState(101)
    emb = rng.randn(384).astype(np.float32)

    res: AppInferenceResponse = pipeline.process_audio_inference(
        speaker_id="F01",
        embedding=emb,
        acoustic_quality=0.95
    )

    assert isinstance(res, AppInferenceResponse)
    assert res.speaker_id == "F01"
    assert res.session_id.startswith("SESS_F01_")
    assert res.predicted_intent in CANONICAL_CLASSES
    assert 0.0 <= res.confidence_score <= 1.0
    assert res.confidence_band in ["HIGH", "MEDIUM", "LOW"]
    assert res.suggested_action in ["CONFIRM_AND_DISPATCH", "REQUEST_CONFIRMATION", "PROCEED", "ABSTAIN_AND_ALERT"]
    assert res.active_model_version == "F01_v0.0.0_baseline"


def test_process_audio_inference_degraded_acoustic_abstention(temp_app_dir):
    """Verify acoustic quality below 0.30 triggers immediate abstention."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    rng = np.random.RandomState(102)
    emb = rng.randn(384).astype(np.float32)

    res = pipeline.process_audio_inference(
        speaker_id="F01",
        embedding=emb,
        acoustic_quality=0.25  # Below 0.30 threshold
    )

    assert res.predicted_intent == "UNKNOWN"
    assert res.suggested_action == "ABSTAIN_AND_ALERT"
    assert res.confidence_band == "LOW"


def test_malformed_input_rejections(temp_app_dir):
    """Verify malformed inputs (wrong dimensions, NaNs, Infs) are rejected at the gate."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)

    # 1. Wrong shape (128-d instead of 384-d)
    bad_shape_emb = np.zeros(128, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 384-dimensional embedding"):
        pipeline.process_audio_inference(speaker_id="F01", embedding=bad_shape_emb)

    # 2. NaN value
    nan_emb = np.zeros(384, dtype=np.float32)
    nan_emb[0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        pipeline.process_audio_inference(speaker_id="F01", embedding=nan_emb)

    # 3. Inf value
    inf_emb = np.zeros(384, dtype=np.float32)
    inf_emb[0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        pipeline.process_audio_inference(speaker_id="F01", embedding=inf_emb)


def test_user_confirmation_accept_workflow(temp_app_dir):
    """Verify user ACCEPT action validates interaction and ingests into experience store."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    emb = np.random.randn(384).astype(np.float32)

    inf_res = pipeline.process_audio_inference(speaker_id="F01", embedding=emb)
    report = pipeline.submit_user_verification(
        session_id=inf_res.session_id,
        user_action=ReviewAction.ACCEPT.value
    )

    assert report.human_action == ReviewAction.ACCEPT.value
    assert report.learning_eligible is True
    assert report.added_to_replay is True
    status = pipeline.get_speaker_status("F01")
    assert status["total_experiences_stored"] == 1
    assert status["replay_buffer_size"] == 1


def test_user_correction_workflow(temp_app_dir):
    """Verify user CORRECT action overrides prediction with canonical ground truth."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    emb = np.random.randn(384).astype(np.float32)

    inf_res = pipeline.process_audio_inference(speaker_id="F01", embedding=emb)
    report = pipeline.submit_user_verification(
        session_id=inf_res.session_id,
        user_action=ReviewAction.CORRECT.value,
        confirmed_intent="SOS_HELP"
    )

    assert report.human_action == ReviewAction.CORRECT.value
    assert report.verified_label == "SOS_HELP"
    assert report.learning_eligible is True


def test_user_rejection_discards_from_learning(temp_app_dir):
    """Verify user REJECT action marks experience as ineligible for replay."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    emb = np.random.randn(384).astype(np.float32)

    inf_res = pipeline.process_audio_inference(speaker_id="F01", embedding=emb)
    report = pipeline.submit_user_verification(
        session_id=inf_res.session_id,
        user_action=ReviewAction.REJECT.value
    )

    assert report.human_action == ReviewAction.REJECT.value
    assert report.learning_eligible is False
    assert report.added_to_replay is False


def test_zero_unverified_learning_barrier(temp_app_dir, mock_val_data):
    """Verify attempting to trigger adaptation without verified experiences raises error."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    emb = np.random.randn(384).astype(np.float32)

    # Inference occurs but user does not confirm
    pipeline.process_audio_inference(speaker_id="F01", embedding=emb)

    with pytest.raises(UnverifiedLearningAttemptError):
        pipeline.trigger_governed_adaptation(speaker_id="F01", val_samples=mock_val_data)


def test_governed_adaptation_promotion_and_rollback(temp_app_dir, mock_val_data):
    """Verify end-to-end cycle: accumulates verified data, governs candidate, and supports rollback."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    rng = np.random.RandomState(777)

    # Accumulate 10 verified interactions
    for i in range(10):
        emb = rng.randn(384).astype(np.float32)
        inf = pipeline.process_audio_inference(speaker_id="F01", embedding=emb)
        lbl = "SOS_HELP" if i < 2 else GENERAL_SPEECH_CLASS
        pipeline.submit_user_verification(
            session_id=inf.session_id,
            user_action=ReviewAction.CORRECT.value if i < 2 else ReviewAction.ACCEPT.value,
            confirmed_intent=lbl if i < 2 else None
        )

    # Trigger governed adaptation
    adapt_res: AppAdaptationResponse = pipeline.trigger_governed_adaptation(
        speaker_id="F01",
        val_samples=mock_val_data,
        candidate_version_tag="F01_v1.0.0_cand"
    )

    assert adapt_res.gate_decision in [PromotionDecision.PROMOTE.value, PromotionDecision.REJECT.value]
    if adapt_res.gate_decision == PromotionDecision.PROMOTE.value:
        assert adapt_res.active_model_version == "F01_v1.0.0_cand"
        # Test rollback restores parent
        restored = pipeline.rollback_model("F01", reason="TEST_ROLLBACK")
        assert restored == "F01_v0.0.0_baseline"
        status = pipeline.get_speaker_status("F01")
        assert status["active_model_version"] == "F01_v0.0.0_baseline"


def test_speaker_isolation_in_pipeline(temp_app_dir):
    """Verify that interaction with speaker F01 has zero cross-talk with speaker M01."""
    pipeline = AtypicalEmergencyAppPipeline(base_dir=temp_app_dir)
    emb = np.random.randn(384).astype(np.float32)

    inf_f = pipeline.process_audio_inference(speaker_id="F01", embedding=emb)
    pipeline.submit_user_verification(session_id=inf_f.session_id, user_action=ReviewAction.ACCEPT.value)

    status_f = pipeline.get_speaker_status("F01")
    status_m = pipeline.get_speaker_status("M01")

    assert status_f["total_experiences_stored"] == 1
    assert status_m["total_experiences_stored"] == 0
    assert status_f["speaker_id"] == "F01"
    assert status_m["speaker_id"] == "M01"
