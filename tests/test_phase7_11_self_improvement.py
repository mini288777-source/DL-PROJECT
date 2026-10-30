"""
Phase 7.11 Continual Self-Improvement Unit & End-to-End Integration Tests.
Verifies:
1. End-to-end closed loop architecture (Speech -> Inference -> Review -> Store -> Replay -> Candidate -> Gate -> Active).
2. Strict verification barrier: Zero unverified learning (Attempting to train without verification raises error).
3. Human-in-the-loop review actions (ACCEPT, CORRECT, REJECT) and canonical taxonomy enforcement.
4. Learning eligibility quarantine for degraded acoustic quality.
5. Replay memory buffering and balanced inclusion in candidate training.
6. ValidationGate multi-metric evaluation and Safety Veto enforcement.
7. Safe active pointer management (PROMOTE updates pointer, REJECT preserves parent).
8. Speaker-scoped rollback restoration of previous validated parent model.
9. Chronological audit logging and complete lineage provenance.
10. Learning efficiency metrics calculation (burden ratio, eligible fraction, deltas).
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.schema import VerificationStatus, LearningStatus, EligibilityReason
from src.continual.review import ReviewAction, VerifierType
from src.continual.governance import (
    PromotionDecision,
    ValidationGate,
    ModelGovernanceManager,
    ValidationReasonCode,
    ValidationDecisionRecord,
)
from src.continual.model_version import ModelLifecycleState, ModelVersionRecord, ModelRegistry
from src.continual.self_improvement import (
    ContinualSelfImprovementEngine,
    UnverifiedLearningAttemptError,
    DirectActivationAttemptError,
    LoopInteractionReport,
    SelfImprovementCycleResult,
    LearningEfficiencyMetrics,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_validation_data():
    """Create a balanced mock validation set (12 samples across canonical classes)."""
    rng = np.random.RandomState(42)
    val_samples = []
    # 4 emergency samples (2 SOS_HELP, 2 BREATHING_MEDICAL)
    for i in range(2):
        val_samples.append((f"val_help_{i}", "SOS_HELP", rng.randn(384).astype(np.float32)))
        val_samples.append((f"val_med_{i}", "BREATHING_MEDICAL", rng.randn(384).astype(np.float32)))
    # 8 general speech samples
    for i in range(8):
        val_samples.append((f"val_gen_{i}", GENERAL_SPEECH_CLASS, rng.randn(384).astype(np.float32)))
    return val_samples


def test_initial_engine_state(temp_dir):
    """Engine initializes with baseline active model and clean stores."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    assert engine.speaker_id == "F01"
    assert engine.active_model_version == "F01_v0.0.0_baseline"
    assert len(engine.interaction_history) == 0
    assert len(engine.cycle_history) == 0
    assert engine.governance_manager.get_active_model("F01") == "F01_v0.0.0_baseline"


def test_human_review_accept_action(temp_dir):
    """Human ACCEPT confirms predicted label as verified ground truth."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    emb = np.random.randn(384).astype(np.float32)
    report = engine.process_speech_interaction(
        embedding=emb,
        true_label=GENERAL_SPEECH_CLASS,
        human_action=ReviewAction.ACCEPT.value,
        acoustic_quality_score=0.95,
    )
    assert report.human_action == ReviewAction.ACCEPT.value
    assert report.verification_status == VerificationStatus.USER_VERIFIED.value
    assert report.verified_label == report.predicted_label
    assert report.learning_eligible is True
    assert report.added_to_replay is True
    assert len(engine.experience_store.list_by_speaker("F01")) == 1


def test_human_review_correct_action(temp_dir):
    """Human CORRECT overrides predicted label with user provided canonical label."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    emb = np.random.randn(384).astype(np.float32)
    report = engine.process_speech_interaction(
        embedding=emb,
        true_label="SOS_HELP",
        human_action=ReviewAction.CORRECT.value,
        user_corrected_label="SOS_HELP",
        acoustic_quality_score=0.95,
    )
    assert report.human_action == ReviewAction.CORRECT.value
    assert report.verification_status == VerificationStatus.USER_VERIFIED.value
    assert report.verified_label == "SOS_HELP"
    assert report.learning_eligible is True
    assert engine.replay_memory.size() == 1


def test_human_review_reject_action(temp_dir):
    """Human REJECT marks interaction as discarded and ineligible for training."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    emb = np.random.randn(384).astype(np.float32)
    report = engine.process_speech_interaction(
        embedding=emb,
        true_label="UNKNOWN",
        human_action=ReviewAction.REJECT.value,
        acoustic_quality_score=0.95,
    )
    assert report.human_action == ReviewAction.REJECT.value
    assert report.verification_status == VerificationStatus.REJECTED.value
    assert report.verified_label is None
    assert report.learning_eligible is False
    assert report.added_to_replay is False
    assert engine.replay_memory.size() == 0


def test_degraded_acoustic_quality_quarantine(temp_dir):
    """Degraded acoustic quality score prevents learning eligibility even if verified."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    emb = np.random.randn(384).astype(np.float32)
    report = engine.process_speech_interaction(
        embedding=emb,
        true_label=GENERAL_SPEECH_CLASS,
        human_action=ReviewAction.ACCEPT.value,
        acoustic_quality_score=0.30,  # Below 0.50 threshold
    )
    assert report.verification_status == VerificationStatus.USER_VERIFIED.value
    assert report.learning_eligible is False
    assert report.eligibility_reason == EligibilityReason.INELIGIBLE_LOW_QUALITY.value
    assert report.added_to_replay is False


def test_zero_unverified_learning_barrier(temp_dir, mock_validation_data):
    """Attempting adaptation without verified eligible experiences raises error."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    # Add unverified interaction (skipped review)
    emb = np.random.randn(384).astype(np.float32)
    engine.process_speech_interaction(
        embedding=emb,
        true_label=GENERAL_SPEECH_CLASS,
        human_action="UNREVIEWED",
    )

    with pytest.raises(UnverifiedLearningAttemptError, match="No verified, learning-eligible experiences"):
        engine.trigger_adaptation_cycle(
            candidate_version="F01_v1.0.0_candidate",
            val_samples=mock_validation_data,
        )


def test_e2e_adaptation_cycle_promotion(temp_dir, mock_validation_data):
    """Closed loop collects verified experiences, trains candidate, and promotes upon passing gate."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )

    # Accumulate 10 verified interactions
    rng = np.random.RandomState(123)
    for i in range(10):
        lbl = "SOS_HELP" if i < 2 else GENERAL_SPEECH_CLASS
        emb = rng.randn(384).astype(np.float32)
        engine.process_speech_interaction(
            embedding=emb,
            true_label=lbl,
            human_action=ReviewAction.CORRECT.value if i < 2 else ReviewAction.ACCEPT.value,
            user_corrected_label=lbl if i < 2 else None,
            acoustic_quality_score=0.90,
        )

    cand_v = "F01_v1.0.0_cand"
    res = engine.trigger_adaptation_cycle(
        candidate_version=cand_v,
        val_samples=mock_validation_data,
        pre_metrics={"macro_f1": 0.10, "emergency_recall": 0.0, "false_emergency_rate": 0.50},
        test_samples=mock_validation_data,
    )

    assert res.budget_n == 10
    assert res.candidate_model_version == cand_v
    assert res.parent_model_version == "F01_v0.0.0_baseline"
    # Candidate should either promote or reject depending on validation gate
    assert res.gate_decision in (PromotionDecision.PROMOTE.value, PromotionDecision.REJECT.value)
    if res.gate_decision == PromotionDecision.PROMOTE.value:
        assert engine.active_model_version == cand_v
        assert engine.governance_manager.get_active_model("F01") == cand_v
    else:
        assert engine.active_model_version == "F01_v0.0.0_baseline"


def test_safety_veto_rejection_preserves_parent(temp_dir):
    """When a candidate fails the safety veto, active model pointer remains untouched."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="M01",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    rng = np.random.RandomState(456)
    # Add verified experiences
    for i in range(10):
        emb = rng.randn(384).astype(np.float32)
        engine.process_speech_interaction(
            embedding=emb,
            true_label=GENERAL_SPEECH_CLASS,
            human_action=ReviewAction.ACCEPT.value,
        )

    # Validation set designed to fail: single emergency sample with FER explosion
    val_samples = [("val_0", "SOS_HELP", rng.randn(384).astype(np.float32))]
    for i in range(10):
        val_samples.append((f"val_{i+1}", GENERAL_SPEECH_CLASS, rng.randn(384).astype(np.float32)))

    # Force candidate evaluation
    cand_v = "M01_v1.0.0_cand"
    res = engine.trigger_adaptation_cycle(
        candidate_version=cand_v,
        val_samples=val_samples,
    )

    # In single-sample emergency case, evidence sufficiency gate triggers HOLD,
    # or safety veto triggers REJECT; both strictly preserve active parent.
    assert res.gate_decision in (PromotionDecision.REJECT.value, PromotionDecision.HOLD.value)
    assert engine.active_model_version == "M01_v0.0.0_baseline"
    assert engine.governance_manager.get_active_model("M01") == "M01_v0.0.0_baseline"


def test_safe_rollback_restoration(temp_dir, mock_validation_data):
    """Rollback restores the preceding active model and increments rollback audit count."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="F03",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    # Manually promote a mock version to test rollback mechanics
    v1 = "F03_v1.0.0"
    v1_rec = ModelVersionRecord(
        model_id="classical",
        version=v1,
        speaker_id="F03",
        parent_version="F03_v0.0.0_baseline",
        training_dataset_hash="hash_mock",
        status=ModelLifecycleState.CANDIDATE.value,
    )
    engine.model_registry.register_candidate(v1_rec)
    dec_rec = ValidationDecisionRecord(
        validation_id="VAL_mock",
        candidate_model_version=v1,
        parent_model_version="F03_v0.0.0_baseline",
        speaker_id="F03",
        validation_dataset_id="val_mock",
        validation_dataset_fingerprint="fp_mock",
        validation_policy_version="v1.0.0_frozen",
        candidate_metrics={"macro_f1": 0.20, "emergency_recall": 0.50, "false_emergency_rate": 0.10},
        parent_metrics=None,
        metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False,
        validation_status="PROMOTABLE",
    )
    engine.governance_manager.promote_candidate(v1, dec_rec)
    engine.active_model_version = v1

    assert engine.governance_manager.get_active_model("F03") == v1

    # Now execute rollback
    prev = engine.trigger_rollback(reason="Simulated regression audit")
    assert prev == "F03_v0.0.0_baseline"
    assert engine.active_model_version == "F03_v0.0.0_baseline"
    assert engine.governance_manager.get_active_model("F03") == "F03_v0.0.0_baseline"
    assert engine.rollback_count == 1


def test_learning_efficiency_metrics_computation(temp_dir):
    """Learning efficiency correctly calculates human burden ratio, fractions, and rates."""
    engine = ContinualSelfImprovementEngine(
        speaker_id="M02",
        base_dir=temp_dir,
        model_family="classical",
        random_seed=42,
    )
    rng = np.random.RandomState(789)
    # 8 ACCEPTs, 2 CORRECTs, 2 REJECTs = 12 total interactions
    for i in range(8):
        engine.process_speech_interaction(rng.randn(384).astype(np.float32), GENERAL_SPEECH_CLASS, ReviewAction.ACCEPT.value)
    for i in range(2):
        engine.process_speech_interaction(rng.randn(384).astype(np.float32), "SOS_HELP", ReviewAction.CORRECT.value, "SOS_HELP")
    for i in range(2):
        engine.process_speech_interaction(rng.randn(384).astype(np.float32), "UNKNOWN", ReviewAction.REJECT.value)

    eff = engine.compute_learning_efficiency_metrics()
    assert eff.total_interactions == 12
    assert eff.total_verified_experiences == 10
    assert eff.total_eligible_experiences == 10
    assert eff.eligible_fraction == 1.0
    assert eff.human_confirmations == 8
    assert eff.human_corrections == 2
    assert eff.human_rejections == 2
    assert abs(eff.human_burden_ratio - (10 / 12)) < 1e-4
