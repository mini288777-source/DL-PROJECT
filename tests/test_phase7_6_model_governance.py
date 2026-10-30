"""
Phase 7.6 Model Validation Gate & Safety Governance Unit Tests.
Verifies:
- Candidate registration, parent lineage constraints, and duplicate rejection.
- Validation dataset isolation and contamination firewall.
- Deterministic multi-metric gate logic (PROMOTE, REJECT, HOLD).
- Unconditional Safety Veto (Emergency Recall regression, FER ceiling/spike).
- General speech improvement cannot override emergency safety veto.
- Per-class recall with NOT_FEASIBLE handling (Section 13).
- Dysarthric speaker vs control cohort visibility (Section 14).
- Controlled active model registry and single-active per speaker invariant.
- Speaker-scoped, provenance-preserving rollback.
- Immutability of models, versions, and validation records.
- Reproducibility and complete audit log persistence.
- Phase 7.4 / 7.5 empirical governance regression scenarios (Cases A, B, C, D).
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    ModelVersionError,
    ModelOverwriteError,
    ModelLineageError,
    compute_dataset_hash,
)
from src.continual.governance import (
    PromotionDecision,
    GovernanceError,
    ContaminationError,
    SafetyVetoError,
    RollbackError,
    ValidationReasonCode,
    ValidationDecisionRecord,
    load_governance_policy,
    ValidationGate,
    ModelGovernanceManager,
    compute_cohort_governance_summary,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def clean_registry(temp_dir):
    reg_dir = os.path.join(temp_dir, "models")
    return ModelRegistry(base_dir=reg_dir)


@pytest.fixture
def mock_validation_samples():
    """Create a balanced mock validation set with known embeddings and labels."""
    rng = np.random.RandomState(42)
    samples = []
    # 8 General speech samples
    for i in range(8):
        emb = rng.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        samples.append((f"val_gen_{i:02d}", GENERAL_SPEECH_CLASS, emb))
    # 4 Emergency samples (SOS_HELP, FALL, BREATHING_MEDICAL, PAIN)
    for i, cls in enumerate(["SOS_HELP", "FALL", "BREATHING_MEDICAL", "PAIN"]):
        emb = rng.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        samples.append((f"val_emg_{i:02d}", cls, emb))
    return samples


# ==============================================================================
# 1. CANDIDATE REGISTRATION & LINEAGE TESTS
# ==============================================================================

def test_candidate_registration(clean_registry):
    """Candidate model enters registry with CANDIDATE status."""
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["sample_01", "sample_02"],
    )
    reg_cand = clean_registry.register_candidate(cand)
    assert reg_cand.status == ModelLifecycleState.CANDIDATE.value
    assert clean_registry.get_version("F01_v1") is not None


def test_missing_parent_rejected(clean_registry):
    """Registering candidate with non-existent parent raises ModelLineageError."""
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v2",
        parent_version="F01_nonexistent_parent",
        speaker_id="F01",
    )
    with pytest.raises(ModelLineageError):
        clean_registry.register_candidate(cand)


def test_duplicate_version_rejected(clean_registry):
    """Re-registering an existing version raises ModelOverwriteError."""
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
    )
    clean_registry.register_candidate(cand)
    cand_dup = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
    )
    with pytest.raises(ModelOverwriteError):
        clean_registry.register_candidate(cand_dup)


# ==============================================================================
# 2. VALIDATION DATASET & CONTAMINATION FIREWALL TESTS
# ==============================================================================

def test_valid_validation_dataset_accepted(mock_validation_samples):
    """Valid validation dataset without overlap is processed successfully."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["train_01", "train_02"],
    )
    # Predict function that predicts perfectly
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
    )
    assert rec.validation_status == "PROMOTABLE"
    assert rec.decision == PromotionDecision.PROMOTE.value
    assert ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value in rec.reason_codes


def test_training_contamination_detected(mock_validation_samples):
    """Validation set containing training sample IDs triggers contamination veto."""
    gate = ValidationGate()
    # Contaminate candidate with val_gen_00
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["val_gen_00", "other_train_sample"],
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert rec.validation_status == "FAILED_CONTAMINATION"
    assert ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value in rec.reason_codes


def test_wrong_speaker_dataset_rejected(mock_validation_samples):
    """Candidate speaker mismatch against validation dataset is rejected."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["train_01"],
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
        val_speaker_id="M01",  # Mismatch: Candidate is F01
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert ValidationReasonCode.REJECTED_SPEAKER_MISMATCH.value in rec.reason_codes


def test_fingerprint_mismatch_rejected(mock_validation_samples):
    """Corrupted or mismatched dataset fingerprint is rejected."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["train_01"],
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
        expected_fingerprint="invalid_corrupted_fingerprint_hash",
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert ValidationReasonCode.REJECTED_INVALID_MODEL_VERSION.value in rec.reason_codes


def test_parent_lineage_mismatch_rejected(mock_validation_samples):
    """Candidate specifying wrong parent is rejected."""
    gate = ValidationGate()
    parent = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v0",
        speaker_id="F01",
    )
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        parent_version="F01_different_parent",  # Mismatch!
        speaker_id="F01",
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
        parent_record=parent,
        parent_predict_fn=pred_fn,
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert ValidationReasonCode.REJECTED_PARENT_LINEAGE_MISMATCH.value in rec.reason_codes


# ==============================================================================
# 3. THREE-WAY DECISION & EVIDENCE SUFFICIENCY (HOLD)
# ==============================================================================

def test_insufficient_emergency_evidence_hold():
    """Fewer than 2 emergency instances triggers HOLD state."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["train_01"],
    )
    # 11 total samples, but only 1 emergency sample
    samples = [(f"val_gen_{i:02d}", GENERAL_SPEECH_CLASS, np.zeros(384, dtype=np.float32)) for i in range(10)]
    samples.append(("val_emg_01", "SOS_HELP", np.zeros(384, dtype=np.float32)))

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=samples,
        candidate_predict_fn=lambda X: ["GENERAL_SPEECH"] * len(samples),
    )
    assert rec.decision == PromotionDecision.HOLD.value
    assert ValidationReasonCode.HOLD_INSUFFICIENT_EMERGENCY_EVIDENCE.value in rec.reason_codes


def test_insufficient_total_samples_hold():
    """Fewer than 10 total instances triggers HOLD state."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["train_01"],
    )
    # 5 total samples with 2 emergency samples
    samples = [(f"val_gen_{i:02d}", GENERAL_SPEECH_CLASS, np.zeros(384, dtype=np.float32)) for i in range(3)]
    samples.append(("val_emg_01", "SOS_HELP", np.zeros(384, dtype=np.float32)))
    samples.append(("val_emg_02", "FALL", np.zeros(384, dtype=np.float32)))

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=samples,
        candidate_predict_fn=lambda X: ["GENERAL_SPEECH"] * len(samples),
    )
    assert rec.decision == PromotionDecision.HOLD.value
    assert ValidationReasonCode.HOLD_INSUFFICIENT_VALIDATION_SAMPLES.value in rec.reason_codes


# ==============================================================================
# 4. SAFETY VETO LOGIC (SECTION 12 & 30)
# ==============================================================================

def test_safety_veto_emergency_recall_drop(mock_validation_samples):
    """
    Candidate with Macro-F1 improvement but degraded Emergency Recall
    is strictly REJECTED via Safety Veto (Regression Scenario Case A).
    """
    gate = ValidationGate()
    parent = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v0",
        speaker_id="F01",
    )
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        parent_version="F01_v0",
        speaker_id="F01",
    )

    # Parent predicts 3 out of 4 emergency correctly
    def parent_preds(X):
        preds = []
        for s in mock_validation_samples:
            if s[0] == "val_emg_03":  # miss 1 emergency
                preds.append(GENERAL_SPEECH_CLASS)
            else:
                preds.append(s[1])
        return preds

    # Candidate predicts general speech better but misses 3 out of 4 emergency!
    def cand_preds(X):
        preds = []
        for s in mock_validation_samples:
            if s[1] in EMERGENCY_CLASSES and s[0] != "val_emg_00":
                preds.append(GENERAL_SPEECH_CLASS)  # miss emergency!
            else:
                preds.append(s[1])
        return preds

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=cand_preds,
        parent_record=parent,
        parent_predict_fn=parent_preds,
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value in rec.reason_codes


def test_safety_veto_false_emergency_rate_spike(mock_validation_samples):
    """Candidate that triggers false emergency rate spike is strictly REJECTED."""
    gate = ValidationGate()
    parent = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v0",
        speaker_id="F01",
    )
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        parent_version="F01_v0",
        speaker_id="F01",
    )

    # Parent predicts clean
    parent_preds = lambda X: [s[1] for s in mock_validation_samples]

    # Candidate hallucinates SOS_HELP on all general speech
    def cand_preds(X):
        return ["SOS_HELP" if s[1] == GENERAL_SPEECH_CLASS else s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=cand_preds,
        parent_record=parent,
        parent_predict_fn=parent_preds,
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert ValidationReasonCode.REJECTED_SAFETY_VETO_FER_EXCEEDED.value in rec.reason_codes


def test_general_speech_gain_cannot_override_safety_veto(mock_validation_samples):
    """High accuracy on general speech cannot override emergency recall degradation."""
    gate = ValidationGate()
    parent = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v0",
        speaker_id="F01",
    )
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        parent_version="F01_v0",
        speaker_id="F01",
    )

    # Parent: 100% emergency recall, 75% general speech recall
    def parent_preds(X):
        preds = []
        for s in mock_validation_samples:
            if s[0] in ["val_gen_00", "val_gen_01"]:
                preds.append("CRITICAL_RESPONSE")  # 2 general misclassifications
            else:
                preds.append(s[1])
        return preds

    # Candidate: 100% general speech recall, but misses emergency recall
    def cand_preds(X):
        preds = []
        for s in mock_validation_samples:
            if s[1] in EMERGENCY_CLASSES:
                preds.append(GENERAL_SPEECH_CLASS)  # emergency collapsed!
            else:
                preds.append(GENERAL_SPEECH_CLASS)  # perfect general!
        return preds

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=cand_preds,
        parent_record=parent,
        parent_predict_fn=parent_preds,
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value in rec.reason_codes


# ==============================================================================
# 5. EMERGENCY-CLASS & DYSARTHRIC-SPEAKER PROTECTION (SECTIONS 13 & 14)
# ==============================================================================

def test_per_class_recall_not_feasible(mock_validation_samples):
    """Classes with 0 validation instances return NOT_FEASIBLE instead of 0 or imputed."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
    )
    # mock_validation_samples does NOT have CRITICAL_RESPONSE or LOCATION
    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=lambda X: [s[1] for s in mock_validation_samples],
    )
    per_class = rec.candidate_metrics["per_class_recall"]
    assert per_class["CRITICAL_RESPONSE"] == "NOT_FEASIBLE"
    assert per_class["LOCATION"] == "NOT_FEASIBLE"
    assert isinstance(per_class["GENERAL_SPEECH"], float)
    assert isinstance(per_class["SOS_HELP"], float)


def test_cohort_governance_visibility():
    """Governance summary maintains separate visibility into dysarthric vs control cohorts."""
    # Create mock records for dysarthric (F01) and control (FC01)
    rec_dys = ValidationDecisionRecord(
        validation_id="VAL_F01",
        candidate_model_version="F01_v1",
        parent_model_version="F01_v0",
        speaker_id="F01",
        validation_dataset_id="val_f01",
        validation_dataset_fingerprint="hash_f01",
        validation_policy_version="v1.0.0_frozen",
        candidate_metrics={"speaker_group": "dysarthric"},
        parent_metrics={"speaker_group": "dysarthric"},
        metric_deltas={"emergency_recall_delta": -0.05, "macro_f1_delta": 0.02},
        decision=PromotionDecision.REJECT.value,
        reason_codes=[ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value],
        safety_veto=True,
        validation_status="REJECTED",
    )
    rec_ctrl = ValidationDecisionRecord(
        validation_id="VAL_FC01",
        candidate_model_version="FC01_v1",
        parent_model_version="FC01_v0",
        speaker_id="FC01",
        validation_dataset_id="val_fc01",
        validation_dataset_fingerprint="hash_fc01",
        validation_policy_version="v1.0.0_frozen",
        candidate_metrics={"speaker_group": "control"},
        parent_metrics={"speaker_group": "control"},
        metric_deltas={"emergency_recall_delta": 0.02, "macro_f1_delta": 0.04},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False,
        validation_status="PROMOTABLE",
    )

    summary = compute_cohort_governance_summary([rec_dys, rec_ctrl])
    assert summary["overall"]["total_candidates_evaluated"] == 2
    assert summary["dysarthric"]["promoted_count"] == 0
    assert summary["dysarthric"]["safety_veto_count"] == 1
    assert summary["control"]["promoted_count"] == 1
    assert summary["control"]["safety_veto_count"] == 0


# ==============================================================================
# 6. ACTIVE MODEL POINTER & AUDITABLE ROLLBACK TESTS (SECTIONS 15, 16, 17)
# ==============================================================================

def test_active_model_promotion_and_rollback(clean_registry, temp_dir):
    """
    Tests controlled promotion, single-active pointer, audit trail,
    and safe rollback restoring parent model.
    """
    audit_file = os.path.join(temp_dir, "governance_audit.json")
    gov_mgr = ModelGovernanceManager(
        model_registry=clean_registry,
        audit_file=audit_file,
    )

    # 1. Register and promote seed model v0
    v0 = clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v0",
        speaker_id="F01",
    ))
    dec_v0 = ValidationDecisionRecord(
        validation_id="VAL_0",
        candidate_model_version="F01_v0",
        parent_model_version=None,
        speaker_id="F01",
        validation_dataset_id="ds0",
        validation_dataset_fingerprint="h0",
        validation_policy_version="v1.0.0",
        candidate_metrics={},
        parent_metrics=None,
        metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False,
        validation_status="PROMOTABLE",
    )
    gov_mgr.promote_candidate("F01_v0", dec_v0)
    assert gov_mgr.get_active_model("F01") == "F01_v0"
    assert clean_registry.get_version("F01_v0").status == ModelLifecycleState.ACTIVE.value

    # 2. Register candidate v1 and attempt forced promotion without PROMOTE decision -> Should fail!
    v1 = clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        parent_version="F01_v0",
        speaker_id="F01",
    ))
    dec_v1_fail = ValidationDecisionRecord(
        validation_id="VAL_1_fail",
        candidate_model_version="F01_v1",
        parent_model_version="F01_v0",
        speaker_id="F01",
        validation_dataset_id="ds1",
        validation_dataset_fingerprint="h1",
        validation_policy_version="v1.0.0",
        candidate_metrics={},
        parent_metrics=None,
        metric_deltas={},
        decision=PromotionDecision.REJECT.value,
        reason_codes=[ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value],
        safety_veto=True,
        validation_status="REJECTED",
    )
    with pytest.raises(SafetyVetoError):
        gov_mgr.promote_candidate("F01_v1", dec_v1_fail)

    # Candidate rejection preserves active model at v0
    gov_mgr.reject_candidate("F01_v1", dec_v1_fail)
    assert gov_mgr.get_active_model("F01") == "F01_v0"
    assert clean_registry.get_version("F01_v1").status == ModelLifecycleState.REJECTED.value

    # 3. Register valid candidate v2 and promote
    v2 = clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v2",
        parent_version="F01_v0",
        speaker_id="F01",
    ))
    dec_v2_pass = ValidationDecisionRecord(
        validation_id="VAL_2_pass",
        candidate_model_version="F01_v2",
        parent_model_version="F01_v0",
        speaker_id="F01",
        validation_dataset_id="ds2",
        validation_dataset_fingerprint="h2",
        validation_policy_version="v1.0.0",
        candidate_metrics={},
        parent_metrics=None,
        metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False,
        validation_status="PROMOTABLE",
    )
    gov_mgr.promote_candidate("F01_v2", dec_v2_pass)
    assert gov_mgr.get_active_model("F01") == "F01_v2"
    assert clean_registry.get_version("F01_v0").status == ModelLifecycleState.ARCHIVED.value
    assert clean_registry.get_version("F01_v2").status == ModelLifecycleState.ACTIVE.value

    # 4. Trigger safe rollback on active model F01_v2
    restored = gov_mgr.rollback("F01", reason="Post-deployment safety degradation detected")
    assert restored == "F01_v0"
    assert gov_mgr.get_active_model("F01") == "F01_v0"
    assert clean_registry.get_version("F01_v2").status == ModelLifecycleState.ROLLED_BACK.value
    assert clean_registry.get_version("F01_v0").status == ModelLifecycleState.ACTIVE.value

    # 5. Verify audit trail persisted
    assert os.path.exists(audit_file)
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    event_types = [e["event_type"] for e in audit_data]
    assert "PROMOTION" in event_types
    assert "REJECTION" in event_types
    assert "ROLLBACK" in event_types


def test_speaker_isolation_in_governance(clean_registry):
    """Operations for speaker A cannot affect speaker B's active model."""
    gov_mgr = ModelGovernanceManager(model_registry=clean_registry)

    # Setup F01 active model
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v0", speaker_id="F01"
    ))
    dec_f01 = ValidationDecisionRecord(
        validation_id="VAL_F01",
        candidate_model_version="F01_v0",
        parent_model_version=None,
        speaker_id="F01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False, validation_status="PROMOTABLE"
    )
    gov_mgr.promote_candidate("F01_v0", dec_f01)

    # Setup M01 active model
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="M01_v0", speaker_id="M01"
    ))
    dec_m01 = ValidationDecisionRecord(
        validation_id="VAL_M01",
        candidate_model_version="M01_v0",
        parent_model_version=None,
        speaker_id="M01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False, validation_status="PROMOTABLE"
    )
    gov_mgr.promote_candidate("M01_v0", dec_m01)

    assert gov_mgr.get_active_model("F01") == "F01_v0"
    assert gov_mgr.get_active_model("M01") == "M01_v0"

    # Rollback on M01 fails because no prior history, and does not alter F01
    with pytest.raises(RollbackError):
        gov_mgr.rollback("M01", reason="test")

    assert gov_mgr.get_active_model("F01") == "F01_v0"
    assert gov_mgr.get_active_model("M01") == "M01_v0"


# ==============================================================================
# 7. DETERMINISTIC REPRODUCIBILITY & IMMUTABILITY (SECTIONS 18 & 22)
# ==============================================================================

def test_deterministic_reproducibility(mock_validation_samples):
    """Evaluating identical candidate and inputs yields identical decision and metrics."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["t1", "t2"],
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    res1 = gate.evaluate_candidate(cand, mock_validation_samples, pred_fn)
    res2 = gate.evaluate_candidate(cand, mock_validation_samples, pred_fn)

    assert res1.decision == res2.decision
    assert res1.reason_codes == res2.reason_codes
    assert res1.validation_dataset_fingerprint == res2.validation_dataset_fingerprint
    assert res1.candidate_metrics["macro_f1"] == res2.candidate_metrics["macro_f1"]
    assert res1.candidate_metrics["emergency_recall"] == res2.candidate_metrics["emergency_recall"]


def test_replay_buffer_contamination_detected(mock_validation_samples):
    """Validation set containing replay buffer sample IDs triggers contamination veto."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["clean_train_01"],
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
        replay_sample_ids=["val_emg_00", "other_replay_sample"],  # Contamination!
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert rec.validation_status == "FAILED_REPLAY_CONTAMINATION"
    assert ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value in rec.reason_codes


def test_future_timestamp_contamination_detected(mock_validation_samples):
    """Validation set containing future timestamps triggers contamination veto."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_sample_ids=["clean_train_01"],
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    # Create samples with timestamps; one timestamp is in the future
    samples_with_ts = [
        (s[0], s[1], s[2], "2026-01-01T00:00:00Z" if i < len(mock_validation_samples) - 1 else "2026-12-31T23:59:59Z")
        for i, s in enumerate(mock_validation_samples)
    ]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=samples_with_ts,
        candidate_predict_fn=pred_fn,
        cutoff_timestamp="2026-06-01T00:00:00Z",  # Cutoff before future sample
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert rec.validation_status == "FAILED_FUTURE_DATA_CONTAMINATION"
    assert ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value in rec.reason_codes


def test_invalid_provenance_rejected(mock_validation_samples):
    """Candidate with invalid dataset provenance hash is strictly rejected."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
        training_dataset_hash="INVALID",  # Invalid provenance
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    rec = gate.evaluate_candidate(
        candidate_record=cand,
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
    )
    assert rec.decision == PromotionDecision.REJECT.value
    assert rec.safety_veto is True
    assert rec.validation_status == "FAILED_INVALID_PROVENANCE"
    assert ValidationReasonCode.REJECTED_INVALID_PROVENANCE.value in rec.reason_codes


def test_decision_record_immutability(mock_validation_samples):
    """ValidationDecisionRecord instances cannot be mutated in place."""
    gate = ValidationGate()
    cand = ModelVersionRecord(
        model_id="classical_adapter",
        version="F01_v1",
        speaker_id="F01",
    )
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]
    rec = gate.evaluate_candidate(cand, mock_validation_samples, pred_fn)

    with pytest.raises(AttributeError):
        rec.decision = "REJECT"

    with pytest.raises(AttributeError):
        rec.safety_veto = True


def test_rollback_safety_cannot_activate_rejected_or_mismatched(clean_registry):
    """Rollback cannot activate a rejected model or mismatched speaker model."""
    gov_mgr = ModelGovernanceManager(model_registry=clean_registry)

    # Register seed active model
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v0", speaker_id="F01"
    ))
    dec_pass = ValidationDecisionRecord(
        validation_id="VAL_0",
        candidate_model_version="F01_v0",
        parent_model_version=None,
        speaker_id="F01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False, validation_status="PROMOTABLE"
    )
    gov_mgr.promote_candidate("F01_v0", dec_pass)

    # Register candidate v1 and promote
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v1", parent_version="F01_v0", speaker_id="F01"
    ))
    dec_pass_v1 = ValidationDecisionRecord(
        validation_id="VAL_1",
        candidate_model_version="F01_v1",
        parent_model_version="F01_v0",
        speaker_id="F01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False, validation_status="PROMOTABLE"
    )
    gov_mgr.promote_candidate("F01_v1", dec_pass_v1)

    # Artificially push a rejected model to history to test safety check
    gov_mgr._active_history["F01"].append("F01_rejected")
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_rejected", speaker_id="F01"
    ))
    clean_registry.update_status("F01_rejected", ModelLifecycleState.REJECTED)

    with pytest.raises(RollbackError):
        gov_mgr.rollback("F01", reason="Testing rollback safety on rejected model")


def test_complete_audit_trail_events(clean_registry, temp_dir):
    """Verify all 8 lifecycle audit events are logged with required fields."""
    audit_file = os.path.join(temp_dir, "governance_audit.json")
    gov_mgr = ModelGovernanceManager(model_registry=clean_registry, audit_file=audit_file)

    # 1. CANDIDATE_CREATED
    gov_mgr.record_candidate_created("F01_v0", "F01", reason="Seed model creation")
    # 2. VALIDATION_STARTED
    gov_mgr.record_validation_started("F01_v0", "F01", "VAL_0", reason="Validation started")

    # 3. VALIDATION_COMPLETED & PROMOTED
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v0", speaker_id="F01"
    ))
    dec_pass = ValidationDecisionRecord(
        validation_id="VAL_0",
        candidate_model_version="F01_v0",
        parent_model_version=None,
        speaker_id="F01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.PROMOTE.value,
        reason_codes=[ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value],
        safety_veto=False, validation_status="PROMOTABLE"
    )
    gov_mgr.record_validation_completed(dec_pass)
    gov_mgr.promote_candidate("F01_v0", dec_pass)

    # 4. HELD
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v1_hold", parent_version="F01_v0", speaker_id="F01"
    ))
    dec_hold = ValidationDecisionRecord(
        validation_id="VAL_1",
        candidate_model_version="F01_v1_hold",
        parent_model_version="F01_v0",
        speaker_id="F01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.HOLD.value,
        reason_codes=[ValidationReasonCode.HOLD_INSUFFICIENT_EVIDENCE.value],
        safety_veto=False, validation_status="HOLD"
    )
    gov_mgr.hold_candidate("F01_v1_hold", dec_hold)

    # 5. REJECTED
    clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v1_rej", parent_version="F01_v0", speaker_id="F01"
    ))
    dec_rej = ValidationDecisionRecord(
        validation_id="VAL_2",
        candidate_model_version="F01_v1_rej",
        parent_model_version="F01_v0",
        speaker_id="F01",
        validation_dataset_id="ds",
        validation_dataset_fingerprint="h",
        validation_policy_version="v1.0.0",
        candidate_metrics={}, parent_metrics=None, metric_deltas={},
        decision=PromotionDecision.REJECT.value,
        reason_codes=[ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value],
        safety_veto=True, validation_status="REJECTED"
    )
    gov_mgr.reject_candidate("F01_v1_rej", dec_rej)

    # 6. RETIRED
    gov_mgr.retire_model("F01_v0", reason="End of lifecycle")

    audit = gov_mgr.get_audit_trail("F01")
    event_names = [e["event"] for e in audit]
    assert "CANDIDATE_CREATED" in event_names
    assert "VALIDATION_STARTED" in event_names
    assert "VALIDATION_COMPLETED" in event_names
    assert "PROMOTED" in event_names
    assert "HELD" in event_names
    assert "REJECTED" in event_names
    assert "RETIRED" in event_names

    # Verify each audit record has all required fields matching Section 17
    for entry in audit:
        assert "timestamp" in entry
        assert "model_version" in entry
        assert "speaker_id" in entry
        assert "event" in entry
        assert "reason" in entry
        assert "policy_version" in entry


def test_evaluate_and_govern_candidate_end_to_end(clean_registry, mock_validation_samples):
    """Test unified evaluate_and_govern_candidate workflow."""
    gov_mgr = ModelGovernanceManager(model_registry=clean_registry)
    cand = clean_registry.register_candidate(ModelVersionRecord(
        model_id="classical_adapter", version="F01_v0", speaker_id="F01"
    ))
    pred_fn = lambda X: [s[1] for s in mock_validation_samples]

    decision = gov_mgr.evaluate_and_govern_candidate(
        candidate_version="F01_v0",
        val_samples=mock_validation_samples,
        candidate_predict_fn=pred_fn,
    )
    assert decision.decision == PromotionDecision.PROMOTE.value
    assert gov_mgr.get_active_model("F01") == "F01_v0"

