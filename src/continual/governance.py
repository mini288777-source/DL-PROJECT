"""
Phase 7.6 Model Validation Gate & Safety Governance Layer.
Implements:
- Three-way promotion decision (PROMOTE, REJECT, HOLD).
- Deterministic multi-metric validation comparing Candidate vs Active Parent Model.
- Unconditional Safety Veto preventing promotion if emergency recall regresses or FER spikes.
- Data contamination detection (anti-leakage verification against training/replay sets).
- Active model pointer management and auditable, speaker-scoped rollback.
- Explainable reason code tracking and immutable governance audit logging.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import yaml

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    ModelVersionError,
)
from src.continual.continual_classical import (
    compute_comprehensive_continual_metrics,
    ContinualEvaluationMetrics,
)


class PromotionDecision(str, Enum):
    """Three-way model promotion decision enum."""
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    HOLD = "HOLD"


class GovernanceError(Exception):
    """Base exception for model governance operations."""
    pass


class ContaminationError(GovernanceError):
    """Raised when validation data is contaminated by training or replay data."""
    pass


class SafetyVetoError(GovernanceError):
    """Raised when an un-promotable model is forcibly requested for promotion."""
    pass


class RollbackError(GovernanceError):
    """Raised when rollback cannot be safely executed."""
    pass


class ValidationReasonCode(str, Enum):
    """Standardized explainable governance reason codes."""
    PROMOTED_ALL_GATES_PASS = "PROMOTED_ALL_GATES_PASS"
    REJECTED_SAFETY_VETO_EMERGENCY_RECALL = "REJECTED_SAFETY_VETO_EMERGENCY_RECALL"
    REJECTED_EMERGENCY_RECALL_REGRESSION = "REJECTED_EMERGENCY_RECALL_REGRESSION"
    REJECTED_SAFETY_VETO_FER_EXCEEDED = "REJECTED_SAFETY_VETO_FER_EXCEEDED"
    REJECTED_MACRO_F1_REGRESSION = "REJECTED_MACRO_F1_REGRESSION"
    REJECTED_PROTECTED_CLASS_REGRESSION = "REJECTED_PROTECTED_CLASS_REGRESSION"
    REJECTED_VALIDATION_CONTAMINATION = "REJECTED_VALIDATION_CONTAMINATION"
    REJECTED_INVALID_PROVENANCE = "REJECTED_INVALID_PROVENANCE"
    REJECTED_INVALID_MODEL_VERSION = "REJECTED_INVALID_MODEL_VERSION"
    REJECTED_MODEL_VERSION = "REJECTED_MODEL_VERSION"
    REJECTED_SPEAKER_MISMATCH = "REJECTED_SPEAKER_MISMATCH"
    REJECTED_PARENT_LINEAGE_MISMATCH = "REJECTED_PARENT_LINEAGE_MISMATCH"
    HOLD_INSUFFICIENT_EVIDENCE = "HOLD_INSUFFICIENT_EVIDENCE"
    HOLD_INSUFFICIENT_EMERGENCY_EVIDENCE = "HOLD_INSUFFICIENT_EMERGENCY_EVIDENCE"
    HOLD_INSUFFICIENT_VALIDATION_SAMPLES = "HOLD_INSUFFICIENT_VALIDATION_SAMPLES"
    HOLD_VALIDATION_INCOMPLETE = "HOLD_VALIDATION_INCOMPLETE"


@dataclass
class ValidationDecisionRecord:
    """Immutable validation record matching Section 15 & Section 19."""
    validation_id: str
    candidate_model_version: str
    parent_model_version: Optional[str]
    speaker_id: str
    validation_dataset_id: str
    validation_dataset_fingerprint: str
    validation_policy_version: str
    candidate_metrics: Dict[str, Any]
    parent_metrics: Optional[Dict[str, Any]]
    metric_deltas: Dict[str, Any]
    decision: str  # PROMOTE, REJECT, HOLD
    reason_codes: List[str]
    safety_veto: bool
    validation_status: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        super().__setattr__("_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError(f"ValidationDecisionRecord is immutable; cannot modify attribute '{name}'.")
        super().__setattr__(name, value)

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.candidate_metrics

    @property
    def metric_comparisons(self) -> Dict[str, Any]:
        return self.metric_deltas

    @property
    def timestamp(self) -> str:
        return self.created_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_initialized", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationDecisionRecord":
        clean_data = {k: v for k, v in data.items() if k != "_initialized"}
        return cls(**clean_data)


def load_governance_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    """Load research governance policy configuration."""
    if policy_path is None:
        policy_path = os.path.join("config", "continual", "governance_policy.yaml")
    if not os.path.exists(policy_path):
        return {
            "policy_metadata": {"policy_version": "v1.0.0_default"},
            "mandatory_safety_gates": {
                "emergency_recall_regression_tolerance": 0.00,
                "max_allowed_false_emergency_rate": 0.50,
                "max_fer_increase_tolerance": 0.05,
                "safety_veto_enabled": True,
            },
            "quality_gates": {
                "macro_f1_regression_tolerance": -0.01,
                "min_general_speech_recall": 0.70,
            },
            "evidence_sufficiency_gates": {
                "min_emergency_test_samples": 2,
                "min_total_validation_samples": 10,
            },
            "contamination_firewall": {
                "require_zero_training_overlap": True,
            },
        }
    with open(policy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ValidationGate:
    """
    Deterministic gatekeeper evaluating candidate models against active parents
    under strict safety-first governance rules.
    """

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy or load_governance_policy()
        self.policy_version = self.policy.get("policy_metadata", {}).get("policy_version", "v1.0.0_frozen")

    def evaluate_candidate(
        self,
        candidate_record: ModelVersionRecord,
        val_samples: List[Any],  # [(sid, lbl, emb)] or [(sid, lbl, emb, ts)]
        candidate_predict_fn: Callable[[np.ndarray], List[str]],
        parent_record: Optional[ModelVersionRecord] = None,
        parent_predict_fn: Optional[Callable[[np.ndarray], List[str]]] = None,
        val_dataset_id: str = "protected_validation_default",
        val_speaker_id: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
        replay_sample_ids: Optional[List[str]] = None,
        cutoff_timestamp: Optional[str] = None,
    ) -> ValidationDecisionRecord:
        """
        Execute deterministic multi-metric evaluation and gate decision.
        """
        val_id = f"VAL_{candidate_record.version}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        # 1. Dataset fingerprint
        hasher = hashlib.sha256()
        val_sample_ids = set()
        true_labels = []
        embs = []
        sample_timestamps = {}

        for item in val_samples:
            sid = item[0]
            lbl = item[1]
            emb = item[2]
            ts = item[3] if len(item) > 3 else None
            val_sample_ids.add(sid)
            true_labels.append(lbl)
            embs.append(emb)
            if ts is not None:
                sample_timestamps[sid] = str(ts)

            hasher.update(sid.encode("utf-8"))
            hasher.update(lbl.encode("utf-8"))
            hasher.update(emb.tobytes())
            if ts is not None:
                hasher.update(str(ts).encode("utf-8"))
        val_fingerprint = hasher.hexdigest()

        X_val = np.stack(embs).astype(np.float32) if len(embs) > 0 else np.empty((0, 384), dtype=np.float32)

        reason_codes: List[str] = []
        safety_veto = False

        # Provenance check (explicitly invalid hash)
        if candidate_record.training_dataset_hash == "INVALID":
            reason_codes.append(ValidationReasonCode.REJECTED_INVALID_PROVENANCE.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version if parent_record else None,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.REJECT.value,
                reason_codes=reason_codes,
                safety_veto=True,
                validation_status="FAILED_INVALID_PROVENANCE",
            )

        # Speaker mismatch check
        if val_speaker_id is not None and val_speaker_id != candidate_record.speaker_id:
            reason_codes.append(ValidationReasonCode.REJECTED_SPEAKER_MISMATCH.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version if parent_record else None,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.REJECT.value,
                reason_codes=reason_codes,
                safety_veto=True,
                validation_status="FAILED_SPEAKER_MISMATCH",
            )

        # Dataset fingerprint mismatch check
        if expected_fingerprint is not None and val_fingerprint != expected_fingerprint:
            reason_codes.append(ValidationReasonCode.REJECTED_INVALID_MODEL_VERSION.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version if parent_record else None,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.REJECT.value,
                reason_codes=reason_codes,
                safety_veto=True,
                validation_status="FAILED_FINGERPRINT_MISMATCH",
            )

        # Parent lineage check
        if parent_record is not None and candidate_record.parent_version != parent_record.version:
            reason_codes.append(ValidationReasonCode.REJECTED_PARENT_LINEAGE_MISMATCH.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.REJECT.value,
                reason_codes=reason_codes,
                safety_veto=True,
                validation_status="FAILED_PARENT_LINEAGE_MISMATCH",
            )

        # 2. Contamination checks
        # 2a. Training data overlap check
        training_ids = set(candidate_record.training_sample_ids)
        contamination_overlap = training_ids.intersection(val_sample_ids)
        if contamination_overlap:
            reason_codes.append(ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version if parent_record else None,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.REJECT.value,
                reason_codes=reason_codes,
                safety_veto=True,
                validation_status="FAILED_CONTAMINATION",
            )

        # 2b. Replay buffer overlap check
        if replay_sample_ids is not None:
            replay_overlap = set(replay_sample_ids).intersection(val_sample_ids)
            if replay_overlap:
                reason_codes.append(ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value)
                return ValidationDecisionRecord(
                    validation_id=val_id,
                    candidate_model_version=candidate_record.version,
                    parent_model_version=parent_record.version if parent_record else None,
                    speaker_id=candidate_record.speaker_id,
                    validation_dataset_id=val_dataset_id,
                    validation_dataset_fingerprint=val_fingerprint,
                    validation_policy_version=self.policy_version,
                    candidate_metrics={},
                    parent_metrics=None,
                    metric_deltas={},
                    decision=PromotionDecision.REJECT.value,
                    reason_codes=reason_codes,
                    safety_veto=True,
                    validation_status="FAILED_REPLAY_CONTAMINATION",
                )

        # 2c. Future timestamp leakage check
        if cutoff_timestamp is not None and sample_timestamps:
            future_samples = [sid for sid, ts in sample_timestamps.items() if ts > cutoff_timestamp]
            if future_samples:
                reason_codes.append(ValidationReasonCode.REJECTED_VALIDATION_CONTAMINATION.value)
                return ValidationDecisionRecord(
                    validation_id=val_id,
                    candidate_model_version=candidate_record.version,
                    parent_model_version=parent_record.version if parent_record else None,
                    speaker_id=candidate_record.speaker_id,
                    validation_dataset_id=val_dataset_id,
                    validation_dataset_fingerprint=val_fingerprint,
                    validation_policy_version=self.policy_version,
                    candidate_metrics={},
                    parent_metrics=None,
                    metric_deltas={},
                    decision=PromotionDecision.REJECT.value,
                    reason_codes=reason_codes,
                    safety_veto=True,
                    validation_status="FAILED_FUTURE_DATA_CONTAMINATION",
                )

        # 3. Evidence sufficiency check
        n_total = len(val_samples)
        n_emerg = sum(1 for lbl in true_labels if lbl in EMERGENCY_CLASSES)
        min_total = self.policy.get("evidence_sufficiency_gates", {}).get("min_total_validation_samples", 10)
        min_emerg = self.policy.get("evidence_sufficiency_gates", {}).get("min_emergency_test_samples", 2)

        if n_total < min_total:
            reason_codes.append(ValidationReasonCode.HOLD_INSUFFICIENT_VALIDATION_SAMPLES.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version if parent_record else None,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.HOLD.value,
                reason_codes=reason_codes,
                safety_veto=False,
                validation_status="HOLD_EVIDENCE",
            )

        if n_emerg < min_emerg:
            reason_codes.append(ValidationReasonCode.HOLD_INSUFFICIENT_EMERGENCY_EVIDENCE.value)
            return ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=candidate_record.version,
                parent_model_version=parent_record.version if parent_record else None,
                speaker_id=candidate_record.speaker_id,
                validation_dataset_id=val_dataset_id,
                validation_dataset_fingerprint=val_fingerprint,
                validation_policy_version=self.policy_version,
                candidate_metrics={},
                parent_metrics=None,
                metric_deltas={},
                decision=PromotionDecision.HOLD.value,
                reason_codes=reason_codes,
                safety_veto=False,
                validation_status="HOLD_EVIDENCE",
            )

        # Speaker cohort identification (TORGO dysarthric vs control)
        dysarthric_speakers = {"F01", "F03", "F04", "M01", "M02", "M04", "M05"}
        control_speakers = {"FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04", "MC05"}
        if candidate_record.speaker_id in dysarthric_speakers:
            speaker_group = "dysarthric"
        elif candidate_record.speaker_id in control_speakers:
            speaker_group = "control"
        else:
            speaker_group = "unknown"

        # 4. Compute candidate metrics
        cand_preds = candidate_predict_fn(X_val)
        cand_metrics_obj = compute_comprehensive_continual_metrics(true_labels, cand_preds)
        cand_metrics = cand_metrics_obj.to_dict()
        cand_metrics["speaker_group"] = speaker_group

        # Per-class recall computation with NOT_FEASIBLE handling (Section 13)
        cand_per_class: Dict[str, Union[float, str]] = {}
        for c in CANONICAL_CLASSES:
            c_indices = [i for i, lbl in enumerate(true_labels) if lbl == c]
            if len(c_indices) == 0:
                cand_per_class[c] = "NOT_FEASIBLE"
            else:
                c_correct = sum(1 for i in c_indices if cand_preds[i] == c)
                cand_per_class[c] = float(c_correct / len(c_indices))
        cand_metrics["per_class_recall"] = cand_per_class

        # 5. Compute parent metrics if parent exists
        parent_metrics = None
        metric_deltas: Dict[str, Any] = {}

        if parent_record and parent_predict_fn:
            parent_preds = parent_predict_fn(X_val)
            parent_metrics_obj = compute_comprehensive_continual_metrics(true_labels, parent_preds)
            parent_metrics = parent_metrics_obj.to_dict()
            parent_metrics["speaker_group"] = speaker_group

            parent_per_class: Dict[str, Union[float, str]] = {}
            per_class_deltas: Dict[str, Union[float, str]] = {}
            for c in CANONICAL_CLASSES:
                c_indices = [i for i, lbl in enumerate(true_labels) if lbl == c]
                if len(c_indices) == 0:
                    parent_per_class[c] = "NOT_FEASIBLE"
                    per_class_deltas[c] = "NOT_FEASIBLE"
                else:
                    p_correct = sum(1 for i in c_indices if parent_preds[i] == c)
                    p_rec = float(p_correct / len(c_indices))
                    parent_per_class[c] = p_rec
                    c_rec = cand_per_class[c]
                    if isinstance(c_rec, float):
                        per_class_deltas[c] = float(c_rec - p_rec)
                    else:
                        per_class_deltas[c] = "NOT_FEASIBLE"

            parent_metrics["per_class_recall"] = parent_per_class
            metric_deltas["per_class_recall_delta"] = per_class_deltas

            metric_deltas["emergency_recall_delta"] = float(
                cand_metrics_obj.emergency_recall - parent_metrics_obj.emergency_recall
            )
            metric_deltas["false_emergency_rate_delta"] = float(
                cand_metrics_obj.false_emergency_rate - parent_metrics_obj.false_emergency_rate
            )
            metric_deltas["macro_f1_delta"] = float(
                cand_metrics_obj.macro_f1 - parent_metrics_obj.macro_f1
            )
        else:
            # First model (B0 seed baseline): compare against zero reference
            metric_deltas["emergency_recall_delta"] = cand_metrics_obj.emergency_recall
            metric_deltas["false_emergency_rate_delta"] = cand_metrics_obj.false_emergency_rate
            metric_deltas["macro_f1_delta"] = cand_metrics_obj.macro_f1

        # 6. Apply Safety Gates (Safety Veto)
        safety_cfg = self.policy.get("mandatory_safety_gates", {})
        recall_tol = safety_cfg.get("emergency_recall_regression_tolerance", 0.00)
        max_fer = safety_cfg.get("max_allowed_false_emergency_rate", 0.50)
        max_fer_inc = safety_cfg.get("max_fer_increase_tolerance", 0.05)

        # Safety Veto 1: Emergency Recall Regression
        if parent_record and metric_deltas.get("emergency_recall_delta", 0.0) < recall_tol:
            safety_veto = True
            reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value)

        # Safety Veto 2: Excessive False Emergency Rate
        if cand_metrics_obj.false_emergency_rate > max_fer:
            safety_veto = True
            reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_FER_EXCEEDED.value)
        elif parent_record and metric_deltas.get("false_emergency_rate_delta", 0.0) > max_fer_inc:
            safety_veto = True
            reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_FER_EXCEEDED.value)

        # 7. Apply Quality Gates
        quality_cfg = self.policy.get("quality_gates", {})
        f1_tol = quality_cfg.get("macro_f1_regression_tolerance", -0.01)
        if parent_record and metric_deltas.get("macro_f1_delta", 0.0) < f1_tol:
            reason_codes.append(ValidationReasonCode.REJECTED_MACRO_F1_REGRESSION.value)

        # 8. Determine Final 3-way Decision
        if safety_veto or reason_codes:
            decision = PromotionDecision.REJECT.value
            val_status = "REJECTED_SAFETY_OR_QUALITY"
        else:
            decision = PromotionDecision.PROMOTE.value
            reason_codes.append(ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value)
            val_status = "PROMOTABLE"

        return ValidationDecisionRecord(
            validation_id=val_id,
            candidate_model_version=candidate_record.version,
            parent_model_version=parent_record.version if parent_record else None,
            speaker_id=candidate_record.speaker_id,
            validation_dataset_id=val_dataset_id,
            validation_dataset_fingerprint=val_fingerprint,
            validation_policy_version=self.policy_version,
            candidate_metrics=cand_metrics,
            parent_metrics=parent_metrics,
            metric_deltas=metric_deltas,
            decision=decision,
            reason_codes=reason_codes,
            safety_veto=safety_veto,
            validation_status=val_status,
        )


class ModelGovernanceManager:
    """
    Manages active model pointers, promotion execution, rollback safety,
    and the permanent governance audit trail.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        validation_gate: Optional[ValidationGate] = None,
        audit_file: Optional[str] = None,
    ):
        self.registry = model_registry
        self.gate = validation_gate or ValidationGate()
        self.audit_file = audit_file

        # Active pointer: speaker_id -> active_version_str
        self._active_pointers: Dict[str, str] = {}
        # Previous active history: speaker_id -> List[version_str]
        self._active_history: Dict[str, List[str]] = {}
        # Audit log of governance actions
        self._governance_audit: List[Dict[str, Any]] = []

        # Load active models currently registered in model_registry
        self._sync_with_registry()

    def _sync_with_registry(self) -> None:
        """Synchronize active pointers from registry."""
        for spk, versions in self.registry._by_speaker.items():
            for v in versions:
                rec = self.registry.get_version(v)
                if rec and rec.status == ModelLifecycleState.ACTIVE.value:
                    self._active_pointers[spk] = v

    def get_active_model(self, speaker_id: str) -> Optional[str]:
        """Return currently active model version for speaker."""
        return self._active_pointers.get(speaker_id)

    def promote_candidate(
        self,
        candidate_version: str,
        decision_record: ValidationDecisionRecord,
    ) -> str:
        """
        Promote candidate to ACTIVE only if decision_record certifies PROMOTE.
        Enforces:
        - Version existence
        - Decision record says PROMOTE
        - Demotes previous active to ARCHIVED
        - Tracks active history for safe rollback
        """
        if decision_record.decision != PromotionDecision.PROMOTE.value:
            raise SafetyVetoError(
                f"Cannot promote candidate '{candidate_version}': Decision is "
                f"'{decision_record.decision}' (Reasons: {decision_record.reason_codes})."
            )

        cand_record = self.registry.get_version(candidate_version)
        if not cand_record:
            raise ModelVersionError(f"Candidate version '{candidate_version}' not found in registry.")

        spk = cand_record.speaker_id
        current_active = self._active_pointers.get(spk)

        # Transition current active to ARCHIVED
        if current_active:
            self.registry.update_status(current_active, ModelLifecycleState.ARCHIVED)
            if spk not in self._active_history:
                self._active_history[spk] = []
            self._active_history[spk].append(current_active)

        # Transition candidate to ACTIVE
        self.registry.update_status(candidate_version, ModelLifecycleState.ACTIVE)
        self._active_pointers[spk] = candidate_version

        # Log audit entry
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "PROMOTED",
            "event_type": "PROMOTION",
            "model_version": candidate_version,
            "speaker_id": spk,
            "promoted_version": candidate_version,
            "superseded_version": current_active,
            "validation_id": decision_record.validation_id,
            "reason": ", ".join(decision_record.reason_codes),
            "reason_codes": decision_record.reason_codes,
            "policy_version": decision_record.validation_policy_version,
        }
        self._governance_audit.append(audit_entry)
        self._persist_audit_log()
        return candidate_version

    def reject_candidate(
        self,
        candidate_version: str,
        decision_record: ValidationDecisionRecord,
    ) -> None:
        """
        Mark candidate as REJECTED in registry. Active pointer remains unchanged.
        """
        cand_record = self.registry.get_version(candidate_version)
        if not cand_record:
            raise ModelVersionError(f"Candidate version '{candidate_version}' not found.")

        self.registry.update_status(candidate_version, ModelLifecycleState.REJECTED)

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "REJECTED",
            "event_type": "REJECTION",
            "model_version": candidate_version,
            "speaker_id": cand_record.speaker_id,
            "candidate_version": candidate_version,
            "validation_id": decision_record.validation_id,
            "reason": ", ".join(decision_record.reason_codes),
            "reason_codes": decision_record.reason_codes,
            "safety_veto": decision_record.safety_veto,
            "policy_version": decision_record.validation_policy_version,
        }
        self._governance_audit.append(audit_entry)
        self._persist_audit_log()

    def hold_candidate(
        self,
        candidate_version: str,
        decision_record: ValidationDecisionRecord,
    ) -> None:
        """
        Keep candidate in CANDIDATE state awaiting further validation evidence.
        """
        cand_record = self.registry.get_version(candidate_version)
        if not cand_record:
            raise ModelVersionError(f"Candidate version '{candidate_version}' not found.")

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "HELD",
            "event_type": "HOLD",
            "model_version": candidate_version,
            "speaker_id": cand_record.speaker_id,
            "candidate_version": candidate_version,
            "validation_id": decision_record.validation_id,
            "reason": ", ".join(decision_record.reason_codes),
            "reason_codes": decision_record.reason_codes,
            "policy_version": decision_record.validation_policy_version,
        }
        self._governance_audit.append(audit_entry)
        self._persist_audit_log()

    def rollback(self, speaker_id: str, reason: str) -> str:
        """
        Roll back active model to the most recent previous validated active model.
        Demotes currently active model to ROLLED_BACK.
        Enforces:
        - Target model existence
        - Speaker isolation (cannot rollback to another speaker's model)
        - Target model was not rejected
        """
        curr_active = self._active_pointers.get(speaker_id)
        if not curr_active:
            raise RollbackError(f"No active model found to roll back for speaker '{speaker_id}'.")

        history = self._active_history.get(speaker_id, [])
        if not history:
            raise RollbackError(f"No prior validated model exists in history for speaker '{speaker_id}'.")

        target_version = history[-1]
        target_rec = self.registry.get_version(target_version)
        if not target_rec:
            raise RollbackError(f"Target rollback model '{target_version}' does not exist in registry.")

        if target_rec.speaker_id != speaker_id:
            raise RollbackError(f"Speaker isolation violation: Target model '{target_version}' belongs to speaker '{target_rec.speaker_id}', not '{speaker_id}'.")

        if target_rec.status == ModelLifecycleState.REJECTED.value:
            raise RollbackError(f"Cannot roll back to rejected model version '{target_version}'.")

        # Pop from history once validated
        history.pop()

        # Demote current active
        self.registry.update_status(curr_active, ModelLifecycleState.ROLLED_BACK)

        # Restore previous active
        self.registry.update_status(target_version, ModelLifecycleState.ACTIVE)
        self._active_pointers[speaker_id] = target_version

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "ROLLBACK",
            "event_type": "ROLLBACK",
            "model_version": curr_active,
            "speaker_id": speaker_id,
            "rolled_back_version": curr_active,
            "restored_version": target_version,
            "reason": reason,
            "policy_version": self.gate.policy_version,
        }
        self._governance_audit.append(audit_entry)
        self._persist_audit_log()
        return target_version

    def record_candidate_created(
        self,
        model_version: str,
        speaker_id: str,
        reason: str = "Candidate model registered",
    ) -> Dict[str, Any]:
        """Record explicit CANDIDATE_CREATED audit event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "CANDIDATE_CREATED",
            "event_type": "CANDIDATE_CREATED",
            "model_version": model_version,
            "speaker_id": speaker_id,
            "reason": reason,
            "policy_version": self.gate.policy_version,
        }
        self._governance_audit.append(entry)
        self._persist_audit_log()
        return entry

    def record_validation_started(
        self,
        model_version: str,
        speaker_id: str,
        validation_id: str,
        reason: str = "Validation evaluation started",
    ) -> Dict[str, Any]:
        """Record explicit VALIDATION_STARTED audit event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "VALIDATION_STARTED",
            "event_type": "VALIDATION_STARTED",
            "model_version": model_version,
            "speaker_id": speaker_id,
            "validation_id": validation_id,
            "reason": reason,
            "policy_version": self.gate.policy_version,
        }
        self._governance_audit.append(entry)
        self._persist_audit_log()
        return entry

    def record_validation_completed(
        self,
        decision_record: ValidationDecisionRecord,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record explicit VALIDATION_COMPLETED audit event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "VALIDATION_COMPLETED",
            "event_type": "VALIDATION_COMPLETED",
            "model_version": decision_record.candidate_model_version,
            "speaker_id": decision_record.speaker_id,
            "validation_id": decision_record.validation_id,
            "decision": decision_record.decision,
            "reason": reason or ", ".join(decision_record.reason_codes),
            "policy_version": decision_record.validation_policy_version,
        }
        self._governance_audit.append(entry)
        self._persist_audit_log()
        return entry

    def retire_model(
        self,
        model_version: str,
        reason: str = "Model retired from active service",
    ) -> Dict[str, Any]:
        """Retire an active or archived model version."""
        rec = self.registry.get_version(model_version)
        if not rec:
            raise ModelVersionError(f"Model version '{model_version}' not found.")
        spk = rec.speaker_id
        if self._active_pointers.get(spk) == model_version:
            del self._active_pointers[spk]
        self.registry.update_status(model_version, ModelLifecycleState.ARCHIVED)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "RETIRED",
            "event_type": "RETIRED",
            "model_version": model_version,
            "speaker_id": spk,
            "reason": reason,
            "policy_version": self.gate.policy_version,
        }
        self._governance_audit.append(entry)
        self._persist_audit_log()
        return entry

    def evaluate_and_govern_candidate(
        self,
        candidate_version: str,
        val_samples: List[Any],
        candidate_predict_fn: Callable[[np.ndarray], List[str]],
        parent_predict_fn: Optional[Callable[[np.ndarray], List[str]]] = None,
        val_dataset_id: str = "protected_validation_default",
        val_speaker_id: Optional[str] = None,
        replay_sample_ids: Optional[List[str]] = None,
        expected_fingerprint: Optional[str] = None,
        cutoff_timestamp: Optional[str] = None,
    ) -> ValidationDecisionRecord:
        """
        Unified end-to-end candidate evaluation and lifecycle execution:
        CANDIDATE -> VALIDATION -> PROMOTE / REJECT / HOLD
        """
        cand_rec = self.registry.get_version(candidate_version)
        if not cand_rec:
            raise ModelVersionError(f"Candidate version '{candidate_version}' not found.")

        parent_rec = None
        if cand_rec.parent_version:
            parent_rec = self.registry.get_version(cand_rec.parent_version)

        val_id = f"VAL_{candidate_version}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self.record_validation_started(candidate_version, cand_rec.speaker_id, val_id)

        decision = self.gate.evaluate_candidate(
            candidate_record=cand_rec,
            val_samples=val_samples,
            candidate_predict_fn=candidate_predict_fn,
            parent_record=parent_rec,
            parent_predict_fn=parent_predict_fn,
            val_dataset_id=val_dataset_id,
            val_speaker_id=val_speaker_id or cand_rec.speaker_id,
            expected_fingerprint=expected_fingerprint,
            replay_sample_ids=replay_sample_ids,
            cutoff_timestamp=cutoff_timestamp,
        )

        self.record_validation_completed(decision)

        if decision.decision == PromotionDecision.PROMOTE.value:
            self.promote_candidate(candidate_version, decision)
        elif decision.decision == PromotionDecision.REJECT.value:
            self.reject_candidate(candidate_version, decision)
        else:
            self.hold_candidate(candidate_version, decision)

        return decision

    def _persist_audit_log(self) -> None:
        """Atomically persist governance audit log."""
        if not self.audit_file:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.audit_file)), exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.audit_file), suffix=".tmp")
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(self._governance_audit, f, indent=2)
            shutil.move(temp_path, self.audit_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_audit_trail(self, speaker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return governance audit trail."""
        if speaker_id:
            return [e for e in self._governance_audit if e.get("speaker_id") == speaker_id]
        return list(self._governance_audit)


def compute_cohort_governance_summary(
    records: List[ValidationDecisionRecord],
) -> Dict[str, Any]:
    """
    Summarize governance decisions across dysarthric, control, and overall cohorts.
    Maintains separate cohort visibility matching Section 14.
    """
    cohorts: Dict[str, List[ValidationDecisionRecord]] = {"dysarthric": [], "control": [], "overall": []}
    for rec in records:
        cohorts["overall"].append(rec)
        spk_grp = rec.candidate_metrics.get("speaker_group", "unknown")
        if spk_grp in cohorts:
            cohorts[spk_grp].append(rec)

    summary: Dict[str, Any] = {}
    for name, grp_records in cohorts.items():
        total = len(grp_records)
        promoted = sum(1 for r in grp_records if r.decision == PromotionDecision.PROMOTE.value)
        rejected = sum(1 for r in grp_records if r.decision == PromotionDecision.REJECT.value)
        held = sum(1 for r in grp_records if r.decision == PromotionDecision.HOLD.value)
        safety_vetoes = sum(1 for r in grp_records if r.safety_veto)

        rec_deltas = [
            r.metric_deltas.get("emergency_recall_delta", 0.0)
            for r in grp_records
            if "emergency_recall_delta" in r.metric_deltas and r.parent_model_version is not None
        ]
        fer_deltas = [
            r.metric_deltas.get("false_emergency_rate_delta", 0.0)
            for r in grp_records
            if "false_emergency_rate_delta" in r.metric_deltas and r.parent_model_version is not None
        ]
        f1_deltas = [
            r.metric_deltas.get("macro_f1_delta", 0.0)
            for r in grp_records
            if "macro_f1_delta" in r.metric_deltas and r.parent_model_version is not None
        ]

        summary[name] = {
            "total_candidates_evaluated": total,
            "promoted_count": promoted,
            "rejected_count": rejected,
            "hold_count": held,
            "safety_veto_count": safety_vetoes,
            "promotion_rate": float(promoted / total) if total > 0 else 0.0,
            "mean_emergency_recall_delta": float(np.mean(rec_deltas)) if len(rec_deltas) > 0 else 0.0,
            "mean_fer_delta": float(np.mean(fer_deltas)) if len(fer_deltas) > 0 else 0.0,
            "mean_macro_f1_delta": float(np.mean(f1_deltas)) if len(f1_deltas) > 0 else 0.0,
        }
    return summary

