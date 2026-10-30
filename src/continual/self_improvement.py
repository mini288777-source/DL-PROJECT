"""
Phase 7.11 Continual Self-Improvement Evaluation Engine.
Implements the complete verified continual-learning closed loop:
Speech -> Prediction -> Confidence/Uncertainty -> Human Verification -> Experience Store
       -> Eligibility -> Replay Memory -> Candidate Training -> Protected Validation
       -> Multi-Metric Gate + Safety Veto -> Promote / Reject / Hold -> Active Model
       -> Next Interaction.

CRITICAL RESEARCH & SAFETY INVARIANTS:
1. Prediction != Confidence != Verification != Ground Truth != Learning Eligibility.
2. Unverified predictions can NEVER enter the training set (Zero Unverified Learning).
3. Candidate models can NEVER bypass Phase 7.6 Model Validation Gate (No Direct Activation).
4. Safety Veto prevents activation if Emergency Recall regresses or False Emergency Rate spikes.
5. All transitions, decisions, and rollbacks are immutably logged with provenance.
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

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent,
)
from src.continual.confidence import (
    ConfidenceRepresentation,
    ConfidenceBand,
    ConfidenceType,
    CalibrationStatus,
    ConfidenceValidator,
    load_confidence_policy,
)
from src.continual.decision import (
    ActionDecision,
    ConfidenceDecision,
    SuggestedAction,
    PrioritizationTag,
    DecisionEngine,
)
from src.continual.review import (
    ReviewState,
    ReviewAction,
    ReviewPriority,
    VerifierType,
    load_review_policy,
)
from src.continual.eligibility import (
    EligibilityEngine,
    evaluate_learning_eligibility,
)
from src.continual.store import (
    ExperienceStore,
    InMemoryExperienceStore,
    PersistentExperienceStore,
)
from src.continual.replay import (
    ReplayMemory,
    ReplayMemoryItem,
    ReplayPolicy,
)
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    compute_dataset_hash,
)
from src.continual.governance import (
    PromotionDecision,
    ValidationReasonCode,
    ValidationDecisionRecord,
    ValidationGate,
    ModelGovernanceManager,
    load_governance_policy,
    SafetyVetoError,
)
from src.continual.continual_classical import (
    compute_comprehensive_continual_metrics,
    ContinualEvaluationMetrics,
)
from src.models.classical_adapter import ClassicalAdapterClassifier
from src.continual.continual_quantum import ContinualQSVCClassifier
from src.continual.trainable_quantum import ContinualTrainableQSVCClassifier
from src.continual.hybrid import (
    HybridClassicalQuantumClassifier,
    ContinualHybridTrainer,
    CANONICAL_HYBRID_CONFIG,
)


class SelfImprovementError(Exception):
    """Base exception for self-improvement loop operations."""
    pass


class UnverifiedLearningAttemptError(SelfImprovementError):
    """Raised when an attempt is made to train on unverified data."""
    pass


class DirectActivationAttemptError(SelfImprovementError):
    """Raised when an attempt is made to activate a model without Phase 7.6 validation."""
    pass


@dataclass
class LoopInteractionReport:
    """Detailed record of a single closed-loop interaction."""
    interaction_id: str
    speaker_id: str
    timestamp: str
    active_model_version: str
    predicted_label: str
    confidence: float
    confidence_band: str
    decision: str  # ACCEPT, CONFIRM, ABSTAIN
    human_action: str  # ACCEPT, CORRECT, REJECT, UNVERIFIED
    verified_label: Optional[str]
    verification_status: str
    stored_in_store: bool
    learning_eligible: bool
    eligibility_reason: str
    added_to_replay: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfImprovementCycleResult:
    """Outcome of an adaptation cycle triggered after experience accumulation."""
    cycle_id: str
    speaker_id: str
    budget_n: int
    candidate_model_version: str
    parent_model_version: Optional[str]
    gate_decision: str  # PROMOTE, REJECT, HOLD
    safety_veto_triggered: bool
    reason_codes: List[str]
    active_model_version_after_cycle: str
    n_train_experiences: int
    n_replay_experiences: int
    pre_cycle_metrics: Dict[str, float]
    post_cycle_metrics: Dict[str, float]
    metric_deltas: Dict[str, float]
    rollback_occurred: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LearningEfficiencyMetrics:
    """Quantitative metrics evaluating learning efficiency and human burden."""
    total_interactions: int
    total_verified_experiences: int
    total_eligible_experiences: int
    eligible_fraction: float
    human_confirmations: int
    human_corrections: int
    human_rejections: int
    human_burden_ratio: float  # (corrections + confirmations) / interactions
    total_adaptation_cycles: int
    promoted_cycles: int
    rejected_cycles: int
    held_cycles: int
    rollback_count: int
    promotion_rate: float
    rejection_rate: float
    delta_macro_f1_per_verified_exp: float
    delta_emergency_recall_per_verified_exp: float
    delta_fer_per_verified_exp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinualSelfImprovementEngine:
    """
    Closed-loop orchestrator executing end-to-end self-improvement with
    absolute verification barriers and safety governance.
    """

    def __init__(
        self,
        speaker_id: str,
        base_dir: str,
        model_family: str = "hybrid",
        governance_manager: Optional[ModelGovernanceManager] = None,
        validation_gate: Optional[ValidationGate] = None,
        confidence_policy: Optional[Dict[str, Any]] = None,
        review_policy: Optional[Dict[str, Any]] = None,
        eligibility_policy: Optional[Dict[str, Any]] = None,
        replay_capacity: int = 20,
        random_seed: int = 42,
    ):
        self.speaker_id = speaker_id
        self.base_dir = base_dir
        self.model_family = model_family
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Storage subdirectories
        self.store_dir = os.path.join(base_dir, "experience_store")
        self.models_dir = os.path.join(base_dir, "models")
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Core registries and stores
        self.experience_store = PersistentExperienceStore(storage_dir=self.store_dir)
        self.replay_memory = ReplayMemory(speaker_id=self.speaker_id, max_capacity=replay_capacity)
        self.model_registry = ModelRegistry(base_dir=self.models_dir)

        # Decision & Policy engines
        self.confidence_policy = confidence_policy or load_confidence_policy()
        self.review_policy = review_policy or load_review_policy()
        self.eligibility_policy = eligibility_policy or self.confidence_policy

        self.decision_engine = DecisionEngine(policy=self.confidence_policy)
        self.eligibility_engine = EligibilityEngine()

        # Governance
        self.validation_gate = validation_gate or ValidationGate()
        self.governance_manager = governance_manager or ModelGovernanceManager(
            model_registry=self.model_registry,
            validation_gate=self.validation_gate,
            audit_file=os.path.join(base_dir, "governance_audit.json"),
        )

        # Active Model State
        self.active_model: Optional[Any] = None
        self.active_model_version: str = "v0.0.0_initial"

        # Initialize Base Initial Model
        self._initialize_base_model()

        # Interaction & Cycle histories
        self.interaction_history: List[LoopInteractionReport] = []
        self.cycle_history: List[SelfImprovementCycleResult] = []
        self.rollback_count: int = 0

    def _initialize_base_model(self) -> None:
        """Register and initialize the baseline model version."""
        init_version = f"{self.speaker_id}_v0.0.0_baseline"
        rec = ModelVersionRecord(
            model_id=f"continual_{self.model_family}",
            version=init_version,
            speaker_id=self.speaker_id,
            parent_version=None,
            training_dataset_hash="hash_baseline_zero",
            training_sample_ids=[],
            status=ModelLifecycleState.ACTIVE.value,
        )
        self.model_registry.register_candidate(rec)
        self.model_registry.update_status(init_version, ModelLifecycleState.ACTIVE)
        self.governance_manager._active_pointers[self.speaker_id] = init_version
        self.active_model_version = init_version

    def set_active_model_instance(self, model_instance: Any, version: str) -> None:
        """Attach an in-memory trained model instance for inference."""
        self.active_model = model_instance
        self.active_model_version = version

    def process_speech_interaction(
        self,
        embedding: np.ndarray,
        true_label: str,
        human_action: str = ReviewAction.ACCEPT.value,
        user_corrected_label: Optional[str] = None,
        verifier_role: str = VerifierType.USER.value,
        acoustic_quality_score: float = 0.95,
    ) -> LoopInteractionReport:
        """
        Execute an end-to-end user speech interaction step:
        1. Forward prediction through active model.
        2. Uncertainty & Confidence quantification.
        3. Decision routing (Accept, Confirm, Abstain).
        4. Human review verification barrier (NEVER automatic ground truth).
        5. Verification status assignment.
        6. Experience storage & Replay eligibility.
        """
        interaction_id = f"INT_{self.speaker_id}_{len(self.interaction_history):04d}_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
        ts = datetime.now(timezone.utc).isoformat()

        # 1. Forward inference through active model
        if self.active_model is not None and hasattr(self.active_model, "predict_proba"):
            X_input = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
            probs = self.active_model.predict_proba(X_input)[0]
            top_idx = int(np.argmax(probs))
            predicted_label = CANONICAL_CLASSES[top_idx]
            raw_confidence = float(probs[top_idx])
            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])
            entropy = float(-np.sum([p * np.log(p + 1e-12) for p in probs if p > 0]))
        elif self.active_model is not None and hasattr(self.active_model, "predict"):
            X_input = embedding.reshape(1, -1) if embedding.ndim == 1 else embedding
            pred = self.active_model.predict(X_input)
            predicted_label = pred[0] if isinstance(pred, (list, np.ndarray)) else str(pred)
            raw_confidence = 0.85
            margin = 0.20
            entropy = 0.50
        else:
            # Fallback for initial un-personalized state
            predicted_label = GENERAL_SPEECH_CLASS
            raw_confidence = 0.50
            margin = 0.05
            entropy = 1.00

        # 2. Decision Engine routing
        conf_rep = ConfidenceRepresentation.create(
            raw_score=raw_confidence,
            confidence_value=raw_confidence,
            source_model=self.active_model_version,
            confidence_type=ConfidenceType.SOFTMAX_PROBABILITY,
            calibration_status=CalibrationStatus.TEMPERATURE_SCALED,
            policy=self.confidence_policy,
        )
        conf_band = conf_rep.confidence_band

        # Decision
        if is_emergency_intent(predicted_label):
            decision = ActionDecision.CONFIRM.value if raw_confidence >= 0.70 else ActionDecision.ABSTAIN.value
        else:
            decision = ActionDecision.ACCEPT.value if raw_confidence >= 0.65 else ActionDecision.CONFIRM.value

        # 3. Human Verification Barrier (Strict Safety)
        verified_label: Optional[str] = None
        ver_status = VerificationStatus.UNVERIFIED.value

        if human_action == ReviewAction.ACCEPT.value:
            # User confirmed predicted label as ground truth
            verified_label = predicted_label
            ver_status = VerificationStatus.USER_VERIFIED.value
        elif human_action == ReviewAction.CORRECT.value:
            # User explicitly corrected prediction to a canonical class
            if user_corrected_label and is_valid_intent(user_corrected_label):
                verified_label = user_corrected_label
                ver_status = VerificationStatus.USER_VERIFIED.value
            else:
                ver_status = VerificationStatus.REJECTED.value
        elif human_action == ReviewAction.REJECT.value:
            # Discarded / unintelligible
            ver_status = VerificationStatus.REJECTED.value
        else:
            # Unreviewed / skipped
            ver_status = VerificationStatus.UNVERIFIED.value

        # 4. Eligibility Check
        is_eligible = False
        elig_reason = EligibilityReason.INELIGIBLE_UNVERIFIED.value
        if ver_status == VerificationStatus.USER_VERIFIED.value and verified_label is not None:
            if acoustic_quality_score >= 0.50:
                is_eligible = True
                elig_reason = EligibilityReason.ELIGIBLE.value
            else:
                elig_reason = EligibilityReason.INELIGIBLE_LOW_QUALITY.value
        else:
            elig_reason = EligibilityReason.INELIGIBLE_UNVERIFIED.value

        # 5. Experience Record Creation & Storage
        rec = ExperienceRecord(
            sample_id=interaction_id,
            speaker_id=self.speaker_id,
            session_id=f"SES_{self.speaker_id}_main",
            timestamp=ts,
            audio_reference=f"audio_{interaction_id}",
            embedding_reference=f"emb_{interaction_id}",
            predicted_intent=predicted_label,
            prediction_confidence=raw_confidence,
            verification_status=ver_status,
            confirmed_intent=verified_label,
            confirmation_source=verifier_role if verified_label else "unconfirmed",
            verification_timestamp=ts if verified_label else None,
            verifier_id=f"human_{self.speaker_id}" if verified_label else None,
            learning_eligible=is_eligible,
            learning_status=LearningStatus.USER_CONFIRMED.value if is_eligible else LearningStatus.UNREVIEWED.value,
            eligibility_reason=elig_reason,
            quality_score=acoustic_quality_score,
        )
        rec.embedding_array = embedding.astype(np.float32)
        self.experience_store.create(rec)

        # 6. Replay buffer update if eligible
        added_to_replay = False
        if is_eligible:
            self.replay_memory.add(rec)
            added_to_replay = True

        report = LoopInteractionReport(
            interaction_id=interaction_id,
            speaker_id=self.speaker_id,
            timestamp=ts,
            active_model_version=self.active_model_version,
            predicted_label=predicted_label,
            confidence=raw_confidence,
            confidence_band=conf_band.value,
            decision=decision,
            human_action=human_action,
            verified_label=verified_label,
            verification_status=ver_status,
            stored_in_store=True,
            learning_eligible=is_eligible,
            eligibility_reason=elig_reason,
            added_to_replay=added_to_replay,
        )
        self.interaction_history.append(report)
        return report

    def trigger_adaptation_cycle(
        self,
        candidate_version: str,
        val_samples: List[Tuple[str, str, np.ndarray]],
        pre_metrics: Optional[Dict[str, float]] = None,
        test_samples: Optional[List[Tuple[str, str, np.ndarray]]] = None,
    ) -> SelfImprovementCycleResult:
        """
        Execute an adaptation training cycle on accumulated eligible experiences:
        1. Fetch all learning-eligible verified experiences (Guaranteed Zero Unverified Learning).
        2. Combine with balanced replay buffer experiences.
        3. Train candidate model matching model_family.
        4. Validate candidate against active parent using Phase 7.6 ValidationGate.
        5. Enforce Multi-Metric Gate & Safety Veto:
           - PROMOTE: Active model pointer updated.
           - REJECT: Candidate discarded, active model pointer retained.
           - HOLD: Retain active model, hold candidate.
        6. Return complete cycle provenance record.
        """
        cycle_id = f"CYC_{self.speaker_id}_{len(self.cycle_history):02d}"

        # 1. Fetch eligible experiences strictly
        eligible_records = [
            r for r in self.experience_store.list_by_speaker(self.speaker_id)
            if r.learning_eligible and r.verification_status == VerificationStatus.USER_VERIFIED.value
        ]

        if len(eligible_records) == 0:
            raise UnverifiedLearningAttemptError(
                f"Cannot run adaptation cycle '{cycle_id}': No verified, learning-eligible experiences found."
            )

        # Safety verification: ensure NO unverified records leaked
        for r in eligible_records:
            if r.verification_status != VerificationStatus.USER_VERIFIED.value:
                raise UnverifiedLearningAttemptError(
                    f"CRITICAL SAFETY VIOLATION: Unverified record '{r.sample_id}' found in eligible set!"
                )

        # 2. Replay experiences
        replay_items = self.replay_memory.sample(policy=ReplayPolicy.BALANCED_REPLAY)
        replay_sids = [item.experience_id for item in replay_items]

        # Combine training set: (sample_id, label, embedding)
        train_samples_map: Dict[str, Tuple[str, np.ndarray]] = {}
        for r in eligible_records:
            emb = getattr(r, "embedding_array", None)
            if emb is not None:
                train_samples_map[r.sample_id] = (r.confirmed_intent, emb)

        for rep in replay_items:
            if rep.experience_id not in train_samples_map:
                emb = rep.embedding_array
                if emb is not None:
                    train_samples_map[rep.experience_id] = (rep.class_label, emb)

        train_sids = list(train_samples_map.keys())
        y_train = [train_samples_map[sid][0] for sid in train_sids]
        X_train = np.stack([train_samples_map[sid][1] for sid in train_sids]).astype(np.float32)
        dataset_hash = compute_dataset_hash(X_train, y_train)

        # Register candidate in registry
        parent_version = self.active_model_version
        parent_record = self.model_registry.get_version(parent_version)

        cand_record = ModelVersionRecord(
            model_id=f"continual_{self.model_family}",
            version=candidate_version,
            speaker_id=self.speaker_id,
            parent_version=parent_version,
            training_sample_ids=train_sids,
            training_dataset_hash=dataset_hash,
            status=ModelLifecycleState.CANDIDATE.value,
        )
        self.model_registry.register_candidate(cand_record)

        # 3. Train Candidate Model
        candidate_model = self._train_candidate_model(X_train, y_train)

        # 4. Predict functions for ValidationGate
        def candidate_predict_fn(X: np.ndarray) -> List[str]:
            return candidate_model.predict(X)

        def parent_predict_fn(X: np.ndarray) -> List[str]:
            if self.active_model is not None and hasattr(self.active_model, "predict"):
                return self.active_model.predict(X)
            return [GENERAL_SPEECH_CLASS] * len(X)

        # 5. Evaluate Candidate via Phase 7.6 Validation Gate
        decision_record = self.validation_gate.evaluate_candidate(
            candidate_record=cand_record,
            val_samples=val_samples,
            candidate_predict_fn=candidate_predict_fn,
            parent_record=parent_record,
            parent_predict_fn=parent_predict_fn,
            val_dataset_id=f"val_{self.speaker_id}",
            val_speaker_id=self.speaker_id,
            replay_sample_ids=replay_sids,
        )

        # 6. Apply Promotion or Rejection via Governance Manager
        active_after = parent_version
        if decision_record.decision == PromotionDecision.PROMOTE.value:
            self.governance_manager.promote_candidate(candidate_version, decision_record)
            self.set_active_model_instance(candidate_model, candidate_version)
            active_after = candidate_version
        elif decision_record.decision == PromotionDecision.REJECT.value:
            self.governance_manager.reject_candidate(candidate_version, decision_record)
            active_after = parent_version
        else:
            # HOLD
            active_after = parent_version

        # 7. Evaluate on held-out test set if provided
        post_metrics = {}
        if test_samples:
            X_test = np.stack([s[2] for s in test_samples]).astype(np.float32)
            y_test = [s[1] for s in test_samples]
            preds_test = candidate_predict_fn(X_test) if decision_record.decision == PromotionDecision.PROMOTE.value else parent_predict_fn(X_test)
            eval_metrics = compute_comprehensive_continual_metrics(y_true=y_test, y_pred=preds_test)
            post_metrics = {
                "macro_f1": eval_metrics.macro_f1,
                "emergency_recall": eval_metrics.emergency_recall,
                "emergency_precision": eval_metrics.emergency_precision,
                "false_emergency_rate": eval_metrics.false_emergency_rate,
                "accuracy": eval_metrics.accuracy,
            }

        pre_m = pre_metrics or {}
        deltas = {}
        for k, v in post_metrics.items():
            if k in pre_m:
                deltas[f"{k}_delta"] = v - pre_m[k]

        cycle_res = SelfImprovementCycleResult(
            cycle_id=cycle_id,
            speaker_id=self.speaker_id,
            budget_n=len(eligible_records),
            candidate_model_version=candidate_version,
            parent_model_version=parent_version,
            gate_decision=decision_record.decision,
            safety_veto_triggered=decision_record.safety_veto,
            reason_codes=decision_record.reason_codes,
            active_model_version_after_cycle=active_after,
            n_train_experiences=len(train_sids),
            n_replay_experiences=len(replay_items),
            pre_cycle_metrics=pre_m,
            post_cycle_metrics=post_metrics,
            metric_deltas=deltas,
            rollback_occurred=False,
        )
        self.cycle_history.append(cycle_res)
        return cycle_res

    def trigger_rollback(self, reason: str = "Empirical post-activation anomaly") -> str:
        """Execute safe speaker-scoped rollback via Governance Manager."""
        previous_version = self.governance_manager.rollback(
            speaker_id=self.speaker_id,
            reason=reason,
        )
        self.active_model_version = previous_version
        self.rollback_count += 1
        return previous_version

    def _train_candidate_model(self, X_train: np.ndarray, y_train: List[str]) -> Any:
        """Internal helper to train a candidate model for the configured family."""
        if self.model_family == "classical":
            clf = ClassicalAdapterClassifier(
                input_dim=384,
                hidden_dim=64,
                num_classes=10,
                learning_rate=0.001,
                weight_decay=0.01,
                epochs=20,
                seed=self.random_seed,
                all_classes=CANONICAL_CLASSES,
            )
            clf.fit(X_train, y_train)
            return clf

        elif self.model_family == "fixed_quantum":
            clf = ContinualQSVCClassifier(
                n_qubits=8,
                seed=self.random_seed,
                C=1.0,
            )
            clf.fit(X_train, y_train)
            return clf

        elif self.model_family == "trainable_quantum":
            clf = ContinualTrainableQSVCClassifier(
                n_qubits=8,
                seed=self.random_seed,
                C=1.0,
                maxiter=5,
            )
            clf.fit(X_train, y_train)
            return clf

        elif self.model_family == "hybrid":
            trainer = ContinualHybridTrainer(
                config=CANONICAL_HYBRID_CONFIG,
                random_seed=self.random_seed,
            )
            res = trainer.fit(X_train, y_train)
            return res.classifier

        else:
            raise ValueError(f"Unknown model family: '{self.model_family}'")

    def compute_learning_efficiency_metrics(self) -> LearningEfficiencyMetrics:
        """Compute end-to-end learning efficiency and human review burden."""
        total_int = len(self.interaction_history)
        total_ver = sum(1 for r in self.interaction_history if r.verification_status == VerificationStatus.USER_VERIFIED.value)
        total_elig = sum(1 for r in self.interaction_history if r.learning_eligible)

        confirms = sum(1 for r in self.interaction_history if r.human_action == ReviewAction.ACCEPT.value)
        corrects = sum(1 for r in self.interaction_history if r.human_action == ReviewAction.CORRECT.value)
        rejects = sum(1 for r in self.interaction_history if r.human_action == ReviewAction.REJECT.value)

        burden_ratio = (confirms + corrects) / total_int if total_int > 0 else 0.0
        elig_frac = total_elig / total_ver if total_ver > 0 else 0.0

        n_cycles = len(self.cycle_history)
        n_prom = sum(1 for c in self.cycle_history if c.gate_decision == PromotionDecision.PROMOTE.value)
        n_rej = sum(1 for c in self.cycle_history if c.gate_decision == PromotionDecision.REJECT.value)
        n_hold = sum(1 for c in self.cycle_history if c.gate_decision == PromotionDecision.HOLD.value)

        prom_rate = n_prom / n_cycles if n_cycles > 0 else 0.0
        rej_rate = n_rej / n_cycles if n_cycles > 0 else 0.0

        # Compute metric gains across trajectory
        delta_f1 = 0.0
        delta_rec = 0.0
        delta_fer = 0.0
        if len(self.cycle_history) > 0:
            first_cycle = self.cycle_history[0]
            last_cycle = self.cycle_history[-1]
            if first_cycle.pre_cycle_metrics and last_cycle.post_cycle_metrics:
                f1_start = first_cycle.pre_cycle_metrics.get("macro_f1", 0.0)
                f1_end = last_cycle.post_cycle_metrics.get("macro_f1", 0.0)
                rec_start = first_cycle.pre_cycle_metrics.get("emergency_recall", 0.0)
                rec_end = last_cycle.post_cycle_metrics.get("emergency_recall", 0.0)
                fer_start = first_cycle.pre_cycle_metrics.get("false_emergency_rate", 0.0)
                fer_end = last_cycle.post_cycle_metrics.get("false_emergency_rate", 0.0)

                delta_f1 = (f1_end - f1_start) / max(total_ver, 1)
                delta_rec = (rec_end - rec_start) / max(total_ver, 1)
                delta_fer = (fer_end - fer_start) / max(total_ver, 1)

        return LearningEfficiencyMetrics(
            total_interactions=total_int,
            total_verified_experiences=total_ver,
            total_eligible_experiences=total_elig,
            eligible_fraction=elig_frac,
            human_confirmations=confirms,
            human_corrections=corrects,
            human_rejections=rejects,
            human_burden_ratio=burden_ratio,
            total_adaptation_cycles=n_cycles,
            promoted_cycles=n_prom,
            rejected_cycles=n_rej,
            held_cycles=n_hold,
            rollback_count=self.rollback_count,
            promotion_rate=prom_rate,
            rejection_rate=rej_rate,
            delta_macro_f1_per_verified_exp=delta_f1,
            delta_emergency_recall_per_verified_exp=delta_rec,
            delta_fer_per_verified_exp=delta_fer,
        )
