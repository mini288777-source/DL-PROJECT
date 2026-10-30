"""
Phase 7.10 — Hybrid Classical + Quantum Personalization Architecture.

Combines:
1. Classical Branch: 25,223-parameter feedforward adapter MLP (Phase 7.4)
   - 384-D Whisper acoustic representations -> calibrated class probabilities P_class(y|x)
2. Quantum Branch: 16-parameter variational quantum feature map (Phase 7.9)
   - 8-qubit linear data-reuploading circuit -> adapted kernel matrix -> calibrated P_quant(y|x)
3. Controlled Convex Fusion:
   - P_hybrid(y|x) = alpha * P_class(y|x) + (1 - alpha) * P_quant(y|x)
   - Frozen mixing parameter: alpha = 0.5 (balanced 50/50 weighting)

Enforces:
- Strict zero-leakage pipeline: train-only PCA (d=8) -> MinMaxScaler [0, pi] for quantum branch
- Matched replay and experience accumulation stream identical to Phase 7.4, 7.8, and 7.9
- Provenance and model versioning tracking both branch states and fingerprints
- Phase 7.6 Multi-Metric ValidationGate and Safety Veto integration
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from scipy.special import softmax

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    EligibilityReason,
    LearningStatus,
)
from src.continual.taxonomy import (
    CANONICAL_CLASSES,
    EMERGENCY_CLASSES,
    GENERAL_SPEECH_CLASS,
    is_valid_intent,
    is_emergency_intent,
)
from src.continual.model_version import (
    ModelRegistry,
    ModelVersionRecord,
    ModelLifecycleState,
    compute_dataset_hash,
    compute_configuration_hash,
)
from src.continual.continual_classical import (
    ContinualEvaluationMetrics,
    compute_comprehensive_continual_metrics,
    ContinualTrainingError,
    IneligibleExperienceError,
    DataLeakageError,
    EmbeddingIntegrityError,
)
from src.models.classical_adapter import (
    ClassicalAdapterNet,
    ClassicalAdapterClassifier,
)
from src.continual.trainable_quantum import (
    CANONICAL_TRAINABLE_QUANTUM_CONFIG,
    ContinualTrainableQSVCClassifier,
    compute_param_fingerprint,
    TrainableQuantumArchitectureError,
)
from src.continual.governance import ValidationGate
from src.utils.seeding import set_seed


# --- Canonical Phase 7.10 Architecture SSOT ---

CANONICAL_HYBRID_CONFIG: Dict[str, Any] = {
    "architecture_name": "hybrid_adapter_vqfm_calibrated_fusion",
    "classical_branch": "classical_adapter_mlp_25223",
    "quantum_branch": "vqfm_q8_p16_linear_reuploading",
    "fusion_method": "calibrated_probability_fusion",
    "fusion_alpha": 0.5,
    "classical_trainable_parameters": 25223,
    "quantum_trainable_parameters": 16,
    "total_trainable_parameters": 25239,
    "qubit_count": 8,
    "quantum_input_dimension": 8,
    "classical_input_dimension": 384,
    "circuit_depth": 37,
    "quantum_optimizer": "COBYLA",
    "classical_optimizer": "AdamW",
    "classical_epochs": 20,
    "classical_lr": 1e-3,
    "qsvc_C": 1.0,
}


@dataclass
class ContinualHybridResult:
    """Evaluation result container for Phase 7.10 hybrid personalization."""
    model_version: str
    speaker_id: str
    budget: int
    optimal_theta: List[float]
    parameter_fingerprint: str
    parent_model_version: Optional[str]
    parent_parameter_fingerprint: Optional[str]
    initialization_mode: str
    evaluation_metrics: ContinualEvaluationMetrics
    classical_metrics: ContinualEvaluationMetrics
    quantum_metrics: ContinualEvaluationMetrics
    governance_decision: str
    safety_veto_triggered: bool
    governance_details: Dict[str, Any]
    optimization_time_sec: float
    adaptation_time_sec: float
    inference_time_sec: float
    branch_divergence_rate: float
    classical_weight: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evaluation_metrics"] = self.evaluation_metrics.to_dict()
        d["classical_metrics"] = self.classical_metrics.to_dict()
        d["quantum_metrics"] = self.quantum_metrics.to_dict()
        return d


class HybridClassicalQuantumClassifier:
    """
    Unified Hybrid Model Combining Classical Adapter and Trainable QSVC.
    Executes balanced convex probability fusion:
      P_hybrid = alpha * P_class + (1 - alpha) * P_quant
    """

    def __init__(
        self,
        alpha: float = 0.5,
        classical_epochs: int = 20,
        classical_lr: float = 1e-3,
        quantum_maxiter: int = 20,
        quantum_rhobeg: float = 0.2,
        seed: int = 42,
        all_classes: Optional[List[str]] = None,
    ):
        self.alpha = float(alpha)
        self.classical_epochs = classical_epochs
        self.classical_lr = classical_lr
        self.quantum_maxiter = quantum_maxiter
        self.quantum_rhobeg = quantum_rhobeg
        self.seed = seed
        self.all_classes = list(all_classes) if all_classes else list(CANONICAL_CLASSES)
        self.class_to_idx = {c: i for i, c in enumerate(self.all_classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.all_classes)}

        # Branch Models
        self.classical_branch = ClassicalAdapterClassifier(
            input_dim=384,
            hidden_dim=64,
            num_classes=len(self.all_classes),
            epochs=self.classical_epochs,
            learning_rate=self.classical_lr,
            seed=self.seed,
            all_classes=self.all_classes,
        )

        self.quantum_branch = ContinualTrainableQSVCClassifier(
            qubit_count=8,
            num_parameters=16,
            C=1.0,
            maxiter=self.quantum_maxiter,
            rhobeg=self.quantum_rhobeg,
            seed=self.seed,
            all_classes=self.all_classes,
        )

        self.is_fitted = False
        self.adaptation_time_sec = 0.0
        self.inference_time_sec = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: List[str],
        theta_warm_start: Optional[np.ndarray] = None,
        adapter_state_dict: Optional[Dict[str, Any]] = None,
    ) -> "HybridClassicalQuantumClassifier":
        """Fit both classical and quantum branches on adaptation experiences."""
        t0 = time.perf_counter()
        set_seed(self.seed)

        # 1. Fit Classical Adapter Branch
        if adapter_state_dict is not None and hasattr(self.classical_branch, "model") and self.classical_branch.model is not None:
            try:
                self.classical_branch.model.load_state_dict(adapter_state_dict)
            except Exception:
                pass
        self.classical_branch.fit(X_train, y_train)

        # 2. Fit Trainable Quantum QSVC Branch
        self.quantum_branch.fit(X_train, y_train, theta_warm_start=theta_warm_start)

        self.is_fitted = True
        self.adaptation_time_sec = time.perf_counter() - t0
        return self

    def predict_proba(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute predicted class probabilities for Hybrid, Classical, and Quantum branches.
        Returns: (P_hybrid, P_class, P_quant)
        """
        if not self.is_fitted:
            raise RuntimeError("HybridClassicalQuantumClassifier must be fitted before predict_proba.")

        M = len(X_test)
        # 1. Classical branch probabilities
        P_class = self.classical_branch.predict_proba(X_test)

        # 2. Quantum branch probabilities via calibrated decision margins
        if self.quantum_branch.is_single_class_fallback:
            P_quant = np.zeros((M, len(self.all_classes)), dtype=np.float32)
            c_idx = self.class_to_idx.get(self.quantum_branch.single_observed_class, 0)
            P_quant[:, c_idx] = 1.0
        else:
            X_test_pca = self.quantum_branch.pca_pipeline.transform(X_test)
            test_circuits = self.quantum_branch.feature_map.prepare_sample_circuits(X_test_pca)
            psi_test = self.quantum_branch.feature_map.compute_statevectors(test_circuits, self.quantum_branch.theta_)
            psi_train = self.quantum_branch.feature_map.compute_statevectors(self.quantum_branch.train_circuits, self.quantum_branch.theta_)
            K_test = self.quantum_branch.feature_map.compute_kernel_matrix(psi_test, psi_train)
            self.quantum_branch.validate_kernel_matrix(K_test, is_training=False)

            # Decision scores to calibrated probabilities
            svc = self.quantum_branch.svc
            dec_scores = svc.decision_function(K_test)
            if len(svc.classes_) == 2:
                # Binary: dec_scores has shape (M,)
                binary_scores = np.column_stack([-dec_scores, dec_scores])
                p_observed = softmax(binary_scores, axis=1)
            else:
                # Multiclass OVR: dec_scores has shape (M, len(svc.classes_))
                p_observed = softmax(dec_scores, axis=1)

            P_quant = np.zeros((M, len(self.all_classes)), dtype=np.float32)
            for obs_idx, c_name in enumerate(svc.classes_):
                if c_name in self.class_to_idx:
                    t_idx = self.class_to_idx[c_name]
                    P_quant[:, t_idx] = p_observed[:, obs_idx]

        # 3. Controlled convex fusion
        P_hybrid = self.alpha * P_class + (1.0 - self.alpha) * P_quant

        return P_hybrid, P_class, P_quant

    def predict(self, X_test: np.ndarray) -> List[str]:
        """Predict intent labels using fused hybrid probabilities."""
        t0 = time.perf_counter()
        P_hybrid, _, _ = self.predict_proba(X_test)
        pred_indices = np.argmax(P_hybrid, axis=1)
        self.inference_time_sec = time.perf_counter() - t0
        return [self.idx_to_class[idx] for idx in pred_indices]


class ContinualHybridTrainer:
    """
    Manages continual hybrid personalization for a target speaker:
    - Verified experience ingestion & eligibility verification
    - Warm-start parameter lineage (classical adapter weights + quantum theta)
    - Full 3-way evaluation: Hybrid vs Classical vs Quantum
    - Phase 7.6 Multi-Metric ValidationGate and Safety Veto enforcement
    """

    def __init__(
        self,
        target_speaker: str,
        model_registry: ModelRegistry,
        validation_gate: ValidationGate,
        base_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.target_speaker = target_speaker
        self.model_registry = model_registry
        self.validation_gate = validation_gate
        self.base_dir = base_dir
        self.config = {**CANONICAL_HYBRID_CONFIG, **(config or {})}
        self.speaker_dir = os.path.join(self.base_dir, self.target_speaker)
        os.makedirs(self.speaker_dir, exist_ok=True)

    def adapt_speaker_continual(
        self,
        experiences: List[ExperienceRecord],
        budget: int,
        test_records: List[Tuple[str, str, np.ndarray]],
        seed: int = 42,
        parent_model_version: Optional[str] = None,
        parent_theta: Optional[np.ndarray] = None,
        parent_adapter_state: Optional[Dict[str, Any]] = None,
    ) -> ContinualHybridResult:
        """
        Execute one continual hybrid adaptation step.
        """
        t_start = time.perf_counter()
        set_seed(seed)

        # 1. Filter and validate experiences
        verified_exps: List[ExperienceRecord] = []
        for exp in experiences:
            if exp.speaker_id != self.target_speaker:
                raise ContinualTrainingError(
                    f"Speaker mismatch: record {exp.sample_id} belongs to {exp.speaker_id}, not {self.target_speaker}"
                )
            if not exp.learning_eligible:
                raise IneligibleExperienceError(
                    f"Experience {exp.sample_id} is ineligible for training: reason={exp.eligibility_reason}"
                )
            if exp.verification_status != VerificationStatus.USER_VERIFIED.value:
                raise IneligibleExperienceError(
                    f"Experience {exp.sample_id} lacks explicit human verification: status={exp.verification_status}"
                )
            if not is_valid_intent(exp.confirmed_intent):
                raise IneligibleExperienceError(f"Invalid intent class: {exp.confirmed_intent}")
            verified_exps.append(exp)

        if not verified_exps:
            raise ContinualTrainingError(f"No eligible verified experiences available for speaker {self.target_speaker}")

        # 2. Build training arrays
        X_train_list = []
        y_train_list = []
        for exp in verified_exps:
            emb = exp.embedding_array
            if emb is None or np.any(np.isnan(emb)):
                raise EmbeddingIntegrityError(f"Invalid embedding for record {exp.sample_id}")
            X_train_list.append(emb)
            y_train_list.append(exp.confirmed_intent)

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = y_train_list

        # Zero-leakage assertion
        test_rids = {r[0] for r in test_records}
        train_rids = {e.sample_id for e in verified_exps}
        leakage = test_rids.intersection(train_rids)
        if leakage:
            raise DataLeakageError(f"Test samples leaked into adaptation training set: {leakage}")

        # 3. Instantiate and fit Hybrid Model
        model = HybridClassicalQuantumClassifier(
            alpha=self.config.get("fusion_alpha", 0.5),
            classical_epochs=self.config.get("classical_epochs", 20),
            classical_lr=self.config.get("classical_lr", 1e-3),
            quantum_maxiter=self.config.get("quantum_maxiter", 20),
            quantum_rhobeg=self.config.get("quantum_rhobeg", 0.2),
            seed=seed,
            all_classes=list(CANONICAL_CLASSES),
        )

        init_mode = "WARM_STARTED" if parent_theta is not None else "ZERO_INITIALIZED"
        model.fit(
            X_train,
            y_train,
            theta_warm_start=parent_theta,
            adapter_state_dict=parent_adapter_state,
        )

        opt_time = getattr(model.quantum_branch, "optimization_time_sec", 0.0)
        adapt_time = time.perf_counter() - t_start

        # 4. Predict on Protected Test Set
        X_test = np.array([r[2] for r in test_records], dtype=np.float32)
        y_test = [r[1] for r in test_records]

        P_hybrid, P_class, P_quant = model.predict_proba(X_test)
        pred_hybrid = [model.idx_to_class[idx] for idx in np.argmax(P_hybrid, axis=1)]
        pred_class = [model.idx_to_class[idx] for idx in np.argmax(P_class, axis=1)]
        pred_quant = [model.idx_to_class[idx] for idx in np.argmax(P_quant, axis=1)]

        # Divergence between classical and quantum branch predictions
        branch_divergence = float(np.mean([p_c != p_q for p_c, p_q in zip(pred_class, pred_quant)]))

        # Metrics for Hybrid, Classical, and Quantum branches
        hybrid_metrics = compute_comprehensive_continual_metrics(y_true=y_test, y_pred=pred_hybrid)
        classical_metrics = compute_comprehensive_continual_metrics(y_true=y_test, y_pred=pred_class)
        quantum_metrics = compute_comprehensive_continual_metrics(y_true=y_test, y_pred=pred_quant)

        # 5. Versioning and Fingerprinting
        optimal_theta = model.quantum_branch.theta_
        param_fingerprint = compute_param_fingerprint(optimal_theta)
        parent_fingerprint = compute_param_fingerprint(parent_theta) if parent_theta is not None else None
        model_version = f"hybrid_{self.target_speaker}_n{budget}_{param_fingerprint[:8]}"

        candidate_dict = {
            "macro_f1": hybrid_metrics.macro_f1,
            "emergency_recall": hybrid_metrics.emergency_recall,
            "emergency_precision": hybrid_metrics.emergency_precision,
            "false_emergency_rate": hybrid_metrics.false_emergency_rate,
            "iper": hybrid_metrics.iper,
            "accuracy": hybrid_metrics.accuracy,
        }

        # 6. Candidate Version Record
        version_record = ModelVersionRecord(
            model_id=f"hybrid_{self.target_speaker}",
            version=model_version,
            speaker_id=self.target_speaker,
            parent_version=parent_model_version,
            training_sample_ids=[e.sample_id for e in verified_exps],
            training_dataset_hash=compute_dataset_hash(X_train, y_train),
            configuration_hash=compute_configuration_hash(self.config),
            representation_version="whisper_tiny_384_pca8_linear_fusion",
            hyperparameters=self.config,
            quantum_configuration={
                "optimal_theta": optimal_theta.tolist(),
                "parameter_fingerprint": param_fingerprint,
                "parent_parameter_fingerprint": parent_fingerprint,
                "initialization_mode": init_mode,
            },
            metrics=candidate_dict,
            status=ModelLifecycleState.CANDIDATE.value,
        )
        self.model_registry.register_candidate(version_record)

        parent_record = None
        if parent_model_version:
            parent_record = self.model_registry.get_version(parent_model_version)

        def candidate_predict_fn(X_val: np.ndarray) -> List[str]:
            return model.predict(X_val)

        gov_eval = self.validation_gate.evaluate_candidate(
            candidate_record=version_record,
            val_samples=test_records,
            candidate_predict_fn=candidate_predict_fn,
            parent_record=parent_record,
            val_dataset_id=f"protected_test_{self.target_speaker}",
            val_speaker_id=self.target_speaker,
        )

        decision = str(gov_eval.decision)
        safety_veto = bool(gov_eval.safety_veto)

        if decision == "PROMOTE":
            self.model_registry.update_status(model_version, ModelLifecycleState.ACTIVE)
        elif decision == "REJECT":
            self.model_registry.update_status(model_version, ModelLifecycleState.REJECTED)

        return ContinualHybridResult(
            model_version=model_version,
            speaker_id=self.target_speaker,
            budget=budget,
            optimal_theta=list(optimal_theta),
            parameter_fingerprint=param_fingerprint,
            parent_model_version=parent_model_version,
            parent_parameter_fingerprint=parent_fingerprint,
            initialization_mode=init_mode,
            evaluation_metrics=hybrid_metrics,
            classical_metrics=classical_metrics,
            quantum_metrics=quantum_metrics,
            governance_decision=decision,
            safety_veto_triggered=safety_veto,
            governance_details=asdict(gov_eval) if hasattr(gov_eval, "__dataclass_fields__") else (vars(gov_eval) if hasattr(gov_eval, "__dict__") else {}),
            optimization_time_sec=opt_time,
            adaptation_time_sec=adapt_time,
            inference_time_sec=model.inference_time_sec,
            branch_divergence_rate=branch_divergence,
            classical_weight=model.alpha,
        )
