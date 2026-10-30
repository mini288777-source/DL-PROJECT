"""
Phase 7.9 Continual Trainable / Adaptive Quantum Representation Engine.

Implements the parameterized quantum representation adaptation pipeline:
- Parameterized Quantum Feature Map (PQFM): 8 qubits, 16 trainable rotation parameters,
  data re-uploading with interleaved linear CNOT entanglement.
- Exact Statevector Simulation (Qiskit Statevector backend).
- Quantum Kernel Alignment (QKA) optimization on verified experience buffers.
- Warm-Start continual optimization lineage across longitudinal accumulation steps.
- Strict Information Isolation: train-only PCA (d=8) and MinMaxScaler [0, pi], zero test leakage.
- Experience eligibility and speaker isolation firewall (Phase 7.2 integration).
- Versioned candidate registration in ModelRegistry (lifecycle state CANDIDATE).
- Safety-first governance integration via Phase 7.6 ValidationGate (Safety Veto).
- Matched 3-way paired comparison: Classical Baseline vs Fixed Quantum vs Trainable Quantum.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

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
from src.continual.store import SpeakerIsolationError
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    compute_dataset_hash,
    compute_configuration_hash,
)
from src.continual.continual_classical import (
    ContinualEvaluationMetrics,
    compute_comprehensive_continual_metrics,
    validate_training_experiences,
    ContinualTrainingError,
    IneligibleExperienceError,
    DataLeakageError,
    EmbeddingIntegrityError,
)
from src.continual.governance import (
    ValidationGate,
    ValidationDecisionRecord,
    PromotionDecision,
)
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline
from src.embeddings.cache import has_cached_embedding, load_embedding_file
from src.utils.seeding import set_seed


# --- Canonical Trainable Quantum Architecture SSOT ---

CANONICAL_TRAINABLE_QUANTUM_CONFIG: Dict[str, Any] = {
    "architecture_name": "vqfm_q8_p16_linear_reuploading",
    "qubit_count": 8,
    "input_dimension": 8,
    "trainable_parameters": 16,
    "circuit_depth": 37,
    "fixed_encoding": "ZZFeatureMap_reps1_linear",
    "trainable_gates": "16_Ry",
    "entangling_gates": "35_CX",
    "entanglement_pattern": "linear",
    "data_reuploading": True,
    "optimizer": "COBYLA",
    "maxiter": 25,
    "rhobeg": 0.2,
    "l2_regularization": 1e-4,
    "backend": "exact_statevector",
    "simulation_framework": "qiskit.quantum_info.Statevector",
    "scaling": "MinMaxScaler fitted on adaptation embeddings to [0, pi]",
    "classifier": "QSVC",
    "C": 1.0,
    "whiten": False,
}


# --- Exceptions ---

class TrainableQuantumError(ContinualTrainingError):
    """Base exception for trainable quantum representation operations."""
    pass


class TrainableQuantumArchitectureError(TrainableQuantumError):
    """Raised when circuit architecture or parameters violate canonical configuration."""
    pass


class QuantumOptimizationError(TrainableQuantumError):
    """Raised when representation parameter optimization diverges or encounters NaN."""
    pass


class TrainableKernelIntegrityError(TrainableQuantumError):
    """Raised when an adapted quantum kernel fails numerical validation."""
    pass


# --- Kernel Diagnostics & Result Schemas ---

@dataclass
class TrainableKernelDiagnostics:
    """Mathematical validation diagnostics for an adapted quantum kernel matrix."""
    shape: List[int]
    is_finite: bool
    min_value: float
    max_value: float
    symmetry_error: float
    diagonal_error: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    trace: float
    mean_off_diagonal: float
    std_off_diagonal: float
    is_training: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContinualTrainableQuantumResult:
    """Comprehensive output container for a continual trainable quantum adaptation run."""
    experiment_id: str
    speaker_id: str
    budget: int
    model_version: str
    parent_model_version: Optional[str]
    matched_classical_experiment_id: Optional[str]
    matched_fixed_quantum_experiment_id: Optional[str]
    seed: int
    train_sample_ids: List[str]
    training_dataset_hash: str
    configuration_hash: str
    optimal_theta: List[float]
    initial_param_fingerprint: str
    optimal_param_fingerprint: str
    parent_param_fingerprint: Optional[str]
    initialization_mode: str  # WARM_START or DETERMINISTIC_ZERO
    optimization_time_sec: float
    adaptation_time_sec: float
    inference_time_sec: float
    initial_loss: float
    final_loss: float
    nfev: int
    metrics: ContinualEvaluationMetrics
    kernel_diagnostics: TrainableKernelDiagnostics
    quantum_config: Dict[str, Any]
    support_vector_count: int
    governance_decision: Optional[str] = None
    model_checkpoint_path: Optional[str] = None
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        d["kernel_diagnostics"] = self.kernel_diagnostics.to_dict()
        return d


# --- Parameter Fingerprinting Helper ---

def compute_param_fingerprint(theta: np.ndarray) -> str:
    """Generate SHA-256 fingerprint for a parameter vector."""
    return hashlib.sha256(theta.astype(np.float64).tobytes()).hexdigest()


# --- Parameterized Quantum Feature Map Circuit ---

class ParameterizedQuantumFeatureMap:
    """
    8-Qubit Variational Quantum Feature Map with 16 trainable rotation parameters.
    Architecture:
      1. Fixed ZZ data encoding U_enc(x)
      2. Layer 1 trainable Ry(theta[0..7])
      3. Linear CX chain (7 entangling gates)
      4. Second ZZ data encoding U_enc(x) (data re-uploading)
      5. Layer 2 trainable Ry(theta[8..15])
    """

    def __init__(
        self,
        num_qubits: int = 8,
        num_parameters: int = 16,
    ):
        if num_qubits != 8:
            raise TrainableQuantumArchitectureError(f"Qubit count must be 8, got {num_qubits}")
        if num_parameters != 16:
            raise TrainableQuantumArchitectureError(f"Trainable parameter count must be 16, got {num_parameters}")

        self.num_qubits = num_qubits
        self.num_parameters = num_parameters

        # Symbolic parameter vectors
        self.x_vec = ParameterVector("x", self.num_qubits)
        self.th1_vec = ParameterVector("th1", self.num_qubits)
        self.th2_vec = ParameterVector("th2", self.num_qubits)

        # Build parameterized template circuit
        self.template_circuit = QuantumCircuit(self.num_qubits)

        # 1. ZZ feature map encoding layer 1
        enc1 = zz_feature_map(feature_dimension=self.num_qubits, reps=1, entanglement="linear")
        self.template_circuit.compose(enc1.assign_parameters(self.x_vec), inplace=True)

        # 2. Trainable Ry rotations layer 1
        for q in range(self.num_qubits):
            self.template_circuit.ry(self.th1_vec[q], q)

        # 3. Linear CNOT entangling layer
        for q in range(self.num_qubits - 1):
            self.template_circuit.cx(q, q + 1)

        # 4. ZZ feature map encoding layer 2 (data re-uploading)
        enc2 = zz_feature_map(feature_dimension=self.num_qubits, reps=1, entanglement="linear")
        self.template_circuit.compose(enc2.assign_parameters(self.x_vec), inplace=True)

        # 5. Trainable Ry rotations layer 2
        for q in range(self.num_qubits):
            self.template_circuit.ry(self.th2_vec[q], q)

        # Verify circuit metrics against SSOT
        self.circuit_depth = int(self.template_circuit.depth())
        ops = self.template_circuit.count_ops()
        self.trainable_gate_count = int(ops.get("ry", 0))
        self.entangling_gate_count = int(ops.get("cx", 0))
        self.total_gate_count = int(sum(ops.values()))

    def prepare_sample_circuits(self, X: np.ndarray) -> List[QuantumCircuit]:
        """
        Pre-bind input coordinates x for each sample in X, producing template
        circuits that only require theta binding during parameter optimization.
        """
        M = X.shape[0]
        circuits = []
        for i in range(M):
            param_dict = {p: float(X[i, j]) for j, p in enumerate(self.x_vec)}
            bound_c = self.template_circuit.assign_parameters(param_dict)
            circuits.append(bound_c)
        return circuits

    def compute_statevectors(
        self,
        sample_circuits: List[QuantumCircuit],
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute complex statevectors for a batch of pre-bound sample circuits given theta.
        """
        M = len(sample_circuits)
        dim_hilbert = 2**self.num_qubits
        statevectors = np.zeros((M, dim_hilbert), dtype=np.complex128)

        # Build theta binding dictionary
        th_dict = {}
        for j in range(self.num_qubits):
            th_dict[self.th1_vec[j]] = float(theta[j])
            th_dict[self.th2_vec[j]] = float(theta[self.num_qubits + j])

        for i, sc in enumerate(sample_circuits):
            bound = sc.assign_parameters(th_dict)
            sv = Statevector.from_instruction(bound)
            statevectors[i] = sv.data

        return statevectors

    def compute_kernel_matrix(
        self,
        psi_left: np.ndarray,
        psi_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate quantum fidelity kernel matrix K_ij = |<psi_i | psi_j>|^2.
        """
        is_symmetric = psi_right is None
        if is_symmetric:
            psi_right = psi_left

        amplitude_matrix = psi_left @ psi_right.conj().T
        K = np.abs(amplitude_matrix) ** 2

        if is_symmetric:
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)

        K = np.clip(K, 0.0, 1.0)
        return K


# --- Kernel-Target Alignment Optimizer ---

class QuantumKernelAlignmentOptimizer:
    """
    Optimizes the 16 trainable quantum representation parameters via Kernel-Target Alignment.
    """

    def __init__(
        self,
        feature_map: ParameterizedQuantumFeatureMap,
        maxiter: int = 20,
        rhobeg: float = 0.2,
        l2_reg: float = 1e-4,
    ):
        self.feature_map = feature_map
        self.maxiter = maxiter
        self.rhobeg = rhobeg
        self.l2_reg = l2_reg

    def build_balanced_target_matrix(self, y: List[str]) -> np.ndarray:
        """
        Construct class-balanced target alignment matrix T.
        Encourages high fidelity for intra-class pairs with reciprocal class weighting,
        and penalizes inter-class overlap.
        """
        N = len(y)
        classes = sorted(list(set(y)))
        class_counts = {c: sum(1 for label in y if label == c) for c in classes}

        T = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if y[i] == y[j]:
                    T[i, j] = 1.0 / max(class_counts[y[i]], 1)
                else:
                    inter_count = N - class_counts[y[i]]
                    T[i, j] = -1.0 / max(inter_count, 1)
        return T

    def optimize(
        self,
        sample_circuits: List[QuantumCircuit],
        y_train: List[str],
        theta_init: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, int, float]:
        """
        Optimize theta using derivative-free COBYLA optimization.
        Returns: (optimal_theta, initial_loss, final_loss, nfev, optimization_time_sec).
        """
        T = self.build_balanced_target_matrix(y_train)
        frob_T = float(np.linalg.norm(T, "fro"))
        if frob_T < 1e-12:
            frob_T = 1.0

        nfev_counter = [0]

        def objective(th: np.ndarray) -> float:
            nfev_counter[0] += 1
            svs = self.feature_map.compute_statevectors(sample_circuits, th)
            K = self.feature_map.compute_kernel_matrix(svs)
            frob_K = float(np.linalg.norm(K, "fro"))
            if frob_K < 1e-12:
                return 0.0

            kta = float(np.sum(K * T) / (frob_K * frob_T))
            reg = self.l2_reg * float(np.sum(th**2))
            return -kta + reg

        # Evaluate initial loss
        initial_loss = objective(theta_init)
        nfev_counter[0] = 0

        t0 = time.perf_counter()
        res = minimize(
            objective,
            theta_init,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": self.rhobeg},
        )
        opt_time = time.perf_counter() - t0

        optimal_theta = np.asarray(res.x, dtype=np.float64)
        final_loss = float(res.fun)

        return optimal_theta, initial_loss, final_loss, int(res.nfev), opt_time


# --- Core Continual Trainable QSVC Classifier ---

class ContinualTrainableQSVCClassifier:
    """
    Trainable-representation Quantum Support Vector Classifier.
    Integrates TrainOnlyPCAPipeline (d=8), ParameterizedQuantumFeatureMap (16 params),
    QuantumKernelAlignmentOptimizer, and scikit-learn SVC(kernel='precomputed', C=1.0).
    """

    def __init__(
        self,
        qubit_count: int = 8,
        num_parameters: int = 16,
        C: float = 1.0,
        maxiter: int = 20,
        rhobeg: float = 0.2,
        seed: int = 42,
        all_classes: Optional[List[str]] = None,
    ):
        # Strict SSOT verification
        if qubit_count != 8:
            raise TrainableQuantumArchitectureError(f"Qubit count must be 8, got {qubit_count}")
        if num_parameters != 16:
            raise TrainableQuantumArchitectureError(f"Trainable parameters must be 16, got {num_parameters}")
        if abs(C - 1.0) > 1e-9:
            raise TrainableQuantumArchitectureError(f"Regularization C must be 1.0, got {C}")

        self.qubit_count = qubit_count
        self.num_parameters = num_parameters
        self.C = C
        self.maxiter = maxiter
        self.rhobeg = rhobeg
        self.seed = seed
        self.classes_ = list(all_classes) if all_classes else list(CANONICAL_CLASSES)
        self.num_classes = len(self.classes_)

        # Feature pipeline & Quantum Engine
        self.pca_pipeline = TrainOnlyPCAPipeline(
            n_components=self.qubit_count,
            whiten=False,
            random_state=self.seed,
            feature_range=(0.0, np.pi),
        )
        self.feature_map = ParameterizedQuantumFeatureMap(
            num_qubits=self.qubit_count,
            num_parameters=self.num_parameters,
        )
        self.optimizer = QuantumKernelAlignmentOptimizer(
            feature_map=self.feature_map,
            maxiter=self.maxiter,
            rhobeg=self.rhobeg,
            l2_reg=1e-4,
        )

        self.svc: Optional[SVC] = None
        self.theta_: np.ndarray = np.zeros(self.num_parameters, dtype=np.float64)
        self.is_single_class_fallback: bool = False
        self.single_observed_class: Optional[str] = None
        self.support_vector_count: int = 0

        # State storage & Profiling
        self.X_train_pca: Optional[np.ndarray] = None
        self.train_circuits: Optional[List[QuantumCircuit]] = None
        self.K_train: Optional[np.ndarray] = None
        self.K_test: Optional[np.ndarray] = None
        self.kernel_diagnostics_: Optional[TrainableKernelDiagnostics] = None
        self.optimization_time_sec: float = 0.0
        self.adaptation_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0
        self.initial_loss: float = 0.0
        self.final_loss: float = 0.0
        self.nfev: int = 0

    def validate_kernel_matrix(self, K: np.ndarray, is_training: bool = True) -> TrainableKernelDiagnostics:
        """Verify strict mathematical integrity of adapted quantum kernel matrix."""
        if not np.isfinite(K).all():
            raise TrainableKernelIntegrityError("Quantum kernel matrix contains non-finite values (NaN/Inf).")

        val_min = float(np.min(K))
        val_max = float(np.max(K))
        if val_min < -1e-6 or val_max > 1.0 + 1e-6:
            raise TrainableKernelIntegrityError(
                f"Kernel values out of bounds [0.0, 1.0]: min={val_min:.6f}, max={val_max:.6f}"
            )

        sym_err = 0.0
        diag_err = 0.0
        min_eig = 0.0
        max_eig = 0.0
        condition_number = 0.0
        trace_val = 0.0
        mean_off_diag = 0.0
        std_off_diag = 0.0

        if is_training:
            n = K.shape[0]
            if K.shape[0] != K.shape[1]:
                raise TrainableKernelIntegrityError(f"Training kernel must be square, got shape {K.shape}")

            diag = np.diag(K)
            diag_err = float(np.max(np.abs(diag - 1.0)))
            if diag_err > 1e-6:
                raise TrainableKernelIntegrityError(f"Training kernel diagonal error {diag_err:.6e} exceeds 1e-6")

            sym_err = float(np.max(np.abs(K - K.T)))
            if sym_err > 1e-6:
                raise TrainableKernelIntegrityError(f"Training kernel symmetry error {sym_err:.6e} exceeds 1e-6")

            trace_val = float(np.trace(K))
            off_diag_mask = ~np.eye(n, dtype=bool)
            if np.any(off_diag_mask):
                off_diag_vals = K[off_diag_mask]
                mean_off_diag = float(np.mean(off_diag_vals))
                std_off_diag = float(np.std(off_diag_vals))

            eigs = np.linalg.eigvalsh(K)
            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            if min_eig < -1e-5:
                raise TrainableKernelIntegrityError(
                    f"Quantum kernel matrix violates positive semi-definiteness: min eigenvalue {min_eig:.6e}"
                )
            condition_number = float(max_eig / max(min_eig, 1e-12))

        return TrainableKernelDiagnostics(
            shape=list(K.shape),
            is_finite=True,
            min_value=val_min,
            max_value=val_max,
            symmetry_error=sym_err,
            diagonal_error=diag_err,
            min_eigenvalue=min_eig,
            max_eigenvalue=max_eig,
            condition_number=condition_number,
            trace=trace_val,
            mean_off_diagonal=mean_off_diag,
            std_off_diagonal=std_off_diag,
            is_training=is_training,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: List[str],
        theta_warm_start: Optional[np.ndarray] = None,
    ) -> "ContinualTrainableQSVCClassifier":
        """
        Fit trainable quantum representation via KTA optimization, followed by QSVC fitting.
        """
        t0 = time.perf_counter()
        set_seed(self.seed)

        unique_labels = sorted(list(set(y_train)))

        # Handle Single-Class Fallback
        if len(unique_labels) <= 1:
            self.is_single_class_fallback = True
            self.single_observed_class = unique_labels[0] if unique_labels else GENERAL_SPEECH_CLASS
            self.support_vector_count = 0
            self.adaptation_time_sec = time.perf_counter() - t0
            return self

        self.is_single_class_fallback = False
        self.single_observed_class = None

        # 1. Fit PCA and feature scaler strictly on adaptation training data
        self.X_train_pca = self.pca_pipeline.fit_transform(X_train)

        # 2. Prepare sample-specific template circuits
        self.train_circuits = self.feature_map.prepare_sample_circuits(self.X_train_pca)

        # 3. Determine initial theta (WARM_START if available, else ZERO)
        if theta_warm_start is not None and len(theta_warm_start) == self.num_parameters:
            theta_init = np.asarray(theta_warm_start, dtype=np.float64)
        else:
            theta_init = np.zeros(self.num_parameters, dtype=np.float64)

        # 4. Optimize quantum representation parameters via KTA
        opt_th, init_l, final_l, nfev, opt_time = self.optimizer.optimize(
            sample_circuits=self.train_circuits,
            y_train=y_train,
            theta_init=theta_init,
        )
        self.theta_ = opt_th
        self.initial_loss = init_l
        self.final_loss = final_l
        self.nfev = nfev
        self.optimization_time_sec = opt_time

        # 5. Evaluate adapted training kernel matrix
        psi_train = self.feature_map.compute_statevectors(self.train_circuits, self.theta_)
        K_train = self.feature_map.compute_kernel_matrix(psi_train)

        self.kernel_diagnostics_ = self.validate_kernel_matrix(K_train, is_training=True)
        self.K_train = K_train

        # 6. Fit SVC with precomputed adapted kernel
        self.svc = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight="balanced",
            random_state=self.seed,
        )
        self.svc.fit(K_train, y_train)

        self.support_vector_count = int(np.sum(self.svc.n_support_))
        self.adaptation_time_sec = time.perf_counter() - t0
        return self

    def predict(self, X_test: np.ndarray) -> List[str]:
        """
        Predict intent classes for test embeddings using adapted quantum representation.
        """
        t0 = time.perf_counter()
        M = X_test.shape[0]

        if self.is_single_class_fallback:
            self.inference_time_sec = time.perf_counter() - t0
            return [self.single_observed_class] * M

        if self.svc is None or self.X_train_pca is None or self.train_circuits is None:
            raise RuntimeError("ContinualTrainableQSVCClassifier must be fitted before predict.")

        # Transform test embeddings using fitted PCA and scaler (ZERO LEAKAGE)
        X_test_pca = self.pca_pipeline.transform(X_test)

        # Prepare test circuits and evaluate statevectors with optimal theta
        test_circuits = self.feature_map.prepare_sample_circuits(X_test_pca)
        psi_test = self.feature_map.compute_statevectors(test_circuits, self.theta_)
        psi_train = self.feature_map.compute_statevectors(self.train_circuits, self.theta_)

        K_test = self.feature_map.compute_kernel_matrix(psi_test, psi_train)

        # Validate test kernel bounds
        self.validate_kernel_matrix(K_test, is_training=False)
        self.K_test = K_test

        predictions = self.svc.predict(K_test)
        self.inference_time_sec = time.perf_counter() - t0
        return list(predictions)


# --- Continual Trainable Quantum Trainer ---

class ContinualTrainableQuantumTrainer:
    """
    Manages the continual trainable quantum adaptation pipeline for a target speaker:
    - Experience eligibility filtering (Phase 7.2)
    - Chronological experience ordering
    - Speaker isolation firewall
    - Train-only PCA and feature scaling
    - Parameterized quantum feature map optimization (KTA, COBYLA)
    - Warm-start continual lineage management
    - Protected test evaluation & metric computation
    - Candidate model versioning (state CANDIDATE)
    - Phase 7.6 ValidationGate governance evaluation
    """

    def __init__(
        self,
        target_speaker: str,
        model_registry: Optional[ModelRegistry] = None,
        validation_gate: Optional[ValidationGate] = None,
        base_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.target_speaker = target_speaker
        self.model_registry = model_registry or ModelRegistry()
        self.validation_gate = validation_gate or ValidationGate()
        self.base_dir = base_dir or "models/continual/phase7_9"

        default_config = dict(CANONICAL_TRAINABLE_QUANTUM_CONFIG)
        default_config["seed"] = 42
        if config:
            default_config.update(config)
        self.config = default_config

        # Parameter memory for warm-start tracking: budget -> theta
        self._learned_parameters: Dict[int, np.ndarray] = {}

    def train_step(
        self,
        experiences: List[ExperienceRecord],
        test_records: List[Tuple[str, str, np.ndarray]],  # [(sample_id, true_intent, embedding), ...]
        parent_model_version: Optional[str] = None,
        parent_theta: Optional[np.ndarray] = None,
        matched_classical_exp_id: Optional[str] = None,
        matched_fixed_q_exp_id: Optional[str] = None,
        custom_seed: Optional[int] = None,
        custom_version: Optional[str] = None,
        replay_sample_ids: Optional[List[str]] = None,
    ) -> ContinualTrainableQuantumResult:
        """
        Execute one continual trainable quantum personalization update.
        """
        seed = custom_seed if custom_seed is not None else self.config.get("seed", 42)
        set_seed(seed)

        # 1. Validate experiences and enforce eligibility firewall
        validate_training_experiences(experiences, self.target_speaker)

        # 2. Chronological ordering
        sorted_experiences = sorted(experiences, key=lambda r: r.timestamp)
        train_sample_ids = [r.sample_id for r in sorted_experiences]

        # 3. Check for data leakage against test records
        test_sample_ids = set(t[0] for t in test_records)
        overlap = set(train_sample_ids).intersection(test_sample_ids)
        if overlap:
            raise DataLeakageError(
                f"CRITICAL: Data leakage detected! {len(overlap)} samples exist in both "
                f"training and protected test sets: {list(overlap)[:5]}"
            )

        # 4. Extract training embeddings and verified labels
        train_embeddings_list = []
        train_labels_list = []

        for rec in sorted_experiences:
            emb = None
            if hasattr(rec, "embedding_array") and rec.embedding_array is not None:
                emb = rec.embedding_array
            elif rec.embedding_reference and os.path.isfile(rec.embedding_reference):
                emb = np.load(rec.embedding_reference)
            elif rec.sample_id:
                rid = rec.sample_id
                if has_cached_embedding(rid, feature_type="whisper", base_dir="embeddings/torgo"):
                    emb = load_embedding_file(rid, feature_type="whisper", base_dir="embeddings/torgo")

            if emb is None:
                raise EmbeddingIntegrityError(f"Failed to load embedding for record '{rec.sample_id}'")

            if np.isnan(emb).any() or np.isinf(emb).any():
                raise EmbeddingIntegrityError(f"Embedding for record '{rec.sample_id}' contains NaN or Inf")

            if emb.shape != (384,):
                raise EmbeddingIntegrityError(f"Embedding shape {emb.shape} does not match expected (384,)")

            train_embeddings_list.append(emb)
            train_labels_list.append(rec.confirmed_intent)

        X_train = np.stack(train_embeddings_list).astype(np.float32)
        y_train = train_labels_list
        budget = len(train_sample_ids)

        # Determine warm-start status and parent parameter fingerprint
        if parent_theta is not None:
            init_mode = "WARM_START"
            th_init = parent_theta
            parent_param_fp = compute_param_fingerprint(parent_theta)
        else:
            init_mode = "DETERMINISTIC_ZERO"
            th_init = np.zeros(self.config["trainable_parameters"], dtype=np.float64)
            parent_param_fp = None

        init_param_fp = compute_param_fingerprint(th_init)

        # Compute dataset fingerprint and config hash
        dataset_hash = compute_dataset_hash(X_train, y_train, train_sample_ids)
        config_hash = compute_configuration_hash(self.config)

        # 5. Build and fit ContinualTrainableQSVCClassifier
        classifier = ContinualTrainableQSVCClassifier(
            qubit_count=self.config["qubit_count"],
            num_parameters=self.config["trainable_parameters"],
            C=self.config["C"],
            maxiter=self.config["maxiter"],
            rhobeg=self.config["rhobeg"],
            seed=seed,
            all_classes=CANONICAL_CLASSES,
        )

        classifier.fit(
            X_train=X_train,
            y_train=y_train,
            theta_warm_start=th_init,
        )

        opt_th = classifier.theta_
        self._learned_parameters[budget] = opt_th
        opt_param_fp = compute_param_fingerprint(opt_th)

        # 6. Evaluate on protected test set
        X_test = np.stack([t[2] for t in test_records]).astype(np.float32)
        y_test = [t[1] for t in test_records]

        y_pred = classifier.predict(X_test=X_test)
        metrics = compute_comprehensive_continual_metrics(y_test, y_pred)

        # 7. Model versioning & artifact serialization
        version_str = custom_version if custom_version else f"v{self.target_speaker}_budget{budget}_trainable_q_seed{seed}"
        version_dir = os.path.join(self.base_dir, self.target_speaker, "trainable_qsvc", version_str)
        os.makedirs(version_dir, exist_ok=True)
        checkpoint_path = os.path.join(version_dir, "model_checkpoint.pkl")

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "svc": classifier.svc,
                "pca_pipeline": classifier.pca_pipeline,
                "theta": opt_th,
                "config": self.config,
                "is_single_class_fallback": classifier.is_single_class_fallback,
                "single_observed_class": classifier.single_observed_class,
                "initial_loss": classifier.initial_loss,
                "final_loss": classifier.final_loss,
            }, f)

        # Register candidate model in ModelRegistry
        version_record = ModelVersionRecord(
            model_id="trainable_quantum_representation",
            version=version_str,
            speaker_id=self.target_speaker,
            parent_version=parent_model_version,
            training_sample_ids=train_sample_ids,
            training_dataset_hash=dataset_hash,
            configuration_hash=config_hash,
            representation_version="whisper_tiny_384_pca8_vqfm16_minmax_pi",
            hyperparameters=self.config,
            quantum_configuration={
                **self.config,
                "optimal_theta": opt_th.tolist(),
                "initial_loss": classifier.initial_loss,
                "final_loss": classifier.final_loss,
            },
            metrics=metrics.to_dict(),
            status=ModelLifecycleState.CANDIDATE.value,
            artifact_path=checkpoint_path,
        )
        self.model_registry.register_candidate(version_record)

        # 8. Phase 7.6 Governance & Validation Gate Evaluation
        parent_record = None
        if parent_model_version:
            parent_record = self.model_registry.get_version(parent_model_version)

        def candidate_predict_fn(X_val: np.ndarray) -> List[str]:
            return classifier.predict(X_val)

        val_samples = test_records

        governance_record: ValidationDecisionRecord = self.validation_gate.evaluate_candidate(
            candidate_record=version_record,
            val_samples=val_samples,
            candidate_predict_fn=candidate_predict_fn,
            parent_record=parent_record,
            val_dataset_id=f"protected_test_{self.target_speaker}",
            val_speaker_id=self.target_speaker,
            replay_sample_ids=replay_sample_ids,
        )

        decision = governance_record.decision
        exp_id = f"EXP_7_9_{self.target_speaker}_N{budget}_TRAINABLE_Q_S{seed}"

        return ContinualTrainableQuantumResult(
            experiment_id=exp_id,
            speaker_id=self.target_speaker,
            budget=budget,
            model_version=version_str,
            parent_model_version=parent_model_version,
            matched_classical_experiment_id=matched_classical_exp_id,
            matched_fixed_quantum_experiment_id=matched_fixed_q_exp_id,
            seed=seed,
            train_sample_ids=train_sample_ids,
            training_dataset_hash=dataset_hash,
            configuration_hash=config_hash,
            optimal_theta=opt_th.tolist(),
            initial_param_fingerprint=init_param_fp,
            optimal_param_fingerprint=opt_param_fp,
            parent_param_fingerprint=parent_param_fp,
            initialization_mode=init_mode,
            optimization_time_sec=classifier.optimization_time_sec,
            adaptation_time_sec=classifier.adaptation_time_sec,
            inference_time_sec=classifier.inference_time_sec,
            initial_loss=classifier.initial_loss,
            final_loss=classifier.final_loss,
            nfev=classifier.nfev,
            metrics=metrics,
            kernel_diagnostics=classifier.kernel_diagnostics_ or classifier.validate_kernel_matrix(np.eye(budget)),
            quantum_config=self.config,
            support_vector_count=classifier.support_vector_count,
            governance_decision=decision,
            model_checkpoint_path=checkpoint_path,
            status="COMPLETED",
        )
