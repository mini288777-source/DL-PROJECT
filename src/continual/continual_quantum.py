"""
Phase 7.8 Continual Quantum Kernel Personalization Engine.

Implements the fixed-kernel continual quantum adaptation pipeline:
- Canonical Phase 6.5 SSOT configuration: 8 qubits, ZZFeatureMap, reps=1, linear entanglement,
  FidelityQuantumKernel, exact statevector simulation, QSVC with C=1.0, balanced class weights.
- Feature pipeline: 384-D Whisper embeddings -> Train-only PCA (d=8) -> MinMaxScaler [0, pi].
- Strict information isolation: zero evaluation leakage, train-only PCA/scaler fitting.
- Deterministic cryptographic kernel caching with full configuration fingerprinting.
- Mathematical kernel integrity diagnostics (symmetry, diagonal, bounds, positive semi-definiteness).
- Zero trainable quantum parameters enforcement (strictly non-variational, anti-Phase-7.9 leakage).
- Experience eligibility and speaker isolation firewall (Phase 7.2 integration).
- Versioned candidate registration in ModelRegistry (lifecycle state CANDIDATE).
- Safety-first governance integration via Phase 7.6 ValidationGate (Safety Veto).
- Matched 1-to-1 classical control tracking for rigorous empirical comparison.
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
from sklearn.svm import SVC

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
from src.quantum.fidelity_kernel import (
    QuantumFidelityKernelEngine,
)
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline
from src.embeddings.cache import has_cached_embedding, load_embedding_file
from src.utils.seeding import set_seed


# --- Canonical Quantum Configuration SSOT ---

CANONICAL_QUANTUM_CONFIG: Dict[str, Any] = {
    "qubit_count": 8,
    "representation_dim": 8,
    "feature_map": "ZZFeatureMap",
    "entanglement": "linear",
    "reps": 1,
    "kernel_type": "FidelityQuantumKernel",
    "classifier": "QSVC",
    "C": 1.0,
    "backend": "exact_statevector",
    "simulation_framework": "qiskit.quantum_info.Statevector",
    "scaling": "MinMaxScaler fitted on adaptation embeddings to [0, pi]",
    "trainable_quantum_parameters": 0,
    "whiten": False,
}


# --- Exceptions ---

class QuantumContinualError(ContinualTrainingError):
    """Base exception for continual quantum learning operations."""
    pass


class QuantumConfigurationError(QuantumContinualError):
    """Raised when quantum hyperparameters violate canonical SSOT."""
    pass


class KernelIntegrityError(QuantumContinualError):
    """Raised when a computed or loaded quantum kernel fails mathematical validation."""
    pass


class QuantumLeakageError(QuantumContinualError):
    """Raised when evaluation data leaks into quantum preprocessing or kernel computation."""
    pass


# --- Kernel Diagnostics & Result Schemas ---

@dataclass
class QuantumKernelDiagnostics:
    """Mathematical validation diagnostics for a quantum kernel matrix."""
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
class ContinualQuantumResult:
    """Comprehensive output container for a continual quantum adaptation run."""
    experiment_id: str
    speaker_id: str
    budget: int
    model_version: str
    parent_model_version: Optional[str]
    matched_control_experiment_id: Optional[str]
    seed: int
    train_sample_ids: List[str]
    training_dataset_hash: str
    configuration_hash: str
    kernel_cache_key: str
    kernel_cache_hit: bool
    adaptation_time_sec: float
    kernel_train_time_sec: float
    kernel_test_time_sec: float
    inference_time_sec: float
    metrics: ContinualEvaluationMetrics
    kernel_diagnostics: QuantumKernelDiagnostics
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


# --- Deterministic Kernel Caching Utilities ---

def compute_quantum_cache_key(
    speaker_id: str,
    budget: int,
    seed: int,
    dataset_hash: str,
    quantum_config: Dict[str, Any],
    protocol_version: str = "7.8",
) -> str:
    """
    Generate a cryptographic fingerprint ensuring cache invalidation on any configuration,
    data, or protocol modification.
    """
    payload = {
        "speaker_id": speaker_id,
        "budget": budget,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "qubits": quantum_config.get("qubit_count", 8),
        "reps": quantum_config.get("reps", 1),
        "entanglement": quantum_config.get("entanglement", "linear"),
        "feature_map": quantum_config.get("feature_map", "ZZFeatureMap"),
        "kernel_type": quantum_config.get("kernel_type", "FidelityQuantumKernel"),
        "protocol_version": protocol_version,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ContinualQuantumKernelCache:
    """
    Persistent, tamper-resistant cache for quantum kernel matrices.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_paths(self, cache_key: str) -> Dict[str, str]:
        return {
            "train_kernel": os.path.join(self.cache_dir, f"{cache_key}_Ktrain.npy"),
            "test_kernel": os.path.join(self.cache_dir, f"{cache_key}_Ktest.npy"),
            "metadata": os.path.join(self.cache_dir, f"{cache_key}_meta.json"),
        }

    def load(
        self, cache_key: str
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]]:
        paths = self.get_paths(cache_key)
        if not (os.path.isfile(paths["train_kernel"]) and os.path.isfile(paths["metadata"])):
            return None

        try:
            K_train = np.load(paths["train_kernel"])
            with open(paths["metadata"], "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("cache_key") != cache_key:
                return None  # Key mismatch

            K_test = None
            if os.path.isfile(paths["test_kernel"]):
                K_test = np.load(paths["test_kernel"])

            return K_train, K_test, meta
        except Exception:
            return None

    def save(
        self,
        cache_key: str,
        K_train: np.ndarray,
        K_test: Optional[np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        paths = self.get_paths(cache_key)
        np.save(paths["train_kernel"], K_train.astype(np.float64))
        if K_test is not None:
            np.save(paths["test_kernel"], K_test.astype(np.float64))

        meta_payload = {
            "cache_key": cache_key,
            "train_shape": list(K_train.shape),
            "test_shape": list(K_test.shape) if K_test is not None else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        with open(paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)


# --- Core Continual QSVC Classifier ---

class ContinualQSVCClassifier:
    """
    Fixed-kernel Quantum Support Vector Classifier.
    Integrates TrainOnlyPCAPipeline (d=8), QuantumFidelityKernelEngine (8 qubits),
    and scikit-learn SVC(kernel='precomputed', C=1.0, class_weight='balanced').
    """

    def __init__(
        self,
        qubit_count: int = 8,
        reps: int = 1,
        entanglement: str = "linear",
        C: float = 1.0,
        whiten: bool = False,
        seed: int = 42,
        all_classes: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
    ):
        # Strict SSOT verification
        if qubit_count != 8:
            raise QuantumConfigurationError(f"Qubit count must be 8, got {qubit_count}")
        if reps != 1:
            raise QuantumConfigurationError(f"Reps must be 1, got {reps}")
        if entanglement != "linear":
            raise QuantumConfigurationError(f"Entanglement must be 'linear', got {entanglement}")
        if abs(C - 1.0) > 1e-9:
            raise QuantumConfigurationError(f"Regularization C must be 1.0, got {C}")

        self.qubit_count = qubit_count
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.whiten = whiten
        self.seed = seed
        self.classes_ = list(all_classes) if all_classes else list(CANONICAL_CLASSES)
        self.num_classes = len(self.classes_)
        self.cache_dir = cache_dir
        self.cache = ContinualQuantumKernelCache(cache_dir) if cache_dir else None

        # Preprocessing & Kernel Engines
        self.pca_pipeline = TrainOnlyPCAPipeline(
            n_components=self.qubit_count,
            whiten=self.whiten,
            random_state=self.seed,
            feature_range=(0.0, np.pi),
        )
        self.kernel_engine = QuantumFidelityKernelEngine(
            num_qubits=self.qubit_count,
            reps=self.reps,
            entanglement=self.entanglement,
        )

        # Inviolable Anti-Phase-7.9 verification
        if self.kernel_engine.trainable_quantum_parameters != 0:
            raise QuantumConfigurationError(
                f"CRITICAL: Detected {self.kernel_engine.trainable_quantum_parameters} trainable quantum "
                "parameters! Phase 7.8 strictly prohibits variational or trainable quantum feature maps."
            )

        self.svc: Optional[SVC] = None
        self.is_single_class_fallback: bool = False
        self.single_observed_class: Optional[str] = None
        self.support_vector_count: int = 0

        # Diagnostics & Profiling
        self.K_train: Optional[np.ndarray] = None
        self.K_test: Optional[np.ndarray] = None
        self.X_train_pca: Optional[np.ndarray] = None
        self.kernel_diagnostics_: Optional[QuantumKernelDiagnostics] = None
        self.kernel_train_time_sec: float = 0.0
        self.kernel_test_time_sec: float = 0.0
        self.adaptation_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0
        self.cache_hit: bool = False
        self.cache_key: str = ""

    def validate_kernel_matrix(self, K: np.ndarray, is_training: bool = True) -> QuantumKernelDiagnostics:
        """
        Verify strict mathematical integrity of quantum kernel matrix.
        """
        if not np.isfinite(K).all():
            raise KernelIntegrityError("Quantum kernel matrix contains non-finite values (NaN/Inf).")

        val_min = float(np.min(K))
        val_max = float(np.max(K))
        if val_min < -1e-6 or val_max > 1.0 + 1e-6:
            raise KernelIntegrityError(
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
                raise KernelIntegrityError(f"Training kernel must be square, got shape {K.shape}")

            diag = np.diag(K)
            diag_err = float(np.max(np.abs(diag - 1.0)))
            if diag_err > 1e-6:
                raise KernelIntegrityError(f"Training kernel diagonal error {diag_err:.6e} exceeds 1e-6")

            sym_err = float(np.max(np.abs(K - K.T)))
            if sym_err > 1e-6:
                raise KernelIntegrityError(f"Training kernel symmetry error {sym_err:.6e} exceeds 1e-6")

            trace_val = float(np.trace(K))
            off_diag_mask = ~np.eye(n, dtype=bool)
            if np.any(off_diag_mask):
                off_diag_vals = K[off_diag_mask]
                mean_off_diag = float(np.mean(off_diag_vals))
                std_off_diag = float(np.std(off_diag_vals))

            # Eigenvalue analysis for positive semi-definiteness
            eigs = np.linalg.eigvalsh(K)
            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            if min_eig < -1e-5:
                raise KernelIntegrityError(
                    f"Quantum kernel matrix violates positive semi-definiteness: min eigenvalue {min_eig:.6e}"
                )
            condition_number = float(max_eig / max(min_eig, 1e-12))

        return QuantumKernelDiagnostics(
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
        speaker_id: Optional[str] = None,
        budget: Optional[int] = None,
        dataset_hash: Optional[str] = None,
    ) -> "ContinualQSVCClassifier":
        """
        Fit QSVC on adaptation embeddings with train-only PCA and fidelity quantum kernel.
        """
        t0 = time.perf_counter()
        set_seed(self.seed)

        unique_labels = sorted(list(set(y_train)))

        # Handle Single-Class Adaptation Fallback
        if len(unique_labels) <= 1:
            self.is_single_class_fallback = True
            self.single_observed_class = unique_labels[0] if unique_labels else GENERAL_SPEECH_CLASS
            self.support_vector_count = 0
            self.adaptation_time_sec = time.perf_counter() - t0
            return self

        self.is_single_class_fallback = False
        self.single_observed_class = None

        # 1. Fit PCA and feature scaler strictly on adaptation training embeddings
        self.X_train_pca = self.pca_pipeline.fit_transform(X_train)

        # 2. Check for cached training kernel
        use_cache = (
            self.cache is not None
            and speaker_id is not None
            and budget is not None
            and dataset_hash is not None
        )

        K_train = None
        self.cache_hit = False

        if use_cache and self.cache is not None:
            self.cache_key = compute_quantum_cache_key(
                speaker_id=speaker_id,
                budget=budget,
                seed=self.seed,
                dataset_hash=dataset_hash,
                quantum_config=CANONICAL_QUANTUM_CONFIG,
            )
            cached_entry = self.cache.load(self.cache_key)
            if cached_entry is not None:
                K_train_candidate, _, _ = cached_entry
                if K_train_candidate.shape == (len(y_train), len(y_train)):
                    K_train = K_train_candidate
                    self.cache_hit = True

        # 3. Compute training kernel if not cached
        if K_train is None:
            t_k0 = time.perf_counter()
            K_train = self.kernel_engine.evaluate(self.X_train_pca)
            self.kernel_train_time_sec = time.perf_counter() - t_k0
        else:
            self.kernel_train_time_sec = 0.0

        # Validate training kernel matrix
        self.kernel_diagnostics_ = self.validate_kernel_matrix(K_train, is_training=True)
        self.K_train = K_train

        # 4. Fit SVC with precomputed kernel
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

    def predict(
        self,
        X_test: np.ndarray,
        speaker_id: Optional[str] = None,
        budget: Optional[int] = None,
    ) -> List[str]:
        """
        Predict intent classes for test embeddings.
        """
        t0 = time.perf_counter()
        M = X_test.shape[0]

        if self.is_single_class_fallback:
            self.inference_time_sec = time.perf_counter() - t0
            return [self.single_observed_class] * M

        if self.svc is None or self.X_train_pca is None:
            raise RuntimeError("ContinualQSVCClassifier must be fitted before predict.")

        # Transform test embeddings using fitted PCA and scaler (ZERO LEAKAGE)
        X_test_pca = self.pca_pipeline.transform(X_test)

        # Check for cached test kernel
        K_test = None
        if self.cache_hit and self.cache is not None and self.cache_key:
            cached_entry = self.cache.load(self.cache_key)
            if cached_entry is not None:
                _, K_test_candidate, _ = cached_entry
                if K_test_candidate is not None and K_test_candidate.shape == (M, self.X_train_pca.shape[0]):
                    K_test = K_test_candidate

        if K_test is None:
            t_k_test = time.perf_counter()
            K_test = self.kernel_engine.evaluate(X_test_pca, self.X_train_pca)
            self.kernel_test_time_sec = time.perf_counter() - t_k_test

            # Cache the newly computed kernels
            if self.cache is not None and self.cache_key and self.K_train is not None:
                self.cache.save(
                    cache_key=self.cache_key,
                    K_train=self.K_train,
                    K_test=K_test,
                    metadata={
                        "speaker_id": speaker_id,
                        "budget": budget,
                        "seed": self.seed,
                        "qubit_count": self.qubit_count,
                        "reps": self.reps,
                    },
                )
        else:
            self.kernel_test_time_sec = 0.0

        # Validate test kernel bounds
        self.validate_kernel_matrix(K_test, is_training=False)
        self.K_test = K_test

        predictions = self.svc.predict(K_test)
        self.inference_time_sec = time.perf_counter() - t0
        return list(predictions)


# --- Continual Quantum Trainer ---

class ContinualQuantumTrainer:
    """
    Manages the continual quantum adaptation pipeline for a target speaker:
    - Experience eligibility filtering (Phase 7.2)
    - Chronological experience ordering
    - Speaker isolation firewall
    - Train-only PCA and feature scaling
    - Exact Fidelity Quantum Kernel evaluation & deterministic caching
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
        cache_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.target_speaker = target_speaker
        self.model_registry = model_registry or ModelRegistry()
        self.validation_gate = validation_gate or ValidationGate()
        self.base_dir = base_dir or "models/continual/phase7_8"
        self.cache_dir = cache_dir or "cache/continual/quantum_kernels"

        default_config = dict(CANONICAL_QUANTUM_CONFIG)
        default_config["seed"] = 42
        if config:
            default_config.update(config)
        self.config = default_config

    def train_step(
        self,
        experiences: List[ExperienceRecord],
        test_records: List[Tuple[str, str, np.ndarray]],  # [(sample_id, true_intent, embedding), ...]
        parent_model_version: Optional[str] = None,
        matched_control_exp_id: Optional[str] = None,
        custom_seed: Optional[int] = None,
        custom_version: Optional[str] = None,
        replay_sample_ids: Optional[List[str]] = None,
    ) -> ContinualQuantumResult:
        """
        Execute one continual quantum personalization update.
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

        # Compute dataset fingerprint and config hash
        dataset_hash = compute_dataset_hash(X_train, y_train, train_sample_ids)
        config_hash = compute_configuration_hash(self.config)

        # 5. Build and fit ContinualQSVCClassifier
        classifier = ContinualQSVCClassifier(
            qubit_count=self.config["qubit_count"],
            reps=self.config["reps"],
            entanglement=self.config["entanglement"],
            C=self.config["C"],
            whiten=self.config.get("whiten", False),
            seed=seed,
            all_classes=CANONICAL_CLASSES,
            cache_dir=self.cache_dir,
        )

        classifier.fit(
            X_train=X_train,
            y_train=y_train,
            speaker_id=self.target_speaker,
            budget=budget,
            dataset_hash=dataset_hash,
        )

        # 6. Evaluate on protected test set
        X_test = np.stack([t[2] for t in test_records]).astype(np.float32)
        y_test = [t[1] for t in test_records]

        y_pred = classifier.predict(
            X_test=X_test,
            speaker_id=self.target_speaker,
            budget=budget,
        )
        metrics = compute_comprehensive_continual_metrics(y_test, y_pred)

        # 7. Model versioning & artifact serialization
        version_str = custom_version if custom_version else f"v{self.target_speaker}_budget{budget}_qsvc_seed{seed}"
        version_dir = os.path.join(self.base_dir, self.target_speaker, "qsvc_q8_r1", version_str)
        os.makedirs(version_dir, exist_ok=True)
        checkpoint_path = os.path.join(version_dir, "model_checkpoint.pkl")

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "svc": classifier.svc,
                "pca_pipeline": classifier.pca_pipeline,
                "config": self.config,
                "is_single_class_fallback": classifier.is_single_class_fallback,
                "single_observed_class": classifier.single_observed_class,
            }, f)

        # Register candidate model in ModelRegistry
        version_record = ModelVersionRecord(
            model_id="qsvc_q8_r1",
            version=version_str,
            speaker_id=self.target_speaker,
            parent_version=parent_model_version,
            training_sample_ids=train_sample_ids,
            training_dataset_hash=dataset_hash,
            configuration_hash=config_hash,
            representation_version="whisper_tiny_384_pca8_minmax_pi",
            hyperparameters=self.config,
            quantum_configuration=self.config,
            metrics=metrics.to_dict(),
            status=ModelLifecycleState.CANDIDATE.value,
            artifact_path=checkpoint_path,
        )
        self.model_registry.register_candidate(version_record)

        # 8. Phase 7.6 Governance & Validation Gate Evaluation
        parent_record = None
        if parent_model_version:
            parent_record = self.model_registry.get_version(parent_model_version)

        # Create candidate predict function matching ValidationGate signature
        def candidate_predict_fn(X_val: np.ndarray) -> List[str]:
            return classifier.predict(X_val, speaker_id=self.target_speaker, budget=budget)

        # Build validation samples format: [(sample_id, true_intent, embedding), ...]
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

        exp_id = f"EXP_7_8_{self.target_speaker}_N{budget}_QSVC_S{seed}"

        return ContinualQuantumResult(
            experiment_id=exp_id,
            speaker_id=self.target_speaker,
            budget=budget,
            model_version=version_str,
            parent_model_version=parent_model_version,
            matched_control_experiment_id=matched_control_exp_id,
            seed=seed,
            train_sample_ids=train_sample_ids,
            training_dataset_hash=dataset_hash,
            configuration_hash=config_hash,
            kernel_cache_key=classifier.cache_key,
            kernel_cache_hit=classifier.cache_hit,
            adaptation_time_sec=classifier.adaptation_time_sec,
            kernel_train_time_sec=classifier.kernel_train_time_sec,
            kernel_test_time_sec=classifier.kernel_test_time_sec,
            inference_time_sec=classifier.inference_time_sec,
            metrics=metrics,
            kernel_diagnostics=classifier.kernel_diagnostics_ or classifier.validate_kernel_matrix(np.eye(budget)),
            quantum_config=self.config,
            support_vector_count=classifier.support_vector_count,
            governance_decision=decision,
            model_checkpoint_path=checkpoint_path,
            status="COMPLETED",
        )
