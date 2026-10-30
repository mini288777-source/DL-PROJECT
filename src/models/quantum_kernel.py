"""
Quantum Support Vector Classifier (QSVC) Model.

Phase 4 Primary Method:
384-D Whisper -> Train-Only PCA -> ZZ Feature Map -> Fidelity Quantum Kernel -> QSVC.
Provides deterministic single-class fallback, caching integration,
support vector tracking, and runtime accounting.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.svm import SVC

from src.quantum.fidelity_kernel import (
    QuantumFidelityKernelEngine,
    load_cached_kernels,
    save_cached_kernels,
)
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline


class QSVCClassifier:
    """
    Quantum Support Vector Classifier using a Fidelity Quantum Kernel
    derived from a ZZ Pauli feature map.
    """

    def __init__(
        self,
        qubit_count: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        C: float = 1.0,
        whiten: bool = False,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        all_classes: Optional[List[str]] = None,
    ):
        self.qubit_count = qubit_count
        self.reps = reps
        self.entanglement = entanglement
        self.C = C
        self.whiten = whiten
        self.seed = seed
        self.cache_dir = cache_dir

        if all_classes is None:
            self.classes_ = [
                "SOS_HELP",
                "FALL",
                "BREATHING_MEDICAL",
                "PAIN",
                "LOCATION",
                "CRITICAL_RESPONSE",
                "GENERAL_SPEECH",
            ]
        else:
            self.classes_ = list(all_classes)

        self.num_classes = len(self.classes_)

        # Pipelines and models
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
        self.svc: Optional[SVC] = None

        # State and accounting
        self.is_single_class_fallback: bool = False
        self.single_observed_class: Optional[str] = None
        self.support_vector_count: int = 0
        self.adaptation_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0
        self.kernel_train_time_sec: float = 0.0
        self.kernel_test_time_sec: float = 0.0

        # Cached kernel storage
        self.K_train: Optional[np.ndarray] = None
        self.K_test: Optional[np.ndarray] = None
        self.X_train_pca: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: List[str],
        speaker_id: Optional[str] = None,
        sample_n: Optional[int] = None,
    ) -> "QSVCClassifier":
        """
        Fit QSVC on adaptation embeddings.

        Args:
            X_train: Embeddings array of shape (N, 384).
            y_train: Target intent labels of length N.
            speaker_id: Optional speaker identifier for persistent cache.
            sample_n: Optional sample budget N for persistent cache.

        Returns:
            self
        """
        t_start = time.time()
        unique_labels = sorted(list(set(y_train)))

        # Handle Single-Class Adaptation Fallback
        if len(unique_labels) <= 1:
            self.is_single_class_fallback = True
            self.single_observed_class = unique_labels[0] if unique_labels else "GENERAL_SPEECH"
            self.support_vector_count = 0
            self.adaptation_time_sec = time.time() - t_start
            return self

        self.is_single_class_fallback = False
        self.single_observed_class = None

        # 1. Fit PCA and feature scaler strictly on adaptation data
        self.X_train_pca = self.pca_pipeline.fit_transform(X_train)

        # 2. Check for persistent cached training kernel
        use_cache = (
            self.cache_dir is not None
            and speaker_id is not None
            and sample_n is not None
        )

        K_train = None
        if use_cache:
            loaded = load_cached_kernels(
                self.cache_dir, speaker_id, sample_n, self.qubit_count, self.reps
            )
            if loaded is not None:
                K_train, _, _ = loaded
                self.K_train = K_train

        # 3. Compute training kernel if not cached
        if K_train is None:
            t_k0 = time.time()
            K_train = self.kernel_engine.evaluate(self.X_train_pca)
            self.kernel_train_time_sec = time.time() - t_k0
            self.K_train = K_train

        # 4. Fit SVC with precomputed kernel
        self.svc = SVC(
            kernel="precomputed",
            C=self.C,
            random_state=self.seed,
        )
        self.svc.fit(K_train, y_train)

        self.support_vector_count = int(np.sum(self.svc.n_support_))
        self.adaptation_time_sec = time.time() - t_start
        return self

    def predict(
        self,
        X_test: np.ndarray,
        speaker_id: Optional[str] = None,
        sample_n: Optional[int] = None,
    ) -> List[str]:
        """
        Predict intent classes for test embeddings.

        Args:
            X_test: Test embeddings array of shape (M, 384).
            speaker_id: Optional speaker identifier for cache lookup.
            sample_n: Optional sample budget N for cache lookup.

        Returns:
            List of predicted intent strings.
        """
        t_start = time.time()
        M = X_test.shape[0]

        # Single-class fallback prediction
        if self.is_single_class_fallback:
            self.inference_time_sec = time.time() - t_start
            return [self.single_observed_class] * M

        if self.svc is None or self.X_train_pca is None:
            raise RuntimeError("QSVCClassifier must be fitted before predict.")

        # Check for cached test kernel
        use_cache = (
            self.cache_dir is not None
            and speaker_id is not None
            and sample_n is not None
        )

        K_test = None
        if use_cache:
            loaded = load_cached_kernels(
                self.cache_dir, speaker_id, sample_n, self.qubit_count, self.reps
            )
            if loaded is not None:
                _, K_test, _ = loaded
                self.K_test = K_test

        if K_test is None:
            # Transform test data using fitted PCA
            X_test_pca = self.pca_pipeline.transform(X_test)

            t_k0 = time.time()
            K_test = self.kernel_engine.evaluate(X_test_pca, self.X_train_pca)
            self.kernel_test_time_sec = time.time() - t_k0
            self.K_test = K_test

            # Save to cache if cache configured
            if use_cache and self.K_train is not None:
                meta = {
                    "pca_explained_variance_ratio": self.pca_pipeline.explained_variance_ratio_.tolist(),
                    "pca_cumulative_explained_variance": self.pca_pipeline.cumulative_explained_variance_,
                    "support_vector_count": self.support_vector_count,
                    "seed": self.seed,
                    "C": self.C,
                }
                save_cached_kernels(
                    self.cache_dir,
                    speaker_id,
                    sample_n,
                    self.qubit_count,
                    self.K_train,
                    self.K_test,
                    meta,
                    reps=self.reps,
                )

        preds = self.svc.predict(K_test)
        self.inference_time_sec = time.time() - t_start
        return list(preds)

    def get_circuit_properties(self) -> Dict[str, Any]:
        """Return circuit complexity and capacity accounting."""
        props = self.kernel_engine.get_circuit_properties()
        props["C"] = self.C
        props["support_vector_count"] = self.support_vector_count
        props["pca_explained_variance_ratio"] = (
            self.pca_pipeline.explained_variance_ratio_.tolist()
            if self.pca_pipeline.explained_variance_ratio_ is not None
            else []
        )
        props["pca_cumulative_explained_variance"] = (
            self.pca_pipeline.cumulative_explained_variance_
        )
        return props
