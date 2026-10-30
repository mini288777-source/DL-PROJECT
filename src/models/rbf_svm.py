"""
Classical RBF-Kernel SVM Baseline Model.

Phase 3 Baseline B: Support Vector Classifier with Radial Basis Function (RBF) kernel
operating on frozen 384-dimensional Whisper acoustic representations.
Supports strictly isolated preprocessing, support vector accounting, and deterministic execution.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ClassicalRBFSVMClassifier:
    """
    RBF-Kernel SVM classifier for few-shot acoustic intent classification.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        class_weight: Optional[str] = None,
        standardize: bool = False,
        probability: bool = False,
        seed: int = 42,
        all_classes: Optional[List[str]] = None,
    ):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.standardize = standardize
        self.probability = probability
        self.seed = seed

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
        self.scaler: Optional[StandardScaler] = None
        self.svc: Optional[SVC] = None
        self.adaptation_time_sec: float = 0.0
        self.inference_time_sec: float = 0.0
        self.is_single_class_fallback: bool = False
        self.single_observed_class: Optional[str] = None
        self.support_vector_count: int = 0

    def fit(self, X: np.ndarray, y: List[str]) -> "ClassicalRBFSVMClassifier":
        """
        Fit RBF-SVM on speaker adaptation embeddings and labels.

        Args:
            X: 2D numpy array of shape (N, input_dim).
            y: List of intent strings of length N.

        Returns:
            self
        """
        start_time = time.perf_counter()
        unique_classes = sorted(list(set(y)))

        # Handle edge-case: single observed class in few-shot budget
        if len(unique_classes) == 1:
            self.is_single_class_fallback = True
            self.single_observed_class = unique_classes[0]
            self.support_vector_count = 0
            self.adaptation_time_sec = time.perf_counter() - start_time
            return self

        self.is_single_class_fallback = False
        self.single_observed_class = None

        # Feature scaling fit ONLY on adaptation data
        X_train = X.copy()
        if self.standardize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        else:
            self.scaler = None

        svc_kwargs = {
            "C": self.C,
            "kernel": "rbf",
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "random_state": self.seed,
        }
        if self.probability:
            svc_kwargs["probability"] = True

        self.svc = SVC(**svc_kwargs)

        self.svc.fit(X_train, y)
        self.support_vector_count = int(np.sum(self.svc.n_support_))
        self.adaptation_time_sec = time.perf_counter() - start_time
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict intent class labels."""
        start_time = time.perf_counter()

        if self.is_single_class_fallback:
            preds = [self.single_observed_class] * len(X)
            self.inference_time_sec = time.perf_counter() - start_time
            return preds

        X_eval = X.copy()
        if self.standardize and self.scaler is not None:
            X_eval = self.scaler.transform(X_eval)

        raw_preds = self.svc.predict(X_eval)
        self.inference_time_sec = time.perf_counter() - start_time
        return [str(p) for p in raw_preds]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities across all global classes."""
        start_time = time.perf_counter()

        if self.is_single_class_fallback:
            probs = np.zeros((len(X), self.num_classes), dtype=np.float32)
            c_idx = self.classes_.index(self.single_observed_class)
            probs[:, c_idx] = 1.0
            self.inference_time_sec = time.perf_counter() - start_time
            return probs

        X_eval = X.copy()
        if self.standardize and self.scaler is not None:
            X_eval = self.scaler.transform(X_eval)

        full_probs = np.zeros((len(X), self.num_classes), dtype=np.float32)

        if self.probability and hasattr(self.svc, "predict_proba"):
            model_probs = self.svc.predict_proba(X_eval)
            model_classes = list(self.svc.classes_)
            for i, c in enumerate(model_classes):
                if c in self.classes_:
                    target_idx = self.classes_.index(c)
                    full_probs[:, target_idx] = model_probs[:, i]
        else:
            # Fall back to one-hot predictions
            preds = self.svc.predict(X_eval)
            for i, p in enumerate(preds):
                if p in self.classes_:
                    c_idx = self.classes_.index(p)
                    full_probs[i, c_idx] = 1.0

        self.inference_time_sec = time.perf_counter() - start_time
        return full_probs

    def get_model_info(self) -> Dict[str, Any]:
        """Return hyperparameters and training details."""
        return {
            "method": "rbf_svm",
            "kernel": "rbf",
            "C": self.C,
            "gamma": str(self.gamma),
            "class_weight": self.class_weight,
            "standardize": self.standardize,
            "support_vector_count": self.support_vector_count,
            "seed": self.seed,
            "is_single_class_fallback": self.is_single_class_fallback,
        }
