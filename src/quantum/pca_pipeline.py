"""
Train-Only PCA and Feature Scaling Pipeline for Quantum Encoding.

Strict Information Isolation Rule:
PCA and feature scalers MUST be fitted exclusively on the adaptation
training samples (N in {20, 30, 40, 50}) for each speaker.
Test data must NEVER be accessed during fitting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class TrainOnlyPCAPipeline:
    """
    Dimensionality reduction and feature scaling pipeline fitted strictly on training data.
    Maps 384-dimensional Whisper embeddings to d-dimensional quantum input coordinates.
    """

    def __init__(
        self,
        n_components: int = 4,
        whiten: bool = False,
        random_state: int = 42,
        feature_range: Tuple[float, float] = (0.0, np.pi),
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.feature_range = feature_range

        self.pca: Optional[PCA] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.is_fitted: bool = False

        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_explained_variance_: float = 0.0

    def fit(self, X_train: np.ndarray) -> "TrainOnlyPCAPipeline":
        """
        Fit PCA and MinMaxScaler exclusively on adaptation training data.

        Args:
            X_train: Training embeddings of shape (N, D), typically (N, 384).

        Returns:
            self
        """
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_train.shape}")

        n_samples, n_features = X_train.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed min(n_samples, n_features) = "
                f"min({n_samples}, {n_features})"
            )

        # 1. Fit PCA on training data only (cast to float64 for quantum circuit precision)
        X_train_f64 = np.asarray(X_train, dtype=np.float64)
        self.pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        X_pca = self.pca.fit_transform(X_train_f64)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_explained_variance_ = float(np.sum(self.explained_variance_ratio_))

        # 2. Fit MinMaxScaler on training PCA coordinates only
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        self.scaler.fit(X_pca)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted PCA and scaler.

        Args:
            X: Embeddings of shape (M, D).

        Returns:
            Quantum input array of shape (M, n_components) within feature_range.
        """
        if not self.is_fitted or self.pca is None or self.scaler is None:
            raise RuntimeError("Pipeline must be fitted before transforming data.")

        X_f64 = np.asarray(X, dtype=np.float64)
        X_pca = self.pca.transform(X_f64)
        X_scaled = self.scaler.transform(X_pca)
        return X_scaled

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return transformed coordinates."""
        return self.fit(X_train).transform(X_train)

    def get_metadata(self) -> Dict[str, Any]:
        """Return diagnostic metadata regarding explained variance."""
        return {
            "n_components": self.n_components,
            "whiten": self.whiten,
            "random_state": self.random_state,
            "feature_range": list(self.feature_range),
            "explained_variance_ratio": (
                self.explained_variance_ratio_.tolist()
                if self.explained_variance_ratio_ is not None
                else []
            ),
            "cumulative_explained_variance": self.cumulative_explained_variance_,
        }
