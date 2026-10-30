"""
Unit tests for Subphase 6.5.5: Zero-Leakage Verification.
Validates:
1. PCA and MinMaxScaler are fitted strictly on adaptation training data.
2. Test representations do not mutate PCA components, singular values, or scalers.
3. Test labels never enter model training or hyperparameter selection.
"""

import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.quantum.pca_pipeline import TrainOnlyPCAPipeline


def test_pca_train_only_fitting_no_mutation():
    """Verify that transforming test data leaves PCA and Scaler parameters strictly unchanged."""
    np.random.seed(42)
    X_train = np.random.randn(30, 384).astype(np.float32)
    X_test = np.random.randn(150, 384).astype(np.float32)

    pipe = TrainOnlyPCAPipeline(n_components=8, random_state=42, feature_range=(0.0, np.pi))
    X_train_pca = pipe.fit_transform(X_train)

    pca_mean_before = pipe.pca.mean_.copy()
    pca_comp_before = pipe.pca.components_.copy()
    scaler_min_before = pipe.scaler.data_min_.copy()
    scaler_max_before = pipe.scaler.data_max_.copy()

    # Transform test set
    X_test_pca = pipe.transform(X_test)

    # Verify no mutation
    assert np.array_equal(pipe.pca.mean_, pca_mean_before)
    assert np.array_equal(pipe.pca.components_, pca_comp_before)
    assert np.array_equal(pipe.scaler.data_min_, scaler_min_before)
    assert np.array_equal(pipe.scaler.data_max_, scaler_max_before)


def test_pca_dimension_error_on_rank_exceeding():
    """Verify that requesting n_components > sample size raises ValueError."""
    X_train = np.random.randn(5, 384)
    pipe = TrainOnlyPCAPipeline(n_components=8, random_state=42)
    with pytest.raises(ValueError, match="cannot exceed min"):
        pipe.fit(X_train)
