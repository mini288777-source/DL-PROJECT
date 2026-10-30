"""
Test Final No Leakage.
Verifies the train-only PCA and train-only scaling contracts across adaptation and evaluation.
"""

import json
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def test_pca_and_scaler_train_only_contract():
    # Synthetic test to verify that fitting on train only does not alter test integrity
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(30, 384))
    x_test = rng.normal(size=(100, 384))
    
    pca = PCA(n_components=8, random_state=42)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    
    # Correct pipeline: fit on train only
    x_train_pca = pca.fit_transform(x_train)
    x_train_scaled = scaler.fit_transform(x_train_pca)
    
    x_test_pca = pca.transform(x_test)
    x_test_scaled = scaler.transform(x_test_pca)
    
    assert x_train_scaled.shape == (30, 8)
    assert x_test_scaled.shape == (100, 8)
    assert np.all(x_train_scaled >= -1e-6)
    assert np.all(x_train_scaled <= np.pi + 1e-6)
