"""
Phase 4.5 Test: No Transformation / PCA / Scaling Leakage

Verifies:
1. TrainOnlyPCAPipeline fit() operates strictly on training samples.
2. transform(X_test) does NOT alter the pipeline's internal parameters (mean_, components_, min_, scale_).
3. Transforming test data with and without prior test exposure produces identical test coordinates.
4. Attempting transform() before fit() raises RuntimeError.
"""

import numpy as np
import pytest
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline


def test_pipeline_internal_parameters_frozen_during_test_transform():
    rng = np.random.default_rng(42)
    N_train = 30
    N_test = 100
    D = 384

    X_train = rng.standard_normal((N_train, D))
    X_test = rng.standard_normal((N_test, D))

    pipeline = TrainOnlyPCAPipeline(n_components=4, random_state=42)
    pipeline.fit(X_train)

    # Capture internal parameters of PCA and Scaler
    pca_mean_before = pipeline.pca.mean_.copy()
    pca_components_before = pipeline.pca.components_.copy()
    scaler_min_before = pipeline.scaler.min_.copy()
    scaler_scale_before = pipeline.scaler.scale_.copy()

    # Transform test set
    X_test_transformed = pipeline.transform(X_test)

    # Verify no parameters changed
    np.testing.assert_allclose(pipeline.pca.mean_, pca_mean_before, err_msg="PCA mean shifted after test transform!")
    np.testing.assert_allclose(pipeline.pca.components_, pca_components_before, err_msg="PCA components shifted after test transform!")
    np.testing.assert_allclose(pipeline.scaler.min_, scaler_min_before, err_msg="Scaler min_ shifted after test transform!")
    np.testing.assert_allclose(pipeline.scaler.scale_, scaler_scale_before, err_msg="Scaler scale_ shifted after test transform!")

    # Verify output shape and bounded range [0, pi] for training data
    X_train_transformed = pipeline.transform(X_train)
    assert X_train_transformed.shape == (N_train, 4)
    assert X_test_transformed.shape == (N_test, 4)
    assert np.all(X_train_transformed >= -1e-12)
    assert np.all(X_train_transformed <= np.pi + 1e-12)


def test_transform_determinism_and_isolation():
    rng = np.random.default_rng(123)
    X_train = rng.standard_normal((20, 384))
    X_test_1 = rng.standard_normal((50, 384))
    X_test_2 = rng.standard_normal((50, 384))

    p1 = TrainOnlyPCAPipeline(n_components=4, random_state=42).fit(X_train)
    p2 = TrainOnlyPCAPipeline(n_components=4, random_state=42).fit(X_train)

    # p1 transforms X_test_1 then X_test_2
    res1_t1 = p1.transform(X_test_1)
    res1_t2 = p1.transform(X_test_2)

    # p2 transforms X_test_2 directly
    res2_t2 = p2.transform(X_test_2)

    np.testing.assert_allclose(res1_t2, res2_t2, atol=1e-14, err_msg="Transforming earlier test data altered subsequent test transformations!")


def test_transform_without_fit_raises():
    pipeline = TrainOnlyPCAPipeline(n_components=4)
    X = np.ones((10, 384))
    with pytest.raises(RuntimeError):
        pipeline.transform(X)
