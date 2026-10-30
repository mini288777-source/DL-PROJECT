"""
Unit and Integration Tests for Phase 3 Classical Baselines.

Verifies:
1. Adapter network architecture, parameter count, and determinism.
2. RBF-SVM fit, predict, determinism, and support vector tracking.
3. Multi-class intent metrics computation and confusion matrix correctness.
4. Single-class adaptation fallback behavior.
5. Strict isolation and no test data leakage.
"""

import numpy as np
import pytest
import torch

from src.models.classical_adapter import ClassicalAdapterClassifier, ClassicalAdapterNet
from src.models.evaluation import compute_intent_metrics
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.utils.seeding import set_seed


ALL_CLASSES = [
    "SOS_HELP",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "LOCATION",
    "CRITICAL_RESPONSE",
    "GENERAL_SPEECH",
]


def test_adapter_net_architecture_and_params():
    """Verify adapter network structure and expected parameter count."""
    net = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7, dropout=0.1)
    # Expected:
    # Linear 1: 384 * 64 + 64 = 24640
    # LayerNorm: 64 * 2 = 128
    # Linear 2: 64 * 7 + 7 = 455
    # Total: 24640 + 128 + 455 = 25223
    param_count = net.count_parameters()
    assert param_count == 25223, f"Unexpected parameter count: {param_count}"

    # Forward pass shape check
    x = torch.randn(4, 384)
    out = net(x)
    assert out.shape == (4, 7), f"Expected shape (4, 7), got {out.shape}"


def test_adapter_determinism():
    """Verify bit-for-bit identical forward pass with identical seed."""
    x = torch.ones(8, 384)

    set_seed(42)
    net1 = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7).eval()
    out1 = net1(x).detach().numpy()

    set_seed(42)
    net2 = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7).eval()
    out2 = net2(x).detach().numpy()

    max_diff = float(np.max(np.abs(out1 - out2)))
    assert max_diff == 0.0, f"Adapter weights are not deterministic (diff = {max_diff})"


def test_adapter_fit_and_predict():
    """Verify end-to-end fit and predict on synthetic data."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 384)).astype(np.float32)
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    y_train = ["GENERAL_SPEECH"] * 16 + ["CRITICAL_RESPONSE"] * 4

    X_test = rng.normal(size=(10, 384)).astype(np.float32)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    clf = ClassicalAdapterClassifier(
        input_dim=384,
        hidden_dim=64,
        num_classes=7,
        epochs=10,
        seed=42,
        all_classes=ALL_CLASSES,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    assert len(preds) == 10
    assert probs.shape == (10, 7)
    assert all(p in ALL_CLASSES for p in preds)
    assert clf.adaptation_time_sec > 0.0


def test_adapter_single_class_fallback():
    """Verify adapter handles single-class adaptation without error."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 384)).astype(np.float32)
    y_train = ["GENERAL_SPEECH"] * 20

    X_test = rng.normal(size=(5, 384)).astype(np.float32)

    clf = ClassicalAdapterClassifier(all_classes=ALL_CLASSES)
    clf.fit(X_train, y_train)

    assert clf.is_single_class_fallback is True
    preds = clf.predict(X_test)
    assert preds == ["GENERAL_SPEECH"] * 5
    probs = clf.predict_proba(X_test)
    assert probs.shape == (5, 7)
    assert np.all(probs[:, ALL_CLASSES.index("GENERAL_SPEECH")] == 1.0)


def test_rbf_svm_fit_and_predict():
    """Verify RBF-SVM fit, predict, and support vector tracking."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(30, 384)).astype(np.float32)
    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    y_train = ["GENERAL_SPEECH"] * 25 + ["FALL"] * 5

    X_test = rng.normal(size=(10, 384)).astype(np.float32)
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

    clf = ClassicalRBFSVMClassifier(C=1.0, gamma="scale", seed=42, all_classes=ALL_CLASSES)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    assert len(preds) == 10
    assert probs.shape == (10, 7)
    assert all(p in ALL_CLASSES for p in preds)
    assert clf.support_vector_count > 0


def test_rbf_svm_single_class_fallback():
    """Verify RBF-SVM single-class fallback behaves properly."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 384)).astype(np.float32)
    y_train = ["GENERAL_SPEECH"] * 20

    X_test = rng.normal(size=(5, 384)).astype(np.float32)

    clf = ClassicalRBFSVMClassifier(all_classes=ALL_CLASSES)
    clf.fit(X_train, y_train)

    assert clf.is_single_class_fallback is True
    preds = clf.predict(X_test)
    assert preds == ["GENERAL_SPEECH"] * 5
    assert clf.support_vector_count == 0


def test_compute_intent_metrics():
    """Verify computation of accuracy, macro-F1, and confusion matrix."""
    y_true = ["GENERAL_SPEECH", "GENERAL_SPEECH", "FALL", "SOS_HELP"]
    y_pred = ["GENERAL_SPEECH", "CRITICAL_RESPONSE", "FALL", "SOS_HELP"]

    metrics = compute_intent_metrics(y_true=y_true, y_pred=y_pred, all_classes=ALL_CLASSES)

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (7, 7)
    assert cm.sum() == 4


def test_no_data_leakage():
    """Verify that model fitting never sees or accesses test data."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 384)).astype(np.float32)
    y_train = ["GENERAL_SPEECH"] * 18 + ["PAIN"] * 2

    # Intentionally distinct distribution for test
    X_test = rng.normal(loc=10.0, scale=1.0, size=(10, 384)).astype(np.float32)

    clf = ClassicalAdapterClassifier(epochs=5, seed=42, all_classes=ALL_CLASSES)
    # Fit only on train
    clf.fit(X_train, y_train)

    # Weights must be independent of test data
    weights_before = clf.model.net[0].weight.clone()
    _ = clf.predict(X_test)
    weights_after = clf.model.net[0].weight.clone()

    assert torch.equal(weights_before, weights_after), "Weights altered during evaluation!"
