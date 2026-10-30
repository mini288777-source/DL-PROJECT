"""
Reproducibility Acceptance Tests for Phase 4 Quantum Kernel.

Verifies:
1. Train-only PCA produces bit-for-bit identical coordinates under identical seeds.
2. ZZ feature map produces bit-for-bit identical statevectors.
3. Fidelity quantum kernel matrix is 100% deterministic.
4. QSVC predictions and computed intent metrics replicate bit-for-bit.
"""

import numpy as np
import pytest

from src.models.evaluation import compute_intent_metrics
from src.models.quantum_kernel import QSVCClassifier
from src.quantum.fidelity_kernel import QuantumFidelityKernelEngine
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline

ALL_CLASSES = [
    "SOS_HELP",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "LOCATION",
    "CRITICAL_RESPONSE",
    "GENERAL_SPEECH",
]


def test_pca_reproducibility():
    """Verify PCA and MinMaxScaler are deterministic given identical input and seed."""
    rng = np.random.default_rng(1234)
    X = rng.normal(size=(30, 384)).astype(np.float32)

    pipe1 = TrainOnlyPCAPipeline(n_components=4, random_state=42)
    X1 = pipe1.fit_transform(X)

    pipe2 = TrainOnlyPCAPipeline(n_components=4, random_state=42)
    X2 = pipe2.fit_transform(X)

    diff = float(np.max(np.abs(X1 - X2)))
    assert diff == 0.0, f"PCA transformation is not bit-for-bit deterministic: max diff = {diff}"


def test_kernel_evaluation_determinism():
    """Verify FidelityQuantumKernel matrix computation is bit-for-bit deterministic."""
    rng = np.random.default_rng(5678)
    X = rng.uniform(0.0, np.pi, size=(20, 4))

    engine1 = QuantumFidelityKernelEngine(num_qubits=4, reps=1, entanglement="linear")
    K1 = engine1.evaluate(X)

    engine2 = QuantumFidelityKernelEngine(num_qubits=4, reps=1, entanglement="linear")
    K2 = engine2.evaluate(X)

    diff = float(np.max(np.abs(K1 - K2)))
    assert diff == 0.0, f"Quantum kernel evaluation is not deterministic: max diff = {diff}"


def test_qsvc_end_to_end_reproducibility():
    """Verify end-to-end fit, predict, and metrics replicate identically."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(25, 384)).astype(np.float32)
    y_train = ["GENERAL_SPEECH"] * 21 + ["CRITICAL_RESPONSE"] * 4
    X_test = rng.normal(size=(15, 384)).astype(np.float32)
    y_test = ["GENERAL_SPEECH"] * 12 + ["CRITICAL_RESPONSE"] * 3

    clf1 = QSVCClassifier(qubit_count=4, C=1.0, seed=42, all_classes=ALL_CLASSES)
    clf1.fit(X_train, y_train)
    p1 = clf1.predict(X_test)
    m1 = compute_intent_metrics(y_test, p1, ALL_CLASSES)

    clf2 = QSVCClassifier(qubit_count=4, C=1.0, seed=42, all_classes=ALL_CLASSES)
    clf2.fit(X_train, y_train)
    p2 = clf2.predict(X_test)
    m2 = compute_intent_metrics(y_test, p2, ALL_CLASSES)

    assert p1 == p2, "Predictions differed across runs under identical seed"
    assert m1["accuracy"] == m2["accuracy"]
    assert m1["macro_f1"] == m2["macro_f1"]
    assert m1["weighted_f1"] == m2["weighted_f1"]
