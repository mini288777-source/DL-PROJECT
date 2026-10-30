"""
Acceptance Test for Phase 4 Quantum Kernel Caching and Zero-Recomputation Reload.

Verifies:
1. Kernel matrix generation and atomic saving to disk.
2. Complete cache reload from disk.
3. Shape, finite, and value tolerance verification.
4. Metadata integrity.
5. QSVC execution directly from cached matrices without recomputing circuits.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.models.quantum_kernel import QSVCClassifier
from src.quantum.fidelity_kernel import (
    QuantumFidelityKernelEngine,
    load_cached_kernels,
    save_cached_kernels,
)

ALL_CLASSES = [
    "SOS_HELP",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "LOCATION",
    "CRITICAL_RESPONSE",
    "GENERAL_SPEECH",
]


def test_kernel_cache_persistence_and_reload():
    """Verify kernel matrix saving, reloading, tolerance, and QSVC fit from cache."""
    temp_dir = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(42)
        X_train = rng.uniform(0.0, np.pi, size=(20, 4))
        X_test = rng.uniform(0.0, np.pi, size=(10, 4))
        y_train = ["GENERAL_SPEECH"] * 16 + ["FALL"] * 4

        speaker_id = "TEST_SPK"
        sample_n = 20
        qubits = 4

        # 1. Generate kernels using engine
        engine = QuantumFidelityKernelEngine(num_qubits=qubits, reps=1, entanglement="linear")
        K_train = engine.evaluate(X_train)
        K_test = engine.evaluate(X_test, X_train)

        # 2. Save kernels
        meta = {"test_metric": 1.0, "qubits": qubits, "reps": 1}
        save_cached_kernels(
            temp_dir, speaker_id, sample_n, qubits, K_train, K_test, meta, reps=1
        )

        # 3. Reload kernels from disk
        loaded = load_cached_kernels(temp_dir, speaker_id, sample_n, qubits, reps=1)
        assert loaded is not None, "Failed to reload cached kernels from disk"
        K_train_loaded, K_test_loaded, meta_loaded = loaded

        # 4. Verify shapes and finite values
        assert K_train_loaded.shape == (20, 20)
        assert K_test_loaded.shape == (10, 20)
        assert np.isfinite(K_train_loaded).all()
        assert np.isfinite(K_test_loaded).all()

        # 5. Value comparison within tolerance
        assert np.allclose(K_train, K_train_loaded, atol=1e-12)
        assert np.allclose(K_test, K_test_loaded, atol=1e-12)

        # 6. Verify metadata
        assert meta_loaded["speaker_id"] == speaker_id
        assert meta_loaded["sample_n"] == sample_n
        assert meta_loaded["qubit_count"] == qubits

        # 7. Run QSVC with cache_dir enabled to ensure reloaded kernel is used directly
        clf = QSVCClassifier(
            qubit_count=qubits,
            cache_dir=temp_dir,
            all_classes=ALL_CLASSES,
        )
        # Mock raw embeddings of dimension 384
        X_raw_train = rng.normal(size=(20, 384)).astype(np.float32)
        X_raw_test = rng.normal(size=(10, 384)).astype(np.float32)

        clf.fit(X_raw_train, y_train, speaker_id=speaker_id, sample_n=sample_n)
        preds = clf.predict(X_raw_test, speaker_id=speaker_id, sample_n=sample_n)

        assert len(preds) == 10
        assert clf.K_train is not None
        assert clf.K_test is not None
        assert np.allclose(clf.K_train, K_train_loaded, atol=1e-12)
        assert np.allclose(clf.K_test, K_test_loaded, atol=1e-12)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
