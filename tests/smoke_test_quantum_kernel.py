#!/usr/bin/env python
"""
Smoke Test #2 -- Quantum Kernel

Proves that the quantum kernel pipeline can operate on small toy
feature vectors:
    toy features -> PCA -> ZZFeatureMap -> FidelityQuantumKernel -> SVM

Uses a statevector simulator only. Keeps qubit count and circuit
depth extremely small for fast execution.

This is NOT the actual experimental configuration.

Usage:
    python tests/smoke_test_quantum_kernel.py
"""

import sys
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.reproducibility import set_seed

# --- Configuration ---
SEED = 42
N_SAMPLES = 20          # tiny toy dataset
FEATURE_DIM_RAW = 16    # raw feature dimension
N_QUBITS = 2            # very small for speed
REPS = 1                # minimal circuit depth


def generate_toy_data(n_samples: int, feature_dim: int, seed: int):
    """Generate a simple linearly separable toy dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, feature_dim)).astype(np.float64)
    # Simple rule: label based on sign of first coordinate
    y = (X[:, 0] > 0).astype(int)
    return X, y


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST #2 -- QUANTUM KERNEL")
    print("=" * 60)

    # Reproducibility
    seed_config = set_seed(SEED)
    print(f"\nSeed: {seed_config['seed']}")

    # Step 1: Generate toy data
    print(f"\nGenerating toy dataset: {N_SAMPLES} samples, {FEATURE_DIM_RAW} features...")
    X, y = generate_toy_data(N_SAMPLES, FEATURE_DIM_RAW, SEED)
    print(f"  X shape: {X.shape}")
    print(f"  y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Step 2: PCA dimensionality reduction
    print(f"\nPCA: {FEATURE_DIM_RAW} -> {N_QUBITS} dimensions...")
    pca = PCA(n_components=N_QUBITS, random_state=SEED)
    X_pca = pca.fit_transform(X)
    print(f"  Reduced shape: {X_pca.shape}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")

    # Normalize to [0, pi] range for quantum feature map
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_pca)
    print(f"  Scaled range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")

    # Step 3: Build quantum feature map
    print(f"\nBuilding ZZFeatureMap: {N_QUBITS} qubits, {REPS} reps...")
    from qiskit.circuit.library import ZZFeatureMap
    feature_map = ZZFeatureMap(
        feature_dimension=N_QUBITS,
        reps=REPS,
        entanglement="linear",
    )
    print(f"  Circuit depth: {feature_map.decompose().depth()}")
    print(f"  Parameters: {feature_map.num_parameters}")

    # Step 4: Build quantum kernel
    print("\nBuilding FidelityQuantumKernel...")
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    t0 = time.time()
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    # Split into train/test
    split = int(0.7 * N_SAMPLES)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # Step 5: Compute kernel matrix
    print(f"\nComputing training kernel matrix ({split}x{split})...")
    t0 = time.time()
    K_train = qkernel.evaluate(X_train)
    kernel_time = time.time() - t0
    print(f"  Kernel matrix shape: {K_train.shape}")
    print(f"  Computation time: {kernel_time:.4f}s")

    # Step 6: Validate kernel matrix
    print("\nValidating kernel matrix...")
    assert K_train.shape == (split, split), f"Wrong shape: {K_train.shape}"
    print(f"  Shape: OK ({K_train.shape})")

    assert np.isfinite(K_train).all(), "Kernel matrix contains NaN/Inf!"
    print("  Finite check: OK")

    # Diagonal should be close to 1 (fidelity of state with itself)
    diag = np.diag(K_train)
    diag_err = np.abs(diag - 1.0).max()
    print(f"  Diagonal values: mean={diag.mean():.6f}, max_err_from_1={diag_err:.2e}")
    assert diag_err < 1e-4, f"Diagonal deviates from 1: max error = {diag_err}"
    print("  Diagonal: OK")

    # Kernel matrix should be symmetric
    sym_err = np.abs(K_train - K_train.T).max()
    print(f"  Symmetry error: {sym_err:.2e}")
    assert sym_err < 1e-6, f"Kernel matrix not symmetric: {sym_err}"
    print("  Symmetry: OK")

    # Values should be in [0, 1] for fidelity kernel
    print(f"  Value range: [{K_train.min():.6f}, {K_train.max():.6f}]")
    assert K_train.min() >= -1e-6, f"Negative kernel values: {K_train.min()}"
    assert K_train.max() <= 1.0 + 1e-6, f"Kernel values > 1: {K_train.max()}"
    print("  Value range: OK")

    # Step 7: Train SVM with precomputed kernel
    print("\nTraining SVM classifier with quantum kernel...")
    svm = SVC(kernel="precomputed", random_state=SEED)
    svm.fit(K_train, y_train)
    print("  SVM fit: OK")

    # Step 8: Predict on test set
    print(f"\nComputing test kernel matrix ({len(X_test)}x{split})...")
    K_test = qkernel.evaluate(X_test, X_train)
    print(f"  Test kernel shape: {K_test.shape}")

    y_pred = svm.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test predictions: {y_pred}")
    print(f"  Test labels:      {y_test}")
    print(f"  Accuracy: {acc:.2%}")

    # Note: accuracy on toy data is not meaningful for research.
    # We only verify the pipeline runs without errors.

    # Summary
    print("\n" + "=" * 60)
    print("QUANTUM KERNEL SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Toy samples:       {N_SAMPLES} (train={split}, test={N_SAMPLES-split})")
    print(f"  Raw feature dim:   {FEATURE_DIM_RAW}")
    print(f"  PCA components:    {N_QUBITS}")
    print(f"  Qubits:            {N_QUBITS}")
    print(f"  Feature map:       ZZFeatureMap (reps={REPS}, entanglement=linear)")
    print(f"  Circuit depth:     {feature_map.decompose().depth()}")
    print(f"  Kernel matrix:     {K_train.shape}")
    print(f"  Kernel time:       {kernel_time:.4f}s")
    print(f"  Classifier:        SVC (precomputed)")
    print(f"  Toy accuracy:      {acc:.2%} (not research-meaningful)")
    print()
    print("QUANTUM KERNEL SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
