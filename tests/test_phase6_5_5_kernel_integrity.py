"""
Unit tests for Subphase 6.5.5: Quantum Kernel Integrity & PSD Diagnostics.
Validates:
1. Symmetry of all cached quantum training kernels (|K - K^T| < 1e-10).
2. Diagonal normalisation (|K_ii - 1.0| < 1e-10).
3. Positive Semi-Definiteness (minimum eigenvalue >= -1e-7).
4. Finite values (no NaNs, no Infs).
5. Value boundedness in [0.0, 1.0] for test transition matrices.
"""

import os
import glob
import json
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL_CACHE_DIR = os.path.join(REPO_ROOT, "results", "phase6_5_5", "kernel_cache")


def test_kernel_cache_exists():
    """Verify kernel cache directory exists and contains matrices."""
    assert os.path.exists(KERNEL_CACHE_DIR), "Kernel cache directory must exist"
    train_files = glob.glob(os.path.join(KERNEL_CACHE_DIR, "*_train.npy"))
    assert len(train_files) >= 6, f"Expected cached training kernels, found {len(train_files)}"


def test_training_kernel_symmetry_and_diagonal():
    """Verify sample training kernels satisfy exact mathematical symmetry and unit diagonal."""
    train_files = glob.glob(os.path.join(KERNEL_CACHE_DIR, "*_train.npy"))
    # Test across sampled matrices
    for f in train_files[:10]:
        K = np.load(f)
        assert K.ndim == 2
        assert K.shape[0] == K.shape[1], f"Kernel {f} must be square"
        sym_err = float(np.max(np.abs(K - K.T)))
        assert sym_err < 1e-10, f"Kernel {f} symmetry error {sym_err} exceeds tolerance"
        diag_err = float(np.max(np.abs(np.diag(K) - 1.0)))
        assert diag_err < 1e-10, f"Kernel {f} diagonal error {diag_err} exceeds tolerance"


def test_training_kernel_psd_property():
    """Verify eigenvalues of training kernels are non-negative (PSD)."""
    train_files = glob.glob(os.path.join(KERNEL_CACHE_DIR, "*_train.npy"))
    for f in train_files[:10]:
        K = np.load(f)
        eigs = np.linalg.eigvalsh(K)
        min_eig = float(np.min(eigs))
        assert min_eig >= -1e-7, f"Kernel {f} has negative eigenvalue {min_eig} (non-PSD)"


def test_test_kernel_bounds():
    """Verify test kernels have transition probabilities bounded in [0, 1]."""
    test_files = glob.glob(os.path.join(KERNEL_CACHE_DIR, "*_test.npy"))
    for f in test_files[:10]:
        K_test = np.load(f)
        assert np.all(np.isfinite(K_test)), f"Test kernel {f} contains NaN/Inf"
        assert np.all(K_test >= 0.0), f"Test kernel {f} contains negative entries"
        assert np.all(K_test <= 1.0 + 1e-7), f"Test kernel {f} contains entries > 1.0"
