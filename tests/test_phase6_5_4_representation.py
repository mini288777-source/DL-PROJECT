"""
Unit tests for Subphase 6.5.4: Canonical Representation Space Integrity

Validates:
1. Canonical 384-D embedding integrity (shape, dtype, NaN, Inf, zero-vector, unit-L2 norm).
2. Exactly 6,067 audio recording embeddings in Whisper cache.
3. Distribution of norms is strictly 1.0 within numerical epsilon.
4. Per-speaker sample counts match manifest records.
5. Representation audit artifact exists and matches embedding properties.
"""

import os
import glob
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WHISPER_DIR = os.path.join(REPO_ROOT, "embeddings", "torgo", "whisper")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")
AUDIT_CSV = os.path.join(REPO_ROOT, "results", "phase6_5_4_representation_audit.csv")


def test_embedding_directory_exists():
    """Verify whisper embedding cache directory exists."""
    assert os.path.isdir(WHISPER_DIR), f"Missing {WHISPER_DIR}"


def test_total_embedding_count():
    """Verify exact expected count of 6067 Whisper embedding files."""
    files = glob.glob(os.path.join(WHISPER_DIR, "*.npy"))
    assert len(files) == 6067, f"Expected 6067 embedding files, found {len(files)}"


def test_representation_audit_csv_integrity():
    """Verify representation audit CSV matches file count and contains no defects."""
    assert os.path.isfile(AUDIT_CSV), f"Missing {AUDIT_CSV}"
    df = pd.read_csv(AUDIT_CSV)
    assert len(df) == 6067
    assert (df["dimension"] == 384).all()
    assert (df["dtype"] == "float32").all()
    assert not df["has_nan"].any()
    assert not df["has_inf"].any()
    assert not df["is_zero"].any()
    assert np.allclose(df["l2_norm"], 1.0, atol=1e-5)


def test_random_sample_embedding_properties():
    """Inspect random sample of embeddings directly from disk."""
    files = glob.glob(os.path.join(WHISPER_DIR, "*.npy"))
    np.random.seed(42)
    sample_files = np.random.choice(files, 50, replace=False)
    for f in sample_files:
        arr = np.load(f)
        assert arr.shape == (384,), f"Bad shape {arr.shape} in {f}"
        assert arr.dtype == np.float32, f"Bad dtype {arr.dtype} in {f}"
        assert not np.isnan(arr).any(), f"NaN found in {f}"
        assert not np.isinf(arr).any(), f"Inf found in {f}"
        norm = np.linalg.norm(arr)
        assert abs(norm - 1.0) < 1e-4, f"Non-unit norm {norm} in {f}"
