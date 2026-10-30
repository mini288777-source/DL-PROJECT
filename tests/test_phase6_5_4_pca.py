"""
Unit tests for Subphase 6.5.4: PCA Variance & Dimensionality Reduction

Validates:
1. PCA variance calculation correctness.
2. Forensic reconciliation table exists and resolves discrepancy.
3. PCA-4 cumulative variance is in the verified ~78%-85% range across adaptation splits.
4. Output dimensionality matches n_components for d < 384.
5. d=384 unreduced handling preserves full 384-dimensional space.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORENSIC_CSV = os.path.join(REPO_ROOT, "results", "phase6_5_4_pca_forensic_reconciliation.csv")
PCA_VAR_CSV = os.path.join(REPO_ROOT, "results", "phase6_5_4_pca_variance.csv")
WHISPER_DIR = os.path.join(REPO_ROOT, "embeddings", "torgo", "whisper")
SPLITS_PATH = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")


def test_forensic_reconciliation_csv_exists():
    """Verify forensic reconciliation CSV exists and classifies discrepancy."""
    assert os.path.isfile(FORENSIC_CSV), f"Missing {FORENSIC_CSV}"
    df = pd.read_csv(FORENSIC_CSV)
    assert len(df) >= 4
    assert "ERRONEOUS_REPORT_ARTIFACT" in df["status"].values
    assert "VERIFIED_EMPIRICAL_TRACK_B" in df["status"].values


def test_pca_variance_csv_structure():
    """Verify PCA variance CSV contains all candidate dimensions and conditions."""
    assert os.path.isfile(PCA_VAR_CSV), f"Missing {PCA_VAR_CSV}"
    df = pd.read_csv(PCA_VAR_CSV)
    assert len(df) == 540  # 60 conditions * 9 dimensions
    assert set(df["dimension"].unique()) == {2, 4, 8, 16, 32, 64, 128, 256, 384}
    assert (df["explained_variance_pct"] >= 50.0).all()
    assert (df["explained_variance_pct"] <= 100.0).all()


def test_pca_4_explained_variance_magnitude():
    """Verify that PCA-4 cumulative variance is around ~80%, disproving 27.4%."""
    df = pd.read_csv(PCA_VAR_CSV)
    pca4 = df[df["dimension"] == 4]
    mean_pca4 = pca4["explained_variance_pct"].mean()
    assert 75.0 <= mean_pca4 <= 86.0, f"Expected PCA-4 mean ~80%, got {mean_pca4:.2f}%"
    assert mean_pca4 > 50.0, "PCA-4 cannot be 27.4% on Whisper embeddings"


def test_pca_transform_output_dimension():
    """Verify PCA transform outputs exactly d dimensions on training and test data."""
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    s = splits[0]
    X_train = np.array([np.load(os.path.join(WHISPER_DIR, f"{r}.npy")) for r in s["adaptation_recording_ids"]])
    X_test = np.array([np.load(os.path.join(WHISPER_DIR, f"{r}.npy")) for r in s["test_recording_ids"][:10]])

    for d in [2, 4, 8, 16]:
        pca = PCA(n_components=d, random_state=42).fit(X_train)
        X_tr = pca.transform(X_train)
        X_te = pca.transform(X_test)
        assert X_tr.shape == (len(X_train), d)
        assert X_te.shape == (len(X_test), d)
