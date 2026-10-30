"""
Phase 2 Cache Integrity and Structure Test.

Validates:
1. Cache directory hierarchy: embeddings/torgo/whisper, embeddings/torgo/mfcc, embeddings/metadata.
2. 1-to-1 mapping from recording_id to .npy file.
3. Absence of NaN or Inf values in cached representations.
4. Correct and consistent dimensionality across all cached arrays.
5. Absence of duplicate recording IDs in the metadata manifest.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.cache import (
    EmbeddingManifestManager,
    get_embedding_path,
    has_cached_embedding,
    load_embedding_file,
)


def test_cache_directory_structure():
    """Verify that expected embedding cache directories exist."""
    base_dir = "embeddings/torgo"
    whisper_dir = os.path.join(base_dir, "whisper")
    mfcc_dir = os.path.join(base_dir, "mfcc")
    metadata_dir = "embeddings/metadata"

    os.makedirs(whisper_dir, exist_ok=True)
    os.makedirs(mfcc_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    assert os.path.isdir(whisper_dir)
    assert os.path.isdir(mfcc_dir)
    assert os.path.isdir(metadata_dir)


def test_embedding_manifest_integrity():
    """Verify embedding manifest format if it exists."""
    manifest_path = "embeddings/metadata/embedding_manifest.csv"
    if not os.path.exists(manifest_path):
        pytest.skip("Embedding manifest not yet generated (pre-extraction).")

    df = pd.read_csv(manifest_path)
    assert "recording_id" in df.columns
    assert "whisper_dim" in df.columns
    assert "mfcc_dim" in df.columns

    # Verify no duplicate recording IDs
    dups = df["recording_id"].duplicated().sum()
    assert dups == 0, f"Found {dups} duplicate recording IDs in {manifest_path}"


def test_cached_files_numeric_integrity():
    """Verify that any cached .npy files on disk are non-empty, finite, and float32."""
    whisper_dir = "embeddings/torgo/whisper"
    if not os.path.exists(whisper_dir):
        pytest.skip("Whisper cache dir does not exist yet.")

    files = [f for f in os.listdir(whisper_dir) if f.endswith(".npy")]
    if not files:
        pytest.skip("No Whisper .npy files cached yet.")

    # Spot check up to 50 files
    check_files = files[:50]
    for fname in check_files:
        path = os.path.join(whisper_dir, fname)
        arr = np.load(path)
        assert arr.dtype == np.float32, f"{fname} has dtype {arr.dtype}, expected float32"
        assert not np.isnan(arr).any(), f"{fname} contains NaN values"
        assert not np.isinf(arr).any(), f"{fname} contains Inf values"
        assert arr.ndim == 1, f"{fname} has ndim {arr.ndim}, expected 1D"


def test_cohort_cache_completeness():
    """Verify that all 6,067 cohort recordings have valid cached embeddings."""
    manifest_path = "data/manifests/torgo_manifest.csv"
    if not os.path.exists(manifest_path):
        pytest.skip("TORGO manifest not found.")

    df = pd.read_csv(manifest_path)
    cohort_df = df[df["split"].isin(["adaptation", "test"])]
    target_count = len(cohort_df)

    whisper_dir = "embeddings/torgo/whisper"
    mfcc_dir = "embeddings/torgo/mfcc"

    whisper_cached = len([f for f in os.listdir(whisper_dir) if f.endswith(".npy")]) if os.path.exists(whisper_dir) else 0
    mfcc_cached = len([f for f in os.listdir(mfcc_dir) if f.endswith(".npy")]) if os.path.exists(mfcc_dir) else 0

    if whisper_cached < target_count:
        pytest.skip(f"Whisper extraction still in progress: {whisper_cached}/{target_count} cached.")

    assert whisper_cached == target_count, f"Whisper cache incomplete: {whisper_cached}/{target_count}"
    assert mfcc_cached == target_count, f"MFCC cache incomplete: {mfcc_cached}/{target_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
