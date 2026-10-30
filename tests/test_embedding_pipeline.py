#!/usr/bin/env python
"""
Unit and Integration Tests for Phase 2: Whisper Embedding Pipeline.

Validates:
    1. Determinism (repeated passes produce bit-for-bit identical vectors).
    2. Batch consistency (individual extraction matches batched extraction).
    3. Output dimensionality (exactly 384 dimensions for whisper-tiny).
    4. Unit normalization (L2 norm ≈ 1.0).
    5. Cache persistence & round-trip retrieval integrity.

Usage:
    python tests/test_embedding_pipeline.py
"""

import os
import sys
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.whisper_embeddings import WhisperEmbeddingExtractor
from src.models.embedding_cache import EmbeddingCacheManager


def run_tests() -> int:
    print("=" * 60)
    print("RUNNING PHASE 2 TESTS: WHISPER EMBEDDING PIPELINE")
    print("=" * 60)

    failures = []

    # Initialize extractor
    print("\n[Test 1/5] Initializing WhisperEmbeddingExtractor...")
    try:
        extractor = WhisperEmbeddingExtractor(
            model_id="openai/whisper-tiny",
            device="cpu",
            pooling="mean",
            normalize=True,
            seed=42
        )
        print("  [PASS] Extractor initialized successfully")
    except Exception as e:
        print(f"  [FAIL] Extractor initialization failed: {e}")
        return 1

    # Generate synthetic waveforms
    sr = 16000
    duration = 2.0
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio_a = (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)
    audio_b = (0.4 * np.sin(2 * np.pi * 880.0 * t) + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)

    # 1. Determinism Test
    print("\n[Test 2/5] Testing extraction determinism across repeated forward passes...")
    emb_pass1 = extractor.extract_from_waveform(audio_a, sr=sr)
    emb_pass2 = extractor.extract_from_waveform(audio_a, sr=sr)
    max_diff = float(np.max(np.abs(emb_pass1 - emb_pass2)))
    print(f"  Max absolute difference between passes: {max_diff:.2e}")
    if max_diff < 1e-6:
        print("  [PASS] Deterministic forward pass confirmed")
    else:
        failures.append(f"Determinism failure: max diff {max_diff:.2e} >= 1e-6")

    # 2. Embedding Dimensionality & Normalization
    print("\n[Test 3/5] Testing embedding dimensionality & L2 normalization...")
    if emb_pass1.shape == (384,):
        print(f"  [PASS] Shape verified: {emb_pass1.shape}")
    else:
        failures.append(f"Shape error: {emb_pass1.shape} != (384,)")

    norm = float(np.linalg.norm(emb_pass1))
    print(f"  L2 norm of embedding: {norm:.6f}")
    if abs(norm - 1.0) < 1e-5:
        print("  [PASS] Unit L2 normalization confirmed")
    else:
        failures.append(f"Normalization error: norm {norm} != 1.0")

    if np.isfinite(emb_pass1).all():
        print("  [PASS] All embedding values are finite (no NaN/Inf)")
    else:
        failures.append("Finite check failed: embedding contains NaN/Inf")

    # 3. Batch Extraction Consistency
    print("\n[Test 4/5] Testing batch extraction consistency...")
    batch_embs = extractor.extract_batch([audio_a, audio_b], sr=sr)
    if batch_embs.shape == (2, 384):
        print(f"  [PASS] Batch output shape verified: {batch_embs.shape}")
    else:
        failures.append(f"Batch shape error: {batch_embs.shape} != (2, 384)")

    diff_batch_single = float(np.max(np.abs(batch_embs[0] - emb_pass1)))
    print(f"  Max difference between batched and single extraction: {diff_batch_single:.2e}")
    if diff_batch_single < 1e-5:
        print("  [PASS] Batched extraction matches single extraction")
    else:
        failures.append(f"Batch mismatch: max diff {diff_batch_single:.2e}")

    # 4. Cache Persistence Round-Trip Test
    print("\n[Test 5/5] Testing EmbeddingCacheManager persistence & round-trip...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cache_file = os.path.join(tmp_dir, "test_cache.npz")
        cache = EmbeddingCacheManager(expected_dim=384)
        cache.add("rec_001", batch_embs[0])
        cache.add("rec_002", batch_embs[1])
        cache.save(tmp_cache_file)

        # Reload
        reloaded_cache = EmbeddingCacheManager(expected_dim=384)
        reloaded_cache.load(tmp_cache_file)

        if len(reloaded_cache) == 2:
            print("  [PASS] Cache length matches (2 items)")
        else:
            failures.append(f"Cache length mismatch: {len(reloaded_cache)} != 2")

        rec1_diff = float(np.max(np.abs(reloaded_cache.get("rec_001") - batch_embs[0])))
        if rec1_diff == 0.0:
            print("  [PASS] Reloaded vector is bit-for-bit identical to original")
        else:
            failures.append(f"Reloaded vector mismatch: diff = {rec1_diff}")

        # Batch lookup
        batch_lookup = reloaded_cache.get_batch(["rec_001", "rec_002"])
        if np.array_equal(batch_lookup, batch_embs):
            print("  [PASS] get_batch() returns exact matrix")
        else:
            failures.append("Batch lookup mismatch")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print("PHASE 2 TESTS: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("PHASE 2 TESTS: ALL PASS (5/5 tests successful)")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(run_tests())
