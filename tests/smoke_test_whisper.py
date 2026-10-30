#!/usr/bin/env python
"""
Smoke Test #1 -- Whisper Encoder

Proves that a synthetic audio sample can pass through the Whisper encoder
and produce valid, deterministic embeddings.

This is NOT the full TORGO extraction pipeline.
This only validates: audio -> preprocessing -> Whisper -> embedding.

Usage:
    python tests/smoke_test_whisper.py
"""

import sys
import os
import time
import tempfile

import numpy as np
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.reproducibility import set_seed

# --- Configuration ---
WHISPER_MODEL_ID = "openai/whisper-tiny"
SAMPLE_RATE = 16000
DURATION_SEC = 2.0
SEED = 42


def generate_synthetic_audio(
    sr: int = SAMPLE_RATE, duration: float = DURATION_SEC, seed: int = SEED
) -> np.ndarray:
    """Generate a synthetic audio signal: 440 Hz sine + light noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.01 * rng.standard_normal(t.shape).astype(np.float32)
    return tone + noise


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST #1 -- WHISPER ENCODER")
    print("=" * 60)

    # Reproducibility
    seed_config = set_seed(SEED)
    print(f"\nSeed config: {seed_config['seed']}")

    # Step 1: Generate synthetic audio
    print(f"\nGenerating synthetic audio: {DURATION_SEC}s at {SAMPLE_RATE} Hz...")
    audio = generate_synthetic_audio()
    assert audio.shape == (int(SAMPLE_RATE * DURATION_SEC),), (
        f"Unexpected audio shape: {audio.shape}"
    )
    assert np.isfinite(audio).all(), "Audio contains NaN/Inf values"
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio dtype: {audio.dtype}")
    print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Step 2: Load Whisper model
    print(f"\nLoading Whisper model: {WHISPER_MODEL_ID}...")
    from transformers import WhisperFeatureExtractor, WhisperModel

    t0 = time.time()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_ID)
    model = WhisperModel.from_pretrained(WHISPER_MODEL_ID)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.2f}s on {device}")

    # Step 3: Preprocess audio for Whisper
    print("\nPreprocessing audio...")
    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    print(f"  Input features shape: {input_features.shape}")

    # Step 4: Run through encoder
    print("\nRunning encoder forward pass...")
    t0 = time.time()
    with torch.no_grad():
        encoder_output = model.encoder(input_features)
    encode_time = time.time() - t0

    # Extract last hidden state as embedding
    embeddings = encoder_output.last_hidden_state  # (batch, seq_len, hidden_dim)
    print(f"  Encoder output shape: {embeddings.shape}")
    print(f"  Encoding time: {encode_time:.4f}s")

    # Step 5: Validate embeddings
    print("\nValidating embeddings...")
    emb_np = embeddings.cpu().numpy()

    # Check shape
    batch_size, seq_len, hidden_dim = emb_np.shape
    assert batch_size == 1, f"Expected batch=1, got {batch_size}"
    assert seq_len > 0, f"Sequence length is 0"
    assert hidden_dim > 0, f"Hidden dim is 0"
    print(f"  Shape OK: ({batch_size}, {seq_len}, {hidden_dim})")

    # Check for NaN / Inf
    assert np.isfinite(emb_np).all(), "Embeddings contain NaN/Inf!"
    print("  Finite check: OK (no NaN/Inf)")

    # Check not all zeros
    assert np.abs(emb_np).max() > 1e-6, "Embeddings are all zeros!"
    print(f"  Value range: [{emb_np.min():.6f}, {emb_np.max():.6f}]")
    print(f"  Mean: {emb_np.mean():.6f}, Std: {emb_np.std():.6f}")

    # Step 6: Save and reload
    print("\nSave/reload round-trip test...")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_path = f.name
        np.save(f, emb_np)

    reloaded = np.load(tmp_path)
    assert np.array_equal(emb_np, reloaded), "Save/reload mismatch!"
    os.unlink(tmp_path)
    print("  Round-trip: OK")

    # Step 7: Determinism check -- run again
    print("\nDeterminism check (second forward pass)...")
    with torch.no_grad():
        encoder_output_2 = model.encoder(input_features)
    emb_np_2 = encoder_output_2.last_hidden_state.cpu().numpy()
    assert np.allclose(emb_np, emb_np_2, atol=1e-6), (
        f"Embeddings differ between runs! Max diff: {np.abs(emb_np - emb_np_2).max()}"
    )
    print("  Determinism: OK (two passes match)")

    # Summary
    print("\n" + "=" * 60)
    print("WHISPER SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Model:           {WHISPER_MODEL_ID}")
    print(f"  Sample rate:     {SAMPLE_RATE} Hz")
    print(f"  Input duration:  {DURATION_SEC}s")
    print(f"  Embedding shape: ({batch_size}, {seq_len}, {hidden_dim})")
    print(f"  Encoding time:   {encode_time:.4f}s")
    print(f"  Device:          {device}")
    print(f"  Deterministic:   Yes")
    print()
    print("WHISPER SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
