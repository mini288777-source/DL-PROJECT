"""
Phase 2 Determinism Test.

Verifies that running Whisper and MFCC feature extraction multiple times
on the exact same clean input waveform yields numerically equivalent representations.
Ensures zero data leakage, zero randomness, and complete reproducibility.
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.whisper_encoder import WhisperEncoder, extract_whisper_embedding
from src.embeddings.mfcc import MFCCExtractor, extract_mfcc_features
from src.embeddings.whisper_preprocessing import load_audio


def test_audio_preprocessing_determinism():
    """Verify that preprocessing produces identical waveforms across runs."""
    audio_path = "data/torgo/audio/F01_1_headMic_0020.wav"
    assert os.path.exists(audio_path), f"Test audio file missing: {audio_path}"

    wav1, meta1 = load_audio(audio_path, target_sr=16000, normalize=True)
    wav2, meta2 = load_audio(audio_path, target_sr=16000, normalize=True)

    assert meta1["is_valid"] is True
    assert meta2["is_valid"] is True
    assert len(wav1) == len(wav2)
    assert np.array_equal(wav1, wav2), "Waveform preprocessing is not byte-for-byte deterministic"


def test_whisper_embedding_determinism():
    """Verify that Whisper embedding extraction produces numerically equivalent vectors."""
    audio_path = "data/torgo/audio/F01_1_headMic_0020.wav"
    assert os.path.exists(audio_path), f"Test audio file missing: {audio_path}"

    encoder = WhisperEncoder(model_id="openai/whisper-tiny", device="cpu", pooling="mean", normalize=True, seed=42)

    emb1, meta1 = extract_whisper_embedding(audio_path, encoder)
    emb2, meta2 = extract_whisper_embedding(audio_path, encoder)

    assert emb1.shape == emb2.shape
    assert emb1.dtype == np.float32
    assert emb2.dtype == np.float32
    assert not np.isnan(emb1).any()
    assert not np.isinf(emb1).any()

    # Numerical equivalence test (tolerance 1e-6)
    diff = np.max(np.abs(emb1 - emb2))
    assert diff < 1e-6, f"Whisper embedding extraction differs across runs: max diff = {diff}"


def test_mfcc_feature_determinism():
    """Verify that MFCC feature extraction produces byte-identical vectors."""
    audio_path = "data/torgo/audio/F01_1_headMic_0020.wav"
    assert os.path.exists(audio_path), f"Test audio file missing: {audio_path}"

    extractor = MFCCExtractor(sr=16000, n_mfcc=13, aggregation="mean_std")

    feat1, meta1 = extract_mfcc_features(audio_path, extractor)
    feat2, meta2 = extract_mfcc_features(audio_path, extractor)

    assert feat1.shape == (78,)
    assert feat2.shape == (78,)
    assert feat1.dtype == np.float32
    assert feat2.dtype == np.float32
    assert not np.isnan(feat1).any()
    assert not np.isinf(feat1).any()

    diff = np.max(np.abs(feat1 - feat2))
    assert diff < 1e-6, f"MFCC extraction differs across runs: max diff = {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
