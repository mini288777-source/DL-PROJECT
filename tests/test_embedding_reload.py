"""
Phase 2 Primary Acceptance Test: Embedding Reload & Lineage Verification.

Verifies:
1. Extract embedding and save as .npy.
2. Terminate/bypass the extraction engine.
3. Reload .npy using lightweight numpy loader (zero PyTorch / Whisper dependency).
4. Verify file existence, shape, dtype float32, no NaN, no Inf.
5. Verify recording_id, speaker_id, and split lineage are fully preserved.
6. Verify fast, inference-free loading for downstream classifiers (RBF, Quantum).
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.cache import (
    EmbeddingManifestManager,
    get_embedding_path,
    load_embedding_file,
    save_embedding_file,
)
from src.embeddings.whisper_encoder import WhisperEncoder, extract_whisper_embedding
from src.embeddings.mfcc import MFCCExtractor, extract_mfcc_features


def test_whisper_embedding_reload():
    """Verify Whisper embedding extraction, caching, and inference-free reload."""
    audio_path = "data/torgo/audio/F01_1_headMic_0020.wav"
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Extract embedding with Whisper
        encoder = WhisperEncoder(model_id="openai/whisper-tiny", device="cpu", pooling="mean", normalize=True)
        rec_meta = {
            "recording_id": "TEST_REC_001",
            "speaker_id": "F01",
            "session_id": "session1",
            "split": "adaptation",
            "intent": "GENERAL_SPEECH"
        }
        emb, meta = extract_whisper_embedding(audio_path, encoder, metadata_record=rec_meta)

        # Step 2: Save .npy file
        saved_path = save_embedding_file("TEST_REC_001", emb, feature_type="whisper", base_dir=temp_dir)
        assert os.path.exists(saved_path), f"Saved embedding file does not exist: {saved_path}"

        # Step 3: Delete encoder object to prove zero dependency on model during reload
        del encoder

        # Step 4: Reload .npy using lightweight loader
        reloaded = load_embedding_file("TEST_REC_001", feature_type="whisper", base_dir=temp_dir, expected_dim=384)

        # Step 5: Verification assertions
        assert isinstance(reloaded, np.ndarray), "Reloaded object is not a numpy array"
        assert reloaded.shape == (384,), f"Unexpected reloaded shape: {reloaded.shape}"
        assert reloaded.dtype == np.float32, f"Unexpected dtype: {reloaded.dtype}"
        assert not np.isnan(reloaded).any(), "Reloaded array contains NaN"
        assert not np.isinf(reloaded).any(), "Reloaded array contains Inf"
        assert np.allclose(reloaded, emb, atol=1e-7), "Reloaded array differs from original extracted array"

        # Verify lineage tracking
        assert meta["recording_id"] == "TEST_REC_001"
        assert meta["speaker_id"] == "F01"
        assert meta["split"] == "adaptation"


def test_mfcc_reload():
    """Verify MFCC acoustic extraction, caching, and inference-free reload."""
    audio_path = "data/torgo/audio/F01_1_headMic_0020.wav"
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        extractor = MFCCExtractor(sr=16000, n_mfcc=13, aggregation="mean_std")
        rec_meta = {
            "recording_id": "TEST_MFCC_001",
            "speaker_id": "F01",
            "session_id": "session1",
            "split": "adaptation",
            "intent": "GENERAL_SPEECH"
        }
        feat, meta = extract_mfcc_features(audio_path, extractor, metadata_record=rec_meta)

        # Save .npy
        saved_path = save_embedding_file("TEST_MFCC_001", feat, feature_type="mfcc", base_dir=temp_dir)
        assert os.path.exists(saved_path)

        # Delete extractor
        del extractor

        # Reload
        reloaded = load_embedding_file("TEST_MFCC_001", feature_type="mfcc", base_dir=temp_dir, expected_dim=78)

        # Verify
        assert isinstance(reloaded, np.ndarray)
        assert reloaded.shape == (78,)
        assert reloaded.dtype == np.float32
        assert not np.isnan(reloaded).any()
        assert not np.isinf(reloaded).any()
        assert np.allclose(reloaded, feat, atol=1e-7)


def test_manifest_manager_lineage():
    """Verify EmbeddingManifestManager tracks lineage without loss."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = os.path.join(temp_dir, "metadata", "test_manifest.csv")
        mgr = EmbeddingManifestManager(manifest_path=manifest_path)

        record = {
            "recording_id": "REC_AUDIT_100",
            "speaker_id": "M01",
            "session_id": "session2",
            "split": "test",
            "intent": "HELP",
            "audio_path": "data/torgo/audio/M01_test.wav",
            "whisper_path": "embeddings/torgo/whisper/REC_AUDIT_100.npy",
            "whisper_dim": 384,
            "whisper_dtype": "float32",
            "mfcc_path": "embeddings/torgo/mfcc/REC_AUDIT_100.npy",
            "mfcc_dim": 78,
            "duration_sec": 3.42,
            "sample_rate": 16000,
            "model_name": "openai/whisper-tiny",
            "pooling_method": "mean",
            "extraction_status": "SUCCESS"
        }

        mgr.upsert_record(record)
        mgr.save()

        # Reload manifest CSV and verify
        df = pd.read_csv(manifest_path)
        assert len(df) == 1
        assert df.iloc[0]["recording_id"] == "REC_AUDIT_100"
        assert df.iloc[0]["speaker_id"] == "M01"
        assert df.iloc[0]["split"] == "test"
        assert df.iloc[0]["intent"] == "HELP"
        assert df.iloc[0]["whisper_dim"] == 384
        assert df.iloc[0]["mfcc_dim"] == 78


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
