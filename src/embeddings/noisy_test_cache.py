"""
Noisy Test Audio and Embedding Cache Manager.

Phase 5: Manages deterministic noise injection and caching of Whisper embeddings
for held-out test audio across clean and noisy SNR conditions (15 dB, 10 dB, 5 dB).
Ensures zero data leakage, identical noise realizations across models, and high
throughput via batched Whisper-tiny inference.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.embeddings.whisper_encoder import WhisperEncoder
from src.embeddings.whisper_preprocessing import load_audio
from src.utils.noise import add_noise_at_snr, derive_noise_seed


class NoisyEmbeddingCacheManager:
    """
    Manages deterministic noisy audio generation and Whisper embedding extraction.
    """

    def __init__(
        self,
        clean_audio_dir: str = "data/torgo/audio",
        clean_embedding_dir: str = "embeddings/torgo/whisper",
        noisy_embedding_base_dir: str = "embeddings/torgo/whisper_noisy",
        manifest_path: str = "data/manifests/torgo_manifest.csv",
        base_seed: int = 42,
        expected_dim: int = 384,
    ):
        self.clean_audio_dir = clean_audio_dir
        self.clean_embedding_dir = clean_embedding_dir
        self.noisy_embedding_base_dir = noisy_embedding_base_dir
        self.manifest_path = manifest_path
        self.base_seed = base_seed
        self.expected_dim = expected_dim

        # Load manifest to get audio file paths if needed
        self.manifest_df = None
        if os.path.isfile(manifest_path):
            self.manifest_df = pd.read_csv(manifest_path).set_index("recording_id", drop=False)

        self._encoder: Optional[WhisperEncoder] = None

    def get_encoder(self) -> WhisperEncoder:
        """Lazy loader for WhisperEncoder."""
        if self._encoder is None:
            self._encoder = WhisperEncoder(
                model_id="openai/whisper-tiny",
                device="cpu",
                pooling="mean",
                normalize=True,
                seed=self.base_seed,
            )
        return self._encoder

    def get_embedding_path(self, recording_id: str, snr_db: Optional[float]) -> str:
        """Get the filesystem path for a clean or noisy embedding."""
        if snr_db is None or snr_db == "clean":
            return os.path.join(self.clean_embedding_dir, f"{recording_id}.npy")
        else:
            snr_int = int(round(float(snr_db)))
            return os.path.join(
                self.noisy_embedding_base_dir,
                f"snr_{snr_int}",
                f"{recording_id}.npy",
            )

    def has_embedding(self, recording_id: str, snr_db: Optional[float]) -> bool:
        """Check if an embedding exists on disk."""
        path = self.get_embedding_path(recording_id, snr_db)
        return os.path.isfile(path)

    def load_embedding(self, recording_id: str, snr_db: Optional[float]) -> np.ndarray:
        """Load and validate an embedding from disk."""
        path = self.get_embedding_path(recording_id, snr_db)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Embedding not found: {path}")

        arr = np.load(path)
        self.validate_embedding(arr, recording_id, snr_db)
        return arr

    def validate_embedding(
        self, arr: np.ndarray, recording_id: str, snr_db: Optional[float]
    ) -> None:
        """Validate shape, dtype, finite values, and unit-L2 normalization."""
        if arr.shape != (self.expected_dim,):
            raise ValueError(
                f"Invalid shape for {recording_id} (snr={snr_db}): expected ({self.expected_dim},), got {arr.shape}"
            )
        if arr.dtype != np.float32:
            raise TypeError(
                f"Invalid dtype for {recording_id} (snr={snr_db}): expected float32, got {arr.dtype}"
            )
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(
                f"Non-finite values found in {recording_id} (snr={snr_db})"
            )
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"Embedding for {recording_id} (snr={snr_db}) is not unit-L2 normalized (norm={norm:.6f})"
            )

    def get_audio_path(self, recording_id: str) -> str:
        """Resolve path to clean audio waveform."""
        if self.manifest_df is not None and recording_id in self.manifest_df.index:
            path_col = self.manifest_df.loc[recording_id, "audio_path"]
            if os.path.isfile(path_col):
                return path_col

        direct_path = os.path.join(self.clean_audio_dir, f"{recording_id}.wav")
        if os.path.isfile(direct_path):
            return direct_path

        raise FileNotFoundError(f"Clean audio file for {recording_id} not found.")

    def ensure_noisy_embeddings(
        self,
        recording_ids: List[str],
        snr_db: float,
        batch_size: int = 32,
    ) -> int:
        """
        Ensure all noisy embeddings for the given recordings at the specified SNR
        exist in cache. Generates missing ones using batched Whisper extraction.

        Args:
            recording_ids: List of unique recording IDs to check/extract.
            snr_db: Target SNR in decibels (e.g. 15.0, 10.0, 5.0).
            batch_size: Whisper inference batch size.

        Returns:
            Number of newly generated embeddings.
        """
        snr_int = int(round(float(snr_db)))
        target_dir = os.path.join(self.noisy_embedding_base_dir, f"snr_{snr_int}")
        os.makedirs(target_dir, exist_ok=True)

        missing_rids = [rid for rid in recording_ids if not self.has_embedding(rid, snr_db)]
        if not missing_rids:
            return 0

        encoder = self.get_encoder()
        generated_count = 0

        for start_idx in range(0, len(missing_rids), batch_size):
            batch_rids = missing_rids[start_idx : start_idx + batch_size]
            waveforms = []

            for rid in batch_rids:
                audio_path = self.get_audio_path(rid)
                clean_audio, sr = load_audio(audio_path, target_sr=16000, normalize=True)

                # Deterministic noise injection using stable seed
                noise_seed = derive_noise_seed(self.base_seed, rid, snr_db)
                rng = np.random.default_rng(noise_seed)
                noisy_audio = add_noise_at_snr(clean_audio, snr_db=snr_db, rng=rng)
                waveforms.append(noisy_audio)

            # Batch extract embeddings
            embeddings = encoder.extract_batch(waveforms, sr=16000)

            # Save and validate each embedding
            for rid, emb in zip(batch_rids, embeddings):
                self.validate_embedding(emb, rid, snr_db)
                out_path = self.get_embedding_path(rid, snr_db)
                np.save(out_path, emb)
                generated_count += 1

        return generated_count
