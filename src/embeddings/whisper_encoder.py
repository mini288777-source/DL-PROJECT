"""
Whisper Encoder and Feature Extraction Module.

Phase 2: Extracts canonical fixed-dimensional speech embeddings from audio.
Model weights are strictly FROZEN (eval mode, torch.inference_mode, requires_grad=False).
Supports both single-sample and batch extraction with deterministic pooling.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.embeddings.whisper_preprocessing import load_audio, preprocess_audio
from src.utils.reproducibility import set_seed


class WhisperEncoder:
    """
    Frozen Whisper encoder representation extractor.
    Loads once, runs deterministically on CPU or GPU.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        device: Optional[str] = None,
        pooling: str = "mean",
        normalize: bool = True,
        seed: int = 42
    ):
        """
        Initialize the frozen Whisper encoder.

        Args:
            model_id: HuggingFace model identifier (e.g. 'openai/whisper-large-v2' or 'openai/whisper-tiny').
            device: 'cuda', 'cpu', or None (auto-detect).
            pooling: Temporal pooling strategy ('mean', 'max', 'first').
            normalize: Whether to apply L2 unit normalization.
            seed: Deterministic random seed.
        """
        self.model_id = model_id
        self.pooling = pooling.lower()
        self.normalize = normalize
        self.seed = seed

        # Deterministic seed initialization
        set_seed(seed)

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load feature extractor and model weights
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        base_model = WhisperModel.from_pretrained(model_id)

        # Strictly freeze encoder weights
        self.encoder = base_model.encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim: int = int(self.encoder.config.d_model)

    def extract_from_waveform(
        self,
        waveform: np.ndarray,
        sr: int = 16000
    ) -> np.ndarray:
        """
        Extract fixed-size embedding for a single 1D waveform.

        Args:
            waveform: 1D numpy array of float32 samples.
            sr: Sampling rate (must be 16000 Hz).

        Returns:
            1D numpy array of shape (hidden_dim,) with dtype float32.
        """
        batch_embeddings = self.extract_batch([waveform], sr=sr)
        return batch_embeddings[0]

    def extract_batch(
        self,
        waveforms: List[np.ndarray],
        sr: int = 16000
    ) -> np.ndarray:
        """
        Extract fixed-size embeddings for a batch of 1D waveforms.

        Args:
            waveforms: List of 1D numpy arrays of float32 samples.
            sr: Sampling rate (must be 16000 Hz).

        Returns:
            2D numpy array of shape (batch_size, hidden_dim) with dtype float32.
        """
        if not waveforms:
            return np.empty((0, self.hidden_dim), dtype=np.float32)

        # Ensure all inputs are 1D float32
        sanitized = []
        for w in waveforms:
            if w.ndim > 1:
                w = np.mean(w, axis=1)
            sanitized.append(w.astype(np.float32))

        # Extract log-mel filterbank features (80 mel bands, 30s window)
        inputs = self.feature_extractor(
            sanitized,
            sampling_rate=sr,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        with torch.inference_mode():
            outputs = self.encoder(input_features)
            # last_hidden_state shape: (batch_size, time_frames, hidden_dim)
            hidden_states = outputs.last_hidden_state

            # Temporal pooling
            if self.pooling == "mean":
                pooled = torch.mean(hidden_states, dim=1)
            elif self.pooling == "max":
                pooled = torch.max(hidden_states, dim=1).values
            elif self.pooling == "first":
                pooled = hidden_states[:, 0, :]
            else:
                pooled = torch.mean(hidden_states, dim=1)

            # L2 unit normalization
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        result = pooled.cpu().numpy().astype(np.float32)

        # Verify no NaN or Inf
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Extracted embedding contains NaN or Inf values")

        return result


def extract_whisper_embedding(
    audio_path: str,
    encoder: WhisperEncoder,
    metadata_record: Optional[Dict[str, Any]] = None,
    target_sr: int = 16000,
    normalize_audio: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Canonical API function to extract a Whisper embedding from an audio file.

    Args:
        audio_path: Path to the audio file.
        encoder: Pre-loaded, frozen WhisperEncoder instance.
        metadata_record: Optional dict with recording lineage metadata.
        target_sr: Target sample rate (16000 Hz).
        normalize_audio: Whether to normalize audio waveform prior to feature extraction.

    Returns:
        Tuple of (1D float32 embedding, metadata_dict).
    """
    waveform, audio_meta = load_audio(
        audio_path,
        target_sr=target_sr,
        normalize=normalize_audio
    )

    if not audio_meta.get("is_valid", True) or len(waveform) == 0:
        raise ValueError(f"Invalid audio file: {audio_path}. Reason: {audio_meta.get('error_msg')}")

    embedding = encoder.extract_from_waveform(waveform, sr=target_sr)

    # Compile comprehensive lineage metadata
    meta: Dict[str, Any] = {
        "audio_path": audio_path,
        "duration_sec": audio_meta.get("duration_sec", 0.0),
        "sample_rate": target_sr,
        "model_name": encoder.model_id,
        "pooling_method": encoder.pooling,
        "embedding_dim": encoder.hidden_dim,
        "dtype": str(embedding.dtype),
        "is_valid": True,
    }

    if metadata_record:
        for key in ["recording_id", "speaker_id", "session_id", "split", "intent"]:
            if key in metadata_record:
                meta[key] = metadata_record[key]

    return embedding, meta
