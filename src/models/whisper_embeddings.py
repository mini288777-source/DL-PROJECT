"""
Whisper Acoustic Representation Extractor Module.

Loads pretrained Whisper encoder and extracts fixed-dimensional
acoustic embedding vectors from audio waveforms.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.reproducibility import set_seed


class WhisperEmbeddingExtractor:
    """
    Deterministic Whisper acoustic embedding extractor.
    Extracts fixed-dimensional representations from audio signals.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        device: str = "cpu",
        pooling: str = "mean",
        normalize: bool = True,
        seed: int = 42
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.pooling = pooling
        self.normalize = normalize
        self.seed = seed

        # Set deterministic seeds
        set_seed(seed)

        print(f"Loading Whisper model: {model_id} on {device}...")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        base_model = WhisperModel.from_pretrained(model_id)
        self.encoder = base_model.encoder.to(self.device).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.d_model  # 384 for whisper-tiny
        print(f"Whisper encoder initialized. Hidden dimension: {self.hidden_dim}")

    def extract_from_waveform(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> np.ndarray:
        """
        Extract embedding vector from a single 1D audio waveform.

        Args:
            audio: 1D numpy array of float audio samples.
            sr: Sampling rate (must be 16000 Hz).

        Returns:
            1D numpy array of shape (hidden_dim,) with dtype float32.
        """
        batch_res = self.extract_batch([audio], sr=sr)
        return batch_res[0]

    def extract_batch(
        self,
        audio_list: List[np.ndarray],
        sr: int = 16000
    ) -> np.ndarray:
        """
        Extract embedding vectors from a batch of 1D audio waveforms.

        Args:
            audio_list: List of 1D numpy arrays of float audio samples.
            sr: Sampling rate (must be 16000 Hz).

        Returns:
            2D numpy array of shape (batch_size, hidden_dim) with dtype float32.
        """
        # Ensure 1D float32 inputs
        processed_audio = []
        for a in audio_list:
            if a.ndim > 1:
                a = np.mean(a, axis=1)
            processed_audio.append(a.astype(np.float32))

        # Extract log-mel spectrogram features
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=sr,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            encoder_outputs = self.encoder(input_features)
            # hidden states shape: (batch_size, time_frames, hidden_dim)
            hidden_states = encoder_outputs.last_hidden_state

            # Temporal pooling
            if self.pooling == "mean":
                pooled = torch.mean(hidden_states, dim=1)
            elif self.pooling == "max":
                pooled = torch.max(hidden_states, dim=1).values
            elif self.pooling == "first":
                pooled = hidden_states[:, 0, :]
            else:
                pooled = torch.mean(hidden_states, dim=1)

            # Normalization (L2 unit sphere)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.cpu().numpy().astype(np.float32)

    def extract_from_file(self, file_path: str) -> np.ndarray:
        """Read audio file from disk and extract its embedding vector."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        data, sr = sf.read(file_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        return self.extract_from_waveform(data, sr=sr)
