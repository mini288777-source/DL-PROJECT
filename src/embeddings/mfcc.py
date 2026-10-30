"""
Classical Acoustic Baseline Representation: MFCC Extractor.

Phase 2: Extracts deterministic MFCC features (13 static + 13 delta + 13 delta-delta = 39 channels),
aggregated temporally via mean and standard deviation into a fixed 78-dimensional vector.
Strictly non-trainable, deterministic acoustic baseline representation.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import librosa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.embeddings.whisper_preprocessing import load_audio, preprocess_audio


class MFCCExtractor:
    """
    Deterministic classical MFCC acoustic feature extractor.
    Extracts 13 MFCCs + deltas + delta-deltas, temporally pooled to 78 dimensions.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 400,
        hop_length: int = 160,
        aggregation: str = "mean_std"
    ):
        """
        Initialize the MFCC extractor.

        Args:
            sr: Sampling rate in Hz (default: 16000).
            n_mfcc: Number of MFCC coefficients to compute (default: 13).
            n_fft: FFT window size in samples (default: 400, corresponds to 25ms at 16kHz).
            hop_length: Frame hop size in samples (default: 160, corresponds to 10ms at 16kHz).
            aggregation: Temporal aggregation strategy ('mean_std' -> 78-dim, 'mean' -> 39-dim).
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.aggregation = aggregation

        if aggregation == "mean_std":
            self.feature_dim = n_mfcc * 3 * 2  # 13 * 3 * 2 = 78
        elif aggregation == "mean":
            self.feature_dim = n_mfcc * 3      # 13 * 3 = 39
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    def extract_from_waveform(
        self,
        waveform: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract fixed-size MFCC vector from 1D waveform.

        Args:
            waveform: 1D numpy array of float32 samples.
            sr: Sampling rate (must match self.sr).

        Returns:
            1D numpy array of shape (feature_dim,) with dtype float32.
        """
        if sr is not None and sr != self.sr:
            # Resample if needed
            waveform, _ = preprocess_audio(waveform, orig_sr=sr, target_sr=self.sr)

        # Handle very short audio gracefully
        if len(waveform) < self.n_fft:
            pad_amount = self.n_fft - len(waveform)
            waveform = np.pad(waveform, (0, pad_amount), mode="constant")

        # Compute static MFCCs
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Compute first derivative (delta) and second derivative (delta-delta)
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Concatenate along feature dimension: shape (39, time_frames)
        features = np.concatenate([mfcc, delta1, delta2], axis=0)

        # Temporal pooling
        if self.aggregation == "mean_std":
            mean_vals = np.mean(features, axis=1)
            std_vals = np.std(features, axis=1)
            pooled = np.concatenate([mean_vals, std_vals], axis=0)
        elif self.aggregation == "mean":
            pooled = np.mean(features, axis=1)
        else:
            pooled = np.mean(features, axis=1)

        result = pooled.astype(np.float32)

        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Extracted MFCC features contain NaN or Inf values")

        return result


def extract_mfcc_features(
    audio_path: str,
    extractor: Optional[MFCCExtractor] = None,
    metadata_record: Optional[Dict[str, Any]] = None,
    sr: int = 16000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract MFCC features from an audio file.

    Args:
        audio_path: Path to the audio file.
        extractor: Optional pre-initialized MFCCExtractor instance.
        metadata_record: Optional dict with recording lineage metadata.
        sr: Sampling rate.

    Returns:
        Tuple of (1D float32 MFCC vector, metadata_dict).
    """
    if extractor is None:
        extractor = MFCCExtractor(sr=sr)

    waveform, audio_meta = load_audio(audio_path, target_sr=sr, normalize=True)

    if not audio_meta.get("is_valid", True) or len(waveform) == 0:
        raise ValueError(f"Invalid audio file: {audio_path}. Reason: {audio_meta.get('error_msg')}")

    features = extractor.extract_from_waveform(waveform, sr=sr)

    meta: Dict[str, Any] = {
        "audio_path": audio_path,
        "duration_sec": audio_meta.get("duration_sec", 0.0),
        "sample_rate": sr,
        "mfcc_dim": extractor.feature_dim,
        "dtype": str(features.dtype),
        "is_valid": True,
    }

    if metadata_record:
        for key in ["recording_id", "speaker_id", "session_id", "split", "intent"]:
            if key in metadata_record:
                meta[key] = metadata_record[key]

    return features, meta
