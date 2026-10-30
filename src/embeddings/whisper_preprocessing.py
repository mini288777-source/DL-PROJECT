"""
Audio Preprocessing Module for Whisper Feature Extraction.

Phase 2: Canonical audio loading, mono conversion, 16 kHz resampling,
float32 conversion, and deterministic peak normalization.
Strictly frozen: no data augmentation, no noise injection, no random cropping.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import scipy.signal


def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Safely load an audio file, convert to mono, resample to target_sr,
    and apply deterministic normalization.

    Args:
        audio_path: Path to the audio file on disk.
        target_sr: Target sample rate in Hz (default: 16000 for Whisper).
        normalize: Whether to apply deterministic peak normalization to [-1.0, 1.0].

    Returns:
        Tuple of:
            - waveform: 1D numpy array of float32 samples.
            - metadata: Dictionary containing audio properties and status.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    # Read audio using soundfile
    data, orig_sr = sf.read(audio_path, dtype="float32")

    return preprocess_audio(
        data=data,
        orig_sr=orig_sr,
        target_sr=target_sr,
        normalize=normalize,
        source_path=audio_path
    )


def preprocess_audio(
    data: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
    normalize: bool = True,
    source_path: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess raw audio waveform into canonical Whisper format.

    Args:
        data: Raw audio waveform (1D or 2D numpy array).
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16000).
        normalize: Whether to apply deterministic peak normalization.
        source_path: Optional source path for metadata tracking.

    Returns:
        Tuple of (preprocessed_1d_float32_waveform, metadata_dict).
    """
    metadata: Dict[str, Any] = {
        "source_path": source_path,
        "orig_sr": orig_sr,
        "target_sr": target_sr,
        "orig_channels": 1 if data.ndim == 1 else data.shape[1],
        "is_valid": True,
        "error_msg": None,
    }

    # Handle empty audio
    if data.size == 0:
        metadata["is_valid"] = False
        metadata["error_msg"] = "Empty audio signal (size=0)"
        return np.array([], dtype=np.float32), metadata

    # Ensure float32 representation
    waveform = data.astype(np.float32)

    # Convert to mono if multi-channel
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Validate non-finite values (NaN / Inf)
    if not np.all(np.isfinite(waveform)):
        metadata["is_valid"] = False
        metadata["error_msg"] = "Audio waveform contains NaN or Inf values"
        # Sanitize to zeros for unrecoverable errors
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample if necessary
    if orig_sr != target_sr and len(waveform) > 0:
        num_samples = int(round(len(waveform) * float(target_sr) / orig_sr))
        if num_samples > 0:
            waveform = scipy.signal.resample(waveform, num_samples).astype(np.float32)
        else:
            metadata["is_valid"] = False
            metadata["error_msg"] = "Resampled signal resulted in 0 samples"
            return np.array([], dtype=np.float32), metadata

    # Deterministic peak normalization
    if normalize and len(waveform) > 0:
        peak = np.max(np.abs(waveform))
        if peak > 1e-6:
            waveform = waveform / peak

    duration_sec = float(len(waveform)) / float(target_sr) if target_sr > 0 else 0.0
    metadata["duration_sec"] = duration_sec
    metadata["num_samples"] = len(waveform)

    return waveform, metadata
