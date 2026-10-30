"""
Noise injection utilities for the speech recognition project.

Provides additive white Gaussian noise injection at a target SNR level.
Designed for reuse across all experimental phases (Phase 0 smoke test
through Phase 5 noise robustness evaluation).
"""

import hashlib
import numpy as np


def derive_noise_seed(base_seed: int, recording_id: str, snr_db: float) -> int:
    """
    Derive a deterministic 32-bit integer seed from a base seed, recording ID, and SNR condition.
    Uses MD5 to ensure complete cross-platform, process-independent determinism.

    Args:
        base_seed: Base integer seed (e.g. 42).
        recording_id: Unique recording identifier string.
        snr_db: Target SNR in decibels.

    Returns:
        32-bit unsigned integer seed suitable for np.random.default_rng.
    """
    key = f"{base_seed}_{recording_id}_{snr_db:.1f}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def add_noise_at_snr(
    audio: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add white Gaussian noise to an audio signal at a specified SNR.

    SNR is defined as:
        SNR_dB = 10 * log10(P_signal / P_noise)

    where P_signal and P_noise are the mean squared power of the signal
    and noise respectively.

    Args:
        audio: 1-D float array of audio samples (clean signal).
        snr_db: Target signal-to-noise ratio in decibels.
        rng: NumPy Generator for reproducible noise generation.

    Returns:
        Noisy audio array (same shape and dtype as input).

    Raises:
        ValueError: If audio is empty or all-zero (cannot compute SNR).
    """
    if audio.size == 0:
        raise ValueError("Audio array is empty.")

    audio = audio.astype(np.float64)

    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-30:
        raise ValueError(
            "Audio signal power is effectively zero; cannot compute SNR."
        )

    # Compute required noise power
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    # Generate noise with the required power
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)

    noisy_audio = audio + noise
    return noisy_audio.astype(np.float32) if audio.dtype == np.float64 else noisy_audio


def estimate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Estimate the SNR of a noisy signal given the original clean signal.

    Args:
        clean: Original clean audio signal.
        noisy: Noisy version of the audio signal.

    Returns:
        Estimated SNR in decibels.
    """
    clean = clean.astype(np.float64)
    noisy = noisy.astype(np.float64)

    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-30:
        return float("inf")

    return 10.0 * np.log10(signal_power / noise_power)
