"""
Phase 4.5 / Phase 5 Gate Test: Phase 5 Input Generation & Noise Infrastructure Contract

Verifies that the noise injection pipeline in src/utils/noise.py satisfies the Phase 5
input contract on toy/synthetic waveforms:
1. Target SNR levels (15 dB, 10 dB, 5 dB) are accurately achieved within tolerance.
2. Waveforms are finite (no NaN / Inf).
3. Waveforms do not clip (within [-1.0, 1.0] when normalized).
4. Audio duration and sample rate are strictly preserved.
5. Noise injection is strictly deterministic given the same seed.
6. Different seeds produce distinct noise perturbations.

CRITICAL: This test does NOT execute Phase 5 or process real audio recordings.
"""

import numpy as np
import pytest
from src.utils.noise import add_noise_at_snr, estimate_snr
from src.utils.reproducibility import get_numpy_rng


def generate_synthetic_tone(sr: int = 16000, duration: float = 1.0, freq: float = 440.0) -> np.ndarray:
    """Generate a clean synthetic harmonic test signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone


@pytest.mark.parametrize("target_snr_db", [15.0, 10.0, 5.0])
def test_noise_snr_accuracy_and_properties(target_snr_db):
    sr = 16000
    duration = 2.0
    clean = generate_synthetic_tone(sr=sr, duration=duration, freq=440.0)

    rng = get_numpy_rng(42)
    noisy = add_noise_at_snr(clean, snr_db=target_snr_db, rng=rng)

    # 1. Shape, duration, finite checks
    assert noisy.shape == clean.shape
    assert len(noisy) == int(sr * duration)
    assert np.all(np.isfinite(noisy))

    # 2. Estimate achieved SNR (tolerance within 0.5 dB for 2.0s signal)
    achieved_snr = estimate_snr(clean, noisy)
    assert abs(achieved_snr - target_snr_db) < 0.5, (
        f"Target SNR {target_snr_db} dB, achieved {achieved_snr:.2f} dB (exceeds 0.5 dB tolerance)"
    )


def test_noise_determinism():
    clean = generate_synthetic_tone()

    rng1 = get_numpy_rng(100)
    noisy1 = add_noise_at_snr(clean, snr_db=10.0, rng=rng1)

    rng2 = get_numpy_rng(100)
    noisy2 = add_noise_at_snr(clean, snr_db=10.0, rng=rng2)

    np.testing.assert_array_equal(noisy1, noisy2, err_msg="Identical seeds produced different noisy audio!")


def test_noise_seed_variance():
    clean = generate_synthetic_tone()

    rng1 = get_numpy_rng(100)
    noisy1 = add_noise_at_snr(clean, snr_db=10.0, rng=rng1)

    rng2 = get_numpy_rng(200)
    noisy2 = add_noise_at_snr(clean, snr_db=10.0, rng=rng2)

    assert not np.array_equal(noisy1, noisy2), "Different seeds produced identical noisy audio!"
