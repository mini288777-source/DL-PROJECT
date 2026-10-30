#!/usr/bin/env python
"""
Smoke Test #3 -- Noise Injection

Proves that the audio noise injection pipeline works correctly.

Tests:
    clean WAV -> noise at target SNR -> output -> reload -> verify

Validates at all four project SNR levels: clean, 15 dB, 10 dB, 5 dB.

Uses deterministic random seed and the reusable add_noise_at_snr() function
from src/utils/noise.py.

Usage:
    python tests/smoke_test_noise.py
"""

import sys
import os
import tempfile

import numpy as np
import soundfile as sf

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.reproducibility import set_seed, get_numpy_rng
from src.utils.noise import add_noise_at_snr, estimate_snr

# --- Configuration ---
SEED = 42
SAMPLE_RATE = 16000
DURATION_SEC = 2.0
SNR_LEVELS = [15, 10, 5]
SNR_TOLERANCE_DB = 0.5  # acceptable deviation from target SNR


def generate_test_audio(sr: int, duration: float, seed: int) -> np.ndarray:
    """Generate a deterministic test signal: 440 Hz + 880 Hz sine mix."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    signal = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 880.0 * t)
    # Add very slight noise to avoid a pathologically clean signal
    signal += 0.001 * rng.standard_normal(t.shape).astype(np.float32)
    return signal


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST #3 -- NOISE INJECTION")
    print("=" * 60)

    seed_config = set_seed(SEED)
    rng = get_numpy_rng(SEED)
    print(f"\nSeed: {seed_config['seed']}")

    # Generate clean audio
    print(f"\nGenerating clean audio: {DURATION_SEC}s at {SAMPLE_RATE} Hz...")
    clean = generate_test_audio(SAMPLE_RATE, DURATION_SEC, SEED)
    print(f"  Shape: {clean.shape}")
    print(f"  Range: [{clean.min():.4f}, {clean.max():.4f}]")
    print(f"  RMS: {np.sqrt(np.mean(clean**2)):.6f}")

    all_passed = True

    # Test clean -> save -> reload
    print("\n--- Clean signal round-trip ---")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        clean_path = f.name
    sf.write(clean_path, clean, SAMPLE_RATE)
    reloaded, sr_reloaded = sf.read(clean_path)
    assert sr_reloaded == SAMPLE_RATE, f"Sample rate mismatch: {sr_reloaded}"
    assert len(reloaded) == len(clean), f"Duration mismatch: {len(reloaded)} vs {len(clean)}"
    print(f"  Save/reload: OK ({sr_reloaded} Hz, {len(reloaded)} samples)")
    os.unlink(clean_path)

    # Test each SNR level
    for snr_target in SNR_LEVELS:
        print(f"\n--- SNR = {snr_target} dB ---")

        # Add noise
        noisy = add_noise_at_snr(clean, snr_target, rng)

        # Basic checks
        assert noisy.shape == clean.shape, (
            f"Shape mismatch: clean={clean.shape}, noisy={noisy.shape}"
        )
        assert np.isfinite(noisy).all(), "Noisy audio contains NaN/Inf"

        # Verify signal is different from clean
        diff = np.abs(noisy.astype(np.float64) - clean.astype(np.float64)).max()
        assert diff > 1e-6, "Noisy signal is identical to clean!"
        print(f"  Max abs difference from clean: {diff:.6f}")

        # Estimate achieved SNR
        achieved_snr = estimate_snr(clean, noisy)
        error = abs(achieved_snr - snr_target)
        print(f"  Target SNR:   {snr_target:.1f} dB")
        print(f"  Achieved SNR: {achieved_snr:.2f} dB")
        print(f"  Error:        {error:.2f} dB (tolerance: +/-{SNR_TOLERANCE_DB} dB)")

        if error > SNR_TOLERANCE_DB:
            print(f"  [FAIL] SNR error exceeds tolerance!")
            all_passed = False
        else:
            print(f"  [OK]")

        # Save/reload round-trip for noisy
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            noisy_path = f.name
        sf.write(noisy_path, noisy, SAMPLE_RATE)
        noisy_reloaded, sr2 = sf.read(noisy_path)
        assert sr2 == SAMPLE_RATE, f"Sample rate mismatch after noise: {sr2}"
        assert len(noisy_reloaded) == len(noisy), "Duration changed after noise"
        print(f"  Round-trip: OK")
        os.unlink(noisy_path)

    # Verify monotonicity: lower SNR -> more noise -> bigger difference
    print("\n--- Monotonicity check ---")
    rng2 = get_numpy_rng(SEED + 100)
    diffs = []
    for snr_target in SNR_LEVELS:
        noisy = add_noise_at_snr(clean, snr_target, rng2)
        d = np.sqrt(np.mean((noisy.astype(np.float64) - clean.astype(np.float64))**2))
        diffs.append((snr_target, d))
        print(f"  SNR {snr_target:2d} dB -> RMS noise = {d:.6f}")

    # Lower SNR should have higher noise
    for i in range(len(diffs) - 1):
        if diffs[i][1] >= diffs[i + 1][1]:
            print(f"  [FAIL] Noise at {diffs[i][0]} dB >= noise at {diffs[i+1][0]} dB")
            all_passed = False
    if all_passed:
        print("  Monotonicity: OK (lower SNR -> more noise)")

    # Summary
    print("\n" + "=" * 60)
    print("NOISE INJECTION SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Sample rate:  {SAMPLE_RATE} Hz")
    print(f"  Duration:     {DURATION_SEC}s")
    print(f"  SNR levels:   clean, {', '.join(str(s) + ' dB' for s in SNR_LEVELS)}")
    print(f"  Tolerance:    +/-{SNR_TOLERANCE_DB} dB")
    print(f"  Seed:         {SEED}")
    print()
    if all_passed:
        print("NOISE INJECTION SMOKE TEST: PASS")
        return 0
    else:
        print("NOISE INJECTION SMOKE TEST: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
