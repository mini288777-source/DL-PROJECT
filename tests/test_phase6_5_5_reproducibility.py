"""
Unit tests for Subphase 6.5.5: Reproducibility, Artifacts & Benchmark Preservation.
Validates:
1. All required result CSVs exist and contain expected row counts.
2. All 9 required publication figures exist and are non-empty.
3. Original Phase 3-6 benchmark artifacts remain bitwise preserved and untouched.
"""

import os
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "phase6_5_5", "figures")


def test_result_csv_artifacts_exist_and_counts():
    """Verify all 7 Phase 6.5.5 result CSVs exist with exact expected row counts."""
    expected_csvs = {
        "phase6_5_5_speaker_results.csv": 420,
        "phase6_5_5_quantum_conditions.csv": 180,
        "phase6_5_5_classical_conditions.csv": 240,
        "phase6_5_5_kernel_diagnostics.csv": 180,
        "phase6_5_5_runtime.csv": 7,
        "phase6_5_5_matched_comparison.csv": 60,
        "phase6_5_5_statistical_tests.csv": 12,
    }

    for filename, expected_rows in expected_csvs.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        assert os.path.exists(filepath), f"Missing expected result CSV: {filename}"
        df = pd.read_csv(filepath)
        assert len(df) == expected_rows, f"{filename} row count mismatch: expected {expected_rows}, got {len(df)}"


def test_publication_figures_exist_and_nonempty():
    """Verify all 9 required publication figures exist and are non-empty."""
    expected_figures = [
        "matched_macro_f1.png",
        "emergency_recall_comparison.png",
        "false_emergency_rate_comparison.png",
        "stage2_macro_f1_comparison.png",
        "stage2_balanced_accuracy_comparison.png",
        "per_n_sample_efficiency.png",
        "quantum_kernel_eigenvalues.png",
        "runtime_comparison.png",
        "speaker_level_comparison.png",
    ]

    for fig_name in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(fig_path), f"Missing publication figure: {fig_name}"
        assert os.path.getsize(fig_path) > 1000, f"Figure {fig_name} is too small or corrupt ({os.path.getsize(fig_path)} bytes)"


def test_historical_benchmarks_preserved():
    """Verify original Phase 3-6 benchmark files are preserved and unmodified."""
    historical_paths = [
        os.path.join(RESULTS_DIR, "phase3", "baseline_curves.csv"),
        os.path.join(RESULTS_DIR, "phase3", "predictions.csv"),
        os.path.join(RESULTS_DIR, "phase4", "quantum_curves.csv"),
        os.path.join(RESULTS_DIR, "phase4", "classical_vs_quantum_comparison.csv"),
        os.path.join(RESULTS_DIR, "phase5", "phase5_results.csv"),
    ]

    for p in historical_paths:
        assert os.path.exists(p), f"Historical benchmark artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Historical benchmark artifact empty: {p}"
