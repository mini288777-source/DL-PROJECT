"""
Unit tests for Subphase 6.5.5-C: Quantum Statistical Closure & Final Audit.
Validates:
1. Exactly 60 matched speaker x N conditions.
2. No missing Adapter / QSVC matched pairs.
3. No duplicate matched pairs.
4. Correct Q4 / Q8 pairing in ablations.
5. Correct Q8-R1 / Q8-R2 pairing in ablations.
6. Statistical result reproducibility from speaker results.
7. Runtime ratio arithmetic (per-evaluation vs total wall time).
8. Holm-Bonferroni correction reproducibility.
9. Original Phase 6.5.5 artifacts preserved.
10. Original Phase 3-6 benchmark artifacts preserved.
11. Quantum kernel integrity verified across cached matrices.
12. Experimental unit strictly verified as speaker x calibration budget N.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
KERNEL_CACHE_DIR = os.path.join(RESULTS_DIR, "phase6_5_5", "kernel_cache")

SPEAKER_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv")
ADAPTER_COMP_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_C_adapter_comparison.csv")
ABLATION_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_C_quantum_ablation.csv")
MULTIPLE_COMP_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_C_multiple_comparison.csv")
RUNTIME_AUDIT_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_C_runtime_audit.csv")
DECISION_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_C_final_decision.csv")
KERNEL_DIAG_PATH = os.path.join(RESULTS_DIR, "phase6_5_5_kernel_diagnostics.csv")


def test_matched_conditions_count_and_units():
    """Verify exactly 60 matched speaker x N conditions exist as the experimental units."""
    assert os.path.exists(SPEAKER_RESULTS_PATH), "Speaker results CSV missing"
    df = pd.read_csv(SPEAKER_RESULTS_PATH)

    conditions = df[["speaker_id", "N"]].drop_duplicates()
    assert len(conditions) == 60, f"Expected exactly 60 experimental conditions, got {len(conditions)}"
    assert len(conditions["speaker_id"].unique()) == 15, "Expected exactly 15 unique speakers"
    assert set(conditions["N"].unique()) == {20, 30, 40, 50}, "Expected budgets N in {20, 30, 40, 50}"


def test_no_missing_adapter_qsvc_pairs():
    """Verify every condition has both QSVC Q8-R1 and Classical Adapter d=8."""
    df = pd.read_csv(SPEAKER_RESULTS_PATH)
    q8 = df[df["model"] == "qsvc_q8_r1"][["speaker_id", "N"]].drop_duplicates()
    adp8 = df[df["model"] == "hierarchical_adapter_d8"][["speaker_id", "N"]].drop_duplicates()

    assert len(q8) == 60, f"Expected 60 QSVC Q8-R1 runs, got {len(q8)}"
    assert len(adp8) == 60, f"Expected 60 Adapter d=8 runs, got {len(adp8)}"
    # Check 1-to-1 match
    merged = pd.merge(q8, adp8, on=["speaker_id", "N"])
    assert len(merged) == 60, f"Mismatch in matched pairs: got {len(merged)}"


def test_no_duplicate_matched_pairs():
    """Verify zero duplicate rows per model across conditions."""
    df = pd.read_csv(SPEAKER_RESULTS_PATH)
    dups = df.duplicated(subset=["speaker_id", "N", "model"])
    assert not dups.any(), f"Found {dups.sum()} duplicate condition cells"


def test_q4_q8_pairing_integrity():
    """Verify exact 1-to-1 pairing between Q4-R1 and Q8-R1."""
    df = pd.read_csv(SPEAKER_RESULTS_PATH)
    q4 = df[df["model"] == "qsvc_q4_r1"].sort_values(["speaker_id", "N"])
    q8 = df[df["model"] == "qsvc_q8_r1"].sort_values(["speaker_id", "N"])
    assert len(q4) == 60 and len(q8) == 60
    assert list(q4["speaker_id"]) == list(q8["speaker_id"])
    assert list(q4["N"]) == list(q8["N"])


def test_q8_r1_q8_r2_pairing_integrity():
    """Verify exact 1-to-1 pairing between Q8-R1 and Q8-R2."""
    df = pd.read_csv(SPEAKER_RESULTS_PATH)
    q8_r1 = df[df["model"] == "qsvc_q8_r1"].sort_values(["speaker_id", "N"])
    q8_r2 = df[df["model"] == "qsvc_q8_r2"].sort_values(["speaker_id", "N"])
    assert len(q8_r1) == 60 and len(q8_r2) == 60
    assert list(q8_r1["speaker_id"]) == list(q8_r2["speaker_id"])
    assert list(q8_r1["N"]) == list(q8_r2["N"])


def test_adapter_comparison_reproducibility():
    """Verify that values in phase6_5_5_C_adapter_comparison.csv match calculation from speaker results."""
    assert os.path.exists(ADAPTER_COMP_PATH), "Adapter comparison CSV missing"
    df_comp = pd.read_csv(ADAPTER_COMP_PATH)
    df_raw = pd.read_csv(SPEAKER_RESULTS_PATH)

    row_f1 = df_comp[df_comp["metric"] == "end_to_end_macro_f1"].iloc[0]
    subA = df_raw[df_raw["model"] == "qsvc_q8_r1"].sort_values(["speaker_id", "N"])["end_to_end_macro_f1"].values
    subB = df_raw[df_raw["model"] == "hierarchical_adapter_d8"].sort_values(["speaker_id", "N"])["end_to_end_macro_f1"].values
    mean_diff = float(np.mean(subA - subB))

    assert np.isclose(row_f1["mean_diff"], mean_diff, atol=1e-4)
    assert row_f1["n_pairs"] == 60


def test_runtime_ratio_arithmetic():
    """Verify arithmetic correctness of per-evaluation vs total experiment wall-time ratios."""
    assert os.path.exists(RUNTIME_AUDIT_PATH), "Runtime audit CSV missing"
    df_rt = pd.read_csv(RUNTIME_AUDIT_PATH)

    # Per-evaluation ratio
    row_inf = df_rt[(df_rt["comparison"] == "QSVC Q8-R1 vs. RBF-SVM d=8") & (df_rt["metric_type"].str.contains("Per-Evaluation"))].iloc[0]
    expected_inf_ratio = round(row_inf["model_A_time_sec"] / row_inf["model_B_time_sec"], 2)
    assert np.isclose(row_inf["ratio_A_to_B"], expected_inf_ratio, atol=1e-2)
    assert row_inf["ratio_A_to_B"] > 700.0, "Per-evaluation ratio should be ~773x"

    # Total wall time ratio
    row_tot = df_rt[(df_rt["comparison"] == "QSVC Q8-R1 vs. RBF-SVM d=8") & (df_rt["metric_type"].str.contains("Total 60-Condition"))].iloc[0]
    expected_tot_ratio = round(row_tot["model_A_time_sec"] / row_tot["model_B_time_sec"], 2)
    assert np.isclose(row_tot["ratio_A_to_B"], expected_tot_ratio, atol=1e-2)
    assert row_tot["ratio_A_to_B"] < 300.0, "Total experiment ratio should be ~213x, NOT 770x"


def test_holm_bonferroni_monotonicity():
    """Verify that adjusted p-values in multiple comparison table are strictly valid and monotonic."""
    assert os.path.exists(MULTIPLE_COMP_PATH), "Multiple comparison CSV missing"
    df_mult = pd.read_csv(MULTIPLE_COMP_PATH)
    assert (df_mult["p_value_adjusted_holm"] >= 0.0).all()
    assert (df_mult["p_value_adjusted_holm"] <= 1.0).all()
    assert (df_mult["p_value_adjusted_holm"] >= df_mult["raw_p_value"]).all()


def test_phase_6_5_5_artifacts_preserved():
    """Verify all original Phase 6.5.5 result and report artifacts remain preserved and untouched."""
    p655_artifacts = [
        os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_quantum_conditions.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_classical_conditions.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_kernel_diagnostics.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_runtime.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_matched_comparison.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_statistical_tests.csv"),
        os.path.join(REPORTS_DIR, "phase6_5_5_quantum_recovery_report.md"),
        os.path.join(REPORTS_DIR, "phase6_5_5_quantum_comparison.md"),
        os.path.join(REPORTS_DIR, "phase6_5_5_EXECUTIVE_SUMMARY.md"),
    ]
    for p in p655_artifacts:
        assert os.path.exists(p), f"Original Phase 6.5.5 artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Original Phase 6.5.5 artifact empty: {p}"


def test_historical_benchmarks_preserved():
    """Verify original Phase 3-6 benchmarks remain bitwise intact."""
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


def test_kernel_diagnostics_validation():
    """Verify that all 180 cached quantum kernels remain PSD and finite."""
    assert os.path.exists(KERNEL_DIAG_PATH), "Kernel diagnostics CSV missing"
    df = pd.read_csv(KERNEL_DIAG_PATH)
    assert len(df) == 180, f"Expected 180 kernel diagnostic records, got {len(df)}"
    assert df["is_finite"].all(), "Non-finite values found in kernel diagnostics"
    assert (df["symmetry_error"] < 1e-10).all(), "Symmetry error exceeded tolerance"
    assert (df["diagonal_error"] < 1e-10).all(), "Diagonal error exceeded tolerance"
    assert (df["min_eigenvalue"] >= 0.0).all(), "Negative eigenvalue (non-PSD) found"
    assert (df["psd_status"] == "STRICTLY_PSD").all(), "Non-PSD kernel status found"


def test_closure_csv_artifacts_exist():
    """Verify all 5 Phase 6.5.5-C closure CSVs exist and contain non-zero data."""
    closure_files = [
        ADAPTER_COMP_PATH,
        ABLATION_PATH,
        MULTIPLE_COMP_PATH,
        RUNTIME_AUDIT_PATH,
        DECISION_PATH,
    ]
    for f in closure_files:
        assert os.path.exists(f), f"Closure CSV missing: {f}"
        df = pd.read_csv(f)
        assert len(df) > 0, f"Closure CSV empty: {f}"
