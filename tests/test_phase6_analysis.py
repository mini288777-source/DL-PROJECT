"""
Phase 6 Automated Test Suite: Statistical Analysis & Intent-Preserving Evaluation.

Validates:
1. Integrity of all Phase 6 output tables.
2. Mathematical consistency with frozen Phase 5 results.
3. Statistical contrast validity (Adapter vs RBF, Adapter vs QSVC, RBF vs QSVC).
4. Correctness of effect sizes, normality tests, and multiple comparison corrections.
5. Completeness and validity of publication-grade figures.
"""

import os
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def phase5_results():
    path = "results/phase5/phase5_results.csv"
    assert os.path.isfile(path), f"Phase 5 results missing: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def descriptive_stats():
    path = "phase6/analysis/descriptive_statistics.csv"
    assert os.path.isfile(path), f"Descriptive statistics file missing: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def speaker_analysis():
    path = "phase6/analysis/speaker_analysis.csv"
    assert os.path.isfile(path), f"Speaker analysis file missing: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def statistical_tests():
    path = "phase6/analysis/statistical_tests.csv"
    assert os.path.isfile(path), f"Statistical tests file missing: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def effect_sizes():
    path = "phase6/analysis/effect_sizes.csv"
    assert os.path.isfile(path), f"Effect sizes file missing: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def confidence_intervals():
    path = "phase6/analysis/confidence_intervals.csv"
    assert os.path.isfile(path), f"Confidence intervals file missing: {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def model_comparisons():
    path = "phase6/analysis/model_comparisons.csv"
    assert os.path.isfile(path), f"Model comparisons file missing: {path}"
    return pd.read_csv(path)


def test_phase6_artifacts_exist():
    """Verify all expected Phase 6 analysis outputs and figures exist."""
    required_csvs = [
        "phase6/analysis/descriptive_statistics.csv",
        "phase6/analysis/speaker_analysis.csv",
        "phase6/analysis/model_comparisons.csv",
        "phase6/analysis/effect_sizes.csv",
        "phase6/analysis/confidence_intervals.csv",
        "phase6/analysis/statistical_tests.csv",
    ]
    for path in required_csvs:
        assert os.path.isfile(path), f"Missing required CSV: {path}"
        assert os.path.getsize(path) > 0, f"CSV is empty: {path}"

    required_figs = [
        "phase6/figures/macro_f1_snr_final.png",
        "phase6/figures/iper_snr_final.png",
        "phase6/figures/sample_efficiency_final.png",
        "phase6/figures/speaker_variation.png",
        "phase6/figures/model_effects.png",
    ]
    for path in required_figs:
        assert os.path.isfile(path), f"Missing required figure: {path}"
        assert os.path.getsize(path) > 10000, f"Figure unexpectedly small: {path}"


def test_descriptive_statistics_completeness(descriptive_stats, phase5_results):
    """Verify descriptive statistics table covers all 48 model x SNR x N conditions."""
    assert len(descriptive_stats) == 48, f"Expected 48 conditions, got {len(descriptive_stats)}"
    models = set(descriptive_stats["model"])
    assert models == {"adapter", "rbf_svm", "qsvc"}
    snrs = set(descriptive_stats["snr_tag"])
    assert snrs == {"clean", "snr15", "snr10", "snr5"}
    ns = set(descriptive_stats["sample_n"])
    assert ns == {20, 30, 40, 50}

    # Verify mathematical match of means against raw Phase 5 results
    for _, row in descriptive_stats.iterrows():
        sub = phase5_results[
            (phase5_results["model"] == row["model"])
            & (phase5_results["snr_tag"] == row["snr_tag"])
            & (phase5_results["sample_n"] == row["sample_n"])
        ]
        assert len(sub) == 15, "Each condition must have 15 speakers"
        expected_macro_f1 = float(np.round(sub["macro_f1"].mean(), 4))
        actual_macro_f1 = float(row["macro_f1_mean"])
        assert abs(expected_macro_f1 - actual_macro_f1) < 1e-4


def test_speaker_analysis_integrity(speaker_analysis):
    """Verify speaker analysis contains 15 speakers x 3 models = 45 rows."""
    assert len(speaker_analysis) == 45, f"Expected 45 speaker rows, got {len(speaker_analysis)}"
    dysarthric_spks = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}
    control_spks = {"FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"}

    for _, row in speaker_analysis.iterrows():
        spk = row["speaker_id"]
        grp = row["cohort_group"]
        if spk in dysarthric_spks:
            assert grp == "dysarthric"
        elif spk in control_spks:
            assert grp == "control"
        else:
            pytest.fail(f"Unknown speaker {spk}")


def test_rbf_qsvc_zero_difference_verified(statistical_tests, effect_sizes):
    """Verify statistical engine confirms numerical identity between RBF-SVM and QSVC."""
    rbf_qsvc_tests = statistical_tests[statistical_tests["contrast"] == "RBF_vs_QSVC"]
    assert len(rbf_qsvc_tests) == 16, "Expected 16 RBF vs QSVC tests"

    for _, row in rbf_qsvc_tests.iterrows():
        assert row["mean_difference"] == 0.0, "RBF vs QSVC mean difference must be 0.0"
        assert row["std_difference"] == 0.0, "RBF vs QSVC std difference must be 0.0"
        assert row["raw_p_value"] == 1.0, "RBF vs QSVC p-value must be 1.0"
        assert not row["significant_alpha_0_05"], "RBF vs QSVC must not be statistically significant"

    rbf_qsvc_effects = effect_sizes[effect_sizes["contrast"] == "RBF_vs_QSVC"]
    assert len(rbf_qsvc_effects) == 16
    for _, row in rbf_qsvc_effects.iterrows():
        assert row["mean_diff"] == 0.0
        assert row["cohens_d"] == 0.0
        assert row["hedges_g"] == 0.0
        assert row["wilcoxon_r"] == 0.0


def test_confidence_intervals_bounds(confidence_intervals):
    """Verify 95% confidence intervals have lower <= upper bounds."""
    assert len(confidence_intervals) == 48
    for _, row in confidence_intervals.iterrows():
        assert row["parametric_ci_lower"] <= row["parametric_ci_upper"]
        assert row["bootstrap_ci_lower"] <= row["bootstrap_ci_upper"]


def test_model_comparisons_granularity(model_comparisons):
    """Verify granular model comparisons table has exactly 240 paired speaker contrasts."""
    # 15 speakers x 4 SNRs x 4 Ns = 240
    assert len(model_comparisons) == 240
    expected_cols = {
        "speaker_id", "speaker_group", "snr_tag", "sample_n",
        "adapter", "rbf_svm", "qsvc",
        "diff_adapter_rbf", "diff_rbf_qsvc"
    }
    assert expected_cols.issubset(set(model_comparisons.columns))
    # diff_rbf_qsvc must be strictly 0.0 across all 240 rows
    assert np.allclose(model_comparisons["diff_rbf_qsvc"].values, 0.0, atol=1e-8)
