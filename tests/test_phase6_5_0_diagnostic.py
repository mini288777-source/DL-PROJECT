"""
Unit and Integration Tests for Subphase 6.5.0 — Forensic Diagnostic & Recovery Plan
"""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE6_5_DIR = REPO_ROOT / "phase6_5"
REPORTS_DIR = PHASE6_5_DIR / "reports"
ANALYSIS_DIR = PHASE6_5_DIR / "analysis"


def test_reports_exist_and_complete():
    """Verify that both required reports exist and contain required content."""
    diag_report = REPORTS_DIR / "phase6_5_0_diagnostic_report.md"
    plan_report = REPORTS_DIR / "PHASE6_5_RECOVERY_PLAN.md"

    assert diag_report.exists(), f"Missing {diag_report}"
    assert plan_report.exists(), f"Missing {plan_report}"

    diag_text = diag_report.read_text(encoding="utf-8")
    required_sections = [
        "## 1. Problem Statement",
        "## 2. Current Failure Symptoms",
        "## 3. Frozen Baseline",
        "## 4. Data Feasibility",
        "## 5. Class Distribution Diagnosis",
        "## 6. Kernel Collapse Diagnosis",
        "## 7. Representation Diagnosis",
        "## 8. Quantum Configuration Audit",
        "## 9. Likely Root Causes",
        "## 10. Root-Cause Ranking",
        "## 11. Recovery Strategy",
        "## 12. Planned Subphases",
        "## 13. Acceptance Criteria",
    ]
    for section in required_sections:
        assert section in diag_text, f"Missing section '{section}' in diagnostic report"

    plan_text = plan_report.read_text(encoding="utf-8")
    for subphase in ["6.5.1", "6.5.2", "6.5.3", "6.5.4", "6.5.5", "6.5.6", "6.5.7", "6.5.8"]:
        assert subphase in plan_text, f"Missing subphase {subphase} in recovery plan"


def test_adaptation_class_support():
    """Verify adaptation class support analysis table."""
    csv_file = ANALYSIS_DIR / "adaptation_class_support.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    df = pd.read_csv(csv_file)
    assert len(df) == 60, f"Expected 60 rows (15 speakers x 4 budgets), got {len(df)}"
    assert "speaker_id" in df.columns
    assert "sample_n" in df.columns
    assert "adapt_GENERAL_SPEECH" in df.columns
    assert "n_emergency_classes_adapt" in df.columns
    assert "n_zero_support_classes" in df.columns

    # Verify that in all adaptation sets, GENERAL_SPEECH makes up the vast majority (>= 80%)
    gen_fractions = df["adapt_GENERAL_SPEECH"] / df["sample_n"]
    assert (gen_fractions >= 0.80).all(), "GENERAL_SPEECH should constitute >= 80% across all adaptation sets"


def test_emergency_support_matrix():
    """Verify emergency support matrix."""
    csv_file = ANALYSIS_DIR / "emergency_support.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    df = pd.read_csv(csv_file)
    assert len(df) == 360, f"Expected 360 rows (60 cells x 6 emergency classes), got {len(df)}"
    assert "speaker_id" in df.columns
    assert "emergency_class" in df.columns
    assert "adapt_support" in df.columns
    assert "zero_support" in df.columns

    # Verify overall high zero-support rate across emergency classes (> 50%)
    overall_zero_rate = df["zero_support"].mean()
    assert overall_zero_rate > 0.50, f"Expected > 50% zero-support, got {overall_zero_rate:.3f}"


def test_speaker_feasibility_matrix():
    """Verify direct 7-class feasibility classification."""
    csv_file = ANALYSIS_DIR / "speaker_feasibility_matrix.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    df = pd.read_csv(csv_file)
    if "recovery_protocol" in df.columns:
        track_df = df[df["recovery_protocol"] == "track_a"]
        assert len(track_df) == 60, f"Expected 60 rows for track_a, got {len(track_df)}"
        target_df = track_df
    else:
        assert len(df) == 60, f"Expected 60 rows, got {len(df)}"
        target_df = df

    # Verify feasibility distribution matches diagnosis
    col = "feasibility_tier" if "feasibility_tier" in target_df.columns else "feasibility"
    at_n20 = target_df[target_df["sample_n"] == 20]
    assert (at_n20[col] != "FEASIBLE").all(), "At N=20, zero cells should be fully FEASIBLE"


def test_prediction_distribution_collapse():
    """Verify that predictions from RBF-SVM and QSVC collapse to GENERAL_SPEECH on clean data."""
    csv_file = ANALYSIS_DIR / "prediction_distribution_by_model.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    df = pd.read_csv(csv_file)
    clean_df = df[df["snr_tag"] == "clean"]

    rbf_clean = clean_df[(clean_df["model"] == "rbf_svm") & (clean_df["predicted_class"] == "GENERAL_SPEECH")]
    qsvc_clean = clean_df[(clean_df["model"] == "qsvc") & (clean_df["predicted_class"] == "GENERAL_SPEECH")]

    assert len(rbf_clean) == 1, "RBF clean should have GENERAL_SPEECH row"
    assert rbf_clean.iloc[0]["fraction"] == 1.0, "RBF clean should be 100% GENERAL_SPEECH"

    assert len(qsvc_clean) == 1, "QSVC clean should have GENERAL_SPEECH row"
    assert qsvc_clean.iloc[0]["fraction"] == 1.0, "QSVC clean should be 100% GENERAL_SPEECH"


def test_pca_explained_variance():
    """Verify PCA cumulative explained variance."""
    csv_file = ANALYSIS_DIR / "pca_explained_variance.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    df = pd.read_csv(csv_file)
    pca4 = df[df["n_components"] == 4]
    assert len(pca4) == 1
    # 4D PCA variance should be approximately 73.2%
    ev_val = pca4.iloc[0]["cumulative_explained_variance"]
    assert 0.70 <= ev_val <= 0.76, f"Expected ~0.73, got {ev_val}"


def test_quantum_config_audit():
    """Verify quantum execution parameters audit."""
    csv_file = ANALYSIS_DIR / "quantum_config_audit.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    df = pd.read_csv(csv_file)
    row = df.iloc[0]

    assert row["qubits"] == 4
    assert row["reps"] == 1
    assert row["entanglement"] == "linear"
    assert row["feature_map"] == "zz"
    assert row["kernel"] == "FidelityQuantumKernel"
    assert row["simulator"] == "Qiskit Aer Statevector"
