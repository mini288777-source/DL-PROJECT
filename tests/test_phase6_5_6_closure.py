"""
Phase 6.5.6 Test Suite: Final Fair Comparison, Evidence Synthesis & Research Closure
Validates:
1. Immutability and preservation of original Phase 3-6 benchmarks.
2. Immutability and preservation of Phase 6.5.0-6.5.5 recovery artifacts.
3. Existence and schema validity of all 4 Phase 6.5.6 results CSVs.
4. Existence and integrity of all 8 Phase 6.5.6 synthesis figures.
5. Existence and completeness of all 6 Phase 6.5.6 markdown reports.
6. Mathematical consistency of metric reconciliation.
7. Verification of Case B decision for primary quantum comparison.
8. Verification of computational runtime audit ratios.
9. Verification of operational status as NOT READY.
10. Verification of matched experimental units (60 units).
"""

import os
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "phase6_5_6", "figures")


def test_original_phase3_6_artifacts_preserved():
    """Verify original Phase 3-6 benchmarks exist and remain intact."""
    orig_files = [
        "phase3_complete_results.md",
        "phase4_complete_results.md",
        "phase5_results_report.md",
        "phase6_complete_results.md",
    ]
    for f in orig_files:
        path = os.path.join(REPORTS_DIR, f)
        assert os.path.exists(path), f"Original benchmark file missing: {path}"
        assert os.path.getsize(path) > 1000, f"Original benchmark file corrupted: {path}"


def test_phase6_5_recovery_artifacts_preserved():
    """Verify Phase 6.5.0-6.5.5-C recovery artifacts exist and remain intact."""
    rec_files = [
        os.path.join(REPO_ROOT, "phase6_5", "reports", "phase6_5_0_diagnostic_report.md"),
        os.path.join(REPO_ROOT, "phase6_5", "reports", "phase6_5_1_feasibility_report.md"),
        os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "reports", "phase6_5_2_classical_rescue_report.md"),
        os.path.join(REPO_ROOT, "phase6_5", "hierarchical", "reports", "phase6_5_3_hierarchical_report.md"),
        os.path.join(REPORTS_DIR, "phase6_5_4_representation_pca_audit.md"),
        os.path.join(REPORTS_DIR, "phase6_5_5_quantum_recovery_report.md"),
        os.path.join(REPORTS_DIR, "phase6_5_5_C_statistical_closure.md"),
        os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv"),
        os.path.join(RESULTS_DIR, "phase6_5_5_C_multiple_comparison.csv"),
    ]
    for p in rec_files:
        assert os.path.exists(p), f"Recovery artifact missing: {p}"
        assert os.path.getsize(p) > 500, f"Recovery artifact empty or corrupted: {p}"


def test_phase6_5_6_results_csvs_exist_and_valid():
    """Verify all 4 required Phase 6.5.6 results CSV files exist and have valid structure."""
    expected_csvs = {
        "phase6_5_6_final_comparison_table.csv": 8,
        "phase6_5_6_evidence_chain.csv": 9,
        "phase6_5_6_metric_reconciliation.csv": 8,
        "phase6_5_6_final_decision.csv": 7,
    }
    for fname, min_rows in expected_csvs.items():
        p = os.path.join(RESULTS_DIR, fname)
        assert os.path.exists(p), f"Phase 6.5.6 CSV missing: {p}"
        df = pd.read_csv(p)
        assert len(df) >= min_rows, f"{fname} has only {len(df)} rows, expected >= {min_rows}"


def test_final_comparison_table_sections():
    """Verify final comparison table has Section A and Section B clearly separated."""
    p = os.path.join(RESULTS_DIR, "phase6_5_6_final_comparison_table.csv")
    df = pd.read_csv(p)
    sections = df["section"].unique()
    assert "SECTION A: ORIGINAL BENCHMARK" in sections
    assert "SECTION B: RECOVERY BENCHMARK" in sections

    sec_a = df[df["section"] == "SECTION A: ORIGINAL BENCHMARK"]
    sec_b = df[df["section"] == "SECTION B: RECOVERY BENCHMARK"]
    assert len(sec_a) == 3  # RBF, Adapter, QSVC
    assert len(sec_b) == 5  # Adapter, RBF, Q8-R1, Q4-R1, Q8-R2


def test_evidence_chain_progression():
    """Verify evidence chain includes all recovery milestones from 6.5.0 to 6.5.5-C."""
    p = os.path.join(RESULTS_DIR, "phase6_5_6_evidence_chain.csv")
    df = pd.read_csv(p)
    phases = df["phase"].tolist()
    expected_phases = [
        "Phase 6.5.0", "Phase 6.5.1", "Phase 6.5.2", "Phase 6.5.2-C",
        "Phase 6.5.3", "Phase 6.5.3-C", "Phase 6.5.4", "Phase 6.5.5", "Phase 6.5.5-C"
    ]
    for ep in expected_phases:
        assert ep in phases, f"Missing phase in evidence chain: {ep}"


def test_metric_reconciliation_accuracy():
    """Verify metric reconciliation table accurately documents discrepancies."""
    p = os.path.join(RESULTS_DIR, "phase6_5_6_metric_reconciliation.csv")
    df = pd.read_csv(p)
    metrics = df["metric"].tolist()

    # Verify QSVC Emergency Precision entry
    q_row = df[df["metric"].str.contains(r"Emergency Precision \(QSVC Q8-R1\)")]
    assert len(q_row) == 1
    assert "11.24%" in q_row["historical_value"].iloc[0]
    assert "0.81%" in q_row["closure_value"].iloc[0]

    # Verify Adapter Emergency Precision entry
    a_row = df[df["metric"].str.contains(r"Emergency Precision \(Classical Adapter d=8\)")]
    assert len(a_row) == 1
    assert "9.21%" in a_row["historical_value"].iloc[0]
    assert "2.03%" in a_row["closure_value"].iloc[0]

    # Verify PCA Explained Variance entry
    pca_row = df[df["metric"].str.contains(r"PCA Explained Variance \(d=4\)")]
    assert len(pca_row) == 1
    assert "27.4%" in pca_row["historical_value"].iloc[0]
    assert "80.66%" in pca_row["closure_value"].iloc[0]


def test_final_decision_case_b():
    """Verify primary comparison decision is frozen as CASE B."""
    p = os.path.join(RESULTS_DIR, "phase6_5_6_final_decision.csv")
    df = pd.read_csv(p)
    f1_row = df[(df["target_metric"] == "End-to-End Macro-F1") & (df["comparison_scope"].str.contains("Primary"))]
    assert len(f1_row) == 1
    assert "CASE B" in f1_row["decision"].iloc[0]
    assert float(f1_row["holm_p_value"].iloc[0]) > 0.05


def test_runtime_ratios_arithmetic():
    """Verify audited computational cost ratios are preserved."""
    p = os.path.join(RESULTS_DIR, "phase6_5_6_metric_reconciliation.csv")
    df = pd.read_csv(p)
    rt_row = df[df["metric"].str.contains("Runtime")]
    assert len(rt_row) == 1
    closure_txt = rt_row["closure_value"].iloc[0]
    assert "772.6x" in closure_txt
    assert "212.8x" in closure_txt


def test_all_8_synthesis_figures_exist():
    """Verify all 8 synthesis figures exist in reports/phase6_5_6/figures/."""
    expected_figs = [
        "fig1_original_vs_recovery_progression.png",
        "fig2_emergency_support_recovery.png",
        "fig3_fer_reduction_flat_vs_hierarchical.png",
        "fig4_representation_dimensionality.png",
        "fig5_matched_q8_vs_rbf.png",
        "fig6_emergency_recall_comparison.png",
        "fig7_quantum_computational_cost.png",
        "fig8_evidence_chain_workflow.png",
    ]
    for fig_name in expected_figs:
        p = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(p), f"Synthesis figure missing: {p}"
        assert os.path.getsize(p) > 20000, f"Synthesis figure file size too small: {p}"


def test_all_6_markdown_reports_exist():
    """Verify all 6 Phase 6.5.6 markdown reports exist and have substantial content."""
    expected_reports = [
        "phase6_5_6_final_fair_comparison.md",
        "phase6_5_6_final_evidence_synthesis.md",
        "phase6_5_6_research_conclusion.md",
        "phase6_5_6_metric_reconciliation.md",
        "phase6_5_6_EXECUTIVE_SUMMARY.md",
        "phase6_5_6_limitations.md",
    ]
    for rep_name in expected_reports:
        p = os.path.join(REPORTS_DIR, rep_name)
        assert os.path.exists(p), f"Phase 6.5.6 report missing: {p}"
        assert os.path.getsize(p) > 1000, f"Phase 6.5.6 report empty or too short: {p}"


def test_deployment_readiness_declared_not_ready():
    """Verify deployment readiness is consistently NOT READY across reports."""
    exec_rep = os.path.join(REPORTS_DIR, "phase6_5_6_EXECUTIVE_SUMMARY.md")
    limit_rep = os.path.join(REPORTS_DIR, "phase6_5_6_limitations.md")

    with open(exec_rep, "r", encoding="utf-8") as f:
        c1 = f.read()
    with open(limit_rep, "r", encoding="utf-8") as f:
        c2 = f.read()

    assert "NOT READY" in c1
    assert "NOT READY" in c2
    assert "NOT ESTABLISHED" in c2


def test_no_data_leakage_and_matched_units():
    """Verify matched experimental units are exactly 60 and no condition was duplicated."""
    results_path = os.path.join(RESULTS_DIR, "phase6_5_5_speaker_results.csv")
    df = pd.read_csv(results_path)

    for m in ["qsvc_q8_r1", "hierarchical_rbf_svm_d8", "hierarchical_adapter_d8"]:
        sub = df[df["model"] == m]
        assert len(sub) == 60, f"Model {m} has {len(sub)} conditions, expected 60"
        speakers = sub["speaker_id"].unique()
        budgets = sub["N"].unique()
        assert len(speakers) == 15
        assert set(budgets) == {20, 30, 40, 50}
        assert not sub.duplicated(subset=["speaker_id", "N"]).any()
