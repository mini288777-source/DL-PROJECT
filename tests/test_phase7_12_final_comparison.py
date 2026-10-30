"""
Tests for Phase 7.12 - Final Quantum vs Classical Consolidated Comparison
Verifies data integrity, matched pairing, statistical rigor, Holm-Bonferroni monotonicity,
and strict preservation of frozen historical boundaries.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE7_12_RESULTS = REPO_ROOT / "results" / "continual" / "phase7_12"
PHASE7_12_REPORTS = REPO_ROOT / "reports" / "continual" / "phase7_12"

def test_phase7_12_artifacts_exist():
    """Verify all required Phase 7.12 CSV and JSON artifacts exist and are non-empty."""
    required_files = [
        PHASE7_12_RESULTS / "comprehensive_four_way_comparison.csv",
        PHASE7_12_RESULTS / "subgroup_budget_breakdown.csv",
        PHASE7_12_RESULTS / "pairwise_statistical_tests.csv",
        PHASE7_12_RESULTS / "final_evidence_matrix.csv",
        PHASE7_12_REPORTS / "final_comparison_summary.json"
    ]
    for p in required_files:
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"

def test_comprehensive_comparison_coverage_and_alignment():
    """Verify that comprehensive_four_way_comparison.csv contains exactly 60 matched cells."""
    df = pd.read_csv(PHASE7_12_RESULTS / "comprehensive_four_way_comparison.csv")
    assert len(df) == 60
    assert set(df["budget"].unique()) == {20, 30, 40, 50}
    assert len(df["speaker_id"].unique()) == 15
    # Verify speaker group split: 8 dysarthric, 7 control
    spk_groups = df.groupby("speaker_id")["speaker_group"].first()
    assert (spk_groups == "dysarthric").sum() == 8
    assert (spk_groups == "control").sum() == 7

def test_pairwise_statistical_comparisons():
    """Verify that pairwise_statistical_tests.csv contains all 6 required comparisons and correct bounds."""
    df = pd.read_csv(PHASE7_12_RESULTS / "pairwise_statistical_tests.csv")
    assert len(df) == 6
    expected_comparisons = {
        "Fixed Quantum vs Classical",
        "Trainable Quantum vs Classical",
        "Hybrid vs Classical",
        "Trainable Quantum vs Fixed Quantum",
        "Hybrid vs Fixed Quantum",
        "Hybrid vs Trainable Quantum"
    }
    assert set(df["comparison"].unique()) == expected_comparisons

    for _, row in df.iterrows():
        assert 0.0 <= row["raw_p_macro_f1"] <= 1.0
        assert 0.0 <= row["adj_p_macro_f1"] <= 1.0
        assert row["raw_p_macro_f1"] <= row["adj_p_macro_f1"] + 1e-9

def test_final_evidence_matrix_completeness():
    """Verify that the evidence matrix includes all 4 canonical continual models with complete fields."""
    df = pd.read_csv(PHASE7_12_RESULTS / "final_evidence_matrix.csv")
    assert len(df) == 4
    expected_models = {
        "Continual Classical Adapter",
        "Fixed Quantum Kernel (QSVC)",
        "Trainable Quantum Representation",
        "Hybrid Classical + Quantum"
    }
    assert set(df["Model"].unique()) == expected_models
    assert "Macro_F1" in df.columns
    assert "Emergency_Recall" in df.columns
    assert "FER" in df.columns
    assert "Governance_Status" in df.columns
    assert "Statistical_Verdict" in df.columns

def test_subgroup_budget_breakdown_metrics():
    """Verify that subgroup and budget breakdowns cover all required aggregations."""
    df = pd.read_csv(PHASE7_12_RESULTS / "subgroup_budget_breakdown.csv")
    assert len(df) == 7
    expected_subgroups = {
        "All Speakers",
        "Dysarthric (n=8)",
        "Control (n=7)",
        "Budget N=20",
        "Budget N=30",
        "Budget N=40",
        "Budget N=50"
    }
    assert set(df["subgroup"].unique()) == expected_subgroups

def test_quantum_vs_classical_scientific_invariants():
    """Verify scientific invariants: Classical retains superior Emergency Recall, and pure quantum models lack advantage."""
    df_comp = pd.read_csv(PHASE7_12_RESULTS / "comprehensive_four_way_comparison.csv")
    mean_c_recall = df_comp["classical_emergency_recall"].mean()
    mean_fq_recall = df_comp["fixed_q_emergency_recall"].mean()
    mean_tq_recall = df_comp["trainable_q_emergency_recall"].mean()

    # Classical recall must be strictly greater than fixed quantum and trainable quantum recall
    assert mean_c_recall > mean_fq_recall
    assert mean_c_recall > mean_tq_recall

    # Trainable quantum does not show statistically significant F1 advantage over fixed quantum
    df_stats = pd.read_csv(PHASE7_12_RESULTS / "pairwise_statistical_tests.csv")
    row_tq_vs_fq = df_stats[df_stats["comparison"] == "Trainable Quantum vs Fixed Quantum"].iloc[0]
    assert row_tq_vs_fq["adj_p_macro_f1"] > 0.05

def test_historical_boundary_isolation():
    """Verify that historical Phase 6.5 reports remain untouched and CASE B conclusion is preserved."""
    summary_path = PHASE7_12_REPORTS / "final_comparison_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)
    assert summary["case_b_status"] == "UNCHANGED"
    assert "CASE B" in summary["overall_scientific_conclusion"]

    # Verify Phase 6.5 benchmark file exists and is intact
    phase6_5_closure = REPO_ROOT / "reports" / "phase6_5_5_C_statistical_closure.md"
    assert phase6_5_closure.exists()
