"""
Tests for Phase 7.15 - Final Phase-7 Closure & Scientific Integrity Verification.
Validates:
1. Completeness and existence of all Phase 7 documentation suite.
2. Completeness and validity of final consolidated artifacts (CSV/JSON).
3. Preservation and immutability of frozen historical boundaries (Phase 1-6.5 and CASE B).
4. Evidence traceability and research question ledger completeness.
5. Deployment status restricted strictly to RESEARCH DEMO ONLY.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_RESULTS_DIR = REPO_ROOT / "results" / "continual" / "final"
FINAL_REPORTS_DIR = REPO_ROOT / "reports" / "continual" / "final"
RESEARCH_DIR = REPO_ROOT / "research" / "continual"


def test_final_closure_artifacts_exist():
    """Verify all final closure CSV and JSON artifacts exist and are non-empty."""
    required = [
        FINAL_RESULTS_DIR / "phase7_master_evidence_matrix.csv",
        FINAL_RESULTS_DIR / "phase7_research_questions_ledger.csv",
        FINAL_REPORTS_DIR / "phase7_final_closure_summary.json"
    ]
    for p in required:
        assert p.exists(), f"Missing final closure artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"


def test_complete_phase7_markdown_documentation_suite():
    """Verify that all Phase 7 subphase scientific reports exist in research/continual/."""
    expected_reports = [
        "PHASE7_1_EXPERIENCE_STORE.md",
        "PHASE7_2_CONFIDENCE_ELIGIBILITY.md",
        "PHASE7_3_HUMAN_IN_LOOP.md",
        "PHASE7_4_CONTINUAL_CLASSICAL_BASELINE.md",
        "PHASE7_5_REPLAY_FORGETTING.md",
        "PHASE7_6_VALIDATION_ROLLBACK.md",
        "PHASE7_7_DRIFT_RECALIBRATION.md",
        "PHASE7_8_CONTINUAL_QUANTUM_KERNEL.md",
        "PHASE7_9_TRAINABLE_QUANTUM_REPRESENTATION.md",
        "PHASE7_10_HYBRID_CLASSICAL_QUANTUM.md",
        "PHASE7_11_SELF_IMPROVEMENT.md",
        "PHASE7_12_FINAL_COMPARISON.md",
        "PHASE7_13_SAFETY_ROBUSTNESS.md",
        "PHASE7_14_APPLICATION_INTEGRATION.md",
        "PHASE7_FINAL_REPORT.md"
    ]
    for rep in expected_reports:
        p = RESEARCH_DIR / rep
        assert p.exists(), f"Missing Phase 7 report: {rep}"
        assert p.stat().st_size > 500, f"Report {rep} is suspiciously small: {p.stat().st_size} bytes"


def test_research_questions_ledger_completeness():
    """Verify all 7 canonical Phase 7 research questions are evaluated with evidence-based verdicts."""
    df = pd.read_csv(FINAL_RESULTS_DIR / "phase7_research_questions_ledger.csv")
    assert len(df) == 7
    expected_rqs = {"RQ7.1", "RQ7.2", "RQ7.3", "RQ7.4", "RQ7.5", "RQ7.6", "RQ7.7"}
    assert set(df["RQ_ID"].unique()) == expected_rqs

    valid_verdicts = {"SUPPORTED", "NOT SUPPORTED", "PARTIALLY SUPPORTED", "INCONCLUSIVE", "NOT FEASIBLE"}
    for _, row in df.iterrows():
        assert row["Verdict"] in valid_verdicts
        assert len(row["Evidence"]) > 10


def test_master_evidence_matrix_integrity():
    """Verify master evidence matrix contains all 4 canonical models with complete performance fields."""
    df = pd.read_csv(FINAL_RESULTS_DIR / "phase7_master_evidence_matrix.csv")
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


def test_frozen_historical_boundaries_preserved():
    """Verify frozen Phase 1-6.5 artifacts exist and CASE B conclusion remains unchanged."""
    # Check Phase 6.5 report
    p6_5_rep = REPO_ROOT / "reports" / "phase6_5_5_C_statistical_closure.md"
    assert p6_5_rep.exists()

    # Check closure summary JSON
    with open(FINAL_REPORTS_DIR / "phase7_final_closure_summary.json", "r") as f:
        data = json.load(f)

    assert data["frozen_evidence_integrity"]["frozen_case_b_conclusion"] == "UNCHANGED"
    assert "NOT SUPPORTED" in data["scientific_verdicts"]["quantum_overall_verdict"]
    assert data["scientific_verdicts"]["deployment_status"] == "READY FOR RESEARCH DEMO (NOT CLINICALLY VALIDATED)"


def test_deployment_status_not_clinically_validated():
    """Verify deployment status never claims independent clinical readiness."""
    app_cfg = REPO_ROOT / "config" / "continual" / "app_config.yaml"
    with open(app_cfg, "r") as f:
        content = f.read()
    assert "NOT CLINICALLY VALIDATED" in content
    assert "RESEARCH DEMO ONLY" in content
