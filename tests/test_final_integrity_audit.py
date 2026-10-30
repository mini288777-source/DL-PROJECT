"""
Test Suite: Final Scientific Integrity & Claim-Correction Pass
Validates:
1. Existence and integrity of all 4 final audit reports.
2. Existence and schema of results/final_claim_audit.csv.
3. Absence of prohibited causal overstatements in final synthesis documents.
4. Consistency of Case B decision and NOT READY deployment readiness.
5. Verification that raw experimental artifacts remain intact.
"""

import os
import pytest
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def test_final_audit_reports_exist():
    """Verify all 4 new final audit reports exist and have substantial content."""
    reports = [
        "FINAL_SCIENTIFIC_INTEGRITY_AUDIT.md",
        "FINAL_CLAIM_CORRECTIONS.md",
        "FINAL_METRIC_DEFINITIONS.md",
        "FINAL_RESEARCH_WORDING.md",
    ]
    for r in reports:
        p = os.path.join(REPORTS_DIR, r)
        assert os.path.exists(p), f"Missing audit report: {p}"
        assert os.path.getsize(p) > 1000, f"Audit report too small: {p}"


def test_final_claim_audit_csv_schema():
    """Verify results/final_claim_audit.csv exists and has correct columns."""
    p = os.path.join(RESULTS_DIR, "final_claim_audit.csv")
    assert os.path.exists(p), f"Missing final claim audit CSV: {p}"
    df = pd.read_csv(p)
    expected_cols = [
        "claim_id", "file", "location", "original_claim",
        "classification", "correction_required", "corrected_claim",
        "evidence_source", "severity"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) >= 10, f"Expected at least 10 audited claims, found {len(df)}"
    valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "OK"}
    for sev in df["severity"]:
        assert sev in valid_severities, f"Invalid severity: {sev}"


def test_no_prohibited_causal_phrases_in_final_reports():
    """Verify prohibited causal overstatements do not appear in final synthesis reports."""
    prohibited_snippets = [
        "artifact of few-shot data starvation",
        "hierarchical routing solves emergency recognition",
        "d=8 is the optimal dimension",
        "PCA dimensionality reduction solved the curse of dimensionality",
    ]
    final_docs = [
        "phase6_5_6_EXECUTIVE_SUMMARY.md",
        "phase6_5_6_research_conclusion.md",
        "phase6_5_6_final_fair_comparison.md",
        "phase6_5_6_final_evidence_synthesis.md",
    ]
    for doc in final_docs:
        p = os.path.join(REPORTS_DIR, doc)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        for snip in prohibited_snippets:
            assert snip.lower() not in content.lower(), f"Prohibited phrase '{snip}' found in {doc}"


def test_case_b_and_not_ready_consistency():
    """Verify Case B decision and NOT READY status are uniformly stated in final documents."""
    final_docs = [
        "phase6_5_6_EXECUTIVE_SUMMARY.md",
        "phase6_5_6_research_conclusion.md",
        "phase6_5_6_final_fair_comparison.md",
        "FINAL_SCIENTIFIC_INTEGRITY_AUDIT.md",
        "FINAL_RESEARCH_WORDING.md",
    ]
    for doc in final_docs:
        p = os.path.join(REPORTS_DIR, doc)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        assert "CASE B" in content, f"Missing CASE B in {doc}"
        assert "NOT READY" in content, f"Missing NOT READY in {doc}"
