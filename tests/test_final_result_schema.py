"""
Test Final Result Schema.
Verifies the schema, columns, and data integrity of canonical result artifacts.
"""

import os
import pandas as pd
import pytest


def test_final_claim_audit_schema():
    claim_csv = "results/final_repair/final_claim_audit.csv"
    assert os.path.exists(claim_csv), f"Missing {claim_csv}"
    df = pd.read_csv(claim_csv)
    
    required_cols = [
        "claim_id",
        "file",
        "location",
        "original_claim",
        "classification",
        "correction_required",
        "corrected_claim",
        "evidence_source",
        "severity",
        "status",
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column in claim audit: {c}"
        
    assert len(df) >= 10, f"Expected at least 10 audited claims, got {len(df)}"
    assert all(df["status"] == "CORRECTED")


def test_final_consistency_matrix_schema():
    matrix_csv = "results/final_repair/final_consistency_matrix.csv"
    assert os.path.exists(matrix_csv), f"Missing {matrix_csv}"
    df = pd.read_csv(matrix_csv)
    
    required_cols = [
        "category",
        "source_A",
        "source_B",
        "match",
        "resolution",
        "authoritative_source",
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing column in consistency matrix: {c}"
        
    for m in df["match"]:
        assert m in ["MATCHED", "EXPLICIT HISTORICAL DIFFERENCE"], f"Invalid match status: {m}"


def test_stage1_reconciliation_schema():
    s1_csv = "results/final_repair/stage1_balanced_accuracy_reconciliation.csv"
    assert os.path.exists(s1_csv), f"Missing {s1_csv}"
    df = pd.read_csv(s1_csv)
    
    assert "TP" in df.columns
    assert "TN" in df.columns
    assert "FP" in df.columns
    assert "FN" in df.columns
    assert "stage1_emergency_recall" in df.columns
    assert "stage1_specificity" in df.columns
    assert "stage1_balanced_accuracy" in df.columns
    assert len(df) > 60
