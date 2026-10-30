"""
Test Final No Hardcoded Metrics.
Ensures report generators do not contain hardcoded synthetic result arrays.
"""

import os
import pytest


def test_no_hardcoded_pca_variances_in_scripts():
    p4_script = "scripts/generate_phase4_reports.py"
    with open(p4_script, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Must NOT have the hardcoded placeholder assignment
    assert "pca_variances = [17.8, 27.4, 34.2, 39.8]" not in content, (
        "Found hardcoded synthetic PCA array in generate_phase4_reports.py!"
    )


def test_reconciled_artifacts_exist_and_populated():
    artifacts = [
        "results/final_repair/phase6_5_5_IPER_reconciled.csv",
        "results/final_repair/stage1_balanced_accuracy_reconciliation.csv",
        "results/final_repair/sample_budget_provenance.csv",
        "results/final_repair/final_claim_audit.csv",
        "results/final_repair/final_consistency_matrix.csv",
    ]
    for art in artifacts:
        assert os.path.exists(art), f"Missing reconciled artifact: {art}"
        assert os.path.getsize(art) > 50, f"Artifact is empty: {art}"
