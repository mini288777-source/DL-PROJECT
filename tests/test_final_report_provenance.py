"""
Test Final Report Provenance.
Verifies that primary quantitative claims in the final reports match source result artifacts exactly.
"""

import os
import pandas as pd
import pytest


def test_primary_matched_comparison_provenance():
    tests_csv = "results/phase6_5_5_statistical_tests.csv"
    assert os.path.exists(tests_csv), f"Missing {tests_csv}"
    df = pd.read_csv(tests_csv)
    
    # P1: Macro-F1
    p1 = df[df["comparison_id"] == "P1"].iloc[0]
    assert round(p1["mean_diff"], 4) == -0.0036
    assert round(p1["p_value_adjusted_holm"], 4) == 0.3428
    
    # P2: Emergency Recall
    p2 = df[df["comparison_id"] == "P2"].iloc[0]
    assert round(p2["mean_diff"] * 100, 2) == -4.37
    assert round(p2["p_value_adjusted_holm"], 4) == 0.0689
    
    # P3: FER
    p3 = df[df["comparison_id"] == "P3"].iloc[0]
    assert round(p3["mean_diff"] * 100, 2) == -1.20
    assert round(p3["p_value_adjusted_holm"], 4) == 0.9505
    
    # P4: IPER
    p4 = df[df["comparison_id"] == "P4"].iloc[0]
    assert round(p4["mean_diff"] * 100, 2) == -3.44
    assert round(p4["p_value_adjusted_holm"], 4) == 0.0582


def test_reconciled_iper_provenance():
    iper_csv = "results/final_repair/phase6_5_5_IPER_reconciled.csv"
    assert os.path.exists(iper_csv), f"Missing {iper_csv}"
    df = pd.read_csv(iper_csv)
    
    p4_rate = df[df["comparison_id"] == "P4_RATE"].iloc[0]
    assert p4_rate["diff_mean"] == -0.0344
    assert round(p4_rate["holm_p_value"], 4) == 0.0582
    
    p4_error = df[df["comparison_id"] == "P4_ERROR"].iloc[0]
    assert p4_error["diff_mean"] == 0.0344
    assert round(p4_error["holm_p_value"], 4) == 0.0582
