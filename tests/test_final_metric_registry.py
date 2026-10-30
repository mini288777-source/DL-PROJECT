"""
Test Final Metric Registry.
Verifies the canonical metric definitions, formulas, and units in src/metrics/metric_registry.py.
"""

import pytest
from src.metrics.metric_registry import METRIC_REGISTRY, get_metric, list_metrics


def test_metric_registry_contains_required_metrics():
    required = [
        "MACRO_F1",
        "END_TO_END_MACRO_F1",
        "EMERGENCY_RECALL",
        "EMERGENCY_PRECISION_STAGE1",
        "EMERGENCY_PRECISION_END_TO_END",
        "FER",
        "IPER",
        "IPER_ERROR",
        "STAGE2_MACRO_F1",
        "STAGE2_BALANCED_ACCURACY",
    ]
    all_metrics = list_metrics()
    for m in required:
        assert m in all_metrics, f"Missing metric: {m}"


def test_iper_and_iper_error_directionality():
    iper = get_metric("IPER")
    iper_error = get_metric("IPER_ERROR")
    
    assert iper.positive_direction is True, "IPER rate should have positive_direction=True"
    assert iper_error.positive_direction is False, "IPER_ERROR should have positive_direction=False"
    assert "1.0 - IPER" in iper_error.formula


def test_emergency_precision_scoping():
    p_s1 = get_metric("EMERGENCY_PRECISION_STAGE1")
    p_e2e = get_metric("EMERGENCY_PRECISION_END_TO_END")
    
    assert p_s1.scope == "stage1"
    assert p_e2e.scope == "end_to_end"
    assert p_s1.metric_id != p_e2e.metric_id
