"""
Test Final Taxonomy Consistency.
Verifies config/taxonomy.yaml and ensures experimental taxonomy aligns across codebase.
"""

import os
import yaml
import pytest


def test_taxonomy_yaml_structure_and_classes():
    tax_path = "config/taxonomy.yaml"
    assert os.path.exists(tax_path), f"Missing {tax_path}"
    with open(tax_path, "r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
        
    classes = tax["canonical_classes"]
    expected_7 = [
        "GENERAL_SPEECH",
        "CRITICAL_RESPONSE",
        "FALL",
        "BREATHING_MEDICAL",
        "PAIN",
        "SOS_HELP",
        "LOCATION",
    ]
    for c in expected_7:
        assert c in classes, f"Missing class in taxonomy: {c}"
        assert classes[c]["canonical_id"] == c
        
    # Check general speech flag
    assert classes["GENERAL_SPEECH"]["emergency_flag"] is False
    
    # Check emergency speech flag
    for c in expected_7:
        if c != "GENERAL_SPEECH":
            assert classes[c]["emergency_flag"] is True, f"{c} must have emergency_flag=True"
            
    # Check disallowed invented labels
    disallowed = tax["hierarchical_mapping"]["stage2"]["disallowed_unmapped_labels"]
    assert "Medical" in disallowed
    assert "Fire" in disallowed
    assert "Police" in disallowed
    assert "Safety" in disallowed
