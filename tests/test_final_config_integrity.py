"""
Test Final Configuration Integrity.
Verifies single source of truth configuration contracts in config/research_protocol.yaml
and config/model_registry.yaml.
"""

import os
import yaml
import pytest


def test_research_protocol_yaml_exists_and_valid():
    proto_path = "config/research_protocol.yaml"
    assert os.path.exists(proto_path), f"Missing {proto_path}"
    with open(proto_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    assert cfg["project"]["team"] == "kernal"
    assert cfg["dataset"]["name"] == "TORGO"
    
    # Cohort checks
    cohort = cfg["cohort"]
    assert cohort["total_speakers"] == 15
    assert cohort["dysarthric"]["count"] == 8
    assert len(cohort["dysarthric"]["speakers"]) == 8
    assert cohort["control"]["count"] == 7
    assert len(cohort["control"]["speakers"]) == 7
    
    # Encoder
    assert cfg["encoder"]["embedding_dim"] == 384
    assert cfg["encoder"]["dtype"] == "float32"
    assert cfg["encoder"]["normalization"] == "unit_L2"
    
    # Calibration budgets
    assert cfg["calibration_budgets"]["N"] == [20, 30, 40, 50]
    
    # Operational status
    assert cfg["project"]["deployment_readiness"] == "NOT READY"
    assert cfg["project"]["clinical_validation"] == "NOT PERFORMED"
    assert cfg["project"]["hardware_execution"] == "NOT TESTED"


def test_model_registry_yaml_integrity():
    model_path = "config/model_registry.yaml"
    assert os.path.exists(model_path), f"Missing {model_path}"
    with open(model_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
        
    models = reg["models"]
    assert "classical_adapter" in models
    assert "classical_rbf_svm" in models
    assert "qsvc_q8_r1" in models
    assert "qsvc_q4_r1" in models
    assert "qsvc_q8_r2" in models
    
    # Parameter counts and architecture
    adapter = models["classical_adapter"]
    assert adapter["architecture"]["parameter_count"]["input_384_classes_7"] == 25223
    assert adapter["training"]["optimizer"] == "AdamW"
    assert adapter["training"]["learning_rate"] == 0.005
    assert adapter["training"]["epochs"] == 60
    
    # Quantum specification
    q8 = models["qsvc_q8_r1"]
    assert q8["parameters"]["qubit_count"] == 8
    assert q8["parameters"]["feature_map"] == "ZZFeatureMap"
    assert q8["parameters"]["backend"] == "exact_statevector"
    assert q8["parameters"]["C"] == 1.0
