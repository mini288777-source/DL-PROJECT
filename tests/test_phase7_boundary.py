"""
Phase 7.0 Boundary and Contract Integrity Tests.
Verifies the isolation of the Phase 7 continual learning extension,
immutability of frozen baselines (Phases 1-6.5), inheritance of taxonomy
and Whisper representation, and structural validity of schemas.
"""

import os
import yaml
import pytest


def test_frozen_configuration_immutability():
    """Verify that frozen research protocol and model registry exist and are marked frozen."""
    frozen_proto = "config/research_protocol.yaml"
    frozen_models = "config/model_registry.yaml"
    assert os.path.exists(frozen_proto), f"Missing frozen protocol: {frozen_proto}"
    assert os.path.exists(frozen_models), f"Missing frozen model registry: {frozen_models}"
    
    with open(frozen_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["project"]["status"] == "COMPLETE_FROZEN"
    assert cfg["statistical_contract"]["primary_quantum_decision"] == "CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE"


def test_phase7_configuration_separation():
    """Verify that Phase 7 continual protocol is strictly separated from frozen protocol."""
    continual_proto = "config/continual/research_protocol.yaml"
    assert os.path.exists(continual_proto), f"Missing continual protocol: {continual_proto}"
    
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    assert cfg["project_extension"]["name"] == "Continual Quantum Personalization"
    assert cfg["project_extension"]["status"] == "experimental"
    assert cfg["project_extension"]["execution_phase"] == "7.0 (Boundary & Contract Freeze Only)"
    assert cfg["baseline"]["immutable"] is True
    assert cfg["baseline"]["frozen_phase"] == "Phase_1_to_6_5"


def test_canonical_taxonomy_inheritance():
    """Verify that Phase 7 inherits the canonical 7-class taxonomy from config/taxonomy.yaml."""
    continual_proto = "config/continual/research_protocol.yaml"
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    expected_7 = [
        "GENERAL_SPEECH",
        "CRITICAL_RESPONSE",
        "FALL",
        "BREATHING_MEDICAL",
        "PAIN",
        "SOS_HELP",
        "LOCATION",
    ]
    assert cfg["taxonomy"]["classes"] == expected_7


def test_whisper_encoder_reference():
    """Verify Whisper-tiny 384-D unit-L2 frozen encoder contract in Phase 7."""
    continual_proto = "config/continual/research_protocol.yaml"
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    enc = cfg["encoder"]
    assert enc["model"] == "openai/whisper-tiny"
    assert enc["embedding_dim"] == 384
    assert enc["frozen"] is True
    assert enc["dtype"] == "float32"
    assert enc["normalization"] == "unit_L2"
    assert enc["trainable"] is False


def test_experience_record_schema_validity():
    """Verify that the experience record schema defines all required fields and learning statuses."""
    continual_proto = "config/continual/research_protocol.yaml"
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    exp_schema = cfg["experience_record_schema"]
    field_names = [list(f.keys())[0] for f in exp_schema["fields"]]
    
    required_fields = [
        "sample_id",
        "speaker_id",
        "audio_reference",
        "embedding_reference",
        "predicted_intent",
        "confirmed_intent",
        "prediction_confidence",
        "confirmation_source",
        "timestamp",
        "model_version",
        "representation_version",
        "learning_status",
        "data_source",
        "quality_score",
        "novelty_score",
        "environment_metadata",
    ]
    for rf in required_fields:
        assert rf in field_names, f"Missing required experience field: {rf}"
        
    expected_statuses = [
        "UNREVIEWED",
        "AUTO_ELIGIBLE",
        "USER_CONFIRMED",
        "REJECTED",
        "QUARANTINED",
        "USED_FOR_TRAINING",
        "VALIDATION_ONLY",
    ]
    assert cfg["experience_record_schema"]["learning_statuses"] == expected_statuses


def test_model_version_schema_validity():
    """Verify that the model version schema defines all required fields and states."""
    continual_proto = "config/continual/research_protocol.yaml"
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    mv_schema = cfg["model_version_schema"]
    field_names = [list(f.keys())[0] for f in mv_schema["fields"]]
    
    required_fields = [
        "model_id",
        "version",
        "parent_version",
        "training_sample_ids",
        "training_dataset_hash",
        "configuration_hash",
        "representation_version",
        "quantum_configuration",
        "metrics",
        "created_at",
        "status",
    ]
    for rf in required_fields:
        assert rf in field_names, f"Missing required model version field: {rf}"
        
    expected_states = [
        "CANDIDATE",
        "VALIDATED",
        "ACTIVE",
        "REJECTED",
        "ROLLED_BACK",
        "ARCHIVED",
    ]
    assert cfg["model_version_schema"]["model_states"] == expected_states


def test_no_automatic_training_or_promotion():
    """Verify that Phase 7.0 explicitly disables automated model training and promotion."""
    continual_proto = "config/continual/research_protocol.yaml"
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    assert cfg["project_extension"]["model_training_allowed"] is False
    assert cfg["project_extension"]["new_experiments_allowed"] is False
    assert cfg["validation_contract"]["promotion_policy"] == "TO_BE_DEFINED_IN_PHASE_7.6"


def test_frozen_artifacts_intact():
    """Verify critical frozen artifacts from Phase 1-6.5 remain intact and present."""
    frozen_files = [
        "results/phase3/classical_sample_efficiency_aggregated.csv",
        "results/phase4/quantum_aggregated.csv",
        "results/phase6_5_5_matched_comparison.csv",
        "results/phase6_5_5_statistical_tests.csv",
        "phase6_5/splits/recovery_splits_hierarchical_ready.json",
        "reports/final/FINAL_RESEARCH_RESULTS.md",
        "reports/final/FINAL_RESEARCH_CONCLUSION.md",
    ]
    for ff in frozen_files:
        assert os.path.exists(ff), f"Frozen artifact missing or moved: {ff}"
        assert os.path.getsize(ff) > 0, f"Frozen artifact corrupted: {ff}"


def test_phase7_documentation_suite_exists():
    """Verify all 4 required Phase 7 architectural and contract documentation files exist."""
    required_docs = [
        "docs/continual/PHASE7_ARCHITECTURE.md",
        "docs/continual/PHASE7_RESEARCH_CONTRACT.md",
        "docs/continual/PHASE7_DATA_CONTRACT.md",
        "docs/continual/PHASE7_MODEL_VERSION_CONTRACT.md",
    ]
    for rd in required_docs:
        assert os.path.exists(rd), f"Missing Phase 7 documentation: {rd}"
        assert os.path.getsize(rd) > 100, f"Documentation file is empty: {rd}"


def test_phase7_anti_hallucination_rules():
    """Verify all 10 anti-hallucination rules are recorded in the protocol."""
    continual_proto = "config/continual/research_protocol.yaml"
    with open(continual_proto, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    rules = cfg["anti_hallucination_rules"]
    assert len(rules) == 10
    assert "Never invent experimental results" in rules["RULE_1"]
    assert "Never start the next phase automatically" in rules["RULE_10"]
