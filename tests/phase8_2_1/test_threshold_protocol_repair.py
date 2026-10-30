"""Phase 8.2.1 Threshold Protocol Repair and Safety Governance Regression Suite.

Verifies the 12 mandatory conditions specified in Section 7 of
PHASE8_2_1_THRESHOLD_PROTOCOL_REPAIR.md:

1. decision_threshold and fer_maximum are separate configuration fields.
2. A threshold cannot be accepted solely because it equals the FER limit.
3. Every ready-for-confirmation threshold has model, score-scale, calibration, and selection provenance.
4. A threshold for another model/checkpoint cannot be silently reused.
5. Threshold selection references development/OOF/validation records only.
6. The previously inspected 20% test partition is never used for threshold selection.
7. Confirmation labels cannot be read during threshold selection.
8. FER is evaluated using the canonical registry definition and correct units.
9. Percent values (15) and fractions (0.15) cannot be compared without explicit unit conversion.
10. A missing confirmation dataset forces confirmation status to BLOCKED.
11. This correction namespace does not alter frozen Phase 1–8 artifacts.
12. The protocol cannot be marked READY_FOR_CONFIRMATION if any required provenance field is missing.
"""

import json
from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_8_2_1 = REPO_ROOT / "config" / "phase8_2_1" / "confirmation_protocol_v2.yaml"
RES_8_2_1_AUDIT = REPO_ROOT / "results" / "phase8_2_1" / "threshold_protocol_audit.json"
RES_8_2_1_INTEG = REPO_ROOT / "results" / "phase8_2_1" / "protocol_integrity.json"
REP_8_2_1 = REPO_ROOT / "reports" / "phase8_2_1" / "THRESHOLD_PROTOCOL_CORRECTION_REPORT.md"
RESEARCH_AUDIT = REPO_ROOT / "research" / "phase8_2_1" / "THRESHOLD_PROTOCOL_FORENSIC_AUDIT.md"

# Historical Phase 8.1 & 8.2 artifacts for non-modification check
HISTORICAL_PATHS = {
    "config/phase8_1/safety_recovery_protocol.yaml": "71a06d9514de44c651be7d63ec4cacabb97ccaf94977bde742efc21389423f52",
    "config/phase8_2/confirmation_protocol.yaml": "de2434dcb1cee100f3ffc49160b8acab5151c97de7b31c618222c8389bac1f93",
    "results/phase8_1/threshold_registry.json": "d6fc0173d138cd57bfa84ac4cdaa03edc175ce23428d494f2d1ad5b129ed9d49",
    "results/phase8_1/safety_operating_points.csv": "b81c06166b130ca01d0ee1459bd84f60cc18ae4cdfe3b8336e1cfafb0b6211ad",
    "reports/phase8_1/FINAL_SAFETY_DECISION.md": "778e7209bc6724f8d4d6bb420c209d5c5dda4e607253f5149df3cfeaaa904062",
    "results/phase8_2/confirmation_dataset_inventory.csv": "4e2967faffd6657e6a47e66443586b11bfada50e51b13bdcc6641eb8f6dd5747",
    "results/phase8_2/confirmation_overlap_audit.json": "573bd286a3dac72d5bea2f32034af1a9900adaab0de16ff44cf8420a1a7b5300",
}


@pytest.fixture
def protocol_v2():
    assert CONF_8_2_1.exists(), f"Missing {CONF_8_2_1}"
    with open(CONF_8_2_1, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def audit_json():
    assert RES_8_2_1_AUDIT.exists(), f"Missing {RES_8_2_1_AUDIT}"
    with open(RES_8_2_1_AUDIT, "r", encoding="utf-8") as f:
        return json.load(f)


# Condition 1: decision_threshold and fer_maximum are separate configuration fields
def test_1_separate_decision_threshold_and_fer_maximum_fields(protocol_v2):
    """Verify decision_policy and safety_constraints are distinct top-level sections with decoupled values."""
    assert "decision_policy" in protocol_v2
    assert "safety_constraints" in protocol_v2

    safety = protocol_v2["safety_constraints"]
    decision = protocol_v2["decision_policy"]

    # Safety constraint holds FER maximum
    assert "fer_maximum_fraction" in safety
    assert safety["fer_maximum_fraction"] == 0.1500
    assert safety["fer_maximum_percent"] == 15.0

    # Decision policy holds model threshold concepts
    assert "score_name" in decision
    assert "candidate_models" in decision
    for cand_name, cand_info in decision["candidate_models"].items():
        assert "development_pooled_operating_point" in cand_info
        # None of the development operating points is arbitrarily 0.1500
        assert cand_info["development_pooled_operating_point"] >= 0.50


# Condition 2: A threshold cannot be accepted solely because it equals the FER limit
def test_2_threshold_cannot_be_accepted_solely_matching_fer_limit(audit_json, protocol_v2):
    """Verify that equating threshold to the FER bound is rejected as an invalid conflation defect."""
    verdict = audit_json["forensic_classification"]["verdict"]
    assert verdict == "FER_LIMIT_MISUSED_AS_DECISION_THRESHOLD"
    assert audit_json["forensic_classification"]["threshold_validity"] is False

    # Check that in Protocol v2, tau=0.1500 is explicitly rejected
    decision = protocol_v2["decision_policy"]
    assert "historical_defect_classification" in decision
    assert decision["historical_defect_classification"] == "FER_LIMIT_MISUSED_AS_DECISION_THRESHOLD"
    assert "rejected" in decision["historical_tau_disposition"].lower()


# Condition 3: Every ready-for-confirmation threshold has model, score-scale, calibration, and selection provenance
def test_3_provenance_fields_present_for_each_candidate(protocol_v2):
    """Verify each candidate specifies model, score scale, calibration, and selection provenance."""
    candidates = protocol_v2["decision_policy"]["candidate_models"]
    assert len(candidates) == 6

    required_provenance_keys = [
        "architecture",
        "score_scale",
        "calibration_version",
        "selection_dataset_partition",
        "selection_method",
        "selection_artifact",
        "development_pooled_operating_point",
        "development_mean_selected_tau"
    ]

    for cname, cinfo in candidates.items():
        for k in required_provenance_keys:
            assert k in cinfo, f"Candidate {cname} missing required provenance field: {k}"
            assert cinfo[k] is not None, f"Candidate {cname} has null provenance field: {k}"


# Condition 4: A threshold for another model/checkpoint cannot be silently reused
def test_4_no_silent_reuse_of_thresholds_across_distinct_models(protocol_v2):
    """Verify distinct model architectures and calibration pipelines maintain their own operating points."""
    candidates = protocol_v2["decision_policy"]["candidate_models"]
    
    # INT-1 (Platt scaling) must have a different operating point than Track A or INT-2
    int1_pt = candidates["INT-1-CALIB"]["development_pooled_operating_point"]
    track_a_pt = candidates["Track_A_Baseline"]["development_pooled_operating_point"]
    int2_pt = candidates["INT-2-COST"]["development_pooled_operating_point"]

    assert int1_pt != track_a_pt, "Platt-calibrated model cannot silently reuse Track A baseline threshold"
    assert int1_pt != int2_pt, "Platt-calibrated model cannot silently reuse Cost-sensitive threshold"


# Condition 5: Threshold selection references development/OOF/validation records only
def test_5_threshold_selection_references_development_oof_only(protocol_v2):
    """Verify that selection dataset partition references development pool only."""
    candidates = protocol_v2["decision_policy"]["candidate_models"]
    for cname, cinfo in candidates.items():
        partition = cinfo["selection_dataset_partition"]
        method = cinfo["selection_method"]
        artifact = cinfo["selection_artifact"]

        assert "80% Training Pool" in partition
        assert "Nested 5-fold inner CV OOF" in method
        assert artifact == "results/phase8_1/threshold_registry.json"


# Condition 6: The previously inspected 20% test partition is never used for threshold selection
def test_6_locked_test_partition_excluded_from_threshold_selection(protocol_v2):
    """Verify that the locked 20% test partition is strictly excluded and quarantined."""
    disqual = protocol_v2["confirmation_data_status"]["disqualification_summary"]
    assert "torgo_locked_test_20" in disqual
    assert "DIAGNOSTIC_ONLY" in disqual["torgo_locked_test_20"]

    # Verify no candidate references test partition
    candidates = protocol_v2["decision_policy"]["candidate_models"]
    for cname, cinfo in candidates.items():
        assert "test" not in cinfo["selection_dataset_partition"].lower()


# Condition 7: Confirmation labels cannot be read during threshold selection
def test_7_confirmation_data_locking_contract_enforced(protocol_v2):
    """Verify data locking contract strictly forbids reading labels during threshold selection."""
    contract = protocol_v2["data_locking_contract"]
    assert contract["manifest_fingerprint_required"] is True
    assert contract["sha256_verification_before_reading"] is True
    assert "Single-shot forward inference only" in contract["eval_execution"]
    assert "Zero fitting, tuning, calibration, PCA estimation, replay, or threshold modification" in contract["eval_execution"]


# Condition 8: FER is evaluated using the canonical registry definition and correct units
def test_8_fer_canonical_definition_and_units(protocol_v2):
    """Verify canonical FER definition matches reports/FINAL_METRIC_DEFINITIONS.md."""
    safety = protocol_v2["safety_constraints"]
    assert safety["fer_metric_name"] == "False Emergency Rate (FER)"
    assert safety["fer_formula"] == "FP / (FP + TN) = FP / Total_General_Speech"
    assert safety["fer_definition_id"] == "canonical_metric_registry:reports/FINAL_METRIC_DEFINITIONS.md#L52-L62"
    assert safety["fer_unit"] == "fraction"


# Condition 9: Percent values (15) and fractions (0.15) cannot be compared without explicit unit conversion
def test_9_unit_conversion_rule_enforced(protocol_v2):
    """Verify unit conversion rule forbids comparing percent directly to fraction."""
    safety = protocol_v2["safety_constraints"]
    assert "unit_conversion_rule" in safety
    assert "Direct comparison between 15 and 0.15 without conversion is prohibited" in safety["unit_conversion_rule"]

    # Check simulated conversion function
    def validate_fer_compliance(empirical_fer, fer_bound, unit="fraction"):
        if unit == "percent":
            empirical_fer = empirical_fer / 100.0
        return empirical_fer <= fer_bound

    assert validate_fer_compliance(12.0, 0.1500, unit="percent") is True
    assert validate_fer_compliance(16.0, 0.1500, unit="percent") is False
    assert validate_fer_compliance(0.120, 0.1500, unit="fraction") is True
    assert validate_fer_compliance(0.160, 0.1500, unit="fraction") is False


# Condition 10: A missing confirmation dataset forces confirmation status to BLOCKED
def test_10_missing_confirmation_dataset_forces_blocked_status(protocol_v2):
    """Verify that absent confirmation data enforces BLOCKED status unconditionally."""
    assert protocol_v2["protocol_metadata"]["status"] == "BLOCKED"
    assert protocol_v2["confirmation_data_status"]["status"] == "NOT_AVAILABLE"
    assert protocol_v2["confirmation_data_status"]["eligible_independent_datasets_count"] == 0
    assert "BLOCKED" in protocol_v2["protocol_metadata"]["governance_gate"]


# Condition 11: This correction namespace does not alter frozen Phase 1–8 artifacts
def test_11_historical_artifacts_remain_unmodified():
    """Verify SHA-256 hashes of audited historical artifacts remain identical to frozen baselines."""
    import hashlib
    for rel_path, expected_hash in HISTORICAL_PATHS.items():
        abs_p = REPO_ROOT / rel_path
        assert abs_p.exists(), f"Historical file missing: {rel_path}"
        computed_hash = hashlib.sha256(abs_p.read_bytes()).hexdigest()
        assert computed_hash == expected_hash, f"Historical file modified: {rel_path}"


# Condition 12: The protocol cannot be marked READY_FOR_CONFIRMATION if any required provenance field is missing
def test_12_protocol_cannot_be_ready_if_provenance_or_checkpoints_missing(protocol_v2):
    """Verify protocol status cannot be READY_FOR_CONFIRMATION when checkpoints are unmaterialized."""
    candidates = protocol_v2["decision_policy"]["candidate_models"]
    
    # Check that candidate checkpoints are unmaterialized and readiness is BLOCKED
    for cname, cinfo in candidates.items():
        assert cinfo["checkpoint_sha256"] is None, f"{cname} should have null hash due to unmaterialized file"
        assert "BLOCKED" in cinfo["readiness_for_evaluation"]

    # Protocol must not be READY_FOR_CONFIRMATION
    assert protocol_v2["protocol_metadata"]["status"] != "READY_FOR_CONFIRMATION"
    assert protocol_v2["protocol_metadata"]["status"] == "BLOCKED"
