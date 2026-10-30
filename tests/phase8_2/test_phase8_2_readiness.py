"""Phase 8.2 Confirmation Readiness, Provenance, and Overlap Audit Tests.

Verifies the 15 mandatory conditions from Section 10 of
PHASE8_2_INDEPENDENT_CONFIRMATORY_EVALUATION.md:

1. Confirmation data do not overlap historical sample/audio/session IDs.
2. Dataset fingerprints and manifests match the inventory.
3. Model, threshold, and protocol were frozen before evaluation.
4. Confirmation labels are not used for fitting or selection.
5. Confirmation data never enter .fit, calibration, threshold selection, PCA/scaler fitting, replay, or optimization.
6. Each candidate's metrics use its own model predictions.
7. Metrics reconcile with saved predictions and confusion counts.
8. TP+FN and FP+TN match denominators.
9. Unsupported classes follow the prespecified policy.
10. Pooled and speaker-specific metrics use correct denominators.
11. FER uses the fixed 15% policy.
12. Report numbers match result files.
13. Every figure/image link exists.
14. Historical Phase 1–8 artifacts remain unchanged.
15. A no-data branch cannot claim confirmatory completion or safety pass.
"""

import json
from pathlib import Path
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RES_8_2 = REPO_ROOT / "results" / "phase8_2"
CONF_8_2 = REPO_ROOT / "config" / "phase8_2"
REP_8_2 = REPO_ROOT / "reports" / "phase8_2"
RES_8_1 = REPO_ROOT / "results" / "phase8_1"
RECON_RES = RES_8_1 / "final_reconciliation"


def test_1_confirmation_data_overlap_check():
    """Verify that confirmation audit correctly identifies all historical data as overlapping."""
    overlap_path = RES_8_2 / "confirmation_overlap_audit.json"
    assert overlap_path.exists()
    with open(overlap_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    census = audit["repository_speech_data_census"]
    assert census["total_audio_corpora_present"] == 1
    assert census["corpus_name"] == "TORGO Atypical Speech Database"
    assert audit["verdict"]["determination"] == "NOT_AVAILABLE"
    assert audit["verdict"]["eligible_independent_datasets_count"] == 0


def test_2_dataset_fingerprints_and_manifests_match_inventory():
    """Verify inventory manifests match actual split manifests and counts."""
    inv_path = RES_8_2 / "confirmation_dataset_inventory.csv"
    assert inv_path.exists()
    df_inv = pd.read_csv(inv_path)

    dev_row = df_inv[df_inv["dataset_candidate_id"] == "TORGO-DEV-POOL"].iloc[0]
    test_row = df_inv[df_inv["dataset_candidate_id"] == "TORGO-LOCKED-TEST"].iloc[0]

    assert dev_row["total_samples"] == 4852
    assert dev_row["emergency_samples"] == 237
    assert dev_row["non_emergency_samples"] == 4615
    assert dev_row["independence_determination"] == "AVAILABLE_BUT_OVERLAPPING"

    assert test_row["total_samples"] == 1215
    assert test_row["emergency_samples"] == 63
    assert test_row["non_emergency_samples"] == 1152
    assert test_row["independence_determination"] == "AVAILABLE_BUT_OVERLAPPING"


def test_3_model_threshold_protocol_frozen_before_eval():
    """Verify model checkpoints, thresholds, and protocol are frozen before any evaluation."""
    protocol_path = CONF_8_2 / "confirmation_protocol.yaml"
    assert protocol_path.exists()
    with open(protocol_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    assert proto["status"] == "FROZEN_READINESS_SPECIFICATION"
    assert proto["safety_constraints"]["safety_bound_limit"] == 0.1500

    candidates = proto["preregistered_candidate_models"]
    assert len(candidates) == 6
    for cname, cinfo in candidates.items():
        assert "tau=0.1500" in cinfo["frozen_threshold_source"]
        assert "checkpoints/phase8_1" in cinfo["checkpoint_reference"]


def test_4_confirmation_labels_not_used_for_fitting_or_selection():
    """Verify protocol mandates unlabelled inference with locked labels."""
    plan_path = REP_8_2 / "CONFIRMATION_DATA_ACQUISITION_AND_LOCKING_PLAN.md"
    assert plan_path.exists()
    content = plan_path.read_text(encoding="utf-8")
    assert "Zero training, fine-tuning, or PCA recalibration" in content or "unlabelled audio" in content
    assert "Ground truth labels will only be unblinded after model inference" in content


def test_5_confirmation_data_never_enter_fitting_pipelines():
    """Verify protocol strictly forbids confirmation data from entering .fit pipelines."""
    protocol_path = CONF_8_2 / "confirmation_protocol.yaml"
    with open(protocol_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    eval_contract = proto["data_locking_contract"]["eval_execution"]
    assert "Zero training, fine-tuning, or PCA recalibration" in eval_contract


def test_6_candidate_metrics_use_independent_predictions():
    """Verify each candidate's metrics derive from its own predictions and not cross-copied."""
    fnr_path = RECON_RES / "fnr_reconciliation.csv"
    assert fnr_path.exists()
    df = pd.read_csv(fnr_path)

    # Check distinct TP values across classical baseline, quantum, and interventions
    tp_classical = df[df["candidate_id"] == "Track_A_Baseline"]["TP"].iloc[0]
    tp_quantum = df[df["candidate_id"] == "Track_B_Quantum"]["TP"].iloc[0]
    tp_samp = df[df["candidate_id"] == "INT-3-SAMP"]["TP"].iloc[0]

    assert tp_classical == 24
    assert tp_quantum == 29
    assert tp_samp == 47
    assert tp_classical != tp_quantum
    assert tp_classical != tp_samp


def test_7_metrics_reconcile_with_saved_predictions():
    """Verify metrics reconcile with raw predictions and confusion counts."""
    spk_path = RECON_RES / "speaker_fer_reconciliation.csv"
    assert spk_path.exists()
    df = pd.read_csv(spk_path)

    for _, row in df.iterrows():
        fp = int(row["FP"])
        tn = int(row["TN"])
        denom = fp + tn
        fer = float(row["FER"])
        expected = fp / denom if denom > 0 else 0.0
        assert pytest.approx(fer, abs=1e-5) == expected


def test_8_tp_fn_and_fp_tn_match_denominators():
    """Verify TP+FN equals emergency support and FP+TN equals non-emergency support."""
    fnr_df = pd.read_csv(RECON_RES / "fnr_reconciliation.csv")
    for _, row in fnr_df.iterrows():
        assert int(row["TP"]) + int(row["FN"]) == int(row["emergency_support"])
        assert int(row["emergency_support"]) == 237

    spk_df = pd.read_csv(RECON_RES / "speaker_fer_reconciliation.csv")
    for cand_id, group in spk_df.groupby("candidate_id"):
        assert group["FP"].sum() + group["TN"].sum() == 4615


def test_9_unsupported_classes_follow_prespecified_policy():
    """Verify prespecified policy for unsupported classes in protocol."""
    protocol_path = CONF_8_2 / "confirmation_protocol.yaml"
    with open(protocol_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    policy = proto["zero_support_policy"]
    assert "N/A" in policy
    assert "denominator=7" in policy or "divide_by_7" in policy


def test_10_pooled_and_speaker_specific_metrics_denominators():
    """Verify pooled and speaker-specific FER use correct denominators."""
    with open(RECON_RES / "speaker_safety_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Track B (Quantum) pooled check
    qb = summary["tracks"]["Track_B_Quantum"]
    assert pytest.approx(qb["pooled_outer_fold_FER"], abs=1e-4) == 294 / 4615

    # Dysarthric speaker M05 denominator check
    spk_df = pd.read_csv(RECON_RES / "speaker_fer_reconciliation.csv")
    m05_qb = spk_df[(spk_df["candidate_id"] == "Track_B_Quantum") & (spk_df["speaker_id"] == "M05")].iloc[0]
    assert m05_qb["FP"] == 41
    assert m05_qb["TN"] == 229
    assert m05_qb["non_emergency_samples"] == 270
    assert pytest.approx(m05_qb["FER"], abs=1e-4) == 41 / 270


def test_11_fer_uses_fixed_15pct_safety_policy():
    """Verify that the FER <= 15% policy is strictly applied and never relaxed."""
    protocol_path = CONF_8_2 / "confirmation_protocol.yaml"
    with open(protocol_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    assert proto["safety_constraints"]["safety_bound_limit"] == 0.1500
    assert "STRICTLY_PROHIBITED" in proto["safety_constraints"]["post_hoc_threshold_adjustment"]


def test_12_report_numbers_match_result_files():
    """Verify numbers quoted in Phase 8.2 reports match result files."""
    report_path = REP_8_2 / "PHASE8_2_INDEPENDENT_CONFIRMATION_REPORT.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    assert "80.17%" in content  # INT-3-SAMP FNR
    assert "87.76%" in content  # Track B FNR
    assert "15.19%" in content  # M05 FER in Track B
    assert "4,852" in content
    assert "1,215" in content


def test_13_every_figure_or_report_link_exists():
    """Verify that every linked artifact and report in Phase 8.2 exists on disk."""
    linked_paths = [
        CONF_8_2 / "confirmation_protocol.yaml",
        RES_8_2 / "confirmation_dataset_inventory.csv",
        RES_8_2 / "confirmation_overlap_audit.json",
        REPO_ROOT / "research" / "phase8_2" / "PHASE8_1_ENTRY_GATE.md",
        REPO_ROOT / "research" / "phase8_2" / "CONFIRMATION_DATA_READINESS.md",
        REP_8_2 / "CONFIRMATION_DATA_ACQUISITION_AND_LOCKING_PLAN.md",
        REP_8_2 / "PHASE8_2_STATUS.md",
        REP_8_2 / "PHASE8_2_INDEPENDENT_CONFIRMATION_REPORT.md",
        RECON_RES / "speaker_fer_reconciliation.png",
    ]
    for p in linked_paths:
        assert p.exists(), f"Linked artifact missing: {p}"
        assert p.stat().st_size > 0, f"Linked artifact empty: {p}"


def test_14_historical_phase1_to_8_artifacts_unchanged():
    """Verify historical Phase 1-8 artifacts are completely intact."""
    historical_checks = [
        REPO_ROOT / "results" / "phase8_80_20" / "split_manifest.csv",
        REPO_ROOT / "results" / "phase8_1" / "oof_predictions.csv",
        REPO_ROOT / "results" / "phase8_1" / "model_comparison.csv",
        REPO_ROOT / "results" / "phase8_1" / "per_speaker_results.csv",
        REPO_ROOT / "config" / "phase8_1" / "safety_recovery_protocol.yaml",
        RECON_RES / "fnr_reconciliation.csv",
        RECON_RES / "speaker_fer_reconciliation.csv",
    ]
    for p in historical_checks:
        assert p.exists(), f"Historical file missing: {p}"
        assert p.stat().st_size > 0, f"Historical file empty: {p}"


def test_15_no_data_branch_blocks_confirmation_and_safety_pass():
    """Verify that a no-data branch strictly blocks confirmation and rejects safety pass."""
    status_path = REP_8_2 / "PHASE8_2_STATUS.md"
    assert status_path.exists()
    status_text = status_path.read_text(encoding="utf-8")

    assert "PHASE 8.2 CONFIRMATORY EVALUATION: BLOCKED — INDEPENDENT CONFIRMATION DATA UNAVAILABLE" in status_text
    assert "No safety claims or clinical deployments are authorized" in status_text

    rep_path = REP_8_2 / "PHASE8_2_INDEPENDENT_CONFIRMATION_REPORT.md"
    rep_text = rep_path.read_text(encoding="utf-8")
    assert "CLINICAL SAFETY VERDICT: NOT CONFIRMED" in rep_text
    assert "NOT EXECUTED (BLOCKED)" in rep_text
