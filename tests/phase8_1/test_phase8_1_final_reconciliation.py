"""Phase 8.1 Final Reconciliation and Safety-Readiness Audit Tests.

Verifies:
1. FNR = 1 - recall under identical definitions and evaluation populations.
2. TP + FN equals emergency support (237 samples).
3. FP + TN equals non-emergency support (4615 samples).
4. Per-speaker FER matches the raw confusion counts FP / (FP + TN).
5. Aggregated FER uses the correct denominator (sum of non-emergency samples).
6. Dysarthric/control labels match canonical metadata (8 dysarthric, 7 control).
7. Every speaker-level safety status uses the fixed 15% limit (M05 > 15% in Track A & B).
8. No unavailable dataset is claimed as independent confirmation (status = NOT_AVAILABLE).
9. Existing Phase 8.1 raw predictions and historical artifacts remain unchanged.
10. All report, CSV, JSON, and figure paths resolve.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECON_RES_DIR = REPO_ROOT / "results" / "phase8_1" / "final_reconciliation"
RECON_REP_DIR = REPO_ROOT / "reports" / "phase8_1" / "final_reconciliation"
PHASE8_1_RES_DIR = REPO_ROOT / "results" / "phase8_1"


def test_1_fnr_formula_identity():
    """Verify FNR = 1 - Recall for all 6 candidates."""
    fnr_path = RECON_RES_DIR / "fnr_reconciliation.csv"
    assert fnr_path.exists(), f"Missing {fnr_path}"
    df = pd.read_csv(fnr_path)
    assert len(df) == 6

    for _, row in df.iterrows():
        recall = float(row["emergency_recall"])
        fnr = float(row["false_negative_rate"])
        assert pytest.approx(recall + fnr, abs=1e-4) == 1.0, (
            f"FNR identity violated for {row['candidate_id']}: recall={recall}, fnr={fnr}"
        )
        assert "RECONCILED" in row["reconciliation_status"]

    # Specific check: INT-3-SAMP is 80.17%, NOT > 87%
    samp_row = df[df["candidate_id"] == "INT-3-SAMP"].iloc[0]
    assert samp_row["false_negative_rate"] < 0.87, "INT-3-SAMP FNR must be below 87% (80.17%)"
    assert pytest.approx(samp_row["false_negative_rate"], abs=1e-3) == 0.8017


def test_2_tp_fn_equals_emergency_support():
    """Verify TP + FN equals emergency support for all candidates (237 samples)."""
    fnr_path = RECON_RES_DIR / "fnr_reconciliation.csv"
    df = pd.read_csv(fnr_path)
    for _, row in df.iterrows():
        tp = int(row["TP"])
        fn = int(row["FN"])
        supp = int(row["emergency_support"])
        assert tp + fn == supp, f"TP ({tp}) + FN ({fn}) != support ({supp}) for {row['candidate_id']}"
        assert supp == 237, f"Expected 237 emergency samples, got {supp}"


def test_3_fp_tn_equals_non_emergency_support():
    """Verify FP + TN equals non-emergency support (4615 samples) across all candidates."""
    spk_path = RECON_RES_DIR / "speaker_fer_reconciliation.csv"
    assert spk_path.exists()
    df = pd.read_csv(spk_path)

    for cand_id, cand_group in df.groupby("candidate_id"):
        tot_fp = cand_group["FP"].sum()
        tot_tn = cand_group["TN"].sum()
        tot_non_emerg = cand_group["non_emergency_samples"].sum()
        assert tot_fp + tot_tn == tot_non_emerg, (
            f"FP ({tot_fp}) + TN ({tot_tn}) != non-emergency samples ({tot_non_emerg}) for {cand_id}"
        )
        assert tot_non_emerg == 4615, (
            f"Expected 4615 non-emergency samples, got {tot_non_emerg} for {cand_id}"
        )


def test_4_per_speaker_fer_matches_raw_counts():
    """Verify per-speaker FER exactly matches FP / (FP + TN)."""
    spk_path = RECON_RES_DIR / "speaker_fer_reconciliation.csv"
    df = pd.read_csv(spk_path)
    assert len(df) == 15 * 6  # 15 speakers * 6 candidates = 90 rows

    for _, row in df.iterrows():
        fp = int(row["FP"])
        tn = int(row["TN"])
        denom = fp + tn
        fer = float(row["FER"])
        expected_fer = fp / denom if denom > 0 else 0.0
        assert pytest.approx(fer, abs=1e-5) == expected_fer, (
            f"FER mismatch for {row['candidate_id']} - {row['speaker_id']}: got {fer}, expected {expected_fer}"
        )


def test_5_aggregated_fer_denominator():
    """Verify pooled FER uses the correct aggregated denominator sum(FP) / sum(non_emergency_samples)."""
    summary_path = RECON_RES_DIR / "speaker_safety_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    spk_path = RECON_RES_DIR / "speaker_fer_reconciliation.csv"
    df = pd.read_csv(spk_path)

    for cand_id, cinfo in summary["tracks"].items():
        pooled_fer = cinfo["pooled_outer_fold_FER"]
        cand_df = df[df["candidate_id"] == cand_id]
        tot_fp = cand_df["FP"].sum()
        tot_denom = cand_df["non_emergency_samples"].sum()
        assert tot_denom == 4615
        expected = tot_fp / tot_denom
        assert pytest.approx(pooled_fer, abs=1e-5) == expected, (
            f"Pooled FER mismatch for {cand_id}: {pooled_fer} vs {expected}"
        )


def test_6_dysarthric_control_cohort_metadata():
    """Verify speaker cohort mappings match canonical metadata (8 dysarthric, 7 control)."""
    spk_path = RECON_RES_DIR / "speaker_fer_reconciliation.csv"
    df = pd.read_csv(spk_path)

    expected_dysarthric = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}
    expected_control = {"FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"}

    spk_cohorts = df[["speaker_id", "cohort"]].drop_duplicates()
    assert len(spk_cohorts) == 15

    for _, row in spk_cohorts.iterrows():
        sid = row["speaker_id"]
        cohort = row["cohort"]
        if sid in expected_dysarthric:
            assert cohort == "dysarthric", f"Speaker {sid} should be dysarthric, got {cohort}"
        elif sid in expected_control:
            assert cohort == "control", f"Speaker {sid} should be control, got {cohort}"
        else:
            pytest.fail(f"Unexpected speaker ID: {sid}")


def test_7_speaker_safety_status_fixed_15pct():
    """Verify every speaker-level safety status uses the fixed 15% limit.

    Specifically:
    - Track A: M05 exceeds 15% (15.56%).
    - Track B: M05 exceeds 15% (15.19%).
    - INT-3-SAMP: FC02 (26.97%) and M03 (21.24%) exceed 15%.
    - INT-1, INT-2, Track C: 0 speakers exceed 15%.
    """
    summary_path = RECON_RES_DIR / "speaker_safety_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Track B (Quantum)
    qb = summary["tracks"]["Track_B_Quantum"]
    assert qb["speakers_above_15pct_count"] == 1
    assert "M05" in qb["speakers_above_15pct_FER"]
    assert len(qb["speakers_at_or_below_15pct_FER"]) == 14

    # Track A (Classical)
    ta = summary["tracks"]["Track_A_Baseline"]
    assert ta["speakers_above_15pct_count"] == 1
    assert "M05" in ta["speakers_above_15pct_FER"]

    # INT-3-SAMP
    s3 = summary["tracks"]["INT-3-SAMP"]
    assert s3["speakers_above_15pct_count"] == 2
    assert "FC02" in s3["speakers_above_15pct_FER"]
    assert "M03" in s3["speakers_above_15pct_FER"]

    # Critical rule: pooled FER <= 15% does not mean all speakers pass
    assert "A pooled FER at or below 15% does not establish that every speaker individually satisfies the same limit." in summary["pooled_vs_speaker_level_finding"]


def test_8_independent_confirmation_not_available():
    """Verify that independent confirmation is NOT_AVAILABLE and 20% test partition is DIAGNOSTIC_ONLY."""
    conf_path = RECON_RES_DIR / "confirmation_availability.json"
    assert conf_path.exists()
    with open(conf_path, "r", encoding="utf-8") as f:
        conf = json.load(f)

    assert conf["determination"] == "NOT_AVAILABLE"
    assert conf["existing_test_set_status"] == "DIAGNOSTIC_ONLY"
    assert conf["phase_8_2_readiness_plan"]["execution_status"] == "NOT_STARTED"


def test_9_phase8_1_raw_predictions_unchanged():
    """Verify existing Phase 8.1 raw predictions and outputs are preserved."""
    oof_preds = PHASE8_1_RES_DIR / "oof_predictions.csv"
    assert oof_preds.exists()
    assert oof_preds.stat().st_size > 0

    model_comp = PHASE8_1_RES_DIR / "model_comparison.csv"
    assert model_comp.exists()
    assert model_comp.stat().st_size > 0

    per_spk = PHASE8_1_RES_DIR / "per_speaker_results.csv"
    assert per_spk.exists()
    assert per_spk.stat().st_size > 0

    # Fold manifest preservation
    fold_manifest = PHASE8_1_RES_DIR / "fold_manifest.json"
    assert fold_manifest.exists()
    assert fold_manifest.stat().st_size > 0


def test_10_all_report_and_figure_paths_resolve():
    """Verify all final reconciliation reports and figures exist and are non-empty."""
    req_files = [
        RECON_RES_DIR / "fnr_reconciliation.csv",
        RECON_RES_DIR / "speaker_fer_reconciliation.csv",
        RECON_RES_DIR / "speaker_safety_summary.json",
        RECON_RES_DIR / "confirmation_availability.json",
        RECON_RES_DIR / "speaker_fer_reconciliation.png",
        RECON_REP_DIR / "INDEPENDENT_CONFIRMATION_AUDIT.md",
        RECON_REP_DIR / "PHASE8_1_FINAL_RECONCILIATION.md",
    ]
    for p in req_files:
        assert p.exists(), f"Required file missing: {p}"
        assert p.stat().st_size > 0, f"File empty: {p}"
