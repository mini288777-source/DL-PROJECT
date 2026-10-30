"""
Generate final reconciliation artifacts for Phase 8.1:
1. fnr_reconciliation.csv
2. speaker_fer_reconciliation.csv
3. speaker_safety_summary.json
4. confirmation_availability.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    out_dir = os.path.join(REPO_ROOT, "results", "phase8_1", "final_reconciliation")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Generate fnr_reconciliation.csv
    comp_path = os.path.join(REPO_ROOT, "results", "phase8_1", "model_comparison.csv")
    df_comp = pd.read_csv(comp_path)

    fnr_rows = []
    for _, r in df_comp.iterrows():
        cand_id = r["track_id"]
        tp = int(r["TP"])
        fn = int(r["FN"])
        support = int(r["true_emergencies"])
        rec = float(r["emergency_recall"])
        fnr = float(fn / support) if support > 0 else 0.0
        
        # Verify reconciliation
        status = "RECONCILED_EXACT" if (tp + fn == support and abs((1.0 - rec) - fnr) < 1e-4) else "DISCREPANCY"

        fnr_rows.append({
            "candidate_id": cand_id,
            "evaluation_population": "Phase 8.1 Outer-Fold Held-out (N=4,852)",
            "TP": tp,
            "FN": fn,
            "emergency_support": support,
            "emergency_recall": rec,
            "false_negative_rate": fnr,
            "calculation_method": "FN / (TP + FN) = 1 - Emergency Recall",
            "reconciliation_status": status
        })

    df_fnr = pd.DataFrame(fnr_rows)
    df_fnr.to_csv(os.path.join(out_dir, "fnr_reconciliation.csv"), index=False)
    print("Saved results/phase8_1/final_reconciliation/fnr_reconciliation.csv.")

    # 2. Generate speaker_fer_reconciliation.csv
    spk_path = os.path.join(REPO_ROOT, "results", "phase8_1", "per_speaker_results.csv")
    df_spk = pd.read_csv(spk_path)

    speaker_fer_rows = []
    for _, r in df_spk.iterrows():
        cand_id = r["track_id"]
        spk_id = r["speaker_id"]
        cohort = r["speaker_group"]
        tp = int(r["TP"])
        fp = int(r["FP"])
        fn = int(r["FN"])
        tn = int(r["TN"])
        em_support = int(r["true_emergencies"])
        non_em_support = fp + tn
        total_samples = int(r["samples"])
        
        rec = float(r["emergency_recall"])
        prec = float(r["emergency_precision"])
        fer = float(r["FER"])
        exceeds_15 = bool(fer > 0.15)

        speaker_fer_rows.append({
            "candidate_id": cand_id,
            "speaker_id": spk_id,
            "cohort": cohort,
            "total_samples": total_samples,
            "true_emergency_samples": em_support,
            "non_emergency_samples": non_em_support,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "emergency_recall": rec,
            "emergency_precision": prec,
            "FER": fer,
            "fer_formula": "FP / (FP + TN)",
            "exceeds_15pct_safety_limit": exceeds_15,
            "evaluation_provenance": "Phase 8.1 5-fold outer cross-validation held-out predictions"
        })

    df_spk_fer = pd.DataFrame(speaker_fer_rows)
    df_spk_fer.to_csv(os.path.join(out_dir, "speaker_fer_reconciliation.csv"), index=False)
    print("Saved results/phase8_1/final_reconciliation/speaker_fer_reconciliation.csv.")

    # 3. Generate speaker_safety_summary.json
    safety_summary = {
        "audit": "Phase 8.1 Speaker-Specific FER and Safety Audit",
        "canonical_fer_formula": "FER = FP / (FP + TN) = FP / Total_Non_Emergency_Speech",
        "safety_bound_limit": 0.15,
        "pooled_vs_speaker_level_finding": "A pooled FER at or below 15% does not establish that every speaker individually satisfies the same limit.",
        "tracks": {}
    }

    for cand_id in df_spk_fer["candidate_id"].unique():
        sub = df_spk_fer[df_spk_fer["candidate_id"] == cand_id]
        above = sub[sub["exceeds_15pct_safety_limit"] == True]
        below_or_equal = sub[sub["exceeds_15pct_safety_limit"] == False]
        
        pooled_fp = int(sub["FP"].sum())
        pooled_tn = int(sub["TN"].sum())
        pooled_non_em = pooled_fp + pooled_tn
        pooled_fer = float(pooled_fp / pooled_non_em)

        dys = sub[sub["cohort"] == "dysarthric"]
        ctrl = sub[sub["cohort"] == "control"]

        dys_fp, dys_tn = int(dys["FP"].sum()), int(dys["TN"].sum())
        ctrl_fp, ctrl_tn = int(ctrl["FP"].sum()), int(ctrl["TN"].sum())

        safety_summary["tracks"][cand_id] = {
            "pooled_outer_fold_FER": pooled_fer,
            "pooled_FER_percentage": f"{pooled_fer*100:.2f}%",
            "pooled_safety_pass": pooled_fer <= 0.15,
            "speakers_above_15pct_FER": above["speaker_id"].tolist(),
            "speakers_above_15pct_count": len(above),
            "speakers_at_or_below_15pct_FER": below_or_equal["speaker_id"].tolist(),
            "speakers_at_or_below_15pct_count": len(below_or_equal),
            "detailed_violating_speakers": [
                {
                    "speaker_id": r["speaker_id"],
                    "cohort": r["cohort"],
                    "FP": int(r["FP"]),
                    "TN": int(r["TN"]),
                    "non_emergency_support": int(r["non_emergency_samples"]),
                    "FER": float(r["FER"]),
                    "FER_percentage": f"{r['FER']*100:.2f}%"
                } for _, r in above.iterrows()
            ],
            "dysarthric_cohort": {
                "total_non_emergency_samples": dys_fp + dys_tn,
                "FP": dys_fp,
                "TN": dys_tn,
                "cohort_FER": float(dys_fp / (dys_fp + dys_tn)),
                "cohort_FER_percentage": f"{(dys_fp / (dys_fp + dys_tn))*100:.2f}%"
            },
            "control_cohort": {
                "total_non_emergency_samples": ctrl_fp + ctrl_tn,
                "FP": ctrl_fp,
                "TN": ctrl_tn,
                "cohort_FER": float(ctrl_fp / (ctrl_fp + ctrl_tn)),
                "cohort_FER_percentage": f"{(ctrl_fp / (ctrl_fp + ctrl_tn))*100:.2f}%"
            }
        }

    with open(os.path.join(out_dir, "speaker_safety_summary.json"), "w", encoding="utf-8") as f:
        json.dump(safety_summary, f, indent=2)
    print("Saved results/phase8_1/final_reconciliation/speaker_safety_summary.json.")

    # 4. Generate confirmation_availability.json
    confirmation_data = {
        "audit": "Phase 8.1 Independent Confirmation Data Availability Audit",
        "determination": "NOT_AVAILABLE",
        "existing_test_set_status": "DIAGNOSTIC_ONLY",
        "audited_datasets": [
            {
                "dataset_name": "TORGO Atypical Speech Database",
                "manifest_path": "data/manifests/torgo_manifest.csv",
                "total_active_utterances": 6067,
                "split_allocation": {
                    "phase8_1_development_training_pool": 4852,
                    "phase8_locked_test_partition": 1215
                },
                "overlap_audit": "100% of TORGO utterances were utilized in Phase 8 and Phase 8.1 development/diagnostics.",
                "eligibility_as_independent_confirmation": False,
                "disqualification_rationale": "The 1,215 locked test partition was inspected across multiple Phase 8 audits and reporting cycles; it is contaminated for confirmation and restricted to diagnostic-only."
            }
        ],
        "other_repository_data_search": {
            "search_paths": ["data/", "embeddings/"],
            "findings": "No secondary, prospective, or uninspected atypical speech dataset exists in the repository.",
            "synthetic_or_relabeled_data_found": False
        },
        "phase_8_2_readiness_plan": {
            "step_1": "Acquire or collect a prospective, consented atypical speech dataset with independent speakers and verified audio provenance.",
            "step_2": "Verify zero sample, speaker, and session overlap with the TORGO corpus.",
            "step_3": "Validate sufficient emergency-class support across clinical emergency intents.",
            "step_4": "Pre-register and freeze exact model checkpoints, thresholds, metric definitions, and acceptance criteria.",
            "step_5": "Execute a single-shot confirmation evaluation without post-hoc threshold adjustment.",
            "execution_status": "NOT_STARTED"
        }
    }

    with open(os.path.join(out_dir, "confirmation_availability.json"), "w", encoding="utf-8") as f:
        json.dump(confirmation_data, f, indent=2)
    print("Saved results/phase8_1/final_reconciliation/confirmation_availability.json.")

if __name__ == "__main__":
    main()
