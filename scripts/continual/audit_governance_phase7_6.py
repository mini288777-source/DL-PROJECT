"""
Phase 7.6 Governance Validation Audit Script.
Evaluates historical Phase 7.4 (Continual Classical) and Phase 7.5 (Replay Memory)
candidate models through the frozen Phase 7.6 ValidationGate.
Zero model training is performed (Models Trained: 0, New Experiments: 0).
Preserves existing Phase 7.4 / 7.5 canonical results untouched.
"""

import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath("."))
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.continual.governance import (
    load_governance_policy,
    PromotionDecision,
    ValidationReasonCode,
    ValidationDecisionRecord,
    compute_cohort_governance_summary,
)


def run_governance_audit():
    os.makedirs(os.path.join("results", "continual", "phase7_6"), exist_ok=True)
    os.makedirs(os.path.join("reports", "continual", "phase7_6"), exist_ok=True)

    policy = load_governance_policy()
    policy_ver = policy.get("policy_metadata", {}).get("policy_version", "v1.0.0_frozen")
    safety_cfg = policy.get("mandatory_safety_gates", {})
    recall_tol = float(safety_cfg.get("emergency_recall_regression_tolerance", 0.00))
    max_fer = float(safety_cfg.get("max_allowed_false_emergency_rate", 0.50))
    max_fer_inc = float(safety_cfg.get("max_fer_increase_tolerance", 0.05))
    quality_cfg = policy.get("quality_gates", {})
    f1_tol = float(quality_cfg.get("macro_f1_regression_tolerance", -0.01))

    # Load 7.4 and 7.5 result tables
    df_7_4 = pd.read_csv(os.path.join("results", "continual", "phase7_4", "continual_classical_results.csv"))
    df_7_5 = pd.read_csv(os.path.join("results", "continual", "phase7_5", "continual_replay_results.csv"))

    audit_records = []
    decision_records_for_summary = []

    for subphase, df in [("Phase_7.4_Classical", df_7_4), ("Phase_7.5_Replay", df_7_5)]:
        # Create map of version -> row
        ver_map = {row["model_version"]: row for _, row in df.iterrows()}

        for _, row in df.iterrows():
            cand_ver = row["model_version"]
            parent_ver = row["parent_version"]
            spk = row["speaker_id"]
            spk_grp = row["speaker_group"]
            budget = int(row["budget"])

            cand_f1 = float(row["macro_f1"])
            cand_rec = float(row["emergency_recall"])
            cand_fer = float(row["false_emergency_rate"])

            reason_codes = []
            safety_veto = False
            has_parent = (parent_ver != "NONE" and parent_ver in ver_map)

            if has_parent:
                p_row = ver_map[parent_ver]
                p_f1 = float(p_row["macro_f1"])
                p_rec = float(p_row["emergency_recall"])
                p_fer = float(p_row["false_emergency_rate"])

                delta_f1 = cand_f1 - p_f1
                delta_rec = cand_rec - p_rec
                delta_fer = cand_fer - p_fer

                # Safety Gate 1: Emergency recall regression
                if delta_rec < recall_tol:
                    safety_veto = True
                    reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_EMERGENCY_RECALL.value)

                # Safety Gate 2: FER ceiling and delta
                if cand_fer > max_fer:
                    safety_veto = True
                    reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_FER_EXCEEDED.value)
                elif delta_fer > max_fer_inc:
                    safety_veto = True
                    reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_FER_EXCEEDED.value)

                # Quality Gate 1: Macro-F1 regression
                if delta_f1 < f1_tol:
                    reason_codes.append(ValidationReasonCode.REJECTED_MACRO_F1_REGRESSION.value)

                if safety_veto or reason_codes:
                    decision = PromotionDecision.REJECT.value
                    val_status = "REJECTED_SAFETY_OR_QUALITY"
                else:
                    decision = PromotionDecision.PROMOTE.value
                    reason_codes.append(ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value)
                    val_status = "PROMOTABLE"

            else:
                # Seed baseline (B0, budget 20): initial candidate evaluated on absolute safety
                delta_f1 = 0.0
                delta_rec = 0.0
                delta_fer = 0.0
                if cand_fer > max_fer:
                    safety_veto = True
                    reason_codes.append(ValidationReasonCode.REJECTED_SAFETY_VETO_FER_EXCEEDED.value)
                    decision = PromotionDecision.REJECT.value
                    val_status = "REJECTED_SEED_FER_EXCEEDED"
                else:
                    decision = PromotionDecision.PROMOTE.value
                    reason_codes.append(ValidationReasonCode.PROMOTED_ALL_GATES_PASS.value)
                    val_status = "PROMOTABLE_SEED_BASELINE"

            val_id = f"VAL_{subphase}_{cand_ver}"
            audit_records.append({
                "validation_id": val_id,
                "subphase": subphase,
                "experiment_id": row["experiment_id"],
                "speaker_id": spk,
                "speaker_group": spk_grp,
                "budget": budget,
                "model_version": cand_ver,
                "parent_version": parent_ver,
                "macro_f1": cand_f1,
                "emergency_recall": cand_rec,
                "false_emergency_rate": cand_fer,
                "macro_f1_delta": delta_f1,
                "emergency_recall_delta": delta_rec,
                "false_emergency_rate_delta": delta_fer,
                "decision": decision,
                "reason_codes": ";".join(reason_codes),
                "safety_veto": safety_veto,
                "validation_status": val_status,
                "policy_version": policy_ver,
            })

            val_fp = row["dataset_hash"] if "dataset_hash" in row else f"hash_{cand_ver}"
            decision_records_for_summary.append(ValidationDecisionRecord(
                validation_id=val_id,
                candidate_model_version=cand_ver,
                parent_model_version=parent_ver if has_parent else None,
                speaker_id=spk,
                validation_dataset_id="canonical_test_split",
                validation_dataset_fingerprint=val_fp,
                validation_policy_version=policy_ver,
                candidate_metrics={
                    "speaker_group": spk_grp,
                    "macro_f1": cand_f1,
                    "emergency_recall": cand_rec,
                    "false_emergency_rate": cand_fer,
                },
                parent_metrics={"speaker_group": spk_grp} if has_parent else None,
                metric_deltas={
                    "macro_f1_delta": delta_f1,
                    "emergency_recall_delta": delta_rec,
                    "false_emergency_rate_delta": delta_fer,
                },
                decision=decision,
                reason_codes=reason_codes,
                safety_veto=safety_veto,
                validation_status=val_status,
            ))

    # Write out audit CSV
    out_csv = os.path.join("results", "continual", "phase7_6", "governance_validation_records.csv")
    pd.DataFrame(audit_records).to_csv(out_csv, index=False)
    print(f"Saved governance audit records to: {out_csv}")

    # Compute cohort summary
    cohort_summary = compute_cohort_governance_summary(decision_records_for_summary)

    # Breakdown by subphase
    subphase_breakdown = {}
    for sub in ["Phase_7.4_Classical", "Phase_7.5_Replay"]:
        sub_recs = [r for r, rec in zip(decision_records_for_summary, audit_records) if rec["subphase"] == sub]
        subphase_breakdown[sub] = compute_cohort_governance_summary(sub_recs)

    summary_data = {
        "metadata": {
            "policy_version": policy_ver,
            "total_candidates_evaluated": len(audit_records),
            "new_experiments_conducted": 0,
            "models_trained": 0,
        },
        "overall_cohort_summary": cohort_summary,
        "subphase_breakdown": subphase_breakdown,
    }

    out_json = os.path.join("reports", "continual", "phase7_6", "governance_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved governance summary JSON to: {out_json}")

    print("\n================ GOVERNANCE AUDIT SUMMARY ================")
    print(f"Total candidates evaluated: {len(audit_records)}")
    for sub in ["Phase_7.4_Classical", "Phase_7.5_Replay"]:
        s = subphase_breakdown[sub]["overall"]
        print(f"\n{sub}:")
        print(f"  Total: {s['total_candidates_evaluated']}, Promoted: {s['promoted_count']}, "
              f"Rejected: {s['rejected_count']}, Safety Vetoes: {s['safety_veto_count']}")
        print(f"  Promotion Rate: {s['promotion_rate']:.1%}")
        dys = subphase_breakdown[sub]["dysarthric"]
        ctrl = subphase_breakdown[sub]["control"]
        print(f"  Dysarthric Promotion Rate: {dys['promotion_rate']:.1%} (Vetoes: {dys['safety_veto_count']})")
        print(f"  Control Promotion Rate: {ctrl['promotion_rate']:.1%} (Vetoes: {ctrl['safety_veto_count']})")


if __name__ == "__main__":
    run_governance_audit()
