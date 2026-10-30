"""
Phase 3.5 Research Validity Audit Script.

Computes:
1. reports/phase3_5_sample_budget_audit.csv
2. reports/phase3_5_intent_validity_audit.csv
3. reports/phase3_5_class_balance_analysis.csv
4. Summary statistics for all narrative markdown reports.
"""

import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC

def main():
    print("=== STARTING PHASE 3.5 AUDIT CALCULATIONS ===")

    # 1. Load splits and manifest
    with open("splits/torgo_splits.json", "r", encoding="utf-8") as f:
        splits_list = json.load(f)

    # Group splits by speaker
    splits = {}
    for entry in splits_list:
        sp = entry["speaker_id"]
        n_val = entry["sample_n"]
        if sp not in splits:
            splits[sp] = {}
        splits[sp][n_val] = {
            "adaptation": entry["adaptation_recording_ids"],
            "test": entry["test_recording_ids"]
        }

    manifest_df = pd.read_csv("data/manifests/torgo_manifest.csv")
    rec_to_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))
    rec_to_transcript = dict(zip(manifest_df["recording_id"], manifest_df["transcript"]))

    # Helper to load whisper embedding
    whisper_dir = "embeddings/torgo/whisper"
    def load_emb(rid):
        return np.load(os.path.join(whisper_dir, f"{rid}.npy"))

    # 2. Audit Sample Budgets (N=20, 30, 40, 50)
    # Target columns: speaker_id, N, adaptation_count, unique_recording_count, unique_intent_count, recording_hash, support_vector_count
    budget_records = []
    nesting_results = {}

    for speaker_id in sorted(splits.keys()):
        sp_data = splits[speaker_id]
        nesting_results[speaker_id] = True
        prev_ids = set()
        for N_val in [20, 30, 40, 50]:
            adapt_ids = sp_data[N_val]["adaptation"]
            
            # Check unique recordings
            unique_ids = sorted(list(set(adapt_ids)))
            intents = [rec_to_intent.get(rid, "UNKNOWN") for rid in unique_ids]
            unique_intents = sorted(list(set(intents)))
            
            # Hash of sorted recording IDs
            id_hash = hashlib.sha256(",".join(unique_ids).encode("utf-8")).hexdigest()[:16]
            
            # Compute support vectors by fitting SVC on the actual embeddings
            X_adapt = np.stack([load_emb(rid) for rid in adapt_ids])
            y_adapt = intents
            
            # If single class, SVC cannot fit multi-class, support vectors is either N or 0 (fallback)
            if len(unique_intents) <= 1:
                sv_count = 0  # Single-class trivial boundary fallback
            else:
                svc = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
                svc.fit(X_adapt, y_adapt)
                sv_count = int(np.sum(svc.n_support_))

            # Verify nesting: prev_ids must be subset of current unique_ids
            if not prev_ids.issubset(set(unique_ids)):
                nesting_results[speaker_id] = False
            prev_ids = set(unique_ids)

            budget_records.append({
                "speaker_id": speaker_id,
                "N": N_val,
                "adaptation_count": len(adapt_ids),
                "unique_recording_count": len(unique_ids),
                "unique_intent_count": len(unique_intents),
                "recording_hash": id_hash,
                "support_vector_count": sv_count
            })

    budget_df = pd.DataFrame(budget_records)
    budget_csv_path = "reports/phase3_5_sample_budget_audit.csv"
    budget_df.to_csv(budget_csv_path, index=False)
    print(f"Generated {budget_csv_path} with {len(budget_df)} rows.")
    print("Nesting verification across all 15 speakers:")
    for sp, is_nested in nesting_results.items():
        print(f"  {sp}: nested={is_nested}")

    # 3. Intent Validity Audit
    raw_intent_map = pd.read_csv("data/manifests/torgo_intent_mapping.csv")
    raw_map_dict = {
        row["recording_id"]: row for _, row in raw_intent_map.iterrows()
    }

    # Get all unique recording IDs in the cohort (adaptation and test)
    cohort_rec_ids = set()
    for sp, sp_dict in splits.items():
        for n_val, sp_data in sp_dict.items():
            cohort_rec_ids.update(sp_data["adaptation"])
            cohort_rec_ids.update(sp_data["test"])

    print(f"\nTotal unique cohort recordings: {len(cohort_rec_ids)}")
    
    intent_audit_records = []
    for rid in sorted(list(cohort_rec_ids)):
        transcript = rec_to_transcript.get(rid, "")
        assigned_intent = rec_to_intent.get(rid, "GENERAL_SPEECH")
        
        raw_info = raw_map_dict.get(rid, None)
        if raw_info is not None:
            rule = raw_info["mapping_rule"]
            status = raw_info["mapping_status"]
            conf = raw_info["mapping_confidence"]
            rev = raw_info["manual_review_required"]
            note = raw_info["notes"]
        else:
            rule = "STANDARD_SPEECH_VOCABULARY"
            status = "UNMAPPED"
            conf = 1.0
            rev = False
            note = "Standard reading vocabulary without emergency semantics."

        # Note explanation
        if assigned_intent == "GENERAL_SPEECH":
            if status == "EXCLUDED":
                validity_note = "Incidental emergency token in multi-word literary reading passage; correctly mapped to GENERAL_SPEECH baseline."
            elif status == "UNMAPPED":
                validity_note = "Standard reading stimulus prompt without emergency semantics; valid GENERAL_SPEECH baseline."
            else:
                validity_note = f"Non-emergency vocabulary ({note})."
        else:
            if status == "VALID":
                validity_note = f"Valid emergency intent: {note}"
            else:
                validity_note = f"Status {status}: {note}"

        intent_audit_records.append({
            "recording_id": rid,
            "transcript": transcript,
            "assigned_intent": assigned_intent,
            "mapping_rule": rule,
            "mapping_status": status,
            "mapping_confidence": conf,
            "manual_review_required": rev,
            "validity_note": validity_note
        })

    intent_audit_df = pd.DataFrame(intent_audit_records)
    intent_audit_csv_path = "reports/phase3_5_intent_validity_audit.csv"
    intent_audit_df.to_csv(intent_audit_csv_path, index=False)
    print(f"Generated {intent_audit_csv_path} with {len(intent_audit_df)} rows.")

    # 4. Class Balance Analysis
    class_balance_records = []
    
    # First: Overall Cohort Unique Recordings Class Distribution
    cohort_manifest = manifest_df[manifest_df["recording_id"].isin(cohort_rec_ids)]
    total_cohort = len(cohort_manifest)
    class_counts = cohort_manifest["intent"].value_counts()
    
    for intent_name, count in class_counts.items():
        pct = (count / total_cohort) * 100.0
        class_balance_records.append({
            "scope": "COHORT_OVERALL",
            "speaker_id": "ALL",
            "N": "ALL",
            "split": "ALL",
            "intent": intent_name,
            "count": count,
            "percentage": round(pct, 2),
            "is_single_class_condition": False
        })

    # Second: Per-Speaker Test Set Class Distribution
    test_recs_by_speaker = {}
    for speaker_id, sp_dict in splits.items():
        test_ids = sp_dict[50]["test"]
        test_recs_by_speaker[speaker_id] = test_ids
        test_df = manifest_df[manifest_df["recording_id"].isin(test_ids)]
        t_total = len(test_df)
        t_counts = test_df["intent"].value_counts()
        for intent_name, count in t_counts.items():
            class_balance_records.append({
                "scope": "SPEAKER_TEST_SET",
                "speaker_id": speaker_id,
                "N": "HELD_OUT_TEST",
                "split": "test",
                "intent": intent_name,
                "count": count,
                "percentage": round((count / t_total) * 100.0, 2),
                "is_single_class_condition": False
            })

    # Third: Per-Speaker Per-N Adaptation Class Distribution
    single_class_count = 0
    total_adaptation_conditions = 0
    for speaker_id, sp_dict in splits.items():
        for N_val in [20, 30, 40, 50]:
            total_adaptation_conditions += 1
            adapt_ids = sp_dict[N_val]["adaptation"]
            adapt_df = manifest_df[manifest_df["recording_id"].isin(adapt_ids)]
            a_total = len(adapt_df)
            a_counts = adapt_df["intent"].value_counts()
            
            is_single = len(a_counts) == 1
            if is_single:
                single_class_count += 1
                
            for intent_name, count in a_counts.items():
                class_balance_records.append({
                    "scope": "ADAPTATION_SET",
                    "speaker_id": speaker_id,
                    "N": str(N_val),
                    "split": "adaptation",
                    "intent": intent_name,
                    "count": count,
                    "percentage": round((count / a_total) * 100.0, 2),
                    "is_single_class_condition": is_single
                })

    class_balance_df = pd.DataFrame(class_balance_records)
    class_balance_csv_path = "reports/phase3_5_class_balance_analysis.csv"
    class_balance_df.to_csv(class_balance_csv_path, index=False)
    print(f"Generated {class_balance_csv_path} with {len(class_balance_df)} rows.")
    print(f"Total single-class adaptation conditions: {single_class_count} / {total_adaptation_conditions} ({round(single_class_count/total_adaptation_conditions*100, 1)}%)")

    # 5. RBF Invariance Audit: Inspect predictions
    preds_df = pd.read_csv("results/phase3/predictions.csv")
    rbf_preds = preds_df[preds_df["method"] == "rbf_svm"].copy()
    
    print("\n=== RBF-SVM PREDICTION ANALYSIS ===")
    rbf_pred_counts = rbf_preds["predicted_intent"].value_counts()
    print("RBF-SVM Predicted Intent Frequencies across all test instances:")
    print(rbf_pred_counts)
    
    gen_pct = (rbf_pred_counts.get("GENERAL_SPEECH", 0) / len(rbf_preds)) * 100.0
    print(f"Percentage of test instances predicted as GENERAL_SPEECH by RBF-SVM: {gen_pct:.4f}%")

    # Check Adapter predictions for comparison
    adapter_preds = preds_df[preds_df["method"] == "adapter"].copy()
    adapt_pred_counts = adapter_preds["predicted_intent"].value_counts()
    print("\n=== CLASSICAL ADAPTER PREDICTION ANALYSIS ===")
    print("Adapter Predicted Intent Frequencies:")
    print(adapt_pred_counts)

    # 6. Leakage Audit Check
    leakage_failures = 0
    for sp, sp_dict in splits.items():
        for n_val, sp_data in sp_dict.items():
            overlap = set(sp_data["adaptation"]).intersection(set(sp_data["test"]))
            if len(overlap) > 0:
                print(f"LEAKAGE DETECTED in {sp} N={n_val}: {len(overlap)} overlapping IDs!")
                leakage_failures += 1
    if leakage_failures == 0:
        print("\nLEAKAGE CHECK: 100% PASS. Zero overlap between adaptation and test sets.")

    print("=== PHASE 3.5 AUDIT CALCULATIONS COMPLETE ===")

if __name__ == "__main__":
    main()
