"""
Canonical Split Manifest Builder for Phase 8 Separated Pipeline.

Builds a versioned, immutable split manifest conforming strictly to the approved Phase 8 protocol:
- 15 TORGO Speakers (8 dysarthric, 7 control)
- 4 Few-shot Budgets N in {20, 30, 40, 50} = 60 matched experimental cells
- Train IDs: adaptation set for that budget N
- Test IDs: locked protected test set from N=50 split (held constant across all budgets)
- Zero leakage: Train ∩ Val == ∅, Train ∩ Test == ∅, Val ∩ Test == ∅
"""

import os
import json
import hashlib
from typing import Dict, List, Any
import pandas as pd
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def compute_dataset_fingerprint(splits: List[Dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for s in sorted(splits, key=lambda x: (x["speaker_id"], x["budget"])):
        hasher.update(s["speaker_id"].encode("utf-8"))
        hasher.update(str(s["budget"]).encode("utf-8"))
        hasher.update(",".join(s["train_ids"]).encode("utf-8"))
        hasher.update(",".join(s["test_ids"]).encode("utf-8"))
    return hasher.hexdigest()

def build_manifest():
    splits_path = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    adapt_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_adaptation_manifest.csv"))
    test_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_test_manifest.csv"))

    label_map = dict(zip(adapt_manifest["recording_id"], adapt_manifest["intent"]))
    label_map.update(dict(zip(test_manifest["recording_id"], test_manifest["intent"])))

    speakers = sorted(list(set(s["speaker_id"] for s in splits)))
    budgets = [20, 30, 40, 50]
    dysarthric_set = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}

    canonical_splits = []

    for spk in speakers:
        spk_splits = {s["sample_n"]: s for s in splits if s["speaker_id"] == spk}
        spk_group = "dysarthric" if spk in dysarthric_set else "control"
        
        # Locked protected test set for this speaker across all budgets
        locked_test_ids = list(spk_splits[50]["test_recording_ids"])
        test_labels = [label_map[rid] for rid in locked_test_ids]

        for budget in budgets:
            train_ids = list(spk_splits[budget]["adaptation_recording_ids"])
            train_labels = [label_map[rid] for rid in train_ids]
            val_ids: List[str] = []
            val_labels: List[str] = []

            # Invariant checks
            assert len(train_ids) == budget, f"Train length {len(train_ids)} != budget {budget}"
            assert len(locked_test_ids) > 0, f"Empty test set for {spk}"
            assert set(train_ids).isdisjoint(set(locked_test_ids)), f"Train/Test leakage detected in {spk} N={budget}!"
            assert set(train_ids).isdisjoint(set(val_ids)), f"Train/Val leakage detected in {spk} N={budget}!"
            assert set(val_ids).isdisjoint(set(locked_test_ids)), f"Val/Test leakage detected in {spk} N={budget}!"

            train_counts = dict(Counter(train_labels))
            test_counts = dict(Counter(test_labels))

            canonical_splits.append({
                "split_id": f"PH8_SEP_{spk}_N{budget}",
                "speaker_id": spk,
                "speaker_group": spk_group,
                "budget": budget,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": locked_test_ids,
                "train_labels": train_labels,
                "val_labels": val_labels,
                "test_labels": test_labels,
                "train_class_counts": train_counts,
                "test_class_counts": test_counts,
                "seed": 42,
                "split_algorithm": "speaker_nested_stratified_frozen"
            })

    dataset_fingerprint = compute_dataset_fingerprint(canonical_splits)

    manifest_dict = {
        "manifest_name": "Phase8_Separated_Canonical_Split_Manifest",
        "manifest_version": "2.0-repaired-separated",
        "dataset_name": "TORGO Database of Dysarthric Speech",
        "dataset_fingerprint": dataset_fingerprint,
        "master_seed": 42,
        "total_cells": len(canonical_splits),
        "total_speakers": len(speakers),
        "budgets": budgets,
        "splits": canonical_splits
    }

    # Save to config/phase8_separated
    config_dir = os.path.join(REPO_ROOT, "config", "phase8_separated")
    os.makedirs(config_dir, exist_ok=True)
    out_json = os.path.join(config_dir, "canonical_split_manifest.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    csv_rows = []
    for s in canonical_splits:
        for tid in s["train_ids"]:
            csv_rows.append({
                "split_id": s["split_id"],
                "speaker_id": s["speaker_id"],
                "speaker_group": s["speaker_group"],
                "budget": s["budget"],
                "split_type": "train",
                "sample_id": tid,
                "intent": label_map[tid]
            })
        for tid in s["test_ids"]:
            csv_rows.append({
                "split_id": s["split_id"],
                "speaker_id": s["speaker_id"],
                "speaker_group": s["speaker_group"],
                "budget": s["budget"],
                "split_type": "test",
                "sample_id": tid,
                "intent": label_map[tid]
            })

    out_csv = os.path.join(config_dir, "canonical_split_manifest.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    # Save duplicate copy to results/phase8_separated
    res_dir = os.path.join(REPO_ROOT, "results", "phase8_separated")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "canonical_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
    pd.DataFrame(csv_rows).to_csv(os.path.join(res_dir, "canonical_split_manifest.csv"), index=False)

    print(f"Successfully generated canonical split manifest across {len(canonical_splits)} cells.")
    print(f"Dataset fingerprint: {dataset_fingerprint}")
    print(f"JSON saved to {out_json}")
    print(f"CSV saved to {out_csv}")

if __name__ == "__main__":
    build_manifest()
