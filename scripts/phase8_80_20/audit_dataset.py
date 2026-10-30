"""
Dataset and Environment Audit for Phase 8 80/20 Benchmark.
"""
import os
import hashlib
import json
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    manifest_csv = os.path.join(REPO_ROOT, "data", "manifests", "torgo_manifest.csv")
    df = pd.read_csv(manifest_csv)
    active = df[df["split"].isin(["adaptation", "test"])].copy()

    print(f"Total rows in torgo_manifest.csv: {len(df)}")
    print(f"Active rows (split in [adaptation, test]): {len(active)}")
    
    # Check speakers
    speakers = sorted(active["speaker_id"].unique().tolist())
    dysarthric_set = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}
    control_set = set(speakers) - dysarthric_set

    print(f"Total speakers: {len(speakers)} ({len(dysarthric_set)} dysarthric, {len(control_set)} control)")
    
    # Check embeddings
    emb_dir = os.path.join(REPO_ROOT, "embeddings", "torgo", "whisper")
    missing_emb = []
    for rid in active["recording_id"]:
        path = os.path.join(emb_dir, f"{rid}.npy")
        if not os.path.exists(path):
            missing_emb.append(rid)
    print(f"Missing embeddings: {len(missing_emb)} / {len(active)}")
    
    # Inspect speaker breakdown
    summary_by_speaker = []
    for spk in speakers:
        spk_df = active[active["speaker_id"] == spk]
        em_count = sum(spk_df["intent"] != "GENERAL_SPEECH")
        gen_count = sum(spk_df["intent"] == "GENERAL_SPEECH")
        vc = spk_df["intent"].value_counts().to_dict()
        summary_by_speaker.append({
            "speaker_id": spk,
            "speaker_group": "dysarthric" if spk in dysarthric_set else "control",
            "total_samples": len(spk_df),
            "emergency_samples": em_count,
            "general_speech_samples": gen_count,
            "class_counts": vc
        })
        print(f"Speaker {spk} ({'dysarthric' if spk in dysarthric_set else 'control'}): total={len(spk_df)}, em={em_count}, gen={gen_count}")
        for c, count in vc.items():
            if c != "GENERAL_SPEECH":
                print(f"    {c}: {count}")

    print("\nTotal Active Intents:")
    print(active["intent"].value_counts())

if __name__ == "__main__":
    main()
