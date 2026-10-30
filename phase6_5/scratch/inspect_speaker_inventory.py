import os
import re
import hashlib
import pandas as pd
import numpy as np

# Load manifest and intent mapping
manifest = pd.read_csv("data/manifests/torgo_manifest.csv")
intent_map = pd.read_csv("data/manifests/torgo_intent_mapping.csv")
merged = manifest.merge(intent_map[["recording_id", "intent_candidate"]], on="recording_id", how="left")

# Active 6,067 recordings
active = merged[merged["split"].isin(["adaptation", "test"])].copy()

CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CLASSES if c != "GENERAL_SPEECH"]

SPEAKERS = sorted(active["speaker_id"].unique())
DYSARTHRIC = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}

print(f"Total active recordings: {len(active)}")
print(f"Total manifest recordings: {len(merged)}")
print(f"Speakers: {len(SPEAKERS)}")

# Let's inspect emergency breakdown per speaker in active pool
summary = []
for spk in SPEAKERS:
    s_df = active[active["speaker_id"] == spk]
    grp = "dysarthric" if spk in DYSARTHRIC else "control"
    counts = s_df["intent_candidate"].value_counts().to_dict()
    
    gen = counts.get("GENERAL_SPEECH", 0)
    emerg_counts = {c: counts.get(c, 0) for c in EMERGENCY_CLASSES}
    total_emerg = sum(emerg_counts.values())
    present_emerg = [c for c, v in emerg_counts.items() if v > 0]
    
    unique_prompts = s_df["transcript"].str.strip().str.upper().nunique()
    unique_sessions = s_df["session_id"].nunique()
    
    summary.append({
        "speaker_id": spk,
        "cohort_group": grp,
        "total_recordings": len(s_df),
        "unique_prompts": unique_prompts,
        "num_sessions": unique_sessions,
        "general_speech": gen,
        "total_emergency": total_emerg,
        "num_emerg_classes": len(present_emerg),
        **emerg_counts
    })

sum_df = pd.DataFrame(summary)
print("\nSpeaker Active Pool Summary:")
print(sum_df.to_string())

# Check prompt grouping and repeats among emergency recordings
print("\n--- Emergency Repeat Analysis ---")
for spk in SPEAKERS:
    s_df = active[active["speaker_id"] == spk]
    emerg_df = s_df[s_df["intent_candidate"].isin(EMERGENCY_CLASSES)]
    prompt_counts = emerg_df["transcript"].str.strip().str.upper().value_counts()
    repeats = prompt_counts[prompt_counts > 1]
    singles = prompt_counts[prompt_counts == 1]
    print(f"Speaker {spk}: {len(emerg_df)} emerg recs, {len(prompt_counts)} unique emerg prompts ({len(repeats)} repeated, {len(singles)} singletons)")
