import os
import re
import hashlib
import json
import pandas as pd
import numpy as np

# Load manifest and intent mapping
manifest = pd.read_csv("data/manifests/torgo_manifest.csv")
intent_map = pd.read_csv("data/manifests/torgo_intent_mapping.csv")
merged = manifest.merge(intent_map[["recording_id", "intent_candidate"]], on="recording_id", how="left")
active = merged[merged["split"].isin(["adaptation", "test"])].copy()

CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CLASSES if c != "GENERAL_SPEECH"]

def get_norm_transcript(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[^A-Za-z0-9 ]+", " ", text).strip().upper()
    return re.sub(r"\s+", " ", t)

active["norm_transcript"] = active["transcript"].apply(get_norm_transcript)
active["prompt_group_id"] = active["norm_transcript"].apply(
    lambda t: "PGRP_" + hashlib.sha256(t.encode("utf-8")).hexdigest()[:10]
)

print(f"Total active recordings: {len(active)}")
print(f"Total unique prompt groups: {active['prompt_group_id'].nunique()}")

# Test pilot on 3 speakers
PILOT_SPEAKERS = ["F01", "M04", "FC02"]

def build_speaker_recovery_sequence(spk: str, master_seed: int = 42, track: str = "A"):
    s_df = active[active["speaker_id"] == spk].copy()
    rng = np.random.default_rng(master_seed + sum(ord(c) for c in spk))
    
    # Group by prompt_group_id
    groups = []
    for pg_id, g in s_df.groupby("prompt_group_id"):
        intent = g["intent_candidate"].iloc[0]
        recs = sorted(g["recording_id"].tolist())
        groups.append({
            "prompt_group_id": pg_id,
            "norm_transcript": g["norm_transcript"].iloc[0],
            "intent": intent,
            "recording_ids": recs,
            "size": len(recs),
            "sessions": sorted(g["session_id"].unique().tolist())
        })
    
    # Separate emergency vs general prompt groups
    emerg_groups = [g for g in groups if g["intent"] != "GENERAL_SPEECH"]
    gen_groups = [g for g in groups if g["intent"] == "GENERAL_SPEECH"]
    
    # Shuffle groups deterministically
    rng.shuffle(emerg_groups)
    rng.shuffle(gen_groups)
    
    # Priority selection of emergency prompt groups
    # Group emergency groups by intent
    intent_to_groups = {}
    for g in emerg_groups:
        intent_to_groups.setdefault(g["intent"], []).append(g)
        
    selected_groups = []
    selected_recs_count = 0
    test_reserved_groups = []
    
    # For each emergency class present for this speaker:
    # If len >= 2, reserve 1 group for test, eligible = len - 1
    # If len == 1, in Track A put in adaptation (test_support_limited = True)
    eligible_emerg_groups = []
    for intent, grp_list in sorted(intent_to_groups.items()):
        if len(grp_list) >= 2:
            test_reserved_groups.append(grp_list[0])
            eligible_emerg_groups.extend(grp_list[1:])
        else:
            eligible_emerg_groups.extend(grp_list)
            
    # Round-robin selection across available emergency classes to maximize class diversity
    by_intent_eligible = {}
    for g in eligible_emerg_groups:
        by_intent_eligible.setdefault(g["intent"], []).append(g)
        
    ordered_emerg_groups = []
    any_added = True
    while any_added:
        any_added = False
        for intent in sorted(by_intent_eligible.keys()):
            if by_intent_eligible[intent]:
                ordered_emerg_groups.append(by_intent_eligible[intent].pop(0))
                any_added = True
                
    # Add emergency groups up to budget constraints
    # In N=50, we want maximum emergency coverage while leaving room for GENERAL_SPEECH
    # For example, max emergency recordings = 25 (or all available if < 25)
    max_emerg_budget = 25 if track == "B" else 20
    
    for g in ordered_emerg_groups:
        if selected_recs_count + g["size"] <= max_emerg_budget:
            selected_groups.append(g)
            selected_recs_count += g["size"]
            
    # Now fill the rest of the 50 recordings using general speech prompt groups
    # To hit EXACTLY 50 recordings, we prioritize singletons (size=1) or groups that fit exactly
    remaining_gen = [g for g in gen_groups if g not in selected_groups]
    # Sort or partition so we can hit exactly 50
    # First add groups that fit
    for g in list(remaining_gen):
        if selected_recs_count + g["size"] <= 50:
            selected_groups.append(g)
            selected_recs_count += g["size"]
            remaining_gen.remove(g)
            
    print(f"Speaker {spk}: Selected {len(selected_groups)} prompt groups, total recs: {selected_recs_count}")
    return selected_groups, selected_recs_count

for spk in PILOT_SPEAKERS:
    build_speaker_recovery_sequence(spk)
