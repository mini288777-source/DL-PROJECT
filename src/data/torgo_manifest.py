"""
TORGO Canonical Manifest and Speaker Shortlist Module.

Consolidates validated audio metadata, transcripts, speaker groups,
and intent classifications into the canonical manifest CSV.
Generates:
    - data/manifests/torgo_manifest.csv
    - data/manifests/torgo_speaker_shortlist.csv
"""

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml


def build_manifest_and_shortlist(
    config_path: str = "configs/torgo_config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build canonical manifest CSV and speaker shortlist CSV.

    Returns:
        manifest_df: Canonical dataset manifest DataFrame.
        shortlist_df: Speaker shortlist DataFrame.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parquet_glob = config["dataset"]["parquet_glob"]
    extracted_audio_dir = config["dataset"]["extracted_audio_dir"]
    manifest_out = config["paths"]["manifest"]
    shortlist_out = config["paths"]["speaker_shortlist"]
    audio_val_path = config["paths"]["audio_validation_csv"]
    intent_map_path = config["paths"]["intent_mapping"]
    eligibility = config["eligibility_rules"]

    # 1. Load audio validation results if available
    val_map = {}
    if os.path.exists(audio_val_path):
        val_df = pd.read_csv(audio_val_path)
        val_map = val_df.set_index("recording_id")["status"].to_dict()

    # 2. Load intent mapping if available
    intent_map = {}
    if os.path.exists(intent_map_path):
        im_df = pd.read_csv(intent_map_path)
        intent_map = im_df.set_index("recording_id")["intent_candidate"].to_dict()

    # 3. Read raw parquets
    parquet_files = sorted(glob.glob(parquet_glob))
    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    raw_df = pd.concat(dfs, ignore_index=True)

    fn_pattern = re.compile(r"^([A-Za-z0-9]+)_([0-9]+)_([A-Za-z0-9]+)_([0-9]+)\.wav$")

    manifest_rows = []
    for idx, row in raw_df.iterrows():
        orig_path = row["audio"]["path"]
        filename = os.path.basename(orig_path.replace("\\", "/"))
        rec_id = filename.rsplit(".", 1)[0]
        wav_path = os.path.join(extracted_audio_dir, filename).replace("\\", "/")

        m = fn_pattern.match(filename)
        if m:
            speaker_id = m.group(1)
            session_id = m.group(2)
            mic_id = m.group(3)
            prompt_id = m.group(4)
        else:
            speaker_id = "UNKNOWN"
            session_id = "UNKNOWN"
            mic_id = "UNKNOWN"
            prompt_id = "UNKNOWN"

        speech_status = row.get("speech_status", "unknown")
        if speech_status == "dysarthria":
            speaker_group = "dysarthric"
        elif speech_status == "healthy":
            speaker_group = "control"
        else:
            speaker_group = str(speech_status)

        transcript = str(row.get("transcription", "")).strip() if row.get("transcription") is not None else ""
        duration_sec = round(float(row.get("duration", 0.0)), 4)
        quality_status = val_map.get(rec_id, "VALID")
        intent = intent_map.get(rec_id, "GENERAL_SPEECH")

        manifest_rows.append({
            "recording_id": rec_id,
            "speaker_id": speaker_id,
            "session_id": session_id,
            "audio_path": wav_path,
            "transcript": transcript,
            "speaker_group": speaker_group,
            "duration_sec": duration_sec,
            "sample_rate": 16000,
            "channels": 1,
            "file_format": "wav",
            "dataset": "TORGO",
            "quality_status": quality_status,
            "intent": intent,
            "split": "unassigned"  # To be populated by torgo_splits.py
        })

    manifest_df = pd.DataFrame(manifest_rows)
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    manifest_df.to_csv(manifest_out, index=False)
    print(f"Saved canonical manifest: {manifest_out} ({len(manifest_df)} utterances)")

    # 4. Build Speaker Shortlist
    shortlist_rows = []
    min_prompts_req = eligibility.get("min_total_unique_prompts", 70)
    min_adapt_req = eligibility.get("min_adaptation_samples", 50)
    min_test_req = eligibility.get("min_test_samples", 20)

    for spk, spk_df in manifest_df.groupby("speaker_id"):
        spk_group = spk_df["speaker_group"].iloc[0]
        total_utts = len(spk_df)
        valid_labeled = len(spk_df[(spk_df["quality_status"] == "VALID") & (spk_df["transcript"] != "")])
        sessions = sorted(spk_df["session_id"].unique().tolist())
        sessions_str = ";".join(str(s) for s in sessions)

        # Count unique prompt IDs
        rec_ids = spk_df["recording_id"].tolist()
        prompts = set()
        for rid in rec_ids:
            parts = rid.split("_")
            if len(parts) >= 4:
                prompts.add(parts[3])
        unique_prompts_count = len(prompts)

        usable = True
        exclusion_reasons = []

        if unique_prompts_count < min_prompts_req:
            usable = False
            exclusion_reasons.append(
                f"Insufficient unique prompts ({unique_prompts_count} < required {min_prompts_req} for N=50 adaptation + N=20 test)"
            )

        if valid_labeled < (min_adapt_req + min_test_req):
            usable = False
            exclusion_reasons.append(
                f"Valid labeled utterances ({valid_labeled}) < {min_adapt_req + min_test_req}"
            )

        shortlist_rows.append({
            "speaker_id": spk,
            "speaker_group": spk_group,
            "available_utterances": total_utts,
            "valid_labeled_utterances": valid_labeled,
            "sessions_available": sessions_str,
            "unique_prompts": unique_prompts_count,
            "usable_for_experiment": usable,
            "exclusion_reason": "; ".join(exclusion_reasons) if exclusion_reasons else "NONE"
        })

    shortlist_df = pd.DataFrame(shortlist_rows).sort_values("speaker_id")
    os.makedirs(os.path.dirname(shortlist_out), exist_ok=True)
    shortlist_df.to_csv(shortlist_out, index=False)
    print(f"Saved speaker shortlist: {shortlist_out} ({len(shortlist_df)} speakers)")
    print("\nSpeaker Shortlist Summary:")
    print(shortlist_df[["speaker_id", "speaker_group", "available_utterances", "unique_prompts", "usable_for_experiment", "exclusion_reason"]].to_string())

    return manifest_df, shortlist_df


if __name__ == "__main__":
    from typing import Tuple
    build_manifest_and_shortlist()
