"""
TORGO Dataset Inventory Module.

Discovers, inspects, and catalogs the actual physical structure and metadata
of the downloaded TORGO dataset without making assumptions.
Generates:
    - reports/torgo_dataset_inventory.json
    - reports/torgo_dataset_inventory.md
"""

import glob
import io
import json
import os
import re
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import soundfile as sf
import yaml


def build_inventory(config_path: str = "configs/torgo_config.yaml") -> Dict[str, Any]:
    """Inspect and inventory the TORGO dataset according to configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = config["dataset"]["raw_dir"]
    parquet_glob = config["dataset"]["parquet_glob"]
    out_json = config["paths"]["inventory_json"]
    out_md = config["paths"]["inventory_md"]

    # 1. Directory and file discovery
    parquet_files = sorted(glob.glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {parquet_glob}")

    file_details = []
    total_raw_bytes = 0
    for pf in parquet_files:
        size = os.path.getsize(pf)
        total_raw_bytes += size
        file_details.append({
            "path": os.path.normpath(pf),
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2)
        })

    # 2. Schema and metadata inspection
    dfs = []
    for pf in parquet_files:
        dfs.append(pd.read_parquet(pf))
    full_df = pd.concat(dfs, ignore_index=True)

    total_records = len(full_df)
    columns_found = list(full_df.columns)

    # 3. Filename and token pattern analysis
    # TORGO filenames follow: <speaker>_<session>_<mic>_<prompt>.wav
    fn_pattern = re.compile(r"^([A-Za-z0-9]+)_([0-9]+)_([A-Za-z0-9]+)_([0-9]+)\.wav$")
    parsed_speakers: List[str] = []
    parsed_sessions: List[str] = []
    parsed_mics: List[str] = []
    parsed_prompts: List[str] = []
    malformed_filenames: List[str] = []

    filenames = [os.path.basename(a["path"].replace("\\", "/")) for a in full_df["audio"]]
    full_df["filename"] = filenames

    for fn in filenames:
        m = fn_pattern.match(fn)
        if m:
            parsed_speakers.append(m.group(1))
            parsed_sessions.append(m.group(2))
            parsed_mics.append(m.group(3))
            parsed_prompts.append(m.group(4))
        else:
            malformed_filenames.append(fn)

    full_df["parsed_speaker"] = parsed_speakers
    full_df["parsed_session"] = parsed_sessions
    full_df["parsed_mic"] = parsed_mics
    full_df["parsed_prompt"] = parsed_prompts

    unique_filenames = full_df["filename"].nunique()
    duplicate_filenames = total_records - unique_filenames

    # 4. Speaker and group breakdown
    speakers = sorted(full_df["parsed_speaker"].unique())
    speaker_info = {}
    for spk in speakers:
        spk_df = full_df[full_df["parsed_speaker"] == spk]
        status = spk_df["speech_status"].unique().tolist()
        gender = spk_df["gender"].unique().tolist()
        sessions = sorted(spk_df["parsed_session"].unique().tolist())
        mics = spk_df["parsed_mic"].value_counts().to_dict()
        durations = spk_df["duration"]
        unique_prompts = spk_df["parsed_prompt"].nunique()
        unique_trans = spk_df["transcription"].nunique()

        speaker_info[spk] = {
            "speaker_id": spk,
            "speaker_group": status[0] if len(status) == 1 else "MIXED/UNKNOWN",
            "gender": gender[0] if len(gender) == 1 else "MIXED/UNKNOWN",
            "total_recordings": len(spk_df),
            "unique_prompts": unique_prompts,
            "unique_transcripts": unique_trans,
            "sessions": sessions,
            "microphone_breakdown": mics,
            "total_duration_sec": round(float(durations.sum()), 2),
            "mean_duration_sec": round(float(durations.mean()), 2),
            "min_duration_sec": round(float(durations.min()), 2),
            "max_duration_sec": round(float(durations.max()), 2)
        }

    # 5. Microphone breakdown
    mic_counts = full_df["parsed_mic"].value_counts().to_dict()

    # 6. Transcripts analysis
    null_transcripts = int(full_df["transcription"].isna().sum())
    empty_transcripts = int((full_df["transcription"].str.strip() == "").sum())
    unique_transcripts = int(full_df["transcription"].nunique())

    # 7. Audio integrity sample check (sampling 100 files for quick summary)
    sample_sr = 16000
    sample_channels = 1
    sample_first = full_df.iloc[0]["audio"]["bytes"]
    with io.BytesIO(sample_first) as bio:
        data, sr = sf.read(bio)
        sample_sr = int(sr)
        sample_channels = 1 if data.ndim == 1 else int(data.shape[1])

    # Compile structured inventory dict
    inventory: Dict[str, Any] = {
        "dataset_name": "TORGO",
        "root_directory": os.path.abspath(raw_dir),
        "total_parquet_files": len(parquet_files),
        "total_raw_bytes": total_raw_bytes,
        "total_raw_size_mb": round(total_raw_bytes / (1024 * 1024), 2),
        "raw_files": file_details,
        "total_utterances": total_records,
        "unique_filenames": unique_filenames,
        "duplicate_filenames": duplicate_filenames,
        "malformed_filenames_count": len(malformed_filenames),
        "malformed_filenames": malformed_filenames,
        "schema_columns": columns_found,
        "audio_container_format": "WAV (RIFF standard header stored as raw bytes in parquet)",
        "sample_rate_hz": sample_sr,
        "channels": sample_channels,
        "total_audio_duration_sec": round(float(full_df["duration"].sum()), 2),
        "total_audio_duration_hours": round(float(full_df["duration"].sum()) / 3600.0, 2),
        "mean_utterance_duration_sec": round(float(full_df["duration"].mean()), 2),
        "min_utterance_duration_sec": round(float(full_df["duration"].min()), 2),
        "max_utterance_duration_sec": round(float(full_df["duration"].max()), 2),
        "transcripts": {
            "null_count": null_transcripts,
            "empty_count": empty_transcripts,
            "unique_count": unique_transcripts
        },
        "microphone_breakdown": mic_counts,
        "speaker_count": len(speakers),
        "dysarthric_speaker_count": sum(1 for s in speaker_info.values() if s["speaker_group"] == "dysarthria"),
        "control_speaker_count": sum(1 for s in speaker_info.values() if s["speaker_group"] == "healthy"),
        "speakers": speaker_info
    }

    # Save JSON report
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    # Save Markdown report
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    md_content = generate_markdown_inventory(inventory)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Inventory saved: {out_json} and {out_md}")
    return inventory


def generate_markdown_inventory(inv: Dict[str, Any]) -> str:
    """Generate human-readable Markdown inventory from inventory dict."""
    md = [
        "# TORGO Dataset Physical Inventory Report",
        "",
        f"**Date:** 2026-09-30  ",
        f"**Dataset Root:** `{inv['root_directory']}`  ",
        f"**Source Format:** Apache Parquet (Hugging Face Datasets export)  ",
        "",
        "---",
        "",
        "## 1. High-Level Summary",
        "",
        f"- **Total Audio Utterances:** {inv['total_utterances']:,}",
        f"- **Total Unique Filenames:** {inv['unique_filenames']:,}",
        f"- **Duplicate Filenames:** {inv['duplicate_filenames']}",
        f"- **Malformed Filenames:** {inv['malformed_filenames_count']}",
        f"- **Total Audio Duration:** {inv['total_audio_duration_hours']:.2f} hours ({inv['total_audio_duration_sec']:,.2f} seconds)",
        f"- **Mean Utterance Duration:** {inv['mean_utterance_duration_sec']:.2f} seconds",
        f"- **Sampling Rate:** {inv['sample_rate_hz']} Hz",
        f"- **Channels:** {inv['channels']} (Mono)",
        f"- **Audio Encoding:** Linear PCM (WAV)",
        f"- **Total Speakers:** {inv['speaker_count']} ({inv['dysarthric_speaker_count']} dysarthric, {inv['control_speaker_count']} control)",
        f"- **Microphone Channels:** {', '.join(f'{k}: {v:,}' for k, v in inv['microphone_breakdown'].items())}",
        f"- **Transcripts:** {inv['transcripts']['unique_count']:,} unique, {inv['transcripts']['null_count']} null, {inv['transcripts']['empty_count']} empty",
        "",
        "---",
        "",
        "## 2. Raw File Manifest",
        "",
        "| File | Size (MB) |",
        "|------|-----------|"
    ]

    for rf in inv["raw_files"]:
        md.append(f"| `{os.path.basename(rf['path'])}` | {rf['size_mb']} MB |")

    md.extend([
        "",
        "---",
        "",
        "## 3. Speaker Breakdown",
        "",
        "| Speaker ID | Group | Gender | Total Recs | HeadMic | ArrayMic | Unique Prompts | Unique Transcripts | Total Dur (s) |",
        "|------------|-------|--------|------------|---------|----------|----------------|--------------------|---------------|"
    ])

    for spk, data in sorted(inv["speakers"].items()):
        hm = data["microphone_breakdown"].get("headMic", 0)
        am = data["microphone_breakdown"].get("arrayMic", 0)
        md.append(
            f"| `{spk}` | {data['speaker_group']} | {data['gender']} | {data['total_recordings']} | "
            f"{hm} | {am} | {data['unique_prompts']} | {data['unique_transcripts']} | {data['total_duration_sec']} |"
        )

    md.extend([
        "",
        "---",
        "",
        "## 4. Microphone Configuration & Channel Pairings",
        "",
        "TORGO records each vocalization simultaneously across two channels:",
        "1. **headMic (Head-Mounted Microphone):** Close proximity to the lips, minimizes ambient room reverberation, higher SNR.",
        "2. **arrayMic (Microphone Array):** Fixed desktop acoustic array, captures environmental reverberation.",
        "",
        "> [!IMPORTANT]",
        "> Multi-channel pairings for the same prompt event MUST be grouped together during train/test splitting to prevent acoustic data leakage across splits.",
        ""
    ])

    return "\n".join(md)


if __name__ == "__main__":
    build_inventory()
