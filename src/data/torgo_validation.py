"""
TORGO Audio and Transcript Validation Module.

Validates every candidate audio file and transcript in the TORGO dataset.
Extracts audio WAV files to data/torgo/audio/ to establish physical disk assets.
Generates:
    - reports/torgo_audio_validation.csv
    - reports/torgo_transcript_validation.json (and markdown summary)
"""

import glob
import io
import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import yaml


def validate_and_extract_torgo(
    config_path: str = "configs/torgo_config.yaml",
    extract_wavs: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate all audio files and transcripts, optionally writing WAV files to disk.

    Returns:
        audio_val_df: DataFrame with audio validation results per recording.
        summary_stats: Dictionary containing high-level validation statistics.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parquet_glob = config["dataset"]["parquet_glob"]
    extracted_audio_dir = config["dataset"]["extracted_audio_dir"]
    out_csv = config["paths"]["audio_validation_csv"]
    audio_rules = config["audio_validation"]

    if extract_wavs:
        os.makedirs(extracted_audio_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matching: {parquet_glob}")

    records = []
    transcript_records = []
    total_processed = 0
    valid_count = 0
    corrupt_count = 0

    t0 = time.time()
    print(f"Starting audio validation and extraction across {len(parquet_files)} parquet files...")

    for pf in parquet_files:
        print(f"  Validating {os.path.basename(pf)}...")
        df = pd.read_parquet(pf)
        for idx, row in df.iterrows():
            total_processed += 1
            b = row["audio"]["bytes"]
            orig_path = row["audio"]["path"]
            filename = os.path.basename(orig_path.replace("\\", "/"))
            rec_id = filename.rsplit(".", 1)[0]
            wav_path = os.path.join(extracted_audio_dir, filename).replace("\\", "/")

            transcript = row.get("transcription", "")
            transcript_str = str(transcript).strip() if transcript is not None else ""

            readable = True
            has_nan = False
            has_inf = False
            status = "VALID"
            reasons: List[str] = []
            duration_sec = 0.0
            sr = 0
            channels = 0

            # Optional physical extraction
            if extract_wavs and not os.path.exists(wav_path):
                try:
                    with open(wav_path, "wb") as f_out:
                        f_out.write(b)
                except Exception as e:
                    status = "WRITE_FAILED"
                    reasons.append(f"Disk write error: {e}")

            # Audio decoding and waveform checks
            try:
                with io.BytesIO(b) as bio:
                    data, sr = sf.read(bio)
                    duration_sec = len(data) / sr if sr > 0 else 0.0
                    channels = 1 if data.ndim == 1 else int(data.shape[1])

                    if np.isnan(data).any():
                        has_nan = True
                        status = "CORRUPT"
                        reasons.append("Waveform contains NaN")

                    if np.isinf(data).any():
                        has_inf = True
                        status = "CORRUPT"
                        reasons.append("Waveform contains Inf")

                    if duration_sec < audio_rules["min_duration_sec"]:
                        status = "INVALID_DURATION"
                        reasons.append(f"Duration {duration_sec:.2f}s < min {audio_rules['min_duration_sec']}s")

                    if duration_sec > audio_rules["max_duration_sec"]:
                        status = "INVALID_DURATION"
                        reasons.append(f"Duration {duration_sec:.2f}s > max {audio_rules['max_duration_sec']}s")

                    if sr != audio_rules["expected_sample_rate"]:
                        status = "UNEXPECTED_SR"
                        reasons.append(f"Sample rate {sr} != expected {audio_rules['expected_sample_rate']}")

                    if channels != audio_rules["expected_channels"]:
                        status = "UNEXPECTED_CHANNELS"
                        reasons.append(f"Channels {channels} != expected {audio_rules['expected_channels']}")

            except Exception as e:
                readable = False
                status = "UNREADABLE"
                reasons.append(f"Audio read error: {e}")

            if status == "VALID":
                valid_count += 1
            else:
                corrupt_count += 1

            records.append({
                "recording_id": rec_id,
                "audio_path": wav_path,
                "readable": readable,
                "duration_sec": round(duration_sec, 4),
                "sample_rate": sr,
                "channels": channels,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "status": status,
                "reason": "; ".join(reasons) if reasons else "OK"
            })

            transcript_records.append({
                "recording_id": rec_id,
                "transcript": transcript_str,
                "has_transcript": len(transcript_str) > 0,
                "word_count": len(transcript_str.split()) if transcript_str else 0
            })

    elapsed = time.time() - t0
    print(f"Validation complete: {total_processed} files in {elapsed:.2f}s ({total_processed/elapsed:.1f} files/s).")

    val_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    val_df.to_csv(out_csv, index=False)
    print(f"Saved audio validation report: {out_csv}")

    # Transcript validation metrics
    tr_df = pd.DataFrame(transcript_records)
    total_audio = len(tr_df)
    with_trans = int(tr_df["has_transcript"].sum())
    without_trans = total_audio - with_trans
    unique_trans = int(tr_df["transcript"].nunique())

    summary_stats = {
        "total_audio_files": total_audio,
        "valid_audio_files": valid_count,
        "corrupted_audio_files": corrupt_count,
        "audio_files_with_transcripts": with_trans,
        "audio_files_without_transcripts": without_trans,
        "unique_transcripts": unique_trans,
        "elapsed_sec": round(elapsed, 2)
    }

    # Save transcript validation JSON
    tr_json_path = "reports/torgo_transcript_validation.json"
    with open(tr_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    return val_df, summary_stats


if __name__ == "__main__":
    validate_and_extract_torgo()
