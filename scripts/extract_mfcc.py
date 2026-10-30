"""
Batch MFCC Feature Extraction Script.

Phase 2: Extracts deterministic classical acoustic MFCC baseline representations (78-dim).
Resumable execution. Multi-threaded processing. Stores individual .npy files per recording.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.cache import (
    EmbeddingManifestManager,
    get_embedding_path,
    has_cached_embedding,
    load_embedding_file,
    save_embedding_file,
)
from src.embeddings.mfcc import MFCCExtractor
from src.embeddings.whisper_preprocessing import load_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Batch MFCC Feature Extraction")
    parser.add_argument("--manifest", type=str, default="data/manifests/torgo_manifest.csv",
                        help="Path to TORGO manifest CSV")
    parser.add_argument("--output-dir", type=str, default="embeddings/torgo",
                        help="Base directory for cached embeddings")
    parser.add_argument("--manifest-out", type=str, default="embeddings/metadata/embedding_manifest.csv",
                        help="Path to output embedding manifest CSV")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("--n-mfcc", type=int, default=13,
                        help="Number of MFCC coefficients (default: 13)")
    parser.add_argument("--aggregation", type=str, default="mean_std",
                        help="Temporal aggregation strategy ('mean_std' -> 78-dim, 'mean' -> 39-dim)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of worker threads (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of recordings to process (for smoke testing/pilot)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cached MFCCs")
    parser.add_argument("--speakers", type=str, default=None,
                        help="Comma-separated list of speaker IDs to extract")
    parser.add_argument("--split-filter", type=str, default="adaptation,test",
                        help="Comma-separated list of splits to process (e.g. 'adaptation,test' or 'all')")
    return parser.parse_args()


def process_single_mfcc(row: pd.Series, extractor: MFCCExtractor, output_dir: str, overwrite: bool) -> Dict[str, Any]:
    """Worker function to process and save MFCC for a single recording."""
    rec_id = str(row["recording_id"])
    audio_path = str(row["audio_path"])

    if not overwrite and has_cached_embedding(rec_id, "mfcc", output_dir):
        try:
            arr = load_embedding_file(rec_id, "mfcc", output_dir, expected_dim=extractor.feature_dim)
            return {
                "recording_id": rec_id,
                "speaker_id": row["speaker_id"],
                "session_id": row["session_id"],
                "split": row["split"],
                "intent": row.get("intent", "UNKNOWN"),
                "audio_path": audio_path,
                "mfcc_path": get_embedding_path(rec_id, "mfcc", output_dir),
                "mfcc_dim": extractor.feature_dim,
                "status": "CACHED"
            }
        except Exception:
            pass  # Corrupt file, recompute

    try:
        wav, meta = load_audio(audio_path, target_sr=extractor.sr, normalize=True)
        if not meta.get("is_valid", True) or len(wav) == 0:
            raise ValueError(meta.get("error_msg", "Invalid audio"))

        feat = extractor.extract_from_waveform(wav, sr=extractor.sr)
        save_path = save_embedding_file(rec_id, feat, "mfcc", output_dir)
        return {
            "recording_id": rec_id,
            "speaker_id": row["speaker_id"],
            "session_id": row["session_id"],
            "split": row["split"],
            "intent": row.get("intent", "UNKNOWN"),
            "audio_path": audio_path,
            "mfcc_path": save_path,
            "mfcc_dim": extractor.feature_dim,
            "status": "EXTRACTED"
        }
    except Exception as e:
        return {
            "recording_id": rec_id,
            "speaker_id": row["speaker_id"],
            "session_id": row["session_id"],
            "split": row["split"],
            "intent": row.get("intent", "UNKNOWN"),
            "audio_path": audio_path,
            "status": f"FAILED: {e}"
        }


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("PHASE 2: BATCH MFCC FEATURE EXTRACTION")
    print("=" * 70)

    extractor = MFCCExtractor(
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_fft=400,
        hop_length=160,
        aggregation=args.aggregation
    )

    print(f"MFCC Configuration:")
    print(f"  Sample Rate:  {args.sr} Hz")
    print(f"  Coefficients: {args.n_mfcc} (+ delta + delta-delta = {args.n_mfcc * 3} channels)")
    print(f"  Aggregation:  {args.aggregation}")
    print(f"  Feature Dim:  {extractor.feature_dim}")
    print(f"  Workers:      {args.num_workers}")
    print(f"  Manifest:     {args.manifest}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Overwrite:    {args.overwrite}")

    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    print(f"Loaded manifest with {len(df)} total rows.")

    if args.speakers:
        target_spks = [s.strip() for s in args.speakers.split(",") if s.strip()]
        df = df[df["speaker_id"].isin(target_spks)]
        print(f"Filtered to {len(df)} recordings across speakers: {target_spks}")

    if args.split_filter and args.split_filter.lower() != "all":
        target_splits = [s.strip() for s in args.split_filter.split(",") if s.strip()]
        df = df[df["split"].isin(target_splits)]
        print(f"Filtered to {len(df)} recordings for splits: {target_splits}")

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit]
        print(f"Applied limit: processing {len(df)} recordings.")

    total_records = len(df)
    if total_records == 0:
        print("No recordings matching criteria. Exiting.")
        return

    manifest_mgr = EmbeddingManifestManager(manifest_path=args.manifest_out)

    num_cached = 0
    num_extracted = 0
    num_failed = 0

    rows_list = [row for _, row in df.iterrows()]
    print(f"\nProcessing {total_records} recordings with {args.num_workers} worker threads...")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_mfcc, r, extractor, args.output_dir, args.overwrite): r for r in rows_list}

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            rec_id = result["recording_id"]
            status = result["status"]

            if status == "CACHED":
                num_cached += 1
                manifest_mgr.upsert_record({
                    "recording_id": rec_id,
                    "speaker_id": result["speaker_id"],
                    "session_id": result["session_id"],
                    "split": result["split"],
                    "intent": result["intent"],
                    "audio_path": result["audio_path"],
                    "mfcc_path": result["mfcc_path"],
                    "mfcc_dim": result["mfcc_dim"],
                })
            elif status == "EXTRACTED":
                num_extracted += 1
                manifest_mgr.upsert_record({
                    "recording_id": rec_id,
                    "speaker_id": result["speaker_id"],
                    "session_id": result["session_id"],
                    "split": result["split"],
                    "intent": result["intent"],
                    "audio_path": result["audio_path"],
                    "mfcc_path": result["mfcc_path"],
                    "mfcc_dim": result["mfcc_dim"],
                })
            else:
                num_failed += 1
                print(f"[FAIL] {rec_id}: {status}")

            if (idx + 1) % 200 == 0 or (idx + 1) == total_records:
                manifest_mgr.save()
                elapsed = time.time() - start_time
                rate = (num_cached + num_extracted) / elapsed if elapsed > 0 else 0
                print(f"Progress: {idx + 1}/{total_records} completed ({num_extracted} new, {num_cached} cached, {rate:.1f} rec/s)")

    manifest_mgr.save()
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("MFCC EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total manifest targets: {total_records}")
    print(f"Previously cached:      {num_cached}")
    print(f"Newly extracted:        {num_extracted}")
    print(f"Total cached valid:     {num_cached + num_extracted}")
    print(f"Failed extractions:     {num_failed}")
    print(f"Total elapsed time:     {elapsed:.2f} s")
    print(f"Manifest updated:       {args.manifest_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
