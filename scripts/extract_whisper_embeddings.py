"""
Batch Whisper Embedding Extraction Script.

Phase 2: Extracts fixed-dimensional canonical Whisper embeddings for audio recordings.
Strictly frozen encoder. Resumable execution. Stores individual .npy files per recording.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings.cache import (
    EmbeddingManifestManager,
    get_embedding_path,
    has_cached_embedding,
    load_embedding_file,
    save_embedding_file,
)
from src.embeddings.whisper_encoder import WhisperEncoder
from src.embeddings.whisper_preprocessing import load_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Whisper Embedding Extraction")
    parser.add_argument("--manifest", type=str, default="data/manifests/torgo_manifest.csv",
                        help="Path to TORGO manifest CSV")
    parser.add_argument("--output-dir", type=str, default="embeddings/torgo",
                        help="Base directory for cached embeddings")
    parser.add_argument("--manifest-out", type=str, default="embeddings/metadata/embedding_manifest.csv",
                        help="Path to output embedding manifest CSV")
    parser.add_argument("--config", type=str, default="configs/embedding_config.yaml",
                        help="Path to embedding configuration YAML")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Whisper model ID (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' or 'cpu' (overrides config)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for Whisper encoder inference (default: 32)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of recordings to process (for smoke testing/pilot)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cached embeddings")
    parser.add_argument("--speakers", type=str, default=None,
                        help="Comma-separated list of speaker IDs to extract")
    parser.add_argument("--split-filter", type=str, default="adaptation,test",
                        help="Comma-separated list of splits to process (e.g. 'adaptation,test' or 'all')")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("PHASE 2: BATCH WHISPER EMBEDDING EXTRACTION")
    print("=" * 70)

    # Load configuration
    cfg = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    extract_cfg = cfg.get("extraction", {})

    model_id = args.model_id or model_cfg.get("model_id", "openai/whisper-tiny")
    device = args.device or model_cfg.get("device", "cpu")
    pooling = extract_cfg.get("pooling", "mean")
    normalize = extract_cfg.get("normalize", True)
    seed = extract_cfg.get("seed", 42)
    batch_size = args.batch_size or extract_cfg.get("batch_size", 8)

    print(f"Configuration:")
    print(f"  Model ID:     {model_id}")
    print(f"  Device:       {device}")
    print(f"  Pooling:      {pooling}")
    print(f"  L2 Normalize: {normalize}")
    print(f"  Batch Size:   {batch_size}")
    print(f"  Manifest:     {args.manifest}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Overwrite:    {args.overwrite}")

    # Load dataset manifest
    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    print(f"Loaded manifest with {len(df)} total rows.")

    # Filter speakers if requested
    if args.speakers:
        target_spks = [s.strip() for s in args.speakers.split(",") if s.strip()]
        df = df[df["speaker_id"].isin(target_spks)]
        print(f"Filtered to {len(df)} recordings across speakers: {target_spks}")

    # Filter splits if requested
    if args.split_filter and args.split_filter.lower() != "all":
        target_splits = [s.strip() for s in args.split_filter.split(",") if s.strip()]
        df = df[df["split"].isin(target_splits)]
        print(f"Filtered to {len(df)} recordings for splits: {target_splits}")

    # Apply limit if requested
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit]
        print(f"Applied limit: processing {len(df)} recordings.")

    total_records = len(df)
    if total_records == 0:
        print("No recordings matching criteria. Exiting.")
        return

    # Initialize frozen Whisper encoder
    print(f"\nInitializing frozen Whisper encoder...")
    encoder = WhisperEncoder(
        model_id=model_id,
        device=device,
        pooling=pooling,
        normalize=normalize,
        seed=seed
    )
    print(f"Encoder ready. Hidden dimension: {encoder.hidden_dim}")

    # Initialize manifest manager
    manifest_mgr = EmbeddingManifestManager(manifest_path=args.manifest_out)

    num_cached = 0
    num_extracted = 0
    num_failed = 0
    failed_recs = []

    # Identify which recordings need extraction vs already cached
    records_to_process = []
    for _, row in df.iterrows():
        rec_id = str(row["recording_id"])
        if not args.overwrite and has_cached_embedding(rec_id, "whisper", args.output_dir):
            try:
                # Validate that existing cached file is valid
                arr = load_embedding_file(rec_id, "whisper", args.output_dir, expected_dim=encoder.hidden_dim)
                num_cached += 1
                manifest_mgr.upsert_record({
                    "recording_id": rec_id,
                    "speaker_id": row["speaker_id"],
                    "session_id": row["session_id"],
                    "split": row["split"],
                    "intent": row.get("intent", "UNKNOWN"),
                    "audio_path": row["audio_path"],
                    "whisper_path": get_embedding_path(rec_id, "whisper", args.output_dir),
                    "whisper_dim": encoder.hidden_dim,
                    "whisper_dtype": str(arr.dtype),
                    "duration_sec": row.get("duration_sec", 0.0),
                    "sample_rate": row.get("sample_rate", 16000),
                    "model_name": model_id,
                    "pooling_method": pooling,
                    "extraction_status": "SUCCESS",
                })
                continue
            except Exception as e:
                # If cached file was corrupted, recompute
                print(f"Cached file for {rec_id} invalid ({e}), will re-extract.")
        records_to_process.append(row)

    print(f"Recordings already cached and valid: {num_cached}/{total_records}")
    print(f"Recordings to extract: {len(records_to_process)}/{total_records}")

    # Batch extraction loop
    from concurrent.futures import ThreadPoolExecutor

    def load_item(item):
        b_idx, row = item
        rec_id = str(row["recording_id"])
        audio_path = str(row["audio_path"])
        try:
            wav, meta = load_audio(audio_path, target_sr=16000, normalize=True)
            if not meta.get("is_valid", True) or len(wav) == 0:
                raise ValueError(meta.get("error_msg", "Invalid audio"))
            return b_idx, rec_id, audio_path, wav, None
        except Exception as e:
            return b_idx, rec_id, audio_path, None, str(e)

    for i in range(0, len(records_to_process), batch_size):
        batch_rows = records_to_process[i:i + batch_size]
        items = list(enumerate(batch_rows))

        with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
            load_results = list(pool.map(load_item, items))

        waveforms = []
        valid_indices = []
        for b_idx, rec_id, audio_path, wav, err in load_results:
            row = batch_rows[b_idx]
            if err is not None:
                num_failed += 1
                failed_recs.append((rec_id, err))
                print(f"[FAIL] Audio load failed for {rec_id}: {err}")
                manifest_mgr.upsert_record({
                    "recording_id": rec_id,
                    "speaker_id": row["speaker_id"],
                    "session_id": row["session_id"],
                    "split": row["split"],
                    "intent": row.get("intent", "UNKNOWN"),
                    "audio_path": audio_path,
                    "extraction_status": f"FAILED_AUDIO: {err}",
                })
            else:
                waveforms.append(wav)
                valid_indices.append(b_idx)

        if not waveforms:
            continue

        try:
            # Run batched inference through frozen encoder
            embeddings = encoder.extract_batch(waveforms, sr=16000)

            # Save individual .npy files
            for local_idx, b_idx in enumerate(valid_indices):
                row = batch_rows[b_idx]
                rec_id = str(row["recording_id"])
                emb = embeddings[local_idx]

                save_path = save_embedding_file(rec_id, emb, "whisper", args.output_dir)
                num_extracted += 1

                manifest_mgr.upsert_record({
                    "recording_id": rec_id,
                    "speaker_id": row["speaker_id"],
                    "session_id": row["session_id"],
                    "split": row["split"],
                    "intent": row.get("intent", "UNKNOWN"),
                    "audio_path": row["audio_path"],
                    "whisper_path": save_path,
                    "whisper_dim": encoder.hidden_dim,
                    "whisper_dtype": str(emb.dtype),
                    "duration_sec": row.get("duration_sec", 0.0),
                    "sample_rate": row.get("sample_rate", 16000),
                    "model_name": model_id,
                    "pooling_method": pooling,
                    "extraction_status": "SUCCESS",
                })
        except Exception as e:
            for b_idx in valid_indices:
                row = batch_rows[b_idx]
                rec_id = str(row["recording_id"])
                num_failed += 1
                failed_recs.append((rec_id, str(e)))
                print(f"[FAIL] Inference failed for {rec_id}: {e}")
                manifest_mgr.upsert_record({
                    "recording_id": rec_id,
                    "speaker_id": row["speaker_id"],
                    "session_id": row["session_id"],
                    "split": row["split"],
                    "intent": row.get("intent", "UNKNOWN"),
                    "audio_path": row["audio_path"],
                    "extraction_status": f"FAILED_INFERENCE: {e}",
                })

        # Save manifest progress periodically
        if (i // batch_size) % 10 == 0:
            manifest_mgr.save()
            progress = min(i + batch_size, len(records_to_process))
            elapsed = time.time() - start_time
            rate = (num_extracted) / elapsed if elapsed > 0 else 0
            print(f"Progress: {progress}/{len(records_to_process)} extracted ({num_extracted} done, {rate:.1f} rec/s)")

    # Final save of manifest
    manifest_mgr.save()
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("WHISPER EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total manifest targets: {total_records}")
    print(f"Previously cached:      {num_cached}")
    print(f"Newly extracted:        {num_extracted}")
    print(f"Total cached valid:     {num_cached + num_extracted}")
    print(f"Failed extractions:     {num_failed}")
    print(f"Total elapsed time:     {elapsed:.2f} s")
    print(f"Manifest written to:    {args.manifest_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
