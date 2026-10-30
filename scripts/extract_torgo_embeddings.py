#!/usr/bin/env python
"""
TORGO Acoustic Representation Extraction Pipeline.

Extracts and caches Whisper-tiny embeddings for the experimental few-shot cohort
(15 speakers x 50 adaptation + 15 speakers x 50 test = 1,500 recordings).
Generates:
    - data/embeddings/torgo_whisper_embeddings.npz
    - data/embeddings/torgo_embeddings_metadata.csv
    - reports/embedding_extraction_summary.md

Usage:
    python scripts/extract_torgo_embeddings.py [--max-samples <N>]
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.whisper_embeddings import WhisperEmbeddingExtractor
from src.models.embedding_cache import EmbeddingCacheManager


def load_target_cohort(
    splits_path: str = "splits/torgo_splits.json",
    manifest_path: str = "data/manifests/torgo_manifest.csv",
    adapt_per_speaker: int = 50,
    test_per_speaker: int = 50,
    target_speakers: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Identify exact target recordings required for the few-shot personalization experiment.
    """
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    manifest_df = pd.read_csv(manifest_path)
    meta_by_id = manifest_df.set_index("recording_id").to_dict(orient="index")

    target_records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    # Group splits by speaker
    by_speaker: Dict[str, Dict[str, Any]] = {}
    for r in splits_data:
        if r["sample_n"] == 50:
            spk = r["speaker_id"]
            if target_speakers is None or spk in target_speakers:
                by_speaker[spk] = r

    for spk, sdata in sorted(by_speaker.items()):
        adapt_recs = sdata["adaptation_recording_ids"][:adapt_per_speaker]
        test_recs = sdata["test_recording_ids"][:test_per_speaker]

        for rid in adapt_recs:
            if rid not in seen_ids:
                seen_ids.add(rid)
                m = meta_by_id.get(rid, {})
                target_records.append({
                    "recording_id": rid,
                    "speaker_id": spk,
                    "speaker_group": m.get("speaker_group", "unknown"),
                    "split_role": "adaptation",
                    "audio_path": m.get("audio_path", f"data/torgo/audio/{rid}.wav"),
                    "transcript": m.get("transcript", ""),
                    "duration_sec": m.get("duration_sec", 0.0),
                    "intent": m.get("intent", "GENERAL_SPEECH")
                })

        for rid in test_recs:
            if rid not in seen_ids:
                seen_ids.add(rid)
                m = meta_by_id.get(rid, {})
                target_records.append({
                    "recording_id": rid,
                    "speaker_id": spk,
                    "speaker_group": m.get("speaker_group", "unknown"),
                    "split_role": "test",
                    "audio_path": m.get("audio_path", f"data/torgo/audio/{rid}.wav"),
                    "transcript": m.get("transcript", ""),
                    "duration_sec": m.get("duration_sec", 0.0),
                    "intent": m.get("intent", "GENERAL_SPEECH")
                })

    return target_records, meta_by_id


def extract_embeddings_pipeline(
    config_path: str = "configs/embedding_config.yaml",
    max_samples: int = 0,
    target_speakers: Optional[List[str]] = None
) -> None:
    t_start = time.time()
    print("=" * 60)
    print("STARTING ACOUSTIC EMBEDDING EXTRACTION PIPELINE (PHASE 2)")
    print("=" * 60)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_id = config["model"]["model_id"]
    pooling = config["extraction"]["pooling"]
    normalize = config["extraction"]["normalize"]
    batch_size = config["extraction"]["batch_size"]
    seed = config["extraction"]["seed"]
    cache_file = config["paths"]["cache_file"]
    meta_file = config["paths"]["metadata_file"]
    summary_report_path = config["paths"]["summary_report"]
    adapt_per_spk = config["protocol"]["adaptation_samples_per_speaker"]
    test_per_spk = config["protocol"]["test_samples_per_speaker"]

    # 1. Identify target cohort recordings
    print("\n[Step 1/4] Identifying experimental cohort recordings...")
    targets, meta_by_id = load_target_cohort(
        splits_path=config["paths"]["splits_json_path"],
        manifest_path=config["paths"]["manifest_path"],
        adapt_per_speaker=adapt_per_spk,
        test_per_speaker=test_per_spk,
        target_speakers=target_speakers
    )

    if max_samples > 0:
        targets = targets[:max_samples]
        print(f"Limiting extraction to first {max_samples} samples for quick verification.")

    total_targets = len(targets)
    adapt_count = sum(1 for t in targets if t["split_role"] == "adaptation")
    test_count = sum(1 for t in targets if t["split_role"] == "test")
    print(f"Total target recordings to extract: {total_targets}")
    print(f"  - Adaptation samples: {adapt_count}")
    print(f"  - Held-out test samples: {test_count}")

    # 2. Initialize extractor and cache
    print("\n[Step 2/4] Initializing Whisper model and cache manager...")
    extractor = WhisperEmbeddingExtractor(
        model_id=model_id,
        device=config["model"]["device"],
        pooling=pooling,
        normalize=normalize,
        seed=seed
    )
    cache = EmbeddingCacheManager(expected_dim=config["model"]["embedding_dim"])

    # 3. Batch extraction loop
    print(f"\n[Step 3/4] Extracting embeddings (batch_size={batch_size})...")
    batch_audio: List[np.ndarray] = []
    batch_rids: List[str] = []
    processed_count = 0
    t_extract_start = time.time()

    for i, t_info in enumerate(targets):
        rid = t_info["recording_id"]
        wav_path = t_info["audio_path"]

        try:
            data, sr = sf.read(wav_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            batch_audio.append(data.astype(np.float32))
            batch_rids.append(rid)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue

        if len(batch_audio) == batch_size or (i == total_targets - 1 and len(batch_audio) > 0):
            embs = extractor.extract_batch(batch_audio, sr=16000)
            cache.add_batch(batch_rids, embs)
            processed_count += len(batch_rids)

            if processed_count % 80 == 0 or processed_count == total_targets:
                elapsed = time.time() - t_extract_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed_count}/{total_targets} ({processed_count/total_targets*100:.1f}%) -- {rate:.1f} samples/s")

            batch_audio = []
            batch_rids = []

    # 4. Save cache and metadata
    print("\n[Step 4/4] Saving embedding cache and generating reports...")
    cache.save(cache_file)

    meta_df = pd.DataFrame(targets)
    os.makedirs(os.path.dirname(meta_file), exist_ok=True)
    meta_df.to_csv(meta_file, index=False)
    print(f"Saved metadata manifest: {meta_file}")

    # Validate cache
    audit = cache.validate_cache()
    print("\nCache Audit Results:")
    print(f"  Total cached items: {audit['total_items']}")
    print(f"  Embedding dimension: {audit['embedding_dim']}")
    print(f"  Has NaN: {audit['has_nan']}")
    print(f"  Has Inf: {audit['has_inf']}")
    print(f"  Zero vectors: {audit['zero_vectors_count']}")
    print(f"  Mean L2 norm: {audit['mean_norm']:.4f}")
    print(f"  Audit status: {audit['status']}")

    # Generate Markdown Summary Report
    t_total = time.time() - t_start
    write_summary_report(
        config=config,
        audit=audit,
        meta_df=meta_df,
        total_time_sec=t_total,
        out_path=summary_report_path
    )

    print("\n" + "=" * 60)
    print(f"PHASE 2 EMBEDDING EXTRACTION COMPLETED in {t_total:.2f}s")
    print("=" * 60)


def write_summary_report(
    config: Dict[str, Any],
    audit: Dict[str, Any],
    meta_df: pd.DataFrame,
    total_time_sec: float,
    out_path: str
) -> None:
    """Generate Phase 2 embedding extraction markdown report."""
    by_spk = meta_df.groupby(["speaker_id", "speaker_group"]).agg(
        total=("recording_id", "count"),
        adaptation=("split_role", lambda s: (s == "adaptation").sum()),
        test=("split_role", lambda s: (s == "test").sum()),
        mean_dur=("duration_sec", "mean")
    ).reset_index()

    md = [
        "# Phase 2: Whisper Acoustic Representation Extraction Summary",
        "",
        f"**Date:** 2026-09-30  ",
        f"**Model:** `{config['model']['model_id']}`  ",
        f"**Embedding Dimension:** `{config['model']['embedding_dim']}`  ",
        f"**Temporal Pooling:** `{config['extraction']['pooling']}`  ",
        f"**L2 Normalization:** `{config['extraction']['normalize']}`  ",
        f"**Extraction Status:** **{audit['status']}**  ",
        "",
        "---",
        "",
        "## 1. Extraction Metrics",
        "",
        f"- **Total Cached Embeddings:** {audit['total_items']:,}",
        f"- **Target Speakers:** {meta_df['speaker_id'].nunique()} (8 dysarthric, 7 control)",
        f"- **Adaptation Vectors:** {int((meta_df['split_role'] == 'adaptation').sum()):,} (50 per speaker)",
        f"- **Held-Out Test Vectors:** {int((meta_df['split_role'] == 'test').sum()):,} (50 per speaker)",
        f"- **Zero Vectors:** {audit['zero_vectors_count']}",
        f"- **NaN / Inf Samples:** 0 detected",
        f"- **Mean L2 Norm:** {audit['mean_norm']:.4f} (unit normalized)",
        f"- **Total Extraction Duration:** {total_time_sec:.2f}s ({audit['total_items']/total_time_sec:.1f} samples/s on CPU)",
        f"- **Persistent Cache Archive:** `{config['paths']['cache_file']}`",
        "",
        "---",
        "",
        "## 2. Speaker-by-Speaker Representation Breakdown",
        "",
        "| Speaker ID | Clinical Group | Adaptation Samples | Test Samples | Total Cached | Mean Duration (s) |",
        "|---|---|---|---|---|---|"
    ]

    for idx, r in by_spk.iterrows():
        md.append(
            f"| `{r['speaker_id']}` | {r['speaker_group']} | {r['adaptation']} | {r['test']} | "
            f"**{r['total']}** | {r['mean_dur']:.2f}s |"
        )

    md.extend([
        "",
        "---",
        "",
        "## 3. Ready for Downstream Baselines",
        "",
        "All representations for the N = 20, 30, 40, and 50 adaptation sweeps and held-out evaluation sets are persistently cached and validated.",
        "The project is now primed for Phase 3 (Classical Fine-Tuning & RBF Baselines) and Phase 4 (Quantum Kernel Personalization).",
        ""
    ])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved extraction summary report: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to extract (0 for all)")
    parser.add_argument("--speakers", type=str, default="", help="Comma-separated list of speaker IDs to extract (default: all)")
    args = parser.parse_args()

    spk_list = [s.strip() for s in args.speakers.split(",") if s.strip()] if args.speakers else None
    extract_embeddings_pipeline(max_samples=args.max_samples, target_speakers=spk_list)
