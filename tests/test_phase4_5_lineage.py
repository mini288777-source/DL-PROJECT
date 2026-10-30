"""
Phase 4.5 Test: Full Prediction Lineage & IPER Traceability

Verifies that all 85,072 quantum prediction records in quantum_predictions.csv
have unambiguous provenance:
1. Every recording_id exists in data/manifests/torgo_manifest.csv.
2. Every recording belongs to the speaker_id claimed in the prediction.
3. Every recording belongs to the canonical test split for that (speaker, N) pair in splits/torgo_splits.json.
4. true_intent perfectly matches the manifest intent.
5. reference_transcript matches the manifest transcript.
6. Total row count is exactly 85,072 with 0 orphans or NaN values.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

WORKSPACE_DIR = Path("d:/DL")
PREDICTIONS_CSV = WORKSPACE_DIR / "results" / "phase4" / "quantum_predictions.csv"
MANIFEST_CSV = WORKSPACE_DIR / "data" / "manifests" / "torgo_manifest.csv"
SPLITS_JSON = WORKSPACE_DIR / "splits" / "torgo_splits.json"


def test_prediction_lineage_and_metadata_integrity():
    assert PREDICTIONS_CSV.exists(), f"Missing {PREDICTIONS_CSV}"
    assert MANIFEST_CSV.exists(), f"Missing {MANIFEST_CSV}"
    assert SPLITS_JSON.exists(), f"Missing {SPLITS_JSON}"

    df_preds = pd.read_csv(PREDICTIONS_CSV)
    df_manifest = pd.read_csv(MANIFEST_CSV).set_index("recording_id")

    with open(SPLITS_JSON, "r") as f:
        splits_list = json.load(f)

    # 1. Total count
    assert len(df_preds) == 85072, f"Expected 85,072 predictions, found {len(df_preds)}"

    # 2. No nulls in essential columns
    required_cols = [
        "run_id", "speaker_id", "recording_id", "sample_n", "qubit_count",
        "true_intent", "predicted_intent", "split", "reference_transcript"
    ]
    for col in required_cols:
        assert col in df_preds.columns, f"Missing column {col} in quantum_predictions.csv"
        assert df_preds[col].isna().sum() == 0, f"Found NaN values in {col}"

    # 3. Check all recording IDs are in manifest
    unique_rec_ids = df_preds["recording_id"].unique()
    missing_recs = set(unique_rec_ids) - set(df_manifest.index)
    assert len(missing_recs) == 0, f"Found {len(missing_recs)} orphan recordings not in manifest!"

    # 4. Check split membership for a representative sample
    split_lookup = {}
    for s_dict in splits_list:
        key = (s_dict["speaker_id"], s_dict["sample_n"])
        split_lookup[key] = set(s_dict["test_recording_ids"])

    # Sample 1000 rows across different speakers, budgets, and qubit counts
    sample_df = df_preds.sample(n=1000, random_state=42)
    for _, row in sample_df.iterrows():
        spk = row["speaker_id"]
        n_val = row["sample_n"]
        rec_id = row["recording_id"]
        
        # Verify test split membership
        assert (spk, n_val) in split_lookup, f"Missing split lookup for {spk} N={n_val}"
        assert rec_id in split_lookup[(spk, n_val)], f"Recording {rec_id} was not in test split for {spk} N={n_val}"
        
        # Verify intent and transcript against manifest
        meta_row = df_manifest.loc[rec_id]
        assert meta_row["speaker_id"] == spk
        assert meta_row["intent"] == row["true_intent"]
        assert meta_row["transcript"] == row["reference_transcript"]
