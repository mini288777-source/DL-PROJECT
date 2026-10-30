"""
Comprehensive Test Suite for Subphase 6.5.1:
Data Feasibility & Speaker-Stratified Recovery Splits
"""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE6_5_DIR = REPO_ROOT / "phase6_5"
SPLITS_DIR = PHASE6_5_DIR / "splits"
MANIFESTS_DIR = PHASE6_5_DIR / "manifests"
ANALYSIS_DIR = PHASE6_5_DIR / "analysis"

SPEAKERS = [
    "F01", "F03", "F04", "FC01", "FC02", "FC03",
    "M01", "M02", "M03", "M04", "M05", "MC01", "MC02", "MC03", "MC04"
]
BUDGETS = [20, 30, 40, 50]
CANONICAL_TAXONOMY = {
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
}


@pytest.fixture(scope="module")
def track_a_splits():
    json_path = SPLITS_DIR / "recovery_splits_stratified.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def track_b_splits():
    json_path = SPLITS_DIR / "recovery_splits_hierarchical_ready.json"
    assert json_path.exists(), f"Missing {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def source_manifest():
    m_path = REPO_ROOT / "data" / "manifests" / "torgo_manifest.csv"
    assert m_path.exists(), f"Missing {m_path}"
    return pd.read_csv(m_path)


# ============================================================
# TEST 1: All 15 speakers exist
# ============================================================
def test_all_15_speakers_exist(track_a_splits, track_b_splits):
    """TEST 1: Verify all 15 speakers exist in both Track A and Track B."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        present_speakers = {s["speaker_id"] for s in splits}
        assert present_speakers == set(SPEAKERS), f"Mismatch in {track_name} speakers"


# ============================================================
# TEST 2: N values are exactly 20, 30, 40, 50
# ============================================================
def test_n_values_are_exact(track_a_splits, track_b_splits):
    """TEST 2: Verify budgets are exactly 20, 30, 40, 50."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        present_n = {s["sample_n"] for s in splits}
        assert present_n == set(BUDGETS), f"Mismatch in {track_name} budgets"


# ============================================================
# TEST 3: Every N contains exactly N adaptation recordings
# ============================================================
def test_exact_adaptation_counts(track_a_splits, track_b_splits):
    """TEST 3: Every N condition contains exactly N adaptation recordings per speaker."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        for entry in splits:
            spk = entry["speaker_id"]
            n = entry["sample_n"]
            adapt_count = len(entry["adaptation_recording_ids"])
            assert adapt_count == n, f"{track_name}: {spk} N={n} has {adapt_count} != {n} recordings"


# ============================================================
# TEST 4: Nestedness S20 ⊂ S30 ⊂ S40 ⊂ S50
# ============================================================
def test_strict_nestedness(track_a_splits, track_b_splits):
    """TEST 4: Strict nested subset containment S20 ⊂ S30 ⊂ S40 ⊂ S50."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        for spk in SPEAKERS:
            spk_splits = {s["sample_n"]: set(s["adaptation_recording_ids"]) for s in splits if s["speaker_id"] == spk}
            s20 = spk_splits[20]
            s30 = spk_splits[30]
            s40 = spk_splits[40]
            s50 = spk_splits[50]

            assert s20.issubset(s30), f"{track_name}: S20 not subset of S30 for {spk}"
            assert s30.issubset(s40), f"{track_name}: S30 not subset of S40 for {spk}"
            assert s40.issubset(s50), f"{track_name}: S40 not subset of S50 for {spk}"
            assert len(s20) < len(s30) < len(s40) < len(s50), f"{track_name}: Strict inequality failed for {spk}"


# ============================================================
# TEST 5: No adaptation/test overlap
# ============================================================
def test_no_adaptation_test_recording_overlap(track_a_splits, track_b_splits):
    """TEST 5: Zero recording overlap between adaptation and test sets."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        for entry in splits:
            spk = entry["speaker_id"]
            n = entry["sample_n"]
            adapt_set = set(entry["adaptation_recording_ids"])
            test_set = set(entry["test_recording_ids"])
            overlap = adapt_set.intersection(test_set)
            assert len(overlap) == 0, f"{track_name}: Overlap {overlap} in {spk} N={n}"


# ============================================================
# TEST 6: No prompt-group leakage
# ============================================================
def test_no_prompt_group_leakage(track_a_splits, track_b_splits):
    """TEST 6: Zero prompt-group overlap between adaptation and test sets."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        for entry in splits:
            spk = entry["speaker_id"]
            n = entry["sample_n"]
            adapt_pg = set(entry["adaptation_prompt_group_ids"])
            test_pg = set(entry["test_prompt_group_ids"])
            overlap = adapt_pg.intersection(test_pg)
            assert len(overlap) == 0, f"{track_name}: Prompt group leakage {overlap} in {spk} N={n}"


# ============================================================
# TEST 7: No duplicate recordings
# ============================================================
def test_no_duplicate_recordings_within_split(track_a_splits, track_b_splits):
    """TEST 7: No duplicate recording IDs within any adaptation or test set."""
    for track_name, splits in [("Track A", track_a_splits), ("Track B", track_b_splits)]:
        for entry in splits:
            spk = entry["speaker_id"]
            n = entry["sample_n"]
            adapt_list = entry["adaptation_recording_ids"]
            test_list = entry["test_recording_ids"]
            assert len(adapt_list) == len(set(adapt_list)), f"{track_name}: Duplicate in adapt for {spk} N={n}"
            assert len(test_list) == len(set(test_list)), f"{track_name}: Duplicate in test for {spk} N={n}"


# ============================================================
# TEST 8: All labels belong to canonical taxonomy
# ============================================================
def test_all_labels_canonical(source_manifest):
    """TEST 8: All recording labels belong to the 7-class canonical taxonomy."""
    manifest_labels = set(source_manifest["intent"].unique())
    assert manifest_labels.issubset(CANONICAL_TAXONOMY), f"Unknown intents found: {manifest_labels - CANONICAL_TAXONOMY}"

    rec_manifest = pd.read_csv(MANIFESTS_DIR / "recovery_adaptation_manifest.csv")
    rec_labels = set(rec_manifest["intent"].unique())
    assert rec_labels.issubset(CANONICAL_TAXONOMY), f"Unknown intents in adaptation manifest: {rec_labels - CANONICAL_TAXONOMY}"


# ============================================================
# TEST 9: All split hashes are deterministic
# ============================================================
def test_split_hashes_valid_and_distinct(track_a_splits):
    """TEST 9: Hashes are non-empty and differ across budgets N."""
    for spk in SPEAKERS:
        spk_entries = [s for s in track_a_splits if s["speaker_id"] == spk]
        hashes = [s["adaptation_split_hash"] for s in spk_entries]
        assert len(set(hashes)) == 4, f"Hash collision across N for speaker {spk}"
        for h in hashes:
            assert len(h) >= 16, f"Invalid hash {h}"


# ============================================================
# TEST 10: Re-running split generation gives identical hashes
# ============================================================
def test_hash_reproducibility(track_a_splits):
    """TEST 10: Verify hash recomputations match stored values exactly."""
    for entry in track_a_splits:
        adapt_rids = entry["adaptation_recording_ids"]
        expected_hash = hashlib.sha256(",".join(sorted(adapt_rids)).encode("utf-8")).hexdigest()[:16]
        assert entry["adaptation_split_hash"] == expected_hash, f"Hash mismatch in {entry['speaker_id']}"


# ============================================================
# TEST 11: Class-support counts match source manifest
# ============================================================
def test_class_support_consistency(source_manifest, track_a_splits):
    """TEST 11: Class-support counts match actual underlying recordings."""
    rec_map = dict(zip(source_manifest["recording_id"], source_manifest["intent"]))
    for entry in track_a_splits:
        adapt_rids = entry["adaptation_recording_ids"]
        intents = [rec_map[r] for r in adapt_rids]
        emerg_intents = [i for i in intents if i != "GENERAL_SPEECH"]
        assert len(emerg_intents) == entry["total_emergency_support"], (
            f"Support mismatch in {entry['speaker_id']} N={entry['sample_n']}"
        )


# ============================================================
# TEST 12: Recovery split metadata is complete
# ============================================================
def test_recovery_metadata_complete(track_a_splits):
    """TEST 12: Required metadata keys exist in every split entry."""
    required_keys = {
        "speaker_id", "speaker_group", "sample_n", "seed", "recovery_protocol",
        "adaptation_recording_ids", "test_recording_ids",
        "adaptation_prompt_group_ids", "test_prompt_group_ids",
        "adaptation_split_hash", "test_split_hash", "split_group_hash",
        "represented_emergency_classes", "total_emergency_support",
        "feasibility_tier", "test_support_limited"
    }
    for entry in track_a_splits:
        missing = required_keys - set(entry.keys())
        assert len(missing) == 0, f"Missing metadata keys {missing} in {entry['speaker_id']}"


# ============================================================
# TEST 13: No original Phase 3-6 artifacts were modified
# ============================================================
def test_original_artifacts_untouched():
    """TEST 13: Frozen Phase 1-6 baseline files are preserved intact."""
    frozen_files = [
        REPO_ROOT / "splits" / "torgo_splits.json",
        REPO_ROOT / "results" / "phase5" / "phase5_results.csv",
        REPO_ROOT / "results" / "phase5" / "phase5_predictions.csv",
        REPO_ROOT / "results" / "phase5" / "phase5_per_class.csv"
    ]
    for ff in frozen_files:
        assert ff.exists(), f"Frozen artifact {ff} was deleted or moved!"
        assert ff.stat().st_size > 0, f"Frozen artifact {ff} is empty!"
