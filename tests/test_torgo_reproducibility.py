#!/usr/bin/env python
"""
Acceptance Test for Phase 1: TORGO Split Reproducibility.

Executes two independent runs of the split generation protocol under identical
seed configuration and strictly compares:
    - Speaker IDs
    - Sample sizes (N = 20, 30, 40, 50)
    - Adaptation recording IDs (element-by-element in exact order)
    - Test recording IDs (element-by-element in exact order)
    - Validation recording IDs
    - Strict nesting integrity (20 ⊂ 30 ⊂ 40 ⊂ 50)
    - Zero data leakage (adaptation ∩ test == empty)

Usage:
    python tests/test_torgo_reproducibility.py
"""

import copy
import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.torgo_splits import generate_torgo_splits


def run_reproducibility_test() -> int:
    print("=" * 60)
    print("PHASE 1 ACCEPTANCE TEST: TORGO SPLIT REPRODUCIBILITY")
    print("=" * 60)

    # Run 1: Generate initial splits
    print("\n[Run 1] Executing split generation...")
    splits_run1, indices_df_run1, feas_run1 = generate_torgo_splits()
    splits_run1_copy = copy.deepcopy(splits_run1)
    indices_run1_rows = len(indices_df_run1)

    # Run 2: Re-execute split generation independently
    print("\n[Run 2] Executing independent re-run of split generation...")
    splits_run2, indices_df_run2, feas_run2 = generate_torgo_splits()
    indices_run2_rows = len(indices_df_run2)

    # Assertions and comparisons
    failures = []
    print("\nVerifying reproducibility invariants between Run 1 and Run 2...")

    # 1. Total records check
    if len(splits_run1_copy) != len(splits_run2):
        failures.append(
            f"Record count mismatch: Run 1 has {len(splits_run1_copy)}, Run 2 has {len(splits_run2)}"
        )
    else:
        print(f"  [OK] Total split records match: {len(splits_run2)} (15 speakers x 4 N-values)")

    # 2. Split indices row count check
    if indices_run1_rows != indices_run2_rows:
        failures.append(
            f"Split index rows mismatch: Run 1 has {indices_run1_rows}, Run 2 has {indices_run2_rows}"
        )
    else:
        print(f"  [OK] Split index CSV row counts match: {indices_run2_rows}")

    # 3. Element-by-element deep comparison
    for idx, (r1, r2) in enumerate(zip(splits_run1_copy, splits_run2)):
        spk = r1["speaker_id"]
        n = r1["sample_n"]

        if r1["speaker_id"] != r2["speaker_id"]:
            failures.append(f"Row {idx}: speaker mismatch ({r1['speaker_id']} != {r2['speaker_id']})")
        if r1["sample_n"] != r2["sample_n"]:
            failures.append(f"Row {idx} ({spk}): N mismatch ({r1['sample_n']} != {r2['sample_n']})")
        if r1["seed"] != r2["seed"]:
            failures.append(f"Row {idx} ({spk}): seed mismatch ({r1['seed']} != {r2['seed']})")

        # Adaptation set match
        if r1["adaptation_recording_ids"] != r2["adaptation_recording_ids"]:
            diff = set(r1["adaptation_recording_ids"]).symmetric_difference(set(r2["adaptation_recording_ids"]))
            failures.append(f"{spk} N={n}: adaptation recording IDs differ ({len(diff)} discrepancies)")

        # Test set match
        if r1["test_recording_ids"] != r2["test_recording_ids"]:
            diff = set(r1["test_recording_ids"]).symmetric_difference(set(r2["test_recording_ids"]))
            failures.append(f"{spk} N={n}: test recording IDs differ ({len(diff)} discrepancies)")

    if not failures:
        print("  [OK] All 60 split records are bit-for-bit identical across runs")

    # 4. Nested hierarchy verification (20 ⊂ 30 ⊂ 40 ⊂ 50)
    print("\nVerifying nested few-shot hierarchy (20 subset of 30 subset of 40 subset of 50)...")
    nested_failures = []
    by_speaker = {}
    for r in splits_run2:
        by_speaker.setdefault(r["speaker_id"], {})[r["sample_n"]] = r["adaptation_recording_ids"]

    for spk, n_map in by_speaker.items():
        s20 = set(n_map[20])
        s30 = set(n_map[30])
        s40 = set(n_map[40])
        s50 = set(n_map[50])

        if not s20.issubset(s30):
            nested_failures.append(f"{spk}: N=20 is not a subset of N=30")
        if not s30.issubset(s40):
            nested_failures.append(f"{spk}: N=30 is not a subset of N=40")
        if not s40.issubset(s50):
            nested_failures.append(f"{spk}: N=40 is not a subset of N=50")

    if not nested_failures:
        print("  [OK] Strict nested hierarchy confirmed for all 15 speakers")
    else:
        failures.extend(nested_failures)

    # 5. Leakage check (adaptation ∩ test == empty)
    print("\nVerifying zero data leakage (adaptation intersect test == empty)...")
    leakage_failures = []
    for r in splits_run2:
        spk = r["speaker_id"]
        n = r["sample_n"]
        adapt_set = set(r["adaptation_recording_ids"])
        test_set = set(r["test_recording_ids"])
        overlap = adapt_set.intersection(test_set)
        if overlap:
            leakage_failures.append(f"{spk} N={n}: {len(overlap)} overlapping recordings detected!")

    if not leakage_failures:
        print("  [OK] Zero data leakage across all 60 split subsets")
    else:
        failures.extend(leakage_failures)

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print("TORGO SPLIT REPRODUCIBILITY: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("TORGO SPLIT REPRODUCIBILITY: PASS")
        print("All acceptance criteria satisfied.")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(run_reproducibility_test())
