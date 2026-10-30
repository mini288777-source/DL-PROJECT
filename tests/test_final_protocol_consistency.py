"""
Test Final Protocol Consistency.
Verifies separation of Track 1 and Track 2, Phase 5 AWGN noise protocol, and few-shot splits.
"""

import os
import json
import yaml
import pytest


def test_track1_vs_track2_separation():
    with open("config/research_protocol.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    t1 = cfg["original_protocol"]
    t2 = cfg["recovery_track"]
    
    assert t1["formulation"] == "flat_7_class"
    assert t1["class_weighting"] is False
    
    assert t2["formulation"]["architecture"] == "two_stage_hierarchical"
    assert t2["class_weighting"] is True
    assert t2["split_protocol"]["per_speaker_emergency_minimum"] == 2


def test_phase5_noise_protocol():
    with open("config/research_protocol.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    noise_cfg = cfg["noise_perturbation_protocol"]
    assert noise_cfg["snr_levels_db"] == ["clean", 15, 10, 5]
    assert "AWGN" in noise_cfg["perturbation_type"].upper()
    assert noise_cfg["sample_rate_hz"] == 16000


def test_track_b_splits_integrity():
    split_path = "phase6_5/splits/recovery_splits_hierarchical_ready.json"
    assert os.path.exists(split_path), f"Missing {split_path}"
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    # Check that budgets are 20, 30, 40, 50
    assert len(splits) == 60, f"Expected 60 splits, got {len(splits)}"
    for s_data in splits:
        n_val = s_data.get("N", s_data.get("sample_n"))
        assert n_val in [20, 30, 40, 50]
        adapt_ids = s_data.get("adaptation_recordings", s_data.get("adaptation_recording_ids"))
        assert len(adapt_ids) == n_val
        test_ids = s_data.get("test_recordings", s_data.get("test_recording_ids"))
        assert len(test_ids) > 0
