"""
Tests for Phase 7.13 - Continual Safety & Robustness Audit
Verifies environmental stress testing, failure mode injections,
safety veto protection, and absence of codebase backdoor bypasses.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.continual.taxonomy import GENERAL_SPEECH_CLASS
from src.continual.governance import ModelGovernanceManager, PromotionDecision
from src.continual.self_improvement import (
    ContinualSelfImprovementEngine,
    UnverifiedLearningAttemptError
)
from src.continual.safety_audit import (
    SafetyRobustnessAuditor,
    StressCondition,
    FailureInjectionType
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE7_13_RESULTS = REPO_ROOT / "results" / "continual" / "phase7_13"
PHASE7_13_REPORTS = REPO_ROOT / "reports" / "continual" / "phase7_13"


def test_phase7_13_artifacts_exist():
    """Verify all Phase 7.13 artifacts exist and are non-empty."""
    required = [
        PHASE7_13_RESULTS / "stress_test_results.csv",
        PHASE7_13_RESULTS / "failure_injection_results.csv",
        PHASE7_13_RESULTS / "codebase_bypass_audit.json",
        PHASE7_13_REPORTS / "safety_robustness_summary.json"
    ]
    for p in required:
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"


def test_awgn_noise_stress_evaluation():
    """Verify auditor correctly evaluates noise perturbations and high abstention under noise."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    rng = np.random.RandomState(42)
    feats = rng.randn(20, 384).astype(np.float32)
    labels = np.zeros(20, dtype=int)
    labels[:4] = 1

    class MockClassifier:
        def predict_proba(self, X):
            # Low confidence predictions under noise
            probs = np.ones((len(X), 7)) / 7.0
            return probs

    res = auditor.audit_awgn_noise_robustness(
        classifier=MockClassifier(),
        features=feats,
        labels=labels,
        snr_levels=[20.0, 10.0, 0.0],
        speaker_id="TEST_SPK"
    )
    assert len(res) == 3
    for r in res:
        assert r.abstention_rate == 1.0  # Low confidence correctly triggered abstention
        assert r.status == "PASS"


def test_scarcity_stress_hold_decision():
    """Verify that severe experience scarcity triggers HOLD or REJECT, never PROMOTE."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        res = auditor.audit_experience_scarcity(engine, "F01", sample_counts=[1, 2])
        for r in res:
            assert r.governance_decision in ["HOLD", "REJECT"]
            assert r.status == "PASS"


def test_unverified_learning_attempt_blocked():
    """Verify deliberate injection of unverified experience is safely blocked."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        inj = auditor.inject_unverified_learning_attempt(engine, "F01")
        assert inj.attempt_blocked is True
        assert inj.status == "PASS"
        assert inj.active_model_safe is True


def test_speaker_isolation_integrity():
    """Verify cross-speaker contamination is prevented."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        inj = auditor.inject_speaker_mismatch(engine, "F01", "M01")
        assert inj.attempt_blocked is True
        assert inj.status == "PASS"


def test_temporal_safety_future_data_blocked():
    """Verify future timestamp injection is detected."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        inj = auditor.inject_future_timestamp_leakage(engine, "F01")
        assert inj.attempt_blocked is True
        assert inj.status == "PASS"


def test_corrupted_features_quarantined():
    """Verify NaN/Inf features are quarantined without poisoning models."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        inj = auditor.inject_corrupted_features(engine, "F01")
        assert inj.attempt_blocked is True
        assert inj.status == "PASS"


def test_direct_activation_bypass_blocked():
    """Verify arbitrary direct model pointer activation is blocked."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        inj = auditor.inject_direct_activation_bypass(engine.governance_manager, "F01")
        assert inj.attempt_blocked is True
        assert inj.status == "PASS"


def test_unsafe_candidate_safety_veto():
    """Verify candidate model with emergency recall drop triggers Safety Veto (REJECT)."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = ContinualSelfImprovementEngine(
            speaker_id="F01",
            base_dir=tmp_dir,
            model_family="classical",
            random_seed=42
        )
        inj = auditor.inject_unsafe_candidate(engine.governance_manager, "F01")
        assert inj.attempt_blocked is True
        assert inj.status == "PASS"


def test_codebase_bypass_audit_clean():
    """Verify that static AST scan over src/continual/ finds 0 unauthorized bypasses."""
    auditor = SafetyRobustnessAuditor(repo_root=REPO_ROOT)
    rep = auditor.audit_codebase_bypasses()
    assert rep.clean_audit_verdict is True
    assert rep.unverified_training_paths_found == 0
    assert rep.direct_activation_paths_found == 0
    assert rep.rollback_bypasses_found == 0
    assert rep.speaker_leakage_paths_found == 0
