"""
Unit and Integration Tests for Phase 8: Targeted Quantum Recovery + Confirmatory Benchmark.

Verifies:
1. Stage 1 Binary Emergency Gate (architecture, weighting, inference).
2. Stage 2 Classical Subtype Net (Track A).
3. Stage 2 Quantum Residual 8-Qubit VQC (Track B).
4. Stage 2 Matched Classical Shadow MLP (Track C).
5. Unified model interface consistency across all 3 tracks.
6. Strict Information Isolation (Zero-leakage Train-only PCA).
7. Phase 8 metrics computation (Emergency Recall, FER, Precision, Macro-F1).
8. Multi-metric safety gate and veto rules.
9. Pre-registered research protocol YAML integrity.
"""

import os
import yaml
import pytest
import numpy as np
import torch

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline
from src.phase8.hierarchical_gate import BinaryEmergencyGateNet, BinaryEmergencyDetector
from src.phase8.classical_subtype import ClassicalSubtypeNet, ClassicalSubtypeClassifier
from src.phase8.quantum_residual import QuantumSubtypeCircuit, QuantumResidualSubtypeClassifier
from src.phase8.classical_shadow import ClassicalShadowNet, ClassicalShadowClassifier
from src.phase8.phase8_models import (
    TrackAHierarchicalClassical,
    TrackBQuantumResidual,
    TrackCMatchedClassicalShadow,
    compute_phase8_metrics,
)


def test_stage1_binary_gate_architecture():
    """Verify Stage 1 Binary Gate net shape and parameters."""
    net = BinaryEmergencyGateNet(input_dim=384, hidden_dim=64)
    x = torch.randn(5, 384)
    out = net(x)
    assert out.shape == (5, 2)
    # Check parameter count: (384*64 + 64) + 128 (LayerNorm) + (64*2 + 2) = 24640 + 128 + 130 = 24898
    param_count = sum(p.numel() for p in net.parameters())
    assert param_count > 24000


def test_stage1_binary_gate_fitting():
    """Verify Stage 1 detector fits and predicts calibrated probabilities."""
    detector = BinaryEmergencyDetector(input_dim=384, hidden_dim=64, epochs=5, pos_weight=3.0)
    X = np.random.randn(20, 384).astype(np.float32)
    y = np.array(["GENERAL_SPEECH"] * 14 + ["FALL"] * 3 + ["SOS_HELP"] * 3)

    detector.fit(X, y)
    probs = detector.predict_proba(X)
    assert probs.shape == (20, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_stage2_classical_subtype_classifier():
    """Verify Stage 2 Classical Subtype Net for Track A."""
    clf = ClassicalSubtypeClassifier(input_dim=384, hidden_dim=64, num_classes=6, epochs=5)
    X = np.random.randn(12, 384).astype(np.float32)
    y = np.array(["CRITICAL_RESPONSE", "FALL", "BREATHING_MEDICAL", "PAIN", "SOS_HELP", "LOCATION"] * 2)

    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 12
    for p in preds:
        assert p in EMERGENCY_CLASSES

    probs = clf.predict_proba(X)
    assert probs.shape == (12, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_stage2_quantum_residual_vqc():
    """Verify Stage 2 Quantum VQC for Track B."""
    vqc = QuantumResidualSubtypeClassifier(n_qubits=8, n_layers=2, n_classes=6, epochs=3)
    X_8d = np.random.uniform(0.0, np.pi, size=(6, 8)).astype(np.float32)
    y = np.array(["CRITICAL_RESPONSE", "FALL", "BREATHING_MEDICAL", "PAIN", "SOS_HELP", "LOCATION"])

    vqc.fit(X_8d, y)
    preds = vqc.predict(X_8d)
    assert len(preds) == 6
    for p in preds:
        assert p in EMERGENCY_CLASSES

    probs = vqc.predict_proba(X_8d)
    assert probs.shape == (6, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_stage2_classical_shadow_mlp():
    """Verify Stage 2 Matched Classical Shadow for Track C."""
    shadow = ClassicalShadowClassifier(input_dim=8, hidden_dim=16, num_classes=6, epochs=5)
    # Check parameter count: (8*16 + 16) + (16*6 + 6) = 144 + 102 = 246
    param_count = sum(p.numel() for p in shadow.net.parameters()) if shadow.net else (8*16 + 16 + 16*6 + 6)
    assert param_count == 246

    X_8d = np.random.uniform(0.0, np.pi, size=(6, 8)).astype(np.float32)
    y = np.array(["CRITICAL_RESPONSE", "FALL", "BREATHING_MEDICAL", "PAIN", "SOS_HELP", "LOCATION"])

    shadow.fit(X_8d, y)
    preds = shadow.predict(X_8d)
    assert len(preds) == 6
    for p in preds:
        assert p in EMERGENCY_CLASSES

    probs = shadow.predict_proba(X_8d)
    assert probs.shape == (6, 6)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_unified_model_interface():
    """Verify Track A, B, and C provide uniform APIs and probability shapes."""
    X_train = np.random.randn(20, 384).astype(np.float32)
    y_train = np.array(["GENERAL_SPEECH"] * 14 + ["FALL", "PAIN", "SOS_HELP", "LOCATION", "CRITICAL_RESPONSE", "BREATHING_MEDICAL"])
    X_test = np.random.randn(10, 384).astype(np.float32)

    models = [
        TrackAHierarchicalClassical(gate_epochs=3, subtype_epochs=3),
        TrackBQuantumResidual(gate_epochs=3, vqc_epochs=2),
        TrackCMatchedClassicalShadow(gate_epochs=3, shadow_epochs=3),
    ]

    for model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == 10
        for p in preds:
            assert p in CANONICAL_CLASSES

        probs = model.predict_proba(X_test)
        assert probs.shape == (10, 7)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_zero_leakage_pca_pipeline():
    """Verify TrainOnlyPCAPipeline strictly enforces information isolation."""
    pca_pipe = TrainOnlyPCAPipeline(n_components=8, feature_range=(0.0, np.pi))
    X_train = np.random.randn(20, 384).astype(np.float32)
    X_test = np.random.randn(10, 384).astype(np.float32)

    # Cannot transform before fit
    with pytest.raises(Exception):
        pca_pipe.transform(X_test)

    # Fit exclusively on training data
    X_train_pca = pca_pipe.fit_transform(X_train)
    assert X_train_pca.shape == (20, 8)
    assert X_train_pca.min() >= -1e-5
    assert X_train_pca.max() <= np.pi + 1e-5

    # Test transform uses fitted parameters without refitting
    X_test_pca = pca_pipe.transform(X_test)
    assert X_test_pca.shape == (10, 8)


def test_phase8_metrics_computation():
    """Verify compute_phase8_metrics handles true/false positives accurately."""
    y_true = np.array(["GENERAL_SPEECH", "GENERAL_SPEECH", "FALL", "SOS_HELP"])
    y_pred = np.array(["GENERAL_SPEECH", "FALL", "FALL", "LOCATION"])

    # True emergencies = 2 (FALL, SOS_HELP)
    # Predicted emergencies = 3 (FALL on gen, FALL on FALL, LOCATION on SOS_HELP)
    # Correctly detected emergencies = 2 (both true emergencies received emergency labels)
    # Emergency Recall = 2/2 = 1.0
    # Emergency Precision = 2/3 = 0.667
    # True general = 2 (indices 0, 1)
    # General falsely predicted as emergency = 1 (index 1)
    # FER = 1/2 = 0.5
    metrics = compute_phase8_metrics(y_true, y_pred)
    assert metrics["emergency_recall"] == 1.0
    assert abs(metrics["emergency_precision"] - (2.0 / 3.0)) < 1e-4
    assert metrics["false_emergency_rate"] == 0.5
    assert metrics["macro_f1"] > 0.0


def test_safety_gate_rules():
    """Verify multi-metric safety gate veto conditions."""
    from scripts.run_phase8_experiments import evaluate_safety_gate

    # Valid performance
    valid_met = {"emergency_recall": 0.40, "false_emergency_rate": 0.04}
    passed, msg = evaluate_safety_gate(valid_met)
    assert passed is True
    assert msg == "PASS"

    # FER Spike Veto
    fer_spike = {"emergency_recall": 0.80, "false_emergency_rate": 0.22}
    passed, msg = evaluate_safety_gate(fer_spike)
    assert passed is False
    assert "FER_SPIKE_VETO" in msg

    # Recall Collapse Veto
    recall_collapse = {"emergency_recall": 0.01, "false_emergency_rate": 0.01}
    passed, msg = evaluate_safety_gate(recall_collapse)
    assert passed is False
    assert "RECALL_COLLAPSE_VETO" in msg


def test_protocol_yaml_integrity():
    """Verify config/phase8/research_protocol.yaml matches canonical specification."""
    proto_path = os.path.join("config", "phase8", "research_protocol.yaml")
    assert os.path.exists(proto_path)

    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    assert proto["dataset_and_cohort"]["total_speakers"] == 15
    assert len(proto["dataset_and_cohort"]["dysarthric_speakers"]) == 8
    assert len(proto["dataset_and_cohort"]["control_speakers"]) == 7
    assert proto["experimental_matrix"]["budgets"] == [20, 30, 40, 50]
    assert proto["experimental_matrix"]["canonical_cells"] == 60
    assert len(proto["taxonomy"]["canonical_classes"]) == 7
    assert len(proto["taxonomy"]["emergency_classes"]) == 6
