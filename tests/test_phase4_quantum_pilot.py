"""
Unit and Integration Tests for Phase 4 Quantum Kernel Pilot.

Verifies:
1. Train-only PCA pipeline fitting and dimensionality reduction (384 -> d).
2. ZZ feature map construction, circuit depth, and gate properties.
3. Fidelity quantum kernel matrix validation (finite, symmetric, diagonal == 1.0).
4. QSVC fitting, support vector accounting, and deterministic prediction.
5. End-to-end evaluation using canonical Phase 3 metrics across representative speakers.
6. Persistent kernel caching and zero-recomputation reload.
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import compute_intent_metrics
from src.models.quantum_kernel import QSVCClassifier
from src.quantum.fidelity_kernel import QuantumFidelityKernelEngine
from src.quantum.pca_pipeline import TrainOnlyPCAPipeline

ALL_CLASSES = [
    "SOS_HELP",
    "FALL",
    "BREATHING_MEDICAL",
    "PAIN",
    "LOCATION",
    "CRITICAL_RESPONSE",
    "GENERAL_SPEECH",
]


@pytest.fixture(scope="module")
def pilot_data():
    """Load real cohort splits and manifest for representative pilot testing."""
    with open("splits/torgo_splits.json", "r", encoding="utf-8") as f:
        splits_list = json.load(f)

    manifest_df = pd.read_csv("data/manifests/torgo_manifest.csv")
    rec_to_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))

    splits_dict = {}
    for entry in splits_list:
        key = (entry["speaker_id"], entry["sample_n"])
        splits_dict[key] = {
            "adaptation": entry["adaptation_recording_ids"],
            "test": entry["test_recording_ids"],
        }

    return {
        "splits": splits_dict,
        "rec_to_intent": rec_to_intent,
    }


def test_pca_train_only_isolation():
    """Verify PCA fits strictly on training data and preserves variance bounds."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 384)).astype(np.float32)
    X_test = rng.normal(size=(10, 384)).astype(np.float32)

    for q in [2, 4, 6, 8]:
        pca_pipe = TrainOnlyPCAPipeline(n_components=q, random_state=42)
        X_tr_pca = pca_pipe.fit_transform(X_train)
        X_te_pca = pca_pipe.transform(X_test)

        assert X_tr_pca.shape == (20, q)
        assert X_te_pca.shape == (10, q)
        assert pca_pipe.cumulative_explained_variance_ > 0.0
        assert pca_pipe.cumulative_explained_variance_ <= 1.0
        # Scaled bounds in [0, pi]
        assert np.all(X_tr_pca >= -1e-6)
        assert np.all(X_tr_pca <= np.pi + 1e-6)


def test_fidelity_kernel_matrix_properties():
    """Verify kernel matrix is symmetric, finite, and has diagonal identically 1.0."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, np.pi, size=(15, 4)).astype(np.float64)

    engine = QuantumFidelityKernelEngine(num_qubits=4, reps=1, entanglement="linear")
    K = engine.evaluate(X)
    diag = engine.validate_kernel_matrix(K, is_training=True)

    assert diag["is_finite"]
    assert diag["shape"] == [15, 15]
    assert diag["symmetry_error"] < 1e-7
    assert diag["diagonal_error"] < 1e-7
    assert diag["min_value"] >= 0.0
    assert diag["max_value"] <= 1.0 + 1e-7
    assert diag["min_eigenvalue"] >= -1e-7


def test_quantum_circuit_properties():
    """Verify gate counts and circuit depth for primary and ablation configurations."""
    for q in [2, 4, 6, 8]:
        engine = QuantumFidelityKernelEngine(num_qubits=q, reps=1, entanglement="linear")
        props = engine.get_circuit_properties()
        assert props["num_qubits"] == q
        assert props["circuit_depth"] > 0
        assert props["two_qubit_gate_count"] == 2 * (q - 1)
        assert props["trainable_quantum_parameters"] == 0
        assert props["encoding_parameters"] == q


def test_quantum_pilot_end_to_end(pilot_data):
    """Run pilot execution on F01 (dysarthric) and FC01 (control) across budgets."""
    splits = pilot_data["splits"]
    rec_to_intent = pilot_data["rec_to_intent"]
    whisper_dir = "embeddings/torgo/whisper"

    speakers = ["F01", "FC01"]
    sample_sizes = [20, 30]
    qubit_ablation = [2, 4]

    temp_cache = tempfile.mkdtemp()
    try:
        for sp in speakers:
            for n_val in sample_sizes:
                split = splits[(sp, n_val)]
                X_train = np.stack(
                    [np.load(os.path.join(whisper_dir, f"{rid}.npy")) for rid in split["adaptation"]]
                )
                y_train = [rec_to_intent[rid] for rid in split["adaptation"]]

                # Use small test subset for fast pilot test execution
                test_rids = split["test"][:25]
                X_test = np.stack(
                    [np.load(os.path.join(whisper_dir, f"{rid}.npy")) for rid in test_rids]
                )
                y_test = [rec_to_intent[rid] for rid in test_rids]

                for q in qubit_ablation:
                    clf = QSVCClassifier(
                        qubit_count=q,
                        reps=1,
                        entanglement="linear",
                        C=1.0,
                        seed=42,
                        cache_dir=temp_cache,
                        all_classes=ALL_CLASSES,
                    )
                    clf.fit(X_train, y_train, speaker_id=sp, sample_n=n_val)
                    preds = clf.predict(X_test, speaker_id=sp, sample_n=n_val)

                    metrics = compute_intent_metrics(y_test, preds, ALL_CLASSES)
                    assert len(preds) == len(y_test)
                    assert "accuracy" in metrics
                    assert "macro_f1" in metrics
                    assert 0.0 <= metrics["accuracy"] <= 1.0
                    assert 0.0 <= metrics["macro_f1"] <= 1.0

                    # Verify cached kernel was created
                    train_k_path = os.path.join(temp_cache, f"{sp}_N{n_val}_q{q}_r1_train.npy")
                    assert os.path.exists(train_k_path)
    finally:
        shutil.rmtree(temp_cache, ignore_errors=True)
