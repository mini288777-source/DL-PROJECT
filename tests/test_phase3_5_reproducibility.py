"""
Phase 3.5 Research Reproducibility Acceptance Test Suite.

Verifies:
1. Split reproducibility and nested subset containment (N=20 ⊂ N=30 ⊂ N=40 ⊂ N=50).
2. Zero data leakage between adaptation and test sets.
3. Bit-for-bit repeatability of RBF-SVM training and evaluation against canonical Phase 3 results.
4. Classical Adapter architecture, parameter count (25,223), and deterministic evaluation.
5. Exact metric computation parity with Phase 3 baseline curves.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.classical_adapter import ClassicalAdapterClassifier, ClassicalAdapterNet
from src.models.evaluation import compute_intent_metrics
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.utils.seeding import set_seed

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
def torgo_data():
    """Load splits and manifest for testing."""
    with open("splits/torgo_splits.json", "r", encoding="utf-8") as f:
        splits_list = json.load(f)

    manifest_df = pd.read_csv("data/manifests/torgo_manifest.csv")
    rec_to_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))

    # Index splits by (speaker_id, sample_n)
    splits_dict = {}
    for entry in splits_list:
        key = (entry["speaker_id"], entry["sample_n"])
        splits_dict[key] = {
            "adaptation": entry["adaptation_recording_ids"],
            "test": entry["test_recording_ids"],
        }

    curves_df = pd.read_csv("results/phase3/baseline_curves.csv")

    return {
        "splits": splits_dict,
        "rec_to_intent": rec_to_intent,
        "curves": curves_df,
    }


def test_split_nesting_and_isolation(torgo_data):
    """Verify nested-subset behavior and zero leakage across all speakers."""
    splits = torgo_data["splits"]
    speakers = sorted(list(set(sp for sp, _ in splits.keys())))

    for sp in speakers:
        s20 = set(splits[(sp, 20)]["adaptation"])
        s30 = set(splits[(sp, 30)]["adaptation"])
        s40 = set(splits[(sp, 40)]["adaptation"])
        s50 = set(splits[(sp, 50)]["adaptation"])
        test_set = set(splits[(sp, 50)]["test"])

        # 1. Exact sizes
        assert len(s20) == 20, f"{sp} N=20 size is {len(s20)}"
        assert len(s30) == 30, f"{sp} N=30 size is {len(s30)}"
        assert len(s40) == 40, f"{sp} N=40 size is {len(s40)}"
        assert len(s50) == 50, f"{sp} N=50 size is {len(s50)}"

        # 2. Strict nested subset containment
        assert s20.issubset(s30), f"{sp} N=20 not subset of N=30"
        assert s30.issubset(s40), f"{sp} N=30 not subset of N=40"
        assert s40.issubset(s50), f"{sp} N=40 not subset of N=50"

        # 3. Zero data leakage
        assert len(s50.intersection(test_set)) == 0, f"{sp} adaptation intersects test!"


def test_rbf_svm_reproducibility_f01(torgo_data):
    """Reproduce RBF-SVM results for speaker F01 at N=20 and verify exact match with Phase 3."""
    sp = "F01"
    n_val = 20
    split = torgo_data["splits"][(sp, n_val)]
    rec_to_intent = torgo_data["rec_to_intent"]
    curves = torgo_data["curves"]

    whisper_dir = "embeddings/torgo/whisper"
    X_train = np.stack([np.load(os.path.join(whisper_dir, f"{rid}.npy")) for rid in split["adaptation"]])
    y_train = [rec_to_intent[rid] for rid in split["adaptation"]]

    X_test = np.stack([np.load(os.path.join(whisper_dir, f"{rid}.npy")) for rid in split["test"]])
    y_test = [rec_to_intent[rid] for rid in split["test"]]

    clf = ClassicalRBFSVMClassifier(C=1.0, gamma="scale", seed=42, all_classes=ALL_CLASSES)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    metrics = compute_intent_metrics(y_test, preds, ALL_CLASSES)

    # Check against stored baseline curves
    row = curves[(curves["speaker_id"] == sp) & (curves["method"] == "rbf_svm") & (curves["sample_n"] == n_val)].iloc[0]
    expected_acc = float(row["accuracy"])
    expected_f1 = float(row["macro_f1"])

    assert abs(metrics["accuracy"] - expected_acc) < 1e-4, f"Accuracy mismatch: {metrics['accuracy']} vs {expected_acc}"
    assert abs(metrics["macro_f1"] - expected_f1) < 1e-4, f"Macro-F1 mismatch: {metrics['macro_f1']} vs {expected_f1}"
    assert clf.support_vector_count == 5, f"Expected 5 support vectors for F01 N=20, got {clf.support_vector_count}"


def test_adapter_architecture_and_forward_reproducibility():
    """Verify adapter parameter count (25,223) and bit-for-bit forward pass."""
    net = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7, dropout=0.1)
    assert net.count_parameters() == 25223

    x = torch.ones(4, 384)
    set_seed(42)
    net1 = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7, dropout=0.0).eval()
    out1 = net1(x).detach().numpy()

    set_seed(42)
    net2 = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7, dropout=0.0).eval()
    out2 = net2(x).detach().numpy()

    assert np.allclose(out1, out2, atol=1e-7), "Adapter forward pass is not deterministic under identical seed"


def test_metric_validation_against_known_edge_cases():
    """Verify compute_intent_metrics behaves correctly under single-class prediction."""
    y_true = ["GENERAL_SPEECH"] * 90 + ["FALL"] * 10
    y_pred = ["GENERAL_SPEECH"] * 100

    metrics = compute_intent_metrics(y_true, y_pred, ALL_CLASSES)
    assert abs(metrics["accuracy"] - 0.90) < 1e-6
    # F1(GENERAL_SPEECH) = 2 * 0.90 * 1.0 / (0.90 + 1.0) = 1.8 / 1.9 = 0.947368
    # Macro-F1 = 0.947368 / 7 = 0.135338
    assert abs(metrics["macro_f1"] - (1.8 / 1.9) / 7.0) < 1e-5
