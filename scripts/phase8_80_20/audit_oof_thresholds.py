"""
Exploratory script to audit and compute training-only Out-Of-Fold (OOF) thresholds.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.special import expit

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.phase8_separated.emergency_gate import ScalarEmergencyGate
from src.embeddings.cache import load_embedding_file
from src.continual.taxonomy import EMERGENCY_CLASSES

def main():
    manifest_path = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_csv = os.path.join(REPO_ROOT, "results", "phase8_80_20", "split_manifest.csv")
    df_manifest = pd.read_csv(manifest_csv)
    label_map = dict(zip(df_manifest["sample_id"], df_manifest["intent"]))

    base_torgo_dir = os.path.join(REPO_ROOT, "embeddings", "torgo")
    needed_rids = set(manifest["all_train_ids"])
    embedding_cache = {}
    for rid in sorted(list(needed_rids)):
        embedding_cache[rid] = load_embedding_file(rid, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=384)

    threshold_grid = np.arange(0.10, 0.95, 0.05)

    print("Checking Out-Of-Fold Threshold Calibration on 80% Training Data:")
    for spk_info in manifest["speakers"]:
        spk = spk_info["speaker_id"]
        train_ids = spk_info["train_ids"]
        X_train = np.array([embedding_cache[r] for r in train_ids], dtype=np.float32)
        y_train = np.array([label_map[r] for r in train_ids])
        binary_y = np.array([1 if label in EMERGENCY_CLASSES else 0 for label in y_train])

        n_pos = int(np.sum(binary_y == 1))
        n_neg = int(np.sum(binary_y == 0))

        # Stratified K-Fold (min 2, max 5 depending on pos count)
        n_splits = min(5, max(2, n_pos)) if n_pos >= 2 else 2
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_probs = np.zeros(len(X_train), dtype=np.float32)
        for train_idx, val_idx in skf.split(X_train, binary_y):
            gate = ScalarEmergencyGate(random_seed=42)
            gate.fit(X_train[train_idx], y_train[train_idx])
            z_val = gate.predict_logits(X_train[val_idx])
            oof_probs[val_idx] = expit(z_val)

        best_tau = 0.5
        best_recall = -1.0
        best_fer_min = 1.0
        is_safety_feasible = False

        for tau in threshold_grid:
            pred_em = oof_probs >= tau
            rec = np.sum((binary_y == 1) & pred_em) / max(1, n_pos)
            fer = np.sum((binary_y == 0) & pred_em) / max(1, n_neg)

            if fer <= 0.15:
                if rec > best_recall:
                    best_recall = rec
                    best_tau = float(tau)
                    is_safety_feasible = True
            else:
                if fer < best_fer_min:
                    best_fer_min = fer

        if not is_safety_feasible:
            # minimize FER
            min_fer = 999.0
            for tau in threshold_grid:
                fer = np.sum((binary_y == 0) & (oof_probs >= tau)) / max(1, n_neg)
                if fer < min_fer:
                    min_fer = fer
                    best_tau = float(tau)

        print(f"Spk: {spk:4s} | Train: {len(X_train):3d} (Em: {n_pos:2d}) | OOF Splits: {n_splits} | Best Tau: {best_tau:.2f} | OOF Rec: {best_recall:.3f} | Feasible: {is_safety_feasible}")

if __name__ == "__main__":
    main()
