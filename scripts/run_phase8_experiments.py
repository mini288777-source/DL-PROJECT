"""
Phase 8 Recovery & Confirmatory Benchmark Runner.

Executes the 60 canonical cells (15 TORGO speakers x 4 budgets N in {20, 30, 40, 50})
across the 3-track controlled recovery ladder:
- Track A: Hierarchical Emergency-Aware Classical Baseline
- Track B: Quantum Residual Emergency Subtype Module
- Track C: Matched Classical Shadow Model

Strict scientific invariants:
- Pre-registered locked protocol (config/phase8/research_protocol.yaml).
- Zero test data leakage (PCA and scalers fitted strictly on adaptation samples).
- Matched inputs, objectives, and capacity between Track B and Track C.
- Honest, unmanipulated reporting of all safety and efficacy metrics.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.phase8.phase8_models import (
    TrackAHierarchicalClassical,
    TrackBQuantumResidual,
    TrackCMatchedClassicalShadow,
    compute_phase8_metrics,
)
from src.embeddings.cache import load_embedding_file
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS


def evaluate_safety_gate(metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    Evaluates Phase 7.6 / Phase 8 multi-metric safety gate.
    Passes only if FER <= 0.15 and Emergency Recall >= 0.05.
    """
    fer = metrics["false_emergency_rate"]
    rec = metrics["emergency_recall"]

    if fer > 0.15:
        return False, f"FER_SPIKE_VETO (FER={fer:.3f} > 0.15)"
    if rec < 0.05:
        return False, f"RECALL_COLLAPSE_VETO (Recall={rec:.3f} < 0.05)"
    return True, "PASS"


def run_phase8_confirmatory_benchmark():
    print("=" * 80, flush=True)
    print("PHASE 8 — TARGETED QUANTUM RECOVERY + CONFIRMATORY VALIDATION", flush=True)
    print("Locked Confirmatory Benchmark across 15 Speakers x 4 Budgets = 60 Cells", flush=True)
    print("=" * 80, flush=True)

    # 1. Output setup
    results_dir = os.path.join(REPO_ROOT, "results", "phase8")
    figures_dir = os.path.join(results_dir, "figures")
    reports_dir = os.path.join(REPO_ROOT, "reports", "phase8")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 2. Load splits and manifests
    splits_path = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    adapt_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_adaptation_manifest.csv"))
    test_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_test_manifest.csv"))

    label_map = dict(zip(adapt_manifest["recording_id"], adapt_manifest["intent"]))
    label_map.update(dict(zip(test_manifest["recording_id"], test_manifest["intent"])))

    # 3. Preload all needed Whisper embeddings
    print("Preloading Whisper embeddings into memory...", flush=True)
    needed_rids = set()
    for s in splits:
        needed_rids.update(s["adaptation_recording_ids"])
        needed_rids.update(s["test_recording_ids"])

    base_torgo_dir = os.path.join(REPO_ROOT, "embeddings", "torgo")
    embedding_cache: Dict[str, np.ndarray] = {}
    for rid in sorted(list(needed_rids)):
        emb = load_embedding_file(rid, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=384)
        embedding_cache[rid] = emb
    print(f"Preloaded {len(embedding_cache)} embeddings.\n", flush=True)

    speakers = sorted(list(set(s["speaker_id"] for s in splits)))
    budgets = [20, 30, 40, 50]
    dysarthric_set = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}

    all_cell_results: List[Dict[str, Any]] = []
    total_cells = len(speakers) * len(budgets)
    cell_counter = 0
    t_start_all = time.perf_counter()

    for spk in speakers:
        spk_splits = {s["sample_n"]: s for s in splits if s["speaker_id"] == spk}
        spk_group = "dysarthric" if spk in dysarthric_set else "control"

        # Fixed locked protected test set for this speaker
        test_rids = spk_splits[50]["test_recording_ids"]
        X_test = np.array([embedding_cache[r] for r in test_rids], dtype=np.float32)
        y_test = np.array([label_map[r] for r in test_rids])

        for budget in budgets:
            cell_counter += 1
            cell_id = f"PH8_{spk}_N{budget}"
            b_rids = spk_splits[budget]["adaptation_recording_ids"]
            X_train = np.array([embedding_cache[r] for r in b_rids], dtype=np.float32)
            y_train = np.array([label_map[r] for r in b_rids])

            n_train_em = int(sum(1 for y in y_train if y in EMERGENCY_CLASSES))
            n_test_em = int(sum(1 for y in y_test if y in EMERGENCY_CLASSES))

            print(f"[{cell_counter:02d}/{total_cells}] Speaker {spk} ({spk_group}), N={budget}: "
                  f"Train={len(X_train)} (Em={n_train_em}), Test={len(X_test)} (Em={n_test_em})", flush=True)

            # -------------------------------------------------------------
            # Track A: Hierarchical Emergency-Aware Classical Baseline
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            model_a = TrackAHierarchicalClassical(pos_weight=3.0, gate_epochs=40, subtype_epochs=40)
            model_a.fit(X_train, y_train)
            t_fit_a = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_a = model_a.predict(X_test)
            t_eval_a = time.perf_counter() - t0
            lat_a = (t_eval_a / len(X_test)) * 1000.0

            met_a = compute_phase8_metrics(y_test, pred_a)
            gate_a_pass, gate_a_msg = evaluate_safety_gate(met_a)

            all_cell_results.append({
                "cell_id": cell_id,
                "speaker_id": spk,
                "speaker_group": spk_group,
                "budget": budget,
                "track": "Track_A_Classical",
                "emergency_recall": met_a["emergency_recall"],
                "emergency_precision": met_a["emergency_precision"],
                "false_emergency_rate": met_a["false_emergency_rate"],
                "macro_f1": met_a["macro_f1"],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_emergencies": n_train_em,
                "test_emergencies": n_test_em,
                "pred_emergencies": met_a["pred_emergencies"],
                "train_time_sec": t_fit_a,
                "eval_time_sec": t_eval_a,
                "latency_ms_per_sample": lat_a,
                "safety_gate_passed": gate_a_pass,
                "safety_gate_status": gate_a_msg,
            })

            # -------------------------------------------------------------
            # Track B: Quantum Residual Emergency Subtype Module
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            model_b = TrackBQuantumResidual(pos_weight=3.0, gate_epochs=40, vqc_epochs=30)
            model_b.fit(X_train, y_train)
            t_fit_b = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_b = model_b.predict(X_test)
            t_eval_b = time.perf_counter() - t0
            lat_b = (t_eval_b / len(X_test)) * 1000.0

            met_b = compute_phase8_metrics(y_test, pred_b)
            gate_b_pass, gate_b_msg = evaluate_safety_gate(met_b)

            all_cell_results.append({
                "cell_id": cell_id,
                "speaker_id": spk,
                "speaker_group": spk_group,
                "budget": budget,
                "track": "Track_B_Quantum",
                "emergency_recall": met_b["emergency_recall"],
                "emergency_precision": met_b["emergency_precision"],
                "false_emergency_rate": met_b["false_emergency_rate"],
                "macro_f1": met_b["macro_f1"],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_emergencies": n_train_em,
                "test_emergencies": n_test_em,
                "pred_emergencies": met_b["pred_emergencies"],
                "train_time_sec": t_fit_b,
                "eval_time_sec": t_eval_b,
                "latency_ms_per_sample": lat_b,
                "safety_gate_passed": gate_b_pass,
                "safety_gate_status": gate_b_msg,
            })

            # -------------------------------------------------------------
            # Track C: Matched Classical Shadow Model
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            model_c = TrackCMatchedClassicalShadow(pos_weight=3.0, gate_epochs=40, shadow_epochs=30)
            model_c.fit(X_train, y_train)
            t_fit_c = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_c = model_c.predict(X_test)
            t_eval_c = time.perf_counter() - t0
            lat_c = (t_eval_c / len(X_test)) * 1000.0

            met_c = compute_phase8_metrics(y_test, pred_c)
            gate_c_pass, gate_c_msg = evaluate_safety_gate(met_c)

            all_cell_results.append({
                "cell_id": cell_id,
                "speaker_id": spk,
                "speaker_group": spk_group,
                "budget": budget,
                "track": "Track_C_Shadow",
                "emergency_recall": met_c["emergency_recall"],
                "emergency_precision": met_c["emergency_precision"],
                "false_emergency_rate": met_c["false_emergency_rate"],
                "macro_f1": met_c["macro_f1"],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_emergencies": n_train_em,
                "test_emergencies": n_test_em,
                "pred_emergencies": met_c["pred_emergencies"],
                "train_time_sec": t_fit_c,
                "eval_time_sec": t_eval_c,
                "latency_ms_per_sample": lat_c,
                "safety_gate_passed": gate_c_pass,
                "safety_gate_status": gate_c_msg,
            })

            print(f"   Track A: Rec={met_a['emergency_recall']:.3f}, FER={met_a['false_emergency_rate']:.3f}, F1={met_a['macro_f1']:.3f} [{gate_a_msg}]", flush=True)
            print(f"   Track B: Rec={met_b['emergency_recall']:.3f}, FER={met_b['false_emergency_rate']:.3f}, F1={met_b['macro_f1']:.3f} [{gate_b_msg}]", flush=True)
            print(f"   Track C: Rec={met_c['emergency_recall']:.3f}, FER={met_c['false_emergency_rate']:.3f}, F1={met_c['macro_f1']:.3f} [{gate_c_msg}]", flush=True)

    elapsed_all = time.perf_counter() - t_start_all
    print(f"\nAll 60 cells completed across 3 tracks in {elapsed_all:.1f}s ({elapsed_all/60.0:.1f} min).", flush=True)

    # 4. Save raw results to CSV
    df = pd.DataFrame(all_cell_results)
    csv_path = os.path.join(results_dir, "phase8_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}", flush=True)

    # 5. Compute summary statistics
    summary: Dict[str, Any] = {
        "benchmark_metadata": {
            "total_cells": total_cells,
            "models_trained": len(df),
            "speakers_evaluated": len(speakers),
            "dysarthric_speakers": len([s for s in speakers if s in dysarthric_set]),
            "control_speakers": len([s for s in speakers if s not in dysarthric_set]),
            "budgets": budgets,
            "total_runtime_seconds": elapsed_all,
        },
        "aggregate_metrics": {},
        "dysarthric_subgroup": {},
        "control_subgroup": {},
        "safety_gate_summary": {},
    }

    for track in ["Track_A_Classical", "Track_B_Quantum", "Track_C_Shadow"]:
        track_df = df[df["track"] == track]
        summary["aggregate_metrics"][track] = {
            "mean_emergency_recall": float(track_df["emergency_recall"].mean()),
            "std_emergency_recall": float(track_df["emergency_recall"].std()),
            "mean_emergency_precision": float(track_df["emergency_precision"].mean()),
            "mean_false_emergency_rate": float(track_df["false_emergency_rate"].mean()),
            "std_false_emergency_rate": float(track_df["false_emergency_rate"].std()),
            "mean_macro_f1": float(track_df["macro_f1"].mean()),
            "std_macro_f1": float(track_df["macro_f1"].std()),
            "mean_train_time_sec": float(track_df["train_time_sec"].mean()),
            "mean_eval_time_sec": float(track_df["eval_time_sec"].mean()),
            "mean_latency_ms": float(track_df["latency_ms_per_sample"].mean()),
        }

        # Dysarthric vs Control splits
        dys_df = track_df[track_df["speaker_group"] == "dysarthric"]
        ctrl_df = track_df[track_df["speaker_group"] == "control"]

        summary["dysarthric_subgroup"][track] = {
            "mean_emergency_recall": float(dys_df["emergency_recall"].mean()),
            "mean_false_emergency_rate": float(dys_df["false_emergency_rate"].mean()),
            "mean_macro_f1": float(dys_df["macro_f1"].mean()),
        }
        summary["control_subgroup"][track] = {
            "mean_emergency_recall": float(ctrl_df["emergency_recall"].mean()),
            "mean_false_emergency_rate": float(ctrl_df["false_emergency_rate"].mean()),
            "mean_macro_f1": float(ctrl_df["macro_f1"].mean()),
        }

        summary["safety_gate_summary"][track] = {
            "pass_count": int(track_df["safety_gate_passed"].sum()),
            "fail_count": int((~track_df["safety_gate_passed"]).sum()),
            "pass_rate": float(track_df["safety_gate_passed"].mean()),
        }

    summary_path = os.path.join(results_dir, "phase8_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)

    return df, summary


if __name__ == "__main__":
    from typing import Tuple
    run_phase8_confirmatory_benchmark()
