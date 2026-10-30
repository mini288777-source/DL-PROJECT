"""
Phase 8 Repaired: Confirmatory Benchmark Runner.

Executes the 60 canonical cells (15 TORGO speakers x 4 budgets N in {20, 30, 40, 50})
across the 3-track repaired residual ladder:
- Track A: Repaired Classical Recovery (384-D Residual)
- Track B: Repaired Quantum Residual (8-Qubit VQC Residual)
- Track C: Repaired Matched Classical Shadow (8-D MLP Residual)

Strict scientific guarantees:
- Residual logit z_residual directly participates in final emergency decision:
  z_final = z_gate + alpha * z_residual
- Metric provenance: Emergency Recall, Precision, and FER are computed from the
  final track-specific emergency prediction.
- Zero test leakage: Threshold tau is locked strictly on training data internal validation.
- Zero leakage PCA: 8-D bottleneck is fitted strictly on adaptation training split.
- Independent outputs preserved under results/phase8_repaired/
"""

import sys
import os
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.phase8_repaired.phase8_repaired_models import (
    RepairedTrackAClassical,
    RepairedTrackBQuantum,
    RepairedTrackCShadow,
    compute_repaired_phase8_metrics,
)
from src.embeddings.cache import load_embedding_file
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS


def evaluate_repaired_safety_gate(metrics: Dict[str, Any]) -> Tuple[bool, str]:
    fer = metrics["false_emergency_rate"]
    rec = metrics["emergency_recall"]

    if fer > 0.15:
        return False, f"FER_SPIKE_VETO (FER={fer:.3f} > 0.15)"
    if rec < 0.05:
        return False, f"RECALL_COLLAPSE_VETO (Recall={rec:.3f} < 0.05)"
    return True, "PASS"


def run_phase8_repaired_benchmark():
    print("=" * 80, flush=True)
    print("PHASE 8 REPAIRED — CORRECTED QUANTUM RECOVERY CONFIRMATORY BENCHMARK", flush=True)
    print("Executing 15 Speakers x 4 Budgets = 60 Cells (180 Track Runs)", flush=True)
    print("=" * 80, flush=True)

    results_dir = os.path.join(REPO_ROOT, "results", "phase8_repaired")
    figures_dir = os.path.join(results_dir, "figures")
    reports_dir = os.path.join(REPO_ROOT, "reports", "phase8_repaired")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    splits_path = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    adapt_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_adaptation_manifest.csv"))
    test_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_test_manifest.csv"))

    label_map = dict(zip(adapt_manifest["recording_id"], adapt_manifest["intent"]))
    label_map.update(dict(zip(test_manifest["recording_id"], test_manifest["intent"])))

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

        test_rids = spk_splits[50]["test_recording_ids"]
        X_test = np.array([embedding_cache[r] for r in test_rids], dtype=np.float32)
        y_test = np.array([label_map[r] for r in test_rids])

        for budget in budgets:
            cell_counter += 1
            cell_id = f"PH8_REP_{spk}_N{budget}"
            b_rids = spk_splits[budget]["adaptation_recording_ids"]
            X_train = np.array([embedding_cache[r] for r in b_rids], dtype=np.float32)
            y_train = np.array([label_map[r] for r in b_rids])

            n_train_em = int(sum(1 for y in y_train if y in EMERGENCY_CLASSES))
            n_test_em = int(sum(1 for y in y_test if y in EMERGENCY_CLASSES))

            print(f"[{cell_counter:02d}/{total_cells}] Speaker {spk} ({spk_group}), N={budget}: "
                  f"Train={len(X_train)} (Em={n_train_em}), Test={len(X_test)} (Em={n_test_em})", flush=True)

            # -------------------------------------------------------------
            # Track A: Repaired Classical Recovery Baseline
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            model_a = RepairedTrackAClassical(alpha=0.5, random_seed=42)
            model_a.fit(X_train, y_train)
            t_fit_a = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_7cl_a, is_em_a = model_a.predict(X_test)
            t_eval_a = time.perf_counter() - t0
            lat_a = (t_eval_a / len(X_test)) * 1000.0

            met_a = compute_repaired_phase8_metrics(y_test, pred_7cl_a, is_em_a, track_id="Track_A_Classical")
            gate_a_pass, gate_a_msg = evaluate_repaired_safety_gate(met_a)

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
                "locked_threshold": model_a.locked_threshold,
                "is_safety_feasible_on_train": model_a.is_safety_feasible,
                "quantum_weight_delta": 0.0,
                "quantum_influence_norm": 0.0,
                "train_time_sec": t_fit_a,
                "eval_time_sec": t_eval_a,
                "latency_ms_per_sample": lat_a,
                "safety_gate_passed": gate_a_pass,
                "safety_gate_status": gate_a_msg,
            })

            # -------------------------------------------------------------
            # Track B: Repaired Quantum Residual Module
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            model_b = RepairedTrackBQuantum(alpha=0.5, random_seed=42)
            model_b.fit(X_train, y_train)
            t_fit_b = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_7cl_b, is_em_b = model_b.predict(X_test)
            t_eval_b = time.perf_counter() - t0
            lat_b = (t_eval_b / len(X_test)) * 1000.0

            # Measure quantum participation on test set
            z_with_b = model_b.predict_emergency_logits(X_test)
            z_without_b = model_b.predict_without_residual(X_test)
            q_influence = float(np.mean(np.abs(z_with_b - z_without_b)))

            met_b = compute_repaired_phase8_metrics(y_test, pred_7cl_b, is_em_b, track_id="Track_B_Quantum")
            gate_b_pass, gate_b_msg = evaluate_repaired_safety_gate(met_b)

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
                "locked_threshold": model_b.locked_threshold,
                "is_safety_feasible_on_train": model_b.is_safety_feasible,
                "quantum_weight_delta": model_b.residual.weight_delta_norm,
                "quantum_influence_norm": q_influence,
                "train_time_sec": t_fit_b,
                "eval_time_sec": t_eval_b,
                "latency_ms_per_sample": lat_b,
                "safety_gate_passed": gate_b_pass,
                "safety_gate_status": gate_b_msg,
            })

            # -------------------------------------------------------------
            # Track C: Repaired Matched Classical Shadow Model
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            model_c = RepairedTrackCShadow(alpha=0.5, random_seed=42)
            model_c.fit(X_train, y_train)
            t_fit_c = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_7cl_c, is_em_c = model_c.predict(X_test)
            t_eval_c = time.perf_counter() - t0
            lat_c = (t_eval_c / len(X_test)) * 1000.0

            met_c = compute_repaired_phase8_metrics(y_test, pred_7cl_c, is_em_c, track_id="Track_C_Shadow")
            gate_c_pass, gate_c_msg = evaluate_repaired_safety_gate(met_c)

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
                "locked_threshold": model_c.locked_threshold,
                "is_safety_feasible_on_train": model_c.is_safety_feasible,
                "quantum_weight_delta": 0.0,
                "quantum_influence_norm": 0.0,
                "train_time_sec": t_fit_c,
                "eval_time_sec": t_eval_c,
                "latency_ms_per_sample": lat_c,
                "safety_gate_passed": gate_c_pass,
                "safety_gate_status": gate_c_msg,
            })

            print(f"   Track A: Rec={met_a['emergency_recall']:.3f}, FER={met_a['false_emergency_rate']:.3f}, F1={met_a['macro_f1']:.3f} [{gate_a_msg}]", flush=True)
            print(f"   Track B: Rec={met_b['emergency_recall']:.3f}, FER={met_b['false_emergency_rate']:.3f}, F1={met_b['macro_f1']:.3f} (q_infl={q_influence:.3f}) [{gate_b_msg}]", flush=True)
            print(f"   Track C: Rec={met_c['emergency_recall']:.3f}, FER={met_c['false_emergency_rate']:.3f}, F1={met_c['macro_f1']:.3f} [{gate_c_msg}]", flush=True)

    elapsed_all = time.perf_counter() - t_start_all
    print(f"\nAll 60 repaired cells completed across 3 tracks in {elapsed_all:.1f}s ({elapsed_all/60.0:.1f} min).", flush=True)

    # Save raw CSV
    df = pd.DataFrame(all_cell_results)
    csv_path = os.path.join(results_dir, "phase8_repaired_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}", flush=True)

    # Save Required Tables
    # 1. corrected_model_comparison.csv
    comp_rows = []
    tracks = ["Track_A_Classical", "Track_B_Quantum", "Track_C_Shadow"]
    track_names = ["Classical Recovery (A)", "Quantum Residual (B)", "Matched Shadow (C)"]
    param_counts = [24962 + 24640 + 25222, 24962 + 16 + 25222, 24962 + 161 + 25222]

    for t, name, params in zip(tracks, track_names, param_counts):
        sub = df[df["track"] == t]
        dys = sub[sub["speaker_group"] == "dysarthric"]
        ctrl = sub[sub["speaker_group"] == "control"]
        comp_rows.append({
            "Track": name,
            "Emergency Recall": f"{sub['emergency_recall'].mean()*100:.2f}%",
            "Emergency Precision": f"{sub['emergency_precision'].mean()*100:.2f}%",
            "FER": f"{sub['false_emergency_rate'].mean()*100:.2f}%",
            "Macro-F1": f"{sub['macro_f1'].mean()*100:.2f}%",
            "Dysarthric Recall": f"{dys['emergency_recall'].mean()*100:.2f}%",
            "Control Recall": f"{ctrl['emergency_recall'].mean()*100:.2f}%",
            "Mean Threshold": f"{sub['locked_threshold'].mean():.2f}",
            "Alpha": 0.5,
            "Train Time": f"{sub['train_time_sec'].mean():.2f}s",
            "Inference Time": f"{sub['latency_ms_per_sample'].mean():.3f}ms",
            "Parameter Count": params,
            "Safety Status": "PASS" if sub["safety_gate_passed"].mean() > 0.5 else "FAIL (FER Spike)",
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(os.path.join(results_dir, "corrected_model_comparison.csv"), index=False)

    # 2. corrected_per_speaker_results.csv
    spk_df = df.groupby(["speaker_id", "speaker_group", "track"]).agg({
        "emergency_recall": "mean",
        "emergency_precision": "mean",
        "false_emergency_rate": "mean",
        "macro_f1": "mean",
        "quantum_influence_norm": "mean",
    }).reset_index()
    spk_df.to_csv(os.path.join(results_dir, "corrected_per_speaker_results.csv"), index=False)

    # 3. corrected_per_budget_results.csv
    bgt_df = df.groupby(["budget", "track"]).agg({
        "emergency_recall": "mean",
        "emergency_precision": "mean",
        "false_emergency_rate": "mean",
        "macro_f1": "mean",
        "quantum_influence_norm": "mean",
    }).reset_index()
    bgt_df.to_csv(os.path.join(results_dir, "corrected_per_budget_results.csv"), index=False)

    # Save summary JSON
    summary: Dict[str, Any] = {
        "benchmark_metadata": {
            "total_cells": total_cells,
            "models_trained": len(df),
            "speakers_evaluated": len(speakers),
            "budgets": budgets,
            "total_runtime_seconds": elapsed_all,
            "alpha": 0.5,
        },
        "aggregate_metrics": {},
        "dysarthric_subgroup": {},
        "control_subgroup": {},
        "safety_gate_summary": {},
    }

    for t in tracks:
        track_df = df[df["track"] == t]
        summary["aggregate_metrics"][t] = {
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
            "mean_quantum_influence": float(track_df["quantum_influence_norm"].mean()),
        }

        dys_df = track_df[track_df["speaker_group"] == "dysarthric"]
        ctrl_df = track_df[track_df["speaker_group"] == "control"]
        summary["dysarthric_subgroup"][t] = {
            "mean_emergency_recall": float(dys_df["emergency_recall"].mean()),
            "mean_false_emergency_rate": float(dys_df["false_emergency_rate"].mean()),
            "mean_macro_f1": float(dys_df["macro_f1"].mean()),
        }
        summary["control_subgroup"][t] = {
            "mean_emergency_recall": float(ctrl_df["emergency_recall"].mean()),
            "mean_false_emergency_rate": float(ctrl_df["false_emergency_rate"].mean()),
            "mean_macro_f1": float(ctrl_df["macro_f1"].mean()),
        }
        summary["safety_gate_summary"][t] = {
            "pass_count": int(track_df["safety_gate_passed"].sum()),
            "fail_count": int((~track_df["safety_gate_passed"]).sum()),
            "pass_rate": float(track_df["safety_gate_passed"].mean()),
        }

    with open(os.path.join(results_dir, "phase8_repaired_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Phase 8 Repaired benchmark artifacts successfully written.", flush=True)
    return df, summary


if __name__ == "__main__":
    run_phase8_repaired_benchmark()
