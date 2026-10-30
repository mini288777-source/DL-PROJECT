"""
Subphase 6.5.2 — Classical Rescue Baseline Experiment Runner

Executes controlled clean-speech evaluation across 4 model conditions on Track A recovery splits:
  1. unweighted_adapter (Control)
  2. weighted_adapter (Loss-level balanced class weighting)
  3. unweighted_rbf_svm (Control)
  4. weighted_rbf_svm (Cost-sensitive balanced class weighting)

Outputs:
  phase6_5/classical_rescue/results/classical_rescue_results.csv (240 cells)
  phase6_5/classical_rescue/results/classical_rescue_predictions.csv
  phase6_5/classical_rescue/results/class_weight_records.csv
  phase6_5/classical_rescue/results/class_support_records.csv
  phase6_5/classical_rescue/results/errors.csv

  phase6_5/classical_rescue/analysis/split_recovery_effect.csv
  phase6_5/classical_rescue/analysis/class_weight_effect.csv
  phase6_5/classical_rescue/analysis/emergency_metrics.csv
  phase6_5/classical_rescue/analysis/sample_efficiency.csv
  phase6_5/classical_rescue/analysis/speaker_analysis.csv
  phase6_5/classical_rescue/analysis/collapse_analysis.csv

  phase6_5/classical_rescue/figures/macro_f1_recovery_comparison.png
  phase6_5/classical_rescue/figures/emergency_recall_comparison.png
  phase6_5/classical_rescue/figures/iper_comparison.png
  phase6_5/classical_rescue/figures/sample_efficiency_comparison.png
  phase6_5/classical_rescue/figures/majority_collapse_comparison.png
"""

import os
import sys
import time
import json
import yaml
import hashlib
import argparse
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.classical_adapter import ClassicalAdapterClassifier
from src.models.rbf_svm import ClassicalRBFSVMClassifier
from src.embeddings.cache import load_embedding_file
from src.utils.seeding import set_seed

CANONICAL_CLASSES = [
    "GENERAL_SPEECH", "SOS_HELP", "FALL", "BREATHING_MEDICAL",
    "PAIN", "LOCATION", "CRITICAL_RESPONSE"
]
EMERGENCY_CLASSES = [c for c in CANONICAL_CLASSES if c != "GENERAL_SPEECH"]
DYSARTHRIC_SET = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}


def compute_iper(y_true: List[str], y_pred: List[str]) -> Tuple[float, int, int]:
    """Canonical IPER calculation from Phase 5."""
    emergency_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != "GENERAL_SPEECH"]
    total_emergency = len(emergency_pairs)
    if total_emergency == 0:
        return 0.0, 0, 0
    preserved_emergency = sum(1 for yt, yp in emergency_pairs if yt == yp)
    return float(preserved_emergency / total_emergency), preserved_emergency, total_emergency


def compute_comprehensive_metrics(y_true: List[str], y_pred: List[str], feasible_classes: List[str]) -> Dict[str, Any]:
    """Calculate all primary, secondary, and diagnostic metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average="weighted", zero_division=0))
    
    # Feasible-class Macro-F1 (diagnostic)
    if feasible_classes:
        feasible_macro_f1 = float(f1_score(y_true, y_pred, labels=feasible_classes, average="macro", zero_division=0))
    else:
        feasible_macro_f1 = 0.0

    # Emergency Macro-F1
    emerg_macro_f1 = float(f1_score(y_true, y_pred, labels=EMERGENCY_CLASSES, average="macro", zero_division=0))

    # Emergency Precision and Recall
    true_emerg_count = sum(1 for yt in y_true if yt in EMERGENCY_CLASSES)
    pred_emerg_count = sum(1 for yp in y_pred if yp in EMERGENCY_CLASSES)
    correct_emerg_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt in EMERGENCY_CLASSES and yp in EMERGENCY_CLASSES and yt == yp)

    emerg_recall = float(correct_emerg_count / true_emerg_count) if true_emerg_count > 0 else 0.0
    emerg_precision = float(correct_emerg_count / pred_emerg_count) if pred_emerg_count > 0 else 0.0

    # IPER
    iper_rate, _, _ = compute_iper(y_true, y_pred)

    # False Emergency Rate on General Speech
    gen_total = sum(1 for yt in y_true if yt == "GENERAL_SPEECH")
    gen_false_emerg = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "GENERAL_SPEECH" and yp in EMERGENCY_CLASSES)
    false_emerg_rate = float(gen_false_emerg / gen_total) if gen_total > 0 else 0.0

    # General speech recall
    gen_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "GENERAL_SPEECH" and yp == "GENERAL_SPEECH")
    gen_recall = float(gen_correct / gen_total) if gen_total > 0 else 0.0

    # Prediction distribution
    total_test = len(y_pred)
    pred_gen_count = sum(1 for yp in y_pred if yp == "GENERAL_SPEECH")
    pred_gen_frac = float(pred_gen_count / total_test) if total_test > 0 else 0.0
    pred_emerg_frac = float(pred_emerg_count / total_test) if total_test > 0 else 0.0
    unique_preds = len(set(y_pred))
    majority_collapse = bool(pred_gen_frac == 1.0)

    # Per-class breakdowns
    p_arr = precision_score(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
    r_arr = recall_score(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
    f_arr = f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)

    per_class = {}
    for i, c in enumerate(CANONICAL_CLASSES):
        per_class[c] = {
            "precision": float(p_arr[i]),
            "recall": float(r_arr[i]),
            "f1": float(f_arr[i]),
            "support": int(sum(1 for yt in y_true if yt == c))
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "feasible_macro_f1": feasible_macro_f1,
        "emergency_macro_f1": emerg_macro_f1,
        "weighted_f1": weighted_f1,
        "emergency_precision": emerg_precision,
        "emergency_recall": emerg_recall,
        "iper": iper_rate,
        "false_emergency_rate": false_emerg_rate,
        "general_speech_recall": gen_recall,
        "predicted_general_fraction": pred_gen_frac,
        "predicted_emergency_fraction": pred_emerg_frac,
        "num_unique_predicted_classes": unique_preds,
        "majority_collapse": majority_collapse,
        "per_class": per_class
    }


def run_experiment(pilot_mode: bool = False):
    config_path = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "config", "classical_rescue_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits_path = os.path.join(REPO_ROOT, cfg["data"]["splits_track_a_path"])
    manifest_path = os.path.join(REPO_ROOT, cfg["data"]["manifest_path"])
    whisper_dir = os.path.join(REPO_ROOT, cfg["data"]["whisper_cache_dir"])

    res_dir = os.path.join(REPO_ROOT, cfg["output"]["results_dir"])
    ana_dir = os.path.join(REPO_ROOT, cfg["output"]["analysis_dir"])
    fig_dir = os.path.join(REPO_ROOT, cfg["output"]["figures_dir"])
    rep_dir = os.path.join(REPO_ROOT, cfg["output"]["reports_dir"])

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    # Load splits JSON
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_list = json.load(f)

    # Load manifest
    manifest = pd.read_csv(manifest_path)
    label_map = dict(zip(manifest["recording_id"], manifest["intent"]))

    # Derive prompt group lookup
    rec_to_pg = {}
    manifest_active = manifest[manifest["split"].isin(["adaptation", "test"])].copy()
    manifest_active["norm_transcript"] = manifest_active["transcript"].astype(str).str.strip().str.upper()
    for _, row in manifest_active.iterrows():
        rid = row["recording_id"]
        t = row["norm_transcript"]
        pg_id = "PGRP_" + hashlib.sha256(t.encode("utf-8")).hexdigest()[:10]
        rec_to_pg[rid] = pg_id

    # Filter conditions if pilot
    if pilot_mode:
        target_speakers = cfg["pilot"]["speakers"]
        target_ns = cfg["pilot"]["sample_sizes"]
        print(f"\n>>> EXECUTING PILOT RUN (Speakers: {target_speakers}, Ns: {target_ns}) <<<")
    else:
        target_speakers = cfg["cohort"]["speakers"]
        target_ns = cfg["cohort"]["sample_sizes"]
        print(f"\n>>> EXECUTING FULL EXPERIMENT (15 Speakers, Ns: [20, 30, 40, 50]) <<<")

    models_cfg = cfg["models"]
    total_conditions = len(target_speakers) * len(target_ns) * len(models_cfg)
    print(f"Total cells to execute: {total_conditions}")

    # Preload embeddings for efficiency
    print("Preloading required Whisper embeddings...")
    needed_rids = set()
    for s in splits_list:
        if s["speaker_id"] in target_speakers and s["sample_n"] in target_ns:
            needed_rids.update(s["adaptation_recording_ids"])
            needed_rids.update(s["test_recording_ids"])

    embedding_cache = {}
    base_torgo_dir = os.path.dirname(whisper_dir)
    for rid in sorted(list(needed_rids)):
        emb = load_embedding_file(rid, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=384)
        embedding_cache[rid] = emb
    print(f"Loaded {len(embedding_cache)} embeddings into RAM.\n")

    results_rows = []
    predictions_rows = []
    weight_records = []
    support_records = []
    error_rows = []

    cell_idx = 0
    t0_all = time.perf_counter()

    for s_entry in splits_list:
        spk = s_entry["speaker_id"]
        n = s_entry["sample_n"]
        if spk not in target_speakers or n not in target_ns:
            continue

        grp = "dysarthric" if spk in DYSARTHRIC_SET else "control"
        adapt_rids = s_entry["adaptation_recording_ids"]
        test_rids = s_entry["test_recording_ids"]
        adapt_hash = s_entry["adaptation_split_hash"]
        test_hash = s_entry["test_split_hash"]

        X_train = np.stack([embedding_cache[r] for r in adapt_rids])
        y_train = [label_map[r] for r in adapt_rids]
        X_test = np.stack([embedding_cache[r] for r in test_rids])
        y_test = [label_map[r] for r in test_rids]

        # Feasible classes in adaptation
        train_counts = pd.Series(y_train).value_counts().to_dict()
        feasible_classes = sorted([c for c in CANONICAL_CLASSES if train_counts.get(c, 0) > 0])
        missing_classes = sorted([c for c in CANONICAL_CLASSES if train_counts.get(c, 0) == 0])
        gen_count = train_counts.get("GENERAL_SPEECH", 0)
        emerg_count = sum(train_counts.get(c, 0) for c in EMERGENCY_CLASSES)

        for m_spec in models_cfg:
            cell_idx += 1
            m_name = m_spec["name"]
            m_family = m_spec["family"]
            cw_param = m_spec["class_weight"]
            b_method = "class_weighted" if cw_param == "balanced" else "unweighted"
            exp_id = f"{spk}_{m_name}_N{n}"

            print(f"[{cell_idx:3d}/{total_conditions:3d}] {exp_id:30s} ... ", end="", flush=True)
            t_train_start = time.perf_counter()

            try:
                set_seed(cfg["cohort"]["seed"])
                preds = []
                t_infer = 0.0

                if m_family == "adapter":
                    clf = ClassicalAdapterClassifier(
                        input_dim=m_spec["input_dim"],
                        hidden_dim=m_spec["hidden_dim"],
                        num_classes=m_spec["num_classes"],
                        dropout=m_spec["dropout"],
                        learning_rate=m_spec["learning_rate"],
                        weight_decay=m_spec["weight_decay"],
                        epochs=m_spec["epochs"],
                        batch_size=m_spec["batch_size"],
                        class_weight=cw_param,
                        seed=cfg["cohort"]["seed"],
                        all_classes=CANONICAL_CLASSES
                    )
                    clf.fit(X_train, y_train)
                    t_train = time.perf_counter() - t_train_start

                    t_infer_start = time.perf_counter()
                    preds = clf.predict(X_test)
                    t_infer = time.perf_counter() - t_infer_start

                elif m_family == "rbf_svm":
                    clf = ClassicalRBFSVMClassifier(
                        C=m_spec["C"],
                        gamma=m_spec["gamma"],
                        class_weight=cw_param,
                        standardize=m_spec["standardize"],
                        probability=m_spec["probability"],
                        seed=cfg["cohort"]["seed"],
                        all_classes=CANONICAL_CLASSES
                    )
                    clf.fit(X_train, y_train)
                    t_train = time.perf_counter() - t_train_start

                    t_infer_start = time.perf_counter()
                    preds = clf.predict(X_test)
                    t_infer = time.perf_counter() - t_infer_start

                # Metrics
                m_res = compute_comprehensive_metrics(y_test, preds, feasible_classes)
                print(f"Macro-F1={m_res['macro_f1']:.4f}  EmergRec={m_res['emergency_recall']:.4f}  IPER={m_res['iper']:.4f}  Collapse={m_res['majority_collapse']}")

                # Record result row
                results_rows.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "cohort_group": grp,
                    "N": n,
                    "model": m_name,
                    "balancing_method": b_method,
                    "split_protocol": "recovery_stratified_track_a",
                    "adaptation_split_hash": adapt_hash,
                    "test_split_hash": test_hash,
                    "adaptation_total": len(adapt_rids),
                    "general_speech_count": gen_count,
                    "emergency_count": emerg_count,
                    "feasible_class_count": len(feasible_classes),
                    "missing_class_count": len(missing_classes),
                    "min_class_support": min(train_counts.values()) if train_counts else 0,
                    "max_class_support": max(train_counts.values()) if train_counts else 0,
                    "macro_f1": round(m_res["macro_f1"], 4),
                    "feasible_macro_f1": round(m_res["feasible_macro_f1"], 4),
                    "emergency_macro_f1": round(m_res["emergency_macro_f1"], 4),
                    "accuracy": round(m_res["accuracy"], 4),
                    "weighted_f1": round(m_res["weighted_f1"], 4),
                    "emergency_precision": round(m_res["emergency_precision"], 4),
                    "emergency_recall": round(m_res["emergency_recall"], 4),
                    "IPER": round(m_res["iper"], 4),
                    "false_emergency_rate": round(m_res["false_emergency_rate"], 4),
                    "predicted_general_fraction": round(m_res["predicted_general_fraction"], 4),
                    "predicted_emergency_fraction": round(m_res["predicted_emergency_fraction"], 4),
                    "num_unique_predicted_classes": m_res["num_unique_predicted_classes"],
                    "majority_collapse": m_res["majority_collapse"],
                    "training_time": round(t_train, 4),
                    "inference_time": round(t_infer, 4),
                    "seed": cfg["cohort"]["seed"],
                    "configuration_hash": hashlib.sha256(f"{m_name}_{b_method}_{n}".encode("utf-8")).hexdigest()[:12],
                    "status": "SUCCESS"
                })

                # Record prediction traces
                for rid, yt, yp in zip(test_rids, y_test, preds):
                    is_corr = (yt == yp)
                    is_em_true = (yt in EMERGENCY_CLASSES)
                    is_em_pred = (yp in EMERGENCY_CLASSES)
                    is_em_corr = (is_em_true and is_em_pred and is_corr)

                    if not is_em_true:
                        iper_stat = "general_speech"
                    elif is_corr:
                        iper_stat = "preserved"
                    else:
                        iper_stat = "corrupted"

                    predictions_rows.append({
                        "experiment_id": exp_id,
                        "speaker_id": spk,
                        "cohort_group": grp,
                        "N": n,
                        "model": m_name,
                        "balancing_method": b_method,
                        "recording_id": rid,
                        "prompt_group_id": rec_to_pg.get(rid, "UNKNOWN"),
                        "true_intent": yt,
                        "predicted_intent": yp,
                        "is_correct": is_corr,
                        "is_emergency_true": is_em_true,
                        "is_emergency_pred": is_em_pred,
                        "is_emergency_correct": is_em_corr,
                        "iper_status": iper_stat
                    })

                # Record class support & per-class metrics
                for c in CANONICAL_CLASSES:
                    c_metrics = m_res["per_class"][c]
                    support_records.append({
                        "experiment_id": exp_id,
                        "speaker_id": spk,
                        "cohort_group": grp,
                        "N": n,
                        "model": m_name,
                        "intent_class": c,
                        "adapt_support": train_counts.get(c, 0),
                        "test_support": c_metrics["support"],
                        "per_class_precision": round(c_metrics["precision"], 4),
                        "per_class_recall": round(c_metrics["recall"], 4),
                        "per_class_f1": round(c_metrics["f1"], 4)
                    })

                # Record class weights
                n_samples = len(y_train)
                n_obs = len(feasible_classes)
                for c in CANONICAL_CLASSES:
                    c_count = train_counts.get(c, 0)
                    is_feas = (c_count > 0)
                    if b_method == "class_weighted" and is_feas:
                        weight_val = n_samples / (n_obs * c_count)
                    elif b_method == "class_weighted" and not is_feas:
                        weight_val = 0.0
                    else:
                        weight_val = 1.0

                    weight_records.append({
                        "experiment_id": exp_id,
                        "speaker_id": spk,
                        "N": n,
                        "model": m_name,
                        "balancing_method": b_method,
                        "intent_class": c,
                        "adaptation_sample_count": c_count,
                        "is_feasible": is_feas,
                        "computed_weight": round(float(weight_val), 4)
                    })

            except Exception as e:
                print(f"FAILED: {e}")
                error_rows.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "N": n,
                    "model": m_name,
                    "error_message": str(e)
                })

    t_elapsed = time.perf_counter() - t0_all
    print(f"\nExecution finished in {t_elapsed:.2f} seconds.")

    # Save Results
    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(os.path.join(res_dir, "classical_rescue_results.csv"), index=False)
    print(f"Saved {os.path.join(res_dir, 'classical_rescue_results.csv')} ({len(df_results)} rows)")

    df_preds = pd.DataFrame(predictions_rows)
    df_preds.to_csv(os.path.join(res_dir, "classical_rescue_predictions.csv"), index=False)
    print(f"Saved {os.path.join(res_dir, 'classical_rescue_predictions.csv')} ({len(df_preds)} rows)")

    df_weights = pd.DataFrame(weight_records)
    df_weights.to_csv(os.path.join(res_dir, "class_weight_records.csv"), index=False)
    print(f"Saved {os.path.join(res_dir, 'class_weight_records.csv')} ({len(df_weights)} rows)")

    df_supp = pd.DataFrame(support_records)
    df_supp.to_csv(os.path.join(res_dir, "class_support_records.csv"), index=False)
    print(f"Saved {os.path.join(res_dir, 'class_support_records.csv')} ({len(df_supp)} rows)")

    df_err = pd.DataFrame(error_rows)
    df_err.to_csv(os.path.join(res_dir, "errors.csv"), index=False)
    print(f"Saved {os.path.join(res_dir, 'errors.csv')} ({len(df_err)} rows)")

    if not pilot_mode:
        # Perform analysis and generate figures
        build_analysis_and_figures(df_results, df_preds, ana_dir, fig_dir)


def build_analysis_and_figures(df_res: pd.DataFrame, df_preds: pd.DataFrame, ana_dir: str, fig_dir: str):
    """Generate statistical analysis tables and publication-ready figures."""
    print("\nComputing statistical analysis tables...")

    # Load frozen Phase 5 baseline for clean audio comparison
    p5_path = os.path.join(REPO_ROOT, "results", "phase5", "phase5_results.csv")
    if os.path.exists(p5_path):
        p5_df = pd.read_csv(p5_path)
        p5_clean = p5_df[p5_df["snr_tag"] == "clean"]
    else:
        p5_clean = None

    # 1. Split Recovery Effect (Original Baseline vs Recovery Unweighted)
    split_rec_rows = []
    for model_short, orig_model_name in [("adapter", "adapter"), ("rbf_svm", "rbf_svm")]:
        rec_unweighted_name = f"unweighted_{model_short}"
        for n in [20, 30, 40, 50]:
            rec_sub = df_res[(df_res["model"] == rec_unweighted_name) & (df_res["N"] == n)]
            if p5_clean is not None:
                p5_sub = p5_clean[(p5_clean["model"] == orig_model_name) & (p5_clean["sample_n"] == n)]
                orig_f1 = float(p5_sub["macro_f1"].mean())
                orig_iper = float(p5_sub["iper"].mean())
            else:
                orig_f1, orig_iper = 0.139, 0.0

            rec_f1 = float(rec_sub["macro_f1"].mean())
            rec_iper = float(rec_sub["IPER"].mean())
            rec_rec = float(rec_sub["emergency_recall"].mean())

            split_rec_rows.append({
                "model": model_short,
                "N": n,
                "original_macro_f1": round(orig_f1, 4),
                "recovery_unweighted_macro_f1": round(rec_f1, 4),
                "delta_macro_f1": round(rec_f1 - orig_f1, 4),
                "original_iper": round(orig_iper, 4),
                "recovery_unweighted_iper": round(rec_iper, 4),
                "delta_iper": round(rec_iper - orig_iper, 4),
                "recovery_unweighted_emergency_recall": round(rec_rec, 4)
            })

    pd.DataFrame(split_rec_rows).to_csv(os.path.join(ana_dir, "split_recovery_effect.csv"), index=False)
    print(f"Saved {os.path.join(ana_dir, 'split_recovery_effect.csv')}")

    # 2. Class Weight Effect (Recovery Unweighted vs Recovery Weighted)
    cw_rows = []
    for model_short in ["adapter", "rbf_svm"]:
        unw_name = f"unweighted_{model_short}"
        w_name = f"weighted_{model_short}"
        for n in [20, 30, 40, 50]:
            unw_sub = df_res[(df_res["model"] == unw_name) & (df_res["N"] == n)]
            w_sub = df_res[(df_res["model"] == w_name) & (df_res["N"] == n)]

            unw_f1 = float(unw_sub["macro_f1"].mean())
            w_f1 = float(w_sub["macro_f1"].mean())
            unw_rec = float(unw_sub["emergency_recall"].mean())
            w_rec = float(w_sub["emergency_recall"].mean())
            unw_prec = float(unw_sub["emergency_precision"].mean())
            w_prec = float(w_sub["emergency_precision"].mean())
            unw_iper = float(unw_sub["IPER"].mean())
            w_iper = float(w_sub["IPER"].mean())

            cw_rows.append({
                "model": model_short,
                "N": n,
                "unweighted_macro_f1": round(unw_f1, 4),
                "weighted_macro_f1": round(w_f1, 4),
                "delta_macro_f1": round(w_f1 - unw_f1, 4),
                "unweighted_emergency_recall": round(unw_rec, 4),
                "weighted_emergency_recall": round(w_rec, 4),
                "delta_emergency_recall": round(w_rec - unw_rec, 4),
                "unweighted_emergency_precision": round(unw_prec, 4),
                "weighted_emergency_precision": round(w_prec, 4),
                "delta_emergency_precision": round(w_prec - unw_prec, 4),
                "unweighted_iper": round(unw_iper, 4),
                "weighted_iper": round(w_iper, 4),
                "delta_iper": round(w_iper - unw_iper, 4)
            })

    pd.DataFrame(cw_rows).to_csv(os.path.join(ana_dir, "class_weight_effect.csv"), index=False)
    print(f"Saved {os.path.join(ana_dir, 'class_weight_effect.csv')}")

    # 3. Emergency Metrics Table
    emerg_table = df_res.groupby(["model", "N"])[[
        "macro_f1", "emergency_macro_f1", "emergency_recall", "emergency_precision", "IPER", "false_emergency_rate"
    ]].mean().reset_index()
    emerg_table.round(4).to_csv(os.path.join(ana_dir, "emergency_metrics.csv"), index=False)
    print(f"Saved {os.path.join(ana_dir, 'emergency_metrics.csv')}")

    # 4. Sample Efficiency Table
    sample_eff = df_res.groupby(["model", "N"])[["macro_f1", "IPER", "emergency_recall"]].mean().unstack(level=1)
    sample_eff.round(4).to_csv(os.path.join(ana_dir, "sample_efficiency.csv"))
    print(f"Saved {os.path.join(ana_dir, 'sample_efficiency.csv')}")

    # 5. Speaker Analysis (Dysarthric vs Control)
    spk_analysis = df_res.groupby(["model", "cohort_group"])[[
        "macro_f1", "emergency_macro_f1", "emergency_recall", "IPER"
    ]].mean().reset_index()
    spk_analysis.round(4).to_csv(os.path.join(ana_dir, "speaker_analysis.csv"), index=False)
    print(f"Saved {os.path.join(ana_dir, 'speaker_analysis.csv')}")

    # 6. Collapse Analysis
    collapse_table = df_res.groupby("model").agg(
        total_cells=("majority_collapse", "count"),
        collapsed_cells=("majority_collapse", "sum"),
        collapse_rate=("majority_collapse", "mean"),
        mean_gen_fraction=("predicted_general_fraction", "mean"),
        mean_emerg_fraction=("predicted_emergency_fraction", "mean"),
        mean_unique_classes=("num_unique_predicted_classes", "mean")
    ).reset_index()
    collapse_table["collapse_percentage"] = (collapse_table["collapse_rate"] * 100).round(2)
    collapse_table.to_csv(os.path.join(ana_dir, "collapse_analysis.csv"), index=False)
    print(f"Saved {os.path.join(ana_dir, 'collapse_analysis.csv')}")

    # Figures Generation
    print("\nGenerating publication-quality comparison figures...")
    models_ordered = ["unweighted_adapter", "weighted_adapter", "unweighted_rbf_svm", "weighted_rbf_svm"]
    colors = {"unweighted_adapter": "#1f77b4", "weighted_adapter": "#aec7e8",
              "unweighted_rbf_svm": "#d62728", "weighted_rbf_svm": "#2ca02c"}
    labels = {
        "unweighted_adapter": "Unweighted Adapter (Control)",
        "weighted_adapter": "Weighted Adapter (Rescue)",
        "unweighted_rbf_svm": "Unweighted RBF-SVM (Control)",
        "weighted_rbf_svm": "Weighted RBF-SVM (Rescue)"
    }

    # Figure 1: Macro-F1 Recovery Comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    for m in models_ordered:
        sub = df_res[df_res["model"] == m].groupby("N")["macro_f1"].mean()
        plt.plot(sub.index, sub.values, marker="o", linewidth=2.2, label=labels[m], color=colors[m])
    plt.title("Figure 1: Macro-F1 across Few-Shot Budgets N (Subphase 6.5.2)", fontsize=13, fontweight="bold")
    plt.xlabel("Sample Budget N (Calibration Set Size)", fontsize=11)
    plt.ylabel("Macro-F1 (7-Class Canonical)", fontsize=11)
    plt.xticks([20, 30, 40, 50])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "macro_f1_recovery_comparison.png"))
    plt.close()

    # Figure 2: Emergency Recall Comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    for m in models_ordered:
        sub = df_res[df_res["model"] == m].groupby("N")["emergency_recall"].mean()
        plt.plot(sub.index, sub.values, marker="s", linewidth=2.2, label=labels[m], color=colors[m])
    plt.title("Figure 2: Mean Emergency Recall across Calibration Budgets N", fontsize=13, fontweight="bold")
    plt.xlabel("Sample Budget N", fontsize=11)
    plt.ylabel("Emergency Recall (True Positive Rate)", fontsize=11)
    plt.xticks([20, 30, 40, 50])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "emergency_recall_comparison.png"))
    plt.close()

    # Figure 3: IPER Comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    for m in models_ordered:
        sub = df_res[df_res["model"] == m].groupby("N")["IPER"].mean() * 100
        plt.plot(sub.index, sub.values, marker="^", linewidth=2.2, label=labels[m], color=colors[m])
    plt.title("Figure 3: Intent Preservation Rate (IPER) across Models", fontsize=13, fontweight="bold")
    plt.xlabel("Sample Budget N", fontsize=11)
    plt.ylabel("IPER (%) [Emergency Recognition]", fontsize=11)
    plt.xticks([20, 30, 40, 50])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "iper_comparison.png"))
    plt.close()

    # Figure 4: Sample Efficiency Comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    width = 0.18
    x = np.arange(4)
    ns = [20, 30, 40, 50]
    for i, m in enumerate(models_ordered):
        vals = [df_res[(df_res["model"] == m) & (df_res["N"] == n)]["emergency_macro_f1"].mean() for n in ns]
        plt.bar(x + i * width, vals, width, label=labels[m], color=colors[m])
    plt.title("Figure 4: Emergency Macro-F1 Sample Efficiency by Budget", fontsize=13, fontweight="bold")
    plt.xlabel("Sample Budget N", fontsize=11)
    plt.ylabel("Emergency Macro-F1", fontsize=11)
    plt.xticks(x + 1.5 * width, ["N=20", "N=30", "N=40", "N=50"])
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sample_efficiency_comparison.png"))
    plt.close()

    # Figure 5: Majority Collapse Comparison
    plt.figure(figsize=(8, 5), dpi=300)
    rates = [collapse_table[collapse_table["model"] == m]["collapse_percentage"].iloc[0] for m in models_ordered]
    bars = plt.bar(["Unw Adapter", "W Adapter", "Unw RBF", "W RBF"], rates, color=[colors[m] for m in models_ordered], width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.title("Figure 5: Majority-Class Collapse Rate (100% General Speech)", fontsize=13, fontweight="bold")
    plt.ylabel("Collapse Percentage (%)", fontsize=11)
    plt.ylim([0, 110])
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "majority_collapse_comparison.png"))
    plt.close()
    print("Figures successfully generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Subphase 6.5.2 Classical Rescue")
    parser.add_argument("--pilot", action="store_true", help="Run 24-cell pilot verification")
    parser.add_argument("--full", action="store_true", help="Run full 240-cell experiment")
    parser.add_argument("--analysis-only", action="store_true", help="Run analysis and figure generation on existing results")
    args = parser.parse_args()

    if args.analysis_only:
        res_dir = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "results")
        ana_dir = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "analysis")
        fig_dir = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "figures")
        df_results = pd.read_csv(os.path.join(res_dir, "classical_rescue_results.csv"))
        df_preds = pd.read_csv(os.path.join(res_dir, "classical_rescue_predictions.csv"))
        build_analysis_and_figures(df_results, df_preds, ana_dir, fig_dir)
    elif args.pilot:
        run_experiment(pilot_mode=True)
    elif args.full:
        run_experiment(pilot_mode=False)
    else:
        run_experiment(pilot_mode=True)
