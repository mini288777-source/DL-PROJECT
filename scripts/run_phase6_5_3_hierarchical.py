"""
Subphase 6.5.3 — Hierarchical Emergency Recognition Experiment Runner

Architecture:
  Stage 1: Binary Emergency Detector (General Speech vs. Emergency)
  Stage 2: Conditional Emergency Intent Router (Trained strictly on emergency adaptation data)
  End-to-End: Hierarchical Gated Routing

Inputs:
  Track B: phase6_5/splits/recovery_splits_hierarchical_ready.json
  384-D Whisper embeddings (clean)
  N in {20, 30, 40, 50}, 15 speakers

Models:
  H1: hierarchical_adapter (Stage 1 Adapter + Stage 2 Adapter)
  H2: hierarchical_rbf_svm (Stage 1 RBF-SVM + Stage 2 RBF-SVM)
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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, precision_recall_curve, auc, roc_auc_score,
    average_precision_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Subphase 6.5.3 Hierarchical Experiment")
    parser.add_argument("--pilot", action="store_true", help="Run 12-cell pilot only")
    parser.add_argument("--config", type=str, default="phase6_5/hierarchical/config/hierarchical_config.yaml")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_iper(y_true: List[str], y_pred: List[str]) -> Tuple[float, int, int]:
    """Canonical IPER calculation from Phase 5/6.5.2."""
    emergency_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yt != "GENERAL_SPEECH"]
    total_emergency = len(emergency_pairs)
    if total_emergency == 0:
        return 0.0, 0, 0
    preserved_emergency = sum(1 for yt, yp in emergency_pairs if yt == yp)
    return float(preserved_emergency / total_emergency), preserved_emergency, total_emergency


def run_experiment(pilot: bool = False, config_path: str = "phase6_5/hierarchical/config/hierarchical_config.yaml"):
    cfg = load_config(config_path)
    seed = cfg["cohort"]["seed"]
    set_seed(seed)
    
    print(f"============================================================")
    print(f" SUBPHASE 6.5.3 — HIERARCHICAL EMERGENCY RECOGNITION")
    print(f" Mode: {'PILOT (12 cells)' if pilot else 'FULL EXPERIMENT (120 cells)'}")
    print(f" Split: Track B ({cfg['data']['splits_track_b_path']})")
    print(f" Master Seed: {seed}")
    print(f"============================================================")
    
    # 1. Load data assets
    with open(cfg["data"]["splits_track_b_path"], "r", encoding="utf-8") as f:
        all_splits = json.load(f)
    
    manifest_df = pd.read_csv(cfg["data"]["manifest_path"])
    rec_intent = dict(zip(manifest_df["recording_id"], manifest_df["intent"]))
    rec_pg = dict(zip(manifest_df["recording_id"], manifest_df.get("prompt_group_id", manifest_df["transcript"])))
    
    emb_dir = cfg["data"]["whisper_cache_dir"]
    emb_dim = cfg["data"]["embedding_dim"]
    base_torgo_dir = os.path.dirname(emb_dir)
    
    # Filter splits if pilot
    if pilot:
        pilot_speakers = set(cfg["pilot"]["speakers"])
        pilot_budgets = set(cfg["pilot"]["sample_sizes"])
        target_splits = [s for s in all_splits if s["speaker_id"] in pilot_speakers and s["sample_n"] in pilot_budgets]
    else:
        target_splits = all_splits

    # Pre-cache embeddings
    print("Pre-caching Whisper embeddings...")
    embedding_cache = {}
    unique_recs = set()
    for s in target_splits:
        unique_recs.update(s["adaptation_recording_ids"])
        unique_recs.update(s["test_recording_ids"])
    
    loaded_count = 0
    for r in sorted(list(unique_recs)):
        try:
            emb = load_embedding_file(r, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=emb_dim)
            embedding_cache[r] = emb.astype(np.float32)
            loaded_count += 1
        except Exception as e:
            print(f"Error loading embedding for {r}: {e}")
            raise
    print(f"Successfully cached {loaded_count} embeddings (dim={emb_dim}).")
        
    model_configs = cfg["models"]
    thresholds = cfg["inference"]["thresholds"]
    primary_threshold = cfg["inference"]["primary_threshold"]
    
    results_records = []
    hierarchical_pred_records = []
    stage1_pred_records = []
    stage2_pred_records = []
    error_taxonomy_records = []
    threshold_records = []
    
    total_cells = len(target_splits) * len(model_configs)
    completed_cells = 0
    start_total_time = time.time()
    
    for split_idx, s in enumerate(target_splits):
        spk = s["speaker_id"]
        n = int(s["sample_n"])
        cohort = "dysarthric" if spk in DYSARTHRIC_SET else "control"
        
        adapt_recs = s["adaptation_recording_ids"]
        test_recs = s["test_recording_ids"]
        
        adapt_split_hash = s.get("adaptation_split_hash", "")
        test_split_hash = s.get("test_split_hash", "")
        
        # Adaptation arrays
        X_adapt = np.stack([embedding_cache[r] for r in adapt_recs])
        y_adapt = [rec_intent[r] for r in adapt_recs]
        
        # Test arrays
        X_test = np.stack([embedding_cache[r] for r in test_recs])
        y_test = [rec_intent[r] for r in test_recs]
        test_pgs = [rec_pg.get(r, "UNKNOWN") for r in test_recs]
        
        # Feasible emergency classes (EXCLUSIVELY from adaptation set)
        feasible_emergency_classes = sorted(list(set(l for l in y_adapt if l != "GENERAL_SPEECH")))
        missing_emergency_classes = [c for c in EMERGENCY_CLASSES if c not in feasible_emergency_classes]
        stage2_num_classes = len(feasible_emergency_classes)
        
        # Stage 1 targets (0 = GENERAL_SPEECH, 1 = EMERGENCY)
        y_adapt_stage1 = ["EMERGENCY" if l != "GENERAL_SPEECH" else "GENERAL_SPEECH" for l in y_adapt]
        y_test_stage1_binary = [1 if l != "GENERAL_SPEECH" else 0 for l in y_test]
        
        # Stage 2 training data (EMERGENCY ONLY)
        emerg_adapt_indices = [i for i, l in enumerate(y_adapt) if l != "GENERAL_SPEECH"]
        X_emerg_adapt = X_adapt[emerg_adapt_indices]
        y_emerg_adapt = [y_adapt[i] for i in emerg_adapt_indices]
        
        for m_cfg in model_configs:
            m_name = m_cfg["name"]
            exp_id = f"{spk}_{m_name}_N{n}"
            
            # ----------------------------------------------------
            # STAGE 1 TRAINING
            # ----------------------------------------------------
            stage1_cfg = m_cfg["stage1"]
            t0 = time.perf_counter()
            
            if stage1_cfg["model_type"] == "adapter":
                stage1_model = ClassicalAdapterClassifier(
                    input_dim=emb_dim,
                    hidden_dim=stage1_cfg["hidden_dim"],
                    num_classes=2,
                    dropout=stage1_cfg["dropout"],
                    learning_rate=stage1_cfg["learning_rate"],
                    weight_decay=stage1_cfg["weight_decay"],
                    epochs=stage1_cfg["epochs"],
                    batch_size=stage1_cfg["batch_size"],
                    class_weight=stage1_cfg["class_weight"],
                    seed=seed,
                    all_classes=["GENERAL_SPEECH", "EMERGENCY"]
                )
                stage1_model.fit(X_adapt, y_adapt_stage1)
                stage1_train_time = time.perf_counter() - t0
                
                # Inference probabilities
                t_inf0 = time.perf_counter()
                stage1_probs_all = stage1_model.predict_proba(X_test)
                stage1_inf_time = time.perf_counter() - t_inf0
                emerg_col = stage1_model.classes_.index("EMERGENCY")
                p_emerg = stage1_probs_all[:, emerg_col]
                
            elif stage1_cfg["model_type"] == "rbf_svm":
                stage1_model = ClassicalRBFSVMClassifier(
                    C=stage1_cfg["C"],
                    gamma=stage1_cfg["gamma"],
                    class_weight=stage1_cfg["class_weight"],
                    standardize=stage1_cfg["standardize"],
                    probability=stage1_cfg["probability"],
                    seed=seed,
                    all_classes=["GENERAL_SPEECH", "EMERGENCY"]
                )
                stage1_model.fit(X_adapt, y_adapt_stage1)
                stage1_train_time = time.perf_counter() - t0
                
                t_inf0 = time.perf_counter()
                stage1_probs_all = stage1_model.predict_proba(X_test)
                stage1_inf_time = time.perf_counter() - t_inf0
                emerg_col = stage1_model.classes_.index("EMERGENCY")
                p_emerg = stage1_probs_all[:, emerg_col]
            else:
                raise ValueError(f"Unknown Stage 1 model type: {stage1_cfg['model_type']}")
            
            # Primary Stage 1 predictions (at threshold 0.50)
            stage1_pred_binary = [1 if p >= primary_threshold else 0 for p in p_emerg]
            stage1_pred_labels = ["EMERGENCY" if b == 1 else "GENERAL_SPEECH" for b in stage1_pred_binary]
            
            # ----------------------------------------------------
            # STAGE 2 TRAINING
            # ----------------------------------------------------
            stage2_cfg = m_cfg["stage2"]
            t2_0 = time.perf_counter()
            
            is_single_emergency_class = (stage2_num_classes == 1)
            single_emergency_class_name = feasible_emergency_classes[0] if is_single_emergency_class else None
            
            if is_single_emergency_class:
                stage2_model = None
                stage2_train_time = 0.0
            else:
                if stage2_cfg["model_type"] == "adapter":
                    stage2_model = ClassicalAdapterClassifier(
                        input_dim=emb_dim,
                        hidden_dim=stage2_cfg["hidden_dim"],
                        num_classes=stage2_num_classes,
                        dropout=stage2_cfg["dropout"],
                        learning_rate=stage2_cfg["learning_rate"],
                        weight_decay=stage2_cfg["weight_decay"],
                        epochs=stage2_cfg["epochs"],
                        batch_size=stage2_cfg["batch_size"],
                        class_weight=stage2_cfg["class_weight"],
                        seed=seed,
                        all_classes=feasible_emergency_classes
                    )
                    stage2_model.fit(X_emerg_adapt, y_emerg_adapt)
                    stage2_train_time = time.perf_counter() - t2_0
                elif stage2_cfg["model_type"] == "rbf_svm":
                    stage2_model = ClassicalRBFSVMClassifier(
                        C=stage2_cfg["C"],
                        gamma=stage2_cfg["gamma"],
                        class_weight=stage2_cfg["class_weight"],
                        standardize=stage2_cfg["standardize"],
                        probability=stage2_cfg["probability"],
                        seed=seed,
                        all_classes=feasible_emergency_classes
                    )
                    stage2_model.fit(X_emerg_adapt, y_emerg_adapt)
                    stage2_train_time = time.perf_counter() - t2_0
                else:
                    raise ValueError(f"Unknown Stage 2 model type: {stage2_cfg['model_type']}")
            
            # ----------------------------------------------------
            # STAGE 2 DIAGNOSTIC EVALUATION (on true emergency test samples)
            # ----------------------------------------------------
            true_emerg_indices = [i for i, l in enumerate(y_test) if l != "GENERAL_SPEECH"]
            if len(true_emerg_indices) > 0:
                X_true_emerg = X_test[true_emerg_indices]
                y_true_emerg = [y_test[i] for i in true_emerg_indices]
                
                if is_single_emergency_class:
                    s2_diag_preds = [single_emergency_class_name] * len(X_true_emerg)
                else:
                    s2_diag_preds = stage2_model.predict(X_true_emerg)
                    
                s2_macro_f1 = float(f1_score(y_true_emerg, s2_diag_preds, labels=feasible_emergency_classes, average="macro", zero_division=0))
                s2_weighted_f1 = float(f1_score(y_true_emerg, s2_diag_preds, labels=feasible_emergency_classes, average="weighted", zero_division=0))
                s2_bal_acc = float(balanced_accuracy_score(y_true_emerg, s2_diag_preds))
                
                for idx_te, r_idx in enumerate(true_emerg_indices):
                    stage2_pred_records.append({
                        "experiment_id": exp_id,
                        "speaker_id": spk,
                        "cohort_group": cohort,
                        "N": n,
                        "model": m_name,
                        "recording_id": test_recs[r_idx],
                        "true_intent": y_test[r_idx],
                        "stage2_predicted_intent": s2_diag_preds[idx_te],
                        "is_stage2_correct": bool(y_test[r_idx] == s2_diag_preds[idx_te])
                    })
            else:
                s2_macro_f1 = 0.0
                s2_weighted_f1 = 0.0
                s2_bal_acc = 0.0
            
            # ----------------------------------------------------
            # END-TO-END HIERARCHICAL ROUTING (at primary threshold 0.50)
            # ----------------------------------------------------
            t_route0 = time.perf_counter()
            final_predictions = []
            routed_to_stage2_indices = [i for i, b in enumerate(stage1_pred_binary) if b == 1]
            
            if len(routed_to_stage2_indices) > 0:
                X_routed = X_test[routed_to_stage2_indices]
                if is_single_emergency_class:
                    stage2_routed_preds = [single_emergency_class_name] * len(X_routed)
                else:
                    stage2_routed_preds = stage2_model.predict(X_routed)
            else:
                stage2_routed_preds = []
                
            routed_map = dict(zip(routed_to_stage2_indices, stage2_routed_preds))
            
            for i in range(len(y_test)):
                if stage1_pred_binary[i] == 0:
                    final_predictions.append("GENERAL_SPEECH")
                else:
                    final_predictions.append(routed_map[i])
                    
            e2e_inf_time = stage1_inf_time + (time.perf_counter() - t_route0)
            
            # ----------------------------------------------------
            # ERROR TAXONOMY & RECORD CREATION
            # ----------------------------------------------------
            false_emerg = 0
            missed_emerg = 0
            wrong_emerg = 0
            correct_emerg = 0
            correct_gen = 0
            
            for i in range(len(y_test)):
                yt = y_test[i]
                yp = final_predictions[i]
                s1_p = stage1_pred_labels[i]
                rec_id = test_recs[i]
                pg_id = test_pgs[i]
                
                # Stage 1 log
                stage1_pred_records.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "cohort_group": cohort,
                    "N": n,
                    "model": m_name,
                    "recording_id": rec_id,
                    "true_binary": y_test_stage1_binary[i],
                    "pred_binary": stage1_pred_binary[i],
                    "prob_emergency": round(float(p_emerg[i]), 4)
                })
                
                # Determine error category
                if yt == "GENERAL_SPEECH" and yp in EMERGENCY_CLASSES:
                    err_cat = "false_emergency"
                    false_emerg += 1
                elif yt in EMERGENCY_CLASSES and yp == "GENERAL_SPEECH":
                    err_cat = "missed_emergency"
                    missed_emerg += 1
                elif yt in EMERGENCY_CLASSES and yp in EMERGENCY_CLASSES and yt != yp:
                    err_cat = "wrong_emergency_type"
                    wrong_emerg += 1
                elif yt in EMERGENCY_CLASSES and yp == yt:
                    err_cat = "correct_emergency"
                    correct_emerg += 1
                elif yt == "GENERAL_SPEECH" and yp == "GENERAL_SPEECH":
                    err_cat = "correct_general"
                    correct_gen += 1
                else:
                    err_cat = "other"
                    
                hierarchical_pred_records.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "cohort_group": cohort,
                    "N": n,
                    "model": m_name,
                    "recording_id": rec_id,
                    "prompt_group_id": pg_id,
                    "true_intent": yt,
                    "stage1_pred": s1_p,
                    "stage2_pred": routed_map.get(i, "NOT_ROUTED"),
                    "final_pred": yp,
                    "is_correct": bool(yt == yp),
                    "error_category": err_cat
                })
                
            assert (false_emerg + missed_emerg + wrong_emerg + correct_emerg + correct_gen) == len(y_test)
            
            error_taxonomy_records.append({
                "experiment_id": exp_id,
                "speaker_id": spk,
                "cohort_group": cohort,
                "N": n,
                "model": m_name,
                "total_samples": len(y_test),
                "false_emergency": false_emerg,
                "missed_emergency": missed_emerg,
                "wrong_emergency_type": wrong_emerg,
                "correct_emergency": correct_emerg,
                "correct_general": correct_gen,
                "false_emergency_rate": round(false_emerg / sum(1 for yt in y_test if yt == "GENERAL_SPEECH"), 4) if sum(1 for yt in y_test if yt == "GENERAL_SPEECH") > 0 else 0.0,
                "emergency_recall": round(correct_emerg / sum(1 for yt in y_test if yt in EMERGENCY_CLASSES), 4) if sum(1 for yt in y_test if yt in EMERGENCY_CLASSES) > 0 else 0.0
            })
            
            # ----------------------------------------------------
            # STAGE 1 METRICS
            # ----------------------------------------------------
            s1_acc = float(accuracy_score(y_test_stage1_binary, stage1_pred_binary))
            s1_macro_f1 = float(f1_score(y_test_stage1_binary, stage1_pred_binary, average="macro", zero_division=0))
            s1_rec = float(recall_score(y_test_stage1_binary, stage1_pred_binary, pos_label=1, zero_division=0))
            s1_prec = float(precision_score(y_test_stage1_binary, stage1_pred_binary, pos_label=1, zero_division=0))
            s1_f1 = float(f1_score(y_test_stage1_binary, stage1_pred_binary, pos_label=1, zero_division=0))
            s1_spec = float(recall_score(y_test_stage1_binary, stage1_pred_binary, pos_label=0, zero_division=0))
            s1_fpr = 1.0 - s1_spec
            
            # Stage 1 PR-AUC and ROC-AUC
            s1_pr_auc = float(average_precision_score(y_test_stage1_binary, p_emerg))
            try:
                s1_roc_auc = float(roc_auc_score(y_test_stage1_binary, p_emerg))
            except Exception:
                s1_roc_auc = 0.5
                
            # ----------------------------------------------------
            # END-TO-END METRICS
            # ----------------------------------------------------
            e2e_macro_f1 = float(f1_score(y_test, final_predictions, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
            
            feasible_classes_7 = sorted(list(set(y_adapt)))
            e2e_feas_macro_f1 = float(f1_score(y_test, final_predictions, labels=feasible_classes_7, average="macro", zero_division=0))
            
            true_emerg_total = sum(1 for yt in y_test if yt in EMERGENCY_CLASSES)
            pred_emerg_total = sum(1 for yp in final_predictions if yp in EMERGENCY_CLASSES)
            correct_emerg_total = sum(1 for yt, yp in zip(y_test, final_predictions) if yt in EMERGENCY_CLASSES and yt == yp)
            
            e2e_emerg_rec = float(correct_emerg_total / true_emerg_total) if true_emerg_total > 0 else 0.0
            e2e_emerg_prec = float(correct_emerg_total / pred_emerg_total) if pred_emerg_total > 0 else 0.0
            e2e_emerg_f1 = float(2 * e2e_emerg_prec * e2e_emerg_rec / (e2e_emerg_prec + e2e_emerg_rec)) if (e2e_emerg_prec + e2e_emerg_rec) > 0 else 0.0
            
            iper_rate, _, _ = compute_iper(y_test, final_predictions)
            
            gen_total = sum(1 for yt in y_test if yt == "GENERAL_SPEECH")
            gen_false_emerg = sum(1 for yt, yp in zip(y_test, final_predictions) if yt == "GENERAL_SPEECH" and yp in EMERGENCY_CLASSES)
            e2e_fer = float(gen_false_emerg / gen_total) if gen_total > 0 else 0.0
            gen_recall = float(sum(1 for yt, yp in zip(y_test, final_predictions) if yt == "GENERAL_SPEECH" and yp == "GENERAL_SPEECH") / gen_total) if gen_total > 0 else 0.0
            
            # ----------------------------------------------------
            # THRESHOLD ANALYSIS
            # ----------------------------------------------------
            for th in thresholds:
                th_s1_pred_bin = [1 if p >= th else 0 for p in p_emerg]
                th_s1_rec = float(recall_score(y_test_stage1_binary, th_s1_pred_bin, pos_label=1, zero_division=0))
                th_s1_prec = float(precision_score(y_test_stage1_binary, th_s1_pred_bin, pos_label=1, zero_division=0))
                th_s1_spec = float(recall_score(y_test_stage1_binary, th_s1_pred_bin, pos_label=0, zero_division=0))
                th_s1_fpr = 1.0 - th_s1_spec
                
                # Routed at threshold th
                th_routed_idx = [i for i, b in enumerate(th_s1_pred_bin) if b == 1]
                if len(th_routed_idx) > 0:
                    if is_single_emergency_class:
                        th_s2_preds = [single_emergency_class_name] * len(th_routed_idx)
                    else:
                        th_s2_preds = stage2_model.predict(X_test[th_routed_idx])
                else:
                    th_s2_preds = []
                th_routed_map = dict(zip(th_routed_idx, th_s2_preds))
                
                th_final_preds = [th_routed_map[i] if th_s1_pred_bin[i] == 1 else "GENERAL_SPEECH" for i in range(len(y_test))]
                th_iper, _, _ = compute_iper(y_test, th_final_preds)
                th_macro_f1 = float(f1_score(y_test, th_final_preds, labels=CANONICAL_CLASSES, average="macro", zero_division=0))
                th_fer = float(sum(1 for yt, yp in zip(y_test, th_final_preds) if yt == "GENERAL_SPEECH" and yp in EMERGENCY_CLASSES) / gen_total) if gen_total > 0 else 0.0
                
                threshold_records.append({
                    "experiment_id": exp_id,
                    "speaker_id": spk,
                    "cohort_group": cohort,
                    "N": n,
                    "model": m_name,
                    "threshold": th,
                    "stage1_recall": round(th_s1_rec, 4),
                    "stage1_precision": round(th_s1_prec, 4),
                    "stage1_fpr": round(th_s1_fpr, 4),
                    "end_to_end_iper": round(th_iper, 4),
                    "end_to_end_macro_f1": round(th_macro_f1, 4),
                    "false_emergency_rate": round(th_fer, 4)
                })
            
            # Record primary result
            config_hash = hashlib.sha256(f"{exp_id}_{seed}".encode("utf-8")).hexdigest()[:12]
            results_records.append({
                "experiment_id": exp_id,
                "speaker_id": spk,
                "cohort_group": cohort,
                "N": n,
                "model": m_name,
                "split_protocol": "recovery_stratified_track_b",
                "adaptation_split_hash": adapt_split_hash,
                "test_split_hash": test_split_hash,
                "feasible_emergency_classes": ";".join(feasible_emergency_classes),
                "missing_emergency_classes": ";".join(missing_emergency_classes),
                "stage2_num_classes": stage2_num_classes,
                # Stage 1
                "stage1_emergency_precision": round(s1_prec, 4),
                "stage1_emergency_recall": round(s1_rec, 4),
                "stage1_emergency_f1": round(s1_f1, 4),
                "stage1_specificity": round(s1_spec, 4),
                "stage1_false_positive_rate": round(s1_fpr, 4),
                "stage1_pr_auc": round(s1_pr_auc, 4),
                "stage1_roc_auc": round(s1_roc_auc, 4),
                # Stage 2
                "stage2_macro_f1": round(s2_macro_f1, 4),
                "stage2_balanced_accuracy": round(s2_bal_acc, 4),
                "stage2_weighted_f1": round(s2_weighted_f1, 4),
                # End-to-End
                "macro_f1": round(e2e_macro_f1, 4),
                "emergency_precision": round(e2e_emerg_prec, 4),
                "emergency_recall": round(e2e_emerg_rec, 4),
                "IPER": round(iper_rate, 4),
                "false_emergency_rate": round(e2e_fer, 4),
                "general_speech_recall": round(gen_recall, 4),
                "feasible_macro_f1": round(e2e_feas_macro_f1, 4),
                # Error counts
                "false_emergency": false_emerg,
                "missed_emergency": missed_emerg,
                "wrong_emergency_type": wrong_emerg,
                "correct_emergency": correct_emerg,
                # Runtime
                "stage1_training_time": round(stage1_train_time, 4),
                "stage2_training_time": round(stage2_train_time, 4),
                "inference_time": round(e2e_inf_time, 4),
                # Reproducibility
                "seed": seed,
                "config_hash": config_hash,
                "status": "SUCCESS"
            })
            
            completed_cells += 1
            if completed_cells % 10 == 0 or completed_cells == total_cells:
                elapsed = time.time() - start_total_time
                print(f"[{completed_cells}/{total_cells}] Completed {exp_id} (elapsed: {elapsed:.1f}s)")
                
    # ----------------------------------------------------
    # WRITE PRIMARY RESULTS
    # ----------------------------------------------------
    res_dir = cfg["output"]["results_dir"]
    analysis_dir = cfg["output"]["analysis_dir"]
    fig_dir = cfg["output"]["figures_dir"]
    
    res_df = pd.DataFrame(results_records)
    res_csv = os.path.join(res_dir, "hierarchical_results.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"\nSaved primary results ({len(res_df)} rows) to {res_csv}")
    
    hier_preds_df = pd.DataFrame(hierarchical_pred_records)
    hier_preds_csv = os.path.join(res_dir, "hierarchical_predictions.csv")
    hier_preds_df.to_csv(hier_preds_csv, index=False)
    print(f"Saved hierarchical predictions ({len(hier_preds_df)} rows) to {hier_preds_csv}")
    
    s1_preds_df = pd.DataFrame(stage1_pred_records)
    s1_preds_csv = os.path.join(res_dir, "stage1_predictions.csv")
    s1_preds_df.to_csv(s1_preds_csv, index=False)
    print(f"Saved Stage 1 predictions ({len(s1_preds_df)} rows) to {s1_preds_csv}")
    
    s2_preds_df = pd.DataFrame(stage2_pred_records)
    s2_preds_csv = os.path.join(res_dir, "stage2_predictions.csv")
    s2_preds_df.to_csv(s2_preds_csv, index=False)
    print(f"Saved Stage 2 predictions ({len(s2_preds_df)} rows) to {s2_preds_csv}")
    
    err_df = pd.DataFrame(error_taxonomy_records)
    err_csv = os.path.join(res_dir, "error_taxonomy.csv")
    err_df.to_csv(err_csv, index=False)
    print(f"Saved error taxonomy ({len(err_df)} rows) to {err_csv}")
    
    th_df = pd.DataFrame(threshold_records)
    th_csv = os.path.join(res_dir, "threshold_analysis.csv")
    th_df.to_csv(th_csv, index=False)
    print(f"Saved threshold analysis ({len(th_df)} rows) to {th_csv}")
    
    # ----------------------------------------------------
    # GENERATE DERIVED ANALYSIS FILES
    # ----------------------------------------------------
    # 1. stage1_metrics.csv
    s1_summary = res_df.groupby(["model", "N"])[[
        "stage1_emergency_recall", "stage1_emergency_precision", "stage1_emergency_f1",
        "stage1_specificity", "stage1_false_positive_rate", "stage1_pr_auc", "stage1_roc_auc"
    ]].mean().round(4).reset_index()
    s1_summary.to_csv(os.path.join(analysis_dir, "stage1_metrics.csv"), index=False)
    
    # 2. stage2_metrics.csv
    s2_summary = res_df.groupby(["model", "N"])[[
        "stage2_macro_f1", "stage2_balanced_accuracy", "stage2_weighted_f1"
    ]].mean().round(4).reset_index()
    s2_summary.to_csv(os.path.join(analysis_dir, "stage2_metrics.csv"), index=False)
    
    # 3. end_to_end_metrics.csv
    e2e_summary = res_df.groupby(["model", "N"])[[
        "macro_f1", "feasible_macro_f1", "emergency_recall", "emergency_precision",
        "IPER", "false_emergency_rate", "general_speech_recall"
    ]].mean().round(4).reset_index()
    e2e_summary.to_csv(os.path.join(analysis_dir, "end_to_end_metrics.csv"), index=False)
    
    # 4. speaker_analysis.csv
    spk_summary = res_df.groupby(["model", "cohort_group"])[[
        "macro_f1", "IPER", "emergency_recall", "emergency_precision", "false_emergency_rate"
    ]].mean().round(4).reset_index()
    spk_summary.to_csv(os.path.join(analysis_dir, "speaker_analysis.csv"), index=False)
    
    # 5. sample_efficiency.csv
    se_summary = res_df.groupby(["model", "N"])[[
        "IPER", "false_emergency_rate", "macro_f1", "emergency_precision"
    ]].mean().round(4).reset_index()
    se_summary.to_csv(os.path.join(analysis_dir, "sample_efficiency.csv"), index=False)
    
    # 6. feasible_class_analysis.csv
    feas_summary = res_df.groupby(["N", "stage2_num_classes"])["experiment_id"].count().reset_index()
    feas_summary.rename(columns={"experiment_id": "condition_count"}, inplace=True)
    feas_summary.to_csv(os.path.join(analysis_dir, "feasible_class_analysis.csv"), index=False)
    
    # 7. flat_vs_hierarchical.csv (compare against Phase 6.5.2 flat weighted results)
    flat_res_path = os.path.join(REPO_ROOT, "phase6_5", "classical_rescue", "results", "classical_rescue_results.csv")
    if os.path.exists(flat_res_path):
        flat_df = pd.read_csv(flat_res_path)
        flat_weighted = flat_df[flat_df["balancing_method"] == "class_weighted"]
        
        flat_agg = flat_weighted.groupby(["model", "N"])[[
            "macro_f1", "emergency_recall", "emergency_precision", "IPER", "false_emergency_rate"
        ]].mean().round(4).reset_index()
        flat_agg["architecture"] = "flat_weighted"
        
        hier_agg = res_df.groupby(["model", "N"])[[
            "macro_f1", "emergency_recall", "emergency_precision", "IPER", "false_emergency_rate"
        ]].mean().round(4).reset_index()
        hier_agg["architecture"] = "hierarchical"
        
        flat_vs_hier = pd.concat([flat_agg, hier_agg], ignore_index=True)
        flat_vs_hier.to_csv(os.path.join(analysis_dir, "flat_vs_hierarchical.csv"), index=False)
        print("Generated flat_vs_hierarchical.csv comparison.")
    
    # ----------------------------------------------------
    # GENERATE PUBLICATION FIGURES
    # ----------------------------------------------------
    print("Generating publication figures...")
    
    # Figure 1: stage1_precision_recall.png
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in res_df["model"].unique():
        sub_th = th_df[th_df["model"] == m].groupby("threshold")[["stage1_recall", "stage1_precision"]].mean().reset_index()
        ax.plot(sub_th["stage1_recall"], sub_th["stage1_precision"], marker="o", label=f"{m} (threshold sweep)")
    ax.set_xlabel("Stage 1 Emergency Recall")
    ax.set_ylabel("Stage 1 Emergency Precision")
    ax.set_title("Stage 1 Precision-Recall Operating Characteristic (Thresholds 0.30 - 0.70)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "stage1_precision_recall.png"), dpi=300)
    plt.close(fig)
    
    # Figure 2: false_alarm_vs_recall.png
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in res_df["model"].unique():
        sub_th = th_df[th_df["model"] == m].groupby("threshold")[["stage1_recall", "false_emergency_rate"]].mean().reset_index()
        ax.plot(sub_th["false_emergency_rate"], sub_th["stage1_recall"], marker="s", label=f"{m}")
    ax.set_xlabel("False Emergency Rate (FPR on Routine Speech)")
    ax.set_ylabel("Stage 1 Emergency Recall")
    ax.set_title("Safety-Critical Trade-Off: False Emergency Rate vs. Recall")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "false_alarm_vs_recall.png"), dpi=300)
    plt.close(fig)
    
    # Figure 3: hierarchical_iper.png
    fig, ax = plt.subplots(figsize=(8, 5))
    iper_by_n = res_df.groupby(["model", "N"])["IPER"].mean().unstack(level=0)
    iper_by_n.plot(kind="bar", ax=ax, width=0.6)
    ax.set_xlabel("Adaptation Budget (N)")
    ax.set_ylabel("End-to-End IPER")
    ax.set_title("Hierarchical Intent-Preserving Emergency Recall (IPER) across N")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(["Hierarchical Adapter", "Hierarchical RBF-SVM"])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "hierarchical_iper.png"), dpi=300)
    plt.close(fig)
    
    # Figure 4: hierarchical_sample_efficiency.png
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in res_df["model"].unique():
        sub = res_df[res_df["model"] == m].groupby("N")["false_emergency_rate"].mean()
        ax.plot(sub.index, sub.values, marker="^", label=f"{m} (False Emerg Rate)")
    ax.set_xlabel("Adaptation Budget (N)")
    ax.set_ylabel("False Emergency Rate")
    ax.set_title("Sample Efficiency: False Emergency Rate vs. Few-Shot Budget N")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "hierarchical_sample_efficiency.png"), dpi=300)
    plt.close(fig)
    
    # Figure 5: hierarchical_speaker_variation.png
    fig, ax = plt.subplots(figsize=(12, 6))
    spk_iper = res_df.groupby(["speaker_id", "model"])["IPER"].mean().unstack(level=1)
    spk_iper.plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Mean IPER")
    ax.set_title("Hierarchical Emergency Recognition across TORGO Speakers (Dysarthric vs. Control)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(["Hierarchical Adapter", "Hierarchical RBF-SVM"])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "hierarchical_speaker_variation.png"), dpi=300)
    plt.close(fig)
    
    # Figure 6: confusion_matrix_end_to_end.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, m in enumerate(["hierarchical_adapter", "hierarchical_rbf_svm"]):
        m_preds = hier_preds_df[hier_preds_df["model"] == m]
        cm = confusion_matrix(m_preds["true_intent"], m_preds["final_pred"], labels=CANONICAL_CLASSES)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        im = axes[ax_idx].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[ax_idx].set_title(f"End-to-End Confusion: {m}")
        axes[ax_idx].set_xticks(range(len(CANONICAL_CLASSES)))
        axes[ax_idx].set_yticks(range(len(CANONICAL_CLASSES)))
        axes[ax_idx].set_xticklabels([c[:4] for c in CANONICAL_CLASSES], rotation=45)
        axes[ax_idx].set_yticklabels([c[:4] for c in CANONICAL_CLASSES])
        axes[ax_idx].set_ylabel("True Intent")
        axes[ax_idx].set_xlabel("Predicted Intent")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "confusion_matrix_end_to_end.png"), dpi=300)
    plt.close(fig)
    
    print("\nAll figures generated successfully.")
    print("Subphase 6.5.3 execution complete!")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(pilot=args.pilot, config_path=args.config)
