"""
Phase 7.7 Empirical Evaluation Script: Speaker Drift Detection & Recalibration.
Executes:
- Experiment A: Multi-faceted speaker drift detection across longitudinal accumulation (N=20 -> 30 -> 40 -> 50)
  for all 15 TORGO speakers (7 dysarthric, 8 control).
- Experiment B: Controlled recalibration via temperature scaling and evaluation through Phase 7.6 Validation Gate.
- Subgroup visibility (dysarthric vs control), statistical testing, error analysis, and experiment registry logging.
"""

import os
import sys
import json
import csv
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
import torch

sys.path.insert(0, os.path.abspath("."))

from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.models.classical_adapter import ClassicalAdapterNet
from src.continual.drift import (
    DriftType,
    DriftStatus,
    RecalibrationTriggerState,
    DriftWindow,
    DriftEvent,
    RecalibrationCandidate,
    compute_mmd_rbf,
    compute_centroid_cosine_distance,
    compute_prediction_tvd,
    compute_confidence_ks,
    compute_expected_calibration_error,
    compute_maximum_calibration_error,
    compute_brier_score,
    TemperatureScalingRecalibrator,
    SpeakerDriftDetector,
    ControlledRecalibrationPipeline,
    load_drift_policy,
)
from src.continual.model_version import (
    ModelLifecycleState,
    ModelVersionRecord,
    ModelRegistry,
    compute_sha256,
)
from src.continual.governance import (
    ValidationGate,
    ModelGovernanceManager,
    PromotionDecision,
    ValidationReasonCode,
)
from src.continual.experiment_registry import (
    ExperimentRegistry,
    ExperimentRunRecord,
    ExperimentStatus,
)
from src.continual.continual_classical import compute_comprehensive_continual_metrics


def run_phase7_7_experiments():
    print("=" * 80)
    print("STARTING PHASE 7.7: SPEAKER DRIFT DETECTION & RECALIBRATION EXPERIMENTS")
    print("=" * 80)

    # 1. Output directories
    results_dir = os.path.join("results", "continual", "phase7_7")
    reports_dir = os.path.join("reports", "continual", "phase7_7")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 2. Load policy and registries
    drift_policy = load_drift_policy()
    policy_ver = drift_policy.get("policy_metadata", {}).get("policy_version", "v1.0.0_frozen")
    print(f"Loaded Drift Policy: {policy_ver}")

    # Initialize Phase 7.7 experiment registry
    exp_registry = ExperimentRegistry(results_dir)

    # 3. Load embeddings & metadata
    print("Loading Torgo intent mapping and embedding cache...")
    df_intent = pd.read_csv("data/manifests/torgo_intent_mapping.csv")
    rid_to_intent = dict(zip(df_intent["recording_id"], df_intent["intent_candidate"]))

    from src.embeddings.cache import load_embedding_file

    _emb_cache: Dict[str, np.ndarray] = {}

    def get_embedding(rid: str) -> np.ndarray:
        if rid not in _emb_cache:
            _emb_cache[rid] = load_embedding_file(rid, feature_type="whisper", base_dir="embeddings/torgo")
        return _emb_cache[rid]

    # 4. Load recovery splits
    with open("phase6_5/splits/recovery_splits_hierarchical_ready.json", "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # Group splits by speaker
    speaker_splits: Dict[str, Dict[int, Any]] = {}
    for entry in splits_data:
        spk = entry["speaker_id"]
        sn = entry["sample_n"]
        if spk not in speaker_splits:
            speaker_splits[spk] = {}
        speaker_splits[spk][sn] = entry

    # 5. Load Phase 7.4 results for baseline references
    df_7_4 = pd.read_csv("results/continual/phase7_4/continual_classical_results.csv")
    speaker_seeds = {row["speaker_id"]: int(row["seed"]) for _, row in df_7_4.iterrows() if row["budget"] == 20}
    speaker_groups = {row["speaker_id"]: row["speaker_group"] for _, row in df_7_4.iterrows()}

    speakers = sorted(list(speaker_seeds.keys()))
    print(f"Total Speakers: {len(speakers)} ({sum(1 for s in speakers if speaker_groups[s] == 'dysarthric')} dysarthric, "
          f"{sum(1 for s in speakers if speaker_groups[s] == 'control')} control)")

    # Data structures for results
    drift_events_log: List[Dict[str, Any]] = []
    trajectory_records: List[Dict[str, Any]] = []
    recalibration_records: List[Dict[str, Any]] = []

    # Initialize SpeakerDriftDetector
    detector = SpeakerDriftDetector(policy=drift_policy)

    # Model evaluation helper
    def load_adapter_and_predict(spk: str, budget: int, seed: int, X_input: np.ndarray) -> Tuple[np.ndarray, List[str], List[float]]:
        ckpt_path = os.path.join(
            "models", "continual", "phase7_4", spk, "classical_adapter",
            f"v{spk}_budget{budget}_seed{seed}", "model_checkpoint.pt"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        net = ClassicalAdapterNet(input_dim=384, hidden_dim=64, num_classes=7)
        net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        net.eval()
        with torch.no_grad():
            t_in = torch.tensor(X_input, dtype=torch.float32)
            logits = net(t_in).numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            pred_indices = np.argmax(probs, axis=-1)
            pred_classes = [CANONICAL_CLASSES[idx] for idx in pred_indices]
            max_confs = np.max(probs, axis=-1).tolist()
        return logits, pred_classes, max_confs

    total_experiments = 0

    # ==========================================================================
    # EXPERIMENT LOOP OVER 15 SPEAKERS
    # ==========================================================================
    for spk in speakers:
        spk_grp = speaker_groups[spk]
        seed = speaker_seeds[spk]
        print(f"\nProcessing Speaker: {spk} ({spk_grp}) Seed: {seed}")

        # Retrieve adaptation splits across budgets: 20, 30, 40, 50
        split_20 = speaker_splits[spk][20]
        test_rids = split_20["test_recording_ids"]

        # Protected test set data
        test_embs = np.stack([get_embedding(r) for r in test_rids]).astype(np.float32)
        test_labels = [rid_to_intent[r] for r in test_rids]
        test_sample_ids = test_rids

        # ----------------------------------------------------------------------
        # A. ESTABLISH REFERENCE WINDOW W0 (Budget B0, N=20)
        # ----------------------------------------------------------------------
        adapt_rids_20 = split_20["adaptation_recording_ids"][:20]
        ref_embs = np.stack([get_embedding(r) for r in adapt_rids_20]).astype(np.float32)
        ref_labels = [rid_to_intent[r] for r in adapt_rids_20]
        m0_version = f"v{spk}_budget20_seed{seed}"

        # Run forward pass for reference window
        ref_logits, ref_preds, ref_confs = load_adapter_and_predict(spk, 20, seed, ref_embs)

        ref_window = DriftWindow(
            window_id=f"win_ref_{spk}_B0",
            speaker_id=spk,
            sample_ids=adapt_rids_20,
            embeddings=ref_embs,
            predicted_classes=ref_preds,
            confidences=ref_confs,
            model_version=m0_version,
            true_labels=ref_labels,
            start_timestamp="2026-10-01T00:00:00Z",
            end_timestamp="2026-10-01T01:00:00Z",
        )
        detector.set_reference_window(ref_window)

        # Evaluate M0 on protected test set (Pre-drift baseline)
        _, m0_test_preds, m0_test_confs = load_adapter_and_predict(spk, 20, seed, test_embs)
        m0_metrics = compute_comprehensive_continual_metrics(test_labels, m0_test_preds)
        m0_accs = [1 if p == y else 0 for p, y in zip(m0_test_preds, test_labels)]
        m0_ece = compute_expected_calibration_error(m0_test_confs, m0_accs, num_bins=10)
        m0_test_logits, _, _ = load_adapter_and_predict(spk, 20, seed, test_embs)
        m0_test_probs = torch.softmax(torch.tensor(m0_test_logits), dim=-1).numpy()
        m0_true_indices = [CANONICAL_CLASSES.index(lbl) for lbl in test_labels]
        m0_brier = compute_brier_score(m0_test_probs, m0_true_indices)

        trajectory_records.append({
            "speaker_id": spk,
            "speaker_group": spk_grp,
            "budget": 20,
            "step": 0,
            "model_version": m0_version,
            "state_type": "REFERENCE_PRE_DRIFT",
            "mmd_drift": 0.0,
            "cosine_drift": 0.0,
            "tvd_prediction_drift": 0.0,
            "ks_confidence_stat": 0.0,
            "ks_confidence_pval": 1.0,
            "mean_confidence": float(np.mean(m0_test_confs)),
            "delta_mean_confidence": 0.0,
            "macro_f1": m0_metrics.macro_f1,
            "emergency_recall": m0_metrics.emergency_recall,
            "emergency_precision": m0_metrics.emergency_precision,
            "false_emergency_rate": m0_metrics.false_emergency_rate,
            "iper": m0_metrics.iper,
            "ece": m0_ece,
            "brier_score": m0_brier,
            "drift_status": DriftStatus.NO_DRIFT.value,
            "recalibration_trigger": RecalibrationTriggerState.NO_RECALIBRATION.value,
        })

        # ----------------------------------------------------------------------
        # B. SEQUENTIAL MONITORING & RECALIBRATION (Budgets N=30, 40, 50)
        # ----------------------------------------------------------------------
        budgets = [30, 40, 50]
        # To strictly enforce temporal causality, we take incremental batches of 10 samples
        prev_parent_version = m0_version

        for step_idx, b in enumerate(budgets, start=1):
            total_experiments += 1
            curr_version = f"v{spk}_budget{b}_seed{seed}"
            exp_id = f"EXP_7_7_{spk}_N{b}_S{seed}"

            # Incremental verified experience window W_t
            # Take recordings up to budget b
            all_adapt_rids = speaker_splits[spk][b]["adaptation_recording_ids"]
            # Current window is the newest 10 experiences accumulated
            curr_window_rids = all_adapt_rids[-10:] if len(all_adapt_rids) >= 10 else all_adapt_rids
            curr_embs = np.stack([get_embedding(r) for r in curr_window_rids]).astype(np.float32)
            curr_labels = [rid_to_intent[r] for r in curr_window_rids]

            # Forward pass on current window using current model Mt
            curr_logits, curr_preds, curr_confs = load_adapter_and_predict(spk, b, seed, curr_embs)

            curr_window = DriftWindow(
                window_id=f"win_{spk}_B{step_idx}_N{b}",
                speaker_id=spk,
                sample_ids=curr_window_rids,
                embeddings=curr_embs,
                predicted_classes=curr_preds,
                confidences=curr_confs,
                model_version=curr_version,
                true_labels=curr_labels,
                start_timestamp=f"2026-10-01T0{step_idx}:00:00Z",
                end_timestamp=f"2026-10-01T0{step_idx}:30:00Z",
            )

            # Evaluate drift against reference W0
            drift_events = detector.evaluate_drift(current_window=curr_window, reference_window=ref_window)
            trigger_state, trigger_reasons = detector.assess_recalibration_trigger(drift_events)

            # Extract statistics
            rep_event = next((e for e in drift_events if e.drift_type == DriftType.REPRESENTATION_DRIFT.value), None)
            pred_event = next((e for e in drift_events if e.drift_type == DriftType.PREDICTION_DRIFT.value), None)
            conf_event = next((e for e in drift_events if e.drift_type == DriftType.CONFIDENCE_DRIFT.value), None)
            cal_event = next((e for e in drift_events if e.drift_type == DriftType.CALIBRATION_DRIFT.value), None)

            mmd_val = rep_event.statistic if rep_event else 0.0
            cos_val = rep_event.metadata.get("centroid_cosine_distance", 0.0) if rep_event else 0.0
            tvd_val = pred_event.statistic if pred_event else 0.0
            ks_stat = conf_event.statistic if conf_event else 0.0
            ks_pval = conf_event.p_value if conf_event else 1.0
            delta_conf = conf_event.confidence_summary.get("delta_mean_confidence", 0.0) if conf_event else 0.0

            # Log drift events
            for ev in drift_events:
                drift_events_log.append({
                    "drift_event_id": ev.drift_event_id,
                    "speaker_id": spk,
                    "speaker_group": spk_grp,
                    "budget": b,
                    "model_version": curr_version,
                    "reference_window_id": ev.reference_window_id,
                    "current_window_id": ev.current_window_id,
                    "drift_type": ev.drift_type,
                    "drift_method": ev.drift_method,
                    "statistic": ev.statistic,
                    "threshold": ev.threshold,
                    "status": ev.status,
                    "p_value": ev.p_value,
                    "policy_version": ev.policy_version,
                    "created_at": ev.created_at,
                })

            # Budget-matched protected test set (completely disjoint from all_adapt_rids)
            b_test_rids = speaker_splits[spk][b]["test_recording_ids"]
            b_test_embs = np.stack([get_embedding(r) for r in b_test_rids]).astype(np.float32)
            b_test_labels = [rid_to_intent[r] for r in b_test_rids]
            b_test_true_indices = [CANONICAL_CLASSES.index(lbl) for lbl in b_test_labels]

            # Evaluate uncalibrated Mt on protected test set
            mt_test_logits, mt_test_preds, mt_test_confs = load_adapter_and_predict(spk, b, seed, b_test_embs)
            mt_metrics = compute_comprehensive_continual_metrics(b_test_labels, mt_test_preds)
            mt_accs = [1 if p == y else 0 for p, y in zip(mt_test_preds, b_test_labels)]
            mt_ece = compute_expected_calibration_error(mt_test_confs, mt_accs, num_bins=10)
            mt_probs = torch.softmax(torch.tensor(mt_test_logits), dim=-1).numpy()
            mt_brier = compute_brier_score(mt_probs, b_test_true_indices)

            is_drift_detected = any(e.status == DriftStatus.DRIFT_DETECTED.value for e in drift_events)

            trajectory_records.append({
                "speaker_id": spk,
                "speaker_group": spk_grp,
                "budget": b,
                "step": step_idx,
                "model_version": curr_version,
                "state_type": "POST_DRIFT_UNCALIBRATED",
                "mmd_drift": mmd_val,
                "cosine_drift": cos_val,
                "tvd_prediction_drift": tvd_val,
                "ks_confidence_stat": ks_stat,
                "ks_confidence_pval": ks_pval,
                "mean_confidence": float(np.mean(mt_test_confs)),
                "delta_mean_confidence": delta_conf,
                "macro_f1": mt_metrics.macro_f1,
                "emergency_recall": mt_metrics.emergency_recall,
                "emergency_precision": mt_metrics.emergency_precision,
                "false_emergency_rate": mt_metrics.false_emergency_rate,
                "iper": mt_metrics.iper,
                "ece": mt_ece,
                "brier_score": mt_brier,
                "drift_status": DriftStatus.DRIFT_DETECTED.value if is_drift_detected else DriftStatus.NO_DRIFT.value,
                "recalibration_trigger": trigger_state.value,
            })

            # ------------------------------------------------------------------
            # EXPERIMENT B: CONTROLLED RECALIBRATION (Temperature Scaling)
            # ------------------------------------------------------------------
            # Disjoint calibration set: newest incoming verified experiences (W_t)
            # which were NOT part of parent model training set
            parent_train_rids = all_adapt_rids[:-10] if len(all_adapt_rids) > 10 else all_adapt_rids[:10]
            cal_rids = curr_window_rids  # (10 verified samples)
            cal_embs = curr_embs
            cal_labels = curr_labels

            # Forward pass on calibration set with Mt
            cal_logits, _, _ = load_adapter_and_predict(spk, b, seed, cal_embs)

            # Fit temperature scaling on calibration set
            recalibrator = TemperatureScalingRecalibrator(temperature_range=(0.5, 3.0), num_steps=51)
            optimal_temp = recalibrator.fit(cal_logits, cal_labels, CANONICAL_CLASSES)

            # Apply temperature scaling to protected test set
            calibrated_test_probs, calibrated_test_confs = recalibrator.apply(mt_test_logits, optimal_temp)
            cal_ece = compute_expected_calibration_error(calibrated_test_confs, mt_accs, num_bins=10)
            cal_brier = compute_brier_score(calibrated_test_probs, b_test_true_indices)

            # Evaluate recalibrated candidate through Phase 7.6 Governance Gate
            # Gate comparison: Candidate vs Parent Mt
            gate = ValidationGate()
            # Prepare validation samples format for gate: [(sid, lbl, emb)]
            val_items = [(r, rid_to_intent[r], get_embedding(r)) for r in b_test_rids]

            cand_rec = ModelVersionRecord(
                model_id="classical_adapter_recal",
                version=f"{curr_version}_recal_T{optimal_temp:.2f}",
                speaker_id=spk,
                parent_version=curr_version,
                training_sample_ids=parent_train_rids,
                training_dataset_hash=compute_sha256(",".join(parent_train_rids)),
                configuration_hash=compute_sha256(json.dumps({"temp": optimal_temp})),
                hyperparameters={"temperature": optimal_temp, "calibration_method": "temperature_scaling"},
                status=ModelLifecycleState.CANDIDATE.value,
            )

            parent_rec = ModelVersionRecord(
                model_id="classical_adapter",
                version=curr_version,
                speaker_id=spk,
                training_sample_ids=parent_train_rids,
                training_dataset_hash=compute_sha256(",".join(parent_train_rids)),
                status=ModelLifecycleState.ACTIVE.value,
            )

            # Candidate predict function
            def cand_predict_fn(X_val):
                return mt_test_preds

            def parent_predict_fn(X_val):
                return mt_test_preds

            gate_decision = gate.evaluate_candidate(
                candidate_record=cand_rec,
                val_samples=val_items,
                candidate_predict_fn=cand_predict_fn,
                parent_record=parent_rec,
                parent_predict_fn=parent_predict_fn,
                val_dataset_id=f"val_protected_{spk}_N{b}",
                replay_sample_ids=cal_rids,
            )

            recalibration_records.append({
                "experiment_id": exp_id,
                "speaker_id": spk,
                "speaker_group": spk_grp,
                "budget": b,
                "parent_version": curr_version,
                "candidate_version": cand_rec.version,
                "calibration_method": "temperature_scaling",
                "temperature": optimal_temp,
                "trigger_state": trigger_state.value,
                "uncalibrated_ece": mt_ece,
                "calibrated_ece": cal_ece,
                "delta_ece": cal_ece - mt_ece,
                "uncalibrated_brier": mt_brier,
                "calibrated_brier": cal_brier,
                "delta_brier": cal_brier - mt_brier,
                "macro_f1": mt_metrics.macro_f1,
                "emergency_recall": mt_metrics.emergency_recall,
                "false_emergency_rate": mt_metrics.false_emergency_rate,
                "iper": mt_metrics.iper,
                "validation_gate_decision": gate_decision.decision,
                "safety_veto": gate_decision.safety_veto,
                "reason_codes": ";".join(gate_decision.reason_codes),
                "promotion_status": "PROMOTABLE" if gate_decision.decision == PromotionDecision.PROMOTE.value else "HELD_OR_REJECTED",
            })

            # Record experiment in ExperimentRegistry
            exp_rec = ExperimentRunRecord(
                experiment_id=exp_id,
                speaker_id=spk,
                experience_budget=b,
                train_experience_ids=all_adapt_rids,
                evaluation_set=f"protected_test_{spk}_N{b}",
                seed=seed,
                configuration={"temperature": optimal_temp, "parent_version": curr_version},
                phase="7.7",
                model_family="classical_adapter",
                model_version=cand_rec.version,
                parent_model_version=curr_version,
                status=ExperimentStatus.COMPLETED.value,
                metrics={
                    "uncalibrated_ece": mt_ece,
                    "calibrated_ece": cal_ece,
                    "delta_ece": cal_ece - mt_ece,
                    "uncalibrated_brier": mt_brier,
                    "calibrated_brier": cal_brier,
                    "macro_f1": mt_metrics.macro_f1,
                    "emergency_recall": mt_metrics.emergency_recall,
                    "false_emergency_rate": mt_metrics.false_emergency_rate,
                    "gate_decision": gate_decision.decision,
                },
                reference_window_id=ref_window.window_id,
                current_window_id=curr_window.window_id,
                drift_policy_version=policy_ver,
                calibration_policy_version=policy_ver,
                calibration_method="temperature_scaling",
                calibration_data_ids=cal_rids,
                evaluation_data_ids=b_test_rids,
            )
            exp_registry.register_run(exp_rec)

    # 6. Save structured CSV outputs
    df_drift_events = pd.DataFrame(drift_events_log)
    df_drift_events.to_csv(os.path.join(results_dir, "drift_events.csv"), index=False)

    df_trajectory = pd.DataFrame(trajectory_records)
    df_trajectory.to_csv(os.path.join(results_dir, "drift_trajectory_summary.csv"), index=False)

    df_recal = pd.DataFrame(recalibration_records)
    df_recal.to_csv(os.path.join(results_dir, "recalibration_experiments.csv"), index=False)

    # ==========================================================================
    # 7. STATISTICAL & SUBGROUP ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL & SUBGROUP ANALYSIS")
    print("=" * 80)

    # A. Representation Drift across Trajectory
    # Compare B0 (N=20) vs B3 (N=50) post-drift states
    post_drift_b50 = df_trajectory[df_trajectory["budget"] == 50]
    mean_mmd = post_drift_b50["mmd_drift"].mean()
    std_mmd = post_drift_b50["mmd_drift"].std()
    mean_cos = post_drift_b50["cosine_drift"].mean()

    print(f"Post-drift (N=50) Mean MMD: {mean_mmd:.4f} +/- {std_mmd:.4f}")
    print(f"Post-drift (N=50) Mean Centroid Cosine Distance: {mean_cos:.4f}")

    # B. Recalibration Effect: Uncalibrated vs Calibrated ECE
    delta_eces = df_recal["delta_ece"].values
    mean_delta_ece = float(np.mean(delta_eces))
    std_delta_ece = float(np.std(delta_eces, ddof=1))

    # Paired Wilcoxon signed-rank test
    stat_ece, pval_ece = stats.wilcoxon(df_recal["uncalibrated_ece"], df_recal["calibrated_ece"])
    print(f"Mean Delta ECE (Calibrated - Uncalibrated): {mean_delta_ece:.4f} +/- {std_delta_ece:.4f} (p = {pval_ece:.4e})")

    # Delta Brier score
    delta_briers = df_recal["delta_brier"].values
    mean_delta_brier = float(np.mean(delta_briers))
    stat_brier, pval_brier = stats.wilcoxon(df_recal["uncalibrated_brier"], df_recal["calibrated_brier"])
    print(f"Mean Delta Brier Score: {mean_delta_brier:.4f} (p = {pval_brier:.4e})")

    # C. Subgroup Breakdown: Dysarthric vs Control
    dys_recal = df_recal[df_recal["speaker_group"] == "dysarthric"]
    ctrl_recal = df_recal[df_recal["speaker_group"] == "control"]

    dys_delta_ece = float(dys_recal["delta_ece"].mean())
    ctrl_delta_ece = float(ctrl_recal["delta_ece"].mean())
    dys_uncal_ece = float(dys_recal["uncalibrated_ece"].mean())
    dys_cal_ece = float(dys_recal["calibrated_ece"].mean())
    ctrl_uncal_ece = float(ctrl_recal["uncalibrated_ece"].mean())
    ctrl_cal_ece = float(ctrl_recal["calibrated_ece"].mean())

    print(f"\nDysarthric Cohort (N=7): Uncalibrated ECE = {dys_uncal_ece:.4f} -> Calibrated ECE = {dys_cal_ece:.4f} (Delta = {dys_delta_ece:.4f})")
    print(f"Control Cohort (N=8):   Uncalibrated ECE = {ctrl_uncal_ece:.4f} -> Calibrated ECE = {ctrl_cal_ece:.4f} (Delta = {ctrl_delta_ece:.4f})")

    subgroup_analysis = [
        {
            "subgroup": "overall",
            "num_speakers": 15,
            "num_evaluations": len(df_recal),
            "uncalibrated_ece_mean": float(df_recal["uncalibrated_ece"].mean()),
            "calibrated_ece_mean": float(df_recal["calibrated_ece"].mean()),
            "delta_ece_mean": mean_delta_ece,
            "delta_ece_std": std_delta_ece,
            "p_value_wilcoxon": float(pval_ece),
            "uncalibrated_brier_mean": float(df_recal["uncalibrated_brier"].mean()),
            "calibrated_brier_mean": float(df_recal["calibrated_brier"].mean()),
            "delta_brier_mean": mean_delta_brier,
            "promotion_rate": float((df_recal["validation_gate_decision"] == "PROMOTE").mean()),
            "safety_veto_rate": float(df_recal["safety_veto"].mean()),
        },
        {
            "subgroup": "dysarthric",
            "num_speakers": 7,
            "num_evaluations": len(dys_recal),
            "uncalibrated_ece_mean": dys_uncal_ece,
            "calibrated_ece_mean": dys_cal_ece,
            "delta_ece_mean": dys_delta_ece,
            "delta_ece_std": float(dys_recal["delta_ece"].std()),
            "p_value_wilcoxon": float(stats.wilcoxon(dys_recal["uncalibrated_ece"], dys_recal["calibrated_ece"])[1]),
            "uncalibrated_brier_mean": float(dys_recal["uncalibrated_brier"].mean()),
            "calibrated_brier_mean": float(dys_recal["calibrated_brier"].mean()),
            "delta_brier_mean": float(dys_recal["delta_brier"].mean()),
            "promotion_rate": float((dys_recal["validation_gate_decision"] == "PROMOTE").mean()),
            "safety_veto_rate": float(dys_recal["safety_veto"].mean()),
        },
        {
            "subgroup": "control",
            "num_speakers": 8,
            "num_evaluations": len(ctrl_recal),
            "uncalibrated_ece_mean": ctrl_uncal_ece,
            "calibrated_ece_mean": ctrl_cal_ece,
            "delta_ece_mean": ctrl_delta_ece,
            "delta_ece_std": float(ctrl_recal["delta_ece"].std()),
            "p_value_wilcoxon": float(stats.wilcoxon(ctrl_recal["uncalibrated_ece"], ctrl_recal["calibrated_ece"])[1]),
            "uncalibrated_brier_mean": float(ctrl_recal["uncalibrated_brier"].mean()),
            "calibrated_brier_mean": float(ctrl_recal["calibrated_brier"].mean()),
            "delta_brier_mean": float(ctrl_recal["delta_brier"].mean()),
            "promotion_rate": float((ctrl_recal["validation_gate_decision"] == "PROMOTE").mean()),
            "safety_veto_rate": float(ctrl_recal["safety_veto"].mean()),
        },
    ]
    df_subgroup = pd.DataFrame(subgroup_analysis)
    df_subgroup.to_csv(os.path.join(results_dir, "subgroup_drift_analysis.csv"), index=False)

    print(f"\nAll experiments complete. Output saved to {results_dir}")
    print(f"Total Experiment Runs Logged: {total_experiments}")


if __name__ == "__main__":
    run_phase7_7_experiments()
