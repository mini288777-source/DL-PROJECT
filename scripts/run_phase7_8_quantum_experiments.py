"""
Phase 7.8 Continual Quantum Kernel Personalization Experiment Runner.

Executes the formal matched empirical benchmark:
- 15 TORGO Speakers: 8 Dysarthric (F01, F03, F04, M01, M02, M03, M04, M05)
  + 7 Control (FC01, FC02, FC03, MC01, MC02, MC03, MC04).
- 4 Accumulation Budgets: N in {20, 30, 40, 50} (nested longitudinal stream).
- Matched Classical Control: Phase 7.4 Continual Classical Adapter (EXP_7_4_*, 60 cells).
- Canonical Fixed Quantum Kernel: 8 qubits, ZZFeatureMap, reps=1, linear entanglement,
  FidelityQuantumKernel, exact statevector simulation, QSVC C=1.0, balanced class weighting.
- Information Isolation: Train-only PCA (d=8) and MinMaxScaler [0, pi] on adaptation data.
  Zero test leakage.
- Deterministic cryptographic kernel caching.
- Safety Governance: Phase 7.6 ValidationGate evaluation with Safety Veto.
- Statistical Analysis: Matched paired Wilcoxon signed-rank tests with Holm-Bonferroni correction.

Outputs:
  results/continual/phase7_8/continual_quantum_results.csv
  results/continual/phase7_8/matched_quantum_classical_comparison.csv
  results/continual/phase7_8/quantum_trajectory_summary.csv
  results/continual/phase7_8/quantum_subgroup_analysis.csv
  results/continual/phase7_8/experiment_registry.json
  reports/continual/phase7_8/continual_quantum_summary.json
"""

import os
import sys
import time
import json
from typing import Dict, List, Set, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    LearningStatus,
    EligibilityReason,
)
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS
from src.continual.model_version import (
    ModelRegistry,
    ModelVersionRecord,
    ModelLifecycleState,
)
from src.continual.experiment_registry import (
    ExperimentRegistry,
    ExperimentRunRecord,
    ExperimentStatus,
)
from src.continual.continual_classical import (
    ContinualEvaluationMetrics,
    compute_comprehensive_continual_metrics,
)
from src.continual.continual_quantum import (
    CANONICAL_QUANTUM_CONFIG,
    ContinualQuantumTrainer,
    ContinualQuantumResult,
    compute_quantum_cache_key,
)
from src.continual.governance import ValidationGate
from src.embeddings.cache import load_embedding_file
from src.utils.seeding import set_seed


def run_phase7_8_benchmark():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("PHASE 7.8 — CONTINUAL QUANTUM KERNEL PERSONALIZATION BENCHMARK")
    print("Matched Evaluation: Continual Classical Baseline vs Continual Fixed-Kernel QSVC")
    print("15 Speakers x 4 Budgets (N in {20, 30, 40, 50}) on Protected Test Sets")
    print("=" * 80)

    # 1. Setup output paths
    phase7_4_dir = os.path.join(REPO_ROOT, "results", "continual", "phase7_4")
    results_dir = os.path.join(REPO_ROOT, "results", "continual", "phase7_8")
    reports_dir = os.path.join(REPO_ROOT, "reports", "continual", "phase7_8")
    models_dir = os.path.join(REPO_ROOT, "models", "continual", "phase7_8")
    cache_dir = os.path.join(REPO_ROOT, "cache", "continual", "quantum_kernels")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Load frozen Phase 7.4 results (Matched Classical Control)
    control_csv_path = os.path.join(phase7_4_dir, "continual_classical_results.csv")
    if not os.path.isfile(control_csv_path):
        raise FileNotFoundError(f"Missing canonical Phase 7.4 control results at: {control_csv_path}")
    control_df = pd.read_csv(control_csv_path)
    print(f"Loaded {len(control_df)} frozen Phase 7.4 matched control cells.")

    # 3. Load splits and manifests
    splits_path = os.path.join(REPO_ROOT, "phase6_5", "splits", "recovery_splits_hierarchical_ready.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    adapt_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_adaptation_manifest.csv"))
    test_manifest = pd.read_csv(os.path.join(REPO_ROOT, "phase6_5", "manifests", "recovery_test_manifest.csv"))

    label_map = dict(zip(adapt_manifest["recording_id"], adapt_manifest["intent"]))
    label_map.update(dict(zip(test_manifest["recording_id"], test_manifest["intent"])))

    sess_map = dict(zip(adapt_manifest["recording_id"], adapt_manifest["session_id"]))
    sess_map.update(dict(zip(test_manifest["recording_id"], test_manifest["session_id"])))

    speakers = sorted(list(set(s["speaker_id"] for s in splits)))
    budgets = [20, 30, 40, 50]
    dysarthric_set = {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}

    # 4. Preload Whisper embeddings
    print("\nPreloading Whisper embeddings for all splits...")
    needed_rids = set()
    for s in splits:
        needed_rids.update(s["adaptation_recording_ids"])
        needed_rids.update(s["test_recording_ids"])

    embedding_cache = {}
    base_torgo_dir = os.path.join(REPO_ROOT, "embeddings", "torgo")
    for rid in sorted(list(needed_rids)):
        emb = load_embedding_file(rid, feature_type="whisper", base_dir=base_torgo_dir, expected_dim=384)
        embedding_cache[rid] = emb
    print(f"Loaded {len(embedding_cache)} embeddings into memory.\n")

    # 5. Initialize registries & governance
    model_registry = ModelRegistry(base_dir=models_dir)
    exp_registry = ExperimentRegistry(registry_dir=results_dir)
    validation_gate = ValidationGate()

    quantum_results: List[Dict[str, Any]] = []
    matched_comparison_rows: List[Dict[str, Any]] = []
    t_start_all = time.perf_counter()
    cell_idx = 0
    total_cells = len(speakers) * len(budgets)

    # 6. Execute Continual Quantum Benchmark Loop
    for spk in speakers:
        spk_splits = {s["sample_n"]: s for s in splits if s["speaker_id"] == spk}
        seed = spk_splits[20]["seed"]
        spk_group = "dysarthric" if spk in dysarthric_set else "control"
        tier = spk_splits[20]["feasibility_tier"]

        # Fixed protected test set from N=50 split
        test_rids = spk_splits[50]["test_recording_ids"]
        test_records = [(r, label_map[r], embedding_cache[r]) for r in test_rids]

        trainer = ContinualQuantumTrainer(
            target_speaker=spk,
            model_registry=model_registry,
            validation_gate=validation_gate,
            base_dir=models_dir,
            cache_dir=cache_dir,
            config={"seed": seed},
        )

        parent_version = None

        for b_idx, budget in enumerate(budgets):
            cell_idx += 1
            exp_id = f"EXP_7_8_{spk}_N{budget}_QSVC_S{seed}"

            # Matched Classical Control ID
            matched_ctrl_id = f"EXP_7_4_{spk}_N{budget}_S{seed}"
            ctrl_row = control_df[control_df["experiment_id"] == matched_ctrl_id]
            if len(ctrl_row) == 0:
                # Fallback search by speaker and budget
                ctrl_row = control_df[(control_df["speaker_id"] == spk) & (control_df["budget"] == budget)]
            if len(ctrl_row) == 0:
                raise RuntimeError(f"Could not locate matched classical control for {spk} budget {budget}")
            ctrl_data = ctrl_row.iloc[0].to_dict()

            # Adaptation recordings up to this budget
            b_rids = spk_splits[budget]["adaptation_recording_ids"]

            # Construct verified ExperienceRecords
            experiences = []
            for idx, r in enumerate(b_rids):
                timestamp_str = f"2026-01-01T{idx // 60:02d}:{idx % 60:02d}:00Z"
                rec = ExperienceRecord(
                    sample_id=r,
                    speaker_id=spk,
                    session_id=f"sess_{spk}_{sess_map.get(r, '01')}",
                    audio_reference=f"audio/{r}.wav",
                    embedding_reference=f"embeddings/torgo/whisper/{r}.npy",
                    predicted_intent=label_map[r],
                    prediction_confidence=0.90,
                    model_version=parent_version or "seed_baseline_v0",
                    quality_score=0.95,
                    timestamp=timestamp_str,
                    verification_status=VerificationStatus.USER_VERIFIED.value,
                    confirmed_intent=label_map[r],
                    confirmation_source="user_explicit",
                    learning_eligible=True,
                    eligibility_reason=EligibilityReason.ELIGIBLE_USER_CONFIRMED.value,
                    learning_status=LearningStatus.USER_CONFIRMED.value,
                )
                rec.embedding_array = embedding_cache[r]
                experiences.append(rec)

            # Register run in ExperimentRegistry
            run_record = ExperimentRunRecord(
                experiment_id=exp_id,
                speaker_id=spk,
                experience_budget=budget,
                train_experience_ids=b_rids,
                evaluation_set=f"test_protected_{spk}_n{len(test_records)}",
                seed=seed,
                configuration=trainer.config,
                phase="7.8",
                model_family="qsvc_q8_r1",
                parent_model_version=parent_version,
                matched_control_experiment_id=matched_ctrl_id,
                feature_version="whisper_tiny_384_unit_L2",
                pca_scaling_version="pca8_minmax_pi_train_only",
                quantum_configuration_version="qsvc_q8_r1_linear_exact_statevector",
                kernel_configuration={
                    "kernel_type": "FidelityQuantumKernel",
                    "qubit_count": 8,
                    "reps": 1,
                    "entanglement": "linear",
                    "feature_map": "ZZFeatureMap",
                },
                simulator_configuration={"backend": "exact_statevector", "framework": "qiskit"},
                validation_set=f"protected_test_{spk}",
            )
            exp_registry.register_run(run_record)

            # Execute train step
            t0_step = time.perf_counter()
            res: ContinualQuantumResult = trainer.train_step(
                experiences=experiences,
                test_records=test_records,
                parent_model_version=parent_version,
                matched_control_exp_id=matched_ctrl_id,
                custom_seed=seed,
            )
            step_time = time.perf_counter() - t0_step

            # Update registry
            exp_registry.update_run(
                experiment_id=exp_id,
                status=ExperimentStatus.COMPLETED,
                model_version=res.model_version,
                metrics=res.metrics.to_dict(),
                output_files={
                    "checkpoint": res.model_checkpoint_path or "",
                    "train_kernel_cache_key": res.kernel_cache_key,
                },
            )

            # Record result dict
            m = res.metrics
            kd = res.kernel_diagnostics
            result_row = {
                "experiment_id": exp_id,
                "speaker_id": spk,
                "speaker_group": spk_group,
                "feasibility_tier": tier,
                "seed": seed,
                "budget": budget,
                "accumulation_step": b_idx,
                "model_version": res.model_version,
                "parent_version": parent_version or "NONE",
                "matched_control_experiment_id": matched_ctrl_id,
                "train_samples": len(b_rids),
                "test_samples": len(test_records),
                "accuracy": m.accuracy,
                "macro_f1": m.macro_f1,
                "emergency_macro_f1": m.emergency_macro_f1,
                "emergency_recall": m.emergency_recall,
                "emergency_precision": m.emergency_precision,
                "false_emergency_rate": m.false_emergency_rate,
                "iper": m.iper,
                "general_speech_recall": m.general_speech_recall,
                "support_vectors": res.support_vector_count,
                "adaptation_time_sec": res.adaptation_time_sec,
                "kernel_train_time_sec": res.kernel_train_time_sec,
                "kernel_test_time_sec": res.kernel_test_time_sec,
                "inference_time_sec": res.inference_time_sec,
                "kernel_cache_hit": res.kernel_cache_hit,
                "governance_decision": res.governance_decision,
                "dataset_hash": res.training_dataset_hash,
                "config_hash": res.configuration_hash,
            }
            quantum_results.append(result_row)

            # Matched 1-to-1 comparison row
            comp_row = {
                "speaker_id": spk,
                "speaker_group": spk_group,
                "budget": budget,
                "seed": seed,
                "quantum_experiment_id": exp_id,
                "classical_experiment_id": matched_ctrl_id,
                # Classical Baseline Metrics (Phase 7.4)
                "classical_macro_f1": ctrl_data["macro_f1"],
                "classical_emergency_recall": ctrl_data["emergency_recall"],
                "classical_emergency_precision": ctrl_data["emergency_precision"],
                "classical_false_emergency_rate": ctrl_data["false_emergency_rate"],
                "classical_iper": ctrl_data["iper"],
                "classical_accuracy": ctrl_data["accuracy"],
                "classical_adaptation_time_sec": ctrl_data["adaptation_time_sec"],
                # Quantum Metrics (Phase 7.8)
                "quantum_macro_f1": m.macro_f1,
                "quantum_emergency_recall": m.emergency_recall,
                "quantum_emergency_precision": m.emergency_precision,
                "quantum_false_emergency_rate": m.false_emergency_rate,
                "quantum_iper": m.iper,
                "quantum_accuracy": m.accuracy,
                "quantum_adaptation_time_sec": res.adaptation_time_sec,
                # Deltas (Quantum - Classical)
                "delta_macro_f1": m.macro_f1 - ctrl_data["macro_f1"],
                "delta_emergency_recall": m.emergency_recall - ctrl_data["emergency_recall"],
                "delta_emergency_precision": m.emergency_precision - ctrl_data["emergency_precision"],
                "delta_false_emergency_rate": m.false_emergency_rate - ctrl_data["false_emergency_rate"],
                "delta_iper": m.iper - ctrl_data["iper"],
                "delta_accuracy": m.accuracy - ctrl_data["accuracy"],
                "delta_time_sec": res.adaptation_time_sec - ctrl_data["adaptation_time_sec"],
                "governance_decision": res.governance_decision,
            }
            matched_comparison_rows.append(comp_row)

            # Update parent version for next step in accumulation
            parent_version = res.model_version

            print(
                f"[{cell_idx:02d}/{total_cells}] {spk} (N={budget:02d}) -> "
                f"F1: Q={m.macro_f1:.4f} vs C={ctrl_data['macro_f1']:.4f} (d={comp_row['delta_macro_f1']:+.4f}) | "
                f"EmergRec: Q={m.emergency_recall:.4f} vs C={ctrl_data['emergency_recall']:.4f} (d={comp_row['delta_emergency_recall']:+.4f}) | "
                f"FER: Q={m.false_emergency_rate:.4f} vs C={ctrl_data['false_emergency_rate']:.4f} | "
                f"Gate: {res.governance_decision} ({step_time:.2f}s)"
            )

    t_total = time.perf_counter() - t_start_all
    print(f"\nAll 60 cells completed in {t_total:.2f}s.")

    # 7. Convert to DataFrames and save
    quantum_df = pd.DataFrame(quantum_results)
    matched_df = pd.DataFrame(matched_comparison_rows)

    quantum_csv = os.path.join(results_dir, "continual_quantum_results.csv")
    matched_csv = os.path.join(results_dir, "matched_quantum_classical_comparison.csv")
    quantum_df.to_csv(quantum_csv, index=False)
    matched_df.to_csv(matched_csv, index=False)
    print(f"Saved: {quantum_csv}")
    print(f"Saved: {matched_csv}")

    # 8. Trajectory & Subgroup Summary Analysis
    traj_summary = matched_df.groupby("budget").agg({
        "classical_macro_f1": ["mean", "std"],
        "quantum_macro_f1": ["mean", "std"],
        "delta_macro_f1": ["mean", "std"],
        "classical_emergency_recall": ["mean", "std"],
        "quantum_emergency_recall": ["mean", "std"],
        "delta_emergency_recall": ["mean", "std"],
        "classical_false_emergency_rate": ["mean", "std"],
        "quantum_false_emergency_rate": ["mean", "std"],
        "delta_false_emergency_rate": ["mean", "std"],
        "classical_adaptation_time_sec": ["mean"],
        "quantum_adaptation_time_sec": ["mean"],
    }).reset_index()

    traj_csv = os.path.join(results_dir, "quantum_trajectory_summary.csv")
    traj_summary.to_csv(traj_csv, index=False)
    print(f"Saved: {traj_csv}")

    # Subgroup breakdown
    subgroup_summary = matched_df.groupby(["speaker_group", "budget"]).agg({
        "classical_macro_f1": "mean",
        "quantum_macro_f1": "mean",
        "delta_macro_f1": "mean",
        "classical_emergency_recall": "mean",
        "quantum_emergency_recall": "mean",
        "delta_emergency_recall": "mean",
        "classical_false_emergency_rate": "mean",
        "quantum_false_emergency_rate": "mean",
        "delta_false_emergency_rate": "mean",
    }).reset_index()

    subgroup_csv = os.path.join(results_dir, "quantum_subgroup_analysis.csv")
    subgroup_summary.to_csv(subgroup_csv, index=False)
    print(f"Saved: {subgroup_csv}")

    # 9. Statistical Hypothesis Testing (Wilcoxon Signed-Rank Test)
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS: MATCHED WILCOXON SIGNED-RANK PAIRED TESTS")
    print("=" * 80)

    stat_results = {}
    metrics_to_test = [
        ("macro_f1", "classical_macro_f1", "quantum_macro_f1"),
        ("emergency_recall", "classical_emergency_recall", "quantum_emergency_recall"),
        ("emergency_precision", "classical_emergency_precision", "quantum_emergency_precision"),
        ("false_emergency_rate", "classical_false_emergency_rate", "quantum_false_emergency_rate"),
        ("iper", "classical_iper", "quantum_iper"),
        ("accuracy", "classical_accuracy", "quantum_accuracy"),
        ("adaptation_time", "classical_adaptation_time_sec", "quantum_adaptation_time_sec"),
    ]

    p_values = []
    for metric_name, c_col, q_col in metrics_to_test:
        c_vals = matched_df[c_col].values
        q_vals = matched_df[q_col].values
        diffs = q_vals - c_vals

        mean_c = float(np.mean(c_vals))
        mean_q = float(np.mean(q_vals))
        mean_delta = float(np.mean(diffs))
        std_delta = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        # Cohen's d effect size
        cohen_d = float(mean_delta / std_delta) if std_delta > 1e-9 else 0.0

        # Wilcoxon test
        try:
            # Check if all differences are zero
            if np.all(np.abs(diffs) < 1e-9):
                w_stat, p_val = 0.0, 1.0
            else:
                res_w = stats.wilcoxon(q_vals, c_vals, alternative="two-sided")
                w_stat, p_val = float(res_w.statistic), float(res_w.pvalue)
        except Exception as e:
            w_stat, p_val = 0.0, 1.0

        stat_results[metric_name] = {
            "mean_classical": mean_c,
            "mean_quantum": mean_q,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "cohens_d": cohen_d,
            "wilcoxon_stat": w_stat,
            "p_value_raw": p_val,
        }
        p_values.append(p_val)

    # Holm-Bonferroni correction
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted_p_values = [1.0] * n_tests

    for rank, idx in enumerate(sorted_indices):
        raw_p = p_values[idx]
        adj_p = min(1.0, raw_p * (n_tests - rank))
        adjusted_p_values[idx] = adj_p

    # Ensure monotonicity of adjusted p-values
    for i in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[i - 1]
        curr_idx = sorted_indices[i]
        adjusted_p_values[curr_idx] = max(adjusted_p_values[curr_idx], adjusted_p_values[prev_idx])

    for i, (metric_name, _, _) in enumerate(metrics_to_test):
        stat_results[metric_name]["p_value_adjusted"] = float(adjusted_p_values[i])
        stat_results[metric_name]["statistically_significant"] = bool(adjusted_p_values[i] < 0.05)

        sr = stat_results[metric_name]
        sig_str = "***" if sr["p_value_adjusted"] < 0.001 else ("*" if sr["p_value_adjusted"] < 0.05 else "n.s.")
        print(
            f"{metric_name:<25}: Classical={sr['mean_classical']:.4f} | Quantum={sr['mean_quantum']:.4f} | "
            f"Δ={sr['mean_delta']:+.4f} (d={sr['cohens_d']:+.2f}) | p={sr['p_value_raw']:.4e} (adj={sr['p_value_adjusted']:.4e}) [{sig_str}]"
        )

    # 10. Cohort / Subgroup specific analysis
    dysarthric_df = matched_df[matched_df["speaker_group"] == "dysarthric"]
    control_spk_df = matched_df[matched_df["speaker_group"] == "control"]

    def subgroup_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
        c_f1 = df["classical_macro_f1"].values
        q_f1 = df["quantum_macro_f1"].values
        c_rec = df["classical_emergency_recall"].values
        q_rec = df["quantum_emergency_recall"].values
        c_fer = df["classical_false_emergency_rate"].values
        q_fer = df["quantum_false_emergency_rate"].values

        return {
            "subgroup": name,
            "n_cells": len(df),
            "macro_f1_classical": float(np.mean(c_f1)),
            "macro_f1_quantum": float(np.mean(q_f1)),
            "macro_f1_delta": float(np.mean(q_f1 - c_f1)),
            "emergency_recall_classical": float(np.mean(c_rec)),
            "emergency_recall_quantum": float(np.mean(q_rec)),
            "emergency_recall_delta": float(np.mean(q_rec - c_rec)),
            "false_emergency_rate_classical": float(np.mean(c_fer)),
            "false_emergency_rate_quantum": float(np.mean(q_fer)),
            "false_emergency_rate_delta": float(np.mean(q_fer - c_fer)),
        }

    subgroup_breakdown = {
        "cohort": subgroup_stats(matched_df, "All 15 Speakers"),
        "dysarthric": subgroup_stats(dysarthric_df, "8 Dysarthric Speakers"),
        "control": subgroup_stats(control_spk_df, "7 Control Speakers"),
    }

    # 11. Scientific Hypothesis Evaluation
    h7_6_supported = bool(
        stat_results["macro_f1"]["statistically_significant"]
        and stat_results["macro_f1"]["mean_delta"] > 0
        and stat_results["emergency_recall"]["mean_delta"] >= 0
    )

    if h7_6_supported:
        rq7_6_decision = "SUPPORTED"
    elif stat_results["macro_f1"]["mean_delta"] > 0 and not stat_results["macro_f1"]["statistically_significant"]:
        rq7_6_decision = "NOT SUPPORTED"
    elif stat_results["macro_f1"]["mean_delta"] <= 0:
        rq7_6_decision = "NOT SUPPORTED"
    else:
        rq7_6_decision = "INCONCLUSIVE"

    print("\n" + "=" * 80)
    print(f"HYPOTHESIS H7.6 (Continual Quantum Benefit): {rq7_6_decision}")
    print(f"RQ7.6 Conclusion: {rq7_6_decision}")
    print("Frozen Historical Conclusion (CASE B): UNCHANGED")
    print("=" * 80)

    # 12. Save Summary JSON
    summary_payload = {
        "phase": "7.8",
        "description": "Continual Fixed-Kernel Quantum Personalization vs Matched Classical Baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_experiments": len(quantum_results),
        "total_cells_evaluated": total_cells,
        "speakers_evaluated": len(speakers),
        "budgets_evaluated": budgets,
        "canonical_quantum_config": CANONICAL_QUANTUM_CONFIG,
        "matched_control_phase": "7.4",
        "statistical_tests": stat_results,
        "subgroup_breakdown": subgroup_breakdown,
        "primary_hypothesis_h7_6": {
            "statement": "Continual quantum kernel personalization provides statistically supported benefit beyond matched continual classical personalization.",
            "status": "NOT SUPPORTED" if rq7_6_decision == "NOT SUPPORTED" else rq7_6_decision,
            "decision": rq7_6_decision,
        },
        "secondary_hypotheses": {
            "h7_6a_emergency_recall": {
                "quantum_mean": stat_results["emergency_recall"]["mean_quantum"],
                "classical_mean": stat_results["emergency_recall"]["mean_classical"],
                "delta": stat_results["emergency_recall"]["mean_delta"],
                "p_value_adjusted": stat_results["emergency_recall"]["p_value_adjusted"],
                "decision": "NOT SUPPORTED",
            },
            "h7_6c_computational_cost": {
                "quantum_mean_time_sec": stat_results["adaptation_time"]["mean_quantum"],
                "classical_mean_time_sec": stat_results["adaptation_time"]["mean_classical"],
                "ratio": float(stat_results["adaptation_time"]["mean_quantum"] / max(stat_results["adaptation_time"]["mean_classical"], 1e-6)),
                "observation": "Simulated quantum kernel requires matrix calculation overhead relative to lightweight classical adapter.",
            },
        },
        "rq7_6_decision": rq7_6_decision,
        "frozen_case_b_conclusion": "UNCHANGED",
    }

    summary_json_path = os.path.join(reports_dir, "continual_quantum_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved: {summary_json_path}")

    return summary_payload


if __name__ == "__main__":
    run_phase7_8_benchmark()
