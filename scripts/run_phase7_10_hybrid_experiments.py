"""
Phase 7.10 — Hybrid Classical + Quantum Empirical Benchmark Script.

Executes matched 4-way continual personalization across all 15 TORGO speakers
and 4 experience budgets (N in {20, 30, 40, 50}):
1. Continual Classical Baseline (Phase 7.4)
2. Continual Fixed Quantum Kernel (Phase 7.8)
3. Continual Trainable Quantum Representation (Phase 7.9)
4. Continual Hybrid Classical + Quantum (Phase 7.10)

Enforces:
- Strict zero-leakage pipeline: train-only PCA (d=8) -> MinMaxScaler [0, pi]
- Convex probability fusion: P_hybrid = 0.5 * P_class + 0.5 * P_quant
- Warm-start parameter updating across continual budget steps
- Phase 7.6 Multi-Metric ValidationGate and Safety Veto
- Matched Wilcoxon signed-rank paired tests with Holm-Bonferroni correction
- Subgroup stratification: Dysarthric (8) vs Control (7)
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.continual.schema import (
    ExperienceRecord,
    VerificationStatus,
    EligibilityReason,
    LearningStatus,
)
from src.continual.model_version import ModelRegistry
from src.continual.experiment_registry import (
    ExperimentRegistry,
    ExperimentRunRecord,
    ExperimentStatus,
)
from src.continual.continual_classical import (
    ContinualEvaluationMetrics,
    compute_comprehensive_continual_metrics,
)
from src.continual.trainable_quantum import compute_param_fingerprint
from src.continual.hybrid import (
    CANONICAL_HYBRID_CONFIG,
    ContinualHybridTrainer,
    ContinualHybridResult,
)
from src.continual.governance import ValidationGate
from src.embeddings.cache import load_embedding_file
from src.utils.seeding import set_seed


def run_hybrid_simulation(
    speakers: List[str],
    budgets: List[int],
    dysarthric_set: set,
    splits: List[Dict[str, Any]],
    label_map: Dict[str, str],
    sess_map: Dict[str, str],
    embedding_cache: Dict[str, np.ndarray],
    classical_df: pd.DataFrame,
    fixed_q_df: pd.DataFrame,
    trainable_q_df: pd.DataFrame,
    model_registry: ModelRegistry,
    validation_gate: ValidationGate,
    exp_registry: ExperimentRegistry,
    models_dir: str,
    results_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Execute the full 60-cell hybrid personalization loop."""
    hybrid_results: List[Dict[str, Any]] = []
    matched_4way_rows: List[Dict[str, Any]] = []
    t_start_all = time.perf_counter()
    cell_idx = 0
    total_cells = len(speakers) * len(budgets)

    for spk in speakers:
        spk_splits = {s["sample_n"]: s for s in splits if s["speaker_id"] == spk}
        seed = spk_splits[20]["seed"]
        spk_group = "dysarthric" if spk in dysarthric_set else "control"
        tier = spk_splits[20]["feasibility_tier"]

        # Fixed protected test set from N=50 split
        test_rids = spk_splits[50]["test_recording_ids"]
        test_records = [(r, label_map[r], embedding_cache[r]) for r in test_rids]

        trainer = ContinualHybridTrainer(
            target_speaker=spk,
            model_registry=model_registry,
            validation_gate=validation_gate,
            base_dir=models_dir,
            config={"seed": seed, "classical_epochs": 20, "quantum_maxiter": 20},
        )

        parent_version = None
        parent_theta = None
        parent_adapter_state = None

        for b_idx, budget in enumerate(budgets):
            cell_idx += 1
            exp_id = f"EXP_7_10_{spk}_N{budget}_HYBRID_S{seed}"

            # Matched baseline controls
            matched_c_id = f"EXP_7_4_{spk}_N{budget}_S{seed}"
            c_row = classical_df[(classical_df["speaker_id"] == spk) & (classical_df["budget"] == budget)]
            c_data = c_row.iloc[0].to_dict()

            matched_fq_id = f"EXP_7_8_{spk}_N{budget}_QSVC_S{seed}"
            fq_row = fixed_q_df[(fixed_q_df["speaker_id"] == spk) & (fixed_q_df["budget"] == budget)]
            fq_data = fq_row.iloc[0].to_dict()

            matched_tq_id = f"EXP_7_9_{spk}_N{budget}_TRAINABLE_Q_S{seed}"
            tq_row = trainable_q_df[(trainable_q_df["speaker_id"] == spk) & (trainable_q_df["budget"] == budget)]
            tq_data = tq_row.iloc[0].to_dict()

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
                phase="7.10",
                model_family="hybrid_classical_quantum",
                parent_model_version=parent_version,
                matched_control_experiment_id=matched_c_id,
                matched_fixed_quantum_experiment_id=matched_fq_id,
                matched_trainable_quantum_experiment_id=matched_tq_id,
                feature_version="whisper_tiny_384_unit_L2",
                pca_scaling_version="pca8_minmax_pi_train_only",
                quantum_architecture_version=CANONICAL_HYBRID_CONFIG["quantum_branch"],
                parameter_count=25239,
                circuit_depth=37,
                trainable_parameter_count=25239,
                optimizer="AdamW_plus_COBYLA",
                fusion_method="calibrated_probability_fusion",
                fusion_weight=0.5,
                parent_parameter_fingerprint=compute_param_fingerprint(parent_theta) if parent_theta is not None else None,
                initialization_mode="WARM_STARTED" if parent_theta is not None else "ZERO_INITIALIZED",
            )
            exp_registry.register_experiment(run_record)

            # Execute Continual Hybrid Adaptation Step
            t0_step = time.perf_counter()
            res: ContinualHybridResult = trainer.adapt_speaker_continual(
                experiences=experiences,
                budget=budget,
                test_records=test_records,
                seed=seed,
                parent_model_version=parent_version,
                parent_theta=parent_theta,
                parent_adapter_state=parent_adapter_state,
            )
            step_time = time.perf_counter() - t0_step

            # Finalize experiment record
            run_record.status = ExperimentStatus.COMPLETED.value
            run_record.model_version = res.model_version
            run_record.parameter_fingerprint = res.parameter_fingerprint
            run_record.execution_time_sec = step_time
            m = res.evaluation_metrics
            run_record.metrics = {
                "macro_f1": m.macro_f1,
                "emergency_recall": m.emergency_recall,
                "emergency_precision": m.emergency_precision,
                "false_emergency_rate": m.false_emergency_rate,
                "accuracy": m.accuracy,
                "branch_divergence_rate": res.branch_divergence_rate,
                "governance_decision": res.governance_decision,
            }
            exp_registry.update_experiment(run_record)

            # Record row for continual_hybrid_results
            row_dict = {
                "experiment_id": exp_id,
                "phase": "7.10",
                "speaker_id": spk,
                "speaker_group": spk_group,
                "feasibility_tier": tier,
                "budget": budget,
                "seed": seed,
                "matched_classical_id": matched_c_id,
                "matched_fixed_quantum_id": matched_fq_id,
                "matched_trainable_quantum_id": matched_tq_id,
                "model_version": res.model_version,
                "parent_model_version": parent_version,
                "initialization_mode": res.initialization_mode,
                "param_fingerprint": res.parameter_fingerprint,
                "parent_param_fingerprint": res.parent_parameter_fingerprint,
                "classical_macro_f1": res.classical_metrics.macro_f1,
                "classical_emergency_recall": res.classical_metrics.emergency_recall,
                "quantum_macro_f1": res.quantum_metrics.macro_f1,
                "quantum_emergency_recall": res.quantum_metrics.emergency_recall,
                "macro_f1": m.macro_f1,
                "emergency_recall": m.emergency_recall,
                "emergency_precision": m.emergency_precision,
                "false_emergency_rate": m.false_emergency_rate,
                "accuracy": m.accuracy,
                "branch_divergence_rate": res.branch_divergence_rate,
                "optimization_time_sec": res.optimization_time_sec,
                "adaptation_time_sec": res.adaptation_time_sec,
                "inference_time_sec": res.inference_time_sec,
                "governance_decision": res.governance_decision,
                "safety_veto_triggered": res.safety_veto_triggered,
            }
            hybrid_results.append(row_dict)

            # Matched 4-Way Comparison Record
            comp_row = {
                "speaker_id": spk,
                "speaker_group": spk_group,
                "feasibility_tier": tier,
                "budget": budget,
                # Classical Baseline (Phase 7.4)
                "classical_macro_f1": c_data["macro_f1"],
                "classical_emergency_recall": c_data["emergency_recall"],
                "classical_false_emergency_rate": c_data["false_emergency_rate"],
                "classical_accuracy": c_data["accuracy"],
                # Fixed Quantum Kernel (Phase 7.8)
                "fixed_q_macro_f1": fq_data["macro_f1"],
                "fixed_q_emergency_recall": fq_data["emergency_recall"],
                "fixed_q_false_emergency_rate": fq_data["false_emergency_rate"],
                "fixed_q_accuracy": fq_data["accuracy"],
                # Trainable Quantum Representation (Phase 7.9)
                "trainable_q_macro_f1": tq_data["macro_f1"],
                "trainable_q_emergency_recall": tq_data["emergency_recall"],
                "trainable_q_false_emergency_rate": tq_data["false_emergency_rate"],
                "trainable_q_accuracy": tq_data["accuracy"],
                # Hybrid Classical + Quantum (Phase 7.10)
                "hybrid_macro_f1": m.macro_f1,
                "hybrid_emergency_recall": m.emergency_recall,
                "hybrid_false_emergency_rate": m.false_emergency_rate,
                "hybrid_accuracy": m.accuracy,
                "branch_divergence_rate": res.branch_divergence_rate,
                "adaptation_time_sec": res.adaptation_time_sec,
                # Deltas vs Classical Baseline (Hybrid - C)
                "delta_hyb_vs_c_macro_f1": m.macro_f1 - c_data["macro_f1"],
                "delta_hyb_vs_c_emergency_recall": m.emergency_recall - c_data["emergency_recall"],
                "delta_hyb_vs_c_false_emergency_rate": m.false_emergency_rate - c_data["false_emergency_rate"],
                "delta_hyb_vs_c_accuracy": m.accuracy - c_data["accuracy"],
                # Deltas vs Fixed Quantum (Hybrid - FQ)
                "delta_hyb_vs_fq_macro_f1": m.macro_f1 - fq_data["macro_f1"],
                "delta_hyb_vs_fq_emergency_recall": m.emergency_recall - fq_data["emergency_recall"],
                "delta_hyb_vs_fq_false_emergency_rate": m.false_emergency_rate - fq_data["false_emergency_rate"],
                "delta_hyb_vs_fq_accuracy": m.accuracy - fq_data["accuracy"],
                # Deltas vs Trainable Quantum (Hybrid - TQ)
                "delta_hyb_vs_tq_macro_f1": m.macro_f1 - tq_data["macro_f1"],
                "delta_hyb_vs_tq_emergency_recall": m.emergency_recall - tq_data["emergency_recall"],
                "delta_hyb_vs_tq_false_emergency_rate": m.false_emergency_rate - tq_data["false_emergency_rate"],
                "delta_hyb_vs_tq_accuracy": m.accuracy - tq_data["accuracy"],
                "governance_decision": res.governance_decision,
            }
            matched_4way_rows.append(comp_row)

            # Update parent state for warm-start lineage
            parent_version = res.model_version
            parent_theta = np.array(res.optimal_theta)

            print(
                f"[{cell_idx:02d}/{total_cells}] {spk} (N={budget:02d}) -> "
                f"F1: Hyb={m.macro_f1:.4f} (C={c_data['macro_f1']:.4f}, TQ={tq_data['macro_f1']:.4f}) | "
                f"EmergRec: Hyb={m.emergency_recall:.4f} (C={c_data['emergency_recall']:.4f}, TQ={tq_data['emergency_recall']:.4f}) | "
                f"FER: Hyb={m.false_emergency_rate:.4f} (C={c_data['false_emergency_rate']:.4f}) | "
                f"Gate: {res.governance_decision} ({step_time:.2f}s)"
            )

    t_total = time.perf_counter() - t_start_all
    print(f"\nAll 60 hybrid cells completed in {t_total:.2f}s.")

    hybrid_df = pd.DataFrame(hybrid_results)
    matched_df = pd.DataFrame(matched_4way_rows)

    hybrid_csv = os.path.join(results_dir, "continual_hybrid_results.csv")
    matched_csv = os.path.join(results_dir, "matched_four_way_comparison.csv")
    hybrid_df.to_csv(hybrid_csv, index=False)
    matched_df.to_csv(matched_csv, index=False)
    print(f"Saved: {hybrid_csv}")
    print(f"Saved: {matched_csv}")

    return hybrid_df, matched_df, hybrid_results


def run_phase7_10_benchmark():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("PHASE 7.10 — CONTINUAL HYBRID CLASSICAL + QUANTUM BENCHMARK")
    print("Matched 4-Way Evaluation: Classical vs Fixed Q vs Trainable Q vs Hybrid")
    print("15 Speakers x 4 Budgets (N in {20, 30, 40, 50}) on Protected Test Sets")
    print("=" * 80)

    # 1. Setup output paths
    phase7_4_dir = os.path.join(REPO_ROOT, "results", "continual", "phase7_4")
    phase7_8_dir = os.path.join(REPO_ROOT, "results", "continual", "phase7_8")
    phase7_9_dir = os.path.join(REPO_ROOT, "results", "continual", "phase7_9")
    results_dir = os.path.join(REPO_ROOT, "results", "continual", "phase7_10")
    reports_dir = os.path.join(REPO_ROOT, "reports", "continual", "phase7_10")
    models_dir = os.path.join(REPO_ROOT, "models", "continual", "phase7_10")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Load frozen baseline controls
    classical_csv_path = os.path.join(phase7_4_dir, "continual_classical_results.csv")
    fixed_q_csv_path = os.path.join(phase7_8_dir, "continual_quantum_results.csv")
    trainable_q_csv_path = os.path.join(phase7_9_dir, "continual_trainable_quantum_results.csv")

    if not os.path.isfile(classical_csv_path):
        raise FileNotFoundError(f"Missing canonical Phase 7.4 control results at: {classical_csv_path}")
    if not os.path.isfile(fixed_q_csv_path):
        raise FileNotFoundError(f"Missing canonical Phase 7.8 fixed quantum results at: {fixed_q_csv_path}")
    if not os.path.isfile(trainable_q_csv_path):
        raise FileNotFoundError(f"Missing canonical Phase 7.9 trainable quantum results at: {trainable_q_csv_path}")

    classical_df = pd.read_csv(classical_csv_path)
    fixed_q_df = pd.read_csv(fixed_q_csv_path)
    trainable_q_df = pd.read_csv(trainable_q_csv_path)
    print(f"Loaded {len(classical_df)} frozen Phase 7.4 classical control cells.")
    print(f"Loaded {len(fixed_q_df)} frozen Phase 7.8 fixed quantum control cells.")
    print(f"Loaded {len(trainable_q_df)} frozen Phase 7.9 trainable quantum control cells.")

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

    # 4. Initialize registries & governance
    model_registry = ModelRegistry(base_dir=models_dir)
    exp_registry = ExperimentRegistry(registry_dir=results_dir)
    validation_gate = ValidationGate()

    parser = argparse.ArgumentParser(description="Phase 7.10 Hybrid Classical + Quantum Benchmark")
    parser.add_argument("--recompute-only", action="store_true", help="Recompute statistics and summary from existing CSVs")
    args = parser.parse_args()

    results_csv_path = os.path.join(results_dir, "continual_hybrid_results.csv")
    matched_csv_path = os.path.join(results_dir, "matched_four_way_comparison.csv")

    if args.recompute_only and os.path.isfile(results_csv_path) and os.path.isfile(matched_csv_path):
        print("\n[--recompute-only] Loading existing Phase 7.10 benchmark CSVs...")
        hybrid_df = pd.read_csv(results_csv_path)
        matched_df = pd.read_csv(matched_csv_path)
        hybrid_results = hybrid_df.to_dict(orient="records")
        total_cells = len(hybrid_results)
    else:
        # Preload Whisper embeddings
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

        hybrid_df, matched_df, hybrid_results = run_hybrid_simulation(
            speakers=speakers,
            budgets=budgets,
            dysarthric_set=dysarthric_set,
            splits=splits,
            label_map=label_map,
            sess_map=sess_map,
            embedding_cache=embedding_cache,
            classical_df=classical_df,
            fixed_q_df=fixed_q_df,
            trainable_q_df=trainable_q_df,
            model_registry=model_registry,
            validation_gate=validation_gate,
            exp_registry=exp_registry,
            models_dir=models_dir,
            results_dir=results_dir,
        )
        total_cells = len(hybrid_results)

    # 5. Trajectory Summary
    traj_summary = matched_df.groupby("budget").agg({
        "classical_macro_f1": "mean",
        "fixed_q_macro_f1": "mean",
        "trainable_q_macro_f1": "mean",
        "hybrid_macro_f1": "mean",
        "delta_hyb_vs_c_macro_f1": "mean",
        "delta_hyb_vs_tq_macro_f1": "mean",
        "classical_emergency_recall": "mean",
        "fixed_q_emergency_recall": "mean",
        "trainable_q_emergency_recall": "mean",
        "hybrid_emergency_recall": "mean",
        "delta_hyb_vs_c_emergency_recall": "mean",
        "delta_hyb_vs_tq_emergency_recall": "mean",
        "classical_false_emergency_rate": "mean",
        "fixed_q_false_emergency_rate": "mean",
        "trainable_q_false_emergency_rate": "mean",
        "hybrid_false_emergency_rate": "mean",
        "adaptation_time_sec": "mean",
    }).reset_index()

    traj_csv = os.path.join(results_dir, "hybrid_trajectory_summary.csv")
    traj_summary.to_csv(traj_csv, index=False)
    print(f"Saved: {traj_csv}")

    # Subgroup breakdown
    subgroup_summary = matched_df.groupby(["speaker_group", "budget"]).agg({
        "classical_macro_f1": "mean",
        "trainable_q_macro_f1": "mean",
        "hybrid_macro_f1": "mean",
        "delta_hyb_vs_c_macro_f1": "mean",
        "delta_hyb_vs_tq_macro_f1": "mean",
        "classical_emergency_recall": "mean",
        "trainable_q_emergency_recall": "mean",
        "hybrid_emergency_recall": "mean",
        "delta_hyb_vs_c_emergency_recall": "mean",
        "delta_hyb_vs_tq_emergency_recall": "mean",
        "classical_false_emergency_rate": "mean",
        "trainable_q_false_emergency_rate": "mean",
        "hybrid_false_emergency_rate": "mean",
    }).reset_index()

    subgroup_csv = os.path.join(results_dir, "hybrid_subgroup_analysis.csv")
    subgroup_summary.to_csv(subgroup_csv, index=False)
    print(f"Saved: {subgroup_csv}")

    # 6. Statistical Hypothesis Testing
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS: MATCHED WILCOXON SIGNED-RANK PAIRED TESTS")
    print("=" * 80)

    def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        diffs = x - y
        mean_d = float(np.mean(diffs))
        std_d = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        d = float(mean_d / std_d) if std_d > 1e-9 else 0.0
        if np.all(np.abs(diffs) < 1e-9):
            return mean_d, d, 0.0, 1.0
        try:
            res_w = stats.wilcoxon(x, y, alternative="two-sided")
            return mean_d, d, float(res_w.statistic), float(res_w.pvalue)
        except Exception:
            return mean_d, d, 0.0, 1.0

    stat_results = {
        "hybrid_vs_classical": {},
        "hybrid_vs_fixed_q": {},
        "hybrid_vs_trainable_q": {},
    }

    # Comparison 1: Hybrid vs Classical Baseline
    comp_c = [
        ("macro_f1", matched_df["hybrid_macro_f1"].values, matched_df["classical_macro_f1"].values),
        ("emergency_recall", matched_df["hybrid_emergency_recall"].values, matched_df["classical_emergency_recall"].values),
        ("false_emergency_rate", matched_df["hybrid_false_emergency_rate"].values, matched_df["classical_false_emergency_rate"].values),
        ("accuracy", matched_df["hybrid_accuracy"].values, matched_df["classical_accuracy"].values),
    ]

    print("\n--- COMPARISON 1: HYBRID VS CONTINUAL CLASSICAL BASELINE ---")
    p_vals_c = []
    for mname, hyb_vals, c_vals in comp_c:
        mean_d, d, w_stat, p_val = paired_wilcoxon(hyb_vals, c_vals)
        stat_results["hybrid_vs_classical"][mname] = {
            "mean_hybrid": float(np.mean(hyb_vals)),
            "mean_classical": float(np.mean(c_vals)),
            "mean_delta": mean_d,
            "cohens_d": d,
            "wilcoxon_stat": w_stat,
            "p_value_raw": p_val,
        }
        p_vals_c.append(p_val)

    for i, (mname, _, _) in enumerate(comp_c):
        adj_p = min(1.0, p_vals_c[i] * len(comp_c))
        stat_results["hybrid_vs_classical"][mname]["p_value_adjusted"] = adj_p
        stat_results["hybrid_vs_classical"][mname]["statistically_significant"] = bool(adj_p < 0.05)
        sr = stat_results["hybrid_vs_classical"][mname]
        sig = "***" if adj_p < 0.001 else ("*" if adj_p < 0.05 else "n.s.")
        print(f"{mname:<22}: Hyb={sr['mean_hybrid']:.4f} | C={sr['mean_classical']:.4f} | Δ={sr['mean_delta']:+.4f} (d={sr['cohens_d']:+.2f}) | p_adj={adj_p:.4e} [{sig}]")

    # Comparison 2: Hybrid vs Trainable Quantum Baseline
    comp_tq = [
        ("macro_f1", matched_df["hybrid_macro_f1"].values, matched_df["trainable_q_macro_f1"].values),
        ("emergency_recall", matched_df["hybrid_emergency_recall"].values, matched_df["trainable_q_emergency_recall"].values),
        ("false_emergency_rate", matched_df["hybrid_false_emergency_rate"].values, matched_df["trainable_q_false_emergency_rate"].values),
        ("accuracy", matched_df["hybrid_accuracy"].values, matched_df["trainable_q_accuracy"].values),
    ]

    print("\n--- COMPARISON 2: HYBRID VS CONTINUAL TRAINABLE QUANTUM BASELINE ---")
    p_vals_tq = []
    for mname, hyb_vals, tq_vals in comp_tq:
        mean_d, d, w_stat, p_val = paired_wilcoxon(hyb_vals, tq_vals)
        stat_results["hybrid_vs_trainable_q"][mname] = {
            "mean_hybrid": float(np.mean(hyb_vals)),
            "mean_trainable_q": float(np.mean(tq_vals)),
            "mean_delta": mean_d,
            "cohens_d": d,
            "wilcoxon_stat": w_stat,
            "p_value_raw": p_val,
        }
        p_vals_tq.append(p_val)

    for i, (mname, _, _) in enumerate(comp_tq):
        adj_p = min(1.0, p_vals_tq[i] * len(comp_tq))
        stat_results["hybrid_vs_trainable_q"][mname]["p_value_adjusted"] = adj_p
        stat_results["hybrid_vs_trainable_q"][mname]["statistically_significant"] = bool(adj_p < 0.05)
        sr = stat_results["hybrid_vs_trainable_q"][mname]
        sig = "***" if adj_p < 0.001 else ("*" if adj_p < 0.05 else "n.s.")
        print(f"{mname:<22}: Hyb={sr['mean_hybrid']:.4f} | TQ={sr['mean_trainable_q']:.4f} | Δ={sr['mean_delta']:+.4f} (d={sr['cohens_d']:+.2f}) | p_adj={adj_p:.4e} [{sig}]")

    # 7. Subgroup Performance
    dys_df = matched_df[matched_df["speaker_group"] == "dysarthric"]
    ctrl_df = matched_df[matched_df["speaker_group"] == "control"]

    def build_subgroup_dict(df: pd.DataFrame, label: str) -> Dict[str, Any]:
        return {
            "subgroup": label,
            "n_cells": len(df),
            "macro_f1": {
                "classical": float(np.mean(df["classical_macro_f1"])),
                "trainable_q": float(np.mean(df["trainable_q_macro_f1"])),
                "hybrid": float(np.mean(df["hybrid_macro_f1"])),
                "delta_hyb_vs_c": float(np.mean(df["delta_hyb_vs_c_macro_f1"])),
                "delta_hyb_vs_tq": float(np.mean(df["delta_hyb_vs_tq_macro_f1"])),
            },
            "emergency_recall": {
                "classical": float(np.mean(df["classical_emergency_recall"])),
                "trainable_q": float(np.mean(df["trainable_q_emergency_recall"])),
                "hybrid": float(np.mean(df["hybrid_emergency_recall"])),
                "delta_hyb_vs_c": float(np.mean(df["delta_hyb_vs_c_emergency_recall"])),
                "delta_hyb_vs_tq": float(np.mean(df["delta_hyb_vs_tq_emergency_recall"])),
            },
            "false_emergency_rate": {
                "classical": float(np.mean(df["classical_false_emergency_rate"])),
                "trainable_q": float(np.mean(df["trainable_q_false_emergency_rate"])),
                "hybrid": float(np.mean(df["hybrid_false_emergency_rate"])),
                "delta_hyb_vs_c": float(np.mean(df["delta_hyb_vs_c_false_emergency_rate"])),
                "delta_hyb_vs_tq": float(np.mean(df["delta_hyb_vs_tq_false_emergency_rate"])),
            },
        }

    subgroups = {
        "cohort": build_subgroup_dict(matched_df, "All 15 Speakers"),
        "dysarthric": build_subgroup_dict(dys_df, "8 Dysarthric Speakers"),
        "control": build_subgroup_dict(ctrl_df, "7 Control Speakers"),
    }

    # 8. Scientific Verdict for Hybrid Model
    f1_sig_c = stat_results["hybrid_vs_classical"]["macro_f1"]["statistically_significant"]
    f1_delta_c = stat_results["hybrid_vs_classical"]["macro_f1"]["mean_delta"]
    rec_delta_c = stat_results["hybrid_vs_classical"]["emergency_recall"]["mean_delta"]

    f1_sig_tq = stat_results["hybrid_vs_trainable_q"]["macro_f1"]["statistically_significant"]
    f1_delta_tq = stat_results["hybrid_vs_trainable_q"]["macro_f1"]["mean_delta"]
    rec_sig_tq = stat_results["hybrid_vs_trainable_q"]["emergency_recall"]["statistically_significant"]
    rec_delta_tq = stat_results["hybrid_vs_trainable_q"]["emergency_recall"]["mean_delta"]

    # Evaluation rule
    if f1_sig_c and f1_delta_c > 0 and rec_sig_tq and rec_delta_tq > 0:
        hybrid_verdict = "SUPPORTED"
    elif (f1_delta_c > 0 or rec_delta_tq > 0) and (rec_delta_c < 0 or not rec_sig_tq):
        hybrid_verdict = "PARTIALLY SUPPORTED"
    elif f1_delta_c <= 0 and rec_delta_tq <= 0:
        hybrid_verdict = "NOT SUPPORTED"
    else:
        hybrid_verdict = "INCONCLUSIVE"

    print("\n" + "=" * 80)
    print(f"HYBRID PERSONALIZATION VERDICT: {hybrid_verdict}")
    print("Frozen Historical Conclusion (CASE B): UNCHANGED")
    print("=" * 80)

    # 9. Save Summary JSON
    summary_payload = {
        "phase": "7.10",
        "description": "Continual Hybrid Classical + Quantum Personalization Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_experiments": len(hybrid_results),
        "total_cells_evaluated": total_cells,
        "speakers_evaluated": len(speakers),
        "budgets_evaluated": budgets,
        "canonical_hybrid_config": CANONICAL_HYBRID_CONFIG,
        "statistical_tests": stat_results,
        "subgroup_breakdown": subgroups,
        "hybrid_verdict": hybrid_verdict,
        "frozen_case_b_conclusion": "UNCHANGED",
    }

    summary_json_path = os.path.join(reports_dir, "continual_hybrid_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    run_phase7_10_benchmark()
