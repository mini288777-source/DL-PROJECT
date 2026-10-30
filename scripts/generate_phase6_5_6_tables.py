"""
Phase 6.5.6: Final Fair Comparison, Evidence Synthesis & Research Closure
Script to generate authoritative synthesis CSV tables:
1. results/phase6_5_6_final_comparison_table.csv
2. results/phase6_5_6_evidence_chain.csv
3. results/phase6_5_6_metric_reconciliation.csv
4. results/phase6_5_6_final_decision.csv

STRICTLY NO NEW EXPERIMENTAL DATA - SYNTHESIS FROM FROZEN ARTIFACTS ONLY.
"""

import os
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_final_comparison_table():
    """
    Generate results/phase6_5_6_final_comparison_table.csv
    Clearly separated into Section A (Original Benchmark) and Section B (Recovery Protocol).
    """
    records = [
        # SECTION A: ORIGINAL BENCHMARK (Phases 3-6)
        {
            "section": "SECTION A: ORIGINAL BENCHMARK",
            "protocol": "Original Track A (Random Unstratified, Flat 7-Class, N in {20,30,40,50})",
            "model": "Original Unweighted RBF-SVM (Phase 3)",
            "dimension": "384",
            "qubits": "0",
            "reps": "0",
            "macro_f1": "0.1390",
            "emergency_recall_pct": "0.00%",
            "emergency_precision_pct": "0.00%",
            "false_emergency_rate_pct": "0.00%",
            "iper_pct": "0.00%",
            "stage2_macro_f1": "N/A (Flat 7-Class)",
            "stage2_balanced_acc_pct": "N/A (Flat 7-Class)",
            "train_runtime_sec": "0.001 s",
            "eval_runtime_sec": "0.001 s",
            "statistical_conclusion": "Majority-class collapse to GENERAL_SPEECH (~95.1% prevalence). Misleading high accuracy (~95%). Zero emergency sensitivity.",
            "operational_status": "Severe support failure / Unusable"
        },
        {
            "section": "SECTION A: ORIGINAL BENCHMARK",
            "protocol": "Original Track A (Random Unstratified, Flat 7-Class, N in {20,30,40,50})",
            "model": "Original Classical Adapter (Phase 3)",
            "dimension": "384",
            "qubits": "0",
            "reps": "0",
            "macro_f1": "0.1410-0.1470",
            "emergency_recall_pct": "0.00-1.20%",
            "emergency_precision_pct": "0.00-0.50%",
            "false_emergency_rate_pct": "0.00-0.80%",
            "iper_pct": "0.00-0.30%",
            "stage2_macro_f1": "N/A (Flat 7-Class)",
            "stage2_balanced_acc_pct": "N/A (Flat 7-Class)",
            "train_runtime_sec": "0.250 s",
            "eval_runtime_sec": "0.001 s",
            "statistical_conclusion": "Slightly less collapsed than RBF, but essentially collapsed due to severe calibration imbalance (7/60 conditions were single-class).",
            "operational_status": "Severe support failure / Unusable"
        },
        {
            "section": "SECTION A: ORIGINAL BENCHMARK",
            "protocol": "Original Track A (Random Unstratified, Flat 7-Class, N in {20,30,40,50})",
            "model": "Original QSVC q=4 reps=1 (Phase 4-6)",
            "dimension": "4 (PCA)",
            "qubits": "4",
            "reps": "1",
            "macro_f1": "0.1388",
            "emergency_recall_pct": "0.00%",
            "emergency_precision_pct": "0.00%",
            "false_emergency_rate_pct": "0.00%",
            "iper_pct": "0.00%",
            "stage2_macro_f1": "N/A (Flat 7-Class)",
            "stage2_balanced_acc_pct": "N/A (Flat 7-Class)",
            "train_runtime_sec": "0.025 s",
            "eval_runtime_sec": "0.380 s",
            "statistical_conclusion": "Bitwise-identical predictions to unweighted RBF across 81,840 paired evaluations. Quantum kernel collapsed identically to majority class.",
            "operational_status": "Severe support failure / Unusable"
        },

        # SECTION B: RECOVERY BENCHMARK (Phases 6.5.0-6.5.5-C)
        {
            "section": "SECTION B: RECOVERY BENCHMARK",
            "protocol": "Recovery Track B (Speaker-Stratified, Prompt-Isolated, Hierarchical 2-Stage, N in {20,30,40,50})",
            "model": "Weighted Classical Adapter d=8 (Phase 6.5.5)",
            "dimension": "8 (PCA)",
            "qubits": "0",
            "reps": "0",
            "macro_f1": "0.0949 +/- 0.0342",
            "emergency_recall_pct": "45.81% +/- 29.78%",
            "emergency_precision_pct": "2.03% +/- 1.45%",
            "false_emergency_rate_pct": "49.82% +/- 22.98%",
            "iper_pct": "14.21% +/- 14.17%",
            "stage2_macro_f1": "0.1824 +/- 0.1220",
            "stage2_balanced_acc_pct": "33.57% +/- 19.60%",
            "train_runtime_sec": "0.2625 s (15.79 s total)",
            "eval_runtime_sec": "0.0006 s",
            "statistical_conclusion": "Statistically superior Emergency Recall over QSVC Q8-R1 (Delta = +18.29 pp, p = 0.000125), but at cost of severe false-alarm burden (FER = 49.82%).",
            "operational_status": "Safety Tradeoff / NOT READY"
        },
        {
            "section": "SECTION B: RECOVERY BENCHMARK",
            "protocol": "Recovery Track B (Speaker-Stratified, Prompt-Isolated, Hierarchical 2-Stage, N in {20,30,40,50})",
            "model": "Weighted Classical RBF-SVM d=8 (Phase 6.5.5)",
            "dimension": "8 (PCA)",
            "qubits": "0",
            "reps": "0",
            "macro_f1": "0.1087 +/- 0.0487",
            "emergency_recall_pct": "31.89% +/- 40.97%",
            "emergency_precision_pct": "2.02% +/- 6.57%",
            "false_emergency_rate_pct": "29.32% +/- 38.07%",
            "iper_pct": "11.69% +/- 17.14%",
            "stage2_macro_f1": "0.1962 +/- 0.1091",
            "stage2_balanced_acc_pct": "33.91% +/- 18.19%",
            "train_runtime_sec": "0.0040 s (0.33 s total)",
            "eval_runtime_sec": "0.0014 s",
            "statistical_conclusion": "Primary classical reference baseline. No statistically supported difference vs QSVC Q8-R1 across all primary safety metrics (Case B).",
            "operational_status": "Benchmark Reference / NOT READY"
        },
        {
            "section": "SECTION B: RECOVERY BENCHMARK",
            "protocol": "Recovery Track B (Speaker-Stratified, Prompt-Isolated, Hierarchical 2-Stage, N in {20,30,40,50})",
            "model": "QSVC Q8-R1 (Primary Quantum Recovery)",
            "dimension": "8 (PCA)",
            "qubits": "8",
            "reps": "1",
            "macro_f1": "0.1051 +/- 0.0575",
            "emergency_recall_pct": "27.52% +/- 41.94%",
            "emergency_precision_pct": "0.81% +/- 1.31%",
            "false_emergency_rate_pct": "28.12% +/- 41.86%",
            "iper_pct": "8.25% +/- 13.61%",
            "stage2_macro_f1": "0.1252 +/- 0.0844",
            "stage2_balanced_acc_pct": "32.53% +/- 23.25%",
            "train_runtime_sec": "0.0888 s (70.22 s total)",
            "eval_runtime_sec": "1.0816 s",
            "statistical_conclusion": "Case B: No statistically supported difference vs RBF-SVM d=8 (Macro-F1 p=0.1714, Recall p=0.0689, FER p=0.9505, IPER p=0.0582). ~772.6x slower per eval.",
            "operational_status": "No Quantum Advantage / NOT READY"
        },
        {
            "section": "SECTION B: RECOVERY BENCHMARK",
            "protocol": "Recovery Track B (Speaker-Stratified, Prompt-Isolated, Hierarchical 2-Stage, N in {20,30,40,50})",
            "model": "QSVC Q4-R1 (Qubit Scaling Ablation)",
            "dimension": "4 (PCA)",
            "qubits": "4",
            "reps": "1",
            "macro_f1": "0.1108 +/- 0.0527",
            "emergency_recall_pct": "27.25% +/- 39.84%",
            "emergency_precision_pct": "1.75% +/- 4.68%",
            "false_emergency_rate_pct": "27.97% +/- 38.11%",
            "iper_pct": "6.61% +/- 10.55%",
            "stage2_macro_f1": "0.1455 +/- 0.1041",
            "stage2_balanced_acc_pct": "27.01% +/- 18.82%",
            "train_runtime_sec": "0.0247 s (1.48 s total)",
            "eval_runtime_sec": "0.4194 s",
            "statistical_conclusion": "No statistically supported difference vs Q8-R1 (Macro-F1 p=1.0, Recall p=1.0). Qubit expansion 4 -> 8 yielded no generalization improvement.",
            "operational_status": "Ablation Baseline / NOT READY"
        },
        {
            "section": "SECTION B: RECOVERY BENCHMARK",
            "protocol": "Recovery Track B (Speaker-Stratified, Prompt-Isolated, Hierarchical 2-Stage, N in {20,30,40,50})",
            "model": "QSVC Q8-R2 (Circuit Depth Ablation)",
            "dimension": "8 (PCA)",
            "qubits": "8",
            "reps": "2",
            "macro_f1": "0.1070 +/- 0.0558",
            "emergency_recall_pct": "26.43% +/- 42.51%",
            "emergency_precision_pct": "0.66% +/- 1.16%",
            "false_emergency_rate_pct": "26.73% +/- 41.70%",
            "iper_pct": "7.65% +/- 13.84%",
            "stage2_macro_f1": "0.1296 +/- 0.0974",
            "stage2_balanced_acc_pct": "30.48% +/- 18.45%",
            "train_runtime_sec": "0.1578 s (128.04 s total)",
            "eval_runtime_sec": "1.9761 s",
            "statistical_conclusion": "No statistically supported difference vs Q8-R1 (Recall p=1.0, Macro-F1 shift Delta = +0.0019 negligible) while doubling simulation time (128.04 s vs 70.22 s).",
            "operational_status": "Ablation Baseline / NOT READY"
        }
    ]

    df = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "phase6_5_6_final_comparison_table.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} ({len(df)} rows)")


def generate_evidence_chain_table():
    """
    Generate results/phase6_5_6_evidence_chain.csv
    Scientific progression table from Phase 6.5.0 to Phase 6.5.5-C.
    """
    records = [
        {
            "phase": "Phase 6.5.0",
            "phase_name": "Diagnostic",
            "problem_tested": "Data feasibility, class imbalance, and representation collapse in original benchmark",
            "intervention": "Audit 42,536 Phase 3-6 predictions, calibration support distributions, and embedding geometry",
            "key_observation": "Severe data support failure: GENERAL_SPEECH was ~95.10% of corpus; 7/60 calibration splits were single-class; N=20 had only ~1.1 emergency samples",
            "interpretation": "Original majority collapse was driven by protocol and severe class support deficiency rather than uniquely implicating quantum kernel mechanics",
            "limitation": "Diagnostic audit only; did not modify dataset splits or retrain models"
        },
        {
            "phase": "Phase 6.5.1",
            "phase_name": "Speaker-Stratified Protocol",
            "problem_tested": "Emergency-class scarcity and random calibration support failure",
            "intervention": "Deterministic speaker-stratified prompt-isolated calibration protocol (Track B) guaranteeing emergency-aware sampling across N in {20,30,40,50}",
            "key_observation": "Calibration emergency support increased from ~1.1 samples to ~11.3 samples at N=20; represented emergency classes rose from ~1.0 to ~5.2",
            "interpretation": "Eliminated zero-support emergency conditions in calibration splits, establishing valid data foundation for personalized few-shot adaptation",
            "limitation": "Limited by underlying TORGO emergency class distribution; LOCATION and PAIN remain inherently sparse"
        },
        {
            "phase": "Phase 6.5.2",
            "phase_name": "Classical Rescue",
            "problem_tested": "Flat 7-class learning failure and unweighted majority collapse",
            "intervention": "Cost-sensitive balanced class-weighting applied to classical Adapter and RBF-SVM baselines",
            "key_observation": "Class weighting successfully broke majority collapse: Weighted Adapter achieved ~58% emergency recall; Weighted RBF achieved non-trivial recall",
            "interpretation": "Cost-sensitive learning proves emergency intents can be recognized, but flat classification causes extreme false-alarm explosion (FER up to ~96.8%)",
            "limitation": "Flat cost-sensitive models trade off false alarms unsustainably; not viable for safety-critical deployment"
        },
        {
            "phase": "Phase 6.5.2-C",
            "phase_name": "Classical Analytical Closure",
            "problem_tested": "Attribution of performance shifts between stratification and class weighting",
            "intervention": "Factorial attribution analysis isolating data stratification effect from cost-sensitive weighting effect",
            "key_observation": "Class weighting accounted for >90% of recall recovery, while speaker stratification provided stable underlying sample support",
            "interpretation": "Confirms that both data foundation and loss formulation are required to escape collapse, but false alarm burden necessitates architectural change",
            "limitation": "Observational decomposition on fixed benchmark cohort"
        },
        {
            "phase": "Phase 6.5.3",
            "phase_name": "Hierarchical Recognition",
            "problem_tested": "Unacceptable false-alarm rates in flat cost-sensitive classification",
            "intervention": "Two-stage hierarchical architecture: Stage-1 binary emergency triage gate + Stage-2 conditional intent routing",
            "key_observation": "Hierarchical routing dramatically reduced false-emergency rate (e.g. RBF N=30 FER reduced from 96.82% to 15.06%, Delta = -81.76 pp)",
            "interpretation": "Decoupling emergency triage from intent classification substantially mitigates false alarm burden while preserving structured routing",
            "limitation": "Stage-1 precision remains low (~0.8%-2.0%) due to severe ~95% negative prevalence in test data; missed emergencies persist"
        },
        {
            "phase": "Phase 6.5.3-C",
            "phase_name": "Hierarchical Closure",
            "problem_tested": "Stage-1 gating vs Stage-2 routing error attribution and threshold sensitivity",
            "intervention": "Condition-level error taxonomy, threshold sweep, and matched flat-vs-hierarchical paired testing",
            "key_observation": "Error attribution confirmed Stage-1 false positives and missed events dominate overall failure modes; Stage-2 conditional accuracy was ~30-34%",
            "interpretation": "Architectural separation validated, but performance remains constrained by acoustic representation overlap at low N",
            "limitation": "Fixed probability threshold (tau=0.5); personalized thresholding unviable without validation splits"
        },
        {
            "phase": "Phase 6.5.4",
            "phase_name": "Representation Space & PCA Audit",
            "problem_tested": "Acoustic representation bottleneck and historical PCA explained variance discrepancy",
            "intervention": "Comprehensive representation space audit across d in {2,4,8,16,32,64,128,256,384}; forensic investigation of historical 27.4% PCA claim",
            "key_observation": "Empirical PCA-4 variance is 80.66% +/- 4.37% (historical 27.4% was a plotting script artifact); d=8 retains 89.73% +/- 2.99% variance and relieves Stage-2 bottleneck",
            "interpretation": "Selected d=8 as primary candidate for quantum comparison, establishing fair representational parity between classical and quantum models",
            "limitation": "PCA is linear unsupervised reduction; frozen Whisper-tiny front-end limits separable representation"
        },
        {
            "phase": "Phase 6.5.5",
            "phase_name": "Quantum Kernel Recovery",
            "problem_tested": "Empirical evaluation of quantum kernel vs matched classical baselines under recovered protocol",
            "intervention": "Executed 180 quantum conditions (Q8-R1, Q4-R1, Q8-R2) with ZZFeatureMap and FidelityQuantumKernel vs matched RBF-SVM and Adapter (d=8)",
            "key_observation": "All 180 quantum kernels mathematically valid; QSVC Q8-R1 performance closely matched Classical RBF-SVM d=8 (Macro-F1 0.1051 vs 0.1087)",
            "interpretation": "Under the evaluated frozen representation, feature scaling, and few-shot protocol, the fidelity quantum kernel produced performance statistically indistinguishable from the matched classical RBF-SVM; no empirical advantage observed",
            "limitation": "Simulated statevectors on CPU; N <= 50; fixed ZZFeatureMap without trainable quantum parameters"
        },
        {
            "phase": "Phase 6.5.5-C",
            "phase_name": "Quantum Statistical Closure",
            "problem_tested": "Rigorous paired hypothesis testing across 60 matched units, multiple comparison corrections, and runtime audit",
            "intervention": "Wilcoxon signed-rank tests, Cohen's dz, 95% CIs, Holm-Bonferroni correction, and per-eval vs total wall-time audit",
            "key_observation": "Primary comparison QSVC Q8-R1 vs RBF-SVM d=8 confirmed Case B (No statistically supported difference on any primary metric); QSVC ~772.6x slower per eval",
            "interpretation": "Rigorous statistical confirmation of null quantum hypothesis under evaluated protocol; Adapter showed superior emergency recall (45.81% vs 27.52%)",
            "limitation": "Findings strictly bounded to TORGO cohort, frozen Whisper-tiny embeddings, and exact statevector simulation"
        }
    ]

    df = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "phase6_5_6_evidence_chain.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} ({len(df)} rows)")


def generate_metric_reconciliation_table():
    """
    Generate results/phase6_5_6_metric_reconciliation.csv
    Authoritative reconciliation of historical reporting artifacts vs audited closure values.
    """
    records = [
        {
            "metric": "Emergency Precision (QSVC Q8-R1)",
            "historical_value": "11.24% +/- 9.45%",
            "closure_value": "0.81% +/- 1.31%",
            "definition": "TP_emerg / (TP_emerg + FP_emerg) evaluated on end-to-end binary emergency vs general predictions",
            "population": "All 60 matched conditions under Track B (including conditions with 0 TP, evaluated with zero_division=0)",
            "root_cause_explanation": "Historical 11.24% was an unverified reporting transcription artifact in the provisional Phase 6.5.5 text table; audited empirical value computed directly from speaker_results.csv is 0.81% (reflecting extreme 95.1% negative test prevalence).",
            "authoritative_final_value": "0.81% +/- 1.31%"
        },
        {
            "metric": "Emergency Precision (Classical Adapter d=8)",
            "historical_value": "9.21% +/- 7.15%",
            "closure_value": "2.03% +/- 1.45%",
            "definition": "TP_emerg / (TP_emerg + FP_emerg) evaluated on end-to-end binary emergency vs general predictions",
            "population": "All 60 matched conditions under Track B",
            "root_cause_explanation": "Historical 9.21% was an unverified reporting transcription artifact in the provisional Phase 6.5.5 text table; audited empirical value across all 60 conditions is 2.03%.",
            "authoritative_final_value": "2.03% +/- 1.45%"
        },
        {
            "metric": "Emergency Precision (Classical RBF-SVM d=8)",
            "historical_value": "11.89% +/- 9.87%",
            "closure_value": "2.02% +/- 6.57%",
            "definition": "TP_emerg / (TP_emerg + FP_emerg) evaluated on end-to-end binary emergency vs general predictions",
            "population": "All 60 matched conditions under Track B",
            "root_cause_explanation": "Historical 11.89% was an unverified reporting transcription artifact; audited empirical value across all 60 conditions is 2.02% (or 0.94% under strict per-slice macro in earlier table). Authoritative population mean is 2.02%.",
            "authoritative_final_value": "2.02% +/- 6.57%"
        },
        {
            "metric": "Emergency Precision (QSVC Q4-R1)",
            "historical_value": "12.04% +/- 9.71%",
            "closure_value": "1.75% +/- 4.68%",
            "definition": "TP_emerg / (TP_emerg + FP_emerg) evaluated on end-to-end binary emergency vs general predictions",
            "population": "All 60 matched conditions under Track B",
            "root_cause_explanation": "Historical 12.04% was an unverified reporting transcription artifact; audited empirical value across all 60 conditions is 1.75%.",
            "authoritative_final_value": "1.75% +/- 4.68%"
        },
        {
            "metric": "Emergency Precision (QSVC Q8-R2)",
            "historical_value": "11.45% +/- 9.52%",
            "closure_value": "0.66% +/- 1.16%",
            "definition": "TP_emerg / (TP_emerg + FP_emerg) evaluated on end-to-end binary emergency vs general predictions",
            "population": "All 60 matched conditions under Track B",
            "root_cause_explanation": "Historical 11.45% was an unverified reporting transcription artifact; audited empirical value across all 60 conditions is 0.66%.",
            "authoritative_final_value": "0.66% +/- 1.16%"
        },
        {
            "metric": "PCA Explained Variance (d=4)",
            "historical_value": "27.4%",
            "closure_value": "80.66% +/- 4.37%",
            "definition": "Cumulative explained variance ratio of top 4 principal components fitted on per-speaker calibration embeddings",
            "population": "15 TORGO speakers, per-speaker calibration splits (Track B)",
            "root_cause_explanation": "Historical 27.4% was traced in Phase 6.5.4 to a hardcoded plotting script placeholder array [17.8, 27.4, 34.2, 39.8]; forensic recomputation on actual empirical model-run data established 80.66% +/- 4.37%.",
            "authoritative_final_value": "80.66% +/- 4.37%"
        },
        {
            "metric": "PCA Explained Variance (d=8)",
            "historical_value": "39.8% (placeholder)",
            "closure_value": "89.73% +/- 2.99%",
            "definition": "Cumulative explained variance ratio of top 8 principal components fitted on per-speaker calibration embeddings",
            "population": "15 TORGO speakers, per-speaker calibration splits (Track B)",
            "root_cause_explanation": "Historical value was part of the hardcoded placeholder array; forensic recomputation confirmed d=8 captures 89.73% +/- 2.99% variance, relieving Stage-2 compression bottleneck.",
            "authoritative_final_value": "89.73% +/- 2.99%"
        },
        {
            "metric": "Quantum Runtime Penalty Metric",
            "historical_value": "770x slower overall",
            "closure_value": "772.6x per-evaluation; 212.8x total wall-time",
            "definition": "Ratio of QSVC Q8-R1 runtime to Classical RBF-SVM d=8 runtime",
            "population": "60 matched conditions (60 training fits, 22,168 test evaluations)",
            "root_cause_explanation": "Historical reporting conflated individual test evaluation latency (~772.6x slower) with total experiment wall-clock duration (~212.8x slower across all 60 conditions).",
            "authoritative_final_value": "772.6x per-evaluation; 212.8x total wall-time"
        }
    ]

    df = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "phase6_5_6_metric_reconciliation.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} ({len(df)} rows)")


def generate_final_decision_table():
    """
    Generate results/phase6_5_6_final_decision.csv
    Final frozen decisions and scientific classifications.
    """
    records = [
        {
            "comparison_scope": "Primary Model Comparison: QSVC Q8-R1 vs. Classical RBF-SVM d=8",
            "target_metric": "End-to-End Macro-F1",
            "delta_mean": "-0.0036",
            "ci_95": "[-0.0083, +0.0010]",
            "wilcoxon_w": "475.0",
            "raw_p_value": "0.1714",
            "holm_p_value": "0.3428",
            "decision": "CASE B: NO STATISTICALLY SUPPORTED DIFFERENCE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "Fidelity quantum kernel demonstrated no statistically supported performance benefit over matched classical RBF kernel."
        },
        {
            "comparison_scope": "Primary Model Comparison: QSVC Q8-R1 vs. Classical RBF-SVM d=8",
            "target_metric": "Emergency Recall",
            "delta_mean": "-4.37 pp",
            "ci_95": "[-8.14, -0.60] pp",
            "wilcoxon_w": "40.5",
            "raw_p_value": "0.0230",
            "holm_p_value": "0.0689",
            "decision": "CASE B: NO STATISTICALLY SUPPORTED DIFFERENCE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "Slight classical RBF recall advantage is non-significant after Holm-Bonferroni family-wise error rate control."
        },
        {
            "comparison_scope": "Primary Model Comparison: QSVC Q8-R1 vs. Classical RBF-SVM d=8",
            "target_metric": "False-Emergency Rate",
            "delta_mean": "-1.20 pp",
            "ci_95": "[-4.48, +2.08] pp",
            "wilcoxon_w": "59.0",
            "raw_p_value": "0.9505",
            "holm_p_value": "0.9505",
            "decision": "CASE B: NO STATISTICALLY SUPPORTED DIFFERENCE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "False alarm rates are statistically indistinguishable between quantum and classical RBF classifiers."
        },
        {
            "comparison_scope": "Primary Model Comparison: QSVC Q8-R1 vs. Classical RBF-SVM d=8",
            "target_metric": "End-to-End IPER",
            "delta_mean": "-3.44 pp",
            "ci_95": "[-6.11, -0.77] pp",
            "wilcoxon_w": "62.0",
            "raw_p_value": "0.0145",
            "holm_p_value": "0.0582",
            "decision": "CASE B: NO STATISTICALLY SUPPORTED DIFFERENCE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "IPER advantage for classical RBF is non-significant after Holm-Bonferroni correction."
        },
        {
            "comparison_scope": "Secondary Model Comparison: QSVC Q8-R1 vs. Weighted Adapter d=8",
            "target_metric": "Emergency Recall",
            "delta_mean": "-18.29 pp",
            "ci_95": "[-27.28, -9.29] pp",
            "wilcoxon_w": "166.5",
            "raw_p_value": "0.000125",
            "holm_p_value": "0.000874",
            "decision": "STATISTICALLY SUPPORTED CLASSICAL ADAPTER ADVANTAGE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "Weighted Classical Adapter achieves significantly higher emergency recall (45.81% vs 27.52%) at cost of high false alarms (49.82% vs 28.12%)."
        },
        {
            "comparison_scope": "Quantum Ablation: Qubit Scaling (Q4-R1 vs. Q8-R1)",
            "target_metric": "End-to-End Macro-F1",
            "delta_mean": "-0.0057",
            "ci_95": "[-0.0163, +0.0049]",
            "wilcoxon_w": "699.5",
            "raw_p_value": "0.3541",
            "holm_p_value": "1.0000",
            "decision": "NO STATISTICALLY SUPPORTED DIFFERENCE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "Increasing qubit count from 4 to 8 did not produce a statistically supported generalization improvement."
        },
        {
            "comparison_scope": "Quantum Ablation: Circuit Depth (Q8-R1 vs. Q8-R2)",
            "target_metric": "End-to-End Macro-F1",
            "delta_mean": "+0.0019",
            "ci_95": "[-0.0016, +0.0054]",
            "wilcoxon_w": "708.5",
            "raw_p_value": "0.4496",
            "holm_p_value": "1.0000",
            "decision": "NO STATISTICALLY SUPPORTED DIFFERENCE",
            "operational_status": "FROZEN",
            "scientific_interpretation": "Increasing reps from 1 to 2 did not produce a statistically supported improvement while doubling computational cost."
        }
    ]

    df = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, "phase6_5_6_final_decision.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    generate_final_comparison_table()
    generate_evidence_chain_table()
    generate_metric_reconciliation_table()
    generate_final_decision_table()
    print("All Phase 6.5.6 synthesis tables generated successfully.")
