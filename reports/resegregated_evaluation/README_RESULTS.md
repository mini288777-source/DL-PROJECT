# TORGO Resegregated Evaluation & Model Comparison Package (Reconciled v2)

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Team Name:** APPY FIZZ  
**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai  
**Date of Release:** 12 October 2026  
**Package Path:** `reports/resegregated_evaluation/`  

---

## 1. Executive Summary & Forensic Reconciliation

This package provides the complete, independently traceable, and mathematically verified experimental evidence comparing hybrid quantum-classical classifiers against matched classical baselines on the **TORGO dysarthric speech corpus**.

Following the forensic audit of evaluation artifacts ("APPY FIZZ — Resegregated Evaluation Evidence Audit (v2)"), this package resolves all 12 identified audit points:
1. **Audit v1 Freeze:** All original v1 artifacts are preserved unmodified in `reports/resegregated_evaluation/audit_v1_snapshot/`.
2. **Unified Test Cohort (5,317 Samples):** All confusion matrices, per-class metrics, and summary tables evaluate strictly on the exact 5,317 locked test samples declared in `splits/torgo_resegregated_v1.json` (5,219 GENERAL_SPEECH, 35 CRITICAL_RESPONSE, 21 FALL, 18 BREATHING_MEDICAL, 13 PAIN, 11 SOS_HELP, and 0 LOCATION).
3. **Sample-by-Sample Prediction Records:** Complete 191,412 prediction-level rows are saved in `results/predictions_resegregated_evaluation.csv` linking every condition ID, speaker, sample ID, true label, and model prediction.
4. **Condition ID Alignment:** All condition identifiers align to the manifest format `RESEG_{speaker}_N{budget}` (60 conditions).
5. **Accuracy Column & Unrounded Differences:** Added `Accuracy` to `condition_level_results.csv` and retained full-precision differences.
6. **Dual Estimand Transparency:** Both Condition-Level Mean (unweighted average across 60 conditions) and Pooled Test-Set Metrics (evaluated on the 5,317 test cohort) are explicitly reported and labeled in `model_comparison_summary.csv` to prevent estimand conflation.
7. **Speaker-Level Re-alignment:** Sample counts in `speaker_level_results.csv` match the manifest per speaker (F01=68, F03=163, ..., MC04=943, total 5,317), and metrics reflect the exact unweighted average across budgets.
8. **Sample-Budget Scaling Trend:** Reconciled curves in Figure 7 and Section 12 explain that increasing budget $N$ from 20 to 50 expands GENERAL_SPEECH calibration data, making the decision boundary more conservative: False Emergency Rate (FER) drops steeply from 53.42% to 13.77% while Emergency Recall adjusts from 54.00% to 13.33%, driving an overall increase in Macro-F1 from 0.0725 to 0.1223.
9. **Full-Precision Inferential Statistics:** Recomputed Wilcoxon signed-rank tests ($W$, $p$) from unrounded differences with Holm-Bonferroni correction and an independent 15-speaker clustered sensitivity test.
10. **PCA Variance Audited:** Local calibration PCA explained variance ($83.48\% \pm 3.12\%$) is documented across all 60 conditions in `pca_explained_variance_by_condition.csv` and distinguished from global corpus PCA (51.84%).
11. **Clinical & Metric Disclosures:** Disclosed FER definition ($\text{FP}/(\text{FP}+\text{TN})$), zero LOCATION test support, and proxy emergency label mapping.
12. **Complete Artifact Integrity:** Recomputed SHA-256 hashes across all 38+ artifacts in `manifests/artifact_inventory.json`.

---

## 2. Directory Structure

```text
reports/resegregated_evaluation/
├── RESEGREGATED_MODEL_COMPARISON_REPORT.docx   # Principal submission document (Word, 19 pages)
├── RESEGREGATED_MODEL_COMPARISON_REPORT.pdf    # Rendered submission document (PDF, 19 pages)
├── RESEGREGATED_MODEL_COMPARISON_REPORT.md     # Markdown results specification
├── README_RESULTS.md                           # Package organization and reconciliation guide
├── audit_v1_snapshot/                          # Frozen audit v1 artifacts snapshot
├── figures/                                    # 12 Publication-quality figures (PNG & SVG)
│   ├── fig1_dataset_distribution.png / .svg
│   ├── fig2_overall_accuracy.png / .svg
│   ├── fig3_macro_f1_comparison.png / .svg
│   ├── fig4_emergency_recall_vs_precision.png / .svg
│   ├── fig5_false_emergency_rate.png / .svg
│   ├── fig6_balanced_accuracy.png / .svg
│   ├── fig7_sample_budget_scaling.png / .svg
│   ├── fig8_speaker_variation.png / .svg
│   ├── fig9_quantum_vs_rbf_matched.png / .svg
│   ├── fig10_confusion_matrices.png / .svg
│   ├── fig11_per_class_performance.png / .svg
│   └── fig12_runtime_latency.png / .svg
├── manifests/                                  # Cryptographic provenance & metadata
│   ├── evaluation_manifest.json                # Complete 60-condition split manifest (v2)
│   └── artifact_inventory.json                 # SHA-256 hashes of all reconciled artifacts
└── results/                                    # Machine-readable evaluation tables
    ├── predictions_resegregated_evaluation.csv # 191,412 prediction-level records
    ├── model_comparison_summary.csv            # Dual estimand benchmark table (10 models)
    ├── condition_level_results.csv             # 60 speaker-by-budget condition metrics
    ├── per_class_metrics.csv                   # 7-class metrics on locked test cohort (5,317)
    ├── speaker_level_results.csv               # 15 individual speaker performances
    ├── statistical_comparisons.csv             # Wilcoxon paired tests & Holm corrections
    ├── pca_explained_variance_by_condition.csv # PCA 8-D and 4-D variance per condition
    ├── runtime_comparison.csv                  # Per-sample latency and wall-clock times
    └── confusion_matrices/                     # Confusion matrices (Row sum = 5,317)
        ├── confusion_matrix_qsvc_q8_r1.csv
        ├── confusion_matrix_rbf_svm8.csv
        ├── confusion_matrix_track_a_classical.csv
        ├── confusion_matrix_track_b_quantum.csv
        └── confusion_matrix_track_c_shadow.csv
```

---

## 3. Reconciled Master Metrics (Dual Estimands)

| Model Identifier | Input Space | Condition Mean Acc (%) | Pooled Test Acc (%) | Condition Mean Macro-F1 | Pooled 7-Class Macro-F1 | Condition Emerg Rec (%) | Pooled Emerg Rec (%) | Condition FER (%) | Pooled FER (%) | Latency / Smp | Scientific Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **QSVC Q8-R1** | $d = 8$ | 72.35% | 63.72% | 0.1100 | 0.1159 | 26.04% | 21.43% | 25.77% | 35.20% | 1.0816 s | Primary Quantum Model |
| **Matched RBF-SVM** | $d = 8$ | 71.75% | 63.51% | 0.1072 | 0.1145 | 26.85% | 20.41% | 26.39% | 35.39% | 0.0014 s | Matched Baseline (CASE B Parity) |
| **Linear SVM** | $d = 8$ | 47.62% | 47.43% | 0.0920 | 0.0973 | 42.35% | 43.88% | 51.37% | 51.91% | 0.0009 s | Linear Subspace Baseline |
| **KNN (k=5)** | $d = 8$ | 65.70% | 61.99% | 0.1082 | 0.1141 | 32.16% | 27.55% | 32.65% | 36.98% | 0.0021 s | Instance Metric Baseline |
| **QSVC Q4-R1** | $d = 4$ | 71.75% | 57.01% | 0.1034 | 0.1118 | 47.42% | 42.86% | 42.08% | 42.21% | 0.0824 s | Compact Quantum Kernel |
| **QSVC Q8-R2** | $d = 8$ | 71.75% | 63.81% | 0.1094 | 0.1160 | 27.00% | 21.43% | 25.78% | 35.10% | 2.1540 s | Circuit Depth Ablation (R=2) |
| **Track A (Classical)** | 384-D | 54.42% | 62.91% | 0.0980 | 0.1185 | 37.77% | 31.63% | 44.19% | 36.12% | 0.0042 s | Classical Residual |
| **Track B (Quantum)** | $d = 8$ | 53.44% | 63.33% | 0.0971 | 0.1186 | 37.59% | 28.57% | 45.20% | 35.68% | 0.0650 s | Quantum VQC Residual |
| **Track C (Shadow)** | $d = 8$ | 49.77% | 53.23% | 0.0929 | 0.1082 | 41.70% | 42.86% | 48.98% | 46.06% | 0.0028 s | Matched Classical Shadow |

---

## 4. Key Inferential Findings

1. **Parity Finding Confirmed (CASE B):**
   - Paired Wilcoxon Signed-Rank Test across 60 conditions:
     - Macro-F1: $W = 219.0$, unadjusted $p = 0.5698$, Holm-Bonferroni adjusted $p = 1.000$.
     - Emergency Recall: $W = 1.0$, unadjusted $p = 0.6547$, Holm-Bonferroni adjusted $p = 1.000$.
     - False Emergency Rate: $W = 153.5$, unadjusted $p = 0.0640$, Holm-Bonferroni adjusted $p = 0.1921$.
   - 15-Speaker Clustered Sensitivity Analysis:
     - Macro-F1: $W = 41.0$, $p = 0.4703$. Accounting for repeated within-speaker measurements confirms no statistically significant advantage.

2. **Computational Simulation Gap:**
   - 8-qubit quantum statevector simulation requires $1.0816\text{ s}$ per sample on CPU compared to $0.0014\text{ s}$ for RBF-SVM ($772.6\times$ latency overhead).

3. **Clinical Feasibility Disclosure:**
   - Both quantum and classical models exhibit False Emergency Rates exceeding 25%, failing the pre-registered clinical feasibility ceiling of 15%. Autonomous emergency communication remains blocked without human triage.
