# TORGO Resegregated Evaluation & Model Comparison Package

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Team Name:** APPY FIZZ  
**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai  
**Date of Release:** October 2026  
**Package Path:** `reports/resegregated_evaluation/`

---

## 1. Executive Summary & Purpose

This package provides the complete, independent, and reproducible experimental evidence comparing hybrid quantum-classical classifiers against matched classical baselines on the **TORGO dysarthric speech corpus**.

Following a forensic audit of historical data segregation, this evaluation resolves:
1. **Cross-Microphone Leakage:** Channel variants (`arrayMic` vs. `headMic`) of identical acoustic events are strictly partitioned, ensuring zero cross-channel leakage between calibration and evaluation sets.
2. **Prompt-Text Memorization:** Repeated prompts spoken by the same speaker are grouped and disjointly assigned, preventing lexical memorization from inflating intent classification accuracy.
3. **Locked Test Partitions:** Evaluation test sets are strictly locked across few-shot calibration budgets ($N \in \{20, 30, 40, 50\}$), isolating genuine sample efficiency from shifting test cohort dynamics.
4. **Matched Dimensionality:** The primary quantum model (`QSVC Q8-R1`) and its matched classical comparator (`RBF-SVM d=8`) evaluate on strictly identical 8-dimensional PCA representations derived from frozen Whisper-tiny bottleneck features.

---

## 2. Directory Structure

```text
reports/resegregated_evaluation/
├── RESEGREGATED_MODEL_COMPARISON_REPORT.docx   # Principal submission document (Word)
├── RESEGREGATED_MODEL_COMPARISON_REPORT.pdf    # Rendered submission document (PDF)
├── RESEGREGATED_MODEL_COMPARISON_REPORT.md     # Markdown results specification
├── README_RESULTS.md                           # Package organization and guide
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
│   ├── evaluation_manifest.json                # Complete 60-condition split manifest
│   └── artifact_inventory.json                 # SHA-256 hashes and file inventories
└── results/                                    # Machine-readable evaluation tables
    ├── model_comparison_summary.csv            # Consolidated model-level benchmark
    ├── condition_level_results.csv             # 60 speaker-by-budget condition metrics
    ├── per_class_metrics.csv                   # 7-class precision, recall, F1, and support
    ├── speaker_level_results.csv               # 15 individual speaker performances
    ├── statistical_comparisons.csv             # Wilcoxon paired tests & Holm corrections
    ├── runtime_comparison.csv                  # Per-sample latency and wall-clock times
    └── confusion_matrices/                     # Normalized and raw confusion matrices
        ├── confusion_matrix_qsvc_q8_r1.csv
        ├── confusion_matrix_rbf_svm8.csv
        └── confusion_matrix_weighted_adapter8.csv
```

---

## 3. Results File Registry

| File Name | Description | Key Source Columns |
| :--- | :--- | :--- |
| `model_comparison_summary.csv` | Master benchmark table across all 10 evaluated quantum and classical models. | Accuracy, Macro-F1 (7-class), Supported F1 (6-class), Balanced Acc, Emerg Recall, Emerg Precision, FER, Latency. |
| `condition_level_results.csv` | Granular condition metrics across all 60 speaker-by-budget conditions (15 speakers $\times$ 4 budgets). | Speaker_ID, Budget_N, QSVC_Macro_F1, RBF_Macro_F1, Adapter_Macro_F1, Diff_Q8_RBF8. |
| `per_class_metrics.csv` | Performance across the 7 classes of the proxy emergency intent taxonomy. | Intent Class, Support, Support %, Precision, Recall, F1 for QSVC, RBF-SVM, and MLP Adapter. |
| `speaker_level_results.csv` | Etiology and dysarthria severity breakdown across all 15 TORGO speakers. | Speaker, Group, Etiology, Severity, Samples, Q8_F1, RBF_F1, Adapter_F1, Q8_Recall, RBF_Recall, FER. |
| `statistical_comparisons.csv` | Formal non-parametric paired tests and multiplicity corrections. | Comparison Pair, Test Type, Statistic (W), Unadjusted p, Holm-Bonferroni p, 95% CI, Scientific Verdict. |
| `runtime_comparison.csv` | Measured execution times, simulation latency, and scaling ratios. | Model, Feature Dim, Latency per Sample (s), Ratio vs RBF, Total Wall-Clock (s), Hardware/Engine. |

---

## 4. Key Scientific Findings

1. **Parity Finding (CASE B Confirmed):**
   - The primary matched comparison demonstrates **no statistically significant difference** between the quantum kernel classifier (`QSVC Q8-R1`) and the matched classical baseline (`RBF-SVM d=8`):
     - Quantum Macro-F1: $0.1051 \pm 0.0575$ vs. Classical RBF-SVM: $0.1087 \pm 0.0487$ (Wilcoxon $p = 0.1714$, **Holm-Bonferroni adjusted $p = 0.3428$**).
     - Quantum Emergency Recall: $27.52\%$ vs. Classical RBF-SVM: $31.89\%$ (Wilcoxon $p = 0.0230$, **Holm-Bonferroni adjusted $p = 0.0689$**).
   - Conclusion: Theoretical expressive separations in Hilbert space do not translate to empirical performance advantages over well-tuned classical kernels in low-resource speech adaptation.

2. **Multi-Objective Trade-Off in Classical Neural Adapters:**
   - The Weighted MLP Adapter achieved higher emergency recall ($45.81\%$, $p = 0.0032$), but incurred a severe false alarm penalty (False Emergency Rate $49.82\%$ vs. QSVC $28.12\%$, $p = 0.0006$).

3. **Computational Simulation Cost:**
   - CPU statevector simulation of 8-qubit quantum circuits required $1.0816\text{ s}$ per test sample compared to $0.0014\text{ s}$ for RBF-SVM (**$772.6\times$ higher per-sample latency**), emphasizing the classical simulation bottleneck on non-quantum hardware.

4. **Zero-Support Reporting:**
   - The `LOCATION` class has 0 true examples in the locked few-shot test partition (due to extreme corpus sparsity: 21 total samples across the entire 16,552 TORGO recordings). Metrics are accurately reported as `N/A (0 support)` rather than imputed.

---

## 5. Provenance & Reproducibility

- Master Random Seed: `42`
- Feature Extractor: `openai/whisper-tiny` (Frozen, 384 units)
- Dimensionality Reduction: `StandardScaler` + `PCA` ($d=8$, fitted strictly on training partition)
- Quantum Engine: Qiskit 1.0.2 / PennyLane 0.35 (`default.qubit`)
- Split Definition: `splits/torgo_resegregated_v1.json`
