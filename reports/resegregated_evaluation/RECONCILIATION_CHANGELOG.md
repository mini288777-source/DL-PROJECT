# RECONCILIATION CHANGELOG — RESEGREGATED MODEL EVALUATION EVIDENCE REVISION (v2 -> Reconciled Final)

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Team Name:** APPY FIZZ  
**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai Campus  
**Date:** 12 October 2026  
**Auditor / Engineer:** Antigravity Research Team  
**Scope:** Resegregated Model Evaluation Package (`reports/resegregated_evaluation/`)  

---

## 1. Executive Summary of Reconciled Corrections

This changelog records the complete experimental and mathematical reconciliation performed across the resegregated evaluation artifacts. Prior audit snapshots (`audit_v1_snapshot` and `audit_v2_snapshot`) have been permanently preserved in the repository as immutable forensic baselines. 

Four primary systemic issues and twelve dependent inconsistencies were audited, mathematically recomputed from the primary prediction ledger (`predictions_resegregated_evaluation.csv`, 191,412 rows), verified against condition-level results, and incorporated into updated tables, figures, CSVs, and reports.

---

## 2. Issue-by-Issue Forensic Reconciliation

### Issue 1: Few-Shot Calibration Sample-Budget Scaling Trend and Narrative (Figure 17.7 & Section 12)

- **Defect Description:** The previous Section 12 narrative and draft reconciliation summary reported provisional condition-level means for QSVC Q8-R1 (e.g., N=20: Macro-F1 = 0.0725, Recall = 54.00%, FER = 53.42%) that conflicted with the actual underlying condition-level results CSV. Furthermore, the narrative lacked a mathematical explanation for why expanding the few-shot budget $N$ reduces emergency recall while improving Macro-F1 and false alarm rejection.
- **Authoritative Data Source:** `reports/resegregated_evaluation/results/condition_level_results.csv` and `predictions_resegregated_evaluation.csv`.
- **Mathematical Calculation:** For each calibration budget $N \in \{20, 30, 40, 50\}$, the 15 speaker-specific conditions were averaged unweighted for both QSVC Q8-R1 and Matched Classical RBF-SVM:
  $$\text{Mean Metric}(N) = \frac{1}{15} \sum_{s=1}^{15} M_{s, N}$$
- **Verified Replacement Values:**
  - **QSVC Q8-R1 (Quantum):**
    - $N = 20$: Macro-F1 = **$0.0767 \pm 0.0659$**, Emergency Recall = **$49.33\% \pm 50.07\%$**, False Emergency Rate (FER) = **$50.15\% \pm 48.81\%$**
    - $N = 30$: Macro-F1 = **$0.1109 \pm 0.0523$**, Emergency Recall = **$26.67\% \pm 45.77\%$**, FER = **$25.09\% \pm 42.19\%$**
    - $N = 40$: Macro-F1 = **$0.1258 \pm 0.0512$**, Emergency Recall = **$14.07\% \pm 35.00\%$**, FER = **$14.19\% \pm 34.85\%$**
    - $N = 50$: Macro-F1 = **$0.1266 \pm 0.0502$**, Emergency Recall = **$14.07\% \pm 35.00\%$**, FER = **$13.64\% \pm 34.40\%$**
  - **Matched Classical RBF-SVM (Comparator):**
    - $N = 20$: Macro-F1 = **$0.0720 \pm 0.0672$**, Emergency Recall = **$53.33\% \pm 51.64\%$**, FER = **$52.39\% \pm 50.62\%$**
    - $N = 30$: Macro-F1 = **$0.1070 \pm 0.0584$**, Emergency Recall = **$26.67\% \pm 45.77\%$**, FER = **$26.36\% \pm 45.05\%$**
    - $N = 40$: Macro-F1 = **$0.1273 \pm 0.0533$**, Emergency Recall = **$14.07\% \pm 35.00\%$**, FER = **$13.43\% \pm 35.15\%$**
    - $N = 50$: Macro-F1 = **$0.1226 \pm 0.0482$**, Emergency Recall = **$13.33\% \pm 35.19\%$**, FER = **$13.36\% \pm 35.11\%$**
- **Physical / Optimization Mechanism Documented:**
  In dysarthric speech under 745:1 class imbalance, calibration budget expansion strictly adds non-emergency baseline utterances (GENERAL_SPEECH), as distress prompts are naturally constrained. At $N=20$, the decision hyperplane is underspecified, classifying ambiguous speech as emergencies (high recall ~50%, but disastrous false alarm rate >50%). Expanding $N$ to 50 provides dense coverage of the non-emergency acoustic manifold. The convex dual quadratic optimizer pushes the support vector margin sharply outward, collapsing false alarms (FER drops by ~37 percentage points to 13.64%). Because false alarms collapse, precision on GENERAL_SPEECH approaches near-perfection, driving 7-class Macro-F1 up from 0.0767 to 0.1266. However, this conservative shift causes ambiguous emergency utterances to be classified as non-emergency, dropping emergency recall to 14.07%. Because this identical trade-off occurs in both quantum and classical models, it is an intrinsic consequence of support vector margin optimization under extreme class imbalance, not a quantum-specific phenomenon.
- **Affected Artifacts:**
  - Created `results/sample_budget_condition_means.csv`.
  - Regenerated Figure 17.7 (`fig7_sample_budget_scaling.png` & `.svg`) showing 3 panels with quantum and classical curves.
  - Updated Section 12, Table 12.1, executive summaries, and conclusion in Word, PDF, and Markdown reports.

---

### Issue 2: Separation of Pooled Test-Set Confusion Matrices by Sample Budget (Figure 17.10 & Results Directory)

- **Defect Description:** Previous evaluation summaries did not make the calibration budget explicit for pooled confusion matrices, or risked conflating predictions across multiple sample budgets into a single 5,317-item matrix.
- **Authoritative Data Source:** `predictions_resegregated_evaluation.csv` filtered strictly by `(model_name, sample_budget_n)`.
- **Mathematical Calculation:** For each of the 9 models and each of the 4 budgets $N \in \{20, 30, 40, 50\}$, predictions from all 15 speakers were pooled across the locked 5,317 test samples.
  - Class support verified across all 36 confusion matrices:
    - GENERAL_SPEECH: 5,219
    - CRITICAL_RESPONSE: 35
    - FALL: 21
    - BREATHING_MEDICAL: 18
    - PAIN: 13
    - SOS_HELP: 11
    - LOCATION: 0
    - Total: 5,317
- **Verified Replacement Artifacts:**
  - 36 distinct confusion matrix CSV files generated in `results/confusion_matrices/`:
    `confusion_matrix_{model}_N{budget}.csv` for all 9 models $\times$ 4 budgets.
  - Dedicated pooled model comparison CSV created:
    `results/model_comparison_pooled_by_budget.csv` containing all 36 budget-specific records.
  - Regenerated Figure 17.10 (`fig10_confusion_matrices.png` & `.svg`) explicitly displaying the locked $N=50$ confusion matrices for QSVC Q8-R1 and Matched RBF-SVM.

---

### Issue 3: Disentanglement of Master Model Comparison Table (Dual Estimands: Table A & Table B)

- **Defect Description:** The master benchmark table previously conflated or omitted aggregation methods, mixing condition-level means with pooled test-set metrics without distinguishing their underlying populations.
- **Authoritative Resolution:** Created two dedicated tables:
  1. **Table A (Table 8.1 / `model_comparison_condition_level.csv`):** Condition-level mean $\pm$ standard deviation across the 60 speaker-by-budget conditions. Gives equal weight to every speaker-budget pair.
     - QSVC Q8-R1: Accuracy = $72.35\% \pm 40.80\%$, Macro-F1 = $0.1100 \pm 0.0576$, Recall = $26.04\% \pm 43.41\%$, FER = $25.77\% \pm 42.18\%$.
     - Matched RBF-SVM: Accuracy = $71.75\% \pm 42.51\%$, Macro-F1 = $0.1072 \pm 0.0599$, Recall = $26.85\% \pm 44.51\%$, FER = $26.39\% \pm 43.96\%$.
  2. **Table B (Table 8.2 / `model_comparison_pooled_by_budget.csv`):** Budget-specific pooled metrics on the locked 5,317 test samples for $N=20, 30, 40, 50$. Weights speakers proportionally to test cohort size.
     - At $N=50$:
       - QSVC Q8-R1: Accuracy = $63.72\%$, 7-Class Macro-F1 = $0.1159$, Supported 6-Class F1 = $0.1352$, Multiclass Balanced Accuracy = $0.1582$, Recall = $21.43\%$, FER = $35.20\%$.
       - Matched RBF-SVM: Accuracy = $63.51\%$, 7-Class Macro-F1 = $0.1145$, Supported 6-Class F1 = $0.1336$, Multiclass Balanced Accuracy = $0.1499$, Recall = $20.41\%$, FER = $35.39\%$.
- **Discrepancy Explained:**
  Speakers with larger test partitions (e.g., MC04 with 943 samples, FC02 with 925 samples) exhibit lower classification accuracy (~55–60%) than speakers with smaller cohorts (e.g., F04 with 200 samples, 98% accuracy). Averaging conditions equally yields ~72%, whereas pooling all 5,317 samples weights speakers by cohort size, yielding ~63.6%. Both models track within 0.6% across both estimands, confirming empirical parity.

---

### Issue 4: Balanced Accuracy and Class Support Formal Definitions (Figure 17.6 & Section 10)

- **Defect Description:** Figure 17.6 plotted balanced accuracy around ~0.50, while pooled multiclass balanced accuracy was reported as ~0.15–0.16. The report lacked formal mathematical definitions of the differing class support policies.
- **Authoritative Resolution:** Derived and documented three distinct balanced accuracy metrics:
  1. **Stage-1 Binary Balanced Accuracy (Upstream Gate):**
     $$\text{BA}_{\text{binary}} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{\text{Recall}_{\text{emerg}} + (1 - \text{FER})}{2}$$
     - Condition Mean: **$0.5014 \pm 0.0183$** (QSVC Q8) vs. **$0.5023 \pm 0.0115$** (Matched RBF).
  2. **Pooled Multiclass Supported 6-Class Balanced Accuracy:**
     $$\text{BA}_{\text{multiclass, supported}} = \frac{1}{6} \sum_{c \in \text{Supported}} \text{Recall}_c$$
     - Evaluates macro-recall strictly over the 6 classes with true support in the test set (excluding LOCATION which has $N=0$).
     - At $N=50$: **$0.1582$** (QSVC Q8) vs. **$0.1499$** (Matched RBF).
  3. **Canonical 7-Class Macro Recall (Zero-Support Policy):**
     $$\text{BA}_{\text{7-class, zero}} = \frac{1}{7} \left( \sum_{c=1}^6 \text{Recall}_c + 0 \right) = \frac{6}{7} \times \text{BA}_{\text{multiclass, supported}}$$
     - At $N=50$: **$0.1356$** (QSVC Q8) vs. **$0.1285$** (Matched RBF).
- **Affected Artifacts:**
  - Regenerated Figure 17.6 (`fig6_balanced_accuracy.png` & `.svg`) showing dual bars for Binary Stage-1 (50%) and Multiclass Supported (15.8%).
  - Updated Section 10, Table 8.1, Table 8.2, and `model_comparison_summary.csv`.

---

## 3. Comprehensive Artifact Reconciliation Registry

| Artifact Path | Previous State | Reconciled State | Verification Method |
|---|---|---|---|
| `results/model_comparison_condition_level.csv` | Missing / Partial | Complete Table A (60 conditions, mean $\pm$ std) | Calculated from `condition_level_results.csv` |
| `results/model_comparison_pooled_by_budget.csv` | Missing | Complete Table B (36 records, N=20..50) | Calculated from `predictions_resegregated_evaluation.csv` |
| `results/sample_budget_condition_means.csv` | Missing | Reconciled condition-means across budgets | Unweighted mean over 15 speakers per budget |
| `results/model_comparison_summary.csv` | Mixed estimands | Dual-estimand consolidated benchmark | Derived from Table A and Table B |
| `results/confusion_matrices/*.csv` | Single / unspecified budget | 36 distinct budget-specific matrices ($5,317$ items) | Direct cross-tabulation of prediction ledger |
| `figures/fig1_dataset_distribution.*` | Stale | Corpus (16,552) vs. Test (5,317) | Reconciled against split manifest |
| `figures/fig2_overall_accuracy.*` | Stale | Condition-mean vs. Pooled N=50 | Programmatic regeneration |
| `figures/fig3_macro_f1_comparison.*` | Stale | Canonical 7-class vs. Supported 6-class | Programmatic regeneration |
| `figures/fig4_emergency_recall_vs_precision.*` | Stale | Sensitivity vs. precision trade-off | Programmatic regeneration |
| `figures/fig5_false_emergency_rate.*` | Stale | FER with 15% safety threshold | Programmatic regeneration |
| `figures/fig6_balanced_accuracy.*` | Ambiguous ~0.50 | Binary Stage-1 vs. Multiclass Supported | Programmatic regeneration |
| `figures/fig7_sample_budget_scaling.*` | Conflicting values | 3-panel quantum vs. classical scaling curves | Programmatic regeneration from Table 12.1 |
| `figures/fig8_speaker_variation.*` | Stale | 15 audited TORGO speakers | Programmatic regeneration |
| `figures/fig9_quantum_vs_rbf_matched.*` | Stale | Paired scatter & diff histogram ($p=0.5698$) | Full-precision Wilcoxon paired records |
| `figures/fig10_confusion_matrices.*` | Single matrix | N=50 locked 5,317 test item matrices | Programmatic regeneration |
| `figures/fig11_per_class_performance.*` | Stale | Per-class sensitivity with true support counts | Programmatic regeneration |
| `figures/fig12_runtime_latency.*` | Stale | 772.6x CPU simulation gap | Measured benchmark records |
| `RESEGREGATED_MODEL_COMPARISON_REPORT.md` | Inconsistent | Fully reconciled Markdown report | Compiled via builder script |
| `RESEGREGATED_MODEL_COMPARISON_REPORT.docx` | Inconsistent | Formatted Word document (15 tables, 12 figures) | Word COM automation export |
| `RESEGREGATED_MODEL_COMPARISON_REPORT.pdf` | Inconsistent | Reconciled publication PDF | Word COM PDF engine export |

---

## 4. Preservation of Immutable Audit Snapshots

In accordance with strict data-provenance standards:
1. `audit_v1_snapshot/`: Contains historical Phase 8 audit artifacts.
2. `audit_v2_snapshot/`: Preserves the pre-reconciliation state of `RESEGREGATED_MODEL_COMPARISON_REPORT.pdf`, `.docx`, `.md`, summary CSVs, and figures prior to this revision.
3. No historical raw prediction records or dataset checkpoints were altered. All recomputations operated directly on the authoritative 191,412-row prediction ledger.
