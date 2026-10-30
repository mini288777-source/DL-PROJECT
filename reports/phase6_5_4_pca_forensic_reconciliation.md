# Subphase 6.5.4: Forensic PCA Reconciliation Report
## Resolution of the PCA Cumulative Explained Variance Discrepancy

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 5, 2026  
**Status:** DISCREPANCY FULLY RESOLVED  
**Reconciliation Decision:** **CASE B** (One value was incorrectly reported in a summary/plotting script, while the underlying model execution and cached metadata were 100% mathematically correct and consistent with empirical reality).

---

## 1. Executive Summary of the Discrepancy

During Phase 6.5 forensic audits, an apparent contradiction was identified between two authoritative project documents:

1. **Phase 4/4.5 Milestone Reports (`reports/phase4_complete_results.md:150`):**
   $$\text{PCA-4 Cumulative Explained Variance} \approx 27.4\% \pm 3.9\%$$
2. **Phase 6.5.0 Diagnostic Report (`phase6_5/reports/phase6_5_0_diagnostic_report.md:224`):**
   $$\text{PCA-4 Cumulative Explained Variance} \approx 73.25\%$$

This discrepancy represented an apparent ~46 percentage-point difference in the perceived information retention of the 4-qubit quantum feature map ($q = 4$).

**Forensic Finding:**
- The actual model execution in Phase 4 (`run_quantum_kernel.py`) used `TrainOnlyPCAPipeline` (`src/quantum/pca_pipeline.py`), which correctly fitted `PCA(n_components=4, random_state=42)` on each speaker's adaptation set. The empirical cumulative explained variance was recorded at runtime into 212 JSON metadata files in `results/phase4/kernel_cache/*_meta.json`.
- The actual empirical mean recorded across all 53 primary Phase 4 runs was **$79.15\% \pm 5.26\%$** (min: 67.92%, max: 89.53%).
- However, during the compilation of the Phase 4 visualization and summary tables, `scripts/generate_phase4_reports.py` (line 308) hardcoded an arbitrary synthetic placeholder list:
  ```python
  pca_variances = [17.8, 27.4, 34.2, 39.8] # mean cumulative variance in %
  pca_vars_std = [3.1, 3.9, 4.2, 4.5]
  ```
  These hardcoded values were copied into the markdown tables of `reports/phase4_complete_results.md` and `reports/phase4_ablation_report.md`, creating the illusion that PCA-4 retained only 27.4% variance.
- When Phase 6.5.0 recomputed PCA on 500 randomly sampled Whisper embeddings from the corpus, it obtained **73.25%**, which closely matches the true empirical reality of Whisper representations.
- When recomputed across all 60 speaker-specific adaptation splits in Track B, the verified PCA-4 cumulative explained variance is **$80.66\% \pm 4.37\%$**.

---

## 2. Provenance and Artifact Investigation Table

| Artifact | File Location | PCA Dims | Input Space | Fit Data Protocol | Scaling | Reported Metric | Result | Forensic Classification |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| **Phase 4/4.5 Report Tables** | `reports/phase4_complete_results.md:150` | 4 | 384-D Whisper | Hardcoded in `scripts/generate_phase4_reports.py:308` | `MinMaxScaler(0, pi)` | Reported Cumulative Variance | $27.4\% \pm 3.9\%$ | **ERRONEOUS_REPORT_ARTIFACT** |
| **Phase 4 Actual Kernel Metadata** | `results/phase4/kernel_cache/*_q4_r1_meta.json` | 4 | 384-D Whisper | Adaptation sets ($N \in \{20, 30, 40, 50\}$, Phase 4 splits) | `MinMaxScaler(0, pi)` | Runtime `pca_cumulative_explained_variance` | **$79.15\% \pm 5.26\%$** (range: 67.92%–89.53%) | **VERIFIED_EMPIRICAL_EXECUTION** |
| **Phase 6.5.0 Diagnostic Report** | `phase6_5/reports/phase6_5_0_diagnostic_report.md:224` | 4 | 384-D Whisper | 500 pooled random samples from TORGO corpus | None (L2 unit norm) | Cumulative Explained Variance | **$73.25\%$** (~26.75% lost) | **VERIFIED_EMPIRICAL_POOLED** |
| **Track B Recovery Splits (Subphase 6.5.4 Recomputed)** | `results/phase6_5_4_pca_variance.csv` | 4 | 384-D Whisper | Per-speaker adaptation sets (60 conditions, Track B) | None (L2 unit norm) | Cumulative Explained Variance | **$80.66\% \pm 4.37\%$** (range: 71.64%–89.38%) | **VERIFIED_EMPIRICAL_TRACK_B** |

---

## 3. Forensic Trace of the Hardcoded Values

In `scripts/generate_phase4_reports.py` (lines 305–319):
```python
# 3. Dedicated Qubit Count Ablation Figure (q=2,4,6,8 vs Variance & Runtime)
fig, (ax_var, ax_time) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
qubits = [2, 4, 6, 8]
pca_variances = [17.8, 27.4, 34.2, 39.8] # mean cumulative variance in %
pca_vars_std = [3.1, 3.9, 4.2, 4.5]

ax_var.bar(qubits, pca_variances, yerr=pca_vars_std, capsize=4, ...)
```

Notice the pattern:
- $q=2 \implies 17.8\%$
- $q=4 \implies 27.4\%$
- $q=6 \implies 34.2\%$
- $q=8 \implies 39.8\%$

The script author hardcoded this array rather than dynamically loading the actual `pca_cumulative_explained_variance` stored in `results/phase4/kernel_cache/*_meta.json`. 

When the actual metadata files saved by `run_quantum_kernel.py` during Phase 4 are queried across all 212 runs:
- **$q=2$:** Mean = **$67.48\% \pm 8.39\%$**
- **$q=4$:** Mean = **$79.15\% \pm 5.26\%$**
- **$q=6$:** Mean = **$84.41\% \pm 4.15\%$**
- **$q=8$:** Mean = **$87.91\% \pm 3.56\%$**

Thus, the Phase 4 model execution never used or experienced 27.4% variance. The model operated with ~79% of the representation variance retained.

---

## 4. Recomputed Empirical Distribution across All Splits

To establish definitive scientific closure, PCA explained variance was systematically computed from the canonical Whisper embeddings across all adaptation sets:

### 4.1 Original Phase 4/5 Split (`splits/torgo_splits.json`, 60 conditions)
- $d=2$: Mean = **$67.69\% \pm 8.27\%$** (min: 49.52%, max: 80.02%)
- $d=4$: Mean = **$79.43\% \pm 5.09\%$** (min: 67.92%, max: 89.53%)
- $d=6$: Mean = **$84.66\% \pm 4.00\%$** (min: 75.27%, max: 92.63%)
- $d=8$: Mean = **$88.18\% \pm 3.45\%$** (min: 80.38%, max: 94.85%)

### 4.2 Track B Recovery Split (`phase6_5/splits/recovery_splits_hierarchical_ready.json`, 60 conditions)
- $d=2$: Mean = **$69.86\% \pm 6.16\%$** (min: 56.19%, max: 82.06%)
- $d=4$: Mean = **$80.66\% \pm 4.37\%$** (min: 71.64%, max: 89.38%)
- $d=6$: Mean = **$86.16\% \pm 3.48\%$** (min: 78.92%, max: 93.28%)
- $d=8$: Mean = **$89.73\% \pm 2.99\%$** (min: 83.59%, max: 95.80%)

### 4.3 Why Pooled Corpus PCA (73.25%) Differs Slightly from Per-Speaker PCA (80.66%)
- **Pooled Corpus PCA ($N=500$, cross-speaker):** 500 utterances randomly drawn across 15 different speakers exhibit inter-speaker acoustic diversity. The cross-speaker variance dilutes the leading components, requiring slightly more dimensions to capture total variance ($d=4 \implies 73.25\%$).
- **Per-Speaker Adaptation PCA ($N \le 50$, single-speaker):** Adaptation data for a single speaker resides on a more constrained, speaker-specific phonetic manifold. Consequently, the leading principal components capture a higher fraction of the within-speaker variance ($d=4 \implies 80.66\%$).

Both numbers are empirically correct and represent mathematically well-defined populations.

---

## 5. Formal Decision Gate

**Discrepancy Category:** **CASE B**  
*"One value (27.4%) was incorrectly reported in a summary/plotting script, but the underlying model execution and implementation were 100% correct and consistent with empirical PCA variance."*

- The true, verified cumulative explained variance for PCA-4 under the recovery protocol is **$80.66\% \pm 4.37\%$**.
- No implementation repair is required for the PCA pipeline.
- The erroneous reference to 27.4% in historical reports is formally documented and superseded by this forensic reconciliation.
