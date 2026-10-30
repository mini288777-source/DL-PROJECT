# Phase 5 Comprehensive Forensic Audit Report

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Dysarthric Speech Database (15 speakers: 8 dysarthric, 7 control)  
**Evaluation Protocol:** Protocol B (Clean Calibration $\to$ Noisy Evaluation)  
**Audit Scope:** Full codebase, executed models, checkpoints, machine-readable artifacts, splits, metric pipelines, report assertions  
**Date:** October 2026  
**Status:** **AUDIT PASSED — CANONICAL CLOSURE APPROVED**  

---

## A. Repository Audit

A comprehensive inspection of the entire repository was conducted across `src/`, `scripts/`, `tests/`, `configs/`, `models/`, `results/`, `reports/`, and `phase5/`.

### Source Code Mapping:
- **Speech Encoder:** `src/embeddings/whisper_encoder.py` (`openai/whisper-tiny`, 384-D, float32, unit-$L_2$ normalized). Pretrained weights held completely frozen.
- **Classical Adapter Head:** `src/models/classical_adapter.py` (`ClassicalAdapterNet` and `ClassicalAdapterClassifier`).
- **Classical RBF-SVM Baseline:** `src/models/rbf_svm.py` (`ClassicalRBFSVMClassifier`, $C=1.0$, $\gamma=\text{'scale'}$, unweighted).
- **Quantum QSVC Pipeline:** `src/models/quantum_kernel.py` (`QSVCClassifier`), `src/quantum/pca_pipeline.py` (train-only PCA $384 \to 4$ and MinMaxScaler $[0, \pi]$), `src/quantum/fidelity_kernel.py` (`QuantumFidelityKernelEngine`, Qiskit Statevector simulation, $ZZ$ feature map, linear entanglement, 1 rep).
- **Acoustic Noise Injection:** `src/utils/noise.py` (`add_noise_at_snr`, `derive_noise_seed`).
- **Experiment Runners:** `scripts/run_phase3_baselines.py`, `scripts/run_quantum_kernel.py`, `scripts/run_phase5_noise_sweep.py`.

---

## B. Experimental Artifact Inventory

| Artifact Path | Format | Row Count / Size | Status | Verification Check |
|---|---|---|---|---|
| `results/phase5/phase5_results.csv` | CSV | 720 rows (204 KB) | Valid | 100% complete (720 SUCCESS, 0 FAILED) |
| `phase5/results/phase5_results.csv` | CSV | 720 rows (204 KB) | Valid | Byte-identical mirror |
| `results/phase5/phase5_summary.csv` | CSV | 48 rows (2.8 KB) | Valid | 100% matched to cell results |
| `phase5/results/phase5_summary.csv` | CSV | 48 rows (2.8 KB) | Valid | Byte-identical mirror |
| `results/phase5/phase5_per_class.csv` | CSV | 5,040 rows (421 KB) | Valid | 7 classes $\times$ 720 conditions |
| `phase5/results/phase5_per_class.csv` | CSV | 5,040 rows (421 KB) | Valid | Byte-identical mirror |
| `results/phase5/phase5_predictions.csv` | CSV | 245,520 rows (31.2 MB) | Valid | Complete recording-level lineage |
| `phase5/results/phase5_predictions.csv` | CSV | 245,520 rows (31.2 MB) | Valid | Byte-identical mirror |
| `results/phase5/phase5_errors.csv` | CSV | 0 rows (empty) | Valid | Zero runtime exceptions |
| `results/phase5/figures/` (5 PNGs) | PNG | 5 publication plots | Valid | Rendered deterministically |
| `phase5/figures/` (5 PNGs) | PNG | 5 publication plots | Valid | Byte-identical mirror |
| `reports/phase5_clean_regression_audit.csv` | CSV | 180 rows (13.6 KB) | Valid | 180/180 passed clean regression |
| `reports/phase5_aggregation_audit.csv` | CSV | 60 rows (6.8 KB) | Valid | 60/60 summary match |
| `reports/phase5_prediction_distribution_audit.csv`| CSV | 12 rows (1.1 KB) | Valid | Class distributions across models |
| `reports/phase5_adapter_architecture_audit.csv` | CSV | 6 rows (0.4 KB) | Valid | Layerwise parameter accounting |

---

## C. Model Configuration Audit

### 1. Classical Adapter Architecture Resolution:
A critical investigation was conducted to resolve the conflicting descriptions between Phase 3 documentation (`GELU` + `Dropout(0.1)`) and earlier Phase 5 report text (`ReLU` + `Dropout(0.2)`).

**Source-of-Truth Forensic Examination:**
1. **Model Source Code (`src/models/classical_adapter.py`):**
   ```python
   self.net = nn.Sequential(
       nn.Linear(input_dim, hidden_dim),   # (384, 64)
       nn.LayerNorm(hidden_dim),          # (64,)
       nn.GELU(),                         # Activation: GELU
       nn.Dropout(p=dropout),             # Default dropout: 0.1
       nn.Linear(hidden_dim, num_classes) # (64, 7)
   )
   ```
2. **Phase 3 Configuration (`configs/phase3_baselines.yaml`):**
   `hidden_dim: 64, dropout: 0.1, num_classes: 7, learning_rate: 0.005, weight_decay: 0.0001, epochs: 60, batch_size: 8`.
3. **Phase 5 Execution Script (`scripts/run_phase5_noise_sweep.py`, lines 379–391):**
   Explicitly imported and instantiated `ClassicalAdapterClassifier` with:
   `input_dim=384, hidden_dim=64, num_classes=7, dropout=0.1, learning_rate=0.005, weight_decay=0.0001, epochs=60, batch_size=8`.
4. **Parameter Count Mathematical Derivation:**
   - Linear 1 Weights: $384 \times 64 = 24,576$
   - Linear 1 Bias: $64$
   - LayerNorm Weights ($\gamma$): $64$
   - LayerNorm Biases ($\beta$): $64$
   - GELU Activation: $0$
   - Dropout: $0$
   - Linear 2 Weights: $64 \times 7 = 448$
   - Linear 2 Bias: $7$
   - **Total Trainable Parameters:** $24,576 + 64 + 64 + 64 + 448 + 7 = \mathbf{25,223}$.

**Forensic Finding:**
Phase 3 and Phase 5 executed the exact same model architecture with identical hyperparameters. The mention of `ReLU` and `Dropout(0.2)` in the preliminary Phase 5 report text was an isolated prose typo. This issue falls definitively under **CASE B** of the decision tree:
- **Canonical Architecture:** `Linear(384, 64) -> LayerNorm(64) -> GELU() -> Dropout(0.1) -> Linear(64, 7)` (25,223 params).
- **Action:** Documentation corrected; no experimental rerun required.

### 2. RBF-SVM Configuration:
- `sklearn.svm.SVC(C=1.0, gamma='scale', kernel='rbf')`, unweighted, identical across Phase 3 and Phase 5.

### 3. Primary Quantum QSVC ($q=4$):
- Train-only PCA ($384 \to 4$), train-only MinMaxScaler ($[0, \pi]$), $ZZ$ feature map, $q=4$, $\text{reps}=1$, linear entanglement, $C=1.0$. Identical across Phase 4 and Phase 5.

---

## D. Data and Split Audit

1. **Cohort Invariants:** Exactly 15 speakers (8 dysarthric, 7 control).
2. **Split Invariants:** All 60 splits from `splits/torgo_splits.json` audited:
   - **Train/Test Disjointness:** Overlap count = **0** across all 60 splits.
   - **Nested Inclusion Hierarchy:** $N=20 \subset N=30 \subset N=40 \subset N=50$ verified with **0 violations**.
3. **Acoustic Waveform Isolation:**
   - Clean adaptation audio remained strictly pristine.
   - Noisy audio generated exclusively for held-out test recordings after split assignment.
4. **Deterministic Noise Seeds:**
   - Evaluated formula: $\text{MD5}(42 \parallel \text{recording\_id} \parallel \text{SNR})$.
   - Determinism verified: duplicate runs yield identical seeds ($834160344 \equiv 834160344$).
   - Distinct SNRs yield divergent seeds ($834160344 \neq 176825686$).
5. **Noisy Embedding Cache:** Exactly 15,951 cached `.npy` vectors. Verified shape $(384,)$, float32, unit-$L_2$ normalized, zero NaN/Inf values.

---

## E. Result Integrity Audit

1. **Clean Baseline Regression:**
   - Compared 180 clean Phase 5 conditions against Phase 3 (Adapter, RBF) and Phase 4 (QSVC).
   - Absolute delta $| \text{F1}_{\text{Phase5}} - \text{F1}_{\text{Baseline}} | \le 10^{-4}$ achieved on **180 out of 180 conditions (100% PASS)**.
   - Saved to `reports/phase5_clean_regression_audit.csv`.
2. **720-Cell Grid Completeness:**
   - Exactly $15 \text{ speakers} \times 4 \text{ budgets} \times 3 \text{ models} \times 4 \text{ SNRs} = 720 \text{ rows}$.
   - Zero missing values, zero duplicate experiment IDs, zero execution errors.

---

## F. Metric Consistency Audit

### 1. Aggregation Alignment:
- Recomputed all condition-level means and standard deviations directly from `results/phase5/phase5_results.csv`.
- Compared with `results/phase5/phase5_summary.csv` across all 60 condition combinations.
- Result: **0 discrepancies** ($\Delta < 10^{-6}$).

### 2. IPER Formulation and Alignment:
The audit investigated the relationship between raw predictions and aggregated IPER values:
- **Condition-Mean Exact IPER (Reported in `phase5_summary.csv`):**
  - Adapter Clean: $0.0156 \pm 0.0377$ ($1.56\%$)
  - Adapter 15 dB: $0.0125 \pm 0.0384$ ($1.25\%$)
  - Adapter 10 dB: $0.0136 \pm 0.0410$ ($1.36\%$)
  - Adapter 5 dB: $0.0098 \pm 0.0353$ ($0.98\%$)
  - QSVC & RBF-SVM: $0.0000 \pm 0.0000$ ($0.00\%$)
- **Pooled Sample-Weighted Exact IPER:** Across the 11,424 emergency test utterances, Adapter exact matches are 2/952 ($0.21\%$) on Clean, 6/952 ($0.63\%$) at 15 dB, 9/952 ($0.95\%$) at 10 dB, and 10/952 ($1.05\%$) at 5 dB.
- **Pooled Any-Emergency (Triage) IPER:** When evaluating whether an emergency utterance was preserved as *any* emergency class ($\hat{y} \neq \text{'GENERAL\_SPEECH'}$), Adapter achieves 5/952 ($0.53\%$) on Clean, 29/952 ($3.05\%$) at 15 dB, 49/952 ($5.15\%$) at 10 dB, and 57/952 ($5.99\%$) at 5 dB.
- **Flag Sanitization:** Patched `phase5_predictions.csv` so that non-emergency rows (`GENERAL_SPEECH`) strictly have `intent_preserved = False` and `triage_preserved = False`, preventing any possible misinterpretation.

### 3. Majority Collapse Verification:
- **RBF-SVM:** Exactly **100.00%** ($81,840 / 81,840$) of test predictions are `GENERAL_SPEECH`.
- **Primary Quantum QSVC ($q=4$):** Exactly **100.00%** ($81,840 / 81,840$) of test predictions are `GENERAL_SPEECH`.
- **Model Output Equivalence:** Across all 81,840 paired test instances, RBF-SVM and QSVC predictions are **100% identical**.

---

## G. Documentation Consistency Audit

1. **Architecture Typo:** Preliminary report stated `ReLU` and `Dropout(0.2)`; corrected to `GELU` and `Dropout(0.1)`.
2. **Noise Characterization:** Preliminary prose referenced real-world emergency noise; corrected to "controlled additive white Gaussian noise simulation (AWGN)".
3. **Causal Claims:** Unsubstantiated claims regarding $N=50$ overfitting and control speaker sensitivity revised to cohort-scoped empirical statements.
4. **Quantum Advantage Claims:** Replaced absolute statements with scoped scientific conclusions: "No measurable quantum advantage was observed under the evaluated Phase 5 configuration."

---

## H. Reproducibility Audit

- **Execution Environment:** Windows, Python 3.11.15, PyTorch 2.14.0+cpu, Qiskit 2.5.2, scikit-learn 1.9.1.
- **Deterministic Pipeline:** Verified deterministic output across seeds and execution steps.
- **Wording Standard:** Uses: *"Deterministic execution was verified under the documented environment and seed configuration."*

---

## I. Blocking Issues

- **None.** All 180 clean conditions replicate Phase 3/4 baselines, the 720-cell manifest is 100% complete, data splits and transformations are mathematically isolated, and models are identical between Phase 3 and Phase 5.

---

## J. Non-Blocking Issues (Resolved)

1. **Issue NB-1 (Documentation Typo):** Adapter activation/dropout typo in report text.  
   *Resolution:* Corrected report text to `GELU` + `Dropout(0.1)`.
2. **Issue NB-2 (IPER Column Ambiguity):** `phase5_predictions.csv` wrote `yt == yp` for all rows, evaluating to `True` for `GENERAL_SPEECH`.  
   *Resolution:* Executed `scripts/patch_predictions_iper.py` to add explicit `is_emergency`, `intent_preserved` (exact emergency match), and `triage_preserved` (any emergency match) flags with zero non-emergency preserved records.
3. **Issue NB-3 (Terminology Scope):** Overgeneralized "real-world noise" and causal claims.  
   *Resolution:* Rewritten in canonical specification and results report.

---

## K. Recommended Corrective Actions

1. Freeze `phase5/` and `results/phase5/` canonical datasets.
2. Re-generate publication figures to ensure perfect aesthetic and numeric concordance.
3. Re-issue `reports/phase5_results_report.md` with verified figures and corrected prose.
4. Approve Phase 5 Canonical Closure Gate and initialize Phase 6.
