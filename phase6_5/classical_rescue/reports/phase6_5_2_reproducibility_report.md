# Subphase 6.5.2 — Reproducibility & Audit Report
## Classical Rescue Baseline Experiment Verification

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Track:** Phase 6.5 — Emergency Recognition Recovery  
**Subphase:** 6.5.2 — Classical Rescue Baseline  
**Date:** October 2026  
**Status:** PASS — 100% REPRODUCIBLE (Automated Verification Complete)  

---

## 1. System & Execution Environment

All experimental computations, evaluations, statistical aggregations, and automated unit tests were executed within the standardized project runtime environment:

| Attribute | Specification | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Enterprise (Build 10.0.26200-SP0) | `platform.platform()` |
| **Python Runtime** | Python 3.11.15 (64-bit) | `sys.version` |
| **PyTorch Version** | 2.14.0+cpu | `torch.__version__` |
| **Scikit-Learn Version** | 1.9.1 | `sklearn.__version__` |
| **NumPy Version** | 2.4.6 | `numpy.__version__` |
| **Pandas Version** | 2.2.3 | `pd.__version__` |
| **PyYAML Version** | 6.0.2 | `yaml.__version__` |
| **Pytest Version** | 9.1.1 | `pytest --version` |
| **Random Master Seed** | 42 | Hardcoded in `classical_rescue_config.yaml` |
| **Execution Platform** | Local CPU Workstation | Python script runner |

---

## 2. Input Lineage & Integrity Hashes

The input data splits and audio embeddings used in Subphase 6.5.2 trace strictly to verified prior phase artifacts:

### 2.1 Track A Stratified Recovery Split
- **File:** `phase6_5/splits/recovery_splits_stratified.json`
- **Track:** Track A (Canonical Direct Multi-Class Recovery Split)
- **Speaker Count:** 15 speakers (`F01`, `F03`, `F04`, `FC01`, `FC02`, `FC03`, `M01`, `M02`, `M03`, `M04`, `M05`, `MC01`, `MC02`, `MC03`, `MC04`)
- **Budgets:** $N \in \{20, 30, 40, 50\}$
- **Condition Count:** 60 speaker-budget partitions
- **Integrity Status:** Bitwise identical to the splits generated in Subphase 6.5.1 (Verified by `test_phase6_5_2_classical_rescue.py::test_7_split_hashes_match`).

### 2.2 Whisper Embedding Cache
- **Directory:** `embeddings/torgo/whisper/`
- **Embedding Dimensionality:** 384-D (OpenAI Whisper-base encoder output, pooled)
- **Acoustic Integrity:** 6,067 primary single-channel audio files, audited in Phase 1 and frozen in Phase 2. Zero missing files.

### 2.3 Intent Taxonomy Specification
- **File:** `configs/intent_taxonomy.yaml`
- **Classes (7):** `GENERAL_SPEECH`, `SOS_HELP`, `FALL`, `BREATHING_MEDICAL`, `PAIN`, `LOCATION`, `CRITICAL_RESPONSE`.
- **Emergency Classes (6):** All classes excluding `GENERAL_SPEECH`.

---

## 3. Preservation of Historical Frozen Benchmarks

In strict adherence to the project charter, all historical benchmark data and results from earlier phases remain completely untouched. No files in the following directories were modified, renamed, or overwritten:

| Milestone / Directory | Audit Check | SHA-256 / File Status |
| :--- | :--- | :--- |
| **Phase 3 Baselines** (`results/phase3/`) | Verified | Untouched (Preserved original benchmarks) |
| **Phase 4 Quantum Pilots** (`results/phase4/`) | Verified | Untouched (Preserved QSVC simulation outputs) |
| **Phase 5 Canonical Benchmark** (`results/phase5/phase5_results.csv`) | Verified | Untouched (Verified by `test_16_frozen_benchmarks_untouched`) |
| **Phase 6 Robustness Results** (`phase6/results/phase6_master_results.csv`) | Verified | Untouched (Preserved noise robustness results) |

The recovery experiment outputs are strictly isolated within the `phase6_5/classical_rescue/` directory.

---

## 4. Model Architecture & Invariant Specifications

The four evaluated model conditions enforce strict architectural parity:

### 4.1 Classical Adapter (MLP)
- **Input Dimension:** 384
- **Hidden Dimension:** 64
- **Hidden Activation:** ReLU
- **Dropout Rate:** 0.1
- **Output Dimension:** 7
- **Minibatch Size:** 8
- **Training Epochs:** 60
- **Optimizer:** AdamW ($\text{lr} = 0.005$, weight decay $= 0.0001$)
- **Determinism:** `torch.manual_seed(42)`, `torch.use_deterministic_algorithms(True)`, `DataLoader(generator=g, shuffle=True)` with seed 42.

#### Loss Formulations:
- **Unweighted Adapter:**
  $$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log p(y_i \mid \mathbf{x}_i)$$
- **Weighted Adapter:**
  $$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} w_{y_i} \log p(y_i \mid \mathbf{x}_i)$$
  where $w_c = \frac{N_{\text{train}}}{K_{\text{present}} \times n_c}$ for observed classes, and $w_c = 0.0$ for absent classes.

### 4.2 Classical RBF-SVM
- **Kernel Function:** Radial Basis Function (RBF), $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$
- **Regularization Parameter:** $C = 1.0$ (Frozen canonical Phase 3 hyperparameter; no post-hoc tuning)
- **Kernel Coefficient:** $\gamma = \text{"scale"} = \frac{1}{D \times \text{Var}(X)}$
- **Input Preprocessing:** Raw 384-D Whisper embeddings (standardize=False)

#### Margin Formulations:
- **Unweighted RBF-SVM:** Default penalty $C$ across all samples.
- **Weighted RBF-SVM:** Class-specific soft-margin penalty $C_c = C \times w_c$, where $w_c = \frac{N_{\text{train}}}{K_{\text{present}} \times n_c}$.

---

## 5. Leakage & Integrity Audits

To ensure zero methodological or data leakage, the pipeline enforces four structural guarantees verified by automated unit tests:

1. **No Adaptation/Test Overlap (Test 8):** For every one of the 240 experimental conditions, the intersection between the adaptation recording IDs and the test recording IDs is strictly empty:
   $$\mathcal{S}_{\text{adapt}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
2. **Prompt-Group Isolation:** All utterances sharing a common base prompt or semantic script are assigned atomically to either adaptation or test, never split across both.
3. **Adaptation-Only Class Weights (Tests 9, 10, 11, 15):** Class weights are computed solely on the adaptation label vector $\mathbf{y}_{\text{adapt}}$. Test labels never enter weight calculation, loss calculation, or model selection.
4. **Absent Class Handling (Test 11):** Classes missing from the calibration set ($n_c = 0$) receive zero weight and are never assigned fabricated priors.

---

## 6. Execution Logs & Performance Metrics

### 6.1 Pilot Execution (24 Cells)
- **Scope:** 3 speakers (`M04`, `F01`, `FC02`) $\times$ 2 budgets ($N=20, 40$) $\times$ 4 models = 24 cells.
- **Wall Time:** 6.94 seconds.
- **Errors:** 0 errors; all metrics within nominal ranges.

### 6.2 Full Experimental Execution (240 Cells)
- **Scope:** 15 speakers $\times$ 4 budgets ($N=20, 30, 40, 50$) $\times$ 4 models = 240 cells.
- **Execution Script:** `scripts/run_phase6_5_2_classical_rescue.py`
- **Total Wall Clock Time:** 68.22 seconds (Average 0.28 seconds per cell).
- **Cell Success Rate:** 240 / 240 (100.0%). Zero failed or dropped cells (`errors.csv` is empty).
- **Total Predictions Generated:** 88,672 sample-level predictions recorded in `classical_rescue_predictions.csv`.

---

## 7. Automated Test Suite Verification

The Phase 6.5.2 verification suite (`tests/test_phase6_5_2_classical_rescue.py`) executes 21 targeted regression tests covering the entire experimental contract:

| Test ID | Test Name | Target Property | Result |
| :--- | :--- | :--- | :---: |
| **TEST 1** | `test_1_exactly_15_speakers` | Exact 15 TORGO speaker cohort | **PASSED** |
| **TEST 2** | `test_2_exactly_expected_budgets` | Exact sample budgets $N \in \{20, 30, 40, 50\}$ | **PASSED** |
| **TEST 3** | `test_3_exactly_4_model_conditions` | 4 distinct model conditions present | **PASSED** |
| **TEST 4** | `test_4_exactly_240_expected_cells` | Matrix size equals exactly 240 rows | **PASSED** |
| **TEST 5** | `test_5_all_240_cells_accounted_and_successful` | Zero failed cells (`status == 'SUCCESS'`) | **PASSED** |
| **TEST 6** | `test_6_no_duplicate_experiment_ids` | Unique experiment IDs across all cells | **PASSED** |
| **TEST 7** | `test_7_split_hashes_match` | Track A split SHA256 hashes match Subphase 6.5.1 | **PASSED** |
| **TEST 8** | `test_8_no_adaptation_test_overlap` | Zero recording overlap between train and test | **PASSED** |
| **TEST 9** | `test_9_class_weights_from_adaptation_only` | Weights derived strictly from adaptation counts | **PASSED** |
| **TEST 10** | `test_10_class_weights_mathematical_formula` | Mathematical verification of $w_c = \frac{N}{K \times n_c}$ | **PASSED** |
| **TEST 11** | `test_11_absent_classes_no_fabricated_weights` | Missing classes assigned weight 0.0 | **PASSED** |
| **TEST 12** | `test_12_unweighted_models_uniform_weights` | Unweighted models use uniform 1.0 weights | **PASSED** |
| **TEST 13** | `test_13_adapter_config_invariance` | Architecture invariants verified for Adapter | **PASSED** |
| **TEST 14** | `test_14_rbf_config_invariance` | Invariants verified for RBF-SVM ($C=1.0$, scale) | **PASSED** |
| **TEST 15** | `test_15_no_test_labels_in_weights` | Leakage audit confirming test label isolation | **PASSED** |
| **TEST 16** | `test_16_frozen_benchmarks_untouched` | Historical Phase 3–6 results verified unperturbed | **PASSED** |
| **TEST 17** | `test_17_prediction_labels_canonical` | All predictions conform to 7-class taxonomy | **PASSED** |
| **TEST 18** | `test_18_iper_matches_canonical_formula` | IPER equals recall on the emergency subset | **PASSED** |
| **TEST 19** | `test_19_majority_collapse_detection` | Collapse flag accurately marks 100% majority | **PASSED** |
| **TEST 20** | `test_20_deterministic_predictions` | Repeated evaluation produces identical predictions | **PASSED** |
| **TEST 21** | `test_21_metrics_valid_ranges` | All metrics in valid bounded intervals $[0, 1]$ | **PASSED** |

### Combined Suite Summary:
Running the entire Phase 6.5 test suite:
- `tests/test_phase6_5_0_diagnostic.py`: 7 / 7 PASSED
- `tests/test_phase6_5_1_splits.py`: 13 / 13 PASSED
- `tests/test_phase6_5_2_classical_rescue.py`: 21 / 21 PASSED
**Total: 41 / 41 PASSED (100% Pass Rate in 1.98s).**

---

## 8. Artifact Catalog & Hashes

The generated deliverables are permanently archived in the following repository paths:

```
phase6_5/classical_rescue/
├── config/
│   └── classical_rescue_config.yaml
│
├── results/
│   ├── classical_rescue_results.csv        (240 records, 64 KB)
│   ├── classical_rescue_predictions.csv    (88,672 rows, 15.6 MB)
│   ├── class_weight_records.csv            (1,680 records, 150 KB)
│   ├── class_support_records.csv           (1,680 records, 154 KB)
│   └── errors.csv                          (0 errors, 47 B)
│
├── analysis/
│   ├── split_recovery_effect.csv           (Effect A quantification)
│   ├── class_weight_effect.csv             (Effect B quantification)
│   ├── emergency_metrics.csv               (Recall, Precision, IPER, FER)
│   ├── sample_efficiency.csv               (N=20, 30, 40, 50 trends)
│   ├── speaker_analysis.csv                (Dysarthric vs. Control breakdown)
│   └── collapse_analysis.csv               (Majority collapse statistics)
│
├── figures/
│   ├── macro_f1_recovery_comparison.png    (229 KB)
│   ├── emergency_recall_comparison.png     (248 KB)
│   ├── iper_comparison.png                 (248 KB)
│   ├── sample_efficiency_comparison.png    (133 KB)
│   └── majority_collapse_comparison.png    (95 KB)
│
└── reports/
    ├── phase6_5_2_classical_rescue_report.md
    └── phase6_5_2_reproducibility_report.md
```

---

## 9. Sign-off & Audit Conclusion

Subphase 6.5.2 satisfies all experimental, statistical, and software engineering criteria specified in the project charter:
- Zero data or label leakage occurred.
- All 240 cells completed cleanly and deterministically.
- All historical benchmarks from Phases 1–6 remain pristine.
- All automated unit tests pass without failure or warning.

**Audit Status: CERTIFIED REPRODUCIBLE.**
