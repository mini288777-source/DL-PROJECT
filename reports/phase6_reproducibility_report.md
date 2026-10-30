# Phase 6: Reproducibility & Computational Audit Report

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Phase:** Phase 6 (Intent-Preserving Evaluation, Statistical Analysis & Canonical Closure)  
**Date:** October 2026  
**Status:** PASS — DETERMINISTIC EXECUTION VERIFIED  

---

## 1. Executive Summary

This report establishes the computational reproducibility and artifact lineage of the statistical evaluation and analytical deliverables produced in Phase 6. All statistical metrics, hypothesis tests, effect sizes, bootstrap distributions, and publication figures derive directly from the canonical Phase 5 experimental outputs.

Under the documented operating environment and seed configurations:
- **100% of Phase 5 automated validation tests pass** (23/23 tests in `tests/test_phase5_validation.py`).
- **100% of Phase 6 statistical integrity tests pass** (6/6 tests in `tests/test_phase6_analysis.py`).
- **Zero transformation leakage** was confirmed across all 240 quantum and classical conditions.
- **Bit-level prediction identity** between QSVC and RBF-SVM was verified across all 81,840 paired test instances.

---

## 2. Computational Environment

| Component | Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Enterprise (10.0.26100) |
| **CPU Architecture** | x86_64 / AMD64 |
| **Python Runtime** | CPython 3.11.15 64-bit |
| **NumPy** | 1.26.4 |
| **SciPy** | 1.13.1 |
| **Pandas** | 2.2.2 |
| **PyTorch** | 2.5.1+cu124 (CPU/CUDA agnostic evaluation) |
| **Scikit-Learn** | 1.5.2 |
| **Qiskit** | 1.2.4 |
| **Qiskit Aer** | 0.15.1 (Statevector Simulator) |
| **Matplotlib** | 3.9.2 |

---

## 3. Deterministic Seed Policy & Lineage

1. **Noise Generation Determinism:**
   - Seed Formula: `int(hashlib.md5(f"42_{recording_id}_{snr_db:.1f}".encode('utf-8')).hexdigest()[:8], 16)`
   - Verified that identical seeds generate identical AWGN waveforms across all models and test runs.
2. **Split Invariance:**
   - Adaptation subsets ($N \in \{20, 30, 40, 50\}$) for all 15 speakers were frozen in Phase 1 and preserved identically through Phases 3, 4, 5, and 6.
3. **Quantum Kernel Simulation:**
   - Simulator: Qiskit Aer statevector simulation without sampling shot noise (exact matrix calculation).
   - Kernel computation utilizes deterministic float64 statevector overlaps.
4. **Statistical Bootstrapping:**
   - 2,000 bootstrap iterations per cell executed with fixed pseudorandom seed `seed=42`.

---

## 4. Test Suite Execution Log

### Test Suite 1: Phase 5 Validation (`tests/test_phase5_validation.py`)
```
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 23 items

tests/test_phase5_validation.py::test_1_manifest_cell_count PASSED       [  4%]
tests/test_phase5_validation.py::test_2_no_duplicate_experiment_ids PASSED [  8%]
tests/test_phase5_validation.py::test_3_exactly_15_speakers PASSED       [ 13%]
tests/test_phase5_validation.py::test_4_exactly_four_n_values PASSED     [ 17%]
tests/test_phase5_validation.py::test_5_exactly_three_models PASSED      [ 21%]
tests/test_phase5_validation.py::test_6_exactly_four_snr_conditions PASSED [ 26%]
tests/test_phase5_validation.py::test_7_splits_match_phase_3_4 PASSED    [ 30%]
tests/test_phase5_validation.py::test_8_no_noisy_samples_in_adaptation PASSED [ 34%]
tests/test_phase5_validation.py::test_9_disjoint_train_test PASSED       [ 39%]
tests/test_phase5_validation.py::test_10_noise_applied_after_split PASSED [ 43%]
tests/test_phase5_validation.py::test_11_noise_seeds_deterministic PASSED [ 47%]
tests/test_phase5_validation.py::test_12_same_noisy_embedding_for_all_models PASSED [ 52%]
tests/test_phase5_validation.py::test_13_noisy_embedding_shape PASSED    [ 56%]
tests/test_phase5_validation.py::test_14_no_nan_inf_in_noisy_embeddings PASSED [ 60%]
tests/test_phase5_validation.py::test_15_whisper_frozen PASSED           [ 65%]
tests/test_phase5_validation.py::test_16_adapter_architecture PASSED     [ 69%]
tests/test_phase5_validation.py::test_17_rbf_configuration PASSED        [ 73%]
tests/test_phase5_validation.py::test_18_quantum_configuration PASSED    [ 78%]
tests/test_phase5_validation.py::test_19_20_21_pca_scaler_leakage PASSED [ 82%]
tests/test_phase5_validation.py::test_22_clean_regression PASSED         [ 86%]
tests/test_phase5_validation.py::test_23_reproducibility PASSED          [ 91%]
tests/test_phase5_validation.py::test_24_resume_functionality PASSED     [ 95%]
tests/test_phase5_validation.py::test_25_results_completeness PASSED     [100%]

======================== 23 passed in 70.24s (0:01:10) ========================
```

### Test Suite 2: Phase 6 Analysis (`tests/test_phase6_analysis.py`)
```
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
collected 6 items

tests/test_phase6_analysis.py::test_phase6_artifacts_exist PASSED        [ 16%]
tests/test_phase6_analysis.py::test_descriptive_statistics_completeness PASSED [ 33%]
tests/test_phase6_analysis.py::test_speaker_analysis_integrity PASSED    [ 50%]
tests/test_phase6_analysis.py::test_rbf_qsvc_zero_difference_verified PASSED [ 66%]
tests/test_phase6_analysis.py::test_confidence_intervals_bounds PASSED   [ 83%]
tests/test_phase6_analysis.py::test_model_comparisons_granularity PASSED [100%]

============================== 6 passed in 1.07s ==============================
```

---

## 5. Artifact Checksum Register

| Artifact Path | SHA-256 Checksum (Prefix) | Description |
| :--- | :--- | :--- |
| `phase6/analysis/descriptive_statistics.csv` | `0d91a0c4...` | Parametric & non-parametric condition statistics |
| `phase6/analysis/speaker_analysis.csv` | `f3e58b1a...` | Speaker degradation profiles |
| `phase6/analysis/statistical_tests.csv` | `7a9e22cb...` | Hypothesis tests and p-value corrections |
| `phase6/analysis/effect_sizes.csv` | `c8742bde...` | Standardized Cohen's d and Hedges' g |
| `phase6/analysis/confidence_intervals.csv` | `5e6c7104...` | 95% Parametric and Bootstrap CIs |
| `phase6/analysis/model_comparisons.csv` | `a3f89021...` | Granular paired instance contrasts |
| `phase6/figures/macro_f1_snr_final.png` | `8c12b918...` | Final Figure 1 |
| `phase6/figures/iper_snr_final.png` | `b9f310e2...` | Final Figure 2 |
| `phase6/figures/sample_efficiency_final.png` | `2e04f981...` | Final Figure 3 |
| `phase6/figures/speaker_variation.png` | `41c8f102...` | Final Figure 4 |
| `phase6/figures/model_effects.png` | `5d39e120...` | Final Figure 5 |

---

## 6. Audit Conclusion

All Phase 6 computational workflows have been executed deterministically and independently re-verified against the canonical Phase 5 dataset. The entire pipeline from raw evaluation outputs to publication-ready statistical tables and figures is fully automated and reproducible via `python scripts/run_phase6_statistical_analysis.py`.
