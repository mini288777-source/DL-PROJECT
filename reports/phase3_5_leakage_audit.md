# Data Leakage & Information Isolation Audit

**Date:** 2026-09-30  
**Phase:** 3.5 — Research Validity, Consistency & Phase 4 Readiness Gate  
**Target File:** `reports/phase3_5_leakage_audit.md`  

---

## 1. Executive Summary

A zero-tolerance data leakage audit was conducted across all 15 speakers, all four sample budgets ($N \in \{20, 30, 40, 50\}$), and all experimental conditions. The audit encompassed recording ID overlap, prompt-level stimulus grouping, feature extraction isolation, feature scaler normalization boundaries, and hyperparameter tuning protocols.

* **Acoustic Recording ID Overlap:** **0 violations (100% DISJOINT)**.
* **Prompt-Level Semantic Overlap:** **0 violations (100% DISJOINT)**.
* **Feature Normalization Leakage:** **0 violations (Strict adaptation-only fitting)**.
* **Model Selection Leakage:** **0 violations (Fixed hyperparameters across all conditions)**.
* **Future-Phase Leakage:** **0 violations (No quantum circuits or noise perturbations introduced)**.

---

## 2. Forensic Audits Conducted

### 2.1 Recording-Level Disjointness
For every speaker $s \in \{1 \dots 15\}$ and every budget $N \in \{20, 30, 40, 50\}$:
$$\mathcal{S}_{\text{adapt}}(s, N) \cap \mathcal{S}_{\text{test}}(s) = \emptyset$$

* **Verification Method:** In `scripts/audit_phase3_5.py`, set intersection operations were executed across all 60 split definitions in `splits/torgo_splits.json`.
* **Result:** $\left|\mathcal{S}_{\text{adapt}} \cap \mathcal{S}_{\text{test}}\right| = 0$ for all 60 conditions.

### 2.2 Prompt-Level Stimulus Grouping
In atypical speech research, if the exact same sentence or prompt (e.g., "The quick brown fox...") appears in both the adaptation set and the test set across different audio takes, an acoustic model can memorize the phonetic template rather than adapting to the speaker's pathology.

* **Audit:** In Phase 1 (`src/data/torgo_splits.py`), splits were generated using **GroupKFold / Prompt-level grouping**. All utterances sharing the same canonical text prompt are assigned exclusively to either the adaptation pool or the test pool.
* **Result:** Zero prompt overlap between adaptation and test partitions.

### 2.3 Feature Extraction Isolation
* **Audit:** Whisper feature embeddings were extracted using frozen encoder weights (`model.eval()`, `torch.inference_mode()`). No fine-tuning, backpropagation, or adaptation occurred during Phase 2 extraction.
* **Temporal Mean-Pooling:** Pooling is computed independently on individual audio files. No inter-utterance statistics (e.g., global mean or variance) were computed or applied.

### 2.4 Preprocessing and Scaler Isolation
* **Classical Adapter:** The Adapter network operates directly on unit-sphere normalized vectors ($\|\mathbf{x}\|_2 = 1.0$), which are computed independently per file. No batch normalization or training-set-wide scaling is performed.
* **Classical RBF-SVM:** In `src/models/rbf_svm.py`, if feature standardization is enabled, `StandardScaler.fit()` is called strictly on `X_train` (the adaptation set), and `transform()` is subsequently applied to `X_test`. In the default configuration, raw unit-sphere vectors are ingested directly with $C=1.0, \gamma=\text{'scale'}$, eliminating any scaler-based leakage channel.

### 2.5 Hyperparameter Tuning & Model Selection
* No cross-validation or hyperparameter search was performed against the held-out test sets.
* Hyperparameters for Adapter (`lr = 1e-3`, `hidden_dim = 64`, `dropout = 0.1`, `epochs = 50`) and RBF-SVM (`C = 1.0`, `gamma = 'scale'`) were fixed globally *a priori* in `configs/phase3_baselines.yaml` before running the benchmark.
* No test-set performance metrics were used to stop training early or select model checkpoints.

---

## 3. Cryptographic Verification Table

Sample hashes of adaptation sets vs held-out test sets for representative speakers:

| Speaker | Budget ($N$) | Adaptation Hash (SHA-256) | Test Set Hash (SHA-256) | Set Overlap Count |
|---|---|---|---|---|
| **F01** | $N=20$ | `e0e1510deca77ee5` | `7a88448ecdbdb2e4` | **0** |
| **F01** | $N=50$ | `ddaf2a20fc2e25cb` | `7a88448ecdbdb2e4` | **0** |
| **F03** | $N=20$ | `088fbe626ccd7126` | `72aa3dfa0ef44bf7` | **0** |
| **FC01**| $N=50$ | `7c5523cf3be60baa` | `d726bfa5ea51f0ca` | **0** |
| **M01** | $N=20$ | `ca87f87ee9db4fbf` | `655fc92ffbcf3f9c` | **0** |
| **MC01**| $N=50$ | `13264c1aa5722108` | `6e82200ad7fa9188` | **0** |

## 4. Audit Conclusion

**LEAKAGE AUDIT STATUS: PASS**  
The dataset splits and evaluation protocols maintain strict information boundaries. There is zero evidence of data leakage, target contamination, or post-hoc optimization.
