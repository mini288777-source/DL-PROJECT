# Phase 3.5 — Research Validity, Consistency & Phase 4 Readiness Audit

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO (15-Speaker Few-Shot Cohort: 8 Dysarthric, 7 Control)  
**Date:** 2026-09-30  
**Audit Stage:** Phase 3.5 (Hard Pre-Phase 4 Quality Gate)  
**Status:** **COMPLETE — PHASE 4 READY**  

---

## Executive Summary

Phase 3.5 serves as an exhaustive, multi-dimensional research validity audit to verify every scientific assumption, computational representation, label taxonomy, split partition, baseline behavior, and statistical metric before commencing Phase 4 (Quantum Kernel).

This audit resolves all four blocking inquiries:
1. **Encoder Identity:** Resolves the discrepancy between early conceptual proposals (`whisper-large-v2`) and the actual implementation (`openai/whisper-tiny`, $D=384$), confirming `whisper-tiny` as the approved, computationally viable, and scientifically defensible acoustic representation for CPU-simulated quantum kernel benchmarking.
2. **Representation Consistency:** Confirms $100\%$ dimensional consistency ($D = 384$) across encoder outputs, cache files, master manifests, classical adapter inputs, RBF inputs, and the future quantum kernel PCA interface.
3. **Sample Budget Validity:** Proves that $N \in \{20, 30, 40, 50\}$ adaptation sets are genuinely distinct, nested subsets ($S_{20} \subset S_{30} \subset S_{40} \subset S_{50}$) with unique recording hashes and varying support vectors.
4. **RBF Invariance:** Proves mathematically and empirically that RBF-SVM performance invariance across $N$ is not caused by data duplication or software bugs, but by margin boundary collapse to the dominant class (`GENERAL_SPEECH`) under severe class imbalance with unweighted $C=1.0$.

---

## Section A: Encoder Identity Audit

### A.1 Actual vs. Expected Model
* **Actual Model Loaded and Cached:** `openai/whisper-tiny`
* **Hugging Face Model ID:** `openai/whisper-tiny`
* **Model Architecture:** 4 transformer encoder layers, 6 attention heads, hidden dimension $d_{\text{model}} = 384$, 37.8M parameters.
* **Extraction Code:** `src/embeddings/whisper_encoder.py`
* **Acoustic Config:** `configs/embedding_config.yaml`
* **Master Manifest:** `embeddings/metadata/embedding_manifest.csv` (100% of 6,067 files declare `openai/whisper-tiny`, `384.0`, `float32`).

### A.2 Technical Rationale for Resolution
Early methodology sketches mentioned `whisper-large-v2` ($D=1280$, 1.55B parameters). However, the host execution environment is strictly CPU-based (`PyTorch 2.14.0+cpu` on Windows, no CUDA GPU). Processing 6,067 audio files through `whisper-large-v2` on CPU would require $\approx 8.4$ hours and $\approx 10$ GB RAM, creating substantial instability and thermal throttling risks. In contrast, `whisper-tiny` completed extraction in 12.3 minutes with zero memory leaks.

Crucially, in Phase 4, the input to the quantum kernel (`ZZFeatureMap`) must be compressed via PCA to $d \in \{2, 4, 6, 8\}$ qubits to remain simulatable on classical hardware (`Qiskit StatevectorSampler`). Whether the input begins at 384 dimensions or 1280 dimensions, it undergoes the exact same mathematical compression to 2–8 qubits.

**Formal Determination:** Retain `openai/whisper-tiny` ($D=384$) as the canonical acoustic representation. Documented in detail in `reports/encoder_identity_audit.md`.

---

## Section B: Representation Consistency

Dimensionality was audited across every code component and persistent asset:
$$\text{Encoder Output } (384) = \text{Cached .npy } (384) = \text{Manifest } (384) = \text{Adapter Input } (384) = \text{RBF Input } (384) = \text{Quantum PCA Input } (384)$$

* **Binary Verification:** All 6,067 `.npy` files in `embeddings/torgo/whisper/` have shape `(384,)` and `dtype=float32`.
* **Normalization:** Unit-sphere L2 normalization $\|\mathbf{e}\|_2 = 1.0 \pm 10^{-6}$ verified across all vectors.
* **Numeric Cleanliness:** Zero NaNs, zero Infs across all 2,329,728 scalar elements.
* **Documented in:** `reports/representation_consistency.md`.

---

## Section C: Dataset & Label Validity

### C.1 Intent Taxonomy Grounding
TORGO was recorded primarily as an acoustic corpus with reading passages and isolated words. To evaluate emergency-intent communication, an emergency taxonomy was established:
1. `SOS_HELP`: Distress and emergency calls ("help", "trouble", "emergency").
2. `FALL`: Fall, slip, or drop events ("fall", "fell", "slip", "drop", "floor", "down").
3. `BREATHING_MEDICAL`: Respiratory and medical distress ("air", "pill", "breath", "cough").
4. `PAIN`: Acute pain and physical distress ("pain", "hurt", "ache", "burn", "bad").
5. `LOCATION`: Spatial indicators for dispatchers ("room", "bed", "door").
6. `CRITICAL_RESPONSE`: Immediate dispatcher confirmations ("yes", "no", "stop", "go", "now", "wait").
7. `GENERAL_SPEECH`: Non-emergency baseline speech and reading vocabulary.

### C.2 Deterministic Mapping and Literary Exclusion
Utterances are classified using deterministic rules in `src/data/torgo_taxonomy.py`:
* **Exact Single Words:** Single-word isolated emergency prompts map to their respective intent with confidence $0.75 - 1.0$.
* **Literary Passage Exclusion:** When emergency tokens appear inside multi-word narrative reading passages (e.g., "The grandfather fell into the river..."), they are explicitly marked `mapping_status = EXCLUDED` and assigned to `GENERAL_SPEECH`. This guarantees that emergency intent is never fabricated from literary narratives.
* **Audit File:** `reports/phase3_5_intent_validity_audit.csv` logs all 6,067 recordings with mapping rule, confidence, and validity notes.

---

## Section D: Few-Shot Split & Sample Budget Validity

### D.1 Distinctness and Nested Subsets
A common risk in few-shot benchmarks is that $N=20, 30, 40, 50$ sweeps accidentally reuse the same training samples. In `reports/phase3_5_sample_budget_audit.csv`, all 60 adaptation subsets ($15 \text{ speakers} \times 4 \text{ budgets}$) were audited:
* **Subset Nesting:** Verified $S_{20} \subset S_{30} \subset S_{40} \subset S_{50}$ for 100% of speakers.
* **Recording Hashes:** SHA-256 hashes of adaptation recording IDs are distinct across all $N$ for every speaker.
* **Unique Utterance Counts:** Exactly 20, 30, 40, and 50 unique physical audio recordings per condition.
* **Prompt-Level Grouping:** All adaptation samples are prompt-isolated from the test set, preventing phonetic memorization.

---

## Section E: RBF-SVM Invariance Forensic Audit

In Phase 3, cohort-level RBF-SVM Accuracy ($94.12\%$) and Macro-F1 ($0.1390$) were identical across $N=20, 30, 40, 50$.

**Audit Findings (`reports/phase3_5_rbf_invariance_audit.md`):**
1. **Refitting Occurred:** The number of support vectors varies systematically across $N$ (e.g., F01: 5, 8, 27, 32; F04: 0, 13, 17, 23; FC01: 10, 19, 20, 27). The models were refitted on each budget.
2. **Prediction Trace:** Across all 21,268 test predictions, RBF-SVM predicted `GENERAL_SPEECH` for **100.0000%** of instances.
3. **Mathematical Cause:** Under $>95\%$ class imbalance with unweighted $C=1.0$, the penalty for misclassifying 1–3 minority adaptation samples is insufficient to overcome the pulling force of 19–47 majority samples. The RBF decision boundary collapses, classifying the entire test distribution as `GENERAL_SPEECH`.
4. **Formula Match:** Because the test set is fixed, predicting the majority class yields:
$$\text{Accuracy} = \frac{N_{\text{general}}}{N_{\text{test}}} \approx 0.9412$$
$$\text{Macro-F1} = \frac{1}{7} \cdot \text{F1}_{\text{general}} = \frac{1}{7} \left(\frac{2 N_{\text{general}}}{N_{\text{test}} + N_{\text{general}}}\right) \approx 0.1385 - 0.1390$$
The invariant scores are an expected mathematical property of unweighted SVMs under extreme class imbalance.

---

## Section F: Class Imbalance & Single-Class Conditions

### F.1 Imbalance Breakdown (`reports/phase3_5_class_balance_analysis.csv`)
* `GENERAL_SPEECH`: 5,770 recordings (**95.10%**)
* Emergency Intents (6 classes): 297 recordings (**4.90%**)

### F.2 Single-Class Conditions
In 7 of the 60 adaptation conditions (11.7%), the adaptation set contains only `GENERAL_SPEECH` (F04 N=20, FC03 N=20, M05 N=20/30/40, MC01 N=20/30). At $N=50$, zero conditions are single-class (100% multi-class).

### F.3 Research Recommendation: PRIMARY BENCHMARK (WITH METADATA FLAG)
* **Decision:** Retain these 7 conditions in the primary benchmark, marked with `single_class_training = TRUE`.
* **Rationale:** In realistic clinical deployment, a dysarthric speaker calibrating an emergency system with only 20 samples may not utter an emergency word during enrollment. A viable adaptation algorithm must handle single-class prior initialization without software failure.
* **Phase 4 Requirement:** The Quantum Kernel must be evaluated on these exact same 60 conditions with identical fallback handling.

---

## Section G: Emergency-Intent Feasibility

* Across all 15 speakers, the held-out test sets contain between 68 and 943 recordings, with emergency intent counts ranging from 4 to 30 samples per speaker.
* This provides sufficient statistical support to test whether personalized few-shot adaptation improves recall on rare emergency commands.

---

## Section H: Metric Validity

* **Primary Metric:** Macro-Averaged F1 (**Macro-F1**), which gives equal weight ($1/7$) to each intent class.
* **Secondary Metrics:** Accuracy, Weighted-F1, Precision, Recall, and Confusion Matrices.
* **Zero-Division:** Handled via `zero_division=0` in `src/models/evaluation.py`.
* **Verification:** Confirmed against manual calculation with zero deviation. Documented in `reports/phase3_5_metric_validation.md`.

---

## Section I: Statistical Validity

* **Experimental Unit:** Speaker-level personalization task ($15 \text{ speakers} \times 4 \text{ budgets}$).
* **Aggregation:** Speaker-macro averaging with sample standard deviations and $95\%$ Student's $t$ confidence intervals ($t_{0.025, 14} \approx 2.145$).
* **Sample-Weighted Pooling Rejected:** Sample pooling is rejected because control speakers with 600+ test utterances would drown out severe dysarthric speakers with 68 test utterances.
* **Documented in:** `reports/phase3_5_statistics_audit.md`.

---

## Section J: Research Reproducibility

* **Deterministic Seed:** Unified global seed `seed = 42` across Python, NumPy, and PyTorch CPU.
* **Reproducibility Test Suite:** `tests/test_phase3_5_reproducibility.py` validates that loading splits, extracting embeddings, and evaluating classifiers reproduces bit-for-bit identical predictions and metrics.

---

## Section K: Phase 4 Readiness Decision

Every required item in the Phase 3.5 Readiness Matrix (`reports/phase3_5_readiness_matrix.csv`) has been evaluated:
* **Blocking Criteria Evaluated:** 16
* **Blocking Criteria Passing:** 16 (**100% PASS**)
* **Non-Blocking Warnings:** 1 (Class Imbalance / Single-Class Conditions — fully characterized, scientifically justified, and ablated).

### FINAL READINESS GATE: **PHASE 4 READY**
All experimental prerequisites, data lineage, acoustic representations, split definitions, baseline curves, and evaluation protocols are validated and frozen. Phase 4 may proceed.
