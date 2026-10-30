# Phase 2 — Speech Embedding Extraction & Caching Summary Report

## 1. Executive Summary
- **Research Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication
- **Dataset:** TORGO (15-speaker few-shot cohort)
- **Primary Acoustic Representation:** Pretrained Frozen Whisper Embeddings
- **Classical Reference Representation:** Deterministic MFCC Baseline Vectors
- **Storage Contract:** Independent, fixed-size `.npy` files per recording + centralized metadata manifest

---

## 2. Experimental Representation Specifications

| Specification | Primary Representation (Whisper) | Classical Baseline (MFCC) |
| :--- | :--- | :--- |
| **Model / Algorithm** | `openai/whisper-tiny` (Configurable `whisper-large-v2` profile) | 13 MFCCs + $\Delta$ + $\Delta\Delta$ (39 channels) |
| **Encoder Weights** | Strictly FROZEN (`eval()`, `inference_mode`, `requires_grad=False`) | Non-trainable deterministic DSP |
| **Input Audio Contract** | 16 kHz, Mono, float32, Peak-normalized [-1.0, 1.0] | 16 kHz, Mono, float32, Peak-normalized [-1.0, 1.0] |
| **Feature Extraction** | Log-Mel Spectrogram (80 mel bands, 25ms window, 10ms hop) | FFT window 400 (25ms), hop 160 (10ms) |
| **Temporal Pooling** | Mean pooling over time frames | Mean + Standard Deviation over time frames |
| **Final Dimension** | **384** (float32) | **78** (float32) |
| **L2 Normalization** | Unit sphere normalized ($\|x\|_2 = 1.0$) | Standard unconstrained feature space |
| **Cache Format** | `embeddings/torgo/whisper/<recording_id>.npy` | `embeddings/torgo/mfcc/<recording_id>.npy` |

---

## 3. Cohort Extraction & Coverage Audit

| Metric | Whisper | MFCC | Total Cohort Target |
| :--- | :--- | :--- | :--- |
| **Cohort Recordings Target** | 6,067 | 6,067 | 6,067 |
| **Successfully Cached** | 6067 | 6067 | 6,067 |
| **Failed Extractions** | 0 | 0 | 0 |
| **Storage Footprint** | 9.63 MB | 2.55 MB | 12.17 MB |
| **Reload Latency** | < 0.2 ms / sample | < 0.2 ms / sample | Zero PyTorch / model inference |

---

## 4. Per-Speaker Few-Shot Cohort Breakdown

| Speaker ID | Group | Manifest Total | Cohort Target | Whisper Cached | MFCC Cached | N=20 Ready | N=30 Ready | N=40 Ready | N=50 Ready |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F01** | dysarthric | 236 | 118 | 118 | 118 | PASS | PASS | PASS | PASS |
| **F03** | dysarthric | 1087 | 213 | 213 | 213 | PASS | PASS | PASS | PASS |
| **F04** | dysarthric | 673 | 250 | 250 | 250 | PASS | PASS | PASS | PASS |
| **FC01** | control | 302 | 151 | 151 | 151 | PASS | PASS | PASS | PASS |
| **FC02** | control | 2187 | 975 | 975 | 975 | PASS | PASS | PASS | PASS |
| **FC03** | control | 1922 | 405 | 405 | 405 | PASS | PASS | PASS | PASS |
| **M01** | dysarthric | 743 | 285 | 285 | 285 | PASS | PASS | PASS | PASS |
| **M02** | dysarthric | 772 | 232 | 232 | 232 | PASS | PASS | PASS | PASS |
| **M03** | dysarthric | 810 | 406 | 406 | 406 | PASS | PASS | PASS | PASS |
| **M04** | dysarthric | 666 | 286 | 286 | 286 | PASS | PASS | PASS | PASS |
| **M05** | dysarthric | 587 | 357 | 357 | 357 | PASS | PASS | PASS | PASS |
| **MC01** | control | 2155 | 419 | 419 | 419 | PASS | PASS | PASS | PASS |
| **MC02** | control | 1123 | 383 | 383 | 383 | PASS | PASS | PASS | PASS |
| **MC03** | control | 1669 | 594 | 594 | 594 | PASS | PASS | PASS | PASS |
| **MC04** | control | 1620 | 993 | 993 | 993 | PASS | PASS | PASS | PASS |

---

## 5. Architectural & Reproducibility Verification
1. **Zero Data Leakage:** Preprocessing and representation extraction are applied unconditionally per recording without access to speaker labels, test set statistics, or train/test splits.
2. **Fixed Dimensionality:** 100% of Whisper embeddings are strictly 384-dimensional float32 arrays; 100% of MFCC vectors are strictly 78-dimensional float32 arrays.
3. **Zero Trainable Parameters in Phase 2:** Whisper encoder remained strictly frozen throughout. No adapters, projection layers, PCA transformations, or classifiers were fitted.
4. **Independent Representations:** Whisper and MFCC embeddings are saved and loaded independently. No automatic concatenation or unexplained fusion exists.
5. **Bit-for-Bit Determinism:** Validated across repeated extraction passes on CPU. Max numerical difference: 0.00e+00.
6. **Zero-Dependency Reload:** Validated via `tests/test_embedding_reload.py` — future phases (Fine-Tuning, RBF, Quantum Kernel) can reload representations using pure numpy without loading Whisper model weights.

---

## 6. Phase 2 Verification Sign-Off
- **Determinism Test:** PASS (`tests/test_embedding_determinism.py`)
- **Reload Acceptance Test:** PASS (`tests/test_embedding_reload.py`)
- **Cache Integrity Test:** PASS (`tests/test_embedding_cache.py`)
- **Interactive Notebook:** PASS (`notebooks/phase2_embedding_validation.ipynb`)
- **Overall Status:** **PHASE 2 COMPLETE**
