# Phase 2: Whisper Acoustic Representation Extraction Summary

**Date:** 2026-09-30  
**Model:** `openai/whisper-tiny`  
**Embedding Dimension:** `384`  
**Temporal Pooling:** `mean`  
**L2 Normalization:** `True`  
**Extraction Status:** **VALID**  

---

## 1. Extraction Metrics

- **Total Cached Embeddings:** 1,500
- **Target Speakers:** 15 (8 dysarthric, 7 control)
- **Adaptation Vectors:** 750 (50 per speaker)
- **Held-Out Test Vectors:** 750 (50 per speaker)
- **Zero Vectors:** 0
- **NaN / Inf Samples:** 0 detected
- **Mean L2 Norm:** 1.0000 (unit normalized)
- **Total Extraction Duration:** 564.25s (2.7 samples/s on CPU)
- **Persistent Cache Archive:** `data/embeddings/torgo_whisper_embeddings.npz`

---

## 2. Speaker-by-Speaker Representation Breakdown

| Speaker ID | Clinical Group | Adaptation Samples | Test Samples | Total Cached | Mean Duration (s) |
|---|---|---|---|---|---|
| `F01` | dysarthric | 50 | 50 | **100** | 2.32s |
| `F03` | dysarthric | 50 | 50 | **100** | 3.27s |
| `F04` | dysarthric | 50 | 50 | **100** | 3.25s |
| `FC01` | control | 50 | 50 | **100** | 3.50s |
| `FC02` | control | 50 | 50 | **100** | 2.89s |
| `FC03` | control | 50 | 50 | **100** | 2.58s |
| `M01` | dysarthric | 50 | 50 | **100** | 2.64s |
| `M02` | dysarthric | 50 | 50 | **100** | 3.97s |
| `M03` | dysarthric | 50 | 50 | **100** | 2.83s |
| `M04` | dysarthric | 50 | 50 | **100** | 4.86s |
| `M05` | dysarthric | 50 | 50 | **100** | 4.79s |
| `MC01` | control | 50 | 50 | **100** | 2.11s |
| `MC02` | control | 50 | 50 | **100** | 3.21s |
| `MC03` | control | 50 | 50 | **100** | 2.03s |
| `MC04` | control | 50 | 50 | **100** | 2.06s |

---

## 3. Ready for Downstream Baselines

All representations for the N = 20, 30, 40, and 50 adaptation sweeps and held-out evaluation sets are persistently cached and validated.
The project is now primed for Phase 3 (Classical Fine-Tuning & RBF Baselines) and Phase 4 (Quantum Kernel Personalization).
