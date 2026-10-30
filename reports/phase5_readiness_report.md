# Phase 5 Readiness Audit Report: Controlled Acoustic Robustness

**Date:** 2026-09-30  
**Phase:** Phase 5 — Controlled Noise Robustness Evaluation  
**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Evaluation Protocol:** Protocol B (Clean Calibration $\to$ Noisy Evaluation)  
**Readiness Status:** **100% READY & VERIFIED (IN EXECUTION)**

---

## 1. Executive Summary

Phase 5 evaluates the acoustic noise generalization of personalized speech recognition models calibrated on clean atypical speech. Unlike unconstrained robustness studies, Phase 5 enforces a strictly frozen experimental contract on top of the validated Phase 4.5 pipeline:
- **Canonical Speech Encoder:** `openai/whisper-tiny` (FROZEN, 384-D, float32, unit-L2 normalized).
- **Speaker Cohort:** Exactly 15 TORGO speakers (8 dysarthric, 7 control).
- **Adaptation Budgets:** Exactly $N \in \{20, 30, 40, 50\}$.
- **Model Triad:**
  1. Classical Adapter Head (25,223 parameters, AdamW, $\text{lr}=10^{-3}$, weight decay $10^{-2}$, epochs=50, batch size=8).
  2. Classical RBF-SVM ($C=1.0, \gamma=\text{"scale"}$, dual support vector formulation).
  3. Primary Quantum QSVC ($q=4, \text{reps}=1, \text{linear entanglement}, \text{ZZFeatureMap}, \text{FidelityQuantumKernel}, C=1.0$, train-only PCA $384 \to 4$, train-only MinMaxScaler to $[0, \pi]$).
- **Test Acoustic Perturbations:** 4 conditions: Clean, 15 dB SNR, 10 dB SNR, 5 dB SNR.
- **Total Experimental Cells:** $15 \times 4 \times 3 \times 4 = 720$ cells.

---

## 2. Invariant & Lineage Verification Matrix

| Component | Specification / Contract | Source Verification | Status |
|---|---|---|---|
| **Phase 0** | Audio ingestion, baseline SNR noise injection, smoke tests | `reports/phase0_summary.md` | **VERIFIED** |
| **Phase 1** | 15 speakers, 60 fixed splits, 7 emergency intents | `splits/torgo_splits.json` | **VERIFIED** |
| **Phase 2** | Frozen Whisper-tiny embeddings (384-D, FP32, L2 norm) | `embeddings/torgo/whisper/` (6,067 files) | **VERIFIED** |
| **Phase 3** | Classical Baselines (Adapter & RBF-SVM, 120 conditions) | `results/phase3/baseline_curves.csv` | **VERIFIED** |
| **Phase 4** | Primary Quantum QSVC ($q=4$, 240 conditions) | `results/phase4/quantum_curves.csv` | **VERIFIED** |
| **Phase 4.5** | Hardening gate: PSD revalidation, single-class audit, null framing | `reports/phase4_5_complete_results.md` | **VERIFIED** |
| **Phase 5 Contract** | Protocol B frozen specification | `configs/phase5_contract.yaml` | **VERIFIED** |

---

## 3. Methodological Controls & Leakage Prevention

1. **Protocol B Strict Invariant:**
   All models (Adapter, RBF-SVM, QSVC) are adapted / fitted **strictly on clean adaptation audio**. Noisy test samples NEVER enter the training set.
2. **Train-Only Preprocessing:**
   PCA ($384 \to 4$) and MinMaxScaler ($[0, \pi]$) for the quantum pipeline are fitted **strictly on the clean adaptation subset** of each speaker $\times$ budget condition. The fitted transforms are then applied to held-out test embeddings (clean and noisy).
3. **Deterministic Noise Realization:**
   Noise injection operates at the raw waveform level via `src.utils.noise.add_noise_at_snr`. Noise seeds are derived deterministically using a cryptographic hash:
   $$\text{seed} = \text{MD5}(\text{base\_seed} \parallel \text{recording\_id} \parallel \text{SNR})$$
   This guarantees that Adapter, RBF-SVM, and QSVC evaluate on **identical noisy waveforms**.
4. **Whisper Embedding Caching:**
   Noisy Whisper embeddings are computed **once per recording $\times$ SNR** and cached in `embeddings/torgo/whisper_noisy/snr_{SNR}/`. Zero redundant encoder evaluations.

---

## 4. Pre-Flight Verification & Pilot Results

### Pilot 1 (12 Cells: F01, N=20, 3 Models, 4 SNRs)
- Noise injection, embedding extraction, and caching verified on 68 test recordings.
- Clean condition Macro-F1:
  - Adapter: $0.1385$ (Phase 3 baseline: $0.1385$, diff: $0.0000$) $\implies$ **PASS**
  - RBF-SVM: $0.1385$ (Phase 3 baseline: $0.1385$, diff: $0.0000$) $\implies$ **PASS**
  - QSVC ($q=4$): $0.1385$ (Phase 4 baseline: $0.1385$, diff: $0.0000$) $\implies$ **PASS**
- Runtime: $29.57$ seconds.

### Pilot 2 (72 Cells: F01, FC01, M01, N in {20, 50}, 3 Models, 4 SNRs)
- 60 new conditions evaluated, 12 resumed from Pilot 1.
- Failures: **0**.
- Resumability: **VERIFIED**.
- Clean regression check: All conditions reproduce Phase 3 & 4 baselines within floating-point tolerance.

---

## 5. Gate Sign-Off

All prerequisites for the complete Phase 5 720-cell experiment are satisfied. Execution of the full sweep is proceeding with zero blocking issues.
