# Phase 5 Experimental & Noise Injection Contract Design Document

**Date:** 2026-09-30  
**Phase:** 4.5 — Research Hardening  
**Target Phase:** Phase 5 — Noise Robustness Evaluation  
**Status:** **SPECIFICATION & DESIGN ONLY — NOT EXECUTED**

---

## 1. Scientific Objective of Phase 5

Phase 5 evaluates the central thesis of the project regarding emergency reliability:
> *"Does a quantum kernel representation exhibit greater degradation resistance (higher robustness) than classical neural and kernel baselines when atypical emergency speech is perturbed by realistic environmental acoustic noise?"*

Under clean speech (Phase 4), quantum and RBF-SVM achieved identical performance ($0.1390 \pm 0.0007$). Phase 5 tests whether acoustic noise corrupts classical representation geometry more rapidly than the non-linear quantum Hilbert space.

---

## 2. Noise Conditions Specification

| Noise Condition | Target SNR | Clinical / Operational Analogy | Implementation Function |
|---|---|---|---|
| **Condition 0: Clean** | $\infty\text{ dB}$ (Unperturbed) | Quiet clinic or home setting | Unmodified original audio (Phase 4 baseline) |
| **Condition 1: SNR 15 dB** | $15.0\text{ dB} \pm 0.5\text{ dB}$ | Mild ambient indoor noise (HVAC, distant talking) | `src.utils.noise.add_noise_at_snr(clean, 15.0, rng)` |
| **Condition 2: SNR 10 dB** | $10.0\text{ dB} \pm 0.5\text{ dB}$ | Moderate public noise (traffic, cafeteria, television) | `src.utils.noise.add_noise_at_snr(clean, 10.0, rng)` |
| **Condition 3: SNR 5 dB** | $5.0\text{ dB} \pm 0.5\text{ dB}$ | Severe emergency setting (ambulance siren, alarm) | `src.utils.noise.add_noise_at_snr(clean, 5.0, rng)` |

---

## 3. Strict Pre-Processing Invariant: The Split-Preservation Rule

```
                                  [Original Clean Audio]
                                            |
                                            v
                                [Phase-1 Fixed Splits]
                      (splits/torgo_splits.json — 60 Fixed Sets)
                                            |
                                            v
                                 [Split Identities Frozen]
                            (Train/Test Recordings Locked)
                                            |
                                            v
                         +------------------+------------------+
                         |                                     |
                         v                                     v
             [Adaptation Recordings]                 [Held-Out Test Recordings]
                         |                                     |
                         v                                     v
             [Noise Injection at SNR]              [Noise Injection at SNR]
             (Deterministic RNG Seed)              (Deterministic RNG Seed)
                         |                                     |
                         v                                     v
             [Whisper Feature Pipeline]            [Whisper Feature Pipeline]
             (384-D Unit-L2 Embeddings)            (384-D Unit-L2 Embeddings)
                         |                                     |
                         v                                     v
             [Identical Adaptation Protocol]       [Identical Evaluation Function]
```

### Critical Rules:
1. **Never Regenerate Splits:** Splits must never be re-partitioned after noise is added.
2. **Identical Perturbations Across Models:** The exact same noisy audio waveforms must be fed to the Whisper encoder for Adapter, RBF-SVM, and QSVC.
3. **No Leakage:** Noise must be injected on an isolated per-file basis without cross-utterance power normalization.
4. **Validation Standards for Generated Audio:**
   - Sample rate: exactly $16,000\text{ Hz}$.
   - Duration: bit-length identical to clean original.
   - Values: finite float32 (zero `NaN` or `Inf`).
   - Clipping: clipped to $[-1.0, 1.0]$ with clipping verification log.

---

## 4. Phase 5 Experimental Grid Matrix

The full Phase 5 evaluation grid encompasses:

$$\text{Total Runs} = 3\text{ Models} \times 4\text{ Sample Budgets } (N) \times 4\text{ Noise Levels (Clean, 15, 10, 5 dB)} \times 15\text{ Speakers} = 720\text{ Runs}$$

| Dimension | Values | Count |
|---|---|---|
| **Models** | Classical Adapter, Classical RBF-SVM, Primary Quantum QSVC ($q=4$) | 3 |
| **Budgets ($N$)** | $20, 30, 40, 50$ | 4 |
| **Noise Levels** | Clean, $15\text{ dB}$, $10\text{ dB}$, $5\text{ dB}$ SNR | 4 |
| **Speakers** | 8 Dysarthric (`F01..M05`), 7 Control (`FC01..MC04`) | 15 |
| **Total Experimental Runs** | $3 \times 4 \times 4 \times 15$ | **720** |

---

## 5. Formal Hard Stop

```
==================================================
PHASE 5 DESIGN COMPLETE
EXECUTION IS PROHIBITED IN PHASE 4.5
==================================================
```
Phase 5 execution will commence only upon explicit user authorization.
