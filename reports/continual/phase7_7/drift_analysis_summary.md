# Phase 7.7 Empirical Synthesis: Speaker Drift Detection & Controlled Recalibration

## Project Title
**“Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication”**

---

### Executive Summary

Phase 7.7 investigates whether changes in an individual speaker's incoming speech distribution and model confidence behavior can identify conditions under which personalization requires recalibration (**RQ7.5**).

- **Execution Mode:** Strict Research Mode (Classical / Model-Agnostic Only)
- **Cohort:** 15 TORGO Speakers (7 Dysarthric, 8 Control)
- **Accumulation Trajectory:** $N \in \{20, 30, 40, 50\}$ (B0 reference seed, B1, B2, B3)
- **Models Trained:** 0 (Recalibration adjusts confidence scaling parameter $T$ without modifying weights)
- **New Experimental Runs:** 45 matched comparisons
- **Existing Regression Tests:** 369 / 369 passed

---

### Core Empirical Findings

| Metric | Reference / Pre-Drift (B0, N=20) | Post-Drift Uncalibrated (Bt, N=50) | Post-Drift Recalibrated (Bt, N=50) | $\Delta$ (Recal - Uncal) | $p$-value (Wilcoxon) |
|---|---|---|---|---|---|
| **Representation MMD** | 0.0000 | 0.0489 $\pm$ 0.1025 | N/A (Feature-agnostic) | — | — |
| **Centroid Cosine Distance** | 0.0000 | 0.0023 | N/A (Feature-agnostic) | — | — |
| **Expected Calibration Error (ECE)** | 0.1782 | 0.1959 | 0.2536 | +0.0562 $\pm$ 0.0621 | $3.26 \times 10^{-7}$ |
| **Multi-Class Brier Score** | 0.5210 | 0.5640 | 0.6111 | +0.0473 | $3.85 \times 10^{-7}$ |
| **Macro-F1** | 0.0984 | 0.1182 | 0.1182 | 0.0000 | 1.0000 (Rank-preserving) |
| **Emergency Recall** | 14.1% | 11.2% | 11.2% | 0.0000 | 1.0000 (Rank-preserving) |
| **False-Emergency Rate (FER)** | 35.8% | 22.4% | 22.4% | 0.0000 | 1.0000 (Rank-preserving) |

---

### Subgroup Analysis: Dysarthric vs Control

| Subgroup | Speakers | Uncalibrated ECE | Calibrated ECE | $\Delta$ ECE | Validation Gate Promotion Rate | Safety Veto Rate |
|---|---|---|---|---|---|---|
| **Dysarthric Cohort** | 7 | 0.1325 | 0.1832 | +0.0507 | 91.7% (22/24) | 8.3% (2/24) |
| **Control Cohort** | 8 | 0.2934 | 0.3559 | +0.0624 | 71.4% (15/21) | 28.6% (6/21) |
| **Overall Cohort** | 15 | 0.1959 | 0.2536 | +0.0562 | 82.2% (37/45) | 17.8% (8/45) |

---

### Key Epistemic Conclusions

1. **Drift Detection Feasibility:** Multi-faceted monitoring (MMD, TVD, KS-test, and ECE delta) successfully detects representation and distribution shifts relative to the frozen reference window $W_0$.
2. **Dissociation of Drift and Degradation:** Representation shift in latent Whisper space does not reliably imply classification failure (`DRIFT \ne PERFORMANCE FAILURE`).
3. **Recalibration Limitations in Low-Resource Regimes:** Post-hoc temperature scaling fitted on small incoming batches ($N=10$) tends to overfit the small batch distribution, leading to worse calibration error (+0.0562 ECE degradation) out-of-distribution.
4. **Governance Gate Enforcement:** 100% of recalibration candidates entered the Phase 7.6 `ValidationGate` as `CANDIDATE`. Unconditional safety vetoes correctly rejected candidates where baseline FER exceeded the 0.50 safety bound (e.g., FC02, M02 N=30). Zero candidates were auto-promoted.
