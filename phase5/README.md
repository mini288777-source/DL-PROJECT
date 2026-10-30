# Phase 5: Controlled Noise Robustness & Acoustic Generalization

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Protocol:** Protocol B (Clean Calibration $\to$ Noisy Evaluation)  
**Status:** Executing / Validating 720 Experimental Cells

---

## Overview

Phase 5 evaluates how speaker-specific emergency intent classification models, calibrated using limited clean atypical speech ($N \in \{20, 30, 40, 50\}$), generalize to degraded acoustic conditions.

### Experimental Grid: 720 Cells
- **15 Speakers:** 8 dysarthric (`F01, F03, F04, M01, M02, M03, M04, M05`) and 7 control (`FC01, FC02, FC03, MC01, MC02, MC03, MC04`).
- **4 Sample Budgets:** $N \in \{20, 30, 40, 50\}$.
- **3 Models:**
  1. `adapter`: Classical Adapter Head (25,223 params).
  2. `rbf_svm`: Classical RBF-Kernel SVM ($C=1.0$).
  3. `qsvc`: Primary Quantum QSVC ($q=4, \text{reps}=1, \text{linear entanglement}, C=1.0$).
- **4 Test Acoustic Conditions:** Clean, 15 dB SNR, 10 dB SNR, 5 dB SNR.
- **Total:** $15 \times 4 \times 3 \times 4 = 720$ experimental conditions.

---

## Directory Structure

```
phase5/
├── README.md                          # This file
├── config/
│   └── phase5_config.yaml             # Frozen experimental configuration
├── manifests/
│   ├── phase5_experiment_manifest.csv # Full 720-cell experiment manifest
│   └── phase5_test_manifest.csv       # Test audio perturbation manifest
├── embeddings/                        # Noisy Whisper embeddings (mirrored)
├── results/
│   ├── phase5_results.csv             # Primary 720-cell results table
│   ├── phase5_summary.csv             # Aggregated mean/std summary table
│   ├── phase5_per_class.csv           # Intent-level precision, recall, F1
│   ├── phase5_predictions.csv         # Recording-level prediction traces
│   └── phase5_errors.csv              # Error tracking log
├── figures/
│   ├── macro_f1_vs_snr.png            # Figure 1: Macro-F1 vs SNR degradation
│   ├── macro_f1_vs_sample_size.png    # Figure 2: Sample efficiency curves
│   ├── noise_sample_efficiency_heatmap.png # Figure 3: N x SNR heatmaps
│   ├── iper_vs_snr.png                # Figure 4: Emergency IPER vs SNR
│   └── model_comparison.png           # Figure 5: Controlled 3-way comparison
└── reports/
    ├── phase5_readiness_report.md     # Readiness audit & pre-flight gate
    ├── phase5_results_report.md       # Complete scientific results analysis
    └── phase5_reproducibility_report.md # Environment & checksum audit
```
