# PRESENTATION EVIDENCE MAP & DEFENSE REFERENCE GUIDE
## Second Review Presentation & Oral Examination Defense Support

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 2026  
**Purpose:** Rapid evidence mapping for oral committee questions and presentation defense.

---

## 1. Core Research Claims to Empirical Evidence Map

| # | Defense Claim / Research Finding | Exact Empirical Value | Supporting Figure / Table | Source File Path |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Classical Emergency Recall Superiority** | **16.20%** (Classical) vs **5.10%** (Fixed Quantum) vs **5.81%** (Trainable Quantum) vs **7.07%** (Hybrid) | Final Evidence Matrix | `results/continual/phase7_12/final_evidence_matrix.csv`<br>`reports/final/FINAL_MODEL_COMPARISON.md` |
| **2** | **Statistical Significance of Quantum Recall Collapse** | Diff: $+11.10\%$ ($p_{\text{adj}} = 1.34 \times 10^{-6}$ vs Fixed)<br>Diff: $+10.39\%$ ($p_{\text{adj}} = 3.82 \times 10^{-5}$ vs Trainable) | Pairwise Statistical Tests | `results/continual/phase7_12/pairwise_statistical_tests.csv` |
| **3** | **Hybrid Macro-F1 & FER Trade-off** | Macro-F1: **0.1184** (Hybrid) vs **0.1012** (Classical)<br>FER: **19.45%** (Hybrid) vs **43.53%** (Classical) | Four-Way Comparison Table | `results/continual/phase7_10/matched_four_way_comparison.csv`<br>`results/continual/phase7_12/comprehensive_four_way_comparison.csv` |
| **4** | **No Quantum Advantage over Fixed in Trainable VQC** | Macro-F1: $0.1187 \pm 0.0039$ vs $0.1180 \pm 0.0041$ ($p = 0.061$, NS)<br>Recall: $5.81\%$ vs $5.10\%$ ($p = 0.559$, NS) | Phase 7.9 Empirical Benchmark | `results/continual/phase7_9/trainable_quantum_benchmark_results.csv` |
| **5** | **Confirmation of Historical Case B** | No statistically significant quantum advantage across static (Phase 4/5/6) or continual (Phase 7) evaluations. | Master Evidence Matrix | `results/continual/final/phase7_master_evidence_matrix.csv`<br>`reports/phase6_5_6/phase6_5_6_final_evidence_synthesis.md` |
| **6** | **Catastrophic Forgetting Prevention** | Forgetting bounded at $\Delta \le 0.0150$ using balanced class replay (100 capacity). | Replay Benchmark Table | `results/continual/phase7_5/matched_replay_comparison.csv` |
| **7** | **Zero Unverified Learning Barrier** | 0 unverified samples admitted; 100% human verification enforcement. | Codebase AST Audit | `results/continual/phase7_13/codebase_bypass_audit.json`<br>`src/continual/self_improvement.py` |
| **8** | **Safety Veto Interception** | 100% of candidate regressions vetoed; 4/60 candidates rejected; 0 regressions deployed. | Governance Decision Log | `results/continual/phase7_11/cycle_logs/`<br>`reports/continual/phase7_6/governance_summary.json` |

---

## 2. Slide-by-Slide Evidence Guide

### Slide 9: Prototype / Proof of Concept
- **Box 1 (Calibration Screen):** `assets/second_review/calibration_screen.png`
  - *Talking Point:* Demonstrates real-time guided sample intake for speaker `M01` (severe dysarthria). Acoustic quality threshold $\ge 0.30$ filters degraded input before intake.
- **Box 2 (Inference Screen):** `assets/second_review/inference_screen.png`
  - *Talking Point:* Shows Whisper-Tiny 384-D feature extraction and forward pass through active `ContinualClassicalAdapter` (24.2 ms latency), producing calibrated probability distributions.
- **Box 3 (Intent / Response):** `assets/second_review/intent_response_screen.png`
  - *Talking Point:* Demonstrates life-critical emergency dispatch trigger paired with the Mandatory Human Verification Barrier (`ACCEPT`, `CORRECT`, `REJECT`), guaranteeing zero unverified learning.

### Slide 12: Initial Results & Key Findings (Phase 3/4 Historical)
- **Box 1 (Macro-F1 vs SNR):** `assets/second_review/results_macro_f1_vs_snr.png`
  - *Talking Point:* Shows graceful degradation of the Classical Adapter under AWGN noise down to 5 dB SNR, while kernel methods collapse to majority speech.
- **Box 2 (Macro-F1 vs N):** `assets/second_review/results_macro_f1_vs_n.png`
  - *Talking Point:* Demonstrates sample efficiency across adaptation budgets $N \in \{20, 30, 40, 50\}$.
- **Box 3 (Noise × Sample Size Matrix):** `assets/second_review/results_noise_sample_heatmap.png`
  - *Talking Point:* 720-cell comparative grid confirming uniform behavior across sample budgets.
- **Box 4 (Model Comparison Matrix):** `assets/second_review/results_model_comparison.png`
  - *Talking Point:* Controlled 3-way evaluation with 95% confidence intervals establishing the initial Case B finding.

### Slide 13: Project Demonstration / Test Workflow
- **Reuses validated prototype captures:** `calibration_screen.png`, `inference_screen.png`, `intent_response_screen.png` alongside `end_to_end_demo.png`.
- *Talking Point:* Illustrates the complete 6-stage lifecycle: Calibrate $\to$ Record $\to$ Encode $\to$ Personalize $\to$ Verify $\to$ Respond.