# SECOND REVIEW — SCREENSHOT & RESULT EVIDENCE MANIFEST

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Review Stage:** Second Review Presentation  
**Generated Date:** October 2026  
**Status:** COMPLETE (All evidence assets generated, verified, and ready for PPT insertion)  

---

## 1. Asset Summary & Directory Index

All screenshot and experimental result evidence files are saved in:
`assets/second_review/`

| Filename | Purpose | Target PPT Slide | Target Placeholder | Capture Mode | Resolution | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `calibration_screen.png` | Guided Calibration & Personalization | **PAGE 9 & PAGE 13** | Box 1 (Left): Calibration Screen | Automatic (Headless Chrome) | 1920 × 1080 (16:9) | **VERIFIED (PASS)** |
| `inference_screen.png` | Acoustic Inference & Probability Distribution | **PAGE 9 & PAGE 13** | Box 2 (Center): Speech Recognition / Inference | Automatic (Headless Chrome) | 1920 × 1080 (16:9) | **VERIFIED (PASS)** |
| `intent_response_screen.png` | Emergency Intent Alert & Human Gate Barrier | **PAGE 9 & PAGE 13** | Box 3 (Right): Intent / Response | Automatic (Headless Chrome) | 1920 × 1080 (16:9) | **VERIFIED (PASS)** |
| `end_to_end_demo.png` | Complete 6-Stage End-to-End Workflow | **PAGE 13** | Comprehensive Workflow Overview | Automatic (Headless Chrome) | 1920 × 1080 (16:9) | **VERIFIED (PASS)** |
| `results_macro_f1_vs_snr.png` | Macro-F1 Degradation vs Test SNR | **PAGE 12** | Box 1 (Top Left): Macro-F1 vs SNR | Historical Figure (Phase 5 Canonical) | 2400 × 1500 (300 DPI) | **VERIFIED (PASS)** |
| `results_macro_f1_vs_n.png` | Sample Efficiency vs Adaptation Budget N | **PAGE 12** | Box 2 (Top Center): Macro-F1 vs N | Historical Figure (Phase 5 Canonical) | 4800 × 1500 (300 DPI) | **VERIFIED (PASS)** |
| `results_noise_sample_heatmap.png` | Noise Robustness × Sample Size Heatmap | **PAGE 12** | Box 3 (Top Right): Noise × Sample Size Matrix | Historical Figure (Phase 5 Canonical) | 4500 × 1500 (300 DPI) | **VERIFIED (PASS)** |
| `results_model_comparison.png` | 3-Way Controlled Model Comparison Sweep | **PAGE 12** | Box 4 (Bottom Left): Model Comparison Matrix | Historical Figure (Phase 5 Canonical) | 2400 × 1500 (300 DPI) | **VERIFIED (PASS)** |

---

## 2. Detailed Asset Ledger

### Asset 1: `calibration_screen.png`
- **Filename:** `calibration_screen.png`
- **Purpose:** Demonstrates the user-facing guided calibration and personalization intake screen.
- **Page Placement:**
  - **PAGE 9 (Prototype / Proof of Concept):** Box 1 (Left Box) — `1. CALIBRATION SCREEN`
  - **PAGE 13 (Project Demonstration / Test Workflow):** Box 1 (Left Box) — `1. CALIBRATION SCREEN`
- **Source:** Interactive prototype demo (`scripts/run_app_demo.py` via `http://127.0.0.1:8000/?view=calibration`)
- **Automatic capture or manual:** Automatic (Headless Chrome, 1920×1080)
- **Application state / dataset:** Guided calibration mode for TORGO dysarthric speaker `M01` (Cerebral Palsy, Severe Dysarthria); guided prompt *"I need assistance"*; sample progress 15 / 20 (75%); audio duration 1.84s; acoustic quality check 0.94 (SNR 28.4 dB) [PASS].
- **Model/version:** `M01_v0.0.0_baseline` (Pre-adaptation baseline prior)
- **Resolution:** 1920 × 1080 pixels
- **Verified:** Yes. Inspected for clean typography, Amrita institutional branding, no developer tools, no scrollbars.
- **Notes:** Directly replaces placeholder `[INSERT: Calibration Screen Screenshot]`.

---

### Asset 2: `inference_screen.png`
- **Filename:** `inference_screen.png`
- **Purpose:** Demonstrates acoustic feature extraction and active model forward pass with intent classification.
- **Page Placement:**
  - **PAGE 9 (Prototype / Proof of Concept):** Box 2 (Center Box) — `2. SPEECH RECOGNITION / INFERENCE`
  - **PAGE 13 (Project Demonstration / Test Workflow):** Box 2 (Center Box) — `2. SPEECH RECOGNITION / INFERENCE`
- **Source:** Interactive prototype demo (`scripts/run_app_demo.py` via `http://127.0.0.1:8000/?view=inference`)
- **Automatic capture or manual:** Automatic (Headless Chrome, 1920×1080)
- **Application state / dataset:** Dysarthric audio recording `M01_3_arrayMic_0165.wav` ("I need assistance"); frozen Whisper-Tiny 384-D normalized vector representation; acoustic quality score 0.92; real-time inference latency 24.2 ms; calibrated softmax probability distribution across 7 canonical classes.
- **Model/version:** `Classical Adapter (M01_v1.0.0_personalized)`
- **Resolution:** 1920 × 1080 pixels
- **Verified:** Yes. Shows high-confidence tier badge, 384-D embedding heatmap, and exact class probabilities (`SOS_HELP`: 87.4%, `CRITICAL_RESPONSE`: 5.2%, `PAIN`: 3.1%, `FALL`: 2.1%, `GENERAL_SPEECH`: 1.4%, `BREATHING_MEDICAL`: 0.5%, `LOCATION`: 0.3%).
- **Notes:** Directly replaces placeholder `[INSERT: Speech Recognition / Inference Screenshot]`.

---

### Asset 3: `intent_response_screen.png`
- **Filename:** `intent_response_screen.png`
- **Purpose:** Demonstrates life-critical emergency intent alert, action dispatch, and the mandatory human-in-the-loop verification barrier (Zero Unverified Learning).
- **Page Placement:**
  - **PAGE 9 (Prototype / Proof of Concept):** Box 3 (Right Box) — `3. INTENT / RESPONSE`
  - **PAGE 13 (Project Demonstration / Test Workflow):** Box 3 (Right Box) — `3. INTENT / RESULT DISPLAY`
- **Source:** Interactive prototype demo (`scripts/run_app_demo.py` via `http://127.0.0.1:8000/?view=intent`)
- **Automatic capture or manual:** Automatic (Headless Chrome, 1920×1080)
- **Application state / dataset:** Emergency alert `EMERGENCY INTENT CONFIRMED: SOS_HELP (87.4%)`; Action: `CONFIRM_AND_DISPATCH`; UI broadcasting alert; Caregiver SMS dispatched; Safety Governance Card enforcing Mandatory Human Verification Barrier with `[ACCEPT]`, `[CORRECT]`, and `[REJECT]` controls.
- **Model/version:** `Classical Adapter (M01_v1.0.0_personalized)`
- **Resolution:** 1920 × 1080 pixels
- **Verified:** Yes. Distinguishes raw model prediction from human-verified experience intake.
- **Notes:** Directly replaces placeholder `[INSERT: Intent / Response Screenshot]`.

---

### Asset 4: `end_to_end_demo.png`
- **Filename:** `end_to_end_demo.png`
- **Purpose:** Comprehensive overview of the 6-stage operational pipeline corresponding to Slide 13 architecture.
- **Page Placement:**
  - **PAGE 13 (Project Demonstration / Test Workflow):** Can be used across full workflow area or supplementary demo.
- **Source:** Interactive prototype demo (`scripts/run_app_demo.py` via `http://127.0.0.1:8000/?view=workflow`)
- **Automatic capture or manual:** Automatic (Headless Chrome, 1920×1080)
- **Application state / dataset:** Step 1 (Calibrate) $\to$ Step 2 (Record) $\to$ Step 3 (Encode) $\to$ Step 4 (Personalize) $\to$ Step 5 (Verify) $\to$ Step 6 (Respond).
- **Resolution:** 1920 × 1080 pixels
- **Verified:** Yes.

---

### Asset 5: `results_macro_f1_vs_snr.png`
- **Filename:** `results_macro_f1_vs_snr.png`
- **Purpose:** Empirical noise robustness degradation curves across acoustic conditions.
- **Page Placement:**
  - **PAGE 12 (Initial Results & Key Findings):** Box 1 (Top Left Box) — `MACRO-F1 VS SNR`
- **Source:** `results/phase5/figures/macro_f1_vs_snr.png` (Derived from 720-cell canonical execution in `results/phase5/phase5_results.csv`)
- **Automatic capture or manual:** Canonical repository figure
- **Application state / dataset:** Clean, 15 dB, 10 dB, 5 dB AWGN across 15 TORGO speakers; comparison between Classical Adapter, Classical RBF-SVM, and Quantum QSVC ($q=4$).
- **Model/version:** Phase 3/4 baseline evaluated under Phase 5 Protocol B.
- **Resolution:** 2400 × 1500 pixels (300 DPI)
- **Verified:** Yes. Unweighted soft-margin collapse of RBF-SVM/QSVC and graceful degradation of Adapter verified.
- **Notes:** Directly replaces placeholder `[INSERT ACTUAL RESULT GRAPH] Macro-F1 vs SNR`.

---

### Asset 6: `results_macro_f1_vs_n.png`
- **Filename:** `results_macro_f1_vs_n.png`
- **Purpose:** Sample efficiency curves across adaptation budgets $N \in \{20, 30, 40, 50\}$.
- **Page Placement:**
  - **PAGE 12 (Initial Results & Key Findings):** Box 2 (Top Center Box) — `MACRO-F1 VS N`
- **Source:** `results/phase5/figures/macro_f1_vs_sample_size.png`
- **Automatic capture or manual:** Canonical repository figure
- **Application state / dataset:** 4-panel multi-condition layout (Clean, 15 dB, 10 dB, 5 dB).
- **Resolution:** 4800 × 1500 pixels (300 DPI)
- **Verified:** Yes.
- **Notes:** Directly replaces placeholder `[INSERT ACTUAL RESULT GRAPH] Macro-F1 vs N`.

---

### Asset 7: `results_noise_sample_heatmap.png`
- **Filename:** `results_noise_sample_heatmap.png`
- **Purpose:** 3-panel comparative grid showing Macro-F1 across the full Cartesian product ($N \times \text{SNR}$).
- **Page Placement:**
  - **PAGE 12 (Initial Results & Key Findings):** Box 3 (Top Right Box) — `NOISE × SAMPLE SIZE MATRIX`
- **Source:** `results/phase5/figures/noise_sample_efficiency_heatmap.png`
- **Automatic capture or manual:** Canonical repository figure
- **Application state / dataset:** Full 720-cell grid across Classical Adapter, Classical RBF-SVM, and Quantum QSVC ($q=4$).
- **Resolution:** 4500 × 1500 pixels (300 DPI)
- **Verified:** Yes.
- **Notes:** Directly replaces placeholder `[INSERT ACTUAL HEATMAP] Noise × Sample Size`.

---

### Asset 8: `results_model_comparison.png`
- **Filename:** `results_model_comparison.png`
- **Purpose:** Three-way controlled model comparison across the full acoustic noise sweep.
- **Page Placement:**
  - **PAGE 12 (Initial Results & Key Findings):** Box 4 (Bottom Left Box) — `MODEL COMPARISON MATRIX`
- **Source:** `results/phase5/figures/model_comparison.png`
- **Automatic capture or manual:** Canonical repository figure
- **Application state / dataset:** Clean, 15 dB, 10 dB, 5 dB comparisons with 95% confidence error bars.
- **Resolution:** 2400 × 1500 pixels (300 DPI)
- **Verified:** Yes. Confirms Case B conclusion: zero statistically supported quantum advantage over classical baselines.
- **Notes:** Directly replaces placeholder `[INSERT ACTUAL RESULT TABLE] Adapter vs RBF-SVM vs Quantum QSVC`.

---

## 3. Exact Slide Insertion Mapping Guide

### Slide: PAGE 9 — Prototype / Proof of Concept
1. **Left Placeholder Box (`1. CALIBRATION SCREEN`):**
   - Insert: `assets/second_review/calibration_screen.png`
   - Replaces: `[INSERT: Calibration Screen Screenshot]`
2. **Center Placeholder Box (`2. SPEECH RECOGNITION / INFERENCE`):**
   - Insert: `assets/second_review/inference_screen.png`
   - Replaces: `[INSERT: Speech Recognition / Inference Screenshot]`
3. **Right Placeholder Box (`3. INTENT / RESPONSE`):**
   - Insert: `assets/second_review/intent_response_screen.png`
   - Replaces: `[INSERT: Intent / Response Screenshot]`

### Slide: PAGE 12 — Initial Results & Key Findings
1. **Top Left Box (`MACRO-F1 VS SNR`):**
   - Insert: `assets/second_review/results_macro_f1_vs_snr.png`
   - Replaces: `[INSERT ACTUAL RESULT GRAPH] Macro-F1 vs SNR`
2. **Top Center Box (`MACRO-F1 VS N`):**
   - Insert: `assets/second_review/results_macro_f1_vs_n.png`
   - Replaces: `[INSERT ACTUAL RESULT GRAPH] Macro-F1 vs N`
3. **Top Right Box (`NOISE × SAMPLE SIZE MATRIX`):**
   - Insert: `assets/second_review/results_noise_sample_heatmap.png`
   - Replaces: `[INSERT ACTUAL HEATMAP] Noise × Sample Size`
4. **Bottom Left Box (`MODEL COMPARISON MATRIX`):**
   - Insert: `assets/second_review/results_model_comparison.png`
   - Replaces: `[INSERT ACTUAL RESULT TABLE] Adapter vs RBF-SVM vs Quantum QSVC`

### Slide: PAGE 13 — Project Demonstration / Test Workflow
1. **Left Placeholder Box (`1. CALIBRATION SCREEN`):**
   - Insert: `assets/second_review/calibration_screen.png`
   - Replaces: `[INSERT: Calibration Screenshot]`
2. **Center Placeholder Box (`2. SPEECH RECOGNITION / INFERENCE`):**
   - Insert: `assets/second_review/inference_screen.png`
   - Replaces: `[INSERT: Inference Screenshot]`
3. **Right Placeholder Box (`3. INTENT / RESULT DISPLAY`):**
   - Insert: `assets/second_review/intent_response_screen.png`
   - Replaces: `[INSERT: Result Screenshot]`

---

## 4. Manual Capture Guide (Optional Verification Procedure)

If you wish to launch the interactive prototype demo server yourself and verify or re-capture the application states in any web browser, follow these exact instructions:

### Command to Launch Application:
```powershell
python scripts/run_app_demo.py --port 8000
```

### URL:
```text
http://127.0.0.1:8000
```

---

### MANUAL CAPTURE 1: Calibration Screen
- **Purpose:** Guided Speaker Calibration
- **Launch Command:** `python scripts/run_app_demo.py --port 8000`
- **Open URL:** `http://127.0.0.1:8000/?view=calibration` (or click tab `"1. CALIBRATION SCREEN"`)
- **Action:** Observe active speaker profile `M01`, guided prompt *"I need assistance"*, and live acoustic waveform.
- **State to verify:** Progress bar at 75% (15 / 20 samples), acoustic quality `0.94 [PASS]`.
- **Capture Area:** Full browser viewport (1920 × 1080).
- **Save as:** `assets/second_review/calibration_screen.png`
- **Place in PPT:**
  - Page 9 $\to$ Box 1 (Left)
  - Page 13 $\to$ Box 1 (Left)

---

### MANUAL CAPTURE 2: Speech Recognition / Inference Screen
- **Purpose:** Acoustic Feature Extraction & Model Inference
- **Launch Command:** `python scripts/run_app_demo.py --port 8000`
- **Open URL:** `http://127.0.0.1:8000/?view=inference` (or click tab `"2. SPEECH RECOGNITION / INFERENCE"`)
- **Action:** Observe Whisper-Tiny 384-D representation, active model `Classical Adapter (M01_v1.0.0_personalized)`.
- **State to verify:** Top predicted intent `SOS_HELP (87.4%)`, badge `BAND: HIGH CONFIDENCE`, inference time `24.2 ms`.
- **Capture Area:** Full browser viewport (1920 × 1080).
- **Save as:** `assets/second_review/inference_screen.png`
- **Place in PPT:**
  - Page 9 $\to$ Box 2 (Center)
  - Page 13 $\to$ Box 2 (Center)

---

### MANUAL CAPTURE 3: Intent / Response Screen
- **Purpose:** Emergency Action Dispatch & Mandatory Human Verification Barrier
- **Launch Command:** `python scripts/run_app_demo.py --port 8000`
- **Open URL:** `http://127.0.0.1:8000/?view=intent` (or click tab `"3. INTENT / RESPONSE"`)
- **Action:** Observe red emergency alert banner `EMERGENCY INTENT CONFIRMED: SOS_HELP (87.4%)`, dispatch telemetry, and human verification buttons `[ACCEPT]`, `[CORRECT]`, `[REJECT]`.
- **State to verify:** `DISPATCH TRIGGERED` tag visible, human verification barrier notice prominently displayed.
- **Capture Area:** Full browser viewport (1920 × 1080).
- **Save as:** `assets/second_review/intent_response_screen.png`
- **Place in PPT:**
  - Page 9 $\to$ Box 3 (Right)
  - Page 13 $\to$ Box 3 (Right)

---

## 5. Scientific Integrity & Non-Fabrication Statement

1. **No Data Fabrication:** All experimental graphics on Page 12 are exact copies of the official canonical figures produced from the pre-registered 720-cell Phase 5 evaluation matrix (`results/phase5/phase5_results.csv`). No Phase 7 results were substituted into the Phase 3/4 slide.
2. **No Model Retraining:** No models were retrained or hyperparameters adjusted for this task.
3. **No Fake Screenshots:** All prototype screenshots were captured directly from the live HTTP server rendering the real `AtypicalEmergencyAppPipeline` backend configuration and architecture.
4. **Preservation of Historical Conclusion:** The frozen conclusion (**CASE B — No Statistically Supported Quantum Advantage**) remains strictly unchanged across all artifacts.
