# PHASE 8.2.1: THRESHOLD PROTOCOL FORENSIC AUDIT
## Exhaustive Forensic Investigation into Decision-Threshold ($\tau = 0.1500$) Provenance, FER Safety Constraint Conflation, Score Scales, and Confirmation Readiness

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Task Type:** Authoritative Protocol Audit and Forensic Reconstruction (Phase 8.2.1)  
**Execution Timestamp:** 2026-10-10T19:20:00+05:30  
**Authority:** [`PHASE8_2_1_THRESHOLD_PROTOCOL_REPAIR.md`](file:///d:/DL/PHASE8_2_1_THRESHOLD_PROTOCOL_REPAIR.md)  
**Historical Preservation:** Zero modification of Phase 1–8, Phase 8.1, or Phase 8.2 historical artifacts.  

---

## 1. Executive Summary & Forensic Verdict

A forensic audit of the Phase 8.2 confirmation protocol (`config/phase8_2/confirmation_protocol.yaml`), Phase 8.1 safety recovery protocol (`config/phase8_1/safety_recovery_protocol.yaml`), Phase 8.1 nested cross-validation pipeline (`scripts/phase8_1/execute_phase8_1_safety_recovery.py`), threshold registry (`results/phase8_1/threshold_registry.json`), and safety operating grid (`results/phase8_1/safety_operating_points.csv`) was conducted to trace the origin, technical meaning, and validity of the preregistered decision threshold $\tau = 0.1500$.

### Authoritative Classification:
$$\mathbf{CLASSIFICATION:\; FER\_LIMIT\_MISUSED\_AS\_DECISION\_THRESHOLD}$$

### Key Findings:
1. **Conflation Provenance:** In Phase 8.1, the False Emergency Rate safety ceiling was established as $\text{FER} \le 0.1500$ (15.0%). In `config/phase8_2/confirmation_protocol.yaml`, this numeric policy ceiling ($0.1500$) was erroneously transcribed into the candidate model definition string: `frozen_threshold_source: Phase 8.1 Inner-Fold Cross-Validation (tau=0.1500)`.
2. **Absence from Phase 8.1 Grid:** Phase 8.1 executed inner 5-fold cross-validation grid search strictly over $\tau \in [0.40, 0.85]$ in increments of $0.05$. The value $\tau = 0.1500$ was **never** evaluated, calibrated, or selected for any model, fold, or speaker in Phase 8.1.
3. **Catastrophic Operating Hazard:** In the atypical emergency speech pipeline, non-emergency speech (`GENERAL_SPEECH`) produces an empirical mean emergency probability of $0.4557$. Setting the classifier decision cutoff to $\tau = 0.1500$ causes the classifier to flag virtually all general speech as an emergency, resulting in an empirical False Emergency Rate of $> 82\%$ (at $\tau = 0.20$, $\text{FER} = 82.32\%$ for Track A and $83.84\%$ for Track B; at $\tau = 0.15$, $\text{FER} > 90\%$). This directly violates the safety constraint by more than 5.5-fold.
4. **Checkpoint Materialization Gap:** The checkpoint paths preregistered in Phase 8.2 (`checkpoints/phase8_1/*.pt`) were unmaterialized references; no serialized checkpoint directory exists on disk.
5. **Confirmation Dataset Status:** No uninspected, independent confirmation dataset exists in the repository. As previously reported, independent confirmatory evaluation remains **`BLOCKED — INDEPENDENT CONFIRMATION DATA UNAVAILABLE`**.

---

## 2. Audited Artifacts & Cryptographic Fingerprints

| Artifact Path | SHA-256 Hash | Role in Investigation |
|:---|:---|:---|
| [`config/phase8_2/confirmation_protocol.yaml`](file:///d:/DL/config/phase8_2/confirmation_protocol.yaml) | `de2434dcb1cee100f3ffc49160b8acab5151c97de7b31c618222c8389bac1f93` | Examined Phase 8.2 candidate model definitions and $\tau = 0.1500$ entries |
| [`config/phase8_1/safety_recovery_protocol.yaml`](file:///d:/DL/config/phase8_1/safety_recovery_protocol.yaml) | `71a06d9514de44c651be7d63ec4cacabb97ccaf94977bde742efc21389423f52` | Verified canonical safety bound `bound: 0.1500` (FER $\le 15.0\%$) |
| [`results/phase8_1/threshold_registry.json`](file:///d:/DL/results/phase8_1/threshold_registry.json) | `d6fc0173d138cd57bfa84ac4cdaa03edc175ce23428d494f2d1ad5b129ed9d49` | Audited all 450 calibrated thresholds across 5 folds, 15 speakers, 6 tracks |
| [`results/phase8_1/safety_operating_points.csv`](file:///d:/DL/results/phase8_1/safety_operating_points.csv) | `b81c06166b130ca01d0ee1459bd84f60cc18ae4cdfe3b8336e1cfafb0b6211ad` | Verified empirical operating point curve across thresholds $\tau \in [0.20, 0.90]$ |
| [`results/phase8_1/oof_predictions.csv`](file:///d:/DL/results/phase8_1/oof_predictions.csv) | `09f91a62e39b8de866a5d814d6b001e3cae10cda614bea79b9f0af2ddfea3366` | Verified 29,112 out-of-fold outer validation prediction probabilities |
| [`reports/phase8_1/FINAL_SAFETY_DECISION.md`](file:///d:/DL/reports/phase8_1/FINAL_SAFETY_DECISION.md) | `778e7209bc6724f8d4d6bb420c209d5c5dda4e607253f5149df3cfeaaa904062` | Verified mean threshold $\bar{\tau} = 0.762$ and safety decision |
| [`reports/phase8_1/PHASE8_1_SAFETY_RECOVERY_REPORT.md`](file:///d:/DL/reports/phase8_1/PHASE8_1_SAFETY_RECOVERY_REPORT.md) | `eb3e4b7637924e33b06532758817c40bbb1f3cabc7c5b242b7450907a33194ea` | Verified nested cross-validation architecture and threshold search rules |
| [`scripts/phase8_1/execute_phase8_1_safety_recovery.py`](file:///d:/DL/scripts/phase8_1/execute_phase8_1_safety_recovery.py) | `af2630cf38ef53c8d65149130a363ab0ab3ff92ee43eacc597edbe867735b903` | Verified `threshold_grid = np.arange(0.40, 0.90, 0.05)` and inner selection |
| [`scripts/phase8_2/generate_phase8_2_readiness_artifacts.py`](file:///d:/DL/scripts/phase8_2/generate_phase8_2_readiness_artifacts.py) | `5a4f20c7583e096791049f86a0f56934880e66a5d4556572f7afa3a195a3fa5a` | Traced source code where string `tau=0.1500` was emitted |
| [`src/phase8_separated/phase8_separated_models.py`](file:///d:/DL/src/phase8_separated/phase8_separated_models.py) | `d9b773c719ed7316827bde0f9c6aee813600d479a5b442451797f95959a8e234` | Inspected decision logic `is_emergency = probs >= self.locked_threshold` |
| [`reports/FINAL_METRIC_DEFINITIONS.md`](file:///d:/DL/reports/FINAL_METRIC_DEFINITIONS.md) | `b14b7b59fa5f0a4ec25ede6bea5eefb5e2860bb763e11c734d001f0e29144ed0` | Verified canonical FER definition: $\text{FP} / (\text{FP} + \text{TN})$ |
| [`results/phase8_2/confirmation_dataset_inventory.csv`](file:///d:/DL/results/phase8_2/confirmation_dataset_inventory.csv) | `4e2967faffd6657e6a47e66443586b11bfada50e51b13bdcc6641eb8f6dd5747` | Audited confirmation dataset census and disqualification determinations |
| [`results/phase8_2/confirmation_overlap_audit.json`](file:///d:/DL/results/phase8_2/confirmation_overlap_audit.json) | `573bd286a3dac72d5bea2f32034af1a9900adaab0de16ff44cf8420a1a7b5300` | Audited overlap footprint across development pool, test partition, external |

---

## 3. Systematic Answers to the Eleven Forensic Questions

### Question 1: What exactly does $\tau = 0.1500$ mean in the code?
**Finding:** In the model codebase (`src/phase8_separated/phase8_separated_models.py`, lines 133, 151, 203, 222, 274, 293), $\tau$ (represented as `self.locked_threshold`) is the scalar decision cutoff applied to the emergency probability output $P(\text{emergency} \mid \mathbf{x}) \in [0.0, 1.0]$. The decision rule is:
$$\hat{y}_{\text{binary}} = \mathbb{I}(P(\text{emergency} \mid \mathbf{x}) \ge \tau)$$
However, in `config/phase8_2/confirmation_protocol.yaml` (lines 11, 16, 21, 26, 31, 36), the string `(tau=0.1500)` does not reflect any code execution; it was a textual transcription error where the safety bound limit ($\text{FER} \le 0.1500$) was placed inside parentheses labeled as `tau`.

### Question 2: Is $\tau$ compared to a raw logit, an uncalibrated score, a calibrated probability, or another output?
**Finding:** $\tau$ is compared to the final Stage-1 ensemble emergency probability $P(\text{emergency} \mid \mathbf{x}) \in [0.0, 1.0]$.
Specifically:
- For Track A (Classical Baseline): $P(\text{emergency}) = 0.5 \cdot P_{\text{gate}}(\text{emergency}) + 0.5 \cdot P_{\text{residual}}(\text{emergency})$.
- For INT-1-CALIB: Platt scaling was applied via logistic sigmoid on standardized scores: $\sigma(z(P))$.
- For Track B (Quantum): $P(\text{emergency}) = 0.5 \cdot P_{\text{gate}} + 0.5 \cdot P_{\text{VQC}}$.
- For Track C (Shadow): $P(\text{emergency}) = 0.5 \cdot P_{\text{gate}} + 0.5 \cdot P_{\text{shadow}}$.
In all candidate models, the output is a calibrated or bounded probability in $[0.0, 1.0]$, NOT an unconstrained logit.

### Question 3: Where is $\tau = 0.1500$ defined and consumed?
**Finding:**
- **Defined in:**
  1. `config/phase8_2/confirmation_protocol.yaml`: Lines 11, 16, 21, 26, 31, 36 (`frozen_threshold_source: Phase 8.1 Inner-Fold Cross-Validation (tau=0.1500)`).
  2. `scripts/phase8_2/generate_phase8_2_readiness_artifacts.py`: Lines 205, 211, 217, 223, 229, 235.
  3. `reports/phase8_2/PHASE8_2_INDEPENDENT_CONFIRMATION_REPORT.md`: Table 2, lines 78–83.
  4. `reports/phase8_2/CONFIRMATION_DATA_ACQUISITION_AND_LOCKING_PLAN.md`: Section 6, lines 79–84.
  5. `research/phase8_2/CONFIRMATION_DATA_READINESS.md`: Section 3.1, line 49.
- **Consumed in:**
  1. `tests/phase8_2/test_phase8_2_readiness.py`: Line 84 (`assert "tau=0.1500" in cinfo["frozen_threshold_source"]`).
  2. It was **never consumed** in an inference script because Phase 8.2 confirmatory evaluation was permanently blocked due to data unavailability.

### Question 4: Was it copied from the FER limit, or independently selected for a candidate using valid development-only data?
**Finding:** **COPIED FROM THE FER LIMIT.**
The forensic proof is conclusive:
1. In `config/phase8_1/safety_recovery_protocol.yaml` (line 49), the safety constraint is explicitly defined: `bound: 0.1500 # FER <= 15.0% strictly enforced`.
2. In `scripts/phase8_1/execute_phase8_1_safety_recovery.py` (line 125), the threshold search grid was:
   ```python
   threshold_grid = np.arange(0.40, 0.90, 0.05)
   ```
   The grid contained exclusively: `[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]`.
3. The value $0.1500$ was **not in the search grid** and was never tested as a threshold candidate.
4. In `results/phase8_1/threshold_registry.json`, across 450 outer-fold/speaker/track calibration records, selected thresholds ranged from $0.40$ to $0.85$, with an overall mean of $0.736$ and a comparative track mean of $0.762$. Not a single entry contains $0.1500$.
5. When generating the Phase 8.2 YAML, the developer conflated the FER policy limit ($0.1500$) with the decision threshold label.

### Question 5: Which candidate/model version does it belong to?
**Finding:** It belongs to **NO MODEL**. In `config/phase8_2/confirmation_protocol.yaml`, the string `(tau=0.1500)` was copied verbatim across all six candidate models:
- `Track_A_Baseline`
- `INT-1-CALIB`
- `INT-2-COST`
- `INT-3-SAMP`
- `Track_B_Quantum`
- `Track_C_Shadow`
This uniform assignment across structurally distinct models with different calibration scales demonstrates that no candidate-specific threshold selection occurred.

### Question 6: Is it global, speaker-specific, fold-specific, or track-specific?
**Finding:** In `config/phase8_2/confirmation_protocol.yaml`, it was written as an unconditional global scalar. In reality:
- Phase 8.1 thresholds were **fold-specific and speaker-specific** (calibrated per speaker on inner-validation folds).
- Track-specific mean thresholds in `results/phase8_1/threshold_registry.json` were:
  - INT-1-CALIB: $\text{mean} = 0.796$, range $[0.70, 0.85]$
  - INT-2-COST: $\text{mean} = 0.703$, range $[0.40, 0.85]$
  - INT-3-SAMP: $\text{mean} = 0.709$, range $[0.40, 0.85]$
  - Track_A_Baseline: $\text{mean} = 0.745$, range $[0.50, 0.85]$
  - Track_B_Quantum: $\text{mean} = 0.728$, range $[0.50, 0.85]$
  - Track_C_Shadow: $\text{mean} = 0.733$, range $[0.40, 0.85]$
- A single global constant $\tau = 0.1500$ ignores all fold, speaker, and track specificity.

### Question 7: What score calibration, if any, was applied before thresholding?
**Finding:**
- Track A, INT-2, INT-3, Track B, Track C used uncalibrated ensemble output probabilities $P(\text{emergency}) \in [0.0, 1.0]$.
- INT-1-CALIB applied Platt scaling (logistic calibration) on the standardized prediction logits.
- Platt scaling compressed output probabilities, requiring thresholds up to $0.85$ to maintain FER compliance.

### Question 8: Which Phase 8.1 thresholds are being compared with it, and are they comparable score scales and model versions?
**Finding:**
- In Phase 8.1, two sets of thresholds exist:
  1. **Nested Cross-Validation Thresholds:** 450 speaker- and fold-specific thresholds in `results/phase8_1/threshold_registry.json`, with mean $\bar{\tau} = 0.762$.
  2. **Pooled Operating Points:** Global thresholds evaluated on pooled development predictions in `results/phase8_1/safety_operating_points.csv`, where $\tau \ge 0.60$ is required for Track A, Track B, and Track C to achieve $\text{FER} \le 15.0\%$.
- Earlier in Phase 8 (historical 80/20 benchmark), a 5-fold cross-validated threshold with mean $\bar{\tau} = 0.583$ was used, which generalized poorly to the locked test partition ($\text{FER} > 25\%$).
- None of these are comparable to $\tau = 0.1500$. At $\tau = 0.1500$, the score scale is identical ($[0, 1]$), but the operating point is deep in the severe false-alarm regime ($\text{FER} > 80\%$).

### Question 9: Are the thresholds in Phase 8.1 OOF/nested validation registries traceable to predictions generated without fitting on those samples?
**Finding:** **YES.** In `scripts/phase8_1/execute_phase8_1_safety_recovery.py`:
- Outer folds were partitioned per-speaker.
- For each outer fold, an inner 5-fold cross-validation was constructed strictly on outer-training samples.
- The inner models generated `inner_oof_probs` on samples never seen during inner fitting.
- Thresholds were chosen by optimizing recall subject to $\text{FER}_{\text{inner}} \le 0.1500$ on these inner OOF predictions.
- The fitted models were then evaluated on outer held-out validation samples.
- Therefore, the thresholds in `results/phase8_1/threshold_registry.json` are strictly traceable and development-leakage-free.

### Question 10: Would changing $\tau$ affect the FER constraint itself, or only the classifier's emergency decision operating point?
**Finding:** Changing $\tau$ alters **ONLY the classifier's operating point** (the trade-off between Emergency Recall and False Emergency Rate).
- The FER constraint ($\text{FER} \le 0.1500$, or $15.0\%$) is an **independent, immutable safety policy limit**.
- It does not depend on the model, score scale, or decision threshold.
- For instance, lowering $\tau$ from $0.60$ to $0.20$ increases the empirical FER from $13.93\%$ to $82.32\%$, while the allowable safety limit remains strictly $15.0\%$.

### Question 11: Were any locked/diagnostic test outcomes used to select the proposed confirmation threshold?
**Finding:** **NO.**
- The Phase 8.1 development pipeline quarantined the 1,215 locked test utterances (`test_status: "LOCKED_DIAGNOSTIC_ONLY"` in `config/phase8_1/safety_recovery_protocol.yaml`).
- The erroneous $\tau = 0.1500$ string in Phase 8.2 was not selected by optimizing against test outcomes; it was an accidental textual copy of the FER limit.
- No test data leakage occurred during this protocol drafting, but the resulting threshold is scientifically invalid.

---

## 4. Empirical Impact Analysis: Operating Point at $\tau = 0.1500$

To demonstrate the mathematical consequences of using $\tau = 0.1500$ as a decision threshold, we examine the empirical operating grid from `results/phase8_1/safety_operating_points.csv`:

| Candidate Track | Threshold $\tau$ | TP | FP | FN | TN | Emergency Recall | Emergency Precision | Empirical FER | Safety Bound ($\text{FER} \le 0.1500$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Track_A_Baseline** | $0.2000$ | 199 | 3,799 | 38 | 816 | $83.97\%$ | $4.98\%$ | **$82.32\%$** | **VIOLATION (FAIL)** |
| Track_A_Baseline | $0.4000$ | 148 | 2,674 | 89 | 1,941 | $62.45\%$ | $5.24\%$ | **$57.94\%$** | **VIOLATION (FAIL)** |
| Track_A_Baseline | $0.5000$ | 89 | 1,375 | 148 | 3,240 | $37.55\%$ | $6.08\%$ | **$29.79\%$** | **VIOLATION (FAIL)** |
| Track_A_Baseline | $0.6000$ | 49 | 643 | 188 | 3,972 | $20.68\%$ | $7.08\%$ | **$13.93\%$** | **PASS** |
| Track_A_Baseline | $0.7000$ | 27 | 263 | 210 | 4,352 | $11.39\%$ | $9.31\%$ | **$5.70\%$** | **PASS** |
| **Track_B_Quantum** | $0.2000$ | 194 | 3,869 | 43 | 746 | $81.86\%$ | $4.77\%$ | **$83.84\%$** | **VIOLATION (FAIL)** |
| Track_B_Quantum | $0.4000$ | 151 | 2,510 | 86 | 2,105 | $63.71\%$ | $5.67\%$ | **$54.39\%$** | **VIOLATION (FAIL)** |
| Track_B_Quantum | $0.5000$ | 90 | 1,281 | 147 | 3,334 | $37.97\%$ | $6.56\%$ | **$27.76\%$** | **VIOLATION (FAIL)** |
| Track_B_Quantum | $0.6000$ | 45 | 563 | 192 | 4,052 | $18.99\%$ | $7.40\%$ | **$12.20\%$** | **PASS** |
| Track_B_Quantum | $0.7000$ | 23 | 220 | 214 | 4,395 | $9.70\%$ | $9.47\%$ | **$4.77\%$** | **PASS** |
| **Track_C_Shadow** | $0.2000$ | 184 | 3,645 | 53 | 970 | $77.64\%$ | $4.81\%$ | **$78.98\%$** | **VIOLATION (FAIL)** |
| Track_C_Shadow | $0.4000$ | 138 | 2,349 | 99 | 2,266 | $58.23\%$ | $5.55\%$ | **$50.90\%$** | **VIOLATION (FAIL)** |
| Track_C_Shadow | $0.5000$ | 93 | 1,216 | 144 | 3,399 | $39.24\%$ | $7.10\%$ | **$26.35\%$** | **VIOLATION (FAIL)** |
| Track_C_Shadow | $0.6000$ | 57 | 574 | 180 | 4,041 | $24.05\%$ | $9.03\%$ | **$12.44\%$** | **PASS** |
| Track_C_Shadow | $0.7000$ | 29 | 275 | 208 | 4,340 | $12.24\%$ | $9.54\%$ | **$5.96\%$** | **PASS** |

### Mathematical Deduction for $\tau = 0.1500$:
At $\tau = 0.2000$, False Emergency Rate already exceeds $82\%$. Extrapolating to $\tau = 0.1500$:
$$\text{FER}(\tau = 0.1500) \approx 90.5\% - 93.0\%$$
Setting $\tau = 0.1500$ would trigger false alarms on over 9 out of every 10 general speech utterances, yielding an emergency precision of under $5\%$.

---

## 5. Audit of Candidate Models and Checkpoint Provenance

Phase 8.2 registered six candidate models citing checkpoint paths in `checkpoints/phase8_1/`:
1. `checkpoints/phase8_1/track_a_fold_models.pt`
2. `checkpoints/phase8_1/int1_calib_models.pt`
3. `checkpoints/phase8_1/int2_cost_models.pt`
4. `checkpoints/phase8_1/int3_samp_models.pt`
5. `checkpoints/phase8_1/track_b_quantum_models.pt`
6. `checkpoints/phase8_1/track_c_shadow_models.pt`

### Forensic Inspection:
- Filesystem command `Test-Path "d:\DL\checkpoints"` returned `False`.
- Recursive search across the entire workspace located no `.pt` files matching `checkpoints/phase8_1/`.
- In `scripts/phase8_1/execute_phase8_1_safety_recovery.py`, models were trained in memory during cross-validation, and only the resulting predictions (`oof_predictions.csv`), fold manifests (`fold_manifest.json`), and metrics (`threshold_registry.json`) were serialized.
- **Finding:** The candidate checkpoints referenced in Phase 8.2 were **unmaterialized placeholders**.
- Under rigorous confirmatory governance, a candidate cannot be evaluated prospective without a verifiable serialized artifact hash.

---

## 6. Audit of Confirmation Dataset Availability

Phase 8.2 registered the following data census:
- TORGO 80% Development Pool: 4,852 utterances (100% contaminated by Phase 8.1 training and tuning).
- TORGO 20% Locked Test Partition: 1,215 utterances (quarantined as diagnostic-only; contaminated by repeated historical inspection).
- External Speech Corpora (UASpeech, Nemours): 0 samples present in repository.
- Prospective Confirmation Cohort: 0 samples collected.

**Finding:** Re-verification confirms that **no independent confirmation dataset is available**.
Under Section 9 of the specification, the confirmation status must strictly remain:
$$\mathbf{BLOCKED\; \text{—}\; INDEPENDENT\; CONFIRMATION\; DATA\; UNAVAILABLE}$$

---

## 7. Forensic Recommendations for Protocol v2

1. **Explicit Schema Separation:** Separate `decision_policy` (model-specific cutoff, score scale, calibration method) from `safety_constraints` (FER ceiling, definition, comparator).
2. **Document Defect in Protocol v2:** Note in `config/phase8_2_1/confirmation_protocol_v2.yaml` that the historical $\tau = 0.1500$ was a conflated FER limit.
3. **Traceable Development Thresholds:** Record the valid development-only operating thresholds derived from Phase 8.1:
   - Pooled operating point: $\tau = 0.6000$ (minimum global cutoff satisfying pooled $\text{FER} \le 0.1500$).
   - Speaker-adapted mean: $\bar{\tau} = 0.762$ (from `threshold_registry.json`).
4. **Preserve Blocked Status:** Since candidate checkpoints are unmaterialized and independent confirmation data are unavailable, the protocol status must be frozen as `BLOCKED`.

---
**END OF FORENSIC AUDIT.**
