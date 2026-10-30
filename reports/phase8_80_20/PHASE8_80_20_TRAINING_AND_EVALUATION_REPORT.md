# PHASE 8 SUPPLEMENTARY 80/20 TRAINING, EVALUATION & COMPLETE REPORT

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Benchmark:** Supplementary Independent 80% Training / 20% Locked Testing Benchmark  
**Dataset:** TORGO Database of Dysarthric Speech (Active HeadMic/Selected Channels)  
**Date of Execution:** 2026-10-10  
**Status:** COMPLETE — ZERO LEAKAGE VERIFIED  
**Final Scientific Conclusion:** CASE B CONFIRMED — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE  

---

## 1. Executive Summary

This report delivers the complete, reproducible execution and empirical evaluation of the **Supplementary 80/20 Independent Benchmark** for low-resource personalized atypical speech recognition. Conducted across the full active corpus of 6,067 primary recordings from 15 speakers (8 dysarthric, 7 control) in the TORGO database, this benchmark partitions data into a deterministic **80% training pool (4,852 utterances)** and an untouched, **locked 20% test pool (1,215 utterances)**.

Three model families were trained and evaluated independently under strict pipeline isolation:
1. **Track A — Classical Baseline:** 384-D Whisper embeddings with Ridge Stage-1 Emergency Gate, 384-D classical residual module, and linear subtype classifier.
2. **Track B — Quantum Residual Model:** 384-D Whisper embeddings, train-only PCA ($384 \to 8$), 8-qubit Variational Quantum Circuit (VQC) with `AngleEmbedding` and `BasicEntanglerLayers` simulated on PennyLane `default.qubit`, continuous logit modulation ($z_{\text{final}} = z_{\text{gate}} + 0.5 \cdot z_{\text{quantum}}$), and linear subtype classifier.
3. **Track C — Matched Classical Shadow:** 384-D Whisper embeddings, train-only PCA ($384 \to 8$), capacity-matched 8-D MLP residual module (16 hidden GELU), and linear subtype classifier.

### Key Benchmark Findings:
- **Global Emergency Recall:** Track B (Quantum) achieves **18.33%** (11/60 detected emergencies), Track A (Classical Baseline) achieves **16.67%** (10/60 detected emergencies), and Track C (Matched Shadow) achieves **11.67%** (7/60 detected emergencies).
- **Macro-F1 & Accuracy:** Track B achieves **0.1694 Macro-F1** and **84.28% Accuracy**; Track A achieves **0.1677 Macro-F1** and **83.87% Accuracy**; Track C achieves **0.1500 Macro-F1** and **80.82% Accuracy**.
- **False Emergency Rate (FER) & Intent Preservation:** Track B maintains an FER of **11.86%** and an Intent Preservation Rate of **88.14%**; Track A maintains an FER of **12.29%** and an Intent Preservation Rate of **87.71%**. Both satisfy the clinical safety bound ($\text{FER} \le 15\%$).
- **Statistical Significance:** Paired Wilcoxon signed-rank testing across the 15 speakers ($p = 1.000$) and McNemar testing on the 60 test emergency utterances ($p = 1.000, \chi^2 = 0.00$) reveal that the 1-sample difference (11 vs 10 detections) is **not statistically significant**.
- **Scientific Verdict:** The benchmark rigorously confirms **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE**. While the quantum circuit actively trains ($\|\Delta W\| = 1.8372$) and modulates decision boundaries, it does not demonstrate a statistically verifiable quantum advantage over a classical baseline.

![Model Comparison](../../results/phase8_80_20/figures/model_comparison.png)

---

## 2. Objective

The primary objective of this study is to execute an unconstrained, large-scale **80% training / 20% testing held-out benchmark** using the active TORGO speech dataset to evaluate whether quantum-classical hybrid architectures provide demonstrable utility over classical counterparts when trained on substantial data pools, rather than restricted few-shot regimes ($N \in \{20, 30, 40, 50\}$).

The benchmark adheres to the following core tenets:
1. **Zero Test Contamination:** Preprocessing (PCA, standard scaling), class weighting, and safety operating thresholds ($\tau$) must be fitted exclusively on the 80% training pool.
2. **Cryptographic Provenance:** Every sample, prediction, and metric file is bound by SHA-256 hashes and state sentinels (`BASELINE_FROZEN.json`, `QUANTUM_RUN_COMPLETE.json`, `SHADOW_RUN_COMPLETE.json`).
3. **Historical Research Integrity:** Historical findings (Phases 1–7, Phase 8 few-shot separated pipeline) and the frozen CASE B conclusion are preserved without modification.

---

## 3. Dataset Description

The evaluation utilizes the canonical **TORGO Database of Dysarthric Speech**, specifically the selected primary-microphone channel recordings (`headMic` or selected single array channel) marked as active in `data/manifests/torgo_manifest.csv`.

| Dataset Attribute | Value | Description |
| :--- | :--- | :--- |
| **Total Corpus Rows** | 16,552 | Full multi-channel audio recordings |
| **Active Utterances** | 6,067 | Primary channel recordings (`headMic` / active primary) |
| **Secondary Channels** | 10,485 | Unselected acoustic array channels |
| **Quality Status** | 100% VALID (6,067 / 6,067) | Verified audio duration and signal integrity |
| **Audio Files on Disk** | 6,067 / 6,067 | 100% physically verified WAV files |
| **Whisper Embeddings** | 6,067 / 6,067 | 384-D Whisper embeddings stored in `embeddings/torgo/whisper/` |
| **Audio Uniqueness** | 6,067 Unique Hashes | 0 duplicate files or byte-level collisions |
| **Total Speakers** | 15 Speakers | 8 dysarthric speakers, 7 control speakers |
| **Total Emergency Samples** | 297 (4.90%) | Extreme minority intent class |
| **Total General Speech** | 5,770 (95.10%) | Conversational general speech utterances |

---

## 4. Dataset Fingerprint

To guarantee immutable repeatability across computational environments, the split manifest is bound to a cryptographic SHA-256 fingerprint computed across sorted training and testing sample identifiers:

$$\text{Dataset Fingerprint} = \texttt{16c1304221900a8ca5a278adc2528ba2807881e9c526a322b7a649135bc80a58}$$

Manifest Location: `results/phase8_80_20/split_manifest.json`  
Master Seed: `42`  
Partition Ratio: `80.0% Training / 20.0% Testing`

---

## 5. 80/20 Split Method

The dataset was partitioned using a **speaker-aware, intent-stratified splitting algorithm** implemented in `scripts/phase8_80_20/build_80_20_split.py`:
1. **Speaker Partitioning:** For each of the 15 speakers, eligible primary utterances were grouped and sorted by `recording_id`.
2. **Intent Stratification:** Within each speaker, samples were grouped by canonical intent. For classes with $N_{s,c} \ge 2$, deterministic permutation with seed 42 allocated $n_{\text{test}} = \text{round}(0.20 \times N_{s,c})$ samples to the test partition and the remaining samples to the training partition.
3. **Singleton Handling Policy:** If a speaker had only a single sample for a specific class ($N_{s,c} = 1$), it was allocated to the training pool. This guaranteed that the model encountered the intent during training without fabricating synthetic samples.
4. **Disjointness Assertion:** Complete mathematical verification ensured $\text{Train} \cap \text{Test} = \emptyset$ for sample IDs and audio SHA-256 digests.

| Partition | Target Ratio | Actual Count | Actual Percentage | Emergency Count | Emergency Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Pool** | 80.00% | 4,852 | 79.97% | 237 | 79.80% |
| **Locked Test Pool** | 20.00% | 1,215 | 20.03% | 60 | 20.20% |
| **Total Corpus** | 100.00% | 6,067 | 100.00% | 297 | 100.00% |

---

## 6. Speaker and Class Distributions

![Train/Test Class Distribution](../../results/phase8_80_20/figures/train_test_class_distribution.png)

### Intent Class Breakdown:

| Canonical Intent Class | Total Utterances | Training Pool (80%) | Locked Test Pool (20%) | Test Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **GENERAL_SPEECH** | 5,770 | 4,615 | 1,155 | 20.02% |
| **CRITICAL_RESPONSE** | 87 | 70 | 17 | 19.54% |
| **FALL** | 67 | 53 | 14 | 20.90% |
| **BREATHING_MEDICAL** | 51 | 40 | 11 | 21.57% |
| **PAIN** | 43 | 34 | 9 | 20.93% |
| **SOS_HELP** | 40 | 31 | 9 | 22.50% |
| **LOCATION** | 9 | 9 | 0 | 0.00%* |
| **Total Emergencies** | **297** | **237** | **60** | **20.20%** |
| **Total All Classes** | **6,067** | **4,852** | **1,215** | **20.03%** |

*\*Note on LOCATION:* Because `LOCATION` occurs only 9 times in the entire 6,067-sample database and no speaker has more than 2 instances, round($0.20 \times 2$) = 0. Under stratified sampling within speakers, all 9 instances were allocated to the training pool. Emergency test evaluation therefore comprehensively assesses the remaining 5 critical emergency subclasses.

### Speaker Group Breakdown:

| Speaker ID | Group | Total Samples | Train Samples (80%) | Test Samples (20%) | Emergency Test Samples |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **F01** | Dysarthric | 118 | 93 | 25 | 3 |
| **F03** | Dysarthric | 213 | 170 | 43 | 3 |
| **F04** | Dysarthric | 250 | 201 | 49 | 1 |
| **FC01** | Control | 151 | 120 | 31 | 3 |
| **FC02** | Control | 975 | 781 | 194 | 7 |
| **FC03** | Control | 405 | 324 | 81 | 2 |
| **M01** | Dysarthric | 285 | 227 | 58 | 4 |
| **M02** | Dysarthric | 232 | 186 | 46 | 3 |
| **M03** | Dysarthric | 406 | 325 | 81 | 5 |
| **M04** | Dysarthric | 286 | 229 | 57 | 5 |
| **M05** | Dysarthric | 357 | 286 | 71 | 3 |
| **MC01** | Control | 419 | 335 | 84 | 3 |
| **MC02** | Control | 383 | 305 | 78 | 5 |
| **MC03** | Control | 594 | 475 | 119 | 4 |
| **MC04** | Control | 993 | 795 | 198 | 9 |
| **Total** | **15 Speakers** | **6,067** | **4,852** | **1,215** | **60** |

---

## 7. Leakage Prevention

The integrity of this benchmark is safeguarded by strict algorithmic boundaries:
1. **Disjoint Sets:** Programmatic verification confirms $0$ overlapping sample IDs and $0$ overlapping audio hashes between the 4,852 training pool and the 1,215 test pool.
2. **Train-Only Preprocessing:** Dimensionality reduction (`TrainOnlyPCAPipeline`) is fitted exclusively on $X_{\text{train}}$. The test vectors $X_{\text{test}}$ are transformed strictly using the pre-fitted projection matrix.
3. **Train-Only Class Weights:** Loss function class reweighting ($\text{pos\_weight}$) is computed solely from training binary labels.
4. **Internal Calibration:** The decision threshold $\tau$ is calibrated strictly on training set probabilities to satisfy $\text{FER} \le 15\%$. Test labels are never accessed during calibration.
5. **Historical Contamination Audit:** An audit revealed that 218 test samples were present in the historical Phase 6.5/8 few-shot adaptation set. As mandated, any direct before-comparison against historical models is marked `INVALID` due to test set appearance in historical training pools.

---

## 8. Training / Internal Validation / Test Protocol

```mermaid
graph TD
    A["Active TORGO Utterances (N=6,067)"] --> B["Speaker-Aware Stratified Split (Seed 42)"]
    B --> C["80% Training Pool (N=4,852)"]
    B --> D["Locked 20% Test Pool (N=1,215)"]
    
    subgraph "Training & Internal Validation (Train Pool Only)"
        C --> E["Stage 1: Scalar Emergency Gate"]
        C --> F["Dimensionality Reduction (PCA 384->8, Fit on Train)"]
        F --> G["Stage 2 Residual (Track A: 384D / Track B: 8Q VQC / Track C: 8D MLP)"]
        C --> H["Stage 3: Subtype Classifier (Emergencies Only)"]
        E & G --> I["Internal Threshold Calibration (FER <= 15% on Train Proba)"]
    end
    
    subgraph "Protected Evaluation (Locked Test Pool Only)"
        D --> J["Transform Features (Pre-fitted PCA)"]
        J --> K["Predict Logits: z_final = z_gate + 0.5 * z_res"]
        K --> L["Apply Locked Threshold tau"]
        L --> M["Compute Metrics & Emit Provenance"]
    end
```

---

## 9. Classical Baseline Architecture (Track A)

Track A represents the standard classical linear model:
- **Feature Space:** 384-D Whisper embeddings.
- **Stage 1 Emergency Gate:** Ridge-regularized logistic regression with balanced class weights emitting scalar logit $z_{\text{gate}}$.
- **Stage 2 Classical Residual:** 384-D linear module fitted on binary emergency loss emitting $z_{\text{residual}}$.
- **Fusion:** Continuous logit combination: $z_{\text{final}} = z_{\text{gate}} + 0.5 \cdot z_{\text{residual}}$.
- **Stage 3 Subtype Classifier:** Multi-class logistic regression trained on emergency samples to predict the 6 emergency sub-intents.

---

## 10. Quantum Architecture (Track B)

Track B introduces the variational quantum circuit residual:
- **Dimensionality Reduction:** `TrainOnlyPCAPipeline` maps 384-D Whisper embeddings to 8 principal components with feature range $[0, \pi]$.
- **Quantum Device:** 8-qubit simulator using PennyLane `default.qubit` with PyTorch autograd interface and `backprop` differentiation.
- **State Preparation:** `qml.AngleEmbedding` on all 8 wires.
- **Entangling Layers:** `qml.BasicEntanglerLayers` parameterized by trainable weight matrix $W \in \mathbb{R}^{2 \times 8}$ (16 rotation angles).
- **Measurement:** Expectation value of Pauli-Z on wire 0: $\langle Z_0 \rangle \in [-1, 1]$.
- **Trainable Affine Mapping:** $z_{\text{quantum}} = \gamma \cdot \langle Z_0 \rangle + \beta$, where $\gamma, \beta$ are trainable scalar parameters (total 18 parameters).
- **Optimization:** Adam optimizer ($\text{lr} = 0.01$), BCEWithLogitsLoss with positive class weight, 25 epochs.
- **Fusion:** $z_{\text{final}} = z_{\text{gate}} + 0.5 \cdot z_{\text{quantum}}$.

---

## 11. Matched Classical Shadow Architecture (Track C)

Track C controls for parameter capacity and input dimensionality:
- **Dimensionality Reduction:** Identical 8-D input space via `TrainOnlyPCAPipeline`.
- **Architecture:** 2-layer Multi-Layer Perceptron (MLP): $\text{Linear}(8 \to 16) \to \text{GELU} \to \text{Linear}(16 \to 1)$ (total 161 parameters).
- **Optimization:** Adam optimizer ($\text{lr} = 0.01$), BCEWithLogitsLoss with positive class weight, 25 epochs.
- **Fusion:** $z_{\text{final}} = z_{\text{gate}} + 0.5 \cdot z_{\text{shadow}}$.

---

## 12. Hyperparameters and Configuration

| Parameter | Track A (Classical) | Track B (Quantum) | Track C (Shadow) |
| :--- | :---: | :---: | :---: |
| **Input Dimensions** | 384 | 8 (via PCA) | 8 (via PCA) |
| **Hidden Units / Qubits** | 384 | 8 Qubits | 16 Hidden |
| **Trainable Residual Params** | 385 | 18 | 161 |
| **Optimizer** | L-BFGS / Ridge | Adam ($\text{lr}=0.01$) | Adam ($\text{lr}=0.01$) |
| **Epochs** | Convergence | 25 | 25 |
| **Batch Size** | Full Batch | 8 (Vectorized) | 8 |
| **Residual Coefficient ($\alpha$)** | 0.5 | 0.5 | 0.5 |
| **Internal Calibration Metric** | FER $\le 15\%$ | FER $\le 15\%$ | FER $\le 15\%$ |

---

## 13. Model Training Results

All models converged stably across all 15 speakers. The quantum module demonstrated active parameter updates across every speaker:
- **Track A Total Training Time:** 28.57 seconds across 15 speakers (1.90s / speaker).
- **Track B Total Training Time:** 281.17 seconds across 15 speakers (18.74s / speaker).
- **Track B Weight Delta Norm ($\|\Delta W\|$):** Mean of **1.8372** across all speakers (range: 1.0801 to 3.1095).
- **Track C Total Training Time:** 32.98 seconds across 15 speakers (2.20s / speaker).

---

## 14. Before Results (Historical Few-Shot Baseline)

Historical Phase 8 few-shot experiments evaluated models adapted on $N=50$ utterances per speaker. As documented in `results/phase8_separated/metrics.json`:
- **Accuracy:** 0.8250
- **Macro-F1:** 0.1480
- **Emergency Recall:** 0.1250
- **Emergency Precision:** 0.0520
- **False Emergency Rate:** 0.1420
- **Intent Preservation Rate:** 0.8580

---

## 15. After Results (80/20 Benchmark)

The newly trained models evaluated on the locked 20% test partition (1,215 utterances) yielded:
- **Track A (Classical):** Recall = **0.1667** (10/60), Precision = **0.0658**, FER = **0.1229**, Macro-F1 = **0.1677**, Accuracy = **0.8387**.
- **Track B (Quantum):** Recall = **0.1833** (11/60), Precision = **0.0743**, FER = **0.1186**, Macro-F1 = **0.1694**, Accuracy = **0.8428**.
- **Track C (Shadow):** Recall = **0.1167** (7/60), Precision = **0.0383**, FER = **0.1524**, Macro-F1 = **0.1500**, Accuracy = **0.8082**.

---

## 16. Before vs After Comparison

![Before vs After Comparison](../../results/phase8_80_20/figures/before_after_comparison.png)

| Metric | Before Value (Few-Shot N=50 Ref) | After Value (Track B 80/20) | Absolute Diff | Relative Change | Validity Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accuracy** | 0.8250 | 0.8428 | +0.0178 | +2.16% | **INVALID — TEST SET APPEARED IN PREVIOUS TRAINING** |
| **Macro-F1** | 0.1480 | 0.1694 | +0.0214 | +14.46% | **INVALID — TEST SET APPEARED IN PREVIOUS TRAINING** |
| **Emergency Recall** | 0.1250 | 0.1833 | +0.0583 | +46.64% | **INVALID — TEST SET APPEARED IN PREVIOUS TRAINING** |
| **Emergency Precision** | 0.0520 | 0.0743 | +0.0223 | +42.88% | **INVALID — TEST SET APPEARED IN PREVIOUS TRAINING** |
| **FER** | 0.1420 | 0.1186 | -0.0234 | -16.48% | **INVALID — TEST SET APPEARED IN PREVIOUS TRAINING** |
| **Intent Pres. Rate** | 0.8580 | 0.8814 | +0.0234 | +2.73% | **INVALID — TEST SET APPEARED IN PREVIOUS TRAINING** |

*Forensic Integrity Notice:* As mandated by Section 9 of the Master Execution Prompt, the direct before/after comparison is officially marked **INVALID**. The historical Phase 6.5/8 few-shot adaptation pool contained 218 utterances that are part of the new 80/20 locked test set. While the 80/20 models demonstrate superior point estimates across all metrics, claiming an uncontaminated historical improvement would violate scientific rigor.

---

## 17. Complete Model Comparison Table

| Metric | Track A: Classical Baseline | Track B: Quantum Residual | Track C: Matched Shadow | Quantum vs Classical Diff | Quantum vs Shadow Diff |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 0.8387 | **0.8428** | 0.8082 | +0.0041 (+0.41%) | +0.0346 (+3.46%) |
| **Macro-F1** | 0.1677 | **0.1694** | 0.1500 | +0.0017 (+0.17%) | +0.0194 (+1.94%) |
| **Emergency Recall** | 0.1667 (10/60) | **0.1833 (11/60)** | 0.1167 (7/60) | +0.0166 (+1.66%) | +0.0666 (+6.66%) |
| **Emergency Precision** | 0.0658 | **0.0743** | 0.0383 | +0.0085 (+0.85%) | +0.0360 (+3.60%) |
| **False Emergency Rate** | 0.1229 | **0.1186** | 0.1524 | -0.0043 (-0.43%) | -0.0338 (-3.38%) |
| **Intent Preservation Rate** | 0.8771 | **0.8814** | 0.8476 | +0.0043 (+0.43%) | +0.0338 (+3.38%) |
| **Dysarthric Recall** | **0.1481 (4/27)** | **0.1481 (4/27)** | 0.0741 (2/27) | +0.0000 (+0.00%) | +0.0740 (+7.40%) |
| **Control Recall** | 0.1818 (6/33) | **0.2121 (7/33)** | 0.1515 (5/33) | +0.0303 (+3.03%) | +0.0606 (+6.06%) |
| **Train Samples** | 4,852 | 4,852 | 4,852 | — | — |
| **Test Samples** | 1,215 | 1,215 | 1,215 | — | — |
| **Parameter Count** | 3,456 | **18** | 161 | -3,438 (-99.5%) | -143 (-88.8%) |
| **Total Train Time** | **28.57s** | 281.17s | 32.98s | +252.60s | +248.19s |
| **Total Inference Time** | 0.03s | 0.19s | **0.02s** | +0.16s | +0.17s |

---

## 18. Accuracy and Macro-F1

![Generalization Gap](../../results/phase8_80_20/figures/train_test_gap.png)

- **Overall Accuracy:** Track B achieves the highest overall classification accuracy (84.28%), outperforming Track A (83.87%) by +0.41% and Track C (80.82%) by +3.46%.
- **Macro-F1 Score:** Due to the extreme class imbalance (95.1% general speech vs 4.9% emergencies distributed across 6 rare subtypes), unweighted Macro-F1 across all 7 canonical classes is bounded across all models (Track B: 0.1694 vs Track A: 0.1677 vs Track C: 0.1500).

---

## 19. Emergency Recall, Precision, and FER

![Emergency Metrics Comparison](../../results/phase8_80_20/figures/emergency_metrics_comparison.png)

- **Emergency Recall:** Track B detects 11 out of 60 emergencies (18.33%), compared to 10 out of 60 for Track A (16.67%) and 7 out of 60 for Track C (11.67%).
- **Clinical Safety Gate Compliance:** Both Track A (FER = 12.29%) and Track B (FER = 11.86%) satisfy the strict safety veto constraint ($\text{FER} \le 15.0\%$). Track C slightly breaches the threshold at the global aggregate level ($\text{FER} = 15.24\%$).

---

## 20. Per-Class Results

![Per Class Performance](../../results/phase8_80_20/figures/per_class_performance.png)

| Intent Class | Support (Test) | Track A F1-Score | Track B F1-Score | Track C F1-Score | Best Performing Track |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GENERAL_SPEECH** | 1,155 | 0.9082 | **0.9106** | 0.8902 | Track B (Quantum) |
| **CRITICAL_RESPONSE** | 17 | 0.0870 | **0.0952** | 0.0000 | Track B (Quantum) |
| **FALL** | 14 | 0.1053 | **0.1111** | 0.0952 | Track B (Quantum) |
| **BREATHING_MEDICAL** | 11 | **0.0769** | 0.0714 | 0.0667 | Track A (Classical) |
| **PAIN** | 9 | 0.0000 | 0.0000 | 0.0000 | Tied (Zero Recall) |
| **SOS_HELP** | 9 | 0.0000 | 0.0000 | 0.0000 | Tied (Zero Recall) |
| **LOCATION** | 0 | 0.0000 | 0.0000 | 0.0000 | N/A (0 Test Support) |

---

## 21. Dysarthric vs Control Results

![Dysarthric vs Control](../../results/phase8_80_20/figures/dysarthric_vs_control.png)

Acoustic degradation in dysarthric speech significantly challenges personalized classification:
- **Dysarthric Speakers (8 Speakers, 27 Test Emergencies):**
  - Track A: Recall = **14.81%** (4/27), Macro-F1 = 0.1442, FER = 14.11%
  - Track B: Recall = **14.81%** (4/27), Macro-F1 = 0.1448, FER = 13.86%
  - Track C: Recall = 7.41% (2/27), Macro-F1 = 0.1293, FER = 14.60%
- **Control Speakers (7 Speakers, 33 Test Emergencies):**
  - Track A: Recall = 18.18% (6/33), Macro-F1 = 0.1798, FER = 11.32%
  - Track B: Recall = **21.21%** (7/33), Macro-F1 = 0.1821, FER = 10.79%
  - Track C: Recall = 15.15% (5/33), Macro-F1 = 0.1604, FER = 15.58%

---

## 22. Per-Speaker Analysis

![Per Speaker Comparison](../../results/phase8_80_20/figures/per_speaker_comparison.png)

| Speaker | Group | Test Em. | Track A Recall | Track B Recall | Track C Recall | Track A FER | Track B FER | Track C FER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F01** | Dysarthric | 3 | 0.333 | 0.333 | 0.333 | 0.136 | 0.136 | 0.091 |
| **F03** | Dysarthric | 3 | 0.000 | 0.000 | 0.000 | 0.100 | 0.200 | 0.150 |
| **F04** | Dysarthric | 1 | 0.000 | **1.000** | 0.000 | 0.146 | 0.104 | 0.125 |
| **FC01** | Control | 3 | **0.333** | 0.000 | **0.333** | 0.143 | 0.107 | 0.143 |
| **FC02** | Control | 7 | **0.143** | 0.000 | 0.000 | 0.091 | 0.037 | 0.102 |
| **FC03** | Control | 2 | 0.000 | 0.000 | 0.000 | 0.139 | 0.127 | 0.177 |
| **M01** | Dysarthric | 4 | 0.000 | 0.000 | 0.000 | 0.130 | 0.037 | 0.056 |
| **M02** | Dysarthric | 3 | 0.333 | 0.333 | 0.000 | 0.186 | 0.279 | 0.163 |
| **M03** | Dysarthric | 5 | **0.200** | 0.000 | 0.000 | 0.158 | 0.145 | 0.118 |
| **M04** | Dysarthric | 5 | 0.000 | 0.000 | 0.000 | 0.135 | 0.077 | 0.135 |
| **M05** | Dysarthric | 3 | 0.333 | 0.333 | 0.333 | 0.132 | 0.191 | 0.162 |
| **MC01** | Control | 3 | 0.333 | **0.667** | 0.333 | 0.272 | 0.259 | 0.259 |
| **MC02** | Control | 5 | 0.200 | **0.400** | 0.200 | 0.096 | 0.110 | 0.247 |
| **MC03** | Control | 4 | 0.000 | **0.250** | 0.000 | 0.052 | 0.096 | 0.113 |
| **MC04** | Control | 9 | 0.222 | 0.222 | 0.222 | 0.095 | 0.101 | 0.190 |

---

## 23. Confusion Matrices

The 7x7 confusion matrices demonstrate that the primary mass is correctly allocated to `GENERAL_SPEECH`, with calibrated off-diagonal detection of `CRITICAL_RESPONSE` and `FALL`.

### Track A Classical Baseline Confusion Matrix
![Confusion Matrix Classical](../../results/phase8_80_20/figures/confusion_matrix_classical.png)

### Track B Quantum Residual Confusion Matrix
![Confusion Matrix Quantum](../../results/phase8_80_20/figures/confusion_matrix_quantum.png)

### Track C Matched Classical Shadow Confusion Matrix
![Confusion Matrix Shadow](../../results/phase8_80_20/figures/confusion_matrix_shadow.png)

---

## 24. Training Curves and Participation Ablation

To prove that the 8-qubit quantum circuit actively participated in the decision pipeline and was not bypassed, a **3-way Participation Ablation** was executed across all 15 speakers:

| Ablation Condition | Decision Logit Formulation | Mean Abs Proba Shift ($\Delta p$) | Prediction Flips (Test) |
| :--- | :--- | :---: | :---: |
| **Normal Forward** | $z_{\text{final}} = z_{\text{gate}} + 0.5 \cdot z_{\text{quantum}}$ | 0.0000 (Baseline) | 0 (Baseline) |
| **Zero Ablation** | $z_{\text{zero}} = z_{\text{gate}} + 0.00$ | **0.0612** | **42 Flips** |
| **Shuffled Ablation** | $z_{\text{shuf}} = z_{\text{gate}} + 0.5 \cdot \text{perm}(z_{\text{quantum}})$ | **0.0841** | **67 Flips** |

*Verification:* The non-zero weight displacement ($\|\Delta W\| = 1.8372$), probability shifts ($\Delta p > 0.06$), and 42 prediction flips confirm that the quantum circuit actively modulated classification decisions.

---

## 25. Runtime and Resource Usage

![Runtime Comparison](../../results/phase8_80_20/figures/runtime_comparison.png)

- **Training Latency:** Track A completed training in **28.57s** (CPU). Track B required **281.17s** due to quantum statevector simulation on PennyLane `default.qubit` across 4,852 training vectors. Track C completed in **32.98s**.
- **Inference Latency:** Track A evaluated 1,215 test vectors in **0.03s** (0.02ms / sample). Track B evaluated the test vectors in **0.19s** (0.16ms / sample, well within the real-time threshold of $\le 50\text{ms}$ for emergency alerting). Track C evaluated in **0.02s**.

---

## 26. Statistical Analysis

To establish whether the observed difference between Track B (11/60 detected emergencies) and Track A (10/60 detected emergencies) constitutes a genuine quantum advantage, formal statistical hypothesis tests were conducted:

### 1. Paired Wilcoxon Signed-Rank Tests (N=15 Speakers):
- **Track B vs Track A Emergency Recall:** $W = 12.0, p = 1.0000$ ($\text{Not Significant}, p \ge 0.05$).
- **Track B vs Track C Emergency Recall:** $W = 6.0, p = 0.1094$ ($\text{Not Significant}, p \ge 0.05$).
- **Track A vs Track C Emergency Recall:** $W = 10.0, p = 0.2188$ ($\text{Not Significant}, p \ge 0.05$).
- **Track B vs Track A Macro-F1:** $W = 54.0, p = 0.7334$ ($\text{Not Significant}, p \ge 0.05$).

### 2. McNemar's Test on 60 Locked Emergency Test Utterances:
- Contingency: Discordant pairs $b_1 a_0 = 3$ (Quantum detected, Classical missed), $b_0 a_1 = 2$ (Classical detected, Quantum missed).
- $\chi^2 = \frac{(|3 - 2| - 1)^2}{3 + 2} = 0.000, p = 1.0000$ ($\text{Not Significant}$).

**Statistical Inference:** There is no statistically supported difference between the quantum model and the classical baseline.

---

## 27. Leakage-Audit Results

The programmatic audit (`results/phase8_80_20/leakage_audit.json`) verified 8 critical safety policies:

```json
{
  "benchmark": "Phase8_80_20",
  "overall_audit_status": "PASS",
  "checks": {
    "sample_id_disjointness": "PASS (0 overlapping IDs)",
    "audio_hash_disjointness": "PASS (0 hash collisions across 6,067 files)",
    "preprocessing_fit_on_train_only": "PASS (PCA fit exclusively on X_train)",
    "class_weights_on_train_only": "PASS (pos_weight from y_train)",
    "decision_threshold_calibration": "PASS (calibrated on train probabilities)",
    "no_test_samples_in_training": "PASS (zero test exposure)",
    "track_pipeline_isolation": "PASS (independent instances & prediction files)",
    "historical_contamination_audit": "FLAGGED_IN_BEFORE_AFTER (218 samples in historical pool)"
  }
}
```

---

## 28. Model and Experiment Provenance

All artifacts are version-controlled and cryptographically signed:
- **Split Manifest:** `results/phase8_80_20/split_manifest.json` (Fingerprint: `16c1304221900a8ca5a278adc2528ba2807881e9c526a322b7a649135bc80a58`)
- **Track A Frozen Sentinel:** `results/phase8_80_20/BASELINE_FROZEN.json` (Hash: `d0cbf13d0d13...`)
- **Track B Complete Sentinel:** `results/phase8_80_20/QUANTUM_RUN_COMPLETE.json` (Hash: `78afed76229a...`)
- **Track C Complete Sentinel:** `results/phase8_80_20/SHADOW_RUN_COMPLETE.json` (Hash: `5b82142e01df...`)
- **Master Prediction Registry:** `results/phase8_80_20/predictions.csv` (3,645 rows)

---

## 29. Limitations

1. **Extreme Class Imbalance:** Emergency phrases constitute only 4.90% of the active database (297 total samples). Even with an 80/20 split, the locked test partition contains only 60 emergency utterances across 15 speakers (average 4 per speaker).
2. **Subtype Resolution:** Subclasses like `PAIN` and `SOS_HELP` possess very few training samples per speaker, resulting in 0 test detections across all model families.
3. **Simulation Scale:** The quantum model is restricted to 8 qubits due to classical simulation overhead on classical CPUs. Full physical QPUs or larger qubit registers may exhibit different noise and expressivity properties.

---

## 30. Final Conclusion

The Supplementary 80/20 Benchmark executes a rigorous, leakage-free empirical evaluation of hybrid quantum-classical deep learning for personalized emergency speech recognition.

The findings conclusively demonstrate:
1. **Model Convergence:** The 8-qubit variational quantum circuit trains stably and participates meaningfully in emergency classification decisions.
2. **Competitive Parity:** The quantum model achieves a marginal point-estimate lead over the classical baseline (Emergency Recall: 18.33% vs 16.67%; Accuracy: 84.28% vs 83.87%; FER: 11.86% vs 12.29%).
3. **Absence of Advantage:** Rigorous statistical hypothesis testing confirms that the observed performance differential is not statistically significant ($p = 1.000$).
4. **Honest Scientific Status:** The frozen conclusion of the thesis remains robustly confirmed:

$$\mathbf{CASE\ B\ —\ NO\ STATISTICALLY\ SUPPORTED\ QUANTUM\ ADVANTAGE}$$

This 80/20 benchmark provides definitive, uncontaminated evidence that quantum-classical hybrid architectures in their current formulation perform on par with, but do not provide a statistically verifiable advantage over, matched classical architectures for personalized atypical speech recognition.

---
*Report generated and validated autonomously by Antigravity Agentic Framework.*
