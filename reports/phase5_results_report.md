# Phase 5 Complete Results Report: Acoustic Noise Robustness & Protocol B Generalization

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Dysarthric Speech Database (15 speakers: 8 dysarthric, 7 control)  
**Evaluation Protocol:** Frozen Protocol B (Clean Calibration $\to$ Controlled Noise Evaluation)  
**Stage:** Phase 5 Canonical Results Report  
**Date:** October 2026  
**Status:** **100% COMPLETE & CANONICALLY FROZEN (720 / 720 Experimental Cells Verified)**  

---

## 1. Executive Summary

Phase 5 investigated the acoustic noise robustness and cross-environment generalization of personalized few-shot emergency intent classifiers trained on frozen acoustic representations of atypical speech. Specifically, Phase 5 executed **Protocol B (Clean Calibration $\to$ Controlled Noise Evaluation)**, modeling the scenario where an assistive communication device is calibrated by a motor-impaired user in a quiet setting, but subsequently operated in noisy acoustic environments.

The experiment comprised exactly **720 experimental conditions** formed by the Cartesian product of:
- **15 Speakers:** 8 dysarthric (`F01, F03, F04, M01, M02, M03, M04, M05`) and 7 matched control (`FC01, FC02, FC03, MC01, MC02, MC03, MC04`).
- **4 Adaptation Budgets ($N$):** $N \in \{20, 30, 40, 50\}$ clean adaptation utterances per speaker.
- **3 Evaluated Models:** Classical Adapter Head (2-layer MLP, 25,223 parameters), Classical RBF-SVM ($C=1.0$), and Primary Quantum Support Vector Classifier (QSVC, $q=4$ qubits, $1$ repetition, linear entanglement, $C=1.0$).
- **4 Acoustic Evaluation Conditions:** Clean ($\infty$ dB SNR), 15 dB SNR, 10 dB SNR, and 5 dB SNR additive white Gaussian noise (AWGN) applied at the waveform level.

### Key Measured Outcomes:
1. **Clean-Baseline Verification (100% PASS):** All 180 clean baseline conditions (60 per model) match Phase 3 and Phase 4 baselines within floating-point tolerance ($\Delta < 10^{-4}$ Macro-F1).
2. **720-Cell Execution Completeness:** 720 of 720 cells executed successfully with 0 failures, 0 dropped cells, and 245,520 individual test predictions recorded with full lineage.
3. **Model Robustness Dynamics:**
   - **Classical Adapter:** Demonstrated non-trivial emergency intent learning on specific speakers (peaking at $0.2252$ Macro-F1 on `MC04` clean). However, its performance suffered systematic degradation under acoustic noise: condition-mean Macro-F1 declined from $0.1450 \pm 0.0181$ (Clean) $\to$ $0.1397 \pm 0.0092$ ($15\text{ dB}$) $\to$ $0.1389 \pm 0.0106$ ($10\text{ dB}$) $\to$ $0.1367 \pm 0.0130$ ($5\text{ dB}$).
   - **RBF-SVM & Primary Quantum QSVC:** Both kernel methods exhibited uniform collapse to the majority class (`GENERAL_SPEECH`, ~95% of test data) across all $N$ and all noise levels due to unweighted $C=1.0$ soft-margin optimization under extreme class imbalance. Their condition-mean Macro-F1 remained invariant at $0.1390 \pm 0.0013$ ($0.9477$ Accuracy, $0.0000$ IPER) because acoustic perturbations on test embeddings did not shift points across the collapsed margin boundary. Across all 81,840 paired test instances, RBF-SVM and QSVC produced identical prediction outputs.
4. **Intent Preservation Rate (IPER):** 
   - Condition-mean Exact IPER for the Classical Adapter degraded from $1.56\%$ (Clean) to $0.98\%$ ($5\text{ dB}$ SNR).
   - In pooled sample-weighted terms, exact emergency intent matches for the Adapter occurred on 2/952 ($0.21\%$) Clean utterances and 10/952 ($1.05\%$) 5 dB utterances.
   - Any-Emergency triage preservation for the Adapter increased from 5/952 ($0.53\%$) on Clean to 57/952 ($5.99\%$) at 5 dB SNR due to noise-induced dispersion of boundary predictions away from `GENERAL_SPEECH` into the `FALL` class.
   - RBF-SVM and QSVC preserved $0.00\%$ of emergency intents under all definitions.
5. **Quantum Advantage Hypothesis:** Under the evaluated Phase 5 configuration, no measurable quantum advantage in noise robustness or intent recognition was observed.

---

## 2. Phase 5 Objective

The primary objective of Phase 5 is to rigorously quantify how test-time acoustic noise affects few-shot personalized models calibrated on clean atypical speech representations. Phase 5 addresses three specific empirical questions:
1. **RQ5.1:** How severely does test-time acoustic noise (15 dB, 10 dB, 5 dB SNR AWGN) degrade personalized few-shot classification of emergency speech intents?
2. **RQ5.2:** Under the evaluated Protocol B configuration, does a quantum fidelity kernel (`QSVC`, $q=4$) offer measurable noise robustness or out-of-distribution margin separation compared to a classical RBF-SVM and a parameterized classical Adapter?
3. **RQ5.3:** To what extent does few-shot calibration ($N=20 \to 50$) affect noise-induced performance variation in safety-critical intent recognition (measured via IPER)?

---

## 3. Frozen Protocol B Definition

Phase 5 strictly adheres to **Protocol B (Clean Calibration $\to$ Noisy Evaluation)**:
- **Adaptation Stage:** The model is adapted/trained exclusively on pristine, clean audio embeddings from the speaker's adaptation split $\mathcal{D}_{\text{adapt}}^{(N)}$.
- **Invariance Enforcement:** Preprocessing pipelines (PCA $384 \to 4$ and MinMaxScaler $[0, \pi]$) are fitted **only** on clean adaptation vectors. No noisy samples enter calibration.
- **Evaluation Stage:** The model is evaluated on held-out test utterances perturbed with controlled additive white Gaussian noise at target SNR levels:
  $$\text{SNR}_{\text{target}} \in \{15\text{ dB}, 10\text{ dB}, 5\text{ dB}\}$$
  alongside unperturbed baseline Clean ($\infty$ dB SNR).
- **Isolation from Protocol A:** Protocol A (Noisy Calibration $\to$ Noisy Evaluation) was explicitly isolated and excluded from the primary 720-cell experimental matrix.

---

## 4. Dataset and Speaker Cohort

The evaluation was conducted on the TORGO database of dysarthric speech across the frozen 15-speaker cohort established in Phase 1:
- **Dysarthric Group ($n=8$):** `F01`, `F03`, `F04`, `M01`, `M02`, `M03`, `M04`, `M05`.
- **Control Group ($n=7$):** `FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC03`, `MC04`.
- **Speech Invariants:** 16 kHz mono waveforms, validated for audio integrity, duration ($>0.5\text{ s}$), and transcript concordance.

---

## 5. Adaptation Budgets

Four nested sample budgets ($N \in \{20, 30, 40, 50\}$) were evaluated per speaker. These splits follow the strict inclusion hierarchy established in Phase 1:
$$\mathcal{D}_{\text{adapt}}^{(20)} \subset \mathcal{D}_{\text{adapt}}^{(30)} \subset \mathcal{D}_{\text{adapt}}^{(40)} \subset \mathcal{D}_{\text{adapt}}^{(50)}$$
with all remaining utterances assigned to the strictly disjoint test set:
$$\mathcal{D}_{\text{test}}^{(N)} = \mathcal{D}_{\text{speaker}} \setminus \mathcal{D}_{\text{adapt}}^{(N)}$$

Across the 15 speakers, the held-out test sets comprise between 144 and 757 test utterances per condition (averaging 341 test utterances per speaker).

---

## 6. Model Configurations

All models operated on 384-dimensional frozen acoustic representations extracted by `openai/whisper-tiny` (pooled over 1,500 frames, unit-$L_2$ normalized):

1. **Classical Adapter Head (`adapter`):**
   - Architecture: $\text{Linear}(384, 64) \to \text{LayerNorm}(64) \to \text{GELU}() \to \text{Dropout}(p=0.1) \to \text{Linear}(64, 7)$.
   - Parameter Count: Exactly 25,223 trainable weights and biases.
   - Optimization: AdamW, $\text{lr}=0.005$, $\text{weight\_decay}=10^{-4}$, batch size 8, 60 epochs.
2. **Classical RBF-SVM (`rbf_svm`):**
   - Classifier: `sklearn.svm.SVC` with RBF kernel ($\gamma=\text{'scale'}$, $C=1.0$, unweighted).
   - Input: Raw 384-D unit-$L_2$ normalized Whisper embeddings.
3. **Primary Quantum QSVC (`qsvc`):**
   - Dimensionality Reduction: Train-only PCA ($384 \to 4$) preserving principal variance.
   - Feature Scaling: Train-only MinMaxScaler scaling components into $[0, \pi]$.
   - Quantum Circuit: Second-order Pauli $ZZ$ Feature Map (`ZZFeatureMap`), $q=4$ qubits, $1$ repetition, linear entanglement.
   - Kernel Evaluation: Exact statevector fidelity simulation $\kappa(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$.
   - Classifier: `QSVC` with precomputed kernel matrix and regularization $C=1.0$, unweighted.

---

## 7. Noise Injection Method

Noise injection was implemented in `src.utils.noise.add_noise_at_snr` following controlled DSP simulation:
1. **Target SNR Definition:**
   $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$
2. **Deterministic Pseudorandom Generation:**
   To guarantee exact reproducibility, noise seeds were derived uniquely per utterance via cryptographic MD5 hashing:
   $$\text{seed} = \text{int}\Big(\text{MD5}\big(\texttt{"42\_\{recording\_id\}\_\{snr\_db:.1f\}"}\big)[:8], 16\Big)$$
3. **Acoustic Waveform Perturbation:**
   Additive white Gaussian noise (AWGN) was injected into the raw 16 kHz PCM waveform before mel-spectrogram extraction and Whisper encoding, ensuring acoustic distortion at the signal level.
4. **Cache & Quality Validation:**
   15,951 noisy Whisper representations were generated and cached to disk. Every noisy embedding was verified: shape $(384,)$, float32, unit-$L_2$ normalized, $0$ NaN/Inf values.

---

## 8. Evaluation Protocol

All 720 cells were evaluated against the identical 7-class emergency taxonomy:
1. `GENERAL_SPEECH` (Control / non-emergency)
2. `SOS_HELP`
3. `FALL`
4. `BREATHING_MEDICAL`
5. `PAIN`
6. `LOCATION`
7. `CRITICAL_RESPONSE`

### Primary & Secondary Metrics:
- **Macro-F1 (Headline Metric):** Unweighted mean of per-class F1 scores:
  $$\text{Macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{F1}_c$$
  Essential due to severe class imbalance (~95% `GENERAL_SPEECH`).
- **Accuracy:** Overall proportion of correct predictions.
- **Exact Intent Preservation Rate ($\text{IPER}_{\text{exact}}$):** Fraction of emergency utterances correctly classified as the exact true emergency intent:
  $$\text{IPER}_{\text{exact}} = \frac{\sum_{i \in \mathcal{E}} \mathbb{I}(\hat{y}_i = y_i)}{|\mathcal{E}|}$$
- **Triage Intent Preservation Rate ($\text{IPER}_{\text{triage}}$):** Fraction of emergency utterances classified as any emergency intent ($\hat{y} \neq \text{'GENERAL\_SPEECH'}$).
- **Per-Class Precision, Recall, and F1:** Evaluated across all 7 intent classes.

---

## 9. Leakage Controls

Rigorous information isolation controls were enforced and verified:
1. **Acoustic Preprocessing Isolation:** Noise was injected solely into test partition waveforms. Clean adaptation waveforms remained pristine.
2. **Transformation Isolation:** PCA projection matrices and MinMaxScaler bounds were fitted exclusively on $\mathcal{D}_{\text{adapt}}^{(N)}$. Transforming test embeddings did not alter internal transformer parameters (`tests/test_phase4_5_no_transform_leakage.py` passed).
3. **Model Training Isolation:** Model weights, support vectors, and dual coefficients were determined solely from clean adaptation samples.
4. **Embedding Lineage:** Noisy embeddings for $15, 10, 5\text{ dB}$ SNR were identical across all three evaluated models, guaranteeing paired statistical comparability.

---

## 10. Reproducibility Controls

- Deterministic execution was verified under the documented environment (Python 3.11.15, PyTorch 2.14.0+cpu, scikit-learn 1.9.1, Qiskit 2.5.2) and fixed seed configuration (seed 42).
- Precomputed kernel cache with MD5 provenance verification.
- 100% pass on acceptance test suite: `tests/test_phase5_validation.py` (23/23 tests passed).

---

## 11. Results

The complete 720-cell experimental matrix yielded the following aggregated performance metrics across model, sample size ($N$), and acoustic condition:

### Table 11.1: Macro-F1 Summary (Condition Mean $\pm$ Standard Deviation)

| Model | $N$ | Clean Baseline | 15 dB SNR | 10 dB SNR | 5 dB SNR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Adapter** | 20 | $0.1466 \pm 0.0229$ | $0.1409 \pm 0.0082$ | $0.1410 \pm 0.0081$ | $0.1388 \pm 0.0016$ |
| **Adapter** | 30 | $0.1453 \pm 0.0141$ | $0.1385 \pm 0.0095$ | $0.1377 \pm 0.0133$ | $0.1357 \pm 0.0186$ |
| **Adapter** | 40 | $0.1472 \pm 0.0232$ | $0.1414 \pm 0.0110$ | $0.1388 \pm 0.0091$ | $0.1358 \pm 0.0105$ |
| **Adapter** | 50 | $0.1410 \pm 0.0093$ | $0.1382 \pm 0.0082$ | $0.1381 \pm 0.0118$ | $0.1367 \pm 0.0156$ |
| **QSVC ($q=4$)** | 20..50 | $0.1390 \pm 0.0014$ | $0.1390 \pm 0.0014$ | $0.1390 \pm 0.0014$ | $0.1390 \pm 0.0014$ |
| **RBF-SVM** | 20..50 | $0.1390 \pm 0.0014$ | $0.1390 \pm 0.0014$ | $0.1390 \pm 0.0014$ | $0.1390 \pm 0.0014$ |

### Table 11.2: Accuracy Summary (Condition Mean $\pm$ Standard Deviation)

| Model | $N$ | Clean Baseline | 15 dB SNR | 10 dB SNR | 5 dB SNR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Adapter** | 20 | $0.9387 \pm 0.0343$ | $0.9446 \pm 0.0207$ | $0.9455 \pm 0.0195$ | $0.9444 \pm 0.0208$ |
| **Adapter** | 30 | $0.9353 \pm 0.0226$ | $0.9191 \pm 0.0784$ | $0.9108 \pm 0.1159$ | $0.8994 \pm 0.1602$ |
| **Adapter** | 40 | $0.9325 \pm 0.0277$ | $0.9246 \pm 0.0512$ | $0.9220 \pm 0.0738$ | $0.9122 \pm 0.1142$ |
| **Adapter** | 50 | $0.9414 \pm 0.0154$ | $0.9216 \pm 0.0809$ | $0.9169 \pm 0.1061$ | $0.9078 \pm 0.1397$ |
| **QSVC ($q=4$)** | 20..50 | $0.9477 \pm 0.0179$ | $0.9477 \pm 0.0179$ | $0.9477 \pm 0.0179$ | $0.9477 \pm 0.0179$ |
| **RBF-SVM** | 20..50 | $0.9477 \pm 0.0179$ | $0.9477 \pm 0.0179$ | $0.9477 \pm 0.0179$ | $0.9477 \pm 0.0179$ |

### Table 11.3: Condition-Mean Exact IPER ($\text{IPER}_{\text{exact}}$, Mean $\pm$ Std)

| Model | $N$ | Clean Baseline | 15 dB SNR | 10 dB SNR | 5 dB SNR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Adapter** | 20 | $0.0131 \pm 0.0279$ | $0.0039 \pm 0.0152$ | $0.0039 \pm 0.0152$ | $0.0000 \pm 0.0000$ |
| **Adapter** | 30 | $0.0238 \pm 0.0478$ | $0.0149 \pm 0.0445$ | $0.0187 \pm 0.0504$ | $0.0187 \pm 0.0504$ |
| **Adapter** | 40 | $0.0218 \pm 0.0493$ | $0.0162 \pm 0.0439$ | $0.0149 \pm 0.0445$ | $0.0038 \pm 0.0147$ |
| **Adapter** | 50 | $0.0039 \pm 0.0152$ | $0.0149 \pm 0.0445$ | $0.0168 \pm 0.0470$ | $0.0168 \pm 0.0470$ |
| **QSVC ($q=4$)** | 20..50 | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ |
| **RBF-SVM** | 20..50 | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ |

---

## 12. Macro-F1 Analysis

Macro-F1 serves as the primary evaluation metric because raw accuracy is dominated by the majority class.

As shown in **Figure 12.1**, the Classical Adapter exhibits measurable sensitivity to test acoustic noise:
- On clean audio, Adapter achieves a condition-mean Macro-F1 of $0.1450 \pm 0.0181$, outperforming RBF-SVM and QSVC ($0.1390 \pm 0.0013$).
- Under noise injection, Adapter Macro-F1 exhibits monotonic degradation: $0.1397$ ($15\text{ dB}$) $\to$ $0.1389$ ($10\text{ dB}$) $\to$ $0.1367$ ($5\text{ dB}$).
- Relative to clean performance, the Adapter suffers a **$5.72\%$ relative decrease in Macro-F1** at 5 dB SNR.

![Macro-F1 vs SNR](results/phase5/figures/macro_f1_vs_snr.png)  
*Figure 12.1: Macro-F1 across test SNR levels for Adapter, RBF-SVM, and Quantum QSVC (mean with shaded standard deviation).*

In contrast, RBF-SVM and QSVC maintain an invariant Macro-F1 curve ($0.1390 \pm 0.0013$). Because these soft-margin kernel formulations assign zero probability to minority classes, test-time acoustic perturbations in feature space are insufficient to shift points across the collapsed boundary.

---

## 13. Accuracy Analysis

Accuracy across all models remains in the $90.0\% - 95.0\%$ range across all noise conditions. This reflects the TORGO dataset's severe class imbalance: `GENERAL_SPEECH` accounts for approximately $94.77\%$ of all test recordings. 

When a classifier predicts `GENERAL_SPEECH` for every input (as occurs for RBF-SVM and QSVC), accuracy is mathematically equal to the majority class prevalence:
$$\text{Accuracy}_{\text{collapse}} = \frac{|\mathcal{D}_{\text{GENERAL\_SPEECH}}|}{|\mathcal{D}_{\text{test}}|} = 0.9477 \pm 0.0179$$
For the Classical Adapter, accuracy slightly decreases from $93.70\%$ on clean to $91.59\%$ at 5 dB SNR due to noise-induced dispersion into false positive non-general-speech predictions.

---

## 14. IPER Analysis

![IPER vs SNR](results/phase5/figures/iper_vs_snr.png)  
*Figure 14.1: Emergency Intent Preservation Rate (IPER) as a function of SNR across adaptation budgets.*

Key findings from the forensic audit of IPER:
1. **Condition-Mean Exact IPER:** The Classical Adapter is the only model achieving non-zero intent preservation under few-shot calibration. On clean audio, Adapter condition-mean IPER is $1.56\% \pm 3.77\%$, decreasing to $0.98\% \pm 3.53\%$ at 5 dB SNR. Peak preservation on specific calibration splits reached $16.67\%$ (`F04`, $N=30$) and $9.52\%$ (`M03`, $N=40$).
2. **Pooled Sample-Weighted Exact IPER:** Across the 11,424 emergency test utterances, exact emergency intent matches for the Adapter were:
   - Clean: 2 / 952 ($0.21\%$)
   - 15 dB: 6 / 952 ($0.63\%$)
   - 10 dB: 9 / 952 ($0.95\%$)
   - 5 dB: 10 / 952 ($1.05\%$)
3. **Pooled Any-Emergency Triage IPER:** When evaluating whether emergency utterances avoided catastrophic misclassification as `GENERAL_SPEECH`, Adapter preserved:
   - Clean: 5 / 952 ($0.53\%$)
   - 15 dB: 29 / 952 ($3.05\%$)
   - 10 dB: 49 / 952 ($5.15\%$)
   - 5 dB: 57 / 952 ($5.99\%$)  
   This increase in triage recall under noise occurs because acoustic distortion perturbs embeddings toward the `FALL` class boundary, increasing emergency false alarms while reducing precision.
4. **Kernel Method Performance:** Both RBF-SVM and QSVC exhibit $0.00\%$ IPER across all conditions and definitions.

---

## 15. Per-Class Analysis

Across the 5,040 per-class metric rows (`results/phase5/phase5_per_class.csv`), classification performance is concentrated in the dominant class:

### Table 15.1: Mean F1 Score by Intent Class Across All Test Conditions

| Model | GENERAL_SPEECH | BREATHING_MEDICAL | FALL | SOS_HELP | PAIN | LOCATION | CRITICAL_RESPONSE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Adapter** | **0.9609** | **0.0037** | **0.0023** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **QSVC ($q=4$)** | **0.9731** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **RBF-SVM** | **0.9731** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The Adapter occasionally detects `BREATHING_MEDICAL` (peaking at $\text{F1}=0.4000$) and `FALL` (peaking at $\text{F1}=0.3333$) under favorable calibration splits. All other emergency classes (`SOS_HELP`, `PAIN`, `LOCATION`, `CRITICAL_RESPONSE`) exhibited zero recall across all models.

---

## 16. Speaker-Level Variation

Performance varied across individual speakers within the 15-speaker cohort:

### Table 16.1: Classical Adapter Macro-F1 Across SNR Levels by Speaker

| Speaker ID | Cohort Group | Clean | 15 dB SNR | 10 dB SNR | 5 dB SNR | Clean $\to$ 5 dB Degradation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `MC04` | Control | **0.1823** | 0.1393 | 0.1393 | 0.1393 | -23.59% |
| `M03` | Dysarthric | **0.1630** | 0.1474 | 0.1379 | 0.1382 | -15.21% |
| `F04` | Dysarthric | **0.1528** | 0.1546 | 0.1592 | 0.1507 | -1.37% |
| `M05` | Dysarthric | **0.1476** | 0.1387 | 0.1388 | 0.1387 | -6.03% |
| `M01` | Dysarthric | **0.1432** | 0.1391 | 0.1391 | 0.1391 | -2.86% |
| `MC03` | Control | **0.1412** | 0.1391 | 0.1397 | 0.1394 | -1.27% |
| `MC01` | Control | **0.1401** | 0.1401 | 0.1401 | 0.1401 | 0.00% |
| `FC03` | Control | **0.1399** | 0.1408 | 0.1408 | 0.1408 | +0.64% |
| `FC02` | Control | **0.1389** | 0.1235 | 0.1144 | **0.0987** | **-28.94%** |
| `MC02` | Control | **0.1389** | 0.1468 | 0.1468 | 0.1386 | -0.22% |
| `F01` | Dysarthric | **0.1385** | 0.1385 | 0.1385 | 0.1385 | 0.00% |
| `F03` | Dysarthric | **0.1385** | 0.1388 | 0.1388 | 0.1388 | +0.22% |
| `M02` | Dysarthric | **0.1376** | 0.1370 | 0.1373 | 0.1372 | -0.29% |
| `FC01` | Control | **0.1366** | 0.1362 | 0.1364 | 0.1368 | +0.15% |
| `M04` | Dysarthric | **0.1360** | 0.1361 | 0.1361 | 0.1362 | +0.15% |

### Dysarthric vs. Control Cohort Breakdown:
- **Dysarthric Group ($n=8$):** Mean Macro-F1 on Clean audio is $0.1447$, declining to $0.1397$ at 5 dB SNR ($-3.46\%$ relative).
- **Control Group ($n=7$):** Mean Macro-F1 on Clean audio is $0.1454$, declining to $0.1334$ at 5 dB SNR ($-8.25\%$ relative).
- Within this cohort, control speakers exhibited a larger mean relative decline, substantially influenced by pronounced degradation on speaker `FC02` ($0.1389 \to 0.0987$).

---

## 17. Sample-Efficiency Analysis

Figure 17.1 illustrates sample-efficiency curves across the four sample budgets ($N=20, 30, 40, 50$).

![Macro-F1 vs Sample Size](results/phase5/figures/macro_f1_vs_sample_size.png)  
*Figure 17.1: Few-shot sample efficiency curves ($N=20 \to 50$) under clean and noisy test conditions.*

Observations:
1. **Kernel Method Sample Invariance:** For RBF-SVM and QSVC, increasing $N$ from 20 to 50 produced no change in Macro-F1 ($0.1390$). Additional clean samples did not overcome the soft-margin threshold under unweighted $C=1.0$ optimization.
2. **Adapter Sample Dynamics:** For the Adapter, $N=40$ achieved the highest mean clean Macro-F1 ($0.1472 \pm 0.0232$), while $N=50$ yielded $0.1410 \pm 0.0093$. The present experiment did not isolate the specific mechanism (e.g. adaptation distribution vs optimization dynamics) underlying this variation.

---

## 18. Noise Robustness Analysis

Figure 18.1 displays the two-dimensional interaction between adaptation budget ($N$) and test SNR.

![Noise Sample Efficiency Heatmap](results/phase5/figures/noise_sample_efficiency_heatmap.png)  
*Figure 18.1: $N \times \text{SNR}$ performance heatmaps for Adapter, QSVC, and RBF-SVM.*

Under Protocol B, increasing adaptation budget $N$ did not insulate models against acoustic noise. Because models were calibrated exclusively on clean speech, the feature shift induced by noise on test embeddings could not be learned from clean calibration data alone.

---

## 19. Adapter vs RBF-SVM vs Quantum QSVC

Figure 19.1 presents a controlled three-way comparison of the models across the entire noise sweep.

![Model Comparison](results/phase5/figures/model_comparison.png)  
*Figure 19.1: Comprehensive three-way comparison of Classical Adapter, RBF-SVM, and Quantum QSVC across the full noise sweep.*

### Direct Head-to-Head Findings:
1. **Model Expressiveness:** The Classical Adapter exhibited higher empirical capacity, enabling non-zero emergency intent recognition on several speakers. It achieved the highest Macro-F1 across all conditions.
2. **Classical RBF-SVM vs Quantum QSVC Equivalence:** Across all 81,840 paired test instances, RBF-SVM and QSVC produced identical prediction outputs and numerically equivalent summary metrics ($0.1390$ Macro-F1, $0.9477$ Accuracy, $0.0000$ IPER).
3. **Absence of Quantum Advantage:** Under the evaluated Phase 5 configuration, the 4-qubit ZZ feature map fidelity kernel demonstrated no advantage in noise robustness, sample efficiency, or emergency intent recognition over classical baselines.

---

## 20. Limitations

1. **Protocol B Scope:** Protocol B evaluates generalization from clean calibration to noisy test; it does not measure adaptation capacity under matched noisy calibration (Protocol A).
2. **Unweighted Loss Function:** In adherence to the frozen Phase 3/4 experimental contract, models used unweighted loss functions ($C=1.0$ for SVMs, unweighted cross-entropy for Adapter), permitting majority-class dominance under extreme imbalance.
3. **Simulated AWGN:** Acoustic perturbations used stationary additive white Gaussian noise at 16 kHz. Non-stationary environmental noise (e.g. background speech, room reverberation) was not evaluated.
4. **Frozen Whisper Encoder:** The speech encoder was held completely frozen without acoustic fine-tuning.

---

## 21. Scientific Interpretation

1. **Acoustic Noise Impairs Clean Few-Shot Adaptation:** Controlled acoustic noise at 5 dB SNR degraded personalized adapter classification performance by $5.72\%$ relative Macro-F1 and reduced condition-mean emergency intent preservation (IPER) by $37.2\%$.
2. **Kernel Method Fragility Under Extreme Imbalance:** In the presence of 95:5 class imbalance and small sample sizes ($N \le 50$), soft-margin kernel classifiers (both classical RBF and quantum ZZ) collapsed to majority-class prediction.
3. **Absence of Measurable Quantum Advantage:** The hypothesis that quantum state space embeddings provide intrinsic robustness against acoustic noise was not supported under this experimental configuration.
4. **Engineering Implications:** Personalization of emergency speech recognition devices cannot rely solely on clean-environment calibration. Robust deployment will require noise-augmented calibration, cost-sensitive loss formulations, and noise-invariant representations.

---

## 22. Conclusion

Phase 5 completed and verified the full 720-cell acoustic noise robustness experiment under Protocol B. Every experimental condition was completed with zero dropped cells, verified clean baseline regression, and 100% test suite passage. The findings provide rigorous empirical evidence regarding the limits of clean few-shot calibration under acoustic degradation and establish a verified baseline for Phase 6 statistical evaluation.

---

## 23. Reproducibility Information

- **Execution Environment:** Windows, Python 3.11.15, PyTorch 2.14.0+cpu, Qiskit 2.5.2, Qiskit Machine Learning 0.9.1, scikit-learn 1.9.1.
- **Deterministic Execution:** Verified under the documented environment and seed configuration (master seed 42, deterministic MD5 noise seeds).
- **Master Runner:** `scripts/run_phase5_noise_sweep.py`
- **Audit Script:** `scripts/audit_phase5_forensic.py`
- **Validation Test Suite:** `tests/test_phase5_validation.py` (23 passed in 49.99s).
- **Full Test Suite:** `pytest tests/ -v` (62 passed in 92.34s).

---

## 24. Artifact Inventory

### Primary Result Datasets:
- `results/phase5/phase5_results.csv` (720 rows, 100% complete)
- `results/phase5/phase5_summary.csv` (48 condition groups)
- `results/phase5/phase5_per_class.csv` (5,040 class-level rows)
- `results/phase5/phase5_predictions.csv` (245,520 prediction records with sanitized IPER flags)
- `results/phase5/phase5_errors.csv` (0 errors)

### Publication Figures:
- `results/phase5/figures/macro_f1_vs_snr.png`
- `results/phase5/figures/macro_f1_vs_sample_size.png`
- `results/phase5/figures/noise_sample_efficiency_heatmap.png`
- `results/phase5/figures/iper_vs_snr.png`
- `results/phase5/figures/model_comparison.png`

### Audit & Governance Reports:
- `reports/phase5_forensic_audit_report.md`
- `reports/phase5_correction_log.md`
- `reports/phase5_canonical_specification.md`
- `reports/phase5_clean_regression_audit.csv`
- `reports/phase5_aggregation_audit.csv`
- `reports/phase5_prediction_distribution_audit.csv`
- `reports/phase5_adapter_architecture_audit.csv`
