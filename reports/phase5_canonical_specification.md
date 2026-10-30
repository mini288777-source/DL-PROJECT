# Phase 5 Canonical Specification: Frozen Reference for Phase 6

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Dysarthric Speech Database  
**Phase:** 5 (Closed and Frozen)  
**Authoritative Reference For:** Phase 6 (Statistical Evaluation & IPER Analysis)  
**Date:** October 2026  

---

## 1. Speech Representation & Acoustic Encoder

- **Encoder Model:** `openai/whisper-tiny` (Encoder-only, pretrained weights strictly frozen).
- **Audio Preprocessing:** 16,000 Hz single-channel PCM WAV audio, padded/truncated to 30.0 seconds, 80-channel log-Mel spectrogram.
- **Embedding Dimension:** 384-dimensional continuous feature vectors.
- **Temporal Pooling:** Mean-pooling across the 1,500 encoder frame representations.
- **Vector Normalization:** Strict unit-$L_2$ Euclidean normalization ($\|x\|_2 = 1.0$).
- **Numeric Precision:** Standard 32-bit floating point (`float32`).

---

## 2. Speaker Cohort & Demographic Distribution

- **Total Cohort Size:** Exactly 15 speakers ($100\%$ adult dysarthric and matched controls).
- **Dysarthric Group ($n=8$):** `F01, F03, F04, M01, M02, M03, M04, M05`.
- **Control Group ($n=7$):** `FC01, FC02, FC03, MC01, MC02, MC03, MC04`.
- **Eligibility Criteria:** Minimum 50 clean adaptation utterances and minimum 20 test utterances.

---

## 3. Fixed Emergency Intent Taxonomy

Exactly seven mutually exclusive classification labels:
1. `GENERAL_SPEECH` (Index 0 / Non-emergency conversational baseline)
2. `SOS_HELP` (Emergency)
3. `FALL` (Emergency)
4. `BREATHING_MEDICAL` (Emergency)
5. `PAIN` (Emergency)
6. `LOCATION` (Emergency)
7. `CRITICAL_RESPONSE` (Emergency)

---

## 4. Few-Shot Calibration Budgets ($N$)

- **Sample Sizes:** $N \in \{20, 30, 40, 50\}$ clean calibration utterances per speaker.
- **Sampling Hierarchy:** Strictly nested subsets:
  $$\mathcal{D}_{\text{adapt}}^{(20)} \subset \mathcal{D}_{\text{adapt}}^{(30)} \subset \mathcal{D}_{\text{adapt}}^{(40)} \subset \mathcal{D}_{\text{adapt}}^{(50)}$$
- **Disjoint Test Sets:**
  $$\mathcal{D}_{\text{test}}^{(N)} = \mathcal{D}_{\text{speaker}} \setminus \mathcal{D}_{\text{adapt}}^{(N)}$$
- **Acoustic Leakage Controls:** Prompt-grouped isolation ensures parallel audio channels (`headMic`, `arrayMic`) of the same vocal event are never split across train and test.

---

## 5. Model Architectures & Hyperparameters

### 1. Classical Adapter Head (`adapter`)
- **Type:** 2-layer Feedforward Multi-Layer Perceptron (MLP).
- **Architecture:**
  $$\text{Input}(384) \to \text{Linear}(384, 64) \to \text{LayerNorm}(64) \to \text{GELU}() \to \text{Dropout}(p=0.1) \to \text{Linear}(64, 7)$$
- **Trainable Parameters:** Exactly **25,223** weights and biases.
- **Optimization:** AdamW ($\text{lr}=0.005$, $\text{weight\_decay}=10^{-4}$), batch size 8, 60 epochs.
- **Loss Formulation:** Standard unweighted Cross-Entropy loss.
- **Weight Initialization:** Xavier uniform initialization with zero biases.

### 2. Classical Support Vector Classifier (`rbf_svm`)
- **Type:** Scikit-learn Support Vector Classifier (`sklearn.svm.SVC`).
- **Kernel Function:** Radial Basis Function (RBF), $K(x, x') = \exp(-\gamma \|x - x'\|^2)$, $\gamma = \text{'scale'}$.
- **Regularization:** $C = 1.0$.
- **Class Weighting:** None (unweighted).

### 3. Primary Quantum Support Vector Classifier (`qsvc`)
- **Dimensionality Reduction:** Principal Component Analysis ($384 \to 4$) fitted exclusively on clean adaptation samples.
- **Feature Scaling:** MinMaxScaler scaling components into $[0, \pi]$ fitted exclusively on clean adaptation samples.
- **Quantum Feature Map:** Second-order Pauli $ZZ$ Feature Map (`ZZFeatureMap`), $q=4$ qubits, $\text{reps}=1$, linear entanglement.
- **Kernel Formulation:** Fidelity Quantum Kernel $\kappa(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$.
- **Backend:** Qiskit statevector simulation (`StatevectorSampler`).
- **Classifier:** Soft-margin SVC with precomputed quantum fidelity kernel, $C = 1.0$, unweighted.

---

## 6. Acoustic Noise Simulation & SNR Levels

- **Protocol:** Protocol B (Clean Calibration $\to$ Noisy Evaluation).
- **Noise Formulation:** Additive White Gaussian Noise (AWGN) applied directly to raw 16 kHz audio waveforms:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$
- **Test Acoustic Conditions:**
  1. `clean`: Unperturbed baseline audio ($\infty$ dB SNR).
  2. `snr15`: 15.0 dB SNR AWGN.
  3. `snr10`: 10.0 dB SNR AWGN.
  4. `snr5`: 5.0 dB SNR AWGN.
- **Deterministic Noise Seed Formula:**
  $$\text{seed} = \text{int}\Big(\text{MD5}\big(\texttt{"42\_\{recording\_id\}\_\{snr\_db:.1f\}"}\big)[:8], 16\Big)$$

---

## 7. Mathematical Definitions of Metrics

### 1. Macro-F1 (Headline Metric)
$$\text{Macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{F1}_c = \frac{1}{7} \sum_{c=1}^7 \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

### 2. Accuracy
$$\text{Accuracy} = \frac{\sum_{i=1}^M \mathbb{I}(\hat{y}_i = y_i)}{M}$$

### 3. Exact Intent Preservation Rate ($\text{IPER}_{\text{exact}}$)
$$\text{IPER}_{\text{exact}} = \frac{\sum_{i \in \mathcal{E}} \mathbb{I}(\hat{y}_i = y_i)}{|\mathcal{E}|}$$
where $\mathcal{E} = \{i : y_i \neq \text{'GENERAL\_SPEECH'}\}$.

### 4. Triage Intent Preservation Rate ($\text{IPER}_{\text{triage}}$)
$$\text{IPER}_{\text{triage}} = \frac{\sum_{i \in \mathcal{E}} \mathbb{I}(\hat{y}_i \neq \text{'GENERAL\_SPEECH'})}{|\mathcal{E}|}$$

---

## 8. Authoritative Data and File Registry

### Machine-Readable Result Files:
- `results/phase5/phase5_results.csv`: 720 rows (Canonical experimental matrix).
- `results/phase5/phase5_summary.csv`: 48 condition rows (Aggregated means and standard deviations).
- `results/phase5/phase5_per_class.csv`: 5,040 rows (Per-class precision, recall, F1, support).
- `results/phase5/phase5_predictions.csv`: 245,520 prediction records with sanitized emergency flags.

### Authoritative Publication Figures:
- `results/phase5/figures/macro_f1_vs_snr.png`
- `results/phase5/figures/macro_f1_vs_sample_size.png`
- `results/phase5/figures/noise_sample_efficiency_heatmap.png`
- `results/phase5/figures/iper_vs_snr.png`
- `results/phase5/figures/model_comparison.png`

---

## 9. Information Isolation & Leakage Guardrails

1. **Zero Preprocessing Leakage:** PCA projection and scaling bounds depend strictly on clean adaptation samples.
2. **Zero Split Leakage:** Test samples never enter model optimization, support vector identification, or dual coefficient calculation.
3. **Paired Test Comparability:** The exact same noisy waveform and 384-D Whisper embedding is evaluated across all three models.
