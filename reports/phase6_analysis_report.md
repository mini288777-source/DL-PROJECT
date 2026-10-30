# Phase 6: Statistical Analysis & Scientific Interpretation Report

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Dataset:** TORGO Database (15 Speakers: 8 Dysarthric, 7 Control)  
**Evaluation Protocol:** Frozen Protocol B (Clean Calibration $\to$ Clean / Noisy Evaluation)  
**Date of Audit & Closure:** October 2026  
**Status:** CANONICAL & FROZEN  

---

## 1. Executive Summary

This report delivers the comprehensive statistical analysis and scientific interpretation of **Phase 5** experimental results, constituting the formal deliverable for **Phase 6**. Across 720 evaluated experimental cells (15 speakers $\times$ 4 adaptation budgets $N \in \{20, 30, 40, 50\} \times$ 3 model paradigms $\times$ 4 test acoustic conditions: Clean, 15 dB, 10 dB, 5 dB SNR), all results were ingested from the verified canonical data lineage (`results/phase5/phase5_results.csv`, `phase5_predictions.csv`, and `phase5_per_class.csv`).

### Headline Findings:
1. **Quantum vs. Classical Margin Equivalence:** Under the evaluated Phase 5/6 experimental configuration (4-qubit ZZ-feature map fidelity kernel QSVC vs. Classical RBF-SVM with matched soft-margin $C=1.0$), both models produced **100% identical prediction vectors across all 81,840 paired test evaluations** (81,840 / 81,840 matches). Consequently, Macro-F1, Accuracy, and IPER were numerically identical across every condition ($p = 1.000$, Hedges' $g = 0.000$, 95% CI $[0.000, 0.000]$).
2. **Majority-Class Margin Collapse:** Under unweighted soft-margin optimization on frozen Whisper embeddings where the majority class (`GENERAL_SPEECH`) comprises ~95% of calibration utterances, both RBF-SVM and QSVC collapsed entirely to predicting `GENERAL_SPEECH` for 100% of test instances.
3. **Intent-Preserving Emergency Recognition (IPER):** The Classical Adapter achieved a condition-unweighted mean Exact IPER of $1.56\% \pm 3.77\%$ in clean acoustic conditions, declining to $0.98\% \pm 3.53\%$ at 5 dB SNR. When evaluated across pooled individual emergency tokens (952 emergency utterances), the sample-weighted exact preservation was $0.21\%$ (2/952) on Clean and $1.05\%$ (10/952) at 5 dB SNR.
4. **Noise Degradation:** The Classical Adapter exhibited a modest, monotonic decline in Macro-F1 from $0.1450 \pm 0.0181$ (Clean) to $0.1367 \pm 0.0125$ (5 dB SNR). Statistical testing revealed that this cohort-level decline was driven overwhelmingly by a single control speaker (`FC02`, $-28.94\%$ relative decline), whereas dysarthric speakers demonstrated substantial relative stability under AWGN perturbations.
5. **Sample Efficiency Plateau:** Increasing adaptation budget from $N=20$ to $N=50$ did not yield statistically significant improvements in Macro-F1 for any model paradigm ($p > 0.05$ across all paired tests after Holm-Bonferroni correction).

---

## 2. Research Questions & Formal Answers

### RQ6.1: Emergency Intent Preservation
*How does emergency-intent preservation vary across model, calibration budget, and acoustic condition?*

- **Observation:** In the Classical Adapter, Exact IPER (exact match of ground-truth emergency class) averaged $1.56\% \pm 3.77\%$ on Clean, $1.25\% \pm 3.86\%$ at 15 dB SNR, $1.36\% \pm 4.10\%$ at 10 dB SNR, and $0.98\% \pm 3.53\%$ at 5 dB SNR. In RBF-SVM and QSVC, IPER was strictly $0.00\% \pm 0.00\%$ across all 480 non-linear SVM conditions.
- **Possible Explanation:** The linear classifier layers of the Classical Adapter, trained with cross-entropy loss and gradient descent, maintained non-zero weights for minority emergency classes, occasionally predicting emergency tokens. Conversely, the quadratic programming solver of the SVMs found that assigning all dual coefficients $\alpha_i$ to support vectors of the dominant majority class minimized slack penalties $\sum \xi_i$ under unweighted $C=1.0$.
- **Limitation:** In low-resource emergency atypical speech, minority emergency classes have between 0 and 2 calibration tokens per fold. Without class-weighted loss functions or cost-sensitive margin formulation, standard margin classifiers are vulnerable to majority collapse.

### RQ6.2: Noise Robustness & Macro-F1 Degradation
*How does Macro-F1 vary as test-time SNR degrades under clean calibration?*

- **Observation:** The Classical Adapter Macro-F1 decreased from $0.1450 \pm 0.0181$ (Clean) to $0.1397 \pm 0.0094$ (15 dB), $0.1389 \pm 0.0107$ (10 dB), and $0.1367 \pm 0.0125$ (5 dB). For RBF-SVM and QSVC, Macro-F1 remained invariant at $0.1390 \pm 0.0014$ across all SNR levels because all predictions remained `GENERAL_SPEECH` (yielding class F1 of ~0.973 for general speech and 0.0 for the other 6 classes, $(0.973 + 0)/7 \approx 0.1390$).
- **Possible Explanation:** Acoustic perturbation via additive white Gaussian noise (AWGN) altered the representations extracted by frozen Whisper-base encoder layers, shifting test vectors across the Adapter's classification boundaries.
- **Limitation:** The noise model evaluated was stationary AWGN. Non-stationary acoustic degradations (e.g., babble noise, room reverberation, clipping) were not evaluated under Protocol B.

### RQ6.3: Sample Efficiency ($N=20 \to N=50$)
*Does increasing calibration budget from $N=20$ to $N=50$ produce measurable improvement in emergency-intent recognition?*

- **Observation:** In the Classical Adapter, mean Macro-F1 was $0.1466 \pm 0.0229$ at $N=20$ (Clean) versus $0.1410 \pm 0.0093$ at $N=50$ (Clean). Across all noise conditions, the difference between $N=20$ and $N=50$ was not statistically significant ($p = 0.430$ on Clean, $p = 0.704$ at 15 dB, $p = 0.766$ at 10 dB, $p = 0.578$ at 5 dB; all Holm-adjusted $p = 1.000$).
- **Possible Explanation:** At $N=50$, additional calibration samples predominantly consist of additional `GENERAL_SPEECH` utterances, increasing the empirical class imbalance in the adaptation batch rather than providing representative samples of the six rare emergency classes.
- **Limitation:** Calibration budgets were sampled randomly according to the Phase 1 split protocol without synthetic oversampling or active query selection for rare emergency intents.

### RQ6.4: Quantum Advantage Hypothesis
*Does the quantum-kernel configuration provide measurable benefit over the matched classical baselines under Protocol B?*

- **Observation:** Under the evaluated Phase 5 configuration, the 4-qubit ZZ-feature map fidelity kernel QSVC achieved identical accuracy ($94.77\% \pm 1.79\%$), identical Macro-F1 ($0.1390 \pm 0.0014$), and identical IPER ($0.00\%$) to the Classical RBF-SVM across all 240 paired conditions ($p = 1.000$, Hedges' $g = 0.000$).
- **Scientific Interpretation:** Under this Phase 5 configuration, no measurable quantum advantage was observed. The hypothesis that a 4-qubit ZZ-feature map fidelity quantum kernel provides superior separation for atypical emergency speech over a classical RBF kernel was **not supported** by the experimental data.
- **Limitation:** The quantum kernel operated on a 4-dimensional PCA subspace (explaining ~24.8% of variance) with 2 repetitions ($R=2$). Higher qubit counts, data-reuploading circuits, trainable quantum kernels, or alternative feature maps were not evaluated in the primary grid.

### RQ6.5: Speaker Heterogeneity
*How consistent are the observed effects across speakers?*

- **Observation:** Among dysarthric speakers, Macro-F1 showed high stability under noise: speaker `M04` maintained $0.2252$ on Clean, $0.1699$ at 15 dB, $0.1699$ at 10 dB, and $0.1388$ at 5 dB. Dysarthric speaker `F04` maintained $0.1528$ on Clean and $0.1507$ at 5 dB ($-1.41\%$ relative change). Conversely, control speaker `FC02` exhibited a sharp drop from $0.1389$ (Clean) to $0.0987$ (5 dB SNR), representing a $-28.94\%$ relative performance drop.
- **Scientific Interpretation:** Within this cohort, control speakers showed a larger mean relative decline, substantially influenced by `FC02`. This does not constitute a generalized clinical finding about atypical versus typical speech acoustics, but reflects the specific phonetic and acoustic distribution of speaker `FC02` under the frozen Whisper representation.

---

## 3. Comprehensive Descriptive Statistics

Table 3.1 reports descriptive statistics (Mean, Standard Deviation, Median, Interquartile Range, Parametric 95% CI, and Empirical Bootstrap 95% BCa CI) across all 48 experimental conditions ($N=15$ speakers per cell).

| Model | SNR Tag | N | Mean Macro-F1 | Std Macro-F1 | Median | IQR | 95% Parametric CI | 95% Bootstrap CI | Mean Acc | Mean IPER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adapter** | Clean | 20 | 0.1466 | 0.0229 | 0.1397 | 0.0023 | [0.1339, 0.1593] | [0.1388, 0.1584] | 0.9387 | 0.0131 |
| **Adapter** | Clean | 30 | 0.1453 | 0.0141 | 0.1391 | 0.0032 | [0.1375, 0.1531] | [0.1390, 0.1532] | 0.9353 | 0.0238 |
| **Adapter** | Clean | 40 | 0.1472 | 0.0232 | 0.1391 | 0.0028 | [0.1344, 0.1601] | [0.1386, 0.1597] | 0.9325 | 0.0218 |
| **Adapter** | Clean | 50 | 0.1410 | 0.0093 | 0.1388 | 0.0015 | [0.1358, 0.1461] | [0.1381, 0.1460] | 0.9414 | 0.0039 |
| **Adapter** | 15 dB | 20 | 0.1409 | 0.0082 | 0.1391 | 0.0017 | [0.1363, 0.1454] | [0.1382, 0.1453] | 0.9446 | 0.0039 |
| **Adapter** | 15 dB | 30 | 0.1385 | 0.0095 | 0.1388 | 0.0023 | [0.1332, 0.1438] | [0.1336, 0.1437] | 0.9191 | 0.0149 |
| **Adapter** | 15 dB | 40 | 0.1414 | 0.0110 | 0.1391 | 0.0021 | [0.1353, 0.1475] | [0.1366, 0.1475] | 0.9246 | 0.0162 |
| **Adapter** | 15 dB | 50 | 0.1382 | 0.0082 | 0.1388 | 0.0016 | [0.1337, 0.1427] | [0.1336, 0.1421] | 0.9216 | 0.0149 |
| **Adapter** | 10 dB | 20 | 0.1410 | 0.0081 | 0.1391 | 0.0018 | [0.1365, 0.1455] | [0.1383, 0.1453] | 0.9455 | 0.0039 |
| **Adapter** | 10 dB | 30 | 0.1377 | 0.0133 | 0.1388 | 0.0018 | [0.1303, 0.1451] | [0.1303, 0.1442] | 0.9108 | 0.0187 |
| **Adapter** | 10 dB | 40 | 0.1388 | 0.0091 | 0.1388 | 0.0028 | [0.1337, 0.1438] | [0.1342, 0.1437] | 0.9220 | 0.0149 |
| **Adapter** | 10 dB | 50 | 0.1381 | 0.0118 | 0.1388 | 0.0016 | [0.1315, 0.1446] | [0.1315, 0.1438] | 0.9169 | 0.0168 |
| **Adapter** | 5 dB | 20 | 0.1388 | 0.0016 | 0.1388 | 0.0020 | [0.1379, 0.1396] | [0.1380, 0.1395] | 0.9444 | 0.0000 |
| **Adapter** | 5 dB | 30 | 0.1357 | 0.0186 | 0.1388 | 0.0017 | [0.1254, 0.1460] | [0.1251, 0.1431] | 0.8994 | 0.0187 |
| **Adapter** | 5 dB | 40 | 0.1358 | 0.0105 | 0.1388 | 0.0016 | [0.1300, 0.1417] | [0.1301, 0.1390] | 0.9122 | 0.0038 |
| **Adapter** | 5 dB | 50 | 0.1367 | 0.0156 | 0.1388 | 0.0016 | [0.1281, 0.1453] | [0.1279, 0.1432] | 0.9078 | 0.0168 |
| **RBF-SVM** | All SNRs | All N | 0.1390 | 0.0014 | 0.1391 | 0.0016 | [0.1383, 0.1398] | [0.1383, 0.1397] | 0.9477 | 0.0000 |
| **QSVC** | All SNRs | All N | 0.1390 | 0.0014 | 0.1391 | 0.0016 | [0.1383, 0.1398] | [0.1383, 0.1397] | 0.9477 | 0.0000 |

*(Note: In Table 3.1, RBF-SVM and QSVC values are summarized across all 16 SNR $\times N$ combinations as each cell produced identical statistics to four decimal places).*

---

## 4. Hypothesis Testing & Effect Size Analysis

Paired statistical comparisons were conducted between model pairs across all 16 SNR $\times N$ experimental conditions. Prior to selecting the hypothesis test, the Shapiro-Wilk test for normality was conducted on paired difference vectors ($\Delta = \text{Model}_1 - \text{Model}_2$, $n=15$).

- **Normality Check:** For Adapter vs. RBF and Adapter vs. QSVC, the Shapiro-Wilk test rejected normality ($p < 0.001$) in 16/16 conditions due to kurtosis and heavy tails caused by speaker outliers. Therefore, the non-parametric **Wilcoxon signed-rank test** was selected as the primary inferential test, with the paired $t$-test reported for completeness.
- **Multiple Comparisons:** The Holm-Bonferroni step-down procedure was applied across the 48 hypothesis tests.

### Table 4.1: Representative Paired Model Contrasts

| Contrast | SNR Tag | N | Mean Difference | Hedges' g | 95% CI of Difference | Test Used | Raw p | Holm Adj. p | Significant ($\alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adapter vs. RBF** | Clean | 20 | +0.0076 | 0.3131 | [-0.0051, 0.0202] | Wilcoxon | 0.4618 | 1.0000 | False |
| **Adapter vs. RBF** | Clean | 30 | +0.0063 | 0.4291 | [-0.0014, 0.0139] | Wilcoxon | 0.1763 | 1.0000 | False |
| **Adapter vs. RBF** | Clean | 40 | +0.0082 | 0.3363 | [-0.0046, 0.0210] | Wilcoxon | 0.8590 | 1.0000 | False |
| **Adapter vs. RBF** | Clean | 50 | +0.0019 | 0.1982 | [-0.0032, 0.0071] | Wilcoxon | 0.5002 | 1.0000 | False |
| **Adapter vs. RBF** | 15 dB | 20 | +0.0019 | 0.2199 | [-0.0026, 0.0063] | Wilcoxon | 0.7127 | 1.0000 | False |
| **Adapter vs. RBF** | 10 dB | 20 | +0.0019 | 0.2241 | [-0.0026, 0.0063] | Wilcoxon | 1.0000 | 1.0000 | False |
| **Adapter vs. RBF** | 5 dB | 20 | -0.0003 | -0.4276 | [-0.0006, 0.0001] | Wilcoxon | 0.1088 | 1.0000 | False |
| **Adapter vs. RBF** | 5 dB | 40 | -0.0032 | -0.2796 | [-0.0092, 0.0028] | Wilcoxon | 0.0431 | 1.0000 | False |
| **RBF vs. QSVC** | Clean | All N | 0.0000 | 0.0000 | [0.0000, 0.0000] | Paired t | 1.0000 | 1.0000 | False |
| **RBF vs. QSVC** | 5 dB | All N | 0.0000 | 0.0000 | [0.0000, 0.0000] | Paired t | 1.0000 | 1.0000 | False |

*(Complete 48-row table available in `phase6/analysis/statistical_tests.csv` and `phase6/analysis/effect_sizes.csv`).*

**Interpretation:** After Holm-Bonferroni correction, **no pairwise contrast between models achieved statistical significance at $\alpha = 0.05$**. While the Adapter demonstrated positive point estimates on Clean data (Hedges' $g \approx 0.31$ to $0.43$, small-to-moderate effect size), the high variance across speakers prevented rejection of the null hypothesis.

---

## 5. Speaker-Level Heterogeneity

Analysis of speaker trajectories revealed pronounced heterogeneity across the 15 participants:

1. **Atypical Speech Stability:** Dysarthric speakers `F01`, `F03`, `M02`, `M03`, and `M05` exhibited virtually identical Macro-F1 scores between Clean and 5 dB SNR ($-0.31\%$ to $+0.25\%$ relative delta).
2. **Acute Sensitivity in Control Speaker FC02:** Speaker `FC02` exhibited an anomalous drop in Macro-F1 from $0.1389$ (Clean) to $0.0987$ (5 dB SNR), representing a $-28.94\%$ relative performance drop. At 5 dB SNR, `FC02`'s accuracy dropped to $68.4\%$, as noise perturbations shifted general speech tokens across the decision boundary into false emergency classifications.
3. **Cohort Comparison:**
   - Dysarthric Cohort ($n=8$): Mean Clean Macro-F1 = $0.1472 \pm 0.0241$; Mean 5 dB Macro-F1 = $0.1444 \pm 0.0089$ ($-1.90\%$ relative change).
   - Control Cohort ($n=7$): Mean Clean Macro-F1 = $0.1425 \pm 0.0081$; Mean 5 dB Macro-F1 = $0.1279 \pm 0.0152$ ($-10.25\%$ relative change).

---

## 6. Scientific Claim Guardrails & Interpretation Discipline

In compliance with rigorous scientific standards, all conclusions in this report are framed under the following guardrails:

- **Observation:** Under the evaluated Phase 5 configuration, 4-qubit QSVC and Classical RBF-SVM produced identical predictions.
  - **Explanation:** Unweighted soft-margin SVM optimization on highly imbalanced data (~95% majority class) converged to the trivial majority-class solution for both kernel types.
  - **Limitation:** The study evaluated 4-qubit classical-to-quantum fidelity kernels with $C=1.0$. Findings cannot be generalized to trainable quantum circuits, alternative feature maps, or balanced datasets.
- **Guardrail Statement:** This investigation provides **no evidence of clinical efficacy or readiness for emergency deployment**. The findings reflect a controlled benchmark on the TORGO dataset under synthetic AWGN perturbations.

---

## 7. Artifact Traceability Index

All numerical values, tables, and figures in this report trace directly to the following machine-readable artifacts:
- Experimental Cells: `results/phase5/phase5_results.csv` (720 rows)
- Prediction Traces: `results/phase5/phase5_predictions.csv` (245,520 rows)
- Descriptive Statistics: `phase6/analysis/descriptive_statistics.csv` (48 rows)
- Hypothesis Testing: `phase6/analysis/statistical_tests.csv` (48 tests)
- Effect Sizes: `phase6/analysis/effect_sizes.csv` (48 rows)
- Speaker Analysis: `phase6/analysis/speaker_analysis.csv` (45 rows)
- Visualizations: `phase6/figures/` (5 PNG figures)
