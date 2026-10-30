# FINAL AUTHORITATIVE METRIC DEFINITIONS & FORMULAS
## Mathematical Formulations, Implementation Contracts & Provenance

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Scope:** Canonical Reference for All Quantitative Evaluation Metrics  
**Status:** COMPLETE & FROZEN  

---

## 1. Overview & Conventions

All evaluation metrics are computed across **60 matched experimental units** ($15$ TORGO speakers $\times 4$ calibration budgets $N \in \{20, 30, 40, 50\}$) evaluated on prompt-isolated Track B test partitions.

### Standard Conventions:
- **Zero-Division Handling:** Where precision or recall has a zero denominator (e.g. no true positives or no positive predictions), standard scikit-learn convention is strictly enforced: `zero_division=0.0`.
- **Prevalence Baseline:** In Track B test partitions, `GENERAL_SPEECH` constitutes approximately **$95.10\%$** of all utterances, while emergency utterances represent only **$\approx 4.90\%$**.
- **Experimental Unit:** Metrics are computed per condition $(s, N)$, then averaged across the cohort ($N=60$). Individual utterance predictions are not treated as independent statistical samples.

---

## 2. Mathematical Metric Definitions

### 2.1 End-to-End Emergency Recall
- **Definition:** The proportion of ground-truth emergency utterances correctly classified as emergency intents by the end-to-end pipeline.
- **Formula:**
  $$\text{Emergency Recall} = \frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FN_{\text{emerg}}} = \frac{\sum_{i \in \mathcal{E}} \mathbb{I}(\hat{y}_i \in \mathcal{E})}{|\mathcal{E}|}$$
  where $\mathcal{E} = \{\text{SOS\_HELP}, \text{FALL}, \text{BREATHING\_MEDICAL}, \text{CRITICAL\_RESPONSE}, \text{LOCATION}, \text{PAIN}\}$.
- **Authoritative Values:**
  - QSVC Q8-R1: **$27.52\% \pm 41.94\%$** (median: $27.27\%$)
  - Classical RBF-SVM $d=8$: **$31.89\% \pm 40.97\%$**
  - Classical Adapter $d=8$: **$45.81\% \pm 29.78\%$** (median: $48.05\%$)

---

### 2.2 End-to-End Emergency Precision vs. Stage-1 Precision
A critical reconciliation established in Phase 6.5.6 is distinguishing Stage-1 Precision from End-to-End Emergency Precision:

- **End-to-End Emergency Precision:**
  $$\text{Emergency Precision}_{\text{e2e}} = \frac{TP_{\text{emerg}}}{TP_{\text{emerg}} + FP_{\text{emerg}}} = \frac{\sum_{i} \mathbb{I}(y_i \in \mathcal{E} \land \hat{y}_i \in \mathcal{E})}{\sum_{i} \mathbb{I}(\hat{y}_i \in \mathcal{E})}$$
  Evaluates whether an utterance predicted as an emergency by the complete two-stage pipeline was truly an emergency.
- **Authoritative Closure Values:**
  - QSVC Q8-R1: **$0.81\% \pm 1.31\%$**
  - Classical RBF-SVM $d=8$: **$2.02\% \pm 6.57\%$**
  - Classical Adapter $d=8$: **$2.03\% \pm 1.45\%$**
  - QSVC Q4-R1: **$1.75\% \pm 4.68\%$**
  - QSVC Q8-R2: **$0.66\% \pm 1.16\%$**
- **Forensic Resolution of Historical Values ($11.24\%$ and $9.21\%$):**
  The preliminary provisional values ($11.24\%$ for QSVC and $9.21\%$ for Adapter) were transcription artifacts from exploratory balanced-subset scripts. The authoritative empirical values across all 60 conditions with natural test prevalence ($95.10\%$ negative) are **$0.81\%$** and **$2.03\%$**.

---

### 2.3 False-Emergency Rate (FER) & Stage-1 False Positive Rate (FPR)
- **Definition:** The proportion of non-emergency (`GENERAL_SPEECH`) utterances incorrectly routed into the emergency triage channel.
- **Formula:**
  $$\text{FER} = \frac{FP_{\text{emerg}}}{FP_{\text{emerg}} + TN_{\text{gen}}} = \frac{\sum_{i \in \mathcal{G}} \mathbb{I}(\hat{y}_i \in \mathcal{E})}{|\mathcal{G}|} = 1.0 - \text{Specificity}$$
  where $\mathcal{G} = \{\text{GENERAL\_SPEECH}\}$.
- **Authoritative Values:**
  - QSVC Q8-R1: **$28.12\% \pm 41.86\%$**
  - Classical RBF-SVM $d=8$: **$29.32\% \pm 38.07\%$**
  - Classical Adapter $d=8$: **$49.82\% \pm 22.98\%$**

---

### 2.4 Intent-Preserving Emergency Recall (IPER)
- **Definition:** The proportion of true emergency utterances that are not only detected as emergencies, but also correctly routed to their exact emergency subtype in Stage 2.
- **Formula:**
  $$\text{IPER} = \frac{\sum_{i \in \mathcal{E}} \mathbb{I}(\hat{y}_i = y_i)}{|\mathcal{E}|}$$
- **Authoritative Values:**
  - QSVC Q8-R1: **$8.25\% \pm 13.61\%$**
  - Classical RBF-SVM $d=8$: **$11.69\% \pm 17.14\%$**
  - Classical Adapter $d=8$: **$14.21\% \pm 14.17\%$**

---

### 2.5 End-to-End Macro-Averaged F1 Score
- **Definition:** The unweighted mean of F1 scores across all 7 canonical classes in the taxonomy:
- **Formula:**
  $$\text{Macro-F1} = \frac{1}{7} \sum_{c=1}^7 \text{F1}_c, \quad \text{where } \text{F1}_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
- **Authoritative Values:**
  - QSVC Q8-R1: **$0.1051 \pm 0.0575$**
  - Classical RBF-SVM $d=8$: **$0.1087 \pm 0.0487$**
  - Classical Adapter $d=8$: **$0.0949 \pm 0.0342$**
  - Primary Paired Difference (QSVC vs. RBF): $\Delta = -0.0036$, $p = 0.3428$ (No statistical difference).

---

### 2.6 Stage-2 Conditional Intent Metrics
Evaluated strictly on the subset of utterances routed through the Stage-1 gate:
- **Stage-2 Feasible Macro-F1:** Macro-F1 computed across emergency classes that were actually present in the speaker's calibration set:
  - QSVC Q8-R1: **$0.1252 \pm 0.0844$** (pooled condition mean: $0.1982$)
  - Classical RBF-SVM $d=8$: **$0.1962 \pm 0.1091$** (pooled condition mean: $0.1993$)
  - Classical Adapter $d=8$: **$0.1824 \pm 0.1220$**
- **Stage-2 Balanced Accuracy:** Macro-averaged recall across feasible emergency classes:
  - QSVC Q8-R1: **$32.53\% \pm 23.25\%$**
  - Classical RBF-SVM $d=8$: **$33.91\% \pm 18.19\%$**
  - Classical Adapter $d=8$: **$33.57\% \pm 19.60\%$**

---

### 2.7 PCA Cumulative Explained Variance
- **Definition:** The fraction of total embedding variance retained by the top $d$ principal components fitted on per-speaker calibration data:
  $$\text{Var}(d) = \frac{\sum_{k=1}^d \lambda_k}{\sum_{j=1}^{384} \lambda_j}$$
- **Authoritative Values:**
  - $d=2$: $69.86\% \pm 5.12\%$
  - $d=4$: **$80.66\% \pm 4.37\%$** (Historical $27.4\%$ was a plotting script artifact)
  - $d=8$: **$89.73\% \pm 2.99\%$** (Selected quantum candidate)
  - $d=16$: $96.48\% \pm 1.45\%$
  - $d=32$: $99.58\% \pm 0.32\%$

---

### 2.8 Computational Runtime & Latency Metrics
- **Per-Evaluation Inference Latency:** Mean wall-clock time required to predict a single test utterance:
  - QSVC Q8-R1: **$1.0816 \text{ s}$ per evaluation**
  - Classical RBF-SVM $d=8$: **$0.0014 \text{ s}$ per evaluation**
  - Ratio: **$772.57\times$ slower**
- **Total 60-Condition Experiment Wall-Clock Runtime:** Cumulative wall-clock time to run training and evaluation across all 60 conditions:
  - QSVC Q8-R1: **$70.22 \text{ s}$**
  - Classical RBF-SVM $d=8$: **$0.33 \text{ s}$**
  - Ratio: **$212.79\times$ slower**
- **Depth Scaling (Q8-R2):**
  - Per-Evaluation: **$1.9761 \text{ s}$** ($1408.8\times$ vs. RBF)
  - Total Wall Time: **$128.04 \text{ s}$** ($388.0\times$ vs. RBF)
