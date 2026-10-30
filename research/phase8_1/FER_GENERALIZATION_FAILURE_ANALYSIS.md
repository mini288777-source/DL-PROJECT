# FORENSIC ROOT CAUSE ANALYSIS: OOF-TO-TEST FER GENERALIZATION FAILURE

**Document:** `research/phase8_1/FER_GENERALIZATION_FAILURE_ANALYSIS.md`  
**Phase:** Phase 8.1 Safety Operating-Point Recovery  
**Target:** Diagnose why Out-of-Fold (OOF) calibrated thresholds ($\text{FER}_{\text{OOF}} \le 15\%$) generalized to test FER values of 27.45% (Classical), 26.06% (Quantum), and 25.97% (Matched Shadow).

---

## 1. Forensic Evidence and Systematic Answers to the 12 Mandatory Questions

### Question 1: What objective selected each threshold?
**Evidence & Finding:**  
In `scripts/phase8_80_20/execute_forensic_repair.py`, lines 195–235, the threshold $\tau_s \in [0.45, 0.85]$ (evaluated in step increments of 0.05) for each speaker $s$ was selected using the following constrained objective on 5-fold Stratified Out-of-Fold predictions:
$$\tau_s^* = \arg\max_{\tau \in [0.45, 0.85]} \left\{ \text{Recall}_{\text{OOF}}(\tau) \;\middle|\; \text{FER}_{\text{OOF}}(\tau) \le 0.15 \right\}$$
When multiple thresholds satisfied the constraint with identical recall, the rule selected the highest (most conservative) $\tau$ that satisfied the condition.

---

### Question 2: Was selection based on genuine OOF predictions?
**Evidence & Finding:**  
**Yes.** For each speaker $s$, a 5-fold Stratified K-Fold split was constructed strictly within that speaker's 80% training pool ($S_{\text{train}}$). For each fold $k \in \{1, \dots, 5\}$:
- The Stage 1 gate and residual models were fit strictly on $S_{\text{train}} \setminus F_k$.
- Probabilities $\hat{p}_{\text{OOF}}$ were evaluated strictly on the held-out fold $F_k$.
Zero test samples from the 20% locked test partition were accessed or included during this process.

---

### Question 3: Was each OOF sample excluded from fitting, including duplicate/derived recordings?
**Evidence & Finding:**  
**Yes.** Each sample was strictly held out when predicting its OOF probability. Furthermore, an exhaustive prompt-level audit of all 6,067 active utterances in `results/phase8_80_20/split_manifest.csv` confirmed that **every single utterance is a distinct prompt/recording** (`sample_id` format: `{speaker}_{session}_{mic}_{prompt_id}`). There are exactly 6,067 unique prompt-level recording groups across the corpus; no microphone pairs (e.g., simultaneous headMic and arrayMic of the exact same prompt) crossed folds or partitions.

---

### Question 4: Were folds grouped by original recording/session and stratified appropriately while preserving within-speaker personalization?
**Evidence & Finding:**  
**Partially.** Folds were stratified by binary emergency label within each speaker's training pool to ensure emergency samples were present in every fold. However, folds were **not grouped by recording session**. In TORGO, speakers completed recording sessions across different days or seated sessions (e.g. session `1`, `2`, `3`). Random stratified splitting intermingled utterances from the same session across training and validation folds. While within-speaker personalization was preserved, intra-session acoustic correlation may have contributed to optimistic validation scores.

---

### Question 5: What were OOF FER, recall, precision, and alarm counts at the selected threshold?
**Evidence & Finding:**  
On the 80% training set (4,852 utterances, comprising 237 true emergencies and 4,615 general speech samples across 15 speakers), the aggregate OOF metrics at the selected per-speaker thresholds $\tau_s^*$ were:
- **Total OOF Alarms Triggered:** 581
- **OOF True Positives (TP):** 43 / 237
- **OOF False Positives (FP):** 538 / 4,615
- **Aggregate OOF Recall:** **18.14%** (43 / 237)
- **Aggregate OOF Precision:** **7.40%** (43 / 581)
- **Aggregate OOF FER:** **11.66%** (538 / 4,615) $\le 15.0\%$
At the validation operating point, the training OOF predictions strictly satisfied the $\le 15\%$ constraint ($11.66\%$).

---

### Question 6: How variable were results across folds and speakers?
**Evidence & Finding:**  
**Extreme inter-speaker variability.** The per-speaker test FER under the OOF thresholds revealed massive disparity:
- **Severe Dysarthric Speakers:**
  - `M02`: FER = **74.42%** (32 FPs on 43 general speech samples)
  - `M01`: FER = **46.30%** (25 FPs on 54 general speech samples)
  - `M03`: FER = **34.21%** (26 FPs on 76 general speech samples)
  - `F01`: FER = **27.27%** (6 FPs on 22 general speech samples)
  - `M04`: FER = **23.08%** (12 FPs on 52 general speech samples)
  - `M05`: FER = **20.59%** (14 FPs on 68 general speech samples)
- **Control Speakers:**
  - `MC04`: FER = **61.90%** (117 FPs on 189 general speech samples — acoustic anomaly)
  - `MC01`: FER = **25.93%** (21 FPs on 81 general speech samples)
  - `MC03`: FER = **15.65%** (18 FPs on 115 general speech samples)
  - `FC01`: FER = **10.71%** (3 FPs on 28 general speech samples)
  - `FC02`: FER = **3.74%** (7 FPs on 187 general speech samples)
  - `FC03`: FER = **1.27%** (1 FP on 79 general speech samples)

Two speakers (`MC04` with 117 FPs and `M02` with 32 FPs) generated **149 out of the 301 total test false alarms (49.5%)**.

---

### Question 7: Was the threshold global or speaker-/track-specific?
**Evidence & Finding:**  
The thresholds were **speaker-specific** (personalized), calibrated per speaker across the 15 speakers, ranging from $\tau = 0.45$ to $\tau = 0.70$ (mean $\bar{\tau} = 0.583$). They were track-specific in intent, but Track B's OOF thresholds were applied across the evaluation comparison.

---

### Question 8: Are class prevalence or score distributions different across OOF folds and the previously examined test set?
**Evidence & Finding:**  
- **Prevalence:** Class prevalence was virtually identical:
  - Training / OOF Emergency Prevalence: **4.88%** (237 / 4,852)
  - Test Emergency Prevalence: **4.94%** (60 / 1,215)
  - Difference: $+0.06\%$ (statistically insignificant, $p = 0.94$).
- **Score Distributions on Test Set:**
  - Test Emergency Scores: Mean = **0.4761**, Median = **0.4844**, Std = **0.2309**
  - Test General Speech Scores: Mean = **0.4557**, Median = **0.4573**, Std = **0.2018**
  - **Two-Sample Kolmogorov-Smirnov Test:**
    $$\text{KS statistic} = 0.1175, \quad p\text{-value} = \mathbf{0.3801} > 0.05$$
  - **Quantitative Conclusion:** The emergency and general speech score distributions on the locked test set are **statistically indistinguishable**. The model achieves virtually zero margin of score separation between true emergencies and non-emergency speech.

---

### Question 9: Were preprocessing and calibration independent of test data?
**Evidence & Finding:**  
**Yes.** All Whisper feature extractions were cached per-sample. PCA (384 $\to$ 8) was fit strictly on training samples for each fold/speaker. Logistic regression weights, gate parameters, and VQC residual parameters were computed strictly without test labels or features.

---

### Question 10: Were metrics computed from each track's final predictions using canonical definitions?
**Evidence & Finding:**  
**Yes.** As audited in Phase 8 forensic repair:
- $\text{Emergency Recall} = \text{TP} / (\text{TP} + \text{FN})$
- $\text{Emergency Precision} = \text{TP} / (\text{TP} + \text{FP})$
- $\text{FER} = \text{FP} / (\text{FP} + \text{TN}) = \text{FP} / \text{Total General Speech}$
All counts reconcile with zero discrepancies.

---

### Question 11: Is the OOF-to-test gap consistent with a measurable split, speaker, session, prevalence, or score-distribution shift?
**Evidence & Finding:**  
**Yes. The primary driver is weak discriminative score separation combined with threshold selection fragility in small-sample regimes:**
1. **Severe Emergency Sparsity During OOF Calibration:**  
   Across the 15 speakers, the median number of emergency training samples per speaker was only **12** (ranging from 7 in F01 to 38 in MC04). In a 5-fold split, each validation fold contained only **1 to 3 emergency samples**!
2. **Fragile Threshold Selection:**  
   To achieve non-zero recall on 1 or 2 emergency samples in a fold, the threshold grid search was forced to select low thresholds ($\tau \approx 0.45 - 0.55$).
3. **General Speech Score Leakage Above Threshold:**  
   Because general speech scores have a mean of 0.456 and standard deviation of 0.202, setting $\tau = 0.55$ causes approximately $1 - \Phi\left(\frac{0.55 - 0.456}{0.202}\right) \approx 32\%$ of general speech utterances to fall above the threshold.
4. **Generalization Breakdown on Test Set:**  
   On the test set, where 1,155 general speech samples were evaluated, this 25–30% tail of general speech scores produced 301 false alarms ($\text{FER} = 26.06\%$), far exceeding the 15% threshold.

---

### Question 12: Is there a feasible validation operating point, and what is its recall/precision trade-off?
**Evidence & Finding:**  
- If thresholds are raised to $\tau \ge 0.75 - 0.80$ to force $\text{FER} \le 15\%$ on test data:
  - Emergency Recall collapses from **33.33%** (20/60) to **11.67%** (7/60) or lower.
  - For 8 out of 15 speakers, emergency recall drops to exactly **0.0%**.
  - Emergency precision remains under **10%** due to the 95:5 class imbalance.
- **Feasibility Verdict:**  
  Under the current single-stage gate + residual architecture on 384-D Whisper embeddings, **no single operating point exists that simultaneously satisfies $\text{FER} \le 15\%$ and achieves clinically meaningful emergency recall ($> 50\%$)**.

---

## 2. Quantitative Summary Table: Root Cause Diagnosis

| Diagnostic Domain | Metric / Property | Observed Value | Forensic Interpretation |
|:---|:---|:---:|:---|
| **Class Imbalance** | Emergency Prevalence (Train vs Test) | 4.88% vs 4.94% | Identical prevalence; no prevalence shift. |
| **Utterance Duplication** | Overlapping prompt groups across splits | **0 / 6,067** | Zero paired mic leakage; clean partition. |
| **Score Separation** | Emergency Score Mean vs General Speech Mean | 0.4761 vs 0.4557 | $\Delta = +0.0204$ (negligible margin). |
| **Statistical Divergence**| Two-Sample Kolmogorov-Smirnov Test | $D = 0.1175, p = 0.3801$ | Distributions are statistically indistinguishable ($p > 0.05$). |
| **Speaker Concentration**| False Alarms in Top 2 Speakers (MC04, M02) | 149 / 301 (49.5%) | Extreme speaker skew; dysarthria degrades specificity. |
| **OOF Sample Size** | Median Emergencies per Speaker Fold | **2.4 samples** | High variance; thresholds overfitted to sparse events. |

---

## 3. Protocol Implications for Phase 8.1

1. **Protocol Freeze:** Do not tune thresholds against the locked 80/20 test set.
2. **Architecture / Discrimination Interventions Required:** Threshold tuning alone cannot solve a zero-margin score distribution problem. Interventions must focus on improving the **representation and discriminative separation** between emergency acoustic tokens and general phonemes.
3. **Safety Gate Realism:** If discrimination cannot be improved within the permitted training data, the scientific verdict must honestly report **`SAFETY-INFEASIBLE`**.
