# PHASE 7.10 — CONTINUAL HYBRID CLASSICAL + QUANTUM PERSONALIZATION
## RESEARCH REPORT AND EMPIRICAL EVALUATION

**Project**: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Subphase**: Phase 7.10 — Hybrid Classical + Quantum Personalization  
**Status**: COMPLETE  
**Primary Research Question**: RQ7.8  
**Verdict**: **PARTIALLY SUPPORTED**  
**Frozen Baseline Conclusion (CASE B)**: **UNCHANGED**  

---

> [!IMPORTANT]
> **Frozen Historical Boundary**:
> Phase 7.10 tests hybrid classical + quantum personalization architectures and does NOT modify the frozen Phase 1–6.5 conclusions.
> The historical CASE B conclusion (*No Statistically Supported Quantum Advantage*) remains completely unchanged.
> All experimental results from Phase 1–6, Phase 6.5, Phase 7.4, Phase 7.5, Phase 7.6, Phase 7.7, Phase 7.8, and Phase 7.9 are frozen and preserved.

---

## 1. Executive Summary & Core Findings

Phase 7.10 executes a rigorous, pre-registered empirical evaluation of **continual hybrid classical + quantum personalization** across 60 matched experimental cells (15 TORGO speakers $\times$ 4 experience budgets $N \in \{20, 30, 40, 50\}$).

The primary research question asks whether fusing classical neural representations (Phase 7.4 continual adapter) with adaptive quantum variational circuits (Phase 7.9 trainable quantum feature map) achieves benefits that neither model family demonstrated in isolation.

### Key Empirical Findings:
1. **Hybrid vs. Continual Classical**:
   - **Macro-F1**: Improved from $0.1012$ (Classical) to $0.1184$ (Hybrid), a statistically significant gain of $\Delta = +0.0172$ ($d = +0.61$, $p_{\text{adj}} = 3.78 \times 10^{-7}$).
   - **Overall Accuracy**: Increased dramatically from $55.23\%$ to $78.44\%$ ($\Delta = +0.2321$, $d = +1.06$, $p_{\text{adj}} = 3.74 \times 10^{-8}$).
   - **False Emergency Rate (FER)**: Reduced by more than half, falling from $43.53\%$ to $19.45\%$ ($\Delta = -0.2407$, $d = -1.06$, $p_{\text{adj}} = 7.28 \times 10^{-8}$).
   - **Emergency Recall**: Regressed from $16.20\%$ to $7.07\%$ ($\Delta = -0.0913$, $d = -0.57$, $p_{\text{adj}} = 7.83 \times 10^{-4}$). The classical baseline retains superior emergency recall because the quantum branch heavily predicts general speech.

2. **Hybrid vs. Continual Trainable Quantum**:
   - **Macro-F1**: $0.1184$ (Hybrid) vs. $0.1187$ (Trainable Quantum) ($\Delta = -0.0004$, $p_{\text{adj}} = 0.275$, not statistically significant).
   - **Emergency Recall**: Directional increase from $5.81\%$ to $7.07\%$ ($\Delta = +0.0127$, $p_{\text{adj}} = 0.250$, not statistically significant).
   - **False Emergency Rate**: Increased slightly from $17.20\%$ to $19.45\%$ ($\Delta = +0.0226$, $p_{\text{adj}} = 4.02 \times 10^{-4}$).
   - **Accuracy**: $78.44\%$ vs. $80.63\%$ ($\Delta = -0.0219$, $p_{\text{adj}} = 0.0059$).

3. **Subgroup Heterogeneity**:
   - **Dysarthric Speakers (n=32 cells)**: Emergency recall for dysarthric speakers remains severely depressed ($3.46\%$ Hybrid vs. $13.95\%$ Classical, and $2.68\%$ Trainable Quantum).
   - **Control Speakers (n=28 cells)**: Emergency recall reached $11.20\%$ (vs. $9.38\%$ Trainable Quantum, but below Classical at $18.78\%$).

4. **Verdict**: **PARTIALLY SUPPORTED**.
   Hybrid fusion successfully transfers the high general accuracy and low false-emergency rate of quantum classifiers while retaining classical representation capacity, achieving significant F1 and FER superiority over pure classical adaptation. However, it fails to outperform trainable quantum on Macro-F1 and fails to recover the emergency recall of the classical baseline.

---

## 2. Research Questions and Hypotheses

### Primary Research Question (RQ7.8)
*Can hybrid classical + quantum personalization provide statistically supported and safety-acceptable benefit beyond matched continual classical, fixed quantum, and trainable quantum baselines?*

### Formal Hypotheses
- **Hypothesis $H_{7.8a}$**: Hybrid fusion achieves higher Macro-F1 and lower False Emergency Rate than the continual classical baseline under matched experience streams.
  - **Result**: **SUPPORTED** ($p_{\text{adj}} < 10^{-6}$ for both).
- **Hypothesis $H_{7.8b}$**: Hybrid fusion produces higher Emergency Recall and Macro-F1 than both standalone quantum baselines (Fixed and Trainable).
  - **Result**: **NOT SUPPORTED** (Macro-F1 difference is non-significant; Emergency Recall gain is directional $+1.27\%$ but not statistically significant, $p_{\text{adj}} = 0.250$).
- **Overall RQ7.8 Verdict**: **PARTIALLY SUPPORTED**.

---

## 3. Canonical Hybrid Architecture (SSOT)

To guarantee exact scientific replicability without hyperparameter search leakage, the hybrid architecture was frozen prior to benchmark execution:

```text
                     Whisper Embedding (384-D)
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   Classical Branch                               Quantum Branch
──────────────────────────────             ──────────────────────────────
2-Layer Adapter MLP (25,223 params)         Train-only PCA (384 -> 8-D)
384 -> 64 -> 10 classes                     MinMaxScaler [0, π]
AdamW (lr=1e-3, 20 epochs)                 8-Qubit VQFM (Depth 37, 16 params)
Softmax Output: P_class(y|x)                COBYLA KTA Optimization
                                           QSVC (C=1.0) Decision Margins
                                           Softmax Calibration: P_quant(y|x)
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
                 Convex Probability Fusion (α = 0.5)
                   P_hybrid = 0.5 * P_class + 0.5 * P_quant
                                 │
                                 ▼
                     Phase 7.6 Governance Gate
             (Protected Validation + Multi-Metric + Safety Veto)
                                 │
                                 ▼
                     PROMOTE / REJECT / HOLD
```

### Architecture Specifications
- **Architecture Identifier**: `hybrid_adapter_vqfm_calibrated_fusion`
- **Classical Branch**: `classical_adapter_mlp_25223` (25,223 trainable weights)
- **Quantum Branch**: `vqfm_q8_p16_linear_reuploading` (16 variational rotation angles)
- **Total Trainable Parameters**: 25,239
- **Fusion Formulation**: Convex probability fusion:
  $$P_{\text{hybrid}}(y = c \mid x) = \alpha P_{\text{class}}(y = c \mid x) + (1 - \alpha) P_{\text{quant}}(y = c \mid x)$$
  with pre-registered $\alpha = 0.5$.
- **Quantum Marginal Calibration**: Scikit-learn `SVC(decision_function_shape='ovr')` distance margins $\mathbf{z}_{\text{quant}}$ are converted to calibrated probabilities via temperature-normalized softmax ($\tau = 1.0$), avoiding small-sample Cross-Validation breakdown.

---

## 4. Matched Four-Way Experimental Comparison

All 60 cells were evaluated across the 4 model families under identical conditions:
- Same 15 TORGO speakers (8 dysarthric, 7 control).
- Same 4 experience budgets ($N \in \{20, 30, 40, 50\}$).
- Same replay buffer protocol ($M=20$, balanced sampling).
- Same held-out evaluation test partitions.

### Comprehensive Cohort Performance (N = 60 Cells)

| Metric | Continual Classical | Fixed Quantum (Phase 7.8) | Trainable Quantum (Phase 7.9) | Hybrid Classical-Quantum (Phase 7.10) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.1012 ± 0.0351 | 0.1179 ± 0.0335 | 0.1187 ± 0.0335 | **0.1184 ± 0.0337** |
| **Emergency Recall** | **0.1620 ± 0.1989** | 0.0510 ± 0.1084 | 0.0581 ± 0.1194 | **0.0707 ± 0.1348** |
| **Emergency Precision** | 0.0312 ± 0.0468 | 0.0387 ± 0.0812 | 0.0441 ± 0.0917 | **0.0401 ± 0.0825** |
| **False Emergency Rate** | 0.4353 ± 0.2312 | **0.1674 ± 0.2014** | **0.1720 ± 0.2058** | **0.1945 ± 0.2241** |
| **Overall Accuracy** | 55.23% ± 18.2% | **81.07% ± 16.4%** | **80.63% ± 16.8%** | **78.44% ± 17.5%** |
| **Trainable Parameters** | 25,223 | 0 (Kernel static) | 16 (Variational) | **25,239** |

---

## 5. Statistical Hypothesis Testing

Statistical significance was evaluated using matched Wilcoxon signed-rank tests across all 60 paired cells, with Holm-Bonferroni correction applied across the family of hypotheses:

### 1. Hybrid vs. Continual Classical Baseline
| Metric | Mean Hybrid | Mean Classical | Delta (Δ) | Cohen's d | Wilcoxon W | Raw p-value | Adjusted p-value | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.1184 | 0.1012 | +0.0172 | +0.61 | 190.0 | $9.44 \times 10^{-8}$ | $3.78 \times 10^{-7}$ | **SIGNIFICANT (***)** |
| **Emergency Recall** | 0.0707 | 0.1620 | -0.0913 | -0.57 | 29.0 | $1.96 \times 10^{-4}$ | $7.83 \times 10^{-4}$ | **REGRESSION (***)** |
| **False Emergency Rate** | 0.1945 | 0.4353 | -0.2407 | -1.06 | 108.0 | $1.82 \times 10^{-8}$ | $7.28 \times 10^{-8}$ | **SIGNIFICANT (***)** |
| **Accuracy** | 0.7844 | 0.5523 | +0.2321 | +1.06 | 135.0 | $9.35 \times 10^{-9}$ | $3.74 \times 10^{-8}$ | **SIGNIFICANT (***)** |

### 2. Hybrid vs. Trainable Quantum Baseline
| Metric | Mean Hybrid | Mean Trainable Q | Delta (Δ) | Cohen's d | Wilcoxon W | Raw p-value | Adjusted p-value | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.1184 | 0.1187 | -0.0004 | -0.05 | 553.0 | 0.0688 | 0.2751 | **Not Significant** |
| **Emergency Recall** | 0.0707 | 0.0581 | +0.0127 | +0.24 | 3.0 | 0.0625 | 0.2500 | **Not Significant** |
| **False Emergency Rate** | 0.1945 | 0.1720 | +0.0226 | +0.36 | 70.0 | $1.00 \times 10^{-4}$ | $4.02 \times 10^{-4}$ | **Higher FER (*)** |
| **Accuracy** | 0.7844 | 0.8063 | -0.0219 | -0.36 | 308.5 | 0.0015 | 0.0059 | **Slightly Lower (*)** |

---

## 6. Subgroup Analysis: Dysarthric vs. Control Cohorts

| Subgroup | Cells | Metric | Classical Baseline | Trainable Quantum | Hybrid | Δ(Hyb - Classical) | Δ(Hyb - Trainable Q) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dysarthric** | 32 | Macro-F1 | 0.1164 | 0.1322 | **0.1320** | +0.0155 | -0.0003 |
| | | Emerg Recall | 0.1395 | 0.0268 | **0.0346** | -0.1049 | +0.0078 |
| | | FER | 0.3333 | 0.0721 | **0.0853** | -0.2480 | +0.0132 |
| **Control** | 28 | Macro-F1 | 0.0837 | 0.1033 | **0.1029** | +0.0191 | -0.0005 |
| | | Emerg Recall | 0.1878 | 0.0938 | **0.1120** | -0.0758 | +0.0182 |
| | | FER | 0.5517 | 0.2861 | **0.3194** | -0.2324 | +0.0333 |

### Interpretation
- **Dysarthric Vulnerability**: In atypical speech, dysarthria alters acoustic realizations heavily. Classical models overfit to emergency classes leading to high recall but catastrophic false alarms (FER 33.3%). Quantum circuits compress features into an 8-D subspace that separates general speech exceptionally well (FER 7.2%) but sacrifices emergency sensitivity (recall 2.7%). Hybrid fusion inherits the general speech conservatism of the quantum branch, suppressing false emergencies (FER 8.5%) but yielding inadequate emergency recall (3.5%).
- **Control Performance**: On control speakers, hybrid fusion maintains higher emergency recall (11.20%) than trainable quantum (9.38%) while slashing classical false emergencies by 23.2 percentage points.

---

## 7. Model Governance and Validation Gate Integration

Every hybrid model candidate was evaluated by the Phase 7.6 `ValidationGate` prior to potential activation:
- **Promotion Rule**: Candidate accepted if and only if:
  1. No safety veto triggered ($\text{FER} \le 0.50$, $\Delta \text{Emergency Recall} \ge 0$).
  2. Macro-F1 delta $\ge -0.01$.
  3. Contamination firewall passes (zero overlap with training/replay).
- **Outcomes Across 60 Cells**:
  - **Promoted**: 36 cells (60.0%) where hybrid candidate maintained or improved parent metrics while staying within safety bounds.
  - **Rejected**: 24 cells (40.0%) due to Safety Veto (e.g. FER exceeding tolerance on speakers FC02, MC04, or Emergency Recall drops).
  - **Direct Activation**: 0 (zero unvalidated model activations permitted).

---

## 8. Artifact Inventory & Provenance

The following phase-specific artifacts were created and registered:
- `results/continual/phase7_10/continual_hybrid_results.csv`: Complete cell-level metrics across all 60 experiments.
- `results/continual/phase7_10/matched_four_way_comparison.csv`: Paired cell-by-cell table comparing Classical, Fixed Quantum, Trainable Quantum, and Hybrid.
- `results/continual/phase7_10/hybrid_trajectory_summary.csv`: Longitudinal trajectory across experience budgets $N \in \{20, 30, 40, 50\}$.
- `results/continual/phase7_10/hybrid_subgroup_analysis.csv`: Subgroup breakdowns for Dysarthric vs. Control cohorts.
- `results/continual/phase7_10/experiment_registry.json`: Full experiment lineage metadata with dataset hashes and hyperparameter records.
- `reports/continual/phase7_10/continual_hybrid_summary.json`: Serialized statistical tests and summary metrics.

---

## 9. Phase 7.10 Conclusion & Gate Exit

1. **Scientific Verdict**: **PARTIALLY SUPPORTED**.
   Hybrid personalization significantly outperforms classical personalization in Macro-F1, Accuracy, and False Emergency Rate. However, it does not achieve superior Macro-F1 over trainable quantum, and classical personalization remains superior on Emergency Recall.
2. **Safety Verdict**: Governed candidates pass safety verification, but unconstrained hybrid deployment is not recommended due to emergency recall degradation on dysarthric speakers.
3. **Frozen Baseline Conclusion (CASE B)**: **UNCHANGED**.
   Historical findings stand immutable.
4. **Transition**: Phase 7.10 is closed. Moving immediately and autonomously to Phase 7.11 (Self-Improvement Evaluation).
