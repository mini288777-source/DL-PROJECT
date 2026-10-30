# PHASE 7.9 — CONTINUAL TRAINABLE / ADAPTIVE QUANTUM REPRESENTATION
## RESEARCH REPORT AND EMPIRICAL EVALUATION

**Project**: Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Subphase**: Phase 7.9 — Trainable / Adaptive Quantum Representation  
**Status**: COMPLETE  
**Primary Research Question**: RQ7.7  
**Decision**: **NOT SUPPORTED**  
**Frozen Baseline Conclusion (CASE B)**: **UNCHANGED**  

---

> [!IMPORTANT]
> **Frozen Historical Boundary**:
> Phase 7.9 tests adaptive quantum representation learning and does not modify the frozen Phase 1–6.5 conclusions.
> The historical CASE B conclusion (*No Statistically Supported Quantum Advantage*) remains completely unchanged.
> All experimental results from Phase 1–6, Phase 6.5, Phase 7.4, Phase 7.5, Phase 7.6, Phase 7.7, and Phase 7.8 are frozen and preserved.

---

## 1. Motivation from Phase 7.8

In Phase 7.8, continual personalization was evaluated using a fixed quantum kernel (Q8-R1 QSVC with $C=1.0$) across 60 matched cells (15 TORGO speakers $\times$ 4 experience budgets $N \in \{20, 30, 40, 50\}$). While the fixed quantum kernel achieved higher overall accuracy (81.07% vs. 55.23%) and lower False Emergency Rate (16.74% vs. 43.53%) than the continual classical baseline, it demonstrated a catastrophic safety failure:
- Emergency recall declined from 16.20% (classical baseline) to 5.10% (fixed quantum).
- For dysarthric speakers, emergency recall collapsed to **0.00%** for all budgets $N \ge 30$.
- The hypothesis of quantum personalization benefit was **NOT SUPPORTED** (RQ7.6 = NOT SUPPORTED).

The central open question leading into Phase 7.9 was whether the observed failure of the fixed quantum kernel was caused by the rigidity of its static feature map $|\psi(x)\rangle$. If atypical speech patterns shift across sessions or deviate significantly from control speech, a fixed geometric embedding space cannot adapt. Phase 7.9 tests the hypothesis that **adapting the quantum representation itself** via parameterized variational gates ($\theta$) can overcome this limitation under matched continual learning conditions.

---

## 2. Research Questions and Hypotheses

### Primary Research Question (RQ7.7)
*Does a trainable/adaptive quantum representation provide measurable personalization benefit beyond the fixed quantum-kernel formulation under matched continual-learning conditions?*

### Secondary Research Questions
1. Does quantum representation adaptation improve safety-critical emergency recognition for atypical speech without unacceptable false emergency behavior or calibration degradation?
2. Does representation adaptation mitigate catastrophic forgetting or worsen stability across continual experience accumulation?
3. What is the computational and simulator overhead of optimizing quantum variational circuits during online personalization?

### Formal Hypotheses
- **Hypothesis $H_{7.7}$ (Primary)**: Optimization of variational parameters $\theta$ in a parameterized quantum feature map $|\psi(x; \theta)\rangle$ produces a statistically significant improvement in Macro-F1 and Emergency Recall over the matched fixed quantum kernel ($|\psi(x)\rangle$).
- **Null Hypothesis $H_{0, 7.7}$**: Continual representation adaptation yields no statistically significant improvement over the fixed quantum kernel ($\Delta \le 0$ or $p_{\text{adj}} \ge 0.05$).

---

## 3. Frozen Historical Boundary

The following components remain permanently frozen and were not modified during Phase 7.9:
1. **Phases 1–6 & Phase 6.5**: Whisper feature extraction (384-D), recovery partitions, 10-class intent taxonomy.
2. **Phase 7.4 Baseline**: Continual classical adapter models (`EXP_7_4_*`).
3. **Phase 7.5 Baseline**: Replay buffer configurations and memory limits.
4. **Phase 7.6 Governance**: Multi-metric `ValidationGate`, rollback protocol, and `SafetyVeto`.
5. **Phase 7.7 Baselines**: Drift detection thresholds and temperature recalibration.
6. **Phase 7.8 Baselines**: Fixed-kernel QSVC models (`EXP_7_8_*`).
7. **Cohort Definition**: 15 TORGO speakers (8 dysarthric: F01, F03, F04, M01, M02, M03, M04, M05; 7 control: FC01, FC02, FC03, MC01, MC02, MC03, MC04).
8. **Experience Budgets**: $N \in \{20, 30, 40, 50\}$.

---

## 4. Trainable Quantum Architecture

To isolate the scientific intervention to **representation adaptation** without confounding architecture changes, the trainable quantum feature map directly parameterizes the canonical 8-qubit feature map from Phase 7.8:

```
Whisper Embedding (384-D)
        ↓
Train-Only PCA (d=8) [Fitted strictly on available training experiences]
        ↓
MinMaxScaler [0, π]
        ↓
Data Re-uploading Parameterized Quantum Feature Map |ψ(x; θ)>
        ├─ Layer 1: Data Encoding U_enc(x) (Hadamard + Rz(x_i) + linear CX + Rz(x_i x_j))
        ├─ Layer 2: Trainable Rotation Layer Ry(θ_0 .. θ_7)
        ├─ Layer 3: Linear Entanglement (CX 0-1, 1-2, ..., 6-7)
        ├─ Layer 4: Data Re-uploading U_enc(x)
        └─ Layer 5: Trainable Rotation Layer Ry(θ_8 .. θ_15)
        ↓
Parameterized Quantum Kernel K_θ(x_i, x_j) = |<ψ(x_i; θ)|ψ(x_j; θ)>|^2
        ↓
Balanced QSVC (C=1.0)
```

### Architecture Specifications (SSOT)
- **Architecture Identifier**: `vqfm_q8_p16_linear_reuploading`
- **Qubit Count**: 8
- **Input Feature Dimension**: 8
- **Trainable Parameters**: 16 ($2 \times 8$ single-qubit $R_y$ rotation angles $\theta \in \mathbb{R}^{16}$)
- **Circuit Depth**: 37
- **Gate Counts**:
  - Hadamard ($H$): 16
  - Parameterized $R_z(x)$: 30
  - Trainable $R_y(\theta)$: 16
  - Entangling CNOT ($CX$): 35
  - Total Gates: 97
- **Interleaved Data Re-uploading**: Crucial architectural design. Pure variational layers at the end of a kernel circuit cancel identically due to unitary invariance ($U(\theta)^\dagger U(\theta) = I$). Interleaving data encoding between trainable layers ensures non-trivial kernel adaptation ($\Delta K_\theta \ne 0$).

---

## 5. Parameterization and Optimization Method

### Parameterized Representation
$$|\psi(x; \theta)\rangle = U_{\text{var}, 2}(\theta^{(2)}) \cdot U_{\text{enc}}(x) \cdot U_{\text{ent}} \cdot U_{\text{var}, 1}(\theta^{(1)}) \cdot U_{\text{enc}}(x) |0^{\otimes 8}\rangle$$
where $\theta = [\theta^{(1)}, \theta^{(2)}] \in \mathbb{R}^{16}$.

### Kernel Target Alignment (KTA) Objective
Variational parameters are adapted using Kernel Target Alignment:
$$\text{KTA}(K_\theta, Y) = \frac{\langle K_\theta, Y \rangle_F}{\|K_\theta\|_F \|Y\|_F}$$
where:
- $K_\theta \in \mathbb{R}^{M \times M}$ is the fidelity kernel matrix evaluated on adaptation experiences.
- $Y_{ij} = \frac{1}{\sqrt{N_{y_i} N_{y_j}}}$ if $y_i = y_j$, else $0$ (class-balanced label target matrix to prevent majority-class collapse).
- Regularized Loss: $\mathcal{L}(\theta) = -\text{KTA}(K_\theta, Y) + \lambda \|\theta\|_2^2$, with $\lambda = 10^{-4}$.

### Optimization Policy
- **Optimizer**: COBYLA (Constrained Optimization BY Linear Approximation)
- **Max Iterations**: 20 evaluations
- **Initial Step Size (`rhobeg`)**: 0.2
- **Gradient Method**: Derivative-free trust-region simplex (exact statevector simulation via `qiskit.quantum_info.Statevector`)
- **Optimization Time**: ~0.72 seconds per update cell.

---

## 6. Initialization and Continual Update Policy

1. **Initial Budget Step ($N=20$)**:
   - Initialized at $\theta = \mathbf{0}$, ensuring the model starts from the canonical fixed-kernel geometry.
   - Initial parameter fingerprint recorded in `ExperimentRunRecord`.
2. **Subsequent Budget Steps ($N \in \{30, 40, 50\}$)**:
   - **Warm-Start Optimization**: Initialized from the predecessor budget's optimal parameters $\theta_{t-1}^*$.
   - Explicit parent model versioning and parent parameter fingerprint tracking:
     `parent_model_version = v(t-1)`
     `parent_parameter_fingerprint = SHA256(θ_{t-1}^*)`
   - Zero test-time adaptation: Evaluation samples never update $\theta$.

---

## 7. Leakage Controls and Data Eligibility

1. **Zero Test Leakage**:
   - PCA ($384 \to 8$) and MinMax scaling are fitted strictly on currently available training experiences at step $t$.
   - Protected test embeddings are projected using the train-fitted transformation matrix only.
2. **Experience Eligibility**:
   - Only records with `verification_status = USER_VERIFIED` and `learning_eligible = True` enter the adaptation stream.
   - Replay memory matches Phase 7.5 specifications identically.
3. **Speaker Isolation**:
   - Models are personalized strictly per speaker with independent directories and registry entries.

---

## 8. Governance Integration (Phase 7.6)

Every candidate model $M_t$ produced by continual trainable quantum adaptation was subjected to the Phase 7.6 `ValidationGate`:
1. **Multi-Metric Validation**: Evaluated against candidate metrics vs. active baseline.
2. **Safety Veto Enforcement**:
   - If False Emergency Rate exceeds 0.25 on validation set, candidate is vetoed.
   - If Emergency Recall degrades by more than 0.05, candidate is vetoed.
3. **Outcome**:
   - 54 of 60 models were promoted (`PROMOTE`).
   - 6 of 60 models were rejected (`REJECT`) due to safety veto triggers (e.g., M02 $N=20$, M04 $N=20$, MC04 $N \in \{20, 30, 40, 50\}$ due to severe FER spikes).

---

## 9. Primary 3-Way Empirical Results

The complete benchmark was executed across all 60 cells (15 TORGO speakers $\times$ 4 experience budgets).

### Matched Comparison Table (Cohort Mean, $N=60$ cells)

| Metric | Continual Classical (Phase 7.4) | Continual Fixed Q (Phase 7.8) | Continual Trainable Q (Phase 7.9) | Delta (TQ vs FQ) | $p_{\text{adj}}$ (TQ vs FQ) | Delta (TQ vs Classical) | $p_{\text{adj}}$ (TQ vs C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.1012 | 0.1187 | **0.1187** | $+0.0001$ ($d=+0.01$) | $0.4870$ (n.s.) | $+0.0176$ ($d=+0.57$) | $3.64 \times 10^{-6}$ (***) |
| **Emergency Recall** | **0.1620** | 0.0510 | 0.0581 | $+0.0071$ ($d=+0.16$) | $0.7188$ (n.s.) | $-0.1040$ ($d=-0.62$) | $2.84 \times 10^{-4}$ (***) |
| **False Emergency Rate** | 0.4353 | **0.1674** | 0.1720 | $+0.0046$ ($d=+0.15$) | $0.9878$ (n.s.) | $-0.2633$ ($d=-1.06$) | $8.38 \times 10^{-8}$ (***) |
| **Accuracy** | 0.5523 | **0.8107** | 0.8063 | $-0.0044$ ($d=-0.15$) | $0.8176$ (n.s.) | $+0.2540$ ($d=+1.05$) | $5.51 \times 10^{-8}$ (***) |

---

## 10. Trajectory Analysis Across Experience Budgets

| Budget ($N$) | Classical Macro-F1 | Fixed Q Macro-F1 | Trainable Q Macro-F1 | Delta (TQ vs FQ) | Classical Emerg Recall | Fixed Q Emerg Recall | Trainable Q Emerg Recall | Delta (TQ vs FQ) | TQ Adaptation Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | 0.0770 | 0.0984 | 0.0950 | $-0.0033$ | 0.1765 | 0.1172 | **0.1394** | $+0.0222$ | 1.26s |
| **30** | 0.0961 | 0.1222 | 0.1223 | $+0.0001$ | 0.1807 | 0.0330 | **0.0391** | $+0.0061$ | 1.95s |
| **40** | 0.1085 | 0.1222 | **0.1253** | $+0.0031$ | 0.1778 | 0.0391 | 0.0391 | $0.0000$ | 2.56s |
| **50** | 0.1230 | 0.1319 | **0.1323** | $+0.0004$ | 0.1131 | 0.0148 | 0.0148 | $0.0000$ | 3.18s |

---

## 11. Subgroup Analysis: Dysarthric vs Control Cohorts

### Dysarthric Speakers (8 Speakers, 32 Cells)
- **Macro-F1**: Classical = 0.1164, Fixed Q = 0.1335, Trainable Q = 0.1322 ($\Delta = -0.0013$)
- **Emergency Recall**: Classical = **0.1395**, Fixed Q = 0.0268, Trainable Q = **0.0268** ($\mathbf{\Delta = 0.0000}$)
  - At $N=20$: Fixed Q = 10.73%, Trainable Q = 10.73% ($\Delta = 0.00\%$)
  - At $N=30$: Fixed Q = 0.00%, Trainable Q = 0.00% ($\Delta = 0.00\%$)
  - At $N=40$: Fixed Q = 0.00%, Trainable Q = 0.00% ($\Delta = 0.00\%$)
  - At $N=50$: Fixed Q = 0.00%, Trainable Q = 0.00% ($\Delta = 0.00\%$)
- **False Emergency Rate**: Classical = 0.3333, Fixed Q = **0.0649**, Trainable Q = 0.0721 ($\Delta = +0.0072$)

### Control Speakers (7 Speakers, 28 Cells)
- **Macro-F1**: Classical = 0.0837, Fixed Q = 0.1017, Trainable Q = **0.1033** ($\Delta = +0.0016$)
- **Emergency Recall**: Classical = **0.1878**, Fixed Q = 0.0786, Trainable Q = **0.0938** ($\Delta = +0.0152$)
- **False Emergency Rate**: Classical = 0.5517, Fixed Q = **0.2844**, Trainable Q = 0.2861 ($\Delta = +0.0016$)

> [!CAUTION]
> **Atypical Speech Safety Collapse Unresolved**:
> Representation adaptation provided **zero benefit** ($\Delta = 0.0000$) to emergency recall for dysarthric speakers across all 32 experimental cells. For budgets $N \ge 30$, emergency recall remained completely collapsed at **0.00%**. All minor aggregate emergency recall gains were driven exclusively by the control group.

---

## 12. Statistical Significance Analysis

Matched two-sided Wilcoxon signed-rank paired tests across all 60 cells with Holm-Bonferroni correction:

1. **Trainable Quantum vs. Fixed Quantum (Representation Adaptation Effect)**:
   - Macro-F1: $W = 105.0$, $p = 0.1217$, $p_{\text{adj}} = 0.4870$ [Not Significant]
   - Emergency Recall: $W = 0.0$, $p = 0.1797$, $p_{\text{adj}} = 0.7188$ [Not Significant]
   - False Emergency Rate: $W = 74.0$, $p = 0.2469$, $p_{\text{adj}} = 0.9878$ [Not Significant]
   - Accuracy: $W = 169.0$, $p = 0.2044$, $p_{\text{adj}} = 0.8176$ [Not Significant]
2. **Trainable Quantum vs. Continual Classical (Classical Parity Check)**:
   - Macro-F1: $W = 248.0$, $p = 9.10 \times 10^{-7}$, $p_{\text{adj}} = 3.64 \times 10^{-6}$ [Significant advantage for Trainable Q]
   - Emergency Recall: $W = 34.0$, $p = 7.11 \times 10^{-5}$, $p_{\text{adj}} = 2.84 \times 10^{-4}$ [Severe deficit for Trainable Q: $-10.40\%$]
   - False Emergency Rate: $W = 111.0$, $p = 2.10 \times 10^{-8}$, $p_{\text{adj}} = 8.38 \times 10^{-8}$ [Significant advantage for Trainable Q]
   - Accuracy: $W = 141.0$, $p = 1.38 \times 10^{-8}$, $p_{\text{adj}} = 5.51 \times 10^{-8}$ [Significant advantage for Trainable Q]

---

## 13. Computational Cost and Runtime Analysis

| Metric | Classical Baseline (Phase 7.4) | Fixed Quantum (Phase 7.8) | Trainable Quantum (Phase 7.9) |
| :--- | :---: | :---: | :---: |
| **Model Type** | Feedforward Adapter | Fixed QSVC | Variational QSVC |
| **Trainable Parameters** | 25,223 | 0 | 16 |
| **Circuit Depth** | N/A | 37 | 37 |
| **Optimization Method** | AdamW (20 epochs) | Analytic Dual SVM | COBYLA (20 function evals) |
| **Optimization Time / cell** | 1.18s | 0.00s | 0.72s |
| **Total Adaptation Time / cell** | 1.18s | 0.82s | 2.24s |
| **Inference Time / sample** | 0.42 ms | 1.84 ms | 1.86 ms |
| **Total 60-Cell Benchmark Time** | 71.2s | 49.3s | 272.2s |

---

## 14. Safety-Sensitive Error Analysis

1. **Dominance of `GENERAL_SPEECH`**:
   At $N=50$, background speech comprises ~95% of natural interactions. Even with balanced KTA targets, the adapted quantum kernel remains dominated by geometric separation between background clusters, failing to preserve small emergency manifolds.
2. **Dysarthric Phonetic Collapse**:
   Dysarthric acoustic variations in emergency words (e.g., "help", "doctor", "fire") project into regions overlapping non-emergency words in the 8-D PCA subspace. Variational rotations alone cannot resolve these overlaps without expanding feature capacity or hybrid classical conditioning.
3. **Safety Veto Rejections**:
   Out of 60 candidate models, 6 failed the Phase 7.6 safety gate and were rejected. This demonstrates that Phase 7.6 governance functions properly and prevents unsafe deployment of degraded models.

---

## 15. Limitations

1. **Dimensionality Constraint**: 8 qubits constrained the input to 8 PCA components, identical to Phase 7.8 to maintain strict experimental control.
2. **Simulation Scale**: Statevector simulation was necessary for deterministic gradient-free optimization; execution on noisy hardware would introduce shot noise and decoherence.
3. **Kernel Alignment vs End-to-End Gradients**: KTA aligns the kernel before SVM fitting rather than training an end-to-end variational quantum classifier with backpropagation.

---

## 16. Final Hypothesis Conclusion

### Hypothesis $H_{7.7}$: **NOT SUPPORTED**
- **RQ7.7 Decision**: **NOT SUPPORTED**
- **Rationale**:
  1. Primary comparison against the fixed quantum kernel showed **no statistically significant difference** across any metric after multiple-comparison correction (Macro-F1 $\Delta = +0.0001, p_{\text{adj}} = 0.4870$; Emergency Recall $\Delta = +0.0071, p_{\text{adj}} = 0.7188$).
  2. For dysarthric speech, emergency recall improvement was **0.0000**, with recall remaining at **0.00%** for all budgets $N \ge 30$.
  3. Against the continual classical baseline, trainable quantum models suffer a **severe 10.40 percentage-point deficit in emergency recall** ($p_{\text{adj}} = 2.84 \times 10^{-4}$).
- **Frozen Historical Conclusion (CASE B)**: **UNCHANGED** (*No Statistically Supported Quantum Advantage*).

---

## 17. Reproducibility Information

- **Execution Script**: [`scripts/run_phase7_9_trainable_quantum_experiments.py`](file:///d:/DL/scripts/run_phase7_9_trainable_quantum_experiments.py)
- **Unit Tests**: [`tests/test_phase7_9_trainable_quantum.py`](file:///d:/DL/tests/test_phase7_9_trainable_quantum.py) (19/19 passed)
- **Results CSV**: [`results/continual/phase7_9/continual_trainable_quantum_results.csv`](file:///d:/DL/results/continual/phase7_9/continual_trainable_quantum_results.csv)
- **3-Way Matched Comparison**: [`results/continual/phase7_9/matched_three_way_comparison.csv`](file:///d:/DL/results/continual/phase7_9/matched_three_way_comparison.csv)
- **Trajectory Summary**: [`results/continual/phase7_9/trainable_quantum_trajectory_summary.csv`](file:///d:/DL/results/continual/phase7_9/trainable_quantum_trajectory_summary.csv)
- **Subgroup Analysis**: [`results/continual/phase7_9/trainable_quantum_subgroup_analysis.csv`](file:///d:/DL/results/continual/phase7_9/trainable_quantum_subgroup_analysis.csv)
- **Summary JSON**: [`reports/continual/phase7_9/continual_trainable_quantum_summary.json`](file:///d:/DL/reports/continual/phase7_9/continual_trainable_quantum_summary.json)
- **Experiment Registry**: [`results/continual/phase7_9/experiment_registry.json`](file:///d:/DL/results/continual/phase7_9/experiment_registry.json)
