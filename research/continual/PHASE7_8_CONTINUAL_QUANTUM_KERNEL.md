# PHASE 7.8: CONTINUAL QUANTUM KERNEL PERSONALIZATION
## Empirical Research Report — Matched Continual Quantum vs. Classical Personalization

> **Inviolable Research Governance Notice:**  
> “Phase 7.8 evaluates continual fixed-kernel quantum personalization and does not modify the frozen Phase 6.5 quantum conclusion.”  
> **Frozen Historical Scientific Conclusion:** `CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE under the evaluated fixed-kernel protocol` (UNCHANGED).

---

## 1. Objective

Phase 7.8 executes the first continual quantum-learning experiment in the research project *“Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication”*.

The primary objective is to evaluate whether a **FIXED quantum kernel** ($d=8$, 8 qubits, $ZZ$ feature map, $\text{reps}=1$, linear entanglement, Fidelity Quantum Kernel, QSVC $C=1.0$), applied within a closed-loop longitudinal personalization stream of verified user experiences, provides measurable performance or safety-relevant benefits over the matched continual classical baseline established in Phase 7.4.

This phase investigates whether the accumulation of verified speaker experiences enables a fixed quantum kernel to adapt more effectively than the canonical classical adapter baseline (25,223 parameters), under identical speaker cohorts, data splits, accumulation budgets, and evaluation partitions.

---

## 2. Research Questions

- **Primary Research Question (RQ7.6):**  
  *Does continual quantum personalization provide benefit beyond continual classical personalization?*
- **Secondary Research Question:**  
  *Under matched speaker, data, budget, replay, and evaluation conditions, does the fixed quantum-kernel model improve safety-relevant performance (Emergency Recall, False Emergency Rate, IPER) as verified experience accumulates?*

---

## 3. Frozen Historical Boundary

The historical boundaries established across prior phases remain strictly immutable:
- **Phases 1–6:** FROZEN.
- **Phase 6.5 (Primary Benchmark & Recovery Protocol):** FROZEN. Historical fixed-dataset quantum evaluation metrics and reports are preserved without modification.
- **Phase 7.0–7.7:** FROZEN. Continual Classical Baseline (7.4), Replay Memory & Catastrophic Forgetting (7.5), Validation Gate & Rollback Governance (7.6), and Speaker Drift Detection & Recalibration (7.7) remain intact.
- **Case B Conclusion:** *No statistically supported quantum advantage.* This conclusion remains unchanged.
- **Zero Fabrication:** No experimental data, metrics, or comparisons are invented. Historical Phase 6.5 results are never modified to alter the relative standing of the new continual experiment.

---

## 4. Quantum Configuration (Canonical SSOT)

The quantum model inherits the canonical Phase 6.5 SSOT configuration (`qsvc_q8_r1`):

| Parameter | Specification | Verification Status |
| :--- | :--- | :---: |
| **Qubit Count** | 8 qubits | VERIFIED (SSOT) |
| **Representation Dimension** | $d=8$ coordinates | VERIFIED (SSOT) |
| **Feature Map** | Second-order Pauli $ZZ$ Feature Map (`ZZFeatureMap`) | VERIFIED (SSOT) |
| **Entanglement** | `linear` (nearest-neighbor CNOT coupling) | VERIFIED (SSOT) |
| **Repetitions (Reps)** | 1 | VERIFIED (SSOT) |
| **Kernel Type** | `FidelityQuantumKernel` ($K_{ij} = \|\langle \psi(x_i) \mid \psi(x_j) \rangle\|^2$) | VERIFIED (SSOT) |
| **Backend / Simulation** | Exact Statevector (`qiskit.quantum_info.Statevector`) | VERIFIED (SSOT) |
| **Classifier** | `QSVC` via scikit-learn `SVC(kernel='precomputed', C=1.0, class_weight='balanced')` | VERIFIED (SSOT) |
| **Trainable Quantum Parameters** | **0** (strictly non-variational; zero Phase 7.9 leakage) | VERIFIED (SSOT) |
| **Feature Scaling** | MinMaxScaler fitted on adaptation training embeddings to $[0, \pi]$ | VERIFIED (SSOT) |

---

## 5. Feature Representation & Leakage Controls

To guarantee absolute fairness between classical and quantum modalities:
1. **Upstream Encoder:** Frozen `openai/whisper-tiny` bottleneck representations (384-D, unit L2 normalized).
2. **Dimensionality Reduction:** `TrainOnlyPCAPipeline` maps 384-D Whisper embeddings to $d=8$ principal components.
3. **Strict Information Isolation Rule:** PCA components and MinMaxScaler bounds $[0, \pi]$ are fitted **strictly on the adaptation training samples available at accumulation step $t$** ($N \in \{20, 30, 40, 50\}$).
4. **Zero Evaluation Leakage:** The held-out protected evaluation set is transformed using the fitted training pipeline without updating components or bounds. Overlap between training and protected evaluation instances raises `DataLeakageError`.

---

## 6. Continual Quantum Update Definition

In Phase 7.8, continual quantum learning is formally defined through support vector set adaptation over expanding verified experience buffers:
- **At Accumulation Step $t$ (Budget $N$):**
  $$\mathcal{D}_{\text{train}}^{(t)} = \{ (x_i, y_i) \}_{i=1}^N$$
  1. Fit `TrainOnlyPCAPipeline` ($d=8$) and $\text{MinMaxScaler}([0, \pi])$ on $\mathcal{D}_{\text{train}}^{(t)}$.
  2. Compute $N \times N$ symmetric quantum fidelity training kernel $K_{\text{train}}^{(t)}$.
  3. Fit QSVC with precomputed kernel $K_{\text{train}}^{(t)}$ and balanced class weighting.
  4. Transform protected test set $\mathcal{D}_{\text{test}}$ and evaluate test kernel $K_{\text{test}}^{(t)} \in \mathbb{R}^{M \times N}$.
  5. Generate candidate version $v^{(t)}$, evaluate via Phase 7.6 `ValidationGate`, and log into `ExperimentRegistry`.

---

## 7. Replay Policy & Matched Conditions

- **Replay Policy in Primary Comparison:** Matched to Phase 7.4 Continual Classical Baseline (`NO_REPLAY`, natural longitudinal stream where training buffer at step $N$ consists of all verified experiences accumulated chronologically up to budget $N$).
- **Experience Ordering:** Deterministic chronological sequence matching Phase 7.4.
- **Verification Policy:** Ingestion restricted strictly to records with `verification_status in (USER_VERIFIED, SYSTEM_VERIFIED)` and `learning_eligible=True`.

---

## 8. Matched Classical Control

Every continual quantum experiment is paired 1-to-1 with its matched classical control from Phase 7.4:
- **Matched Control Model:** Canonical Classical Adapter (25,223 parameters, AdamW, $\text{lr}=0.005$, $\text{epochs}=60$, $\text{batch\_size}=8$, balanced class weights).
- **Matching Invariants:** Same speaker, same random seed, same budget $N \in \{20, 30, 40, 50\}$, same adaptation sample IDs, same protected held-out test split, same 7-class taxonomy.
- **Tracking:** `matched_control_experiment_id = "EXP_7_4_{spk}_N{budget}_S{seed}"`.

---

## 9. Speaker Cohort

The full canonical cohort of 15 TORGO speakers was evaluated across all 4 budgets (60 matched cells):
- **8 Dysarthric Speakers:** `F01`, `F03`, `F04`, `M01`, `M02`, `M03`, `M04`, `M05` (32 experimental cells).
- **7 Control Speakers:** `FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC03`, `MC04` (28 experimental cells).

---

## 10. Experience Budgets

The 4 canonical accumulation budgets established in Phase 7.4 were evaluated:
- $B_0 = 20$ (initial few-shot calibration seed)
- $B_1 = 30$ ($B_0 + 10$ verified experiences)
- $B_2 = 40$ ($B_1 + 10$ verified experiences)
- $B_3 = 50$ ($B_2 + 10$ verified experiences)

---

## 11. Kernel Caching & Mathematical Integrity

1. **Deterministic Cache Key:**
   $$\text{Key} = \text{SHA256}(\text{speaker\_id} \,\|\, \text{budget} \,\|\, \text{seed} \,\|\, \text{dataset\_hash} \,\|\, \text{quantum\_config} \,\|\, \text{protocol\_version})$$
   Guarantees that changing any data or hyperparameter produces a cache miss and prevents stale kernel contamination.
2. **Numerical Validation Invariants:**
   - Values strictly bounded: $0.0 \le K_{ij} \le 1.0$.
   - Symmetry error: $\max |K - K^T| < 10^{-10}$.
   - Diagonal error: $\max |K_{ii} - 1.0| < 10^{-10}$.
   - Positive semi-definiteness: $\lambda_{\min}(K) \ge -10^{-8}$.

---

## 12. Simulator Environment

- **Simulator:** Local ideal statevector simulation via `qiskit.quantum_info.Statevector`.
- **Hardware Context:** Local CPU execution (`Windows 11, Intel Core, Python 3.11`).
- **Explicit Label:** `SIMULATED QUANTUM EXECUTION (Statevector)`. No physical QPU execution or hardware noise was introduced.

---

## 13. Safety & Governance (Phase 7.6 Integration)

Every newly fitted continual quantum model entered the model registry with status `CANDIDATE` and was evaluated against its active predecessor through the Phase 7.6 `ValidationGate`:
- **Unconditional Safety Veto:** If Emergency Recall drops relative to active baseline or False Emergency Rate exceeds safety limits, the candidate model is flagged with a Safety Veto and denied promotion (`REJECT`).
- **Promotion Integrity:** No quantum model bypassed governance directly to `ACTIVE`.

---

## 14. Empirical Results & Primary Comparison

### Full Cohort (15 Speakers, 60 Matched Cells)

| Metric | Classical Baseline (Phase 7.4) | Continual QSVC (Phase 7.8) | Delta ($\Delta = Q - C$) | Cohen's $d$ | Wilcoxon $p$-value (adj) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 0.1012 | 0.1187 | **+0.0175** | +0.55 | $1.99 \times 10^{-5}$ | $***$ |
| **Emergency Recall** | **0.1620 (16.2%)** | **0.0510 (5.1%)** | **-0.1110 (-11.1%)** | **-0.69** | **$3.12 \times 10^{-5}$** | **$***$** |
| **Emergency Precision**| 0.0074 | 0.0010 | -0.0064 | -0.59 | $1.99 \times 10^{-5}$ | $***$ |
| **False Emergency Rate**| 0.4353 | 0.1674 | -0.2679 | -1.06 | $8.67 \times 10^{-8}$ | $***$ |
| **IPER** | 0.1620 | 0.0510 | -0.1110 | -0.69 | $3.12 \times 10^{-5}$ | $***$ |
| **Accuracy** | 0.5523 | 0.8107 | +0.2584 | +1.06 | $6.68 \times 10^{-8}$ | $***$ |
| **Adaptation Time (s)**| 0.3521 | 0.0615 | -0.2906 | -3.48 | $1.14 \times 10^{-10}$ | $***$ |

---

## 15. Subgroup Analysis: Dysarthric vs. Control Cohorts

### A. Dysarthric Speakers (8 Speakers, 32 Cells)

| Budget | Classical Macro-F1 | Quantum Macro-F1 | $\Delta$ F1 | Classical Emerg. Recall | Quantum Emerg. Recall | $\Delta$ Emerg. Recall | Classical FER | Quantum FER |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=20** | 0.0915 | 0.1127 | +0.0211 | 0.1726 (17.3%) | 0.1073 (10.7%) | -0.0653 | 0.5250 | 0.2531 |
| **N=30** | 0.1162 | 0.1406 | +0.0244 | 0.1576 (15.8%) | **0.0000 (0.0%)** | **-0.1576** | 0.3543 | 0.0008 |
| **N=40** | 0.1254 | 0.1404 | +0.0150 | 0.1281 (12.8%) | **0.0000 (0.0%)** | **-0.1281** | 0.2710 | 0.0033 |
| **N=50** | 0.1327 | 0.1404 | +0.0078 | 0.0997 (10.0%) | **0.0000 (0.0%)** | **-0.0997** | 0.1830 | 0.0025 |
| **Mean** | **0.1164** | **0.1335** | **+0.0171** | **0.1395 (14.0%)** | **0.0268 (2.7%)** | **-0.1127** | **0.3333** | **0.0649** |

### B. Control Speakers (7 Speakers, 28 Cells)

| Budget | Classical Macro-F1 | Quantum Macro-F1 | $\Delta$ F1 | Classical Emerg. Recall | Quantum Emerg. Recall | $\Delta$ Emerg. Recall | Classical FER | Quantum FER |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N=20** | 0.0604 | 0.0821 | +0.0217 | 0.1811 (18.1%) | 0.1284 (12.8%) | -0.0527 | 0.7099 | 0.4248 |
| **N=30** | 0.0732 | 0.1012 | +0.0280 | 0.2071 (20.7%) | 0.0707 (7.1%) | -0.1364 | 0.6141 | 0.2861 |
| **N=40** | 0.0893 | 0.1014 | +0.0122 | 0.2347 (23.5%) | 0.0837 (8.4%) | -0.1510 | 0.5342 | 0.2860 |
| **N=50** | 0.1120 | 0.1222 | +0.0102 | 0.1284 (12.8%) | 0.0317 (3.2%) | -0.0967 | 0.3487 | 0.1409 |
| **Mean** | **0.0837** | **0.1017** | **+0.0180** | **0.1878 (18.8%)** | **0.0786 (7.9%)** | **-0.1092** | **0.5517** | **0.2844** |

---

## 16. Detailed Scientific Findings & Error Analysis

### Finding 1: The Quantum Majority-Margin Collapse Phenomenon
While raw Accuracy (+25.84%) and Macro-F1 (+0.0175, $p < 0.001$) show apparent mathematical gains, these gains are an artifact of **majority class conservative collapse**.
Because real-world atypical speech streams are heavily dominated by `GENERAL_SPEECH` (~95% background prevalence), projecting samples into the fixed $ZZ$ Hilbert space causes the QSVC margin to collapse around the dominant class. The classifier learns to predict `GENERAL_SPEECH` almost exclusively, drastically reducing false emergency alarms (FER drops from 43.53% to 16.74%) but devastating emergency detection.

### Finding 2: Catastrophic Emergency Recall Eradication in Dysarthric Speech
For dysarthric speakers, as verified experience accumulates from $N=20$ to $N=50$, **quantum emergency recall collapses from 10.73% at $N=20$ to exactly 0.000% across all budgets $N \ge 30$**.
In a life-critical emergency communication system, a model with 0.00% emergency recall is completely unusable and clinically dangerous.

### Finding 3: Phase 7.6 Governance Veto Confirmation
Under the multi-metric safety policy enforced by `ValidationGate`:
- Even when candidate models achieved lower FER and nominally higher Macro-F1, candidates exhibiting emergency recall regression were denied promotion or flagged with a Safety Veto.
- Out of 60 evaluation steps, whenever early emergency recall was observed to regress, candidate promotion was blocked.

---

## 17. Runtime & Computational Cost

- **Statevector Simulation Runtime:** The fixed quantum kernel evaluation for small support budgets ($N \in \{20, 30, 40, 50\}$) evaluated rapidly on CPU ($0.0615\text{ s}$ average adaptation time per step).
- **Physical Hardware Caution:** Simulator runtime does **not** equal physical QPU runtime. On physical quantum processors, shot noise, circuit compilation latency, and state preparation overhead would introduce substantial latency without resolving the representational margin collapse identified above.

---

## 18. Hypothesis Decisions

### Primary Hypothesis H7.6:
> **H7.6:** *Continual quantum kernel personalization provides statistically supported benefit beyond matched continual classical personalization.*
- **Decision:** **NOT SUPPORTED** (under safety-governed criteria) / **INCONCLUSIVE** (due to severe safety-critical emergency recall collapse).
- **Rationale:** Although Macro-F1 increased nominally by $+0.0175$ ($p < 0.001$) due to general speech conservatism, Emergency Recall suffered a catastrophic, statistically significant collapse of $-11.10\%$ ($p = 3.12 \times 10^{-5}, d = -0.69$), reaching $0.00\%$ for dysarthric speakers at $N \ge 30$. Under clinical safety constraints, no quantum advantage is established.

### Secondary Hypotheses:
- **H7.6a (Emergency Recall Benefit):** **NOT SUPPORTED**. Quantum emergency recall regressed by $-11.10\%$ overall and collapsed to $0.00\%$ in dysarthric cohorts.
- **H7.6b (Forgetting Reduction):** **NOT SUPPORTED**. The fixed quantum kernel suffers severe majority-class forgetting of emergency support vectors.
- **H7.6c (Safety-Preserving Computational Cost):** **NOT SUPPORTED**. The computational profile does not yield safety-compliant predictions.

---

## 19. Reproducibility & Provenance Metadata

- **Phase:** 7.8
- **Execution Date:** 2026-10-06
- **Total Experiments:** 60 matched continual quantum runs
- **Total Cells Evaluated:** 60
- **Cohorts:** 15 TORGO speakers (8 dysarthric, 7 control)
- **Budgets:** $N \in \{20, 30, 40, 50\}$
- **Primary Artifacts:**
  - `results/continual/phase7_8/continual_quantum_results.csv`
  - `results/continual/phase7_8/matched_quantum_classical_comparison.csv`
  - `results/continual/phase7_8/quantum_trajectory_summary.csv`
  - `results/continual/phase7_8/quantum_subgroup_analysis.csv`
  - `results/continual/phase7_8/experiment_registry.json`
  - `reports/continual/phase7_8/continual_quantum_summary.json`

---

## 20. Conclusion

Phase 7.8 establishes that **a fixed quantum kernel does NOT provide a viable continual personalization mechanism for low-resource atypical speech emergency recognition**. The fixed quantum geometry suffers from severe majority-class margin collapse, reducing emergency recall to $0.00\%$ in dysarthric speakers.

The historical finding **CASE B — NO STATISTICALLY SUPPORTED QUANTUM ADVANTAGE** remains fully validated, uncompromised, and unchanged.
