# PHASE 8 PIPELINE FORENSIC AUDIT
## Comprehensive Investigation into the Emergency Metric Identicality Anomaly

**Project:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Date:** October 2026  
**Status:** Audit Complete — Root Cause Formally Verified and Replicated

---

### 1. Executive Summary & Verification of the Anomaly

During the evaluation of the original Phase 8 benchmark across 60 experimental cells (15 TORGO speakers $\times$ 4 few-shot adaptation budgets $N \in \{20, 30, 40, 50\}$), the emergency detection metrics exhibited exact bitwise equality across all three evaluation tracks (Track A: Hierarchical Classical Baseline, Track B: Quantum Residual Emergency Subtype Module, Track C: Matched Classical Shadow Model):

- **Aggregate Emergency Recall:** Exactly `46.83465608465608%` across Track A, Track B, and Track C ($\sigma = 39.58\%$).
- **Aggregate Emergency Precision:** Exactly `1.604082187630006%` across Track A, Track B, and Track C.
- **Aggregate False Emergency Rate (FER):** Exactly `52.52518148068819%` across Track A, Track B, and Track C ($\sigma = 36.61\%$).
- **Dysarthric Subgroup Recall:** Exactly `38.090277777777776%` across all three tracks.
- **Dysarthric Subgroup FER:** Exactly `43.48242951928555%` across all three tracks.
- **Control Subgroup Recall:** Exactly `56.82823129251701%` across all three tracks.
- **Control Subgroup FER:** Exactly `62.85975515086263%` across all three tracks.
- **Safety Gate Decisions:** 2 PASS / 58 FAIL across all three tracks (identical 3.33% pass rate).
- **Macro-F1 (7-Class):** Only metric with slight variance across tracks (Track A: 8.17%, Track B: 8.10%, Track C: 8.23%).

Programmatic verification against `results/phase8/phase8_results.csv` confirms:
- `Recall A == B`: **TRUE** (all 60 cells)
- `Recall B == C`: **TRUE** (all 60 cells)
- `Precision A == B`: **TRUE** (all 60 cells)
- `Precision B == C`: **TRUE** (all 60 cells)
- `FER A == B`: **TRUE** (all 60 cells)
- `FER B == C`: **TRUE** (all 60 cells)
- `Macro-F1 A == B`: **FALSE** (differs across tracks)
- `Macro-F1 B == C`: **FALSE** (differs across tracks)

---

### 2. End-to-End Data and Prediction Flow Trace

We traced the execution path across `scripts/run_phase8_experiments.py`, `src/phase8/phase8_models.py`, `src/phase8/hierarchical_gate.py`, `src/phase8/quantum_residual.py`, `src/phase8/classical_shadow.py`, and `scripts/generate_phase8_analysis_and_plots.py`:

```
Input Embedding X (384-D Whisper)
        │
        ▼
Stage 1: BinaryEmergencyDetector.predict_proba(X)
        │
        ▼
gate_probs[:, 1] >= threshold (0.5) ───► is_emergency (Boolean Array)
        │                                      │
        ├──────────────────────┬───────────────┤
        │                      │               │
  [is_emergency = False]       │        [is_emergency = True]
        │                      │               │
        ▼                      │               ▼
  Assign GENERAL_SPEECH        │        Stage 2 Subtype Classifier
                               │        - Track A: ClassicalSubtypeClassifier (384-D)
                               │        - Track B: QuantumSubtypeCircuit (8-D VQC)
                               │        - Track C: ClassicalShadowClassifier (8-D MLP)
                               │               │
                               │               ▼
                               │        em_preds (chosen strictly from EMERGENCY_CLASSES)
                               │               │
                               ▼               ▼
                        preds[em_indices] = em_preds
                               │
                               ▼
               Final 7-Class Predictions (preds)
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
is_pred_emergency = (label in EMERGENCY_CLASSES)    Macro-F1
[EXACTLY EQUAL TO is_emergency!]               (computed on 7 classes)
        │
        ▼
Emergency Metrics:
- Recall = (true_em & is_pred_em) / true_em
- Precision = (true_em & is_pred_em) / pred_em
- FER = (true_gen & is_pred_em) / true_gen
```

---

### 3. Answers to the Six Mandatory Forensic Audit Questions

#### Question 1: Which prediction vector feeds Emergency Recall, Precision, and FER?
**Answer:** In `compute_phase8_metrics` (`src/phase8/phase8_models.py`, lines 301–324):
```python
is_true_emergency = np.array([label in EMERGENCY_CLASSES for label in y_true])
is_pred_emergency = np.array([label in EMERGENCY_CLASSES for label in y_pred])
```
The vector `y_pred` comes directly from `model.predict(X_test)`. Within `model.predict`, any sample where Stage 1 evaluated `is_emergency = True` was assigned a prediction from Stage 2 (`em_preds`). Because all Stage 2 models (Classical, Quantum, Shadow) are 6-class emergency subtype classifiers predicting solely from `EMERGENCY_CLASSES`, **every sample where Stage 1 predicted emergency remained an emergency class**. Therefore, `is_pred_emergency` was mathematically identical to `is_emergency` from Stage 1 for all samples.

#### Question 2: Does each track affect its own final emergency score?
**Answer:** **NO.** In the original Phase 8 architecture:
- Track A's Stage 2 classical subtype net did not affect the emergency score.
- Track B's 8-qubit Variational Quantum Circuit (VQC) did not produce an emergency detection score or residual; it only classified the emergency subtype among the 6 classes *after* the sample was already categorized as an emergency.
- Track C's 8-D MLP shadow similarly had no connection to the emergency detection score.
The quantum residual had **zero mathematical participation** in the primary emergency detection decision.

#### Question 3: Do metrics and plots reference the correct track-specific prediction files/columns?
**Answer:** Yes, the plotting script (`scripts/generate_phase8_analysis_and_plots.py`) correctly read the `track` column in `results/phase8/phase8_results.csv`. However, because `results/phase8/phase8_results.csv` recorded identical numbers for all three tracks, the plots accurately displayed identical bars and trajectories for Emergency Recall, Precision, and FER.

#### Question 4: Are subgroup and per-budget results taken from the same shared column?
**Answer:** Subgroup (dysarthric vs. control) and per-budget metrics were aggregated from the `emergency_recall`, `emergency_precision`, and `false_emergency_rate` columns of `results/phase8/phase8_results.csv` filtered by `speaker_group` and `budget`. Because every underlying cell had identical values across tracks, every slice (by speaker, group, or budget) naturally produced identical aggregated values across all tracks.

#### Question 5: Do the three tracks share a Stage-1 gate intentionally, and if so, which metrics are Stage-1-only versus end-to-end?
**Answer:** The models did not literally share a single instantiated Python object; each model instantiated its own `BinaryEmergencyDetector(random_seed=42)`. However:
1. In `config/phase8/research_protocol.yaml` (lines 84–86 and 104–105), the research protocol declared:
   - Track B Stage 1 gate: "Identical to Track A (Linear 384->64->2), frozen weights"
   - Track C Stage 1 gate: "Identical to Track A and B"
2. Because each track instantiated the identical PyTorch network architecture with `random_seed=42`, `pos_weight=3.0`, and trained it on the identical `X_train` with identical epochs (40), deterministic execution produced bitwise identical weights and bitwise identical probability outputs across all tracks.
3. Because Stage 2 did not feed back into the emergency decision, the reported Emergency Recall, Precision, and FER were purely Stage-1-only metrics disguised as end-to-end metrics.

#### Question 6: Are Track A, Track B, and Track C actually independently trained?
**Answer:** Technically, three separate model instances were trained in sequence within the loop in `scripts/run_phase8_experiments.py`. However:
- Their Stage 1 training was completely redundant (3 identical trainings of the same net with the same seed).
- Track B and Track C were never given an independent training or inference pipeline for the emergency score; they were hard-gated by the Stage 1 output.
- Therefore, there was no independent pipeline evaluation for emergency communication reliability.

---

### 4. Confirmed Root Cause Statement

The verified root cause of identical emergency metrics in Phase 8 is a **two-fold structural defect**:
1. **Decision Bottleneck Defect:** The Stage 1 binary gate functioned as an unconditional, irreversible bottleneck. The Stage 2 models (including the Track B quantum VQC) were strictly confined to conditional subtype categorization among the 6 emergency classes. No residual logit from Stage 2 was fused into the emergency triage logit.
2. **Deterministic Triplicate Training:** Each track deterministically trained an identical Stage 1 network with `random_seed=42` on identical training samples. Consequently, all three tracks produced bitwise identical binary emergency predictions $P(\text{Emergency}) \ge 0.5$, forcing Emergency Recall, Emergency Precision, and False Emergency Rate to be 100% identical.

---

### 5. Architectural Specification for the Separated Pipeline Repair

To establish genuine scientific separation, fair comparison, and verified quantum participation, the pipeline is refactored into `phase8_separated` under the following specification:

#### 1. Continuous Scalar Logit Fusion:
For each track $t \in \{\text{Track A}, \text{Track B}, \text{Track C}\}$:
$$z_{\text{final}}^{(t)} = z_{\text{gate}}^{(t)} + \alpha \cdot z_{\text{residual}}^{(t)}$$
$$p_{\text{emergency}}^{(t)} = \sigma(z_{\text{final}}^{(t)}) = \frac{1}{1 + e^{-z_{\text{final}}^{(t)}}}$$
$$\hat{y}_{\text{emergency}}^{(t)} = \mathbb{I}\left(p_{\text{emergency}}^{(t)} \ge \tau^{(t)}\right)$$

Where:
- $z_{\text{gate}}^{(t)}$ is produced by an independently trained Stage 1 Classical Emergency Gate.
- $z_{\text{residual}}^{(t)}$ is produced by track-specific modules:
  - **Track A:** `ClassicalResidualModule` (384-D MLP $\to$ scalar residual logit).
  - **Track B:** `QuantumResidualModule` (Train-only 8-D PCA $\to$ 8-qubit PennyLane VQC with AngleEmbedding and BasicEntanglerLayers measuring Pauli-Z $\to$ linear projection $\to$ scalar residual logit).
  - **Track C:** `MatchedShadowResidualModule` (Train-only 8-D PCA $\to$ capacity-matched 8-D MLP $\to$ scalar residual logit).
- $\alpha = 0.5$ (registered fusion coefficient).
- $\tau^{(t)}$ is an operating threshold calibrated strictly on internal validation data with a pre-registered safety target ($\text{FER} \le 15\%$). Test data is never accessed during calibration.

#### 2. Sequential Execution & Freezing Order:
1. **Track A (Classical Baseline):** Trained and evaluated across all 60 cells. Provenance and prediction records written to `results/phase8_separated/track_a_classical/`. Outputs frozen and validated by writing `BASELINE_FROZEN.json`.
2. **Track B (Quantum Residual):** Verified sentinel `BASELINE_FROZEN.json` before execution. Trained independently without reading Track A predictions or weights. Evaluated across the same locked test IDs. Provenance written to `results/phase8_separated/track_b_quantum/`. Verified quantum influence ablation before writing `QUANTUM_RUN_COMPLETE.json`.
3. **Track C (Matched Classical Shadow):** Executed independently. Provenance written to `results/phase8_separated/track_c_classical_shadow/`.
4. **Comparison & Synthesis:** Conducted only after all three tracks have completed and passed integrity verification.

---
**Audit Approved for Implementation:** Antigravity Research Engineering
